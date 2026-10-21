#!/usr/bin/env python3
"""Lidar mapping on the drone: no ground truth.

Pose  : the autopilot's own EKF estimate over MAVLink (LOCAL_POSITION_NED + ATTITUDE), i.e. GPS-based in the
        open and VIO-based under the deck, interpolated to the exact capture time of each scan: the scan's
        Gazebo sim-time stamp, converted to the autopilot's clock (time_boot_ms) with the offset between the
        two clocks measured from /world/<world>/clock (time only, no pose).
Refine: every scan is registered to the map built so far with point-to-plane ICP (lidar odometry on
        top of the EKF): the correction absorbs EKF / VIO drift and timing error so the map stays
        consistent. Only geometrically constrained directions are corrected, corrections are bounded
        per scan, and while GPS has a fix the correction is pulled back towards the GPS-based EKF
        (mapping.icp_*).
Map   : corrected scans merged into a `mapping.voxel` grid -> <out>/map_points.npy every few seconds
        and on exit; <out>/trajectory.npy (t, corrected x y z, qx qy qz qw);
        <out>/nav_status.json (corrected pose, GPS fix as reported by the autopilot) for other tools.
The lidar mounting pose comes from the drone model SDF (vehicle design, not truth).

With --ros, for RViz2 (frame "map" = world ENU):
  /bridge/map    PointCloud2 xyz, the growing map without water / ground (structure only)
  /bridge/scan   PointCloud2, latest corrected scan
  /bridge/path   nav_msgs/Path, the drone's own (corrected) trajectory, plus TF map -> iris
  /bridge/gps    MarkerArray: planned path (blue GPS / red VIO), flown path coloured by the autopilot's
                 GPS fix (green fix / red none), lost / regained points, status label

Usage: GZ_PARTITION=... record_lidar.py <config.yaml> [--ros]"""
import argparse, json, math, os, signal, sys, threading, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bridge_autonomy"))
from bridge_model import load_config, voxel_down, FloorMap
from frames import ned_to_enu, att_to_R
from sdf_utils import model_sdf_for_include, link_pose
from gz.transport13 import Node
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.msgs10.clock_pb2 import Clock
from pymavlink import mavutil

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--ros", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config); sc, mp = cfg["scene"], cfg["mapping"]
out = sc["out_dir"]; os.makedirs(out, exist_ok=True)
LIDAR_R, LIDAR_T, _ = link_pose(model_sdf_for_include(sc["world_sdf"], sc["vehicle_model"]), mp["lidar_link"])
VOX = mp["voxel"]
print("lidar mount", LIDAR_T.round(3).tolist(), flush=True)

# ---- autopilot pose stream --------------------------------------------------------------
lock = threading.Lock()
pos_buf, att_buf = [], []                       # (autopilot time s, enu p) / (autopilot time s, R)
nav = {"fix": None, "sats": 0, "home": None}     # home: first pose = where the drone took off (the deck)
clk = {"sim": None, "offs": [], "locked": None}   # latest Gazebo sim time; (sim - autopilot) clock offset samples


def on_clock(m):
    clk["sim"] = m.sim.sec + m.sim.nsec * 1e-9


def clock_offset():
    """The autopilot runs in lockstep with Gazebo, so the offset between the two clocks is constant: measure it
    over the first clock_offset_samples attitude messages (median, robust to message latency), then lock it."""
    if clk["locked"] is None and len(clk["offs"]) >= mp["clock_offset_samples"]:
        clk["locked"] = float(np.median(clk["offs"]))
        print(f"clock offset (Gazebo - autopilot) locked at {clk['locked']:.3f} s", flush=True)
    return clk["locked"]
mav = mavutil.mavlink_connection(mp["mavlink"], source_system=1, source_component=198)
mav.wait_heartbeat()
for mid, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 50), (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50),
                (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 5)):
    mav.mav.command_long_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                              mid, 1e6 / hz, 0, 0, 0, 0, 0)


def mav_loop():
    while True:
        m = mav.recv_match(blocking=True, timeout=1)
        if m is None:
            continue
        k = m.get_type()
        with lock:
            # keep each stream strictly increasing in time (UDP can duplicate / reorder)
            if k == "LOCAL_POSITION_NED" and (not pos_buf or m.time_boot_ms * 1e-3 > pos_buf[-1][0]):
                pos_buf.append((m.time_boot_ms * 1e-3, ned_to_enu(m.x, m.y, m.z)))
                if nav["home"] is None:
                    nav["home"] = pos_buf[-1][1].copy()
                del pos_buf[:-500]
            elif k == "ATTITUDE" and (not att_buf or m.time_boot_ms * 1e-3 > att_buf[-1][0]):
                att_buf.append((m.time_boot_ms * 1e-3, att_to_R(m.roll, m.pitch, m.yaw)))
                del att_buf[:-500]
                if clk["sim"] is not None and clk["locked"] is None:
                    clk["offs"].append(clk["sim"] - m.time_boot_ms * 1e-3)
            elif k == "GPS_RAW_INT":
                nav["fix"], nav["sats"] = bool(m.fix_type >= 3), int(m.satellites_visible)


threading.Thread(target=mav_loop, daemon=True).start()


def ekf_pose_at(t):
    """EKF pose at autopilot time t (s since its boot)."""
    with lock:
        if len(pos_buf) < 2 or len(att_buf) < 2:
            return None
        P, A = list(pos_buf[-200:]), list(att_buf[-200:])
    tp = np.array([x[0] for x in P]); ta = np.array([x[0] for x in A])
    if t < max(tp[0], ta[0]) or t > min(tp[-1], ta[-1]) + mp["max_pose_extrapolation"]:
        return None
    t_p, t_a = min(t, tp[-1]), min(t, ta[-1])
    p = np.array([np.interp(t_p, tp, [x[1][k] for x in P]) for k in range(3)])
    i = int(np.clip(np.searchsorted(ta, t_a), 1, len(ta) - 1))
    if ta[i] - ta[i - 1] < 1e-6:
        R = A[i][1]
    else:
        R = Slerp([ta[i - 1], ta[i]], Rotation.from_matrix([A[i - 1][1], A[i][1]]))(t_a).as_matrix()
    return p, R


# ---- map + scan registration ------------------------------------------------------------------
chunks, merged = [], np.zeros((0, 3), np.float32)
raw, raw_k = [], 0                                # raw scans (body frame) + EKF pose, for offline map building
raw_dir = os.path.join(out, "raw_scans")
if mp.get("record_raw"):
    os.makedirs(raw_dir, exist_ok=True)
    for f_ in os.listdir(raw_dir):
        os.remove(os.path.join(raw_dir, f_))
reg = {"pts": np.zeros((0, 3)), "tree": None, "nrm": None, "built": 0.0, "new": []}
corr_R, corr_t = np.eye(3), np.zeros(3)         # map <- EKF correction from ICP
traj, scans, icp_stats = [], 0, {"ok": 0, "rejected": 0, "skipped": 0}
latest = {}


def rebuild_registration_map(center):
    pts = np.concatenate([reg["pts"]] + reg["new"]) if reg["new"] else reg["pts"]
    pts = voxel_down(pts, mp["icp_map_voxel"]); reg["new"] = []
    reg["pts"] = pts
    near = pts[np.linalg.norm(pts - center, axis=1) < mp["icp_radius"]]
    if len(near) < 200:
        reg["tree"] = None; return
    tree = cKDTree(near)
    _, nb = tree.query(near, k=8)
    Q = near[nb] - near[nb].mean(1, keepdims=True)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    reg.update(tree=tree, nrm=Vt[:, 2, :], tgt=near, built=time.time())


def icp(src, center):
    """Point-to-plane ICP of src (already in the map frame by the current estimate) to the registration
    map, solved about `center` (the sensor). Returns (dR, dt, inlier fraction) in the map frame or None."""
    R, t = np.eye(3), np.zeros(3)
    tree, tgt, nrm = reg["tree"], reg["tgt"], reg["nrm"]
    S0 = src - center
    for dmax in np.linspace(mp["icp_max_corr"], mp["icp_min_corr"], mp["icp_iterations"]):
        X = S0 @ R.T + t
        d, i = tree.query(X + center, distance_upper_bound=dmax)
        ok = np.isfinite(d)
        if ok.sum() < mp["icp_min_pairs"]:
            return None
        A, B, N = X[ok], tgt[i[ok]] - center, nrm[i[ok]]
        r = np.sum((A - B) * N, 1)
        J = np.hstack([np.cross(A, N), N])
        w = 1.0 / np.maximum(1.0, np.abs(r) / (dmax / 3))            # Huber-like
        H = (J * w[:, None]).T @ J; g = (J * w[:, None]).T @ r
        x = -np.linalg.lstsq(H, g, rcond=None)[0]
        # only move along directions the geometry constrains (a flat deck says nothing about sliding
        # along it): normalised eigen-analysis of the rotation and translation blocks
        for blk, scale in ((slice(0, 3), np.sum(w * np.sum(A * A, 1))), (slice(3, 6), np.sum(w))):
            lam, E = np.linalg.eigh(H[blk, blk] / max(scale, 1e-9))
            good = E[:, lam >= mp["icp_min_eig"]]
            x[blk] = good @ (good.T @ x[blk])
        dR = Rotation.from_rotvec(x[:3]).as_matrix()
        R, t = dR @ R, dR @ t + x[3:]
    return R, t + center - R @ center, ok.mean()


skips = {"no_clock": 0, "no_pose": 0, "error": 0}


def on_cloud(msg):
    try:
        _on_cloud(msg)
    except Exception as e:                                                # gz transport swallows callback errors
        skips["error"] += 1
        if skips["error"] <= 5:
            import traceback; traceback.print_exc(); sys.stdout.flush()


def _on_cloud(msg):
    global scans, corr_R, corr_t
    t_rx = time.time()
    t_scan = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9          # capture time, Gazebo sim time
    off = clock_offset()
    if off is None:
        skips["no_clock"] += 1
        return
    for _ in range(20):                                                   # the pose stream may lag the scan slightly
        pose = ekf_pose_at(t_scan - off - mp["pose_latency"])
        if pose is not None:
            break
        time.sleep(0.01)
    if pose is None:
        skips["no_pose"] += 1
        return
    off = {f.name: f.offset for f in msg.field}
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                   "offsets": [off["x"], off["y"], off["z"]], "itemsize": msg.point_step})
    arr = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
    P = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
    r = np.linalg.norm(P, axis=1)
    P = P[np.isfinite(r) & (r > mp["min_range"]) & (r < mp["max_range"])]
    if not len(P):
        return
    p_ekf, R_ekf = pose
    Wb = (P @ LIDAR_R.T + LIDAR_T)                                     # body frame
    if mp.get("record_raw"):
        rb = voxel_down(Wb, mp["raw_voxel"])
        with lock:
            raw.append((t_scan, p_ekf, Rotation.from_matrix(R_ekf).as_quat(), np.round(rb * 100).astype(np.int16), nav["fix"]))
    R0, t0 = corr_R @ R_ekf, corr_R @ p_ekf + corr_t                   # current best pose
    W = Wb @ R0.T + t0
    if mp["icp"] and reg["tree"] is not None:
        res = icp(voxel_down(W, mp["icp_scan_voxel"]), t0)
        if res is not None:
            dR, dtv, inl = res
            step = np.linalg.norm(dtv); ang = math.degrees(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()))
            if inl >= mp["icp_min_inlier"] and step <= mp["icp_max_step"] and ang <= mp["icp_max_step_deg"]:
                corr_R, corr_t = dR @ corr_R, dR @ corr_t + dtv
                R0, t0 = corr_R @ R_ekf, corr_R @ p_ekf + corr_t
                W = Wb @ R0.T + t0
                icp_stats["ok"] += 1
            else:
                icp_stats["rejected"] += 1
        else:
            icp_stats["skipped"] += 1
    if nav["fix"]:
        # GPS is an absolute anchor: while it is good, pull the correction back towards the EKF
        k = mp["icp_gps_pull"]
        corr_t = (1 - k) * corr_t
        corr_R = Rotation.from_rotvec((1 - k) * Rotation.from_matrix(corr_R).as_rotvec()).as_matrix()
    W = voxel_down(W.astype(np.float32), VOX)
    q = Rotation.from_matrix(R0).as_quat()
    with lock:
        chunks.append(W); traj.append([t_scan, *t0, *q])
        latest.update(scan=W, pose=(t0, q, R0), t=t_scan)
        scans += 1
    reg["new"].append(W.astype(np.float64))
    if reg["tree"] is None or time.time() - reg["built"] > mp["icp_rebuild_s"]:
        rebuild_registration_map(t0)
    st = {"t": t_scan, "p": [round(float(x), 3) for x in t0], "q": [round(float(x), 5) for x in q],
          "gps_fix": nav["fix"], "sats": nav["sats"],
          "correction_m": round(float(np.linalg.norm(corr_t)), 3), "icp": dict(icp_stats)}
    tmp = os.path.join(out, "nav_status.json.tmp"); json.dump(st, open(tmp, "w")); os.replace(tmp, os.path.join(out, "nav_status.json"))


def save():
    global merged, raw_k
    with lock:
        new, chunks[:] = chunks[:], []
        tr = np.array(traj, np.float64)
        rw, raw[:] = raw[:], []
    if rw:
        np.savez(os.path.join(raw_dir, f"chunk_{raw_k:05d}.npz"), t=np.array([x[0] for x in rw]), p=np.array([x[1] for x in rw]),
                 q=np.array([x[2] for x in rw]), n=np.array([len(x[3]) for x in rw]), pts_cm=np.concatenate([x[3] for x in rw]),
                 fix=np.array([bool(x[4]) for x in rw]))
        raw_k += 1
    merged = voxel_down(np.concatenate([merged] + new), VOX) if new else merged
    np.save(os.path.join(out, "map_points.tmp.npy"), merged)
    os.replace(os.path.join(out, "map_points.tmp.npy"), os.path.join(out, "map_points.npy"))
    np.save(os.path.join(out, "trajectory.npy"), tr)
    return len(merged)


class RosOut:
    def __init__(self):
        import rclpy
        from sensor_msgs.msg import PointCloud2
        from nav_msgs.msg import Path
        from visualization_msgs.msg import MarkerArray
        from tf2_ros import TransformBroadcaster
        rclpy.init(); self.rclpy = rclpy
        self.n = rclpy.create_node("bridge_mapper")
        self.pub_map = self.n.create_publisher(PointCloud2, "/bridge/map", 1)
        self.pub_scan = self.n.create_publisher(PointCloud2, "/bridge/scan", 1)
        self.pub_path = self.n.create_publisher(Path, "/bridge/path", 1)
        self.pub_gps = self.n.create_publisher(MarkerArray, "/bridge/gps", 1)
        self.tfb = TransformBroadcaster(self.n)
        self.path = Path(); self.path.header.frame_id = "map"
        self.last_t, self.flown, self.fix, self.events, self.prev_fix = None, [], [], [], None
        self.plan, self.plan_mtime = None, 0
        self.vio_path = os.path.join(out, "vio_status.json")

    def _cloud(self, P):
        from sensor_msgs.msg import PointCloud2, PointField
        m = PointCloud2(); m.header.frame_id = "map"; m.header.stamp = self.n.get_clock().now().to_msg()
        m.height, m.width = 1, len(P)
        m.fields = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, k in enumerate("xyz")]
        m.point_step = 12; m.row_step = 12 * len(P); m.is_dense = True
        m.data = np.asarray(P, np.float32).tobytes()
        return m

    def publish_map(self, M):
        """Show the structure only: water / ground / banks (the lowest layer of returns below the deck the drone
        took off from) are left out; the deck / road itself is kept."""
        if len(M) < 1000:
            return
        D = voxel_down(M, mp["display_voxel"]).astype(np.float64)
        try:
            D = D[D[:, 2] - FloorMap(cfg, D, nav["home"]).floor_at(D[:, :2]) > cfg["cloud_analysis"]["struct_above_floor"]]
        except Exception:
            pass
        self.pub_map.publish(self._cloud(D))

    def _load_plan(self):
        f = os.path.join(out, "current_plan.npz")
        try:
            mt = os.path.getmtime(f)
        except OSError:
            return
        if mt != self.plan_mtime:
            d = np.load(f); k = max(1, int(0.5 / max(np.median(np.diff(d["S"])), 1e-3)))
            self.plan, self.plan_mtime = (d["P"][::k], d["zone"][::k]), mt

    def tick(self):
        from geometry_msgs.msg import PoseStamped, TransformStamped
        with lock:
            W, pq, t = latest.get("scan"), latest.get("pose"), latest.get("t")
        if W is None or t == self.last_t:
            self.rclpy.spin_once(self.n, timeout_sec=0); return
        self.last_t = t
        p, q, _ = pq; now = self.n.get_clock().now().to_msg()
        self.pub_scan.publish(self._cloud(W))
        tf = TransformStamped(); tf.header.stamp = now; tf.header.frame_id = "map"; tf.child_frame_id = "iris"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, p)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
        self.tfb.sendTransform(tf)
        ps = PoseStamped(); ps.header.frame_id = "map"; ps.header.stamp = now
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p); ps.pose.orientation = tf.transform.rotation
        self.path.poses.append(ps); self.path.header.stamp = now; self.pub_path.publish(self.path)
        fix = nav["fix"]
        if self.prev_fix is not None and fix is not None and fix != self.prev_fix:
            self.events.append((p.copy(), fix))
        if fix is not None:
            self.prev_fix = fix
        if not self.flown or np.linalg.norm(p - self.flown[-1]) > 0.3:
            self.flown.append(p.copy()); self.fix.append(bool(fix))
        self._load_plan()
        self.publish_markers(p, now)
        self.rclpy.spin_once(self.n, timeout_sec=0)

    def publish_markers(self, p, now):
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        col = lambda r, g, b, al=1.0: ColorRGBA(r=r, g=g, b=b, a=al)
        pt = lambda v: Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
        arr = MarkerArray()

        def mk(ns, i, typ):
            m = Marker(); m.header.frame_id = "map"; m.header.stamp = now; m.ns, m.id, m.type = ns, i, typ
            m.action = Marker.ADD; m.pose.orientation.w = 1.0; arr.markers.append(m)
            return m
        if self.plan is not None:
            P, Z = self.plan
            m = mk("planned_path", 0, Marker.LINE_STRIP); m.scale.x = 0.06
            m.points = [pt(v) for v in P]
            m.colors = [col(1.0, 0.3, 0.3, 0.45) if z else col(0.3, 0.6, 1.0, 0.45) for z in Z]
        if len(self.flown) > 1:
            m = mk("flown_gps", 0, Marker.LINE_STRIP); m.scale.x = 0.15
            m.points = [pt(v) for v in self.flown]
            m.colors = [col(0.1, 1.0, 0.3) if f else col(1.0, 0.1, 0.1) for f in self.fix]
        for i, (e, f) in enumerate(self.events):
            s = mk("gps_events", 2 * i, Marker.SPHERE); s.pose.position = pt(e); s.scale.x = s.scale.y = s.scale.z = 0.6
            s.color = col(0.1, 1.0, 0.3) if f else col(1.0, 0.1, 0.1)
            t = mk("gps_events", 2 * i + 1, Marker.TEXT_VIEW_FACING); t.pose.position = pt(e + [0, 0, 0.8]); t.scale.z = 0.6
            t.color = s.color; t.text = "GPS back" if f else "GPS lost"
        h = mk("drone_status", 0, Marker.TEXT_VIEW_FACING); h.pose.position = pt(p + [0, 0, 1.5]); h.scale.z = 0.8
        try:
            vio = json.load(open(self.vio_path))
        except (OSError, ValueError):
            vio = {}
        if nav["fix"]:
            h.text, h.color = f"GPS OK ({nav['sats']} sats)", col(0.1, 1.0, 0.3)
        else:
            h.text, h.color = "NO GPS - flying on VIO", col(1.0, 0.2, 0.2)
        h.text += "\nVIO tracking" if vio.get("healthy") else "\nVIO lost"
        self.pub_gps.publish(arr)


stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set()); signal.signal(signal.SIGINT, lambda *_: stop.set())
node = Node(); node.subscribe(Clock, f"/world/{sc['world_name']}/clock", on_clock)
node.subscribe(PointCloudPacked, mp["lidar_topic"], on_cloud)
print(f"mapping {mp['lidar_topic']} with the autopilot's pose -> {out}", flush=True)
ros = RosOut() if a.ros else None
last = last_map = time.time()
while not stop.is_set():
    time.sleep(0.2)
    if ros:
        ros.tick()
    if time.time() - last > mp["save_every_s"]:
        n = save(); last = time.time()
        print(f"scans {scans}  map points {n}  skipped {skips}  clock offset {clock_offset()}", flush=True)
    if ros and time.time() - last_map > mp["display_every_s"]:
        with lock:
            new = list(chunks)
        ros.publish_map(np.concatenate([merged] + new) if new else merged); last_map = time.time()
print(f"final: scans {scans}  map points {save()}", flush=True)

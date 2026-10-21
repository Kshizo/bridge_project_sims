#!/usr/bin/env python3
"""EVALUATION ONLY (referee). Nothing on the drone reads what this writes.

Compares the drone's own estimates with Gazebo's true pose:
  vio  : vio_status.json "p"   (what ArduPilot is fed under the deck)
  map  : nav_status.json "p"   (the mapper's pose: autopilot EKF + ICP; what pc_b2 is built with)
Each estimate is compared with the true position at the time the estimate was made (VIO: frame sim
time, mapper: scan wall time). Writes <out>/eval/truth_log.npy (t, true xyz, vio xyz, map xyz, gps fix,
true xyz at the VIO's time, true xyz at the mapper's time, mapper quaternion, true quaternion at the
mapper's time) and, on exit,
<out>/eval/accuracy.json (error stats with GPS available / denied).
With --ros publishes /eval/markers: the true GPS shadow (zone box), the true path (white), and a
label at the true position with the live errors.

Usage: truth_monitor.py <config.yaml> [--ros]"""
import argparse, json, math, os, signal, sys, threading, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge_autonomy"))
from bridge_model import load_config
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--ros", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config); sc = cfg["scene"]; out = sc["out_dir"]
ev = os.path.join(out, "eval"); os.makedirs(ev, exist_ok=True)
truth = {}
hist = []                                          # (sim time, wall time, true xyz): estimates are compared at their own time


def on_pose(m):
    for p in m.pose:
        if p.name == sc["vehicle_model"]:
            truth["p"] = np.array([p.position.x, p.position.y, p.position.z])
            hist.append((m.header.stamp.sec + m.header.stamp.nsec * 1e-9, time.time(), *truth["p"],
                         p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))
            if len(hist) > 20000:
                del hist[:10000]


def truth_at(t, col):
    """True position at time t (col 0: sim time, col 1: wall time)."""
    H = np.array(hist[-5000:])
    if not len(H) or t < H[0, col] or t > H[-1, col] + 0.2:
        return np.full(3, np.nan)
    return np.array([np.interp(t, H[:, col], H[:, k]) for k in (2, 3, 4)])


def truth_q_at(t):
    """True orientation (xyzw) closest to sim time t."""
    H = np.array(hist[-5000:])
    if not len(H) or t < H[0, 0] or t > H[-1, 0] + 0.2:
        return np.full(4, np.nan)
    return H[np.argmin(np.abs(H[:, 0] - t)), 5:9]


def read(name):
    try:
        return json.load(open(os.path.join(out, name)))
    except (OSError, ValueError):
        return {}


node = Node(); node.subscribe(Pose_V, f"/world/{sc['world_name']}/dynamic_pose/info", on_pose)
stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set()); signal.signal(signal.SIGINT, lambda *_: stop.set())
ros = None
if a.ros:
    import rclpy
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    rclpy.init(); rn = rclpy.create_node("truth_monitor"); ros = rn.create_publisher(MarkerArray, "/eval/markers", 1)
log, path, last_print, last_save = [], [], 0.0, 0.0
nan3 = [float("nan")] * 3
print("truth monitor: evaluation only", flush=True)
while not stop.is_set():
    time.sleep(0.2)
    if "p" not in truth:
        continue
    tp = truth["p"].copy(); vio, nav, gps = read("vio_status.json"), read("nav_status.json"), read("gps_status.json")
    vp = vio.get("p", nan3) if vio.get("healthy") else nan3
    mpos = nav.get("p", nan3)
    fix = gps.get("fix")
    tv = truth_at(vio["t"], 0) if "t" in vio else np.full(3, np.nan)          # VIO stamps frames in sim time
    tm = truth_at(nav["t"], 0) if "t" in nav else np.full(3, np.nan)          # the mapper stamps scans in sim time
    tq = truth_q_at(nav["t"]) if "t" in nav else np.full(4, np.nan)
    log.append([time.time(), *tp, *vp, *mpos, float("nan") if fix is None else float(fix), *tv, *tm,
                *nav.get("q", [np.nan] * 4), *tq])
    if not path or np.linalg.norm(tp - path[-1]) > 0.3:
        path.append(tp)
    e_vio = np.linalg.norm(np.subtract(vp, tv)); e_map = np.linalg.norm(np.subtract(mpos, tm))
    if time.time() - last_print > 30:
        last_print = time.time()
        print(f"truth ({tp[0]:.1f}, {tp[1]:.1f}, {tp[2]:.1f})  VIO err {e_vio:.2f} m  map-pose err {e_map:.2f} m  "
              f"GPS {'fix' if fix else 'denied'}", flush=True)
    if time.time() - last_save > 10:
        last_save = time.time(); np.save(os.path.join(ev, "truth_log.npy"), np.array(log))
    if ros is not None:
        now = rn.get_clock().now().to_msg(); arr = MarkerArray()
        col = lambda r, g, b, al=1.0: ColorRGBA(r=r, g=g, b=b, a=al)
        pt = lambda v: Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))

        def mk(ns, i, typ):
            m = Marker(); m.header.frame_id = "map"; m.header.stamp = now; m.ns, m.id, m.type = ns, i, typ
            m.action = Marker.ADD; m.pose.orientation.w = 1.0; arr.markers.append(m)
            return m
        z = gps.get("zone")
        if z:
            m = mk("true_gps_shadow", 0, Marker.CUBE); m.pose.position = pt(z["center"])
            m.pose.orientation.z, m.pose.orientation.w = math.sin(z["yaw"] / 2), math.cos(z["yaw"] / 2)
            m.scale.x, m.scale.y, m.scale.z = map(float, z["size"]); m.color = col(1.0, 0.15, 0.15, 0.10)
        if len(path) > 1:
            m = mk("true_path", 0, Marker.LINE_STRIP); m.scale.x = 0.05; m.color = col(1.0, 1.0, 1.0, 0.8)
            m.points = [pt(v) for v in path]
        t = mk("truth_label", 0, Marker.TEXT_VIEW_FACING); t.pose.position = pt(tp + [0, 0, -1.2]); t.scale.z = 0.6
        t.color = col(0.9, 0.9, 0.9)
        t.text = (f"[referee] VIO err {e_vio:.2f} m" if np.isfinite(e_vio) else "[referee] VIO not tracking") + \
                 (f" | map pose err {e_map:.2f} m" if np.isfinite(e_map) else "")
        ros.publish(arr); rclpy.spin_once(rn, timeout_sec=0)

L = np.array(log)
np.save(os.path.join(ev, "truth_log.npy"), L)
if len(L):
    Vp, Mp, F, Tv, Tm = L[:, 4:7], L[:, 7:10], L[:, 10], L[:, 11:14], L[:, 14:17]
    stats = lambda e: {"samples": int(np.isfinite(e).sum()), **({k: round(float(f(e[np.isfinite(e)])), 3) for k, f in
                       (("mean", np.mean), ("median", np.median), ("p95", lambda x: np.percentile(x, 95)), ("max", np.max))}
                       if np.isfinite(e).any() else {})}
    ev_vio, ev_map = np.linalg.norm(Vp - Tv, axis=1), np.linalg.norm(Mp - Tm, axis=1)
    res = {"vio": {"gps_available": stats(ev_vio[F == 1]), "gps_denied": stats(ev_vio[F == 0])},
           "map_pose": {"gps_available": stats(ev_map[F == 1]), "gps_denied": stats(ev_map[F == 0])}}
    json.dump(res, open(os.path.join(ev, "accuracy.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)

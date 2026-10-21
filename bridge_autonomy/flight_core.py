"""Flying a plan with ArduCopter in GUIDED mode (shared by mission.py and flight0.py).

Streams position + velocity feed-forward + yaw at 20 Hz; the reference only advances while the
vehicle (EKF estimate - the only position the drone has) keeps up. Navigation source management:
  * before the path enters the predicted GPS-denied zone (switch_lookahead metres ahead) the EKF is
    switched to source set 2 (external nav = VIO), once VIO has been healthy for healthy_hold s;
  * after leaving it, once GPS has been healthy for gps_recover_hold seconds, back to set 1;
  * VIO lost inside the zone: hold, then retrace the path back out.
"""
import json, math, os, time
import numpy as np
from pymavlink import mavutil
from frames import ned_to_enu

MASK = 0b0000100111000000


class Stalled(RuntimeError):
    pass


class Vehicle:
    def __init__(self, cfg, log=print):
        self.cfg, self.log = cfg, log
        self.gd, self.vc = cfg["gps_denied"], cfg["vio"]
        self.m = m = mavutil.mavlink_connection(cfg["mavlink"]["mission"], source_system=250)
        m.wait_heartbeat(); log("heartbeat")
        for mid, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 20), (mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 2),
                        (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 5)):
            m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, mid, 1e6 / hz, 0, 0, 0, 0, 0)
        self.pos, self.gps_ok, self.gps_ok_since, self.source = np.zeros(3), False, None, 1
        self.vio = {"healthy": False, "since": None, "read": 0.0}
        self.status_path = os.path.join(cfg["scene"]["out_dir"], "vio_status.json")

    # ---- telemetry -------------------------------------------------------------------
    def update(self, timeout=0.02):
        msg = self.m.recv_match(type=["LOCAL_POSITION_NED", "STATUSTEXT", "GPS_RAW_INT"], blocking=True, timeout=timeout)
        if msg is None:
            return
        t = msg.get_type()
        if t == "STATUSTEXT":
            self.log(f"  [ap] {msg.text}")
        elif t == "LOCAL_POSITION_NED":
            self.pos = ned_to_enu(msg.x, msg.y, msg.z)
        else:
            ok = msg.fix_type >= 3 and msg.satellites_visible >= 6
            if ok and not self.gps_ok:
                self.gps_ok_since = time.monotonic()
            self.gps_ok = ok

    def settle(self, seconds):
        t = time.monotonic()
        while time.monotonic() - t < seconds:
            self.update(0.1)

    def vio_ok_for(self, seconds):
        now, v = time.monotonic(), self.vio
        if now - v["read"] > 0.2:
            v["read"] = now
            try:
                h = json.load(open(self.status_path))["healthy"]
            except (OSError, ValueError, KeyError):
                h = False
            if h and not v["healthy"]:
                v["since"] = now
            v["healthy"] = h
        return v["healthy"] and now - v["since"] >= seconds

    def set_source(self, k):
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, mavutil.mavlink.MAV_CMD_SET_EKF_SOURCE_SET,
                                     0, k, 0, 0, 0, 0, 0, 0)
        self.source = k; self.log(f"  EKF source set -> {k} ({'GPS' if k == 1 else 'VIO'})")

    # ---- take-off / land ------------------------------------------------------------------
    def wait_ekf(self, timeout=300):
        t0 = time.monotonic()
        while True:
            e = self.m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=2)
            if e is not None and (e.flags & 0x18) == 0x18 and not (e.flags & 0x400):
                break
            if time.monotonic() - t0 > timeout:
                raise SystemExit("EKF never ready")
        self.log("EKF ready"); time.sleep(3)
        self.set_source(1)
        self.settle(4)

    def takeoff(self, climb):
        """Arm in GUIDED and climb `climb` m above the current point (works from a raised deck)."""
        m = self.m
        z0 = self.pos[2]
        # NAV_TAKEOFF checks the altitude against the EKF origin but targets it above home, and home is
        # reset to wherever we arm (e.g. a bridge deck). Re-home to the EKF origin after arming.
        m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                                mavutil.mavlink.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN, 0, 0, 0, 0, 0, 0)
        org = m.recv_match(type="GPS_GLOBAL_ORIGIN", blocking=True, timeout=10)
        if org is None:
            raise SystemExit("no GPS_GLOBAL_ORIGIN")
        self.log(f"EKF origin {org.latitude*1e-7:.7f}, {org.longitude*1e-7:.7f}; we are {z0:.2f} m above it")
        for _ in range(10):
            m.set_mode("GUIDED")
            for _ in range(30):
                m.arducopter_arm(); m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
                if m.motors_armed():
                    break
                time.sleep(2)
            m.mav.command_int_send(m.target_system, m.target_component, mavutil.mavlink.MAV_FRAME_GLOBAL, mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                                   0, 0, 0, 0, 0, 0, org.latitude, org.longitude, org.altitude / 1000.0)
            m.recv_match(type="COMMAND_ACK", condition="COMMAND_ACK.command==179", blocking=True, timeout=3)
            m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0,
                                    z0 + climb)
            t1 = time.monotonic()
            while time.monotonic() - t1 < 20 and self.pos[2] < z0 + climb / 2:
                self.update(0.2)
            if self.pos[2] >= z0 + climb / 2:
                break
            self.log("takeoff did not start, retrying")
        self.settle(3)
        self.log(f"airborne at {self.pos.round(2).tolist()}")

    def land(self):
        if self.source == 2:
            self.set_source(1)
        self.m.set_mode("LAND")
        while self.m.motors_armed():                             # armed state comes from HEARTBEAT messages
            self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        self.log("landed")

    # ---- plan following -------------------------------------------------------------------
    def send_setpoint(self, p, vel, yaw):
        """yaw None: keep the current heading."""
        mask = MASK | (1 << 10 if yaw is None else 0)
        self.m.mav.set_position_target_local_ned_send(0, self.m.target_system, self.m.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                                                      mask, p[1], p[0], -p[2], vel[1], vel[0], -vel[2], 0, 0, 0,
                                                      0.0 if yaw is None else math.pi / 2 - yaw, 0)

    def fly(self, plan, join_window=None, label="plan"):
        """Follow plan (dict with P, V, T, yaw, S, zone). join_window: join at the closest sample within
        the first join_window metres (None: start at sample 0). Returns when the end is reached."""
        P, V, T, YAW, S, ZONE = plan["P"], plan["V"], plan["T"], plan["yaw"], plan["S"], plan["zone"]
        gd, vc = self.gd, self.vc
        nxt = np.full(len(P), np.inf); d = np.inf                # path distance to the next GPS-denied sample
        for i in range(len(P) - 1, -1, -1):
            d = 0.0 if ZONE[i] else d + (S[i + 1] - S[i] if i + 1 < len(P) else 0)
            nxt[i] = d
        self.log(f"{label}: {S[-1]:.0f} m, {len(P)} samples, {100*ZONE.mean():.0f}% predicted GPS-denied")
        i = 0
        if join_window is not None:
            col = np.where(S <= S[0] + join_window)[0]
            i = int(col[np.argmin(np.linalg.norm(P[col] - self.pos, axis=1))])
        s_ref, last_t, last_print = S[i], time.monotonic(), 0
        direction, unhealthy_since, state = 1, None, "fly"
        mo = self.cfg["motion"]; behind_since = None; last_i, last_move = i, time.monotonic()
        while i < len(P) - 1:
            self.update()
            now = time.monotonic(); dt = min(now - last_t, 0.2); last_t = now      # monotonic + capped: no jumps
            err = np.linalg.norm(self.pos - P[i])
            want = 2 if nxt[i] <= gd["switch_lookahead"] or ZONE[i] else 1
            # advance while the vehicle keeps up; in wind it may sit a little off the reference, so after
            # track_stuck_s of not catching up the reference moves on anyway (it is collision-checked)
            close = err < (mo["track_tol_slow"] if V[i] < 1.0 else mo["track_tol_fast"])
            behind_since = None if close else (behind_since or now)
            advance = close or (err < mo["track_tol_fast"] and now - behind_since > mo["track_stuck_s"])
            if self.source == 2 and not self.vio_ok_for(0):
                unhealthy_since = unhealthy_since or now
                advance = False
                if now - unhealthy_since > gd["vio_lost_retreat"]:
                    if state != "retreat":
                        self.log("  VIO unhealthy in the GPS-denied zone - retracing the path out")
                    state, direction, advance = "retreat", -1, err < 1.0
            else:
                unhealthy_since = None
            if state == "retreat" and not ZONE[i] and nxt[i] > gd["switch_lookahead"]:
                state, direction = "wait", 1
            if want == 2 and self.source == 1:
                if self.vio_ok_for(vc["healthy_hold"]) and state != "wait":
                    self.set_source(2)
                else:
                    advance = False                              # hold short of the zone until VIO is trusted
            elif self.source == 2 and (want == 1 or state == "wait") and self.gps_ok and self.gps_ok_since \
                    and now - self.gps_ok_since > gd["gps_recover_hold"]:
                self.set_source(1)
            if state == "wait" and self.source == 1 and self.vio_ok_for(vc["healthy_hold"]):
                state = "fly"; self.log("  GPS + VIO healthy again - resuming")
            if state == "wait":
                advance = False
            if advance:
                s_ref += direction * max(V[i], 0.3 if direction < 0 else 0) * dt
                while direction > 0 and i < len(P) - 1 and S[i + 1] <= s_ref:
                    i += 1
                while direction < 0 and i > 0 and S[i] > s_ref:
                    i -= 1
            if i != last_i:
                last_i, last_move = i, now
            elif now - last_move > mo["stall_abort_s"]:
                raise Stalled(f"no progress along the {label} for {mo['stall_abort_s']} s at {P[i].round(1).tolist()}")
            vel = direction * T[i] * (V[i] if (err < 1.0 and advance) else 0.0)
            self.send_setpoint(P[i], vel, YAW[i])
            if now - last_print > 10:
                last_print = now
                self.log(f"{label} {100*S[i]/S[-1]:5.1f}%  ref ({P[i][0]:.1f}, {P[i][1]:.1f}, {P[i][2]:.1f})  err {err:.2f} m  "
                         f"nav {'VIO' if self.source == 2 else 'GPS'}  vio {'ok' if self.vio['healthy'] else 'LOST'}  {state}"
                         f"{'  [GPS-denied zone]' if ZONE[i] else ''}")
            time.sleep(0.03)
        while np.linalg.norm(self.pos - P[-1]) > 0.5:
            self.send_setpoint(P[-1], np.zeros(3), YAW[-1]); self.update(0.1)
        if self.source == 2 and not ZONE[-1]:
            self.set_source(1)

    def goto(self, target, yaw, speed):
        """Straight, slow move in open air (used before any map exists, e.g. the first climb)."""
        start = self.pos.copy(); L = np.linalg.norm(target - start); t0 = time.monotonic()
        u = (target - start) / max(L, 1e-6)
        while True:
            self.update()
            s = min(L, speed * (time.monotonic() - t0))
            ref = start + u * s
            self.send_setpoint(ref, u * speed if s < L else np.zeros(3), yaw)
            if s >= L and np.linalg.norm(self.pos - target) < 0.5:
                return
            time.sleep(0.03)

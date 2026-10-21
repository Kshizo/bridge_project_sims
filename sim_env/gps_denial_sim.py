#!/usr/bin/env python3
"""SIMULATED WORLD. Simulates GPS loss under the bridge: watches the drone's true pose in Gazebo and turns
SITL's GPS off (SIM_GPS1_ENABLE) while it is inside the GPS-denied zone derived from the
bridge geometry (deck footprint + margin, below the deck soffit). Environment side only;
the flight stack never sees this script's truth data.

It also watches the GPS fix the autopilot actually reports (GPS_RAW_INT) and writes
<out_dir>/gps_status.json for the evaluation display: the zone box, the current state and every point
where the fix was lost / regained (true position at that moment).

Usage: gps_denial_sim.py <config.yaml>"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge_autonomy"))
from bridge_model import load_config
from world_model import WorldTruth
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from pymavlink import mavutil

cfg = load_config(sys.argv[1]); sc = cfg["scene"]
br = WorldTruth(cfg)
pos = {}
status_path = os.path.join(sc["out_dir"], "gps_status.json")


def on_pose(m):
    for p in m.pose:
        if p.name == sc["vehicle_model"]:
            pos["p"] = np.array([p.position.x, p.position.y, p.position.z])


node = Node(); node.subscribe(Pose_V, f"/world/{sc['world_name']}/dynamic_pose/info", on_pose)
mav = mavutil.mavlink_connection(cfg["mavlink"]["gps_sim"], source_system=254)
mav.wait_heartbeat()
mav.mav.command_long_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                          mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 2e5, 0, 0, 0, 0, 0)


def set_gps(on):
    for _ in range(5):
        mav.mav.param_set_send(mav.target_system, mav.target_component, b"SIM_GPS1_ENABLE", 1.0 if on else 0.0,
                               mavutil.mavlink.MAV_PARAM_TYPE_INT8)
        m = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if m and m.param_id == "SIM_GPS1_ENABLE" and int(m.param_value) == int(on):
            return True
    return False


status = {"zone": br.gps_denied_box(), "in_zone": False, "fix": None, "sats": 0, "events": []}


def write_status():
    tmp = status_path + ".tmp"
    json.dump(status, open(tmp, "w")); os.replace(tmp, status_path)


state = None; last_write = 0; had_fix = False
print(f"GPS-denied zone: deck footprint s {np.round(br.s_range,1)} c {np.round(br.c_range,1)} "
      f"(+{cfg['gps_denied']['footprint_margin']} m), below z {br.deck_bottom + cfg['gps_denied']['top_offset']:.2f}", flush=True)
while True:
    if "p" in pos:
        inside = bool(br.in_gps_denied_zone(pos["p"])[0])
        if inside != state:
            ok = set_gps(not inside)
            print(f"{time.strftime('%H:%M:%S')}  drone at {pos['p'].round(1)} -> GPS {'OFF (under bridge)' if inside else 'ON'}"
                  f"{'' if ok else ' (param not acknowledged)'}", flush=True)
            state = inside; status["in_zone"] = inside
    while (m := mav.recv_match(blocking=False)) is not None:
        if m.get_type() != "GPS_RAW_INT":
            continue
        fix = m.fix_type >= 3
        if status["fix"] is not None and fix != status["fix"] and "p" in pos and had_fix:
            status["events"].append({"t": time.time(), "pos": pos["p"].round(2).tolist(), "fix": fix})
            print(f"{time.strftime('%H:%M:%S')}  autopilot GPS fix {'regained' if fix else 'LOST'} at {pos['p'].round(1)}", flush=True)
        status["fix"], status["sats"] = fix, int(m.satellites_visible)
        had_fix = had_fix or fix                            # the first fix after boot is not an event
    if time.time() - last_write > 0.5:
        if "p" in pos:
            status["pos"] = pos["p"].round(2).tolist()
        write_status(); last_write = time.time()
    time.sleep(0.1)

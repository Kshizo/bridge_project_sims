#!/usr/bin/env python3
"""flight0: the piloted mapping flight that produces pc_b2, the point cloud every later flight plans from.

flight0 stands in for a manual flight: the pilot knows where the bridge is, so its route comes from
the real bridge layout (sim_env/world_model.MeshBridge). The route only covers the bridge itself
(deck end to deck end, at most top_pass_offset above the railing):
  * exterior : side passes on both sides at the survey.side_levels heights + one pass over the deck
  * sections : under the deck, for every span between supports, a loop in front of the piers /
               abutment at the top of the pillars, then the middle, then the bottom
What the flight PRODUCES uses no ground truth: the point cloud comes from the drone's lidar placed with
its own pose (autopilot EKF on GPS / VIO, refined by ICP; mapping/record_lidar.py), with sensor noise
and wind on. After landing, this flight's map is written as the bridge cloud (scene.prior_cloud: only
the bridge - water, ground and banks removed) + a coarse surroundings file + json (prior_cloud.py).
Usage: flight0.py <config.yaml>"""
import json, os, shutil, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sim_env"))
from bridge_model import load_config
from prior_cloud import write_prior
from plan_survey import Planner, save_plan
from flight_core import Vehicle
from world_model import MeshBridge

cfg = load_config(sys.argv[1]); sc, f0 = cfg["scene"], cfg["flight0"]
out = sc["out_dir"]; map_file = os.path.join(out, "map_points.npy")
log = lambda s: print(s, flush=True)
t_start = time.time()

pilot = MeshBridge(cfg)
pl = Planner(cfg, br=pilot)
plan, info = pl.plan(("exterior", "sections"))
save_plan(pl, plan, info, out, "flight0_route", plot=True)
shutil.copy(os.path.join(out, "flight0_route.npz"), os.path.join(out, "current_plan.npz"))
log(f"flight0 route: {info['length_m']} m, ~{info['duration_min']} min, highest {plan['P'][:, 2].max():.1f} m, "
    f"{100*info['gps_denied_fraction']:.0f}% under the deck")

v = Vehicle(cfg, log)
v.wait_ekf()
v.takeoff(cfg["vehicle"]["takeoff_climb"])
try:
    v.fly(plan, join_window=cfg["vehicle"]["takeoff_climb"] + 1.0, label="flight0")
except Exception as e:                                          # anything goes wrong: back to the take-off point
    log(f"ABORT: {e} - returning to the take-off point")
    home = pilot.spawn + [0, 0, cfg["vehicle"]["takeoff_climb"]]
    hp = Planner(cfg, br=pilot, extra_bounds=np.array([v.pos, home]))
    back, _ = hp.route([v.pos.copy(), home])
    v.fly(back, label="return")
v.land()

# the bridge point cloud from THIS flight's own map ------------------------------------------------
v.settle(cfg["mapping"]["save_every_s"] + 3)
t = time.time()
while os.path.getmtime(map_file) < t:                        # one more save after landing
    v.settle(1.0)
M = np.load(map_file).astype(np.float64)
bridge, sur, meta = write_prior(cfg, M, deck_hint=v.pos.copy(), meta={
    "made_by": "flight0 (piloted route; map from this flight's lidar placed with the drone's own pose estimate)",
    "date": time.strftime("%Y-%m-%d %H:%M"), "flight_minutes": round((time.time() - t_start) / 60, 1),
    "pose_source": "autopilot EKF (GPS / VIO), scans timed by capture",
    "sensor_noise": bool(cfg.get("sensor_noise", {}).get("enabled")),
    "wind_mean_mps": cfg["wind"]["mean_speed"] if cfg.get("wind", {}).get("enabled") else 0})
log(f"bridge cloud: {len(bridge)} points -> {sc['prior_cloud']} (surroundings {len(sur)} coarse points kept separately)")
log(json.dumps(meta["bridge_as_measured"]))
log("flight0 done")

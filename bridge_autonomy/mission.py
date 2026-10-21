#!/usr/bin/env python3
"""Inspection survey: fly <out_dir>/plan.npz (planned by plan_survey.py from the point cloud pc_b2).

Usage: mission.py <config.yaml>"""
import os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_model import load_config
from flight_core import Vehicle

cfg = load_config(sys.argv[1]); out = cfg["scene"]["out_dir"]
plan = dict(np.load(os.path.join(out, "plan.npz")))
shutil.copy(os.path.join(out, "plan.npz"), os.path.join(out, "current_plan.npz"))     # shown in RViz
log = lambda s: print(s, flush=True)
v = Vehicle(cfg, log)
v.wait_ekf()
v.takeoff(cfg["vehicle"]["takeoff_climb"])
try:
    v.fly(plan, join_window=cfg["vehicle"]["takeoff_climb"] + 1.0, label="survey")
except Exception as e:                                          # anything goes wrong: back to the take-off point
    log(f"ABORT: {e} - returning to the take-off point")
    from bridge_model import Bridge
    from plan_survey import Planner
    br = Bridge(cfg)                                            # the obstacle map from pc_b2
    home = br.spawn + [0, 0, cfg["vehicle"]["takeoff_climb"]]
    back, _ = Planner(cfg, br=br, extra_bounds=np.array([v.pos, home])).route([v.pos.copy(), home])
    v.fly(back, label="return")
v.land()
log("mission done")

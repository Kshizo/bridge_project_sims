#!/usr/bin/env python3
"""Bring the drone back to its take-off point and land, from wherever it is.

The route is planned through the obstacle map of the drone's own lidar map (<out_dir>/map_points.npy)
to takeoff_climb above the take-off point (the drone's spawn on the deck), then it lands.
Usage: return_home.py <config.yaml>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_model import Bridge, load_config, spawn_pose
from plan_survey import Planner
from flight_core import Vehicle

cfg = load_config(sys.argv[1]); sc = cfg["scene"]
log = lambda s: print(s, flush=True)
spawn, _ = spawn_pose(sc["world_sdf"], sc["vehicle_model"])
above = spawn + [0, 0, cfg["vehicle"]["takeoff_climb"]]
v = Vehicle(cfg, log)
v.m.set_mode("GUIDED")
v.settle(2)
log(f"at {v.pos.round(2).tolist()}, going to {above.round(2).tolist()}")
pl = Planner(cfg, Bridge(cfg, cloud=np.load(os.path.join(sc["out_dir"], "map_points.npy")).astype(float)),
             extra_bounds=np.array([v.pos, above]))
plan, info = pl.route([v.pos.copy(), above])
v.fly(plan, label="home")
v.land()
log("home and landed")

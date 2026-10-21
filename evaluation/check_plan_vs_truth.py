#!/usr/bin/env python3
"""EVALUATION: how close does a plan (made from the point cloud) really get to the real bridge?
Distance from every plan sample to the ground-truth part meshes (bridge + terrain).

Usage: check_plan_vs_truth.py <config.yaml> [<plan.npz>]"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bridge_autonomy")); sys.path.insert(0, os.path.join(ROOT, "sim_env"))
from bridge_model import load_config, load_obj
from world_model import sample_tris, load_parts

cfg = load_config(sys.argv[1]); sc = cfg["scene"]
plan = np.load(sys.argv[2] if len(sys.argv) > 2 else os.path.join(sc["out_dir"], "plan.npz"))
P, tag, names = plan["P"], plan["tag"], plan["names"]
parts, _ = load_parts(sc["parts_dir"], sc.get("bridge_model"))   # all parts: the region only limits the survey
rng = np.random.default_rng(0)
lo, hi = P.min(0) - 6, P.max(0) + 6
pts = []
for p in parts:
    if p["class"] == "water":
        continue
    T = load_obj(p["obj"])
    T = T[np.all((T.max(1) >= lo) & (T.min(1) <= hi), axis=1)]          # only triangles near the plan
    pts.append(sample_tris(T, 0.25 if p["env"] else 0.05, rng))
tree = cKDTree(np.concatenate(pts))
d, _ = tree.query(P)
water = [p["bbox_max"][2] for p in parts if p["class"] == "water"]
if water:
    d = np.minimum(d, P[:, 2] - max(water))
skip = int(5 / cfg["motion"]["sample_step"])                       # take-off column from the deck
print(f"plan {len(P)} samples: true distance to structure / terrain / water  min {d[skip:].min():.2f} m  "
      f"p1 {np.percentile(d[skip:], 1):.2f} m  median {np.median(d[skip:]):.2f} m   (min_clearance {cfg['safety']['min_clearance']} m)")
for g in sorted(set(tag[tag >= 0])):
    m = tag == g
    print(f"  {names[g]:24s} min {d[m].min():.2f} m  median {np.median(d[m]):.2f} m")

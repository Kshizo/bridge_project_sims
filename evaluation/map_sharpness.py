#!/usr/bin/env python3
"""EVALUATION: how sharp is a map? Distance from every map point near the bridge to the real (ground
truth) bridge surface. A crisp map has p50 ~ the lidar noise and p90 of a few cm; smear shows up as
large p90 / p99.

Usage: map_sharpness.py <config.yaml> <map_points.npy | cloud.ply>"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bridge_autonomy")); sys.path.insert(0, os.path.join(ROOT, "sim_env"))
from bridge_model import load_config, load_obj, read_ply, voxel_down
from world_model import sample_tris, load_parts

cfg = load_config(sys.argv[1]); sc = cfg["scene"]
M = read_ply(sys.argv[2]) if sys.argv[2].endswith(".ply") else np.load(sys.argv[2]).astype(np.float64)
M = voxel_down(M, 0.05)
parts, _ = load_parts(sc["parts_dir"], sc.get("bridge_model"), sc.get("region"))
rng = np.random.default_rng(0)
lo, hi = M.min(0) - 1, M.max(0) + 1
S, E = [], []
for p in parts:
    T = load_obj(p["obj"])
    T = T[np.all((T.max(1) >= lo) & (T.min(1) <= hi), axis=1)]
    (E if p["env"] else S).append(sample_tris(T, 0.1 if p["env"] else 0.02, rng))
d, _ = cKDTree(np.concatenate(S)).query(M, distance_upper_bound=1.0)
de, _ = cKDTree(np.concatenate(E)).query(M, distance_upper_bound=2.0) if E else (np.full(len(M), np.inf), None)
water = [p["bbox_max"][2] for p in parts if p["class"] == "water"]
dw = np.abs(M[:, 2] - max(water)) if water else np.full(len(M), np.inf)
near = np.isfinite(d) & (d < de) & (d < dw)                  # points that belong to the bridge, not the banks / water
p = np.percentile(d[near], [50, 90, 99])
print(f"{len(M)} points (5 cm), {near.sum()} within 1 m of the bridge ({100*near.mean():.0f}%): "
      f"distance to the real surface p50 {p[0]:.3f}  p90 {p[1]:.3f}  p99 {p[2]:.3f} m; "
      f"within 5 cm {100*np.mean(d[near] < 0.05):.0f}%, within 10 cm {100*np.mean(d[near] < 0.10):.0f}%")

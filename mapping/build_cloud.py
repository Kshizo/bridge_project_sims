#!/usr/bin/env python3
"""Rebuild a map offline from the raw scans the mapper recorded (<out_dir>/raw_scans: body-frame points
+ the autopilot's EKF pose at each scan's capture time). No ground truth.

Usage: build_cloud.py <config.yaml> <out.npy> [--max-range R] [--min-range r] [--voxel v]"""
import argparse, glob, os, sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge_autonomy"))
from bridge_model import load_config, voxel_down


def build(raw_dir, max_range, min_range, voxel, lidar_t=None):
    out, n_scans = [], 0
    for f in sorted(glob.glob(os.path.join(raw_dir, "chunk_*.npz"))):
        d = np.load(f)
        pts = d["pts_cm"].astype(np.float64) / 100.0
        starts = np.r_[0, np.cumsum(d["n"])[:-1]]
        for p, q, a, n in zip(d["p"], d["q"], starts, d["n"]):
            B = pts[a:a + n]
            r = np.linalg.norm(B - (lidar_t if lidar_t is not None else 0), axis=1)
            B = B[(r <= max_range) & (r >= min_range)]
            out.append(voxel_down(B @ Rotation.from_quat(q).as_matrix().T + p, voxel))
            n_scans += 1
        if len(out) > 400:
            out = [voxel_down(np.concatenate(out), voxel)]
    return voxel_down(np.concatenate(out), voxel), n_scans


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("out")
    ap.add_argument("--max-range", type=float, default=None); ap.add_argument("--min-range", type=float, default=None)
    ap.add_argument("--voxel", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config); mp = cfg["mapping"]
    M, n = build(os.path.join(cfg["scene"]["out_dir"], "raw_scans"), a.max_range or mp["max_range"],
                 a.min_range or mp["min_range"], a.voxel or mp["voxel"])
    np.save(a.out, M.astype(np.float32))
    print(f"{n} scans -> {len(M)} points -> {a.out}")

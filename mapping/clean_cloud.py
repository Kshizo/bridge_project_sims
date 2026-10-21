#!/usr/bin/env python3
"""Clean a lidar map into a crisp point cloud (no ground truth involved).

  1. statistical outlier removal: drop points whose mean distance to their k nearest neighbours is far
     above the typical value (ghost returns, stray scans)
  2. surface consolidation (moving least squares, plane fit): a surface seen on many passes is a few
     centimetres thick because every pass is posed slightly differently; each point is projected onto
     the plane through its neighbours where that neighbourhood is planar, so the surface becomes thin
  3. voxel grid

Usage: clean_cloud.py <config.yaml> <in.ply | map_points.npy> <out.ply>   (settings: config cloud_cleaning)"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge_autonomy"))
from bridge_model import load_config, read_ply, write_ply, voxel_down


def remove_outliers(P, k, std_ratio, chunk=500000):
    tree = cKDTree(P)
    md = np.empty(len(P))
    for a in range(0, len(P), chunk):
        d, _ = tree.query(P[a:a + chunk], k=k + 1)
        md[a:a + chunk] = d[:, 1:].mean(1)
    keep = md <= md.mean() + std_ratio * md.std()
    return P[keep]


def consolidate(P, k, planarity, chunk=300000):
    """Project points onto the local plane of their k neighbours where the neighbourhood is planar."""
    tree = cKDTree(P)
    out = P.copy()
    for a in range(0, len(P), chunk):
        _, nb = tree.query(P[a:a + chunk], k=k)
        Q = P[nb]                                                     # (n, k, 3)
        mu = Q.mean(1)
        C = np.einsum("nki,nkj->nij", Q - mu[:, None], Q - mu[:, None]) / k
        w, V = np.linalg.eigh(C)                                      # ascending eigenvalues
        n = V[:, :, 0]
        flat = w[:, 0] < planarity * w[:, 1]                          # thin along the normal compared with in-plane
        d = np.einsum("ni,ni->n", P[a:a + chunk] - mu, n)
        out[a:a + chunk][flat] -= (d[flat, None] * n[flat])
    return out


def clean(P, cc):
    n0 = len(P)
    P = remove_outliers(P, cc["outlier_k"], cc["outlier_std"])
    n1 = len(P)
    for _ in range(cc["consolidate_passes"]):
        P = consolidate(P, cc["consolidate_k"], cc["planarity"])
    P = voxel_down(P, cc["voxel"])
    print(f"clean: {n0} -> {n1} after outlier removal -> {len(P)} after consolidation + {cc['voxel']} m voxels", flush=True)
    return P


if __name__ == "__main__":
    cfg = load_config(sys.argv[1])
    src = sys.argv[2]
    P = read_ply(src) if src.endswith(".ply") else np.load(src).astype(np.float64)
    write_ply(sys.argv[3], clean(P, cfg["cloud_cleaning"]), comment="cleaned lidar map")

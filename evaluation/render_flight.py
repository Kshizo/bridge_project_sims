#!/usr/bin/env python3
"""EVALUATION: render a point cloud with the flown paths (true, from the referee log, and the drone's
own estimate) from several viewpoints -> PNG.

Usage: render_flight.py <cloud.ply | map_dir> <out.png> [--eval <map_dir>/eval/truth_log.npy]"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge_autonomy"))
from bridge_model import read_ply

ap = argparse.ArgumentParser(); ap.add_argument("cloud"); ap.add_argument("out"); ap.add_argument("--eval", default=None)
a = ap.parse_args()
P = read_ply(a.cloud) if os.path.isfile(a.cloud) else np.load(os.path.join(a.cloud, "map_points.npy")).astype(float)
rng = np.random.default_rng(0)
S = P[rng.choice(len(P), min(250000, len(P)), replace=False)]
L = np.load(a.eval) if a.eval and os.path.exists(a.eval) else None
c = np.median(S, 0); r = np.percentile(np.abs(S - c), 98, axis=0).max()
views = [("oblique", 25, -60), ("side (along the bridge)", 2, 0), ("top", 89, -90), ("end-on", 2, 90), ("from below", -20, -60)]
fig = plt.figure(figsize=(26, 11))
for k, (title, el, az) in enumerate(views):
    ax = fig.add_subplot(2, 3, k + 1, projection="3d")
    ax.scatter(*S.T, s=0.05, c=S[:, 2], cmap="viridis", alpha=0.6)
    if L is not None and len(L):
        ax.plot(*L[:, 1:4].T, c="white", lw=0.6, label="true path")
        ok = np.isfinite(L[:, 7])
        ax.plot(*L[ok, 7:10].T, c="orange", lw=0.6, label="drone's estimate")
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r / 3, c[2] + r / 3)
    ax.view_init(el, az); ax.set_title(title); ax.set_facecolor("0.15")
ax = fig.add_subplot(2, 3, 6)
band = S[np.abs(S[:, 0] - c[0]) < 2.0]                                   # a 4 m slice across the bridge axis
ax.scatter(band[:, 1], band[:, 2], s=0.05, c="k"); ax.set_aspect("equal"); ax.set_title("4 m slice (y-z)")
plt.tight_layout(); plt.savefig(a.out, dpi=70)
print("wrote", a.out)

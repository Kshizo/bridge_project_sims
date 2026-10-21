#!/usr/bin/env python3
"""Paste real CODEBRIM defect photos (crack / spalling / exposed rebar) onto Scene-3's
concrete as thin decal quads, for testing a drone-side defect detector.

Each decal is ray-cast onto the bridge mesh so it sits flush on a real face (lifted
OFFSET metres to avoid z-fighting), and its annotation polygons are projected into world
coordinates in defects_gt.json, so detections can be scored against ground truth.

Usage: tools_scene3_defects.py <scene3_tex/scene3.obj> <images_dir> <annotations.json> <out_dir>
Then include model://<out_dir>/defects.obj as a static visual in scene3_ardupilot.sdf."""
import json, os, sys, shutil
import numpy as np

obj, img_dir, ann_path, out = sys.argv[1:5]
OFFSET = 0.015       # m off the surface
SIZE = 2.0           # default decal edge, m (CODEBRIM tiles are close-ups, ~1-2 m of wall)

X, Y, Z = np.eye(3)
# (id, class, image, centre guess, outward normal, up, size)
PIER_N, PIER_S = (0, -1, 0), (0, 1, 0)     # pier faces at y~155 / y~-217 that face mid-span
DECALS = [
    # north pier wall (y~155) + its columns
    ("N1", "crack",         "image_0000652_1712_0.jpg",    (-5.3, 155, -5.5), PIER_N, Z, SIZE),
    ("N2", "spalling",      "image_0000683_1792_0.jpg",    (-2.0, 155, -5.5), PIER_N, Z, SIZE),
    ("N3", "exposed_rebar", "image_0000587_0_0.jpg",       ( 1.3, 155, -5.5), PIER_N, Z, SIZE),
    ("N4", "crack",         "image_0000646_896_0.jpg",     (-20.3, 155, -6.0), PIER_N, Z, 1.8),
    ("N5", "spalling",      "image_0000587_0_1792.jpg",    ( 15.8, 155, -6.0), PIER_N, Z, 1.8),
    # south pier wall (y~-217) + its columns
    ("S1", "exposed_rebar", "image_0000683_1792_896.jpg",  ( 1.3, -217, -5.5), PIER_S, Z, SIZE),
    ("S2", "crack",         "image_0000652_0_2624.jpg",    (-2.0, -217, -5.5), PIER_S, Z, SIZE),
    ("S3", "spalling",      "image_0000683_896_0.jpg",     (-5.3, -217, -5.5), PIER_S, Z, SIZE),
    ("S4", "exposed_rebar", "image_0000683_896_896.jpg",   ( 15.8, -217, -6.0), PIER_S, Z, 1.8),
    ("S5", "crack",         "image_0000652_896_0.jpg",     (-20.3, -217, -6.0), PIER_S, Z, 1.8),
    # outer girder faces of the mid-span arch (x~19.4 east, x~-23.4 west)
    ("E1", "crack",         "image_0000652_896_1792.jpg",  (19.4, 57.0, -6.0), X, Z, SIZE),
    ("E2", "exposed_rebar", "image_0000587_896_0.jpg",     (19.4, 61.0, -6.0), X, Z, SIZE),
    ("W1", "spalling",      "image_0000587_0_896.jpg",     (-23.4, 58.0, -6.0), -X, Z, SIZE),
    ("W2", "crack",         "image_0000652_896_896.jpg",   (-23.4, 62.0, -6.0), -X, Z, SIZE),
    # deck soffit (z~-2.6), seen from below
    ("U1", "spalling",      "image_0000579_0_896.jpg",     (-10.0, -40.0, -2.6), -Z, Y, SIZE),
    ("U2", "exposed_rebar", "image_0000683_1792_1712.jpg", ( 5.0, -20.0, -2.6), -Z, Y, SIZE),
]

# ---- load mesh (triangulated fan, all materials except water) ----
V, T, mat = [], [], None
for l in open(obj):
    if l.startswith("v "):
        V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith("usemtl"):
        mat = l.split()[1]
    elif l.startswith("f ") and mat != "Water":
        idx = [int(t.split("/")[0]) - 1 for t in l.split()[1:]]
        T += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
V, T = np.array(V), np.array(T)
A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
lo, hi = np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)


def raycast(o, d, reach=3.0):
    """Nearest hit distance along d from o (Moller-Trumbore), or None."""
    m = np.all((lo <= o + reach) & (hi >= o - reach), axis=1)
    a, b, c = A[m], B[m], C[m]
    e1, e2 = b - a, c - a
    p = np.cross(d, e2); det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-9
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    s = o - a; u = (s * p).sum(1) * inv
    q = np.cross(s, e1); v = (q * d).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return t[hit].min() if hit.any() else None


anns = {e["file"]: e for e in json.load(open(ann_path))}
os.makedirs(os.path.join(out, "textures"), exist_ok=True)
objl, mtll, gt, bad = ["mtllib defects.mtl"], [], [], []
vt_base = 0
for k, (did, cls, img, c, n, up, size) in enumerate(DECALS):
    c, n, up = np.array(c, float), np.array(n, float), np.array(up, float)
    r = np.cross(up, n)                       # viewer's right when looking at the face
    # snap the centre onto the surface, then require all corners to land on the same plane
    t0 = raycast(c + n * 1.5, -n)
    if t0 is None:
        bad.append(f"{did}: no surface under centre {c}"); continue
    c = c + n * (1.5 - t0)
    for sx, sy in [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]:
        p = c + r * sx * size + up * sy * size
        t = raycast(p + n * 0.5, -n, 1.0)
        if t is None or abs(t - 0.5) > 0.05:
            bad.append(f"{did}: corner {sx,sy} off-surface (t={t})")
    c = c + n * OFFSET
    corners = [c + r * sx * size + up * sy * size for sx, sy in [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]]

    shutil.copy(os.path.join(img_dir, img), os.path.join(out, "textures", f"{did}.jpg"))
    mtll += [f"newmtl defect_{did}", "Ka 1 1 1", "Kd 1 1 1", "Ks 0 0 0", f"map_Kd textures/{did}.jpg", ""]
    objl += [f"g defect_{did}", f"usemtl defect_{did}"]
    objl += ["v %.4f %.4f %.4f" % tuple(p) for p in corners]
    objl += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1", "vn %.4f %.4f %.4f" % tuple(n)]
    i = 4 * k + 1
    objl += [f"f {i}/{i}/{k+1} {i+1}/{i+1}/{k+1} {i+2}/{i+2}/{k+1}",
             f"f {i}/{i}/{k+1} {i+2}/{i+2}/{k+1} {i+3}/{i+3}/{k+1}"]

    # ground truth: pixel (x right, y down) -> world
    e = anns.get(img, {"counts": {}, "instances": []})
    W = H = 1024.0
    def w(x, y): return (c + r * (x / W - .5) * size + up * (.5 - y / H) * size).round(3).tolist()
    gt.append({
        "id": did, "label": cls, "image": img, "size_m": size,
        "centre": c.round(3).tolist(), "normal": n.tolist(), "up": up.tolist(),
        "corners": [p.round(3).tolist() for p in corners],
        "counts": e["counts"],
        "instances": [{"class": ins["class"],
                       "polygon_world": [w(*xy) for xy in zip(ins["segmentation"][0][::2], ins["segmentation"][0][1::2])]}
                      for ins in e["instances"]],
    })

open(os.path.join(out, "defects.obj"), "w").write("\n".join(objl) + "\n")
open(os.path.join(out, "defects.mtl"), "w").write("\n".join(mtll))
json.dump({"frame": "scene3 world ENU, metres", "decals": gt}, open(os.path.join(out, "defects_gt.json"), "w"), indent=1)
print(f"wrote {len(gt)} decals to {out}")
for g in gt: print(f"  {g['id']:3s} {g['label']:14s} centre {g['centre']}")
if bad:
    print("WARNINGS:"); print("\n".join("  " + b for b in bad))

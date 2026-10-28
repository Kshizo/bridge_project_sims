"""Paste real CODEBRIM defect photos onto a scene's bridge as thin decal meshes, for testing a
drone-side defect detector (used by sceneN/tools_sceneN_defects.py).

Each decal is ray-cast onto the bridge parts so it sits flush on a real face (lifted OFFSET m to
avoid z-fighting); a decal with a radius is wrapped round a round column instead. Its YOLO-seg
annotation polygons are projected into world coordinates in defects_gt.json, so detections can be
scored against ground truth. Images should be CODEBRIM val tiles (1024 px, never trained on).

A decal is (id, class, image, centre, outward normal, up, size_m[, column_radius_m]):
  plane:  centre is a guess near the face, snapped onto it along the normal
  column: centre is a point on the column axis, the normal picks the side it faces"""
import json, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_model import load_parts, load_obj

OFFSET = 0.015       # m off the surface
NSEG = 12            # facets across a wrapped column decal
TILE = 1024.0        # CODEBRIM tile size, px
CLASSES = ["crack", "spalling", "exposed_rebar"]
X, Y, Z = np.eye(3)


def load_labels(path):
    """YOLO-seg label file -> counts + instances with pixel polygons."""
    e = {"counts": dict.fromkeys(CLASSES, 0), "instances": []}
    for l in open(path):
        p = l.split()
        if not p:
            continue
        cls = CLASSES[int(p[0])]
        e["counts"][cls] += 1
        xy = [float(v) * TILE for v in p[1:]]
        e["instances"].append({"class": cls, "polygon_px": list(zip(xy[::2], xy[1::2]))})
    return e


class Mesh:
    """All bridge (non-environment, non-defect) triangles of a parts dir, for ray casting."""
    def __init__(self, parts_dir):
        parts, _ = load_parts(parts_dir)
        T = np.concatenate([load_obj(p["obj"]) for p in parts if not p["env"] and p["class"] != "defect"])
        self.A, self.B, self.C = T[:, 0], T[:, 1], T[:, 2]
        self.lo, self.hi = T.min(1), T.max(1)

    def raycast(self, o, d, reach=3.0):
        """Nearest hit distance along d from o (Moller-Trumbore), or None."""
        m = np.all((self.lo <= o + reach) & (self.hi >= o - reach), axis=1)
        a, b, c = self.A[m], self.B[m], self.C[m]
        e1, e2 = b - a, c - a
        p = np.cross(d, e2); det = (e1 * p).sum(1)
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        s = o - a; u = (s * p).sum(1) * inv
        q = np.cross(s, e1); v = (q * d).sum(1) * inv
        t = (e2 * q).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        return t[hit].min() if hit.any() else None


def build(scene, parts_dir, decals, img_dir, lbl_dir, out):
    mesh = Mesh(parts_dir)
    os.makedirs(os.path.join(out, "textures"), exist_ok=True)
    objl, mtll, gt, bad = ["mtllib defects.mtl"], [], [], []
    nv = 0
    for did, cls, img, c, n, up, size, *rad in decals:
        c, n, up = np.array(c, float), np.array(n, float), np.array(up, float)
        if rad:
            # wrapped round a column: u is arc length round the axis through c
            R = rad[0] + OFFSET
            phi0 = np.arctan2(n[1], n[0])
            def surf(u, v, c=c, size=size, R=R, phi0=phi0):
                a = phi0 + u * size / R                         # +u = viewer's right when facing it
                r = np.array([np.cos(a), np.sin(a), 0])
                return c + R * r + Z * v * size, r
            t = mesh.raycast(c + n * 1.5, -n)
            if t is None or abs((1.5 - t) - rad[0]) > 0.02:
                bad.append(f"{did}: column surface not at R={rad[0]} (t={t})")
            nseg = NSEG
        else:
            r = np.cross(up, n)                                 # viewer's right when facing the face
            t0 = mesh.raycast(c + n * 1.5, -n)
            if t0 is None:
                bad.append(f"{did}: no surface under centre {c.tolist()}"); continue
            c = c + n * (1.5 - t0)
            for sx, sy in [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]:
                t = mesh.raycast(c + r * sx * size + up * sy * size + n * 0.5, -n, 1.0)
                if t is None or abs(t - 0.5) > 0.05:
                    bad.append(f"{did}: corner {sx, sy} off-surface (t={t})")
            c = c + n * OFFSET
            def surf(u, v, c=c, r=r, up=up, n=n, size=size):
                return c + r * u * size + up * v * size, n
            nseg = 1

        shutil.copy(os.path.join(img_dir, img), os.path.join(out, "textures", f"{did}.jpg"))
        mtll += [f"newmtl defect_{did}", "Ka 1 1 1", "Kd 1 1 1", "Ks 0 0 0", f"map_Kd textures/{did}.jpg", ""]
        objl += [f"g defect_{did}", f"usemtl defect_{did}"]
        for j in range(nseg + 1):
            for v in (0.0, 1.0):
                p, nrm = surf(j / nseg - .5, v - .5)
                objl += ["v %.4f %.4f %.4f" % tuple(p), "vt %.4f %.1f" % (j / nseg, v), "vn %.4f %.4f %.4f" % tuple(nrm)]
        for j in range(nseg):
            a, b, cc, d = nv + 2 * j + 1, nv + 2 * j + 3, nv + 2 * j + 4, nv + 2 * j + 2   # (u,0) (u+,0) (u+,1) (u,1)
            objl += [f"f {a}/{a}/{a} {b}/{b}/{b} {cc}/{cc}/{cc}", f"f {a}/{a}/{a} {cc}/{cc}/{cc} {d}/{d}/{d}"]
        nv += 2 * (nseg + 1)

        # ground truth: pixel (x right, y down) -> world
        e = load_labels(os.path.join(lbl_dir, os.path.splitext(img)[0] + ".txt"))
        def w(x, y): return surf(x / TILE - .5, .5 - y / TILE)[0].round(3).tolist()
        centre, normal = surf(0, 0)
        gt.append({
            "id": did, "label": cls, "image": img, "size_m": size,
            "surface": "cylinder" if rad else "plane",
            "centre": centre.round(3).tolist(), "normal": normal.round(4).tolist(), "up": up.tolist(),
            "corners": [w(0, TILE), w(TILE, TILE), w(TILE, 0), w(0, 0)],
            "counts": e["counts"],
            "instances": [{"class": i["class"], "polygon_world": [w(*xy) for xy in i["polygon_px"]]}
                          for i in e["instances"]],
        })

    open(os.path.join(out, "defects.obj"), "w").write("\n".join(objl) + "\n")
    open(os.path.join(out, "defects.mtl"), "w").write("\n".join(mtll))
    json.dump({"frame": f"{scene} world ENU, metres", "decals": gt},
              open(os.path.join(out, "defects_gt.json"), "w"), indent=1)
    print(f"wrote {len(gt)} decals to {out}")
    for g in gt:
        print(f"  {g['id']:3s} {g['label']:9s} {g['surface']:8s} centre {g['centre']}")
    if bad:
        print("WARNINGS:"); print("\n".join("  " + b for b in bad))
    return not bad

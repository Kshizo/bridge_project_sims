#!/usr/bin/env python3
"""Split Scene-1 into per-component bridge parts + environment pieces (see BRIDGE_PARTS_SPEC.md).

Source is the textured SketchUp OBJ export (Y-up, mm, group hierarchy). Its hierarchy
already names some objects (Component_4_* = piers, elastomeric_bearing*, Bolt*,
Component_8_* = vehicles, Component_9_1 = a cabin, ske1D* = road barriers), but the deck,
girders, parapets, east abutment and the terrain are all fused into one ungrouped mesh,
so those are split geometrically:
  bridge runs along +X, deck ends x=-232.0 / 56.66, deck slab z 11.29-11.46,
  3 I-girders (y centres -10.53 / -8.94 / -7.34) z 9.61-11.29, parapets above the deck,
  piers z 6.64-9.92 (column below z 8.86, cap above); each pier cap carries an inspection
  platform whose steel handrails (Component_1/2/3_*) become railing_17..23.
Girders/deck/parapets are clipped at pier centrelines so each span is its own part.

Usage: tools_scene1_parts.py <Scene-1.obj> <scene1_tex dir (mtl+textures)> <out_dir>
Writes <out_dir>/<part>.obj (+ collision/<part>.obj convex hulls), parts.mtl, textures/,
parts.json and sdf_parts.xml (the <model>s to paste into the world)."""
import json, os, re, shutil, sys
from collections import defaultdict
import numpy as np
from scipy.spatial import ConvexHull

src, texdir, out = sys.argv[1:4]
LABEL = {"deck": 1, "girder": 2, "cross_beam": 3, "pier_cap": 4, "pier_column": 5, "footing": 6,
         "abutment": 7, "railing": 8, "bearing": 9, "arch_truss": 10, "bridge_misc": 11,
         "terrain": 20, "water": 21, "vegetation": 22, "vehicle": 23, "road": 24, "other": 25}

X_W, X_E = -232.0, 56.66                     # deck ends (abutment back faces)
GIRDER_Y = (-10.53, -8.94, -7.34)            # south -> north
GIRDER_HALF = 0.45
DECK_Z0, DECK_Z1 = 11.28, 11.47
GIRDER_Z0 = 9.55
PIER_CAP_Z = 8.86

# ---------------------------------------------------------------- parse (same transform as tools_skp_obj_to_gz.py)
V, VT, VN = [], [], []
groups, mats = {}, {}
tv, tt, tn, tg, tm = [], [], [], [], []
g = m = 0
with open(src, errors="replace") as f:
    for line in f:
        c = line[:2]
        if c == "v ":
            x, y, z = map(float, line.split()[1:4]); V.append((x * 1e-3, -z * 1e-3, y * 1e-3))
        elif c == "vt":
            p = line.split(); VT.append((float(p[1]), float(p[2])))
        elif c == "vn":
            x, y, z = map(float, line.split()[1:4]); VN.append((x, -z, y))
        elif c == "f ":
            p = [q.split("/") for q in line.split()[1:]]
            vi = [int(q[0]) - 1 for q in p]
            ti = [int(q[1]) - 1 if len(q) > 1 and q[1] else -1 for q in p]
            ni = [int(q[2]) - 1 if len(q) > 2 and q[2] else -1 for q in p]
            for k in range(1, len(p) - 1):
                tv.append((vi[0], vi[k], vi[k + 1])); tt.append((ti[0], ti[k], ti[k + 1]))
                tn.append((ni[0], ni[k], ni[k + 1])); tg.append(g); tm.append(m)
        elif c == "g ":
            g = groups.setdefault(line[2:].strip(), len(groups))
        elif line.startswith("usemtl"):
            m = mats.setdefault(line.split(None, 1)[1].strip(), len(mats))
V = np.array(V); V[:, 2] -= V[:, 2].min()
VT = np.array(VT + [(0.0, 0.0)]); VN = np.array(VN + [(0.0, 0.0, 1.0)])
tv, tt, tn = np.array(tv), np.array(tt), np.array(tn)
tg, tm = np.array(tg), np.array(tm)
MATS = list(mats)
paths = [[x for x in s.split()[1:] if x != "Model"][::-1] for s in groups]   # top-level first

# triangle corner attributes: P (n,3,3), UV (n,3,2), N (n,3,3), mat (n,)
P = V[tv]; UV = VT[tt]; NN = VN[tn]

# ---------------------------------------------------------------- 1. assign by hierarchy
n = len(tv)
kind = np.full(n, "", dtype=object)     # coarse source kind
inst = np.full(n, "", dtype=object)     # instance key from hierarchy
for gi, p in enumerate(paths):
    sel = tg == gi
    if not sel.any():
        continue
    top = p[0] if p else ""
    names = " ".join(p)
    if not p:
        k, key = "root", ""
    elif top.startswith("Component_8"):
        k, key = "vehicle", top
    elif top.startswith("Component_9") or top.startswith("ske1D"):
        k, key = "other", top
    elif "elastomeric_bearing" in names:
        k, key = "bearing", next(x for x in p if x.startswith("elastomeric_bearing"))
    elif any(x.startswith("Bolt") for x in p):
        k, key = "bolt", ""
    elif any(x.startswith("Component_4_") for x in p):
        pier = next(x for x in p if x.startswith("Component_4_"))
        k, key = ("pier" if p[-1] == pier else "pier_detail"), pier
    elif top.startswith("Component_7"):
        k, key = "abut_w", top
    elif top.startswith("Component_6") and len(p) == 1:
        k, key = "super", ""
    else:
        k, key = "unknown", top
    kind[sel] = k; inst[sel] = key
assert not (kind == "unknown").any(), set(inst[kind == "unknown"])

# ---------------------------------------------------------------- 2. clipping helpers
def clip_poly(poly, c, keep_left, ax=0):
    """Sutherland-Hodgman on the plane p[ax]=c. poly: list of (p, uv, n)."""
    res = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        da, db = a[0][ax] - c, b[0][ax] - c
        ina, inb = (da <= 0) if keep_left else (da >= 0), (db <= 0) if keep_left else (db >= 0)
        if ina:
            res.append(a)
        if ina != inb:
            t = da / (da - db)
            res.append(tuple(a[j] + t * (b[j] - a[j]) for j in range(3)))
    return res


def cut(tris, planes, ax=0):
    """tris: list of (P,UV,N,mat). Split every triangle crossing any plane p[ax]=c."""
    for c in planes:
        nxt = []
        for P3, U3, N3, mt in tris:
            xs = P3[:, ax]
            if xs.min() < c - 1e-6 and xs.max() > c + 1e-6:
                poly = list(zip(P3, U3, N3))
                for side in (True, False):
                    q = clip_poly(poly, c, side, ax)
                    for k in range(1, len(q) - 1):
                        a, b, d = q[0], q[k], q[k + 1]
                        nxt.append((np.array([a[0], b[0], d[0]]), np.array([a[1], b[1], d[1]]),
                                    np.array([a[2], b[2], d[2]]), mt))
            else:
                nxt.append((P3, U3, N3, mt))
        tris = nxt
    return tris


parts = defaultdict(list)       # part name -> list of (P,UV,N,mat)
pclass = {}


def add(name, cls, tri):
    parts[name].append(tri); pclass[name] = cls


def T(i):
    return (P[i], UV[i], NN[i], tm[i])


# ---------------------------------------------------------------- 3. substructure
piers = sorted({k for k in inst[(kind == "pier") | (kind == "pier_detail")]},
               key=lambda k: P[inst == k][:, :, 0].mean())
pier_x = {k: P[(inst == k) & (kind == "pier")][:, :, 0].mean() for k in piers}
PIER_X = [pier_x[k] for k in piers]
for i, k in enumerate(piers, 1):
    for tri in cut([T(t) for t in np.where((inst == k) & (kind == "pier"))[0]], [PIER_CAP_Z], ax=2):
        col = tri[0][:, 2].mean() < PIER_CAP_Z
        add(f"pier_column_{i:02d}" if col else f"pier_cap_{i:02d}", "pier_column" if col else "pier_cap", tri)
    for t in np.where((inst == k) & (kind == "pier_detail"))[0]:      # inspection-platform handrails
        add(f"railing_{16 + i:02d}", "railing", T(t))                   # 01-16 = deck parapets

bkeys = sorted({k for k in inst[kind == "bearing"]},
               key=lambda k: (round(P[inst == k][:, :, 0].mean(), 0), P[inst == k][:, :, 1].mean()))
bcent = np.array([P[inst == k].reshape(-1, 3).mean(0) for k in bkeys])
bname = {k: f"bearing_{i:02d}" for i, k in enumerate(bkeys, 1)}
for t in np.where(kind == "bearing")[0]:
    add(bname[inst[t]], "bearing", T(t))
bolt = np.where(kind == "bolt")[0]                     # anchor bolts -> nearest bearing
near = np.argmin(((P[bolt].mean(1)[:, None, :2] - bcent[None, :, :2]) ** 2).sum(-1), 1)
for t, j in zip(bolt, near):
    add(bname[bkeys[j]], "bearing", T(t))

for t in np.where(kind == "abut_w")[0]:
    add("abutment_01", "abutment", T(t))
for t in np.where(kind == "vehicle")[0]:
    add("env_vehicle", "vehicle", T(t))
for t in np.where(kind == "other")[0]:
    add("env_other", "other", T(t))

# ---------------------------------------------------------------- 4. fused root mesh + superstructure
mname = np.array(MATS, dtype=object)[tm]
root = kind == "root"
for t in np.where(root & (mname == "_13"))[0]:
    add("env_water", "water", T(t))
for t in np.where(root & np.isin(mname, ["_19", "_18", "FrontColor"]))[0]:
    add("env_terrain", "terrain", T(t))
for t in np.where(root & (mname == "_14"))[0]:
    add("env_road", "road", T(t))

rest = np.where((root & ~np.isin(mname, ["_13", "_19", "_18", "FrontColor", "_14"])) | (kind == "super"))[0]
cuts = [X_W, X_E] + PIER_X
pieces = cut(cut([T(t) for t in rest], cuts), [DECK_Z1 - 0.005], ax=2)   # parapet faces run down the slab edge
supports = [X_W] + PIER_X + [X_E]
cross = []
for P3, U3, N3, mt in pieces:
    c = P3.mean(0); x, y, z = c
    ylo, yhi = P3[:, 1].min(), P3[:, 1].max()
    if abs(y + 8.94) > 7 or c[2] < 6.5:                         # terrain block side faces
        add("env_terrain", "terrain", (P3, U3, N3, mt)); continue
    if x < X_W or x > X_E:                                      # beyond the deck ends
        if x > X_E and x < 58.5 and P3[:, 2].max() < DECK_Z0 - 0.05:   # east abutment wings
            add("abutment_02", "abutment", (P3, U3, N3, mt))
        else:                                                   # approach road + its parapets
            add("env_road", "road", (P3, U3, N3, mt))
        continue
    if ylo < -11.7 or yhi > -6.1:                               # east abutment back/wing walls
        add("abutment_01" if x < 0 else "abutment_02", "abutment", (P3, U3, N3, mt)); continue
    span = int(np.searchsorted(supports, x)) - 1               # 0..7
    span = min(max(span, 0), len(supports) - 2)
    if c[2] > DECK_Z1 and MATS[mt] != "_12":
        side = 0 if y < -8.94 else 1
        add(f"railing_{span * 2 + side + 1:02d}", "railing", (P3, U3, N3, mt)); continue
    edge = yhi - ylo < 0.05 and abs(y + 8.94) > 2.6                # slab fascia below its soffit
    if P3[:, 2].min() >= DECK_Z0 - 0.01 or MATS[mt] == "_12" or edge:
        add(f"deck_{span + 1:02d}", "deck", (P3, U3, N3, mt)); continue
    if P3[:, 2].min() >= GIRDER_Z0 and P3[:, 2].max() <= DECK_Z0 + 0.02:
        gi = [j for j, gy in enumerate(GIRDER_Y) if ylo >= gy - GIRDER_HALF and yhi <= gy + GIRDER_HALF]
        if gi:
            add(f"girder_{span * 3 + gi[0] + 1:02d}", "girder", (P3, U3, N3, mt))
        else:
            cross.append((P3, U3, N3, mt))
        continue
    # below the girders at the ends: abutments
    add("abutment_01" if x < 0 else "abutment_02", "abutment", (P3, U3, N3, mt))

# cross beams: 1-D clusters along X (diaphragms over supports + intermediate bracing)
cross.sort(key=lambda t: t[0][:, 0].mean())
k, last = 0, None
for tri in cross:
    x = tri[0][:, 0].mean()
    if last is None or x - last > 2.0:
        k += 1
    last = x
    add(f"cross_beam_{k:02d}", "cross_beam", tri)

# ---------------------------------------------------------------- 5. write
os.makedirs(os.path.join(out, "collision"), exist_ok=True)
for d_ in (out, os.path.join(out, "collision")):                 # drop parts from earlier runs
    for f_ in os.listdir(d_):
        if f_.endswith(".obj"):
            os.remove(os.path.join(d_, f_))
if os.path.isdir(os.path.join(out, "textures")):
    shutil.rmtree(os.path.join(out, "textures"))
shutil.copytree(os.path.join(texdir, "textures"), os.path.join(out, "textures"))
shutil.copy(os.path.join(texdir, "scene1.mtl"), os.path.join(out, "parts.mtl"))


def write_obj(path, tris):
    with open(path, "w") as o:
        o.write("mtllib parts.mtl\n")
        byM = defaultdict(list)
        for tri in tris:
            byM[tri[3]].append(tri)
        vi = 0
        for mt, tl in byM.items():
            o.write(f"g {re.sub(r'[^A-Za-z0-9_]', '_', MATS[mt])}\nusemtl {MATS[mt]}\n")
            buf = []
            for P3, U3, N3, _ in tl:
                for j in range(3):
                    buf.append(f"v {P3[j,0]:.5f} {P3[j,1]:.5f} {P3[j,2]:.5f}\nvt {U3[j,0]:.6f} {U3[j,1]:.6f}\nvn {N3[j,0]:.4f} {N3[j,1]:.4f} {N3[j,2]:.4f}\n")
                buf.append(f"f {vi+1}/{vi+1}/{vi+1} {vi+2}/{vi+2}/{vi+2} {vi+3}/{vi+3}/{vi+3}\n")
                vi += 3
            o.write("".join(buf))


def write_hull(path, pts):
    lo, hi = pts.min(0), pts.max(0)
    if len(pts) < 4 or (hi - lo).min() < 0.02:                # flat: pad to a thin slab
        pad = np.maximum(0.02 - (hi - lo), 0) / 2
        lo, hi = lo - pad, hi + pad
        pts = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]) if len(pts) < 4 else np.vstack([pts, lo, hi])
    h = ConvexHull(pts, qhull_options="QJ")       # joggle: some parts are near-planar
    with open(path, "w") as o:
        o.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in pts[h.vertices])
        idx = {v: i + 1 for i, v in enumerate(h.vertices)}
        cen = pts[h.vertices].mean(0)
        for k, s in enumerate(h.simplices, 1):          # dartsim needs normals on every vertex
            a, b, c = pts[s]
            nrm = np.cross(b - a, c - a)
            if np.dot(nrm, a - cen) < 0:
                s, nrm = s[[0, 2, 1]], -nrm
            nrm /= np.linalg.norm(nrm) + 1e-12
            o.write(f"vn {nrm[0]:.5f} {nrm[1]:.5f} {nrm[2]:.5f}\nf " + " ".join(f"{idx[v]}//{k}" for v in s) + "\n")


def order(name):
    cls = pclass[name]
    return (not name.startswith("env_"), list(LABEL).index(cls), name)


meta = []
for name in sorted(parts, key=order):
    tris = parts[name]
    pts = np.concatenate([t[0] for t in tris])
    write_obj(os.path.join(out, name + ".obj"), tris)
    if not name.startswith("env_"):
        write_hull(os.path.join(out, "collision", name + ".obj"), np.unique(np.round(pts, 4), axis=0))
    note = {"railing": "deck parapet" if name < "railing_17" else "pier-cap inspection platform handrail"}.get(pclass[name])
    meta.append({"name": name, "class": pclass[name], "label": LABEL[pclass[name]], **({"note": note} if note else {}),
                 "centroid": np.round(pts.mean(0), 3).tolist(), "bbox_min": np.round(pts.min(0), 3).tolist(),
                 "bbox_max": np.round(pts.max(0), 3).tolist(), "tris": len(tris)})
json.dump(meta, open(os.path.join(out, "parts.json"), "w"), indent=1)

# SDF snippet
uri = "model://" + os.path.basename(os.path.normpath(out))
lab = lambda c: f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{LABEL[c]}</label></plugin>'
s = ['    <model name="scene1_bridge">\n      <static>true</static>']
for p in meta:
    if p["name"].startswith("env_"):
        continue
    s.append(f'      <link name="{p["name"]}">\n'
             f'        <visual name="visual"><geometry><mesh><uri>{uri}/{p["name"]}.obj</uri></mesh></geometry>{lab(p["class"])}</visual>\n'
             f'        <collision name="collision"><geometry><mesh><uri>{uri}/collision/{p["name"]}.obj</uri></mesh></geometry></collision>\n'
             f'      </link>')
s.append("    </model>\n")
for p in meta:
    if not p["name"].startswith("env_"):
        continue
    c = p["class"]
    coll = c in ("terrain", "water", "road")
    s.append(f'    <model name="scene1_{c}">\n      <static>true</static>\n      <link name="link">\n'
             f'        <visual name="visual"><geometry><mesh><uri>{uri}/{p["name"]}.obj</uri></mesh></geometry>{lab(c)}</visual>\n'
             + (f'        <collision name="collision"><geometry><mesh><uri>{uri}/{p["name"]}.obj</uri></mesh></geometry></collision>\n' if coll else "")
             + '      </link>\n    </model>\n')
open(os.path.join(out, "sdf_parts.xml"), "w").write("\n".join(s))

cnt = defaultdict(int)
for p in meta:
    cnt[p["class"]] += 1
print(f"{len(meta)} parts:", dict(cnt))
print("supports x:", np.round(supports, 2).tolist())

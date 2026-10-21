#!/usr/bin/env python3
"""Break Scene-3 into per-component bridge parts (see BRIDGE_PARTS_SPEC.md).

Scene-3's OBJ export has the main viaduct exploded into ONE welded mesh (~108k faces), so
parts are cut out geometrically:
  main bridge (along +Y, y -500..439):
    approaches  y<-218.5 / y>156.5 : main deck x -15.1..11 (top z 0.6, slab bottom 0.2,
                3-cell box girder down to -2.6) on single flared columns; two sloping ramps
                (x<-15.1, x>11), deck slab to 2.1 m and box girder to 4.9 m below barrier top,
                and columns.
    central     -218.5<y<156.5      : full-width deck/girders (z>=-2.65); below it two
                V-strut "arches" springing from trunks at y~-127 / y~59.5 (trunk z<-17.5).
    5-column piers at y~-218.5 and y~156.5.
  bridge 2 : small two-pier bridge to the west (x<-150, deck z~-18).
Separate SketchUp components: railings, ramp columns, green median cards (-> vegetation).
Within each class, instances are numbered along the bridge axis.
FrontColor faces get the scene3_tex look (deck tops Blacktop, rest Concrete, box UVs).

Usage: tools_scene3_parts.py <Scene-3.obj> <scene3_tex dir> <scene3_defects dir> <out_dir> <world.sdf>
Writes out_dir/*.obj (visuals), col/*.stl (per-part collision), parts.mtl,
textures/, parts.json, debug_world.sdf (flat colour per class) and the world file."""
import json, os, shutil, sys
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

src, texdir, defdir, out, world_path = sys.argv[1:6]
NAME = os.path.basename(os.path.normpath(out))           # model:// prefix, e.g. scene3_parts
LABEL = {"deck": 1, "girder": 2, "cross_beam": 3, "pier_cap": 4, "pier_column": 5, "footing": 6,
         "abutment": 7, "railing": 8, "bearing": 9, "arch_truss": 10, "bridge_misc": 11,
         "terrain": 20, "water": 21, "vegetation": 22, "vehicle": 23, "road": 24, "other": 25,
         "defect": 30}
DEBUG_RGB = {"deck": (0.2, 0.2, 0.2), "girder": (0.1, 0.4, 0.9), "cross_beam": (0, 0.8, 0.8),
             "pier_cap": (1, 0.5, 0), "pier_column": (0.9, 0.1, 0.1), "footing": (0.5, 0.3, 0.1),
             "abutment": (0.6, 0, 0.6), "railing": (1, 1, 0), "bearing": (1, 0, 1),
             "arch_truss": (0.1, 0.8, 0.1), "bridge_misc": (1, 1, 1), "vegetation": (0.75, 1, 0.35),
             "water": (0.3, 0.5, 0.8), "defect": (1, 0.4, 0.7)}
TILE, WATER_Z = 4.0, -12.0
XW, XE = -15.1, 11.0                       # main deck / ramp split (approaches)
Y_S, Y_N = -218.5, 156.5                   # central section limits (5-column piers)
V_ST = (-127.0, 59.5)                      # V-strut trunk stations
RAMP_Y = (-498.35, 436.55)                 # ramp deck ends
STATIONS = [-500, -428.5, -358.5, -288.5, Y_S, V_ST[0], V_ST[1], Y_N, 226.5, 296.5, 366.5, 439]

# ---------------------------------------------------------------- parse (Y-up -> Z-up)
V, VT, VN, polys = [], [], [], []
grp = mat = None
with open(src, errors="replace") as f:
    for line in f:
        t = line.split()
        if not t:
            continue
        k = t[0]
        if k == "v":
            x, y, z = map(float, t[1:4]); V.append((x, -z, y))
        elif k == "vt":
            VT.append((float(t[1]), float(t[2])))
        elif k == "vn":
            x, y, z = map(float, t[1:4]); VN.append((x, -z, y))
        elif k == "usemtl":
            mat = t[1]
        elif k == "f":
            ix = [(p.split("/") + ["", ""])[:3] for p in t[1:]]
            polys.append((mat, [int(a) - 1 for a, _, _ in ix], [int(b) - 1 if b else -1 for _, b, _ in ix],
                          [int(c) - 1 if c else -1 for _, _, c in ix]))
V, VT, VN = np.array(V), np.array(VT + [(0, 0)]), np.array(VN + [(0, 0, 1)])

tv, tt, tn, tmat = [], [], [], []
for m, vi, ti, ni in polys:
    for k in range(1, len(vi) - 1):
        tv.append((vi[0], vi[k], vi[k + 1])); tt.append((ti[0], ti[k], ti[k + 1]))
        tn.append((ni[0], ni[k], ni[k + 1])); tmat.append(m)
tv, tt, tn, tmat = np.array(tv), np.array(tt), np.array(tn), np.array(tmat)
# mirrored SketchUp copies (the east ramps) come out with reversed winding: flip any
# triangle whose winding disagrees with the exported vertex normals
_P = V[tv]
_fn = np.cross(_P[:, 1] - _P[:, 0], _P[:, 2] - _P[:, 0])
_vn = np.where((tn >= 0).all(1)[:, None], VN[tn].sum(1), _fn)
flip = (_fn * _vn).sum(1) < 0
tv[flip] = tv[flip][:, ::-1]; tt[flip] = tt[flip][:, ::-1]; tn[flip] = tn[flip][:, ::-1]
print(f"flipped {flip.sum()} mirrored triangles")
P = V[tv]
nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
area = np.linalg.norm(nrm, axis=1) / 2
nrm /= (2 * area[:, None] + 1e-12)
C = P.mean(1)
N = len(tv)
print(f"{N} triangles")

_, weld = np.unique(np.round(V * 1000).astype(np.int64), axis=0, return_inverse=True)
W = weld.ravel()[tv]


def cc(idx):
    """Connected components (shared welded vertex) of the triangle subset idx."""
    idx = np.asarray(idx)
    if len(idx) == 0:
        return np.zeros(0, int)
    nv = W.max() + 1
    rows = np.repeat(np.arange(len(idx)), 3)
    g = coo_matrix((np.ones(len(rows)), (rows, len(idx) + W[idx].ravel())), shape=(len(idx) + nv,) * 2)
    _, lab = connected_components(g, directed=False)
    _, lab = np.unique(lab[:len(idx)], return_inverse=True)
    return lab


comp = cc(np.arange(N))
ncomp = comp.max() + 1
csize = np.bincount(comp)
blob = csize.argmax()
cmin = np.full((ncomp, 3), np.inf); cmax = np.full((ncomp, 3), -np.inf)
np.minimum.at(cmin, comp, P.min(1)); np.maximum.at(cmax, comp, P.max(1))
ccen = (cmin + cmax) / 2

# per-triangle assignment: model, class, instance key
model = np.empty(N, object); cls = np.empty(N, object); key = np.empty(N, object)
span = np.clip(np.searchsorted(STATIONS, C[:, 1]) - 1, 0, len(STATIONS) - 2)


def merge_small(idx, labels, min_tris=40, min_size=1.0):
    """Fold tiny connected pieces into the nearest big piece (by centroid)."""
    labels = labels.copy()
    big = [l for l in np.unique(labels) if (labels == l).sum() >= min_tris and
           np.ptp(P[idx[labels == l]].reshape(-1, 3), 0).max() >= min_size]
    if not big:
        return labels
    cen = {l: C[idx[labels == l]].mean(0) for l in big}
    for l in np.unique(labels):
        if l in cen:
            continue
        c = C[idx[labels == l]].mean(0)
        labels[labels == l] = min(big, key=lambda b: np.linalg.norm(cen[b] - c))
    return labels


# ---- bridge 2 (west) ------------------------------------------------------------------
b2 = ccen[comp, 0] < -150
A0, A1 = np.array([-342.5, -36.5]), np.array([-182.0, -146.0])
axis2 = (A1 - A0) / np.linalg.norm(A1 - A0)
for c in np.unique(comp[b2]):
    m = comp == c
    ext = cmax[c] - cmin[c]
    model[m] = "scene3_bridge2"
    if csize[c] > 1700 and ext[2] > 20:            # deck girder + one pier, welded
        low = m & (C[:, 2] < -22.6)
        cls[low], key[low] = "pier_column", "b2pier_a"
        cls[m & ~low], key[m & ~low] = "girder", "b2girder"
    elif ext[2] > 10:
        cls[m], key[m] = "pier_column", f"b2pier_{c}"
    elif csize[c] > 2000:
        cls[m], key[m] = "deck", "b2deck"
    elif max(ext[0], ext[1]) > 20 or (ext[2] > 0.9 and csize[c] < 30):   # rails + posts
        side = np.sign(np.cross(axis2, ccen[c, :2] - A0))
        cls[m], key[m] = "railing", f"b2rail_{side:+.0f}"
    else:                                             # lamps / plates at the ends
        cls[m], key[m] = "bridge_misc", f"b2misc_{np.dot(ccen[c, :2] - A0, axis2) > 80}"

# ---- main bridge: separate components -------------------------------------------------
for c in range(ncomp):
    m = comp == c
    if c == blob or b2[m][0]:
        continue
    model[m] = "scene3_bridge"
    ext = cmax[c] - cmin[c]
    if (tmat[m] == "_Color_F16_1").mean() > 0.5:
        model[m], cls[m], key[m] = "scene3_vegetation", "vegetation", "veg"
    elif ext[2] > 10:
        cls[m], key[m] = "pier_column", f"col_{c}"
    elif max(ext[0], ext[1]) > 10:
        cls[m], key[m] = "railing", f"rail_{c}"
    else:
        cls[m], key[m] = "bridge_misc", "misc"

# ---- main bridge: the welded blob -----------------------------------------------------
bi = np.where(comp == blob)[0]
model[bi] = "scene3_bridge"
x, y, z = C[bi, 0], C[bi, 1], C[bi, 2]
central = (y > Y_S) & (y < Y_N)
# faces on the x=XW/XE seam below the main girders belong to the ramp girders
in_main = (x >= XW - 0.05) & (x <= XE + 0.05) & ((z >= -2.65) | (np.abs(x + 2) < 9))
unit = np.where(central | in_main, "main", np.where(x < -2, "rampW", "rampE"))
layer = np.empty(len(bi), object)
mm = unit == "main"
layer[mm] = np.where(z[mm] >= 0.15, "deck", np.where(z[mm] >= -2.65, "girder", "sub"))
for u in ("rampW", "rampE"):                          # sloping ramps: depth below local top
    for half in (y < 0, y >= 0):
        r = (unit == u) & half
        if not r.any():
            continue
        up = r & (nrm[bi, 2] > 0.95)
        bins = np.arange(-504, 444, 4.0)
        bc, tops = [], []
        for b0 in bins:
            k = up & (y >= b0) & (y < b0 + 4)
            if k.any():
                bc.append(b0 + 2); tops.append(z[k].max())
        top = np.interp(y[r], bc, tops)
        d = top - z[r]
        # depth below barrier top: road ~1.3, slab soffit ~2.0, box soffit ~4.7
        layer[r] = np.where(d < 2.1, "deck", np.where(d <= 4.9, "girder", "sub"))
        layer[np.where(r)[0][(y[r] < RAMP_Y[0]) | (y[r] > RAMP_Y[1])]] = "sub"   # columns past the deck ends

sp = span[bi]
for i, (u, l) in enumerate(zip(unit, layer)):
    if l == "deck":
        cls[bi[i]], key[bi[i]] = "deck", ("deck", u, sp[i])
# girders: the decks sit on box girders -> one girder per (deck unit, span); in the central
# section the two carriageways have separate boxes either side of the x~-1.5 median gap
gi = np.where(layer == "girder")[0]
for j in gi:
    side = bool(central[j] and x[j] < -1.5)
    cls[bi[j]], key[bi[j]] = "girder", ("girder", unit[j], sp[j], side)
# substructure
si = np.where(layer == "sub")[0]
cen_sub = si[(y[si] > Y_S + 2.5) & (y[si] < Y_N - 2.5)]
app_sub = np.setdiff1d(si, cen_sub)
lab = merge_small(bi[app_sub], cc(bi[app_sub]))
for j, l in zip(app_sub, lab):
    cls[bi[j]], key[bi[j]] = "pier_column", ("col", l)
for j in cen_sub:
    st = int(np.argmin([abs(y[j] - s) for s in V_ST]))
    if z[j] < -17.5 and abs(y[j] - V_ST[st]) < 12:
        cls[bi[j]], key[bi[j]] = "pier_column", ("vtrunk", st)
    else:
        cls[bi[j]], key[bi[j]] = "arch_truss", ("arch", st, x[j] < -1.5, y[j] < V_ST[st])

assert all(c is not None for c in cls), "unassigned triangles"
if os.environ.get("PARTS_DUMP"):
    import pickle
    pickle.dump(dict(P=P, C=C, nrm=nrm, area=area, cls=cls, key=key, model=model, bi=bi, layer=layer, unit=unit),
                open(os.environ["PARTS_DUMP"], "wb"))

# ---------------------------------------------------------------- materials / UVs
os.makedirs(os.path.join(out, "textures"), exist_ok=True)
for f in os.listdir(os.path.join(texdir, "textures")):
    shutil.copy(os.path.join(texdir, "textures", f), os.path.join(out, "textures", f))
shutil.copy(os.path.join(texdir, "scene3.mtl"), os.path.join(out, "parts.mtl"))
newmat = tmat.copy()
fc = tmat == "FrontColor"
newmat[fc] = np.where((cls[fc] == "deck") & (nrm[fc, 2] > 0.9), "Blacktop", "Concrete")
ax = np.argmax(np.abs(nrm), 1)
UVW = [(1, 2), (0, 2), (0, 1)]

# ---------------------------------------------------------------- group into named parts
inst = defaultdict(list)
for i in range(N):
    inst[(model[i], cls[i], key[i])].append(i)


def along(mdl, c):
    return np.dot(c[:2] - A0, axis2) if mdl == "scene3_bridge2" else c[1]


parts = []
bycls = defaultdict(list)
for (mdl, c, k), ids in inst.items():
    ids = np.array(ids)
    pc = C[ids].mean(0)
    bycls[(mdl, c)].append((along(mdl, pc), pc[0], ids))
for (mdl, c), lst in sorted(bycls.items()):
    lst.sort(key=lambda t: (round(t[0], 0), t[1]))
    for n, (_, _, ids) in enumerate(lst, 1):
        name = f"{c}_{n:02d}" if c != "vegetation" else "env_vegetation"
        parts.append((mdl, c, name, ids))


def write_obj(path, ids):
    """One OBJ per part; corners deduplicated on (position, uv, normal)."""
    ids = np.asarray(ids)
    pos = P[ids].reshape(-1, 3)
    uv = VT[tt[ids]].reshape(-1, 2)
    f = fc[ids]
    if f.any():                                      # box-projected UVs for retextured faces
        a = np.array([u for u, _ in UVW])[ax[ids[f]]]; b = np.array([w for _, w in UVW])[ax[ids[f]]]
        pf = P[ids[f]]
        uv.reshape(-1, 3, 2)[f] = np.stack([np.take_along_axis(pf, a[:, None, None].repeat(3, 1), 2)[..., 0],
                                            np.take_along_axis(pf, b[:, None, None].repeat(3, 1), 2)[..., 0]], -1) / TILE
    nn = np.where((tn[ids] >= 0)[..., None], VN[tn[ids]], nrm[ids][:, None, :]).reshape(-1, 3)
    rows = np.hstack([np.round(pos, 4), np.round(uv, 5), np.round(nn, 3)])
    uniq, inv = np.unique(rows, axis=0, return_inverse=True)
    inv = inv.ravel().reshape(-1, 3) + 1
    with open(path, "w") as o:
        o.write(f"# {os.path.basename(path)} - generated by tools_scene3_parts.py (Z-up, m)\nmtllib parts.mtl\n")
        o.writelines(f"v {r[0]:.4f} {r[1]:.4f} {r[2]:.4f}\n" for r in uniq)
        o.writelines(f"vt {r[3]:.5f} {r[4]:.5f}\n" for r in uniq)
        o.writelines(f"vn {r[5]:.3f} {r[6]:.3f} {r[7]:.3f}\n" for r in uniq)
        mats = newmat[ids]
        for mname in np.unique(mats):
            o.write(f"usemtl {mname}\n")
            o.writelines(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" for a, b, c in inv[mats == mname])


COL_TILE = None          # m tile edge for collision meshes; None = one mesh per part (A/B: 204 shapes
#                          RTF 0.42, 25 m tiles = 835 shapes RTF 0.27) ...
BIG_TRIS, BIG_TILE = 8000, 50.0   # ... except the big central decks, which the iris sits on
os.makedirs(os.path.join(out, "col"), exist_ok=True)
for _f in os.listdir(os.path.join(out, "col")):     # drop stale tiles from a previous run
    if _f.endswith(".stl"):
        os.remove(os.path.join(out, "col", _f))


def write_col(fname, ids):
    """Binary STL(s) of the part (optionally cut into COL_TILE (x, y) tiles); returns names."""
    ids = np.asarray(ids)
    tile = COL_TILE if len(ids) <= BIG_TRIS else BIG_TILE
    t = np.zeros((len(ids), 2), int) if tile is None else np.floor(C[ids, :2] / tile).astype(int)
    names = []
    for k in np.unique(t, axis=0):
        sel = ids[(t == k).all(1)]
        nm = f"{fname}.stl" if tile is None else f"{fname}_{k[0]}_{k[1]}.stl"
        rec = np.zeros(len(sel), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        rec["n"], rec["v"] = nrm[sel], P[sel]
        with open(os.path.join(out, "col", nm), "wb") as o:
            o.write(b"tools_scene3_parts collision tile".ljust(80, b" ") + np.uint32(len(sel)).tobytes() + rec.tobytes())
        names.append(nm)
    return names


meta = []
for mdl, c, name, ids in parts:
    write_obj(os.path.join(out, name + ".obj") if mdl != "scene3_bridge2" else os.path.join(out, "b2_" + name + ".obj"), ids)
    pts = P[ids].reshape(-1, 3)
    meta.append({"model": mdl, "name": name, "class": c, "label": LABEL[c],
                 "file": ("b2_" if mdl == "scene3_bridge2" else "") + name + ".obj",
                 "centroid": np.round(C[ids].mean(0), 3).tolist(),
                 "bbox_min": np.round(pts.min(0), 3).tolist(), "bbox_max": np.round(pts.max(0), 3).tolist(),
                 "tris": int(len(ids)),
                 "collision": [] if c == "vegetation" else
                 write_col(("b2_" if mdl == "scene3_bridge2" else "") + name, ids)})

# water quad (same extent as scene3_tex)
x0, x1, y0, y1 = V[:, 0].min() - 300, V[:, 0].max() + 300, V[:, 1].min() - 150, V[:, 1].max() + 150
with open(os.path.join(out, "env_water.obj"), "w") as o:
    o.write("mtllib parts.mtl\n")
    for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        o.write(f"v {xx:.3f} {yy:.3f} {WATER_Z}\nvt {xx/8:.4f} {yy/8:.4f}\n")
    o.write("vn 0 0 1\nusemtl Water\nf 1/1/1 2/2/1 3/3/1 4/4/1\n")
meta.append({"model": "scene3_water", "name": "env_water", "class": "water", "label": 21, "file": "env_water.obj",
             "centroid": [(x0 + x1) / 2, (y0 + y1) / 2, WATER_Z], "bbox_min": [x0, y0, WATER_Z],
             "bbox_max": [x1, y1, WATER_Z], "tris": 2})
gt = os.path.join(defdir, "defects_gt.json")
meta.append({"model": "scene3_defects", "name": "defects", "class": "defect", "label": 30,
             "file": "../scene3_defects/defects.obj", "ground_truth": os.path.relpath(gt, out)})
json.dump({"scene": "scene3", "labels": LABEL, "parts": meta}, open(os.path.join(out, "parts.json"), "w"), indent=1)

# ---------------------------------------------------------------- SDF
_w = open(world_path).read()
HEADER = _w.split("    <!-- PARTS BEGIN")[0] if "<!-- PARTS BEGIN" in _w else _w.split("    <!-- Scene-2 (SketchUp STL")[0]


def label_plugin(l):
    return f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{l}</label></plugin>'


def link(name, uri, l, collide=None, debug=False, shadows=True):
    mat = ""
    if debug:
        r, g, b = DEBUG_RGB[[k for k, v in LABEL.items() if v == l][0]]
        mat = f"<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>"
    s = (f'      <link name="{name}">\n'
         f'        <visual name="visual">{"" if shadows else "<cast_shadows>false</cast_shadows>"}'
         f'<geometry><mesh><uri>{uri}</uri></mesh></geometry>{mat}{label_plugin(l)}</visual>\n')
    for k, c in enumerate(collide or []):
        s += (f'        <collision name="c{k}"><geometry><mesh><uri>model://{NAME}/col/{c}</uri></mesh>'
              f'</geometry></collision>\n')
    return s + "      </link>\n"


def models(debug=False):
    s = ("    <!-- PARTS BEGIN: generated by tools_scene3_parts.py. One link per bridge component,\n"
         "         labelled per BRIDGE_PARTS_SPEC.md (gz-sim-label-system); parts.json has bboxes. -->\n")
    for mdl in ("scene3_bridge", "scene3_bridge2", "scene3_vegetation"):
        s += f'    <model name="{mdl}">\n      <static>true</static>\n'
        for p in meta:
            if p["model"] == mdl:
                s += link(p["name"], f"model://{NAME}/{p['file']}", p["label"],
                          collide=p["collision"], debug=debug)
        s += "    </model>\n\n"
    s += (f'    <model name="scene3_water">\n      <static>true</static>\n'
          + link("env_water", f"model://{NAME}/env_water.obj", 21, debug=debug, shadows=False).replace(
              "      </link>\n",
              f'        <collision name="collision"><pose>{(x0+x1)/2:.1f} {(y0+y1)/2:.1f} {WATER_Z-0.05} 0 0 0</pose>'
              f'<geometry><box><size>{x1-x0:.0f} {y1-y0:.0f} 0.1</size></box></geometry></collision>\n      </link>\n')
          + "    </model>\n\n")
    s += ('    <!-- CODEBRIM defect decals (tools_scene3_defects.py); ground truth in scene3_defects/defects_gt.json -->\n'
          '    <model name="scene3_defects">\n      <static>true</static>\n'
          + link("defects", "model://scene3_defects/defects.obj", 30, debug=debug, shadows=False)
          + "    </model>\n    <!-- PARTS END -->\n\n")
    return s


TAIL = ('    <!-- ArduPilot-controlled quad (SITL connects on UDP 9002). Spawn on the z=0.6 m deck at origin. -->\n'
        '    <include>\n      <uri>model://iris_with_gimbal</uri>\n      <name>iris</name>\n'
        '      <pose degrees="true">0 0 0.8 0 0 90</pose>\n    </include>\n  </world>\n</sdf>\n')
open(world_path, "w").write(HEADER + models() + TAIL)
open(os.path.join(out, "debug_world.sdf"), "w").write(HEADER + models(debug=True) + TAIL)

cnt = defaultdict(int)
for p in meta:
    cnt[(p["model"], p["class"])] += 1
for k, v in sorted(cnt.items()):
    print(f"{k[0]:18s} {k[1]:12s} {v}")

#!/usr/bin/env python3
"""Scene-3's only OBJ export (Aug 4) is mostly SketchUp default material and has no terrain.
This bakes it Z-up and paints the untextured faces with textures pulled from Scene-3.skp:
up-facing deck faces -> Blacktop, everything else -> bare concrete (box-projected UVs),
and adds a water surface under the bridge.
Usage: tools_scene3_retexture.py <Scene-3.obj> <Scene-3.skp> <out_dir>"""
import os, sys, zipfile
from collections import defaultdict

src, skp, out = sys.argv[1:4]
os.makedirs(os.path.join(out, "textures"), exist_ok=True)
srcdir = os.path.dirname(os.path.abspath(src))

TILE = 4.0          # metres per texture repeat for generated UVs
WATER_Z, WATER_TILE = -12.0, 8.0
SKP_TEX = {          # new material -> (texture inside .skp, fallback Kd)
    "Blacktop":  ("materials/[Blacktop New]/Blacktop New.jpg", "1 1 1"),
    "Concrete":  ("materials/[Color_A08]/ConcreteBare0280_39_S.jpg", "1 1 1"),
    "Water":     ("materials/[Water Pool]/Water_Pool.jpg", "1 1 1"),
}
EXPORT_TEX = {       # textures the export ships but its MTL forgot to reference
    "Polished_Concrete_Old": "Polished_Concrete_Old.jpg",
    "Metal_Aluminum_Anodized": "Metal_Aluminum_Anodized.jpg",
    "Asphalt_New": "Asphalt_New.jpg",
}

z = zipfile.ZipFile(skp)
for m, (path, _) in SKP_TEX.items():
    open(os.path.join(out, "textures", m + ".jpg"), "wb").write(z.read(path))

V, VT, VN, faces, mat = [], [], [], defaultdict(list), None
with open(src, errors="replace") as f:
    for line in f:
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            x, y, zz = map(float, t[1:4]); V.append((x, -zz, y))      # Y-up -> Z-up
        elif t[0] == "vt":
            VT.append((float(t[1]), float(t[2])))
        elif t[0] == "vn":
            x, y, zz = map(float, t[1:4]); VN.append((x, -zz, y))
        elif t[0] == "usemtl":
            mat = t[1]
        elif t[0] == "f":
            faces[mat].append([tuple(int(i) - 1 if i else -1 for i in (p.split("/") + ["", ""])[:3]) for p in t[1:]])

out_faces = defaultdict(list)   # mat -> list of [(v, vt, vn)]
for m, fl in faces.items():
    if m not in ("FrontColor",):
        out_faces[m].extend(fl)
        continue
    for f in fl:
        nx, ny, nz = VN[f[0][2]] if f[0][2] >= 0 else (0, 0, 1)
        zc = sum(V[v][2] for v, _, _ in f) / len(f)
        new = "Blacktop" if nz > 0.9 and zc > -1.0 else "Concrete"
        ax = max(range(3), key=lambda i: abs((nx, ny, nz)[i]))      # box projection
        uvs = []
        for v, _, vn in f:
            p = V[v]
            u, w = [(p[1], p[2]), (p[0], p[2]), (p[0], p[1])][ax]
            VT.append((u / TILE, w / TILE)); uvs.append((v, len(VT) - 1, vn))
        out_faces[new].append(uvs)

# water quad (two-sided not needed: viewed from above)
xs, ys = [p[0] for p in V], [p[1] for p in V]
x0, x1, y0, y1 = min(xs) - 300, max(xs) + 300, min(ys) - 150, max(ys) + 150
base = len(V)
for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
    V.append((x, y, WATER_Z)); VT.append((x / WATER_TILE, y / WATER_TILE))
VN.append((0, 0, 1))
out_faces["Water"].append([(base + i, len(VT) - 4 + i, len(VN) - 1) for i in range(4)])

with open(os.path.join(out, "scene3.obj"), "w") as o:
    o.write("# Scene-3 retextured, Z-up metres\nmtllib scene3.mtl\n")
    o.writelines(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in V)
    o.writelines(f"vt {u:.5f} {w:.5f}\n" for u, w in VT)
    o.writelines(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in VN)
    for m, fl in out_faces.items():
        o.write(f"g {m}\nusemtl {m}\n")
        for f in fl:
            o.write("f " + " ".join(f"{v+1}/{t+1 if t >= 0 else ''}/{n+1 if n >= 0 else ''}".replace("//", "//") for v, t, n in f) + "\n")

# MTL: original colours + texture maps
with open(os.path.join(srcdir, "Scene-3.mtl"), errors="replace") as f, open(os.path.join(out, "scene3.mtl"), "w") as o:
    cur = None
    for line in f:
        line = line.rstrip("\r\n"); o.write(line + "\n")
        if line.startswith("newmtl"):
            cur = line.split()[1]
        if line.startswith("Ks") and cur in EXPORT_TEX:
            img = EXPORT_TEX[cur]
            os.system(f'cp "{os.path.join(srcdir, "Scene-3", img)}" "{os.path.join(out, "textures", img)}"')
            o.write(f"map_Kd textures/{img}\n")
    for m, (_, kd) in SKP_TEX.items():
        o.write(f"\nnewmtl {m}\nKa 0 0 0\nKd {kd}\nKs 0.1 0.1 0.1\nmap_Kd textures/{m}.jpg\n")

for m, fl in out_faces.items():
    print(f"{m:26s} {len(fl)}")
print(f"water {x0:.0f}..{x1:.0f} x {y0:.0f}..{y1:.0f} at z={WATER_Z}")

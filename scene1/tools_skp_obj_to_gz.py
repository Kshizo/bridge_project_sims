#!/usr/bin/env python3
"""Convert a SketchUp OBJ export (Y-up, mm, thousands of groups) into a Gazebo-friendly
textured OBJ: Z-up, metres, one group per material, textures copied next to the MTL.
Usage: skp_obj_to_gz.py <in.obj> <out_dir> <name>
Prints the bounding box (m) of the result."""
import os, re, shutil, sys
from collections import defaultdict

src, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
srcdir = os.path.dirname(os.path.abspath(src))
os.makedirs(os.path.join(out, "textures"), exist_ok=True)

S = 0.001  # mm -> m
V, VT, VN, faces = [], [], [], defaultdict(list)
mtllib, mat = None, "FrontColor"
with open(src, "r", errors="replace") as f:
    for line in f:
        line = line.rstrip("\r\n")
        if line.startswith("v "):
            x, y, z = map(float, line.split()[1:4])
            V.append((x * S, -z * S, y * S))           # roll +90deg: Y-up -> Z-up
        elif line.startswith("vt "):
            VT.append(line)
        elif line.startswith("vn "):
            x, y, z = map(float, line.split()[1:4])
            VN.append((x, -z, y))
        elif line.startswith("f "):
            faces[mat].append(line)
        elif line.startswith("usemtl "):
            mat = line.split(None, 1)[1].strip()
        elif line.startswith("mtllib "):
            mtllib = line.split(None, 1)[1].strip()

zmin = min(v[2] for v in V)
xs, ys = [v[0] for v in V], [v[1] for v in V]
with open(os.path.join(out, name + ".obj"), "w") as o:
    o.write(f"# Converted from SketchUp export, Z-up metres, base lifted {-zmin:.3f} m\nmtllib {name}.mtl\n")
    o.writelines(f"v {x:.5f} {y:.5f} {z - zmin:.5f}\n" for x, y, z in V)
    o.writelines(l + "\n" for l in VT)
    o.writelines(f"vn {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in VN)
    for m, fl in faces.items():
        o.write(f"g {re.sub(r'[^A-Za-z0-9_]', '_', m)}\nusemtl {m}\n")
        o.writelines(l + "\n" for l in fl)

# MTL: LF endings, textures flattened into textures/
with open(os.path.join(srcdir, mtllib), errors="replace") as f, \
     open(os.path.join(out, name + ".mtl"), "w") as o:
    for line in f:
        line = line.rstrip("\r\n")
        if line.startswith("map_Kd "):
            tex = line.split(None, 1)[1].strip()
            base = os.path.basename(tex.replace("\\", "/"))
            shutil.copy2(os.path.join(srcdir, tex), os.path.join(out, "textures", base))
            line = f"map_Kd textures/{base}"
        o.write(line + "\n")

print(f"verts={len(V)} mats={len(faces)} faces={sum(map(len, faces.values()))}")
print(f"bbox x {min(xs):.2f}..{max(xs):.2f}  y {min(ys):.2f}..{max(ys):.2f}  z 0..{max(v[2] for v in V) - zmin:.2f}  lift={-zmin:.3f}")

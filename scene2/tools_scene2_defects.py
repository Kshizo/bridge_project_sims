#!/usr/bin/env python3
"""Scene-2 crack / spalling decals for testing a defect detector (engine: sim_env/defect_decals.py).

Usage: tools_scene2_defects.py <codebrim tiles/images/val> <codebrim tiles/labels/val>
Writes scene2_defects/{defects.obj, defects.mtl, textures/, defects_gt.json}; the world includes it as
model://scene2_defects/defects.obj (label 30) and scene2_parts/parts.json lists it as the "defect" part."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim_env"))
from defect_decals import build, X, Y, Z

PY, NY = (0, 1, 0), (0, -1, 0)
# Scene-2: deck along +Y (x -9.2..-0.4), piers at y 20.475/38.475/56.475/74.475 with round
# columns (R 0.5 m) at x -7.9/-4.8/-1.7, caps z 7.15..8.15, girders z 8.25..10.25, soffit z 10.25.
# (id, class, image, centre, outward normal, up, size[, column radius]) - see sim_env/defect_decals.py
DECALS = [
    # pier columns are round (R 0.5 m): centre = axis point, decal wrapped round it
    ("C1", "crack",    "image_0000646_896_0.jpg",     (-7.902, 20.475, 3.0),  -X, Z, 0.9, 0.5),
    ("C2", "spalling", "image_0001386_2688_0.jpg",    (-4.802, 20.475, 2.0),  PY, Z, 0.9, 0.5),
    ("C3", "crack",    "image_0000592_896_0.jpg",     (-1.702, 38.475, 4.0),  X,  Z, 0.9, 0.5),
    ("C4", "spalling", "image_0001355_896_896.jpg",   (-7.902, 38.475, 1.8),  NY, Z, 0.9, 0.5),
    ("C5", "crack",    "image_0000652_896_896.jpg",   (-4.802, 56.475, 5.0),  PY, Z, 0.9, 0.5),
    ("C6", "spalling", "image_0001192_1792_1792.jpg", (-1.702, 56.475, 2.5),  X,  Z, 0.9, 0.5),
    ("C7", "crack",    "image_0000646_896_896.jpg",   (-7.902, 74.475, 4.5),  -X, Z, 0.9, 0.5),
    ("C8", "spalling", "image_0001191_3584_896.jpg",  (-1.702, 74.475, 3.0),  NY, Z, 0.9, 0.5),
    # pier caps (1 m tall faces)
    ("P1", "crack",    "image_0000652_896_1792.jpg",  (-6.4, 19.775, 7.65),   NY, Z, 0.9),
    ("P2", "spalling", "image_0000755_896_896.jpg",   (-3.0, 39.175, 7.65),   PY, Z, 0.9),
    ("P3", "crack",    "image_0000652_0_1792.jpg",    (-7.0, 57.175, 7.65),   PY, Z, 0.9),
    ("P4", "spalling", "image_0001365_2688_1792.jpg", (-2.5, 73.775, 7.65),   NY, Z, 0.9),
    # outer girder webs (2 m tall)
    ("G1", "crack",    "image_0000592_0_680.jpg",     (-8.502, 12.0, 9.25),   -X, Z, 1.8),
    ("G2", "spalling", "image_0000758_1792_896.jpg",  (-8.502, 47.0, 9.25),   -X, Z, 1.8),
    ("G3", "crack",    "image_0000633_176_0.jpg",     (-1.102, 30.0, 9.25),   X,  Z, 1.8),
    ("G4", "spalling", "image_0001192_896_896.jpg",   (-1.102, 65.0, 9.25),   X,  Z, 1.8),
    ("G5", "crack",    "image_0000579_0_1088.jpg",    (-1.102, 84.0, 9.25),   X,  Z, 1.8),
    # deck soffit between girders, seen from below
    ("U1", "spalling", "image_0001191_2688_896.jpg",  (-7.07, 29.0, 10.25),   -Z, Y, 1.5),
    ("U2", "crack",    "image_0000652_0_2624.jpg",    (-4.80, 47.5, 10.25),   -Z, Y, 1.5),
    ("U3", "spalling", "image_0001489_1792_896.jpg",  (-2.54, 65.0, 10.25),   -Z, Y, 1.5),
    # abutment front walls facing the river
    ("A1", "crack",    "image_0000652_1712_0.jpg",    (-4.8, 3.15, 5.0),      PY, Z, 2.0),
    ("A2", "spalling", "image_0000683_1792_0.jpg",    (-4.8, 91.8, 5.0),      NY, Z, 2.0),
]

if __name__ == "__main__":
    ok = build("scene2", os.path.join(HERE, "scene2_parts"), DECALS, sys.argv[1], sys.argv[2],
               os.path.join(HERE, "scene2_defects"))
    sys.exit(0 if ok else 1)

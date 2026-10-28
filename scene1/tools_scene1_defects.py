#!/usr/bin/env python3
"""Scene-1 crack / spalling decals for testing a defect detector (engine: sim_env/defect_decals.py).

Usage: tools_scene1_defects.py <codebrim tiles/images/val> <codebrim tiles/labels/val>
Writes scene1_defects/{defects.obj, defects.mtl, textures/, defects_gt.json}; the world includes it as
model://scene1_defects/defects.obj (label 30) and scene1_parts/parts.json lists it as the "defect" part."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim_env"))
from defect_decals import build, X, Y, Z

PY, NY = (0, 1, 0), (0, -1, 0)
# Scene-1: deck along X (y -11.6..-6.1), water z 6.58. Wall piers (x centres below) stand 6.64..~9.1 with a
# recessed 1.65 m panel on each +-X side (y centred on the pier) and a 1.2 m flat strip on each +-Y end.
# Outer girder webs (y -10.89 / -6.99) are split by a rib at z 10.5-10.65: the decals use the 0.55 m
# lower band. Abutment fronts are battered except a flat band z ~7.2-8.2. Caps carry platforms: skipped.
PIER = [(-196.111, -9.053), (-160.23, -9.001), (-124.58, -9.053), (-88.699, -9.001), (-53.049, -9.053),
        (-17.168, -9.001), (18.482, -9.053)]


def side(i, sgn):                     # point in front of pier i's +-X panel
    x, y = PIER[i]
    return (x + sgn * 1.1, y, 7.85)


def end(i, sgn):                      # point in front of pier i's +-Y end strip
    x, y = PIER[i]
    return (x + sgn * 0.05, y + sgn * 2.2, 7.8)       # the strips sit 0.05 m off-centre, mirrored


# (id, class, image, centre, outward normal, up, size) - see sim_env/defect_decals.py
DECALS = [
    # pier side panels
    ("S1", "crack",    "image_0001543_896_2432.jpg",  side(0, +1), X,  Z, 1.4),
    ("S2", "spalling", "image_0001355_896_1792.jpg",  side(1, -1), -X, Z, 1.4),
    ("S3", "spalling", "image_0001192_1792_2432.jpg", side(2, +1), X,  Z, 1.4),
    ("S4", "crack",    "image_0001571_0_896.jpg",     side(3, -1), -X, Z, 1.4),
    ("S5", "crack",    "image_0001543_896_896.jpg",   side(4, +1), X,  Z, 1.4),
    ("S6", "spalling", "image_0001191_3584_1792.jpg", side(5, -1), -X, Z, 1.4),
    ("S7", "crack",    "image_0001501_3584_896.jpg",  side(6, -1), -X, Z, 1.4),
    # pier ends, facing up/down river
    ("E1", "spalling", "image_0000716_0_1712.jpg",    end(0, -1),  NY, Z, 1.0),
    ("E2", "crack",    "image_0001352_2688_896.jpg",  end(2, +1),  PY, Z, 1.0),
    ("E3", "spalling", "image_0000755_0_896.jpg",     end(4, -1),  NY, Z, 1.0),
    ("E4", "spalling", "image_0000758_896_896.jpg",   end(6, +1),  PY, Z, 1.0),
    # outer girder webs, lower band
    ("G1", "crack",    "image_0001396_896_896.jpg",   (-214.0, -10.89, 10.18), NY, Z, 0.5),
    ("G2", "spalling", "image_0001365_2688_896.jpg",  (-178.0, -6.994, 10.18), PY, Z, 0.5),
    ("G3", "crack",    "image_0001543_0_2432.jpg",    (-106.0, -10.89, 10.18), NY, Z, 0.5),
    ("G4", "crack",    "image_0001396_1792_896.jpg",  (-70.0, -6.994, 10.18),  PY, Z, 0.5),
    ("G5", "spalling", "image_0001086_1792_1792.jpg", (-35.0, -10.89, 10.18),  NY, Z, 0.5),
    ("G6", "crack",    "image_0001571_2688_896.jpg",  (1.0, -6.994, 10.18),    PY, Z, 0.5),
    ("G7", "spalling", "image_0001365_1792_896.jpg",  (36.0, -10.89, 10.18),   NY, Z, 0.5),
    # abutment fronts, facing the spans
    ("A1", "crack",    "image_0001352_896_896.jpg",   (-230.415, -11.0, 7.68), X,  Z, 0.9),
    ("A2", "spalling", "image_0001191_2688_1792.jpg", (-230.415, -7.0, 7.68),  X,  Z, 0.9),
    ("A3", "spalling", "image_0001192_896_0.jpg",     (52.578, -11.0, 7.78),   -X, Z, 0.9),
    ("A4", "crack",    "image_0001501_2688_0.jpg",    (52.578, -7.0, 7.78),    -X, Z, 0.9),
]

if __name__ == "__main__":
    ok = build("scene1", os.path.join(HERE, "scene1_parts"), DECALS, sys.argv[1], sys.argv[2],
               os.path.join(HERE, "scene1_defects"))
    sys.exit(0 if ok else 1)

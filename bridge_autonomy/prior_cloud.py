#!/usr/bin/env python3
"""Write the prior cloud every later flight plans from, from a lidar map (no ground truth):
  <prior_cloud>.ply               the BRIDGE only: structure above the floor inside the measured bridge
                                  footprint (+ cloud_analysis.bridge_margin), at flight0.pc_structure_voxel
  <prior>_surroundings.ply        everything else (water, ground, banks, trees...), coarse
                                  (flight0.pc_surroundings_voxel), only so the planner knows the floor and
                                  nearby obstacles
  <prior_cloud>.json              how it was made + the bridge as measured from it

Usage: prior_cloud.py <config.yaml> <map_points.npy | cloud.ply> [--hint x y z]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_model import load_config, read_ply, write_ply, voxel_down, split_bridge, surroundings_path, Bridge


def write_prior(cfg, M, deck_hint=None, meta=None):
    sc, f0 = cfg["scene"], cfg["flight0"]
    M = voxel_down(np.asarray(M, np.float64), f0["pc_structure_voxel"])
    bridge, sur, br = split_bridge(cfg, M, deck_hint)
    sur = voxel_down(sur, f0["pc_surroundings_voxel"])
    os.makedirs(os.path.dirname(sc["prior_cloud"]), exist_ok=True)
    write_ply(sc["prior_cloud"], bridge, comment="bridge only: lidar map, autopilot EKF pose, world ENU metres")
    write_ply(surroundings_path(sc["prior_cloud"]), sur, comment="surroundings (floor, banks, obstacles), coarse")
    info = dict(meta or {})
    info.update({"bridge_points": int(len(bridge)), "surroundings_points": int(len(sur)),
                 "frame": "world ENU, metres (ArduPilot EKF origin)",
                 "deck_hint": None if deck_hint is None else [round(float(x), 3) for x in deck_hint]})
    json.dump(info, open(os.path.splitext(sc["prior_cloud"])[0] + ".json", "w"), indent=1)   # Bridge() reads deck_hint
    try:
        info["bridge_as_measured"] = Bridge(cfg).summary()          # exactly what the planner will see
    except RuntimeError as e:
        info["bridge_as_measured"] = f"failed: {e}"
    json.dump(info, open(os.path.splitext(sc["prior_cloud"])[0] + ".json", "w"), indent=1)
    return bridge, sur, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("map")
    ap.add_argument("--hint", type=float, nargs=3, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    M = read_ply(a.map) if a.map.endswith(".ply") else np.load(a.map)
    t = time.time()
    b, s, info = write_prior(cfg, M, np.array(a.hint) if a.hint else None,
                             {"made_by": f"prior_cloud.py from {os.path.basename(a.map)}", "date": time.strftime("%Y-%m-%d %H:%M")})
    print(f"bridge {len(b)} pts -> {cfg['scene']['prior_cloud']}, surroundings {len(s)} pts ({time.time()-t:.0f} s)")
    print(json.dumps(info["bridge_as_measured"])[:600])

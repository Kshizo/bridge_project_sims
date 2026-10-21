# Scene 1 - 289 m girder bridge

8 spans over a river: 5.5 m wide deck at 11.2-11.5 m, girders down to 9.6 m, 7 wall piers
(2.2 x 4.7 m) from the water (6.6 m) to the pier caps, abutments at both ends.

There are only ~3 m between the water and the girders, so the drone cannot fly under this deck with
`safety.min_clearance` 1.6 m: the planner skips the under-deck loops ("no room under the deck") and the
narrow deck is mapped from the side passes (one just above the water looks under it) and from above.
The drone takes off from a pad on the deck at mid-span (x -70.8).

```bash
./run.sh flight0 --no-gui      # ~1.8 km, ~50 min sim time -> pointcloud/pc_b1.ply
./run.sh survey --no-gui       # planned from pc_b1.ply
./run.sh stop
```

Use `--no-gui`: the railing meshes are very heavy and the Gazebo window slows the simulation down by
~30x. RViz2 still shows the live map, the drone and the camera.

Files: `scene1_ardupilot.sdf` (world), `scene1_parts/` (meshes + parts.json, ground truth),
`config.yaml`, `tools_scene1_parts.py` / `tools_skp_obj_to_gz.py` (how the parts were made from the
SketchUp export).

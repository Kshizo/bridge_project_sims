# Scene 2 - 107 m girder bridge

5 spans over a river: 8.8 m wide deck at 10.6 m, soffit 8.25 m, 4 piers of 3 columns each
(caps at 7.15 m), footings at the water line, abutments at both ends. The water is 0 m.

flight0 maps it completely: side passes at three heights, a pass over the deck, and in each of the
5 spans a loop 2 m in front of the piers at the top, middle and bottom of the pillars. Under the deck
GPS is denied and the drone flies on VIO; each loop's outer sides are just outside the GPS shadow, so
it gets GPS back twice per loop. The drone takes off from a pad on the deck at mid-bridge.

```bash
./run.sh flight0               # ~1.6 km, ~23-29 min sim time -> pointcloud/pc_b2.ply
./run.sh survey                # planned from pc_b2.ply
./run.sh stop
```

Files: `scene2_ardupilot.sdf` (world), `scene2_parts/` (meshes + parts.json, ground truth),
`config.yaml`, `source/` (the SketchUp model and its structure mesh), `tools_scene2_rebuild.py`
(rebuilds `scene2_parts/` and the world's bridge block from `source/`).

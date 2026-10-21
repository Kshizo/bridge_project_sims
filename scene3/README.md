# Scene 3 - 938 m arch bridge (a 110 m window is mapped)

A long bridge with a 46 m wide deck at ~0.6-2 m, two large arch spans over water at -12 m, lines of
3-5 columns, and ramps at both ends that slope down into the water. `parts.json` also contains a second,
fully submerged bridge (`scene3_bridge2`) and a defect part; `scene.bridge_model: scene3_bridge` keeps
the main bridge only.

Mapping the whole bridge takes hours, so `config.yaml` sets a window:

```yaml
scene:
  region: {min: [-40.0, -225.0], max: [40.0, -115.0]}   # x/y box: pier line + arch span + arch support
```

Change it to map another part (or a bigger part). The drone takes off from a pad on the deck inside the
window (4, -170). In the window: side passes at three heights, three passes over the wide deck (one per
`survey.top_lane_spacing` of width), and the under-deck loops in the span.

```bash
./run.sh flight0               # ~1.6 km, ~20 min sim time -> pointcloud/pc_b3.ply
./run.sh survey                # planned from pc_b3.ply
./run.sh stop
```

Files: `scene3_ardupilot.sdf` (world), `scene3_parts/` + `scene3_defects/` (meshes + parts.json, ground
truth), `config.yaml`, `tools_scene3_*.py` (how the parts, defects and textures were made).

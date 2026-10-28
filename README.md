# Bridge project sims

Gazebo Harmonic + ArduPilot SITL simulations of a drone that maps bridges with a 3D lidar and an
RGB-D camera, flies under the deck with GPS denied on its own visual-inertial odometry (VIO), and
turns what it measured into a **bridge-only point cloud** that later flights plan from.

Three scenes, one folder each:

| Folder | Bridge | Notes |
|---|---|---|
| [`scene1/`](scene1/README.md) | 289 m, 8-span girder bridge over a river, wall piers | only 3 m between water and girders, so it is mapped from the sides and above |
| [`scene2/`](scene2/README.md) | 107 m, 5-span girder bridge, 4 piers x 3 columns | full under-deck mapping with GPS denied |
| [`scene3/`](scene3/README.md) | 938 m arch bridge, 46 m wide deck | a ~110 m window (pier line + arch span) is mapped, see `scene.region` |

## Layout

| Path | What |
|---|---|
| `sceneN/sceneN_ardupilot.sdf` | the world: bridge (one labelled link per component), river, banks, the drone on a deck pad |
| `sceneN/sceneN_parts/` | one mesh per bridge component + `parts.json` (ground truth, used only by the simulated world and the evaluation tools) |
| `sceneN/sceneN_defects/` | CODEBRIM crack / spalling photos pasted on the bridge (label 30) + `defects_gt.json` (every defect polygon in world coordinates), for testing a defect detector on the camera footage; built by `sceneN/tools_sceneN_defects.py` |
| `sceneN/config.yaml` | **every** tunable for that scene (route, speeds, clearances, wind, noise, mapping, cloud analysis) |
| `sceneN/run.sh` | `./run.sh flight0`, `./run.sh survey`, `./run.sh stop` |
| `sceneN/output/` | created by a run: map, trajectory, plans, camera video, logs (not in git) |
| `sceneN/pointcloud/pc_bN.ply` | created by flight0: the bridge point cloud (not in git, regenerate with flight0) |
| `bridge_autonomy/` | the drone: planner, VIO, flight control, flight0, survey mission, launcher |
| `mapping/` | lidar mapper (RViz2 output), camera recorder, cloud tools |
| `sim_env/` | the simulated world: wind, GPS shadow under the deck, referee (ground truth, evaluation only) |
| `evaluation/` | scoring tools (coverage, sharpness, plan safety) - they use the ground truth meshes |
| `models/` | `iris_mapper` (Iris + gimbal + 3D lidar + RGB-D camera) and a wind-enabled Iris body |

## Requirements

* Ubuntu 24.04, **Gazebo Harmonic**, **ROS 2 Jazzy** (RViz2)
* **ArduPilot** with SITL built (`~/ardupilot`, `sim_vehicle.py` on PATH) and
  **[ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo)** built in `~/ardupilot_gazebo`
  (other locations: `sitl:` section of each `config.yaml`)
* Python: `pip install numpy scipy opencv-python pymavlink pyyaml matplotlib`, plus the Gazebo Python
  bindings (`gz.transport13`, `gz.msgs10`, installed with Gazebo Harmonic)

## Running

```bash
cd scene2                 # or scene1 / scene3
./run.sh flight0          # map the bridge -> pointcloud/pc_b2.ply   (Gazebo + RViz2 open)
./run.sh survey           # inspection survey planned only from pc_b2.ply (needs flight0 first)
./run.sh stop             # stop everything
```

`run.sh` passes extra flags through: `./run.sh flight0 --no-gui` (no Gazebo window, much faster -
recommended for scene1, whose railing meshes make the Gazebo window very slow), `--no-rviz`,
`./run.sh survey --replan` after changing the config. The same thing without the script:
`python3 ../bridge_autonomy/run_mission.py config.yaml --flight flight0`.

A run starts Gazebo, ArduPilot SITL, the wind and GPS-shadow simulators, the referee, the VIO, the lidar
mapper, the camera recorder, RViz2 and the flight. Logs: `sceneN/output/logs/`. The drone takes off from
the deck, flies the route, returns to the take-off point and lands (also if it gets stuck for
`motion.stall_abort_s`). Progress: `tail -f sceneN/output/logs/mission.log`.

## How it works

**Simulated world** (`sim_env/`, allowed to use ground truth because it *is* the world)
* wind: mean wind + gusts + turbulence on the drone body through Gazebo's WindEffects (`wind:`)
* GPS shadow: under the deck (from the true deck footprint) SITL's GPS is switched off
* sensor noise: gyro noise and bias, image noise, depth noise, lidar noise, SITL IMU/baro noise (`sensor_noise:`)
* referee (`truth_monitor.py`): compares the drone's estimates with the true pose; evaluation only

**Drone** (never sees ground truth)
* navigation: ArduPilot EKF on GPS in the open, on VIO under the deck (switched automatically before
  the deck and back after); `flight_core.py` follows the route, faces the direction of travel, returns home on a stall
* VIO (`vio.py`): KLT feature tracking on the RGB-D camera + PnP + gyro, fed to ArduPilot as external vision
* mapper (`mapping/record_lidar.py`): every lidar scan is placed with the autopilot's pose at the scan's
  capture time (Gazebo and autopilot clocks are synchronised once at start). RViz2 shows the map
  without water / ground, the drone's path coloured by GPS fix, GPS lost/regained points and the camera

**flight0** (`bridge_autonomy/flight0.py`) is the mapping flight. It stands in for a manual flight, so its
*route* is planned from the real bridge layout (a pilot knows where the bridge is): side passes at three
heights from deck end to deck end, passes over the deck, and - where there is room under the deck -
loops in front of the piers at the top, middle and bottom of the pillars. What it *produces* uses no ground
truth: after landing, this flight's map is split into
* `pointcloud/pc_bN.ply` - **the bridge only** (structure above the water / ground inside the measured
  bridge footprint; the deck the drone took off from is always bridge, so the road stays in)
* `pointcloud/pc_bN_surroundings.ply` - water / ground / banks, coarse, only so later flights know the floor
* `pointcloud/pc_bN.json` - how it was made and the bridge as measured from the cloud

**Later flights** (`./run.sh survey`) measure the bridge from `pc_bN` alone (`bridge_model.py`: deck,
heights, piers and columns, abutments) and plan the inspection from that (`plan_survey.py`): clearance
`safety.min_clearance` from every measured point, fast transit (`motion.v_transit`) between passes,
mapping passes at `motion.v_max`.

## Evaluation (uses ground truth)

```bash
python3 evaluation/eval_coverage.py scene2/scene2_parts scene2/pointcloud/pc_b2.ply   # coverage per part / class / facing
python3 evaluation/map_sharpness.py scene2/config.yaml scene2/pointcloud/pc_b2.ply   # distance of map points to the real surfaces
python3 evaluation/check_plan_vs_truth.py scene2/config.yaml scene2/output/plan.npz  # how close a plan really gets
```

## Known limitations

* flight0's route uses the true bridge layout (piloted flight). Only the point cloud and later flights are ground-truth free.
* VIO is fed noisy sensors and is the weak part: about 0.4 m median error, metres after long GPS gaps. In the
  simulator the autopilot rides out the short (~15 s) gaps on its very good simulated IMU.
* Under-deck surfaces are a few centimetres thicker than real ones (multi-pass pose differences of ~2-7 cm).
* The Gazebo window is slow on large scenes (scene1); use `--no-gui` and watch RViz2.

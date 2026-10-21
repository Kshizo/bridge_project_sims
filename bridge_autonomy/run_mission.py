#!/usr/bin/env python3
"""Launch a full bridge mission from one config file:
  simulated world : Gazebo (+ wind) + ArduCopter SITL + GPS-denial simulator + wind simulator
                    + truth monitor (referee: evaluation only)
  drone           : VIO + lidar mapper (EKF pose + ICP) + camera recorder + the flight itself
  display         : RViz2

Usage: run_mission.py <config.yaml> --flight flight0 [--no-gui] [--no-rviz]   map the bridge -> pc_b2
       run_mission.py <config.yaml> [--replan] [--no-gui] [--no-rviz]            survey planned from pc_b2
       run_mission.py <config.yaml> --stop
Logs: <out_dir>/logs/*.log"""
import argparse, os, signal, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bridge_model import load_config
from sdf_utils import model_sdf_for_include, link_pose

ap = argparse.ArgumentParser(); ap.add_argument("config")
ap.add_argument("--replan", action="store_true"); ap.add_argument("--no-gui", action="store_true")
ap.add_argument("--no-rviz", action="store_true"); ap.add_argument("--stop", action="store_true")
ap.add_argument("--flight", choices=["survey", "flight0"], default="survey",
                help="flight0: map the bridge from scratch -> pc_b2; survey: inspection planned from pc_b2")
a = ap.parse_args()
cfg = load_config(a.config); sc = cfg["scene"]; cfgp = os.path.abspath(a.config)
out = sc["out_dir"]; logs = os.path.join(out, "logs"); os.makedirs(logs, exist_ok=True)
sim_root = os.path.dirname(sc["world_sdf"])
env = dict(os.environ, GZ_PARTITION=sc["gz_partition"])
agz = os.path.expanduser(cfg["sitl"].get("ardupilot_gazebo_dir", "~/ardupilot_gazebo"))
env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = ":".join([os.path.join(agz, "build"), env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")])
env["GZ_SIM_RESOURCE_PATH"] = ":".join([os.path.join(cfg["root"], "models"), os.path.join(agz, "models"),
                                        os.path.join(agz, "worlds"), sim_root, env.get("GZ_SIM_RESOURCE_PATH", "")])
os.environ.update(env)
PROGS = {"arducopter", "rviz2", "mavproxy.py"}
SCRIPTS = {"sim_vehicle.py", "mavproxy.py", "record_lidar.py", "record_camera.py", "vio.py", "gps_denial_sim.py", "mission.py",
           "run_mission.py", "flight0.py", "wind_sim.py", "truth_monitor.py"}


def ours(argv):
    """True if this process is part of the stack (matched on program / script name, not on text)."""
    prog = os.path.basename(argv[0])
    if prog in PROGS:
        return True
    if prog.startswith("python") and len(argv) > 1:
        script = next((x for x in argv[1:] if not x.startswith("-")), "")
        return os.path.basename(script) in SCRIPTS
    ours_world = any(x in (sc["world_sdf"], os.path.join(out, "world_runtime.sdf")) for x in argv)
    return prog == "gz" and argv[1:3] == ["sim", "-v4"] and ours_world or argv[:3] in (["gz", "sim", "server"], ["gz", "sim", "gui"])


def stop_all():
    me = os.getpid()
    for line in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]:
        pid, args = line.strip().split(None, 1)
        if int(pid) != me and ours(args.split()):
            try:
                os.kill(int(pid), signal.SIGKILL if "arducopter" in args else signal.SIGTERM)
            except ProcessLookupError:
                pass


if a.stop:
    stop_all(); sys.exit()
procs = []


def start(name, cmd, cwd=None):
    f = open(os.path.join(logs, name + ".log"), "w")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    procs.append(p); print(f"started {name} (log {f.name})", flush=True)
    return p


def wait_log(name, text, timeout=180):
    t = time.time()
    while text not in open(os.path.join(logs, name + ".log"), errors="replace").read():
        if time.time() - t > timeout:
            raise SystemExit(f"{name}: '{text}' not seen in {timeout}s")
        time.sleep(1)


stop_all(); time.sleep(2)
py = sys.executable
ROOT = os.path.dirname(HERE)
if a.flight == "survey":
    if not os.path.exists(sc["prior_cloud"]):
        raise SystemExit(f"{sc['prior_cloud']} not found - run flight0 first:  run_mission.py {a.config} --flight flight0")
    if a.replan or not os.path.exists(os.path.join(out, "plan.npz")):
        subprocess.run([py, os.path.join(HERE, "plan_survey.py"), cfgp, "--plot"], check=True)

# SITL params: config + camera offset (FRD body frame) from the drone model
st = cfg["sitl"]
_, t_cam, _ = link_pose(model_sdf_for_include(sc["world_sdf"], sc["vehicle_model"]), cfg["vio"]["camera_link"])
params = dict(st["params"]); params.update(VISO_POS_X=round(t_cam[0], 3), VISO_POS_Y=round(-t_cam[1], 3), VISO_POS_Z=round(-t_cam[2], 3))
if cfg.get("sensor_noise", {}).get("enabled"):
    params.update(cfg["sensor_noise"].get("autopilot", {}))
parm = os.path.join(out, "sitl_generated.parm")
open(parm, "w").write("".join(f"{k} {v}\n" for k, v in params.items()))

# runtime world: the scene + Gazebo's WindEffects system (wind velocity comes from sim_env/wind_sim.py)
import xml.etree.ElementTree as ET
tree = ET.parse(sc["world_sdf"]); world_el = tree.getroot().find("world")
W = cfg.get("wind", {})
if W.get("enabled"):
    pl = ET.SubElement(world_el, "plugin", {"filename": "gz-sim-wind-effects-system", "name": "gz::sim::systems::WindEffects"})
    ET.SubElement(pl, "force_approximation_scaling_factor").text = str(W["drag"])
    wind_el = ET.SubElement(world_el, "wind"); ET.SubElement(wind_el, "linear_velocity").text = "0 0 0"
runtime_world = os.path.join(out, "world_runtime.sdf")
tree.write(runtime_world)

for f in ("map_points.npy", "trajectory.npy", "vio_log.npy", "gps_status.json", "vio_status.json", "nav_status.json",
          "current_plan.npz"):
    if os.path.exists(os.path.join(out, f)):
        os.remove(os.path.join(out, f))
start("gz", ["gz", "sim", "-v4", "-r", runtime_world] + (["-s", "--headless-rendering"] if a.no_gui else []))
wait_log("gz", "Serving world names"); time.sleep(3)
ardu = st["ardupilot_dir"]
cmd = ["sim_vehicle.py", "-v", st["vehicle"], "-f", st["frame"], "--model", "JSON", "--no-rebuild", "-w",
       "--mavproxy-args=--non-interactive"]
cmd += [f"--add-param-file={p}" for p in st["param_files"]] + [f"--add-param-file={parm}"]
cmd += [f"--out={o}" for o in cfg["mavlink"]["sitl_outputs"]]
# SITL home = the world's <spherical_coordinates>, so SITL GPS and Gazebo agree
_sp = world_el.find("spherical_coordinates")
if _sp is not None:
    _g = lambda k, d=0.0: float(_sp.findtext(k, d))
    cmd.append(f"--custom-location={_g('latitude_deg')},{_g('longitude_deg')},{_g('elevation')},{_g('heading_deg')}")
start("sitl", cmd, cwd=ardu)
mp = cfg["mapping"]
ros_env = lambda c: ["bash", "-c", "source /opt/ros/jazzy/setup.bash && exec " + " ".join(f"'{x}'" for x in c)]
rviz = not a.no_rviz
# --- simulated world (may use ground truth) ---
start("gps_denial", [py, "-u", os.path.join(ROOT, "sim_env", "gps_denial_sim.py"), cfgp])
if W.get("enabled"):
    start("wind", [py, "-u", os.path.join(ROOT, "sim_env", "wind_sim.py"), cfgp])
if cfg.get("evaluation", {}).get("enabled"):
    tm = [py, "-u", os.path.join(ROOT, "sim_env", "truth_monitor.py"), cfgp]
    start("truth_monitor", ros_env(tm + ["--ros"]) if rviz else tm)
# --- drone (no ground truth) ---
start("vio", [py, "-u", os.path.join(HERE, "vio.py"), cfgp])
rec = [py, "-u", os.path.join(ROOT, "mapping", "record_lidar.py"), cfgp]
start("recorder", ros_env(rec + ["--ros"]) if rviz else rec)
cr = cfg.get("camera_recording", {})
if cr.get("enabled", True):
    cam = [py, "-u", os.path.join(ROOT, "mapping", "record_camera.py"), out, "--topic", cfg["vio"]["rgbd_topic"]] + \
          (["--depth", "--depth-range", *map(str, cfg["vio"]["depth_range"])] if cr.get("depth") else [])
    start("camera", ros_env(cam + ["--ros"]) if rviz else cam)
if rviz:
    start("rviz", ["bash", "-c", f"source /opt/ros/jazzy/setup.bash && exec rviz2 -d {mp['rviz_config']}"])
mission = start("mission", [py, "-u", os.path.join(HERE, "flight0.py" if a.flight == "flight0" else "mission.py"), cfgp])
try:
    with open(os.path.join(logs, "mission.log"), errors="replace") as f:
        while mission.poll() is None:
            line = f.readline()
            if line:
                print("[mission]", line, end="", flush=True)
            else:
                time.sleep(0.5)
        print(f.read(), end="")
    print(f"mission exited ({mission.returncode}); map in {out}. Ctrl-C closes everything.", flush=True)
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop_all()

#!/usr/bin/env python3
"""SIMULATED WORLD. Wind for Gazebo's WindEffects system.

Publishes the world wind velocity on /world/<world>/wind at `rate_hz`:
  mean wind      : `mean_speed` m/s from `from_deg` (compass, 0 = from north, 90 = from east)
  turbulence     : first-order Gauss-Markov per axis (`turbulence_std` m/s, correlation `turbulence_tau` s),
                   vertical with `vertical_std`
  gusts          : random events (mean `gust_interval` s apart) along the mean direction, raised-cosine
                   profile `gust_duration` s long, up to `gust_speed` m/s on top of the mean
WindEffects turns it into a force on every link with <enable_wind> (the drone's base_link):
F = mass * drag * (wind - link velocity), `drag` set in the generated world. The autopilot does not
know the wind; it has to reject it.

Usage: wind_sim.py <config.yaml>"""
import math, os, signal, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge_autonomy"))
from bridge_model import load_config
from gz.transport13 import Node
from gz.msgs10.wind_pb2 import Wind

cfg = load_config(sys.argv[1]); W = cfg["wind"]; sc = cfg["scene"]
rng = np.random.default_rng(W.get("seed", 0))
to = math.radians(W["from_deg"] + 180.0)                    # direction the air moves towards (compass)
d = np.array([math.sin(to), math.cos(to), 0.0])             # ENU unit vector
mean = W["mean_speed"] * d
node = Node(); pub = node.advertise(f"/world/{sc['world_name']}/wind", Wind)
stop = []
signal.signal(signal.SIGTERM, lambda *_: stop.append(1)); signal.signal(signal.SIGINT, lambda *_: stop.append(1))
dt = 1.0 / W["rate_hz"]; turb = np.zeros(3); sig = np.array([W["turbulence_std"]] * 2 + [W["vertical_std"]])
gust_t, gust_a, next_gust = None, 0.0, rng.exponential(W["gust_interval"])
t, last_print = 0.0, 0.0
log = open(os.path.join(sc["out_dir"], "logs", "wind.csv"), "w"); log.write("t,east,north,up,speed\n")
print(f"wind: {W['mean_speed']} m/s from {W['from_deg']} deg, turbulence {W['turbulence_std']} m/s, "
      f"gusts up to +{W['gust_speed']} m/s every ~{W['gust_interval']} s", flush=True)
while not stop:
    a = dt / W["turbulence_tau"]
    turb = turb * (1 - a) + sig * math.sqrt(2 * a) * rng.normal(size=3)
    if gust_t is None and t >= next_gust:
        gust_t, gust_a = t, rng.uniform(0.5, 1.0) * W["gust_speed"]
    g = 0.0
    if gust_t is not None:
        x = (t - gust_t) / W["gust_duration"]
        if x >= 1:
            gust_t, next_gust = None, t + rng.exponential(W["gust_interval"])
        else:
            g = gust_a * 0.5 * (1 - math.cos(2 * math.pi * x))
    v = mean + g * d + turb
    m = Wind(); m.enable_wind = True
    m.linear_velocity.x, m.linear_velocity.y, m.linear_velocity.z = map(float, v)
    pub.publish(m)
    log.write(f"{t:.2f},{v[0]:.3f},{v[1]:.3f},{v[2]:.3f},{np.linalg.norm(v):.3f}\n")
    if t - last_print >= 10:
        last_print = t; log.flush()
        print(f"t {t:6.0f}  wind {np.linalg.norm(v[:2]):.1f} m/s{'  GUST' if g > 0.5 else ''}", flush=True)
    t += dt
    time.sleep(dt)

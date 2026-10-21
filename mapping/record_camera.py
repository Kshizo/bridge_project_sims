#!/usr/bin/env python3
"""Record what the drone's RGB-D camera sees.

Writes <out>/camera_rgb.mp4 (the RGB stream, frame rate = camera rate, so the video runs in
sim time) and <out>/camera_frames.csv (frame, sim time, the drone's own estimated x y z and GPS fix
from the mapper's nav_status.json - no ground truth) so every frame can be geotagged.
With --depth also <out>/camera_depth.mp4 (depth colourised, near = warm).
With --ros the RGB frames are also published as sensor_msgs/Image on /bridge/camera.

Usage: GZ_PARTITION=scene2 record_camera.py <out_dir> --topic /mapper/rgbd [--depth --depth-range 0.3 25 --ros]"""
import argparse, csv, json, os, queue, signal, threading, time
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--topic", required=True, help="<topic> of the rgbd_camera (…/image, …/depth_image)")
ap.add_argument("--depth", action="store_true"); ap.add_argument("--depth-range", type=float, nargs=2, default=[0.3, 25.0])
ap.add_argument("--ros", action="store_true")
ap.add_argument("--fps", type=float, default=0, help="video frame rate (default: measured camera rate)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

frames = queue.Queue(maxsize=60)
nav_file = os.path.join(a.out, "nav_status.json")


def stamp(h):
    return h.stamp.sec + h.stamp.nsec * 1e-9


def on_rgb(m):
    try:
        frames.put_nowait(("rgb", stamp(m.header), np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1).copy()))
    except queue.Full:
        pass                                                # writer behind: drop rather than block gz transport


def on_depth(m):
    try:
        frames.put_nowait(("depth", stamp(m.header), np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()))
    except queue.Full:
        pass


stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set()); signal.signal(signal.SIGINT, lambda *_: stop.set())
node = Node()
node.subscribe(Image, a.topic + "/image", on_rgb)
if a.depth:
    node.subscribe(Image, a.topic + "/depth_image", on_depth)

ros = None
if a.ros:
    import rclpy
    from sensor_msgs.msg import Image as RImage
    rclpy.init(); rn = rclpy.create_node("bridge_camera"); ros = rn.create_publisher(RImage, "/bridge/camera", 1)

writers, nav, nav_read = {}, {}, 0.0
log = csv.writer(open(os.path.join(a.out, "camera_frames.csv"), "w", newline=""))
log.writerow(["frame", "sim_time", "est_x", "est_y", "est_z", "gps_fix"])
first_stamps, n = [], 0
print(f"recording {a.topic} -> {a.out}", flush=True)


def writer(kind, shape, fps):
    w = cv2.VideoWriter(os.path.join(a.out, f"camera_{kind}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (shape[1], shape[0]))
    if not w.isOpened():
        raise SystemExit(f"cannot open camera_{kind}.mp4 for writing")
    print(f"camera_{kind}.mp4: {shape[1]}x{shape[0]} @ {fps:.1f} fps", flush=True)
    return w


while not stop.is_set():
    try:
        kind, t, img = frames.get(timeout=0.5)
    except queue.Empty:
        continue
    if kind == "rgb" and not a.fps and len(first_stamps) < 10:   # frame rate from the camera's own stamps
        first_stamps.append(t)
        if len(first_stamps) < 10:
            continue
    if not a.fps and len(first_stamps) < 10:
        continue                                            # depth before the rate is known
    fps = a.fps or (len(first_stamps) - 1) / max(first_stamps[-1] - first_stamps[0], 1e-3)
    if kind == "depth":
        lo, hi = a.depth_range
        d = np.nan_to_num(img, nan=hi, posinf=hi, neginf=hi)
        g = (255 * (1 - np.clip((d - lo) / (hi - lo), 0, 1))).astype(np.uint8)
        if "depth" not in writers:
            writers["depth"] = writer("depth", g.shape, fps)
        writers["depth"].write(cv2.applyColorMap(g, cv2.COLORMAP_TURBO))
        continue
    if "rgb" not in writers:
        writers["rgb"] = writer("rgb", img.shape, fps)
    writers["rgb"].write(cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR))
    if time.time() - nav_read > 0.1:
        nav_read = time.time()
        try:
            nav = json.load(open(nav_file))
        except (OSError, ValueError):
            pass
    p = nav.get("p", (float("nan"),) * 3); gps = nav.get("gps_fix")
    log.writerow([n, f"{t:.3f}", *(f"{v:.3f}" for v in p), "" if gps is None else int(gps)])
    n += 1
    if ros is not None:
        msg = RImage(); msg.header.frame_id = "camera"; msg.header.stamp = rn.get_clock().now().to_msg()
        msg.height, msg.width = img.shape[:2]; msg.encoding = "rgb8"; msg.step = msg.width * 3
        msg.data = img[:, :, :3].tobytes(); ros.publish(msg)
    if n % 300 == 0:
        print(f"frames {n}  sim t {t:.0f}", flush=True)
for w in writers.values():
    w.release()
print(f"final: {n} RGB frames", flush=True)

"""Ground-truth bridge geometry: the SIMULATED WORLD, plus the pilot of flight0.

The world side (GPS-denial simulator, evaluation) knows where the bridge really is, because it is
the simulator. flight0 is a piloted mapping flight, so its route may use where the bridge is
(MeshBridge). Nothing else on the drone imports this module: the VIO, the mapper and every flight
after flight0 only know what the drone measured (pc_b2, VIO, EKF).
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge_autonomy"))
from bridge_model import load_obj, spawn_pose                             # plain OBJ / SDF readers
from scipy.spatial import cKDTree

ENV_SOLID = {"terrain", "road", "vegetation", "vehicle", "other"}


def load_parts(parts_dir, bridge_model=None, region=None):
    """parts.json in any of the scene formats (a bare list, or {"parts": [...], "labels": ...}). Each part gets
    "obj" (its mesh path: "file" if given, else <name>.obj) and a missing bbox is computed from the mesh.
    bridge_model: keep only bridge parts of this "model" (scenes with several bridges); environment parts
    (terrain / water / vegetation / ...) are always kept.
    region: {"min": [x, y], "max": [x, y]}: keep only bridge parts whose bbox centre lies inside (map part of a
    long bridge)."""
    meta = json.load(open(os.path.join(parts_dir, "parts.json")))
    parts = meta if isinstance(meta, list) else meta["parts"]
    labels = {} if isinstance(meta, list) else meta.get("labels", {})
    out = []
    for p in parts:
        p = dict(p)
        p["obj"] = os.path.join(parts_dir, p.get("file") or p["name"] + ".obj")
        env = p["class"] in ENV_SOLID | {"water"}
        if bridge_model and not env and p.get("model") not in (None, bridge_model):
            continue
        if "bbox_min" not in p:
            T = load_obj(p["obj"]).reshape(-1, 3)
            if not len(T):
                continue
            p["bbox_min"], p["bbox_max"] = T.min(0).tolist(), T.max(0).tolist()
        p["env"] = env
        if region and not env:
            c = (np.array(p["bbox_min"][:2]) + np.array(p["bbox_max"][:2])) / 2
            if np.any(c < region["min"]) or np.any(c > region["max"]):
                continue
        out.append(p)
    return out, labels


SUPER = {"deck", "girder", "cross_beam", "bearing", "railing", "arch_truss"}
SUB = {"pier_cap", "pier_column", "footing", "abutment"}


class WorldTruth:
    def __init__(self, cfg):
        sc = cfg["scene"]; self.cfg = cfg
        self.parts, _ = load_parts(sc["parts_dir"], sc.get("bridge_model"), sc.get("region"))
        deck = [p for p in self.parts if p["class"] == "deck"]
        D = np.concatenate([load_obj(p["obj"]).reshape(-1, 3) for p in deck])
        self.origin = D[:, :2].mean(0)
        w, V = np.linalg.eigh(np.cov((D[:, :2] - self.origin).T))
        self.u = V[:, np.argmax(w)]
        self.v = np.array([-self.u[1], self.u[0]])
        s, c = self.to_sc(D[:, :2]).T
        self.s_range = (s.min(), s.max()); self.c_range = (np.percentile(c, 1), np.percentile(c, 99))
        self.deck_bottom = float(min(p["bbox_min"][2] for p in deck))
        self.z_low = float(min(p["bbox_min"][2] for p in self.parts if not p["env"]))

    def to_sc(self, xy):
        d = np.atleast_2d(xy) - self.origin
        return np.stack([d @ self.u, d @ self.v], 1)

    def to_xy(self, s, c):
        return self.origin + np.multiply.outer(s, self.u) + np.multiply.outer(c, self.v)

    def in_gps_denied_zone(self, P):
        """True GPS shadow: under the deck footprint (+ margin), below the deck."""
        g = self.cfg["gps_denied"]; m = g["footprint_margin"]
        P = np.atleast_2d(P); s, c = self.to_sc(P[:, :2]).T
        return ((s >= self.s_range[0] - m) & (s <= self.s_range[1] + m) & (c >= self.c_range[0] - m) &
                (c <= self.c_range[1] + m) & (P[:, 2] < self.deck_bottom + g["top_offset"]))

    def gps_denied_box(self):
        g = self.cfg["gps_denied"]; m = g["footprint_margin"]
        s0, s1 = self.s_range[0] - m, self.s_range[1] + m
        c0, c1 = self.c_range[0] - m, self.c_range[1] + m
        z0, z1 = self.z_low, self.deck_bottom + g["top_offset"]
        x, y = self.to_xy((s0 + s1) / 2, (c0 + c1) / 2)
        return {"center": [float(x), float(y), (z0 + z1) / 2], "size": [s1 - s0, c1 - c0, z1 - z0],
                "yaw": float(math.atan2(self.u[1], self.u[0]))}


def sample_tris(T, spacing, rng):
    if not len(T):
        return np.zeros((0, 3))
    A = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    n = rng.poisson(A / spacing ** 2) + 1
    ti = np.repeat(np.arange(len(T)), n)
    r = rng.random((len(ti), 2)); f = r.sum(1) > 1; r[f] = 1 - r[f]
    t = T[ti]
    return t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])


class MeshBridge:
    """The real bridge from the part meshes, with the same interface as bridge_autonomy.bridge_model.Bridge.
    Used ONLY by flight0, which is a piloted flight: the pilot knows where the bridge is."""
    def __init__(self, cfg, rng_seed=0):
        sc = cfg["scene"]
        self.cfg = cfg
        rng = np.random.default_rng(rng_seed)
        self.parts, self.labels = load_parts(sc["parts_dir"], sc.get("bridge_model"), sc.get("region"))
        for p in self.parts:                                   # unique keys (several models may reuse names)
            p["name"] = p["obj"]
        self.tris = {p["name"]: load_obj(p["obj"]) for p in self.parts}
        cls = lambda c: [p for p in self.parts if p["class"] == c]
        bridge_parts = [p for p in self.parts if p["class"] in SUPER | SUB]

        # --- axis (PCA of deck vertices in xy) ---------------------------------------
        D = np.concatenate([self.tris[p["name"]].reshape(-1, 3) for p in cls("deck")])
        self.origin = D[:, :2].mean(0)
        w, V = np.linalg.eigh(np.cov((D[:, :2] - self.origin).T))
        self.u = V[:, np.argmax(w)]                      # along the bridge
        if self.u[np.argmax(np.abs(self.u))] < 0:
            self.u = -self.u
        self.v = np.array([-self.u[1], self.u[0]])       # across (left of u)
        sd, cd = self.to_sc(D[:, :2]).T
        self.s_range = (sd.min(), sd.max())
        reg = sc.get("region")
        if reg:                                            # long parts can stick out of the region: clip the extent
            corners = np.array([[x, y] for x in (reg["min"][0], reg["max"][0]) for y in (reg["min"][1], reg["max"][1])])
            rs = self.to_sc(corners)[:, 0]
            self.s_range = (max(self.s_range[0], rs.min()), min(self.s_range[1], rs.max()))
        self.c_range = (np.percentile(cd, 1), np.percentile(cd, 99))
        self.c_mid = sum(self.c_range) / 2
        self.half_width = (self.c_range[1] - self.c_range[0]) / 2

        # --- heights -----------------------------------------------------------------
        self.deck_top = float(np.percentile(D[:, 2], 99))
        self.deck_bottom = float(min(p["bbox_min"][2] for p in cls("deck")))
        soff = [p["bbox_min"][2] for p in self.parts if p["class"] in ("girder", "cross_beam", "deck")]
        self.soffit = float(min(soff))
        rails = cls("railing")
        self.railing_top = float(max([p["bbox_max"][2] for p in rails] + [self.deck_top]))

        # --- floor: highest of terrain / road / water under each xy ------------------
        floor_parts = [p for p in self.parts if p["class"] in ENV_SOLID | {"water"}]
        self._floor_tris = np.concatenate([self.tris[p["name"]] for p in floor_parts]) if floor_parts else np.zeros((0, 3, 3))
        water = cls("water")
        self.water_z = max(p["bbox_max"][2] for p in water) if water else None

        # --- piers: cluster columns along the axis -------------------------------------
        gap = cfg["survey"]["pier_cluster_gap"]
        cols = []
        for p in cls("pier_column"):
            T = self.tris[p["name"]].reshape(-1, 3)
            s, c = self.to_sc(T[:, :2]).T
            cols.append({"name": p["name"], "s": (s.min() + s.max()) / 2, "c": (c.min() + c.max()) / 2,
                         "hs": (s.max() - s.min()) / 2, "hc": (c.max() - c.min()) / 2,
                         "zmin": T[:, 2].min(), "zmax": T[:, 2].max()})
        cols.sort(key=lambda k: k["s"])
        self.piers = []
        for col in cols:
            if self.piers and col["s"] - self.piers[-1]["cols"][-1]["s"] < gap:
                self.piers[-1]["cols"].append(col)
            else:
                self.piers.append({"cols": [col]})
        caps = cls("pier_cap")
        for pr in self.piers:
            cs = pr["cols"]
            pr["s"] = float(np.mean([k["s"] for k in cs]))
            pr["s_lo"] = min(k["s"] - k["hs"] for k in cs); pr["s_hi"] = max(k["s"] + k["hs"] for k in cs)
            pr["c_lo"] = min(k["c"] - k["hc"] for k in cs); pr["c_hi"] = max(k["c"] + k["hc"] for k in cs)
            pr["c"] = (pr["c_lo"] + pr["c_hi"]) / 2
            pr["zmin"] = min(k["zmin"] for k in cs)
            top = [k["zmax"] for k in cs]
            for cp in caps:
                T = self.tris[cp["name"]].reshape(-1, 3); s = self.to_sc(T[:, :2])[:, 0]
                if s.min() - 0.5 <= pr["s"] <= s.max() + 0.5:
                    top.append(T[:, 2].min())
            pr["ztop"] = float(min(top))
            pr["cols"] = sorted(cs, key=lambda k: k["c"])

        # --- abutments + supports ------------------------------------------------------
        self.abutments = []
        for p in cls("abutment"):
            T = self.tris[p["name"]].reshape(-1, 3); s = self.to_sc(T[:, :2])[:, 0]
            # the face toward the spans: the side of the abutment nearer the bridge middle
            mid = sum(self.s_range) / 2
            face = s.max() if abs(s.max() - mid) < abs(s.min() - mid) else s.min()
            self.abutments.append({"name": p["name"], "face_s": float(face), "dir": float(np.sign(mid - face))})
        sup = sorted([a["face_s"] for a in self.abutments] + [p["s"] for p in self.piers])
        self.supports = sup if sup else list(self.s_range)
        self.midspans = [(a + b) / 2 for a, b in zip(self.supports, self.supports[1:])]

        # --- structure samples (for look-at and clearance) ----------------------------
        res = cfg["safety"]["grid_resolution"]
        # obstacles: EVERY bridge part, also those outside scene.region (the region only limits what is surveyed)
        every, _ = load_parts(sc["parts_dir"], sc.get("bridge_model"))
        obst = [p for p in every if p["class"] in SUPER | SUB]
        self.struct_pts = np.concatenate([sample_tris(self.tris[p["obj"]] if p["obj"] in self.tris else load_obj(p["obj"]),
                                                      res / 2.5, rng) for p in obst])
        reg = sc.get("region")
        if reg:                                            # ...but only those near the region matter for the plan
            mg = cfg["safety"]["grid_margin"]
            near = lambda Q: np.all((Q[:, :2] >= np.array(reg["min"]) - mg) & (Q[:, :2] <= np.array(reg["max"]) + mg), axis=1)
            self.struct_pts = self.struct_pts[near(self.struct_pts)]
        self.struct_tree = cKDTree(self.struct_pts)
        env_pts = [sample_tris(self.tris[p["name"]], res / 2.5, rng) for p in self.parts
                   if p["class"] in ENV_SOLID and p["name"] in self.tris]
        self.env_pts = np.concatenate(env_pts) if env_pts else np.zeros((0, 3))
        if reg:
            self.env_pts = self.env_pts[near(self.env_pts)]
        self.spawn, self.spawn_yaw = spawn_pose(sc["world_sdf"], sc["vehicle_model"])

    # --- frames ------------------------------------------------------------------------
    def to_sc(self, xy):
        d = np.atleast_2d(xy) - self.origin
        return np.stack([d @ self.u, d @ self.v], 1)

    def to_xy(self, s, c):
        return self.origin + np.multiply.outer(s, self.u) + np.multiply.outer(c, self.v)

    def sc_point(self, s, c, z):
        x, y = self.to_xy(s, c)
        return np.array([x, y, z])

    # --- floor height --------------------------------------------------------------------
    def floor_at(self, xy):
        xy = np.atleast_2d(xy); out = np.full(len(xy), -np.inf)
        for v0, v1, v2 in self._floor_tris:
            d = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
            if abs(d) < 1e-9:
                continue
            l0 = ((v1[1] - v2[1]) * (xy[:, 0] - v2[0]) + (v2[0] - v1[0]) * (xy[:, 1] - v2[1])) / d
            l1 = ((v2[1] - v0[1]) * (xy[:, 0] - v2[0]) + (v0[0] - v2[0]) * (xy[:, 1] - v2[1])) / d
            l2 = 1 - l0 - l1
            ins = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            z = l0 * v0[2] + l1 * v1[2] + l2 * v2[2]
            out = np.where(ins & (z > out), z, out)
        return out

    # --- GPS-denied zone ---------------------------------------------------------------
    def in_gps_denied_zone(self, P):
        g = self.cfg["gps_denied"]
        P = np.atleast_2d(P)
        s, c = self.to_sc(P[:, :2]).T
        m = g["footprint_margin"]
        return ((s >= self.s_range[0] - m) & (s <= self.s_range[1] + m) &
                (c >= self.c_range[0] - m) & (c <= self.c_range[1] + m) &
                (P[:, 2] < self.deck_bottom + g["top_offset"]))

    def summary(self):
        return {"axis": self.u.round(3).tolist(), "s_range": np.round(self.s_range, 2).tolist(),
                "deck_width": round(2 * self.half_width, 2), "deck_top": round(self.deck_top, 2),
                "soffit": round(self.soffit, 2), "railing_top": round(self.railing_top, 2),
                "water_z": self.water_z, "piers": [{"s": round(p["s"], 2), "columns": len(p["cols"]),
                                                   "top": round(p["ztop"], 2)} for p in self.piers],
                "abutment_faces_s": [round(a["face_s"], 2) for a in self.abutments],
                "spawn": self.spawn.round(2).tolist()}

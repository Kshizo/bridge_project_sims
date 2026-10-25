#!/usr/bin/env python3
"""Plan a full inspection/mapping survey for any bridge described by parts.json.

Primitives (all placed from Bridge geometry + config offsets):
  side passes at several heights and a pass over the deck (outside of the bridge), and one
  section per span between supports: a loop in front of the piers / abutment at both ends of
  the section and past the outer columns, flown at the top of the pillars, then the middle,
  then the bottom (survey.section_levels). Each pier face is mapped from the section it faces.
Every sample is checked against the occupancy clearance field; anything closer than
safety.min_clearance is cut out. Primitives are ordered greedily and joined with A* routes
through free space, smoothed, time-parameterised (speed from clearance) and given a yaw
that looks at the nearest structure. Each sample is flagged if it lies in the GPS-denied zone.

The bridge is whatever the point cloud cfg.scene.prior_cloud (pc_b2, made by flight0) contains;
the planner never sees meshes, labels or ground truth.

Usage: plan_survey.py <config.yaml> [--plot]      -> <out_dir>/plan.npz (+ plan.png, plan.json)
"""
import argparse, heapq, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_model import Bridge, Occupancy, load_config


def resample(P, step):
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    d = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 1e-6:
        return P[:1]
    t = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)


class Planner:
    def __init__(self, cfg, br=None, extra_bounds=None):
        """br: a Bridge measured from a point cloud (default: cfg.scene.prior_cloud, i.e. pc_b2).
        extra_bounds: points the occupancy grid must also cover (e.g. the drone's current position)."""
        self.cfg = cfg
        t = time.time(); self.br = br or Bridge(cfg); self.occ = Occupancy(self.br, extra_bounds)
        print(f"model + occupancy in {time.time()-t:.1f} s; grid {self.occ.shape.tolist()}", flush=True)
        self.minc = cfg["safety"]["min_clearance"]
        self.step = cfg["motion"]["sample_step"]

    # ---- helpers -----------------------------------------------------------------
    def level(self, ref, offset, xy=None):
        br = self.br
        base = {"deck_top": br.deck_top, "soffit": br.soffit, "railing_top": br.railing_top,
                "deck_bottom": br.deck_bottom}.get(ref)
        if ref == "floor":
            base = float(np.max(br.floor_at(xy))) if xy is not None else (br.water_z or 0.0)
        return base + offset

    def valid_pieces(self, P, name):
        """Cut a primitive where it violates clearance; keep pieces longer than 1 m."""
        P = resample(P, self.step)
        ok = self.occ.clearance(P) >= self.minc
        pieces, cur = [], []
        for p, o in zip(P, ok):
            if o:
                cur.append(p)
            elif cur:
                pieces.append(np.array(cur)); cur = []
        if cur:
            pieces.append(np.array(cur))
        pieces = [q for q in pieces if len(q) * self.step > 1.0]
        dropped = 1 - sum(len(q) for q in pieces) / max(len(P), 1)
        return [(name, q) for q in pieces], dropped

    def line_sc(self, s0, c0, s1, c1, z0, z1=None):
        z1 = z0 if z1 is None else z1
        n = max(2, int(math.hypot(s1 - s0, c1 - c0) / self.step))
        s = np.linspace(s0, s1, n); c = np.linspace(c0, c1, n)
        xy = self.br.to_xy(s, c)
        return np.column_stack([xy, np.linspace(z0, z1, n)])

    # ---- survey primitives ---------------------------------------------------------
    def primitives(self, kinds=("exterior", "sections")):
        br, sv = self.br, self.cfg["survey"]
        prims, report = [], {}
        self.fixed = set()

        def add(name, P, fixed=False):
            pcs, dropped = self.valid_pieces(P, name)
            prims.extend(pcs); report[name] = round(dropped, 3)
            if fixed:
                self.fixed.add(name)                         # flown in the given direction only

        s0 = br.s_range[0] - sv["side_extension"]; s1 = br.s_range[1] + sv["side_extension"]   # the bridge's ends
        for side in ((-1, 1) if "exterior" in kinds else ()):
            c = br.c_mid + side * (br.half_width + sv["standoff_side"])
            for lv in sv["side_levels"]:
                xy = br.to_xy(np.linspace(s0, s1, 50), np.full(50, c))
                z = self.level(lv["ref"], lv["offset"], xy)
                add(f"side{'LR'[side > 0]}_{lv['ref']}{lv['offset']:+.1f}", self.line_sc(s0, c, s1, c, z))
        if "exterior" in kinds:
            # passes over the deck: one per top_lane_spacing of deck width (one lane on a narrow deck)
            n = max(1, int(math.ceil(2 * br.half_width / sv["top_lane_spacing"])))
            for i in range(n):
                c = br.c_mid - br.half_width + (i + 0.5) * 2 * br.half_width / n
                add("top" if n == 1 else f"top{i+1}", self.line_sc(s0, c, s1, c, br.railing_top + sv["top_pass_offset"]))
        if "sections" not in kinds:
            return prims, report

        # one section per span between supports. In every section a closed loop in front of the supports
        # at both ends (section_standoff from pier faces, abutment_standoff from abutment faces) and
        # section_side_standoff beyond the outer columns, flown once per section_levels height, in order:
        # top of the pillars, middle, bottom. A pier face is mapped from the section it faces, its back
        # face from the section on the other side.
        ends = {pr["s"]: ("pier", pr) for pr in br.piers}
        ends.update({a["face_s"]: ("abutment", a) for a in br.abutments})

        def front(s, toward):
            kind, o = ends[s]
            if kind == "pier":
                return (o["s_hi"] if toward > 0 else o["s_lo"]) + toward * sv["section_standoff"]
            return o["face_s"] + toward * sv["abutment_standoff"]
        for k, (sa, sb) in enumerate(zip(br.supports, br.supports[1:])):
            prs = [ends[s][1] for s in (sa, sb) if s in ends and ends[s][0] == "pier"]
            if not prs:
                continue
            s_from, s_to = front(sa, 1), front(sb, -1)
            c_lo = min(p["c_lo"] for p in prs) - sv["section_side_standoff"]
            c_hi = max(p["c_hi"] for p in prs) + sv["section_side_standoff"]
            corners = np.array([(s_from, c_lo), (s_from, c_hi), (s_to, c_hi), (s_to, c_lo), (s_from, c_lo)], float)
            loop = resample(np.column_stack([corners, np.zeros(len(corners))]), self.step)
            xy = br.to_xy(loop[:, 0], loop[:, 1])
            z = {}
            for i, lv in enumerate(sv["section_levels"]):
                if lv["ref"] == "pier_top":
                    z[i] = min(p["ztop"] for p in prs) + lv["offset"]
                elif lv["ref"] != "middle":
                    z[i] = self.level(lv["ref"], lv["offset"], xy)
            for i, lv in enumerate(sv["section_levels"]):
                if lv["ref"] == "middle":
                    z[i] = (max(z.values()) + min(z.values())) / 2 + lv["offset"]
            # only fly under the deck if there is room: every level between floor + clearance and the
            # deck underside - clearance (the loops run beside the pier caps, not under them); keep top -> bottom
            z_lo = float(np.max(br.floor_at(xy))) + self.minc
            z_hi = br.soffit - self.minc
            levels = sorted({round(v, 2) for v in z.values() if z_lo <= v <= z_hi}, reverse=True)
            if not levels:
                report[f"section{k+1}"] = f"skipped: no room under the deck ({z_hi - z_lo + 2 * self.minc:.1f} m clear)"
                continue
            P = np.concatenate([np.column_stack([xy, np.full(len(xy), v)]) for v in levels])
            add(f"section{k+1}", P, fixed=True)
        return prims, report

    # ---- routing -------------------------------------------------------------------
    def free(self, i):
        return self.occ.clear[tuple(i)] >= self.minc

    def astar(self, a, b):
        occ = self.occ
        ia, ib = tuple(occ.idx(a)[0]), tuple(occ.idx(b)[0])
        if ia == ib:
            return np.array([a, b])
        nb = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
        cost = [math.sqrt(dx * dx + dy * dy + dz * dz) for dx, dy, dz in nb]
        clear = occ.clear; shape = occ.shape; minc = self.minc
        wh = self.cfg["safety"]["astar_weight"]                       # weighted A*: >1 trades optimality for speed
        h = lambda i: wh * math.dist(i, ib)
        g = {ia: 0.0}; came = {}; pq = [(h(ia), ia)]; seen = set()
        while pq:
            _, cur = heapq.heappop(pq)
            if cur == ib:
                break
            if cur in seen:
                continue
            seen.add(cur)
            if len(seen) > self.cfg["safety"]["astar_max_nodes"]:
                raise RuntimeError("A* exhausted")
            gc = g[cur]
            for (dx, dy, dz), w in zip(nb, cost):
                n = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                if not (0 <= n[0] < shape[0] and 0 <= n[1] < shape[1] and 0 <= n[2] < shape[2]):
                    continue
                cl = clear[n]
                if cl < minc and n != ib:
                    continue
                ng = gc + w * (1 + 2.0 / max(cl, 0.5))          # prefer roomier space
                if ng < g.get(n, 1e18):
                    g[n] = ng; came[n] = cur; heapq.heappush(pq, (ng + h(n), n))
        if ib not in came and ia != ib:
            raise RuntimeError(f"no route {a} -> {b}")
        path = [ib]
        while path[-1] != ia:
            path.append(came[path[-1]])
        P = np.array([occ.center(i) for i in path[::-1]])
        P[0], P[-1] = a, b
        return self.shortcut(P)

    def los(self, a, b):
        Q = resample(np.array([a, b]), self.occ.res / 2)
        return bool(np.all(self.occ.clearance(Q) >= self.minc))

    def shortcut(self, P):
        out = [P[0]]; i = 0
        while i < len(P) - 1:
            j = len(P) - 1
            while j > i + 1 and not self.los(P[i], P[j]):
                j -= 1
            out.append(P[j]); i = j
        return np.array(out)

    def connect(self, a, b):
        if self.los(a, b):
            return np.array([a, b])
        return self.astar(a, b)

    # ---- full plan -------------------------------------------------------------------
    def plan(self, kinds=("exterior", "sections"), start=None, home=True):
        """start=None: take off from the spawn point; otherwise start in the air at `start`.
        home: end back above the take-off point."""
        br, mo = self.br, self.cfg["motion"]
        prims, report = self.primitives(kinds)
        print(f"{len(prims)} primitive pieces", flush=True)
        takeoff = start is None
        if takeoff:
            start = br.spawn + [0, 0, self.cfg["vehicle"]["takeoff_climb"]]
            if self.occ.clearance(start)[0] < self.minc:
                raise RuntimeError("takeoff point violates clearance - raise vehicle.takeoff_climb")
            path = [resample(np.array([br.spawn + [0, 0, 0.3], start]), self.step)]
        else:
            start = np.asarray(start, float); path = [start[None]]
        tags = [np.full(len(path[0]), -1)]
        cur = start; todo = list(range(len(prims))); order = []
        while todo:                                          # greedy nearest-end ordering
            best = min([(math.dist(cur, prims[k][1][0]), k, False) for k in todo] +
                       [(math.dist(cur, prims[k][1][-1]), k, True) for k in todo if prims[k][0] not in self.fixed])
            _, k, rev = best; todo.remove(k)
            P = prims[k][1][::-1] if rev else prims[k][1]
            link = resample(self.connect(cur, P[0]), self.step)
            path.append(link); tags.append(np.full(len(link), -1))
            path.append(P); tags.append(np.full(len(P), k)); order.append(prims[k][0])
            cur = P[-1]
        if home is not None and home is not False:
            hp = br.spawn + [0, 0, self.cfg["vehicle"]["takeoff_climb"]] if home is True else np.asarray(home, float)
            link = resample(self.connect(cur, hp), self.step)
            path.append(link); tags.append(np.full(len(link), -1))
        info = {"primitives_flown": order, "dropped_fraction_per_primitive": report}
        return self.parametrize(np.concatenate(path), np.concatenate(tags), takeoff, [n for n, _ in prims], info)

    def route(self, points):
        """Collision-free path through `points` (A* where the straight line is blocked), parametrised."""
        path = [np.asarray(points[0], float)[None]]
        for a, b in zip(points, points[1:]):
            path.append(resample(self.connect(np.asarray(a, float), np.asarray(b, float)), self.step))
        P = np.concatenate(path)
        return self.parametrize(P, np.full(len(P), -1), False, [], {})

    def parametrize(self, P, T, takeoff, names, info):
        br, mo = self.br, self.cfg["motion"]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6])
        P, T = P[keep], T[keep]
        clr = self.occ.clearance(P)
        # grid quantisation: route endpoints snap from voxel centres, allow one cell of slack
        skip = int(5 / self.step) if takeoff else 0                                       # takeoff column excepted
        bad = (clr < self.minc - self.occ.res) & (np.arange(len(P)) > skip)
        if bad.any():
            raise RuntimeError(f"{bad.sum()} samples violate clearance, worst {clr[bad].min():.2f} m")

        # yaw: "travel" faces the direction of travel (smooth, straight flight; the lidar sees 360 deg anyway);
        # "look_at" faces the nearest structure (can swing a lot when the nearest structure changes side)
        from scipy.ndimage import uniform_filter1d
        if mo["yaw_mode"] == "look_at":
            d, i = br.struct_tree.query(P, distance_upper_bound=mo["look_at_radius"])
            tgt = np.where(np.isfinite(d)[:, None], br.struct_pts[np.minimum(i, len(br.struct_pts) - 1)], P + np.gradient(P, axis=0) * 50)
            yaw = np.arctan2(tgt[:, 1] - P[:, 1], tgt[:, 0] - P[:, 0])
        else:
            dxy = np.gradient(P[:, :2], axis=0)
            moving = np.linalg.norm(dxy, axis=1) > 0.3 * self.step              # horizontal motion (not a climb)
            yaw = np.full(len(P), br.spawn_yaw if takeoff else np.nan)
            yaw[moving] = np.arctan2(dxy[moving, 1], dxy[moving, 0])
            for j in range(1, len(yaw)):                                         # climbs / descents keep the last heading
                if not np.isfinite(yaw[j]):
                    yaw[j] = yaw[j - 1]
            first = np.flatnonzero(np.isfinite(yaw))
            yaw[np.isnan(yaw)] = yaw[first[0]] if len(first) else 0.0
        if takeoff:
            yaw[:int(1.0 / self.step)] = br.spawn_yaw
        yaw = uniform_filter1d(np.unwrap(yaw), max(1, int(mo["yaw_smooth_m"] / self.step)))
        # speed from clearance, turn rate, yaw rate, acceleration
        D = np.linalg.norm(np.diff(P, axis=0), axis=1); S = np.concatenate([[0], np.cumsum(D)])
        f = np.clip((clr - self.minc) / max(mo["slow_clearance"] - self.minc, 1e-3), 0, 1)
        vtop = np.where(T < 0, mo["v_transit"], mo["v_max"])          # transit between passes fast, mapping passes not
        V = mo["v_min"] + f * (vtop - mo["v_min"])
        Tg = np.gradient(P, S, axis=0); Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-9
        k = max(1, int(0.5 / self.step)); turn = np.zeros(len(P))
        turn[k:-k] = np.degrees(np.arccos(np.clip(np.sum(Tg[2 * k:] * Tg[:-2 * k], 1), -1, 1)))
        V = np.minimum(V, np.where(turn > 60, 0.3, np.where(turn > 25, 0.6, 99)))
        V = np.minimum(V, mo["v_vertical_max"] / np.maximum(np.abs(Tg[:, 2]), 1e-3))   # climbs / descents within the autopilot's limits
        dyaw = np.abs(np.gradient(yaw, S)) + 1e-6
        V = np.minimum(V, math.radians(mo["yaw_rate_max_deg"]) / dyaw)
        V[0] = V[-1] = 0
        for j in range(1, len(V)):
            V[j] = min(V[j], math.sqrt(V[j - 1] ** 2 + 2 * mo["accel"] * D[j - 1]))
        for j in range(len(V) - 2, -1, -1):
            V[j] = min(V[j], math.sqrt(V[j + 1] ** 2 + 2 * mo["accel"] * D[j]))
        V = np.maximum(V, 0.05)
        t = np.concatenate([[0], np.cumsum(D / ((V[1:] + V[:-1]) / 2))])
        zone = br.in_gps_denied_zone(P)
        info = {"length_m": round(float(S[-1]), 1), "duration_min": round(float(t[-1]) / 60, 1),
                "min_clearance_m": round(float(clr[skip:].min()), 2),
                "gps_denied_fraction": round(float(zone.mean()), 3), **info, "bridge": br.summary()}
        return dict(P=P, V=V, T=Tg, yaw=yaw, S=S, t=t, zone=zone, tag=T, clear=clr,
                    names=np.array(names if names else [""])), info        # names[tag] = primitive of a sample


def save_plan(pl, plan, info, out, name="plan", plot=False):
    np.savez_compressed(os.path.join(out, name + ".npz"), **plan)
    json.dump(info, open(os.path.join(out, name + ".json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("primitives_flown", "dropped_fraction_per_primitive", "bridge")}), flush=True)
    if plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        br, P, z = pl.br, plan["P"], plan["zone"]
        sub = br.struct_pts[np.random.default_rng(0).choice(len(br.struct_pts), min(80000, len(br.struct_pts)), replace=False)]
        fig = plt.figure(figsize=(20, 10))
        for k, (el, az) in enumerate([(25, -60), (5, 0), (89, -90)]):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            ax.scatter(*sub.T, s=0.1, c="0.6")
            ax.scatter(*P[~z][::5].T, s=0.3, c="tab:blue"); ax.scatter(*P[z][::5].T, s=0.3, c="tab:red")
            c = (sub.min(0) + sub.max(0)) / 2; r = (sub.max(0) - sub.min(0)).max() / 2 + 10
            ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r / 3, c[2] + r / 3)
            ax.view_init(el, az); ax.set_title("blue: GPS, red: GPS-denied (VIO)")
        plt.tight_layout(); plt.savefig(os.path.join(out, name + ".png"), dpi=70); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config); out = cfg["scene"]["out_dir"]; os.makedirs(out, exist_ok=True)
    if not os.path.exists(cfg["scene"]["prior_cloud"]):
        raise SystemExit(f"{cfg['scene']['prior_cloud']} not found - fly flight0 first to map the bridge")
    pl = Planner(cfg)
    plan, info = pl.plan()
    save_plan(pl, plan, info, out, plot=a.plot)


if __name__ == "__main__":
    main()

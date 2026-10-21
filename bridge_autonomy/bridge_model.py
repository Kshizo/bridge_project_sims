"""Bridge geometry MEASURED FROM A POINT CLOUD (the drone's own lidar map, e.g. pc_b2.ply).

No meshes, no part labels, no ground truth: everything the planner needs is extracted from xyz
points with generic rules (all thresholds in the config's `cloud_analysis` section):
  floor      per cell, the top of the lowest layer of returns (water / ground under a deck, the surface
             itself on a bank); layers are split by vertical gaps, sparse layers are noise
  deck       the largest connected footprint of cells whose top has >= `min_clearance_under` of
             free height to the floor; axis = PCA of that footprint
  heights    deck top / railing top from the footprint tops; superstructure underside per cell by
             walking down from the top until a vertical gap; soffit / deck bottom = percentiles
  supports   connected components of the points in the lower part of the clear height under the
             deck: small ones are columns (clustered along the axis into piers), ones spanning most
             of the deck width are abutments; pier top = lowest structure above a pier's columns
The drone spawn is read from the world SDF (that is where the operator put the drone).
"""
import math, os, xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy import ndimage
from scipy.spatial import cKDTree


def load_config(path):
    """Load a scene config. Relative paths are resolved against `root` (itself relative to the
    config file's directory; default: the project root two levels up), '~' paths are expanded."""
    path = os.path.abspath(os.path.expanduser(path))
    cfg = yaml.safe_load(open(path))
    root = os.path.normpath(os.path.join(os.path.dirname(path), cfg.get("root", "../..")))
    cfg["root"] = root

    def res(p):
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))
    for k in ("world_sdf", "parts_dir", "out_dir", "prior_cloud"):
        if k in cfg["scene"]:
            cfg["scene"][k] = res(cfg["scene"][k])
    if "mapping" in cfg and "rviz_config" in cfg["mapping"]:
        cfg["mapping"]["rviz_config"] = res(cfg["mapping"]["rviz_config"])
    cfg["sitl"]["ardupilot_dir"] = res(cfg["sitl"]["ardupilot_dir"])
    cfg["sitl"]["param_files"] = [res(p) for p in cfg["sitl"]["param_files"]]
    return cfg


def load_obj(path):
    V, F = [], []
    for l in open(path):
        t = l.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            idx = [int(x.split("/")[0]) - 1 for x in t[1:]]
            F += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.array(V)[np.array(F)] if F else np.zeros((0, 3, 3))


def read_ply(path):
    """xyz of a PLY (binary little endian or ascii, float/double vertex x y z first)."""
    with open(path, "rb") as f:
        n, fmt, props = 0, "ascii", []
        while True:
            l = f.readline().decode().strip()
            if l.startswith("format"):
                fmt = l.split()[1]
            elif l.startswith("element vertex"):
                n = int(l.split()[2])
            elif l.startswith("property"):
                props.append(l.split()[1])
            elif l == "end_header":
                break
        if fmt == "ascii":
            return np.loadtxt(f, max_rows=n)[:, :3]
        dt = np.dtype([(f"p{i}", "<f4" if t in ("float", "float32") else "<f8") for i, t in enumerate(props)])
        a = np.frombuffer(f.read(n * dt.itemsize), dt, count=n)
        return np.stack([a["p0"], a["p1"], a["p2"]], 1).astype(np.float64)


def write_ply(path, P, comment=""):
    P = np.asarray(P, np.float32)
    hdr = (f"ply\nformat binary_little_endian 1.0\ncomment {comment}\nelement vertex {len(P)}\n"
           "property float x\nproperty float y\nproperty float z\nend_header\n")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(hdr.encode()); f.write(P.tobytes())
    os.replace(tmp, path)


def voxel_down(P, size):
    if not len(P):
        return P
    k = np.floor(P / size).astype(np.int64) + (1 << 20)
    _, first = np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2], return_index=True)
    return P[first]


def spawn_pose(world_sdf, model_name):
    """(x, y, z, yaw_rad) of <include><name>model_name</name> in the world."""
    root = ET.parse(world_sdf).getroot()
    for inc in root.iter("include"):
        n = inc.find("name")
        if n is not None and n.text.strip() == model_name:
            pe = inc.find("pose")
            v = [float(x) for x in pe.text.split()]
            if pe.get("degrees") == "true":
                v[3:] = [math.radians(a) for a in v[3:]]
            return np.array(v[:3]), v[5]
    raise KeyError(f"no <include> named {model_name} in {world_sdf}")


def surroundings_path(prior_cloud):
    return os.path.splitext(prior_cloud)[0] + "_surroundings.ply"


def split_bridge(cfg, P, deck_hint=None):
    """Split a lidar map into the bridge (structure above the floor inside the measured bridge footprint
    + bridge_margin) and its surroundings (water / ground / banks / anything else). Returns
    (bridge points, surroundings points, Bridge measured from the whole map)."""
    P = np.asarray(P, np.float64)
    br = Bridge(cfg, cloud=P, deck_hint=deck_hint)
    ca = cfg["cloud_analysis"]; m = ca["bridge_margin"]
    above = P[:, 2] - br.floor_at(P[:, :2]) > ca["struct_above_floor"]
    keep = above & br._in_footprint(P, m)
    return P[keep], P[~keep], br


class FloorMap:
    """Just the floor height map of a cloud (see Bridge._build_floor): fast structure / ground split."""

    def __init__(self, cfg, P, deck_hint=None):
        Bridge._build_floor(self, np.asarray(P, np.float64), cfg["cloud_analysis"],
                            None if deck_hint is None else np.asarray(deck_hint, float))

    floor_at = lambda self, xy: Bridge.floor_at(self, xy)


class Bridge:
    """Bridge geometry from a point cloud. `cloud` defaults to cfg.scene.prior_cloud.
    `deck_hint` = a point on the deck (e.g. the take-off point) to pick the deck footprint."""

    def __init__(self, cfg, cloud=None, deck_hint=None):
        sc, ca = cfg["scene"], cfg["cloud_analysis"]
        self.cfg = cfg
        if cloud is None:                                  # the bridge cloud + its coarse surroundings (floor, banks)
            P = read_ply(sc["prior_cloud"])
            sur = surroundings_path(sc["prior_cloud"])
            if os.path.exists(sur):
                P = np.concatenate([P, read_ply(sur)])
        else:
            P = np.asarray(cloud, np.float64)
        P = P[np.all(np.isfinite(P), 1)]
        self.n_points = len(P)
        if deck_hint is None and cloud is None:            # the deck height the cloud was recorded from
            meta = os.path.splitext(sc["prior_cloud"])[0] + ".json"
            if os.path.exists(meta):
                import json
                deck_hint = json.load(open(meta)).get("deck_hint")
        deck_hint = None if deck_hint is None else np.asarray(deck_hint, float)
        self._build_floor(P, ca, deck_hint)
        h = P[:, 2] - self.floor_at(P[:, :2])
        S = P[h > ca["struct_above_floor"]]
        self._deck(S, ca, deck_hint)
        self._underside(S, ca)
        self._supports(P, ca)
        res = cfg["safety"]["grid_resolution"]
        self.struct_pts = voxel_down(S, res / 2.5)
        self.struct_tree = cKDTree(self.struct_pts)
        self.env_pts = voxel_down(P[h <= ca["struct_above_floor"]], res / 2.5)
        self.water_z = None
        self.spawn, self.spawn_yaw = spawn_pose(sc["world_sdf"], sc["vehicle_model"])

    # --- floor (ground / water) height map -----------------------------------------------
    def _build_floor(self, P, ca, deck_hint=None):
        """Floor of each cell = top of its lowest layer of returns (layers split by vertical gaps larger than
        floor_layer_gap; layers with fewer than floor_min_layer_points are noise). Under a deck that is the
        water / ground seen through the gap, on a bank it is the bank surface itself."""
        g = ca["floor_cell"]
        self._fl_lo = P[:, :2].min(0) - g
        ij = np.floor((P[:, :2] - self._fl_lo) / g).astype(int)
        shp = ij.max(0) + 2
        key = ij[:, 0] * shp[1] + ij[:, 1]
        o = np.lexsort((P[:, 2], key)); k, z = key[o], P[o, 2]
        n = len(k)
        new_cell = np.r_[True, k[1:] != k[:-1]]
        new_layer = new_cell | np.r_[False, np.diff(z) > ca["floor_layer_gap"]]
        layer_id = np.cumsum(new_layer) - 1
        l_start = np.flatnonzero(new_layer); l_end = np.r_[l_start[1:], n]
        l_cell, l_count, l_top = k[l_start], l_end - l_start, z[l_end - 1]
        ok = l_count >= ca["floor_min_layer_points"]
        # lowest supported layer per cell (layers are ordered bottom-up within a cell)
        l_cell, l_top = l_cell[ok], l_top[ok]
        first = np.r_[True, l_cell[1:] != l_cell[:-1]]
        fl = np.full(shp[0] * shp[1], np.nan); fl[l_cell[first]] = l_top[first]
        fl = fl.reshape(shp)
        if deck_hint is not None:
            # the lowest thing seen at deck height is the deck (nothing below it was seen yet), not the ground:
            # leave those cells unknown so they take the floor seen next to the bridge
            fl[fl >= deck_hint[2] - ca["deck_height_tol"]] = np.nan
        known = np.isfinite(fl)
        if not known.any():                                # nothing below the deck seen yet
            fl[:] = P[:, 2].min() - ca["min_clearance_under"]; known[:] = True
        if not known.all():
            _, (ii, jj) = ndimage.distance_transform_edt(~known, return_indices=True)
            fl = fl[ii, jj]
        # columns are solid from water to deck (one layer): take the lowest floor among the neighbours
        w = max(1, int(round(ca["floor_min_window"] / g)) | 1)
        self._fl, self._fl_g = ndimage.grey_erosion(fl, size=(w, w)), g

    def floor_at(self, xy):
        ij = np.floor((np.atleast_2d(xy) - self._fl_lo) / self._fl_g).astype(int)
        ij = np.clip(ij, 0, np.array(self._fl.shape) - 1)
        return self._fl[ij[:, 0], ij[:, 1]]

    # --- deck footprint, axis, top heights ---------------------------------------------------
    def _deck(self, S, ca, hint):
        g = ca["deck_cell"]
        lo = S[:, :2].min(0) - g
        ij = np.floor((S[:, :2] - lo) / g).astype(int); shp = ij.max(0) + 2
        top = np.full(shp, -np.inf); np.maximum.at(top, (ij[:, 0], ij[:, 1]), S[:, 2])
        I, J = np.nonzero(np.isfinite(top))
        cxy = lo + (np.stack([I, J], 1) + 0.5) * g
        cand = np.zeros(shp, bool)
        cand[I, J] = top[I, J] - self.floor_at(cxy) >= ca["min_clearance_under"]
        lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
        if n == 0:
            raise RuntimeError("no elevated structure with open space under it in the cloud")
        sizes = ndimage.sum(cand, lab, range(1, n + 1)) * g * g
        meds = np.array([np.median(top[lab == k + 1]) for k in range(n)])
        if hint is not None:                                                 # deck height = the hint's height
            ref = float(hint[2])
        else:
            ref = meds[np.argmax(sizes)]
        keep = [k + 1 for k in range(n) if sizes[k] >= ca["deck_min_area"] and abs(meds[k] - ref) <= ca["deck_height_tol"]]
        if not keep:
            raise RuntimeError("no deck-sized footprint found near the expected deck height")
        m = np.isin(lab, keep)
        I, J = np.nonzero(m)
        C = lo + (np.stack([I, J], 1) + 0.5) * g
        self.origin = C.mean(0)
        w, V = np.linalg.eigh(np.cov((C - self.origin).T))
        self.u = V[:, np.argmax(w)]
        if self.u[np.argmax(np.abs(self.u))] < 0:
            self.u = -self.u
        self.v = np.array([-self.u[1], self.u[0]])
        s, c = self.to_sc(C).T
        self.s_range = (s.min() - g / 2, s.max() + g / 2)
        # width: median edges of 1 m slices along the axis, so wider things at the ends (abutments) don't count
        sb = np.floor(s).astype(int); lo_e, hi_e = [], []
        for b in np.unique(sb):
            cc = c[sb == b]
            if len(cc) >= 3:
                lo_e.append(cc.min()); hi_e.append(cc.max())
        self.c_range = (float(np.median(lo_e)) - g / 2, float(np.median(hi_e)) + g / 2)
        self.c_mid = sum(self.c_range) / 2
        self.half_width = (self.c_range[1] - self.c_range[0]) / 2
        tops = top[I, J]
        self.deck_top = float(np.median(tops))
        inside = self._in_footprint(S, 0.0)
        self.railing_top = float(np.percentile(S[inside, 2], 99.5)) if inside.any() else self.deck_top
        self._deck_cells = (lo, g, I, J, tops)

    def _in_footprint(self, P, margin):
        s, c = self.to_sc(P[:, :2]).T
        return ((s >= self.s_range[0] - margin) & (s <= self.s_range[1] + margin) &
                (c >= self.c_range[0] - margin) & (c <= self.c_range[1] + margin))

    # --- underside of the superstructure ------------------------------------------------------
    def _underside(self, S, ca):
        lo, g, I, J, tops = self._deck_cells
        ij = np.floor((S[:, :2] - lo) / g).astype(int)
        key_cells = I * 100000 + J
        k = ij[:, 0] * 100000 + ij[:, 1]
        sel = np.isin(k, key_cells)
        k, z = k[sel], S[sel, 2]
        o = np.lexsort((-z, k)); k, z = k[o], z[o]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]]); ends = np.r_[starts[1:], len(k)]
        bottoms, cells = [], []
        for a, b in zip(starts, ends):
            zz = z[a:b]
            gap = np.flatnonzero(-np.diff(zz) > ca["underside_gap"])
            bottoms.append(zz[gap[0]] if len(gap) else zz[-1]); cells.append(k[a])
        bottoms, cells = np.array(bottoms), np.array(cells)
        cxy = lo + (np.stack([cells // 100000, cells % 100000], 1) + 0.5) * g
        span = bottoms > self.floor_at(cxy) + ca["min_clearance_under"] / 2
        self.has_underside = bool(span.any() and self.deck_top - np.median(bottoms[span]) > ca["underside_gap"])
        if self.has_underside:
            self.soffit = float(np.percentile(bottoms[span], ca["soffit_percentile"]))
            self.deck_bottom = float(np.median(bottoms[span]))
        else:                                                                # only seen from above so far
            self.soffit = self.deck_bottom = self.deck_top

    # --- piers + abutments -----------------------------------------------------------------------
    def _supports(self, P, ca):
        """Supports = solid things standing in the lower part of the clear height under the deck.
        The band is absolute: from the deck's water / ground level (median floor under the spanning part of
        the deck) up to band_frac of the clear height. Components on a fine grid; a wide component that is
        not solid across its width is a row of columns and is split; the outermost wide solid wall at each
        end near the deck end is an abutment; everything else up to pier_max_size is a pier (column or wall)."""
        self.piers, self.abutments = [], []
        if not self.has_underside:
            return
        lo_d, gd, I, J, _ = self._deck_cells
        C = lo_d + (np.stack([I, J], 1) + 0.5) * gd
        fl = self.floor_at(C)
        open_ = fl < self.soffit - ca["min_clearance_under"] / 2
        if not open_.any():
            return
        self.base_z = base = float(np.median(fl[open_]))
        z0, z1 = base + ca["band_low"], base + ca["band_frac"] * (self.soffit - base)
        B = P[self._in_footprint(P, ca["support_search_margin"]) & (P[:, 2] >= z0) & (P[:, 2] <= z1)]
        if not len(B):
            return
        g = ca["support_cell"]
        sc = self.to_sc(B[:, :2]); lo = sc.min(0) - g
        ij = np.floor((sc - lo) / g).astype(int); shp = ij.max(0) + 2
        cnt = np.zeros(shp, int); np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
        # real surfaces are dense; smear between them (multi-pass pose differences) is orders of magnitude
        # sparser, so the threshold is relative to the typical surface density
        thr = max(ca["support_min_points"], ca["support_density_frac"] * np.percentile(cnt[cnt > 0], 90))
        dense = cnt >= thr
        lab, n = ndimage.label(ndimage.binary_dilation(dense), structure=np.ones((3, 3)))   # grouping only
        lab = np.where(dense, lab, 0)                                                         # extents: dense cells
        width = self.c_range[1] - self.c_range[0]
        pid = lab[ij[:, 0], ij[:, 1]]
        cands = []                                                 # (cells I, cells J) per candidate
        for k in range(1, n + 1):
            Ik, Jk = np.nonzero(lab == k)
            span = Jk.max() - Jk.min() + 1
            fill = len(np.unique(Jk)) / span
            if (span * g >= ca["abutment_min_width_frac"] * width and fill < ca["wall_min_fill"]):
                # a row of columns: split where the projection across the bridge has gaps
                js = np.unique(Jk); runs = np.split(js, np.flatnonzero(np.diff(js) > 1) + 1)
                for r in runs:
                    m = np.isin(Jk, r); cands.append((Ik[m], Jk[m], k))
            else:
                cands.append((Ik, Jk, k))
        sups = []
        for Ik, Jk, k in cands:
            s0, s1 = lo[0] + Ik.min() * g, lo[0] + (Ik.max() + 1) * g
            c0, c1 = lo[1] + Jk.min() * g, lo[1] + (Jk.max() + 1) * g
            inb = (pid == k) & (sc[:, 1] >= c0) & (sc[:, 1] <= c1)
            pts = B[inb]
            if not len(pts) or np.ptp(pts[:, 2]) < ca["support_min_height_frac"] * (z1 - z0):
                continue                                           # not standing through the band
            sups.append({"s": (s0 + s1) / 2, "c": (c0 + c1) / 2, "hs": (s1 - s0) / 2, "hc": (c1 - c0) / 2,
                         "zmin": float(pts[:, 2].min()), "zmax": float(pts[:, 2].max()),
                         "wide": (c1 - c0) >= ca["abutment_min_width_frac"] * width})
        # abutments: the outermost wide wall at each end, near that end of the deck
        for d, end in ((1, self.s_range[0]), (-1, self.s_range[1])):
            wide = [q for q in sups if q["wide"] and abs(q["s"] - end) <= ca["abutment_end_margin"]]
            if wide:
                q = min(wide, key=lambda q: abs(q["s"] - end))
                face = q["s"] + d * q["hs"]
                self.abutments.append({"face_s": float(face), "dir": float(d)})
                sups = [r for r in sups if (r["s"] - face) * d > 0]   # everything behind it is abutment / embankment
        self.abutments = [dict(a, name=f"abutment_{i+1:02d}") for i, a in enumerate(sorted(self.abutments, key=lambda q: q["face_s"]))]
        cols = sorted([dict(q, name=f"support_{i+1:02d}") for i, q in enumerate(sups)
                       if ca["pier_min_size"] <= 2 * min(q["hs"], q["hc"]) and 2 * max(q["hs"], q["hc"]) <= ca["pier_max_size"]],
                      key=lambda q: q["s"])
        for col in cols:
            if self.piers and col["s"] - self.piers[-1]["cols"][-1]["s"] < self.cfg["survey"]["pier_cluster_gap"]:
                self.piers[-1]["cols"].append(col)
            else:
                self.piers.append({"cols": [col]})
        ssc = self.to_sc(P[:, :2])
        for pr in self.piers:
            cs = pr["cols"]
            pr["s"] = float(np.mean([q["s"] for q in cs]))
            pr["s_lo"] = min(q["s"] - q["hs"] for q in cs); pr["s_hi"] = max(q["s"] + q["hs"] for q in cs)
            pr["c_lo"] = min(q["c"] - q["hc"] for q in cs); pr["c_hi"] = max(q["c"] + q["hc"] for q in cs)
            pr["c"] = (pr["c_lo"] + pr["c_hi"]) / 2
            pr["zmin"] = min(q["zmin"] for q in cs)
            pr["cols"] = sorted(cs, key=lambda q: q["c"])
            # pier top: lowest structure above the band over / around the pier that is not the pier itself
            near = ((ssc[:, 0] >= pr["s_lo"] - 0.5) & (ssc[:, 0] <= pr["s_hi"] + 0.5) &
                    (ssc[:, 1] >= pr["c_lo"] - 1.0) & (ssc[:, 1] <= pr["c_hi"] + 1.0) & (P[:, 2] > z1))
            on_col = np.zeros(len(P), bool)
            for q in cs:
                on_col |= ((np.abs(ssc[:, 0] - q["s"]) <= q["hs"] + 0.15) & (np.abs(ssc[:, 1] - q["c"]) <= q["hc"] + 0.15))
            above = P[near & ~on_col, 2]
            pr["ztop"] = float(np.percentile(above, 2)) if len(above) else self.soffit
        sup = sorted([a["face_s"] for a in self.abutments] + [p["s"] for p in self.piers])
        self.supports = sup if sup else list(self.s_range)
        self.midspans = [(a + b) / 2 for a, b in zip(self.supports, self.supports[1:])]

    # --- frames ------------------------------------------------------------------------
    def to_sc(self, xy):
        d = np.atleast_2d(xy) - self.origin
        return np.stack([d @ self.u, d @ self.v], 1)

    def to_xy(self, s, c):
        return self.origin + np.multiply.outer(s, self.u) + np.multiply.outer(c, self.v)

    # --- predicted GPS-denied zone (from the measured deck) ----------------------------------
    def in_gps_denied_zone(self, P):
        g = self.cfg["gps_denied"]
        P = np.atleast_2d(P)
        s, c = self.to_sc(P[:, :2]).T
        m = g["footprint_margin"]
        return ((s >= self.s_range[0] - m) & (s <= self.s_range[1] + m) &
                (c >= self.c_range[0] - m) & (c <= self.c_range[1] + m) &
                (P[:, 2] < self.deck_bottom + g["top_offset"]))

    def summary(self):
        return {"points": self.n_points, "axis": self.u.round(3).tolist(), "s_range": np.round(self.s_range, 2).tolist(),
                "deck_width": round(2 * self.half_width, 2), "deck_top": round(self.deck_top, 2),
                "soffit": round(self.soffit, 2), "deck_bottom": round(self.deck_bottom, 2),
                "railing_top": round(self.railing_top, 2), "underside_seen": self.has_underside,
                "piers": [{"s": round(p["s"], 2), "columns": len(p["cols"]), "top": round(p["ztop"], 2),
                           "c": [round(p["c_lo"], 2), round(p["c_hi"], 2)]} for p in self.piers],
                "abutment_faces_s": [round(a["face_s"], 2) for a in self.abutments],
                "spawn": self.spawn.round(2).tolist()}


class Occupancy:
    """Voxel occupancy (every measured point + everything below the floor) and clearance field."""

    def __init__(self, br, extra_bounds=None):
        cfg = br.cfg; self.res = r = cfg["safety"]["grid_resolution"]; mg = cfg["safety"]["grid_margin"]
        P = np.concatenate([br.struct_pts, br.env_pts]) if len(br.env_pts) else br.struct_pts
        lo = br.struct_pts.min(0) - mg; hi = br.struct_pts.max(0) + mg
        if extra_bounds is not None:
            lo = np.minimum(lo, np.min(extra_bounds, 0) - 5); hi = np.maximum(hi, np.max(extra_bounds, 0) + 5)
        lo[2] = min(br.struct_pts[:, 2].min(), P[:, 2].min()) - 2.0; hi[2] = max(br.railing_top + mg / 2, hi[2])
        self.lo, self.shape = lo, np.ceil((hi - lo) / r).astype(int) + 1
        occ = np.zeros(self.shape, bool)
        idx = np.floor((P - lo) / r).astype(int)
        ok = np.all((idx >= 0) & (idx < self.shape), axis=1)
        occ[tuple(idx[ok].T)] = True
        # everything at or below the floor (terrain / water surface) is solid
        gx = lo[0] + r * (np.arange(self.shape[0]) + 0.5); gy = lo[1] + r * (np.arange(self.shape[1]) + 0.5)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        fl = br.floor_at(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
        fl[~np.isfinite(fl)] = lo[2]
        kz = np.clip(np.floor((fl - lo[2]) / r).astype(int), -1, self.shape[2] - 1)
        zz = np.arange(self.shape[2])[None, None, :]
        occ |= zz <= kz[:, :, None]
        self.occ = occ
        self.clear = ndimage.distance_transform_edt(~occ) * r
        self.floor = fl

    def idx(self, P):
        return np.clip(np.floor((np.atleast_2d(P) - self.lo) / self.res).astype(int), 0, self.shape - 1)

    def clearance(self, P):
        P = np.atleast_2d(P)
        i = np.floor((P - self.lo) / self.res).astype(int)
        inside = np.all((i >= 0) & (i < self.shape), axis=1)
        out = np.full(len(P), 99.0)
        out[inside] = self.clear[tuple(i[inside].T)]
        return out

    def center(self, i):
        return self.lo + (np.asarray(i) + 0.5) * self.res

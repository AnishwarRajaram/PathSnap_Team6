"""Observation-to-route-arc alignment that produces training labels.

The reference route lists arcs in driving order but does not say which GPS
observation lies on which arc. Two alignment methods are provided.

A. nearest: each observation takes the nearest arc among the route arcs.
   Order is ignored.
B. ordered: dynamic programming over route positions. The route position may
   stay or move forward between consecutive observations but never moves back.
   The cost is the truncated squared distance from observation to arc.
   This resolves out-and-back trips, loops and repeated arcs.

Both return a route position per observation. The arc label is route[pos].
"""
from __future__ import annotations

import numpy as np

from pathsnap.data import Trip

RELIABLE_M = 50.0     # labels farther than this from the GPS point are unreliable
TRUNC_M = 100.0       # distance cap inside the DP cost
CHUNK = 1024


def route_segments(trip: Trip) -> np.ndarray:
    """(R, 4) array of x0, y0, x1, y1 in local metres for each route position."""
    nxy = trip.nodes_xy()
    a = trip.arcs.iloc[trip.route]
    return np.hstack([nxy[a.u.to_numpy()], nxy[a.v.to_numpy()]]).astype(np.float64)


def point_seg_dist(p: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Distances (P, S) from points p (P, 2) to segments seg (S, 4)."""
    ax, ay, bx, by = seg.T
    dx, dy = bx - ax, by - ay
    l2 = np.where(dx * dx + dy * dy == 0, 1e-12, dx * dx + dy * dy)
    px, py = p[:, :1], p[:, 1:2]
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _dist_rows(pts: np.ndarray, seg: np.ndarray):
    for s in range(0, len(pts), CHUNK):
        yield s, point_seg_dist(pts[s:s + CHUNK], seg).astype(np.float32)


def align_nearest(trip: Trip) -> tuple[np.ndarray, np.ndarray]:
    pts, seg = trip.track_xy(), route_segments(trip)
    pos = np.empty(len(pts), np.int32)
    dist = np.empty(len(pts), np.float32)
    for s, d in _dist_rows(pts, seg):
        pos[s:s + len(d)] = d.argmin(1)
        dist[s:s + len(d)] = d.min(1)
    return pos, dist


def align_ordered(trip: Trip) -> tuple[np.ndarray, np.ndarray]:
    """Monotone alignment: pos[i] <= pos[i+1]. Minimises sum of min(d, TRUNC)^2."""
    pts, seg = trip.track_xy(), route_segments(trip)
    n, r = len(pts), len(seg)
    back = np.empty((n, r), np.int16 if r < 32767 else np.int32)
    dist_all = np.empty((n, r), np.float32) if n * r <= 60_000_000 else None
    prev = None
    idx = np.arange(r)
    for s, d in _dist_rows(pts, seg):
        if dist_all is not None:
            dist_all[s:s + len(d)] = d
        cost = np.minimum(d, TRUNC_M) ** 2
        for k in range(len(d)):
            i = s + k
            if prev is None:
                cur = cost[k].astype(np.float64)
                back[i] = idx
            else:
                # best predecessor among positions <= j: running prefix minimum.
                pm = np.minimum.accumulate(prev)
                arg = np.where(prev == pm, idx, 0)
                arg = np.maximum.accumulate(arg)
                cur = pm + cost[k]
                back[i] = arg
            prev = cur
    pos = np.empty(n, np.int32)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    if dist_all is not None:
        dist = dist_all[np.arange(n), pos]
    else:
        dist = np.concatenate([d[np.arange(len(d)), pos[s:s + len(d)]] for s, d in _dist_rows(pts, seg)])
    return pos, dist


def canonical_arc(trip: Trip) -> np.ndarray:
    """Map each arc_id to the smallest arc_id with the same (u, v). Duplicates share a key."""
    a = trip.arcs.reset_index()
    return a.groupby(["u", "v"]).arc_id.transform("min").to_numpy()

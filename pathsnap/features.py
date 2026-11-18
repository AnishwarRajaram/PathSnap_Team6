"""Feature engineering for (observation, candidate arc) pairs.

Every feature is place-independent: it describes geometry, motion or local
network shape, never an arc identifier or a location. The same code runs at
training time and inside the API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.candidates import ArcIndex, build_index, generate
from pathsnap.data import Trip

HEADING_K = 3        # half-window (observations) for smoothed heading and speed
NEIGHBOUR_K = 2      # half-window for neighbour distance to the same arc
GAP_S = 30.0         # windows do not reach across gaps longer than this

FEATURES = [
    "dist_m", "rank", "n_cand", "dist_minus_best", "dist_ratio_best", "fallback",
    "proj_frac", "dist_to_end_m", "arc_len_m",
    "heading_diff_deg", "heading_diff_undirected_deg", "speed_mps", "turn_rate_dps",
    "dt_prev_s", "dt_next_s",
    "nbr_mean_dist_m", "nbr_max_dist_m",
    "arc_oneway", "end_degree_max", "end_degree_min",
]

# Locked model input after step 6 (eda/06_features.py). Grouped 5-fold CV on train
# gave the same Top-1, macro F1 and log loss as all 20 features.
# Dropped: fallback, dt_prev_s, dt_next_s (no signal); nbr_max_dist_m (|rho| 0.93 with
# nbr_mean_dist_m); heading_diff_undirected_deg (covered by heading_diff_deg);
# end_degree_max (weak, end_degree_min kept).
SELECTED = [
    "dist_m", "rank", "n_cand", "dist_minus_best", "dist_ratio_best",
    "proj_frac", "dist_to_end_m", "arc_len_m",
    "heading_diff_deg", "speed_mps", "turn_rate_dps",
    "nbr_mean_dist_m", "arc_oneway", "end_degree_min",
]


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def _segment_ids(t: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.diff(t) > GAP_S)])


def _window_index(n: int, seg: np.ndarray, k: int, offset: int) -> np.ndarray:
    """Index i+offset clipped to [0, n) and to the same gap-free segment as i."""
    i = np.arange(n)
    j = np.clip(i + offset, 0, n - 1)
    return np.where(seg[j] == seg, j, i)


def motion(trip: Trip):
    """Smoothed heading (deg, 0 = north), speed (m/s), turn rate (deg/s) per observation."""
    xy, t = trip.track_xy(), trip.track.t.to_numpy()
    n, seg = len(t), _segment_ids(t)
    a = _window_index(n, seg, HEADING_K, -HEADING_K)
    b = _window_index(n, seg, HEADING_K, HEADING_K)
    d = xy[b] - xy[a]
    span = np.maximum(t[b] - t[a], 1e-6)
    heading = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
    speed = np.hypot(d[:, 0], d[:, 1]) / span
    h_prev = heading[_window_index(n, seg, 1, -1)]
    h_next = heading[_window_index(n, seg, 1, 1)]
    turn = np.abs(_wrap180(h_next - h_prev)) / np.maximum(
        t[_window_index(n, seg, 1, 1)] - t[_window_index(n, seg, 1, -1)], 1.0)
    dt = np.diff(t)
    dt_prev = np.concatenate([[0.0], dt]); dt_next = np.concatenate([dt, [0.0]])
    return heading, speed, turn, dt_prev, dt_next, seg


def _point_seg(px, py, s):
    dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
    l2 = np.where(dx * dx + dy * dy == 0, 1e-12, dx * dx + dy * dy)
    t = np.clip(((px - s[:, 0]) * dx + (py - s[:, 1]) * dy) / l2, 0, 1)
    return np.hypot(px - (s[:, 0] + t * dx), py - (s[:, 1] + t * dy)), t


def build(trip: Trip, index: ArcIndex | None = None) -> pd.DataFrame:
    """One row per (observation, candidate arc) with FEATURES plus obs_idx and arc_id."""
    index = index or build_index(trip)
    pts = trip.track_xy()
    n = len(pts)
    obs, arc, dist, rank, fb = generate(index, pts)

    nxy = trip.nodes_xy()
    u = trip.arcs.u.to_numpy()[arc]; v = trip.arcs.v.to_numpy()[arc]
    s = np.hstack([nxy[u], nxy[v]])
    length = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    _, frac = _point_seg(pts[obs, 0], pts[obs, 1], s)
    bearing = np.degrees(np.arctan2(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]))

    heading, speed, turn, dt_prev, dt_next, seg = motion(trip)
    hd = np.abs(_wrap180(heading[obs] - bearing))

    # Distances of neighbouring observations to the same arc.
    nd = []
    for off in range(-NEIGHBOUR_K, NEIGHBOUR_K + 1):
        if off == 0:
            continue
        j = _window_index(n, seg, NEIGHBOUR_K, off)[obs]
        nd.append(_point_seg(pts[j, 0], pts[j, 1], s)[0])
    nd = np.vstack(nd)

    # Network context: one-way flag and undirected degree at both ends.
    au, av = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    pairs = pd.MultiIndex.from_arrays([au, av])
    oneway = ~pd.MultiIndex.from_arrays([v, u]).isin(pairs)
    und = pd.DataFrame({"a": np.minimum(au, av), "b": np.maximum(au, av)})
    und = und[und.a != und.b].drop_duplicates()
    deg = np.bincount(np.concatenate([und.a, und.b]), minlength=len(nxy))

    n_cand = np.bincount(obs, minlength=n)[obs]
    best = dist[np.searchsorted(obs, obs)]  # rows are sorted by obs then dist
    df = pd.DataFrame({
        "obs_idx": obs, "arc_id": arc,
        "dist_m": dist, "rank": rank, "n_cand": n_cand,
        "dist_minus_best": dist - best, "dist_ratio_best": (dist + 1.0) / (best + 1.0),
        "fallback": fb,
        "proj_frac": frac, "dist_to_end_m": np.minimum(frac, 1 - frac) * length, "arc_len_m": length,
        "heading_diff_deg": hd, "heading_diff_undirected_deg": np.minimum(hd, 180 - hd),
        "speed_mps": speed[obs], "turn_rate_dps": turn[obs],
        "dt_prev_s": dt_prev[obs], "dt_next_s": dt_next[obs],
        "nbr_mean_dist_m": nd.mean(0), "nbr_max_dist_m": nd.max(0),
        "arc_oneway": oneway.astype(np.int8),
        "end_degree_max": np.maximum(deg[u], deg[v]), "end_degree_min": np.minimum(deg[u], deg[v]),
    })
    return df

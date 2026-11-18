"""Candidate road-segment generation.

Each arc is densified into sample points no more than STEP_M apart and placed
in a KD-tree. A radius query on the tree returns every arc that may lie within
the radius. Exact point-to-segment distances then filter the result.
Self-loop arcs are dropped. Duplicate arcs (same u, v) are collapsed to their
canonical arc so each geometry appears once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from pathsnap.data import Trip
from pathsnap.labels import canonical_arc

STEP_M = 5.0

# Candidate rule chosen from the step 5 sweep on train (see eda/05_candidate_sweep.py).
RADIUS_M = 50.0     # take every arc within this distance
CAP_K = 16          # keep at most this many nearest arcs
FALLBACK_K = 4      # if none within RADIUS_M, take this many nearest arcs
FALLBACK_MAX_M = 500.0


@dataclass
class ArcIndex:
    seg: np.ndarray      # (A, 4) x0, y0, x1, y1 for each kept arc
    arc_id: np.ndarray   # (A,) arc ids of kept arcs (canonical only)
    tree: cKDTree
    owner: np.ndarray    # (S,) row in seg for each sample point


def build_index(trip: Trip) -> ArcIndex:
    nxy = trip.nodes_xy()
    u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    canon = canonical_arc(trip)
    keep = (u != v) & (canon == np.arange(len(u)))
    ids = np.flatnonzero(keep)
    seg = np.hstack([nxy[u[ids]], nxy[v[ids]]])
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    nseg = np.maximum(1, np.ceil(length / STEP_M).astype(int))
    owner = np.repeat(np.arange(len(ids)), nseg + 1)
    frac = np.concatenate([np.linspace(0, 1, k + 1) for k in nseg])
    pts = seg[owner, :2] + frac[:, None] * (seg[owner, 2:] - seg[owner, :2])
    return ArcIndex(seg, ids, cKDTree(pts), owner)


def query(index: ArcIndex, pts: np.ndarray, radius: float):
    """Return (obs_idx, arc_id, dist_m) for every arc within radius of each point."""
    hits = index.tree.query_ball_point(pts, r=radius + STEP_M / 2)
    obs, rows = [], []
    for i, h in enumerate(hits):
        if h:
            r = np.unique(index.owner[h])
            obs.append(np.full(len(r), i)); rows.append(r)
    if not obs:
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    obs = np.concatenate(obs); rows = np.concatenate(rows)
    s = index.seg[rows]
    p = pts[obs]
    dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
    l2 = np.where(dx * dx + dy * dy == 0, 1e-12, dx * dx + dy * dy)
    t = np.clip(((p[:, 0] - s[:, 0]) * dx + (p[:, 1] - s[:, 1]) * dy) / l2, 0, 1)
    d = np.hypot(p[:, 0] - (s[:, 0] + t * dx), p[:, 1] - (s[:, 1] + t * dy))
    m = d <= radius
    return obs[m], index.arc_id[rows[m]], d[m]


def generate(index: ArcIndex, pts: np.ndarray):
    """Apply the locked candidate rule.

    Returns (obs_idx, arc_id, dist_m, rank, fallback) sorted by observation then distance.
    fallback is 1 for observations that had no arc within RADIUS_M.
    """
    n = len(pts)
    obs, arc, d = query(index, pts, RADIUS_M)
    have = np.zeros(n, bool); have[obs] = True
    miss = np.flatnonzero(~have)
    fb = np.zeros(len(obs), np.int8)
    if len(miss):
        k = min(FALLBACK_K * 8, len(index.owner))
        dd, ii = index.tree.query(pts[miss], k=k, distance_upper_bound=FALLBACK_MAX_M)
        o2, a2, d2 = [], [], []
        for j, i in enumerate(miss):
            ok = np.isfinite(dd[j])
            rows = pd_unique(index.owner[ii[j][ok]])[:FALLBACK_K]
            if len(rows) == 0:
                continue
            s = index.seg[rows]
            dist = _seg_dist(pts[i], s)
            o2.append(np.full(len(rows), i)); a2.append(index.arc_id[rows]); d2.append(dist)
        if o2:
            obs = np.concatenate([obs, *o2]); arc = np.concatenate([arc, *a2]); d = np.concatenate([d, *d2])
            fb = np.concatenate([fb, np.ones(sum(len(x) for x in o2), np.int8)])
    order = np.lexsort((d, obs))
    obs, arc, d, fb = obs[order], arc[order], d[order], fb[order]
    first = np.searchsorted(obs, np.arange(n))
    rank = np.arange(len(obs)) - first[obs]
    keep = rank < CAP_K
    return obs[keep], arc[keep], d[keep], rank[keep], fb[keep]


def pd_unique(a: np.ndarray) -> np.ndarray:
    """Unique values in first-seen order."""
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


def _seg_dist(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
    l2 = np.where(dx * dx + dy * dy == 0, 1e-12, dx * dx + dy * dy)
    t = np.clip(((p[0] - s[:, 0]) * dx + (p[1] - s[:, 1]) * dy) / l2, 0, 1)
    return np.hypot(p[0] - (s[:, 0] + t * dx), p[1] - (s[:, 1] + t * dy))

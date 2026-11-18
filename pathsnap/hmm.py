"""HMM and Viterbi route reconstruction.

Hidden states   candidate arcs of each observation (contract v1.1, section 7).
Emission        alpha * log of the classifier probability normalised over the
                observation's candidates. A distance-only variant uses a Gaussian
                on the point-to-arc distance instead.
Transition      -|road distance - GPS distance| / beta, the standard map-matching
                form. Road distance follows the directed road graph.
Chain breaks    when no candidate of step i reaches any candidate of step i+1,
                the chain ends and a new chain starts. With jump_penalty set,
                unreachable transitions instead receive a large finite penalty.
Route output    the per-observation arc sequence, plus a drivable route where
                consecutive distinct arcs are joined by shortest paths.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pathsnap.data import Trip

BACK_TOL_M = 10.0        # backward movement allowed on the same arc (GPS noise)
MIN_LIMIT_M = 200.0      # minimum search radius for road distance
MAX_SPEED_MPS = 50.0     # search radius grows with dt at this speed
MAX_LIMIT_M = 5000.0
P_FLOOR = 1e-6


@dataclass
class DecoderParams:
    alpha: float = 1.0           # weight of the classifier emission
    beta: float = 10.0           # transition scale in metres
    emission: str = "classifier" # "classifier" or "distance"
    sigma: float = 7.0           # Gaussian sigma for the distance-only emission
    jump_penalty: float | None = None  # log-score for unreachable transitions; None = hard break


@dataclass
class RoadGraph:
    """Directed graph with bounded Dijkstra and a cache keyed by source node."""
    adj_ptr: np.ndarray
    adj_to: np.ndarray
    adj_w: np.ndarray
    arc_of: dict
    cache: dict = field(default_factory=dict)

    @classmethod
    def from_trip(cls, trip: Trip) -> "RoadGraph":
        nxy = trip.nodes_xy()
        u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
        keep = u != v
        u, v, ids = u[keep], v[keep], np.flatnonzero(keep)
        w = np.hypot(*(nxy[v] - nxy[u]).T)
        order = np.argsort(u, kind="stable")
        u, v, w, ids = u[order], v[order], w[order], ids[order]
        ptr = np.searchsorted(u, np.arange(len(nxy) + 1))
        arc_of = {}
        for a, b, i in zip(u.tolist(), v.tolist(), ids.tolist()):
            arc_of.setdefault((a, b), i)   # smallest id first, matches canonical arcs
        return cls(ptr, v, w, arc_of)

    def dist(self, src: int, limit: float) -> tuple[dict, dict]:
        hit = self.cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1], hit[2]
        d, pred = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            du, n = heapq.heappop(heap)
            if du > d.get(n, np.inf) or du > limit:
                continue
            for k in range(self.adj_ptr[n], self.adj_ptr[n + 1]):
                m, nd = int(self.adj_to[k]), du + float(self.adj_w[k])
                if nd <= limit and nd < d.get(m, np.inf):
                    d[m] = nd; pred[m] = n
                    heapq.heappush(heap, (nd, m))
        self.cache[src] = (limit, d, pred)
        return d, pred

    def path_arcs(self, src: int, dst: int, limit: float) -> list[int] | None:
        """Arc ids along the shortest path from node src to node dst, or None."""
        if src == dst:
            return []
        d, pred = self.dist(src, limit)
        if dst not in d:
            return None
        nodes = [dst]
        while nodes[-1] != src:
            nodes.append(pred[nodes[-1]])
        nodes.reverse()
        return [self.arc_of[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]


def _emission(g: pd.DataFrame, prm: DecoderParams) -> np.ndarray:
    if prm.emission == "distance":
        return -0.5 * (g.dist_m.to_numpy() / prm.sigma) ** 2
    p = np.maximum(g.p.to_numpy(), P_FLOOR)
    return prm.alpha * np.log(p / p.sum())


def decode_trip(trip: Trip, cand: pd.DataFrame, prm: DecoderParams, graph: RoadGraph | None = None):
    """Viterbi over one trip.

    cand: rows with obs_idx, arc_id, proj_frac, arc_len_m, dist_m and p (for classifier emission).
    Returns (per-observation arc array with -1 where undecided, number of chains, filled route arcs).
    """
    graph = graph or RoadGraph.from_trip(trip)
    xy = trip.track_xy()
    t = trip.track.t.to_numpy()
    u_all, v_all = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    n = len(xy)
    groups = {o: g for o, g in cand.groupby("obs_idx", sort=True)}
    obs_list = sorted(groups)
    chosen = np.full(n, -1, np.int64)
    chains = 0

    def backtrack(steps):
        k = int(np.argmax(steps[-1][1]))
        for obs, delta, back, arcs in reversed(steps):
            chosen[obs] = arcs[k]
            k = back[k] if back is not None else k

    steps = []   # (obs, delta, backpointer into previous step, arc ids)
    prev = None
    for obs in obs_list:
        g = groups[obs]
        arcs = g.arc_id.to_numpy(); frac = g.proj_frac.to_numpy(); alen = g.arc_len_m.to_numpy()
        em = _emission(g, prm)
        if prev is None:
            steps = [(obs, em, None, arcs)]; chains += 1
            prev = (obs, arcs, frac, alen, em)
            continue
        po, parcs, pfrac, palen, pdelta = prev[0], prev[1], prev[2], prev[3], steps[-1][1]
        gps = float(np.hypot(*(xy[obs] - xy[po])))
        limit = min(MAX_LIMIT_M, max(MIN_LIMIT_M, (t[obs] - t[po]) * MAX_SPEED_MPS, 3 * gps))
        trans = np.full((len(parcs), len(arcs)), -np.inf)
        for i, a in enumerate(parcs):
            d, _ = graph.dist(int(v_all[a]), limit)
            rest = (1 - pfrac[i]) * palen[i]
            for j, b in enumerate(arcs):
                if a == b:
                    road = (frac[j] - pfrac[i]) * alen[j]
                    if road < -BACK_TOL_M:
                        continue
                    road = max(road, 0.0)
                else:
                    sp = d.get(int(u_all[b]))
                    if sp is None:
                        continue
                    road = rest + sp + frac[j] * alen[j]
                trans[i, j] = -abs(road - gps) / prm.beta
        if prm.jump_penalty is not None:
            trans = np.where(np.isfinite(trans), trans, -prm.jump_penalty - gps / prm.beta)
        score = pdelta[:, None] + trans
        back = np.argmax(score, axis=0)
        best = score[back, np.arange(len(arcs))]
        if not np.isfinite(best).any():
            backtrack(steps)
            steps = [(obs, em, None, arcs)]; chains += 1
        else:
            delta = best + em
            delta = delta - np.max(delta[np.isfinite(delta)])
            steps.append((obs, delta, back, arcs))
        prev = (obs, arcs, frac, alen, em)
    if steps:
        backtrack(steps)

    # Drivable route: join consecutive distinct arcs by shortest paths.
    seq = chosen[chosen >= 0]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    route = []
    for i, a in enumerate(seq):
        if i and route:
            gap = None
            for lim in (MIN_LIMIT_M, 1000.0, MAX_LIMIT_M):
                gap = graph.path_arcs(int(v_all[route[-1]]), int(u_all[a]), lim)
                if gap is not None:
                    break
            if gap:
                route.extend(gap)
        route.append(int(a))
    return chosen, chains, route

"""End-to-end prediction for one trip: candidates, features, classifier, HMM.

Used by the API and by the offline check. The prediction path reads only the
track, nodes and arcs. Scoring against the reference route is a separate
function and runs on the server side only (contract v1.1, section 8.3).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pathsnap.candidates import build_index
from pathsnap.data import Trip
from pathsnap.features import build
from pathsnap.hmm import DecoderParams, RoadGraph, decode_trip
from pathsnap.labels import RELIABLE_M, align_ordered, canonical_arc

THRESHOLD = 0.5


@dataclass
class Model:
    estimator: object
    features: list[str]
    decoder: DecoderParams
    meta: dict


def load_model(path: Path) -> Model:
    b = joblib.load(path)
    meta = {k: v for k, v in b.items() if k not in ("estimator", "features")}
    return Model(b["estimator"], list(b["features"]), DecoderParams(**b["decoder"]), meta)


@dataclass
class Prediction:
    cand: pd.DataFrame        # candidate rows with p
    chosen: np.ndarray        # arc id per observation, -1 where no candidate
    route: list[int]          # drivable route arcs
    chains: int
    timing_ms: dict


def predict_trip(trip: Trip, model: Model, keep_obs: np.ndarray | None = None) -> Prediction:
    """keep_obs: optional boolean mask of observations to decode (offline checks only)."""
    t0 = time.perf_counter()
    index = build_index(trip)
    t1 = time.perf_counter()
    cand = build(trip, index)
    if keep_obs is not None:
        cand = cand[keep_obs[cand.obs_idx.to_numpy()]].reset_index(drop=True)
    t2 = time.perf_counter()
    cand["p"] = model.estimator.predict_proba(cand[model.features].to_numpy(np.float32))[:, 1]
    t3 = time.perf_counter()
    chosen, chains, route = decode_trip(trip, cand, model.decoder, RoadGraph.from_trip(trip))
    t4 = time.perf_counter()
    ms = lambda a, b: round((b - a) * 1000, 1)  # noqa: E731
    timing = {"candidates": ms(t0, t1), "features": ms(t1, t2), "classifier": ms(t2, t3),
              "decoder": ms(t3, t4), "total": ms(t0, t4)}
    return Prediction(cand, chosen, route, chains, timing)


def _continuity(seq: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[int, int]:
    seq = seq[seq >= 0]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    if len(seq) < 2:
        return 0, 0
    return int(np.sum(v[seq[:-1]] == u[seq[1:]])), len(seq) - 1


def score_trip(trip: Trip, pred: Prediction) -> dict:
    """Contract section 9 metrics. Needs trip.route; server side only."""
    pos, dist = align_ordered(trip)
    canon = canonical_arc(trip)
    target = canon[trip.route[pos]]
    rel = dist <= RELIABLE_M
    u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    nxy = trip.nodes_xy(); length = np.hypot(*(nxy[v] - nxy[u]).T)

    c = pred.cand[rel[pred.cand.obs_idx.to_numpy()]]
    y = (c.arc_id.to_numpy() == target[c.obs_idx.to_numpy()]).astype(int)
    covered = np.zeros(len(rel), bool); covered[c.obs_idx.to_numpy()[y == 1]] = True
    hits = int(np.sum((pred.chosen == target) & rel))
    ok, tot = _continuity(pred.chosen, u, v)
    fok, ftot = _continuity(np.asarray(pred.route, np.int64), u, v)

    P = set(canon[np.asarray(pred.route, int)].tolist()) if pred.route else set()
    R = set(canon[trip.route].tolist())
    inter = sum(length[a] for a in P & R)
    prec = inter / max(sum(length[a] for a in P), 1e-9)
    recall = inter / max(sum(length[a] for a in R), 1e-9)
    n_rel = int(rel.sum())
    return {
        "top1_accuracy": hits / max(n_rel, 1),
        "macro_f1": float(f1_score(y, (c.p.to_numpy() >= THRESHOLD).astype(int), average="macro")) if len(c) else None,
        "candidate_miss_rate": 1 - covered[rel].mean() if n_rel else None,
        "route_continuity": ok / tot if tot else None,
        "filled_route_continuity": fok / ftot if ftot else None,
        "route_f1": 2 * prec * recall / (prec + recall) if prec + recall else 0.0,
        "n_obs_scored": n_rel,
        # raw counts for pooled (micro) aggregation across trips
        "_counts": {"top1_hits": hits, "n_rel": n_rel, "cont_ok": ok, "cont_tot": tot,
                    "filled_ok": fok, "filled_tot": ftot, "route_precision": prec, "route_recall": recall},
    }


def to_geo(trip: Trip, pred: Prediction) -> tuple[list[dict], dict]:
    """Matched points and the drivable route as GeoJSON, in lon/lat.

    The local projection is affine in lon/lat, so the projected point on an arc is
    the same fraction along the arc in lon/lat as in metres.
    """
    lon, lat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    frac = pred.cand.set_index(["obs_idx", "arc_id"]).proj_frac
    matched = []
    for i, a in enumerate(pred.chosen.tolist()):
        if a < 0:
            matched.append({"obs_idx": i, "arc_id": None, "lon": None, "lat": None})
            continue
        f = float(frac.loc[(i, a)])
        matched.append({"obs_idx": i, "arc_id": int(a),
                        "lon": float(lon[u[a]] + f * (lon[v[a]] - lon[u[a]])),
                        "lat": float(lat[u[a]] + f * (lat[v[a]] - lat[u[a]]))})
    coords = []
    for a in pred.route:
        for node in (u[a], v[a]):
            pt = [float(lon[node]), float(lat[node])]
            if not coords or coords[-1] != pt:
                coords.append(pt)
    return matched, {"type": "LineString", "coordinates": coords}

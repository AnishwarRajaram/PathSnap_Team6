"""Evaluate route reconstruction: classifier only, classifier + HMM, distance-only HMM.

Usage: PATHSNAP_DATA=<data root> python hmm/run_hmm.py --model hgb --split train_oof|student_test
       [--alpha A --beta B] [--variants classifier,hmm,distance_hmm] [--out <tag>]
Emissions come from artifacts/oof/<model>_<split>.pkl, written by models/harness.py.
Train predictions are out-of-fold, so no trip is scored by a model that saw it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.hmm import DecoderParams, RoadGraph, decode_trip  # noqa: E402

ART = ROOT / "artifacts"
OUT = ROOT / "results" / "hmm"
WORKERS = 6


def continuity(seq: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[int, int]:
    seq = seq[seq >= 0]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    if len(seq) < 2:
        return 0, 0
    return int(np.sum(v[seq[:-1]] == u[seq[1:]])), len(seq) - 1


def route_prf(pred: list[int], ref: np.ndarray, canon: np.ndarray, length: np.ndarray):
    """Length-weighted precision, recall over distinct canonical arcs."""
    P = set(canon[np.asarray(pred, int)].tolist()) if len(pred) else set()
    R = set(canon[ref].tolist())
    inter = sum(length[a] for a in P & R)
    prec = inter / max(sum(length[a] for a in P), 1e-9)
    rec = inter / max(sum(length[a] for a in R), 1e-9)
    return prec, rec


def run(job):
    part, tid, cand, variants, prm_d = job
    trip = load_trip(part, tid)
    lab = np.load(ART / "labels" / part / f"{tid}.npz")
    target, rel = lab["arc_canon"], lab["reliable"]
    u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    nxy = trip.nodes_xy(); length = np.hypot(*(nxy[v] - nxy[u]).T)
    canon = trip.arcs.reset_index().groupby(["u", "v"]).arc_id.transform("min").to_numpy()
    n = len(trip.track)
    graph = RoadGraph.from_trip(trip)
    out = []
    for var in variants:
        t0 = time.perf_counter()
        if var == "classifier":
            best = cand.loc[cand.groupby("obs_idx").p.idxmax()]
            chosen = np.full(n, -1, np.int64); chosen[best.obs_idx.to_numpy()] = best.arc_id.to_numpy()
            seq = chosen[chosen >= 0]
            route = seq[np.r_[True, seq[1:] != seq[:-1]]].tolist() if len(seq) else []
            chains = np.nan
        else:
            prm = DecoderParams(**{**prm_d, "emission": "distance" if var == "distance_hmm" else "classifier"})
            chosen, chains, route = decode_trip(trip, cand, prm, graph)
        secs = time.perf_counter() - t0
        ok, tot = continuity(chosen, u, v)
        prec, rec = route_prf(route, trip.route, canon, length)
        filled_ok, filled_tot = continuity(np.asarray(route, np.int64), u, v)
        out.append(dict(partition=part, trip_id=tid, variant=var, n_obs=n, n_rel=int(rel.sum()),
                        top1_hits=int(np.sum((chosen == target) & rel)), cont_ok=ok, cont_tot=tot,
                        filled_cont_ok=filled_ok, filled_cont_tot=filled_tot,
                        route_precision=prec, route_recall=rec, chains=chains, secs=secs))
    return out


def evaluate(model: str, split: str, prm: dict, variants: list[str], workers: int = WORKERS) -> pd.DataFrame:
    df = pd.read_pickle(ART / "oof" / f"{model}_{split}.pkl")
    part = "train" if split == "train_oof" else "student_test"
    jobs = [(part, tid, g.drop(columns=["trip_id"]), variants, prm) for tid, g in df.groupby("trip_id")]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for rs in ex.map(run, jobs) for r in rs]
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("variant")
    s = pd.DataFrame({
        "top1": g.top1_hits.sum() / g.n_rel.sum(),
        "route_continuity": g.cont_ok.sum() / g.cont_tot.sum(),
        "filled_route_continuity": g.filled_cont_ok.sum() / g.filled_cont_tot.sum(),
        "route_precision": g.route_precision.mean(), "route_recall": g.route_recall.mean(),
        "chains_per_trip": g.chains.mean(), "secs_per_trip": g.secs.mean(),
        "trips_top1_worse_than_classifier": np.nan,
    })
    s["route_f1"] = 2 * s.route_precision * s.route_recall / (s.route_precision + s.route_recall)
    if "classifier" in res.variant.values:
        base = res[res.variant == "classifier"].set_index("trip_id")
        for var in s.index:
            cur = res[res.variant == var].set_index("trip_id")
            s.loc[var, "trips_top1_worse_than_classifier"] = int(
                ((cur.top1_hits / cur.n_rel.clip(lower=1)) < (base.top1_hits / base.n_rel.clip(lower=1)) - 1e-9).sum())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb"); ap.add_argument("--split", default="train_oof")
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=7.0)
    ap.add_argument("--jump", type=float, default=None)
    ap.add_argument("--variants", default="classifier,hmm,distance_hmm"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prm = {"alpha": a.alpha, "beta": a.beta, "sigma": a.sigma, "jump_penalty": a.jump}
    t0 = time.perf_counter()
    res = evaluate(a.model, a.split, prm, a.variants.split(","))
    s = summarise(res)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string())
    print(f"wall {time.perf_counter()-t0:.1f}s")
    if a.out:
        OUT.mkdir(parents=True, exist_ok=True)
        res.to_csv(OUT / f"{a.out}_per_trip.csv", index=False)
        s.to_csv(OUT / f"{a.out}_summary.csv")
        (OUT / f"{a.out}_params.json").write_text(json.dumps({"model": a.model, "split": a.split, **prm}, indent=2))


if __name__ == "__main__":
    main()

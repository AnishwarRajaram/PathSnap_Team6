"""EDA step 4: build observation labels by aligning GPS points to route arcs.

Runs method A (nearest route arc) and method B (ordered dynamic programming)
on every trip with a process pool, saves per-trip labels to artifacts/labels
(gitignored) and writes per-trip aggregates to results/eda/label_alignment.csv.
Prints aggregates only.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import list_partitions, list_trips, load_trip  # noqa: E402
from pathsnap.labels import RELIABLE_M, align_nearest, align_ordered, canonical_arc  # noqa: E402

OUT = ROOT / "results" / "eda"
ART = ROOT / "artifacts" / "labels"
WORKERS = 6


def run(job):
    part, tid = job
    trip = load_trip(part, tid)
    t0 = time.perf_counter(); pa, da = align_nearest(trip); ta = time.perf_counter() - t0
    t0 = time.perf_counter(); pb, db = align_ordered(trip); tb = time.perf_counter() - t0
    canon = canonical_arc(trip)
    arc_a, arc_b = trip.route[pa], trip.route[pb]
    same_arc = canon[arc_a] == canon[arc_b]
    rel_b = db <= RELIABLE_M
    (ART / part).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ART / part / f"{tid}.npz", pos=pb, arc=arc_b, arc_canon=canon[arc_b],
                        dist=db, reliable=rel_b, pos_nearest=pa, arc_nearest=arc_a, dist_nearest=da)
    r = len(trip.route)
    return dict(
        partition=part, trip_id=tid, n_obs=len(pb), n_route=r,
        secs_A=ta, secs_B=tb,
        agree_arc=float(same_arc.mean()),
        agree_arc_reliable=float(same_arc[rel_b].mean()) if rel_b.any() else np.nan,
        disagree_far_pos=float(np.mean(np.abs(pa - pb) > 3)),
        reliable_A=float(np.mean(da <= RELIABLE_M)), reliable_B=float(rel_b.mean()),
        distA_p50=float(np.median(da)), distB_p50=float(np.median(db)),
        distB_p90=float(np.percentile(db, 90)), distB_p99=float(np.percentile(db, 99)),
        cost_gap_B_minus_A_p50=float(np.median(db - da)),
        route_pos_covered_B=float(len(np.unique(pb)) / r),
        monotone_B=int(np.all(np.diff(pb) >= 0)),
        start_pos_B=float(pb[0] / max(r - 1, 1)), end_pos_B=float(pb[-1] / max(r - 1, 1)),
    )


def main():
    jobs = [(p, t) for p in list_partitions() for t in list_trips(p)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(run, jobs))
    wall = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "label_alignment.csv", index=False)

    pd.set_option("display.width", 160)
    cols = ["agree_arc", "agree_arc_reliable", "disagree_far_pos", "reliable_A", "reliable_B",
            "distA_p50", "distB_p50", "distB_p90", "distB_p99", "route_pos_covered_B",
            "start_pos_B", "end_pos_B", "secs_A", "secs_B"]
    print(df.groupby("partition")[cols].describe(percentiles=[.1, .5]).T
          .loc[(slice(None), ["mean", "min", "10%", "50%", "max"]), :].round(3).to_string())
    for p, g in df.groupby("partition"):
        w = g.n_obs / g.n_obs.sum()
        print(f"\n[{p}] obs-weighted: agree={np.sum(w*g.agree_arc):.3%} reliable_B={np.sum(w*g.reliable_B):.3%} "
              f"far-disagree={np.sum(w*g.disagree_far_pos):.3%}  trips<90% reliable: {(g.reliable_B<0.9).sum()}  "
              f"trips<95% agree: {(g.agree_arc<0.95).sum()}")
    print(f"\nAll monotone: {bool(df.monotone_B.all())}   wall time {wall:.1f}s with {WORKERS} workers")
    print(f"Wrote {OUT/'label_alignment.csv'}; labels in {ART} (gitignored)")


if __name__ == "__main__":
    main()

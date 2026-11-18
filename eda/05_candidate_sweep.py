"""EDA step 5: candidate radius sweep.

For each reliable labelled observation, find all arcs within R_MAX and measure,
for several radii and top-k limits, how often the labelled arc is missing
(candidate miss rate) and how many candidates each observation gets.
Train decides the radius. Student-test is reported as a check only.
Writes results/eda/candidate_sweep.csv and a figure. Prints aggregates only.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.candidates import build_index, query  # noqa: E402
from pathsnap.data import list_partitions, list_trips, load_trip  # noqa: E402

OUT = ROOT / "results" / "eda"
ART = ROOT / "artifacts" / "labels"
RADII = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
TOPK = [2, 4, 8, 12, 16, 24, 32]
R_MAX = max(RADII)
WORKERS = 6


def run(job):
    part, tid = job
    trip = load_trip(part, tid)
    lab = np.load(ART / part / f"{tid}.npz")
    rel = lab["reliable"]
    target = lab["arc_canon"]
    t0 = time.perf_counter()
    idx = build_index(trip)
    t_build = time.perf_counter() - t0
    pts = trip.track_xy()
    t0 = time.perf_counter()
    obs, arc, d = query(idx, pts, R_MAX)
    t_query = time.perf_counter() - t0
    n = len(pts)
    # rank of each candidate by distance within its observation
    order = np.lexsort((d, obs))
    obs, arc, d = obs[order], arc[order], d[order]
    first = np.searchsorted(obs, np.arange(n))
    rank = np.arange(len(obs)) - first[obs]
    is_true = arc == target[obs]
    true_rank = np.full(n, np.inf); true_d = np.full(n, np.inf)
    true_rank[obs[is_true]] = rank[is_true]; true_d[obs[is_true]] = d[is_true]
    rows = []
    for r in RADII:
        cnt = np.bincount(obs[d <= r], minlength=n)
        rows.append(dict(partition=part, trip_id=tid, kind="radius", value=r, n_rel=int(rel.sum()),
                         miss=float(np.mean(true_d[rel] > r)) if rel.any() else np.nan,
                         cand_mean=float(cnt.mean()), cand_p90=float(np.percentile(cnt, 90)),
                         cand_p99=float(np.percentile(cnt, 99)), zero_cand=float(np.mean(cnt == 0)),
                         rows_total=int(cnt.sum())))
    for k in TOPK:
        cnt = np.minimum(np.bincount(obs, minlength=n), k)
        rows.append(dict(partition=part, trip_id=tid, kind="topk", value=k, n_rel=int(rel.sum()),
                         miss=float(np.mean(true_rank[rel] >= k)) if rel.any() else np.nan,
                         cand_mean=float(cnt.mean()), cand_p90=float(np.percentile(cnt, 90)),
                         cand_p99=float(np.percentile(cnt, 99)), zero_cand=float(np.mean(cnt == 0)),
                         rows_total=int(cnt.sum())))
    for r in rows:
        r.update(t_build=t_build, t_query=t_query, n_obs=n)
    return rows


def main():
    jobs = [(p, t) for p in list_partitions() for t in list_trips(p)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        rows = [r for rs in ex.map(run, jobs) for r in rs]
    wall = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "candidate_sweep.csv", index=False)

    def pooled(g):
        w = g.n_rel / g.n_rel.sum()
        return pd.Series({"miss_pooled": np.sum(w * g.miss), "miss_trip_p90": g.miss.quantile(.9),
                          "miss_trip_max": g.miss.max(), "cand_mean": np.average(g.cand_mean, weights=g.n_obs),
                          "cand_p99_max": g.cand_p99.max(), "zero_cand": np.average(g.zero_cand, weights=g.n_obs),
                          "rows_M": g.rows_total.sum() / 1e6})
    pd.set_option("display.width", 160)
    summ = df.groupby(["kind", "value", "partition"]).apply(pooled, include_groups=False).unstack("partition")
    print(summ.round(4).to_string())
    per = df.drop_duplicates(["partition", "trip_id"])
    print(f"\nIndex build s: p50={per.t_build.median():.2f} max={per.t_build.max():.2f} | "
          f"query s (R={R_MAX}): p50={per.t_query.median():.2f} max={per.t_query.max():.2f} | wall {wall:.1f}s")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for p in ["train", "student_test"]:
        s = summ.xs("radius", level="kind")
        ax[0].plot(s.index, s[("miss_pooled", p)] * 100, marker="o", label=p)
        ax[1].plot(s.index, s[("cand_mean", p)], marker="o", label=p)
    ax[0].set(title="Candidate miss rate vs radius", xlabel="radius (m)", ylabel="miss %", yscale="log")
    ax[1].set(title="Mean candidates per observation", xlabel="radius (m)", ylabel="candidates")
    for a in ax: a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "figs" / "05_candidate_sweep.png", dpi=120)
    print(f"Wrote {OUT/'candidate_sweep.csv'} and figs/05_candidate_sweep.png")


if __name__ == "__main__":
    main()

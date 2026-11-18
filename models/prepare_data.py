"""Rebuild labels and feature tables from the raw data in one command.

Usage: PATHSNAP_DATA=<data root> python models/prepare_data.py
Writes artifacts/labels/<partition>/<trip>.npz and artifacts/features/<partition>.pkl.
Both directories are gitignored. Uses the same library code as EDA steps 4 and 6.
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
from pathsnap.features import build  # noqa: E402
from pathsnap.labels import RELIABLE_M, align_nearest, align_ordered, canonical_arc  # noqa: E402

ART = ROOT / "artifacts"
WORKERS = 6


def label_job(job):
    part, tid = job
    trip = load_trip(part, tid)
    pa, da = align_nearest(trip)
    pb, db = align_ordered(trip)
    canon = canonical_arc(trip)
    (ART / "labels" / part).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ART / "labels" / part / f"{tid}.npz", pos=pb, arc=trip.route[pb],
                        arc_canon=canon[trip.route[pb]], dist=db, reliable=db <= RELIABLE_M,
                        pos_nearest=pa, arc_nearest=trip.route[pa], dist_nearest=da)


def feature_job(job):
    part, tid = job
    df = build(load_trip(part, tid))
    lab = np.load(ART / "labels" / part / f"{tid}.npz")
    o = df.obs_idx.to_numpy()
    df["y"] = (df.arc_id.to_numpy() == lab["arc_canon"][o]).astype(np.int8)
    df["reliable"] = lab["reliable"][o]
    df.insert(0, "trip_id", tid)
    return part, df


def main():
    jobs = [(p, t) for p in list_partitions() for t in list_trips(p)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(label_job, jobs))
        print(f"labels: {len(jobs)} trips in {time.perf_counter()-t0:.1f}s", flush=True)
        parts: dict[str, list] = {}
        for p, df in ex.map(feature_job, jobs):
            parts.setdefault(p, []).append(df)
    (ART / "features").mkdir(parents=True, exist_ok=True)
    for p, dfs in parts.items():
        d = pd.concat(dfs, ignore_index=True)
        d.to_pickle(ART / "features" / f"{p}.pkl")
        print(f"features[{p}]: {len(d):,} rows, {int(d.reliable.sum()):,} reliable")
    print(f"done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

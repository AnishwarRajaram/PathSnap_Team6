"""EDA step 2: GPS track quality.

Per trip: sampling intervals, gaps, implied speed, jumps, stationary periods
and heading change. Flags anomalous trips with rules fixed in advance.
Writes results/eda/track_quality.csv and aggregate figures under results/eda/figs.
Prints partition-level aggregates only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import list_partitions, list_trips, load_trip  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "eda"
FIGS = OUT / "figs"

# Flag rules, fixed before looking at results.
GAP_LONG_S = 300          # any single gap longer than 5 min
TELEPORT_MPS = 70         # implied speed above 252 km/h
STATIONARY_MPS = 0.5
STATIONARY_SHARE = 0.30   # more than 30% of steps stationary
SEGMENT_GAP_S = 30        # gap that splits a track into segments


def longest_run(mask: np.ndarray, weights: np.ndarray) -> float:
    best = cur = 0.0
    for m, w in zip(mask, weights):
        cur = cur + w if m else 0.0
        best = max(best, cur)
    return best


def analyse(trip):
    xy = trip.track_xy()
    t = trip.track.t.to_numpy()
    dt = np.diff(t)
    step = np.hypot(*np.diff(xy, axis=0).T)
    speed = step / np.where(dt > 0, dt, np.nan)
    moving = speed >= STATIONARY_MPS
    hdg = np.degrees(np.arctan2(np.diff(xy[:, 0]), np.diff(xy[:, 1])))
    hmov = hdg[moving]
    dh = np.abs((np.diff(hmov) + 180) % 360 - 180) if len(hmov) > 1 else np.array([0.0])
    row = dict(
        partition=trip.partition, trip_id=trip.trip_id, n=len(t),
        dt_share_1s=float(np.mean(dt == 1)), dt_share_nonint=float(np.mean(np.mod(dt, 1) != 0)),
        gaps_gt2=int((dt > 2).sum()), gaps_gt5=int((dt > 5).sum()), gaps_gt30=int((dt > 30).sum()),
        gaps_gt300=int((dt > 300).sum()), max_gap_s=float(dt.max()),
        time_in_gaps_gt30_share=float(dt[dt > 30].sum() / max(t[-1] - t[0], 1)),
        n_segments=int((dt > SEGMENT_GAP_S).sum() + 1),
        speed_p50=float(np.nanpercentile(speed, 50)), speed_p95=float(np.nanpercentile(speed, 95)),
        speed_max=float(np.nanmax(speed)), n_teleport=int(np.nansum(speed > TELEPORT_MPS)),
        step_p99_m=float(np.percentile(step, 99)),
        stationary_share=float(np.mean(~moving)),
        longest_stationary_s=float(longest_run(~moving, dt)),
        heading_change_p50=float(np.median(dh)), heading_change_p95=float(np.percentile(dh, 95)),
        track_km=float(step.sum() / 1000),
    )
    row["flag_long_gap"] = int(row["max_gap_s"] > GAP_LONG_S)
    row["flag_teleport"] = int(row["n_teleport"] > 0)
    row["flag_stationary"] = int(row["stationary_share"] > STATIONARY_SHARE)
    row["flag_any"] = int(row["flag_long_gap"] or row["flag_teleport"] or row["flag_stationary"])
    return row, dt, speed


def main():
    rows, dts, speeds = [], {}, {}
    for p in list_partitions():
        dts[p], speeds[p] = [], []
        for tid in list_trips(p):
            r, dt, sp = analyse(load_trip(p, tid, with_route=False))
            rows.append(r); dts[p].append(dt); speeds[p].append(sp)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "track_quality.csv", index=False)

    pd.set_option("display.width", 160)
    cols = ["dt_share_1s", "dt_share_nonint", "gaps_gt5", "gaps_gt30", "gaps_gt300", "max_gap_s",
            "time_in_gaps_gt30_share", "n_segments", "speed_p50", "speed_p95", "speed_max",
            "n_teleport", "stationary_share", "longest_stationary_s", "heading_change_p95"]
    print(df.groupby("partition")[cols].describe(percentiles=[.5, .9]).T.round(3).to_string())
    print("\nFlag counts:\n", df.groupby("partition")[["flag_long_gap", "flag_teleport", "flag_stationary", "flag_any"]].sum().to_string())
    for p in dts:
        dt = np.concatenate(dts[p]); sp = np.concatenate(speeds[p])
        print(f"\n[{p}] pooled steps={len(dt)}  dt==1: {np.mean(dt==1):.3%}  dt in (1,2]: {np.mean((dt>1)&(dt<=2)):.3%}  "
              f"dt<1: {np.mean(dt<1):.3%}  dt>5: {np.mean(dt>5):.3%}")
        print(f"[{p}] pooled speed m/s: p50={np.nanpercentile(sp,50):.1f} p90={np.nanpercentile(sp,90):.1f} "
              f"p99={np.nanpercentile(sp,99):.1f} p99.9={np.nanpercentile(sp,99.9):.1f} max={np.nanmax(sp):.1f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p in dts:
        dt = np.concatenate(dts[p]); sp = np.concatenate(speeds[p])
        ax[0].hist(np.clip(dt, 0, 30), bins=60, alpha=.6, label=p, log=True, density=True)
        ax[1].hist(np.clip(sp[np.isfinite(sp)], 0, 60), bins=60, alpha=.6, label=p, density=True)
    ax[0].set(title="Sampling interval (clipped at 30 s)", xlabel="dt (s)", ylabel="density (log)")
    ax[1].set(title="Implied speed (clipped at 60 m/s)", xlabel="m/s", ylabel="density")
    for p, g in df.groupby("partition"):
        ax[2].scatter(g.n, g.max_gap_s, label=p, alpha=.7)
    ax[2].set(xscale="log", yscale="log", title="Max gap per trip", xlabel="points in trip", ylabel="max gap (s)")
    ax[2].axhline(GAP_LONG_S, ls="--", c="grey")
    for a in ax: a.legend()
    fig.tight_layout(); fig.savefig(FIGS / "02_track_quality.png", dpi=120)
    print(f"\nWrote {OUT/'track_quality.csv'} and {FIGS/'02_track_quality.png'}")


if __name__ == "__main__":
    main()

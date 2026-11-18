"""EDA step 1: dataset inventory and basic validity checks.

Writes one summary row per trip to results/eda/inventory.csv and prints
partition-level aggregates only.

Usage: PATHSNAP_DATA=<data root> python -m eda.01_inventory   (or run as a script)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import list_partitions, list_trips, load_trip, trip_files, haversine_m, to_local_xy  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "eda"


def inspect(partition: str, trip_id: str) -> dict:
    files = trip_files(partition, trip_id)
    row = {"partition": partition, "trip_id": trip_id}
    row.update({f"missing_{k}": int(not p.exists()) for k, p in files.items()})
    row.update({f"kb_{k}": round(p.stat().st_size / 1024, 1) if p.exists() else np.nan for k, p in files.items()})
    tr = load_trip(partition, trip_id)
    t, n, a, r = tr.track, tr.nodes, tr.arcs, tr.route

    row.update(n_track=len(t), n_nodes=len(n), n_arcs=len(a), n_route=len(r))
    row.update(nan_track=int(t.isna().sum().sum()), nan_nodes=int(n.isna().sum().sum()))

    def bad_coord(df):
        return int(((df.lon.abs() > 180) | (df.lat.abs() > 90) | ((df.lon == 0) & (df.lat == 0))).sum())
    row.update(bad_coord_track=bad_coord(t), bad_coord_nodes=bad_coord(n))

    dt = np.diff(t.t.to_numpy())
    row.update(dup_t=int((dt == 0).sum()), backward_t=int((dt < 0).sum()),
               t_is_integer=int(np.all(np.mod(t.t.to_numpy(), 1) == 0)),
               duration_s=float(t.t.iloc[-1] - t.t.iloc[0]))
    same_pos = (np.diff(t.lon.to_numpy()) == 0) & (np.diff(t.lat.to_numpy()) == 0)
    row["dup_consecutive_pos"] = int(same_pos.sum())

    u, v = a.u.to_numpy(), a.v.to_numpy()
    row.update(arc_bad_node=int(((u < 0) | (u >= len(n)) | (v < 0) | (v >= len(n))).sum()),
               arc_self_loop=int((u == v).sum()),
               arc_duplicate=int(a.duplicated().sum()),
               route_bad_arc=int(((r < 0) | (r >= len(a))).sum()))

    # Coordinate-order plausibility: share of track points inside the node bounding box,
    # as stored and with lon/lat swapped.
    def inside(lon, lat):
        return float(np.mean((lon >= n.lon.min()) & (lon <= n.lon.max()) & (lat >= n.lat.min()) & (lat <= n.lat.max())))
    row["track_in_node_bbox"] = inside(t.lon, t.lat)
    row["track_in_node_bbox_swapped"] = inside(t.lat, t.lon)

    # Projection check: local equirectangular vs haversine over the node extent.
    lon0, lat0 = tr.origin
    corners_lon = np.array([n.lon.min(), n.lon.max(), n.lon.min(), n.lon.max()])
    corners_lat = np.array([n.lat.min(), n.lat.min(), n.lat.max(), n.lat.max()])
    h = haversine_m(lon0, lat0, corners_lon, corners_lat)
    xy = to_local_xy(corners_lon, corners_lat, lon0, lat0)
    row["proj_max_rel_err_pct"] = float(np.max(np.abs(np.hypot(*xy.T) - h) / np.maximum(h, 1)) * 100)
    row["node_extent_km"] = float(max(haversine_m(n.lon.min(), lat0, n.lon.max(), lat0),
                                      haversine_m(lon0, n.lat.min(), lon0, n.lat.max())) / 1000)
    row["centre_lon"], row["centre_lat"] = round(lon0, 1), round(lat0, 1)
    return row


def main() -> None:
    rows = [inspect(p, tid) for p in list_partitions() for tid in list_trips(p)]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "inventory.csv", index=False)

    pd.set_option("display.width", 160)
    print("Trips per partition:\n", df.groupby("partition").size().to_string(), "\n")
    count_cols = ["missing_track", "missing_nodes", "missing_arcs", "missing_route",
                  "nan_track", "nan_nodes", "bad_coord_track", "bad_coord_nodes",
                  "dup_t", "backward_t", "dup_consecutive_pos",
                  "arc_bad_node", "arc_self_loop", "arc_duplicate", "route_bad_arc"]
    print("Issue totals (sum over trips):\n", df.groupby("partition")[count_cols].sum().T.to_string(), "\n")
    print("Trips with non-integer timestamps:", int((df.t_is_integer == 0).sum()))
    size_cols = ["n_track", "n_nodes", "n_arcs", "n_route", "duration_s", "node_extent_km",
                 "kb_track", "kb_nodes", "kb_arcs", "kb_route"]
    print("\nSize distribution by partition:")
    print(df.groupby("partition")[size_cols].describe(percentiles=[.5, .9]).T.round(1).to_string())
    print("\nCoordinate order: mean share of track inside node bbox "
          f"as stored = {df.track_in_node_bbox.mean():.3f}, swapped = {df.track_in_node_bbox_swapped.mean():.3f}")
    print("Trips with <99% of track inside node bbox:", int((df.track_in_node_bbox < 0.99).sum()))
    print(f"Projection max relative error over all trips: {df.proj_max_rel_err_pct.max():.3f}%")
    print(f"\nWrote {OUT / 'inventory.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()

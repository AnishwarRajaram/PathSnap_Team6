"""Data access and geodesy helpers shared by every PathSnap stage.

The data root is read from the PATHSNAP_DATA environment variable. It must
contain one directory per partition (train, student_test), each holding one
directory per trip with <trip>.track, <trip>.nodes, <trip>.arcs, <trip>.route.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8
EXTS = ("track", "nodes", "arcs", "route")


def data_root() -> Path:
    root = os.environ.get("PATHSNAP_DATA")
    if not root:
        raise RuntimeError("Set PATHSNAP_DATA to the directory that holds the partitions.")
    return Path(root)


def list_partitions() -> list[str]:
    return sorted(p.name for p in data_root().iterdir() if p.is_dir())


def list_trips(partition: str) -> list[str]:
    return sorted(p.name for p in (data_root() / partition).iterdir() if p.is_dir())


def trip_files(partition: str, trip_id: str) -> dict[str, Path]:
    d = data_root() / partition / trip_id
    return {ext: d / f"{trip_id}.{ext}" for ext in EXTS}


@dataclass
class Trip:
    partition: str
    trip_id: str
    track: pd.DataFrame   # lon, lat, t
    nodes: pd.DataFrame   # lon, lat ; index = node_id
    arcs: pd.DataFrame    # u, v ; index = arc_id
    route: np.ndarray     # ordered arc_ids

    @property
    def origin(self) -> tuple[float, float]:
        """Projection origin: mean of the track (lon, lat)."""
        return float(self.track.lon.mean()), float(self.track.lat.mean())

    def track_xy(self) -> np.ndarray:
        return to_local_xy(self.track.lon.to_numpy(), self.track.lat.to_numpy(), *self.origin)

    def nodes_xy(self) -> np.ndarray:
        return to_local_xy(self.nodes.lon.to_numpy(), self.nodes.lat.to_numpy(), *self.origin)


def _read(path: Path, names: list[str], dtype) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(partition: str, trip_id: str, with_route: bool = True) -> Trip:
    f = trip_files(partition, trip_id)
    track = _read(f["track"], ["lon", "lat", "t"], float)
    nodes = _read(f["nodes"], ["lon", "lat"], float)
    arcs = _read(f["arcs"], ["u", "v"], np.int64)
    route = (_read(f["route"], ["arc_id"], np.int64).arc_id.to_numpy()
             if with_route else np.empty(0, np.int64))
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"
    return Trip(partition, trip_id, track, nodes, arcs, route)


def to_local_xy(lon, lat, lon0: float, lat0: float) -> np.ndarray:
    """Local equirectangular projection in metres around (lon0, lat0).

    Trip extents are under about 80 km, where this projection stays within a
    fraction of a percent of geodesic distance. It is fast and needs no CRS lookup.
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))

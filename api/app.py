"""PathSnap prediction service (laptop 1). Contract v1.1, section 8.

Environment:
  PATHSNAP_DATA       data root that holds the evaluation partition
  PATHSNAP_PARTITION  partition directory to serve (default: student_test)
  PATHSNAP_MODEL      model bundle (default: artifacts/models/final.joblib)

Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000
The reference route, when present, is read on the server only to compute the
metrics. Reference arcs and labels are never returned.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import list_trips, load_trip, trip_files  # noqa: E402
from pathsnap.predict import load_model, predict_trip, score_trip, to_geo  # noqa: E402

PARTITION = os.environ.get("PATHSNAP_PARTITION", "student_test")
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "artifacts" / "models" / "final.joblib"))

app = FastAPI(title="PathSnap", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

_model = None
_cache: dict[str, dict] = {}
_lock = threading.Lock()


def model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    trip_id: str


def _known(trip_id: str) -> None:
    # Only identifiers from the served partition are accepted; nothing else reaches the filesystem.
    if trip_id not in list_trips(PARTITION):
        raise HTTPException(status_code=404, detail="unknown trip_id")


@app.get("/health")
def health():
    m = model()
    return {"status": "ok", "partition": PARTITION, "n_trips": len(list_trips(PARTITION)),
            "model": {"config_source": m.meta.get("config_source"), "trained_on": m.meta.get("trained_on"),
                      "decoder": m.meta.get("decoder")}}


@app.get("/trips")
def trips():
    return {"trips": list_trips(PARTITION)}


@app.get("/trips/{trip_id}/track")
def track(trip_id: str):
    """Raw GPS observations of a trip, for display. Input data only."""
    _known(trip_id)
    t = load_trip(PARTITION, trip_id, with_route=False).track
    return {"trip_id": trip_id, "points": t[["lon", "lat", "t"]].to_numpy().tolist()}


@app.post("/predict")
def predict(req: PredictRequest):
    _known(req.trip_id)
    with _lock:
        if req.trip_id in _cache:
            return _cache[req.trip_id] | {"cached": True}
    t0 = time.perf_counter()
    has_route = trip_files(PARTITION, req.trip_id)["route"].exists()
    trip = load_trip(PARTITION, req.trip_id, with_route=has_route)
    pred = predict_trip(trip, model())
    matched, geo = to_geo(trip, pred)
    metrics = None
    if has_route:
        metrics = {k: v for k, v in score_trip(trip, pred).items() if not k.startswith("_")}
    timing = pred.timing_ms | {"total": round((time.perf_counter() - t0) * 1000, 1)}
    out = {"trip_id": req.trip_id, "matched": matched, "route_arcs": [int(a) for a in pred.route],
           "route_geojson": geo, "metrics": metrics, "timing_ms": timing, "cached": False}
    with _lock:
        _cache[req.trip_id] = out
    return out

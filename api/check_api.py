"""Offline check of the deployed pipeline.

1. Parity: decodes every student-test trip on reliable observations only, the
   setting of hmm/final.py, and compares pooled Top-1 and continuity with
   results/hmm/final_<model>_student_test_summary.csv.
2. API: calls GET /health, GET /trips, POST /predict for every trip through the
   FastAPI test client. Checks the response fields of contract section 8.2, that
   no reference route is returned, that unknown ids are rejected, and reports
   pooled metrics on all observations together with request times.
Usage: PATHSNAP_DATA=<data root> python api/check_api.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import api.app as app_module  # noqa: E402
from api.app import MODEL_PATH, PARTITION, app  # noqa: E402
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.labels import RELIABLE_M, align_ordered  # noqa: E402
from pathsnap.predict import load_model, predict_trip, score_trip  # noqa: E402

TOL = 1e-9
FIELDS = {"trip_id", "matched", "route_arcs", "route_geojson", "metrics", "timing_ms"}


def pooled(counts: list[dict]) -> dict:
    c = pd.DataFrame(counts).sum()
    return {"top1": c.top1_hits / c.n_rel, "route_continuity": c.cont_ok / c.cont_tot,
            "filled_route_continuity": c.filled_ok / c.filled_tot}


def main():
    ok = True
    m = load_model(MODEL_PATH)
    print(f"model: {m.meta['model']} from {m.meta.get('config_source')}, decoder {m.meta.get('decoder')}")

    counts = []
    for tid in list_trips(PARTITION):
        trip = load_trip(PARTITION, tid)
        _, dist = align_ordered(trip)
        pred = predict_trip(trip, m, keep_obs=dist <= RELIABLE_M)
        counts.append(score_trip(trip, pred)["_counts"])
    got = pooled(counts)
    ref = pd.read_csv(ROOT / "results" / "hmm" / f"final_{m.meta['model']}_student_test_summary.csv", index_col=0).loc["hmm"]
    print("== parity with hmm/final.py (reliable observations)")
    for k in got:
        good = abs(got[k] - ref[k]) < TOL
        ok &= good
        print(f"  {k:24s} expected {ref[k]:.6f} got {got[k]:.6f} {'OK' if good else 'DIFF'}")

    print("== API")
    app_module._model = m          # reuse the loaded bundle instead of loading it a second time
    cl = TestClient(app)
    h = cl.get("/health"); print(f"  /health {h.status_code} {h.json()['n_trips']} trips")
    ids = cl.get("/trips").json()["trips"]
    bad = cl.post("/predict", json={"trip_id": "../train"}).status_code
    print(f"  unknown trip_id -> {bad}"); ok &= bad == 404
    rows, counts = [], []
    for tid in ids:
        r = cl.post("/predict", json={"trip_id": tid}); j = r.json()
        ok &= r.status_code == 200 and FIELDS <= set(j) and "route" not in j
        ok &= len(j["matched"]) == len(load_trip(PARTITION, tid, with_route=False).track)
        mt = j["metrics"]
        counts.append({"hits": mt["top1_accuracy"] * mt["n_obs_scored"], "n": mt["n_obs_scored"]})
        rows.append({"trip_id": tid, **j["metrics"], "total_ms": j["timing_ms"]["total"],
                     "decoder_ms": j["timing_ms"]["decoder"]})
        again = cl.post("/predict", json={"trip_id": tid}).json()
        ok &= again["cached"] and again["route_arcs"] == j["route_arcs"]
    df = pd.DataFrame(rows)
    c = pd.DataFrame(counts).sum()
    print(f"  all observations, pooled: top1={c.hits / c.n:.4f}")
    print(f"  per-trip mean: macro_f1={df.macro_f1.mean():.4f} route_f1={df.route_f1.mean():.4f} "
          f"miss={df.candidate_miss_rate.mean():.4f}")
    print(f"  request time ms: median {df.total_ms.median():.0f}, max {df.total_ms.max():.0f}; "
          f"decoder share {np.sum(df.decoder_ms) / np.sum(df.total_ms):.0%}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

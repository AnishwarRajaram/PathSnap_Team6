"""Fit the deployment model and bundle it with the tuned decoder parameters.

The classifier is fitted on every reliable training row (80 trips). Student-test
trips are not used, so they remain a clean check of the deployed pipeline.
Default hyperparameters are those of models/hgb.py. With --tuned, the best
parameters in results/tuning/hgb_best.json are used instead.
Decoder parameters come from results/hmm/decoder_hgb.json.

Writes artifacts/models/final.joblib (gitignored).
Usage: PATHSNAP_DATA=<data root> python api/train_final.py [--tuned]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import hgb  # noqa: E402
from models.harness import load_rows  # noqa: E402
from pathsnap.features import SELECTED  # noqa: E402

OUT = ROOT / "artifacts" / "models" / "final.joblib"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tuned", action="store_true"); a = ap.parse_args()
    if a.tuned:
        cfg = json.loads((ROOT / "results" / "tuning" / "hgb_best.json").read_text())["best_params"]
        source = "results/tuning/hgb_best.json"
    else:
        cfg, source = dict(hgb.CONFIG), "models/hgb.py"
    dec = json.loads((ROOT / "results" / "hmm" / "decoder_hgb.json").read_text())
    decoder = {k: dec[k] for k in ("alpha", "beta", "jump_penalty")}

    train = load_rows("train")
    t0 = time.perf_counter()
    est = HistGradientBoostingClassifier(random_state=0, **cfg).fit(train[SELECTED].to_numpy(), train.y.to_numpy())
    fit_s = time.perf_counter() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": est, "features": SELECTED, "config": cfg, "config_source": source,
                 "decoder": decoder, "trained_on": "train (80 trips, reliable rows)"}, OUT)
    print(f"fitted on {len(train):,} rows in {fit_s:.1f}s with {source}; decoder {decoder}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

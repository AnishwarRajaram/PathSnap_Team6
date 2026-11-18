"""Fit the deployment model and bundle it with its tuned decoder parameters.

The model is any module in models/ that follows the harness interface. By default
the model selected in results/models/comparison.csv is used, in its tuned form
models/<selected>_tuned.py when that module exists (guidelines section 7: refit the
selected model with the chosen configuration). The classifier is
fitted on every reliable training row (80 trips), or on the module's
MAX_TRAIN_ROWS sample. Student-test trips are not used, so they remain a clean
check of the deployed pipeline.

Hyperparameters come from the module CONFIG. Decoder parameters come from
results/hmm/decoder_<model>.json, which hmm/tune.py writes.

Writes artifacts/models/final.joblib (gitignored).
Usage: PATHSNAP_DATA=<data root> python api/train_final.py [--model rf_tuned]
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.harness import load_rows, subsample_obs  # noqa: E402
from pathsnap.features import SELECTED  # noqa: E402

OUT = ROOT / "artifacts" / "models" / "final.joblib"


def selected_model() -> str:
    c = pd.read_csv(ROOT / "results" / "models" / "comparison.csv")
    name = str(c.loc[c.selected.astype(bool), "model"].iloc[0])
    return f"{name}_tuned" if (ROOT / "models" / f"{name}_tuned.py").exists() else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="module name in models/; default: selected model")
    a = ap.parse_args()
    name = a.model or selected_model()
    module = importlib.import_module(f"models.{name}")

    cfg, source = dict(module.CONFIG), f"models/{name}.py"
    dec_path = ROOT / "results" / "hmm" / f"decoder_{name}.json"
    if not dec_path.exists():
        raise SystemExit(f"{dec_path.relative_to(ROOT)} missing. Run: python hmm/tune.py --model {name}")
    dec = json.loads(dec_path.read_text())
    decoder = {k: dec[k] for k in ("alpha", "beta", "jump_penalty")}

    train = subsample_obs(load_rows("train"), getattr(module, "MAX_TRAIN_ROWS", None))
    t0 = time.perf_counter()
    est = module.build(cfg)
    est.fit(train[SELECTED].to_numpy(), train.y.to_numpy())
    fit_s = time.perf_counter() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": est, "features": SELECTED, "model": name, "config": cfg,
                 "config_source": source, "decoder": decoder,
                 "trained_on": f"train (80 trips, {len(train):,} reliable rows)"}, OUT)
    print(f"{name}: fitted on {len(train):,} rows in {fit_s:.1f}s with {source}; decoder {decoder}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

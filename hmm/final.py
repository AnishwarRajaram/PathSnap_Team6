"""Final route-reconstruction comparison with the tuned decoder.

Classifier + HMM uses results/hmm/decoder_<model>.json. The distance-only HMM uses
its own tuned beta with hard chain breaks. Writes final_<model>_<split>_{per_trip,summary}.csv.
Usage: PATHSNAP_DATA=<data root> python hmm/final.py --model hgb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmm.run_hmm import OUT, evaluate, summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="hgb"); a = ap.parse_args()
    d = json.loads((OUT / f"decoder_{a.model}.json").read_text())
    for split in ("train_oof", "student_test"):
        r1 = evaluate(a.model, split, {"alpha": d["alpha"], "beta": d["beta"], "sigma": d["sigma"],
                                       "jump_penalty": d["jump_penalty"]}, ["classifier", "hmm"])
        r2 = evaluate(a.model, split, {"alpha": 1.0, "beta": d["distance_hmm_beta"], "sigma": d["sigma"],
                                       "jump_penalty": None}, ["distance_hmm"])
        res = pd.concat([r1, r2], ignore_index=True)
        s = summarise(res)
        res.to_csv(OUT / f"final_{a.model}_{split}_per_trip.csv", index=False)
        s.to_csv(OUT / f"final_{a.model}_{split}_summary.csv")
        pd.set_option("display.width", 200)
        print(f"== {split}\n{s.round(4).to_string()}")


if __name__ == "__main__":
    main()

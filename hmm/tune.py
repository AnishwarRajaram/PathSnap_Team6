"""Tune decoder parameters on TRAIN out-of-fold predictions only.

Viterbi depends on alpha and beta only through their product, so alpha is fixed at 1
and the search covers beta and the soft-break jump penalty (None = hard chain break).

Objective (fixed in advance): observation-level Top-1 accuracy on reliable training
observations. Student-test data is not used. The distance-only HMM sigma is set from
training labels as 1.4826 x median point-to-label distance (robust standard deviation).
Usage: PATHSNAP_DATA=<data root> python hmm/tune.py --model hgb
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmm.run_hmm import OUT, evaluate, summarise  # noqa: E402

BETAS = [2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
JUMPS = [None, 5.0, 10.0, 25.0, 50.0]


def train_sigma() -> float:
    d = np.concatenate([np.load(f)["dist"][np.load(f)["reliable"]]
                        for f in sorted((ROOT / "artifacts" / "labels" / "train").glob("*.npz"))])
    return float(1.4826 * np.median(d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="hgb"); a = ap.parse_args()
    sigma = train_sigma()
    t0 = time.perf_counter()
    rows = []
    for beta, jump in itertools.product(BETAS, JUMPS):
        s = summarise(evaluate(a.model, "train_oof", {"alpha": 1.0, "beta": beta, "sigma": sigma, "jump_penalty": jump}, ["classifier", "hmm"]))
        r = s.loc["hmm"].to_dict(); r.update(alpha=1.0, beta=beta, jump_penalty=jump)
        rows.append(r)
        print(f"beta={beta:<5} jump={str(jump):<5} top1={r['top1']:.4f} cont={r['route_continuity']:.4f} "
              f"route_f1={r['route_f1']:.4f}", flush=True)
    sweep_d = []
    for beta in BETAS:
        s = summarise(evaluate(a.model, "train_oof", {"alpha": 1.0, "beta": beta, "sigma": sigma, "jump_penalty": None}, ["distance_hmm"]))
        r = s.loc["distance_hmm"].to_dict(); r.update(beta=beta); sweep_d.append(r)
    grid = pd.DataFrame(rows); dist = pd.DataFrame(sweep_d)
    best = grid.sort_values("top1", ascending=False).iloc[0]
    bd = dist.sort_values("top1", ascending=False).iloc[0]
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / f"tune_{a.model}_grid.csv", index=False)
    dist.to_csv(OUT / f"tune_{a.model}_distance_hmm.csv", index=False)
    jp = None if pd.isna(best.jump_penalty) else float(best.jump_penalty)
    chosen = {"model": a.model, "alpha": 1.0, "beta": float(best.beta), "jump_penalty": jp, "sigma": sigma,
              "distance_hmm_beta": float(bd.beta), "objective": "train OOF Top-1",
              "train_oof_top1": float(best.top1)}
    (OUT / f"decoder_{a.model}.json").write_text(json.dumps(chosen, indent=2))
    print(f"\nsigma={sigma:.2f}  best beta={best.beta} jump={jp} top1={best.top1:.4f} | "
          f"distance-only best beta={bd.beta} top1={bd.top1:.4f} | {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()

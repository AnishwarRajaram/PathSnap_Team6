"""Bayesian optimisation of the HistGradientBoosting classifier.

Optuna TPE sampler over five hyperparameters. Each trial runs the harness folds
(results/models/folds.csv, trip-grouped, 5 folds) and returns the mean CV macro F1
on reliable candidate rows, the harness primary metric. Trial 0 is the current
untuned configuration from models/hgb.py, so the baseline is part of the study.
Folds are reported one at a time and a median pruner stops weak trials early.
Student-test data is not read.

Outputs: results/tuning/hgb_trials.csv, results/tuning/hgb_best.json,
results/tuning/hgb_history.png. The study is stored in artifacts/tuning/hgb.db
(gitignored) so an interrupted run resumes.
Usage: PATHSNAP_DATA=<data root> python tuning/tune_hgb.py --trials 60 --timeout 4800
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import hgb  # noqa: E402
from models.harness import N_FOLDS, SEED, THRESHOLD, folds, load_rows  # noqa: E402
from pathsnap.features import SELECTED  # noqa: E402

OUT = ROOT / "results" / "tuning"
DB = ROOT / "artifacts" / "tuning" / "hgb.db"
FIXED = {"early_stopping": False, "class_weight": None, "random_state": 0}


def space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 400, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=4800, help="seconds")
    a = ap.parse_args()

    train = load_rows("train")
    fid = train.trip_id.map(folds(train)).to_numpy()
    X, y = train[SELECTED].to_numpy(), train.y.to_numpy()
    splits = [(np.flatnonzero(fid != k), np.flatnonzero(fid == k)) for k in range(N_FOLDS)]

    def objective(trial: optuna.Trial) -> float:
        cfg = space(trial)
        scores = []
        for k, (tr, va) in enumerate(splits):
            est = HistGradientBoostingClassifier(**cfg, **FIXED).fit(X[tr], y[tr])
            p = est.predict_proba(X[va])[:, 1]
            scores.append(f1_score(y[va], (p >= THRESHOLD).astype(int), average="macro"))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_std", float(np.std(scores, ddof=1)))
        return float(np.mean(scores))

    DB.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="hgb", storage=f"sqlite:///{DB}", load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    if not study.trials:
        study.enqueue_trial({k: hgb.CONFIG[k] for k in
                             ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf")}
                            | {"l2_regularization": 1e-4})
    t0 = time.time()
    remaining = max(a.trials - len(study.trials), 0)
    study.optimize(objective, n_trials=remaining, timeout=a.timeout, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number} {t.state.name} value={t.value} "
                                                 f"best={s.best_value:.4f} elapsed={time.time()-t0:.0f}s",
                                                 flush=True)])

    OUT.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration", "user_attrs"))
    df.to_csv(OUT / "hgb_trials.csv", index=False)
    done = df[df.state == "COMPLETE"]
    base = df.loc[df.number == 0, "value"].iloc[0]
    best = study.best_trial
    (OUT / "hgb_best.json").write_text(json.dumps({
        "objective": "mean CV macro F1, 5 trip-grouped folds, reliable rows",
        "baseline_trial0_macro_f1": base, "best_macro_f1": best.value,
        "best_fold_std": best.user_attrs.get("fold_std"), "best_trial": best.number,
        "best_params": best.params | {k: v for k, v in FIXED.items() if k != "random_state"},
        "n_trials": len(df), "n_complete": len(done), "n_pruned": int((df.state == "PRUNED").sum()),
    }, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(done.number, done.value, s=18, label="completed trial")
    ax.plot(done.number, done.value.cummax(), c="#d62728", label="best so far")
    ax.axhline(base, ls="--", c="grey", label="untuned baseline (trial 0)")
    ax.set(title="HGB Bayesian optimisation (TPE)", xlabel="trial", ylabel="CV macro F1"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "hgb_history.png", dpi=120)
    print(f"baseline={base:.4f} best={best.value:.4f} (trial {best.number}) params={best.params}")


if __name__ == "__main__":
    main()

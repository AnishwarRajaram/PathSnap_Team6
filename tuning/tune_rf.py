"""Bayesian optimisation of the Random Forest classifier with a reduced budget.

A full Random Forest fold takes about seven minutes, so each trial trains on a
fixed sample of whole training observations per fold (--rows, default 150,000
rows) and scores the full validation fold. Folds, metric and pruning match
tuning/tune_hgb.py: the fixed trip folds, mean CV macro F1 on reliable rows and a
median pruner. Trial 0 is the configuration of models/rf.py on the same sample,
so the comparison inside the study is fair. Student-test data is not read.

Outputs: results/tuning/rf_trials.csv, results/tuning/rf_best.json,
results/tuning/rf_history.png. Study stored in artifacts/tuning/rf.db.
Usage: PATHSNAP_DATA=<data root> python tuning/tune_rf.py --trials 30 --timeout 3600
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import rf  # noqa: E402
from models.harness import N_FOLDS, SEED, THRESHOLD, folds, load_rows, subsample_obs  # noqa: E402
from pathsnap.features import SELECTED  # noqa: E402

OUT = ROOT / "results" / "tuning"
DB = ROOT / "artifacts" / "tuning" / "rf.db"


def space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [None, 16, 24, 32]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced_subsample"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds")
    ap.add_argument("--rows", type=int, default=150_000, help="training rows sampled per fold")
    ap.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    ap.add_argument("--study", default="rf", help="study name; outputs are named after it")
    ap.add_argument("--jobs", type=int, default=6, help="cores per forest")
    a = ap.parse_args()
    FIXED = {"random_state": rf.CONFIG["random_state"], "n_jobs": a.jobs}

    train = load_rows("train")
    fid = train.trip_id.map(folds(train)).to_numpy()
    splits = []
    for k in range(N_FOLDS):
        tr = subsample_obs(train[fid != k], a.rows)
        va = train[fid == k]
        splits.append((tr[SELECTED].to_numpy(), tr.y.to_numpy(), va[SELECTED].to_numpy(), va.y.to_numpy()))

    def objective(trial: optuna.Trial) -> float:
        cfg = space(trial)
        scores = []
        for k, (Xtr, ytr, Xva, yva) in enumerate(splits):
            est = RandomForestClassifier(**cfg, **FIXED).fit(Xtr, ytr)
            p = est.predict_proba(Xva)[:, 1]
            scores.append(f1_score(yva, (p >= THRESHOLD).astype(int), average="macro"))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", [float(s) for s in scores])
        return float(np.mean(scores))

    DB.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=a.study, storage=f"sqlite:///{DB}", load_if_exists=True, direction="maximize",
        sampler=(optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6) if a.sampler == "tpe"
                 else optuna.samplers.RandomSampler(seed=SEED)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1))
    if not study.trials:
        study.enqueue_trial({"n_estimators": rf.CONFIG["n_estimators"], "max_features": rf.CONFIG["max_features"],
                             "min_samples_leaf": rf.CONFIG["min_samples_leaf"], "max_depth": None, "class_weight": None})
    t0 = time.time()
    study.optimize(objective, n_trials=max(a.trials - len(study.trials), 0), timeout=a.timeout,
                   callbacks=[lambda s, t: print(f"trial {t.number} {t.state.name} value={t.value} "
                                                 f"best={s.best_value:.4f} elapsed={time.time()-t0:.0f}s", flush=True)])

    OUT.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration", "user_attrs"))
    df.to_csv(OUT / f"{a.study}_trials.csv", index=False)
    done = df[df.state == "COMPLETE"]
    base, best = study.trials[0], study.best_trial
    diff = np.array(best.user_attrs["fold_scores"]) - np.array(base.user_attrs["fold_scores"])
    (OUT / f"{a.study}_best.json").write_text(json.dumps({
        "sampler": a.sampler, "objective": f"mean CV macro F1, 5 trip-grouped folds, {a.rows:,} sampled training rows per fold",
        "baseline_trial0_macro_f1": base.value, "best_macro_f1": best.value, "best_trial": best.number,
        "paired_fold_gain_mean": float(diff.mean()), "folds_improved": int((diff > 0).sum()),
        "best_params": best.params, "n_trials": len(df), "n_complete": len(done),
        "n_pruned": int((df.state == "PRUNED").sum()),
    }, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(done.number, done.value, s=18, label="completed trial")
    ax.plot(done.number, done.value.cummax(), c="#d62728", label="best so far")
    ax.axhline(base.value, ls="--", c="grey", label="models/rf.py configuration (trial 0)")
    ax.set(title=f"Random Forest Bayesian optimisation (TPE, {a.rows:,} rows per fold)",
           xlabel="trial", ylabel="CV macro F1"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / f"{a.study}_history.png", dpi=120)
    print(f"baseline={base.value:.4f} best={best.value:.4f} (trial {best.number}) gain on {int((diff > 0).sum())}/5 folds")


if __name__ == "__main__":
    main()

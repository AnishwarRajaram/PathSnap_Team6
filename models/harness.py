"""Shared evaluation harness for the PathSnap model comparison.

Every model is evaluated with the same folds, features, metrics and outputs.

Protocol (fixed before training, see docs/MODEL_REPORT_TEMPLATE.md):
  * Input: the 14 SELECTED features of contract v1.0. Unreliable observations are excluded.
  * Folds: results/models/folds.csv assigns each training trip to one of 5 folds (seed 0).
  * Primary metric: CV macro F1 over candidate rows at threshold 0.5.
  * Tie-break: CV log loss when macro F1 means differ by less than one fold standard deviation.
  * Reported: Top-1 accuracy, precision, recall, log loss, route continuity of the
    classifier-only top pick, student-test metrics, training and inference time.
  * Student-test results are never used for model or hyperparameter choice.

A model module defines NAME, FAMILY, CONFIG and build(config) -> estimator with
fit and predict_proba. Optional: MAX_TRAIN_ROWS (observation-level subsample of
each training set) and history(estimator, X_train, y_train, X_val, y_val) ->
DataFrame with an 'iteration' column and loss columns, for models that train
iteratively.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from pathsnap.data import load_trip
from pathsnap.features import SELECTED

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "artifacts" / "features"
MODELS_DIR = ROOT / "artifacts" / "models"
RESULTS = ROOT / "results" / "models"
FOLDS_CSV = RESULTS / "folds.csv"
N_FOLDS = 5
SEED = 0
THRESHOLD = 0.5
EPS = 1e-7


# ---------------------------------------------------------------- data

def load_rows(partition: str) -> pd.DataFrame:
    """Reliable candidate rows for a partition, built by models/prepare_data.py."""
    path = FEATURES_DIR / f"{partition}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python models/prepare_data.py")
    df = pd.read_pickle(path)
    df = df[df.reliable].reset_index(drop=True)
    df[SELECTED] = df[SELECTED].astype(np.float32)
    return df


def folds(train: pd.DataFrame) -> pd.Series:
    """Fold id per training trip. Created once, then read from results/models/folds.csv."""
    if FOLDS_CSV.exists():
        f = pd.read_csv(FOLDS_CSV, dtype={"trip_id": str})
    else:
        trips = np.array(sorted(train.trip_id.unique()))
        rng = np.random.default_rng(SEED)
        rng.shuffle(trips)
        f = pd.DataFrame({"trip_id": trips, "fold": np.arange(len(trips)) % N_FOLDS}).sort_values("trip_id")
        FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(FOLDS_CSV, index=False)
    return f.set_index("trip_id").fold


def subsample_obs(df: pd.DataFrame, max_rows: int | None, seed: int = SEED) -> pd.DataFrame:
    """Keep whole observations (all their candidates) until about max_rows rows remain."""
    if not max_rows or len(df) <= max_rows:
        return df
    keys = df[["trip_id", "obs_idx"]].drop_duplicates()
    frac = max_rows / len(df)
    keep = keys.sample(frac=frac, random_state=seed)
    return df.merge(keep, on=["trip_id", "obs_idx"]).reset_index(drop=True)


# ---------------------------------------------------------------- metrics

_ARC_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _arc_ends(partition: str, trip_id: str):
    key = (partition, trip_id)
    if key not in _ARC_CACHE:
        t = load_trip(partition, trip_id, with_route=False)
        _ARC_CACHE[key] = (t.arcs.u.to_numpy(), t.arcs.v.to_numpy())
    return _ARC_CACHE[key]


def top_pick(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Highest-probability candidate per observation, ordered by trip and observation."""
    d = df[["trip_id", "obs_idx", "arc_id", "y"]].assign(p=p)
    best = d.loc[d.groupby(["trip_id", "obs_idx"], sort=False).p.idxmax()]
    return best.sort_values(["trip_id", "obs_idx"])


def route_continuity(best: pd.DataFrame, partition: str) -> float:
    """Share of consecutive distinct predicted arcs where end node of one is start node of the next."""
    ok = total = 0
    for tid, g in best.groupby("trip_id"):
        arcs = g.arc_id.to_numpy()
        arcs = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
        if len(arcs) < 2:
            continue
        u, v = _arc_ends(partition, tid)
        ok += int(np.sum(v[arcs[:-1]] == u[arcs[1:]]))
        total += len(arcs) - 1
    return ok / total if total else float("nan")


def score(df: pd.DataFrame, p: np.ndarray, partition: str) -> dict:
    y = df.y.to_numpy()
    yhat = (p >= THRESHOLD).astype(int)
    best = top_pick(df, p)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]),
        "top1": float(best.y.mean()),
        "route_continuity": route_continuity(best, partition),
    }


# ---------------------------------------------------------------- run

def _predict(est, X: np.ndarray, n_obs: int):
    t0 = time.perf_counter()
    p = est.predict_proba(X)[:, 1]
    return p, (time.perf_counter() - t0) * 1000 / max(n_obs, 1) * 1000


def evaluate(module, skip_student_test: bool = False) -> dict:
    """Run grouped CV on train, refit on all train, score student_test, write outputs."""
    name, cfg = module.NAME, dict(module.CONFIG)
    max_rows = getattr(module, "MAX_TRAIN_ROWS", None)
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)

    train = load_rows("train")
    fold_of = folds(train)
    fid = train.trip_id.map(fold_of).to_numpy()

    rows, histories = [], []
    for k in range(N_FOLDS):
        tr = subsample_obs(train[fid != k], max_rows)
        va = train[fid == k].reset_index(drop=True)
        est = module.build(cfg)
        t0 = time.perf_counter()
        est.fit(tr[SELECTED].to_numpy(), tr.y.to_numpy())
        fit_s = time.perf_counter() - t0
        p, ms = _predict(est, va[SELECTED].to_numpy(), va[["trip_id", "obs_idx"]].drop_duplicates().shape[0])
        r = score(va, p, "train")
        r.update(fold=k, fit_s=fit_s, infer_ms_per_1k_obs=ms, train_rows=len(tr), val_rows=len(va))
        rows.append(r)
        print(f"[{name}] fold {k}: macro_f1={r['macro_f1']:.4f} top1={r['top1']:.4f} "
              f"log_loss={r['log_loss']:.4f} fit={fit_s:.1f}s", flush=True)
        if hasattr(module, "history"):
            h = module.history(est, tr[SELECTED].to_numpy(), tr.y.to_numpy(), va[SELECTED].to_numpy(), va.y.to_numpy())
            if h is not None:
                histories.append(h.assign(fold=k))
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv_folds.csv", index=False)

    summary = {"name": name, "family": module.FAMILY, "config": cfg, "max_train_rows": max_rows,
               "cv_mean": cv.drop(columns="fold").mean().to_dict(),
               "cv_std": cv.drop(columns="fold").std().to_dict()}

    if not skip_student_test:
        full = subsample_obs(train, max_rows)
        est = module.build(cfg)
        t0 = time.perf_counter()
        est.fit(full[SELECTED].to_numpy(), full.y.to_numpy())
        fit_s = time.perf_counter() - t0
        st = load_rows("student_test")
        p, ms = _predict(est, st[SELECTED].to_numpy(), st[["trip_id", "obs_idx"]].drop_duplicates().shape[0])
        s = score(st, p, "student_test")
        s.update(fit_s=fit_s, infer_ms_per_1k_obs=ms)
        summary["student_test"] = s
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": est, "features": SELECTED, "config": cfg}, MODELS_DIR / f"{name}.joblib")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix(st.y, (p >= THRESHOLD).astype(int)),
                               display_labels=["not true arc", "true arc"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{name}: student-test candidate rows")
        fig.tight_layout(); fig.savefig(out / "confusion_student_test.png", dpi=110); plt.close(fig)
        print(f"[{name}] student_test: macro_f1={s['macro_f1']:.4f} top1={s['top1']:.4f} log_loss={s['log_loss']:.4f}")

    if histories:
        h = pd.concat(histories)
        h.to_csv(out / "history.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        for col in [c for c in h.columns if c not in ("iteration", "fold")]:
            m = h.groupby("iteration")[col].mean()
            ax.plot(m.index, m.values, label=f"{col} (mean of folds)")
        ax.set(title=f"{name}: training history", xlabel="iteration", ylabel="log loss"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "history.png", dpi=110); plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    m, s = summary["cv_mean"], summary["cv_std"]
    print(f"[{name}] CV macro_f1={m['macro_f1']:.4f}±{s['macro_f1']:.4f} top1={m['top1']:.4f}±{s['top1']:.4f} "
          f"log_loss={m['log_loss']:.4f} continuity={m['route_continuity']:.4f}")
    return summary

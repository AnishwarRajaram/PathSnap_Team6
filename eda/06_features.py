"""EDA step 6: feature engineering and feature analysis.

Builds (observation, candidate) feature rows for every trip with a process pool,
attaches labels from step 4, and analyses features on TRAIN ONLY:
univariate AUC, mutual information, Spearman correlation, PCA, and grouped-CV
permutation importance from a quick gradient-boosting model.
Feature tables go to artifacts/features (gitignored). Aggregates and figures go to results/eda.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import list_partitions, list_trips, load_trip  # noqa: E402
from pathsnap.features import FEATURES, build  # noqa: E402

OUT = ROOT / "results" / "eda"; FIGS = OUT / "figs"
ART = ROOT / "artifacts"
WORKERS = 6
SEED = 0


def run(job):
    part, tid = job
    trip = load_trip(part, tid)
    df = build(trip)
    lab = np.load(ART / "labels" / part / f"{tid}.npz")
    df["y"] = (df.arc_id.to_numpy() == lab["arc_canon"][df.obs_idx.to_numpy()]).astype(np.int8)
    df["reliable"] = lab["reliable"][df.obs_idx.to_numpy()]
    df.insert(0, "trip_id", tid)
    return part, df


def top1(df, score):
    d = df.assign(s=score)
    best = d.loc[d.groupby(["trip_id", "obs_idx"]).s.idxmax()]
    return float(best.y.mean())


def main():
    t0 = time.perf_counter()
    jobs = [(p, t) for p in list_partitions() for t in list_trips(p)]
    parts = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for p, df in ex.map(run, jobs):
            parts.setdefault(p, []).append(df)
    (ART / "features").mkdir(parents=True, exist_ok=True)
    tables = {}
    for p, dfs in parts.items():
        tables[p] = pd.concat(dfs, ignore_index=True)
        tables[p].to_pickle(ART / "features" / f"{p}.pkl")
    print(f"Feature build: {time.perf_counter()-t0:.1f}s")
    for p, d in tables.items():
        r = d[d.reliable]
        obs = r.groupby(["trip_id", "obs_idx"]).y.max()
        print(f"[{p}] rows={len(d):,} reliable rows={len(r):,} positive share={r.y.mean():.3%} "
              f"obs with true arc in candidates={obs.mean():.3%}")

    tr = tables["train"]; tr = tr[tr.reliable].reset_index(drop=True)
    X, y, g = tr[FEATURES].astype(float), tr.y.to_numpy(), tr.trip_id.to_numpy()
    rng = np.random.default_rng(SEED)
    samp = rng.choice(len(tr), size=min(300_000, len(tr)), replace=False)

    # 1. Univariate: medians by class, AUC, mutual information.
    auc = {f: roc_auc_score(y, X[f]) for f in FEATURES}
    mi = mutual_info_classif(X.iloc[samp], y[samp], random_state=SEED,
                             discrete_features=[X[f].nunique() < 20 for f in FEATURES])
    uni = pd.DataFrame({
        "median_true": X[y == 1].median(), "median_false": X[y == 0].median(),
        "auc": pd.Series(auc), "auc_strength": pd.Series(auc).sub(.5).abs().add(.5),
        "mutual_info": pd.Series(mi, index=FEATURES),
    }).sort_values("auc_strength", ascending=False)
    pd.set_option("display.width", 160)
    print("\nUnivariate (train, reliable rows):\n", uni.round(4).to_string())

    # 2. Spearman correlation.
    corr = X.iloc[samp].corr(method="spearman")
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]
             if abs(corr.loc[a, b]) >= 0.8]
    print("\nPairs with |Spearman| >= 0.8:")
    for a, b, c in sorted(pairs, key=lambda x: -abs(x[2])):
        print(f"  {a:28s} {b:28s} {c:+.3f}")

    # 3. PCA on standardised features.
    Z = StandardScaler().fit_transform(X.iloc[samp])
    pca = PCA(random_state=SEED).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print("\nPCA cumulative variance: " + " ".join(f"PC{i+1}={c:.3f}" for i, c in enumerate(cum[:10])))
    print("Components for 90/95/99% variance: "
          + "/".join(str(int(np.searchsorted(cum, q) + 1)) for q in (.9, .95, .99)))
    load = pd.DataFrame(pca.components_[:3].T, index=FEATURES, columns=["PC1", "PC2", "PC3"])
    print("Top loadings:\n" + "\n".join(
        f"  {c}: " + ", ".join(f"{f}({load.loc[f, c]:+.2f})" for f in load[c].abs().sort_values(ascending=False).index[:4])
        for c in load.columns))

    # 4. Quick grouped CV with gradient boosting, plus permutation importance.
    gkf = GroupKFold(n_splits=5)
    folds, imp = [], []
    for k, (a, b) in enumerate(gkf.split(X, y, g)):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)
        m.fit(X.iloc[a], y[a])
        p = m.predict_proba(X.iloc[b])[:, 1]
        vb = tr.iloc[b]
        folds.append(dict(fold=k, top1=top1(vb, p), top1_nearest=top1(vb, -vb.dist_m.to_numpy()),
                          macro_f1=f1_score(y[b], p >= .5, average="macro"), log_loss=log_loss(y[b], p)))
        if k < 2:
            sb = rng.choice(b, size=min(60_000, len(b)), replace=False)
            pi = permutation_importance(m, X.iloc[sb], y[sb], scoring="neg_log_loss",
                                        n_repeats=3, random_state=SEED, n_jobs=1)
            imp.append(pi.importances_mean)
    cv = pd.DataFrame(folds)
    print("\nGrouped 5-fold CV, HistGradientBoosting (default-ish):\n", cv.round(4).to_string(index=False))
    print("mean:", cv.drop(columns="fold").mean().round(4).to_dict())
    imp = pd.Series(np.mean(imp, axis=0), index=FEATURES).sort_values(ascending=False)
    print("\nPermutation importance (log-loss increase, 2 folds):\n", imp.round(5).to_string())

    uni.join(imp.rename("perm_importance")).to_csv(OUT / "feature_analysis.csv")
    cv.to_csv(OUT / "feature_quick_cv.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    uni.auc_strength.sort_values().plot.barh(ax=ax[0, 0], title="Univariate AUC strength (train)")
    ax[0, 0].axvline(.5, c="grey", ls="--")
    im = ax[0, 1].imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax[0, 1].set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=7)
    ax[0, 1].set_yticks(range(len(FEATURES)), FEATURES, fontsize=7)
    ax[0, 1].set_title("Spearman correlation"); fig.colorbar(im, ax=ax[0, 1])
    ax[1, 0].plot(range(1, len(cum) + 1), cum, marker="o"); ax[1, 0].axhline(.95, c="grey", ls="--")
    ax[1, 0].set(title="PCA cumulative explained variance", xlabel="components", ylabel="share")
    imp.sort_values().plot.barh(ax=ax[1, 1], title="Permutation importance (log-loss, grouped CV)")
    fig.tight_layout(); fig.savefig(FIGS / "06_features.png", dpi=110)
    print(f"\nTotal {time.perf_counter()-t0:.0f}s. Wrote feature_analysis.csv, feature_quick_cv.csv, figs/06_features.png")


if __name__ == "__main__":
    main()

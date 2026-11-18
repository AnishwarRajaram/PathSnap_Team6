"""EDA step 8: PCA elbow (scree) plot for the engineered features.

Fits PCA on standardised training rows (reliable observations, 300,000-row sample,
seed 0) for all 20 engineered features and for the 14 selected features.
Marks the elbow with the maximum distance to the chord of the scree curve and the
components needed for 90% and 95% of variance.
Writes results/eda/figs/08_pca_elbow.png. Prints aggregates only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.features import FEATURES, SELECTED  # noqa: E402

SEED = 0
SAMPLE = 300_000


def elbow(ratio: np.ndarray) -> int:
    """Index (1-based) of the point farthest from the line joining the first and last points."""
    x = np.arange(1, len(ratio) + 1, dtype=float)
    p1, p2 = np.array([x[0], ratio[0]]), np.array([x[-1], ratio[-1]])
    v, w = p2 - p1, np.c_[x, ratio] - p1
    d = np.abs(v[0] * w[:, 1] - v[1] * w[:, 0]) / np.linalg.norm(v)
    return int(np.argmax(d) + 1)


def main():
    tr = pd.read_pickle(ROOT / "artifacts" / "features" / "train.pkl")
    tr = tr[tr.reliable]
    tr = tr.sample(n=min(SAMPLE, len(tr)), random_state=SEED)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for cols, label, c in ((FEATURES, "all 20 features", "#1f4fd1"), (SELECTED, "14 selected features", "#d62728")):
        Z = StandardScaler().fit_transform(tr[cols].astype(float))
        r = PCA(random_state=SEED).fit(Z).explained_variance_ratio_
        cum = np.cumsum(r)
        k = elbow(r)
        k90, k95 = int(np.searchsorted(cum, .90) + 1), int(np.searchsorted(cum, .95) + 1)
        x = np.arange(1, len(r) + 1)
        ax[0].plot(x, r, marker="o", c=c, label=f"{label} (elbow at PC{k})")
        ax[0].scatter([k], [r[k - 1]], s=160, facecolors="none", edgecolors=c, linewidths=2)
        ax[1].plot(x, cum, marker="o", c=c, label=f"{label}: 90% at {k90}, 95% at {k95}")
        print(f"{label}: PC1={r[0]:.3f} elbow=PC{k} cum@elbow={cum[k-1]:.3f} 90%->{k90} 95%->{k95} of {len(r)}")
    ax[0].set(title="Scree plot: variance explained per component", xlabel="principal component",
              ylabel="explained variance ratio")
    ax[1].axhline(.90, ls="--", c="grey"); ax[1].axhline(.95, ls=":", c="grey")
    ax[1].set(title="Cumulative explained variance", xlabel="number of components", ylabel="cumulative ratio",
              ylim=(0, 1.02))
    for a in ax:
        a.set_xticks(range(1, len(FEATURES) + 1))
        a.grid(alpha=.3); a.legend()
    fig.suptitle("PCA on standardised training features (reliable rows, 300k sample)")
    fig.tight_layout()
    out = ROOT / "results" / "eda" / "figs" / "08_pca_elbow.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

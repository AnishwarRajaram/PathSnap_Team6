"""Model 2: HistGradientBoosting (boosting family)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

NAME = "hgb"
FAMILY = "Boosting"
CONFIG = {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20,
          "l2_regularization": 0.0, "early_stopping": False, "class_weight": None}
HISTORY_EVERY = 10
HISTORY_ROWS = 200_000


def build(cfg):
    return HistGradientBoostingClassifier(random_state=0, **cfg)


def history(est, Xtr, ytr, Xva, yva):
    """Real per-iteration log loss on a fixed sample of the training and validation folds."""
    rng = np.random.default_rng(0)
    it = rng.choice(len(Xtr), size=min(HISTORY_ROWS, len(Xtr)), replace=False)
    iv = rng.choice(len(Xva), size=min(HISTORY_ROWS, len(Xva)), replace=False)
    rows = []
    for i, (pt, pv) in enumerate(zip(est.staged_predict_proba(Xtr[it]), est.staged_predict_proba(Xva[iv])), 1):
        if i % HISTORY_EVERY == 0 or i == 1:
            rows.append({"iteration": i, "train_log_loss": log_loss(ytr[it], pt[:, 1], labels=[0, 1]),
                         "val_log_loss": log_loss(yva[iv], pv[:, 1], labels=[0, 1])})
    return pd.DataFrame(rows)

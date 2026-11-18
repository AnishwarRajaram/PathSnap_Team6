"""Model 4: Multilayer Perceptron (neural network family).

Runs through the shared harness unchanged:
    PATHSNAP_DATA=<data root> python models/run.py mlp

Training loop. scikit-learn's MLPClassifier stores only the training loss per
epoch. The guidelines ask for training and validation loss curves, so the network
is trained one epoch at a time with partial_fit (Adam state carries over between
epochs). After every epoch the log loss is measured on a fixed held-out share of
the training rows. Training stops when that validation loss has not improved by
more than `tol` for `n_iter_no_change` epochs, and the weights of the best epoch
are restored.

Leakage. The held-out rows come from the training fold that the harness passes to
fit. The harness validation fold and the student-test partition are never seen
during fitting. StandardScaler sits in the Pipeline, so it is fitted on the
training rows only and is saved with the model (contract section 6).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "mlp"
FAMILY = "Neural network"

# Every hyperparameter, including the seed, so that summary.json records all of them.
CONFIG = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,             # L2 penalty
    "batch_size": 2048,
    "learning_rate_init": 0.001,
    "max_iter": 100,             # maximum epochs
    "validation_fraction": 0.1,  # held-out share of the training rows for early stopping
    "n_iter_no_change": 10,      # patience in epochs
    "tol": 0.0001,               # minimum validation log-loss improvement
    "random_state": 0,
}

EPS = 1e-7
HISTORY_ROWS = 200_000  # training rows scored after each epoch for the training-loss curve


class EpochMLP(ClassifierMixin, BaseEstimator):
    """MLPClassifier trained epoch by epoch with validation-loss early stopping.

    After fit: `history_` holds one row per epoch with train_log_loss (on a fixed
    sample of up to HISTORY_ROWS training rows) and val_log_loss (on the held-out
    rows), both measured at the end of the epoch. `best_epoch_` is the restored epoch.
    """

    def __init__(self, hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.0001,
                 batch_size=2048, learning_rate_init=0.001, max_iter=100, validation_fraction=0.1,
                 n_iter_no_change=10, tol=0.0001, random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(X))
        n_val = int(round(self.validation_fraction * len(X)))
        va, tr = idx[:n_val], idx[n_val:]
        tr_eval = tr[:HISTORY_ROWS]  # fixed random sample of the training rows for the loss curve

        net = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver,
            alpha=self.alpha, batch_size=self.batch_size, learning_rate_init=self.learning_rate_init,
            shuffle=True, random_state=self.random_state)
        classes = np.array([0, 1])
        rows, best, best_state, wait = [], np.inf, None, 0
        for epoch in range(1, self.max_iter + 1):
            net.partial_fit(X[tr], y[tr], classes=classes)
            train_ll = self._log_loss(net, X[tr_eval], y[tr_eval])
            val_ll = self._log_loss(net, X[va], y[va])
            if not np.isfinite(train_ll) or not np.isfinite(val_ll):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            rows.append({"iteration": epoch, "train_log_loss": train_ll, "val_log_loss": val_ll})
            if val_ll < best - self.tol:
                best, best_state, wait = val_ll, copy.deepcopy((net.coefs_, net.intercepts_, epoch)), 0
            else:
                wait += 1
                if wait >= self.n_iter_no_change:
                    break
        net.coefs_, net.intercepts_, self.best_epoch_ = best_state
        self.net_ = net
        self.n_epochs_run_ = len(rows)
        self.history_ = pd.DataFrame(rows)
        self.classes_ = net.classes_
        return self

    @staticmethod
    def _log_loss(net, X, y):
        return log_loss(y, np.clip(net.predict_proba(X)[:, 1], EPS, 1 - EPS), labels=[0, 1])

    def predict_proba(self, X):
        return self.net_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return self.net_.predict(np.asarray(X, dtype=np.float32))


def build(cfg):
    """Return an unfitted scaler + MLP pipeline that exposes predict_proba."""
    return make_pipeline(StandardScaler(), EpochMLP(**cfg))


def history(est, X_train, y_train, X_val, y_val):
    """Per-epoch training and held-out validation log loss recorded during fit."""
    return est[-1].history_

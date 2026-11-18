"""Model N: <algorithm> (<family>). Owner: <teammate>.

Copy this file to models/<name>.py, fill in NAME, FAMILY, CONFIG and build, then run:
    PATHSNAP_DATA=<data root> python models/run.py <name>
Do not change models/harness.py, the folds file or the feature list.
"""
NAME = "<name>"            # short id, used for the results folder
FAMILY = "<family>"        # e.g. "Bagging ensemble", "Neural network", "Instance-based"
CONFIG = {}                # every hyperparameter you set, with its value
# MAX_TRAIN_ROWS = 100_000  # set only if the model cannot fit all training rows


def build(cfg):
    """Return an unfitted estimator with fit(X, y) and predict_proba(X).
    Wrap it in a Pipeline with a scaler if the algorithm is scale-sensitive."""
    raise NotImplementedError


# Optional, only for models that train iteratively (for example MLP):
# def history(est, X_train, y_train, X_val, y_val):
#     """Return a DataFrame with an 'iteration' column and real loss columns."""

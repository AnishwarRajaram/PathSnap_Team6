"""Random Forest with the configuration selected by Bayesian optimisation (results/tuning/rf_best.json)."""
from sklearn.ensemble import RandomForestClassifier

NAME = "rf_tuned"
FAMILY = "Ensemble (bagging of decision trees)"
CONFIG = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "max_depth": None,
    "class_weight": "balanced_subsample",
    "random_state": 2026,
    "n_jobs": -1,
}


def build(cfg):
    return RandomForestClassifier(**cfg)

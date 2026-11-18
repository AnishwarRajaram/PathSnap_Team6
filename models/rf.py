"""Random Forest classifier using bagging of decision trees."""

from sklearn.ensemble import RandomForestClassifier


NAME = "rf"
FAMILY = "Ensemble (bagging of decision trees)"
CONFIG = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "random_state": 2026,
    "n_jobs": -1,
}


def build(cfg):
    """Build an unfitted forest that provides candidate probabilities."""
    # Tree splits do not depend on feature scale, so no scaler is needed.
    return RandomForestClassifier(**cfg)

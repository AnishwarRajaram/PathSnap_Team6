"""Model 1: Logistic Regression (linear and probabilistic family)."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "logreg"
FAMILY = "Linear / probabilistic"
CONFIG = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "class_weight": None}


def build(cfg):
    return make_pipeline(StandardScaler(), LogisticRegression(random_state=0, **cfg))

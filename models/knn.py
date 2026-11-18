"""Model 5: Distance-weighted kNN (instance-based). Owner: Teammate 3."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "knn"
FAMILY = "Instance-based"
CONFIG = {"n_neighbors": 15, "weights": "distance", "n_jobs": 4}
MAX_TRAIN_ROWS = 100_000


def build(cfg):
    """Return an unfitted pipeline that predicts candidate-arc probabilities."""
    # Fit the scaler with each training fold so feature units do not dominate kNN distances.
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**cfg))

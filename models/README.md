# Model Comparison Workflow

## 1. Assignment

| Model | Family | Owner | File |
|---|---|---|---|
| Logistic Regression | Linear and probabilistic | [Lead pair] | `models/logreg.py` |
| HistGradientBoosting | Boosting | [Lead pair] | `models/hgb.py` |
| Random Forest | Bagging ensemble | [Teammate 1] | `models/rf.py` |
| Multilayer Perceptron | Neural network | [Teammate 2] | `models/mlp.py` |
| Distance-weighted kNN | Instance-based | [Teammate 3] | `models/knn.py` |

## 2. Setup

1. Check out the `anishwar` branch.
2. Create the environment: `python3 -m venv .venv` and `.venv/bin/pip install -r requirements.txt`.
3. Place the issued student folder outside version control and set `PATHSNAP_DATA` to its path.
4. Build labels and features: `.venv/bin/python models/prepare_data.py`. The command takes about 10 seconds.

## 3. Adding a Model

1. Copy `models/_template.py` to `models/<name>.py`.
2. Set `NAME`, `FAMILY` and `CONFIG`. Implement `build`. The estimator must provide `predict_proba`.
3. Set `MAX_TRAIN_ROWS` only if the algorithm cannot fit the full training set. kNN uses 100,000.
4. Implement `history` only if the algorithm trains iteratively. Report real losses only.
5. Run `.venv/bin/python models/run.py <name>`.
6. Run `.venv/bin/python models/compare.py` to refresh the comparison table.
7. Write the model report from `docs/MODEL_REPORT_TEMPLATE.md`.

## 4. Fixed Rules

4.1. `models/harness.py`, `results/models/folds.csv` and the feature list are shared. Changes need team agreement.

4.2. The primary metric is cross-validated macro F1. Ties within one fold standard deviation go to the lower log loss.

4.3. Student-testing results are never used to choose a model or a hyperparameter.

4.4. Trained models are written to `artifacts/models/` and are not committed.

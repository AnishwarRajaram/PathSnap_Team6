# Model Report: Distance-Weighted k-Nearest Neighbours

| Item | Value |
|---|---|
| Model | Distance-weighted k-Nearest Neighbours (kNN) |
| Algorithm family | Instance-based |
| Owner | [Team Name / Author Name] |
| Code | `models/knn.py` |
| Results | `results/models/knn/` |

## 1. Evaluation Protocol

kNN was evaluated with the shared protocol in `models/harness.py`. Its input is the 14 selected features from input contract version 1.1, in the fixed contract order. Unreliable observations are excluded. The harness uses grouped five-fold cross-validation with the fixed trip assignments in `results/models/folds.csv`; each fold holds out complete trips. The primary metric is candidate-row macro F1 at threshold 0.5. Log loss is the tie-breaker when macro F1 results are within one fold standard deviation.

The reported kNN results are from cross-validation on the training partition. Student-testing was skipped, so no student-testing results were used for model selection or are reported here.

## 2. Model Understanding

kNN scores a candidate by finding the 15 nearest training examples in the scaled feature space. With distance weighting, nearer examples have more influence on the class vote than farther examples. The positive-class vote is returned as a probability through `predict_proba`.

This is an instance-based method: it keeps the training examples for neighbour lookup rather than learning model weights through gradient-based training. It can represent local, non-linear patterns, but prediction can be expensive because it must search the stored examples. Scaling is important because the 14 features use different units and ranges.

## 3. Training and Configuration

The saved configuration in `summary.json` is:

| Setting | Value |
|---|---:|
| Neighbours (`n_neighbors`) | 15 |
| Weighting (`weights`) | `distance` |
| Parallel neighbour search (`n_jobs`) | 4 |
| Training-row target (`MAX_TRAIN_ROWS`) | 100,000 |
| Preprocessing | `StandardScaler` in a scikit-learn `Pipeline` |

The scaler is fitted as part of the estimator on each fold's training data, then applied to that fold's validation data. No scaler statistics are learned from the held-out fold. The harness samples whole observations together with their candidate rows, so the realized row count can vary slightly around the 100,000 target; the saved summary reports a mean of about 99,644 training rows per fold.

kNN does not minimize a gradient-based loss during training. It stores the scaled examples and uses distance-weighted neighbour voting when it predicts.

## 4. Experimental Results

The values below are means and standard deviations across the five training cross-validation folds, copied from `results/models/knn/summary.json`.

| Metric | CV mean ± fold standard deviation |
|---|---:|
| Macro F1 | 0.9151 ± 0.0315 |
| Top-1 accuracy | 0.8620 ± 0.0683 |
| Log loss | 0.1480 ± 0.0384 |
| Precision | 0.8759 ± 0.0513 |
| Recall | 0.8259 ± 0.0698 |
| Route continuity | 0.7240 ± 0.0962 |
| Inference time (ms per 1,000 observations) | 3,093 ± 765 |

The mean fit time was about 0.36 seconds per fold. Student-testing was skipped: student-testing metrics and a student-testing confusion matrix are not available.

For context, the supplied baseline CV macro F1 values are 0.932 for HistGradientBoosting and 0.908 for Logistic Regression. The kNN score of 0.9151 is between those two reported values; this comparison is limited to macro F1 and does not establish that kNN is better than every model.

The fold scores vary, but these results alone are not sufficient to conclude that kNN overfits or underfits. No training-history curves are produced for this non-iterative model.

## 5. Visualisations

No visualisations are available in `results/models/knn/`; the saved files are `summary.json` and `cv_folds.csv`. In particular, student-testing was skipped, so the student-testing confusion matrix is unavailable. No iteration-wise history plot is applicable to kNN.

## 6. Observations

- The mean inference time is approximately 3,093 ms per 1,000 observations. This is a prediction-cost measure and is not directly comparable to fit time per fold.
- The macro F1 standard deviation across folds is 0.0315, so performance varies between held-out trip groups.
- The summary does not identify particular error cases, such as reverse-twin arcs, junctions, or GPS gaps; no such failure case is claimed here.
- The student-testing partition was not evaluated for this report, and its results must not be used to tune this model.

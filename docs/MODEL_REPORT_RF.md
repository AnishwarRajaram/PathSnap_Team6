# Model Report: Random Forest

| Item | Value |
|---|---|
| Model | Random Forest (`rf`) |
| Algorithm family | Ensemble (bagging of decision trees) |
| Owner | [Team Name / Author Name] |
| Code | `models/rf.py` |
| Results | `results/models/rf/` |

## 1. Evaluation Protocol

All models use the shared harness, the 14 selected float32 features, reliable
candidate rows, and the fixed grouped five-fold assignment in
`results/models/folds.csv`. The primary metric is cross-validated macro F1 at
threshold 0.5. Student-testing results are reported only and are not used to
choose or tune the model.

## 2. Model Understanding

Random Forest is a bagging ensemble: each tree is trained on a bootstrap sample
and uses a random feature subset when considering splits. It can represent
nonlinear interactions among candidate distance, heading, speed, and local
network features, while averaging trees reduces the variance of a single tree.

Tree splits do not depend on feature scales, so the model receives the feature
matrix without a scaler. The ensemble is more expensive to train and store than
the linear baseline. The measured student-test probabilities include many
exact zeros and values on the 1/300 grid; this is discussed in Section 6.

## 3. Training and Configuration

The initial configuration is copied from `models/rf.py`:

| Hyperparameter | Initial value |
|---|---:|
| `n_estimators` | 300 |
| `max_features` | `"sqrt"` |
| `min_samples_leaf` | 2 |
| `random_state` | 2026 |
| `n_jobs` | -1 |

The classifier uses the default Gini-impurity split criterion. No scaling,
training-row cap, or class weighting is applied. The harness fits on the
training folds and refits on all reliable training rows before reporting
student-testing metrics.

## 4. Experimental Results

### 4.1. Random Forest

Values below are copied from `results/models/rf/summary.json`.

| Metric | Cross-validation (mean ± fold SD) | Student-test |
|---|---:|---:|
| Macro F1 | 0.9335 ± 0.0304 | 0.9315 |
| Top-1 accuracy | 0.8821 ± 0.0638 | 0.8816 |
| Log loss | 0.0720 ± 0.0216 | 0.0667 |
| Precision | 0.9046 ± 0.0462 | 0.8993 |
| Recall | 0.8617 ± 0.0680 | 0.8578 |
| Route continuity | 0.7582 ± 0.0917 | 0.7418 |

| Timing | Value |
|---|---:|
| Mean CV fit time per fold | 421.23 s |
| CV fit-time standard deviation | 49.24 s |
| Full-training refit time | 544.06 s |
| Student-test inference time (ms per 1,000 observations) | 256.58 |

### 4.2. Reference Model Comparison

The values below are from the existing HGB and Logistic Regression summary
files. They are reported for context and were not used to tune Random Forest.

| Model | CV Macro F1 (mean ± SD) | CV Log Loss | Student Macro F1 | Student Top-1 | Student Log Loss | Refit (s) | Inference (ms/1k observations) |
|---|---:|---:|---:|---:|---:|---:|---:|
| HGB | 0.9316 ± 0.0305 | 0.0803 | 0.9329 | 0.8850 | 0.0692 | 8.768 | 14.927 |
| Logistic Regression | 0.9080 ± 0.0354 | 0.1160 | 0.9106 | 0.8666 | 0.1027 | 0.735 | 0.118 |
| Random Forest | 0.9335 ± 0.0304 | 0.0720 | 0.9315 | 0.8816 | 0.0667 | 544.057 | 256.578 |

Random Forest's CV Macro F1 exceeds HGB by 0.0019, which is within the larger
fold standard deviation (0.0304). The shared tie rule therefore compares CV Log
Loss; Random Forest is lower (0.0720 versus 0.0803), so `compare.py` selects RF.
Its student-test Macro F1 and Top-1 are slightly below HGB, while its student
Log Loss is lower. RF exceeds Logistic Regression on the listed CV and
student-test metrics, but costs substantially more to train and infer.
Student-test metrics remain reporting-only.

### 4.3. Generalisation Evidence

CV Macro F1 (0.9335) and student-test Macro F1 (0.9315) are close, with a
student-test decrease of about 0.0021. This does not show a large generalization
gap. However, the harness does not record training-set scores for each fold, so
overfitting within the training data cannot be ruled out from this comparison
alone. Fold variability is material (Macro F1 SD 0.0304).

## 5. Visualisations

The shared harness produces the student-test confusion matrix at
`results/models/rf/confusion_student_test.png`.

Random Forest does not expose an iteration-wise training history in this
implementation, so no history plot is produced.

## 6. Observations

### 6.1. Probability Resolution and HMM Emissions

The aggregate-only audit in `results/models/rf/probability_diagnostics.csv`
covered 169,408 reliable student-test candidate rows across 19,133 observation
groups. It found 89,875 exact zeros (53.05%) and 295 exact ones (0.17%), with
34,221 distinct probability values. About 55.48% of probabilities fell on the
1/300 grid. No observation group had an all-zero candidate probability sum;
15,192 groups had at least one zero-probability candidate.

`predict_proba` averages tree leaf class probabilities. When leaves are pure,
these can behave like coarse vote fractions and can include exact zero or one;
mixed leaves need not produce exact multiples of 1/300. The shared harness clips
probabilities to `[1e-7, 1 - 1e-7]` for Log Loss. For HMM emissions, probabilities
are normalized across each observation's candidates: exact zeros remain zero,
so those candidates would receive zero emission mass. No all-zero candidate
groups were observed in the student-test audit, so normalization had positive
mass for every audited group. This does not establish that zero emissions are
harmless for sequence decoding; a later HMM may need a small emission floor.
No HMM changes are made in this model task.

The RF produced the highest CV Macro F1 among the three compared models and
lower CV/student Log Loss than either reference model. HGB had slightly higher
student Macro F1 and Top-1, and much shorter training and inference times.
Logistic Regression was much faster but had lower F1 and higher Log Loss.
RF's measured exact-zero share and slower inference are notable limitations.
All route-based labels and scores remain subject to the contract's caveat that
the reference route is not verified ground truth.

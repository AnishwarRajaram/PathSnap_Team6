# Model Report: Multilayer Perceptron

| Item | Value |
|---|---|
| Model | Multilayer Perceptron (`mlp`) |
| Algorithm family | Neural network |
| Owner | Team 6 |
| Code | `models/mlp.py` |
| Results | `results/models/mlp/` |

## 1. Evaluation Protocol

The MLP runs through the shared harness `models/harness.py` without changes, using `python models/run.py mlp`. The input is the 14 `SELECTED` features of contract version 1.1, cast to float32, on reliable candidate rows only. Cross-validation uses the fixed trip-grouped five-fold assignment in `results/models/folds.csv`, so no trip appears in both a training and a validation fold. The primary metric is cross-validated macro F1 over candidate rows at threshold 0.5. Ties within one fold standard deviation are broken by the lower log loss. Student-testing results are reported only and were not used to choose or tune any setting.

## 2. Model Understanding

2.1. Candidate scoring depends on interactions between features. A small distance matters less when the heading disagrees with the arc bearing, and the reverse-twin arcs that make up about 93% of the network can only be separated by heading. An MLP with ReLU units learns such interactions without hand-made cross terms and outputs a probability for each candidate, which the HMM uses as its emission score.

2.2. Strengths seen in the results:
- **Best probability quality in cross-validation:** CV log loss is 0.0792, against 0.0803 for HGB and 0.1160 for logistic regression.
- **Most connected classifier-only picks:** CV route continuity is 0.764, against 0.737 for HGB.
- **Fast inference:** 2.5 ms per 1,000 observations, against 14.9 ms for HGB.

Limitations seen in the results:
- **Slow training on the CPU:** fitting takes 256 to 475 s per fold, against about 7 s for HGB. scikit-learn's MLP has no GPU support.
- **Lower recall than precision:** CV recall is 0.845 and precision 0.903, so at threshold 0.5 the model misses some true arcs rather than over-predicting them.

## 3. Training and Configuration

3.1. Hyperparameters, copied from `summary.json`:

| Hyperparameter | Value |
|---|---|
| Hidden layer sizes | (64, 32) |
| Activation | ReLU |
| Optimiser | Adam |
| L2 penalty `alpha` | 0.0001 |
| Batch size | 2048 |
| Initial learning rate | 0.001 |
| Maximum epochs `max_iter` | 100 |
| Held-out share for early stopping | 0.1 |
| Patience `n_iter_no_change` | 10 epochs |
| Minimum improvement `tol` | 0.0001 |
| Seed `random_state` | 0 |

3.2. The training objective is binary cross-entropy with an L2 penalty, minimised by Adam. scikit-learn's `MLPClassifier` records only the training loss, so `models/mlp.py` trains one epoch at a time with `partial_fit`, and the Adam state carries over between epochs. After each epoch it measures log loss on a fixed sample of up to 200,000 training rows and on a held-out 10% of the training rows. Training stops once the held-out loss fails to improve by more than `tol` for 10 epochs. The weights of the last epoch that improved by more than `tol` are then restored.

3.3. `StandardScaler` sits inside the Pipeline, so it is fitted on each training fold only and is saved with the model. No training-row cap and no class weighting are applied. The early-stopping hold-out comes from the training fold that the harness passes to `fit`. The harness validation fold and the student-test partition never influence training.

## 4. Experimental Results

4.1. Cross-validation, mean and standard deviation over 5 trip-grouped folds:

| Metric | CV mean | Fold std |
|---|---:|---:|
| Macro F1 | 0.9280 | 0.0316 |
| Top-1 accuracy | 0.8798 | 0.0639 |
| Log loss | 0.0792 | 0.0236 |
| Precision | 0.9035 | 0.0449 |
| Recall | 0.8446 | 0.0768 |
| Route continuity (classifier-only top pick) | 0.7643 | 0.0906 |
| Fit time per fold (s) | 359.8 | 92.3 |

Per fold, macro F1 is 0.8945, 0.9419, 0.9637, 0.8946 and 0.9451. Folds 0 and 3 are the weakest for all three models in the comparison. The spread therefore follows the trips in each fold, not the MLP.

4.2. Student-testing partition, model refitted on all 80 training trips:

| Metric | Student-test |
|---|---:|
| Macro F1 | 0.9252 |
| Top-1 accuracy | 0.8727 |
| Log loss | 0.0724 |
| Precision | 0.8976 |
| Recall | 0.8379 |
| Route continuity (classifier-only top pick) | 0.7569 |
| Fit time (s) | 412.1 |
| Inference time (ms per 1,000 observations) | 2.51 |

4.3. Overfitting and underfitting. The model does not overfit:
- **Train against validation:** at the restored epoch, held-out loss exceeds training loss by between 0.0001 and 0.0026, depending on the fold.
- **CV against student-test:** CV macro F1 is 0.9280 and student-test 0.9252, a gap far below the fold standard deviation of 0.032.

The model is slightly under-trained:
- **Epoch cap reached:** four of five folds and the final refit stopped at the 100-epoch limit, not through early stopping. Only fold 4 stopped early, at epoch 81, restoring epoch 71.
- **Loss still falling:** held-out loss was still decreasing slowly at epoch 100 (see `history.png`), so a larger epoch budget could improve it a little. That belongs to the Bayesian optimisation stage, on training data only.

## 5. Comparison with the Other Models

The table comes from `models/compare.py` on `main`, run unchanged over the `logreg`, `hgb` and `knn` summaries on `main` (commit `afd6dbe`) and the MLP summary on this branch. Random Forest is not yet on `main`. kNN was run with `--skip-student-test`, so it has no student-test values.

| Model | Family | CV macro F1 | CV Top-1 | CV log loss | CV route continuity | ST macro F1 | ST Top-1 | ST log loss | Inference (ms per 1k obs) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | Boosting | **0.9316** ± 0.0305 | 0.8795 | 0.0803 | 0.7368 | **0.9329** | **0.8850** | **0.0692** | 14.90 |
| MLP | Neural network | 0.9280 ± 0.0316 | **0.8798** | **0.0792** | **0.7643** | 0.9252 | 0.8727 | 0.0724 | 2.51 |
| Distance-weighted kNN (100,000-row cap) | Instance-based | 0.9151 ± 0.0315 | 0.8620 | 0.1480 | 0.7240 | n/a | n/a | n/a | n/a |
| Logistic Regression | Linear / probabilistic | 0.9080 ± 0.0354 | 0.8693 | 0.1160 | 0.7185 | 0.9106 | 0.8666 | 0.1027 | 0.12 |

Among the four models in this table HGB has the highest CV macro F1 and the MLP trails it by 0.0036, within one fold standard deviation. The five-model comparison in `results/models/comparison.csv` adds Random Forest, which has the highest CV macro F1 and the lowest log loss and is selected.

## 6. Visualisations

6.1. `results/models/mlp/confusion_student_test.png`: candidate-row confusion matrix on the student-testing partition.

6.2. `results/models/mlp/history.png`: training and held-out log loss per epoch, as the mean over the five folds. Fold 4 stopped at epoch 81, so later points average four folds. This causes the small step at epoch 81. The per-fold values are in `history.csv`.

## 7. Observations

7.1. The MLP's main gain over HGB is in route continuity: its classifier-only top picks join up more often (0.764 against 0.737 in CV). That suggests smoother probabilities between neighbouring arcs, which should help the HMM.

7.2. The MLP's recall is lower than HGB's (0.845 against 0.870 in CV). At threshold 0.5 it is more conservative on true arcs. This affects macro F1 but not Top-1 accuracy, which is the same for both models (0.880).

7.3. Reproduce with:
```bash
export PATHSNAP_DATA="/path/to/Student folder"
python models/prepare_data.py
python models/run.py mlp
```
The run takes about 40 minutes on a laptop CPU.

# Model Report: [Model Name]

| Item | Value |
|---|---|
| Model | [Model Name] |
| Algorithm family | [Family] |
| Owner | Team 6 |
| Code | `models/[name].py` |
| Results | `results/models/[name]/` |

## 1. Evaluation Protocol

All models follow the shared protocol in `models/harness.py`. The input is the 14-feature schema of contract version 1.0. Unreliable observations are excluded. Grouped five-fold cross-validation uses the fixed trip assignment in `results/models/folds.csv`. The primary metric is cross-validated macro F1 over candidate rows at threshold 0.5. Ties within one fold standard deviation are broken by lower log loss. Student-testing results are reported but never used for selection.

## 2. Model Understanding

2.1. State why the algorithm suits the PathSnap candidate-scoring problem.

2.2. State the main strengths and limitations of the algorithm for this task. Relate each point to an observed result.

## 3. Training and Configuration

3.1. List the important hyperparameters and their initial values. Copy them from `summary.json`.

3.2. State the loss function or optimisation objective.

3.3. State any training-row cap, scaling step or class-imbalance treatment.

## 4. Experimental Results

4.1. Report cross-validated macro F1, Top-1 accuracy, log loss, precision, recall and route continuity with fold standard deviations. Copy them from `summary.json`.

4.2. Report the same metrics on the student-testing partition.

4.3. State whether the model overfits or underfits. Support the statement with the fold results or the training history.

## 5. Visualisations

5.1. Include `confusion_student_test.png`.

5.2. Include `history.png` only if the model produces an iteration-wise history. Do not create curves for models without one.

## 6. Observations

Record behaviour specific to this model, such as failure cases on twin arcs, junctions or GPS gaps.

# PathSnap: Map-Matching and Route Reconstruction from Noisy GPS

Team 6

UE24CS352A Machine Learning.

## 1. Introduction

PathSnap reconstructs the road route of a vehicle from noisy GPS observations sampled at 1 Hz. The system assigns one road arc to every observation. The ordered arcs form a drivable route. The pipeline has four stages: candidate generation, candidate scoring by a binary classifier, sequence decoding by a hidden Markov model and deployment as a two-machine service.

Every metric, figure and map in this report is produced by code in the repository. Section 9 lists the commands.

## 2. Dataset

2.1. The training partition holds 80 trips. The student-testing partition holds 10 trips. The organisers hold 10 evaluator-testing trips. Each trip has a track file, a node file, an arc file and an ordered reference route.

2.2. Each trip carries its own road network. The 90 issued trips use 90 distinct networks whose centres lie up to about 10,000 km apart. No component can learn place-specific parameters. All learned components use place-independent features.

2.3. Sampling is 1 Hz for 96.5% of training steps. Eleven training trips contain a gap longer than 300 s. Nodes trace road geometry and are not limited to junctions. The median arc length is about 25 m. About 92% of arcs have a reverse twin with identical geometry. Direction therefore cannot be inferred from position alone.

2.4. The reference route lists arcs but does not assign observations to arcs. It is not verified ground truth. Section 8 states its limitations.

## 3. Methodology

### 3.1. Projection

All distances are computed in a local equirectangular projection centred on the mean track position of each trip. A point at longitude λ and latitude φ maps to x = R cos φ₀ (λ − λ₀) and y = R (φ − φ₀) in metres, with angles in radians and R = 6,371,008.8 m. The maximum relative error against the haversine distance is 0.18% across the issued trips. The projection is affine in longitude and latitude, so segment interpolation is exact in both systems.

### 3.2. Training Labels

An order-preserving dynamic-programming alignment assigns each training observation to a position on the reference route. The alignment minimises the sum of min(d, 100 m)² subject to non-decreasing route position. It agrees with a nearest-route-arc baseline on 93.5% of training observations. Observations farther than 50 m from their aligned arc are unreliable and are excluded from training and scoring. They form 11.1% of training observations and 0.1% of student-testing observations.

### 3.3. Candidate Generation

The candidate set of an observation contains every arc within 50 m, capped at the 16 nearest. When no arc lies within 50 m the 4 nearest arcs within 500 m are taken. Self-loops are dropped and duplicate arcs collapse to one canonical identifier. The rule misses the label arc for 0.48% of reliable training observations and 0.03% of student-testing observations. It yields 7.6 candidates per observation.

### 3.4. Features

Twenty place-independent features were engineered for each observation and candidate pair. They describe distance to the arc, position along the arc, heading agreement, motion, neighbouring observations and network shape. Heading is smoothed over three observations on each side, and windows never cross gaps longer than 30 s. Permutation importance and a grouped cross-validation ablation retained fourteen features. Principal component analysis shows an elbow at the third component for both sets. Ninety-five percent of variance requires 14 of 20 or 10 of 14 components, so the retained features are not redundant. The input contract `docs/INPUT_CONTRACT.md` fixes the schema.

### 3.5. Evaluation Protocol

All classifiers share one harness, `models/harness.py`. Training trips are assigned to five fixed folds, so no trip contributes rows to both the training and validation sides of a fold. The primary metric is cross-validated macro F1 over candidate rows at threshold 0.5. When two models differ by less than one fold standard deviation, the lower log loss decides. Student-testing results are reported but never used for selection or tuning.

### 3.6. Sequence Decoding

A hidden Markov model decodes each trip with the Viterbi algorithm. Hidden states are the candidate arcs of each observation. The emission score is α log p, where p is the classifier probability normalised over the candidates of the observation. The transition score is −|road distance − GPS distance| / β, where the road distance follows the directed road graph of the trip. A transition with no road path inside the search radius receives −(jump penalty + GPS distance / β). The decoder therefore continues through gaps instead of splitting the trip. Consecutive distinct arcs of the decoded sequence are joined by shortest paths to form the drivable route. α, β and the jump penalty are tuned on out-of-fold training predictions only.

## 4. Model Comparison

Five classifiers from five algorithm families were evaluated under the protocol of section 3.5.

| Model | Family | CV macro F1 | CV log loss | CV Top-1 | Student-test macro F1 | Student-test Top-1 | Refit time (s) |
|---|---|---|---|---|---|---|---|
| Random Forest | Bagging ensemble | 0.9335 ± 0.0304 | 0.0720 | 88.2% | 0.9315 | 88.2% | 316 |
| HistGradientBoosting | Boosting | 0.9316 ± 0.0305 | 0.0803 | 88.0% | 0.9329 | 88.5% | 9 |
| Multilayer Perceptron | Neural network | 0.9280 ± 0.0316 | 0.0792 | 88.0% | 0.9252 | 87.3% | 412 |
| Distance-weighted kNN | Instance-based | 0.9151 ± 0.0315 | 0.1480 | 86.2% | n/a | n/a | n/a |
| Logistic Regression | Linear | 0.9080 ± 0.0354 | 0.1160 | 86.9% | 0.9106 | 86.7% | 0.8 |

Random Forest has the highest mean macro F1. HistGradientBoosting and the multilayer perceptron lie within one fold standard deviation of it, so the tie-break applies. Random Forest has the lowest log loss and is selected. All five models exceed the nearest-arc baseline of 55.0% Top-1 by more than 30 points. Fold-to-fold variation of about 0.03 exceeds every difference between the three leading models. The variation arises from trip difficulty rather than model choice. The table is produced by `models/compare.py`. Individual model reports are in `docs/`.

## 5. Hyperparameter Optimisation

5.1. Bayesian optimisation used the Optuna tree-structured Parzen estimator. Each trial ran the five fixed folds and returned mean macro F1. A median pruner stopped weak trials after the second fold. The untuned configuration was evaluated as trial 0 of every study. Student-testing data was not read.

5.2. A full Random Forest fold takes several minutes, so the Random Forest search trained each fold on a fixed sample of 150,000 rows of whole observations and scored the full validation fold. The space covered the number of trees, the features per split, the minimum leaf size, the maximum depth and class weighting. The search stopped when ten consecutive completed trials gave no gain above 0.0005.

| Study | Trials completed | Untuned | Best | Folds improved |
|---|---|---|---|---|
| Random Forest, TPE | 20 | 0.9319 | 0.9327 | 4 of 5 |
| Random Forest, random search | 9 | 0.9319 | 0.9321 | 4 of 5 |

5.3. The search converged. The five leading trials share the features-per-split rule, the leaf size and class weighting and differ by less than 0.0001. The best configuration adds `class_weight="balanced_subsample"` to the `models/rf.py` configuration. Random search with the same budget reached a lower best value.

5.4. A full-size harness run of the best configuration reached CV macro F1 0.9337 against 0.9335, improved one fold of five and raised log loss from 0.0720 to 0.0732. The difference lies far inside one fold standard deviation and the lower log loss decides. The untuned Random Forest therefore remains the deployed model. The sample gain did not survive the full training set.

5.5. A study on HistGradientBoosting with all training rows ran 50 completed trials. It raised macro F1 from 0.9316 to 0.9341 with a gain on all five folds. HistGradientBoosting was not selected, so this configuration is not deployed.

## 6. Route Reconstruction

The decoder was tuned on out-of-fold predictions of the classifier. Top-1 accuracy is pooled over reliable observations. Route continuity is the share of consecutive distinct arcs that connect in the road graph before path filling.

| Configuration | Train OOF Top-1 | Train OOF continuity | Student-test Top-1 | Student-test continuity | Student-test route F1 |
|---|---|---|---|---|---|
| HistGradientBoosting only | 88.4% | 73.6% | 88.5% | 73.4% | 0.926 |
| Distance-only HMM | 89.0% | 89.7% | 89.7% | 89.4% | 0.968 |
| HistGradientBoosting with HMM | 92.4% | 90.0% | 92.8% | 90.3% | 0.965 |
| Random Forest only | 88.7% | 75.7% | 88.2% | 74.2% | 0.942 |
| Random Forest with HMM | 92.4% | 89.4% | 92.2% | 89.9% | 0.976 |

The decoder tuned on Random Forest predictions uses β = 2 and a jump penalty of 10. Random Forest with HMM matches HistGradientBoosting with HMM on training out-of-fold Top-1, at 92.37% against 92.41%. The classifier ranking of section 4 does not carry through the decoder. The decoder removes most of the per-observation difference between the two classifiers.

The HMM raises Top-1 accuracy by about four points and continuity by about sixteen points over the classifier alone. The distance-only HMM, which ignores the classifier, reaches 89.0% Top-1, so the classifier contributes beyond geometry. With HistGradientBoosting the decoder made one training trip worse than the classifier alone and no student-testing trip worse. The drivable route reaches 99.9% filled continuity. Figures for the selected model are in `results/hmm/`.

## 7. Deployment

7.1. Laptop 1 runs a FastAPI service. It exposes `GET /health`, `GET /trips`, `GET /trips/{id}/track` and `POST /predict`. A prediction runs candidate generation, feature construction, the classifier and the decoder on the track, node and arc files of the trip. The service accepts only trip identifiers present in the served partition. When a reference route is present on the server it is used only to compute the metrics. Reference arcs and labels are never returned.

7.2. Laptop 2 serves a static page that lists the trips, sends a trip identifier and draws the GPS observations, the matched points and the predicted route on a map together with the metrics.

7.3. The check `api/check_api.py` decodes every student-testing trip through the deployed path and reproduces the offline Top-1 accuracy and continuity exactly. The deployed model is Random Forest with its tuned decoder.

## 8. Limitations

8.1. The reference route is not verified ground truth. Three training trips account for most unreliable observations. Labels depend on the alignment rule of section 3.2.

8.2. Arcs are treated as straight segments between consecutive nodes. Curved roads are represented by their shape points, so the median arc of 25 m keeps the approximation error small.

8.3. Reverse-twin arcs share geometry. Their separation relies on heading, which is unreliable at low speed and after gaps.

8.4. Fold variation exceeds the differences between the leading classifiers. The ranking of Random Forest, HistGradientBoosting and the multilayer perceptron is not statistically separated.

8.5. Evaluator-testing trips played no part in any modelling decision. Their score is unknown until evaluation.

## 9. Reproducibility

| Step | Command |
|---|---|
| Analysis and model selection | `bash scripts/reproduce.sh` |
| One model through the harness | `python models/run.py <model>` |
| Comparison and selection | `python models/compare.py` |
| Bayesian optimisation | `python tuning/tune_rf.py --sampler tpe` and `python tuning/tune_hgb.py` |
| Decoder tuning and comparison | `python hmm/tune.py --model <model>` and `python hmm/final.py --model <model>` |
| HMM figures and map | `python hmm/report.py --model <model>` |
| Deployment model and check | `python api/train_final.py` and `python api/check_api.py` |
| Service | `uvicorn api.app:app --host 0.0.0.0 --port 8000` |
| Interface | `python3 -m http.server 8080 -d ui` |

The data root is read from the `PATHSNAP_DATA` environment variable. Raw data, derived data, trained models and maps are not committed.

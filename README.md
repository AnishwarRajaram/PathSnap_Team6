# PathSnap: Map-Matching and Route Reconstruction

PathSnap matches noisy 1 Hz GPS observations to a road network and reconstructs the driven route. A classifier scores the candidate road arcs of each observation. A hidden Markov model with Viterbi decoding then selects a connected route through the road graph. A FastAPI service and an evaluator interface run the pipeline on two laptops.

Team 6. Course: UE24CS352A Machine Learning.

## 1. Results

### 1.1. Classifier Comparison

Grouped five-fold cross-validation on the 80 training trips. Student-testing trips are scored once and play no part in selection.

| Model | Family | CV macro F1 | CV log loss | CV Top-1 | Student-test macro F1 | Student-test Top-1 |
|---|---|---|---|---|---|---|
| Random Forest | Bagging ensemble | 0.9335 ± 0.0304 | 0.0720 | 88.2% | 0.9315 | 88.2% |
| HistGradientBoosting | Boosting | 0.9316 ± 0.0305 | 0.0803 | 88.0% | 0.9329 | 88.5% |
| Multilayer Perceptron | Neural network | 0.9280 ± 0.0316 | 0.0792 | 88.0% | 0.9252 | 87.3% |
| Distance-weighted kNN | Instance-based | 0.9151 ± 0.0315 | 0.1480 | 86.2% | n/a | n/a |
| Logistic Regression | Linear | 0.9080 ± 0.0354 | 0.1160 | 86.9% | 0.9106 | 86.7% |
| Nearest arc only | Baseline | n/a | n/a | 55.0% | n/a | n/a |

Random Forest is selected. HistGradientBoosting and the multilayer perceptron lie within one fold standard deviation of it, and the tie goes to the lowest log loss.

### 1.2. Route Reconstruction

| Configuration | Train OOF Top-1 | Train OOF continuity | Student-test Top-1 | Student-test continuity | Student-test route F1 |
|---|---|---|---|---|---|
| Random Forest only, untuned | 88.7% | 75.7% | 88.2% | 74.2% | 0.942 |
| Random Forest with HMM, untuned | 92.4% | 89.4% | 92.2% | 89.9% | 0.976 |
| Tuned Random Forest only | 88.6% | 75.4% | 88.1% | 74.2% | 0.944 |
| Tuned Random Forest with HMM (deployed) | 92.5% | 89.9% | 92.2% | 90.3% | 0.980 |
| Distance-only HMM | 89.0% | 89.7% | 89.7% | 89.4% | 0.968 |

Continuity is measured before path filling. The drivable route returned by the decoder reaches 99.98% continuity on student testing. The deployed service runs the tuned Random Forest with its own decoder parameters (α = 1, β = 3, jump penalty 10) and reproduces the student-testing figures exactly.

### 1.3. Hyperparameter Optimisation

Bayesian optimisation used the Optuna tree-structured Parzen estimator on the five fixed trip folds with CV macro F1 as the objective. Each study evaluates the untuned configuration as its first trial.

| Study | Budget | Trials completed | Untuned | Best | Folds improved |
|---|---|---|---|---|---|
| Random Forest, TPE | 150,000 rows per fold | 20 | 0.9319 | 0.9327 | 4 of 5 |
| Random Forest, random search | 150,000 rows per fold | 9 | 0.9319 | 0.9321 | 4 of 5 |
| HistGradientBoosting, TPE | All rows | 50 | 0.9316 | 0.9341 | 5 of 5 |

The Random Forest search converged: ten consecutive completed trials gave no gain above 0.0005, and the leading trials share one region of the space. Its best configuration adds `class_weight="balanced_subsample"` to the `models/rf.py` configuration. The guidelines require the selected model to be refitted with the chosen configuration and its outputs to feed the decoder and the interface. The tuned configuration is therefore `models/rf_tuned.py`. It was refitted on all training rows, the decoder was retuned on its out-of-fold predictions and it is the deployed model. A full-size harness run reached CV macro F1 0.9337 against 0.9335 for the untuned configuration, with log loss 0.0732 against 0.0720 and inference time 33.7 against 60.1 ms per 1,000 observations. The full-size check is in `results/tuning/rf_fullsize_check.json` and the tuned harness results are in `results/tuning/rf_tuned/`.

## 2. Repository Layout

| Path | Contents |
|---|---|
| `pathsnap/` | Shared library: data loading and projection, labels, candidates, features, HMM decoder, end-to-end prediction |
| `eda/` | Exploratory analysis, steps 01 to 08 |
| `models/` | Evaluation harness, one module per model, data preparation and model comparison |
| `hmm/` | Decoder tuning, evaluation and figures |
| `tuning/` | Bayesian optimisation of the classifiers |
| `api/` | FastAPI service, deployment model training and service check |
| `ui/` | Evaluator interface, a static page for the second laptop |
| `scripts/` | `reproduce.sh` reruns the analysis and model comparison and checks the reference metrics |
| `results/` | Metrics, CSV summaries and figures |
| `docs/` | Report, input contract, model report template and model reports |

## 3. Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATHSNAP_DATA="/path/to/Student folder"
```

`PATHSNAP_DATA` must point to the issued student folder, which contains `train/` and `student_test/`. The data is never committed. Python 3.12 is used. All dependencies are pinned.

## 4. Running the Pipeline

### 4.1. Analysis and Model Comparison

```bash
bash scripts/reproduce.sh          # add --maps for the Folium maps
```

The script runs EDA steps 1 to 6, trains every model through the harness, applies the selection rule and checks the reference metrics. It prints `REPRODUCED` on success. Random Forest and the multilayer perceptron dominate the run time of about 90 minutes.

### 4.2. Individual Steps

```bash
.venv/bin/python models/prepare_data.py          # labels and features, about 10 s
.venv/bin/python models/run.py rf                # one model through the harness
.venv/bin/python models/compare.py               # comparison table and selection
.venv/bin/python hmm/tune.py --model rf          # decoder tuning on training out-of-fold predictions
.venv/bin/python hmm/final.py --model rf         # classifier against classifier with HMM
.venv/bin/python hmm/report.py --model rf        # HMM figures
```

### 4.3. Hyperparameter Optimisation

```bash
.venv/bin/python tuning/tune_rf.py --sampler tpe --study rf          # Random Forest, TPE
.venv/bin/python tuning/tune_rf.py --sampler random --study rf_random # Random Forest, random search
.venv/bin/python tuning/tune_hgb.py --trials 60                      # HistGradientBoosting, TPE
.venv/bin/python models/run.py rf_tuned                              # refit of the tuned configuration
.venv/bin/python hmm/tune.py --model rf_tuned && .venv/bin/python hmm/final.py --model rf_tuned
```

### 4.4. Deployment

Laptop 1 builds the deployment model and serves the evaluation trips. `train_final.py` uses the model selected in `results/models/comparison.csv` in its tuned form `models/<model>_tuned.py`, unless `--model` is given.

```bash
.venv/bin/python api/train_final.py              # deployment model and decoder parameters
.venv/bin/python api/check_api.py                # parity with hmm/final.py and endpoint checks
PATHSNAP_PARTITION=student_test .venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The service exposes `GET /health`, `GET /trips`, `GET /trips/{id}/track` and `POST /predict`. It also serves the interface at its root address, so a browser on laptop 2 can open `http://<laptop 1 address>:8000/` directly. The interface can instead be served from laptop 2 and pointed at laptop 1.

```bash
python3 -m http.server 8080 -d ui                # then open http://localhost:8080
```

The request and response formats are fixed in `docs/INPUT_CONTRACT.md`.

### 4.5. Model Weights

The fitted model files are not stored in the repository. The deployed bundle `artifacts/models/final.joblib` holds a 300-tree Random Forest fitted on 1.48 million training rows and is about 1.25 GB, above the 100 MB file limit of GitHub. Every input needed to rebuild it is committed: the model configuration (`models/rf_tuned.py`), the feature list (`pathsnap/features.py`), the decoder parameters (`results/hmm/decoder_rf_tuned.json`) and the training code (`api/train_final.py`). The fit is deterministic through a fixed random seed and takes about four minutes on a laptop CPU.

```bash
export PATHSNAP_DATA=/path/to/"Student folder"
.venv/bin/python api/train_final.py              # writes artifacts/models/final.joblib
.venv/bin/python api/check_api.py                # must print PASS
```

`check_api.py` confirms the rebuilt bundle against the committed student-testing results in `results/hmm/final_rf_tuned_student_test_summary.csv` to a tolerance of 1e-9.

## 5. Method Summary

| Step | Method |
|---|---|
| Projection | Local equirectangular projection to metres around each trip centre, within 0.18% of haversine |
| Labels | Order-preserving dynamic-programming alignment of observations to reference route arcs |
| Candidates | Arcs within 50 m, at most 16, with the 4 nearest within 500 m as fallback |
| Features | 14 place-independent features on distance, position on arc, heading, motion, neighbouring observations and network shape |
| Classifier | Binary: is this candidate the true arc. Selected by cross-validated macro F1 with trip-grouped folds, then tuned by Bayesian optimisation |
| Decoder | HMM over candidate arcs. Emission from classifier probabilities. Transition from road distance against GPS distance on the directed graph. Viterbi with soft breaks |

The full method and its evaluation are in `docs/REPORT.md`.

## 6. Data Rules

Raw data, derived data, trained models and Folium maps are not committed. Evaluator-testing data plays no part in any modelling decision. Reference routes stay on the server and never reach the model or the interface.

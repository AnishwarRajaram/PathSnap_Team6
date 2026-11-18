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
| Random Forest only | 88.7% | 75.7% | 88.2% | 74.2% | 0.942 |
| Random Forest with HMM | 92.4% | 89.4% | 92.2% | 89.9% | 0.976 |
| Distance-only HMM | 89.0% | 89.7% | 89.7% | 89.4% | 0.968 |

Continuity is measured before path filling. The drivable route returned by the decoder reaches 99.9% continuity. The deployed service reproduces the student-testing figures exactly.

## 2. Repository Layout

| Path | Contents |
|---|---|
| `pathsnap/` | Shared library: data loading and projection, labels, candidates, features, HMM decoder, end-to-end prediction |
| `eda/` | Exploratory analysis, steps 01 to 08 |
| `models/` | Evaluation harness, one module per model, data preparation and model comparison |
| `hmm/` | Decoder tuning, evaluation and figures |
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

### 4.3. Deployment

Laptop 1 builds the deployment model and serves the evaluation trips. `train_final.py` uses the model selected in `results/models/comparison.csv` unless `--model` is given.

```bash
.venv/bin/python api/train_final.py              # deployment model and decoder parameters
.venv/bin/python api/check_api.py                # parity with hmm/final.py and endpoint checks
PATHSNAP_PARTITION=student_test .venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The service exposes `GET /health`, `GET /trips`, `GET /trips/{id}/track` and `POST /predict`. Laptop 2 serves the interface and connects to the address of laptop 1.

```bash
python3 -m http.server 8080 -d ui                # then open http://localhost:8080
```

The request and response formats are fixed in `docs/INPUT_CONTRACT.md`.

## 5. Method Summary

| Step | Method |
|---|---|
| Projection | Local equirectangular projection to metres around each trip centre, within 0.18% of haversine |
| Labels | Order-preserving dynamic-programming alignment of observations to reference route arcs |
| Candidates | Arcs within 50 m, at most 16, with the 4 nearest within 500 m as fallback |
| Features | 14 place-independent features on distance, position on arc, heading, motion, neighbouring observations and network shape |
| Classifier | Binary: is this candidate the true arc. Selected by cross-validated macro F1 with trip-grouped folds |
| Decoder | HMM over candidate arcs. Emission from classifier probabilities. Transition from road distance against GPS distance on the directed graph. Viterbi with soft breaks |

The full method and its evaluation are in `docs/REPORT.md`.

## 6. Data Rules

Raw data, derived data, trained models and Folium maps are not committed. Evaluator-testing data plays no part in any modelling decision. Reference routes stay on the server and never reach the model or the interface.

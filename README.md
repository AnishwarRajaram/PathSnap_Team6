# PathSnap: Map-Matching and Route Reconstruction

PathSnap matches noisy 1 Hz GPS observations to a road network and reconstructs the driven route. A classifier scores candidate road segments for each observation. An HMM with Viterbi decoding then selects a connected route through the road graph. A FastAPI service and an evaluator interface run the pipeline on two laptops.

Team: [Team Name / Author Name]. Course: UE24CS352A Machine Learning.

## 1. Results

| Stage | Training, grouped 5-fold CV | Student testing |
|---|---|---|
| Nearest arc only, Top-1 | 55.0% | n/a |
| Logistic Regression, macro F1 / Top-1 | 0.908 / 86.9% | 0.911 / 86.7% |
| HistGradientBoosting, macro F1 / Top-1 | 0.932 / 88.0% | 0.933 / 88.5% |
| HistGradientBoosting with HMM, Top-1 | 92.4% | 92.8% |
| Route continuity before and after HMM | 73.6% to 90.0% | 73.4% to 90.3% |

The model comparison covers five model families. Three are in progress. Results come from `results/` and are reproducible with the commands in section 4.

## 2. Repository Layout

| Path | Contents |
|---|---|
| `pathsnap/` | Shared library: data loading and projection, label alignment, candidate generation, features, HMM decoder |
| `eda/` | Exploratory analysis, steps 01 to 08 |
| `models/` | Evaluation harness, one file per model, data preparation and model comparison |
| `hmm/` | Route reconstruction experiments, decoder tuning and figures |
| `scripts/` | `reproduce_viva1.sh` reruns everything up to model selection |
| `results/` | Metrics, CSV summaries and figures |
| `docs/` | Input contract and model report template |
| `AGENTS.md` | Team rules for people and AI agents |

The `main` branch is the integration branch. Each team member develops on a personal branch cut from `main`, and finished work is merged back after review.

## 3. Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATHSNAP_DATA="/path/to/Student folder"
```

`PATHSNAP_DATA` must point to the issued student folder, which contains `train/` and `student_test/`. The data is never committed. Python 3.12 is used. All dependencies are pinned.

## 4. Running the Pipeline

### 4.1. Reproduce Everything up to Model Selection

```bash
bash scripts/reproduce_viva1.sh          # add --maps for the Folium maps
```

The script runs EDA steps 1 to 6, trains every model through the harness, applies the selection rule and checks the reference metrics. It prints `REPRODUCED` on success. It takes about three minutes on a laptop.

### 4.2. Individual Steps

```bash
.venv/bin/python models/prepare_data.py          # labels and features, about 10 s
.venv/bin/python models/run.py hgb               # one model through the harness
.venv/bin/python models/compare.py               # comparison table and selection
.venv/bin/python hmm/tune.py --model hgb         # decoder tuning on training out-of-fold predictions
.venv/bin/python hmm/final.py --model hgb        # classifier vs HMM comparison
.venv/bin/python hmm/report.py                   # HMM figures
```

### 4.3. API and Interface

Pending implementation. Laptop 1 will run the FastAPI service with `GET /trips` and `POST /predict`. Laptop 2 will run the evaluator interface. The request and response formats are fixed in `docs/INPUT_CONTRACT.md`.

## 5. Method Summary

| Step | Method |
|---|---|
| Projection | Local equirectangular projection to metres around each trip centre, within 0.18% of haversine |
| Labels | Order-preserving dynamic-programming alignment of observations to reference route arcs |
| Candidates | Arcs within 50 m, at most 16, with the 4 nearest as fallback |
| Features | 14 place-independent features on distance, position on arc, heading, motion, neighbouring observations and network shape |
| Classifier | Binary: is this candidate the true arc. Selected by cross-validated macro F1 with trip-grouped folds |
| Decoder | HMM over candidate arcs. Emission from classifier probabilities. Transition from road distance against GPS distance on the directed graph. Viterbi with soft breaks |

## 6. Data Rules

Raw data, derived data, trained models and Folium maps are not committed. Evaluator-testing data plays no part in any modelling decision. Ground-truth routes stay on the server side and never reach the model or the interface. See `AGENTS.md` for the full rules.

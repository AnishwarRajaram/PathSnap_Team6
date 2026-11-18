# PathSnap Team 6: Agent Operating Context

This document is the shared operating context for every team member and every AI agent working in this repository. Each contributor reads it before starting work. The team updates it when a decision changes.

## 1. Branch Policy

1.1. The `main` branch holds three artefacts: this document, the exploratory data analysis under `eda/` and the agreed input contract. No other work is committed to `main`.

1.2. The `anishwar` branch is the working branch. All modelling code, experiments, results and report sources are committed there. The team does not merge `anishwar` into `main`.

1.3. The team may create a separate repository for the final presentation. Until that decision is made, the history on `anishwar` remains unaltered. Contributors do not rewrite history or force-push.

## 2. Problem Statement

PathSnap reconstructs vehicle routes by matching noisy 1 Hz GPS observations to a road network. The required pipeline has six stages.

| Stage | Description |
|---|---|
| 1. Candidate generation | Select candidate road segments (arcs) near each GPS observation. |
| 2. Model comparison | Train at least five classifiers from different algorithm families to score each candidate. |
| 3. Model selection and tuning | Select the best classifier on measured evidence. Tune it by Bayesian optimisation with trip-grouped cross-validation. |
| 4. Sequence decoding | Decode the route with an HMM and the Viterbi algorithm. Classifier scores define emissions. Road connectivity and distance define transitions. |
| 5. Deployment | Laptop 1 runs a FastAPI service. Laptop 2 runs a user interface that lists the ten evaluator trips, sends a trip identifier and displays the predicted route and its score. |
| 6. Reporting | The team submits one PDF report with Folium maps and verifiable results. |

## 3. Dataset Description

3.1. The dataset has three partitions. The training partition holds 80 trips. The student-testing partition holds 10 trips. The organisers hold the 10 evaluator-testing trips.

3.2. Each trip directory contains four headerless whitespace-separated files.

| File | Columns | Notes |
|---|---|---|
| `.track` | longitude, latitude, timestamp (s) | Nominal sampling rate is 1 Hz. |
| `.nodes` | longitude, latitude | The row index is the node identifier. |
| `.arcs` | from_node, to_node | The row index is the arc identifier. |
| `.route` | arc identifier | Ordered reference route. |

3.3. Coordinates are stored with longitude first. Distance calculations use a projected coordinate system or the haversine formula. Euclidean distance in degrees is not used.

3.4. The reference route is not verified ground truth. The report states its limitations.

## 4. Mandatory Rules

4.1. Raw data is never committed. Code reads the data root from the `PATHSNAP_DATA` environment variable. Code never contains a local path.

4.2. Derived data, credentials and API keys are never committed.

4.3. AI agents report aggregates only from the dataset. Row-level values do not appear in logs, chat transcripts or commits.

4.4. Evaluator-testing data does not influence any modelling decision. Preprocessing is fitted on training data only. Decoder parameters are tuned on training and student-testing data only.

4.5. Predictions are never hardcoded. Ground-truth labels remain on the server and never reach the model or the user interface.

4.6. Every metric, graph and map in the report is produced by code in this repository.

4.7. The repository carries no licence file by team decision.

4.8. Commit messages carry no AI attribution lines.

4.9. Each contributor must be able to explain any code they commit.

## 5. Repository Layout

| Directory | Contents |
|---|---|
| `eda/` | Dataset exploration scripts and Folium maps |
| `features/` | Candidate generation and feature engineering |
| `models/` | Training script for each model and the comparison harness |
| `tuning/` | Bayesian optimisation configuration and trial logs |
| `hmm/` | HMM and Viterbi decoder |
| `api/` | FastAPI backend |
| `ui/` | Evaluator interface |
| `results/` | Small CSV files, logs and plots |
| `report/` | Report sources |

## 6. Input Contract

The team defines the input contract after exploratory analysis, dimensionality analysis and feature selection are complete. The contract specifies the candidate generation rule, the feature schema with units and types, and the request and response formats of the API. The agreed contract is committed to `main` alongside this document.

## 7. Status Record

| Item | Status |
|---|---|
| Repository and branches | Created |
| Data received | Training 80 trips. Student-testing 10 trips. |
| Exploratory analysis | In progress |
| Input contract | Not started |

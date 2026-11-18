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

3.5. The upstream source is the student folder issued by the organisers. It contains the guideline PDFs and the raw trip data. It stays local and is excluded by `.gitignore`. It is never committed.

3.6. Each trip carries its own road network. The 90 issued trips use 90 distinct node files. Trip centres lie up to about 10,000 km apart. No shared global map exists. Within a trip the nodes and arcs describe the area that the track crosses.

3.7. Nodes trace road geometry and are not limited to junctions. The median arc length is about 25 m. About 93% of arcs have a reverse twin with identical geometry, so direction cannot be inferred from position alone.

## 4. Task Framing

4.1. The system input for one trip is the track, the nodes and the arcs. The system output is one arc identifier per GPS observation. The ordered arcs form the predicted route.

4.2. The reference route is used differently in each partition.

| Partition | Use of the reference route |
|---|---|
| Training | Source of supervision labels. The team aligns each observation to a route arc. |
| Student-testing | Hidden from the model. Used only to score and compare models. |
| Evaluator-testing | Held on the server only. Used only to score the final prediction. |

4.3. The reference route lists arcs but does not assign observations to arcs. The team builds the observation-to-arc alignment that produces training labels.

4.4. Learned components use place-independent features such as distance to arc, heading difference and speed. HMM transitions are computed from each trip's own road graph. No component learns arc-specific or place-specific parameters.

## 5. Mandatory Rules

5.1. Raw data is never committed. Code reads the data root from the `PATHSNAP_DATA` environment variable. Code never contains a local path.

5.2. Derived data, credentials and API keys are never committed.

5.3. AI agents report aggregates only from the dataset. Row-level values do not appear in logs, chat transcripts or commits.

5.4. Evaluator-testing data does not influence any modelling decision. Preprocessing is fitted on training data only. Decoder parameters are tuned on training and student-testing data only.

5.5. Predictions are never hardcoded. Ground-truth labels remain on the server and never reach the model or the user interface.

5.6. Every metric, graph and map in the report is produced by code in this repository.

5.7. The repository carries no licence file by team decision.

5.8. Commit messages carry no AI attribution lines.

5.9. Each contributor must be able to explain any code they commit.

## 6. Repository Layout

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

## 7. Input Contract

The team defines the input contract after exploratory analysis, dimensionality analysis and feature selection are complete. The contract specifies the candidate generation rule, the feature schema with units and types, and the request and response formats of the API. The agreed contract is committed to `main` alongside this document.

## 8. Status Record

| Item | Status |
|---|---|
| Repository and branches | Created |
| Data received | Training 80 trips. Student-testing 10 trips. |
| Exploratory analysis | In progress |
| Input contract | Not started |

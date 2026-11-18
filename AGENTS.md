# PathSnap Team 6: Agent Operating Context

This document is the shared operating context for every team member and every AI agent working in this repository. Each contributor reads it before starting work. The team updates it when a decision changes.

## 1. Branch Policy

1.1. The `main` branch is the integration branch. It holds this document, the input contract, the shared library `pathsnap/`, the model harness and templates under `models/`, the fixed folds file, the dependency list, the exploratory data analysis scripts under `eda/`, the reproduction script `scripts/reproduce_viva1.sh` and the merged results of every team member.

1.2. Each team member works on a personal branch cut from the latest `main`. The branches are `anishwar`, `anusha`, `bharath` and `bhuvi`. A member may open an additional branch for one model, such as `bhuvi-rf`, or commit a finished model to `anishwar` after review. Only the model file, its summary results and its report are carried into `main`. Exploratory work on personal branches stays there.

1.3. Finished work is merged back into `main` after review. A merge must not change shared files such as the harness, the folds file, the feature list or `requirements.txt` without team agreement. A merge must pass the checks in `AGENTS.md` section 5.

1.4. Team members update their branches from `main` before starting a new stage so that every branch uses the current contract and shared code.

1.5. Contributors do not rewrite history or force-push. The team may create a separate repository for the final presentation.

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

5.8. Commits made with AI assistance carry a `Co-Authored-By` line that names the AI assistant. This disclosure applies from the candidate-generation commit onward. Earlier commits are not rewritten. The README and the report carry no separate AI-use section.

5.9. Each contributor must be able to explain any code they commit.

## 6. Repository Layout

| Directory | Contents |
|---|---|
| `pathsnap/` | Shared library: data loading and projection, labels, candidates, features, HMM decoder, end-to-end prediction |
| `eda/` | Exploratory analysis scripts, steps 01 to 08 |
| `models/` | Evaluation harness, one module per model, data preparation and model comparison |
| `tuning/` | Bayesian optimisation scripts |
| `hmm/` | Decoder evaluation, tuning and figures |
| `api/` | FastAPI service, deployment model training and service check |
| `ui/` | Evaluator interface |
| `scripts/` | Reproduction scripts |
| `results/` | Small CSV files, JSON summaries and figures |
| `docs/` | Input contract, model report template and model reports |

## 7. Input Contract

7.1. The input contract is `docs/INPUT_CONTRACT.md`, version 1.1. It fixes the candidate generation rule, the 14-feature schema with units and types, the label rule, the classifier and decoder interfaces, the API request and response formats and the metric definitions.

7.2. Every stage conforms to the contract. A change to any field, unit or rule requires a new contract version committed to `main`.

7.3. The full exploratory analysis that supports the contract is merged into `main` under `eda/` and `results/eda/`.

## 8. Status Record

| Item | Status |
|---|---|
| Repository and branches | Created |
| Data received | Training 80 trips. Student-testing 10 trips. |
| Exploratory analysis | Complete |
| Input contract | Version 1.1 approved |
| Model comparison | Four of five models complete: Logistic Regression, HistGradientBoosting, Random Forest and kNN. The multilayer perceptron is outstanding. |
| Model selection | Random Forest leads under the selection rule. Final selection follows the fifth model. |
| Bayesian optimisation | Script and HistGradientBoosting study complete. The selected model is tuned after final selection. |
| HMM and Viterbi | Complete for HistGradientBoosting. Retuned on the selected model's out-of-fold predictions after final selection. |
| API and interface | Complete and verified on one machine. Two-laptop rehearsal outstanding. |
| Report | Outstanding |

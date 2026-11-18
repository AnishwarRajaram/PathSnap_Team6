# AGENTS.md — PathSnap Team 6

Running context for every human and AI agent working in this repo. Read it before touching anything. Update it when a decision changes.

## Branches

- `main` holds only this setup. Do not commit to it and do not merge into it.
- `anishwar` is the working branch. All work goes here.
- The team may cut a fresh, clean repo for the final presentation near the end. Until then, keep history honest. Do not rewrite or force-push.

## The task (UE24CS352A hackathon)

Map-match noisy 1 Hz GPS traces to a road network and rebuild the driven route.

1. Generate candidate road segments (arcs) near each GPS point.
2. Train at least 5 classifiers from different families to score "is this candidate the true arc?".
3. Pick the best one on evidence. Tune it with Bayesian optimisation (group-aware CV, grouped by trip).
4. Decode the sequence with an HMM and Viterbi. Classifier scores feed the emissions. Road connectivity and distance feed the transitions.
5. Serve it with FastAPI on laptop 1. A UI on laptop 2 lists the 10 evaluator trips, sends a trip ID, and shows the route and the score.
6. Write one PDF report. Include Folium maps.

## Data

- Partitions: `train` (80 trips), `student_test` (10 trips), evaluator test (10 trips, held by organisers).
- Each trip folder has 4 headerless, whitespace-separated files:
  - `.track`: lon, lat, t (seconds)
  - `.nodes`: lon, lat. The row index is the node ID.
  - `.arcs`: from_node, to_node. The row index is the arc ID.
  - `.route`: the arc IDs of the reference route, in order.
- Coordinates are **lon first**. Do not use Euclidean distance in degrees. Use a projected CRS or haversine.
- The reference route is not perfect ground truth. Note its limits in the report.

## Hard rules

- Raw data is never committed. Read the data root from the `PATHSNAP_DATA` env variable. Never hardcode a local path.
- Never commit derived data, secrets or API keys.
- AI agents may print **aggregates only** from the data (counts, stats, distributions). No row-level dumps in logs, chat or commits.
- No evaluator-test leakage. Fit preprocessing on train only. Tune decoder parameters on train and student_test only.
- No hardcoded predictions. Ground truth stays on the server and never reaches the model or the UI.
- Every metric, graph and map in the report must come from code in this repo.
- No licence file, by team decision.
- Commit messages carry no AI attribution lines.
- Every team member must be able to explain any code they commit.

## Required repo layout (target)

```
eda/          dataset exploration, Folium maps
features/     candidate generation and feature engineering
models/       training scripts for each model, plus comparison
tuning/       Bayesian optimisation configs and trial logs
hmm/          HMM and Viterbi decoder
api/          FastAPI backend
ui/           evaluator UI
results/      CSVs, logs, plots (small files only)
report/       report sources
```

## Status log

- Repo set up. Data received: train 80, student_test 10.

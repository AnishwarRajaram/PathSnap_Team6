# PathSnap Input Contract

| Item | Value |
|---|---|
| Project | PathSnap |
| Team | [Team Name / Author Name] |
| Version | 1.0 |
| Status | Approved |
| Location | `main` branch. The working copy on `anishwar` must match it. |

## 1. Scope

This contract fixes the interfaces between the pipeline stages: candidate generation, feature construction, the classifier, the HMM decoder and the API. Any change to a field, unit or rule in this document requires a new contract version.

## 2. Raw Inputs

2.1. One trip consists of four files: `.track`, `.nodes`, `.arcs` and `.route`. The formats are defined in `AGENTS.md` section 3.

2.2. The prediction pipeline reads `.track`, `.nodes` and `.arcs` only. The label builder reads the `.route` file during training. The server-side scorer reads it during evaluation. No other component reads it.

2.3. All distances are computed in a local equirectangular projection centred on the mean track position. Its maximum relative error against haversine is 0.18% across the issued trips.

## 3. Candidate Generation

| Rule | Value |
|---|---|
| Search radius | 50 m from each GPS observation |
| Cap | The 16 nearest arcs within the radius |
| Fallback | If no arc lies within 50 m, the 4 nearest arcs within 500 m |
| Excluded arcs | Self-loops, where from_node equals to_node |
| Duplicate arcs | Arcs with identical (from_node, to_node) collapse to the smallest arc_id |
| Ordering | Rows are sorted by observation index, then by distance |

The rule is implemented in `pathsnap/candidates.py` by the function `generate`.

## 4. Feature Schema

Each row describes one pair of a GPS observation and a candidate arc. The model input is a float32 matrix with the 14 columns below in this exact order.

| # | Name | Type | Unit | Definition |
|---|---|---|---|---|
| 1 | dist_m | float | m | Distance from the observation to the arc segment |
| 2 | rank | int | none | Rank of the arc by distance within the observation, 0 is nearest |
| 3 | n_cand | int | none | Number of candidates for the observation |
| 4 | dist_minus_best | float | m | dist_m minus the smallest dist_m of the observation |
| 5 | dist_ratio_best | float | none | (dist_m + 1) / (smallest dist_m + 1) |
| 6 | proj_frac | float | 0 to 1 | Position of the projected point along the arc from its start node |
| 7 | dist_to_end_m | float | m | Distance from the projected point to the nearer arc end |
| 8 | arc_len_m | float | m | Arc length |
| 9 | heading_diff_deg | float | deg, 0 to 180 | Difference between the smoothed GPS heading and the arc bearing from start to end node |
| 10 | speed_mps | float | m/s | Displacement over a window of 3 observations each side divided by elapsed time |
| 11 | turn_rate_dps | float | deg/s | Change of smoothed heading between neighbouring observations per second |
| 12 | nbr_mean_dist_m | float | m | Mean distance of the 2 preceding and 2 following observations to the same arc |
| 13 | arc_oneway | int | 0 or 1 | 1 if the reverse arc does not exist |
| 14 | end_degree_min | int | none | Smaller undirected node degree of the two arc ends |

4.1. Smoothing windows never cross a time gap longer than 30 s.

4.2. Row keys `trip_id`, `obs_idx` and `arc_id` accompany every row but are never model inputs.

4.3. The schema is implemented in `pathsnap/features.py` by the function `build` and the column list `SELECTED`.

## 5. Training Labels

5.1. Labels come from an order-preserving dynamic-programming alignment of observations to reference route positions. The function `align_ordered` in `pathsnap/labels.py` implements it.

5.2. A row has label 1 when its arc_id equals the canonical arc_id of the aligned route arc. All other rows have label 0.

5.3. An observation whose aligned arc lies more than 50 m away is unreliable. Unreliable observations are excluded from training and from scoring.

## 6. Classifier Interface

| Item | Specification |
|---|---|
| Input | float32 matrix (rows, 14) in the column order of section 4 |
| Output | One probability p in [0, 1] per row that the candidate is the true arc |
| Calibration | Output must be a usable probability. Log loss is reported. |
| Preprocessing | Any scaler is fitted on training rows only and is saved with the model |

## 7. Decoder Interface

7.1. Hidden states are the candidate arcs of each observation.

7.2. Emission score for candidate c of observation i is p(i, c) normalised over the candidates of observation i.

7.3. Transition scores use the road graph of the same trip. The exact form is fixed by the HMM stage and recorded in its configuration file.

7.4. Output is one arc_id per observation and the ordered list of distinct consecutive arcs as the predicted route.

## 8. API Interface

8.1. `GET /trips` returns the list of available evaluation trip identifiers.

8.2. `POST /predict` accepts `{"trip_id": "<id>"}` and returns:

| Field | Type | Content |
|---|---|---|
| trip_id | string | Echo of the request |
| matched | list | One entry per observation: obs_idx, arc_id, lon, lat of the projected point |
| route_arcs | list of int | Predicted route as ordered distinct arcs |
| route_geojson | object | LineString of the predicted route for display |
| metrics | object | top1_accuracy, macro_f1, candidate_miss_rate, route_continuity, n_obs_scored |
| timing_ms | object | candidates, features, classifier, decoder, total |

8.3. The response never contains reference-route arcs or labels.

## 9. Metric Definitions

| Metric | Definition |
|---|---|
| Top-1 accuracy | Share of reliable observations whose predicted canonical arc equals the label arc |
| Macro F1 | Macro-averaged F1 over the binary candidate rows at threshold 0.5 |
| Candidate miss rate | Share of reliable observations whose label arc is absent from the candidate set |
| Route continuity | Share of consecutive distinct predicted arcs where the end node of the first equals the start node of the second |
| Log loss | Binary cross-entropy over candidate rows |

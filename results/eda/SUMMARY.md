# PathSnap Exploratory Data Analysis Summary

Team 6

All figures are produced by the scripts in `eda/`. Values are computed on the training partition unless stated otherwise.

## 1. Dataset Inventory

The training partition holds 80 trips and the student-testing partition holds 10. Every trip has all four files. No missing values, invalid coordinates or out-of-range identifiers exist. Coordinates are stored longitude first. The arc files contain 1,903 duplicate arcs and 24 self-loops. Source: `eda/01_inventory.py`.

## 2. GPS Track Quality

Sampling is 1 Hz for 96.5% of training steps and 98.6% of student-testing steps. Eleven training trips contain a gap longer than 300 s. One training trip spans 43 hours. Raw point-to-point heading changes by a median of 41 degrees per second, so heading must be smoothed before use. Source: `eda/02_track_quality.py`.

## 3. Road Network

Each trip carries its own road network. About 73% of nodes are shape points with two neighbours. About 92% of arcs have a reverse twin, but only 60% of route arcs do. Routes therefore use one-way carriageways more often than the network average. The largest strongly connected component holds 95% of nodes. Source: `eda/03_road_network.py`.

## 4. Label Alignment

An order-preserving alignment assigns each observation to a route arc. It agrees with a nearest-route-arc baseline on 93.5% of training observations. The baseline selects a distant route position for 8.5% of student-testing observations, which confirms the need for order. Observations farther than 50 m from their label are unreliable. They form 11.1% of training observations and 0.1% of student-testing observations. Three training trips account for most of them. Source: `eda/04_label_alignment.py`.

## 5. Candidate Generation

The adopted rule takes arcs within 50 m, caps at 16 and falls back to the 4 nearest arcs. It misses the label arc for 0.48% of reliable training observations and 0.03% of student-testing observations. It yields 7.6 candidates per observation in training. Source: `eda/05_candidate_sweep.py`.

## 6. Features

Twenty features were engineered and analysed. Heading difference and distance relative to the best candidate dominate permutation importance. Fourteen features were retained. A gradient-boosting baseline under grouped five-fold cross-validation reaches 88.3% Top-1 accuracy against 55.0% for the nearest arc. Source: `eda/06_features.py`.

## 7. Geographical Visualisation

The script `eda/07_maps.py` writes Folium maps for a typical trip, an out-and-back trip and the trip with the lowest label reliability. Each map has layer control and a legend.

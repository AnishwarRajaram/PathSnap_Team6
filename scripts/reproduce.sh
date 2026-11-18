#!/usr/bin/env bash
# Reproduce every step up to model selection.
#
#   1. Environment     .venv from requirements.txt (skipped if PYTHON is set)
#   2. EDA             eda/01 to eda/06: inventory, track quality, road network,
#                      label alignment, candidate sweep, feature analysis
#   3. Maps            eda/07 Folium maps (optional, pass --maps)
#   4. Models          every model module in models/ through the shared harness
#   5. Selection       models/compare.py builds the comparison table and applies the rule
#   6. Check           reference metrics of all five models must match within 0.002
#
# Random Forest and the multilayer perceptron take most of the run time (about 90 minutes in all).
#
# Usage:
#   export PATHSNAP_DATA=/path/to/"Student folder"
#   bash scripts/reproduce.sh [--maps]
#
# Outputs go to results/ and artifacts/ (artifacts are gitignored).
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PATHSNAP_DATA:?Set PATHSNAP_DATA to the student folder that holds train/ and student_test/}"
[ -d "$PATHSNAP_DATA/train" ] || { echo "No train/ under PATHSNAP_DATA"; exit 1; }

MAPS=0
[ "${1:-}" = "--maps" ] && MAPS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export PYTHONWARNINGS=ignore

if [ -z "${PYTHON:-}" ]; then
  if [ ! -x .venv/bin/python ]; then
    echo "== creating .venv"
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r requirements.txt
  fi
  PYTHON=.venv/bin/python
fi

step() { echo; echo "== $1"; local t0=$SECONDS; shift; "$@"; echo "   done in $((SECONDS - t0)) s"; }

step "EDA 1: dataset inventory"        "$PYTHON" eda/01_inventory.py
step "EDA 2: GPS track quality"        "$PYTHON" eda/02_track_quality.py
step "EDA 3: road network"             "$PYTHON" eda/03_road_network.py
step "EDA 4: label alignment"          "$PYTHON" eda/04_label_alignment.py
step "EDA 5: candidate radius sweep"   "$PYTHON" eda/05_candidate_sweep.py
step "EDA 6: features and analysis"    "$PYTHON" eda/06_features.py
[ "$MAPS" = 1 ] && step "EDA 7: Folium maps" "$PYTHON" eda/07_maps.py

# Every model module except the shared infrastructure files and tuned refits.
MODELS=$(ls models/*.py | xargs -n1 basename | sed 's/\.py$//' \
  | grep -vE '^(__init__|_template|harness|run|compare|prepare_data|.*_tuned)$')
for m in $MODELS; do
  step "Model: $m" "$PYTHON" models/run.py "$m"
done

step "Model selection" "$PYTHON" models/compare.py

echo; echo "== Reproducibility check"
"$PYTHON" - <<'PY'
import json, sys
from pathlib import Path
# Reference values recorded from the harness runs (fixed folds, fixed seeds).
REF = {
    "logreg": {"cv_macro_f1": 0.9080, "cv_top1": 0.8693, "st_macro_f1": 0.9106, "st_top1": 0.8666},
    "hgb":    {"cv_macro_f1": 0.9316, "cv_top1": 0.8795, "st_macro_f1": 0.9329, "st_top1": 0.8850},
    "rf":     {"cv_macro_f1": 0.9335, "cv_top1": 0.8821, "st_macro_f1": 0.9315, "st_top1": 0.8816},
    "mlp":    {"cv_macro_f1": 0.9280, "cv_top1": 0.8798, "st_macro_f1": 0.9252, "st_top1": 0.8727},
    "knn":    {"cv_macro_f1": 0.9151, "cv_top1": 0.8620},
}
TOL = 0.002
bad = 0
for name, ref in REF.items():
    f = Path("results/models") / name / "summary.json"
    if not f.exists():
        print(f"  {name}: missing"); bad += 1; continue
    s = json.loads(f.read_text())
    st = s.get("student_test") or {}
    got = {"cv_macro_f1": s["cv_mean"]["macro_f1"], "cv_top1": s["cv_mean"]["top1"],
           "st_macro_f1": st.get("macro_f1"), "st_top1": st.get("top1")}
    for k, v in ref.items():
        ok = abs(got[k] - v) <= TOL
        bad += not ok
        print(f"  {name:7s} {k:12s} expected {v:.4f} got {got[k]:.4f} {'OK' if ok else 'MISMATCH'}")
print("REPRODUCED" if bad == 0 else f"{bad} MISMATCH(ES)")
sys.exit(1 if bad else 0)
PY

"""Run one model through the shared harness.

Usage: PATHSNAP_DATA=<data root> python models/run.py <model_module> [--skip-student-test]
Example: python models/run.py logreg
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.harness import evaluate  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    evaluate(importlib.import_module(f"models.{sys.argv[1]}"), skip_student_test="--skip-student-test" in sys.argv)

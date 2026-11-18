"""Collect every results/models/<model>/summary.json into one comparison table and plots.

Selection rule (fixed in advance): highest CV macro F1. If the top two means differ by
less than the larger fold standard deviation, the lower CV log loss wins.
Usage: python models/compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results" / "models"


def main():
    rows = []
    for f in sorted(RESULTS.glob("*/summary.json")):
        if f.parent.name.endswith("_tuned"):  # tuned refits are not candidates for selection
            continue
        s = json.loads(f.read_text())
        m, sd, st = s["cv_mean"], s["cv_std"], s.get("student_test", {})
        rows.append({
            "model": s["name"], "family": s["family"],
            "key_config": ", ".join(f"{k}={v}" for k, v in s["config"].items()),
            "train_rows_cap": s.get("max_train_rows") or "all",
            "cv_macro_f1": m["macro_f1"], "cv_macro_f1_std": sd["macro_f1"],
            "cv_top1": m["top1"], "cv_top1_std": sd["top1"], "cv_log_loss": m["log_loss"],
            "cv_precision": m["precision"], "cv_recall": m["recall"],
            "cv_route_continuity": m["route_continuity"],
            "st_macro_f1": st.get("macro_f1", np.nan), "st_top1": st.get("top1", np.nan),
            "st_log_loss": st.get("log_loss", np.nan), "st_route_continuity": st.get("route_continuity", np.nan),
            "fit_s": st.get("fit_s", np.nan), "infer_ms_per_1k_obs": st.get("infer_ms_per_1k_obs", np.nan),
        })
    if not rows:
        raise SystemExit("No summaries found. Run models/run.py first.")
    df = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)
    a = df.iloc[0]
    if len(df) > 1:
        b = df.iloc[1]
        if a.cv_macro_f1 - b.cv_macro_f1 < max(a.cv_macro_f1_std, b.cv_macro_f1_std) and b.cv_log_loss < a.cv_log_loss:
            a = b
    df["selected"] = df.model == a.model
    df.to_csv(RESULTS / "comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.drop(columns=["key_config"]).round(4).to_string(index=False))
    print(f"\nSelected by rule: {a.model}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(df))
    ax[0].bar(x, df.cv_macro_f1, yerr=df.cv_macro_f1_std, capsize=4)
    ax[0].set(title="CV macro F1 (mean ± fold std)", xticks=x, xticklabels=df.model, ylim=(df.cv_macro_f1.min() - .05, 1))
    w = .38
    ax[1].bar(x - w / 2, df.st_top1, w, label="Top-1 accuracy"); ax[1].bar(x + w / 2, df.st_macro_f1, w, label="macro F1")
    ax[1].set(title="Student-test", xticks=x, xticklabels=df.model, ylim=(min(df.st_top1.min(), df.st_macro_f1.min()) - .05, 1))
    ax[1].legend()
    ax[2].bar(x - w / 2, df.cv_log_loss, w, label="CV"); ax[2].bar(x + w / 2, df.st_log_loss, w, label="student-test")
    ax[2].set(title="Log loss (lower is better)", xticks=x, xticklabels=df.model); ax[2].legend()
    fig.tight_layout(); fig.savefig(RESULTS / "comparison.png", dpi=110)
    print(f"Wrote {RESULTS/'comparison.csv'} and comparison.png")


if __name__ == "__main__":
    main()

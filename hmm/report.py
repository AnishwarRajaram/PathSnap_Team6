"""Figures and Folium map for the HMM section of the report.

Reads results/hmm/final_hgb_<split>_{summary,per_trip}.csv written by hmm/run_hmm.py.
Writes results/hmm/hmm_comparison.png, results/hmm/hmm_per_trip.png and, for one
training trip chosen by a fixed rule, a Folium map in results/maps (gitignored).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from branca.element import Element

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.hmm import DecoderParams, decode_trip  # noqa: E402

RES = ROOT / "results" / "hmm"
VARIANTS = ["classifier", "distance_hmm", "hmm"]
LABELS = {"classifier": "Classifier only", "distance_hmm": "Distance-only HMM", "hmm": "Classifier + HMM"}


def figures():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, (metric, title) in enumerate([("top1", "Top-1 accuracy"), ("route_continuity", "Route continuity (per-point sequence)"),
                                         ("route_f1", "Route F1 (length-weighted)")]):
        for j, split in enumerate(["train_oof", "student_test"]):
            s = pd.read_csv(RES / f"final_hgb_{split}_summary.csv", index_col=0).loc[VARIANTS]
            ax[k].bar(np.arange(3) + (j - .5) * .38, s[metric], .38, label=split.replace("_", " "))
        ax[k].set(title=title, xticks=range(3), xticklabels=[LABELS[v] for v in VARIANTS], ylim=(.7, 1))
        ax[k].legend()
    fig.tight_layout(); fig.savefig(RES / "hmm_comparison.png", dpi=110); plt.close(fig)

    r = pd.read_csv(RES / "final_hgb_train_oof_per_trip.csv", dtype={"trip_id": str})
    acc = r.assign(acc=r.top1_hits / r.n_rel.clip(lower=1)).pivot(index="trip_id", columns="variant", values="acc")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(acc.classifier, acc.hmm, s=18)
    ax.plot([0, 1], [0, 1], c="grey", ls="--")
    ax.set(title="Per-trip Top-1, train out-of-fold", xlabel="classifier only", ylabel="classifier + HMM", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout(); fig.savefig(RES / "hmm_per_trip.png", dpi=110); plt.close(fig)
    return acc


def folium_map(acc: pd.DataFrame):
    la = pd.read_csv(ROOT / "results" / "eda" / "label_alignment.csv", dtype={"trip_id": str})
    tid = la[la.partition == "train"].sort_values("disagree_far_pos").trip_id.iloc[-1]   # out-and-back trip
    trip = load_trip("train", tid)
    oof = pd.read_pickle(ROOT / "artifacts" / "oof" / "hgb_train_oof.pkl")
    cand = oof[oof.trip_id == tid].drop(columns="trip_id")
    prm = json.loads((RES / "decoder_hgb.json").read_text())
    chosen, _, route = decode_trip(trip, cand, DecoderParams(alpha=prm["alpha"], beta=prm["beta"], jump_penalty=prm.get("jump_penalty")))
    best = cand.loc[cand.groupby("obs_idx").p.idxmax()]
    nodes, arcs, t = trip.nodes, trip.arcs, trip.track

    def arc_line(a):
        u, v = arcs.u.iat[a], arcs.v.iat[a]
        return [(nodes.lat.iat[u], nodes.lon.iat[u]), (nodes.lat.iat[v], nodes.lon.iat[v])]

    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    layers = [("Reference route", trip.route, "#1b7f3b", 7), ("Classifier only (top pick)", np.unique(best.arc_id), "#ff7f0e", 4),
              ("Classifier + HMM route", np.unique(route), "#6a3d9a", 3)]
    for name, arcs_, col, w in layers:
        g = folium.FeatureGroup(name)
        for a in arcs_:
            folium.PolyLine(arc_line(int(a)), color=col, weight=w, opacity=.85).add_to(g)
        g.add_to(m)
    gps = folium.FeatureGroup("GPS observations")
    for i in range(0, len(t), max(1, len(t) // 3000)):
        folium.CircleMarker((t.lat.iat[i], t.lon.iat[i]), radius=2, color="#1f4fd1", fill=True, fill_opacity=.8, weight=0).add_to(gps)
    gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[t.lat.min(), t.lon.min()], [t.lat.max(), t.lon.max()]])
    m.get_root().html.add_child(Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;border:1px solid #333;'
        'padding:8px 10px;font:12px sans-serif;line-height:1.6"><b>Train trip, out-and-back</b><br>'
        '<span style="color:#1b7f3b">&#9473;</span> reference route<br><span style="color:#ff7f0e">&#9473;</span> classifier only<br>'
        '<span style="color:#6a3d9a">&#9473;</span> classifier + HMM<br><span style="color:#1f4fd1">&#9679;</span> GPS observation</div>'))
    (ROOT / "results" / "maps").mkdir(parents=True, exist_ok=True)
    m.save(ROOT / "results" / "maps" / "hmm_train_out_and_back.html")
    print(f"map written; trip Top-1 classifier={acc.loc[tid,'classifier']:.3f} hmm={acc.loc[tid,'hmm']:.3f}")


if __name__ == "__main__":
    acc = figures()
    d = acc.hmm - acc.classifier
    print(f"per-trip Top-1 change (train OOF): improved={int((d>1e-9).sum())} worse={int((d<-1e-9).sum())} "
          f"same={int((d.abs()<=1e-9).sum())}  median change={d.median():+.4f}  worst={d.min():+.4f}  best={d.max():+.4f}")
    folium_map(acc)

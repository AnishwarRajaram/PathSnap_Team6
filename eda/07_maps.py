"""EDA step 7: Folium maps and trajectory plots for the report.

Trips are chosen from train by fixed rules using earlier step outputs:
  typical      unflagged trip with observation count closest to the train median
  out_and_back trip where nearest-arc and ordered labels disagree most often
  problem      trip with the lowest share of reliable labels
Maps show GPS observations, nearby road network, reference route, and points
where the nearest-arc baseline disagrees with the ordered labels.
HTML maps go to results/maps (gitignored). A static trajectory figure goes to results/eda/figs.
"""
from __future__ import annotations

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
from pathsnap.candidates import build_index, query  # noqa: E402
from pathsnap.data import load_trip  # noqa: E402

EDA = ROOT / "results" / "eda"; MAPS = ROOT / "results" / "maps"
NETWORK_M = 150.0
MAX_POINTS = 4000

LEGEND = """<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;
border:1px solid #333;padding:8px 10px;font:12px sans-serif;line-height:1.6">
<b>{title}</b><br>
<span style="color:#9e9e9e">&#9473;</span> road network (within 150 m)<br>
<span style="color:#1b7f3b">&#9473;</span> reference route<br>
<span style="color:#1f4fd1">&#9679;</span> GPS observation<br>
<span style="color:#d62728">&#9679;</span> nearest-arc label differs from ordered label</div>"""


def pick_trips() -> dict[str, str]:
    tq = pd.read_csv(EDA / "track_quality.csv", dtype={"trip_id": str})
    la = pd.read_csv(EDA / "label_alignment.csv", dtype={"trip_id": str})
    tq, la = tq[tq.partition == "train"], la[la.partition == "train"]
    ok = tq[tq.flag_any == 0]
    typical = ok.iloc[(ok.n - tq.n.median()).abs().argsort()].trip_id.iloc[0]
    return {"typical": typical,
            "out_and_back": la.sort_values("disagree_far_pos").trip_id.iloc[-1],
            "problem": la.sort_values("reliable_B").trip_id.iloc[0]}


def make_map(kind: str, tid: str):
    trip = load_trip("train", tid)
    lab = np.load(ROOT / "artifacts" / "labels" / "train" / f"{tid}.npz")
    t, nodes, arcs = trip.track, trip.nodes, trip.arcs
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    net = folium.FeatureGroup("Road network", show=True)
    idx = build_index(trip)
    _, near_arcs, _ = query(idx, trip.track_xy(), NETWORK_M)
    for a in np.unique(near_arcs):
        u, v = arcs.u.iat[a], arcs.v.iat[a]
        folium.PolyLine([(nodes.lat.iat[u], nodes.lon.iat[u]), (nodes.lat.iat[v], nodes.lon.iat[v])],
                        color="#9e9e9e", weight=2, opacity=.7).add_to(net)
    route = folium.FeatureGroup("Reference route", show=True)
    ru, rv = arcs.u.to_numpy()[trip.route], arcs.v.to_numpy()[trip.route]
    line = [(nodes.lat.iat[ru[0]], nodes.lon.iat[ru[0]])] + [(nodes.lat.iat[x], nodes.lon.iat[x]) for x in rv]
    folium.PolyLine(line, color="#1b7f3b", weight=5, opacity=.8).add_to(route)
    gps = folium.FeatureGroup("GPS observations", show=True)
    step = max(1, len(t) // MAX_POINTS)
    for i in range(0, len(t), step):
        folium.CircleMarker((t.lat.iat[i], t.lon.iat[i]), radius=2, color="#1f4fd1", fill=True,
                            fill_opacity=.8, weight=0).add_to(gps)
    diff = folium.FeatureGroup("Nearest vs ordered label mismatch", show=True)
    for i in np.flatnonzero(lab["arc_nearest"] != lab["arc"])[::step]:
        folium.CircleMarker((t.lat.iat[i], t.lon.iat[i]), radius=3, color="#d62728", fill=True,
                            fill_opacity=.9, weight=0).add_to(diff)
    for g in (net, route, gps, diff):
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[t.lat.min(), t.lon.min()], [t.lat.max(), t.lon.max()]])
    m.get_root().html.add_child(Element(LEGEND.format(title=f"Train trip ({kind})")))
    MAPS.mkdir(parents=True, exist_ok=True)
    m.save(MAPS / f"train_{kind}.html")
    return trip, lab


def main():
    picks = pick_trips()
    fig, ax = plt.subplots(1, len(picks), figsize=(6 * len(picks), 6))
    for a, (kind, tid) in zip(ax, picks.items()):
        trip, lab = make_map(kind, tid)
        xy = trip.track_xy(); nxy = trip.nodes_xy()
        ru, rv = trip.arcs.u.to_numpy()[trip.route], trip.arcs.v.to_numpy()[trip.route]
        for p, q in zip(ru, rv):
            a.plot(*np.c_[nxy[p], nxy[q]], c="#1b7f3b", lw=2.5, alpha=.7)
        a.scatter(*xy.T, s=1, c="#1f4fd1")
        bad = ~lab["reliable"]
        a.scatter(*xy[bad].T, s=3, c="#d62728")
        a.set(title=f"Train trip, {kind}", xlabel="east (m)", ylabel="north (m)", aspect="equal")
        print(f"{kind}: map written, points={len(xy)}, unreliable={bad.mean():.1%}")
    fig.legend(handles=[plt.Line2D([], [], c="#1b7f3b", lw=2.5, label="reference route"),
                        plt.Line2D([], [], c="#1f4fd1", marker=".", ls="", label="GPS observation"),
                        plt.Line2D([], [], c="#d62728", marker=".", ls="", label="unreliable label (>50 m)")],
               loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(EDA / "figs" / "07_trajectories.png", dpi=110)
    print(f"Maps in {MAPS} (gitignored); figure figs/07_trajectories.png")


if __name__ == "__main__":
    main()

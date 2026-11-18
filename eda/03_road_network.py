"""EDA step 3: road network structure.

Per trip: size, degree mix, junction share, connectivity, reverse twins,
duplicate and self-loop arcs, arc length, and how much of the map the trip uses.
Writes results/eda/road_network.csv and an aggregate figure. Prints aggregates only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import list_partitions, list_trips, load_trip  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "eda"
FIGS = OUT / "figs"


def analyse(trip):
    n_nodes, arcs = len(trip.nodes), trip.arcs
    u, v = arcs.u.to_numpy(), arcs.v.to_numpy()
    nxy = trip.nodes_xy()
    alen = np.hypot(*(nxy[v] - nxy[u]).T)

    # Undirected neighbour count = number of distinct adjacent nodes.
    und = pd.DataFrame({"a": np.minimum(u, v), "b": np.maximum(u, v)})
    und = und[und.a != und.b].drop_duplicates()
    nbrs = np.bincount(np.concatenate([und.a, und.b]), minlength=n_nodes)

    pairs = pd.MultiIndex.from_arrays([u, v])
    rev = pd.MultiIndex.from_arrays([v, u])
    has_twin = rev.isin(pairs) & (u != v)
    dup_mask = pd.DataFrame({"u": u, "v": v}).duplicated(keep=False).to_numpy()

    g = coo_matrix((np.ones(len(u)), (u, v)), shape=(n_nodes, n_nodes))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    giant_w = np.bincount(lab_w).max() / n_nodes
    giant_s = np.bincount(lab_s).max() / n_nodes

    r = trip.route
    route_nodes = np.unique(np.concatenate([u[r], v[r]]))
    giant_s_label = np.bincount(lab_s).argmax()
    return dict(
        partition=trip.partition, trip_id=trip.trip_id, n_nodes=n_nodes, n_arcs=len(arcs),
        arcs_per_node=len(arcs) / n_nodes,
        share_dead_end=float(np.mean(nbrs == 1)), share_shape=float(np.mean(nbrs == 2)),
        share_junction=float(np.mean(nbrs >= 3)), share_isolated=float(np.mean(nbrs == 0)),
        share_twin=float(np.mean(has_twin)), share_oneway=float(np.mean(~has_twin & (u != v))),
        share_duplicate=float(np.mean(dup_mask)), n_self_loop=int((u == v).sum()),
        arc_len_p50=float(np.median(alen)), arc_len_p90=float(np.percentile(alen, 90)),
        arc_len_max=float(alen.max()), share_arc_lt1m=float(np.mean(alen < 1)),
        weak_components=n_weak, giant_weak=giant_w, strong_components=n_strong, giant_strong=giant_s,
        route_arcs_used_share=float(len(np.unique(r)) / len(arcs)),
        route_in_giant_strong=float(np.mean(lab_s[route_nodes] == giant_s_label)),
        route_has_duplicate_arc=float(np.mean(dup_mask[r])),
        route_twin_share=float(np.mean(has_twin[r])),
        route_len_km=float(alen[r].sum() / 1000),
    ), alen, nbrs


def main():
    rows, lens, nb = [], {}, {}
    for p in list_partitions():
        lens[p], nb[p] = [], []
        for tid in list_trips(p):
            r, alen, nbrs = analyse(load_trip(p, tid))
            rows.append(r); lens[p].append(alen); nb[p].append(nbrs)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "road_network.csv", index=False)

    pd.set_option("display.width", 160)
    cols = [c for c in df.columns if c not in ("partition", "trip_id")]
    print(df.groupby("partition")[cols].describe(percentiles=[.5]).T
          .loc[(slice(None), ["mean", "min", "50%", "max"]), :].round(3).to_string())
    for p in nb:
        n = np.concatenate(nb[p]); L = np.concatenate(lens[p])
        print(f"\n[{p}] undirected neighbours share: " + ", ".join(f"{k}:{np.mean(n==k):.1%}" for k in range(5)) + f", 5+:{np.mean(n>=5):.1%}")
        print(f"[{p}] pooled arc length m: p10={np.percentile(L,10):.1f} p50={np.median(L):.1f} p90={np.percentile(L,90):.1f} p99={np.percentile(L,99):.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for p in lens:
        L = np.concatenate(lens[p]); ax[0].hist(np.log10(np.clip(L, .1, None)), bins=60, alpha=.6, density=True, label=p)
        n = np.concatenate(nb[p]); ax[1].bar(np.arange(7) + (0.2 if p == "train" else -0.2),
                                          [np.mean(np.minimum(n, 6) == k) for k in range(7)], width=.4, label=p)
    ax[0].set(title="Arc length", xlabel="log10 metres", ylabel="density")
    ax[1].set(title="Distinct neighbours per node (6 = 6+)", xlabel="neighbours", ylabel="share")
    for a in ax: a.legend()
    fig.tight_layout(); fig.savefig(FIGS / "03_road_network.png", dpi=120)
    print(f"\nWrote {OUT/'road_network.csv'} and {FIGS/'03_road_network.png'}")


if __name__ == "__main__":
    main()

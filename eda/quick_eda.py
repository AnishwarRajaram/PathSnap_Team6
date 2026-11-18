"""Throwaway EDA. Prints aggregates only, no row-level data.
Usage: PATHSNAP_DATA=<data root> python3 eda/quick_eda.py
"""
import sys, os, hashlib
import numpy as np

root = os.environ["PATHSNAP_DATA"]
R = 6371000.0

def load(p, dtype=float):
    a = np.loadtxt(p, dtype=dtype, ndmin=2)
    return a

def proj(lon, lat, lon0, lat0):
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y

def q(a, ps=(50, 90, 99)):
    a = np.asarray(a, float)
    return " ".join(f"p{p}={np.percentile(a, p):.1f}" for p in ps) + f" min={a.min():.1f} max={a.max():.1f} mean={a.mean():.1f}"

def components(n, e):
    par = np.arange(n)
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in e:
        ra, rb = f(a), f(b)
        if ra != rb: par[ra] = rb
    roots = np.array([f(i) for i in range(n)])
    _, c = np.unique(roots, return_counts=True)
    return len(c), c.max() / n

def seg_dist(px, py, ax, ay, bx, by):
    # px,py: (P,) ; segments (S,) -> (P,S)
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    L2 = np.where(L2 == 0, 1e-9, L2)
    t = ((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / L2
    t = np.clip(t, 0, 1)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px[:, None] - cx, py[:, None] - cy)

rng = np.random.default_rng(0)
for part in sorted(os.listdir(root)):
    pdir = os.path.join(root, part)
    if not os.path.isdir(pdir): continue
    trips = sorted(d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d)))
    print(f"\n===== partition={part} trips={len(trips)}")
    S = {k: [] for k in ["npts", "dur", "dt1", "maxgap", "gaps_gt5", "nonmono", "nnodes", "narcs",
                          "arclen", "recip", "deg", "ncomp", "giant", "nroute", "routecont", "routerep",
                          "d_node", "d_arc", "d_route", "nearest_in_route", "cand20", "cand50",
                          "miss20", "miss50", "bbox_km", "routelen_km", "tracklen_km", "isolated"]}
    hashes, centres, files_missing = [], [], 0
    for t in trips:
        td = os.path.join(pdir, t)
        fs = {ext: [f for f in os.listdir(td) if f.endswith("." + ext)] for ext in ["track", "nodes", "arcs", "route"]}
        if any(len(v) != 1 for v in fs.values()):
            files_missing += 1; continue
        tr = load(os.path.join(td, fs["track"][0]))
        nd = load(os.path.join(td, fs["nodes"][0]))
        ar = load(os.path.join(td, fs["arcs"][0]), int)
        rt = load(os.path.join(td, fs["route"][0]), int).ravel()
        hashes.append(hashlib.md5(open(os.path.join(td, fs["nodes"][0]), "rb").read()).hexdigest())
        lon0, lat0 = tr[:, 0].mean(), tr[:, 1].mean()
        centres.append((lon0, lat0))
        tx, ty = proj(tr[:, 0], tr[:, 1], lon0, lat0)
        nx, ny = proj(nd[:, 0], nd[:, 1], lon0, lat0)
        dt = np.diff(tr[:, 2])
        S["npts"].append(len(tr)); S["dur"].append(tr[-1, 2] - tr[0, 2])
        S["dt1"].append(np.mean(dt == 1) * 100); S["maxgap"].append(dt.max())
        S["gaps_gt5"].append(int((dt > 5).sum())); S["nonmono"].append(int((dt <= 0).sum()))
        S["tracklen_km"].append(np.hypot(np.diff(tx), np.diff(ty)).sum() / 1000)
        S["nnodes"].append(len(nd)); S["narcs"].append(len(ar))
        ax, ay, bx, by = nx[ar[:, 0]], ny[ar[:, 0]], nx[ar[:, 1]], ny[ar[:, 1]]
        alen = np.hypot(bx - ax, by - ay); S["arclen"].extend(alen.tolist())
        S["bbox_km"].append(max(nx.ptp(), ny.ptp()) / 1000)
        pairs = set(map(tuple, ar.tolist()))
        S["recip"].append(np.mean([(b, a) in pairs for a, b in pairs]) * 100)
        deg = np.bincount(ar.ravel(), minlength=len(nd)); S["deg"].extend(deg.tolist())
        S["isolated"].append(int((deg == 0).sum()))
        nc, g = components(len(nd), ar); S["ncomp"].append(nc); S["giant"].append(g * 100)
        S["nroute"].append(len(rt))
        S["routelen_km"].append(alen[rt].sum() / 1000)
        S["routecont"].append(np.mean(ar[rt[:-1], 1] == ar[rt[1:], 0]) * 100 if len(rt) > 1 else 100)
        S["routerep"].append((len(rt) - len(np.unique(rt))))
        # proximity on a sample of points
        idx = rng.choice(len(tr), size=min(400, len(tr)), replace=False)
        px, py = tx[idx], ty[idx]
        dn = np.hypot(px[:, None] - nx, py[:, None] - ny).min(1); S["d_node"].extend(dn.tolist())
        D = seg_dist(px, py, ax, ay, bx, by)
        S["d_arc"].extend(D.min(1).tolist())
        rset = np.unique(rt)
        S["d_route"].extend(D[:, rset].min(1).tolist())
        S["nearest_in_route"].append(np.mean(np.isin(D.argmin(1), rset)) * 100)
        for r in (20, 50):
            S[f"cand{r}"].extend((D <= r).sum(1).tolist())
            S[f"miss{r}"].append(np.mean(D[:, rset].min(1) > r) * 100)
    print(f"trips with missing/duplicate files: {files_missing}")
    print(f"distinct .nodes files: {len(set(hashes))} of {len(hashes)}  (1 => one shared map)")
    c = np.array(centres)
    print(f"region centre lon~{c[:,0].mean():.1f} lat~{c[:,1].mean():.1f}, spread of trip centres: "
          f"{np.hypot(*proj(c[:,0], c[:,1], c[:,0].mean(), c[:,1].mean())).max()/1000:.0f} km")
    print("--- GPS track")
    print("points/trip:", q(S["npts"]))
    print("duration s :", q(S["dur"]))
    print("track km   :", q(S["tracklen_km"]))
    print("% dt==1s   :", q(S["dt1"]))
    print("max gap s  :", q(S["maxgap"]))
    print(f"gaps >5s total: {sum(S['gaps_gt5'])}, non-increasing timestamps total: {sum(S['nonmono'])}")
    print("--- road network (per trip)")
    print("nodes      :", q(S["nnodes"]))
    print("arcs       :", q(S["narcs"]))
    print("bbox km    :", q(S["bbox_km"]))
    print("arc len m  :", q(S["arclen"]))
    print("% arcs with reverse arc:", q(S["recip"]))
    dg = np.array(S["deg"]); print("node degree (in+out) share: " + ", ".join(f"{k}:{np.mean(dg==k)*100:.0f}%" for k in range(0, 7)) + f", >6:{np.mean(dg>6)*100:.0f}%")
    print(f"isolated nodes total: {sum(S['isolated'])}")
    print("components :", q(S["ncomp"]), "| giant comp % nodes:", q(S["giant"]))
    print("--- reference route")
    print("arcs/route :", q(S["nroute"]))
    print("route km   :", q(S["routelen_km"]))
    print("% consecutive arcs head-to-tail connected:", q(S["routecont"]))
    print(f"repeated arcs in routes total: {sum(S['routerep'])}")
    print("--- proximity (400 sampled pts/trip, metres)")
    print("GPS->nearest node :", q(S["d_node"]))
    print("GPS->nearest arc  :", q(S["d_arc"]))
    print("GPS->nearest ROUTE arc:", q(S["d_route"]))
    print("% pts whose nearest arc is on route:", q(S["nearest_in_route"]))
    print("candidates within 20m:", q(S["cand20"]), "| 50m:", q(S["cand50"]))
    print("candidate miss % @20m:", q(S["miss20"]), "| @50m:", q(S["miss50"]))

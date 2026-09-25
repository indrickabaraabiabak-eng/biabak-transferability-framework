"""Structure tests restricted to boreholes inside or outside the dense cluster.
Permutations are performed within each subset, so the reference distribution
keeps the subset's own variance."""
import pickle, numpy as np, pandas as pd
from biabak import config as C
from biabak import variography as V
OUT = C.OUT_DIR / "step1"
S = pickle.load(open(OUT / "step1_store.pkl", "rb")); T = pickle.load(open(OUT / "step1_tables.pkl", "rb"))
rng = np.random.default_rng(C.SEED + 7)
edges = np.arange(0, C.HMAX_KM + C.LAG_KM, C.LAG_KM)
rows, bins = [], []
for key in ["log10_Sc", "log10_T", "status"]:
    s = S[key]
    for name, m in [("inside cluster", s["inside"]), ("outside cluster", ~s["inside"])]:
        sub = s["sub"][m]; z = s["z"][m]
        i, j, h, _ = V.pairs(sub.x_km.values, sub.y_km.values)
        for minp in [C.MIN_PAIRS]:
            hm, g, n, keep = V.experimental(z, i, j, h, edges, min_pairs=minp)
            sims = V.permutation_envelope(z, i, j, h, edges, C.N_PERMUTATIONS, rng)
            lo, hi = np.percentile(sims, [2.5, 97.5], axis=0)
            tg, pg = V.global_deviation_test(g, sims, keep)
            for cut in [10.0, 20.0, 30.0]:
                sh = keep & (edges[1:] <= cut)
                if sh.sum() == 0:
                    continue
                o, e, p = V.short_lag_test(g, n, sims, sh)
                rows.append(dict(target=key, subset=name, n=int(m.sum()), variance=float(np.var(z, ddof=1)),
                                 bins_retained=int(keep.sum()), global_test_p=pg, short_lag_cut_km=cut,
                                 short_lag_bins=int(sh.sum()), short_lag_pairs=int(n[sh].sum()),
                                 gamma_short=o, gamma_perm_mean=e, ratio=o / e, one_sided_p_continuity=p))
            for b in range(len(edges) - 1):
                bins.append(dict(target=key, subset=name, bin_from_km=edges[b], bin_to_km=edges[b + 1],
                                 mean_lag_km=hm[b], n_pairs=int(n[b]), retained=bool(keep[b]), gamma=g[b],
                                 perm_p2_5=lo[b], perm_p97_5=hi[b]))
            s.setdefault("subset", {})[name] = (hm, g, n, keep, lo, hi, float(np.var(z, ddof=1)))
r = pd.DataFrame(rows)
# Holm over the three cut-offs x three targets x two subsets (18 one-sided tests)
p = r.one_sided_p_continuity.values; o = np.argsort(p); adj = np.empty(len(p)); run = 0
for k, idx in enumerate(o):
    run = max(run, (len(p) - k) * p[idx]); adj[idx] = min(run, 1)
r["p_holm_18_tests"] = adj
T["S17b_structure_tests_by_subset"] = r
T["S17c_variograms_by_subset"] = pd.DataFrame(bins)
pickle.dump(T, open(OUT / "step1_tables.pkl", "wb"))
pickle.dump(S, open(OUT / "step1_store.pkl", "wb"))
print(r.round(3).to_string())

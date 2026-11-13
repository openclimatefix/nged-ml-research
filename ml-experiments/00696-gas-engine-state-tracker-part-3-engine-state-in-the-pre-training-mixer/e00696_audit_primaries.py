# ruff: noqa: E501, ANN001, ANN201, B023, D103, F405, FURB192, I001, PTH123, PTH207, SIM115, UP031
"""Audit of experiment 696, step 3: S1 minus A0 at the other Primaries, and probe M1."""

import glob
import json

import numpy as np
import polars as pl
from scipy import stats
from e00696_audit_common import *  # noqa: F403

A = pl.read_parquet(OUT / "losses_all_primaries.parquet")
g = A.group_by("time_series_id", "arm", "seed", "month").agg(
    pl.col("crps").sum().alias("s"), pl.len().alias("n")
)
print("== other Primaries: S1-A0, all seeds, month bootstrap 95%; and seed-group split ==")
rng = np.random.default_rng(3)
rows = []
for sid in sorted(g["time_series_id"].unique()):
    x = g.filter(pl.col("time_series_id") == sid)

    def tab(arm, ss):
        y = (
            x.filter(pl.col("arm") == arm, pl.col("seed").is_in(ss))
            .group_by("month")
            .agg((pl.col("s") / pl.col("n")).mean().alias("m"), pl.col("n").first())
            .sort("month")
        )
        return y["m"].to_numpy(), y["n"].to_numpy().astype(float)

    a, n = tab("A0", list(range(10)))
    s, _ = tab("S1", list(range(10)))
    d = s - a
    c = rng.multinomial(len(d), np.ones(len(d)) / len(d), size=20000)
    dr = (c @ (d * n)) / (c @ n)
    a1, _ = tab("A0", [0, 1, 2])
    s1, _ = tab("S1", [0, 1, 2])
    a2, _ = tab("A0", list(range(3, 10)))
    s2, _ = tab("S1", list(range(3, 10)))
    est = (d * n).sum() / n.sum()
    g1 = ((s1 - a1) * n).sum() / n.sum()
    g2 = ((s2 - a2) * n).sum() / n.sum()
    rows.append(
        (
            sid,
            est,
            *np.percentile(dr, [2.5, 97.5]),
            g1,
            g2,
            (a1 * n).sum() / n.sum() - (a2 * n).sum() / n.sum(),
            (s1 * n).sum() / n.sum() - (s2 * n).sum() / n.sum(),
        )
    )
    print(
        f"id {sid}: S1-A0 {est:+.3f} [{rows[-1][2]:+.3f},{rows[-1][3]:+.3f}]  s0-2 {g1:+.3f}  s3-9 {g2:+.3f} | A0 s0-2 minus s3-9 {rows[-1][6]:+.3f}, S1 s0-2 minus s3-9 {rows[-1][7]:+.3f}"
    )
r = np.array(rows)
print(
    "mean over 15 others (excl 26): group gap A0 %+.4f, S1 %+.4f; S1-A0 s0-2 %+.4f s3-9 %+.4f"
    % tuple(r[r[:, 0] != 26][:, [6, 7, 4, 5]].mean(0))
)
print("== probe M1 from jsons ==")


def pv(arm):
    out = {}
    for s in range(10):
        folder = "W0p" if arm == "W0" and s < 3 else arm
        for f in sorted(glob.glob(f"{E6}/runs/{folder}/probe/*_s{s}.json")):
            m = f.split("/")[-1][:7]
            out[(m, s)] = json.load(open(f))["mae_points"]
    return out


w, t = pv("W0"), pv("WST")
ms = sorted({k[0] for k in w})
print("n", len(w), len(t))
pm = np.array(
    [np.mean([t[(m, s)] for s in range(10)]) - np.mean([w[(m, s)] for s in range(10)]) for m in ms]
)
h = stats.t.ppf(0.975, 7) * pm.std(ddof=1) / np.sqrt(8)
print(
    "M1 months: est %+.4f [%+.4f,%+.4f]" % (pm.mean(), pm.mean() - h, pm.mean() + h),
    " ".join("%+.3f" % x for x in pm),
)
ps = np.array(
    [np.mean([t[(m, s)] for m in ms]) - np.mean([w[(m, s)] for m in ms]) for s in range(10)]
)
h = stats.t.ppf(0.975, 9) * ps.std(ddof=1) / np.sqrt(10)
print(
    "M1 seeds: est %+.4f [%+.4f,%+.4f]" % (ps.mean(), ps.mean() - h, ps.mean() + h),
    " ".join("%+.3f" % x for x in ps),
)
wv = np.array([np.mean([w[(m, s)] for m in ms]) for s in range(10)])
tv = np.array([np.mean([t[(m, s)] for m in ms]) for s in range(10)])
print(
    "probe W0 per seed",
    " ".join("%.3f" % x for x in wv),
    "| s0-2 (695 weights via W0p) %.4f s3-9 %.4f" % (wv[:3].mean(), wv[3:].mean()),
)
print(
    "probe WST per seed",
    " ".join("%.3f" % x for x in tv),
    "| s0-2 %.4f s3-9 %.4f" % (tv[:3].mean(), tv[3:].mean()),
)
print("M1 relative to W0 probe MAE: %.2f%%" % (100 * pm.mean() / wv.mean()))
j = json.load(open(sorted(glob.glob(f"{E6}/runs/WST/probe/*.json"))[0]))
print("probe json keys", list(j.keys()))

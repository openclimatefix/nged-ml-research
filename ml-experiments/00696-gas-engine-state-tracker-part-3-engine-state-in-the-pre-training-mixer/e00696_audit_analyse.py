# ruff: noqa: E501, ANN001, ANN201, ANN202, B007, B023, C408, D103, E731, F405, F541, I001, N802, PTH123, RET504, SIM115
"""Audit of experiment 696, step 2: the four planned contrasts, the seed-group check, power, and S1d."""

import json

import numpy as np
import polars as pl
from scipy import stats
from e00696_audit_common import *  # noqa: F403

L = pl.read_parquet(OUT / "losses.parquet")
MONTHS = sorted(L["month"].unique().to_list())


def mat(arm, metric="crps", seeds=range(10), filt=None):
    d = L.filter(pl.col("arm") == arm, pl.col("seed").is_in(list(seeds)))
    if filt is not None:
        d = d.filter(filt)
    w = d.pivot(on="seed", index=["origin", "valid_time", "month"], values=metric).sort(
        "origin", "valid_time"
    )
    return w


def contrast(t, r, metric="crps", seeds=range(10), level=98.75, filt=None, nb=40000, label=""):
    seeds = list(seeds)
    a = mat(t, metric, seeds, filt)
    b = mat(r, metric, seeds, filt)
    j = a.join(b, on=["origin", "valid_time", "month"], suffix="_r")
    assert len(j) == len(a) == len(b), (len(j), len(a), len(b))
    D = np.stack([j[str(s)].to_numpy() - j[f"{s}_r"].to_numpy() for s in seeds])  # seeds x rows
    mo = j["month"].to_numpy()
    sm = D.mean(0)
    ms = sorted(set(mo))
    sums = np.array([sm[mo == m].sum() for m in ms])
    n = np.array([(mo == m).sum() for m in ms], float)
    monthly = sums / n
    per_seed = D.mean(1)

    def tint(v):
        h = stats.t.ppf(0.5 + level / 200, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return (v.mean() - h, v.mean() + h)

    rng = np.random.default_rng(7)
    c = rng.multinomial(len(ms), np.ones(len(ms)) / len(ms), size=nb)
    dr = (c @ sums) / (c @ n)
    tail = (100 - level) / 2
    bt = tuple(np.percentile(dr, [tail, 100 - tail]))
    # seed x month table, to estimate variance components
    SM = np.array([[D[s][mo == m].mean() for m in ms] for s in range(len(seeds))])
    res = dict(
        label=label,
        est=float(D.mean()),
        t_month=tint(monthly),
        boot=bt,
        t_seed=tint(per_seed),
        monthly=monthly,
        per_seed=per_seed,
        SM=SM,
    )
    return res


def fmt(r):
    return f"{r['label']}: est {r['est']:+.4f}; t-month [{r['t_month'][0]:+.3f},{r['t_month'][1]:+.3f}]; boot [{r['boot'][0]:+.3f},{r['boot'][1]:+.3f}]; t-seed [{r['t_seed'][0]:+.3f},{r['t_seed'][1]:+.3f}]"


out = []


def P(s):
    print(s)
    out.append(s)


P("== Planned contrasts (98.75%), own code ==")
C = {}
for lab, t, r in (("C1", "S1", "A0"), ("C2", "S1", "SP"), ("C3", "WST", "W0"), ("C4", "WS1", "W0")):
    C[lab] = contrast(t, r, label=f"{lab} {t}-{r}")
    P(fmt(C[lab]))
    P(
        "   per-seed "
        + " ".join(f"{x:+.3f}" for x in C[lab]["per_seed"])
        + "   monthly "
        + " ".join(f"{x:+.3f}" for x in C[lab]["monthly"])
    )
for lab, t, r in (("C1mae", "S1", "A0"), ("C3mae", "WST", "W0")):
    P(fmt(contrast(t, r, metric="mae", label=lab, level=95)))
P("== Seed-group check (a) ==")
for arm in ("A0", "S1", "SP", "W0", "WST", "WS1"):
    g = L.filter(pl.col("arm") == arm).group_by("seed").agg(pl.col("crps").mean()).sort("seed")
    v = g["crps"].to_numpy()
    P(
        f"{arm}: per-seed CRPS "
        + " ".join(f"{x:.3f}" for x in v)
        + f" | mean s0-2 {v[:3].mean():.4f}  s3-9 {v[3:].mean():.4f}  diff {v[:3].mean() - v[3:].mean():+.4f}  sd(seed) {v.std(ddof=1):.4f}"
    )
for lab, t, r in (
    ("S1-A0", "S1", "A0"),
    ("SP-A0", "SP", "A0"),
    ("S1-SP", "S1", "SP"),
    ("WST-W0", "WST", "W0"),
    ("WS1-W0", "WS1", "W0"),
):
    a = contrast(t, r, seeds=range(3), level=95, label=lab + " s0-2")
    b = contrast(t, r, seeds=range(3, 10), level=95, label=lab + " s3-9")
    P(fmt(a))
    P(fmt(b))


# Welch test of A0 group difference minus S1 group difference (DiD) using per-seed means
def pseed(arm):
    return (
        L.filter(pl.col("arm") == arm)
        .group_by("seed")
        .agg(pl.col("crps").mean())
        .sort("seed")["crps"]
        .to_numpy()
    )


for t, r in (("S1", "A0"), ("SP", "A0"), ("WST", "W0")):
    d = pseed(t) - pseed(r)
    tt = stats.ttest_ind(d[:3], d[3:], equal_var=False)
    P(
        f"{t}-{r} per-seed diff: s0-2 mean {d[:3].mean():+.3f} vs s3-9 {d[3:].mean():+.3f}; Welch p={tt.pvalue:.3f}"
    )
# treatment arms average (S1,SP) minus A0 by group: both treatments share the same A0 seeds
d = (pseed("S1") + pseed("SP")) / 2 - pseed("A0")
P(
    f"mean(S1,SP)-A0 per seed: "
    + " ".join(f"{x:+.3f}" for x in d)
    + f"; s0-2 {d[:3].mean():+.3f}, s3-9 {d[3:].mean():+.3f}, Welch p={stats.ttest_ind(d[:3], d[3:], equal_var=False).pvalue:.3f}"
)
# per month seed-group A0 effect
P("A0 minus mean(S1,SP), by month, seeds 0-2 vs 3-9:")
for m in MONTHS:
    x = L.filter(pl.col("month") == m).group_by("arm", "seed").agg(pl.col("crps").mean())
    g = lambda arm, ss: float(
        x.filter(pl.col("arm") == arm, pl.col("seed").is_in(ss))["crps"].mean()  # ty: ignore[invalid-argument-type]
    )  # ty: ignore[invalid-argument-type]
    P(
        f"  {m}: s0-2 {g('A0', [0, 1, 2]) - (g('S1', [0, 1, 2]) + g('SP', [0, 1, 2])) / 2:+.3f}  s3-9 {g('A0', list(range(3, 10))) - (g('S1', list(range(3, 10))) + g('SP', list(range(3, 10)))) / 2:+.3f}"
    )
P("== Variance components and power (c) ==")
for lab in ("C1", "C3"):
    SM = C[lab]["SM"]
    ns, nm = SM.shape
    # two-way: seed x month table of mean differences
    gm = SM.mean()
    ms_month = ns * ((SM.mean(0) - gm) ** 2).sum() / (nm - 1)
    ms_seed = nm * ((SM.mean(1) - gm) ** 2).sum() / (ns - 1)
    resid = SM - SM.mean(0) - SM.mean(1)[:, None] + gm
    ms_res = (resid**2).sum() / ((ns - 1) * (nm - 1))
    P(
        f"{lab}: sd single seed-month diff {SM.std(ddof=1):.3f}; sd monthly seed-mean {C[lab]['monthly'].std(ddof=1):.3f}; sd per-seed mean {C[lab]['per_seed'].std(ddof=1):.3f}; MS month {ms_month:.4f} seed {ms_seed:.4f} resid {ms_res:.4f}"
    )
    hw = [(r[1] - r[0]) / 2 for r in (C[lab]["t_month"], C[lab]["boot"], C[lab]["t_seed"])]
    P(
        f"   half-widths t-month {hw[0]:.3f} boot {hw[1]:.3f} t-seed {hw[2]:.3f}; minimal detectable effect (80% power, two-sided at 98.75%) ~ {max(hw[0], hw[2]) * (1 + 0.84 / stats.t.ppf(0.99375, 7)):.3f}"
    )
# Relative effect at host: what would a "clean encoder" gain look like? oracle bound: compare WST with W0 on running rows
host = L.filter(pl.col("arm").is_in(["W0", "WST", "WS1"]))
P("== Slices ==")
# engine running proxy at host: truth below real reading -> need the real reading; approximate by negative engine via stored diff
gt = (
    pl.read_parquet(SH / "ground_truth_primaries.parquet")
    .filter(pl.col("time_series_id") == 11)
    .select("valid_time", real=pl.col("power") * 100 / HOST_CAPACITY_MVA)
)
for arm in ("W0", "WST", "WS1"):
    d = (
        L.filter(pl.col("arm") == arm)
        .join(gt, on="valid_time", how="left")
        .with_columns(run=(pl.col("real") - pl.col("act")) > 0.5)
    )
    P(
        f"{arm}: CRPS engine running {d.filter(pl.col('run'))['crps'].mean():.3f} (n {d['run'].sum()}) off {d.filter(~pl.col('run'))['crps'].mean():.3f}"
    )
for t, r in (("WST", "W0"), ("S1", "A0"), ("S1d", "A0")):
    pass
P("== S1d (d) ==")
for lab, t, r in (
    ("S1d-A0", "S1d", "A0"),
    ("S1d-S1", "S1d", "S1"),
    ("S1-A0", "S1", "A0"),
    ("SP-A0", "SP", "A0"),
    ("S1d-SP", "S1d", "SP"),
):
    rr = contrast(t, r, seeds=range(3), level=95, label=lab + " s0-2")
    P(fmt(rr) + "  monthly " + " ".join(f"{x:+.3f}" for x in rr["monthly"]))
# S1d by lead band
for lo, hi in ((24, 48), (48, 96), (96, 168), (168, 241)):
    f = (pl.col("lead") >= lo) & (pl.col("lead") < hi)
    rr = contrast(
        "S1d", "S1", seeds=range(3), level=95, filt=f, label=f"S1d-S1 lead {lo}-{hi}", nb=4000
    )
    P(fmt(rr))
# median bias
for arm in ("A0", "S1", "SP", "S1d"):
    d = L.filter(pl.col("arm") == arm, pl.col("seed") < 3)
    P(
        f"{arm} s0-2: mean(q50-actual) {(d['q50'] - d['act']).mean():+.3f}, CRPS {d['crps'].mean():.3f}, MAE {d['mae'].mean():.3f}, mean-median {d['mean_minus_med'].mean():+.3f}"
    )
json.dump(out, open(OUT / "analyse_out.json", "w"), indent=0)

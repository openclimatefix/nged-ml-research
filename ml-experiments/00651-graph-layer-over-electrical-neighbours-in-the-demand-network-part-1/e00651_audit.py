# ruff: noqa: E501, D103, ANN001, ANN201, PLR0915, PLR0912, ANN202,D205,F841,UP031
"""Experiment 651 audit: re-derive the scorer's event-free CRPS from the stored forecasts and check it.

Own code for the CRPS (twice the mean pinball loss over the 11 quantile levels, written out here),
the mean forecast of a quantile distribution, leave-one-Primary-out contrasts, the bank M against
bank X change in each arm's median, and the trigger checks. Reuses only the row loaders of
experiment 613 (`e00613_nbscore.load_rows` through `GraphArm`).

Usage: python e00651_audit.py   (writes CSVs under data/.../00651-.../score/audit/)
"""

import e00583_score as s583  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00651_score import ARMS, ROOT, ROOT612, GraphArm

OUT = ROOT / "score" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
MONTHS = ["2025-10", "2025-12"]
SEEDS = [0, 1]
TAUS = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
QCOLS = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
SIX = "signed-MW six"
NAMES = list(ARMS)  # C, G0, GCN-E, GAT-E, GAT-R, GAT-F, FLOW-E
RNG = np.random.default_rng(651)
N_BOOT = 4000


def hdr(x: str) -> None:
    print("\n" + "=" * 8, x, flush=True)


def crps(*, y, q):
    """Twice the mean pinball over the 11 levels (the scorer's approximation), in capacity share."""
    d = y[:, None] - q
    pin = np.maximum(TAUS[None, :] * d, (TAUS[None, :] - 1.0) * d)
    return 2.0 * pin.mean(axis=1)


def mean_weights():
    """Weights w with mean forecast = q @ w: the mean of a quantile function that is linear between
    the 11 knots and flat outside [0.01, 0.99] (trapezoid rule on u in [0, 1]).
    """
    w = np.zeros(11)
    w[0] += 0.01  # u in [0, 0.01] at q01
    w[-1] += 0.01  # u in [0.99, 1] at q99
    for j in range(10):
        du = TAUS[j + 1] - TAUS[j]
        w[j] += du / 2
        w[j + 1] += du / 2
    assert abs(w.sum() - 1.0) < 1e-12
    return w


W = mean_weights()


def scale_about_median(*, q, factor):
    med = q[:, 5:6]
    return med + factor * (q - med)


def boot(*, diff, blocks, alphas=(0.05, 0.0025)):
    """Block bootstrap (blocks = origin days) of the mean of diff; returns mean and intervals."""
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(ub))
    cnt = np.bincount(inv, minlength=len(ub)).astype(float)
    idx = RNG.integers(0, len(ub), size=(N_BOOT, len(ub)))
    m = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    out = {"mean": float(diff.mean())}
    for a in alphas:
        out[f"lo{a}"], out[f"hi{a}"] = (
            float(np.percentile(m, 100 * a / 2)),
            float(np.percentile(m, 100 * (1 - a / 2))),
        )
    return out


def main() -> None:
    inp = k597.Inputs(arm="a1")
    t = inp.t
    kk = s583.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk)
    sorted_keys = kk[order]
    arms = {}
    for n in NAMES:
        a = GraphArm(name=n, folder=ARMS[n][0], bank="M", months=MONTHS, seeds=SEEDS)
        assert a.available(), n
        a.load(t=t, order=order, sorted_keys=sorted_keys)
        arms[n] = a
    use = np.isin(t.month, MONTHS)
    fin = np.ones(t.n, bool)
    for a in arms.values():
        for sd in SEEDS:
            fin &= np.isfinite(a.q[sd]).all(axis=1)
    fin &= np.isfinite(t.arms["V"]).all(axis=1) & np.isfinite(t.arms["M"]).all(axis=1)
    common = use & fin
    print("rows in table", t.n, "common rows", int(common.sum()))

    factors = pl.read_csv(ROOT / "score" / "M" / "factors.csv")

    def fac(system, m, sd):
        f = factors.filter(
            (pl.col("system") == system) & (pl.col("month") == m) & (pl.col("seed") == sd)
        )
        return float(f["factor"][0])

    # systems: raw seed means and the stored-factor rescaled ("-rs") seed means
    Q: dict[str, np.ndarray] = {}
    QSEED: dict[str, dict[int, np.ndarray]] = {}
    for n, a in arms.items():
        QSEED[n] = {sd: a.q[sd] for sd in SEEDS}
        Q[n] = np.sort(np.mean([a.q[sd] for sd in SEEDS], axis=0), axis=1)
        rs_seeds = []
        for sd in SEEDS:
            rs = np.full_like(a.q[sd], np.nan)
            for m in MONTHS:
                sel = use & (t.month == m)
                rs[sel] = scale_about_median(q=a.q[sd][sel], factor=fac(f"{n}-rs", m, sd))
            rs_seeds.append(rs)
        Q[f"{n}-rs"] = np.sort(np.mean(rs_seeds, axis=0), axis=1)

    y = t.obs
    CR, AE50, AEM, SEM, MEAN = {}, {}, {}, {}, {}
    for n, q in Q.items():
        ok = common
        CR[n] = np.full(t.n, np.nan)
        CR[n][ok] = crps(y=y[ok], q=q[ok])
        mean_f = np.full(t.n, np.nan)
        mean_f[ok] = q[ok] @ W
        MEAN[n] = mean_f
        AE50[n] = np.abs(y - q[:, 5])
        AEM[n] = np.abs(y - mean_f)
        SEM[n] = (y - mean_f) ** 2

    # =================================================== Q1
    hdr("Q1: six-MW event-free CRPS 24-120 h (points of capacity = share x 100)")
    sel6 = t.free & (t.group_of_row == SIX) & (t.band == 1) & common
    print(
        "n rows", int(sel6.sum()), "months", {m: int((sel6 & (t.month == m)).sum()) for m in MONTHS}
    )
    ef = pl.read_csv(ROOT / "score" / "M" / "event_free.csv")
    ref = ef.filter((pl.col("group") == SIX) & (pl.col("band") == "24-120 h"))
    print("csv n_rows:", ref["n_rows"].unique().to_list())
    rows = []
    for n in Q:
        mine = float(CR[n][sel6].mean() * 100)
        csv = ref.filter(pl.col("system") == n)
        csv_v = float(csv["crps"][0]) if csv.height else np.nan
        rows.append({"system": n, "mine": mine, "csv": csv_v, "diff": mine - csv_v})
    q1 = pl.DataFrame(rows)
    print(q1)
    q1.write_csv(OUT / "q1_levels.csv")
    print("max |mine - csv| over systems:", float(np.abs(q1["diff"]).max()))

    def per_month_contrast(a, b, metric, sel):
        d = []
        for m in MONTHS:
            ss = sel & (t.month == m)
            d.append((metric[a][ss] - metric[b][ss]).mean() * 100)
        return float(np.mean(d)), d

    pairs = []
    for suffix in ("", "-rs"):
        for n in NAMES:
            if n != "C":
                pairs.append((n + suffix, "C" + suffix))
            if n not in ("G0", "C"):
                pairs.append((n + suffix, "G0" + suffix))
        pairs.append(("GAT-E" + suffix, "GAT-R" + suffix))
        pairs.append(("GAT-F" + suffix, "GAT-E" + suffix))
    pairs = list(dict.fromkeys(p for p in pairs if p is not None))
    rows = []
    blocks6 = t.origin_day[sel6]
    for a, b in pairs:
        pooled = float((CR[a][sel6] - CR[b][sel6]).mean() * 100)
        pm, per = per_month_contrast(a, b, CR, sel6)
        bt = boot(diff=(CR[a][sel6] - CR[b][sel6]) * 100, blocks=blocks6)
        csv_a = ref.filter(pl.col("system") == a)["crps"][0]
        csv_b = ref.filter(pl.col("system") == b)["crps"][0]
        mae50_pm, _ = per_month_contrast(a, b, AE50, sel6)
        maem_pm, _ = per_month_contrast(a, b, AEM, sel6)
        rmse_a = float(np.sqrt(SEM[a][sel6].mean()) * 100)
        rmse_b = float(np.sqrt(SEM[b][sel6].mean()) * 100)
        rows.append(
            {
                "a": a,
                "b": b,
                "crps_pooled": pooled,
                "crps_csv": float(csv_a - csv_b),
                "crps_month_avg": pm,
                "crps_month0": per[0],
                "crps_month1": per[1],
                "boot95_lo": bt["lo0.05"],
                "boot95_hi": bt["hi0.05"],
                "boot99.75_lo": bt["lo0.0025"],
                "boot99.75_hi": bt["hi0.0025"],
                "mae_q50_month_avg": mae50_pm,
                "mae_mean_month_avg": maem_pm,
                "rmse_mean_diff": rmse_a - rmse_b,
                "same_sign_crps_mae50": bool(np.sign(pm) == np.sign(mae50_pm)),
                "same_sign_crps_maemean": bool(np.sign(pm) == np.sign(maem_pm)),
            }
        )
    q12 = pl.DataFrame(rows)
    q12.write_csv(OUT / "q1_q2_contrasts.csv")
    with pl.Config(tbl_rows=100, tbl_cols=30, fmt_str_lengths=20, float_precision=4):
        print(
            q12.select(
                "a",
                "b",
                "crps_pooled",
                "crps_csv",
                "crps_month_avg",
                "crps_month0",
                "crps_month1",
                "boot95_lo",
                "boot95_hi",
                "boot99.75_lo",
                "boot99.75_hi",
            )
        )
        hdr(
            "Q2: same contrasts in MAE of q50 and of the mean forecast (points of capacity, month-averaged)"
        )
        print(
            q12.select(
                "a",
                "b",
                "crps_month_avg",
                "mae_q50_month_avg",
                "mae_mean_month_avg",
                "rmse_mean_diff",
                "same_sign_crps_mae50",
                "same_sign_crps_maemean",
            )
        )
    for n in ("C", "G0", "GCN-E", "GAT-E", "GAT-R", "GAT-F", "FLOW-E"):
        print(
            n,
            "six-MW event-free: MAE q50 %.4f  MAE mean %.4f  RMSE mean %.4f"
            % (
                AE50[n][sel6].mean() * 100,
                AEM[n][sel6].mean() * 100,
                np.sqrt(SEM[n][sel6].mean()) * 100,
            ),
        )

    # =================================================== Q3
    hdr("Q3: median minus mean forecast per Primary (event-free, 24-120 h; points of capacity)")
    labels = [str(x) for x in t.label[:16]]
    ef_all = t.free & (t.band == 1) & common
    gap = np.full((16, len(NAMES)), np.nan)
    for p in range(16):
        sp = ef_all & (t.prim == p)
        for k, n in enumerate(NAMES):
            gap[p, k] = float(((Q[n][sp, 5]) - MEAN[n][sp]).mean() * 100)
    gdf = pl.DataFrame(
        {
            "primary": range(16),
            "label": labels,
            "group": [str(g) for g in t.group[:16]],
            **{n: gap[:, k] for k, n in enumerate(NAMES)},
        }
    )
    for n in NAMES[1:]:
        gdf = gdf.with_columns((pl.col(n) - pl.col("C")).alias(f"{n}-C"))
    gdf.write_csv(OUT / "q3_median_minus_mean.csv")
    with pl.Config(tbl_rows=40, tbl_cols=30, float_precision=3):
        print(gdf)
    THR = 0.05
    for n in NAMES[1:]:
        big = [
            (labels[p], round(gap[p, NAMES.index(n)] - gap[p, 0], 3))
            for p in range(16)
            if abs(gap[p, NAMES.index(n)] - gap[p, 0]) > THR
        ]
        print(n, "Primaries with |gap - C gap| >", THR, ":", big)
    print(
        "max |arm - C| over arms and Primaries:", float(np.nanmax(np.abs(gap[:, 1:] - gap[:, [0]])))
    )

    # =================================================== Q4
    hdr("Q4: leave one six-MW Primary out (event-free CRPS 24-120 h, month-averaged, points)")
    six_p = [p for p in range(16) if str(t.group[p]) == SIX]
    print("six-MW Primaries:", [labels[p] for p in six_p])
    lopo_rows = []
    cmp_pairs = [
        (a, b)
        for a, b in pairs
        if (b in ("C", "C-rs") or (a.startswith("GAT-E") and b.startswith("GAT-R")))
    ]
    for a, b in cmp_pairs:
        full, _ = per_month_contrast(a, b, CR, sel6)
        for p in six_p:
            ss = sel6 & (t.prim != p)
            v, _ = per_month_contrast(a, b, CR, ss)
            lopo_rows.append(
                {
                    "a": a,
                    "b": b,
                    "left_out": labels[p],
                    "value": v,
                    "full": full,
                    "flip": bool(np.sign(v) != np.sign(full)),
                }
            )
    lopo = pl.DataFrame(lopo_rows)
    lopo.write_csv(OUT / "q4_lopo_crps.csv")
    flips = lopo.filter(pl.col("flip"))
    print("sign flips (CRPS):", flips.height, "of", lopo.height)
    with pl.Config(tbl_rows=100, float_precision=4):
        print(flips)
        summ = lopo.group_by("a", "b", maintain_order=True).agg(
            pl.col("full").first(),
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
            pl.col("flip").sum().alias("n_flips"),
        )
        print(summ)

    # follow on bank M, arm minus C, leave one Primary out (all scored primaries; days4-end, 24-120 h, >=15 d)
    bep = pl.read_csv(ROOT / "score" / "M" / "bank_event_phase.csv", infer_schema_length=None)
    f = bep.filter(
        (pl.col("phase") == "days4-end")
        & (pl.col("band") == "24-120 h")
        & (pl.col("length_days") >= 15)
    )
    cl = f.group_by("system", "month", "primary").agg(pl.col("follow_q50").mean())
    fl_rows = []

    def follow_mean(system, excl=None):
        c_ = cl.filter(pl.col("system") == system)
        if excl is not None:
            c_ = c_.filter(pl.col("primary") != excl)
        return float(c_["follow_q50"].mean())  # ty: ignore[invalid-argument-type]

    for suffix in ("", "-rs"):
        for n in NAMES:
            if n == "C":
                continue
            full = follow_mean(n + suffix) - follow_mean("C" + suffix)
            for p in sorted(cl["primary"].unique().to_list()):
                v = follow_mean(n + suffix, p) - follow_mean("C" + suffix, p)
                fl_rows.append(
                    {
                        "a": n + suffix,
                        "b": "C" + suffix,
                        "left_out": labels[p],
                        "value": v,
                        "full": full,
                        "flip": bool(np.sign(v) != np.sign(full)),
                    }
                )
        full = follow_mean("GAT-E" + suffix) - follow_mean("GAT-R" + suffix)
        for p in sorted(cl["primary"].unique().to_list()):
            v = follow_mean("GAT-E" + suffix, p) - follow_mean("GAT-R" + suffix, p)
            fl_rows.append(
                {
                    "a": "GAT-E" + suffix,
                    "b": "GAT-R" + suffix,
                    "left_out": labels[p],
                    "value": v,
                    "full": full,
                    "flip": bool(np.sign(v) != np.sign(full)),
                }
            )
    fl = pl.DataFrame(fl_rows)
    fl.write_csv(OUT / "q4_lopo_follow_M.csv")
    with pl.Config(tbl_rows=100, float_precision=4):
        print(
            fl.group_by("a", "b", maintain_order=True).agg(
                pl.col("full").first(),
                pl.col("value").min().alias("min"),
                pl.col("value").max().alias("max"),
                pl.col("flip").sum().alias("n_flips"),
                pl.col("left_out").filter(pl.col("flip")).alias("flip_at"),
            )
        )

    # =================================================== Q5 / Q7: bank X vs bank M
    hdr(
        "Q5: bank M vs bank X change in q50 (points of capacity), event Primaries not mirrored / mirrored / others"
    )
    ev_all = pl.read_parquet(s583.BANK_PARQUET)
    nbm = inp.nb
    cat = {}  # (month, r) -> (set event primaries, set mirror primaries)
    depths = []
    for (m, r), g in ev_all.group_by("month", "pass"):
        E = set(g["primary"].to_list())
        Mi = set()
        for e in g.iter_rows(named=True):
            nb = nbm.get(e["primary"], [])
            if nb:
                Mi.add(int(nb[(r + e["primary"]) % len(nb)]))
        cat[(m, r)] = (E, Mi)
        depths.extend(g["depth"].to_list())
    print("mean event depth (points of capacity):", float(np.mean(np.abs(depths)) * 100))
    sm = s583.c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))

    def folders(n, bank):
        if n == "C":
            return (
                (ROOT612 / "P7_D50_e3" / "work" / "inject")
                if bank == "M"
                else (ROOT / "P7_D50" / "work" / "inject_X")
            )
        return ARMS[n][0] / "work" / ("inject" if bank == "M" else "inject_X")

    acc = {}  # (arm, category) -> [sum abs, count]
    g0c = {
        "q50_abs": [0.0, 0],
        "all_abs": [0.0, 0],
        "sxx": 0.0,
        "syy": 0.0,
        "sxy": 0.0,
        "sx": 0.0,
        "sy": 0.0,
        "n": 0,
    }
    g0c_err = {"sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0, "n": 0}
    xbank_names = ["C", *[n for n in NAMES if n != "C"]]
    n_missing = 0
    for m in MONTHS:
        for r in range(20):
            if (m, r) not in cat:
                continue
            E, Mi = cat[(m, r)]
            for d in (0, 14):
                for sd in SEEDS:
                    fn = f"{m}_r{r:02d}_d{d}_s{sd}.parquet"
                    dfm, dfx = {}, {}
                    ok = True
                    for n in xbank_names:
                        fm, fx = folders(n, "M") / fn, folders(n, "X") / fn
                        if not (fm.exists() and fx.exists()):
                            ok = False
                            n_missing += 1
                            break
                        dfm[n] = pl.read_parquet(fm, columns=[*KEYS, *QCOLS]).sort(KEYS)
                        dfx[n] = pl.read_parquet(fx, columns=[*KEYS, "q50"]).sort(KEYS)
                    if not ok:
                        continue
                    base = dfm["G0"].select(KEYS)
                    for n in xbank_names:
                        j = dfm[n].join(dfx[n].rename({"q50": "q50x"}), on=KEYS, how="inner")
                        pr = np.array([prim_of[k] for k in j["time_series_id"].to_list()])
                        dq = np.abs(j["q50"].to_numpy() - j["q50x"].to_numpy())
                        cats = np.where(
                            np.isin(pr, list(Mi)),
                            "mirrored",
                            np.where(np.isin(pr, list(E)), "event_not_mirrored", "other"),
                        )
                        for c_ in ("event_not_mirrored", "mirrored", "other"):
                            a_ = acc.setdefault((n, c_), [0.0, 0])
                            mk = cats == c_
                            a_[0] += float(dq[mk].sum())
                            a_[1] += int(mk.sum())
                    # Q7 on the injected rows: G0 vs C bank M
                    j = dfm["G0"].join(dfm["C"], on=KEYS, how="inner", suffix="_c")
                    gq = j.select(QCOLS).to_numpy()
                    cq = j.select([c_ + "_c" for c_ in QCOLS]).to_numpy()
                    g0c["q50_abs"][0] += float(np.abs(gq[:, 5] - cq[:, 5]).sum())
                    g0c["q50_abs"][1] += len(gq)
                    g0c["all_abs"][0] += float(np.abs(gq - cq).sum())
                    g0c["all_abs"][1] += gq.size
                    x_, y_ = gq[:, 5].astype(float), cq[:, 5].astype(float)
                    for k_, v_ in (
                        ("sx", x_.sum()),
                        ("sy", y_.sum()),
                        ("sxx", (x_ * x_).sum()),
                        ("syy", (y_ * y_).sum()),
                        ("sxy", (x_ * y_).sum()),
                    ):
                        g0c[k_] += float(v_)  # ty: ignore[unsupported-operator]
                    g0c["n"] += len(x_)
        print("done month", m, flush=True)
    print("missing file pairs:", n_missing)
    rows = []
    for n in xbank_names:
        r_ = {"arm": n}
        for c_ in ("event_not_mirrored", "mirrored", "other"):
            sm_, cn = acc[(n, c_)]
            r_[c_ + "_mean_abs_dq50_pts"] = sm_ / max(cn, 1) * 100
            r_[c_ + "_rows"] = cn
        rows.append(r_)
    q5 = pl.DataFrame(rows)
    q5.write_csv(OUT / "q5_graph_use_M_vs_X.csv")
    with pl.Config(tbl_rows=20, tbl_cols=12, float_precision=5):
        print(q5)

    # =================================================== Q6 / Q7 on real rows
    hdr("Q7: G0 vs C on the real event-free rows")
    for label_, sel in (
        ("six-MW event-free 24-120h", sel6),
        ("all common rows", common),
        ("all Primaries event-free 24-120h", ef_all),
    ):
        g, c_ = Q["G0"][sel], Q["C"][sel]
        r_all = np.corrcoef(g[:, 5], c_[:, 5])[0, 1]
        err_g, err_c = g[:, 5] - y[sel], c_[:, 5] - y[sel]
        print(
            f"{label_}: n={int(sel.sum())} corr(q50)={r_all:.5f} MAD q50={np.abs(g[:, 5] - c_[:, 5]).mean() * 100:.4f} pts "
            f"MAD all quantiles={np.abs(g - c_).mean() * 100:.4f} pts corr(errors)={np.corrcoef(err_g, err_c)[0, 1]:.5f} "
            f"MAD width90={np.abs((g[:, 9] - g[:, 1]) - (c_[:, 9] - c_[:, 1])).mean() * 100:.4f} pts; mean q50 diff (G0-C)={(g[:, 5] - c_[:, 5]).mean() * 100:.4f}"
        )
    # per seed
    for sd in SEEDS:
        g, c_ = QSEED["G0"][sd][sel6], QSEED["C"][sd][sel6]
        print(
            f"seed {sd}: G0 vs C same seed, six-MW: corr={np.corrcoef(g[:, 5], c_[:, 5])[0, 1]:.5f}, MAD q50={np.abs(g[:, 5] - c_[:, 5]).mean() * 100:.4f}"
        )
    g, c_ = QSEED["C"][0][sel6], QSEED["C"][1][sel6]
    print(
        f"C seed0 vs C seed1: corr={np.corrcoef(g[:, 5], c_[:, 5])[0, 1]:.5f}, MAD q50={np.abs(g[:, 5] - c_[:, 5]).mean() * 100:.4f}"
    )
    g, c_ = QSEED["G0"][0][sel6], QSEED["G0"][1][sel6]
    print(
        f"G0 seed0 vs G0 seed1: corr={np.corrcoef(g[:, 5], c_[:, 5])[0, 1]:.5f}, MAD q50={np.abs(g[:, 5] - c_[:, 5]).mean() * 100:.4f}"
    )
    n_ = g0c["n"]
    cov_ = g0c["sxy"] / n_ - g0c["sx"] / n_ * g0c["sy"] / n_
    corr_ = cov_ / np.sqrt(
        (g0c["sxx"] / n_ - (g0c["sx"] / n_) ** 2) * (g0c["syy"] / n_ - (g0c["sy"] / n_) ** 2)
    )
    print(
        f"injected rows bank M: n={n_} corr(q50)={corr_:.5f} MAD q50={g0c['q50_abs'][0] / g0c['q50_abs'][1] * 100:.4f} pts MAD all quantiles={g0c['all_abs'][0] / g0c['all_abs'][1] * 100:.4f} pts"
    )

    hdr("Q6e: seed noise, six-MW event-free CRPS 24-120 h by single seed (points)")
    rows = []
    cr_seed = {}
    for n in NAMES:
        for sd in SEEDS:
            cr_seed[(n, sd)] = np.full(t.n, np.nan)
            cr_seed[(n, sd)][common] = crps(y=y[common], q=np.sort(QSEED[n][sd][common], axis=1))
    for n in NAMES:
        v0 = cr_seed[(n, 0)][sel6].mean() * 100
        v1 = cr_seed[(n, 1)][sel6].mean() * 100
        rows.append(
            {
                "arm": n,
                "seed0": float(v0),
                "seed1": float(v1),
                "seed1_minus_seed0": float(v1 - v0),
                "seedmean": float(CR[n][sel6].mean() * 100),
                "minus_C_seed0": float(v0 - cr_seed[("C", 0)][sel6].mean() * 100),
                "minus_C_seed1": float(v1 - cr_seed[("C", 1)][sel6].mean() * 100),
            }
        )
    q6e = pl.DataFrame(rows)
    q6e.write_csv(OUT / "q6e_seed_noise.csv")
    with pl.Config(tbl_rows=20, tbl_cols=12, float_precision=4):
        print(q6e)
    # follow by seed is not stored per seed in the scorer's tables; say so
    # seed-pair gap on the GAT-E - GAT-R contrast
    for sd in SEEDS:
        print(
            "GAT-E minus GAT-R seed",
            sd,
            float((cr_seed[("GAT-E", sd)][sel6] - cr_seed[("GAT-R", sd)][sel6]).mean() * 100),
        )

    hdr("Q6a: controls A1/R/W/G4T in 651 vs experiment 613 (six-MW and eight, 24-120 h event-free)")
    c613 = pl.read_csv(next((ROOT.parent).glob("00613-*/score/c2_e3/event_free_cells.csv")))
    for grp in ("signed-MW six", "MVA-only eight"):
        g613 = c613.filter(
            (pl.col("group") == grp)
            & (pl.col("band") == "24-120 h")
            & pl.col("month").is_in(MONTHS)
        )
        for sysn in g613["system"].unique().sort().to_list():
            h = g613.filter(pl.col("system") == sysn)
            print(
                f"613 {grp} {sysn}: n={h['n_rows'].sum()} crps={float((h['crps'] * h['n_rows']).sum() / h['n_rows'].sum()) * 100:.4f} cover90={float((h['cover90'] * h['n_rows']).sum() / h['n_rows'].sum()):.4f} width90={float((h['width90'] * h['n_rows']).sum() / h['n_rows'].sum()) * 100:.3f}"  # ty: ignore[unsupported-operator]
            )
        h651 = ef.filter(
            (pl.col("group") == grp)
            & (pl.col("band") == "24-120 h")
            & pl.col("system").is_in(["A1", "R", "W", "G4T", "C", "G0"])
        )
        for r_ in h651.iter_rows(named=True):
            print(
                f"651 {grp} {r_['system']}: n={r_['n_rows']} crps={r_['crps']:.4f} cover90={r_['cover90']:.4f} width90={r_['width90']:.3f}"
            )

    hdr("Q6a-bis: own re-derivation of the controls A1, R, W from the stored arrays")
    v, mm = t.arms["V"], t.arms["M"]
    wide_r = inp.mode[t.prim, t.origin_day]
    q_r = k597.assemble(v=v, m=mm, wide=wide_r)
    w_ = np.full_like(v, np.nan)
    for mo in MONTHS:
        sl = use & (t.month == mo)
        w_[sl] = scale_about_median(q=v[sl], factor=fac("W", mo, -1))
    for grp in (SIX, "MVA-only eight"):
        sg = t.free & (t.group_of_row == grp) & (t.band == 1) & common
        for nm, qq in (("A1", v), ("R", q_r), ("W", w_)):
            cr_ = crps(y=y[sg], q=qq[sg])
            cov = ((y[sg] >= qq[sg][:, 1]) & (y[sg] <= qq[sg][:, 9])).mean()
            csvv = ef.filter(
                (pl.col("group") == grp) & (pl.col("band") == "24-120 h") & (pl.col("system") == nm)
            )
            print(
                f"{grp} {nm}: mine crps {cr_.mean() * 100:.4f} csv {csvv['crps'][0]:.4f}; mine cover90 {cov:.4f} csv {csvv['cover90'][0]:.4f}; share of rows R wide {wide_r[sg].mean():.4f}"
            )

    print("\nAUDIT SCRIPT DONE")


if __name__ == "__main__":
    main()

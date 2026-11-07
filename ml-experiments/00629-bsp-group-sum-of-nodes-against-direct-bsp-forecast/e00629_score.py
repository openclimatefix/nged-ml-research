# ruff: noqa: E501, PLR0912, PLR0915, B007, F841
"""Score the BSP sums of experiment 629 against the BSP meters.

Reads `work/runs/<network>/*.npz` (written by `e00629_build.py`) and writes
`work/scores_<network>.parquet` (one row per network, version, group, source, lead band, side, and
statistic) and `work/rows_<network>.npz` (the per-row quantities the figures use), and prints a report.

Sources: the four scenario sums J0, J2, J3, J5, the direct forecasts of the group meter
(`d622_lagged` or `d622_lagfree` is the headline one for the network; the others are reported
beside it), and the same four sums plus a gap correction (`+g`, post hoc: the mean of meter minus the
Primaries' observed sum by half-hour of the day, fitted on the other four origin months).

Conventions. Flow is positive when power flows into the group (import) and negative on export. The
quantile of a set of 51 scenario sums is numpy's `weibull` quantile (position k/52). The export side
scores the negated flow. A direct forecast's quantiles are the stored ones. The independent unit of
an exceedance rate is the valid calendar day, so its binomial error uses the number of valid days.
twCRPS uses the chaining function max(x, L) with L = 0.9 times the largest export (import) of the
meter before 2025-10-01, and 51 values (for a direct forecast, the 11 quantiles interpolated at the
levels k/52 as in experiment 434). Rows with a valid time on or after 2026-07-01 are dropped.
"""

import argparse
import os
from datetime import UTC, datetime

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00434_score as s434  # noqa: E402  # ty: ignore[unresolved-import]
import e00629_common as c  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from studies.power import scan_power  # noqa: E402

BANDS = {
    "0-12h": (0, 12),
    "12-36h": (12, 36),
    "36-120h": (36, 120),
    "120-240h": (120, 240.01),
    "0-240h": (0, 240.01),
}
TAUS = (0.5, 0.9, 0.95)
BOOT = 20000
BSP_GROUPS = (0, 1)  # positions in GROUP_IDS


def limits() -> dict[int, tuple[float, float]]:
    """Return (export limit, import limit) in MW per group id: 0.9 times the largest before 2025-10-01."""
    df = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(list(c.GROUP_IDS)) & (pl.col("time") < c.LIMIT_BEFORE)
        )
        .collect()
    )
    out = {}
    for g in c.GROUP_IDS:
        p = df.filter(pl.col("time_series_id") == g)["power"].to_numpy().astype(float)
        out[g] = (0.9 * float(-p.min()), 0.9 * float(p.max()))
    return out


def twcrps(*, x: np.ndarray, y: np.ndarray, limit: float) -> np.ndarray:
    """Threshold-weighted CRPS, chaining function max(., limit); x (..., n) values, y (...)."""
    v = np.maximum(x, limit)
    vy = np.maximum(y, limit)
    first = np.abs(v - vy[..., None]).mean(-1)
    vs = np.sort(v, axis=-1)
    n = vs.shape[-1]
    w = 2 * np.arange(1, n + 1) - n - 1
    second = (2 * (vs * w).sum(-1)) / n**2
    return first - 0.5 * second


def load_rows(*, network: str) -> dict:
    """Return all runs of a network stacked: arrays with a leading row axis, and the row labels."""
    files = sorted((c.WORK / "runs" / network).glob("*.npz"))
    cols: dict[str, list] = {
        k: []
        for k in (
            "sums",
            "median_sum",
            "meter",
            "prim_obs_sum",
            "direct",
            "lead",
            "valid",
            "origin_month",
            "init",
        )
    }
    for f in files:
        z = np.load(f)
        init = datetime.strptime(f.stem, "%Y-%m-%d").replace(hour=9, tzinfo=UTC)
        ok = c.valid_mask(init=init)
        cols["sums"].append(z["sums"][..., ok].transpose(3, 0, 1, 2))  # (rows, arm, group, 51)
        cols["median_sum"].append(z["median_sum"][:, ok].T)
        cols["meter"].append(z["meter"][:, ok].T)
        cols["prim_obs_sum"].append(z["prim_obs_sum"][:, ok].T)
        cols["direct"].append(z["direct"][:, :, ok].transpose(2, 0, 1, 3))  # (rows, src, group, 11)
        steps = np.flatnonzero(ok)
        cols["lead"].append((steps + 1) / 2.0)
        cols["valid"].append(
            np.array(
                [np.datetime64((init + (s + 1) * c.STEP).replace(tzinfo=None), "m") for s in steps]  # ty: ignore[unsupported-operator]
            )
        )
        cols["origin_month"].append(np.full(steps.size, str(z["month"])))
        cols["init"].append(np.full(steps.size, np.datetime64(init.replace(tzinfo=None), "m")))
    return {k: np.concatenate(v) for k, v in cols.items()}


def gap_shift(*, rows: dict) -> np.ndarray:
    """Return the gap correction per row and group (rows, 4): mean(meter - Primaries' sum) by half-hour of day, other months."""
    valid = rows["valid"].astype("datetime64[m]")
    hod = ((valid - valid.astype("datetime64[D]")) / np.timedelta64(30, "m")).astype(int)
    gap = rows["meter"] - rows["prim_obs_sum"]
    shift = np.zeros_like(gap)
    vmon = np.array([str(v)[:7] for v in valid.astype("datetime64[M]")])
    months = rows["origin_month"]
    for m in c.MONTHS:
        test = months == m
        for g in range(4):
            for h in range(48):
                tr = (~test) & (hod == h) & np.isfinite(gap[:, g]) & (vmon != m)
                te = test & (hod == h)
                shift[te, g] = gap[tr, g].mean() if tr.any() else 0.0
    return shift


def boot_ratio(
    *, a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the relative difference mean(a)/mean(b) - 1 (percent) of monthly means with a month-block interval."""
    idx = rng.integers(0, len(a), size=(BOOT, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1) - 1
    return 100 * (a.mean() / b.mean() - 1), 100 * np.quantile(r, 0.025), 100 * np.quantile(r, 0.975)


def main() -> None:
    """Score the stored sums of one network."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="K1")
    args = parser.parse_args()
    lim = limits()
    print(
        "scaled limits (MW) export, import:",
        {c.GROUP_NAMES[g]: tuple(round(v, 1) for v in lim[g]) for g in c.GROUP_IDS},
    )
    rows = load_rows(network=args.network)
    n = rows["lead"].size
    direct_key = c.HEADLINE_DIRECT[args.network]
    dkeys = list(c.DIRECT)
    shift = gap_shift(rows=rows)
    valid = rows["valid"]
    vhour = ((valid - valid.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(float)
    vmonth = valid.astype("datetime64[M]").astype(int) % 12 + 1
    vday = valid.astype("datetime64[D]")
    slices = {
        "all": np.ones(n, bool),
        "summer midday": (vhour >= 10) & (vhour < 14) & np.isin(vmonth, [4, 5, 6, 7, 8, 9]),
        "winter evening": (vhour >= 16) & (vhour < 20) & np.isin(vmonth, [11, 12, 1, 2]),
    }
    out: list[dict] = []
    ph = {}
    for gi, g in enumerate(c.GROUP_IDS):
        yy = rows["meter"][:, gi].astype(float)
        yy = yy[np.isfinite(yy)]
        # post hoc limits: the 2nd percentile of the meter (export side) and the 98th (import side)
        ph[g] = (float(-np.quantile(yy, 0.02)), float(np.quantile(yy, 0.98)))
        for side, lm, idx in (("export", lim[g][0], 0), ("import", lim[g][1], 1)):
            beyond = (-yy if side == "export" else yy) > lm
            out.append(
                {
                    "network": args.network,
                    "group": g,
                    "source": "census",
                    "band": "0-240h",
                    "side": side,
                    "stat": "n_half_hours_beyond_scaled_limit",
                    "value": float(beyond.sum()),
                    "n_rows": int(yy.size),
                }
            )
        out.append(
            {
                "network": args.network,
                "group": g,
                "source": "census",
                "band": "0-240h",
                "side": "both",
                "stat": "posthoc_limits_export_import",
                "value": ph[g][0],
                "se_nominal": ph[g][1],
                "n_rows": int(yy.size),
            }
        )
    rng = np.random.default_rng(629)
    months = rows["origin_month"]
    for gi, g in enumerate(c.GROUP_IDS):
        y = rows["meter"][:, gi].astype(float)
        have = np.isfinite(y) & np.isfinite(rows["sums"][:, 0, gi, 0])
        sources: dict[str, dict] = {}
        common_ok = have.copy()
        for a, arm in enumerate(c.ARMS):
            for corr in (False, True):
                s = rows["sums"][:, a, gi, :].astype(float) + (shift[:, gi, None] if corr else 0.0)
                sources[arm + ("+g" if corr else "")] = {
                    "values": s,
                    "median": np.median(s, axis=1),
                    "q": {
                        t: np.quantile(s, t, axis=1, method="weibull")
                        for t in (0.05, 0.1, 0.5, 0.9, 0.95)
                    },
                }
        for k, dk in enumerate(dkeys):
            dq = rows["direct"][:, k, gi, :].astype(float)
            lv = {
                t: dq[:, int(np.argmin(np.abs(c.LEVELS - t)))] for t in (0.05, 0.1, 0.5, 0.9, 0.95)
            }
            sources["D:" + dk] = {
                "values": s434.fifty_one_values(quantiles=dq),
                "median": dq[:, 5],
                "q": lv,
            }
        for src in sources.values():
            common_ok &= np.isfinite(src["median"]) & np.isfinite(src["values"]).all(1)
        for name, src in sources.items():
            ok = common_ok
            for band, (lo, hi) in BANDS.items():
                bm = ok & (rows["lead"] > lo) & (rows["lead"] <= hi)
                for side in ("export", "import"):
                    for tau in TAUS:
                        if side == "import":
                            hit = y > src["q"][round(tau, 2)]
                        else:
                            hit = y < src["q"][round(1 - tau, 2)]
                        sel = bm
                        if sel.sum() < 100:
                            continue
                        rate = hit[sel].mean()
                        ndays = len(np.unique(vday[sel]))
                        se = (
                            np.sqrt((1 - tau) * tau / ndays)
                            if tau != 0.5
                            else np.sqrt(0.25 / ndays)
                        )
                        cnt = np.array(
                            [
                                [hit[sel & (months == m)].sum(), (sel & (months == m)).sum()]
                                for m in c.MONTHS
                            ],
                            float,
                        )
                        idx = rng.integers(0, len(c.MONTHS), size=(2000, len(c.MONTHS)))
                        rb = cnt[idx, 0].sum(1) / np.maximum(cnt[idx, 1].sum(1), 1)
                        out.append(
                            {
                                "network": args.network,
                                "group": g,
                                "source": name,
                                "band": band,
                                "side": side,
                                "stat": f"exceed_{tau}",
                                "value": float(rate),
                                "se_nominal": float(se),
                                "lo": float(np.quantile(rb, 0.025)),
                                "hi": float(np.quantile(rb, 0.975)),
                                "n_rows": int(sel.sum()),
                                "n_days": int(ndays),
                            }
                        )
                # twCRPS
                for side, limit, tag in (
                    ("export", lim[g][0], ""),
                    ("import", lim[g][1], ""),
                    ("export", ph[g][0], "_ph"),
                    ("import", ph[g][1], "_ph"),
                ):
                    sgn = -1.0 if side == "export" else 1.0
                    tw = twcrps(
                        x=sgn * src["values"][bm],
                        y=sgn * y[bm],
                        limit=limit,
                    )
                    for m in c.MONTHS:
                        sel_m = months[bm] == m
                        if sel_m.any():
                            out.append(
                                {
                                    "network": args.network,
                                    "group": g,
                                    "source": name,
                                    "band": band,
                                    "side": side,
                                    "stat": f"twcrps{tag}_month",
                                    "month": m,
                                    "value": float(tw[sel_m].mean()),
                                    "n_rows": int(sel_m.sum()),
                                }
                            )
                    crps = twcrps(x=src["values"][bm], y=y[bm], limit=-1e9)
                    out.append(
                        {
                            "network": args.network,
                            "group": g,
                            "source": name,
                            "band": band,
                            "side": "both",
                            "stat": "crps",
                            "value": float(crps.mean()),
                            "n_rows": int(bm.sum()),
                        }
                    )
            # bias by slice (all leads 0-240)
            for sl, mask in slices.items():
                sel = ok & mask
                if sel.sum() >= 100:
                    out.append(
                        {
                            "network": args.network,
                            "group": g,
                            "source": name,
                            "band": "0-240h",
                            "side": "both",
                            "stat": f"bias_median_{sl}",
                            "value": float((y[sel] - src["median"][sel]).mean()),
                            "n_rows": int(sel.sum()),
                        }
                    )
        # gap between the sum of Primaries and the meter
        po = rows["prim_obs_sum"][:, gi].astype(float)
        for sl, mask in slices.items():
            sel = np.isfinite(y) & np.isfinite(po) & mask
            if sel.sum() >= 100:
                out.append(
                    {
                        "network": args.network,
                        "group": g,
                        "source": "gap_meter_minus_primaries_observed",
                        "band": "0-240h",
                        "side": "both",
                        "stat": f"gap_{sl}",
                        "value": float((y[sel] - po[sel]).mean()),
                        "n_rows": int(sel.sum()),
                    }
                )
    table = pl.DataFrame(out, infer_schema_length=None)
    table.write_parquet(c.WORK / f"scores_{args.network}.parquet")
    keep = {
        k: rows[k]
        for k in ("meter", "prim_obs_sum", "median_sum", "lead", "valid", "origin_month", "init")
    }
    np.savez_compressed(c.WORK / f"rows_{args.network}.npz", shift=shift, **keep)
    print(table.height, "score rows written")


if __name__ == "__main__":
    main()

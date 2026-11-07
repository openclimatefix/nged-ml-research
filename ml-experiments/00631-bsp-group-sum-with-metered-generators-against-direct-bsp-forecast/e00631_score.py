# ruff: noqa: E501, PLR0912, PLR0915
"""Score the BSP sums of experiment 631 against the BSP meters.

Reads `work/runs/*.npz` and `work/offsets.npz` (written by `e00631_build.py`) and writes
`work/scores.parquet` (one row per row set, BSP, source, lead band, side, and statistic) and
`work/rows.npz` (the per-row quantities the figures use).

Sources are named `<arm>|<variant>|<remainder>`: arm J0, J2, J3, or J5; variant `wo` (Primaries
only) or `w` (Primaries minus the metered farms); remainder `none`, `const`, or `tod` (the offsets
of `offsets.npz`: the mean of the observed meter minus the observed combination, constant or by
half-hour of day, over the history before the test month). `D:<key>` are the stored direct BSP forecasts.

Row sets. `common`: half-hours where the meter, every sum, and every direct forecast have a value
(the farms' stored forecasts omit idle-wind and missing-reading half-hours, so `common` is smaller than
the Primaries' full row set). `full`: the same without the farms' sources, for the sources that
need no farm (this repeats experiment 629's sums on a new Schaake pool).

Conventions are experiment 629's: flow into the BSP is positive; the quantile of 51 scenario sums is
numpy's `weibull` quantile; the export side scores the negated flow; the independent unit of an
exceedance rate is the valid calendar day. twCRPS uses the chaining function max(x, L) with L the
90th percentile of the positive exports (imports) of the meter in the `common` rows (post hoc).
"""

import argparse
import os
from datetime import UTC, datetime

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00434_score as s434  # noqa: E402  # ty: ignore[unresolved-import]
import e00629_common as c629  # noqa: E402  # ty: ignore[unresolved-import]
import e00629_score as s629  # noqa: E402  # ty: ignore[unresolved-import]
import e00631_common as c  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

BANDS = s629.BANDS
ARMS = c629.ARMS
SLICES_HOURS = {
    "summer midday": (10, 14, (4, 5, 6, 7, 8, 9)),
    "winter evening": (16, 20, (11, 12, 1, 2)),
}
REMAINDERS = ("none", "const", "tod")


def load_rows() -> dict:
    """Return all runs stacked along a leading row axis, with the row labels."""
    files = sorted((c.WORK / "runs").glob("*.npz"))
    cols: dict[str, list] = {
        k: []
        for k in (
            "sums",
            "meter",
            "prim_obs_sum",
            "farm_obs_sum",
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
        ok = c629.valid_mask(init=init)
        cols["sums"].append(
            z["sums"][..., ok].transpose(4, 0, 1, 2, 3)
        )  # (rows, arm, variant, bsp, 51)
        for k in ("meter", "prim_obs_sum", "farm_obs_sum"):
            cols[k].append(z[k][:, ok].T)
        cols["direct"].append(z["direct"][:, :, ok].transpose(2, 0, 1, 3))  # (rows, src, bsp, 11)
        steps = np.flatnonzero(ok)
        cols["lead"].append((steps + 1) / 2.0)
        cols["valid"].append(
            np.array(
                [np.datetime64((init + (s + 1) * c.STEP).replace(tzinfo=None), "m") for s in steps]
            )  # ty: ignore[unsupported-operator]
        )
        cols["origin_month"].append(np.full(steps.size, str(z["month"])))
        cols["init"].append(np.full(steps.size, np.datetime64(init.replace(tzinfo=None), "m")))
    return {k: np.concatenate(v) for k, v in cols.items()}


def remainder_shift(*, rows: dict, kind: str, variant: int, gi: int, offsets: dict) -> np.ndarray:
    """Return the remainder term per row (MW) for one BSP, variant (0 wo, 1 w), and kind."""
    if kind == "none":
        return np.zeros(rows["lead"].size)
    valid = rows["valid"].astype("datetime64[m]")
    hod = ((valid - valid.astype("datetime64[D]")) / np.timedelta64(30, "m")).astype(int)
    out = np.zeros(rows["lead"].size)
    for m in c.MONTHS:
        sel = rows["origin_month"] == m
        if kind == "const":
            out[sel] = offsets[f"const_{m}"][variant, gi]
        else:
            out[sel] = offsets[f"tod_{m}"][variant, gi][hod[sel]]
    return out


def exceed(*, y: np.ndarray, q: np.ndarray, side: str) -> np.ndarray:
    """Return whether the actual is beyond the 5% tail on one side of the sum's quantile."""
    return y < q[0.05] if side == "export" else y > q[0.95]  # ty: ignore[invalid-argument-type]


def main() -> None:
    """Score every source at both BSPs and write the long table."""
    argparse.ArgumentParser().parse_args()
    rows = load_rows()
    offsets = dict(np.load(c.WORK / "offsets.npz"))
    lim = s629.limits()
    n = rows["lead"].size
    valid = rows["valid"]
    vhour = ((valid - valid.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(float)
    vmonth = valid.astype("datetime64[M]").astype(int) % 12 + 1
    vday = valid.astype("datetime64[D]")
    months = rows["origin_month"]
    slices = {"all": np.ones(n, bool)}
    for name, (h0, h1, mons) in SLICES_HOURS.items():
        slices[name] = (vhour >= h0) & (vhour < h1) & np.isin(vmonth, list(mons))
    dkeys = list(c629.DIRECT)
    out: list[dict] = []
    rng = np.random.default_rng(631)
    for gi, g in enumerate(c.BSP_GROUPS):
        y = rows["meter"][:, gi].astype(float)
        sources: dict[str, dict] = {}
        for a, arm in enumerate(ARMS):
            for variant, vname in enumerate(("wo", "w")):
                base = rows["sums"][:, a, variant, gi, :].astype(float)
                if np.isnan(base).all():
                    continue
                for kind in REMAINDERS:
                    s = (
                        base
                        + remainder_shift(
                            rows=rows, kind=kind, variant=variant, gi=gi, offsets=offsets
                        )[:, None]
                    )
                    sources[f"{arm}|{vname}|{kind}"] = {
                        "values": s,
                        "median": np.median(s, axis=1),
                        "q": {
                            t: np.quantile(s, t, axis=1, method="weibull")
                            for t in (0.05, 0.5, 0.95)
                        },
                        "needs_farms": vname == "w",
                    }
        for k, dk in enumerate(dkeys):
            dq = rows["direct"][:, k, gi, :].astype(float)
            sources["D:" + dk] = {
                "values": s434.fifty_one_values(quantiles=dq),
                "median": dq[:, 5],
                "q": {t: dq[:, int(np.argmin(np.abs(c629.LEVELS - t)))] for t in (0.05, 0.5, 0.95)},
                "needs_farms": False,
            }
        full = np.isfinite(y)
        common = np.isfinite(y)
        for src in sources.values():
            fin = np.isfinite(src["median"]) & np.isfinite(src["values"]).all(1)
            common &= fin
            if not src["needs_farms"]:
                full &= fin
        # the post hoc limits: 90th percentile of positive exports and imports in the common rows
        yy = y[common]
        l_exp = float(np.quantile(-yy[yy < 0], 0.9)) if (yy < 0).any() else float("nan")
        l_imp = float(np.quantile(yy[yy > 0], 0.9))
        out.append(
            {
                "rowset": "common",
                "group": g,
                "source": "census",
                "band": "0-240h",
                "side": "both",
                "stat": "limits_posthoc_export_import_scaled_export_scaled_import",
                "value": l_exp,
                "se_nominal": l_imp,
                "lo": lim[g][0],
                "hi": lim[g][1],
                "n_rows": int(yy.size),
            }
        )
        out.append(
            {
                "rowset": "common",
                "group": g,
                "source": "census",
                "band": "0-240h",
                "side": "export",
                "stat": "n_half_hours_beyond_scaled_limit",
                "value": float((-yy > lim[g][0]).sum()),
                "n_rows": int(yy.size),
            }
        )
        out.append(
            {
                "rowset": "common",
                "group": g,
                "source": "census",
                "band": "0-240h",
                "side": "both",
                "stat": "n_rows_full_and_common",
                "value": float(full.sum()),
                "se_nominal": float(common.sum()),
                "n_rows": int(np.isfinite(y).sum()),
            }
        )
        for rowset, ok in (("common", common), ("full", full)):
            for name, src in sources.items():
                if rowset == "full" and src["needs_farms"]:
                    continue
                for band, (lo, hi) in BANDS.items():
                    bm = ok & (rows["lead"] > lo) & (rows["lead"] <= hi)
                    if bm.sum() < 100:
                        continue
                    for side in ("export", "import"):
                        tau = 0.05
                        hit = exceed(y=y, q=src["q"], side=side)
                        rate = hit[bm].mean()
                        ndays = len(np.unique(vday[bm]))
                        cnt = np.array(
                            [
                                [hit[bm & (months == m)].sum(), (bm & (months == m)).sum()]
                                for m in c.MONTHS
                            ],
                            float,
                        )
                        idx = rng.integers(0, len(c.MONTHS), size=(2000, len(c.MONTHS)))
                        rb = cnt[idx, 0].sum(1) / np.maximum(cnt[idx, 1].sum(1), 1)
                        out.append(
                            {
                                "rowset": rowset,
                                "group": g,
                                "source": name,
                                "band": band,
                                "side": side,
                                "stat": "exceed_0.05",
                                "value": float(rate),
                                "se_nominal": float(np.sqrt(tau * (1 - tau) / ndays)),
                                "lo": float(np.quantile(rb, 0.025)),
                                "hi": float(np.quantile(rb, 0.975)),
                                "n_rows": int(bm.sum()),
                                "n_days": int(ndays),
                            }
                        )
                    crps = s629.twcrps(x=src["values"][bm], y=y[bm], limit=-1e9)
                    for m in c.MONTHS:
                        sel_m = months[bm] == m
                        if sel_m.any():
                            out.append(
                                {
                                    "rowset": rowset,
                                    "group": g,
                                    "source": name,
                                    "band": band,
                                    "side": "both",
                                    "stat": "crps_month",
                                    "month": m,
                                    "value": float(crps[sel_m].mean()),
                                    "n_rows": int(sel_m.sum()),
                                }
                            )
                    for side, limit in (("export", l_exp), ("import", l_imp)):
                        sgn = -1.0 if side == "export" else 1.0
                        tw = s629.twcrps(x=sgn * src["values"][bm], y=sgn * y[bm], limit=limit)
                        for m in c.MONTHS:
                            sel_m = months[bm] == m
                            if sel_m.any():
                                out.append(
                                    {
                                        "rowset": rowset,
                                        "group": g,
                                        "source": name,
                                        "band": band,
                                        "side": side,
                                        "stat": "twcrps_month",
                                        "month": m,
                                        "value": float(tw[sel_m].mean()),
                                        "n_rows": int(sel_m.sum()),
                                    }
                                )
                for sl, mask in slices.items():
                    sel = ok & mask
                    if sel.sum() >= 100:
                        out.append(
                            {
                                "rowset": rowset,
                                "group": g,
                                "source": name,
                                "band": "0-240h",
                                "side": "both",
                                "stat": f"bias_median_{sl}",
                                "value": float((y[sel] - src["median"][sel]).mean()),
                                "n_rows": int(sel.sum()),
                            }
                        )
        # observed remainder: meter minus Primaries minus farms, by slice, in the common rows
        po, fo = (
            rows["prim_obs_sum"][:, gi].astype(float),
            rows["farm_obs_sum"][:, gi].astype(float),
        )
        for sl, mask in slices.items():
            sel = np.isfinite(y) & np.isfinite(po) & np.isfinite(fo) & mask
            if sel.sum() >= 100:
                out.append(
                    {
                        "rowset": "observed",
                        "group": g,
                        "source": "observed_remainder",
                        "band": "0-240h",
                        "side": "both",
                        "stat": f"gap_wo_{sl}",
                        "value": float((y[sel] - po[sel]).mean()),
                        "n_rows": int(sel.sum()),
                    }
                )
                out.append(
                    {
                        "rowset": "observed",
                        "group": g,
                        "source": "observed_remainder",
                        "band": "0-240h",
                        "side": "both",
                        "stat": f"gap_w_{sl}",
                        "value": float((y[sel] - po[sel] + fo[sel]).mean()),
                        "n_rows": int(sel.sum()),
                    }
                )
                out.append(
                    {
                        "rowset": "observed",
                        "group": g,
                        "source": "observed_remainder",
                        "band": "0-240h",
                        "side": "both",
                        "stat": f"farm_output_{sl}",
                        "value": float(fo[sel].mean()),
                        "n_rows": int(sel.sum()),
                    }
                )
    table = pl.DataFrame(out, infer_schema_length=None)
    table.write_parquet(c.WORK / "scores.parquet")
    keep = {
        k: rows[k]
        for k in ("meter", "prim_obs_sum", "farm_obs_sum", "lead", "valid", "origin_month", "init")
    }
    np.savez_compressed(c.WORK / "rows.npz", **keep)
    print(table.height, "score rows written")


if __name__ == "__main__":
    main()

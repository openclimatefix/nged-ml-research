# ruff: noqa: E501, D103, PLR0912, PLR0915
"""Experiment 578: DESCRIPTIVE gates G1 and G2 with action A2 (mask the lags) on stored forecasts.

Nothing is tuned: m = 3 days, k = 3 event-free standard deviations, theta = 5% of capacity, release after 2
days below half the threshold. The lag-free expectation is the lags-masked v3+D5 day-ahead forecast (24-48 h
lead), so it exists in every fold and is issued at least 24 h before its target. A gate evaluated at origin O
(09:00 UTC on day D) reads only the daily mean gap (observed minus expectation) of the 3 complete UTC days
D-3 to D-1, observations at least 9 h old (the brief's limit is 3.5 h). G1: own gap sustained with one sign
and above both thresholds. G2: G1 and a neighbour (Primary Substation Interconnections.csv, metered
neighbours) with a sustained gap of the opposite sign above the same thresholds. A2: for a gated origin, the
whole forecast is replaced by the lags-masked twin's. Writes `work/gate_events.csv` and `work/gate_cost.csv`.
"""

import datetime as dt

import e00400_blend as b400  # ty: ignore[unresolved-import]
import e00578_common as k  # ty: ignore[unresolved-import]
import e00578_nra as nra  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

M, K_SD, THETA, RELEASE_DAYS = 3, 3.0, 0.05, 2


def daily_gap(*, j: pl.DataFrame) -> np.ndarray:
    """Return (16, days) daily mean of (observed - lags-masked day-ahead forecast), NaN with under 40 half-hours."""
    d = (
        j.filter((pl.col("lead") >= 24) & (pl.col("lead") < 48))
        .group_by("prim", "idx")
        .agg(gap=(pl.col("obs") - pl.col("r")).mean())
    )
    n_days = len(k.clean()[0]) // k.PER_DAY
    s = np.zeros((16, n_days))
    c = np.zeros((16, n_days))
    p, i, g = d["prim"].to_numpy(), d["idx"].to_numpy() // k.PER_DAY, d["gap"].to_numpy()
    ok = np.isfinite(g)
    np.add.at(s, (p[ok], i[ok]), g[ok])
    np.add.at(c, (p[ok], i[ok]), 1)
    return np.where(c >= 40, s / np.maximum(c, 1) / 1.0, np.nan) * 0 + np.where(
        c >= 40, s / np.maximum(c, 1), np.nan
    )


def main() -> None:
    j = nra.table()
    gap = daily_gap(j=j)
    ev_all = pl.read_parquet(k.EVENTS)
    sm = k.sites()
    lab2p = dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True))
    nbrs = {
        lab2p[a]: [lab2p[x] for x in v if x in lab2p]
        for a, v in b400.nged_neighbours().items()
        if a in lab2p
    }
    n_days = gap.shape[1]
    near = np.zeros((16, n_days), bool)
    for r in ev_all.iter_rows(named=True):
        near[lab2p[r["label"]], max(r["onset_idx"] // 48 - 3, 0) : r["end_idx"] // 48 + 4] = True
    sd = np.array([np.nanstd(gap[p][~near[p]]) for p in range(16)])
    thr = np.maximum(K_SD * sd, THETA)
    origin_days = sorted(
        {(o - k.GRID_START).days for o in j["power_fcst_init_time"].unique().to_list()}
    )

    def sustained(p: int, d: int, sign: float | None) -> float:
        g = gap[p, d - M : d]
        if not np.isfinite(g).all():
            return 0.0
        s = np.sign(g)
        if (s != s[0]).any() or np.abs(g).min() <= thr[p]:
            return 0.0
        return float(s[0]) if sign is None or s[0] == sign else 0.0

    state = {v: np.zeros((16, n_days), bool) for v in ("G1", "G2")}
    for variant in ("G1", "G2"):
        for p in range(16):
            on = False
            for d in origin_days:
                s = sustained(p, d, None)
                if variant == "G2" and s != 0:
                    s = s if any(sustained(q, d, -s) != 0 for q in nbrs.get(p, [])) else 0.0
                if not on and s != 0:
                    on = True
                elif on:
                    g = gap[p, d - RELEASE_DAYS : d]
                    if np.isfinite(g).all() and (np.abs(g) < 0.5 * thr[p]).all():
                        on = False
                state[variant][p, d] = on
    # evaluation on the 12 long events
    ev = pl.read_csv(k.WORK / "step0_events.csv").filter(pl.col("long"))
    j24 = j.filter((pl.col("lead") >= 24) & (pl.col("lead") < 120)).with_columns(
        od=((pl.col("power_fcst_init_time") - k.GRID_START).dt.total_days()).cast(pl.Int64)
    )
    rows, cost = [], []
    for q in ev.iter_rows(named=True):
        p = q["primary"]
        t0 = k.GRID_START + dt.timedelta(minutes=30 * q["onset_idx"])
        t1 = k.GRID_START + dt.timedelta(minutes=30 * q["end_idx"])
        depth = q["depth_share"] / 100 * np.sign(q["depth_mw"])
        d0 = j24.filter(
            (pl.col("prim") == p) & pl.col("valid_time").is_between(t0, t1 + dt.timedelta(days=7))
        )
        if d0.height < 200:
            continue
        for variant in ("G1", "G2"):
            on = state[variant][p]
            d1 = d0.with_columns(gated=pl.Series(on[d0["od"].to_numpy()])).with_columns(
                a2=pl.when(pl.col("gated")).then(pl.col("r")).otherwise(pl.col("v"))
            )
            for part, sel in (
                ("during", pl.col("valid_time") <= t1),
                ("after", pl.col("valid_time") > t1),
            ):
                s = d1.filter(sel)
                if s.height < 100:
                    continue
                days_on = sorted({d for d in s["od"].unique().to_list() if on[d]})
                first = (
                    next((d - (t0 - k.GRID_START).days for d in days_on), None)
                    if part == "during"
                    else None
                )
                rows.append(
                    {
                        "event_id": q["event_id"],
                        "label": q["label"],
                        "variant": variant,
                        "part": part,
                        "share_of_origin_days_gated": float(
                            s.group_by("od").agg(pl.col("gated").first())["gated"].mean()  # ty: ignore[invalid-argument-type]
                        ),
                        "first_trigger_day_after_onset": first,
                        "follow_ungated": float(((s["v"] - s["c"]) / depth).mean()),
                        "follow_A2": float(((s["a2"] - s["c"]) / depth).mean()),
                        "error_ungated": float((s["v"] - s["obs"]).abs().mean()) / abs(depth),  # ty: ignore[invalid-argument-type]
                        "error_A2": float((s["a2"] - s["obs"]).abs().mean()) / abs(depth),  # ty: ignore[invalid-argument-type]
                    }
                )
    # false triggers and cost on event-free days (not within 3 days of any labelled event)
    for variant in ("G1", "G2"):
        free_days = [
            (p, d)
            for p in range(16)
            for d in origin_days
            if not near[p, d] and np.isfinite(gap[p, d - M : d]).all()
        ]
        trig = [(p, d) for p, d in free_days if state[variant][p, d]]
        years = len(free_days) / 365.25
        ff = j24.filter(~pl.col("valid_time").is_null())
        mask_free = np.zeros((16, n_days), bool)
        for p, d in free_days:
            mask_free[p, d] = True
        sel = ff.filter(pl.Series(mask_free[ff["prim"].to_numpy(), ff["od"].to_numpy()]))
        g = state[variant][sel["prim"].to_numpy(), sel["od"].to_numpy()]
        a2 = np.where(g, sel["r"].to_numpy(), sel["v"].to_numpy())
        obs = sel["obs"].to_numpy()
        cost.append(
            {
                "variant": variant,
                "event_free_origin_days": len(free_days),
                "gated_origin_days": len(trig),
                "false_triggers_per_primary_year": len(trig) / years,
                "mae_ungated": float(np.abs(sel["v"].to_numpy() - obs).mean()),
                "mae_A2": float(np.abs(a2 - obs).mean()),
            }
        )
    r = pl.DataFrame(rows)
    r.write_csv(k.WORK / "gate_events.csv")
    c = pl.DataFrame(cost)
    c.write_csv(k.WORK / "gate_cost.csv")
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=250):
        print(
            pl.DataFrame(
                {"label": sm["label"], "event_free_gap_sd": sd, "threshold": thr}
            ).with_columns(pl.col(pl.Float64).round(3))
        )
        print(r.with_columns(pl.col(pl.Float64).round(3)))
        print(c)
        for v in ("G1", "G2"):
            for part in ("during", "after"):
                s = r.filter((pl.col("variant") == v) & (pl.col("part") == part))
                print(
                    v,
                    part,
                    "events",
                    s.height,
                    "events triggered",
                    int((s["share_of_origin_days_gated"] > 0).sum()),
                    "mean follow ungated",
                    round(float(s["follow_ungated"].mean()), 3),  # ty: ignore[invalid-argument-type]
                    "A2",
                    round(float(s["follow_A2"].mean()), 3),  # ty: ignore[invalid-argument-type]
                    "mean error ungated",
                    round(float(s["error_ungated"].mean()), 3),  # ty: ignore[invalid-argument-type]
                    "A2",
                    round(float(s["error_A2"].mean()), 3),  # ty: ignore[invalid-argument-type]
                )


main()

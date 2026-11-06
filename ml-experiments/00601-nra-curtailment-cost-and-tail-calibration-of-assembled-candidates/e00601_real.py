# ruff: noqa: E501, D103, PLW2901
"""Experiment 601, real data: curtailment cost and tail calibration of the stored forecasts.

No fit. Reads the stored exports (`backtest_export_2026-11-06` for C0 and E2,
`backtest_export_candidates_2026-11-06`
for C1, C1 + G4, A1, and A1 + G4), keeps the rows every system holds, and writes under
`data/ml-experiments/00601-*/`:

- `limits.csv`: each six-MW Primary's limit at f = 0.9, 0.75, 0.5 (f times the largest metered
  power before 2025-10-01).
- `known_answer.csv`: the per-row cost summed, against `metrics.avoidable_curtailment_cost`, on
  every system.
- `tail_real.csv`: tail calibration per system, group, row set, and lead band.
- `cost_real.csv`, `cost_real_per_primary.csv`: avoidable cost per system, row set, and fraction
  (six MW Primaries).
- `contrast_real.csv`: (gated minus plain) contrasts of the cost and the tail pinball losses,
  month blocks.
- `real_pair_phase.csv`: the December 2025 pair (D5, D16) by phase.

Usage: python e00601_real.py
"""

from datetime import UTC, datetime

import e00601_common as k
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.power import scan_power

SCORED_START = datetime(2025, 10, 1, tzinfo=UTC)
BANDS = {"0-24 h": (0.0, 24.0), "24-120 h": (24.0, 120.0)}
ROW_SETS = {  # name -> expression on the stored columns
    "event-free": pl.col("event_free"),
    "event rows": ~pl.col("event_free"),
    "all rows": pl.lit(value=True),
}
PHASES = ("before", "days0-3", "days4-end", "after0-7", "after7-14")
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]


def load_system(*, name: str) -> pl.DataFrame:
    folder, prefix = k.SYSTEMS[name]
    return pl.concat(
        [pl.read_parquet(folder / f"{prefix}_{m}.parquet") for m in k.MONTHS]
    ).with_columns(time_series_id=pl.col("time_series_id").cast(pl.Int64))


def limits_table(*, subs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    six = subs.filter(pl.col("group") == k.SIX)
    flows = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(six["time_series_id"].to_list()))
        .select("time_series_id", valid_time="time", flow_mw="power")
        .collect()
        .with_columns(pl.col("time_series_id").cast(pl.Int64))
    )
    peak = (
        flows.filter(pl.col("valid_time") < SCORED_START, pl.col("flow_mw").is_finite())
        .group_by("time_series_id")
        .agg(max_mw=pl.col("flow_mw").max(), first=pl.col("valid_time").min())
    )
    out = six.join(peak, on="time_series_id").select(
        "time_series_id", "primary_name", "max_mw", "first"
    )
    for f in k.FRACTIONS:
        out = out.with_columns((pl.col("max_mw") * f).alias(f"limit_{f}"))
    return out.sort("primary_name"), flows


def add_row_metrics(*, df: pl.DataFrame, lim: pl.DataFrame) -> pl.DataFrame:
    """Add the exceedance indicators, pinball losses, over/under shares, and the avoidable cost."""
    df = df.join(
        lim.select("time_series_id", *[f"limit_{f}" for f in k.FRACTIONS]),
        on="time_series_id",
        how="left",
    )
    y = df["nra_power"].to_numpy().astype(np.float64)
    yo = df["observed_power"].to_numpy().astype(np.float64)
    q = {c: df[c].to_numpy().astype(np.float64) for c in k.Q}

    def pin(tau: float, c: str) -> np.ndarray:
        d = y - q[c]
        return np.maximum(tau * d, (tau - 1) * d)

    cols = {
        "ex95": (y > q["q95"]).astype(float), "ex99": (y > q["q99"]).astype(float),
        "lo05": (y < q["q05"]).astype(float), "lo01": (y < q["q01"]).astype(float),
        "over": (q["q50"] > y).astype(float), "under": (q["q50"] < y).astype(float),
        "pin05": pin(0.05, "q05"), "pin95": pin(0.95, "q95"), "pin99": pin(0.99, "q99"),
        "cover90": ((q["q05"] <= y) & (y <= q["q95"])).astype(float),
    }  # fmt: skip
    for f in k.FRACTIONS:
        lim_f = df[f"limit_{f}"].to_numpy().astype(np.float64)
        cols[f"cost_{f}"] = k.avoidable_cost(
            u=q["q95"], y=yo, limit=lim_f
        )  # NaN where no limit (not six MW)
    in_cost = (df["lead_hours"].to_numpy() >= k.LEAD_COST[0]) & (
        df["lead_hours"].to_numpy() < k.LEAD_COST[1]
    )
    return df.with_columns(pl.Series(n, v) for n, v in cols.items()).with_columns(
        in_cost=pl.Series(in_cost), month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
    )


def known_answer(
    *, frames: dict[str, pl.DataFrame], flows: pl.DataFrame, subs: pl.DataFrame
) -> pl.DataFrame:
    """Compare the per-row cost with the metric itself, on every system and fraction (all six-MW rows)."""
    six = subs.filter(pl.col("group") == k.SIX).select("time_series_id", label="primary_name")
    out = []
    for f in k.FRACTIONS:
        fc = {
            n: d.filter(pl.col("group") == k.SIX, pl.col("observed_power").is_finite())
            .select(*KEYS, *k.Q, actual="observed_power")
            .with_columns(pl.col(c).cast(pl.Float64) for c in ("actual", *k.Q))
            for n, d in frames.items()
        }
        res = metrics.avoidable_curtailment_cost(
            forecasts=fc, series=six, flows=flows, scored_start=SCORED_START, fraction=f
        )
        mine = {
            n: d.filter(
                pl.col("group") == k.SIX, pl.col("in_cost"), pl.col("observed_power").is_finite()
            )[f"cost_{f}"].sum()
            for n, d in frames.items()
        }
        for n, total in res.group_by("arm").agg(pl.col("avoidable_cost_gbp").sum()).iter_rows():
            out.append(
                {
                    "fraction": f,
                    "system": n,
                    "module_gbp": total,
                    "per_row_gbp": mine[n],
                    "diff_gbp": total - mine[n],
                }
            )
    return pl.DataFrame(out)


def block_ratio(
    *, num: np.ndarray, den: np.ndarray, blocks: np.ndarray, alpha: float
) -> tuple[float, float, float]:
    """Return a ratio of sums (mean per row) and its block bootstrap interval; `blocks` labels the rows."""
    keys, inv = np.unique(blocks, return_inverse=True)
    s = np.bincount(inv, weights=num, minlength=len(keys))
    n = np.bincount(inv, weights=den, minlength=len(keys))
    rng = np.random.default_rng(601)
    idx = rng.integers(0, len(keys), size=(k.N_BOOT, len(keys)))
    r = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return (
        float(s.sum() / n.sum()),
        float(np.quantile(r, alpha / 2)),
        float(np.quantile(r, 1 - alpha / 2)),
    )


def summarise(
    *, frames: dict[str, pl.DataFrame]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    tail, cost, per_p = [], [], []
    for name, d in frames.items():
        for gname in (k.EIGHT, k.SIX, k.REPORT_ONLY):
            g = d.filter(pl.col("group") == gname)
            for rs, expr in ROW_SETS.items():
                sub = g.filter(expr)
                for bname, (lo, hi) in BANDS.items():
                    b = sub.filter(pl.col("lead_hours") >= lo, pl.col("lead_hours") < hi)
                    if b.height < 200:
                        continue
                    rec = {
                        "system": name,
                        "group": gname,
                        "row_set": rs,
                        "band": bname,
                        "n_rows": b.height,
                    }
                    blocks = (b["time_series_id"].cast(pl.String) + b["month"]).to_numpy()
                    for m in (
                        "ex95",
                        "ex99",
                        "lo05",
                        "lo01",
                        "over",
                        "under",
                        "pin05",
                        "pin95",
                        "pin99",
                        "cover90",
                    ):
                        v = b[m].to_numpy()
                        mean, lo_, hi_ = block_ratio(
                            num=v, den=np.ones(len(v)), blocks=blocks, alpha=0.05
                        )
                        rec[m], rec[f"{m}_lo"], rec[f"{m}_hi"] = mean, lo_, hi_  # ty: ignore[invalid-assignment]
                    tail.append(rec)
                if gname != k.SIX:
                    continue
                c = sub.filter(pl.col("in_cost"), pl.col("observed_power").is_finite())
                for f in k.FRACTIONS:
                    col = f"cost_{f}"
                    v = c[col].to_numpy()
                    blocks = (c["time_series_id"].cast(pl.String) + c["month"]).to_numpy()
                    mean, lo_, hi_ = block_ratio(
                        num=v, den=np.ones(len(v)), blocks=blocks, alpha=0.05
                    )
                    cost.append({
                        "system": name, "row_set": rs, "fraction": f, "n_rows": c.height, "total_gbp": float(v.sum()),
                        "gbp_per_primary_day": mean * 48, "lo": lo_ * 48, "hi": hi_ * 48,
                        "n_exceed_rows": int((c["observed_power"].to_numpy() > c[f"limit_{f}"].to_numpy()).sum()),
                    })  # fmt: skip
                    for (sid, pn), pp in c.group_by("time_series_id", "primary_name"):
                        per_p.append(
                            {
                                "system": name,
                                "row_set": rs,
                                "fraction": f,
                                "time_series_id": sid,
                                "primary": pn,
                                "n_rows": pp.height,
                                "gbp_per_primary_day": float(pp[col].mean()) * 48,  # ty: ignore[invalid-argument-type]
                            }
                        )
    return pl.DataFrame(tail), pl.DataFrame(cost), pl.DataFrame(per_p)


def contrasts(*, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Gated minus plain, on shared rows: cost (six MW) and the tail pinball losses, by month blocks."""
    out = []
    for plain, gated in (("C1", "C1 + G4"), ("A1", "A1 + G4")):
        for gname in (k.SIX, k.EIGHT):
            for rs, expr in (
                ("event-free", ROW_SETS["event-free"]),
                ("event rows", ROW_SETS["event rows"]),
            ):
                a = frames[plain].filter(pl.col("group") == gname).filter(expr).sort(KEYS)
                b = frames[gated].filter(pl.col("group") == gname).filter(expr).sort(KEYS)
                assert a.height == b.height
                measures = [
                    ("pin05", None),
                    ("pin95", None),
                    ("pin99", None),
                    ("ex95", None),
                    ("ex99", None),
                ]
                if gname == k.SIX:
                    measures += [(f"cost_{f}", True) for f in k.FRACTIONS]
                for m, is_cost in measures:
                    sel = (
                        (b["in_cost"] & b["observed_power"].is_finite()).to_numpy()
                        if is_cost
                        else (b["lead_hours"] >= 24).to_numpy()
                    )
                    d = (b[m].to_numpy() - a[m].to_numpy())[sel]
                    blocks = b["month"].to_numpy()[sel]
                    scale = 48.0 if is_cost else 1.0
                    for level, alpha in (("95%", 0.05), ("98.75%", k.BONFERRONI_ALPHA)):
                        mean, lo_, hi_ = block_ratio(
                            num=d, den=np.ones(len(d)), blocks=blocks, alpha=alpha
                        )
                        out.append({"plain": plain, "gated": gated, "group": gname, "row_set": rs, "measure": m, "level": level, "n_rows": int(sel.sum()), "diff": mean * scale, "lo": lo_ * scale, "hi": hi_ * scale, "mde": (hi_ - lo_) / 2 * scale})  # fmt: skip
    return pl.DataFrame(out)


def real_pair(*, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The December pair by phase; the event window is the verified one (rows with `event_id`)."""
    out = []
    ref = frames["C1"]
    win = (
        ref.filter(pl.col("event_id").is_not_null())
        .group_by("time_series_id")
        .agg(a=pl.col("valid_time").min(), b=pl.col("valid_time").max())
    )
    day = pl.duration(days=1)
    for sid, a, b in win.iter_rows():
        spans = {
            "before": (a - 7 * day, a), "days0-3": (a, a + 3 * day), "days4-end": (a + 3 * day, b),
            "after0-7": (b, b + 7 * day), "after7-14": (b + 7 * day, b + 14 * day),
        }  # fmt: skip
        for name, d in frames.items():
            for phase, (lo, hi) in spans.items():
                inside = d.filter(
                    pl.col("time_series_id") == sid,
                    pl.col("valid_time") >= lo,
                    pl.col("valid_time") < hi,
                )
                p = inside.filter(pl.col("lead_hours") >= 24)
                if p.height < 100:
                    continue
                c = inside.filter(pl.col("in_cost"), pl.col("observed_power").is_finite())
                out.append({
                    "primary": ref.filter(pl.col("time_series_id") == sid)["time_series_id"][0],
                    "time_series_id": sid, "system": name, "phase": phase, "n_rows": p.height,
                    **{m: float(p[m].mean()) for m in ("ex95", "ex99", "pin05", "pin95", "pin99", "over", "under", "cover90")},  # ty: ignore[invalid-argument-type]
                    **{f"cost_gbp_per_day_{f}": float(c[f"cost_{f}"].mean() * 48) if c.height else None for f in k.FRACTIONS},  # ty: ignore[unsupported-operator]
                    "n_cost_rows": c.height,
                })  # fmt: skip
    return pl.DataFrame(out)


def main() -> None:
    k.WORK.mkdir(parents=True, exist_ok=True)
    subs = pl.read_csv(k.EXPORT / "substations.csv").with_columns(
        pl.col("time_series_id").cast(pl.Int64)
    )
    lim, flows = limits_table(subs=subs)
    lim.write_csv(k.WORK / "limits.csv")
    print(lim, flush=True)
    raw = {n: load_system(name=n) for n in k.SYSTEMS}
    keys = None
    for d in raw.values():
        s = d.select(KEYS)
        keys = s if keys is None else keys.join(s, on=KEYS, how="semi")
    print(
        "rows held by every system:",
        keys.height,  # ty: ignore[unresolved-attribute]
        "of",
        {n: d.height for n, d in raw.items()},
        flush=True,
    )
    frames = {}
    for n, d in raw.items():
        d = d.join(keys, on=KEYS, how="semi").join(  # ty: ignore[invalid-argument-type]
            subs.select("time_series_id", "group", "capacity", "primary_name"), on="time_series_id"
        )
        d = d.filter(pl.col("nra_power").is_finite())
        frames[n] = add_row_metrics(df=d, lim=lim).sort(KEYS)
    ka = known_answer(frames=frames, flows=flows, subs=subs)
    ka.write_csv(k.WORK / "known_answer.csv")
    print(ka, flush=True)
    tail, cost, per_p = summarise(frames=frames)
    tail.write_csv(k.WORK / "tail_real.csv")
    cost.write_csv(k.WORK / "cost_real.csv")
    per_p.write_csv(k.WORK / "cost_real_per_primary.csv")
    contrasts(frames=frames).write_csv(k.WORK / "contrast_real.csv")
    real_pair(frames=frames).write_csv(k.WORK / "real_pair_phase.csv")
    print("done", flush=True)


if __name__ == "__main__":
    main()

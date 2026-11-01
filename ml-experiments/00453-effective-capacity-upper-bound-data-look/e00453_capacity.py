"""Experiment 453, step 3: compare effective-capacity rules on the stored forecasts of 429 and 430.

For each test month and each series, the capacity under a rule is the maximum, the 99.99th, the
99.9th, or the 99th percentile of the series' cleaned power in the months before the test month (no
test-month data), or the stored whole-record 99th percentile (`current`, which uses later data).
The stored quantiles and the actuals are converted to MW with the stored capacity and divided by the
rule's capacity, and the quantiles are clipped to 0 to 1. No model is refitted, so the model never
saw the rule's target. Wind: experiment 429's `member_plain` (5 seeds averaged in quantile space,
experiment 210's idle-band mask applied). Solar: experiment 430's `a2_member` (3 seeds). Rows are
forecasts at true days 1 to 10 (lead 24 to 240 hours). Output: `work/capacity_rows.parquet` (one
row per series, month, and rule).

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00451-*/env.sh
uv run --no-sync python ml-experiments/00453-*/e00453_capacity.py
```
"""

from typing import Final

import e00230_common as c230  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00450_score as score450  # ty: ignore[unresolved-import]
import e00450_step1 as step1  # ty: ignore[unresolved-import]
import e00453_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

RULES: Final[dict[str, float]] = {"max": 1.0, "p99.99": 0.9999, "p99.9": 0.999, "p99": 0.99}
LEVEL_NAMES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
SOLAR_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00430-unified-forecaster-solar-and-pretraining-ablations"
    / "predictions"
    / "a2_member"
)
LEAD_MIN_H: Final[float] = 24.0
LEAD_MAX_H: Final[float] = 240.0
WIND_LABEL: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}


def wind_rows() -> pl.DataFrame:
    """Return experiment 429's `member_plain` scored rows: site, month, actual_mw, q*_mw columns."""
    cases = data347.scored_frame()
    caps = c230.site_table().select("site", cap=pl.col("effective_capacity_mw"))
    tables = score450.load_seed_tables(name="base", seeds=score450.run450.SEEDS)
    q = np.sort(np.mean(np.stack(tables), axis=0), axis=1)
    frame = cases.select("site", "month", "masked", "lead_hours", "actual_norm").with_columns(
        *[pl.Series(c, q[:, i]) for i, c in enumerate(LEVEL_NAMES)]
    )
    frame = frame.filter(
        ~pl.col("masked") & pl.col("lead_hours").is_between(LEAD_MIN_H, LEAD_MAX_H)
        & pl.col("actual_norm").is_finite()
    ).join(caps, on="site")  # fmt: skip
    return frame.select(
        *LEVEL_NAMES,
        site=pl.col("site").replace_strict(WIND_LABEL),
        month=pl.col("month"),
        lead_hours=pl.col("lead_hours"),
        cap=pl.col("cap").cast(pl.Float64),
        actual=pl.col("actual_norm"),
    )


def solar_rows() -> pl.DataFrame:
    """Return experiment 430's `a2_member` rows: site, month, stored-capacity fractions."""
    sites = data.labels().filter(pl.col("tech") == "solar")
    cap = {r["site"].removeprefix("Solar "): r["cap"] for r in sites.iter_rows(named=True)}
    frame = pl.concat([pl.read_parquet(f) for f in sorted(SOLAR_DIR.glob("s*_*.parquet"))])
    keys = ["time", "init_time", "time_series_id", "lead_hours", "actual"]
    mean = frame.group_by(keys).agg(*[pl.col(c).mean() for c in LEVEL_NAMES]).sort(keys)
    q = np.sort(mean.select(list(LEVEL_NAMES)).to_numpy(), axis=1)
    mean = mean.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(LEVEL_NAMES)])
    mean = mean.filter(
        pl.col("lead_hours").is_between(LEAD_MIN_H, LEAD_MAX_H) & pl.col("actual").is_finite()
    )
    return mean.select(
        *LEVEL_NAMES,
        site="Solar " + pl.col("time_series_id"),
        month=pl.col("time").dt.strftime("%Y-%m"),
        lead_hours=pl.col("lead_hours"),
        cap=pl.col("time_series_id").replace_strict(cap, return_dtype=pl.Float64),
        actual=pl.col("actual"),
    )


def earlier_capacity(*, series: pl.DataFrame, month: str, site: str) -> dict[str, float]:
    """Return each rule's capacity (MW) from the site's power before the first day of `month`."""
    start = pl.datetime(int(month[:4]), int(month[5:]), 1, time_zone="UTC")
    power = series.filter((pl.col("site") == site) & (pl.col("time") < start))["power"].to_numpy()
    out = {"max": float(power.max())}
    out.update({name: float(np.quantile(power, q)) for name, q in RULES.items() if name != "max"})
    return out


def rows_in_rule_units(
    *, rows: pl.DataFrame, series: pl.DataFrame, month: str, rule: str
) -> pl.DataFrame:
    """Return the rows of `month` and earlier, in units of the rule's capacity at `month`.

    Quantiles are clipped to 0 to 1; actuals are not. `lead_band` is added for table P.
    """
    caps = {}
    for site, stored in rows.group_by("site").agg(pl.col("cap").first()).iter_rows():
        caps[site] = (
            stored
            if rule == "current"
            else earlier_capacity(series=series, month=month, site=site)[rule]
        )
    scale = pl.col("site").replace_strict({s: caps[s] for s in caps}, return_dtype=pl.Float64)
    return rows.filter(pl.col("month") <= month).with_columns(
        (pl.col("cap") / scale).alias("ratio")
    ).with_columns(
        *[(pl.col(c) * pl.col("ratio")).clip(0.0, 1.0).alias(c) for c in LEVEL_NAMES],
        actual=pl.col("actual") * pl.col("ratio"),
        new_cap=scale,
        lead_band=step1.lead_band_expr(),
    ).drop("ratio")  # fmt: skip


def cell_metrics(*, frame: pl.DataFrame, rule: str, variant: str) -> list[dict]:
    """Return the share and score columns of `frame`, one dict per (site, month)."""
    out = []
    for (site, month), part in frame.group_by(["site", "month"], maintain_order=True):
        q = part.select(list(LEVEL_NAMES)).to_numpy()
        actual = part["actual"].to_numpy()
        cap = float(part["new_cap"][0])
        score = c230.quantile_score(quantiles=q, actual=np.clip(actual, 0.0, None))
        out.append({
            "site": site, "month": month, "rule": rule, "variant": variant, "cap_mw": cap,
            "n": len(actual), "n_gt_1": int((actual > 1.0).sum()),
            "n_gt_q99": int((actual > q[:, -1]).sum()),
            "max_actual": float(actual.max()),
            "score_pts_sum": float(score.sum()) * 100.0, "score_mw_sum": float(score.sum()) * cap,
        })  # fmt: skip
    return out


def run() -> None:
    """Write `work/capacity_rows.parquet`: one row per series, test month, rule, and variant."""
    series = pl.read_parquet(data.SERIES_PATH).select("site", "time", "power")
    out: list[dict] = []
    for rows in (wind_rows(), solar_rows()):
        months = sorted(rows["month"].unique())
        for month in months[1:]:
            for rule in ("current", *RULES):
                frame = rows_in_rule_units(rows=rows, series=series, month=month, rule=rule)
                held = frame.filter(pl.col("month") == month)
                out += cell_metrics(frame=held, rule=rule, variant="raw")
                factors = step1.fit_factors(fit=frame.filter(pl.col("month") < month))
                recal = step1.apply_factors(frame=held, factors=factors)
                out += cell_metrics(frame=recal, rule=rule, variant="P")
    data.WORK_DIR.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(out)
    table.write_parquet(data.WORK_DIR / "capacity_rows.parquet")
    summary = (
        table.group_by("site", "rule", "variant")
        .agg(
            pl.col("n").sum(), pl.col("n_gt_1").sum(), pl.col("n_gt_q99").sum(),
            pl.col("score_mw_sum").sum(), pl.col("cap_mw").mean(),
        )
        .with_columns(
            share_gt_1=pl.col("n_gt_1") / pl.col("n") * 100,
            share_gt_q99=pl.col("n_gt_q99") / pl.col("n") * 100,
            score_mw=pl.col("score_mw_sum") / pl.col("n"),
        )
        .sort("site", "variant", "cap_mw")
        .drop("n_gt_1", "n_gt_q99", "score_mw_sum")
    )  # fmt: skip
    with pl.Config(tbl_rows=120, tbl_cols=20, tbl_width_chars=200):
        print(summary)


if __name__ == "__main__":
    run()

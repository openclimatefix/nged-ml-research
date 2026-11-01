"""Experiment 451: where Farm A's upper-tail exceedances sit, and finer or looser table P variants.

Step 1 compares, for each farm, the share of observations above q99 (kept rows, true days 1 to 10,
11 scored months) under five versions of `member_plain`'s forecast: none; experiment 450's table P
(farm x lead band); P_med (also by band of the forecast median); P_wind (also by tercile of the
ensemble-mean 100 m wind speed, with edges from all scored months since they only label rows); and
table P with the upper clip raised to the farm's largest earlier power over capacity. Every
rescaled quantile is clipped, and every cell and bound is fitted on earlier months only. Step 2
splits Farm A's exceedances of the table P q99 by month, hour, wind-speed bin, forecast-median
band, lead band, and experiment 341's capacity state, and counts those where q99 is at the clip
(0.999 or more). Output: `work/step1.parquet`, `work/step2_<split>.parquet`.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00451-*/env.sh
uv run --no-sync python ml-experiments/00451-*/e00451_exceedances.py
```
"""

from itertools import pairwise
from pathlib import Path
from typing import Final

import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00429_data as data429  # ty: ignore[unresolved-import]
import e00450_score as score450  # ty: ignore[unresolved-import]
import e00450_step1 as step1  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00451-farm-a-upper-tail-exceedances-and-capacity-state"
)
WORK_DIR: Final[Path] = DATA_DIR / "work"
CAPACITY_STATE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00341-wind-capacity-steps-from-strong-wind"
    / "work"
    / "capacity_state.parquet"
)
MEDIAN_BANDS: Final[tuple[float, float, float]] = (0.1, 0.4, 0.7)
"""Edges of the forecast-median bands as shares of capacity, as in experiment 435."""
AT_CLIP: Final[float] = 0.999
"""A q99 at or above this value counts as clipped at 1.0."""
WIND_BINS: Final[tuple[float, ...]] = (4.0, 8.0, 12.0, 16.0)
"""Edges of the ensemble-mean 100 m wind speed bins in m/s."""


def wind_bin_expr() -> pl.Expr:
    """Return the label of the ensemble-mean wind-speed bin, such as `(4, 8]`."""
    edges = [None, *WIND_BINS, None]
    expr = pl.lit(None, dtype=pl.String)
    for low, high in pairwise(edges):
        label = (
            f"({'-inf' if low is None else f'{low:g}'}, {'inf' if high is None else f'{high:g}'}]"
        )
        inside = pl.lit(value=True)
        if low is not None:
            inside = inside & (pl.col("wind") > low)
        if high is not None:
            inside = inside & (pl.col("wind") <= high)
        expr = pl.when(inside).then(pl.lit(label)).otherwise(expr)
    return expr


def frame_with_extras() -> pl.DataFrame:
    """Return `member_plain`'s scored rows with quantiles, bands, wind speed, and capacity state."""
    cases = data347.scored_frame()
    weather, _ = data429.load(set_name="scored")
    wind = weather[:, :, 1].mean(axis=1)  # ensemble-mean 100 m wind speed
    assert len(wind) == cases.height
    tables = score450.load_seed_tables(name="base", seeds=score450.run450.SEEDS)
    q = np.sort(np.mean(np.stack(tables), axis=0), axis=1)
    frame = cases.select(
        "site", "month", "masked", "lead_hours", "valid_time", actual=pl.col("actual_norm")
    ).with_columns(
        *[pl.Series(c, q[:, i]) for i, c in enumerate(step1.LEVELS)],
        wind=pl.Series(wind),
    )
    med = pl.col("q50")
    frame = frame.with_columns(
        median_band=pl.when(med < MEDIAN_BANDS[0]).then(pl.lit("m0"))
        .when(med < MEDIAN_BANDS[1]).then(pl.lit("m1"))
        .when(med < MEDIAN_BANDS[2]).then(pl.lit("m2"))
        .otherwise(pl.lit("m3")),
        wind_bin=wind_bin_expr(),
        hour=pl.col("valid_time").dt.hour(),
    )  # fmt: skip
    state = pl.read_parquet(CAPACITY_STATE).select("site", "valid_time", "state", "reduced_share")
    frame = frame.join(state, on=["site", "valid_time"], how="left")
    return (
        frame.filter(pl.col("actual").is_finite())
        .with_columns(lead_band=step1.lead_band_expr())
        .filter(pl.col("lead_band").is_not_null() & ~pl.col("masked"))
    )


def with_cell(*, frame: pl.DataFrame, cell: tuple[str, ...]) -> pl.DataFrame:
    """Return the frame with `lead_band` renamed to a combined cell key; `site` stays the farm."""
    key = pl.concat_str([pl.col(c).cast(pl.String) for c in cell], separator="|")
    return frame.with_columns(lead_band=key)


def recalibrated(*, frame: pl.DataFrame, extra: tuple[str, ...]) -> pl.DataFrame:
    """Return the scored rows after table P with cells (lead band, *extra), fitted earlier."""
    cell = ("lead_band", *extra)
    work = with_cell(frame=frame, cell=cell)
    months = sorted(work["month"].unique())
    parts = []
    for i, month in enumerate(months[1:], start=1):
        earlier = work.filter(pl.col("month").is_in(months[:i]) & (pl.col("lead_hours") >= 24))
        held = work.filter(pl.col("month") == month)
        factors = step1.fit_factors(fit=earlier)
        parts.append(step1.apply_factors(frame=held, factors=factors))
    return pl.concat(parts)


def recalibrated_with_bound(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return table P (farm x lead band) with the upper clip raised to the farm's earlier maximum.

    The upper bound of every rescaled quantile is the largest observed power over capacity at that
    farm in the months before the scored month (at least 1.0), so an output a few per cent above the
    effective capacity can fall below q99. Fitted on earlier months only.
    """
    months = sorted(frame["month"].unique())
    parts = []
    for i, month in enumerate(months[1:], start=1):
        earlier = frame.filter(pl.col("month").is_in(months[:i]))
        bound = earlier.group_by("site").agg(
            pl.max_horizontal(pl.col("actual").max(), pl.lit(1.0)).alias("upper")
        )
        factors = step1.fit_factors(fit=earlier.filter(pl.col("lead_hours") >= 24))
        held = frame.filter(pl.col("month") == month)
        # apply_factors clips to [0, 1]; redo the scaling unclipped and clip to the farm's bound.
        out = held
        for name in step1.TAILS:
            lookup = pl.DataFrame(
                [(s, b, f) for (s, b, n), f in factors.items() if n == name],
                schema={"site": pl.String, "lead_band": pl.String, "factor": pl.Float64},
                orient="row",
            )
            out = out.join(lookup, on=["site", "lead_band"], how="left").join(bound, on="site")
            out = out.with_columns(
                (pl.col("q50") + pl.col("factor").fill_null(1.0) * (pl.col(name) - pl.col("q50")))
                .clip(0.0, pl.col("upper"))
                .alias(name)
            ).drop("factor", "upper")
        parts.append(out)
    return pl.concat(parts)


def share_above(*, frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return rows and the share of observations above q99 at true days 1 to 10, per `by`."""
    return (
        frame.filter(pl.col("lead_hours") >= 24)
        .group_by(by)
        .agg(
            pl.len().alias("n"),
            (pl.col("actual") > pl.col("q99")).mean().mul(100).alias("above_q99_pct"),
        )
        .sort(by)
    )


def main() -> None:
    """Run both steps and print the tables."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    frame = frame_with_extras()
    tercile = frame.group_by("site").agg(
        pl.col("wind").quantile(1 / 3).alias("t1"), pl.col("wind").quantile(2 / 3).alias("t2")
    )
    frame = frame.join(tercile, on="site").with_columns(
        wind_tercile=pl.when(pl.col("wind") < pl.col("t1")).then(pl.lit("w1"))
        .when(pl.col("wind") < pl.col("t2")).then(pl.lit("w2")).otherwise(pl.lit("w3"))
    ).drop("t1", "t2")  # fmt: skip
    variants: dict[str, tuple[str, ...] | str | None] = {
        "none": None,
        "P (farm x lead band)": (),
        "P_med (+ forecast-median band)": ("median_band",),
        "P_wind (+ wind-speed tercile)": ("wind_tercile",),
        "P with the upper clip raised to the farm's earlier maximum": "bound",
    }
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    rows = []
    adjusted: dict[str, pl.DataFrame] = {}
    for name, extra in variants.items():
        if extra is None:
            f = frame.filter(pl.col("month") != min(frame["month"].unique()))
        elif isinstance(extra, str):
            f = recalibrated_with_bound(frame=frame)
        else:
            f = recalibrated(frame=frame, extra=extra)
        adjusted[name] = f
        by_farm = share_above(frame=f, by=["site"]).with_columns(variant=pl.lit(name))
        pooled = share_above(frame=f.with_columns(site=pl.lit("all")), by=["site"]).with_columns(
            variant=pl.lit(name)
        )
        rows += [by_farm, pooled]
    step1_table = pl.concat(rows)
    step1_table.write_parquet(WORK_DIR / "step1.parquet")
    print(step1_table.pivot(on="site", index="variant", values="above_q99_pct"))
    # Step 2: Farm A exceedances of table P's q99.
    a = adjusted["P (farm x lead band)"].filter(
        (pl.col("site") == "W1") & (pl.col("lead_hours") >= 24)
    )
    a = a.with_columns(
        over=pl.col("actual") > pl.col("q99"),
        at_cap=pl.col("actual") >= 0.95,
        excess=pl.when(pl.col("actual") > pl.col("q99")).then(pl.col("actual") - pl.col("q99")),
    )
    print(
        "Farm A exceedances:",
        int(a["over"].sum()),
        "of",
        a.height,
        "; at/above 0.95 of capacity:",
        int(a.filter(pl.col("over"))["at_cap"].sum()),
    )
    at_clip = pl.col("q99") >= AT_CLIP
    clip_split = a.select(
        (at_clip.mean() * 100).alias("pct_rows_q99_at_clip"),
        ((pl.col("over") & at_clip).mean() * 100).alias("points_exceed_q99_at_clip"),
        ((pl.col("over") & ~at_clip).mean() * 100).alias("points_exceed_q99_below_clip"),
        (pl.col("actual") > 1.0).mean().mul(100).alias("pct_rows_actual_above_1"),
    )
    clip_split.write_parquet(WORK_DIR / "step2_clip_split.parquet")
    print(clip_split)
    for split in ("month", "hour", "wind_bin", "state", "median_band", "lead_band"):
        t = (
            a.group_by(split)
            .agg(
                pl.len().alias("n"),
                pl.col("over").sum().alias("exceed"),
                (pl.col("over").mean() * 100).alias("pct"),
                pl.col("excess").mean().alias("mean_excess"),
            )
            .sort(split)
        )
        t.write_parquet(WORK_DIR / f"step2_{split}.parquet")
        print(t)
    own = a.filter(pl.col("over"))
    print("reduced_share among exceedances:", own["reduced_share"].describe())


if __name__ == "__main__":
    main()

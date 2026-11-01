"""Experiment 483: tail calibration of A-pool, A-gam-in, and K2-425A, by widening and table P.

Reads experiment 470's stored forecasts and row table, rejoins the actuals and the
normal-running-arrangement (NRA) correction, and writes under
`data/ml-experiments/00483-*/work/`:

- `cache`: one parquet per arm and month with the keys, row flags, actual, NRA, and the 11
  quantiles (cells averaged in quantile space);
- `free`: tails by month, the widening sweep 1.0 to 1.15, and the worst 20 half-hours by q99
  pinball loss;
- `tablep`: table P fitted leave-one-fold-out, with a cold-night class added (the five folds are
  not contiguous, so each fold is fitted on the other four).

Rows are the `gam14` set of experiment 470, plus D15 and D18 (`main16` rows) for every arm except
A-gam-in, which has no forecast for them. D15 and D18 are the `report_only` group. Units are
points of effective capacity (fractions times 100). July 2026 is not used.

```bash
source /tmp/e483env.sh   # DATA_PATH_INTERNAL and a PYTHONPATH of every ml-experiments folder
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00483-*/e00483_tails.py cache
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import widen  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00483-uncertainty-rerun-on-the-first-release-demand-candidates"
    / "work"
)
ARMS: Final[tuple[str, ...]] = ("A-pool", "A-gam-in", "K2-425A")
MONTHS: Final[tuple[str, ...]] = score470.MONTHS
LEVELS: Final[tuple[str, ...]] = build470.LEVELS
KEYS: Final[list[str]] = build470.KEYS
SWEEP: Final[tuple[float, ...]] = (1.0, 1.05, 1.10, 1.15)
FLAGS: Final[tuple[str, ...]] = (
    "group", "cls", "lead_hours", "lead_band", "tgt_real", "cold", "xmas", "band", "day",
)  # fmt: skip


def cache_path(*, arm: str, month: str) -> Path:
    """Return where an arm-month frame is cached."""
    return WORK / "cache" / f"{arm}_{month}.parquet"


def cache() -> None:
    """Write one frame per arm and month."""
    (WORK / "cache").mkdir(parents=True, exist_ok=True)
    ctx = prep425.cleaned_scoring_context(series=adapter.series_table())
    sites = build470.site_map()
    rows = score470.load_rows()
    for month in MONTHS:
        full = score470.month_context(month=month, sites=sites, ctx=ctx)
        month_rows = rows.filter(pl.col("month") == month)
        for arm in ARMS:
            keep_expr = pl.col("gam14") | (
                (pl.col("main16") & (pl.col("group") == "report_only")) & pl.lit(arm != "A-gam-in")
            )
            keep = month_rows.filter(keep_expr).select(KEYS)
            base = full.join(keep, on=KEYS, how="semi")
            frames = [
                pl.read_parquet(build470.prediction_path(arm=arm, month=month, cell=c))
                for c in range(score470.arm_cells(arm=arm))
            ]
            assert all(f.select(KEYS).equals(frames[0].select(KEYS)) for f in frames), (
                "cells differ"
            )
            q = np.mean([f.select(list(LEVELS)).to_numpy().astype(np.float64) for f in frames], 0)
            mean = frames[0].select(KEYS).hstack(pl.DataFrame(q, schema=list(LEVELS), orient="row"))
            out = base.join(mean, on=KEYS, how="left", maintain_order="left").filter(
                pl.all_horizontal(pl.col(*LEVELS, "actual", "nra").is_finite())
            )
            assert out.select(KEYS).is_duplicated().sum() == 0, "duplicate keys"
            dropped = base.height - out.height
            if dropped:
                print("  dropped non-finite rows:", dropped)
            out = out.select(
                *KEYS, "time_series_id", "valid_time", "actual", "nra", *FLAGS[:-1], "day", *LEVELS
            ).with_columns(month=pl.lit(month))
            out.write_parquet(cache_path(arm=arm, month=month))
            print(arm, month, base.height, out.height, flush=True)


def load(*, arm: str) -> pl.DataFrame:
    """Return all months of an arm."""
    return pl.concat([pl.read_parquet(cache_path(arm=arm, month=m)) for m in MONTHS])


def summarise(*, frame: pl.DataFrame, by: list[str], actual: str = "actual") -> pl.DataFrame:
    """Return the row count, CRPS, coverage, and pinball loss (points) at the tails, by `by`."""
    m = score470.row_metrics(frame=frame, actual=actual)
    cols = [pl.Series(k, v) for k, v in m.items()]
    return (
        frame.select(*by)
        .with_columns(cols)
        .group_by(by)
        .agg(pl.len().alias("n"), pl.all().mean())
        .sort(by)
    )


def slices(*, frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the named row slices of a frame (group and flags decide each)."""
    clean = frame.filter(pl.col("cls") == "clean")
    return {
        "clean": clean,
        "clean_12_36h": clean.filter(pl.col("band")),
        "clean_cold": clean.filter(pl.col("cold")),
        "clean_xmas": clean.filter(pl.col("xmas")),
        "real_event_nra": frame.filter(pl.col("tgt_real")).with_columns(
            actual=pl.col("actual") - pl.col("nra")
        ),
        "real_event_metered": frame.filter(pl.col("tgt_real")),
    }


def free() -> None:
    """Write tails by month, the widening sweep, and the worst 20 half-hours."""
    parts: list[pl.DataFrame] = []
    worst: list[pl.DataFrame] = []
    for arm in ARMS:
        frame = load(arm=arm)
        for factor in SWEEP:
            wide = frame if factor == 1.0 else widen(frame=frame, factor=factor)
            for name, part in slices(frame=wide).items():
                for by in (["group"], ["group", "month"], ["group", "lead_band"]):
                    summary = summarise(frame=part, by=by)
                    parts += [
                        summary.with_columns(
                            arm=pl.lit(arm),
                            widening=pl.lit(factor),
                            slice=pl.lit(name),
                            by=pl.lit("+".join(by)),
                        )
                    ]
        for factor in (1.0, 1.15):
            wide = frame if factor == 1.0 else widen(frame=frame, factor=factor)
            wide = wide.filter((pl.col("group") != "report_only") & (pl.col("cls") == "clean"))
            m = score470.row_metrics(frame=wide)
            top = (
                wide.with_columns(pin_q99=pl.Series(m["pin_q99"]))
                .sort("pin_q99", descending=True)
                .head(20)
                .with_columns(arm=pl.lit(arm), widening=pl.lit(factor))
            )
            worst.append(top)
        print("free", arm, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    pl.concat(parts, how="diagonal").write_parquet(WORK / "free_tails.parquet")
    pl.concat(worst).write_parquet(WORK / "worst20.parquet")
    # Share of actuals beyond q99 and q01, by month, at the headline groups.
    beyond = []
    for arm in ARMS:
        for factor in SWEEP:
            f = load(arm=arm)
            f = f if factor == 1.0 else widen(frame=f, factor=factor)
            f = f.filter(pl.col("cls") == "clean")
            beyond.append(
                f.group_by("group", "month")
                .agg(
                    pl.len().alias("n"),
                    (pl.col("actual") > pl.col("q99")).mean().alias("above_q99"),
                    (pl.col("actual") < pl.col("q01")).mean().alias("below_q01"),
                )
                .with_columns(arm=pl.lit(arm), widening=pl.lit(factor))
            )
    pl.concat(beyond).write_parquet(WORK / "beyond.parquet")


def fit_with_fallback(*, fit: pl.DataFrame, cold_class: bool) -> pl.DataFrame:
    """Fit table P; a `_cold` cell with too few rows takes the plain group's factor."""
    plain = prov.fit_factors(fit=fit.with_columns(group=pl.col("group_orig")))
    if not cold_class:
        return plain
    cold = prov.fit_factors(fit=fit)
    have = cold.select("group", "lead_band", "tail")
    fallback = (
        plain.with_columns(group=pl.col("group") + "_cold")
        .join(have, on=["group", "lead_band", "tail"], how="anti")
        .with_columns(n_fit=pl.lit(0, dtype=pl.Int64))
    )
    return pl.concat([cold, fallback])


def tablep() -> None:
    """Fit table P leave-one-fold-out (group by lead band, plus a cold-night class) and score it.

    A factor missing for a group, lead band, and tail is 1.0 (no change) in `apply_factors`; the
    cold class falls back to the plain group's factor, and `n_fit == 0` in the saved factors marks
    each fallback.
    """
    results: list[pl.DataFrame] = []
    factors_out: list[pl.DataFrame] = []
    for arm in ARMS:
        frame = load(arm=arm)
        for variant, cold_class in (("P", False), ("P+cold", True)):
            tag = frame.with_columns(
                group_orig=pl.col("group"),
                group=pl.when(pl.col("cold") & pl.lit(cold_class))
                .then(pl.col("group") + "_cold")
                .otherwise(pl.col("group")),
            )
            held_all: list[pl.DataFrame] = []
            for month in MONTHS:
                fit = tag.filter((pl.col("month") != month) & (pl.col("cls") == "clean"))
                factors = fit_with_fallback(fit=fit, cold_class=cold_class)
                factors_out.append(
                    factors.with_columns(
                        arm=pl.lit(arm), month=pl.lit(month), variant=pl.lit(variant)
                    )
                )
                held = tag.filter(pl.col("month") == month)
                held_all.append(
                    prov.apply_factors(frame=held, factors=factors).with_columns(
                        group=pl.col("group_orig")
                    )
                )
            adj = pl.concat(held_all)
            for name, part in slices(frame=adj).items():
                for by in (["group"], ["group", "month"], ["group", "lead_band"]):
                    summary = summarise(frame=part, by=by)
                    results += [
                        summary.with_columns(
                            arm=pl.lit(arm),
                            variant=pl.lit(variant),
                            slice=pl.lit(name),
                            by=pl.lit("+".join(by)),
                        )
                    ]
            print("tablep", arm, variant, flush=True)
    pl.concat(results, how="diagonal").write_parquet(WORK / "tablep_tails.parquet")
    pl.concat(factors_out).write_parquet(WORK / "tablep_factors.parquet")


def strict() -> None:
    """Fit table P on earlier folds only (2025-10 is never scored) and score the other four."""
    results: list[pl.DataFrame] = []
    for arm in ARMS:
        frame = load(arm=arm).with_columns(group_orig=pl.col("group"))
        held_all = []
        for i, month in enumerate(MONTHS[1:], start=1):
            fit = frame.filter(pl.col("month").is_in(MONTHS[:i]) & (pl.col("cls") == "clean"))
            factors = fit_with_fallback(fit=fit, cold_class=False)
            held = frame.filter(pl.col("month") == month)
            held_all.append(prov.apply_factors(frame=held, factors=factors))
        adj = pl.concat(held_all)
        for name, part in slices(frame=adj).items():
            results.extend(
                summarise(frame=part, by=by).with_columns(
                    arm=pl.lit(arm), variant=pl.lit("P-earlier"), slice=pl.lit(name),
                    by=pl.lit("+".join(by)),
                )
                for by in (["group"], ["group", "month"])
            )  # fmt: skip
        # The same four folds, raw and widened, so the comparison is on identical rows.
        rest = frame.filter(pl.col("month").is_in(MONTHS[1:]))
        for label, variant_frame in (("raw", rest), ("w1.15", widen(frame=rest, factor=1.15))):
            for name, part in slices(frame=variant_frame).items():
                results.append(
                    summarise(frame=part, by=["group"]).with_columns(
                        arm=pl.lit(arm), variant=pl.lit(f"{label}-4folds"), slice=pl.lit(name),
                        by=pl.lit("group"),
                    )
                )  # fmt: skip
        print("strict", arm, flush=True)
    pl.concat(results, how="diagonal").write_parquet(WORK / "strict_tails.parquet")


def upper() -> None:
    """Widen only q95 and q99 (distance from the median times 1.10 or 1.15) and score clean rows."""
    parts: list[pl.DataFrame] = []
    for arm in ARMS:
        frame = load(arm=arm).filter(pl.col("cls") == "clean")
        for factor in (1.10, 1.15):
            wide = frame.with_columns(
                [
                    (pl.col("q50") + factor * (pl.col(q) - pl.col("q50"))).alias(q)
                    for q in ("q95", "q99")
                ]
            )
            parts.append(
                summarise(frame=wide, by=["group"]).with_columns(
                    arm=pl.lit(arm), variant=pl.lit(f"upper{factor:.2f}")
                )
            )
    pl.concat(parts).write_parquet(WORK / "upper_tails.parquet")


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cache", "free", "tablep", "strict", "upper"))
    args = parser.parse_args()
    {"cache": cache, "free": free, "tablep": tablep, "strict": strict, "upper": upper}[
        args.command
    ]()


if __name__ == "__main__":
    main()

"""Experiment 435, step 3: table P on stored experiment 425 (demand), 429 (wind), and 430 (solar).

Each scored month is fitted on the earlier months only (the first month is never scored). For
demand (`425`) the fitting cell is (metering group, lead band), and a per-Primary variant `P_site`
fits (Primary, lead band) as well, to test the D14 and D4 shortfall. For wind (`429`) the cell is
(farm, lead band) and an extra per-band-of-forecast-median variant `P_med` stands in for the
per-wind-speed band, because the scored frame carries no wind speed. For solar (`430`) the cell is
(solar site, lead band). Every cell is fitted as in `e00435_table_p_provisional.fit_factors`
(minimum 200 rows). Output: `work/table_p_<source>.parquet` with coverage and pinball loss at the
four tail quantiles raw, widened by 1.15, and after P, by month.

```bash
# PYTHONPATH as for e00435_tails.py (425) or e00435_tails_wind_solar.py (429 and 430)
uv run --no-sync python ml-experiments/00435-*/e00435_table_p_other.py {425,429,430}
```
"""

import sys
from pathlib import Path
from typing import Final

import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import polars as pl
from e00435_tailcore import WIDENING, tail_table, widen
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = prov.WORK_DIR


def with_cell(*, frame: pl.DataFrame, cell: str) -> pl.DataFrame:
    """Return the frame with `group` set to the fitting-cell column `cell`."""
    return frame.with_columns(group=pl.col(cell).cast(pl.String))


def run(
    *,
    frame: pl.DataFrame,
    source: str,
    arm: str,
    cells: dict[str, str],
    bounds: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Score each month after the first with P fitted on the earlier months, per cell choice."""
    months = sorted(frame["month"].unique())
    out: list[pl.DataFrame] = []
    for i, month in enumerate(months[1:], start=1):
        earlier = frame.filter(pl.col("month").is_in(months[:i]))
        held = frame.filter(pl.col("month") == month)
        variants = {"raw": held, "w1.15": widen(frame=held, factor=WIDENING, bounds=bounds)}
        for name, cell in cells.items():
            factors = prov.fit_factors(fit=with_cell(frame=earlier, cell=cell))
            variants[name] = prov.apply_factors(
                frame=with_cell(frame=held, cell=cell), factors=factors, bounds=bounds
            )
        for variant, variant_frame in variants.items():
            reported = variant_frame.with_columns(group=pl.col("report_group"))
            for by in (("group", "lead_band"), ("site_report", "lead_band")):
                table = tail_table(frame=reported, by=by).rename({by[0]: "key"})
                out.append(
                    table.with_columns(
                        source=pl.lit(source),
                        arm=pl.lit(arm),
                        month=pl.lit(month),
                        variant=pl.lit(variant),
                        by=pl.lit(by[0]),
                    )
                )
    return pl.concat(out)


def demand_425() -> pl.DataFrame:
    """Score arms R and A of experiment 425."""
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    import e00425_prep as prep  # ty: ignore[unresolved-import]
    import e00425_score as score425  # ty: ignore[unresolved-import]
    import e00435_tails as tails  # ty: ignore[unresolved-import]

    months = tuple(
        f"{y}-{m:02d}"
        for y, m in [(2025, m) for m in range(7, 13)] + [(2026, m) for m in range(1, 7)]
    )
    series = adapter.series_table()
    actuals = prep.cleaned_scoring_context(series=series)["actuals"].filter(
        pl.col("actual").is_finite()
    )
    frames = score425.load_frames(series=series, months=months, with_references=False)
    out: list[pl.DataFrame] = []
    for arm in tails.ARMS:
        scored = tails.with_group_and_band(
            frame=frames[f"{arm}_raw"].join(actuals, on=["time_series_id", "valid_time"]),
            series=series,
        ).filter(pl.col("group").is_not_null() & pl.col("lead_band").is_not_null())
        scored = scored.with_columns(
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            report_group=pl.col("group"),
            site=pl.col("label"),
            site_report=pl.col("label"),
        ).with_columns(
            *[pl.col(c).cast(pl.Float64) for c in ("actual", "q50", "q01", "q05", "q95", "q99")]
        )
        out.append(run(frame=scored, source="425", arm=arm, cells={"P": "group", "P_site": "site"}))
    return pl.concat(out)


def wind_429() -> pl.DataFrame:
    """Score every arm of experiment 429."""
    import e00347_s4_data as data347  # ty: ignore[unresolved-import]
    import e00429_run as run429  # ty: ignore[unresolved-import]
    import e00435_tails_wind_solar as ws  # ty: ignore[unresolved-import]
    import numpy as np

    cases = data347.scored_frame()
    base = cases.select("site", "month", "masked", "lead_hours", actual=pl.col("actual_norm"))
    out: list[pl.DataFrame] = []
    for name in run429.ARMS:
        present = [
            np.load(p)
            for s in run429.SEEDS
            if (p := run429.PARTS_DIR / f"{name}_h{s}.npy").exists()
        ]
        if not present:
            continue
        q = np.sort(np.mean(np.stack(present), axis=0), axis=1)
        frame = base.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(ws.LEVELS)])
        frame = (
            frame.filter(pl.col("actual").is_finite())
            .with_columns(lead_band=ws.lead_band_expr(), median_band=ws.median_band_expr())
            .filter(pl.col("lead_band").is_not_null())
        )
        frame = frame.with_columns(
            report_group=pl.lit("all farms"),
            site_report=pl.col("site"),
            cell_med=pl.col("site") + pl.col("median_band"),
        )
        out.append(
            run(
                frame=frame,
                source="429",
                arm=name,
                cells={"P": "site", "P_med": "cell_med"},
                bounds=(0.0, 1.0),
            )
        )
    return pl.concat(out)


def solar_430() -> pl.DataFrame:
    """Score every arm of experiment 430."""
    import e00435_tails_wind_solar as ws  # ty: ignore[unresolved-import]
    import numpy as np

    out: list[pl.DataFrame] = []
    for arm_dir in sorted(p for p in ws.SOLAR_DIR.iterdir() if p.is_dir()):
        files = sorted(arm_dir.glob("s*_*.parquet"))
        if not files:
            continue
        frame = pl.concat([pl.read_parquet(f) for f in files])
        keys = ["time", "init_time", "time_series_id", "lead_hours", "actual"]
        mean = frame.group_by(keys).agg(*[pl.col(c).mean() for c in ws.LEVELS]).sort(keys)
        q = np.sort(mean.select(list(ws.LEVELS)).to_numpy(), axis=1)
        mean = mean.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(ws.LEVELS)])
        mean = (
            mean.filter(pl.col("actual").is_finite())
            .with_columns(
                lead_band=ws.lead_band_expr(),
                month=pl.col("init_time").dt.strftime("%Y-%m"),
                report_group=pl.lit("all sites"),
                site_report=pl.col("time_series_id"),
            )
            .filter(pl.col("lead_band").is_not_null())
        )
        out.append(
            run(
                frame=mean,
                source="430",
                arm=arm_dir.name,
                cells={"P": "time_series_id"},
                bounds=(0.0, 1.0),
            )
        )
    return pl.concat(out)


def stale(*, source: str) -> pl.DataFrame:
    """Score every arm of experiment 422's network `T<k>` (as `422:T0`) or of experiment 410.

    The stored quantiles are fractions of capacity; the adapter converts them to MW.
    """
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    import e00425_prep as prep  # ty: ignore[unresolved-import]
    import e00435_tails as tails  # ty: ignore[unresolved-import]
    import numpy as np

    root = ML_EXPERIMENTS_DATA_DIR
    if source.startswith("422:"):
        folder = (
            root
            / "00422-mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2"
            / "work"
            / "preds"
            / source[4:]
        )
    else:
        folder = (
            root / "00410-mdn-stale-weather-and-delayed-power-rescore" / "work" / "preds" / "pooled"
        )
    series = adapter.series_table()
    actuals = prep.cleaned_scoring_context(series=series)["actuals"].filter(
        pl.col("actual").is_finite()
    )
    levels = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
    out: list[pl.DataFrame] = []
    for arm_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        parts: list[pl.DataFrame] = []
        for month in ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06"):
            seeds = [
                pl.read_parquet(f).sort("time_series_id", "power_fcst_init_time", "valid_time")
                for f in sorted(arm_dir.glob(f"{month}_s*.parquet"))
            ]
            if not seeds:
                continue
            mean = np.sort(np.mean([s.select(levels).to_numpy() for s in seeds], axis=0), axis=1)
            stored = (
                seeds[0]
                .select("time_series_id", "power_fcst_init_time", "valid_time")
                .with_columns(*[pl.Series(c, mean[:, i]) for i, c in enumerate(levels)])
            )
            parts.append(
                adapter.forecast_frame(stored=stored, series=series, unit="fraction").with_columns(
                    month=pl.lit(month)
                )
            )
        frame = pl.concat(parts).join(actuals, on=["time_series_id", "valid_time"])
        scored = tails.with_group_and_band(frame=frame, series=series).filter(
            pl.col("group").is_not_null() & pl.col("lead_band").is_not_null()
        )
        scored = scored.with_columns(
            report_group=pl.col("group"), site_report=pl.col("label")
        ).with_columns(*[pl.col(c).cast(pl.Float64) for c in ("actual", *levels)])
        out.append(run(frame=scored, source=source, arm=arm_dir.name, cells={"P": "group"}))
        print(source, arm_dir.name, flush=True)
    return pl.concat(out)


def main() -> None:
    """Run one source named on the command line."""
    source = sys.argv[1]
    simple = {"425": demand_425, "429": wind_429, "430": solar_430}
    result = simple[source]() if source in simple else stale(source=source)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    result.write_parquet(WORK_DIR / f"table_p_{source.replace(':', '_')}.parquet")
    print(source, result.height)


if __name__ == "__main__":
    main()

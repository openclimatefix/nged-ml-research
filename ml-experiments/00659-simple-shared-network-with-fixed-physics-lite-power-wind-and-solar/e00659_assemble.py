# ruff: noqa: E501
"""Put experiment 659's forecasts in experiment 615's table layout and print the headline scores.

Every power is a fraction of experiment 615's scoring capacity (wind: the 99.9th percentile of power
before 2025-10-01; solar: the old capacity), so the tables compare with 615's. Seeds are averaged
quantile by quantile and the 11 quantiles sorted. Wind tables carry `actual` (idle-mask rows set to
zero: "cleaned zeros"), `actual_raw` (the reading as metered), and `masked`.

    uv run --no-sync python ml-experiments/00659-*/e00659_assemble.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00659_common as common
import numpy as np
import polars as pl

Q = [f"q{p}" for p in ("01", "05", "10", "25", "40", "50", "60", "75", "90", "95", "99")]
LEVELS = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
SCORING = json.loads((common.E615_DIR / "wind_default2" / "scoring_capacity.json").read_text())
WIND_ARMS = ("S", "PLP", "PL", "S_cal", "PLP_cal")
SOLAR_ARMS = ("S", "PLP", "PL")
BANDS = ((0, 24), (24, 120), (120, 1e9))


def sorted_mean(*, frame: pl.DataFrame, keys: list[str], extra: list[str]) -> pl.DataFrame:
    """Average the seeds, then sort across the 11 levels."""
    mean = (
        frame.group_by(keys)
        .agg(pl.col([f"q{i}" for i in range(11)]).mean(), *[pl.col(c).first() for c in extra])
        .sort(keys)
    )
    matrix = np.sort(mean.select([f"q{i}" for i in range(11)]).to_numpy(), axis=1)
    return mean.drop([f"q{i}" for i in range(11)]).with_columns(
        *[pl.Series(q, matrix[:, i]) for i, q in enumerate(Q)]
    )


def keep_development(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep origins in the development months and valid times before 2026-07-01."""
    return frame.filter(
        pl.col("init_time").dt.strftime("%Y-%m").is_in(common.DEV_MONTHS)
        & (pl.col("valid_time") < common.VALID_LIMIT)
    )


def wind(
    *, arm: str, directory: Path | None = None, pattern: str | None = None, name: str | None = None
) -> pl.DataFrame:
    """Return one wind arm in the 615 layout (fits of this experiment, or another folder's refits)."""
    directory = directory or common.DATA_DIR / "wind_fits"
    paths = sorted(directory.glob(pattern or f"{arm}_refit=*_seed=*.parquet"))
    frame = pl.concat([pl.read_parquet(p) for p in paths])
    if (
        "actual_norm" not in frame.columns
    ):  # experiment 615's refits keep the observed power in the cases
        cases = pl.concat(
            [
                pl.read_parquet(common.E615_DIR / "wind_inputs" / f"cases_{f}.parquet")
                for f in SCORING
            ]
        ).select("site", "time_series_id", "power_fcst_init_time", "valid_time", "actual_norm")
        frame = frame.join(
            cases,
            on=["site", "time_series_id", "power_fcst_init_time", "valid_time"],
            validate="m:1",
        )
    if "actual_clean" not in frame.columns:  # experiment 615's refits: no idle-mask cleaning column
        frame = frame.with_columns(
            actual_clean=pl.when(pl.col("masked")).then(0.0).otherwise(pl.col("actual_norm"))
        )
    scale = pl.col("site").replace_strict(
        {f: float(v["new_mw"]) for f, v in SCORING.items()}, return_dtype=pl.Float64
    )
    old = pl.col("site").replace_strict(
        {f: float(v["old_mw"]) for f, v in SCORING.items()}, return_dtype=pl.Float64
    )
    frame = frame.with_columns(
        series=pl.col("site"),
        init_time=pl.col("power_fcst_init_time"),
        actual=pl.col("actual_clean") * old / scale,
        actual_raw=pl.col("actual_norm") * old / scale,
        **{f"q{i}": pl.col(f"q{i}") / scale for i in range(11)},
    )
    keys = ["series", "init_time", "valid_time", "lead_hours"]
    out = sorted_mean(frame=frame, keys=keys, extra=["actual", "actual_raw", "masked"])
    out = keep_development(frame=out.filter(pl.col("actual_raw").is_finite()))
    return out.with_columns(tech=pl.lit("wind"), candidate=pl.lit(name or f"wind_{arm}"))


def solar(*, arm: str) -> pl.DataFrame:
    """Return one solar arm in the 615 layout."""
    paths = sorted((common.DATA_DIR / "solar_fits").glob(f"{arm}_refit=*_seed=*.parquet"))
    frame = pl.concat([pl.read_parquet(p) for p in paths]).with_columns(
        series=pl.lit("Solar ") + pl.col("series"),
        valid_time=pl.col("time"),
    )
    keys = ["series", "init_time", "valid_time", "lead_hours"]
    out = sorted_mean(frame=frame, keys=keys, extra=["actual"])
    return keep_development(frame=out).with_columns(
        tech=pl.lit("solar"), candidate=pl.lit(f"solar_{arm}")
    )


def solar_730() -> pl.DataFrame | None:
    """Return `a2_member` refitted with the 730-day capacity (this experiment's refits), if present."""
    paths = sorted((common.DATA_DIR / "solar_a2_member_730" / "a2_member").glob("s?_*.parquet"))
    if len(paths) < 12:
        return None
    frame = pl.concat(
        [pl.read_parquet(p).filter(pl.col("setting") == "own_history") for p in paths]
    )
    frame = frame.select(
        series=pl.lit("Solar ") + pl.col("time_series_id"),
        init_time=pl.col("init_time"),
        valid_time=pl.col("time"),
        lead_hours=pl.col("lead_hours"),
        actual=pl.col("actual"),
        **{q: pl.col(q) for q in Q},
    ).rename({q: f"q{i}" for i, q in enumerate(Q)})
    keys = ["series", "init_time", "valid_time", "lead_hours"]
    out = sorted_mean(frame=frame, keys=keys, extra=["actual"])
    return keep_development(frame=out).with_columns(
        tech=pl.lit("solar"), candidate=pl.lit("solar_a2_member_730")
    )


def scores(*, frame: pl.DataFrame, actual: str, drop_masked: bool = False) -> dict[str, float]:
    """Return the quantile score and normalised MAE of the median (points), coverage, tail share."""
    data = frame.filter(~pl.col("masked")) if drop_masked else frame
    y = data[actual].to_numpy()
    q = data.select(Q).to_numpy()
    diff = y[:, None] - q
    pinball = np.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean()
    return {
        "rows": float(len(y)),
        "score": 100 * pinball,
        "mae_median": 100 * np.abs(diff[:, 5]).mean(),
        "cov_q05_q95": float(((y >= q[:, 1]) & (y <= q[:, 9])).mean()),
        "above_q99": float((y > q[:, 10]).mean()),
        "below_q01": float((y < q[:, 0]).mean()),
    }


def main() -> None:
    """Write every arm's table and print the overall scores."""
    common.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in WIND_ARMS:
        table = wind(arm=arm)
        table.write_parquet(common.TABLE_DIR / f"wind_{arm}.parquet")
        for variant, col, drop in (
            ("cleaned_zeros", "actual", False),
            ("kept_rows", "actual_raw", True),
            ("all_rows", "actual_raw", False),
        ):
            rows.append(
                {
                    "candidate": f"wind_{arm}",
                    "variant": variant,
                    **scores(frame=table, actual=col, drop_masked=drop),
                }
            )
    for directory, name in (
        (common.E615_DIR / "wind_default2", "wind_default2_old_capacity"),
        (common.DATA_DIR / "wind_default2_730", "wind_default2_730"),
        (common.DATA_DIR / "wind_fits_single", "wind_S_single"),
        (common.DATA_DIR / "wind_fits_quarterly", "wind_S_quarterly"),
    ):
        if not list(directory.glob("*refit=*_seed=*.parquet")):
            continue
        table = wind(
            arm="D2",
            directory=directory,
            pattern="refit=*_seed=*.parquet" if "default2" in name else "S_refit=*_seed=*.parquet",
            name=name,
        )
        table.write_parquet(common.TABLE_DIR / f"{name}.parquet")
        for variant, col, drop in (
            ("cleaned_zeros", "actual", False),
            ("kept_rows", "actual_raw", True),
            ("all_rows", "actual_raw", False),
        ):
            rows.append(
                {
                    "candidate": name,
                    "variant": variant,
                    **scores(frame=table, actual=col, drop_masked=drop),
                }
            )
    for arm in SOLAR_ARMS:
        table = solar(arm=arm).with_columns(masked=pl.lit(value=False))
        table.write_parquet(common.TABLE_DIR / f"solar_{arm}.parquet")
        rows.append(
            {
                "candidate": f"solar_{arm}",
                "variant": "all_rows",
                **scores(frame=table, actual="actual"),
            }
        )
    table = solar_730()
    if table is not None:
        table = table.with_columns(masked=pl.lit(value=False))
        table.write_parquet(common.TABLE_DIR / "solar_a2_member_730.parquet")
        rows.append(
            {
                "candidate": "solar_a2_member_730",
                "variant": "all_rows",
                **scores(frame=table, actual="actual"),
            }
        )
    out = pl.DataFrame(rows)
    out.write_csv(common.TABLE_DIR / "overall_scores.csv")
    with pl.Config(tbl_rows=60, tbl_cols=12, float_precision=3):
        print(out)


if __name__ == "__main__":
    main()

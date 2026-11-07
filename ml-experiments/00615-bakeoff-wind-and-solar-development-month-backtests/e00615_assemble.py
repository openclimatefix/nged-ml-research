# ruff: noqa: E501, PERF401
"""Put every candidate's development-month forecasts in one layout, one parquet per candidate.

Layout (fractions of the scoring capacity, which is the series' old capacity for solar and the
99.9th percentile of power before 2025-10-01 for wind): `tech`, `candidate`, `series`
(`Solar A` to `Solar F`, `W1` to `W3`), `init_time`, `valid_time`, `lead_hours`, `actual`,
`q01` to `q99`. Rows keep forecast origins in the development months and valid times before
2026-07-01. Solar rows come from stored forecasts (experiments 310, 430, and 493), which are
daylight half-hours. Wind rows come from this experiment's refits, experiment 347's stored Tuesday
forecasts, and experiment 303's stored default 1.

    source <env>; uv run --no-sync python ml-experiments/00615-*/e00615_assemble.py
"""

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00303_score as score303  # ty: ignore[unresolved-import]
import e00615_common as common
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

Q = list(common.QUANTILES)
E430 = ML_EXPERIMENTS_DATA_DIR / "00430-unified-forecaster-solar-and-pretraining-ablations"
E493 = ML_EXPERIMENTS_DATA_DIR / "00493-solar-annual-against-quarterly-refits"
E310 = ML_EXPERIMENTS_DATA_DIR / "00310-metered-farm-long-lead-scoring"
E347 = ML_EXPERIMENTS_DATA_DIR / "00347-wind-physics-hybrid-plant-then-mdn"
E303 = ML_EXPERIMENTS_DATA_DIR / "00303-wind-adapter-in-production-setting"
WIND_DIR = common.DATA_DIR / "wind_default2"


def keep_development(*, frame: pl.DataFrame, init: str, valid: str) -> pl.DataFrame:
    """Keep origins in the development months and valid times before 2026-07-01."""
    return frame.filter(
        pl.col(init).dt.strftime("%Y-%m").is_in(common.DEV_MONTHS)
        & (pl.col(valid) < common.VALID_LIMIT)
    )


def seed_mean(*, frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Average the quantiles over seeds and sort each row across the 11 levels."""
    mean = frame.group_by(keys).agg(pl.col(Q).mean(), pl.col("actual").first()).sort(keys)
    matrix = np.sort(mean.select(Q).to_numpy().astype(np.float64), axis=1)
    return mean.with_columns(*[pl.Series(q, matrix[:, i]) for i, q in enumerate(Q)])


def finish(*, frame: pl.DataFrame, tech: str, candidate: str) -> pl.DataFrame:
    """Return `frame` in the common layout and write it."""
    out = frame.select(
        pl.lit(tech).alias("tech"),
        pl.lit(candidate).alias("candidate"),
        "series",
        "init_time",
        "valid_time",
        "lead_hours",
        pl.col("actual").cast(pl.Float64),
        *[pl.col(q).cast(pl.Float64) for q in Q],
    ).sort("series", "init_time", "valid_time")
    common.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(common.TABLE_DIR / f"{candidate}.parquet")
    print(f"{candidate}: {out.height:,} rows", flush=True)
    return out


def solar_stored(*, paths: list, seeds: bool, setting_column: bool) -> pl.DataFrame:
    """Read stored solar forecast files into the common keys."""
    frames = []
    for path in paths:
        frame = pl.read_parquet(path)
        if setting_column:
            frame = frame.filter(pl.col("setting") == "own_history")
        frames.append(
            frame.select(
                series=pl.lit("Solar ") + pl.col("time_series_id"),
                init_time=pl.col("init_time"),
                valid_time=pl.col("time"),
                lead_hours=pl.col("lead_hours"),
                actual=pl.col("actual"),
                seed=pl.col("seed"),
                **{q: pl.col(q) for q in Q},
            )
        )
    frame = keep_development(frame=pl.concat(frames), init="init_time", valid="valid_time")
    keys = ["series", "init_time", "valid_time", "lead_hours"]
    return seed_mean(frame=frame, keys=keys) if seeds else frame.drop("seed")


def solar() -> None:
    """Assemble the solar candidates from stored forecasts."""
    quarterly = sorted((E430 / "predictions" / "a2_member").glob("s?_*.parquet"))
    finish(
        frame=solar_stored(paths=quarterly, seeds=True, setting_column=True),
        tech="solar",
        candidate="solar_a2_member_quarterly",
    )
    annual = sorted((E493 / "predictions" / "annual_a2_member").glob("s?_2025-09.parquet"))
    finish(
        frame=solar_stored(paths=annual, seeds=True, setting_column=True),
        tech="solar",
        candidate="solar_a2_member_annual",
    )
    for arm, candidate in (("gam", "solar_gam"), ("xgb", "solar_xgboost")):
        paths = sorted(E310.glob(f"predictions/d*/own_history/{arm}_s0_?_????-??.parquet"))
        finish(
            frame=solar_stored(paths=paths, seeds=False, setting_column=False),
            tech="solar",
            candidate=candidate,
        )
    floors = []
    for band in ("d1", "d2", "d3", "d4_5", "d6_7", "d8_10"):
        floors.append(pl.read_parquet(E310 / f"floors_{band}.parquet"))
    floors = pl.concat(floors)
    for arm, candidate in (
        ("floor_climatology", "solar_climatology"),
        ("floor_heuristic", "solar_manual_heuristic"),
    ):
        part = floors.filter(pl.col("arm") == arm).select(
            series=pl.lit("Solar ") + pl.col("time_series_id"),
            init_time=pl.col("init_time"),
            valid_time=pl.col("time"),
            lead_hours=pl.col("lead_hours"),
            actual=pl.col("actual"),
            **{q: pl.col(q) for q in Q},
        )
        part = keep_development(frame=part, init="init_time", valid="valid_time")
        finish(
            frame=part.with_columns(pl.col(Q).fill_nan(None)).drop_nulls(Q),
            tech="solar",
            candidate=candidate,
        )


def wind_default2() -> pl.DataFrame:
    """Assemble wind default 2 from this experiment's refits (seed mean) in fractions of scoring capacity."""
    scoring = json.loads((WIND_DIR / "scoring_capacity.json").read_text())
    parts = [pl.read_parquet(p) for p in sorted(WIND_DIR.glob("refit=*_seed=*.parquet"))]
    frame = pl.concat(parts)
    cap_new = {f: float(v["new_mw"]) for f, v in scoring.items()}
    cap_old = {f: float(v["old_mw"]) for f, v in scoring.items()}
    case = pl.read_parquet  # noqa: F841
    actual = pl.concat(
        [pl.read_parquet(common.DATA_DIR / "wind_inputs" / f"cases_{f}.parquet") for f in cap_new]
    ).select("site", "time_series_id", "power_fcst_init_time", "valid_time", "actual_norm")
    frame = frame.join(
        actual, on=["site", "time_series_id", "power_fcst_init_time", "valid_time"], validate="m:1"
    ).with_columns(
        series=pl.col("site"),
        init_time=pl.col("power_fcst_init_time"),
        scale=pl.col("site").replace_strict(cap_new, return_dtype=pl.Float64),
        old=pl.col("site").replace_strict(cap_old, return_dtype=pl.Float64),
    )
    frame = frame.with_columns(
        actual=pl.col("actual_norm") * pl.col("old") / pl.col("scale"),
        **{f"q{i}": pl.col(f"q{i}") / pl.col("scale") for i in range(11)},
    )
    renamed = frame.rename(dict(zip([f"q{i}" for i in range(11)], Q, strict=True)))
    kept = renamed.filter(~pl.col("masked")).filter(pl.col("actual").is_finite())
    kept = keep_development(frame=kept, init="init_time", valid="valid_time")
    keys = ["series", "init_time", "valid_time", "lead_hours"]
    return seed_mean(frame=kept, keys=keys)


def wind_climatology(*, cases: pl.DataFrame) -> pl.DataFrame:
    """Per farm, month, and hour quantiles of power before 2025-10-01, for the wind cases."""
    scoring = json.loads((WIND_DIR / "scoring_capacity.json").read_text())
    series = pl.read_parquet(common.SERIES_PATH).filter(
        pl.col("site").is_in(list(common.WIND_SITE.values()))
        & (pl.col("time") < pl.datetime(2025, 10, 1, time_zone="UTC"))
        & pl.col("power").is_finite()
    )
    labels = {v: k for k, v in common.WIND_SITE.items()}
    series = series.with_columns(
        series=pl.col("site").replace_strict(labels),
        scale=pl.col("site").replace_strict(
            {common.WIND_SITE[f]: float(v["new_mw"]) for f, v in scoring.items()},
            return_dtype=pl.Float64,
        ),
    ).with_columns(
        value=pl.col("power") / pl.col("scale"),
        month=pl.col("time").dt.month(),
        hour=pl.col("time").dt.hour(),
    )
    table = series.group_by("series", "month", "hour").agg(
        [
            pl.col("value").quantile(level, interpolation="linear").alias(q)
            for level, q in zip(common.LEVELS, Q, strict=True)
        ]
    )
    keyed = cases.select(
        "series",
        "init_time",
        "valid_time",
        "lead_hours",
        "actual",
        month=pl.col("valid_time").dt.month(),
        hour=pl.col("valid_time").dt.hour(),
    )
    joined = keyed.join(table, on=["series", "month", "hour"], how="left", maintain_order="left")
    matrix = np.sort(joined.select(Q).to_numpy(), axis=1)
    return joined.drop("month", "hour", *Q).with_columns(
        *[pl.Series(q, matrix[:, i]) for i, q in enumerate(Q)]
    )


def wind_stored_tuesdays() -> None:
    """Assemble experiment 347's y2 and experiment 303's default 1 on the stored Tuesday cases."""
    scoring = json.loads((WIND_DIR / "scoring_capacity.json").read_text())
    cases = score303.ordered_cases().select(
        "site", "power_fcst_init_time", "valid_time", "lead_hours", "actual_norm", "masked"
    )
    old_cap = {f: float(v["old_mw"]) for f, v in scoring.items()}
    new_cap = {f: float(v["new_mw"]) for f, v in scoring.items()}
    stored = {
        "wind_default2_stored_tuesdays": np.mean(
            [np.load(E347 / "s4" / "parts" / f"y2_h{s}.npy") for s in range(5)], axis=0
        ),
        "wind_default1_stored_tuesdays": np.load(E303 / "work" / "final" / "P1.npy").mean(axis=0),
    }
    for candidate, quantiles in stored.items():
        if quantiles.shape[0] != cases.height:
            msg = f"{candidate}: {quantiles.shape} does not match {cases.height} cases"
            raise ValueError(msg)
        scale = np.array([old_cap[s] / new_cap[s] for s in cases["site"]])[:, None]
        sorted_q = np.sort(quantiles * scale, axis=1)
        frame = cases.with_columns(
            series=pl.col("site"),
            init_time=pl.col("power_fcst_init_time"),
            actual=pl.col("actual_norm")
            * pl.col("site").replace_strict(old_cap, return_dtype=pl.Float64)
            / pl.col("site").replace_strict(new_cap, return_dtype=pl.Float64),
            **{q: pl.Series(sorted_q[:, i]) for i, q in enumerate(Q)},
        )
        frame = frame.filter(~pl.col("masked") & pl.col("actual").is_finite())
        frame = keep_development(frame=frame, init="init_time", valid="valid_time")
        finish(frame=frame, tech="wind", candidate=candidate)


def wind() -> None:
    """Assemble the wind candidates."""
    default2 = wind_default2()
    finish(frame=default2, tech="wind", candidate="wind_default2")
    finish(frame=wind_climatology(cases=default2), tech="wind", candidate="wind_climatology")
    wind_stored_tuesdays()


def main() -> None:
    """Assemble every candidate the stored files and refits allow."""
    solar()
    if any(WIND_DIR.glob("refit=*_seed=*.parquet")):
        wind()


if __name__ == "__main__":
    main()

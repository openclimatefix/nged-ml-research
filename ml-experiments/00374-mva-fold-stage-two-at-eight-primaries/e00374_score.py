"""Score the rule-chosen fit with experiment 252's stage-2 corrections at the 8 MVA-only Primaries.

The raw forecasts are the re-run of experiment 152's engine in this experiment's work folder
(`e00374_rerun.py`). Stage 2 is experiment 252's `build_arms`, imported unchanged and applied to
each raw arm as experiment 353 did at D8. D8 has no tuning month of its own and borrows the donor
Primaries' spread, as in experiment 353.

| Arm | Built from |
|---|---|
| `a1` | experiment 252's stage-2 arm d, as stored |
| `n2` | the re-run of arm N plus stage-2 arm d |
| `l02` | the re-run of arm A (the fit through the meter) plus stage-2 arm d |
| `chosen2` | `l02` at D12, where experiment 366's rule picks it; else `n2` |
| `n_raw`, `l0_raw`, `chosen_raw` | the same arms without stage 2 (control for experiment 372) |

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries \
        uv run --no-sync python ml-experiments/00374-*/e00374_score.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from pathlib import Path
from typing import Final

import e00252_common as common  # ty: ignore[unresolved-import]
import e00252_score as score_module  # ty: ignore[unresolved-import]
import e00252s2_common as s2common  # ty: ignore[unresolved-import]
import e00252s2_corrections as corr  # ty: ignore[unresolved-import]
import e00252s2_run as stage2  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_calendar import HOLIDAY_PERIODS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITES: Final[tuple[str, ...]] = common.SCORED_SITES
L0_SITES: Final[frozenset[str]] = frozenset({"D12"})
QUANTILES: Final[tuple[str, ...]] = score_module.QUANTILES
KEY: Final[list[str]] = score_module.KEY
REFERENCES: Final[tuple[str, ...]] = tuple(score_module.REFERENCES)
ARMS: Final[tuple[str, ...]] = (
    "a1", "n2", "l02", "chosen2", "n_raw", "l0_raw", "chosen_raw",
)  # fmt: skip
SLUG: Final[str] = "mva-fold-stage-two-at-eight-primaries"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00374-{SLUG}" / "work"
PREDS: Final[Path] = WORK / "preds" / "real"
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
SLICES: Final[tuple[str, ...]] = score_module.SLICES


def read_raw(*, arm: str, site: str) -> pl.DataFrame:
    """Return the re-run's raw forecasts of one arm at one Primary, all three periods."""
    frames = [
        pl.read_parquet(PREDS / f"{period}_{site}.parquet")
        .filter(pl.col("arm") == arm)
        .select(list(s2common.BASE_COLUMNS))
        .with_columns(period=pl.lit(period))
        for period in PERIODS
        if (PREDS / f"{period}_{site}.parquet").exists()
    ]
    return pl.concat(frames).sort(["power_fcst_init_time", "valid_time"])


def with_flags(*, frame: pl.DataFrame, cache, site: str) -> pl.DataFrame:  # noqa: ANN001
    """Add experiment 252's slice flags, computed from the series alone."""
    features = cache.sites[site]
    index = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    unit = features.pv_unit_shape[index]
    local_month = cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % 12 + 1
    holiday = cache.calendar.holiday_period[index].astype(np.int64)
    daylight = unit > score_module.SUN_UP
    edge = float(np.quantile(unit[daylight], 2 / 3))
    winter = np.isin(local_month, [11, 12, 1, 2, 3])
    night = features.elevation_deg[index] < -5
    christmas = [HOLIDAY_PERIODS.index(n) for n in ("Christmas Day", "Boxing Day")]
    return frame.with_columns(
        site=pl.lit(site),
        month=frame["valid_time"].dt.convert_time_zone("Europe/London").dt.strftime("%Y-%m"),
        all=pl.lit(True),
        sunny=pl.Series(unit >= edge),
        winter=pl.Series(winter),
        winter_night=pl.Series(winter & night),
        cold_night=pl.Series(night & (features.temperature_c[index] < 2.0)),
        christmas=pl.Series(np.isin(holiday, christmas)),
        bank_holiday=pl.Series(holiday >= 0),
    )


def site_arms(*, site: str) -> dict[str, pl.DataFrame]:
    """Return every arm's scored-year frame at one Primary."""
    raw = {name: read_raw(arm=arm, site=site) for name, arm in (("n", "N"), ("l0", "A"))}
    staged = {
        name: stage2.build_arms(site=site, frame=corr.add_calendar(frame=frame))["d"]
        for name, frame in raw.items()
    }
    columns = list(s2common.BASE_COLUMNS)
    frames = {
        "a1": pl.read_parquet(s2common.out_path(arm="d", period="scored", site=site)),
        "n2": staged["n"].filter(pl.col("period") == "scored"),
        "l02": staged["l0"].filter(pl.col("period") == "scored"),
        "n_raw": raw["n"].filter(pl.col("period") == "scored"),
        "l0_raw": raw["l0"].filter(pl.col("period") == "scored"),
    }
    chosen = "l0" if site in L0_SITES else "n"
    frames["chosen2"] = frames["l02" if chosen == "l0" else "n2"]
    frames["chosen_raw"] = frames["l0_raw" if chosen == "l0" else "n_raw"]
    return {k: v.select(columns) for k, v in frames.items()}


def site_wide(*, site: str, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return the rows every arm and reference share at one Primary, with scores and flags."""
    frames = site_arms(site=site)
    base = (
        frames["a1"]
        .filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= score_module.HORIZON_HOURS)
        )
        .select([*KEY, "lead_hours", "actual", "run"])
    )
    for arm, frame in frames.items():
        base = base.join(
            frame.select(
                *KEY,
                pl.col("mean").alias(f"{arm}__mean"),
                *[pl.col(q).alias(f"{arm}__{q}") for q in QUANTILES],
            ),
            on=KEY,
            how="inner",
            validate="1:1",
        )
    for name in REFERENCES:
        base = base.join(
            score_module.reference_frame(name=name, site=site), on=KEY, how="inner", validate="1:1"
        )
    base = with_flags(frame=base, cache=cache, site=site)
    actual = base["actual"].to_numpy().astype(np.float64)
    scores = [
        pl.Series(
            f"s_{name}",
            score_module.score(
                actual=actual,
                quantiles=base.select(
                    [f"{name}__{q}" for q in QUANTILES]
                    if name in ARMS
                    else [f"{name}_{q}" for q in QUANTILES]
                ).to_numpy(),
            ),
        )
        for name in (*ARMS, *REFERENCES)
    ]
    return base.with_columns(*scores).with_columns(
        c1=pl.col("s_chosen2") - pl.col("s_a1"),
        c2=pl.col("s_l02") - pl.col("s_n2"),
        c3=pl.col("s_chosen2") - pl.col("s_bg2"),
        c4=pl.col("s_chosen2") - pl.col("s_chosen_raw"),
        a1_bg2=pl.col("s_a1") - pl.col("s_bg2"),
    )


def tails(*, wide: pl.DataFrame, flag: str) -> dict:
    """Return each arm's share of readings at or below q01, q05, q95, and q99 over a slice."""
    part = wide.filter(pl.col(flag))
    actual = part["actual"].to_numpy()
    out: dict = {}
    for name in (*ARMS, *REFERENCES):
        prefix = f"{name}__" if name in ARMS else f"{name}_"
        out[name] = {
            q: float((actual <= part[f"{prefix}{q}"].to_numpy()).mean())
            for q in ("q01", "q05", "q95", "q99")
        }
    return out


def main() -> None:
    """Score everything and write `work/scores.json`."""
    cache = common.load_cache()
    wides = {site: site_wide(site=site, cache=cache) for site in SITES}
    out: dict = {"rows": {site: w.height for site, w in wides.items()}}
    out["absolute"] = {
        site: {
            flag: {
                name: float(w.filter(pl.col(flag))[f"s_{name}"].to_numpy().mean())
                for name in (*ARMS, *REFERENCES)
            }
            for flag in SLICES
        }
        for site, w in wides.items()
    }
    out["contrast"] = {
        column: {
            flag: score_module.contrast(frames=wides, column=column, flag=flag) for flag in SLICES
        }
        for column in ("c1", "c2", "c3", "c4", "a1_bg2")
    }
    out["contrast_d12"] = {
        column: {
            flag: score_module.contrast(frames={"D12": wides["D12"]}, column=column, flag=flag)
            for flag in SLICES
        }
        for column in ("c1", "c2", "c3", "c4")
    }
    out["contrast_leave_one_out"] = {
        column: {
            left_out: score_module.contrast(
                frames={s: w for s, w in wides.items() if s != left_out}, column=column, flag="all"
            )["pooled"]
            for left_out in SITES
        }
        for column in ("c1", "c3")
    }
    out["tails"] = {
        site: {flag: tails(wide=w, flag=flag) for flag in ("all", "sunny", "cold_night")}
        for site, w in wides.items()
    }
    out["checks"] = {
        site: {
            "median_minus_mean": {
                arm: float((w[f"{arm}__q50"] - w[f"{arm}__mean"]).to_numpy().mean())
                for arm in ("a1", "chosen2")
            },
            "mae_mean_forecast": {
                arm: float((w["actual"] - w[f"{arm}__mean"]).abs().to_numpy().mean())
                for arm in ("a1", "chosen2")
            },
        }
        for site, w in wides.items()
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "scores.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["absolute"], indent=1, default=float))
    for column, by_flag in out["contrast"].items():
        for flag, result in by_flag.items():
            point, low, high = result["pooled"]
            print(f"contrast {column} {flag}: {point:+.3f} ({low:+.3f} to {high:+.3f})")


if __name__ == "__main__":
    main()

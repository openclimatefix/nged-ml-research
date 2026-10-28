"""Score the rule-chosen raw fit at the 8 MVA-only Primaries, and the references at all 16 Primaries.

No stage-2 correction is applied: this is close to a repeat of experiment 152 and makes no claim
against experiment 252's arm A1. Arms at each MVA-only Primary (scored year 2025-07-02 to
2026-06-30, leads 0.5 to 168 hours), all as stored by experiment 152:

| Arm | Built from |
|---|---|
| `n` | arm N, the reading taken as signed demand |
| `l0` | arm A, the fit through the meter |
| `chosen` | `l0` at Stickney (D12), where experiment 366's rule picks the fit through the meter; `n` elsewhere |

References: BG2, climatology, and the manual heuristic, as stored by experiment 87, at all 16
Primaries. Intervals come from experiment 252's paired bootstrap that resamples whole valid-time
months. The report also counts each Primary's hand-filtered half-hours in the scored year.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries \
        uv run --no-sync python ml-experiments/00372-*/e00372_score.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from pathlib import Path
from typing import Final

import e00252_common as common  # ty: ignore[unresolved-import]
import e00252_score as score_module  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_calendar import HOLIDAY_PERIODS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

MVA_SITES: Final[tuple[str, ...]] = common.SCORED_SITES
ALL_SITES: Final[tuple[str, ...]] = (
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D18", "D19",
    "D21", "D22",
)  # fmt: skip
L0_SITES: Final[frozenset[str]] = frozenset({"D12"})
"""The Primaries where the rule of experiment 366 chooses the fit through the meter."""
QUANTILES: Final[tuple[str, ...]] = score_module.QUANTILES
KEY: Final[list[str]] = score_module.KEY
ARMS: Final[tuple[str, ...]] = ("n", "l0", "chosen")
REFERENCES: Final[tuple[str, ...]] = tuple(score_module.REFERENCES)
STORED_152: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries"
    / "work"
    / "preds"
    / "real"
)
OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00372-mva-fold-rule-chosen-fit-at-all-primaries" / "work"
)
SLICES: Final[tuple[str, ...]] = score_module.SLICES


def read_152(*, arm: str, site: str) -> pl.DataFrame:
    """Return experiment 152's stored scored-year quantiles of one arm at one Primary."""
    return (
        pl.read_parquet(STORED_152 / f"scored_{site}.parquet")
        .filter(pl.col("arm") == arm)
        .select(*KEY, *QUANTILES)
    )


def with_flags(*, frame: pl.DataFrame, cache, site: str) -> pl.DataFrame:  # noqa: ANN001
    """Add experiment 252's slice flags and drop hand-filtered rows, from the series alone."""
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
        masked=pl.Series(features.switching_mask[index]),
    )


def site_wide(*, site: str, cache) -> pl.DataFrame:  # noqa: ANN001
    """Return the rows every arm and reference share at one Primary, with scores and flags."""
    base = (
        pl.scan_parquet(
            score_module.PREDICTIONS / score_module.REFERENCES["bg2"] / "predictions.parquet"
        )
        .filter(pl.col("site") == site)
        .select(*KEY, "lead_hours", (pl.col("actual") * 100.0).alias("actual"))
        .collect()
    )
    base = with_flags(frame=base, cache=cache, site=site)
    base = base.filter(
        ~pl.col("masked")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= score_module.HORIZON_HOURS)
    )
    for name in REFERENCES:
        base = base.join(
            score_module.reference_frame(name=name, site=site), on=KEY, how="inner", validate="1:1"
        )
    arms = []
    if site in MVA_SITES:
        frames = {"n": read_152(arm="N", site=site), "l0": read_152(arm="A", site=site)}
        frames["chosen"] = frames["l0" if site in L0_SITES else "n"]
        for arm, frame in frames.items():
            base = base.join(
                frame.rename({q: f"{arm}__{q}" for q in QUANTILES}),
                on=KEY,
                how="inner",
                validate="1:1",
            )
        arms = list(ARMS)
    actual = base["actual"].to_numpy().astype(np.float64)
    scores = [
        pl.Series(
            f"s_{name}",
            score_module.score(
                actual=actual,
                quantiles=base.select(
                    [f"{name}__{q}" for q in QUANTILES]
                    if name in arms
                    else [f"{name}_{q}" for q in QUANTILES]
                ).to_numpy(),
            ),
        )
        for name in (*arms, *REFERENCES)
    ]
    base = base.with_columns(*scores)
    if arms:
        base = base.with_columns(
            c1=pl.col("s_chosen") - pl.col("s_n"),
            c2=pl.col("s_l0") - pl.col("s_n"),
            c3=pl.col("s_chosen") - pl.col("s_bg2"),
        )
    return base


def main() -> None:
    """Score everything and write `work/scores.json`."""
    cache = common.load_cache()
    wides = {site: site_wide(site=site, cache=cache) for site in ALL_SITES}
    scored_year = (np.datetime64("2025-07-02T00:00") <= cache.end_times) & (
        cache.end_times <= np.datetime64("2026-06-30T23:30")
    )
    out: dict = {
        "rows": {site: w.height for site, w in wides.items()},
        "masked_half_hours_scored_year": {
            site: int((cache.sites[site].switching_mask & scored_year).sum()) for site in ALL_SITES
        },
        "capacity_mw": {site: cache.sites[site].table.effective_capacity_mw for site in ALL_SITES},
    }
    out["absolute"] = {
        site: {
            flag: {
                name: float(w.filter(pl.col(flag))[f"s_{name}"].to_numpy().mean())
                for name in (*(ARMS if site in MVA_SITES else ()), *REFERENCES)
            }
            for flag in ("all", "sunny", "winter_night", "cold_night")
        }
        for site, w in wides.items()
    }
    mva = {site: wides[site] for site in MVA_SITES}
    out["contrast"] = {
        column: {
            flag: score_module.contrast(frames=mva, column=column, flag=flag) for flag in SLICES
        }
        for column in ("c1", "c2", "c3")
    }
    out["contrast_d12"] = {
        column: {
            flag: score_module.contrast(frames={"D12": wides["D12"]}, column=column, flag=flag)
            for flag in SLICES
        }
        for column in ("c1", "c2", "c3")
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scores.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

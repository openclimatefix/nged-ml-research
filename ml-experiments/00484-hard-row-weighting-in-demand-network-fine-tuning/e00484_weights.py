"""Row weights for experiment 484's fine-tuning arms, defined before any fit.

Every weight multiplies the recency weight that experiment 425 already gives each fine-tuning row
(`0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)`, `e00425_prep.build_ens_fold`), so the matched reference
`W0` is experiment 425's own fit. The early-stopping loss is the unweighted mean negative
log-likelihood on the held-out weeks, as in experiment 425, so every arm stops by the same rule.

- `W-cold`: weight `COLD_WEIGHT` on a night half-hour (the sun below `NIGHT_ELEVATION_DEG`) whose
  ERA5 temperature is at most `COLD_NIGHT_MAX_TEMPERATURE_C`, the exact rule of the scorer's
  `cold_nights` slice (`studies.eval_protocol.slices`), which experiment 470 uses; 1 elsewhere.
  The scorer's rule is a fixed temperature, not a per-Primary decile, so the weighted rows and the
  scored rows are the same kind of row. The weight reads only the row's own valid time, which is
  at or before the fold's refit cutoff.
- `W-curt`: weight `1 + CURT_EXTRA` on a half-hour of one of the six signed-MW Primaries whose
  metered flow is at least `CURT_FRACTION` times that Primary's maximum flow in the fold's
  fine-tuning window (the scaled limit `L = f x maximum` of `studies.node_curtailment`, with
  `f = 0.75`; the avoidable cost is the cost of capping generators when the forecast crosses `L`
  and the energy lost when the flow does). The indicator needs no forecast. 1 elsewhere, and 1 at
  every other Primary because the cost is defined only for the signed-MW six.
- `W-season`: weight `SEASON_WEIGHT` on every half-hour whose local (Europe/London) month is
  December, January, or February; 1 elsewhere.
"""

from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as refs448  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol.slices import (
    COLD_NIGHT_MAX_TEMPERATURE_C,
    NIGHT_ELEVATION_DEG,
    SIGNED_MW_LABELS,
    WINTER_MONTHS,
)

COLD_WEIGHT: Final[float] = 3.0
CURT_FRACTION: Final[float] = 0.75
CURT_EXTRA: Final[float] = 2.0
SEASON_WEIGHT: Final[float] = 3.0
WEIGHTED_ARMS: Final[tuple[str, ...]] = ("W-cold", "W-curt", "W-season")
FLAGS_FILE: Final[str] = "cold_night_flags.parquet"


def cold_flags(*, work_dir: Path) -> pl.DataFrame:
    """Return `time_series_id`, `valid_us`, `cold` for every half-hour of every series (cached).

    `cold` is the scorer's `cold_nights` rule on the ERA5 temperature and the sun's elevation.
    """
    target = work_dir / FLAGS_FILE
    if target.exists():
        return pl.read_parquet(target)
    ctx = prep425.cleaned_scoring_context(series=adapter.series_table())
    context = ctx["context"]
    elevation = context["elevation_deg"].fill_nan(90.0).fill_null(90.0)
    temperature = context["temperature_c"].fill_nan(float("inf")).fill_null(float("inf"))
    out = context.select(
        "time_series_id",
        valid_us=pl.col("valid_time").dt.epoch("us"),
        cold=(elevation < NIGHT_ELEVATION_DEG) & (temperature <= COLD_NIGHT_MAX_TEMPERATURE_C),
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.rename(target)
    return out


def _signed_primaries() -> np.ndarray:
    """Return the Primary indices of the six signed-MW Primaries."""
    sites = refs448.sites_table()
    return sites.filter(pl.col("label").is_in(list(SIGNED_MW_LABELS)))["primary"].to_numpy()


def row_multiplier(
    *, arm: str, valid_us: np.ndarray, primary: np.ndarray, y: np.ndarray, work_dir: Path
) -> np.ndarray:
    """Return an arm's weight on each fine-tuning row, to multiply by the recency weight.

    Args:
        arm: One of `WEIGHTED_ARMS`.
        valid_us: Valid time of each row, microseconds since the epoch.
        primary: The Primary index of each row.
        y: The target of each row in the fold's own units (the fold's flow, so the maximum over
            the window is the Primary's maximum flow).
        work_dir: Where the cold-night flag table is cached.

    Returns:
        A float32 array of the same length: 1 where the arm adds no weight.
    """
    out = np.ones(len(valid_us), dtype=np.float32)
    if arm == "W-cold":
        sites = refs448.sites_table().select("primary", "time_series_id")
        frame = pl.DataFrame({"primary": primary.astype(np.int64), "valid_us": valid_us}).join(
            sites.with_columns(pl.col("primary").cast(pl.Int64)), on="primary", how="left"
        )
        flags = cold_flags(work_dir=work_dir).with_columns(pl.col("time_series_id").cast(pl.Int64))
        joined = frame.select("time_series_id", "valid_us").join(
            flags.unique(subset=["time_series_id", "valid_us"]),
            on=["time_series_id", "valid_us"],
            how="left",
            maintain_order="left",
        )
        cold = joined["cold"].fill_null(value=False).to_numpy()
        out[cold] = COLD_WEIGHT
    elif arm == "W-curt":
        for p in _signed_primaries():
            sel = primary == p
            if not sel.any():
                continue
            limit = CURT_FRACTION * float(y[sel].max())
            if limit <= 0:
                continue
            out[sel & (y >= limit)] = 1.0 + CURT_EXTRA
    elif arm == "W-season":
        local = (
            pl.Series(valid_us).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Europe/London").dt.month().to_numpy()
        )  # fmt: skip
        out[np.isin(local, WINTER_MONTHS)] = SEASON_WEIGHT
    else:
        msg = f"unknown weighted arm {arm!r}"
        raise ValueError(msg)
    return out

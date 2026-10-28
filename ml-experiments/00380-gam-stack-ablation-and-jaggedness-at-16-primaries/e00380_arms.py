"""Build the ablation arms of the experiment 377 stack at all 16 Primaries.

Components of the stack, each switched on or off: the Kalman filter on stage 1's residuals
(`K`), the winter night offset (`O`), recipe R (`R`), and the per-fold stage 2 spread (`F`; off
means stage 1's tuning-month spread, shifted with the mean). Stage 1 with the filter on is
experiment 377's stored `a2`; with the filter off it is `e00380_kalman_off.py`'s output.

| Arm | K | O | R | Spread |
|---|---|---|---|---|
| `full` | on | on | on | per fold (experiment 377's stack) |
| `build0` | off | off | off | stage 1 tuning months |
| `build1` | off | off | off | per fold |
| `build2` | on | off | off | per fold |
| `build3` | on | on | off | per fold (also drop R) |
| `drop_k` | off | on | on | per fold |
| `drop_o` | on | off | on | per fold |
| `drop_f` | on | on | on | stage 1 tuning months |

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
    E00377_SOURCE=control E00377_ARM=a0 uv run --no-sync python \
        ml-experiments/00380-*/e00380_arms.py
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377s2_common as common  # ty: ignore[unresolved-import]
import e00377s2_corrections as corr  # ty: ignore[unresolved-import]
import e00377s3_stack as stack  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
E377_WORK: Final[Path] = DATA / "00377-demand-v1-first-run-feasibility" / "work"
OWN: Final[Path] = DATA / "00380-gam-stack-ablation-and-jaggedness-at-16-primaries" / "work"
SOURCES: Final[dict[str, tuple[str, ...]]] = {
    "control": ("D5", "D9", "D10", "D11", "D14", "D16", "D18"),
    "d15": ("D15",),
    "mva": ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
}
# arm: (kalman, offset, recipe_r, fold_spread)
ARMS: Final[dict[str, tuple[bool, bool, bool, bool]]] = {
    "full": (True, True, True, True),
    "build0": (False, False, False, False),
    "build1": (False, False, False, True),
    "build2": (True, False, False, True),
    "build3": (True, True, False, True),
    "drop_k": (False, True, True, True),
    "drop_o": (True, False, True, True),
    "drop_f": (True, True, True, False),
}
COLUMNS: Final[list[str]] = [*common.BASE_COLUMNS, "arm"]


def read_base(*, site: str, kalman: bool) -> pl.DataFrame:
    """Return stage 1 at a site, with the filter on (experiment 377's `a2`) or off (own)."""
    source = next(s for s, sites in SOURCES.items() if site in sites)
    folder = (
        E377_WORK / source / "arms" / "a0" / "dp_preds"
        if kalman
        else OWN / "kalman_off" / source / "dp_preds"
    )
    pieces = [
        pl.read_parquet(folder / f"a2_{period}_{site}.parquet")
        .select(list(common.BASE_COLUMNS))
        .with_columns(period=pl.lit(period))
        for period in common.PERIODS
        if (folder / f"a2_{period}_{site}.parquet").exists()
    ]
    return corr.add_calendar(frame=pl.concat(pieces).sort(["power_fcst_init_time", "valid_time"]))


def shifted(*, base: pl.DataFrame, offset: bool, recipe_r: bool) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the frame with the corrected mean and the residual, and the shift added."""
    mean_a = base["mean"].to_numpy().astype(np.float64)
    shift = np.zeros(base.height)
    if offset:
        shift, _ = corr.winter_offset_shift(frame=base, variant="winter")
    if recipe_r:
        shift = shift + corr.recipe_r_shift(frame=base, forecast=mean_a + shift)
    mean = mean_a + shift
    frame = base.with_columns(
        pl.Series("mean", mean.astype(np.float32)),
        resid=pl.col("actual").cast(pl.Float64) - pl.Series(mean),
    )
    return frame, shift


def first_readings(*, sites: list[str]) -> dict[str, datetime]:
    """Return each Primary's first reading time from the feature cache."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=common.CACHE_DIR)
    return {
        site: next(
            t.replace(tzinfo=UTC) if t.tzinfo is None else t
            for t, p in zip(cache.times_series(), cache.sites[site].present, strict=True)
            if p
        )
        for site in sites
    }


def build_site(*, site: str, first: datetime, arm_names: list[str]) -> dict[str, pl.DataFrame]:
    """Return the requested arms' frames at one Primary."""
    out = {}
    bases = {kalman: read_base(site=site, kalman=kalman) for kalman in (True, False)}
    for name in arm_names:
        kalman, offset, recipe_r, fold = ARMS[name]
        base = bases[kalman]
        frame, shift = shifted(base=base, offset=offset, recipe_r=recipe_r)
        if fold:
            if site == "D8":
                months = stack.eligible_months(
                    first_reading=first, first_month=stack.D8_FIRST_OWN_MONTH
                )
            else:
                months = stack.eligible_months(first_reading=first)
            result = stack.with_fold_spread(frame=frame, pools=[frame], months=months, arm=name)
        else:
            result = base.with_columns(
                pl.Series("mean", frame["mean"]),
                *[(pl.col(q) + pl.Series(shift)).cast(pl.Float32) for q in common.QUANTILES],
                arm=pl.lit(name),
            )
            result = result.filter(pl.col("period") != "warm_up")
        out[name] = result.select(COLUMNS)
    return out


def main() -> None:
    """Build and write every arm at every Primary."""
    sites = [s for sites in SOURCES.values() for s in sites]
    firsts = first_readings(sites=sites)
    (OWN / "arms").mkdir(parents=True, exist_ok=True)
    for site in sites:
        frames = build_site(site=site, first=firsts[site], arm_names=list(ARMS))
        for name, frame in frames.items():
            target = OWN / "arms" / f"{name}_{site}.parquet"
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            frame.write_parquet(temporary)
            temporary.rename(target)
        print(site, {n: f.height for n, f in frames.items()}, flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

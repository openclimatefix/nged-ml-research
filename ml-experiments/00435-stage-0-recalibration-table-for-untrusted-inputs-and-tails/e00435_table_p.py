"""Experiment 435, step 3: table P on stored experiment 428 forecasts, by condition class.

Every row of a stored forecast gets one condition class, assigned in this order:

1. `real_event`: the target lies in a real load-shift event (`tgt_real`); scored only, never fitted;
2. `lag_missing`: one of the row's three lags (newest, same time yesterday, same time last week) is
   unobserved in the world;
3. `surprise_gt4` and `surprise_2_4`: the largest of the three lags' absolute surprise (the world's
   power minus the frozen lagless median, at the lag's valid time) over the lagless 90% band width
   is above 4, or from 2 to 4;
4. `inj_harm`: an injected event touches the lag window (`win_inj`);
5. `clean`: every other row (world 0 and the injected worlds outside every event).

Rows whose target is in a post-event week (`tgt_post`, unless the row is a real event) or in an
injected event (`tgt_inj`) are dropped, as in experiment 428's own scoring. `real_event` rows are
scored three ways: against the metered actual, and against the NRA-corrected actual at depth
factors 0.75, 1.0, and 1.25 (`basis` column). The surprise classes are read from seed 0 of
the frozen lagless table of experiment 404 (commit `63d3799f`). Seeds of an arm that has three
are averaged in quantile space before fitting. Each scored fold is fitted on the earlier folds
only, so 2025-10 (the first fold) is never scored.

```bash
# PYTHONPATH: experiments 87, 48, 404, 422 and their imports, plus this folder
uv run --no-sync python ml-experiments/00435-*/e00435_table_p.py
```
"""

from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00404_rows as rows404  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import WIDENING, lead_band_expr, tail_table, widen
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = prov.WORK_DIR
PRED_DIR: Final[Path] = prov.PRED_DIR
LAGLESS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00404-mdn-lagless-and-calibration-module-trust-weights"
    / "work"
    / "lagless"
)
WORLD_DIR: Final[Path] = PRED_DIR.parent / "worlds"
EVAL_WORLD_SEEDS: Final[dict[int, int]] = {4: 200, 5: 201, 6: 202}
MONTHS: Final[tuple[str, ...]] = prov.MONTHS
ARM_SEEDS: Final[dict[str, tuple[int, ...]]] = {
    "R-inj": (0, 1, 2),
    "E3": (0, 1, 2),
    "E3-surp": (0, 1, 2),
    "R": (0,),
    "L0": (0,),
    "Bmix": (0,),
    "E1": (0,),
    "E2": (0,),
}
CLASSES: Final[tuple[str, ...]] = (
    "clean",
    "inj_harm",
    "surprise_2_4",
    "surprise_gt4",
    "lag_missing",
    "real_event",
)
FITTED: Final[tuple[str, ...]] = CLASSES[:-1]
KEYS: Final[list[str]] = ["primary", "origin_us", "valid_us", "world"]
LEVELS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
_GRID_START_US: Final[int] = int(common.GRID_START.timestamp() * 1_000_000)
_STEP_US: Final[int] = 1_800_000_000


def grid_of(*, us: np.ndarray) -> np.ndarray:
    """Return the dense-grid index of microsecond timestamps."""
    return (us - _GRID_START_US) // _STEP_US


def world_powers() -> dict[int, np.ndarray]:
    """Return each world's power over capacity, shape (16, grid), by world number."""
    clean = prep422.cleaned_arrays()
    out = {0: np.stack([clean[s] for s in prov.LABELS]).astype(np.float32)}
    for world, seed in EVAL_WORLD_SEEDS.items():
        out[world] = np.load(WORLD_DIR / f"seed{seed}.npz")["power"]
    return out


def lagless_dense(*, month: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the lagless median and 90% band width on the dense grid, shape (16, grid)."""
    table = pl.read_parquet(LAGLESS_DIR / f"month={month}" / "seed=0.parquet")
    size = int(table["grid"].max()) + 1  # ty: ignore[invalid-argument-type]
    median = np.full((len(prov.LABELS), size), np.nan, dtype=np.float32)
    width = np.full_like(median, np.nan)
    p = table["primary"].to_numpy().astype(np.int64)
    g = table["grid"].to_numpy().astype(np.int64)
    median[p, g] = table["q50"].to_numpy()
    width[p, g] = (table["q95"] - table["q05"]).to_numpy()
    return median, width


def add_classes(
    *, frame: pl.DataFrame, power: np.ndarray, median: np.ndarray, width: np.ndarray
) -> pl.DataFrame:
    """Add the largest lag surprise (in band widths), the missing-lag flag, and `cls`."""
    p = frame["primary"].to_numpy().astype(np.int64)
    origin = grid_of(us=frame["origin_us"].to_numpy())
    target = grid_of(us=frame["valid_us"].to_numpy())
    lags = rows404.lag_grids(target=target, origin=origin)
    size = min(power.shape[1], median.shape[1])
    inside = (lags >= 0) & (lags < size)
    safe = np.where(inside, lags, 0)
    observed = power[p[:, None], safe]
    surprise = np.abs(observed - median[p[:, None], safe]) / width[p[:, None], safe]
    surprise = np.where(inside, surprise, np.nan)
    missing = np.isnan(surprise).any(axis=1)
    largest = np.nanmax(np.where(np.isnan(surprise), -np.inf, surprise), axis=1)
    return frame.with_columns(
        surprise=pl.Series(np.where(missing, np.nan, largest)),
        lag_missing=pl.Series(missing),
    ).with_columns(
        cls=pl.when(pl.col("tgt_real"))
        .then(pl.lit("real_event"))
        .when(pl.col("lag_missing"))
        .then(pl.lit("lag_missing"))
        .when(pl.col("surprise") > 4)
        .then(pl.lit("surprise_gt4"))
        .when(pl.col("surprise") > 2)
        .then(pl.lit("surprise_2_4"))
        .when(pl.col("win_inj"))
        .then(pl.lit("inj_harm"))
        .otherwise(pl.lit("clean"))
    )


def nra_corrections() -> dict[float, np.ndarray]:
    """Return the NRA correction (share of capacity, (16, grid)) for each depth factor."""
    z = np.load(WORK_DIR / "nra_correction_share.npz")
    return {f: z[f"f{f}"] for f in (0.75, 1.0, 1.25)}


def load_arm(*, arm: str, month: str, powers: dict[int, np.ndarray]) -> pl.DataFrame:
    """Return one fold of one arm: seed-averaged quantiles, flags, classes, band, and group."""
    median, width = lagless_dense(month=month)
    parts: list[pl.DataFrame] = []
    for world in (0, 4, 5, 6):
        seeds = [
            pl.read_parquet(PRED_DIR / arm / f"{month}_s{s}_w{world}.parquet").sort(KEYS[:3])
            for s in ARM_SEEDS[arm]
        ]
        first = seeds[0].with_columns(world=pl.lit(world))
        for other in seeds[1:]:
            assert other.select(KEYS[:3]).equals(first.select(KEYS[:3])), "seed keys differ"
        mean = np.sort(np.mean([s.select(list(LEVELS)).to_numpy() for s in seeds], axis=0), axis=1)
        first = first.with_columns(*[pl.Series(c, mean[:, i]) for i, c in enumerate(LEVELS)])
        parts.append(add_classes(frame=first, power=powers[world], median=median, width=width))
    frame = pl.concat(parts)
    grid = grid_of(us=frame["valid_us"].to_numpy())
    frame = frame.with_columns(
        **{
            f"nra_{f}": pl.Series(c[frame["primary"].to_numpy().astype(np.int64), grid])
            for f, c in nra_corrections().items()
        }
    )
    frame = frame.filter(
        (~pl.col("tgt_post") | pl.col("tgt_real"))
        & ~pl.col("tgt_inj")
        & pl.col("actual").is_finite()
    )
    return (
        frame.with_columns(
            label=pl.col("primary").replace_strict(
                dict(enumerate(prov.LABELS)), return_dtype=pl.String
            ),
            lead_hours=pl.col("lead").cast(pl.Float64),
            **{c: pl.col(c).cast(pl.Float64) for c in ("actual", *LEVELS)},
        )
        .with_columns(
            group=pl.col("label").replace_strict(prov.GROUP_OF), lead_band=lead_band_expr()
        )
        .filter(pl.col("lead_band").is_not_null())
    )


def main() -> None:
    """Fit on earlier folds, score the next fold, per arm and class."""
    powers = world_powers()
    results: list[pl.DataFrame] = []
    factors_out: list[pl.DataFrame] = []
    for arm in ARM_SEEDS:
        folds = {m: load_arm(arm=arm, month=m, powers=powers) for m in MONTHS}
        for i, month in enumerate(MONTHS[1:], start=1):
            earlier = pl.concat([folds[m] for m in MONTHS[:i]])
            held = folds[month]
            for cls in CLASSES:
                part_held = held.filter(pl.col("cls") == cls)
                if part_held.is_empty():
                    continue
                # `real_event` rows use the factors of `inj_harm`, the closest fitted class.
                fit_cls = cls if cls in FITTED else "inj_harm"
                part_fit = earlier.filter(pl.col("cls") == fit_cls)
                factors = prov.fit_factors(fit=part_fit)
                factors_out.append(
                    factors.with_columns(arm=pl.lit(arm), month=pl.lit(month), cls=pl.lit(cls))
                )
                bases = {"metered": part_held}
                if cls == "real_event":
                    for f in (0.75, 1.0, 1.25):
                        bases[f"nra_x{f}"] = part_held.with_columns(
                            actual=pl.col("actual") - pl.col(f"nra_{f}")
                        )
                for basis, base in bases.items():
                    for variant, frame in {
                        "raw": base,
                        "w1.15": widen(frame=base, factor=WIDENING),
                        "P": prov.apply_factors(frame=base, factors=factors),
                    }.items():
                        results.append(
                            tail_table(frame=frame, by=("group", "lead_band")).with_columns(
                                arm=pl.lit(arm),
                                month=pl.lit(month),
                                cls=pl.lit(cls),
                                variant=pl.lit(variant),
                                basis=pl.lit(basis),
                            )
                        )
        print("done", arm, flush=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(results).write_parquet(WORK_DIR / "table_p_428.parquet")
    pl.concat(factors_out).write_parquet(WORK_DIR / "table_p_factors_full_428.parquet")


if __name__ == "__main__":
    main()

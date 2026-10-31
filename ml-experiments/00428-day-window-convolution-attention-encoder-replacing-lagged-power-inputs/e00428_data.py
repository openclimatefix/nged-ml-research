# ruff: noqa: E501
"""Power worlds, fold rows, and standardisation for experiment 428.

**Worlds.** A world is the 16 Primaries' cleaned power over capacity (experiment 422's
`cleaned_arrays`: flagged half-hours and the D15 and D18 start cut are NaN) with load-conserving
events injected by experiment 402's `inject` over 2024-05-01 to 2026-07-01 at 10% coverage.
World 0 is the clean series. Worlds 1 to 3 (seeds 100 to 102) train; worlds 4 to 6 (seeds 200 to
202) score harm rows and are never trained on. Each world is stored as one `.npz` of power
(primaries, grid) and the injected-event code (0 none, 1 donor, 2 recipient, 3 spike).

**Rows.** A row is one (forecast run, valid time) pair of experiment 397's summary-scheme raw
matrices, with the three lag columns removed. Targets come from the clean series, so a row whose
target is flagged or start-cut is dropped. Folds, purge, held-out weeks, and recency weights are
experiment 422's.

```bash
PYTHONPATH=... DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00428-*/e00428_data.py worlds
```
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00402_inject as inj  # ty: ignore[unresolved-import]
import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00428-day-window-convolution-attention-encoder-replacing-lagged-power-inputs"
    / "work"
)
WORLD_DIR: Final[Path] = WORK_DIR / "worlds"
TRAIN_WORLD_SEEDS: Final[tuple[int, ...]] = (100, 101, 102)
EVAL_WORLD_SEEDS: Final[tuple[int, ...]] = (200, 201, 202)
WORLD_FIRST: Final[datetime] = datetime(2024, 5, 1, tzinfo=UTC)
WORLD_LAST: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
COVERAGE: Final[float] = 0.10
N_TRAIN_WORLDS: Final[int] = len(TRAIN_WORLD_SEEDS)
"""Training draws from worlds 0 (clean) to `N_TRAIN_WORLDS` with equal probability."""
HEADLINE_MAX_HOURS: Final[float] = 240.0
HOLDOUT: Final[datetime] = common.HOLDOUT
"""No experiment scores a valid time on or after 2026-07-01."""


def world_path(*, seed: int) -> Path:
    """Return the file of an injected world."""
    return WORLD_DIR / f"seed{seed}.npz"


def grid_length() -> int:
    """Return the number of half-hours on the dense grid."""
    return len(common.grid_times())


def sites() -> list[str]:
    """Return the 16 Primaries in the order of `prep.primaries()` (the Primary index)."""
    return prep.primaries()["site"].to_list()


def build_world(*, seed: int, ctx: worlds402.Context, clean: dict[str, np.ndarray]) -> str:
    """Inject events into the cleaned power and write the world; return a progress line."""
    target = world_path(seed=seed)
    if target.exists():
        return f"seed {seed}: exists"
    labels = sites()
    assert set(labels) == set(ctx.labels), "the label table and the Primaries disagree"
    injected = inj.inject(
        power={s: clean[s].astype(np.float64) for s in labels},
        capacity_mw=ctx.capacity,
        neighbours=ctx.neighbours,
        magnitude_only=ctx.magnitude_only,
        spike_mad_mw=ctx.stage1_mad_mw,
        pool=worlds402.make_pool(ctx=ctx),
        first=int(common.grid_index(times=WORLD_FIRST)),
        last=int(common.grid_index(times=WORLD_LAST)),
        coverage=COVERAGE,
        seed=seed,
        allowed={s: ~ctx.event_mask[s] & np.isfinite(clean[s]) for s in labels},
    )
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez(
        temporary,
        power=np.stack([injected.power[s] for s in labels]).astype(np.float32),
        code=np.stack([injected.mask[s] for s in labels]).astype(np.int8),
    )
    temporary.rename(target)
    return f"seed {seed}: {injected.events.height:,} events"


def load_stack(*, include_eval: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return power (worlds, primaries, grid) and event codes, world 0 clean, then the injected.

    World order: 0 clean; 1 to 3 the training worlds; 4 to 6 the scoring worlds (if requested).
    """
    clean = prep422.cleaned_arrays()
    base = np.stack([clean[s] for s in sites()]).astype(np.float32)
    seeds = (*TRAIN_WORLD_SEEDS, *(EVAL_WORLD_SEEDS if include_eval else ()))
    powers, codes = [base], [np.zeros(base.shape, dtype=np.int8)]
    for seed in seeds:
        z = np.load(world_path(seed=seed))
        powers.append(z["power"])
        codes.append(z["code"])
    return np.stack(powers), np.stack(codes)


def real_event_mask() -> np.ndarray:
    """Return the padded real-event mask (primaries, grid) from the corrected label table."""
    ctx = worlds402.load_context()
    return np.stack([ctx.event_mask[s] for s in sites()])


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


@dataclass
class Rows:
    """Rows of one fold, ready for the GPU. `x` holds the non-lag inputs, standardised."""

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    origin_idx: np.ndarray
    target_idx: np.ndarray
    lead: np.ndarray
    weight: np.ndarray
    is_val: np.ndarray
    standardiser: prep.Standardiser
    target_mean: np.ndarray
    target_std: np.ndarray
    power_mean: float
    power_std: float
    columns: list[str]
    dropped: int


def non_lag_layout() -> tuple[list[int], list[str], np.ndarray]:
    """Return the kept column positions, their names, and their standardise mask."""
    names, mask, layout = prep.feature_names(scheme="summary")
    drop = set(layout.columns_of(groups=["lags"]))
    keep = [i for i in range(len(names)) if i not in drop]
    return keep, [names[i] for i in keep], mask[keep]


def build_fold(*, refit: datetime, clean: np.ndarray) -> Rows:
    """Return a fold's training and early-stopping rows (24 months before the refit).

    Args:
        refit: 00:00 UTC on the first day of the test month.
        clean: Cleaned power over capacity (primaries, grid), NaN where flagged or cut.

    Returns:
        The training and early-stopping rows.
    """
    keep_cols, columns, mask = non_lag_layout()
    table = prep.primaries()
    parts: list[dict[str, np.ndarray]] = []
    dropped = 0
    for row in table.iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= _us(t=refit))
            & (z["valid"] > _us(t=refit - timedelta(days=prep.WINDOW_DAYS)))
        )
        if not keep.any():
            continue
        target_idx = ((z["valid"][keep] - prep422.GRID_START_US) // prep422.STEP_US).astype(
            np.int64
        )
        origin_idx = ((z["origin"][keep] - prep422.GRID_START_US) // prep422.STEP_US).astype(
            np.int64
        )
        y = clean[row["primary"]][target_idx]
        usable = np.isfinite(y)
        dropped += int((~usable).sum())
        parts.append(
            {
                "x": z["x"][keep][usable][:, keep_cols],
                "y": y[usable],
                "origin": z["origin"][keep][usable],
                "valid": z["valid"][keep][usable],
                "origin_idx": origin_idx[usable],
                "target_idx": target_idx[usable],
                "lead": z["lead"][keep][usable],
                "primary": np.full(int(usable.sum()), row["primary"], dtype=np.int64),
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - _us(t=prep.FIRST_RUN)) // (86_400 * 1_000_000)).astype(np.int64)
    held, purged = prep422.purge_masks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = prep.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / (86_400 * 1_000_000)
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(table)
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    kept = np.nonzero(train | held)[0]
    x_kept = np.empty((len(kept), cat["x"].shape[1]), dtype=np.float32)
    for start in range(0, len(kept), 1_000_000):
        chunk = kept[start : start + 1_000_000]
        x_kept[start : start + len(chunk)] = standardiser.apply(x=cat["x"][chunk])
    return Rows(
        x=x_kept,
        y=cat["y"][kept],
        primary=cat["primary"][kept],
        origin_idx=cat["origin_idx"][kept],
        target_idx=cat["target_idx"][kept],
        lead=cat["lead"][kept],
        weight=weight[kept].astype(np.float32),
        is_val=held[kept],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        power_mean=float(cat["y"][train].mean()),
        power_std=float(cat["y"][train].std()),
        columns=columns,
        dropped=dropped,
    )


@dataclass
class TestRows:
    """Rows of one test month: the same inputs as `Rows`, no target filter."""

    x: np.ndarray
    primary: np.ndarray
    origin_idx: np.ndarray
    target_idx: np.ndarray
    lead: np.ndarray
    origin: np.ndarray
    valid: np.ndarray


def build_test(*, month: str, standardiser: prep.Standardiser) -> TestRows:
    """Return the rows of the daily runs whose origin falls in `month`, leads up to 240 hours."""
    keep_cols, _, _ = non_lag_layout()
    start = _us(t=prep.refit_of(month=month))
    stop = _us(t=(prep.refit_of(month=month) + timedelta(days=32)).replace(day=1))
    parts: list[dict[str, np.ndarray]] = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["origin"] >= start)
            & (z["origin"] < stop)
            & (z["lead"] <= HEADLINE_MAX_HOURS)
            & (z["valid"] < _us(t=HOLDOUT))
        )
        if not keep.any():
            continue
        parts.append(
            {
                "x": standardiser.apply(x=z["x"][keep][:, keep_cols]),
                "origin": z["origin"][keep],
                "valid": z["valid"][keep],
                "origin_idx": ((z["origin"][keep] - prep422.GRID_START_US) // prep422.STEP_US),
                "target_idx": ((z["valid"][keep] - prep422.GRID_START_US) // prep422.STEP_US),
                "lead": z["lead"][keep],
                "primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64),
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return TestRows(
        x=cat["x"],
        primary=cat["primary"],
        origin_idx=cat["origin_idx"].astype(np.int64),
        target_idx=cat["target_idx"].astype(np.int64),
        lead=cat["lead"],
        origin=cat["origin"],
        valid=cat["valid"],
    )


def main() -> None:
    """Build the injected worlds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worlds", "check"))
    args = parser.parse_args()
    clean = prep422.cleaned_arrays()
    if args.command == "worlds":
        ctx = worlds402.load_context()
        for seed in (*TRAIN_WORLD_SEEDS, *EVAL_WORLD_SEEDS):
            print(build_world(seed=seed, ctx=ctx, clean=clean), flush=True)
    else:
        stack = np.stack([clean[s] for s in sites()])
        rows = build_fold(refit=prep.refit_of(month="2026-01"), clean=stack)
        print(
            f"fold 2026-01: {len(rows.x):,} rows ({int(rows.is_val.sum()):,} held out), "
            f"{rows.dropped:,} unusable targets, {len(rows.columns)} columns, "
            f"power mean {rows.power_mean:.3f} sd {rows.power_std:.3f}"
        )
        test = build_test(month="2026-01", standardiser=rows.standardiser)
        print(f"test 2026-01: {len(test.x):,} rows")


if __name__ == "__main__":
    main()

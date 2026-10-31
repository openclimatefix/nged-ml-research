# ruff: noqa: E501, PT018
"""Training and held-out worlds with genuine level changes for experiment 446 (arm U1n and U2).

**What a genuine level change is.** A step in a Primary's normal running arrangement (NRA) level
that is real: the target moves with the recent power. Experiment 428's injected worlds change the
recent power only (a load-shift event leaves the target alone); a genuine level change adds the same
step to the power at every half-hour of its span, so the lags, the window and the target all see it.
A jump at the lags therefore cannot be told apart from a load-shift event at its start, which is
what the scale-only path of arms U2, U2i, and U2c learns to widen for.

**Events.** Depth and duration are drawn together from one row of the corrected label table
(`sw_block_detector_label_table_corrected.csv` in experiment 395's folder): the depth is
`abs(corrected_depth_first_mw)` over the effective capacity of the row's first Primary (a share of
capacity), the duration is `duration_hours` in half-hours. The sign is random. Events are placed at
random starts, never overlapping, until the covered half-hours reach `rate` of the grid. The
untrimmed duration distribution is the table's own; the trimmed one keeps only rows of at most 7
days. MVA-only Primaries cannot go below zero, so `apply_shift` clips their shifted power at zero
(lags and target alike).

**Files.** One `.npz` per (rate, duration, seed) in this experiment's private work folder, holding
`shift` (primaries, grid) as the share of capacity added, and `onset` (primaries, grid) the grid
index of the first half-hour of the event covering each half-hour (-1 where none).

**Seeds.** Training worlds 0 to 3 of experiment 428's mix draw seeds 300 to 303; the three held-out
worlds (worlds 7 to 9 of this experiment's evaluation stack, built on the real clean series) draw
seeds 400 to 402 at `HELD_OUT_RATE`.

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00446-* DATA_PATH_INTERNAL=... \
  uv run --no-sync python ml-experiments/00446-*/e00446_worlds.py check
uv run --no-sync python ml-experiments/00446-*/e00446_worlds.py build
uv run --no-sync python ml-experiments/00446-*/e00446_worlds.py selftest
```
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00446-honest-bands-and-tail-calibration-on-the-s1-network" / "work"
)
"""This experiment's private work folder: the only folder this experiment writes."""
WORLD_DIR: Final[Path] = WORK_DIR / "worlds"
LABEL_TABLE: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts"
    / "sw_block_detector_label_table_corrected.csv"
)
TRAIN_SEEDS: Final[tuple[int, ...]] = (300, 301, 302, 303)
"""One seed per training world 0 to 3."""
HELD_OUT_SEEDS: Final[tuple[int, ...]] = (400, 401, 402)
RATES: Final[tuple[float, ...]] = (0.02, 0.05)
HELD_OUT_RATE: Final[float] = 0.03
"""Between the two training rates. The plan fixes no rate for the held-out world."""
TRIM_HOURS: Final[float] = 168.0
"""The trimmed duration distribution keeps label rows of at most 7 days."""
MAX_DRAWS: Final[int] = 200_000


@dataclass(frozen=True)
class LabelRanges:
    """Depth (share of capacity) and duration (half-hours) of every usable label-table row."""

    depth: np.ndarray
    steps: np.ndarray


def label_ranges(*, capacity: dict[str, float], trimmed: bool) -> LabelRanges:
    """Return the label table's depths as shares of capacity and durations in half-hours.

    Args:
        capacity: Effective capacity in MW of each Primary label.
        trimmed: Keep only rows lasting at most `TRIM_HOURS`.

    Returns:
        The paired depth and duration arrays (rows with a missing depth or capacity are dropped).
    """
    table = pl.read_csv(LABEL_TABLE).filter(
        pl.col("corrected_depth_first_mw").is_not_null() & pl.col("duration_hours").is_not_null()
    )
    table = table.filter(pl.col("first_primary").is_in(list(capacity)))
    depth = np.array(
        [
            abs(d) / capacity[p]
            for d, p in zip(
                table["corrected_depth_first_mw"].to_list(),
                table["first_primary"].to_list(),
                strict=True,
            )
        ]
    )
    hours = table["duration_hours"].to_numpy().astype(np.float64)
    steps = np.maximum(np.ceil(hours * 2.0), 1.0).astype(np.int64)
    keep = (depth > 0) & np.isfinite(depth)
    if trimmed:
        keep &= hours <= TRIM_HOURS
    return LabelRanges(depth=depth[keep], steps=steps[keep])


def duration_check(*, table: Path = LABEL_TABLE) -> dict[str, float]:
    """Return the median duration in hours and the share of events above 7 days (the plan's check)."""
    hours = pl.read_csv(table)["duration_hours"].drop_nulls().to_numpy().astype(np.float64)
    return {
        "events": len(hours),
        "median_hours": float(np.median(hours)),
        "share_above_7_days": float((hours > TRIM_HOURS).mean()),
    }


def draw_events(
    *, ranges: LabelRanges, observed: np.ndarray, rate: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(shift, onset)` of random non-overlapping level changes covering `rate` of the data.

    Args:
        ranges: The label table's paired depths and durations.
        observed: (primaries, grid) bool, True where the Primary has a clean observation.
        rate: Share of each Primary's observed half-hours inside an event.
        seed: Seed of the draw.

    Returns:
        `shift` (primaries, grid) float32, the signed share of capacity added, and `onset`
        (primaries, grid) int32, the grid index of the event's first half-hour (-1 outside).
    """
    rng = np.random.default_rng(seed)
    n_primaries, grid = observed.shape
    shift = np.zeros((n_primaries, grid), dtype=np.float32)
    onset = np.full((n_primaries, grid), -1, dtype=np.int32)
    for p in range(n_primaries):
        wanted = int(rate * observed[p].sum())
        covered = 0
        for _ in range(MAX_DRAWS):
            if covered >= wanted:
                break
            i = int(rng.integers(len(ranges.depth)))
            start = int(rng.integers(0, grid))
            end = min(grid, start + int(ranges.steps[i]))
            if (onset[p, start:end] >= 0).any():
                continue
            sign = 1.0 if rng.random() < 0.5 else -1.0
            shift[p, start:end] = sign * ranges.depth[i]
            onset[p, start:end] = start
            covered += int(observed[p, start:end].sum())
    return shift, onset


def apply_shift(*, power: np.ndarray, shift: np.ndarray, nonnegative: np.ndarray) -> np.ndarray:
    """Return `power + shift`, clipped at zero for the Primaries flagged in `nonnegative`.

    Used for the lags and the target alike, so a genuine level change moves both together.

    Args:
        power: Power over capacity, shape (..., primaries, grid); NaN stays NaN.
        shift: The level change, shape (primaries, grid).
        nonnegative: (primaries,) True for MVA-only Primaries.

    Returns:
        The shifted power, same shape as `power`.
    """
    moved = power + shift
    floor = np.where(nonnegative[:, None], 0.0, -np.inf)
    return np.maximum(moved, floor).astype(power.dtype)


def world_file(*, rate: float, trimmed: bool, seed: int) -> Path:
    """Return the file of a (rate, duration distribution, seed) world."""
    pct = round(rate * 100)
    return WORLD_DIR / f"shift_r{pct}_{'trim' if trimmed else 'full'}_seed{seed}.npz"


def load_shift(*, rate: float, trimmed: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return `(shift, onset)` of a stored world, or fail with the command that builds it."""
    path = world_file(rate=rate, trimmed=trimmed, seed=seed)
    if not path.exists():
        msg = f"{path} is missing: run `e00446_worlds.py build`"
        raise FileNotFoundError(msg)
    z = np.load(path)
    return z["shift"], z["onset"]


def build_one(*, ranges: LabelRanges, observed: np.ndarray, rate: float, trimmed: bool,
              seed: int) -> str:  # fmt: skip
    """Draw and store one world unless it exists; return a progress line."""
    path = world_file(rate=rate, trimmed=trimmed, seed=seed)
    if path.exists():
        return f"{path.name}: exists"
    shift, onset = draw_events(ranges=ranges, observed=observed, rate=rate, seed=seed)
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, shift=shift, onset=onset)
    temporary.rename(path)
    share = ((onset >= 0) & observed).sum(axis=1) / observed.sum(axis=1)
    shares = " ".join(f"{v:.2%}" for v in share)
    return f"{path.name}: share of observed half-hours inside an event per Primary: {shares}"


def build_all() -> list[str]:
    """Build the training worlds (both rates, both duration distributions) and the held-out ones."""
    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
    import e00428_data as data  # ty: ignore[unresolved-import]

    ctx = worlds402.load_context()
    stack, _ = data.load_stack(include_eval=False)
    observed = np.isfinite(stack[0])
    assert observed.shape == (len(data.sites()), len(common.grid_times())), observed.shape
    lines = []
    for trimmed in (False, True):
        ranges = label_ranges(capacity=ctx.capacity, trimmed=trimmed)
        for rate in RATES:
            lines += [
                build_one(ranges=ranges, observed=observed, rate=rate, trimmed=trimmed, seed=s)
                for s in TRAIN_SEEDS
            ]
    full = label_ranges(capacity=ctx.capacity, trimmed=False)
    lines += [
        build_one(ranges=full, observed=observed, rate=HELD_OUT_RATE, trimmed=False, seed=s)
        for s in HELD_OUT_SEEDS
    ]
    return lines


def selftest() -> None:
    """Check the draw and the shift on a toy grid (no data needed)."""
    ranges = LabelRanges(depth=np.array([0.1, 0.3]), steps=np.array([10, 40]))
    observed = np.ones((3, 2_000), dtype=bool)
    observed[1, :1_000] = False  # half of Primary 1's grid is unobserved
    shift, onset = draw_events(ranges=ranges, observed=observed, rate=0.05, seed=1)
    covered = ((onset >= 0) & observed).sum(axis=1) / observed.sum(axis=1)
    assert (covered >= 0.05).all() and (covered < 0.12).all(), covered
    assert ((shift != 0) == (onset >= 0)).all(), "shift and onset must cover the same half-hours"
    assert {round(float(v), 3) for v in np.unique(np.abs(shift[shift != 0]))} <= {0.1, 0.3}
    first = np.nonzero(onset[0] >= 0)[0][0]
    assert onset[0, first] == first, "the onset of an event is its first half-hour"
    # Lags and target move together: the same function builds both, so a target read from the
    # shifted clean series equals the shifted series at the target's index.
    rng = np.random.default_rng(0)
    clean = rng.uniform(0.0, 1.0, (3, 2_000)).astype(np.float32)
    nonneg = np.array([True, False, False])
    series = apply_shift(power=clean, shift=shift, nonnegative=nonneg)
    target_index = rng.integers(0, 2_000, 500)
    lag_index = target_index - 336
    p = rng.integers(0, 3, 500)
    target_from_world = series[p, target_index]
    in_event = shift[p, target_index] != 0
    assert in_event.any(), "the toy draw must put some targets inside an event"
    free = in_event & (p != 0)
    unclipped = clean[p, target_index] + shift[p, target_index]
    assert np.allclose(target_from_world[free], unclipped[free]), "target must carry the shift"
    usable = (p != 0) & (lag_index >= 0)
    lag_moved = series[p, lag_index.clip(0)] - clean[p, lag_index.clip(0)]
    assert np.allclose(lag_moved[usable], shift[p, lag_index.clip(0)][usable], atol=1e-6), (
        "a lag carries the same shift as a target at its index would"
    )
    assert (series[0] >= 0).all(), "MVA-only power must not go below zero"
    nan_clean = clean.copy()
    nan_clean[1, 5] = np.nan
    assert np.isnan(apply_shift(power=nan_clean, shift=shift, nonnegative=nonneg)[1, 5])
    print("selftest passed")


def main() -> None:
    """Check the label table, build the worlds, or run the selftest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "build", "selftest"))
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "check":
        report = duration_check()
        print(json.dumps(report, indent=1))
        print(
            "plan: median near 48 h and about 10% above 7 days (unverified); "
            f"table: median {report['median_hours']:.1f} h, "
            f"{report['share_above_7_days']:.1%} above 7 days"
        )
    else:
        for line in build_all():
            print(line, flush=True)


if __name__ == "__main__":
    main()

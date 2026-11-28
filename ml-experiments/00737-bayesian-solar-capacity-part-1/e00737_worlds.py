"""The worlds of experiment 737: a Primary's real record with a synthetic fleet added.

A world is one Primary's cleaned signed power with an injected fleet subtracted (generation lowers
the reading), and, in two world types, a positive load added. Each world is a function of the
Primary's inputs, the world's seed, and the Primary's index, and of nothing else: the power never
enters the injected output. The fleet's capacity follows a ramp, a step, or an accelerating path
over the 18 quarters. The orientation mix, the tilt, and the DC:AC ratio of a fleet are drawn from
the seed, except in a matched world, which takes the prior's own.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Final

import numpy as np
from e00737_common import (
    CLOUD_LOAD_SHARE,
    DIRICHLET_CONCENTRATION,
    FARM_MIN_FINITE_SHARE,
    FARM_PHYSICS_SHARE,
    FLEET_MIX,
    G2_BRIGHT_W_M2,
    G2_MIN_NONZERO_SHARE,
    G2_RTOL,
    GRID_LENGTH,
    MAIN_SEEDS,
    MATCHED_SEEDS,
    N_QUARTERS,
    PRIMARIES,
    RATIO_RANGE,
    SHAPES,
    SIZES_MW,
    SPECIAL_SEED,
    STEP_QUARTER,
    TILT_RANGE_DEG,
    TRANSFER_COUNT,
    TRANSFER_DAYS,
    TRANSFER_MW,
    TUNING_SEEDS,
    TUNING_SIZES_MW,
)
from e00737_forward import PRIOR_FLEET, FleetParams
from e00737_inputs import PrimaryInputs, load_farm_per_unit, unit_term

TRANSFER_HALF_HOURS: Final[int] = TRANSFER_DAYS * 48
MIN_TRANSFER_FINITE_SHARE: Final[float] = 0.8
"""Each transfer window must hold finite power in at least this share of its half-hours."""
KINDS: Final[tuple[str, ...]] = (
    "main",
    "matched",
    "farm",
    "cloud",
    "transfer",
    "placebo",
    "control",
    "tuning",
)


@dataclass(frozen=True)
class WorldSpec:
    """The recipe of one world.

    Attributes:
        primary: The Primary's label.
        kind: One of `KINDS`.
        size_mw: The fleet's final AC capacity (zero for a placebo or a control).
        shape: The growth shape (`none` for a placebo or a control).
        seed: The world's seed.
    """

    primary: str
    kind: str
    size_mw: float
    shape: str
    seed: int

    @property
    def name(self) -> str:
        """The world's file name stem."""
        if self.kind in ("placebo", "control"):
            return self.kind
        return f"{self.kind}_{self.size_mw:g}_{self.shape}_{self.seed}"

    @property
    def primary_index(self) -> int:
        """The Primary's position in `PRIMARIES`."""
        return PRIMARIES.index(self.primary)


def world_specs(*, primary: str) -> list[WorldSpec]:
    """Return every scored world of one Primary: 27 main, 4 matched, 2 farm, and 5 more.

    Args:
        primary: The Primary's label.

    Returns:
        The 37 specs: 27 main, 4 matched, 2 real-farm, 1 cloud, 1 transfer, 1 placebo, 1 control.
    """
    specs = [
        WorldSpec(primary, "main", size, shape, seed)
        for size in SIZES_MW
        for shape in SHAPES
        for seed in MAIN_SEEDS
    ]
    specs += [
        WorldSpec(primary, "matched", size, "ramp", seed)
        for size in (2.5, 5.0)
        for seed in MATCHED_SEEDS
    ]
    specs += [WorldSpec(primary, "farm", 2.5, "ramp", seed) for seed in MATCHED_SEEDS]
    specs += [
        WorldSpec(primary, "cloud", 2.5, "ramp", SPECIAL_SEED),
        WorldSpec(primary, "transfer", 2.5, "ramp", SPECIAL_SEED),
        WorldSpec(primary, "placebo", 0.0, "none", SPECIAL_SEED),
        WorldSpec(primary, "control", 0.0, "none", SPECIAL_SEED),
    ]
    return specs


def tuning_specs(*, primary: str) -> list[WorldSpec]:
    """Return the 20 tuning worlds of one Primary: 10 seeds at 2 MW and 4 MW.

    The growth shape follows from the seed: its remainder modulo 3 picks ramp, step, or
    accelerating.

    Args:
        primary: The Primary's label.

    Returns:
        The specs.
    """
    return [
        WorldSpec(primary, "tuning", size, SHAPES[seed % 3], seed)
        for seed in TUNING_SEEDS
        for size in TUNING_SIZES_MW
    ]


def capacity_path(*, shape: str, size_mw: float) -> np.ndarray:
    """Return a fleet's AC capacity in each quarter.

    Args:
        shape: `ramp` (linear from zero), `step` (zero for 8 quarters, then the final capacity),
            `accelerating` (quadratic from zero), or `none`.
        size_mw: The final AC capacity.

    Returns:
        18 capacities in megawatts.

    Raises:
        ValueError: When the shape is not one of the four.
    """
    q = np.arange(N_QUARTERS)
    last = N_QUARTERS - 1
    if shape == "ramp":
        return size_mw * q / last
    if shape == "step":
        return size_mw * (q >= STEP_QUARTER).astype(float)
    if shape == "accelerating":
        return size_mw * (q / last) ** 2
    if shape == "none":
        return np.zeros(N_QUARTERS)
    msg = f"unknown growth shape {shape!r}"
    raise ValueError(msg)


def draw_fleet(*, seed: int, primary_index: int, matched: bool) -> FleetParams:
    """Draw a fleet's orientation mix, tilt, and DC:AC ratio from a seed.

    Args:
        seed: The world's seed.
        primary_index: The Primary's position, so that Primaries get different fleets.
        matched: When true, return the prior's own fleet.

    Returns:
        The fleet.
    """
    if matched:
        return PRIOR_FLEET
    rng = np.random.default_rng([seed, primary_index])
    mix = rng.dirichlet(DIRICHLET_CONCENTRATION * np.array(FLEET_MIX))
    return FleetParams(
        mix=(float(mix[0]), float(mix[1]), float(mix[2])),
        tilt_deg=float(rng.uniform(*TILT_RANGE_DEG)),
        dc_ac_ratio=float(rng.uniform(*RATIO_RANGE)),
    )


def fleet_of(*, spec: WorldSpec) -> FleetParams | None:
    """Return the fleet a world injects, or None for a placebo or a control.

    Args:
        spec: The world.

    Returns:
        The fleet.
    """
    if spec.kind in ("placebo", "control"):
        return None
    return draw_fleet(
        seed=spec.seed, primary_index=spec.primary_index, matched=spec.kind == "matched"
    )


def farm_index(*, primary_index: int, seed: int) -> int:
    """Return the farm that a real-farm world uses: farm k for the first seed, k + 1 for the second.

    Args:
        primary_index: The Primary's position in `PRIMARIES`.
        seed: The world's seed.

    Returns:
        A farm number 0 to 5.
    """
    return (primary_index + MATCHED_SEEDS.index(seed)) % 6


def quarterly_finite_share(
    *, finite: np.ndarray, sun_up: np.ndarray, quarter: np.ndarray
) -> np.ndarray:
    """Return, for each quarter, the share of its daylight half-hours with a finite value.

    Args:
        finite: Whether each half-hour holds a finite value.
        sun_up: Whether each half-hour is a daylight half-hour.
        quarter: Each half-hour's quarter.

    Returns:
        18 shares (zero where a quarter has no daylight half-hour).
    """
    total = np.bincount(quarter[sun_up], minlength=N_QUARTERS).astype(float)
    have = np.bincount(quarter[sun_up & finite], minlength=N_QUARTERS).astype(float)
    return np.divide(have, total, out=np.zeros(N_QUARTERS), where=total > 0)


def farm_unit_output(
    *, physics: np.ndarray, farm_per_unit: np.ndarray, sun_up: np.ndarray, quarter: np.ndarray
) -> np.ndarray:
    """Return the per-megawatt output of a fleet that is half a metered farm and half physics.

    A half-hour in which the farm has no finite reading takes the physics output alone. A quarter
    in which the farm has finite readings in under 60% of its daylight half-hours takes the physics
    output alone. A farm reading is clipped at zero, and a half-hour with the sun below 5 degrees
    takes the physics output (which is exactly zero there).

    Args:
        physics: The physics fleet's unit output.
        farm_per_unit: The farm's power per unit of its capacity, NaN where missing.
        sun_up: Whether the sun is more than 5 degrees above the horizon.
        quarter: Each half-hour's quarter.

    Returns:
        The unit output of the mixed fleet.
    """
    finite = np.isfinite(farm_per_unit)
    share = quarterly_finite_share(finite=finite, sun_up=sun_up, quarter=quarter)
    quarter_ok = share[quarter] >= FARM_MIN_FINITE_SHARE
    use_farm = finite & sun_up & quarter_ok
    farm = np.clip(np.nan_to_num(farm_per_unit), 0.0, None)
    return np.where(
        use_farm,
        FARM_PHYSICS_SHARE * physics + (1.0 - FARM_PHYSICS_SHARE) * farm,
        physics,
    )


def cloud_load_shape(*, inputs: PrimaryInputs) -> np.ndarray:
    """Return experiment 40's `flat` confound shape on the grid: the day's cloud, scaled to mean 1.

    The cloud response of a half-hour is `1 - min(GHI / clear-sky GHI, 1)` of its CAMS hour. The
    shape is taken from `e00040_arms.confound_shapes` over the Primary's usable daylight
    half-hours (the window) and is zero elsewhere.

    Args:
        inputs: The Primary's inputs.

    Returns:
        An array of `GRID_LENGTH` values.
    """
    import e00040_arms as arms40  # ty: ignore[unresolved-import]
    import e00040_model as model40  # ty: ignore[unresolved-import]

    ghi = np.repeat(inputs.hourly["ghi_w_m2"].to_numpy(), 2)
    clear = np.repeat(inputs.hourly["clear_sky_ghi_w_m2"].to_numpy(), 2)
    window_rows = inputs.usable_daylight & (clear > 0)
    cloud = 1.0 - np.minimum(ghi[window_rows] / clear[window_rows], 1.0)
    zeros = np.zeros(int(window_rows.sum()))
    window = model40.Window(
        power=zeros,
        u=zeros,
        hdd=zeros,
        cloud=cloud,
        cells=zeros.astype(int),
        days=zeros.astype(int),
        month=np.ones(len(zeros), dtype=int),
        half_hour=zeros.astype(int),
        off=zeros,
        u_clear=zeros,
        day_cloud=zeros,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        flat = arms40.confound_shapes(window=window)["flat"]
    out = np.zeros(GRID_LENGTH)
    out[window_rows] = flat
    return out


def transfer_windows(*, seed: int, primary_index: int, power: np.ndarray) -> list[tuple[int, int]]:
    """Place three 30-day windows without overlap, each with mostly finite power.

    Args:
        seed: The world's seed.
        primary_index: The Primary's position.
        power: The Primary's real power on the grid.

    Returns:
        Three `(start, end)` half-hour index pairs, sorted by start.

    Raises:
        RuntimeError: When no placement is found in 10,000 tries.
    """
    rng = np.random.default_rng([seed, primary_index, 7])
    finite = np.isfinite(power)
    cumulative = np.concatenate([[0], np.cumsum(finite)])
    for _ in range(10_000):
        starts = sorted(
            int(s) for s in rng.integers(0, GRID_LENGTH - TRANSFER_HALF_HOURS, TRANSFER_COUNT)
        )
        if any(b - a < TRANSFER_HALF_HOURS for a, b in itertools.pairwise(starts)):
            continue
        counts = [cumulative[s + TRANSFER_HALF_HOURS] - cumulative[s] for s in starts]
        if all(c >= MIN_TRANSFER_FINITE_SHARE * TRANSFER_HALF_HOURS for c in counts):
            return [(s, s + TRANSFER_HALF_HOURS) for s in starts]
    msg = "no placement of three transfer windows was found"
    raise RuntimeError(msg)


def transfer_addition(*, windows: list[tuple[int, int]], power: np.ndarray) -> np.ndarray:
    """Return the flat 1.5 MW addition of the windows, on half-hours that have a reading.

    This is a direct array addition. A test compares it with `e00660_generator.apply_events`.

    Args:
        windows: The `(start, end)` pairs.
        power: The Primary's real power (its NaN positions get no addition).

    Returns:
        An array of `GRID_LENGTH` values, megawatts.
    """
    out = np.zeros(GRID_LENGTH)
    for start, end in windows:
        out[start:end] = TRANSFER_MW
    return np.where(np.isfinite(power), out, 0.0)


@dataclass(frozen=True)
class World:
    """One built world.

    Attributes:
        spec: The recipe.
        power: The world's power on the grid: the real power minus the injection plus the addition.
        injected: The injected output, positive, in megawatts.
        addition: The positive load added to the power (the cloud load or the transfers).
        truth: The injected AC capacity of each quarter.
        fleet: The injected fleet's shape (None when nothing is injected).
        windows: The transfer windows (empty except in the transfer world).
    """

    spec: WorldSpec
    power: np.ndarray
    injected: np.ndarray
    addition: np.ndarray
    truth: np.ndarray
    fleet: FleetParams | None
    windows: list[tuple[int, int]]


def injected_output(
    *, inputs: PrimaryInputs, spec: WorldSpec, fleet: FleetParams, physics: np.ndarray | None = None
) -> np.ndarray:
    """Return a world's injected output: the quarter's capacity times the fleet's unit output.

    A real-farm world uses `farm_unit_output` in place of the physics output alone. The output
    reads only the sky, the weather, the sun's position, and the spec.

    Args:
        inputs: The Primary's inputs.
        spec: The world.
        fleet: The fleet's shape.
        physics: The fleet's unit output when already computed.

    Returns:
        The injected output in megawatts at every half-hour of the grid.
    """
    unit = physics if physics is not None else unit_term(inputs=inputs, fleet=fleet)
    if spec.kind == "farm":
        unit = farm_unit_output(
            physics=unit,
            farm_per_unit=load_farm_per_unit(
                farm=farm_index(primary_index=spec.primary_index, seed=spec.seed)
            ),
            sun_up=inputs.sun_up,
            quarter=inputs.quarter,
        )
    path = capacity_path(shape=spec.shape, size_mw=spec.size_mw)
    return path[inputs.quarter] * unit


def build_world(*, inputs: PrimaryInputs, spec: WorldSpec) -> World:
    """Build one world from a Primary's inputs and a spec.

    Args:
        inputs: The Primary's inputs.
        spec: The world's recipe.

    Returns:
        The world.
    """
    fleet = fleet_of(spec=spec)
    truth = capacity_path(shape=spec.shape, size_mw=spec.size_mw)
    zeros = np.zeros(GRID_LENGTH)
    injected = zeros if fleet is None else injected_output(inputs=inputs, spec=spec, fleet=fleet)
    addition = zeros
    windows: list[tuple[int, int]] = []
    if spec.kind == "cloud":
        addition = CLOUD_LOAD_SHARE * truth[inputs.quarter] * cloud_load_shape(inputs=inputs)
    elif spec.kind == "transfer":
        windows = transfer_windows(
            seed=spec.seed, primary_index=spec.primary_index, power=inputs.power
        )
        addition = transfer_addition(windows=windows, power=inputs.power)
    return World(
        spec=spec,
        power=inputs.power - injected + addition,
        injected=injected,
        addition=addition,
        truth=truth,
        fleet=fleet,
        windows=windows,
    )


def check_injection(*, inputs: PrimaryInputs, world: World) -> dict[str, object]:
    """Gate G2 for one world: the injection equals its formula, is nonzero, and is zero at night.

    The formula is recomputed by a separate loop over the quarters. The check is skipped for the
    formula (but not for the nonzero and zero-at-night parts) in a real-farm world, whose output
    is not the physics formula alone.

    Args:
        inputs: The Primary's inputs.
        world: The built world.

    Returns:
        A record with `passed`, and the worst share of bright half-hours with nonzero output.
    """
    if world.fleet is None:
        return {"passed": bool(np.all(world.injected == 0.0)), "worst_nonzero_share": None}
    unit = unit_term(inputs=inputs, fleet=world.fleet)
    formula_ok = True
    if world.spec.kind != "farm":
        expected = np.zeros(GRID_LENGTH)
        for q in range(N_QUARTERS):
            pick = inputs.quarter == q
            expected[pick] = world.truth[q] * unit[pick]
        formula_ok = bool(np.allclose(world.injected, expected, rtol=G2_RTOL, atol=0.0))
    bright = inputs.sky.ghi_w_m2 > G2_BRIGHT_W_M2
    shares = []
    for q in range(N_QUARTERS):
        pick = bright & (inputs.quarter == q)
        if world.truth[q] > 0 and pick.any():
            shares.append(float((world.injected[pick] > 0).mean()))
    worst = min(shares) if shares else 1.0
    night_zero = bool(np.all(world.injected[~inputs.sun_up] == 0.0))
    return {
        "passed": formula_ok
        and night_zero
        and worst >= G2_MIN_NONZERO_SHARE
        and bool(world.injected.any()),
        "formula_ok": formula_ok,
        "night_zero": night_zero,
        "worst_nonzero_share": worst,
    }


def world_arrays(*, world: World) -> dict[str, np.ndarray]:
    """Return the arrays of a world file.

    Args:
        world: The world.

    Returns:
        `injected`, `addition`, `truth`, and the fleet's parameters and the windows.
    """
    fleet = world.fleet
    shape = np.array([*fleet.mix, fleet.tilt_deg, fleet.dc_ac_ratio]) if fleet else np.zeros(5)
    return {
        "injected": world.injected,
        "addition": world.addition,
        "truth": world.truth,
        "fleet": shape,
        "windows": np.array(world.windows, dtype=np.int64).reshape(-1, 2),
    }

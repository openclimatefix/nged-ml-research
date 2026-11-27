# ruff: noqa: E501
"""Experiment 718, WP5b: the known-answer worlds WS and WN, their references, and the checks of the
weather path (design section 10.2).

World WS replaces the power with `y = 0.5 + c[p, era, regime of the block] + noise`, where the
regime is a synthetic weather level written into six of the seven weather columns (every column
except the 2 m temperature), so that only a past block of the same regime reveals the target's
level. World WN is experiment 713's world W0 with the same regimes, with `c` equal to zero, and
with a per-block level `e` that nothing at the origin reveals. This module builds the arrays and
the NumPy references. The fits are in `e00718_known_answer.py`.

The 21 date-time columns are the columns 14 to 34 of experiment 397's summary scheme. By name
they are `cos_zenith`, `sin_azimuth`, `cos_azimuth`, `sin_hour`, `cos_hour`, `sin_hour_2`,
`cos_hour_2`, `dow_1` to `dow_7`, `covid`, `christmas_inside`, `christmas_days`,
`christmas_tent`, `bank_holiday`, `nearest_days`, and `nearest_inside`. None of them increases
with the year, but `covid` is a flag that is 1 in a fixed span of 2020 and 2021 and tells the
network which era a training day lies in. The worlds set `covid` to zero, and `trending_columns`
names any column whose correlation with the year exceeds 0.3, so that a run on real data can show
the list.
"""  # noqa: D205

from dataclasses import dataclass
from typing import Final

import e00713_known_answer as kn  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00718_retrieval as ret
import e00718_tables as tab
import numpy as np
import torch  # ty: ignore[unresolved-import]

PRIMARIES: Final = kn.PRIMARIES
N_REGIMES: Final = 32
REGIME_CENTRE: Final = 15.5
TEMPERATURE_COLUMN: Final = 2
LEVEL_SD: Final = 0.10
NOISE_SD: Final = 0.01
E_SD: Final = 0.05
ERA_DAYS: Final = 730
LAST_ERA_DAYS: Final = 400
HELD_OUT_DAYS: Final = 60
DATETIME_NAMES: Final = (
    "cos_zenith", "sin_azimuth", "cos_azimuth", "sin_hour", "cos_hour", "sin_hour_2", "cos_hour_2",
    "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6", "dow_7", "covid", "christmas_inside",
    "christmas_days", "christmas_tent", "bank_holiday", "nearest_days", "nearest_inside",
)  # fmt: skip
ZEROED_COLUMNS: Final = ("covid",)
GAP_MINIMUM: Final = 0.010
SEED: Final = 0


def datetime_names_match_prep() -> bool:
    """Return whether the 21 names above are the names of columns 14 to 34 of experiment 397's summary scheme."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    names, _, _ = prep422.feature_names(age_inputs=False)
    return tuple(names[14:35]) == DATETIME_NAMES


def trending_columns(
    *, dense_datetime: torch.Tensor, year: np.ndarray, threshold: float = 0.3
) -> list[str]:
    """Return the names of date-time columns whose correlation with `year` (per grid index) exceeds
    `threshold` in absolute value.
    """  # noqa: D205
    x = dense_datetime[:, :, :].float().mean(dim=0).cpu().numpy()
    out = []
    for k, name in enumerate(DATETIME_NAMES):
        col = x[:, k]
        ok = np.isfinite(col)
        if (
            ok.sum() > 2
            and np.std(col[ok]) > 0
            and abs(np.corrcoef(col[ok], year[ok])[0, 1]) > threshold
        ):
            out.append(name)
    return out


def neutralised_datetime(
    *, dense_datetime: torch.Tensor, extra: list[str] | None = None
) -> torch.Tensor:
    """Return a copy of the date-time array with the columns of `ZEROED_COLUMNS` and `extra` set to
    zero (the standardised mean).
    """  # noqa: D205
    out = dense_datetime.clone()
    for name in (*ZEROED_COLUMNS, *(extra or [])):
        out[..., DATETIME_NAMES.index(name)] = 0.0
    return out


def block_of(*, t: np.ndarray) -> np.ndarray:
    """Return the block (24 hours from 12:00 UTC) of each grid index, `(t + 24) // 48`."""
    return (np.asarray(t) + 24) // lay713.STEPS_PER_DAY


@dataclass
class WorldArrays:
    """The arrays of one world (shares of capacity and standardised weather)."""

    kind: str
    y: np.ndarray  # (16, G), NaN outside PRIMARIES
    regime: np.ndarray  # (16, n_blocks) int
    level: np.ndarray  # (16, n_blocks) the block's level above the base (c for WS, e for WN)
    weather: torch.Tensor  # (16, G, 21) float16 with the regimes written in
    datetime: torch.Tensor
    boundary: int  # first grid index of the last era (WS)
    cutoff: int  # the fold's cutoff grid index


def draw_regimes(*, n_blocks: int, seed: int = SEED) -> np.ndarray:
    """Return (16, n_blocks) regimes, one of 32 levels, drawn independently and uniformly for each
    block and Primary.
    """  # noqa: D205
    return np.random.default_rng([seed, 11]).integers(
        0, N_REGIMES, size=(mdl.N_PRIMARIES, n_blocks)
    )


def write_regimes(*, weather: torch.Tensor, regime: np.ndarray) -> torch.Tensor:
    """Return `weather` with six of the seven weather columns set to `regime - 15.5` for every
    half-hour of the block; temperature and the B1 columns stay real.
    """  # noqa: D205
    out = weather.clone()
    n_grid = weather.shape[1]
    q = torch.from_numpy(
        regime[:, block_of(t=np.arange(n_grid))].astype(np.float32) - REGIME_CENTRE
    ).half()
    for col in range(mdl.N_WEATHER):
        if col != TEMPERATURE_COLUMN:
            out[:, :, col] = q
    return out


def era_of(*, n_grid: int, boundary: int) -> np.ndarray:
    """Return the era of each grid index: 0 from `boundary` on (the last era), then 1, 2, ...
    backward in blocks of 730 days.
    """  # noqa: D205
    t = np.arange(n_grid)
    return np.where(t >= boundary, 0, (boundary - 1 - t) // (ERA_DAYS * lay713.STEPS_PER_DAY) + 1)


def build_ws(
    *, template: tr.bt.Source, cutoff: int, seed: int = SEED, last_era_days: int = LAST_ERA_DAYS
) -> WorldArrays:  # ty: ignore[unresolved-attribute]
    """Return world WS: `y = 0.5 + c[p, era, regime] + noise`, with the last era starting 400 days before `cutoff`."""
    n_grid = template.n_grid
    n_blocks = int(block_of(t=np.array([n_grid - 1]))[0]) + 1
    regime = draw_regimes(n_blocks=n_blocks, seed=seed)
    boundary = cutoff - last_era_days * lay713.STEPS_PER_DAY
    era = era_of(n_grid=n_grid, boundary=boundary)
    rng = np.random.default_rng([seed, 12])
    table = rng.normal(0.0, LEVEL_SD, size=(mdl.N_PRIMARIES, int(era.max()) + 1, N_REGIMES))
    y = np.full((mdl.N_PRIMARIES, n_grid), np.nan)
    level = np.zeros((mdl.N_PRIMARIES, n_blocks))
    blocks = block_of(t=np.arange(n_grid))
    starts = np.clip(lay713.STEPS_PER_DAY * np.arange(n_blocks) - 24, 0, n_grid - 1)
    for p in PRIMARIES:
        c = table[p, era, regime[p, blocks]]
        y[p] = 0.5 + c + rng.normal(0.0, NOISE_SD, size=n_grid)
        level[p] = table[p, era[starts], regime[p]]
    return WorldArrays(
        kind="WS", y=y, regime=regime, level=level, weather=write_regimes(weather=template.dense_weather, regime=regime),
        datetime=neutralised_datetime(dense_datetime=template.dense_datetime), boundary=boundary, cutoff=cutoff,
    )  # fmt: skip


def build_wn(
    *,
    template: tr.bt.Source,
    temperature: np.ndarray,
    half_hour: np.ndarray,
    cutoff: int,
    seed: int = SEED,
) -> WorldArrays:  # ty: ignore[unresolved-attribute]
    """Return world WN: experiment 713's `W0`, the same regimes in the weather columns, `c = 0`, and
    a block level `e` (sd 0.05) that nothing at the origin reveals.
    """  # noqa: D205
    n_grid = template.n_grid
    n_blocks = int(block_of(t=np.array([n_grid - 1]))[0]) + 1
    regime = draw_regimes(n_blocks=n_blocks, seed=seed)
    syn = kn.build_world(
        kind="W0", temperature=temperature, day_type=template.day_type.cpu().numpy(), half_hour=half_hour,
        primaries=PRIMARIES, seed=seed,
    )  # fmt: skip
    rng = np.random.default_rng([seed, 13])
    e = rng.normal(0.0, E_SD, size=(mdl.N_PRIMARIES, n_blocks))
    y = syn.y.copy()
    blocks = block_of(t=np.arange(n_grid))
    for p in PRIMARIES:
        y[p] = y[p] + e[p, blocks]
    return WorldArrays(
        kind="WN", y=y, regime=regime, level=e, weather=write_regimes(weather=template.dense_weather, regime=regime),
        datetime=neutralised_datetime(dense_datetime=template.dense_datetime), boundary=cutoff, cutoff=cutoff,
    )  # fmt: skip


def to_world(
    *, arrays: WorldArrays, template: tr.bt.Source, device: str | torch.device
) -> tuple[tr.World, np.ndarray, np.ndarray]:  # ty: ignore[unresolved-attribute]
    """Return the training `World` of a world: its targets end at the last era's start for WS (the
    era is never a training target), and at the cutoff for WN.
    """  # noqa: D205
    shell = tr.bt.Source(  # ty: ignore[unresolved-attribute]
        power_z=template.power_z, dense_weather=arrays.weather.to(device), dense_datetime=arrays.datetime.to(device),
        day_type=template.day_type, ens=None,
    )  # fmt: skip
    cutoff = arrays.boundary + tr.CUTOFF_AGE_STEPS if arrays.kind == "WS" else arrays.cutoff
    syn = kn.Synthetic(
        y=arrays.y,
        shift=np.zeros_like(arrays.y),
        step_start=np.full(arrays.y.shape, -1, dtype=np.int64),
        kind=arrays.kind,
    )
    world, mean, std = kn.to_world(syn=syn, template=shell, cutoff_idx=cutoff, device=device)
    return world, mean, std


def held_out_samples(*, arrays: WorldArrays) -> tr.SampleSet:
    """Return the held-out origins: the 60 days from the fold's cutoff, for the world's Primaries,
    with all 14 lead days read at prediction.
    """  # noqa: D205
    days = np.arange(
        arrays.cutoff // lay713.STEPS_PER_DAY, arrays.cutoff // lay713.STEPS_PER_DAY + HELD_OUT_DAYS
    )
    prim, c = np.meshgrid(np.array(PRIMARIES), days * lay713.STEPS_PER_DAY + 24, indexing="ij")
    return tr.SampleSet(
        primary=prim.ravel(), c_idx=c.ravel(), is_val=np.zeros(prim.size, dtype=bool)
    )


def held_out_floor(*, arrays: WorldArrays) -> float:
    """Return the mean absolute `e` over the held-out targets: each (origin day, lead day) pair
    counts once, so a block counts as often as pairs reach it.

    The origin on day `d` has contract index `48 d + 24`, whose lead-day-`L` target is block `d +
    1 + L`.
    """  # noqa: D205
    first = arrays.cutoff // lay713.STEPS_PER_DAY
    weights = np.zeros(first + HELD_OUT_DAYS + lay713.N_LEAD_DAYS + 1)
    for d in range(first, first + HELD_OUT_DAYS):
        weights[d + 1 : d + 1 + lay713.N_LEAD_DAYS] += 1.0
    blocks = np.flatnonzero(weights)
    level = np.abs(arrays.level[np.array(PRIMARIES)][:, blocks])
    return float(
        (level * weights[blocks][None, :]).sum() / (weights[blocks].sum() * level.shape[0])
    )


# ------------------------------------------------------------------------------------ references


def reference_gap(
    *, arrays: WorldArrays, rows_a: np.ndarray, rows_b: np.ndarray, samples: tr.SampleSet, lead_days: range = range(lay713.N_LEAD_DAYS),
) -> tuple[float, float]:  # fmt: skip
    """Return the reference MAE of two row sets, and so their gap (`mae_a - mae_b`) is `a - b`.

    `rows_a` and `rows_b` are (S, 14, n_rows) age arrays for the held-out samples and lead days,
    with -1 for an absent row. For a target block the reference estimates the level as the mean,
    over the rows whose regime equals the target's, of each row's mean power minus 0.5, with
    weights `max(0, 1 - age / 730)`, and estimates zero where no row matches. Its MAE is taken
    against the target block's true level `y - 0.5` over every half-hour of the 14 target blocks.
    """
    maes = []
    for rows in (rows_a, rows_b):
        errs = []
        for s in range(len(samples.primary)):
            p, c = int(samples.primary[s]), int(samples.c_idx[s])
            for lead in lead_days:
                top = c + lay713.STEPS_PER_DAY * lead
                target = arrays.y[p, top : top + lay713.STEPS_PER_DAY]
                if not np.isfinite(target).all():
                    continue
                q_t = arrays.regime[p, int(block_of(t=np.array([top]))[0])]
                num = den = 0.0
                for age in rows[s, lead]:
                    if age < 0:
                        continue
                    lo = c - lay713.STEPS_PER_DAY * int(age)
                    if arrays.regime[p, int(block_of(t=np.array([lo]))[0])] != q_t:
                        continue
                    w = max(0.0, 1.0 - age / ERA_DAYS)
                    num += w * (
                        float(
                            np.nanmean(
                                arrays.y[
                                    p,
                                    lo : min(
                                        lo + lay713.STEPS_PER_DAY,
                                        c - ret.NEWEST_VISIBLE_BEFORE_CONTRACT + 1,
                                    ),
                                ]
                            )
                        )
                        - 0.5
                    )
                    den += w
                est = num / den if den > 0 else 0.0
                errs.append(np.abs(est - (target - 0.5)))
        maes.append(float(np.concatenate(errs).mean()) if errs else float("nan"))
    return maes[0], maes[1]


def world_tables(
    *,
    arm: str,
    template: tr.bt.Source,
    arrays: WorldArrays,
    days: np.ndarray,
    device: str | torch.device = "cpu",
) -> ret.RowTable:  # ty: ignore[unresolved-attribute]
    """Return the stage 1 (`era5`) row table of `arm` in a world: the pool reads the world's regimes
    and `y` as the finite flags.
    """  # noqa: D205
    z = torch.from_numpy(np.where(np.isfinite(arrays.y), 0.0, np.nan).astype(np.float32)).to(device)
    src = tr.bt.Source(
        power_z=z,
        dense_weather=arrays.weather.to(device),
        dense_datetime=arrays.datetime.to(device),
        day_type=template.day_type,
        ens=None,
    )  # ty: ignore[unresolved-attribute]
    pool = tab.pool_arrays_from_source(source=src, with_calendar=False)
    return ret.build_table(
        arm=arm, arrays=pool, primaries=np.array(PRIMARIES), days=days, target_provider=tab.era5_provider(arrays=pool),
    )  # fmt: skip

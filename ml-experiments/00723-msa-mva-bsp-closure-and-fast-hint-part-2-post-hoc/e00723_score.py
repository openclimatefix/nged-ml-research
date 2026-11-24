# ruff: noqa: E501
"""Experiment 723: building and scoring one world, and the pooled statistics of job W2.

A world is one host's signed series (with a farm's output injected, or un-injected), the folded
reading that a magnitude meter would show, and four sensors: the farm sensor (which binds), the
meter sensor, the CAMS sensor, and the farm sensor shifted by 28 days. `score_world` returns the
stretch table of the world on the later span.

**The injected farm never enters its own world's sensor.** The sensor farms of a world are the
host's sensor membership less the injected farm. The injected farm's series (experiment 720's
`farm_norm`; a gap or dropout is NaN) goes into the host's signed series and into the meter
sensor's residual, because the BSP meter also receives it.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import e00722_core as core722  # ty: ignore[unresolved-import]
import e00723_common as common
import e00723_core as core
import e00723_world as world
import numpy as np
import polars as pl

LEVEL_REFERENCE_INSTANT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""The injection level's reference rows lie before this instant."""

UNINJECTED_FALSE_EXPORT_HOSTS: Final[tuple[str, ...]] = ("D5", "D10")
"""The un-injected hosts of the false-export bar."""

TRANSFER_HOSTS: Final[tuple[str, ...]] = ("D9", "D11", "D14", "D16")
"""The un-injected hosts of the transfer check."""


@dataclass(frozen=True)
class WorldSpec:
    """One world of job W2.

    Attributes:
        name: A unique name that holds no generator value.
        role: `injected`, `uninjected`, `d10_folded`, `d15_folded`, or `real` (a Primary's own
            reading, not folded, which job W3 uses for D12 and D3).
        host: The host Primary's label.
        farm: The injected farm's label (`F1` to `F6`), or None.
        level: The injection level, or None.
    """

    name: str
    role: str
    host: str
    farm: str | None
    level: float | None


def world_specs(*, qualifying: Sequence[tuple[str, str]]) -> list[WorldSpec]:
    """Return every world of job W2 in a fixed order: injected, un-injected, D10 folded, D15 folded.

    With 12 qualifying pairs and 2 levels this gives 24 + 6 + 1 + 1 = 32 worlds, the count of the
    timing gate. D10 appears twice, as un-injected D10 (false-export bar) and as D10 folded
    (recall bar); the world is the same and scored twice.
    """
    specs = [
        WorldSpec(f"inj_{host}_{farm}_{level}", "injected", host, farm, level)
        for host, farm in qualifying
        for level in core.LEVELS
    ]
    specs += [WorldSpec(f"unin_{h}", "uninjected", h, None, None) for h in common.SIGNED_SIX]
    specs += [WorldSpec("d10_folded", "d10_folded", "D10", None, None)]
    specs += [WorldSpec("d15_folded", "d15_folded", "D15", None, None)]
    return specs


@dataclass(frozen=True)
class SkippedWorld:
    """A world that job W2 did not score, with the reason and the evidence for it.

    Attributes:
        name: The world's name.
        reason: `too_few_sensor_farms` (fewer than 2 farms remain in the sensor) or
            `injection_scale_not_finite` (the host has too few reference rows with a finite value
            of the farm's injected series).
        n_reference_rows: Reference rows of the injection level with a finite host value and farm
            output; 0 for an un-injected world.
    """

    name: str
    reason: str
    n_reference_rows: int


def expected_to_score(*, spec: WorldSpec, membership: Mapping[str, Sequence[str]]) -> bool:
    """Return True when the world has a sensor of at least 2 farms once the injected farm is removed."""
    return len([f for f in membership.get(spec.host, ()) if f != spec.farm]) >= 2


@dataclass
class WorldScore:
    """The result of scoring one world on the later span.

    Attributes:
        spec: The world.
        table: The stretch table (columns of `world.STRETCH_SCHEMA`).
        counts: R1's counts at each filter stage under the farm sensor.
        diagnostics: R3's diagnostics for each sensor.
        lag: The lag at which the reading's and the farm sensor's fast changes correlate most on
            anchor stretches, and that correlation.
        scale: The injection scale (1.0 for an un-injected world).
        seconds: Core-seconds the world took.
    """

    spec: WorldSpec
    table: pl.DataFrame
    counts: dict[str, int]
    diagnostics: dict[str, dict[str, float]]
    lag: tuple[int, float]
    scale: float
    seconds: float


def level_rows(*, ctx: common.Context, host: str) -> np.ndarray:
    """Return the host's reference rows for the injection level (cached)."""
    key = ("level_rows", host)
    if key not in ctx.cache:
        ctx.cache[key] = world.level_reference_rows(
            k=ctx.k(host=host),
            month=world.month_of_rows(n=core.N_GRID),
            hour=world.utc_hour_of_rows(n=core.N_GRID),
            before_row=world.row_of_instant(instant=LEVEL_REFERENCE_INSTANT),
        )
    return ctx.cache[key]


def windows_mask(*, windows: Mapping[str, Sequence[tuple[int, int]]], host: str) -> np.ndarray:
    """Return the transfer mask of a host (all False when no window exists for it)."""
    return world.window_mask(windows=windows.get(host, ()), n=core.N_GRID)


def score_world(
    *,
    ctx: common.Context,
    spec: WorldSpec,
    membership: Mapping[str, Sequence[str]],
    windows: Mapping[str, Sequence[tuple[int, int]]],
    skips: list[SkippedWorld] | None = None,
) -> WorldScore | None:
    """Build and score one world; return None, and append the reason to `skips`, when the world cannot be scored.

    Args:
        ctx: The inputs.
        spec: The world.
        membership: The farms of each host's sensor, from job W0.
        windows: Each host's switching-log windows in grid rows, from job W0.
        skips: Receives a `SkippedWorld` for a world that is not scored, so that no world drops
            out of the pool silently.

    Returns:
        The score, or None when fewer than 2 farms remain in the sensor or the injection scale is
        not finite.
    """
    t0 = time.process_time()
    host = spec.host
    cap = ctx.capacity(host=host)
    farms = [f for f in membership.get(host, ()) if f != spec.farm]
    if len(farms) < 2:
        if skips is not None:
            skips.append(SkippedWorld(spec.name, "too_few_sensor_farms", 0))
        return None
    signed = ctx.series[host].copy()
    residual = ctx.residual(host=host)
    scale = 1.0
    if spec.farm is not None and spec.level is not None:
        out = ctx.farm_norm(farm=spec.farm)
        rows = level_rows(ctx=ctx, host=host)
        scale = world.injection_scale(
            host_signed_mw=ctx.series[host], farm_norm_series=out, rows=rows, level=spec.level
        )
        if not np.isfinite(scale):
            if skips is not None:
                n_ref = world.reference_row_count(
                    host_signed_mw=ctx.series[host], farm_norm_series=out, rows=rows
                )
                skips.append(SkippedWorld(spec.name, "injection_scale_not_finite", n_ref))
            return None
        signed = world.inject(signed_mw=signed, farm_norm_series=out, scale=scale)
        residual = residual - scale * out
    reading = signed if spec.role == "real" else world.fold(signed_mw=signed, capacity_mw=cap)
    start = ctx.start_row
    lit = ctx.lit(host=host)
    masked = windows_mask(windows=windows, host=host)
    dm = world.reading_fast_change(
        reading_mw=reading, capacity_mw=cap, lit=lit, key=ctx.eng.key,
        overcast_row=ctx.overcast_row(host=host), start=start,
    )  # fmt: skip
    farm = ctx.sensor(host=host, farms=farms)
    full = {
        "farm": farm,
        "meter": world.meter_sensor(residual=residual, lit=lit, farm_dk=farm.dk),
        "cams": world.cams_sensor(s=ctx.sky_of[host], lit=lit),
        "shifted": world.shifted_sensor(sensor=farm),
    }
    sensors = {k: v.from_row(start) for k, v in full.items()}
    stretches = world.stretches_of(
        reading_mw=reading[start:], signed_mw=signed[start:], capacity_mw=cap, lit=lit[start:]
    )
    table = world.stretch_table(
        stretches=stretches, signed_mw=signed[start:], dm=dm, sensors=sensors, masked=masked[start:]
    )
    counts = world.stage_counts(
        reading_mw=reading[start:], signed_mw=signed[start:], lit=lit[start:], masked=masked[start:],
        dm=dm, sensor=sensors["farm"], table=table, n_days=ctx.n_days,
    )  # fmt: skip
    diagnostics = {
        k: world.threshold_diagnostics(dm=dm, sensor=s, masked=masked[start:])
        for k, s in sensors.items()
    }
    lag = world.peak_lag_on_anchors(dm=dm, dz=sensors["farm"].dz, stretches=stretches)
    return WorldScore(spec, table, counts, diagnostics, lag, scale, time.process_time() - t0)


# ---------------------------------------------------------------------------------------------
# Pooled statistics
# ---------------------------------------------------------------------------------------------
def tagged(
    *,
    scores: Sequence[WorldScore],
    sensor: str,
    roles: Sequence[str],
    population_sensor: str | None = None,
) -> pl.DataFrame:
    """Return the scorable export and import stretches of the worlds of the given roles, tagged with their world.

    A stretch is scorable when it is post-bounce, on a one-stretch day, and holds at least 8 usable
    pairs under `population_sensor` (default: `sensor`). Mixed stretches are excluded. The columns
    are `world`, `host`, `farm`, `level`, `day`, `c`, `n_pairs`, and `is_export`.

    RULED (the placebo): the shifted sensor is read on the stretches that are scorable under the
    farm sensor, so pass `sensor="shifted"` and `population_sensor="farm"`. A stretch on which the
    shifted sensor has no finite pair scores `c = 0`, a stretch with no evidence, and shows
    `n_pairs = 0`.
    """
    frames = []
    pop = sensor if population_sensor is None else population_sensor
    for s in scores:
        if s.spec.role not in roles:
            continue
        t = (
            world.scorable(table=s.table, sensor=pop)
            .filter(pl.col("truth") != "mixed")
            .with_columns(pl.col(f"c_{sensor}").fill_nan(0.0).fill_null(0.0))
        )
        frames.append(
            t.select(
                pl.lit(s.spec.name).alias("world"),
                pl.lit(s.spec.host).alias("host"),
                pl.lit(s.spec.farm, dtype=pl.String).alias("farm"),
                pl.lit(s.spec.level, dtype=pl.Float64).alias("level"),
                "day",
                pl.col(f"c_{sensor}").alias("c"),
                pl.col(f"n_{sensor}").alias("n_pairs"),
                (pl.col("truth") == "export").alias("is_export"),
            )
        )
    schema = {
        "world": pl.String, "host": pl.String, "farm": pl.String, "level": pl.Float64,
        "day": pl.Int64, "c": pl.Float64, "n_pairs": pl.Int64, "is_export": pl.Boolean,
    }  # fmt: skip
    return pl.concat(frames) if frames else pl.DataFrame(schema=schema)


def area_of(*, rows: pl.DataFrame, weights: np.ndarray | None) -> dict[str, float]:
    """Return the pooled area of `C` for export against import stretches, with a bootstrap if `weights` is given.

    The keys are `area`, `se`, `lo`, `hi`, `n_export`, `n_import`, `n_days`, and `n_pairs`.
    """
    n_export = int(rows.filter(pl.col("is_export")).height)
    n_import = rows.height - n_export
    base = {
        "n_export": n_export,
        "n_import": n_import,
        "n_days": int(rows["day"].n_unique()) if rows.height else 0,
        "n_pairs": int(rows["n_pairs"].sum()) if rows.height else 0,
    }
    if rows.height == 0 or n_export == 0 or n_import == 0:
        return {
            "area": float("nan"),
            "se": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            **base,
        }
    score, label, day = rows["c"].to_numpy(), rows["is_export"].to_numpy(), rows["day"].to_numpy()
    if weights is None:
        point = core722.weighted_auc(score=score, label=label.astype(int), w=np.ones(len(score)))
        return {"area": point, "se": float("nan"), "lo": float("nan"), "hi": float("nan"), **base}
    point, se, lo, hi = core.bootstrap_area(score=score, is_export=label, day=day, weights=weights)
    return {"area": point, "se": se, "lo": lo, "hi": hi, **base}


def time_draws(
    *, rows: pl.DataFrame, weights: np.ndarray, n: int = 10, replicate: int = 1
) -> float:
    """Return the core-seconds per bootstrap draw, timed on the first `n` draws of the rows given.

    `replicate` repeats the rows (with the same days) so that a world's few stretches stand in for
    the pooled population, which holds about that many times as many stretches. The draw's cost
    grows with the number of stretches, so the timing gate must time a draw at the pooled size.
    """
    if rows.height == 0:
        return 0.0
    big = pl.concat([rows] * max(replicate, 1)) if replicate > 1 else rows
    score, label, day = big["c"].to_numpy(), big["is_export"].to_numpy(), big["day"].to_numpy()
    present, compact = np.unique(day, return_inverse=True)
    sub = core722.subset_weights(weights=weights[:n], present=present)
    t0 = time.process_time()
    for b in range(sub.shape[0]):
        core722.weighted_auc(score=score, label=label.astype(int), w=sub[b][compact])
    return (time.process_time() - t0) / max(sub.shape[0], 1)


def evidence_counts(*, rows: pl.DataFrame) -> dict[str, int]:
    """Return how many stretches give export evidence, import evidence, and none."""
    c = rows["c"].to_numpy()
    return {
        "n": rows.height,
        "n_export_evidence": int((c > core.DEAD_BAND).sum()),
        "n_import_evidence": int((c < -core.DEAD_BAND).sum()),
        "n_no_evidence": int((np.abs(c) <= core.DEAD_BAND).sum()),
    }


def dead_band_population(*, scores: Sequence[WorldScore], sensor: str = "farm") -> dict[str, int]:
    """Return D15's counts: every post-bounce stretch on a one-stretch day, and those that give no evidence.

    The dead-band bar keeps every stretch as its denominator, scorable or not. A stretch with
    fewer than 8 usable pairs gives no evidence. The scorable share is reported beside.
    """
    out = {"n_stretches": 0, "n_no_evidence": 0, "n_scorable": 0, "n_scorable_no_evidence": 0}
    for s in scores:
        if s.spec.role != "d15_folded":
            continue
        t = s.table.filter(pl.col("is_post_bounce") & (pl.col("n_post_bounce_in_day") == 1))
        ev = [
            core.evidence_of(c=c, n_usable=n)
            for c, n in zip(t[f"c_{sensor}"].to_list(), t[f"n_{sensor}"].to_list(), strict=True)
        ]
        scorable_flags = [n >= core.MIN_USABLE_PAIRS for n in t[f"n_{sensor}"].to_list()]
        out["n_stretches"] += t.height
        out["n_no_evidence"] += sum(1 for e in ev if e == "none")
        out["n_scorable"] += sum(scorable_flags)
        out["n_scorable_no_evidence"] += sum(
            1 for e, ok in zip(ev, scorable_flags, strict=True) if ok and e == "none"
        )
    return out


def alignment(*, scores: Sequence[WorldScore]) -> dict[str, Any]:
    """Return the alignment check over the un-injected worlds that hold a farm sensor.

    The check passes when every evaluated host's cross-correlation of the reading's and the
    sensor's fast changes on anchor stretches peaks at lag 0. A host whose correlation could not
    be evaluated (too few rows) is listed and not counted.
    """
    rows = [
        (s.spec.host, s.lag[0], s.lag[1])
        for s in scores
        if s.spec.role in ("uninjected", "d15_folded") and np.isfinite(s.lag[1])
    ]
    return {
        "per_host": {h: lag for h, lag, _ in rows},
        "n_hosts_evaluated": len(rows),
        "ok": bool(rows) and all(lag == 0 for _, lag, _ in rows),
    }

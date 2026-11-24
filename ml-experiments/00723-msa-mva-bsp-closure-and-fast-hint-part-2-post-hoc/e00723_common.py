# ruff: noqa: E501
"""Experiment 723: the loading and the setup that jobs W0, W2, and W3 share.

This module imports experiment 722's loaders (`e00722_data`, `e00722_common`) and never copies
them. Run a job with experiment 722's folder on `PYTHONPATH` (`run.sh` sets it).

**A job refuses to start unless the environment and the earlier jobs allow it.**
`require_w0` raises when job W0's check file is missing, and `require_w2_supported` raises when job
W2's verdict is not `supported`, which is the only case in which job W3 runs.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import e00722_common as common722  # ty: ignore[unresolved-import]
import e00722_core as core722  # ty: ignore[unresolved-import]
import e00722_data as data  # ty: ignore[unresolved-import]
import e00722_sky as sky  # ty: ignore[unresolved-import]
import e00723_core as core
import e00723_guard as guard
import e00723_world as world
import numpy as np
import polars as pl

SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The signed-MW Primaries: the hosts of the injected worlds and of the un-injected worlds."""

SENSOR_HOSTS: Final[tuple[str, ...]] = (*SIGNED_SIX, "D15", "D12", "D3")
"""Every host that job W0 builds a sensor for: the six, D15 (dead-band check), D12 and D3 (job W3)."""

W0_CHECKS_NAME: Final[str] = "w0_checks.json"
"""Job W0's check file."""

W0_PAIRS_NAME: Final[str] = "w0_pairs.csv"
"""Job W0's polygon-filtered pair table, with the powered flags."""

W0_MEMBERSHIP_NAME: Final[str] = "w0_membership.csv"
"""Job W0's sensor membership: one row per host and farm that stays in the host's sensor."""

W0_WINDOWS_NAME: Final[str] = "w0_windows.parquet"
"""Job W0's switching-log windows in grid rows; it lives under `PRIVATE/`."""

W2_VERDICT_NAME: Final[str] = "w2_verdict.json"
"""Job W2's verdict file, which job W3 reads."""

W3_VERDICT_NAME: Final[str] = "w3_verdict.json"
"""Job W3's verdict file."""

ENV_DEFAULTS: Final[dict[str, str]] = common722.ENV_DEFAULTS
"""The thread and GPU settings every job asserts (the launch wrapper sets them)."""


def assert_launch() -> None:
    """Raise unless the thread variables are set and the process is pinned to exactly one core."""
    try:
        common722.assert_environment(env=dict(os.environ))
        common722.assert_one_core()
    except common722.JobRefusedError as exc:
        raise guard.JobRefusedError(str(exc)) from exc


def require_w0(*, out_dir: Path = guard.OUT_DIR) -> dict[str, Any]:
    """Return job W0's check file, or raise when job W0 has not run."""
    return guard.read_json(name=W0_CHECKS_NAME, out_dir=out_dir)


def require_w2_supported(*, out_dir: Path = guard.OUT_DIR) -> dict[str, Any]:
    """Return job W2's verdict file, and raise unless the verdict is `supported`."""
    verdict = guard.read_json(name=W2_VERDICT_NAME, out_dir=out_dir)
    if verdict.get("verdict") != "supported":
        msg = f"job W3 runs only when job W2 is supported; the verdict is {verdict.get('verdict')}"
        raise guard.JobRefusedError(msg)
    return verdict


@dataclass
class Context:
    """The inputs of a job on the master grid, with lazily built per-host arrays.

    Attributes:
        topology: The Primaries, BSPs, and generators.
        classes: Job 1 of experiment 722's generator-class table with ids.
        power: Cleaned power by time series id.
        series: The Primaries' power by label.
        sky_of: CAMS and sun geometry of each Primary by label.
        farm_sky: CAMS and output of each farm by label (`F1` to `F6`).
        farm_cap: Each farm's capacity in megawatts.
        farm_id: Each farm's time series id.
        eng: The engine frame, which supplies the matched key and the quiet mask.
        start_row: First grid row of the later span (the start of the first day with both transformers).
        n_days: Days of the later span.
        inside: The metered generators inside each Primary's polygon, by label.
    """

    topology: data.Topology
    classes: pl.DataFrame
    power: dict[int, np.ndarray]
    series: dict[str, np.ndarray]
    sky_of: dict[str, data.Sky]
    farm_sky: dict[str, data.Sky]
    farm_cap: dict[str, float]
    farm_id: dict[str, int]
    eng: data.EngineFrame
    start_row: int
    n_days: int
    inside: dict[str, set[int]]
    cache: dict[Any, Any] = field(default_factory=dict)

    def lit(self, *, host: str) -> np.ndarray:
        """Return the daylight rows (sun at 10 degrees or more) of a host's CAMS point."""
        return sky.daylight(
            zenith_deg=self.sky_of[host].zenith, min_elevation_deg=core.DAYLIGHT_ELEVATION_DEG
        )

    def k(self, *, host: str) -> np.ndarray:
        """Return the clear-sky index of a host's CAMS point."""
        s = self.sky_of[host]
        return sky.clear_sky_index(ghi=s.ghi, ghi_clear=s.ghi_clear)

    def overcast_row(self, *, host: str) -> np.ndarray:
        """Return True on every row of a day that is overcast at the host's CAMS point."""
        key = ("overcast", host)
        if key not in self.cache:
            classes = sky.classify_days(
                k=self.k(host=host), lit=self.lit(host=host), n_days=world.n_days_of_grid()
            )
            self.cache[key] = np.repeat(classes == sky.CLASS_OVERCAST, core.HALF_HOURS_PER_DAY)[
                : core.N_GRID
            ]
        return self.cache[key]

    def capacity(self, *, host: str) -> float:
        """Return a Primary's capacity in megawatts."""
        return self.topology.capacity(label=host)

    def farms_inside(self, *, host: str) -> list[str]:
        """Return the farms whose metered series lies inside the host's service-area polygon."""
        ids = self.inside.get(host, set())
        return sorted(f for f, ts in self.farm_id.items() if ts in ids)

    def residual(self, *, host: str) -> np.ndarray:
        """Return the residual that leaves `host` out: the BSP meter less every other child plus subtracted generators."""
        bsp = self.topology.bsp_of(label=host)
        labels = core722.BSP_A_PRIMARIES if bsp == "A" else core722.BSP_B_PRIMARIES
        gens = [
            self.power[g]
            for g in common722.generator_ids_to_subtract(
                classes=self.classes, bsp=bsp, variant="primary"
            )
        ]
        transformers = core.TWO_TRANSFORMERS_2025 if bsp == "A" else ()
        start = (
            common722.both_start(series=self.series, transformers=transformers) if bsp == "A" else 0
        )
        return common722.spanned_residual(
            bsp=self.power[self.topology.bsp_meter[bsp]],
            series=self.series,
            children=labels,
            transformers=transformers,
            leave_out=host,
            generators=gens,
            start=start,
        )

    def farm_output(self, *, farm: str) -> np.ndarray:
        """Return a farm's recorded output in megawatts (NaN where missing)."""
        out = self.farm_sky[farm].power
        if out is None:
            msg = f"farm {farm} has no recorded output"
            raise ValueError(msg)
        return out

    def farm_mean_share(self, *, farms: list[str]) -> np.ndarray:
        """Return the sensor series (the mean share of capacity) of the named farms."""
        return world.sensor_series(
            shares=[
                world.farm_share(output_mw=self.farm_output(farm=f), capacity_mw=self.farm_cap[f])
                for f in farms
            ]
        )

    def sensor(self, *, host: str, farms: list[str]) -> world.SensorChange:
        """Return the farm sensor of a host built from the named farms (cached)."""
        key = ("sensor", host, tuple(sorted(farms)))
        if key not in self.cache:
            self.cache[key] = world.farm_sensor(
                z=self.farm_mean_share(farms=farms),
                u_clear=self.sky_of[host].u_clear,
                lit=self.lit(host=host),
            )
        return self.cache[key]


@dataclass
class SkyStage:
    """The sources that hold no Primary's reading: CAMS, the farms' output, the topology, and polygons.

    Attributes:
        topology: The Primaries, BSPs, and generators.
        classes: Job 1 of experiment 722's generator-class table with ids.
        sky_of: CAMS and sun geometry of each Primary by label.
        farm_sky: CAMS and output of each farm by label.
        farm_cap: Each farm's capacity in megawatts.
        farm_id: Each farm's time series id.
        inside: The metered generators inside each Primary's polygon, by label.
    """

    topology: data.Topology
    classes: pl.DataFrame
    sky_of: dict[str, data.Sky]
    farm_sky: dict[str, data.Sky]
    farm_cap: dict[str, float]
    farm_id: dict[str, int]
    inside: dict[str, set[int]]


def load_sky_stage(*, log: guard.ReadLog, inputs: list[Path]) -> SkyStage:
    """Read the sources that hold no Primary's reading, so that the sensor audit can run before any reading is read."""
    inputs += [data.SITE_MAP_PATH, data.METADATA_PATH, data.HIERARCHY_PATH, data.FRAME39_PATH]
    inputs += [
        data.SITES39_PATH,
        data.ENGINE_FRAME_PATH,
        guard.OUT_DIR_722 / common722.CLASSES_NAME,
    ]
    topology = common722.load_topology()
    primaries_sky, farm_sky, farm_cap, farm_id, _ = data.load_sky(log=log)
    label_of = dict(
        zip(topology.primaries["time_series_id"], topology.primaries["label"], strict=True)
    )
    return SkyStage(
        topology=topology,
        classes=common722.read_classes(),
        sky_of={label_of[ts]: s for ts, s in primaries_sky.items() if ts in label_of},
        farm_sky=farm_sky,
        farm_cap=farm_cap,
        farm_id=farm_id,
        inside=data.polygon_membership(topology=topology),
    )


def load_context(
    *, log: guard.ReadLog, inputs: list[Path], stage: SkyStage | None = None
) -> Context:
    """Read every Primary's reading, the BSP meters, and the generators, and return the context.

    Args:
        log: The read log, which gets each source in order.
        inputs: The list that fills the run log with the input paths.
        stage: The sky stage, read first by job W0 so that its audit precedes every reading; read
            here when None.

    Returns:
        The context. The later span's day count is not asserted here; `assert_later_span` does it.
    """
    st = stage if stage is not None else load_sky_stage(log=log, inputs=inputs)
    ids = [
        *st.topology.ids_of(labels=(*core722.BSP_A_PRIMARIES, *core722.BSP_B_PRIMARIES)).values(),
        *st.topology.bsp_meter.values(),
        *st.classes["generator_id"].unique().to_list(),
    ]
    power = data.load_power(ids=ids, log=log, name="primaries_bsp_meters_generators")
    series = common722.label_arrays(topology=st.topology, power=power)
    eng = data.load_engine_frame(log=log)
    first_row = common722.both_start(series=series, transformers=core.TWO_TRANSFORMERS_2025)
    return Context(
        topology=st.topology,
        classes=st.classes,
        power=power,
        series=series,
        sky_of=st.sky_of,
        farm_sky=st.farm_sky,
        farm_cap=st.farm_cap,
        farm_id=st.farm_id,
        eng=eng,
        start_row=world.start_row_of_span(first_row=first_row),
        n_days=world.later_span_days(first_row=first_row),
        inside=st.inside,
    )


def assert_later_span(*, n_days: int) -> bool:
    """Return True when the later span holds exactly 413 days; a job treats False as a code defect."""
    return n_days == core.N_LATER_DAYS

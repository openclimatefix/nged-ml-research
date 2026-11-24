# ruff: noqa: E501
"""Experiment 722: the loaders. The only module that reads real data.

**Every loader puts its source on the master grid of `e00722_core` at the end-of-period stamp.**
`scan_power.time` and the `time` of the frame of experiment 39 are already period-ending, and the
`start` of the engine frame of experiment 697 is converted by adding 30 minutes. No row on or after
the holdout is kept, and the loaders count how many were dropped.

The topology builder `build_topology` is pure (frames in, tables out), so a test can run it on
synthetic frames. Each loader that touches a Primary's reading or a generator appends to the
`ReadLog`, so that the class audit can prove that it finished first.
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00722_core as core
import e00722_guard as guard
import numpy as np
import polars as pl
from studies.dp_calendar import LOCAL_TIME_ZONE, calendar_columns
from studies.power import CLEANED_POWER_DELTA_URI, scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

NGED_DIR: Final[Path] = REPO_DATA_DIR / "NGED"
"""The folder of NGED's metadata and hierarchy files."""

METADATA_PATH: Final[Path] = NGED_DIR / "metadata.parquet"
"""One row per time series: type, units, substation number, and substation type."""

HIERARCHY_PATH: Final[Path] = NGED_DIR / "substation_hierarchy_2026-10-15" / "Substations.csv"
"""Every substation's BSP and GSP."""

SWITCHING_LOG_PATH: Final[Path] = (
    NGED_DIR / "substation_hierarchy_2026-10-15" / "Switching Logs.xlsx"
)
"""The switching logs, read if present."""

EFFECTIVE_CAPACITY_PATH: Final[Path] = REPO_DATA_DIR / "effective_capacity"
"""The Delta table of each series' effective capacity over time."""

ENS_MEMBERS_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/downloads/NWP/ENS_SITE_EXTRACT/ens_members.parquet"
)
"""The stored ENS table, which holds `temp_c` and `lead_hours`."""

SITE_MAP_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free"
    / "site_map.csv"
)
"""The site map of experiment 521: label, time series id, capacity, and group of the 16 Primaries."""

FRAME39_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection/work/frame.parquet"
)
"""Experiment 39's half-hourly CAMS frame of 20 substation series and 6 farms (period-ending)."""

SITES39_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection/work/sites.parquet"
)
"""Experiment 39's site table: label, coordinates, effective capacity, and time series id."""

ENGINE_FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00697-gas-engine-state-tracker-part-4-running-patterns/frames/frame_d8.parquet"
)
"""Experiment 697's engine-state frame (`start` is the start of the half-hour)."""

PV_REGISTER_GLOB: Final[str] = "0071[67]-*/**/pv_register_capacity*.parquet"
"""Where experiments 716 and 717 store the register capacity table (columns `primary` and `capacity_mw`)."""

ENGINE_RECORD_GLOB: Final[str] = "00194-*/g2_window.parquet"
"""Where experiment 194 stored the 44-day independent engine record (private)."""

D8_BSP_STUDY_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "briefs/msa_architecture/d8_bsp_residual"
"""The private folder of the stored BSP run, whose CSV job 1 reads."""

GENERATOR_TYPES: Final[frozenset[str]] = frozenset(
    {
        "BESS", "Biofuel", "CHP", "Energy from Waste", "Geothermal", "Hydro",
        "Mixed (Generation)", "Other (Generation)", "Other (Storage)", "Peaking Plant", "PV", "Wind",
    }
)  # fmt: skip
"""Metadata `time_series_type` values that are metered generation or storage."""

SKY_COLUMNS: Final[tuple[str, ...]] = (
    "time",
    "site",
    "u",
    "u_clear",
    "ghi_u",
    "ghi_u_clear",
    "zenith",
)
"""The columns read for a Primary's CAMS. A Primary's own `power` column is never selected."""


class TopologyError(AssertionError):
    """Raised when the joined topology differs from the one the design asserts."""


@dataclass(frozen=True)
class Topology:
    """The join of Primaries, BSPs, and generators.

    Attributes:
        primaries: One row per Primary: `label`, `time_series_id`, `capacity_mw`, `units`, `bsp`.
        generators: One row per metered generator below a BSP: `time_series_id`, `type`,
            `units`, `bsp`, `substation_number`.
        bsp_meter: The BSP meter's time series id by `A` and `B`.
    """

    primaries: pl.DataFrame
    generators: pl.DataFrame
    bsp_meter: dict[str, int]

    def ids_of(self, *, labels: tuple[str, ...]) -> dict[str, int]:
        """Return the time series id of each Primary label."""
        rows = self.primaries.filter(pl.col("label").is_in(labels))
        return dict(zip(rows["label"].to_list(), rows["time_series_id"].to_list(), strict=True))

    def capacity(self, *, label: str) -> float:
        """Return a Primary's capacity in megawatts."""
        return float(self.primaries.filter(pl.col("label") == label)["capacity_mw"][0])

    def bsp_of(self, *, label: str) -> str:
        """Return `A` or `B`."""
        return str(self.primaries.filter(pl.col("label") == label)["bsp"][0])


def build_topology(
    *, site_map: pl.DataFrame, metadata: pl.DataFrame, hierarchy: pl.DataFrame
) -> Topology:
    """Join each Primary to its BSP, unit, and BSP meter, and list the generators below each BSP.

    Args:
        site_map: Columns `label`, `time_series_id`, `capacity_mw`.
        metadata: Columns `time_series_id`, `time_series_type`, `units`, `substation_number`,
            `substation_type`.
        hierarchy: Columns `substation_number` and `bsp_substation_number`.

    Returns:
        The Primaries, the metered generators below each BSP, and each BSP's meter.

    Raises:
        TopologyError: When a Primary has no BSP or no meter, or the BSP has no unique meter.
    """
    hier = hierarchy.select("substation_number", "bsp_substation_number").unique()
    meta = metadata.select(
        "time_series_id", "time_series_type", "units", "substation_number", "substation_type"
    ).with_columns(pl.col("units").cast(pl.String), pl.col("time_series_type").cast(pl.String))
    prim = (
        site_map.select("label", "time_series_id", "capacity_mw")
        .join(meta, on="time_series_id", how="left", maintain_order="left")
        .join(hier, on="substation_number", how="left", maintain_order="left")
    )
    if prim["bsp_substation_number"].null_count() or prim["units"].null_count():
        msg = "a Primary has no unit or no BSP in the hierarchy table"
        raise TopologyError(msg)
    bsp_numbers: dict[str, int] = {}
    for letter, anchor in (("A", "D3"), ("B", "D2")):
        bsp_numbers[letter] = int(
            prim.filter(pl.col("label") == anchor)["bsp_substation_number"][0]
        )
    prim = prim.with_columns(
        bsp=pl.col("bsp_substation_number").replace_strict(
            {v: k for k, v in bsp_numbers.items()}, default="other", return_dtype=pl.String
        )
    )
    meters: dict[str, int] = {}
    for letter, number in bsp_numbers.items():
        cand = meta.filter(
            (pl.col("substation_type") == "BSP")
            & (pl.col("substation_number") == number)
            & (pl.col("units") == "MW")
        )
        raw = cand.filter(pl.col("time_series_type") == "Raw Flow")
        pick = raw if raw.height == 1 else cand
        if pick.height != 1:
            msg = f"BSP {letter} has {pick.height} candidate MW meters, not one"
            raise TopologyError(msg)
        meters[letter] = int(pick["time_series_id"][0])
    gens = (
        meta.filter(pl.col("time_series_type").is_in(sorted(GENERATOR_TYPES)))
        .join(hier, on="substation_number", how="inner")
        .with_columns(
            bsp=pl.col("bsp_substation_number").replace_strict(
                {v: k for k, v in bsp_numbers.items()}, default="other", return_dtype=pl.String
            )
        )
        .filter(pl.col("bsp") != "other")
        .select(
            "time_series_id",
            pl.col("time_series_type").alias("type"),
            "units",
            "bsp",
            "substation_number",
        )
        .sort("bsp", "time_series_id")
    )
    primaries = prim.select("label", "time_series_id", "capacity_mw", "units", "bsp").sort("label")
    return Topology(primaries=primaries, generators=gens, bsp_meter=meters)


def assert_topology(*, topology: Topology) -> dict[str, bool]:
    """Check 1 of job 1: BSP A holds the nine Primaries and BSP B the seven the design names.

    Args:
        topology: The joined topology.

    Raises:
        TopologyError: On any mismatch, which stops the slice.

    Returns:
        Whether each BSP's generator count matches the design's (8 under A, 5 under B), which is
        reported and does not stop the slice.
    """
    for letter, labels in (("A", core.BSP_A_PRIMARIES), ("B", core.BSP_B_PRIMARIES)):
        held = set(topology.primaries.filter(pl.col("bsp") == letter)["label"].to_list())
        if held != set(labels):
            msg = f"BSP {letter} holds {sorted(held)}, not {sorted(labels)}"
            raise TopologyError(msg)
    unsigned = set(topology.primaries.filter(pl.col("units") == "MVA")["label"].to_list())
    expected = set(core.BSP_A_UNSIGNED) | set(core.BSP_B_UNSIGNED)
    if not expected <= unsigned:
        msg = f"MVA-metered Primaries {sorted(unsigned)} lack {sorted(expected - unsigned)}"
        raise TopologyError(msg)
    counts = topology.generators.group_by("bsp").len()
    n = dict(zip(counts["bsp"].to_list(), counts["len"].to_list(), strict=True))
    return {
        "generators_under_a_are_8": n.get("A") == 8,
        "generators_under_b_are_5": n.get("B") == 5,
    }


# ---------------------------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------------------------
def grid_index(*, end_time: pl.Expr) -> pl.Expr:
    """Return the expression of the master-grid index of a period-ending stamp."""
    return ((end_time - pl.lit(core.GRID_FIRST_END)).dt.total_minutes() // 30).alias("i")


def to_grid(*, index: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place values on the master grid; rows outside the grid are ignored."""
    out = np.full(core.N_GRID, np.nan)
    ok = (index >= 0) & (index < core.N_GRID)
    out[index[ok]] = values[ok]
    return out


def load_power(*, ids: list[int], log: guard.ReadLog, name: str) -> dict[int, np.ndarray]:
    """Return each series' cleaned power on the master grid (NaN where absent).

    Rows at or after the holdout are not read. `name` goes into the read log, and the log gets the
    count of rows on or after the holdout that this read returned (which must be 0) and the count
    of those rows among the requested series that the filter dropped.
    """
    log.record(name=name, kind="primary")
    frame = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(ids) & (pl.col("time") < pl.lit(core.HOLDOUT)))
        .with_columns(grid_index(end_time=pl.col("time")))
        .select("time", "time_series_id", "i", "power")
        .collect()
    )
    log.record_holdout(
        name=name,
        read_after=int(frame.filter(pl.col("time") >= pl.lit(core.HOLDOUT)).height),
        dropped=count_power_after_holdout(ids=ids),
    )
    out: dict[int, np.ndarray] = {}
    for ts in ids:
        sub = frame.filter(pl.col("time_series_id") == ts)
        out[ts] = to_grid(index=sub["i"].to_numpy(), values=sub["power"].to_numpy().astype(float))
    return out


def count_power_after_holdout(*, ids: list[int]) -> int:
    """Return the number of power rows of the requested series whose end stamp is on or after the holdout."""
    return int(
        scan_power()
        .filter(pl.col("time_series_id").is_in(ids) & (pl.col("time") >= pl.lit(core.HOLDOUT)))
        .select(pl.len())
        .collect()
        .item()
    )


def power_table_version() -> str:
    """Return the Delta version of the cleaned power table, or `unavailable` if it cannot be read."""
    try:
        from deltalake import DeltaTable

        return str(DeltaTable(CLEANED_POWER_DELTA_URI).version())
    except Exception:  # noqa: BLE001
        return "unavailable"


def calendar_key() -> np.ndarray:
    """Return the matched key of every master-grid half-hour from the calendar alone.

    The key is the local half-hour, the local month, and the day type (weekend or bank holiday
    against other days), the same key that the engine frame holds on its own rows. The engine
    record lies outside the engine frame, so a key that exists only on the frame's rows would make
    the engine test unreadable.
    """
    end_times = pl.Series(
        "t",
        pl.datetime_range(
            core.GRID_FIRST_END,
            core.GRID_FIRST_END + timedelta(minutes=30 * (core.N_GRID - 1)),
            interval="30m",
            time_zone="UTC",
            eager=True,
        ),
    )
    cal = calendar_columns(end_times=end_times)
    month = (
        end_times.dt.offset_by("-30m").dt.convert_time_zone(LOCAL_TIME_ZONE).dt.month().to_numpy()
    )
    return core.matched_key(
        local_half_hour=cal.half_hour.astype(np.int64),
        month=month.astype(np.int64),
        day_type=(cal.day_type != 0).astype(np.int64),
    )


@dataclass(frozen=True)
class Sky:
    """CAMS and sun geometry of one site on the master grid, plus the farm's output if a farm."""

    u: np.ndarray
    u_clear: np.ndarray
    ghi: np.ndarray
    ghi_clear: np.ndarray
    zenith: np.ndarray
    power: np.ndarray | None


def _sky_of(*, sub: pl.DataFrame, with_power: bool) -> Sky:
    idx = sub["i"].to_numpy()

    def col(name: str) -> np.ndarray:
        return to_grid(index=idx, values=sub[name].to_numpy().astype(float))

    return Sky(
        u=col("u"),
        u_clear=col("u_clear"),
        ghi=col("ghi_u"),
        ghi_clear=col("ghi_u_clear"),
        zenith=col("zenith"),
        power=col("power") if with_power else None,
    )


def load_sky(
    *, log: guard.ReadLog
) -> tuple[dict[int, Sky], dict[str, Sky], dict[str, float], dict[str, int], int]:
    """Return CAMS by Primary time series id, and CAMS plus output by farm.

    A Primary's `power` column is never selected. The farms are labelled `F1`, `F2`, ... in a
    fixed order of the site table, and the farm's time series id is returned beside the label.

    Args:
        log: The read log, which gets the sky read and the holdout counts.

    Returns:
        The Primaries' sky by time series id; the farms' sky by label; the farms' capacity in
        megawatts by label; the farms' time series id by label; and the count of frame rows on or
        after the holdout.
    """
    sites = pl.read_parquet(SITES39_PATH)
    lazy = pl.scan_parquet(FRAME39_PATH)
    n_after = int(
        lazy.filter(pl.col("time") >= pl.lit(core.HOLDOUT)).select(pl.len()).collect().item()
    )
    before = pl.col("time") < pl.lit(core.HOLDOUT)
    log.record(name="cams_at_primaries", kind="sky")
    subs = (
        lazy.filter((pl.col("kind") == "substation") & before)
        .select(*SKY_COLUMNS)
        .with_columns(grid_index(end_time=pl.col("time")))
        .collect()
    )
    log.record(name="cams_and_output_of_farms", kind="farm")
    farms = (
        lazy.filter((pl.col("kind") == "farm") & before)
        .select(*SKY_COLUMNS, "power")
        .with_columns(grid_index(end_time=pl.col("time")))
        .collect()
    )
    log.record_holdout(
        name="frame_39",
        read_after=int(subs.filter(pl.col("time") >= pl.lit(core.HOLDOUT)).height)
        + int(farms.filter(pl.col("time") >= pl.lit(core.HOLDOUT)).height),
        dropped=n_after,
    )
    site_ids = dict(zip(sites["site"].to_list(), sites["time_series_id"].to_list(), strict=True))
    caps = dict(zip(sites["site"].to_list(), sites["effective_capacity_mw"].to_list(), strict=True))
    primaries = {
        int(site_ids[s]): _sky_of(sub=subs.filter(pl.col("site") == s), with_power=False)
        for s in subs["site"].unique().to_list()
        if s in site_ids
    }
    farm_sky: dict[str, Sky] = {}
    farm_cap: dict[str, float] = {}
    farm_id: dict[str, int] = {}
    for n, s in enumerate(sorted(farms["site"].unique().to_list()), start=1):
        label = f"F{n}"
        farm_sky[label] = _sky_of(sub=farms.filter(pl.col("site") == s), with_power=True)
        farm_cap[label] = float(caps[s])
        farm_id[label] = int(site_ids[s])
    return primaries, farm_sky, farm_cap, farm_id, n_after


@dataclass(frozen=True)
class EngineFrame:
    """The engine-state frame of experiment 697 on the master grid.

    Attributes:
        has_row: True where the frame has a half-hour.
        k_path: The smoothed number of units running (NaN where absent).
        era: The era name of each half-hour (empty where absent).
        change: True at the first half-hour with a new engine state (never at an era's first row).
        key: The calendar matched key of every half-hour of the grid, built from the calendar alone.
        frame_key: The frame's own matched key (-1 where the frame has no row), kept for a check.
        in_drop: True inside the multi-day drops that experiment 159 flagged.
        n_after_holdout: Frame rows whose end stamp is on or after the holdout.
    """

    has_row: np.ndarray
    k_path: np.ndarray
    era: np.ndarray
    change: np.ndarray
    key: np.ndarray
    frame_key: np.ndarray
    in_drop: np.ndarray
    n_after_holdout: int


def engine_changes(*, k_path: np.ndarray, has_row: np.ndarray, era: np.ndarray) -> np.ndarray:
    """Return True where `k_path` differs from the previous half-hour inside one era.

    The first row of every era and of every run after a missing row has no change.
    """
    change = np.zeros(k_path.shape, dtype=bool)
    both = has_row[1:] & has_row[:-1] & (era[1:] == era[:-1])
    diff = np.where(both, k_path[1:] - k_path[:-1], 0.0)
    change[1:] = np.nan_to_num(diff) != 0
    return change


def load_engine_frame(*, log: guard.ReadLog) -> EngineFrame:
    """Read experiment 697's frame (the columns that hold no reading) and put it on the grid."""
    log.record(name="engine_state_frame", kind="primary")
    cols = (
        "start", "era", "k_path", "local_half_hour", "month", "weekend", "bank_holiday", "in_drop",
    )  # fmt: skip
    frame = (
        pl.scan_parquet(ENGINE_FRAME_PATH)
        .select(*cols)
        .with_columns(pl.col("start").dt.offset_by("30m").alias("end"))
        .collect()
    )
    n_after = int(frame.filter(pl.col("end") >= pl.lit(core.HOLDOUT)).height)
    frame = frame.filter(pl.col("end") < pl.lit(core.HOLDOUT)).with_columns(
        grid_index(end_time=pl.col("end"))
    )
    log.record_holdout(
        name="engine_frame",
        read_after=int(frame.filter(pl.col("end") >= pl.lit(core.HOLDOUT)).height),
        dropped=n_after,
    )
    idx = frame["i"].to_numpy()
    has_row = np.zeros(core.N_GRID, dtype=bool)
    ok = (idx >= 0) & (idx < core.N_GRID)
    has_row[idx[ok]] = True
    k = to_grid(index=idx, values=frame["k_path"].to_numpy().astype(float))
    era = np.full(core.N_GRID, "", dtype=object)
    era[idx[ok]] = frame["era"].to_numpy()[ok]
    lhh = np.full(core.N_GRID, -1)
    lhh[idx[ok]] = frame["local_half_hour"].to_numpy()[ok]
    mon = np.full(core.N_GRID, -1)
    mon[idx[ok]] = frame["month"].to_numpy()[ok]
    dtype = np.full(core.N_GRID, -1)
    dtype[idx[ok]] = (frame["weekend"].to_numpy() | frame["bank_holiday"].to_numpy()).astype(int)[
        ok
    ]
    drop = np.zeros(core.N_GRID, dtype=bool)
    drop[idx[ok]] = frame["in_drop"].to_numpy()[ok]
    return EngineFrame(
        has_row=has_row,
        in_drop=drop,
        k_path=k,
        era=era,
        change=engine_changes(k_path=k, has_row=has_row, era=era),
        key=calendar_key(),
        frame_key=core.matched_key(local_half_hour=lhh, month=mon, day_type=dtype),
        n_after_holdout=n_after,
    )


def find_engine_record() -> Path:
    """Locate experiment 194's engine record, or raise when there is not exactly one."""
    hits = sorted(ML_EXPERIMENTS_DATA_DIR.glob(ENGINE_RECORD_GLOB))
    if len(hits) != 1:
        msg = f"expected one engine record, found {len(hits)}"
        raise FileNotFoundError(msg)
    return hits[0]


def load_engine_record(*, log: guard.ReadLog) -> tuple[np.ndarray, Path, int]:
    """Return the 44-day engine record's power on the master grid, its path, and rows after holdout."""
    log.record(name="engine_record", kind="primary")
    path = find_engine_record()
    frame = pl.read_parquet(path).select("time", "power")
    n_after = int(frame.filter(pl.col("time") >= pl.lit(core.HOLDOUT)).height)
    frame = frame.filter(pl.col("time") < pl.lit(core.HOLDOUT)).with_columns(
        grid_index(end_time=pl.col("time"))
    )
    log.record_holdout(
        name="engine_record",
        read_after=int(frame.filter(pl.col("time") >= pl.lit(core.HOLDOUT)).height),
        dropped=n_after,
    )
    out = to_grid(index=frame["i"].to_numpy(), values=frame["power"].to_numpy().astype(float))
    return out, path, n_after


def load_effective_capacity(*, ids: list[int]) -> dict[int, float]:
    """Return each series' last effective capacity before the holdout, in megawatts."""
    frame = (
        pl.scan_delta(str(EFFECTIVE_CAPACITY_PATH))
        .filter(pl.col("time_series_id").is_in(ids) & (pl.col("time") < pl.lit(core.HOLDOUT)))
        .sort("time")
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").last())
        .collect()
    )
    return dict(
        zip(
            frame["time_series_id"].to_list(),
            frame["effective_capacity_mw"].to_list(),
            strict=True,
        )
    )


def register_pv_capacity_mw(*, primary: str) -> float | None:
    """Return the register's PV capacity at one Primary in megawatts, or None when the table is missing.

    The table is the register capacity table of experiments 716 and 717. A missing or ambiguous
    table, or a Primary with no row, gives None, which job 1 reports as an `unknown` band.
    """
    hits = sorted(ML_EXPERIMENTS_DATA_DIR.glob(PV_REGISTER_GLOB))
    if len(hits) != 1:
        return None
    table = pl.read_parquet(hits[0])
    if not {"primary", "capacity_mw"} <= set(table.columns):
        return None
    rows = table.filter(pl.col("primary") == primary)
    return float(rows["capacity_mw"].sum()) if rows.height else None


def ens_temperature_leads() -> tuple[int, int]:
    """Return the number of distinct leads with a temperature, and rows on or after the holdout."""
    lazy = pl.scan_parquet(ENS_MEMBERS_PATH)
    leads = int(
        lazy.filter(pl.col("temp_c").is_not_null())
        .select(pl.col("lead_hours").n_unique())
        .collect()
        .item()
    )
    after = int(
        lazy.filter(pl.col("valid_time") >= pl.lit(core.HOLDOUT)).select(pl.len()).collect().item()
    )
    return leads, after

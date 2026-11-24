# ruff: noqa: E501, PERF401, PLR0915
"""Experiment 723, job W0: sensor audit, power table, and switching-log mask.

Job W0 asks which tests can be read, whether the sky sensors agree, and what the switching logs
mask.

Run it only through `run.sh --go e00723_w0.py`, after the Opus code review. Job W0 has no
hypothesis and no verdict.

**Job W0 computes no statistic that depends on the direction of a co-movement between a sky
sensor, or a Primary's own leave-out residual, and the reading of a Primary that enters a bar of
job W2 or W3.** It computes counts, standard deviations of one series, farm-to-farm correlations,
and counts that depend only on the sizes of two changes. The one correlation that involves a
leave-out residual is the sky's share at D12, written as `r^2` only, and the residual that leaves
D12 out holds no reading of D12. The sensor audit runs before any Primary's reading is read.

Outputs (in `data/ml-experiments/00723-*/`): `w0_checks.json`, `w0_sensor_audit.csv`,
`w0_pairs.csv`, `w0_membership.csv`, `w0_power_table.csv`, `w0_share_pairs.csv`,
`w0_switching_sheets.csv`, `run_log_w0.json`, and under `PRIVATE/` the switching-log windows.
Stdout prints counts, shares, and correlations of series only.
"""

import re
from pathlib import Path
from typing import Any, Final

import e00722_common as common722  # ty: ignore[unresolved-import]
import e00722_data as data  # ty: ignore[unresolved-import]
import e00722_sky as sky  # ty: ignore[unresolved-import]
import e00723_common as common
import e00723_core as core
import e00723_guard as guard
import e00723_score as score
import e00723_world as world
import numpy as np
import polars as pl

FOLDER: Final[Path] = guard.FOLDER
"""This experiment's code folder."""

D10_EXPORT_MIN_ELEVATION_DEG: Final[float] = 30.0
"""The review's recount of D10's export hours takes a sun at 30 degrees or more (amendment, section 3)."""

OUTPUTS: Final[tuple[str, ...]] = (
    common.W0_CHECKS_NAME, "w0_sensor_audit.csv", common.W0_PAIRS_NAME, common.W0_MEMBERSHIP_NAME,
    "w0_power_table.csv", "w0_share_pairs.csv", "w0_switching_sheets.csv",
)  # fmt: skip
"""Every file the job writes outside `PRIVATE/`; a run deletes them first so that a crash leaves none stale."""


# ---------------------------------------------------------------------------------------------
# Pure helpers (tested on synthetic frames)
# ---------------------------------------------------------------------------------------------
def sheet_summary(*, sheets: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return each sheet's name, column names, and row count."""
    return pl.DataFrame(
        {
            "sheet": list(sheets),
            "columns": ["; ".join(f.columns) for f in sheets.values()],
            "n_columns": [f.width for f in sheets.values()],
            "n_rows": [f.height for f in sheets.values()],
        },
        schema={
            "sheet": pl.String,
            "columns": pl.String,
            "n_columns": pl.Int64,
            "n_rows": pl.Int64,
        },
    )


LOCAL_ZONE: Final[str] = "Europe/London"
"""RULED: a switching-log time with no time zone is read as local time in this zone."""

START_WORDS: Final[tuple[str, ...]] = ("start", "from", "begin")
"""Whole words of a column name that mark the start time."""

END_WORDS: Final[tuple[str, ...]] = ("end", "finish", "restore", "restored", "restoration", "until")
"""Whole words of a column name that mark the end time."""

NOT_TIME_WORDS: Final[tuple[str, ...]] = ("transfer", "transferred", "transferring", "name")
"""Whole words that rule a column out as a start or end time (for example "transferred from")."""

TIMED_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
)  # fmt: skip
"""The text formats that carry a time of day."""

DATE_ONLY_FORMATS: Final[tuple[str, ...]] = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")
"""The text formats that carry a date and no time of day."""

DATETIME_FORMATS: Final[tuple[str, ...]] = (*TIMED_FORMATS, *DATE_ONLY_FORMATS)
"""The formats tried on a text time; each row takes the first format that parses it, timed formats first."""


class SwitchingLogDefectError(RuntimeError):
    """Raised when a sheet has a substation column, a start column, and rows, but no row matches a Primary.

    That case is an input defect (row 1 of the verdict table), not "no mask".
    """


def _words(*, name: str) -> set[str]:
    """Return the lower-case whole words of a column name."""
    return {w for w in re.split(r"[^a-z0-9]+", name.lower()) if w}


def _find(
    *, names: dict[str, str], parts: tuple[str, ...], not_parts: tuple[str, ...] = ()
) -> str | None:
    """Return the first column that has one of `parts` as a whole word and none of `not_parts`."""
    return next(
        (
            v
            for k, v in names.items()
            if _words(name=k) & set(parts) and not _words(name=k) & set(not_parts)
        ),
        None,
    )


def _substation_text(series: pl.Series) -> pl.Series:
    """Return the substation column as the text of an integer ("12345.0" and 12345.0 both give "12345")."""
    return (
        series.cast(pl.String, strict=False)
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .round(0)
        .cast(pl.Int64, strict=False)
        .cast(pl.String)
    )


PLAUSIBLE_YEARS: Final[tuple[int, int]] = (2000, 2100)
"""A parsed switching-log time outside these years (a two-digit year read as year 25, for example) is an error."""


def _parse_local(series: pl.Series) -> tuple[pl.DataFrame, pl.Series]:
    """Return a column as local (Europe/London) naive times with a date-only flag, and the implausible cells.

    The first frame has `local` (naive local time, null when the cell does not parse) and
    `date_only` (True for a `Date`, and for a text with a date-only format). The second result is a
    Boolean series that marks a cell that parsed to a year outside `PLAUSIBLE_YEARS`; such a cell is
    null in `local`. A cell with a zone is converted to local time. A text cell takes the first of
    `DATETIME_FORMATS` that parses it, timed formats first, and a text that matches none ("ongoing")
    is null. A workbook read with every cell as text holds a date as `2025-06-10 00:00:00`, which
    matches a timed format, so a date-only cell of a workbook is a timed midnight here; the rules of
    `windows_with_diagnostics` read it per row.
    """
    n = len(series)
    if isinstance(series.dtype, pl.Datetime) and series.dtype.time_zone is not None:
        local = series.dt.convert_time_zone(LOCAL_ZONE).dt.replace_time_zone(None)
        date_only = pl.Series([False] * n)
    elif series.dtype == pl.Date:
        local = series.cast(pl.Datetime("us"))
        date_only = local.is_not_null()
    elif isinstance(series.dtype, pl.Datetime):
        local = series.cast(pl.Datetime("us"))
        date_only = pl.Series([False] * n)
    else:
        text = pl.DataFrame({"t": series.cast(pl.String, strict=False).str.strip_chars()})
        parsed = text.select(
            timed=pl.coalesce(
                [
                    pl.col("t").str.to_datetime(f, strict=False, time_zone=None)
                    for f in TIMED_FORMATS
                ]
            ),
            dated=pl.coalesce(
                [
                    pl.col("t").str.to_datetime(f, strict=False, time_zone=None)
                    for f in DATE_ONLY_FORMATS
                ]
            ),
        )
        local = parsed.select(pl.coalesce("timed", "dated").cast(pl.Datetime("us"))).to_series()
        date_only = parsed["timed"].is_null() & parsed["dated"].is_not_null()
    year = local.dt.year()
    implausible = (year < PLAUSIBLE_YEARS[0]) | (year > PLAUSIBLE_YEARS[1])
    implausible = implausible.fill_null(False)
    local = (
        pl.DataFrame({"t": local, "bad": implausible})
        .select(pl.when(pl.col("bad")).then(None).otherwise(pl.col("t")))
        .to_series()
    )
    return pl.DataFrame({"local": local, "date_only": date_only.fill_null(False)}), implausible


def _local_to_utc(series: pl.Series) -> pl.Series:
    """Return naive local (Europe/London) times as naive UTC; a time in the spring-forward gap is null."""
    return (
        series.dt.replace_time_zone(LOCAL_ZONE, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
    )


def window_times(
    *, start: pl.Series, end: pl.Series, end_date_only: pl.Series
) -> tuple[pl.Series, pl.Series]:
    """Return the start and end of every window as naive UTC, from naive local start and end cells.

    RULED (design lead): a workbook stores a date as a datetime at midnight, so two per-row rules
    read a date with no time.

    1. An end at exactly local 00:00 on a row whose end is not after a start on the same local date
       is local 24:00 of that date (the start of the next day). A text end with a date-only format
       is local 24:00 of its date whatever the start. An end at 00:00 on a later date than the start
       stays at 00:00.
    2. A start with no end covers the local calendar day of the start, from local 00:00 to local
       24:00, whether the start is timed or not.

    A null start gives a null window. A day is added in local time, so a window may last 23 or 25 hours.
    """
    frame = pl.DataFrame({"s": start, "e": end, "eo": end_date_only.fill_null(False)})
    day = pl.col("s").dt.truncate("1d")
    midnight_end = pl.col("e") == pl.col("e").dt.truncate("1d")
    same_day_not_after = (pl.col("e").dt.truncate("1d") == day) & (pl.col("e") <= pl.col("s"))
    extend = pl.col("eo") | (midnight_end & same_day_not_after)
    out = frame.select(
        s=pl.when(pl.col("e").is_null()).then(day).otherwise(pl.col("s")),
        e=pl.when(pl.col("e").is_null())
        .then(day.dt.offset_by("1d"))
        .when(extend)
        .then(pl.col("e").dt.truncate("1d").dt.offset_by("1d"))
        .otherwise(pl.col("e")),
    )
    return _local_to_utc(out["s"]), _local_to_utc(out["e"])


def windows_with_diagnostics(
    *, sheets: dict[str, pl.DataFrame], numbers: dict[str, int]
) -> tuple[dict[str, list[tuple[int, int]]], list[dict[str, Any]]]:
    """Return one window per logged event and Primary in grid rows (first, last), and each sheet's diagnostics.

    A sheet is used when it has a column whose name holds the whole word "substation" (and not
    "name") and a column with a start word ("start", "from", "begin") that is not a transfer
    column. The end column is the first with an end word ("end", "finish", "restore", "until") and
    no start word. A time with no zone is read as Europe/London. The window runs from the start to
    the end, with the two per-row rules of `window_times` for a date with no time and for a start
    with no end. A cell that parses to nothing (a text such as "ongoing") is a missing time, and a
    row whose start cannot be parsed is skipped. The diagnostics count the rows with a start that
    does not parse, an end that does not parse, a time outside `PLAUSIBLE_YEARS`, an end that
    precedes its start (skipped), and a window that lies wholly outside the grid.

    Args:
        sheets: The workbook's sheets by name.
        numbers: Each Primary's substation number by label.

    Returns:
        The windows by Primary label, and one diagnostics row per sheet.

    Raises:
        SwitchingLogDefectError: A sheet has the two columns and rows but no row matches a Primary.
    """
    out: dict[str, list[tuple[int, int]]] = {label: [] for label in numbers}
    diagnostics: list[dict[str, Any]] = []
    for sheet_name, frame in sheets.items():
        names = {c.lower(): c for c in frame.columns}
        sub = _find(names=names, parts=("substation",), not_parts=("name",))
        beg = _find(names=names, parts=START_WORDS, not_parts=NOT_TIME_WORDS)
        end = _find(names=names, parts=END_WORDS, not_parts=(*START_WORDS, *NOT_TIME_WORDS))
        diag: dict[str, Any] = {
            "sheet": sheet_name, "substation_column": sub, "start_column": beg, "end_column": end,
            "n_rows": frame.height, "n_rows_start_parsed": 0, "n_rows_matched": 0,
            "n_rows_start_not_parsed": 0, "n_rows_end_not_parsed": 0, "n_rows_time_implausible": 0,
            "n_rows_end_before_start": 0, "n_windows_outside_grid": 0, "used": False,
        }  # fmt: skip
        diagnostics.append(diag)
        if sub is None or beg is None:
            continue
        diag["used"] = True
        start_cells, start_bad = _parse_local(frame[beg])
        if end:
            end_cells, end_bad = _parse_local(frame[end])
        else:
            end_cells = pl.DataFrame(
                {
                    "local": pl.Series([None] * frame.height, dtype=pl.Datetime("us")),
                    "date_only": pl.Series([False] * frame.height),
                }
            )
            end_bad = pl.Series([False] * frame.height)
        begin, finish = window_times(
            start=start_cells["local"], end=end_cells["local"], end_date_only=end_cells["date_only"]
        )
        # A start and an end that both survive the rules give the window; compare in UTC.
        sub_text = _substation_text(frame[sub])
        start_given = frame[beg].is_not_null()
        diag["n_rows_start_parsed"] = int(start_cells["local"].is_not_null().sum())
        diag["n_rows_start_not_parsed"] = int(
            (start_given & start_cells["local"].is_null() & ~start_bad).sum()
        )
        if end:
            diag["n_rows_end_not_parsed"] = int(
                (frame[end].is_not_null() & end_cells["local"].is_null() & ~end_bad).sum()
            )
        diag["n_rows_time_implausible"] = int((start_bad | end_bad).sum())
        matched_rows: set[int] = set()
        for label, number in numbers.items():
            idx = [
                i
                for i, (sv, b) in enumerate(zip(sub_text.to_list(), begin.to_list(), strict=True))
                if sv == str(number) and b is not None
            ]
            matched_rows.update(idx)
            for i in idx:
                b_t, e_t = begin[i], finish[i]
                if e_t is None:  # an end in the spring-forward gap
                    diag["n_rows_end_not_parsed"] += 1
                    continue
                if e_t < b_t:
                    diag["n_rows_end_before_start"] += 1
                    continue
                first, last = world.utc_window_rows(start=b_t, end=e_t)
                if last < 0 or first >= core.N_GRID:
                    diag["n_windows_outside_grid"] += 1
                out[label].append((first, last))
        diag["n_rows_matched"] = len(matched_rows)
        if frame.height > 0 and diag["n_rows_start_parsed"] > 0 and not matched_rows:
            msg = (
                f"sheet {sheet_name!r} has a substation column and a start column with rows, "
                "but no row matches a Primary: an input defect"
            )
            raise SwitchingLogDefectError(msg)
    return out, diagnostics


def read_workbook(*, path: Path) -> dict[str, pl.DataFrame]:
    """Return every sheet of the switching log with every cell as text.

    `infer_schema_length=0` reads a column as text whatever it holds, so one text cell such as
    "ongoing" in a column of dates cannot make the workbook unreadable. The windows parse each cell.
    """
    return pl.read_excel(path, sheet_id=0, engine="openpyxl", infer_schema_length=0)


def windows_from_sheets(
    *, sheets: dict[str, pl.DataFrame], numbers: dict[str, int]
) -> dict[str, list[tuple[int, int]]]:
    """Return the windows of `windows_with_diagnostics`, without the diagnostics."""
    return windows_with_diagnostics(sheets=sheets, numbers=numbers)[0]


def d10_counts(
    *, signed: np.ndarray, zenith: np.ndarray, start: int
) -> dict[str, float | int | str]:
    """Return the recount of D10's export hours and export days, with the base stated.

    An export hour is a hour-ending row of the later span on which `X < 0` at the row and at the
    row before it, with the sun at 30 degrees or more at both. The base is the hour-ending rows with
    a finite reading at both ends.
    """
    elev = 90.0 - zenith
    hour = world.hour_end_rows(n=len(signed))
    finite = np.isfinite(signed)
    finite2 = finite.copy()
    finite2[1:] &= finite[:-1]
    sun = np.isfinite(elev) & (elev >= D10_EXPORT_MIN_ELEVATION_DEG)
    sun2 = sun.copy()
    sun2[1:] &= sun[:-1]
    neg = signed < 0
    neg2 = neg.copy()
    neg2[1:] &= neg[:-1]
    span = np.arange(len(signed)) >= start
    base = hour & finite2 & span
    export = base & neg2 & sun2
    day = sky.day_of(n=len(signed))
    lit10 = np.isfinite(elev) & (elev >= core.DAYLIGHT_ELEVATION_DEG)
    half = finite & span
    return {
        "base": "hour-ending rows of the later span with a reading at both ends",
        "n_base_hours": int(base.sum()),
        "n_export_hours": int(export.sum()),
        "n_export_days": int(np.unique(day[export]).size),
        "share_negative_half_hours": float(neg[half].mean()) if half.any() else float("nan"),
        "share_negative_daylight_half_hours": float(neg[half & lit10].mean())
        if (half & lit10).any()
        else float("nan"),
    }


# ---------------------------------------------------------------------------------------------
# Stage 1: the sensor audit (no Primary's reading is read)
# ---------------------------------------------------------------------------------------------
def sensor_audit(*, stage: common.SkyStage) -> dict[str, Any]:
    """Run the three audit conditions, the same-sky table, the polygon filter, and the sensor membership.

    Returns the audit rows, the pair table without the powered columns, the membership by host,
    the farms' own fast changes, and the flags the job writes to its check file.
    """
    hosts = common.SENSOR_HOSTS
    farms = sorted(stage.farm_sky)
    months = world.month_of_rows(n=core.N_GRID)
    in_months = np.isin(months, core.TEST_MONTHS)
    lit_of: dict[str, np.ndarray] = {}
    k_of: dict[str, np.ndarray] = {}
    for name, s in {**{h: stage.sky_of[h] for h in hosts}, **stage.farm_sky}.items():
        lit_of[name] = sky.daylight(
            zenith_deg=s.zenith, min_elevation_deg=core.DAYLIGHT_ELEVATION_DEG
        )
        k_of[name] = sky.clear_sky_index(ghi=s.ghi, ghi_clear=s.ghi_clear)
    audit_rows: list[dict[str, Any]] = []
    # Condition 1: resolution. RULED: the arc term is void if any site's share of hour-ending half-hours
    # on which u_clear does not change exceeds 1% (0% is expected); W2 then builds no world.
    share_u: list[float] = []
    share_u_names: list[str] = []
    for name, s in {**{h: stage.sky_of[h] for h in hosts}, **stage.farm_sky}.items():
        res = world.resolution_shares(k=k_of[name], u_clear=s.u_clear, lit=lit_of[name])
        share_u.append(res["share_u_clear_unchanged"])
        share_u_names.append(name)
        audit_rows.append({"condition": 1, "scope": name, "kind": "resolution", **_as_row(res)})
    unchanged_by_site = dict(zip(share_u_names, share_u, strict=True))
    arc_void = core.arc_term_void(unchanged_share_by_site=unchanged_by_site)
    # Condition 2: arc removal, per farm.
    farm_dz: dict[str, np.ndarray] = {}
    arc_ok: dict[str, bool] = {}
    for f in farms:
        s = stage.farm_sky[f]
        share = world.farm_share(output_mw=_output(s), capacity_mw=stage.farm_cap[f])
        dz = world.fast_sensor_change(z=share, u_clear=s.u_clear, lit=lit_of[f])
        farm_dz[f] = dz
        r, n = world.arc_removal_r(
            dz=dz, d_u_clear=world.arc_change(u_clear=s.u_clear, lit=lit_of[f])
        )
        arc_ok[f] = core.arc_removal_ok(r=r)
        audit_rows.append(
            {
                "condition": 2,
                "scope": f,
                "kind": "arc_removal",
                "r": r,
                "n_rows": n,
                "passes": arc_ok[f],
            }
        )
    # The same-sky table and the polygon filter.
    pair_rows: list[dict[str, Any]] = []
    for h in hosts:
        inside = set(_farms_inside(stage=stage, host=h))
        for f in farms:
            r, n = sky.correlation_of_changes(
                a=sky.fast_k_change(k=k_of[h], lit=lit_of[h]),
                b=sky.fast_k_change(k=k_of[f], lit=lit_of[f]),
            )
            pair_rows.append(
                {
                    "host": h, "farm": f, "r_cams": r, "n_rows": n,
                    "same_sky": bool(r >= core.HALF), "outside_polygon": f not in inside,
                    "arc_ok": arc_ok[f],
                }
            )  # fmt: skip
    pairs = pl.DataFrame(pair_rows)
    # Condition 3: coherence, per farm and per host sensor.
    membership: dict[str, list[str]] = {}
    for h in hosts:
        cand = pairs.filter(
            (pl.col("host") == h)
            & pl.col("same_sky")
            & pl.col("outside_polygon")
            & pl.col("arc_ok")
        )["farm"].to_list()
        in_season = in_months & lit_of[h]
        keep, r_of = world.keep_coherent(dz_of={f: farm_dz[f] for f in cand}, in_season=in_season)
        membership[h] = keep
        for f in cand:
            audit_rows.append(
                {"condition": 3, "scope": f"{h}/{f}", "kind": "coherence", "r": r_of[f], "n_rows": int((np.isfinite(farm_dz[f]) & in_season).sum()), "passes": f in keep}
            )  # fmt: skip
    return {
        "audit": pl.DataFrame(audit_rows, infer_schema_length=None),
        "pairs": pairs,
        "membership": membership,
        "farm_dz": farm_dz,
        "in_months": in_months,
        "lit_of": lit_of,
        "arc_void": arc_void,
        "u_clear_unchanged_share_by_site": unchanged_by_site,
    }


def _output(s: data.Sky) -> np.ndarray:
    if s.power is None:
        msg = "a farm has no output"
        raise ValueError(msg)
    return s.power


def _farms_inside(*, stage: common.SkyStage, host: str) -> list[str]:
    ids = stage.inside.get(host, set())
    return sorted(f for f, ts in stage.farm_id.items() if ts in ids)


def _as_row(res: dict[str, float]) -> dict[str, Any]:
    return {
        "r": float("nan"),
        "n_rows": int(res["n_rows_index"]),
        "passes": None,
        "share_index_unchanged": res["share_index_unchanged"],
        "share_u_clear_unchanged": res["share_u_clear_unchanged"],
    }


# ---------------------------------------------------------------------------------------------
# Stage 2 (after the audit): everything that reads one series at a time
# ---------------------------------------------------------------------------------------------
def powered_table(
    *, ctx: common.Context, audit: dict[str, Any], membership: dict[str, list[str]]
) -> pl.DataFrame:
    """Return the polygon-filtered pair table with the keeps-a-sensor, coherence, dilution, and powered columns.

    A pair qualifies when it is same-sky, lies outside the host's polygon, and the host's sensor
    keeps at least 2 farms after the injected farm is removed. Coherence is the farm-to-farm
    correlation of the injected farm's `dZ~` with the mean `dZ~` of the remaining farms of the
    host's sensor, on daylight half-hours from April to September. The dilution reads the standard
    deviation of one series twice (the injected output's fast change and the host's reading's fast
    change, both in points of the host's capacity), and no co-movement.
    """
    pairs: pl.DataFrame = audit["pairs"]
    rows: list[dict[str, Any]] = []
    start = ctx.start_row
    for rec in pairs.iter_rows(named=True):
        host, farm = rec["host"], rec["farm"]
        remaining = [f for f in membership.get(host, []) if f != farm]
        qualifies = bool(
            rec["same_sky"]
            and rec["outside_polygon"]
            and len(remaining) >= 2
            and host in common.SIGNED_SIX
        )
        row: dict[str, Any] = {**rec, "keeps_sensor": len(remaining) >= 2, "qualifies": qualifies}
        in_season = audit["in_months"] & audit["lit_of"][host]
        dz_of = {f: audit["farm_dz"][f] for f in remaining}
        if qualifies:
            dz_of[farm] = audit["farm_dz"][farm]
            coh, _ = world.coherence_r(dz_of=dz_of, farm=farm, in_season=in_season)
            cap = ctx.capacity(host=host)
            reading = world.fold(signed_mw=ctx.series[host], capacity_mw=cap)
            dm = world.reading_fast_change(
                reading_mw=reading, capacity_mw=cap, lit=audit["lit_of"][host], key=ctx.eng.key,
                overcast_row=ctx.overcast_row(host=host), start=start,
            )  # fmt: skip
            s_h = float(np.nanstd(dm)) if np.isfinite(dm).any() else float("nan")
            row["coherence_injected"] = coh
            row["sd_host_points"] = s_h
            ref_rows = score.level_rows(ctx=ctx, host=host)
            farm_f = ctx.farm_norm(farm=farm)
            n_ref = world.reference_row_count(
                host_signed_mw=ctx.series[host], farm_norm_series=farm_f, rows=ref_rows
            )
            # The farm-sign flag reads the farm's recorded output, before the clipping at 0 that
            # `farm_norm` applies, so that a farm that reads negative for generation shows. Job W2
            # drops a pair whose flag is set (`common.qualifying_pairs`).
            raw_mw = ctx.farm_output(farm=farm)
            ref_ok = ref_rows & np.isfinite(ctx.series[host]) & np.isfinite(raw_mw)
            share_ref = (
                float(np.median(raw_mw[ref_ok] / ctx.farm_cap[farm]))
                if ref_ok.any()
                else float("nan")
            )
            row["n_reference_rows"] = n_ref
            row["farm_median_share_reference"] = share_ref
            row["farm_reads_negative"] = bool(np.isfinite(share_ref) and share_ref <= 0)
            for level in core.LEVELS:
                scale = world.injection_scale(
                    host_signed_mw=ctx.series[host], farm_norm_series=farm_f,
                    rows=ref_rows, level=level,
                )  # fmt: skip
                row[f"scale_finite_level_{level}"] = bool(np.isfinite(scale))
                dz = world.fast_sensor_change(
                    z=scale * farm_f / cap * 100.0,
                    u_clear=ctx.farm_sky[farm].u_clear, lit=audit["lit_of"][farm],
                )  # fmt: skip
                s_f = (
                    float(np.nanstd(dz[start:])) if np.isfinite(dz[start:]).any() else float("nan")
                )
                d = core.dilution(sd_injected=s_f, sd_host=s_h)
                row[f"sd_injected_points_level_{level}"] = s_f
                row[f"dilution_level_{level}"] = d
                row[f"powered_level_{level}"] = core.is_powered(coherence=coh, dilution_value=d)
        rows.append(row)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return common.with_pair_columns(frame=frame)


def power_table(
    *,
    ctx: common.Context,
    membership: dict[str, list[str]],
    windows: dict[str, list[tuple[int, int]]],
) -> pl.DataFrame:
    """Return the power table: for each fitted slope of jobs W2 and W3, the row count and the two standard deviations.

    The table is a read-out. No slope in jobs W2 or W3 carries a bar. The standard error is
    predicted as 1.25 times the naive figure.
    """
    start = ctx.start_row
    key, quiet = ctx.eng.key, core722_quiet(ctx=ctx)
    rows: list[dict[str, Any]] = []

    def add(
        *,
        name: str,
        host: str,
        kind: str,
        signed_regressor: bool,
        extra: np.ndarray | None = None,
        min_elev: float | None = None,
    ) -> None:
        masked = score.windows_mask(windows=windows, host=host)
        sel = world.closure_rows(
            zenith=ctx.sky_of[host].zenith,
            masked=masked,
            start=start,
            kind=kind,
            min_elevation_deg=min_elev,
        )
        if extra is not None:
            sel &= extra
        resid = ctx.residual(host=host)
        x = (
            ctx.series[host]
            if signed_regressor
            else world.fold(signed_mw=ctx.series[host], capacity_mw=ctx.capacity(host=host))
        )
        d_r = world.hourly_demeaned(x=resid, key=key, quiet=quiet)
        d_x = world.hourly_demeaned(x=x, key=key, quiet=quiet)
        rows.append(
            {
                "slope": name,
                "host": host,
                **world.power_row(response_change=d_r, regressor_change=d_x, rows=sel),
            }
        )

    add(name="d15_lit_closure", host="D15", kind="lit", signed_regressor=True)
    add(name="d15_dark_closure", host="D15", kind="dark", signed_regressor=True)
    add(name="d10_dark_closure", host="D10", kind="dark", signed_regressor=True)
    neg_hour = _hours_with_sign(signed=ctx.series["D10"], negative=True)
    pos_hour = _hours_with_sign(signed=ctx.series["D10"], negative=False)
    add(
        name="d10_fold_ratio_export_hours",
        host="D10",
        kind="lit",
        signed_regressor=False,
        extra=neg_hour,
        min_elev=D10_EXPORT_MIN_ELEVATION_DEG,
    )
    add(
        name="d10_fold_ratio_import_hours",
        host="D10",
        kind="lit",
        signed_regressor=False,
        extra=pos_hour,
        min_elev=D10_EXPORT_MIN_ELEVATION_DEG,
    )
    add(name="d12_dark_fold_ratio", host="D12", kind="dark", signed_regressor=True)
    add(name="d12_pooled_fold_ratio_lit_hours", host="D12", kind="lit", signed_regressor=True)
    return pl.DataFrame(rows, infer_schema_length=None)


def core722_quiet(*, ctx: common.Context) -> np.ndarray:
    """Return the quiet mask of the engine frame (half-hours at least 6 away from an engine step)."""
    return core.core722.quiet_mask(engine_change=ctx.eng.change, calendar_ok=ctx.eng.has_row)


def _hours_with_sign(*, signed: np.ndarray, negative: bool) -> np.ndarray:
    """Return True on hour-ending rows where the signed series has one sign at the row and at the row before."""
    s = signed < 0 if negative else signed > 0
    out = s.copy()
    out[1:] &= s[:-1]
    return out


def share_pairs(
    *,
    ctx: common.Context,
    membership: dict[str, list[str]],
    windows: dict[str, list[tuple[int, int]]],
) -> pl.DataFrame:
    """Return R3's table: for each host and sensor, the thresholds and the share of daylight pairs that pass.

    The host's reading is its un-injected folded reading, and the pass shares depend only on the
    sizes of the two changes. The meter sensor needs a residual and is not listed.
    """
    rows: list[dict[str, Any]] = []
    start = ctx.start_row
    for host in common.SENSOR_HOSTS:
        farms = membership.get(host, [])
        if len(farms) < 2:
            continue
        cap = ctx.capacity(host=host)
        lit = ctx.lit(host=host)
        signed = ctx.series[host]
        reading = (
            world.fold(signed_mw=signed, capacity_mw=cap)
            if host in (*common.SIGNED_SIX, "D15")
            else signed
        )
        dm = world.reading_fast_change(
            reading_mw=reading, capacity_mw=cap, lit=lit, key=ctx.eng.key,
            overcast_row=ctx.overcast_row(host=host), start=start,
        )  # fmt: skip
        masked = score.windows_mask(windows=windows, host=host)[start:]
        sensors = {
            "farm": ctx.sensor(host=host, farms=farms),
            "cams": world.cams_sensor(s=ctx.sky_of[host], lit=lit),
        }
        for name, s in sensors.items():
            sp = s.from_row(start)
            rows.append(
                {
                    "host": host,
                    "sensor": name,
                    **world.threshold_diagnostics(dm=dm, sensor=sp, masked=masked),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def sky_share_d12(*, ctx: common.Context, membership: dict[str, list[str]]) -> float:
    """Return `r^2` of the residual that leaves D12 out against D12's farm sensor change.

    The change is read on lit pairs.

    Only the squared correlation is returned, never the correlation or a slope, because a squared
    correlation carries no direction. The residual holds no reading of D12.
    """
    farms = membership.get("D12", [])
    if len(farms) < 2:
        return float("nan")
    lit = ctx.lit(host="D12")
    resid = ctx.residual(host="D12")
    d_r = np.full(resid.shape, np.nan)
    d_r[1:] = resid[1:] - resid[:-1]
    sensor = ctx.sensor(host="D12", farms=farms)
    where = np.arange(core.N_GRID) >= ctx.start_row
    r, _ = world.correlation(a=d_r, b=sensor.dz, where=where & lit)
    return float(r * r) if np.isfinite(r) else float("nan")


# ---------------------------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------------------------
def run(*, gate: guard.TimingGate, inputs: list[Path], info: dict[str, Any]) -> None:
    """Run job W0; `inputs` and `info` fill the run log as the job reads."""
    common.assert_launch()
    info["attempt"] = guard.begin_attempt(
        job="w0", names=(*OUTPUTS, "run_log_w0.json"), private_names=(common.W0_WINDOWS_NAME,)
    )
    log = guard.ReadLog()
    guard.say("JOB W0 STARTING")
    stage = common.load_sky_stage(log=log, inputs=inputs)
    audit = sensor_audit(stage=stage)
    log.mark_audit_done()
    ctx = common.load_context(log=log, inputs=inputs, stage=stage)
    log.assert_audit_before_primary_reads()
    info["power_table_delta_version"] = data.power_table_version()
    info["read_order"] = log.as_list()
    info["holdout"] = log.holdout
    later_ok = common.assert_later_span(n_days=ctx.n_days)
    guard.say("later span days:", ctx.n_days, "expected:", core.N_LATER_DAYS, "ok:", later_ok)
    membership: dict[str, list[str]] = audit["membership"]
    pairs = powered_table(ctx=ctx, audit=audit, membership=membership)
    counts_pairs = common.pair_counts(pairs=pairs)
    n_keep = counts_pairs["n_qualifying_pairs"]
    n_powered = counts_pairs["n_powered_pairs_level_2"]
    n_hosts = counts_pairs["n_powered_hosts"]
    n_nan_scale = counts_pairs["n_pairs_non_finite_injection_scale"]
    n_farm_reads_negative = counts_pairs["n_pairs_farm_reads_negative"]
    guard.say(
        "qualifying pairs with a non-finite injection scale:",
        n_nan_scale,
        "| pairs dropped because the farm reads negative:",
        n_farm_reads_negative,
    )
    void_before = core.void_before_run(
        n_pairs_keeping_sensor=n_keep,
        n_powered_pairs_level_2=n_powered,
        n_powered_hosts=n_hosts,
        arc_term_void_flag=bool(audit["arc_void"]),
    )
    guard.say(
        "qualifying pairs:",
        n_keep,
        "powered at level 2.0:",
        n_powered,
        "on hosts:",
        n_hosts,
        "void before run:",
        void_before,
    )
    # Switching logs.
    inputs.append(data.SWITCHING_LOG_PATH)
    switching: dict[str, Any] = {"readable": False}
    windows: dict[str, list[tuple[int, int]]] = {}
    sheets_frame = pl.DataFrame(
        schema={"sheet": pl.String, "columns": pl.String, "n_columns": pl.Int64, "n_rows": pl.Int64}
    )
    try:
        sheets = read_workbook(path=data.SWITCHING_LOG_PATH)
    except Exception as exc:  # noqa: BLE001
        sheets = None
        switching = {"readable": False, "reason": type(exc).__name__}
    if sheets is not None:
        sheets_frame = sheet_summary(sheets=sheets)
        meta = pl.read_parquet(data.METADATA_PATH).select("time_series_id", "substation_number")
        numbers = {
            r["label"]: int(r["substation_number"])
            for r in ctx.topology.primaries.join(meta, on="time_series_id").iter_rows(named=True)
            if r["substation_number"] is not None
        }
        windows, sheet_diag = windows_with_diagnostics(sheets=sheets, numbers=numbers)
        n_windows = {h: len(w) for h, w in windows.items()}
        switching = {
            "readable": True,
            "n_windows_total": sum(n_windows.values()),
            "n_windows_by_primary": n_windows,
            "n_primaries_without_a_number": int(ctx.topology.primaries.height - len(numbers)),
            "sheets": sheet_diag,
        }
    mask_available = bool(switching.get("n_windows_total", 0) > 0)
    guard.say(
        "switching logs readable:",
        switching["readable"],
        "windows:",
        switching.get("n_windows_total"),
        "mask available:",
        mask_available,
    )
    win_rows = [
        {"label": h, "first_row": a, "last_row": b} for h, ws in windows.items() for a, b in ws
    ]
    guard.write_private_parquet(
        frame=pl.DataFrame(
            win_rows, schema={"label": pl.String, "first_row": pl.Int64, "last_row": pl.Int64}
        ),
        name=common.W0_WINDOWS_NAME,
    )
    # Power table, share of pairs, sky share, D10 counts, D15's farm class.
    power = power_table(ctx=ctx, membership=membership, windows=windows)
    shares = share_pairs(ctx=ctx, membership=membership, windows=windows)
    r2 = sky_share_d12(ctx=ctx, membership=membership)
    counts = d10_counts(
        signed=ctx.series["D10"], zenith=ctx.sky_of["D10"].zenith, start=ctx.start_row
    )
    guard.say(
        "D10 export hours:",
        counts["n_export_hours"],
        "on days:",
        counts["n_export_days"],
        "| sky share at D12 (r squared):",
        round(r2, 3),
    )
    d15_farms = ctx.farms_inside(host="D15")
    cls_rows = (
        ctx.classes.filter(pl.col("generator_id").is_in([ctx.farm_id[f] for f in d15_farms]))
        if d15_farms
        else ctx.classes.head(0)
    )
    d15_class = str(cls_rows["generator_class"][0]) if cls_rows.height else None
    d15_addback = bool(
        set(common722.addback_candidates(classes=ctx.classes, primary="D15"))
        & {ctx.farm_id[f] for f in d15_farms}
    )
    guard.say(
        "D15's farm inside the polygon, job 1's class:",
        d15_class,
        "| add-back candidate:",
        d15_addback,
    )
    # Write.
    membership_frame = pl.DataFrame(
        [
            {"host": h, "farm": f, "n_farms_in_sensor": len(fs)}
            for h, fs in membership.items()
            for f in fs
        ],
        schema={"host": pl.String, "farm": pl.String, "n_farms_in_sensor": pl.Int64},
    )
    guard.write_public_csv(frame=audit["audit"], name="w0_sensor_audit.csv")
    guard.write_public_csv(frame=pairs, name=common.W0_PAIRS_NAME)
    guard.write_public_csv(frame=membership_frame, name=common.W0_MEMBERSHIP_NAME)
    guard.write_public_csv(frame=power, name="w0_power_table.csv")
    guard.write_public_csv(frame=shares, name="w0_share_pairs.csv")
    guard.write_public_csv(frame=sheets_frame, name="w0_switching_sheets.csv")
    gate.check(done=1, total=1)
    checks = {
        "attempt": info["attempt"],
        "later_span_days": ctx.n_days,
        "later_span_ok": later_ok,
        "holdout_clean": log.holdout_clean(),
        "arc_term_void": audit["arc_void"],
        "u_clear_unchanged_share_by_site": audit["u_clear_unchanged_share_by_site"],
        "n_pairs_non_finite_injection_scale": n_nan_scale,
        "n_pairs_farm_reads_negative": n_farm_reads_negative,
        "n_qualifying_pairs": n_keep,
        "n_powered_pairs_level_2": n_powered,
        "n_powered_hosts": n_hosts,
        "w2_void_before_run": void_before,
        "sensor_hosts": {h: len(fs) for h, fs in membership.items()},
        "switching_logs": switching,
        "mask_available": mask_available,
        "d15_farm_class": d15_class,
        "d15_add_back_candidate": d15_addback,
        "d15_dead_band_binds": core.d15_dead_band_binds(
            farm_class=d15_class, is_addback_candidate=d15_addback
        ),
        "d12_sky_share_r_squared": r2,
        "d10_counts": counts,
        "core_seconds": round(gate.elapsed(), 1),
    }
    guard.write_public_json(payload=checks, name=common.W0_CHECKS_NAME)
    guard.say("JOB W0 DONE; core-seconds", round(gate.elapsed(), 1))


def main() -> None:
    """Run job W0 and write its run log even when the job crashes."""
    gate = guard.TimingGate(cap_core_seconds=core.CAP_CORE_SECONDS)
    inputs: list[Path] = []
    info: dict[str, Any] = {}
    with guard.run_log(job="w0", gate=gate, inputs=inputs, info=info):
        run(gate=gate, inputs=inputs, info=info)


if __name__ == "__main__":
    main()

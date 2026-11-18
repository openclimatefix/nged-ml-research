# ruff: noqa: E501
"""Experiment 701, GAM side, step 1: extend experiment 103's rows table to 2026-11-14.

    PYTHONPATH=ml-experiments/00103-differentiable-physics-1-data-look:ml-experiments/00701-* \
    DATA_PATH_INTERNAL=<real data folder> uv run --no-sync python ml-experiments/00701-*/e00701_gam_inputs.py

Writes `<ROOT>/gam/inputs/{rows,series_table,sites}.parquet` and
`cams_demand_points_combined.parquet`, and never writes under a stored folder.

**The rows table is rebuilt by experiment 103's own builder functions** (`e00103_build.py`'s
`power_columns`, `temperature_columns`, `irradiance_columns`, and `add_switching_mask`, called with
the extended weather files and the panel end moved to 2026-11-15), followed by the rule of
`e00103_decide.py`'s `write_handed_on` that turns the panel into rows:

- `present`: the cleaned power table holds a reading; `flagged`: the cleaned table gives a
  `drop_reason` (the `substation_zero` rule and the others), and `clean = present and not flagged`.
  **The cleaned table was rewritten on 2026-11-16 with a new rule, `initial_analogue_noise`, which
  flags the first 90 days of every series (91,858 half-hours of the 22 series that the stored panel
  held as clean, all before 2025-06). On the half-hours that carry that reason, `flagged` is the
  stored panel's value, which reproduces the stored rows exactly. The power values are unchanged;**
- `early_cut`: the half-hour ends before the series' `first_usable` date of experiment 103's
  `sites.parquet`; `lockdown`: the local date is 2020-03-23 to 2021-07-19;
- `in_pool`: the series' role is `mw_long`, `mw_short`, or `mva_dark`, the row is not `early_cut`,
  and it is `clean`; `default_rows`: `in_pool` and not `lockdown` and not `switching_mask`;
- `switching_mask`: within 24 hours of a hand-filtered switching event of experiment 41's labelled
  set; the set holds no event after 2026-07, so the mask is empty after that month;
- `dark`: CAMS irradiance below 1 W/m2; rows start at each series' first clean half-hour.

**`sites.parquet` and `series_table.parquet` are copies of the stored files.** They hold each
substation's role, first usable date, coordinates, and effective capacity, which are decisions made
on data to 2026-06-30. The script checks that `load_series()` of experiment 103 still gives the same
series table, and that each series' first clean half-hour equals the stored first clean time.

**The stored and extended tables are compared on every shared (series, time) key** for every column
of the stored table, and the script stops if one differs, printing the count of differences.
"""

from datetime import UTC, datetime
from typing import Final

import e00103_build as build  # ty: ignore[unresolved-import]
import e00103_common as c103  # ty: ignore[unresolved-import]
import e00103_stats as stats103  # ty: ignore[unresolved-import]
import e00701_gam_extend as ext
import holidays
import polars as pl

STORED: Final = c103.WORK_DIR
CLEANED_END: Final = ext.ROWS_END_EXCLUSIVE
CAMS_DIR: Final = ext.REAL_DATA / "studies" / "downloads" / "reanalysis" / "CAMS" / "demand_points"
ROLES_IN_POOL: Final[tuple[str, ...]] = ("mw_long", "mw_short", "mva_dark")
NEW_REASON: Final[str] = "initial_analogue_noise"
"""A drop reason that the cleaned power table gained after experiment 103 built its rows.

It marks the first 90 days of each series. The extended rows keep the stored panel's `flagged` value
on those half-hours, which reproduces every stored row. Every training window and scoring row of the
2026-07 to 2026-10 origins starts after 2024-07, so the rule changes no row the GAM reads."""


def combined_cams() -> pl.DataFrame:
    """Return the stored CAMS table and its extension joined, with one row per (site, time)."""
    files = sorted(CAMS_DIR.glob("cams_demand_points*.parquet"))
    print("CAMS files:", [f.name for f in files])
    frames = [pl.read_parquet(f) for f in files]
    joined = pl.concat(frames, how="vertical_relaxed")
    unique = joined.unique(["site", "time"]).sort("site", "time")
    assert unique.height == joined.height, "the CAMS files overlap or repeat a key"
    stored = frames[0]
    shared = stored.join(unique, on=["site", "time"], suffix="_new")
    assert shared.height == stored.height
    for column in ("ghi_w_m2", "clear_sky_ghi_w_m2"):
        assert (shared[column].fill_null(-1) == shared[f"{column}_new"].fill_null(-1)).all()
    return unique


def restore_stored_flags(*, panel: pl.DataFrame) -> pl.DataFrame:
    """Give the half-hours that carry the new `initial_analogue_noise` reason their stored `flagged` value.

    The stored panel (built 2026-10-19) says whether each of those half-hours was flagged by an
    older rule. Nothing outside the first 90 days of a series carries the new reason, which the
    assertion checks.
    """
    stored = pl.read_parquet(
        STORED / "panel.parquet", columns=["time_series_id", "time", "flagged"]
    )
    new = panel.filter(pl.col("drop_reason") == NEW_REASON)
    last = new["time"].max()
    print(f"the new reason covers {new.height} half-hours of the Primaries, the last ending {last}")
    assert last is not None
    assert isinstance(last, datetime)
    assert last < datetime(2025, 7, 1, tzinfo=UTC)
    joined = panel.join(
        stored.rename({"flagged": "flagged_stored"}), on=["time_series_id", "time"], how="left"
    )
    assert joined.filter(
        (pl.col("drop_reason") == NEW_REASON) & pl.col("flagged_stored").is_null()
    ).is_empty()
    changed = joined.filter(
        (pl.col("drop_reason") == NEW_REASON) & (pl.col("flagged") != pl.col("flagged_stored"))
    )
    print(f"of those, {changed.height} were clean in the stored panel and keep that status here")
    return joined.with_columns(
        flagged=pl.when(pl.col("drop_reason") == NEW_REASON)
        .then(pl.col("flagged_stored"))
        .otherwise(pl.col("flagged"))
    ).drop("flagged_stored")


def build_rows(*, sites: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Rebuild the rows table of the Primaries to 2026-11-14 from the cleaned power and the weather."""
    build.PANEL_END = CLEANED_END  # ty: ignore[invalid-assignment]
    build.ERA5_PATH = ext.ERA5_EXTENDED  # ty: ignore[invalid-assignment]
    build.CAMS_PATH = ext.CAMS_COMBINED  # ty: ignore[invalid-assignment]
    primaries = series.filter(pl.col("site").is_in(list(c103.PRIMARIES)))
    grid = build.half_hour_grid(series=primaries)
    panel = (
        grid.join(build.power_columns(), on=["time_series_id", "time"], how="left")
        .with_columns(
            present=pl.col("power").is_not_null(),
            flagged=pl.col("drop_reason").is_not_null(),
            cams_hour_end=pl.col("time").dt.truncate("1h")
            + pl.duration(hours=pl.when(pl.col("time").dt.minute() == 0).then(0).otherwise(1)),
        )
        .join(build.temperature_columns(grid=grid), on=["time_series_id", "time"], how="left")
        .join(
            build.irradiance_columns(series=primaries),
            on=["time_series_id", "cams_hour_end"],
            how="left",
        )
        .with_columns(
            dark=pl.col("ghi_w_m2") < 1.0,
            local_time=pl.col("time").dt.convert_time_zone(c103.LOCAL_TIME_ZONE),
        )
        .with_columns(
            local_date=(pl.col("local_time") - pl.duration(minutes=30)).dt.date(),
        )
        .drop("local_time")
    )
    panel = restore_stored_flags(panel=panel)
    panel = build.add_switching_mask(panel=panel.drop("drop_reason"))
    panel = panel.with_columns(
        clean=pl.col("present") & ~pl.col("flagged"),
        weekday=pl.col("local_date").dt.weekday(),
        lockdown=stats103.in_lockdown_window(local_date=pl.col("local_date")),
        season=stats103.season_year(date=pl.col("local_date")),
    )
    first_usable = dict(sites.select("time_series_id", "first_usable").iter_rows())
    role_of = dict(sites.select("time_series_id", "role").iter_rows())
    england = holidays.UnitedKingdom(subdiv="ENG", years=range(2019, 2027))
    bank_days = pl.Series(sorted(england.keys())).cast(pl.Date)
    day_type = (
        pl.when(pl.col("local_date").is_in(bank_days.implode()))
        .then(pl.lit("bank_holiday"))
        .when(pl.col("weekday") == 6)
        .then(pl.lit("saturday"))
        .when(pl.col("weekday") == 7)
        .then(pl.lit("sunday"))
        .otherwise(pl.lit("weekday"))
    )
    christmas = (
        (pl.col("local_date").dt.month() == 12) & (pl.col("local_date").dt.day() >= 24)
    ) | ((pl.col("local_date").dt.month() == 1) & (pl.col("local_date").dt.day() == 1))
    panel = (
        panel.with_columns(
            early_cut=pl.col("time")
            < pl.col("time_series_id").replace_strict(
                first_usable, return_dtype=pl.Datetime("us", "UTC")
            ),
            role=pl.col("time_series_id").replace_strict(role_of, return_dtype=pl.String),
            day_type=day_type,
            christmas_run=christmas,
        )
        .with_columns(
            in_pool=pl.col("role").is_in(list(ROLES_IN_POOL))
            & ~pl.col("early_cut")
            & pl.col("clean"),
            scarce_kind=pl.when(
                pl.col("temperature_c").mean().over("time_series_id", "local_date") < 0
            )
            .then(pl.lit("cold_day"))
            .when(pl.col("day_type") == "bank_holiday")
            .then(pl.lit("bank_holiday"))
            .when(pl.col("christmas_run"))
            .then(pl.lit("christmas_run")),
        )
        .with_columns(
            default_rows=pl.col("in_pool") & ~pl.col("lockdown") & ~pl.col("switching_mask")
        )
    )
    first_clean = dict(
        panel.filter(pl.col("clean"))
        .group_by("time_series_id")
        .agg(t=pl.col("time").min())
        .iter_rows()
    )
    stored_cuts = pl.read_parquet(STORED / "tables" / "cuts.parquet")
    for series_id, first in stored_cuts.select("time_series_id", "first_clean").iter_rows():
        if series_id in first_clean:
            assert first_clean[series_id] == first, (
                f"first clean time of {series_id} changed: {first} stored, {first_clean[series_id]} now"
            )
    return (
        panel.filter(pl.col("time") >= pl.col("time_series_id").replace_strict(first_clean))
        .select(
            "time_series_id",
            "site",
            "time",
            "power",
            "present",
            "in_pool",
            "default_rows",
            "early_cut",
            "lockdown",
            "switching_mask",
            "dark",
            "scarce_kind",
            "season",
            "day_type",
            "role",
        )
        .rename({"season": "season_year"})
        .sort("site", "time")
    )


def compare(*, stored: pl.DataFrame, extended: pl.DataFrame) -> None:
    """Assert that every column of every shared row is equal, and print the counts."""
    keys = ["time_series_id", "time"]
    shared = stored.join(extended, on=keys, how="full", suffix="_ext", coalesce=True)
    in_both = shared.filter(pl.col("site").is_not_null() & pl.col("site_ext").is_not_null())
    only_stored = shared.filter(pl.col("site_ext").is_null()).height
    print(
        f"stored rows {stored.height}; extended rows {extended.height}; shared {in_both.height}; stored-only {only_stored}"
    )
    bad_total = 0
    for column in stored.columns:
        if column in keys:
            continue
        a, b = in_both[column], in_both[f"{column}_ext"]
        differ = (a != b).fill_null(False) | (a.is_null() != b.is_null())
        count = int(differ.sum())
        bad_total += count
        print(f"  {column:15s} differences on shared rows: {count}")
    print(
        "  extended rows before 2026-07-01 00:00 not in the stored table:",
        extended.filter(pl.col("time") <= datetime(2026, 7, 1, tzinfo=UTC)).height - in_both.height,
    )
    assert only_stored == 0, "stored rows are missing from the extended table"
    assert bad_total == 0, f"{bad_total} differences on shared rows"


def main() -> None:
    """Write the extended input tables and verify them against the stored ones."""
    ext.INPUTS.mkdir(parents=True, exist_ok=True)
    stored_series = pl.read_parquet(STORED / "series_table.parquet")
    assert c103.load_series().equals(stored_series), "series table changed"
    sites = pl.read_parquet(STORED / "sites.parquet")
    stored_series.write_parquet(ext.INPUTS / "series_table.parquet")
    sites.write_parquet(ext.INPUTS / "sites.parquet")
    combined_cams().write_parquet(ext.CAMS_COMBINED)
    sites_ids = sites.filter(pl.col("time_series_id").is_not_null()).with_columns(
        pl.col("time_series_id").cast(pl.Int32)
    )
    rows = build_rows(sites=sites_ids, series=stored_series)
    stored_rows = pl.read_parquet(STORED / "rows.parquet")
    compare(stored=stored_rows, extended=rows)
    rows.write_parquet(ext.INPUTS / "rows.parquet")
    last = (
        rows.group_by("site")
        .agg(
            last=pl.col("time").max(),
            present=pl.col("present").sum(),
            in_pool=pl.col("in_pool").sum(),
            masked=pl.col("switching_mask")
            .filter(pl.col("time") > datetime(2026, 7, 1, tzinfo=UTC))
            .sum(),
        )
        .sort("site")
    )
    print(last)
    new = rows.filter(pl.col("time") > datetime(2026, 7, 1, tzinfo=UTC))
    print(
        "new rows:",
        new.height,
        "; present",
        int(new["present"].sum()),
        "; in_pool",
        int(new["in_pool"].sum()),
        "; switching_mask",
        int(new["switching_mask"].sum()),
    )
    print(
        "new rows by month, present:",
        new.group_by(pl.col("time").dt.truncate("1mo").alias("m"))
        .agg(pl.col("present").sum())
        .sort("m")
        .rows(),
    )
    print("wrote", ext.INPUTS)


if __name__ == "__main__":
    main()

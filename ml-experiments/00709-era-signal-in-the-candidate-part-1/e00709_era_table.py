r"""Experiment 709, step 1: the table of known eras and the table of known switching events.

The era table (`era_table.parquet`) comes from NGED's Embedded Capacity Register (ECR), read with
`studies.ecr_reader.read_sheet` because `read_ecr` drops the `Date Connected` column, plus the
telemetry dates of the plan. The table holds no customer name. It lists:

- D8's two register connections (a 6 MW gas engine of 2023-06-13 that the register files under
  D14's name, and an 8.91 MW solar farm of 2025-03-24) and the undated 0.7 MW solar row, which is
  *not* an era because it has no date;
- D8's fade from 2026-03, which only telemetry shows and is unresolved;
- every other register connection of 1 MW or more since 2019-09 at the 16 Primaries, as
  `register only`.

The known switching events table (`known_events.parquet`) holds the visible events of the
experiment 41 labelled set (read through `load_labels` of experiment 86) and the verified events
of experiment 580, with a flag for onsets inside the scored window 2025-11-01 to 2026-06-30.

Run with the folders of experiments 86 on `PYTHONPATH`:

    PYTHONPATH=ml-experiments/00086-sw2-12-detector-labels-for-nra-forecast \\
        uv run --no-sync python ml-experiments/00709-*/e00709_era_table.py
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00086_common as common86  # ty: ignore[unresolved-import]
import e00086_labels as labels86  # ty: ignore[unresolved-import]
import e00709_features as features
import numpy as np
import polars as pl
from studies.ecr_reader import ECR_PATH, REGISTER_SHEETS, read_sheet
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ERA_TABLE_PATH: Final[Path] = features.DATA_DIR / "era_table.parquet"
KNOWN_EVENTS_PATH: Final[Path] = features.DATA_DIR / "known_events.parquet"
VERIFIED_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00580-verified-load-shift-event-set-and-label-audit/work/events_verified.parquet"
)
LABEL_ORIGIN: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
"""Origin of the half-hour indices in experiment 580's label tables."""
SCORED_WINDOW: Final[tuple[date, date]] = (date(2025, 11, 1), date(2026, 6, 30))
EXCEL_EPOCH: Final[date] = date(1899, 12, 30)
MIN_ERA_MW: Final[float] = 1.0
FIRST_ERA_DATE: Final[date] = date(2019, 9, 1)
PRIMARY_NAMES: Final[dict[str, str]] = {
    "D5": "Kirton", "D9": "Trusthorpe", "D10": "Spilsby", "D11": "Horncastle",
    "D14": "Mount Bridge Boston", "D15": "Wrangle T1", "D16": "Donington", "D18": "Wrangle T2",
    "D2": "Warth Lane Skegness", "D3": "Sleaford Road Boston", "D4": "Ingoldmells",
    "D8": "Marsh Lane", "D12": "Stickney", "D19": "Chapel St Leonards", "D21": "Alford",
    "D22": "Langrick",
}  # fmt: skip
"""Substring of the register's Primary column that identifies each site (as in experiment 191)."""

D8_GAS_DATE: Final[date] = date(2023, 6, 13)
D8_GAS_TELEMETRY_DATE: Final[date] = date(2023, 8, 15)
D8_SOLAR_DATE: Final[date] = date(2025, 3, 24)
D8_FADE_DATE: Final[date] = date(2026, 3, 1)
"""The month of the fade; only the month is known, and the plan lists it as telemetry only."""
D8_GAS_MW: Final[float] = 6.0
D8_SOLAR_MW: Final[float] = 8.91
D8_SMALL_SOLAR_MW: Final[float] = 0.7
D8_SOLAR_RECOLLECTION_MW: Final[float] = 0.7
"""The maintainer's recollection of a solar farm of about 700 kW in 2025."""

ERA_SCHEMA: Final[dict[str, pl.DataType | type[pl.DataType]]] = {
    "primary": pl.String,
    "era_index": pl.Int32,
    "register_date": pl.Date,
    "telemetry_date": pl.Date,
    "technology": pl.String,
    "size_mw_register": pl.Float64,
    "size_mw_telemetry": pl.Float64,
    "size_mw_recollection": pl.Float64,
    "source": pl.String,
    "confidence": pl.String,
    "note": pl.String,
}


def read_register_assets() -> pl.DataFrame:
    """Read every generation and storage slot under the 16 Primaries, without customer names.

    Returns:
        One row per technology slot with `site`, `technology`, `chp`, `mw`, `status`, and
        `connected_date` (null where the register holds no date).
    """
    pieces = []
    for sheet in REGISTER_SHEETS:
        frame = read_sheet(path=ECR_PATH, sheet_name=sheet)
        for slot in (1, 2, 3):
            pieces.append(  # noqa: PERF401
                frame.select(
                    primary=pl.col("Primary"),
                    source=pl.col(f"Energy Source {slot}"),
                    technology=pl.col(f"Energy Conversion Technology {slot}"),
                    mw=pl.col(
                        f"Energy Source & Energy Conversion Technology {slot} - "
                        "Registered Capacity (MW)"
                    ),
                    status=pl.col("Connection Status"),
                    connected=pl.col("Date Connected"),
                )
            )
    assets = pl.concat(pieces).filter(pl.col("source").is_not_null())
    assets = assets.with_columns(
        mw=pl.col("mw").cast(pl.Float64, strict=False),
        connected_date=(
            pl.lit(EXCEL_EPOCH)
            + pl.duration(days=pl.col("connected").cast(pl.Float64, strict=False).cast(pl.Int64))
        ).cast(pl.Date),
    )
    labelled = [
        assets.filter(pl.col("primary").str.to_lowercase().str.contains(name.lower())).with_columns(
            site=pl.lit(site)
        )
        for site, name in PRIMARY_NAMES.items()
    ]
    return pl.concat(labelled).select(
        "site",
        (pl.col("source") + " / " + pl.col("technology")).alias("technology"),
        "mw",
        "status",
        "connected_date",
    )


def _is_gas_engine() -> pl.Expr:
    text = pl.col("technology").str.to_lowercase()
    return text.str.contains("engine") & (text.str.contains("gas") | text.str.contains("fossil"))


def _is_solar() -> pl.Expr:
    return pl.col("technology").str.to_lowercase().str.contains("solar|photovoltaic")


def d8_rows(*, assets: pl.DataFrame) -> list[dict[str, object]]:
    """Build the era-table rows of D8 and print the register rows that support them.

    The rows come from the plan's dates and sizes. Each one is cross-checked against the workbook,
    and the check is printed, so a mismatch is visible and the table is never silently wrong.

    Args:
        assets: The output of `read_register_assets`.

    Returns:
        The rows for era 1 (gas engine), era 2 (solar farm), the undated solar row, and the fade.
    """
    gas = assets.filter(
        (pl.col("site") == "D14") & _is_gas_engine() & (pl.col("mw") >= D8_GAS_MW - 0.5)
    )
    solar = assets.filter((pl.col("site") == "D8") & _is_solar())
    with pl.Config(tbl_rows=30, fmt_str_lengths=60):
        print("Register rows behind the plan's D8 eras (no customer names):")
        print("Gas engine of about 6 MW listed under D14:")
        print(gas)
        print("Solar rows listed under D8:")
        print(solar)
    found_gas = gas.filter(pl.col("connected_date") == D8_GAS_DATE).height > 0
    found_solar = solar.filter(pl.col("connected_date") == D8_SOLAR_DATE).height > 0
    small = solar.filter(pl.col("connected_date").is_null() & pl.col("mw").is_between(0.6, 0.8))
    print(
        f"cross-check: gas row dated {D8_GAS_DATE} found: {found_gas}; solar row dated "
        f"{D8_SOLAR_DATE} found: {found_solar}; undated 0.7 MW solar row found: {small.height > 0}"
    )
    source = "ECR workbook"
    return [
        {
            "primary": "D8", "era_index": 1, "register_date": D8_GAS_DATE,
            "telemetry_date": D8_GAS_TELEMETRY_DATE, "technology": "gas engine",
            "size_mw_register": D8_GAS_MW, "size_mw_telemetry": None,
            "size_mw_recollection": None, "source": f"{source}, listed under D14's name",
            "confidence": "register date confirmed" if found_gas else "plan value, not re-found",
            "note": "D8's reading stopped including the engine's output on 2023-08-15, 63 days "
            "after the register date (experiment 194). The oracle switches on the telemetry date.",
        },
        {
            "primary": "D8", "era_index": 2, "register_date": D8_SOLAR_DATE,
            "telemetry_date": None, "technology": "solar farm",
            "size_mw_register": D8_SOLAR_MW, "size_mw_telemetry": None,
            "size_mw_recollection": D8_SOLAR_RECOLLECTION_MW, "source": source,
            "confidence": "register date confirmed" if found_solar else "plan value, not re-found",
            "note": "Telemetry date and size are measured in stage 0. The maintainer recalls about "
            "700 kW, the dated register row is 8.91 MW.",
        },
        {
            "primary": "D8", "era_index": None, "register_date": None, "telemetry_date": None,
            "technology": "solar", "size_mw_register": D8_SMALL_SOLAR_MW,
            "size_mw_telemetry": None, "size_mw_recollection": None, "source": source,
            "confidence": "unknown date",
            "note": "Unknown: the register row has no connection date, so it cannot define an era.",
        },
        {
            "primary": "D8", "era_index": None, "register_date": None,
            "telemetry_date": D8_FADE_DATE, "technology": "unknown", "size_mw_register": None,
            "size_mw_telemetry": None, "size_mw_recollection": None,
            "source": "telemetry only (experiment 697)", "confidence": "unresolved",
            "note": "A fade from 2026-03 that no register row explains; the date is the month.",
        },
    ]  # fmt: skip


def other_register_rows(*, assets: pl.DataFrame) -> list[dict[str, object]]:
    """Build `register only` rows for connections of 1 MW or more since 2019-09 at other Primaries.

    Args:
        assets: The output of `read_register_assets`.

    Returns:
        One row per connection, numbered by date within each Primary. The 6 MW gas engine that the
        plan assigns to D8 is left out of D14's rows.
    """
    kept = assets.filter(
        (pl.col("site") != "D8")
        & (pl.col("status") == "Connected")
        & (pl.col("mw") >= MIN_ERA_MW)
        & (pl.col("connected_date") >= FIRST_ERA_DATE)
    ).filter(
        ~((pl.col("site") == "D14") & (pl.col("connected_date") == D8_GAS_DATE) & _is_gas_engine())
    )
    kept = kept.sort("site", "connected_date").with_columns(
        era_index=pl.int_range(pl.len(), dtype=pl.Int32).over("site") + 1
    )
    return [
        {
            "primary": row["site"], "era_index": row["era_index"],
            "register_date": row["connected_date"], "telemetry_date": None,
            "technology": row["technology"], "size_mw_register": row["mw"],
            "size_mw_telemetry": None, "size_mw_recollection": None, "source": "ECR workbook",
            "confidence": "register only", "note": "",
        }
        for row in kept.iter_rows(named=True)
    ]  # fmt: skip


def build_era_table() -> pl.DataFrame:
    """Build the era table from the register.

    Returns:
        The table with the columns of `ERA_SCHEMA`.
    """
    assets = read_register_assets()
    rows = d8_rows(assets=assets) + other_register_rows(assets=assets)
    return pl.DataFrame(rows, schema=ERA_SCHEMA)


def _runs(*, flag: np.ndarray) -> list[tuple[int, int]]:
    """Return (first index, last index) of each run of True in a boolean array."""
    padded = np.concatenate([[False], flag, [False]]).astype(np.int8)
    change = np.diff(padded)
    starts, stops = np.flatnonzero(change == 1), np.flatnonzero(change == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]


def _event_row(
    *, source: str, event_id: str, site: str, other: str | None, first: int, last: int
) -> dict[str, object]:
    onset = LABEL_ORIGIN + timedelta(minutes=30 * int(first))
    end = LABEL_ORIGIN + timedelta(minutes=30 * (int(last) + 1))
    return {
        "source": source,
        "event_id": event_id,
        "site": site,
        "other_site": other,
        "onset_utc": onset,
        "end_utc": end,
        "duration_days": (end - onset) / timedelta(days=1),
        "in_scored_window": SCORED_WINDOW[0] <= onset.date() <= SCORED_WINDOW[1],
    }


def labelled_events() -> list[dict[str, object]]:
    """Return the visible events of the experiment 41 labelled set as one row per Primary and run.

    Returns:
        Rows from the runs of `in_event` of each Primary that `load_labels` builds.
    """
    series = common86.series_table()
    labels = labels86.load_labels(series=series)
    rows = []
    for site in features.PRIMARIES:
        if site not in labels:
            continue
        for k, (first, last) in enumerate(_runs(flag=labels[site].in_event)):
            rows.append(
                _event_row(
                    source="experiment 41 labelled set",
                    event_id=f"{site}-{k}",
                    site=site,
                    other=None,
                    first=first,
                    last=last,
                )
            )
    return rows


def verified_events() -> list[dict[str, object]]:
    """Return the verified events of experiment 580, one row per transfer and side.

    Returns:
        Rows for side a and side b of each verified transfer that has an onset and an end.
    """
    table = pl.read_parquet(VERIFIED_PATH)
    rows = []
    for row in table.iter_rows(named=True):
        for side, other in (("a", "b"), ("b", "a")):
            first, last = row[f"onset_{side}_idx"], row[f"end_{side}_idx"]
            if first is None or last is None:
                continue
            rows.append(
                _event_row(
                    source="experiment 580 verified",
                    event_id=f"{row['transfer_id']}{side}",
                    site=row[f"primary_{side}"],
                    other=row[f"primary_{other}"],
                    first=first,
                    last=last - 1,
                )
            )
    return rows


def build_known_events() -> pl.DataFrame:
    """Build the table of known switching events from experiments 41 and 580.

    Returns:
        One row per event and side, with the columns `source`, `event_id`, `site`, `other_site`,
        `onset_utc`, `end_utc`, `duration_days`, and `in_scored_window`.
    """
    schema = {
        "source": pl.String, "event_id": pl.String, "site": pl.String, "other_site": pl.String,
        "onset_utc": pl.Datetime("us", "UTC"), "end_utc": pl.Datetime("us", "UTC"),
        "duration_days": pl.Float64, "in_scored_window": pl.Boolean,
    }  # fmt: skip
    return pl.DataFrame(labelled_events() + verified_events(), schema=schema)


def print_known_events(*, events: pl.DataFrame) -> None:
    """Print how many known switching events each source holds and how many start in the window.

    Args:
        events: The output of `build_known_events`.
    """
    verified = events.filter(pl.col("source") == "experiment 580 verified").filter(
        pl.col("event_id").str.ends_with("a")
    )
    labelled = events.filter(pl.col("source") == "experiment 41 labelled set")
    print(
        f"experiment 41 labelled set: n={labelled.height} visible runs of events at the Primaries, "
        f"{labelled['in_scored_window'].sum()} with onsets in "
        f"{SCORED_WINDOW[0]}..{SCORED_WINDOW[1]}"
    )
    print(
        f"experiment 580 verified set: n={verified.height} transfers, "
        f"{verified['in_scored_window'].sum()} with onsets in "
        f"{SCORED_WINDOW[0]}..{SCORED_WINDOW[1]}"
    )
    print(
        events.group_by("source", "in_scored_window")
        .agg(n=pl.len(), median_duration_days=pl.col("duration_days").median())
        .sort("source", "in_scored_window")
    )


def main() -> None:
    """Build, print, and store the era table and the known switching events."""
    features.DATA_DIR.mkdir(parents=True, exist_ok=True)
    eras = build_era_table()
    with pl.Config(tbl_rows=60, tbl_cols=11, fmt_str_lengths=50):
        print(eras)
    eras.write_parquet(ERA_TABLE_PATH)
    events = build_known_events()
    print_known_events(events=events)
    events.write_parquet(KNOWN_EVENTS_PATH)


if __name__ == "__main__":
    main()

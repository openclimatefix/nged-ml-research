r"""Download the extra free signals for experiment 697's relationship panels, with checkpoints.

    uv run --no-sync python $E/00697-*/e00697_signals_download.py [--only NAME ...] [--max-chunks N]

Every signal is free and open, with no login. Each request chunk is written to its own file under
`signals/_cache/<name>/` before the next chunk is requested, and a chunk that is already cached is
skipped, so a crash costs one chunk. The combined table of each signal is written to
`signals/<name>.parquet`, validated (gaps, duplicates, time convention, range), and described in
`signals/README.md` with a `lineage.json`. **Every half-hourly table is keyed on the half-hour
start in UTC (`start`)**, the convention of the experiment's frames.

The signals, added after the experiment's design and not pre-registered as separate questions:

- `apx_mid`: Elexon Market Index Data, APX provider only (`APXMIDP`), from 2023-06. The N2EX
  provider (`N2EXMIDP`) returns 0.00 in 2023 and is not used.
- `stor_windows` and `stor_auction`: NESO Short Term Operating Reserve (STOR) committed windows,
  and the day-ahead auction results with the unit identifier and company name dropped.
- `imbalngc`: Elexon indicated imbalance forecast, national boundary, one vintage per day (the
  publication nearest 09:45 UTC), covering that day's later half-hours and the early hours after.
- `demand_forecast`: NESO day-ahead demand forecast archive (cardinal points), from 2023-06.
- `historic_demand`: NESO half-hourly national demand, embedded wind, and embedded solar, 2023 to
  2026.
- `fuelhh_wind`: Elexon FUELHH transmission wind, 2023-06 to 2024-06, filling the gap before the
  held wind table.
- `bank_holidays`: gov.uk bank holidays for England and Wales.
- `syswarn`: Elexon system warnings (margin, capacity market, and demand flexibility notices).
- `carbon_intensity`: the national Carbon Intensity API, forecast and actual.
"""

import argparse
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl
import requests
from e00697_common import DATA_DIR, LOCAL_TZ  # ty: ignore[unresolved-import]

SIGNALS_DIR: Final[Path] = DATA_DIR / "signals"
CACHE_DIR: Final[Path] = SIGNALS_DIR / "_cache"
FIRST_DAY: Final[date] = date(2023, 6, 1)
LAST_DAY: Final[date] = date(2026, 7, 1)
"""The day after the last day to fetch."""
FUELHH_LAST_DAY: Final[date] = date(2024, 7, 1)
"""FUELHH is fetched only up to the start of the held national wind table (2024-07)."""
WORKERS: Final[int] = 4
"""Requests in flight at once; each also sleeps `SLEEP_SECONDS` after its reply."""
SLEEP_SECONDS: Final[float] = 1.0
TIMEOUT_SECONDS: Final[int] = 60
MAX_TRIES: Final[int] = 4
ELEXON: Final[str] = "https://data.elexon.co.uk/bmrs/api/v1"
NESO: Final[str] = "https://api.neso.energy/api/3/action"
MID_WINDOW_DAYS: Final[int] = 7
SYSWARN_WINDOW_DAYS: Final[int] = 100
CARBON_WINDOW_DAYS: Final[int] = 13
"""The Carbon Intensity API allows 14 days a request."""
PAGE_SIZE: Final[int] = 32000
STOR_WINDOWS_ID: Final[str] = "90d4a1a4-d311-4a22-9138-84cb7ead2599"
STOR_AUCTION_ID: Final[str] = "844a1a65-0a3c-4704-b327-d1685d0c397b"
DEMAND_FORECAST_ID: Final[str] = "9847e7bb-986e-49be-8138-717b25933fbb"
DEMAND_PACKAGE: Final[str] = "historic-demand-data"
DEMAND_YEARS: Final[tuple[int, ...]] = (2023, 2024, 2025, 2026)
BANK_HOLIDAY_URL: Final[str] = "https://www.gov.uk/bank-holidays.json"
CARBON_URL: Final[str] = "https://api.carbonintensity.org.uk/intensity"
IMBALANCE_BOUNDARY: Final[str] = "N"
IMBALANCE_PUBLISH_FROM: Final[str] = "09:30"
IMBALANCE_PUBLISH_TO: Final[str] = "10:00"
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
STOR_DROPPED_COLUMNS: Final[tuple[str, ...]] = ("Unit ID", "Company Name")
"""A unit name and its company are never kept, because nothing maps a unit to D8's location."""
BYTE_BUDGET: Final[int] = 500_000_000
"""Stop and report if one signal's cache holds more than this many bytes."""


def utc_text(*, moment: datetime) -> str:
    """Return a UTC instant in the form the Elexon API accepts."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def get_json(*, url: str, params: dict[str, Any] | None = None) -> Any:
    """Return the JSON reply of a GET, retrying a failure a few times, then sleeping politely."""
    last: Exception | None = None
    for attempt in range(MAX_TRIES):
        try:
            reply = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
            reply.raise_for_status()
            time.sleep(SLEEP_SECONDS)
            return reply.json()
        except (requests.RequestException, ValueError) as error:
            last = error
            time.sleep(SLEEP_SECONDS * 2 ** (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {MAX_TRIES} tries: {last}")


def cached_chunk(*, name: str, key: str, fetch: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    """Return the table of one request chunk, fetching it only if it is not cached.

    The table is written to a `.partial` file and renamed, so a crash never leaves a truncated file
    that a later run would take as complete.
    """
    directory = CACHE_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    frame = fetch()
    partial = path.with_suffix(".parquet.partial")
    frame.write_parquet(partial)
    partial.rename(path)
    total = sum(f.stat().st_size for f in directory.glob("*.parquet"))
    if total > BYTE_BUDGET:
        raise SystemExit(
            f"{name}: the cache holds {total} bytes, above the budget; stop and report"
        )
    return frame


def day_windows(*, first: date, last: date, days: int) -> list[tuple[date, date]]:
    """Split `first` up to `last` (exclusive) into consecutive windows of at most `days` days."""
    out, start = [], first
    while start < last:
        end = min(start + timedelta(days=days), last)
        out.append((start, end))
        start = end
    return out


def at_midnight(*, day: date) -> datetime:
    """Return midnight UTC at the start of a date."""
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def settlement_start(*, date_column: str, period_column: str) -> pl.Expr:
    """Return the UTC start of a settlement period given its local date and period number.

    Period 1 starts at local midnight and each period lasts 30 minutes, so on a clock-change day a
    period count of 46 or 50 is handled by counting from the local day's first instant.
    """
    midnight = (
        pl.col(date_column)
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone(LOCAL_TZ, ambiguous="earliest", non_existent="raise")
    )
    return midnight.dt.convert_time_zone("UTC") + pl.duration(
        minutes=(pl.col(period_column) - 1) * 30
    )


def chunked(
    *,
    name: str,
    windows: list[tuple[date, date]],
    fetch: Callable[[date, date], pl.DataFrame],
    max_chunks: int | None,
) -> pl.DataFrame:
    """Fetch every window of a signal through the cache and return the stacked table."""
    todo = windows if max_chunks is None else windows[:max_chunks]

    def one(window: tuple[date, date]) -> pl.DataFrame:
        start, end = window
        return cached_chunk(
            name=name, key=f"{start}_{end}", fetch=lambda s=start, e=end: fetch(s, e)
        )

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        parts = []
        for i, part in enumerate(pool.map(one, todo)):
            parts.append(part)
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                print(f"{name}: {i + 1} of {len(windows)} chunks done", flush=True)
    return pl.concat([p for p in parts if p.height], how="vertical_relaxed")


# --------------------------------------------------------------------------- the signals


def apx_mid(*, max_chunks: int | None) -> pl.DataFrame:
    """Return the APX day-ahead price index (GBP per MWh) and volume (MWh) by half-hour start."""

    def fetch(start: date, end: date) -> pl.DataFrame:
        reply = get_json(
            url=f"{ELEXON}/balancing/pricing/market-index",
            params={
                "from": utc_text(moment=at_midnight(day=start)),
                "to": utc_text(moment=at_midnight(day=end)),
                "dataProviders": "APXMIDP",
                "format": "json",
            },
        )
        rows = reply["data"]
        return pl.DataFrame(
            rows,
            schema={"startTime": pl.String, "price": pl.Float64, "volume": pl.Float64},
            strict=False,
        )

    windows = day_windows(first=FIRST_DAY - timedelta(days=1), last=LAST_DAY, days=MID_WINDOW_DAYS)
    frame = chunked(name="apx_mid", windows=windows, fetch=fetch, max_chunks=max_chunks)
    return (
        frame.select(
            start=pl.col("startTime").str.to_datetime(time_zone="UTC", time_unit="us"),
            apx_price=pl.col("price"),
            apx_volume=pl.col("volume"),
        )
        .unique(subset="start", keep="first", maintain_order=True)
        .sort("start")
    )


def neso_records(*, resource_id: str, filters: str | None = None) -> list[dict]:
    """Return every record of a NESO datastore resource, paging by offset."""
    out: list[dict] = []
    offset = 0
    while True:
        params: dict[str, Any] = {
            "resource_id": resource_id,
            "limit": PAGE_SIZE,
            "offset": offset,
        }
        if filters:
            params["filters"] = filters
        result = get_json(url=f"{NESO}/datastore_search", params=params)["result"]
        rows = result["records"]
        out.extend(rows)
        offset += len(rows)
        if not rows or offset >= result["total"]:
            return out


def neso_table(*, name: str, resource_id: str, max_chunks: int | None) -> pl.DataFrame:
    """Fetch one NESO resource in a single cached chunk (the table is small) and return it."""
    del max_chunks
    return cached_chunk(
        name=name,
        key="all",
        fetch=lambda: pl.DataFrame(neso_records(resource_id=resource_id), infer_schema_length=None),
    )


def stor_windows(*, max_chunks: int | None) -> pl.DataFrame:
    """Return the committed STOR windows per date, from 2023-06, weekday or non-working day."""
    frame = neso_table(name="stor_windows", resource_id=STOR_WINDOWS_ID, max_chunks=max_chunks)
    return (
        frame.rename({c: c.lower().replace(" ", "_") for c in frame.columns})
        .with_columns(date=pl.col("date").str.to_date())
        .filter(pl.col("date") >= FIRST_DAY)
        .drop("_id")
        .sort("date")
    )


def stor_auction(*, max_chunks: int | None) -> pl.DataFrame:
    """Return day-ahead STOR auction results from 2023-06 without the unit and company names."""
    frame = neso_table(name="stor_auction", resource_id=STOR_AUCTION_ID, max_chunks=max_chunks)
    keep = frame.drop([c for c in STOR_DROPPED_COLUMNS if c in frame.columns])
    return (
        keep.rename({c: c.lower().replace(" ", "_") for c in keep.columns})
        .drop("_id")
        .filter(pl.col("service_delivery_from") >= str(FIRST_DAY))
    )


def imbalngc(*, max_chunks: int | None) -> pl.DataFrame:
    """Return one national imbalance forecast vintage per day (the one published near 09:45 UTC)."""

    def fetch(start: date, end: date) -> pl.DataFrame:
        del end
        reply = get_json(
            url=f"{ELEXON}/datasets/IMBALNGC",
            params={
                "publishDateTimeFrom": f"{start}T{IMBALANCE_PUBLISH_FROM}Z",
                "publishDateTimeTo": f"{start}T{IMBALANCE_PUBLISH_TO}Z",
                "format": "json",
            },
        )
        rows = reply["data"] if isinstance(reply, dict) else reply
        schema = {
            "imbalance": pl.Float64,
            "publishTime": pl.String,
            "startTime": pl.String,
            "boundary": pl.String,
        }
        return pl.DataFrame(
            [{k: r[k] for k in schema} for r in rows if r["boundary"] == IMBALANCE_BOUNDARY],
            schema=schema,
        )

    windows = [
        (d, d + timedelta(days=1))
        for d in (FIRST_DAY + timedelta(days=i) for i in range((LAST_DAY - FIRST_DAY).days))
    ]
    frame = chunked(name="imbalngc", windows=windows, fetch=fetch, max_chunks=max_chunks)
    return frame.select(
        start=pl.col("startTime").str.to_datetime(time_zone="UTC", time_unit="us"),
        published=pl.col("publishTime").str.to_datetime(time_zone="UTC", time_unit="us"),
        imbalance_mw=pl.col("imbalance"),
    ).sort("start")


def demand_forecast(*, max_chunks: int | None) -> pl.DataFrame:
    """Return NESO's day-ahead demand forecast by cardinal point, from 2023-06."""
    frame = neso_table(
        name="demand_forecast", resource_id=DEMAND_FORECAST_ID, max_chunks=max_chunks
    )
    return (
        frame.with_columns(
            target_date=pl.col("TARGETDATE").str.to_date(),
            forecast_timestamp=pl.col("FORECAST_TIMESTAMP").str.to_datetime(),
        )
        .filter(pl.col("target_date") >= FIRST_DAY)
        .select(
            "target_date",
            days_ahead=pl.col("DAYSAHEAD"),
            cardinal_point=pl.col("CARDINALPOINT"),
            point_type=pl.col("CP_TYPE"),
            start_hhmm=pl.col("CP_ST_TIME"),
            end_hhmm=pl.col("CP_END_TIME"),
            forecast_demand_mw=pl.col("FORECASTDEMAND"),
            forecast_timestamp=pl.col("forecast_timestamp"),
        )
    )


def historic_demand(*, max_chunks: int | None) -> pl.DataFrame:
    """Return national demand, embedded wind, and embedded solar by half-hour start."""
    package = get_json(url=f"{NESO}/package_show", params={"id": DEMAND_PACKAGE})["result"]
    ids = {
        int(r["name"].split()[-1]): r["id"]
        for r in package["resources"]
        if r["name"].startswith("Historic Demand Data ") and r["name"].split()[-1].isdigit()
    }
    parts = []
    for year in DEMAND_YEARS:
        if max_chunks is not None and len(parts) >= max_chunks:
            break
        frame = cached_chunk(
            name="historic_demand",
            key=str(year),
            fetch=lambda y=year: pl.DataFrame(
                neso_records(resource_id=ids[y]), infer_schema_length=None
            ),
        )
        parts.append(frame)
        print(f"historic_demand: {year} done", flush=True)
    frame = pl.concat(parts, how="diagonal_relaxed")
    return (
        frame.with_columns(
            date=pl.coalesce(
                pl.col("SETTLEMENT_DATE")
                .cast(pl.String)
                .str.slice(0, 10)
                .str.to_date("%Y-%m-%d", strict=False),
                pl.col("SETTLEMENT_DATE").cast(pl.String).str.to_date("%d-%b-%y", strict=False),
            )
        )
        .with_columns(start=settlement_start(date_column="date", period_column="SETTLEMENT_PERIOD"))
        .select(
            "start",
            national_demand_mw=pl.col("ND"),
            transmission_demand_mw=pl.col("TSD"),
            embedded_wind_mw=pl.col("EMBEDDED_WIND_GENERATION"),
            embedded_solar_mw=pl.col("EMBEDDED_SOLAR_GENERATION"),
        )
        .filter(
            pl.col("start") >= at_midnight(day=FIRST_DAY),
            pl.col("start") < at_midnight(day=LAST_DAY),
        )
        .unique(subset="start", keep="first", maintain_order=True)
        .sort("start")
    )


def fuelhh_wind(*, max_chunks: int | None) -> pl.DataFrame:
    """Return transmission-connected wind generation (MW) by half-hour start, 2023-06 to 2024-06."""

    def fetch(start: date, end: date) -> pl.DataFrame:
        reply = get_json(
            url=f"{ELEXON}/datasets/FUELHH/stream",
            params={
                "settlementDateFrom": str(start),
                "settlementDateTo": str(end - timedelta(days=1)),
                "fuelType": "WIND",
            },
        )
        rows = reply if isinstance(reply, list) else reply["data"]
        return pl.DataFrame(
            {
                "startTime": [r["startTime"] for r in rows],
                "generation": [r["generation"] for r in rows],
            },
            schema={"startTime": pl.String, "generation": pl.Float64},
        )

    windows = day_windows(first=FIRST_DAY, last=FUELHH_LAST_DAY, days=31)
    frame = chunked(name="fuelhh_wind", windows=windows, fetch=fetch, max_chunks=max_chunks)
    return (
        frame.select(
            start=pl.col("startTime").str.to_datetime(time_zone="UTC", time_unit="us"),
            fuelhh_wind_mw=pl.col("generation"),
        )
        .unique(subset="start", keep="first", maintain_order=True)
        .sort("start")
    )


def bank_holidays(*, max_chunks: int | None) -> pl.DataFrame:
    """Return England and Wales bank holiday dates from gov.uk."""
    del max_chunks

    def fetch() -> pl.DataFrame:
        reply = get_json(url=BANK_HOLIDAY_URL)
        events = reply["england-and-wales"]["events"]
        return pl.DataFrame(
            {"date": [e["date"] for e in events], "title": [e["title"] for e in events]}
        )

    return cached_chunk(name="bank_holidays", key="all", fetch=fetch).with_columns(
        date=pl.col("date").str.to_date()
    )


def syswarn(*, max_chunks: int | None) -> pl.DataFrame:
    """Return Elexon system warnings: publication time, type, and text length only."""

    def fetch(start: date, end: date) -> pl.DataFrame:
        reply = get_json(
            url=f"{ELEXON}/datasets/SYSWARN",
            params={
                "publishDateTimeFrom": utc_text(moment=at_midnight(day=start)),
                "publishDateTimeTo": utc_text(moment=at_midnight(day=end)),
                "format": "json",
            },
        )
        rows = reply["data"] if isinstance(reply, dict) else reply
        return pl.DataFrame(
            {
                "published": [r.get("publishTime") for r in rows],
                "warning_type": [r.get("warningType") for r in rows],
                "warning_text": [r.get("warningText") for r in rows],
            },
            schema={"published": pl.String, "warning_type": pl.String, "warning_text": pl.String},
        )

    windows = day_windows(first=FIRST_DAY, last=LAST_DAY, days=SYSWARN_WINDOW_DAYS)
    frame = chunked(name="syswarn", windows=windows, fetch=fetch, max_chunks=max_chunks)
    return (
        frame.with_columns(
            published=pl.col("published").str.to_datetime(time_zone="UTC", time_unit="us")
        )
        .unique(subset=["published", "warning_type"], keep="first", maintain_order=True)
        .sort("published")
    )


def carbon_intensity(*, max_chunks: int | None) -> pl.DataFrame:
    """Return the national carbon intensity (gCO2 per kWh), forecast and actual."""

    def fetch(start: date, end: date) -> pl.DataFrame:
        reply = get_json(
            url=f"{CARBON_URL}/{start}T00:00Z/{end}T00:00Z",
        )
        rows = reply["data"]
        return pl.DataFrame(
            {
                "start": [r["from"] for r in rows],
                "forecast": [r["intensity"]["forecast"] for r in rows],
                "actual": [r["intensity"]["actual"] for r in rows],
            },
            schema={"start": pl.String, "forecast": pl.Float64, "actual": pl.Float64},
        )

    windows = day_windows(first=FIRST_DAY, last=LAST_DAY, days=CARBON_WINDOW_DAYS)
    frame = chunked(name="carbon_intensity", windows=windows, fetch=fetch, max_chunks=max_chunks)
    return (
        frame.select(
            start=pl.col("start").str.to_datetime(time_zone="UTC", time_unit="us"),
            carbon_forecast=pl.col("forecast"),
            carbon_actual=pl.col("actual"),
        )
        .unique(subset="start", keep="first", maintain_order=True)
        .sort("start")
    )


SIGNALS: Final = {
    "apx_mid": apx_mid,
    "stor_windows": stor_windows,
    "stor_auction": stor_auction,
    "imbalngc": imbalngc,
    "demand_forecast": demand_forecast,
    "historic_demand": historic_demand,
    "fuelhh_wind": fuelhh_wind,
    "bank_holidays": bank_holidays,
    "syswarn": syswarn,
    "carbon_intensity": carbon_intensity,
}
HALF_HOURLY: Final[tuple[str, ...]] = (
    "apx_mid",
    "historic_demand",
    "fuelhh_wind",
    "carbon_intensity",
)
"""Signals that should hold a row for every half-hour start."""
SOURCES: Final[dict[str, str]] = {
    "apx_mid": "Elexon BMRS Market Index Data, https://bmrs.elexon.co.uk/market-index-data",
    "stor_windows": "NESO Data Portal, STOR committed windows, https://www.neso.energy/data-portal",
    "stor_auction": "NESO Data Portal, STOR day-ahead auction results",
    "imbalngc": "Elexon BMRS dataset IMBALNGC (indicated imbalance), https://bmrs.elexon.co.uk",
    "demand_forecast": "NESO Data Portal, day-ahead demand forecast archive",
    "historic_demand": "NESO Data Portal, Historic Demand Data (one resource per year)",
    "fuelhh_wind": "Elexon BMRS dataset FUELHH, fuel type WIND",
    "bank_holidays": "https://www.gov.uk/bank-holidays (Open Government Licence)",
    "syswarn": "Elexon BMRS dataset SYSWARN (system warnings)",
    "carbon_intensity": "https://carbonintensity.org.uk (CC BY 4.0)",
}


def validate(*, name: str, frame: pl.DataFrame, partial: bool) -> dict[str, object]:
    """Return computed facts: rows, range, duplicates, half-hour gaps, nulls per column.

    A partial run (a few chunks) skips the gap count, because its range is not the full range.
    """
    facts: dict[str, object] = {
        "rows": frame.height,
        "columns": {c: str(t) for c, t in frame.schema.items()},
    }
    facts["nulls"] = {
        c: int(n) for c, n in zip(frame.columns, frame.null_count().row(0), strict=True)
    }
    if "start" in frame.columns:
        starts = frame["start"]
        facts["first_start_utc"] = str(starts.min())
        facts["last_start_utc"] = str(starts.max())
        facts["duplicate_starts"] = int(starts.is_duplicated().sum())
        facts["off_half_hour_starts"] = int(
            frame.filter(
                (pl.col("start").dt.minute() % 30 != 0) | (pl.col("start").dt.second() != 0)
            ).height
        )
        if name in HALF_HOURLY and not partial:
            expected = int((starts.max() - starts.min()) / HALF_HOUR) + 1  # ty: ignore[unsupported-operator]
            facts["expected_rows"] = expected
            facts["missing_half_hours"] = expected - int(starts.n_unique())
    numeric = [c for c, t in frame.schema.items() if t.is_float() or t.is_integer()]
    facts["range"] = {c: [frame[c].min(), frame[c].max()] for c in numeric}
    return facts


def write_readme(*, facts: dict[str, dict], partial: bool) -> None:
    """Write `README.md` and `lineage.json`, with every fact taken from the written tables."""
    lineage = {
        "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "partial_run": partial,
        "sources": {n: SOURCES[n] for n in facts},
        "facts": facts,
    }
    (SIGNALS_DIR / "lineage.json").write_text(json.dumps(lineage, indent=1, default=str))
    lines = [
        "# Extra free signals for experiment 697",
        "",
        f"Written by `{Path(__file__).name}`. All data are free and open with no login.",
        "The signals were added after the experiment's design.",
        "They are not pre-registered questions.",
        "",
        "**Time convention.** Every half-hourly table has `start`, the UTC start of the half-hour.",
        "NESO settlement dates and periods become starts by counting",
        "30-minute periods from local midnight.",
        "",
        "**Unit names.** The STOR auction table has its unit and company columns dropped.",
        "A unit is never mapped to D8.",
        "",
        "**The imbalance forecast** keeps the national boundary and one vintage a day:",
        "the publication between 09:30 and 10:00 UTC, which covers that day's later",
        "half-hours and the early hours of the next day.",
        "",
    ]
    for name, fact in facts.items():
        lines += [f"## `{name}`", "", f"Source: {SOURCES[name]}.", ""]
        lines += [f"- {key}: {value}" for key, value in fact.items()]
        lines.append("")
    (SIGNALS_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    """Fetch the chosen signals, write and validate each, and write the README and lineage note."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(SIGNALS), choices=list(SIGNALS))
    parser.add_argument(
        "--max-chunks", type=int, default=None, help="Fetch only this many chunks per signal."
    )
    args = parser.parse_args()
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    partial = args.max_chunks is not None
    facts: dict[str, dict] = {}
    for name in args.only:
        frame = SIGNALS[name](max_chunks=args.max_chunks)
        suffix = "_partial" if partial else ""
        frame.write_parquet(SIGNALS_DIR / f"{name}{suffix}.parquet")
        facts[name] = validate(name=name, frame=frame, partial=partial)
        print(f"{name}: {facts[name]}", flush=True)
    if partial:
        return
    stored = SIGNALS_DIR / "lineage.json"
    earlier = json.loads(stored.read_text())["facts"] if stored.exists() else {}
    write_readme(facts={**earlier, **facts}, partial=partial)


if __name__ == "__main__":
    main()

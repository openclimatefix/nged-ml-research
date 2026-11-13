r"""Extra relationship panels: D8's running share against the signals fetched after the design.

    uv run --no-sync python $E/00697-*/e00697_signals.py

**These panels are descriptive. The signals were added after the design, were not pre-registered as
separate questions, and none of them changes the Q1 verdict rule.** For each signal the script
joins the signal to the half-hour frames on the half-hour start, takes the top third minus the
bottom third of the signal within a cell (local half-hour, weekday class, calendar month), and
draws that difference of running shares with the experiment's week-block interval, for D8's
smoothed flags in the net and solar eras and for the null control's labels. A signal that the
injected control engine cannot respond to should show no pattern in the control. A pattern in D8
that the control lacks is a coincidence of timing and is not read as a cause. In particular, the
engine's holding of a STOR or Capacity Market agreement is unknown, so a match with a STOR window is
only a coincidence of timing.

Capacity Market notices and system warnings form a descriptive panel with few notice days, so
the panel's intervals are wide, and groups under 20 days are not drawn. The page describes the
engine only as a gas engine with a long-term capacity contract.

The figure holds no date, no unit name, and no series. Numbers go to `printed/signals_checks.txt`.
"""

from typing import Final

import altair as alt
import polars as pl
from e00697_analysis import (  # ty: ignore[unresolved-import]
    cell_bins,
    stable_seed,
    two_group_share,
    usable,
    with_state,
)
from e00697_common import (  # ty: ignore[unresolved-import]
    EVENING_FIRST,
    EVENING_LAST,
    MIN_DAYS_PER_BIN,
    PRINTED_DIR,
    Estimate,
)
from e00697_drivers import bank_holiday_dates  # ty: ignore[unresolved-import]
from e00697_figures import BLUE, ORANGE, SKY, host_null_frame, save  # ty: ignore[unresolved-import]
from e00697_frame import load_frame  # ty: ignore[unresolved-import]
from e00697_signals_download import SIGNALS_DIR  # ty: ignore[unresolved-import]

HALF_HOURLY_SIGNALS: Final[dict[str, tuple[str, str]]] = {
    "apx_price": ("apx_mid", "Within-day price (APX index)"),
    "imbalance_mw": ("imbalngc", "Forecast imbalance (same-day vintage)"),
    "national_demand_mw": ("historic_demand", "National demand"),
    "embedded_wind_mw": ("historic_demand", "Embedded wind"),
    "embedded_solar_mw": ("historic_demand", "Embedded solar"),
    "fuelhh_wind_mw": ("fuelhh_wind", "Transmission wind (first year)"),
    "carbon_actual": ("carbon_intensity", "Carbon intensity"),
}
"""Signal column to its table and its panel title."""
WARNING_TYPES: Final[tuple[str, ...]] = ("MARGIN", "CAPACITY MARKET")
"""A system warning whose type contains one of these marks a day of system stress."""
SERIES: Final[dict[str, str]] = {
    "net": "D8 net era",
    "solar": "D8 solar era",
    "null": "Null control on host series",
}
COLOURS: Final[list[str]] = [BLUE, SKY, ORANGE]


def read_signal(*, name: str) -> pl.DataFrame:
    """Return a stored signal table."""
    return pl.read_parquet(SIGNALS_DIR / f"{name}.parquet")


def attach(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Join every signal to a half-hour frame: by start, or by local date for daily signals."""
    out = frame
    for name in ("apx_mid", "imbalngc", "historic_demand", "fuelhh_wind", "carbon_intensity"):
        table = read_signal(name=name)
        keep = [c for c in table.columns if c in {"start", *HALF_HOURLY_SIGNALS}]
        out = out.join(table.select(keep), on="start", how="left", maintain_order="left")
    forecast = read_signal(name="demand_forecast")
    daily = (
        forecast.filter(pl.col("days_ahead") == 1)
        .group_by("target_date")
        .agg(forecast_peak_mw=pl.col("forecast_demand_mw").max())
        .rename({"target_date": "local_date"})
    )
    out = out.join(daily, on="local_date", how="left", maintain_order="left")
    windows = read_signal(name="stor_windows").rename({"date": "local_date"})
    out = out.join(windows, on="local_date", how="left", maintain_order="left")
    minutes = pl.col("local_half_hour") * 30
    start2 = pl.col("window_2_start_time").str.slice(0, 2).cast(pl.Int32) * 60 + pl.col(
        "window_2_start_time"
    ).str.slice(3, 2).cast(pl.Int32)
    end2 = pl.col("window_2_end_time").str.slice(0, 2).cast(pl.Int32) * 60 + pl.col(
        "window_2_end_time"
    ).str.slice(3, 2).cast(pl.Int32)
    out = out.with_columns(in_stor_window_2=(minutes >= start2) & (minutes < end2))
    warned = (
        read_signal(name="syswarn")
        .filter(pl.col("warning_type").str.contains("|".join(WARNING_TYPES)))
        .select(local_date=pl.col("published").dt.convert_time_zone("Europe/London").dt.date())
        .unique()
        .with_columns(warned_day=pl.lit(value=True))
    )
    out = out.join(warned, on="local_date", how="left", maintain_order="left")
    gov = (
        read_signal(name="bank_holidays")
        .select(local_date=pl.col("date"))
        .with_columns(gov_bank_holiday=pl.lit(value=True))
    )
    out = out.join(gov, on="local_date", how="left", maintain_order="left")
    return out.with_columns(
        warned_day=pl.col("warned_day").fill_null(value=False),
        gov_bank_holiday=pl.col("gov_bank_holiday").fill_null(value=False),
    )


def top_minus_bottom(*, view: pl.DataFrame, column: str, seed: int) -> Estimate | None:
    """Return the running share in the top third of `column` within cell minus the bottom third."""
    keep = usable(frame=view).filter(pl.col(column).is_not_null())
    if keep.height == 0:
        return None
    bins = cell_bins(view=keep, column=column, n_bins=3)
    return two_group_share(
        rows=keep, mask_a=bins == 2, mask_b=bins == 0, seed=seed, min_days=MIN_DAYS_PER_BIN
    )


def views(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Return the three views the panels compare: D8's net and solar flags, and the host controls.

    The key `c1_null` holds the stacked null controls of the C2 hosts, which cover the period of
    every signal (the C1 controls end before 2023-06-12).
    """
    d8 = frames["d8"]
    return {
        "net": with_state(frame=d8.filter(pl.col("era") == "net"), column="ts"),
        "solar": with_state(frame=d8.filter(pl.col("era") == "solar"), column="ts"),
        "null": with_state(frame=frames["c1_null"], column="ts"),
    }


def third_rows(*, series: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the top-minus-bottom difference with its interval, per signal and series."""
    rows = []
    for column, (_, title) in HALF_HOURLY_SIGNALS.items():
        for key, view in series.items():
            estimate = top_minus_bottom(
                view=view, column=column, seed=stable_seed("signal", column, key)
            )
            if estimate is not None:
                rows.append(
                    {
                        "signal": title,
                        "series": SERIES[key],
                        "difference": estimate.point,
                        "lo": estimate.lo,
                        "hi": estimate.hi,
                    }
                )
    for key, view in series.items():
        estimate = top_minus_bottom(
            view=view, column="forecast_peak_mw", seed=stable_seed("signal", "peak", key)
        )
        if estimate is not None:
            rows.append(
                {
                    "signal": "Day-ahead forecast peak demand",
                    "series": SERIES[key],
                    "difference": estimate.point,
                    "lo": estimate.lo,
                    "hi": estimate.hi,
                }
            )
    return pl.DataFrame(rows)


def event_rows(*, series: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return weekday evening running shares by STOR window, warning day, and bank holiday.

    The warning-day and bank-holiday groups hold under 20 days, so the floor drops them.
    """
    rows = []
    for key, view in series.items():
        evening = usable(frame=view).filter(
            ~pl.col("weekend"), pl.col("local_half_hour").is_between(EVENING_FIRST, EVENING_LAST)
        )
        groups = {
            "Inside STOR window 2": evening.filter(
                pl.col("in_stor_window_2").fill_null(value=False)
            ),
            "Outside STOR window 2": evening.filter(
                ~pl.col("in_stor_window_2").fill_null(value=True)
            ),
            "System warning day": evening.filter(pl.col("warned_day")),
            "Bank holiday (gov.uk)": evening.filter(pl.col("gov_bank_holiday")),
        }
        rows.extend(
            {
                "group": name,
                "series": SERIES[key],
                "share": group["state"].mean(),
                "days": group["local_date"].n_unique(),
            }
            for name, group in groups.items()
            if group.height and group["local_date"].n_unique() >= MIN_DAYS_PER_BIN
        )
    return pl.DataFrame(rows)


def checks(*, frames: dict[str, pl.DataFrame]) -> list[str]:
    """Return text lines: overlap of the APX index with the held day-ahead price, and calendars."""
    lines = []
    both = frames["d8"].filter(
        pl.col("day_ahead_price").is_not_null(), pl.col("apx_price").is_not_null()
    )
    if both.height:
        difference = (both["apx_price"] - both["day_ahead_price"]).abs()
        lines.append(
            "APX within-day index against the held N2EX day-ahead price, "
            f"{both.height} overlapping half-hours: "
            f"median absolute difference {difference.median():.2f}, "
            f"share within 0.05 GBP per MWh {(difference <= 0.05).mean():.3f}. "
            "The held table is NESO's N2EX hourly day-ahead price, not the APX "
            "half-hourly index, so the two are not expected to match."
        )
    local = frames["d8"].select("local_date").unique()
    ours = bank_holiday_dates()
    gov = set(read_signal(name="bank_holidays")["date"].to_list())
    window = set(local["local_date"].to_list())
    lines.append(
        f"bank holidays in the frame's dates: package {len(ours & window)}, "
        f"gov.uk {len(gov & window)}, "
        f"in both {len(ours & gov & window)}"
    )
    lines.append(
        "share of all D8 half-hours inside STOR window 2: "
        f"{frames['d8']['in_stor_window_2'].fill_null(value=False).mean():.3f}"
    )
    for era in ("net", "solar"):
        evening = frames["d8"].filter(
            pl.col("era") == era,
            ~pl.col("weekend"),
            pl.col("local_half_hour").is_between(EVENING_FIRST, EVENING_LAST),
            pl.col("observed"),
        )
        lines.append(
            f"{era} era weekday evenings: "
            f"{evening.filter(pl.col('warned_day'))['local_date'].n_unique()} system warning days "
            f"({evening.filter(pl.col('warned_day')).height} half-hours), "
            f"{evening.filter(pl.col('gov_bank_holiday'))['local_date'].n_unique()} gov.uk bank "
            "holidays; groups under 20 days are not drawn in G10"
        )
    return lines


def figures(*, series: dict[str, pl.DataFrame]) -> None:
    """Draw the two panels: top-minus-bottom differences, and weekday evening shares by event."""
    data = third_rows(series=series)
    domain = list(SERIES.values())
    base = alt.Chart(data)  # ty: ignore[unresolved-attribute]
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=domain, range=COLOURS),
        legend=alt.Legend(title=None, labelLimit=300),
    )
    chart = (
        base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("signal:N", title=None, sort=None, axis=alt.Axis(labelLimit=240)),
            yOffset="series:N",
            x=alt.X("lo:Q", title="Running share, top third minus bottom third (95% interval)"),
            x2="hi:Q",
            color=color,
        )
        + base.mark_point(aria=False, filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("signal:N", sort=None),
            yOffset="series:N",
            x="difference:Q",
            color=color,
        )
    ).properties(width=240, height=200)
    save(chart=chart, name="g9_other_signals_by_third")
    events = (
        alt.Chart(event_rows(series=series))  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False)
        .encode(
            y=alt.Y("group:N", title=None, sort=None),
            yOffset="series:N",
            x=alt.X(
                "share:Q",
                title="Running share, weekdays 16:00 to 19:00",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=domain, range=COLOURS),
                legend=alt.Legend(title=None, labelLimit=300),
            ),
        )
        .properties(width=260, height=200)
    )
    save(chart=events, name="g10_stor_windows_warnings_and_holidays")


def main() -> None:
    """Attach the signals to the frames, write the checks, and draw the panels."""
    frames = {
        "d8": attach(frame=load_frame(name="d8")),
        "c1_null": attach(frame=host_null_frame()),
    }
    lines = checks(frames=frames)
    series = views(frames=frames)
    lines.append("G9 values: running share in the top third minus the bottom third of each signal")
    lines.extend(
        f"  {row['signal']}, {row['series']}: {row['difference']:+.3f} "
        f"[{row['lo']:+.3f}, {row['hi']:+.3f}]"
        for row in third_rows(series=series).iter_rows(named=True)
    )
    PRINTED_DIR.mkdir(parents=True, exist_ok=True)
    (PRINTED_DIR / "signals_checks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figures(series=series)


if __name__ == "__main__":
    main()

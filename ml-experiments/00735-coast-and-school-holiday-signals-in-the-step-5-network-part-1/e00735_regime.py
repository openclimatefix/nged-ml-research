# ruff: noqa: E501, PERF401, ANN001
"""Experiment 735: how each Primary's demand changed in each school holiday and bank holiday (CPU only).

The changes are given year by year.

README. For each Primary, school-holiday block, and bank holiday from 2021-22 to October 2026, this
script computes the mean demand on the block's days minus the mean demand on the same weekdays and
the same half-hours of the term-time days in the 14 days before and the 14 days after the block, in
points of capacity (a point is 1% of the Primary's capacity). A bank holiday is compared with the
same weekday in the 14 days either side that are neither school-holiday days nor bank holidays.
Term dates before 2021-09 are not on disk. It relies on `e00735_columns` (the school calendar and the
local-date convention), `e00735_windows` (the Primary table), `e00735_diagnostic` (the cut date of
D15 and D18), and `e00668_score_lib` (the cleaned power). D15 and D18 are set to NaN before
2025-03-01. Run with the experiment folders on `PYTHONPATH` (source `ml-experiments/00524*/env.sh`).
It writes `regime/regime.csv` and `regime/regime_by_year.svg` under experiment 735's data folder. No
fit is involved. Series are Primary demand, never a metered generator.
"""

import os
from pathlib import Path
from typing import Final

import e00735_columns as cols  # ty: ignore[unresolved-import]
import e00735_windows as win  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1"
ROOT: Final[Path] = Path(os.environ.get("R735", ML_EXPERIMENTS_DATA_DIR / NAME))
REFERENCE_DAYS: Final = 14
MIN_REFERENCE_DAYS: Final = 2


def daily_profiles() -> tuple[np.ndarray, int]:
    """Return (primaries, local days, 48) mean power in points of capacity per local half-hour.

    The first day code is returned beside the array.
    """
    import e00668_score_lib as lib  # ty: ignore[unresolved-import]
    import e00735_diagnostic as diag  # ty: ignore[unresolved-import]

    table = win.sites()
    ids = sorted(table["time_series_id"].to_list())
    primary_of = dict(
        zip(table["time_series_id"].to_list(), table["primary"].to_list(), strict=True)
    )
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    cap = (
        capacity.filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")["capacity"]
        .to_numpy()
    )
    raw = lib.cleaned_grid(ids=ids)
    cut_index = int(lib.grid_index(times=pl.Series([diag.CUT_START]))[0])
    label_of = dict(zip(table["time_series_id"].to_list(), table["d_label"].to_list(), strict=True))
    for row, tid in enumerate(ids):
        if label_of[tid] in win.REPORTED_ONLY:  # D15 and D18 are not trusted before 2025-03-01
            raw[row, :cut_index] = np.nan
    grid = raw * (100.0 / cap)[:, None]
    stamps = (
        int(lib.GRID_START.timestamp()) * 1_000_000
        + np.arange(grid.shape[1], dtype=np.int64) * lib.STEP_MINUTES * 60_000_000
    )
    code = cols.local_dates(stamp_us=stamps).code.astype(np.int64)
    local = (
        pl.Series("t", stamps)
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .dt.offset_by("-15m")
        .dt.convert_time_zone("Europe/London")
    )
    half = (local.dt.hour() * 2 + local.dt.minute() // 30).to_numpy().astype(np.int64)
    first = int(code[0])
    n_days = int(code[-1]) - first + 1
    out = np.full((16, n_days, 48), np.nan)
    for row, tid in enumerate(ids):
        values = grid[row]
        ok = np.isfinite(values)
        flat = (code[ok] - first) * 48 + half[ok]
        sums = np.bincount(flat, weights=values[ok], minlength=n_days * 48)
        counts = np.bincount(flat, minlength=n_days * 48)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[primary_of[tid]] = np.where(
                counts > 0, sums / np.maximum(counts, 1), np.nan
            ).reshape(n_days, 48)
    return out, first


def mean_difference(
    *, profiles: np.ndarray, first: int, target: list[int], reference_for
) -> np.ndarray:
    """Return per Primary the mean over target days of the day's profile minus the reference mean.

    The reference mean is the mean profile of the day's reference days.
    """
    n_days = profiles.shape[1]
    diffs = np.full((profiles.shape[0], len(target)), np.nan)
    for i, day in enumerate(target):
        ref = [d - first for d in reference_for(day) if 0 <= d - first < n_days]
        if len(ref) < MIN_REFERENCE_DAYS or not 0 <= day - first < n_days:
            continue
        reference = np.nanmean(profiles[:, ref, :], axis=1)
        diffs[:, i] = np.nanmean(profiles[:, day - first, :] - reference, axis=1)
    return diffs


def main() -> None:
    """Write the table and the chart."""
    import holidays

    profiles, first = daily_profiles()
    table = win.sites()
    cal = cols.dense_calendar()
    bank = holidays.country_holidays("GB", subdiv="ENG", years=range(2021, 2028))
    last_data = first + profiles.shape[1] - 1

    def term_time(code: int) -> bool:
        i = code - cal.first_code
        return (
            0 <= i < len(cal.school_off)
            and cal.school_off[i] == 0
            and cols.date_of(code=code) not in bank
        )

    rows = []
    for block in cols.blocks():
        lo, hi = cols.code_of(day=block.first), cols.code_of(day=block.last)
        if hi > last_data or lo < first + REFERENCE_DAYS:
            continue
        target = list(range(lo, hi + 1))
        around = [*range(lo - REFERENCE_DAYS, lo), *range(hi + 1, hi + 1 + REFERENCE_DAYS)]
        diffs = mean_difference(
            profiles=profiles, first=first, target=target,
            reference_for=lambda d, around=around: [r for r in around if (r - d) % 7 == 0 and term_time(r)],
        )  # fmt: skip
        mean = np.nanmean(diffs, axis=1) if np.isfinite(diffs).any() else np.full(16, np.nan)
        year = block.first.year if block.first.month >= 8 else block.first.year - 1
        kind = (
            "summer"
            if block.days > 30
            else ("half-term" if block.days <= 10 else "Christmas or Easter")
        )
        for p in range(16):
            rows.append({"event": f"school {block.first}", "kind": kind, "academic_year": f"{year}-{(year + 1) % 100:02d}", "primary": p, "d_label": table["d_label"][p], "coast_km": table["coast_km"][p], "difference_points": float(mean[p])})  # fmt: skip
    for holiday_date in sorted(bank):
        code = cols.code_of(day=holiday_date)
        if code > last_data or code < first + REFERENCE_DAYS:
            continue
        diffs = mean_difference(
            profiles=profiles, first=first, target=[code],
            reference_for=lambda d: [d + 7 * k for k in (-2, -1, 1, 2) if term_time(d + 7 * k)],
        )  # fmt: skip
        year = holiday_date.year if holiday_date.month >= 8 else holiday_date.year - 1
        for p in range(16):
            rows.append({"event": f"bank holiday {holiday_date}", "kind": "bank holiday", "academic_year": f"{year}-{(year + 1) % 100:02d}", "primary": p, "d_label": table["d_label"][p], "coast_km": table["coast_km"][p], "difference_points": float(diffs[p, 0])})  # fmt: skip
    frame = pl.DataFrame(rows)
    out = ROOT / "regime"
    out.mkdir(parents=True, exist_ok=True)
    frame.write_csv(out / "regime.csv")
    summary = (
        frame.group_by("kind", "academic_year", "d_label", "coast_km")
        .agg(pl.col("difference_points").mean())
        .sort("kind", "academic_year", "coast_km")
    )
    corr = (
        summary.filter(pl.col("difference_points").is_finite())
        .group_by("kind", "academic_year")
        .agg(pl.corr("coast_km", "difference_points").alias("correlation with coast distance"), pl.len().alias("Primaries"))
        .sort("kind", "academic_year")
    )  # fmt: skip
    print(corr)
    chart(summary=summary, path=out / "regime_by_year.svg")
    print("wrote", out)


def chart(*, summary: pl.DataFrame, path: Path) -> None:
    """Draw demand change against coast distance, one panel per academic year, coloured by kind of holiday."""
    import altair as alt
    import plotting.ocf_theme as theme  # noqa: F401

    colours = {
        "half-term": "#FF4901",
        "Christmas or Easter": "#306BFF",
        "summer": "#10C5F7",
        "bank holiday": "#B701FF",
    }
    frame = summary.filter(pl.col("difference_points").is_finite())
    base = alt.Chart(frame).encode(
        x=alt.X("coast_km:Q", title="Distance from the coast (km)"),
        y=alt.Y("difference_points:Q", title="Demand on holiday days minus nearby term-time days (points of capacity)"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=list(colours), range=list(colours.values())), legend=alt.Legend(orient="top", title=None)),
    )  # fmt: skip
    layered = base.mark_circle(size=40) + base.mark_rule(strokeDash=[2, 2], color="#888").encode(  # ty: ignore[unresolved-attribute]
        y=alt.datum(0)
    )
    chart = layered.properties(width=620, height=120).facet(row=alt.Row("academic_year:N", title=None)).properties(
        title=alt.TitleParams(
            "Holiday demand change by distance from the coast, each academic year",
            subtitle=["Each dot: one kind of holiday, mean over its blocks and the Primaries' demand (points of capacity; zero means no change).", "Reference: same weekdays and half-hours in the 14 days either side that are term time."],
        )
    )  # fmt: skip
    chart.save(str(path))


if __name__ == "__main__":
    main()

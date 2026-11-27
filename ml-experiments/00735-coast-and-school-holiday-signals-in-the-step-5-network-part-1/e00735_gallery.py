# ruff: noqa: E501
"""Experiment 735: the diagnostic gallery of the new input columns, shown to the maintainer before the first fit (CPU only).

README. This script prints and draws what the new columns hold, using the calendar, the coast
closeness, and daily weather, with no ENS rows and no fit. It relies on `e00735_columns`,
`e00735_windows`, and `plotting.ocf_theme`. Run with the experiment folders on `PYTHONPATH` (source
`ml-experiments/00524*/env.sh`). It writes to `gallery/` under experiment 735's data folder:

- `closeness.csv`: each Primary's coast distance, true closeness `c`, and permuted closeness, and the
  correlation of the two;
- `columns_by_block.csv`: for each school-holiday block and each Primary group, the days on which S1,
  S2, S3, K2, K3, K4, and K5 are non-zero, and the mean K2 at the coastal and inland Primaries;
- `weather_split_counts.csv`: the Primary-days of each weather kind in each window, held-in months;
- `columns_by_date.svg`: K2 (`c` times school holiday) for the 4 coastal and 3 inland Primaries, with
  the school holidays and the placebo calendar, over 2025-11 to 2026-11.

- `w_panels.svg` and `w_panels.csv`: W1 to W4 for D19 (coastal) and D11 (inland), 2026-09-14 to
  2026-11-15, drawn from realised daily weather with the same formulas as the columns. The
  anomalies are those of the daily maximum temperature and the daily mean shortwave against the
  Primary's mean and standard deviation in the same calendar month over 2019-09 to 2025-12, clipped
  to +-3 (the columns use the 09:00-origin B1 columns, fitted on the training rows); W3 uses the
  7-day baseline of `e00735_columns.week_baseline` with origins at 12:00 UTC on the day before.

**The W1 to W4 panels do not use the real ENS rows.** The real columns need a month's stage-2 rows
(`e.k.build_month` of experiment 677's runner, which loads the cleaned power, the weather caches, and
`torch`, and has only been run on the GPU chain), so the smoke test prints the real columns and this
gallery shows the realised-weather proxy.
"""

import os
from datetime import date
from pathlib import Path
from typing import Final

import e00735_columns as cols  # ty: ignore[unresolved-import]
import e00735_windows as win  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1"
ROOT: Final[Path] = Path(os.environ.get("R735", ML_EXPERIMENTS_DATA_DIR / NAME))
FIRST: Final = date(2025, 11, 1)
LAST: Final = date(2026, 11, 30)
HELD_IN_LAST: Final = date(2026, 6, 30)


def main() -> None:
    """Write the gallery tables and the chart."""
    out = ROOT / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    table = win.sites()
    labels = table["d_label"].to_list()
    c_true, c_perm = cols.coast_arrays(save=True)
    closeness = pl.DataFrame(
        {"primary": table["primary"], "d_label": labels, "coast_km": table["coast_km"], "closeness": c_true, "permuted_closeness": c_perm}
    )  # fmt: skip
    closeness.write_csv(out / "closeness.csv")
    print(closeness)
    print(
        "correlation of permuted with true closeness:",
        round(float(np.corrcoef(c_true, c_perm)[0, 1]), 3),
    )
    print(
        "Primaries within 3 km of the coast by distance:",
        win.coastal_by_distance(table=table),
        "(plan: D9, D2, D4, D19)",
    )
    n = (LAST - FIRST).days + 1
    dates = [date.fromordinal(FIRST.toordinal() + i) for i in range(n)]
    codes = np.asarray([cols.code_of(day=d) for d in dates], dtype=np.int64)
    school = cols.school_columns(codes=codes)
    bank, weekend = cols.day_flags(codes=codes)
    rows = []
    for block in cols.blocks():
        if block.last < FIRST or block.first > LAST:
            continue
        sel = np.asarray([block.first <= d <= block.last for d in dates], dtype=bool)
        for name, idx in win.group_indices(labels=labels).items():
            if not name.startswith(("4 coastal", "3 inland", "six", "eight")):
                continue
            rows.append(
                {
                    "block": f"{block.first} to {block.last}", "days": block.days, "group": name,
                    "S1_days": int(school[sel, 0].sum()), "S2_days": int(school[sel, 1].sum()),
                    "S3_days": int(school[sel, 2].sum()), "K2_mean": float((c_true[idx].mean()) * school[sel, 0].mean()),
                    "K3_mean": float(c_true[idx].mean() * school[sel, 1].mean()),
                    "K4_days_non_zero": int(bank[sel].sum()), "K5_days_non_zero": int(weekend[sel].sum()),
                }
            )  # fmt: skip
    pl.DataFrame(rows).write_csv(out / "columns_by_block.csv")
    print(pl.DataFrame(rows).filter(pl.col("group") == "4 coastal"))
    daily = win.daily_weather(expected_dir=win.EXPECTED_DIR, cache=ROOT / win.WEATHER_CACHE_NAME)
    medians = win.monthly_medians(weather=daily)
    prim = np.repeat(np.arange(16), n)
    day_codes = np.tile(codes, 16)
    kinds = win.weather_split(weather=daily, medians=medians, primary=prim, codes=day_codes)
    held = day_codes <= cols.code_of(day=HELD_IN_LAST)
    masks = {k: v & held for k, v in win.window_masks(codes=day_codes).items()}
    counts = win.split_counts(kinds=kinds, masks=masks)
    counts.write_csv(out / "weather_split_counts.csv")
    print("Primary-days by weather kind in each window (held-in months, realised weather):")
    print(counts)
    print(
        "days from the expected-weather proxy in the range:",
        int(
            daily.proxy[np.clip(codes - daily.first_code, 0, len(daily.proxy) - 1)][
                codes - daily.first_code < len(daily.proxy)
            ].sum()
        ),
        "; the table ends on",
        cols.date_of(code=daily.first_code + len(daily.proxy) - 1),
    )
    w_panels(daily=daily, c_true=c_true, labels=labels, out=out)
    chart(
        dates=dates,
        codes=codes,
        school=school,
        c_true=c_true,
        labels=labels,
        path=out / "columns_by_date.svg",
        daily=daily,
    )
    print("wrote", out)


def monthly_anomaly(*, weather: win.DailyWeather, values: np.ndarray) -> np.ndarray:
    """Return (Primaries, days) anomalies of `values` against each Primary's calendar-month mean over 2019-09 to 2025-12, over its standard deviation, clipped to +-3."""
    n_p, n_days = values.shape
    codes = weather.first_code + np.arange(n_days)
    month = (
        cols.local_dates(
            stamp_us=codes.astype(np.int64) * 86_400_000_000 + 12 * 3_600_000_000
        ).month.astype(np.int64)
        - 1
    )
    inside = (codes >= cols.code_of(day=win.MEDIAN_FIRST)) & (
        codes <= cols.code_of(day=win.MEDIAN_LAST)
    )
    out = np.full(values.shape, np.nan)
    for p in range(n_p):
        for m in range(12):
            ref = values[p][inside & (month == m)]
            ref = ref[np.isfinite(ref)]
            if len(ref) > 1:
                z = (values[p][month == m] - ref.mean()) / max(float(ref.std()), 1e-6)
                out[p, month == m] = np.clip(z, -cols.ANOMALY_CLIP, cols.ANOMALY_CLIP)
    return out


def w_panels(*, daily: win.DailyWeather, c_true: np.ndarray, labels: list[str], out: Path) -> None:
    """Draw W1 to W4 of D19 and D11 from realised daily weather and write the values."""
    first, last = date(2026, 9, 14), date(2026, 11, 15)
    dates = [date.fromordinal(first.toordinal() + i) for i in range((last - first).days + 1)]
    codes = np.asarray([cols.code_of(day=d) for d in dates], dtype=np.int64)
    school = cols.school_columns(codes=codes)
    z_t = monthly_anomaly(weather=daily, values=daily.tmax)
    z_s = monthly_anomaly(weather=daily, values=daily.shortwave)
    max_daily = cols.DailyMax(
        first_code=daily.first_code,
        value=daily.tmax,
        run_us=np.zeros(daily.tmax.shape, dtype=np.int64),
    )
    rows = []
    for lab in ("D19", "D11"):
        p = labels.index(lab)
        idx = codes - daily.first_code
        origin_us = (codes - 1) * 86_400_000_000 + 12 * 3_600_000_000
        baseline = cols.week_baseline(
            daily=max_daily, primary=np.full(len(codes), p), origin_us=origin_us
        )
        w3 = cols.first_warm_day(tmax_celsius=daily.tmax[p, idx], baseline=baseline)
        c = float(c_true[p])
        values = {
            "W1 (c x holiday x temperature anomaly)": c * school[:, 0] * np.nan_to_num(z_t[p, idx]),
            "W2 (c x holiday x sunshine anomaly)": c * school[:, 0] * np.nan_to_num(z_s[p, idx]),
            "W3 (c x holiday x warmer than the week before)": c * school[:, 0] * w3,
            "W4 (c x temperature anomaly)": c * np.nan_to_num(z_t[p, idx]),
        }
        for name, v in values.items():
            rows.extend(
                {"date": d, "Primary": lab, "column": name, "value": float(x)}
                for d, x in zip(dates, v, strict=True)
            )
    frame = pl.DataFrame(rows)
    frame.write_csv(out / "w_panels.csv")
    import altair as alt
    import plotting.ocf_theme  # noqa: F401

    names = list(dict.fromkeys(frame["column"].to_list()))
    chart_ = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%d %b", tickCount=8)),
            y=alt.Y("value:Q", title="Column value"),
            color=alt.Color("Primary:N", scale=alt.Scale(domain=["D19", "D11"], range=["#FF4901", "#306BFF"]), legend=alt.Legend(orient="top", title=None)),
        )
        .properties(width=620, height=70)
        .facet(row=alt.Row("column:N", sort=names, title=None, header=alt.Header(labelAngle=0, labelAlign="left", labelOrient="top")))
        .properties(title=alt.TitleParams("The weather columns W1 to W4 around the October 2026 half-term, D19 (coastal) and D11 (inland)", subtitle=["This picture uses realised daily weather (ERA5 to 2026-07-31, then the expected-weather table) for illustration only; the model inputs do not (they use the ensemble forecast issued at the origin). The half-term is 2026-10-24 to 2026-11-01. Substation demand only."]))
    )  # fmt: skip
    chart_.save(str(out / "w_panels.svg"))


def chart(
    *,
    dates: list[date],
    codes: np.ndarray,
    school: np.ndarray,
    c_true: np.ndarray,
    labels: list[str],
    path: Path,
    daily: win.DailyWeather,
) -> None:
    """Draw K2 and the placebo K2 for coastal and inland Primaries over the scored year."""
    import altair as alt
    import plotting.ocf_theme  # noqa: F401

    shifted = cols.school_columns(codes=codes, shift_days=cols.PLACEBO_SHIFT_DAYS)
    show = ["D9", "D4", "D19", "D2", "D11", "D10", "D22"]
    rows = []
    for lab in show:
        p = labels.index(lab)
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "Primary": lab,
                    "calendar": "real school holidays",
                    "K2": float(c_true[p] * school[i, 0]),
                }
            )
            rows.append(
                {
                    "date": d,
                    "Primary": lab,
                    "calendar": "placebo (21 days later)",
                    "K2": float(c_true[p] * shifted[i, 0]),
                }
            )
    frame = pl.DataFrame(rows)
    chart_ = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_area(opacity=0.8, interpolate="step-after")
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y", tickCount=7)),
            y=alt.Y("K2:Q", title="K2", stack=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("calendar:N", scale=alt.Scale(domain=["real school holidays", "placebo (21 days later)"], range=["#FF4901", "#306BFF"]), legend=alt.Legend(orient="top", title=None)),
        )
        .properties(width=620, height=55)
        .facet(row=alt.Row("Primary:N", sort=show, title=None, header=alt.Header(labelAngle=0, labelAlign="left", labelOrient="top")))
        .properties(title=alt.TitleParams("The school-holiday column K2 of each Primary, real and placebo", subtitle=["K2 is the coast closeness c (D9, D4, D19, and D2 are within 3 km of the coast; D11, D10, and D22 are inland) on school-holiday days and zero otherwise.", "Substation demand only; no metered generator appears. November 2025 to November 2026."]))
    )  # fmt: skip
    chart_.save(str(path))
    del daily


if __name__ == "__main__":
    main()

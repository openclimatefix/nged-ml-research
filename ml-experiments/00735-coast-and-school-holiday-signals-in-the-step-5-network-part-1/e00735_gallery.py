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

W1 to W4 need the stage-2 rows, so the gallery draws a daily proxy of the weather factor `W1 = K2 *
(daily maximum temperature anomaly)` from realised weather. The smoke test prints the real columns.
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
    codes = np.asarray([cols.code_of(day=d) for d in dates])
    school = cols.school_columns(codes=codes)
    bank, weekend = cols.day_flags(codes=codes)
    rows = []
    for block in cols.blocks():
        if block.last < FIRST or block.first > LAST:
            continue
        sel = np.asarray([block.first <= d <= block.last for d in dates])
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
    daily = win.daily_weather(expected_dir=win.EXPECTED_DIR, cache=ROOT / "weather_daily.npz")
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
        alt.Chart(frame)
        .mark_area(opacity=0.8, interpolate="step-after")
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y", tickCount=7)),
            y=alt.Y("K2:Q", title="K2 (closeness to coast x school holiday)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("calendar:N", scale=alt.Scale(domain=["real school holidays", "placebo (21 days later)"], range=["#FF4901", "#306BFF"]), legend=alt.Legend(orient="top", title=None)),
        )
        .properties(width=620, height=55)
        .facet(row=alt.Row("Primary:N", sort=show, title=None))
        .properties(title=alt.TitleParams("The school-holiday column K2 of each Primary, real and placebo", subtitle=["K2 is the coast closeness c (D9, D4, D19, and D2 are within 3 km of the coast; D11, D10, and D22 are inland) on school-holiday days and zero otherwise.", "Substation demand only; no metered generator appears. November 2025 to November 2026."]))
    )  # fmt: skip
    chart_.save(str(path))
    del daily


if __name__ == "__main__":
    main()

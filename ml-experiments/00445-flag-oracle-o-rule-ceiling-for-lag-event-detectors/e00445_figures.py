# ruff: noqa: E501
r"""Draw the figures of experiment 445 from the stored scores and forecasts (no fit).

- Figure 1: arm minus B-mix, pooled over 5 months, with 95% event-cluster intervals.
- Figure 2: mean score of the lagged head, the lagless head, and B-mix by number of flagged lags.
- Figures 3 and 4: a day-ahead and a 14-day forecast of one Primary (B-mix), winter and summer.
- Figure 5: six real events (three Primaries from each metering group) with the lagged head, the
  lagless head, and B-mix.

```bash
<PYTHONPATH as e00445_orule.py shows> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00445-*/e00445_figures.py
```
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00445_ceiling as ceiling  # ty: ignore[unresolved-import]
import e00445_fit as fit  # ty: ignore[unresolved-import]
import e00445_orule as orule  # ty: ignore[unresolved-import]
import e00445_report as report  # ty: ignore[unresolved-import]
import e00445_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00445-flag-oracle-o-rule-ceiling-for-lag-event-detectors"
)
WIDTH: Final[int] = 700
POINTS: Final[float] = 100.0
ORANGE, BLUE, GREEN, PURPLE, SKY = (
    theme.BRAND_ORANGE,
    theme.DATA_BLUE,
    theme.DATA_GREEN,
    theme.DATA_PURPLE,
    theme.DATA_SKY,
)
CONTRAST_ARMS: Final[list[str]] = ["O-hard", "O-soft", "O-floor", "O-lagless", "A0-inj", "B1", "B3"]
CONTRAST_COLOURS: Final[list[str]] = [ORANGE, PURPLE, GREEN, BLUE, SKY, "#888888", "#444444"]
HEAD_ARMS: Final[list[str]] = ["A0-inj", "O-lagless", "B-mix"]
HEAD_LABELS: Final[dict[str, str]] = {
    "A0-inj": "lagged head",
    "O-lagless": "lagless head",
    "B-mix": "B-mix (50/50)",
}
MONTHS: Final[tuple[str, ...]] = ("2026-01", "2025-12", "2025-10", "2026-03", "2026-06")


def figure_contrasts() -> None:
    """Figure 1: arm minus B-mix with 95% intervals, pooled over the five seed-0 months."""
    c = (
        pl.read_parquet(orule.OUT_DIR / "contrasts_pooled.parquet")
        .filter(
            ~pl.col("largest_removed"),
            pl.col("cls").is_in(["tgt_real", "harm_real", "harm_inj"]),
            pl.col("arm").is_in(CONTRAST_ARMS),
        )
        .with_columns(
            row=pl.col("cls").replace_strict(ceiling.CLASS_LABEL),
            group=pl.col("group").replace_strict(
                {"signed_six": "signed-MW six", "mva_eight": "MVA-only eight"}
            ),
        )
    )
    base = alt.Chart(c.to_pandas())
    y = alt.Y("arm:N", sort=CONTRAST_ARMS, title=None)
    colour = alt.Color(
        "arm:N", scale=alt.Scale(domain=CONTRAST_ARMS, range=CONTRAST_COLOURS), legend=None
    )
    rules = base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        y=y, x=alt.X("low:Q", title="Arm minus B-mix, score (points of capacity; negative is better)"),
        x2="high:Q", color=colour,
    )  # fmt: skip
    dots = base.mark_point(filled=True, size=40).encode(y=y, x="contrast:Q", color=colour)  # ty: ignore[unresolved-attribute]
    zero = base.mark_rule(color="#888", strokeDash=[3, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        (zero + rules + dots)
        .properties(width=WIDTH - 260, height=105)
        .facet(
            row=alt.Row(
                "row:N", title=None, sort=["inside real events", "real harm", "injected harm"]
            ),
            column=alt.Column("group:N", title=None),
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Flag-oracle rules (O-) and stored arms minus B-mix (interim, unaudited)",
                    "Dot: estimate. Line: 95% interval from resampling whole events. Dashed: same as B-mix.",
                    "Five test months, fit seed 0. Zero or above means no better than B-mix.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_1_contrasts.svg")


def figure_by_count() -> None:
    """Figure 2: mean score by number of flagged lags, signed-MW six."""
    d = pl.read_parquet(orule.OUT_DIR / "by_count.parquet").filter(
        pl.col("cls").is_in(["harm_inj", "tgt_real", "harm_real"])
    )
    long = d.unpivot(
        on=HEAD_ARMS,
        index=["kind", "cls", "n_flagged", "rows"],
        variable_name="arm",
        value_name="score",
    ).with_columns(
        arm=pl.col("arm").replace_strict(HEAD_LABELS),
        row=pl.col("cls").replace_strict(ceiling.CLASS_LABEL),
    )
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(HEAD_LABELS.values()), range=[ORANGE, BLUE, GREEN]),
        legend=alt.Legend(title=None, orient="top"),
    )
    chart = (
        alt.Chart(long.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X(
                "n_flagged:O",
                title="Number of the 3 lags that lie inside an event",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset="arm:N",
            y=alt.Y("score:Q", title="Mean score (points; smaller is better)"),
            color=colour,
        )
        .properties(width=WIDTH - 120, height=110)
        .facet(
            row=alt.Row(
                "row:N", title=None, sort=["injected harm", "real harm", "inside real events"]
            )
        )
        .properties(
            title=alt.TitleParams(
                [
                    "The lagless head barely beats the lagged head when lags are flagged (interim)",
                    "Signed-MW six, five test months, fit seed 0. B-mix beats both heads in every group.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_2_by_count.svg")


def read_arm(*, arm: str, month: str, seed: int = 0) -> pl.DataFrame:
    """Return an arm's clean-world rows with the quantiles in points of capacity."""
    d = pl.read_parquet(ceiling.arm_path(arm=arm, month=month, seed=seed, world=-1))
    cols = [c for c in d.columns if c.startswith("q")]
    return d.with_columns(*[pl.col(c) * POINTS for c in [*cols, "actual"]])


def figure_examples(*, site: str) -> None:
    """Figures 3 and 4: day-ahead and 14-day forecasts of B-mix, one winter and one summer origin."""
    times = common.grid_times()
    primary = fit.prep.primaries()["site"].to_list().index(site)
    for number, name, hours in ((3, "day_ahead", 36.0), (4, "14_day", 400.0)):
        panels = []
        for month in ("2026-01", "2026-06"):
            d = read_arm(arm="B-mix", month=month, seed=0).filter(pl.col("primary") == primary)
            origins = np.sort(d["origin"].unique().to_numpy())
            origin = int(origins[len(origins) // 2])
            one = d.filter(pl.col("origin") == origin, pl.col("lead_hours") <= hours).with_columns(
                time=pl.Series([times[i] for i in d.filter(pl.col("origin") == origin, pl.col("lead_hours") <= hours)["target"].to_list()])
            )  # fmt: skip
            x = alt.X("time:T", title="Valid time (UTC)")
            pdata = one.to_pandas()
            outer = alt.Chart(pdata).mark_area(opacity=0.15, color=BLUE).encode(  # ty: ignore[unresolved-attribute]
                x=x, y=alt.Y("q05:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)), y2="q95:Q"
            )  # fmt: skip
            inner = (
                alt.Chart(pdata)
                .mark_area(opacity=0.3, color=BLUE)
                .encode(x=x, y="q25:Q", y2="q75:Q")
            )  # ty: ignore[unresolved-attribute]
            median = alt.Chart(pdata).mark_line(color=BLUE).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
            truth = (
                alt.Chart(pdata)
                .mark_line(color="#333", strokeDash=[2, 2])
                .encode(x=x, y="actual:Q")
            )  # ty: ignore[unresolved-attribute]
            panels.append(
                (outer + inner + median + truth).properties(
                    width=WIDTH - 90,
                    height=170,
                    title=f"{month[:4]}-{month[5:]}: Primary {site}, forecast issued {times[origin]:%Y-%m-%d %H:%M}",
                )  # fmt: skip
            )
        chart = alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                [
                    f"B-mix {name.replace('_', '-')} forecast (interim)",
                    "Line: median. Dark band: 25% to 75%. Light band: 5% to 95%. Dashed: the cleaned reading.",
                ],
                anchor="start",
            )
        )
        chart.save(FIGURE_DIR / f"figure_{number}_example_{name}.svg")


def padded_runs(*, real: np.ndarray, p: int) -> list[tuple[int, int]]:
    """Return (start, end) grid indices of a Primary's padded real-event runs."""
    edges = np.diff(np.concatenate([[0], real[p].astype(np.int8), [0]]))
    return list(
        zip(np.nonzero(edges == 1)[0].tolist(), np.nonzero(edges == -1)[0].tolist(), strict=True)
    )


def pick_events(*, real: np.ndarray, per_group: int = 3) -> list[dict]:
    """Return events (with their month cell) spread over both metering groups."""
    sites = fit.prep.primaries()["site"].to_list()
    groups = report.group_of(sites=sites)
    ranges = {}
    for month in MONTHS:
        f = pl.read_parquet(
            ceiling.arm_path(arm="B-mix", month=month, seed=0, world=-1), columns=["target"]
        )
        ranges[month] = (int(f["target"].min()), int(f["target"].max()))  # ty: ignore[invalid-argument-type]
    found: dict[str, list[dict]] = {"signed_six": [], "mva_eight": []}
    for p, site in enumerate(sites):
        if groups[p] not in found:
            continue
        for start, end in padded_runs(real=real, p=p):
            for month, (lo, hi) in ranges.items():
                if start - 96 >= lo and end + 96 <= hi and 96 <= end - start <= 480:
                    found[groups[p]].append(
                        {
                            "p": p,
                            "site": site,
                            "start": start,
                            "end": end,
                            "month": month,
                            "group": groups[p],
                        }
                    )
    out = []
    for group in found:
        events = sorted(found[group], key=lambda e: e["end"] - e["start"], reverse=True)
        seen: set[str] = set()
        for e in events:
            if e["site"] not in seen and len([o for o in out if o["group"] == group]) < per_group:
                out.append(e)
                seen.add(e["site"])
    return out


def figure_events() -> None:
    """Figure 5: six real events, forecasts made the day before each event starts."""
    times = common.grid_times()
    real = score.real_event_mask()
    panels = []
    for i, e in enumerate(pick_events(real=real), start=1):
        frames = []
        for arm, label in (
            ("A0-inj", "lagged head"),
            ("O-lagless", "lagless head"),
            ("B-mix", "B-mix"),
        ):
            d = read_arm(arm=arm, month=e["month"]).filter(pl.col("primary") == e["p"])
            origins = np.sort(d["origin"].unique().to_numpy())
            origin = int(origins[np.argmin(np.abs(origins - (e["start"] - 48)))])
            frames.append(
                d.filter(pl.col("origin") == origin, pl.col("lead_hours") <= 120).with_columns(
                    arm=pl.lit(label)
                )
            )
        d = pl.concat(frames, how="diagonal_relaxed").with_columns(
            time=pl.Series(
                [times[t] for t in pl.concat(frames, how="diagonal_relaxed")["target"].to_list()]
            )
        )
        lo, hi = times[e["start"]], times[e["end"]]
        t0, t1 = lo - timedelta(days=1), hi + timedelta(days=2)
        x = alt.X("time:T", title=None, scale=alt.Scale(domain=[t0, t1]))
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=["lagged head", "lagless head", "B-mix"], range=[ORANGE, BLUE, GREEN]
            ),
            legend=alt.Legend(title=None, orient="top") if i == 1 else None,
        )
        shade = (
            alt.Chart(pl.DataFrame({"a": [lo], "b": [hi]}).to_pandas())
            .mark_rect(opacity=0.12, color="#888")
            .encode(x="a:T", x2="b:T")
        )  # ty: ignore[unresolved-attribute]
        mix = d.filter(pl.col("arm") == "B-mix").to_pandas()
        band = (
            alt.Chart(mix)
            .mark_area(opacity=0.12, color=GREEN)
            .encode(
                x=x,
                y=alt.Y("q05:Q", title="Power (points)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
            )
        )  # ty: ignore[unresolved-attribute]
        median = alt.Chart(d.to_pandas()).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
        actual = alt.Chart(mix).mark_line(color="#333").encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        group = "signed-MW six" if e["group"] == "signed_six" else "MVA-only eight"
        panels.append(
            (shade + band + median + actual).properties(
                width=WIDTH - 90,
                height=130,
                title=f"Event {i}: Primary {e['site']} ({group})",
            )  # fmt: skip
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Labelled load-shift events: lagged head, lagless head, and B-mix (interim)",
                "Forecasts issued the day before the padded event starts (shaded). Black: the cleaned reading.",
                "Band: B-mix 5% to 95%. Forecasts reach 5 days. Some real events are not logged.",
            ],
            anchor="start",
        )
    )
    chart.save(FIGURE_DIR / "figure_5_real_events.svg")


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    figure_by_count()
    figure_examples(site="D5")
    figure_events()


if __name__ == "__main__":
    main()

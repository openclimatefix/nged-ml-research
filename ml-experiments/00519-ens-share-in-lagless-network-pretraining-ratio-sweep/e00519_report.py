# ruff: noqa: E501
"""Horizon-band table and figures of experiment 519, from the tables the scorer stored.

Commands: `bands` (CRPS by horizon band per arm, and paired differences to F0 with per-fold signs; writes
`scores/bands.csv`) and `figures` (SVG files in the experiment's docs folder; optimise each with
`npx svgo@4 --multipass --precision=1 --final-newline`).

Figures: 1 CRPS against ENS share with Bonferroni intervals against F0; 2 the flag ablation; 3 the
horizon bands; 4 q99 pinball loss against ENS share; 5 example forecasts.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00519_fit as fit  # ty: ignore[unresolved-import]
import e00519_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00519-ens-share-in-lagless-network-pretraining-ratio-sweep"
)
SCORES: Final[Path] = fit.W / "scores"
WIDTH: Final[int] = 600
SHARE: Final[dict[str, int]] = {"F0": 0, "E10": 10, "E25": 25, "L-mix": 50, "E75": 75, "E100": 100}
BANDS: Final[dict[str, tuple[float, float]]] = {
    "day 1": (0.0, 24.0),
    "days 2 to 3": (24.0, 72.0),
    "days 4 to 10": (72.0, 241.0),
}
GROUPS: Final[dict[str, str]] = {"eight": "mva_eight", "six_clean": "signed_six"}
GROUP_LABEL: Final[dict[str, str]] = {"eight": "MVA-only eight", "six_clean": "Signed-MW six"}
ARMS_BANDS: Final[tuple[str, ...]] = ("L-Ww", "F0", "E10", "E25", "L-mix", "E75", "E100", "TS")
EXAMPLE_ORIGIN: Final[str] = "2026-01-12"
EXAMPLES: Final[dict[str, str]] = {"D10": "signed-MW Primary", "D4": "MVA-only Primary"}


def _rows() -> pl.DataFrame:
    """Return the scored rows of the five folds (main16 row set) with the arms' CRPS, the lead, and the month."""
    keep = ["primary", "origin", "target", "group", "cls"]
    frames = []
    for path in sorted(SCORES.glob("rows_*.parquet")):
        frame = pl.read_parquet(path)
        frame = frame.filter(pl.col("L-Ww__q50").is_finite())
        arms = [f"{a}__crps" for a in ARMS_BANDS]
        ts = pl.read_parquet(fit.W / "scores_ts" / path.name).select(
            "primary", "origin", "target", "TS__crps"
        )
        frame = frame.join(ts, on=["primary", "origin", "target"], how="left")
        cols = [c for c in (*keep, *arms) if c in frame.columns]
        frames.append(
            frame.select(cols).with_columns(
                month=pl.lit(path.stem.removeprefix("rows_")),
                lead=(pl.col("target") - pl.col("origin")).cast(pl.Float64) / 2.0,
            )
        )
    return pl.concat(frames, how="diagonal_relaxed")


def bands() -> None:
    """Write CRPS by horizon band, arm, group and month, and print the paired differences to F0."""
    rows = _rows()
    out = []
    for group, name in GROUPS.items():
        part = rows.filter(pl.col("group") == name)
        if group == "six_clean":
            part = part.filter(pl.col("cls") == "clean")
        for band, (lo, hi) in BANDS.items():
            sub = part.filter(pl.col("lead") >= lo, pl.col("lead") < hi)
            for arm in ARMS_BANDS:
                col = f"{arm}__crps"
                if col not in sub.columns:
                    continue
                for month, d in sub.group_by("month"):
                    out.append(
                        {
                            "group": group, "band": band, "arm": arm, "month": month[0],
                            "n": d.height, "crps_sum": float(d[col].sum()), "crps": float(d[col].mean()),
                        }
                    )  # fmt: skip
    table = pl.DataFrame(out)
    table.write_csv(SCORES / "bands.csv")
    pooled = table.group_by("group", "band", "arm").agg(
        (pl.col("crps_sum").sum() / pl.col("n").sum()).alias("crps")
    )
    for group in GROUPS:
        for band in BANDS:
            ref = pooled.filter(
                pl.col("group") == group, pl.col("band") == band, pl.col("arm") == "F0"
            )["crps"][0]
            line = []
            for arm in ARMS_BANDS:
                v = pooled.filter(
                    pl.col("group") == group, pl.col("band") == band, pl.col("arm") == arm
                )["crps"]
                if len(v):
                    line.append(f"{arm} {float(v[0]) - ref:+.3f}")
            print(group, band, "minus F0:", ", ".join(line))


def _t(*, name: str) -> pl.DataFrame:
    return pl.read_csv(SCORES / name)


def _contrasts() -> pl.DataFrame:
    return pl.read_csv(SCORES / "contrasts.csv").filter(
        pl.col("metric") == "crps",
        pl.col("rowset") == "main16",
        pl.col("slice").is_in(list(GROUP_LABEL)),
    )


def figures() -> None:
    """Draw the figures."""
    con = _contrasts()
    panels = []
    vs = con.filter(pl.col("contrast").is_in([f"{a} - F0" for a in SHARE if a != "F0"]))
    vs = vs.with_columns(
        share=pl.col("contrast").str.replace(" - F0", "").replace_strict(SHARE),
        group=pl.col("slice").replace_strict(GROUP_LABEL),
    ).to_pandas()
    for label in GROUP_LABEL.values():
        part = vs[vs["group"] == label]
        base = alt.Chart(part).encode(
            x=alt.X(
                "share:Q",
                title="ENS share of every pretraining batch (%)",
                scale=alt.Scale(domain=[0, 105]),
            )
        )  # ty: ignore[unresolved-attribute]
        bar = base.mark_rule(strokeWidth=3, color=theme.DATA_BLUE).encode(
            y=alt.Y("low_bonferroni:Q", title="CRPS minus 0% control F0"), y2="high_bonferroni:Q"
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=70, color=theme.DATA_BLUE).encode(y="difference:Q")  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())
            .mark_rule(color="black")
            .encode(y="y:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=150, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "CRPS of each ENS share minus the 0% control F0 (points of capacity; lower is better),",
                "five folds, two seeds, Bonferroni intervals for 32 tests. The 50% point is experiment 511's L-mix.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_1_headline.svg")

    flag = (
        con.filter(
            pl.col("contrast").is_in(
                ["NF25 - E25", "NF50 - L-mix", "Z25 - E25", "Z50 - L-mix", "F0 - L-Ww", "E25 - E10"]
            )
        )
        .with_columns(group=pl.col("slice").replace_strict(GROUP_LABEL))
        .to_pandas()
    )
    panels = []
    order = ["NF25 - E25", "NF50 - L-mix", "Z25 - E25", "Z50 - L-mix", "F0 - L-Ww"]
    for label in GROUP_LABEL.values():
        part = flag[(flag["group"] == label) & flag["contrast"].isin(order)]
        base = alt.Chart(part).encode(y=alt.Y("contrast:N", sort=order, title=None))  # ty: ignore[unresolved-attribute]
        bar = base.mark_rule(strokeWidth=3, color=theme.DATA_SKY).encode(
            x=alt.X("low_bonferroni:Q", title="CRPS difference"), x2="high_bonferroni:Q"
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=70, color=theme.DATA_SKY).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
            .mark_rule(color="black")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=130, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Flag ablation. NF: no flag column. Z: flag 0 at fine-tuning and scoring (E25 and L-mix weights).",
                "F0 - L-Ww: the flag-column control against the 37-input network. Bonferroni intervals.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_2_flag_ablation.svg")

    table = (
        _t(name="bands.csv")
        .group_by("group", "band", "arm")
        .agg((pl.col("crps_sum").sum() / pl.col("n").sum()).alias("crps"))
    )
    ref = table.filter(pl.col("arm") == "F0").select("group", "band", ref=pl.col("crps"))
    diff = (
        table.join(ref, on=["group", "band"])
        .with_columns(
            diff=pl.col("crps") - pl.col("ref"), label=pl.col("group").replace_strict(GROUP_LABEL)
        )
        .filter(pl.col("arm").is_in(["E10", "E25", "L-mix", "E75", "E100"]))
        .to_pandas()
    )
    scale = alt.Scale(
        domain=["E10", "E25", "L-mix", "E75", "E100"],
        range=[
            theme.DATA_GREEN,
            theme.DATA_SKY,
            theme.DATA_BLUE,
            theme.DATA_PURPLE,
            theme.BRAND_ORANGE,
        ],
    )
    alt.Chart(diff).mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("band:N", sort=list(BANDS), title=None),
        xOffset="arm:N",
        y=alt.Y("diff:Q", title="CRPS minus F0 (points of capacity)"),
        color=alt.Color("arm:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
    ).properties(width=240, height=200).facet(column=alt.Column("label:N", title=None)).properties(  # ty: ignore[unresolved-attribute]
        title=alt.TitleParams(
            "CRPS by horizon band minus the 0% control F0 (pooled over five folds; no interval)",
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_3_horizon.svg")

    q99 = (
        pl.read_csv(SCORES / "absolute.csv")
        .filter(
            pl.col("rowset") == "main16",
            pl.col("slice").is_in(list(GROUP_LABEL)),
            pl.col("arm").is_in(list(SHARE)),
        )
        .with_columns(
            share=pl.col("arm").replace_strict(SHARE),
            label=pl.col("slice").replace_strict(GROUP_LABEL),
        )
        .to_pandas()
    )
    alt.Chart(q99).mark_line(point=True, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("share:Q", title="ENS share of every pretraining batch (%)"),
        y=alt.Y("pin_q99:Q", title="q99 pinball loss", scale=alt.Scale(zero=False)),
    ).properties(width=240, height=200).facet(column=alt.Column("label:N", title=None)).properties(  # ty: ignore[unresolved-attribute]
        title=alt.TitleParams(
            "q99 pinball loss against ENS share (points of capacity)", anchor="start"
        )
    ).save(DOCS_DIR / "figure_4_q99.svg")
    examples()


def examples() -> None:
    """Draw a day-ahead and a 10-day forecast of F0, L-mix, and E100."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    month = EXAMPLE_ORIGIN[:7]
    colours = {"F0": theme.DATA_GREEN, "L-mix": theme.DATA_BLUE, "E100": theme.BRAND_ORANGE}
    frames = {}
    for arm in colours:
        pooled = build470.pool_stored(
            folder=score.SEED_PREDS / arm, month=month, sites=sites, seeds=(0, 1)
        )
        frames[arm] = adapter.forecast_frame(
            stored=s407.widen_global(frame=pooled.drop("primary", "origin", "target")),
            series=series,
            unit="fraction",
        )
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    origin = datetime.fromisoformat(EXAMPLE_ORIGIN).replace(hour=9, tzinfo=UTC)
    for label, kind in EXAMPLES.items():
        one = {
            a: f.filter(pl.col("time_series_id") == ids[label], pl.col("power_fcst_init_time") == origin)
            .with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0).sort("valid_time")
            for a, f in frames.items()
        }  # fmt: skip
        actual = (
            ctx["actuals"]
            .filter(pl.col("time_series_id") == ids[label])
            .join(one["F0"].select("valid_time", "day"), on="valid_time")
        )
        for horizon, window in (("day-ahead", (1.0, 2.0)), ("10 days", (0.0, 10.0))):
            x = alt.X(
                "day:Q",
                title="Days after the forecast was made",
                scale=alt.Scale(domain=list(window), nice=False),
            )
            main = one["F0"].filter(pl.col("day").is_between(*window)).to_pandas()
            band = alt.Chart(main).encode(x=x)  # ty: ignore[unresolved-attribute]
            layers = [
                band.mark_area(opacity=0.2, color=colours["F0"]).encode(
                    y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
                ),
                band.mark_area(opacity=0.35, color=colours["F0"]).encode(y="q25:Q", y2="q75:Q"),
            ]  # ty: ignore[unresolved-attribute]
            for arm, f in one.items():
                sub = f.filter(pl.col("day").is_between(*window)).to_pandas()
                layers.append(
                    alt.Chart(sub)
                    .mark_line(color=colours[arm], strokeWidth=2)
                    .encode(x=x, y="q50:Q")
                )  # ty: ignore[unresolved-attribute]
            act = actual.filter(pl.col("day").is_between(*window)).to_pandas()
            layers.append(
                alt.Chart(act)
                .mark_line(color="black", strokeWidth=1.5, strokeDash=[4, 3])
                .encode(x=x, y="actual:Q")
            )  # ty: ignore[unresolved-attribute]
            alt.layer(*layers).properties(
                width=WIDTH, height=200,
                title=alt.TitleParams([f"{label} ({kind}), {horizon} forecast made on {EXAMPLE_ORIGIN}, widened 1.15.", "Bands: F0 50% and 90%. Medians: F0 green (0% ENS), L-mix blue (50%), E100 orange (100%). Dashed: actual."], anchor="start"),
            ).save(DOCS_DIR / f"figure_5_example_{label}_{horizon.replace('-', '_').replace(' ', '_')}.svg")  # fmt: skip


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bands", "figures"))
    args = parser.parse_args()
    {"bands": bands, "figures": figures}[args.command]()


if __name__ == "__main__":
    main()

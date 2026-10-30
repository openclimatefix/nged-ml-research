# ruff: noqa: E501
"""Rebuild `docs/ml-experiments/scoreboard-demand.md` and its figure from the stored scored rows.

This is the second half of the scoreboard; `scoreboard_demand_rows.py` writes the rows. Every
metric and every interval comes from `studies.eval_protocol`. Intervals are cached in the
scoreboard's private work folder, because the 99% headline contrasts take about a minute each on
one core. Delete `intervals.parquet` there to recompute them.

    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
        uv run --no-sync python scripts/scoreboard_demand_page.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

REPO: Final[Path] = Path(__file__).resolve().parents[1]
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "scoreboard-demand" / "work"
PAGE: Final[Path] = REPO / "docs" / "ml-experiments" / "scoreboard-demand.md"
FIGURE: Final[Path] = REPO / "docs" / "ml-experiments" / "scoreboard-demand-figure.svg"
HEADLINE: Final[str] = "lead_days_1_10"
BANDS: Final[dict[str, str]] = {
    "lead_day_1": "Day 1",
    "lead_day_2": "Day 2",
    "lead_days_3_4": "Days 3 to 4",
    "lead_days_5_7": "Days 5 to 7",
    "lead_days_8_10": "Days 8 to 10",
    "lead_days_11_14": "Days 11 to 14",
}
ARM_NAMES: Final[dict[str, str]] = {
    "climatology": "`climatology`",
    "manual_heuristic": "`manual_heuristic`",
    "xgboost_baseline": "`xgboost_baseline` (production-style)",
    "bg2": "BG2",
    "stack_without_s": "Stack without S",
    "default_1": "Primary demand research default 1",
    "default_2": "Primary demand research default 2",
    "default_3": "Primary demand research default 3",
}
VERSIONS: Final[tuple[str, ...]] = ("default_1", "default_2", "default_3")
CONTRAST_ARMS: Final[tuple[str, ...]] = ("stack_without_s", *VERSIONS)
SHORT_NAMES: Final[dict[str, str]] = {
    "stack_without_s": "Stack without S",
    "default_1": "Default 1",
    "default_2": "Default 2",
    "default_3": "Default 3",
}
GROUPS: Final[dict[str, tuple[str, pl.Expr]]] = {
    "signed": ("Signed-MW six (D5, D9, D10, D11, D14, D16)", pl.col("group") == "signed-MW"),
    "mva": (
        "MVA-only eight (D2, D3, D4, D8, D12, D19, D21, D22), reactive power assumed zero",
        pl.col("group") == "MVA-only",
    ),
    "mva_no_d8": (
        "MVA-only seven without D8, reactive power assumed zero",
        (pl.col("group") == "MVA-only") & (pl.col("label") != "D8"),
    ),
    "short": (
        "Short history (D15, D18), report only",
        pl.col("group") == "short-history",
    ),
}
BENCHMARK_COLOURS: Final[dict[str, str]] = {
    "BG2": ocf.DATA_BLUE,
    "manual_heuristic": ocf.DATA_PURPLE,
    "climatology": ocf.DATA_SKY,
    "xgboost_baseline": ocf.DATA_GREEN,
}
USED_COLUMNS: Final[list[str]] = [
    "arm",
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "label",
    "group",
    "crps",
    "mae_median",
    "twcrps",
    *BANDS,
    HEADLINE,
]


def load(*, row_set: str) -> pl.DataFrame:
    """Return the stored scored rows of one row set, with only the columns the page reads."""
    return pl.read_parquet(WORK / f"rows_{row_set}.parquet", columns=USED_COLUMNS)


def mean_by_group(
    *, rows: pl.DataFrame, band: str, metric: str, group: pl.Expr
) -> dict[str, float]:
    """Return the mean of `metric` over the half-hours of one band and group, for every arm."""
    out = rows.filter(pl.col(band), group).group_by("arm").agg(pl.col(metric).mean().alias("mean"))
    return dict(zip(out["arm"], out["mean"], strict=True))


def intervals(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the 95% interval of each arm's mean CRPS and the 99% contrast with BG2, days 1 to 10."""
    cache = WORK / "intervals.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    headline = rows.filter(pl.col(HEADLINE))
    pieces = []
    for key, (_, group) in GROUPS.items():
        in_group = headline.filter(group)
        bg2 = in_group.filter(pl.col("arm") == "bg2")
        zero = bg2.with_columns(crps=pl.lit(0.0))
        for arm in ARM_NAMES:
            treatment = in_group.filter(pl.col("arm") == arm)
            absolute = metrics.month_bootstrap_contrast(
                treatment=treatment, reference=zero, metric="crps"
            )
            row = {
                "group": key,
                "arm": arm,
                "mean": absolute["difference"],
                "mean_lower": absolute["lower"],
                "mean_upper": absolute["upper"],
                "n_half_hours": absolute["n_half_hours"],
            }
            if arm in CONTRAST_ARMS and key != "short":
                contrast = metrics.headline_month_bootstrap_contrast(
                    treatment=treatment, reference=bg2, metric="crps"
                )
                row |= {
                    "minus_bg2": contrast["difference"],
                    "minus_bg2_lower": contrast["lower"],
                    "minus_bg2_upper": contrast["upper"],
                }
            pieces.append(row)
            print(f"interval {key} {arm}", flush=True)
    table = pl.DataFrame(pieces, infer_schema_length=None)
    table.write_parquet(cache)
    return table


def fmt(value: float | None, digits: int = 2) -> str:
    """Format a score, or an en dash for a missing one."""
    return "–" if value is None else f"{value:.{digits}f}"


def signed(value: float) -> str:
    """Format a difference with its sign."""
    return f"{value:+.2f}"


def headline_table(*, rows: pl.DataFrame, table: pl.DataFrame, key: str) -> str:
    """Return the days 1 to 10 table of one group."""
    group = GROUPS[key][1]
    mae = mean_by_group(rows=rows, band=HEADLINE, metric="mae_median", group=group)
    tw = mean_by_group(rows=rows, band=HEADLINE, metric="twcrps", group=group)
    lines = [
        "| Forecast | CRPS (95% interval) | MAE of the median | twCRPS | Minus BG2 (99% interval) |",
        "|---|---|---|---|---|",
    ]
    for arm, name in ARM_NAMES.items():
        r = table.filter(pl.col("group") == key, pl.col("arm") == arm).row(0, named=True)
        contrast = (
            "reference"
            if arm == "bg2"
            else "–"
            if r.get("minus_bg2") is None
            else f"{signed(r['minus_bg2'])} ({signed(r['minus_bg2_lower'])} to {signed(r['minus_bg2_upper'])})"
        )
        lines.append(
            f"| {name} | {r['mean']:.2f} ({r['mean_lower']:.2f} to {r['mean_upper']:.2f}) "
            f"| {mae[arm]:.2f} | {fmt(tw.get(arm))} | {contrast} |"
        )
    return "\n".join(lines)


def band_table(*, rows_240: pl.DataFrame, rows_336: pl.DataFrame, key: str) -> str:
    """Return the CRPS by lead band of one group, with days 11 to 14 from the 336-hour rows."""
    group = GROUPS[key][1]
    by_band = {
        band: mean_by_group(rows=rows_240, band=band, metric="crps", group=group)
        for band in list(BANDS)[:-1]
    }
    late = mean_by_group(rows=rows_336, band="lead_days_11_14", metric="crps", group=group)
    header = " | ".join(BANDS.values())
    lines = [f"| Forecast | {header} |", "|---|---|---|---|---|---|---|"]
    for arm, name in ARM_NAMES.items():
        cells = [fmt(by_band[b].get(arm)) for b in list(BANDS)[:-1]]
        cells.append(fmt(late.get(arm)) if arm in late else "not on these rows")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    return "\n".join(lines)


def per_primary(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the mean CRPS of each Primary and arm over days 1 to 10."""
    return (
        rows.filter(pl.col(HEADLINE))
        .group_by("label", "group", "arm")
        .agg(pl.col("crps").mean())
        .pivot(on="arm", index=["label", "group"], values="crps")
        .sort("group", pl.col("label").str.slice(1).cast(pl.Int32))
    )


def floor_summary(*, scores: pl.DataFrame) -> tuple[str, str]:
    """Return the floor summary of each version and the per-Primary floor table of default 3."""
    fourteen = scores.filter(pl.col("group") != "short-history")
    summary = [
        "| Forecast | Primaries above 1.10 times BG2 | Worst ratio to BG2 | Primaries worse than `manual_heuristic` | Worst difference from `manual_heuristic` |",
        "|---|---|---|---|---|",
    ]
    for arm in CONTRAST_ARMS:
        frame = fourteen.with_columns(
            ratio=pl.col(arm) / pl.col("bg2"),
            minus_heuristic=pl.col(arm) - pl.col("manual_heuristic"),
        )
        worst_ratio = frame.sort("ratio", descending=True).row(0, named=True)
        worst_diff = frame.sort("minus_heuristic", descending=True).row(0, named=True)
        n_ratio = frame.filter(pl.col("ratio") > 1.10).height
        n_diff = frame.filter(pl.col("minus_heuristic") > 0).height
        summary.append(
            f"| {ARM_NAMES[arm]} | {n_ratio} of 14 | {worst_ratio['ratio']:.2f} ({worst_ratio['label']}) "
            f"| {n_diff} of 14 | {worst_diff['minus_heuristic']:+.2f} ({worst_diff['label']}) |"
        )
    detail = [
        "| Primary | Group | BG2 | `manual_heuristic` | Default 3 | Ratio to BG2 | Minus `manual_heuristic` |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in fourteen.iter_rows(named=True):
        ratio = r["default_3"] / r["bg2"]
        diff = r["default_3"] - r["manual_heuristic"]
        flag_ratio = " **above 1.10**" if ratio > 1.10 else ""
        flag_diff = " **worse**" if diff > 0 else ""
        detail.append(
            f"| {r['label']} | {r['group']} | {r['bg2']:.2f} | {r['manual_heuristic']:.2f} "
            f"| {r['default_3']:.2f} | {ratio:.2f}{flag_ratio} | {diff:+.2f}{flag_diff} |"
        )
    return "\n".join(summary), "\n".join(detail)


def per_primary_table(*, scores: pl.DataFrame) -> str:
    """Return the CRPS of every forecast at every Primary, days 1 to 10."""
    names = [ARM_NAMES[a] for a in ARM_NAMES]
    lines = [f"| Primary | Group | {' | '.join(names)} |", f"|---|---|{'---|' * len(names)}"]
    lines.extend(
        f"| {r['label']} | {r['group']} | {' | '.join(fmt(r[a]) for a in ARM_NAMES)} |"
        for r in scores.iter_rows(named=True)
    )
    return "\n".join(lines)


def wrap_prose(*, text: str) -> str:
    """Rewrap paragraphs to 100 columns, leaving tables, headings, lists, and images alone."""
    blocks = []
    for block in text.split("\n\n"):
        if block.startswith(("|", "#", "-", "!", "<")):
            blocks.append(block)
        else:
            blocks.append(textwrap.fill(" ".join(block.split("\n")), width=100))
    return "\n\n".join(blocks)


def draw_figure(*, table: pl.DataFrame) -> None:
    """Draw the cumulative-progress chart: absolute CRPS of each version beside the benchmarks."""
    order = [SHORT_NAMES[a] for a in CONTRAST_ARMS]
    panels = []
    for key in ("signed", "mva", "mva_no_d8"):
        group = table.filter(pl.col("group") == key)
        versions = group.filter(pl.col("arm").is_in(CONTRAST_ARMS)).with_columns(
            version=pl.col("arm").replace_strict(SHORT_NAMES)
        )
        bench = group.filter(pl.col("arm").is_in(list(BENCHMARK_COLOURS_ARMS))).with_columns(
            benchmark=pl.col("arm").replace_strict(BENCHMARK_LABELS)
        )
        title = {
            "signed": "Signed-MW six",
            "mva": "MVA-only eight (reactive power assumed zero)",
            "mva_no_d8": "MVA-only seven without D8 (reactive power assumed zero)",
        }[key]
        base = alt.Chart(versions.to_pandas()).encode(
            x=alt.X("version:N", sort=order, title=None, axis=alt.Axis(labelAngle=0))
        )
        interval = base.mark_rule(color=ocf.BRAND_ORANGE, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "mean_lower:Q",
                title="CRPS, days 1 to 10 (points of capacity)",
                scale=alt.Scale(zero=False),
            ),
            y2="mean_upper:Q",
        )
        dots = base.mark_point(color=ocf.BRAND_ORANGE, filled=True, size=70).encode(y="mean:Q")  # ty: ignore[unresolved-attribute]
        lines = (
            alt.Chart(bench.to_pandas())
            .mark_rule(strokeDash=[5, 3], strokeWidth=2)
            .encode(  # ty: ignore[unresolved-attribute]
                y="mean:Q",
                color=alt.Color(
                    "benchmark:N",
                    scale=alt.Scale(
                        domain=list(BENCHMARK_COLOURS), range=list(BENCHMARK_COLOURS.values())
                    ),
                    legend=alt.Legend(title="Benchmark (dashed line)", orient="top")
                    if key == "signed"
                    else None,
                ),
            )
        )
        panels.append((lines + interval + dots).properties(title=title, width=620, height=190))
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Absolute score of each Primary demand forecast, days 1 to 10, from the current rows",
            subtitle=[
                "Dot: mean CRPS (smaller is better). Orange line: 95% interval from resampling whole months.",
                "Dashed lines: the benchmarks on the same half-hours. July 2025 to June 2026, reactive power zero.",
            ],
        )
    )
    chart.save(FIGURE)
    print(f"wrote {FIGURE}")


BENCHMARK_LABELS: Final[dict[str, str]] = {
    "bg2": "BG2",
    "manual_heuristic": "manual_heuristic",
    "climatology": "climatology",
    "xgboost_baseline": "xgboost_baseline",
}
BENCHMARK_COLOURS_ARMS: Final[tuple[str, ...]] = tuple(BENCHMARK_LABELS)


def main() -> None:
    """Write the figure and the page."""
    rows_240 = load(row_set="to_240h")
    rows_336 = load(row_set="to_336h")
    table = intervals(rows=rows_240)
    draw_figure(table=table)
    scores = per_primary(rows=rows_240)
    floor_summary_md, floor_detail_md = floor_summary(scores=scores)
    n_240 = rows_240.filter(pl.col("arm") == "bg2").height
    n_336 = rows_336.filter(pl.col("arm") == "bg2").height
    sections = {key: headline_table(rows=rows_240, table=table, key=key) for key in GROUPS}
    bands = {key: band_table(rows_240=rows_240, rows_336=rows_336, key=key) for key in GROUPS}

    def cell(key: str, arm: str, field: str = "mean") -> float:
        return table.filter(pl.col("group") == key, pl.col("arm") == arm)[field].item()

    def gap_text(key: str) -> str:
        low, high = (
            cell(key, "default_3", "minus_bg2_lower"),
            cell(key, "default_3", "minus_bg2_upper"),
        )
        return f"{signed(cell(key, 'default_3', 'minus_bg2'))} ({signed(low)} to {signed(high)})"

    text = TEMPLATE.read_text().format(
        n_240=f"{n_240:,}",
        n_336=f"{n_336:,}",
        signed_bg2=f"{cell('signed', 'bg2'):.2f}",
        signed_d3=f"{cell('signed', 'default_3'):.2f}",
        signed_gap_text=gap_text("signed"),
        mva_bg2=f"{cell('mva', 'bg2'):.2f}",
        mva_d3=f"{cell('mva', 'default_3'):.2f}",
        mva_gap_text=gap_text("mva"),
        mva7_gap_text=gap_text("mva_no_d8"),
        headline_signed=sections["signed"],
        headline_mva=sections["mva"],
        headline_mva_no_d8=sections["mva_no_d8"],
        headline_short=sections["short"],
        bands_signed=bands["signed"],
        bands_mva=bands["mva"],
        bands_mva_no_d8=bands["mva_no_d8"],
        bands_short=bands["short"],
        floor_summary=floor_summary_md,
        floor_detail=floor_detail_md,
        per_primary=per_primary_table(scores=scores),
    )
    PAGE.write_text(wrap_prose(text=text))
    print(f"wrote {PAGE}")


TEMPLATE: Final[Path] = Path(__file__).resolve().parent / "scoreboard_demand_template.md"


if __name__ == "__main__":
    main()

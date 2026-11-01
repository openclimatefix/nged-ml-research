# ruff: noqa: E501
"""Draw experiment 435's figures as SVG from the tables of its scripts (aggregates only).

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00435-*/e00435_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf  # noqa: F401  (registers the theme)
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
)
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
)
WIDTH: Final[int] = 480
VARIANT_ORDER: Final[list[str]] = ["raw", "w1.15", "P"]
VARIANT_LABELS: Final[dict[str, str]] = {"raw": "Raw", "w1.15": "Widened by 1.15", "P": "Table P"}


def _weighted(*, columns: list[str]) -> list[pl.Expr]:
    return [((pl.col(c) * pl.col("n")).sum() / pl.col("n").sum()).alias(c) for c in columns]


def clean_tails() -> alt.TopLevelMixin:
    """Return q95 and q99 coverage on clean rows of each demand source, raw, widened, and P."""
    t428 = pl.read_parquet(WORK_DIR / "table_p_428.parquet").filter(
        (pl.col("arm") == "R-inj") & (pl.col("cls") == "clean") & (pl.col("basis") == "metered")
    )
    t425 = pl.read_parquet(WORK_DIR / "table_p_425.parquet").filter(
        (pl.col("arm") == "A") & (pl.col("by") == "group") & (pl.col("variant") != "P_site")
    )
    rows = []
    for label, t, key in (("428 R-inj, clean rows", t428, "group"), ("425 arm A", t425, "key")):
        rows.append(
            t.with_columns(source=pl.lit(label), grp=pl.col(key))
            .group_by("source", "grp", "variant")
            .agg(*_weighted(columns=["cov_q95", "cov_q99"]))
        )
    data = (
        pl.concat(rows)
        .filter(pl.col("grp") != "D15 and D18")
        .unpivot(index=["source", "grp", "variant"], variable_name="tail", value_name="coverage")
        .with_columns(
            variant=pl.col("variant").replace_strict(VARIANT_LABELS),
            tail=pl.col("tail").replace_strict(
                {"cov_q95": "q95 (nominal 0.95)", "cov_q99": "q99 (nominal 0.99)"}
            ),
        )
    )
    return (
        alt.Chart(data)
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "coverage:Q",
                scale=alt.Scale(domain=[0.84, 1.0]),
                title="Share of actuals at or below the quantile",
            ),
            y=alt.Y("grp:N", title=None),
            color=alt.Color("variant:N", sort=list(VARIANT_LABELS.values()), title=None),
            row=alt.Row("source:N", title=None),
            column=alt.Column("tail:N", title=None),
        )
        .properties(
            width=WIDTH // 2,
            height=60,
            title="Table P does not bring clean-row q95 coverage into 0.93 to 0.97",
        )
    )


def cold_nights() -> alt.TopLevelMixin:
    """Return q95 coverage on cold winter nights per fold at the signed-MW six, arm A."""
    t = pl.read_parquet(WORK_DIR / "cold_nights_425.parquet").filter(
        (pl.col("arm") == "A")
        & (pl.col("cls") == "cold_night")
        & (pl.col("group") == "signed-MW six")
    )
    data = (
        t.group_by("fold", "variant")
        .agg(*_weighted(columns=["cov_q95", "cov_q99"]), pl.col("n").sum())
        .unpivot(index=["fold", "variant", "n"], variable_name="tail", value_name="coverage")
        .with_columns(variant=pl.col("variant").replace_strict(VARIANT_LABELS))
    )
    base = alt.Chart(data).encode(
        x=alt.X("fold:N", title="Held-out fold"),
        y=alt.Y(
            "coverage:Q",
            scale=alt.Scale(domain=[0.7, 1.02]),
            title="Coverage on cold winter nights",
        ),
        color=alt.Color("variant:N", sort=list(VARIANT_LABELS.values()), title=None),
        column=alt.Column("tail:N", title=None),
    )
    return (base.mark_line(point=True)).properties(
        width=WIDTH // 2,
        height=220,
        title="Cold winter nights at the signed-MW six: targets are q95 at least 0.92 and q99 at least 0.975",
    )


def real_events() -> alt.TopLevelMixin:
    """Return q05 and q95 coverage of real-event rows against metered and NRA-corrected actuals."""
    t = pl.read_parquet(WORK_DIR / "table_p_428.parquet").filter(
        (pl.col("arm") == "R-inj")
        & (pl.col("cls") == "real_event")
        & (pl.col("group") == "signed-MW six")
        & (pl.col("variant") == "w1.15")
    )
    data = (
        t.group_by("basis")
        .agg(*_weighted(columns=["cov_q05", "cov_q95"]))
        .unpivot(index="basis", variable_name="tail", value_name="coverage")
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("basis:N", title="Actual used for the real-event rows"),
            y=alt.Y("coverage:Q", scale=alt.Scale(domain=[0, 1]), title="Coverage"),
            color=alt.Color("basis:N", legend=None),
            column=alt.Column("tail:N", title=None),
        )
        .properties(
            width=WIDTH // 3,
            height=200,
            title="Real events at the signed-MW six (R-inj, widened by 1.15): nominal q05 0.05, q95 0.95",
        )
    )


def main() -> None:
    """Write the figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    clean_tails().save(DOCS_DIR / "figure_1_clean_tails.svg")
    cold_nights().save(DOCS_DIR / "figure_2_cold_nights.svg")
    real_events().save(DOCS_DIR / "figure_3_real_events.svg")


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Experiment 720: the figure scripts. Every figure goes to the private folder unless it is a pooled aggregate.

Each function builds a chart from a private table and writes an SVG. `write_chart` is the only writer.
It runs `e00720_guard.refuse_dated_generator_values` on the table the chart was drawn from, and it
writes to a publishable folder only when the table also passes `e00720_guard.public_aggregate` (pooled
scores and counts, no dates, no generator values). A chart of a generator's inferred state or profile
carries generator columns, so the guard refuses to publish it and the chart stays private. Charts use
`plotting.ocf_theme` colours, are 680 px wide (the text column), have no calendar dates on any axis, and
turn off per-point accessibility text.

`python e00720_figures.py --selftest` draws each chart from synthetic tables into a temporary folder and
checks that the guard refuses a dated generator table and a column outside the allow-list.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00720_guard as guard
import polars as pl
from plotting import ocf_theme as theme

WIDTH_PX: Final[int] = 680
SKY_COLOURS: Final[dict[str, str]] = {"real": theme.DATA_BLUE, "shifted": theme.BRAND_ORANGE}


def write_chart(
    *,
    chart: Any,
    table: pl.DataFrame,
    name: str,
    publish_dir: Path | None = None,
    private_dir: Path | None = None,
) -> Path:
    """Write `chart` as SVG under the private folder, or under `publish_dir` when `table` is a public aggregate.

    Args:
        chart: The Altair chart.
        table: The table the chart was drawn from; the guard checks it.
        name: File name without extension.
        publish_dir: A publishable folder, or `None` (the default) to write privately.
        private_dir: Override of the private folder (used by the self-test).

    Raises:
        ValueError: When `publish_dir` is given and the table is not a public aggregate, or when the
            table holds a dated generator value (the guard raises in both cases).
    """
    guard.refuse_dated_generator_values(frame=table)
    if publish_dir is not None:
        guard.public_aggregate(frame=table)
        out = publish_dir / f"{name}.svg"
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = (
            (private_dir / f"{name}.svg")
            if private_dir
            else guard.private_path(name=f"figures/{name}.svg")
        )
        out.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(out))
    return out


def sign_auc_chart(*, rows: pl.DataFrame) -> alt.Chart:
    """Area under the curve of each sign read-out with the real and the shifted sky, pooled over sites and folds.

    Columns: `readout`, `sky`, `level`, `auc`. Larger is better; 0.5 is no skill.
    """
    base = alt.Chart(rows).encode(
        y=alt.Y("readout:N", title=None),
        x=alt.X(
            "auc:Q",
            title="Area under the curve (larger is better; 0.5 is no skill)",
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.Color(
            "sky:N",
            title="Sky",
            scale=alt.Scale(domain=list(SKY_COLOURS), range=list(SKY_COLOURS.values())),
            legend=alt.Legend(orient="top"),
        ),
        yOffset="sky:N",
    )
    return (
        base.mark_point(filled=True, size=70, aria=False)
        .properties(  # ty: ignore[unresolved-attribute]
            width=WIDTH_PX - 200,
            height=24 * rows["readout"].n_unique() + 40,
            title=[
                "Sign read-outs on injected worlds, real against shifted sky.",
                "Dot: pooled estimate over the six signed Primaries and folds.",
            ],
        )
        .facet(
            row=alt.Row("level:N", title="Injection level", header=alt.Header(labelOrient="top"))
        )
    )


def paired_f1_chart(*, per_cluster: pl.DataFrame) -> alt.Chart:
    """Half-hour F1 of the hindsight model against experiment 694's M2, one dot per cluster, no cluster names.

    Columns: `f1_m2`, `f1_hindsight`. Points above the diagonal favour the hindsight model.
    """
    line = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(color=theme.BLACK_2, strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    pts = (
        alt.Chart(per_cluster)
        .mark_point(filled=True, color=theme.DATA_BLUE, opacity=0.7, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("f1_m2:Q", title="M2 F1 (larger is better)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(
                "f1_hindsight:Q",
                title="Hindsight model F1 (larger is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
        )
    )
    return (line + pts).properties(
        width=WIDTH_PX - 120,
        height=320,
        title=[
            "Paired F1 on the same injected segments.",
            "Dot: one cluster of segments. Dashed line: equal F1.",
        ],
    )


def profile_chart(*, profile: pl.DataFrame) -> alt.Chart:
    """Mean half-hourly profile difference, running minus idle days, matched and unmatched (private).

    Columns: `half_hour`, `matched`, `unmatched`, `profile`. Hour axis labelled at 0, 6, 12, 18, 24 and carries no date.
    """
    long = profile.unpivot(
        index=["half_hour", "profile"],
        on=["matched", "unmatched"],
        variable_name="matching",
        value_name="difference",
    )
    return (
        alt.Chart(long)
        .mark_line(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "half_hour:Q",
                title="Half-hour of the day (hour = value / 2)",
                axis=alt.Axis(values=[0, 12, 24, 36, 48]),
            ),
            y=alt.Y("difference:Q", title="Running minus idle (MW)"),
            color=alt.Color(
                "matching:N",
                scale=alt.Scale(
                    domain=["matched", "unmatched"], range=[theme.DATA_BLUE, theme.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=WIDTH_PX - 120, height=180)
        .facet(row=alt.Row("profile:N", title=None))
    )


def selftest() -> int:
    """Draw each chart from a synthetic table, and check the guard's refusals."""
    rows = pl.DataFrame(
        {
            "readout": ["anchored", "hindsight"] * 4,
            "sky": ["real", "real", "shifted", "shifted"] * 2,
            "level": [1.0] * 4 + [2.0] * 4,
            "auc": [0.9, 0.97, 0.6, 0.7, 0.95, 0.99, 0.55, 0.7],
        }
    )
    cluster = pl.DataFrame({"f1_m2": [0.2, 0.5, 0.7, 0.9], "f1_hindsight": [0.4, 0.7, 0.9, 0.95]})
    prof = pl.DataFrame(
        {
            "half_hour": list(range(48)) * 2,
            "profile": ["reading"] * 48 + ["gross"] * 48,
            "matched": [0.1] * 96,
            "unmatched": [0.5] * 96,
        }
    )
    with tempfile.TemporaryDirectory(dir=str(Path.home() / ".cache")) as d:
        tmp = Path(d)
        a = write_chart(
            chart=sign_auc_chart(rows=rows), table=rows, name="auc", publish_dir=tmp / "pub"
        )
        b = write_chart(
            chart=paired_f1_chart(per_cluster=cluster),
            table=cluster,
            name="paired",
            private_dir=tmp / "priv",
        )
        c = write_chart(
            chart=profile_chart(profile=prof), table=prof, name="profile", private_dir=tmp / "priv"
        )
        assert all(p.exists() and p.stat().st_size > 500 for p in (a, b, c))
        try:
            write_chart(
                chart=profile_chart(profile=prof),
                table=prof,
                name="profile",
                publish_dir=tmp / "pub",
            )
        except ValueError:
            pass
        else:
            msg = "the guard let a profile table go to a publishable folder"
            raise AssertionError(msg)
        dated = pl.DataFrame({"valid_time": [1], "inferred_generator_output": [1.0]})
        try:
            write_chart(
                chart=alt.Chart(dated).mark_point(),
                table=dated,
                name="bad",
                private_dir=tmp / "priv",
            )
        except ValueError:
            pass
        else:
            msg = "the guard let a dated generator table be drawn"
            raise AssertionError(msg)
    print(
        "PASS figures: three charts drawn, the guard refuses a profile table and a dated generator table for publication"
    )
    return 0


def main() -> int:
    """Parse arguments and run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    ap.error(
        "give --selftest; the real figures are drawn by e00720_report.py from the private tables"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

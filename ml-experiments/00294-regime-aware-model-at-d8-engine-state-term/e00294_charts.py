# ruff: noqa: E501
"""Draw the figures of experiment 294 from the stored predictions and `score.json`.

uv run --no-sync python ml-experiments/00294-*/e00294_charts.py
"""

import sys
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00294_common as common
import e00294_score as sc
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_figure_guard as guard

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.DATA_DIR / "png"
WIDTH: Final[int] = 560
LABELS: Final[dict[str, str]] = {
    "arm2": "Arm 2: experiment 191's rows",
    "arm3_indicator": "Break indicator only",
    "arm3": "Arm 3: engine state, last day",
    "arm3_matched": "Post hoc: trained on lagged state",
    "arm3_hindsight": "Engine state at the target (leaky)",
    "arm4_emp": "Arm 4: bands from past residuals",
    "arm4_2b": "Arm 4b: both tails widened",
}
COLOURS: Final[dict[str, str]] = {
    "arm2": "#7A7A7A",
    "arm3_indicator": ocf.DATA_SKY,
    "arm3": ocf.DATA_BLUE,
    "arm3_matched": ocf.DATA_PURPLE,
    "arm3_hindsight": ocf.BRAND_ORANGE,
    "arm4_emp": "#222222",
    "arm4_2b": "#B5B5B5",
}


def _chart(data: pl.DataFrame) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    guard.save_chart(chart=chart, path=DOCS_DIR / f"{name}.svg")
    guard.save_chart(chart=chart, path=PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 80),
        subtitle=textwrap.wrap(subtitle, 105),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _scale(keys: list[str]) -> alt.Scale:
    return alt.Scale(domain=[LABELS[k] for k in keys], range=[COLOURS[k] for k in keys])


def d8_losses() -> tuple[dict, np.ndarray]:
    """Return D8's per-row losses of every arm and the month of each row."""
    frames = {name: sc.load_arm(config=c, kind=k, site="D8") for name, (c, k) in sc.D8_ARMS.items()}
    frames["arm4_emp"], frames["arm4_2b"] = sc.load_e264(arm="arm_emp"), sc.load_e264(arm="arm2b")
    frames = sc.align(frames=frames)
    base = frames["arm1"]
    actual = base["actual"].to_numpy().astype(np.float64)
    thr, low = float(np.quantile(actual, 0.10)), float(np.quantile(actual, 0.02))
    return {
        a: sc.row_losses(frame=f, threshold=thr, low=low) for a, f in frames.items()
    }, sc.month_of(frame=base)


def headline() -> None:
    """Relative quantile score and twCRPS of each D8 arm against arm 1."""
    losses, months = d8_losses()
    rows = []
    for arm in LABELS:
        for metric, label in (
            ("qs", "Quantile score (CRPS stand-in)"),
            ("tw", "Lower-tail twCRPS"),
        ):
            r = sc.relative_contrast(
                losses=losses, arm=arm, reference="arm1", metric=metric, months=months
            )
            rows.append(
                {
                    "arm": LABELS[arm],
                    "metric": label,
                    "est": r["relative"] * 100,
                    "lo": r["low"] * 100,
                    "hi": r["high"] * 100,
                }
            )
    data = pl.DataFrame(rows)
    order = [LABELS[k] for k in LABELS]
    base = _chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.Color("arm:N", scale=_scale(list(LABELS)), legend=None),
    )
    rule = (
        _chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        (
            base.mark_rule(strokeWidth=2).encode(
                x=alt.X("lo:Q", title="Relative to arm 1, % (negative is better)"), x2="hi:Q"
            )
            + base.mark_point(filled=True, size=70).encode(x="est:Q")
            + rule
        )
        .properties(width=WIDTH - 250, height=170)
        .facet(
            row=alt.Row(
                "metric:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=12)
            )
        )
        .properties(
            title=_title(
                title="Only the arm that sees the engine's state at the target beats arm 1; the last-day state makes the forecast worse",
                subtitle="D8, scored year, leads 0.5 to 168 hours. Dot: estimate. Line: 98.75% interval from resampling whole months. Dashed line: same as arm 1 (the single model, whole history).",
            )
        )
    )
    _save(chart, name="figure_1_relative_score")


def controls() -> None:
    """Arm 3 and the post hoc arm against arm 1 at the three Primaries without an engine."""
    rows = []
    for site in common.CONTROL_SITES:
        frames = sc.align(
            frames={
                n: sc.load_arm(config=c, kind=k, site=site) for n, (c, k) in sc.CONTROL_ARMS.items()
            }
        )
        actual = frames["arm1"]["actual"].to_numpy().astype(np.float64)
        losses = {
            a: sc.row_losses(
                frame=f,
                threshold=float(np.quantile(actual, 0.1)),
                low=float(np.quantile(actual, 0.02)),
            )
            for a, f in frames.items()
        }
        months = sc.month_of(frame=frames["arm1"])
        for arm in ("arm3_indicator", "arm3", "arm3_matched"):
            r = sc.relative_contrast(
                losses=losses, arm=arm, reference="arm1", metric="qs", months=months
            )
            rows.append(
                {
                    "site": site,
                    "arm": LABELS[arm],
                    "est": r["relative"] * 100,
                    "lo": r["low"] * 100,
                    "hi": r["high"] * 100,
                }
            )
    keys = ["arm3_indicator", "arm3", "arm3_matched"]
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=[LABELS[k] for k in keys], title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.Color("arm:N", scale=_scale(keys), legend=None),
    )
    rule = (
        _chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#555555", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        (
            base.mark_rule(strokeWidth=2).encode(
                x=alt.X("lo:Q", title="Quantile score relative to arm 1, % (negative is better)"),
                x2="hi:Q",
            )
            + base.mark_point(filled=True, size=70).encode(x="est:Q")
            + rule
        )
        .properties(width=WIDTH - 250, height=90)
        .facet(
            row=alt.Row(
                "site:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=12)
            )
        )
        .properties(
            title=_title(
                title="At the three Primaries without an engine, the engine term moves the score by under 2%",
                subtitle="D5, D10, D11, scored year. Dot: estimate. Line: 98.75% interval from resampling whole months. Dashed line: same as arm 1 at that Primary.",
            )
        )
    )
    _save(chart, name="figure_5_controls")


def main() -> None:
    """Draw every figure.

    The engine offset per refit and the example weeks are not drawn, because they show the
    inferred engine state or its fitted size on calendar dates.
    """
    headline()
    controls()


if __name__ == "__main__":
    main()

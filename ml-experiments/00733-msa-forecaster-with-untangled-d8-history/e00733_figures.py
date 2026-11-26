# ruff: noqa: E501
"""Experiment 733: the result figures that read stored files (loss curves, contrasts.

The sign forecast).

Every figure is an Altair chart that `save_svg` writes as SVG, to be optimised with `svgo`. Each
figure's data is built by a function that returns the plotted rows, so a check can read what the
chart
draws. No chart carries a series identifier or a generator value; the sign-forecast panels carry
synthetic days only.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_fit as fit
import e00733_paths as paths  # noqa: F401
import numpy as np
import polars as pl

# isort: split
import altair as alt
import e00729_figures as figures729

WIDTH_PX: Final = figures729.WIDTH_PX
save_svg = figures729.save_svg
ARM_COLOURS: Final = {
    "M*": figures729.GREY, "F0r": "#999999", "FU": figures729.BLUE, "FU-S": figures729.ORANGE, "FS": figures729.GREEN,
    "Y0": figures729.GREY, "YU": figures729.BLUE, "YT": figures729.PURPLE, "B-hist": figures729.ORANGE, "B-clim": "#999999", "O": figures729.SKY,
}  # fmt: skip
"""One colour per arm, so an arm keeps its colour in every figure."""
LOSS_CURVE_SEED: Final = 0


class HistoryError(ValueError):
    """A stored fit record has no usable per-pass history."""


def curve_rows(*, record: Mapping[str, object]) -> pl.DataFrame:
    """Return one fit's plotted rows: panel, pass, split (train or validation), term, value.

    And the stored pass.

    Args:
        record: A fit record (`fits/<job>.json`).

    Returns:
        The rows, one per panel (stage), pass, split, and term.

    Raises:
        HistoryError: When the record holds no pass, a curve is not finite, or the stored pass lies
            outside the passes run.
    """
    try:
        data = fit.loss_curve_data(record=record)  # ty: ignore[invalid-argument-type]
    except (ValueError, KeyError) as exc:
        raise HistoryError(str(exc)) from exc
    parts = []
    for series in data["series"]:
        n = len(series["val"])
        if len(series["train"]) != n or not np.isfinite(series["train"] + series["val"]).all():
            msg = "a loss curve is not as long as the validation curve, or holds a non-finite value"
            raise HistoryError(msg)
        if not 1 <= int(series["stored_pass"]) <= n:
            msg = "the stored pass lies outside the passes run"
            raise HistoryError(msg)
        terms = {"train": ("loss", series["train"]), "validation": ("loss", series["val"])}
        if series.get("val_nll"):
            terms["validation mixture"] = ("mixture negative log-likelihood", series["val_nll"])
            terms["validation sign"] = ("sign cross-entropy", series["val_bce"])
        for split, (term, values) in terms.items():
            parts.append(
                pl.DataFrame({
                    "panel": series["label"], "pass": np.arange(1, n + 1), "split": split, "term": term,
                    "value": np.asarray(values, dtype=float), "stored_pass": int(series["stored_pass"]),
                })
            )  # fmt: skip
    return pl.concat(parts)


def loss_panel(*, record: Mapping[str, object], title: str) -> alt.LayerChart:
    """Return one fit's panel of loss per pass (train solid, validation dashed).

    With the stored pass marked.
    """
    rows = curve_rows(record=record)
    splits = list(dict.fromkeys(rows["split"].to_list()))
    lines = (
        alt.Chart(rows)
        .mark_line(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("pass:Q", title="Pass"),
            y=alt.Y("value:Q", title="Loss", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(
                    domain=splits,
                    range=[figures729.BLUE, figures729.ORANGE, figures729.GREEN, figures729.PURPLE][
                        : len(splits)
                    ],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash("panel:N", legend=None),
        )
    )
    mark = (
        alt.Chart(rows.select("stored_pass", "panel").unique())  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#000000", strokeDash=[2, 2], aria=False)
        .encode(x="stored_pass:Q")
    )  # ty: ignore[unresolved-attribute]
    return (lines + mark).properties(
        width=WIDTH_PX, height=150, title=alt.TitleParams(title, anchor="start")
    )


def loss_curves_chart(*, records: Sequence[Mapping[str, object]]) -> alt.VConcatChart:
    """Return the figure: one panel per fit record, stacked, each marking the stored pass."""
    if not records:
        msg = "no fit record to draw loss curves from"
        raise HistoryError(msg)
    panels = [
        loss_panel(
            record=r,
            title=f"{r['arm']}, {r['fold'] or 'synthetic world'}, seed {r['seed']}: stored pass {_stored(r)}",
        )
        for r in records
    ]
    return alt.vconcat(*panels)


def _stored(record: Mapping[str, object]) -> object:
    if "stored_pass" in record:
        return record["stored_pass"]
    return "/".join(str(s["stored_pass"]) for s in record["stages"])  # ty: ignore[invalid-argument-type, not-iterable]


def pick_loss_records(*, root: Path) -> list[dict]:
    """Return the fit records the loss-curve figure draws: seed 0 of each arm.

    The first fold on the real folds.

    Args:
        root: The data folder.

    Returns:
        The fit records, in `core.FIT_ORDER`.

    Raises:
        HistoryError: When no fit record exists.
    """
    folder = root / "fits"
    picked: list[dict] = []
    for arm in core.FIT_ORDER:
        name = "FU" if arm == "FU-screen" else arm
        if any(r["arm"] == name for r in picked):
            continue
        records = [
            json.loads(f.read_text())
            for f in sorted(folder.glob("*.json"))
            if ".failed" not in f.name
        ]
        mine = [r for r in records if r.get("arm") == name and r.get("seed") == LOSS_CURVE_SEED]
        if mine:
            picked.append(mine[0])
    if not picked:
        msg = "no fit record exists, so there is no loss curve to draw"
        raise HistoryError(msg)
    return picked


def contrast_rows(*, readings: Mapping[str, Any]) -> pl.DataFrame:
    """Return the rows of a contrast dot chart from `readings.json` entries.

    An entry with `relative` or with `point`, `lower`, and `upper` gives a row; void entries are left out.
    """
    rows = []
    for name, v in readings.items():
        if not isinstance(v, Mapping):
            continue
        r = v.get("relative", v)
        if all(k in r for k in ("point", "lower", "upper")):
            rows.append(
                {
                    "contrast": name,
                    "point": float(r["point"]),
                    "lower": float(r["lower"]),
                    "upper": float(r["upper"]),
                }
            )
    return pl.DataFrame(
        rows,
        schema={"contrast": pl.Utf8, "point": pl.Float64, "lower": pl.Float64, "upper": pl.Float64},
    )


def contrast_dot_chart(
    *, readings: Mapping[str, Any], names: Sequence[str], title: str, x_title: str
) -> alt.Chart:
    """Return a dot chart of the named contrasts with their intervals, one colour per contrast.

    And a rule at zero.
    """
    data = contrast_rows(readings=readings).filter(pl.col("contrast").is_in(list(names)))
    if data.is_empty():
        msg = "none of the named contrasts has a reading to draw"
        raise HistoryError(msg)
    order = [n for n in names if n in set(data["contrast"].to_list())]
    colours = [
        figures729.BLUE,
        figures729.ORANGE,
        figures729.GREEN,
        figures729.PURPLE,
        figures729.SKY,
        figures729.GREY,
    ]
    base = alt.Chart(data).encode(y=alt.Y("contrast:N", sort=order, title=None))  # ty: ignore[unresolved-attribute]
    color = alt.Color(
        "contrast:N",
        scale=alt.Scale(domain=order, range=[colours[i % len(colours)] for i in range(len(order))]),
        legend=None,
    )
    bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title=x_title), x2="upper:Q", color=color
    )  # ty: ignore[unresolved-attribute]
    dots = base.mark_circle(size=70).encode(x="point:Q", color=color)  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    return (bars + dots + zero).properties(
        width=WIDTH_PX, height=26 * len(order), title=alt.TitleParams(title, anchor="start")
    )


def sign_series_rows(*, truth: np.ndarray, forecasts: Mapping[str, np.ndarray]) -> pl.DataFrame:
    """Return the rows of the sign-forecast panel for one synthetic segment: half-hour index.

    Series name, and value.

    Args:
        truth: The indicator of X < 0 over the target days `(n,)`.
        forecasts: Each forecast's probability over the same half-hours (YU, Y0, B-hist).

    Returns:
        The rows.

    Raises:
        HistoryError: When a series is not as long as the truth.
    """
    parts = [
        pl.DataFrame(
            {
                "half_hour": np.arange(len(truth)),
                "series": "true sign",
                "value": np.asarray(truth, dtype=float),
            }
        )
    ]
    for name, p in forecasts.items():
        if len(p) != len(truth):
            msg = f"the forecast {name} is not as long as the truth"
            raise HistoryError(msg)
        parts.append(
            pl.DataFrame(
                {
                    "half_hour": np.arange(len(truth)),
                    "series": name,
                    "value": np.asarray(p, dtype=float),
                }
            )
        )
    return pl.concat(parts)


def sign_series_chart(*, rows: pl.DataFrame, title: str) -> alt.Chart:
    """Return the line chart of forecast probabilities against the true sign.

    The chart covers one synthetic segment.
    """
    names = list(dict.fromkeys(rows["series"].to_list()))
    return (
        alt.Chart(rows)
        .mark_line(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("half_hour:Q", title="Half-hour of the 10 target days"),
            y=alt.Y("value:Q", title="Probability of reverse flow", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=names, range=[ARM_COLOURS.get(n, "#555555") for n in names]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH_PX, height=120, title=alt.TitleParams(title, anchor="start"))
    )


def reliability_rows(
    *, p: np.ndarray, truth: np.ndarray, n_bins: int = 10
) -> tuple[list[float], list[float]]:
    """Return the mean forecast and the observed share of each bin.

    The bins have equal width, and only a bin that holds a half-hour gives a point.
    """
    p, t = np.asarray(p, dtype=float).ravel(), np.asarray(truth, dtype=float).ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    xs, ys = [], []
    for k in range(n_bins):
        m = b == k
        if m.any():
            xs.append(float(p[m].mean()))
            ys.append(float(t[m].mean()))
    return xs, ys


def write_loss_curves(*, root: Path, out: Path) -> Path:
    """Write the loss-curve figure from the stored fit records of seed 0 of each arm.

    Args:
        root: The data folder.
        out: The SVG path.

    Returns:
        The path written.
    """
    save_svg(chart=loss_curves_chart(records=pick_loss_records(root=root)), path=out)
    return out

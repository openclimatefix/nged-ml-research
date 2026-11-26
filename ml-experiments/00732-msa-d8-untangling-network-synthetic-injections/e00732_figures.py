# ruff: noqa: E501
"""Experiment 732: the result figures that read stored files only (the loss curves).

Every figure is an Altair chart that `save_svg` writes as SVG, to be optimised with `svgo`. The loss
curves read the stored per-epoch histories of the fit records and refit nothing. Each figure's data is
built by a function that returns the plotted arrays, so a check can read what the chart draws.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import altair as alt
import e00729_figures as figures729
import e00732_core as core
import numpy as np
import polars as pl

WIDTH_PX: Final = figures729.WIDTH_PX
save_svg = figures729.save_svg
TERM_COLOURS: Final = {
    "total": figures729.GREY, "sign": figures729.BLUE, "solar": figures729.ORANGE, "step": figures729.GREEN,
    "size": figures729.SKY, "injection": figures729.PURPLE, "night": "#7A7A7A", "monotone": "#1F3A93",
}  # fmt: skip
"""One colour per loss term (the main data colours of the brand guidelines, then neutral shades)."""
CURVE_ARMS: Final = tuple(core.ARMS)
LOSS_CURVE_SEED: Final = 0
"""The seed whose history each panel shows (one seed per arm)."""


class HistoryError(ValueError):
    """A stored fit record has no usable per-epoch history."""


def curve_rows(*, record: Mapping[str, object]) -> pl.DataFrame:
    """Return one fit's plotted rows: epoch, split ("train" or "validation"), term, and value.

    Args:
        record: The fit record (`fits/<name>.json`) with `train_terms`, `val_terms`, `epochs`, and `best_epoch`.

    Returns:
        The rows, one per epoch, split, and term.

    Raises:
        HistoryError: When a history is empty, the two splits hold different terms, a curve is not
            as long as `epochs`, a
            value is not finite, or `best_epoch` is outside the epochs run.
    """
    train, val = record.get("train_terms") or {}, record.get("val_terms") or {}
    if not train or not val:
        msg = f"the fit record of {record.get('arm')} seed {record.get('seed')} holds no per-epoch loss history"
        raise HistoryError(msg)
    if set(train) != set(val):  # ty: ignore[invalid-argument-type]
        msg = "the training and validation histories hold different loss terms"
        raise HistoryError(msg)
    n = int(record["epochs"])  # ty: ignore[invalid-argument-type]
    if not 1 <= int(record["best_epoch"]) <= n:  # ty: ignore[invalid-argument-type]
        msg = "best_epoch lies outside the epochs run"
        raise HistoryError(msg)
    parts = []
    for split, terms in (("train", train), ("validation", val)):
        for term, values in terms.items():  # ty: ignore[unresolved-attribute]
            v = np.asarray(values, dtype=float)
            if len(v) != n or not np.isfinite(v).all():
                msg = f"the {split} history of term {term} is not {n} finite values"
                raise HistoryError(msg)
            parts.append(
                pl.DataFrame(
                    {"epoch": np.arange(1, n + 1), "split": split, "term": term, "value": v}
                )
            )
    return pl.concat(parts)


def best_epoch_of(*, record: Mapping[str, object]) -> int:
    """Return the marked epoch of a fit: its stored `best_epoch`.

    Checked against the lowest validation Brier score of the sign.
    """
    history = np.asarray(record.get("history") or [], dtype=float)
    if history.size == 0:
        msg = "the fit record holds no validation history of the sign"
        raise HistoryError(msg)
    best = int(record["best_epoch"])  # ty: ignore[invalid-argument-type]
    if best != int(np.argmin(history)) + 1:
        msg = (
            "best_epoch differs from the epoch of the lowest validation Brier score in the history"
        )
        raise HistoryError(msg)
    return best


def loss_panel(*, record: Mapping[str, object]) -> alt.LayerChart:
    """Return one arm's panel of train (solid) and validation (dashed) curves per term.

    The panel marks the epoch of the stored weights.
    """
    rows = curve_rows(record=record)
    best = best_epoch_of(record=record)
    terms = list(dict.fromkeys(rows["term"].to_list()))
    lines = (
        alt.Chart(rows)
        .mark_line(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y(
                "value:Q",
                title="Loss term (symmetric log scale)",
                scale=alt.Scale(type="symlog", constant=1e-3),
            ),
            color=alt.Color(
                "term:N",
                scale=alt.Scale(domain=terms, range=[TERM_COLOURS[t] for t in terms]),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash(
                "split:N",
                scale=alt.Scale(domain=["train", "validation"], range=[[1, 0], [5, 3]]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    mark = (
        alt.Chart(pl.DataFrame({"epoch": [best], "label": [f"stored model: epoch {best}"]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#000000", strokeDash=[2, 2], aria=False)
        .encode(x="epoch:Q")
    )  # ty: ignore[unresolved-attribute]
    title = f"{record['arm']}, seed {record['seed']}, {record['arch']}: stored model at epoch {best} of {record['epochs']}"
    return (lines + mark).properties(
        width=WIDTH_PX, height=150, title=alt.TitleParams(title, anchor="start")
    )


def loss_curves_chart(*, records: Sequence[Mapping[str, object]]) -> alt.VConcatChart:
    """Return the figure: one panel per arm, stacked, each from one seed's stored history."""
    if not records:
        msg = "no fit record to draw loss curves from"
        raise HistoryError(msg)
    panels = [loss_panel(record=r) for r in records]
    title = alt.TitleParams(
        [
            "Training and validation loss terms by epoch, one fit per arm",
            f"Seed {LOSS_CURVE_SEED} of each arm; solid: training batches, dashed: one fixed set of validation triples (the validation transfer overlay)",
        ],
        anchor="start",
    )
    return alt.vconcat(*panels, title=title).resolve_scale(color="independent")


def loss_curve_checks(*, records: Sequence[Mapping[str, object]]) -> dict[str, bool]:
    """Return the checks on the plotted arrays: every arm drawn, every curve finite and long enough.

    Every marker equal to the stored best epoch.
    """
    out = {"every_arm_drawn": {r["arm"] for r in records} == set(CURVE_ARMS)}
    for r in records:
        rows = curve_rows(record=r)
        out[f"{r['arm']}_markers_match"] = best_epoch_of(record=r) == int(r["best_epoch"])  # ty: ignore[invalid-argument-type]
        out[f"{r['arm']}_has_sign_curves"] = {"train", "validation"} <= set(
            rows.filter(pl.col("term") == "sign")["split"].to_list()
        )
    return out


def stage_loss_curves(*, root: Path, log: Callable[[str], None]) -> list[Path]:
    """Draw `figures/loss_curves.svg` from the stored fit records of seed 0 of every arm (no refit).

    Args:
        root: The data folder.
        log: Receives progress lines.

    Returns:
        The paths of the figures written.

    Raises:
        HistoryError: When a record is missing or its history is empty.
        FileNotFoundError: When an arm has no record of seed 0.
    """
    import json

    records = []
    for arm, spec in core.ARMS.items():
        path = (
            root / "fits" / f"{core.job_name(arm=arm, arch=spec.arch, seed=LOSS_CURVE_SEED)}.json"
        )
        if not path.exists():
            msg = f"no fit record {path.name}: the loss curves need the stored history of seed {LOSS_CURVE_SEED} of every arm"
            raise FileNotFoundError(msg)
        records.append(json.loads(path.read_text()))
    checks = loss_curve_checks(records=records)
    if not all(checks.values()):
        msg = f"the loss-curve checks failed: {[k for k, v in checks.items() if not v]}"
        raise HistoryError(msg)
    path = root / "figures" / "loss_curves.svg"
    save_svg(chart=loss_curves_chart(records=records), path=path)
    log(f"loss curves drawn for {len(records)} arms; checks {sum(checks.values())}/{len(checks)}")
    return [path]


def contrast_rows(*, readings: Mapping[str, Mapping[str, float | bool]]) -> pl.DataFrame:
    """Return one row per counted contrast, with its point and interval ends.

    Each row also says whether the reading is void (a void reading draws nothing).
    """
    return pl.DataFrame(
        [
            {"contrast": name, "point": float(r["point"]), "lower": float(r["lower"]), "upper": float(r["upper"]), "void": bool(r["void"])}
            for name, r in readings.items()
        ],
        schema={"contrast": pl.Utf8, "point": pl.Float64, "lower": pl.Float64, "upper": pl.Float64, "void": pl.Boolean},
    )  # fmt: skip


def contrast_chart(*, readings: Mapping[str, Mapping[str, float | bool]]) -> alt.LayerChart:
    """Return the counted contrasts as points with intervals.

    The chart draws a labelled zero rule (a negative value favours the first arm).

    Args:
        readings: The counted contrasts' readings by name (`reading_dict` output).

    Returns:
        The chart.

    Raises:
        HistoryError: When every reading is void, because the chart would be empty.
    """
    data = contrast_rows(readings=readings).filter(~pl.col("void"))
    if data.height == 0:
        msg = "every counted contrast is void"
        raise HistoryError(msg)
    base = alt.Chart(data).encode(y=alt.Y("contrast:N", title=None, sort=list(readings)))
    bars = base.mark_rule(color=figures729.BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Contrast in mean score (negative: the first arm is better; Bonferroni 99.286% interval)",
        ),
        x2="upper:Q",
    )  # ty: ignore[unresolved-attribute]
    dots = base.mark_point(filled=True, color=figures729.BLUE, size=60, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q"
    )  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#000000", strokeDash=[3, 3], aria=False)
        .encode(x="z:Q")
    )  # ty: ignore[unresolved-attribute]
    return (bars + dots + zero).properties(
        width=WIDTH_PX,
        height=160,
        title=alt.TitleParams(
            "The seven counted contrasts C1 to C7, with the zero rule", anchor="start"
        ),
    )

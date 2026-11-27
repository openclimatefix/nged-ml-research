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
        .mark_line(aria=False, point=True)
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
    entries: dict[str, Any] = {}
    for name, v in readings.items():
        if isinstance(v, Mapping) and isinstance(v.get("bands"), Mapping):
            entries.update({f"{name}, {band}": b for band, b in v["bands"].items()})
        else:
            entries[name] = v
    for name, v in entries.items():
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
            x=alt.X("half_hour:Q", title="Half-hour of the target days (lead days 2 to 10)"),
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


# --------------------------------------------------------------------------- part 5c: the `figures` stage

REAL_LOSS_ARMS: Final = ("F0r", "FU", "FU-S", "FS")
SYNTHETIC_LOSS_ARMS: Final = ("Y0", "YU", "YT")
CONTRAST_GROUPS: Final = {
    "real": (("C1", "C2", "R1", "R2", "R3", "R4", "R8"), "Relative CRPS difference at D8 (negative: first arm better)"),
    "synthetic": (("C3", "C4", "R5", "R7"), "Difference in Brier score of the probability of reverse flow (negative: first arm better)"),
}  # fmt: skip
"""The contrasts of each dot chart (an R8 row is one lead band) and the axis title."""
COVERAGE_LEVELS: Final = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def pick_loss_groups(*, root: Path) -> dict[str, list[dict]]:
    """Return the fit records of the loss-curve figure by arm: seed 0 of each fold, or every seed.

    The real-fold arms (F0r, FU, FU-S, FS) are drawn at seed 0 in every fold they were fitted; the
    synthetic arms (Y0, YU, YT) are drawn at every seed.

    Args:
        root: The data folder.

    Returns:
        The records of each arm that has any, in `REAL_LOSS_ARMS` then `SYNTHETIC_LOSS_ARMS` order.

    Raises:
        HistoryError: When no fit record exists.
    """
    records = [
        json.loads(f.read_text())
        for f in sorted((root / "fits").glob("*.json"))
        if ".failed" not in f.name
    ]
    groups: dict[str, list[dict]] = {}
    for arm in (*REAL_LOSS_ARMS, *SYNTHETIC_LOSS_ARMS):
        mine = [r for r in records if r.get("arm") == arm]
        if arm in REAL_LOSS_ARMS:
            mine = [r for r in mine if r.get("seed") == LOSS_CURVE_SEED]
        mine.sort(key=lambda r: (str(r.get("fold")), int(r.get("seed", 0))))
        if mine:
            groups[arm] = mine
    if not groups:
        msg = "no fit record exists, so there is no loss curve to draw"
        raise HistoryError(msg)
    return groups


def loss_grid_chart(*, groups: Mapping[str, Sequence[Mapping[str, object]]]) -> alt.VConcatChart:
    """Return the loss figure: one row per arm, one panel per fold (real) or seed (synthetic).

    Each panel draws the training loss and the validation loss against the pass and marks the stored pass.
    """
    rows = []
    for arm, records in groups.items():
        n = len(records)
        panels = [
            loss_panel(
                record=r,
                title=f"{arm}, {r['fold'] or 'seed ' + str(r['seed'])}: stored pass {_stored(r)}",
            ).properties(width=max(WIDTH_PX // max(n, 1) - 60, 120), height=120)
            for r in records
        ]
        rows.append(alt.hconcat(*panels))
    return alt.vconcat(*rows)


def write_loss_grid(*, root: Path, out: Path) -> dict[str, int]:
    """Write the loss-curve figure of every fitted arm and return the number of panels by arm."""
    groups = pick_loss_groups(root=root)
    save_svg(chart=loss_grid_chart(groups=groups), path=out)
    return {arm: len(records) for arm, records in groups.items()}


def coverage_rows(*, r9: Mapping[str, Any]) -> pl.DataFrame:
    """Return the observed share below each tail quantile, by arm (R9), with the nominal level."""
    rows = []
    for arm, table in (r9.get("arms") or {}).items():
        for q, level in COVERAGE_LEVELS.items():
            value = table.get(f"coverage_{q}")
            if value is not None and np.isfinite(value):
                rows.append(
                    {
                        "arm": arm,
                        "quantile": q,
                        "nominal": level,
                        "observed": float(value),
                        "n": int(table["n"]),
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "arm": pl.Utf8,
            "quantile": pl.Utf8,
            "nominal": pl.Float64,
            "observed": pl.Float64,
            "n": pl.Int64,
        },
    )


def coverage_chart(*, r9: Mapping[str, Any]) -> alt.LayerChart:
    """Return the coverage figure: observed share of readings below each tail quantile of D8.

    The share is drawn against the nominal level, one dot per arm and quantile, with the diagonal
    of perfect coverage.

    Args:
        r9: The R9 reading.

    Returns:
        The chart.

    Raises:
        HistoryError: When R9 holds no coverage.
    """
    data = coverage_rows(r9=r9)
    if data.is_empty():
        msg = "R9 holds no coverage to draw"
        raise HistoryError(msg)
    arms = list(dict.fromkeys(data["arm"].to_list()))
    line = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#555555", strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    dots = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=80)
        .encode(
            x=alt.X(
                "nominal:Q", title="Nominal level of the quantile", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y(
                "observed:Q",
                title="Share of readings below the quantile",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=arms, range=[ARM_COLOURS.get(a, "#555555") for a in arms]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )  # ty: ignore[unresolved-attribute]
    return (line + dots).properties(
        width=WIDTH_PX // 2,
        height=WIDTH_PX // 2,
        title=alt.TitleParams("Coverage of D8's tail quantiles", anchor="start"),
    )


def reliability_chart(*, curves: Mapping[str, tuple[list[float], list[float]]]) -> alt.LayerChart:
    """Return the reliability diagram of the forecasts of reverse flow: observed share by mean forecast.

    Args:
        curves: Each forecast's `reliability_rows`.

    Returns:
        The chart, with the diagonal of a reliable forecast.

    Raises:
        HistoryError: When no forecast has a bin that holds a half-hour.
    """
    parts = [
        pl.DataFrame(
            {"forecast": name, "mean_forecast": xs, "observed": ys},
            schema={"forecast": pl.Utf8, "mean_forecast": pl.Float64, "observed": pl.Float64},
        )
        for name, (xs, ys) in curves.items()
        if xs
    ]
    if not parts:
        msg = "no forecast has a half-hour in any reliability bin"
        raise HistoryError(msg)
    data = pl.concat(parts)
    names = list(dict.fromkeys(data["forecast"].to_list()))
    colour = alt.Color(
        "forecast:N",
        scale=alt.Scale(domain=names, range=[ARM_COLOURS.get(n, "#555555") for n in names]),
        legend=alt.Legend(orient="top", title=None),
    )
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#555555", strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    base = alt.Chart(data).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "mean_forecast:Q",
            title="Mean forecast probability of reverse flow",
            scale=alt.Scale(domain=[0, 1]),
        ),
        y=alt.Y(
            "observed:Q", title="Observed share of reverse flow", scale=alt.Scale(domain=[0, 1])
        ),
        color=colour,
    )
    return (diagonal + base.mark_line(aria=False) + base.mark_circle(size=60)).properties(
        width=WIDTH_PX // 2,
        height=WIDTH_PX // 2,
        title=alt.TitleParams("Reliability of the forecast of reverse flow", anchor="start"),
    )


def sign_reliability_curves(*, data: Any) -> dict[str, tuple[list[float], list[float]]]:
    """Return the reliability bins of YU, Y0, and B-hist over the scored half-hours of the bank."""
    keep = data.keep
    return {
        name: reliability_rows(p=data.probabilities[name][keep], truth=data.truth[keep])
        for name in ("YU", "Y0", "B-hist")
    }


def run_figures(*, root: Path, rows: str) -> dict[str, Any]:
    """Draw every figure that reads stored files and return what was written, with its checks.

    The figures are the loss curves, the two contrast dot charts, the coverage of D8's tail
    quantiles, and the reliability diagram of the forecast of reverse flow.

    Args:
        root: The data folder.
        rows: "test" or "validation" (the rows the sign forecasts are read from).

    Returns:
        `files` (the SVG names written), `loss_panels` (the panels by arm), and `contrasts` (the
        rows each dot chart drew).

    Raises:
        HistoryError: When a figure has nothing to draw.
    """
    import e00733_signdata as signdata

    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    readings = json.loads((root / "scores" / "readings.json").read_text())
    result: dict[str, Any] = {"files": [], "contrasts": {}}
    result["loss_panels"] = write_loss_grid(root=root, out=out / "loss_curves.svg")
    result["files"].append("loss_curves.svg")
    every = contrast_rows(readings=readings)["contrast"].to_list()
    for name, (heads, x_title) in CONTRAST_GROUPS.items():
        chosen = [c for c in every if c.split(",")[0] in heads]
        chart = contrast_dot_chart(
            readings=readings, names=chosen, title=f"Contrasts ({name} forecasts)", x_title=x_title
        )
        save_svg(chart=chart, path=out / f"contrasts_{name}.svg")
        result["files"].append(f"contrasts_{name}.svg")
        result["contrasts"][name] = chosen
    save_svg(chart=coverage_chart(r9=readings["R9"]), path=out / "coverage.svg")
    result["files"].append("coverage.svg")
    data = signdata.load_sign_data(root=root, rows=rows)
    curves = sign_reliability_curves(data=data)
    save_svg(chart=reliability_chart(curves=curves), path=out / "reliability_sign.svg")
    result["files"].append("reliability_sign.svg")
    result["reliability_bins"] = {name: len(xs) for name, (xs, _) in curves.items()}
    return result

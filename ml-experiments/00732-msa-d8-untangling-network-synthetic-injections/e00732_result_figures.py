# ruff: noqa: E501
"""Experiment 732: the result figures 1, 3, 4, 5, 6, 7, 8, and 9, drawn from the scored test rows.

Every function takes the test bank's truth and the seed-averaged outputs of each arm (the flat rows
of `e00732_score`) and returns an Altair chart. A figure that has no fitted arm to draw raises
`FiguresError` rather than drawing an empty plot, and an arm that lacks an output is left out of a
figure (the caption of each chart lists the arms drawn). No figure shows a calendar date or a level
of the real solar output.
"""

from collections.abc import Mapping
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import altair as alt
import e00730_score as score730
import e00732_core as core
import e00732_score as score
import numpy as np
import polars as pl

WIDTH_PX: Final = 640
COLOURS: Final = (
    "#FF4901", "#306BFF", "#10C5F7", "#B701FF", "#17E58F", "#009C75", "#FC9700", "#666666", "#000000",
    "#8B5E3C", "#E91E63", "#795548",
)  # fmt: skip
"""One colour per arm: the five main data colours first, then the internal-use ones."""
RELIABILITY_EDGES: Final = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0000001)
"""The bin edges of the reliability diagram's predicted probability."""
RECOVERY_BINS: Final = 8
"""The number of equal-count bins of delta in the recovery figure."""
SCENARIO_COLUMNS: Final = ("brier_sign", "solar_mae", "brier_step")


class FiguresError(RuntimeError):
    """A figure needs fitted outputs that do not exist (no empty plot is drawn)."""


def _domain(*, names: list[str]) -> alt.Scale:
    return alt.Scale(domain=names, range=list(COLOURS[: len(names)]))


def dot_chart(
    *, data: pl.DataFrame, x: str, y: str, color: str, title: str, x_title: str, height: int = 160
) -> alt.Chart:
    """Return dots of one value per row and arm, with the arm's name in the legend.

    Args:
        data: The rows.
        x: The value column.
        y: The row-label column.
        color: The arm column.
        title: The chart's title (what is plotted, and in which direction is better).
        x_title: The value axis' title.
        height: The chart's height in pixels.

    Returns:
        The chart.

    Raises:
        FiguresError: When the data holds no finite value.
    """
    data = data.filter(pl.col(x).is_finite())
    if data.height == 0:
        msg = f"{title}: no finite value to draw (no fitted arm has this output)"
        raise FiguresError(msg)
    names = sorted(set(data[color].to_list()))
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, aria=False)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title),
            y=alt.Y(f"{y}:N", title=None, sort=list(dict.fromkeys(data[y].to_list()))),
            color=alt.Color(
                f"{color}:N",
                scale=_domain(names=names),
                legend=alt.Legend(orient="top", title=None),
            ),
            yOffset=alt.YOffset(f"{color}:N"),
        )
        .properties(width=WIDTH_PX, height=height, title=alt.TitleParams(title, anchor="start"))
    )


def _arms_with(*, arms: Mapping[str, score.ArmScores], field: str) -> dict[str, np.ndarray]:
    return {k: getattr(a, field) for k, a in arms.items() if getattr(a, field) is not None}


def fig1_brier_by_group(*, truth: score.Truth, arms: Mapping[str, score.ArmScores]) -> alt.Chart:
    """Figure 1: the Brier score of P(X<0) of each arm.

    The score is read on all rows, on import rows, on reverse-flow rows, and by reading band.
    """
    rows = truth.rows
    groups = {
        "all rows": np.ones(rows.n, dtype=bool),
        "import": rows.x >= 0.0,
        "reverse flow": rows.x < 0.0,
    }
    groups |= {f"import, band {k}": v for k, v in score.band_masks(rows=rows).items()}
    records = []
    for name, p in _arms_with(arms=arms, field="p").items():
        brier = score730.row_brier(p=p, x=rows.x)
        records += [
            {"arm": name, "group": g, "value": float(brier[m].mean()) if m.any() else float("nan")}
            for g, m in groups.items()
        ]
    return dot_chart(
        data=pl.DataFrame(records, schema={"arm": pl.Utf8, "group": pl.Utf8, "value": pl.Float64}) if records else pl.DataFrame(schema={"arm": pl.Utf8, "group": pl.Utf8, "value": pl.Float64}),
        x="value", y="group", color="arm", height=260,
        title="Figure 1: Brier score of P(X<0) on the plain test rows (smaller is better)",
        x_title="Mean Brier score",
    )  # fmt: skip


def fig3_harm(
    *, truth: score.Truth, arms: Mapping[str, score.ArmScores], limits: Mapping[str, core.BandLimit]
) -> alt.LayerChart:
    """Figure 3: the mean P(X<0) on import half-hours, pooled and by band, against the three limits."""
    records, rules = [], []
    for name, p in _arms_with(arms=arms, field="p").items():
        for band, reading in score.harm_by_band(rows=truth.rows, p=p, limits=limits).items():
            records.append({"arm": name, "band": band, "value": float(reading["mean_p"])})
    rules = [{"band": b, "limit": float(v.limit)} for b, v in limits.items()]
    data = pl.DataFrame(records, schema={"arm": pl.Utf8, "band": pl.Utf8, "value": pl.Float64})
    dots = dot_chart(
        data=data, x="value", y="band", color="arm", height=140,
        title="Figure 3: mean P(X<0) on import half-hours against each band's limit (smaller is better; ticks: the limits)",
        x_title="Mean P(X<0)",
    )  # fmt: skip
    ticks = (
        alt.Chart(pl.DataFrame(rules))  # ty: ignore[unresolved-attribute]
        .mark_tick(color="#000000", thickness=2, size=30, aria=False)
        .encode(x="limit:Q", y=alt.Y("band:N", sort=list(limits)))
    )
    return dots + ticks


def fig4_reliability(*, truth: score.Truth, arms: Mapping[str, score.ArmScores]) -> alt.LayerChart:
    """Figure 4: the reliability diagram of P(X<0).

    Each bin of the predicted probability shows the share of reverse-flow rows.
    """
    x = truth.rows.x
    records = []
    for name, p in _arms_with(arms=arms, field="p").items():
        which = np.digitize(p, RELIABILITY_EDGES) - 1
        for b in range(len(RELIABILITY_EDGES) - 1):
            m = which == b
            if m.sum() >= 5:
                records.append(
                    {
                        "arm": name,
                        "predicted": float(p[m].mean()),
                        "observed": float((x[m] < 0).mean()),
                    }
                )
    if not records:
        msg = "Figure 4: no arm has P(X<0) in a bin of at least 5 rows"
        raise FiguresError(msg)
    data = pl.DataFrame(records)
    names = sorted(set(data["arm"].to_list()))
    lines = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, aria=False)
        .encode(
            x=alt.X("predicted:Q", title="Mean predicted P(X<0) in the bin", scale=alt.Scale(type="sqrt")),
            y=alt.Y("observed:Q", title="Share of rows with reverse flow", scale=alt.Scale(type="sqrt")),
            color=alt.Color("arm:N", scale=_domain(names=names), legend=alt.Legend(orient="top", title=None)),
        )
    )  # fmt: skip
    diagonal = (
        alt.Chart(pl.DataFrame({"a": [0.0, 1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#000000", strokeDash=[3, 3], aria=False)
        .encode(x="a:Q", y="a:Q")
    )
    return (lines + diagonal).properties(
        width=WIDTH_PX, height=260,
        title=alt.TitleParams("Figure 4: reliability of P(X<0) (the dashed line is perfect calibration)", anchor="start"),
    )  # fmt: skip


def fig5_solar_error(*, truth: score.Truth, arms: Mapping[str, score.ArmScores]) -> alt.Chart:
    """Figure 5: the solar error by world type and source.

    The error is read on sun-up half-hours, and the unreliable half-hours are left out.
    """
    pv = _arms_with(arms=arms, field="pv")
    cosz = np.where(truth.exclude, 0.0, truth.cosz)
    table = score.solar_error_table(rows=truth.rows, cosz=cosz, pv_by_arm=pv)
    records = [
        {"arm": a, "group": g, "value": cell["mae"]}
        for a, groups in table.items()
        for g, cell in groups.items()
    ]
    return dot_chart(
        data=pl.DataFrame(records, schema={"arm": pl.Utf8, "group": pl.Utf8, "value": pl.Float64}),
        x="value", y="group", color="arm", height=240,
        title="Figure 5: mean absolute error of the solar output by world type and source (points; smaller is better)",
        x_title="Mean absolute error (points)",
    )  # fmt: skip


def fig6_recovery(*, truth: score.Truth, arms: Mapping[str, score.ArmScores]) -> alt.VConcatChart:
    """Figure 6: the change of the injected member's solar output against delta.

    The panels split by amount and by import and reverse-flow half-hours.
    """
    rows = truth.rows
    records = []
    for name, a in arms.items():
        if a.pv is None or set(a.pv_inj) != set(core.TEST_INJECTION_POINTS):
            continue
        for amount in core.TEST_INJECTION_POINTS:
            keep = score.injection_keep(
                delta=truth.delta[amount],
                amount=amount,
                cosz=truth.cosz,
                pair=truth.pair[amount].astype(float),
                exclude=truth.exclude,
            )
            change = a.pv_inj[amount] - a.pv
            for flow, flow_mask in (("import", rows.x >= 0.0), ("reverse flow", rows.x < 0.0)):
                m = keep & flow_mask
                if m.sum() < RECOVERY_BINS:
                    continue
                order = np.argsort(truth.delta[amount][m])
                records.extend(
                    {
                        "arm": name,
                        "amount": f"a = {amount:g}",
                        "flow": flow,
                        "delta": float(truth.delta[amount][m][chunk].mean()),
                        "change": float(change[m][chunk].mean()),
                    }
                    for chunk in np.array_split(order, RECOVERY_BINS)
                )
    if not records:
        msg = "Figure 6: no arm has the solar output of the injected member"
        raise FiguresError(msg)
    data = pl.DataFrame(records)
    names = sorted(set(data["arm"].to_list()))
    top = float(max(core.TEST_INJECTION_POINTS))
    exact = (
        alt.Chart(pl.DataFrame({"a": [0.0, top]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#000000", strokeDash=[3, 3], aria=False)
        .encode(x="a:Q", y="a:Q")
    )
    flows = []
    for flow in ("import", "reverse flow"):
        panels = []
        for amount in core.TEST_INJECTION_POINTS:
            part = data.filter((pl.col("flow") == flow) & (pl.col("amount") == f"a = {amount:g}"))
            if part.height == 0:
                continue
            line = (
                alt.Chart(part)  # ty: ignore[unresolved-attribute]
                .mark_line(point=True, aria=False)
                .encode(
                    x=alt.X("delta:Q", title="Injected output delta (points)"),
                    y=alt.Y("change:Q", title="PV(w+) - PV(w) (points)"),
                    color=alt.Color("arm:N", scale=_domain(names=names), legend=alt.Legend(orient="top", title=None)),
                )
            )  # fmt: skip
            panels.append(
                (line + exact).properties(width=200, height=160, title=f"{flow}, a = {amount:g}")
            )
        if panels:
            flows.append(alt.hconcat(*panels))
    return alt.vconcat(*flows).properties(
        title=alt.TitleParams(
            "Figure 6: injection recovery (the dashed line is exact recovery)", anchor="start"
        )
    )


def fig7_solar_checks(*, truth: score.Truth, arms: Mapping[str, score.ArmScores]) -> alt.Chart:
    """Figure 7: the night output, the monotonicity violations, and the response to the sky."""
    records = []
    night = truth.cosz <= core.NIGHT_COSZ
    for name, a in arms.items():
        if a.pv is None:
            continue
        records.append(
            {
                "arm": name,
                "check": "night output, mean (points; smaller is better)",
                "value": float(a.pv[night].mean()) if night.any() else float("nan"),
            }
        )
        if a.pv_raised is not None and truth.raisable.any():
            records.append(
                {
                    "arm": name,
                    "check": "monotonicity violations, share (smaller is better)",
                    "value": float(
                        score.violations(pv=a.pv, pv_raised=a.pv_raised)[truth.raisable].mean()
                    ),
                }
            )
            records.append(
                {
                    "arm": name,
                    "check": "raised-sky response, mean change (points; larger is better)",
                    "value": float((a.pv_raised - a.pv)[truth.raisable].mean()),
                }
            )
    return dot_chart(
        data=pl.DataFrame(records, schema={"arm": pl.Utf8, "check": pl.Utf8, "value": pl.Float64}),
        x="value", y="check", color="arm", height=200,
        title="Figure 7: the solar head's night output, monotonicity, and response to the sky",
        x_title="Value of the check (its direction is in the row label)",
    )  # fmt: skip


def fig8_transfer_steps(
    *, truth: score.Truth, arms: Mapping[str, score.ArmScores]
) -> alt.VConcatChart:
    """Figure 8: step precision and recall against the threshold, and recall by size in D8's noise units."""
    thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    pr, bands = [], []
    for name, step in _arms_with(arms=arms, field="step").items():
        for th in thresholds:
            called = (step > th)[None, :]
            true = (truth.event > 0.5)[None, :]
            if not called.any() or not true.any():
                continue
            call_ok, true_hit = score.match_steps(called=called, true=true)
            pr.append(
                {
                    "arm": name,
                    "threshold": th,
                    "precision": float(call_ok[called].mean()),
                    "recall": float(true_hit[true].mean()),
                }
            )
        table = score.step_table(
            step_p=step,
            size_hat=arms[name].size_hat,
            event=truth.event,
            size=truth.size,
            tr=truth.tr,
        )
        bands += [
            {
                "arm": name,
                "group": f"size {label.replace('_', ' ')} noise units",
                "value": float(cell["recall"]),
            }
            for label, cell in table["size_noise_units"].items()
        ]
    if not pr:
        msg = "Figure 8: no arm has a step output with a call and a true step"
        raise FiguresError(msg)
    data = pl.DataFrame(pr).unpivot(
        index=["arm", "threshold"],
        on=["precision", "recall"],
        variable_name="measure",
        value_name="value",
    )
    names = sorted(set(data["arm"].to_list()))
    curve = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, aria=False)
        .encode(
            x=alt.X("threshold:Q", title="Step probability threshold"),
            y=alt.Y("value:Q", title="Precision or recall (larger is better)"),
            color=alt.Color("arm:N", scale=_domain(names=names), legend=alt.Legend(orient="top", title=None)),
            strokeDash="measure:N",
        )
        .properties(width=WIDTH_PX, height=200, title=alt.TitleParams("Figure 8: precision and recall of the step probability (dashed: recall)", anchor="start"))
    )  # fmt: skip
    recall = dot_chart(
        data=pl.DataFrame(bands, schema={"arm": pl.Utf8, "group": pl.Utf8, "value": pl.Float64}),
        x="value", y="group", color="arm", height=160,
        title="Recall of the step by the size of its transfer (a line at 4 noise units separates the two upper bands)",
        x_title="Recall (larger is better)",
    )  # fmt: skip
    return alt.vconcat(curve, recall)


def fig9_scenarios(*, truth: score.Truth, arms: Mapping[str, score.ArmScores]) -> alt.VConcatChart:
    """Figure 9: the four load-transfer scenarios of table T2, one panel for each score."""
    table = score.scenario_table(truth=truth, arms=arms)
    panels = []
    labels = {
        "brier_sign": "Brier score of P(X<0) (smaller is better)",
        "solar_mae": "Solar mean absolute error (points; smaller is better)",
        "brier_step": "Step Brier score (smaller is better)",
    }
    for column in SCENARIO_COLUMNS:
        records = [
            {"arm": a, "group": s.replace("_", " "), "value": float(cell[column])}
            for a, scenarios in table.items()
            for s, cell in scenarios.items()
            if column in cell
        ]
        if not records:
            continue
        panels.append(
            dot_chart(
                data=pl.DataFrame(records),
                x="value",
                y="group",
                color="arm",
                height=130,
                title=f"Figure 9: {labels[column]}",
                x_title=labels[column],
            )
        )
    if not panels:
        msg = "Figure 9: no arm has an output on the transfer overlay"
        raise FiguresError(msg)
    return alt.vconcat(*panels)


def all_figures(
    *, truth: score.Truth, arms: Mapping[str, score.ArmScores], limits: Mapping[str, core.BandLimit]
) -> dict[str, alt.TopLevelMixin]:
    """Return figures 1, 3, 4, 5, 6, 7, 8, and 9 by file name.

    Args:
        truth: The truth of the test bank.
        arms: The scores of each arm that was fitted.
        limits: The band limits.

    Returns:
        The figures, keyed by file name.

    Raises:
        FiguresError: When no arm has the output a figure needs.
    """
    return {
        "fig01_brier_by_group": fig1_brier_by_group(truth=truth, arms=arms),
        "fig03_harm": fig3_harm(truth=truth, arms=arms, limits=limits),
        "fig04_reliability": fig4_reliability(truth=truth, arms=arms),
        "fig05_solar_error": fig5_solar_error(truth=truth, arms=arms),
        "fig06_recovery": fig6_recovery(truth=truth, arms=arms),
        "fig07_solar_checks": fig7_solar_checks(truth=truth, arms=arms),
        "fig08_transfer_steps": fig8_transfer_steps(truth=truth, arms=arms),
        "fig09_scenarios": fig9_scenarios(truth=truth, arms=arms),
    }

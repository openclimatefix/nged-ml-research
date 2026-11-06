# ruff: noqa: E501, D103, ANN001, ANN201, ANN202
"""Uncluttered time-series panels for the forecast figures of experiments 588 and 597.

Standing rules (maintainer, 2026-11-06): at least 280 px per panel, at most two panels per figure,
one forecast arm (median and q05-q95 band) per panel beside observed power, the gate-on
bar in a thin strip under the panel, and a legend outside the data area. Demand series only.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import polars as pl

W = 620
PANEL_HEIGHT = 280
STRIP_HEIGHT = 14
SKY = "#10C5F7"
SHADE_COLOUR = "#E3D3FA"  # pale lilac: the load-shift
RULE_COLOUR = "#7A1FA2"  # dark purple: the maintainer's switch times
FORECAST_COLOUR = "#FF4901"  # Brand Orange: every forecast band
FORECAST_MEDIAN = "#B33000"


def to_none(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in a]


def _xs(v, xtype: str) -> str | float:
    return str(v) if xtype == "T" else float(v)


def _x_parts(*, xdom, xtype: str, labels: bool) -> tuple[alt.Scale, alt.Axis]:
    if xtype == "T":
        return (
            alt.Scale(domain=[str(v) for v in xdom]),
            alt.Axis(format="%d %b", labels=labels, tickCount=8),
        )
    return (
        alt.Scale(domain=list(xdom), zero=False, nice=False),
        alt.Axis(tickMinStep=1, labels=labels),
    )


def arm_panel(
    *,
    x,
    q,
    iq: dict,
    obs,
    xdom,
    title: str,
    ydom,
    xtype: str = "Q",
    xtitle: str | None = None,
    strip: list[dict] | None = None,
    span=None,
    rules=(),
    label_nra=None,
):
    """One panel unit: one arm's median and q05-q95 band, observed power, and a gate strip.

    `strip` is a list of `{"x0", "x1"}` origin-day spans with the gate on or held; None means the
    arm has no gate and the strip stays empty. `span` is the (start, end) of the shaded load-shift.
    `rules` are x positions of dashed vertical lines. `xtype` is "Q" (days) or "T" (datetimes).
    """
    cols = {
        "x": [_xs(a, xtype) for a in x],
        "lo": to_none(np.asarray(q[:, iq[0.05]] * 100, float)),
        "hi": to_none(np.asarray(q[:, iq[0.95]] * 100, float)),
        "med": to_none(np.asarray(q[:, iq[0.5]] * 100, float)),
        "obs": to_none(np.asarray(obs * 100, float)),
    }
    if label_nra is not None:
        cols["lab"] = to_none(np.asarray(label_nra * 100, float))
    df = pl.DataFrame(cols)
    # Keep only rows inside the plotted window: marks outside it widen the exported SVG.
    inside = [xdom[0] <= v <= xdom[1] for v in x]
    df = df.filter(pl.Series(inside))
    sy = alt.Scale(domain=[float(ydom[0]), float(ydom[1])], nice=False)
    scale, top_axis = _x_parts(xdom=xdom, xtype=xtype, labels=False)
    _, bot_axis = _x_parts(xdom=xdom, xtype=xtype, labels=True)
    top_x = alt.X(f"x:{xtype}", scale=scale, axis=top_axis, title=None)
    if span is not None:
        span = (max(span[0], xdom[0]), min(span[1], xdom[1]))
        if span[0] >= span[1]:
            span = None
    shade = (
        pl.DataFrame({"a": [_xs(span[0], xtype)], "b": [_xs(span[1], xtype)]})
        if span is not None
        else None
    )
    rules = [v for v in rules if xdom[0] <= v <= xdom[1]]
    rule_df = pl.DataFrame({"x": [_xs(v, xtype) for v in rules]}) if rules else None

    def shade_layer(axis):
        return (
            alt.Chart(shade)  # ty: ignore[unresolved-attribute]
            .mark_rect(color=SHADE_COLOUR, opacity=0.7, clip=True, aria=False)
            .encode(x=alt.X(f"a:{xtype}", scale=scale, axis=axis), x2="b")
        )

    def rule_layer(axis):
        return (
            alt.Chart(rule_df)  # ty: ignore[unresolved-attribute]
            .mark_rule(color=RULE_COLOUR, strokeDash=[6, 3], strokeWidth=1.3, aria=False, clip=True)
            .encode(x=alt.X(f"x:{xtype}", scale=scale, axis=axis))
        )

    base = alt.Chart(df)
    layers = []
    if shade is not None:
        layers.append(shade_layer(top_axis))
    layers += [
        base.mark_area(color=FORECAST_COLOUR, opacity=0.30, aria=False, clip=True).encode(x=top_x, y=alt.Y("lo:Q", scale=sy, title="% of capacity"), y2="hi:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_line(color=FORECAST_MEDIAN, strokeWidth=1.6, aria=False, clip=True).encode(x=top_x, y=alt.Y("med:Q", scale=sy)),  # ty: ignore[unresolved-attribute]
        base.mark_line(color="black", strokeWidth=1.0, aria=False, clip=True).encode(x=top_x, y=alt.Y("obs:Q", scale=sy)),  # ty: ignore[unresolved-attribute]
    ]  # fmt: skip
    if label_nra is not None:
        layers.append(
            base.mark_line(
                color="#306BFF", strokeDash=[5, 3], strokeWidth=1.3, aria=False, clip=True
            ).encode(x=top_x, y=alt.Y("lab:Q", scale=sy))  # ty: ignore[unresolved-attribute]
        )
    if rule_df is not None:
        layers.append(rule_layer(top_axis))
    main = alt.layer(*layers).properties(
        width=W, height=PANEL_HEIGHT, title=alt.TitleParams(title, fontSize=12, anchor="start")
    )

    strip_layers = []
    if shade is not None:
        strip_layers.append(shade_layer(bot_axis))
    if strip:
        sdf = pl.DataFrame(
            {"x0": [_xs(r["x0"], xtype) for r in strip], "x1": [_xs(r["x1"], xtype) for r in strip]}
        )
        strip_layers.append(
            alt.Chart(sdf)  # ty: ignore[unresolved-attribute]
            .mark_rect(color=SKY, opacity=0.95, clip=True, aria=False)
            .encode(x=alt.X(f"x0:{xtype}", scale=scale, axis=bot_axis, title=xtitle), x2="x1")
        )
    else:  # invisible mark that only carries the x axis
        strip_layers.append(
            alt.Chart(df.head(1))  # ty: ignore[unresolved-attribute]
            .mark_point(opacity=0, aria=False)
            .encode(x=alt.X(f"x:{xtype}", scale=scale, axis=bot_axis, title=xtitle))
        )
    if rule_df is not None:
        strip_layers.append(rule_layer(bot_axis))
    strip_chart = alt.layer(*strip_layers).properties(width=W, height=STRIP_HEIGHT)
    return alt.vconcat(main, strip_chart, spacing=2)


def entry(label: str, kind: str, colour: str, alpha: float = 1.0) -> dict:
    return {"label": label, "kind": kind, "colour": colour, "alpha": alpha}


FORECAST_ENTRIES = [
    entry("q05-q95 band", "band", FORECAST_COLOUR, 0.30),
    entry("Median", "line", FORECAST_MEDIAN),
]
OBS = entry("Observed power", "line", "black")
SHADE = entry("Load-shift", "band", SHADE_COLOUR, 0.7)
SHADE_LABELLED = entry("Labelled load-shift", "band", SHADE_COLOUR, 0.7)
GATE = entry("Gate on, or held for 7 days", "band", SKY, 0.95)
RULES = entry("Switch times given by the maintainer", "rule", RULE_COLOUR)
LABEL_NRA = entry("NRA estimate if label is right", "dashed", "#306BFF")


def legend(entries: list[dict]):
    """A key drawn under the panels, outside the data area."""
    cols = 3
    col_w = W / cols
    rows = []
    for i, e in enumerate(entries):
        r, c = divmod(i, cols)
        rows.append(
            {**e, "x": c * col_w, "y": float(-r), "x2": c * col_w + 22, "tx": c * col_w + 28}
        )
    df = pl.DataFrame(rows)
    n_rows = (len(entries) + cols - 1) // cols
    ys = alt.Scale(domain=[-(n_rows - 0.5), 0.5])
    xs = alt.Scale(domain=[0, W])
    layers = []
    rect = df.filter(pl.col("kind") == "band")
    if rect.height:
        layers.append(
            alt.Chart(rect).mark_rect(height=11, aria=False).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("x:Q", scale=xs, axis=None), x2="x2:Q", y=alt.Y("y:Q", scale=ys, axis=None),
                color=alt.Color("colour:N", scale=None, legend=None), opacity=alt.Opacity("alpha:Q", scale=None, legend=None),
            )
        )  # fmt: skip
    for kind, dash, width in (
        ("line", [1, 0], 2.0),
        ("dashed", [4, 3], 1.3),
        ("rule", [6, 3], 1.3),
    ):
        sub = df.filter(pl.col("kind") == kind)
        if sub.height:
            layers.append(
                alt.Chart(sub).mark_rule(strokeDash=dash, strokeWidth=width, aria=False).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X("x:Q", scale=xs, axis=None), x2="x2:Q", y=alt.Y("y:Q", scale=ys, axis=None),
                    color=alt.Color("colour:N", scale=None, legend=None),
                )
            )  # fmt: skip
    layers.append(
        alt.Chart(df)
        .mark_text(align="left", baseline="middle", fontSize=11, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("tx:Q", scale=xs, axis=None),
            y=alt.Y("y:Q", scale=ys, axis=None),
            text="label:N",
        )
    )
    return alt.layer(*layers).properties(width=W, height=18 * n_rows + 4)


def figure(*, units: list, title: list[str] | str, key: list[dict]):
    """Stack at most two panel units above the key."""
    assert len(units) <= 2, "never more than two panels in one figure"
    return (
        alt.vconcat(*units, legend(key), spacing=14)
        .properties(title=alt.TitleParams(title, fontSize=12, anchor="start"))
        .resolve_scale(x="independent", y="independent")
        .configure_view(stroke=None)
    )


def common_ydom(pairs, *, iq: dict) -> tuple[float, float]:
    """Shared y range (% of capacity) over the bands and observed power of several panels."""
    lo, hi = np.inf, -np.inf
    for q, obs in pairs:
        lo = min(lo, float(np.nanmin(q[:, iq[0.05]])), float(np.nanmin(obs)))
        hi = max(hi, float(np.nanmax(q[:, iq[0.95]])), float(np.nanmax(obs)))
    return lo * 100 - 2, hi * 100 + 2

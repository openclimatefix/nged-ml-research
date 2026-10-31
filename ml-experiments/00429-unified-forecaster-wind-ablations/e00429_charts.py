# ruff: noqa: E501
"""Draw experiment 00429's figures as SVG. No calendar date or capacity in megawatts appears.

    source ml-experiments/00429-*/env.sh
    uv run --no-sync python ml-experiments/00429-*/e00429_charts.py

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`.
"""

import os
import re
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import altair as alt  # noqa: E402

import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00429_data as data  # noqa: E402
import e00429_score as sc  # noqa: E402
import numpy as np  # noqa: E402
import plotting.ocf_theme as ocf  # noqa: E402
import polars as pl  # noqa: E402

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / ("00429-unified-forecaster-wind-ablations")
)
WIDTH: Final[int] = 440
GREY: Final[str] = "#999999"
GROUPS: Final[dict[str, tuple[str, str]]] = {
    "g0": ("Fixed plant summary (G0)", ocf.GREY_2),
    "one_pass_ms": ("One pass, mean and spread", ocf.DATA_SKY),
    "one_pass_m": ("One pass, mean only", ocf.DATA_SKY),
    "member_plain": ("Member route, plain", ocf.DATA_BLUE),
    "member_frozen": ("Member route, plant", ocf.BRAND_ORANGE),
    "member_e2e": ("Member route, plant", ocf.BRAND_ORANGE),
    "memberwise": ("Member-wise quantiles", ocf.DATA_PURPLE),
    "memberwise_frozen": ("Member-wise quantiles", ocf.DATA_PURPLE),
    "member_plain_allrows": ("Trained on all rows", ocf.DATA_GREEN),
    "member_frozen_allrows": ("Trained on all rows", ocf.DATA_GREEN),
    "member_plain_censored": ("Censored head", "#555555"),
}
LABELS: Final[dict[str, str]] = {
    "g0": "G0 (rebuild of 347's y2)",
    "one_pass_ms": "one pass, mean and spread",
    "one_pass_m": "one pass, mean only",
    "member_plain": "member route, plain",
    "member_frozen": "member route, frozen plant",
    "member_e2e": "member route, trained plant",
    "memberwise": "member-wise, plain",
    "memberwise_frozen": "member-wise, frozen plant",
    "member_plain_allrows": "member route, plain, all rows",
    "member_frozen_allrows": "member route, frozen plant, all rows",
    "member_plain_censored": "member route, plain, censored head",
}


def colour_scale() -> alt.Scale:
    """Return the colour scale of the arm groups."""
    seen: dict[str, str] = {}
    for name, (label, colour) in GROUPS.items():
        if name != "g0":
            seen.setdefault(label, colour)
    return alt.Scale(domain=list(seen), range=list(seen.values()))


def wrap(text: str, width: int = 95) -> list[str]:
    """Wrap a title line to the plot width."""
    import textwrap

    return textwrap.wrap(text, width)


def load() -> tuple[pl.DataFrame, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the scored cases, actual, per-arm quantiles, and per-arm row scores."""
    cases = data347.scored_frame().with_columns(lead_day=pl.col("lead_day").clip(0, 10))
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    arms = sc.load_arms()
    qs = {
        n: score303.row_metrics(quantiles=q, actual=actual)["qs"].mean(axis=0, keepdims=True)
        for n, q in arms.items()
    }
    return cases, actual, arms, qs


def fig_headline(*, cases: pl.DataFrame, qs: dict[str, np.ndarray]) -> alt.TopLevelMixin:
    """Each arm minus experiment 347's y2 at true days 1 to 10, kept rows."""
    keep = (~cases["masked"] & cases["lead_hours"].is_between(24, 240)).to_numpy()
    resampler = sc.Intervals(month=cases["month"].to_numpy(), seeds=1)
    rows = []
    for name, (group, _) in GROUPS.items():
        if name == "g0" or name not in qs:
            continue
        entry = resampler.read(values=qs[name] - qs["ref_347_y2"], keep=keep)
        low95, high95 = entry["0.9500"]
        low, high = entry[f"{sc.LEVEL:.4f}"]
        rows.append(
            {
                "arm": LABELS[name],
                "group": group,
                "mean": entry["mean"],
                "low95": low95,
                "high95": high95,
                "low": low,
                "high": high,
            }
        )
    frame = pl.DataFrame(rows)
    order = [LABELS[n] for n in GROUPS if n != "g0"]
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=240))
    )
    wide = base.mark_rule(strokeWidth=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q",
            title="Quantile score minus experiment 347's y2 (points of capacity; negative is better)",
        ),
        x2="high:Q",
        color=alt.Color(
            "group:N",
            scale=colour_scale(),
            legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=400),
        ),
    )
    narrow = base.mark_rule(strokeWidth=4, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x="low95:Q", x2="high95:Q", color=alt.Color("group:N", scale=colour_scale(), legend=None)
    )
    dot = base.mark_point(filled=True, size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x="mean:Q", color=alt.Color("group:N", scale=colour_scale(), legend=None)
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black", aria=False).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    margin = (
        alt.Chart(pl.DataFrame({"x": [-sc.MARGIN, sc.MARGIN]}))
        .mark_rule(color=GREY, strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(margin, zero, wide, narrow, dot).properties(
        width=WIDTH,
        height=26 * len(rows),
        title=alt.TitleParams(
            text=[
                "No design beats experiment 347's plant-plus-head forecast by the 0.10-point margin"
            ],
            subtitle=wrap(
                "Dot: estimate. Thick line: 95% interval. Thin line: 99.8% interval, both from resampling whole months. Black rule: same as 347's y2 (Wind farms research default 2). Dashed: the 0.10-point margin. Kept rows, true days 1 to 10, wind farms W1 to W3, 53 Tuesday runs, 5 seeds."
            ),
            anchor="start",
        ),
    )


def fig_tails(
    *, cases: pl.DataFrame, arms: dict[str, np.ndarray], actual: np.ndarray
) -> alt.TopLevelMixin:
    """Share of observations above q99 and below q01 at true days 1 to 10."""
    keep = (~cases["masked"] & cases["lead_hours"].is_between(24, 240)).to_numpy()
    rows = []
    for name in (*GROUPS, "ref_default", "ref_clim"):
        if name not in arms:
            continue
        q = arms[name]
        above = float(np.mean([(actual[keep] > s[keep, 10]).mean() for s in q]))
        below = float(np.mean([(actual[keep] < s[keep, 0]).mean() for s in q]))
        label = LABELS.get(
            name,
            {"ref_default": "research default 1 (XGBoost)", "ref_clim": "climatology"}.get(
                name, name
            ),
        )
        rows.append({"arm": label, "side": "above q99", "share": above * 100})
        rows.append({"arm": label, "side": "below q01", "share": below * 100})
    frame = pl.DataFrame(rows)
    order = list(dict.fromkeys(frame["arm"].to_list()))
    chart = (
        alt.Chart(frame)
        .mark_point(filled=True, size=60, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("share:Q", title="Share of observations beyond the quantile (%; ideal 1%)"),
            color=alt.Color(
                "side:N",
                scale=alt.Scale(
                    domain=["above q99", "below q01"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [1.0, 1.5]}))
        .mark_rule(color=GREY, strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (chart + rule).properties(
        width=WIDTH,
        height=20 * len(order),
        title=alt.TitleParams(
            text=[
                "Every neural arm leaves 2.2% to 2.5% of observations above q99; the target is 1%"
            ],
            subtitle=wrap(
                "Dashed rules: the ideal 1% and the 1.5% upper-tail limit. Kept rows, true days 1 to 10, wind farms W1 to W3, 53 Tuesday runs; mean over 5 seeds."
            ),
            anchor="start",
        ),
    )


def fig_lead(*, cases: pl.DataFrame, qs: dict[str, np.ndarray]) -> alt.TopLevelMixin:
    """Quantile score by calendar lead day, kept rows."""
    days = (cases["lead_hours"].to_numpy() // 24).astype(int)
    kept = ~cases["masked"].to_numpy()
    chosen = {
        "one_pass_ms": "one pass, mean and spread",
        "member_plain": "member route, plain",
        "memberwise": "member-wise, plain",
        "ref_347_y2": "347's y2 (frozen plant + head)",
        "ref_default": "research default 1 (XGBoost)",
        "ref_clim": "climatology",
    }
    colours = [ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, GREY]
    rows = [
        {"day": d, "arm": label, "score": float(qs[n][0, kept & (days == d)].mean())}
        for n, label in chosen.items()
        for d in range(15)
        if (kept & (days == d)).any()
    ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:Q",
                title="Lead (days from the start of the run; day 0 is the first 24 hours)",
                axis=alt.Axis(values=list(range(0, 15, 2))),
            ),
            y=alt.Y("score:Q", title="Quantile score (points; smaller is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(chosen.values()), range=colours),
                legend=alt.Legend(orient="top", title=None, columns=2),
            ),
        )
        .properties(
            width=WIDTH,
            height=240,
            title=alt.TitleParams(
                text=[
                    "Forecast skill falls with lead for every design, and the neural designs track each other"
                ],
                subtitle=wrap(
                    "Kept rows, wind farms W1 to W3, 53 Tuesday runs, mean over 5 seeds."
                ),
                anchor="start",
            ),
        )
    )


def fig_coherence() -> alt.TopLevelMixin:
    """Variogram and energy scores of the day blocks, from the stored coherence report."""
    text = (data.DATA_DIR / "coherence_report.txt").read_text()
    day = text.split("[day blocks")[1].split("[run blocks")[0]
    scores = {}
    for line in day.splitlines():
        m = re.match(
            r"(\w+)\s+(memberwise_median|memberwise|ecc|schaake|shifted|independent)\s*energy ([\d.]+)\s+variogram ([\d.]+)",
            line,
        )
        if m:
            scores[m.group(1), m.group(2)] = (float(m.group(3)), float(m.group(4)))
    chosen = [
        (("one_pass_ms", "independent"), "one pass, independent draws (no coherence)", GREY),
        (("one_pass_ms", "ecc"), "one pass, ECC", ocf.DATA_SKY),
        (("one_pass_ms", "schaake"), "one pass, Schaake shuffle", ocf.DATA_GREEN),
        (("member_plain", "shifted"), "member route, members' own paths", ocf.DATA_BLUE),
        (("memberwise", "memberwise"), "member-wise quantiles at ECC ranks", ocf.DATA_PURPLE),
    ]
    rows = []
    reference = scores["one_pass_ms", "ecc"]
    for key, label, _ in chosen:
        for k, metric in enumerate(("Energy score", "Variogram score")):
            rows.append({"path": label, "metric": metric, "score": scores[key][k] - reference[k]})
    frame = pl.DataFrame(rows)
    order = [label for _, label, _ in chosen]
    return (
        alt.Chart(frame)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("path:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X(
                "score:Q",
                title="Score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "path:N",
                scale=alt.Scale(domain=order, range=[c for _, _, c in chosen]),
                legend=None,
            ),
            row=alt.Row("metric:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .resolve_scale(x="independent")
        .properties(
            width=WIDTH - 40,
            height=90,
            title=alt.TitleParams(
                text=["The Schaake shuffle gives the most coherent paths across the three farms"],
                subtitle=wrap(
                    "51 member paths per farm, scored over one run and one calendar day jointly across W1 to W3 (529 blocks, 53 Tuesday runs, kept half-hours), mean over 5 seeds. Bars show the difference from the one-pass ECC paths."
                ),
                anchor="start",
            ),
        )
    )


def fig_hp(
    *,
    cases: pl.DataFrame,
    actual: np.ndarray,
    arms: dict[str, np.ndarray],
    qs: dict[str, np.ndarray],
) -> alt.TopLevelMixin:
    """Plant minus plain for the 2 x 2 of training rows, on four row sets."""
    resampler = sc.Intervals(month=cases["month"].to_numpy(), seeds=1)
    masked = cases["masked"].to_numpy()
    clean = np.where(masked, 0.0, actual)
    clean_qs = {
        n: score303.row_metrics(quantiles=arms[n], actual=clean)["qs"].mean(axis=0, keepdims=True)
        for n in ("member_plain", "member_frozen", "member_plain_allrows", "member_frozen_allrows")
    }
    sets = {
        "Kept rows, true days 1 to 10": (
            (~masked) & cases["lead_hours"].is_between(24, 240).to_numpy(),
            qs,
        ),
        "Idle-band rows only": (masked, qs),
        "All rows, raw readings": (cases["lead_hours"].is_between(24, 240).to_numpy(), qs),
        "All rows, idle band set to zero": (
            cases["lead_hours"].is_between(24, 240).to_numpy(),
            clean_qs,
        ),
    }
    rows = []
    for training, (plant, plain) in {
        "Trained without the idle band": ("member_frozen", "member_plain"),
        "Trained on all rows": ("member_frozen_allrows", "member_plain_allrows"),
    }.items():
        for label, (keep, source) in sets.items():
            entry = resampler.read(values=source[plant] - source[plain], keep=keep)
            low, high = entry["0.9500"]
            rows.append(
                {
                    "training": training,
                    "rows": label,
                    "mean": entry["mean"],
                    "low": low,
                    "high": high,
                }
            )
    frame = pl.DataFrame(rows)
    order = list(sets)
    base = alt.Chart(frame).encode(
        y=alt.Y("rows:N", sort=order, title=None, axis=alt.Axis(labelLimit=210)),
        color=alt.Color(
            "training:N",
            scale=alt.Scale(
                domain=["Trained without the idle band", "Trained on all rows"],
                range=[ocf.DATA_BLUE, ocf.DATA_GREEN],
            ),
            legend=alt.Legend(orient="top", title=None, labelLimit=400),
        ),
        yOffset="training:N",
    )
    chart = base.mark_rule(strokeWidth=3, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q", title="Frozen-plant arm minus plain arm (points; negative favours the plant)"
        ),
        x2="high:Q",
    ) + base.mark_point(filled=True, size=70, aria=False).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black", aria=False).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (zero + chart).properties(
        width=WIDTH,
        height=170,
        title=alt.TitleParams(
            text=["A physics power curve gives no gain over a plain network, trained either way"],
            subtitle=wrap(
                "Dot: estimate. Line: 95% interval from resampling whole months. Black rule: the two arms tie. Wind farms W1 to W3, 53 Tuesday runs, 5 seeds. The idle band is experiment 210's mask."
            ),
            anchor="start",
        ),
    )


def fig_example(
    *, cases: pl.DataFrame, actual: np.ndarray, arms: dict[str, np.ndarray], window: str
) -> alt.TopLevelMixin:
    """One farm, one winter and one summer run: median, 50% and 90% bands against the observation."""
    q = arms["member_plain"].mean(axis=0)
    months = cases["month"].to_numpy()
    site = cases["site"].to_numpy()
    lead = cases["lead_hours"].to_numpy()
    init = cases["power_fcst_init_time"].to_numpy()
    panels = []
    for season, month_pick in (
        ("winter run", ("2025-12", "2026-01", "2026-02")),
        ("summer run", ("2025-07", "2025-08", "2026-06")),
    ):
        sel = np.isin(months, month_pick) & (site == "W1")
        run_time = np.unique(init[sel])[2]
        rows = sel & (init == run_time) & ~cases["masked"].to_numpy()
        if window == "day":
            rows &= (lead >= 12) & (lead < 36)
        order = np.argsort(lead[rows])
        frame = pl.DataFrame(
            {
                "lead": (lead[rows][order] / 24.0).tolist()
                if window == "14"
                else lead[rows][order].tolist(),
                "median": (q[rows, 5][order] * 100).tolist(),
                "q05": (q[rows, 1][order] * 100).tolist(),
                "q95": (q[rows, 9][order] * 100).tolist(),
                "q25": (q[rows, 3][order] * 100).tolist(),
                "q75": (q[rows, 7][order] * 100).tolist(),
                "actual": (actual[rows][order] * 100).tolist(),
            }
        )
        title = (
            "Lead (hours after the run starts)"
            if window == "day"
            else "Lead (days after the run starts)"
        )
        x = alt.X("lead:Q", title=title)
        base = alt.Chart(frame).encode(x=x)
        band90 = base.mark_area(opacity=0.25, color=ocf.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Output (% of capacity)"), y2="q95:Q"
        )
        band50 = base.mark_area(opacity=0.45, color=ocf.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y="q25:Q", y2="q75:Q"
        )
        med = base.mark_line(color=ocf.DATA_BLUE, aria=False).encode(y="median:Q")  # ty: ignore[unresolved-attribute]
        obs = base.mark_line(color=ocf.BRAND_ORANGE, aria=False).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            (band90 + band50 + med + obs).properties(
                width=WIDTH,
                height=150,
                title=alt.TitleParams(text=f"Farm W1, {season}", anchor="start", fontSize=12),
            )
        )
    heading = (
        "Day-ahead forecast (leads 12 to 36 hours)"
        if window == "day"
        else "Full 14-day horizon of one run"
    )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            text=[heading + ": the bands widen with lead"] if window == "14" else [heading],
            subtitle=wrap(
                "Orange: observed output. Blue line: median forecast. Dark band: 50% interval. Light band: 90% interval. Member route, plain; mean of 5 seeds' quantiles. One Tuesday run per season; idle-band half-hours left out."
            ),
            anchor="start",
        )
    )


def main() -> None:
    """Write every figure."""
    DOCS.mkdir(parents=True, exist_ok=True)
    cases, actual, arms, qs = load()
    figures = {
        "headline": fig_headline(cases=cases, qs=qs),
        "upper_tail": fig_tails(cases=cases, arms=arms, actual=actual),
        "by_lead": fig_lead(cases=cases, qs=qs),
        "coherence": fig_coherence(),
        "physics_layer": fig_hp(cases=cases, actual=actual, arms=arms, qs=qs),
        "example_day_ahead": fig_example(cases=cases, actual=actual, arms=arms, window="day"),
        "example_14_day": fig_example(cases=cases, actual=actual, arms=arms, window="14"),
    }
    for name, chart in figures.items():
        chart.save(DOCS / f"{name}.svg")
        print(name, flush=True)


if __name__ == "__main__":
    main()

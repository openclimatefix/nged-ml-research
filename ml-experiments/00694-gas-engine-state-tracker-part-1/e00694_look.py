# ruff: noqa: E501, PLR0912, PLR0915
r"""Phase A: the data look. Draws D8's reading and its differences in four eras; fits no model.

    OMP_NUM_THREADS=1 PYTHONPATH=src:ml-experiments/00694-gas-engine-state-tracker-part-1 \
        uv run --no-sync python ml-experiments/00694-*/e00694_look.py

Gas A is drawn only as a histogram of output levels, with no dates. D8 is a substation.
"""

import os
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00694_common import (
    DATA_DIR,
    DOCS_DIR,
    ENGINE_CONNECTED,
    EXPECTED_UNIT_COUNTS,
    FADE_START,
    GRID_START,
    NET_END,
    NET_START,
    SOLAR_START,
    causal_spread,
    load_panel,
    panel_power,
    quantise_spread,
    record_arrays,
    runs_of,
    stamp,
)
from e00694_trackers import U_GRID  # noqa: F401  (kept so a changed grid shows in the printout)

WIDTH: Final[int] = 560
ERAS: Final[dict[str, tuple[datetime, datetime]]] = {
    "pre-engine": (GRID_START, ENGINE_CONNECTED),
    "net": (NET_START, NET_END),
    "net with 11 kV solar": (SOLAR_START, FADE_START),
    "from 2026-03": (FADE_START, datetime(2026, 7, 1, tzinfo=UTC)),
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_PURPLE]
LINES: list[str] = []


def say(text: str) -> None:
    """Print a line and keep it for printed_look.txt."""
    print(text)
    LINES.append(text)


def wrap(text: str, width: int = 95) -> list[str]:
    """Wrap a title or subtitle to the plot width."""
    return textwrap.wrap(text, width)


def merged_steps(*, y: np.ndarray, lvl: np.ndarray, min_size: float = 0.6) -> dict[str, np.ndarray]:
    """Merge consecutive same-sign differences above 4 spreads; keep merged steps above min_size."""
    from e00694_common import SPREAD_LEVELS

    spread = SPREAD_LEVELS[lvl]
    d = np.zeros(len(y))
    d[1:] = np.nan_to_num(np.diff(y))
    big = np.abs(d) > 4.0 * spread
    first, last, size, before = [], [], [], []
    t = 1
    while t < len(y):
        if not big[t]:
            t += 1
            continue
        end = t
        while end + 1 < len(y) and big[end + 1] and np.sign(d[end + 1]) == np.sign(d[t]):
            end += 1
        total = float(d[t : end + 1].sum())
        if abs(total) >= min_size:
            first.append(t)
            last.append(end)
            size.append(total)
            before.append(float(y[t - 1]))
        t = end + 1
    return {
        "first": np.array(first),
        "last": np.array(last),
        "size": np.array(size),
        "before": np.array(before),
    }


def save(*, chart: alt.Chart | alt.VConcatChart, name: str) -> None:
    """Write an SVG beside the write-up."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    if png_dir := os.environ.get("PNG_DIR"):
        chart.save(Path(png_dir) / f"{name}.png", scale_factor=1.0)


def main() -> None:
    """Draw figures 1 to 5 and print the Phase A numbers."""
    panel = load_panel()
    y_all = panel_power(panel=panel, label="D8")
    spread = causal_spread(y=y_all)
    lvl_all = quantise_spread(spread=spread)
    base = np.datetime64(GRID_START.replace(tzinfo=None), "s")

    # ---- reproduction of experiment 194's counts (the gate for the experiment's validity)
    g, units = record_arrays()
    counts = tuple(int(c) for c in np.bincount(units, minlength=4))
    say(f"Gas A: rows {len(units)}, half-hours at 0/1/2/3 units {counts}")
    say(f"Experiment 194 counts {EXPECTED_UNIT_COUNTS}: match = {counts == EXPECTED_UNIT_COUNTS}")
    spells = [d for k, d in runs_of(units=(units > 0).astype(int)) if k > 0]
    say(
        f"Gas A running spells (runs with at least 1 unit): {len(spells)}, median {np.median(spells):.0f} half-hours"
    )
    say(
        f"Gas A output per unit: median g/units {np.median(g[units > 0] / units[units > 0]):.3f} MW"
    )
    if counts != EXPECTED_UNIT_COUNTS or len(units) != 2054:
        raise SystemExit("Phase A cannot reproduce experiment 194's counts: stop (verdict invalid)")

    era_data: dict[str, dict[str, Any]] = {}
    for name, (lo, hi) in ERAS.items():
        a, b = stamp(when=lo), min(stamp(when=hi), len(y_all))
        yy, lv = y_all[a:b], lvl_all[a:b]
        d1 = np.diff(yy)
        d2 = yy[2:] - yy[:-2]
        steps = merged_steps(y=yy, lvl=lv)
        hours = (
            (base + (a + steps["first"]) * np.timedelta64(1800, "s")).astype("datetime64[h]")
            - (base + (a + steps["first"]) * np.timedelta64(1800, "s")).astype("datetime64[D]")
        ).astype(int)
        era_data[name] = {
            "a": a,
            "b": b,
            "y": yy,
            "d1": d1,
            "d2": d2,
            "steps": steps,
            "hours": hours,
        }
        say(
            f"{name}: {b - a} half-hours, nan share {np.isnan(yy).mean():.3f}, "
            f"median {np.nanmedian(yy):.2f} MVA, merged steps {len(steps['size'])}"
        )
        sz = steps["size"]
        hist, edges = np.histogram(sz, bins=np.arange(-8, 8.01, 0.5))
        peaks = [
            f"{edges[i]:.1f}..{edges[i + 1]:.1f}: {hist[i]}"
            for i in range(1, len(hist) - 1)
            if hist[i] >= 8 and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]
        ]
        say(f"  merged-step bins with local maxima (count >= 8): {peaks}")
        for lo_l, hi_l in ((0, 3), (3, 6), (6, 8), (8, 10)):
            m = (steps["before"] >= lo_l) & (steps["before"] < hi_l)
            r_n = int(((sz >= 1.5) & m).sum())
            f_n = int(((sz <= -1.5) & m).sum())
            say(
                f"  steps of at least 1.5 MVA from a level in [{lo_l}, {hi_l}) MVA: {r_n} rises, {f_n} falls"
            )

    # ---- Figure 1: 14-day panels, one per era
    panels = []
    for i, (name, ed) in enumerate(era_data.items()):
        start = 28 * 48
        seg = ed["y"][start : start + 14 * 48]
        when = (base + (ed["a"] + start) * np.timedelta64(1800, "s")).astype("datetime64[D]")
        frame = pl.DataFrame({"day": np.arange(len(seg)) / 48.0, "mva": seg})
        panels.append(
            alt.Chart(frame)  # ty: ignore[unresolved-attribute]
            .mark_line(color=COLOURS[i], strokeWidth=1, clip=True)
            .encode(
                x=alt.X(
                    "day:Q",
                    title="Days from the start of the 14-day window",
                    axis=alt.Axis(values=list(range(0, 15, 2))),
                ),
                y=alt.Y("mva:Q", title="MVA", scale=alt.Scale(domain=[0, 14])),
            )
            .properties(width=WIDTH, height=110, title=f"{name}: 14 days from {when}")
        )
    fig1 = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            wrap(
                "D8's half-hourly reading in four eras, 14 days each, on one axis (apparent power in MVA)"
            ),
            anchor="start",
        )
    )
    save(chart=fig1, name="figure_1_d8_reading_by_era")

    # ---- Figure 2: histograms of lag-1 and lag-2 differences
    rows = []
    for name, ed in era_data.items():
        for lag, d in (("lag 1", ed["d1"]), ("lag 2", ed["d2"])):
            h, e = np.histogram(d[np.isfinite(d)], bins=np.arange(-8, 8.01, 0.25))
            share = h / max(h.sum(), 1)
            for c, cnt in zip(e[:-1], share, strict=True):
                if cnt > 0:
                    rows.append(
                        {"era": name, "lag": lag, "step": float(c + 0.125), "share": float(cnt)}
                    )
    frame = pl.DataFrame(rows)
    fig2 = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(interpolate="step-after", strokeWidth=1.5)
        .encode(
            x=alt.X(
                "step:Q",
                title="Change in the reading over the lag (MVA)",
                scale=alt.Scale(domain=[-8, 8]),
            ),
            y=alt.Y(
                "share:Q", title="Share of half-hours (log scale)", scale=alt.Scale(type="log")
            ),
            color=alt.Color(
                "era:N",
                scale=alt.Scale(domain=list(ERAS), range=COLOURS),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH, height=150)
        .facet(row=alt.Row("lag:N", title=None))
        .properties(
            title=alt.TitleParams(
                wrap(
                    "Change in D8's reading over 1 and 2 half-hours, by era. A peak near 4 to 5 MVA appears only after the engine is connected."
                ),
                anchor="start",
            )
        )
    )
    save(chart=fig2, name="figure_2_difference_histograms")

    # ---- Figure 3: merged fall size against the level before the fall
    rows = []
    for name, ed in era_data.items():
        st = ed["steps"]
        m = st["size"] <= -0.6
        lev, fall = st["before"][m], -st["size"][m]
        for lo_l in np.arange(0, 14, 1.0):
            sel = (lev >= lo_l) & (lev < lo_l + 1.0)
            if sel.sum() >= 5:
                rows.append(
                    {
                        "era": name,
                        "level": float(lo_l + 0.5),
                        "fall": float(np.median(fall[sel])),
                        "n": int(sel.sum()),
                    }
                )
    frame = pl.DataFrame(rows)
    theory = pl.DataFrame(
        [
            {
                "curve": f"{k} unit{'s' if k > 1 else ''} of 2 MW",
                "level": float(L),
                "fall": float(L - abs(L - 2.0 * k)),
            }
            for k in (1, 2, 3)
            for L in np.arange(0.5, 13.5, 0.5)
        ]
    )
    pts = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X("level:Q", title="Reading before the fall (MVA, 1 MVA bins)"),
            y=alt.Y(
                "fall:Q", title="Median size of the fall (MVA)", scale=alt.Scale(domain=[0, 7])
            ),
            color=alt.Color(
                "era:N",
                scale=alt.Scale(domain=list(ERAS), range=COLOURS),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    curves = (
        alt.Chart(theory)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeDash=[4, 3], strokeWidth=1.5, color=ocf.BLACK_2)
        .encode(x="level:Q", y="fall:Q", detail="curve:N")
    )
    fig3 = (pts + curves).properties(
        width=WIDTH,
        height=220,
        title=alt.TitleParams(
            wrap(
                "Size of D8's merged falls against the reading before them. Dashed black: the fall a folded reading would show when k units of 2 MW start, for k = 1, 2, 3 (lowest to highest curve)."
            ),
            anchor="start",
        ),
    )
    save(chart=fig3, name="figure_3_fall_size_by_level")

    # ---- Figure 4: sign test, dwell, hour of day
    rows = []
    for name, ed in era_data.items():
        st = ed["steps"]
        for lo_l, hi_l in ((0, 3), (8, 10)):
            m = (st["before"] >= lo_l) & (st["before"] < hi_l)
            for kind, cond in (("rise", st["size"] >= 1.5), ("fall", st["size"] <= -1.5)):
                rows.append(
                    {
                        "era": name,
                        "band": f"from {lo_l} to {hi_l} MVA",
                        "kind": kind,
                        "count": int((cond & m).sum()),
                    }
                )
    sign = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("era:N", title=None, sort=list(ERAS), axis=alt.Axis(labelAngle=0)),
            xOffset="kind:N",
            y=alt.Y("count:Q", title="Merged steps of at least 1.5 MVA"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["fall", "rise"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH, height=120)
        .facet(row=alt.Row("band:N", title="Reading before the step"))
    )
    # dwell: time from each fall of at least 1.5 MVA to the next rise of at least 1.5 MVA
    rows = []
    for name, ed in era_data.items():
        st = ed["steps"]
        idx = np.argsort(st["first"])
        sz, f0, l0 = st["size"][idx], st["first"][idx], st["last"][idx]
        for i in range(len(sz) - 1):
            if sz[i] <= -1.5:
                nxt = next((j for j in range(i + 1, len(sz)) if sz[j] >= 1.5), None)
                if nxt is not None:
                    rows.append({"era": name, "dwell": int(f0[nxt] - l0[i])})
    dw = pl.DataFrame(rows)
    for name in ERAS:
        sub = dw.filter(pl.col("era") == name)["dwell"]
        if len(sub):
            say(
                f"{name}: dwell from a fall to the next rise (half-hours): n {len(sub)}, median {sub.median():.0f}, share at 1 {float(np.mean(sub.to_numpy() <= 1)):.2f}"
            )
    dwell = (
        alt.Chart(dw.filter(pl.col("dwell") <= 48))  # ty: ignore[unresolved-attribute]
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "dwell:Q",
                bin=alt.Bin(step=2),
                title="Half-hours from a fall to the next rise (up to 48)",
            ),
            y=alt.Y("count()", title="Count"),
            color=alt.Color(
                "era:N", scale=alt.Scale(domain=list(ERAS), range=COLOURS), legend=None
            ),
        )
        .properties(width=WIDTH, height=120)
    )
    net = era_data["net"]
    hod = pl.DataFrame(
        [
            {"hour": int(h), "kind": kind, "count": int(((net["hours"] == h) & cond).sum())}
            for h in range(24)
            for kind, cond in (
                ("fall", net["steps"]["size"] <= -1.5),
                ("rise", net["steps"]["size"] >= 1.5),
            )
        ]
    )
    hourly = (
        alt.Chart(hod)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "hour:Q",
                title="Hour of day (UTC) of the step in the net era",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
            ),
            y=alt.Y("count:Q", title="Merged steps"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["fall", "rise"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH, height=120)
    )
    fig4 = alt.vconcat(sign, dwell, hourly).properties(
        title=alt.TitleParams(
            wrap(
                "Direction, duration, and timing of D8's large steps. If the reading folds, a start from below about 3 MVA shows as a rise."
            ),
            anchor="start",
        )
    )
    save(chart=fig4, name="figure_4_step_direction_dwell_hour")

    # ---- Figure 5: Gas A level histogram, no time axis
    levels = np.round(g / 0.25) * 0.25
    h = pl.DataFrame({"mw": levels}).group_by("mw").len().sort("mw")
    fig5 = (
        alt.Chart(h)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE, size=8)
        .encode(
            x=alt.X(
                "mw:Q", title="Gas A output (MW, 0.25 MW bins)", axis=alt.Axis(values=[0, 2, 4, 6])
            ),
            y=alt.Y("len:Q", title="Half-hours (log scale)", scale=alt.Scale(type="symlog")),
        )
        .properties(
            width=WIDTH,
            height=150,
            title=alt.TitleParams(
                wrap("Output levels of Gas A over its 44-day record: three units of about 2 MW."),
                anchor="start",
            ),
        )
    )
    save(chart=fig5, name="figure_5_gas_a_levels")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "printed_look.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()

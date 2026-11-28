"""Figures of experiment 736, written under the data folder only (private, never published).

Every figure shows synthetic quantities or aggregates over injected transfers: histograms of the
injected durations and sizes, nine injected additions in megawatts on an axis of hours from the
start, and nine detector rows in noise units on the same axis. No figure shows a date or a level
of D8's real series: the decoded offset is shifted to zero before it is drawn.
"""

from __future__ import annotations

from typing import Any, Final

import altair as alt
import numpy as np
import polars as pl
from e00736_common import OUT, POINT_MW
from e00736_world import months_and_draws, transfer_table

ORANGE: Final[str] = "#FF4901"
BLUE: Final[str] = "#306BFF"
WIDTH: Final[int] = 600
FIG_DIR: Final[str] = "figures"
TARGET_HOURS: Final[tuple[float, ...]] = (6.0, 30.0, 90.0)
SHAPES: Final[tuple[str, ...]] = ("constant", "demand", "solar")


def _save(*, chart: alt.TopLevelMixin, name: str) -> None:
    path = OUT / FIG_DIR
    path.mkdir(parents=True, exist_ok=True)
    chart.save(str(path / f"{name}.svg"))


def pick_transfers(*, table: pl.DataFrame) -> pl.DataFrame:
    """Pick one transfer per duration (about 6, 30, and 90 hours) and shape where one exists.

    Args:
        table: The transfer table.

    Returns:
        Up to nine rows, the transfers of at least 2 step noise units nearest each target.
    """
    rows = []
    for shape in SHAPES:
        for hours in TARGET_HOURS:
            pool = table.filter((pl.col("shape") == shape) & (pl.col("units") >= 2.0))
            if pool.height:
                rows.append(
                    pool.with_columns(d=(pl.col("hours") - hours).abs()).sort("d").head(1).drop("d")
                )
    return pl.concat(rows).unique(subset=["seg", "k"], maintain_order=True).head(9)


def histogram_figure(*, table: pl.DataFrame) -> None:
    """Draw the histograms of injected durations and sizes.

    Args:
        table: The transfer table.
    """
    dur = (
        alt.Chart(alt.Data(values=table.select("hours", "units").to_dicts()))
        .mark_bar(color=BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hours:Q", bin=alt.Bin(maxbins=24), title="Duration (hours)"),
            y=alt.Y("count()", title="Injected transfers"),
        )
        .properties(
            width=WIDTH,
            height=120,
            title="Injected durations (24 and 96 hours bound the counted cell)",
        )
    )
    size = (
        alt.Chart(alt.Data(values=table.select("hours", "units").to_dicts()))
        .mark_bar(color=ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "units:Q",
                bin=alt.Bin(maxbins=24),
                title="Size (step noise units; 4 or more is counted)",
            ),
            y=alt.Y("count()", title="Injected transfers"),
        )
        .properties(width=WIDTH, height=120, title="Injected sizes")
    )
    _save(chart=alt.vconcat(dur, size), name="injected_transfers_histograms")


def addition_figures(*, table: pl.DataFrame, env: Any) -> None:
    """Draw each picked transfer's injected addition and its mirror, in megawatts, from its start.

    Args:
        table: The picked transfers.
        env: The regenerated bank and overlays.
    """
    for n, r in enumerate(table.iter_rows(named=True)):
        o = env.overlays[r["seg"]]
        tr = sorted(o.transfers, key=lambda x: x.start_tick)[r["k"]]
        level = tr.mean_levels() * POINT_MW
        hours = np.arange(len(level)) / 2.0
        rows = [
            {"hours": float(h), "mw": float(v), "series": "D8 gains"}
            for h, v in zip(hours, level, strict=True)
        ]
        if tr.mirrored:
            rows += [
                {"hours": float(h), "mw": float(-tr.jitter * v), "series": f"{tr.partner} loses"}
                for h, v in zip(hours, level, strict=True)
            ]
        chart = (
            alt.Chart(alt.Data(values=rows))
            .mark_line(interpolate="step-after", aria=False)
            .encode(  # ty: ignore[unresolved-attribute]  # ty: ignore[unresolved-attribute]
                x=alt.X("hours:Q", title="Hours from the injected start"),
                y=alt.Y("mw:Q", title="Added power (MW)"),
                color=alt.Color("series:N", scale=alt.Scale(range=[BLUE, ORANGE]), title=None),
            )
            .properties(
                width=WIDTH,
                height=130,
                title=f"Injected addition, {tr.shape} shape, {r['hours']:.0f} hours (synthetic)",
            )
        )
        _save(chart=chart, name=f"addition_{n}")


def detector_figures(
    *,
    table: pl.DataFrame,
    head_p: dict[int, tuple[np.ndarray, np.ndarray]],
    draw_files: bool = True,
) -> None:
    """Draw, for each picked transfer, the head's step probability and the offset posterior.

    The offset posterior mean of D8 is drawn in step noise units after subtracting its median over
    the window, so no D8 level is visible. A transfer whose world has no decoded posterior shows
    the head's row only.

    Args:
        table: The picked transfers.
        head_p: Segment to `(dense half-hours, calibrated step probability)`.
        draw_files: Whether to read the decoded posteriors.
    """
    for n, r in enumerate(table.iter_rows(named=True)):
        times, prob = head_p[r["seg"]]
        lo, hi = r["start_hh"] - 24, r["end_hh"] + 24
        keep = (times >= lo) & (times <= hi)
        rows = [
            {
                "hours": float((t - r["start_hh"]) / 2.0),
                "value": float(p),
                "row": "Step head: step probability",
            }
            for t, p in zip(times[keep], prob[keep], strict=True)
        ]
        post = OUT / "decode" / f"world_{r['draw']}_H-nbr_d8_posterior.npz"
        if draw_files and post.exists():
            with np.load(post) as z:
                mean, sigma = z["mean_mw"], float(z["sigma"])
            blocks = np.arange((r["start_panel"] - 24) // 12, (r["end_panel"] + 24) // 12 + 1)
            vals = mean[blocks] / sigma
            vals = vals - np.nanmedian(vals)
            rows += [
                {
                    "hours": float((b * 12 - r["start_panel"]) / 2.0),
                    "value": float(v),
                    "row": "Offset model: switched power (noise units)",
                }
                for b, v in zip(blocks, vals, strict=True)
            ]
        chart = (
            alt.Chart(alt.Data(values=rows))
            .mark_line(color=BLUE, interpolate="step-after", aria=False)
            .encode(  # ty: ignore[unresolved-attribute]  # ty: ignore[unresolved-attribute]
                x=alt.X("hours:Q", title="Hours from the injected start"),
                y=alt.Y("value:Q", title=None),
            )
            .properties(width=WIDTH, height=100)
            .facet(row=alt.Row("row:N", title=None))
            .resolve_scale(y="independent")
        )
        _save(chart=chart, name=f"detectors_{n}")


def make_figures() -> None:
    """Draw every figure from the regenerated overlays and the stored head outputs."""
    import e00736_heads as heads
    from e00736_world import load_env

    env = load_env()
    table = transfer_table(segments=env.segments, overlays=env.overlays, noise_d8=env.noise_d8)
    histogram_figure(table=table)
    picked = pick_transfers(table=table)
    addition_figures(table=picked, env=env)
    frozen = heads.frozen_thresholds()
    out = heads.load_outputs(arm=heads.HEAD_ARM)
    xs, ys, _ = heads.frozen_threshold(arm=heads.HEAD_ARM, frozen=frozen)
    cal = heads.iso_apply(p=out.step_p, xs=xs, ys=ys)
    head_p = {}
    for seg in months_and_draws(segments=env.segments)["seg"].to_list():
        first = int(np.asarray(env.segments[seg].dates).astype("datetime64[D]")[0].astype(int))
        pick = out.row_seg == seg
        head_p[seg] = ((out.row_date[pick] - first) * 48 + out.row_hh[pick], cal[pick])
    detector_figures(table=picked, head_p=head_p)

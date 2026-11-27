# ruff: noqa: E501
"""Draw experiment 361's injection figure from stored outputs and the experiment 252 cache.

Injection panels use host D2, one draw per size with a fixed seed. Each draw shows the first
missed spell (none of its half-hours detected, and at least 2 days into the window so a baseline
exists) and the first false alarm (a detected segment that overlaps no injected spell).

    PYTHONPATH=$E252:$E264:$E354:$E361 uv run --no-sync python ml-experiments/00361-*/e00361_timeseries.py
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00361_detect as detect
import e00361_inject as inject
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_figure_guard as guard

SLUG: Final[str] = "00361-d8-step-catalogue-detector-injection-and-gam-stack"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 480
YEAR_START: Final[datetime] = datetime(2025, 7, 7)  # noqa: DTZ001
N_WEEKS: Final[int] = 51
RED: Final[str] = "#B30000"


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    guard.save_chart(chart=chart, path=DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def _pairs(seg: detect.Segments) -> list[tuple[int, int]]:
    return list(zip(seg.start.tolist(), seg.end.tolist(), strict=True))


def _py(t: np.datetime64) -> datetime:
    return t.astype("datetime64[us]").item()


def _host_series(*, host: str) -> tuple[np.ndarray, np.ndarray]:
    times, power = detect.load_power_mw(site=host)
    lo = int(np.searchsorted(times, np.datetime64(inject.WINDOW[0])))
    hi = int(np.searchsorted(times, np.datetime64(inject.WINDOW[1])))
    return times[lo:hi], power[lo:hi]


def figure_5(*, catalogue: pl.DataFrame) -> None:
    """Simulated spells injected into D2 with the detector's output, for 2 MW and 4 MW."""
    dwells = catalogue["dwell_days"].to_numpy()
    gaps = (catalogue["start"].to_numpy()[1:] - catalogue["end"].to_numpy()[:-1]) / np.timedelta64(
        1, "D"
    )
    gaps = gaps[gaps > 0.25]
    times, base = _host_series(host="D2")
    n = len(base)
    panels = []
    for size in (2.0, 4.0):
        rng = np.random.default_rng(inject.SEED + int(size))
        spells = inject.spells_for(rng=rng, n=n, dwells=dwells, gaps=gaps)
        injected = np.zeros(n)
        for s, e in spells:
            injected[s:e] += size
        series = base - injected
        found = _pairs(detect.detect(power_mw=series)) + _pairs(
            detect.detect_drops(power_mw=series)
        )
        flag, level = inject.union_flag(n=n, power=series)
        scored = inject.score_spells(spells=spells, truth=injected, flag=flag, level=level)
        missed = [
            sp
            for sp, r in zip(spells, scored, strict=True)
            if r["covered"] == 0 and sp[0] >= 2 * 48
        ]
        false = [seg for seg in found if not any(seg[0] < e and s < seg[1] for s, e in spells)]
        picks = []
        if missed:
            picks.append(("Missed spell", missed[0][0], missed[0][1]))
        if false:
            picks.append(("False alarm", false[0][0], false[0][1]))
        print(size, "spells", len(spells), "missed", len(missed), "false alarms", len(false), picks)
        for label, s, e in picks:
            mid = (s + e) // 2
            lo, hi = max(mid - 4 * 48, 0), min(mid + 4 * 48, n)
            line = pl.DataFrame({"time": times[lo:hi], "mw": series[lo:hi]})
            a, b = (t.astype("datetime64[us]").item() for t in (times[lo], times[hi - 1]))
            x = alt.X(
                "time:T", title=None, scale=alt.Scale(domain=[a, b]), axis=alt.Axis(format="%d %b")
            )
            layers = [
                alt.Chart(line)
                .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.2)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x, y=alt.Y("mw:Q", title="D2 + simulated (MW)", scale=alt.Scale(zero=False))
                )
            ]
            truth = [(s2, e2) for s2, e2 in spells if s2 < hi and e2 > lo]
            det = [(s2, e2) for s2, e2 in found if s2 < hi and e2 > lo]
            rects = [
                {
                    "start": _py(times[max(s2, lo)]),
                    "end": _py(times[min(e2, hi) - 1]),
                    "what": "Injected spell",
                }
                for s2, e2 in truth
            ]
            rects += [
                {
                    "start": _py(times[max(s2, lo)]),
                    "end": _py(times[min(e2, hi) - 1]),
                    "what": "Detected segment",
                }
                for s2, e2 in det
            ]
            if rects:
                frame = pl.DataFrame(rects).with_columns(
                    row=pl.when(pl.col("what") == "Injected spell")
                    .then(pl.lit(1.0))
                    .otherwise(pl.lit(0.0))
                )
                layers.insert(0, alt.Chart(frame).mark_rect(opacity=0.3).encode(x="start:T", x2="end:T", color=alt.Color("what:N", scale=alt.Scale(domain=["Injected spell", "Detected segment"], range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE]), legend=alt.Legend(title=None, orient="top"))))  # ty: ignore[unresolved-attribute]  # fmt: skip
            panels.append(
                alt.layer(*layers).properties(
                    width=WIDTH,
                    height=120,
                    title=f"{size:.0f} MW injected: {label.lower()} (from {str(times[lo])[:10]})",
                )
            )
    _save(
        alt.vconcat(*panels).resolve_scale(color="shared"),
        name="figure_5_injected_spells_with_detector",
    )


def main() -> None:
    """Draw the injection figure.

    Figures 4 and 6 (dated weeks of D8 with the inferred spells shaded) are not drawn: they show a
    series inferred from a metered generator's record on calendar dates.
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    cat = (
        pl.concat(
            [
                pl.read_parquet(WORK / "d8_negative_segments.parquet"),
                pl.read_parquet(WORK / "d8_drops.parquet"),
            ]
        )
        .unique()
        .sort("start", "end")
    )
    figure_5(catalogue=cat)


if __name__ == "__main__":
    main()

"""The synthetic world gallery of experiment 738, committed before the first fit.

Each panel shows one component of the world separately, on a real Primary other than D8: the real
cleaned power, one injected shift of each family (constant step, proportional shape, ramp, short,
long, and held-out), the same shift in the history rows and in the target, the strata marked on one
40-day record, the fitted survival curve against the generator's true curve, and A1's output on one
shift. Axes count days from the start of the panel, so no panel carries a calendar date. The charts
are private (written under the data folder), and the published gallery is a separate step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import altair as alt
import e00738_common as common
import e00738_survival as surv
import e00738_world as world
import numpy as np
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, DATA_GREEN, DATA_PURPLE, DATA_SKY

PANEL_DAYS: Final[int] = 40
WIDTH: Final[int] = 600
HEIGHT: Final[int] = 180
FIG_DIR: Final[str] = "figures/gallery"


def _save(*, chart: alt.TopLevelMixin, name: str, out: Path | None = None) -> Path:
    path = (out if out is not None else common.OUT) / FIG_DIR
    path.mkdir(parents=True, exist_ok=True)
    chart.save(str(path / f"{name}.svg"))
    return path / f"{name}.svg"


def _lines(
    *, days: np.ndarray, series: dict[str, np.ndarray], y_title: str, colours: list[str]
) -> alt.Chart:
    rows = [
        {"day": float(d), "series": name, "power": float(v)}
        for name, arr in series.items()
        for d, v in zip(days, arr, strict=True)
        if np.isfinite(v)
    ]
    return (
        alt.Chart(alt.Data(values=rows))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            alt.X("day:Q", title="Days from the start of the panel"),
            alt.Y("power:Q", title=y_title),
            alt.Color("series:N", scale=alt.Scale(range=colours), title=None),
        )
        .properties(width=WIDTH, height=HEIGHT)
    )


def make_shift(
    *,
    kind: str,
    family: str,
    onset_day: float,
    days: float,
    ramp: int,
    depth: float,
    sign: int,
    end_cap: int,
) -> world.Shift:
    """Build one shift for a panel from its parameters (onset in days from the panel start)."""
    onset = int(onset_day * world.PER_DAY)
    return world.Shift(
        primary=0,
        kind=kind,
        onset_idx=onset,
        ramp_half_hours=ramp,
        end_idx=min(onset + int(days * world.PER_DAY), end_cap),
        drawn_days=days,
        depth=depth,
        sign=sign,
        family=family,
    )


def draw_gallery(*, clean_row: np.ndarray, out: Path | None = None) -> list[Path]:
    """Draw the gallery panels from the cleaned power of one real Primary (not D8).

    Args:
        clean_row: The Primary's cleaned power over capacity, at least 40 days of half-hours, NaN
        where dropped.
        out: The output folder (`OUT` by default).

    Returns:
        The files written.
    """
    n = PANEL_DAYS * world.PER_DAY
    base = clean_row[:n].astype(np.float64)
    days = np.arange(n) / world.PER_DAY
    paths = [
        _save(
            chart=_lines(
                days=days,
                series={"real": base * 100},
                y_title="Power (points of capacity)",
                colours=[DATA_BLUE],
            ),
            name="01_real",
            out=out,
        )
    ]
    specs = {
        "02_constant_step": ("constant", "train", 12.0, 10, 1, 0.15, 1),
        "03_proportional": ("proportional", "train", 12.0, 10, 1, 0.15, 1),
        "04_ramp": ("constant", "train", 12.0, 10, 72, -0.15, -1),
        "05_short": ("constant", "train", 12.0, 3, 1, 0.20, 1),
        "06_long": ("constant", "train", 5.0, 30, 1, 0.20, -1),
        "07_held_out": ("constant", "held_out", 8.0, 21, 200, 0.40, 1),
    }
    for name, (kind, fam, onset, dur, ramp, depth, sign) in specs.items():
        s = make_shift(
            kind=kind,
            family=fam,
            onset_day=onset,
            days=dur,
            ramp=ramp,
            depth=abs(depth),
            sign=sign,
            end_cap=n,
        )
        own = world.build_own(clean=base[None, :], shifts=[s], is_mw=np.array([True]))[0]
        paths.append(
            _save(
                chart=_lines(
                    days=days,
                    series={"real": base * 100, "injected": (base + own) * 100},
                    y_title="Power (points of capacity)",
                    colours=[DATA_BLUE, BRAND_ORANGE],
                ),
                name=name,
                out=out,
            )
        )
    # the same shift in the history rows and in the target of one origin
    s = make_shift(
        kind="constant",
        family="train",
        onset_day=12.0,
        days=10,
        ramp=1,
        depth=0.15,
        sign=1,
        end_cap=n,
    )
    own = world.build_own(clean=base[None, :], shifts=[s], is_mw=np.array([True]))[0]
    origin = int(18.5 * world.PER_DAY)
    rows = [
        {"day": float(d), "part": part, "power": float(v)}
        for part, lo, hi in (
            ("history rows (14 days)", origin - 14 * world.PER_DAY, origin),
            ("target (14 days)", origin, origin + 14 * world.PER_DAY),
        )
        for d, v in zip(days[lo:hi], ((base + own) * 100)[lo:hi], strict=True)
        if np.isfinite(v)
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            alt.X("day:Q", title="Days from the start of the panel"),
            alt.Y("power:Q", title="Power (points of capacity)"),
            alt.Color("part:N", scale=alt.Scale(range=[DATA_GREEN, DATA_PURPLE]), title=None),
        )
        .properties(width=WIDTH, height=HEIGHT)
    )
    paths.append(_save(chart=chart, name="08_history_and_target", out=out))
    # strata marked on one 40-day record
    s = make_shift(
        kind="constant",
        family="train",
        onset_day=10.0,
        days=14,
        ramp=1,
        depth=0.15,
        sign=1,
        end_cap=n,
    )
    origins = np.arange(0, n - 10 * world.PER_DAY, world.PER_DAY) + 24
    lab = world.strata(
        origin_idx=origins,
        valid_idx=origins + 5 * world.PER_DAY,
        onset_idx=np.full(len(origins), s.onset_idx),
        end_idx=np.full(len(origins), s.end_idx),
        clipped=np.full(len(origins), False),
    )
    chart = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            alt.Data(
                values=[
                    {"day": float(o / world.PER_DAY), "stratum": str(x)}
                    for o, x in zip(origins, lab, strict=True)
                ]
            )
        )
        .mark_tick(size=14)
        .encode(
            alt.X("day:Q", title="Origin, days from the start of the panel"),
            alt.Y("stratum:N", title=None, sort=list(world.STRATA)),
            alt.Color(
                "stratum:N",
                legend=None,
                scale=alt.Scale(
                    range=[
                        DATA_BLUE,
                        DATA_SKY,
                        DATA_GREEN,
                        DATA_PURPLE,
                        BRAND_ORANGE,
                        "#888888",
                        "#bbbbbb",
                    ]
                ),
            ),
        )
        .properties(width=WIDTH, height=HEIGHT)
    )
    paths.append(_save(chart=chart, name="09_strata", out=out))
    # fitted and true survival
    rng = np.random.default_rng(0)
    drawn = np.where(
        rng.random(4000) < 0.5, rng.integers(3, 15, 4000), rng.integers(28, 113, 4000)
    ).astype(float)
    curve = surv.fit_curve(durations_days=drawn)
    t = np.linspace(0, 120, 241)
    rows = [
        {"day": float(a), "curve": name, "survival": float(b)}
        for name, f in (
            ("fitted " + curve.family, curve.sf),
            ("generator", surv.GeneratorCurve().sf),
        )
        for a, b in zip(t, f(t), strict=True)
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            alt.X("day:Q", title="Duration (days)"),
            alt.Y("survival:Q", title="Chance the shift lasts longer"),
            alt.Color("curve:N", scale=alt.Scale(range=[BRAND_ORANGE, DATA_BLUE]), title=None),
        )
        .properties(width=WIDTH, height=HEIGHT)
    )
    paths.append(_save(chart=chart, name="10_survival_curves", out=out))
    # A1's output on one shift, for origins walking through the shift
    s = make_shift(
        kind="constant",
        family="train",
        onset_day=14.0,
        days=20,
        ramp=1,
        depth=0.20,
        sign=1,
        end_cap=n,
    )
    own = world.build_own(clean=base[None, :], shifts=[s], is_mw=np.array([True]))[0]
    z = np.nan_to_num((base + own) - np.nanmean(base), nan=0.0)
    unit = surv.noise_unit_from_panel(
        z=(base[None, :] - np.nanmean(base)) / max(np.nanstd(base), 1e-6), cutoff_idx=n
    )
    ages = []
    for c in range(16 * world.PER_DAY + 24, n - world.PER_DAY, world.PER_DAY):
        idx = c - world.PER_DAY * np.arange(2, 16)[:, None] + np.arange(world.PER_DAY)[None, :]
        vis = idx <= c - 54
        ages.append(
            (
                c / world.PER_DAY,
                float(
                    surv.age_rule_a1(
                        hist_power=(z[idx] / max(np.nanstd(base), 1e-6))[None],
                        visible=vis[None],
                        noise=unit,
                    )[0]
                ),
                (c - s.onset_idx) / world.PER_DAY,
            )
        )
    rows = [
        {"origin_day": a, "series": name, "age": v}
        for a, found, true in ages
        for name, v in (("A1 age", found), ("true age", max(true, 0.0)))
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            alt.X("origin_day:Q", title="Origin, days from the start of the panel"),
            alt.Y("age:Q", title="Age of the shift (days)"),
            alt.Color("series:N", scale=alt.Scale(range=[BRAND_ORANGE, DATA_BLUE]), title=None),
        )
        .properties(width=WIDTH, height=HEIGHT)
    )
    paths.append(_save(chart=chart, name="11_a1_on_one_shift", out=out))
    return paths

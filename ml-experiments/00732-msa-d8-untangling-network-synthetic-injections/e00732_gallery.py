# ruff: noqa: E501, PLR0911
"""Experiment 732: the gallery panels G10 to G21 and their automatic checks, drawn before any fit.

Every panel is built from arrays, and its check reads those same arrays (never a chart), so a
generator that silently produces nothing fails its panel. A panel function returns the charts, the
result of its check, and a counts-only note. `stage_gallery` draws every panel from the test and
validation banks, the training overlay of seed 0, the neighbour table, and one built input tensor,
writes the published SVGs to `figures/gallery/` and the private ones (G17 only) to
`PRIVATE/gallery/`, and writes `PRIVATE/gallery/auto_check.json`. `record_panel_check` combines that
file with the reading of the person who checked each panel against the written list
(`core.PANEL_REQUIREMENTS`) into `PRIVATE/gallery/panel_check.json`, which the fit stages read.

No axis carries a calendar date: a window is labelled by the hours from the start of its day 1.
Every published window comes from an engine-only, S2, or no-injection segment, and a neighbour is
drawn as a multiple of its own median.
"""

import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import altair as alt
import e00729_core as core729
import e00729_figures as figures729
import e00729_sim as sim729
import e00730_banks as banks730
import e00730_masks as masks
import e00732_arms as arms
import e00732_core as core
import e00732_leakage as leak
import e00732_neighbours as nb
import e00732_overlays as ov
import e00732_stages as stages
import numpy as np
import polars as pl

WIDTH_PX: Final = figures729.WIDTH_PX
BLUE, ORANGE, GREEN = figures729.BLUE, figures729.ORANGE, figures729.GREEN
PURPLE, SKY, GREY = figures729.PURPLE, figures729.SKY, figures729.GREY
HIDDEN_FILL: Final = "#BBBBBB"
PRIVATE_SUBFOLDER: Final = "gallery"
PANEL_FILE: Final = "panel_check.json"
AUTO_FILE: Final = "auto_check.json"
PAIR_ORDER_SEED: Final = 732_011
"""Seeds the order in which G11 and G17 visit their candidate windows (the same order at every amount)."""
N_DRAWS: Final = 3
"""Each panel shows this many draws, from different segments."""
TOL: Final = 1e-3
"""The tolerance of an equality between two float32 arrays in points."""
PARTNER_SHARE_TOLERANCE: Final = 0.08
"""How far the share of transfers under BSP A may lie from the registered 0.8."""
SIZE_NOISE_LINE: Final = 4.0
"""The size in D8's noise units that the plan expects to separate detectable from undetectable."""
DURATION_BAND_LABELS: Final = ("1-3 h", "3-12 h", "12 h-2 d", "2-4 d")
COLOURS: Final = (BLUE, ORANGE, GREEN, PURPLE, SKY, GREY)
PAIR_SUFFIX: Final = re.compile(
    r" (without( the transfer)?|with( the transfer| the injection)?|w\+?)$"
)
"""Strips the suffix that tells the thin member of a pair from the thick one, so that both take one colour."""


def gallery_private_dir(*, root: Path) -> Path:
    """Return the private folder that holds G17, `auto_check.json`, and `panel_check.json`."""
    return root / "PRIVATE" / PRIVATE_SUBFOLDER


# --------------------------------------------------------------------------- charts


def lines_chart(
    *,
    series: Mapping[str, np.ndarray],
    title: str,
    widths: Mapping[str, float] | None = None,
    rules: Sequence[float] = (),
    hidden: np.ndarray | None = None,
    marks: np.ndarray | None = None,
    y_title: str = "Points of capacity",
    height: int = 120,
    n_days: int = 2,
    colours: Mapping[str, str] | None = None,
) -> alt.LayerChart | alt.Chart:
    """Return lines over a window of `n_days` days with the edges as rules.

    The chart shades a hidden span and marks the half-hours.

    Args:
        series: Named arrays of equal length over the window's half-hours.
        title: What the panel shows, in one or two short lines.
        widths: Stroke width by series name (default 1.5), so that a thin line can lie under a thick one.
        rules: Hours from the start of day 1 at which a dotted vertical rule is drawn (transfer edges).
        hidden: Positions whose reading is hidden, shaded grey.
        marks: Half-hour positions drawn as small ticks along the bottom (the half-hours with an injection).
        y_title: The y-axis title with its unit.
        height: Plot height in pixels.
        n_days: The window length in days.
        colours: Colour by series name (the default order of `COLOURS` otherwise), so that a thin
            and a thick line of one quantity share a colour.

    Returns:
        The chart, with no calendar date anywhere.
    """
    names = list(series)
    roots = [PAIR_SUFFIX.sub("", k) for k in names]
    palette = {r: COLOURS[i % len(COLOURS)] for i, r in enumerate(dict.fromkeys(roots))}
    colours = {k: (colours or {}).get(k, palette[r]) for k, r in zip(names, roots, strict=True)}
    n = len(series[names[0]])
    data = pl.DataFrame(
        {
            "hour": np.tile(np.arange(n) / 2.0, len(names)),
            "value": np.concatenate([np.asarray(series[k], dtype=float) for k in names]),
            "series": np.repeat(names, n),
        }
    )
    w = {k: (widths or {}).get(k, 1.5) for k in names}
    step = 12 if n_days <= 2 else 24
    lines = (
        alt.Chart(data)
        .mark_line(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                title="Hours from the start of day 1 (UTC)",
                axis=alt.Axis(values=list(range(0, 24 * n_days + 1, step))),
            ),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=names, range=[colours[k] for k in names]),
                legend=alt.Legend(
                    orient="top",
                    title=None,
                    symbolType="stroke",
                    symbolStrokeWidth=3,
                    labelLimit=400,
                    columns=2,
                ),
            ),
            size=alt.Size(
                "series:N",
                scale=alt.Scale(domain=names, range=[w[k] for k in names]),
                legend=None,
            ),
        )
    )
    layers: list[alt.Chart] = []
    if hidden is not None and hidden.any():
        idx = np.flatnonzero(hidden)
        span = pl.DataFrame({"a": [idx[0] / 2.0], "b": [(idx[-1] + 1) / 2.0]})
        layers.append(
            alt.Chart(span)
            .mark_rect(color=HIDDEN_FILL, opacity=0.35, aria=False)
            .encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
        )
    layers.append(lines)
    if len(rules):
        layers.append(
            alt.Chart(pl.DataFrame({"h": [float(h) for h in rules]}))
            .mark_rule(color="#000000", strokeDash=[2, 3], aria=False)
            .encode(x="h:Q")  # ty: ignore[unresolved-attribute]
        )
    if marks is not None and np.asarray(marks).size:
        layers.append(
            alt.Chart(pl.DataFrame({"h": (np.asarray(marks, dtype=float) / 2.0)}))
            .mark_tick(color=PURPLE, thickness=2, size=6, aria=False, y=height - 4)
            .encode(x="h:Q")  # ty: ignore[unresolved-attribute]
        )
    return alt.layer(*layers).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH_PX, height=height, title=alt.TitleParams(title, anchor="start")
    )


def stack(
    *, charts: Sequence[alt.TopLevelMixin], title: str, subtitle: str | Sequence[str] = ""
) -> alt.VConcatChart:
    """Stack panels vertically under one title, and an optional subtitle of one or more lines."""
    head = alt.TitleParams(title, anchor="start", subtitle=subtitle or alt.Undefined)
    return alt.vconcat(*charts, title=head).resolve_scale(  # ty: ignore[invalid-argument-type]
        color="independent", size="independent"
    )


# --------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class Win:
    """The arrays of one window of a transfer-overlaid segment, with and without the overlay.

    Attributes:
        index: The segment's position in its bank.
        day0: The dense day the window starts on.
        n_days: The window length in days.
        world: The segment's world type.
        source: The segment's solar source.
        transfer: The transfer the window was chosen for (None for a window chosen without one).
        beta: The segment's BSP coupling.
        x0: The stored truth.
        x1: The truth with the overlay.
        r0: The stored reading.
        r1: The reading with the overlay.
        base1: The base demand with the overlay.
        bsp0: The stored BSP meter.
        bsp1: The BSP meter with the overlay.
        t: The load that D8 gains in each half-hour.
        t_bsp: The part of `t` that moves the BSP meter.
        event: 1 on a half-hour that holds a transfer edge.
        size: The signed step size at each edge.
        solar: The solar target.
        engine: The engine output.
        csi: The clear-sky index.
        cosz: The cosine of the zenith angle.
        nb_before: Each neighbour's stored series (points, signed).
        nb_after: Each neighbour's series after the mirrors.
        floors: Each neighbour's fold floor in points.
    """

    index: int
    day0: int
    n_days: int
    world: str
    source: str
    transfer: ov.Transfer | None
    beta: float
    x0: np.ndarray
    x1: np.ndarray
    r0: np.ndarray
    r1: np.ndarray
    base1: np.ndarray
    bsp0: np.ndarray
    bsp1: np.ndarray
    t: np.ndarray
    t_bsp: np.ndarray
    event: np.ndarray
    size: np.ndarray
    solar: np.ndarray
    engine: np.ndarray
    csi: np.ndarray
    cosz: np.ndarray
    nb_before: Mapping[str, np.ndarray] = field(default_factory=dict)
    nb_after: Mapping[str, np.ndarray] = field(default_factory=dict)
    floors: Mapping[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """The window's half-hours."""
        return self.n_days * 48

    def rel(self, *, half_hour: int) -> int:
        """Return a dense half-hour as a position inside the window."""
        return half_hour - self.day0 * 48

    @property
    def edges(self) -> list[int]:
        """The positions of the window's transfer edges (the chosen transfer's start and end that lie inside)."""
        if self.transfer is None:
            return []
        pos = [
            self.rel(half_hour=self.transfer.start_half_hour),
            self.rel(half_hour=self.transfer.end_half_hour),
        ]
        return [p for p in pos if 0 <= p < self.n]


def window_of(
    *, overlay: ov.TransferOverlay, original: sim729.Segment, index: int, day0: int, n_days: int,
    transfer: ov.Transfer | None,
) -> Win:  # fmt: skip
    """Slice `n_days` days of an overlay and of its stored segment, from dense day `day0`."""

    def cut(a: np.ndarray) -> np.ndarray:
        return np.asarray(a)[day0 : day0 + n_days].reshape(-1).astype(np.float64)

    seg = overlay.seg
    return Win(
        index=index, day0=day0, n_days=n_days, world=seg.world, source=seg.source, transfer=transfer,
        beta=overlay.beta, x0=cut(original.x), x1=cut(seg.x), r0=cut(original.r), r1=cut(seg.r),
        base1=cut(seg.base), bsp0=cut(original.bsp), bsp1=cut(seg.bsp), t=cut(overlay.t),
        t_bsp=cut(overlay.t_bsp), event=cut(overlay.event), size=cut(overlay.size),
        solar=cut(seg.solar), engine=cut(seg.engine), csi=cut(seg.csi), cosz=cut(seg.cosz),
        nb_before={n: cut(overlay.neighbours_real[n]) for n in nb.NEIGHBOURS},
        nb_after={n: cut(overlay.neighbours[n]) for n in nb.NEIGHBOURS},
        floors=dict(overlay.floors),
    )  # fmt: skip


def publishable(*, seg: sim729.Segment) -> bool:
    """Return whether a segment may appear in a published panel (an engine-only, S2.

    Or no-injection segment, never a metered farm).
    """
    return seg.source != "S1"


def choose_windows(
    *,
    overlays: Sequence[ov.TransferOverlay],
    originals: Sequence[sim729.Segment],
    keep: Callable[[ov.TransferOverlay, ov.Transfer], bool],
    n: int = N_DRAWS,
    n_days: int = 2,
    public: bool = True,
) -> list[Win]:
    """Return up to `n` windows of different segments, each centred on a transfer that `keep` accepts.

    The window starts on the day of the transfer's start edge (so both edges show for a transfer that
    ends within `n_days`), moved inside the segment's dense axis, and every day of it lies on a base day.

    Args:
        overlays: The bank's transfer overlays.
        originals: The bank's stored segments, in the same order.
        keep: Whether a transfer of an overlay is wanted.
        n: The number of draws.
        n_days: The window length in days.
        public: Draw only from segments that may be published.

    Returns:
        The windows, in segment order.
    """
    out: list[Win] = []
    for i, o in enumerate(overlays):
        if public and not publishable(seg=o.seg):
            continue
        for tr in o.transfers:
            if not keep(o, tr):
                continue
            last = o.seg.x.shape[0] - n_days
            day0 = int(min(max(tr.start_half_hour // 48, 1), last))
            days = slice(day0, day0 + n_days)
            if (o.seg.cal_index[days] < 0).any() or not np.isfinite(o.seg.x[days]).all():
                continue
            lo, hi = day0 * 48, (day0 + n_days) * 48
            if any(
                t is not tr and t.start_half_hour < hi and t.end_half_hour >= lo
                for t in o.transfers
            ):
                continue
            out.append(
                window_of(
                    overlay=o, original=originals[i], index=i, day0=day0, n_days=n_days, transfer=tr
                )
            )
            break
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------- G10: load transfers at D8


def check_g10(*, wins: Sequence[Win], expect_bsp_moves: bool | None, min_days: int = 0) -> bool:
    """Check one G10 panel: the truth changes by the drawn load and the edges are marked.

    The check also requires that the BSP steps follow the partner.

    Every draw needs `x1 - x0 == t` on every half-hour, a transfer that changes the truth, an edge
    mark at each edge inside the window, a partial half-hour at an edge no larger than the drawn size,
    a BSP meter that moves by `beta t` for a transfer under another BSP and stays put otherwise.
    """
    if not wins:
        return False
    for w in wins:
        tr = w.transfer
        if tr is None or not np.allclose(w.x1 - w.x0, w.t, atol=TOL):
            return False
        if not np.abs(w.t).max() > 0.0 or not np.isfinite(w.x1).all():
            return False
        for edge in w.edges:
            if (
                w.event[edge] != 1.0
                or abs(w.t[edge])
                > abs(tr.size) * (np.max(tr.profile) if tr.profile is not None else 1.0) + TOL
            ):
                return False
        step = np.nan_to_num(w.bsp1 - w.bsp0, nan=0.0)
        if not tr.inside_bsp_a:
            if not np.allclose(step, w.beta * w.t_bsp, atol=TOL) or not np.abs(step).max() > 0.0:
                return False
        elif np.abs(step).max() > TOL:
            return False
        if expect_bsp_moves is not None and expect_bsp_moves != (not tr.inside_bsp_a):
            return False
        if min_days and (tr.ticks / (6 * 48) < min_days):
            return False
    return True


BASE_LABEL: Final = "base demand with the transfer, before generation"
G10_SUBTITLE: Final = (
    "Import is positive. A transfer that adds load lifts the thick truth line above the thin one by the load added; one that removes load lowers it.",
    "The truth also holds the engine and solar output, which lowers it below the base demand line by the generation.",
)


def g10_charts(*, wins: Sequence[Win], title: str, n_days: int) -> alt.VConcatChart:
    """Return a panel of draws: the truth with and without the overlay, and the base demand.

    The panel also draws the reading and the BSP steps.
    """
    charts: list[alt.TopLevelMixin] = []
    for k, w in enumerate(wins):
        rules = [p / 2.0 for p in w.edges]
        top = lines_chart(
            series={
                "truth without the transfer": w.x0, "truth with the transfer": w.x1,
                "reading": w.r1, BASE_LABEL: w.base1,
            },
            widths={"truth without the transfer": 0.8, "truth with the transfer": 2.2},
            colours={"truth without the transfer": SKY, "truth with the transfer": BLUE},
            title=f"Draw {k + 1}: D8 (dotted: transfer edges)", rules=rules, n_days=n_days,
        )  # fmt: skip
        d = np.concatenate([[0.0], np.diff(np.nan_to_num(w.bsp1, nan=0.0))])
        d0 = np.concatenate([[0.0], np.diff(np.nan_to_num(w.bsp0, nan=0.0))])
        bottom = lines_chart(
            series={"BSP step without the transfer": d0, "BSP step with the transfer": d},
            widths={"BSP step without the transfer": 0.8, "BSP step with the transfer": 2.2},
            title="The BSP meter's half-hour steps", rules=rules, height=70, n_days=n_days,
        )  # fmt: skip
        charts += [top, bottom]
    return stack(charts=charts, title=title, subtitle=G10_SUBTITLE)


def panel_g10(
    *, overlays: Sequence[ov.TransferOverlay], originals: Sequence[sim729.Segment]
) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:
    """Draw G10: four kinds of load transfer at D8, three draws each."""

    def adds_on_import_day(o: ov.TransferOverlay, tr: ov.Transfer) -> bool:
        d0 = int(tr.start_half_hour // 48)
        return tr.size > 0 and tr.inside_bsp_a and bool(np.nanmin(o.seg.x[d0 : d0 + 2]) > 0)

    def removes_in_solar_spell(o: ov.TransferOverlay, tr: ov.Transfer) -> bool:
        d0 = int(tr.start_half_hour // 48)
        return (
            tr.size < 0
            and tr.inside_bsp_a
            and o.seg.world in ov.SOLAR_WORLDS
            and bool(np.nanmax(o.seg.solar[d0 : d0 + 2]) > 0)
        )

    cases = {
        "g10_adds_load": ("A transfer that adds load on an import day, inside BSP A", adds_on_import_day, False, 0, 2),
        "g10_removes_load_in_solar_spell": ("A transfer that removes load during a solar spell", removes_in_solar_spell, False, 0, 2),
        "g10_other_bsp": ("A transfer to a Primary under another BSP", lambda o, tr: tr.partner == core.OTHER_BSP_PARTNER and bool(np.isfinite(o.seg.bsp[max(tr.start_half_hour // 48, 1) : max(tr.start_half_hour // 48, 1) + 2]).all()), True, 0, 2),
        "g10_long": ("A transfer of 2 days or more in a 5-day window", lambda o, tr: tr.ticks >= 2 * 48 * 6 and (tr.inside_bsp_a or bool(np.isfinite(o.seg.bsp[max(tr.start_half_hour // 48, 1) : max(tr.start_half_hour // 48, 1) + 5]).all())), None, 2, 5),
    }  # fmt: skip
    charts: dict[str, alt.TopLevelMixin] = {}
    ok = True
    counts = []
    for name, (title, keep, bsp, min_days, n_days) in cases.items():
        wins = choose_windows(overlays=overlays, originals=originals, keep=keep, n_days=n_days)
        counts.append(len(wins))
        ok &= check_g10(wins=wins, expect_bsp_moves=bsp, min_days=min_days)
        if wins:
            charts[name] = g10_charts(wins=wins, title=title, n_days=n_days)
    return charts, ok, f"draws per panel {counts}"


# --------------------------------------------------------------------------- G11 and G17: contrastive pairs


@dataclass(frozen=True)
class PairWin:
    """One window and the same window with an extra solar injection.

    Attributes:
        amount: The injected amount a in points.
        sibling_source: The solar source of the sibling that lends the shape.
        delta: The injection over the window's 97 half-hours (the half-hour before it first).
        x0: The truth of the window `(96,)`.
        x1: The truth less the injection.
        r0: The reading.
        r1: The re-folded reading.
        bsp0: The BSP meter.
        bsp1: The BSP meter less `beta delta`.
        cosz: The cosine of the zenith angle.
    """

    amount: float
    sibling_source: str
    delta: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    r0: np.ndarray
    r1: np.ndarray
    bsp0: np.ndarray
    bsp1: np.ndarray
    cosz: np.ndarray


def window_cols(*, day: int) -> slice:
    """Return the dense half-hours of the 96-half-hour window around a scored day."""
    start = day * 48 - core729.MIDDLE_START
    return slice(start, start + core729.WINDOW_STEPS)


def pair_windows(
    *, segments: Sequence[sim729.Segment], overlays: Sequence[ov.TransferOverlay] | None,
    index: ov.PairIndex, amount: float, source: str, public: bool, n: int = N_DRAWS,
) -> list[PairWin]:  # fmt: skip
    """Return up to `n` pair windows with a sibling of the given source.

    The windows differ in segment, in calendar day, and in sibling. Segments of one bank share a
    calendar, so taking the first candidates would give the same day and the same sibling shape to
    every draw. The candidates are visited in a fixed pseudo-random order instead, which is the same
    for every amount.

    Args:
        segments: The bank's stored segments (the plain window).
        overlays: Unused (the pair is built on the plain window); kept so that callers pass one bank.
        index: The bank's pair index.
        amount: The injected amount in points.
        source: "S1" or "S2", the sibling's source.
        public: Draw the window's own segment from publishable segments only.
        n: The number of draws.

    Returns:
        The pairs.
    """
    out: list[PairWin] = []
    used: set[int] = set()
    days: set[int] = set()
    siblings: set[int] = set()
    candidates = sorted(index.siblings.items())
    order = np.random.default_rng(PAIR_ORDER_SEED).permutation(len(candidates))
    for k in order:
        (i, d), sibs = candidates[int(k)]
        seg = segments[i]
        if i in used or d < 1 or (public and not publishable(seg=seg)):
            continue
        if public and seg.world in ov.SOLAR_WORLDS:
            continue
        pick = [int(j) for j in sibs if segments[int(j)].source == source]
        if not pick:
            continue
        cols = window_cols(day=d)
        if cols.start < 0 or cols.stop > seg.x.size or seg.cal_index[d] < 0:
            continue
        if int(seg.cal_index[d]) in days or pick[0] in siblings:
            continue
        delta97 = ov.window_delta(index=index, sibling=pick[0], amount=amount, day=d)
        delta = delta97[1:].astype(np.float64)
        sun_up = seg.cosz.reshape(-1)[cols] > core.NIGHT_COSZ
        if not (delta > 0).any() or (delta[~sun_up] > 0).any():
            continue
        x = seg.x.reshape(-1)[cols].astype(np.float64)
        q = ov.reactive_floor(seg=seg).reshape(-1)[cols]
        bsp = seg.bsp.reshape(-1)[cols]
        x_new, r_new, bsp_new = ov.inject(
            x=x, q=q, delta=delta, bsp=bsp, beta=float(seg.params[ov.BETA_KEY])
        )
        out.append(
            PairWin(
                amount=amount, sibling_source=segments[pick[0]].source, delta=delta,
                x0=x, x1=x_new.astype(np.float64), r0=seg.r.reshape(-1)[cols].astype(np.float64),
                r1=r_new.astype(np.float64), bsp0=bsp.astype(np.float64),
                bsp1=bsp_new.astype(np.float64), cosz=seg.cosz.reshape(-1)[cols].astype(np.float64),
            )
        )  # fmt: skip
        used.add(i)
        days.add(int(seg.cal_index[d]))
        siblings.add(pick[0])
        if len(out) == n:
            break
    return out


def check_pairs(*, pairs: Sequence[PairWin]) -> bool:
    """Check one pair panel: delta is above zero somewhere and only on sun-up half-hours.

    The truth drops by delta there. The re-folded reading equals the original where delta is zero
    and differs where it is not. No two draws may share one injection or one truth.
    """
    if not pairs:
        return False
    for a, b in itertools.combinations(pairs, 2):
        if np.array_equal(a.delta, b.delta) or np.array_equal(a.x0, b.x0, equal_nan=True):
            return False
    for p in pairs:
        active = p.delta > 0.0
        if not active.any() or (active & ~(p.cosz > core.NIGHT_COSZ)).any():
            return False
        seen = np.isfinite(p.x0)
        if not np.allclose((p.x0 - p.x1)[seen], p.delta[seen], atol=TOL):
            return False
        ok = np.isfinite(p.r0) & np.isfinite(p.r1)
        if (np.abs(p.r1 - p.r0)[ok & ~active] > TOL).any() or not (
            np.abs(p.r1 - p.r0)[ok & active] > 0
        ).any():
            return False
    return True


def pair_charts(*, pairs: Sequence[PairWin], title: str) -> alt.VConcatChart:
    """Return a panel for one amount: the injection, the truth pair, the reading pair, and the BSP pair."""
    charts: list[alt.TopLevelMixin] = []
    for k, p in enumerate(pairs):
        widths = {"original": 0.8, "with the injection": 2.2}
        same = {"original": BLUE, "with the injection": BLUE}
        marks = np.flatnonzero(p.delta > 0)
        charts += [
            lines_chart(series={"extra injection": p.delta}, title=f"Draw {k + 1}: the extra solar injection (a = {p.amount:g} points)", height=60, marks=None),
            lines_chart(series={"original": p.x0, "with the injection": p.x1}, widths=widths, colours=same, title="Signed truth", height=90),
            lines_chart(series={"original": np.nan_to_num(p.r0), "with the injection": np.nan_to_num(p.r1)}, widths=widths, colours=same, title="Reading (the injection re-folded; ticks: half-hours with the injection)", height=90, marks=marks),
            lines_chart(series={"original": np.nan_to_num(p.bsp0), "with the injection": np.nan_to_num(p.bsp1)}, widths=widths, colours=same, title="BSP meter", height=70),
        ]  # fmt: skip
    return stack(charts=charts, title=title)


def panel_g11(
    *,
    segments: Sequence[sim729.Segment],
    index: ov.PairIndex,
    public: bool = True,
    source: str = "S2",
) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:
    """Draw G11 (published, S2 siblings) or G17 (private, S1 siblings): one panel for each tested amount."""
    charts: dict[str, alt.TopLevelMixin] = {}
    ok = True
    counts = []
    for a in core.TEST_INJECTION_POINTS:
        pairs = pair_windows(
            segments=segments, overlays=None, index=index, amount=a, source=source, public=public
        )
        counts.append(len(pairs))
        ok &= check_pairs(pairs=pairs)
        if pairs:
            charts[f"g{'11' if public else '17'}_pairs_{a:g}"] = pair_charts(
                pairs=pairs,
                title=f"Contrastive pairs at a = {a:g} points, extra injection from a source {source} sibling",
            )
    return charts, ok, f"pairs per amount {counts}"


# --------------------------------------------------------------------------- G12: the raised sky


def check_g12(*, windows: Sequence[Mapping[str, np.ndarray]]) -> bool:
    """Check that the raised index is above the original on every raisable half-hour.

    Equal elsewhere, and that the window has raisable half-hours.
    """
    if not windows:
        return False
    for w in windows:
        raisable = ov.raisable(csi=w["csi"], cosz=w["cosz"])
        if not raisable.any():
            return False
        if not (w["raised"][raisable] > w["csi"][raisable]).all():
            return False
        elsewhere = ~(w["cosz"] > core.SUN_UP_COSZ)
        if not np.allclose(
            np.nan_to_num(w["raised"][elsewhere]), np.nan_to_num(w["csi"][elsewhere])
        ):
            return False
        if not (np.nan_to_num(w["raised"]) <= core.CSI_CLIP + 1e-6).all():
            return False
    return True


def panel_g12(
    *, segments: Sequence[sim729.Segment]
) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:
    """Draw G12: the clear-sky index of a window and of its raised-sky member, on a clear day and a cloudy day."""
    best: dict[str, tuple[float, int, int]] = {}
    for i, seg in enumerate(segments):
        if not publishable(seg=seg):
            continue
        for d in range(1, seg.x.shape[0] - 2):
            up = seg.cosz[d] > core.SUN_UP_COSZ
            if seg.cal_index[d] < 0 or up.sum() < 6 or not np.isfinite(seg.csi[d : d + 2]).any():
                continue
            level = float(np.nanmean(seg.csi[d][up]))
            if not np.isfinite(level):
                continue
            if "clear" not in best or level > best["clear"][0]:
                best["clear"] = (level, i, d)
            if "cloudy" not in best or level < best["cloudy"][0]:
                best["cloudy"] = (level, i, d)
    windows: list[dict[str, np.ndarray]] = []
    charts: list[alt.TopLevelMixin] = []
    for label, (_, i, d) in best.items():
        seg = segments[i]
        csi, cosz = seg.csi[d : d + 2].reshape(-1), seg.cosz[d : d + 2].reshape(-1)
        raised = ov.raise_sky(csi=csi, cosz=cosz, eps=core.TEST_EPS)
        windows.append(
            {
                "csi": csi.astype(np.float64),
                "cosz": cosz.astype(np.float64),
                "raised": raised.astype(np.float64),
            }
        )
        mask = ov.raisable(csi=csi, cosz=cosz)
        charts.append(
            lines_chart(
                series={"clear-sky index": np.nan_to_num(csi), "raised by 20%": np.nan_to_num(raised), "reading (unchanged)": np.nan_to_num(seg.r[d : d + 2].reshape(-1)) / 100.0},
                widths={"clear-sky index": 0.9, "raised by 20%": 2.2, "reading (unchanged)": 1.0},
                title=f"A {label} day: ticks mark the raisable half-hours; the reading is divided by 100 and is the same in both members",
                marks=np.flatnonzero(mask), y_title="Clear-sky index", height=110,
            )
        )  # fmt: skip
    ok = check_g12(windows=windows) and len(windows) == 2
    return (
        ({"g12_raised_sky": stack(charts=charts, title="G12: the raised sky")} if charts else {}),
        ok,
        f"windows {len(windows)}",
    )


# --------------------------------------------------------------------------- G13: the target stack


def check_g13(*, windows: Sequence[Mapping[str, object]]) -> bool:
    """Check the target stack of each world type.

    The solar target is zero in a world with no solar farm and above zero on some sun-up half-hour
    (and never at night) in a solar world; every window holds a transfer edge, the step size is
    non-zero exactly where the event is 1, and the hidden span is shaded.
    """
    if len(windows) < 4:
        return False
    for w in windows:
        solar, cosz = np.asarray(w["solar"]), np.asarray(w["cosz"])
        event, size = np.asarray(w["event"]), np.asarray(w["size"])
        if (
            not event.any()
            or ((size != 0) != (event > 0)).any()
            or not np.asarray(w["hidden"]).any()
        ):
            return False
        if (solar[cosz <= core.NIGHT_COSZ] > TOL).any():
            return False
        if w["world"] in ov.SOLAR_WORLDS:
            if not (solar > 0).any():
                return False
        elif np.abs(solar).max() > TOL:
            return False
    return True


def panel_g13(
    *, overlays: Sequence[ov.TransferOverlay], originals: Sequence[sim729.Segment]
) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:
    """Draw G13: the solar target and the step events of one window of each world type.

    The panel also draws the step sizes and shades the hidden span of style R.
    """
    windows: list[dict[str, object]] = []
    charts: list[alt.TopLevelMixin] = []
    for world in core729.WORLD_TYPES:
        wins = choose_windows(
            overlays=[o for o in overlays if o.seg.world == world], originals=[originals[i] for i, o in enumerate(overlays) if o.seg.world == world],
            keep=lambda o, tr: True, n=1,
        )  # fmt: skip
        if not wins:
            continue
        w = wins[0]
        hidden = masks.hidden_positions(
            style=core.STYLE,
            rng=masks.fixed_rng(style=core.STYLE, index=len(windows)),
            utc_offset=0.0,
        )
        hidden = hidden[: w.n] if hidden.size >= w.n else np.pad(hidden, (0, w.n - hidden.size))
        windows.append(
            {
                "world": world,
                "solar": w.solar,
                "cosz": w.cosz,
                "event": w.event,
                "size": w.size,
                "hidden": hidden,
            }
        )
        charts.append(
            lines_chart(
                series={"solar target": w.solar, "step event (x 10)": 10.0 * w.event, "step size": w.size},
                title=f"World type {world}: the solar target, the step events, and the step sizes (grey: hidden span of style R)",
                hidden=hidden, rules=[p / 2.0 for p in w.edges], height=110,
            )
        )  # fmt: skip
    ok = check_g13(windows=windows)
    return (
        ({"g13_targets": stack(charts=charts, title="G13: the target stack")} if charts else {}),
        ok,
        f"worlds drawn {len(windows)}",
    )


# --------------------------------------------------------------------------- G14: the drawn transfer parameters


def duration_hours(*, transfer: ov.Transfer) -> float:
    """Return a transfer's duration in hours (the grid has 12 ticks an hour)."""
    return transfer.ticks / 12.0


def duration_band(*, hours: float) -> int:
    """Return the index of a transfer's duration band (1-3 h, 3-12 h, 12 h-2 d, 2-4 d)."""
    return int(np.digitize(hours, core.DURATION_BANDS_HOURS[1:-1]))


def settled_table(
    *, overlays_by_name: Mapping[str, Sequence[ov.TransferOverlay]]
) -> list[dict[str, object]]:
    """Return the settled T0 (the mean absolute size after the donor rule, in points) by direction.

    Partner, and duration band, with counts.
    """
    cells: dict[tuple[str, str, int], list[float]] = {}
    for overlays in overlays_by_name.values():
        for o in overlays:
            for tr in o.transfers:
                key = (
                    "D8 gains load" if tr.size > 0 else "D8 loses load",
                    tr.partner,
                    duration_band(hours=duration_hours(transfer=tr)),
                )
                cells.setdefault(key, []).append(abs(tr.size))
    return [
        {"direction": d, "partner": p, "duration": DURATION_BAND_LABELS[b], "mean_t0": float(np.mean(v)), "n": len(v)}
        for (d, p, b), v in sorted(cells.items())
    ]  # fmt: skip


def parameter_arrays(*, overlays: Sequence[ov.TransferOverlay]) -> dict[str, np.ndarray]:
    """Return the drawn parameters of a bank: the transfers per 30 days of each segment.

    The durations in hours, the sizes in points, and whether the partner sits under BSP A.
    """
    rate = np.array([len(o.transfers) * 30.0 / o.seg.n_dense for o in overlays])
    drawn = np.array([o.rate for o in overlays])
    trs = [tr for o in overlays for tr in o.transfers]
    return {
        "per_30_days": rate, "drawn_rate": drawn,
        "n_transfers": np.array([len(o.transfers) for o in overlays], dtype=float),
        "n_expected": np.array([round(o.rate * o.seg.n_dense / 30.0) for o in overlays], dtype=float),
        "hours": np.array([duration_hours(transfer=t) for t in trs]),
        "points": np.array([abs(t.size) for t in trs]),
        "under_bsp_a": np.array([t.inside_bsp_a for t in trs], dtype=float),
        "gains": np.array([t.size > 0 for t in trs], dtype=float),
    }  # fmt: skip


def check_g14(
    *, params: Mapping[str, Mapping[str, np.ndarray]], table: Sequence[Mapping[str, object]]
) -> bool:
    """Check that every histogram lies inside its registered range.

    The check also requires that the settled table splits both directions and at least two partners.

    The transfers per 30 days of a segment never exceed the drawn rate by more than rounding (and the
    drawn rate lies in its range: 0 to 6 in training, 6 in the test overlay), each duration lies in 1
    to 96 hours, each size in 3 to 40 points, and the share of transfers under BSP A lies within
    `PARTNER_SHARE_TOLERANCE` of 0.8.
    """
    if not params or not table:
        return False
    for name, a in params.items():
        if a["points"].size < 20 or a["per_30_days"].size == 0:
            return False
        lo, hi = (
            core.TRANSFER_RATE_FIXED if name == "test" else core.TRANSFER_RATE_TRAIN[0],
            core.TRANSFER_RATE_FIXED if name == "test" else core.TRANSFER_RATE_TRAIN[1],
        )
        if not ((a["drawn_rate"] >= lo - 1e-9) & (a["drawn_rate"] <= hi + 1e-9)).all():
            return False
        if (a["n_transfers"] > a["n_expected"]).any():
            return False
        if not (
            (a["hours"] >= core.TRANSFER_HOURS[0] - 1e-9)
            & (a["hours"] <= core.TRANSFER_HOURS[1] + 1e-9)
        ).all():
            return False
        if not (
            (a["points"] >= core.TRANSFER_POINTS[0] - 1e-6)
            & (a["points"] <= core.TRANSFER_POINTS[1] + 1e-6)
        ).all():
            return False
        want = core.PARTNER_MIX["read"] + core.PARTNER_MIX[core.UNREAD_PARTNER]
        if abs(float(a["under_bsp_a"].mean()) - want) > PARTNER_SHARE_TOLERANCE:
            return False
    return {str(r["direction"]) for r in table} == {"D8 gains load", "D8 loses load"} and len(
        {str(r["partner"]) for r in table}
    ) >= 2


def histogram_chart(
    *,
    values: Mapping[str, np.ndarray],
    bounds: tuple[float, float],
    title: str,
    x_title: str,
    log: bool = False,
) -> alt.LayerChart:
    """Return a share histogram of one parameter for each overlay, with the registered range as two rules."""
    rows = []
    lo, hi = bounds
    pooled = np.concatenate([v for v in values.values() if v.size])
    edges = (
        np.geomspace(max(min(pooled.min(), lo), 1e-3), max(pooled.max(), hi), 25)
        if log
        else np.linspace(min(pooled.min(), lo), max(pooled.max(), hi) + 1e-9, 25)
    )
    for name, v in values.items():
        h, _ = np.histogram(v, bins=edges)
        share = h / max(h.sum(), 1)
        rows += [
            {
                "overlay": name,
                "a": float(edges[k]),
                "b": float(edges[k + 1]),
                "share": float(share[k]),
            }
            for k in range(len(h))
        ]
    x = alt.X("a:Q", title=x_title, scale=alt.Scale(type="log" if log else "linear"))
    bars = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar(opacity=0.6, aria=False)
        .encode(
            x=x,
            x2="b:Q",
            y=alt.Y("share:Q", title="Share"),
            color=alt.Color(
                "overlay:N",
                scale=alt.Scale(range=[BLUE, ORANGE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )  # ty: ignore[unresolved-attribute]
    rules = (
        alt.Chart(pl.DataFrame({"v": [lo, hi]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#000000", strokeDash=[3, 3], aria=False)
        .encode(x="v:Q")
    )  # ty: ignore[unresolved-attribute]
    return (bars + rules).properties(
        width=WIDTH_PX, height=110, title=alt.TitleParams(title, anchor="start")
    )


def table_chart(*, table: Sequence[Mapping[str, object]]) -> alt.LayerChart:
    """Return the settled T0 by direction and partner (rows) and duration band (columns).

    As a printed heat map.
    """
    data = pl.DataFrame(
        [
            {
                "row": f"{r['direction']}, {r['partner']}",
                "duration": r["duration"],
                "t0": float(r["mean_t0"]),  # ty: ignore[invalid-argument-type]
                "n": int(r["n"]),  # ty: ignore[invalid-argument-type]
            }
            for r in table
        ]  # ty: ignore[invalid-argument-type]
    ).with_columns(
        label=pl.col("t0").round(1).cast(pl.Utf8)
        + pl.lit(" (")
        + pl.col("n").cast(pl.Utf8)
        + pl.lit(")")
    )
    base = alt.Chart(data).encode(
        x=alt.X("duration:N", sort=list(DURATION_BAND_LABELS), title="Duration band"),
        y=alt.Y("row:N", title=None),
    )  # ty: ignore[unresolved-attribute]
    rect = base.mark_rect(aria=False).encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color("t0:Q", title="Settled T0 (points)", scale=alt.Scale(scheme="blues"))
    )
    text = base.mark_text(aria=False, fontSize=9).encode(text="label:N")  # ty: ignore[unresolved-attribute]
    return (rect + text).properties(
        width=WIDTH_PX,
        height=22 * max(len({r["row"] for r in data.to_dicts()}), 1),
        title=alt.TitleParams(
            "Settled T0 in points after the donor rule (mean, and the number of transfers in brackets)",
            anchor="start",
        ),
    )


def panel_g14(
    *, overlays_by_name: Mapping[str, Sequence[ov.TransferOverlay]]
) -> tuple[dict[str, alt.TopLevelMixin], bool, str, list[dict[str, object]]]:
    """Draw G14: the drawn parameters of seed 0's training overlay and of the test overlay.

    The panel also returns the settled T0 table.
    """
    params = {name: parameter_arrays(overlays=o) for name, o in overlays_by_name.items()}
    table = settled_table(overlays_by_name=overlays_by_name)
    ok = check_g14(params=params, table=table)
    ranges = {
        "drawn_rate": (core.TRANSFER_RATE_TRAIN[0], core.TRANSFER_RATE_TRAIN[1]),
        "hours": core.TRANSFER_HOURS, "points": core.TRANSFER_POINTS,
    }  # fmt: skip
    charts = [
        histogram_chart(values={n: p["drawn_rate"] for n, p in params.items()}, bounds=ranges["drawn_rate"], title="Drawn transfer rate of each segment (rules: the registered range)", x_title="Transfers per 30 days"),
        histogram_chart(values={n: p["hours"] for n, p in params.items()}, bounds=ranges["hours"], title="Duration of a transfer", x_title="Hours", log=True),
        histogram_chart(values={n: p["points"] for n, p in params.items()}, bounds=ranges["points"], title="Settled size T0 after the donor rule", x_title="Points of capacity", log=True),
        histogram_chart(values={n: p["under_bsp_a"] for n, p in params.items()}, bounds=(0.0, 1.0), title="Partner under BSP A (1) or under another BSP (0); registered share 0.8", x_title="Under BSP A"),
    ]  # fmt: skip
    if table:
        charts.append(table_chart(table=table))
    notes = {n: int(p["points"].size) for n, p in params.items()}
    return (
        {"g14_parameters": stack(charts=charts, title="G14: the drawn transfer parameters")},
        ok,
        f"transfers drawn {notes}; table cells {len(table)}",
        table,
    )


# --------------------------------------------------------------------------- G15: everything at once


def check_g15(*, windows: Sequence[Mapping[str, np.ndarray]]) -> bool:
    """Check that each window holds an engine run, solar output, and a load transfer.

    Each window also holds an extra injection. The truth must be base demand less engine and solar,
    and the injected member must differ from the plain member.
    """
    if not windows:
        return False
    for w in windows:
        if not all(np.abs(w[k]).max() > 0 for k in ("engine", "solar", "t", "delta")):
            return False
        if not np.allclose(w["x"], w["base"] - w["engine"] - w["solar"], atol=TOL):
            return False
        if not np.allclose(w["x_inj"], w["x"] - w["delta"], atol=TOL):
            return False
        if (
            not (np.abs(w["r_inj"] - w["r"]) > 0).any()
            or not (np.abs(w["bsp_inj"] - w["bsp"]) > 0).any()
        ):
            return False
    return True


def panel_g15(
    *,
    overlays: Sequence[ov.TransferOverlay],
    originals: Sequence[sim729.Segment],
    index: ov.PairIndex,
) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:
    """Draw G15: windows that carry an engine run, S2 solar output, and a load transfer.

    Each window also carries an extra injection.
    """
    windows: list[dict[str, np.ndarray]] = []
    charts: list[alt.TopLevelMixin] = []
    for i, o in enumerate(overlays):
        if len(windows) == N_DRAWS:
            break
        seg = o.seg
        if seg.world != "both" or seg.source != "S2":
            continue
        for tr in o.transfers:
            d = int(min(max(tr.start_half_hour // 48, 1), seg.x.shape[0] - 3))
            sibs = [
                int(j) for j in index.siblings.get((i, d), []) if originals[int(j)].source == "S2"
            ]
            cols = window_cols(day=d)
            if not sibs or cols.start < 0 or cols.stop > seg.x.size or seg.cal_index[d] < 0:
                continue

            def flat(a: np.ndarray, cols: slice = cols) -> np.ndarray:
                return np.asarray(a).reshape(-1)[cols].astype(np.float64)

            if (
                not all(np.abs(flat(a)).max() > 0 for a in (seg.engine, seg.solar, o.t))
                or not np.isfinite(flat(seg.bsp)).all()
            ):
                continue
            delta = ov.window_delta(index=index, sibling=sibs[0], amount=20.0, day=d)[1:].astype(
                np.float64
            )
            if not (delta > 0).any():
                continue
            x, bsp = flat(seg.x), flat(seg.bsp)
            x_inj, r_inj, bsp_inj = ov.inject(x=x, q=flat(o.q), delta=delta, bsp=bsp, beta=o.beta)
            w = {
                "x": x, "base": flat(seg.base), "engine": flat(seg.engine), "solar": flat(seg.solar),
                "t": flat(o.t), "delta": delta, "r": flat(seg.r), "bsp": np.nan_to_num(bsp),
                "x_inj": x_inj.astype(np.float64), "r_inj": r_inj.astype(np.float64),
                "bsp_inj": np.nan_to_num(bsp_inj.astype(np.float64)),
            }  # fmt: skip
            windows.append(w)
            charts += [
                lines_chart(series={"base demand": w["base"], "engine output": w["engine"], "solar output": w["solar"], "transferred load": w["t"], "extra injection": delta}, title=f"Draw {len(windows)}: the components", height=100),
                lines_chart(series={"truth w": x, "truth w+": w["x_inj"], "reading w": np.nan_to_num(w["r"]), "reading w+": w["r_inj"]}, widths={"truth w": 0.8, "reading w": 0.8, "truth w+": 2.2, "reading w+": 2.2}, title="The window w and its injected member w+", height=100),
                lines_chart(series={"BSP w": w["bsp"], "BSP w+": w["bsp_inj"]}, widths={"BSP w": 0.8, "BSP w+": 2.2}, title="The BSP meter", height=70),
            ]  # fmt: skip
            break
    ok = check_g15(windows=windows)
    return (
        (
            {"g15_everything": stack(charts=charts, title="G15: every component at once")}
            if charts
            else {}
        ),
        ok,
        f"windows {len(windows)}",
    )


# --------------------------------------------------------------------------- G16: pairing coverage


def check_g16(*, paired: Mapping[str, Sequence[float]]) -> bool:
    """Check that every training seed has months with at least 95% of windows paired.

    Every month's share of windows with a sibling pair must reach 95%.
    """
    return bool(paired) and all(
        len(v) > 0 and min(v) >= 1.0 - core.MAX_UNPAIRED_SHARE for v in paired.values()
    )


def panel_g16(*, summary: Mapping[str, object]) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:
    """Draw G16 from the unpaired shares that `overlays/summary.json` recorded.

    The panel has one line for each training seed and month position.
    """
    banks = summary["banks"]
    paired = {
        name: [1.0 - float(u) for u in rec["share_unpaired_by_month_position"]]  # ty: ignore[invalid-argument-type, not-subscriptable]
        for name, rec in banks.items()  # ty: ignore[unresolved-attribute]
        if name.startswith("train_s")
    }
    rows = [
        {"seed": n.removeprefix("train_s"), "month": k + 1, "share": v}
        for n, vs in paired.items()
        for k, v in enumerate(vs)
    ]
    if not rows:
        return {}, False, "no training bank in the summary"
    bars = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, aria=False)
        .encode(
            x=alt.X("month:Q", title="Month, in time order", axis=alt.Axis(format="d")),
            y=alt.Y(
                "share:Q",
                title="Share of windows with a sibling pair",
                scale=alt.Scale(domain=[0.8, 1.0]),
            ),
            color=alt.Color(
                "seed:N",
                title="Training seed",
                scale=alt.Scale(range=list(COLOURS[:5])),
                legend=alt.Legend(orient="top"),
            ),
        )
    )  # ty: ignore[unresolved-attribute]
    bound = (
        alt.Chart(pl.DataFrame({"y": [1.0 - core.MAX_UNPAIRED_SHARE]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#000000", strokeDash=[3, 3], aria=False)
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = (bars + bound).properties(
        width=WIDTH_PX,
        height=160,
        title=alt.TitleParams(
            "G16: windows with a sibling pair, by training seed and month (rule: the bound of 95%)",
            anchor="start",
        ),
    )
    return {"g16_pairing": chart}, check_g16(paired=paired), f"seeds {len(paired)}"


# --------------------------------------------------------------------------- G18: D8 and the neighbours


def mirror_change(*, w: Win, name: str) -> tuple[float, float]:
    """Return a neighbour's mean change over the transfer's half-hours and the mean transferred load there."""
    inside = (np.abs(w.t) > 0) & np.isfinite(w.nb_before[name]) & np.isfinite(w.nb_after[name])
    if not inside.any():
        return float("nan"), float("nan")
    return float(np.mean((w.nb_after[name] - w.nb_before[name])[inside])), float(
        np.mean(w.t[inside])
    )


def check_g18(*, wins: Mapping[str, Sequence[Win]]) -> bool:
    """Check the mirrors of the four G18 panels.

    A read partner moves by minus `j` times the transferred load (within `MIRROR_TOLERANCE_POINTS`),
    and the other two neighbours do not move; where the partner is not read (or sits under another
    BSP), no neighbour moves. A panel with no draw fails.
    """
    for case, group in wins.items():
        if not group:
            return False
        for w in group:
            tr = w.transfer
            if tr is None:
                return False
            for name in nb.NEIGHBOURS:
                moved = np.nan_to_num(w.nb_after[name] - w.nb_before[name])
                if name == case and tr.mirrored:
                    change, load = mirror_change(w=w, name=name)
                    if (
                        not abs(change + tr.jitter * load) <= core.MIRROR_TOLERANCE_POINTS
                        or abs(change) < TOL
                    ):
                        return False
                elif np.abs(moved).max() > TOL:
                    return False
            if case not in nb.NEIGHBOURS and tr.mirrored:
                return False
    return True


def g18_charts(
    *, wins: Sequence[Win], medians: Mapping[str, float], title: str
) -> alt.VConcatChart:
    """Return a panel: D8, the three neighbours before the fold, and the folded channels.

    Each without (thin) and with (thick) the overlay.
    """
    charts: list[alt.TopLevelMixin] = []
    for k, w in enumerate(wins):
        rules = [p / 2.0 for p in w.edges]
        charts.append(
            lines_chart(
                series={
                    "truth without": w.x0,
                    "truth with": w.x1,
                    "reading without": np.nan_to_num(w.r0),
                    "reading with": np.nan_to_num(w.r1),
                },
                widths={
                    "truth without": 0.8,
                    "reading without": 0.8,
                    "truth with": 2.2,
                    "reading with": 2.2,
                },
                title=f"Draw {k + 1}: D8 (dotted: transfer edges)",
                rules=rules,
                height=90,
            )
        )
        for label, fold in (("before the fold, signed", False), ("folded", True)):
            series: dict[str, np.ndarray] = {}
            widths: dict[str, float] = {}
            for name in nb.NEIGHBOURS:
                for tag, arr, width in (
                    ("without", w.nb_before[name], 0.8),
                    ("with", w.nb_after[name], 2.2),
                ):
                    values = nb.fold(signed=arr, floor=w.floors.get(name, 0.0)) if fold else arr
                    series[f"{name} {tag}"] = np.nan_to_num(values / medians[name])
                    widths[f"{name} {tag}"] = width
            charts.append(
                lines_chart(
                    series=series,
                    widths=widths,
                    title=f"The neighbours {label}, as multiples of each one's own median",
                    rules=rules,
                    y_title="Multiple of the neighbour's median",
                    height=100,
                )
            )
    return stack(charts=charts, title=title)


def panel_g18(
    *,
    overlays: Sequence[ov.TransferOverlay],
    originals: Sequence[sim729.Segment],
    medians: Mapping[str, float],
) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:
    """Draw G18: a transfer whose partner is D3, D5, or D14.

    The panel also draws a transfer whose partner the neural network does not read.
    """
    cases = {
        "D3": "D3",
        "D5": "D5",
        "D14": "D14",
        core.UNREAD_PARTNER: "A partner that the neural network does not read",
    }
    groups: dict[str, list[Win]] = {}
    for key in cases:
        groups[key] = choose_windows(
            overlays=overlays,
            originals=originals,
            keep=lambda o, tr, key=key: (  # ty: ignore[invalid-argument-type]
                tr.partner == key
                and np.isfinite(
                    o.seg.x[max(tr.start_half_hour // 48, 1) : max(tr.start_half_hour // 48, 1) + 2]
                ).all()
                and (
                    key not in nb.NEIGHBOURS
                    or np.isfinite(
                        o.neighbours_real[key][
                            max(tr.start_half_hour // 48, 1) : max(tr.start_half_hour // 48, 1) + 2
                        ]
                    ).all()
                )
            ),
        )
    charts = {
        f"g18_{'partner_' + k if k in nb.NEIGHBOURS else 'unread'}": g18_charts(
            wins=v, medians=medians, title=f"G18: transfer with partner {cases[k]}"
        )
        for k, v in groups.items()
        if v
    }
    return charts, check_g18(wins=groups), f"draws {[len(v) for v in groups.values()]}"  # ty: ignore[invalid-return-type]


# --------------------------------------------------------------------------- G19: the three transfer shapes


def shape_windows(
    *, overlays: Sequence[ov.TransferOverlay], originals: Sequence[sim729.Segment], hours: float = 20.0, t0: float = 15.0,
) -> dict[str, Win] | None:  # fmt: skip
    """Build a constant, a demand-shaped, and a solar-shaped transfer of the same T0.

    All three transfers sit on one segment (a D3 partner, D8 gaining the load).

    The first publishable S2 solar-world segment whose three shapes hold the floor is used. The
    profiles come from
    `overlays.shape_profile` as the generator builds them.
    """
    for i, o in enumerate(overlays):
        seg = originals[i]
        if (
            not publishable(seg=seg)
            or seg.world not in ov.SOLAR_WORLDS
            or o.neighbours_real["D3"].size == 0
        ):
            continue
        for d in range(2, seg.x.shape[0] - 4):
            if (seg.cal_index[d : d + 3] < 0).any() or not np.isfinite(
                o.neighbours_real["D3"][d : d + 3]
            ).all():
                continue
            if np.abs(seg.solar[d : d + 3]).max() <= 0:
                continue
            start_tick = (d * 48 + 14) * core.TICKS_PER_HALF_HOUR + 2
            ticks = int(hours * 12)
            probe = ov.Transfer(start_tick=start_tick, ticks=ticks, size=t0, partner="D3")
            lo, hi = probe.start_half_hour, probe.end_half_hour + 1
            sun = (seg.csi.reshape(-1) * seg.clear.reshape(-1))[lo:hi]
            donor = seg.base.reshape(-1)[lo:hi]
            built = {}
            for shape in core.SHAPES:
                profile = ov.shape_profile(
                    probe=probe, shape=shape, donor=donor, sun=sun, kappa=0.8
                )
                if profile is None:
                    break
                tr = ov.Transfer(
                    start_tick=start_tick,
                    ticks=ticks,
                    size=t0,
                    partner="D3",
                    shape=shape,
                    profile=profile,
                )
                built[shape] = ov.apply_transfers(
                    seg=seg,
                    transfers=[tr],
                    neighbours=dict(o.neighbours_real),
                    floors=o.floors,
                )
                built[shape] = (built[shape], tr)  # ty: ignore[invalid-assignment]
            if len(built) == len(core.SHAPES):
                return {
                    s: window_of(overlay=ov_, original=seg, index=i, day0=d, n_days=3, transfer=tr)  # ty: ignore[not-iterable]
                    for s, (ov_, tr) in built.items()
                }
    return None


def check_g19(*, shapes: Mapping[str, Win] | None) -> bool:
    """Check the three shapes: the same mean T0, never below 0.1 T0, and different from one another.

    The check also requires an unchanged solar target.
    """
    if not shapes or set(shapes) != set(core.SHAPES):
        return False
    t0 = shapes["constant"].transfer.size  # ty: ignore[unresolved-attribute]
    spread = {}
    for name, w in shapes.items():
        tr = w.transfer
        assert tr is not None
        if abs(float(w.t.sum()) / (tr.ticks / core.TICKS_PER_HALF_HOUR) - t0) > TOL * abs(t0):
            return False
        full = tr.used_weights() == core.TICKS_PER_HALF_HOUR
        levels = w.t[w.rel(half_hour=tr.start_half_hour) : w.rel(half_hour=tr.end_half_hour) + 1]
        if (levels[full] < core.SHAPE_FLOOR * t0 - 1e-9).any():
            return False
        spread[name] = float(np.std(levels[full]))
    constant_solar = shapes["constant"].solar
    if not all(np.array_equal(w.solar, constant_solar) for w in shapes.values()):
        return False
    return (
        spread["constant"] < TOL
        and spread["demand"] > 0.01 * abs(t0)
        and spread["solar"] > 0.01 * abs(t0)
    )


def panel_g19(
    *, overlays: Sequence[ov.TransferOverlay], originals: Sequence[sim729.Segment]
) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:
    """Draw G19: the three transfer shapes with the same T0."""
    shapes = shape_windows(overlays=overlays, originals=originals)
    ok = check_g19(shapes=shapes)
    if not shapes:
        return {}, False, "no segment holds all three shapes"
    charts = []
    for name, w in shapes.items():
        t0 = w.transfer.size  # ty: ignore[unresolved-attribute]
        charts.append(
            lines_chart(
                series={
                    "transferred load T": w.t,
                    "T0": np.full(w.n, t0),
                    "D8 truth": w.x1,
                    "partner D3 change": np.nan_to_num(w.nb_after["D3"] - w.nb_before["D3"]),
                    "solar target": w.solar,
                },
                title=f"A {name} transfer of T0 = {t0:g} points: the solar target is the same in all three",
                n_days=3,
                rules=[p / 2.0 for p in w.edges],
                height=120,
            )
        )
    return (
        {"g19_shapes": stack(charts=charts, title="G19: the three transfer shapes")},
        ok,
        "shapes drawn 3",
    )


# --------------------------------------------------------------------------- G20: the neighbour table and the noise units


def monthly_coverage(
    *, base_dates: np.ndarray, reading: np.ndarray, arrays: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return each neighbour's coverage of D8's present half-hours in each calendar month, in time order."""
    months = base_dates.astype("datetime64[M]")
    out: dict[str, list[float]] = {n: [] for n in nb.NEIGHBOURS}
    for m in np.unique(months):
        rows = months == m
        if not np.isfinite(reading[rows]).any():
            continue
        shares = nb.coverage(
            table={n: arrays[n][rows] for n in nb.NEIGHBOURS}, reading=reading[rows]
        )
        for n in nb.NEIGHBOURS:
            out[n].append(shares[n])
    return {n: np.array(v) for n, v in out.items()}


def mirror_points(*, overlays: Sequence[ov.TransferOverlay]) -> list[dict[str, object]]:
    """Return, for every mirrored transfer, the partner's mean change over the transfer.

    Minus the transfer's mean load, and the largest half-hour gap to the mirror that the overlay
    should hold.

    The gap compares the partner's change in each half-hour with minus the summed `j` times the load of
    every transfer that mirrors onto that partner, so a transfer that shares an edge half-hour with another
    transfer is compared with both loads. Half-hours where the partner holds no reading are left out.
    """
    rows = []
    for o in overlays:
        shape = o.neighbours_real["D3"].shape
        expected = {n: np.zeros(shape[0] * shape[1]) for n in core.READ_PARTNERS}
        for tr in o.transfers:
            if tr.mirrored:
                expected[tr.partner][tr.start_half_hour : tr.end_half_hour + 1] += (
                    tr.jitter * tr.mean_levels()
                )
        for tr in o.transfers:
            if not tr.mirrored:
                continue
            span = slice(tr.start_half_hour, tr.end_half_hour + 1)
            diff = (
                o.neighbours[tr.partner].reshape(-1) - o.neighbours_real[tr.partner].reshape(-1)
            )[span]
            level = tr.mean_levels()
            seen = np.isfinite(diff)
            if not seen.any():
                continue
            rows.append({
                "partner": tr.partner, "jittered": tr.jitter != 1.0, "j": float(tr.jitter),
                "change": float(np.mean(diff[seen])), "minus_load": -float(np.mean(level[seen])),
                "max_gap": float(np.max(np.abs(diff[seen] + expected[tr.partner][span][seen]))),
            })  # fmt: skip
    return rows  # ty: ignore[invalid-return-type]


def check_g20(
    *,
    overall: Mapping[str, float],
    noise_sizes: Mapping[str, np.ndarray],
    points: Sequence[Mapping[str, object]],
) -> bool:
    """Check the neighbour table and the drawn sizes.

    Every neighbour covers at least 90% of D8's present half-hours over the whole table; the drawn
    sizes in D8's noise units of the test overlay reach both sides of 4; and every mirror point lies
    on the line of equality `change = -j load` within `MIRROR_TOLERANCE_POINTS`, with at least 10 points.
    """
    if set(overall) != set(nb.NEIGHBOURS) or any(
        not v >= nb.MIN_COVERAGE for v in overall.values()
    ):
        return False
    test = noise_sizes.get("test")
    if (
        test is None
        or test.size == 0
        or not ((test < SIZE_NOISE_LINE).any() and (test >= SIZE_NOISE_LINE).any())
    ):
        return False
    return len(points) >= 10 and all(
        float(p["max_gap"]) <= core.MIRROR_TOLERANCE_POINTS  # ty: ignore[invalid-argument-type]
        for p in points
    )


def panel_g20(
    *, base_dates: np.ndarray, reading: np.ndarray, arrays: Mapping[str, np.ndarray], overlays_by_name: Mapping[str, Sequence[ov.TransferOverlay]], noise_d8: float,
) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:  # fmt: skip
    """Draw G20: the coverage per month index, the drawn sizes in D8's noise units, and the mirror check's scatter."""
    coverage = monthly_coverage(base_dates=base_dates, reading=reading, arrays=arrays)
    overall = nb.coverage(table=arrays, reading=reading)
    sizes = {
        n: np.array([abs(t.size) / noise_d8 for o in os_ for t in o.transfers])
        for n, os_ in overlays_by_name.items()
    }
    points = [
        {**p, "overlay": n}
        for n, os_ in overlays_by_name.items()
        for p in mirror_points(overlays=os_)
    ]
    cov_rows = [
        {"neighbour": n, "month": k + 1, "share": float(v)}
        for n, vs in coverage.items()
        for k, v in enumerate(vs)
    ]
    lowest = min(float(r["share"]) for r in cov_rows)  # ty: ignore[invalid-argument-type]
    coverage_floor = min(0.5, float(np.floor(lowest * 10.0) / 10.0))
    cov = (
        alt.Chart(pl.DataFrame(cov_rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, aria=False)
        .encode(
            x=alt.X("month:Q", title="Month, in time order", axis=alt.Axis(format="d")),
            y=alt.Y(
                "share:Q",
                title="Coverage of D8's present half-hours",
                scale=alt.Scale(domain=[coverage_floor, 1.0]),
            ),
            color=alt.Color(
                "neighbour:N",
                scale=alt.Scale(domain=list(nb.NEIGHBOURS), range=[BLUE, ORANGE, GREEN]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"y": [nb.MIN_COVERAGE]}))
        .mark_rule(color="#000000", strokeDash=[3, 3], aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH_PX,
        height=130,
        title=alt.TitleParams(
            "Coverage of each neighbour by month, information only (rule: the bound of 90% applies to the whole table)",
            anchor="start",
        ),
    )
    hist = histogram_chart(
        values={n: v for n, v in sizes.items() if v.size},
        bounds=(SIZE_NOISE_LINE, SIZE_NOISE_LINE),
        title="Drawn transfer sizes in D8's noise units (rule: 4 noise units)",
        x_title="Size in D8's noise units",
        log=True,
    )
    scatter_rows = [
        {
            "change": p["change"],
            "minus_load": p["minus_load"],
            "kind": "jittered mirror" if p["jittered"] else "exact mirror",
        }
        for p in points
    ]
    lim = max([abs(float(r["minus_load"])) for r in scatter_rows] + [1.0])  # ty: ignore[invalid-argument-type]
    diag = (
        alt.Chart(pl.DataFrame({"a": [-lim, lim]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#000000", strokeDash=[3, 3], aria=False)
        .encode(x="a:Q", y="a:Q")
    )  # ty: ignore[unresolved-attribute]
    dots = (
        alt.Chart(pl.DataFrame(scatter_rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, opacity=0.6, aria=False)
        .encode(
            x=alt.X("minus_load:Q", title="Minus the mean transferred load (points)"),
            y=alt.Y("change:Q", title="Mean change of the partner (points)"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["exact mirror", "jittered mirror"], range=[BLUE, ORANGE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )  # ty: ignore[unresolved-attribute]
    scatter = (
        (diag + dots).properties(
            width=WIDTH_PX,
            height=200,
            title=alt.TitleParams(
                "The mirror check: a jittered mirror lies on the line change = -j x load, so only j = 1 sits on the diagonal",
                anchor="start",
            ),
        )
        if scatter_rows
        else None
    )
    charts = [cov, hist] + ([scatter] if scatter is not None else [])
    ok = check_g20(overall=overall, noise_sizes=sizes, points=points) if scatter_rows else False
    return (
        {
            "g20_neighbour_table": stack(
                charts=charts, title="G20: the neighbour table and the noise units"
            )
        },
        ok,
        f"mirror points {len(points)}; overall coverage "
        + ", ".join(f"{n} {v:.3f}" for n, v in overall.items()),
    )


# --------------------------------------------------------------------------- G21: one built input tensor


@dataclass(frozen=True)
class Tensor:
    """One built training triple and its variants.

    Attributes:
        plain: The plain window of U* `(C, 96)`.
        injected: The injected member.
        raised: The raised-sky member.
        hidden: The hidden positions `(96,)`.
        delta: The extra injection over the window's 97 half-hours.
        edges: The positions of the transfer edges in the window.
        partner: The transfer's partner.
        no_neighbours: The same window as built for U-N `(N_BASE, 96)`.
        no_bsp: The same window as built for U-noBSP `(C, 96)`.
    """

    plain: np.ndarray
    injected: np.ndarray
    raised: np.ndarray
    hidden: np.ndarray
    delta: np.ndarray
    edges: list[int]
    partner: str
    no_neighbours: np.ndarray
    no_bsp: np.ndarray


def bsp_step_outside(*, x: np.ndarray, hidden: np.ndarray) -> bool:
    """Return whether a window shows the BSP meter: a non-zero step with its mask set, outside the hidden span."""
    step = np.abs(x[arms.ROW["d_bsp"]]) > 0.0
    present = x[arms.ROW["bsp_mask"]] > 0.5
    return bool((step & present & ~hidden).any())


def build_tensor(
    *, prepared: Sequence[arms.Prepared], index: ov.PairIndex, rng_seed: int = 0
) -> Tensor | None:
    """Build one training triple of U* with a mirrored edge, an extra injection, and a hidden span.

    The function builds the same window for U-N and U-noBSP.

    The window comes from an engine-only or no-injection segment (never a metered farm) that has a
    BSP meter (only 1 segment in 7 does), with a sibling of source S2, an edge of a transfer that a
    read neighbour mirrors outside the hidden span, and a BSP step outside the hidden span.
    """
    full = arms.eval_spec(inject=True, raised=True)
    for p in prepared:
        if p.world in ov.SOLAR_WORLDS or not p.has_bsp:
            continue
        days = {int(x) for x in p.days}
        for tr in p.transfers:
            if not tr.mirrored:
                continue
            for hh in (tr.start_half_hour, tr.end_half_hour):
                d = hh // 48
                if d not in days or d < 1:
                    continue
                sib = ov.fixed_sibling(index=index, segment=p.source_index, day=d)
                if sib is None:
                    continue
                delta = ov.window_delta(index=index, sibling=sib, amount=20.0, day=d)
                if not (delta[1 + arms.MID : 1 + arms.MID + 48] > 0).any():
                    continue
                start = d * 48 - arms.MID
                edges = [
                    e
                    for e in (tr.start_half_hour - start, tr.end_half_hour - start)
                    if 0 <= e < arms.STEPS
                ]
                hidden = None
                for k in range(50):
                    cand = masks.hidden_positions(
                        style=core.STYLE,
                        rng=np.random.default_rng([rng_seed, k]),
                        utc_offset=float(p.offsets[d]),
                    )
                    if cand.any() and not any(cand[e] for e in edges):
                        hidden = cand
                        break
                if hidden is None or not edges:
                    continue
                trip = arms.build_triple(
                    p=p, day=d, spec=full, hidden=hidden, delta=delta, eps=core.TEST_EPS
                )
                un = arms.build_triple(
                    p=p,
                    day=d,
                    spec=arms.eval_spec(inject=True, raised=True, neighbours=False),
                    hidden=hidden,
                    delta=delta,
                    eps=core.TEST_EPS,
                )
                nobsp = arms.build_triple(
                    p=p,
                    day=d,
                    spec=arms.eval_spec(inject=True, raised=True, bsp_zeroed=True),
                    hidden=hidden,
                    delta=delta,
                    eps=core.TEST_EPS,
                )
                if (
                    trip.xi is None
                    or trip.xc is None
                    or not bsp_step_outside(x=trip.x, hidden=hidden)
                ):
                    continue
                return Tensor(
                    trip.x, trip.xi, trip.xc, hidden, delta, edges, tr.partner, un.x, nobsp.x
                )
    return None


def check_g21(*, t: Tensor | None) -> bool:
    """Check one built tensor against the written list.

    Every neighbour channel of every member is zero inside the hidden span (and so are the reading and
    its mask); the partner's change channel is non-zero at an edge outside the span; the injected
    member differs from the plain one only in the channels the plan allows and on half-hours with an
    injection (the reading), and the raised member only in the clear-sky channel, above it on every
    raisable half-hour; and U-N and U-noBSP differ from U* only in the neighbour and the BSP channels.
    """
    if t is None or not t.hidden.any():
        return False
    h = t.hidden
    nbr = list(leak.NEIGHBOUR_ROWS)
    for member in (t.plain, t.injected, t.raised):
        if (
            np.abs(member[nbr][:, h]).max() > 0.0
            or np.abs(member[[arms.ROW["r"], arms.ROW["r_mask"]]][:, h]).max() > 0.0
        ):
            return False
    partner_change = t.plain[arms.NB_ROW[f"{t.partner}_change"]]
    if not t.edges or not any(abs(partner_change[e]) > 0.0 for e in t.edges):
        return False
    changed = np.flatnonzero(np.abs(t.injected - t.plain).max(axis=1) > 0.0)
    if not set(changed) <= set(leak.INJECTION_MAY_CHANGE) or arms.ROW["r"] not in changed:
        return False
    on = t.delta[1:] > 0.0
    r_diff = np.abs(t.injected[arms.ROW["r"]] - t.plain[arms.ROW["r"]]) > 0.0
    if r_diff[~on].any() or not r_diff[on & ~h].any():
        return False
    raised_rows = np.flatnonzero(np.abs(t.raised - t.plain).max(axis=1) > 0.0)
    if set(raised_rows) != {arms.ROW["csi"]}:
        return False
    raisable = ov.raisable(csi=t.plain[arms.ROW["csi"]], cosz=t.plain[arms.ROW["cosz"]])
    if (
        raisable.any()
        and not (t.raised[arms.ROW["csi"]][raisable] > t.plain[arms.ROW["csi"]][raisable]).all()
    ):
        return False
    if not np.array_equal(t.no_neighbours, t.plain[: arms.N_BASE]):
        return False
    bsp_rows = [arms.ROW["d_bsp"], arms.ROW["bsp_mask"]]
    if not bsp_step_outside(x=t.plain, hidden=h):
        return False
    rest = np.setdiff1d(np.arange(arms.N_CHANNELS), bsp_rows)
    return bool(np.array_equal(t.no_bsp[rest], t.plain[rest]) and not t.no_bsp[bsp_rows].any())


def variant_chart(
    *, t: Tensor, label: str, window: np.ndarray, row: int, rules: Sequence[float]
) -> alt.LayerChart | alt.Chart:
    """Return one channel of a variant window drawn over the same channel of U*."""
    short = label.split(":", maxsplit=1)[0]
    return lines_chart(
        series={"U*": t.plain[row], short: window[row]},
        widths={"U*": 2.2, short: 1.0},
        title=f"{label}: {arms.ALL_NAMES[row]}",
        rules=rules, hidden=t.hidden, y_title=arms.ALL_NAMES[row], height=44,
    )  # fmt: skip


def tensor_charts(*, t: Tensor) -> dict[str, alt.TopLevelMixin]:
    """Return the two G21 figures: every channel of w, w+, and w^c as small panels.

    The second figure shows the same window as built for U-N and U-noBSP.
    """
    shown = {"w": t.plain, "w+": t.injected, "w^c": t.raised}
    rules = [e / 2.0 for e in t.edges]
    marks = np.flatnonzero(t.delta[1:] > 0.0)
    panels = []
    for row, name in enumerate(arms.ALL_NAMES):
        panels.append(
            lines_chart(
                series={k: m[row] for k, m in shown.items()},
                widths={"w": 2.4, "w+": 1.4, "w^c": 0.8},
                title=name,
                rules=rules,
                hidden=t.hidden,
                marks=marks,
                y_title=name,
                height=44,
            )
        )
    subtitle = (
        "The extra injection comes from a sibling of source S2 (a published farm, never a metered one).",
        "The window comes from a segment with a BSP meter, so the BSP step is not flat.",
    )
    first = stack(
        charts=panels,
        title="G21: every input channel of one training window w, its injected member w+, and its raised-sky member w^c (grey: hidden span; dotted: transfer edges; ticks: half-hours with an injection)",
        subtitle=subtitle,
    )
    variants = []
    absent = np.zeros((arms.N_CHANNELS - t.no_neighbours.shape[0], t.no_neighbours.shape[1]))
    for label, window in (
        (
            "U-N: the neighbour channels are absent (drawn as zero)",
            np.concatenate([t.no_neighbours, absent]),
        ),
        ("U-noBSP: the BSP step and its mask are zero", t.no_bsp),
    ):
        rows = (
            arms.ROW["r"],
            arms.ROW["d_bsp"],
            arms.ROW["bsp_mask"],
            *(arms.NB_ROW[f"{n}_change"] for n in nb.NEIGHBOURS),
        )
        variants.extend(
            variant_chart(t=t, label=label, window=window, row=row, rules=rules) for row in rows
        )
    return {
        "g21_tensor": first,
        "g21_variants": stack(
            charts=variants,
            title="G21: the same window as built for U-N and U-noBSP against U*",
            subtitle=subtitle,
        ),
    }


def assert_siblings_are_published(
    *, index: ov.PairIndex, segments: Sequence[sim729.Segment]
) -> None:
    """Check that every sibling that can lend G21 its extra injection is a published (S2) farm.

    Args:
        index: The pair index G21 draws its injected member from.
        segments: The segments the index was built from.

    Raises:
        CorrectnessFailureError: When a sibling has a source outside `core.TEST_SIBLING_SOURCES`,
            or the index and the segments do not match (a metered farm could reach a published file).
    """
    if len(index.months) != len(segments):
        msg = "G21's pair index was not built from the segments it is checked against"
        raise leak.CorrectnessFailureError(msg)
    bad = sum(
        1
        for sibs in index.siblings.values()
        for j in sibs
        if segments[int(j)].source not in core.TEST_SIBLING_SOURCES
    )
    if bad:
        msg = f"{bad} G21 siblings are not of a published source: a metered farm could reach a published file"
        raise leak.CorrectnessFailureError(msg)


def panel_g21(
    *,
    prepared: Sequence[arms.Prepared],
    index: ov.PairIndex,
    segments: Sequence[sim729.Segment],
) -> tuple[dict[str, alt.TopLevelMixin], bool, str]:
    """Draw G21 from the training bank of seed 0.

    G21 is published, so its injected member may take its extra output from a sibling of a published
    source only.

    Args:
        prepared: The prepared training segments.
        index: The pair index (built with `core.TEST_SIBLING_SOURCES`).
        segments: The segments of the bank the index was built from.

    Returns:
        The charts, whether the check passes, and a counts-only note.
    """
    assert_siblings_are_published(index=index, segments=segments)
    t = build_tensor(prepared=prepared, index=index)
    return (
        (tensor_charts(t=t) if t is not None else {}),
        check_g21(t=t),
        "tensor built" if t is not None else "no window met the conditions",
    )


# --------------------------------------------------------------------------- the stage


@dataclass
class Panels:
    """What the gallery stage draws.

    Attributes:
        public: The published charts by file stem.
        private: The private charts (G17 only) by file stem.
        results: Whether each panel of `core.PANEL_REQUIREMENTS` passes its check.
        notes: A counts-only note per panel.
        tables: The printed tables (the settled T0 of G14).
    """

    public: dict[str, alt.TopLevelMixin] = field(default_factory=dict)
    private: dict[str, alt.TopLevelMixin] = field(default_factory=dict)
    results: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    tables: dict[str, object] = field(default_factory=dict)


def check_private_only(*, public: Mapping[str, object], private: Mapping[str, object]) -> bool:
    """Check that G17 exists and sits in the private set only (no S1 pair in a published file)."""
    return (
        bool(private)
        and not any(k.startswith("g17") for k in public)
        and all(k.startswith("g17") for k in private)
    )


def build_panels(
    *, test_overlays: Sequence[ov.TransferOverlay], test_segments: Sequence[sim729.Segment], train_overlays: Sequence[ov.TransferOverlay],
    train_prepared: Sequence[arms.Prepared], train_index: ov.PairIndex, test_index: ov.PairIndex, test_index_both: ov.PairIndex, overlay_summary: Mapping[str, object],
    base_dates: np.ndarray, reading: np.ndarray, arrays: Mapping[str, np.ndarray], medians: Mapping[str, float], noise_d8: float,
) -> Panels:  # fmt: skip
    """Draw every panel G10 to G21 and run its check."""
    out = Panels()

    def add(
        key: str,
        charts: Mapping[str, alt.TopLevelMixin],
        ok: bool,
        note: str,
        private: bool = False,
    ) -> None:
        (out.private if private else out.public).update(charts)
        out.results[key] = bool(ok)
        out.notes[key] = note

    add("G10_transfers", *panel_g10(overlays=test_overlays, originals=test_segments))
    add("G11_pairs", *panel_g11(segments=test_segments, index=test_index, public=True, source="S2"))
    add("G12_raised_sky", *panel_g12(segments=test_segments))
    add("G13_targets", *panel_g13(overlays=test_overlays, originals=test_segments))
    charts, ok, note, table = panel_g14(
        overlays_by_name={"train seed 0": train_overlays, "test": test_overlays}
    )
    add("G14_parameters", charts, ok, note)
    out.tables["g14_settled_t0"] = table
    add(
        "G15_everything",
        *panel_g15(overlays=test_overlays, originals=test_segments, index=test_index),
    )
    add("G16_pairing", *panel_g16(summary=overlay_summary))
    s1_charts, s1_ok, s1_note = panel_g11(
        segments=test_segments, index=test_index_both, public=False, source="S1"
    )
    add(
        "G17_s1_pairs",
        s1_charts,
        s1_ok and check_private_only(public=out.public, private=s1_charts),
        s1_note,
        private=True,
    )
    add(
        "G18_neighbours",
        *panel_g18(overlays=test_overlays, originals=test_segments, medians=medians),
    )
    add("G19_shapes", *panel_g19(overlays=test_overlays, originals=test_segments))
    add(
        "G20_neighbour_table",
        *panel_g20(
            base_dates=base_dates,
            reading=reading,
            arrays=arrays,
            overlays_by_name={"train seed 0": train_overlays, "test": test_overlays},
            noise_d8=noise_d8,
        ),
    )
    add(
        "G21_input_tensor",
        *panel_g21(
            prepared=train_prepared, index=train_index, segments=[o.seg for o in train_overlays]
        ),
    )
    return out


def save_panels(*, panels: Panels, public_dir: Path, private_dir: Path) -> list[Path]:
    """Write the published SVGs to `public_dir` and the private ones to `private_dir`."""
    written = []
    for folder, charts in ((public_dir, panels.public), (private_dir, panels.private)):
        for name, chart in charts.items():
            path = folder / f"gallery_{name}.svg"
            figures729.save_svg(chart=chart, path=path)
            written.append(path)
    return written


def auto_check_record(*, panels: Panels, banks_digest: str) -> dict:
    """Return the content of `auto_check.json`: one flag per required panel and the notes.

    The record also holds the id of the banks in use.
    """
    flags = {name: bool(panels.results.get(name, False)) for name in core.PANEL_REQUIREMENTS}
    return {
        "panels": flags, "notes": dict(panels.notes), "requirements": dict(core.PANEL_REQUIREMENTS),
        "all_passed": all(flags.values()), "banks_digest": banks_digest,
    }  # fmt: skip


def stage_gallery(
    *, root: Path, root730: Path, env: stages.Env, base: stages.BaseDays, log: Callable[[str], None]
) -> dict[str, bool]:
    """Draw the gallery G10 to G21 on the CPU from the banks, neighbours, and overlays.

    The function also writes `auto_check.json`.

    Args:
        root: The data folder of this experiment.
        root730: The data folder of experiment 730 (read only).
        env: The stages' `Env` (the neighbour table, the medians, the standardisation constants).
        base: The stages' `BaseDays`.
        log: Receives progress lines.

    Returns:
        Whether each panel passes its automatic check. The stage never writes `panel_check.json`.
    """
    import e00729_banks as banks729

    stages.guard_run(root=root, root730=root730, stage="gallery")
    summary = stages.read_json(path=root / "overlays" / "summary.json")
    te = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    tr = banks729.load_bank(path=banks730.bank_path(root=root730, kind="train", seed=0))
    arrays = env.neighbours  # ty: ignore[unresolved-attribute]
    test_overlays = ov.overlay_segments(segments=te.segments, kind="test", neighbours=arrays)
    train_overlays = ov.overlay_segments(
        segments=tr.segments, kind="train", seed=0, neighbours=arrays
    )
    prepared = arms.prepare(
        overlays=train_overlays,
        holidays=env.holidays,
        stats=env.stats,
        medians=env.medians,  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    limits = stages.read_json(path=root / "limits.json")
    panels = build_panels(
        test_overlays=test_overlays, test_segments=te.segments, train_overlays=train_overlays, train_prepared=prepared,
        train_index=ov.build_pair_index(segments=tr.segments, sources=core.TEST_SIBLING_SOURCES), test_index=ov.build_pair_index(segments=te.segments, sources=core.TEST_SIBLING_SOURCES),
        test_index_both=ov.build_pair_index(segments=te.segments), overlay_summary=summary, base_dates=base.dates, reading=base.reading,  # ty: ignore[unresolved-attribute]
        arrays=arrays, medians=env.medians, noise_d8=float(limits["noise_units"]["D8"]),  # ty: ignore[unresolved-attribute]
    )  # fmt: skip
    private = gallery_private_dir(root=root)
    save_panels(panels=panels, public_dir=root / "figures" / "gallery", private_dir=private)
    stages.write_json(
        path=private / AUTO_FILE,
        payload=auto_check_record(
            panels=panels, banks_digest=banks730.banks_digest_id(root=root730)
        ),
    )
    stages.write_json(
        path=private / "g14_settled_t0.json", payload={"cells": panels.tables["g14_settled_t0"]}
    )  # ty: ignore[invalid-argument-type]
    log(
        f"gallery drawn: {sum(panels.results.values())} of {len(panels.results)} panels pass the check"
    )
    return panels.results


def record_panel_check(*, root: Path, root730: Path, confirms: Mapping[str, bool]) -> dict:
    """Write `panel_check.json`: a panel passes when its automatic check passes.

    The reader's confirmation must also say that the panel passes.

    Args:
        root: The data folder.
        root730: Experiment 730's data folder (for the banks' id).
        confirms: For each panel, whether the reader checked it against the written list and found
            that it shows what the list requires.

    Returns:
        The record written.
    """
    auto = stages.read_json(path=gallery_private_dir(root=root) / AUTO_FILE)
    digest = banks730.banks_digest_id(root=root730)
    if auto.get("banks_digest") != digest:
        msg = "auto_check.json belongs to other banks: run `gallery` on the banks now in use"
        raise RuntimeError(msg)
    results = {
        k: bool(auto["panels"].get(k)) and bool(confirms.get(k)) for k in core.PANEL_REQUIREMENTS
    }
    record = {
        "panels": results, "requirements": dict(core.PANEL_REQUIREMENTS),
        "notes": {"confirmed_by": "the session that runs the experiment"},
        "all_passed": all(results.values()), "banks_digest": digest,
    }  # fmt: skip
    stages.write_json(path=gallery_private_dir(root=root) / PANEL_FILE, payload=record)
    return record

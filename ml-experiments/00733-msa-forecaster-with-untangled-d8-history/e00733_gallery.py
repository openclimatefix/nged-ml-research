# ruff: noqa: E501
"""Experiment 733: the data, the automatic checks.

And the charts of the new gallery panels G10 to G15.

Experiment 730 drew G1 to G9 from the banks this experiment reads. Each new panel is a function that
returns the plotted arrays and the result of its written list (`core.GALLERY_PANELS`), so a panel
that
shows a silent component fails before any fit: `panel_check` writes
`PRIVATE/gallery/panel_check.json`,
which `e00733_stages.guard_run` reads. The checks count every threshold over the half-hours where
the
quantity exists (the solar channel over sun-up half-hours only). Panels that carry a single real S1
window
go to `PRIVATE/gallery/` only.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_paths as paths  # noqa: F401
import e00733_synth as sy
import numpy as np

SUN_UP_SHORTWAVE: Final = 10.0
"""The shortwave (in the units of the weather column) above which a half-hour counts as sun-up."""
MOST: Final = 0.5
"""'Most' means more than this share of the half-hours the check counts."""
CORRELATION_FLOOR: Final = 0.0
ZERO_TOLERANCE: Final = 1e-6


@dataclass(frozen=True)
class PanelResult:
    """What one panel shows and whether its written list holds.

    Attributes:
        panel: The panel name (G10 to G15).
        passed: Whether every item of the written list holds.
        items: The result of each item of the list, by short name.
        data: The plotted arrays.
    """

    panel: str
    passed: bool
    items: Mapping[str, bool]
    data: Mapping[str, Any]


def _result(*, panel: str, items: Mapping[str, bool], data: Mapping[str, Any]) -> PanelResult:
    return PanelResult(
        panel=panel, passed=bool(items) and all(items.values()), items=dict(items), data=data
    )


def pick_origin(*, seg: sy.SegmentArrays) -> int:
    """Return the first origin day with at least 7 history rows, a covered ensemble run.

    And the most history rows (ties to the earliest).
    """
    days = np.flatnonzero(seg.present & seg.covered)
    if not len(days):
        msg = "the segment has no origin on a covered base day"
        raise sy.SynthError(msg)
    n = sy.history_row_exists(seg=seg, origin_day=days).sum(axis=1)
    return int(days[int(np.argmax(n))])


def g10_layout(*, seg: sy.SegmentArrays, origin_day: int) -> PanelResult:
    """G10: the history rows and target rows of one origin on the segment's axis of days.

    With the edges.

    The check requires that every history row and every target row of lead days 1 to 14 that lies
    inside the
    segment is listed, none outside it, and that the scored lead days are 2 to 10.
    """
    d = np.array([origin_day])
    hist_slots = sy.history_slots(origin_day=d)[0]
    hist_days = hist_slots[:, 0] // sy.STEPS
    hist_exists = sy.history_row_exists(seg=seg, origin_day=d)[0]
    tgt = [
        int(sy.target_slots(origin_day=d, lead_index=np.array([k]))[0, 0] // sy.STEPS)
        for k in range(sy.N_LEAD)
    ]
    n_days = seg.n_days
    tgt_inside = [0 <= t < n_days and bool(seg.present[t]) for t in tgt]
    hist_inside = [0 <= int(h) < n_days and bool(seg.present[int(h)]) for h in hist_days]
    items = {
        "history_rows_match_the_segment": hist_inside == hist_exists.tolist(),
        "some_history_row_shown": bool(hist_exists.any()),
        "target_rows_inside_listed": sum(tgt_inside)
        == len([t for t in tgt if 0 <= t < n_days and seg.present[t]]),
        "scored_lead_days_are_2_to_10": tuple(k + 1 for k in sy.SCORED_K0)
        == core.SYNTHETIC_LEAD_DAYS,
    }
    return _result(
        panel="G10", items=items,
        data={"n_days": n_days, "present": seg.present.copy(), "origin_day": origin_day, "history_days": hist_days, "history_exists": hist_exists, "target_days": np.array(tgt), "target_inside": np.array(tgt_inside), "scored_lead_days": np.array(core.SYNTHETIC_LEAD_DAYS)},
    )  # fmt: skip


def g11_target_days(
    *,
    segments: Sequence[sy.SegmentArrays],
    origin_days: Sequence[int],
    engine: Sequence[np.ndarray] | None = None,
) -> PanelResult:
    """G11: the folded reading, the signed truth, the injected solar output, and.

    (when given) the injected engine over 10 target days of three segments.

    The check requires reverse flow on target days in at least two of the three segments and a
    nonzero injected solar output in at least one, so a silent component fails the panel.
    """
    reverse, windows, solar_nonzero = [], [], []
    for i, (seg, d) in enumerate(zip(segments, origin_days, strict=True)):
        t = sy.target_slots(origin_day=np.array([d]), lead_index=np.arange(1, 10))[:, :].reshape(-1)
        windows.append(
            {
                "reading": seg.reading_z[t],
                "x_neg": seg.x_neg[t],
                "solar": seg.solar_points[t],
                "engine": None if engine is None else np.asarray(engine[i])[t],
            }
        )
        reverse.append(bool(np.nansum(seg.x_neg[t]) > 0))
        solar_nonzero.append(bool(np.nanmax(np.abs(seg.solar_points[t])) > 0))
    items = {
        "reverse_flow_in_at_least_two_segments": sum(reverse) >= 2,
        "injected_solar_nonzero_somewhere": any(solar_nonzero),
        "engine_nonzero_when_given": engine is None
        or any(bool(np.nanmax(np.abs(w["engine"])) > 0) for w in windows),
    }
    return _result(panel="G11", items=items, data={"windows": windows, "reverse": reverse})


def g12_weather_correlation(
    *, segments: Sequence[sy.SegmentArrays], ens_col: int, cams_col: int
) -> PanelResult:
    """G12.

    The correlation by lead day between the ENS shortwave column of the target row and the observed
    sky column of the same half-hours.

    The check requires a correlation above zero on every lead day and a correlation at lead day 10
    below the one at lead day 1.
    """
    ens: list[list[np.ndarray]] = [[] for _ in range(sy.N_LEAD)]
    obs: list[list[np.ndarray]] = [[] for _ in range(sy.N_LEAD)]
    for seg in segments:
        days = np.flatnonzero(seg.present & seg.covered)
        for k in range(sy.N_LEAD):
            t = sy.target_slots(origin_day=days, lead_index=np.full(len(days), k))
            ok = (
                np.array([(s.min() >= 0) and (s.max() < seg.n_days * sy.STEPS) for s in t])
                if len(days)
                else np.zeros(0, bool)
            )
            if not ok.any():
                continue
            ens[k].append(seg.top_weather[days[ok], k, :, ens_col].ravel())
            obs[k].append(seg.hist_weather[t[ok], cams_col].ravel())
    corr = np.full(sy.N_LEAD, np.nan)
    for k in range(sy.N_LEAD):
        if ens[k]:
            a, b = np.concatenate(ens[k]), np.concatenate(obs[k])
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0:
                corr[k] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    scored = corr[list(sy.SCORED_K0)]
    items = {
        "correlation_above_zero_on_scored_lead_days": bool(
            np.isfinite(scored).all() and (scored > CORRELATION_FLOOR).all()
        ),
        "correlation_falls_with_lead_day": bool(
            np.isfinite(scored).all() and scored[-1] < scored[0]
        ),
    }
    return _result(
        panel="G12",
        items=items,
        data={"lead_day": np.arange(1, sy.N_LEAD + 1), "correlation": corr},
    )


def g13_channels_against_truth(
    *,
    segments: Sequence[sy.SegmentArrays],
    shortwave_col: int,
    min_reverse_points: float,
    reading_points: Sequence[np.ndarray],
) -> PanelResult:
    """G13: the untangled P(X<0) against the indicator of X < 0.

    And the untangled solar output against the injected solar output.

    The check counts the solar channel over sun-up half-hours of segments that inject solar output
    (a nonzero
    channel there), and the probability over the reverse-flow half-hours whose reading is at least
    `min_reverse_points`.

    Args:
        segments: The test segments drawn.
        shortwave_col: The index of the shortwave column of `hist_weather`.
        min_reverse_points: 1 MVA in points of D8's capacity.
        reading_points: Each segment's folded reading in points `(n_days * 48,)`.

    Returns:
        The panel result.
    """
    nonzero_share, p_above, n_rev = [], [], 0
    for seg, rp in zip(segments, reading_points, strict=True):
        sun = (
            (seg.hist_weather[:, shortwave_col] > SUN_UP_SHORTWAVE)
            & seg.u_ok
            & (seg.solar_points > 0)
        )
        if sun.any():
            nonzero_share.append(
                float(
                    np.mean(
                        np.abs(seg.u_raw[sun, core.CHANNEL_NAMES.index("solar")]) > ZERO_TOLERANCE
                    )
                )
            )
        rev = (seg.x_neg > 0.5) & seg.u_ok & (np.asarray(rp) >= min_reverse_points)
        n_rev += int(rev.sum())
        if rev.any():
            p_above.append(float(np.mean(seg.u_raw[rev, core.P_NEG] > 0.5)))
    items = {
        "solar_channel_nonzero_on_sun_up_half_hours": bool(nonzero_share)
        and float(np.mean(nonzero_share)) > MOST,
        "p_neg_above_half_on_most_reverse_half_hours": bool(p_above)
        and float(np.mean(p_above)) > MOST,
    }
    return _result(
        panel="G13",
        items=items,
        data={
            "solar_nonzero_share": nonzero_share,
            "p_neg_above_half_share": p_above,
            "n_reverse_half_hours": n_rev,
        },
    )


def g14_rerun_difference(*, segments: Sequence[sy.SegmentArrays]) -> PanelResult:
    """G14: the stored output minus the re-run output of days D - 2 and D - 1.

    By hour of the day after the visibility limit.

    The check requires a difference of zero on every half-hour before the limit and at least one
    origin with a re-run.
    """
    diffs, after = [], []
    for seg in segments:
        if seg.u_rerun is None or seg.u_rerun_ok is None:
            continue
        for d in np.flatnonzero(seg.present & seg.u_rerun_ok.any(axis=(1, 2))):
            for r in (0, 1):
                day = int(d) - 2 + r
                if day < 0:
                    continue
                stored = seg.u_raw[day * sy.STEPS : (day + 1) * sy.STEPS, core.P_NEG]
                diffs.append(
                    np.where(
                        seg.u_rerun_ok[d, r], seg.u_rerun[d, r, :, core.P_NEG] - stored, np.nan
                    )
                )
                # stamp of half-hour j of day D - 2 + r is 48 (D - 2 + r) + j + 1; the limit stamp is 48 D + 24 - 54
                after.append(
                    sy.STEPS * (r - 2)
                    + np.arange(sy.STEPS)
                    + 1
                    + core.NEWEST_VISIBLE_BEFORE_CONTRACT
                    - 24
                )
    if not diffs:
        return _result(panel="G14", items={"some_rerun_exists": False}, data={})
    stack, hours = np.stack(diffs), np.stack(after) / 2.0
    before = hours <= 0
    zero_before = bool(
        not np.isfinite(stack[before]).any() or np.nanmax(np.abs(stack[before])) <= ZERO_TOLERANCE
    )
    items = {
        "some_rerun_exists": True,
        "difference_zero_before_the_limit": zero_before,
        "some_half_hour_after_the_limit": bool((~before).any()),
    }
    return _result(
        panel="G14", items=items, data={"differences": stack, "hours_after_limit": hours}
    )


def g15_base_rate(
    *, segments: Sequence[sy.SegmentArrays], clim: np.ndarray, shortwave_col: int
) -> PanelResult:
    """G15: the share of reverse flow in the target rows by lead day and by half-hour of the day.

    With B-clim beside it.

    The check requires a share above zero on sun-up half-hours.
    """
    num = np.zeros((sy.N_LEAD, sy.STEPS))
    den = np.zeros((sy.N_LEAD, sy.STEPS))
    sun_hh = np.zeros(sy.STEPS, dtype=bool)
    for seg in segments:
        days = np.flatnonzero(seg.present & seg.covered)
        for k in range(sy.N_LEAD):
            t = sy.target_slots(origin_day=days, lead_index=np.full(len(days), k))
            inside = (t.min(axis=1) >= 0) & (t.max(axis=1) < seg.n_days * sy.STEPS)
            if not inside.any():
                continue
            x = seg.x_neg[t[inside]]
            ok = np.isfinite(x)
            num[k] += np.where(ok, x, 0.0).sum(axis=0)
            den[k] += ok.sum(axis=0)
        sun_hh |= (
            seg.hist_weather[:, shortwave_col].reshape(seg.n_days, sy.STEPS) > SUN_UP_SHORTWAVE
        ).mean(axis=0) > MOST
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(den > 0, num / den, np.nan)
    items = {
        "share_above_zero_on_sun_up_half_hours": bool(
            sun_hh.any() and np.nanmax(share[:, sun_hh]) > 0
        ),
        "b_clim_given": bool(np.asarray(clim).size > 0),
    }
    return _result(
        panel="G15", items=items, data={"share": share, "clim": np.asarray(clim), "sun_up": sun_hh}
    )


def panel_check(*, results: Sequence[PanelResult], banks_digest: str) -> dict[str, Any]:
    """Return the content of `panel_check.json`: each panel's pass, its items.

    And the banks' digest id.

    A panel of `core.GALLERY_PANELS` that is not among the results is recorded as failed.
    """
    got = {r.panel: r for r in results}
    panels = {
        name: bool(got[name].passed) if name in got else False for name in core.GALLERY_PANELS
    }
    return {
        "banks_digest": banks_digest,
        "panels": panels,
        "items": {n: dict(got[n].items) for n in got},
        "written_lists": {n: list(v) for n, v in core.GALLERY_PANELS.items()},
    }


def write_panel_check(*, root: Path, payload: Mapping[str, Any]) -> Path:
    """Write `PRIVATE/gallery/panel_check.json` atomically."""
    path = root / "PRIVATE" / "gallery" / "panel_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    tmp.rename(path)
    return path


def draw(*, result: PanelResult, path: Path) -> Path:
    """Draw one panel as SVG with Altair (a heat map of the share for G15.

    Lines for the rest) and return the path.

    A panel that failed its list is still drawn, so the person who runs the experiment can see why.
    """
    import altair as alt
    import e00733_figures as figs
    import polars as pl

    d = result.data
    if result.panel == "G15":
        share = np.asarray(d["share"])
        rows = pl.DataFrame(
            {
                "lead_day": np.repeat(np.arange(1, share.shape[0] + 1), share.shape[1]),
                "half_hour": np.tile(np.arange(share.shape[1]), share.shape[0]),
                "share": share.ravel(),
            }
        ).drop_nans()
        chart = (
            alt.Chart(rows)  # ty: ignore[unresolved-attribute]
            .mark_rect()
            .encode(
                x="half_hour:O",
                y="lead_day:O",
                color=alt.Color("share:Q", scale=alt.Scale(scheme="blues")),
            )
        )  # ty: ignore[unresolved-attribute]
    elif result.panel == "G12":
        rows = pl.DataFrame(
            {"lead_day": d["lead_day"], "correlation": d["correlation"]}
        ).drop_nans()
        chart = (
            alt.Chart(rows)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True, color=figs.figures729.BLUE)
            .encode(x="lead_day:Q", y=alt.Y("correlation:Q", scale=alt.Scale(domain=[0, 1])))
        )  # ty: ignore[unresolved-attribute]
    else:
        items = pl.DataFrame(
            {"item": list(result.items), "passed": [float(v) for v in result.items.values()]}
        )
        chart = (
            alt.Chart(items)  # ty: ignore[unresolved-attribute]
            .mark_bar()
            .encode(
                y=alt.Y("item:N", title=None),
                x=alt.X(
                    "passed:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="Written-list item holds (1) or fails (0)",
                ),
            )
        )  # ty: ignore[unresolved-attribute]
    figs.save_svg(
        chart=chart.properties(
            title=alt.TitleParams(
                f"{result.panel}: {'pass' if result.passed else 'FAIL'}", anchor="start"
            ),
            width=figs.WIDTH_PX,
        ),
        path=path,
    )
    return path

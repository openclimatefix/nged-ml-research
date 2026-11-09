"""Decode a world (real or injected) and score the decoded events against the truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, Panel, block_means
from e00653_inject import Truth
from e00653_segment import Event, Params, classify, segment_primary
from e00653_slow import fit_slow, noise_unit, to_half_hours

ARMS: tuple[str, ...] = ("masked", "huber", "rw30")


@dataclass
class Frozen:
    """Noise units and slow levels fitted once on the uninjected residual."""

    sigma: dict[str, float]
    sigma3: dict[str, float]
    slow: dict[str, dict[str, np.ndarray]]


def three_day_unit(*, y: np.ndarray) -> float:
    """Robust spread of the 3-day-against-3-day step of daily means (the design answer's unit)."""
    d = np.nanmean(y[: len(y) // 4 * 4].reshape(-1, 4), axis=1)
    k = 3
    steps = np.array([np.nanmean(d[i : i + k]) - np.nanmean(d[i - k : i]) for i in range(k, len(d) - k)])
    steps = steps[np.isfinite(steps)]
    return float(1.4826 * np.median(np.abs(steps - np.median(steps))))


def freeze(*, panel: Panel, residual: dict[str, np.ndarray], upto: int | None = None) -> Frozen:
    """Fit the slow levels and fix the noise units on the uninjected residual (before `upto`)."""
    sigma, sigma3, slow = {}, {}, {}
    for s in panel.sites:
        x = residual[s] if upto is None else np.where(np.arange(len(residual[s])) < upto, residual[s], np.nan)
        y = block_means(x=x)
        if np.isfinite(y).sum() < 1000:
            sigma[s], sigma3[s], slow[s] = float("nan"), float("nan"), {}
            continue
        base = noise_unit(y=y)
        sl = fit_slow(y=y, sigma=base, arm="masked")
        r = y - sl
        sigma[s] = float(1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r))))
        sigma3[s] = three_day_unit(y=r)
        slow[s] = {"masked": sl}
        for arm in ("huber", "rw30"):
            slow[s][arm] = fit_slow(y=y, sigma=sigma[s], arm=arm)
    return Frozen(sigma, sigma3, slow)


def decode_world(*, panel: Panel, residual: dict[str, np.ndarray], frozen: Frozen, arm: str,
                 params: Params, edge_ok: set[tuple[str, str]] | None = None,
                 neighbours: dict[str, list[str]] | None = None,
                 refit_slow: bool = True) -> tuple[list[Event], dict[str, np.ndarray]]:
    """Fit the slow level (refitted per world), decode every Primary, and pair across neighbours."""
    events: list[Event] = []
    slows: dict[str, np.ndarray] = {}
    record_end = 0
    for s in panel.sites:
        ok = np.flatnonzero(np.isfinite(residual[s]))
        record_end = max(record_end, int(ok[-1]) + 1 if len(ok) else 0)
    for s in panel.sites:
        if not np.isfinite(frozen.sigma[s]):
            continue
        y = block_means(x=residual[s])
        sl = fit_slow(y=y, sigma=frozen.sigma[s], arm=arm) if refit_slow else frozen.slow[s][arm]
        slows[s] = sl
        xh = residual[s] - to_half_hours(slow=sl)
        events += segment_primary(name=s, y=y, xh=xh, slow=sl, sigma=frozen.sigma[s], p=params)
    classify(events=events, sigmas={k: (v if np.isfinite(v) else 1.0) for k, v in frozen.sigma.items()}, neighbours=neighbours or panel.neighbours,
             p=params, edge_ok=edge_ok, record_end=record_end)
    return events, slows


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match(*, truth: list[Truth], events: list[Event], start_tol: int = 24,
          world_end: int = GRID_LENGTH) -> list[dict]:
    """Score every truth event: found, class, timing error."""
    rows = []
    taken: set[int] = set()
    for t in sorted(truth, key=lambda x: -abs(x.amp_mw)):
        t_end = world_end if t.end is None else t.end
        best, best_iou = None, 0.0
        for e in events:
            if id(e) in taken or e.primary != t.site or np.sign(e.amp_mw) != np.sign(t.amp_mw):
                continue
            e_end = world_end if e.end is None else e.end
            iou = _iou((t.start, t_end), (e.start, e_end))
            if abs(e.start - t.start) <= start_tol and iou >= 0.5 and iou > best_iou:
                best, best_iou = e, iou
        if best is not None:
            taken.add(id(best))
        edge_truth = t.kind in ("edge", "permanent_transfer") and bool(t.partners)
        right_class = False
        if best is not None and edge_truth and best.cls in ("edge_transfer", "permanent_transfer"):
            if t.amp_mw < 0:
                total = sum(t.partner_amp.values()) or 1.0
                got = sum(v for k, v in t.partner_amp.items() if k in best.partners)
                right_class = got / total >= 0.5
            else:
                right_class = bool(set(t.partners) & set(best.partners))
        days = None if t.end is None else (t.end - t.start) / DAY
        rows.append({
            "site": t.site, "kind": t.kind, "group": t.group, "units": t.units,
            "amp_mw": t.amp_mw, "start": t.start, "end": -1 if t.end is None else t.end,
            "days": -1.0 if days is None else days, "edge_truth": edge_truth,
            "found": best is not None, "right_class": right_class,
            "decoded_cls": "none" if best is None else best.cls,
            "decoded_kind": "none" if best is None else best.kind,
            "start_err_hh": np.nan if best is None else float(best.start - t.start),
            "end_err_hh": np.nan if best is None or best.end is None or t.end is None
            else float(best.end - t.end),
        })
    return rows


def absorption(*, truth: list[Truth], slow_world: dict[str, np.ndarray],
               slow_base: dict[str, np.ndarray]) -> list[float]:
    """Share of each event's megawatts taken by the slow level (mean over the event, first 21 days)."""
    out = []
    for t in truth:
        b0 = t.start // 12
        b1 = (min(t.end if t.end is not None else GRID_LENGTH, t.start + 21 * DAY)) // 12
        if b1 <= b0 + 1 or t.site not in slow_world or t.kind in ("pv_growth", "seasonal_bias"):
            out.append(float("nan"))
            continue
        d = (slow_world[t.site][b0:b1] - slow_base[t.site][b0:b1]) * np.sign(t.amp_mw)
        out.append(float(np.nanmean(d) / abs(t.amp_mw)))
    return out


def events_frame(*, events: list[Event]) -> pl.DataFrame:
    """Decoded events as a table."""
    return pl.DataFrame([
        {"primary": e.primary, "start": e.start, "end": -1 if e.end is None else e.end,
         "amp_mw": e.amp_mw, "units": e.units, "confidence": e.confidence, "cls": e.cls,
         "kind": e.kind, "band": e.band, "partners": ",".join(e.partners),
         "ramp_fraction": e.ramp_fraction}
        for e in events
    ], schema={"primary": pl.Utf8, "start": pl.Int64, "end": pl.Int64, "amp_mw": pl.Float64,
               "units": pl.Float64, "confidence": pl.Float64, "cls": pl.Utf8, "kind": pl.Utf8,
               "band": pl.Utf8, "partners": pl.Utf8, "ramp_fraction": pl.Float64})

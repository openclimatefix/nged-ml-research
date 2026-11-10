"""Check 2: what are the false events that solar growth causes in experiment 653's masked arm?

    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 PYTHONPATH=<folders of 653 and 657>:. \
        uv run --no-sync python e00686_false_events.py

``<folders of 653 and 657>`` are the two experiment folders under ``ml-experiments/``, joined by a
colon.

Re-injects experiment 653's solar-growth world (three Primaries drawn per world, a growth in the
midday residual of 0.2 to 1.0 MW per year) with the same generator, the same tuned penalties, and
the same rule for a false event (a decoded event at an injected Primary, starting in 2024-01-01 or
later, that overlaps no event decoded on the un-injected record by 0.2 or more). Worlds 0 and 1 use
the seeds of experiment 653; the other worlds are new draws. Each false event gets the
measurements that decide its class.
"""

from __future__ import annotations

from datetime import timedelta
from multiprocessing import get_context
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, GRID_START, SCORED_FROM  # ty: ignore[unresolved-import]
from e00653_core import decode_world  # ty: ignore[unresolved-import]
from e00653_inject import GenParams, inject  # ty: ignore[unresolved-import]
from e00653_run import _overlap, load_params, setup  # ty: ignore[unresolved-import]
from e00653_slow import to_half_hours  # ty: ignore[unresolved-import]
from e00657_solar import haurwitz, load_irradiance  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00686-event-table-transfer-timing-and-false-solar-growth-events"
)
N_WORLDS: Final[int] = 24
ARM: Final[str] = "masked"
MIDDAY: Final[slice] = slice(20, 30)
"""Half-hours 10:00 to 14:30 UTC within a day."""
SUNNY: Final[float] = 0.7


def seed_of(*, k: int) -> int:
    """The seed experiment 653 used for world `k` of the solar-growth spec."""
    return 1000 + 17 * k + len("pv_growth")


def clearness_by_site() -> dict[str, np.ndarray]:
    """ERA5 shortwave over Haurwitz clear-sky irradiance per half-hour, for the 15 Primaries."""
    from e00653_common import SERIES_PATH  # ty: ignore[unresolved-import]

    series = pl.read_parquet(SERIES_PATH)
    ids = {r["site"]: int(r["time_series_id"]) for r in series.iter_rows(named=True)}
    sw, zen = load_irradiance(site_ids=ids)
    out = {}
    for s in sw:
        clear = haurwitz(zenith_deg=zen[s])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s] = np.where(clear > 20, sw[s] / clear, np.nan)
    return out


def season_boundary_distance_days(*, start: int) -> float:
    """Days from the event start to the nearest 1 April or 1 October."""
    when = GRID_START + timedelta(minutes=30 * start)
    best = 1e9
    for year in (when.year - 1, when.year, when.year + 1):
        for month in (4, 10):
            d = abs((when - GRID_START.replace(year=year, month=month, day=1)).total_seconds())
            best = min(best, d / 86400)
    return best


def midday_edge_width(*, xh: np.ndarray, start: int, sigma: float) -> tuple[float, bool]:
    """Days the midday level takes to go from 10% to 90% of its rise, and whether it completes.

    The series is the daily mean of the 10:00 to 14:30 UTC half-hours of `xh`, smoothed over 3
    days. The rise runs from the median of the 14 days before the start to the median of days 4
    to 10 after it. A rise under half a noise unit, or one whose 90% level is not reached within
    14 days, is reported as not complete.
    """
    days = xh[: len(xh) // DAY * DAY].reshape(-1, DAY)[:, MIDDAY]
    with np.errstate(invalid="ignore"):
        mid = np.nanmean(days, axis=1)
    d0 = start // DAY
    if d0 < 14 or d0 + 17 >= len(mid):
        return float("nan"), False
    pre = np.nanmedian(mid[d0 - 14 : d0])
    post = np.nanmedian(mid[d0 + 4 : d0 + 11])
    rise = post - pre
    if not np.isfinite(rise) or abs(rise) < 0.5 * sigma:
        return float("nan"), False
    seg = np.convolve(np.nan_to_num(mid[d0 - 3 : d0 + 18], nan=pre), np.ones(3) / 3, mode="same")
    frac = (seg - pre) / rise
    up10 = np.flatnonzero(frac > 0.1)
    up90 = np.flatnonzero(frac > 0.9)
    if len(up10) == 0 or len(up90) == 0:
        return float("nan"), False
    return float(max(0, up90[0] - up10[0])), True


def first_days_clearness(*, clearness: np.ndarray, start: int) -> tuple[float, float]:
    """Mean ERA5 clearness, and the share at 0.7 or more, over midday of the first 3 days."""
    idx = start + np.arange(3 * DAY)
    if idx[-1] >= GRID_LENGTH:
        return float("nan"), float("nan")
    mid = clearness[idx[(idx % DAY >= 20) & (idx % DAY < 30)]]
    if not np.isfinite(mid).any():
        return float("nan"), float("nan")
    return float(np.nanmean(mid)), float(np.nanmean(mid[np.isfinite(mid)] >= SUNNY))


def midday_share(*, x: np.ndarray, start: int) -> float:
    """Share of the change in daily profile (7 days after minus 7 days before) in 10:00 to 15:00."""
    if start < 7 * DAY or start + 7 * DAY > len(x):
        return float("nan")

    def profile(lo: int) -> np.ndarray:
        idx = lo + np.arange(7 * DAY)
        v = x[idx]
        ok = np.isfinite(v)
        tot = np.bincount(idx[ok] % DAY, weights=v[ok], minlength=DAY)
        cnt = np.bincount(idx[ok] % DAY, minlength=DAY)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    d = profile(start) - profile(start - 7 * DAY)
    total = float(np.nansum(d))
    return float(np.nansum(d[MIDDAY]) / total) if abs(total) > 1e-9 else float("nan")


def one_world(k: int) -> list[dict]:
    """Inject, decode the masked arm, and describe every false event."""
    panel, res, fz = setup()
    p = load_params()
    base_ev, base_slow = decode_world(panel=panel, residual=res, frozen=fz, arm=ARM, params=p)
    clearness = clearness_by_site()
    world, truth = inject(
        panel=panel,
        residual=res,
        sigma=fz.sigma,
        rng=np.random.default_rng(seed_of(k=k)),
        lo=SCORED_FROM,
        hi=GRID_LENGTH,
        gen=GenParams(),
        kinds=("pv_growth",),
    )
    ev, slows = decode_world(panel=panel, residual=world, frozen=fz, arm=ARM, params=p)
    sites = {t.site for t in truth}
    rows = [{"world": k, "kind": "world", "n_sites": len(sites), "primary": ""}]
    for e in ev:
        if e.primary not in sites or e.start < SCORED_FROM:
            continue
        if any(b.primary == e.primary and _overlap(b, e) >= 0.2 for b in base_ev):
            continue
        site = e.primary
        sigma = fz.sigma[site]
        xh = world[site] - to_half_hours(slow=slows[site])
        injected = world[site] - res[site]
        edge_days, edge_complete = midday_edge_width(xh=xh, start=e.start, sigma=sigma)
        # Accumulated injected growth over the first 30 days that the slow level has not absorbed.
        a, b = e.start, min(e.start + 30 * DAY, GRID_LENGTH)
        diff = to_half_hours(slow=slows[site] - base_slow[site])[a:b]
        misfit = (np.nanmean(injected[a:b]) - np.nanmean(diff)) / sigma
        mean_clear, sunny_share = first_days_clearness(clearness=clearness[site], start=e.start)
        share_mid = midday_share(x=world[site], start=e.start)
        rows.append(
            {
                "world": k,
                "kind": "false_event",
                "primary": site,
                "start": e.start,
                "end": -1 if e.end is None else e.end,
                "days": float("nan") if e.end is None else (e.end - e.start) / DAY,
                "amp_mw": e.amp_mw,
                "units": e.units,
                "ramp_fraction": e.ramp_fraction,
                "edge_days": edge_days,
                "edge_complete": edge_complete,
                "cls": e.cls,
                "dist_season_boundary_days": season_boundary_distance_days(start=e.start),
                "mean_clearness_3d": mean_clear,
                "sunny_share_3d": sunny_share,
                "share_midday": share_mid,
                "slow_misfit_units": float(misfit),
                "n_sites": len(sites),
            }
        )
    print("world", k, "false events", sum(r["kind"] == "false_event" for r in rows), flush=True)
    return rows


def main() -> None:
    """Run the worlds on two worker processes and write the table."""
    WORK.mkdir(parents=True, exist_ok=True)
    ctx = get_context("spawn")
    with ctx.Pool(2) as pool:
        parts = pool.map(one_world, range(N_WORLDS), chunksize=1)
    pl.DataFrame([r for part in parts for r in part], infer_schema_length=None).write_parquet(
        WORK / "false_events.parquet"
    )


if __name__ == "__main__":
    main()

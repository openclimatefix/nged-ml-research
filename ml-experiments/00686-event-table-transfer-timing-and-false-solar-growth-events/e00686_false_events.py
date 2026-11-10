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

import sys
from datetime import timedelta
from multiprocessing import get_context
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, GRID_START, SCORED_FROM  # ty: ignore[unresolved-import]
from e00653_core import decode_world  # ty: ignore[unresolved-import]
from e00653_inject import GenParams, inject  # ty: ignore[unresolved-import]
from e00653_run import _overlap, load_params, setup  # ty: ignore[unresolved-import]
from e00653_segment import step_width  # ty: ignore[unresolved-import]
from e00653_slow import to_half_hours  # ty: ignore[unresolved-import]
from e00657_solar import haurwitz, load_irradiance  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

sys.path.insert(0, "ml-experiments/00686-event-table-transfer-timing-and-false-solar-growth-events")
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
    import polars as pl
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
        width = step_width(xh=xh, t=e.start, span=2 * DAY)
        b0 = e.start // 12
        # Slow level's misfit: injected change over the 3 days after the start, less the slow
        # level's own change relative to the un-injected record, in noise units.
        a, b = e.start, min(e.start + 3 * DAY, GRID_LENGTH)
        pre = slice(max(0, e.start - 3 * DAY), e.start)
        inj_change = np.nanmean(injected[a:b]) - np.nanmean(injected[pre])
        slow_change = float(
            np.nanmean((slows[site] - base_slow[site])[b0 : b0 + 3 * 4 * 2])
            - np.nanmean((slows[site] - base_slow[site])[max(0, b0 - 12) : b0])
        )
        misfit = (inj_change - slow_change) / sigma
        clear3 = clearness[site][a:b].reshape(-1, DAY)[:, MIDDAY] if (b - a) % DAY == 0 else None
        mean_clear = float(np.nanmean(clear3)) if clear3 is not None else float("nan")
        sunny_share = float(np.nanmean(clear3 >= SUNNY)) if clear3 is not None else float("nan")
        # Share of the event's megawatts in 10:00 to 15:00 UTC, from the diurnal difference between
        # the 7 days after and the 7 days before the start.
        post_d = world[site][e.start : e.start + 7 * DAY]
        pre_d = world[site][max(0, e.start - 7 * DAY) : e.start]
        if len(post_d) == 7 * DAY and len(pre_d) == 7 * DAY:
            dprof = np.nanmean(post_d.reshape(7, DAY), axis=0) - np.nanmean(
                pre_d.reshape(7, DAY), axis=0
            )
            total = float(np.nansum(dprof))
            share_mid = (
                float(np.nansum(dprof[MIDDAY]) / total) if abs(total) > 1e-9 else float("nan")
            )
        else:
            share_mid = float("nan")
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
                "width_hh": width,
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
    pl.DataFrame([r for part in parts for r in part]).write_parquet(WORK / "false_events.parquet")


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Experiment 652: shared paths, duration mixtures, and per-row shift state for the injected bank.

Experiment 650's banks (`head_data/bank_<month>.npz`) and event tables (`k1_bank/events_*.parquet`)
are reused. A bank row is one (Primary, origin, valid time) with a lead of 24 h or more.
"""

from typing import Final

import e00650_bank as gen  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00652-two-output-forecast-with-explicit-rules-and-survival-hazard"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME
MONTHS: Final = c650.MONTHS
SEEDS: Final[tuple[int, ...]] = (0, 1)
SHORT_MAX_DAYS: Final[int] = 14
MIXTURES: Final[dict[str, float]] = {"m50": 0.5, "mshort": 0.8, "mlong": 0.2}
"""Mixture name -> share of shifts that are short (3 to 14 days). The generator draws 0.5."""
GENERATOR_SHORT_SHARE: Final[float] = 0.5
N_DAYS_HORIZON: Final[int] = 10
STRATA: Final = ("free", "far", "middle", "near", "just_ended")
POST_END_DAYS: Final[int] = 7


def events_table(*, month: str) -> pl.DataFrame:
    """Return the events of every pass of `month`."""
    return pl.concat(
        [pl.read_parquet(c650.ROWS / f"events_{month}_r{r:02d}.parquet") for r in c650.ALL_PASSES]
    )


def _unclipped_days(*, month_idx: int, r: int, primary: int, nb_of: dict, is_mw: np.ndarray) -> int:
    """Return the generator's drawn duration in days, before the clip at the window edge."""
    ev = gen.draw(
        month_idx=month_idx,
        r=r,
        primary=primary,
        nb_of=nb_of,
        month_start_idx=month_idx,
        window_end_idx=10**9,
        is_mw=is_mw,
    )
    return (ev.end_idx - ev.onset_idx) // 48


def class_weight(*, short: np.ndarray, mixture: str) -> np.ndarray:
    """Return the per-event-class weight that turns the generator's duration mixture into `mixture`."""
    s = MIXTURES[mixture]
    return np.where(short, s / GENERATOR_SHORT_SHARE, (1 - s) / (1 - GENERATOR_SHORT_SHARE))


def row_state(*, z: dict, month: str) -> dict[str, np.ndarray]:
    """Return per-row shift state of a bank file `z`: onset, end, short class, in force, still on, stratum.

    `in_force` means the Primary's shift is on at the origin. `still_on` is the survival label (the
    shift is still on at the valid time, defined only for in-force rows). `event_w` makes each event count equally (the mean row count of an event over the row count of this
    event), so that a mixture is a mixture of events and not of rows. `stratum` is one of
    `STRATA`, `censored` (shift clipped at the window edge, so its end is false), or `onset_ahead` for rows whose shift starts after the origin and before the valid
    time (no forecast can see an onset), or `other` (an origin before the onset whose valid time is
    also before it is `free`; an origin more than 7 days after the end is `free`).
    """
    ev = events_table(month=month)
    onset = {
        (int(r["pass"]), int(r["primary"])): int(r["onset_idx"]) for r in ev.iter_rows(named=True)
    }
    end = {(int(r["pass"]), int(r["primary"])): int(r["end_idx"]) for r in ev.iter_rows(named=True)}
    pa, pr = z["pass_"].astype(int), z["prim"].astype(int)
    on = np.array([onset[(p, q)] for p, q in zip(pa, pr, strict=True)])
    en = np.array([end[(p, q)] for p, q in zip(pa, pr, strict=True)])
    oi, vi = z["oi"].astype(np.int64), z["vi"].astype(np.int64)
    in_force = (oi >= on) & (oi < en)
    remaining_days = (en - oi) / 48.0
    after_days = (oi - en) / 48.0
    stratum = np.full(len(oi), "free", dtype="<U12")
    stratum[in_force & (remaining_days > N_DAYS_HORIZON)] = "far"
    stratum[in_force & (remaining_days <= N_DAYS_HORIZON) & (remaining_days > 1.0)] = "middle"
    stratum[in_force & (remaining_days <= 1.0)] = "near"
    stratum[(oi >= en) & (after_days <= POST_END_DAYS)] = "just_ended"
    stratum[(oi < on) & (vi >= on)] = "onset_ahead"
    nb_of, _cap, is_mw = c650.world()
    ms = c650.month_start_idx(month=month)
    days = {
        (r, p): _unclipped_days(month_idx=ms, r=r, primary=p, nb_of=nb_of, is_mw=is_mw)
        for r, p in set(zip(pa.tolist(), pr.tolist(), strict=True))
    }
    short = np.array([days[(p, q)] <= SHORT_MAX_DAYS for p, q in zip(pa, pr, strict=True)])
    key = pa * 100 + pr
    _u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    event_w = (cnt.mean() / cnt[inv]).astype(np.float32)
    censored = en >= c650.WINDOW_START_IDX
    stratum[in_force & censored] = "censored"
    return {
        "onset": on, "end": en, "short": short, "in_force": in_force,
        "still_on": in_force & (vi < en), "stratum": stratum, "censored": censored, "event_w": event_w,
    }  # fmt: skip

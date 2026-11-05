# ruff: noqa: D103, E501
"""Experiment 580, step 5: the injected event bank (design table and an apply function).

A deterministic design in the spirit of experiment 402's `inject()` (load-conserving: the donor changes
by a depth and one connected recipient gains the same MW). `inject()` itself draws depths, lengths and
dates at random, so it cannot pin the design the review asked for; this module fixes it:

- depth: 15%, 30% or 50% of the smaller capacity of the pair (so neither Primary exceeds that share);
- length: 3, 7, 15 or 30 days; onset at 00:00 (night) or 12:00 (day) UTC;
- 3 start dates (days 3, 12 and 21) in each of the five fold months of experiments 565/566/578;
- one event per world, with donor and recipient taken round-robin from the connected pairs of
  experiment 400's interconnection list, excluding D15 and D18 (report-only), and only where both
  Primaries have at least 95% finite cleaned power from 35 days before the onset to 14 days after the
  end and carry no labelled event or telemetry-found transfer within 14 days of the window.
- the sign of the donor's change alternates between specs, so half the donors gain load.

Use the bank twice, as the review describes: inject into the lags only (forecast-time following)
and into the lags and the fine-tune targets (absorption). `apply_spec` returns the injected power; `correct`
from `e00580_verify` removes it.
"""

from datetime import UTC, datetime
from itertools import product
from typing import Final

import e00400_blend as b400  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00580_verify as v  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

DEPTHS: Final[tuple[float, ...]] = (0.15, 0.30, 0.50)
LENGTHS_DAYS: Final[tuple[int, ...]] = (3, 7, 15, 30)
ONSET_HOURS: Final[tuple[int, ...]] = (0, 12)
START_DAYS: Final[tuple[int, ...]] = (3, 12, 21)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
EXCLUDED: Final[frozenset[str]] = frozenset({"D15", "D18"})
FLOOR: Final[float] = 0.1
GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
DAY: Final[int] = 48


def month_index(month: str) -> int:
    d = datetime(int(month[:4]), int(month[5:]), 1, tzinfo=UTC)
    return int((d - GRID_START).total_seconds() // 1800)


def build() -> pl.DataFrame:
    ev, labels, x, cap, _, _, _ = v.load_series()
    row = {lab: i for i, lab in enumerate(labels)}
    nb = b400.nged_neighbours()
    pairs = sorted(
        (d, r)
        for d, rs in nb.items()
        for r in rs
        if d in row and r in row and d not in EXCLUDED and r not in EXCLUDED
    )
    tr = pl.read_parquet(v.WORK / "events_verified.parquet")
    busy: dict[str, list[tuple[int, int]]] = {lab: [] for lab in labels}
    for r in ev.iter_rows(named=True):
        busy[r["label"]].append((r["onset_idx"], r["end_idx"]))
    for r in tr.iter_rows(named=True):
        busy[r["primary_a"]].append((r["onset_a_idx"], r["end_a_idx"]))
        busy[r["primary_b"]].append((r["onset_b_idx"], r["end_b_idx"]))
    mva = MVA_ONLY(labels)
    specs = []
    k = 0
    for month, day, depth, length, hour in product(
        MONTHS, START_DAYS, DEPTHS, LENGTHS_DAYS, ONSET_HOURS
    ):
        onset = month_index(month) + (day - 1) * DAY + hour * 2
        end = onset + length * DAY
        chosen = None
        for j in range(len(pairs)):
            d, r = pairs[(k + j) % len(pairs)]
            ok = True
            for lab in (d, r):
                seg = x[row[lab], onset - 35 * DAY : end + 14 * DAY]
                if np.isfinite(seg).mean() < 0.95 or any(
                    lo - 14 * DAY <= end and hi + 14 * DAY >= onset for lo, hi in busy[lab]
                ):
                    ok = False
            if ok:
                mw = depth * min(cap[d], cap[r])
                sign = 1 if k % 2 == 0 else -1
                dd, dr = sign * mw / cap[d], -sign * mw / cap[r]
                for lab, delta in ((d, dd), (r, dr)):
                    w = x[row[lab], onset:end]
                    low = np.nanmin(w + delta)
                    if lab in mva and low <= FLOOR:
                        ok = False
                    if np.nanmax(w + delta) > 1.5 or low < -0.5:
                        ok = False
                if ok:
                    chosen = (d, r, sign, mw)
                    break
        k += 1
        if chosen is None:
            continue
        d, r, sign, mw = chosen
        specs.append(
            {
                "spec_id": len(specs),
                "month": month,
                "start_day": day,
                "depth_share_of_smaller_capacity": depth,
                "length_days": length,
                "onset_hour_utc": hour,
                "onset_idx": onset,
                "end_idx": end,
                "donor": d,
                "recipient": r,
                "donor_depth_mw": sign * mw,
                "recipient_depth_mw": -sign * mw,
            }
        )
    return pl.DataFrame(specs)


def MVA_ONLY(labels: list[str]) -> set[str]:  # noqa: N802
    sm = k578.sites()
    return {
        lab
        for lab, g in zip(sm["label"].to_list(), sm["group"].to_list(), strict=True)
        if g.startswith("MVA")
    }


def apply_spec(
    *, power: np.ndarray, labels: list[str], cap: dict[str, float], spec: dict
) -> np.ndarray:
    """Return `power` (primaries by grid, share of capacity) with one spec injected; the input is not changed."""
    out = power.copy()
    for lab, mw in (
        (spec["donor"], spec["donor_depth_mw"]),
        (spec["recipient"], spec["recipient_depth_mw"]),
    ):
        out[labels.index(lab), spec["onset_idx"] : spec["end_idx"]] += mw / cap[lab]
    return out


if __name__ == "__main__":
    bank = build()
    bank.write_parquet(v.WORK / "injected_event_bank.parquet")
    print(
        bank.height,
        "specs of",
        len(MONTHS) * len(START_DAYS) * len(DEPTHS) * len(LENGTHS_DAYS) * len(ONSET_HOURS),
    )
    print(bank.group_by("length_days").len().sort("length_days").rows())
    ev, labels, x, cap, *_ = v.load_series()
    del ev
    s = bank.row(0, named=True)
    inj = apply_spec(power=x, labels=labels, cap=cap, spec=s)
    for lab, mw in ((s["donor"], s["donor_depth_mw"]), (s["recipient"], s["recipient_depth_mw"])):
        back = v.correct(
            observed=inj[labels.index(lab)],
            depth=mw / cap[lab],
            onset=s["onset_idx"],
            end=s["end_idx"],
        )
        assert np.allclose(back, x[labels.index(lab)], equal_nan=True, atol=1e-5)
    assert abs(s["donor_depth_mw"] + s["recipient_depth_mw"]) < 1e-9
    print("known-answer check passed: correct() removes the injected event and MW is conserved")

"""Phase 0 check of the injected worlds: write each fold's shifts to disk, then read them back.

The check reads the files it wrote, so it tests what the scoring code will read, and not a
simulation of it. It makes two tests. First, the share of scored rows in each stratum (built from
the drawn end and the drawn shift) must match the plan. Second, no shift reaches the holdout, and
the forecast's valid-time filter drops rows of the 2026-06 fold and of no other fold, which needs
no fit.
"""

from __future__ import annotations

import calendar
from pathlib import Path
from typing import Any

import e00738_common as common
import e00738_panel as panel
import e00738_world as world
import numpy as np
import polars as pl

LEADS_PER_ORIGIN: int = 48
"""Half-hours in a day: each scored origin and lead has 48 target rows."""
LAST_HOLDOUT_FOLD: str = "2026-06"
"""The only fold whose scored rows reach the holdout."""


def worlds_path(*, month: str, out: Path | None = None) -> Path:
    """Return the file that holds one fold's injected shifts."""
    return (out if out is not None else common.OUT) / "worlds" / f"{month}.parquet"


def write_worlds(*, month: str, out: Path | None = None) -> Path:
    """Draw every scoring panel's 16 shifts of a fold and write them to one parquet file.

    Args:
        month: The fold month.
        out: The output folder (`OUT` by default).

    Returns:
        The path written.
    """
    import e00738_fit as fit
    import e00738_predict as pred

    is_mw = fit.fold_world()[2]
    import e00650_common as c650  # ty: ignore[unresolved-import]

    shifts = pred.all_panel_shifts(
        month=month,
        month_start=int(c650.month_start_idx(month=month)),
        window_end=fit.window_end_idx(),
        is_mw=is_mw,
        real_days=world.load_real_durations_days(),
    )
    rows = [
        {
            "panel": name,
            "primary": s.primary,
            "onset_idx": s.onset_idx,
            "end_idx": s.end_idx,
            "drawn_days": s.drawn_days,
        }
        for name, lst in shifts.items()
        for s in lst
    ]
    path = worlds_path(month=month, out=out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    pl.DataFrame(rows).write_parquet(tmp)
    tmp.rename(path)
    return path


def scored_valid_idx(*, month_start_idx: int, month: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the origin and valid grid indices (flat) of the rows a fold scores.

    The origins are the 12:00 UTC contract origins of the month's days. The leads are the pooled
    leads of `world.POOLED_LEAD_DAYS` (1 to 9.5 days).
    """
    year, mon = (int(v) for v in month.split("-"))
    days = calendar.monthrange(year, mon)[1]
    origins = month_start_idx + np.arange(days) * world.PER_DAY + 24
    lead = np.array([int(v * world.PER_DAY) for v in world.POOLED_LEAD_DAYS])
    o, ld = np.meshgrid(origins, lead, indexing="ij")
    return o.ravel(), (o + ld).ravel()


def check_worlds(*, months: tuple[str, ...], out: Path | None = None) -> dict:
    """Read the files of the given folds and return the stratum shares and the holdout findings.

    Args:
        months: The fold months, each with a file under `worlds/`.
        out: The output folder (`OUT` by default).

    Returns:
        A dict with `shares` (pooled over the training-family panels of every fold), `dropped_rows`
        (the rows the valid-time filter drops, by fold), `latest_shift_end_idx`, and `holdout_idx`.
    """
    import e00650_common as c650  # ty: ignore[unresolved-import]
    import e00738_score as score

    holdout_idx = (common.HOLDOUT_START_US - common.GRID_START_US) // common.STEP_US
    labels: list[np.ndarray] = []
    dropped: dict[str, int] = {}
    latest = 0
    for month in months:
        table = pl.read_parquet(worlds_path(month=month, out=out))
        start = int(c650.month_start_idx(month=month))
        o, v = scored_valid_idx(month_start_idx=start, month=month)
        valid_us = common.GRID_START_US + v * common.STEP_US
        # a row's 48 target half-hours all sit after its valid-day start, so the last one is +47
        last_us = valid_us + (LEADS_PER_ORIGIN - 1) * common.STEP_US
        dropped[month] = int((~panel.before_holdout(valid_us=last_us)).sum())
        latest = max(latest, int(table["end_idx"].max()))  # ty: ignore[invalid-argument-type]
        for name in table["panel"].unique().to_list():
            if score.world_of(name=name) != "train_family":
                continue
            sub = table.filter(pl.col("panel") == name).sort("primary")
            assert sub.height == 16, (month, name, sub.height)
            onset = sub["onset_idx"].to_numpy()
            drawn_end = onset + np.round(sub["drawn_days"].to_numpy() * world.PER_DAY).astype(int)
            n = o.size
            # one shift per Primary: every Primary is scored at every row
            labels.extend(
                world.strata(
                    origin_idx=o,
                    valid_idx=v,
                    onset_idx=np.full(n, onset[p]),
                    end_idx=np.full(n, drawn_end[p]),
                    clipped=np.zeros(n, dtype=bool),
                )
                for p in range(16)
            )
    flat = np.concatenate(labels)
    shares = {name: float((flat == name).mean()) for name in world.STRATA}
    return {
        "shares": shares,
        "dropped_rows": dropped,
        "latest_shift_end_idx": latest,
        "holdout_idx": int(holdout_idx),
    }


def verdict(*, found: dict) -> dict[str, Any]:
    """Return the shares that are off the plan and the holdout faults of a `check_worlds` result."""
    bad = {}
    for k, want in common.EXPECTED_SHARES.items():
        tol = (
            common.STRATUM_TOLERANCE_END if k in {"near", "ended_24h"} else common.STRATUM_TOLERANCE
        )
        if abs(found["shares"][k] - want) > tol:
            bad[k] = (found["shares"][k], want)
    faults = []
    if found["latest_shift_end_idx"] > found["holdout_idx"]:
        faults.append("a shift reaches past 2026-07-01")
    for month, n in found["dropped_rows"].items():
        if (n > 0) != (month == LAST_HOLDOUT_FOLD):
            faults.append(f"the valid-time filter drops {n} rows of fold {month}")
    return {"off": bad, "faults": faults}

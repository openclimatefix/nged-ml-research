# ruff: noqa: D103
"""Experiment 620: two descriptive second readings of the scored window (CPU, stored tables only).

Reads `rows.npz` that `e00620_score.py` wrote (per-row CRPS and median of every system) and the
cleaned power grid. It re-runs no network. Neither reading is used to rank the systems.

- Event-free rule R1, fixed by the audit before any score on the removed rows was computed. For each
  Primary and UTC day, take the daily mean of observed power minus the median of the heuristic with
  all 13 members, and the same for the climatology (history before 2026-07-01). Subtract the day's
  median across the 16 Primaries from each series, average the two residuals, and standardise each
  Primary by its median and 1.4826 x the median absolute deviation over 2026-05-01 to 2026-06-30.
  Flag days with |z| > 3, bridge single-day gaps, keep runs of 3 or more days, and drop every row of
  that Primary whose valid day lies in a kept run, for all five systems alike.
- Without D15 and D18, the two report-only Primaries with short histories.

Usage: E00620_PHASE=predict python e00620_second_reading.py SCORE_DIR
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00583_common as c  # ty: ignore[unresolved-import]
import e00620_extend as ext  # ty: ignore[unresolved-import]
import e00620_score as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SYSTEMS = s.SYSTEMS
FIRST_DAY = datetime(2026, 5, 1, tzinfo=UTC)
CUTOFF = "2026-07-01"


def day_index(when: datetime) -> int:
    return int((when - s.START_US_DT) / (s.STEP_DT * 48))


def residual_days(*, clean: np.ndarray, cutoff_idx: int) -> np.ndarray:
    """Return the (16, n_days) average of the two standardised-before-standardising residuals."""
    d0, d1 = day_index(FIRST_DAY), clean.shape[1] // 48  # to the end of the extended grid
    halves = np.arange(d0 * 48, d1 * 48)
    prim = np.repeat(np.arange(16), len(halves))
    idx = np.tile(halves, 16)
    heur = s.manual_heuristic(clean=clean, prim=prim, idx=idx, lead_h=np.zeros(len(idx)))[:, 5]
    clim = s.climatology(clean=clean, prim=prim, idx=idx, cutoff_idx=cutoff_idx)[:, 5]
    obs = clean[prim, idx]
    n_days = d1 - d0
    out = []
    for ref in (heur, clim):
        diff = (obs - ref).reshape(16, n_days, 48)
        out.append(np.nanmean(diff, axis=2))
    for k, a in enumerate(out):
        out[k] = a - np.nanmedian(a, axis=0, keepdims=True)
    return (out[0] + out[1]) / 2


def kept_runs(*, resid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return z (16, n_days) and the boolean mask of days inside kept runs."""
    d0 = day_index(FIRST_DAY)
    base = slice(0, day_index(datetime(2026, 7, 1, tzinfo=UTC)) - d0)
    med = np.nanmedian(resid[:, base], axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(resid[:, base] - med), axis=1, keepdims=True) * 1.4826
    z = (resid - med) / mad
    flag = np.abs(z) > 3
    keep = np.zeros_like(flag)
    for p in range(16):
        f = flag[p].copy()
        for i in range(1, len(f) - 1):
            if not f[i] and f[i - 1] and f[i + 1]:
                f[i] = True
        i = 0
        while i < len(f):
            if f[i]:
                j = i
                while j < len(f) and f[j]:
                    j += 1
                if j - i >= 3:
                    keep[p, i:j] = True
                i = j
            else:
                i += 1
    return z, keep


def main(score: Path) -> None:
    ext.redirect()
    clean = c.k578.clean()
    cutoff_idx = int(
        (datetime.fromisoformat(CUTOFF).replace(tzinfo=UTC) - s.START_US_DT) / s.STEP_DT
    )
    z = np.load(score / "rows.npz")
    prim, idx, y, day = z["prim"], z["idx"], z["y"], z["day"]
    labels = pl.read_csv(score / "primaries.csv")["label"].to_list()
    resid = residual_days(clean=clean, cutoff_idx=cutoff_idx)
    zz, keep = kept_runs(resid=resid)
    d0 = day_index(FIRST_DAY)
    vday = idx // 48 - d0
    removed = keep[prim, vday]
    pl.DataFrame(
        [
            {
                "primary": labels[p],
                "day": str((s.START_US_DT + (d + d0) * s.STEP_DT * 48).date()),
                "z": float(zz[p, d]),
            }
            for p, d in zip(*np.nonzero(keep), strict=True)
        ]
    ).write_csv(score / "second_reading_removed_days.csv")
    print(
        "R1 removed rows",
        int(removed.sum()),
        "of",
        len(removed),
        "Primary-days",
        int(keep.sum()),
        flush=True,
    )
    d15_18 = np.isin(prim, [labels.index("D15"), labels.index("D18")])
    readings = {
        "all rows": np.ones(len(prim), bool),
        "event-free rows (rule R1)": ~removed,
        "removed by rule R1": removed,
        "without D15 and D18": ~d15_18,
    }
    rows, wins = [], []
    for name, m in readings.items():
        per = {
            n: np.array(
                [
                    z[f"crps_{n}"][m & (prim == p)].mean() if (m & (prim == p)).any() else np.nan
                    for p in range(16)
                ]
            )
            for n in SYSTEMS
        }
        for n in SYSTEMS:
            crps_ = z[f"crps_{n}"]
            mae = np.abs(y - z[f"q50_{n}"]) * 100
            mc, lc, hc, _ = s.day_boot(day=day[m], vals=crps_[m])
            mm, lm, hm, _ = s.day_boot(day=day[m], vals=mae[m])
            rec = {
                "reading": name,
                "system": n,
                "rows": int(m.sum()),
                "crps": mc,
                "crps_lo": lc,
                "crps_hi": hc,
                "mae": mm,
                "mae_lo": lm,
                "mae_hi": hm,
            }
            if n != "Manual heuristic":
                _, dl, dh, _ = s.day_boot(day=day[m], vals=crps_[m] - z["crps_Manual heuristic"][m])
                rec |= {
                    "crps_minus_heuristic": float(
                        (crps_[m] - z["crps_Manual heuristic"][m]).mean()
                    ),
                    "diff_lo": dl,
                    "diff_hi": dh,
                }
                ok = np.isfinite(per[n]) & np.isfinite(per["Manual heuristic"])
                wins.append(
                    {
                        "reading": name,
                        "system": n,
                        "primaries_scored": int(ok.sum()),
                        "primaries_lower_crps_than_heuristic": int(
                            (per[n][ok] < per["Manual heuristic"][ok]).sum()
                        ),
                    }
                )
            rows.append(rec)
    pl.DataFrame(rows).write_csv(score / "second_reading.csv")
    pl.DataFrame(wins).write_csv(score / "second_reading_wins.csv")
    print(
        pl.DataFrame(rows).select(
            "reading", "system", "rows", "crps", "mae", "crps_minus_heuristic", "diff_lo", "diff_hi"
        ),
        flush=True,
    )
    print(pl.DataFrame(wins), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

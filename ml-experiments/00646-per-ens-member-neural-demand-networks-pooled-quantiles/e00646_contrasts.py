# ruff: noqa: D103, E501
"""Experiment 646: paired CRPS contrasts of each arm against its comparator, by fold group.

Reads `rows.npz` from `e00646_score.py` (per-row event-free CRPS flag, lead, month, Primary) and
prints, for each `ARM:COMPARATOR` pair and each variant, the event-free CRPS difference (arm minus
comparator, points of capacity, 14 Primaries without D15 and D18, days 1 to 10) pooled over the five
months with a month-block 95% interval, by fold group, and by lead band. A negative value favours the
arm. The material-gain bar is 0.06 points in at least two of three fold groups.

Usage: python e00646_contrasts.py SCORE_DIR ARM:COMPARATOR [...]
"""

import sys
from pathlib import Path

import e00645_score as s645  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BANDS = {
    "day 1": (0, 24),
    "day 2": (24, 48),
    "days 3-5": (48, 120),
    "days 6-10": (120, 240),
    "days 1-10": (0, 240),
}
FOLD_GROUPS = {"2025-10 + 2025-12": (0, 1), "2026-01 + 2026-03": (2, 3), "2026-06": (4,)}
BAR = 0.06


def main(score: Path, pairs: list[str]) -> None:
    r = np.load(score / "rows.npz", allow_pickle=True)
    month, lead, flagged, _prim = r["month"], r["lead"], r["flagged"], r["prim"]
    short = r["short"]
    base = ~flagged & ~short
    rows = []
    for pair in pairs:
        arm, comp = pair.split(":")
        for variant in ("unscaled", "scaled"):
            diff = r[f"crps_{arm}_{variant}"] - r[f"crps_{comp}_{variant}"]
            for band, (lo, hi) in BANDS.items():
                m_ = base & (lead >= lo) & (lead < hi)
                mean, mlo, mhi = s645.month_boot(block=month[m_], vals=diff[m_], seed=646)
                rows.append(
                    {
                        "pair": pair,
                        "variant": variant,
                        "band": band,
                        "group": "all months",
                        "diff": mean,
                        "lo": mlo,
                        "hi": mhi,
                    }
                )
            for fname, months in FOLD_GROUPS.items():
                m_ = base & np.isin(month, months) & (lead < 240)
                rows.append(
                    {
                        "pair": pair,
                        "variant": variant,
                        "band": "days 1-10",
                        "group": fname,
                        "diff": float(diff[m_].mean()),
                        "lo": float("nan"),
                        "hi": float("nan"),
                    }
                )
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(160)
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])

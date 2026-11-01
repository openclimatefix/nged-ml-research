"""Score G1-trust-12 against B-mix, G1-trust (6 folds), G1-insample (6 folds), and K2-S1 (no fit).

Reuses experiment 463's scoring (`e00463_score`) with this experiment's arms, contrast list and
output folder. K2-S1 is experiment 454's stored forecast averaged over its stored cells in
quantile space (no widening, as in the other arms). `refs` writes that average once; `cells`,
`pool`, `weights` and `median` then score.

```bash
PYTHONPATH=<as e00469_crossfit12.py shows> DATA_PATH_INTERNAL=<private data folder> \
    uv run --no-sync python ml-experiments/00469-*/e00469_score.py refs
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00463_score as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00469-lag-trust-cross-fitted-trust-fit-twelve-folds"
)
DATA_463: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00463-lag-trust-cross-fitted-trust-fit-seed-zero" / "predictions"
)
DATA_454: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00454-lag-trust-mixture-and-ensemble-constructs-for-lagged-and-lagless-heads"
    / "work"
    / "predictions"
)
ARM_DIRS: Final[dict[str, Path]] = {
    "B-mix": s.DATA_404 / "B-mix",
    "A0-inj": s.DATA_404 / "A0-inj",
    "K2-S1": OUT_DIR / "predictions" / "K2-S1-mean",
    "G1-insample": DATA_463 / "G1-insample",
    "G1-trust": DATA_463 / "G1-trust",
    "G1-trust-12": OUT_DIR / "predictions" / "G1-trust-12",
}
FOCUS: Final[tuple[str, ...]] = ("G1-trust-12", "G1-trust", "K2-S1")
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("G1-trust-12", "B-mix"),
    ("G1-trust-12", "G1-trust"),
    ("G1-trust-12", "K2-S1"),
    ("G1-trust-12", "G1-insample"),
    ("G1-trust", "B-mix"),
    ("K2-S1", "B-mix"),
    ("A0-inj", "B-mix"),
)
"""The first three are the pre-registered contrasts; the rest are descriptive."""
CONTRAST_COUNT: Final[int] = 24
"""Pre-registered: three contrasts on four slices (no-label, harm, event with NRA 1.0, event
metered) at both groups = 3 x 4 x 2 = 24."""
KEYS: Final[list[str]] = ["primary", "origin", "target"]


def refs() -> None:
    """Write K2-S1 averaged over experiment 454's stored cells, month by month."""
    target = ARM_DIRS["K2-S1"]
    target.mkdir(parents=True, exist_ok=True)
    for month in s.MONTHS:
        paths = sorted((DATA_454 / "K2-S1").glob(f"{month}_s*_w-1.parquet"))
        frames = [pl.read_parquet(p) for p in paths]
        first = frames[0]
        for f in frames[1:]:
            if not f.select(KEYS).equals(first.select(KEYS)):
                msg = f"K2-S1 cells differ in rows in {month}"
                raise ValueError(msg)
        q = np.mean([f.select(s.LEVEL_NAMES).to_numpy().astype(np.float64) for f in frames], axis=0)
        q.sort(axis=1)
        out = first.with_columns(
            [pl.Series(name, q[:, i].astype(np.float32)) for i, name in enumerate(s.LEVEL_NAMES)]
        )
        out.write_parquet(target / f"{month}_s0_w-1.parquet")
        print(f"{month}: K2-S1 averaged over {len(paths)} cells")


def _patch() -> None:
    """Point experiment 463's scoring at this experiment's arms, contrasts, and folder."""
    s.ARM_DIRS = ARM_DIRS  # ty: ignore[invalid-assignment]
    s.PAIRS = PAIRS  # ty: ignore[invalid-assignment]
    s.CONTRAST_COUNT = CONTRAST_COUNT  # ty: ignore[invalid-assignment]
    s.OUT_DIR = OUT_DIR  # ty: ignore[invalid-assignment]


def weights() -> None:
    """Write the mean learned lagged weight by arm, group, lead band, and month."""
    groups = s.report.group_of(sites=s.report.fit.prep.primaries()["site"].to_list())
    rows = []
    for month in s.MONTHS:
        for arm in ("G1-trust-12", "G1-trust", "G1-insample"):
            frame = s.load_arm(arm=arm, month=month)
            w = frame["trust_w"].to_numpy()
            group = groups[frame["primary"].to_numpy().astype(np.int64)]
            band = np.searchsorted(s.BAND_EDGES, frame["lead_hours"].to_numpy(), side="right")
            for g in s.GROUPS:
                for b, name in enumerate(s.BAND_NAMES):
                    sel = (group == g) & (band == b)
                    if sel.any():
                        rows.append(
                            {"arm": arm, "month": month, "group": g, "band": name,
                             "weight": float(w[sel].mean()), "n": int(sel.sum())}
                        )  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv(OUT_DIR / "weights.csv", float_precision=3)
    with pl.Config(tbl_rows=100):
        print(
            out.group_by("arm", "group", "band")
            .agg(mean=pl.col("weight").mean(), swing=pl.col("weight").max() - pl.col("weight").min())
            .sort("arm", "group", "band")
        )
        print(out.group_by("arm", "month").agg(mean=pl.col("weight").mean()).sort("arm", "month"))


def median() -> None:
    """Write the error of the median forecast of each focus arm beside B-mix, by slice and group."""
    groups = s.report.group_of(sites=s.report.fit.prep.primaries()["site"].to_list())
    rows = []
    for month in s.MONTHS:
        frames = {a: s.load_arm(arm=a, month=month) for a in ("B-mix", *FOCUS)}
        ref = frames["B-mix"]
        primary = ref["primary"].to_numpy().astype(np.int64)
        actual = ref["actual"].to_numpy().astype(np.float64)
        for setname in ("nolabel", "harm", "event"):
            mask = s.row_set(frame=ref, name=setname)
            for g in s.GROUPS:
                sel = mask & (groups[primary] == g)
                if not sel.any():
                    continue
                base = np.abs(ref["q50"].to_numpy().astype(np.float64) - actual)[sel]
                for a in FOCUS:
                    err = np.abs(frames[a]["q50"].to_numpy().astype(np.float64) - actual)[sel]
                    rows.append(
                        {"month": month, "arm": a, "group": g, "set": setname, "n": int(sel.sum()),
                         "mae_median_diff_vs_bmix": float((err - base).mean())}
                    )  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv(OUT_DIR / "median_error.csv", float_precision=4)
    print(out.group_by("arm", "group", "set").agg(pl.col("n").sum(), pl.col("mae_median_diff_vs_bmix").mean()))


def main() -> None:
    """Run one step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("refs", "cells", "pool", "weights", "median"))
    args = parser.parse_args()
    _patch()
    {"refs": refs, "cells": s.cells, "pool": s.pool, "weights": weights, "median": median}[
        args.command
    ]()


if __name__ == "__main__":
    main()

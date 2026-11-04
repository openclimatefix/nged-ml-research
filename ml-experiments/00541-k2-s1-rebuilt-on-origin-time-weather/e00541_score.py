"""Score experiment 541's rebuilt K2-S1 on experiment 511's rows, with 521's statistics.

Commands: `rows`, `stats`.

The rebuilt K2-S1 is scored cell by cell and as the mean of three cells. The old K2-S1 is experiment
511's stored `K2-S1__crps` (cell c0). Contrasts: rebuilt minus old K2-S1 and rebuilt minus L-1d
(both counted); rebuilt minus L-Ww and the old K2-S1 minus L-1d are report-only. A-pool, A-pool
minus lags (537) and R-DW (539) come from their stored absolute tables, because their per-row scores
sit in other folders.
"""

import argparse
import json
import math

import e00521_refs as r521  # ty: ignore[unresolved-import]
import e00521_stats as s521  # ty: ignore[unresolved-import]
import e00541_common as m  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

KEYS = r521.KEYS
NEW = "K2-S1-origin"
TRAINED = "K2-S1-origin-trained"
BUILT = (NEW, TRAINED)
OLD = "K2-S1"
ARMS = (OLD, NEW, TRAINED, "L-1d", "L-Ww")
CONTRASTS = (
    (NEW, OLD, True),
    (NEW, "L-1d", True),
    (TRAINED, OLD, True),
    (TRAINED, "L-1d", True),
    (TRAINED, NEW, False),
    (TRAINED, "L-Ww", False),
    (OLD, "L-1d", False),
)
CELL_COLS = {arm: [f"{arm}-c{c}__crps" for c in range(3)] for arm in BUILT}
REFERENCE_TABLES = {
    "A-pool": ("00537-a-pool-minus-lags", "A-pool"),
    "A-pool minus lags": ("00537-a-pool-minus-lags", "A-nolag"),
    "R-DW": ("00539-l1d-plus-day-and-week-lags", "R-DW"),
}


def rows() -> None:
    """Write each month's per-row scores of the rebuilt K2-S1 cells."""
    m.WORK.mkdir(parents=True, exist_ok=True)
    for month in r521.MONTHS:
        base = pl.read_parquet(r521.E511 / "scores" / f"rows_{month}.parquet", columns=KEYS)
        parts = [base]
        for arm in BUILT:
            cells = [
                pl.read_parquet(m.WORK / "cells" / arm / f"{month}_c{c}.parquet").select(
                    *KEYS, "actual", *r521.LEVELS
                )
                for c in range(3)
            ]
            for c, cell in enumerate(cells):
                parts.append(r521.score_cells(rows=base, cells=[cell], name=f"{arm}-c{c}"))
            parts.append(r521.score_cells(rows=base, cells=cells, name=arm))
        frame = pl.concat(parts, how="horizontal")
        frame.write_parquet(m.WORK / f"new_{month}.parquet")
        print("rows", month, frame.height, flush=True)


def cell_mde(*, part: pl.DataFrame, arm: str) -> float:
    """Return `1.96 sqrt(2) sd / sqrt(k)` of the per-fold means of the three cells (as 531)."""
    values = (
        part.group_by("month")
        .agg([pl.col(c).mean() for c in CELL_COLS[arm]])
        .sort("month")
        .select(CELL_COLS[arm])
        .to_numpy()
    )
    folds, k = values.shape
    resid = values - values.mean(axis=1, keepdims=True)
    sd = math.sqrt(float((resid**2).sum()) / (folds * (k - 1)))
    return 1.96 * math.sqrt(2.0) * sd / math.sqrt(k)


def load() -> pl.DataFrame:
    """Return 511's rows with the references of experiment 521 and the rebuilt K2-S1 scores."""
    base = s521.load()
    extra = pl.concat(
        [pl.read_parquet(m.WORK / f"new_{mo}.parquet").drop(KEYS) for mo in r521.MONTHS]
    )
    extra = extra.select([c for c in extra.columns if c not in base.columns])
    return pl.concat([base, extra], how="horizontal").with_columns(band=s521.band_expr())


def stats() -> None:
    """Write the absolute and contrast tables."""
    m.SCORES.mkdir(parents=True, exist_ok=True)
    table = load()
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    n_counted = sum(1 for *_, c in CONTRASTS if c) * len(s521.BANDS) * 2  # 4 x 7 x 2 = 56
    for name in ("eight", "six", "report"):
        part = s521.row_set(table=table, name=name)
        for band in (*s521.BANDS, s521.ALL):
            sub = s521.in_band(part=part, band=band)
            for arm in ARMS:
                v = sub[f"{arm}__crps"]
                absolute.append(
                    {
                        "set": name,
                        "band": band,
                        "arm": arm,
                        "n": int(v.is_finite().sum()),
                        "crps": float(v.filter(v.is_finite()).mean()),
                    }
                )
            for a, b, counted in CONTRASTS:
                diff = sub.select(
                    "month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps")
                ).filter(pl.col("d").is_finite())
                mde = cell_mde(part=sub, arm=a) if a in BUILT else float("nan")
                contrasts.append(
                    {
                        "set": name,
                        "band": band,
                        "contrast": f"{a} - {b}",
                        "counted": counted,
                        "mde": mde,
                        **s521.block_stats(diff=diff, rng=rng, tests=n_counted),
                    }
                )
    for label, (folder, arm) in REFERENCE_TABLES.items():
        ref = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / folder / "scores" / "absolute.csv").filter(
            pl.col("arm") == arm
        )
        absolute.extend(
            {"set": r["set"], "band": r["band"], "arm": label, "n": r["n"], "crps": r["crps"]}
            for r in ref.iter_rows(named=True)
        )
    pl.DataFrame(absolute).write_csv(m.SCORES / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(m.SCORES / "contrasts.csv")
    (m.SCORES / "counts.json").write_text(json.dumps({"bonferroni_counted_tests": n_counted}))
    print("wrote tables; counted tests", n_counted)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rows", "stats"))
    {"rows": rows, "stats": stats}[parser.parse_args().command]()


if __name__ == "__main__":
    main()

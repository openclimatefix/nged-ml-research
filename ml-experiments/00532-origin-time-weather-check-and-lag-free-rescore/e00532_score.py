# ruff: noqa: E501
"""Score experiment 532's origin-time lag-free arms with 521's statistics against L-1d, L-Ww, S2-J and K2-S1. Commands: `rows`, `stats`."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e00521_refs as r521  # ty: ignore[unresolved-import]
import e00521_stats as s521  # ty: ignore[unresolved-import]
import e00531_score as s531  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA = ML_EXPERIMENTS_DATA_DIR / "00532-origin-time-weather-check-and-lag-free-rescore"
KEYS = r521.KEYS
LL = ("LL-ens3-origin", "LL-ens4-origin", "LL-ens5-origin")
OTHERS = ("K2-S1", "L-Ww", "L-1d", "S2-J-0.4x", "S2-J-0.1x")
ARMS = (*OTHERS, "LL-ens2-origin", *LL)
CONTRASTS = tuple((a, b) for a in ("LL-ens2-origin", *LL) for b in OTHERS)
SEED_COLS = {"LL-ens3-origin": None}


def rows() -> None:
    """Write per-row scores of the new cells."""
    for month in r521.MONTHS:
        base = pl.read_parquet(r521.E511 / "scores" / f"rows_{month}.parquet", columns=KEYS)
        parts = [base]
        for name in LL:
            cell = pl.read_parquet(DATA / "work" / "cells" / name / f"{month}_c0.parquet").select(
                *KEYS, "actual", *r521.LEVELS
            )
            parts.append(r521.score_cells(rows=base, cells=[cell], name=name))
        pl.concat(parts, how="horizontal").write_parquet(DATA / "work" / f"new_{month}.parquet")
        print("rows", month, flush=True)


def stats() -> None:
    """Write absolute and contrast tables."""
    (DATA / "scores").mkdir(exist_ok=True)
    base = s531.load()  # includes LL-ens2-origin__crps, L-1d, K2-S1, L-Ww, S2-J
    extra = pl.concat(
        [pl.read_parquet(DATA / "work" / f"new_{mo}.parquet").drop(KEYS) for mo in r521.MONTHS]
    )
    extra = extra.select([c for c in extra.columns if c not in base.columns])
    table = pl.concat([base, extra], how="horizontal").with_columns(band=s521.band_expr())
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    n_counted = len(CONTRASTS) * len(s521.BANDS) * 2
    for name in ("eight", "six"):
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
            for a, b in CONTRASTS:
                diff = sub.select(
                    "month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps")
                ).filter(pl.col("d").is_finite())
                cols = s531.SEED_COLS.get(a) if a == "LL-ens2-origin" else None
                mde = s531.seed_mde(part=sub, cols=cols) if cols else float("nan")
                contrasts.append(
                    {
                        "set": name,
                        "band": band,
                        "contrast": f"{a} - {b}",
                        "mde": mde,
                        **s521.block_stats(diff=diff, rng=rng, tests=n_counted),
                    }
                )
    pl.DataFrame(absolute).write_csv(DATA / "scores" / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(DATA / "scores" / "contrasts.csv")
    (DATA / "scores" / "counts.json").write_text(
        json.dumps({"bonferroni_counted_tests": n_counted})
    )
    print("counted tests", n_counted)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("rows", "stats"))
    {"rows": rows, "stats": stats}[p.parse_args().command]()

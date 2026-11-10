# ruff: noqa: PERF401
"""Experiment 531, `e00531_score.py`: kept for the record.

Score experiment 531's arms on experiment 511's rows with 521's statistics. Commands: `rows`,
`stats`.
"""

import argparse
import json
import math
from typing import Final

import e00521_refs as r521  # ty: ignore[unresolved-import]
import e00521_stats as s521  # ty: ignore[unresolved-import]
import e00531_common as m  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E527: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00527-lag-gap-head-count-test-k2-s1-vs-four-seed-daily-lag-free"
    / "work"
)
E529: Final = ML_EXPERIMENTS_DATA_DIR / "00529-lag-free-gap-recency-restart-and-cadence" / "work"
KEYS: Final = r521.KEYS
REF: Final = "K2-S1"
NEW_ARMS: Final = ("L1d-origin", "L1d-fresh", "LL-ens2-fresh", "LL-ens2-origin")


def read_seed(
    *, mode: str, month: str, seed: int, sites: pl.DataFrame, actual: pl.DataFrame
) -> pl.DataFrame:
    """Return a part-B seed table on the grid keys with `actual` joined."""
    frame = pl.read_parquet(m.WORK / "seed_preds" / f"L1d-{mode}" / f"{month}_s{seed}.parquet")
    frame = frame.join(sites.select("time_series_id", "primary"), on="time_series_id").with_columns(
        primary=pl.col("primary").cast(pl.Int16),
        origin=(
            (pl.col("power_fcst_init_time").dt.epoch("us") - r521.GRID_START_US) // r521.STEP_US
        ).cast(pl.Int32),
        target=((pl.col("valid_time").dt.epoch("us") - r521.GRID_START_US) // r521.STEP_US).cast(
            pl.Int32
        ),
    )
    return frame.select(*KEYS, *r521.LEVELS).join(actual, on=KEYS, how="inner")


def rows_command() -> None:
    """Write each month's per-row scores of the new arms."""
    sites = pl.read_csv(r521.DATA / "site_map.csv").with_columns(pl.col("primary").cast(pl.Int16))
    for month in r521.MONTHS:
        rows = pl.read_parquet(r521.E511 / "scores" / f"rows_{month}.parquet", columns=KEYS)
        actual = pl.read_parquet(r521.E454 / "K2-S1" / f"{month}_s0_w-1.parquet").select(
            *KEYS, "actual"
        )
        parts = [rows]
        for mode in ("origin", "fresh"):
            tables = [
                read_seed(mode=mode, month=month, seed=s, sites=sites, actual=actual)
                for s in (0, 1)
            ]
            for s, t in enumerate(tables):
                parts.append(r521.score_cells(rows=rows, cells=[t], name=f"L1d-{mode}-seed{s}"))
            parts.append(
                r521.score_cells(
                    rows=rows, cells=[r521.pooled_seeds(tables=tables)], name=f"L1d-{mode}"
                )
            )
        for mode in ("fresh", "origin"):
            cells = [
                pl.read_parquet(
                    m.WORK / "cells" / f"LL-ens2-{mode}" / f"{month}_c{c}.parquet"
                ).select(*KEYS, "actual", *r521.LEVELS)
                for c in range(3)
            ]
            for c, t in enumerate(cells):
                parts.append(r521.score_cells(rows=rows, cells=[t], name=f"LL-ens2-{mode}-c{c}"))
            parts.append(r521.score_cells(rows=rows, cells=cells, name=f"LL-ens2-{mode}"))
        frame = pl.concat(parts, how="horizontal")
        frame.write_parquet(m.WORK / f"new_{month}.parquet")
        print("rows", month, frame.height, flush=True)


ARMS: Final[tuple[str, ...]] = (
    "K2-S1",
    "L-Ww",
    "L-1d",
    "LL-ens2",
    "S2-J-0.4x",
    "S2-J-0.1x",
    *NEW_ARMS,
)
# (a, b, counted): a minus b
CONTRASTS: Final[tuple[tuple[str, str, bool], ...]] = (
    ("LL-ens2-origin", "LL-ens2", True),  # A: effect of origin-time weather on the lag-free head
    ("L1d-fresh", "L-1d", True),  # B: effect of look-ahead weather on the L-1d network
    ("LL-ens2-fresh", "LL-ens2", False),  # A reproduction (refit minus stored)
    ("L1d-origin", "L-1d", False),  # B reproduction (rerun minus stored)
    ("LL-ens2", REF, False), ("LL-ens2-origin", REF, False), ("LL-ens2-fresh", REF, False),
    ("L-1d", REF, False), ("L1d-fresh", REF, False), ("L1d-origin", REF, False),
    ("L-Ww", REF, False), ("S2-J-0.4x", REF, False), ("S2-J-0.1x", REF, False),
    ("L1d-fresh", "LL-ens2", False), ("L1d-fresh", "LL-ens2-origin", False),
)  # fmt: skip
SEED_COLS: Final[dict[str, list[str]]] = {
    "LL-ens2-origin": [f"LL-ens2-origin-c{c}__crps" for c in range(3)],
    "LL-ens2-fresh": [f"LL-ens2-fresh-c{c}__crps" for c in range(3)],
    "L1d-fresh": [f"L1d-fresh-seed{s}__crps" for s in (0, 1)],
    "L1d-origin": [f"L1d-origin-seed{s}__crps" for s in (0, 1)],
    "L-1d": ["L-1d-seed0__crps", "L-1d-seed1__crps"],
}


def seed_mde(*, part: pl.DataFrame, cols: list[str]) -> float:
    """Return `1.96 sqrt(2) sd / sqrt(k)` of the per-fold means of `cols` (as 527 and 530)."""
    values = (
        part.group_by("month")
        .agg([pl.col(c).mean() for c in cols])
        .sort("month")
        .select(cols)
        .to_numpy()
    )
    folds, k = values.shape
    resid = values - values.mean(axis=1, keepdims=True)
    sd = math.sqrt(float((resid**2).sum()) / (folds * (k - 1)))
    return 1.96 * math.sqrt(2.0) * sd / math.sqrt(k)


def load() -> pl.DataFrame:
    """Return 511's rows with the references (521, 527, 529) and this experiment's scores."""
    base = s521.load()
    frames = []
    for mo in r521.MONTHS:
        old = pl.read_parquet(E527 / f"old_{mo}.parquet").drop(KEYS).select("LL-ens2__crps")
        e529 = (
            pl.read_parquet(E529 / f"new_{mo}.parquet")
            .drop(KEYS)
            .select("S2-J-0.4x__crps", "S2-J-0.1x__crps")
        )
        mine = pl.read_parquet(m.WORK / f"new_{mo}.parquet").drop(KEYS)
        frames.append(pl.concat([old, e529, mine], how="horizontal"))
    extra = pl.concat(frames)
    extra = extra.select([c for c in extra.columns if c not in base.columns])
    return pl.concat([base, extra], how="horizontal").with_columns(band=s521.band_expr())


def stats() -> None:
    """Write the absolute and contrast tables."""
    m.SCORES.mkdir(parents=True, exist_ok=True)
    table = load()
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    n_counted = sum(1 for *_, c in CONTRASTS if c) * len(s521.BANDS) * 2
    for name in ("eight", "six", "six_all", "report"):
        part = s521.row_set(table=table, name=name)
        for band in (*s521.BANDS, s521.ALL):
            sub = s521.in_band(part=part, band=band)
            for arm in ARMS:
                absolute.append(
                    {
                        "set": name,
                        "band": band,
                        "arm": arm,
                        "n": int(sub[f"{arm}__crps"].is_finite().sum()),
                        "crps": float(
                            sub[f"{arm}__crps"].filter(sub[f"{arm}__crps"].is_finite()).mean()
                        ),
                    }
                )
            for a, b, counted in CONTRASTS:
                diff = sub.select(
                    "month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps")
                ).filter(pl.col("d").is_finite())
                cols = SEED_COLS.get(a)
                mde = seed_mde(part=sub, cols=cols) if cols else float("nan")
                stat = s521.block_stats(diff=diff, rng=rng, tests=n_counted)
                contrasts.append(
                    {
                        "set": name,
                        "band": band,
                        "contrast": f"{a} - {b}",
                        "counted": counted,
                        "mde": mde,
                        **stat,
                    }
                )
    pl.DataFrame(absolute).write_csv(m.SCORES / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(m.SCORES / "contrasts.csv")
    (m.SCORES / "counts.json").write_text(json.dumps({"bonferroni_counted_tests": n_counted}))
    print("wrote tables; counted tests", n_counted)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rows", "stats"))
    {"rows": rows_command, "stats": stats}[parser.parse_args().command]()


if __name__ == "__main__":
    main()

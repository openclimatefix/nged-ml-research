# ruff: noqa: E501, PERF401
r"""Experiment 529, `e00529_score.py`: kept for the record.

Score the arms of experiment 529 on experiment 511's rows, with 527's statistics.

Commands: `rows` (per-row scores of each new arm and each seed into `work/new_{month}.parquet`,
plus experiment 524's S2-J columns), `stats` (absolute scores, contrasts, per-fold signs, MDE).
"""

import argparse
import json
import math
from typing import Final

import e00521_refs as r521  # ty: ignore[unresolved-import]
import e00521_stats as s521  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final = ML_EXPERIMENTS_DATA_DIR / "00529-lag-free-gap-recency-restart-and-cadence"
WORK: Final = DATA / "work"
SCORES: Final = DATA / "scores"
E524: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00524-two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator"
    / "work"
    / "scores"
)
KEYS: Final = r521.KEYS
LEVELS: Final = r521.LEVELS
NEW: Final = ("W-none", "W-365", "W-scratch", "M-90")
S2J: Final = ("S2-J-0.4x", "S2-J-0.1x")
REF: Final = "K2-S1"


def read_seed(
    *, arm: str, month: str, seed: int, sites: pl.DataFrame, actual: pl.DataFrame
) -> pl.DataFrame:
    """Return an arm's seed table on the grid keys with `actual` joined."""
    frame = pl.read_parquet(WORK / "seed_preds" / arm / f"{month}_s{seed}.parquet")
    frame = frame.join(sites.select("time_series_id", "primary"), on="time_series_id").with_columns(
        primary=pl.col("primary").cast(pl.Int16),
        origin=(
            (pl.col("power_fcst_init_time").dt.epoch("us") - r521.GRID_START_US) // r521.STEP_US
        ).cast(pl.Int32),
        target=((pl.col("valid_time").dt.epoch("us") - r521.GRID_START_US) // r521.STEP_US).cast(
            pl.Int32
        ),
    )
    return frame.select(*KEYS, *LEVELS).join(actual, on=KEYS, how="inner")


def rows_command() -> None:
    """Write each month's new per-row scores."""
    torch.set_num_threads(1)
    sites = pl.read_csv(r521.DATA / "site_map.csv").with_columns(pl.col("primary").cast(pl.Int16))
    for month in r521.MONTHS:
        rows = pl.read_parquet(r521.E511 / "scores" / f"rows_{month}.parquet", columns=KEYS)
        actual = pl.read_parquet(r521.E454 / "K2-S1" / f"{month}_s0_w-1.parquet").select(
            *KEYS, "actual"
        )
        parts = [rows]
        for arm in NEW:
            tables = [
                read_seed(arm=arm, month=month, seed=s, sites=sites, actual=actual) for s in (0, 1)
            ]
            for s, t in enumerate(tables):
                parts.append(r521.score_cells(rows=rows, cells=[t], name=f"{arm}-seed{s}"))
            parts.append(
                r521.score_cells(rows=rows, cells=[r521.pooled_seeds(tables=tables)], name=arm)
            )
        # per-seed columns of the stored L-1d, to give the seed spread of the reference
        for s in (0, 1):
            t = r521.read_seed(month=month, seed=s, sites=sites, actual=actual)
            parts.append(r521.score_cells(rows=rows, cells=[t], name=f"L-1d-seed{s}"))
        s2 = pl.read_parquet(
            E524 / f"rows_{month}.parquet", columns=[*KEYS, *[f"{a}__crps" for a in S2J]]
        )
        assert s2.select(KEYS).equals(rows), f"{month}: 524 rows differ from 511 rows"
        parts.append(s2.drop(KEYS))
        frame = pl.concat(parts, how="horizontal")
        frame.write_parquet(WORK / f"new_{month}.parquet")
        print("rows", month, frame.height, flush=True)


ARMS: Final[tuple[str, ...]] = ("K2-S1", "L-1d", "L-Ww", *NEW, *S2J)
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    *[(a, REF, f"gap to K2-S1: {a}") for a in ("L-1d", "L-Ww", *NEW, *S2J)],
    *[(a, "L-1d", f"{a} minus daily L-1d") for a in NEW],
    *[(a, "L-Ww", f"{a} minus weekly L-Ww (same harness)") for a in NEW],
    ("M-90", "L-Ww", "cadence: monthly minus weekly"), ("L-Ww", "L-1d", "cadence: weekly minus daily"),
)  # fmt: skip
COUNTED: Final[int] = (
    12  # 4 new arms x 3 contrasts (vs K2-S1, vs L-1d, vs L-Ww); x 7 bands x 2 sets = 168
)


def seed_mde(*, part: pl.DataFrame, cols: list[str]) -> float:
    """Return `1.96 sqrt(2) sd / sqrt(k)` of the per-fold means of `cols` (experiment 470's pooling, as in 527)."""
    means = part.group_by("month").agg([pl.col(c).mean() for c in cols]).sort("month")
    values = means.select(cols).to_numpy()
    folds, k = values.shape
    resid = values - values.mean(axis=1, keepdims=True)
    sd = math.sqrt(float((resid**2).sum()) / (folds * (k - 1)))
    return 1.96 * math.sqrt(2.0) * sd / math.sqrt(k)


def load() -> pl.DataFrame:
    """Return 511's rows with reference scores and this experiment's."""
    base = s521.load()
    extra = pl.concat([pl.read_parquet(WORK / f"new_{m}.parquet").drop(KEYS) for m in r521.MONTHS])
    extra = extra.select([c for c in extra.columns if c not in base.columns])
    return pl.concat([base, extra], how="horizontal").with_columns(band=s521.band_expr())


def stats() -> None:
    """Write the absolute and contrast tables."""
    SCORES.mkdir(parents=True, exist_ok=True)
    table = load()
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    n_counted = COUNTED * 7 * 2
    seeds = {a: [f"{a}-seed0__crps", f"{a}-seed1__crps"] for a in (*NEW, "L-1d")}
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
                        "n": sub.height,
                        "crps": float(sub[f"{arm}__crps"].mean()),
                    }
                )
            for a, b, label in CONTRASTS:
                diff = sub.select(
                    "month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps")
                ).filter(pl.col("d").is_finite())
                cols = seeds.get(a, seeds["L-1d"])
                mde = seed_mde(part=sub, cols=cols)
                stat = s521.block_stats(diff=diff, rng=rng, tests=n_counted)
                contrasts.append(
                    {
                        "set": name,
                        "band": band,
                        "contrast": f"{a} - {b}",
                        "label": label,
                        "mde_2seed": mde,
                        **stat,
                    }
                )
    pl.DataFrame(absolute).write_csv(SCORES / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(SCORES / "contrasts.csv")
    (SCORES / "counts.json").write_text(json.dumps({"bonferroni_counted_tests": n_counted}))
    print("wrote tables; counted tests", n_counted)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rows", "stats"))
    {"rows": rows_command, "stats": stats}[parser.parse_args().command]()


if __name__ == "__main__":
    main()

# ruff: noqa: E501, PERF401, SIM222
r"""Score the arms of experiment 530 on experiment 511's rows, with 527's and 529's statistics.

Commands: `rows` (per-row scores of each new arm and each seed into `work/new_{month}.parquet`, plus
experiment 524's S2-J columns), `stats` (absolute scores, contrasts, per-fold signs, MDE).
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

DATA: Final = ML_EXPERIMENTS_DATA_DIR / "00530-lag-free-gap-network-family-ladder"
E527: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00527-lag-gap-head-count-test-k2-s1-vs-four-seed-daily-lag-free"
    / "work"
)
E529: Final = ML_EXPERIMENTS_DATA_DIR / "00529-lag-free-gap-recency-restart-and-cadence" / "work"
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
NEW: Final = ("N1", "N2", "N3", "N4")
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
        cells = [r521.read_454(arm="LL-ens2", month=month, cell=cc) for cc in range(3)]
        repro = r521.score_cells(rows=rows, cells=cells, name="LL-ens2-repro")
        stored = pl.read_parquet(E527 / f"old_{month}.parquet", columns=["LL-ens2__crps"])
        diff = float((repro["LL-ens2-repro__crps"] - stored["LL-ens2__crps"]).abs().max())
        print("LL-ens2 reproduction max |difference| in row CRPS", month, diff, flush=True)
        assert diff < 1e-6, "LL-ens2 does not reproduce"
        parts.append(repro)
        for arm in NEW:
            tables = [
                read_seed(arm=arm, month=month, seed=s, sites=sites, actual=actual) for s in (0, 1)
            ]
            for s, t in enumerate(tables):
                parts.append(r521.score_cells(rows=rows, cells=[t], name=f"{arm}-seed{s}"))
            parts.append(
                r521.score_cells(rows=rows, cells=[r521.pooled_seeds(tables=tables)], name=arm)
            )
        frame = pl.concat(parts, how="horizontal")
        frame.write_parquet(WORK / f"new_{month}.parquet")
        print("rows", month, frame.height, flush=True)


LADDER: Final = ("LL-ens2", *NEW, "M-90")
ARMS: Final[tuple[str, ...]] = ("K2-S1", "L-Ww", "L-1d", "LL-ens2", *NEW, "M-90", *S2J)
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    *[(a, REF, f"gap to K2-S1: {a}") for a in ("LL-ens2", *NEW, "M-90", "L-Ww", *S2J)],
    *[(LADDER[i + 1], LADDER[i], f"step {i + 1}: {LADDER[i + 1]} minus {LADDER[i]}") for i in range(len(LADDER) - 1)],
    ("M-90", "LL-ens2", "whole ladder: M-90 minus LL-ens2"),
)  # fmt: skip
STEP_BANDS: Final = ("0-6h", "6-12h", "120-240h")
COUNTED: Final[int] = 5  # the five ladder steps; x 3 bands x 2 sets = 30 Bonferroni tests


def seed_mde(*, part: pl.DataFrame, cols: list[str]) -> float:
    """Return `1.96 sqrt(2) sd / sqrt(k)` of the per-fold means of `cols` (experiment 470's pooling, as in 527)."""
    means = part.group_by("month").agg([pl.col(c).mean() for c in cols]).sort("month")
    values = means.select(cols).to_numpy()
    folds, k = values.shape
    resid = values - values.mean(axis=1, keepdims=True)
    sd = math.sqrt(float((resid**2).sum()) / (folds * (k - 1)))
    return 1.96 * math.sqrt(2.0) * sd / math.sqrt(k)


def load() -> pl.DataFrame:
    """Return 511's rows with the references (521, 527, 529, 524) and this experiment's scores."""
    base = s521.load()
    frames = []
    for m in r521.MONTHS:
        old = pl.read_parquet(E527 / f"old_{m}.parquet").drop(KEYS).select("LL-ens2__crps")
        e529 = (
            pl.read_parquet(E529 / f"new_{m}.parquet")
            .drop(KEYS)
            .select(
                "M-90__crps",
                "M-90-seed0__crps",
                "M-90-seed1__crps",
                "S2-J-0.4x__crps",
                "S2-J-0.1x__crps",
            )
        )
        mine = pl.read_parquet(WORK / f"new_{m}.parquet").drop(KEYS)
        mine = mine.select([c for c in mine.columns if c != "LL-ens2-repro__crps" or True])
        frames.append(pl.concat([old, e529, mine], how="horizontal"))
    extra = pl.concat(frames)
    extra = extra.select([c for c in extra.columns if c not in base.columns])
    return pl.concat([base, extra], how="horizontal").with_columns(band=s521.band_expr())


def stats() -> None:
    """Write the absolute and contrast tables."""
    SCORES.mkdir(parents=True, exist_ok=True)
    table = load()
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    n_counted = COUNTED * len(STEP_BANDS) * 2
    seeds = {a: [f"{a}-seed0__crps", f"{a}-seed1__crps"] for a in (*NEW, "M-90")}
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
                cols = seeds.get(a, seeds["M-90"])
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

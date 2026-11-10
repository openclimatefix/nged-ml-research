# ruff: noqa: E501, PERF401
r"""Experiment 527, `e00527_score.py`: kept for the record.

Score the arms of experiment 527 on experiment 511's rows (stored outputs plus L-1d seeds 2 and
3).

Commands:

- `old`: score experiment 454's `LL-ens2`, `S1-ens2` (three pair cells each) and a new `LL-ens4`
  (the four lag-free heads of fit seeds 0 to 3, mixed with equal weights, built from 454's
  code).
- `seeds`: score L-1d seeds 2 and 3, and the 4-seed L-1d averaged by quantile (`L-1d-4q`) and by
  an approximate mixture (`L-1d-4m`). Only the 11 quantiles of each seed are stored, so the
  mixture is built from a piecewise-linear CDF through each seed's quantiles (0 at q01 - 5% of
  the q01-q99 range, 1 at q99 + 5%); it is an approximation of 454's Gaussian-mixture solve.
- `stats`: the contrasts, the minimum detectable effects, per-fold signs.

Everything is written to `work/new_{month}.parquet` and `scores/` of this experiment's data
folder.
"""

import argparse
import json
import math
from typing import Final

import e00454_build as b454  # ty: ignore[unresolved-import]
import e00521_refs as r521  # ty: ignore[unresolved-import]
import e00521_stats as s521  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00527-lag-gap-head-count-test-k2-s1-vs-four-seed-daily-lag-free"
)
WORK: Final = DATA / "work"
SCORES: Final = DATA / "scores"
SEED_DIR: Final = WORK / "seed_preds" / "L-1d"
LEVELS: Final = r521.LEVELS
KEYS: Final = r521.KEYS
TESTS: Final[int] = 24
"""Counted tests: 7 lead bands x 2 row sets x 2 contrasts (H1, H3), plus H2's 7 x 2 = 14 report rows excluded."""


def mix_quantiles(*, tables: list[np.ndarray]) -> np.ndarray:
    """Return the 11 quantiles of the equal-weight mixture of per-seed piecewise-linear CDFs.

    `tables` hold (rows, 11) quantile arrays; the output is (rows, 11).
    """
    levels = np.asarray(r521.common.LEVELS, dtype=np.float64)
    n_rows = tables[0].shape[0]
    out = np.empty((n_rows, len(levels)))
    for i in range(n_rows):
        xs, fs = [], []
        for t in tables:
            q = np.sort(t[i])
            pad = 0.05 * (q[-1] - q[0]) + 1e-9
            xs.append(np.concatenate([[q[0] - pad], q, [q[-1] + pad]]))
            fs.append(np.concatenate([[0.0], levels, [1.0]]))
        knots = np.unique(np.concatenate(xs))
        cdf = np.mean([np.interp(knots, x, f) for x, f in zip(xs, fs, strict=True)], axis=0)
        out[i] = np.interp(levels, cdf, knots)
    return out


def with_levels(*, base: pl.DataFrame, q: np.ndarray) -> pl.DataFrame:
    """Return `base` (keys and actual) with the quantile columns `q`."""
    return base.hstack(pl.DataFrame(q, schema=list(LEVELS), orient="row"))


def old_month(*, month: str) -> pl.DataFrame:
    """Return the per-row scores of 454's two-head arms and the new four-head lag-free mixture."""
    rows = pl.read_parquet(r521.E511 / "scores" / f"rows_{month}.parquet", columns=KEYS)
    parts = [rows]
    for arm in ("LL-ens2", "S1-ens2"):
        cells = [r521.read_454(arm=arm, month=month, cell=c) for c in range(3)]
        parts.append(r521.score_cells(rows=rows, cells=cells, name=arm))
    heads, keys = b454.load_heads(month=month)
    stored = pl.read_parquet(r521.E454 / "LL-ens3" / f"{month}_s0_w-1.parquet")
    for k in KEYS:
        assert np.array_equal(stored[k].to_numpy(), keys[k]), f"{month}: {k} differs"
    q = b454.mix_equal(heads=[heads["L"][s] for s in range(4)])
    cell = with_levels(base=stored.select(*KEYS, "actual"), q=q)
    parts.append(r521.score_cells(rows=rows, cells=[cell], name="LL-ens4"))
    return pl.concat(parts, how="horizontal")


def seed_table(*, month: str, seed: int, sites: pl.DataFrame, actual: pl.DataFrame) -> pl.DataFrame:
    """Return one L-1d seed's table (seeds 0 and 1 from 511, 2 and 3 from this experiment)."""
    if seed < 2:
        return r521.read_seed(month=month, seed=seed, sites=sites, actual=actual)
    frame = pl.read_parquet(SEED_DIR / f"{month}_s{seed}.parquet")
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


def seeds_month(*, month: str, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the per-row scores of L-1d seeds 2 and 3 and the four-seed averages."""
    rows = pl.read_parquet(r521.E511 / "scores" / f"rows_{month}.parquet", columns=KEYS)
    actual = pl.read_parquet(r521.E454 / "K2-S1" / f"{month}_s0_w-1.parquet").select(
        *KEYS, "actual"
    )
    tables = [seed_table(month=month, seed=s, sites=sites, actual=actual) for s in range(4)]
    parts = [rows]
    for s in (2, 3):
        parts.append(r521.score_cells(rows=rows, cells=[tables[s]], name=f"L-1d-seed{s}"))
    pooled = r521.pooled_seeds(tables=tables)
    parts.append(r521.score_cells(rows=rows, cells=[pooled], name="L-1d-4q"))
    base = pooled.select(*KEYS, "actual")
    aligned = [
        base.select(*KEYS)
        .join(t.select(*KEYS, *LEVELS), on=KEYS, how="left", maintain_order="left")
        .select(list(LEVELS))
        .to_numpy()
        .astype(np.float64)
        for t in tables
    ]
    ok = np.all([np.isfinite(a).all(axis=1) for a in aligned], axis=0)
    mixed = np.full_like(aligned[0], np.nan)
    mixed[ok] = mix_quantiles(tables=[a[ok] for a in aligned])
    parts.append(
        r521.score_cells(rows=rows, cells=[with_levels(base=base, q=mixed)], name="L-1d-4m")
    )
    return pl.concat(parts, how="horizontal")


def run(*, which: str) -> None:
    """Write `work/{which}_{month}.parquet` for each month."""
    torch.set_num_threads(1)
    sites = pl.read_csv(r521.DATA / "site_map.csv").with_columns(pl.col("primary").cast(pl.Int16))
    for month in r521.MONTHS:
        frame = old_month(month=month) if which == "old" else seeds_month(month=month, sites=sites)
        frame.write_parquet(WORK / f"{which}_{month}.parquet")
        print(which, month, frame.height, flush=True)


# ---- statistics -------------------------------------------------------------------------------

REF: Final = "K2-S1"
"""Reference arm: experiment 511's K2-S1 column (cell c0), as in experiment 521. The 3-cell mean is reported beside it."""
ARMS: Final[tuple[str, ...]] = (
    "K2-S1", "K2-S1-3cell", "L-1d", "L-1d-4q", "L-1d-4m", "LL", "LL-ens2", "LL-ens3", "LL-ens4",
    "LL-ens5", "S1-ens2", "L-Ww",
)  # fmt: skip
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("L-1d", REF, "base: 2-seed daily lag-free minus K2-S1"),
    ("L-1d-4q", REF, "H1: 4-seed daily lag-free (quantile) minus K2-S1"),
    ("L-1d-4m", REF, "H1: 4-seed daily lag-free (approx. mixture) minus K2-S1"),
    ("LL-ens2", REF, "H3: K2-LL, 2 lag-free heads only, minus K2-S1"),
    ("LL-ens4", REF, "H3: K2-LL, 4 lag-free heads, minus K2-S1"),
    ("S1-ens2", REF, "report: lagged heads alone minus K2-S1"),
    ("LL-ens4", "L-1d-4q", "H2: monthly 4-head minus daily 4-head (quantile)"),
    ("LL-ens4", "L-1d-4m", "H2: monthly 4-head minus daily 4-head (approx. mixture)"),
    ("L-1d-4q", "L-1d", "head count at daily cadence: 4 minus 2 seeds"),
    ("K2-S1-3cell", REF, "check: 3-cell K2-S1 minus cell c0"),
)
"""(A, B, label): A minus B, positive = B better. The first rows through H3 are counted tests."""
COUNTED: Final[int] = (
    4  # H1 quantile, H1 mixture, H3 (2 heads), H3 (4 heads): x 7 bands x 2 sets = 56
)


def load() -> pl.DataFrame:
    """Return 511's rows joined with 521's reference scores and this experiment's scores."""
    base = s521.load()
    extra = []
    for month in r521.MONTHS:
        old = pl.read_parquet(WORK / f"old_{month}.parquet").drop(KEYS)
        new = pl.read_parquet(WORK / f"seeds_{month}.parquet").drop(KEYS)
        extra.append(pl.concat([old, new], how="horizontal"))
    table = pl.concat([base, pl.concat(extra)], how="horizontal")
    return table.with_columns(band=s521.band_expr())


def seed_mde(*, part: pl.DataFrame, cols: list[str]) -> float:
    """Return `1.96 sqrt(2) sd / sqrt(k)` of the per-fold means of `cols` (experiment 470's pooling)."""
    means = part.group_by("month").agg([pl.col(c).mean() for c in cols]).sort("month")
    values = means.select(cols).to_numpy()
    folds, k = values.shape
    resid = values - values.mean(axis=1, keepdims=True)
    sd = math.sqrt(float((resid**2).sum()) / (folds * (k - 1)))
    return 1.96 * math.sqrt(2.0) * sd / math.sqrt(k)


def stats() -> None:
    """Write the absolute and contrast tables."""
    SCORES.mkdir(parents=True, exist_ok=True)
    table = load()
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    n_counted = COUNTED * 7 * 2
    for name in ("eight", "six", "six_all", "report"):
        part = s521.row_set(table=table, name=name)
        no_event = part.filter(~pl.col("tgt_real"))
        for band in (*s521.BANDS, s521.ALL):
            sub = s521.in_band(part=part, band=band)
            for arm in ARMS:
                absolute.append(
                    {"set": name, "band": band, "arm": arm, "n": sub.height,
                     "crps": float(sub[f"{arm}__crps"].mean())}
                )  # fmt: skip
            mde_2 = seed_mde(part=sub, cols=["L-1d-seed0__crps", "L-1d-seed1__crps"])
            mde_4 = seed_mde(part=sub, cols=[f"L-1d-seed{s}__crps" for s in range(4)])
            for a, b, label in CONTRASTS:
                for rows_name, rows in (
                    ("all rows", sub),
                    ("tgt_real false", s521.in_band(part=no_event, band=band)),
                ):
                    diff = rows.select(
                        "month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps")
                    )
                    diff = diff.filter(pl.col("d").is_finite())
                    contrasts.append(
                        {"set": name, "band": band, "rows": rows_name, "contrast": f"{a} - {b}",
                         "label": label, "mde_2seed": mde_2, "mde_4seed": mde_4,
                         **s521.block_stats(diff=diff, rng=rng, tests=n_counted)}
                    )  # fmt: skip
    pl.DataFrame(absolute).write_csv(SCORES / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(SCORES / "contrasts.csv")
    (SCORES / "counts.json").write_text(json.dumps({"bonferroni_counted_tests": n_counted}))
    print("wrote tables; counted tests", n_counted)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("old", "seeds", "stats"))
    args = parser.parse_args()
    if args.command == "stats":
        stats()
    else:
        run(which=args.command)


if __name__ == "__main__":
    main()

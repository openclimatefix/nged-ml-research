"""Experiment 448, post hoc stage: experiment 435's table P applied to S1 and its five-seed pool.

Nothing is fitted. The factors are experiment 435's stored `table_p_factors_full_428.parquet`, arm
`R-inj`, which that experiment fitted on the forecasts of experiment 428's arm `R-inj` in the folds
before each scored month. A factor multiplies the distance of one tail quantile (q01, q05, q95, or
q99) from the median, per condition class, metering group, and lead band. Demand quantiles are not
clipped, as in experiment 435. Each row takes the factors of its own condition class, assigned by
experiment 435's `add_classes` (real event, lag missing, surprise above 4, surprise 2 to 4,
injected harm, clean).

Rows are experiment 448's `clean` row set: world 0, no real event in the three windows, target
outside the padded event mask and the 96 half-hours after it, lead above 0 and up to 240 hours.
Months are the ones experiment 435 scored for its 428 tables (2026-01, 2026-03, and 2026-06),
because table P for a month needs the earlier folds. Scopes:

- `clean_448`: all those rows;
- `clean_435_class`: the rows whose condition class is `clean`, experiment 435's headline class;
- `cold_nights_425`: `clean_448` rows flagged `cold_nights` by `studies.eval_protocol.slices`
  (sun more than 5 degrees below the horizon, ERA5 temperature at or below 3 degrees Celsius);
- `cold_winter_night_441`: `clean_448` rows in experiment 441's `cold_winter_night` mask, the slice
  experiment 435 used.

Variants are `raw`, `w1.15` (every spread about the median multiplied by 1.15), and `P` (table P,
applied to the raw quantiles). Every variant is sorted across the 11 levels. Scores follow
experiment 428: CRPS (points of effective capacity, the quantile score at the 11 levels) is the mean
of the per-seed scores, and the tail coverage and pinball loss are read on the seed-mean quantiles.
`S1` uses seeds 0 to 4; `S1-pool5` is one stored file (its five seed files are identical copies).

```bash
source /tmp/e448_env.sh
systemd-run --user --scope -q -p MemoryMax=10G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00448-*/e00448_table_p.py [--check]
```
"""

import argparse
from typing import Final

import e00428_score as score  # ty: ignore[unresolved-import]
import e00435_table_p as t435  # ty: ignore[unresolved-import]
import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import e00435_tailcore as tailcore  # ty: ignore[unresolved-import]
import e00448_refs as refs  # ty: ignore[unresolved-import]
import e00448_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PRED_DIR = run.WORK_DIR / "predictions"
TABLE_PATH = run.WORK_DIR / "figure_tables" / "table_p_stage.csv"
FACTORS_PATH = prov.WORK_DIR / "table_p_factors_full_428.parquet"
FACTOR_ARM: Final[str] = "R-inj"
MASKS_441 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00441-difficult-row-rescore-of-the-neural-demand-input-bundles"
    / "work"
    / "masks.parquet"
)
P_MONTHS: Final[tuple[str, ...]] = ("2026-01", "2026-03", "2026-06")
ARM_SEEDS: Final[dict[str, tuple[int, ...]]] = {"S1": (0, 1, 2, 3, 4), "S1-pool5": (0,)}
VARIANTS: Final[tuple[str, ...]] = ("raw", "w1.15", "P")
LEVELS: Final[tuple[str, ...]] = tuple(refs.LEVEL_NAMES)
TAILS: Final[tuple[str, ...]] = tuple(refs.TAILS)
POINTS: Final[float] = refs.POINTS
GROUPS: Final[tuple[str, ...]] = ("signed-MW six", "MVA-only eight", "D15 and D18")
METRICS: Final[tuple[str, ...]] = (
    "crps", "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99",
    "pinball_q95", "pinball_q99",
)  # fmt: skip


def read_month(
    *, arm: str, month: str, seeds: tuple[int, ...], powers: np.ndarray
) -> tuple[pl.DataFrame, list[np.ndarray]]:
    """Return the clean world-0 rows with classes and keys, and one raw quantile matrix per seed."""
    files = [pl.read_parquet(PRED_DIR / arm / f"{month}_s{s}_w0.parquet") for s in seeds]
    keys = ["primary", "origin_us", "valid_us"]
    for other in files[1:]:
        assert other.select(keys).equals(files[0].select(keys)), "seed files differ in row order"
    median, width = t435.lagless_dense(month=month)
    classed = t435.add_classes(frame=files[0], power=powers, median=median, width=width)
    sites = refs.sites_table().select("primary", "time_series_id", "label")
    rows = (
        classed.with_row_index("_r")
        .with_columns(
            lead_hours=pl.col("lead").cast(pl.Float64),
            actual=pl.col("actual").cast(pl.Float64),
        )
        .with_columns(lead_band=tailcore.lead_band_expr())
        .filter(
            ~pl.col("win_real")
            & ~pl.col("tgt_post")
            & pl.col("actual").is_finite()
            & pl.col("lead_band").is_not_null()
        )
        .join(sites, on="primary")
        .with_columns(
            group=pl.col("label").replace_strict(prov.GROUP_OF, return_dtype=pl.String),
            power_fcst_init_time=pl.from_epoch("origin_us", time_unit="us").dt.replace_time_zone(
                "UTC"
            ),
            valid_time=pl.from_epoch("valid_us", time_unit="us").dt.replace_time_zone("UTC"),
            time_series_id=pl.col("time_series_id").cast(pl.Int32),
        )
        .sort("_r")
    )
    index = rows["_r"].to_numpy()
    mats = [f.select(list(LEVELS)).to_numpy().astype(np.float64)[index] for f in files]
    assert all(np.isfinite(m).all() for m in mats), "non-finite stored quantile"
    return rows, mats


def apply_table_p(*, rows: pl.DataFrame, q: np.ndarray, factors: pl.DataFrame) -> np.ndarray:
    """Return `q` with each row's tail quantiles scaled by the factors of its own class."""
    out = q.copy()
    classes = rows["cls"].to_numpy()
    for cls in np.unique(classes):
        index = np.flatnonzero(classes == cls)
        part = (
            rows[index]
            .select("group", "lead_band")
            .with_columns(
                _i=pl.Series(np.arange(len(index))),
                **{name: pl.Series(q[index, i]) for i, name in enumerate(LEVELS)},
            )
        )
        cell = factors.filter(pl.col("cls") == cls).select("group", "lead_band", "tail", "factor")
        scaled = prov.apply_factors(frame=part, factors=cell).sort("_i")
        for tail in tailcore.TAIL_LEVELS:
            out[index, LEVELS.index(tail)] = scaled[tail].to_numpy()
    return out


def variant_matrices(
    *, variant: str, rows: pl.DataFrame, mats: list[np.ndarray], factors: pl.DataFrame
) -> list[np.ndarray]:
    """Return the sorted quantile matrix of each seed under one variant."""
    if variant == "raw":
        out = mats
    elif variant == "w1.15":
        out = [score.widened(q=m, factor=score.WIDENING) for m in mats]
    else:
        out = [apply_table_p(rows=rows, q=m, factors=factors) for m in mats]
    return [np.sort(m, axis=1) for m in out]


def row_metrics(*, actual: np.ndarray, mats: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Return per-row metrics: CRPS as the mean of seed scores, tails on the seed-mean quantiles."""
    crps = np.mean([refs.row_scores(actual=actual, q=m)["crps"] for m in mats], axis=0)
    return refs.row_scores(actual=actual, q=np.mean(mats, axis=0)) | {"crps": crps}


def cold_flags(*, rows: pl.DataFrame, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the `cold_nights` (protocol) and `cold_winter_night` (experiment 441) row flags."""
    protocol = refs.slice_flags(rows=rows, ctx=ctx).select(*refs.KEYS, "cold_nights")
    masks = (
        pl.read_parquet(MASKS_441)
        .select("power_fcst_init_time", "valid_time", "label", "cold_winter_night")
        .unique(subset=["power_fcst_init_time", "valid_time", "label"])
    )
    joined = (
        rows.select("_r", *refs.KEYS, "label")
        .join(protocol, on=list(refs.KEYS), how="left")
        .join(masks, on=["power_fcst_init_time", "valid_time", "label"], how="left")
        .sort("_r")
    )
    return (
        joined["cold_nights"].fill_null(value=False).to_numpy(),
        joined["cold_winter_night"].fill_null(value=False).to_numpy(),
    )


def summarise(
    *, rows: pl.DataFrame, metrics: dict[str, np.ndarray], scopes: dict[str, np.ndarray]
) -> list[dict]:
    """Return one record per scope and group: row count and the mean of each metric."""
    frame = rows.select("group").with_columns(**{k: pl.Series(v) for k, v in metrics.items()})
    out: list[dict] = []
    for scope, mask in scopes.items():
        part = (
            frame.filter(pl.Series(mask))
            .group_by("group")
            .agg(pl.len().alias("n"), *[pl.col(m).mean() for m in METRICS])
        )
        out += [{"scope": scope, **r} for r in part.iter_rows(named=True)]
    return out


def main() -> None:
    """Score S1 and the pool raw, widened, and after table P; write the table."""
    powers = t435.world_powers()[0]
    factors_all = pl.read_parquet(FACTORS_PATH).filter(pl.col("arm") == FACTOR_ARM)
    ctx = refs.prep425.cleaned_scoring_context(series=refs.adapter.series_table())
    records: list[dict] = []
    for month in P_MONTHS:
        factors = factors_all.filter(pl.col("month") == month)
        for arm, seeds in ARM_SEEDS.items():
            rows, mats = read_month(arm=arm, month=month, seeds=seeds, powers=powers)
            cold, cold_441 = cold_flags(rows=rows, ctx=ctx)
            everything = np.ones(rows.height, dtype=bool)
            scopes = {
                "clean_448": everything,
                "clean_435_class": (rows["cls"] == "clean").to_numpy(),
                "cold_nights_425": cold,
                "cold_winter_night_441": cold_441,
            }
            actual = rows["actual"].to_numpy()
            for variant in VARIANTS:
                matrices = variant_matrices(variant=variant, rows=rows, mats=mats, factors=factors)
                if variant == "P":
                    unsorted = apply_table_p(rows=rows, q=mats[0], factors=factors)
                    crossing = float((np.diff(unsorted, axis=1).min(axis=1) < 0).mean())
                    print(f"{arm} {month}: crossing rows before sorting: {crossing:.3f}")
                metrics = row_metrics(actual=actual, mats=matrices)
                records += [
                    {"arm": arm, "variant": variant, "month": month, **r}
                    for r in summarise(rows=rows, metrics=metrics, scopes=scopes)
                ]
            print("done", arm, month, flush=True)
    monthly = pl.DataFrame(records)
    pooled = (
        monthly.group_by("arm", "variant", "scope", "group")
        .agg(
            pl.col("n").sum(),
            *[(pl.col(m) * pl.col("n")).sum() / pl.col("n").sum() for m in METRICS],
        )
        .with_columns(month=pl.lit("pooled"))
    )
    table = pl.concat([monthly, pooled.select(monthly.columns)]).sort(
        "scope", "group", "arm", "variant", "month"
    )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(TABLE_PATH)
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=200):
        print(table.filter(pl.col("month") == "pooled").drop("month"))


def check() -> None:
    """Reproduce experiment 435's R-inj clean-class P coverage, and compare this script's rows.

    Experiment 435's own `load_arm` (four worlds, three seeds pooled) is run on 2026-06 against
    experiment 448's R-inj forecast files and scored through `prov.apply_factors`. Its q95 coverage
    by group is compared with the stored `table_p_428.parquet`, then with this script's world-0
    rows of the same class.
    """
    month, arm, seeds = "2026-06", "R-inj", (0, 1, 2)
    t435.PRED_DIR = PRED_DIR
    powers = t435.world_powers()
    frame = t435.load_arm(arm=arm, month=month, powers=powers)
    factors_all = pl.read_parquet(FACTORS_PATH).filter(
        (pl.col("arm") == arm) & (pl.col("month") == month)
    )
    clean = frame.filter(pl.col("cls") == "clean")
    p_frame = prov.apply_factors(frame=clean, factors=factors_all.filter(pl.col("cls") == "clean"))
    ours = tailcore.tail_table(frame=p_frame, by=("group",)).select(
        "group", "n", ours_cov_q95="cov_q95"
    )
    stored = (
        pl.read_parquet(prov.WORK_DIR / "table_p_428.parquet")
        .filter(
            (pl.col("arm") == arm)
            & (pl.col("month") == month)
            & (pl.col("cls") == "clean")
            & (pl.col("variant") == "P")
        )
        .group_by("group")
        .agg(stored_cov_q95=(pl.col("cov_q95") * pl.col("n")).sum() / pl.col("n").sum())
    )
    print(ours.join(stored, on="group"))
    rows, mats = read_month(arm=arm, month=month, seeds=seeds, powers=powers[0])
    rows_clean = rows.filter(pl.col("cls") == "clean")
    keep = np.flatnonzero((rows["cls"] == "clean").to_numpy())
    q = np.mean([apply_table_p(rows=rows, q=m, factors=factors_all)[keep] for m in mats], axis=0)
    cov = (rows_clean["actual"].to_numpy() <= q[:, LEVELS.index("q95")]).astype(float)
    world0 = frame.filter((pl.col("cls") == "clean") & (pl.col("world") == 0) & ~pl.col("win_real"))
    mine = (
        pl.DataFrame({"group": rows_clean["group"], "cov": cov})
        .group_by("group")
        .agg(n=pl.len(), this_script_cov_q95=pl.col("cov").mean())
    )
    p0 = prov.apply_factors(frame=world0, factors=factors_all.filter(pl.col("cls") == "clean"))
    ref0 = tailcore.tail_table(frame=p0, by=("group",)).select(
        "group", n_435_loader="n", loader_435_cov_q95="cov_q95"
    )
    print(mine.join(ref0, on="group"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="reproduce experiment 435's numbers")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        main()

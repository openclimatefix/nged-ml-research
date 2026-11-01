r"""Score experiment 491's weight-averaged fine-tunes against experiment 425's stored fits.

Commands, run in order:

- `build`: write each arm's cells on experiment 454's rows (the layout of experiment 470) under
  `work/predictions/<arm>/<month>_c<cell>.parquet`. Stored references are copied or averaged from
  experiment 425's arm A seeds 0 to 4 (`S0`, `Pool2`, `Pool3`, `Pool5`); the new arms are read from
  this experiment's `seed_preds`.
- `rows`: write the row-level metrics of every arm (cells averaged, 1.15 widening) beside the slice
  flags of experiment 470, restricted to experiment 470's `gam14` rows.
- `stats`: print the minimum detectable effect and the gaps from S0f and S0 to Pool5 first, then the
  absolute scores, the contrasts with month-block bootstrap intervals (Bonferroni count `TESTS`),
  the closure shares, and the tails.
- `widening`: the mean CRPS of each arm at widenings 1.0 to 1.15 on the eight and the six.

```bash
source /tmp/e491env.sh
uv run --no-sync python ml-experiments/00491-*/e00491_score.py build
uv run --no-sync python ml-experiments/00491-*/e00491_score.py rows
uv run --no-sync python ml-experiments/00491-*/e00491_score.py stats
```
"""

import argparse
import math
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00491_fit as fit491  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import TAIL_LEVELS  # ty: ignore[unresolved-import]

MONTHS: Final[tuple[str, ...]] = fit491.FOLDS
KEYS: Final[list[str]] = build470.KEYS
LEVELS: Final[tuple[str, ...]] = build470.LEVELS
SEEDS: Final[tuple[int, ...]] = fit491.SEEDS
WORK: Final[Path] = fit491.WORK
SCORE_DIR: Final[Path] = WORK / "scores"
PRED_DIR: Final[Path] = WORK / "predictions"
SINGLES: Final[tuple[str, ...]] = tuple(f"A-s{s}" for s in build470.POOL_SEEDS)
NEW_ARMS: Final[tuple[str, ...]] = ("EMA", "SWA")
FIXED_BUDGET: Final[str] = "S0f"
SINGLE_SEED: Final[str] = "S0"
POOL_REFERENCES: Final[tuple[str, ...]] = ("Pool2", "Pool3", "Pool5")
POOL_OF_NEW: Final[dict[str, str]] = {"PoolS0f3": "S0f", "PoolEMA3": "EMA", "PoolSWA3": "SWA"}
PRE_ARMS: Final[tuple[str, ...]] = ("P0f", "Pswa", "Pswa+EMA", "Pswa+SWA")
PRE_MONTH: Final[str] = fit491.SWA_PRE_FOLD
MAIN_ARMS: Final[tuple[str, ...]] = (
    SINGLE_SEED, FIXED_BUDGET, *NEW_ARMS, "EMA99", *POOL_REFERENCES, *POOL_OF_NEW,
)  # fmt: skip
CELLS: Final[dict[str, int]] = {
    SINGLE_SEED: 3, FIXED_BUDGET: 3, "EMA": 3, "EMA99": 3, "SWA": 3, "Pool2": 3,
    **dict.fromkeys(PRE_ARMS, 3),
}  # fmt: skip
PAIRS: Final[tuple[tuple[int, int], ...]] = ((0, 1), (0, 2), (1, 2))
CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (a, b) for b in (FIXED_BUDGET, "Pool2", "Pool5") for a in NEW_ARMS
)
DESCRIPTIVE: Final[tuple[tuple[str, str], ...]] = (
    (FIXED_BUDGET, SINGLE_SEED), ("EMA", SINGLE_SEED), ("SWA", SINGLE_SEED),
    ("EMA99", FIXED_BUDGET),
    ("PoolEMA3", "Pool3"), ("PoolSWA3", "Pool3"), ("PoolS0f3", "Pool3"),
    ("Pool5", SINGLE_SEED), ("Pool2", SINGLE_SEED),
)  # fmt: skip
TESTS: Final[int] = 54
GATING_CRPS: Final[tuple[str, ...]] = score470.GATING_CRPS
GATING_Q99: Final[tuple[str, ...]] = score470.GATING_Q99
assert len(CONTRASTS) * (len(GATING_CRPS) + len(GATING_Q99)) == TESTS
WIDENINGS: Final[tuple[float, ...]] = (1.0, 1.05, 1.10, 1.15)
MDE_SLICES: Final[dict[str, tuple[str, str]]] = {
    "eight": ("mva_eight", "all"),
    "six (clean rows)": ("signed_six", "clean"),
    "cold_eight": ("mva_eight", "cold"),
    "cold_six": ("signed_six", "cold"),
}


def cells_of(*, arm: str) -> int:
    """Return the number of stored cells of an arm."""
    return CELLS.get(arm, 1)


def pred_path(*, arm: str, month: str, cell: int) -> Path:
    """Return where an arm's cell is stored on experiment 454's rows."""
    return PRED_DIR / arm / f"{month}_c{cell}.parquet"


def write_cell(*, arm: str, month: str, cell: int, frame: pl.DataFrame) -> None:
    """Write one cell's keys and quantiles."""
    target = pred_path(arm=arm, month=month, cell=cell)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.select(*KEYS, *LEVELS).write_parquet(target)


def build_command() -> None:
    """Write every arm's cells on experiment 454's rows."""
    sites = build470.site_map()
    stored = build470.heads470.WORK_DIR / "predictions"
    a425 = build470.E425 / "seed_preds" / "A"
    for month in MONTHS:
        rows = pl.read_parquet(stored / "A-s0" / f"{month}_c0.parquet").select(*KEYS)
        for seed in SEEDS:
            frame = pl.read_parquet(stored / f"A-s{seed}" / f"{month}_c0.parquet")
            write_cell(arm=SINGLE_SEED, month=month, cell=seed, frame=frame)
        write_cell(
            arm="Pool5", month=month, cell=0,
            frame=pl.read_parquet(stored / "A-pool" / f"{month}_c0.parquet"),
        )  # fmt: skip
        for cell, pair in enumerate(PAIRS):
            one = build470.pool_stored(folder=a425, month=month, sites=sites, seeds=pair)
            write_cell(
                arm="Pool2", month=month, cell=cell, frame=build470.on_rows(rows=rows, frame=one)
            )
        three = build470.pool_stored(folder=a425, month=month, sites=sites, seeds=SEEDS)
        write_cell(arm="Pool3", month=month, cell=0, frame=build470.on_rows(rows=rows, frame=three))
        for arm in (FIXED_BUDGET, *NEW_ARMS, "EMA99"):
            for seed in SEEDS:
                one = build470.pool_stored(
                    folder=WORK / "seed_preds" / arm, month=month, sites=sites, seeds=(seed,)
                )
                write_cell(
                    arm=arm, month=month, cell=seed, frame=build470.on_rows(rows=rows, frame=one)
                )
        for arm, source in POOL_OF_NEW.items():
            pooled = build470.pool_stored(
                folder=WORK / "seed_preds" / source, month=month, sites=sites, seeds=SEEDS
            )
            write_cell(
                arm=arm, month=month, cell=0, frame=build470.on_rows(rows=rows, frame=pooled)
            )
        if month == PRE_MONTH:
            for arm in PRE_ARMS:
                for seed in SEEDS:
                    one = build470.pool_stored(
                        folder=WORK / "seed_preds" / arm, month=month, sites=sites, seeds=(seed,)
                    )
                    write_cell(
                        arm=arm,
                        month=month,
                        cell=seed,
                        frame=build470.on_rows(rows=rows, frame=one),
                    )
        print(f"{month}: wrote arms", flush=True)


def arms_of(*, month: str) -> tuple[str, ...]:
    """Return the arms stored for a month (the pretraining arms exist for one fold)."""
    return (*MAIN_ARMS, *PRE_ARMS) if month == PRE_MONTH and _pre_stored() else MAIN_ARMS


def _pre_stored() -> bool:
    """Return whether the pretraining arms have been built."""
    return pred_path(arm=PRE_ARMS[0], month=PRE_MONTH, cell=0).exists()


def month_rows(*, month: str, base: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Return a month's row-level metrics of every arm (cells averaged) at one widening."""
    cols: dict[str, np.ndarray] = {}
    ok_all = np.ones(base.height, dtype=bool)
    for arm in arms_of(month=month):
        cells = []
        for cell in range(cells_of(arm=arm)):
            frame = base.select(*KEYS, "actual", "nra", "tgt_real").join(
                pl.read_parquet(pred_path(arm=arm, month=month, cell=cell)),
                on=KEYS,
                how="left",
                maintain_order="left",
            )
            frame = score470.widen_arm(arm=arm, frame=frame, factor=factor)
            ok = frame["q50"].is_not_null().to_numpy() & np.isfinite(frame["q50"].to_numpy())
            frame = frame.with_columns(*[pl.col(c).fill_null(0.0) for c in LEVELS])
            metered = score470.row_metrics(frame=frame)
            nra = score470.row_metrics(
                frame=frame.with_columns(actual_nra=pl.col("actual") - pl.col("nra")),
                actual="actual_nra",
            )
            if arm not in PRE_ARMS:
                ok_all &= ok
            cells.append(
                (
                    {k: np.where(ok, v, np.nan) for k, v in metered.items()},
                    {k: np.where(ok, v, np.nan) for k, v in nra.items()},
                )
            )
        for metric in cells[0][0]:
            cols[f"{arm}__{metric}"] = np.mean([c[0][metric] for c in cells], axis=0)
            cols[f"{arm}__nra_{metric}"] = np.mean([c[1][metric] for c in cells], axis=0)
    out = base.select(*KEYS).hstack(
        pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items()})
    )
    return out.hstack(
        base.select(
            "group", "cls", "lead_hours", "lead_band", "tgt_real", "cold", "xmas", "band", "day"
        )
    ).with_columns(month=pl.lit(month), all_arms=pl.Series(ok_all))


def rows_command(*, factor: float = score470.HEADLINE, tag: str = "rows") -> None:
    """Write the row-level table, restricted to experiment 470's gam14 rows."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    gam = score470.load_rows().select(*KEYS, "month", "gam14", *[f"{a}__crps" for a in SINGLES])
    for month in MONTHS:
        base = score470.month_context(month=month, sites=sites, ctx=ctx)
        table = month_rows(month=month, base=base, factor=factor)
        table = table.join(
            gam.filter(pl.col("month") == month).drop("month"),
            on=KEYS,
            how="left",
            maintain_order="left",
        )
        table = table.filter(pl.col("gam14").fill_null(value=False) & pl.col("all_arms"))
        table.write_parquet(SCORE_DIR / f"{tag}_{month}.parquet")
        print(f"{month}: base rows {base.height}, scored rows {table.height}", flush=True)


def load_rows(*, tag: str = "rows") -> pl.DataFrame:
    """Return the row-level table of all months."""
    return pl.concat(
        [pl.read_parquet(SCORE_DIR / f"{tag}_{m}.parquet") for m in MONTHS], how="diagonal"
    )


def slice_rows(*, table: pl.DataFrame, name: str) -> tuple[pl.DataFrame, str]:
    """Return the rows of one of experiment 470's slices and the metric prefix."""
    return score470.slice_filter(table=table, name=name)


def mde_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 428's minimum detectable effect from arm A's five single seeds.

    The effect is for a contrast of two three-seed means, per slice, so the divisor is the square
    root of 3 (the cells an arm averages). The slices include the cold winter nights.
    """
    records = []
    for label, (group, kind) in MDE_SLICES.items():
        part = table.filter(pl.col("group") == group)
        if kind == "clean":
            part = part.filter(pl.col("cls") == "clean")
        elif kind == "cold":
            part = part.filter(pl.col("cold"))
        columns = [f"{a}__crps" for a in SINGLES]
        values = np.array(
            [
                [part.filter(pl.col("month") == m)[c].mean() for c in columns]
                for m in MONTHS
                if part.filter(pl.col("month") == m).height
            ]
        )
        folds, k = values.shape
        resid = values - values.mean(axis=1, keepdims=True)
        sd = math.sqrt((resid**2).sum() / (folds * (k - 1)))
        records.append(
            {
                "slice": label,
                "seed_sd": sd,
                "mde_single_seed": 1.96 * math.sqrt(2.0) * sd,
                "mde_three_seed_mean": 1.96 * math.sqrt(2.0) * sd / math.sqrt(3.0),
            }
        )
    return pl.DataFrame(records)


def closure(
    *,
    table: pl.DataFrame,
    arm: str,
    sname: str,
    rng: np.random.Generator,
    start: str = FIXED_BUDGET,
) -> dict:
    """Return the share of the `start`-to-Pool5 gap an arm closes, with a block interval."""
    part, prefix = slice_rows(table=table, name=sname)
    frame = part.select(
        "month",
        "day",
        gap=pl.col(f"{start}__{prefix}crps") - pl.col(f"Pool5__{prefix}crps"),
        gain=pl.col(f"{start}__{prefix}crps") - pl.col(f"{arm}__{prefix}crps"),
    ).drop_nulls()
    blocks = frame.group_by("month", "day").agg(
        pl.len().alias("n"), pl.col("gap").sum().alias("g"), pl.col("gain").sum().alias("s")
    )
    n, g, s = (blocks[c].to_numpy().astype(float) for c in ("n", "g", "s"))
    draws = rng.integers(0, len(n), (score470.RESAMPLES, len(n)))
    ratio = s[draws].sum(1) / g[draws].sum(1)
    return {
        "arm": arm,
        "start": start,
        "slice": sname,
        "gap_points": float(g.sum() / n.sum()),
        "gain_points": float(s.sum() / n.sum()),
        "closure_share": float(s.sum() / g.sum()),
        "low": float(np.percentile(ratio, 2.5)),
        "high": float(np.percentile(ratio, 97.5)),
    }


def stats_command() -> None:
    """Print the minimum detectable effect and the gap first, then scores, contrasts, closure."""
    table = load_rows()
    minimum = mde_table(table=table)
    print("minimum detectable effect (printed before any contrast):")
    print(minimum)
    minimum.write_csv(SCORE_DIR / "mde.csv")
    rng = np.random.default_rng(0)
    gaps = [
        closure(table=table, arm="Pool5", sname=s, rng=rng, start=start)
        for start in (FIXED_BUDGET, SINGLE_SEED)
        for s in ("eight", "six_clean", "cold_eight", "cold_six")
    ]
    print("gap from S0f and from S0 to Pool5 in points:")
    print(pl.DataFrame(gaps).select("start", "slice", "gap_points"))
    metrics = ["crps", *[f"pin_{t}" for t in TAIL_LEVELS], *[f"cov_{t}" for t in TAIL_LEVELS]]
    arms = [c.removesuffix("__crps") for c in table.columns if c.endswith("__crps")]
    arms = [a for a in arms if a not in SINGLES]
    absolute = []
    for sname in score470.SLICES:
        part, prefix = slice_rows(table=table, name=sname)
        if part.height == 0:
            continue
        absolute.extend(
            {"slice": sname, "arm": arm, "n": part.height}
            | {m: part[f"{arm}__{prefix}{m}"].mean() for m in metrics}
            for arm in arms
        )
    pl.DataFrame(absolute).write_csv(SCORE_DIR / "absolute.csv")
    score470.TESTS = TESTS
    records = []
    for a, b in (*CONTRASTS, *DESCRIPTIVE):
        counted_pair = (a, b) in CONTRASTS
        for sname in (*GATING_CRPS, "xmas_eight", "xmas_six", "report"):
            for metric in (
                "crps",
                *(
                    [f"pin_{t}" for t in TAIL_LEVELS]
                    if sname in ("eight", "six_clean", "report")
                    else []
                ),
            ):
                part, prefix = slice_rows(table=table, name=sname)
                diff = (
                    part.select(
                        "month",
                        "day",
                        d=pl.col(f"{a}__{prefix}{metric}") - pl.col(f"{b}__{prefix}{metric}"),
                    )
                    .drop_nulls()
                    .drop_nans()
                )
                if diff.height == 0:
                    continue
                records.append(
                    {
                        "contrast": f"{a} - {b}",
                        "slice": sname,
                        "metric": metric,
                        "counted": counted_pair
                        and (
                            (sname in GATING_CRPS and metric == "crps")
                            or (sname in GATING_Q99 and metric == "pin_q99")
                        ),
                        **score470.block_bootstrap(diff=diff, rng=rng),
                    }
                )
    contrasts = pl.DataFrame(records)
    contrasts.write_csv(SCORE_DIR / "contrasts.csv")
    print(f"counted tests: {int(contrasts['counted'].sum())} (pre-registered {TESTS})")
    shares = [
        closure(table=table, arm=arm, sname=s, rng=rng, start=start)
        for start in (FIXED_BUDGET, SINGLE_SEED)
        for arm in (*NEW_ARMS, "EMA99", "Pool2", "PoolEMA3", "PoolSWA3")
        for s in ("eight", "six_clean", "cold_eight", "cold_six")
    ]
    pl.DataFrame(shares).write_csv(SCORE_DIR / "closure.csv")
    with pl.Config(tbl_rows=300, tbl_width_chars=220):
        print(pl.DataFrame(shares).with_columns(pl.col(pl.Float64).round(3)))
        for metric in ("crps", "pin_q99"):
            print(
                contrasts.filter(pl.col("metric") == metric)
                .drop("by_month")
                .with_columns(pl.col(pl.Float64).round(3))
            )


def widening_command() -> None:
    """Write the mean CRPS and the q99 and q95 coverage of each arm at each widening."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    records = []
    for factor in WIDENINGS:
        parts = []
        for month in MONTHS:
            base = score470.month_context(month=month, sites=sites, ctx=ctx)
            table = month_rows(month=month, base=base, factor=factor)
            wanted = score470.load_rows().filter(pl.col("month") == month)
            table = table.join(
                wanted.select(*KEYS, "gam14"), on=KEYS, how="left", maintain_order="left"
            )
            parts.append(table.filter(pl.col("gam14").fill_null(value=False) & pl.col("all_arms")))
        table = pl.concat(parts, how="diagonal")
        for sname in ("eight", "six_clean"):
            part, prefix = slice_rows(table=table, name=sname)
            records.extend(
                {"factor": factor, "slice": sname, "arm": arm}
                | {
                    m: part[f"{arm}__{prefix}{m}"].mean()
                    for m in ("crps", "pin_q99", "cov_q99", "cov_q95")
                }
                for arm in MAIN_ARMS
            )
    out = pl.DataFrame(records)
    out.write_csv(SCORE_DIR / "widening.csv")
    best = out.sort("crps").group_by("slice", "arm").first().sort("slice", "arm")
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(best.with_columns(pl.col(pl.Float64).round(3)))


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "rows", "stats", "widening"))
    args = parser.parse_args()
    {
        "build": build_command,
        "rows": rows_command,
        "stats": stats_command,
        "widening": widening_command,
    }[args.command]()


if __name__ == "__main__":
    main()

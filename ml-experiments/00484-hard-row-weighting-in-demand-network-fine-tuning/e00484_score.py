r"""Score experiment 484's weighted fine-tunes against experiment 425's stored fits.

Three commands, run in order:

- `build`: write each arm's three seed cells on experiment 454's rows (the layout of experiment
  470) under `work/predictions/<arm>/<month>_c<seed>.parquet`. The arm `W0` is experiment 470's
  `A-s0`, `A-s1`, and `A-s2` (experiment 425's stored arm A fits, seeds 0 to 2) copied unchanged.
- `rows`: write the row-level metrics of every arm (cells averaged, 1.15 widening) beside the
  slice flags of experiment 470, restricted to experiment 470's `gam14` rows.
- `stats`: print the minimum detectable effect first, then the absolute scores, the contrasts
  against `W0` with month-block bootstrap intervals (Bonferroni count `TESTS`), and the tails.

```bash
source /tmp/e484env.sh
uv run --no-sync python ml-experiments/00484-*/e00484_score.py build
uv run --no-sync python ml-experiments/00484-*/e00484_score.py rows
uv run --no-sync python ml-experiments/00484-*/e00484_score.py stats
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
import e00484_fit as fit484  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import TAIL_LEVELS  # ty: ignore[unresolved-import]

MONTHS: Final[tuple[str, ...]] = fit484.FOLDS
KEYS: Final[list[str]] = build470.KEYS
LEVELS: Final[tuple[str, ...]] = build470.LEVELS
SEEDS: Final[tuple[int, ...]] = fit484.SEEDS
NEW_ARMS: Final[tuple[str, ...]] = ("W-cold", "W-curt", "W-season")
REFERENCE: Final[str] = "W0"
ARMS: Final[tuple[str, ...]] = (REFERENCE, *NEW_ARMS)
WORK: Final[Path] = fit484.WORK
SCORE_DIR: Final[Path] = WORK / "scores"
PRED_DIR: Final[Path] = WORK / "predictions"
SINGLES: Final[tuple[str, ...]] = tuple(f"A-s{s}" for s in build470.POOL_SEEDS)
TESTS: Final[int] = 27
GATING_CRPS: Final[tuple[str, ...]] = score470.GATING_CRPS
GATING_Q99: Final[tuple[str, ...]] = score470.GATING_Q99
assert len(NEW_ARMS) * (len(GATING_CRPS) + len(GATING_Q99)) == TESTS
MDE_SLICES: Final[dict[str, tuple[str, str]]] = {
    "eight": ("mva_eight", "all"),
    "six (clean rows)": ("signed_six", "clean"),
    "cold_eight": ("mva_eight", "cold"),
    "cold_six": ("signed_six", "cold"),
}


def pred_path(*, arm: str, month: str, cell: int) -> Path:
    """Return where an arm's seed cell is stored on experiment 454's rows."""
    return PRED_DIR / arm / f"{month}_c{cell}.parquet"


def build_command() -> None:
    """Write every arm's cells on experiment 454's rows."""
    sites = build470.site_map()
    for month in MONTHS:
        stored = build470.heads470.WORK_DIR / "predictions"
        rows = pl.read_parquet(stored / "A-s0" / f"{month}_c0.parquet").select(*KEYS)
        for seed in SEEDS:
            target = pred_path(arm=REFERENCE, month=month, cell=seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            frame = pl.read_parquet(stored / f"A-s{seed}" / f"{month}_c0.parquet")
            frame.write_parquet(target)
        for arm in NEW_ARMS:
            for seed in SEEDS:
                one = build470.pool_stored(
                    folder=fit484.WORK / "seed_preds" / arm, month=month, sites=sites, seeds=(seed,)
                )
                frame = build470.on_rows(rows=rows, frame=one)
                target = pred_path(arm=arm, month=month, cell=seed)
                target.parent.mkdir(parents=True, exist_ok=True)
                frame.select(*KEYS, *LEVELS).write_parquet(target)
        print(f"{month}: wrote {len(ARMS)} arms x {len(SEEDS)} cells", flush=True)


def month_rows(*, month: str, base: pl.DataFrame) -> pl.DataFrame:
    """Return a month's row-level metrics of every arm (headline widening, cells averaged)."""
    cols: dict[str, np.ndarray] = {}
    ok_all = np.ones(base.height, dtype=bool)
    for arm in ARMS:
        cells = []
        for cell in SEEDS:
            frame = base.select(*KEYS, "actual", "nra", "tgt_real").join(
                pl.read_parquet(pred_path(arm=arm, month=month, cell=cell)),
                on=KEYS,
                how="left",
                maintain_order="left",
            )
            frame = score470.widen_arm(arm=arm, frame=frame, factor=score470.HEADLINE)
            ok = frame["q50"].is_not_null().to_numpy() & np.isfinite(frame["q50"].to_numpy())
            frame = frame.with_columns(*[pl.col(c).fill_null(0.0) for c in LEVELS])
            metered = score470.row_metrics(frame=frame)
            nra = score470.row_metrics(
                frame=frame.with_columns(actual_nra=pl.col("actual") - pl.col("nra")),
                actual="actual_nra",
            )
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
        for i, c in enumerate(cells):
            cols[f"{arm}__crps_cell{i}"] = c[0]["crps"]
    out = base.select(*KEYS).hstack(
        pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items()})
    )
    return out.hstack(
        base.select(
            "group", "cls", "lead_hours", "lead_band", "tgt_real", "cold", "xmas", "band", "day"
        )
    ).with_columns(month=pl.lit(month), all_arms=pl.Series(ok_all))


def rows_command() -> None:
    """Write the row-level table, restricted to experiment 470's gam14 rows."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    gam = score470.load_rows().select(*KEYS, "month", "gam14", *[f"{a}__crps" for a in SINGLES])
    for month in MONTHS:
        base = score470.month_context(month=month, sites=sites, ctx=ctx)
        table = month_rows(month=month, base=base)
        table = table.join(
            gam.filter(pl.col("month") == month).drop("month"),
            on=KEYS,
            how="left",
            maintain_order="left",
        )
        table = table.filter(pl.col("gam14").fill_null(value=False) & pl.col("all_arms"))
        table.write_parquet(SCORE_DIR / f"rows_{month}.parquet")
        print(f"{month}: base rows {base.height}, scored rows {table.height}", flush=True)


def load_rows() -> pl.DataFrame:
    """Return the row-level table of all months."""
    return pl.concat([pl.read_parquet(SCORE_DIR / f"rows_{m}.parquet") for m in MONTHS])


def slice_rows(*, table: pl.DataFrame, name: str) -> tuple[pl.DataFrame, str]:
    """Return the rows of one of experiment 470's slices and the metric prefix."""
    return score470.slice_filter(table=table, name=name)


def mde_command(*, table: pl.DataFrame) -> pl.DataFrame:
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


def stats_command() -> None:
    """Print the minimum detectable effect, then absolute scores, contrasts, and tails."""
    table = load_rows()
    minimum = mde_command(table=table)
    print("minimum detectable effect (printed before any contrast):")
    print(minimum)
    minimum.write_csv(SCORE_DIR / "mde.csv")
    metrics = ["crps", *[f"pin_{t}" for t in TAIL_LEVELS], *[f"cov_{t}" for t in TAIL_LEVELS]]
    absolute = []
    for sname in score470.SLICES:
        part, prefix = slice_rows(table=table, name=sname)
        if part.height == 0:
            continue
        absolute.extend(
            {"slice": sname, "arm": arm, "n": part.height}
            | {m: part[f"{arm}__{prefix}{m}"].mean() for m in metrics}
            for arm in ARMS
        )
    pl.DataFrame(absolute).write_csv(SCORE_DIR / "absolute.csv")
    rng = np.random.default_rng(0)
    score470.TESTS = TESTS
    records = []
    for arm in NEW_ARMS:
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
                        d=pl.col(f"{arm}__{prefix}{metric}")
                        - pl.col(f"{REFERENCE}__{prefix}{metric}"),
                    )
                    .drop_nulls()
                    .drop_nans()
                )
                if diff.height == 0:
                    continue
                records.append(
                    {
                        "contrast": f"{arm} - {REFERENCE}",
                        "slice": sname,
                        "metric": metric,
                        "counted": (sname in GATING_CRPS and metric == "crps")
                        or (sname in GATING_Q99 and metric == "pin_q99"),
                        **score470.block_bootstrap(diff=diff, rng=rng),
                    }
                )
    contrasts = pl.DataFrame(records)
    contrasts.write_csv(SCORE_DIR / "contrasts.csv")
    print(f"counted tests: {int(contrasts['counted'].sum())} (pre-registered {TESTS})")
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(
            contrasts.filter(pl.col("metric") == "crps")
            .drop("by_month")
            .with_columns(pl.col(pl.Float64).round(3))
        )
        print(
            contrasts.filter(pl.col("metric") == "pin_q99")
            .drop("by_month")
            .with_columns(pl.col(pl.Float64).round(3))
        )


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "rows", "stats"))
    args = parser.parse_args()
    {"build": build_command, "rows": rows_command, "stats": stats_command}[args.command]()


if __name__ == "__main__":
    main()

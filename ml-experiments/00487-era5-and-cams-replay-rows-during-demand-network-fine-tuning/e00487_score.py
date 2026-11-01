# ruff: noqa: E501
"""Score experiment 487's replay arms on experiment 470's rows.

Rows are experiment 470's `main16` set (the eight MVA-only Primaries, the signed-MW six, and the
report-only D15 and D18), restricted to the `gam14` set (no D15 or D18) for every gating slice.
Slices are experiment 470's. Every arm is scored cell by cell (cell = fit seed) and the per-row
metrics are averaged over the three cells; scores are CRPS in points of effective capacity with the
1.15 widening unless a column says otherwise. All scores are selection-biased: the five folds that
choose the candidate also score it. July 2026 is not used.

Arms: `R0` (experiment 425's stored arm A fits, seeds 0 to 2), `R25`, `R50`, `Rcold` (this
experiment's fits), `A-pool` (five-seed quantile average, context), `A-s3` and `A-s4` (with `R0`'s
three cells, the five single seeds that give the minimum detectable effect).

Contrasts: each replay arm minus `R0`, 3 contrasts times 9 counted tests (7 CRPS slices and the q99
pinball loss at the eight and at the six on rows with no labelled event) = 27 Bonferroni tests,
fixed before any score was read.

```bash
source .e_replay_env.sh   (sets PYTHONPATH; run from the repo root)
uv run --no-sync python ml-experiments/00487-*/e00487_score.py rows
uv run --no-sync python ml-experiments/00487-*/e00487_score.py stats
```
"""

import argparse
import json
import math
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00487_build as build  # ty: ignore[unresolved-import]
import e00487_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import TAIL_LEVELS, widen  # ty: ignore[unresolved-import]

KEYS: Final[list[str]] = build470.KEYS
LEVELS: Final[tuple[str, ...]] = build470.LEVELS
SCORE_DIR = fit.WORK_DIR / "scores"
ARMS: Final[tuple[str, ...]] = ("R0", "R25", "R50", "Rcold")
CONTEXT: Final[tuple[str, ...]] = ("A-pool", "A-s3", "A-s4")
CELLS: Final[dict[str, int]] = {"R0": 3, "R25": 3, "R50": 3, "Rcold": 3}
NEW: Final[tuple[str, ...]] = ("R25", "R50", "Rcold")
WIDEN_SWEEP: Final[tuple[float, ...]] = (1.0, 1.05, 1.10, 1.15)
HEADLINE: Final[float] = 1.15
TESTS: Final[int] = 27
RESAMPLES: Final[int] = 20_000
GATING_CRPS: Final[tuple[str, ...]] = score470.GATING_CRPS
GATING_Q99: Final[tuple[str, ...]] = score470.GATING_Q99
assert len(NEW) * (len(GATING_CRPS) + len(GATING_Q99)) == TESTS
METRICS: Final[tuple[str, ...]] = (
    "crps",
    *[f"pin_{t}" for t in TAIL_LEVELS],
    *[f"cov_{t}" for t in TAIL_LEVELS],
)


def column(*, arm: str, metric: str, factor: float = HEADLINE, nra: bool = False) -> str:
    """Return a row-table column name."""
    if nra:
        return f"{arm}__nra_{metric}"
    return f"{arm}__{metric}" if factor == HEADLINE else f"{arm}__w{factor}__{metric}"


def read_cell(*, arm: str, month: str, cell: int) -> pl.DataFrame:
    """Return an arm cell's quantiles (fractions) on the scored rows."""
    return pl.read_parquet(build.prediction_path(arm=arm, month=month, cell=cell))


def month_rows(*, month: str, base: pl.DataFrame, flags: pl.DataFrame) -> pl.DataFrame:
    """Return a month's row-level metrics: cell-averaged, at each widening factor."""
    base = base.join(flags, on=KEYS, how="inner", maintain_order="left")
    cols: dict[str, np.ndarray] = {}
    for arm in (*ARMS, *CONTEXT):
        cells = CELLS.get(arm, 1)
        per_cell: list[dict[str, np.ndarray]] = []
        ok_all = []
        for cell in range(cells):
            frame = base.select(*KEYS, "actual", "nra").join(
                read_cell(arm=arm, month=month, cell=cell),
                on=KEYS,
                how="left",
                maintain_order="left",
            )
            ok = frame["q50"].is_not_null().to_numpy() & np.isfinite(frame["q50"].to_numpy())
            ok_all.append(ok)
            frame = frame.with_columns(*[pl.col(c).fill_null(0.0) for c in LEVELS])
            metrics: dict[str, np.ndarray] = {}
            for factor in WIDEN_SWEEP:
                m = score470.row_metrics(frame=widen(frame=frame, factor=factor))
                for name, values in m.items():
                    metrics[column(arm=arm, metric=name, factor=factor)] = np.where(
                        ok, values, np.nan
                    )
            nra = score470.row_metrics(
                frame=widen(frame=frame, factor=HEADLINE).with_columns(
                    actual_nra=pl.col("actual") - pl.col("nra")
                ),
                actual="actual_nra",
            )
            metrics[column(arm=arm, metric="crps", nra=True)] = np.where(ok, nra["crps"], np.nan)
            per_cell.append(metrics)
        every = np.logical_and.reduce(ok_all)
        for name in per_cell[0]:
            cols[name] = np.where(every, np.mean([c[name] for c in per_cell], axis=0), np.nan)
        if cells > 1:
            for i, c in enumerate(per_cell):
                cols[f"{arm}__crps_cell{i}"] = c[column(arm=arm, metric="crps")]
    out = base.select(
        *KEYS, "group", "cls", "lead_hours", "lead_band", "tgt_real", "cold", "xmas", "band", "day",
        "gam14", "main16",
    ).hstack(pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items()}))  # fmt: skip
    return out.with_columns(month=pl.lit(month))


def rows_command() -> None:
    """Write the row-level table (one parquet per month) and print the row-set sizes."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    rows470 = score470.load_rows().select(*KEYS, "month", "gam14", "main16")
    for month in fit.MONTHS:
        base = score470.month_context(month=month, sites=sites, ctx=ctx)
        flags = rows470.filter((pl.col("month") == month) & pl.col("main16")).drop("month")
        table = month_rows(month=month, base=base, flags=flags)
        for arm in (*ARMS, *CONTEXT):
            held = table.filter(pl.col("main16"))[column(arm=arm, metric="crps")]
            assert held.is_not_null().all(), f"{arm} {month}: missing rows"
            assert held.is_finite().all(), f"{arm} {month}: non-finite scores"
        table.write_parquet(SCORE_DIR / f"rows_{month}.parquet")
        print(
            f"{month}: base {base.height}, main16 {int(table['main16'].sum())}, gam14 {int(table['gam14'].sum())}",
            flush=True,
        )


def load_rows() -> pl.DataFrame:
    """Return the row-level table of all months."""
    return pl.concat([pl.read_parquet(SCORE_DIR / f"rows_{m}.parquet") for m in fit.MONTHS])


def block_bootstrap(*, diff: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Return the difference and its intervals from month-by-origin-day blocks (diff in `d`)."""
    blocks = diff.group_by("month", "day").agg(pl.len().alias("n"), pl.col("d").sum().alias("s"))
    n, s = blocks["n"].to_numpy().astype(float), blocks["s"].to_numpy()
    draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
    boot = s[draws].sum(1) / n[draws].sum(1)
    alpha = 100.0 * 0.05 / TESTS / 2.0
    by_month = diff.group_by("month").agg(pl.col("d").mean())
    return {
        "n": int(n.sum()), "difference": float(s.sum() / n.sum()),
        "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5)),
        "low_bonferroni": float(np.percentile(boot, alpha)),
        "high_bonferroni": float(np.percentile(boot, 100.0 - alpha)),
        "by_month": json.dumps({r["month"]: round(r["d"], 4) for r in by_month.iter_rows(named=True)}),
    }  # fmt: skip


def mde(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the minimum detectable effect from the seed spread of arm A's five single seeds.

    The spread is the residual standard deviation of the five seeds' fold-mean CRPS (experiment
    428's formula, as in experiment 470). `mde_1` is for two single-seed arms; `mde_3` is for two
    arms that each average three seeds, which is this experiment's contrast.
    """
    columns = [f"R0__crps_cell{i}" for i in range(3)] + [
        column(arm=a, metric="crps") for a in ("A-s3", "A-s4")
    ]
    records = []
    for label, slice_name in (
        ("eight", "eight"), ("six (rows with no labelled event)", "six_clean"),
        ("six (all rows)", None), ("cold nights, eight", "cold_eight"), ("cold nights, six", "cold_six"),
    ):  # fmt: skip
        if slice_name is None:
            part = table.filter((pl.col("group") == "signed_six") & pl.col("gam14"))
        else:
            part, _ = score470.slice_filter(table=table, name=slice_name)
            part = part.filter(pl.col("gam14"))
        values = np.array(
            [
                [part.filter(pl.col("month") == m)[c].mean() for c in columns]
                for m in fit.MONTHS
                if part.filter(pl.col("month") == m).height
            ]
        )
        folds, k = values.shape
        resid = values - values.mean(axis=1, keepdims=True)
        sd = math.sqrt((resid**2).sum() / (folds * (k - 1)))
        records.append(
            {"slice": label, "seed_sd": sd, "mde_1": 1.96 * math.sqrt(2.0) * sd,
             "mde_3": 1.96 * math.sqrt(2.0) * sd / math.sqrt(3.0)}
        )  # fmt: skip
    return pl.DataFrame(records)


def stats_command() -> None:
    """Write the minimum detectable effect, absolute scores, contrasts, and tails."""
    table = load_rows()
    minimum = mde(table=table)
    print("minimum detectable effect, printed before any contrast (points of effective capacity):")
    with pl.Config(tbl_rows=20, tbl_width_chars=160):
        print(minimum.with_columns(pl.col(pl.Float64).round(4)))
    minimum.write_csv(SCORE_DIR / "mde.csv")
    absolute = []
    for sname in (*score470.SLICES,):
        for arm in (*ARMS, *CONTEXT):
            part, prefix = score470.slice_filter(table=table, name=sname)
            part = part.filter(pl.col("main16" if sname == "report" else "gam14"))
            for factor in WIDEN_SWEEP:
                record = {"slice": sname, "arm": arm, "factor": factor, "n": part.height}
                for m in METRICS:
                    if prefix and m != "crps":
                        continue
                    record[m] = part[
                        column(arm=arm, metric=m, factor=factor, nra=bool(prefix))
                    ].mean()
                if factor != HEADLINE and sname not in (
                    "eight",
                    "six_clean",
                    "cold_eight",
                    "cold_six",
                    "report",
                ):
                    continue
                absolute.append(record)
    pl.DataFrame(absolute).write_csv(SCORE_DIR / "absolute.csv")
    rng = np.random.default_rng(0)
    records = []
    for a, b in [*[(x, "R0") for x in NEW], *[(x, "A-pool") for x in (*NEW, "R0")]]:
        counted_pair = b == "R0"
        for sname in (*GATING_CRPS, "xmas_eight", "xmas_six", "report"):
            for metric in (
                "crps",
                *(
                    [f"pin_{t}" for t in TAIL_LEVELS]
                    if sname in ("eight", "six_clean", "report")
                    else []
                ),
            ):
                part, prefix = score470.slice_filter(table=table, name=sname)
                part = part.filter(pl.col("main16" if sname == "report" else "gam14"))
                diff = (
                    part.select(
                        "month", "day",
                        d=pl.col(column(arm=a, metric=metric, nra=bool(prefix))) - pl.col(column(arm=b, metric=metric, nra=bool(prefix))),
                    )
                    .drop_nulls()
                    .drop_nans()
                )  # fmt: skip
                if diff.height == 0:
                    continue
                records.append(
                    {"contrast": f"{a} - {b}", "slice": sname, "metric": metric,
                     "counted": counted_pair and ((sname in GATING_CRPS and metric == "crps") or (sname in GATING_Q99 and metric == "pin_q99")),
                     **block_bootstrap(diff=diff, rng=rng)}
                )  # fmt: skip
    contrasts = pl.DataFrame(records)
    contrasts.write_csv(SCORE_DIR / "contrasts.csv")
    counted = int(contrasts["counted"].sum())
    print(f"counted tests: {counted} (pre-registered {TESTS})")
    assert counted == TESTS
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(
            contrasts.filter((pl.col("metric") == "crps") & pl.col("contrast").str.ends_with("R0"))
            .drop("by_month")
            .with_columns(pl.col(pl.Float64).round(3))
        )


def repro_command() -> None:
    """Compare the R0 refit (2026-01, seed 0) with experiment 425's stored fit.

    Prints the largest quantile difference (capacity fraction) and the epochs and validation
    negative log-likelihood of both fits. The pass rule of the plan (CRPS within the printed
    single-seed standard deviation) is read from `repro_crps` below and `mde.csv`.
    """
    month, seed = "2026-01", 0
    new = pl.read_parquet(fit.seed_pred_path(arm="R0", month=month, seed=seed))
    old = pl.read_parquet(fit.E425_WORK / "seed_preds" / "A" / f"{month}_s{seed}.parquet")
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    joined = new.join(old, on=keys, suffix="_old")
    diff = max(float((joined[q] - joined[f"{q}_old"]).abs().max()) for q in LEVELS)  # ty: ignore[invalid-argument-type]
    print(f"rows {new.height} vs {old.height}; max abs quantile difference {diff:.5f}")
    new_log = json.loads((fit.WORK_DIR / "logs" / f"{month}.json").read_text())[f"R0_s{seed}"]
    old_log = json.loads((fit.E425_WORK / "logs" / f"{month}.json").read_text())[f"A_s{seed}"]
    print("refit", new_log, "stored", old_log)
    # CRPS of both fits, 1.15 widening, on the eight and the six (all rows of the month).
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    base = score470.month_context(month=month, sites=build470.site_map(), ctx=ctx)
    for label, frame in (("refit", new), ("stored", old)):
        grid = build470.grid_keys(
            frame=frame.join(
                build470.site_map().select("time_series_id", "primary"), on="time_series_id"
            )
        )
        rows = base.select(*KEYS, "actual", "group").join(
            grid.select(*KEYS, *LEVELS), on=KEYS, how="inner"
        )
        scored = score470.row_metrics(frame=widen(frame=rows, factor=HEADLINE))
        for group in ("mva_eight", "signed_six"):
            mask = (rows["group"] == group).to_numpy()
            print(f"repro_crps {label} {group}: {scored['crps'][mask].mean():.4f}")


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rows", "stats", "repro"))
    args = parser.parse_args()
    {"rows": rows_command, "stats": stats_command, "repro": repro_command}[args.command]()


if __name__ == "__main__":
    main()

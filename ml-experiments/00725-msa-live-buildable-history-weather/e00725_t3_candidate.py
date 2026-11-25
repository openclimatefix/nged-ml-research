# ruff: noqa: E501, PT018
"""Experiment 725: the stored Candidate predicts with the target row's CAMS day summary moved to D - 3 (read-out `T3`, arm `T3-C`).

Usage: e00725_t3_candidate.py T3-C MONTH SEED [--device cuda] [--step all|table|fit]

**What moves.** The Candidate reads the CAMS mean and the CAMS clear-sky ratio of the UTC day before the
forecast origin in two columns of its per-Primary table, `cams_prev_day_mean` and `cams_prev_day_csi`
(experiment 437's `site_table` fills them with `cams.get(origin_day - 1)`). In `T3` those two columns hold
the values of UTC day `origin_day - 3` for every row whose origin is on or after the first day of the fold
month, which are the rows that the fold scores. Every other row and every other column keeps the stored
value, so the standardiser, the training rows, and the lags of the Candidate are unchanged. No state is
trained: the stored stage states and the stored step 2 state of experiment 667 predict.

**How it runs.** The script follows `e00713_swap_c.py` (read only). It mirrors experiment 667's `inputs` as
links under `t3_candidate/<month>` of this experiment's data folder, replaces the 16 table links of
`inputs/trailing437` with the patched tables, links the stored states, the standardiser hash, and the chain,
and runs experiment 667's own code twice with `E00667_PHASE=predict`: the `table` command builds the lag-free
table `T` from the stored step 2 state on the patched inputs, and the `fit` command loads the stored stage
state (the fine-tune is skipped) and writes the forecasts to
`t3_candidate/<month>/P7_D50_<month>/work/fits/P5/<month>_d0_s<seed>.parquet`. `--step all` runs the two
commands as two processes, because experiment 667's module reads its command at import.

**Checks the script makes.** Before it writes a table, it rebuilds the stored columns with `origin_day - 1`
and asserts that they equal the stored columns on every row, which proves that the day arithmetic is
experiment 437's. After the `fit` step it asserts that the forecast keys equal the stored Candidate's keys,
that the quantiles differ from the stored Candidate's somewhere, and that nothing under experiment 667's
folder changed. It records the shares of finite values and the size of the differences in
`t3_candidate/<month>/log_s<seed>.json`.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00725_common as cm
import numpy as np
import polars as pl

ARM: Final = "T3-C"
CAMS_COLUMNS: Final = ("cams_prev_day_mean", "cams_prev_day_csi")
US_PER_DAY: Final = 86_400 * 1_000_000
BACK_DAYS: Final = 3
"""The target row reads the CAMS day summary of UTC day `origin_day - BACK_DAYS` (the stored Candidate reads 1)."""
TRAIL_TAG: Final = "P7_D50"
CAND_ARM: Final = "P7+D50"
"""The Candidate's arm name in experiment 667's command line (`e00667_cand.py P7+D50 ...`)."""


def root(*, month: str) -> Path:
    """Return the folder that holds one fold's `T3` inputs, tables, states, and forecasts."""
    return cm.ROOT / "t3_candidate" / month


def fit_file(*, month: str, seed: int) -> Path:
    """Return the forecast file of arm `T3-C`, in the layout of the Candidate's own forecast files."""
    return (
        root(month=month) / f"{TRAIL_TAG}_{month}" / "work" / "fits" / "P5" / f"{month}_d0_s{seed}.parquet"
    )  # fmt: skip


def log_file(*, month: str, seed: int) -> Path:
    """Return the JSON log of one `T3-C` forecast."""
    return root(month=month) / f"log_s{seed}.json"


def month_start_us(*, month: str) -> int:
    """Return the first microsecond of the fold month, in UTC."""
    year, number = (int(x) for x in month.split("-"))
    return int(datetime(year, number, 1, tzinfo=UTC).timestamp()) * 1_000_000


def day_columns(
    *, origin_us: np.ndarray, cams: dict[int, tuple[float, float]], back_days: int
) -> np.ndarray:
    """Return (rows, 2): the CAMS mean and clear-sky ratio of UTC day `origin day - back_days`, NaN where CAMS has none.

    The origin day is the floor of the origin in days since the epoch, as in experiment 437's `site_table`.
    """
    day = origin_us // US_PER_DAY
    days, inverse = np.unique(day, return_inverse=True)
    values = np.array(
        [cams.get(int(d) - back_days, (np.nan, np.nan)) for d in days], dtype=np.float64
    ).reshape(len(days), 2)
    return values[inverse]


def patched_table(
    *, source: pl.DataFrame, cams: dict[int, tuple[float, float]], start_us: int
) -> pl.DataFrame:
    """Return a Primary's table with the two CAMS columns of every row on or after `start_us` moved to D - 3.

    The function first rebuilds both stored columns from `cams` with the stored day offset of 1 and asserts that
    they equal the stored columns on every row. The patched table equals `source` in every other column and in
    every row before `start_us`.
    """
    origin = source["origin_us"].to_numpy()
    rebuilt = day_columns(origin_us=origin, cams=cams, back_days=1)
    moved = day_columns(origin_us=origin, cams=cams, back_days=BACK_DAYS)
    scored = origin >= start_us
    columns = []
    for k, name in enumerate(CAMS_COLUMNS):
        stored = source[name].to_numpy()
        same = (rebuilt[:, k].astype(stored.dtype) == stored) | (
            np.isnan(stored) & np.isnan(rebuilt[:, k])
        )
        assert same.all(), (
            f"{name}: the stored column differs from CAMS day origin - 1 in {int((~same).sum()):,} of {len(same):,} rows"
        )
        values = np.where(scored, moved[:, k].astype(stored.dtype), stored)
        columns.append(pl.Series(name, values, dtype=source.schema[name]))
    return source.with_columns(*columns)


def build_tables(*, month: str, swap) -> dict:  # noqa: ANN001
    """Replace the table links of `inputs/trailing437` under the fold's folder with the patched tables; return a report."""
    import e00437_features as f437  # ty: ignore[unresolved-import]

    source_dir = swap.inp.ROOT_667 / "inputs" / "trailing437"
    target_dir = swap.ROOT_SWAP / "inputs" / "trailing437"
    start = month_start_us(month=month)
    report: dict[str, dict] = {}
    for row in f437.prep.primaries().iter_rows(named=True):
        site = row["site"]
        target = target_dir / f"{site}.parquet"
        if target.exists() and not target.is_symlink():
            continue  # a patched table from an earlier run of this fold
        source = pl.read_parquet(source_dir / f"{site}.parquet")
        table = patched_table(source=source, cams=f437.cams_previous_day(site=site), start_us=start)
        origin = source["origin_us"].to_numpy()
        scored = origin >= start
        report[site] = {
            "rows": source.height,
            "scored_rows": int(scored.sum()),
            "finite_before": int(np.isfinite(source[CAMS_COLUMNS[0]].to_numpy()[scored]).sum()),
            "finite_after": int(np.isfinite(table[CAMS_COLUMNS[0]].to_numpy()[scored]).sum()),
            "changed_scored_rows": int((source[CAMS_COLUMNS[0]].to_numpy()[scored] != table[CAMS_COLUMNS[0]].to_numpy()[scored]).sum()),
        }  # fmt: skip
        tmp = target.with_name(target.name + ".tmp")
        table.write_parquet(tmp)
        tmp.replace(target)  # replaces the link itself and never writes through it
        print(site, report[site], flush=True)
    return report


def link_stored(*, month: str, seed: int, swap) -> None:  # noqa: ANN001
    """Link the stored stage state and the standardiser hash of experiment 667; the stored `T` table is not linked."""
    fold = swap.inp.ROOT_667 / f"{TRAIL_TAG}_{month}"
    stored = fold / "work" / "models" / "P5" / f"{month}_d0_s{seed}.pt"
    assert stored.exists(), stored
    swap.link_file(
        source=stored,
        dest=swap.ROOT_SWAP / f"{TRAIL_TAG}_{month}" / "work" / "models" / "P5" / stored.name,
    )
    sha = fold / f"standardiser_{month}.sha256"
    assert sha.exists(), sha
    swap.link_file(source=sha, dest=swap.ROOT_SWAP / f"{TRAIL_TAG}_{month}" / sha.name)


def run_step(*, month: str, seed: int, step: str, device: str) -> int:
    """Run experiment 667's `table` or `fit` command on the patched inputs, in this process."""
    import e00713_swap_c as swap  # ty: ignore[unresolved-import]
    import e00725_frozen as fz

    fz.assert_frozen_checkout()
    fz.limit_gpu_memory(device=device)
    before = swap.snapshot_667()
    swap.ROOT_SWAP = root(month=month)
    link_stored(month=month, seed=seed, swap=swap)
    swap.redirect_inputs()
    report = build_tables(month=month, swap=swap)
    os.environ.update(
        E00667_ROOT=str(swap.ROOT_SWAP), E00667_MONTH=month, E00667_PHASE="predict",
        CHAIN667=str(swap.link_chain(month=month)),
    )  # fmt: skip
    sys.argv = [
        "e00667_cand.py", CAND_ARM, step, "--work-suffix", f"_{month}", "--months", month, "--seeds", str(seed),
    ]  # fmt: skip
    started = time.monotonic()
    import e00667_cand as cand  # ty: ignore[unresolved-import]

    if step == "table":
        cand.e.k.build_month = cand.build_month_syn
        cand.e.k.GpuMonth = cand.SigGpuMonth
        cand.run_prepare(device=device, command="table")
    else:
        cand.run_fit(device=device)
    assert swap.snapshot_667() == before, "the T3 run changed a path under experiment 667's folder"
    if step == "fit":
        write_log(
            month=month, seed=seed, seconds=time.monotonic() - started, tables=report, swap=swap
        )
    return 0


def write_log(*, month: str, seed: int, seconds: float, tables: dict, swap) -> None:  # noqa: ANN001
    """Compare the forecasts with the stored Candidate's, assert that the patch took effect, and write the log."""
    import e00713_score as sc713  # ty: ignore[unresolved-import]

    moved = pl.read_parquet(fit_file(month=month, seed=seed))
    stored_path = (
        swap.inp.ROOT_667 / f"{TRAIL_TAG}_{month}" / "work" / "fits" / "P5" / f"{month}_d0_s{seed}.parquet"
    )  # fmt: skip
    stored = pl.read_parquet(stored_path)
    keys = list(sc713.KEYS)
    assert moved.height == stored.height, (moved.height, stored.height)
    moved, stored = moved.sort(keys), stored.sort(keys)
    for key in keys:
        assert moved[key].equals(stored[key]), f"the forecast keys differ in {key}"
    difference = {
        q: float(np.nanmax(np.abs(moved[q].to_numpy() - stored[q].to_numpy()))) for q in sc713.QCOLS
    }
    median = sc713.QCOLS[len(sc713.QCOLS) // 2]
    assert max(difference.values()) > 0.0, (
        "T3-C equals the stored Candidate: the patched columns were not read"
    )
    cm.write_json_atomic(
        obj={
            "arm": ARM, "month": month, "seed": seed, "back_days": BACK_DAYS, "rows": moved.height,
            "max_abs_quantile_difference": difference,
            "mean_abs_median_difference": float(np.nanmean(np.abs(moved[median].to_numpy() - stored[median].to_numpy()))),
            "seconds": round(seconds, 1), "tables": tables,
        },
        path=log_file(month=month, seed=seed),
    )  # fmt: skip
    print(
        json.dumps({"month": month, "seed": seed, "max_abs_quantile_difference": difference}),
        flush=True,
    )


def main() -> int:
    """Run the `table` and `fit` commands of one fold and seed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step", choices=("all", "table", "fit"), default="all")
    args = parser.parse_args()
    assert args.arm == ARM, f"this script writes arm {ARM} only, not {args.arm!r}"
    assert args.month in cm.ALL_FOLDS and args.seed in cm.TWO_SEEDS, (args.month, args.seed)
    if args.step == "all":
        for step in ("table", "fit"):
            status = subprocess.run(
                [sys.executable, __file__, args.arm, args.month, str(args.seed), "--device", args.device, "--step", step],
                check=False,
            ).returncode  # fmt: skip
            if status != 0:
                return status
        return 0
    return run_step(month=args.month, seed=args.seed, step=args.step, device=args.device)


if __name__ == "__main__":
    sys.exit(main())

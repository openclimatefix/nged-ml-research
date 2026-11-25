# ruff: noqa: E501, PT018
"""Experiment 725: the stored Candidate predicts with the target row's CAMS day summary moved to D - 3 (read-out `T3`, arm `T3-C`).

Usage: e00725_t3_candidate.py T3-C MONTH SEED --step table|fit [--device cuda]

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
and runs experiment 667's own code with `E00667_PHASE=predict`: the `table` step builds the lag-free table
`T` from the stored step 2 state on the patched inputs, and the `fit` step loads the stored stage state (the
fine-tune is skipped, and the script asserts that) and writes the forecasts to
`t3_candidate/<month>/P7_D50_<month>/work/fits/P5/<month>_d0_s<seed>.parquet`. The two steps are two jobs
of one chain, because experiment 667's module reads its command at import. The phase runner puts both seeds
of a fold in one chain, so the two seeds of a fold never run at once.

**The CAMS file.** Experiment 667 built the stored tables from experiment 620's extended CAMS file, so the
script points `e00418_prep.CAMS_PATH` at `e00667_tables.CAMS_EXTENDED` (the path only) before it reads CAMS.

**Checks the script makes.** Before it writes a table, it rebuilds the stored columns with `origin_day - 1`
and asserts that they equal the stored columns on every row, which proves that the day arithmetic and the
CAMS file are experiment 437's. After the `fit` step it asserts that the run reused the stored state, that the
forecast keys equal the stored Candidate's keys, that the share of rows whose median changed is at least half
the share of scored table rows whose CAMS value changed, and that nothing under experiment 667's folder
changed. The log `t3_candidate/<month>/log_s<seed>.json` is the done marker of the arm, and it records the
shares, the size of the differences, the code record, and the hash of this script.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Final, Literal

import e00725_common as cm
import numpy as np
import polars as pl

Step = Literal["table", "fit"]
ARM: Final[str] = "T3-C"
CAMS_COLUMNS: Final[tuple[str, str]] = ("cams_prev_day_mean", "cams_prev_day_csi")
US_PER_DAY: Final[int] = 86_400 * 1_000_000
BACK_DAYS: Final[int] = 3
"""The target row reads the CAMS day summary of UTC day `origin_day - BACK_DAYS` (the stored Candidate reads 1)."""
TRAIL_TAG: Final[str] = "P7_D50"
CAND_ARM: Final[str] = "P7+D50"
"""The Candidate's arm name in experiment 667's command line (`e00667_cand.py P7+D50 ...`)."""


def root(*, month: str) -> Path:
    """Return the folder that holds one fold's `T3` inputs, tables, states, and forecasts."""
    return cm.ROOT / "t3_candidate" / month


def fit_file(*, month: str, seed: int) -> Path:
    """Return the forecast file of arm `T3-C`, in the layout of the Candidate's own forecast files."""
    return (
        root(month=month) / f"{TRAIL_TAG}_{month}" / "work" / "fits" / "P5" / f"{month}_d0_s{seed}.parquet"
    )  # fmt: skip


def table_file(*, month: str, seed: int) -> Path:
    """Return the lag-free table that the `table` step writes (the done marker of that step)."""
    return root(month=month) / "tables" / "predict" / f"T_{month}_s{seed}.npz"


def log_file(*, month: str, seed: int) -> Path:
    """Return the JSON log of one `T3-C` forecast, which the `fit` step writes after every check passed (the done marker of the arm)."""
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
        columns.append(pl.Series(name=name, values=values, dtype=source.schema[name]))
    return source.with_columns(*columns)


def use_stored_cams() -> Path:
    """Point experiment 437's CAMS reader at the file that experiment 667 built the stored tables from; return the path.

    Only the path changes: `e00667_tables.setup()` would also write a combined file, and the script never calls it.
    """
    import e00418_prep as prep418  # ty: ignore[unresolved-import]
    import e00667_tables as t667  # ty: ignore[unresolved-import]

    assert t667.CAMS_EXTENDED.exists(), t667.CAMS_EXTENDED
    prep418.CAMS_PATH = t667.CAMS_EXTENDED
    return t667.CAMS_EXTENDED


def table_report(*, source: pl.DataFrame, table: pl.DataFrame, start_us: int) -> dict[str, int]:
    """Return the counts of scored rows, finite values, and rows whose first CAMS column changed (NaN on both sides is no change)."""
    scored = source["origin_us"].to_numpy() >= start_us
    before = source[CAMS_COLUMNS[0]].to_numpy()[scored]
    after = table[CAMS_COLUMNS[0]].to_numpy()[scored]
    same = (before == after) | (np.isnan(before) & np.isnan(after))
    return {
        "rows": source.height,
        "scored_rows": int(scored.sum()),
        "finite_before": int(np.isfinite(before).sum()),
        "finite_after": int(np.isfinite(after).sum()),
        "changed_scored_rows": int((~same).sum()),
    }


def build_tables(*, month: str, swap: ModuleType) -> dict[str, dict[str, int]]:
    """Replace the table links of `inputs/trailing437` under the fold's folder with the patched tables; return a report.

    A patched table from an earlier run of the fold is kept, and the report is rebuilt from it, so that the second
    seed of a fold reports the same counts as the first. The temporary file carries the process number.
    """
    import e00437_features as f437  # ty: ignore[unresolved-import]

    use_stored_cams()
    source_dir = swap.inp.ROOT_667 / "inputs" / "trailing437"
    target_dir = swap.ROOT_SWAP / "inputs" / "trailing437"
    start = month_start_us(month=month)
    report: dict[str, dict[str, int]] = {}
    for row in f437.prep.primaries().iter_rows(named=True):
        site = row["site"]
        target = target_dir / f"{site}.parquet"
        source = pl.read_parquet(source_dir / f"{site}.parquet")
        if target.exists() and not target.is_symlink():
            table = pl.read_parquet(target)
        else:
            table = patched_table(
                source=source, cams=f437.cams_previous_day(site=site), start_us=start
            )
            tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            table.write_parquet(tmp)
            tmp.replace(target)  # replaces the link itself and never writes through it
        report[site] = table_report(source=source, table=table, start_us=start)
        print(site, report[site], flush=True)
    return report


def link_stored(*, month: str, seed: int, swap: ModuleType) -> None:
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


def run_step(*, month: str, seed: int, step: Step, device: str) -> int:
    """Run experiment 667's `table` or `fit` command on the patched inputs, in this process."""
    import e00713_swap_c as swap  # ty: ignore[unresolved-import]
    import e00725_frozen as fz
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
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
        "e00667_cand.py", CAND_ARM, step, "--work-suffix", f"_{month}", "--months", month,
        "--seeds", str(seed),
    ]  # fmt: skip
    started = time.monotonic()
    import e00667_cand as cand  # ty: ignore[unresolved-import]

    if step == "table":
        cand.e.k.build_month = cand.build_month_syn
        cand.e.k.GpuMonth = cand.SigGpuMonth
        cand.run_prepare(device=device, command="table")
        assert table_file(month=month, seed=seed).exists(), table_file(month=month, seed=seed)
    else:
        cand.run_fit(device=device)
    assert swap.snapshot_667() == before, "the T3 run changed a path under experiment 667's folder"
    if step == "fit":
        write_log(
            month=month, seed=seed, seconds=time.monotonic() - started, tables=report, swap=swap
        )
    return 0


def write_log(
    *, month: str, seed: int, seconds: float, tables: dict[str, dict[str, int]], swap: ModuleType
) -> None:
    """Check the forecasts against the stored Candidate's and the stored state's reuse, then write the log (the arm's done marker)."""
    import e00713_score as sc713  # ty: ignore[unresolved-import]
    import e00725_frozen as fz

    work = root(month=month) / f"{TRAIL_TAG}_{month}" / "work"
    run_log = json.loads((work / "logs" / "P5" / f"{month}_d0_s{seed}_predict.json").read_text())
    assert run_log.get("reused"), (
        f"the fine-tune ran instead of loading the stored state: {run_log}"
    )
    moved = pl.read_parquet(fit_file(month=month, seed=seed))
    stored = pl.read_parquet(
        swap.inp.ROOT_667 / f"{TRAIL_TAG}_{month}" / "work" / "fits" / "P5" / f"{month}_d0_s{seed}.parquet"
    )  # fmt: skip
    keys = list(sc713.KEYS)
    assert moved.height == stored.height, (moved.height, stored.height)
    moved, stored = moved.sort(keys), stored.sort(keys)
    for key in keys:
        assert moved[key].equals(stored[key]), f"the forecast keys differ in {key}"
    median = sc713.QCOLS[len(sc713.QCOLS) // 2]
    delta = {q: np.abs(moved[q].to_numpy() - stored[q].to_numpy()) for q in sc713.QCOLS}
    share_changed = float((delta[median] > 0.0).mean())
    scored = sum(t["scored_rows"] for t in tables.values())
    expected = sum(t["changed_scored_rows"] for t in tables.values()) / max(scored, 1)
    assert share_changed >= 0.5 * expected and share_changed > 0.0, (
        f"the patched columns were barely read: {share_changed:.4f} of the forecast rows changed, "
        f"{expected:.4f} of the scored table rows changed"
    )
    cm.write_json_atomic(
        obj={
            "arm": ARM, "month": month, "seed": seed, "back_days": BACK_DAYS, "rows": moved.height,
            "share_of_rows_with_a_changed_median": share_changed,
            "share_of_scored_table_rows_changed": expected,
            "max_abs_quantile_difference": {q: float(np.nanmax(d)) for q, d in delta.items()},
            "mean_abs_median_difference": float(np.nanmean(delta[median])),
            "seconds": round(seconds, 1), "tables": tables,
            "code_record": fz.code_record(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        path=log_file(month=month, seed=seed),
    )  # fmt: skip
    print(json.dumps({"month": month, "seed": seed, "share_changed": share_changed}), flush=True)


def main() -> int:
    """Run the `table` or the `fit` step of one fold and seed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step", choices=("table", "fit"), required=True)
    args = parser.parse_args()
    assert args.arm == ARM, f"this script writes arm {ARM} only, not {args.arm!r}"
    assert args.month in cm.ALL_FOLDS and args.seed in cm.TWO_SEEDS, (args.month, args.seed)
    return run_step(month=args.month, seed=args.seed, step=args.step, device=args.device)


if __name__ == "__main__":
    sys.exit(main())

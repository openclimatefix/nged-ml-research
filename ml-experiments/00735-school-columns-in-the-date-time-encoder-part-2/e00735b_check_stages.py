# ruff: noqa: E501, PT018
"""Experiment 735 part 2: the "every stage" gate of the 8 columns (CPU).

README. For each arm (R0, R1, SKE, SKE-S21, SKE-CP), each stage (1, 2, 3), and each input path of
the date-time encoder (the row, the day lag, the week lag), this script feeds a batch through the arm's
network class and reads the tensor that reaches the date-time encoder (a forward pre-hook on
`network.datetime`). It asserts that:

- an arm with the columns has an encoder input of 37 columns whose columns 21 to 28 equal the arm's 8
  columns at that path's valid time and Primary (the row's valid time, or the grid time of the lag), and
  an arm without them (R0, R1) has an encoder input of 29 columns and no new-column table;
- each of the 8 columns is non-constant (more than one distinct value), and each school column holds
  both 0 and 1;
- SKE-S21's school columns S1 to S3 differ from SKE's on at least 5% of the rows, and SKE-CP's coast
  columns K1 to K5 differ from SKE's at 14 or more of the 16 Primaries.

R0 is experiment 677's stored control, so its network is built from experiment 591's `AuxLagMdn`, the
class that experiment 677's step-5 network is.

**What a batch is.** The batch of a stage is `ROWS` rows with valid times drawn uniformly over the
stage's valid-time range, random Primaries, and day and week lags 1 to 3 days and 1 to 2 weeks before
the row (5% missing). The ranges are those of the refit month 2026-03: stage 1 trains on ERA5 and CAMS
rows from 2019-09-17, and stages 2 and 3 on ensemble rows from 2024-04-01, both to 3.5 hours before
the refit. Stage 3 runs in training mode with the two label columns attached; stages 1 and 2 run without
labels. The other columns of a row are random numbers, because the gate checks the new columns' path.
This synthetic mode does not read the real rows of a month.

**The real-run mode** (`--from-json FOLDER`) reads the files `checks_stage_<stage>_<month>_s<seed>.json`
that the runner writes in stages 1, 2, and 3 (`e00735b_stage1.gate_stage`, run on 2,048 random
training rows and 2,048 random validation rows of the real matrix, through a forward pre-hook on
`network.datetime`). It prints, for each arm, stage, batch, path, and column, the number of distinct
values and the share of ones, and asserts that an arm with the columns shows them equal to the
expected columns on every path, that each column of the training batch is non-constant, and that R1
shows none of the columns. R0 is experiment 677's stored control and writes no file.

Usage: `python e00735b_check_stages.py [--out PATH] [--from-json FOLDER]`. PYTHONPATH: source
`ml-experiments/00524*/env.sh`. Exit code 0 when every assertion holds, and the table is printed (and
written to PATH as csv).
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import e00735b_columns as cols  # ty: ignore[unresolved-import]
import e00735b_model as model735  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

ROWS: Final = 20_000
N_PRIMARIES: Final = 16
N_B1: Final = 14
N_KNOTS: Final = 6
STEP_US: Final = 1_800_000_000
FIRST_US: Final = int(common.GRID_START.timestamp() * 1e6)
N_GRID: Final = int((common.GRID_END - common.GRID_START).total_seconds() // 1800) + 1
STAGE_RANGES: Final = {
    1: (datetime(2019, 9, 17, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
    2: (datetime(2024, 4, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
    3: (datetime(2024, 4, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
}
PATHS: Final = ("row", "day lag", "week lag")
MIN_SHIFT_SHARE: Final = 0.05
MIN_PERMUTED_PRIMARIES: Final = 14


def grid_of(*, when: datetime) -> int:
    """Return the grid index of a time."""
    return int((when.timestamp() * 1e6 - FIRST_US) // STEP_US)


def make_batch(*, stage: int) -> dict[str, np.ndarray]:
    """Return a batch of `ROWS` rows: target grid index, Primary, and day and week lag indices."""
    rng = np.random.default_rng(100 + stage)
    lo, hi = (grid_of(when=w) for w in STAGE_RANGES[stage])
    target = rng.integers(lo + 400, hi, size=ROWS)
    day = target - 48 * rng.integers(1, 4, size=ROWS)
    week = target - 336 * rng.integers(1, 3, size=ROWS)
    day[rng.random(ROWS) < 0.05] = -1
    week[rng.random(ROWS) < 0.05] = -1
    return {
        "target": target,
        "primary": rng.integers(0, N_PRIMARIES, size=ROWS),
        "day": day,
        "week": week,
        "noise": rng.normal(size=(ROWS, 60)).astype(np.float32),
        "labels": rng.normal(size=(ROWS, 2)).astype(np.float32),
    }


def run_arm(
    *,
    arm: cols.ArmB,
    stage: int,
    batch: dict[str, np.ndarray],
    c_true: np.ndarray,
    c_permuted: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the date-time encoder's input at each path (name to array), after a forward pass of the arm's network."""
    width = arm.width
    stage_name = "pre" if stage == 1 else "ft"
    if arm.name == "R0":
        net: torch.nn.Module = aux.AuxLagMdn(
            n_primaries=N_PRIMARIES,
            stage=stage_name,
            n_knots=N_KNOTS,
            n_b1=N_B1,
            kind="shared",
            detector=False,
        )
    else:
        net = model735.make_network(
            n_primaries=N_PRIMARIES, stage=stage_name, n_knots=N_KNOTS, n_b1=N_B1, n_new=width
        )
    valid = FIRST_US + batch["target"].astype(np.int64) * STEP_US
    x = np.concatenate([batch["noise"], np.zeros((ROWS, width), dtype=np.float32)], axis=1)
    x[:, 43] = batch["day"]
    x[:, 44] = batch["week"]
    if width:
        x[:, 60:] = cols.extra_columns(
            valid_us=valid, primary=batch["primary"], arm=arm, c_true=c_true, c_permuted=c_permuted
        )
        net.table_new = torch.from_numpy(  # ty: ignore[invalid-assignment]
            cols.grid_table(
                arm=arm,
                first_us=FIRST_US,
                step_us=STEP_US,
                n_grid=N_GRID,
                c_true=c_true,
                c_permuted=c_permuted,
            )
        )
    net.table = torch.zeros(N_PRIMARIES, N_GRID, 28)  # ty: ignore[invalid-assignment]
    seen: list[torch.Tensor] = []
    net.datetime.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))  # ty: ignore[unresolved-attribute]
    x_in = torch.from_numpy(x)
    if stage == 3:
        x_in = torch.cat([x_in, torch.from_numpy(batch["labels"])], dim=1)
        net.train()
    else:
        net.eval()
    out = net(x_in, torch.from_numpy(batch["primary"]))
    assert torch.isfinite(out).all(), (arm.name, stage, "the forward pass is not finite")
    assert len(seen) == 3, (arm.name, stage, len(seen))
    return dict(zip(PATHS, (s.numpy() for s in seen), strict=True))


def expected_columns(
    *,
    arm: cols.ArmB,
    batch: dict[str, np.ndarray],
    path: str,
    c_true: np.ndarray,
    c_permuted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the columns that the path's encoder input must hold, and the mask of rows that have a value there."""
    idx = batch["target"] if path == "row" else batch["day" if path == "day lag" else "week"]
    present = idx >= 0
    valid = FIRST_US + np.where(present, idx, batch["target"]).astype(np.int64) * STEP_US
    block = cols.extra_columns(
        valid_us=valid, primary=batch["primary"], arm=arm, c_true=c_true, c_permuted=c_permuted
    )
    return np.where(present[:, None], block, 0.0).astype(np.float32), present


PLAN_ARMS: Final = ("R1", "SKE", "SKE-S21", "SKE-CP")
PLAN_MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
PLAN_SEEDS: Final = (0, 1, 2)
MIN_SCHOOL_DIFFERENCE: Final = 0.05
MIN_COAST_PRIMARIES_DIFFERING: Final = 14


def read_json_checks(
    *, folder: Path, arms: tuple[str, ...] = PLAN_ARMS, months: tuple[str, ...] = PLAN_MONTHS,
    seeds: tuple[int, ...] = PLAN_SEEDS,
) -> pl.DataFrame:  # fmt: skip
    """Read the runner's stage-gate files under a folder (one subfolder per arm), assert their coverage and contents, and return the table.

    The files found must be exactly those of every arm, stage 1 to 3, month, and seed that the
    arguments name. Every batch has rows. An arm with the columns shows them equal to the expected
    columns on every path; its training row path has S1 to S3 with both 0 and 1 and every column
    non-constant. R1 shows none. Across arms (for the arms present): SKE-S21's S1 to S3 differ from
    SKE's on at least 5% of the rows of the same sample, and SKE-CP's coast columns differ from SKE's
    at 14 or more of the 16 Primaries.
    """
    records: dict[tuple[str, int, str, int], dict] = {}
    for file in sorted(folder.rglob("checks_stage_*_s*.json")):
        record = json.loads(file.read_text())
        key = (record["arm"], record["stage"], record["month"], record["seed"])
        assert key not in records, ("two files for", key)
        records[key] = record
    expected = {(a, st, m, sd) for a in arms for st in (1, 2, 3) for m in months for sd in seeds}
    assert set(records) == expected, (
        "missing:", sorted(expected - set(records))[:6], "unexpected:", sorted(set(records) - expected)[:6],
    )  # fmt: skip
    rows: list[dict[str, object]] = []
    for (arm, stage, month, seed), record in sorted(records.items()):
        assert arm in cols.ARMS and arm != "R0", arm
        spec = cols.ARMS[arm]
        assert record["n_new"] == spec.width, (arm, stage, month, seed, record["n_new"])
        for batch, body in record["batches"].items():
            assert body["n_rows"] > 0, (arm, stage, month, seed, batch, "no rows")
            for path, entry in body["paths"].items():
                where = (arm, stage, month, seed, batch, path)
                if not spec.columns:
                    assert (
                        entry["columns"] == []
                        and entry["input_width"] == 21 + record["embedding_dim"]
                    ), where
                    rows.append(
                        {
                            "arm": arm, "stage": stage, "month": month, "seed": seed, "batch": batch,
                            "path": path, "column": "none", "distinct": 0, "share_of_ones": 0.0,
                        }
                    )  # fmt: skip
                    continue
                assert entry["equals_expected"] is True, where
                assert len(entry["columns"]) == cols.N_COLUMNS, where
                for j, column in enumerate(entry["columns"]):
                    if batch == "train" and path == "row":
                        assert column["distinct"] > 1, (where, column)
                        if j < cols.SCHOOL_SLICE.stop:
                            assert column["min"] == 0.0 and column["max"] == 1.0, (
                                where,
                                column,
                                "lacks 0 or 1",
                            )
                    rows.append(
                        {
                            "arm": arm, "stage": stage, "month": month, "seed": seed, "batch": batch,
                            "path": path, "column": column["column"], "distinct": column["distinct"],
                            "share_of_ones": round(column["share_one"], 4),
                        }
                    )  # fmt: skip
    _check_contrasts(records=records, arms=arms, months=months, seeds=seeds)
    return pl.DataFrame(rows)


def _check_contrasts(
    *, records: dict[tuple[str, int, str, int], dict], arms: tuple[str, ...], months: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:  # fmt: skip
    """Assert the plan's contrasts between arms on the same rows of the training batch."""
    for stage in (1, 2, 3):
        for month in months:
            for seed in seeds:
                if "SKE" not in arms:
                    continue
                ske = records[("SKE", stage, month, seed)]["batches"]["train"]
                where = (stage, month, seed)
                if "SKE-S21" in arms:
                    other = records[("SKE-S21", stage, month, seed)]["batches"]["train"]
                    assert other["row_hash"] == ske["row_hash"], (
                        where,
                        "SKE and SKE-S21 sampled different rows",
                    )
                    a = np.asarray(ske["paths"]["row"]["school_sample"])
                    b = np.asarray(other["paths"]["row"]["school_sample"])
                    differ = float((a != b).any(axis=1).mean())
                    assert differ >= MIN_SCHOOL_DIFFERENCE, (
                        where,
                        "SKE-S21's school columns differ from SKE's on only",
                        differ,
                    )
                if "SKE-CP" in arms:
                    other = records[("SKE-CP", stage, month, seed)]["batches"]["train"]
                    assert other["row_hash"] == ske["row_hash"], (
                        where,
                        "SKE and SKE-CP sampled different rows",
                    )
                    means_a = ske["paths"]["row"]["per_primary_mean"]
                    means_b = other["paths"]["row"]["per_primary_mean"]
                    assert set(means_a) == set(means_b) and len(means_a) == N_PRIMARIES, (
                        where,
                        sorted(means_a),
                    )
                    differing = sum(
                        not np.allclose(
                            means_a[p][cols.SCHOOL_SLICE.stop :],
                            means_b[p][cols.SCHOOL_SLICE.stop :],
                        )
                        for p in means_a
                    )
                    assert differing >= MIN_COAST_PRIMARIES_DIFFERING, (
                        where,
                        "SKE-CP's coast columns differ at only",
                        differing,
                    )


def main() -> int:
    """Run the gate, print the table, and return 0 (an assertion raises otherwise)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--from-json", default=None, help="a folder of the runner's checks_stage_*.json files"
    )
    args = parser.parse_args()
    if args.from_json:
        frame = read_json_checks(
            folder=Path(args.from_json),
            arms=tuple(args.arms),
            months=tuple(args.months),
            seeds=tuple(args.seeds),
        )
        with pl.Config(tbl_rows=2000, tbl_cols=10, tbl_width_chars=180):
            print(frame)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            frame.write_csv(args.out)
        print("REAL-RUN STAGE GATE PASSED")
        return 0
    c_true, c_permuted = cols.coast_arrays()
    table_rows: list[dict[str, object]] = []
    captured: dict[tuple[str, int, str], np.ndarray] = {}
    for stage in (1, 2, 3):
        batch = make_batch(stage=stage)
        for name in ("R0", "R1", "SKE", "SKE-S21", "SKE-CP"):
            arm = cols.ARMS[name]
            inputs = run_arm(
                arm=arm, stage=stage, batch=batch, c_true=c_true, c_permuted=c_permuted
            )
            for path in PATHS:
                seen = inputs[path]
                if not arm.columns:
                    assert seen.shape[1] == 29, (name, stage, path, seen.shape)
                    table_rows.append(
                        {
                            "arm": name,
                            "stage": stage,
                            "path": path,
                            "column": "none",
                            "distinct": 0,
                            "share_of_ones": 0.0,
                            "share_nonzero": 0.0,
                            "rows": ROWS,
                        }
                    )
                    continue
                assert seen.shape[1] == 21 + cols.N_COLUMNS + 8, (name, stage, path, seen.shape)
                got = seen[:, 21 : 21 + cols.N_COLUMNS]
                want, present = expected_columns(
                    arm=arm, batch=batch, path=path, c_true=c_true, c_permuted=c_permuted
                )
                assert np.array_equal(got, want), (
                    name,
                    stage,
                    path,
                    "the encoder input is not the arm's columns",
                )
                captured[(name, stage, path)] = got
                use = got[present]
                for j, column in enumerate(cols.COLUMN_NAMES):
                    values = use[:, j]
                    distinct = int(np.unique(values).size)
                    assert distinct > 1, (name, stage, path, column, "is constant")
                    if j < 3:
                        assert values.min() == 0.0 and values.max() == 1.0, (
                            name,
                            stage,
                            path,
                            column,
                            "lacks 0 or 1",
                        )
                    table_rows.append(
                        {
                            "arm": name, "stage": stage, "path": path, "column": column,
                            "distinct": distinct, "share_of_ones": float((values == 1.0).mean()),
                            "share_nonzero": float((values != 0.0).mean()), "rows": int(present.sum()),
                        }
                    )  # fmt: skip
        for path in PATHS:
            ske, s21, cp = (captured[(n, stage, path)] for n in ("SKE", "SKE-S21", "SKE-CP"))
            differ = (s21[:, cols.SCHOOL_SLICE] != ske[:, cols.SCHOOL_SLICE]).any(axis=1).mean()
            assert differ >= MIN_SHIFT_SHARE, (
                stage,
                path,
                "SKE-S21's school columns differ on too few rows",
                differ,
            )
            primary_differs = [
                bool((cp[batch["primary"] == p][:, cols.COAST_SLICE] != ske[batch["primary"] == p][:, cols.COAST_SLICE]).any())
                for p in range(N_PRIMARIES)
            ]  # fmt: skip
            assert sum(primary_differs) >= MIN_PERMUTED_PRIMARIES, (
                stage,
                path,
                "SKE-CP's coast columns differ at too few Primaries",
                sum(primary_differs),
            )
            print(
                f"stage {stage} {path}: SKE-S21 school columns differ from SKE's on {differ:.1%} of rows;"
                f" SKE-CP coast columns differ at {sum(primary_differs)} of {N_PRIMARIES} Primaries",
                flush=True,
            )
    frame = pl.DataFrame(table_rows)
    with pl.Config(tbl_rows=400, tbl_cols=10, tbl_width_chars=160):
        print(frame)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(args.out)
    print("EVERY-STAGE GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

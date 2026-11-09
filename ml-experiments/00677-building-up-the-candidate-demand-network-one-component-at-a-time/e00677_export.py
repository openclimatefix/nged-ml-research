# ruff: noqa: D103, E501, PT018
"""Experiment 677: turn each ladder step's fits into per-seed contract rows (private Parquet).

The networks run at their native origin of 09:00 UTC. Experiment 667's `contract_rows` relabels it
as the contract origin 12:00 UTC and keeps the 672 half-hours after it. This module applies that
function to each seed separately, so the file holds one row per (seed, series, origin, valid time),
and adds the columns `seed`, `step`, `version`, and `NRA_or_prevailing` (always "NRA": the survival
head exists only in experiment 667's file).

Usage: python e00677_export.py STEP SPAN MONTH [MONTH ...] [--seeds 0 1 2]
STEP is one of 1, 2, 3a, 3b, 4, 5w, 5wo, 6w, 6wo, 7w, 7wo, 8w, 8wo. Writes
`backtests/step<N>_<with|without|ens_only|...>.parquet` once per call (all MONTHs given).
"""

import argparse
import os
from pathlib import Path
from typing import Final

import e00667_export as x667  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
ROOT: Final[Path] = Path(os.environ["E00677_ROOT"]) if os.environ.get("E00677_ROOT") else ML_EXPERIMENTS_DATA_DIR / NAME
CAND_ARMS: Final = {"6": "P3_D50", "7": "P5_D50", "8": "P7_D50"}
VERSION: Final = {
    "1": "ens_only", "2": "ens_only", "3a": "ens_only", "3b": "pretrained", "4": "pretrained",
    "5w": "with_injected", "5wo": "without_injected", "6w": "with_injected", "6wo": "without_injected",
    "7w": "with_injected", "7wo": "without_injected", "8w": "with_injected", "8wo": "without_injected",
}


def fits_dir(*, step: str, span: int, month: str) -> Path:
    if step[0] in CAND_ARMS:
        return ROOT / f"{CAND_ARMS[step[0]]}_{month}_{step}_sp{span}" / "work" / "fits" / "P5"
    return ROOT / "arms" / f"k1_{step}_sp{span}" / "work" / "fits" / "P5"


def step_rows(*, step: str, span: int, months: list[str], seeds: list[int]) -> pl.DataFrame:
    parts = []
    for month in months:
        for seed in seeds:
            raw = pl.read_parquet(
                fits_dir(step=step, span=span, month=month) / f"{month}_d0_s{seed}.parquet",
                columns=[*x667.KEYS, *x667.QCOLS],
            )
            assert raw.height == raw.unique(x667.KEYS).height, f"duplicate keys: {step} {month} s{seed}"
            parts.append(x667.contract_rows(mean=raw.sort(x667.KEYS)).with_columns(seed=pl.lit(seed, pl.Int8)))
    out = pl.concat(parts).with_columns(
        step=pl.lit(step),
        version=pl.lit(VERSION[step]),
        NRA_or_prevailing=pl.lit("NRA"),
    )
    assert out.height == out.unique(["seed", *x667.KEYS]).height, "duplicate keys across folds"
    assert out["valid_time"].dtype == pl.Datetime("us", "UTC")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=tuple(VERSION))
    parser.add_argument("span", type=int)
    parser.add_argument("months", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    frame = step_rows(step=args.step, span=args.span, months=args.months, seeds=args.seeds)
    out = ROOT / "backtests"
    out.mkdir(exist_ok=True)
    target = out / f"step{args.step}_{VERSION[args.step]}.parquet"
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(target)
    print(target.name, frame.height, "rows;", frame["power_fcst_init_time"].n_unique(), "origins;", sorted(frame["seed"].unique().to_list()))


if __name__ == "__main__":
    main()

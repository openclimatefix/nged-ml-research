# ruff: noqa: E501
"""Experiment 702: turn each arm's per-Primary fits into per-seed contract rows (private Parquet).

Usage: python e00702_export.py ARM MONTH [MONTH ...] [--seeds 0 1 2] [--suffix _b8192] [--name STEM]
       [--primaries D14,D2,D15]

ARM is `split1`, `split2`, `split3`, `hybrid3`, `global_pair`, or `global_pilot`. `--primaries`
restricts the output to the listed Primaries (the first run's three), and then checks only that
number of series. The per-substation arms store one
forecast file per fold, Primary, and seed (`arms/<arm>/<month>/<label>/work/fits/P5/`). This script
joins the 16 Primaries' files of each fold and seed, checks that each file holds one Primary and that
the 16 Primaries and all seeds share one set of keys, and applies experiment 667's `contract_rows`,
which relabels the native 09:00 UTC origin as the contract origin 12:00 UTC. The output
`backtests/<arm><suffix>.parquet` has the columns of experiment 677's backtest files plus `arm`.

`global_pair` and `global_pilot` read the fits of this experiment's `global` arm
(`e00702_fit.py global ...`), which hold all 16 Primaries in one file: `global_pair` the seeds 3 to 5
under `arms/global/<month>/pooled/work/fits/P5/`, and `global_pilot` the first run's seed 0 under
`arms/global_pilot/<month>/pooled/work/fits/P5/`. Run `e00702_export.py global_pilot 2026-05
--seeds 0 --name global_pilot` to write `backtests/global_pilot.parquet`, which `e00702_score.py
match-refit` reads.
"""

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00667_export as x667  # ty: ignore[unresolved-import]
import e00702_core as core
import e00702_score as score702
import numpy as np
import polars as pl

LABELS: Final[tuple[str, ...]] = (
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D18", "D19", "D21", "D22",
)  # fmt: skip
"""The 16 Primaries in `prep.primaries()` order."""


POOLED_FILE_ARMS: Final[dict[str, str]] = {"global_pair": "global", "global_pilot": "global_pilot"}
"""The export names whose fits sit in one file of all 16 Primaries, with the `e00702_fit.py` arm folder."""


def subset_contract_rows(*, mean: pl.DataFrame, n_series: int) -> pl.DataFrame:
    """Return experiment 667's `contract_rows` of a frame that holds `n_series` Primaries, fewer than 16.

    Experiment 667's function asserts 16 series. This copy makes the same relabelling, window,
    capacity scaling, and checks, and asserts `n_series` instead; `arm_rows` calls experiment 667's
    own function whenever all 16 Primaries are exported, and the tests compare the two on 16 series.
    """
    assert n_series < len(LABELS), "use experiment 667's `contract_rows` for all 16 Primaries"
    assert (mean["power_fcst_init_time"].dt.hour() == x667.NETWORK_ORIGIN_HOUR).all()
    assert (mean["power_fcst_init_time"].dt.minute() == 0).all()
    shift = timedelta(hours=x667.CONTRACT_HOUR - x667.NETWORK_ORIGIN_HOUR)
    cap = x667.capacity_mw().select(pl.col("time_series_id").cast(pl.Int64), "capacity_mw")
    out = (
        mean.with_columns(
            contract_origin=pl.col("power_fcst_init_time") + shift,
            time_series_id=pl.col("time_series_id").cast(pl.Int64),
        )
        .filter(
            (pl.col("valid_time") >= pl.col("contract_origin"))
            & (pl.col("valid_time") < pl.col("contract_origin") + timedelta(days=14))
        )
        .join(cap, on="time_series_id", how="left")
        .with_columns(
            *[
                (pl.col(q).cast(pl.Float64) * pl.col("capacity_mw")).cast(pl.Float32)
                for q in x667.QCOLS
            ]
        )
        .select(
            pl.col("time_series_id").cast(pl.Int8),
            pl.col("valid_time"),
            pl.col("contract_origin").dt.truncate("1d").alias("nwp_init_time"),
            pl.col("contract_origin").alias("power_fcst_init_time"),
            *x667.QCOLS,
        )
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )
    assert out.schema == pl.Schema(x667.CONTRACT_SCHEMA), out.schema
    assert out["q99"].is_finite().all()
    assert out.select(pl.all_horizontal(pl.col(x667.QCOLS).is_not_null())).to_series().all()
    q = out.select(x667.QCOLS).to_numpy()
    assert (np.diff(q, axis=1) >= 0).all(), "quantiles cross"
    per = out.group_by("time_series_id", "power_fcst_init_time").len()["len"].unique().to_list()
    assert per == [x667.HALF_HOURS], per
    assert out.height == out.unique(["time_series_id", "power_fcst_init_time", "valid_time"]).height
    assert out["time_series_id"].n_unique() == n_series
    return out


def fit_files(
    *, arm: str, suffix: str, month: str, seed: int, labels: tuple[str, ...]
) -> list[Path]:
    """Return the forecast files of one fold and seed: one per Primary, or one pooled file."""
    if arm in POOLED_FILE_ARMS:
        folder = core.ROOT / "arms" / f"{POOLED_FILE_ARMS[arm]}{suffix}" / month / "pooled" / "work"
        return [folder / "fits" / "P5" / f"{month}_d0_s{seed}.parquet"]
    folder = core.ROOT / "arms" / f"{arm}{suffix}" / month
    return [
        folder / label / "work" / "fits" / "P5" / f"{month}_d0_s{seed}.parquet" for label in labels
    ]


def seed_frame(
    *, arm: str, suffix: str, month: str, seed: int, labels: tuple[str, ...] = LABELS
) -> pl.DataFrame:
    """Return one fold and seed's native-origin forecasts of the Primaries in `labels`."""
    ids = score702.sites().filter(pl.col("label").is_in(list(labels)))["time_series_id"].to_list()
    assert len(ids) == len(labels), (labels, ids)
    parts = []
    for path in fit_files(arm=arm, suffix=suffix, month=month, seed=seed, labels=labels):
        assert path.exists(), path
        part = pl.read_parquet(path, columns=[*x667.KEYS, *x667.QCOLS])
        if arm not in POOLED_FILE_ARMS:
            assert part["time_series_id"].n_unique() == 1, f"{path} holds more than one Primary"
        parts.append(part.filter(pl.col("time_series_id").is_in(ids)))
    frame = pl.concat(parts).sort(x667.KEYS)
    assert frame["time_series_id"].n_unique() == len(labels), frame["time_series_id"].n_unique()
    assert frame.height == frame.unique(x667.KEYS).height, f"duplicate keys: {arm} {month} s{seed}"
    return frame


def arm_rows(
    *, arm: str, suffix: str, months: list[str], seeds: list[int], labels: tuple[str, ...] = LABELS
) -> pl.DataFrame:
    """Return the arm's contract rows for every month and seed."""
    out = []
    for month in months:
        keys = None
        for seed in seeds:
            raw = seed_frame(arm=arm, suffix=suffix, month=month, seed=seed, labels=labels)
            this = raw.select(x667.KEYS)
            keys = this if keys is None else keys
            assert this.equals(keys), f"seeds differ in keys: {arm} {month} s{seed}"
            rows = (
                x667.contract_rows(mean=raw)
                if len(labels) == len(LABELS)
                else subset_contract_rows(mean=raw, n_series=len(labels))
            )
            out.append(rows.with_columns(seed=pl.lit(seed, pl.Int8)))
    frame = pl.concat(out).with_columns(arm=pl.lit(arm + suffix))
    assert frame.height == frame.unique(["seed", *x667.KEYS]).height
    return frame


def main() -> None:
    """Write `backtests/<arm><suffix>.parquet`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=(*core.ARMS, *POOLED_FILE_ARMS))
    parser.add_argument("months", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--primaries", default="all", help="comma-separated labels, or all")
    parser.add_argument(
        "--name", default=None, help="output file stem (default: the arm plus the suffix)"
    )
    args = parser.parse_args()
    seeds = args.seeds or list(core.GLOBAL_PAIR_SEEDS if args.arm == "global_pair" else core.SEEDS)
    labels = LABELS if args.primaries == "all" else tuple(args.primaries.split(","))
    assert set(labels) <= set(LABELS), labels
    frame = arm_rows(
        arm=args.arm, suffix=args.suffix, months=args.months, seeds=seeds, labels=labels
    )
    target = core.ROOT / "backtests" / f"{args.name or args.arm + args.suffix}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(target)
    print(
        target.name,
        frame.height,
        "rows;",
        frame["power_fcst_init_time"].n_unique(),
        "origins;",
        seeds,
    )


if __name__ == "__main__":
    main()

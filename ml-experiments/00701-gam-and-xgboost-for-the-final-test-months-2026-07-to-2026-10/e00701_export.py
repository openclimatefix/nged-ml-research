# ruff: noqa: E501
"""Experiment 701: export the XGBoost forecasts of 2026-07 to 2026-10 in experiment 700's layout.

Reads the arms that `e00701_assemble.py build` wrote (`p_s6_bg2`, BG2, and
`p_a3_monthly_in_weekly_slot`, its monthly-only variant), keeps the origins from 2026-07-01 09:00
UTC, and writes `backtests/xgb_bg2_year_sp1.parquet` and `backtests/xgb_monthly_year_sp1.parquet`
under this experiment's private folder. The conversion is experiment 700's `to_contract` and
`write`, imported unchanged: the native origin (09:00 UTC) is relabelled to the 12:00 UTC contract
origin, valid times from 12:00 UTC to 14 days later are kept (672 half-hours), `nwp_init_time` is
the 00:00 UTC run, and the quantiles are converted from fractions of effective capacity to MW (MVA
at the MVA-only Primaries). `e00700_export.ROOT` is pointed at this experiment's folder, so the
files are not written next to experiment 700's.

The file's `step` column holds the arm name. A frame of the stored origins would also pass through
the same code, so `--check-stored` exports experiment 87's stored arms for 2026-06 and compares them
with experiment 700's stored files on the shared rows (a check of the export, not a forecast).

    PYTHONPATH=<folders of experiments 87, 33, 48, 700, 377, 701> \
        uv run --no-sync python ml-experiments/00701-*/e00701_export.py [--arms ...] [--check-stored]
"""

import argparse
from datetime import UTC, datetime

import e00700_export as e700  # ty: ignore[unresolved-import]
import e00701_common as mine
import numpy as np
import polars as pl

e700.ROOT = mine.ROOT
FIRST_ORIGIN = datetime(2026, 7, 1, 9, tzinfo=UTC)
LAST_ORIGIN = datetime(2026, 10, 31, 9, tzinfo=UTC)
OUTPUTS = {
    "p_s6_bg2": "xgb_bg2_year_sp1",
    "p_a3_monthly_in_weekly_slot": "xgb_monthly_year_sp1",
}


def arm_rows(*, arm: str) -> pl.DataFrame:
    """Return an assembled arm's new-run Primary rows, quantiles as fractions of capacity."""
    return (
        pl.scan_parquet(mine.ASSEMBLE_DIR / "arms" / f"{arm}.parquet")
        .filter(
            (pl.col("power_fcst_init_time") >= FIRST_ORIGIN)
            & (pl.col("power_fcst_init_time") <= LAST_ORIGIN)
            & (pl.col("group") == "Primary")
        )
        .select("site", "power_fcst_init_time", "valid_time", "lead_hours", *e700.Q)
        .collect()
    )


def export(*, arms: list[str]) -> None:
    """Write the forecast files of the named arms and check their row counts."""
    for arm in arms:
        name = OUTPUTS[arm]
        rows = arm_rows(arm=arm)
        runs = rows["power_fcst_init_time"].n_unique()
        sites = rows["site"].n_unique()
        print(f"{arm}: {rows.height:,} native rows, {runs} origins, {sites} Primaries")
        if runs != (LAST_ORIGIN - FIRST_ORIGIN).days + 1 or sites != 16:
            msg = f"{arm}: expected 123 origins and 16 Primaries"
            raise SystemExit(msg)
        contract = e700.to_contract(frame=rows, unit="fraction")
        per = (
            contract.group_by("time_series_id", "power_fcst_init_time")
            .len()["len"]
            .unique()
            .to_list()
        )
        if contract["time_series_id"].n_unique() != 16:
            msg = f"{arm}: the contract rows hold {contract['time_series_id'].n_unique()} series, not 16"
            raise SystemExit(msg)
        if per != [672]:
            msg = f"{arm}: rows per series and origin are {per}, not 672"
            raise SystemExit(msg)
        e700.write(frame=contract, name=name, step=arm)


def check_stored() -> None:
    """Export experiment 87's stored arms for 2026-06 and compare them with experiment 700's files."""
    for arm, name in OUTPUTS.items():
        stored = (
            pl.scan_parquet(mine.E87_WORK / "arms" / f"{arm}.parquet")
            .filter(
                (pl.col("power_fcst_init_time") >= datetime(2026, 6, 1, 9, tzinfo=UTC))
                & (pl.col("power_fcst_init_time") <= datetime(2026, 6, 30, 9, tzinfo=UTC))
                & (pl.col("group") == "Primary")
            )
            .select("site", "power_fcst_init_time", "valid_time", "lead_hours", *e700.Q)
            .collect()
        )
        mine_rows = e700.to_contract(frame=stored, unit="fraction")
        theirs = pl.read_parquet(
            e700.ML_EXPERIMENTS_DATA_DIR / e700.NAME / "backtests" / f"{name}.parquet"
        ).filter(
            (pl.col("power_fcst_init_time") >= datetime(2026, 6, 1, 12, tzinfo=UTC))
            & (pl.col("power_fcst_init_time") <= datetime(2026, 6, 30, 12, tzinfo=UTC))
        )
        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
        joined = mine_rows.join(theirs, on=keys, suffix="_700")
        gap = np.abs(
            joined.select(e700.Q).to_numpy()
            - joined.select([f"{q}_700" for q in e700.Q]).to_numpy()
        ).max()
        print(
            f"{name}: {joined.height:,} rows shared with experiment 700's file "
            f"({mine_rows.height:,} here, {theirs.height:,} there); largest difference {gap:.3g} MW"
        )
        if joined.height != theirs.height or gap > 0.0:
            msg = f"{name}: the export does not reproduce experiment 700's file"
            raise SystemExit(msg)


def main() -> None:
    """Export, or check the export on the stored origins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-stored", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=sorted(OUTPUTS), default=sorted(OUTPUTS))
    args = parser.parse_args()
    if args.check_stored:
        check_stored()
    else:
        export(arms=args.arms)


if __name__ == "__main__":
    main()

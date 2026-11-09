# ruff: noqa: D103, E501
"""Experiment 667, prevailing rows, step 4: write the contract rows of each fold month.

Reads the head's real-row quantiles (`fits/Rhaz_<month>_s<seed>.npz`, seed mean then sorted, or
the fallback `fits/Bhaz_<month>.npz`) and turns them into the contract's columns with experiment
667's own `contract_rows` (origin relabelled 09:00 to 12:00, valid times 12:00 to +14 days,
capacity in MW or MVA, 672 half-hours per series and origin), plus `NRA_or_prevailing =
"prevailing"` as the last column. Writes `exports/prevailing_<month>.parquet` and appends the
fold's method to `exports/prevailing_folds.csv`.

Usage: python p667_export.py MONTH ...
"""

import sys
from datetime import timedelta
from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00667_export as x667  # ty: ignore[unresolved-import]
import numpy as np
import p667_data as dat
import polars as pl

FITS: Final = dat.ROOT / "prevailing" / "fits"
SEEDS: Final = x667.SEEDS
MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
)  # fmt: skip


def head_quantiles(*, month: str) -> tuple[np.ndarray, str]:
    """Return (real-row quantiles of the head as a fraction of capacity, method name)."""
    rhaz = [FITS / f"Rhaz_{month}_s{s}.npz" for s in SEEDS]
    if all(f.exists() for f in rhaz):
        q = np.mean([np.load(f)["real_q"] for f in rhaz], axis=0)
        return np.sort(q, axis=1).astype(np.float32), "Rhaz"
    bhaz = FITS / f"Bhaz_{month}.npz"
    assert bhaz.exists(), f"no head fit for {month}"
    return np.load(bhaz)["real_q"].astype(np.float32), "Bhaz"


def prevailing_month(*, month: str) -> tuple[pl.DataFrame, str]:
    q, method = head_quantiles(month=month)
    r = np.load(dat.HEAD_DATA / f"real_{month}.npz")
    assert len(q) == len(r["lead"]), (len(q), len(r["lead"]))
    sm = k578.sites()
    ts_of = dict(zip(sm["primary"].to_list(), sm["time_series_id"].to_list(), strict=True))
    grid0 = k578.GRID_START
    frame = pl.DataFrame(
        {
            "time_series_id": [ts_of[int(p)] for p in r["prim"]],
            "power_fcst_init_time": [grid0 + timedelta(minutes=30 * int(i)) for i in r["oi"]],
            "valid_time": [grid0 + timedelta(minutes=30 * int(i)) for i in r["vi"]],
            **{name: q[:, j] for j, name in enumerate(x667.QCOLS)},
        }
    ).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )
    return x667.contract_rows(mean=frame).with_columns(
        NRA_or_prevailing=pl.lit("prevailing")
    ), method


def main() -> None:
    out = x667.ROOT / "exports"
    out.mkdir(exist_ok=True)
    for month in sys.argv[1:]:
        frame, method = prevailing_month(month=month)
        earlier = [m for m in MONTHS if m < month]
        tmp = out / f"prevailing_{month}.parquet.tmp"
        frame.write_parquet(tmp)
        tmp.rename(out / f"prevailing_{month}.parquet")
        print(
            month,
            method,
            frame.height,
            "rows;",
            frame["power_fcst_init_time"].n_unique(),
            "origins",
            flush=True,
        )
        folds = out / "prevailing_folds.csv"
        lines = [
            ln
            for ln in (
                folds.read_text().splitlines()
                if folds.exists()
                else ["month,method,earlier_banks,trained_on"]
            )
            if not ln.startswith(month)
        ]
        folds.write_text(
            "\n".join(
                [
                    *lines,
                    f"{month},{method},{len(earlier)},{' '.join(earlier) if method == 'Rhaz' else 'none'}",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

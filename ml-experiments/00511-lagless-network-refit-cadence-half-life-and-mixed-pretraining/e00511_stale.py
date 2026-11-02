# ruff: noqa: E501
"""Staleness curve and injected-shift leak of experiment 511 (free: reads fitted forecasts only).

- `curve`: CRPS (points of effective capacity) by model age, for each arm. Each origin day is scored by
  the model whose cutoff is the latest at least `age` days earlier (`e00511_fit.py stale`). Rows are
  those scored by every age 0 to 7 of every listed arm, rows with no labelled event, at the eight and
  the signed-MW six.
- `leak`: the injected-shift leak of experiment 510's definition for L-1d (by block age) and L-Ww-30.
"""

import argparse
from pathlib import Path
from typing import Final

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00511_common as k
import e00511_score as score
import numpy as np
import polars as pl

W: Final[Path] = k.WORK_DIR
ARMS: Final[tuple[str, ...]] = ("L-3d", "L-1d", "L-Ww-30", "L-Ww-60", "L-mix")
AGES: Final[range] = range(8)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
GROUPS: Final[dict[str, pl.Expr]] = {
    "eight": pl.col("group") == "mva_eight",
    "six": (pl.col("group") == "signed_six") & (pl.col("cls") == "clean"),
}


def curve() -> None:
    """Write CRPS by age per arm, group, and month-pooled rows."""
    score.configure()
    sites = build470.site_map()
    records = []
    for month in score.FIVE:
        rows = pl.read_parquet(W / "scores" / f"rows_{month}.parquet").select(
            "primary", "origin", "target", "group", "cls", "main16"
        )
        base = score454.month_base(month=month).join(
            rows, on=["primary", "origin", "target"], how="inner"
        )
        per = {}
        for arm in ARMS:
            for age in AGES:
                folder = W / "stale_seed" / f"{arm}_a{age}"
                folder.mkdir(parents=True, exist_ok=True)
                for s in (0, 1):
                    src = W / "stale" / arm / f"{month}_a{age}_s{s}.parquet"
                    if src.exists() and not (folder / f"{month}_s{s}.parquet").exists():
                        (folder / f"{month}_s{s}.parquet").symlink_to(src)
                if not all((folder / f"{month}_s{s}.parquet").exists() for s in (0, 1)):
                    continue
                frame = build470.on_rows(
                    rows=base.select(*build470.KEYS),
                    frame=build470.pool_stored(
                        folder=folder, month=month, sites=sites, seeds=(0, 1)
                    ),
                )
                frame = base.select(*build470.KEYS, "actual").join(
                    frame, on=build470.KEYS, how="left", maintain_order="left"
                )
                frame = score470.widen_arm(arm=arm, frame=frame, factor=score470.HEADLINE)
                ok = np.isfinite(frame["q50"].to_numpy())
                crps = score470.row_metrics(
                    frame=frame.with_columns(*[pl.col(c).fill_null(0.0) for c in build470.LEVELS])
                )["crps"]
                per[(arm, age)] = np.where(ok, crps, np.nan)
        if not per:
            continue
        table = base.select("group", "cls", "main16").with_columns(
            **{f"{a}__{g}": pl.Series(v) for (a, g), v in per.items()}
        )
        common_ok = pl.all_horizontal([pl.col(f"{a}__{g}").is_finite() for (a, g) in per]) & pl.col(
            "main16"
        )
        for gname, expr in GROUPS.items():
            part = table.filter(common_ok & expr)
            for arm, age in per:
                records.append(
                    {
                        "month": month,
                        "group": gname,
                        "arm": arm,
                        "age": age,
                        "n": part.height,
                        "crps": float(part[f"{arm}__{age}"].mean()),
                    }
                )
    out = pl.DataFrame(records)
    pooled = out.group_by("group", "arm", "age", maintain_order=True).agg(
        pl.col("crps").mean(), pl.col("n").sum()
    )
    pooled.write_csv(W / "scores" / "staleness_curve.csv")
    out.write_csv(W / "scores" / "staleness_curve_by_month.csv")
    with pl.Config(tbl_rows=100):
        print(
            pooled.pivot(on="age", index=["group", "arm"], values="crps").with_columns(
                pl.col(pl.Float64).round(3)
            )
        )


def leak() -> None:
    """Return the injected-shift leak (median shift in points of capacity, signed by the injected sign)."""
    records = []
    for arm, days in (("L-1d", list(range(14, 21))), ("L-Ww-30", [14])):
        for month in score.FIVE:
            for day in days:
                control = pl.read_parquet(W / "fits" / arm / f"{month}_d{day}_s0.parquet").select(
                    *KEYS, c=pl.col("q50")
                )
                signed = []
                for name, sign in (("p", 1), ("n", -1)):
                    st = pl.read_parquet(
                        W / "fits" / arm / f"{month}_d{day}_s0_stress{name}.parquet"
                    ).select(*KEYS, s=pl.col("q50"))
                    signed.append(
                        control.join(st, on=KEYS).with_columns(
                            lead=(
                                pl.col("valid_time") - pl.col("power_fcst_init_time")
                            ).dt.total_hours(),
                            shift=sign * 100.0 * (pl.col("s") - pl.col("c")),
                        )
                    )
                row = {"arm": arm, "month": month, "block_age_days": day - 14}
                for label, cond in {
                    "all": pl.lit(True),
                    "day1": pl.col("lead") <= 24,
                    "days2_3": (pl.col("lead") > 24) & (pl.col("lead") <= 72),
                    "days4_10": pl.col("lead") > 72,
                }.items():
                    row[label] = float(np.mean([p.filter(cond)["shift"].mean() for p in signed]))
                records.append(row)
    table = pl.DataFrame(records)
    table.write_csv(W / "scores" / "stress_leak_by_fold.csv")
    summary = table.group_by("arm", "block_age_days", maintain_order=True).agg(
        pl.col("all").mean(),
        pl.col("all").min().alias("min"),
        pl.col("all").max().alias("max"),
        pl.col("day1").mean(),
        pl.col("days2_3").mean(),
        pl.col("days4_10").mean(),
    )
    summary.write_csv(W / "scores" / "stress_leak.csv")
    with pl.Config(tbl_rows=40):
        print(summary.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("curve", "leak"))
    {"curve": curve, "leak": leak}[parser.parse_args().command]()

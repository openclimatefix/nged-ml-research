# ruff: noqa: E501
"""Audit checks for experiment 523's pretrained-minus-unpretrained contrast (L0PT minus LL).

1. The same contrast on the absolute error of the median forecast (audit trigger 3).
2. The CRPS contrast with each Primary left out in turn (audit trigger 4), at 120 to 240 h.
3. The median minus the mean of the stored quantiles per arm (a crude mean forecast: the mean of
   the eleven quantiles weighted by the atom masses), per Primary group.

```bash
source /tmp/e523_env.sh
uv run --no-sync python ml-experiments/00523-*/e00523_checks.py
```
"""

import e00087_common as common  # ty: ignore[unresolved-import]
import e00521_stats as st  # ty: ignore[unresolved-import]
import e00523_panel as panel  # ty: ignore[unresolved-import]
import e00523_scores as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl


def main() -> None:
    """Write `checks_median_and_leave_one_out.csv`."""
    table = sc.load_panel().filter(pl.col("month").is_in(list(sc.PT_MONTHS)))
    table = table.filter(
        pl.Series(
            sc.finite_rows(table=table, cells=("PTs0", "PTs1", "PTs2", "LLc0", "LLc1", "LLc2"))
        )
    )
    y = table["actual"].to_numpy().astype(np.float64)
    cols = {}
    for arm in ("L0PT", "LL"):
        cells = sc.CELLS[arm]
        cols[f"{arm}__mae"] = np.mean(
            [100.0 * np.abs(sc.quantiles(table=table, cell=c)[:, sc.MEDIAN] - y) for c in cells],
            axis=0,
        )
        cols[f"{arm}__crps"] = np.mean(
            [
                sc.metrics(q=sc.widen(q=sc.quantiles(table=table, cell=c), factor=sc.FLAT), y=y)[
                    "crps"
                ]
                for c in cells
            ],
            axis=0,
        )
        cols[f"{arm}__median_minus_mean"] = np.mean(
            [
                100.0
                * (
                    sc.quantiles(table=table, cell=c)[:, sc.MEDIAN]
                    - sc.quantiles(table=table, cell=c)
                    @ common.ATOM_MASSES
                    / common.ATOM_MASSES.sum()
                )
                for c in cells
            ],
            axis=0,
        )
    scored = table.select("group", "cls", "main16", "month", "day", "band", "primary").hstack(
        pl.DataFrame(cols)
    )
    records = []
    for name in sc.SETS2:
        part = st.row_set(table=scored, name=name)
        for band in ("0-6h", "12-24h", "120-240h", st.ALL):
            sub = part if band == st.ALL else part.filter(pl.col("band") == band)
            records.append(
                {"check": "mean |median error| L0PT minus LL (points)", "set": name, "band": band,
                 "value": float(sub["L0PT__mae"].mean() - sub["LL__mae"].mean())}
            )  # fmt: skip
            records.append(
                {"check": "median minus mean forecast, L0PT (points)", "set": name, "band": band,
                 "value": float(sub["L0PT__median_minus_mean"].mean())}
            )  # fmt: skip
            records.append(
                {"check": "median minus mean forecast, LL (points)", "set": name, "band": band,
                 "value": float(sub["LL__median_minus_mean"].mean())}
            )  # fmt: skip
        long = part.filter(pl.col("band") == "120-240h")
        for primary in sorted(long["primary"].unique().to_list()):
            rest = long.filter(pl.col("primary") != primary)
            records.append(
                {"check": f"CRPS L0PT minus LL at 120-240h without Primary index {primary}", "set": name,
                 "band": "120-240h", "value": float(rest["L0PT__crps"].mean() - rest["LL__crps"].mean())}
            )  # fmt: skip
    out = pl.DataFrame(records)
    out.write_csv(panel.DATA / "scores" / "checks_median_and_leave_one_out.csv")
    with pl.Config(tbl_rows=80, tbl_width_chars=200):
        print(out.with_columns(pl.col("value").round(3)))


if __name__ == "__main__":
    main()

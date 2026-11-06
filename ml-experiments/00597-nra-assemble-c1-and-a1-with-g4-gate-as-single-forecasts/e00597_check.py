# ruff: noqa: E501, D103, ANN001
"""Experiment 597 check: the exported candidate files reproduce the scored numbers.

Reads only the exported Parquet files (and experiment 593's export for the C1 comparison):

- C1 equals experiment 593's C1 file exactly, and C1_G4 has the same median as C1 and differs from C1 only on rows with `gate_on`
  (the union band moves every level except q50).
- The CRPS of C1 on event-free rows at 24-120 h reproduces experiment 584 (3.081, 2.465).
- Every quantile row is sorted, and `gate_on` is filled in the `_G4` files only.
- The share of rows with `gate_on` per system and the coverage of the NRA on the verified December pair.
"""

import e00597_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

Q = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
TAU = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
OLD = ML_EXPERIMENTS_DATA_DIR / "backtest_export_2026-11-06"
SYSTEMS = ("C1", "C1_G4", "A1", "A1_G4")


def read(*, folder, system: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(folder / f"{system}_{m}.parquet") for m in k.c.MONTHS])


def crps(df: pl.DataFrame, *, y: str = "nra_power") -> pl.DataFrame:
    yy = pl.col(y)
    pin = [
        pl.max_horizontal(t * (yy - pl.col(q)), (t - 1) * (yy - pl.col(q)))
        for q, t in zip(Q, TAU, strict=True)
    ]
    sub = pl.read_csv(k.EXPORT / "substations.csv")
    rows = df.join(sub, on="time_series_id").filter(
        pl.col("event_free") & (pl.col("lead_hours") >= 24) & yy.is_not_null()
    )
    return rows.group_by("group").agg(
        (2 * pl.mean_horizontal(*pin) / pl.col("capacity") * 100).mean().alias("crps")
    )


def main() -> None:
    tbl = {s: read(folder=k.EXPORT, system=s) for s in SYSTEMS}
    old_c1 = read(folder=OLD, system="C1")
    same = tbl["C1"].select(Q).equals(old_c1.select(Q))
    print("C1 quantiles equal experiment 593's C1 file:", same, tbl["C1"].height, old_c1.height)
    for arm in ("C1", "A1"):
        a, b = tbl[arm], tbl[f"{arm}_G4"]
        assert a.height == b.height
        gate = b["gate_on"].to_numpy()
        print(
            arm,
            "gate_on null in the plain file:",
            a["gate_on"].null_count() == a.height,
            "| null in the G4 file:",
            b["gate_on"].null_count(),
            "| share of rows with gate_on:",
            round(float(gate.mean()), 4),
        )
        for q in Q:
            diff = (a[q] != b[q]).to_numpy()
            ok = not diff.any() if q == "q50" else bool((diff <= gate).all())
            print(
                f"  {arm}_G4 vs {arm} {q}: " + ("as expected" if ok else "UNEXPECTED CHANGE"),
                "| rows changed",
                int(diff.sum()),
            )
    for s, df in tbl.items():
        q = df.select(Q).to_numpy()
        print(
            s,
            "rows",
            df.height,
            "quantiles sorted:",
            bool((np.diff(q, axis=1) >= 0).all()),
            "| columns",
            len(df.columns),
        )
        print(" ", crps(df).sort("group").to_dicts())
    print(
        "Experiment 584 for C1: eight MVA Primaries 3.081, six MW Primaries 2.465 (24-120 h, event-free rows)"
    )
    print(
        "\nCoverage of the NRA by q05-q95, verified December pair (event_id set), 0-7 days after the end, 24-120 h"
    )
    for s, df in tbl.items():
        d = df.filter(pl.col("lead_hours") >= 24)
        out = []
        for (ev, pid), g in d.filter(pl.col("in_verified_event")).group_by(
            "event_id", "time_series_id"
        ):
            end = g["valid_time"].max()
            after = d.filter(
                (pl.col("time_series_id") == pid)
                & (pl.col("valid_time") > end)
                & (pl.col("valid_time") <= end + pl.duration(days=7))
            )
            out.append(
                (
                    int(ev),
                    pid,
                    after.height,
                    float(
                        (  # ty: ignore[invalid-argument-type]
                            (after["q05"] <= after["nra_power"])
                            & (after["nra_power"] <= after["q95"])
                        ).mean()
                    ),
                )
            )
        print(s, sorted(out))


if __name__ == "__main__":
    main()

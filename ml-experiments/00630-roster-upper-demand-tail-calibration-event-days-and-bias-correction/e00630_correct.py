# ruff: noqa: E501
"""Experiment 630, part 2: fit a shift (S) or rescale (R) on the first three months, score the last two.

The correction is fitted per Primary and lead band on event-free rows of 2025-10, 2025-12 and
2026-01, applied to every row of 2026-03 and 2026-06, and scored on event-free rows (the gating
rows) and on all rows.
"""

import numpy as np
import polars as pl
from e00630_common import (
    BAND_LABELS,
    NETWORKS,
    TABLES,
    TAU,
    TEST_MONTHS,
    TRAIN_MONTHS,
    Q,
    bootstrap_ratio,
    day_table,
    load,
)

MEDIAN: int = Q.index("q50")
Q90: int = Q.index("q90")
Q95: int = Q.index("q95")
GRID = np.round(np.arange(1.0, 3.0001, 0.05), 2)


def crps(*, y: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Twice the mean pinball loss over the 11 levels, per row."""
    err = y[:, None] - quantiles
    return 2 * np.maximum(TAU * err, (TAU - 1) * err).mean(axis=1)


def fit(*, train: pl.DataFrame) -> pl.DataFrame:
    """Fit the shift and the rescale factor per Primary and lead band on the training rows."""
    rows = []
    for (pid, band), g in train.group_by("time_series_id", "band"):
        y = g["observed_power"].to_numpy().astype(np.float64)
        q = g.select(Q).to_numpy().astype(np.float64)
        d = float(np.median(y - q[:, MEDIAN]))
        up = q[:, Q90] - q[:, MEDIAN]
        shares = np.array([np.mean(y > q[:, MEDIAN] + c * up) for c in GRID])
        rows.append(
            {
                "time_series_id": pid,
                "band": band,
                "shift_mw": d,
                "factor": float(GRID[np.argmin(np.abs(shares - 0.10))]),
                "train_q90_exceedance": float(np.mean(y > q[:, Q90])),
                "n_train": len(y),
            }
        )
    return pl.DataFrame(rows)


def apply(*, test: pl.DataFrame, params: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the uncorrected, shifted (S), and rescaled (R) quantile arrays for the test rows."""
    t = test.join(
        params.select("time_series_id", "band", "shift_mw", "factor"),
        on=["time_series_id", "band"],
        how="left",
        maintain_order="left",
    )
    assert t["valid_time"].equals(test["valid_time"]), "join changed the row order"
    q = t.select(Q).to_numpy().astype(np.float64)
    d = t["shift_mw"].fill_null(0.0).to_numpy()
    c = t["factor"].fill_null(1.0).to_numpy()
    out = {"base": q}
    s = q.copy()
    s[:, MEDIAN + 1 :] += d[:, None]
    out["S"] = np.sort(s, axis=1)
    r = q.copy()
    r[:, MEDIAN + 1 :] = q[:, [MEDIAN]] + c[:, None] * (q[:, MEDIAN + 1 :] - q[:, [MEDIAN]])
    out["R"] = np.sort(r, axis=1)
    return out


def main() -> None:
    """Fit on the first three months, score the last two, and write the tables."""
    all_params = []
    for network in NETWORKS:
        df = load(network=network)
        train = df.filter(pl.col("month").is_in(TRAIN_MONTHS) & pl.col("event_free"))
        test = df.filter(pl.col("month").is_in(TEST_MONTHS))
        params = fit(train=train)
        all_params.append(params.with_columns(network=pl.lit(network)))
        arms = apply(test=test, params=params)
        y = test["observed_power"].to_numpy().astype(np.float64)
        cols = {}
        for arm, q in arms.items():
            cols[f"a90_{arm}"] = y > q[:, Q90]
            cols[f"a95_{arm}"] = y > q[:, Q95]
            cols[f"crps_{arm}"] = crps(y=y, quantiles=q)
        scored = test.select(
            "month", "origin_date", "event_free", "group", "primary_name", "band"
        ).with_columns([pl.Series(k, v) for k, v in cols.items()])
        values = list(cols)
        ratios = {f"crps_ratio_{a}": (f"crps_{a}", "crps_base") for a in ("S", "R")}
        res = []
        for subset, sdf in (("event-free", scored.filter(pl.col("event_free"))), ("all", scored)):
            for name, keys in (
                ("overall", []),
                ("group", ["group"]),
                ("band", ["band"]),
                ("month", ["month"]),
            ):
                days = day_table(df=sdf, keys=keys, values=values)
                res.append(
                    bootstrap_ratio(
                        days=days, keys=keys, values=values, ratios=ratios
                    ).with_columns(
                        subset=pl.lit(subset), slice=pl.lit(name), network=pl.lit(network)
                    )
                )
        out = pl.concat(res, how="diagonal")
        out.write_csv(TABLES / f"correction_{network}.csv")
        print(f"\n== {network}: fitted shifts (MW) and factors; train q90 exceedance")
        print(
            params.group_by("band")
            .agg(
                pl.col("shift_mw").median().round(3).alias("median_shift_mw"),
                pl.col("factor").median().alias("median_factor"),
                pl.col("train_q90_exceedance").mean().round(3),
            )
            .sort("band")
        )
        with pl.Config(tbl_rows=40, tbl_cols=30, fmt_float="mixed"):
            o = out.filter(pl.col("slice") == "overall").select(
                "subset",
                "n",
                "n_days",
                *[
                    pl.col(c).round(4)
                    for c in (
                        "a90_base",
                        "a90_S",
                        "a90_R",
                        "a95_base",
                        "a95_S",
                        "a95_R",
                        "crps_base",
                        "crps_ratio_S",
                        "crps_ratio_R",
                    )
                ],
                *[
                    pl.col(c).round(3)
                    for c in (
                        "a90_S_lo",
                        "a90_S_hi",
                        "a90_R_lo",
                        "a90_R_hi",
                        "a95_S_lo",
                        "a95_S_hi",
                        "a95_R_lo",
                        "a95_R_hi",
                        "crps_ratio_S_lo",
                        "crps_ratio_S_hi",
                        "crps_ratio_R_lo",
                        "crps_ratio_R_hi",
                    )
                ],
            )
            print(o)
    pl.concat(all_params).write_csv(TABLES / "correction_parameters.csv")
    _ = BAND_LABELS


if __name__ == "__main__":
    main()

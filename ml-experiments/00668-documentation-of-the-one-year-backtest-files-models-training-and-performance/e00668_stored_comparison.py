"""Compare BG2 and Primary demand research default 3 on their stored rows (6 MW Primaries).

uv run --no-sync python ml-experiments/00668-*/e00668_stored_comparison.py

Reads experiment 322's per-row losses (default 3, default 2, manual heuristic, climatology; 09:00
UTC origins, leads 0.5 to 240 h) and experiment 87's stored BG2 quantiles (`p_s6_bg2`, 09:00 UTC
origins, leads to 168 h), joined on series, origin, and valid time. BG2's quantiles are fractions of
capacity, so its CRPS is 100 times twice the mean pinball loss over the 11 levels, in points of
capacity like the losses of experiment 322. Rows are limited to valid times from 2025-11-01 12:00
UTC (the first valid time of the one-year backtest). Writes `scores_668/stored_comparison/` in the
backtest folder (private).
"""

import e00668_score_lib as lib
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

R = ML_EXPERIMENTS_DATA_DIR
START = "2025-11-01 12:00:00"
SYSTEMS = {
    "BG2": "loss_bg2",
    "Default 3": "loss_d3",
    "Default 2": "loss_default",
    "Manual heuristic": "loss_manual_heuristic",
    "Climatology": "loss_climatology",
}


def load() -> pl.DataFrame:
    """Return the joined rows with a `loss_bg2` column."""
    a = pl.read_parquet(next(R.glob("00322-*/work/report/scored_rows.parquet")))
    b = (
        pl.scan_parquet(next(R.glob("00087-*/work/arms/p_s6_bg2.parquet")))
        .filter(pl.col("site").is_in(a["site"].unique().to_list()))
        .select("site", "power_fcst_init_time", "valid_time", *lib.QCOLS, "actual")
        .rename({"actual": "actual_bg2"})
        .collect()
    )
    k = a.join(b, on=["site", "power_fcst_init_time", "valid_time"], how="inner")
    q = k.select(lib.QCOLS).to_numpy().astype(np.float64)
    d = k["actual_bg2"].to_numpy().astype(np.float64)[:, None] - q
    loss = 100.0 * 2.0 * np.maximum(lib.TAUS * d, (lib.TAUS - 1.0) * d).mean(axis=1)
    return k.with_columns(loss_bg2=pl.Series(loss)).filter(
        pl.col("valid_time") >= pl.lit(START).str.to_datetime(time_zone="UTC")
    )


def main() -> None:
    """Score the common rows by lead band, origin month, and condition."""
    k = load()
    k = k.filter(pl.col("loss_bg2").is_finite() & (pl.col("lead_hours") <= 168.0))
    out = lib.FOLDER / "scores_668" / "stored_comparison"
    out.mkdir(parents=True, exist_ok=True)
    day = (k["power_fcst_init_time"].dt.epoch("s") // 86400).to_numpy()
    loss = {name: k[col].to_numpy().astype(np.float64) for name, col in SYSTEMS.items()}
    lead = k["lead_hours"].to_numpy()
    month = k["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    rows: list[dict] = []

    def add(mask: np.ndarray, **tag: str) -> None:
        draws = lib.day_bootstrap(day=day[mask], values={n: v[mask] for n, v in loss.items()})
        for name, v in loss.items():
            lo, hi = np.quantile(draws[name], [0.025, 0.975])
            rows.append(
                {
                    **tag,
                    "system": name,
                    "crps": float(v[mask].mean()),
                    "lo": float(lo),
                    "hi": float(hi),
                    "rows": int(mask.sum()),
                }
            )

    add(np.ones(len(k), dtype=bool), slice="all rows", group="all")
    for name, (lo, hi) in lib.LEAD_BANDS.items():
        if lo < 168.0:
            add((lead >= lo) & (lead < min(hi, 168.5)), slice="lead band", group=name)
    for m in sorted(set(month)):
        add(month == m, slice="origin month", group=m)
    add(k["cold_winter_night"].to_numpy(), slice="condition", group="cold winter nights")
    add(k["sunny_daylight"].to_numpy(), slice="condition", group="sunny daylight")
    pl.DataFrame(rows).write_csv(out / "scores.csv")
    print(len(k), "common rows", k["valid_time"].min(), k["valid_time"].max(), flush=True)


if __name__ == "__main__":
    main()

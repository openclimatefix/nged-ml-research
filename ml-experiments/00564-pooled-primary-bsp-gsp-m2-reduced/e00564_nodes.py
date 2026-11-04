# ruff: noqa: E501
"""Score the 4 extra series (2 GSPs, 2 BSPs) of experiment 564's arms, beside a climatology reference.

The row set of experiment 511 holds Primaries only, so the extra series are scored here on every
half-hour of the five test months with a finite cleaned actual. CRPS is computed from the 11
quantiles with the atom masses of `e00087_common.ATOM_MASSES`, in points of effective capacity.

The reference is a climatology of the fold's own training window: for each (UTC month, half-hour
of day, weekend flag) cell, the empirical 11 quantile levels of the cleaned power over capacity in
the 24 months before the test month's first day, using only valid times at least 3.5 h earlier. The
single-network L-1d of experiment 511 was never fitted to these series, so it is not a reference.

Usage: python e00564_nodes.py ARM [ARM ...]
"""

import sys
from datetime import timedelta

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00564_pool as pool
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = ML_EXPERIMENTS_DATA_DIR / "00564-pooled-primary-bsp-gsp-m2-reduced" / "work"
BANDS = ((0, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 120), (120, 240))
POOLED = (24, 120)
MASSES = common.ATOM_MASSES


def crps(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the per-row CRPS from sorted quantiles (atom-mass form), in capacity fractions."""
    q = np.sort(q, axis=1)
    diff = actual[:, None] - q
    levels = common.LEVELS[None, :]
    pinball = np.where(diff >= 0, diff * levels, -diff * (1.0 - levels))
    return 2.0 * (pinball * MASSES[None, :]).sum(axis=1) / MASSES.sum()


def climatology(*, power: np.ndarray, month: str) -> dict[tuple[int, int, int], np.ndarray]:
    """Return the 11 quantiles of each (UTC month, half-hour, weekend) cell from the fold's training window."""
    grid = common.grid_times()
    start = c.month_start(month=month)
    hi = int(common.grid_index(times=start - regimes.REFIT_CUTOFF_AGE))
    lo = int(common.grid_index(times=start - timedelta(days=730)))
    idx = np.arange(max(lo, 0), hi + 1)
    t = grid.gather(idx.tolist())
    cell = np.stack(
        [t.dt.month().to_numpy(), (t.dt.hour() * 2 + t.dt.minute() // 30).to_numpy(), (t.dt.weekday() >= 6).to_numpy().astype(int)], axis=1
    )
    values = power[idx]
    out = {}
    for key in {tuple(r) for r in cell.tolist()}:
        sel = (cell == np.array(key)).all(axis=1) & np.isfinite(values)
        if sel.sum() >= 5:
            out[key] = np.quantile(values[sel], common.LEVELS)
    return out


def main(arms: list[str]) -> None:
    """Print each arm's CRPS by lead band at each extra series, with the climatology reference."""
    pool.install()
    clean = c.clean_stack()
    table = pool.pooled_table().filter(pl.col("primary") >= pool.N_PRIMARIES)
    id_to_row = dict(zip(table["time_series_id"], table["primary"], strict=True))
    records = []
    for month in c.MONTHS:
        for node in table.iter_rows(named=True):
            clim = climatology(power=clean[node["primary"]], month=month)
            for arm in ["climatology", *arms]:
                if arm == "climatology":
                    pred = pl.read_parquet(W / "node_preds" / arms[0] / f"{month}_s0.parquet")
                else:
                    pred = pl.concat(
                        [
                            pl.read_parquet(W / "node_preds" / arm / f"{month}_s{seed}.parquet")
                            for seed in (0, 1)
                        ]
                    )
                pred = pred.filter(pl.col("time_series_id") == node["time_series_id"])
                valid_idx = np.asarray(common.grid_index(times=pred["valid_time"]))
                actual = clean[node["primary"]][valid_idx]
                lead = (pred["valid_time"] - pred["power_fcst_init_time"]).dt.total_hours().to_numpy()
                if arm == "climatology":
                    t = pred["valid_time"]
                    keys = list(
                        zip(t.dt.month().to_list(), (t.dt.hour() * 2 + t.dt.minute() // 30).to_list(), (t.dt.weekday() >= 6).cast(int).to_list(), strict=True)
                    )
                    q = np.stack([clim.get(k, np.full(len(common.LEVELS), np.nan)) for k in keys])
                else:
                    q = pred.select(common.LEVEL_COLUMNS).to_numpy()
                score = crps(actual=actual, q=q)
                ok = np.isfinite(score)
                assert id_to_row[node["time_series_id"]] == node["primary"]
                for lo, hi in (*BANDS, POOLED):
                    sel = ok & (lead >= lo) & (lead < hi)
                    if sel.any():
                        records.append(
                            {"month": month, "site": node["site"], "kind": node["substation_type"], "arm": arm, "band": f"{lo}-{hi}h", "n": int(sel.sum()), "crps": 100.0 * float(score[sel].mean())}
                        )
    out = pl.DataFrame(records)
    (W / "scores").mkdir(parents=True, exist_ok=True)
    out.write_csv(W / "scores" / "nodes.csv")
    summary = (
        out.filter(pl.col("band") == "24-120h")
        .group_by("site", "kind", "arm")
        .agg((pl.col("crps") * pl.col("n")).sum() / pl.col("n").sum())
        .sort("site", "arm")
    )
    with pl.Config(tbl_rows=100):
        print(summary.pivot(on="arm", index=["site", "kind"], values="crps"))


if __name__ == "__main__":
    main(sys.argv[1:])

# ruff: noqa: E501
"""Experiment 640: score the stored fit-age forecasts into cells (CPU only).

For one network (`k1`: tag P5; `k3`: tag D5) and each of the 25 folds `YYYY-MM-AA`, this reads the
seed-mean quantiles of seeds 0 to 2, takes the actual power from the cleaned power (fraction of
capacity), keeps only the rows that experiment 613's roster scored (the same Primary, forecast
origin, and valid time, with its `free` flag, lead band and metering group), and writes one row per
(network, month, fit age, Primary, event-free flag, lead band, whole week since the fit) with
the row count and the sums of CRPS, absolute error of the median, q05-q95 and q01-q99 coverage, and the
pinball loss at 0.01, 0.05, 0.95, 0.99. The network is scored on its own quantiles (no G4 gate).

Usage: python e00640_score.py {k1|k3}
Output: `data/ml-experiments/00640-*/score/cells_{net}.parquet`.
"""

import sys
from datetime import timedelta

import e00422_prep as prep  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00640_fold as fold
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NET = sys.argv[1]
TAG = {"k1": "P5", "k3": "D5"}[NET]
SEEDS = (0, 1, 2)
ROOT = ML_EXPERIMENTS_DATA_DIR / "00640-demand-fit-age-refit-k1-k3"
REF = (
    ML_EXPERIMENTS_DATA_DIR
    / "00613-nra-roster-contrasts-with-intervals-five-seeds/score/c2_e3/event_free_rows.npz"
)
KEYS = a.KEYS
TAUS = {0.01: "q01", 0.05: "q05", 0.95: "q95", 0.99: "q99"}


def reference() -> pl.DataFrame:
    """Return the roster's scored rows with their event-free flag, band, group and month."""
    z = np.load(REF, allow_pickle=True)
    return pl.DataFrame(
        {
            k: z[k]
            for k in ("prim", "origin_idx", "idx", "band", "free", "group", "month", "crps_K2")
        }
    )


def main() -> None:
    """Score the network's 25 folds into cells."""
    ref = reference()
    clean = c.clean_stack()
    sm = k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    out = []
    for month in fold.MONTHS:
        ref_m = ref.filter(pl.col("month") == month)
        for age in fold.AGES_WEEKS:
            name = fold.fold_name(month=month, age_weeks=age)
            cutoff = c.month_start(month=month) - timedelta(weeks=age)
            parts = [
                pl.read_parquet(
                    ROOT / "arms" / NET / "work" / "fits" / TAG / f"{name}_d0_s{s}.parquet",
                    columns=[*KEYS, *a.QCOLS],
                )
                for s in SEEDS
            ]
            q = pl.concat(parts).group_by(KEYS).agg([pl.col(x).mean() for x in a.QCOLS])
            q = q.with_columns(
                prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
                idx=(
                    (pl.col("valid_time").dt.epoch("us") - prep.GRID_START_US) // prep.STEP_US
                ).cast(pl.Int64),
                origin_idx=(
                    (pl.col("power_fcst_init_time").dt.epoch("us") - prep.GRID_START_US)
                    // prep.STEP_US
                ).cast(pl.Int64),
            )
            j = q.join(ref_m, on=["prim", "origin_idx", "idx"], how="inner")
            assert j.height > 0.95 * ref_m.height, (name, j.height, ref_m.height)
            y = clean[j["prim"].to_numpy(), j["idx"].to_numpy()]
            ok = np.isfinite(y)
            j, y = j.filter(pl.Series(ok)), y[ok]
            qm = np.sort(j.select(a.QCOLS).to_numpy().astype(np.float32), axis=1)
            iq = a.IQ
            lead_h = (j["idx"].to_numpy() - j["origin_idx"].to_numpy()) / 2.0
            origin_us = j["origin_idx"].to_numpy() * prep.STEP_US + prep.GRID_START_US
            weeks = np.floor((origin_us - cutoff.timestamp() * 1e6) / (7 * 86_400 * 1e6)).astype(
                int
            )
            cols = {
                "crps": a.crps_from_quantiles(y=y, q=qm),
                "abs_err": np.abs(y - qm[:, iq[0.5]]),
                "cover90": ((y >= qm[:, iq[0.05]]) & (y <= qm[:, iq[0.95]])).astype(float),
                "cover98": ((y >= qm[:, iq[0.01]]) & (y <= qm[:, iq[0.99]])).astype(float),
                **{f"pin{int(t * 100):02d}": a.pinball(y=y, q=qm[:, iq[t]], tau=t) for t in TAUS},
                "ref_crps_k2": j["crps_K2"].to_numpy(),
            }
            df = pl.DataFrame(
                {"prim": j["prim"], "free": j["free"], "band": j["band"], "group": j["group"],
                 "week": weeks, "lead": lead_h, **cols}
            )  # fmt: skip
            cell = (
                df.group_by(["prim", "group", "free", "band", "week"])
                .agg(pl.len().alias("n"), *[pl.col(k).sum() for k in (*cols,)])
                .with_columns(net=pl.lit(NET), month=pl.lit(month), age=pl.lit(age))
            )
            out.append(cell)
            print(name, j.height, flush=True)
    res = pl.concat(out)
    (ROOT / "score").mkdir(exist_ok=True)
    res.write_parquet(ROOT / "score" / f"cells_{NET}.parquet")


main()

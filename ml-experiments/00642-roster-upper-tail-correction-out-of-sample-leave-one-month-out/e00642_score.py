# ruff: noqa: E501
"""Experiment 642: leave-one-month-out scoring of five upper-tail corrections.

Reads experiment 630's joined frames (`e00630_prep.py`), fits each correction on the other four
months, and writes one per-row table of indicators and scores per network, plus the quantiles of
the six signed-MW Primaries at leads 12 to 36 hours for the curtailment cost.
Run: PYTHONPATH=ml-experiments/00630-... uv run --no-sync python e00642_score.py
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00642_arms as arms
import numpy as np
import polars as pl
from e00630_common import BAND_LABELS, MONTHS, NETWORKS, Q, load
from studies.eval_protocol.metrics import tail_weighted_score
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00642-roster-upper-tail-correction-out-of-sample-leave-one-month-out"
)
ARMS: Final[tuple[str, ...]] = ("base", "S", "B", "C", "Q", "M")
SIGNED_MW: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
UPPER_TAU: Final[np.ndarray] = arms.TAU[arms.UPPER]
POOL_DAYS_C: Final[int] = 56
POOL_DAYS_M: Final[int] = 28
EXCLUDE_DAYS: Final[int] = 10
EXPECTED_ROWS: Final[dict[str, int]] = {"all": 1_140_221, "event_free": 905_502}


def fold_span(*, month: str) -> tuple[datetime, datetime]:
    """The held-out month's valid-time span extended by `EXCLUDE_DAYS` each side."""
    y, m = int(month[:4]), int(month[5:])
    start = datetime(y, m, 1, tzinfo=UTC)
    end = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=UTC)
    return start - timedelta(days=EXCLUDE_DAYS), end + timedelta(days=EXCLUDE_DAYS)


def crps(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Twice the mean pinball loss over the 11 levels, per row."""
    err = y[:, None] - q
    return 2 * np.maximum(arms.TAU * err, (arms.TAU - 1) * err).mean(axis=1)


def rolling_tables(*, df: pl.DataFrame) -> pl.DataFrame:
    """Causal conformal adjustments (5 upper levels) and median shift per (Primary, band, origin day).

    The pool for origin day d is the event-free rows of the same Primary and lead band whose origin
    day lies in the previous 56 days (28 for the shift) and whose valid time is before 00:00 UTC of d.
    A cell with fewer than `MIN_POOL_ROWS` pool rows gets null adjustments (no correction).
    """
    rows = []
    ef = df.filter(pl.col("event_free")).with_columns(
        od=pl.col("origin_date").dt.epoch("d"), vt=pl.col("valid_time").dt.epoch("s") / 86400.0
    )
    days = df.select("time_series_id", "band", "origin_date").unique()
    for (sid, band), g in ef.group_by("time_series_id", "band"):
        g_sorted = g.sort("od")
        od = g_sorted["od"].to_numpy()
        vt = g_sorted["vt"].to_numpy()
        y = g_sorted["observed_power"].to_numpy().astype(np.float64)
        q = g_sorted.select(Q).to_numpy().astype(np.float64)
        res_up = y[:, None] - q[:, arms.UPPER]
        res_med = y - q[:, arms.MEDIAN]
        for d in days.filter((pl.col("time_series_id") == sid) & (pl.col("band") == band))[
            "origin_date"
        ]:
            d_int = (d - datetime(1970, 1, 1, tzinfo=UTC).date()).days
            rec = {"time_series_id": sid, "band": band, "origin_date": d}
            for tag, win in (("c", POOL_DAYS_C), ("m", POOL_DAYS_M)):
                lo, hi = np.searchsorted(od, [d_int - win, d_int])
                mask = vt[lo:hi] < d_int
                n = int(mask.sum())
                rec[f"n_{tag}"] = n
                if tag == "c":
                    if n >= arms.MIN_POOL_ROWS:
                        r = res_up[lo:hi][mask]
                        for k, lev in enumerate(UPPER_TAU):
                            rec[f"adj{k}"] = float(np.quantile(r[:, k], lev))
                    else:
                        for k in range(5):
                            rec[f"adj{k}"] = None
                else:
                    rec["shift"] = (
                        float(np.median(res_med[lo:hi][mask])) if n >= arms.MIN_POOL_ROWS else None
                    )
            rows.append(rec)
    schema = {f"adj{k}": pl.Float64 for k in range(5)} | {"shift": pl.Float64}
    return pl.DataFrame(rows, schema_overrides=schema)


def attach(*, df: pl.DataFrame, tables: pl.DataFrame) -> pl.DataFrame:
    """Join the rolling tables to the rows, keeping the row order."""
    out = df.join(
        tables, on=["time_series_id", "band", "origin_date"], how="left", maintain_order="left"
    )
    assert out["valid_time"].equals(df["valid_time"]), "join changed the row order"
    return out.with_columns(block=(pl.col("hour") // 4).cast(pl.Int8))


def shifted(*, d: pl.DataFrame) -> np.ndarray:
    """Quantiles after adding the rolling median shift (zero where the window is too small)."""
    return d.select(Q).to_numpy().astype(np.float64) + d["shift"].fill_null(0.0).to_numpy()[:, None]


def fit_and_apply(*, train: pl.DataFrame, test: pl.DataFrame) -> dict[str, np.ndarray]:
    """Fit S, B, Q, M on `train` (event-free rows) and return every arm's test quantiles."""
    qt = test.select(Q).to_numpy().astype(np.float64)
    out = {"base": qt}
    s_fac, pit_maps, m_fac = {}, {}, {}
    for sid, g in train.group_by("time_series_id"):
        y = g["observed_power"].to_numpy().astype(np.float64)
        s_fac[sid[0]] = arms.fit_scale(y=y, q=g.select(Q).to_numpy().astype(np.float64))
        gm = g.filter(pl.col("shift").is_not_null())
        m_fac[sid[0]] = arms.fit_scale(
            y=gm["observed_power"].to_numpy().astype(np.float64), q=shifted(d=gm)
        )
    for key, g in train.group_by("time_series_id", "band"):
        pit_maps[key] = arms.fit_pit_map(
            y=g["observed_power"].to_numpy().astype(np.float64),
            q=g.select(Q).to_numpy().astype(np.float64),
        )
    sid_t = test["time_series_id"].to_numpy()
    f_s = np.array([s_fac.get(s, 1.0) for s in sid_t])
    out["S"] = arms.scale_upper(q=qt, factor=f_s)
    # B: pooled factor per (band, 4-hour block) fitted on the S-scaled training rows.
    qs_train = arms.scale_upper(
        q=train.select(Q).to_numpy().astype(np.float64),
        factor=np.array([s_fac.get(s, 1.0) for s in train["time_series_id"].to_numpy()]),
    )
    y_train = train["observed_power"].to_numpy().astype(np.float64)
    b_fac = {}
    keys_train = list(zip(train["band"].to_list(), train["block"].to_list(), strict=True))
    idx: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys_train):
        idx.setdefault(k, []).append(i)
    for k, ii in idx.items():
        rows_k = np.array(ii)
        b_fac[k] = arms.fit_scale(y=y_train[rows_k], q=qs_train[rows_k])
    f_b = np.array(
        [
            b_fac.get(k, 1.0)
            for k in zip(test["band"].to_list(), test["block"].to_list(), strict=True)
        ]
    )
    out["B"] = arms.scale_upper(q=out["S"], factor=f_b)
    # C: rolling conformal (causal; independent of the fold).
    adj = test.select([f"adj{k}" for k in range(5)]).to_numpy().astype(np.float64)
    adj = np.where(np.isnan(adj), 0.0, adj)
    out["C"] = arms.add_upper_adjustment(q=qt, adjustment=adj)
    # Q: quantile mapping per (Primary, band).
    q_out = qt.copy()
    keys_test = list(zip(test["time_series_id"].to_list(), test["band"].to_list(), strict=True))
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys_test):
        groups.setdefault(k, []).append(i)
    for k, ii in groups.items():
        rows_k = np.array(ii)
        q_out[rows_k] = arms.apply_pit_map(q=qt[rows_k], p=pit_maps.get(k, arms.TAU))
    out["Q"] = q_out
    # M: rolling 28-day median shift, then the per-Primary scale about the shifted median.
    out["M"] = arms.scale_upper(
        q=shifted(d=test), factor=np.array([m_fac.get(s, 1.0) for s in sid_t])
    )
    out["_s_fac"] = f_s
    out["_m_fac"] = np.array([m_fac.get(s, 1.0) for s in sid_t])
    return out


def main() -> None:
    """Score every arm of every network on each held-out month."""
    OUT.mkdir(parents=True, exist_ok=True)
    sub = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07" / "substations.csv"
    )
    for network in NETWORKS:
        df = load(network=network)
        assert df.height == EXPECTED_ROWS["all"], df.height
        assert df.filter(pl.col("event_free")).height == EXPECTED_ROWS["event_free"]
        df = df.filter(pl.col("month").is_in(MONTHS)).with_columns(
            block=(pl.col("hour") // 4).cast(pl.Int8)
        )
        df = attach(df=df, tables=rolling_tables(df=df))
        pieces, cost_pieces = [], []
        for fold in MONTHS:
            lo, hi = fold_span(month=fold)
            test = df.filter(pl.col("month") == fold)
            train = df.filter(
                (pl.col("month") != fold)
                & pl.col("event_free")
                & ~((pl.col("valid_time") >= lo) & (pl.col("valid_time") < hi))
            )
            res = fit_and_apply(train=train, test=test)
            y = test["observed_power"].to_numpy().astype(np.float64)
            thr = (
                train.group_by("time_series_id")
                .agg(pl.col("observed_power").quantile(0.95).alias("thr"))
                .join(
                    test.select("time_series_id"),
                    on="time_series_id",
                    how="right",
                    maintain_order="right",
                )["thr"]
                .to_numpy()
            )
            cols = {}
            for arm in ARMS:
                q = res[arm]
                cols[f"a90_{arm}"] = y > q[:, arms.Q90]
                cols[f"a95_{arm}"] = y > q[:, arms.Q95]
                cols[f"b10_{arm}"] = y < q[:, 2]
                cols[f"b05_{arm}"] = y < q[:, 1]
                cols[f"crps_{arm}"] = crps(y=y, q=q)
                cols[f"tw_{arm}"] = tail_weighted_score(actual=y, quantiles=q, threshold=thr)
                cols[f"gap_{arm}"] = y - q[:, arms.MEDIAN]
            cols["corrected_C"] = test["adj0"].is_not_null().to_numpy()
            cols["corrected_M"] = test["shift"].is_not_null().to_numpy()
            cols["scale_S"] = res["_s_fac"]
            cols["scale_M"] = res["_m_fac"]
            piece = test.select(
                "time_series_id",
                "primary_name",
                "group",
                "month",
                "origin_date",
                "event_free",
                "band",
                "block",
                "cold_night",
                "lead_hours",
            ).with_columns([pl.Series(k, v) for k, v in cols.items()])
            pieces.append(piece)
            keep = (
                test["primary_name"].is_in(SIGNED_MW)
                & (test["lead_hours"] >= 12)
                & (test["lead_hours"] < 36)
            ).to_numpy()
            cost_pieces.extend(
                (
                    test.filter(pl.Series(keep))
                    .select("time_series_id", "power_fcst_init_time", "valid_time", "primary_name")
                    .with_columns(
                        pl.Series("actual", y[keep]),
                        arm=pl.lit(arm),
                        **{
                            qn: pl.Series(qn, res[arm][keep, i].astype(np.float32))
                            for i, qn in enumerate(Q)
                        },
                    )
                )
                for arm in ARMS
            )
            print(network, fold, test.height, train.height, flush=True)
        pl.concat(pieces).write_parquet(OUT / f"scored_{network}.parquet", compression="zstd")
        pl.concat(cost_pieces).write_parquet(
            OUT / f"cost_quantiles_{network}.parquet", compression="zstd"
        )
    _ = (sub, BAND_LABELS)


if __name__ == "__main__":
    main()

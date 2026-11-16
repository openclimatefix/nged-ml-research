# ruff: noqa: D103, E501, PLR0912
"""Experiment 700: extra scores of the origin months 2025-11 to 2026-06, on the rows all arms share.

The scores are those that the protocol slices lack.

    PYTHONPATH=<folders of experiments 377, 378, 668, 677> E00677_ROOT=<this experiment's private folder> \
        uv run --no-sync python ml-experiments/00700-*/e00700_extra.py

Reads the per-row keys of `scores/protocol_year/rows.parquet` (the rows every arm shares) and the
same forecast files as `e00700_protocol.py`, and prints, for the 13 Primaries of the two protocol
groups (six signed-MW, seven MVA-only), in points of capacity: the MAE of the median by forecast day
(days 0 to 13, so days 10 to 14 are covered), the pinball loss by quantile level, the coverage of
q05 to q95 and q01 to q99, and the verified D5 and D16 load transfer (inside, the 7 days after, and
outside it). Writes `scores/protocol_year/extra_*.csv`.
"""

from datetime import UTC, datetime, timedelta

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00677_score as sc  # ty: ignore[unresolved-import]
import e00700_protocol as proto
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics

VERIFIED = (datetime(2025, 12, 29, tzinfo=UTC), datetime(2026, 1, 14, tzinfo=UTC))
TAUS = lib.TAUS


def main() -> None:
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=cp.CACHE_DIR)
    actuals, _ = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=metrics.HOLDOUT_START,
    )
    keep = pl.read_parquet(sc.ROOT / "scores" / "protocol_year" / "rows.parquet").filter(
        pl.col("label").is_in(cp.GROUPS["signed-MW six"] + cp.GROUPS["MVA-only seven"])
    )
    keys = keep.filter(pl.col("arm") == "GAM").select(*cp.KEYS, "label")
    cap = series.select("time_series_id", "capacity_mw").with_columns(
        pl.col("time_series_id").cast(pl.Int8)
    )
    ids = series["time_series_id"].cast(pl.Int8).to_list()
    frames = {}
    for label, (name, seeds) in proto.FILES.items():
        frames[label] = sc.seed_mean(name=name, seeds=seeds).filter(
            pl.col("time_series_id").is_in(ids)
        )
    label_of = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    for label, arm in proto.REFERENCES.items():
        stored = (
            pl.scan_parquet(
                next(proto.ML_EXPERIMENTS_DATA_DIR.glob("00087-*"))
                / "work"
                / "arms"
                / f"{arm}.parquet"
            )
            .filter(
                (pl.col("group") == "Primary")
                & (pl.col("power_fcst_init_time") >= proto.START)
                & (pl.col("power_fcst_init_time") < proto.END)
            )
            .collect()
            .with_columns(
                time_series_id=pl.col("site").replace_strict(label_of, return_dtype=pl.Int32)
            )
        )
        f = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
        frames[label] = cp.relabel(frame=f).with_columns(pl.col("time_series_id").cast(pl.Int8))
    truth = actuals.with_columns(pl.col("time_series_id").cast(pl.Int8)).select(
        "time_series_id", "valid_time", pl.col("actual").alias("y_mw")
    )
    by_day, pin, cov, event = [], [], [], []
    for arm, f in frames.items():
        j = (
            keys.join(
                f.with_columns(pl.col("time_series_id").cast(pl.Int8)), on=cp.KEYS, how="inner"
            )
            .join(truth, on=["time_series_id", "valid_time"])
            .join(cap, on="time_series_id")
        )
        scale = 100.0 / j["capacity_mw"].to_numpy()
        y = j["y_mw"].to_numpy() * scale
        q = j.select(cp.Q).to_numpy().astype(np.float64) * scale[:, None]
        lead_h = (
            j["valid_time"] - j["power_fcst_init_time"]
        ).dt.total_seconds().to_numpy() / 3600.0
        day = np.minimum(np.floor(lead_h / 24.0), 13).astype(int)
        mae = np.abs(y - q[:, 5])
        pb = lib.pinball(y=y, q=q)
        for d in range(14):
            m = day == d
            by_day.append(
                {
                    "arm": arm,
                    "lead_day": d,
                    "rows": int(m.sum()),
                    "mae_median": float(mae[m].mean()),
                    "crps": float((2 * pb[m].mean(axis=1)).mean()),
                }
            )
        for k, tau in enumerate(TAUS):
            pin.append(
                {
                    "arm": arm,
                    "level": float(tau),
                    "pinball": float(pb[:, k].mean()),
                    "share_below": float((y <= q[:, k]).mean()),
                }
            )
        cov.append(
            {
                "arm": arm,
                "rows": len(y),
                "q05_q95": float(((y >= q[:, 1]) & (y <= q[:, 9])).mean()),
                "q01_q99": float(((y >= q[:, 0]) & (y <= q[:, 10])).mean()),
            }
        )
        vt = j["valid_time"].to_numpy()
        inside = (vt >= np.datetime64(VERIFIED[0].replace(tzinfo=None))) & (
            vt < np.datetime64(VERIFIED[1].replace(tzinfo=None))
        )
        after = (vt >= np.datetime64(VERIFIED[1].replace(tzinfo=None))) & (
            vt < np.datetime64((VERIFIED[1] + timedelta(days=7)).replace(tzinfo=None))
        )
        outside = vt >= np.datetime64((VERIFIED[1] + timedelta(days=7)).replace(tzinfo=None))
        lab = j["label"].to_numpy()
        for site in ("D5", "D16"):
            for name, m in (
                ("inside", inside),
                ("7 days after", after),
                ("outside (7+ days after)", outside),
                ("all rows", np.ones(len(y), bool)),
            ):
                mm = m & (lab == site)
                if mm.any():
                    event.append(
                        {
                            "arm": arm,
                            "site": site,
                            "scenario": name,
                            "rows": int(mm.sum()),
                            "mae_median": float(mae[mm].mean()),
                            "crps": float((2 * pb[mm].mean(axis=1)).mean()),
                        }
                    )
    out = sc.ROOT / "scores" / "protocol_year"
    for name, rows in (
        ("by_day", by_day),
        ("pinball", pin),
        ("coverage", cov),
        ("verified_transfer", event),
    ):
        frame = pl.DataFrame(rows)
        frame.write_csv(out / f"extra_{name}.csv")
        print("==", name)
        if name == "by_day":
            print(frame.pivot(on="arm", index="lead_day", values="mae_median").sort("lead_day"))
        elif name == "pinball":
            print(frame.pivot(on="arm", index="level", values="pinball").sort("level"))
        elif name == "verified_transfer":
            print(
                frame.pivot(on="arm", index=["site", "scenario"], values="mae_median").sort(
                    "site", "scenario"
                )
            )
        else:
            print(frame)


if __name__ == "__main__":
    main()

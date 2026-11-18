# ruff: noqa: D103, E501
"""Experiment 701: extra scores of the origin months 2026-07 to 2026-10, on the rows all arms share.

The scores are those that the protocol slices lack.

    PYTHONPATH=<experiment folders> \
        uv run --no-sync python ml-experiments/00701-*/e00701_extra.py [--smoke]

Reads the per-row keys of `scores/protocol_final/rows.parquet` (the rows every arm shares) and the
same forecast files as `e00701_protocol.py`, and prints, for the 13 Primaries of the two protocol
groups (six signed-MW, seven MVA-only), in points of capacity: the MAE of the median by forecast day
(days 0 to 13, so days 10 to 14 are covered), the pinball loss by quantile level, and the coverage of
q05 to q95 and q01 to q99. Writes `scores/protocol_final/extra_{by_day,pinball,coverage}.csv`.
`--smoke` reads and writes `scores/smoke_protocol_final/` instead.
"""

import argparse

import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00701_protocol as proto
import numpy as np
import polars as pl

TAUS = lib.TAUS


def main(*, setting: proto.Setting) -> None:
    series, actuals, *_ = proto.load_inputs()
    keep = pl.read_parquet(setting.out / "rows.parquet").filter(
        pl.col("label").is_in(cp.GROUPS["signed-MW six"] + cp.GROUPS["MVA-only seven"])
    )
    keys = keep.filter(pl.col("arm") == "GAM").select(*cp.KEYS, "label")
    cap = series.select("time_series_id", "capacity_mw").with_columns(
        pl.col("time_series_id").cast(pl.Int8)
    )
    frames = proto.load_forecasts(setting=setting, series=series)
    truth = actuals.with_columns(pl.col("time_series_id").cast(pl.Int8)).select(
        "time_series_id", "valid_time", pl.col("actual").alias("y_mw")
    )
    by_day, pin, cov = [], [], []
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
    for name, rows in (("by_day", by_day), ("pinball", pin), ("coverage", cov)):
        frame = pl.DataFrame(rows)
        frame.write_csv(setting.out / f"extra_{name}.csv")
        print("==", name)
        if name == "by_day":
            print(frame.pivot(on="arm", index="lead_day", values="mae_median").sort("lead_day"))
        elif name == "pinball":
            print(frame.pivot(on="arm", index="level", values="pinball").sort("level"))
        else:
            print(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="read experiment 700's June files")
    main(setting=proto.SMOKE if parser.parse_args().smoke else proto.FINAL)

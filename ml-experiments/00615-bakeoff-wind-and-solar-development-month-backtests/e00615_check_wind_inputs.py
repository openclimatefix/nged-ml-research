"""Gate: the rebuilt plant summaries match experiment 347's stored scored summaries on shared cases.

The 53 scored Tuesday runs of experiment 347 include Tuesdays after 2025-06-17, which this
experiment rebuilt from the same plant. For each farm, the script joins the two on the case key and
prints the largest absolute difference in each summary column and in the observed power.
"""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import e00303_score as score303  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00615_common as common
import e00615_wind_inputs as inputs
import numpy as np
import polars as pl


def main() -> None:
    """Print the largest differences per farm."""
    scored = score303.ordered_cases()
    stored = np.concatenate(
        [np.load(data347.S4_DIR / f"summary_y2_scored_{f}.npy") for f in data347.FARMS]
    )
    frame = scored.select(
        "site", "time_series_id", "power_fcst_init_time", "valid_time", "actual_norm"
    ).with_columns(**{n: stored[:, i] for i, n in enumerate(inputs.SUMMARY_NAMES)})
    for farm in data347.FARMS:
        new = pl.read_parquet(common.DATA_DIR / "wind_inputs" / f"cases_{farm}.parquet")
        joined = frame.filter(pl.col("site") == farm).join(
            new, on=["time_series_id", "power_fcst_init_time", "valid_time"], suffix="_new"
        )
        diffs = {
            n: float((joined[n] - joined[f"{n}_new"]).abs().max()) for n in inputs.SUMMARY_NAMES
        }
        diffs["actual_norm"] = float(
            (joined["actual_norm"] - joined["actual_norm_new"]).abs().max()
        )
        print(farm, "shared cases", joined.height, {k: round(v, 6) for k, v in diffs.items()})


if __name__ == "__main__":
    main()

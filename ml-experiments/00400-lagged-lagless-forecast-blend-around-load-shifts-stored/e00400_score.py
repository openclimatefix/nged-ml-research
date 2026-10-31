"""Experiment 400, score step: per-row losses of every arm, as in experiment 395 Part F.

Reads `arm_<name>.parquet` (keys and 11 quantiles in MW) from the blend step, scores each arm with
experiment 390's `trimmed_rows` on experiment 395's Part F slice table, and writes
`rows_<name>.parquet` with the flagged half-hours tagged (`flagged`), the median, q95, q99, and
actual in MW. Also writes the exact CRPS of every mixture arm (`exact_crps.parquet`).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00390-xgboost-d11-d14-gap-and-load-shift-event-lags:\
ml-experiments/00395-xgboost-level-state-winter-spread-anchor-stored-forecasts:\
ml-experiments/00400-lagged-lagless-forecast-blend-around-load-shifts-stored \
DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00400-*/e00400_score.py
```
"""

import gc
import sys
from pathlib import Path

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00390_build as b390  # ty: ignore[unresolved-import]
import e00400_blend as blend
import e00400_mixture as mix
import e00400_partf as pf  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol.metrics import QUANTILE_COLUMNS

KEYS = blend.KEYS
WORK = blend.WORK


def exact_crps(*, scored: pl.DataFrame) -> pl.DataFrame:
    """Return the exact CRPS (points) of every mixture arm on the scored rows."""
    weights = pl.read_parquet(WORK / "weights.parquet")
    q = {}
    for name in ("lags", "lagless"):
        frame = pl.read_parquet(WORK / f"arm_{name}.parquet")
        q[name] = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    cap = scored["capacity_mw"].to_numpy()
    actual = scored["actual_mw"].to_numpy() / cap
    out = {k: scored[k] for k in KEYS}
    for arm in ("fixed", "b1", "b2", "b3"):
        w = weights[f"w_{arm}"].to_numpy().astype(np.float64)
        crps = np.empty(scored.height)
        for start in range(0, scored.height, blend.CHUNK):
            sl = slice(start, start + blend.CHUNK)
            pair = mix.Pair(
                q1=q["lags"][sl] / cap[sl, None],
                q2=q["lagless"][sl] / cap[sl, None],
                actual=actual[sl],
            )
            crps[sl] = pair.crps(w=w[sl]) * 100.0
        out[f"exact_crps_{arm}"] = pl.Series(crps)
    return pl.DataFrame(out)


def main() -> None:
    """Score every arm, one at a time."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    table = pl.read_parquet(pf.WORK / "slice_table.parquet")
    flags = pf.flagged_keys(series=series)
    scored = pl.read_parquet(
        pf.WORK / "rows_bundle.parquet",
        columns=[*KEYS, "capacity_mw", "actual_mw"],
    ).sort(KEYS)
    names = ["lags", "lagless"] + [
        f"{kind}_{arm}" for kind in ("mix", "avg") for arm in ("fixed", "b1", "b2", "b3")
    ]
    for name in names:
        frame = pl.read_parquet(WORK / f"arm_{name}.parquet").with_columns(
            pl.col("time_series_id").cast(series.schema["time_series_id"])
        )
        use = table.join(frame.select(*KEYS), on=KEYS, how="semi")
        rows = b390.trimmed_rows(name=name, frame=frame, table=use, ctx=ctx, series=series)
        rows = rows.join(flags, on=["time_series_id", "valid_time"], how="left").with_columns(
            pl.col("flagged").fill_null(value=False)
        )
        extra = frame.join(ctx["actuals"], on=["time_series_id", "valid_time"]).select(
            *KEYS,
            q50_mw=pl.col("q50"),
            actual_mw=pl.col("actual"),
            q95_mw=pl.col("q95"),
            q99_mw=pl.col("q99"),
            q05_mw=pl.col("q05"),
        )
        rows.join(extra, on=KEYS, how="left", validate="1:1").write_parquet(
            WORK / f"rows_{name}.parquet"
        )
        print("scored", name, rows.height, flush=True)
        del frame, rows, extra, use
        gc.collect()
    exact_crps(scored=scored).write_parquet(WORK / "exact_crps.parquet")
    print("exact CRPS written", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

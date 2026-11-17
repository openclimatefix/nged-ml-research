# ruff: noqa: E501, I001
"""Experiment 701, GAM side: experiment 378's solar summary S and blend, on the extended inputs.

    export E00377_SOURCE=control OMP_NUM_THREADS=1 DATA_PATH_INTERNAL=SANDBOX
    export PYTHONPATH=FOLDERS_OF_184_202_319_320_377_378_700_701
    uv run --no-sync python ml-experiments/00701-*/e00701_gam_arms.py low|long [--smoke 2026-06]

`low` is `e00378_arms_low.py` (S at 0.5 to 168 hours, column `x1_*`, file
`low_lead_<source>.parquet`) and `long` is `e00378_arms.py` (S, the blend with the manual heuristic,
and default 3 at 168 to 240 hours, columns `x1_*`, `r319_*`, `d3_*`, file
`long_lead_<source>.parquet`). **This wrapper imports `e00701_gam_extend377` first and then calls
experiment 378's own `main`, so the arms read the extended feature cache (their daylight flag and
the heuristic's lagged readings need grid rows after 2026-06-30), and nothing else of experiment
378's code changes.** Both replay every fold month from the sandbox's stage 2 and member files, so
the months to 2026-06 are recomputed, and `--check-old` (see `e00701_gam_run.sh`) compares them with
the files that experiment 700 left.

`--smoke MONTH` replays only that month for the sites of `E00377_SOURCE` (the fit windows still read
the earlier months), compares the result with the stored columns of the sandbox files, prints
the largest differences, and writes nothing. For `low` this repeats the body of
`e00378_arms_low.main` restricted to one month.
"""

import json
import sys
from datetime import timedelta

import e00701_gam_extend377  # noqa: F401  (dates and paths first)
import e00701_gam_extend as ext
import e00202_arms as a202  # ty: ignore[unresolved-import]
import e00377_common as c377  # ty: ignore[unresolved-import]
import e00378_arms as arms  # ty: ignore[unresolved-import]
import e00378_arms_low as low  # ty: ignore[unresolved-import]
import e00378_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours

QUANTILES = common.QUANTILES


def smoke_long(*, source: str, month_text: str) -> pl.DataFrame:
    """Replay one month of the long-lead arms and return its rows with the new `x1`, `r319`, and `d3` columns."""
    cache = c377.load_cache()
    frame, delta = arms.load_rows(cache=cache, source=source)
    in_fit = np.array([s in arms.FIT_GROUPS[arms.APPLY_GROUP[s]] for s in frame["site"].to_numpy()])
    scored = (frame["power_fcst_init_time"].dt.strftime("%Y-%m") == month_text).to_numpy()
    out, _ = arms.replay(frame=frame, delta=delta, in_fit=in_fit, scored=scored, label=source)
    keep = frame.select(
        "site",
        "run",
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        "kind",
        "heuristic_median",
        *QUANTILES,
    ).filter(pl.Series(scored))
    index = np.flatnonzero(scored)
    return keep.with_columns(
        [
            pl.Series(f"{name}_{q}", values[index, k].astype(np.float32))
            for name, values in out.items()
            for k, q in enumerate(QUANTILES)
        ]
    )


def smoke_low(*, source: str, month_text: str) -> pl.DataFrame:
    """Replay one month of the low-lead arm S and return its rows with the new `x1` columns."""
    cache = c377.load_cache()
    frame, delta = low.load_low(cache=cache, source=source)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    month = frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    usable = (~frame["switching_mask"] & frame["actual"].is_finite()).to_numpy()
    group = frame["group"].to_numpy()
    in_fit = np.array([s in arms.FIT_GROUPS[arms.APPLY_GROUP[s]] for s in frame["site"].to_numpy()])
    day = ((frame["kind"] == "daylight") & (frame["n_members"] > 0)).to_numpy()
    lead_band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    horizon = np.timedelta64(int(timedelta(hours=common.STORED_HORIZON_HOURS).total_seconds()), "s")
    out = frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
    for g in sorted(set(group)):
        rows_m = np.flatnonzero((month == month_text) & (group == g))
        first = origin[rows_m].min()
        fit_rows = np.flatnonzero(
            (origin + horizon <= first)
            & (origin >= first - np.timedelta64(common.WINDOW_DAYS, "D"))
            & usable
            & in_fit
            & (group == g)
        )
        counts = [
            int((day[fit_rows] & (lead_band[fit_rows] == b)).sum()) for b in range(N_LEAD_BANDS)
        ]
        if min(counts) < arms.MIN_ROWS:
            factors = [(1.0, 1.0)] * N_LEAD_BANDS
        else:
            sub = frame[fit_rows]
            stats = {
                k: v.astype(np.float32).astype(np.float64)
                for k, v in a202.summaries(delta=delta[fit_rows]).items()
            }
            factors = a202.fit_factors(
                frame=sub,
                make_points=lambda s, stats=stats: np.nan_to_num(
                    a202.gaussian_points(stats=stats, spread=s)
                ),
                fit_spread=True,
            )
        stats_m = {
            k: v.astype(np.float32).astype(np.float64)
            for k, v in a202.summaries(delta=delta[rows_m]).items()
        }
        q = a202.apply_arm(
            frame=frame[rows_m],
            make_points=lambda s, stats=stats_m: np.nan_to_num(
                a202.gaussian_points(stats=stats, spread=s)
            ),
            factors=factors,
        )
        out[rows_m] = np.sort(q, axis=1)
    rows = np.flatnonzero(month == month_text)
    return (
        frame.select("site", "run", "power_fcst_init_time", "valid_time", "lead_hours")
        .filter(pl.Series(month == month_text))
        .with_columns(
            [pl.Series(f"x1_{q}", out[rows, k].astype(np.float32)) for k, q in enumerate(QUANTILES)]
        )
    )


def compare(*, new: pl.DataFrame, stored_path, columns: list[str]) -> dict:  # noqa: ANN001
    """Return row counts and the largest absolute difference of `columns` against a stored file."""
    stored = pl.read_parquet(stored_path)
    joined = new.join(
        stored.select("site", "run", "valid_time", *columns),
        on=["site", "run", "valid_time"],
        suffix="_old",
    )
    diffs = {
        c: float(np.nanmax(np.abs(joined[c].to_numpy() - joined[f"{c}_old"].to_numpy())))
        for c in columns
    }
    return {
        "new_rows": new.height,
        "joined_rows": joined.height,
        "max_abs_diff": max(diffs.values()),
        "per_column": diffs,
    }


def main() -> None:
    """Run experiment 378's arm, or the one-month smoke replay."""
    arm = sys.argv[1]
    source = c377.SOURCE
    if "--smoke" in sys.argv:
        month_text = sys.argv[sys.argv.index("--smoke") + 1]
        if arm == "low":
            new = smoke_low(source=source, month_text=month_text)
            result = compare(
                new=new,
                stored_path=common.ARMS_DIR / f"low_lead_{source}.parquet",
                columns=[f"x1_{q}" for q in QUANTILES],
            )
        else:
            new = smoke_long(source=source, month_text=month_text)
            result = compare(
                new=new,
                stored_path=common.ARMS_DIR / f"long_lead_{source}.parquet",
                columns=[f"{n}_{q}" for n in ("x1", "r319", "d3") for q in QUANTILES],
            )
        print(
            json.dumps(
                {
                    "arm": arm,
                    "source": source,
                    "month": month_text,
                    **{k: v for k, v in result.items() if k != "per_column"},
                }
            )
        )
        return
    assert ext.SANDBOX in common.ARMS_DIR.parents
    (low if arm == "low" else arms).main()


if __name__ == "__main__":
    main()

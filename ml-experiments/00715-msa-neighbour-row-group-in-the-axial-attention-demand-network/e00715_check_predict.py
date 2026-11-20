# ruff: noqa: E501, PT018, C408
"""Experiment 715: check the prediction path on the real fold with an untrained network and no fit.

Usage: e00715_check_predict.py MONTH   (CPU; run through `run.sh`)

An untrained `F4Net` (zero-initialised, so it equals an untrained `M*` of the same seed) forecasts every
scored origin of the fold. The check asserts the output's keys are unique, every valid time is before
2026-07-01, the quantiles are sorted, a masked-history and a masked-neighbour run give the same keys,
`compare_forecasts` reports a difference of 0 for a frame against itself and a positive one against a
shifted copy (a negative control), and the origin-day bound raises a readable error for a day outside
the ensemble table. No state is loaded, nothing is trained, and nothing is written.
"""

import os
import sys

os.environ["E00667_PHASE"] = "predict"

import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00715_arms as arms
import e00715_batch as fb
import e00715_model as f4
import e00715_neighbours as nb
import e00715_predict as pr
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.experiment_store import HOLDOUT_START


def main() -> int:
    """Run the checks on the fold named on the command line."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    month = sys.argv[1]
    arms.install_own_cache()
    pipe = inp.load_pipeline(month=month, phase="predict")
    data = inp.build_fold_data(pipe=pipe, device="cpu")
    table = nb.load_table()
    source = arms.f4_source(data=data, table=table, mode="real", rewire_seed=0)
    torch.manual_seed(0)
    model = f4.F4Net(switches=mdl.ARM_SWITCHES["M"]).eval()
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    common = dict(
        model=model,
        neighbours=True,
        spec=arms.resolve(arm="F4"),
        data=data,
        source=source,
        seed=0,
        grid_start_us=prep422.GRID_START_US,
        time_series_ids=ids,
    )
    frame, counts = pr.forecast_frame(**common, mask_history=False, mask_neighbours=False)  # ty: ignore[invalid-argument-type]
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    assert frame.select(keys).is_duplicated().sum() == 0, "duplicated keys"
    assert (frame["valid_time"] < pl.lit(HOLDOUT_START).cast(pl.Datetime("us", "UTC"))).all(), (
        "a valid time is in the holdout"
    )
    q = frame.select(pr.QCOLS).to_numpy()
    assert (np.diff(q, axis=1) >= -1e-9).all(), "the quantiles are not sorted"
    assert np.isfinite(q).all(), "a quantile is not finite"
    assert counts["visible_neighbour_tokens"] > 0
    nohist, _ = pr.forecast_frame(**common, mask_history=True, mask_neighbours=False)  # ty: ignore[invalid-argument-type]
    nonbr, _ = pr.forecast_frame(**common, mask_history=False, mask_neighbours=True)  # ty: ignore[invalid-argument-type]
    assert frame.select(keys).equals(nohist.select(keys)) and frame.select(keys).equals(
        nonbr.select(keys)
    )
    moved = float(np.abs(nohist["q50"].to_numpy() - frame["q50"].to_numpy()).max())
    assert moved > 0, "masking the host history changed nothing"
    # a zero-initialised F4Net ignores its neighbours, so masking them changes nothing
    assert float(np.abs(nonbr["q50"].to_numpy() - frame["q50"].to_numpy()).max()) < 1e-6
    same = pr.compare_forecasts(new=frame, stored=frame)
    assert same["same_keys"] and same["max_abs_difference"] == 0.0 and same["pass"]
    shifted = frame.with_columns(*(pl.col(c) + 1e-3 for c in pr.QCOLS))
    diff = pr.compare_forecasts(new=frame, stored=shifted)
    assert diff["max_abs_difference"] > pr.KN13_TOLERANCE and not diff["pass"], diff
    fewer = pr.compare_forecasts(new=frame, stored=frame.head(frame.height - 1))
    assert not fewer["same_keys"]
    ens = data.source.ens
    assert ens is not None
    try:
        fb.check_origin_days(
            ens=ens, c_idx=np.array([(int(ens.first_day) - 5) * 48 + 24]), what="a negative control"
        )
    except IndexError as err:
        print("origin-day bound raises:", str(err)[:120])
    else:
        msg = "an origin day before the ensemble table did not raise"
        raise AssertionError(msg)
    print(
        f"predict check passed on {month}: {frame.height} rows, visible neighbour share {counts['visible_neighbour_tokens'] / max(counts['tokens_of_filled_slots'], 1):.3f}, masking the history moves the median by up to {moved:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

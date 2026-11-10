# ruff: noqa: ANN001, ANN201, D103, E501, RET504
"""Experiment 537: experiment 425 arm A with the 9 lag input columns removed (46 -> 37 inputs).

Wraps 425's fit code unchanged; only the input matrix loses the "lags" columns, in the fine-tuning rows,
the pretraining rows, and the test-month inputs. Cuts and rows are the same as A-pool's.
Run from the worktree root after `source ml-experiments/00524*/env.sh`.
"""

from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_fit as fit  # ty: ignore[unresolved-import]
import e00425_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

WORK = Path("/home/jack/dev/nged-substation-forecast/data/ml-experiments/00537-a-pool-minus-lags")
_, _, LAYOUT = base.feature_names(scheme="summary")
LAG_COLS = LAYOUT.columns_of(groups=["lags"])
assert len(LAG_COLS) == 9, LAG_COLS
KEEP = np.array([i for i in range(LAYOUT.n_inputs) if i not in set(LAG_COLS)])
assert len(KEEP) == 37

fit.OUT_DIR = WORK / "work"
ORIG_BUILD, ORIG_PRE, ORIG_RUN = prep.build_ens_fold, prep.fold_pretrain, base.run_inputs
FULL = {}


def build(*, refit, cut=True):
    fold = ORIG_BUILD(refit=refit, cut=cut)
    FULL["fold"] = fold
    fold = type(fold)(**{**fold.__dict__, "x": np.ascontiguousarray(fold.x[:, KEEP])})
    return fold


def pretrain_rows(*, fold, refit, cut=True):
    full = FULL[
        "fold"
    ]  # the 46-wide fold: the pretraining builder standardises full-width raw rows
    pre, dropped = ORIG_PRE(fold=full, refit=refit, cut=cut)
    pre.x = np.ascontiguousarray(pre.x[:, KEEP])
    return pre, dropped


CHECKED = {"rows": 0, "runs": 0, "example": None}


def run_inputs(*, origin, scheme, standardiser):
    out = ORIG_RUN(origin=origin, scheme=scheme, standardiser=standardiser)
    run = origin - common.PUBLICATION_DELAY
    frame = pl.read_parquet(common.demand_frame_path(run=run))
    summ = base.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
    assert summ.exists()
    # weather run of every scored row: the NWP init time in the frame, and the file's run, vs the origin
    assert (frame["nwp_init_time"] <= frame["power_fcst_init_time"]).all()
    assert (frame["power_fcst_init_time"] == origin).all(), (
        "frame origin differs from scored origin"
    )
    assert run <= origin
    new = []
    for row, keys, x in out:
        assert (keys["power_fcst_init_time"] == origin).all()
        n = keys.height
        CHECKED["rows"] += n
        if CHECKED["example"] is None:
            CHECKED["example"] = (  # ty: ignore[invalid-assignment]
                f"series {row['site']}, origin {origin}, weather run {run} "
                f"(nwp_init_time {frame['nwp_init_time'][0]}), first target {keys['valid_time'][0]}, "
                f"lead_hours {keys['lead_hours'][0]}"
            )
            print("EXAMPLE ROW:", CHECKED["example"], flush=True)
        new.append((row, keys, np.ascontiguousarray(x[:, KEEP])))
    CHECKED["runs"] += 1  # ty: ignore[unsupported-operator]
    return new


prep.build_ens_fold = build
prep.fold_pretrain = pretrain_rows
base.run_inputs = run_inputs

if __name__ == "__main__":
    import json

    fit.main()
    (WORK / "weather_run_check.json").write_text(json.dumps(CHECKED, indent=1, default=str))
    print("WEATHER RUN CHECK PASSED:", CHECKED["rows"], "scored rows over", CHECKED["runs"], "runs")

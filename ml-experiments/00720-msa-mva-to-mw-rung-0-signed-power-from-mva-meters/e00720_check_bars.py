# ruff: noqa: E501
"""Experiment 720: known-answer checks of the bar functions and of the grid-length rule of the loader."""

from pathlib import Path

import e00720_bars as bars
import e00720_real as real
import numpy as np
import polars as pl


def main() -> int:
    """Run the checks."""
    assert bars.v1_bar(auc_real=0.9, auc_shifted=0.6, false_reverse_forced_open=0.005)["passed"]
    assert not bars.v1_bar(auc_real=0.9, auc_shifted=0.85, false_reverse_forced_open=0.005)[
        "passed"
    ]
    assert not bars.v1_bar(auc_real=0.9, auc_shifted=0.6, false_reverse_forced_open=0.02)["passed"]
    assert bars.paired_bar(lower=0.01, point=0.06, min_point=0.05)["passed"]
    assert not bars.paired_bar(lower=-0.01, point=0.06, min_point=0.05)["passed"]
    assert not bars.paired_bar(lower=0.01, point=0.04, min_point=0.05)["passed"]
    ok_p = {"flagged_share": 0.01, "active_days_per_month": 0.0, "forced_k3_flagged_share": 0.02}
    assert bars.v3_all_periods(periods={"a": ok_p, "b": ok_p})["passed"]
    assert not bars.v3_all_periods(
        periods={"a": ok_p, "b": {**ok_p, "forced_k3_flagged_share": 0.2}}
    )["passed"]
    assert not bars.v3_all_periods(periods={})["passed"]
    assert bars.k1_bar(f1=0.78)["passed"]
    assert not bars.k1_bar(f1=0.7)["passed"]
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = rng.uniform(0, 1, 20000) < p
    assert bars.sign_probability_bar(prob=p, outcome=y)["passed"]
    assert not bars.sign_probability_bar(prob=np.clip(p * 0.5 + 0.5, 0, 1), outcome=y)["passed"]
    assert real.check_grid_lengths(n_grid=121248, dense_raw_len=126624, cams_len=121248) == 121248
    for bad in ((121248, 100000, 121248), (121248, 126624, 126624)):
        try:
            real.check_grid_lengths(n_grid=bad[0], dense_raw_len=bad[1], cams_len=bad[2])
        except ValueError:
            continue
        msg = "a bad grid length was accepted"
        raise AssertionError(msg)
    base = np.datetime64("2019-09-01T00:00") + np.arange(400) * np.timedelta64(30, "m")
    us = base.astype("datetime64[us]").astype(np.int64)
    weekday = (
        pl.Series(us)
        .cast(pl.Datetime("us", "UTC"))
        .dt.convert_time_zone("Europe/London")
        .dt.weekday()
        .to_numpy()
        - 1
    )
    onehot = np.eye(7)[weekday]
    assert real.check_dense_raw_alignment(day_of_week_columns=onehot, stamp_us=us) == 400
    try:
        real.check_dense_raw_alignment(day_of_week_columns=onehot[48:], stamp_us=us[:352])
    except ValueError:
        pass
    else:
        msg = "a shifted dense_raw start was accepted"
        raise AssertionError(msg)
    try:
        real.check_f7_commit(module_file=Path(__file__))
    except RuntimeError:
        pass
    else:
        msg = "a tree at another commit was accepted as experiment 716"
        raise AssertionError(msg)
    f7 = Path(__file__).resolve().parents[3] / "msa-f7-pv-part-1" / "ml-experiments"
    for mod in f7.glob("00716-*/e00716_pv.py"):
        assert real.check_f7_commit(module_file=mod) == real.F7_COMMIT
    print("PASS bars, grid-length rule, and the experiment 716 commit pin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

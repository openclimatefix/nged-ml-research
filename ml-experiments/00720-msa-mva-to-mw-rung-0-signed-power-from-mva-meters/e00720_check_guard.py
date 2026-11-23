# ruff: noqa: E501
"""Experiment 720: checks of the anonymisation guard."""

import sys
from datetime import UTC, datetime

import e00720_guard as guard
import numpy as np
import polars as pl


def raises(*, fn, **kw) -> bool:  # noqa: ANN001, ANN003
    """Return whether `fn(**kw)` raises `ValueError`."""
    try:
        fn(**kw)
    except ValueError:
        return True
    return False


def main() -> int:
    """Run the guard checks."""
    dated = pl.DataFrame({"valid_time": [datetime(2025, 1, 1, tzinfo=UTC)], "n_units": [2]})
    assert raises(fn=guard.refuse_dated_generator_values, frame=dated)
    assert raises(
        fn=guard.refuse_dated_generator_values,
        frame=pl.DataFrame({"month": ["2025-01"], "inferred_generator_output": [1.0]}),
    )
    assert not raises(
        fn=guard.refuse_dated_generator_values,
        frame=pl.DataFrame({"primary": ["D5"], "auc": [0.9]}),
    )
    assert not raises(
        fn=guard.refuse_dated_generator_values,
        frame=pl.DataFrame({"valid_time": [datetime(2025, 1, 1, tzinfo=UTC)], "reading": [1.0]}),
    )
    assert raises(
        fn=guard.public_aggregate, frame=pl.DataFrame({"primary": ["D8"], "hours_run": [3.0]})
    )
    assert not raises(
        fn=guard.public_aggregate,
        frame=pl.DataFrame({"primary": ["D12"], "auc": [0.9], "n_rows": [10]}),
    )
    assert guard.hours_band(hours=0.0) == "none"
    assert guard.hours_band(hours=50.0) == "1 to 50"
    assert guard.hours_band(hours=120.0) == "51 to 150"
    assert guard.hours_band(hours=301.0) == "over 300"
    pooled = guard.pool_bands(monthly_hours=np.array([0.0, 10.0, 60.0, 400.0, 400.0]))
    assert pooled["n_months"].to_list() == [1, 1, 1, 0, 2]
    for col in ("hours_run", "running_share", "p_any", "e_n", "flag", "signed_mean", "p_negative"):
        assert raises(
            fn=guard.refuse_dated_generator_values,
            frame=pl.DataFrame({"valid_time": [1], col: [0.5]}),
        ), col
    assert raises(
        fn=guard.public_aggregate,
        frame=pl.DataFrame({"primary": ["D8"], "era": ["a date range"], "f1": [0.5]}),
    )
    guard.public_aggregate(frame=pl.DataFrame({"primary": ["D8"], "era": ["era_1"], "f1": [0.5]}))
    p = guard.private_path(name="selftest/guard.txt")
    assert "private" in p.parts
    print(
        "PASS guard: dated generator frames refused, allow-list enforced, bands pooled with no month labels"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

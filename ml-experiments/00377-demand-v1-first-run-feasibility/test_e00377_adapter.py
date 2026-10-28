"""Tests of the unit argument of `e00377_adapter.forecast_frame`."""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from studies.eval_protocol.metrics import QUANTILE_COLUMNS
from studies.eval_protocol.slices import KEY_COLUMNS

_SPEC = importlib.util.spec_from_file_location(
    "e00377_adapter", Path(__file__).with_name("e00377_adapter.py")
)
assert _SPEC is not None
assert _SPEC.loader is not None
adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapter)

CAPACITY_MW = 40.0
SERIES = pl.DataFrame(
    {"time_series_id": [1], "label": ["D5"], "capacity_mw": [CAPACITY_MW]},
    schema={"time_series_id": pl.Int32, "label": pl.String, "capacity_mw": pl.Float64},
)


def stored_frame(*, quantile: float) -> pl.DataFrame:
    """Return one stored row whose 11 quantiles all equal `quantile`."""
    return pl.DataFrame(
        {
            "time_series_id": [1],
            "power_fcst_init_time": [datetime(2025, 7, 1, 9, tzinfo=UTC)],
            "valid_time": [datetime(2025, 7, 1, 12, tzinfo=UTC)],
            **dict.fromkeys(QUANTILE_COLUMNS, quantile),
        }
    )


@pytest.mark.parametrize(
    ("unit", "stored", "expected_mw"),
    [
        ("mw", 3.0, 3.0),
        ("points", 5.0, 5.0 * 0.01 * CAPACITY_MW),
        ("fraction", 0.25, 0.25 * CAPACITY_MW),
    ],
)
def test_forecast_frame_converts_each_unit_to_mw(
    unit: str, stored: float, expected_mw: float
) -> None:
    """Each stored unit becomes the same quantile in MW."""
    frame = adapter.forecast_frame(stored=stored_frame(quantile=stored), series=SERIES, unit=unit)  # ty: ignore[invalid-argument-type]
    assert frame.columns == [*KEY_COLUMNS, *QUANTILE_COLUMNS]
    assert frame["q50"].to_list() == pytest.approx([expected_mw])


def test_forecast_frame_rejects_a_repeated_key() -> None:
    """A key held twice is an error, not a silent pick."""
    stored = pl.concat([stored_frame(quantile=1.0), stored_frame(quantile=2.0)])
    with pytest.raises(ValueError, match="twice"):
        adapter.forecast_frame(stored=stored, series=SERIES, unit="mw")

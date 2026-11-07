# ruff: noqa: D101, D102, D107
"""Experiment 614: two dummy frozen candidates that prove the final-test runner works.

Neither is a forecast anyone would use. `PersistencePredictor` (a stand-in for a lag network)
forecasts
the day lag with a fixed band; `WeatherOnlyPredictor` (a stand-in for a lag-free network) forecasts
from
each Primary's mean and standard deviation before May 2026 only (it ignores weather). Each loads one
JSON artefact whose numbers are the frozen weights.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

Z = np.array([-2.326, -1.645, -1.282, -0.674, -0.253, 0.0, 0.253, 0.674, 1.282, 1.645, 2.326])
"""Standard normal quantiles at the 11 levels."""


class PersistencePredictor:
    def __init__(self, artefacts: dict[str, Path]) -> None:
        self.params = json.loads(artefacts["persistence.json"].read_text())

    def predict(self, inputs: pl.DataFrame) -> np.ndarray:
        assert "observed_power" not in inputs.columns, (
            "a predictor never sees the observed power at the valid time"
        )
        lag = inputs["power_lag_day"].to_numpy().astype(np.float64)
        median = np.where(np.isfinite(lag), lag, self.params["fallback"])
        return median[:, None] + self.params["scale"] * np.abs(median)[:, None] * Z[None, :]


class WeatherOnlyPredictor:
    def __init__(self, artefacts: dict[str, Path]) -> None:
        self.params = json.loads(artefacts["weather_only.json"].read_text())

    def predict(self, inputs: pl.DataFrame) -> np.ndarray:
        ids = inputs["time_series_id"].to_numpy()
        mean = np.array([self.params["mean"][str(i)] for i in ids])
        std = np.array([self.params["std"][str(i)] for i in ids])
        return mean[:, None] + std[:, None] * Z[None, :]

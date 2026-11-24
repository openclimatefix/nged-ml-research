"""Fixtures of the tests of experiment 722."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(722)

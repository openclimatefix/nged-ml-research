"""Tests of the shared term, its loading, and its on/off switch (synthetic arrays only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import e00734_panel as pan

N = 2000
SITES = [f"S{i}" for i in range(12)]


def make(*, loaded: bool = True) -> tuple[dict, dict, dict]:
    """Twelve Primaries that share a weather wave (loading 0.8) with independent noise."""
    rng = np.random.default_rng(0)
    wave = np.cumsum(rng.standard_normal(N)) * 0.1
    wave -= np.convolve(wave, np.ones(40) / 40, mode="same")
    y, slow, sigma = {}, {}, {}
    for i, s in enumerate(SITES):
        sigma[s] = 1.0 + 0.1 * i
        slow[s] = np.zeros(N)
        y[s] = 0.5 * rng.standard_normal(N) + (0.8 * sigma[s] * wave * 3 if loaded else 0.0)
    return y, slow, sigma


def test_the_shared_term_leaves_a_primary_out_of_its_own_median() -> None:
    """Test a large event at one Primary does not enter that Primary's shared term."""
    y, slow, sigma = make()
    y["S0"] = y["S0"] + 50.0
    c = pan.shared_term(y=y, slow=slow, sigma=sigma)
    assert np.abs(c["S0"]).max() < 10.0
    assert np.abs(c["S1"]).max() < 10.0
    base = pan.shared_term(y=make()[0], slow=slow, sigma=sigma)
    assert np.abs(c["S1"] - base["S1"]).max() < 2.0


def test_the_shared_term_is_zero_with_fewer_than_eight_other_readings() -> None:
    """Test the term is 0 at a block where fewer than 8 other Primaries have a reading."""
    y, slow, sigma = make()
    for s in SITES[1:6]:
        y[s][100:200] = np.nan
    c = pan.shared_term(y=y, slow=slow, sigma=sigma)
    assert (c["S0"][100:200] == 0.0).all()
    assert (c["S0"][300:400] != 0.0).any()


def test_the_loading_is_recovered_and_the_switch_is_on_only_where_the_term_helps() -> None:
    """Test a loading of 0.8 is found by the Huber fit and the switch follows the noise gain."""
    y, slow, sigma = make()
    c = pan.shared_term(y=y, slow=slow, sigma=sigma)
    beta = pan.fit_loading(y=y["S3"], slow=slow["S3"], c=c["S3"], sigma=sigma["S3"], upto_block=N)
    assert beta == pytest.approx(0.8, abs=0.3)
    loading = {
        s: pan.fit_loading(y=y[s], slow=slow[s], c=c[s], sigma=sigma[s], upto_block=N)
        for s in SITES
    }
    on = pan.shared_switch(y=y, slow=slow, c=c, sigma=sigma, loading=loading, upto_block=N)
    assert sum(on.values()) >= 10
    y0, slow0, sigma0 = make(loaded=False)
    c0 = pan.shared_term(y=y0, slow=slow0, sigma=sigma0)
    load0 = {
        s: pan.fit_loading(y=y0[s], slow=slow0[s], c=c0[s], sigma=sigma0[s], upto_block=N)
        for s in SITES
    }
    off = pan.shared_switch(y=y0, slow=slow0, c=c0, sigma=sigma0, loading=load0, upto_block=N)
    assert sum(off.values()) <= 2


def test_block_scale_is_the_mad_of_each_block_of_the_day_outside_the_mask() -> None:
    """Test the Student-t scale per block of the day ignores masked blocks."""
    rng = np.random.default_rng(1)
    r = rng.standard_normal(4000) * np.tile([1.0, 2.0, 2.0, 1.0], 1000)
    mask = np.zeros(4000, dtype=bool)
    r[100:140] += 50.0
    mask[100:140] = True
    scale = pan.block_scale(r=r, mask=mask)
    np.testing.assert_allclose(scale, [1.0, 2.0, 2.0, 1.0], rtol=0.15)


def test_the_candidate_mask_covers_a_long_step_and_widens_it() -> None:
    """Test blocks of a 10-day step and two days either side are masked."""
    y = np.random.default_rng(2).standard_normal(1000) * 0.2
    y[300:340] += 3.0
    mask = pan.candidate_mask(y=y, slow=np.zeros(1000), sigma=0.2)
    assert mask[310:330].all()
    assert not mask[:200].any()
    assert mask[295]

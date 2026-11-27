# ruff: noqa: PT018
"""Tests of `e00733_score`: paired contrasts on the same rows, the t widening, the guards, the
Brier readings, and the verdict files."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00733_core as core
import e00733_score as sc
import numpy as np
import polars as pl
import pytest


def _frame(
    *, loss: float, months: int = 8, per: int = 30, noise: np.ndarray | None = None
) -> pl.DataFrame:
    rows: dict[str, list] = {
        "time_series_id": [],
        "power_fcst_init_time": [],
        "valid_time": [],
        "crps": [],
    }
    gen = np.random.default_rng(1)
    for m in range(months):
        for i in range(per):
            init = datetime(2025 + (10 + m) // 12, (10 + m) % 12 + 1, 1 + i % 20, 9, tzinfo=UTC)
            rows["time_series_id"].append(26)
            rows["power_fcst_init_time"].append(init)
            rows["valid_time"].append(init + timedelta(hours=30, minutes=30 * i))
            rows["crps"].append(
                loss + float(gen.normal(0, 0.1))
                if noise is None
                else loss + float(noise[m * per + i])
            )
    return pl.DataFrame(rows).with_columns(pl.col("time_series_id").cast(pl.Int32))


def test_real_contrast_pairs_on_keys_and_widens_by_t() -> None:
    """A clear gain has an upper end below zero, the interval is widened by the 7-df t ratio, and
    the sign is relative to the control."""
    ref = _frame(loss=2.0)
    treat = ref.with_columns(crps=pl.col("crps") - 0.2)
    c = sc.real_contrast(treatment=treat, reference=ref, n_resamples=2000)
    assert c.point == pytest.approx(-0.2)
    assert c.relative().gain and c.n_rows == 240
    assert c.widening == pytest.approx(core.t_widening(df=7))
    assert c.widening > 1.0 and c.lower <= c.raw_lower and c.upper >= c.raw_upper
    assert len(c.monthly) == 8


def test_real_contrast_refuses_different_keys() -> None:
    """A treatment frame on other keys is an error, never a position-based pairing."""
    ref = _frame(loss=2.0)
    other = ref.with_columns(valid_time=pl.col("valid_time") + pl.duration(days=1))
    with pytest.raises(sc.ScoreError):
        sc.paired_difference(treatment=other, reference=ref, metric="crps")
    with pytest.raises(sc.ScoreError):
        sc.paired_difference(treatment=ref.head(5), reference=ref, metric="crps")


def test_guards() -> None:
    """The leave-one-month-out guard fails when one month carries the sign, and the tails and the
    MAE guards are read as the plan states."""
    assert sc.leave_one_month_out_keeps_sign(monthly=[-1, -1, -1, -1], point=-1.0)
    assert not sc.leave_one_month_out_keeps_sign(monthly=[-10, 1, 1, 1], point=-1.75)
    assert sc.mae_moves_with_crps(crps_point=-1.0, mae_point=-0.1)
    assert not sc.mae_moves_with_crps(crps_point=-1.0, mae_point=0.1)
    assert sc.tails_not_worse(q01=0.05, q99=-1.0) and not sc.tails_not_worse(q01=0.06, q99=0.0)


def test_lead_masks() -> None:
    """The headline band includes 24 h and excludes 240 h; lead day k covers hours 24 (k - 1) up to
    24 k."""
    lead = np.array([23.5, 24.0, 239.5, 240.0])
    assert sc.headline_mask(lead_hours=lead).tolist() == [False, True, True, False]
    assert sc.lead_band_mask(lead_hours=lead, days=(2, 3)).tolist() == [False, True, False, False]


def test_brier_and_log_loss_known_answers() -> None:
    """Known-answer test of the Brier score and the log loss (verdict row 1 if it failed)."""
    p, t = np.array([0.0, 1.0, 0.5, 0.2]), np.array([0.0, 1.0, 1.0, np.nan])
    b = sc.brier_rows(p=p, truth=t)
    assert np.nanmean(b) == pytest.approx(0.25 / 3)
    ll = sc.log_loss_rows(p=np.array([0.5]), truth=np.array([1.0]))
    assert ll[0] == pytest.approx(np.log(2))
    assert sc.reliability_term(
        p=np.full(100, 0.3), truth=np.r_[np.ones(30), np.zeros(70)]
    ) == pytest.approx(0.0)
    assert sc.reliability_term(p=np.full(100, 0.9), truth=np.zeros(100)) == pytest.approx(0.81)


def _synth(*, n_seg: int = 6, n_days: int = 30) -> tuple[sc.SynthRows, np.ndarray]:
    seg = np.repeat(np.arange(n_seg), n_days)
    day = np.tile(np.arange(n_days), n_seg)
    rows = sc.synthetic_rows(seg=seg, origin_day=day, seg_month=np.arange(n_seg) % 2)
    truth = (np.random.default_rng(0).random(len(seg) * 48) < 0.2).astype(float)
    return rows, truth


def test_synthetic_contrast_uses_two_way_bootstrap_and_drops_nan() -> None:
    """A probability closer to the truth than the reference's is a gain; half-hours with a NaN in
    either arm are left out of both."""
    rows, truth = _synth()
    good = np.where(truth > 0.5, 0.8, 0.1)
    ref = np.full_like(truth, 0.2)
    ref[::7] = np.nan
    r = sc.synthetic_contrast(
        rows=rows,
        first=sc.brier_rows(p=good, truth=truth),
        second=sc.brier_rows(p=ref, truth=truth),
        n_resamples=500,
    )
    assert sc.reading_of(contrast=r).gain and r.n_rows == int(np.isfinite(ref).sum())
    assert r.n_resamples == 500 and r.kish_blocks > 0


def test_sign_forecast_inputs() -> None:
    """Skill is one minus the ratio of Brier scores to B-clim, and the reverse-flow count is of
    half-hours."""
    truth = np.r_[np.ones(600), np.zeros(1400)]
    out = sc.sign_forecast_inputs(
        p_yu=truth * 0.9 + 0.05, p_clim=np.full(2000, 0.3), truth=truth, seeds_present=True
    )
    assert (
        out["skill"] > 0.9
        and out["n_reverse"] == 600
        and out["share_reverse"] == pytest.approx(0.3)
    )
    assert core.sign_forecast_word(c3=core.Reading(-0.1, -0.2, -0.05), **out) == "skilful"


def test_assemble_follows_the_verdict_table_and_voids_dropped_arms(tmp_path: Path) -> None:
    """Row 3 needs C1 and C2 gains, the point at or below -g, and the guards; a dropped arm voids
    its readings; the files are written."""
    g = lambda p, hw: core.Reading(p, p - hw, p + hw)  # noqa: E731

    class R:
        def __init__(self, r: core.Reading) -> None:
            self.reading = r
            self.kish_blocks = self.kish_segments = 30.0
            self.widening = 1.0
            self.n_rows = 10
            self.n_resamples = 5

    contrasts = {
        "C1": R(g(-0.04, 0.01)),
        "C2": R(g(-0.03, 0.01)),
        "C3": None,
        "C4": None,
        "R1": None,
    }
    guards = {"all": True}
    v, r = sc.assemble(
        stopped=False,
        invalid=False,
        failed=False,
        c=contrasts,
        guards=guards,
        sign_inputs=None,
        dropped_arms=("FS",),
        coverage={"n_gap_base_dates": 3},
    )
    assert v["row"] == 3 and v["word"] == "supported" and v["sign_forecast_word"] == "void"
    assert r["R1"] == "void" and v["synthetic_coverage"] == {"n_gap_base_dates": 3}
    v4, _ = sc.assemble(
        stopped=False,
        invalid=False,
        failed=False,
        c={**contrasts, "C1": R(g(-0.015, 0.01))},
        guards=guards,
        sign_inputs=None,
    )
    assert v4["row"] == 4
    v5, _ = sc.assemble(
        stopped=False,
        invalid=False,
        failed=False,
        c=contrasts,
        guards={"all": False},
        sign_inputs=None,
    )
    assert v5["row"] == 5
    v0, _ = sc.assemble(
        stopped=True, invalid=True, failed=False, c=contrasts, guards=guards, sign_inputs=None
    )
    assert v0["row"] == 0 and v0["front_matter"] == "inconclusive"
    sc.write_results(root=tmp_path, verdict=v, readings=r)
    assert json.loads((tmp_path / "scores" / "verdict.json").read_text())["row"] == 3

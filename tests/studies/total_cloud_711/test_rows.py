"""Tests of the Primary demand and solar row builders of experiment 711 on synthetic arrays."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00711_arms as arms
import e00711_common as common
import e00711_placebo as placebo
import e00711_rowcloud as rowcloud
import e00711_solar as solar
import numpy as np
import polars as pl
import pytest

US_PER_DAY = 86_400_000_000
FIRST = datetime(2024, 3, 25, tzinfo=UTC)
N_RUNS = 12


def _store() -> rowcloud.RunArrays:
    """Return a store of 12 runs from 2024-03-25, with NaN cloud on the gap runs (from 04-01)."""
    rng = np.random.default_rng(0)
    values = rng.uniform(0.1, 0.9, (N_RUNS, rowcloud.STEPS, len(rowcloud.CHANNELS))).astype(
        np.float32
    )
    for i in range(N_RUNS):
        if common.is_missing_run(run=FIRST + timedelta(days=i)):
            values[i, :, :5] = np.nan
    return rowcloud.RunArrays(first_day=placebo.day_number(run=FIRST), values=values)


def _rows() -> tuple[np.ndarray, np.ndarray]:
    """Return 4 rows: runs 03-30, 03-31, 04-02 and 04-03, all at lead 1 hour."""
    days = [
        placebo.day_number(run=datetime(2024, m, d, tzinfo=UTC))
        for m, d in ((3, 30), (3, 31), (4, 2), (4, 3))
    ]
    return np.array(days, dtype=np.int64) * US_PER_DAY, np.full(4, 1.0)


def _donors() -> np.ndarray:
    """Return a map: each run reads the run four days earlier, and gap runs are unmapped."""
    days = [placebo.day_number(run=FIRST + timedelta(days=i)) for i in range(N_RUNS)]
    return np.array(
        [
            placebo.UNMAPPED if common.is_missing_run(run=FIRST + timedelta(days=i)) else d - 4
            for i, d in enumerate(days)
        ]
    )


def test_row_extras_real_reads_the_rows_own_run_and_masks_the_gap():
    store = _store()
    nwp_init, lead = _rows()
    values, flag = rowcloud.row_extras(
        arm=arms.ARMS["D1"],
        store=store,
        nwp_init=nwp_init,
        lead_hours=lead,
        donors=None,
        donors_first_day=0,
    )
    assert flag.tolist() == [False, False, True, True]
    assert np.isnan(values[2:]).all()
    run_index = int(nwp_init[0] // US_PER_DAY) - store.first_day
    assert values[0].tolist() == pytest.approx(store.values[run_index, 1, :2].tolist())


def test_row_extras_placebo_reads_the_donor_at_the_same_step_with_the_real_flag():
    store = _store()
    nwp_init, lead = _rows()
    first_day = placebo.day_number(run=FIRST)
    values, flag = rowcloud.row_extras(
        arm=arms.ARMS["D1p"],
        store=store,
        nwp_init=nwp_init,
        lead_hours=lead,
        donors=_donors(),
        donors_first_day=first_day,
    )
    assert flag.tolist() == [False, False, True, True]
    donor_index = int(nwp_init[0] // US_PER_DAY) - 4 - store.first_day
    assert values[0].tolist() == pytest.approx(store.values[donor_index, 1, :2].tolist())
    assert np.isnan(values[2:]).all()
    with pytest.raises(ValueError, match="needs a donor map"):
        rowcloud.row_extras(
            arm=arms.ARMS["D1p"],
            store=store,
            nwp_init=nwp_init,
            lead_hours=lead,
            donors=None,
            donors_first_day=first_day,
        )


def test_a_hole_in_a_present_run_is_refused_instead_of_becoming_a_silent_mask():
    store = _store()
    nwp_init, lead = _rows()
    run_index = int(nwp_init[0] // US_PER_DAY) - store.first_day
    store.values[run_index, 1, 0] = np.nan
    with pytest.raises(ValueError, match="differ from the cloud_missing flag"):
        rowcloud.row_extras(
            arm=arms.ARMS["D1"],
            store=store,
            nwp_init=nwp_init,
            lead_hours=lead,
            donors=None,
            donors_first_day=0,
        )
    rowcloud.assert_missing_is_nan(values=np.array([[np.nan], [1.0]]), flag=np.array([True, False]))


def test_extend_columns_puts_the_flag_last_and_the_temperature_back_as_masked_extras():
    base = ["a", *arms.TEMPERATURE_BASE_COLUMNS, "b"]
    mask = np.ones(len(base), dtype=bool)
    names, new_mask, kept = rowcloud.extend_columns(
        base_names=base, base_mask=mask, arm=arms.ARMS["D1"]
    )
    assert names == [*base, "cloud_mean", "cloud_spread", "cloud_missing"]
    assert new_mask.tolist() == [True] * len(base) + [True, True, False]
    assert kept.tolist() == list(range(len(base)))
    k_names, k_mask, k_kept = rowcloud.extend_columns(
        base_names=base, base_mask=mask, arm=arms.ARMS["K1"]
    )
    assert k_names == ["a", "b", "temp_mean", "temp_std", "dew_mean", "dew_std", "cloud_missing"]
    assert k_kept.tolist() == [0, 5]
    assert k_mask.tolist() == [True, True, True, True, True, True, False]
    plain, _, _ = rowcloud.extend_columns(base_names=base, base_mask=mask, arm=arms.ARMS["D0n"])
    assert plain == base


def test_extend_matrix_orders_kept_base_summaries_and_flag():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = rowcloud.extend_matrix(
        x=x,
        kept=np.array([0, 3]),
        summaries=np.array([[0.5], [np.nan], [0.7]]),
        flag=np.array([False, True, False]),
    )
    assert out.shape == (3, 4)
    assert out[:, 0].tolist() == [0, 4, 8]
    assert out[:, 1].tolist() == [3, 7, 11]
    assert np.isnan(out[1, 2])
    assert out[:, 3].tolist() == [0, 1, 0]
    assert rowcloud.extend_matrix(
        x=x, kept=np.arange(4), summaries=np.zeros((3, 0)), flag=None
    ).shape == (3, 4)


# --- solar ------------------------------------------------------------------------------------


def _solar_frame(*, runs: list[datetime], with_actual: bool = True) -> pl.DataFrame:
    rows = []
    for run in runs:
        for step in range(2, 8):
            time = run + timedelta(hours=12, minutes=30 * step)
            missing = common.is_missing_run(run=run)
            rows.append(
                {
                    "series": "A",
                    "init_time": run,
                    "time": time,
                    "lead_hours": 12.0 + 0.5 * step,
                    "actual": 0.4 if with_actual else None,
                    "cloud_missing": missing,
                    "cloud_mean": None if missing else 0.1 * step + run.day / 100,
                    "cloud_spread": None if missing else 0.05 * step,
                    "cloud_clear_share": None if missing else 0.2,
                    "cloud_overcast_share": None if missing else 0.3,
                    "cloud_daylight_mean": None if missing else 0.5,
                }
            )
    return pl.DataFrame(rows).cast(
        {"init_time": pl.Datetime("us", "UTC"), "time": pl.Datetime("us", "UTC")}
    )


def test_select_rows_cuts_training_at_the_refit_and_targets_the_month():
    runs = [datetime(2026, 5, 20, tzinfo=UTC) + timedelta(days=i) for i in range(30)]
    frame = _solar_frame(runs=runs)
    refit = datetime(2026, 6, 1, tzinfo=UTC)
    selected = solar.select_rows(frame=frame, refit=refit, arm=arms.ARMS["P1"])
    holdout = datetime(2026, 7, 1, tzinfo=UTC)
    gap = timedelta(days=solar.REFIT_GAP_DAYS)
    assert selected.train.filter(pl.col("init_time") >= refit).is_empty()
    assert selected.train.filter(pl.col("time") >= refit - gap).is_empty()
    assert selected.target.filter(pl.col("init_time") < refit).is_empty()
    assert selected.target.filter(pl.col("init_time") >= holdout).is_empty()
    assert selected.target.filter(pl.col("time") >= holdout).is_empty()
    assert selected.farm_order == ("A",)
    with pytest.raises(ValueError, match="training rows"):
        solar.select_rows(
            frame=frame.filter(pl.col("actual").is_null()), refit=refit, arm=arms.ARMS["P1"]
        )


def test_select_rows_drops_gap_runs_from_training_for_the_same_row_pair():
    gap = [datetime(2024, 5, 1, tzinfo=UTC) + timedelta(days=i) for i in range(5)]
    recent = [datetime(2025, 11, 25, tzinfo=UTC) + timedelta(days=i) for i in range(40)]
    frame = _solar_frame(runs=[*gap, *recent])
    refit = datetime(2025, 12, 15, tzinfo=UTC)
    kept = solar.select_rows(frame=frame, refit=refit, arm=arms.ARMS["P1"])
    assert kept.train["cloud_missing"].any()
    dropped = solar.select_rows(frame=frame, refit=refit, arm=arms.ARMS["P1g"])
    assert not dropped.train["cloud_missing"].any()
    assert dropped.target.height == kept.target.height


def test_apply_solar_donors_reads_the_donor_runs_cloud_at_the_same_lead():
    runs = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)]
    full = _solar_frame(runs=runs)
    rows = full.filter(pl.col("init_time").is_in(runs[3:5]))
    donors = np.array(
        [
            *[placebo.UNMAPPED] * 3,
            placebo.day_number(run=runs[0]),
            placebo.day_number(run=runs[1]),
            placebo.UNMAPPED,
        ]
    )
    out = solar.apply_solar_donors(rows=rows, full=full, targets=runs, donors=donors)
    assert out.height == rows.height
    first = out.filter(pl.col("init_time") == runs[3]).sort("lead_hours")
    donor = full.filter(pl.col("init_time") == runs[0]).sort("lead_hours")
    assert first["cloud_mean"].to_list() == donor["cloud_mean"].to_list()
    unmapped = solar.apply_solar_donors(
        rows=full.filter(pl.col("init_time") == runs[5]), full=full, targets=runs, donors=donors
    )
    assert unmapped["cloud_mean"].is_null().all()
    with pytest.raises(ValueError, match="no donor half-hour"):
        solar.apply_solar_donors(
            rows=rows, full=full.filter(pl.col("init_time") != runs[0]), targets=runs, donors=donors
        )


def test_every_solar_donor_of_both_maps_is_a_run_that_the_farm_table_keeps():
    targets = solar.target_runs()
    available = common.cache_runs()
    held = {placebo.day_number(run=r) for r in available}
    shifted = placebo.shifted_donor_map(targets=targets, available=available)
    mapped = shifted[shifted != placebo.UNMAPPED]
    assert set(mapped.tolist()) <= held
    assert placebo.day_number(run=common.CACHE_FIRST_RUN) in mapped.tolist()
    cut = datetime(2026, 1, 1, tzinfo=UTC)
    shuffled = placebo.shuffled_donor_map(
        targets=targets, available=available, cut=cut, seed=0, series_key=1000
    )
    assert set(shuffled[shuffled != placebo.UNMAPPED].tolist()) <= held
    assert common.CACHE_FIRST_RUN + timedelta(days=14) == solar.ROW_FIRST_RUN


def test_solar_standardisation_fills_masked_rows_with_the_mean_so_they_standardise_to_zero():
    arm = arms.ARMS["P1"]
    columns = solar.input_columns(arm=arm)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, len(columns)))
    cloud = [columns.index(c) for c in ("cloud_mean", "cloud_spread")]
    flag = columns.index(arms.FLAG_COLUMN)
    masked = np.arange(50) % 5 == 0
    x[np.ix_(masked, cloud)] = np.nan
    x[:, flag] = masked.astype(float)
    mean, std, filled = solar.standardisation(x=x, is_fit=np.ones(50, bool), columns=columns)
    standardised = (filled - mean) / std
    assert standardised[np.ix_(masked, cloud)] == pytest.approx(0.0, abs=1e-9)
    assert mean[flag] == 0.0
    assert std[flag] == 1.0
    assert mean[cloud[0]] == pytest.approx(np.nanmean(x[:, cloud[0]]))
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="base column"):
        solar.standardisation(x=bad, is_fit=np.ones(50, bool), columns=columns)
    all_masked = x.copy()
    all_masked[:, cloud[0]] = np.nan
    with pytest.raises(ValueError, match="no unmasked fit row"):
        solar.standardisation(x=all_masked, is_fit=np.ones(50, bool), columns=columns)


def test_the_stored_plp_reader_keeps_the_month_and_drops_the_origin_half_hour(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import e00711_score as score

    pytest.importorskip("e00666_common")
    from studies import pv_dataset

    run = datetime(2025, 12, 3, tzinfo=UTC)
    quantiles = [f"q{i}" for i in range(11)]
    frame = pl.DataFrame(
        {
            "time_series_id": [7, 7, 7],
            "init_time": [run, run, datetime(2025, 11, 30, tzinfo=UTC)],
            "time": [
                run + timedelta(hours=12),
                run + timedelta(hours=12, minutes=30),
                run + timedelta(hours=14),
            ],
            **{q: [2.0, 2.0, 2.0] for q in quantiles},
        }
    ).cast({"init_time": pl.Datetime("us", "UTC"), "time": pl.Datetime("us", "UTC")})
    path = tmp_path / "refit=2025-11-01_seed=0.parquet"
    frame.write_parquet(path)
    monkeypatch.setattr(score, "stored_p0_files", lambda *, seed: [path])
    monkeypatch.setattr(
        pv_dataset,
        "pv_sites",
        lambda: pl.DataFrame({"time_series_id": [7], "effective_capacity_mw": [4.0]}),
    )
    import e00666_common

    monkeypatch.setattr(e00666_common, "QUANTILES", tuple(quantiles))
    out = score.read_stored_solar(month="2025-12", seed=0)
    assert out.height == 1
    assert out["power_fcst_init_time"][0] == run + timedelta(hours=12)
    assert out["valid_time"][0] == run + timedelta(hours=12, minutes=30)
    assert out["q50"][0] == pytest.approx(0.5)


def test_the_replicate_fine_tune_window_keeps_nothing_before_2021_07_20():
    pytest.importorskip("torch")
    import e00707_folds as folds707
    import e00711_fit as fit

    refit = datetime(2023, 4, 1, tzinfo=UTC)
    start = datetime(2021, 4, 1, tzinfo=UTC)
    valid = [start + timedelta(days=i) for i in range(400)]
    micro = np.array([int(v.timestamp() * 1e6) for v in valid], dtype=np.int64)
    z = {"valid": micro, "origin": micro - 24 * 3_600_000_000, "lead": np.full(len(valid), 24.0)}
    window = fit.fine_tune_window(arm=arms.ARMS["R1"])
    assert fit.fine_tune_window(arm=arms.ARMS["D1"]) != window
    keep = folds707.window_mask(z=z, refit=refit, window=window)
    floor = int(datetime(2021, 7, 20, tzinfo=UTC).timestamp() * 1e6)
    assert keep.any()
    assert (micro[keep] >= floor).all()
    assert ((micro[keep] - 7 * 24 * 3_600_000_000) >= floor).all()


def test_take_with_no_channels_returns_an_empty_column_block():
    """The no-cloud reference asks for no channels; the lookup must not fail on a float index."""
    store = _store()
    block = store.take(run_days=np.array([store.first_day]), steps=np.array([0]), channels=[])
    assert block.shape == (1, 0)

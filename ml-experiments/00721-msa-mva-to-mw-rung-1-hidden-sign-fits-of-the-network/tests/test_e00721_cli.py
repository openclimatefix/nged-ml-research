# ruff: noqa: E501
"""Tests of the command-line entry's pure parts, the driver's MDE and power-check steps, and the synthetic world on a real-style grid."""

import datetime
import json
from pathlib import Path

import e00721_cli as cli
import e00721_core as core
import e00721_driver as drv
import e00721_fit as fit
import e00721_score as score
import e00721_worlds as worlds
import numpy as np
import polars as pl
import pytest

IDS = {"D5": 5, "D10": 10}


def _rows(*, q: float, n_days: int = 12, ids: tuple[int, ...] = (5, 10, 3)) -> pl.DataFrame:
    rows = []
    for series in ids:
        for d in range(n_days):
            base = {
                "time_series_id": series, "power_fcst_init_time": datetime.datetime(2026, 5, 1) + datetime.timedelta(days=d),
                "valid_time": datetime.datetime(2026, 5, 2) + datetime.timedelta(days=d), "lead_hours": 30.0, "fold": "2026-05", "seed": 0,
                "actual": -0.02 if series != 3 else 0.5, "reading": 0.02,
            }  # fmt: skip
            base |= dict.fromkeys(score.QCOLS, q) | {c: abs(q) for c in score.FOLDED_QCOLS}
            rows.append(base)
    return pl.DataFrame(rows)


def test_seed_difference_is_the_crps_gap_on_the_reverse_flow_rows_of_the_hidden_pair(
    tmp_path: Path,
) -> None:
    for seed, q in ((0, -0.02), (1, -0.01)):
        job = fit.Job(arm="SG-0", world="W2", fold=core.MDE_FOLD, seed=seed)
        path = fit.job_paths(data_dir=tmp_path, job=job)["predictions"]
        path.parent.mkdir(parents=True, exist_ok=True)
        _rows(q=q).with_columns(seed=pl.lit(seed)).write_parquet(path)
    got = drv.seed_difference_points(data_dir=tmp_path, hidden_ids=IDS)
    assert got == pytest.approx(-1.0, abs=1e-6)


def test_seed_difference_names_the_missing_fit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="seed 0 is missing"):
        drv.seed_difference_points(data_dir=tmp_path, hidden_ids=IDS)


def test_power_check_passes_when_the_stored_seeds_agree_and_names_a_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="power check needs"):
        drv.power_check(m1_dir=tmp_path, folds=("2026-05",), series_ids=IDS)
    for seed in range(5):
        _rows(q=-0.02).drop("fold", "seed").with_columns(seed=pl.lit(seed)).write_parquet(
            tmp_path / f"2026-05_s{seed}.parquet"
        )
    got = drv.power_check(m1_dir=tmp_path, folds=("2026-05",), series_ids=IDS)
    assert got["n_assignments"] == 60
    assert got["joint_pass_rate"] == pytest.approx(1.0)
    assert got["pass"]


def test_power_check_drops_leads_outside_24_to_240_hours(tmp_path: Path) -> None:
    for seed in range(5):
        frame = (
            pl.concat([_rows(q=-0.02), _rows(q=1.0).with_columns(lead_hours=pl.lit(3.0))])
            .drop("fold", "seed")
            .with_columns(seed=pl.lit(seed))
        )
        frame.write_parquet(tmp_path / f"2026-05_s{seed}.parquet")
    assert drv.power_check(m1_dir=tmp_path, folds=("2026-05",), series_ids=IDS)["pass"]


def test_score_all_writes_the_mde_file_from_the_seed_difference_once(tmp_path: Path) -> None:
    mde = tmp_path / "mde.json"
    drv.score_all(
        data_dir=tmp_path, folds=("2026-05",), seeds=(0, 1), hidden_ids=IDS, mva_ids=[3], n_resamples=20, mde_path=mde, seed_difference=0.9,
    )  # fmt: skip
    assert core.read_mde_file(path=mde) == pytest.approx(0.9)
    drv.score_all(
        data_dir=tmp_path, folds=("2026-05",), seeds=(0, 1), hidden_ids=IDS, mva_ids=[3], n_resamples=20, mde_path=mde, seed_difference=5.0,
    )  # fmt: skip
    assert core.read_mde_file(path=mde) == pytest.approx(0.9)


def test_jobs_of_a_wave_or_one_named_job() -> None:
    assert cli.jobs_of(wave=1, arm=None, world=None, fold=None, seed=None) == fit.wave_jobs(wave=1)
    assert cli.jobs_of(wave=None, arm="P0", world="W2", fold="2026-05", seed=1) == [
        fit.Job(arm="P0", world="W2", fold="2026-05", seed=1)
    ]
    with pytest.raises(ValueError, match="name a --wave"):
        cli.jobs_of(wave=None, arm="P0", world=None, fold=None, seed=None)


def test_spent_hours_sums_the_stored_curves(tmp_path: Path) -> None:
    folder = fit.data_root(data_dir=tmp_path) / "fits"
    folder.mkdir(parents=True)
    for i, hours in enumerate((0.1, 0.25)):
        (folder / f"j{i}.curve.json").write_text(json.dumps({"hours": hours}))
    assert cli.spent_hours(data_dir=tmp_path) == pytest.approx(0.35)
    assert cli.spent_hours(data_dir=tmp_path / "none") == 0.0


def test_site_ids_order_the_primaries_by_label_number() -> None:
    table = [
        {"site": "D10", "time_series_id": 9, "substation_type": "Primary"}, {"site": "D2", "time_series_id": 4, "substation_type": "Primary"},
        {"site": "D1", "time_series_id": 1, "substation_type": "BSP"},
    ]  # fmt: skip
    sites, ids = cli.site_ids(table=table)
    assert sites == ["D2", "D10"]
    assert ids == {"D2": 4, "D10": 9}


def test_synthetic_on_real_grid_fills_only_the_chosen_primaries_with_the_real_calendar() -> None:
    pytest.importorskip("e00720_worlds")
    n_grid = 48 * 400
    stamp = (
        np.datetime64("2025-01-01T00:30:00", "us").astype(np.int64)
        + np.arange(n_grid, dtype=np.int64) * 1_800_000_000
    )
    rng = np.random.default_rng(0)
    day = np.arange(n_grid) // 48
    half_hour = np.arange(n_grid) % 48
    bell = np.clip(np.sin(np.pi * (half_hour - 12) / 24.0), 0.0, None) ** 2
    u = np.stack([bell * rng.uniform(0.2, 1.0, day.max() + 1)[day] for _ in range(6)])
    temperature = 10.0 + 5.0 * np.sin(2 * np.pi * day / 365.0)[None, :] + np.zeros((6, n_grid))
    table = np.random.default_rng(0).uniform(-0.1, 0.1, size=(8, 48))
    day_type = (day % 7).astype(np.int64)

    def build() -> tuple[worlds.SyntheticArrays, np.ndarray, np.ndarray]:
        return worlds.synthetic_on_real_grid(
            temperature_c=temperature, u=u, stamp_us=stamp, day_type=day_type, half_hour=half_hour, table=table,
            primaries=(1, 2, 4, 5), exporters=(1, 2), seed=3,
        )  # fmt: skip

    syn, signed, reading = build()
    assert signed.shape == reading.shape == (6, n_grid)
    assert np.isnan(signed[[0, 3]]).all()
    assert np.isnan(reading[[0, 3]]).all()
    np.testing.assert_array_equal(signed[[1, 2, 4, 5]], syn.x)
    assert (syn.export_share[:2] > 0.5).all()
    assert (syn.export_share[2:] == 0.0).all()
    again, _, _ = build()
    np.testing.assert_array_equal(syn.x, again.x)
    demand = worlds.known_demand(
        temperature_c=temperature[[1]],
        day_type=day_type,
        half_hour=half_hour,
        table=table,
    )
    np.testing.assert_allclose(syn.demand_mean[0], demand[0])

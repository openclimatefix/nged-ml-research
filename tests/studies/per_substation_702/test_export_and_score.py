# ruff: noqa: ANN001, E501, PT018
"""Tests of `e00702_export.py` and the pilot and bank-match modes of `e00702_score.py` on synthetic files."""

from datetime import UTC, datetime, timedelta

import e00702_core as core
import polars as pl
import pytest

x667 = pytest.importorskip("e00667_export")
export = pytest.importorskip("e00702_export")
score = pytest.importorskip("e00702_score")


def forecast_file(path, *, series_id: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 3
    frame = pl.DataFrame(
        {
            "time_series_id": [series_id] * n,
            "power_fcst_init_time": [datetime(2026, 5, 1, 9, tzinfo=UTC)] * n,
            "valid_time": [datetime(2026, 5, 1, 9 + i, tzinfo=UTC) for i in range(n)],
            **{q: [float(i)] * n for i, q in enumerate(x667.QCOLS)},
        }
    ).with_columns(
        pl.col("time_series_id").cast(pl.Int32),
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )
    frame.write_parquet(path)


def test_export_reads_only_the_listed_primaries(tmp_path, monkeypatch):
    """With `--primaries`, the export joins those Primaries' files and checks their count, not 16."""
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(
        score,
        "sites",
        lambda: pl.DataFrame(
            {
                "primary": [0, 1, 2],
                "label": ["D2", "D3", "D4"],
                "time_series_id": [10, 11, 12],
                "capacity_mw": [1.0] * 3,
                "group": ["g"] * 3,
            }
        ),
    )
    for label, ts in (("D2", 10), ("D4", 12)):
        forecast_file(
            tmp_path
            / "arms"
            / "split1"
            / "2026-05"
            / label
            / "work"
            / "fits"
            / "P5"
            / "2026-05_d0_s0.parquet",
            series_id=ts,
        )
    frame = export.seed_frame(arm="split1", suffix="", month="2026-05", seed=0, labels=("D2", "D4"))
    assert sorted(frame["time_series_id"].unique().to_list()) == [10, 12]
    with pytest.raises(AssertionError):
        export.seed_frame(arm="split1", suffix="", month="2026-05", seed=0, labels=("D2", "D3"))


def test_pooled_export_keeps_only_the_listed_primaries_of_the_pooled_file(tmp_path, monkeypatch):
    """A `global_pilot` file holds every Primary; `--primaries` filters it to the listed ones."""
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(
        score,
        "sites",
        lambda: pl.DataFrame(
            {
                "primary": [0, 1],
                "label": ["D2", "D3"],
                "time_series_id": [10, 11],
                "capacity_mw": [1.0] * 2,
                "group": ["g"] * 2,
            }
        ),
    )
    path = (
        tmp_path
        / "arms"
        / "global_pilot"
        / "2026-05"
        / "pooled"
        / "work"
        / "fits"
        / "P5"
        / "2026-05_d0_s0.parquet"
    )
    forecast_file(path, series_id=10)
    both = pl.concat(
        [
            pl.read_parquet(path),
            pl.read_parquet(path).with_columns(time_series_id=pl.lit(11, pl.Int32)),
        ]
    )
    both.write_parquet(path)
    frame = export.seed_frame(
        arm="global_pilot", suffix="", month="2026-05", seed=0, labels=("D3",)
    )
    assert frame["time_series_id"].unique().to_list() == [11]


def test_pilot_primaries_follow_the_rule():
    """The largest signed-MW Primary, the largest MVA-only Primary, and D15."""
    chosen = score.pilot_primaries()
    assert len(chosen) == 3 and chosen[2] == "D15"
    site = score.sites()
    assert site.filter(pl.col("label") == chosen[0])["group"][0] == "signed-MW six"
    assert site.filter(pl.col("label") == chosen[1])["group"][0] == "MVA-only eight"


def _native_frame(*, ids: list[int]) -> pl.DataFrame:
    """Return 14 days and 2 hours of half-hourly native-origin rows (09:00 UTC) for each series."""
    origin = datetime(2026, 5, 1, 9, tzinfo=UTC)
    valid = pl.datetime_range(
        origin + timedelta(minutes=30), origin + timedelta(days=14, hours=5), "30m", eager=True,
        time_zone="UTC",
    )  # fmt: skip
    n = len(valid)
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "time_series_id": [ts] * n,
                    "power_fcst_init_time": [origin] * n,
                    "valid_time": valid,
                    **{q: [0.01 * (i + 1) * (ts + 1)] * n for i, q in enumerate(x667.QCOLS)},
                }
            )
            for ts in ids
        ]
    ).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        pl.col(x667.QCOLS).cast(pl.Float32),
    )


def test_subset_contract_rows_equal_experiment_667s_rows_of_those_primaries(monkeypatch):
    """The copy gives experiment 667's rows for a subset, and still refuses a wrong series count."""
    ids = list(range(16))
    monkeypatch.setattr(
        x667,
        "capacity_mw",
        lambda: pl.DataFrame(
            {"time_series_id": ids, "capacity_mw": [1.0 + i for i in ids], "group": ["g"] * 16}
        ),
    )
    full = x667.contract_rows(mean=_native_frame(ids=ids))
    pick = [2, 7, 11]
    part = export.subset_contract_rows(mean=_native_frame(ids=pick), n_series=3)
    assert part.equals(full.filter(pl.col("time_series_id").is_in(pick)))
    with pytest.raises(AssertionError):
        export.subset_contract_rows(mean=_native_frame(ids=pick), n_series=2)
    with pytest.raises(AssertionError):
        export.subset_contract_rows(mean=_native_frame(ids=ids), n_series=16)

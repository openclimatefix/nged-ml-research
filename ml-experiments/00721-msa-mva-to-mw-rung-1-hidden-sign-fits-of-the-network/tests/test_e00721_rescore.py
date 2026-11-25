# ruff: noqa: E501
"""Tests of `e00721_rescore` on synthetic arrays (no stored file is read)."""

import datetime
import json
from pathlib import Path

import e00721_core as core
import e00721_paths
import numpy as np
import polars as pl
import pytest

e00721_paths.pin_threads()

import e00721_fit as fit  # noqa: E402
import e00721_rescore as rescore  # noqa: E402
import e00721_score as score  # noqa: E402

IDS = {"D5": 5, "D10": 10}
OFFSET = {"P0": 0.0, "SG-0": 0.02, "SG-A": 0.05}


def _frame(*, arm: str) -> pl.DataFrame:
    rng = np.random.default_rng(1)  # the same truth for every arm
    noise = np.random.default_rng({"P0": 10, "SG-0": 11, "SG-A": 12}[arm])
    rows = []
    for sid in IDS.values():
        for day in range(40):
            origin = datetime.datetime(2026, 5, 1, 12) + datetime.timedelta(days=day)
            for lead in (30.0, 60.0, 130.0):
                actual = float(rng.normal(0.0, 0.1))
                q = np.sort(actual + OFFSET[arm] + noise.normal(0.0, 0.03, len(score.QCOLS)))
                rows.append(
                    {
                        "time_series_id": sid, "power_fcst_init_time": origin, "valid_time": origin + datetime.timedelta(hours=lead),
                        "fold": "2026-05", "seed": 0, "actual": actual, "reading": abs(actual), "signed_known": True,
                    }
                    | dict(zip(score.QCOLS, q.tolist(), strict=True))
                    | {f"f{c}": abs(v) for c, v in zip(score.QCOLS, q.tolist(), strict=True)}
                )  # fmt: skip
    return score.with_crps(frame=pl.DataFrame(rows))


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> dict[str, pl.DataFrame]:
    frames = {arm: _frame(arm=arm) for arm in OFFSET}
    monkeypatch.setattr(rescore.drv, "load_frame", lambda *, data_dir, job: frames[job.arm])
    return frames


def test_the_point_is_the_mean_paired_difference_and_the_row_classes_add_up(
    stored: dict, tmp_path: Path
) -> None:
    result = rescore.rescore(data_dir=tmp_path, hidden_ids=IDS, n_resamples=200)
    body = result["contrasts"]["SG-A_minus_P0"]
    paired = score.pair_arms(a=stored["SG-A"], b=stored["P0"])
    imp = paired.filter(pl.col("actual_a") >= 0.0)
    want = float(imp.select((pl.col("crps_a") - pl.col("crps_b")).mean()).item())
    assert body["rows"]["pooled/import"]["point"] == pytest.approx(want)
    d5 = imp.filter(pl.col("time_series_id") == 5)
    assert body["rows"]["D5/import"]["n_rows"] == d5.height
    for scope in ("pooled", "D5", "D10"):
        assert (
            body["rows"][f"{scope}/import"]["n_rows"] + body["rows"][f"{scope}/reverse"]["n_rows"]
            == body["rows"][f"{scope}/all"]["n_rows"]
        )
        bands = sum(v["n_rows"] for k, v in body["lead_bands"].items() if k.startswith(f"{scope}/"))
        assert bands == body["rows"][f"{scope}/import"]["n_rows"]
    assert (
        sum(h["n_rows"] for h in body["hours"]["pooled/import"])
        == body["rows"]["pooled/import"]["n_rows"]
    )
    assert set(result["contrasts"]) == {"SG-0_minus_P0", "SG-A_minus_P0", "SG-A_minus_SG-0"}


def test_the_interval_is_the_unchanged_block_interval_and_is_widened_below_20_blocks(
    stored: dict, tmp_path: Path
) -> None:
    del stored
    result = rescore.rescore(data_dir=tmp_path, hidden_ids=IDS, n_resamples=300)
    r = result["contrasts"]["SG-A_minus_P0"]["rows"]["pooled/import"]
    assert r["units_below_minimum"] is True
    assert r["lower"] <= r["raw_lower"] <= r["upper"]  # widening never narrows
    assert r["n_units"] == 5  # 40 + 2 days of valid dates in blocks of 10 dates
    assert "fold stage" in r["note"]
    again = rescore.rescore(data_dir=tmp_path, hidden_ids=IDS, n_resamples=300)
    assert (
        again["contrasts"]["SG-A_minus_P0"]["rows"]["pooled/import"] == r
    )  # the generator is built inside the call


def test_lead_bands_use_the_12_utc_origin(stored: dict, tmp_path: Path) -> None:
    del stored
    body = rescore.rescore(data_dir=tmp_path, hidden_ids=IDS, n_resamples=50)["contrasts"][
        "SG-A_minus_P0"
    ]
    counts = {
        k.rsplit("/", 1)[1]: v["n_rows"]
        for k, v in body["lead_bands"].items()
        if k.startswith("D5/")
    }
    assert counts["24-48h"] > 0
    assert counts["48-120h"] > 0
    assert counts["120-240h"] > 0
    assert not any(k.endswith("0-24h") for k in counts)


def test_run_writes_only_the_rescore_folder_and_prints_aggregates(
    stored: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del stored
    rescore.run(data_dir=tmp_path, hidden_ids=IDS, n_resamples=50)
    folder = fit.data_root(data_dir=tmp_path) / "rescore"
    assert {p.name for p in folder.iterdir()} == {"rescore.json", "rescore.md"}
    assert {p.name for p in fit.data_root(data_dir=tmp_path).iterdir()} == {"rescore"}
    printed = capsys.readouterr().out
    assert "interim" in printed
    assert "valid_time" not in printed
    assert json.loads((folder / "rescore.json").read_text())["n_resamples"] == 50


def test_a_missing_arm_names_itself(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="P0"):
        rescore.load_arms(data_dir=tmp_path, ids=[5])
    assert core.N_RESAMPLES == 20_000
    assert core.GENERATOR_SEED == 721

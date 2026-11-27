"""Tests of stage 5, the report of experiment 734, on small synthetic folders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import e00734_report as rep
import e00734_score as score
import e00734_store as store

DAY = 48


def match_rows(
    *, arm: str, found_share: float, n_worlds: int = 12, per_world: int = 10
) -> pl.DataFrame:
    """Match rows of long, shallow edge events in `n_worlds` worlds with a given found share."""
    rows = [
        {
            "world": 900 + w,
            "site": "A",
            "kind": "edge",
            "group": i,
            "units": 4.0,
            "amp_mw": 1.0,
            "start": (3 * w + i) * 30 * DAY // 3 + 100,
            "end": -1,
            "days": 20.0,
            "edge_truth": True,
            "found": (i + w) % 10 < round(found_share * 10),
            "right_class": (i + w) % 10 < round(found_share * 10),
            "decoded_kind": "neither",
            "start_err_hh": 2.0,
            "end_err_hh": float("nan"),
            "prop_share": 0.3,
            "cascaded": False,
            "method": arm,
            "set": "default",
        }
        for w in range(n_worlds)
        for i in range(per_world)
    ]
    return pl.DataFrame(rows)


def write_set(*, folder: Path, arm: str, rows: pl.DataFrame) -> None:
    """Write per-world match files as the scoring tasks do."""
    folder.mkdir(parents=True, exist_ok=True)
    for w in rows["world"].unique().to_list():
        rows.filter(pl.col("world") == w).write_parquet(folder / f"{arm}_{w}.parquet")


def test_read_files_picks_up_only_the_files_of_the_kind_and_arm_asked_for(tmp_path: Path) -> None:
    """Test `H-nbr` does not read call tables, posterior files, or `H-nbr-noramp` files."""
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "H-nbr_900.parquet")
    pl.DataFrame({"a": [2]}).write_parquet(tmp_path / "H-nbr_901.parquet")
    pl.DataFrame({"a": [9]}).write_parquet(tmp_path / "H-nbr_900_calls.parquet")
    pl.DataFrame({"a": [8]}).write_parquet(tmp_path / "H-nbr-noramp_900.parquet")
    pl.DataFrame({"a": [7]}).write_parquet(tmp_path / "post_H-nbr_900.parquet")
    assert sorted(rep.read_files(folder=tmp_path, arm="H-nbr")["a"].to_list()) == [1, 2]
    assert rep.read_files(folder=tmp_path, arm="H-nbr", suffix="_calls")["a"].to_list() == [9]
    assert rep.read_files(folder=tmp_path, arm="H-nbr", prefix="post_")["a"].to_list() == [7]
    assert rep.read_files(folder=tmp_path, arm="PELT").height == 0


def test_calibration_covers_the_truth_when_the_band_surrounds_it() -> None:
    """Test the coverage of the 80% interval, after subtracting the uninjected median."""
    n = 6
    post = pl.DataFrame(
        {
            "world": [1] * n,
            "arm": ["H-nbr"] * n,
            "site": ["A"] * n,
            "start": [0] * n,
            "kind": ["edge"] * n,
            "block": list(range(n)),
            "truth_mw": [2.0] * n,
            **{
                c: [v + 1.0] * n
                for c, v in zip(store.Q_COLS, [0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0], strict=True)
            },
            **{c: [None] * n for c in store.S_COLS},
        },
        schema_overrides=dict.fromkeys(store.S_COLS, pl.Float64),
    )
    uninj = pl.DataFrame(
        {"site": ["A"] * n, "block": list(range(n)), "q50": [1.0] * n, "s50": [0.0] * n}
    )
    cells = pl.DataFrame(
        {"world": [1], "site": ["A"], "start": [0], "cell": ["long, shallow"], "found": [True]}
    )
    out = rep.calibration_by_cell(post=post, uninjected=uninj, rows=cells)
    row = out.row(0, named=True)
    assert row["coverage_80"] == pytest.approx(1.0)
    assert row["event_coverage_80"] == pytest.approx(1.0)
    assert row["n_blocks"] == n
    assert row["pinball_0.50"] == pytest.approx(0.0)
    narrow = post.with_columns(q10=pl.col("q10") + 5.0, q90=pl.col("q90") + 5.0)
    assert (
        rep.calibration_by_cell(post=narrow, uninjected=uninj, rows=cells)["coverage_80"][0] == 0.0
    )


def test_conservation_residual_counts_each_mirrored_signed_pair_once() -> None:
    """Test the sum of the two shifts as a share of the donor's shift."""
    calls = pl.DataFrame(
        {
            "grade": ["mirrored and returned"] * 3 + ["neither"],
            "signed_pair": [True, True, False, True],
            "amp_mw": [-2.0, 2.0, -2.0, -2.0],
            "post_mean_mw": [-2.0, 2.0, -2.0, -2.0],
            "partner_post_mean_mw": [1.5, -2.0, 2.0, 2.0],
        }
    )
    out = rep.conservation_residual(calls=calls)
    assert out["n"] == 1
    assert out["median"] == pytest.approx(-0.25)
    assert rep.conservation_residual(calls=calls.clear())["n"] == 0


def test_the_predictions_hold_or_fail_by_their_pre_registered_numbers() -> None:
    """Test each prediction's rule on a case that passes and a case that fails."""
    c1 = score.Contrast(0.12, 0.05, 0.2, 0.04, 0.21)
    good = {
        "c1": c1,
        "c2": score.Contrast(0.2, 0.1, 0.3, 0.1, 0.3),
        "p3": 0.05,
        "p4": 0.03,
        "p5": 0.06,
        "p6": {"short, shallow": 0.01, "short, deep": -0.02},
        "p7": {"long, shallow": 0.8, "medium, deep": 0.75},
        "p8": (0.01, 0.06),
        "p9": 0.4,
        "p10": (1.0, 0.95),
    }
    assert all(p["held"] is True for p in rep.evaluate_predictions(d=good))
    bad = {
        **good,
        "p3": 0.1,
        "p6": {"short, shallow": 0.05},
        "p7": {"long, shallow": 0.95},
        "p9": 0.7,
        "p10": (1.0, 1.2),
    }
    held = {p["id"]: p["held"] for p in rep.evaluate_predictions(d=bad)}
    assert [k for k, v in held.items() if v is False] == ["P3", "P6", "P7", "P9", "P10"]
    missing = {p["id"]: p["held"] for p in rep.evaluate_predictions(d={})}
    assert set(missing.values()) == {None}
    nan = rep.evaluate_predictions(d={**good, "p3": float("nan")})
    assert nan[2]["held"] is None


def test_leave_one_out_flags_a_row_that_crosses_zero_or_the_bar() -> None:
    """Test the flags against the full-sample point estimate."""
    loo = pl.DataFrame({"world": [1, 2, 3], "point": [0.12, 0.08, -0.01]})
    out = rep.leave_one_out_flags(loo=loo, point=0.12)
    assert out["flips"].to_list() == [False, True, True]
    assert out["crosses_zero"].to_list() == [False, False, True]


def test_the_scenario_table_pools_the_worlds() -> None:
    """Test the pooled error is the sum of errors over the sum of half-hours."""
    scen = pl.DataFrame(
        {
            "arm": ["H-nbr"] * 2,
            "scenario": ["all rows"] * 2,
            "n": [10, 30],
            "sum_before": [1.0, 3.0],
            "sum_after": [0.5, 1.5],
        }
    )
    out = rep.scenario_table(scen=scen)
    assert out["nmae_before"][0] == pytest.approx(0.1)
    assert out["nmae_after"][0] == pytest.approx(0.05)


def test_the_report_computes_the_verdict_from_stored_files(tmp_path: Path) -> None:
    """Test a small synthetic bank end to end: contrasts, vetoes, gates, and the verdict word."""
    scored = tmp_path / "scored" / "default"
    for arm, share in (("H-nbr", 0.8), ("H-own", 0.5), ("PELT", 0.2)):
        write_set(folder=scored, arm=arm, rows=match_rows(arm=arm, found_share=share))
    stored = rep.Stored(
        e1jsym=match_rows(arm="E1-J-sym", found_share=0.4),
        e1=match_rows(arm="E1", found_share=0.3),
    )
    (tmp_path / "real").mkdir()
    summary = {
        "H-nbr": {"rates_scored": [1.0, 0.5], "rates_event_free": [0.95, 0.48]},
        "E1-J-sym": {"rates_scored": [1.0, 0.5], "rates_event_free": [1.0, 0.5]},
    }
    (tmp_path / "real" / "summary.json").write_text(json.dumps(summary))
    res = rep.compute(out=tmp_path, stored=stored)
    assert res["d"]["c1"].point == pytest.approx(0.4)
    assert res["d"]["c2"].point == pytest.approx(0.6)
    assert res["vetoes"]["V1"]["passes"]
    assert res["verdict"].startswith("invalid")
    gates = {k: {"passed": True} for k in ("G1", "G2", "G4", "G5")}
    (tmp_path / "gates.json").write_text(json.dumps(gates))
    res = rep.compute(out=tmp_path, stored=stored)
    assert res["verdict"] == "supported"
    held = {p["id"]: p["held"] for p in res["predictions"]}
    assert held["P1"] is True
    assert held["P2"] is True
    assert held["P10"] is True
    assert res["shares"].filter(
        (pl.col("arm") == "E1-J-sym") & (pl.col("cell") == "long, shallow")
    )["found"][0] == pytest.approx(0.4)
    assert res["loo_world"].height == 12
    assert np.isfinite(res["d"]["p4"])

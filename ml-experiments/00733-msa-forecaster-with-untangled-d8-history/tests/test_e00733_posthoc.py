# ruff: noqa: PT018, E501
"""Tests of the post hoc arms YS and YO, the post hoc readings P1 to P6, and the post hoc smoke test."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import e00733_core as core
import e00733_fit as fit
import e00733_posthoc as ph
import e00733_scoring as scoring
import e00733_scoring_synth as ss
import e00733_smoke as smoke
import e00733_smokerun as smokerun
import e00733_stages as st
import e00733_synth as sy
import numpy as np
import polars as pl
import pytest
from synth_fixture import make_segment
from test_e00733_scoring import _losses
from test_e00733_scoring_synth import world_root  # noqa: F401

pytest.importorskip("torch")


# --------------------------------------------------------------------------- the post hoc arms


def test_the_b_arms_are_post_hoc_synthetic_arms_outside_the_chain() -> None:
    """Y0b, YUb, and YOb are one-seed synthetic arms that stop on the sign BCE and join no chain family."""
    for name in ("Y0b", "YUb", "YOb"):
        spec = core.ARMS[name]
        assert spec.post_hoc and spec.where == "synthetic" and spec.sign_head and not spec.counted
        assert spec.seeds == (0,) and spec.stop_on == core.STOP_ON_SIGN_BCE
        assert name not in st.FAMILIES and name not in core.FIT_ORDER
        assert name not in core.REPORT_ONLY_ARMS and name not in core.DROP_ORDER
        assert st.family_jobs(family=name) == [(name, None, 0)]
    assert core.ARMS["YUb"].channels and core.ARMS["YOb"].true_channels
    assert not core.ARMS["Y0b"].channels and not core.ARMS["Y0b"].true_channels
    assert core.POST_HOC_ARMS == ("FS", "Y0b", "YUb", "YOb")
    assert st.POSTHOC_FAMILIES == core.POST_HOC_ARMS
    assert {a for a, s_ in core.ARMS.items() if s_.post_hoc} == {"Y0b", "YUb", "YOb"}
    assert "YS" not in core.ARMS and "YO" not in core.ARMS
    assert st.family_jobs(family="FS") == [
        ("FS", f, s) for f in core.ALL_FOLDS for s in core.PLACEBO_SEEDS
    ]  # fmt: skip


def test_the_main_arms_keep_stopping_on_the_total_loss() -> None:
    """Every arm of the main chain stops on the sum of the two losses, as before."""
    assert {n for n, a in core.ARMS.items() if a.stop_on == core.STOP_ON_SIGN_BCE} == {
        "Y0b", "YUb", "YOb",
    }  # fmt: skip


def test_the_b_arm_names_do_not_collide_with_the_main_arms_files() -> None:
    """No job name, record, or prediction folder of a b-arm equals one of the main arms, in any case."""
    main = [a for a, spec in core.ARMS.items() if not spec.post_hoc]
    b_arms = [a for a, spec in core.ARMS.items() if spec.post_hoc]
    names = {
        a: [fit.job_name(arm=a, fold=f, seed=s) for f in (core.ARMS[a].folds or (None,)) for s in core.SYNTHETIC_SEEDS]
        for a in (*main, *b_arms)
    }  # fmt: skip
    main_files = {n.lower() for a in main for n in names[a]}
    b_files = [n.lower() for a in b_arms for n in names[a]]
    assert not main_files & set(b_files)
    assert len(set(b_files)) == len(b_files)
    assert "y0b_s0" in b_files and "y0_s0" in main_files
    sets = {
        st_
        for a in b_arms
        for s in core.SYNTHETIC_SEEDS
        if (st_ := ss.store.set_name(arm=a, fold=None, seed=s))
    }
    assert not sets & {
        ss.store.set_name(arm=a, fold=None, seed=s) for a in main for s in core.SYNTHETIC_SEEDS
    }


def test_the_b_arms_change_neither_the_projection_nor_the_timing_gate() -> None:
    """The projected hours count the 92 real and 9 synthetic fits of the chain only."""
    hours = core.projected_hours(seconds_per_real_fit=100.0, seconds_per_synthetic_fit=1000.0)
    assert hours == pytest.approx((92 * 100.0 + 9 * 1000.0) / 3600.0)
    decision = st.timing_decision(
        seconds_per_real_fit=288.0,
        seconds_per_synthetic_fit=1080.0,
        peak_gpu_gb=3.0,
        peak_rss_gb=8.0,
    )
    for name in ("Y0b", "YUb", "YOb"):
        assert name not in decision["dropped"] and name not in decision["not_fitted"]


def test_the_synthetic_mode_of_each_b_arm_is_that_of_its_main_arm() -> None:
    """Y0b reads no channel, YUb the untangled channels, and YOb the true sign and solar output."""
    modes = {a: fit.synthetic_mode(arm=a) for a in ("Y0", "YU", "YT", "Y0b", "YUb", "YOb")}
    assert modes == {
        "Y0": "absent", "YU": "untangled", "YT": "true", "Y0b": "absent", "YUb": "untangled",
        "YOb": "true",
    }  # fmt: skip


@pytest.mark.parametrize("arm", ["YU", "YUb", "YOb"])
def test_the_origin_time_weather_assertion_runs_for_yu_and_the_b_arms(
    arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every synthetic batch asserts that the weather is that of the origin time."""
    calls: list[int] = []
    real = sy.OriginTimes.assert_origin_time_weather

    def counted(self: sy.OriginTimes) -> int:
        calls.append(1)
        return real(self)

    monkeypatch.setattr(sy.OriginTimes, "assert_origin_time_weather", counted)
    seg = make_segment()
    idx = sy.enumerate_samples(segments=[seg])
    sy.gather_batch(
        segments=[seg], index=idx, rows=np.arange(4), d8_index=0, target_std=2.0,
        mode=fit.synthetic_mode(arm=arm),
    )  # fmt: skip
    assert calls == [1]


def _perturbed(
    seg: sy.SegmentArrays, *, first: int, last: int | None, flip: bool = True
) -> sy.SegmentArrays:
    """Return a copy of `seg` whose true sign and solar output differ on the slots `first` to `last`."""
    x, solar = seg.x_neg.copy(), seg.solar_points.copy()
    end = len(x) if last is None else last + 1
    x[first:end] = (
        np.where(np.isnan(x[first:end]), np.nan, 1.0 - x[first:end]) if flip else x[first:end]
    )
    solar[first:end] = solar[first:end] + 7.0
    return replace(seg, x_neg=x, solar_points=solar)


def test_yob_reads_the_true_channels_of_history_half_hours_before_the_origin_only() -> None:
    """Perturbing the truth at and after the origin leaves YOb's input tensors unchanged, perturbing a
    visible history half-hour changes them, and the channels are the true sign and solar output."""
    seg = make_segment(n_days=40, seed=4)
    idx = sy.enumerate_samples(segments=[seg])
    day, lead = 25, 3
    rows = np.flatnonzero((idx.origin_day == day) & (idx.lead_index == lead))
    assert len(rows) == 1
    origin = day * sy.STEPS + 24
    mode = fit.synthetic_mode(arm="YOb")

    def gather(s: sy.SegmentArrays) -> dict[str, np.ndarray]:
        return sy.gather_batch(
            segments=[s], index=idx, rows=rows, d8_index=0, target_std=2.0, mode=mode
        ).arrays

    clean = gather(seg)
    for first in (origin - 1, origin, origin + 5, origin - core.NEWEST_VISIBLE_BEFORE_CONTRACT):
        got = gather(_perturbed(seg, first=first, last=None))
        for name in ("hist_u", "hist_u_ok"):
            assert np.array_equal(clean[name], got[name]), (name, first)
    hs = sy.history_slots(origin_day=np.array([day]))[0]
    visible = sy.visible_slots(origin_day=np.array([day]), slots=hs[None])[0]
    assert (hs[visible] < origin).all(), "every channel half-hour lies before the origin"
    ok = clean["hist_u_ok"][0]
    r, j = (int(v[0]) for v in np.nonzero(ok))
    slot = int(hs[r, j])
    moved = gather(_perturbed(seg, first=slot, last=slot))
    assert not np.array_equal(clean["hist_u"], moved["hist_u"])
    diff = np.argwhere(clean["hist_u"][0] != moved["hist_u"][0])
    assert {(int(a), int(b)) for a, b, _ in diff} == {(r, j)}
    u = clean["hist_u"][0]
    assert np.array_equal(u[..., 0][ok], np.nan_to_num(seg.x_neg[hs])[ok])
    assert np.allclose(
        u[..., 1][ok], sy.solar_unit(points=seg.solar_points[hs], target_std=2.0)[ok]
    )
    assert not u[..., 2].any(), "the transfer-step channel is absent"
    assert ok.any() and np.isfinite(u).all()
    # The cut-off is exact: the slot c - 55 (stamp c - 54) is the newest visible and carries the
    # truth, and the slot c - 54 is not visible, so a cut-off moved by one half-hour fails either way.
    assert core.NEWEST_VISIBLE_BEFORE_CONTRACT == 54  # pinned: the cut-off itself must not move
    newest, first_hidden = (
        origin - core.NEWEST_VISIBLE_BEFORE_CONTRACT - 1,
        origin - core.NEWEST_VISIBLE_BEFORE_CONTRACT,
    )
    at_newest = np.argwhere(hs == newest)
    at_hidden = np.argwhere(hs == first_hidden)
    assert len(at_newest) == 1 and len(at_hidden) == 1
    rn, jn = (int(v) for v in at_newest[0])
    rh, jh = (int(v) for v in at_hidden[0])
    assert np.isfinite(seg.x_neg[newest]) and ok[rn, jn] and not ok[rh, jh]
    assert u[rn, jn, 0] == seg.x_neg[newest]
    assert np.isclose(
        u[rn, jn, 1], sy.solar_unit(points=seg.solar_points[newest : newest + 1], target_std=2.0)[0]
    )


# --------------------------------------------------------------------------- the readings


B_ARMS = ("y0b_s0", "yub_s0", "yob_s0")


def _fit_root_with_b_arms(root: Path, *, skip: tuple[str, ...] = ()) -> Path:
    """Copy YU's archive to each b-arm's place in the post hoc folder, so every b-arm equals YU."""
    fit_root = st.posthoc_fit_root(root=root)
    for arm in B_ARMS:
        if arm not in skip:
            shutil.copytree(root / "predictions" / "yu_s0", fit_root / "predictions" / arm)
    return fit_root


def test_synthetic_contrasts_read_the_b_arms_from_the_post_hoc_folder_as_point_differences(
    world_root: Path,  # noqa: F811
) -> None:
    """With every b-arm equal to YU, P3 and P4 are zero, carry the single-seed note, and have no interval."""
    fit_root = _fit_root_with_b_arms(world_root)
    got = ph.synthetic_contrasts(root=world_root, fit_root=fit_root, rows="validation")
    assert set(got) == {"P3", "P4"} and all(v["post_hoc"] is True for v in got.values())
    assert got["P3"]["treatment"] == "YOb" and got["P3"]["reference"] == "Y0b"
    assert got["P4"]["treatment"] == "YUb" and got["P4"]["reference"] == "Y0b"
    assert got["P3"]["point"] == pytest.approx(got["P4"]["point"])
    assert not {"lower", "upper"} & set(got["P3"]) and got["P3"]["seeds"] == [0]
    assert got["P3"]["note"] == "single seed, post hoc, no interval claim"
    assert "post hoc" in got["P3"]["label"] and "post hoc" in got["P4"]["label"]
    # b-arms equal YU, so the reference Y0b equals YU too and every difference is zero
    assert got["P3"]["point"] == pytest.approx(0.0) and got["P4"]["n_rows"] > 0


def test_a_missing_b_arm_leaves_only_its_contrasts_unread(world_root: Path) -> None:  # noqa: F811
    """Without the YOb archive, P3 says so and P4 is read."""
    fit_root = _fit_root_with_b_arms(world_root, skip=("yob_s0",))
    got = ph.synthetic_contrasts(root=world_root, fit_root=fit_root, rows="validation")
    assert [got[n]["status"] for n in ("P3", "P4")] == ["void", "read"]
    assert "YOb" in got["P3"]["reason"]
    none = ph.synthetic_contrasts(root=world_root, fit_root=world_root / "empty", rows="validation")
    assert {v["status"] for v in none.values()} == {"void"} and "Y0b" in none["P4"]["reason"]


def _write_record(*, root: Path, arm: str, stored: int, val_bce: list[float], stop_on: str) -> None:
    """Write a minimal synthetic fit record under `root` (the fields the table reads)."""
    st.write_json(
        path=fit.record_path(root=root, arm=arm, fold=None, seed=0),
        payload={
            "arm": arm, "stored_pass": stored, "stop_on": stop_on,
            "history": {"val": [1.0] * len(val_bce), "val_bce": val_bce},
        },
    )  # fmt: skip


def test_the_stored_pass_table_lists_each_arm_beside_the_main_runs_y0_and_yu(
    world_root: Path,  # noqa: F811
) -> None:
    """Each arm's stored pass, passes run, validation sign BCE at the stored pass, and Brier score are read."""
    fit_root = _fit_root_with_b_arms(world_root)
    for arm in ("Y0", "YU"):
        _write_record(
            root=world_root, arm=arm, stored=1, val_bce=[0.116] + [0.12] * 20, stop_on="total"
        )
    for arm in ("Y0b", "YUb", "YOb"):
        _write_record(
            root=fit_root,
            arm=arm,
            stored=4,
            val_bce=[0.3, 0.2, 0.15, 0.1, 0.12, 0.13],
            stop_on="sign_bce",
        )
    table = ph.synthetic_arm_table(root=world_root, fit_root=fit_root, rows="validation")
    assert set(table) == {"Y0", "YU", "Y0b", "YUb", "YOb"}
    assert table["YU"]["stored_pass"] == 1 and table["YU"]["passes"] == 21
    assert table["YU"]["val_sign_bce"] == pytest.approx(0.116)
    assert table["YOb"]["stored_pass"] == 4 and table["YOb"]["passes"] == 6
    assert (
        table["YOb"]["val_sign_bce"] == pytest.approx(0.1) and table["YOb"]["stop_on"] == "sign_bce"
    )
    assert 0.0 < table["Y0b"]["brier"] < 1.0 and table["Y0b"]["brier"] == pytest.approx(
        table["YUb"]["brier"]
    )
    assert all(v["post_hoc"] is True for v in table.values())
    line = ph.arm_line(arm="YOb", entry=table["YOb"])
    assert (
        "stored pass 4 of 6 passes" in line
        and "sign cross-entropy alone" in line
        and "post hoc" in line
    )


def test_real_contrasts_read_fs_from_the_post_hoc_folder_and_the_rest_from_the_main_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1 is FS minus FU and P2 is FS minus M* (F0r on validation rows), on D8, with the MAE."""
    import e00733_baselines as baselines

    root, fit_root = tmp_path / "main", tmp_path / "main" / "posthoc" / "main"
    offsets = {"FS": -0.2, "FU": -0.1, "F0r": 0.0}
    seen: list[tuple[str, Path]] = []

    def arm_losses(*, root: Path, arm: tuple[str, str | None], **_: Any) -> pl.DataFrame:
        seen.append((arm[0], root))
        return _losses(n=40, offset=offsets[arm[0]])

    monkeypatch.setattr(scoring, "arm_losses", arm_losses)
    monkeypatch.setattr(scoring, "has_job", lambda **_: True)
    monkeypatch.setattr(scoring, "validation_truth", lambda **_: None)
    monkeypatch.setattr(baselines, "d8_series_id", lambda: 26)
    got = ph.real_contrasts(root=root, fit_root=fit_root, rows="validation", n_resamples=50)
    assert set(got) == {"P1", "P2"} and all(v["post_hoc"] is True for v in got.values())
    assert got["P1"]["reading"]["point"] == pytest.approx(-0.1)
    assert got["P2"]["reading"]["point"] == pytest.approx(-0.2)
    assert got["P1"]["mae_of_median_difference"] == pytest.approx(-0.1)
    assert got["P2"]["reference"] == "F0r" and got["P1"]["interim"] is True
    assert {(a, r) for a, r in seen} == {("FS", fit_root), ("FU", root), ("F0r", root)}
    assert got["P1"]["reading"]["widening"] >= 1.0


def test_real_contrasts_are_unread_when_a_forecast_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FS forecasts that do not exist leave P1 and P2 unread."""
    import e00733_baselines as baselines

    monkeypatch.setattr(scoring, "has_job", lambda **kw: kw["arm"][0] != "FS")
    monkeypatch.setattr(scoring, "validation_truth", lambda **_: None)
    monkeypatch.setattr(baselines, "d8_series_id", lambda: 26)
    monkeypatch.setattr("e00733_scoring._partial_plan", lambda **_: None)
    got = ph.real_contrasts(
        root=tmp_path, fit_root=tmp_path / "p", rows="validation", n_resamples=50
    )
    assert {v["status"] for v in got.values()} == {"void"}


def _canned(*, rows: str = "test") -> dict[str, Any]:
    real = lambda name, p: {  # noqa: E731
        "post_hoc": True, "label": ph.contrast_label(name=name), "status": "read",
        "treatment": "FS", "reference": "FU" if name == "P1" else "M*", "metric": "crps",
        "reading": {"relative": {"point": p, "lower": p - 0.01, "upper": p + 0.01}, "n_rows": 1234},
        "mae_of_median_relative": p / 2, "interim": False,
    }  # fmt: skip
    syn = lambda name, p: {  # noqa: E731
        "post_hoc": True, "label": ph.contrast_label(name=name), "status": "read",
        "treatment": "YOb" if name == "P3" else "YUb", "reference": "Y0b", "metric": "brier",
        "point": p, "n_rows": 999, "seeds": [0], "note": ph.SINGLE_SEED_NOTE,
    }  # fmt: skip
    arm = lambda stop_on, stored, passes: {  # noqa: E731
        "post_hoc": True, "stop_on": stop_on, "stored_pass": stored, "passes": passes,
        "val_sign_bce": 0.117, "brier": 0.0391,
    }  # fmt: skip
    return {
        "post_hoc": True, "rows": rows, "level": core.LEVEL,
        "contrasts": {"P1": real("P1", 0.02), "P2": real("P2", -0.01), "P3": syn("P3", -0.002), "P4": syn("P4", 0.001)},
        "synthetic_arms": {"Y0": arm("total", 1, 21), "YU": arm("total", 1, 21), "Y0b": arm("sign_bce", 6, 26), "YUb": arm("sign_bce", 9, 29), "YOb": arm("sign_bce", 12, 32)},
        "cost": {"main_chain": 7.2, "per_arm": {"FS": 1.5, "Y0b": 0.6, "YUb": 0.6, "YOb": 0.6}, "posthoc": 3.3, "running_total": 10.5},
    }  # fmt: skip


def test_the_report_is_short_plain_and_labels_every_number_post_hoc() -> None:
    """At most 30 lines, the verdict stays, and each line that holds a digit says post hoc."""
    text = ph.render_report(readings=_canned(), verdict_row=7)
    lines = text.splitlines()
    assert len(lines) <= 30
    assert "after seeing the verdict" in text and "The verdict stays row 7 whatever" in text
    assert "dropped FS" in text and "a lead for a follow-up, not a finding" in text
    assert "stored pass 1" in text and "all six synthetic fits" in text.lower().replace("\n", " ")
    for line in lines:
        if any(ch.isdigit() for ch in line) and "verdict stays" not in line:
            assert "post hoc" in line.lower(), line
    assert "post hoc fits cost 3.30 GPU-process-hours" in text
    assert "FS 1.50, Y0b 0.60, YUb 0.60, YOb 0.60" in text
    assert "running total is 10.50, against the stop line of 12" in text
    assert all(f"P{i} (post hoc)" in text for i in range(1, 5)) and "P5" not in text
    assert text.count("single seed, post hoc, no interval claim") == 2
    p3_line = next(line for line in lines if line.startswith("P3 (post hoc)"))
    assert "significant" not in text.lower() and "interval (" not in p3_line
    unread = _canned()
    unread["contrasts"]["P1"] = ph.unread(
        name="P1", reason="a stored forecast of the plan is missing"
    )
    assert "P1 (post hoc) was not read" in ph.render_report(readings=unread, verdict_row=None)


def test_the_figure_says_post_hoc_in_every_label_and_the_title() -> None:
    """Each contrast label and the title carry the words post hoc, and the synthetic dots have no interval."""
    data = ph.figure_rows(readings=_canned())
    assert data.height == 4 and all("post hoc" in x for x in data["label"].to_list())
    synthetic = data.filter(pl.col("panel") == "synthetic")
    assert synthetic.height == 2 and synthetic["lower"].null_count() == 2
    spec = json.dumps(ph.contrast_figure(readings=_canned()).to_dict())
    assert (
        spec.count("post hoc") >= 4
        and "Post hoc contrasts P1 to P4 of FS, Y0b, YUb, and YOb" in spec
    )
    assert f"{100 * core.LEVEL:.2f}% interval" in spec


def test_the_stage_writes_three_files_under_posthoc_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The readings, the report, and the figure land in `posthoc/`; the main run is not touched."""
    root = tmp_path / "ml" / "00733-x"
    (root / "scores").mkdir(parents=True)
    (root / "scores" / "verdict.json").write_text('{"row": 7}')
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(ph, "posthoc_readings", lambda **_: _canned(rows="validation"))
    args = argparse.Namespace(validation_only=True, data_root=str(root))
    ph.stage_posthoc_score(args, root)
    assert {p.name for p in (root / "posthoc").iterdir()} == {
        "readings.json", "report.md", "posthoc_contrasts.svg",
    }  # fmt: skip
    assert {p: p.read_bytes() for p in before} == before
    assert "post hoc" in (root / "posthoc" / "posthoc_contrasts.svg").read_text().lower()
    assert json.loads((root / "posthoc" / "readings.json").read_text())["post_hoc"] is True
    assert "row 7" in (root / "posthoc" / "report.md").read_text()
    with pytest.raises(RuntimeError):
        ph.stage_posthoc_score(argparse.Namespace(validation_only=True, data_root=None), root)


# --------------------------------------------------------------------------- the smoke test


def test_the_post_hoc_smoke_takes_only_the_post_hoc_arms() -> None:
    """`--arms FS,Y0b,YUb,YOb` is accepted, and an arm of the chain or an unknown arm is refused."""
    assert smokerun.parse_arms(text="FS,Y0b,YUb,YOb") == ("FS", "Y0b", "YUb", "YOb")
    assert smokerun.parse_arms(text="YOb") == ("YOb",)
    for bad in ("FU", "FS,YU", "", "Z", "YT", "YS", "YO"):
        with pytest.raises(smokerun.rt.RuntimeError733):
            smokerun.parse_arms(text=bad)


def test_the_post_hoc_smoke_writes_its_record_inside_the_scratch_folder_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record `scores/smoke_posthoc.json` is in the scratch folder; the data folder gets no file."""
    import e00733_predictions as pr

    root = tmp_path / "data" / "00733-x"
    (root / "scores").mkdir(parents=True)
    ran: list[tuple[str, str | None]] = []

    def fit_one(arm: str, fold: str | None, seed: int) -> dict:
        ran.append((arm, fold))
        return {"stored_pass": 1}

    monkeypatch.setattr(smokerun, "link_inputs", lambda **_: [])
    monkeypatch.setattr(smokerun.rt, "make_run_job", lambda **_: fit_one)
    monkeypatch.setattr(pr, "predict_job", lambda **kw: [tmp_path / f"{kw['arm']}.out"])
    monkeypatch.setattr(smokerun, "read_stored_step", lambda **_: lambda arm, path: 5)
    before = sorted(p for p in root.rglob("*"))
    args = argparse.Namespace(
        scratch=str(tmp_path / "scratch"), device="cpu", arms="FS,Y0b,YUb,YOb"
    )
    record = smokerun.stage_smoke(args, root)
    assert (
        record["pass"] is True
        and record["post_hoc"] is True
        and record["arms"] == ["FS", "Y0b", "YUb", "YOb"]
    )
    assert ran == [("FS", core.ARMS["FS"].folds[0]), ("Y0b", None), ("YUb", None), ("YOb", None)]
    written = tmp_path / "scratch" / root.name / smokerun.POSTHOC_SMOKE_FILE
    assert json.loads(written.read_text())["pass"] is True
    assert sorted(root.rglob("*")) == before
    assert (
        not (root / st.SMOKE_FILE).exists()
        and not (root / st.SMOKE_FILE.replace("smoke", "smoke_cpu")).exists()
    )


def test_the_main_smoke_still_runs_every_chain_arm() -> None:
    """The chain's smoke test covers the arms of `FIT_ORDER`, and no post hoc arm joins it."""
    assert tuple(a for a in core.FIT_ORDER if a != "FU-screen") == smoke.SMOKE_ARMS
    assert not set(smoke.SMOKE_ARMS) & {a for a, s in core.ARMS.items() if s.post_hoc}


# --------------------------------------------------------------------------- a stopped run is read as far as it went


def _copy_archive(*, src_root: Path, src_arm: str, dst_root: Path, dst_arm: str, seed: int) -> None:
    """Copy the validation archive of `src_arm` seed 0 to the test archive of `dst_arm` at `seed`."""
    dst = ss.archive_path(root=dst_root, arm=dst_arm, seed=seed, rows="test")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ss.archive_path(root=src_root, arm=src_arm, seed=0, rows="validation"), dst)


def test_a_stopped_run_is_read_as_far_as_it_went(
    world_root: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the status `stopped` and YOb missing, P3 is void and names YOb, and the others are computed."""
    fit_root = st.posthoc_fit_root(root=world_root)
    for arm in ("Y0", "YU"):
        _copy_archive(src_root=world_root, src_arm=arm, dst_root=world_root, dst_arm=arm, seed=0)
    for arm in ("Y0b", "YUb"):
        _copy_archive(src_root=world_root, src_arm="YU", dst_root=fit_root, dst_arm=arm, seed=0)
    st.write_posthoc_status(
        root=world_root,
        state="stopped",
        reason=st.COST_CAP_MESSAGE,
        skipped=[{"job": "YOb_s0"}],
        failed=[],
    )
    canned = _canned()["contrasts"]
    monkeypatch.setattr(ph, "real_contrasts", lambda **_: {"P1": canned["P1"], "P2": canned["P2"]})
    readings = ph.posthoc_readings(root=world_root, rows="test")
    status = {n: e["status"] for n, e in readings["contrasts"].items()}
    assert status == {"P1": "read", "P2": "read", "P3": "void", "P4": "read"}
    assert "YOb holds no archive of seed 0" in readings["contrasts"]["P3"]["reason"]
    assert readings["post_hoc_run"]["state"] == "stopped"
    text = ph.render_report(readings=readings, verdict_row=7)
    assert "ended stopped" in text and "P3 (post hoc) was not read: YOb holds no archive" in text
    assert len(text.splitlines()) <= 30


def test_the_scratch_folder_of_the_post_hoc_smoke_may_not_lie_in_or_contain_the_data_folder(
    tmp_path: Path,
) -> None:
    """`smoke --arms` fits after the first test score, so it refuses a scratch inside the data folder."""
    root = tmp_path / "data" / "00733-x"
    (root / "scores").mkdir(parents=True)
    for scratch in (root / "scratch", root, tmp_path / "data", tmp_path):
        args = argparse.Namespace(scratch=str(scratch), device="cpu", arms="YOb")
        with pytest.raises(st.GuardError):
            smokerun.stage_smoke_arms(args, root)
    assert not (root / "scratch").exists()

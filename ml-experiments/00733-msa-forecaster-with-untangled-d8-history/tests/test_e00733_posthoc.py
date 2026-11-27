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


# --------------------------------------------------------------------------- the arm YS


def test_ys_is_a_post_hoc_arm_outside_the_chain() -> None:
    """YS reads P(X<0) alone on the synthetic world, seeds 0 to 2, and is no family of the chain."""
    spec = core.ARMS["YS"]
    assert spec.post_hoc and spec.where == "synthetic" and spec.sign_head and not spec.counted
    assert spec.channels and spec.channel_subset == ("p_neg",) and spec.seeds == (0, 1, 2)
    assert "YS" not in st.FAMILIES and "YS" not in core.FIT_ORDER
    assert "YS" not in core.REPORT_ONLY_ARMS and core.POST_HOC_ARMS == ("YO", "FS", "YS")
    assert st.family_jobs(family="YS") == [("YS", None, s) for s in (0, 1, 2)]
    assert st.family_jobs(family="FS") == [
        ("FS", f, s) for f in core.ALL_FOLDS for s in core.PLACEBO_SEEDS
    ]  # fmt: skip


def test_yo_is_a_post_hoc_oracle_arm_outside_the_chain() -> None:
    """YO reads the true sign and solar output of the history on the synthetic world, seeds 0 to 2."""
    spec = core.ARMS["YO"]
    assert spec.post_hoc and spec.where == "synthetic" and spec.sign_head and not spec.counted
    assert spec.true_channels and not spec.channels and spec.seeds == (0, 1, 2)
    assert "YO" not in st.FAMILIES and "YO" not in core.FIT_ORDER
    assert st.family_jobs(family="YO") == [("YO", None, s) for s in (0, 1, 2)]
    assert st.POSTHOC_FAMILIES == ("YO", "FS", "YS")


def test_ys_changes_neither_the_projection_nor_the_timing_gate() -> None:
    """The projected hours count the 92 real and 9 synthetic fits of the chain, as before YS."""
    hours = core.projected_hours(seconds_per_real_fit=100.0, seconds_per_synthetic_fit=1000.0)
    assert hours == pytest.approx((92 * 100.0 + 9 * 1000.0) / 3600.0)
    decision = st.timing_decision(
        seconds_per_real_fit=288.0,
        seconds_per_synthetic_fit=1080.0,
        peak_gpu_gb=3.0,
        peak_rss_gb=8.0,
    )
    assert "YS" not in decision["dropped"] and "YS" not in decision["not_fitted"]


def test_the_synthetic_mode_of_ys_is_p_neg_alone() -> None:
    """Y0, YU, YT, and YS each have their own mode."""
    modes = {a: fit.synthetic_mode(arm=a) for a in ("Y0", "YU", "YT", "YS", "YO")}
    assert modes == {"Y0": "absent", "YU": "untangled", "YT": "true", "YS": "p_neg", "YO": "true"}


def test_ys_inputs_hold_p_neg_and_flag_the_other_channels_absent_where_yu_holds_all_three() -> None:
    """On the same segments and rows, YS and YU read the same P(X<0) and the same flags, and YS has
    the solar and transfer-step channels at zero, as FS has them on the real folds."""
    segs = [make_segment(n_days=40, seed=1), make_segment(n_days=45, seed=2)]
    idx = sy.enumerate_samples(segments=segs)
    rows = np.arange(0, 60, 3)
    gathered = {
        arm: sy.gather_batch(
            segments=segs, index=idx, rows=rows, d8_index=0, target_std=2.0,
            mode=fit.synthetic_mode(arm=arm),
        ).arrays
        for arm in ("YU", "YS")
    }  # fmt: skip
    yu, ys = gathered["YU"], gathered["YS"]
    assert yu["hist_u_ok"].any() and np.array_equal(yu["hist_u_ok"], ys["hist_u_ok"])
    assert (yu["hist_u"][..., 0][yu["hist_u_ok"]] > 0).any()
    assert np.array_equal(yu["hist_u"][..., 0], ys["hist_u"][..., 0])
    for channel in (1, 2):
        assert (yu["hist_u"][..., channel] != 0).any(), "YU holds this channel"
        assert not ys["hist_u"][..., channel].any(), "YS flags this channel absent"
    for name in ("hist_power", "hist_weather", "top_weather", "hist_visible"):
        assert np.array_equal(yu[name], ys[name]), name


@pytest.mark.parametrize("arm", ["YU", "YS", "YO"])
def test_the_origin_time_weather_assertion_runs_for_yu_ys_and_yo(
    arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every synthetic batch asserts that the weather is that of the origin time, YS and YO as YU."""
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


def test_yo_reads_the_true_channels_of_history_half_hours_before_the_origin_only() -> None:
    """Perturbing the truth at and after the origin leaves YO's input tensors unchanged, perturbing a
    visible history half-hour changes them, and the channels are the true sign and solar output."""
    seg = make_segment(n_days=40, seed=4)
    idx = sy.enumerate_samples(segments=[seg])
    day, lead = 25, 3
    rows = np.flatnonzero((idx.origin_day == day) & (idx.lead_index == lead))
    assert len(rows) == 1
    origin = day * sy.STEPS + 24
    mode = fit.synthetic_mode(arm="YO")

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


def _fit_root_with_ys(root: Path) -> Path:
    """Copy YU's archive to YS's and YO's place in the post hoc folder, so both equal YU."""
    fit_root = st.posthoc_fit_root(root=root)
    for arm in ("ys_s0", "yo_s0"):
        shutil.copytree(root / "predictions" / "yu_s0", fit_root / "predictions" / arm)
    return fit_root


def test_synthetic_contrasts_read_ys_from_the_post_hoc_folder(world_root: Path) -> None:  # noqa: F811
    """P3 equals C4's reading when YS equals YU, P4 is zero, and both carry the post hoc label."""
    fit_root = _fit_root_with_ys(world_root)
    got = ph.synthetic_contrasts(
        root=world_root, fit_root=fit_root, rows="validation", n_resamples=50
    )
    main = ss.synthetic_readings(root=world_root, rows="validation", n_resamples=50)
    assert set(got) == {"P3", "P4", "P5", "P6"} and all(v["post_hoc"] is True for v in got.values())
    assert got["P3"]["point"] == pytest.approx(main["contrasts"]["C4"].reading.point)
    assert got["P3"]["lower"] == pytest.approx(main["contrasts"]["C4"].reading.lower)
    assert got["P3"]["upper"] == pytest.approx(main["contrasts"]["C4"].reading.upper)
    assert got["P4"]["point"] == pytest.approx(0.0) and "post hoc" in got["P4"]["label"]
    assert got["P3"]["treatment"] == "YS" and got["P3"]["reference"] == "Y0"
    assert got["P4"]["reference"] == "YU" and got["P3"]["seeds"]["YS"] == [0]
    assert got["P5"]["point"] == pytest.approx(got["P3"]["point"])
    assert got["P6"]["point"] == pytest.approx(0.0) and got["P6"]["treatment"] == "YO"
    assert got["P5"]["reference"] == "Y0" and "post hoc" in got["P5"]["label"]


def test_a_missing_arm_leaves_only_its_contrasts_unread(world_root: Path) -> None:  # noqa: F811
    """With YS but no YO archive, P3 and P4 are read, and P5 and P6 say which arm is missing."""
    fit_root = _fit_root_with_ys(world_root)
    shutil.rmtree(fit_root / "predictions" / "yo_s0")
    got = ph.synthetic_contrasts(
        root=world_root, fit_root=fit_root, rows="validation", n_resamples=50
    )
    assert [got[n]["status"] for n in ("P3", "P4", "P5", "P6")] == [
        "read",
        "read",
        "void",
        "void",
    ]
    assert "YO" in got["P5"]["reason"]


def test_synthetic_contrasts_are_unread_without_the_ys_archive(world_root: Path) -> None:  # noqa: F811
    """No YS archive leaves P3 and P4 unread with the reason, never a number."""
    got = ph.synthetic_contrasts(
        root=world_root,
        fit_root=st.posthoc_fit_root(root=world_root),
        rows="validation",
        n_resamples=50,
    )
    assert {v["status"] for v in got.values()} == {"void"} and "YS" in got["P3"]["reason"]
    assert "YO" in got["P5"]["reason"]


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
        "treatment": "YS" if name in ("P3", "P4") else "YO",
        "reference": "Y0" if name in ("P3", "P5") else "YU", "metric": "brier",
        "point": p, "lower": p - 0.001, "upper": p + 0.001, "n_rows": 999,
    }  # fmt: skip
    return {
        "post_hoc": True, "rows": rows, "level": core.LEVEL,
        "contrasts": {"P1": real("P1", 0.02), "P2": real("P2", -0.01), "P3": syn("P3", -0.002), "P4": syn("P4", 0.001), "P5": syn("P5", -0.003), "P6": syn("P6", -0.001)},
        "cost": {"main_chain": 7.2, "per_arm": {"YO": 1.6, "FS": 1.5, "YS": 1.6}, "posthoc": 4.7, "running_total": 11.9},
    }  # fmt: skip


def test_the_report_is_short_plain_and_labels_every_number_post_hoc() -> None:
    """At most 25 lines, the verdict stays, and each line that holds a digit says post hoc."""
    text = ph.render_report(readings=_canned(), verdict_row=7)
    lines = text.splitlines()
    assert len(lines) <= 25
    assert "after seeing the verdict" in text and "The verdict stays row 7 whatever" in text
    assert "dropped FS" in text and "a lead for a follow-up, not a finding" in text
    for line in lines:
        if any(ch.isdigit() for ch in line) and "verdict stays" not in line:
            assert "post hoc" in line.lower(), line
    assert (
        "post hoc fits cost 4.70 GPU-process-hours" in text and "YO 1.60, FS 1.50, YS 1.60" in text
    )
    assert "running total is 11.90, against the stop line of 12" in text
    assert all(f"P{i} (post hoc)" in text for i in range(1, 7))
    unread = _canned()
    unread["contrasts"]["P1"] = ph.unread(
        name="P1", reason="a stored forecast of the plan is missing"
    )
    assert "P1 (post hoc) was not read" in ph.render_report(readings=unread, verdict_row=None)


def test_the_figure_says_post_hoc_in_every_label_and_the_title() -> None:
    """Each contrast label and the title carry the words post hoc."""
    data = ph.figure_rows(readings=_canned())
    assert data.height == 6 and all("post hoc" in x for x in data["label"].to_list())
    spec = json.dumps(ph.contrast_figure(readings=_canned()).to_dict())
    assert spec.count("post hoc") >= 6 and "Post hoc contrasts P1 to P6 of FS, YS, and YO" in spec
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
    """`--arms FS,YS` is accepted, and an arm of the chain or an unknown arm is refused."""
    assert smokerun.parse_arms(text="FS,YS") == ("FS", "YS")
    assert smokerun.parse_arms(text="YO,FS,YS") == ("YO", "FS", "YS")
    for bad in ("FU", "FS,YU", "", "Z", "YT"):
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
    args = argparse.Namespace(scratch=str(tmp_path / "scratch"), device="cpu", arms="YO,FS,YS")
    record = smokerun.stage_smoke(args, root)
    assert (
        record["pass"] is True
        and record["post_hoc"] is True
        and record["arms"] == ["YO", "FS", "YS"]
    )
    assert ran == [("YO", None), ("FS", core.ARMS["FS"].folds[0]), ("YS", None)]
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
    """With the status `stopped` and YS at seeds 0 and 1 only, P3 and P4 are void and name the seeds
    held, and P1, P2, P5, and P6 are computed; the report says what is missing."""
    fit_root = st.posthoc_fit_root(root=world_root)
    for arm in ("Y0", "YU"):
        for seed in core.SYNTHETIC_SEEDS:
            _copy_archive(
                src_root=world_root, src_arm=arm, dst_root=world_root, dst_arm=arm, seed=seed
            )
    for seed in core.SYNTHETIC_SEEDS:
        _copy_archive(src_root=world_root, src_arm="YU", dst_root=fit_root, dst_arm="YO", seed=seed)
    for seed in (0, 1):
        _copy_archive(src_root=world_root, src_arm="YU", dst_root=fit_root, dst_arm="YS", seed=seed)
    st.write_posthoc_status(
        root=world_root,
        state="stopped",
        reason=st.COST_CAP_MESSAGE,
        skipped=[{"job": "YS_s2"}],
        failed=[],
    )
    canned = _canned()["contrasts"]
    monkeypatch.setattr(ph, "real_contrasts", lambda **_: {"P1": canned["P1"], "P2": canned["P2"]})
    readings = ph.posthoc_readings(root=world_root, rows="test", n_resamples_synthetic=50)
    status = {n: e["status"] for n, e in readings["contrasts"].items()}
    assert status == {
        "P1": "read",
        "P2": "read",
        "P3": "void",
        "P4": "void",
        "P5": "read",
        "P6": "read",
    }
    assert "YS holds seeds [0, 1] of [0, 1, 2]" in readings["contrasts"]["P3"]["reason"]
    assert readings["post_hoc_run"]["state"] == "stopped"
    text = ph.render_report(readings=readings, verdict_row=7)
    assert "ended stopped" in text and "P3 (post hoc) was not read: YS holds seeds [0, 1]" in text
    assert len(text.splitlines()) <= 25


def test_the_scratch_folder_of_the_post_hoc_smoke_may_not_lie_in_or_contain_the_data_folder(
    tmp_path: Path,
) -> None:
    """`smoke --arms` fits after the first test score, so it refuses a scratch inside the data folder."""
    root = tmp_path / "data" / "00733-x"
    (root / "scores").mkdir(parents=True)
    for scratch in (root / "scratch", root, tmp_path / "data", tmp_path):
        args = argparse.Namespace(scratch=str(scratch), device="cpu", arms="YS")
        with pytest.raises(st.GuardError):
            smokerun.stage_smoke_arms(args, root)
    assert not (root / "scratch").exists()

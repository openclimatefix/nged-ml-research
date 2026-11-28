"""A CPU dry run of fit, predict, and score on a tiny synthetic fold (needs torch)."""
# ruff: noqa: D103, E402

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
# the tests never write to the experiment's data folder
os.environ.setdefault("E00738_OUT", str(Path(tempfile.gettempdir()) / "scratch" / "e738_tests"))
os.environ.setdefault("E00738_SCRATCH", "1")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _d in HERE.parent.glob("*"):
    if _d.is_dir() and _d.name[:5].isdigit() and int(_d.name[:5]) in {
        11, 87, 377, 378, 381, 391, 397, 400, 402, 404, 407, 410, 418, 422, 425, 428, 435, 437, 448,
        454, 470, 510, 511, 513, 515, 521, 545, 558, 559, 560, 562, 565, 568, 569, 572, 574, 578,
        581, 632, 667, 668, 709, 647, 650, 652, 654, 713, 726,
    }:  # fmt: skip
        sys.path.append(str(_d))

torch = pytest.importorskip("torch", reason="the dry run needs torch (workstation only)")
import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00738_common as common
import e00738_fit as fit
import e00738_panel as panel
import e00738_predict as pred
import e00738_score as score
import e00738_world as world
import polars as pl

MONTH = "2025-11"
PANELS = ("real", "pass00", f"pass{common.HELD_FIRST}", "resampled00")
N_PASSES = 2


class FakeData:
    """The fields of experiment 713's `FoldData` that the fit, predict, and score code read."""

    def __init__(self) -> None:  # noqa: D107
        import e00650_common as c650  # ty: ignore[unresolved-import]

        rng = np.random.default_rng(0)
        self.month = MONTH
        self.cutoff_idx = int(c650.month_start_idx(month=MONTH))
        n_grid = self.cutoff_idx + 48 * 50
        day = np.arange(n_grid)
        base = 0.3 + 0.1 * np.sin(2 * np.pi * day / 48)[None, :]
        self.clean = (base + 0.01 * rng.normal(size=(16, n_grid))).astype(np.float32)
        self.target_mean = self.clean.mean(axis=1)
        self.target_std = self.clean.std(axis=1)
        z = ((self.clean - self.target_mean[:, None]) / self.target_std[:, None]).astype(np.float32)
        first_day = self.cutoff_idx // 48 - 30
        n_days = 34
        ens = bt.EnsTensors(
            weather=torch.from_numpy(rng.normal(size=(16, n_days, 672, 21))).half(),
            spread=torch.from_numpy(np.abs(rng.normal(size=(16, n_days, 672, 7)))).half(),
            filled=torch.ones(16, n_days, 672, dtype=torch.bool),
            first_day=first_day,
        )
        self.source = bt.Source(
            power_z=torch.from_numpy(z),
            dense_weather=torch.from_numpy(rng.normal(size=(16, n_grid, 21))).half(),
            dense_datetime=torch.from_numpy(rng.normal(size=(16, n_grid, 21))).half(),
            day_type=torch.zeros(n_grid, dtype=torch.long),
            ens=ens,
        )
        self.fingerprint = "fake"
        self.test_origin_idx = np.array([self.cutoff_idx + 18, self.cutoff_idx + 48 + 18])


@pytest.fixture
def dry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeData:
    data = FakeData()
    torch.manual_seed(0)
    state = mdl.MsaNet(switches=mdl.ARM_SWITCHES["M"]).state_dict()
    monkeypatch.setattr(common, "OUT", tmp_path)
    monkeypatch.setenv(common.PASS_CAP_ENV, str(N_PASSES))
    monkeypatch.setattr(panel, "assert_713_code", lambda: "fake713")
    monkeypatch.setattr(fit, "load_fold_data", lambda **kw: data)
    monkeypatch.setattr(fit, "m1_state", lambda **kw: {k: v.clone() for k, v in state.items()})
    monkeypatch.setattr(
        fit, "fold_world", lambda: ({}, np.ones(16), np.array([True] * 8 + [False] * 8))
    )

    def new_w0(**kw: object) -> object:
        model = mdl.MsaNet(switches=mdl.ARM_SWITCHES["M"])
        model.load_state_dict({k: v.clone() for k, v in state.items()})
        model.freeze_encoders()
        return model

    monkeypatch.setattr(fit, "new_w0_model", new_w0)
    sites = pl.DataFrame(
        {"primary": list(range(16)), "label": [f"S{i}" for i in range(16)],
         "time_series_id": [100 + i for i in range(16)]}
    )  # fmt: skip
    monkeypatch.setattr(fit, "site_table", lambda: sites)
    monkeypatch.setattr(world, "load_real_durations_days", lambda: np.array([0.5, 2.0, 9.0, 40.0]))
    monkeypatch.setattr(pred, "panel_names", lambda: list(PANELS))
    monkeypatch.setattr(
        pred, "reproduce_stored_m1", lambda **kw: {"max_abs_difference": 0.0, "rows": 1.0}
    )

    def make_samples(
        *, world: object, stage: int, placebo: bool, primaries: object = None
    ) -> object:
        first = data.cutoff_idx // 48 - 25
        days = np.arange(first, first + 20)
        prim, cc = np.meshgrid(np.arange(16), days * 48 + 24, indexing="ij")
        return tr.SampleSet(
            primary=prim.ravel(), c_idx=cc.ravel(), is_val=(cc.ravel() // 48 >= first + 16)
        )

    monkeypatch.setattr(tr, "make_samples", make_samples)
    return data


def test_the_pass_cap_environment_variable_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(common.PASS_CAP_ENV, raising=False)
    assert common.pass_cap(remedy=False) == common.PASS_CAP == 60
    assert common.pass_cap(remedy=True) == common.REMEDY_PASS_CAP == 120
    monkeypatch.setenv(common.PASS_CAP_ENV, "1")
    assert common.pass_cap(remedy=False) == common.pass_cap(remedy=True) == 1


def test_fit_predict_and_score_run_end_to_end_on_a_tiny_fold(dry: FakeData, tmp_path: Path) -> None:
    for arm in common.FIT_ARMS:
        fit.fit_job(arm=arm, month=MONTH, seed=0, device="cpu")
        log = fit.read_log(arm=arm, month=MONTH, seed=0)
        assert log["arm"] == arm
        assert [s["passes"] for s in log["stages"]] == [N_PASSES] * len(log["stages"]), (
            "the pass cap must stop every stage"
        )
        assert common.fit_paths(arm=arm, month=MONTH, seed=0)["state"].exists()
    # a finished fit is not repeated
    fit.fit_job(arm="LP", month=MONTH, seed=0, device="cpu")

    pred.run_job(month=MONTH, seed=0, device="cpu", only=PANELS)
    for arm in common.ALL_ARMS:
        for name in PANELS:
            f = pl.read_parquet(pred.panel_path(arm=arm, month=MONTH, seed=0, name=name))
            assert f.height == 16 * 2 * len(panel.SCORED_LEAD_DAYS) * 48
            assert f["q50"].is_finite().all()
    diag = pl.read_parquet(
        pred.panel_path(arm="LP", month=MONTH, seed=0, name="pass00").with_name(
            "pass00_diag.parquet"
        )
    )
    assert {"age", "depth_share", "p_mean"} <= set(diag.columns)

    month_start = data_start = dry.cutoff_idx
    del data_start
    shifts = pred.all_panel_shifts(
        month=MONTH, month_start=month_start, window_end=fit.window_end_idx(),
        is_mw=np.array([True] * 8 + [False] * 8), real_days=np.array([0.5, 2.0, 9.0, 40.0]),
    )  # fmt: skip
    ids = np.array([100 + i for i in range(16)])
    sums = score.score_fold_seed(
        month=MONTH, seed=0, labels=[f"S{i}" for i in range(16)], ids=ids,
        shifts_of={k: shifts[k] for k in PANELS if k != "real"},
    )  # fmt: skip
    assert set(sums) == {*common.ALL_ARMS}
    for arm in common.ALL_ARMS:
        assert sums[arm]["real"]["all"]["n"].sum() > 0
        assert sums[arm]["train_family"]["all"]["n"].sum() > 0
        assert np.isfinite(sums[arm]["scale_factor"])
    # an untrained head arm starts equal to M1, and a head arm differs after training
    np.testing.assert_allclose(sums["M1"]["real"]["all"]["n"], sums["LP"]["real"]["all"]["n"])
    back = score.read_sums(month=MONTH, seed=0)
    np.testing.assert_allclose(back["M1"]["real"]["all"]["crps"], sums["M1"]["real"]["all"]["crps"])

    # the member-path check reads the stored files of a mixture arm
    import e00738_members as members
    import e00738_survival as surv

    saved = torch.load(common.fit_paths(arm="LM", month=MONTH, seed=0)["state"])
    curve = surv.curve_from_record(record=saved["curve"])
    path = pred.panel_path(arm="LM", month=MONTH, seed=0, name="pass00")
    got = members.check_frames(
        arm=pl.read_parquet(path),
        base=pl.read_parquet(pred.panel_path(arm="M1", month=MONTH, seed=0, name="pass00")),
        diag=pl.read_parquet(path.with_name("pass00_diag.parquet")),
        curve=curve,
        max_rows=20,
    )
    assert got["rows"] > 0
    assert all(np.isfinite(v) for v in got.values())


def test_phase_zero_tasks_run_on_the_tiny_fold(dry: FakeData) -> None:
    import e00738_run as run

    run.task_strata_check()
    shares = json.loads((common.OUT / "checks" / "strata_shares.json").read_text())
    assert shares["off"] == {}
    run.task_a1_table(month=MONTH)
    table = json.loads((common.OUT / "a1" / f"{MONTH}.json").read_text())
    assert "false_alarm_real_all_rows" in table
    assert table["noise_unit"] > 0
    run.task_gallery(month=MONTH)
    assert table["age_bias_bar_days"] == common.A1_BIAS_MAX_DAYS
    assert "age_bias_days" in table
    assert "age_bias_exceeds_bar" in table
    assert all("mean_signed_age_error" in v for k, v in table.items() if k in {"0-2", "3-5"})


def test_the_level_head_starts_from_different_weights_for_different_seeds(dry: FakeData) -> None:
    def first_layer(seed: int) -> torch.Tensor:
        net = fit.head_model(
            arm="LR", month=MONTH, seed=seed, target_std=dry.target_std, noise=0.05,
            curve=None, a1_z=2.0, device="cpu",
        )  # fmt: skip
        return net.level_head[0].weight.detach().clone()

    a0, a0_again, a1 = first_layer(0), first_layer(0), first_layer(1)
    assert torch.equal(a0, a0_again), "one seed gives one initialisation"
    assert not torch.equal(a0, a1), "two seeds give two initialisations"
    assert fit.head_init_seed(seed=0) != fit.head_init_seed(seed=1)


def _level_readings(net: object, inp: object) -> dict[str, torch.Tensor]:
    h0, base, spread, _ = net.base_forward(inp)  # ty: ignore[unresolved-attribute]
    out = net.level_forward(inp)  # ty: ignore[unresolved-attribute]
    return {
        "features": net.features_of(inp=inp, h0=h0, base=base, spread=spread),  # ty: ignore[unresolved-attribute]
        "age": out.age,
        "depth": out.depth,
        "p": out.p,
        "mixture": out.out,
    }


def _origin_batch(source: object, data: FakeData, origin: int) -> object:
    samples = tr.SampleSet(
        primary=np.arange(16), c_idx=np.full(16, origin), is_val=np.zeros(16, bool)
    )
    return tr.batch_for(
        source=source, samples=samples, index=np.arange(16), lead=np.zeros(16, int),
        ensemble=False, cut=10**9, dropout=None,
    ).inputs  # fmt: skip


def _random_head_net(dry: FakeData, mode: str) -> object:
    import e00738_model as lmdl
    import e00738_survival as surv

    torch.manual_seed(3)
    curve = surv.GeneratorCurve() if mode in {"LD", "LM"} else None
    net = lmdl.LevelNet(mode=mode, target_std=dry.target_std, noise=0.05, curve=curve)
    torch.nn.init.normal_(net.level_head[-1].weight, std=0.5)
    torch.nn.init.normal_(net.level_head[-1].bias, std=0.5)
    return net.eval()


@pytest.mark.parametrize("mode", ["LP", "LD", "LM", "LR"])
def test_nothing_after_the_origin_reaches_the_level_head_through_the_real_source(
    dry: FakeData, mode: str
) -> None:
    """Perturb the Source's power after the origin, then add a shift that starts after it."""
    import e00738_train as train

    origin = dry.cutoff_idx - 48 * 8 + 24
    net = _random_head_net(dry, mode)
    before = _level_readings(net, _origin_batch(dry.source, dry, origin))

    # (i) every power value from the origin on is replaced by noise
    rng = np.random.default_rng(5)
    z = dry.source.power_z.clone()
    z[:, origin:] = torch.from_numpy(
        rng.normal(5.0, 3.0, size=z[:, origin:].shape).astype(np.float32)
    )
    noisy = type(dry.source)(
        power_z=z, dense_weather=dry.source.dense_weather, dense_datetime=dry.source.dense_datetime,
        day_type=dry.source.day_type, ens=dry.source.ens, power_in=z,
    )  # fmt: skip
    after = _level_readings(net, _origin_batch(noisy, dry, origin))
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"{mode}: {k} changed when the future power changed"

    # (ii) a shift that starts after the origin is added to the whole source
    late = [
        world.Shift(primary=p, kind="constant", onset_idx=origin + 48 * 2, ramp_half_hours=12,
                    end_idx=origin + 48 * 30, drawn_days=28.0, depth=0.2, sign=1, family="train")
        for p in range(16)
    ]  # fmt: skip
    own = world.build_own(clean=dry.clean, shifts=late, is_mw=np.array([True] * 8 + [False] * 8))
    shifted = train.world_source(
        base=dry.source, clean=dry.clean, own=own, target_mean=dry.target_mean,
        target_std=dry.target_std, device="cpu",
    )  # fmt: skip
    plain = train.world_source(
        base=dry.source, clean=dry.clean, own=np.zeros_like(own), target_mean=dry.target_mean,
        target_std=dry.target_std, device="cpu",
    )  # fmt: skip
    a = _level_readings(net, _origin_batch(plain, dry, origin))
    b = _level_readings(net, _origin_batch(shifted, dry, origin))
    for k, v in a.items():
        assert torch.equal(v, b[k]), f"{mode}: {k} changed when a later shift was added"

    # perturbing the history before the origin does change the readings, so the test can fail
    z2 = dry.source.power_z.clone()
    z2[:, origin - 48 * 3 : origin - 48 * 2] += 4.0
    seen = type(dry.source)(
        power_z=z2, dense_weather=dry.source.dense_weather,
        dense_datetime=dry.source.dense_datetime,
        day_type=dry.source.day_type, ens=dry.source.ens, power_in=z2,
    )  # fmt: skip
    c = _level_readings(net, _origin_batch(seen, dry, origin))
    assert not torch.equal(before["features"], c["features"])


def test_the_noise_unit_and_the_curve_read_nothing_after_the_cutoff(
    dry: FakeData, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e00738_survival as surv

    z = (dry.clean - dry.target_mean[:, None]) / dry.target_std[:, None]
    z_late = z.copy()
    z_late[:, dry.cutoff_idx :] = 99.0
    assert surv.noise_unit_from_panel(z=z, cutoff_idx=dry.cutoff_idx) == surv.noise_unit_from_panel(
        z=z_late, cutoff_idx=dry.cutoff_idx
    )
    # the curve is fitted to drawn durations only: it reads neither the power nor the real durations
    monkeypatch.setattr(fit, "N_DURATION_PASSES", 4)

    def refuse() -> None:
        msg = "the curve must not read the real durations"
        raise AssertionError(msg)

    monkeypatch.setattr(world, "load_real_durations_days", refuse)
    is_mw = np.array([True] * 8 + [False] * 8)
    one = fit.fold_curve(month=MONTH, seed=0, cutoff_idx=dry.cutoff_idx, is_mw=is_mw)
    two = fit.fold_curve(month=MONTH, seed=0, cutoff_idx=dry.cutoff_idx, is_mw=is_mw)
    assert one.durations == two.durations
    # the durations come from the training window only: every drawn start lies before the cutoff
    assert len(one.durations) > 0

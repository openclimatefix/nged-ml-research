"""Tests that need torch (skipped in CI): shapes, loss masks, and the sign logit's independence."""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import e00730_arms as arms  # noqa: E402
import e00730_banks as banks  # noqa: E402
import e00730_core as core  # noqa: E402
import e00730_nets as nets  # noqa: E402
import numpy as np  # noqa: E402


@pytest.mark.parametrize("arch", core.ARCHITECTURES)
def test_every_architecture_maps_a_window_to_the_outputs(arch: str) -> None:
    spec = arms.arm_specs(style="K")["SD*"]
    net = nets.build_net(arch=arch, spec=spec)
    x = torch.randn(2, arms.n_channels(spec=spec), 96)
    assert net(x).shape == (2, 3, 96)
    assert nets.count_weights(net=net) > 0


def test_transformer_shape_follows_the_plan() -> None:
    spec = arms.arm_specs(style="K")["SD*"]
    net = nets.build_net(arch="transformer", spec=spec)
    assert core.TRANSFORMER_LAYERS == 3
    assert core.TRANSFORMER_WIDTH == 128
    assert core.TRANSFORMER_HEADS == 4
    assert core.TRANSFORMER_FF == 512
    assert len([m for m in net.backbone.modules() if isinstance(m, nets.Block)]) == 3


@pytest.mark.parametrize("arch", core.ARCHITECTURES)
def test_the_sign_logit_does_not_depend_on_the_demand_rows(arch: str) -> None:
    spec = arms.arm_specs(style="K")["SD*"]
    net = nets.build_net(arch=arch, spec=spec).eval()
    x = torch.randn(1, arms.n_channels(spec=spec), 96)
    before = net(x)[:, 0].detach().clone()
    with torch.no_grad():
        head = net.head
        head.weight[1:] += 10.0
        assert head.bias is not None
        head.bias[1:] += 10.0
    after = net(x)[:, 0]
    assert torch.allclose(before, after)


def test_masked_loss_keeps_demand_and_drops_sign() -> None:
    raw = torch.zeros(1, 3, 48)
    targets = torch.zeros(1, 6, 48)
    targets[0, 0] = -1.0
    targets[0, 1] = 1.0
    keeps = torch.zeros(1, 2, 48)
    keeps[0, 1] = 1.0
    sign_dropped = nets.batch_loss(raw=raw, targets=targets, keeps=keeps, lam=0.1)
    keeps[0, 0] = 1.0
    sign_kept = nets.batch_loss(raw=raw, targets=targets, keeps=keeps, lam=0.1)
    assert sign_kept > sign_dropped
    only_sign = nets.batch_loss(raw=raw[:, :1], targets=targets, keeps=keeps, lam=0.1)
    assert only_sign == pytest.approx(float(np.log(2.0)), abs=1e-4)


def test_decode_gives_a_probability_and_a_positive_spread() -> None:
    out = nets.decode(raw=torch.randn(2, 3, 48))
    assert ((out["p"] >= 0) & (out["p"] <= 1)).all()
    assert (out["demand_s"] > 0).all()


# --------------------------------------------------------------------------- review 2 (M1, N4, S9)


def test_m1_the_fit_seed_is_defined_and_distinct_for_every_architecture_and_arm() -> None:
    import e00730_train as train

    arms_all = [*core.ARMS_NEW, *core.REFIT_729_ARMS]
    seeds = {
        train.fit_seed(arm=arm, arch=arch, seed=seed)
        for arm in arms_all
        for arch in core.ARCHITECTURES
        for seed in core.ARCH_SEEDS
    }
    assert len(seeds) == len(arms_all) * len(core.ARCHITECTURES) * len(core.ARCH_SEEDS)


@pytest.fixture
def fit_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    import e00729_arms as arms729
    import e00729_core as core729
    import e00730_stages as stages
    from fixture730 import make_root, open_gallery, write_choices

    root, holidays, _ = make_root(tmp_path, monkeypatch)
    monkeypatch.setattr(core, "EPOCH_WINDOWS", 64)
    monkeypatch.setattr(core729, "EPOCH_WINDOWS", 64)
    open_gallery(root)
    write_choices(root, style="W")
    banks.write_json(path=root / "scores" / "timing.json", payload={"pass": True})
    monkeypatch.setattr(stages, "real_sets", lambda ctx: {})
    ctx = SimpleNamespace(
        cal=SimpleNamespace(holidays=holidays),
        stats=arms729.InputStats(t_mean=10.0, t_sd=3.0, w_mean=5.0, w_sd=2.0),
    )
    return root, ctx


def encoder_for(root: Path) -> arms.EncoderTable:
    import e00729_banks as banks729

    days = []
    for kind, seed in (("train", 0), ("validation", None), ("shifted", None), ("test", None)):
        bank = banks729.load_bank(path=banks.bank_path(root=root, kind=kind, seed=seed))
        days += [s.dates for s in bank.segments]
    dates = np.unique(np.concatenate(days)).astype("datetime64[D]")
    values = np.random.default_rng(0).normal(size=(len(dates), 48, 28)).astype(np.float32)
    return arms.EncoderTable(dates=dates, values=values)


FIT_CASES = (
    ("S", "bilstm"), ("SD-W", "transformer"), ("SD*-C", "gru_transformer"),
    ("SDe-FR", "unet"), ("SDe-PT", "bilstm"),
)  # fmt: skip


@pytest.mark.parametrize(("arm", "arch"), FIT_CASES)
def test_n4_a_fit_runs_on_every_architecture_and_writes_every_expected_file(
    fit_world: tuple, arm: str, arch: str
) -> None:
    import e00729_banks as banks729
    import e00729_score as score729
    import e00730_stages as stages
    import e00730_train as train

    root, ctx = fit_world
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    n_rows = score729.rows_from_segments(segments=test.segments, holidays=ctx.cal.holidays).n
    kwargs: dict = {}
    if arm.startswith("SDe"):
        import e00545_model as model

        pretrained = root / "pretrained" / "M0x" / "2026-07_s0.pt"
        pretrained.parent.mkdir(parents=True)
        state = model.ModularMdn(n_primaries=16, aux=False, stage="pre").state_dict()
        torch.save(state, pretrained)
        kwargs = {
            "encoder": encoder_for(root), "d8_index": 3,
            "pretrained": pretrained if arm == "SDe-PT" else None,
        }  # fmt: skip
    record = stages.run_fit(
        root=root, ctx=ctx, arm=arm, arch=arch, seed=0, style="W", device="cpu",
        log=lambda _: None, max_epochs=2, **kwargs,
    )  # fmt: skip
    folder = root / "predictions" / stages.job_name(arm=arm, arch=arch, seed=0)
    for name in ("validation", "test", "shifted"):
        assert np.load(folder / f"{name}.npz")["p"].shape[0] >= 1
    assert np.load(folder / "test.npz")["p"].shape[0] == n_rows
    names = {p.name for p in folder.iterdir()}
    if arm.startswith("SD"):
        assert {
            f"{kind}_masked_{s}.npz" for kind in ("validation", "test") for s in core.STYLES
        } <= names
    if arm == "SD*-C":
        assert {f"test_causal_k{k}.npz" for k in core.CAUSAL_K} <= names
        assert np.load(folder / "test_causal_k0.npz")["p"].shape[0] == n_rows
    if arm == "SD-W" and arch in core.TRANSFORMER_VARIANTS:
        assert stages.ATTENTION_FILE in names
    if arm == "SDe-PT":
        assert record["pretrained_digest"]
    del train


def test_m9_sde_fr_refuses_a_pretrained_file_and_sde_pt_needs_one(fit_world: tuple) -> None:
    import e00730_stages as stages

    root, ctx = fit_world
    pretrained = root / "pretrained" / "x.pt"
    pretrained.parent.mkdir(parents=True)
    torch.save({}, pretrained)
    base = {"encoder": encoder_for(root), "d8_index": 3}
    with pytest.raises(ValueError, match=r"fresh|pretrained|pre-trained"):
        stages.run_fit(
            root=root, ctx=ctx, arm="SDe-FR", arch="bilstm", seed=0, style="W", device="cpu",
            log=lambda _: None, max_epochs=1, pretrained=pretrained, **base,
        )  # fmt: skip
    with pytest.raises(ValueError, match=r"fresh|pretrained|pre-trained"):
        stages.run_fit(
            root=root, ctx=ctx, arm="SDe-PT", arch="bilstm", seed=0, style="W", device="cpu",
            log=lambda _: None, max_epochs=1, pretrained=None, **base,
        )  # fmt: skip


def test_m10_the_encoder_row_is_the_d8_row_of_the_pretrained_embedding() -> None:
    net = nets.build_net(arch="bilstm", spec=arms.arm_specs(style="K")["SDe-FR"], d8_index=7)
    assert net.front is not None
    assert net.front.d8_index == 7


@pytest.mark.parametrize("arch", core.ARCHITECTURES)
def test_m20_a_causal_output_does_not_move_when_the_data_after_the_cut_changes(arch: str) -> None:
    import e00729_core as core729
    import e00729_sim as sim
    import e00730_masks as masks
    import e00730_score as score
    import e00730_train as train
    from world730 import make_calendar

    cal = make_calendar()
    segment = sim.simulate_segment(
        cal=cal, month="2022-03", use=np.ones(len(cal.dates), dtype=bool), world="both",
        rng=np.random.default_rng(1),
    )  # fmt: skip
    spec = arms.arm_specs(style="K")["SD*-C"]
    stats = arms729_stats()
    base = arms.prepare_segments(segments=[segment], spec=spec, holidays=cal.holidays, stats=stats)[
        0
    ]
    net = nets.build_net(arch=arch, spec=spec).eval()
    observed = [i for i, n in enumerate(arms.stack_names(spec=spec)) if n in masks.OBSERVED_ROWS]
    rng = np.random.default_rng(5)
    day = int(base.days[3])
    start = day * 48 - core729.MIDDLE_START
    for t, k in ((0, 0), (10, 2), (30, 6)):
        pos = score.causal_cut(t=t, k=k)
        assert pos is not None
        cut = core729.MIDDLE_START + pos
        stack = base.stack.copy()
        for row in observed:
            stack[row, start + cut + 1 :] = rng.normal(size=stack.shape[1] - start - cut - 1) * 50
        hourly = base.hourly.copy()
        hour = (start + cut) // 2
        hourly[:3, hour + 1 :] = rng.normal(size=hourly[:3, hour + 1 :].shape) * 50
        hourly[3, hour:] = rng.normal(size=hourly[3, hour:].shape) * 50
        scrambled = arms.PreparedSegment(**{**base.__dict__, "stack": stack, "hourly": hourly})  # ty: ignore[invalid-argument-type]
        picks, cuts = [(0, day)], np.array([cut])
        a, _, _ = train.predict(
            net=net, prepared=[base], spec=spec, device="cpu", picks=picks, cut_pos=cuts
        )
        b, _, _ = train.predict(
            net=net, prepared=[scrambled], spec=spec, device="cpu", picks=picks, cut_pos=cuts
        )
        assert a["p"][0, t] == b["p"][0, t], f"{arch}: P(X<0) at {t} moved with k = {k}"
    # not vacuous: with no cut, the scrambled data do move the output
    c, _, _ = train.predict(net=net, prepared=[base], spec=spec, device="cpu", picks=[(0, day)])
    d, _, _ = train.predict(
        net=net, prepared=[scrambled], spec=spec, device="cpu", picks=[(0, day)]
    )
    assert not np.allclose(c["p"], d["p"])


def arms729_stats():
    import e00729_arms as arms729

    return arms729.InputStats(10.0, 3.0, 5.0, 2.0)


def test_s9_the_attention_maps_have_one_row_per_scored_half_hour(fit_world: tuple) -> None:
    import e00729_banks as banks729
    import e00730_train as train

    root, ctx = fit_world
    spec = arms.arm_specs(style="K")["SD-W"]
    net = nets.build_net(arch="transformer", spec=spec)
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    prepared = arms.prepare_segments(
        segments=test.segments, spec=spec, holidays=ctx.cal.holidays, stats=ctx.stats
    )
    maps = train.attention_maps(net=net, prepared=prepared, spec=spec, device="cpu")
    assert set(maps) <= set(train.ATTENTION_GROUPS)
    assert maps
    for grid in maps.values():
        assert grid.shape == (48, 96)
        assert np.allclose(grid.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.parametrize("arm", core.REFIT_729_ARMS)
def test_n4_each_refit_of_an_experiment_729_arm_runs_and_stores_its_predictions(
    fit_world: tuple, arm: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e00729_core as core729
    import e00730_stages as stages
    import e00730_train as train

    root, ctx = fit_world
    monkeypatch.setattr(core729, "MAX_EPOCHS", 2)
    monkeypatch.setattr(train.core729, "MAX_EPOCHS", 2)
    stages.run_refit_729(root=root, ctx=ctx, arm=arm, arch="bilstm", seed=0, device="cpu")
    folder = root / "predictions" / stages.job_name(arm=arm, arch="bilstm", seed=0)
    assert np.load(folder / "test.npz")["mu_i"].shape[0] >= 1


def test_s7_the_causal_read_out_is_timed_on_the_test_bank(
    fit_world: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e00730_stages as stages

    root, ctx = fit_world
    monkeypatch.setattr(core, "TIMING_EPOCHS", 1)
    assert stages.measure_causal_seconds(root=root, ctx=ctx, arch="bilstm", device="cpu") > 0.0
    assert (
        stages.measure_seconds_per_epoch(
            root=root, ctx=ctx, arch="bilstm", device="cpu", epochs=1, arm="SD-W"
        )
        > 0.0
    )

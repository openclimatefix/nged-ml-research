# ruff: noqa: E501, PT018
"""Tests of `e00733_train`: the batch wiring, the optimiser groups, both loops, and the host conversion."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import e00733_core as core
import e00733_leakage as leak
import e00733_model as model
import e00733_paths as paths
import e00733_synth as sy
import numpy as np
import pytest
from synth_fixture import make_segment

torch = pytest.importorskip("torch")
import e00713_layout as lay  # noqa: E402  # ty: ignore[unresolved-import]
import e00733_train as tn  # noqa: E402
from torch_inputs import FakeSource, make_channel_tensors  # noqa: E402


class _FakeTensor:
    """A stand-in for a CUDA tensor: `numpy()` before `cpu()` fails, as a real CUDA tensor does."""

    def __init__(self) -> None:
        self.on_host = False

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        self.on_host = True
        return self

    def numpy(self) -> np.ndarray:
        if not self.on_host:
            msg = "can't convert cuda:0 device type tensor to numpy"
            raise TypeError(msg)
        return np.zeros(2)


def test_to_numpy_moves_to_the_host_first() -> None:
    """The conversion calls `.detach().cpu()` before `.numpy()` (a CUDA tensor raises otherwise)."""
    assert tn.to_numpy(tensor=_FakeTensor()).shape == (2,)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        _FakeTensor().numpy()


def test_param_groups_put_every_new_parameter_in_a_group_without_warmup() -> None:
    """The trunk groups warm up; the new parameters form their own groups, the gain without weight decay."""
    net = model.UntangledNet(seed=0, sign_head=False)
    net.freeze_encoders()
    groups = tn.param_groups(net=net, stage=2, encoders="frozen")
    new_ids = {id(p) for n, p in net.named_parameters() if n.startswith(model.NEW_PREFIXES)}
    in_new = {id(p) for g in groups if not g["trunk"] for p in g["params"]}
    assert in_new == new_ids
    assert all(g["warmup"] for g in groups if g["trunk"])
    gain = next(g for g in groups if any(p is net.u_gamma for p in g["params"]))
    assert gain["weight_decay"] == 0.0 and gain["lr"] == pytest.approx(3e-3)
    other = next(g for g in groups if g["lr"] == pytest.approx(2e-3) and not g["trunk"])
    assert other["weight_decay"] == pytest.approx(1e-3)
    scaled = tn.param_groups(net=net, stage=2, encoders="frozen", new_lr_scale=0.0)
    assert all(g["lr"] == 0.0 for g in scaled if not g["trunk"])
    trunk, new = tn.split_params(optimiser=torch.optim.AdamW(groups), net=net)
    assert len(new) == len(new_ids) and not {id(p) for p in trunk} & new_ids


def test_unt_source_adds_channels_only_at_d8_and_checks_the_run() -> None:
    """The batch carries channels at D8 only; a run issued at the wrong time fails the origin-time assertion."""
    n_grid = 48 * 400
    src = tn.UntSource(source=FakeSource(), tensors=make_channel_tensors(n_grid=n_grid))
    c_idx = torch.tensor([48 * 300 + 24, 48 * 301 + 24])
    primary = torch.tensor([3, 5])
    got = src.assemble(
        primary=primary, c_idx=c_idx, lead_day=torch.tensor([0, 2]), cams_rule=None, convention=None
    )
    assert got.inputs.hist_u_ok[0].any() and not got.inputs.hist_u_ok[1].any()
    assert (got.inputs.hist_u[1] == 0).all()
    late = tn.UntSource(
        source=FakeSource(run_offset_us=3 * lay.HOUR_US),
        tensors=make_channel_tensors(n_grid=n_grid),
    )
    with pytest.raises(leak.CorrectnessFailureError):
        late.assemble(primary=primary, c_idx=c_idx, lead_day=torch.tensor([0, 2]))


def test_channels_are_absent_where_stage_3_shifts_the_power() -> None:
    """Tokens whose history power differs from the clean power carry no channel; the rest keep it."""
    fake = FakeSource()
    c_idx, primary, lead = torch.tensor([48 * 300 + 24]), torch.tensor([3]), torch.tensor([0])
    src = tn.UntSource(source=fake, tensors=make_channel_tensors(n_grid=48 * 400))
    clean = src.assemble(primary=primary, c_idx=c_idx, lead_day=lead)
    assert clean.inputs.hist_u_ok[0, 5].any()
    shifted_power = fake.power_z.clone()
    row = 5
    idx = clean.hist_idx[0, row]
    shifted_power[3, idx] += 1.0
    shifted_power[3, clean.hist_idx[0, 7, 0]] = float("nan")  # NaN in both: not a shift
    fake.power_z[3, clean.hist_idx[0, 7, 0]] = float("nan")
    fake.power_in = shifted_power
    got = src.assemble(primary=primary, c_idx=c_idx, lead_day=lead)
    ok = got.inputs.hist_u_ok[0]
    assert not ok[row].any() and (got.inputs.hist_u[0, row] == 0).all()
    keep = [r for r in range(14) if r != row]
    assert torch.equal(ok[keep], clean.inputs.hist_u_ok[0][keep])


def test_check_batch_rejects_history_after_the_target() -> None:
    """A history token at or after the first target token fails the batch check."""
    src = tn.UntSource(source=FakeSource(), tensors=make_channel_tensors(n_grid=48 * 400))
    b = src.assemble(
        primary=torch.tensor([3]), c_idx=torch.tensor([48 * 300 + 24]), lead_day=torch.tensor([0])
    )
    b.hist_idx = b.target_idx[:, :1, None].expand(-1, 14, 48).clone()
    with pytest.raises(leak.CorrectnessFailureError):
        tn.check_batch(batch=b)


@pytest.mark.parametrize(
    ("field", "extra_us"), [("avail_extra_us", 1_800_000_000), ("other_extra_us", 9 * lay.HOUR_US)]
)
def test_a_target_row_column_one_step_past_the_native_origin_fails_the_batch(
    field: str, extra_us: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving a run's availability, or another target-row column's valid time, to the origin fails."""
    monkeypatch.setattr(paths, "private_dir", lambda **_: Path(tempfile.mkdtemp()))
    fake = FakeSource()
    c_idx, primary = torch.tensor([48 * 300 + 24]), torch.tensor([3])
    ok = tn.UntSource(source=fake, tensors=make_channel_tensors(n_grid=48 * 400))
    ok.assemble(primary=primary, c_idx=c_idx, lead_day=torch.tensor([0]))
    assert tn._EXAMPLE_PRINTED and tn._EXAMPLE_PRINTED[-1].endswith("pass")
    setattr(fake, field, extra_us)
    bad = tn.UntSource(source=fake, tensors=make_channel_tensors(n_grid=48 * 400))
    with pytest.raises(leak.CorrectnessFailureError):
        bad.assemble(primary=primary, c_idx=c_idx, lead_day=torch.tensor([0]))


def test_a_source_without_run_file_times_cannot_be_checked() -> None:
    """Experiment 713's own run times are the origin minus 9 hours, so they are never read."""
    fake = FakeSource()
    c_idx, primary = torch.tensor([48 * 300 + 24]), torch.tensor([3])
    src = tn.UntSource(source=fake, tensors=make_channel_tensors(n_grid=48 * 400))
    src.assemble(primary=primary, c_idx=c_idx, lead_day=torch.tensor([0]))
    del fake.ens.file_run_avail_us
    with pytest.raises(leak.CorrectnessFailureError):
        tn.run_times_of(source=src, primary=primary, c_idx=c_idx)


# --------------------------------------------------------------------------- the synthetic loop


def _world(*, n_days: int = 40, seed: int = 0) -> tuple[list[sy.SegmentArrays], sy.SampleIndex]:
    segs = [make_segment(n_days=n_days, seed=seed)]
    idx = sy.enumerate_samples(segments=segs)
    sel = np.arange(len(idx))[:: max(1, len(idx) // 48)]
    return segs, sy.SampleIndex(idx.seg[sel], idx.origin_day[sel], idx.lead_index[sel])


def test_synthetic_loop_records_both_losses_and_marks_the_stored_pass() -> None:
    """Every pass has a training loss, a validation loss and its two parts; the stored pass is the best one."""
    segs, idx = _world()
    net = model.UntangledNet(seed=0, sign_head=True, clim_logit=-1.5)
    hist = tn.train_synthetic(
        net=net, train_segments=segs, train_index=idx, val_segments=segs, val_index=idx,
        seed=0, mode="untangled", d8_index=3, target_std=1.0, device="cpu", max_epochs=3, patience=2, batch_size=16,
    )  # fmt: skip
    n = len(hist.val)
    assert 1 <= n <= 3 and len(hist.train) == len(hist.val_nll) == len(hist.val_bce) == n
    assert hist.stored_pass == int(np.argmin(hist.val)) + 1
    assert hist.val[hist.stored_pass - 1] == pytest.approx(
        hist.val_nll[hist.stored_pass - 1] + hist.val_bce[hist.stored_pass - 1], rel=1e-3
    )
    assert all(np.isfinite(hist.train))
    assert not net.training


def test_synthetic_loop_moves_the_sign_head_and_is_reproducible() -> None:
    """The sign head's output layer leaves zero, and two runs with one seed agree."""
    segs, idx = _world()
    kwargs = {
        "train_segments": segs,
        "train_index": idx,
        "val_segments": segs,
        "val_index": idx,
        "seed": 1,
        "mode": "untangled",
        "d8_index": 3,
        "target_std": 1.0,
        "device": "cpu",
        "max_epochs": 2,
        "patience": 5,
        "batch_size": 16,
    }
    torch.manual_seed(5)
    a = model.UntangledNet(seed=0, sign_head=True)
    torch.manual_seed(5)
    b = model.UntangledNet(seed=0, sign_head=True)
    ha = tn.train_synthetic(net=a, **kwargs)  # ty: ignore[invalid-argument-type]
    hb = tn.train_synthetic(net=b, **kwargs)  # ty: ignore[invalid-argument-type]
    assert float(a.sign_out.weight.detach().abs().sum()) > 0
    assert ha.val == hb.val


def test_synthetic_y0_reads_no_channel() -> None:
    """The control arm sees absent channels in every batch, so its path never receives a flagged token."""
    segs, idx = _world()
    b = sy.gather_batch(
        segments=segs, index=idx, rows=np.arange(8), d8_index=3, target_std=1.0, mode="absent"
    )
    x = tn.synthetic_arrays_to_inputs(batch=b, device="cpu")
    assert not bool(x.hist_u_ok.any())
    net = model.UntangledNet(seed=0, sign_head=True)
    loss, nll, bce = tn.synthetic_loss(net=net, batch=b, device="cpu")
    assert float(loss.detach()) == pytest.approx(
        float(nll.detach()) + float(bce.detach()), rel=1e-5
    )


def test_cap_hit_and_still_falling_flags() -> None:
    """A loop that ends at the cap says so and whether the validation loss was still falling."""
    segs, idx = _world()
    net = model.UntangledNet(seed=0, sign_head=True)
    hist = tn.train_synthetic(
        net=net, train_segments=segs, train_index=idx, val_segments=segs, val_index=idx,
        seed=2, mode="absent", d8_index=3, target_std=1.0, device="cpu", max_epochs=2, patience=9, batch_size=16,
    )  # fmt: skip
    assert hist.cap_hit and len(hist.val) == 2
    assert hist.still_falling == (hist.val[-1] < hist.val[-2])
    assert set(hist.as_dict()) >= {"train", "val", "stored_pass", "cap_hit", "still_falling"}


# --------------------------------------------------------------------------- the real-fold loop with stubs


class _World:
    def __init__(self, source: tn.UntSource) -> None:
        self.source = source


def _samples(*, n: int = 40) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        primary=np.full(n, 3), c_idx=(300 + rng.integers(0, 50, n)) * 48 + 24,
        is_val=np.arange(n) % 5 == 0, ages=None,
    )  # fmt: skip


def test_real_fold_loop_trains_the_new_parameters_and_records_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With stubs for the world's helpers, the loop returns a history with the stored pass and moves the gain."""
    monkeypatch.setattr(tn.tr, "token_weights", lambda **_: torch.ones(1, 1))
    monkeypatch.setattr(tn.tr, "target_end", lambda **_: 10**9)
    monkeypatch.setattr(tn.tr, "BATCH", 8)
    monkeypatch.setattr(tn.tr, "VALIDATION_LEAD_DAYS", (0, 3))
    monkeypatch.setattr(
        tn,
        "block_dropout_masks",
        lambda **kw: (np.zeros(len(kw["origin_idx"]), bool), np.zeros(len(kw["origin_idx"]), bool)),
    )
    src = tn.UntSource(source=FakeSource(), tensors=make_channel_tensors(n_grid=48 * 400))
    net = model.UntangledNet(seed=0)
    net.freeze_encoders()
    before = {
        n: p.detach().clone()
        for n, p in net.named_parameters()
        if not n.startswith(model.NEW_PREFIXES)
    }
    log, hist = tn.train_stage(
        net=net,
        world=_World(src),
        stage=2,
        samples=_samples(),
        seed=0,
        encoders="frozen",
        max_passes=2,
        patience=3,
    )
    assert hist.stage == 2 and len(hist.train) == len(hist.val) == 2 and hist.stored_pass >= 1
    assert log.passes == 2 and log.cap_hit
    assert float(net.u_gamma.detach().abs().sum()) > 0
    frozen = [n for n in before if n.startswith(("weather.", "datetime.", "embedding."))]
    assert all(torch.equal(before[n], dict(net.named_parameters())[n]) for n in frozen)


def test_real_fold_loop_with_zero_new_rate_leaves_the_gain_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`new_lr_scale = 0` keeps every new parameter at its start (the trunk-only check)."""
    monkeypatch.setattr(tn.tr, "token_weights", lambda **_: torch.ones(1, 1))
    monkeypatch.setattr(tn.tr, "target_end", lambda **_: 10**9)
    monkeypatch.setattr(tn.tr, "BATCH", 8)
    monkeypatch.setattr(tn.tr, "VALIDATION_LEAD_DAYS", (0,))
    monkeypatch.setattr(
        tn,
        "block_dropout_masks",
        lambda **kw: (np.zeros(len(kw["origin_idx"]), bool), np.zeros(len(kw["origin_idx"]), bool)),
    )
    src = tn.UntSource(source=FakeSource(), tensors=make_channel_tensors(n_grid=48 * 400))
    net = model.UntangledNet(seed=0)
    net.freeze_encoders()
    tn.train_stage(
        net=net,
        world=_World(src),
        stage=2,
        samples=_samples(),
        seed=0,
        encoders="frozen",
        max_passes=1,
        new_lr_scale=0.0,
        max_steps=2,
    )
    assert float(net.u_gamma.detach().abs().sum()) == 0.0
    assert core.SYNTHETIC_PATIENCE == 20  # the registered patience of the synthetic loop


# --------------------------------------------------------------------------- the prefetcher


def test_prefetched_gives_the_results_in_order_whatever_the_thread_count() -> None:
    """Four threads give the sequence of one thread, even when the early items are the slow ones."""
    import time

    def make(i: int) -> int:
        time.sleep(0.003 * (5 - i % 5))
        return i * i

    one = list(tn.prefetched(items=range(23), make=make, workers=1))
    many = list(tn.prefetched(items=range(23), make=make, workers=4, depth=6))
    assert one == many == [i * i for i in range(23)]


def test_prefetched_raises_where_the_result_would_be_taken() -> None:
    """An error in `make` reaches the consumer in the item's own position."""

    def make(i: int) -> int:
        if i == 3:
            msg = "boom"
            raise ValueError(msg)
        return i

    stream = tn.prefetched(items=range(10), make=make, workers=3, depth=4)
    assert [next(stream), next(stream), next(stream)] == [0, 1, 2]
    with pytest.raises(ValueError, match="boom"):
        next(stream)


def test_the_synthetic_loop_gives_the_same_history_with_and_without_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefetch threads change the speed only: two runs, one thread and four, agree to the bit."""
    segs, idx = _world()
    hists = []
    for workers in (1, 4):
        monkeypatch.setattr(tn, "PREFETCH_WORKERS", workers)
        torch.manual_seed(5)
        net = model.UntangledNet(seed=0, sign_head=True)
        hists.append(
            tn.train_synthetic(
                net=net,
                train_segments=make_world_copy(segs),
                train_index=idx,
                val_segments=make_world_copy(segs),
                val_index=idx,
                seed=1,
                mode="untangled",
                d8_index=3,
                target_std=1.0,
                device="cpu",
                max_epochs=2,
                patience=5,
                batch_size=16,
            )
        )
    assert hists[0].train == hists[1].train and hists[0].val == hists[1].val


def make_world_copy(segs: list[sy.SegmentArrays]) -> list[sy.SegmentArrays]:
    """Return a deep copy of the segments (packing rebinds a segment's arrays)."""
    import copy

    return copy.deepcopy(segs)


# --------------------------------------------------------------------------- no write into the real data folder


def _listing(folder: Path) -> list[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*")) if folder.exists() else []


def test_the_example_row_goes_to_the_given_root_and_never_to_the_data_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a root the example row lands under it; without one it is printed only; the folder that
    `DATA_PATH_INTERNAL` names gains no file either way."""
    internal = tmp_path / "internal"
    internal.mkdir()
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(internal))
    before = _listing(internal)
    for root_name in (None, "root"):
        monkeypatch.setattr(tn, "_EXAMPLE_PRINTED", [])
        root = tmp_path / root_name if root_name else None
        src = tn.UntSource(
            source=FakeSource(), tensors=make_channel_tensors(n_grid=48 * 400), root=root
        )
        src.assemble(
            primary=torch.tensor([3]),
            c_idx=torch.tensor([48 * 300 + 24]),
            lead_day=torch.tensor([0]),
        )
        assert len(tn._EXAMPLE_PRINTED) == 1
        written = tmp_path / "root" / "PRIVATE" / "origin_time_example_real.txt"
        assert written.exists() == (root is not None)
    assert _listing(internal) == before


def test_the_prefetch_stream_is_closed_when_the_loss_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `NonFiniteLossError` in the loop closes the prefetch generator, so no worker thread stays alive."""
    import threading

    streams = []
    real = tn.prefetched

    def tracking(**kw):  # noqa: ANN003, ANN202
        gen = real(**kw)
        streams.append(gen)
        return gen

    def failing_loss(**_):  # noqa: ANN003, ANN202
        msg = "loss is not finite"
        raise tn.NonFiniteLossError(msg)

    monkeypatch.setattr(tn, "prefetched", tracking)
    monkeypatch.setattr(tn, "PREFETCH_WORKERS", 4)
    monkeypatch.setattr(tn, "synthetic_loss", failing_loss)
    segs, idx = _world()
    net = model.UntangledNet(seed=0, sign_head=True)
    before = threading.active_count()
    with pytest.raises(tn.NonFiniteLossError):
        tn.train_synthetic(
            net=net, train_segments=segs, train_index=idx, val_segments=make_world_copy(segs), val_index=idx,
            seed=0, mode="untangled", d8_index=3, target_std=1.0, device="cpu", max_epochs=2, patience=2, batch_size=16,
        )  # fmt: skip
    assert len(streams) == 1
    assert streams[0].gi_frame is None  # a closed (finished) generator holds no frame
    assert threading.active_count() == before

# ruff: noqa: E501, PT018
"""Tests of `e00733_train`: the batch wiring, the optimiser groups, both loops, and the host conversion."""

import tempfile
from pathlib import Path

import e00733_leakage as leak
import e00733_model as model
import e00733_paths as paths
import numpy as np
import pytest

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

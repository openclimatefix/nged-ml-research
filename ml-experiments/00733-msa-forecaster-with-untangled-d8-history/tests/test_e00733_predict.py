# ruff: noqa: E501, PT018
"""Tests of `e00733_predict`: the forecast frame, the synthetic archive, and the reads of stored output."""

from datetime import UTC, datetime
from pathlib import Path

import e00733_core as core
import e00733_model as model
import e00733_synth as sy
import numpy as np
import pytest
from synth_fixture import make_segment

torch = pytest.importorskip("torch")
import e00713_layout as lay  # noqa: E402  # ty: ignore[unresolved-import]
import e00733_predict as pr  # noqa: E402
import e00733_train as tn  # noqa: E402
from torch_inputs import FakeSource, make_channel_tensors  # noqa: E402


def _frame(
    *, net: model.UntangledNet, tensors: object, seed: int = 0, holdout: datetime | None = None
) -> object:
    src = tn.UntSource(source=FakeSource(), tensors=tensors)  # ty: ignore[invalid-argument-type]
    origins = np.array([300 * 48 + 18, 301 * 48 + 18])
    return pr.forecast_frame(
        net=net, test_origin_idx=origins, source=src, target_mean=np.linspace(0, 1, 16),
        target_std=np.linspace(1, 2, 16), seed=seed, grid_start_us=lay.GRID_START_US,
        time_series_ids=np.arange(16) + 100, holdout_start=holdout or datetime(2099, 1, 1, tzinfo=UTC),
    )  # fmt: skip


def _net() -> model.UntangledNet:
    torch.manual_seed(0)
    net = model.UntangledNet(seed=0)
    with torch.no_grad():
        for p in net.head.parameters():
            p.normal_(std=0.1)
    return net.eval()


def test_forecast_frame_has_the_columns_and_the_row_count_of_713() -> None:
    """16 Primaries x 2 origins x 14 lead days x 48 half-hours, sorted, with sorted quantiles."""
    frame = _frame(net=_net(), tensors=make_channel_tensors(n_grid=48 * 400))
    assert tuple(frame.columns) == pr.FRAME_COLUMNS  # ty: ignore[unresolved-attribute]
    assert frame.height == 16 * 2 * 14 * 48  # ty: ignore[unresolved-attribute]
    q = frame.select(pr.QCOLS).to_numpy()  # ty: ignore[unresolved-attribute]
    assert (np.diff(q, axis=1) >= 0).all()
    assert frame["time_series_id"].min() == 100  # ty: ignore[not-subscriptable, unresolved-attribute]
    assert frame["lead_hours"].min() == 3.0  # ty: ignore[not-subscriptable, unresolved-attribute]


def test_holdout_rows_are_dropped() -> None:
    """Valid times at or after the holdout start are not in the frame."""
    full = _frame(net=_net(), tensors=make_channel_tensors(n_grid=48 * 400))
    cut = _frame(
        net=_net(),
        tensors=make_channel_tensors(n_grid=48 * 400),
        holdout=datetime(2019, 9, 1, tzinfo=UTC)
        + (datetime(2019, 9, 1, tzinfo=UTC) - datetime(2019, 9, 1, tzinfo=UTC))
        + __import__("datetime").timedelta(days=305),
    )
    assert cut.height < full.height  # ty: ignore[unresolved-attribute]


def test_an_absent_channel_run_equals_the_masked_run() -> None:
    """A prediction with every channel masked equals one on tensors with every channel absent.

    The R4 readout is a source built from other tensors.
    """
    import e00733_inputs as inp

    net = _net()
    with torch.no_grad():
        net.u_gamma.fill_(0.4)
        net.u_head_in.weight.normal_(std=0.2)
    full = make_channel_tensors(n_grid=48 * 400)
    a = _frame(net=net, tensors=inp.mask_all(tensors=full))
    b = _frame(net=net, tensors=inp.absent_tensors(n_primaries=16, n_grid=48 * 400, d8_index=3))
    c = _frame(net=net, tensors=full)
    assert a.select(pr.QCOLS).equals(b.select(pr.QCOLS))  # ty: ignore[unresolved-attribute]
    assert not a.select(pr.QCOLS).equals(c.select(pr.QCOLS))  # ty: ignore[unresolved-attribute]


def test_compare_forecasts_checks_keys_and_tolerance() -> None:
    """Identical frames pass; a shift beyond 1e-6 or other keys fail."""
    f = _frame(net=_net(), tensors=make_channel_tensors(n_grid=48 * 400))
    assert "largest difference 0.00e+00" in pr.compare_forecasts(new=f, stored=f)  # ty: ignore[invalid-argument-type]
    import polars as pl

    shifted = f.with_columns(pl.col("q50") + 1e-3)  # ty: ignore[unresolved-attribute]
    with pytest.raises(AssertionError):
        pr.compare_forecasts(new=shifted, stored=f)  # ty: ignore[invalid-argument-type]
    with pytest.raises(AssertionError):
        pr.compare_forecasts(new=f.reverse(), stored=f)  # ty: ignore[unresolved-attribute, invalid-argument-type]


def test_write_frame_is_atomic(tmp_path: Path) -> None:
    """The frame is written under a temporary name and renamed; no temporary file stays."""
    import polars as pl

    path = pr.write_frame_atomic(frame=pl.DataFrame({"a": [1]}), path=tmp_path / "x" / "f.parquet")
    assert path.exists() and not list(path.parent.glob("*.tmp"))


def _split() -> tuple[list[sy.SegmentArrays], sy.SampleIndex]:
    segs = [make_segment(n_days=40)]
    idx = sy.enumerate_samples(segments=segs)
    sel = np.arange(len(idx))[::40]
    return segs, sy.SampleIndex(idx.seg[sel], idx.origin_day[sel], idx.lead_index[sel])


def test_predict_synthetic_stores_probabilities_and_keys(tmp_path: Path) -> None:
    """The archive holds probabilities in 0 to 1, sorted quantiles, the targets.

    And the keys; the stored read checks them.
    """
    segs, idx = _split()
    torch.manual_seed(1)
    net = model.UntangledNet(seed=0, sign_head=True, clim_logit=-2.0).eval()
    out = pr.predict_synthetic(
        net=net,
        segments=segs,
        index=idx,
        d8_index=3,
        target_std=1.0,
        mode="untangled",
        device="cpu",
        batch_size=7,
    )
    assert out["p_neg"].shape == (len(idx), 48) and out["quantiles"].shape == (len(idx), 48, 11)
    path = pr.write_synthetic(arrays=out, path=tmp_path / "p" / "YU_s0.npz")
    back = pr.read_synthetic(path=path)
    pr.check_stored_synthetic(arrays=back, n_expected=len(idx))
    assert np.array_equal(back["origin_day"], idx.origin_day)
    assert (np.diff(back["quantiles"], axis=-1) >= -1e-6).all()
    with pytest.raises(ValueError, match="another number"):
        pr.check_stored_synthetic(arrays=back, n_expected=len(idx) + 1)
    bad = dict(back)
    bad["p_neg"] = back["p_neg"] + 2.0
    with pytest.raises(ValueError, match="outside 0 to 1"):
        pr.check_stored_synthetic(arrays=bad, n_expected=len(idx))


def test_predict_synthetic_needs_the_sign_head() -> None:
    """A network without the sign head cannot forecast the synthetic world."""
    segs, idx = _split()
    with pytest.raises(ValueError, match="no sign head"):
        pr.predict_synthetic(
            net=model.UntangledNet(seed=0),
            segments=segs,
            index=idx,
            d8_index=3,
            target_std=1.0,
            mode="absent",
            device="cpu",
        )


def test_batch_size_does_not_change_the_synthetic_forecast() -> None:
    """Two batch sizes give the same probabilities (evaluation mode has no dropout)."""
    segs, idx = _split()
    torch.manual_seed(2)
    net = model.UntangledNet(seed=0, sign_head=True).eval()
    a = pr.predict_synthetic(
        net=net,
        segments=segs,
        index=idx,
        d8_index=3,
        target_std=1.0,
        mode="untangled",
        device="cpu",
        batch_size=5,
    )
    b = pr.predict_synthetic(
        net=net,
        segments=segs,
        index=idx,
        d8_index=3,
        target_std=1.0,
        mode="untangled",
        device="cpu",
        batch_size=64,
    )
    assert np.allclose(a["p_neg"], b["p_neg"], atol=1e-5)
    assert core.SYNTHETIC_BATCH == 256

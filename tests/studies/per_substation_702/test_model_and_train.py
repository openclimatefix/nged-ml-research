# ruff: noqa: ANN001, ANN202, ANN003, E501, PT018
"""CPU tests of experiment 702's network variants and training loop on tiny random data.

They need `torch` and the earlier experiments' folders on `PYTHONPATH` (`source
ml-experiments/00524*/env.sh`), so they skip in CI.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
m591 = pytest.importorskip("e00591_model")
model_702 = pytest.importorskip("e00702_model")
train_702 = pytest.importorskip("e00702_train")

N_KNOTS = 4
N_B1 = 14
WIDTH = 46 + N_B1  # 45 lag-network columns, the knot position, then the B1 columns
N_PRIMARIES = 16


def _table(*, n_primaries: int) -> torch.Tensor:
    """Return a random expected-input table: 7 weather and 21 date-time columns per grid index."""
    return torch.randn(n_primaries, 50, 28)


def _rows(*, n: int, n_primaries: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return random inputs and Primary indices that satisfy the network's index columns."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, WIDTH, generator=gen)
    x[:, 38] = (torch.rand(n, generator=gen) < 0.1).float()  # day lag missing
    x[:, 41] = (torch.rand(n, generator=gen) < 0.1).float()  # week lag missing
    x[:, 43] = torch.randint(0, 50, (n,), generator=gen).float()
    x[:, 44] = torch.randint(0, 50, (n,), generator=gen).float()
    x[:, 45] = torch.rand(n, generator=gen) * (N_KNOTS - 1)
    return x, torch.randint(0, n_primaries, (n,), generator=gen)


def _k1(*, stage: str = "ft", n_primaries: int = N_PRIMARIES):
    net = m591.AuxLagMdn(
        n_primaries=n_primaries, stage=stage, n_knots=N_KNOTS, n_b1=N_B1, kind="shared"
    )
    net.table = _table(n_primaries=n_primaries)
    return net


def _hybrid():
    net = model_702.Hybrid3Mdn(n_primaries=N_PRIMARIES, n_knots=N_KNOTS)
    net.table = _table(n_primaries=N_PRIMARIES)
    return net


def test_k1_mixer_has_145_inputs_and_hybrid3_has_209():
    """The fine-tuning mixer of K1 reads 145 columns and Hybrid 3's reads 64 more."""
    assert _k1().mixer_ft[0].in_features == 145
    assert _hybrid().mixer_ft[0].in_features == 209


def test_the_new_encoders_have_no_embedding_input_and_the_right_shape():
    """The new weather encoder reads 7 + 14 columns and the new date-time encoder reads 21, both to 32."""
    net = _hybrid()
    assert net.weather2[0].in_features == 21 and net.weather2[-1].out_features == 32
    assert net.datetime2[0].in_features == 21 and net.datetime2[-1].out_features == 32
    assert net.weather2[0].out_features == 64 and net.datetime2[0].out_features == 64


def test_hybrid3_forward_depends_on_the_new_encoders_and_gives_nine_outputs():
    """Changing the new encoders' weights changes the output, so the mixer reads them."""
    net = _hybrid().eval()
    with torch.no_grad():  # the mixer's head starts at zero, which hides every input
        for q in net.mixer_ft.parameters():
            q.normal_(std=0.1)
    x, p = _rows(n=8, n_primaries=N_PRIMARIES)
    out = net(x, p)
    assert out.shape == (8, 9)
    with torch.no_grad():
        for q in net.weather2.parameters():
            q.add_(1.0)
    assert not torch.allclose(net(x, p), out)


def test_new_encoders_are_redrawn_from_the_seed_and_the_global_state_is_untouched():
    """The same seed gives the same new encoders, another seed gives other ones, and the RNG state is kept."""
    a, b, c = _hybrid(), _hybrid(), _hybrid()
    torch.manual_seed(5)
    before = torch.get_rng_state()
    a.reset_new_encoders(seed=1)
    assert torch.equal(torch.get_rng_state(), before)
    b.reset_new_encoders(seed=1)
    c.reset_new_encoders(seed=2)
    assert torch.equal(a.weather2[0].weight, b.weather2[0].weight)
    assert not torch.equal(a.weather2[0].weight, c.weather2[0].weight)


def test_loading_a_pooled_pretraining_into_hybrid3_keeps_the_encoders_and_leaves_the_new_ones_random():
    """The pooled `pre` weights fill K1's parts; only `mixer_ft`, the new encoders, and the knots differ."""
    pre = _k1(stage="pre")
    state = {a: v.clone() for a, v in pre.state_dict().items() if not a.startswith("mixer_ft.")}
    net = _hybrid()
    model_702.load_pretrained_into_hybrid3(model=net, state=state, seed=3)
    assert torch.equal(net.weather[0].weight, pre.weather[0].weight)
    assert torch.equal(net.embedding.weight, pre.embedding.weight)
    assert torch.equal(net.lag[0][0].weight, pre.lag[0][0].weight)
    assert torch.equal(net.knots_old, pre.knots[:, :-1])
    assert torch.equal(net.knot_new, pre.knots[:, -1])


def test_zero_embedding_has_no_parameter_and_returns_zeros():
    """The split-at-1 embedding holds nothing to train and reads as zero."""
    net = _k1(n_primaries=1)
    model_702.zero_primary_embedding(model=net)
    assert not list(net.embedding.parameters())
    assert not any("embedding" in n for n, _ in net.named_parameters())
    assert torch.count_nonzero(net.embedding(torch.zeros(5, dtype=torch.long))) == 0
    x, p = _rows(n=6, n_primaries=1)
    assert net(x, p).shape == (6, 9)


def test_zero_embedding_survives_training_without_gaining_a_parameter():
    """`train` re-enables every parameter at its end; the zero embedding has none to re-enable."""
    net = _k1(n_primaries=1)
    model_702.zero_primary_embedding(model=net)
    x, p = _rows(n=400, n_primaries=1)
    train_702.train(
        model=net,
        x=x,
        z=torch.randn(400),
        w=torch.ones(400),
        p=p,
        is_val=torch.arange(400) % 10 == 0,
        seed=0,
        lr=1e-3,
        max_epochs=2,
        fresh_lr=None,
        frozen=False,
        batch=64,
        device="cpu",
        row_nll=lambda *, out, z, p: (out[:, 0] - z) ** 2,
    )
    assert torch.count_nonzero(net.embedding(torch.zeros(3, dtype=torch.long))) == 0


def _fit(net, *, n=600, fixed=None, max_epochs=3, fresh_lr=None, frozen=True, batch=64, seed=0):
    x, p = _rows(n=n, n_primaries=N_PRIMARIES)
    return train_702.train(
        model=net,
        x=x,
        z=torch.randn(n),
        w=torch.ones(n),
        p=p,
        is_val=torch.arange(n) % 7 == 0,
        seed=seed,
        lr=1e-3,
        max_epochs=max_epochs,
        fresh_lr=fresh_lr,
        frozen=frozen,
        batch=batch,
        device="cpu",
        row_nll=lambda *, out, z, p: (out[:, 0] - z) ** 2,
        fixed_epochs=fixed,
    )


def test_fixed_epochs_run_exactly_that_many_and_keep_the_last_epoch():
    """With `fixed_epochs`, the epoch count equals the argument, whatever the patience would do."""
    epochs, best = _fit(_k1(), fixed=4, max_epochs=1)
    assert epochs == 4 and np.isnan(best)


def test_early_stopping_stops_before_the_epoch_limit_on_a_flat_validation_loss():
    """A constant target the network cannot improve on stops after patience + 1 stale epochs."""
    net = _k1()
    x, p = _rows(n=600, n_primaries=N_PRIMARIES)
    epochs, best = train_702.train(
        model=net,
        x=x,
        z=torch.zeros(600),
        w=torch.ones(600),
        p=p,
        is_val=torch.arange(600) % 7 == 0,
        seed=0,
        lr=0.0,
        max_epochs=40,
        fresh_lr=None,
        frozen=False,
        batch=64,
        device="cpu",
        row_nll=lambda *, out, z, p: (out[:, 0] - z) ** 2,
    )
    assert epochs == 1 + train_702.SETTINGS.patience
    assert np.isfinite(best)


def test_batch_sets_the_number_of_gradient_steps_per_epoch():
    """A step takes `batch` of the 20% of rows an epoch uses, so a smaller batch takes proportionally more steps."""
    steps = {}
    for batch in (50, 800):
        calls = []
        net = _k1()
        x, p = _rows(n=16_000, n_primaries=N_PRIMARIES)

        def nll(*, out, z, p, calls=calls, net=net):
            if net.training:
                calls.append(1)
            return (out[:, 0] - z) ** 2

        train_702.train(
            model=net,
            x=x,
            z=torch.randn(16_000),
            w=torch.ones(16_000),
            p=p,
            is_val=torch.arange(16_000) % 10 == 0,
            seed=0,
            lr=1e-4,
            max_epochs=1,
            fresh_lr=None,
            frozen=False,
            batch=batch,
            device="cpu",
            row_nll=nll,
        )
        steps[batch] = len(calls)
    assert steps[800] == int(np.ceil(0.2 * 14_400 / 800))
    assert steps[50] == int(np.ceil(0.2 * 14_400 / 50))
    assert steps[50] > 10 * steps[800]


def test_frozen_encoders_stay_fixed_and_new_encoders_train_in_both_steps(monkeypatch):
    """Weather and date-time encoders are frozen; the new encoders and the mixer move; groups follow the plan."""
    seen = {}
    real = torch.optim.AdamW

    def spy(groups, **kwargs):
        seen["groups"] = groups
        return real(groups, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    for fresh_lr, expected_groups in ((5e-4, 2), (None, 1)):
        net = _hybrid()
        net.reset_new_encoders(seed=0)
        before = {n: p.detach().clone() for n, p in net.named_parameters()}
        _fit(net, fresh_lr=fresh_lr, frozen=True, max_epochs=2)
        after = dict(net.named_parameters())
        assert len(seen["groups"]) == expected_groups
        assert torch.equal(before["weather.0.weight"], after["weather.0.weight"])
        assert torch.equal(before["datetime.0.weight"], after["datetime.0.weight"])
        assert not torch.equal(before["weather2.0.weight"], after["weather2.0.weight"])
        assert not torch.equal(before["datetime2.0.weight"], after["datetime2.0.weight"])
        assert not torch.equal(before["mixer_ft.0.weight"], after["mixer_ft.0.weight"])
        if fresh_lr is not None:
            fresh = {id(q) for q in seen["groups"][1]["params"]}
            names = {n for n, q in net.named_parameters() if id(q) in fresh}
            assert any(n.startswith("weather2.") for n in names) and any(
                n.startswith("mixer_ft.") for n in names
            )
            assert seen["groups"][1]["lr"] == 5e-4 and seen["groups"][0]["lr"] == 1e-3
        else:
            assert seen["groups"][0]["lr"] == 1e-3


def test_the_loop_settings_equal_experiment_510s():
    """The copied loop's settings are the module constants of experiment 510."""
    c = pytest.importorskip("e00510_common")
    s = train_702.SETTINGS
    assert (s.patience, s.row_fraction, s.weight_decay, s.grad_clip, s.validation_rows) == (
        c.PATIENCE,
        c.ROW_FRACTION,
        c.WEIGHT_DECAY,
        c.GRAD_CLIP,
        c.VALIDATION_ROWS,
    )

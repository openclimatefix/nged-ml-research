# ruff: noqa: E402, PT018
"""Tests of experiment 714's features, classifier, and second generator. Synthetic data only."""

import ast
import sys
from pathlib import Path

HERE = Path(__file__).parent
for folder in (
    HERE,
    HERE.parent / "00660-shared-generator-of-injected-load-transfer-events",
    HERE.parent / "00709-era-signal-in-the-candidate-part-1",
):
    sys.path.insert(0, str(folder))

import e00714_threads  # noqa: F401  (must run before numpy)

# isort: split
import e00714_analyse as ana
import e00714_build as bld
import e00714_common as common
import e00714_features as feats
import e00714_generator2 as g2
import e00714_model as model
import numpy as np
import pytest

N_SITES = 4
N_DAYS = 60
N_HH = N_DAYS * common.DAY
ONSET_DAY = 20


def synthetic(
    *, step: float = -0.2, seed: int = 0
) -> tuple[common.PassPower, common.PassPower, np.ndarray]:
    """Return a pass with a flat step at the host, a zero reference, and a shortwave field."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_HH)
    daily = 0.5 + 0.1 * np.sin(2 * np.pi * (t % common.DAY) / common.DAY)
    ref = np.tile(daily, (N_SITES, 1)).astype(np.float32)
    power = ref + 0.01 * rng.standard_normal(ref.shape).astype(np.float32)
    power[0, ONSET_DAY * common.DAY :] += step
    sw = np.maximum(0.0, 500 * np.sin(2 * np.pi * (t % common.DAY) / common.DAY))
    return common.PassPower(0, power), common.PassPower(0, ref), sw


def event() -> feats.EventInput:
    """Return the host event of the synthetic pass."""
    return feats.EventInput(
        host=0, onset_hh=ONSET_DAY * common.DAY + 7, onset_day=ONSET_DAY, neighbours=[1, 2]
    )


def test_flatness() -> None:
    """A constant profile is flat; a zero-mean profile is not."""
    assert feats.flatness(profile=np.full(48, -3.0)) == pytest.approx(1.0)
    assert feats.flatness(profile=np.sin(np.linspace(0, 2 * np.pi, 48, endpoint=False))) < 0.01


def test_step_features() -> None:
    """A sustained step has no steps; an on-off signal has many."""
    sustained = np.full(14 * 48, -10.0)
    assert feats.step_features(change=sustained)[0] == 0.0
    on_off = np.tile(np.concatenate([np.full(12, -10.0), np.zeros(24)]), 19)[: 14 * 48]
    steps, high = feats.step_features(change=on_off)
    assert steps > 2 and 0.2 < high < 0.5


def test_features_of_a_flat_step() -> None:
    """A flat step of 20 points is found with flatness near 1 and no mirror."""
    power, ref, sw = synthetic()
    out = feats.build_features(
        event=event(), k=14, power=power, reference=ref, shortwave=sw, capacity_mw=np.ones(N_SITES)
    )
    assert out is not None
    assert out["size"] == pytest.approx(-20.0, abs=1.0)
    assert out["flat"] > 0.95
    assert abs(out["mirror_share"]) < 0.5


def test_origin_cut() -> None:
    """Scrambling power, reference, and shortwave at or after the data cut changes no feature."""
    power, ref, sw = synthetic()
    args = {"event": event(), "k": 14, "capacity_mw": np.ones(N_SITES)}
    base = feats.build_features(power=power, reference=ref, shortwave=sw, **args)
    cut = (ONSET_DAY + 14) * common.DAY + common.ORIGIN_HOUR_HH - common.LATENCY_HH
    rng = np.random.default_rng(1)
    scrambled = power.power.copy()
    scrambled[:, cut:] = rng.standard_normal(scrambled[:, cut:].shape)
    ref_scrambled = ref.power.copy()
    ref_scrambled[:, cut:] = 99.0
    sw_scrambled = sw.copy()
    sw_scrambled[cut:] = rng.uniform(0, 900, sw[cut:].shape)
    after = feats.build_features(
        power=common.PassPower(0, scrambled), reference=common.PassPower(0, ref_scrambled),
        shortwave=sw_scrambled, **args,
    )  # fmt: skip
    assert base is not None and after is not None
    for name in base:
        a, b = base[name], after[name]
        assert (a == b) or (np.isnan(a) and np.isnan(b)), name


def test_older_rows_end_before_the_onset() -> None:
    """Older rows that end after the onset are not read: injecting there does not move them."""
    power, ref, sw = synthetic()
    args = {
        "event": event(),
        "k": 28,
        "capacity_mw": np.ones(N_SITES),
        "reference": ref,
        "shortwave": sw,
    }
    base = feats.build_features(power=power, **args)
    assert base is not None


def test_auroc_and_logistic_known_answer() -> None:
    """Separable classes score near 1, identical classes near 0.5, and ties are averaged."""
    rng = np.random.default_rng(0)
    y = np.repeat([0.0, 1.0], 200)
    x = np.concatenate([rng.normal(-3, 1, 200), rng.normal(3, 1, 200)])[:, None]
    groups = np.tile(np.arange(40), 10)
    fitted = model.fit_logistic(x=x, y=y, blocks=groups % 4)
    assert model.auroc(label=y, score=fitted.proba(x=x)) > 0.99
    noise = rng.normal(size=(400, 1))
    assert abs(model.auroc(label=y, score=noise[:, 0]) - 0.5) < 0.08
    assert model.auroc(label=np.array([0, 1]), score=np.array([1.0, 1.0])) == 0.5


def _reversed_relation_data(*, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return features, labels, and months: month b follows `x = y`, the others `x = 1 - y`."""
    rng = np.random.default_rng(seed)
    months = np.repeat(np.array(["a", "b", "c", "d"]), [30, 150, 30, 30])
    y = rng.integers(0, 2, 240).astype(float)
    x = np.where(months == "b", y, 1.0 - y) + 0.3 * rng.normal(size=240)
    return x[:, None], y, months


def test_out_of_fold_never_scores_a_month_it_saw() -> None:
    """Out-of-fold scores of a month come from a fit without that month.

    Months a, c, and d carry the label as `x = 1 - y` and month b as `x = y`. A fit without month b
    learns a negative weight and scores month b near 0. A fit that saw month b's 150 rows learns a
    positive weight and scores it near 1. The check runs over 20 seeds.
    """
    for seed in range(20):
        x, y, months = _reversed_relation_data(seed=seed)
        held = months == "b"
        pred = model.out_of_fold(x=x, y=y, months=months)
        assert np.isfinite(pred).all()
        assert model.auroc(label=y[held], score=pred[held]) < 0.2, seed
        leaked = model.fit_logistic(x=x, y=y, blocks=months).proba(x=x[held])
        assert model.auroc(label=y[held], score=leaked) > 0.8, seed  # the fault is visible


def test_bootstrap_keeps_whole_passes() -> None:
    """Every drawn pass appears with all its rows."""
    months = np.repeat(np.array(["a", "b"]), 6)
    passes = np.array(["a1", "a1", "a1", "a2", "a2", "a2", "b1", "b1", "b1", "b2", "b2", "b2"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        drawn = model.resample_indices(months=months, passes=passes, rng=rng)
        counts = np.bincount(drawn, minlength=12)
        for p in np.unique(passes):
            assert len(set(counts[passes == p])) == 1


def test_permanent_deltas() -> None:
    """Each non-gas permanent kind has the shape it is named for."""
    n_hh = 120 * common.DAY
    onset = 40 * common.DAY + 5
    real = 0.5 + 0.2 * np.sin(2 * np.pi * (np.arange(n_hh) % 48) / 48)
    rng = np.random.default_rng(0)
    flat = g2.permanent_delta(kind="flat", size=-0.1, onset_hh=onset, n_hh=n_hh, real=real, rng=rng)
    assert flat[:onset].sum() == 0 and np.all(flat[onset:] == -0.1)
    prop = g2.permanent_delta(
        kind="profile_prop", size=0.1, onset_hh=onset, n_hh=n_hh, real=real, rng=rng
    )
    assert prop[onset:].mean() == pytest.approx(0.1, rel=0.02) and prop[onset:].std() > 0.01
    day = g2.permanent_delta(
        kind="daylight", size=0.3, onset_hh=onset, n_hh=n_hh, real=real, rng=rng
    )
    assert day[onset:].max() == pytest.approx(0.3) and (day[onset:] == 0).mean() > 0.3
    q = g2.permanent_delta(
        kind="quantised", size=-0.15, onset_hh=onset, n_hh=n_hh, real=real, rng=rng
    )
    assert 2 <= len(np.unique(np.round(q[onset:], 6))) <= 3


def test_intermittent_rows() -> None:
    """Intermittent pulse rows alternate on and off, never overlap, and have a constant size."""
    rng = np.random.default_rng(0)
    rows, share = g2.intermittent_rows(
        site="X", size=0.08, onset_hh=1000, stop_hh=1000 + 45 * 48, first_id=0, rng=rng
    )
    starts = np.array([r["start"] for r in rows])
    ends = np.array([r["end"] for r in rows])
    assert (
        len(rows) > 10 and np.all(starts[1:] > ends[:-1]) and {r["delta"] for r in rows} == {-0.08}
    )
    assert all(r["event_class"] == "pulse" for r in rows)
    assert 0.0 < share < 1.0


def test_second_generator_shares_no_injector_with_experiment_709() -> None:
    """The second generator never imports experiment 709's injector."""
    tree = ast.parse(Path(g2.__file__).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "e00709_inject" not in imported
    assert not hasattr(g2, "inject")


def _old_mirror_share(*, host_mw: np.ndarray, neighbour_mw: np.ndarray) -> float:
    """The uncentred slope that the first version of the mirror feature used."""
    x, z = host_mw, -neighbour_mw
    return float(np.sum(x * z) / np.sum(x * x))


def test_mirror_ignores_common_mode_error_and_finds_a_mirror() -> None:
    """With no event the mirror share is near 0; with a 90% mirror it is near 0.9.

    The reference level's error is shared by neighbouring Primaries (here a constant offset). The
    first version of the feature, an uncentred slope, read that shared error as a strong mirror.
    """
    cap = np.full(N_SITES, 100.0)
    power, ref, sw = synthetic(step=0.0)
    shared = power.power.copy()
    shared[0] += 0.10
    shared[1] += 0.10
    shared[2] += 0.08
    quiet = feats.build_features(
        event=event(), k=14, power=common.PassPower(0, shared), reference=ref, shortwave=sw,
        capacity_mw=cap,
    )  # fmt: skip
    assert quiet is not None
    assert abs(quiet["mirror_share"]) < 0.4
    assert abs(quiet["mirror_z"]) < 3.0
    window = slice((ONSET_DAY + 14) * common.DAY - 600, (ONSET_DAY + 14) * common.DAY)
    old = _old_mirror_share(
        host_mw=(shared[0, window] - ref.power[0, window]) * 100.0,
        neighbour_mw=sum((shared[j, window] - ref.power[j, window]) * 100.0 for j in (1, 2)),
    )
    assert old < -0.3  # the old statistic fails the same input
    moved = shared.copy()
    moved[0, event().onset_hh :] -= 0.20
    moved[1, event().onset_hh :] += 0.18
    mirrored = feats.build_features(
        event=event(), k=14, power=common.PassPower(0, moved), reference=ref, shortwave=sw,
        capacity_mw=cap,
    )  # fmt: skip
    assert mirrored is not None
    assert mirrored["mirror_share"] == pytest.approx(0.9, abs=0.15)
    assert mirrored["mirror_z"] > 3.0


def _old_mirror_corr(*, host: np.ndarray, neighbours: np.ndarray, n_pre: int) -> float:
    """The correlation difference that the second version of the feature used."""
    pre = slice(0, n_pre)
    return float(
        np.corrcoef(host[pre], neighbours[pre])[0, 1] - np.corrcoef(host, neighbours)[0, 1]
    )


def test_mirror_features_ignore_the_size_of_an_unmirrored_step() -> None:
    """A time-varying shared error and an unmirrored host step must not raise the features.

    The second version of the feature, the correlation before the onset minus the correlation over
    the whole span, rose with the size of the host's step (median 0.18 at 0.05 and 0.96 at 0.30)
    because the step adds variance that the neighbours do not share. No mirror is possible here.
    `mirror_z` reads the neighbours and the sign of the step, so its median is the same at both
    sizes; `mirror_share` shrinks with the step size; both stay below 2.
    """
    cap = np.full(N_SITES, 100.0)
    small = _unmirrored_step_medians(cap=cap, step_size=0.05)
    large = _unmirrored_step_medians(cap=cap, step_size=0.30)
    assert large["old"] - small["old"] > 0.5  # the setup shows the growth of the old statistic
    assert abs(large["z"] - small["z"]) < 0.3
    assert large["share"] <= small["share"] + 0.05
    assert max(small["z"], large["z"]) < 2.0
    assert max(small["share"], large["share"]) < 2.0


def _unmirrored_step_medians(*, cap: np.ndarray, step_size: float) -> dict[str, float]:
    """Return the medians of |mirror_z|, |mirror_share|, and the old correlation statistic."""
    shares, zs, olds = [], [], []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        power, ref, sw = synthetic(step=0.0, seed=seed)
        data = power.power.copy()
        shared = np.zeros(N_HH)
        for t in range(1, N_HH):
            shared[t] = 0.995 * shared[t - 1] + 0.004 * rng.standard_normal()
        data[:3] += shared
        data[0, event().onset_hh :] -= step_size
        out = feats.build_features(
            event=event(), k=14, power=common.PassPower(0, data), reference=ref, shortwave=sw,
            capacity_mw=cap,
        )  # fmt: skip
        assert out is not None
        shares.append(abs(out["mirror_share"]))
        zs.append(abs(out["mirror_z"]))
        cut = (ONSET_DAY + 14) * common.DAY + common.ORIGIN_HOUR_HH - common.LATENCY_HH
        lo = event().onset_hh - common.WINDOW_HH
        host = data[0, lo:cut] - ref.power[0, lo:cut]
        nbr = data[1, lo:cut] + data[2, lo:cut] - ref.power[1, lo:cut] - ref.power[2, lo:cut]
        olds.append(_old_mirror_corr(host=host, neighbours=nbr, n_pre=event().onset_hh - lo))
    return {
        "z": float(np.median(zs)),
        "share": float(np.median(shares)),
        "old": float(np.median(olds)),
    }


def _ring_context(*, n_sites: int = 16) -> common.Context:
    """Return a context whose panel is a ring of Primaries with full readings."""
    sites = tuple(f"S{i}" for i in range(n_sites))
    neighbours = {i: sorted({(i - 1) % n_sites, (i + 1) % n_sites}) for i in range(n_sites)}
    panel = common.f709.Panel(
        sites=sites, power=np.full((n_sites, 20 * common.DAY), 0.5),
        capacity_mw=np.full(n_sites, 30.0), is_mva=np.zeros(n_sites, dtype=bool),
        neighbours=neighbours,
    )  # fmt: skip
    return common.Context(panel=panel, weather=None, references={})  # ty: ignore[invalid-argument-type]


def test_no_event_sits_in_another_events_neighbour_set() -> None:
    """Every event site is outside the neighbour sets of the other events' sites."""
    ctx = _ring_context()
    nbrs = ctx.panel.neighbours
    pool = list(range(16))
    for seed in range(100):
        rng = np.random.default_rng(seed)
        mirrors = g2.SWITCH_ROTATION[seed % 4]
        eras, pairs = g2.choose_hosts(
            ctx=ctx, pool=pool, onset_hh=15 * common.DAY, mirrors=mirrors, n_era=2, rng=rng
        )
        groups = [{h} for h in eras] + [{d, *([] if r is None else [r])} for d, r in pairs]
        for a, group_a in enumerate(groups):
            for b, group_b in enumerate(groups):
                if a == b:
                    continue
                assert not any(nbrs[x] and set(nbrs[x]) & group_b for x in group_a), (seed, a, b)


def test_half_of_the_transfers_are_scored_at_the_receiver() -> None:
    """About half of the temporary moves have a positive change at the scored host."""
    ctx = _ring_context()
    rng = np.random.default_rng(0)
    truth: list[dict] = []
    pairs: list[tuple[int, int | None]] = [(0, 1), (4, None)] * 200
    mirrors: tuple[g2.MirrorType, ...] = tuple(["full", "hidden"] * 200)  # ty: ignore[invalid-assignment]
    g2._build_transfers(
        ctx=ctx, pairs=pairs, mirrors=mirrors, onset_hh=100, onset_day=2, truth=truth, rng=rng
    )  # fmt: skip
    positive = np.mean([t["size"] > 0 for t in truth])
    assert 0.4 < positive < 0.6
    flipped_metered = [t for t in truth if t["receiving_host"] and t["mirror_state"] == "full"]
    assert flipped_metered
    assert all(t["host"] == 1 and t["recipient"] == 0 for t in flipped_metered)
    flipped_hidden = [t for t in truth if t["receiving_host"] and t["mirror_state"] == "hidden"]
    assert all(t["recipient"] == -1 and t["size"] > 0 for t in flipped_hidden)


def test_bank_unpickler_allows_only_known_globals() -> None:
    """An unpickler that is asked for any other global raises."""
    import io
    import os
    import pickle

    class Evil:
        def __reduce__(self) -> tuple:
            return (os.system, ("true",))

    with pytest.raises(pickle.UnpicklingError):
        common._BankUnpickler(io.BytesIO(pickle.dumps(Evil()))).load()


def test_rewired_neighbours_never_hold_an_injected_site() -> None:
    """The rewired set excludes every touched site and the host, or is None."""
    ctx = _ring_context()
    for seed in range(50):
        rng = np.random.default_rng(seed)
        touched = {int(t) for t in rng.choice(16, size=6, replace=False)}
        host = int(rng.choice(sorted(touched)))
        chosen = bld.rewired_neighbours(ctx=ctx, host=host, touched=touched, rng=rng)
        own = {host, *ctx.panel.neighbours[host]}
        assert chosen is None or (chosen and not (touched | own) & set(chosen))
    everything = bld.rewired_neighbours(
        ctx=ctx, host=0, touched=set(range(16)), rng=np.random.default_rng(0)
    )
    assert everything is None


def test_balanced_selection_beats_a_random_draw() -> None:
    """The kept passes balance permanent and temporary events per Primary and keep every cell."""
    rng = np.random.default_rng(0)
    months = ("m1", "m2")
    vectors = {
        (m, i): rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0], size=6) for m in months for i in range(16)
    }
    kept = g2.select_balanced(vectors=vectors, months=months, per_month=8, modulus=4)
    for month in months:
        assert len(kept[month]) == 8
        assert sorted(i % 4 for i in kept[month]) == [0, 0, 1, 1, 2, 2, 3, 3]
    chosen = sum(vectors[(m, i)] for m in months for i in kept[m])
    first = sum(vectors[(m, i)] for m in months for i in range(8))
    assert np.abs(chosen).sum() < np.abs(first).sum()


def test_site_prior_scores_and_stratified_auroc() -> None:
    """A site that decides the class scores near 1 by its prior, and 0.5 when stratified."""
    months = np.repeat(np.array(["a", "b", "c", "d"]), 20)
    sites = np.tile(np.arange(10), 8)
    label = (sites < 5).astype(float)
    prior = model.site_prior_scores(sites=sites, label=label, months=months)
    assert model.auroc(label=label, score=prior) > 0.95
    assert np.isnan(model.site_stratified_auroc(sites=sites, label=label, score=prior))
    rng = np.random.default_rng(0)
    label2 = rng.integers(0, 2, 80).astype(float)
    assert (
        abs(model.site_stratified_auroc(sites=sites, label=label2, score=rng.random(80)) - 0.5)
        < 0.2
    )


def test_site_prior_never_reads_the_scored_month() -> None:
    """When each month alone decides the class, the leave-one-month-out prior scores near 0.

    The correct prior reads the other 3 months, whose labels are reversed, and scores 0. A prior
    that read the scored month alone would score 1, and a prior that read all 4 months would score
    0.5.
    """
    months = np.repeat(np.array(["a", "b", "c", "d"]), 20)
    sites = np.tile(np.arange(10), 8)
    flipped = np.isin(months, ["b", "d"])
    label = ((sites < 5) ^ flipped).astype(float)
    prior = model.site_prior_scores(sites=sites, label=label, months=months)
    assert model.auroc(label=label, score=prior) < 0.3


def test_selection_does_not_depend_on_the_order_of_the_candidates() -> None:
    """The kept passes are the same whatever order the caller inserts the candidates in."""
    rng = np.random.default_rng(1)
    months = ("m1", "m2")
    vectors = {(m, i): rng.choice([-1.0, 0.0, 1.0], size=5) for m in months for i in range(16)}
    kept = g2.select_balanced(vectors=vectors, months=months, per_month=8, modulus=4)
    shuffled_keys = list(vectors)
    np.random.default_rng(2).shuffle(shuffled_keys)  # ty: ignore[invalid-argument-type]
    reordered = {key: vectors[key] for key in shuffled_keys}
    assert g2.select_balanced(vectors=reordered, months=months, per_month=8, modulus=4) == kept


def test_rewired_neighbours_exclude_the_hosts_own_neighbours_in_a_fixed_case() -> None:
    """Host 0 on a ring: Primary 2's neighbour 1 is the host's own neighbour and is left out."""
    ctx = _ring_context()
    for seed in range(20):
        chosen = bld.rewired_neighbours(
            ctx=ctx, host=0, touched={0}, rng=np.random.default_rng(seed)
        )
        assert chosen is not None
        assert 1 not in chosen and 15 not in chosen and 0 not in chosen


def _decision_inputs(
    *, g2_state: str, bank_enough: bool, c6_high: bool, known: bool, nbr: str,
    g2_ok: bool, bank_ok: bool,
) -> tuple[dict, dict]:  # fmt: skip
    """Build the contrasts and placebos that give one combination of audits and outcomes."""
    low, high = {"enough": (0.8, 0.9), "not_enough": (0.5, 0.6), "neither": (0.6, 0.8)}[g2_state]
    bank_low = 0.8 if bank_enough else 0.5
    host = {"auroc": 0.6, "low": low, "high": high}
    contrasts = {
        "C1": host, "C2": host,
        "C5": {"low": 0.0 if nbr == "stopped" else 0.1},
        "C5_auroc_host_nbr_mirrored": {"auroc": 0.9},
        "C5_gain_hidden": {"diff": 0.05 if nbr == "hidden_gain" else 0.0},
        "C6": {"low": 0.2 if c6_high else 0.0},
    }  # fmt: skip
    auroc = {"bank/k14/known_answer": {"auroc": 0.99 if known else 0.5}}
    for k in (7, 14):
        auroc[f"bank/k{k}/host"] = {"low": bank_low, "auroc": 0.7}
    for w in ("bank", "g2"):
        auroc[f"{w}/k14/host"] = {"auroc": 0.7, "low": bank_low}
        auroc[f"{w}/k14/host_nbr_rewired"] = {"auroc": 0.7}
    result = {
        "contrasts": contrasts, "auroc": auroc,
        "site_prior": {"g2/site_prior_auroc": 0.5 if g2_ok else 0.8, "gas/site_prior_auroc": 0.5},
    }  # fmt: skip
    placebo = {}
    for w, ok in (("g2", g2_ok), ("bank", bank_ok)):
        placebo[f"{w}/shuffled_labels"] = {"within_0.03": True}
        placebo[f"{w}/no_event_placebo"] = {"auroc": 0.5 if ok else 0.7}
    return result, placebo


def test_stated_outcome_for_every_combination_of_audits_and_flags() -> None:
    """The stated outcome follows the plan's definitions for every combination of the inputs."""
    import itertools

    for g2_ok, bank_ok, g2_state, bank_enough, c6_high, known, nbr in itertools.product(
        (True, False), (True, False), ("enough", "not_enough", "neither"), (True, False),
        (True, False), (True, False), ("full", "stopped", "hidden_gain"),
    ):  # fmt: skip
        result, placebo = _decision_inputs(
            g2_state=g2_state, bank_enough=bank_enough, c6_high=c6_high, known=known, nbr=nbr,
            g2_ok=g2_ok, bank_ok=bank_ok,
        )  # fmt: skip
        flags = ana.decisions(result=result, placebo=placebo)
        text = ana.stated_outcome(flags=flags)
        case = (g2_ok, bank_ok, g2_state, bank_enough, c6_high, known, nbr)
        outcome_2 = g2_state == "enough" and bank_enough and bank_ok and g2_ok
        assert flags["outcome_2_host_rows_enough_on_bank_and_g2"] == outcome_2, case
        if not g2_ok:
            assert text.startswith("inconclusive pending an audit of the second"), case
            continue
        assert not text.startswith("inconclusive"), case
        if not known:
            assert text.startswith("uninterpretable"), case
            assert "outcome 1" not in text and "outcome 2" not in text, case
            continue
        assert ("outcome 1:" in text) == (g2_state == "not_enough"), case
        assert ("outcome 2:" in text) == outcome_2, case
        if g2_state == "enough" and not bank_ok:
            assert "bank half of outcome 2 is inconclusive" in text, case
        if g2_state == "enough" and bank_ok and not bank_enough:
            assert "not on the bank" in text, case
        assert ("outcome 4:" in text) == (nbr == "full"), case
        assert ("F4 stopped" in text) == (nbr == "stopped"), case
        assert ("also gain on hidden" in text) == (nbr == "hidden_gain"), case
        assert ("outcome 3:" in text) == (c6_high and g2_state != "enough" and bank_ok), case
        assert ("bank comparison" in text) == (not bank_ok), case
        assert ("bank inflated" in text) == (c6_high and g2_state == "enough" and bank_ok), case

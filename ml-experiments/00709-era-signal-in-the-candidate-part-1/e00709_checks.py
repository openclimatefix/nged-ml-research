# ruff: noqa: E402, E501, ANN001, ANN201, D103
r"""Experiment 709's checks: identity, anchor, double counting, liveness, signal cut, alignment, weather.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00709-*/e00709_checks.py --synthetic        # CPU, no stored rows
    uv run --no-sync python ml-experiments/00709-*/e00709_checks.py --signals          # CPU, the stored signal tables
    uv run --no-sync python ml-experiments/00709-*/e00709_checks.py --alignment 2026-03  # CPU, builds one fold's rows
    uv run --no-sync python ml-experiments/00709-*/e00709_checks.py --weather ARM ...    # after a fit: origin-time weather

Every mode merges its result into `checks.json` (`synthetic_passed`, `signals_passed`,
`alignment_passed`, `origin_weather_passed`). `e00709_chain.sh` refuses to fit until the first
three are true. No mode fits a stage 3 network on stored rows; the liveness and identity checks
train tiny networks on random inputs.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

_p = argparse.ArgumentParser()
_p.add_argument("--synthetic", action="store_true")
_p.add_argument("--signals", action="store_true")
_p.add_argument("--alignment", default=None, help="a fold month, for example 2026-03")
_p.add_argument("--weather", nargs="*", default=None, help="arms whose stored forecasts to inspect")
_p.add_argument("--fold", default="2026-03", help="the fold of the stored forecasts for --weather")
_p.add_argument("--seed", type=int, default=0)
ARGS = _p.parse_args()
FOLD_FOR_IMPORT: Final = ARGS.alignment or ARGS.fold
# the candidate module reads its own arguments at import; `none` keeps it from reading any signal table
sys.argv = [
    "e00709_cand.py", "fit", "--era", "none", "--fold", FOLD_FOR_IMPORT, "--seed", "0", "--phase",
    "train",
]  # fmt: skip

import e00613_nbmodel as nb  # ty: ignore[unresolved-import]
import e00709_features as features
import e00709_model as me
import e00709_signal as sg
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

CHECKS_PATH: Final[Path] = features.DATA_DIR / "checks.json"
IDENTITY_INIT_TOLERANCE: Final[float] = 1e-6
IDENTITY_TRAINED_TOLERANCE: Final[float] = 1e-5
N_PRIMARIES: Final[int] = 4
N_KNOTS: Final[int] = 5
N_SIGNAL: Final[int] = 5
N_ROWS: Final[int] = 96
N_GRID: Final[int] = 200
RESULTS: dict[str, object] = {}


def record(*, name: str, ok: bool, detail: str = "") -> bool:
    """Print and keep one check's outcome."""
    RESULTS[name] = {"passed": bool(ok), "detail": detail}
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    return bool(ok)


def merge_into_checks(*, flag: str, passed: bool) -> None:
    """Merge this mode's outcome into `checks.json`, keeping the other modes' entries."""
    current = json.loads(CHECKS_PATH.read_text()) if CHECKS_PATH.exists() else {}
    current[flag] = bool(passed)
    current[f"{flag}_details"] = RESULTS
    CHECKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=1, default=str))
    tmp.rename(CHECKS_PATH)
    print(f"{flag} = {passed} -> {CHECKS_PATH}", flush=True)


# ---------------------------------------------------------------- synthetic network checks


def make_pair(*, seed: int, arm: str):
    """Return a `PermLagMdn` and an `EraPermLagMdn` that holds its weights, with one shared table."""
    torch.manual_seed(seed)
    base = nb.PermLagMdn(
        n_primaries=N_PRIMARIES,
        stage="ft",
        n_knots=N_KNOTS,
        n_b1=0,
        inputs=nb.InputSet(n_extra=N_SIGNAL),
    )
    era = me.new_era_model(
        n_primaries=N_PRIMARIES, n_knots=N_KNOTS, n_b1=0, n_signal=N_SIGNAL, arm=arm
    )
    result = era.load_state_dict(base.state_dict())
    assert not result.unexpected_keys
    table = torch.randn(N_PRIMARIES, N_GRID, 28).half()
    base.table = era.table = table  # ty: ignore[invalid-assignment]
    return base, era


def random_rows(*, seed: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return random rows in the candidate's layout (37 lag-free, 8 lag, tau, then signal columns)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N_ROWS, width, generator=g)
    x[:, 43] = torch.randint(0, N_GRID, (N_ROWS,), generator=g).float()
    x[:, 44] = torch.randint(0, N_GRID, (N_ROWS,), generator=g).float()
    x[:, 45] = torch.rand(N_ROWS, generator=g) * (N_KNOTS - 1)
    p = torch.randint(0, N_PRIMARIES, (N_ROWS,), generator=g)
    return x, p


def era_block(
    *, seed: int, active: float, mode: int = 0, anchor: float = 0.0, onset_tau: float = 0.0
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(N_ROWS, sg.ERA_WIDTH)
    out[:, : sg.ENCODER_INPUTS] = torch.randn(N_ROWS, sg.ENCODER_INPUTS, generator=g) * active
    out[:, sg.COL_ACTIVE] = active
    out[:, sg.COL_ANCHOR] = anchor
    out[:, sg.COL_MODE] = mode
    out[:, sg.COL_ONSET_TAU] = onset_tau
    return out


def check_identity_at_init() -> bool:
    ok = True
    for arm in ("all", "stat", "oracle-enc"):
        base, era = make_pair(seed=1, arm=arm)
        base.eval()
        era.eval()
        x, p = random_rows(seed=2, width=base.label_col + N_SIGNAL)
        with torch.no_grad():
            ref = base(x, p)
            absent = era(x, p)
            zero = era(torch.cat([x, era_block(seed=3, active=0.0)], dim=1), p)
            live = era(torch.cat([x, era_block(seed=3, active=1.0)], dim=1), p)
        gaps = {
            "absent": float((absent - ref).abs().max()),
            "zero": float((zero - ref).abs().max()),
            "live_encoder_but_no_anchor": float((live - ref).abs().max()),
        }
        ok &= record(
            name=f"identity_at_init[{arm}]",
            ok=max(gaps.values()) < IDENTITY_INIT_TOLERANCE,
            detail=str(gaps),
        )
    return ok


def train_steps(*, model: nn.Module, x, p, z, steps: int, seed: int) -> None:
    """Train with the candidate's recipe in miniature: encoders frozen, AdamW, the model's penalty."""
    torch.manual_seed(seed)
    frozen = {id(q) for part in (model.weather, model.datetime) for q in part.parameters()}  # ty: ignore[unresolved-attribute]
    for part in (model.weather, model.datetime):  # ty: ignore[unresolved-attribute]
        for q in part.parameters():  # ty: ignore[unresolved-attribute]
            q.requires_grad_(False)
    params = [q for q in model.parameters() if id(q) not in frozen]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    model.train()
    model.weather.eval()  # ty: ignore[unresolved-attribute]
    model.datetime.eval()  # ty: ignore[unresolved-attribute]
    for _ in range(steps):
        out = model(x, p)  # ty: ignore[call-non-callable]
        nll = nb.aux.base.mixture_nll(out=out, z=z, components=3)
        loss = nll.mean() + (model.penalty() if hasattr(model, "penalty") else 0.0)  # ty: ignore[call-non-callable]
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


def check_identity_after_training() -> bool:
    ok = True
    for arm in ("all", "stat", "oracle-enc"):
        base, era = make_pair(seed=4, arm=arm)
        x, p = random_rows(seed=5, width=base.label_col + N_SIGNAL)
        z = torch.randn(N_ROWS) * 0.5
        xe = torch.cat([x, era_block(seed=6, active=0.0)], dim=1)
        train_steps(model=base, x=x, p=p, z=z, steps=25, seed=7)
        train_steps(model=era, x=xe, p=p, z=z, steps=25, seed=7)
        with torch.no_grad():
            gap = float((era(xe, p) - base(x, p)).abs().max())
        ok &= record(
            name=f"identity_after_training_on_none_rows[{arm}]",
            ok=gap < IDENTITY_TRAINED_TOLERANCE,
            detail=f"max gap {gap:.2e}",
        )
    return ok


def check_anchor_moves_output() -> bool:
    """A known anchor moves the three means by exactly the anchor and leaves logits and scales alone."""
    ok = True
    base, era = make_pair(seed=8, arm="all")
    x, p = random_rows(seed=9, width=base.label_col + N_SIGNAL)
    for mode, anchor in ((1, 0.4), (1, -0.25), (0, 0.4)):
        e0 = era_block(seed=10, active=1.0, mode=0, anchor=anchor)
        e1 = era_block(seed=10, active=1.0, mode=mode, anchor=anchor)
        with torch.no_grad():
            before = era(torch.cat([x, e0], dim=1), p)
            after = era(torch.cat([x, e1], dim=1), p)
        diff = after - before
        want = anchor if mode == 1 else 0.0
        means_ok = float((diff[:, 3:6] - want).abs().max()) < 1e-6
        rest_ok = float(diff[:, :3].abs().max()) < 1e-7 and float(diff[:, 6:].abs().max()) < 1e-7
        ok &= record(
            name=f"anchor_moves_means[mode {mode}, anchor {anchor}]",
            ok=means_ok and rest_ok,
            detail=f"mean shift {float(diff[:, 3].mean()):.4f}",
        )
    return ok


def check_no_double_counting() -> bool:
    """An era already carried by the offset knots must not be added a second time.

    The knots move by the step between the onset and the newest knot. A mode 2 row (confirmed after
    the cut) adds `clip(full - moved)`, so the total shift against the same network without the
    knot move and without the anchor is the step once, never twice. A mode 0 row adds nothing.
    """
    ok = True
    base, era = make_pair(seed=11, arm="all")
    x, p = random_rows(seed=12, width=base.label_col + N_SIGNAL)
    x[:, 45] = 2.0
    step = 0.4
    onset_tau = 1.0
    zero_block = era_block(seed=13, active=1.0, mode=0, anchor=step, onset_tau=onset_tau)
    mode2 = era_block(seed=13, active=1.0, mode=2, anchor=step, onset_tau=onset_tau)
    with torch.no_grad():
        plain = era(torch.cat([x, zero_block], dim=1), p)
        # knots 0 and 1 stay at 0, knots 2 to 4 carry the step: the newest knot minus the onset knot is `step`
        era.knots_old[:, 2:, :] = step  # ty: ignore[invalid-assignment]
        era.knot_new[:] = step  # ty: ignore[invalid-assignment]
        carried = era(torch.cat([x, zero_block], dim=1), p)
        carried_mode2 = era(torch.cat([x, mode2], dim=1), p)
    knots_only = float((carried[:, 3:6] - plain[:, 3:6]).mean())
    total = float((carried_mode2[:, 3:6] - plain[:, 3:6]).mean())
    ok &= record(
        name="double_counting[knots carry the step]",
        ok=abs(knots_only - step) < 1e-5,
        detail=f"knots shift {knots_only:.4f}",
    )
    ok &= record(
        name="double_counting[mode 2 adds nothing on top]",
        ok=abs(total - step) < 1e-5,
        detail=f"total shift {total:.4f}, step {step}",
    )
    # the offset moved by less than the step: the anchor adds only the difference
    with torch.no_grad():
        era.knots_old[:, 2:, :] = step / 2  # ty: ignore[invalid-assignment]
        era.knot_new[:] = step / 2  # ty: ignore[invalid-assignment]
        half = era(torch.cat([x, mode2], dim=1), p)
        half_plain = era(torch.cat([x, zero_block], dim=1), p)
    adds = float((half[:, 3:6] - half_plain[:, 3:6]).mean())
    ok &= record(
        name="double_counting[mode 2 adds the missing half]",
        ok=abs(adds - step / 2) < 1e-5,
        detail=f"added {adds:.4f}",
    )
    # the offset fell: the anchor is capped at the full step
    with torch.no_grad():
        era.knots_old[:, 2:, :] = -0.3  # ty: ignore[invalid-assignment]
        era.knot_new[:] = -0.3  # ty: ignore[invalid-assignment]
        fell = era(torch.cat([x, mode2], dim=1), p)
        fell_plain = era(torch.cat([x, zero_block], dim=1), p)
    capped = float((fell[:, 3:6] - fell_plain[:, 3:6]).mean())
    ok &= record(
        name="double_counting[the anchor is capped at the full step]",
        ok=abs(capped - step) < 1e-5,
        detail=f"added {capped:.4f}",
    )
    return ok


def check_oenc_liveness_and_frozen() -> bool:
    """O-enc's layers move in one epoch of the candidate's training loop, and the frozen encoders do not."""
    import e00709_cand as cand

    ok = True
    _base, era = make_pair(seed=14, arm="oracle-enc")
    x, p = random_rows(seed=15, width=era.label_col + N_SIGNAL)
    live = era_block(seed=16, active=1.0)
    live[:, sg.COL_MODE] = sg.MODE_FULL
    live[:, sg.COL_ANCHOR] = 0.3
    xe = torch.cat([x, live], dim=1)
    # a target that depends on the era inputs, so the vector carries a signal
    z = (live[:, 3] * 0.8).detach()
    before = {n: q.detach().clone() for n, q in era.named_parameters()}
    is_val = torch.zeros(N_ROWS, dtype=torch.bool)
    is_val[-16:] = True
    cand.train_curriculum(
        model=era, x=xe, z=z, w=torch.ones(N_ROWS), p=p, is_val=is_val, seed=0, lr=1e-2, max_epochs=3,
        fresh_lr=None, frozen=True, anchor=None, lam=0.0, device="cpu",
    )  # fmt: skip
    moved = {n: float((q.detach() - before[n]).abs().max()) for n, q in era.named_parameters()}
    for name in ("oenc_weather.weight", "oenc_datetime.weight"):
        ok &= record(
            name=f"oenc_moves[{name}]",
            ok=moved[name] > 0.0,
            detail=f"largest change {moved[name]:.2e}",
        )
    frozen_moved = {
        n: v for n, v in moved.items() if n.startswith(("weather.", "datetime.")) and v > 0.0
    }
    ok &= record(
        name="encoders_stay_frozen", ok=not frozen_moved, detail=str(frozen_moved) or "none moved"
    )
    registered = {n for n, _ in era.named_parameters()}
    inside = {n for n in registered if n.startswith(("weather.oenc", "datetime.oenc"))}
    ok &= record(
        name="oenc_outside_the_frozen_modules",
        ok=not inside and "oenc_weather.weight" in registered,
        detail=f"inside: {sorted(inside)}",
    )
    return ok


def check_oenc_off_on_lag_side() -> bool:
    """With O-enc weights set non-zero, the lag side's expected-input embeddings are unchanged."""
    _base, era = make_pair(seed=17, arm="oracle-enc")
    era.eval()
    x, p = random_rows(seed=18, width=era.label_col + N_SIGNAL)
    emb = era.embedding(p)
    with torch.no_grad():
        era.oenc_weather.weight.normal_()  # ty: ignore[unresolved-attribute]
        era.oenc_datetime.weight.normal_()  # ty: ignore[unresolved-attribute]
        era._vec = torch.randn(N_ROWS, me.ERA_DIM)
        era._oenc_on = True
        got = era._expected(x=x, primary=p, emb=emb)
        era._vec, era._oenc_on = None, False
        want = era._expected(x=x, primary=p, emb=emb)
    gap = max(
        float((a - b).abs().max())
        for ga, gb in zip(got, want, strict=True)
        for a, b in zip(ga, gb, strict=True)
    )
    return record(name="oenc_off_on_the_lag_side", ok=gap == 0.0, detail=f"max gap {gap:.2e}")


def run_synthetic() -> bool:
    ok = check_identity_at_init()
    ok &= check_identity_after_training()
    ok &= check_anchor_moves_output()
    ok &= check_no_double_counting()
    ok &= check_oenc_liveness_and_frozen()
    ok &= check_oenc_off_on_lag_side()
    ok &= check_translation_example()
    return ok


# ---------------------------------------------------------------- the signal tables


def check_translation_example() -> bool:
    """A known band change translates the lags before the onset by the band's change, and only those."""
    power = np.zeros(features.HALF_HOURS_PER_DAY * 100)
    bands = np.array([0.1, 0.2, 0.3, 0.4])
    out = sg.translate_panel_row(
        power=power, onset_day=50, era_scale=1.0, bands=bands, switch_scale=0.0, depth=0.0
    )
    hh = np.arange(features.HALF_HOURS_PER_DAY * 100)
    band_of = sg.BAND_OF_HALF_HOUR[hh % features.HALF_HOURS_PER_DAY]
    before = hh < 50 * features.HALF_HOURS_PER_DAY
    want = np.where(before & (hh >= (50 - 60) * features.HALF_HOURS_PER_DAY), bands[band_of], 0.0)
    ok = bool(np.allclose(out, want))
    sw = sg.translate_panel_row(
        power=power + 1.0, onset_day=50, era_scale=0.0, bands=bands, switch_scale=1.0, depth=0.3
    )
    ok &= bool(np.allclose(sw[~before], 0.7) and np.allclose(sw[before], 1.0))
    both = sg.translate_panel_row(
        power=power, onset_day=50, era_scale=0.5, bands=bands, switch_scale=0.5, depth=0.2
    )
    ok &= bool(np.allclose(both[~before], -0.1) and np.allclose(both[before], 0.5 * want[before]))
    return record(name="translation_known_answer", ok=ok)


def perturb_from(*, signal: sg.Signal, first_day: int, seed: int) -> sg.Signal:
    """Return the signal with every entry from `first_day` on replaced by random values."""
    rng = np.random.default_rng(seed)
    cols = {k: v.copy() for k, v in signal.columns.items()}
    for name, values in cols.items():
        tail = values[:, first_day:]
        if name == "state":
            values[:, first_day:] = rng.integers(0, 4, tail.shape)
        elif name in {"onset_day", "confirm_day"}:
            values[:, first_day:] = rng.integers(first_day, first_day + 200, tail.shape)
        elif name in {"withdrawn", "era_id"}:
            values[:, first_day:] = rng.integers(0, 3, tail.shape)
        else:
            values[:, first_day:] = rng.normal(size=tail.shape)
    return sg.Signal(arm=signal.arm, causal=signal.causal, columns=cols, episodes=signal.episodes)


def check_signal_arm(*, arm: str) -> bool:
    signal = sg.read_signal(arm=arm)
    ok = signal.causal == (arm not in ("oracle", "placebo"))
    rng = np.random.default_rng(0)
    n = 4000
    site = rng.integers(0, signal.n_sites, n)
    day = rng.integers(400, signal.n_days - 40, n)
    cut = int(day.min()) + 200
    std = np.ones(signal.n_sites)
    tau = np.linspace(0, 80, signal.n_days)
    slots = sg.era_slots(signal=signal, cut_day=cut) if arm == "stat" else None
    ref = sg.era_block(
        signal=signal,
        site=site,
        origin_day=day,
        cut_day=cut,
        target_std=std,
        tau_of_day=tau,
        slots=slots,
    )
    ok_cut = True
    if signal.causal:
        for first in (1, 7, 30):
            changed = perturb_from(signal=signal, first_day=int(day.max()) + 1, seed=first)
            alt = sg.era_block(
                signal=changed,
                site=site,
                origin_day=day,
                cut_day=cut,
                target_std=std,
                tau_of_day=tau,
                slots=slots,
            )
            ok_cut &= bool(np.array_equal(ref, alt, equal_nan=True))
        # an origin on day D reads entry D - 1: perturbing entry D itself must not change it
        test_days = [int(d) for d in np.unique(np.quantile(day, [0.2, 0.5, 0.8]).astype(int))]
        ok_cut &= all(
            (day == d).any() for d in test_days
        )  # fails if no sampled row hits a test day
        for d in test_days:
            changed = perturb_from(signal=signal, first_day=d, seed=d)
            alt = sg.era_block(
                signal=changed,
                site=site,
                origin_day=day,
                cut_day=cut,
                target_std=std,
                tau_of_day=tau,
                slots=slots,
            )
            ok_cut &= bool(np.array_equal(ref[day <= d], alt[day <= d], equal_nan=True))
        hold = sg.hold_mask(signal=signal, cut_day=cut)
        later = perturb_from(signal=signal, first_day=cut, seed=3)
        ok_cut &= bool(np.array_equal(hold, sg.hold_mask(signal=later, cut_day=cut)))
        tr = sg.translation_rows(signal=signal, site=site, origin_day=day)
        tr2 = sg.translation_rows(
            signal=perturb_from(signal=signal, first_day=int(day.max()) + 1, seed=9),
            site=site,
            origin_day=day,
        )
        ok_cut &= bool(np.array_equal(tr.row, tr2.row)) and bool(np.allclose(tr.bands, tr2.bands))
    encoded = ref[:, sg.COL_MODE]
    ok_modes = set(np.unique(encoded)) <= {0.0, 1.0, 2.0}
    ok_width = ref.shape[1] == sg.ERA_WIDTH and bool(np.isfinite(ref).all())
    return record(
        name=f"signal_cut[{arm}]", ok=bool(ok and ok_cut and ok_modes and ok_width),
        detail=f"causal={signal.causal} flag_ok={ok} cut_ok={ok_cut} modes={sorted(set(encoded.tolist()))}",
    )  # fmt: skip


def run_signals() -> bool:
    ok = True
    for arm in sg.SIGNAL_ARMS:
        path = sg.SIGNALS_DIR / f"{arm}.parquet"
        if not path.exists():
            ok &= record(
                name=f"signal_cut[{arm}]",
                ok=False,
                detail=f"{path} is missing: run e00709_signal.py first",
            )
            continue
        ok &= check_signal_arm(arm=arm)
    ok &= check_as_of_alignment()
    return ok


def check_as_of_alignment() -> bool:
    """The era block of an origin on day D decodes entry D - 1 (a signal whose age column holds its own day)."""
    n_sites, n_days = 3, 300
    cols: dict[str, np.ndarray] = {
        k: np.zeros((n_sites, n_days), dtype=np.int64 if k in sg.INT_COLUMNS else np.float64)
        for k in sg.COLUMN_NAMES
    }
    cols["state"][:] = sg.STATE_ERA
    cols["age"] = np.tile(np.arange(n_days, dtype=float), (n_sites, 1))
    signal = sg.Signal(arm="all", causal=True, columns=cols, episodes=sg._episode_frame(rows=[]))
    site = np.array([0, 1, 2, 1])
    origin = np.array([10, 20, 30, 299])
    block = sg.era_block(
        signal=signal,
        site=site,
        origin_day=origin,
        cut_day=1000,
        target_std=np.ones(n_sites),
        tau_of_day=np.zeros(n_days),
    )
    decoded = np.rint(np.expm1(block[:, 2] * np.log1p(sg.AGE_CAP_DAYS)))
    ok = bool(np.array_equal(decoded, origin - 1))
    return record(
        name="as_of_is_day_before_origin",
        ok=ok,
        detail=f"decoded {decoded.tolist()} for origins {origin.tolist()}",
    )


# ---------------------------------------------------------------- alignment on one fold's rows


def check_bank_lag_columns(*, cand, test) -> bool:
    """A host's lag column 37 on the injected panel equals the clean lag before the onset and differs after it.

    The bank passes `lag_x` built from the injected panel, as experiment 693 does. Without it, the
    host's lag columns would read the clean panel and the bank would show no step to the model.
    """
    ctx, _models, _stage0 = cand.stage0_context()
    days = test.origin_idx // cand.PER_DAY
    onset_day = int(np.median(days))
    injected = cand.inject.build_pass(
        panel=ctx.panel, weather_shortwave=ctx.weather.shortwave, ready_days=ctx.ready_days,
        verified_days=ctx.verified_days, world="bank", pass_index=0, onset_day=onset_day,
    )  # fmt: skip
    clean = cand.e.c.clean_stack()
    n_hh = ctx.panel.power.shape[1]
    delta = np.where(
        np.isfinite(injected.power) & np.isfinite(ctx.panel.power),
        injected.power - ctx.panel.power, 0.0,
    )  # fmt: skip
    lag_power = clean.copy()
    lag_power[:, :n_hh] += delta
    blk_clean = cand.e.block_of(
        clean=clean, primary=test.primary, target_idx=test.target_idx, origin_idx=test.origin_idx
    )
    blk_world = cand.e.block_of(
        clean=lag_power, primary=test.primary, target_idx=test.target_idx,
        origin_idx=test.origin_idx,
    )  # fmt: skip
    ok, details = True, []
    for ev in injected.eras:
        host = test.primary == ev.host
        onset = ev.onset_hh // cand.PER_DAY
        before = host & (days < onset)
        after = host & (days >= onset + 2)
        if not before.any() or not after.any():
            continue
        gap = np.abs(blk_world[:, 0] - blk_clean[:, 0])
        valid = ~np.isnan(gap)
        same_before = bool(np.all(gap[before & valid] < 1e-6))
        moved_after = float(np.nanmedian(gap[after & valid]))
        ok &= same_before and moved_after > 1e-3
        details.append(
            f"host {ev.host}: before equal {same_before}, median gap after {moved_after:.3f}"
        )
    ok &= bool(details)  # fails when no era host has rows on both sides of its onset
    return record(name="bank_lag_column_37_reads_injected_panel", ok=ok, detail="; ".join(details))


def run_alignment(*, month: str) -> bool:
    """Row alignment on real rows: the Primary order, the origin day of training and forecast rows."""
    import e00709_cand as cand

    ok = True
    labels = list(cand.k578.sites()["label"])
    ok &= record(
        name="primary_order[k578 sites == features.PRIMARIES]",
        ok=labels == list(features.PRIMARIES),
        detail=f"{labels[:4]}... vs {list(features.PRIMARIES)[:4]}...",
    )
    clean = cand.e.c.clean_stack()
    mr = cand.build_month_syn(month=month, clean=clean)
    test = cand.e.build_test(month=month, standardiser=mr.base.standardiser)
    train_day = cand.origin_idx_of(origin_us=mr.origin_us) // cand.PER_DAY
    stamp_day = ((mr.origin_us - cand.GRID) // cand.DAY_US).astype(np.int64)
    ok &= record(
        name="training_origin_day",
        ok=bool(np.array_equal(train_day, stamp_day)),
        detail=f"{len(train_day):,} rows",
    )
    ok &= record(
        name="training_origin_before_cut",
        ok=bool((train_day < cand.CUT_DAY).all()),
        detail=f"latest origin day {int(train_day.max())}, cut day {cand.CUT_DAY}",
    )
    if test is not None:
        test_day = test.origin_idx // cand.PER_DAY
        want = ((np.asarray(test.origin) - cand.GRID) // cand.DAY_US).astype(np.int64)
        ok &= record(
            name="forecast_origin_day",
            ok=bool(np.array_equal(test_day, want)),
            detail=f"{len(test_day):,} rows",
        )
        # the training phase returns the previous month's rows (the noise calibration), the predict phase the month's own
        first = cand.CUT_DAY if cand.fold.PHASE == "predict" else cand.CUT_DAY - 32
        ok &= record(
            name="forecast_origin_in_window",
            ok=bool(((test_day >= first) & (test_day < first + 32)).all()),
            detail=f"phase {cand.fold.PHASE}: days {int(test_day.min())} to {int(test_day.max())}, cut day {cand.CUT_DAY}",
        )
        ok &= record(name="target_after_origin", ok=bool((test.target_idx > test.origin_idx).all()))
        good, example = check_weather_runs(origins_us=np.asarray(test.origin))
        ok &= record(name="origin_time_weather_of_test_rows", ok=good, detail=example)
        ok &= check_bank_lag_columns(cand=cand, test=test)
    ok &= record(
        name="tau_of_cut_is_newest_knot",
        ok=bool(
            abs(float(cand.tau_of_days(n_days=cand.CUT_DAY + 1)[cand.CUT_DAY]) - (cand.e.KNOTS - 1))
            < 1e-6
        ),
        detail=f"knots {cand.e.KNOTS}",
    )
    return ok


# ---------------------------------------------------------------- origin-time weather on stored forecasts


def check_weather_runs(*, origins_us: np.ndarray) -> tuple[bool, str]:
    """For every origin, the weather run is 00:00 UTC of the origin's day, its demand frame and summary file exist.

    Args:
        origins_us: Origin times in microseconds since the epoch.

    Returns:
        Whether every origin passes, and one example.
    """
    from datetime import UTC, datetime

    import e00709_cand as cand

    pub = cand.e.common.PUBLICATION_DELAY
    ok, example = True, ""
    for o in np.unique(origins_us):
        origin = datetime.fromtimestamp(int(o) / 1e6, UTC)
        run = origin - pub
        ok &= run.hour == 0 and run.minute == 0 and run.date() == origin.date()
        frame = pl.read_parquet(cand.e.common.demand_frame_path(run=run))
        ok &= bool((frame["nwp_init_time"] == run).all()) and bool(
            (frame["power_fcst_init_time"] == origin).all()
        )
        ok &= (cand.e.prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet").exists()
        example = f"origin {origin}, weather run {run}"
    return bool(ok), example


def run_weather(*, arms: list[str], month: str, seed: int) -> bool:
    """Every stored forecast's weather run is the 00:00 UTC run of the origin day; print one example row per arm."""
    import e00667_export as exp  # ty: ignore[unresolved-import]

    ok = True
    for arm in arms:
        folder = (
            features.DATA_DIR
            / "fits"
            / f"{arm.replace('-', '_')}_{month}_s{seed}"
            / "work"
            / "fits"
            / "P5"
        )
        files = sorted(folder.glob(f"{month}_d0_s{seed}.parquet"))
        if not files:
            ok &= record(
                name=f"origin_weather[{arm}]", ok=False, detail=f"no stored forecast in {folder}"
            )
            continue
        raw = pl.read_parquet(
            files[0]
        )  # the network origins (09:00), before the contract relabel to 12:00
        rows = exp.contract_rows(mean=raw)
        origins = raw["power_fcst_init_time"].dt.epoch("us").to_numpy()
        good, example = check_weather_runs(origins_us=origins)
        good &= bool((rows["nwp_init_time"] <= rows["power_fcst_init_time"]).all())
        ok &= record(name=f"origin_weather[{arm}]", ok=good, detail=example)
    return ok


def main() -> None:
    if ARGS.synthetic:
        merge_into_checks(flag="synthetic_passed", passed=run_synthetic())
    if ARGS.signals:
        merge_into_checks(flag="signals_passed", passed=run_signals())
    if ARGS.alignment:
        merge_into_checks(flag="alignment_passed", passed=run_alignment(month=ARGS.alignment))
    if ARGS.weather is not None:
        merge_into_checks(
            flag="origin_weather_passed",
            passed=run_weather(arms=ARGS.weather, month=ARGS.fold, seed=ARGS.seed),
        )


if __name__ == "__main__":
    main()

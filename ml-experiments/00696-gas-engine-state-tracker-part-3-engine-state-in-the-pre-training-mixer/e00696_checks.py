# ruff: noqa: E501, PT018, PLR0915, B023
r"""Experiment 696's checks: synthetic leakage, alignment, identity, liveness, and fit-time gates.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00696-*/e00696_checks.py --write   # synthetic checks (no GPU, no stored data)
    uv run --no-sync python ml-experiments/00696-*/e00696_checks.py --real 2026-02   # leakage and file checks on stored data
    uv run --no-sync python ml-experiments/00696-*/e00696_checks.py --fits   # after the fits: identity control, G0, G0W, liveness, alignment

The synthetic checks write nothing but `checks.json`. The chain script refuses to fit a network until
`checks.json` holds `synthetic_passed: true`. The real-series checks read experiment 695's stored states
and write their results into `checks.json`. The fit-time gates read the fits and the per-run check files
that `e00696_k1.py` wrote.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00695_checks as c695  # ty: ignore[unresolved-import]
import e00695_tracker as tracker  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00695_common import FOLD_MONTHS, READING_DELAY, fold_refit, grid_index  # ty: ignore[unresolved-import]
from e00696_common import DATA_DIR, E695_DIR, N_STATE
from e00696_model import ORIGINAL_CLASS, StateAuxLagMdn
from e00696_probe import fit_probe
from e00696_state import permute_days, rows_state, state_path, tracked_state, unit_vector

CHECKS_PATH: Final[Path] = DATA_DIR / "checks.json"
IDENTITY_TOLERANCE: Final[float] = 1e-4
"""Largest gap, in fractions of capacity, between arm SZ's (or G0's) forecast and experiment 695's A0."""
N_KNOTS: Final[int] = 2
N_B1: Final[int] = 4
N_ROWS: Final[int] = 64


def _models(*, stage: str, seed: int = 1) -> tuple[torch.nn.Module, StateAuxLagMdn]:
    """Return experiment 591's network and the widened network, built under the same seed."""
    torch.manual_seed(seed)
    original = ORIGINAL_CLASS(n_primaries=3, stage=stage, n_knots=N_KNOTS, n_b1=N_B1, kind="shared")
    torch.manual_seed(seed)
    widened = StateAuxLagMdn(n_primaries=3, stage=stage, n_knots=N_KNOTS, n_b1=N_B1, kind="shared")
    for model in (original, widened):
        model.table = torch.zeros(3, 4, 28)  # ty: ignore[invalid-assignment]
    return original, widened


def _batch(*, label_col: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random input without lags (index columns -1) and its Primary index."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(N_ROWS, label_col, generator=generator)
    x[:, 43:45] = -1.0
    x[:, 45] = 0.0
    return x, torch.arange(N_ROWS) % 3


def check_widening_identity() -> float:
    """Return the largest gap between the widened network at zero state and experiment 591's network.

    Both are built under one seed. Every parameter of the original must appear unchanged in the
    widened network, except the first layer of `mixer_pre`, whose original columns must be unchanged
    and whose 3 new columns must be zero. The forward pass of a stage `pre` network with and without
    state columns of zeros must equal the original's.
    """
    worst = 0.0
    for stage in ("pre", "ft"):
        original, widened = _models(stage=stage)
        base, wide = original.state_dict(), widened.state_dict()
        assert base.keys() == wide.keys(), set(base) ^ set(wide)
        for name, value in base.items():
            if name == "mixer_pre.0.weight":
                assert wide[name].shape[1] == value.shape[1] + N_STATE
                assert torch.equal(wide[name][:, :-N_STATE], value), "an original column moved"
                assert torch.count_nonzero(wide[name][:, -N_STATE:]) == 0, (
                    "a new column is not zero"
                )
            else:
                worst = (
                    max(worst, float((wide[name] - value).abs().max()))
                    if value.is_floating_point()
                    else worst
                )
                assert torch.equal(wide[name], value), (
                    f"{stage}: {name} differs from experiment 591's"
                )
    original, widened = _models(stage="pre")
    original.eval(), widened.eval()
    x, p = _batch(label_col=original.label_col)  # ty: ignore[invalid-argument-type]
    with torch.no_grad():
        reference = original(x, p)
        without = widened(x, p)
        zeros = widened(torch.cat([x, torch.zeros(N_ROWS, N_STATE)], dim=1), p)
    return max(
        worst, float((reference - without).abs().max()), float((reference - zeros).abs().max())
    )


def check_state_reaches_the_mixer() -> float:
    """Return the output change when the state columns change, with new weights set to 1."""
    _, widened = _models(stage="pre")
    widened.eval()
    with torch.no_grad():
        widened.mixer_pre[0].weight[:, -N_STATE:] = 1.0  # ty: ignore[not-subscriptable, unresolved-attribute]
        widened.mixer_pre[
            -1
        ].weight.normal_()  # the head starts near zero, which would hide any change  # ty: ignore[not-subscriptable, unresolved-attribute]
        x, p = _batch(label_col=widened.label_col)
        off = widened(torch.cat([x, torch.zeros(N_ROWS, N_STATE)], dim=1), p)
        on = widened(torch.cat([x, torch.ones(N_ROWS, N_STATE)], dim=1), p)
    change = float((on - off).abs().max())
    assert change > 1e-3, "the state columns do not reach the mixer"
    return change


def check_forecast_time_rejects_state() -> None:
    """Assert that a stage 2, stage 3, or forecast-time pass raises when its input carries state columns."""
    _, widened = _models(stage="ft")
    x, p = _batch(label_col=widened.label_col)
    for width in (N_STATE,):
        try:
            widened(torch.cat([x, torch.zeros(N_ROWS, width)], dim=1), p)
        except ValueError:
            continue
        msg = "a stage ft pass accepted state columns"
        raise AssertionError(msg)
    out = widened(x, p)  # the original layout passes
    assert out.shape[1] == 9


def check_liveness_on_synthetic_data() -> dict[str, float]:
    """Train a tiny stage `pre` network on a target that depends on the state, and on one that does not.

    Returns the largest absolute weight on the first two state columns for each case. With a
    state-dependent target the weights must become non-zero. With zero state columns they must stay
    exactly zero, because the extra weights get no gradient and AdamW's decay of a zero is zero.
    """
    result = {}
    for label, scale in (("state_dependent", 1.0), ("zero_state", 0.0)):
        _, model = _models(stage="pre")
        x, p = _batch(label_col=model.label_col)
        state = torch.rand(N_ROWS, N_STATE) * scale
        state[:, 2] = 0.0
        target = 3.0 * state[:, 0] + 2.0 * state[:, 1]
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
        model.train()
        for _ in range(30):
            loss = ((model(torch.cat([x, state], dim=1), p)[:, 0] - target) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        weight = model.mixer_pre[0].weight.detach()[:, -N_STATE:]  # ty: ignore[not-subscriptable, unresolved-attribute]
        result[label] = float(weight[:, :2].abs().max())
        assert float(weight[:, 2].abs().max()) == 0.0, (
            "the hidden-flag weight moved with a zero flag"
        )
    assert result["state_dependent"] > 0.0, "a state weight stayed at zero"
    assert result["zero_state"] == 0.0, "a state weight moved with zero inputs"
    return result


def check_probe_on_synthetic_features() -> dict[str, float]:
    """Fit the probe on a linear target of random features; return its error against the mean."""
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(6_000, 8, generator=generator)
    target = (
        5.0 * features[:, 0]
        - 3.0 * features[:, 1]
        + 0.1 * torch.randn(6_000, generator=generator)
        + 10.0
    )
    mae = fit_probe(features=features, target=target, n_train=4_000, n_stop=1_000, seed=0)
    baseline = float((target[5_000:] - target[:4_000].mean()).abs().mean())
    assert mae < 0.5 * baseline, (mae, baseline)
    return {"probe_mae": mae, "mean_baseline_mae": baseline}


def check_state_cut() -> float:
    """Return the largest change in a state column before the cut when readings from the cut on are changed.

    The tracker states are tracked with a cut, `tracked_state` turns them into columns, and
    `permute_days` shuffles them. Raising every reading from `refit - 7` on must leave every column
    of the half-hours before the cut, tracked or permuted, unchanged.
    """
    y, _ = c695.synthetic_series()
    fit = tracker.fit_era(y=y, lvl=tracker.spread_levels(y=y))
    refit = 1488  # a multiple of 48, so the placebo's span starts on a day boundary
    perturbed = y.copy()
    perturbed[refit - READING_DELAY :] += 3.0
    columns = []
    for series in (y, perturbed):
        states = tracker.track_segment(y=series, fit=fit, refit=refit)
        grid = tracked_state(
            pk=states.pk,
            unit=unit_vector(n_grid=len(y), era2_first=len(y), unit1=fit.unit, unit2=fit.unit),
            capacity=10.0,
        )
        permuted = permute_days(state=grid, first=0, stop=refit - READING_DELAY, seed=0)
        columns.append((grid, permuted))
    return max(
        float(np.abs(columns[0][0][:refit] - columns[1][0][:refit]).max()),
        float(
            np.abs(
                columns[0][1][: refit - READING_DELAY] - columns[1][1][: refit - READING_DELAY]
            ).max()
        ),
    )


def check_alignment_on_synthetic_rows() -> None:
    """Assert that `rows_state` reads each row's own half-hour and gives zeros for other Primaries."""
    rng = np.random.default_rng(0)
    grid = rng.random((500, N_STATE)).astype(np.float32)
    primary = rng.integers(0, 16, size=3_000)
    target = rng.integers(0, 500, size=3_000)
    got = rows_state(grid_state=grid, row=5, primary=primary, target_idx=target)
    for i in range(len(primary)):
        want = grid[target[i]] if primary[i] == 5 else np.zeros(N_STATE)
        assert np.array_equal(got[i], want), i


def check_hooks_on_synthetic_rows(*, probe: bool = False) -> dict:
    """Run the pre-training hooks on synthetic rows with a fake `train` that takes a few real AdamW steps.

    A small stage `pre` network, a fake `pretrain_rows` result, and a synthetic state file stand in for
    one fold. The hooks must widen the tensor, fill the 3 columns from the file, pass the alignment
    check, let the network read the columns (the state weights become non-zero), leave the original
    columns of the tensor equal to the input, and, for a stage `ft` call, pass the tensor through
    unchanged and refuse one that carries state columns.
    """
    import tempfile
    from types import SimpleNamespace

    from e00696_common import Arm
    from e00696_hooks import StateHooks

    n_grid, n = 4_000, 3_000
    rng = np.random.default_rng(0)
    grid = rng.random((n_grid, N_STATE)).astype(np.float32)
    grid[:, 2] = 0.0
    primary = rng.integers(0, 3, size=n)
    target_idx = rng.integers(0, n_grid - 100, size=n)
    refit = n_grid - 1
    rows = SimpleNamespace(
        primary=primary, target_idx=target_idx, origin_idx=target_idx - 48,
        y=rng.random(n).astype(np.float32), is_val=rng.random(n) < 0.1,
    )  # fmt: skip
    results = {}
    for kind in ("tracked", "zero", "hidden"):
        with tempfile.TemporaryDirectory() as folder:
            state_file = Path(folder) / "state.npz"
            np.savez(
                state_file,
                state=np.zeros_like(grid) if kind == "zero" else grid,
                row=1,
                refit=refit,
                stop=refit - READING_DELAY,
            )
            spec = Arm(kind.upper(), "d8", kind, None, None, probe=probe)  # ty: ignore[invalid-argument-type]
            hooks = StateHooks(
                spec=spec, month="2026-01", run_root=Path(folder), state_file=state_file, host_row=1,
                raw_stack=lambda: rng.random((3, n_grid)), stored_pretrained=None, probe_only=False,
            )  # fmt: skip
            hooks.wrap_pretrain_rows(lambda **_: rows)()
            _, model = _models(stage="pre")
            x = torch.randn(n, model.label_col)
            x[:, 43:45], x[:, 45] = -1.0, 0.0
            primary_t = torch.from_numpy(primary).long()
            seen = {}

            def fake_train(
                *, model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor, seed: int, **_: object
            ) -> tuple[int, float]:
                seen["width"] = x.shape[1]
                seen["state"] = x[:, -N_STATE:].clone()
                optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
                model.train()
                target = 3.0 * x[:, -N_STATE] + 2.0 * x[:, -N_STATE + 1]
                for _step in range(20):
                    loss = ((model(x[:512], p[:512])[:, 0] - target[:512]) ** 2).mean()
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                return 20, float(loss.detach())

            train = hooks.wrap_train(fake_train)
            model.table = torch.zeros(3, 4, 28)  # ty: ignore[invalid-assignment]
            train(model=model, x=x, p=primary_t, seed=0)
            assert seen["width"] == model.label_col + N_STATE
            expected = rows_state(
                grid_state=grid if kind != "zero" else np.zeros_like(grid),
                row=1,
                primary=primary,
                target_idx=target_idx,
            )
            if kind != "hidden":
                assert np.array_equal(seen["state"].numpy(), expected)
            assert hooks.wide is not None and torch.equal(hooks.wide[:, : model.label_col], x)
            record = json.loads((Path(folder) / "checks" / "2026-01_s0_pretrain.json").read_text())
            results[kind] = {
                "alignment": record["alignment"],
                "liveness": record["liveness"]["largest_abs_weight_per_state_column"],
            }
            _, ft = _models(stage="ft")
            ft.table = torch.zeros(3, 4, 28)  # ty: ignore[invalid-assignment]
            train(model=ft, x=x, p=primary_t, seed=0)  # the original layout passes through
            assert seen["width"] == ft.label_col
            try:
                train(
                    model=ft, x=torch.cat([x, torch.zeros(n, N_STATE)], dim=1), p=primary_t, seed=0
                )
            except AssertionError:
                pass
            else:
                msg = "a stage ft call accepted state columns"
                raise AssertionError(msg)
    return results


def check_real_state_cut(*, source: str, month: str) -> float:
    """Return the largest change in a fold's training-time states when readings from the cut are raised.

    As experiment 695's `check_real_stack_cut`, for the series whose states the arm reads (`A1` for
    D8, `W1` for the host). The states before the refit must not change.
    """
    import e00695_clean as clean  # ty: ignore[unresolved-import]

    setup = clean.load_setup(arm=source)
    refit = grid_index(when=fold_refit(month=month))
    _, full, _, _ = clean.fold_states(setup=setup, month=month, hindsight=False, mode="folded")
    perturbed = setup.y.copy()
    perturbed[refit - READING_DELAY :] += 3.0
    _, other, _, _ = clean.fold_states(
        setup=replace(setup, y=perturbed), month=month, hindsight=False, mode="folded"
    )
    window = slice(grid_index(when=clean.ERA1_START), refit)
    return float(
        max(
            np.abs(full.pk[window] - other.pk[window]).max(),
            np.abs(full.lmean[window] - other.lmean[window])[np.isfinite(full.lmean[window])].max(),
        )
    )


def check_state_files(*, arm: str, month: str) -> dict[str, float]:
    """Check a stored state file's shape, ranges, and zeros outside the fold; return summary numbers."""
    data = np.load(state_path(arm=arm, month=month))
    state, refit, stop = data["state"], int(data["refit"]), int(data["stop"])
    assert state.ndim == 2 and state.shape[1] == N_STATE and np.isfinite(state).all()
    assert (state[:, 1] >= 0).all() and (state[:, 1] <= 1.0 + 1e-6).all()
    assert (state[:, 2] == 0).all(), "the hidden flag is set on a state file"
    assert stop == refit - READING_DELAY
    before_era1 = state[: grid_index(when=c695_era1_start())]
    assert not before_era1.any(), "a state before the tracker starts is not zero"
    return {
        "mean_share": float(state[:, 0].mean()),
        "share_of_half_hours_running": float((state[:, 1] > 0.5).mean()),
    }


def c695_era1_start():  # noqa: ANN201
    """Return the first half-hour of the tracker's era 1."""
    from e00695_common import ERA1_START  # ty: ignore[unresolved-import]

    return ERA1_START


def _quantiles_of(*, arm: str, month: str, seed: int, root: Path) -> pl.DataFrame:
    """Return one stored fit file, sorted by its keys."""
    path = (
        root
        / "runs"
        / arm
        / "arms"
        / "k1_5w_sp1"
        / "work"
        / "fits"
        / "P5"
        / f"{month}_d0_s{seed}.parquet"
    )
    frame = pl.read_parquet(path)
    keys = [name for name, dtype in frame.schema.items() if not dtype.is_float()]
    return frame.sort(keys)


def compare_fits(
    *, arm: str, seeds: tuple[int, ...], month: str = "2025-11", reference: str = "A0"
) -> dict[str, float]:
    """Return the largest gap between `arm`'s fits and experiment 695's `reference` fits, per seed."""
    out = {}
    for seed in seeds:
        mine = _quantiles_of(arm=arm, month=month, seed=seed, root=DATA_DIR)
        theirs = _quantiles_of(arm=reference, month=month, seed=seed, root=E695_DIR)
        assert mine.height == theirs.height, (mine.height, theirs.height)
        numeric = [name for name, dtype in mine.schema.items() if dtype.is_float()]
        out[f"s{seed}"] = max(float((mine[name] - theirs[name]).abs().max()) for name in numeric)  # ty: ignore[invalid-argument-type]
    return out


def aggregate_pretrain_records() -> dict:
    """Read every per-run pre-training record and summarise the alignment and liveness checks.

    `e00696_k1.py` asserts both checks and writes their results, so a record exists only for a run
    that passed. This lists, for each arm, the folds and seeds that have a record, and the smallest
    state-weight magnitude among the live arms.
    """
    summary: dict[str, dict] = {}
    for path in sorted((DATA_DIR / "runs").glob("*/checks/*_pretrain.json")):
        arm = path.parent.parent.name
        record = json.loads(path.read_text())
        entry = summary.setdefault(
            arm, {"runs": 0, "alignment_folds": [], "smallest_state_weight": float("inf")}
        )
        entry["runs"] += 1
        if "alignment" in record:
            entry["alignment_folds"].append(record["month"])
            assert record["alignment"]["largest_gap"] == 0.0
        if "liveness" in record and arm != "SZ":
            entry["smallest_state_weight"] = min(
                entry["smallest_state_weight"],
                *record["liveness"]["largest_abs_weight_per_state_column"][:2],
            )
    return summary


def main() -> None:
    """Run the requested checks and update `checks.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write checks.json")
    parser.add_argument(
        "--real",
        nargs="+",
        default=[],
        metavar="MONTH",
        help="months, or `all`: leakage and file checks on stored data",
    )
    parser.add_argument(
        "--fits",
        action="store_true",
        help="the identity control, G0, and the liveness and alignment records",
    )
    args = parser.parse_args()
    previous = json.loads(CHECKS_PATH.read_text()) if CHECKS_PATH.exists() else {}
    results: dict = {}
    synthetic = {
        "cut_at_train_end": c695.check_cut_at_train_end(),
        "origin_cut": c695.check_origin_cut(),
        "state_cut": check_state_cut(),
        "widening_identity": check_widening_identity(),
        "state_reaches_mixer": check_state_reaches_the_mixer(),
        "liveness_synthetic": check_liveness_on_synthetic_data(),
        "probe": check_probe_on_synthetic_features(),
    }
    check_forecast_time_rejects_state()
    check_alignment_on_synthetic_rows()
    synthetic["hooks"] = check_hooks_on_synthetic_rows()
    synthetic["hooks_with_probe"] = check_hooks_on_synthetic_rows(probe=True)
    assert synthetic["cut_at_train_end"] == 0.0 and synthetic["origin_cut"] == 0.0
    assert synthetic["state_cut"] == 0.0, synthetic["state_cut"]
    assert synthetic["widening_identity"] == 0.0, synthetic["widening_identity"]
    results["synthetic"] = synthetic
    results["synthetic_passed"] = True
    months = list(FOLD_MONTHS) if args.real == ["all"] else args.real
    real = dict(previous.get("real_state_cut", {}))
    for month in months:
        real[month] = {
            source: check_real_state_cut(source=source, month=month) for source in ("A1", "W1")
        }
        results["state_files"] = {
            f"{arm}/{month}": check_state_files(arm=arm, month=month)
            for arm in ("S1", "WS1")
            if state_path(arm=arm, month=month).exists()
        }
    if months:
        results["real_state_cut"] = real
        results["leakage_passed"] = bool(
            set(real) >= set(FOLD_MONTHS)
            and max(v for per in real.values() for v in per.values()) == 0.0
        )
    if args.fits:
        results["identity_sz"] = compare_fits(arm="SZ", seeds=(0, 1, 2))
        results["g0"] = compare_fits(arm="G0", seeds=(0,))
        results["identity_passed"] = max(results["identity_sz"].values()) <= IDENTITY_TOLERANCE
        results["g0w"] = compare_fits(arm="G0W", seeds=(0,), reference="W0")
        results["g0_passed"] = (
            max(results["g0"].values()) <= IDENTITY_TOLERANCE
            and max(results["g0w"].values()) <= IDENTITY_TOLERANCE
        )
        results["pretrain_records"] = aggregate_pretrain_records()
        results["gate_failed"] = not (results["identity_passed"] and results["g0_passed"])
    print(json.dumps(results, indent=1, default=float))
    if args.write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CHECKS_PATH.write_text(json.dumps(previous | results, indent=1, default=float))


if __name__ == "__main__":
    main()

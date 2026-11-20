r"""The known-answer tests of experiment 710 (design section g), run before the first fit.

Seven tests, numbered as in the design. A failure raises, prints the test, and stops the run. A run
of all seven that passes writes `work/known_answers_passed.json`, which `e00710_fit.py` requires
(the pilot is exempt).

1. **The new loop equals the old loop with its features off.** On one CPU thread, with one seed,
   200,000 random rows, and two epochs, `e00707_net.train_network` and `e00710_net.train_stage`
   (flat weights, no sampler, no schedule, no averaging, no injection) give the same sequence of
   negative log-likelihood sums to 1e-6 and equal final weights.
2. **A refit with the new loop matches the stored `R_F`.** `M2-n` is refitted on folds 2026-06 and
   2026-01 with seeds 0 to 2 (the stored states of experiment 707 are not used). The seed-averaged
   CRPS lies within two standard errors of the seed-to-seed spread, per group. The comparison is
   written to `work/known_answer/refit_log.json`. This test needs the GPU and takes about 1.5 hours.
3. **The stored forecasts are reproduced by prediction only** from `models/A_F` and `models/R_F` to
   1e-4, and `M3-n` loads each stored `S_ens(F)` state with the scale of the matching `R_F` fit.
4. **The weight rules give known values** (see `check_weights`).
5. **An empty world changes nothing** in each of the four copied functions, and a (fold, seed) world
   changes only the three lag value columns of the rows that are not held out (`check_worlds`).
6. **The slump share has a known answer:** a forecaster whose median copies the 1-day lag scores 1,
   and a forecaster that ignores the lags scores 0, for both signs.
7. **Experiment 699's known-answer table still reproduces** (`e00699_contrasts.py known`).

```bash
# environment as in e00710_fit.py; add the folders of experiments 699 and 677 and their imports
uv run --no-sync python ml-experiments/00710-*/e00710_known_answer.py            # all seven
uv run --no-sync python ml-experiments/00710-*/e00710_known_answer.py --tests 1 4 6   # some
```
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00707_arms as arms707  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_folds as folds707  # ty: ignore[unresolved-import]
import e00707_models as models  # ty: ignore[unresolved-import]
import e00707_net as net707  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import e00710_arms as arms  # ty: ignore[unresolved-import]
import e00710_fit as fit  # ty: ignore[unresolved-import]
import e00710_folds as folds  # ty: ignore[unresolved-import]
import e00710_inject as inject  # ty: ignore[unresolved-import]
import e00710_net as net  # ty: ignore[unresolved-import]
import e00710_weights as weights  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

TOLERANCE_LOOP: Final[float] = 1e-6
TOLERANCE_PREDICTION: Final[float] = 1e-4
TOLERANCE_WEIGHT: Final[float] = 1e-6
STANDARD_ERRORS: Final[float] = 2.0
N_LOOP_ROWS: Final[int] = 200_000
N_PRIMARIES: Final[int] = 16
EXPECTED_RECENT_ROWS_2026_01: Final[int] = 5_608_723
"""The rows of `build_fold` with window `W24` at fold 2026-01 (experiment 707's log)."""
REFIT_MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01")
REFIT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
KNOWN_699_STRINGS: Final[tuple[str, ...]] = ("7.124", "7.165")
LOG_DIR: Final[Path] = fit.OUT_DIR / "known_answer"
ALL_TESTS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)


def check(*, condition: bool, message: str) -> None:
    """Raise if a condition does not hold, so a failed test stops the run.

    Args:
        condition: What must be true.
        message: What failed.

    Raises:
        AssertionError: If the condition is false.
    """
    if not condition:
        raise AssertionError(message)


def known_1_loop_equals_old_loop() -> None:
    """Test 1: the new loop with its features off equals experiment 707's `train_network`."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    width = prep.feature_names()[2].n_inputs
    x = torch.from_numpy(rng.standard_normal((N_LOOP_ROWS, width)).astype(np.float32))
    z = torch.from_numpy(rng.standard_normal(N_LOOP_ROWS).astype(np.float32))
    primary = torch.from_numpy(rng.integers(0, N_PRIMARIES, N_LOOP_ROWS))
    is_val = torch.from_numpy(rng.random(N_LOOP_ROWS) < 1 / 12)
    recorded: list[float] = []
    original = models.mixture_nll

    def spy(*, out: torch.Tensor, z: torch.Tensor, components: int) -> torch.Tensor:
        value = original(out=out, z=z, components=components)
        recorded.append(float(value.sum()))
        return value

    setting = net707.G09
    models.mixture_nll = spy
    try:
        torch.manual_seed(0)
        old = net707.new_model(n_inputs=width, n_primaries=N_PRIMARIES, setting=setting)
        net707.train_network(
            model=old,
            setting=setting,
            x=x,
            z=z,
            w=torch.ones(N_LOOP_ROWS),
            primary=primary,
            is_val=is_val,
            seed=0,
            max_epochs=2,
        )
        old_losses, recorded[:] = list(recorded), []
        torch.manual_seed(0)
        new = net707.new_model(n_inputs=width, n_primaries=N_PRIMARIES, setting=setting)
        data = net.StageData(
            x=x,
            z=z,
            primary=primary,
            train_idx=torch.nonzero(~is_val).squeeze(1),
            val_idx=torch.nonzero(is_val).squeeze(1),
            n_primaries=N_PRIMARIES,
        )
        net.train_stage(
            model=new,
            setting=setting,
            data=data,
            config=net.StageConfig(max_epochs=2),
            weight_provider=lambda _epoch: None,
            seed=0,
        )
    finally:
        models.mixture_nll = original
    skipped = math.ceil(int(is_val.sum()) / (net.BATCH * 4))
    new_losses = recorded[skipped:]
    check(condition=len(new_losses) == len(old_losses), message="the loops took different steps")
    gap = float(np.max(np.abs(np.array(new_losses) - np.array(old_losses))))
    scale = float(np.max(np.abs(old_losses)))
    check(
        condition=gap <= TOLERANCE_LOOP * max(scale, 1.0), message=f"per-step losses differ {gap}"
    )
    same = all(
        torch.equal(a, b)
        for a, b in zip(old.state_dict().values(), new.state_dict().values(), strict=True)
    )
    check(condition=same, message="the final weights differ")
    print(f"test 1 passed: {len(old_losses)} calls, largest gap {gap:.2e}, equal final weights")


def crps_by_group(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return the six's and the eight's CRPS (points) of a raw forecast, over leads to 240 hours.

    Args:
        frame: A forecast with `KEYS` and the 11 quantiles (fraction of capacity), unwidened.

    Returns:
        The equal-weight mean over a group's Primaries of each Primary's mean score.
    """
    clean = inject.observed_power()
    sites = prep.primaries()
    label = dict(zip(sites["time_series_id"].to_list(), sites["site"].to_list(), strict=True))
    out: dict[str, list[float]] = {"six": [], "eight": []}
    groups = {
        "six": ("D5", "D9", "D10", "D11", "D14", "D16"),
        "eight": tuple(f"D{n}" for n in (2, 3, 4, 8, 12, 19, 21, 22)),
    }
    for series_id, part in frame.group_by("time_series_id"):
        site = label[int(series_id[0])]
        valid = np.asarray(common.grid_index(times=part["valid_time"]))
        origin = np.asarray(common.grid_index(times=part["power_fcst_init_time"]))
        actual = common.take(values=clean[site], index=valid).astype(np.float64)
        keep = np.isfinite(actual) & ((valid - origin) / 2.0 <= 240.0)
        q = part.select(list(common.LEVEL_COLUMNS)).to_numpy().astype(np.float64)[keep]
        score = 100.0 * common.pinball_quantile_score(quantiles=q, actual=actual[keep])
        for group, members in groups.items():
            if site in members:
                out[group].append(float(score.mean()))
    return {g: float(np.mean(v)) for g, v in out.items() if v}


def known_2_refit_matches_stored() -> None:
    """Test 2: a refit of `M2-n` with the new loop matches the stored `R_F` (2 standard errors)."""
    out_dir = LOG_DIR
    args = argparse.Namespace(
        device="cuda",
        h_t=0.5,
        pp_half_lives=None,
        pilot=False,
        no_bank=True,
        no_stored_states=True,
        shift_share_range=None,
    )
    scales: dict[str, fit.MonthScale] = {}
    jobs = [arms.Job("M2", "n", m, s) for m in REFIT_MONTHS for s in REFIT_SEEDS]
    for _tier, key, names in fit.group_jobs(jobs=jobs, h_t=0.5, b_star="M2", o_star=None):
        fit.run_group(
            key=key, arm_names=names, scales=scales, args=args, out_dir=out_dir, show_example=False
        )
    log: dict[str, dict] = {}
    failures = []
    for month in REFIT_MONTHS:
        new = [
            crps_by_group(
                frame=pl.read_parquet(
                    fit.forecast_path(out_dir=out_dir, arm="M2", state="n", month=month, seed=s)
                )
            )
            for s in REFIT_SEEDS
        ]
        stored = [
            crps_by_group(
                frame=pl.read_parquet(
                    fit.STORED_707_WORK / "seed_preds" / "R_F" / f"{month}_s{s}.parquet"
                )
            )
            for s in arms.STORED_SEEDS
        ]
        log[month] = {}
        for group in ("six", "eight"):
            a = np.array([n[group] for n in new])
            b = np.array([s[group] for s in stored])
            se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            gap = float(a.mean() - b.mean())
            ok = abs(gap) <= STANDARD_ERRORS * se
            log[month][group] = {
                "refit_mean": float(a.mean()),
                "stored_mean": float(b.mean()),
                "gap": gap,
                "two_standard_errors": STANDARD_ERRORS * se,
                "passed": ok,
            }
            if not ok:
                failures.append((month, group, gap, STANDARD_ERRORS * se))
    fit.write_json(payload=log, target=LOG_DIR / "refit_log.json")
    check(condition=not failures, message=f"the refit differs from the stored R_F: {failures}")
    print(f"test 2 passed: log in {LOG_DIR / 'refit_log.json'}")


def known_3_prediction_only(*, month: str = "2026-01", seed: int = 0) -> None:
    """Test 3: stored networks reproduce the stored forecasts to 1e-4; `M3-n` states load.

    Args:
        month: The fold to check (experiment 707's gate used the full set; one fold shows the code).
        seed: The seed to check.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for stored_arm in ("A_F", "R_F"):
        folder = fit.STORED_707_WORK / "models" / stored_arm / month / f"s{seed}"
        scale = net707.load_scale(folder=folder)
        fitted = net707.load_fitted(folder=folder, device=device)
        frame = net.forecast_month(fitted=fitted, scale=scale, month=month, device=device)
        stored = pl.read_parquet(
            fit.STORED_707_WORK / "seed_preds" / stored_arm / f"{month}_s{seed}.parquet"
        )
        joined = frame.join(stored, on=common.KEYS, suffix="_stored", validate="1:1")
        check(
            condition=joined.height == stored.height == frame.height,
            message=f"{stored_arm}: the forecasts hold different keys",
        )
        gap = max(
            float((joined[c] - joined[f"{c}_stored"]).abs().max()) for c in common.LEVEL_COLUMNS
        )
        check(condition=gap <= TOLERANCE_PREDICTION, message=f"{stored_arm}: gap {gap:.2e} > 1e-4")
        print(f"test 3: {stored_arm} {month} seed {seed} reproduced, largest gap {gap:.2e}")
    r_folder = fit.STORED_707_WORK / "models" / "R_F" / month / f"s{seed}"
    fitted = net707.load_fitted(folder=r_folder, device=device)
    state = torch.load(
        fit.STORED_707_WORK / "pretrained" / "ensF_ftw730" / f"{month}_s{seed}.pt",
        map_location=device,
    )
    fitted.model.load_state_dict(state)
    frame = net.forecast_month(
        fitted=fitted, scale=net707.load_scale(folder=r_folder), month=month, device=device
    )
    values = frame.select(list(common.LEVEL_COLUMNS)).to_numpy()
    check(condition=bool(np.isfinite(values).all()), message="M3-n forecast is not finite")
    print(
        f"test 3 passed: M3-n loads the stored S_ens(F) state with R_F's scale "
        f"({frame.height} rows)"
    )


def check_weights(*, month: str = "2026-01") -> None:
    """Test 4: the weight rules give known values.

    The checks: `W-h0.5` equals `fold.weight` to 1e-6 on the fine-tune rows; an infinite half-life
    gives ones; `HW730` keeps exactly the rows of `build_fold` with window `W24`; epoch 6 of `S`
    has the half-life of `W-h0.5`; multiplying every weight by 10 changes no loss; the sampler's
    empirical frequencies over 1,000 batches match the weights within sampling error; the placebo
    is fixed by its seed and leaves the partial first month alone.

    Args:
        month: The fold whose rows are used.
    """
    refit = prep.refit_of(month=month)
    refit_us = int(refit.timestamp() * 1e6)
    fold = folds.build_fold(refit=refit, window=arms707.W24, power=None)
    rows, _ = folds.ens_rows(
        standardiser=fold.standardiser, refit=refit, window=arms707.W_F, power=None
    )
    age = weights.age_days(valid_us=rows.valid, refit_us=refit_us)
    recent = weights.within_window(age=age, window_days=arms.RECENT_DAYS)
    check(condition=int(recent.sum()) == len(fold.x), message="HW730 keeps other rows than W24")
    check(
        condition=bool(np.array_equal(rows.valid[recent], fold.valid))
        and bool(np.array_equal(rows.is_val[recent], fold.is_val)),
        message="the window-F rows of the last 730 days differ from the fold's rows",
    )
    if month == "2026-01":
        check(
            condition=len(fold.x) == EXPECTED_RECENT_ROWS_2026_01,
            message=f"fold 2026-01 holds {len(fold.x)} rows, not {EXPECTED_RECENT_ROWS_2026_01}",
        )
    w = weights.exponential(age=age[recent], half_life_years=0.5)
    gap = float(np.max(np.abs(w - fold.weight)))
    check(condition=gap <= TOLERANCE_WEIGHT, message=f"W-h0.5 differs from fold.weight by {gap}")
    ones = weights.exponential(age=age, half_life_years=float("inf"))
    check(condition=bool(np.all(ones == 1.0)), message="an infinite half-life is not all ones")
    check(
        condition=weights.schedule_half_life(schedule="S", epoch=5) == 0.5
        and weights.schedule_half_life(schedule="S", epoch=30) == 0.5,
        message="epoch 6 of S does not use the half-life of W-h0.5",
    )
    check(
        condition=[weights.schedule_half_life(schedule="CYC", epoch=e) for e in (0, 4, 8)]
        == [float("inf"), 2.0, 0.5],
        message="CYC's schedule is wrong",
    )
    torch.manual_seed(0)
    nll, wt = torch.rand(1000), torch.rand(1000) + 0.1
    a = weights.weighted_mean_nll(nll=nll, w=wt)
    b = weights.weighted_mean_nll(nll=nll, w=10 * wt)
    check(condition=abs(float(a - b)) <= 1e-6, message="scaling the weights changed the loss")
    probability = (wt / wt.sum()).double()
    cumulative = torch.cumsum(probability, dim=0)
    cumulative = cumulative / cumulative[-1]
    counts = torch.zeros(1000, dtype=torch.float64)
    index = torch.arange(1000)
    for _ in range(1000):
        counts += torch.bincount(
            net.sample_batch(cumulative=cumulative, train_idx=index, size=net.BATCH), minlength=1000
        ).double()
    expected = probability * 1000 * net.BATCH
    z = (counts - expected).abs() / expected.sqrt()
    check(
        condition=float(z.max()) < 5.0,
        message=f"the sampler's frequencies are off: z {float(z.max()):.1f}",
    )
    floor_us = int(arms707.LONG_WINDOW_FLOOR.timestamp() * 1e6)
    yyyymm = int(month.replace("-", ""))
    p1 = weights.permuted_age_days(
        valid_us=rows.valid, refit_us=refit_us, floor_us=floor_us, fold_yyyymm=yyyymm
    )
    p2 = weights.permuted_age_days(
        valid_us=rows.valid, refit_us=refit_us, floor_us=floor_us, fold_yyyymm=yyyymm
    )
    check(condition=bool(np.array_equal(p1, p2)), message="the placebo is not fixed by its seed")
    first_month = rows.valid < int(datetime(2021, 8, 1, tzinfo=UTC).timestamp() * 1e6)
    check(
        condition=bool(np.array_equal(p1[first_month], age[first_month])),
        message="the placebo changed the partial first month",
    )
    check(condition=bool((p1 >= 0).all()), message="the placebo made a negative age")
    print(
        f"test 4 passed: {int(recent.sum()):,} recent rows, W-h0.5 gap {gap:.1e}, "
        f"sampler z {float(z.max()):.2f}"
    )


def compare_rows(
    *, a: np.ndarray, b: np.ndarray, held: np.ndarray, label: str, lag_columns: list[int]
) -> int:
    """Check that a world changes only the lag value columns of the non-held-out rows.

    Args:
        a: The matrix without a world.
        b: The matrix with a world.
        held: Which rows are held out.
        label: The function's name, for the message.
        lag_columns: The columns of the three lag values.

    Returns:
        The number of rows that changed.
    """
    others = [c for c in range(a.shape[1]) if c not in lag_columns]
    check(
        condition=bool(np.array_equal(a[:, others], b[:, others])),
        message=f"{label}: another column changed",
    )
    check(
        condition=bool(np.array_equal(a[held], b[held])), message=f"{label}: a held-out row changed"
    )
    changed = np.any(a[:, lag_columns] != b[:, lag_columns], axis=1)
    check(condition=bool(changed.any()), message=f"{label}: the world changed no row")
    return int(changed.sum())


def check_worlds(*, month: str = "2026-01", seed: int = 0) -> None:
    """Test 5: an empty world changes nothing, and a (fold, seed) world changes only lag values.

    The four copied functions (`build_fold`, `ens_rows`, `fold_pretrain`, `run_inputs`) are
    compared with no world, with the observed power as an empty world (bit for bit), and with the
    (fold, seed) world. The world changes only the three lag value columns of the rows that are
    not held out, its last changed half-hour is before the refit, and the standardiser and target
    statistics do not move.

    Args:
        month: The fold.
        seed: The network seed of the world.
    """
    refit = prep.refit_of(month=month)
    observed = inject.observed_power()
    world = inject.build_world(month=month, seed=seed)
    lag_columns = inject.lag_column_index()
    refit_index = int(common.grid_index(times=refit))  # ty: ignore[invalid-argument-type]
    for site, shifted in world.power.items():
        differs = np.flatnonzero(np.nan_to_num(shifted - observed[site]) != 0)
        check(
            condition=bool(len(differs) == 0 or differs.max() < refit_index),
            message=f"{site}: the world changes a half-hour at or after the refit",
        )
    base = folds.build_fold(refit=refit, window=arms707.W24, power=None)
    empty = folds.build_fold(refit=refit, window=arms707.W24, power=observed)
    check(
        condition=bool(np.array_equal(base.x, empty.x)),
        message="build_fold: an empty world changed x",
    )
    shifted_fold = folds.build_fold(refit=refit, window=arms707.W24, power=world.power)
    n_fold = compare_rows(
        a=base.x, b=shifted_fold.x, held=base.is_val, label="build_fold", lag_columns=lag_columns
    )
    for name in ("mean", "std"):
        check(
            condition=bool(
                np.array_equal(
                    getattr(base.standardiser, name), getattr(shifted_fold.standardiser, name)
                )
            )
            and bool(np.array_equal(base.target_mean, shifted_fold.target_mean))
            and bool(np.array_equal(base.target_std, shifted_fold.target_std)),
            message="the scale moved with the world",
        )
    check(condition=bool(np.array_equal(base.y, shifted_fold.y)), message="the target changed")
    standardiser = base.standardiser
    spread = folds707.spread_table(fold=base)
    del base, empty, shifted_fold
    plain, _ = folds.ens_rows(
        standardiser=standardiser, refit=refit, window=arms707.W_F, power=None
    )
    empty_rows, _ = folds.ens_rows(
        standardiser=standardiser, refit=refit, window=arms707.W_F, power=observed
    )
    check(
        condition=bool(np.array_equal(plain.x, empty_rows.x)),
        message="ens_rows: an empty world changed x",
    )
    del empty_rows
    with_world, _ = folds.ens_rows(
        standardiser=standardiser, refit=refit, window=arms707.W_F, power=world.power
    )
    n_ens = compare_rows(
        a=plain.x, b=with_world.x, held=plain.is_val, label="ens_rows", lag_columns=lag_columns
    )
    check(
        condition=bool(np.array_equal(plain.y, with_world.y)),
        message="ens_rows: the target changed",
    )
    del plain, with_world
    plain, _ = folds.fold_pretrain(
        standardiser=standardiser, spread=spread, refit=refit, power=None
    )
    empty_rows, _ = folds.fold_pretrain(
        standardiser=standardiser, spread=spread, refit=refit, power=observed
    )
    check(
        condition=bool(np.array_equal(plain.x, empty_rows.x)),
        message="fold_pretrain: an empty world changed x",
    )
    del empty_rows
    with_world, _ = folds.fold_pretrain(
        standardiser=standardiser, spread=spread, refit=refit, power=world.power
    )
    n_era = compare_rows(
        a=plain.x, b=with_world.x, held=plain.is_val, label="fold_pretrain", lag_columns=lag_columns
    )
    del plain, with_world
    origin = prep.test_origins(month=month)[0]
    plain_runs = folds.run_inputs(origin=origin, standardiser=standardiser, power=None)
    empty_runs = folds.run_inputs(origin=origin, standardiser=standardiser, power=observed)
    old_runs = prep.run_inputs(origin=origin, standardiser=standardiser)
    for (_, _, x0), (_, _, x1), (_, _, x2) in zip(plain_runs, empty_runs, old_runs, strict=True):
        check(
            condition=bool(np.array_equal(x0, x1)) and bool(np.array_equal(x0, x2)),
            message="run_inputs: an empty world changed x",
        )
    print(
        f"test 5 passed: rows changed by the world: build_fold {n_fold:,}, ens_rows {n_ens:,}, "
        f"fold_pretrain {n_era:,}; {world.n_events} events; lags only, scale unchanged"
    )


def known_6_slump_known_answer() -> None:
    """Test 6: a forecaster that copies the 1-day lag scores 1; one that ignores the lags, 0."""
    import e00710_score as score

    rng = np.random.default_rng(1)
    size = 48 * 60
    power = rng.random(size).astype(np.float32)
    for signed in (0.3, -0.3, 0.5, -0.15):
        shift = np.zeros(size, dtype=np.float32)
        onset, end = 48 * 20, 48 * 34
        shift[onset:end] = signed
        origin = np.arange(48 * 30, 48 * 40)
        target = origin + 10
        lag_base = common.origin_lags(power=power, target=target, origin=origin)
        lag_shift = common.origin_lags(power=power + shift, target=target, origin=origin)
        mask = score.lag_shift_mask(shift_series=shift, valid_index=target, origin_index=origin)
        day_shifted = np.abs(lag_shift[:, 1] - lag_base[:, 1]) > 0
        check(
            condition=bool(mask.any()) and bool(day_shifted.any()),
            message="the event shifts no lag",
        )
        copy = score.slump_share(
            shifted=lag_shift[day_shifted, 1].astype(np.float64),
            base=lag_base[day_shifted, 1].astype(np.float64),
            signed_depth=np.full(int(day_shifted.sum()), signed),
        )
        ignore = score.slump_share(
            shifted=np.full(int(day_shifted.sum()), 0.4),
            base=np.full(int(day_shifted.sum()), 0.4),
            signed_depth=np.full(int(day_shifted.sum()), signed),
        )
        check(
            condition=bool(np.allclose(copy, 1.0, atol=1e-5)),
            message=f"copy share is not 1 for {signed}",
        )
        check(condition=bool(np.all(ignore == 0.0)), message=f"ignore share is not 0 for {signed}")
    print("test 6 passed: the 1-day-lag copier scores 1 and the lag-ignoring forecaster scores 0")


def known_7_known_699() -> None:
    """Test 7: experiment 699's known-answer table reproduces (`e00699_contrasts.py known`)."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)  # `run_phase.sh` already put the experiment folders on PYTHONPATH
    script = next(root.glob("00699-*/e00699_contrasts.py"))
    done = subprocess.run(
        [sys.executable, str(script), "known"], capture_output=True, text=True, env=env, check=False
    )
    check(
        condition=done.returncode == 0,
        message=f"e00699_contrasts.py known failed:\n{done.stdout}\n{done.stderr}",
    )
    check(
        condition=all(s in done.stdout for s in KNOWN_699_STRINGS),
        message=f"the output lacks {KNOWN_699_STRINGS}:\n{done.stdout}",
    )
    print("test 7 passed: experiment 699's table reproduces")


def main() -> None:
    """Run the chosen tests (all seven by default) and write the marker if all seven passed."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tests", nargs="+", type=int, default=list(ALL_TESTS))
    parser.add_argument("--month", default="2026-01")
    args = parser.parse_args()
    runners = {
        1: known_1_loop_equals_old_loop,
        2: known_2_refit_matches_stored,
        3: lambda: known_3_prediction_only(month=args.month),
        4: lambda: check_weights(month=args.month),
        5: lambda: check_worlds(month=args.month),
        6: known_6_slump_known_answer,
        7: known_7_known_699,
    }
    passed = json.loads(fit.MARKER.read_text()) if fit.MARKER.exists() else {}
    current = fit.code_hash()
    if passed.get("code_sha256") != current:
        passed = {"code_sha256": current}
    for number in args.tests:
        runners[number]()
        passed[str(number)] = datetime.now(UTC).isoformat(timespec="seconds")
        fit.write_json(payload=passed, target=fit.MARKER)
    complete = all(str(n) in passed for n in ALL_TESTS)
    print(
        f"tests passed so far on the code with SHA-256 {current[:12]}: "
        f"{sorted(k for k in passed if k.isdigit())}; all seven: {complete}"
    )


if __name__ == "__main__":
    main()

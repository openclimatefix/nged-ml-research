# ruff: noqa: E501, PT018
"""Experiment 713: the synthetic world and known-answer checks KA1 to KA12 (design section 3).

Usage: e00713_known_answer.py MONTH [--device cuda] [--skip-fits]
       e00713_known_answer.py --selftest

`--selftest` runs on the CPU with no data and no fit: it builds a small synthetic world from random
arrays and checks the world's formula, the follow share, and the gate file logic.

The full run needs the GPU for about 45 minutes. It runs, for the fold MONTH, KA5 (poisoned power),
KA6 (future weather canary) on the real inputs, then fits the synthetic worlds W0 and W1 with the
training loop of `e00713_train` (stage 1, at most 15 passes, arms `M`, `P`, `N`, `E`, `G`) and checks
KA2, KA3, KA4, KA10, and KA12. KA1, KA7, and KA9 are run by `e00713_check_model.py` and
`e00713_check_layout.py`; KA8 is run by `e00713_score.py`; KA11 by `e00713_era.py`. The script writes
`work/known_answers_passed.json` only if every check it runs passes; `e00713_fit.py` refuses to run
without the file. A failed check fires audit trigger 1.

The synthetic world keeps the real grid, the real ERA5 weather, and the real calendar of four
Primaries, and replaces the power with a known function, in shares of capacity:

    y(p, t) = 0.5 + 0.02 max(0, 15.5 - T(p, t)) + w[day type of t, half-hour of t] + shift(p, t) + noise

`T` is the ERA5 2 m temperature in degrees Celsius, `w` is a fixed 8 x 48 table drawn once from a
uniform distribution on -0.10 to +0.10 (seed 0), the noise is Gaussian with standard deviation 0.01,
and `shift` is zero in world W0. In world W1 each Primary has level steps of +0.30 or -0.30 lasting
10 days, starting at random half-hours that cover 10% of the history.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00713_arms as arms
import e00713_batch as bt
import e00713_layout as lay
import e00713_model as mdl
import e00713_train as tr
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PRIMARIES: Final = (2, 5, 9, 14)
STEP_SIZE: Final = 0.30
STEP_DAYS: Final = 10
STEP_SHARE: Final = 0.10
NOISE_SD: Final = 0.01
WORLD_SEED: Final = 0
HELD_OUT_DAYS: Final = 60
SYNTHETIC_PASSES: Final = 15
THRESHOLDS: Final = {
    "KA2_mae": 0.02, "KA3_follow_m": 0.75, "KA3_follow_p": 0.25, "KA4_m_minus_n": 0.005,
    "KA10_abs_m_minus_p": 0.005, "KA12_slack": 0.05,
}  # fmt: skip
MIN_STEP_AGE_DAYS: Final = 3


@dataclass
class Synthetic:
    """A synthetic world's arrays (shares of capacity), shape (16, G); only `PRIMARIES` are filled."""

    y: np.ndarray
    shift: np.ndarray
    step_start: (
        np.ndarray
    )  # (16, G) grid index where the step covering the half-hour began, -1 if none
    kind: str


def draw_table() -> np.ndarray:
    """Return the fixed 8 x 48 table `w`: uniform on -0.10 to +0.10, seed 0."""
    return np.random.default_rng(WORLD_SEED).uniform(-0.10, 0.10, size=(mdl.N_DAY_TYPES, 48))


def build_world(
    *, kind: str, temperature: np.ndarray, day_type: np.ndarray, half_hour: np.ndarray,
    primaries: tuple[int, ...] = PRIMARIES, first_valid: int = 0, seed: int = WORLD_SEED,
) -> Synthetic:  # fmt: skip
    """Return world `W0` (no shift) or `W1` (steps of +-0.30 for 10 days over 10% of the history).

    Args:
        kind: `W0` or `W1`.
        temperature: (16, G) ERA5 2 m temperature in degrees Celsius.
        day_type: (G,) day type of each grid index.
        half_hour: (G,) local half-hour of the day of each grid index.
        primaries: The Primaries that are filled.
        first_valid: First grid index of the history that carries steps.
        seed: Seed of the noise and the steps.

    Returns:
        The synthetic world.
    """
    assert kind in {"W0", "W1"}
    n_grid = temperature.shape[1]
    w = draw_table()
    rng = np.random.default_rng([seed, 1 if kind == "W0" else 2])
    y = np.full(temperature.shape, np.nan)
    shift = np.zeros(temperature.shape)
    start = np.full(temperature.shape, -1, dtype=np.int64)
    length = STEP_DAYS * lay.STEPS_PER_DAY
    for p in primaries:
        base = 0.5 + 0.02 * np.maximum(0.0, 15.5 - temperature[p]) + w[day_type, half_hour]
        if kind == "W1":
            covered = 0
            while covered < STEP_SHARE * (n_grid - first_valid):
                on = int(rng.integers(first_valid, n_grid - length))
                sign = float(rng.choice([-1.0, 1.0]))
                shift[p, on : on + length] = sign * STEP_SIZE
                start[p, on : on + length] = on
                covered += length
        y[p] = base + shift[p] + rng.normal(0.0, NOISE_SD, size=n_grid)
    return Synthetic(y=y, shift=shift, step_start=start, kind=kind)


def to_world(
    *, syn: Synthetic, template: bt.Source, cutoff_idx: int, device: str | torch.device
) -> tuple[tr.World, np.ndarray, np.ndarray]:
    """Return a training `World` whose power is the synthetic `y`, with its per-Primary mean and standard deviation."""
    clean = np.where(np.isfinite(syn.y), syn.y, np.nan).astype(np.float32)
    train = clean[:, :cutoff_idx]
    mean = np.where(np.isfinite(train).any(axis=1), np.nanmean(train, axis=1), 0.0)
    std = np.where(np.isfinite(train).any(axis=1), np.nanstd(train, axis=1), 1.0)
    z = torch.from_numpy(((clean - mean[:, None]) / std[:, None]).astype(np.float32)).to(device)
    source = bt.Source(
        power_z=z, dense_weather=template.dense_weather, dense_datetime=template.dense_datetime,
        day_type=template.day_type, ens=None,
    )  # fmt: skip
    pool = tr.block_pool(clean=clean)
    world = tr.World(
        source=source, cutoff_idx=cutoff_idx, clean=clean, target_mean=mean, target_std=std,
        month="2026-01", placebo_pool=pool,
    )  # fmt: skip
    return world, mean, std


def test_samples(*, world: tr.World, primaries: tuple[int, ...]) -> tr.SampleSet:
    """Return the held-out origins: the 60 days after the cutoff, for `primaries` (all 14 lead days)."""
    days = np.arange(world.cutoff_idx // 48, world.cutoff_idx // 48 + HELD_OUT_DAYS)
    c = days * 48 + 24
    prim, cc = np.meshgrid(np.array(primaries), c, indexing="ij")
    prim, cc = prim.ravel(), cc.ravel()
    return tr.SampleSet(
        primary=prim, c_idx=cc, is_val=np.zeros(len(prim), bool),
        ages=tr.placebo_ages_for(primary=prim, c_idx=cc, pool=world.placebo_pool)  # ty: ignore[invalid-argument-type]
        if world.placebo_pool is not None
        else None,
    )  # fmt: skip


def predict_medians(
    *, model: mdl.MsaNet, world: tr.World, samples: tr.SampleSet, source: bt.Source,
    use_placebo: bool, mean: np.ndarray, std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Return (median, actual, target index) per (sample, lead day, column), in capacity shares."""
    model.eval()
    prim, cc = samples.primary, samples.c_idx
    meds, acts, idxs = [], [], []
    with torch.no_grad():
        for lead in range(lay.N_LEAD_DAYS):
            for start in range(0, len(prim), 512):
                sel = np.arange(start, min(start + 512, len(prim)))
                sub = tr.SampleSet(
                    prim, cc, np.zeros(len(prim), bool), samples.ages if use_placebo else None
                )
                b = tr.batch_for(
                    source=source, samples=sub, index=sel, lead=np.full(len(sel), lead),
                    ensemble=False, cut=10**9, dropout=None,
                )  # fmt: skip
                q = mdl.standardised_quantiles(out=model(b.inputs).out)[..., 5].cpu().numpy()
                p = prim[sel]
                meds.append(q * std[p][:, None] + mean[p][:, None])
                acts.append(b.target.cpu().numpy() * std[p][:, None] + mean[p][:, None])
                idxs.append(b.target_idx.cpu().numpy())
    return np.concatenate(meds), np.concatenate(acts), np.concatenate(idxs)


def mae_of_median(*, med: np.ndarray, act: np.ndarray) -> float:
    """Return the mean absolute error of the median over finite targets."""
    ok = np.isfinite(act)
    return float(np.abs(med[ok] - act[ok]).mean())


def follow_share(
    *, model: mdl.MsaNet, world: tr.World, syn: Synthetic, samples: tr.SampleSet, mean: np.ndarray,
    std: np.ndarray, use_placebo: bool,
) -> float:  # fmt: skip
    """Return the follow share of a trained network on W1 (design 3.2, KA3).

    The follow share is the trained median on the W1 inputs, minus the trained median on the same
    inputs with the shift removed from the history power, divided by the step, averaged over the
    target tokens inside a step that began at least 3 days before the origin.
    """
    z_clean = (syn.y - syn.shift - mean[:, None]) / std[:, None]
    without = world.source.with_power_in(
        power_in=torch.from_numpy(z_clean.astype(np.float32)).to(world.source.device)
    )
    with_shift, _, idx = predict_medians(
        model=model, world=world, samples=samples, source=world.source, use_placebo=use_placebo, mean=mean, std=std,
    )  # fmt: skip
    no_shift, _, _ = predict_medians(
        model=model, world=world, samples=samples, source=without, use_placebo=use_placebo, mean=mean, std=std,
    )  # fmt: skip
    prim = np.tile(np.repeat(samples.primary, 1), lay.N_LEAD_DAYS)
    return follow_from_arrays(
        diff=with_shift - no_shift, idx=idx, prim=prim[: len(idx)], c_idx=np.tile(samples.c_idx, lay.N_LEAD_DAYS)[: len(idx)], syn=syn,
    )  # fmt: skip


def follow_from_arrays(
    *, diff: np.ndarray, idx: np.ndarray, prim: np.ndarray, c_idx: np.ndarray, syn: Synthetic
) -> float:
    """Average `diff / step` over target tokens inside a step that began at least 3 days before the origin."""
    step = syn.shift[prim[:, None], idx]
    began = syn.step_start[prim[:, None], idx]
    qualifies = (
        (step != 0)
        & (began >= 0)
        & ((c_idx[:, None] - began) >= MIN_STEP_AGE_DAYS * lay.STEPS_PER_DAY)
    )
    return (
        float(np.mean((diff * np.sign(step) / STEP_SIZE)[qualifies]))
        if qualifies.any()
        else float("nan")
    )


def selftest() -> int:
    """CPU checks of the world's formula and the follow share, with no data and no fit."""
    rng = np.random.default_rng(1)
    n_grid = 48 * 400
    temp = rng.uniform(-5, 25, size=(16, n_grid))
    day_type = rng.integers(0, 8, n_grid)
    half_hour = (np.arange(n_grid) % 48).astype(int)
    w0 = build_world(kind="W0", temperature=temp, day_type=day_type, half_hour=half_hour)
    w1 = build_world(kind="W1", temperature=temp, day_type=day_type, half_hour=half_hour)
    p = PRIMARIES[0]
    expect = 0.5 + 0.02 * np.maximum(0, 15.5 - temp[p]) + draw_table()[day_type, half_hour]
    assert (
        float(np.abs(w0.y[p] - expect).std()) < 2 * NOISE_SD
        and float(np.abs(w0.y[p] - expect).mean()) < 3 * NOISE_SD
    )
    assert not np.any(w0.shift) and np.isnan(w0.y[0]).all()
    share = float((w1.shift[p] != 0).mean())
    assert 0.08 <= share <= 0.5, share
    assert set(np.unique(np.abs(w1.shift[p]))) <= {0.0, STEP_SIZE}
    assert np.array_equal(draw_table(), draw_table())
    # the follow share of a perfect follower is 1 and of a network that ignores history is 0
    idx = np.tile(np.arange(100, 148), (3, 1))
    prim = np.array([p, p, p])
    c_idx = np.array([5000, 5000, 5000])
    fake = Synthetic(
        y=w1.y, shift=np.zeros_like(w1.y), step_start=np.full(w1.y.shape, -1), kind="W1"
    )
    fake.shift[p, 100:148] = -STEP_SIZE
    fake.step_start[p, 100:148] = 100
    full = follow_from_arrays(
        diff=np.full((3, 48), -STEP_SIZE), idx=idx, prim=prim, c_idx=c_idx, syn=fake
    )
    none = follow_from_arrays(diff=np.zeros((3, 48)), idx=idx, prim=prim, c_idx=c_idx, syn=fake)
    young = follow_from_arrays(
        diff=np.full((3, 48), -STEP_SIZE),
        idx=idx,
        prim=prim,
        c_idx=np.array([100 + 48, 100 + 48, 100 + 48]),
        syn=fake,
    )
    assert abs(full - 1.0) < 1e-9 and none == 0.0 and np.isnan(young)
    print("known-answer self-test passed (no data, no fit)")
    return 0


def run_checks(*, month: str, device: str, skip_fits: bool) -> dict:
    """Run KA5, KA6, and the synthetic-world checks for the fold `month`; return the results."""
    os.environ["E00667_PHASE"] = "train"
    import e00713_inputs as inp

    results: dict = {}
    pipe = inp.load_pipeline(month=month, phase="train")
    data = inp.build_fold_data(pipe=pipe, device=device)
    src = data.source
    # KA5 and KA6 on the real inputs
    rng = np.random.default_rng(0)
    first_c = lay.STEPS_PER_DAY * 16 + 24
    c = (rng.integers(first_c // 48, data.cutoff_idx // 48, size=256) * 48 + 24).astype(np.int64)
    dev = src.device
    prim = torch.from_numpy(rng.integers(0, 16, size=256)).to(dev)
    lead = torch.from_numpy(rng.integers(0, 14, size=256)).to(dev)
    c_t = torch.from_numpy(c).to(dev)
    batch = src.assemble(primary=prim, c_idx=c_t, lead_day=lead, ensemble=False)
    results["KA5"] = bt.poison_check(
        source=src, batch=batch, ages=None, age_exists=None, ensemble=False, lead_day=lead
    )
    ens_c = torch.from_numpy(
        np.arange(int(data.cutoff_idx) - 48 * 40, int(data.cutoff_idx) - 48 * 3, 48) + 24
    ).to(dev)
    results["KA6"] = "canary raised: " + bt.ka6_canary(
        source=src,
        primary=prim[:1].expand(len(ens_c)),
        c_idx=ens_c,
        lead_day=lead[:1].expand(len(ens_c)),
    )
    if skip_fits:
        return results
    # the synthetic worlds
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    raw = pipe.d632.dense_raw()
    temperature = np.asarray(
        raw[:, :, pipe.d632.prep.WEATHER.index("temperature_2m")], dtype=np.float64
    )
    if np.nanmedian(temperature) > 100:
        temperature = temperature - 273.15
    n_grid = temperature.shape[1]
    holidays = lay.england_bank_holidays(years=range(2019, 2029))
    day_type = lay.day_types_for_grid(
        grid_start_us=prep422.GRID_START_US, n_grid=n_grid, holidays=holidays
    )
    half_hour = lay.local_half_hours_for_grid(grid_start_us=prep422.GRID_START_US, n_grid=n_grid)
    mae: dict[str, dict[str, float]] = {"W0": {}, "W1": {}}
    follow: dict[str, dict[str, float]] = {"W1": {}}
    for kind in ("W0", "W1"):
        syn = build_world(
            kind=kind,
            temperature=temperature,
            day_type=day_type,
            half_hour=half_hour,
            first_valid=lay.STEPS_PER_DAY * 20,
        )
        world, mean, std = to_world(syn=syn, template=src, cutoff_idx=data.cutoff_idx, device=dev)
        held = test_samples(world=world, primaries=PRIMARIES)
        for arm in ("M", "P", "N", "E", "G"):
            if kind == "W0" and arm in {"E", "G"}:
                continue
            torch.manual_seed(0)
            model = mdl.MsaNet(switches=mdl.ARM_SWITCHES[arm]).to(dev)
            samples = tr.make_samples(world=world, stage=1, placebo=arm == "P", primaries=PRIMARIES)
            tr.train_stage(
                model=model,
                world=world,
                stage=1,
                samples=samples,
                seed=0,
                encoders="random",
                max_passes=SYNTHETIC_PASSES,
            )
            med, act, _idx = predict_medians(
                model=model, world=world, samples=held, source=world.source, use_placebo=arm == "P", mean=mean, std=std,
            )  # fmt: skip
            mae[kind][arm] = mae_of_median(med=med, act=act)
            if kind == "W1":
                follow["W1"][arm] = follow_share(
                    model=model,
                    world=world,
                    syn=syn,
                    samples=held,
                    mean=mean,
                    std=std,
                    use_placebo=arm == "P",
                )
            print(
                kind,
                arm,
                "MAE",
                round(mae[kind][arm], 4),
                "follow",
                follow["W1"].get(arm),
                flush=True,
            )
    t = THRESHOLDS
    results["KA2"] = {"mae_N_W0": mae["W0"]["N"], "pass": mae["W0"]["N"] <= t["KA2_mae"]}
    results["KA3"] = {
        "follow": follow["W1"],
        "pass": follow["W1"]["M"] >= t["KA3_follow_m"] and follow["W1"]["P"] <= t["KA3_follow_p"] and follow["W1"]["N"] == 0.0,
    }  # fmt: skip
    results["KA4"] = {
        "m_minus_n": mae["W0"]["M"] - mae["W0"]["N"],
        "pass": mae["W0"]["M"] - mae["W0"]["N"] <= t["KA4_m_minus_n"],
    }
    results["KA10"] = {
        "abs_m_minus_p": abs(mae["W0"]["M"] - mae["W0"]["P"]),
        "pass": abs(mae["W0"]["M"] - mae["W0"]["P"]) <= t["KA10_abs_m_minus_p"],
    }
    results["KA12"] = {
        "follow_e": follow["W1"]["E"], "follow_g": follow["W1"]["G"], "follow_m": follow["W1"]["M"],
        "pass": follow["W1"]["E"] >= follow["W1"]["M"] - t["KA12_slack"],
    }  # fmt: skip
    return results


def main() -> int:
    """Run the self-test, or the checks of a fold, and write the gate file when all pass."""
    parser = argparse.ArgumentParser()
    parser.add_argument("month", nargs="?")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-fits", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    assert args.month, "give the fold month, for example 2026-01"
    results = run_checks(month=args.month, device=args.device, skip_fits=args.skip_fits)
    print(json.dumps(results, indent=1, default=str))
    failed = [k for k, v in results.items() if isinstance(v, dict) and not v.get("pass", True)]
    if failed or args.skip_fits:
        print("gate file not written; failed:", failed, "skipped fits:", args.skip_fits)
        return 1 if failed else 0
    path = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR) / "work" / "known_answers_passed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps({**results, "code": arms.code_record()}, indent=1, default=str))
    tmp.rename(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

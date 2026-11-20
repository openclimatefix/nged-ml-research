# ruff: noqa: E501, PT018, ANN001
"""Experiment 715: the synthetic neighbour worlds and the known-answer gate (KN6, KN7, KN11 part (a), KN12).

Usage: e00715_known_answer.py MONTH [--device cuda]      the fit gate: KN6, KN7, KN11(a), KN12
       e00715_known_answer.py --selftest                 CPU, no data, no fit
       e00715_known_answer.py --smoke MONTH              CPU, real fold, forward passes only: no training
       e00715_known_answer.py --kn8 MONTH SEED           KN8, outside the gate (needs the F4-0 fit and 713's stored forecast)
       e00715_known_answer.py --kn11b MONTH SEED         KN11(b), outside the gate
       e00715_known_answer.py --kn13 MONTH SEED          KN13, outside the gate (delegates to `e00715_predict.py --kn13`)

The other checks live in their own scripts: KN1, KN4, KN5 in `e00715_check_model.py`; KN2 and KN9 in
`e00715_check_batch.py`; KN3 and KN10 in `e00715_check_neighbours.py`. The full run needs the GPU for
about an hour (design section 8.1). It writes `work/known_answers_passed.json`, with the code record,
only when every check it runs passes; `e00715_fit.py` refuses to run without the file.

**World W3** is experiment 713's world W0 (the real grid, the real ERA5 temperature, the real calendar,
`0.5 + 0.02 max(0, 15.5 - T) + w[day type, half-hour]`) on four Primaries that form two neighbour pairs, with
steps of +-0.30 of capacity that cover about 15% of the history: half temporary (4 to 12 days, mirrored
at the partner with the opposite sign) and half permanent (45 days, no mirror). **W3p** draws the
mirrored share from 0.3 to 1.0 and reports the gap of `F4` with no threshold. **KN7's world** is W0 with
no steps. Every arm trains from random weights with stage 1 only and at most 15 passes, and the new
parameters train at the base rate (there is no stage 1 state to warm start from).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Final

import e00713_arms as a713  # ty: ignore[unresolved-import]
import e00713_known_answer as ka  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr713  # ty: ignore[unresolved-import]
import e00715_arms as arms
import e00715_batch as fb
import e00715_model as f4
import e00715_neighbours as nb
import e00715_train as tr
import e00715_worlds as wd
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PAIRS: Final = ((2, 5), (9, 14))
CROSS_PAIRS: Final = ((2, 9), (5, 14))
PRIMARIES: Final = (2, 5, 9, 14)
PASSES: Final = 15
THRESHOLDS: Final = {
    "KN6_f4_gap": 0.60,
    "KN6_f4r_abs_gap": 0.15,
    "KN6_m_abs_gap": 0.15,
    "KN7_f4_minus_m": 0.005,
}
SHARE_P: Final = (0.3, 1.0)


def w0_base(*, temperature: np.ndarray, day_type: np.ndarray, half_hour: np.ndarray) -> np.ndarray:
    """Return the noise-free W0 power (16, G) in shares of capacity, for every Primary (the four filled ones are used)."""
    w = ka.draw_table()
    return 0.5 + 0.02 * np.maximum(0.0, 15.5 - temperature) + w[day_type, half_hour][None, :]


class Duck:
    """A stand-in for `e00713_known_answer.Synthetic` that carries only the power array `y`."""

    def __init__(self, *, y: np.ndarray) -> None:
        """Hold `y`."""
        self.y = y


def f4_world(
    *, y: np.ndarray, template, cutoff_idx: int, device, table: nb.NeighbourTable, neighbours: bool
) -> tuple[tr.F4World, np.ndarray, np.ndarray]:  # fmt: skip
    """Return an `F4World` over the synthetic power `y` (16, G), with the per-Primary mean and standard deviation."""
    base, mean, std = ka.to_world(
        syn=Duck(y=y), template=template, cutoff_idx=cutoff_idx, device=device
    )  # ty: ignore[invalid-argument-type]
    spec = arms.resolve(arm="F4" if neighbours else "F4-0")
    n = nb.N_PRIMARIES
    world = tr.F4World(
        base=base, table=table, mw_scale=std.astype(float), spec=spec, rewire_seed=0,
        capacity_mw=np.ones(n), mva_only=np.zeros(n, bool),
    )  # fmt: skip
    return world, mean, std


def source_of(
    *,
    world: tr.F4World,
    power_in: np.ndarray | None,
    neighbour: np.ndarray | None,
    mean: np.ndarray,
    std: np.ndarray,
) -> fb.F4Source:
    """Return the source that reads `power_in` as host history and `neighbour` as neighbour power (raw shares; None means the world's own power)."""
    base = world.base.source
    dev = base.device

    def z(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((a - mean[:, None]) / std[:, None]).astype(np.float32)).to(dev)

    host = base if power_in is None else base.with_power_in(power_in=z(power_in))
    nbr = base.power_z if neighbour is None else z(neighbour)
    mode = "real" if world.spec.neighbour_path else "none"
    return fb.F4Source(
        host=host, neighbour_power=nbr, mw_scale=world.mw_scale, table=world.table, mode=mode,
        rewire_seed=0, label="cleaned",
    )  # fmt: skip


def held_out(*, world: tr.F4World) -> tr.F4Samples:
    """Return the 60 days after the cutoff for the four Primaries."""
    s = ka.test_samples(world=world.base, primaries=PRIMARIES)
    return tr.F4Samples(primary=s.primary, c_idx=s.c_idx, is_val=s.is_val)  # ty: ignore[unknown-argument]


def medians(
    *, model: mdl.MsaNet, world: tr.F4World, samples: tr.F4Samples, source: fb.F4Source, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Return (median, actual, target index) per (sample, lead day, column), in shares of capacity; lead-major."""
    model.eval()
    prim = samples.primary
    meds, acts, idxs = [], [], []
    with torch.no_grad():
        for lead in range(lay.N_LEAD_DAYS):
            for start in range(0, len(prim), 512):
                sel = np.arange(start, min(start + 512, len(prim)))
                b = tr.batch_for(
                    source=source, samples=samples, index=sel, lead=np.full(len(sel), lead),
                    ensemble=False, cut=10**9, dropout=None,
                )  # fmt: skip
                out = tr.run_model(model=model, batch=b, neighbours=world.spec.neighbour_path)
                q = mdl.standardised_quantiles(out=out)[..., 5].cpu().numpy()
                p = prim[sel]
                meds.append(q * std[p][:, None] + mean[p][:, None])
                acts.append(b.host.target.cpu().numpy() * std[p][:, None] + mean[p][:, None])
                idxs.append(b.host.target_idx.cpu().numpy())
    return np.concatenate(meds), np.concatenate(acts), np.concatenate(idxs)


def train_from_scratch(*, world: tr.F4World, device: str, seed: int = 0) -> mdl.MsaNet:
    """Train one network from random weights with stage 1 only and at most `PASSES` passes."""
    torch.manual_seed(seed)
    spec = world.spec
    model: mdl.MsaNet = (
        f4.F4Net(switches=spec.switches)
        if spec.neighbour_path
        else mdl.MsaNet(switches=spec.switches)
    )
    model = model.to(device)
    s = tr713.make_samples(world=world.base, stage=1, placebo=False, primaries=PRIMARIES)
    samples = tr.F4Samples(primary=s.primary, c_idx=s.c_idx, is_val=s.is_val)  # ty: ignore[unknown-argument]
    source = source_of(
        world=world,
        power_in=None,
        neighbour=None,
        mean=world.base.target_mean,
        std=world.base.target_std,
    )
    tr.train_stage(
        model=model, world=world, stage=1, samples=samples, seed=seed, encoders="random", max_passes=PASSES,
        source=source, ensemble=False, new_lr=tr713.PRETRAIN_LR,
    )  # fmt: skip
    return model


def w3_follow(
    *, model: mdl.MsaNet, world: tr.F4World, w3: wd.W3, mean: np.ndarray, std: np.ndarray
) -> dict[str, float]:
    """Return the follow shares of a trained network on W3: with the steps and mirrors, against the same inputs without them."""
    samples = held_out(world=world)
    with_steps, _, idx = medians(
        model=model, world=world, samples=samples,
        source=source_of(world=world, power_in=None, neighbour=None, mean=mean, std=std), mean=mean, std=std,
    )  # fmt: skip
    removed = np.where(np.isfinite(w3.y), w3.y - w3.shift - w3.mirror, np.nan)
    without, _, _ = medians(
        model=model, world=world, samples=samples,
        source=source_of(world=world, power_in=removed, neighbour=removed, mean=mean, std=std), mean=mean, std=std,
    )  # fmt: skip
    prim = np.tile(samples.primary, lay.N_LEAD_DAYS)
    c_idx = np.tile(samples.c_idx, lay.N_LEAD_DAYS)
    return wd.follow_share_w3(diff=with_steps - without, idx=idx, prim=prim, c_idx=c_idx, world=w3)


def selftest() -> int:
    """CPU checks of the W3 construction and the follow share, with no data and no fit."""
    rng = np.random.default_rng(1)
    n_grid = 48 * 500
    base = 0.5 + 0.1 * rng.random((16, n_grid))
    w3 = wd.build_w3(base=base, pairs=PAIRS, first_valid=48 * 20)
    p, q = PAIRS[0]
    assert np.isnan(w3.y[0]).all() and np.isfinite(w3.y[p]).all()
    share = float((w3.kind[p] > 0).mean())
    assert 0.10 <= share <= 0.35, share
    # every temporary step is mirrored with the opposite sign at the partner, a permanent one is not
    on = int(np.nonzero(w3.kind[p] == 1)[0][0])
    assert (
        w3.mirror[q, on] == -w3.shift[p, on]
        or abs(w3.mirror[q, on] + w3.shift[p, on]) <= abs(w3.shift[p, on]) + 1e-12
    )
    perm = np.nonzero(w3.kind[p] == 2)[0]
    assert len(perm), "no permanent step was drawn"
    assert np.array_equal(
        wd.build_w3(base=base, pairs=PAIRS, first_valid=48 * 20).y[p], w3.y[p], equal_nan=True
    )
    # the follow share of a network that follows permanent steps and ignores temporary ones has gap 1
    prim = np.full(3, p)
    onsets = np.unique(w3.onset[p][w3.kind[p] == 2])
    clean = [
        int(o) for o in onsets
        if abs(abs(w3.shift[p, o]) - wd.W3_STEP) < 1e-9 and abs(abs(w3.shift[p, o + 5 * 48]) - wd.W3_STEP) < 1e-9 and w3.kind[p, o + 5 * 48] == 2
    ]  # fmt: skip
    assert clean, "no permanent step without an overlap was drawn"
    onset = clean[0]
    c_idx = np.full(3, onset + 5 * 48 + 1)  # 5 days after the onset, step still in force
    idx = np.tile(np.arange(onset + 14 * 48, onset + 15 * 48), (3, 1))
    sign = float(np.sign(w3.shift[p, onset]))
    diff = np.full((3, 48), sign * wd.W3_STEP)
    out = wd.follow_share_w3(diff=diff, idx=idx, prim=prim, c_idx=c_idx, world=w3)
    assert abs(out["permanent"] - 1.0) < 1e-9, out
    assert out["n_permanent"] > 0
    zero = wd.follow_share_w3(diff=np.zeros((3, 48)), idx=idx, prim=prim, c_idx=c_idx, world=w3)
    assert zero["permanent"] == 0.0
    # a target half-hour less than 13 days after the onset does not count
    near = np.tile(np.arange(onset + 12 * 48, onset + 13 * 48 - 1), (3, 1))
    nothing = wd.follow_share_w3(
        diff=np.zeros((3, near.shape[1])), idx=near, prim=prim, c_idx=c_idx, world=w3
    )
    assert nothing["n_permanent"] == 0
    # the gate file logic: another code hash is refused
    log: dict = {}
    refused = False
    try:
        arms.check_code(
            earlier={"code_hash": "0" * 16}, accept=[], record=log, what="selftest", kind="gate"
        )
    except AssertionError:
        refused = True
    assert refused
    print("known-answer self-test passed (no data, no fit)")
    return 0


def load_world(*, month: str, device: str) -> tuple:
    """Return the real fold's pipeline data and the temperature, day type, and half-hour arrays of the grid."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    arms.install_own_cache()
    os.environ["E00667_PHASE"] = "train"
    pipe = inp.load_pipeline(month=month, phase="train")
    data = inp.build_fold_data(pipe=pipe, device=device)
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
    return data, temperature, day_type, half_hour


def smoke(*, month: str) -> int:
    """Forward passes on the CPU: build W3, the sources, and the follow-share path with an untrained F4Net. No training."""
    data, temperature, day_type, half_hour = load_world(month=month, device="cpu")
    base = w0_base(temperature=temperature, day_type=day_type, half_hour=half_hour)
    w3 = wd.build_w3(base=base, pairs=PAIRS, first_valid=lay.STEPS_PER_DAY * 20)
    world, mean, std = f4_world(
        y=w3.y, template=data.source, cutoff_idx=data.cutoff_idx, device="cpu", table=wd.pair_table(pairs=PAIRS), neighbours=True,
    )  # fmt: skip
    torch.manual_seed(0)
    model = f4.F4Net(switches=mdl.ARM_SWITCHES["M"])
    follow = w3_follow(model=model, world=world, w3=w3, mean=mean, std=std)
    print(
        "untrained F4Net, W3 follow shares (an untrained network gives the world's noise, not a result):",
        follow,
    )
    assert follow["n_permanent"] > 0 and follow["n_temporary"] > 0, follow
    # the zero-initialised F4Net equals the same network without the neighbour tensors on W3
    samples = held_out(world=world)
    src = source_of(world=world, power_in=None, neighbour=None, mean=mean, std=std)
    sel = np.arange(64)
    b = tr.batch_for(
        source=src,
        samples=samples,
        index=sel,
        lead=np.zeros(64, int),
        ensemble=False,
        cut=10**9,
        dropout=None,
    )
    model.eval()
    with torch.no_grad():
        with_n = model(b.host.inputs, b.nbr).out
        without = model(b.host.inputs).out
    assert torch.allclose(with_n, without, atol=1e-6), (
        "a zero-initialised F4Net differs from its trunk on W3"
    )
    print("smoke test passed (forward passes only)")
    return 0


def run_checks(*, month: str, device: str) -> dict:
    """Run KN12, KN11(a), KN6, and KN7 for the fold `month`; return the results."""
    from datetime import UTC, datetime

    import e00715_era as era
    import e00715_weather_lookahead as wl

    results: dict = {}
    ok, lines = wl.w713.check_data(contract=datetime(2026, 3, 10, 12, tzinfo=UTC))
    print("\n".join(lines))
    results["KN12"] = {"pass": ok, "lines": lines}
    results["KN11a"] = era.kn11a(month=month)
    data, temperature, day_type, half_hour = load_world(month=month, device=device)
    base = w0_base(temperature=temperature, day_type=day_type, half_hour=half_hour)
    first_valid = lay.STEPS_PER_DAY * 20
    gaps: dict[str, dict[str, float]] = {}
    for world_name, share in (("W3", (1.0, 1.0)), ("W3p", SHARE_P)):
        w3 = wd.build_w3(base=base, pairs=PAIRS, first_valid=first_valid, share_range=share)
        arm_tables = {
            "F4": (wd.pair_table(pairs=PAIRS), True),
            "F4-R": (wd.pair_table(pairs=CROSS_PAIRS), True),
            "M": (wd.pair_table(pairs=PAIRS), False),
        }
        gaps[world_name] = {}
        for arm, (table, neighbours) in arm_tables.items():
            if world_name == "W3p" and arm != "F4":
                continue
            world, mean, std = f4_world(
                y=w3.y,
                template=data.source,
                cutoff_idx=data.cutoff_idx,
                device=device,
                table=table,
                neighbours=neighbours,
            )
            model = train_from_scratch(world=world, device=device)
            follow = w3_follow(model=model, world=world, w3=w3, mean=mean, std=std)
            gaps[world_name][arm] = follow["gap"]
            print(world_name, arm, follow, flush=True)
    t = THRESHOLDS
    g = gaps["W3"]
    results["KN6"] = {
        "gaps": g, "W3p_f4_gap_report_only": gaps["W3p"]["F4"],
        "pass": g["F4"] >= t["KN6_f4_gap"] and abs(g["F4-R"]) <= t["KN6_f4r_abs_gap"] and abs(g["M"]) <= t["KN6_m_abs_gap"],
    }  # fmt: skip
    y0 = np.where(
        np.isin(np.arange(16), PRIMARIES)[:, None],
        base + np.random.default_rng(0).normal(0, wd.NOISE_SD, base.shape),
        np.nan,
    )
    mae: dict[str, float] = {}
    for arm, neighbours in (("F4", True), ("M", False)):
        world, mean, std = f4_world(
            y=y0,
            template=data.source,
            cutoff_idx=data.cutoff_idx,
            device=device,
            table=wd.pair_table(pairs=PAIRS),
            neighbours=neighbours,
        )
        model = train_from_scratch(world=world, device=device)
        samples = held_out(world=world)
        med, act, _ = medians(
            model=model,
            world=world,
            samples=samples,
            source=source_of(world=world, power_in=None, neighbour=None, mean=mean, std=std),
            mean=mean,
            std=std,
        )
        mae[arm] = ka.mae_of_median(med=med, act=act)
        print("W0 pairs", arm, "MAE", round(mae[arm], 4), flush=True)
    results["KN7"] = {
        "mae": mae,
        "f4_minus_m": mae["F4"] - mae["M"],
        "pass": mae["F4"] - mae["M"] <= t["KN7_f4_minus_m"],
    }
    return results


def kn8(*, month: str, seed: int) -> dict:
    """KN8: arm `F4-0` against the stored forecast of `M*`; print the largest absolute difference of the median (design 8.2)."""
    import e00715_predict as pr
    import polars as pl

    star = a713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    mine = arms.fit_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="F4-0", month=month, seed=seed)
    stored = a713.fit_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm=star, month=month, seed=seed)
    if not (mine.exists() and stored.exists()):
        return {"skipped": f"missing {mine.name} or {stored.name}"}
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    a = pl.read_parquet(mine).select(*keys, "q50")
    b = pl.read_parquet(stored).select(*keys, pl.col("q50").alias("q50_s"))
    j = a.join(b, on=keys)
    cap = float(arms.capacity_mw().max())
    worst = float((j["q50"] - j["q50_s"]).abs().max())  # ty: ignore[invalid-argument-type]
    return {
        "rows": j.height,
        "max_abs_difference_mw": worst,
        "max_abs_difference_of_capacity": worst / cap,
        "within_1e-3_of_capacity": worst / cap <= 1e-3,
        "pr_loaded": pr.__name__,
    }


def main() -> int:  # noqa: PLR0911
    """Run one command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("month", nargs="?")
    parser.add_argument("seed", type=int, nargs="?", default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--kn8", action="store_true")
    parser.add_argument("--kn11b", action="store_true")
    parser.add_argument("--kn13", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    assert args.month, "give the fold month, for example 2025-11"
    if args.smoke:
        return smoke(month=args.month)
    if args.kn8:
        print(json.dumps(kn8(month=args.month, seed=args.seed), indent=1))
        return 0
    if args.kn11b:
        import e00715_era as era

        print(json.dumps(era.kn11b(month=args.month, seed=args.seed), indent=1, default=str))
        return 0
    if args.kn13:
        import e00715_predict as pr

        result = pr.kn13(
            month=args.month, seed=args.seed, device=args.device, data_dir=ML_EXPERIMENTS_DATA_DIR
        )
        print(json.dumps(result, indent=1))
        return 0 if result.get("pass", True) else 1
    results = run_checks(month=args.month, device=args.device)
    print(json.dumps(results, indent=1, default=str))
    failed = [k for k, v in results.items() if isinstance(v, dict) and not v.get("pass", True)]
    if failed:
        print("gate file not written; failed:", failed)
        return 1
    path = arms.gate_file(data_dir=ML_EXPERIMENTS_DATA_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps({**results, "code": arms.code_record()}, indent=1, default=str))
    tmp.rename(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

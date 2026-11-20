# ruff: noqa: E501, PT018, PLR0915
"""Experiment 713: report-only additions that keep the pre-registered contrasts unchanged.

None of these changes H1, H2, the verdict rule, or the thresholds of R1 to R3. Each is a separate
sub-command, so a reviewer can skip the file.

    e00713_extras.py splits-era          # (1) R2 by kind and by week since the onset, R3 inside/after
    e00713_extras.py splits-r1           # (1) R1 before and after 2026-03-01, and 16:00-19:00 weekdays
    e00713_extras.py internals ARM MONTH SEED [--device cuda]   # (2) read-out weights, (3) target vector
    e00713_extras.py w2 MONTH [--device cuda]   # (5) mixed-step diagnostic world; --selftest has no data
    (4) the two bank read-outs are `e00713_era.py bank ARM MONTH SEED --variant shuffle|mask9`.

**Splits (1).** R2's two slices (0 to 7 and 7 to 28 days after the onset) are cut into weeks, by the
`kind` column of the truth table (the three level steps and the solar-shaped reduction). R3 is split
into the days inside the move and the 7 days after its end. R1 is read on the months before and from
2026-03-01, and on weekday half-hours from 16:00 to 19:00 local time against the other half-hours.

**Internals (2) and (3).** For every host sample of the bank and every D8 test sample at lead days
1, 7, and 14, the file `internals/<arm>_<month>_s<seed>.npz` holds the read-out weights (14, 48), the
weighted mean `base`, the weighted spread, and the 64-number vector per target token that enters the
head (float16). The attention hook of the network is the source (`return_attention=True`).

**Diagnostic world W2 (5).** Each Primary has lasting steps (they stay to the end of the history) and
10-day steps. The Bayes-optimal follow share is printed by days since the onset beside the network's:
a step that is still active d days after its onset is lasting with probability `p_l / (p_l + p_t)`
while `d < 10` and is lasting for certain from `d >= 10`, and a 10-day step is still active at lead
`k` days when `d + k < 10`. The world is a diagnostic and gates nothing.
"""

import argparse
import sys
from typing import Final

import e00713_arms as arms
import e00713_layout as lay
import numpy as np
import polars as pl

STEP_DAYS: Final = 10
WEEK_WINDOWS: Final = ((0, 7), (7, 14), (14, 21), (21, 28))
R1_SPLIT_MONTH: Final = "2026-03"
INTERNAL_LEADS: Final = (0, 6, 13)
"""Lead days 1, 7, and 14 as zero-based indices."""


def bayes_follow(
    *, days_since_onset: np.ndarray, lead_days: np.ndarray, p_lasting: float
) -> np.ndarray:
    """Return the Bayes-optimal follow share of a step seen `days_since_onset` days after its onset.

    The share is the probability that the step is still active at the target, `lead_days` after the
    origin, given that it is active at the origin. Steps are lasting or last `STEP_DAYS` days.
    """
    d = np.asarray(days_since_onset, dtype=np.float64)
    k = np.asarray(lead_days, dtype=np.float64)
    p_l = np.where(d < STEP_DAYS, p_lasting / (p_lasting + (1.0 - p_lasting)), 1.0)
    still_ten = (d + k < STEP_DAYS).astype(np.float64)
    return p_l + (1.0 - p_l) * still_ten


def mixed_steps(
    *, n_grid: int, first_valid: int, rng: np.random.Generator, share: float = 0.10,
    n_lasting: int = 2, size: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:  # fmt: skip
    """Return (shift, step_start, step_kind, n_lasting_events, n_ten_day_events) of one Primary.

    `step_kind` is 1 for a lasting step and 2 for a 10-day step, 0 elsewhere.
    """
    length = STEP_DAYS * lay.STEPS_PER_DAY
    shift = np.zeros(n_grid)
    start = np.full(n_grid, -1, dtype=np.int64)
    kind = np.zeros(n_grid, dtype=np.int8)
    n_l = n_t = 0
    for _ in range(n_lasting):
        on = int(rng.integers(first_valid, first_valid + int(0.6 * (n_grid - first_valid))))
        sign = float(rng.choice([-1.0, 1.0]))
        free = shift[on:] == 0
        shift[on:][free] = sign * size
        start[on:][free] = on
        kind[on:][free] = 1
        n_l += 1
    covered = 0
    while covered < share * (n_grid - first_valid):
        on = int(rng.integers(first_valid, n_grid - length))
        sign = float(rng.choice([-1.0, 1.0]))
        window = slice(on, on + length)
        free = shift[window] == 0
        shift[window][free] = sign * size
        start[window][free] = on
        kind[window][free] = 2
        covered += length
        n_t += 1
    return shift, start, kind, n_l, n_t


def selftest() -> int:
    """Check the Bayes share and the mixed steps without data."""
    d = np.array([0, 5, 9, 10, 20])
    k = np.array([1, 1, 3, 2, 10])
    got = bayes_follow(days_since_onset=d, lead_days=k, p_lasting=0.5)
    assert np.allclose(got, [1.0, 1.0, 0.5, 1.0, 1.0]), got
    assert np.allclose(
        bayes_follow(days_since_onset=np.array([2]), lead_days=np.array([12]), p_lasting=0.25),
        [0.25],
    )
    shift, _start, kind, n_l, n_t = mixed_steps(
        n_grid=48 * 400, first_valid=48 * 20, rng=np.random.default_rng(0)
    )
    assert n_l == 2 and n_t >= 1 and set(np.unique(kind)) <= {0, 1, 2}
    assert bool((kind[-1] == 1) or (shift[-1] == 0)), "a lasting step stays to the end"
    assert set(np.unique(np.abs(shift))) <= {0.0, 0.30}
    print("extras self-test passed: Bayes follow share and the mixed steps")
    return 0


def weekly_rows(
    *, forecast: pl.DataFrame, events: pl.DataFrame, ids: np.ndarray, fold: str, seed: int, arm: str
) -> list[dict]:
    """Return the host's mean CRPS by event kind and window (weeks for an era; inside, after for a move)."""
    import e00709_features as features  # ty: ignore[unresolved-import]
    from studies.eval_protocol.metrics import quantile_score

    q_cols = [c for c in forecast.columns if c.startswith("q")]
    origin_day = (
        (
            forecast["power_fcst_init_time"].dt.replace_time_zone(None)
            - pl.Series([features.GRID_START.replace(tzinfo=None)] * forecast.height)
        )
        .dt.total_days()
        .to_numpy()
    )
    site = forecast["time_series_id"].to_numpy()
    q = forecast.select(q_cols).to_numpy().astype(np.float64)
    out = []
    for index, event in enumerate(events.iter_rows(named=True)):
        host = site == ids[event["host"]]
        onset = event["onset_day"]
        if event["truth"] == "era":
            windows = {f"week {lo // 7 + 1}": (onset + lo, onset + hi) for lo, hi in WEEK_WINDOWS}
            column = "truth_share"
        else:
            windows = {
                "inside the move": (onset, event["end_day"]),
                "7 days after the end": (event["end_day"], event["end_day"] + 7),
            }
            column = "clean_share"
        truth = forecast[column].to_numpy().astype(np.float64)
        for label, (lo, hi) in windows.items():
            sel = host & (origin_day >= lo) & (origin_day < hi) & np.isfinite(truth)
            if sel.any():
                crps = float(
                    np.mean(quantile_score(actual=truth[sel] * 100.0, quantiles=q[sel] * 100.0))
                )
                out.append(
                    {
                        "arm": arm,
                        "fold": fold,
                        "seed": seed,
                        "pass": index,
                        "kind": event["kind"],
                        "truth": event["truth"],
                        "window": label,
                        "crps": crps,
                    }
                )
    return out


def cmd_splits_era() -> None:
    """Write R2 by kind and week, and R3 inside and after, as the mean difference of arm minus `C` per pass."""
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_era as era
    import e00713_score as score

    star = arms.m_star(data_dir=era.ML_EXPERIMENTS_DATA_DIR)
    ids = k578.sites()["time_series_id"].to_numpy()
    rows = []
    for arm in (star, "E", "P", "K"):
        for month in arms.BANK_FOLDS:
            for seed in arms.SCREEN_SEEDS:
                for path in sorted(
                    (era.BANK713 / f"{arm}_{month}_s{seed}").glob("pass[0-9][0-9].parquet")
                ):
                    forecast = pl.read_parquet(path)
                    events = pl.read_parquet(path.with_name(path.stem + "_truth.parquet"))
                    rows += weekly_rows(
                        forecast=forecast, events=events, ids=ids, fold=month, seed=seed, arm=arm
                    )
                    ref = pl.read_parquet(era.D709 / "bank" / f"none_{month}_s{seed}" / path.name)
                    rows += weekly_rows(
                        forecast=ref, events=events, ids=ids, fold=month, seed=seed, arm="C"
                    )
    frame = (
        pl.DataFrame(rows)
        .group_by("arm", "fold", "pass", "kind", "window")
        .agg(pl.col("crps").mean())
    )
    base = frame.filter(pl.col("arm") == "C").rename({"crps": "crps_c"}).drop("arm")
    out = {}
    for arm in (star, "E", "P", "K"):
        d = frame.filter(pl.col("arm") == arm).join(base, on=["fold", "pass", "kind", "window"])
        d = d.with_columns(diff=pl.col("crps") - pl.col("crps_c"))
        for (kind, window), g in d.group_by("kind", "window"):
            by_pass = g.group_by("fold", "pass").agg(pl.col("diff").mean())["diff"].to_numpy()
            lo, hi = era.pass_bootstrap(diffs=by_pass)
            out[f"{arm}-C:{kind}:{window}"] = {
                "mean": float(by_pass.mean()),
                "lower": lo,
                "upper": hi,
                "n_passes": len(by_pass),
            }
    score.write(name="era_splits_report_only", obj=out)


def cmd_splits_r1() -> None:
    """Write R1 (D8 alone) before and from 2026-03-01, and for weekday 16:00-19:00 against the other half-hours."""
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_era as era
    import e00713_score as score

    star = arms.m_star(data_dir=era.ML_EXPERIMENTS_DATA_DIR)
    sites = k578.sites()
    groups = {"d8": sites.filter(pl.col("label") == "D8")["time_series_id"].to_numpy()}
    truth = score.truth_stack()
    names = [star, "E", "C"]
    loaded = {
        n: score.load_scored(arm=n, seeds=arms.SCREEN_SEEDS, months=arms.ALL_FOLDS, truth=truth)
        for n in names
    }
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    peak = (local.dt.weekday() <= 5) & (local.dt.hour() >= 16) & (local.dt.hour() < 19)
    cuts = {
        f"months before {R1_SPLIT_MONTH}": pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
        < R1_SPLIT_MONTH,
        f"months from {R1_SPLIT_MONTH}": pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
        >= R1_SPLIT_MONTH,
        "weekday 16:00 to 19:00": peak,
        "other half-hours": ~peak,
    }
    out = {}
    for label, expr in cuts.items():
        per_seed = {n: loaded[n][0].filter(expr) for n in names}
        averaged = {n: loaded[n][1].filter(expr) for n in names}
        for t, r in (("E", star), ("E", "C"), (star, "C")):
            out[f"{label}:{t}-{r}"] = score.contrast(
                per_seed=per_seed,
                averaged=averaged,
                treatment=t,
                reference=r,
                group="d8",
                groups=groups,
                level=95.0,
            )
    score.write(name="era_r1_splits_report_only", obj=out)


def cmd_internals(*, arm: str, month: str, seed: int, device: str) -> None:
    """Store the read-out weights and the target-row vector of the D8 test samples at three lead days."""
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_bank as bank
    import e00713_inputs as inp
    import e00713_model as mdl
    import e00713_train as tr
    import torch  # ty: ignore[unresolved-import]
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    path = (
        arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR)
        / "work"
        / "internals"
        / f"{arm}_{month}_s{seed}.npz"
    )
    if path.exists():
        return
    spec = arms.resolve(arm=arm, data_dir=ML_EXPERIMENTS_DATA_DIR)
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    model = bank.load_model(
        spec=spec, arm=arm, month=month, seed=seed, device=device, data_dir=ML_EXPERIMENTS_DATA_DIR
    )
    d8 = int(k578.sites().filter(pl.col("label") == "D8")["primary"][0])
    assert data.test_origin_idx is not None
    c = data.test_origin_idx + lay.NATIVE_OFFSET_STEPS
    samples = tr.SampleSet(np.full(len(c), d8), c, np.zeros(len(c), bool))
    keep: dict[str, list[np.ndarray]] = {
        "alpha": [],
        "base": [],
        "spread": [],
        "vector": [],
        "c_idx": [],
        "lead": [],
    }
    for lead in INTERNAL_LEADS:
        sel = np.arange(len(c))
        b = tr.batch_for(
            source=data.source,
            samples=samples,
            index=sel,
            lead=np.full(len(sel), lead),
            ensemble=True,
            cut=10**9,
            dropout=None,
        )
        with torch.no_grad():
            res = model(b.inputs, return_attention=True)
        keep["alpha"].append(res.alpha.cpu().numpy().astype(np.float16))
        keep["base"].append(res.base.cpu().numpy().astype(np.float32))
        keep["spread"].append(res.spread.cpu().numpy().astype(np.float32))
        assert res.target_vector is not None
        keep["vector"].append(res.target_vector.cpu().numpy().astype(np.float16))
        keep["c_idx"].append(c)
        keep["lead"].append(np.full(len(c), lead))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **{k: np.concatenate(v) for k, v in keep.items()},  # ty: ignore[invalid-argument-type]
        primary=d8,
        mdl_width=mdl.WIDTH,  # ty: ignore[invalid-argument-type]
    )
    tmp.rename(path)
    print("stored", path, flush=True)


def cmd_w2(*, month: str, device: str) -> None:
    """Fit arm `M` on the mixed-step world and print its follow share beside the Bayes-optimal share."""
    import os

    os.environ["E00667_PHASE"] = "train"
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00713_inputs as inp
    import e00713_known_answer as ka
    import e00713_model as mdl
    import e00713_train as tr
    import torch  # ty: ignore[unresolved-import]

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
    syn = ka.build_world(
        kind="W0",
        temperature=temperature,
        day_type=day_type,
        half_hour=half_hour,
        first_valid=lay.STEPS_PER_DAY * 20,
    )
    rng = np.random.default_rng([ka.WORLD_SEED, 3])
    kind = np.zeros(syn.y.shape, dtype=np.int8)
    n_l = n_t = 0
    for p in ka.PRIMARIES:
        shift, start, k, a, b = mixed_steps(
            n_grid=n_grid, first_valid=lay.STEPS_PER_DAY * 20, rng=rng
        )
        syn.shift[p], syn.step_start[p], kind[p] = shift, start, k
        syn.y[p] = syn.y[p] + shift
        n_l, n_t = n_l + a, n_t + b
    p_lasting = n_l / (n_l + n_t)
    world, mean, std = ka.to_world(
        syn=syn, template=data.source, cutoff_idx=data.cutoff_idx, device=device
    )
    held = ka.test_samples(world=world, primaries=ka.PRIMARIES)
    torch.manual_seed(0)
    model = mdl.MsaNet(switches=mdl.ARM_SWITCHES["M"]).to(device)
    samples = tr.make_samples(world=world, stage=1, placebo=False, primaries=ka.PRIMARIES)
    tr.train_stage(
        model=model,
        world=world,
        stage=1,
        samples=samples,
        seed=0,
        encoders="random",
        max_passes=ka.SYNTHETIC_PASSES,
    )
    z_clean = (syn.y - syn.shift - mean[:, None]) / std[:, None]
    without = world.source.with_power_in(
        power_in=torch.from_numpy(z_clean.astype(np.float32)).to(device)
    )
    with_shift, _, idx = ka.predict_medians(
        model=model,
        world=world,
        samples=held,
        source=world.source,
        use_placebo=False,
        mean=mean,
        std=std,
    )
    no_shift, _, _ = ka.predict_medians(
        model=model,
        world=world,
        samples=held,
        source=without,
        use_placebo=False,
        mean=mean,
        std=std,
    )
    prim = np.tile(held.primary, lay.N_LEAD_DAYS)[: len(idx)]
    c_idx = np.tile(held.c_idx, lay.N_LEAD_DAYS)[: len(idx)]
    lead = np.repeat(np.arange(lay.N_LEAD_DAYS), len(held.primary))[: len(idx)]
    step = syn.shift[prim[:, None], idx]
    began = syn.step_start[prim[:, None], idx]
    d = (c_idx[:, None] - began) / lay.STEPS_PER_DAY
    ok = (step != 0) & (began >= 0) & (d >= 0)
    net = (with_shift - no_shift) * np.sign(step) / ka.STEP_SIZE
    k_days = np.broadcast_to((lead + 1)[:, None], d.shape)
    best = bayes_follow(days_since_onset=d, lead_days=k_days, p_lasting=p_lasting)
    print(
        f"W2: p_lasting {p_lasting:.2f}; follow share by whole days since the onset (network | Bayes-optimal)"
    )
    for lo in range(0, 21, 2):
        m = ok & (d >= lo) & (d < lo + 2)
        if m.any():
            print(
                f"  days {lo:2d}-{lo + 2:2d}: {net[m].mean():.3f} | {best[m].mean():.3f}  (n={int(m.sum())})"
            )


def main() -> int:
    """Dispatch the sub-command."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("splits-era")
    sub.add_parser("splits-r1")
    i = sub.add_parser("internals")
    i.add_argument("arm")
    i.add_argument("month")
    i.add_argument("seed", type=int)
    w = sub.add_parser("w2")
    w.add_argument("month", nargs="?")
    w.add_argument("--selftest", action="store_true")
    for p in (i, w):
        p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "splits-era":
        cmd_splits_era()
    elif args.command == "splits-r1":
        cmd_splits_r1()
    elif args.command == "internals":
        cmd_internals(arm=args.arm, month=args.month, seed=args.seed, device=args.device)
    elif args.selftest:
        return selftest()
    else:
        cmd_w2(month=args.month, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ruff: noqa: E501
"""Experiment 729: the stages of the run (banks, gate, fits, predictions, scores, verdict).

Every stage reads and writes files under this experiment's private data folder, and writes
aggregates only to the `scores/` JSON files. No stage prints a month, a date, an engine size, a
series identifier, or a real value; log lines carry counts and scores. Files are written atomically.
The stages that need torch import `e00729_train` inside the function.
"""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import e00729_arms as arms
import e00729_banks as banks
import e00729_baselines as base
import e00729_core as core
import e00729_data as data
import e00729_leakage as leak
import e00729_paths as paths
import e00729_real as real
import e00729_score as score
import e00729_sim as sim
import numpy as np

Log = Callable[[str], None]
SPLIT_ROW = re.compile(r"^\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d{4}-\d{2})\s*\|\s*(\w+)\s*\|")
SIMULATOR_SOURCES = ("e00729_sim.py", "e00729_banks.py")


def write_json(*, path: Path, payload: dict) -> None:
    """Write a JSON file atomically."""
    data.atomic_write_text(path=path, text=json.dumps(payload, indent=1, default=float))


def read_json(*, path: Path) -> dict:
    """Read a JSON file."""
    return json.loads(path.read_text())


@dataclass
class Context:
    """The day table, the split, and the per-split day masks, built once per stage.

    Attributes:
        table: Every D8 day with its fields.
        cal: The calendar of base days only.
        month_split: The split of each base month.
        day_split: The split of each base day of `cal` (or "dropped").
        use: For each split, for each month, the mask over `cal` of days a segment may use.
        stats: Temperature standardisation from the training days.
        cut_day: The private cut as a day (never printed).
    """

    table: data.DayTable
    cal: sim.Calendar
    month_split: dict[str, str]
    day_split: np.ndarray
    use: dict[str, dict[str, np.ndarray]]
    stats: arms.InputStats
    cut_day: np.datetime64

    def months(self, *, split: str) -> list[str]:
        """Return the months of a split."""
        return sorted(m for m, s in self.month_split.items() if s == split)


def parse_private_splits(*, path: Path) -> dict[str, str]:
    """Read the month-to-split table of the private `splits.md`."""
    out = {}
    for line in path.read_text().splitlines():
        m = SPLIT_ROW.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def build_context(*, closure_days: frozenset[date] = frozenset()) -> Context:
    """Load the days, derive the split from the rule, and check it against the private file.

    Args:
        closure_days: Days within 24 hours of a logged closure at D8 (empty when no log loads).

    Returns:
        The context.

    Raises:
        CorrectnessFailureError: When the derived split differs from `PRIVATE/splits.md`.
    """
    table = data.load_day_table(closure_days=closure_days)
    cal = data.select_days(cal=table.calendar, mask=table.is_base)
    months, n_days, bsp_days = data.month_table(cal=cal)
    month_split = core.month_splits(months=months, n_days=n_days, bsp_days=bsp_days)
    private = paths.data_root() / "PRIVATE" / "splits.md"
    if private.exists() and parse_private_splits(path=private) != month_split:
        msg = "the split derived from the rule differs from PRIVATE/splits.md"
        raise leak.CorrectnessFailureError(msg)
    days = [d.astype("datetime64[D]").astype(date) for d in cal.dates]
    day_split = np.array(core.assign_day_splits(days=days, splits=month_split))
    leak.assert_split_integrity(days=days, day_splits=list(day_split), month_split=month_split)
    leak.assert_not_in_lockdown(month_split=month_split)
    keys = np.array([str(d)[:7] for d in cal.dates])
    use = {
        split: {
            m: (keys == m) & (day_split == split) for m in months if month_split.get(m) == split
        }
        for split in ("training", "validation", "test")
    }
    train_days = np.flatnonzero(day_split == "training")
    stats = arms.input_stats(cal=cal, train_days=train_days)
    cut_day = np.datetime64(data.private_cut().date(), "D")
    leak.assert_cut_applied(base_dates=cal.dates, cut_day=cut_day)
    leak.assert_before_holdout(dates=table.calendar.dates)
    leak.assert_cams_differs(cal=cal)
    return Context(table, cal, month_split, day_split, use, stats, cut_day)


def bank_path(*, kind: str, seed: int | None = None) -> Path:
    """Return the file of a bank."""
    name = f"{kind}_s{seed}.pkl" if seed is not None else f"{kind}.pkl"
    return paths.data_root() / "banks" / name


def stage_banks(*, ctx: Context, train_seeds: Sequence[int], log: Log) -> dict:
    """Build the validation, test, shifted-mix, and training banks, and run the leakage checks.

    Args:
        ctx: The context.
        train_seeds: The training seeds to build (0 to 4).
        log: Receives progress lines.

    Returns:
        The summary written to `banks/summary.json` (hashes, counts, reachability).
    """
    built: dict[str, banks.Bank] = {}
    for kind, split in (("validation", "validation"), ("test", "test"), ("shifted", "test")):
        built[kind] = banks.build_bank(
            cal=ctx.cal, kind=kind, seed=None, months=ctx.months(split=split),  # ty: ignore[invalid-argument-type]
            month_use=ctx.use[split],
        )  # fmt: skip
        banks.save_bank(bank=built[kind], path=bank_path(kind=kind))
        log(f"built the {kind} bank: {len(built[kind].segments)} segments")
    for seed in train_seeds:
        b = banks.build_bank(
            cal=ctx.cal, kind="train", seed=seed, months=ctx.months(split="training"),
            month_use=ctx.use["training"],
        )  # fmt: skip
        banks.save_bank(bank=b, path=bank_path(kind="train", seed=seed))
        built[f"train_s{seed}"] = b
        log(f"built the training bank of seed {seed}: {len(b.segments)} segments")
    leak.assert_seed_spaces(
        train_seeds=train_seeds, evaluation=[built[k] for k in ("validation", "test", "shifted")]
    )
    every = [s for b in built.values() for s in b.segments]
    leak.assert_parameter_ranges(segments=every)
    folder = Path(__file__).parent
    leak.assert_simulator_source_clean(paths=[folder / n for n in SIMULATOR_SOURCES])
    for seg in built["test"].segments[:: max(1, len(built["test"].segments) // 40)]:
        leak.assert_bsp_construction(seg=seg, cal=ctx.cal)
        leak.assert_s1_same_day(seg=seg, cal=ctx.cal)
    leak.assert_no_calendar_memorisation(specs=arms.ARM_SPECS)
    rows = score.rows_from_segments(segments=built["test"].segments, holidays=ctx.cal.holidays)
    summary = {
        "digests": {k: b.digest for k, b in built.items()},
        "reachability": score.reachability(rows=rows),
        "n_segments": {k: len(b.segments) for k, b in built.items()},
        "dropped_half_hours": {
            s: int(ctx.table.dropped_half_hours[ctx.table.is_base][ctx.day_split == s].sum())
            for s in ("training", "validation", "test")
        },
        "dropped_days_by_guard": int((ctx.day_split == "dropped").sum()),
        "closure_log_loaded": False,
        "leakage_pass": True,
    }
    write_json(path=paths.data_root() / "banks" / "summary.json", payload=summary)
    return summary


def stage_gate(*, ctx: Context, log: Log) -> dict:
    """Run the realism gate on folded aggregates and the BSP step relation, and write `gate.json`.

    Args:
        ctx: The context.
        log: Receives progress lines.

    Returns:
        The gate record (aggregates only).
    """
    train = banks.load_bank(path=bank_path(kind="train", seed=0))
    test = banks.load_bank(path=bank_path(kind="test"))
    cal_all = ctx.table.calendar
    out: dict = {"shares": {}}
    passes = []
    for label, era, worlds in (
        ("net", data.ERA_NET, ("engine",)),
        ("solar", data.ERA_SOLAR, ("both",)),
    ):
        real_days = cal_all.reading[ctx.table.era == era]
        real_day, real_hh = banks.near_zero_shares(r=real_days)
        syn_day, syn_hh = banks.bank_shares_by_world(bank=train, worlds=worlds)
        ok = core.gate_ratio_ok(synthetic=syn_day, real=real_day) and core.gate_ratio_ok(
            synthetic=syn_hh, real=real_hh
        )
        passes.append(ok)
        out["shares"][label] = {
            "real_day": real_day, "real_half_hour": real_hh, "synthetic_day": syn_day,
            "synthetic_half_hour": syn_hh, "pass": ok,
        }  # fmt: skip
        log(f"gate {label}: pass {ok}")
    record = data.metered_record()
    day_axis = cal_all.dates
    rec = data.record_by_day(record=record, days=day_axis)
    days = np.flatnonzero(np.isfinite(rec).any(axis=1))
    d_eng = real_step(values=rec[days])
    d_bsp = real_step(values=cal_all.bsp[days])
    fit = banks.step_fit(d_bsp=d_bsp, d_eng=d_eng)
    ci = (fit.slope - 1.96 * fit.se, fit.slope + 1.96 * fit.se)
    placebo = placebo_slopes(rec=rec[days], bsp=cal_all.bsp[days])
    edges = banks.real_bin_edges(d_eng=d_eng)
    synthetic = synthetic_stretches(
        bank=test, length=int(np.isfinite(rec[days]).sum()), edges=edges
    )
    gate = banks.bsp_gate(real=fit, real_ci=ci, synthetic=synthetic)
    out["bsp"] = {
        **gate, "real_slope_z_vs_placebo": float((fit.slope - np.mean(placebo)) / np.std(placebo)),
        "n_synthetic_stretches": len(synthetic),
    }  # fmt: skip
    passes.append(bool(gate["pass"]))
    out["pass"] = all(passes)
    write_json(path=paths.data_root() / "scores" / "gate.json", payload=out)
    return out


def real_step(*, values: np.ndarray) -> np.ndarray:
    """Return the half-hour steps of consecutive days' values, flattened (NaN across gaps)."""
    flat = values.reshape(-1)
    out = np.full_like(flat, np.nan)
    out[1:] = flat[1:] - flat[:-1]
    return out


def placebo_slopes(*, rec: np.ndarray, bsp: np.ndarray) -> np.ndarray:
    """Return the slopes with the record shifted by 3 to 20 whole days either way (circular)."""
    lo, hi = core.GATE_PLACEBO_SHIFTS
    slopes = []
    d_bsp = real_step(values=bsp)
    for shift in [*range(lo, hi + 1), *range(-hi, -lo + 1)]:
        shifted = np.roll(rec, shift, axis=0)
        slopes.append(banks.step_fit(d_bsp=d_bsp, d_eng=real_step(values=shifted)).slope)
    return np.asarray(slopes)


def synthetic_stretches(*, bank: banks.Bank, length: int, edges: np.ndarray) -> list[banks.StepFit]:
    """Fit the size-matched BSP regression on engine-only segments cut into stretches of `length`."""
    fits = []
    for seg in bank.segments:
        if seg.world != "engine":
            continue
        eng = np.where(seg.cal_index[:, None] >= 0, seg.engine, np.nan).reshape(-1)
        bsp = seg.bsp.reshape(-1)
        for a in range(0, len(eng) - length + 1, length):
            e, b = eng[a : a + length], bsp[a : a + length]
            d_e = np.full_like(e, np.nan)
            d_e[1:] = e[1:] - e[:-1]
            d_b = np.full_like(b, np.nan)
            d_b[1:] = b[1:] - b[:-1]
            if np.isfinite(d_e).sum() < 100:
                continue
            fits.append(
                banks.step_fit(
                    d_bsp=d_b, d_eng=d_e, weights=banks.size_matched_weights(d_eng=d_e, edges=edges)
                )
            )
    return fits


# --------------------------------------------------------------------------- fits and predictions


def fit_dir() -> Path:
    """Return the folder of fit records."""
    return paths.data_root() / "fits"


def job_name(*, arm: str, arch: str, seed: int) -> str:
    """Return the file stem of a fit."""
    return f"{arm}_{arch}_s{seed}"


def real_sets(*, ctx: Context) -> dict[str, list[sim.Segment]]:
    """Return the real-day pseudo-segments that every fit predicts: R1, K1, K2, and the eras.

    Args:
        ctx: The context.

    Returns:
        A map from set name ("r1", "k1", "k2", "eras") to its segments. K1 feeds D8's reading with
        the metered record subtracted and folded; K2 feeds the reading unchanged.
    """
    cal = ctx.table.calendar
    reading = cal.reading
    test_dates = ctx.cal.dates[ctx.day_split == "test"]
    r1_use = np.isin(cal.dates, test_dates)
    rec = data.record_by_day(record=data.metered_record(), days=cal.dates)
    k_days = np.isfinite(rec).any(axis=1)
    folded, x = real.folded_record_days(reading=reading, record=np.nan_to_num(rec))
    era_use = np.isin(ctx.table.era, [data.ERA_NET, data.ERA_SOLAR, data.ERA_FADE])
    leak.assert_record_not_base(record_days=cal.dates[k_days], base_dates=ctx.cal.dates)
    return {
        "r1": real.real_segments(cal=cal, use=r1_use, reading=reading, truth=reading),
        "k1": real.real_segments(cal=cal, use=k_days, reading=folded, truth=x),
        "k2": real.real_segments(cal=cal, use=k_days, reading=reading, truth=reading),
        "eras": real.real_segments(cal=cal, use=era_use, reading=reading, truth=reading),
    }


def run_fit(
    *, ctx: Context, arm: str, arch: str, seed: int, device: str, log: Log,
    max_epochs: int = core.MAX_EPOCHS,
) -> dict:  # fmt: skip
    """Fit one network, save it, and store its predictions on every bank and real set.

    A fit whose record already exists is skipped, so an interrupted run resumes.

    Args:
        ctx: The context.
        arm: The arm.
        arch: The architecture.
        seed: The seed.
        device: "cuda" or "cpu".
        log: Receives progress lines.
        max_epochs: The cap on epochs.

    Returns:
        The fit record.
    """
    import e00729_train as train

    name = job_name(arm=arm, arch=arch, seed=seed)
    record_path = fit_dir() / f"{name}.json"
    if record_path.exists():
        return read_json(path=record_path)
    train.limit_gpu_memory()
    spec = arms.ARM_SPECS[arm]
    holidays = ctx.cal.holidays
    train_bank = banks.load_bank(path=bank_path(kind="train", seed=seed))
    val_bank = banks.load_bank(path=bank_path(kind="validation"))
    prep_train = arms.prepare_segments(
        segments=train_bank.segments, spec=spec, holidays=holidays, stats=ctx.stats
    )
    prep_val = arms.prepare_segments(
        segments=val_bank.segments, spec=spec, holidays=holidays, stats=ctx.stats
    )
    net, result = train.fit(
        spec=spec, arch=arch, seed=seed, train=prep_train, val=prep_val, device=device,
        max_epochs=max_epochs, out_path=paths.data_root() / "networks" / f"{name}.pt", log=log,
    )  # fmt: skip
    folder = paths.data_root() / "predictions" / name
    for kind in ("validation", "test", "shifted"):
        bank = val_bank if kind == "validation" else banks.load_bank(path=bank_path(kind=kind))
        prep = (
            prep_val
            if kind == "validation"
            else arms.prepare_segments(
                segments=bank.segments, spec=spec, holidays=holidays, stats=ctx.stats
            )
        )
        train.save_predictions(
            arrays=train.predictions(net=net, prepared=prep, spec=spec, device=device),
            path=folder / f"{kind}.npz",
        )
    for set_name, segments in real_sets(ctx=ctx).items():
        prep = arms.prepare_segments(
            segments=segments, spec=spec, holidays=holidays, stats=ctx.stats
        )
        train.save_predictions(
            arrays=train.predictions(net=net, prepared=prep, spec=spec, device=device),
            path=folder / f"real_{set_name}.npz",
        )
    write_json(path=record_path, payload=result.as_dict())
    return result.as_dict()


def stage_choose_architecture() -> dict:
    """Pick the architecture from arm D's validation scores (seeds 0 to 2) and write it once.

    Returns:
        The selection record, including the weights and the mean scores of the three architectures.
    """
    import e00729_nets as nets

    scores, weights = {}, {}
    for arch in core.ARCHITECTURES:
        recs = [
            read_json(path=fit_dir() / f"{job_name(arm='D', arch=arch, seed=s)}.json")
            for s in core.ARCH_SEEDS
        ]
        scores[arch] = float(np.mean([r["best_val_crps"] for r in recs]))
        weights[arch] = int(recs[0]["n_weights"])
    del nets
    chosen = core.choose_architecture(scores=scores, weights=weights)
    record = {"chosen": chosen, "scores": scores, "weights": weights, "chosen_on": "validation"}
    path = paths.data_root() / "selection.json"
    if not path.exists():
        write_json(path=path, payload=record)
    return read_json(path=path)


def stage_margin() -> dict:
    """Write `margin.json` once from arm D's validation signed CRPS over seeds 0 to 4."""
    chosen = read_json(path=paths.data_root() / "selection.json")["chosen"]
    vals = [
        read_json(path=fit_dir() / f"{job_name(arm='D', arch=chosen, seed=s)}.json")[
            "best_val_crps"
        ]
        for s in core.MARGIN_SEEDS
    ]
    path = paths.data_root() / "margin.json"
    if path.exists():
        return read_json(path=path)
    return core.write_margin_file(path=path, val_crps_by_seed=vals)


# --------------------------------------------------------------------------- scoring


def load_predictions(*, arm: str, arch: str, seed: int, name: str) -> dict[str, np.ndarray]:
    """Read one stored prediction file."""
    path = (
        paths.data_root() / "predictions" / job_name(arm=arm, arch=arch, seed=seed) / f"{name}.npz"
    )
    with np.load(path) as f:
        return {k: f[k].astype(np.float64) for k in f.files}


def metrics_of(*, pred: dict[str, np.ndarray], x: np.ndarray) -> score.RowMetrics:
    """Return the per-row metrics of one stored prediction."""
    if "q" in pred:
        return score.quantile_metrics(q=pred["q"], x=x)
    lobes = score.dist.Lobes(pred["p"], pred["mu_i"], pred["s_i"], pred["mu_r"], pred["s_r"])
    return score.lobe_metrics(lobes=lobes, x=x)


def arm_metrics(
    *, arm: str, arch: str, name: str, x: np.ndarray, seeds: Sequence[int]
) -> score.RowMetrics:
    """Return the seed-averaged metrics of an arm on one stored prediction set."""
    per_seed = []
    for s in seeds:
        pred = load_predictions(arm=arm, arch=arch, seed=s, name=name)
        if len(next(iter(pred.values()))) != len(x):
            msg = "a stored prediction does not match the scored rows"
            raise leak.CorrectnessFailureError(msg)
        per_seed.append(metrics_of(pred=pred, x=x))
    return score.average_metrics(per_seed=per_seed)


def baseline_metrics(
    *, rows: score.Rows, climatology: base.Climatology, regression: base.Regression, pi: float
) -> dict[str, score.RowMetrics]:
    """Return the metrics of P0, P0f, B1, and B2 on a set of rows with prior `pi`."""
    f = rows.features
    b1 = base.sign_prior_lobes(f=f, prob_below=climatology.prob_below(r=f.r, f=f), pi=pi)
    b2 = base.sign_prior_lobes(f=f, prob_below=regression.prob_below(r=f.r, f=f), pi=pi)
    lobes = {"P0": base.p0(f=f), "P0f": base.p0f(f=f), "B1": b1, "B2": b2}
    return {k: score.lobe_metrics(lobes=v, x=rows.x) for k, v in lobes.items()}


def fit_baselines(*, ctx: Context) -> tuple[base.Climatology, base.Regression]:
    """Fit the climatology and the regression on the training base days."""
    train = ctx.day_split == "training"
    cal = ctx.cal
    demand = cal.reading[train]
    n = demand.shape[0]
    hh = np.tile(np.arange(48), (n, 1))
    offsets = np.array(
        [sim.utc_offset_hours(day=d.astype("datetime64[D]").astype(date)) for d in cal.dates[train]]
    )
    local_hh = (hh + np.round(2 * offsets)[:, None].astype(int)) % 48
    weekday = (cal.dates[train].astype("datetime64[D]").astype(int) + 3) % 7
    month = np.array([int(str(d)[5:7]) for d in cal.dates[train]])
    holiday = np.isin(cal.dates[train], cal.holidays)

    def tile(v: np.ndarray) -> np.ndarray:
        return np.repeat(v, 48)

    f = base.Features(
        r=demand.reshape(-1), local_hh=local_hh.reshape(-1), local_month=tile(month),
        weekend=tile(weekday >= 5), weekday=tile(weekday), holiday=tile(holiday),
        t2m=cal.t2m[train].reshape(-1),
    )  # fmt: skip
    return base.Climatology.fit(demand=f.r, f=f), base.Regression.fit(demand=f.r, f=f)


def contrast_plan() -> dict[str, list[tuple[str, str, str]]]:
    """Return, for each contrast, the readings to take as (label, row subset, month set)."""
    plan: dict[str, list[tuple[str, str, str]]] = {
        c: [("all", "all", "all")] for c in core.CONTRASTS
    }
    plan["C3"] += [("source_S1", "source_S1", "all"), ("source_S2", "source_S2", "all")]
    for c in core.BSP_CONTRASTS:
        plan[c] = [("bsp_all", "all", "bsp")] + [
            (f"bsp_{w}", f"world_{w}", "bsp") for w in ("engine", "solar", "both")
        ]
    return plan


def stage_score(*, ctx: Context, n_resamples: int = core.N_RESAMPLES, log: Log) -> dict:
    """Score every arm and baseline on the test, validation, and shifted banks and read the verdict.

    The margin file must exist before this stage reads any test contrast.

    Args:
        ctx: The context.
        n_resamples: The bootstrap resamples per reading.
        log: Receives progress lines.

    Returns:
        The verdict record, also written to `verdict.json`.
    """
    root = paths.data_root()
    margin = core.read_margin_file(path=root / "margin.json")
    arch = read_json(path=root / "selection.json")["chosen"]
    clim, reg = fit_baselines(ctx=ctx)
    seeds = core.ARCH_SEEDS
    rows_val = score.rows_from_segments(
        segments=banks.load_bank(path=bank_path(kind="validation")).segments,
        holidays=ctx.cal.holidays,
    )
    val_b = baseline_metrics(
        rows=rows_val, climatology=clim, regression=reg, pi=core.TEST_MIX_INJECTED
    )
    bstar = core.choose_baseline(
        b1=float(val_b["B1"].crps.mean()), b2=float(val_b["B2"].crps.mean())
    )
    write_json(
        path=root / "scores" / "bstar.json", payload={"bstar": bstar, "chosen_on": "validation"}
    )
    rows = score.rows_from_segments(
        segments=banks.load_bank(path=bank_path(kind="test")).segments, holidays=ctx.cal.holidays
    )
    base_m = baseline_metrics(
        rows=rows, climatology=clim, regression=reg, pi=core.TEST_MIX_INJECTED
    )
    metrics: dict[str, score.RowMetrics] = {**base_m, "Bstar": base_m[bstar]}
    for arm in core.ARMS:
        metrics[arm] = arm_metrics(arm=arm, arch=arch, name="test", x=rows.x, seeds=seeds)
    masks = score.subset_masks(rows=rows)
    bsp = score.bsp_months(rows=rows)
    readings: dict[str, dict[str, dict]] = {}
    for cname, (a, b) in core.CONTRASTS.items():
        metric = "pinball" if cname in core.PINBALL_CONTRASTS else "crps"
        readings[cname] = {}
        for label, subset, months in contrast_plan()[cname]:
            r = score.contrast(
                rows=rows, a=getattr(metrics[a], metric), b=getattr(metrics[b], metric),
                keep=masks[subset], months=bsp if months == "bsp" else None, n_resamples=n_resamples,
            )  # fmt: skip
            readings[cname][label] = {
                **r.reading.__dict__,
                "kish_blocks": r.kish_blocks,
                "kish_segments": r.kish_segments,
                "widening": r.widening,
                "n_rows": r.n_rows,
            }
        log(f"contrast {cname} read")
    d = metrics["D"]
    calib = score.sign_calibration(rows=rows, p_neg=d.p_neg)
    lobe = score.lobe_coverage_bar(covers=d.covers)
    h1 = score.harm_check(
        d=d, p0f_metrics=metrics["P0f"], p0_metrics=metrics["P0"], keep=masks["import"], real=False
    )
    tables = {name: score.subset_means(metrics=m, masks=masks) for name, m in metrics.items()}
    coverage = {
        name: {"q05": float(m.q05_cover.mean()), "q95": float(m.q95_cover.mean())}
        for name, m in metrics.items()
    }
    write_json(
        path=root / "scores" / "tables.json",
        payload={"tables": tables, "coverage": coverage, "lobe_width_mean": float(d.width.mean())},
    )
    shifted = shifted_read_out(ctx=ctx, arch=arch, clim=clim, reg=reg, bstar=bstar)
    gate = read_json(path=root / "scores" / "gate.json")
    reach = score.reachability(rows=rows)
    real_out = stage_real(ctx=ctx, arch=arch, metrics=metrics, clim=clim, log=log)
    c1 = readings["C1"]["all"]
    verdict_in = core.VerdictInputs(
        stopped=not (gate["pass"] and reach["pass"]),
        invalid=not read_json(path=root / "banks" / "summary.json")["leakage_pass"],
        failed=any(fit_dir().glob("*.failed.json")),
        h1_pass=bool(h1["pass"]), h2_pass=bool(real_out["h2"]["pass"]),
        c1=core.Reading(c1["point"], c1["lower"], c1["upper"]), margin=margin,
        calibration_pass=bool(calib["pass"]), lobe_pass=bool(lobe["pass"]),
    )  # fmt: skip
    row = core.verdict_row(v=verdict_in)
    record = {
        "row": row, "verdict": core.VERDICT_WORDS[row], "margin": margin, "bstar": bstar,
        "architecture": arch, "c1": c1, "calibration": calib, "lobe": lobe, "h1": h1,
        "reachability": reach, "contrasts": readings, "shifted": shifted, "real": real_out,
    }  # fmt: skip
    write_json(path=root / "verdict.json", payload=record)
    return record


def shifted_read_out(
    *, ctx: Context, arch: str, clim: base.Climatology, reg: base.Regression, bstar: str
) -> dict:
    """The shifted-mix read-out: arm D and B* on the 50%-injected bank (enters no verdict row)."""
    rows = score.rows_from_segments(
        segments=banks.load_bank(path=bank_path(kind="shifted")).segments, holidays=ctx.cal.holidays
    )
    pi = 1.0 - core.SHIFTED_NONE_SHARE
    bm = baseline_metrics(rows=rows, climatology=clim, regression=reg, pi=pi)[bstar]
    d = arm_metrics(arm="D", arch=arch, name="shifted", x=rows.x, seeds=core.ARCH_SEEDS)
    out = {}
    keep = np.abs(rows.x) >= core.DEAD_BAND_POINTS
    for name, m in (("D", d), ("Bstar", bm)):
        rel = score.dist.reliability_term(prob=m.p_neg[keep], observed=rows.x[keep] < 0)
        out[name] = {
            "crps": float(m.crps.mean()),
            "reliability": rel,
            "share_above_half": score.false_sign_share(p_neg=m.p_neg),
        }
    return out


def stage_real(
    *,
    ctx: Context,
    arch: str,
    metrics: dict[str, score.RowMetrics],
    clim: base.Climatology,
    log: Log,
) -> dict:
    """Run R1 (H2), K1, K2, R2, and R3 from the stored real-set predictions (aggregates only)."""
    sets = real_sets(ctx=ctx)
    cal = ctx.table.calendar
    rows_r1 = score.rows_from_segments(segments=sets["r1"], holidays=cal.holidays)
    f = rows_r1.features
    p0f_m = score.lobe_metrics(lobes=base.p0f(f=f), x=rows_r1.x)
    p0_m = score.lobe_metrics(lobes=base.p0(f=f), x=rows_r1.x)
    d_r1 = arm_metrics(arm="D", arch=arch, name="real_r1", x=rows_r1.x, seeds=core.ARCH_SEEDS)
    h2 = real.r1_false_alarms(d=d_r1, p0f_metrics=p0f_m, p0_metrics=p0_m)
    rows_k1 = score.rows_from_segments(segments=sets["k1"], holidays=cal.holidays)
    rows_k2 = score.rows_from_segments(segments=sets["k2"], holidays=cal.holidays)
    rec_days = data.record_by_day(record=data.metered_record(), days=cal.dates)
    rec_rows = record_values_for_rows(segments=sets["k1"], record=rec_days, cal=cal)
    aux = [load_predictions(arm="G2", arch=arch, seed=s, name="real_k1") for s in core.ARCH_SEEDS]
    head = real.gaussian_mixture_median(
        mu=np.stack([a["aux_mu"] for a in aux]), sigma=np.stack([a["aux_s"] for a in aux])
    )
    running = rec_rows > core.FLOOR_POINTS
    p_any = base.m2_p_any(reading_points=rows_k1.r)
    k1 = real.k1_engine_head(
        head_median=head, record=rec_rows, running=running, m2_flag=p_any > 0.5
    )
    d_k1 = arm_metrics(arm="D", arch=arch, name="real_k1", x=rows_k1.x, seeds=core.ARCH_SEEDS)
    b1_k1 = score.lobe_metrics(
        lobes=real.b_star_on_rows(rows=rows_k1, climatology=clim, pi=core.TEST_MIX_INJECTED),
        x=rows_k1.x,
    )
    k1["crps_d"] = float(d_k1.crps.mean())
    k1["crps_bstar"] = float(b1_k1.crps.mean())
    aux2 = [load_predictions(arm="G2", arch=arch, seed=s, name="real_k2") for s in core.ARCH_SEEDS]
    head2 = real.gaussian_mixture_median(
        mu=np.stack([a["aux_mu"] for a in aux2]), sigma=np.stack([a["aux_s"] for a in aux2])
    )
    d_k2 = arm_metrics(arm="D", arch=arch, name="real_k2", x=rows_k2.x, seeds=core.ARCH_SEEDS)
    k2 = real.k2_placebo(p_neg=d_k2.p_neg, head_median=head2)
    rows_e = score.rows_from_segments(segments=sets["eras"], holidays=cal.holidays)
    d_e = arm_metrics(arm="D", arch=arch, name="real_eras", x=rows_e.x, seeds=core.ARCH_SEEDS)
    eras = real.era_groups(dates=rows_e.date, day_dates=cal.dates, day_eras=list(ctx.table.era))
    out = {
        "h2": h2, "k1": k1, "k2": k2,
        "r2": {**real.r2_by_group(median=d_e.median, reading=rows_e.r, groups=eras), "k2_days": real.refold_share(median=d_k2.median, reading=rows_k2.r)},
        "r3": {
            "share_above_half_by_era": real.r3_sign_by_group(p_neg=d_e.p_neg, groups=eras),
            "share_above_half_by_era_and_hour": real.r3_sign_by_group(p_neg=d_e.p_neg, groups=np.array([f"{e}_{h // 2}" for e, h in zip(eras, rows_e.hh, strict=True)])),
        },
    }  # fmt: skip
    log("real checks done")
    return out


def record_values_for_rows(
    *, segments: Sequence[sim.Segment], record: np.ndarray, cal: sim.Calendar
) -> np.ndarray:
    """Return the record's value for each valid half-hour of the K1 segments, in `Rows` order."""
    out = []
    for seg in segments:
        for d in np.flatnonzero(seg.cal_index >= 0):
            ok = seg.valid[d] & np.isfinite(seg.x[d])
            out.append(np.nan_to_num(record[seg.cal_index[d]])[ok])
    return np.concatenate(out)

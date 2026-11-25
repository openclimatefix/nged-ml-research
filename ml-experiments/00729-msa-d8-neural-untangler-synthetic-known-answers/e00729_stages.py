# ruff: noqa: E501
"""Experiment 729: the stages of the run (banks, gate, fits, predictions, scores, verdict).

Every stage reads and writes files under this experiment's private data folder, and writes
aggregates only to the `scores/` JSON files. No stage prints a month, a date, an engine size, a
series identifier, or a real value; log lines carry counts and scores. Files are written atomically.
The stages that need torch import `e00729_train` inside the function.
"""

import json
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

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
SPLIT_ROW = re.compile(r"^\|\s*\d+\s*\|\s*\w+\s*\|\s*(\d{4}-\d{2})\s*\|\s*(\w+)\s*\|")
"""A row of `splits.md`. The position column holds `None` for a lockdown month."""
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
        closure_status: `loaded`, `missing`, or `unreadable` (the switching logs).
        closure_on_base: Whether any closure day fell on a base day, so that the log changed the base.
    """

    table: data.DayTable
    cal: sim.Calendar
    month_split: dict[str, str]
    day_split: np.ndarray
    use: dict[str, dict[str, np.ndarray]]
    stats: arms.InputStats
    cut_day: np.datetime64
    closure_status: str = "missing"
    closure_on_base: bool = False

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


def build_context() -> Context:
    """Load the days, derive the split from the rule, and check it against the private file.

    The switching logs are loaded when they can be, and the days within 24 hours of a logged closure
    at D8 are dropped from the base.

    Returns:
        The context.

    Raises:
        CorrectnessFailureError: When the derived split differs from `PRIVATE/splits.md`.
    """
    closure_days, closure_status = data.load_closure_days()
    table = data.load_day_table(closure_days=closure_days)
    cal = data.select_days(cal=table.calendar, mask=table.is_base)
    months, n_days, bsp_days = data.month_table(cal=cal)
    month_split = core.month_splits(months=months, n_days=n_days, bsp_days=bsp_days)
    private = paths.data_root() / "PRIVATE" / "splits.md"
    if not private.exists():
        msg = "PRIVATE/splits.md is missing, so the derived split cannot be checked (verdict row 1)"
        raise leak.CorrectnessFailureError(msg)
    if parse_private_splits(path=private) != month_split:
        msg = (
            "the split derived from the rule, with the closure days dropped, differs from "
            "PRIVATE/splits.md (verdict row 1): the design lead rules which calendar is registered"
        )
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
    closed = np.array(sorted(closure_days), dtype="datetime64[D]")
    on_base = bool(np.isin(table.calendar.dates[table.is_base], closed).any())
    return Context(table, cal, month_split, day_split, use, stats, cut_day, closure_status, on_base)  # ty: ignore[invalid-argument-type]


def bank_path(*, kind: str, seed: int | None = None) -> Path:
    """Return the file of a bank."""
    name = f"{kind}_s{seed}.pkl" if seed is not None else f"{kind}.pkl"
    return paths.data_root() / "banks" / name


def stage_banks(*, ctx: Context, train_seeds: Sequence[int], log: Log) -> dict:
    """Build the validation, test, shifted-mix, and training banks, and run the leakage checks.

    The bank-level checks run on every segment of every evaluation bank, because they are cheap.

    Args:
        ctx: The context.
        train_seeds: The training seeds to build (0 to 4).
        log: Receives progress lines.

    Returns:
        The summary written to `banks/summary.json` (hashes, counts, reachability).
    """
    summary_path = paths.data_root() / "banks" / "summary.json"
    before_path = summary_path.with_name("summary_before.json")
    earlier = read_json(path=summary_path)["digests"] if summary_path.exists() else None
    if earlier is not None:
        shutil.copy2(summary_path, before_path)
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
    evaluation = [built[k] for k in ("validation", "test", "shifted")]
    leak.assert_seed_spaces(train_seeds=train_seeds, evaluation=evaluation)
    leak.assert_banks_independent(evaluation=evaluation)
    every = [s for b in built.values() for s in b.segments]
    leak.assert_parameter_ranges(segments=every)
    folder = Path(__file__).parent
    leak.assert_simulator_source_clean(paths=[folder / n for n in SIMULATOR_SOURCES])
    for bank in evaluation:
        for seg in bank.segments:
            leak.assert_bsp_construction(seg=seg, cal=ctx.cal)
            leak.assert_s1_same_day(seg=seg, cal=ctx.cal)
    leak.assert_no_calendar_memorisation(specs=arms.ARM_SPECS)
    min_unit = min(
        float(s.params["min_unit_points"]) for s in every if "min_unit_points" in s.params
    )
    log(f"smallest engine unit after the size cap: {min_unit:.1f} points")
    rows = score.rows_from_segments(segments=built["test"].segments, holidays=ctx.cal.holidays)
    digests = {k: b.digest for k, b in built.items()}
    unchanged = None if earlier is None else all(earlier.get(k) == v for k, v in digests.items())
    log(f"digests unchanged from the earlier summary: {unchanged}")
    summary = {
        "digests": digests,
        "digests_unchanged": unchanged,
        "reachability": score.reachability(rows=rows),
        "n_segments": {k: len(b.segments) for k, b in built.items()},
        "injected_share": {b.kind: banks.injected_share(bank=b) for b in evaluation},
        "min_unit_points_after_cap": min_unit,
        "dropped_half_hours": {
            s: int(ctx.table.dropped_half_hours[ctx.table.is_base][ctx.day_split == s].sum())
            for s in ("training", "validation", "test")
        },
        "dropped_days_by_guard": int((ctx.day_split == "dropped").sum()),
        "closure_log_status": ctx.closure_status,
        "closure_day_on_base_day": ctx.closure_on_base,
        "leakage_pass": True,
    }
    write_json(path=summary_path, payload=summary)
    return summary


def stage_gate(*, ctx: Context, log: Log) -> dict:
    """Run the realism gate on folded aggregates and the BSP step relation, and write `gate.json`.

    The record spans more days than one synthetic segment (one month), so the BSP half of the gate
    cuts the real record and every engine-only synthetic segment into stretches of one common
    length, the longest that a segment can hold. The real standard error and residual standard
    deviation are the medians over the real stretches; the real slope's interval comes from the
    whole record. A gate that finds no synthetic stretch is `void`, which does not pass.

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
    rec = data.record_by_day(record=record, days=cal_all.dates)
    days = np.flatnonzero(np.isfinite(rec).any(axis=1))
    d_eng = real_step(values=rec[days], days=cal_all.dates[days])
    d_bsp = real_step(values=cal_all.bsp[days], days=cal_all.dates[days])
    fit = banks.step_fit(d_bsp=d_bsp, d_eng=d_eng)
    ci = (fit.slope - 1.96 * fit.se, fit.slope + 1.96 * fit.se)
    placebo = placebo_slopes(rec=rec[days], bsp=cal_all.bsp[days], days=cal_all.dates[days])
    edges = banks.real_bin_edges(d_eng=d_eng)
    shares = banks.real_bin_shares(d_eng=d_eng, edges=edges)
    synthetic, length = synthetic_stretches(
        bank=test, record_length=len(d_eng), edges=edges, shares=shares
    )
    real_fits = banks.chunk_fits(d_bsp=d_bsp, d_eng=d_eng, length=length)
    real_ref = banks.StepFit(
        fit.slope,
        float(np.nanmedian([f.se for f in real_fits])) if real_fits else float("nan"),
        float(np.nanmedian([f.resid_sd for f in real_fits])) if real_fits else float("nan"),
        int(np.median([f.n for f in real_fits])) if real_fits else 1,
    )
    gate = banks.bsp_gate(real=real_ref, real_ci=ci, synthetic=synthetic)
    out["bsp"] = {
        **gate, "real_slope_z_vs_placebo": float((fit.slope - np.mean(placebo)) / np.std(placebo)),
        "n_synthetic_stretches": gate["n_finite_fits"], "n_synthetic_stretches_all": len(synthetic),
        "n_real_stretches": len(real_fits),
        "stretch_half_hours": length,
    }  # fmt: skip
    passes.append(bool(gate["pass"]))
    out["void"] = bool(gate["void"])
    out["pass"] = all(passes)
    log(f"gate BSP steps: pass {gate['pass']}, void {gate['void']}")
    write_json(path=paths.data_root() / "scores" / "gate.json", payload=out)
    return out


def real_step(*, values: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Return the half-hour steps of the days' values, flattened.

    A step is NaN where either half-hour is missing and at the first half-hour of a day whose
    previous calendar day is not in `days`, so that no step spans a gap.

    Args:
        values: The values, `(n_days, 48)`.
        days: The calendar day of each row, as `datetime64[D]`, sorted.

    Returns:
        The steps, `(n_days * 48,)`.
    """
    flat = values.reshape(-1)
    out = np.full_like(flat, np.nan)
    out[1:] = flat[1:] - flat[:-1]
    first = np.ones(len(days), dtype=bool)
    first[1:] = (days[1:] - days[:-1]).astype(int) != 1
    out[np.flatnonzero(first) * 48] = np.nan
    return out


def placebo_slopes(*, rec: np.ndarray, bsp: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Return the slopes with the record shifted by 3 to 20 whole days either way (circular)."""
    lo, hi = core.GATE_PLACEBO_SHIFTS
    slopes = []
    d_bsp = real_step(values=bsp, days=days)
    for shift in [*range(lo, hi + 1), *range(-hi, -lo + 1)]:
        shifted = np.roll(rec, shift, axis=0)
        slopes.append(banks.step_fit(d_bsp=d_bsp, d_eng=real_step(values=shifted, days=days)).slope)
    return np.asarray(slopes)


def synthetic_stretches(
    *, bank: banks.Bank, record_length: int, edges: np.ndarray, shares: np.ndarray
) -> tuple[list[banks.StepFit], int]:
    """Fit the size-matched BSP regression on the engine-only segments of a bank.

    Each segment is one month, so a stretch is as long as the record or as the shortest engine-only
    segment, whichever is smaller; the real record is cut to the same length by the caller.

    Args:
        bank: The test bank.
        record_length: The real record's length in half-hours.
        edges: The real record's size-bin edges.
        shares: The real record's share of steps per size bin.

    Returns:
        The fits (one per stretch) and the common stretch length in half-hours.
    """
    steps = []
    for seg in bank.segments:
        if seg.world != "engine":
            continue
        eng = np.where(seg.cal_index[:, None] >= 0, seg.engine, np.nan)
        steps.append((step_flat(values=eng), step_flat(values=seg.bsp)))
    if not steps:
        return [], record_length
    length = min(record_length, *(len(e) for e, _ in steps))
    fits = []
    for d_e, d_b in steps:
        fits.extend(
            banks.chunk_fits(d_bsp=d_b, d_eng=d_e, length=length, edges=edges, shares=shares)
        )
    return fits, length


def step_flat(*, values: np.ndarray) -> np.ndarray:
    """Return the half-hour steps of a dense `(days, 48)` array (NaN where a half-hour is missing)."""
    flat = values.reshape(-1)
    out = np.full_like(flat, np.nan)
    out[1:] = flat[1:] - flat[:-1]
    return out


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
    folded, x = real.folded_record_days(reading=reading, record=rec)
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
    digests = {
        "validation": val_bank.digest,
        "test": banks.load_bank(path=bank_path(kind="test")).digest,
        "shifted": banks.load_bank(path=bank_path(kind="shifted")).digest,
    }
    payload = {**result.as_dict(), "digests": digests}
    write_json(path=record_path, payload=payload)
    (fit_dir() / f"{name}.failed.json").unlink(missing_ok=True)
    return payload


def run_fit_or_record_failure(
    *, ctx: Context, arm: str, arch: str, seed: int, device: str, log: Log
) -> dict | None:
    """Run one fit; when it raises, write `fits/<name>.failed.json` and return None (no retry).

    A failed fit makes the verdict row 2 (`failed`) at the scoring stage. A correctness error is not
    caught here, because it must stop the line as row 1.

    Args:
        ctx: The context.
        arm: The arm.
        arch: The architecture.
        seed: The seed.
        device: "cuda" or "cpu".
        log: Receives progress lines.

    Returns:
        The fit record, or None when the fit failed.
    """
    try:
        return run_fit(ctx=ctx, arm=arm, arch=arch, seed=seed, device=device, log=log)
    except leak.CorrectnessFailureError:
        raise
    except Exception as error:  # noqa: BLE001
        name = job_name(arm=arm, arch=arch, seed=seed)
        write_json(path=fit_dir() / f"{name}.failed.json", payload={"error": type(error).__name__})
        log(f"fit {name} failed with {type(error).__name__}; it is not retried")
        return None


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


AMBIGUITY_ARMS: Final = (*(a for a in core.ARMS if a != "Dq"), "Bstar")
"""The arms with a probability output, and B*: the ambiguity read-out covers them (not the quantile head)."""


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


def segment_p_any(*, segments: Sequence[sim.Segment]) -> np.ndarray:
    """Return experiment 694's M2 probability that an engine unit runs, in `Rows` order.

    M2 reads each segment as one continuous series of folded readings (a missing half-hour is filled
    by linear interpolation, because the tracker's 48-step lags need an unbroken series), and the
    probabilities are then picked out at the valid half-hours that `rows_from_segments` scores.

    Args:
        segments: The segments (synthetic, or the real K1 pseudo-segments).

    Returns:
        P(an engine unit runs) at each scored half-hour.
    """
    out = []
    for seg in segments:
        base_days = np.flatnonzero(seg.cal_index >= 0)
        y = seg.r[base_days[0] : base_days[-1] + 1].reshape(-1)
        ok = np.isfinite(y)
        idx = np.arange(len(y))
        filled = np.interp(idx, idx[ok], y[ok]) if ok.any() else np.zeros(len(y))
        p_any = base.m2_p_any(reading_points=filled).reshape(-1, 48)
        for d in base_days:
            keep = seg.valid[d] & np.isfinite(seg.x[d])
            out.append(p_any[d - base_days[0]][keep])
    return np.concatenate(out)


def needed_job_names() -> set[str]:
    """Return the fits whose predictions a counted reading reads: every counted arm at seeds 0 to 2.

    The report-only arm D-sign is left out, so its failure cannot make verdict row 2. The margin
    also reads arm D at the other seeds. The architecture's choice has been made by then,
    so the architecture is read from `selection.json`; before it exists, no fit is needed.
    """
    path = paths.data_root() / "selection.json"
    if not path.exists():
        return set()
    arch = read_json(path=path)["chosen"]
    counted = [a for a in core.ARMS if a not in core.REPORT_ONLY_ARMS]
    names = {job_name(arm=a, arch=arch, seed=s) for a in counted for s in core.ARCH_SEEDS}
    return names | {job_name(arm="D", arch=arch, seed=s) for s in core.MARGIN_SEEDS}


def arm_fitted(*, arm: str, arch: str, seeds: Sequence[int]) -> bool:
    """Return whether every seed of an arm has a fit record (and so stored predictions)."""
    return all((fit_dir() / f"{job_name(arm=arm, arch=arch, seed=s)}.json").exists() for s in seeds)


def void_reading() -> dict[str, float | bool]:
    """Return the record of a contrast that could not be read: `void` is true and no number is set."""
    nan = float("nan")
    return {
        "void": True, "point": nan, "lower": nan, "upper": nan, "kish_blocks": nan,
        "kish_segments": nan, "widening": nan, "n_rows": 0,
    }  # fmt: skip


def needed_fits_failed() -> bool:
    """Return whether a needed fit has a failure record and no fit record.

    A failure record is stale once a sanctioned rerun of the fit has written its fit record, and a
    failure of a fit that no counted reading needs (a screen fit of another architecture) is not row 2.
    """
    needed = needed_job_names()
    for path in fit_dir().glob("*.failed.json"):
        name = path.name.removesuffix(".failed.json")
        if name in needed and not (fit_dir() / f"{name}.json").exists():
            return True
    return False


def terminal_verdict(
    *, root: Path, stopped: bool = False, invalid: bool = False, failed: bool = False
) -> dict:
    """Write and return a verdict record for a line that ended before the counted readings.

    Args:
        root: The experiment's data folder.
        stopped: Row 0 applies.
        invalid: Row 1 applies (a correctness check failed).
        failed: Row 2 applies (a fit raised).

    Returns:
        The record, also written to `verdict.json`.
    """
    nan = float("nan")
    inputs = core.VerdictInputs(
        stopped=stopped, invalid=invalid, failed=failed, h1_pass=False, h2_pass=False,
        c1=core.Reading(nan, nan, nan), margin=nan, calibration_pass=False, lobe_pass=False,
    )  # fmt: skip
    row = core.verdict_row(v=inputs)
    record = {"row": row, "verdict": core.VERDICT_WORDS[row], "terminal": True}
    write_json(path=root / "verdict.json", payload=record)
    return record


def stage_score(*, ctx: Context, n_resamples: int = core.N_RESAMPLES, log: Log) -> dict:
    """Score every arm and baseline, read the verdict, and route a failure to its verdict row.

    A fit that failed makes row 2. A correctness error raised while scoring makes row 1.

    Args:
        ctx: The context.
        n_resamples: The bootstrap resamples per reading.
        log: Receives progress lines.

    Returns:
        The verdict record, also written to `verdict.json`.
    """
    root = paths.data_root()
    if needed_fits_failed():
        log("a fit that a counted reading needs failed: verdict row 2")
        return terminal_verdict(root=root, failed=True)
    try:
        return score_everything(ctx=ctx, n_resamples=n_resamples, log=log)
    except leak.CorrectnessFailureError:
        log("a correctness check failed while scoring: verdict row 1")
        return terminal_verdict(root=root, invalid=True)


def write_bstar_once(*, path: Path, bstar: str) -> None:
    """Write the choice of B* once; a later run that would choose another baseline is an error."""
    if path.exists():
        if read_json(path=path)["bstar"] != bstar:
            msg = "B* chosen on the validation months differs from the stored choice"
            raise leak.CorrectnessFailureError(msg)
        return
    write_json(path=path, payload={"bstar": bstar, "chosen_on": "validation"})


TEST_MARKER: Final = "first_test_score.marker"
"""A file that the first scoring run creates and never rewrites; its time is the first test score."""


def choice_events(*, root: Path) -> list[str]:
    """Return the names of the choices and of the first test score in order of file time.

    The architecture, B*, and margin files are compared with the marker of the first test score,
    which is created now when no earlier run made it. A choice file rewritten after the first test
    score therefore sorts after it, and `assert_choices_before_test` fails.

    Args:
        root: The experiment's data folder.

    Returns:
        The event names, earliest first.
    """
    marker = root / "scores" / TEST_MARKER
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
    stamps = {
        "architecture": (root / "selection.json").stat().st_mtime_ns,
        "bstar": (root / "scores" / "bstar.json").stat().st_mtime_ns,
        "margin": (root / "margin.json").stat().st_mtime_ns,
        "test_score": marker.stat().st_mtime_ns,
    }
    return sorted(stamps, key=lambda k: (stamps[k], k == "test_score"))


def fit_digests(*, arch: str, seeds: Sequence[int]) -> dict[str, dict[str, str]]:
    """Return the bank digests that each stored fit recorded, and those of the bank files."""
    out = {
        f"{arm}_s{s}": read_json(path=fit_dir() / f"{job_name(arm=arm, arch=arch, seed=s)}.json")[
            "digests"
        ]
        for arm in core.ARMS
        for s in seeds
        if (fit_dir() / f"{job_name(arm=arm, arch=arch, seed=s)}.json").exists()
    }
    out["baselines"] = {
        kind: banks.load_bank(path=bank_path(kind=kind)).digest
        for kind in ("validation", "test", "shifted")
    }
    return out


def score_everything(*, ctx: Context, n_resamples: int, log: Log) -> dict:  # noqa: PLR0915
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
    seeds = core.ARCH_SEEDS
    leak.assert_hashes_match(digests_by_arm=fit_digests(arch=arch, seeds=seeds))
    clim, reg = fit_baselines(ctx=ctx)
    val_bank = banks.load_bank(path=bank_path(kind="validation"))
    rows_val = score.rows_from_segments(segments=val_bank.segments, holidays=ctx.cal.holidays)
    val_b = baseline_metrics(
        rows=rows_val, climatology=clim, regression=reg, pi=banks.injected_share(bank=val_bank)
    )
    bstar = core.choose_baseline(
        b1=float(val_b["B1"].crps.mean()), b2=float(val_b["B2"].crps.mean())
    )
    write_bstar_once(path=root / "scores" / "bstar.json", bstar=bstar)
    leak.assert_choices_before_test(events=choice_events(root=root))
    test_bank = banks.load_bank(path=bank_path(kind="test"))
    rows = score.rows_from_segments(segments=test_bank.segments, holidays=ctx.cal.holidays)
    pi_test = banks.injected_share(bank=test_bank)
    base_m = baseline_metrics(rows=rows, climatology=clim, regression=reg, pi=pi_test)
    metrics: dict[str, score.RowMetrics] = {**base_m, "Bstar": base_m[bstar]}
    if paths.sibling_available(prefix="00694-"):
        m2 = base.m2_lobes(
            f=rows.features,
            prob_below=clim.prob_below(r=rows.features.r, f=rows.features),
            p_any=segment_p_any(segments=test_bank.segments),
        )
        metrics["M2"] = score.lobe_metrics(lobes=m2, x=rows.x)
    report_only_missing = {
        a for a in core.REPORT_ONLY_ARMS if not arm_fitted(arm=a, arch=arch, seeds=seeds)
    }
    for arm in core.ARMS:
        if arm not in report_only_missing:
            metrics[arm] = arm_metrics(arm=arm, arch=arch, name="test", x=rows.x, seeds=seeds)
    masks = score.subset_masks(rows=rows)
    bsp = score.bsp_months(rows=rows)
    readings: dict[str, dict[str, dict]] = {}
    for cname, (a, b) in core.CONTRASTS.items():
        metric = "pinball" if cname in core.PINBALL_CONTRASTS else "crps"
        readings[cname] = {}
        if a in report_only_missing or b in report_only_missing:
            readings[cname] = {"all": void_reading()}
            log(f"contrast {cname} is void: a report-only arm has no complete fit")
            continue
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
    ambiguity = {
        name: score.ambiguity_readout(rows=rows, p_neg=m.p_neg)
        for name, m in metrics.items()
        if name in AMBIGUITY_ARMS
    }
    coverage = {
        name: {"q05": float(m.q05_cover.mean()), "q95": float(m.q95_cover.mean())}
        for name, m in metrics.items()
    }
    write_json(
        path=root / "scores" / "tables.json",
        payload={
            "tables": tables, "coverage": coverage, "ambiguity": ambiguity, "lobe_width_mean": float(d.width.mean()),
            "realised_injected_share": pi_test,
        },
    )  # fmt: skip
    shifted = shifted_read_out(ctx=ctx, arch=arch, clim=clim, reg=reg, bstar=bstar)
    gate = read_json(path=root / "scores" / "gate.json")
    reach = score.reachability(rows=rows)
    real_out = stage_real(
        ctx=ctx, arch=arch, metrics=metrics, clim=clim, reg=reg, bstar=bstar, log=log
    )
    c1 = readings["C1"]["all"]
    verdict_in = core.VerdictInputs(
        stopped=not (gate["pass"] and reach["pass"]),
        invalid=not read_json(path=root / "banks" / "summary.json")["leakage_pass"],
        failed=needed_fits_failed(),
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
    bank = banks.load_bank(path=bank_path(kind="shifted"))
    rows = score.rows_from_segments(segments=bank.segments, holidays=ctx.cal.holidays)
    pi = banks.injected_share(bank=bank)
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


def flag_grid(*, rows: score.Rows, flagged: np.ndarray, day_dates: np.ndarray) -> np.ndarray:
    """Place per-row flags on the calendar's `(days, 48)` grid (False where no row exists)."""
    lookup = {int(d): i for i, d in enumerate(day_dates.astype("datetime64[D]").astype(int))}
    grid = np.zeros((len(day_dates), 48), dtype=bool)
    for date_int, hh, flag in zip(rows.date, rows.hh, flagged, strict=True):
        grid[lookup[int(date_int)], int(hh)] = bool(flag)
    return grid


def r3_extras(
    *, rows: score.Rows, p_neg: np.ndarray, cal: sim.Calendar
) -> dict[str, float | dict[str, float]]:
    """R3's two further readings: sign changes beside near-zero half-hours, and the BSP level.

    Args:
        rows: The era rows.
        p_neg: Arm D's P(X < 0) on those rows.
        cal: The calendar.

    Returns:
        The mean over the months' shares of sign changes that sit next to a half-hour below
        1.0 MVA, and the BSP residual level on flagged half-hours against the others.
    """
    shares = []
    for seg in np.unique(rows.seg):
        on = rows.seg == seg
        shares.append(real.near_zero_neighbour_share(p_neg=p_neg[on], reading=rows.r[on]))
    finite = [v for v in shares if np.isfinite(v)]
    flagged = flag_grid(rows=rows, flagged=p_neg > 0.5, day_dates=cal.dates)
    on_days = np.isin(cal.dates.astype("datetime64[D]").astype(int), np.unique(rows.date))
    return {
        "sign_change_near_zero_share": float(np.mean(finite)) if finite else float("nan"),
        "bsp_residual": real.bsp_residual_level(bsp=cal.bsp[on_days], flagged=flagged[on_days]),
    }


def stage_real(
    *,
    ctx: Context,
    arch: str,
    metrics: dict[str, score.RowMetrics],
    clim: base.Climatology,
    reg: base.Regression,
    bstar: str,
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
    rec_rows = record_values_for_rows(
        segments=sets["k1"], record=data.record_by_day(record=data.metered_record(), days=cal.dates)
    )
    units_rows = record_values_for_rows(
        segments=sets["k1"],
        record=data.record_by_day(record=data.metered_record(), days=cal.dates, column="units"),
    )
    aux = [load_predictions(arm="G2", arch=arch, seed=s, name="real_k1") for s in core.ARCH_SEEDS]
    head = real.gaussian_mixture_median(
        mu=np.stack([a["aux_mu"] for a in aux]), sigma=np.stack([a["aux_s"] for a in aux])
    )
    running = np.nan_to_num(units_rows) >= 1
    m2_flag = (
        segment_p_any(segments=sets["k1"]) > 0.5
        if paths.sibling_available(prefix="00694-")
        else np.zeros(len(rec_rows), dtype=bool)
    )
    k1 = real.k1_engine_head(head_median=head, record=rec_rows, running=running, m2_flag=m2_flag)
    d_k1 = arm_metrics(arm="D", arch=arch, name="real_k1", x=rows_k1.x, seeds=core.ARCH_SEEDS)
    b_k1 = score.lobe_metrics(
        lobes=real.b_star_on_rows(
            rows=rows_k1, model=clim if bstar == "B1" else reg, pi=core.TEST_MIX_INJECTED
        ),
        x=rows_k1.x,
    )
    k1["crps_d"] = float(d_k1.crps.mean())
    k1["crps_bstar"] = float(b_k1.crps.mean())
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
            "share_below_1mva_by_era_and_hour": {
                f"{e}_{h}": float(np.mean(rows_e.r[(eras == e) & (rows_e.hh // 2 == h)] < real.NEAR_ZERO_POINTS))
                for e in np.unique(eras)
                for h in range(24)
                if ((eras == e) & (rows_e.hh // 2 == h)).any()
            },
            **r3_extras(rows=rows_e, p_neg=d_e.p_neg, cal=cal),
        },
    }  # fmt: skip
    log("real checks done")
    return out


def record_values_for_rows(*, segments: Sequence[sim.Segment], record: np.ndarray) -> np.ndarray:
    """Return a per-day record array's value for each valid half-hour of the K1 segments.

    The values come in `Rows` order. A half-hour with no record row stays NaN.
    """
    out = []
    for seg in segments:
        for d in np.flatnonzero(seg.cal_index >= 0):
            ok = seg.valid[d] & np.isfinite(seg.x[d])
            out.append(record[seg.cal_index[d]][ok])
    return np.concatenate(out)


# --------------------------------------------------------------------------- figures 1, 2, and 6

MONTH_LABELS: Final = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def era_month_shares(*, ctx: Context) -> dict[str, dict[str, float]]:
    """Return, per era and calendar month of the year, the share of days under 1.0 MVA at the minimum.

    Only the month of the year is kept, so that the figure shows no calendar date.
    """
    cal = ctx.table.calendar
    month = np.array([int(str(d)[5:7]) for d in cal.dates])
    out: dict[str, dict[str, float]] = {}
    for era in np.unique(ctx.table.era):
        on = ctx.table.era == era
        out[str(era)] = {}
        for m in range(1, 13):
            sel = on & (month == m) & np.isfinite(cal.reading).any(axis=1)
            if sel.sum() >= core.MIN_DAYS_IN_MONTH // 2:
                share, _ = banks.near_zero_shares(r=cal.reading[sel])
                out[str(era)][MONTH_LABELS[m - 1]] = share
    return out


def gallery_series(*, seg: sim.Segment) -> dict[str, np.ndarray]:
    """Return one segment's first two base days as the series of figure 2 (96 half-hours)."""
    days = np.flatnonzero(seg.cal_index >= 0)[:2]
    pick = np.concatenate([np.arange(d * 48, d * 48 + 48) for d in days])

    def flat(values: np.ndarray) -> np.ndarray:
        return values.reshape(-1)[pick]

    bsp = flat(seg.bsp)
    return {
        "base demand": flat(seg.base), "engine": flat(seg.engine), "solar": flat(seg.solar),
        "signed truth": flat(seg.x), "folded reading": flat(seg.r),
        "BSP step": np.r_[np.nan, np.diff(bsp)],
    }  # fmt: skip


def example_windows(*, rows: score.Rows, segments: Sequence[sim.Segment]) -> dict[str, np.ndarray]:
    """Pick one scored day for each of the four examples of figure 6 and return its row indices.

    The four are an import-only day, an engine-reversal day, a solar-reversal day, and a combined
    day. A kind with no matching day is left out.
    """
    picked: dict[str, np.ndarray] = {}
    for label, world in (
        ("import only", "none"), ("engine reversal", "engine"), ("solar reversal", "solar"),
        ("engine and solar", "both"),
    ):  # fmt: skip
        for k, seg in enumerate(segments):
            if seg.world != world:
                continue
            for day_int in np.unique(rows.date[rows.seg == k]):
                day = np.flatnonzero((rows.seg == k) & (rows.date == day_int))
                complete = len(day) == 48
                reverses = bool((rows.x[day] < -core.DEAD_BAND_POINTS).any())
                if complete and (reverses or world == "none"):
                    picked[label] = day
                    break
            if label in picked:
                break
    return picked


def stage_figures(*, ctx: Context, arch: str, log: Log) -> list[Path]:
    """Write figures 1, 2, and 6 from the real aggregates, the synthetic banks, and arm D's predictions.

    Figure 1 keeps the month of the year only. Figures 2 and 6 show synthetic worlds built on real base
    days, with hours of the day on the axis and no calendar date.

    Args:
        ctx: The context.
        arch: The chosen architecture.
        log: Receives progress lines.

    Returns:
        The written paths (run `svgo` on each before committing).
    """
    import e00729_figures as figures

    out_dir = paths.data_root() / "figures"
    test = banks.load_bank(path=bank_path(kind="test"))
    written: list[Path] = []
    shares = era_month_shares(ctx=ctx)
    months = [m for m in MONTH_LABELS if any(m in v for v in shares.values())]
    chart1 = figures.dot_chart(
        scores=shares, rows=months, title="Days with a daily minimum below 1.0 MVA, by era and month",
        x_title="Share of days",
    )  # fmt: skip
    figures.save_svg(chart=chart1, path=out_dir / "figure_1_eras.svg")
    written.append(out_dir / "figure_1_eras.svg")
    s2 = [s for s in test.segments if s.source == "S2" and s.world in ("solar", "both")][:6]
    if s2:
        gallery = figures.alt.vconcat(
            *(
                figures.series_chart(
                    series=gallery_series(seg=s), title=f"Synthetic segment {i + 1}"
                )
                for i, s in enumerate(s2)
            )
        )
        figures.save_svg(chart=gallery, path=out_dir / "figure_2_gallery.svg")
        written.append(out_dir / "figure_2_gallery.svg")
    rows = score.rows_from_segments(segments=test.segments, holidays=ctx.cal.holidays)
    pred = load_predictions(arm="D", arch=arch, seed=0, name="test")
    lobes = score.dist.Lobes(pred["p"], pred["mu_i"], pred["s_i"], pred["mu_r"], pred["s_r"])
    q = score.dist.mixture_quantiles(lobes=lobes, levels=(0.05, 0.5, 0.95))
    panels = []
    for label, day in example_windows(rows=rows, segments=test.segments).items():
        panels.append(
            figures.band_chart(
                truth=rows.x[day], lower=q[day, 0], median=q[day, 1], upper=q[day, 2],
                p_neg=score.dist.prob_negative(lobes=lobes)[day], title=label,
            )
        )  # fmt: skip
    if panels:
        figures.save_svg(chart=figures.alt.vconcat(*panels), path=out_dir / "figure_6_windows.svg")
        written.append(out_dir / "figure_6_windows.svg")
    log(f"wrote {len(written)} figures")
    return written


def stage_series(*, ctx: Context, arch: str, log: Log) -> list[Path]:
    """Write the 12 time-series figures from stored predictions only (no fit, no re-score).

    Args:
        ctx: The context (the real days of set R1 need its calendar).
        arch: The chosen architecture.
        log: Receives one line per figure.

    Returns:
        The written paths (run `svgo` on each before committing).
    """
    import e00729_series_figures as series

    test = banks.load_bank(path=bank_path(kind="test"))
    holidays = ctx.cal.holidays
    rows = score.rows_from_segments(segments=test.segments, holidays=holidays)
    sets = real_sets(ctx=ctx)
    real_rows = score.rows_from_segments(segments=sets["r1"], holidays=holidays)
    written = series.write_series_figures(
        arch=arch, test_segments=test.segments, rows=rows, real_segments=sets["r1"],
        real_rows=real_rows, out_dir=paths.data_root() / "figures", log=log,
    )  # fmt: skip
    log(f"wrote {len(written)} time-series figures")
    return written

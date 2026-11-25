"""Experiment 730: the stages, with the data root passed to every function.

A stage runs `banks.guard_run` first, so a stage on the outputs of another run, before the gate, or
before the gallery check, refuses to start. The stages that read real data (`real_context`,
`real_gate_inputs`, `real_sets`) are thin wrappers over experiment 729's loaders and need the real
files; every other function here takes arrays or files under `root`. Stages print counts and scores
only.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import e00729_banks as banks729
import e00729_score as score729
import e00729_stages as stages729
import e00730_arms as arms
import e00730_banks as banks
import e00730_core as core
import e00730_leakage as leak
import e00730_score as score
import e00730_signed as signed
import numpy as np

TEST_MARKER: Final = "test_started"
Log = banks.Log


def fit_dir(*, root: Path) -> Path:
    """Return the folder of the fit records."""
    return root / "fits"


def job_name(*, arm: str, arch: str, seed: int) -> str:
    """Return the file stem of a fit (`SD*` becomes `SDstar`, so no name holds an asterisk)."""
    return f"{arm.replace('*', 'star')}_{arch}_s{seed}"


# --------------------------------------------------------------------------- real-data wrappers


def real_context() -> stages729.Context:
    """Build experiment 729's context (days, split, masks) from the real files. Needs real data."""
    return stages729.build_context()


def bank_inputs(*, ctx: stages729.Context) -> banks.BankInputs:
    """Return the bank build's inputs from a context."""
    months = {s: ctx.months(split=s) for s in ("training", "validation", "test")}
    return banks.BankInputs(cal=ctx.cal, months=months, use=ctx.use)


def real_gate_inputs(*, ctx: stages729.Context) -> banks.GateInputs:
    """Return the realism gate's real arrays from a context. Needs real data."""
    data = stages729.data
    cal_all = ctx.table.calendar
    rec = data.record_by_day(record=data.metered_record(), days=cal_all.dates)
    days = np.flatnonzero(np.isfinite(rec).any(axis=1))
    return banks.GateInputs(
        real_days_by_era={
            "net": cal_all.reading[ctx.table.era == data.ERA_NET],
            "solar": cal_all.reading[ctx.table.era == data.ERA_SOLAR],
        },
        d_eng=stages729.real_step(values=rec[days], days=cal_all.dates[days]),
        d_bsp=stages729.real_step(values=cal_all.bsp[days], days=cal_all.dates[days]),
        placebo_slopes=stages729.placebo_slopes(
            rec=rec[days], bsp=cal_all.bsp[days], days=cal_all.dates[days]
        ),
    )


# --------------------------------------------------------------------------- choices


def choose_architecture(*, root: Path) -> dict:
    """Pick the architecture from the screen's validation Brier scores and write it once.

    Args:
        root: The data folder.

    Returns:
        The selection record, read back from `selection.json`.
    """
    banks.guard_run(root=root, stage="choose")
    limits = core.read_limits_file(path=root / "limits.json")
    scores, weights = {}, {}
    for arch in core.ARCHITECTURES:
        recs = [
            banks.read_json(path=fit_dir(root=root) / f"{job_name(arm=a, arch=arch, seed=s)}.json")
            for a in core.SCREEN_ARMS
            for s in core.ARCH_SEEDS
        ]
        scores[arch] = float(np.mean([r["best_val_brier"] for r in recs]))
        weights[arch] = int(recs[0]["n_weights"])
    chosen = core.choose_architecture(scores=scores, weights=weights, floor=limits.floor)
    record = {"chosen": chosen, "scores": scores, "weights": weights, "chosen_on": "validation"}
    path = root / "selection.json"
    if not path.exists():
        core.write_once_json(path=path, payload=record)
    return banks.read_json(path=path)


def write_margin(*, root: Path) -> dict:
    """Write `margin.json` once from arm S's validation Brier scores over seeds 0 to 4."""
    banks.guard_run(root=root, stage="margin")
    chosen = banks.read_json(path=root / "selection.json")["chosen"]
    limits = core.read_limits_file(path=root / "limits.json")
    vals = [
        banks.read_json(path=fit_dir(root=root) / f"{job_name(arm='S', arch=chosen, seed=s)}.json")[
            "best_val_brier"
        ]
        for s in core.MARGIN_SEEDS
    ]
    path = root / "margin.json"
    if path.exists():
        return banks.read_json(path=path)
    return core.write_margin_file(path=path, floor=limits.floor, val_brier_s_by_seed=vals)


def choose_style(*, root: Path, whole_day_demand_mae: dict[str, float]) -> dict:
    """Pick the masking style of SD* from the validation scores of SD-0, SD-K, SD-W, SD-R.

    Args:
        root: The data folder.
        whole_day_demand_mae: Each style's validation demand error on whole-day-hidden windows.

    Returns:
        The style record, read back from `style.json`.
    """
    banks.guard_run(root=root, stage="styles")
    chosen_arch = banks.read_json(path=root / "selection.json")["chosen"]
    limits = core.read_limits_file(path=root / "limits.json")
    val = {
        s: float(
            np.mean(
                [
                    banks.read_json(
                        path=fit_dir(root=root)
                        / f"{job_name(arm=f'SD-{s}', arch=chosen_arch, seed=seed)}.json"
                    )["best_val_brier"]
                    for seed in core.ARCH_SEEDS
                ]
            )
        )
        for s in core.STYLES
    }
    chosen = core.choose_style(
        val_brier=val, whole_day_demand_mae=whole_day_demand_mae, floor=limits.floor
    )
    path = root / "style.json"
    if not path.exists():
        core.write_once_json(
            path=path, payload={"chosen": chosen, "val_brier": val, "chosen_on": "validation"}
        )
    return banks.read_json(path=path)


def choice_events(*, root: Path) -> list[str]:
    """Return the choices and the first test score in order of file time (the marker is made now).

    A choice file rewritten after the first test score sorts after it, and
    `assert_choices_before_test` fails.
    """
    marker = root / "scores" / TEST_MARKER
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
    stamps = {
        "architecture": (root / "selection.json").stat().st_mtime_ns,
        "style": (root / "style.json").stat().st_mtime_ns,
        "margin": (root / "margin.json").stat().st_mtime_ns,
        "test_score": marker.stat().st_mtime_ns,
    }
    return sorted(stamps, key=lambda k: (stamps[k], k == "test_score"))


# --------------------------------------------------------------------------- scoring


def verdict(
    *,
    root: Path,
    rows: score729.Rows,
    p: dict[str, np.ndarray],
    per_row: dict[str, np.ndarray],
    n_resamples: int = core.N_RESAMPLES,
    stopped: bool = False,
    invalid: bool = False,
    failed: bool = False,
    h2_pass: bool = True,
) -> dict:
    """Score the test bank and return the verdict inputs, the checks, and the row of the table.

    Args:
        root: The data folder.
        rows: The test bank's rows.
        p: The seed-mean P(X < 0) of each arm on those rows (must hold "SD*").
        per_row: The seed-averaged per-row Brier score of each arm and architecture.
        n_resamples: Resamples of every counted contrast.
        stopped: Row 0 applies.
        invalid: Row 1 applies.
        failed: Row 2 applies.
        h2_pass: H2 (the real held-out days) passed.

    Returns:
        The record, with the counted contrasts, H1, the bars, and the verdict word.
    """
    banks.guard_run(root=root, stage="score")
    leak.assert_choices_before_test(events=choice_events(root=root))
    limits = core.read_limits_file(path=root / "limits.json")
    m = core.read_margin_file(path=root / "margin.json")
    readings = score.counted_contrasts(rows=rows, per_row=per_row, n_resamples=n_resamples)
    h1 = score.harm_check(rows=rows, p=p["SD*"], limit=limits.harm_limit)
    det = score.detection(rows=rows, p=p["SD*"])
    cal = score.calibration(rows=rows, p=p["SD*"], bar=limits.reliability_bar)
    v = core.VerdictInputs(
        stopped=stopped,
        invalid=invalid,
        failed=failed,
        h1_pass=bool(h1["pass"]),
        h2_pass=h2_pass,
        c1=readings["C1"].reading,
        margin=m,
        calibration_pass=bool(cal["pass"]),
        detection_pass=bool(det["pass"]),
    )
    return {
        "contrasts": {k: score.contrast_dict(reading=r) for k, r in readings.items()},
        "h1": h1,
        "detection": det,
        "calibration": {k: val for k, val in cal.items() if k != "bins"},
        "margin": m,
        "row": core.verdict_row(v=v),
        "verdict": core.verdict_word(v=v),
    }


def stage_names() -> Sequence[str]:
    """Return the stage names of the command line, in the order of the plan."""
    return (
        "banks", "gallery", "gate", "pretrain", "timing", "screen", "choose", "margin", "styles",
        "later", "refit729", "fit", "score", "real", "report", "figures", "series", "components",
    )  # fmt: skip


def spec_for(*, arm: str, style: str) -> arms.ArmSpec:
    """Return the arm named `arm`, with SD* using the chosen masking style."""
    return arms.arm_specs(style=style)[arm]


def run_fit(
    *,
    root: Path,
    ctx: stages729.Context,
    arm: str,
    arch: str,
    seed: int,
    style: str,
    device: str,
    log: Log,
    max_epochs: int = core.MAX_EPOCHS,
    encoder: arms.EncoderTable | None = None,
    pretrained: Path | None = None,
) -> dict:
    """Fit one network, save it, and store its probabilities on the validation and test banks.

    A fit whose record exists is skipped, so an interrupted run resumes. The pre-trained weight file
    must lie under this experiment's `pretrained/` folder.

    Args:
        root: The data folder.
        ctx: The context (needs real data).
        arm: The arm.
        arch: The architecture.
        seed: The seed.
        style: The chosen masking style of SD*.
        device: "cuda" or "cpu".
        log: Receives progress lines.
        max_epochs: The cap on epochs.
        encoder: D8's encoder input table (arms SDe-PT and SDe-FR).
        pretrained: The pre-trained weight file of an encoder arm.

    Returns:
        The fit record.
    """
    import e00729_train as train729
    import e00730_train as train

    banks.guard_run(root=root, stage="fit")
    name = job_name(arm=arm, arch=arch, seed=seed)
    record_path = fit_dir(root=root) / f"{name}.json"
    if record_path.exists():
        return banks.read_json(path=record_path)
    if pretrained is not None:
        leak.assert_no_foreign_weights(used=[pretrained], pretrained_root=root / "pretrained")
    train729.limit_gpu_memory()
    spec = spec_for(arm=arm, style=style)
    train_bank = banks729.load_bank(path=banks.bank_path(root=root, kind="train", seed=seed))
    val_bank = banks729.load_bank(path=banks.bank_path(root=root, kind="validation"))
    kwargs = {"holidays": ctx.cal.holidays, "stats": ctx.stats, "encoder": encoder}
    prep_train = arms.prepare_segments(segments=train_bank.segments, spec=spec, **kwargs)
    prep_val = arms.prepare_segments(segments=val_bank.segments, spec=spec, **kwargs)
    net, record = train.fit(
        spec=spec,
        arch=arch,
        seed=seed,
        train=prep_train,
        val=prep_val,
        device=device,
        max_epochs=max_epochs,
        out_path=root / "networks" / f"{name}.pt",
        frozen_encoders=arm == "SDe-FR",
        pretrained=pretrained,
        log=log,
    )
    store_predictions(
        root=root, name=name, net=net, spec=spec, device=device, ctx=ctx, kwargs=kwargs,
        val_bank=val_bank,
    )  # fmt: skip
    banks.write_json(path=record_path, payload=record.as_dict())
    return record.as_dict()


def store_predictions(
    *,
    root: Path,
    name: str,
    net: object,
    spec: arms.ArmSpec,
    device: str,
    ctx: stages729.Context,
    kwargs: dict,
    val_bank: banks729.Bank,
) -> None:
    """Store a fit's probabilities and demand outputs on every bank, the real sets, and read-outs.

    Every array is flattened over the valid half-hours in `Rows` order. Arms with the demand output
    also store the demand read-out under the fixed masks of each style, and the causal arm stores
    its probabilities at each of the 6 values of k on the test bank.
    """
    import e00729_train as train729
    import e00730_train as train

    folder = root / "predictions" / name
    banks_by_kind = {"validation": val_bank}
    for kind in ("test", "shifted"):
        banks_by_kind[kind] = banks729.load_bank(path=banks.bank_path(root=root, kind=kind))
    prepared = {
        kind: arms.prepare_segments(segments=bank.segments, spec=spec, **kwargs)
        for kind, bank in banks_by_kind.items()
    }
    for kind, prep in prepared.items():
        decoded, targets, _ = train.predict(net=net, prepared=prep, spec=spec, device=device)
        flat = {k: train.valid_flat(values=v, targets=targets) for k, v in decoded.items()}
        train729.save_predictions(arrays=flat, path=folder / f"{kind}.npz")
    for set_name, segments in real_sets(ctx=ctx).items():
        prep = arms.prepare_segments(segments=segments, spec=spec, **kwargs)
        decoded, targets, _ = train.predict(net=net, prepared=prep, spec=spec, device=device)
        flat = {k: train.valid_flat(values=v, targets=targets) for k, v in decoded.items()}
        train729.save_predictions(arrays=flat, path=folder / f"real_{set_name}.npz")
    if spec.demand:
        for kind in ("validation", "test"):
            for style in core.STYLES:
                out = train.masked_demand(
                    net=net, prepared=prepared[kind], spec=spec, device=device, style=style
                )
                train729.save_predictions(arrays=out, path=folder / f"{kind}_masked_{style}.npz")
    if spec.causal:
        for k in core.CAUSAL_K:
            p = train.causal_predictions(
                net=net, prepared=prepared["test"], spec=spec, device=device, k=k
            )
            train729.save_predictions(arrays={"p": p}, path=folder / f"test_causal_k{k}.npz")


def real_sets(*, ctx: stages729.Context) -> dict:
    """Return experiment 729's real-day pseudo-segments (needs real data)."""
    return stages729.real_sets(ctx=ctx)


# --------------------------------------------------------------------------- job loops


def arm_alias(*, arm: str, style: str) -> str:
    """Return the fitted arm that supplies `arm`: SD* is the fit of the style chosen."""
    return f"SD-{style}" if arm == "SD*" else arm


def run_jobs(
    *,
    root: Path,
    jobs: Sequence[tuple[str, str, int]],
    run_one: Callable[[str, str, int], dict],
    log: Log,
) -> dict[str, int]:
    """Run fits one at a time; a fit that raises is recorded, not rerun, and the run goes on.

    A fit whose record exists is skipped, so an interrupted run resumes. A raised fit writes
    `<name>.failed.json` (counts only), which makes the verdict row 2 when the fit was needed.

    Args:
        root: The data folder.
        jobs: The (arm, architecture, seed) of each fit.
        run_one: Runs one fit and writes its record.
        log: Receives progress lines.

    Returns:
        The numbers of fits run, skipped, and failed.
    """
    counts = {"run": 0, "skipped": 0, "failed": 0}
    for arm, arch, seed in jobs:
        name = job_name(arm=arm, arch=arch, seed=seed)
        if (fit_dir(root=root) / f"{name}.json").exists():
            counts["skipped"] += 1
            continue
        try:
            run_one(arm, arch, seed)
            counts["run"] += 1
        except Exception as exc:  # noqa: BLE001
            counts["failed"] += 1
            banks.write_json(
                path=fit_dir(root=root) / f"{name}.failed.json",
                payload={"error": type(exc).__name__},
            )
            log(f"fit {name} failed: {type(exc).__name__}")
    return counts


def timing_passed(*, root: Path) -> bool:
    """Return whether `scores/timing.json` records a passed timing gate."""
    path = root / "scores" / "timing.json"
    return path.exists() and bool(banks.read_json(path=path).get("pass"))


def stage_timing(
    *,
    root: Path,
    seconds_per_epoch: Mapping[str, float],
    peak_memory_gb: float,
    manager_approved: bool,
    log: Log,
) -> dict:
    """Write `scores/timing.json` from the measured seconds an epoch of each architecture.

    Args:
        root: The data folder.
        seconds_per_epoch: Measured by 5 epochs of S on each architecture (the CLI measures them).
        peak_memory_gb: The peak resident memory of the pre-training row build.
        manager_approved: The manager approved a projection above the stop line.
        log: Receives progress lines.

    Returns:
        The report with `pass`.

    Raises:
        StoppedError: When the projection or the peak memory is over its line.
    """
    banks.guard_run(root=root, stage="timing")
    report = core.timing_gate_report(
        seconds_per_epoch=seconds_per_epoch, peak_memory_gb=peak_memory_gb
    )
    passes = core.timing_gate_passes(report=report, manager_approved=manager_approved)
    out = {**report, "pass": passes}
    banks.write_json(path=root / "scores" / "timing.json", payload=out)
    log(f"timing gate: projection {report['hours_at_cap']:.2f} h, pass {passes}")
    if not passes:
        msg = "the timing gate failed: the projection or the peak memory is over its line (row 0)"
        raise banks.StoppedError(msg)
    return out


def _fit_stage(
    *,
    root: Path,
    stage: str,
    jobs: Sequence[tuple[str, str, int]],
    run_one: Callable[[str, str, int], dict],
    log: Log,
) -> dict[str, int]:
    """Guard a fit stage, check the timing gate, and run its jobs."""
    banks.guard_run(root=root, stage=stage)
    if not timing_passed(root=root):
        msg = f"`{stage}` needs scores/timing.json with a pass: run `timing` first"
        raise RuntimeError(msg)
    return run_jobs(root=root, jobs=jobs, run_one=run_one, log=log)


def chosen_architecture(*, root: Path) -> str:
    """Return the architecture that `selection.json` records."""
    return banks.read_json(path=root / "selection.json")["chosen"]


def stage_screen(*, root: Path, run_one: Callable[[str, str, int], dict], log: Log) -> dict:
    """Fit arms S-A and S on the 4 architectures, seeds 0 to 2 (24 fits)."""
    return _fit_stage(root=root, stage="screen", jobs=core.screen_jobs(), run_one=run_one, log=log)


def stage_styles(*, root: Path, run_one: Callable[[str, str, int], dict], log: Log) -> dict:
    """Fit SD-0, SD-K, SD-W, and SD-R on the chosen architecture (12 fits)."""
    jobs = core.style_jobs(arch=chosen_architecture(root=root))
    return _fit_stage(root=root, stage="styles", jobs=jobs, run_one=run_one, log=log)


def stage_later(*, root: Path, run_one: Callable[[str, str, int], dict], log: Log) -> dict:
    """Fit S seeds 3 and 4, SDe-PT, SDe-FR, SD*-C, and SD*-lambda (14 fits)."""
    jobs = core.later_jobs(arch=chosen_architecture(root=root))
    return _fit_stage(root=root, stage="later", jobs=jobs, run_one=run_one, log=log)


def stage_refit729(*, root: Path, run_one: Callable[[str, str, int], dict], log: Log) -> dict:
    """Refit experiment 729's arms A, D, and D-sign on the bilstm (9 fits, cap 200)."""
    return _fit_stage(root=root, stage="refit729", jobs=core.refit_jobs(), run_one=run_one, log=log)


def whole_day_demand_mae(*, root: Path, arch: str) -> dict[str, float]:
    """Return each style's validation demand error on windows with the whole day hidden."""
    out = {}
    for style in core.STYLES:
        vals = []
        for seed in core.ARCH_SEEDS:
            name = job_name(arm=f"SD-{style}", arch=arch, seed=seed)
            z = np.load(root / "predictions" / name / "validation_masked_W.npz")
            vals.append(float(np.abs(z["demand_mu"] - z["base"]).mean()))
        out[style] = float(np.mean(vals))
    return out


# --------------------------------------------------------------------------- gallery stage


def stage_gallery(*, root: Path, shares_real: Mapping[str, float] | None, log: Log) -> dict:
    """Draw the synthetic world gallery from the rebuilt banks and write the automatic check.

    The published panels go to `figures/gallery/` and every single metered-farm day to
    `PRIVATE/gallery/`. The automatic check goes to `PRIVATE/gallery/auto_check.json`. This stage
    never writes `panel_check.json`, which records the design lead's reading of each panel
    (`record_panel_check`).

    Args:
        root: The data folder.
        shares_real: The real aggregates for panel G8, or None.
        log: Receives progress lines.

    Returns:
        The automatic result of each panel.
    """
    import e00730_figures as figures
    import e00730_gallery as gallery

    banks.guard_run(root=root, stage="gallery")
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    train0 = banks729.load_bank(path=banks.bank_path(root=root, kind="train", seed=0))
    segments = [*test.segments, *train0.segments]
    summary = banks.read_json(path=root / "banks" / "summary.json")
    components = {
        f"{k}/{n}": float(v)
        for k, shares in summary["components"]["shares"].items()
        for n, v in shares.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }
    near = {}
    for world in ("none", "engine", "solar", "both"):
        parts = [s.r[1:-1].reshape(-1) for s in segments if s.world == world]
        r = np.concatenate(parts) if parts else np.array([np.nan])
        near[world] = float(np.nanmean(r < core.DECIDABLE_READING_POINTS))
    out = figures.build_gallery(
        segments=segments,
        shares_synthetic=near,
        shares_real=shares_real,
        component_shares=components,
    )
    private = gallery.gallery_private_dir(root=root)
    figures.save_gallery(out=out, public_dir=root / "figures" / "gallery", private_dir=private)
    banks.write_json(
        path=private / "auto_check.json",
        payload=gallery.build_panel_check(results=out.results, notes=out.notes),
    )
    log(f"gallery drawn: {sum(out.results.values())} of {len(out.results)} panels pass the check")
    return out.results


def record_panel_check(*, root: Path, design_lead_confirms: Mapping[str, bool]) -> dict:
    """Write `panel_check.json`: a panel passes if the automatic check and the design lead agree.

    Args:
        root: The data folder.
        design_lead_confirms: For each panel, whether the design lead read it against the written
            list and found that it shows what the list requires.

    Returns:
        The record written.
    """
    import e00730_gallery as gallery

    auto = banks.read_json(path=gallery.gallery_private_dir(root=root) / "auto_check.json")[
        "panels"
    ]
    results = {
        k: bool(auto.get(k)) and bool(design_lead_confirms.get(k)) for k in core.PANEL_REQUIREMENTS
    }
    record = gallery.build_panel_check(results=results, notes={"confirmed_by": "the design lead"})
    gallery.write_panel_check(root=root, record=record)
    return record


# --------------------------------------------------------------------------- scoring


def p_neg_of(*, pred: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Return P(X < 0) of a stored prediction: the sign head's `p`, or the lobes' negative mass."""
    if "mu_i" in pred:
        s = score729
        lobes = s.dist.Lobes(pred["p"], pred["mu_i"], pred["s_i"], pred["mu_r"], pred["s_r"])
        return s.lobe_metrics(lobes=lobes, x=x).p_neg
    return pred["p"]


def load_p(
    *, root: Path, arm: str, arch: str, seeds: Sequence[int], name: str, x: np.ndarray
) -> np.ndarray:
    """Return the seed-mean P(X < 0) of a fitted arm on a stored set."""
    ps = []
    for seed in seeds:
        folder = root / "predictions" / job_name(arm=arm, arch=arch, seed=seed)
        z = np.load(folder / f"{name}.npz")
        ps.append(p_neg_of(pred={k: z[k] for k in z.files}, x=x))
    return score.seed_mean(p_by_seed=ps)


def load_preds(
    *, root: Path, arm: str, arch: str, seeds: Sequence[int], name: str
) -> list[dict[str, np.ndarray]]:
    """Return the stored predictions of an arm on one set, one dictionary per seed."""
    out = []
    for seed in seeds:
        z = np.load(root / "predictions" / job_name(arm=arm, arch=arch, seed=seed) / f"{name}.npz")
        out.append({k: z[k] for k in z.files})
    return out


def real_h2_passed(*, root: Path) -> bool | None:
    """Return the H2 verdict of `scores/real.json`, or None when the real stage has not run."""
    path = root / "scores" / "real.json"
    return None if not path.exists() else bool(banks.read_json(path=path)["h2"]["pass"])


def stage_score(
    *,
    root: Path,
    holidays: np.ndarray,
    n_resamples: int = core.N_RESAMPLES,
    baselines: tuple[object, object] | None = None,
    log: Log,
) -> dict:
    """Score the test bank: the counted contrasts, the checks, the report-only readings, verdict.

    Needs the choices, the margin, the limits, and the real stage's H2 reading on disk. Writes
    `scores/verdict.json`.

    Args:
        root: The data folder.
        holidays: Bank-holiday dates (the rows' features).
        n_resamples: Resamples of every contrast.
        baselines: Experiment 729's fitted climatology and regression (from `fit_baselines`), or
            None to leave out the readings that need B1 (R2 to R6, T1, and figures 2 and 9).
        log: Receives progress lines.

    Returns:
        The verdict record.
    """
    banks.guard_run(root=root, stage="score")
    arch = chosen_architecture(root=root)
    style = banks.read_json(path=root / "style.json")["chosen"]
    h2 = real_h2_passed(root=root)
    failed = any(fit_dir(root=root).glob("*.failed.json"))
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    p: dict[str, np.ndarray] = {}
    for arm in (*core.ARMS_NEW, *core.REFIT_729_ARMS):
        if arm == "SD*":
            continue
        a = core.REFIT_729_ARCH if arm in core.REFIT_729_ARMS else arch
        try:
            p[arm] = load_p(
                root=root, arm=arm, arch=a, seeds=core.ARCH_SEEDS, name="test", x=rows.x
            )
        except FileNotFoundError:
            continue
    p["SD*"] = p[f"SD-{style}"]
    per_row = {arm: score.row_brier(p=v, x=rows.x) for arm, v in p.items()}
    for a in ("transformer", "bilstm"):
        pa = load_p(root=root, arm="S", arch=a, seeds=core.ARCH_SEEDS, name="test", x=rows.x)
        per_row[a] = score.row_brier(p=pa, x=rows.x)
    result = verdict(
        root=root,
        rows=rows,
        p=p,
        per_row=per_row,
        n_resamples=n_resamples,
        failed=failed,
        h2_pass=bool(h2),
    )
    result["h2_pending"] = h2 is None
    limits = core.read_limits_file(path=root / "limits.json")
    result["checks_by_arm"] = {
        arm: {
            "h1": score.harm_check(rows=rows, p=v, limit=limits.harm_limit),
            "detection": score.detection(rows=rows, p=v),
            "brier": float(per_row[arm].mean()),
            "log_loss": float(score.row_log_loss(p=v, x=rows.x).mean()),
        }
        for arm, v in p.items()
    }
    result["report_only"] = report_only(rows=rows, per_row=per_row, n_resamples=n_resamples)
    result["signed"] = signed_stage(
        root=root, rows=rows, p=p, test=test, arch=arch, style=style, baselines=baselines,
        n_resamples=n_resamples,
    )  # fmt: skip
    result["figure2"] = signed.brier_by_subset(rows=rows, p_by_arm={"P0": np.zeros(rows.n), **p})
    result["figure5"] = {
        arm: signed.harm_detection_by_group(rows=rows, p=p[arm]) for arm in ("S", "SD*") if arm in p
    }
    result["causal"] = causal_readout(
        root=root, rows=rows, arch=arch, p_s=p["S"], n_resamples=n_resamples
    )
    result["shifted_mix"] = shifted_table(
        root=root, arch=arch, style=style, limit=limits.harm_limit, holidays=holidays
    )
    banks.write_json(path=root / "scores" / "verdict.json", payload=result)
    log(f"verdict row {result['row']}: {result['verdict']}")
    return result


def signed_stage(
    *,
    root: Path,
    rows: score729.Rows,
    p: Mapping[str, np.ndarray],
    test: banks729.Bank,
    arch: str,
    style: str,
    baselines: tuple[object, object] | None,
    n_resamples: int,
) -> dict:
    """Return R2 to R6, the signed R1, T1, and figure 9's data (empty without the baselines)."""
    if baselines is None:
        return {"skipped": "no fitted baselines were passed"}
    clim, reg = baselines
    metrics = dict(
        stages729.baseline_metrics(
            rows=rows, climatology=clim, regression=reg, pi=banks729.injected_share(bank=test)
        )
    )
    seeds = core.ARCH_SEEDS
    for arm, a in (
        ("S", arch),
        ("SD*", arch),
        *((r, core.REFIT_729_ARCH) for r in core.REFIT_729_ARMS),
    ):
        name = f"SD-{style}" if arm == "SD*" else arm
        try:
            preds = load_preds(root=root, arm=name, arch=a, seeds=seeds, name="test")
        except FileNotFoundError:
            continue
        metrics[arm] = signed.arm_signed_metrics(preds=preds, rows=rows)
    out = signed.signed_readout(rows=rows, metrics=metrics, n_resamples=n_resamples)
    if "S" in p and "D-sign" in p:
        out["R1_log_loss"] = score.contrast_dict(
            reading=score.report_only_contrast(
                rows=rows,
                a=score.row_log_loss(p=p["S"], x=rows.x),
                b=score.row_log_loss(p=p["D-sign"], x=rows.x),
                n_resamples=n_resamples,
            )
        )
    return out


def report_only(
    *, rows: score729.Rows, per_row: Mapping[str, np.ndarray], n_resamples: int
) -> dict:
    """Return the report-only Brier contrasts R1 and R7 to R10 that the arms here supply."""
    pairs = {
        "R1": ("S", "D-sign"),
        "R7": ("SD-0", "S"),
        "R8": ("SDe-FR", "SD*"),
        "R9": ("SD*-lambda", "SD*"),
        "R10": ("SD*-C", "SD*"),
    }
    out = {}
    for name, (a, b) in pairs.items():
        if a in per_row and b in per_row:
            reading = score.report_only_contrast(
                rows=rows, a=per_row[a], b=per_row[b], n_resamples=n_resamples
            )
            out[name] = score.contrast_dict(reading=reading)
    return out


def causal_readout(
    *, root: Path, rows: score729.Rows, arch: str, p_s: np.ndarray, n_resamples: int
) -> dict:
    """Return the causal protocol: SD*-C's Brier score at each k, and R11 to R22."""
    out: dict = {"brier_by_k": {}, "vs_p0": {}, "vs_s": {}}
    zeros = score.row_brier(p=np.zeros(rows.n), x=rows.x)
    per_s = score.row_brier(p=p_s, x=rows.x)
    for k in core.CAUSAL_K:
        files = [
            root
            / "predictions"
            / job_name(arm="SD*-C", arch=arch, seed=s)
            / f"test_causal_k{k}.npz"
            for s in core.ARCH_SEEDS
        ]
        if not all(f.exists() for f in files):
            continue
        b = score.row_brier(p=score.seed_mean(p_by_seed=[np.load(f)["p"] for f in files]), x=rows.x)
        keep = score.causal_keep(hh=rows.hh, k=k)
        out["brier_by_k"][k] = float(b[keep].mean())
        for label, ref in (("vs_p0", zeros), ("vs_s", per_s)):
            r = score.report_only_contrast(
                rows=rows, a=b, b=ref, keep=keep, n_resamples=n_resamples
            )
            out[label][k] = score.contrast_dict(reading=r)
    return out


def shifted_table(*, root: Path, arch: str, style: str, limit: float, holidays: np.ndarray) -> dict:
    """Return T2: S and SD* on the shifted-mix bank (Brier score, reliability, H1; no bar)."""
    bank = banks729.load_bank(path=banks.bank_path(root=root, kind="shifted"))
    rows = score729.rows_from_segments(segments=bank.segments, holidays=holidays)
    out = {}
    for label, arm in (("S", "S"), ("SD*", f"SD-{style}")):
        p = load_p(root=root, arm=arm, arch=arch, seeds=core.ARCH_SEEDS, name="shifted", x=rows.x)
        rel = score.calibration(rows=rows, p=p, bar=float("inf"))
        out[label] = {
            "brier": score.mean_brier(p=p, x=rows.x),
            "reliability": float(rel["reliability"]),
            "h1": score.harm_check(rows=rows, p=p, limit=limit),
        }
    return out


# --------------------------------------------------------------------------- real, report, figures


def stage_real(
    *,
    root: Path,
    real_rows: Mapping[str, score729.Rows],
    era_of_row: np.ndarray,
    baselines: tuple[object, object] | None = None,
    log: Log,
) -> dict:
    """Read H2, K1, K2, R2, and R3 from the stored real predictions and write `scores/real.json`.

    Args:
        root: The data folder.
        real_rows: The rows of the real sets `r1`, `k1`, `k2`, and `eras`, from the pseudo-segments.
        era_of_row: The era label of each row of the `eras` set.
        baselines: Experiment 729's fitted climatology and regression. With them, K1 also reports
            B1's scores and the demand error against the demand climatology, and K1 passes only
            where both of its bars hold. Without them, K1 reads the Brier bar alone.
        log: Receives progress lines.

    Returns:
        The record (aggregates only).
    """
    import e00730_real as real

    banks.guard_run(root=root, stage="real")
    arch = chosen_architecture(root=root)
    style = banks.read_json(path=root / "style.json")["chosen"]
    limits = core.read_limits_file(path=root / "limits.json")
    arm = f"SD-{style}"

    def p_of(name: str, rows: score729.Rows) -> np.ndarray:
        return load_p(
            root=root, arm=arm, arch=arch, seeds=core.ARCH_SEEDS, name=f"real_{name}", x=rows.x
        )

    r1, k1, k2, eras = (real_rows[n] for n in ("r1", "k1", "k2", "eras"))
    p_r1, p_k1, p_k2, p_eras = p_of("r1", r1), p_of("k1", k1), p_of("k2", k2), p_of("eras", eras)
    decidable_import = (k1.r >= core.DECIDABLE_READING_POINTS) & (k1.x >= 0)
    n_reverse = int(((k1.x < 0) & (k1.r >= core.DECIDABLE_READING_POINTS)).sum())
    k1_rec = real.k_reading(p=p_k1, x=k1.x, p0=0.0)
    if baselines is not None:
        mu_k1 = np.mean(
            [
                z["demand_mu"]
                for z in load_preds(
                    root=root, arm=arm, arch=arch, seeds=core.ARCH_SEEDS, name="real_k1"
                )
            ],
            axis=0,
        )
        test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
        k1_rec |= real.k1_against_baselines(
            rows=k1, p=p_k1, mu=mu_k1, climatology=baselines[0],
            pi=banks729.injected_share(bank=test),
        )  # fmt: skip
    if n_reverse < core.K1_MIN_REVERSE_DECIDABLE:
        k1_status = "void"
    else:
        bars = [k1_rec["brier"] < k1_rec["brier_p0"]]
        if "demand_mae" in k1_rec:
            bars.append(k1_rec["demand_mae"] < k1_rec["demand_mae_climatology"])
        k1_status = "pass" if all(bars) else "fail"
    k2_rec = score.harm_from_p(p=p_k2, limit=limits.harm_limit)
    z = np.load(root / "predictions" / job_name(arm=arm, arch=arch, seed=0) / "real_eras.npz")
    above = z["demand_mu"] > eras.r if "demand_mu" in z.files else np.zeros(eras.n, dtype=bool)
    era_hour = np.char.add(np.char.add(era_of_row.astype(str), "/"), (eras.hh // 2).astype(str))
    out = {
        "h2": real.h2(p=p_r1, limit=limits.harm_limit),
        "k1": {
            **k1_rec,
            "status": k1_status,
            "n_reverse_decidable": n_reverse,
            "h1": score.harm_from_p(p=p_k1[decidable_import], limit=limits.harm_limit),
        },
        "k2": {**k2_rec, "status": "pass" if k2_rec["pass"] else "fail"},
        "r2": {
            "share_demand_above_reading": real.share_by(values=above, groups=era_of_row),
            "share_demand_above_by_era_and_hour": real.share_by(values=above, groups=era_hour),
            "share_p_above_half": real.share_by(values=p_eras > 0.5, groups=era_of_row),
        },
        "r3": {
            "share_p_above_half_by_era": real.share_by(values=p_eras > 0.5, groups=era_of_row),
            "share_by_era_and_hour": real.share_by(values=p_eras > 0.5, groups=era_hour),
        },
    }
    banks.write_json(path=root / "scores" / "real.json", payload=out)
    log(f"real checks: H2 pass {out['h2']['pass']}, K1 {k1_status}, K2 {out['k2']['status']}")
    return out


def stage_report(*, root: Path) -> str:
    """Return the printed report of `scores/verdict.json` (counts and scores only)."""
    import e00730_report as report

    banks.guard_run(root=root, stage="report")
    return report.format_verdict(record=banks.read_json(path=root / "scores" / "verdict.json"))


def stage_components(*, root: Path) -> dict:
    """Return the component checks recorded in `banks/summary.json`."""
    banks.guard_run(root=root, stage="components")
    return banks.read_json(path=root / "banks" / "summary.json")["components"]


def stage_figures(*, root: Path, log: Log) -> list[Path]:
    """Draw the result figures that the stored scores support, and write them as SVG."""
    import e00730_figures as figures

    banks.guard_run(root=root, stage="figures")
    v = banks.read_json(path=root / "scores" / "verdict.json")
    folder = root / "figures"
    readings = {k: (r["point"], r["lower"], r["upper"]) for k, r in v["contrasts"].items()}
    written = []
    figures.save_svg(
        chart=figures.contrast_chart(readings=readings, margin=v["margin"]),
        path=folder / "contrasts.svg",
    )
    written.append(folder / "contrasts.svg")
    by_k = v["causal"]["brier_by_k"]
    if by_k:
        chart = figures.causal_readout_chart(
            brier_by_k={int(k): b for k, b in by_k.items()},
            two_sided=v["checks_by_arm"]["SD*"]["brier"],
            p0=v["checks_by_arm"]["SD*"].get("p0_brier", 0.0),
        )
        figures.save_svg(chart=chart, path=folder / "causal.svg")
        written.append(folder / "causal.svg")
    figure_data = (
        ("brier_by_subset.svg", v.get("figure2"), figures.brier_dot_chart),
        ("signed_crps.svg", v.get("signed", {}).get("figure9"), figures.signed_crps_dot_chart),
    )
    for fname, data, draw in figure_data:
        if data:
            figures.save_svg(chart=draw(scores=data), path=folder / fname)
            written.append(folder / fname)
    if v.get("figure5"):
        limit = core.read_limits_file(path=root / "limits.json").harm_limit
        chart = figures.harm_detection_chart(by_arm=v["figure5"], limit=limit)
        figures.save_svg(chart=chart, path=folder / "harm_detection.svg")
        written.append(folder / "harm_detection.svg")
    real_path = root / "scores" / "real.json"
    if real_path.exists():
        r = banks.read_json(path=real_path)
        chart = figures.real_eras_chart(
            above_half=r["r3"]["share_by_era_and_hour"],
            demand_above=r["r2"]["share_demand_above_by_era_and_hour"],
        )
        figures.save_svg(chart=chart, path=folder / "real_eras.svg")
        written.append(folder / "real_eras.svg")
    log(f"{len(written)} figures written")
    return written


def stage_series(
    *, root: Path, table: object, windows: Mapping[str, Mapping[str, np.ndarray]], log: Log
) -> list[Path]:
    """Draw the time series figures of the rules S1 to S16 (S1 windows to the private folder only).

    Args:
        root: The data folder.
        table: The `WindowTable` of the scored bank.
        windows: For each table index (as a string), the arrays of `series_figures.window_figure`.
        log: Receives progress lines.

    Returns:
        The paths written.
    """
    import e00730_figures as figures
    import e00730_series_figures as sf

    banks.guard_run(root=root, stage="series")
    written = []
    for private in (False, True):
        plan = sf.plan_windows(t=table, published=not private)  # ty: ignore[invalid-argument-type]
        folder = root / ("PRIVATE" if private else "figures") / "series"
        for rule, idx in plan.items():
            if idx is None or str(idx) not in windows:
                continue
            if private and table.source[idx] != "S1":  # ty: ignore[unresolved-attribute]
                continue
            chart = sf.window_figure(
                window=windows[str(idx)], title=f"{rule}: window picked by the fixed rule"
            )
            figures.save_svg(chart=chart, path=folder / f"{rule}.svg")
            written.append(folder / f"{rule}.svg")
    log(f"{len(written)} time series figures written")
    return written


# --------------------------------------------------------------------------- real-data drivers


def run_refit_729(
    *, root: Path, ctx: stages729.Context, arm: str, arch: str, seed: int, device: str
) -> dict:
    """Refit one of experiment 729's arms (A, D, D-sign) and store its predictions (real data)."""
    import e00729_arms as arms729
    import e00729_train as train729
    import e00730_train as train

    banks.guard_run(root=root, stage="refit729")
    name = job_name(arm=arm, arch=arch, seed=seed)
    record_path = fit_dir(root=root) / f"{name}.json"
    if record_path.exists():
        return banks.read_json(path=record_path)
    train729.limit_gpu_memory()
    spec = arms729.ARM_SPECS[arm]
    kwargs = {"spec": spec, "holidays": ctx.cal.holidays, "stats": ctx.stats}
    train_bank = banks729.load_bank(path=banks.bank_path(root=root, kind="train", seed=seed))
    val_bank = banks729.load_bank(path=banks.bank_path(root=root, kind="validation"))
    prep_train = arms729.prepare_segments(segments=train_bank.segments, **kwargs)
    prep_val = arms729.prepare_segments(segments=val_bank.segments, **kwargs)
    net, record = train.refit_729_arm(
        arm=arm, arch=arch, seed=seed, train=prep_train, val=prep_val, device=device,
        out_path=root / "networks" / f"{name}.pt",
    )  # fmt: skip
    sets = {"validation": val_bank.segments}
    for kind in ("test", "shifted"):
        sets[kind] = banks729.load_bank(path=banks.bank_path(root=root, kind=kind)).segments
    sets.update({f"real_{k}": v for k, v in real_sets(ctx=ctx).items()})
    for kind, segments in sets.items():
        prep = arms729.prepare_segments(segments=segments, **kwargs)
        arrays = train729.predictions(net=net, prepared=prep, spec=spec, device=device)
        train729.save_predictions(arrays=arrays, path=root / "predictions" / name / f"{kind}.npz")
    banks.write_json(path=record_path, payload=record)
    return record


def measure_seconds_per_epoch(
    *, root: Path, ctx: stages729.Context, arch: str, device: str, epochs: int = core.TIMING_EPOCHS
) -> float:
    """Return the measured seconds an epoch of arm S on one architecture (nothing is saved)."""
    import e00730_train as train

    spec = spec_for(arm="S", style="K")
    kwargs = {"holidays": ctx.cal.holidays, "stats": ctx.stats}
    tr = banks729.load_bank(path=banks.bank_path(root=root, kind="train", seed=0))
    va = banks729.load_bank(path=banks.bank_path(root=root, kind="validation"))
    _, record = train.fit(
        spec=spec, arch=arch, seed=0, device=device, max_epochs=epochs,
        train=arms.prepare_segments(segments=tr.segments, spec=spec, **kwargs),
        val=arms.prepare_segments(segments=va.segments, spec=spec, **kwargs),
    )  # fmt: skip
    return record.seconds_per_epoch


def encoder_inputs(*, root: Path) -> tuple[arms.EncoderTable, Path]:
    """Return D8's encoder input table and the pre-trained weight folder of this experiment."""
    import e00730_pretrain as pretrain

    folder = root / "pretrained"
    dates, values = pretrain.read_table(path=folder / pretrain.D8_TABLE_FILE)
    return arms.EncoderTable(dates=dates, values=values), folder


def fit_one(
    *, root: Path, ctx: stages729.Context, arm: str, arch: str, seed: int, device: str, log: Log
) -> dict:
    """Fit one new arm; the style of SD* and the encoders come from files under `root`."""
    import e00730_pretrain as pretrain

    style_file = root / "style.json"
    style = banks.read_json(path=style_file)["chosen"] if style_file.exists() else "K"
    encoder = pretrained = None
    if arm.startswith("SDe"):
        encoder, folder = encoder_inputs(root=root)
        pretrained = pretrain.weight_path(folder=folder, seed=seed)
    return run_fit(
        root=root, ctx=ctx, arm=arm, arch=arch, seed=seed, style=style, device=device, log=log,
        encoder=encoder, pretrained=pretrained,
    )  # fmt: skip


# --------------------------------------------------------------------------- real rows and windows


def real_baselines(*, ctx: stages729.Context) -> tuple[object, object]:
    """Return experiment 729's climatology and regression, fitted on the training base days."""
    return stages729.fit_baselines(ctx=ctx)


def real_inputs(*, ctx: stages729.Context) -> tuple[dict, np.ndarray]:
    """Return the rows of the real sets and the era of each row of the `eras` set (real data)."""
    sets = real_sets(ctx=ctx)
    rows = {
        k: score729.rows_from_segments(segments=v, holidays=ctx.cal.holidays)
        for k, v in sets.items()
    }
    cal_dates = ctx.table.calendar.dates
    day = np.datetime64("1970-01-01", "D") + rows["eras"].date.astype("timedelta64[D]")
    return rows, np.asarray(ctx.table.era)[np.searchsorted(cal_dates, day)]


def _scatter(*, keys: np.ndarray, hh: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Scatter per-row values onto an `(n windows, 48)` array of NaN."""
    out = np.full((n, 48), np.nan)
    out[keys, hh] = values
    return out


def series_inputs(
    *, root: Path, holidays: np.ndarray, real_rows: Mapping[str, score729.Rows] | None
) -> tuple[object, dict[str, dict[str, np.ndarray]]]:
    """Build the per-window table and the window arrays of the time series figures.

    One window is one scored day of the test bank (and, with `real_rows`, one real held-out
    pre-engine day). The arrays carry no date.

    Args:
        root: The data folder.
        holidays: Bank-holiday dates.
        real_rows: The rows of the real sets, or None to leave the real windows out.

    Returns:
        The `WindowTable` and, for each window index as a string, the arrays of one figure.
    """
    import e00730_series_figures as sf

    arch = chosen_architecture(root=root)
    style = banks.read_json(path=root / "style.json")["chosen"]
    test = banks729.load_bank(path=banks.bank_path(root=root, kind="test"))
    rows = score729.rows_from_segments(segments=test.segments, holidays=holidays)
    seeds = core.ARCH_SEEDS

    def mean_of(arm: str, key: str, name: str = "test") -> np.ndarray:
        zs = [
            np.load(root / "predictions" / job_name(arm=arm, arch=arch, seed=s) / f"{name}.npz")[
                key
            ]
            for s in seeds
        ]
        return np.mean(zs, axis=0)

    p_s = load_p(root=root, arm="S", arch=arch, seeds=seeds, name="test", x=rows.x)
    p_sd = load_p(root=root, arm=f"SD-{style}", arch=arch, seeds=seeds, name="test", x=rows.x)
    mu = mean_of(f"SD-{style}", "demand_mu")
    p_c = mean_of("SD*-C", "p", "test_causal_k0")
    keys_all = rows.seg.astype(np.int64) * 1_000_000 + rows.date.astype(np.int64)
    uniq, key = np.unique(keys_all, return_inverse=True)
    n = len(uniq)
    base = np.full((n, 48), np.nan)
    for w, k in enumerate(uniq):
        seg = test.segments[int(k // 1_000_000)]
        day = np.datetime64("1970-01-01", "D") + np.timedelta64(int(k % 1_000_000), "D")
        base[w] = seg.base[int(np.searchsorted(seg.dates, day))]
    sc = lambda v: _scatter(keys=key, hh=rows.hh, values=v, n=n)  # noqa: E731
    truth, reading, engine, solar = sc(rows.x), sc(rows.r), sc(rows.engine), sc(rows.solar)
    a_s, a_sd, a_c, a_mu = sc(p_s), sc(p_sd), sc(p_c), sc(mu)
    decidable = sc((rows.r >= core.DECIDABLE_READING_POINTS).astype(float)) > 0
    source = np.array(["none", "S1", "S2"])[
        np.bincount(key, weights=rows.source, minlength=n).astype(int) // 48 % 3
    ]
    worlds = np.array([test.segments[int(k // 1_000_000)].world for k in uniq])
    table = sf.WindowTable(
        world=worlds,
        solar_run=np.nan_to_num(engine).max(axis=1) > 0,
        solar_out=np.nan_to_num(solar).max(axis=1) > 0,
        source=source,
        depth=np.nanmax(-truth, axis=1),
        near_zero=(np.abs(reading) < core.DECIDABLE_READING_POINTS * 0.3).sum(axis=1),
        evening=np.nan_to_num(engine)[:, 28:].max(axis=1)
        > np.nan_to_num(engine)[:, :28].max(axis=1),
        sd_max_p=np.nanmax(a_sd, axis=1),
        sd_mean_p=np.nan_to_num(np.nanmean(np.where(decidable, a_sd, np.nan), axis=1), nan=-1.0),
        gain=np.nanmean((a_s - (truth < 0)) ** 2 - (a_sd - (truth < 0)) ** 2, axis=1),
        real=np.zeros(n, dtype=bool),
    )
    windows = {
        str(i): {
            "truth": truth[i], "reading": reading[i], "demand": base[i], "demand_mean": a_mu[i],
            "p_S": a_s[i], "p_SD*": a_sd[i], "p_SD*-C": a_c[i],
        }
        for i in range(n)
    }  # fmt: skip
    if real_rows is not None:
        table, windows = add_real_windows(
            table=table, windows=windows, rows=real_rows["r1"], root=root, arch=arch, style=style
        )
    return table, windows


def add_real_windows(
    *,
    table: object,
    windows: dict[str, dict[str, np.ndarray]],
    rows: score729.Rows,
    root: Path,
    arch: str,
    style: str,
) -> tuple[object, dict[str, dict[str, np.ndarray]]]:
    """Append one window per real held-out pre-engine day (the rows of set `r1`) to the table.

    The real windows carry the reading, SD*'s and S's P(X<0), and SD*'s demand output. They have
    no clean demand and no causal probability, so those arrays are NaN.

    Args:
        table: The `WindowTable` of the synthetic windows.
        windows: The synthetic windows' arrays by index.
        rows: The rows of the real set `r1`.
        root: The data folder.
        arch: The chosen architecture.
        style: The chosen masking style.

    Returns:
        The extended table and windows.
    """
    import dataclasses

    import e00730_series_figures as sf

    seeds = core.ARCH_SEEDS
    p_s = load_p(root=root, arm="S", arch=arch, seeds=seeds, name="real_r1", x=rows.x)
    arm = f"SD-{style}"
    p_sd = load_p(root=root, arm=arm, arch=arch, seeds=seeds, name="real_r1", x=rows.x)
    preds = load_preds(root=root, arm=arm, arch=arch, seeds=seeds, name="real_r1")
    mu = np.mean([z["demand_mu"] for z in preds], axis=0)
    keys = rows.seg.astype(np.int64) * 1_000_000 + rows.date.astype(np.int64)
    _, key = np.unique(keys, return_inverse=True)
    n = int(key.max()) + 1 if key.size else 0
    sc = lambda v: _scatter(keys=key, hh=rows.hh, values=v, n=n)  # noqa: E731
    reading, a_s, a_sd, a_mu = sc(rows.r), sc(p_s), sc(p_sd), sc(mu)
    truth = sc(rows.x)
    dec = sc((rows.r >= core.DECIDABLE_READING_POINTS).astype(float)) > 0
    near_zero_points = core.DECIDABLE_READING_POINTS * 0.3
    extra = sf.WindowTable(
        world=np.full(n, "real"), solar_run=np.zeros(n, dtype=bool),
        solar_out=np.zeros(n, dtype=bool), source=np.full(n, "none"),
        depth=np.nanmax(-truth, axis=1),
        near_zero=(np.abs(np.nan_to_num(reading, nan=1e9)) < near_zero_points).sum(axis=1),
        evening=np.zeros(n, dtype=bool), sd_max_p=np.nanmax(a_sd, axis=1),
        sd_mean_p=np.nan_to_num(np.nanmean(np.where(dec, a_sd, np.nan), axis=1), nan=-1.0),
        gain=np.zeros(n), real=np.ones(n, dtype=bool),
    )  # fmt: skip
    merged = sf.WindowTable(
        **{
            f.name: np.concatenate([getattr(table, f.name), getattr(extra, f.name)])
            for f in dataclasses.fields(extra)
        }
    )
    offset = len(table.world)  # ty: ignore[unresolved-attribute]
    nan = np.full(48, np.nan)
    out = dict(windows)
    for i in range(n):
        out[str(offset + i)] = {
            "truth": truth[i], "reading": reading[i], "demand": nan, "demand_mean": a_mu[i],
            "p_S": a_s[i], "p_SD*": a_sd[i], "p_SD*-C": nan,
        }  # fmt: skip
    return merged, out

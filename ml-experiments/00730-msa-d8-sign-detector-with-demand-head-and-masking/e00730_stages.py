"""Experiment 730: the stages, with the data root passed to every function.

A stage runs `banks.guard_run` first, so a stage on the outputs of another run, before the gate, or
before the gallery check, refuses to start. The stages that read real data (`real_context`,
`real_gate_inputs`, `real_sets`) are thin wrappers over experiment 729's loaders and need the real
files; every other function here takes arrays or files under `root`. Stages print counts and scores
only.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import e00729_banks as banks729
import e00729_stages as stages729
import e00730_arms as arms
import e00730_banks as banks
import e00730_core as core
import e00730_leakage as leak
import e00730_score as score
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
    rows: banks729.score729.Rows,
    p: dict[str, np.ndarray],
    per_row: dict[str, np.ndarray],
    n_resamples: int = core.N_RESAMPLES,
    stopped: bool = False,
    invalid: bool = False,
    failed: bool = False,
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
        h2_pass=True,
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
        "h2_pending": True,
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
    for kind in ("validation", "test", "shifted"):
        bank = (
            val_bank
            if kind == "validation"
            else banks729.load_bank(path=banks.bank_path(root=root, kind=kind))
        )
        prep = arms.prepare_segments(segments=bank.segments, spec=spec, **kwargs)
        decoded, targets, _ = train.predict(net=net, prepared=prep, spec=spec, device=device)
        flat = {k: train.valid_flat(values=v, targets=targets) for k, v in decoded.items()}
        train729.save_predictions(arrays=flat, path=root / "predictions" / name / f"{kind}.npz")
    banks.write_json(path=record_path, payload=record.as_dict())
    return record.as_dict()

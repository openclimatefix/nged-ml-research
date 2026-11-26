# ruff: noqa: E402, PLR0912, PLR0915, E501
"""Experiment 732: the command line. Run it through `run.sh`, never directly.

    SLOT_CPUS=<list> run.sh --go e00732_cli.py <stage> [options]

Stages, in the order of the plan: `banks`, `neighbours`, `limits`, `overlays`, `gallery` (then `panels`
once the session has read the gallery), `tensor`, `timing`, `fit`, `chain`, `real`, `score`, `report`,
`figures`, `loss_curves`, and `series`. Every stage prints counts and scores only, and every stage but
`banks` runs its guard first. A stop prints a verdict-row-0 line and exits with status 3, and a
correctness failure writes `scores/invalid.json` once and exits with status 4.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import e00732_paths as paths

paths.pin_threads()

import e00729_stages as stages729
import e00730_banks as banks730
import e00732_core as core
import e00732_leakage as leak
import e00732_neighbours as nb
import e00732_stages as stages
import numpy as np

CPU_ONLY_STAGES = (
    "banks", "neighbours", "limits", "overlays", "tensor", "gallery", "panels", "real", "score",
    "report", "figures", "loss_curves", "series",
)  # fmt: skip


def log(line: str) -> None:
    """Print one progress line with a UTC time stamp."""
    print(f"{datetime.now(UTC):%H:%M:%S} {line}", flush=True)


def real_context() -> stages729.Context:
    """Build experiment 729's context (days, split, masks) from the real files. Needs real data."""
    return stages729.build_context()


def real_base(*, ctx: stages729.Context) -> stages.BaseDays:
    """Return D8's base calendar as the neighbour, limits, and gallery stages read it."""
    cal = ctx.cal
    return stages.BaseDays(
        dates=cal.dates, reading=cal.reading, bsp=cal.bsp, holidays=cal.holidays,
        split=np.asarray(ctx.day_split),
    )  # fmt: skip


def build_neighbours(*, base: stages.BaseDays) -> dict[str, np.ndarray]:
    """Read D3, D5, and D14 from the stored power files onto D8's base calendar."""
    import e00722_guard as guard

    return nb.build_arrays(dates=base.dates, log=guard.ReadLog())


def real_env(
    *, root: Path, ctx: stages729.Context, base: stages.BaseDays, with_real: bool
) -> stages.Env:
    """Return the stages' environment from the real files (the neighbour table and medians must exist).

    Args:
        root: The data folder.
        ctx: The context.
        base: D8's base calendar.
        with_real: Whether to add the real-day sets (the fits need them; the gallery does not).

    Returns:
        The environment. The neighbours of the real-day sets sit on D8's whole day table, which
        the stage builds from the same files (not from the base calendar).
    """
    arrays, medians = stages.read_neighbour_inputs(root=root, base=base)
    if not with_real:
        return stages.Env(
            holidays=ctx.cal.holidays, stats=ctx.stats, neighbours=arrays, medians=medians
        )
    import e00722_guard as guard

    table_dates = ctx.table.calendar.dates
    real_neighbours = nb.build_arrays(
        dates=table_dates, log=guard.ReadLog(), name="neighbours_real_days"
    )
    return stages.Env(
        holidays=ctx.cal.holidays, stats=ctx.stats, neighbours=arrays, medians=medians,
        real_sets=lambda: stages729.real_sets(ctx=ctx), real_neighbours=real_neighbours,
    )  # fmt: skip


def main(argv: list[str]) -> int:
    """Run one stage and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(core.stage_names()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--confirm-panels", help="JSON file of the session's per-panel reading")
    parser.add_argument("--manager-approved", action="store_true")
    parser.add_argument("--cores", default="?", help="The cores of the slot, for the printed lines")
    args = parser.parse_args(argv)
    root = paths.data_root()
    root.mkdir(parents=True, exist_ok=True)
    try:
        return dispatch(args=args, root=root, root730=paths.root_730())
    except stages.StoppedError as exc:
        log(f"STOPPED (verdict row 0): {exc}")
        return 3
    except leak.CorrectnessFailureError as exc:
        stages.record_invalid(root=root, rule=str(exc))
        log(f"A CORRECTNESS CHECK FAILED (verdict row 1): {exc}")
        return 4


def dispatch(*, args: argparse.Namespace, root: Path, root730: Path) -> int:
    """Run the stage that `args` names."""
    stage = args.stage
    if stage == "panels":
        import e00732_gallery as gallery

        confirms = json.loads(Path(args.confirm_panels).read_text())
        record = gallery.record_panel_check(root=root, root730=root730, confirms=confirms)
        log(f"panel check recorded: all passed {record['all_passed']}")
        return 0
    if stage == "loss_curves":
        import e00732_figures as figures

        figures.stage_loss_curves(root=root, log=log)
        return 0
    ctx = real_context()
    base = real_base(ctx=ctx)
    holidays = ctx.cal.holidays
    if stage == "banks":
        stages.stage_banks(root=root, root730=root730, holidays=holidays, log=log)
    elif stage == "neighbours":
        stages.stage_neighbours(
            root=root,
            root730=root730,
            base=base,
            build=lambda: build_neighbours(base=base),
            log=log,
        )
    elif stage == "limits":
        arrays, _ = stages.read_neighbour_inputs(root=root, base=base)
        stages.stage_limits(
            root=root, root730=root730, holidays=holidays, base=base, arrays=arrays, log=log
        )
    elif stage == "overlays":
        arrays, _ = stages.read_neighbour_inputs(root=root, base=base)
        stages.stage_overlays(
            root=root, root730=root730, holidays=holidays, neighbours=arrays, log=log
        )
    elif stage == "tensor":
        stages.stage_tensor_checks(
            root=root,
            root730=root730,
            env=real_env(root=root, ctx=ctx, base=base, with_real=False),
            log=log,
        )
    elif stage == "gallery":
        import e00732_gallery as gallery

        results = gallery.stage_gallery(
            root=root,
            root730=root730,
            env=real_env(root=root, ctx=ctx, base=base, with_real=False),
            base=base,
            log=log,
        )
        log(f"gallery: {sum(results.values())} of {len(results)} panels pass their automatic check")
    elif stage in ("timing", "fit", "chain"):
        import e00732_fit as fit

        env = real_env(root=root, ctx=ctx, base=base, with_real=True)
        if stage == "timing":
            fit.stage_timing(
                root=root, root730=root730, env=env, device=args.device,
                manager_approved=args.manager_approved, log=log, cores=args.cores,
            )  # fmt: skip
        else:

            def run_one(arm: str, arch: str, seed: int) -> dict:
                del arch
                return fit.fit_one(
                    root=root,
                    root730=root730,
                    env=env,
                    arm=arm,
                    seed=seed,
                    device=args.device,
                    log=log,
                )

            if stage == "fit":
                run_one(args.arm, core.ARMS[args.arm].arch, args.seed)
            else:
                status = fit.stage_chain(
                    root=root, root730=root730, run_one=run_one, log=log, cores=args.cores
                )
                log(f"chain: {status['state']}")
    else:
        import e00732_report as report

        env = real_env(root=root, ctx=ctx, base=base, with_real=False)
        getattr(report, f"stage_{stage}")(
            root=root, root730=root730, env=env, base=base, ctx=ctx, log=log
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

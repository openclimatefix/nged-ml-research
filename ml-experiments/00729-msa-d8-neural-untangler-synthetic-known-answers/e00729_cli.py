# ruff: noqa: E402, E501, PLR0912, PLR0915
"""Experiment 729: the command line. Run it through `run.sh`, never directly.

    SLOT_CPU=<n> run.sh --go e00729_cli.py <stage> [options]

Stages, in the order of the plan: `banks`, `gate`, `timing`, `screen`, `choose`, `ladder`,
`margin`, `score`, `report`, `figures`, `series`, `events`. `fit` runs one named fit.
Every stage prints counts and scores only.
"""

import argparse
import sys
import time
from datetime import UTC, datetime

import e00729_paths as paths

paths.pin_threads()

import e00729_arms as arms
import e00729_banks as banks
import e00729_core as core
import e00729_report as report
import e00729_stages as stages


def log(line: str) -> None:
    """Print one progress line with a UTC time stamp."""
    print(f"{datetime.now(UTC):%H:%M:%S} {line}", flush=True)


def run_jobs(*, ctx: stages.Context, jobs: list[tuple[str, str, int]], device: str) -> None:
    """Run fits one at a time (one GPU process)."""
    for arm, arch, seed in jobs:
        started = time.perf_counter()
        rec = stages.run_fit_or_record_failure(
            ctx=ctx, arm=arm, arch=arch, seed=seed, device=device, log=log
        )
        if rec is None:
            continue
        log(
            f"fit {arm} {arch} seed {seed}: validation CRPS {rec['best_val_crps']:.4f}, {time.perf_counter() - started:.0f} s"
        )


def prepared_for(
    *, ctx: stages.Context, bank: banks.Bank, spec: arms.ArmSpec
) -> list[arms.PreparedSegment]:
    """Return the prepared segments of a bank for an arm."""
    return arms.prepare_segments(
        segments=bank.segments, spec=spec, holidays=ctx.cal.holidays, stats=ctx.stats
    )


def main(argv: list[str]) -> int:
    """Run one stage and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "banks",
            "gate",
            "timing",
            "screen",
            "choose",
            "ladder",
            "margin",
            "score",
            "report",
            "figures",
            "series",
            "events",
            "fit",
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", choices=core.ARMS)
    parser.add_argument("--arch", choices=core.ARCHITECTURES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resamples", type=int, default=core.N_RESAMPLES)
    parser.add_argument(
        "--approve-over-budget",
        action="store_true",
        help="the manager approved a projection above the budget",
    )
    args = parser.parse_args(argv)
    if args.stage == "report":
        root = paths.data_root()
        verdict = stages.read_json(path=root / "verdict.json")
        tables = stages.read_json(path=root / "scores" / "tables.json")
        gate = stages.read_json(path=root / "scores" / "gate.json")
        sel = stages.read_json(path=root / "selection.json")
        timing_path = root / "scores" / "timing.json"
        timing = stages.read_json(path=timing_path) if timing_path.exists() else None
        out = report.report_paths()
        out["report"].parent.mkdir(parents=True, exist_ok=True)
        out["report"].write_text(
            report.render_report(verdict=verdict, tables=tables, gate=gate, timing=timing)
        )
        for p in report.write_figures(
            verdict=verdict, tables=tables, gate=gate, selection=sel, out_dir=out["figures"]
        ):
            log(f"wrote a figure: {report.figures.optimise_command(path=p)}")
        return 0
    if args.stage == "margin":
        log(f"margin written: {stages.stage_margin()['margin_points']:.3f} points")
        return 0
    if args.stage == "choose":
        log(f"architecture chosen: {stages.stage_choose_architecture()['chosen']}")
        return 0
    if args.stage == "events":
        for p in stages.stage_events(log=log):
            log(f"wrote a figure: {report.figures.optimise_command(path=p)}")
        return 0
    ctx = stages.build_context()
    if args.stage == "banks":
        summary = stages.stage_banks(ctx=ctx, train_seeds=core.MARGIN_SEEDS, log=log)
        log(f"banks built; reachability {summary['reachability']}")
    elif args.stage == "gate":
        log(f"gate pass: {stages.stage_gate(ctx=ctx, log=log)['pass']}")
    elif args.stage == "timing":
        import e00729_train as train

        spec = arms.ARM_SPECS["D"]
        tr = banks.load_bank(path=stages.bank_path(kind="train", seed=0))
        va = banks.load_bank(path=stages.bank_path(kind="validation"))
        result = train.timing_gate(
            train=prepared_for(ctx=ctx, bank=tr, spec=spec),
            val=prepared_for(ctx=ctx, bank=va, spec=spec),
            device=args.device,
            log=log,
        )
        stages.write_json(path=paths.data_root() / "scores" / "timing.json", payload=result)
        log(f"timing gate: {result}")
        if not result["within_budget"] and not args.approve_over_budget:
            log("the projection exceeds the budget: send it to the manager before any fit")
            return 2
    elif args.stage == "screen":
        import e00729_train as train

        run_jobs(ctx=ctx, jobs=train.screen_jobs(), device=args.device)
    elif args.stage == "ladder":
        import e00729_train as train

        arch = stages.read_json(path=paths.data_root() / "selection.json")["chosen"]
        run_jobs(ctx=ctx, jobs=train.ladder_jobs(arch=arch), device=args.device)
    elif args.stage == "figures":
        arch = stages.read_json(path=paths.data_root() / "selection.json")["chosen"]
        for p in stages.stage_figures(ctx=ctx, arch=arch, log=log):
            log(f"wrote a figure: {report.figures.optimise_command(path=p)}")
    elif args.stage == "series":
        arch = stages.read_json(path=paths.data_root() / "selection.json")["chosen"]
        for p in stages.stage_series(ctx=ctx, arch=arch, log=log):
            log(f"wrote a figure: {report.figures.optimise_command(path=p)}")
    elif args.stage == "fit":
        run_jobs(ctx=ctx, jobs=[(args.arm, args.arch, args.seed)], device=args.device)
    elif args.stage == "score":
        record = stages.stage_score(ctx=ctx, n_resamples=args.resamples, log=log)
        log(f"verdict row {record['row']}: {record['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

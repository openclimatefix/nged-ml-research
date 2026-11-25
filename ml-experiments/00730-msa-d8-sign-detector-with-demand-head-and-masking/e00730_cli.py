# ruff: noqa: E402
"""Experiment 730: the command line. Run it through `run.sh`, never directly.

    SLOT_CPUS=<list> run.sh --go e00730_cli.py <stage> [options]

Stages, in the order of the plan: `banks`, `gallery`, `gate`, `pretrain`, `timing`, `screen`,
`choose`, `margin`, `styles`, `later`, `refit729`, `fit`, `score`, `real`, `report`, `figures`,
`series`, and `components`. Every stage prints counts and scores only.
"""

import argparse
import sys
from datetime import UTC, datetime

import e00730_paths as paths

paths.pin_threads()

import e00730_banks as banks
import e00730_core as core
import e00730_stages as stages


def log(line: str) -> None:
    """Print one progress line with a UTC time stamp."""
    print(f"{datetime.now(UTC):%H:%M:%S} {line}", flush=True)


def main(argv: list[str]) -> int:
    """Run one stage and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=stages.stage_names())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm")
    parser.add_argument("--arch", choices=core.ARCHITECTURES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--style", choices=core.STYLES, default="K")
    args = parser.parse_args(argv)
    root = paths.data_root()
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "banks":
            banks.guard_run(root=root, stage="banks")
            ctx = stages.real_context()
            banks.stage_banks(
                root=root, inputs=stages.bank_inputs(ctx=ctx), train_seeds=range(5), log=log
            )
        elif args.stage == "gate":
            ctx = stages.real_context()
            banks.stage_gate(root=root, gate_inputs=stages.real_gate_inputs(ctx=ctx), log=log)
        elif args.stage == "choose":
            log(str(stages.choose_architecture(root=root)))
        elif args.stage == "margin":
            log(str(stages.write_margin(root=root)))
        elif args.stage == "fit":
            ctx = stages.real_context()
            rec = stages.run_fit(
                root=root, ctx=ctx, arm=args.arm, arch=args.arch, seed=args.seed,
                style=args.style, device=args.device, log=log,
            )  # fmt: skip
            log(f"fit {args.arm}: validation Brier {rec['best_val_brier']:.5f}")
        else:
            banks.guard_run(root=root, stage=args.stage)
            log(f"stage `{args.stage}` passed its guards; its driver is not written yet")
            return 2
    except banks.StoppedError as exc:
        log(f"STOPPED (verdict row 0): {exc}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

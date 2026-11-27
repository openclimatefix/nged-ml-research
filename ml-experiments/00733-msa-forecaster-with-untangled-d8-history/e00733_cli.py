# ruff: noqa: TRY300
"""Experiment 733: the command line, one stage per sub-command.

(torch is imported by the stages that fit).

Run it through `run.sh --go e00733_cli.py <stage> ...`. Every stage runs `guard_run` first. The
exit code is 0
for success, 2 for a refused guard, 3 for a stop (verdict row 0), 4 for a correctness failure
(verdict row 1),
and 5 for any failed fit or chain; never 1, which the launcher reserves for "slot busy". A failure
leaves
`scores/failure_<stage>.json` with the message and the traceback.

Stages built: `guard`, `banks`, `channels`, `limits`, `gallery`, `smoke`, `timing`, `fit`, `chain`,
`predict`, `score`, `report`, and `loss_curves`. `main` installs the stage functions of
`e00733_runtime` into `STAGE_FUNCTIONS`, never at import. The stages not built yet (`figures`,
`attention`, `series`) have no function, and such a stage exits 2 with a message, never silently.
"""

import argparse
import json
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import e00733_paths as paths
import e00733_stages as st

STAGES: Final = (
    "guard",
    "banks",
    "channels",
    "limits",
    "gallery",
    "smoke",
    "timing",
    "fit",
    "chain",
    "predict",
    "score",
    "report",
    "figures",
    "loss_curves",
    "attention",
    "series",
)
STAGE_FUNCTIONS: dict[str, Callable[[argparse.Namespace, Path], Any]] = {}
"""Functions of the stages that need the real-file adapters, by stage name (filled by the
real-environment run)."""


def write_failure(*, root: Path, stage: str, exc: BaseException) -> None:
    """Write `scores/failure_<stage>.json` with the exception's type, message, and traceback."""
    st.write_json(
        path=root / "scores" / f"failure_{stage}.json",
        payload={"stage": stage, **st.fit.failure_payload(exc=exc)},
    )


def banks_digest(*, root730: Path) -> str:
    """Return the digest id of experiment 730's banks."""
    import e00730_banks as banks730  # ty: ignore[unresolved-import]

    return banks730.banks_digest_id(root=root730)


def run_stage(*, args: argparse.Namespace, root: Path, digest: str) -> int:
    """Run one stage under its guard and return its exit code."""
    try:
        st.guard_run(
            root=root, stage="chain" if args.stage == "guard" else args.stage, banks_digest=digest
        ) if args.stage != "banks" else st.guard_run(root=root, stage="banks", banks_digest=digest)
        if args.stage == "guard":
            print("guard passed")
            return st.EXIT_OK
        if (
            args.stage == "score"
            and not getattr(args, "validation_only", False)
            and not st.marker_exists(root=root)
        ):
            st.write_test_marker(root=root)
        fn = STAGE_FUNCTIONS.get(args.stage) or BUILTIN.get(args.stage)
        if fn is None:
            print(f"stage {args.stage} has no function in this build", file=sys.stderr)
            return st.EXIT_GUARD
        result = fn(args, root)
        if isinstance(result, dict) and "state" in result:
            return st.exit_code(status=result)
        return st.EXIT_OK
    except BaseException as exc:
        if isinstance(exc, st.CorrectnessFailureError):
            st.record_invalid(root=root, rule=str(exc))
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise
        write_failure(root=root, stage=args.stage, exc=exc)
        traceback.print_exc()
        return st.exit_code(exc=exc)


def install_runtime() -> None:
    """Put the stage functions of `e00733_runtime` into `STAGE_FUNCTIONS`."""
    import e00733_runtime as rt
    import e00733_scorestage as scorestage
    import e00733_smokerun as smokerun

    STAGE_FUNCTIONS.update(
        {
            "banks": rt.stage_banks, "channels": rt.stage_channels, "limits": rt.stage_limits,
            "gallery": rt.stage_gallery, "fit": rt.stage_fit, "chain": rt.stage_chain,
            "predict": rt.stage_predict, "score": scorestage.stage_score,
            "smoke": smokerun.stage_smoke,
        }
    )  # fmt: skip


def _timing(args: argparse.Namespace, root: Path) -> dict:
    if args.measured is None:
        import e00733_runtime as rt

        measured = rt.measure_timing(args, root)
    else:
        measured = json.loads(Path(args.measured).read_text())
    choice_path = paths.channels_json(root=root)
    same = (
        bool(json.loads(choice_path.read_text()).get("fu_equals_fs"))
        if choice_path.exists()
        else False
    )
    decision = st.timing_decision(fu_equals_fs=same, **measured)
    st.write_json(path=root / st.TIMING_FILE, payload=decision)
    print(json.dumps(decision, indent=1))
    return decision


def _report(args: argparse.Namespace, root: Path) -> Path:
    import e00733_report as rep

    return rep.write_report(root=root, counted_read=not args.interim)


def _loss_curves(args: argparse.Namespace, root: Path) -> Path:
    import e00733_figures as figs

    return figs.write_loss_curves(root=root, out=root / "figures" / "loss_curves.svg")


BUILTIN: Final[dict[str, Callable[[argparse.Namespace, Path], Any]]] = {
    "timing": _timing,
    "report": _report,
    "loss_curves": _loss_curves,
}


def parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    p = argparse.ArgumentParser(prog="e00733_cli.py")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument(
        "--measured", default=None, help="JSON of the timing gate's measurements (stage timing)"
    )
    p.add_argument(
        "--interim", action="store_true", help="label every number interim (stage report)"
    )
    p.add_argument(
        "--limit-segments", type=int, default=None, help="scratch root only: segments per bank"
    )
    p.add_argument(
        "--max-passes", type=int, default=None, help="scratch root only: cap every fit's passes"
    )
    p.add_argument(
        "--skip-gates", action="store_true", help="scratch root only: no F0r gate, no screen"
    )
    p.add_argument(
        "--first-job-per-family",
        action="store_true",
        help="scratch root only: the chain fits one job of each family",
    )
    p.add_argument(
        "--rows",
        default="test",
        choices=("test", "validation"),
        help="stage predict: the rows predicted (validation rows are held out of the fits)",
    )
    p.add_argument(
        "--validation-only",
        action="store_true",
        help="scratch root only: score validation rows, F0r in place of M*, no marker",
    )
    p.add_argument(
        "--scratch", default=None, help="stage smoke: a scratch folder outside the data folder"
    )
    p.add_argument("--arm", default=None)
    p.add_argument("--fold", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--data-root",
        default=None,
        help="a scratch data root (default: DATA_PATH_INTERNAL's folder of this experiment)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments, pin the threads, and run the stage."""
    args = parser().parse_args(argv)
    paths.pin_threads()
    install_runtime()
    root = Path(args.data_root) if args.data_root else paths.data_root()
    return run_stage(args=args, root=root, digest=banks_digest(root730=paths.root_730()))


if __name__ == "__main__":
    sys.exit(main())

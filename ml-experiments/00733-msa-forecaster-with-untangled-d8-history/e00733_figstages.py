# ruff: noqa: E501
"""Experiment 733: the stage functions `figures`, `loss_curves`, `series`, and `attention`.

Each stage reads stored files only, writes its SVG files to `figures/`, optimises them with `svgo`,
checks that each file is under 500 KB, and writes a summary to `scores/<stage>_stage.json`. The
`--validation-only` flag (a scratch root) reads the validation rows, and the official run reads the
test rows.
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any, Final

import e00733_stages as st

MAX_SVG_BYTES: Final = 500_000
SVGO_ARGS: Final = ("svgo@4", "--multipass", "--precision=1", "--final-newline")


class FigureStageError(RuntimeError):
    """A figure is missing, too large, or could not be optimised."""


def rows_of(*, args: argparse.Namespace) -> str:
    """Return "validation" for a `--validation-only` run and "test" otherwise."""
    return "validation" if getattr(args, "validation_only", False) else "test"


def optimise(*, folder: Path, names: list[str]) -> dict[str, int]:
    """Run `svgo` on each file in place and return the sizes in bytes.

    Args:
        folder: The folder of the files.
        names: The SVG file names.

    Returns:
        The size of each file after optimisation.

    Raises:
        FigureStageError: When no file was written, `npx` is missing, `svgo` fails, or a file exceeds
            `MAX_SVG_BYTES`.
    """
    if not names:
        msg = "the stage wrote no figure"
        raise FigureStageError(msg)
    npx = shutil.which("npx")
    if npx is None:
        msg = "npx is not installed, so the figures cannot be optimised with svgo"
        raise FigureStageError(msg)
    sizes: dict[str, int] = {}
    for name in names:
        path = folder / name
        done = subprocess.run(
            [npx, "--no-install", *SVGO_ARGS, str(path)], capture_output=True, text=True, check=False, timeout=600
        )  # fmt: skip
        if done.returncode != 0 or not path.exists():
            msg = f"svgo failed on {name}: {(done.stdout + done.stderr)[-300:]}"
            raise FigureStageError(msg)
        sizes[name] = path.stat().st_size
        if sizes[name] > MAX_SVG_BYTES:
            msg = f"{name} is {sizes[name]} bytes after svgo, over the limit of {MAX_SVG_BYTES}"
            raise FigureStageError(msg)
    return sizes


def finish(*, root: Path, stage: str, result: dict[str, Any]) -> dict[str, Any]:
    """Optimise the files in `result["files"]`, record the sizes, and write the stage's summary."""
    result["sizes"] = optimise(folder=root / "figures", names=result["files"])
    st.write_json(path=root / "scores" / f"{stage}_stage.json", payload=result)
    return result


def stage_figures(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Draw the loss curves, the contrast dot charts, the coverage figure, and the reliability diagram."""
    import e00733_figures as figs

    return finish(
        root=root, stage="figures", result=figs.run_figures(root=root, rows=rows_of(args=args))
    )


def stage_loss_curves(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Draw the loss curves of every fitted arm (training and validation loss per pass)."""
    import e00733_figures as figs

    del args
    panels = figs.write_loss_grid(root=root, out=root / "figures" / "loss_curves.svg")
    return finish(
        root=root, stage="loss_curves", result={"files": ["loss_curves.svg"], "loss_panels": panels}
    )


def stage_series(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Draw the forecast time series of the real folds and the synthetic world."""
    import e00733_series as series

    return finish(
        root=root, stage="series", result=series.run_series(root=root, rows=rows_of(args=args))
    )


def stage_attention(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Draw the attention maps of the real cases and of the synthetic case (loads the fitted networks)."""
    import e00733_attention as attention

    result = attention.run_attention(root=root, rows=rows_of(args=args), device=args.device)
    return finish(root=root, stage="attention", result=result)

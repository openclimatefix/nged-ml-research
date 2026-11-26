# ruff: noqa: E501, PERF401
"""Experiment 733: attention maps of the forecaster (torch).

For D8's target tokens, the across-day attention (which of the 15 rows each half-hour of the target
row reads) and the within-row attention of the target row come from a forward pass of a saved state
with `return_attention=True`. Every tensor leaves the device through `e00733_train.to_numpy`. The
learned gain `gamma` of the history path is returned beside each map, as the plan prints it.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import e00733_figures as figs
import e00733_model as model
import e00733_paths as paths  # noqa: F401
import e00733_train as train
import numpy as np
import polars as pl

# isort: split
import altair as alt
import torch  # ty: ignore[unresolved-import]

TARGET_ROW: Final = 0
"""The index of the target row among the 15 rows of the across-day attention."""


class AttentionError(ValueError):
    """The attention weights are not in the shape the maps need."""


def extract(
    *, net: model.UntangledNet, inp: model.UntInputs, device: str, sample: int = 0
) -> dict[str, np.ndarray]:
    """Return the attention maps and the readout weights of one sample from a forward pass.

    Args:
        net: The fitted network.
        inp: A batch of inputs (on `device`).
        device: "cuda" or "cpu".
        sample: The batch row to extract.

    Returns:
        `across_<i>` `(48, 15)` (head mean), `within_<i>` `(48, 48)` (target row, head mean),
        `alpha`
        `(14, 48)` the similar-day read-out weights, and `gamma` `(64,)` (empty when the arm has no
        path).

    Raises:
        AttentionError: When the network returns no attention or a weight has the wrong shape.
    """
    net.eval()
    with torch.no_grad():
        out = net(inp.to(device), return_attention=True)
    if out.attention is None:
        msg = "the network returned no attention weights"
        raise AttentionError(msg)
    maps: dict[str, np.ndarray] = {}
    for name, weights in out.attention.items():
        w = train.to_numpy(tensor=weights)[sample]
        if name.startswith("across"):
            if w.ndim != 3 or w.shape[0] != 48:
                msg = f"the across-day weights {name} have shape {w.shape}, not (48, heads, rows)"
                raise AttentionError(msg)
            maps[name] = w.mean(axis=1)
        else:
            if w.ndim != 4:
                msg = (
                    f"the within-row weights {name} have shape {w.shape}, not (rows, heads, 48, 48)"
                )
                raise AttentionError(msg)
            maps[name] = w[TARGET_ROW].mean(axis=0)
    maps["alpha"] = train.to_numpy(tensor=out.alpha)[sample]
    gamma = getattr(net, "u_gamma", None)
    maps["gamma"] = train.to_numpy(tensor=gamma) if gamma is not None else np.zeros(0)
    return maps


def gamma_line(*, gamma: np.ndarray) -> str:
    """Return the line printed beside a map: the size of the learned gain of the history path.

    (no value of a series).
    """
    if gamma.size == 0:
        return "no untangled history path"
    return f"learned gain of the history path: mean absolute {float(np.abs(gamma).mean()):.3f}, largest {float(np.abs(gamma).max()):.3f}"


def heatmap(*, matrix: np.ndarray, x_title: str, y_title: str, title: str) -> alt.Chart:
    """Return a heatmap of an attention matrix (rows on the y axis).

    With one colour scale from 0 to the largest weight.
    """
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        msg = "an attention map must be a finite two-dimensional array"
        raise AttentionError(msg)
    rows, cols = matrix.shape
    data = pl.DataFrame(
        {
            "x": np.tile(np.arange(cols), rows),
            "y": np.repeat(np.arange(rows), cols),
            "weight": matrix.ravel(),
        }
    )
    return (
        alt.Chart(data)
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:O", title=x_title, axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("y:O", title=y_title, axis=alt.Axis(labelOverlap=True)),
            color=alt.Color(
                "weight:Q",
                scale=alt.Scale(scheme="blues", domain=[0, float(matrix.max())]),
                title="Attention weight",
            ),
        )
        .properties(
            width=figs.WIDTH_PX / 2 - 40, height=150, title=alt.TitleParams(title, anchor="start")
        )
    )


def attention_figure(*, maps: Mapping[str, np.ndarray], label: str) -> alt.VConcatChart:
    """Return one case's figure.

    The across-day and the within-row maps of each block and the read-out weights, with the gain
    line.
    """
    panels = []
    for name in sorted(k for k in maps if k.startswith("across")):
        panels.append(
            heatmap(
                matrix=maps[name].T,
                x_title="Half-hour of the target day",
                y_title="Row (0 target, 1 to 14 history)",
                title=f"{label}: across-day attention, block {name[-1]}",
            )
        )
    panels.append(
        heatmap(
            matrix=maps["alpha"],
            x_title="Half-hour of the target day",
            y_title="History row",
            title=f"{label}: read-out weights of the similar-day base",
        )
    )
    first_within = sorted(k for k in maps if k.startswith("within"))[:1]
    for name in first_within:
        panels.append(
            heatmap(
                matrix=maps[name],
                x_title="Half-hour read",
                y_title="Half-hour of the target row",
                title=f"{label}: within-row attention of the target row, block {name[-1]}",
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(gamma_line(gamma=maps["gamma"]), anchor="start", fontSize=11)
    )


def write_case(*, maps: Mapping[str, np.ndarray], label: str, path: Path) -> None:
    """Draw one case and write it as SVG (run `svgo` on the file before committing it)."""
    figs.save_svg(chart=attention_figure(maps=maps, label=label), path=path)

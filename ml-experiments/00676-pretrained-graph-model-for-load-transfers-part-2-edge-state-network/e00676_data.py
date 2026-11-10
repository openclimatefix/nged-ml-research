"""Labels and batches of experiment 676 from the cached worlds.

Link labels come from the generator's truth table: a link is "a donates" (1) or "b donates" (2)
over the donor's hours, the lag class is the recipient's start minus the donor's start in hours, and
all other time is "off" (0). The node label is 0 (no step), 1 (inside a downward event), or 2
(inside an upward event). Labels enter the loss only.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00675_inputs import node_inputs
from e00676_net import N_LAGS


def labels(
    *, truth: pl.DataFrame, sites: list[str], edges: list[tuple[int, int]], hours: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Link state, lag class (both (links, hours)), and node state ((nodes, hours))."""
    index = {s: i for i, s in enumerate(sites)}
    where = {(a, b): k for k, (a, b) in enumerate(edges)}
    link = np.zeros((len(edges), hours), np.int8)
    lag = np.zeros((len(edges), hours), np.int8)
    node = np.zeros((len(sites), hours), np.int8)
    rows = truth.to_dicts()
    for r in rows:
        h0, h1 = r["start"] // 2, hours if r["end"] < 0 else min(hours, r["end"] // 2 + 1)
        node[index[r["site"]], h0:h1] = 1 if r["amp_mw"] < 0 else 2
    by_group: dict[int, list[dict]] = {}
    for r in rows:
        if r["kind"] == "edge":
            by_group.setdefault(r["group"], []).append(r)
    for grp in by_group.values():
        for d in (r for r in grp if r["amp_mw"] < 0 and r["partners"]):
            h0 = d["start"] // 2
            h1 = hours if d["end"] < 0 else min(hours, d["end"] // 2 + 1)
            for p in d["partners"].split(","):
                rec = next((r for r in grp if r["site"] == p and r["amp_mw"] > 0), None)
                if rec is None or p not in index:
                    continue
                a, b = index[d["site"]], index[p]
                key, state = ((a, b), 1) if (a, b) in where else ((b, a), 2)
                if key not in where:
                    continue
                k = where[key]
                link[k, h0:h1] = state
                lag[k, h0:h1] = int(np.clip(round((rec["start"] - d["start"]) / 2), 0, N_LAGS - 1))
    return link, lag, node


def window_inputs(
    *, resid: np.ndarray, start: int, span: int, sigma: np.ndarray, mva: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Node inputs of one window of the hourly residual (no masking)."""
    x = torch.tensor(resid[:, start : start + span], dtype=torch.float64)
    return node_inputs(x=x, sigma=torch.tensor(sigma), is_mva=mva, hour0=start)

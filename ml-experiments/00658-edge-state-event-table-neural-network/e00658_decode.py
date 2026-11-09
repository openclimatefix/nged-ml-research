"""Turn the network's per-block outputs into an event table (Viterbi decode plus edge pairing)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00653_common import BLOCK, DAY, GRID_LENGTH, N_BLOCKS, Panel, block_means
from e00653_segment import PERMANENT_DAYS, Event, merge_fragments, refine
from e00658_data import World, graph_arrays
from e00658_net import EdgeStateNet

NONE, UP, DOWN, OTHER = 0, 1, 2, 3
CHUNK: Final[int] = 960
STRIDE: Final[int] = 480


@dataclass(frozen=True)
class DecodeCosts:
    """Fixed switching costs of the Viterbi decode and the pairing cut-off."""

    on: float = 4.0
    flip: float = 8.0
    edge_cutoff: float = 0.5
    none_boost: float = 1.0
    """Factor on the 'none' probability: the 0.1 weight on unlabelled time distorts the prior."""


@torch.no_grad()
def predict(
    *, net: EdgeStateNet, world: World, nbr: np.ndarray, edges: np.ndarray, device: torch.device
) -> dict[str, np.ndarray]:
    """Run the network over the whole grid in overlapping chunks.

    Args:
        net: The trained network.
        world: The world (features for every block).
        nbr: Neighbour matrix.
        edges: Edge index pairs.
        device: Torch device.

    Returns:
        Probabilities: `cls` (primaries, blocks, 4), `pos` (primaries, blocks), `edge`
        (edges, blocks, 3).
    """
    net.eval()
    seen = np.flatnonzero((world.feats[:, :, 1] > 0).any(axis=0))
    last_obs = int(seen[-1]) + 1 if len(seen) else N_BLOCKS
    x = torch.from_numpy(world.feats[:, :last_obs]).to(device)
    n_b = last_obs
    nb = torch.from_numpy(nbr).to(device)
    ed = torch.from_numpy(edges).to(device)
    cls = np.zeros((x.shape[0], n_b, 4), np.float32)
    pos = np.zeros((x.shape[0], n_b), np.float32)
    edge = np.zeros((len(edges), n_b, 3), np.float32)
    starts = list(range(0, max(1, n_b - CHUNK) + 1, STRIDE))
    if starts[-1] + CHUNK < n_b:
        starts.append(max(0, n_b - CHUNK))
    for k, s in enumerate(starts):
        out = net(x=x[None, :, s : s + CHUNK], nbr=nb, edges=ed)
        lo = 0 if k == 0 else STRIDE // 2
        hi = CHUNK if k == len(starts) - 1 else CHUNK - STRIDE // 2
        sl = slice(s + lo, s + hi)
        cls[:, sl] = torch.softmax(out.cls[0, :, lo:hi], dim=-1).cpu().numpy()
        pos[:, sl] = torch.sigmoid(out.sign[0, :, lo:hi]).cpu().numpy()
        edge[:, sl] = torch.softmax(out.edge[0, :, lo:hi], dim=-1).cpu().numpy()
    pad = N_BLOCKS - n_b
    cls = np.pad(cls, ((0, 0), (0, pad), (0, 0)))
    pos = np.pad(pos, ((0, 0), (0, pad)))
    edge = np.pad(edge, ((0, 0), (0, pad), (0, 0)))
    return {"cls": cls, "pos": pos, "edge": edge}


def viterbi(*, prob: np.ndarray, costs: DecodeCosts) -> np.ndarray:
    """Four-state path (none, up, down, other) minimising negative log probability plus switch cost.

    Args:
        prob: Per-block state probabilities, shape (blocks, 4).
        costs: Switching costs.

    Returns:
        State per block.
    """
    emit = -np.log(np.clip(prob, 1e-6, 1.0))
    n = len(prob)
    t = np.full((4, 4), costs.on)
    t[1, 2] = t[2, 1] = costs.flip
    t[1, 3] = t[3, 1] = t[2, 3] = t[3, 2] = costs.flip
    np.fill_diagonal(t, 0.0)
    dp = emit[0].copy()
    back = np.zeros((n, 4), np.int8)
    for b in range(1, n):
        m = dp[:, None] + t
        back[b] = m.argmin(axis=0)
        dp = m[back[b], np.arange(4)] + emit[b]
    path = np.zeros(n, np.int8)
    path[-1] = int(dp.argmin())
    for b in range(n - 1, 0, -1):
        path[b - 1] = back[b, path[b]]
    return path


def state_probs(*, cls: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Per-block probabilities of none, up, down, other for one Primary."""
    p_shift = cls[:, 1]
    return np.stack([cls[:, 0], p_shift * pos, p_shift * (1 - pos), cls[:, 2] + cls[:, 3]], axis=1)


def primary_events(
    *,
    name: str,
    world: World,
    index: int,
    sigma: float,
    pred: dict[str, np.ndarray],
    costs: DecodeCosts,
) -> list[Event]:
    """Decode one Primary into events with half-hour starts and ends."""
    y = block_means(x=world.residual[name])
    x = y - world.slow[index]
    valid = np.flatnonzero(np.isfinite(x))
    if len(valid) < 1000:
        return []
    last = int(valid[-1]) + 1
    prob = state_probs(cls=pred["cls"][index, :last], pos=pred["pos"][index, :last])
    prob[:, NONE] *= costs.none_boost
    prob /= prob.sum(axis=1, keepdims=True)
    path = viterbi(prob=prob, costs=costs)
    slow_hh = np.interp(
        np.arange(N_BLOCKS * BLOCK), (np.arange(N_BLOCKS) + 0.5) * BLOCK, world.slow[index]
    )
    xh = world.residual[name] - slow_hh
    events: list[Event] = []
    b = 0
    while b < last:
        if path[b] == NONE:
            b += 1
            continue
        e = b
        while e + 1 < last and path[e + 1] == path[b]:
            e += 1
        seg = x[b : e + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) and abs(float(np.median(seg))) >= sigma:
            amp = float(np.median(seg))
            start, width = refine(xh=xh, boundary=b * BLOCK, reach=36)
            end = None if e == last - 1 else refine(xh=xh, boundary=(e + 1) * BLOCK, reach=36)[0]
            conf = float(np.mean(prob[b : e + 1, path[b]]))
            ev = Event(name, start, end, amp, abs(amp) / sigma, conf, width)
            if path[b] == OTHER:
                ramp = float(pred["cls"][index, b : e + 1, 2].mean())
                rout = float(pred["cls"][index, b : e + 1, 3].mean())
                ev.cls = "ramp" if ramp >= rout else "routine"
            events.append(ev)
        b = e + 1
    shifts = merge_fragments(events=[e for e in events if e.cls == "shift_no_partner"])
    return shifts + [e for e in events if e.cls != "shift_no_partner"]


def decode_net(
    *,
    panel: Panel,
    world: World,
    sigma: dict[str, float],
    pred: dict[str, np.ndarray],
    costs: DecodeCosts,
    edges: np.ndarray,
    neighbours: dict[str, list[str]] | None = None,
    pair: str = "net",
) -> list[Event]:
    """Decode every Primary and pair events across links.

    Args:
        panel: The panel.
        world: The world.
        sigma: Noise units in megawatts.
        pred: Output of `predict`.
        costs: Decode costs and edge cut-off.
        edges: The edge index pairs the prediction used.
        neighbours: The neighbour lists for the experiment 653 pairing rule.
        pair: `net` pairs with the edge posterior, `e653` applies experiment 653's `classify`.

    Returns:
        The events, with classes set.
    """
    events: list[Event] = []
    for i, s in enumerate(panel.sites):
        if np.isfinite(sigma[s]):
            events += primary_events(
                name=s, world=world, index=i, sigma=sigma[s], pred=pred, costs=costs
            )
    record_end = 0
    for s in panel.sites:
        ok = np.flatnonzero(np.isfinite(world.residual[s]))
        record_end = max(record_end, int(ok[-1]) + 1 if len(ok) else 0)
    if pair == "e653":
        from e00653_run import load_params
        from e00653_segment import classify

        pairable = [e for e in events if e.cls == "shift_no_partner"]
        classify(
            events=pairable,
            sigmas=sigma,
            neighbours=neighbours or panel.neighbours,
            p=load_params(),
            record_end=record_end,
        )
        return events
    _pair_with_edge_head(
        events=events, panel=panel, pred=pred, edges=edges, cutoff=costs.edge_cutoff
    )
    for e in events:
        if e.end is None:
            e.kind = "permanent" if (record_end - e.start) / DAY >= PERMANENT_DAYS else "open"
        else:
            e.kind = "temporary"
        if e.cls == "shift_no_partner" and e.kind == "permanent":
            e.cls = "permanent_shift"
        elif e.cls == "edge_transfer" and e.kind == "permanent":
            e.cls = "permanent_transfer"
    return events


def _pair_with_edge_head(
    *,
    events: list[Event],
    panel: Panel,
    pred: dict[str, np.ndarray],
    edges: np.ndarray,
    cutoff: float,
) -> None:
    """Pair opposite-sign events at linked Primaries whose mean edge posterior reaches `cutoff`."""
    index = {s: i for i, s in enumerate(panel.sites)}
    e_of = {(int(a), int(b)): k for k, (a, b) in enumerate(edges)}
    shifts = [e for e in events if e.cls == "shift_no_partner"]
    used: set[int] = set()
    for d in sorted((e for e in shifts if e.amp_mw < 0), key=lambda e: e.amp_mw):
        scored = []
        d_end = d.end if d.end is not None else GRID_LENGTH
        for r in shifts:
            if r.amp_mw <= 0 or id(r) in used:
                continue
            i, j = index[d.primary], index[r.primary]
            key, donor_state = (
                ((i, j), 1) if (i, j) in e_of else (((j, i), 2) if (j, i) in e_of else (None, 0))
            )
            if key is None:
                continue
            r_end = r.end if r.end is not None else GRID_LENGTH
            if abs(r.start - d.start) > 12:
                continue
            lo, hi = max(d.start, r.start), min(d_end, r_end)
            inter = hi - lo
            union = (d_end - d.start) + (r_end - r.start) - inter
            if inter <= 0 or inter / union < 0.5:
                continue
            b0, b1 = lo // BLOCK, max(hi // BLOCK, lo // BLOCK + 1)
            score = float(pred["edge"][e_of[key], b0:b1, donor_state].mean())
            if score >= cutoff:
                scored.append((score, r))
        scored.sort(key=lambda t: -t[0])
        chosen = [r for _, r in scored[:3]]
        total = sum(r.amp_mw for r in chosen)
        if chosen and 0.3 <= total / abs(d.amp_mw) <= 1.5:
            for r in chosen:
                used.add(id(r))
                r.cls, r.partners, r.partner_amp = (
                    "edge_transfer",
                    [d.primary],
                    {d.primary: d.amp_mw},
                )
            used.add(id(d))
            d.cls, d.partners = "edge_transfer", [r.primary for r in chosen]
            d.partner_amp = {r.primary: r.amp_mw for r in chosen}
            d.share = total / abs(d.amp_mw)


def graph_for(
    *, panel: Panel, neighbours: dict[str, list[str]] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour matrix and edges of the real or a rewired graph."""
    return graph_arrays(panel=panel, neighbours=neighbours)

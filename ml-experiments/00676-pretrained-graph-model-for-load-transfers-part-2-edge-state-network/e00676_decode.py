"""Decode the link and node states of an edge-state neural network into events (experiment 676).

`predict_world` runs a trained network over a whole hourly world in windows of 2,880 hours and
keeps the middle half of each. `decode` runs a three-state chain per link (fixed stay probability,
an onset cost, and the node evidence of the two ends added), accepts link segments greedily by
posterior mass, and turns each accepted segment into a donor event and a recipient event of
experiment 653's `Event` class, with sizes measured on the residual by experiment 670's `step_size`.
Labels never enter. There is no post hoc pairing window.
"""

# ruff: noqa: E501, RUF007

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00653_segment import Event
from e00670_search import step_size
from e00676_data import window_inputs
from e00676_net import N_LAGS, EdgeStateNet

SPAN = 2880
STAY = 0.999
NODE_WEIGHT = 0.5


@dataclass
class Posterior:
    """Network outputs over a world: link log-probabilities, lag log-probabilities, node ones."""

    link: np.ndarray  # (links, hours, 3)
    lag: np.ndarray  # (links, hours, 2, N_LAGS)
    node: np.ndarray  # (nodes, hours, 3)


def predict_world(
    *, net: EdgeStateNet, resid: np.ndarray, sigma: np.ndarray, mva: torch.Tensor,
    adj: torch.Tensor, device: torch.device, hours: int | None = None,
) -> Posterior:  # fmt: skip
    """Run `net` over the first `hours` hours (default all) in overlapping windows."""
    total = (hours or resid.shape[1]) // 24 * 24
    n_links, n_nodes = len(net.ia), resid.shape[0]  # ty: ignore[invalid-argument-type]
    link = np.zeros((n_links, total, 3), np.float32)
    lag = np.zeros((n_links, total, 2, N_LAGS), np.float32)
    node = np.zeros((n_nodes, total, 3), np.float32)
    quarter = SPAN // 4
    starts = list(range(0, max(1, total - SPAN + 1), SPAN // 2))
    if starts[-1] + SPAN < total:
        starts.append(total - SPAN)
    net.eval()
    for k, s in enumerate(starts):
        lo = 0 if k == 0 else s + quarter
        hi = total if k == len(starts) - 1 else s + SPAN - quarter
        h, d = window_inputs(resid=resid, start=s, span=SPAN, sigma=sigma, mva=mva)
        with torch.no_grad():
            lk, lg, nd = net(h.float()[None].to(device), d.float()[None].to(device), adj.float().to(device), mva.float().to(device))  # fmt: skip
        sl = slice(lo - s, hi - s)
        link[:, lo:hi] = torch.log_softmax(lk[0], -1)[:, sl].cpu().numpy()
        lag[:, lo:hi] = torch.log_softmax(lg[0], -1)[:, sl].cpu().numpy()
        node[:, lo:hi] = torch.log_softmax(nd[0], -1)[:, sl].cpu().numpy()
    return Posterior(link=link, lag=lag, node=node)


def transitions(*, onset: float) -> np.ndarray:
    """Log transition matrix of the link chain; entering a donating state costs `onset` nats."""
    move = np.log((1 - STAY) / 2)
    trans = np.full((3, 3), move)
    np.fill_diagonal(trans, np.log(STAY))
    trans[0, 1:] -= onset
    trans[1, 2] = trans[2, 1] = move - onset
    return trans


def viterbi(*, emit: np.ndarray, onset: float) -> np.ndarray:
    """Most likely state path of each link's 3-state chain: emit (links, hours, 3) to (links, hours)."""
    n, t, _ = emit.shape
    trans = transitions(onset=onset)
    score = emit[:, 0].copy()
    back = np.zeros((t, n, 3), np.int8)
    for i in range(1, t):
        cand = score[:, :, None] + trans[None]
        back[i] = cand.argmax(1)
        score = cand.max(1) + emit[:, i]
    path = np.zeros((n, t), np.int8)
    path[:, -1] = score.argmax(1)
    for i in range(t - 1, 0, -1):
        path[:, i - 1] = back[i, np.arange(n), path[:, i]]
    return path


def segments(
    *, post: Posterior, edges: list[tuple[int, int]], onset: float, cutoff: float, lo: int, hi: int
) -> list[dict]:
    """Link segments in hours [lo, hi): donor, recipient, start, end, lag class, mean posterior."""
    nd = post.node[:, lo:hi]
    t = np.arange(hi - lo)
    emit = post.link[:, lo:hi].astype(np.float64).copy()
    lag_mode = post.lag[:, lo:hi].argmax(-1)  # (links, hours, 2)
    for k, (a, b) in enumerate(edges):
        for s, (d, r) in ((1, (a, b)), (2, (b, a))):
            tr = np.clip(t + lag_mode[k, :, s - 1], 0, hi - lo - 1)
            emit[k, :, s] += NODE_WEIGHT * (nd[d, :, 1] + nd[r, tr, 2] - nd[d, :, 0] - nd[r, tr, 0])
    paths = viterbi(emit=emit, onset=onset)
    p = np.exp(post.link[:, lo:hi])
    out = []
    for k, (a, b) in enumerate(edges):
        path = paths[k]
        t0 = 0
        bounds = np.concatenate([[0], np.flatnonzero(np.diff(path) != 0) + 1, [len(path)]])
        for t0, t1 in zip(bounds[:-1], bounds[1:], strict=True):
            state = int(path[t0])
            if state == 0 or t0 == 0:  # a segment already running at the window start has no onset
                continue
            mean_p = float(p[k, t0:t1, state].mean())
            if mean_p >= cutoff:
                d, r = (a, b) if state == 1 else (b, a)
                lag_c = int(np.bincount(post.lag[k, lo + t0 : lo + t1, state - 1].argmax(-1), minlength=N_LAGS).argmax())  # fmt: skip
                out.append({"donor": d, "recipient": r, "start": lo + int(t0), "end": lo + int(t1), "lag": lag_c, "mass": mean_p * (t1 - t0)})  # fmt: skip
    return out


def accept(*, segs: list[dict]) -> list[dict]:
    """Greedy MAP: keep segments by mass, dropping any whose donor hours overlap an accepted one."""
    kept: list[dict] = []
    for s in sorted(segs, key=lambda x: -x["mass"]):
        clash = any(
            k["donor"] == s["donor"] and abs(k["start"] - s["start"]) > 12
            and min(k["end"], s["end"]) - max(k["start"], s["start"]) > 0
            for k in kept
        )  # fmt: skip
        dup = any(k["donor"] == s["donor"] and k["recipient"] == s["recipient"] and min(k["end"], s["end"]) > max(k["start"], s["start"]) for k in kept)  # fmt: skip
        if not clash and not dup:
            kept.append(s)
    return kept


def to_events(
    *,
    segs: list[dict],
    sites: list[str],
    xh: dict[str, np.ndarray],
    sigma: dict[str, float],
    hours: int,
) -> list[Event]:
    """Donor and recipient events (half-hour grid) from accepted segments; sizes from `step_size`."""
    events: list[Event] = []
    donors: list[Event] = []
    for s in sorted(segs, key=lambda x: x["start"]):
        d_site, r_site = sites[s["donor"]], sites[s["recipient"]]
        start = 2 * s["start"]
        end = None if s["end"] >= hours - 2 else 2 * s["end"]
        d_step = step_size(xh=xh[d_site], start=start, end=end)
        if not np.isfinite(d_step) or d_step >= 0:
            d_step = -abs(sigma[d_site]) * 4.0
        r_start = start + 2 * s["lag"]
        if end is not None:
            r_start = min(r_start, end - 2)
        r_step = step_size(xh=xh[r_site], start=r_start, end=end)
        if not np.isfinite(r_step) or r_step <= 0:
            r_step = abs(d_step) * 0.5
        same = next((e for e in donors if e.primary == d_site and abs(e.start - start) <= 24), None)
        if same is not None:
            same.partners.append(r_site)
            same.partner_amp[r_site] = r_step
        else:
            kind, cls = ("permanent", "permanent_transfer") if end is None else ("temporary", "edge_transfer")  # fmt: skip
            ev = Event(primary=d_site, start=start, end=end, amp_mw=d_step, units=abs(d_step) / sigma[d_site], confidence=0.5, ramp_fraction=0.0, kind=kind, cls=cls, partners=[r_site], partner_amp={r_site: r_step})  # fmt: skip
            donors.append(ev)
            events.append(ev)
        kind, cls = ("permanent", "permanent_transfer") if end is None else ("temporary", "edge_transfer")  # fmt: skip
        events.append(
            Event(
                primary=r_site,
                start=r_start,
                end=end,
                amp_mw=r_step,
                units=r_step / sigma[r_site],
                confidence=0.5,
                ramp_fraction=0.0,
                kind=kind,
                cls=cls,
                partners=[d_site],
                partner_amp={d_site: d_step},
            )
        )
    return events


def decode(
    *, post: Posterior, edges: list[tuple[int, int]], sites: list[str],
    xh: dict[str, np.ndarray], sigma: dict[str, float], onset: float, cutoff: float, lo: int, hi: int,
) -> list[Event]:  # fmt: skip
    """Full decode of hours [lo, hi) of a world into events."""
    segs = accept(segs=segments(post=post, edges=edges, onset=onset, cutoff=cutoff, lo=lo, hi=hi))
    return to_events(segs=segs, sites=sites, xh=xh, sigma=sigma, hours=post.link.shape[1])

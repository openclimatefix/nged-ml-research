"""Experiment 676 diagnostic (post hoc, CPU).

Edge events per Primary-year of a trained network on the real, un-injected record, by period, with the decoder chosen on validation worlds.

    uv run --no-sync python e00676_diag.py <arm> <seed>   (PYTHONPATH as in `e00676_train.py`)
"""

# ruff: noqa: D103, E501

from __future__ import annotations

import json
import sys

import torch  # ty: ignore[unresolved-import]
from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]
from e00675_pretrain import hourly_residual  # ty: ignore[unresolved-import]
from e00676_decode import decode, predict_world
from e00676_net import EdgeStateNet
from e00676_run import xh_of
from e00676_train import graph_of
from e00676_worlds import ROOT

PERIODS = {"2022-01 to 2023-06 (training years)": (0, 13032), "2023-07 to 2023-12": (13032, 17520), "2024-01 to 2026-06 (scored years)": (17520, 39408)}  # fmt: skip


def main() -> None:
    arm, seed = sys.argv[1], int(sys.argv[2])
    rig = Rig()
    sites, mva, edges, adj = graph_of(rig=rig, arm=arm)
    net = EdgeStateNet(edges=edges, no_links=arm == "G-0")
    net.load_state_dict(torch.load(ROOT / "weights" / f"{arm}_seed{seed}.pt", map_location="cpu"))
    dec = json.loads((ROOT / "scored" / f"{arm}_seed{seed}_decoder.json").read_text())
    x_all, sig_t, _, _ = hourly_residual(rig=rig)
    post = predict_world(net=net, resid=x_all.numpy(), sigma=sig_t.numpy(), mva=mva, adj=adj, device=torch.device("cpu"))  # fmt: skip
    xh = xh_of(resid=x_all.numpy(), sigma=sig_t.numpy(), sites=sites)
    sigma = {s: float(sig_t[i]) for i, s in enumerate(sites)}
    out = {}
    for name, (lo, hi) in PERIODS.items():
        ev = decode(post=post, edges=edges, sites=sites, xh=xh, sigma=sigma, onset=dec["onset"], cutoff=dec["cutoff"], lo=lo, hi=min(hi, post.link.shape[1]))  # fmt: skip
        n = sum(e.amp_mw < 0 and e.cls.endswith("transfer") for e in ev)
        out[name] = n / (15 * (hi - lo) / (24 * 365.25))
        print(name, round(out[name], 2), flush=True)
    (ROOT / "scored" / f"{arm}_seed{seed}_real_rate.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

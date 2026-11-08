# ruff: noqa: E501, D103, ANN001, ANN201, E402,RET504
"""Experiment 651: the design's leak guard on the real input builders (CPU, before any fit).

For one origin of 2025-10, every power reading from 24 h before the origin onwards (all 16 Primaries,
neighbours included) is replaced with noise. The inputs a row carries as its own, and the inputs its
neighbours carry for it in the bank, must be bit-identical, and so must a graph network's forecasts
(forecasts depend on nothing else). Also asserts that the bank holds no target.
"""

import sys

sys.argv = ["e00651_arm.py", "G0+D50+GAT", "check", "--months", "2025-10", "--seeds", "0"]


import e00651_arm as arm  # ty: ignore[unresolved-import]
import e00651_nbmodel as mm  # ty: ignore[unresolved-import]
import numpy as np
import torch

e = arm.e
MONTH, SEED = "2025-10", 0
clean = e.c.clean_stack()
mr = e.k.build_month(month=MONTH, clean=clean)
arm.STD["std"] = mr.base.target_std
test = e.build_test(month=MONTH, standardiser=mr.base.standardiser)
z = np.load(arm.table_file(month=MONTH, seed=SEED))
fs = arm.FoldSignals.__new__(arm.FoldSignals)
fs.table = arm.s598.LagFreeTable(median=z["median"], origin_idx=z["origin_idx"])
fs.weather_origin = np.asarray(e.CTX["origin_ft"])

o0 = int(np.median(np.unique(test.origin_idx)))
sel = test.origin_idx == o0
sub = arm.subset(test=test, sel=sel)
print(f"origin {o0}: {len(sub.x)} rows, {len(np.unique(sub.primary))} Primaries")

rng = np.random.default_rng(0)
noisy = clean.copy()
import os

cut = (
    o0 - int(os.environ.get("LEAK_HOURS_STEPS", "47"))
)  # readings at grid index o0 - 47 or later are within 24 h of the origin; 60 must fail (positive control)
noisy[:, cut:] = rng.uniform(-1.0, 1.0, size=noisy[:, cut:].shape)


def build(power):
    own_x = arm.msg_lags(
        x=sub.x, power=power, prim=sub.primary, target_idx=sub.target_idx, origin_idx=sub.origin_idx
    )
    own = np.concatenate([own_x, fs.test_block(test=sub, power_lag=power, power_sig=power)], axis=1)
    return own


a, b = build(clean), build(noisy)
same_x = np.array_equal(a, b, equal_nan=True)
print(
    "rows' inputs (own and as the neighbours see them) bit-identical under noisy recent power:",
    same_x,
)
if not same_x:
    bad = np.flatnonzero(~np.all((a == b) | (np.isnan(a) & np.isnan(b)), axis=0))
    print("differing columns:", bad.tolist())
assert set(vars(mm.BANK)) <= {"x", "p", "nbr"} or set(dir(mm.Bank)) >= {"x", "p", "nbr"}
assert not any(
    "target" in n or n == "z"
    for n in vars(mm.Bank) | {k: 1 for k in dir(mm.Bank) if not k.startswith("_")}
), "the bank holds a target"

# forecasts: a random-weight graph network on both versions
arm.e.KNOTS = max(arm.e.KNOTS, 14)
arm.GRAPH_KIND, arm.GRAPH_MOD = "gat", "GAT"
arm._ADJ.clear()
torch.manual_seed(0)
m = arm.new_model(n_primaries=16).eval()
with torch.no_grad():
    for q in m.parameters():
        q.add_(torch.randn_like(q) * 0.2)
m.table = e.CTX["table_ft"].cpu().float()


def forecast(power):
    xm = arm.msg_lags(
        x=sub.x, power=power, prim=sub.primary, target_idx=sub.target_idx, origin_idx=sub.origin_idx
    )
    block = fs.test_block(test=sub, power_lag=power, power_sig=power)
    own = np.concatenate([xm, block], axis=1)
    arm.set_bank(
        x=own,
        prim=sub.primary,
        nbr=arm.neighbour_table(
            prim=sub.primary,
            origin_idx=sub.origin_idx,
            target_idx=sub.target_idx,
            adj=arm.adjacency(seed=SEED),
        ),
        device="cpu",
    )
    ids = np.arange(len(own), dtype=np.float32)[:, None]
    with torch.no_grad():
        return m(
            torch.from_numpy(np.concatenate([own, ids], axis=1)), torch.from_numpy(sub.primary)
        )


fa, fb = forecast(clean), forecast(noisy)
same_f = torch.equal(fa, fb)
print("forecasts bit-identical under noisy recent power:", same_f)
sys.exit(0 if same_x and same_f else 1)

# ruff: noqa: E501, D103, ANN001, ANN201, S101, INP001
"""Experiment 651: CPU tests of the graph network, run before any fit (`pytest`-free script; exit code 0 means pass).

1. Zero start: every graph kind gives exactly the own-signal network's output before training.
2. Locality: GAT-E and GCN-E outputs do not change when a non-neighbour's bank row changes, and do change
   (once the layer is non-zero) when a neighbour's bank row changes, so the test can fail.
3. No pooling over nodes: shuffling the order of the batch rows leaves each row's output unchanged.
4. Flow conservation: the per-edge flows in MW cancel between the two ends of every edge.
"""

import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "G0+D50+GAT"
sys.argv = ["e00651_arm.py", mode, "check", "--months", "2025-10", "--seeds", "0"]

import e00651_arm as arm  # noqa: E402  # ty: ignore[unresolved-import]
import e00651_nbmodel as mm  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(0)
N_PRIM, GRID, N_ROWS = 16, 200, 400
arm.STD["std"] = np.full(N_PRIM, 0.1, np.float32)
arm.e.CTX["stats"] = np.ones((2, 2), np.float32) * 0.7
arm.e.KNOTS = 14  # the fit sets this from the month list; any value of 2 or more builds the network


def make(kind_mod: str):
    arm.GRAPH_KIND = arm.GRAPH_MODS[kind_mod]
    arm.GRAPH_MOD = kind_mod
    arm._ADJ.clear()
    m = arm.new_model(n_primaries=N_PRIM)
    m.table = torch.randn(N_PRIM, GRID, 28).half()
    return m


def rows(*, width: int, n: int = N_ROWS):
    x = torch.randn(n, width) * 0.5
    x[:, 38] = (torch.rand(n) < 0.2).float()
    x[:, 41] = (torch.rand(n) < 0.2).float()
    x[:, 43] = torch.randint(0, GRID, (n,)).float()
    x[:, 44] = torch.randint(0, GRID, (n,)).float()
    return x


def bank_for(*, x: torch.Tensor, adj: dict[int, list[int]]):
    prim = torch.randint(0, N_PRIM, (len(x),))
    # one row per (primary, time) over 25 times so that neighbours exist
    t = torch.arange(len(x)) % 25
    prim = (torch.arange(len(x)) // 25) % N_PRIM
    nbr = arm.neighbour_table(
        prim=prim.numpy(), origin_idx=t.numpy(), target_idx=t.numpy(), adj=adj
    )
    mm.BANK.x, mm.BANK.p, mm.BANK.nbr = x.clone(), prim, torch.from_numpy(nbr)
    return prim, nbr


def feed(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.arange(len(x), dtype=torch.float32)[:, None]], dim=1)


fails = []
for kind_mod in ("GCN", "GAT", "GATR", "GATF", "FLOW"):
    m = make(kind_mod).eval()
    width = m.label_col + m.inputs.n_extra
    x = rows(width=width)
    prim, nbr = bank_for(x=x, adj=arm.adjacency(seed=0))
    assert (nbr >= 0).any(), "the test bank has no neighbour pairs"
    # 1. zero start equals the own-signal network
    plain = mm.PermLagMdn(
        n_primaries=N_PRIM, stage="ft", n_knots=arm.e.KNOTS, n_b1=arm.e.N_B1, inputs=arm.INPUTS
    ).eval()
    plain.table = m.table
    with torch.no_grad():  # the output head starts at zero, which would hide every difference
        for q_ in plain.parameters():
            q_.add_(torch.randn_like(q_) * 0.3)
    state = {a: v for a, v in plain.state_dict().items()}
    miss = m.load_state_dict(state)
    with torch.no_grad():
        got = m(feed(x), prim)
        want = plain(x, prim)
    ok = torch.allclose(got, want, atol=1e-6)
    print(f"{kind_mod}: zero start equals own-signal network: {ok}")
    if not ok:
        fails.append((kind_mod, "zero start", float((got - want).abs().max())))
    # make the layer non-zero
    with torch.no_grad():
        for n, p_ in m.named_parameters():
            if n.startswith(("graph_out", "flow_scale")):
                p_.copy_(torch.randn_like(p_) * 0.5)
    adj = arm.adjacency(seed=0)
    cols = list(range(37)) + list(range(m.label_col, width))
    with torch.no_grad():
        base = m(feed(x), prim)
        # 2a. change one row's bank entry that is NOT a neighbour of row 0: row 0 must not move
        r0 = 0
        p0 = int(prim[r0])
        non = [i for i in range(len(x)) if int(prim[i]) != p0 and int(prim[i]) not in adj[p0]]
        nb_rows = {int(j) for j in nbr[r0] if j >= 0}
        far = next((i for i in non if i not in nb_rows), None)
        stable = True
        if far is not None:
            mm.BANK.x[far, cols] += 3.0
            moved = m(feed(x), prim)
            stable = torch.equal(moved[r0], base[r0])
            mm.BANK.x[far, cols] -= 3.0
        # 2b. change a neighbour's bank entry: row 0's output must move (the test can fail)
        if nb_rows:
            j = next(iter(nb_rows))
            mm.BANK.x[j, cols] += 3.0
            moved2 = m(feed(x), prim)
            mm.BANK.x[j, cols] -= 3.0
            reacts = not torch.equal(moved2[r0], base[r0])
        else:
            reacts = True
        # 3. batch order
        perm = torch.randperm(len(x))
        shuffled = m(
            torch.cat([x[perm], torch.arange(len(x), dtype=torch.float32)[perm][:, None]], dim=1),
            prim[perm],
        )
        same = torch.allclose(shuffled, base[perm], rtol=1e-4, atol=1e-3)
    full = kind_mod == "GATF"
    print(
        kind_mod,
        "row0 neighbours",
        sorted(nb_rows)[:6],
        "max |change|",
        float((moved2[r0] - base[r0]).abs().max()) if nb_rows else None,
    )
    print(
        f"{kind_mod}: non-neighbour unchanged {stable or full}, neighbour reacts {reacts}, batch-order invariant {same}"
    )
    if not (stable or full) or not reacts or not same:
        fails.append((kind_mod, "locality", stable, reacts, same))
    # 4. flow conservation
    if kind_mod == "FLOW":
        m.train(False)
        xb = feed(x)
        _, mw = m.flow_correction(x=x, p=prim, rid=xb[:, -1].long())
        idx = mm.BANK.nbr
        tot, cnt = 0.0, 0
        for i in range(len(x)):
            for k in range(idx.shape[1]):
                j = int(idx[i, k])
                if j >= 0:
                    # the reverse edge: row j's slot that points back to i's primary at the same time
                    back = [kk for kk in range(idx.shape[1]) if int(idx[j, kk]) == i]
                    if back:
                        tot = max(tot, float((mw[i, k] + mw[j, back[0]]).abs().max()))
                        cnt += 1
        print(f"FLOW: largest MW imbalance over {cnt} edge ends: {tot:.2e}")
        if cnt == 0 or tot > 1e-4:
            fails.append(("FLOW", "conservation", tot, cnt))

print("FAILS", fails)
sys.exit(1 if fails else 0)

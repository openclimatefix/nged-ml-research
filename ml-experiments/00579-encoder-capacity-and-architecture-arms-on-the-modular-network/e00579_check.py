# ruff: noqa: E501
"""Experiment 579 offline checks (CPU): the default `ArmMdn` equals experiment 572's `LagOffsetMdn`, and parameter counts per arm."""

import e00572_model as m572  # ty: ignore[unresolved-import]
import e00579_arms as arms  # ty: ignore[unresolved-import]
import e00579_model as mm  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

N_PRIMARIES, KNOTS, ROWS = 16, 6, 64


def equivalence() -> float:
    """Return the largest absolute difference between the two networks' outputs at the default config."""
    torch.manual_seed(0)
    old = m572.LagOffsetMdn(n_primaries=N_PRIMARIES, stage="pre", n_knots=KNOTS)
    for p in old.parameters():
        torch.nn.init.normal_(p, std=0.2)
    new = mm.ArmMdn(n_primaries=N_PRIMARIES, stage="pre", n_knots=KNOTS, cfg=mm.Cfg())
    sd = {}
    for a, v in old.state_dict().items():
        if a.startswith("mixer_pre.6."):
            a = a.replace("mixer_pre.6.", "mixer_pre.out.")
        elif a.startswith("mixer_pre."):
            a = a.replace("mixer_pre.", "mixer_pre.trunk.")
        elif a.startswith("mixer_ft."):
            a = a.replace("mixer_ft.6.", "mixer_ft.out.") if ".6." in a else a.replace("mixer_ft.", "mixer_ft.trunk.")
        sd[a] = v
    res = new.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    table = torch.randn(N_PRIMARIES, 100, 28)
    x = torch.randn(ROWS, 46)
    x[:, 43] = torch.randint(0, 100, (ROWS,)).float()
    x[:, 44] = torch.randint(0, 100, (ROWS,)).float()
    x[:, 45] = torch.rand(ROWS) * (KNOTS - 1)
    p = torch.randint(0, N_PRIMARIES, (ROWS,))
    old.table, new.table = table, table
    old.eval()
    new.eval()
    with torch.no_grad():
        return float((old(x, p) - new(x, p)).abs().max())


def counts() -> None:
    """Print each arm's parameter counts per component (stage `pre`)."""
    for tag, arm in arms.ARMS.items():
        n_extra = arm.cfg.n_extra
        model = mm.ArmMdn(n_primaries=N_PRIMARIES, stage="pre", n_knots=KNOTS, cfg=arm.cfg)
        table = torch.randn(N_PRIMARIES, 100, 28 + n_extra)
        if arm.cfg.p2:
            table[..., -1] = torch.randint(0, 48, (N_PRIMARIES, 100)).float()
        model.table = table
        x = torch.randn(8, 46 + n_extra)
        x[:, 43:45] = 5.0
        x[:, 45] = 2.0
        if arm.cfg.p2:
            x[:, -1] = 7.0
        p = torch.randint(0, N_PRIMARIES, (8,))
        model.eval()
        out = model(x, p)
        assert out.shape == (8, 3 * arm.cfg.k), (tag, out.shape)
        c = model.parameter_counts()
        print(tag, {a: v for a, v in c.items()}, "total", sum(v.numel() for v in model.parameters()))


if __name__ == "__main__":
    print("max abs difference, default ArmMdn vs experiment 572's LagOffsetMdn:", equivalence())
    counts()

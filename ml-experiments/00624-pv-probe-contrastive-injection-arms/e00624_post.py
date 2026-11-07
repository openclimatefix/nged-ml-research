# ruff: noqa: E501
"""Experiment 624, arms (b) and (c): post-hoc calibration check by refitting only the twin embeddings.

Arm (b) starts from experiment 621's plain network of the same world, cutoff, and seed; arm (c)
starts from this experiment's arm (a) network. The twin embedding vectors (16 to 27) are initialised
from the real Primary's vector, every other parameter is frozen, and the twin embeddings are fitted
by negative log-likelihood on the twin rows alone, for `EPOCHS` epochs at `LR`. Nothing about the
real Primaries changes, so the check reads how much of a known +1 MW PV injection the network's
existing sky response lets the twin embedding absorb.

Usage: python e00624_post.py --cutoff 2026-06 --source plain --conditions control farm_mw4_r0
"""

import argparse
import time

import e00397_models as base  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00624_common as k
import torch  # ty: ignore[unresolved-import]

EPOCHS = 4
LR = 1e-2


def load(
    *, source: str, cutoff: str, condition: str, seed: int, six: list[int], device: str
) -> m.ModularMdn:
    """Load the source network into a 28-vector network, copying real vectors into the twin slots."""
    model = m.ModularMdn(n_primaries=k.N_EMBED, stage="pre")
    path = (
        k.plain_weights_path(cutoff=cutoff, condition=condition, seed=seed)
        if source == "plain"
        else k.weights_path(family="a", cutoff=cutoff, condition=condition, seed=seed)
    )
    state = torch.load(path, map_location="cpu")
    if source == "plain":
        table = model.embedding.weight.data.clone()
        table[: k.N_REAL] = state["embedding.weight"]
        state = {**state, "embedding.weight": table}
    model.load_state_dict(state)
    with torch.no_grad():
        for j, p in enumerate(six):
            model.embedding.weight[k.PHYS_BASE + j] = model.embedding.weight[p]
            model.embedding.weight[k.PROFILE_BASE + j] = model.embedding.weight[p]
    return model.to(device).eval()


def main() -> None:
    """Refit the twin embeddings of every requested network and store the weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--source", required=True, choices=["plain", "a"])
    parser.add_argument("--conditions", nargs="+", default=list(k.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    args = parser.parse_args()
    device = "cuda"
    family = "b" if args.source == "plain" else "c"
    fold = k.Fold(cutoff=args.cutoff)
    six = list(fold.six.values())
    for name in args.conditions:
        d = fold.combined(condition=name, device=device)
        idx = torch.nonzero(d["is_twin"] & ~d["is_val"]).squeeze(1)
        for seed in args.seeds:
            target = k.WORK / f"pretrained_{family}" / args.cutoff / f"{name}_s{seed}.pt"
            if target.exists():
                continue
            started = time.monotonic()
            model = load(
                source=args.source,
                cutoff=args.cutoff,
                condition=name,
                seed=seed,
                six=six,
                device=device,
            )
            for p in model.parameters():
                p.requires_grad_(False)
            model.embedding.weight.requires_grad_(True)
            mask = torch.zeros_like(model.embedding.weight)
            mask[k.PHYS_BASE :] = 1.0
            optimiser = torch.optim.Adam([model.embedding.weight], lr=LR)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for _ in range(EPOCHS):
                order = idx[torch.randperm(len(idx), generator=generator).to(device)]
                order = order[: int(c.ROW_FRACTION * len(order))]
                for start in range(0, len(order), c.BATCH):
                    b = order[start : start + c.BATCH]
                    out = model(d["x"][d["row_idx"][b]], d["emb"][b])
                    loss = base.mixture_nll(out=out, z=d["z"][b], components=m.COMPONENTS).mean()
                    optimiser.zero_grad()
                    loss.backward()
                    model.embedding.weight.grad *= mask
                    optimiser.step()
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  {args.cutoff} {family} {name} s{seed}: {time.monotonic() - started:.0f} s",
                flush=True,
            )
        del d
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

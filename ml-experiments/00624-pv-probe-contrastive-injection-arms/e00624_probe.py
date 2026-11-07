# ruff: noqa: E501
"""Experiment 624: the donor-day probe of experiment 623, applied to the real and twin embedding vectors.

The probe code is experiment 623's (and experiment 621's) and is imported, not copied. A wrapper
replaces the Primary index that the network sees, so that one Primary's probe cells (its datetime
and donor weather) can be run through the real Primary's embedding vector or through either twin's.

Usage: python e00624_probe.py --family a --cutoff 2026-06 --conditions control farm_mw4_r0
"""

import argparse

import e00554_model as m  # ty: ignore[unresolved-import]
import e00623_probe as p623  # ty: ignore[unresolved-import]
import e00624_common as k
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]


class FixedEmbedding(nn.Module):
    """Run the wrapped network with every row's Primary index replaced by `index`."""

    def __init__(self, *, inner: m.ModularMdn, index: int) -> None:
        """Hold the network and the embedding index."""
        super().__init__()
        self.inner, self.index = inner, index

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the wrapped network's raw outputs."""
        assert primary is not None
        return self.inner(x, torch.full_like(primary, self.index))


def main() -> None:
    """Probe every requested network at its real and twin embedding vectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["a", "b", "c", "nopair"])
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--conditions", nargs="+", default=list(k.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    frame = p623.load_frame(cutoff=args.cutoff)
    six = k.k.six_indices()
    cap = k.k.effective_capacity_mw()
    probes = {site: p623.cells_and_donors(frame=frame, primary=i) for site, i in six.items()}
    for name in args.conditions:
        out = k.WORK / f"probes_{args.family}" / args.cutoff / f"{name}.parquet"
        if out.exists():
            continue
        records = []
        for seed in args.seeds:
            path = (
                k.weights_path(family=args.family, cutoff=args.cutoff, condition=name, seed=seed)
                if args.family in {"a", "nopair"}
                else k.WORK / f"pretrained_{args.family}" / args.cutoff / f"{name}_s{seed}.pt"
            )
            assert path.exists(), path
            net = m.ModularMdn(n_primaries=k.N_EMBED, stage="pre")
            net.load_state_dict(torch.load(path, map_location="cpu"))
            net.eval()
            for j, (site, i) in enumerate(six.items()):
                for kind, index in (
                    ("real", i),
                    ("twin_phys", k.PHYS_BASE + j),
                    ("twin_prof", k.PROFILE_BASE + j),
                ):
                    rng = np.random.default_rng(1000 * seed + i)
                    res = p623.probe_network(
                        model=FixedEmbedding(inner=net, index=index),
                        frame=frame,
                        primary=i,
                        probe=probes[site],
                        cap=float(cap[i]),
                        rng=rng,
                    )
                    for model_kind in ("plain", "cloud_tpl"):
                        point, boot = res[model_kind]
                        records.append(
                            {
                                "cutoff": args.cutoff, "condition": name, "seed": seed, "site": site, "emb_kind": kind,
                                "model": model_kind, "capacity_mw": point, "boot": boot.tolist(),
                            }
                        )  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(records).write_parquet(out)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

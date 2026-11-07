# ruff: noqa: E501
"""Experiment 624, arm (a): pretrain with the twin rows and the paired-difference term (a training loss).

Usage: python e00624_fit.py --cutoff 2026-06 --conditions control farm_mw4_r0 --seeds 0 1 2
"""

import argparse
import json
import time

import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00624_common as k
import torch  # ty: ignore[unresolved-import]


def main() -> None:
    """Pretrain the requested worlds and seeds for one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--conditions", nargs="+", default=list(k.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    parser.add_argument("--family", default="a", choices=["a", "nopair"])
    args = parser.parse_args()
    device = "cuda"
    fold = k.Fold(cutoff=args.cutoff)
    weight = k.PAIR_WEIGHT if args.family == "a" else 0.0
    for name in args.conditions:
        d = fold.combined(condition=name, device=device)
        print(f"{args.cutoff} {name}: {len(d['row_idx']):,} rows incl. twins", flush=True)
        for seed in args.seeds:
            target = k.weights_path(
                family=args.family, cutoff=args.cutoff, condition=name, seed=seed
            )
            if target.exists():
                continue
            started = time.monotonic()
            torch.manual_seed(seed)
            model = m.ModularMdn(n_primaries=k.N_EMBED, stage="pre").to(device)
            epochs, best = k.train(
                model=model, d=d, seed=seed, lr=c.PRETRAIN_LR, pair_weight=weight, device=device
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            seconds = time.monotonic() - started
            print(
                f"  {args.cutoff} {name} s{seed}: {epochs} epochs, val NLL {best:.4f}, {seconds:.0f} s",
                flush=True,
            )
            log = k.WORK / "logs" / args.family / args.cutoff / f"{name}_s{seed}.json"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps({"epochs": epochs, "val_nll": best, "seconds": seconds}))
        del d
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

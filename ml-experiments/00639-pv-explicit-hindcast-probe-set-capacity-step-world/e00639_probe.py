# ruff: noqa: E501
"""Experiment 639: probe the free networks of both worlds (five seeds) on the CPU.

The probe is experiment 634's donor-day probe.

Usage: python e00639_probe.py --cutoff 2026-06
"""

import argparse

import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00634_probe as p  # ty: ignore[unresolved-import]
import e00639_common as k
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]


def main() -> None:
    """Probe the free networks of each world at one cutoff and write one parquet per world."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    p.SHIFT_DAYS = 0
    frame = p.load_frame(cutoff=args.cutoff)
    six = k634.six_indices()
    cap = k634.effective_capacity_mw()
    probes = {site: p.cells_and_donors(frame=frame, primary=i) for site, i in six.items()}
    for world in k.WORLDS:
        out = k.probe_path(cutoff=args.cutoff, world=world)
        if out.exists():
            continue
        records = []
        for seed in k.SEEDS:
            model = m.ModularMdn(n_primaries=16, stage="pre")
            model.load_state_dict(
                torch.load(
                    k.free_weights_path(cutoff=args.cutoff, world=world, seed=seed),
                    map_location="cpu",
                )
            )
            model.eval()
            for site, i in six.items():
                rng = np.random.default_rng(1000 * seed + i)
                res = p.probe_network(
                    model=model,
                    frame=frame,
                    primary=i,
                    probe=probes[site],
                    cap=float(cap[i]),
                    rng=rng,
                )
                point, boot = res["plain"]
                records.append(
                    {
                        "cutoff": args.cutoff, "world": world, "seed": seed, "site": site, "model": "plain",
                        "capacity_mw": point, "boot": boot.tolist(), "n_rows": res["n_rows"],
                    }
                )  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(records).write_parquet(out)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Experiment 638: the donor-day probe of experiment 634 applied to this experiment's networks (CPU).

The probe, its frames, its probe dates, and its regression are experiment 634's `e00634_probe.py`.
`--shift-days 364` moves every probe date back 52 weeks (the previous-year base).

Usage: python e00638_probe.py --cutoff 2026-06 --shift-days 0
"""

import argparse

import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00634_probe as p  # ty: ignore[unresolved-import]
import e00638_common as k
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]


def main() -> None:
    """Probe every existing network of the requested conditions and write one parquet per condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--conditions", nargs="+", default=list(k.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    parser.add_argument("--shift-days", type=int, default=0)
    args = parser.parse_args()
    p.SHIFT_DAYS = args.shift_days
    torch.set_num_threads(2)
    frame = p.load_frame(cutoff=args.cutoff)
    six = k634.six_indices()
    cap = k634.effective_capacity_mw()
    probes = {site: p.cells_and_donors(frame=frame, primary=i) for site, i in six.items()}
    for name in args.conditions:
        paths = [k.weights_path(cutoff=args.cutoff, condition=name, seed=s) for s in args.seeds]
        if not all(path.exists() for path in paths):
            print(f"{args.cutoff} {name}: no networks; skipped", flush=True)
            continue
        out = k.probes_dir(shift_days=args.shift_days) / args.cutoff / f"{name}.parquet"
        if out.exists():
            continue
        records = []
        for seed, path in zip(args.seeds, paths, strict=True):
            model = m.ModularMdn(n_primaries=16, stage="pre")
            model.load_state_dict(torch.load(path, map_location="cpu"))
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
                for kind in ("plain", "cloud_tpl"):
                    point, boot = res[kind]
                    records.append(
                        {
                            "cutoff": args.cutoff, "condition": name, "seed": seed, "site": site, "model": kind,
                            "capacity_mw": point, "boot": boot.tolist(), "n_rows": res["n_rows"],
                            "swing_mw": res["mean_swing_mw"],
                        }
                    )  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(records).write_parquet(out)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

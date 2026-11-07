# ruff: noqa: E501
"""Experiment 624: validation-row CRPS of the uninjected arm (a) networks against experiment 621's plain ones.

Scores the real rows only (never the twin rows) of the pretraining validation set, as in experiment
623's `e00623_skill.py`, through the real Primary's embedding vector.

Usage: python e00624_skill.py --cutoff 2026-06
"""

import argparse
import json

import e00554_model as m  # ty: ignore[unresolved-import]
import e00623_skill as s623  # ty: ignore[unresolved-import]
import e00624_common as k
import numpy as np
import torch  # ty: ignore[unresolved-import]

N_ROWS = 600_000


def main() -> None:
    """Score the control networks of both arms and write `work/skill_<cutoff>.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    args = parser.parse_args()
    fold = k.Fold(cutoff=args.cutoff)
    pre = fold.pre
    keep = fold.keep & pre.is_val
    idx = np.nonzero(keep)[0]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(idx, size=min(N_ROWS, len(idx)), replace=False))
    std = fold.rows.target_std[pre.primary[idx]]
    z = torch.from_numpy(
        ((pre.y[idx] - fold.rows.target_mean[pre.primary[idx]]) / std).astype(np.float32)
    )
    x, p = torch.from_numpy(pre.x[idx]), torch.from_numpy(pre.primary[idx])
    scale = std * fold.cap[pre.primary[idx]]
    six = np.isin(pre.primary[idx], list(fold.six.values()))
    result: dict = {"rows": len(idx), "six_rows": int(six.sum())}
    for arm in ("plain", "a"):
        for seed in k.SEEDS:
            model = m.ModularMdn(n_primaries=k.N_EMBED if arm == "a" else k.N_REAL, stage="pre")
            path = (
                k.plain_weights_path(cutoff=args.cutoff, condition="control", seed=seed)
                if arm == "plain"
                else k.weights_path(family="a", cutoff=args.cutoff, condition="control", seed=seed)
            )
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            with torch.no_grad():
                crps = (
                    torch.cat(
                        [
                            s623.mixture_crps(
                                out=model(x[i : i + 65_536], p[i : i + 65_536]), z=z[i : i + 65_536]
                            )
                            for i in range(0, len(x), 65_536)
                        ]
                    ).numpy()
                    * scale
                )
            result[f"{arm}_s{seed}"] = {
                "crps_mw_all": float(crps.mean()),
                "crps_mw_six": float(crps[six].mean()),
            }
            print(args.cutoff, arm, seed, result[f"{arm}_s{seed}"], flush=True)
    (k.WORK / f"skill_{args.cutoff}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

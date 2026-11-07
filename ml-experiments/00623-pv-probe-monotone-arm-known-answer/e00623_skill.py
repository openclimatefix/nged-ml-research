# ruff: noqa: E501, ANN001, ANN201
"""Experiment 623: validation-row CRPS of the uninjected control networks, plain arm against monotone arm.

The CRPS of a Gaussian mixture is exact: sum_k w_k A(y - mu_k, s_k^2) - 0.5 sum_kl w_k w_l A(mu_k - mu_l,
s_k^2 + s_l^2), with A(m, v) = m (2 Phi(m / sqrt v) - 1) + 2 sqrt(v) phi(m / sqrt v). It is computed on the
pretraining validation rows (the rows each network's early stopping reads, a time-ordered hold-out
inside the pretraining window, so no row is dated 2026-07-01 or later), in megawatts, for all 16
Primaries and for the six MW Primaries alone. The same rows score both arms.

Usage: python e00623_skill.py --cutoff 2026-06
"""

import argparse
import json
import math

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00623_common as k
import e00623_model as mono
import numpy as np
import torch  # ty: ignore[unresolved-import]

N_ROWS = 600_000
SQRT2 = math.sqrt(2.0)


def _a(*, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    sd = var.sqrt()
    t = mean / sd
    cdf = 0.5 * (1.0 + torch.erf(t / SQRT2))
    pdf = torch.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    return mean * (2.0 * cdf - 1.0) + 2.0 * sd * pdf


def mixture_crps(*, out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Return the exact CRPS (standardised units) of each row's Gaussian mixture."""
    log_w, mu, sd = base.mixture_parts(out=out.double(), components=m.COMPONENTS)
    w, var = log_w.exp(), sd**2
    first = (w * _a(mean=z[:, None].double() - mu, var=var)).sum(-1)
    dm = mu[:, :, None] - mu[:, None, :]
    dv = var[:, :, None] + var[:, None, :]
    second = 0.5 * (w[:, :, None] * w[:, None, :] * _a(mean=dm, var=dv)).sum((-1, -2))
    return first - second


def load_model(*, arm: str, path):
    """Load a stored network of the given arm."""
    model = (
        mono.MonotoneModularMdn(n_primaries=16)
        if arm == "monotone"
        else m.ModularMdn(n_primaries=16, stage="pre")
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def main() -> None:
    """Score the control networks of both arms and write `work/skill_<cutoff>.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    args = parser.parse_args()
    clean = c.clean_stack()
    refit_t = c.month_start(month=args.cutoff)
    rows = data.build_fold(refit=refit_t, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
    inj = k.Injector(
        primary=pre.primary, target_idx=pre.target_idx, x=pre.x, standardiser=rows.standardiser
    )
    keep = ~inj.gap & pre.is_val
    idx = np.nonzero(keep)[0]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(idx, size=min(N_ROWS, len(idx)), replace=False))
    tgt_mean, tgt_std = rows.target_mean[pre.primary[idx]], rows.target_std[pre.primary[idx]]
    z = torch.from_numpy(((pre.y[idx] - tgt_mean) / tgt_std).astype(np.float32))
    x = torch.from_numpy(pre.x[idx])
    p = torch.from_numpy(pre.primary[idx])
    cap = k.effective_capacity_mw()[pre.primary[idx]]
    scale_mw = tgt_std * cap
    six = np.isin(pre.primary[idx], list(k.six_indices().values()))
    result: dict = {"rows": len(idx), "six_rows": int(six.sum())}
    for arm, root in (("plain", k.PLAIN_WORK), ("monotone", k.WORK)):
        for seed in args.seeds:
            path = root / "pretrained" / args.cutoff / f"control_s{seed}.pt"
            model = load_model(arm=arm, path=path)
            with torch.no_grad():
                crps_z = torch.cat(
                    [
                        mixture_crps(
                            out=model(x[i : i + 65_536], p[i : i + 65_536]), z=z[i : i + 65_536]
                        )
                        for i in range(0, len(x), 65_536)
                    ]
                ).numpy()
            crps_mw = crps_z * scale_mw
            result[f"{arm}_s{seed}"] = {
                "crps_mw_all": float(crps_mw.mean()),
                "crps_mw_six": float(crps_mw[six].mean()),
            }
            print(args.cutoff, arm, seed, result[f"{arm}_s{seed}"], flush=True)
    (k.WORK / f"skill_{args.cutoff}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

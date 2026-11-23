# ruff: noqa: E501, PT018
"""Experiment 716: the read-outs of design section 7.5 that need no refit, and the learned-gain tables (WP7).

Usage: run.sh e00716_readouts.py contrasts ARM      # ARM is H (part 1) or HT (part 2)
       run.sh e00716_readouts.py gains ARM FOLD SEED
       run.sh e00716_readouts.py --selftest

The no-refit read-outs are forecast files that `e00716_predict.py --readout NAME` wrote under
`work/fits/<ARM>-<NAME>/`. This module scores each one against its parent arm and against `M*`, at
95%, in the six and the eight, and measures what each fitted PV path learned. Nothing here changes a
verdict. The learned gains are read from the stage 3 state.

**The reachable range of a gain.** AdamW moves a parameter by at most about its rate per optimiser
step, so a gain with rate `lr` that has taken `n` steps can have moved by at most `lr * n`. The
report states each learned gain as a share of that range, and the share of a PV path in the length of
the token it is added to.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_model as pvm
import e00716_score as score
import e00716_train as pvt
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

READOUTS: Final = ("cams", "convention", "nopv", "scale0.7", "scale1.3", "lookahead")
LOOKAHEAD_FOLDS: Final = ("2025-11", "2025-12", "2026-01", "2026-02")
UNUSED_SHARE: Final = 0.05


def reachable_range(*, rate: float, steps: int) -> float:
    """Return the largest distance that a parameter with Adam rate `rate` can move in `steps` optimiser steps."""
    return rate * steps


def token_length_share(*, model: pvm.PvNet, inp: pvm.PvInputs) -> dict[str, float]:
    """Return the mean length of each PV path's output as a share of the mean length of the token it is added to (experiment 713's token before the paths)."""
    model.eval()
    with torch.no_grad():
        base, _ = mdl.MsaNet._tokens(model, inp)  # ty: ignore[invalid-argument-type]
        length = base.norm(dim=-1)
        out = {}
        if model.pv_hist_w1 is not None:
            hist = model.pv_hist_path(inp).norm(dim=-1)
            out["history"] = (
                float((hist[inp.hist_pv_ok]).mean() / length[:, 1:][inp.hist_pv_ok].mean())
                if bool(inp.hist_pv_ok.any())
                else 0.0
            )
        if model.pv_top_w1 is not None:
            top = model.pv_top_path(inp).norm(dim=-1)
            out["target"] = (
                float(top[inp.top_pv_ok].mean() / length[:, 0].mean())
                if bool(inp.top_pv_ok.any())
                else 0.0
            )
    return out


def learned_gains(*, model: pvm.PvNet, steps: int) -> dict:
    """Return `a` and `c` per Primary, the norms of the token-path gains, and each as a share of its reachable range at `steps` steps."""
    out: dict = {}
    for name, rate in (
        ("pv_a", 1e-2),
        ("pv_c", 1e-2),
        ("pv_hist_gamma", 3e-3),
        ("pv_top_gamma", 3e-3),
    ):
        param = getattr(model, name, None)
        if param is None:
            continue
        values = param.detach().cpu().numpy()
        reach = reachable_range(rate=rate, steps=steps)
        out[name] = {
            "values": values.tolist(), "max_abs": float(np.abs(values).max()), "reachable": reach, "share_of_reachable": float(np.abs(values).max() / reach),
            "used": bool(np.abs(values).max() >= UNUSED_SHARE * reach),
        }  # fmt: skip
    return out


def readout_contrasts(*, base: str, months: tuple[str, ...], seeds: tuple[int, ...]) -> dict:
    """Return, at 95%, each read-out against its parent arm and against `M*`, per group (the look-ahead on the dark folds at the six)."""
    star = arms713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    names = [base, star, *[f"{base}-{r}" for r in READOUTS]]
    per_seed, averaged, groups = score.load_all(
        names=names, months=months, seeds=dict.fromkeys(names, seeds)
    )
    out: dict = {}
    for r in READOUTS:
        for ref in (base, star):
            for g in score.GROUPS_COUNTED:
                out[f"{base}-{r} minus {ref}:{g}"] = score.contrast(
                    per_seed=per_seed, averaged=averaged, treatment=f"{base}-{r}", reference=ref, group=g, groups=groups, level=score.LEVEL_SECONDARY, n_seeds=len(seeds),
                )  # fmt: skip
    for g in score.GROUPS_COUNTED:
        ps = score.apply_slice(per_seed, solar=False, folds=LOOKAHEAD_FOLDS)
        av = score.apply_slice(averaged, solar=False, folds=LOOKAHEAD_FOLDS)
        out[f"look-ahead folds {base}-lookahead minus {base}:{g}"] = score.contrast(
            per_seed=ps, averaged=av, treatment=f"{base}-lookahead", reference=base, group=g, groups=groups, level=score.LEVEL_SECONDARY, n_seeds=len(seeds),
        )  # fmt: skip
    return out


def load_state(*, arm: str, month: str, seed: int) -> pvm.PvNet:
    """Load the stage 3 state of an arm on the CPU."""
    spec = arms.ARMS[arm]
    model = pvm.PvNet(arm=spec.net, seed=seed)  # ty: ignore[invalid-argument-type]
    model.load_state_dict(
        torch.load(
            arms.model_file(
                data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed, stage=3
            ),
            map_location="cpu",
        )
    )
    return model


def steps_of(*, arm: str, month: str, seed: int) -> int:
    """Return the optimiser steps of stage 3 recorded in the fit log (passes times samples over 256), or 0 when the log lacks them."""
    path = arms.log_file(
        data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed, kind="fit"
    )
    log = json.loads(path.read_text())
    entry = log["stages"][-1]
    return int(entry["passes"] * -(-entry["pairs"] // pvt.tr.BATCH))


def write(*, name: str, obj: object, part: int) -> Path:
    """Write `score/<name>.json` of a part."""
    score.write(name=name, obj=obj, score_dir=score.score_dir_of(part=part))
    return score.score_dir_of(part=part) / f"{name}.json"


def selftest() -> int:
    """Check the gain table and the token-length share on a random `HT` network."""
    import e00716_check_model as ckm

    model = pvm.PvNet(arm="HT", seed=0)
    inp = ckm.random_inputs(seed=1, b=4)
    model.calibrate_scale(batches=[inp])
    zero = token_length_share(model=model, inp=inp)
    assert zero == {"history": 0.0, "target": 0.0} or all(v < 1e-9 for v in zero.values()), (
        "the paths start at zero"
    )
    with torch.no_grad():
        model.pv_hist_gamma.add_(0.5)
        model.pv_top_gamma.add_(0.2)
        model.pv_c.add_(-1.0)
    share = token_length_share(model=model, inp=inp)
    assert share["history"] > 0 and share["target"] > 0, share
    gains = learned_gains(model=model, steps=100)
    assert gains["pv_c"]["used"] and gains["pv_c"]["share_of_reachable"] > 0.5, gains["pv_c"]
    assert not learned_gains(model=pvm.PvNet(arm="HT", seed=0), steps=100)["pv_a"]["used"]
    assert reachable_range(rate=1e-2, steps=100) == 1.0
    print(
        "token-length share and learned-gain tables: ok", {k: round(v, 3) for k, v in share.items()}
    )
    print("selftest passed")
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("contrasts", "gains"))
    parser.add_argument("arm", nargs="?")
    parser.add_argument("month", nargs="?")
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    part = arms.ARMS[args.arm].part
    if args.command == "contrasts":
        seeds = score.available_seeds(names=[args.arm], months=arms.ALL_FOLDS)
        write(
            name=f"readouts_{args.arm}",
            obj=readout_contrasts(base=args.arm, months=arms.ALL_FOLDS, seeds=seeds),
            part=part,
        )
        return 0
    model = load_state(arm=args.arm, month=args.month, seed=args.seed)
    write(
        name=f"gains_{args.arm}_{args.month}_s{args.seed}",
        obj=learned_gains(
            model=model, steps=steps_of(arm=args.arm, month=args.month, seed=args.seed)
        ),
        part=part,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ruff: noqa: E501
"""Experiment 661, step 5: the explicit-minus-free contrast on native, 3-hour, and 6-hour known weather.

CRPS, the day-block bootstrap, and the sunny-midday mask are experiment 639's (`e00639_analyse`,
`e00639_common`). The rows are experiment 639's real-world test rows (the six Primaries, the 60 days
after each cutoff); the sunny mask comes from experiment 639's stored undegraded clear-sky index so
that every variant scores the same rows. A positive contrast favours the explicit network.

Usage: python e00661_analyse5.py --cutoffs 2026-06 --seeds 0 1 2
"""

import argparse
import json
from typing import Final

import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00639_common as k  # ty: ignore[unresolved-import]
import e00661_degrade as g
import e00661_fit5 as f5
import numpy as np

N_CONTRASTS: Final = 12
LEVEL: Final = 1.0 - a639.ALPHA / N_CONTRASTS


def load_variant(*, variant: str, cutoffs: list[str], seeds: list[int]) -> dict:
    """Return per-seed CRPS of arms `free` and `cal` for a variant ("native" is experiment 639's own)."""
    out: dict = {
        "free": [],
        "cal": [],
        "y": [],
        "kt": [],
        "hour": [],
        "t_us": [],
        "primary": [],
        "day": [],
    }
    for ci, cutoff in enumerate(cutoffs):
        ref = np.load(k.test_rows_path(cutoff=cutoff, world="real"))
        six = ref["inc"] > 0
        if variant == "native":
            rows = ref
        else:
            rows = np.load(f5.WORK5 / variant / "predictions" / cutoff / "real_rows.npz")
            assert np.array_equal(rows["primary"], ref["primary"]), "primary keys differ from 639's"
            assert np.array_equal(rows["t_us"], ref["t_us"]), "time keys differ from 639's"
        for arm in ("free", "cal"):
            scores = []
            for seed in seeds:
                path = (
                    k.prediction_path(cutoff=cutoff, world="real", arm=arm, seed=seed)
                    if variant == "native"
                    else f5.WORK5 / variant / "predictions" / cutoff / f"real_{arm}_s{seed}.npz"
                )
                q = np.load(path)["q"][six]
                scores.append(
                    100.0
                    * a639.crps(actual=rows["y"][six].astype(np.float64), q=q.astype(np.float64))
                )
            out[arm].append(np.stack(scores))
        for name in ("y", "kt", "hour", "t_us", "primary"):
            out[name].append(
                ref[name][six]
            )  # the mask always comes from experiment 639's undegraded rows
        out["day"].append(ci * 100_000 + ref["t_us"][six] // k.k634.DAY_US % 100_000)
    res = {
        a: np.concatenate(v, axis=1) if a in ("free", "cal") else np.concatenate(v)
        for a, v in out.items()
    }
    res["sunny"] = (
        (res["kt"] >= k.SUNNY_KT)
        & (res["hour"] >= k.SUNNY_HOURS[0])
        & (res["hour"] <= k.SUNNY_HOURS[1])
    )
    return res


def main() -> None:
    """Print the contrasts and write them to JSON."""
    p = argparse.ArgumentParser()
    p.add_argument("--cutoffs", nargs="+", default=["2026-06"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = p.parse_args()
    data = {
        v: load_variant(variant=v, cutoffs=a.cutoffs, seeds=a.seeds) for v in ("native", *g.WIDTH_H)
    }
    res: dict = {"cutoffs": a.cutoffs, "seeds": a.seeds, "level": LEVEL, "variants": {}}
    for v, d in data.items():
        gain = (d["free"] - d["cal"]).mean(axis=0)
        r = {}
        for name, mask in {
            "sunny_middays": d["sunny"],
            "all_half_hours": np.ones_like(d["sunny"]),
        }.items():
            mean, lo, hi = a639.bootstrap(diff=gain[mask], day=d["day"][mask], level=LEVEL)
            r[name] = {
                "free_minus_explicit": mean,
                "lo": lo,
                "hi": hi,
                "n_rows": int(mask.sum()),
                "free_crps": float(d["free"][:, mask].mean()),
                "explicit_crps": float(d["cal"][:, mask].mean()),
            }
        r["per_seed_gain_sunny"] = [
            float((d["free"][s] - d["cal"][s])[d["sunny"]].mean()) for s in range(len(a.seeds))
        ]
        res["variants"][v] = r
        print(
            v,
            {n: round(x["free_minus_explicit"], 3) for n, x in r.items() if isinstance(x, dict)},
            r["per_seed_gain_sunny"],
            flush=True,
        )
    base = data["native"]
    for v in g.WIDTH_H:
        d = data[v]
        diff = ((d["free"] - d["cal"]) - (base["free"] - base["cal"])).mean(axis=0)
        mean, lo, hi = a639.bootstrap(diff=diff[d["sunny"]], day=d["day"][d["sunny"]], level=LEVEL)
        res["variants"][v]["gain_minus_native_gain_sunny"] = {"diff": mean, "lo": lo, "hi": hi}
        share = (
            res["variants"][v]["sunny_middays"]["free_minus_explicit"]
            / res["variants"]["native"]["sunny_middays"]["free_minus_explicit"]
        )
        res["variants"][v]["share_of_native_gain"] = share
        print(
            v,
            "gain minus native gain",
            round(mean, 3),
            round(lo, 3),
            round(hi, 3),
            "share of native gain",
            round(share, 3),
        )
    out = g.WORK / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"step5_{'_'.join(a.cutoffs)}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

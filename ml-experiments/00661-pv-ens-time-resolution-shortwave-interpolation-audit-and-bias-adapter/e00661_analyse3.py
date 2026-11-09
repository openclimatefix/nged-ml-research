# ruff: noqa: ANN001
"""Experiment 661, Part 3: score the adapter arms against experiment 644's stored arms.

Rows, masks, CRPS, bootstrap, and the seed-split bar are experiment 644's (`e00644_analyse`). A
contrast is a control's CRPS minus arm A's, so a positive value favours arm A. Controls: `a` the
free network, `b` the calibrated explicit network, `d` the 14-day tracker, `A0` arm A without its
adapter, and `p` the capacity-permuted placebo of experiment 644.

Usage: python e00661_analyse3.py --seeds 0 1 2 3 4
"""

import argparse
import json
from typing import Final

import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00644_analyse as a644  # ty: ignore[unresolved-import]
import e00644_fit as fit  # ty: ignore[unresolved-import]
import e00661_adapter as ad
import numpy as np

N_CONTRASTS: Final = 12
LEVEL: Final = 1.0 - a639.ALPHA / N_CONTRASTS
a644.LEVEL = LEVEL  # the contrast helpers of experiment 644 read this module-level value


def adapter_quantiles(*, arm: str, seed: int, rows: dict) -> np.ndarray:
    """Return arm A or A0's stored quantiles in the order of `rows`."""
    out = np.empty((len(rows["y"]), 11), dtype=np.float32)
    for i, (cutoff, month) in enumerate(rows["files"]):
        out[rows["file"] == i] = np.load(ad.WORK / "fits" / cutoff / f"{month}_{arm}_s{seed}.npz")[
            "q"
        ]
    return out


def crps_of(*, q_fn, rows: dict, seeds: list[int]) -> np.ndarray:
    """Return CRPS (points) per seed and row."""
    return np.stack(
        [
            100.0 * a639.crps(actual=rows["y"].astype(np.float64), q=q_fn(s).astype(np.float64))
            for s in seeds
        ]
    )


def main() -> None:
    """Print and store the Part 3 scores and contrasts."""
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--arms", nargs="+", default=["adapter", "nested"])
    a = p.parse_args()
    seeds = a.seeds
    rows = a644.load(world="real")
    masks = a644._masks(rows=rows)
    scores = {
        arm: a644.arm_crps(world="real", arm=arm, rows=rows, seeds=tuple(seeds))
        for arm in ("a", "b", "d", "p")
    }
    for arm in a.arms:
        scores[arm] = crps_of(
            q_fn=lambda s, arm=arm: adapter_quantiles(arm=arm, seed=s, rows=rows),
            rows=rows,
            seeds=seeds,
        )
    out: dict = {
        "seeds": seeds,
        "level": LEVEL,
        "scores": {},
        "contrasts": {},
        "bars": {},
        "tails": {},
    }
    for name, mask in masks.items():
        out["scores"][name] = {arm: float(s[:, mask].mean()) for arm, s in scores.items()}
    for band in a644.BANDS:
        name = f"sunny_{band}"
        out["bars"][name] = max(
            a644.MDE_639["real"], a644.null_bar(scores_a=scores["a"], rows=rows, mask=masks[name])
        )
        out["contrasts"][name] = {}
        for ctrl in ("a", "b", "d", "p", *[x for x in a.arms if x != "adapter"]):
            out["contrasts"][name][f"{ctrl}_minus_adapter"] = a644._contrast(
                rows=rows, control=scores[ctrl], expl=scores["adapter"], mask=masks[name]
            )
    out["contrasts"]["all_half_hours"] = {
        "a_minus_adapter": a644._contrast(
            rows=rows, control=scores["a"], expl=scores["adapter"], mask=masks["all"]
        )
    }
    sun = rows["sunny"] & rows["in_window"]
    qs = {arm: [adapter_quantiles(arm=arm, seed=s, rows=rows) for s in seeds] for arm in a.arms}
    qs["a"] = [a644.arm_quantiles(world="real", arm="a", seed=s, rows=rows) for s in seeds]
    qs["b"] = [a644.arm_quantiles(world="real", arm="b", seed=s, rows=rows) for s in seeds]
    for arm, lst in qs.items():
        q50 = np.mean([q[:, 5] for q in lst], axis=0)
        out["tails"][arm] = {
            "mae_median_points": float(np.abs(q50 - rows["y"])[sun].mean() * 100),
            "coverage_q05_q95": float(
                np.mean(
                    [((rows["y"] >= q[:, 1]) & (rows["y"] <= q[:, 9]))[sun].mean() for q in lst]
                )
            ),
            "pinball": {
                str(k639.LEVELS[i]): float(
                    np.mean(
                        [
                            100
                            * np.where(
                                rows["y"] - q[:, i] >= 0,
                                k639.LEVELS[i] * (rows["y"] - q[:, i]),
                                (k639.LEVELS[i] - 1) * (rows["y"] - q[:, i]),
                            )[sun].mean()
                            for q in lst
                        ]
                    )
                )
                for i in (0, 1, 9, 10)
            },
            "coverage_q01": float(np.mean([(rows["y"] <= q[:, 0])[sun].mean() for q in lst])),
            "coverage_q99": float(np.mean([(rows["y"] <= q[:, 10])[sun].mean() for q in lst])),
        }
    mix = {
        arm: np.mean(
            [np.load(f)["mix"] for f in sorted((ad.WORK / "fits").glob(f"*/*_{arm}_s*.npz"))],
            axis=0,
        )
        for arm in a.arms
    }
    out["mixture_mean_over_fits"] = {arm: v.tolist() for arm, v in mix.items()}
    for name in ("sunny_d1-2", "sunny_d3-5", "sunny_d6-10"):
        c = out["contrasts"][name]["a_minus_adapter"]
        print(
            name,
            "free minus adapter",
            round(c["diff"], 3),
            round(c["lo"], 3),
            round(c["hi"], 3),
            "bar",
            round(out["bars"][name], 3),
            "| b minus adapter",
            round(out["contrasts"][name]["b_minus_adapter"]["diff"], 3),
            flush=True,
        )
    fit.WORK.mkdir(parents=True, exist_ok=True)
    (ad.WORK / "analysis").mkdir(parents=True, exist_ok=True)
    (ad.WORK / "analysis" / "part3.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

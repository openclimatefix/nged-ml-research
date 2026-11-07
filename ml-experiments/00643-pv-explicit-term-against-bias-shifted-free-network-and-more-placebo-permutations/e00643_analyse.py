# ruff: noqa: E501, ANN001, ANN202, PERF401
"""Experiment 643: score every arm and print the planned contrasts.

CRPS and the day-block bootstrap are experiment 639's. A contrast is the control's CRPS minus
`cal`'s, so a positive value favours `cal`. Intervals are at the Bonferroni level over `N_CONTRASTS`.

Usage: python e00643_analyse.py contrasts [--seeds 0 1 2 3 4]
"""

import argparse
import json

import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00643_common as k
import numpy as np

LEVEL = 1.0 - a639.ALPHA / k.N_CONTRASTS


def _arm_scores(*, world: str, arm: str, seeds: tuple[int, ...]) -> np.ndarray:
    """Return CRPS (points) per seed and six-Primary test row, stacked over cutoffs: shape (seeds, rows).

    Arm `fhind` shifts each free network's sunny-midday forecast by its mean median error on the test window (a hindsight bound).
    """
    out = []
    for seed in seeds:
        parts = []
        for cutoff in k639.CUTOFFS:
            z = np.load(k639.test_rows_path(cutoff=cutoff, world=world))
            six = z["inc"] > 0
            arm_read = "free" if arm == "fhind" else arm
            q = np.load(k.prediction_path(cutoff=cutoff, world=world, arm=arm_read, seed=seed))[
                "q"
            ][six].astype(np.float64)
            y = z["y"][six].astype(np.float64)
            if arm == "fhind":
                sun = _sunny_mask(z=z)[six]
                prim = z["primary"][six]
                shifted = q.copy()
                for p in np.unique(prim):
                    m = sun & (prim == p)
                    shifted[m] -= (q[m, 5] - y[m]).mean()
                q = shifted
            parts.append(100.0 * a639.crps(actual=y, q=q))
        out.append(np.concatenate(parts))
    return np.stack(out)


def _sunny_mask(*, z) -> np.ndarray:
    return (
        (z["kt"] >= k639.SUNNY_KT)
        & (z["hour"] >= k639.SUNNY_HOURS[0])
        & (z["hour"] <= k639.SUNNY_HOURS[1])
    )


def _contrast(*, rows, control: np.ndarray, cal: np.ndarray, subset: str) -> dict:
    """Return control minus `cal` (mean over seeds) on a subset, with the interval and per-seed spread."""
    mask = rows["sunny"] if subset == "sunny" else np.ones(len(rows["y"]), dtype=bool)
    diff = (control - cal).mean(axis=0)
    mean, lo, hi = a639.bootstrap(diff=diff[mask], day=rows["day"][mask], level=LEVEL)
    per_seed = (control - cal)[:, mask].mean(axis=1)
    return {
        "diff": mean,
        "lo": lo,
        "hi": hi,
        "n_rows": int(mask.sum()),
        "per_seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
    }


def main() -> None:
    """Print every contrast; write them to the analysis folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("contrasts",))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.STAGE1_SEEDS))
    parser.add_argument("--tag", default="stage1")
    parser.add_argument(
        "--perm-only", action="store_true", help="stage 2: only the placebo, step world"
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    results = []
    for world in ("step",) if args.perm_only else k639.WORLDS:
        rows = a639.load(world=world)
        cal = _arm_scores(world=world, arm="cal", seeds=seeds)
        absolute = {"cal": float(cal.mean(axis=0)[rows["sunny"]].mean())}
        controls = [] if args.perm_only else ["free", "fbias", "foff", "fhind"]
        for arm in controls:
            scores = _arm_scores(world=world, arm=arm, seeds=seeds)
            absolute[arm] = float(scores.mean(axis=0)[rows["sunny"]].mean())
            for subset in ("sunny", "all"):
                results.append(
                    {
                        "world": world,
                        "arm": arm,
                        "subset": subset,
                        **_contrast(rows=rows, control=scores, cal=cal, subset=subset),
                    }
                )
        if world == "step":
            perms = {
                f"perm{s}": _arm_scores(world=world, arm=f"perm{s}", seeds=seeds) for s in k.SHIFTS
            }
            for name, scores in perms.items():
                absolute[name] = float(scores.mean(axis=0)[rows["sunny"]].mean())
                results.append(
                    {
                        "world": world,
                        "arm": name,
                        "subset": "sunny",
                        **_contrast(rows=rows, control=scores, cal=cal, subset="sunny"),
                    }
                )
            mean_perm = np.mean(list(perms.values()), axis=0)
            results.append(
                {
                    "world": world,
                    "arm": "perm_mean",
                    "subset": "sunny",
                    **_contrast(rows=rows, control=mean_perm, cal=cal, subset="sunny"),
                }
            )
            results.append(
                {
                    "world": world,
                    "arm": "perm_mean",
                    "subset": "all",
                    **_contrast(rows=rows, control=mean_perm, cal=cal, subset="all"),
                }
            )
        results.append(
            {"world": world, "arm": "ABSOLUTE_SUNNY_CRPS", "subset": "sunny", "absolute": absolute}
        )
    out_dir = k.WORK / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"contrasts_{args.tag}.json").write_text(json.dumps(results, indent=1))
    for r in results:
        if "absolute" in r:
            print(
                r["world"],
                "absolute sunny CRPS:",
                {a: round(v, 3) for a, v in r["absolute"].items()},
            )
            continue
        bar = k.MDE[r["world"]]
        print(
            f"{r['world']:5s} {r['arm']:10s} minus cal {r['subset']:5s} n={r['n_rows']:6d} {r['diff']:+.4f} "
            f"[{r['lo']:+.4f}, {r['hi']:+.4f}] seed sd {r['per_seed_sd']:.4f}  gain-by-rule={r['lo'] > 0 and r['diff'] >= bar}"
        )


if __name__ == "__main__":
    main()

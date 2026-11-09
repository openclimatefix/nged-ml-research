# ruff: noqa: E501
"""Experiment 664: score every arm, print the planned contrasts, and store the numbers the figures use.

Rows, CRPS, and the day-block bootstrap are experiment 644's (`e00644_analyse`, `e00639_analyse`).
A contrast is the control's CRPS minus the arm's, so a positive value favours the arm (points of
effective capacity). The planned contrasts are in `CONTRASTS`; their Bonferroni level is
`LEVEL`. Arms are listed in `e00664_run.ARMS`.

Usage: python e00664_analyse.py contrasts [--seeds 0 1 2 3 4]
"""

import argparse
import json
from typing import Final

import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00644_analyse as a644  # ty: ignore[unresolved-import]
import e00664_run as run
import numpy as np

OUT: Final = run.WORK / "analysis"
BANDS: Final = a644.BANDS
BARS: Final[dict[str, float]] = {"d1-2": 0.084, "d3-5": 0.093, "d6-10": 0.123}
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"),
    ("a", "cc"), ("a", "dcc"), ("b", "cc"), ("b", "dcc"), ("c", "p"), ("c", "s"), ("c", "c0"),
)  # fmt: skip
"""(control, arm) pairs, per lead band (sunny middays). The last three are checks: placebos must lose to `c`, and `c0` shows what the member draw adds to the sampling."""
N_CONTRASTS: Final = len(CONTRASTS) * len(BANDS)
LEVEL: Final = 1.0 - a639.ALPHA / N_CONTRASTS
PIT_BINS: Final = 10


def load_arm(*, arm: str, kind: str, seed: int, rows: dict) -> np.ndarray:
    """Return `q` (rows, 11) or `pit` (rows) of one arm and seed in the order of `rows`."""
    shape = (len(rows["y"]), 11) if kind == "q" else (len(rows["y"]),)
    out = np.empty(shape, dtype=np.float32)
    for i, (cutoff, month) in enumerate(rows["files"]):
        z = np.load(run.out_path(cutoff=cutoff, month=month, seed=seed))
        out[rows["file"] == i] = z[f"{kind}_{arm}"]
    return out


def pit_hist(*, pit: np.ndarray, mask: np.ndarray) -> tuple[list[float], float]:
    """Return the share of PIT values per bin and the total-variation distance from uniform."""
    h = np.histogram(pit[mask], bins=PIT_BINS, range=(0, 1))[0] / max(int(mask.sum()), 1)
    return h.tolist(), float(0.5 * np.abs(h - 1.0 / PIT_BINS).sum())


def contrasts(*, seeds: tuple[int, ...]) -> dict:
    """Return scores, planned contrasts, coverage, tails, and PIT summaries."""
    rows = a644.load(world="real")
    masks = a644._masks(rows=rows)
    q = {arm: [load_arm(arm=arm, kind="q", seed=s, rows=rows) for s in seeds] for arm in run.ARMS}
    pit = {
        arm: [load_arm(arm=arm, kind="pit", seed=s, rows=rows) for s in seeds] for arm in run.ARMS
    }
    scores = {
        arm: np.stack([100.0 * a639.crps(actual=rows["y"].astype(np.float64), q=x.astype(np.float64)) for x in q[arm]])
        for arm in run.ARMS
    }  # fmt: skip
    out: dict = {"seeds": list(seeds), "level": LEVEL, "n_contrasts": N_CONTRASTS, "bars": BARS, "scores": {}, "contrasts": {}, "coverage": {}, "tails": {}, "pit": {}, "by_primary": {}}  # fmt: skip
    for name, mask in masks.items():
        out["scores"][name] = {arm: float(scores[arm][:, mask].mean()) for arm in run.ARMS}
    for band in BANDS:
        name = f"sunny_{band}"
        out["contrasts"][name] = {}
        for ctl, arm in CONTRASTS:
            diff = (scores[ctl] - scores[arm]).mean(axis=0)
            mean, lo, hi = a639.bootstrap(
                diff=diff[masks[name]], day=rows["day"][masks[name]], level=LEVEL
            )
            out["contrasts"][name][f"{ctl}-{arm}"] = {
                "diff": mean, "lo": lo, "hi": hi,
                "per_seed": (scores[ctl] - scores[arm])[:, masks[name]].mean(axis=1).tolist(),
            }  # fmt: skip
    for ctl, arm in (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")):
        diff = (scores[ctl] - scores[arm]).mean(axis=0)
        mean, lo, hi = a639.bootstrap(
            diff=diff[masks["all"]], day=rows["day"][masks["all"]], level=LEVEL
        )
        out["contrasts"].setdefault("all_half_hours", {})[f"{ctl}-{arm}"] = {
            "diff": mean,
            "lo": lo,
            "hi": hi,
        }
    # Coverage, width, tails and PIT on sunny middays by band, and on all half-hours.
    slices = {
        **{f"sunny_{b}": masks[f"sunny_{b}"] for b in BANDS},
        "sunny_all": masks["sunny_all"],
        "all": masks["all"],
    }
    y = rows["y"]
    for sname, mask in slices.items():
        out["coverage"][sname], out["tails"][sname], out["pit"][sname] = {}, {}, {}
        for arm in run.ARMS:
            qs = q[arm]
            cov = np.mean([((y >= x[:, 1]) & (y <= x[:, 9]))[mask].mean() for x in qs])
            width = np.mean([100 * (x[:, 9] - x[:, 1])[mask].mean() for x in qs])
            t = {}
            for i, lev in ((0, "q01"), (1, "q05"), (9, "q95"), (10, "q99")):
                d = [y - x[:, i] for x in qs]
                t[f"pinball_{lev}"] = float(np.mean([100 * np.where(dd >= 0, k639.LEVELS[i] * dd, (k639.LEVELS[i] - 1) * dd)[mask].mean() for dd in d]))  # fmt: skip
                t[f"coverage_{lev}"] = float(np.mean([(y <= x[:, i])[mask].mean() for x in qs]))
            out["coverage"][sname][arm] = {
                "coverage_5_95": float(cov),
                "mean_width_5_95": float(width),
            }
            out["tails"][sname][arm] = t
            hist = [pit_hist(pit=p, mask=mask) for p in pit[arm]]
            out["pit"][sname][arm] = {"hist": np.mean([h for h, _ in hist], axis=0).tolist(), "distance": float(np.mean([d for _, d in hist]))}  # fmt: skip
    # CRPS by Primary on sunny middays, day 1-2 (the table the take-home may cite).
    sun = masks["sunny_all"]
    for p in np.unique(rows["primary"]):
        mp = sun & (rows["primary"] == p)
        out["by_primary"][int(p)] = {arm: float(scores[arm][:, mp].mean()) for arm in run.ARMS}
    out["by_lead_day"] = {
        str(dd): {arm: float(scores[arm][:, rows["sunny"] & (rows["lead_day"] == dd) & rows["in_window"]].mean()) for arm in run.ARMS}
        for dd in range(1, 11)
    }  # fmt: skip
    # Median minus mean forecast per arm (audit trigger): mean of the 11 quantiles is not the mean forecast, so report the median-to-quantile-mean gap.
    out["median_minus_quantile_mean"] = {arm: float(np.mean([100 * (x[:, 5] - x.mean(axis=1))[masks["all"]].mean() for x in q[arm]])) for arm in run.ARMS}  # fmt: skip
    out["mae_median_sunny"] = {arm: float(np.mean([100 * np.abs(x[:, 5] - y)[masks["sunny_all"]].mean() for x in q[arm]])) for arm in run.ARMS}  # fmt: skip
    # Validation statistics (correlations, cell sizes, fallbacks).
    out["val"] = [json.loads(run.val_path(cutoff=c, month=m, seed=s).read_text()) | {"cutoff": c, "month": m, "seed": s} for c, m in rows["files"] for s in seeds]  # fmt: skip
    out["n_rows"] = len(y)
    out["share_without_realised_kt"] = float(rows["share_without_realised_kt"])
    return out


def main() -> None:
    """Print and store the contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("contrasts",))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k639.SEEDS))
    a = parser.parse_args()
    res = contrasts(seeds=tuple(a.seeds))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contrasts.json").write_text(json.dumps(res, indent=1))
    for name in ("sunny_d1-2", "sunny_d3-5", "sunny_d6-10", "all"):
        print(name, {k: round(v, 3) for k, v in res["scores"][name].items()})
    for band in BANDS:
        for key, v in res["contrasts"][f"sunny_{band}"].items():
            print(
                f"{band} {key}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] (bar {BARS[band]})"
            )
    for sname in ("sunny_d1-2", "sunny_all"):
        print(
            sname,
            "coverage",
            {k: round(v["coverage_5_95"], 3) for k, v in res["coverage"][sname].items()},
        )
        print(
            sname,
            "PIT distance",
            {k: round(v["distance"], 3) for k, v in res["pit"][sname].items()},
        )


if __name__ == "__main__":
    main()

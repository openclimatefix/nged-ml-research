# ruff: noqa: E501
"""Experiment 655: score every arm and print the planned contrasts.

Rows, CRPS, and the day-block bootstrap are experiment 644's (`e00644_analyse`, `e00639_analyse`).
A contrast is the control's CRPS minus the arm's, so a positive value favours the arm (points of
capacity). Arms: `a` free and `b` experiment 644's explicit network (both read from experiment
644's fits); `k` (c0), `c`, `d`, `e`, `p`, `s` from `e00655_fit`; `f` experiment 644's tracker
(observed sunniness); `tb`, `tc` the trackers by forecast sunniness (binary and continuous);
`tbs`, `tcs` the same corrections shuffled across days.

Usage: python e00655_analyse.py contrasts [--seeds 0 1 2 3 4]
"""

import argparse
import json
from typing import Final

import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00644_analyse as a644  # ty: ignore[unresolved-import]
import e00655_fit as fit
import numpy as np

OUT: Final = fit.WORK / "analysis"
BANDS: Final = a644.BANDS
N_CONTRASTS: Final = 40
LEVEL: Final = 1.0 - a639.ALPHA / N_CONTRASTS
BARS: Final[dict[str, float]] = {"d1-2": 0.084, "d3-5": 0.093, "d6-10": 0.123}
"""Experiment 644's real-month bars (the larger of experiment 639's value and the seed-split null)."""
GAIN_644: Final[dict[str, float]] = {"d1-2": 0.34, "d3-5": 0.46, "d6-10": 0.57}
FIT_ARMS: Final = ("k", "c", "d", "e", "p", "s")
KT_LO, KT_HI = 0.5, 0.8
"""The continuous weight is `clip((kt - 0.5) / 0.3, 0, 1)` on the ensemble-mean clear-sky index."""
SHUFFLE_SEED: Final = 655


def _fit_quantiles(*, arm: str, seed: int, rows: dict) -> np.ndarray:
    out = np.empty((len(rows["y"]), 11), dtype=np.float32)
    for i, (cutoff, month) in enumerate(rows["files"]):
        path = fit.fits_path(cutoff=cutoff, month=month, arm=arm, seed=seed).with_suffix(".npz")
        out[rows["file"] == i] = np.load(path)["q"]
    return out


def _weight(*, rows: dict, kind: str) -> np.ndarray:
    """Return the per-row weight of a tracker: 0 or 1 (`bin`), or the continuous ramp (`cont`)."""
    midday = rows["midday"].astype(np.float64)
    if kind == "bin":
        return midday * (rows["kt_forecast"] >= k639.SUNNY_KT)
    return midday * np.clip((rows["kt_forecast"] - KT_LO) / (KT_HI - KT_LO), 0.0, 1.0)


def _corrections(*, q: np.ndarray, rows: dict, w: np.ndarray) -> np.ndarray:
    """Return, per row, the weighted mean day-1 error over the 14 dates before the row's origin date.

    The history holds the rows with lead below 24 hours, weighted by `w`, and a date counts only
    if it ends before the origin date (`valid_date < origin_date`). Fewer than
    `MIN_HISTORY_DAYS` dates with weight give no correction (zero).
    """
    err = q[:, 5] - rows["y"]
    corr = np.zeros(len(err))
    hist = (w > 0) & (rows["lead"] < 24)
    for p in np.unique(rows["primary"]):
        m = rows["primary"] == p
        h = m & hist
        dates, inv = np.unique(rows["valid_date"][h], return_inverse=True)
        s = np.bincount(inv, weights=w[h] * err[h])
        n = np.bincount(inv, weights=w[h])
        target = m & (w > 0)
        for od in np.unique(rows["origin_date"][target]):
            win = (dates >= od - a644.TRAIL_DAYS) & (dates < od)
            if win.sum() >= a644.MIN_HISTORY_DAYS and n[win].sum() > 0:
                corr[target & (rows["origin_date"] == od)] = s[win].sum() / n[win].sum()
    return corr


def _shuffled(*, corr: np.ndarray, rows: dict, w: np.ndarray) -> np.ndarray:
    """Return the corrections with the per-origin-date value permuted across dates (same Primary, same cutoff)."""
    out = np.zeros_like(corr)
    rng = np.random.default_rng(SHUFFLE_SEED)
    for cut in np.unique(rows["cutoff"]):
        for p in np.unique(rows["primary"]):
            m = (rows["primary"] == p) & (rows["cutoff"] == cut) & (w > 0)
            od = rows["origin_date"][m]
            dates = np.unique(od)
            vals = np.array([corr[m][od == d][0] for d in dates])
            perm = rng.permutation(len(dates))
            lookup = dict(zip(dates.tolist(), vals[perm].tolist(), strict=True))
            out[m] = [lookup[d] for d in od.tolist()]
    return out


def arm_quantiles(*, arm: str, seed: int, rows: dict) -> np.ndarray:
    """Return the quantiles (rows, 11) of any arm."""
    if arm in FIT_ARMS:
        return _fit_quantiles(arm=arm, seed=seed, rows=rows)
    if arm in ("a", "b"):
        return a644._quantiles(world="real", arm=arm, seed=seed, rows=rows)
    base = a644._quantiles(world="real", arm="a", seed=seed, rows=rows)
    if arm == "f":
        return a644._tracker(q=base, rows=rows)
    kind = "bin" if arm in ("tb", "tbs") else "cont"
    w = _weight(rows=rows, kind=kind)
    corr = _corrections(q=base, rows=rows, w=w)
    if arm in ("tbs", "tcs"):
        corr = _shuffled(corr=corr, rows=rows, w=w)
    return base - (w * corr)[:, None]


def arm_crps(*, arm: str, rows: dict, seeds: tuple[int, ...]) -> np.ndarray:
    """Return CRPS (points) per seed and row: shape (seeds, rows)."""
    return np.stack(
        [
            100.0
            * a639.crps(
                actual=rows["y"].astype(np.float64),
                q=arm_quantiles(arm=arm, seed=s, rows=rows).astype(np.float64),
            )
            for s in seeds
        ]
    )


def contrast(*, rows: dict, control: np.ndarray, arm: np.ndarray, mask: np.ndarray) -> dict:
    """Return the control-minus-arm mean with its day-block interval at the Bonferroni level."""
    diff = (control - arm).mean(axis=0)
    mean, lo, hi = a639.bootstrap(diff=diff[mask], day=rows["day"][mask], level=LEVEL)
    per_seed = (control - arm)[:, mask].mean(axis=1)
    return {
        "diff": mean,
        "lo": lo,
        "hi": hi,
        "n_rows": int(mask.sum()),
        "per_seed": per_seed.tolist(),
    }


PART1: Final[tuple[tuple[str, str], ...]] = (
    ("a", "d"), ("b", "d"), ("c", "d"), ("k", "c"), ("k", "d"), ("d", "e"), ("p", "d"), ("s", "d"),
)  # fmt: skip
"""(control, arm) pairs of part 1, per lead band."""
PART2: Final[tuple[tuple[str, str], ...]] = (
    ("a", "tb"), ("a", "tc"), ("tbs", "tb"), ("tcs", "tc"), ("f", "tb"),
)  # fmt: skip
"""(control, arm) pairs of part 2, per lead band."""


def contrasts(*, seeds: tuple[int, ...]) -> dict:
    """Return every planned contrast, scores, tails, and checks."""
    rows = a644.load(world="real")
    masks = a644._masks(rows=rows)
    arms = ("a", "b", "k", "c", "d", "e", "p", "s", "f", "tb", "tc", "tbs", "tcs")
    scores = {arm: arm_crps(arm=arm, rows=rows, seeds=seeds) for arm in arms}
    disagree = rows["midday"] & (rows["sunny"] != rows["sunny_forecast"]) & rows["in_window"]
    out: dict = {
        "seeds": list(seeds),
        "level": LEVEL,
        "n_contrasts": N_CONTRASTS,
        "bars": BARS,
        "scores": {},
        "contrasts": {},
        "extra": {},
    }
    slices = {**masks, "disagree": disagree}
    for name, mask in slices.items():
        out["scores"][name] = {arm: float(scores[arm][:, mask].mean()) for arm in arms}
    for band in BANDS:
        name = f"sunny_{band}"
        out["contrasts"][name] = {
            f"{ctl}-{arm}": contrast(
                rows=rows, control=scores[ctl], arm=scores[arm], mask=masks[name]
            )
            for ctl, arm in (*PART1, *PART2)
        }
        dis = disagree & masks[f"allhh_{band}"]
        out["extra"][f"disagree_{band}"] = {
            f"{ctl}-{arm}": contrast(rows=rows, control=scores[ctl], arm=scores[arm], mask=dis)
            for ctl, arm in (("a", "tb"), ("a", "tc"), ("a", "f"))
        }
        out["extra"][f"lopo_{band}"] = {
            int(p): {
                f"{ctl}-{arm}": contrast(
                    rows=rows,
                    control=scores[ctl],
                    arm=scores[arm],
                    mask=masks[name] & (rows["primary"] != p),
                )["diff"]
                for ctl, arm in (("a", "d"), ("a", "tb"), ("k", "c"), ("c", "d"))
            }
            for p in np.unique(rows["primary"])
        }
    out["contrasts"]["all_half_hours"] = {
        "a-d": contrast(rows=rows, control=scores["a"], arm=scores["d"], mask=masks["all"])
    }
    out["by_lead_day"] = {
        str(dd): {
            arm: float(scores[arm][:, rows["sunny"] & (rows["lead_day"] == dd)].mean())
            for arm in arms
        }
        for dd in range(1, 11)
    }
    sun = rows["sunny"] & rows["in_window"]
    qs_all = {arm: [arm_quantiles(arm=arm, seed=s, rows=rows) for s in seeds] for arm in arms}
    out["tails_sunny"] = {}
    for arm, qs in qs_all.items():
        t = {"mae_median": float(np.mean([np.abs(q[:, 5] - rows["y"])[sun].mean() * 100 for q in qs])), "coverage_5_95": float(np.mean([((rows["y"] >= q[:, 1]) & (rows["y"] <= q[:, 9]))[sun].mean() for q in qs]))}  # fmt: skip
        for i, lev in ((0, "q01"), (1, "q05"), (9, "q95"), (10, "q99")):
            diff = [rows["y"] - q[:, i] for q in qs]
            t[f"pinball_{lev}"] = float(
                np.mean(
                    [
                        100
                        * np.where(d >= 0, k639.LEVELS[i] * d, (k639.LEVELS[i] - 1) * d)[sun].mean()
                        for d in diff
                    ]
                )
            )
            t[f"coverage_{lev}"] = float(np.mean([(rows["y"] <= q[:, i])[sun].mean() for q in qs]))
        out["tails_sunny"][arm] = t
    # Median minus mean forecast per Primary (arms a and d), a check for a spread that shifts the forecast.
    out["extra"]["median_minus_mean_q"] = {}
    for arm in ("a", "d"):
        q = qs_all[arm][0]
        mean_q = q.mean(axis=1)
        out["extra"]["median_minus_mean_q"][arm] = {
            int(p): float(100 * (q[:, 5] - mean_q)[rows["primary"] == p].mean())
            for p in np.unique(rows["primary"])
        }
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
            print(f"{band} {key}: {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    print("all half-hours a-d:", res["contrasts"]["all_half_hours"]["a-d"])


if __name__ == "__main__":
    main()

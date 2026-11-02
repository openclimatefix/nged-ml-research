# ruff: noqa: E501
"""Experiment 514: the frozen p tables on real rows (report only), after the freeze.

Reports: the share of rows with no labelled event flagged, the weight and Kish effective sample size
of w = (1 - p)^g over experiment 511's training windows, the area under the curve of p on the labelled
events, p on D8's unexplained dips and D22's dip days, and the national same-sign shift. All numbers
are interim and unaudited until the results review.
"""

import json
import pickle
from typing import Final

import e00514_data as d514  # ty: ignore[unresolved-import]
import e00514_tables as tables  # ty: ignore[unresolved-import]
import e00514_train as train  # ty: ignore[unresolved-import]
import numpy as np

HALF_LIFE_DAYS: Final[float] = 90.0
FLOOR: Final[float] = 0.1
BOOT: Final[int] = 500
SHIFT_AGES: Final[tuple[int, ...]] = (0, 1, 2, 3)
SHIFT_BLOCK_DAYS: Final[tuple[int, ...]] = (8, 14, 20)


def weights(*, p: np.ndarray, g: int) -> np.ndarray:
    """Return the training weight (1 - p)^g with the floor."""
    return np.maximum((1.0 - p) ** g, FLOOR)


def kish(*, w: np.ndarray) -> float:
    """Return the Kish effective sample size of weights `w`."""
    return float(w.sum() ** 2 / np.sum(w**2))


def fold_real_stats(*, month: str) -> dict:
    """Return flag shares and weight statistics of a fold at every daily cutoff."""
    t = pickle.loads((d514.WORK_DIR / f"p_{month}.pkl").read_bytes())
    clean = d514.clean_primary_power()
    events = d514.event_mask(n=clean.shape[1])
    out = {
        "month": month,
        "flag_share": [],
        "flag_share_unlabelled": [],
        "mean_w": {1: [], 2: []},
        "kish_ratio": {1: [], 2: []},
        "kish_recency_only_ratio": [],
        "kish_ratio_vs_recency": {1: [], 2: []},
    }
    for k, c in enumerate(t["cutoffs"]):
        origin = t["table_origin"][k]
        p = t["p"][k][:, : c - origin]
        valid = (
            np.isfinite(clean[:, origin:c]) & (np.arange(origin, c) < c - d514.CUT_STEPS)[None, :]
        )
        ev = events[:, origin:c]
        age = (c - np.arange(origin, c)) / 48.0
        recency = 0.5 ** (age / HALF_LIFE_DAYS)
        flag = (p >= 0.5) & valid
        out["flag_share"].append(float(flag.sum() / valid.sum()))
        unl = valid & ~ev
        out["flag_share_unlabelled"].append(float(((p >= 0.5) & unl).sum() / unl.sum()))
        rec = np.broadcast_to(recency, valid.shape)[unl]
        k0 = kish(w=rec)
        out["kish_recency_only_ratio"].append(k0 / unl.sum())
        for g in (1, 2):
            w = weights(p=p[unl], g=g)
            out["mean_w"][g].append(float(w.mean()))
            out["kish_ratio"][g].append(kish(w=w) / unl.sum())
            out["kish_ratio_vs_recency"][g].append(kish(w=rec * w) / k0)
    return out


def auc(*, pos: np.ndarray, neg: np.ndarray) -> float:
    """Return the area under the curve of scores `pos` against `neg` (ties count one half)."""
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def event_auc(*, month: str, rng: np.random.Generator) -> dict:
    """Return the AUC of mean p on labelled events against matched windows, from the fold's last cutoff table."""
    t = pickle.loads((d514.WORK_DIR / f"p_{month}.pkl").read_bytes())
    k = len(t["cutoffs"]) - 1
    c, origin = int(t["cutoffs"][k]), int(t["table_origin"][k])
    p = t["p"][k]
    events = d514.labelled_event_table()
    mask = d514.event_mask(n=d514.grid_length())
    names = d514.sites()
    pos, neg, owner, days = [], [], [], []
    for r in events.iter_rows(named=True):
        i = names.index(r["label"])
        a, b = int(r["onset_idx"]), int(r["end_idx"])
        if a < origin + 48 * 30 or b > c - 14 * 48:
            continue
        if np.isfinite(d514.clean_primary_power()[i, a : b + 1]).mean() < 0.8:
            continue
        pos.append(float(p[i, a - origin : b - origin + 1].mean()))
        days.append(a // 48)
        length = b - a + 1
        draws = []
        for _ in range(200):
            s = int(rng.integers(origin + 48 * 30, c - 14 * 48 - length))
            if (
                not mask[i, s : s + length].any()
                and np.isfinite(d514.clean_primary_power()[i, s : s + length]).mean() > 0.8
            ):
                draws.append(float(p[i, s - origin : s - origin + length].mean()))
            if len(draws) == 5:
                break
        neg.append(draws)
        owner.append(i)
        if not draws:
            pos.pop()
            days.pop()
            neg.pop()
            owner.pop()
    pos_a = np.array(pos)
    all_neg = [x for d in neg for x in d]
    base = auc(pos=pos_a, neg=np.array(all_neg))
    boots = []
    day_arr = np.array(days)
    blocks = np.unique(day_arr)
    for _ in range(BOOT):
        chosen = rng.choice(blocks, size=len(blocks))
        pick = np.concatenate([np.flatnonzero(day_arr == b) for b in chosen])
        boots.append(auc(pos=pos_a[pick], neg=np.array([x for j in pick for x in neg[j]])))
    return {"month": month, "bootstrap_blocks": len(blocks), "events": len(pos_a), "matched_windows": len(all_neg), "auc": base,
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "mean_p_events": float(pos_a.mean()), "mean_p_matched": float(np.mean(all_neg))}  # fmt: skip


def dip_p(*, month: str) -> dict:
    """Return p on D8's unexplained dip days and D22's dip days (rows 10:00 to 15:00) from the last table."""
    t = pickle.loads((d514.WORK_DIR / f"p_{month}.pkl").read_bytes())
    k = len(t["cutoffs"]) - 1
    c, origin = int(t["cutoffs"][k]), int(t["table_origin"][k])
    p = t["p"][k]
    _explained, unexplained = train.dip_windows()
    names = d514.sites()
    out = {}
    d8_unexplained = unexplained[: len(unexplained) - 12]
    d22_days = unexplained[len(d8_unexplained) :]
    for label, days, site in (("D8_unexplained", d8_unexplained, "D8"), ("D22", d22_days, "D22")):
        i = names.index(site)
        shares, n = [], 0
        for day in days:
            lo, hi = day + 20, day + 30
            if origin <= lo and hi < c - 14 * 48:
                row = p[i, lo - origin : hi - origin]
                shares.append(float((row < 0.5).mean()))
                n += 1
        out[label] = {"days_in_window": n, "share_of_days_rows_below_half": float(np.mean(shares)) if shares else None,
                      "share_days_mostly_below_half": float(np.mean([s >= 0.5 for s in shares])) if shares else None}  # fmt: skip
    return out


def national_shift(*, month: str) -> dict:
    """Return the share of shifted rows with p >= 0.5 after experiment 511's national same-sign shift."""
    model = pickle.loads((d514.WORK_DIR / "frozen_model.pkl").read_bytes())
    fold = d514.load_fold(month=month)
    shapes = d514.load_shapes()
    scales = tables.fold_scales(fold=fold)
    clean = d514.clean_primary_power()
    days = d514.daily_cutoffs(month=month)
    res: dict = {}
    for start_day in SHIFT_BLOCK_DAYS:
        block_end = d514.cutoff_index(cutoff=days[start_day])
        size = d514.c510.shift_sizes(clean=clean, cutoff=days[start_day]) / fold.half_width
        for sign in (1, -1):
            for age in SHIFT_AGES:
                cutoff = block_end + age * 48
                if start_day + age >= len(days):
                    continue
                lo, hi = block_end - 7 * 48, block_end - d514.CUT_STEPS
                p, _ = tables.p_at_cutoff(
                    model=model, fold=fold, scales=scales, shapes=shapes, cutoff_idx=cutoff,
                    previous=None, full=True, add=(lo, block_end, sign * size),
                )  # fmt: skip
                rows = p[:, lo:hi][size > 0]
                valid = np.isfinite(fold.z[:, lo:hi])[size > 0]
                res.setdefault(age, []).append(float(((rows >= 0.5) & valid).sum() / valid.sum()))
    return {
        int(a): {"mean_share_p_ge_half": float(np.mean(v)), "n": len(v)} for a, v in res.items()
    }


def main() -> None:
    """Print and store the real-row reports."""
    rng = np.random.default_rng(1514)
    out: dict = {"folds": {}, "auc": {}, "dips": {}, "shift": {}}
    for month in d514.FOLDS:
        s = fold_real_stats(month=month)
        out["folds"][month] = {
            "flag_share_mean": float(np.mean(s["flag_share"])),
            "flag_share_unlabelled_mean": float(np.mean(s["flag_share_unlabelled"])),
            "mean_w": {g: float(np.mean(v)) for g, v in s["mean_w"].items()},
            "kish_ratio": {g: float(np.mean(v)) for g, v in s["kish_ratio"].items()},
            "kish_recency_only_ratio": float(np.mean(s["kish_recency_only_ratio"])),
            "kish_ratio_vs_recency": {g: float(np.mean(v)) for g, v in s["kish_ratio_vs_recency"].items()},
        }  # fmt: skip
        print(month, json.dumps(out["folds"][month]), flush=True)
    for month in ("2026-06", "2026-03"):
        out["auc"][month] = event_auc(month=month, rng=rng)
        out["dips"][month] = dip_p(month=month)
        print(month, out["auc"][month], out["dips"][month], flush=True)
    for month in d514.FOLDS:
        out["shift"][month] = national_shift(month=month)
        print("national shift", month, out["shift"][month], flush=True)
    (d514.WORK_DIR / "eval_real.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

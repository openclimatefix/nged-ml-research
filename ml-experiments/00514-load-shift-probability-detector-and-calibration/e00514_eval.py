# ruff: noqa: E501
"""Experiment 514: calibration, recall, and false flags of the frozen classifier on injected worlds.

Run after `e00514_train.py fit`. Every number is interim and unaudited until the results review.
Bootstrap unit: the Primary-month (a (month, series) pair), 300 resamples.
"""

import json
import pickle
from typing import Final

import e00514_data as d514  # ty: ignore[unresolved-import]
import e00514_train as train  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BOOT: Final[int] = 300
PREVALENCES: Final[tuple[float, ...]] = (0.07, 0.02)
BINS: Final[int] = 10
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
EIGHT: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")


def group_of(*, series: int) -> str:
    """Return `six`, `eight`, or `short` (D15 and D18) for a Primary index."""
    name = d514.sites()[series]
    return "six" if name in SIX else "eight" if name in EIGHT else "short"


def unit_bins(*, p: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return a (BINS, 3) array: sum of weight, weight times p, weight times y per bin."""
    which = np.clip(np.digitize(p, np.linspace(0, 1, BINS + 1)[1:-1]), 0, BINS - 1)
    out = np.zeros((BINS, 3))
    np.add.at(out[:, 0], which, w)
    np.add.at(out[:, 1], which, w * p)
    np.add.at(out[:, 2], which, w * y)
    return out


def ece_of(*, bins: np.ndarray) -> float:
    """Return the weighted expected calibration error from summed bin statistics."""
    total = bins[:, 0].sum()
    ok = bins[:, 0] > 0
    return float(np.sum(np.abs(bins[ok, 1] - bins[ok, 2])) / total)


def brier_of(*, sums: np.ndarray) -> float:
    """Return the weighted Brier score from (sum w, sum w (p - y)^2) per unit."""
    return float(sums[:, 1].sum() / sums[:, 0].sum())


def evaluate_set(
    *,
    model: train.Model,
    records: list[dict],
    pi_val: float,
    target: float,
    rng: np.random.Generator,
) -> dict:
    """Return row-level calibration of a set at a target prevalence, with bootstrap intervals."""
    shift = train.logit(target) - train.logit(pi_val)
    p_all, y_all, k_all = train.row_arrays(model=model, records=records, shift=shift)
    w_all = train.prior_weights(y=y_all, target=target)
    units: dict[tuple[str, int], list[int]] = {}
    for k, r in enumerate(records):
        units.setdefault((r["month"], r["series"]), []).append(k)
    keys = list(units)
    unit_stats, unit_sums = [], []
    for key in keys:
        m = np.isin(k_all, units[key])
        unit_stats.append(unit_bins(p=p_all[m], y=y_all[m], w=w_all[m]))
        unit_sums.append([w_all[m].sum(), float(np.sum(w_all[m] * (p_all[m] - y_all[m]) ** 2)),
                          float(np.sum(w_all[m] * (target - y_all[m]) ** 2))])  # fmt: skip
    stats, sums = np.stack(unit_stats), np.array(unit_sums)
    ece = ece_of(bins=stats.sum(axis=0))
    brier = brier_of(sums=sums)
    base = float(sums[:, 2].sum() / sums[:, 0].sum())
    boot_ece, boot_skill = [], []
    for _ in range(BOOT):
        pick = rng.integers(len(keys), size=len(keys))
        boot_ece.append(ece_of(bins=stats[pick].sum(axis=0)))
        boot_skill.append(1 - sums[pick, 1].sum() / sums[pick, 2].sum())
    table = pl.DataFrame(
        {"bin": range(BINS), "weight": stats.sum(axis=0)[:, 0],
         "mean_p": stats.sum(axis=0)[:, 1] / np.maximum(stats.sum(axis=0)[:, 0], 1e-12),
         "observed": stats.sum(axis=0)[:, 2] / np.maximum(stats.sum(axis=0)[:, 0], 1e-12)}
    )  # fmt: skip
    return {
        "prevalence": target, "rows": len(y_all), "units": len(keys), "positives": int(y_all.sum()),
        "ece": ece, "ece_ci": [float(np.percentile(boot_ece, 2.5)), float(np.percentile(boot_ece, 97.5))],
        "brier": brier, "brier_constant": base, "skill": 1 - brier / base,
        "skill_ci": [float(np.percentile(boot_skill, 2.5)), float(np.percentile(boot_skill, 97.5))],
        "reliability": table.to_dicts(),
    }  # fmt: skip


def recall_tables(
    *, model: train.Model, records: list[dict], threshold: float, rng: np.random.Generator
) -> pl.DataFrame:
    """Return the recall of grid pulses (at least half their rows at p >= threshold) by cell."""
    rows = []
    for r in records:
        if r["pulse"] is None or not r["truth"].any():
            continue
        p = model.row_p(record=r)
        inside = r["truth"]
        size, dur, shape, _sign, mirrored = r["pulse"]
        rows.append({
            "size": size, "duration": dur, "shape": shape, "mirrored": mirrored,
            "month": r["month"], "series": r["series"], "group": group_of(series=r["series"]),
            "hit": float((p[inside] >= threshold).mean() >= 0.5),
        })  # fmt: skip
    return pl.DataFrame(rows)


def boot_mean(*, frame: pl.DataFrame, rng: np.random.Generator) -> tuple[float, float, float, int]:
    """Return the mean of `hit`, a bootstrap interval over (month, series) units, and the count."""
    g = frame.group_by(["month", "series"]).agg(pl.col("hit").sum().alias("s"), pl.len().alias("n"))
    s, n = g["s"].to_numpy(), g["n"].to_numpy()
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(len(s), size=len(s))
        vals.append(s[pick].sum() / max(n[pick].sum(), 1))
    return (
        float(s.sum() / n.sum()),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
        int(n.sum()),
    )


def false_flags(
    *, model: train.Model, records: list[dict], threshold: float, rng: np.random.Generator
) -> dict:
    """Return false-flag runs (p >= threshold with no injected row) per Primary-year in null worlds."""
    counts, years = [], []
    for r in records:
        p = model.row_p(record=r)
        flag = p >= threshold
        starts = np.flatnonzero(flag & ~np.concatenate([[False], flag[:-1]]))
        counts.append(len(starts))
        years.append(len(p) / (48 * 365.25))
    c, y = np.array(counts), np.array(years)
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(len(c), size=len(c))
        vals.append(c[pick].sum() / y[pick].sum())
    return {"threshold": threshold, "rate": float(c.sum() / y.sum()), "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "primary_years": float(y.sum()), "runs": int(c.sum())}  # fmt: skip


def main() -> None:
    """Print and store the injected-world evaluation."""
    model = pickle.loads((d514.WORK_DIR / "frozen_model.pkl").read_bytes())
    info = json.loads((d514.WORK_DIR / "frozen_model.json").read_text())
    pi_val = info["validation_row_prevalence"]
    rng = np.random.default_rng(514)
    out: dict = {"calibration": {}, "recall": {}, "false_flags": {}, "genuine": {}}
    for name in ("train", "val", "held_seed", "held_month", "held_shape", "held_primary"):
        records = train.load_set(name=name)
        for target in PREVALENCES:
            res = evaluate_set(model=model, records=records, pi_val=pi_val, target=target, rng=rng)
            out["calibration"][f"{name}@{target}"] = res
            print(
                f"{name}{' (fit statistic)' if name in ('train', 'val') else ''} @ {target}: ECE {res['ece']:.4f} {res['ece_ci']}, skill {res['skill']:.3f} {res['skill_ci']}, positives {res['positives']}",
                flush=True,
            )
    grid = train.load_set(name="grid")
    for threshold in (0.5,):
        table = recall_tables(model=model, records=grid, threshold=threshold, rng=rng)
        table.write_parquet(d514.WORK_DIR / "recall_grid_rows.parquet")
        for key in ("size", "duration", "shape"):
            for value in sorted(table[key].unique().to_list()):
                m, lo, hi, n = boot_mean(frame=table.filter(pl.col(key) == value), rng=rng)
                out["recall"][f"{key}={value}"] = {"recall": m, "ci": [lo, hi], "pulses": n}
                print(f"recall by {key}={value}: {m:.3f} [{lo:.3f}, {hi:.3f}] n={n}")
        big = table.filter((pl.col("size") >= 1.0) & (pl.col("duration") >= 12))
        m, lo, hi, n = boot_mean(frame=big, rng=rng)
        out["recall"]["size>=1_duration>=6h"] = {"recall": m, "ci": [lo, hi], "pulses": n}
        print(f"recall size>=1 and duration>=6 h: {m:.3f} [{lo:.3f}, {hi:.3f}] n={n}")
        for g in ("six", "eight"):
            m, lo, hi, n = boot_mean(frame=big.filter(pl.col("group") == g), rng=rng)
            out["recall"][f"size>=1_duration>=6h_{g}"] = {"recall": m, "ci": [lo, hi], "pulses": n}
            print(f"  {g}: {m:.3f} [{lo:.3f}, {hi:.3f}] n={n}")
    nulls = train.load_set(name="null")
    for threshold in (0.5, 0.3, 0.7):
        res = false_flags(model=model, records=nulls, threshold=threshold, rng=rng)
        out["false_flags"][str(threshold)] = res
        print("false flags per Primary-year", res)
    # genuine level changes: share of rows from day 15 after the step start with p >= 0.5
    gen = [r for r in train.load_set(name="held_month") if r["genuine"].any()]
    shares = []
    for r in gen:
        p = model.row_p(record=r)
        late = np.flatnonzero(r["genuine"])
        late = late[late >= r["genuine_start"] + 15 * 48]
        if len(late):
            shares.append(float((p[late] >= 0.5).mean()))
    out["genuine"] = {
        "worlds": len(shares),
        "mean_share_p_ge_half_after_day_15": float(np.mean(shares)) if shares else None,
    }
    print("genuine level changes, share of rows from day 15 with p>=0.5:", out["genuine"])
    (d514.WORK_DIR / "eval_injected.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

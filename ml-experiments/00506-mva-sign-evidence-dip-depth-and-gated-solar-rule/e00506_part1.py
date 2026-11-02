"""Experiment 506, Part 1: the apparent-power dip at sunny middays and its calibration.

For each Primary and each UK day, the dip statistic is the minimum reading over the sunny-midday
window (UK clock 10:00 to 15:00, CAMS clear-sky index at least 0.7, at least 6 such half-hours).
At the six signed-MW Primaries a day is a crossing if the signed power (after an injected solar
farm, as in experiment 408) falls below zero in the window. The reading is built four ways:
`|P|` (reactive floor 0), `sqrt(P^2 + Q^2)` with Q = 2 and 4 points, and Q = 0.25 times the
dark-hour demand level (power factor 0.97). The area under the curve uses a month-block bootstrap.

    PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-* \
        uv run --no-sync python ml-experiments/00506-*/e00506_part1.py
"""

import json
from typing import Final

import e00506_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

LEVELS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0)
ARMS: Final[tuple[str, ...]] = ("Q0", "Q2", "Q4", "PF097")
WINDOW: Final[tuple[float, float]] = (10.0, 15.0)
MIN_HALF_HOURS: Final[int] = 6
N_RESAMPLES: Final[int] = 20_000
N_BINS: Final[int] = 400
TAN_PHI_097: Final[float] = 0.2506
SEED: Final[int] = 506


def day_frame(*, site: str, level: float) -> pl.DataFrame:
    """Return one row per UK day with the signed minimum and each arm's minimum reading."""
    stamps = data.grid()
    p = data.power_points(site=site)
    u, k = data.cams(site=site)
    hour = data.local_hour_start(stamps=stamps)
    local = stamps.dt.offset_by("-30m").dt.convert_time_zone("Europe/London")
    date = local.dt.date().to_numpy()
    month = local.dt.month().to_numpy()
    year = local.dt.year().to_numpy()
    weekend = (local.dt.weekday().to_numpy() >= 6).astype(int)
    sunny = (
        (hour >= WINDOW[0])
        & (hour < WINDOW[1])
        & (np.nan_to_num(k) >= data.SUNNY_K)
        & np.isfinite(p)
    )
    dark = (np.nan_to_num(u, nan=1.0) < 0.01) & np.isfinite(p)
    # Dark-hour demand level per (month, weekend): the median of the dark-hour power.
    key = month * 2 + weekend
    level_by_key = {
        int(c): float(np.median(p[dark & (key == c)]))
        for c in np.unique(key)
        if (dark & (key == c)).sum() > 50
    }
    demand = np.array([level_by_key.get(int(c), np.nan) for c in key])
    signed = p.copy()
    if level > 0 and site in data.FARM_OF_SITE:
        farm = data.farm_norm(farm=data.FARM_OF_SITE[site])
        before = (stamps < data.LEVEL_WINDOW_END).to_numpy() & sunny & np.isin(month, range(4, 10))
        scale = level * float(np.median(np.abs(p[before])))
        signed = p - scale * farm
    if site in data.MVA_ONLY:
        raw = np.abs(p)
        readings = dict.fromkeys(ARMS, raw)
    else:
        readings = {
            "Q0": np.abs(signed),
            "Q2": np.sqrt(signed**2 + 2.0**2),
            "Q4": np.sqrt(signed**2 + 4.0**2),
            "PF097": np.sqrt(signed**2 + (TAN_PHI_097 * np.abs(demand)) ** 2),
        }
        if not np.isfinite(demand[sunny]).all():
            msg = f"{site}: dark-hour demand level missing on sunny rows"
            raise ValueError(msg)
    frame = pl.DataFrame(
        {"date": date, "year": year, "month": month, "sunny": sunny, "signed": signed}
        | dict(readings)
    ).filter(pl.col("sunny"), pl.col("signed").is_finite())
    return (
        frame.group_by("date")
        .agg(
            pl.col("year").first(),
            pl.col("month").first(),
            n=pl.len(),
            signed_min=pl.col("signed").min(),
            signed_max=pl.col("signed").max(),
            **{a: pl.col(a).min() for a in ARMS},
        )
        .filter(pl.col("n") >= MIN_HALF_HOURS)
        .with_columns(
            site=pl.lit(site),
            level=pl.lit(level),
            crossing=(pl.col("signed_min") < 0) & (pl.col("signed_max") > 0),
            any_negative=pl.col("signed_min") < 0,
        )
    )


def hist_by_block(
    *, stat: np.ndarray, label: np.ndarray, block: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (blocks, positives per bin, negatives per bin) with one row per block."""
    bins = np.clip(np.searchsorted(edges, stat, side="right") - 1, 0, N_BINS - 1)
    ids, inv = np.unique(block, return_inverse=True)
    pos = np.zeros((len(ids), N_BINS))
    neg = np.zeros((len(ids), N_BINS))
    np.add.at(pos, (inv[label], bins[label]), 1)
    np.add.at(neg, (inv[~label], bins[~label]), 1)
    return ids, pos, neg


def auc_from_hist(*, pos: np.ndarray, neg: np.ndarray) -> float:
    """Return the area under the curve for a low statistic meaning crossing (ties count half)."""
    p, n = pos.sum(), neg.sum()
    if p == 0 or n == 0:
        return float("nan")
    # P(stat_pos < stat_neg) + 0.5 P(tie).
    neg_above = n - np.cumsum(neg)
    return float((pos * (neg_above + 0.5 * neg)).sum() / (p * n))


def bootstrap(*, pos: np.ndarray, neg: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the AUC of each resample; `weights` (resamples x blocks) is shared across arms."""
    p = weights @ pos
    n = weights @ neg
    neg_above = n.sum(1, keepdims=True) - np.cumsum(n, axis=1)
    return (p * (neg_above + 0.5 * n)).sum(1) / (p.sum(1) * n.sum(1))


def best_threshold(*, stat: np.ndarray, label: np.ndarray) -> float:
    """Return the threshold (days with a lower dip are called crossings) with the best Youden J."""
    cand = np.unique(np.quantile(stat, np.linspace(0, 1, 401)))
    best, best_j = cand[0], -1.0
    for c in cand:
        called = stat < c
        j = called[label].mean() - called[~label].mean()
        if j > best_j:
            best, best_j = float(c), float(j)
    return best


def main() -> None:  # noqa: PLR0915
    """Build the day tables, calibrate and freeze thresholds at the six, then apply to the eight."""
    data.WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    six_days = pl.concat([day_frame(site=s, level=lv) for s in data.SIGNED_SIX for lv in LEVELS])
    six_days.write_parquet(data.WORK / "part1_days_six.parquet")
    print("== crossing counts (days, sign-change days, any-negative days) per Primary and level")
    print(
        six_days.group_by("site", "level")
        .agg(
            days=pl.len(),
            crossings=pl.col("crossing").sum(),
            any_negative=pl.col("any_negative").sum(),
        )
        .sort("site", "level")
    )

    rows = []
    thresholds: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    boot: dict[tuple[str, float], np.ndarray] = {}
    for level in LEVELS:
        sub = six_days.filter(pl.col("level") == level)
        label = sub["crossing"].to_numpy()
        block = (sub["year"].to_numpy() * 100 + sub["month"].to_numpy()).astype(int)
        n_blocks = len(np.unique(block))
        weights = np.stack(
            [
                np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
                for _ in range(N_RESAMPLES)
            ]
        ).astype(float)
        for arm in ARMS:
            stat = sub[arm].to_numpy()
            edges = np.quantile(stat, np.linspace(0, 1, N_BINS + 1)[:-1])
            _, pos, neg = hist_by_block(stat=stat, label=label, block=block, edges=edges)
            auc = auc_from_hist(pos=pos.sum(0), neg=neg.sum(0))
            draws = bootstrap(pos=pos, neg=neg, weights=weights)
            boot[(arm, level)] = draws
            threshold = best_threshold(stat=stat, label=label)
            thresholds[arm][str(level)] = threshold
            called = stat < threshold
            # Leave one Primary out: threshold from the other five, rates on the held-out one.
            site_arr = sub["site"].to_numpy()
            loo_tpr, loo_fpr = [], []
            for held in data.SIGNED_SIX:
                m = site_arr == held
                if label[m].sum() == 0 or label[~m].sum() == 0:
                    continue
                c = best_threshold(stat=stat[~m], label=label[~m])
                loo_tpr.append((stat[m & label] < c).mean())
                loo_fpr.append((stat[m & ~label] < c).mean())
            rows.append(
                {
                    "level": level,
                    "arm": arm,
                    "n_days": len(sub),
                    "n_crossing": int(label.sum()),
                    "auc": auc,
                    "auc_lo": float(np.percentile(draws, 2.5)),
                    "auc_hi": float(np.percentile(draws, 97.5)),
                    "se": float(np.std(draws)),
                    "threshold_points": threshold,
                    "tpr_in_sample": float(called[label].mean()),
                    "fpr_in_sample": float(called[~label].mean()),
                    "tpr_loo": float(np.mean(loo_tpr)) if loo_tpr else float("nan"),
                    "fpr_loo": float(np.mean(loo_fpr)) if loo_fpr else float("nan"),
                    "n_loo_primaries": len(loo_tpr),
                }
            )
    table = pl.DataFrame(rows)
    table.write_parquet(data.WORK / "part1_auc.parquet")
    (data.WORK / "part1_thresholds.json").write_text(json.dumps(thresholds, indent=1))
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    print("== area under the curve, pooled over the six")
    print(table)

    print("== AUC difference Q0 minus PF097 (shared resamples), minimum detectable effect 2.8 SE")
    for level in LEVELS:
        d = boot[("Q0", level)] - boot[("PF097", level)]
        print(
            level,
            f"diff {d.mean():.3f} [{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]"
            f" MDE {2.8 * float(np.std(d)):.3f}",
        )

    # Thresholds are now written to disk; only then are the eight read.
    frozen = json.loads((data.WORK / "part1_thresholds.json").read_text())
    eight = pl.concat([day_frame(site=s, level=0.0) for s in data.MVA_ONLY])
    eight.write_parquet(data.WORK / "part1_days_eight.parquet")
    apply = []
    for site in (*data.SIGNED_SIX, *data.MVA_ONLY):
        sub = (
            six_days.filter(pl.col("level") == 0.0) if site in data.SIGNED_SIX else eight
        ).filter(pl.col("site") == site)
        for arm in ARMS:
            t = frozen[arm]["2.0"]
            stat = sub[arm].to_numpy()
            apply.append(
                {
                    "site": site,
                    "arm": arm,
                    "days": len(sub),
                    "median_dip": float(np.median(stat)),
                    "p10_dip": float(np.percentile(stat, 10)),
                    "min_dip": float(stat.min()),
                    "share_below": float((stat < t).mean()),
                    "threshold": t,
                }
            )
    applied = pl.DataFrame(apply)
    applied.write_parquet(data.WORK / "part1_apply.parquet")
    print("== frozen level-2.0 thresholds applied (the eight use the raw MVA reading in every arm)")
    print(applied)
    print("Bonferroni count: 70")


if __name__ == "__main__":
    main()

"""Experiment 508, Part A: shoulder-hour dips, calibrated at the six, applied to the eight.

    PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-*:ml-experiments/00508-* \
        uv run --no-sync python ml-experiments/00508-*/e00508_part_a.py

April to September only. One label for every window: the day's signed power changes sign in
daylight (elevation above zero). Days qualify only if every window has enough finite half-hours.
Thresholds are frozen on the six and written to disk before any of the eight is read.
"""

import json
from typing import Final

import e00506_data as data  # ty: ignore[unresolved-import]
import e00508_common as common  # ty: ignore[unresolved-import]
import e00508_geometry as geo  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

N_RESAMPLES: Final[int] = 20_000
N_BINS: Final[int] = 400
SEED: Final[int] = 508
RATIO_EDGES: Final[tuple[float, ...]] = (0.0, 1.5, 3.0, 6.0, 1e9)
MIN_CLASS: Final[int] = 30
MIN_BLOCKS: Final[int] = 10
DARK_SLOTS: Final = "adjacent"
PLACEBO_SHARES: Final[tuple[float, ...]] = (0.01, 0.02, 0.03)
N_AUC: Final[int] = 32
N_CONTRAST: Final[int] = 16
N_PLACEBO: Final[int] = 16


def day_stats(*, mats: geo.Mats, level: float, band: tuple[float, float]) -> dict:
    """Return per-day window minima (both arms), label, qualification, and sky and dark placebos."""
    sig, extended = geo.signed(mats=mats, level=level)
    read = geo.readings(mats=mats, sig=sig)
    masks = geo.window_masks(mats=mats, band=band)
    finite = np.isfinite(sig)
    day_light = mats.el > 0
    out: dict = {"extended_share": extended}
    qualifies = mats.in_months().copy()
    for name, mask in masks.items():
        m = mask & finite
        qualifies &= m.sum(axis=1) >= geo.MIN_HH[name]
        for arm in common.ARMS:
            out[f"{name}_{arm}"] = np.where(m, read[arm], np.inf).min(axis=1)
        k = np.where(m, mats.k, np.nan)
        with np.errstate(all="ignore"):
            out[f"{name}_k"] = np.nanmean(k, axis=1)
    lit = finite & day_light
    smin = np.where(lit, sig, np.inf).min(axis=1)
    smax = np.where(lit, sig, -np.inf).max(axis=1)
    out["crossing"] = (smin < 0) & (smax > 0)
    out["qualifies"] = qualifies
    # Adjacent dark window of the same number of half-hours as each shoulder.
    for name in geo.MIN_HH:
        if name not in {"morning", "evening"}:
            continue
        n_days = len(mats.dates)
        dark_min = {arm: np.full(n_days, np.nan) for arm in common.ARMS}
        for i in range(n_days):
            m = masks[name][i] & finite[i]
            if not m.any():
                continue
            slots = np.where(m)[0]
            n = len(slots)
            idx = (
                np.arange(slots[0] - n, slots[0])
                if name == "morning"
                else np.arange(slots[-1] + 1, slots[-1] + 1 + n)
            )
            if idx.min() < 0 or idx.max() > 47 or not finite[i, idx].all():
                continue
            for arm in common.ARMS:
                dark_min[arm][i] = read[arm][i, idx].min()
        for arm in common.ARMS:
            out[f"{name}_dark_{arm}"] = dark_min[arm]
    midday_sunny = masks["midday"] & finite & (mats.k >= data.SUNNY_K)
    dark = finite & (mats.el < geo.DARK_ELEVATION)
    summer = np.isin(mats.month, common.RATIO_MONTHS)
    out["ratio"] = {
        arm: float(
            np.median(read[arm][midday_sunny & summer[:, None]])
            / np.median(read[arm][dark & summer[:, None]])
        )
        for arm in common.ARMS
    }
    return out


def hist(*, stat: np.ndarray, label: np.ndarray, block_idx: np.ndarray, n_blocks: int) -> tuple:
    """Return (positives, negatives) as (blocks x bins) counts on quantile bins."""
    edges = np.quantile(stat, np.linspace(0, 1, N_BINS + 1)[:-1])
    bins = np.clip(np.searchsorted(edges, stat, side="right") - 1, 0, N_BINS - 1)
    pos = np.zeros((n_blocks, N_BINS))
    neg = np.zeros((n_blocks, N_BINS))
    np.add.at(pos, (block_idx[label], bins[label]), 1)
    np.add.at(neg, (block_idx[~label], bins[~label]), 1)
    return pos, neg


def auc_of(*, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Return AUC for (resamples x bins) counts; a low statistic means crossing, ties count half."""
    n_total = neg.sum(axis=1, keepdims=True)
    above = n_total - np.cumsum(neg, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (above + 0.5 * neg)).sum(axis=1) / (pos.sum(axis=1) * n_total[:, 0])


def best_threshold(*, stat: np.ndarray, label: np.ndarray) -> float:
    """Return the threshold (days below it are called crossings) with the best Youden J."""
    best, best_j = float(stat.min()), -1.0
    for c in np.unique(np.quantile(stat, np.linspace(0, 1, 401))):
        called = stat < c
        j = called[label].mean() - called[~label].mean()
        if j > best_j:
            best, best_j = float(c), float(j)
    return best


def bin_of(*, ratio: float) -> int:
    """Return the index of the export-to-import ratio bin."""
    return int(np.searchsorted(RATIO_EDGES, ratio, side="right") - 1)


def world_frames(*, band: tuple[float, float]) -> pl.DataFrame:
    """Return one row per (site, level, day) for the six with the window statistics."""
    parts = []
    for site in data.SIGNED_SIX:
        farm = data.farm_norm(farm=data.FARM_OF_SITE[site])
        mats = geo.Mats(site=site, extra={"farm": farm})
        for level in common.ALL_LEVELS:
            st = day_stats(mats=mats, level=level, band=band)
            q = st["qualifies"]
            cols = {
                "site": np.full(q.sum(), site),
                "level": np.full(q.sum(), level),
                "date": mats.dates[q],
                "block": mats.block[q],
                "crossing": st["crossing"][q],
                "ratio_Q0": np.full(q.sum(), st["ratio"]["Q0"]),
                "ratio_PF097": np.full(q.sum(), st["ratio"]["PF097"]),
                "extended_share": np.full(q.sum(), st["extended_share"]),
            }
            for key, val in st.items():
                if (
                    isinstance(val, np.ndarray)
                    and val.shape == q.shape
                    and key not in cols
                    and key not in {"qualifies", "crossing"}
                ):
                    cols[key] = val[q]
            parts.append(pl.DataFrame(cols))
    return pl.concat(parts)


def main() -> None:  # noqa: PLR0912, PLR0915
    """Calibrate on the six, freeze thresholds, then apply to the eight."""
    common.WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    ref = geo.Mats(site="D12", extra={"farm": np.zeros(len(data.grid()))})
    band = geo.calibrate_shoulder(mats=ref)
    print(f"== shoulder band by sun elevation (degrees): {band[0]:.2f} to {band[1]:.2f}")
    six = world_frames(band=band)
    six.write_parquet(common.WORK / "partA_days_six.parquet")
    print("== extended farm share of half-hours (farm shape below 15 degrees)")
    print(six.group_by("site").agg(pl.col("extended_share").first()))
    ratio_table = (
        six.group_by("site", "level")
        .agg(
            pl.col("ratio_Q0").first(),
            pl.col("ratio_PF097").first(),
            days=pl.len(),
            crossings=pl.col("crossing").sum(),
        )
        .sort("site", "level")
    )
    ratio_table.write_parquet(common.WORK / "partA_ratio_six.parquet")
    print("== observed export-to-import ratio, days, and crossing days per Primary and level")
    print(ratio_table)

    injected = six.filter(pl.col("level") > 0)
    rows, thresholds, boot = [], {}, {}
    all_blocks = np.sort(injected["block"].unique().to_numpy())
    idx_of = {int(b): i for i, b in enumerate(all_blocks)}
    weights = np.stack(
        [
            np.bincount(
                rng.integers(0, len(all_blocks), len(all_blocks)), minlength=len(all_blocks)
            )
            for _ in range(N_RESAMPLES)
        ]
    ).astype(float)
    for arm in common.ARMS:
        ratio_col = f"ratio_{arm}"
        binned = injected.with_columns(
            bin=pl.col(ratio_col).map_elements(lambda r: bin_of(ratio=r), return_dtype=pl.Int64)
        )
        for b in range(len(RATIO_EDGES) - 1):
            sub = binned.filter(pl.col("bin") == b)
            if sub.height == 0:
                continue
            label = sub["crossing"].to_numpy()
            block_idx = np.array([idx_of[int(x)] for x in sub["block"].to_numpy()])
            n_blocks_both = len(
                set(sub.filter(pl.col("crossing"))["block"])
                & set(sub.filter(~pl.col("crossing"))["block"])
            )
            for window in common.WINDOWS:
                stat = sub[f"{window}_{arm}"].to_numpy()
                row = {
                    "arm": arm,
                    "ratio_bin": f"{RATIO_EDGES[b]:g}-{min(RATIO_EDGES[b + 1], 99):g}",
                    "window": window,
                    "n_days": sub.height,
                    "n_crossing": int(label.sum()),
                    "n_ordinary": int((~label).sum()),
                    "n_worlds": sub.select(["site", "level"]).unique().height,
                    "blocks_with_both": n_blocks_both,
                    "auc": float("nan"),
                    "auc_lo": float("nan"),
                    "auc_hi": float("nan"),
                    "threshold": float("nan"),
                    "tpr_loo": float("nan"),
                    "fpr_loo": float("nan"),
                    "estimable": False,
                }
                if (
                    label.sum() >= MIN_CLASS
                    and (~label).sum() >= MIN_CLASS
                    and n_blocks_both >= MIN_BLOCKS
                ):
                    pos, neg = hist(
                        stat=stat, label=label, block_idx=block_idx, n_blocks=len(all_blocks)
                    )
                    draws = auc_of(pos=weights @ pos, neg=weights @ neg)
                    boot[(arm, b, window)] = draws
                    row["auc"] = float(auc_of(pos=pos.sum(0)[None], neg=neg.sum(0)[None])[0])
                    row["auc_lo"] = float(np.nanpercentile(draws, 2.5))
                    row["auc_hi"] = float(np.nanpercentile(draws, 97.5))
                    row["threshold"] = best_threshold(stat=stat, label=label)
                    row["estimable"] = True
                    thresholds[f"{arm}|{b}|{window}"] = row["threshold"]
                    site_arr, tpr, fpr = sub["site"].to_numpy(), [], []
                    for held in data.SIGNED_SIX:
                        m = site_arr == held
                        if label[m].sum() == 0 or label[~m].sum() == 0 or (~label[m]).sum() == 0:
                            continue
                        c = best_threshold(stat=stat[~m], label=label[~m])
                        tpr.append((stat[m & label] < c).mean())
                        fpr.append((stat[m & ~label] < c).mean())
                    row["tpr_loo"] = float(np.mean(tpr)) if tpr else float("nan")
                    row["fpr_loo"] = float(np.mean(fpr)) if fpr else float("nan")
                rows.append(row)
    auc_table = pl.DataFrame(rows)
    auc_table.write_parquet(common.WORK / "partA_auc.parquet")
    print("== area under the curve by arm, ratio bin, window (pooled injected worlds)")
    print(auc_table.sort("arm", "ratio_bin", "window"))

    contrasts = []
    for arm in common.ARMS:
        for b in range(len(RATIO_EDGES) - 1):
            for shoulder in common.SHOULDERS:
                ka, kb = (arm, b, shoulder), (arm, b, "midday")
                if ka not in boot or kb not in boot:
                    continue
                d = boot[ka] - boot[kb]
                contrasts.append(
                    {
                        "arm": arm,
                        "ratio_bin": b,
                        "contrast": f"{shoulder} minus midday",
                        "diff": float(np.nanmean(d)),
                        "lo95": float(np.nanpercentile(d, 2.5)),
                        "hi95": float(np.nanpercentile(d, 97.5)),
                        "lo_bonf": float(
                            np.nanpercentile(d, 100 * 0.025 / (N_AUC + N_CONTRAST + N_PLACEBO))
                        ),
                        "hi_bonf": float(
                            np.nanpercentile(
                                d, 100 * (1 - 0.025 / (N_AUC + N_CONTRAST + N_PLACEBO))
                            )
                        ),
                        "mde": float(2.8 * np.nanstd(d)),
                    }
                )
    contrast_table = pl.DataFrame(contrasts)
    contrast_table.write_parquet(common.WORK / "partA_contrasts.parquet")
    print("== named contrasts (shoulder AUC minus midday AUC), minimum detectable effect")
    print(contrast_table)

    (common.WORK / "partA_thresholds.json").write_text(json.dumps(thresholds, indent=1))
    frozen = json.loads((common.WORK / "partA_thresholds.json").read_text())

    apply_rows = []
    for site in data.MVA_ONLY:
        mats = geo.Mats(site=site, extra={})
        st = day_stats(mats=mats, level=0.0, band=band)
        in_months = st["qualifies"]
        for arm in common.ARMS:
            b = bin_of(ratio=st["ratio"][arm])
            for window in common.SHOULDERS:
                t = frozen.get(f"{arm}|{b}|{window}")
                stat = st[f"{window}_{arm}"]
                dark = st[f"{window}_dark_{arm}"]
                k = st[f"{window}_k"]
                have = in_months & np.isfinite(stat) & (stat < np.inf)
                sunny = have & (k >= data.SUNNY_K)
                cloudy = have & (k < common.CLOUDY_K)
                row = {
                    "site": site,
                    "arm": arm,
                    "window": window,
                    "ratio": st["ratio"][arm],
                    "bin": b,
                    "threshold": t if t is not None else float("nan"),
                    "n_sunny": int(sunny.sum()),
                    "n_cloudy": int(cloudy.sum()),
                }
                if t is not None:
                    dip = stat < t
                    ks, ns, kc, nc = (
                        int((dip & sunny).sum()),
                        int(sunny.sum()),
                        int((dip & cloudy).sum()),
                        int(cloudy.sum()),
                    )
                    row |= {
                        "k_sunny": ks,
                        "k_cloudy": kc,
                        "share_sunny": ks / max(ns, 1),
                        "share_cloudy": kc / max(nc, 1),
                    }
                    ratio_est = ((ks + 0.5) / (ns + 1)) / ((kc + 0.5) / (nc + 1))
                    uniq, inv = np.unique(mats.block, return_inverse=True)
                    per_block = np.stack(
                        [
                            np.bincount(inv, weights=x.astype(float), minlength=len(uniq))
                            for x in (dip & sunny, sunny, dip & cloudy, cloudy)
                        ]
                    )
                    w = np.stack(
                        [
                            np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
                            for _ in range(N_RESAMPLES)
                        ]
                    ).astype(float)
                    ks_, ns_, kc_, nc_ = (w @ per_block[r] for r in range(4))
                    draws = ((ks_ + 0.5) / (ns_ + 1)) / ((kc_ + 0.5) / (nc_ + 1))
                    row |= {
                        "sunny_to_cloudy": ratio_est,
                        "ratio_lo": float(np.percentile(draws, 2.5)),
                        "ratio_hi": float(np.percentile(draws, 97.5)),
                    }
                    ok = sunny & np.isfinite(dark)
                    dark_dip = dark < t
                    row["share_adjacent_dark_dip_on_sunny"] = (
                        float(dark_dip[ok].mean()) if ok.any() else float("nan")
                    )
                    free = ok & ~dark_dip
                    row["share_shoulder_dip_without_dark_dip"] = (
                        float(dip[free].mean()) if free.any() else float("nan")
                    )
                    row["n_free"] = int(free.sum())
                apply_rows.append(row)
    applied = pl.DataFrame(apply_rows, infer_schema_length=None)
    applied = pl.concat(
        [
            applied.filter(pl.col("site").is_in(["D12", "D8"])),
            applied.filter(~pl.col("site").is_in(["D12", "D8"])),
        ],
        how="vertical",
    )
    applied.write_parquet(common.WORK / "partA_apply.parquet")
    print("== frozen thresholds applied to the eight (D12 and D8 first)")
    print(applied)
    print(f"Bonferroni families: {N_AUC} AUC + {N_CONTRAST} contrasts + {N_PLACEBO} placebo ratios")


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Experiment 699: print every score and contrast that the page quotes.

Usage: python e00699_contrasts.py COMMAND [FOLD ...]

- `known`: the known-answer table from the cut copies of experiment 677's stored arms (needs no GPU).
- `a0`: the reproduction check of fold 2026-01 against the stored forecasts, value by value.
- `noise`: the standard deviation between seeds of each fold in the refitted clean chain, and the fits
  flagged as possible failed fits.
- `sizing`: the expected half-width of family H on all-rows MAE from the noise of the measured folds.
- `families`: the two intervals of families H, S2, and S3, the descriptive readings, the predictions
  P1 to P6, and the verdict.
- `tables`: the scores by metering group, lead band, and the four load-shift scenarios, with the band width and
  the median minus the mean forecast per Primary.
- `all`: `families` and `tables`.

The folds default to those that every arm has scored. Reads the `rows.parquet` files that
`e00677_score.py` writes under each sub-folder of `roots/` and keeps the rows that every arm, every
seed, and both baselines can be scored on.
"""

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import e00521_stats as s521  # ty: ignore[unresolved-import]
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00699_bank import arrays, copied_share, family_contrast, non_inferior
from e00699_common import (
    ARMS,
    FOLDS,
    HOLDOUT,
    LEVEL_FAMILY,
    PHASE_A_FOLDS,
    ROOT,
    ROOT_677,
    SEEDS,
    STORED_SEEDS,
    seeds_of,
)
from e00699_stats import Interval, Term, bootstrap_combination, contrast, t_combination
from scipy import stats
from studies.bootstrap import bootstrap_month_totals

GRID_START = np.datetime64("2019-09-01T00:00")
SIZE_MAE = -0.022
SIZE_CRPS = -0.014
DAY = 48


def month_names(*, origin_idx: np.ndarray) -> np.ndarray:
    """Return each origin's UTC month as `YYYY-MM`."""
    when = GRID_START + origin_idx.astype("timedelta64[30m]")
    return when.astype("datetime64[M]").astype(str)


@dataclass
class Frame:
    """The shared rows with each arm's per-row loss (points of capacity) for each seed."""

    months: list[str]
    series: np.ndarray
    lead_h: np.ndarray
    valid_idx: np.ndarray
    origin_idx: np.ndarray
    month_idx: np.ndarray
    mae: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    crps: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    seed_mean_mae: dict[str, np.ndarray] = field(default_factory=dict)
    seed_mean_crps: dict[str, np.ndarray] = field(default_factory=dict)
    baselines: dict[str, np.ndarray] = field(default_factory=dict)

    def counts(self, *, mask: np.ndarray | None = None) -> np.ndarray:
        """Return the number of rows in each month."""
        idx = self.month_idx if mask is None else self.month_idx[mask]
        return np.bincount(idx, minlength=len(self.months)).astype(float)

    def arrays(
        self, *, arm: str, metric: str, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the numerators and denominators of an arm's mean loss, shape (months, seeds)."""
        store = self.mae if metric == "mae" else self.crps
        seeds = seeds_of(arm=ARMS[arm])
        count = self.counts(mask=mask)
        num = np.zeros((len(self.months), len(seeds)))
        for j, seed in enumerate(seeds):
            values = store[(arm, seed)]
            keep = np.ones(len(values), dtype=bool) if mask is None else mask
            num[:, j] = np.bincount(
                self.month_idx[keep],
                weights=values[keep].astype(np.float64),
                minlength=len(self.months),
            )
        return num, np.repeat(count[:, None], len(seeds), axis=1)


def row_key(*, series: np.ndarray, origin: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return one integer per (series, origin, valid time)."""
    return (
        (series.astype(np.int64) << 40) | (origin.astype(np.int64) << 20) | valid.astype(np.int64)
    )


def read_scores(*, arm: str, tag: str) -> dict[str, np.ndarray]:
    """Read one scored file's rows that every system could be scored on, sorted by key."""
    spec = ARMS[arm]
    path = spec.folder / "scores" / f"{spec.file_stem}_s{tag}" / "rows.parquet"
    frame = pl.read_parquet(path).filter(pl.col("ok"))
    key = row_key(
        series=frame["time_series_id"].to_numpy(),
        origin=frame["origin_idx"].to_numpy(),
        valid=frame["valid_idx"].to_numpy(),
    )
    order = np.argsort(key)
    out = {"key": key[order]}
    for name in (
        "time_series_id", "origin_idx", "valid_idx", "lead_h", "Model_mae", "Model_crps",
        "Manual heuristic_mae", "Manual heuristic_crps", "Climatology_mae", "Climatology_crps",
    ):  # fmt: skip
        out[name] = frame[name].to_numpy()[order]
    return out


def load(
    *, arms: list[str], months: tuple[str, ...] | None = None, with_seed_mean: bool = True
) -> Frame:
    """Load the per-row scores of the arms and keep the rows that every file can be scored on."""
    files: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for arm in arms:
        seeds = seeds_of(arm=ARMS[arm])
        tags = [str(s) for s in seeds]
        if with_seed_mean:
            tags.append("".join(str(s) for s in seeds))
        for tag in tags:
            files[(arm, tag)] = read_scores(arm=arm, tag=tag)
    common = None
    for data in files.values():
        common = (
            data["key"]
            if common is None
            else np.intersect1d(common, data["key"], assume_unique=True)
        )
    assert common is not None
    assert len(common)
    first = next(iter(files.values()))
    where = np.searchsorted(first["key"], common)
    origin = first["origin_idx"][where]
    month = month_names(origin_idx=origin)
    keep = np.ones(len(common), dtype=bool) if months is None else np.isin(month, list(months))
    common, where, month = common[keep], where[keep], month[keep]
    names = sorted(set(month))
    frame = Frame(
        months=names,
        series=first["time_series_id"][where],
        lead_h=first["lead_h"][where],
        valid_idx=first["valid_idx"][where],
        origin_idx=first["origin_idx"][where],
        month_idx=np.searchsorted(names, month),
    )
    assert datetime(2026, 6, 30, tzinfo=UTC) < HOLDOUT
    for (arm, tag), data in files.items():
        at = np.searchsorted(data["key"], common)  # ty: ignore[no-matching-overload]
        assert np.array_equal(data["key"][at], common)
        mae, crps = data["Model_mae"][at], data["Model_crps"][at]
        if len(tag) == 1:  # one seed; the seed-mean file's tag lists three or more seeds
            seed = int(tag)
            frame.mae[(arm, seed)], frame.crps[(arm, seed)] = mae, crps
        else:
            frame.seed_mean_mae[arm], frame.seed_mean_crps[arm] = mae, crps
        if not frame.baselines:
            frame.baselines = {
                "Manual heuristic mae": data["Manual heuristic_mae"][at],
                "Manual heuristic crps": data["Manual heuristic_crps"][at],
                "Climatology mae": data["Climatology_mae"][at],
                "Climatology crps": data["Climatology_crps"][at],
            }
    return frame


def hdr(text: str) -> None:
    """Print a section heading."""
    print(f"\n=== {text}", flush=True)


def fmt(iv: Interval) -> str:
    """Return a contrast as text."""
    return (
        f"{iv.point:+.4f} [bootstrap {iv.boot_lo:+.4f} to {iv.boot_hi:+.4f}; "
        f"t {iv.t_lo:+.4f} to {iv.t_hi:+.4f}] at {iv.level:.2f}%"
    )


def skill(
    *,
    frame: Frame,
    a: str,
    b: str,
    metric: str,
    paired: bool,
    level: float = LEVEL_FAMILY,
    mask: np.ndarray | None = None,
    months: list[int] | None = None,
    resample_seeds: bool = True,
) -> Interval:
    """Return the mean loss of arm b minus arm a, with both intervals."""
    na, da = frame.arrays(arm=a, metric=metric, mask=mask)
    nb, db = frame.arrays(arm=b, metric=metric, mask=mask)
    if months is not None:
        na, da, nb, db = na[months], da[months], nb[months], db[months]
    return contrast(
        num_a=na, den_a=da, num_b=nb, den_b=db, level=level, paired_seeds=paired,
        resample_seeds=resample_seeds,
    )  # fmt: skip


# ----------------------------------------------------------------------------- known answer


def month_only(
    *, num_a: np.ndarray, den_a: np.ndarray, num_b: np.ndarray, den_b: np.ndarray, level: float
) -> tuple[float, float, float]:
    """Return the pooled difference and its month-only interval from `studies.bootstrap`.

    The arrays have shape (months, seeds). Whole months are resampled and one seed is drawn per resample,
    with the random stream that produced the plan's reference numbers. Both arms must hold the same
    denominator in each month and seed, which holds because they score the same rows.
    """
    assert np.allclose(den_a, den_b), "the arms' denominators differ"
    assert np.allclose(den_a, den_a[:, :1]), "the denominator differs between seeds"
    point = float((num_b - num_a).sum() / den_a.sum())
    lo, hi = bootstrap_month_totals(
        month_sums=(num_b - num_a).T, month_counts=den_a[:, 0], level=level
    )
    return point, lo, hi


KNOWN: dict[str, tuple[float, float, float, float, float]] = {
    "mae": (7.169, 7.213, 0.045, 0.024, 0.066),
    "crps": (4.021, 4.049, 0.028, 0.014, 0.043),
    "share": (0.123, 0.075, -0.047, -0.061, -0.036),
}
"""Experiment 677's stored table: no injection, stage 3 only, difference, and the interval around it."""


def known() -> None:
    """Print the known-answer table from the cut copies of the stored arms."""
    hdr("known-answer table: stored arms, seed-mean forecast of seeds 0 to 2")
    frame = load(arms=["none_stored", "s3_stored"], months=FOLDS)
    print("rows", len(frame.series), "months", frame.months)
    level = 100.0 - 5.0 / 6.0
    for _metric, label, store in (
        ("mae", "All-rows MAE of the median", frame.seed_mean_mae),
        ("crps", "All-rows CRPS", frame.seed_mean_crps),
    ):
        a, b = store["none_stored"], store["s3_stored"]
        num_a = np.bincount(frame.month_idx, weights=a.astype(np.float64))[:, None]
        num_b = np.bincount(frame.month_idx, weights=b.astype(np.float64))[:, None]
        den = frame.counts()[:, None]
        point, lo, hi = month_only(num_a=num_a, den_a=den, num_b=num_b, den_b=den, level=level)
        print(
            f"{label}: no injection {num_a.sum() / den.sum():.3f}; stage 3 only {num_b.sum() / den.sum():.3f}; "
            f"difference {point:+.3f}; month bootstrap {lo:+.3f} to {hi:+.3f} at {level:.2f}%"
        )
        want = KNOWN[_metric]
        for what, got, ref in zip(
            ("no injection", "stage 3 only", "difference"),
            (num_a.sum() / den.sum(), num_b.sum() / den.sum(), point),
            want[:3],
            strict=True,
        ):
            assert abs(got - ref) <= 0.0005, (label, what, got, ref)
        assert abs(lo - want[3]) <= 0.0015, (label, lo, want)
        assert abs(hi - want[4]) <= 0.0015, (label, hi, want)
    na, da = arrays(arm_key="none_stored", bank="581", months=FOLDS)
    nb, db = arrays(arm_key="s3_stored", bank="581", months=FOLDS)
    point, lo, hi = month_only(num_a=na, den_a=da, num_b=nb, den_b=db, level=level)
    print(
        f"Copied share, days 0 to 14: no injection {na.sum() / da.sum():.3f}; stage 3 only {nb.sum() / db.sum():.3f}; "
        f"difference {point:+.3f}; month bootstrap {lo:+.3f} to {hi:+.3f} at {level:.2f}%"
    )
    for what, got, ref in zip(
        ("no injection", "stage 3 only", "difference"),
        (na.sum() / da.sum(), nb.sum() / db.sum(), point),
        KNOWN["share"][:3],
        strict=True,
    ):
        assert abs(got - ref) <= 0.0005, ("copied share", what, got, ref)
    assert abs(lo - KNOWN["share"][3]) <= 0.0015, (lo, KNOWN["share"])
    assert abs(hi - KNOWN["share"][4]) <= 0.0015, (hi, KNOWN["share"])
    print("known-answer table reproduced within 0.0005 (points) and 0.0015 (interval ends)")
    for name, arr in frame.baselines.items():
        print(f"{name} on the shared rows: {arr.astype(np.float64).mean():.3f}")
    print(
        "per-seed all-rows MAE, stage 3 only (stored):",
        [round(float(frame.mae[("s3_stored", s)].mean()), 3) for s in STORED_SEEDS],
    )


# ----------------------------------------------------------------------------- A0


def a0() -> None:
    """Compare the refitted fold 2026-01 with the stored forecasts, value by value and by per-seed MAE."""
    hdr("phase A0: refit of fold 2026-01 against experiment 677's stored forecasts (stage 3 only)")
    q = [f"q{t}" for t in ("01", "05", "10", "25", "40", "50", "60", "75", "90", "95", "99")]
    identical = True
    for seed in STORED_SEEDS:
        name = f"2026-01_d0_s{seed}.parquet"
        new = pl.read_parquet(
            ARMS["s3"].folder / "arms" / "k1_5w_sp1" / "work" / "fits" / "P5" / name
        )
        old = pl.read_parquet(ROOT_677 / "arms" / "k1_5w_sp1" / "work" / "fits" / "P5" / name)
        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
        joined = new.join(old, on=keys, suffix="_old", how="inner")
        assert joined.height == new.height == old.height, (joined.height, new.height, old.height)
        worst = max(float((joined[c] - joined[f"{c}_old"]).abs().max()) for c in q)  # ty: ignore[invalid-argument-type]
        print(
            f"seed {seed}: {joined.height:,} rows; largest absolute difference in any quantile {worst:.6g} (share of capacity)"
        )
        identical &= worst == 0.0
    print("forecasts identical:", identical)
    new_rows, old_rows = {}, {}
    for seed in STORED_SEEDS:
        new_rows[seed] = read_scores(arm="s3", tag=str(seed))
        old_rows[seed] = read_scores(arm="s3_stored", tag=str(seed))
    common = None
    for data in [*new_rows.values(), *old_rows.values()]:
        month = month_names(origin_idx=data["origin_idx"])
        sel = data["key"][month == "2026-01"]
        common = sel if common is None else np.intersect1d(common, sel)
    new_mae, old_mae = [], []
    for seed in STORED_SEEDS:
        for store, out in ((new_rows, new_mae), (old_rows, old_mae)):
            data = store[seed]
            at = np.searchsorted(data["key"], common)  # ty: ignore[no-matching-overload]
            out.append(float(data["Model_mae"][at].astype(np.float64).mean()))
    sd = float(np.std(old_mae, ddof=1))
    print(
        f"fold 2026-01 all-rows MAE by seed: refit {np.round(new_mae, 4).tolist()}, stored {np.round(old_mae, 4).tolist()}"
    )
    print(
        f"SD between the stored seeds {sd:.4f}; per-seed |refit - stored| {np.round(np.abs(np.array(new_mae) - old_mae), 4).tolist()}"
    )
    ok = identical or bool((np.abs(np.array(new_mae) - old_mae) <= sd).all())
    print("A0 passes (identical, or every seed within one SD of the stored value):", ok)


# ----------------------------------------------------------------------------- noise and sizing


def noise(*, frame: Frame) -> dict[str, np.ndarray]:
    """Print the SD between seeds of each fold and the flagged fits; return the per-fold variances."""
    hdr("noise between seeds in the refitted clean chain (stage 3 only and no injection)")
    variances = {}
    for metric in ("mae", "crps"):
        rows = []
        for m, month in enumerate(frame.months):
            per = {}
            for arm in ("s3", "none"):
                num, den = frame.arrays(arm=arm, metric=metric)
                per[arm] = num[m] / den[m]
            contrast_seed = per["s3"] - per["none"]
            rows.append(
                {
                    "fold": month,
                    "sd_s3": float(per["s3"].std(ddof=1)),
                    "sd_none": float(per["none"].std(ddof=1)),
                    "sd_within_seed_contrast": float(contrast_seed.std(ddof=1)),
                    "mean_s3": float(per["s3"].mean()),
                    "mean_none": float(per["none"].mean()),
                }
            )
            variances.setdefault(metric, []).append(
                float(np.mean([per["s3"].var(ddof=1), per["none"].var(ddof=1)]))
            )
        print(f"-- {metric} (points of capacity), SD over the {len(SEEDS)} seeds")
        print(pl.DataFrame(rows))
    print(
        "-- flagged fits: a fit more than 3 SD from the mean of the other seven seeds of its fold and arm"
    )
    flagged = 0
    for metric in ("mae",):
        for arm in ("s3", "none"):
            num, den = frame.arrays(arm=arm, metric=metric)
            for m, month in enumerate(frame.months):
                vals = num[m] / den[m]
                for s in range(len(vals)):
                    others = np.delete(vals, s)
                    z = (vals[s] - others.mean()) / others.std(ddof=1)
                    if abs(z) > 3:
                        flagged += 1
                        log = (
                            ROOT
                            / "checks"
                            / ARMS[arm].folder.name
                            / ARMS[arm].step
                            / "logs"
                            / f"{month}_d0_s{s}.json"
                        )
                        extra = log.read_text()[:300] if log.exists() else "no log"
                        print(
                            f"FLAG {ARMS[arm].label} fold {month} seed {s}: MAE {vals[s]:.3f}, z {z:+.1f}; {extra}"
                        )
    print("flagged fits:", flagged)
    return {k: np.array(v) for k, v in variances.items()}


def sizing(*, frame: Frame, variances: dict[str, np.ndarray]) -> None:
    """Print the expected half-width of family H on MAE and CRPS for 8, 12, and 16 seeds."""
    hdr(
        "sizing: expected half-width of family H under no effect (98.33% t interval on the monthly differences)"
    )
    t_crit = float(stats.t.ppf(0.5 + LEVEL_FAMILY / 200.0, len(FOLDS) - 1))
    counts = frame.counts()
    mean_count = counts.mean()
    for metric in ("mae", "crps"):
        var = variances[metric]
        median_var = float(np.median(var))
        full_var = np.full(len(FOLDS), median_var)
        full_n = np.full(len(FOLDS), mean_count)
        for k, month in enumerate(frame.months):  # key the measured folds by name, not by slot
            full_var[FOLDS.index(month)] = var[k]
            full_n[FOLDS.index(month)] = counts[k]
        w = full_n / full_n.sum()
        for n_seeds in (8, 12, 16):
            se = float(np.sqrt((w**2 * 2.0 * full_var / n_seeds).sum()))
            print(
                f"{metric}: {n_seeds} seeds: standard error {se:.4f}, expected half-width {t_crit * se:.4f} (t multiplier {t_crit:.2f}; measured folds {len(var)} of {len(FOLDS)})"
            )
    print("Family S2 resolution is not known until phase A2 (its arms share stage-1 weights).")
    print(
        "Rule: 8 seeds if the MAE half-width with 8 seeds is at most 0.022; else 12 or 16 if one of them reaches 0.022; else family H cannot resolve the predicted size."
    )


# ----------------------------------------------------------------------------- families


def months_without(*, frame: Frame, drop: str) -> list[int]:
    """Return the indices of the months other than `drop`."""
    return [i for i, m in enumerate(frame.months) if m != drop]


def families(*, frame: Frame) -> dict:
    """Print every family, the descriptive readings, P1 to P6, and the verdict."""
    out: dict[str, Any] = {}
    bank_months = tuple(frame.months)
    hdr(f"families at {LEVEL_FAMILY:.2f}% over the folds {frame.months}")
    for name, a, b, paired in (
        ("H", "s3", "s123", False),
        ("S2", "s3", "s23", True),
        ("S3", "s12", "s123", True),
        ("clean chain (stage 3 only minus no injection)", "none", "s3", True),
    ):
        if not all(any(k[0] == x for k in frame.mae) for x in (a, b)):
            continue
        mae = skill(frame=frame, a=a, b=b, metric="mae", paired=paired)
        crps = skill(frame=frame, a=a, b=b, metric="crps", paired=paired)
        print(f"{name}: {ARMS[b].label} minus {ARMS[a].label}")
        print("  MAE ", fmt(mae))
        print("  CRPS", fmt(crps))
        share581 = family_contrast(a=a, b=b, bank="581", months=bank_months, paired_seeds=paired)
        print(
            "  copied share, 581 bank:",
            fmt(share581),
            "| non-inferior:",
            non_inferior(interval=share581),
            "| higher:",
            share581.lower_above(bound=0.0),
        )
        share660 = None
        try:
            share660 = family_contrast(
                a=a, b=b, bank="660", months=bank_months, paired_seeds=paired
            )
            print(
                "  copied share, 660 bank:",
                fmt(share660),
                "| non-inferior:",
                non_inferior(interval=share660),
            )
        except FileNotFoundError as exc:
            print("  copied share, 660 bank: not run", type(exc).__name__)
        out[name.split(" ")[0]] = {
            "mae": mae,
            "crps": crps,
            "share581": share581,
            "share660": share660,
        }
    if "H" in out:
        descriptive_h(frame=frame)
        out["verdict"] = verdict(h=out["H"])
        predictions(out=out, frame=frame)
    return out


def descriptive_h(*, frame: Frame) -> None:
    """Print the descriptive readings beside family H."""
    hdr("descriptive readings of family H (stages 1 to 3 minus stage 3 only)")
    a, b = "s3", "s123"
    for metric in ("mae", "crps"):
        na, da = frame.arrays(arm=a, metric=metric)
        nb, db = frame.arrays(arm=b, metric=metric)
        by_month = nb.sum(axis=1) / db.sum(axis=1) - na.sum(axis=1) / da.sum(axis=1)
        print(
            f"{metric}: contrast in each fold",
            dict(zip(frame.months, np.round(by_month, 4).tolist(), strict=True)),
        )
        by_seed = nb.sum(axis=0) / db.sum(axis=0) - na.sum(axis=0) / da.sum(axis=0)
        print(
            f"{metric}: contrast for each seed number (seed s against seed s)",
            np.round(by_seed, 4).tolist(),
        )
        paired = skill(frame=frame, a=a, b=b, metric=metric, paired=True)
        print(f"{metric}: one draw of seeds per month serving both arms:", fmt(paired))
        seed_mean = frame.seed_mean_mae if metric == "mae" else frame.seed_mean_crps
        sa = np.bincount(frame.month_idx, weights=seed_mean[a].astype(np.float64))[:, None]
        sb = np.bincount(frame.month_idx, weights=seed_mean[b].astype(np.float64))[:, None]
        den = frame.counts()[:, None]
        mo_point, mo_lo, mo_hi = month_only(
            num_a=sa, den_a=den, num_b=sb, den_b=den, level=LEVEL_FAMILY
        )
        print(
            f"{metric}: seed-mean forecast, months only (bootstrap {mo_lo:+.4f} to {mo_hi:+.4f}), point {mo_point:+.4f}; scores {sa.sum() / den.sum():.3f} and {sb.sum() / den.sum():.3f}"
        )
        keep = months_without(frame=frame, drop="2025-11")
        if len(keep) < len(frame.months):
            print(
                f"{metric}: without fold 2025-11:",
                fmt(skill(frame=frame, a=a, b=b, metric=metric, paired=False, months=keep)),
            )
        loo = {}
        for tid in np.unique(frame.series):
            iv = skill(
                frame=frame,
                a=a,
                b=b,
                metric=metric,
                paired=False,
                mask=frame.series != tid,
                level=95.0,
            )
            loo[int(tid)] = (round(iv.point, 4), iv.boot_hi < 0 or iv.boot_lo > 0)
        print(
            f"{metric}: leave one Primary out (time_series_id: point, bootstrap interval excludes zero at 95%)",
            loo,
        )
    terms = [
        Term(*frame.arrays(arm="s12", metric="mae"), -1.0, 0),
        Term(*frame.arrays(arm="s123", metric="mae"), 1.0, 0),
        Term(*frame.arrays(arm="none", metric="mae"), 1.0, 1),
        Term(*frame.arrays(arm="s3", metric="mae"), -1.0, 1),
    ]
    lo, hi = bootstrap_combination(terms=terms, level=95.0)
    centre, t_lo, t_hi = t_combination(terms=terms, level=95.0)
    print(
        f"MAE: finding S3 minus the clean chain's stage 3 only minus no injection, independent chains: {centre:+.4f} [bootstrap {lo:+.4f} to {hi:+.4f}; t {t_lo:+.4f} to {t_hi:+.4f}] at 95%"
    )


def passes(*, iv: Interval) -> bool:
    """Return whether both intervals lie below zero."""
    return iv.below_zero()


def verdict(*, h: dict) -> str:
    """Apply the decision rule to family H and print the verdict."""
    mae, crps, share = h["mae"], h["crps"], h["share581"]
    skill_ok = passes(iv=mae) and passes(iv=crps)
    sizes_met = mae.point <= SIZE_MAE and crps.point <= SIZE_CRPS
    non_inf = non_inferior(interval=share)
    opposite = (mae.below_zero() and crps.above_zero()) or (crps.below_zero() and mae.above_zero())
    if opposite:
        result = "inconclusive (MAE and CRPS in opposite directions: audit trigger 3)"
    elif skill_ok and sizes_met and non_inf:
        result = "supported"
    elif skill_ok and non_inf:
        result = "small gain"
    elif skill_ok:
        result = "partly supported"
    elif (mae.above_zero() and not passes(iv=crps)) or (crps.above_zero() and not passes(iv=mae)):
        result = "refuted"
    elif (
        mae.includes_zero()
        and crps.includes_zero()
        and mae.boot_lo > SIZE_MAE
        and mae.t_lo > SIZE_MAE
        and crps.boot_lo > SIZE_CRPS
        and crps.t_lo > SIZE_CRPS
    ):
        result = "not supported"
    else:
        result = "inconclusive"
    hdr(f"VERDICT (family H, rule applied to the 581 bank): {result}")
    return result


def predictions(*, out: dict, frame: Frame) -> None:
    """Print whether each prediction P1 to P6 passed."""
    hdr("predictions")
    h = out["H"]
    print(
        "P1 (H, MAE point <= -0.022 and both intervals below zero):",
        h["mae"].point <= SIZE_MAE and h["mae"].below_zero(),
    )
    print(
        "P2 (H, CRPS point <= -0.014 and both intervals below zero):",
        h["crps"].point <= SIZE_CRPS and h["crps"].below_zero(),
    )
    print(
        "P3 (H, copied share non-inferior at 0.012 on both intervals, 581 bank):",
        non_inferior(interval=h["share581"]),
        "| 660 bank:",
        None if h["share660"] is None else non_inferior(interval=h["share660"]),
    )
    if "S2" in out:
        s2 = out["S2"]
        print(
            "P4 (S2, MAE and CRPS points below zero with both intervals below zero, copied share non-inferior):",
            s2["mae"].point < 0
            and s2["mae"].below_zero()
            and s2["crps"].point < 0
            and s2["crps"].below_zero()
            and non_inferior(interval=s2["share581"]),
        )
    if "S3" in out:
        s3 = out["S3"]["share581"]
        print(
            "P5 (S3, copied share point <= -0.012 and both intervals below zero):",
            s3.point <= -0.012 and s3.below_zero(),
            "| 660 bank same sign:",
            None if out["S3"]["share660"] is None else out["S3"]["share660"].point < 0,
        )
    clean = out.get("clean")
    if clean:
        mae = clean["mae"].point
        shares = [
            copied_share(arm_key=k, bank="581", months=tuple(frame.months)) for k in ("s3", "none")
        ]
        print(
            f"P6 (reproduction): stage 3 only minus no injection MAE {mae:+.4f} in [+0.015, +0.075]: {0.015 <= mae <= 0.075}; "
            f"copied shares {shares[0]:.3f} (stored 0.075) and {shares[1]:.3f} (stored 0.123), within 0.020: "
            f"{abs(shares[0] - 0.075) <= 0.020 and abs(shares[1] - 0.123) <= 0.020}"
        )


# ----------------------------------------------------------------------------- tables


def events_all(*, site: pl.DataFrame) -> pl.DataFrame:
    """Return every logged event of experiment 653's table that overlaps the folds, plus the verified transfer."""
    table = pl.read_parquet(next(s668.R.glob("00653-*/real_events_masked.parquet"))).filter(
        pl.col("start") >= s668.SCORED_FROM_653
    )
    if table["primary"].dtype == pl.String:
        label = pl.col("primary")
    else:
        label = pl.col("primary").replace_strict(
            dict(zip(site["primary"].to_list(), site["label"].to_list(), strict=True))
        )
    table = table.with_columns(
        label=label, start=pl.col("start") + s668.OFFSET_653, end=pl.col("end") + s668.OFFSET_653
    )
    first, last = s668.idx_of(datetime(2025, 11, 1, tzinfo=UTC)), s668.idx_of(HOLDOUT)
    logged = table.filter((pl.col("end") > first) & (pl.col("start") < last)).select(
        "label", "start", "end"
    )
    verified = pl.DataFrame(
        [
            {"label": k, "start": s668.idx_of(a), "end": s668.idx_of(b)}
            for k, (a, b) in s668.VERIFIED.items()
        ]
    )
    return pl.concat([logged, verified.cast(logged.schema)]).unique()


def scenario_masks(
    *, frame: Frame, site: pl.DataFrame
) -> tuple[dict[str, np.ndarray], pl.DataFrame]:
    """Return the four scenario masks over the shared rows and the event table."""
    label = site.select("time_series_id", "label")
    rows = pl.DataFrame(
        {"time_series_id": frame.series.astype(np.int64), "valid_idx": frame.valid_idx}
    ).join(
        label.cast({"time_series_id": pl.Int64}),
        on="time_series_id",
        how="left",
        maintain_order="left",
    )
    events = events_all(site=site)
    parts = s668.masks(rows=rows, events=events)
    inside, after = parts["inside"], parts["7 days after"]
    masks = {
        "all rows": np.ones(len(frame.series), dtype=bool),
        "outside every logged event and 7 days after": ~inside & ~after,
        "inside a logged event": inside,
        "7 days after a logged event": after,
    }
    return masks, events


def tables(*, frame: Frame) -> None:
    """Print the scores by metering group, lead band, scenario, band width, and median minus mean."""
    site = pl.read_csv(s521.DATA / "site_map.csv")
    arms = [k for k in ("none", "s3", "s23", "s12", "s123") if any(a == k for a, _ in frame.mae)]
    hdr("all-rows mean loss by arm (points of capacity), mean over seeds, and the baselines")
    for arm in arms:
        for metric in ("mae", "crps"):
            num, den = frame.arrays(arm=arm, metric=metric)
            print(
                f"{ARMS[arm].label} {metric}: {num.sum() / den.sum():.4f} (seed SD {np.std(num.sum(axis=0) / den.sum(axis=0), ddof=1):.4f})"
            )
    for name, arr in frame.baselines.items():
        print(f"{name}: {arr.astype(np.float64).mean():.4f}")
    slices: dict[str, np.ndarray] = {}
    group_of = dict(zip(site["time_series_id"].to_list(), site["group"].to_list(), strict=True))
    groups = np.array([group_of[int(t)] for t in frame.series], dtype=object)
    for g in sorted(set(groups)):
        slices[f"group: {g}"] = groups == g
    slices["days 1 to 10 (NGED horizon)"] = (frame.lead_h >= 24) & (frame.lead_h < 240)
    for name, (lo, hi) in lib.LEAD_BANDS.items():
        slices[f"lead band: {name}"] = (frame.lead_h >= lo) & (frame.lead_h < hi)
    scen, events = scenario_masks(frame=frame, site=site)
    print(
        f"events: {events.height} logged events overlap the folds (including the verified transfer)"
    )
    slices.update({f"scenario: {k}": v for k, v in scen.items()})
    for name, mask in slices.items():
        print(
            f"-- {name}: {int(mask.sum()):,} rows, {len(np.unique(frame.origin_idx[mask] // DAY))} origin days"
        )
        for pair, paired in (
            (("s3", "s123"), False),
            (("s3", "s23"), True),
            (("s12", "s123"), True),
        ):
            if not all(x in arms for x in pair) or not mask.any():
                continue
            for metric in ("mae", "crps"):
                iv = skill(
                    frame=frame,
                    a=pair[0],
                    b=pair[1],
                    metric=metric,
                    paired=paired,
                    mask=mask,
                    level=95.0,
                )
                print(f"   {ARMS[pair[1]].label} minus {ARMS[pair[0]].label} {metric}: {fmt(iv)}")
        for arm in arms:
            num, den = frame.arrays(arm=arm, metric="mae", mask=mask)
            print(f"   {ARMS[arm].label} MAE {num.sum() / den.sum():.3f}")
        for name_b, arr in frame.baselines.items():
            if name_b.endswith("mae"):
                print(f"   {name_b}: {arr[mask].astype(np.float64).mean():.3f}")


def extras(*, frame: Frame) -> None:
    """Print the band width of every arm and the median minus the mean forecast per Primary."""
    hdr(
        "width of the q05 to q95 band and median minus mean forecast per Primary (points of capacity)"
    )
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet").select(
        pl.col("time_series_id").cast(pl.Int64), pl.col("capacity").cast(pl.Float64)
    )
    shared = pl.DataFrame(
        {
            "time_series_id": frame.series.astype(np.int64),
            "origin_idx": frame.origin_idx.astype(np.int64),
            "valid_idx": frame.valid_idx.astype(np.int64),
        }
    ).with_columns(shared=pl.lit(True))
    taus = lib.TAUS
    weights = np.zeros(len(taus))
    edges = np.concatenate([[0.0], (taus[1:] + taus[:-1]) / 2, [1.0]])
    weights = np.diff(edges)
    for arm in ("none", "s3", "s23", "s12", "s123"):
        spec = ARMS[arm]
        path = spec.folder / "backtests" / f"{spec.file_stem}.parquet"
        if not path.exists():
            continue
        lf = pl.scan_parquet(path).select(
            pl.col("time_series_id").cast(pl.Int64),
            "power_fcst_init_time",
            "valid_time",
            "seed",
            *lib.QCOLS,
        )
        df = lf.collect().with_columns(
            origin_idx=(
                (
                    pl.col("power_fcst_init_time").dt.epoch("us")
                    - int(datetime(2019, 9, 1, tzinfo=UTC).timestamp() * 1e6)
                )
                // 1_800_000_000
            ),
            valid_idx=(
                (
                    pl.col("valid_time").dt.epoch("us")
                    - int(datetime(2019, 9, 1, tzinfo=UTC).timestamp() * 1e6)
                )
                // 1_800_000_000
            ),
        )
        df = df.join(shared, on=["time_series_id", "origin_idx", "valid_idx"], how="inner").join(
            capacity, on="time_series_id"
        )
        scale = 100.0 / df["capacity"].to_numpy()
        q = df.select(lib.QCOLS).to_numpy().astype(np.float64)
        width = (q[:, 9] - q[:, 1]) * scale
        mean_fc = (q * weights).sum(axis=1) * scale
        diff = q[:, 5] * scale - mean_fc
        out = df.select("time_series_id").with_columns(
            width=pl.Series(width), median_minus_mean=pl.Series(diff)
        )
        print(
            f"{spec.label}: mean q05 to q95 width {width.mean():.3f}; mean median minus mean {diff.mean():+.3f}"
        )
        per = (
            out.group_by("time_series_id")
            .agg(pl.col("median_minus_mean").mean())
            .sort("time_series_id")
        )
        print(
            "  median minus mean by Primary:",
            dict(
                zip(
                    per["time_series_id"].to_list(),
                    np.round(per["median_minus_mean"].to_numpy(), 3).tolist(),
                    strict=True,
                )
            ),
        )
    print(
        "(the mean forecast is the quantile function integrated over the 11 stored levels, an approximation)"
    )


def main() -> None:
    """Run the command named on the command line."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    command = sys.argv[1]
    folds = tuple(sys.argv[2:]) or None
    if command == "known":
        known()
        return
    if command == "a0":
        a0()
        return
    if command in {"noise", "sizing"}:
        frame = load(arms=["s3", "none"], months=folds or PHASE_A_FOLDS, with_seed_mean=False)
        variances = noise(frame=frame)
        if command == "sizing":
            sizing(frame=frame, variances=variances)
        return
    arms = [k for k in ("none", "s3", "s23", "s12", "s123") if (ARMS[k].folder / "scores").exists()]
    frame = load(arms=arms, months=folds)
    print("arms", arms, "folds", frame.months, "shared rows", len(frame.series))
    if command in {"families", "all"}:
        families(frame=frame)
    if command in {"tables", "all"}:
        tables(frame=frame)
        extras(frame=frame)


if __name__ == "__main__":
    main()

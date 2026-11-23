# ruff: noqa: E501, PT018, PERF401, PLR0915, E731
"""Experiment 716: scoring, contrasts, guards, verdicts, and secondary readings (WP6a and WP6b; design section 7).

Usage: run.sh e00716_score.py baseline       # WP6a: experiment 713's scorer on M* and C, solar-block slices
       run.sh e00716_score.py zrule          # the rule on arm Z
       run.sh e00716_score.py part1 --n-seeds 5   # A1 and A2, guards, readings, verdict, S1, S2, S4, report-only cuts
       run.sh e00716_score.py part2 --n-seeds 5   # B1 and B2 (the 16-Primary table is in the file)
       run.sh e00716_score.py screen         # part 1's stop rule on the four screen folds (interim)
       run.sh e00716_score.py secondary1 --n-seeds 5  # S3 and S5 (secondary2 for part 2)
       run.sh e00716_score.py --selftest     # the decision logic on synthetic frames (no data)

Experiment 713's scorer does the loading of forecast files, the losses, the seed average, the
month-block bootstrap, and the t and seed and noise-floor guards. This module adds F7's arms, the
slices, F7's reading and verdict tables (design 7.4, including the guard on a harmful placebo), the
`supported` guards generalised to any treatment and reference, the rule on `Z`, the shifted-`M*`
control of S3, and the secondary readings. Part 2's `e00717_score.py` imports this module and changes
the arm names alone. Every number is in points of capacity, from seed-averaged forecasts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00713_arms as arms713  # ty: ignore[unresolved-import]
import e00713_score as s713  # ty: ignore[unresolved-import]
import e00716_arms as arms
import numpy as np
import polars as pl
from scipy import stats
from studies.sources import ML_EXPERIMENTS_DATA_DIR

KEYS: Final = s713.KEYS
QCOLS: Final = s713.QCOLS
SUNNY_FOLDS: Final = ("2026-04", "2026-05", "2026-06")
DARK_FOLDS: Final = ("2025-11", "2025-12", "2026-01", "2026-02")
TAU: Final = 0.02
DELTA_5: Final = {"six": 0.06, "eight": 0.16}
DELTA_2: Final = {"six": 0.094, "eight": 0.25}
FLOOR_5: Final = {"six": 0.034, "eight": 0.090}
FLOOR_2: Final = {"six": 0.054, "eight": 0.142}
LEVEL_COUNTED: Final = 98.75
LEVEL_SENSITIVITY: Final = 99.375
LEVEL_SECONDARY: Final = 95.0
SOLAR_BLOCK_MINUTES: Final = (6 * 60, 14 * 60 + 30)
"""Valid-time stamps 06:00 to 14:30 UTC, both included."""
LEAD_BANDS: Final = ((24.0, 72.0), (72.0, 120.0), (120.0, 240.0))
HARMFUL: Final = frozenset({"loss", "clear loss"})
GROUPS_COUNTED: Final = ("six", "eight")


def thresholds(*, n_seeds: int) -> tuple[dict[str, float], dict[str, float]]:
    """Return (delta, noise floor) per group: the 5-seed values, or F4's two-seed values when fewer than 5 seeds exist."""
    return (DELTA_5, FLOOR_5) if n_seeds >= 5 else (DELTA_2, FLOOR_2)


# ---------------------------------------------------------------- loading


def fit_path(*, name: str, month: str, seed: int, data_dir: Path = ML_EXPERIMENTS_DATA_DIR) -> Path:
    """Return the per-seed forecast file of an F7 arm or of a no-refit read-out (`H-cams`, `HT-nopv`, ...)."""
    if name in arms.ARMS:
        return arms.fit_file(data_dir=data_dir, arm=name, month=month, seed=seed)
    base = next(b for b in sorted(arms.ARMS, key=len, reverse=True) if name.startswith(b + "-"))
    return (
        arms.fit_file(data_dir=data_dir, arm=base, month=month, seed=seed).parent.parent
        / name
        / f"{month}_s{seed}.parquet"
    )


def load_arm(*, arm: str, seeds: tuple[int, ...], months: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm's per-seed rows (contract origin, capacity shares); F7 arms and read-outs from F7's files, others from experiment 713's loader."""
    is_f7 = arm in arms.ARMS or any(arm.startswith(b + "-") for b in arms.ARMS)
    if not is_f7:
        return s713.load_arm(arm=arm, seeds=seeds, months=months)
    parts = []
    for month in months:
        for seed in seeds:
            frame = pl.read_parquet(
                fit_path(name=arm, month=month, seed=seed), columns=[*KEYS, *QCOLS]
            )
            parts.append(
                s713.to_contract_shares(frame=frame.with_columns(seed=pl.lit(seed, dtype=pl.Int64)))
            )
    out = pl.concat(parts)
    assert out.select(*KEYS, "seed").is_duplicated().sum() == 0, (
        f"arm {arm!r} holds a duplicated (key, seed) row"
    )
    assert out["seed"].n_unique() == len(seeds), f"arm {arm!r} does not hold every seed of {seeds}"
    return out


def load_all(
    *,
    names: list[str],
    months: tuple[str, ...],
    seeds: dict[str, tuple[int, ...]],
    truth: np.ndarray | None = None,
) -> tuple[dict, dict, dict]:
    """Return every arm's per-seed and seed-averaged scored rows and the Primary groups; assert equal keys.

    `seeds[name]` are the seeds of that arm; a contrast uses the seeds both arms hold, so the arms are
    scored on seeds they share (the caller passes the shared set).
    """
    truth = s713.truth_stack() if truth is None else truth
    per_seed, averaged = {}, {}
    for name in names:
        raw = load_arm(arm=name, seeds=seeds[name], months=months)
        per_seed[name] = s713.add_losses(frame=raw, truth=truth)
        averaged[name] = s713.add_losses(frame=s713.seed_average(frame=raw), truth=truth)
    s713.assert_same_keys(frames=averaged)
    return per_seed, averaged, s713.group_ids()


# ---------------------------------------------------------------- slices


def solar_block(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows whose valid time is stamped 06:00 to 14:30 UTC."""
    minutes = frame["valid_time"].dt.hour().cast(pl.Int32) * 60 + frame[
        "valid_time"
    ].dt.minute().cast(pl.Int32)
    return frame.filter((minutes >= SOLAR_BLOCK_MINUTES[0]) & (minutes <= SOLAR_BLOCK_MINUTES[1]))


def in_folds(frame: pl.DataFrame, folds: tuple[str, ...]) -> pl.DataFrame:
    """Keep the rows whose origin month is one of `folds`."""
    return frame.filter(pl.col("power_fcst_init_time").dt.strftime("%Y-%m").is_in(list(folds)))


def apply_slice(
    frames: dict[str, pl.DataFrame], *, solar: bool, folds: tuple[str, ...] | None
) -> dict[str, pl.DataFrame]:
    """Apply the same slice to every frame (rows stay aligned because the slice reads only keys)."""
    out = {}
    for name, frame in frames.items():
        f = frame
        if folds is not None:
            f = in_folds(f, folds)
        if solar:
            f = solar_block(f)
        out[name] = f
    return out


# ---------------------------------------------------------------- one contrast, with F7's guards


def contrast(
    *, per_seed: dict, averaged: dict, treatment: str, reference: str, group: str, groups: dict, level: float,
    metric: str = "crps", band: tuple[float, float] | None = s713.HEADLINE_BAND_HOURS, n_seeds: int,
) -> dict:  # fmt: skip
    """Return 713's contrast with the noise floor and the 4-of-5 seed guard set for `n_seeds` (design 7.2)."""
    c = s713.contrast(
        per_seed=per_seed, averaged=averaged, treatment=treatment, reference=reference, group=group, groups=groups,
        metric=metric, level=level, band=band,
    )  # fmt: skip
    delta, floor = thresholds(n_seeds=n_seeds)
    c["floor"] = floor.get(group)
    c["floor_guard"] = bool(group in floor and abs(c["point"]) >= floor[group])
    need = (
        len(c["seed_differences"]) - 1
        if len(c["seed_differences"]) >= 5
        else len(c["seed_differences"])
    )
    c["seed_guard_4_of_5"] = bool(c["seeds_same_sign"] >= max(need, 1))
    c["delta"] = delta.get(group)
    return c


def reading(*, c: dict) -> str:
    """Return a group's reading by the first row of the table of design 7.4 that holds."""
    d, lo, hi = c["point"], c["lower"], c["upper"]
    guards = c["t_guard"] and c["seed_guard"] and c["floor_guard"]
    delta = c["delta"]
    if lo > TAU and guards:
        return "clear loss"
    if lo > 0 and guards:
        return "loss"
    if hi < 0 and d <= -TAU and guards:
        return "gain"
    if hi < 0 and c["seed_guard_4_of_5"]:
        return "small gain"
    if delta is not None and lo > -delta and hi < delta:
        return "no detectable difference"
    return "cannot tell"


@dataclass
class PlaceboReading:
    """What the verdict reads of the placebo, per group.

    Attributes:
        upper: The upper limit of the placebo contrast (A2 or B2) per group.
        point: Its point estimate per group.
        control_reading: The reading of the placebo arm minus its control (`H-S` minus `M*`, or `HT-S` minus `H`) per group.
    """

    upper: dict[str, float]
    point: dict[str, float]
    control_reading: dict[str, str]

    @property
    def harmful(self) -> bool:
        """Return whether the placebo arm is a loss or a clear loss against its control in either group."""
        return any(r in HARMFUL for r in self.control_reading.values())


def verdict(*, readings: dict[str, str], placebo: PlaceboReading, guards_hold: bool) -> str:
    """Return the verdict by the first row of the verdict table of design 7.4 (with the guard on a harmful placebo in rows 3 to 5)."""
    r = list(readings.values())
    if "clear loss" in r:
        return "refuted"
    if "loss" in r or all(x == "no detectable difference" for x in r):
        return "not supported"
    placebo_clean = all(placebo.upper[g] < 0 for g in readings) and not placebo.harmful
    if all(x == "gain" for x in r) and placebo_clean and guards_hold:
        return "supported"
    gains = [x == "gain" for x in r]
    if (any(gains) and "loss" not in r) or all(gains):
        return "partly supported"
    small = [g for g, x in readings.items() if x == "small gain"]
    if (
        small
        and "gain" not in r
        and "loss" not in r
        and all(placebo.point[g] < 0 for g in small)
        and not placebo.harmful
    ):
        return "small gain"
    return "inconclusive"


def closes_line(*, verdict_word: str) -> bool:
    """Return whether the verdict closes the line (`not supported` or `refuted` adds one to the stop-rule count)."""
    return verdict_word in {"not supported", "refuted"}


# ---------------------------------------------------------------- guards of `supported`


def supported_guards(
    *, treatment: str, reference: str, averaged: dict, groups: dict, mask_points: dict[str, dict[str, float]] | None,
    cold_night_difference: dict[str, float] | None, first_sign: dict[str, float],
) -> dict:  # fmt: skip
    """Return the guards of `supported` for the first contrast of a part (design 7.4).

    The sign of the first contrast holds with each Primary and each month left out; the q01 and q99
    pinball loss of the treatment is within 0.05 of the reference's (a one-sided 95% t bound over the
    months); the normalised MAE of the median moves the same way as CRPS; the treatment is not worse
    than the reference by more than 0.05 on cold winter nights (`cold_night_difference` per group, from
    the protocol's `cold_nights` slice); and, in part 1, the first contrast keeps its sign under both
    masks (`mask_points[mask][group]`, the contrast of the masked read-out against the reference).
    """
    out: dict = {}
    for g in GROUPS_COUNTED:
        sel = {
            n: s713.restrict(
                frame=averaged[n], group=g, band=s713.HEADLINE_BAND_HOURS, groups=groups
            )
            for n in (treatment, reference)
        }
        crps = s713.paired(treatment=sel[treatment], reference=sel[reference], metric="crps")
        month = crps["power_fcst_init_time"].dt.strftime("%Y-%m")
        by_month = (
            crps.with_columns(m=month)
            .group_by("m")
            .agg(pl.col("d").mean())
            .sort("m")["d"]
            .to_numpy()
        )
        by_id = crps.group_by("time_series_id").agg(
            pl.col("d").sum().alias("s"), pl.len().alias("n")
        )
        total, count = by_id["s"].sum(), by_id["n"].sum()
        sign = np.sign(float(total) / float(count))
        loo_primary = [
            np.sign((total - s_) / (count - n_)) == sign
            for s_, n_ in zip(by_id["s"], by_id["n"], strict=True)
        ]
        loo_month = [np.sign((by_month.sum() - x) / (len(by_month) - 1)) == sign for x in by_month]
        tails = {}
        for col in ("pin_q01", "pin_q99"):
            d = (
                s713.paired(treatment=sel[treatment], reference=sel[reference], metric=col)
                .with_columns(m=month)
                .group_by("m")
                .agg(pl.col("d").mean())["d"]
                .to_numpy()
            )
            upper = d.mean() + stats.t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            tails[col] = {"upper_95": float(upper), "ok": bool(upper <= 0.05)}
        mae = float(
            s713.paired(treatment=sel[treatment], reference=sel[reference], metric="abs_err")[
                "d"
            ].mean()
        )  # ty: ignore[invalid-argument-type]
        cold = None if cold_night_difference is None else bool(cold_night_difference[g] <= 0.05)
        masks = (
            None
            if mask_points is None
            else bool(all(np.sign(mask_points[m][g]) == sign for m in mask_points))
        )
        out[g] = {
            "leave_one_primary_out": bool(all(loo_primary)), "leave_one_month_out": bool(all(loo_month)),
            "tails": tails, "mae_moves_with_crps": bool(np.sign(mae) == sign), "cold_nights_ok": cold, "masks_keep_sign": masks,
        }  # fmt: skip
        out[g]["all_hold"] = bool(
            out[g]["leave_one_primary_out"] and out[g]["leave_one_month_out"] and all(t["ok"] for t in tails.values())
            and out[g]["mae_moves_with_crps"] and cold is not False and cold is not None and masks is not False
        )  # fmt: skip
    out["all_hold"] = bool(all(out[g]["all_hold"] for g in GROUPS_COUNTED))
    return out


# ---------------------------------------------------------------- cold winter nights (guard of `supported`)

COLD_NIGHT_MAX_TEMPERATURE_C: Final = 3.0
"""The protocol's `cold_nights` slice: the ERA5 temperature at or below this, and the sun more than 5 degrees below the horizon."""
NIGHT_COS_ZENITH_MAX: Final = float(np.sin(np.radians(-5.0)))
"""The sun is more than 5 degrees below the horizon when the cosine of the zenith is below this."""


def cold_night_grid(*, raw: np.ndarray) -> np.ndarray:
    """Return a (16, G) bool array of the half-hours that the protocol's `cold_nights` slice would hold: the sun more than 5 degrees below the horizon and the temperature at or below 3 C.

    `raw` is the dense table of experiment 713 (column 2 is the temperature in degrees Celsius, column
    7 the cosine of the zenith at the stamp). The guard reads this re-implementation of the protocol's
    two thresholds, in points of capacity, because the protocol's own scorer loads only experiment 713's arms.
    """
    return (np.asarray(raw[:, :, 7]) < NIGHT_COS_ZENITH_MAX) & (
        np.asarray(raw[:, :, 2]) <= COLD_NIGHT_MAX_TEMPERATURE_C
    )


def cold_night_difference(
    *, averaged: dict, groups: dict, treatment: str, reference: str, cold: np.ndarray, prim_of: dict[int, int], grid_start_us: int,
    step_us: int = 1_800_000_000,
) -> dict[str, float]:  # fmt: skip
    """Return, per counted group, the treatment's mean CRPS minus the reference's on the cold winter nights of the headline band (points of capacity; positive is worse)."""
    out = {}
    for g in GROUPS_COUNTED:
        sel = {
            n: s713.restrict(
                frame=averaged[n], group=g, band=s713.HEADLINE_BAND_HOURS, groups=groups
            )
            for n in (treatment, reference)
        }
        both = s713.paired(treatment=sel[treatment], reference=sel[reference], metric="crps")
        prim = np.array([prim_of[t] for t in both["time_series_id"].to_list()])
        idx = ((both["valid_time"].dt.epoch("us").to_numpy() - grid_start_us) // step_us).astype(
            np.int64
        )
        winter = both["valid_time"].dt.month().is_in([12, 1, 2]).to_numpy()
        keep = cold[prim, np.clip(idx, 0, cold.shape[1] - 1)] & winter
        assert keep.any(), f"no cold winter night in group {g}: the guard cannot be read"
        out[g] = float(both["d"].to_numpy()[keep].mean())
    return out


def load_cold_inputs() -> tuple[np.ndarray, dict[int, int], int]:
    """Read the dense table of experiment 713 and the Primary map of the scorer, and return the cold-night grid, the map, and the grid start."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    raw = np.load(
        ML_EXPERIMENTS_DATA_DIR
        / "00713-msa-axial-attention-demand-network-part-1"
        / "cache"
        / "dense_raw.npy",
        mmap_mode="r",
    )
    sites = k578.sites()
    prim_of = dict(zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True))
    return cold_night_grid(raw=raw), prim_of, prep422.GRID_START_US


# ---------------------------------------------------------------- the rule on Z


def z_rule(*, per_seed: dict, averaged: dict, groups: dict, star: str) -> dict:
    """Return the rule on arm `Z`: `Z` minus `M*` on its 5 folds, in absolute value, against the two-seed floor (audit trigger 1)."""
    out: dict = {}
    for g in GROUPS_COUNTED:
        c = contrast(
            per_seed=per_seed,
            averaged=averaged,
            treatment="Z",
            reference=star,
            group=g,
            groups=groups,
            level=LEVEL_SECONDARY,
            n_seeds=2,
        )
        out[g] = {
            "point": c["point"],
            "floor_two_seed": FLOOR_2[g],
            "fires": bool(abs(c["point"]) > FLOOR_2[g]),
            "lower": c["lower"],
            "upper": c["upper"],
        }
    out["audit_trigger_1_fires"] = bool(any(out[g]["fires"] for g in GROUPS_COUNTED))
    return out


# ---------------------------------------------------------------- secondary readings (95%, never change a verdict)


def secondary_s1_s2(
    *, per_seed: dict, averaged: dict, groups: dict, pairs: dict[str, tuple[str, str]], n_seeds: int
) -> dict:
    """Return S1 (first contrast) and S2 (placebo contrast) on the solar block of the sunny folds, in each group."""
    ps = apply_slice(per_seed, solar=True, folds=SUNNY_FOLDS)
    av = apply_slice(averaged, solar=True, folds=SUNNY_FOLDS)
    out = {}
    for label, (t, r) in pairs.items():
        for g in GROUPS_COUNTED:
            c = contrast(
                per_seed=ps,
                averaged=av,
                treatment=t,
                reference=r,
                group=g,
                groups=groups,
                level=LEVEL_SECONDARY,
                band=None,
                n_seeds=n_seeds,
            )
            c["holds"] = bool(c["point"] <= -0.05 and c["upper"] < 0)
            out[f"{label}:{g}"] = c
    return out


def secondary_s4(
    *, per_seed: dict, averaged: dict, groups: dict, pairs: dict[str, tuple[str, str]], n_seeds: int
) -> dict:
    """Return S4 (no harm in the dark): the first contrast on the whole band of the dark folds; it holds when the interval lies inside plus or minus `delta`."""
    ps = apply_slice(per_seed, solar=False, folds=DARK_FOLDS)
    av = apply_slice(averaged, solar=False, folds=DARK_FOLDS)
    out = {}
    for label, (t, r) in pairs.items():
        for g in GROUPS_COUNTED:
            c = contrast(
                per_seed=ps,
                averaged=av,
                treatment=t,
                reference=r,
                group=g,
                groups=groups,
                level=LEVEL_SECONDARY,
                n_seeds=n_seeds,
            )
            c["holds"] = bool(c["lower"] > -c["delta"] and c["upper"] < c["delta"])
            out[f"{label}:{g}"] = c
    return out


def shift_control(*, star_frame: pl.DataFrame, treated_frame: pl.DataFrame) -> pl.DataFrame:
    """Return `M*`'s forecasts with every quantile shifted, per Primary and fold, by the mean difference between the treated arm's median and `M*`'s median over the scored rows.

    Both frames are seed-averaged, hold the same keys, and carry no losses (call `s713.add_losses` on the result).
    """
    assert star_frame.height == treated_frame.height
    joined = star_frame.select(*KEYS, "q50").join(
        treated_frame.select(*KEYS, pl.col("q50").alias("_q50_t")), on=list(KEYS), how="inner"
    )
    assert joined.height == star_frame.height, "the arms hold different keys"
    month = pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
    shift = joined.group_by("time_series_id", month.alias("_m")).agg(
        (pl.col("_q50_t") - pl.col("q50")).mean().alias("_shift")
    )
    out = star_frame.with_columns(_m=month).join(shift, on=["time_series_id", "_m"], how="left")
    out = out.with_columns([(pl.col(q) + pl.col("_shift")).alias(q) for q in QCOLS]).drop(
        "_shift", "_m"
    )
    q = np.sort(out.select(QCOLS).to_numpy().astype(np.float64), axis=1)
    return out.with_columns([pl.Series(name, q[:, i]) for i, name in enumerate(QCOLS)])


def daily_clear_sky_index(
    *, ghi: np.ndarray, clear: np.ndarray, floor_w_m2: float = 50.0
) -> np.ndarray:
    """Return the UTC-day clear-sky index of each Primary, (n_primary, n_days): summed irradiance over summed clear-sky irradiance, in half-hours with clear-sky irradiance above `floor_w_m2`."""
    n_days = ghi.shape[1] // 48
    g = ghi[:, : n_days * 48].reshape(ghi.shape[0], n_days, 48)
    c = clear[:, : n_days * 48].reshape(ghi.shape[0], n_days, 48)
    day = c > floor_w_m2
    num = np.where(day, g, 0.0).sum(axis=2)
    den = np.where(day, c, 0.0).sum(axis=2)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def dullness_thirds(
    *, frame: pl.DataFrame, csi: np.ndarray, prim_of: dict[int, int], grid_start_us: int
) -> pl.DataFrame:
    """Add `third` (0 duller, 1 middle, 2 sunnier) to each row: the valid day's clear-sky index minus the mean over the UTC days 2 to 15 before the origin's date, split into thirds within each Primary and fold."""
    day_us = 86_400 * 1_000_000
    prim = np.array([prim_of[t] for t in frame["time_series_id"].to_list()])
    vday = ((frame["valid_time"].dt.epoch("us").to_numpy() - grid_start_us) // day_us).astype(
        np.int64
    )
    oday = (
        (frame["power_fcst_init_time"].dt.epoch("us").to_numpy() - grid_start_us) // day_us
    ).astype(np.int64)
    window = np.full(len(frame), np.nan)
    for age_lo, age_hi in ((2, 15),):
        cs = np.nancumsum(np.nan_to_num(csi, nan=0.0), axis=1)
        cnt = np.cumsum(np.isfinite(csi), axis=1)
        hi, lo = oday - age_lo, oday - age_hi - 1
        s = cs[prim, hi] - np.where(lo >= 0, cs[prim, np.clip(lo, 0, None)], 0.0)
        n = cnt[prim, hi] - np.where(lo >= 0, cnt[prim, np.clip(lo, 0, None)], 0)
        window = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
    delta = csi[prim, vday] - window
    out = frame.with_columns(
        pl.Series("_delta", delta),
        pl.col("power_fcst_init_time").dt.strftime("%Y-%m").alias("_fold"),
    )
    cuts = out.group_by("time_series_id", "_fold").agg(
        pl.col("_delta").quantile(1 / 3).alias("_c1"), pl.col("_delta").quantile(2 / 3).alias("_c2")
    )
    out = out.join(cuts, on=["time_series_id", "_fold"])
    third = (
        pl.when(pl.col("_delta") <= pl.col("_c1"))
        .then(0)
        .when(pl.col("_delta") <= pl.col("_c2"))
        .then(1)
        .otherwise(2)
    )
    return (
        out.with_columns(third.alias("third"))
        .filter(pl.col("_delta").is_not_nan())
        .drop("_delta", "_fold", "_c1", "_c2")
    )


def secondary_s3(
    *,
    treated: pl.DataFrame,
    star: pl.DataFrame,
    control: pl.DataFrame,
    group: str,
    groups: dict,
    n_resamples: int = 10_000,
    seed: int = 3,
) -> dict:
    """Return S3 for one group: A1 on the duller third minus A1 on the sunnier third, for the treated arm and for the shifted-`M*` control, and the gap between the two differences with a bootstrap interval over origin days.

    All three frames are seed-averaged, hold the same keys, carry losses, and have a `third` column
    (the solar block of the sunny folds). The reading holds when the treated arm's difference of thirds
    lies at least 0.10 below the control's and the interval of the gap is below zero.
    """
    sel = {
        n: s713.restrict(frame=f, group=group, band=s713.HEADLINE_BAND_HOURS, groups=groups)
        for n, f in (("t", treated), ("s", star), ("c", control))
    }
    a1 = s713.paired(treatment=sel["t"], reference=sel["s"], metric="crps").join(
        sel["t"].select(*KEYS, "third"), on=list(KEYS)
    )
    ctl = s713.paired(treatment=sel["c"], reference=sel["s"], metric="crps").select(
        *KEYS, pl.col("d").alias("d_c")
    )
    a1 = a1.join(ctl, on=list(KEYS), how="inner")
    assert a1.height == ctl.height, "the control holds different keys"
    day = a1["power_fcst_init_time"].dt.date().cast(pl.Utf8).to_numpy()
    third = a1["third"].to_numpy()
    d_t, d_c = a1["d"].to_numpy(), a1["d_c"].to_numpy()

    def diff_of_thirds(x: np.ndarray, idx: np.ndarray) -> float:
        return float(x[idx][third[idx] == 0].mean() - x[idx][third[idx] == 2].mean())

    all_idx = np.arange(len(day))
    point_t, point_c = diff_of_thirds(d_t, all_idx), diff_of_thirds(d_c, all_idx)
    labels, inv = np.unique(day, return_inverse=True)
    members = [np.nonzero(inv == i)[0] for i in range(len(labels))]
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_resamples // 10):
        pick = rng.integers(0, len(labels), size=len(labels))
        idx = np.concatenate([members[i] for i in pick])
        gaps.append(diff_of_thirds(d_t, idx) - diff_of_thirds(d_c, idx))
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    gap = point_t - point_c
    return {
        "treated_dull_minus_sunny": point_t, "control_dull_minus_sunny": point_c, "gap": gap, "gap_lower": float(lo), "gap_upper": float(hi),
        "holds": bool(gap <= -0.10 and hi < 0), "n_origin_days": len(labels),
    }  # fmt: skip


def secondary_s5(
    *,
    first_by_primary: dict[int, float],
    pv_share: dict[int, float],
    n_resamples: int = 10_000,
    seed: int = 5,
) -> dict:
    """Return S5: the Spearman correlation between the per-Primary first contrast (solar block, sunny folds) and the PV share, with a bootstrap interval over Primaries."""
    ids = sorted(set(first_by_primary) & set(pv_share))
    x = np.array([pv_share[i] for i in ids])
    y = np.array([first_by_primary[i] for i in ids])
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(ids), size=len(ids))
        if len(np.unique(x[pick])) > 1 and len(np.unique(y[pick])) > 1:
            draws.append(float(stats.spearmanr(x[pick], y[pick]).statistic))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"rho": rho, "lower": float(lo), "upper": float(hi), "n_primaries": len(ids)}


# ---------------------------------------------------------------- report-only tables


def report_only_tables(
    *, per_seed: dict, averaged: dict, groups: dict, treatment: str, reference: str, n_seeds: int
) -> dict:
    """Return one contrast's report-only cuts at 95%: all 16 Primaries pooled, D15 with D18, three lead bands, each Primary, and the normalised MAE of the median beside CRPS."""
    out: dict = {}
    pooled = {**groups, "all16": np.concatenate([groups["six"], groups["eight"], groups["other"]])}
    for g in ("all16", "other"):
        out[f"pooled:{g}"] = contrast(
            per_seed=per_seed,
            averaged=averaged,
            treatment=treatment,
            reference=reference,
            group=g,
            groups=pooled,
            level=LEVEL_SECONDARY,
            n_seeds=n_seeds,
        )
    for g in GROUPS_COUNTED:
        for band in LEAD_BANDS:
            out[f"band{int(band[0])}-{int(band[1])}:{g}"] = contrast(
                per_seed=per_seed,
                averaged=averaged,
                treatment=treatment,
                reference=reference,
                group=g,
                groups=groups,
                level=LEVEL_SECONDARY,
                band=band,
                n_seeds=n_seeds,
            )
        out[f"mae:{g}"] = contrast(
            per_seed=per_seed,
            averaged=averaged,
            treatment=treatment,
            reference=reference,
            group=g,
            groups=groups,
            level=LEVEL_SECONDARY,
            metric="abs_err",
            n_seeds=n_seeds,
        )
    sel = {
        n: s713.restrict(
            frame=averaged[n], group="all16", band=s713.HEADLINE_BAND_HOURS, groups=pooled
        )
        for n in (treatment, reference)
    }
    both = s713.paired(treatment=sel[treatment], reference=sel[reference], metric="crps")
    out["by_primary"] = {
        int(k): float(v)
        for k, v in both.group_by("time_series_id").agg(pl.col("d").mean()).iter_rows()
    }
    return out


# ---------------------------------------------------------------- audit trigger 5, the 16-Primary table, leave-one-out tables


def audit_trigger_5(
    *, per_seed: dict, averaged: dict, groups: dict, part: int, n_seeds: int
) -> dict:
    """Return audit trigger 5 (713's): a nested arm beating its container, `H-S` beating `H` in part 1 and `H` beating `HT` in part 2.

    The trigger fires when the nested arm's contrast against its container reads "gain" or "small gain" in either group.
    """
    nested, container = ("H-S", "H") if part == 1 else ("H", "HT")
    readings = {
        g: reading(
            c=contrast(
                per_seed=per_seed, averaged=averaged, treatment=nested, reference=container, group=g, groups=groups,
                level=LEVEL_COUNTED, n_seeds=n_seeds,
            )
        )
        for g in GROUPS_COUNTED
    }  # fmt: skip
    return {
        "nested": nested,
        "container": container,
        "readings": readings,
        "fires": bool(any(r in {"gain", "small gain"} for r in readings.values())),
    }


def per_primary_table(
    *, per_seed: dict, averaged: dict, pairs: dict[str, tuple[str, str]], n_seeds: int
) -> dict:
    """Return, for every one of the 16 Primaries, each contrast's CRPS point and 95% month-block interval and the normalised MAE of the median (points of capacity).

    Design 7.1 and 11.2: part 2 is scored at every Primary. Each Primary is its own group, so the
    thresholds that belong to the six and the eight do not apply and only the point and the interval are read.
    """
    out: dict = {}
    for pid in range(16):
        groups = {f"p{pid}": np.array([pid])}
        row: dict = {}
        for label, (t, r) in pairs.items():
            crps = contrast(
                per_seed=per_seed,
                averaged=averaged,
                treatment=t,
                reference=r,
                group=f"p{pid}",
                groups=groups,
                level=LEVEL_SECONDARY,
                n_seeds=n_seeds,
            )
            mae = contrast(per_seed=per_seed, averaged=averaged, treatment=t, reference=r, group=f"p{pid}", groups=groups, level=LEVEL_SECONDARY, metric="abs_err", n_seeds=n_seeds)  # fmt: skip
            row[label] = {
                "point": crps["point"],
                "lower": crps["lower"],
                "upper": crps["upper"],
                "mae_point": mae["point"],
            }
        out[pid] = row
    return out


def leave_one_out_tables(*, averaged: dict, groups: dict, treatment: str, reference: str) -> dict:
    """Return the first contrast's CRPS point (headline band) with each Primary left out and with each month left out, per counted group."""
    out: dict = {}
    for g in GROUPS_COUNTED:
        sel = {
            n: s713.restrict(
                frame=averaged[n], group=g, band=s713.HEADLINE_BAND_HOURS, groups=groups
            )
            for n in (treatment, reference)
        }
        d = s713.paired(treatment=sel[treatment], reference=sel[reference], metric="crps")
        by_id = d.group_by("time_series_id").agg(pl.col("d").sum().alias("s"), pl.len().alias("n"))
        total, count = float(by_id["s"].sum()), float(by_id["n"].sum())
        month = (
            d.with_columns(m=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
            .group_by("m")
            .agg(pl.col("d").mean())
            .sort("m")
        )
        months = month["d"].to_numpy()
        out[g] = {
            "all": total / count,
            "without_primary": {int(i): (total - float(s_)) / (count - float(n_)) for i, s_, n_ in by_id.iter_rows()},
            "without_month": {m: float((months.sum() - x) / (len(months) - 1)) for m, x in zip(month["m"], months, strict=True)},
        }  # fmt: skip
    return out


# ---------------------------------------------------------------- one part

PART_PAIRS: Final = {
    1: {"first": ("H", "STAR"), "placebo": ("H", "H-S"), "control": ("H-S", "STAR")},
    2: {"first": ("HT", "H"), "placebo": ("HT", "HT-S"), "control": ("HT-S", "H")},
}
"""Per part: the first contrast (A1 or B1), the placebo contrast (A2 or B2), and the placebo arm against its control."""


def pairs_of_part(*, part: int, star: str) -> dict[str, tuple[str, str]]:
    """Return the three pairs of a part with `STAR` replaced by the name of `M*`."""
    return {k: tuple(star if x == "STAR" else x for x in v) for k, v in PART_PAIRS[part].items()}  # ty: ignore[invalid-return-type]


def score_part(
    *, part: int, star: str, months: tuple[str, ...], seeds: tuple[int, ...], truth: np.ndarray | None = None,
    mask_points: dict | None = None, cold_night_difference: dict | None = None,
) -> dict:  # fmt: skip
    """Return a part's counted contrasts at 98.75% and 99.375%, readings, guards, and verdict, and the report-only cuts."""
    pairs = pairs_of_part(part=part, star=star)
    names = sorted({n for p in pairs.values() for n in p})
    per_seed, averaged, groups = load_all(
        names=names, months=months, seeds=dict.fromkeys(names, seeds), truth=truth
    )
    n_seeds = len(seeds)
    out: dict = {
        "part": part,
        "star": star,
        "seeds": list(seeds),
        "months": list(months),
        "n_common_keys": averaged[names[0]].height,
    }
    for level_name, level in (
        ("counted", LEVEL_COUNTED),
        ("sensitivity_99_375", LEVEL_SENSITIVITY),
    ):
        for label, (t, r) in pairs.items():
            for g in GROUPS_COUNTED:
                out[f"{level_name}:{label}:{g}"] = contrast(
                    per_seed=per_seed,
                    averaged=averaged,
                    treatment=t,
                    reference=r,
                    group=g,
                    groups=groups,
                    level=level,
                    n_seeds=n_seeds,
                )
    readings = {g: reading(c=out[f"counted:first:{g}"]) for g in GROUPS_COUNTED}
    control_reading = {
        g: reading(
            c=contrast(
                per_seed=per_seed,
                averaged=averaged,
                treatment=pairs["control"][0],
                reference=pairs["control"][1],
                group=g,
                groups=groups,
                level=LEVEL_COUNTED,
                n_seeds=n_seeds,
            )
        )
        for g in GROUPS_COUNTED
    }
    placebo = PlaceboReading(
        upper={g: out[f"counted:placebo:{g}"]["upper"] for g in GROUPS_COUNTED}, point={g: out[f"counted:placebo:{g}"]["point"] for g in GROUPS_COUNTED}, control_reading=control_reading,
    )  # fmt: skip
    assert cold_night_difference is not None, (
        "the cold-night guard needs its input: pass `cold_night_difference` (a missing input never passes)"
    )
    assert part == 2 or mask_points is not None, (
        "part 1's mask guard needs the two masked read-outs: pass `mask_points`"
    )
    out["guards"] = supported_guards(
        treatment=pairs["first"][0], reference=pairs["first"][1], averaged=averaged, groups=groups,
        mask_points=mask_points if part == 1 else {"none": {g: out[f"counted:first:{g}"]["point"] for g in GROUPS_COUNTED}},
        cold_night_difference=cold_night_difference, first_sign={g: out[f"counted:first:{g}"]["point"] for g in GROUPS_COUNTED},
    )  # fmt: skip
    out["readings"] = readings
    out["placebo_control_readings"] = control_reading
    out["harmful_placebo"] = placebo.harmful
    out["verdict"] = verdict(
        readings=readings, placebo=placebo, guards_hold=out["guards"]["all_hold"]
    )
    out["closes_line"] = closes_line(verdict_word=out["verdict"])
    out["secondary_s1_s2"] = secondary_s1_s2(
        per_seed=per_seed,
        averaged=averaged,
        groups=groups,
        pairs={"S1": pairs["first"], "S2": pairs["placebo"]},
        n_seeds=n_seeds,
    )
    out["secondary_s4"] = secondary_s4(
        per_seed=per_seed,
        averaged=averaged,
        groups=groups,
        pairs={"S4": pairs["first"]},
        n_seeds=n_seeds,
    )
    out["report_only_first"] = report_only_tables(
        per_seed=per_seed,
        averaged=averaged,
        groups=groups,
        treatment=pairs["first"][0],
        reference=pairs["first"][1],
        n_seeds=n_seeds,
    )
    out["audit_trigger_5"] = audit_trigger_5(
        per_seed=per_seed, averaged=averaged, groups=groups, part=part, n_seeds=n_seeds
    )
    out["per_primary"] = per_primary_table(
        per_seed=per_seed, averaged=averaged, pairs={"first": pairs["first"], "placebo": pairs["placebo"]}, n_seeds=n_seeds
    )  # fmt: skip
    out["leave_one_out"] = leave_one_out_tables(
        averaged=averaged, groups=groups, treatment=pairs["first"][0], reference=pairs["first"][1]
    )
    out["secondary_count_note"] = (
        "14 secondary readings across S1 to S5 (S1, S2, and S4 give 4 each, S3 and S5 one each); one false positive among them is likely"
    )
    return out


def write(*, name: str, obj: object, score_dir: Path) -> None:
    """Write `score/<name>.json` atomically."""
    score_dir.mkdir(parents=True, exist_ok=True)
    path = score_dir / f"{name}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)


def score_dir_of(*, part: int) -> Path:
    """Return the `score/` folder of a part's data folder."""
    return arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="Z" if part == 1 else "T") / "score"


def pinned_seeds(*, n_seeds: int, names: list[str], months: tuple[str, ...]) -> tuple[int, ...]:
    """Return the seed set the manager chose (5 seeds or F4's 2), and refuse when any arm lacks a file of it.

    The set is never read from the files that happen to exist, so that a half-finished phase cannot be
    scored silently at 2 seeds.
    """
    assert n_seeds in {2, 5}, "the seed set is 5 (seeds 0 to 4) or 2 (seeds 0 and 1)"
    seeds = arms.HEADLINE_SEEDS if n_seeds == 5 else arms.TWO_SEEDS
    missing = [
        str(fit_path(name=n, month=m, seed=sd))
        for n in names
        if n in arms.ARMS or "-" in n
        for m in months
        for sd in seeds
        if not fit_path(name=n, month=m, seed=sd).exists()
    ]
    assert not missing, (
        f"{len(missing)} forecast files of the {n_seeds}-seed set are missing, for example {missing[:3]}"
    )
    return seeds


MASK_READOUTS: Final = {"cams": "H-cams", "convention": "H-convention"}
"""Part 1's two masked read-outs of arm `H` (no refit): the CAMS-rule mask and 713's convention mask."""


def mask_points_of(
    *, star: str, months: tuple[str, ...], seeds: tuple[int, ...], truth: np.ndarray | None
) -> dict[str, dict[str, float]]:
    """Return each masked read-out's contrast point against `M*` per counted group (a missing read-out raises)."""
    names = [star, *MASK_READOUTS.values()]
    per_seed, averaged, groups = load_all(
        names=names, months=months, seeds=dict.fromkeys(names, seeds), truth=truth
    )
    return {
        key: {
            g: contrast(per_seed=per_seed, averaged=averaged, treatment=name, reference=star, group=g, groups=groups, level=LEVEL_COUNTED, n_seeds=len(seeds))["point"]
            for g in GROUPS_COUNTED
        }
        for key, name in MASK_READOUTS.items()
    }  # fmt: skip


def cmd_baseline() -> int:
    """WP6a: experiment 713's scorer on `M*` and `C` on all 8 folds, with the solar-block table per group and fold."""
    star = arms713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    seeds = arms713.HEADLINE_SEEDS
    per_seed, averaged, groups = load_all(
        names=[star, "C"], months=arms.ALL_FOLDS, seeds={star: seeds, "C": seeds}
    )
    out: dict = {"star": star, "seeds": list(seeds)}
    for g in (*GROUPS_COUNTED, "other"):
        out[f"crps:{g}"] = {
            n: float(
                s713.restrict(
                    frame=averaged[n], group=g, band=s713.HEADLINE_BAND_HOURS, groups=groups
                )["crps"].mean()
            )
            for n in (star, "C")
        }  # ty: ignore[invalid-argument-type]
        for fold in arms.ALL_FOLDS:
            a = apply_slice(averaged, solar=True, folds=(fold,))
            p = apply_slice(per_seed, solar=True, folds=(fold,))
            res = s713.contrast(
                per_seed=p,
                averaged=a,
                treatment=star,
                reference="C",
                group=g,
                groups=groups,
                level=95.0,
            )
            out[f"solar_block:{g}:{fold}"] = {
                "point": res["point"],
                "lower": res["lower"],
                "upper": res["upper"],
            }
            b = apply_slice(averaged, solar=False, folds=(fold,))
            q = apply_slice(per_seed, solar=False, folds=(fold,))
            res2 = s713.contrast(
                per_seed=q,
                averaged=b,
                treatment=star,
                reference="C",
                group=g,
                groups=groups,
                level=95.0,
            )
            out[f"whole_band:{g}:{fold}"] = {"point": res2["point"]}
    write(name="baseline", obj=out, score_dir=score_dir_of(part=1))
    print(json.dumps(out, indent=1, default=float))
    return 0


def cmd_zrule() -> int:
    """The rule on `Z`: `Z` minus `M*` on its 5 folds, seeds 0 and 1."""
    star = arms713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    per_seed, averaged, groups = load_all(
        names=["Z", star], months=arms.Z_FOLDS, seeds={"Z": arms.TWO_SEEDS, star: arms.TWO_SEEDS}
    )
    out = z_rule(per_seed=per_seed, averaged=averaged, groups=groups, star=star)
    write(name="zrule", obj=out, score_dir=score_dir_of(part=1))
    print(json.dumps(out, indent=1, default=float))
    return 0


def cmd_part(*, part: int, n_seeds: int) -> int:
    """Score a part and write `score/part<N>.json`.

    The cold-night guard reads the dense table, and part 1's mask guard reads the two masked read-outs
    of `H`; a missing input raises, so the verdict never passes a guard it could not compute.
    """
    star = arms713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    pairs = pairs_of_part(part=part, star=star)
    names = sorted({n for p in pairs.values() for n in p if n != star})
    seeds = pinned_seeds(n_seeds=n_seeds, names=names, months=arms.ALL_FOLDS)
    truth = s713.truth_stack()
    mask_points = (
        mask_points_of(star=star, months=arms.ALL_FOLDS, seeds=seeds, truth=truth)
        if part == 1
        else None
    )
    first_t, first_r = pairs["first"]
    per_seed_names = sorted({first_t, first_r})
    _, averaged, groups = load_all(
        names=per_seed_names,
        months=arms.ALL_FOLDS,
        seeds=dict.fromkeys(per_seed_names, seeds),
        truth=truth,
    )
    cold, prim_of, grid_start = load_cold_inputs()
    cold_diff = cold_night_difference(
        averaged=averaged, groups=groups, treatment=first_t, reference=first_r, cold=cold, prim_of=prim_of, grid_start_us=grid_start,
    )  # fmt: skip
    out = score_part(
        part=part, star=star, months=arms.ALL_FOLDS, seeds=seeds, truth=truth, mask_points=mask_points, cold_night_difference=cold_diff,
    )  # fmt: skip
    out["cold_night_difference"] = cold_diff
    out["mask_points"] = mask_points
    out["extra_contrasts"] = report_only_extra(part=part, star=star, seeds=seeds, truth=truth)
    write(name=f"part{part}", obj=out, score_dir=score_dir_of(part=part))
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "readings",
                    "placebo_control_readings",
                    "verdict",
                    "closes_line",
                    "audit_trigger_5",
                )
            },
            indent=1,
        )
    )
    return 0


def extra_pairs(*, part: int, star: str) -> dict[str, tuple[str, str]]:
    """Return the report-only contrasts of design 7.5 that are not counted: nested and cross-arm readings, and every arm against `C`."""
    if part == 1:
        pairs = {"H-X": ("H", "H-X"), "H-S vs M*": ("H-S", star), "H vs Z": ("H", "Z")}
        arm_list = ("H", "H-S", "H-X", "Z")
    else:
        pairs = {"HT vs M*": ("HT", star), "T vs M*": ("T", star), "HT vs T": ("HT", "T")}
        arm_list = ("HT", "HT-S", "T")
    return {**pairs, **{f"{a} vs C": (a, "C") for a in arm_list}}


def report_only_extra(
    *, part: int, star: str, seeds: tuple[int, ...], truth: np.ndarray | None
) -> dict:
    """Return each extra contrast at 95% in both groups, or the reason it could not be read (a missing arm is reported, never guessed)."""
    out: dict = {}
    for label, (t, r) in extra_pairs(part=part, star=star).items():
        try:
            names = [t, r]
            ps, av, groups = load_all(
                names=names, months=arms.ALL_FOLDS, seeds=dict.fromkeys(names, seeds), truth=truth
            )
            out[label] = {
                g: contrast(
                    per_seed=ps,
                    averaged=av,
                    treatment=t,
                    reference=r,
                    group=g,
                    groups=groups,
                    level=LEVEL_SECONDARY,
                    n_seeds=len(seeds),
                )
                for g in GROUPS_COUNTED
            }
        except (FileNotFoundError, AssertionError) as err:
            out[label] = {"not_read": str(err)[:200]}
    return out


def screen_stop_rule(*, first: dict[str, dict], monthly: dict[str, np.ndarray]) -> dict:
    """Return 713's screen stop rule for part 1: `H` minus `M*` above +0.15 in both groups, or above +0.10 in both groups and positive in at least 3 of 4 folds in both groups."""
    big = all(first[g]["point"] > 0.15 for g in GROUPS_COUNTED)
    moderate = all(
        first[g]["point"] > 0.10 and int((np.asarray(monthly[g]) > 0).sum()) >= 3
        for g in GROUPS_COUNTED
    )
    return {
        "stop": bool(big or moderate),
        "above_0_15_both": bool(big),
        "above_0_10_and_3_of_4_both": bool(moderate),
    }


def cmd_screen() -> int:
    """Part 1's stop rule on the four screen folds, seeds 0 and 1 (every number is interim), with fold 2025-11 left out as a second reading."""
    star = arms713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    names = ["H", star]
    seeds = pinned_seeds(n_seeds=2, names=["H"], months=arms.SCREEN_FOLDS)
    per_seed, averaged, groups = load_all(
        names=names, months=arms.SCREEN_FOLDS, seeds=dict.fromkeys(names, seeds)
    )
    first = {
        g: contrast(
            per_seed=per_seed,
            averaged=averaged,
            treatment="H",
            reference=star,
            group=g,
            groups=groups,
            level=LEVEL_SECONDARY,
            n_seeds=2,
        )
        for g in GROUPS_COUNTED
    }
    monthly = {g: np.array(first[g]["monthly"]) for g in GROUPS_COUNTED}
    out = {
        "interim": True,
        "folds": list(arms.SCREEN_FOLDS),
        "first": first,
        "rule": screen_stop_rule(first=first, monthly=monthly),
    }
    without = {g: {**first[g], "point": float(np.mean(monthly[g][1:]))} for g in GROUPS_COUNTED}
    out["rule_without_2025_11"] = screen_stop_rule(
        first=without, monthly={g: monthly[g][1:] for g in GROUPS_COUNTED}
    )
    write(name="screen", obj=out, score_dir=score_dir_of(part=1))
    print(
        json.dumps(
            {"rule": out["rule"], "rule_without_2025_11": out["rule_without_2025_11"]}, indent=1
        )
    )
    return 0


def capacity_share_of_fold(*, fold: str) -> dict[int, float]:
    """Return each Primary's PV capacity over its power capacity at `fold` (the S5 share), from the as-of capacity table."""
    import e00397_prep as prep397  # ty: ignore[unresolved-import]
    import e00716_pv as pv

    rows = list(prep397.primaries().iter_rows(named=True))
    caps = pv.capacity_table(fold=fold, sites=[r["site"] for r in rows])
    return {
        i: (c.mw / float(r["effective_capacity_mw"]) if c.source is not None else 0.0)
        for i, (c, r) in enumerate(zip(caps, rows, strict=True))
    }


def cmd_secondary(*, part: int, n_seeds: int) -> int:
    """S3 (the dull-day gap against the shifted-`M*` control) and S5 (the per-Primary contrast against the PV share), on the solar block of the sunny folds."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00716_pv as pv

    star = arms713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    pairs = pairs_of_part(part=part, star=star)
    treated, reference = pairs["first"]
    names = sorted({treated, reference})
    seeds = pinned_seeds(n_seeds=n_seeds, names=names, months=SUNNY_FOLDS)
    truth = s713.truth_stack()
    raw_t = s713.seed_average(frame=load_arm(arm=treated, seeds=seeds, months=SUNNY_FOLDS))
    raw_r = s713.seed_average(frame=load_arm(arm=reference, seeds=seeds, months=SUNNY_FOLDS))
    raw_c = shift_control(star_frame=raw_r, treated_frame=raw_t)
    frames = {
        n: s713.add_losses(frame=f, truth=truth)
        for n, f in (("t", raw_t), ("r", raw_r), ("c", raw_c))
    }
    sites = k578.sites()
    prim_of = dict(zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True))
    raw = np.load(
        ML_EXPERIMENTS_DATA_DIR
        / "00713-msa-axial-attention-demand-network-part-1"
        / "cache"
        / "dense_raw.npy",
        mmap_mode="r",
    )
    clear = pv.clear_sky(cos_zenith=np.asarray(raw[:, :, 7], dtype=np.float64))
    csi = daily_clear_sky_index(ghi=np.asarray(raw[:, :, 0], dtype=np.float64), clear=clear)
    groups = s713.group_ids()
    sliced = {k: solar_block(in_folds(f, SUNNY_FOLDS)) for k, f in frames.items()}
    with_thirds = {
        k: dullness_thirds(frame=f, csi=csi, prim_of=prim_of, grid_start_us=prep422.GRID_START_US)
        for k, f in sliced.items()
    }
    out: dict = {
        "part": part,
        "seeds": list(seeds),
        "S3": {
            g: secondary_s3(
                treated=with_thirds["t"],
                star=with_thirds["r"],
                control=with_thirds["c"],
                group=g,
                groups=groups,
            )
            for g in GROUPS_COUNTED
        },
    }
    both = s713.paired(treatment=sliced["t"], reference=sliced["r"], metric="crps")
    first_by_primary = {
        int(k): float(v)
        for k, v in both.group_by("time_series_id").agg(pl.col("d").mean()).iter_rows()
    }
    first_by_primary = {prim_of[k]: v for k, v in first_by_primary.items()}
    out["S5"] = secondary_s5(
        first_by_primary=first_by_primary, pv_share=capacity_share_of_fold(fold="2026-06")
    )
    write(name=f"secondary_part{part}", obj=out, score_dir=score_dir_of(part=part))
    print(json.dumps(out, indent=1, default=float))
    return 0


# ---------------------------------------------------------------- self-test


def synthetic_arm(
    *,
    rng: np.random.Generator,
    effect: float,
    months: tuple[str, ...],
    seeds: tuple[int, ...],
    n_origins: int = 6,
    primaries: tuple[int, ...] = tuple(range(16)),
) -> dict:
    """Return per-seed rows with losses for a toy arm: `crps = 1 + effect + noise` on every row of the toy panel."""
    rows = []
    for m_i, month in enumerate(months):
        for p in primaries:
            for o in range(n_origins):
                origin = datetime.fromisoformat(f"{month}-{1 + o:02d}T12:00:00")
                for step in range(0, 14 * 48, 12):
                    rows.append((p, origin, origin + timedelta(minutes=30 * (step + 1)), m_i))
    frame = pl.DataFrame(
        rows, schema=["time_series_id", "power_fcst_init_time", "valid_time", "m"], orient="row"
    ).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )
    out = []
    base = rng.normal(0, 0.3, size=frame.height)
    for seed in seeds:
        noise = rng.normal(0, 0.05, size=frame.height)
        crps = 1.0 + effect + base + noise
        out.append(
            frame.with_columns(
                seed=pl.lit(seed, dtype=pl.Int64),
                crps=pl.Series(crps),
                abs_err=pl.Series(crps * 0.9),
                pin_q01=pl.Series(crps * 0.1),
                pin_q99=pl.Series(crps * 0.1),
                lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time"))
                .dt.total_minutes()
                .cast(pl.Float64)
                / 60.0,
            ).drop("m")
        )
    return {"frame": pl.concat(out), "base": base}


def selftest() -> int:
    """Check the reading table, the verdict table, the placebo guard, the shifted-`M*` control, and a contrast on synthetic frames."""
    s713.N_RESAMPLES = 2000
    c = lambda point, lo, hi, guards=True, delta=0.06, g4=True: {
        "point": point, "lower": lo, "upper": hi, "t_guard": guards, "seed_guard": guards, "floor_guard": guards, "seed_guard_4_of_5": g4, "delta": delta,
    }  # fmt: skip
    assert reading(c=c(0.10, 0.03, 0.2)) == "clear loss"
    assert reading(c=c(0.04, 0.01, 0.07)) == "loss"
    assert reading(c=c(-0.05, -0.1, -0.01)) == "gain"
    assert reading(c=c(-0.01, -0.05, -0.001, guards=False)) == "small gain"
    assert reading(c=c(-0.01, -0.05, -0.001, guards=False, g4=False)) == "no detectable difference"
    assert reading(c=c(0.0, -0.2, 0.2)) == "cannot tell"
    clean = PlaceboReading(
        upper={"six": -0.01, "eight": -0.01},
        point={"six": -0.03, "eight": -0.03},
        control_reading={"six": "cannot tell", "eight": "cannot tell"},
    )
    harmful = PlaceboReading(
        upper={"six": -0.01, "eight": -0.01},
        point={"six": -0.03, "eight": -0.03},
        control_reading={"six": "loss", "eight": "cannot tell"},
    )
    assert harmful.harmful and not clean.harmful
    gain = {"six": "gain", "eight": "gain"}
    assert verdict(readings=gain, placebo=clean, guards_hold=True) == "supported"
    assert verdict(readings=gain, placebo=harmful, guards_hold=True) == "partly supported"
    assert verdict(readings=gain, placebo=clean, guards_hold=False) == "partly supported"
    small = {"six": "small gain", "eight": "no detectable difference"}
    assert verdict(readings=small, placebo=clean, guards_hold=True) == "small gain"
    assert verdict(readings=small, placebo=harmful, guards_hold=True) == "inconclusive", (
        "the S7 guard: a harmful placebo turns a small gain into inconclusive"
    )
    positive = PlaceboReading(
        upper={"six": 0.0, "eight": 0.0},
        point={"six": 0.01, "eight": -0.01},
        control_reading={"six": "cannot tell", "eight": "cannot tell"},
    )
    assert verdict(readings=small, placebo=positive, guards_hold=True) == "inconclusive"
    assert (
        verdict(readings={"six": "clear loss", "eight": "gain"}, placebo=clean, guards_hold=True)
        == "refuted"
    )
    assert (
        verdict(
            readings={"six": "no detectable difference", "eight": "no detectable difference"},
            placebo=clean,
            guards_hold=True,
        )
        == "not supported"
    )
    assert closes_line(verdict_word="not supported") and not closes_line(verdict_word="small gain")
    print("reading table, verdict table, and the harmful-placebo guard: ok")
    # a contrast on a toy panel with a built-in effect
    months = ("2026-01", "2026-02", "2026-03", "2026-04")
    seeds = (0, 1, 2, 3, 4)
    rng = np.random.default_rng(0)
    star = synthetic_arm(rng=rng, effect=0.0, months=months, seeds=seeds)
    treated_frame = star["frame"].with_columns(
        crps=pl.col("crps") - 0.08, abs_err=pl.col("abs_err") - 0.07
    )
    per_seed = {"star": star["frame"], "H": treated_frame}
    averaged = {
        k: v.group_by(*KEYS)
        .agg(
            pl.col("crps").mean(),
            pl.col("abs_err").mean(),
            pl.col("pin_q01").mean(),
            pl.col("pin_q99").mean(),
            pl.col("lead_hours").first(),
        )
        .with_columns(seed=pl.lit(0, dtype=pl.Int64))
        for k, v in per_seed.items()
    }
    ids = np.arange(16)
    groups = {"six": ids[:6], "eight": ids[6:14], "other": ids[14:]}
    res = contrast(
        per_seed=per_seed,
        averaged=averaged,
        treatment="H",
        reference="star",
        group="six",
        groups=groups,
        level=LEVEL_COUNTED,
        n_seeds=5,
    )
    assert (
        abs(res["point"] + 0.08) < 1e-6
        and res["upper"] < 0
        and res["seed_guard"]
        and res["t_guard"]
        and res["floor_guard"]
    ), res
    assert reading(c=res) == "gain"
    res2 = contrast(
        per_seed=per_seed,
        averaged=averaged,
        treatment="H",
        reference="star",
        group="six",
        groups=groups,
        level=LEVEL_COUNTED,
        n_seeds=2,
    )
    assert res2["floor"] == FLOOR_2["six"] and res2["floor_guard"], (
        "0.08 is above the two-seed floor of 0.054"
    )
    assert (
        not contrast(
            per_seed=per_seed,
            averaged=averaged,
            treatment="H",
            reference="star",
            group="six",
            groups=groups,
            level=LEVEL_COUNTED,
            n_seeds=5,
        )["floor"]
        < 0.034
    )
    print(
        f"toy contrast: point {res['point']:.3f}, interval ({res['lower']:.3f}, {res['upper']:.3f}), reading {reading(c=res)}"
    )
    guards = supported_guards(
        treatment="H", reference="star", averaged=averaged, groups=groups, mask_points={"a": {"six": -0.07, "eight": -0.07}, "b": {"six": -0.09, "eight": -0.09}},
        cold_night_difference={"six": 0.0, "eight": 0.0}, first_sign={"six": -0.08, "eight": -0.08},
    )  # fmt: skip
    assert guards["all_hold"], guards
    print("guards of `supported` hold on the toy contrast")
    # the shifted-M* control moves M* by the mean median difference, per Primary and fold
    keys = averaged["star"].select(*KEYS)
    q_star = keys.with_columns([pl.lit(0.1 * (i + 1)).alias(q) for i, q in enumerate(QCOLS)])
    q_treated = keys.with_columns(
        [pl.lit(0.1 * (i + 1) - 0.03).alias(q) for i, q in enumerate(QCOLS)]
    )
    shifted = shift_control(star_frame=q_star, treated_frame=q_treated)
    assert np.allclose(
        shifted["q50"].to_numpy(), q_star["q50"].to_numpy() - 0.03
    ) and shifted.select(KEYS).equals(q_star.select(KEYS))
    print("shifted-M* control moves every quantile by the median difference: ok")
    # daily clear-sky index and thirds
    steps = np.arange(48 * 40)
    clear = np.clip(800 * np.sin(np.pi * ((steps % 48) / 48)), 0, None)[None].repeat(2, axis=0)
    ghi = clear * np.repeat(np.linspace(0.3, 1.0, 40), 48)[None].repeat(2, axis=0)
    csi = daily_clear_sky_index(ghi=ghi, clear=clear)
    assert np.allclose(csi[0], np.linspace(0.3, 1.0, 40)), csi[0][:5]
    print("daily clear-sky index recovers the sky factor: ok")

    def averaged_of(frames: dict) -> dict:
        return {
            k: v.group_by(*KEYS)
            .agg(pl.col("crps").mean(), pl.col("abs_err").mean(), pl.col("pin_q01").mean(), pl.col("pin_q99").mean(), pl.col("lead_hours").first())
            .with_columns(seed=pl.lit(0, dtype=pl.Int64))
            for k, v in frames.items()
        }  # fmt: skip

    # the cold-night guard reads a number and refuses a missing input
    from datetime import UTC

    grid_start = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1e6)
    cold_all = np.ones((16, 48 * 120), dtype=bool)
    diff = cold_night_difference(averaged=averaged, groups=groups, treatment="H", reference="star", cold=cold_all, prim_of={i: i for i in range(16)}, grid_start_us=grid_start)  # fmt: skip
    assert all(abs(v + 0.08) < 1e-6 for v in diff.values()), diff
    try:
        cold_night_difference(averaged=averaged, groups=groups, treatment="H", reference="star", cold=~cold_all, prim_of={i: i for i in range(16)}, grid_start_us=grid_start)  # fmt: skip
    except AssertionError:
        pass
    else:
        raise AssertionError("an empty cold-night slice must raise, not pass")
    print("cold-night guard: reads the difference on cold winter nights, and an empty slice raises")
    # the stop rule of the screen
    first = {g: {"point": 0.2} for g in GROUPS_COUNTED}
    assert screen_stop_rule(
        first=first, monthly={g: np.array([0.3, 0.2, 0.1, -0.1]) for g in GROUPS_COUNTED}
    )["stop"]
    first = {g: {"point": 0.12} for g in GROUPS_COUNTED}
    assert screen_stop_rule(
        first=first, monthly={g: np.array([0.3, 0.2, 0.1, -0.1]) for g in GROUPS_COUNTED}
    )["stop"]
    assert not screen_stop_rule(
        first=first, monthly={g: np.array([0.3, 0.2, -0.1, -0.1]) for g in GROUPS_COUNTED}
    )["stop"]
    assert not screen_stop_rule(
        first={g: {"point": 0.05} for g in GROUPS_COUNTED},
        monthly={g: np.ones(4) for g in GROUPS_COUNTED},
    )["stop"]
    print("screen stop rule: above +0.15, or above +0.10 with 3 of 4 folds, in both groups")
    # audit trigger 5, the 16-Primary table, and the leave-one-out tables
    ps5 = {
        **per_seed,
        "H-S": star["frame"].with_columns(
            crps=pl.col("crps") - 0.30, abs_err=pl.col("abs_err") - 0.27
        ),
    }
    av5 = averaged_of(ps5)
    trig = audit_trigger_5(per_seed=ps5, averaged=av5, groups=groups, part=1, n_seeds=5)
    assert trig["fires"] and trig["nested"] == "H-S", trig
    ps5["H-S"] = star["frame"].with_columns(crps=pl.col("crps") + 0.10)
    assert not audit_trigger_5(
        per_seed=ps5, averaged=averaged_of(ps5), groups=groups, part=1, n_seeds=5
    )["fires"]
    table = per_primary_table(
        per_seed=per_seed,
        averaged=averaged,
        pairs={"first": ("H", "star"), "placebo": ("H", "star")},
        n_seeds=5,
    )
    assert len(table) == 16 and all(
        abs(r["first"]["point"] + 0.08) < 1e-6 and abs(r["first"]["mae_point"] + 0.07) < 1e-6
        for r in table.values()
    )
    loo = leave_one_out_tables(averaged=averaged, groups=groups, treatment="H", reference="star")
    assert (
        len(loo["six"]["without_primary"]) == 6
        and len(loo["six"]["without_month"]) == 4
        and abs(loo["six"]["all"] + 0.08) < 1e-6
    )
    print(
        "audit trigger 5 fires when a nested arm beats its container; the 16-Primary table and the leave-one-out tables have the right shape"
    )
    # the reading `supported` is reachable once every guard has an input
    assert (
        verdict(
            readings={"six": "gain", "eight": "gain"}, placebo=clean, guards_hold=guards["all_hold"]
        )
        == "supported"
    )
    print("selftest passed")
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        choices=("baseline", "zrule", "part1", "part2", "screen", "secondary1", "secondary2"),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        choices=(2, 5),
        help="the manager's seed set (required to score a part): 5 seeds or 2",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.command in {"part1", "part2", "secondary1", "secondary2"}:
        assert args.n_seeds is not None, "pass --n-seeds 5 or --n-seeds 2"
    return {
        "baseline": cmd_baseline,
        "zrule": cmd_zrule,
        "screen": cmd_screen,
        "part1": lambda: cmd_part(part=1, n_seeds=args.n_seeds),
        "part2": lambda: cmd_part(part=2, n_seeds=args.n_seeds),
        "secondary1": lambda: cmd_secondary(part=1, n_seeds=args.n_seeds),
        "secondary2": lambda: cmd_secondary(part=2, n_seeds=args.n_seeds),
    }[args.command]()


if __name__ == "__main__":
    sys.exit(main())

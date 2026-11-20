# ruff: noqa: E501
"""Experiment 713: losses, contrasts, the three guards, the decision rule, and the screen's rules.

Usage:
    e00713_score.py candidate            # KA8: the stored Candidate alone (run this first)
    e00713_score.py screen               # sanity and stop rules on the 4 screen folds
    e00713_score.py headline             # H1 and H2 on the 8 folds, 5 seeds, guards, and the verdict
    e00713_score.py report-only          # the 95% report-only readings of section 1.4 (seeds 0 and 1)
    e00713_score.py swaps                # history-weather swaps (a), (b), (c) of section 4.3
    e00713_score.py write-predictions    # seed-averaged sets into the experiment store (once per arm)

Every table is printed and written to `score/<command>.json` in the experiment's data folder. All
losses are in points of capacity (CRPS is twice the mean pinball loss over the 11 quantile levels,
times 100; the normalised MAE of the median is the absolute error of the median times 100, over
capacity). A forecast file is converted to the contract's origin (12:00 UTC, the native origin plus
3 hours) with valid times from 12:00 UTC to 14 days later, valid times from 2026-07-01 are dropped, and
rows with no cleaned actual are dropped for every arm alike.

The climatology row, the slices, the threshold-weighted CRPS, the tails, the four load-shift
scenarios, the leave-one-out tables, and the avoidable-curtailment cost are in `e00713_protocol.py`.
"""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00667_export as exp  # ty: ignore[unresolved-import]
import e00667_score as s667  # ty: ignore[unresolved-import]
import e00713_arms as arms
import e00713_inputs as inp
import numpy as np
import polars as pl
from scipy import stats
from studies.eval_protocol.metrics import quantile_score
from studies.sources import ML_EXPERIMENTS_DATA_DIR

from studies import bootstrap, experiment_store

QCOLS: Final = exp.QCOLS
KEYS: Final = exp.KEYS
SIX: Final = ("D5", "D9", "D10", "D11", "D14", "D16")
EIGHT: Final = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
OTHER: Final = ("D15", "D18")
TAU: Final = 0.05
DELTA: Final = {"six": 0.06, "eight": 0.16}
NOISE_FLOOR: Final = {"six": 0.034, "eight": 0.090}
"""Twice the seed-noise standard error at 5 seeds (design 1.4, 1.5)."""
N_COUNTED: Final = 4
LEVEL: Final = 100.0 * (1.0 - 0.05 / N_COUNTED)
N_RESAMPLES: Final = 100_000
HEADLINE_BAND_HOURS: Final = (24.0, 240.0)
KA8_EXPECTED: Final = {
    "six": (3.203, 3.175, 0.001, 0.108, 611_055),
    "eight": (3.704, 3.694, 0.059, 0.285, 816_768),
    "other": (8.106, 8.002, 0.127, 0.974, 203_040),
}
"""Section 1.5: pooled CRPS of the seed mean, mean of monthly means, mean monthly seed difference, its sd, half-hours."""
SCORE_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / arms.NAME / "score"
MONTH_ARM: Final = "avg"


def group_ids() -> dict[str, np.ndarray]:
    """Return the `time_series_id`s of the three Primary groups."""
    sites = k578.sites()
    out = {}
    for name, labels in (("six", SIX), ("eight", EIGHT), ("other", OTHER)):
        out[name] = sites.filter(pl.col("label").is_in(labels))["time_series_id"].to_numpy()
    return out


def to_contract_shares(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Relabel the native origin (09:00) as the contract origin (12:00), keep 12:00 to +14 days, drop the holdout."""
    origin = pl.col("power_fcst_init_time") + timedelta(hours=3)
    return (
        frame.with_columns(power_fcst_init_time=origin)
        .filter(
            (pl.col("valid_time") >= pl.col("power_fcst_init_time"))
            & (pl.col("valid_time") < pl.col("power_fcst_init_time") + timedelta(days=14))
            & (pl.col("valid_time") < experiment_store.HOLDOUT_START)
        )
        .with_columns(pl.col("time_series_id").cast(pl.Int64))
    )


def candidate_files(*, month: str, seed: int) -> Path:
    """Return the Candidate's per-seed forecast file: experiment 667's for seeds 0 and 1, else the refit."""
    root = exp.ROOT if seed in {0, 1} else inp.CANDIDATE_SEEDS_ROOT
    return root / f"P7_D50_{month}" / "work" / "fits" / "P5" / f"{month}_d0_s{seed}.parquet"


SWAP_C: Final = "C-swapc"
"""The Candidate predicting with ERA5 at its lag times (design 4.3, swap (c))."""


def swap_c_file(*, month: str, seed: int) -> Path:
    """Return the forecast file that `e00713_swap_c.py` wrote."""
    return (
        ML_EXPERIMENTS_DATA_DIR / arms.NAME / "swap_c" / f"P7_D50_{month}" / "work" / "fits" / "P5"
        / f"{month}_d0_s{seed}.parquet"
    )  # fmt: skip


def load_arm(*, arm: str, seeds: tuple[int, ...], months: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm's per-seed rows in contract origin and capacity shares, with a `seed` column."""
    parts = []
    for month in months:
        for seed in seeds:
            path = (
                candidate_files(month=month, seed=seed)
                if arm == "C"
                else swap_c_file(month=month, seed=seed)
                if arm == SWAP_C
                else arms.fit_file(
                    data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed
                )
            )
            frame = pl.read_parquet(path, columns=[*KEYS, *QCOLS]).with_columns(
                seed=pl.lit(seed, dtype=pl.Int64)
            )
            parts.append(to_contract_shares(frame=frame))
    return pl.concat(parts)


def truth_stack() -> np.ndarray:
    """Return the cleaned actual power over capacity, (16, grid), NaN where missing."""
    return s667.observed_stack(grid_start_us=prep422.GRID_START_US, step_us=prep422.STEP_US)


def add_losses(*, frame: pl.DataFrame, truth: np.ndarray) -> pl.DataFrame:
    """Add the actual, `crps`, `abs_err`, `lead_hours`, coverage and pinball columns; drop unscored rows."""
    sites = k578.sites()
    prim_of = dict(zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True))
    prim = np.array([prim_of[t] for t in frame["time_series_id"].to_list()])
    idx = (
        (frame["valid_time"].dt.epoch("us").to_numpy() - prep422.GRID_START_US) // prep422.STEP_US
    ).astype(int)
    ok_idx = (idx >= 0) & (idx < truth.shape[1])
    actual = np.where(ok_idx, truth[prim, np.clip(idx, 0, truth.shape[1] - 1)], np.nan)
    q = frame.select(QCOLS).to_numpy().astype(np.float64)
    pts = 100.0
    crps = quantile_score(actual=actual * pts, quantiles=q * pts)
    lead = (
        frame["valid_time"] - frame["power_fcst_init_time"]
    ).dt.total_minutes().to_numpy() / 60.0
    pin = lambda i: np.maximum(  # noqa: E731
        LEVELS_LIST[i] * (actual - q[:, i]), (LEVELS_LIST[i] - 1.0) * (actual - q[:, i])
    )
    out = frame.with_columns(
        pl.Series("actual", actual),
        pl.Series("crps", crps),
        pl.Series("abs_err", pts * np.abs(actual - q[:, 5])),
        pl.Series("lead_hours", lead),
        pl.Series("pin_q01", pts * pin(0)),
        pl.Series("pin_q05", pts * pin(1)),
        pl.Series("pin_q95", pts * pin(9)),
        pl.Series("pin_q99", pts * pin(10)),
        pl.Series("cover_q05_q95", ((actual >= q[:, 1]) & (actual <= q[:, 9])).astype(float)),
        pl.Series("width_q05_q95", pts * (q[:, 9] - q[:, 1])),
    )
    return out.filter(pl.col("actual").is_not_nan() & pl.col("actual").is_not_null())


LEVELS_LIST: Final = (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99)


def seed_average(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the seed mean of each quantile, sorted within the row (experiment 667's rule), with `seed` 0."""
    cat = frame.group_by(KEYS).agg([pl.col(q).mean() for q in QCOLS]).sort(KEYS)
    q = np.sort(cat.select(QCOLS).to_numpy().astype(np.float32), axis=1)
    mean = cat.with_columns([pl.Series(name, q[:, i]) for i, name in enumerate(QCOLS)])
    return mean.with_columns(seed=pl.lit(0, dtype=pl.Int64))


def assert_same_keys(*, frames: dict[str, pl.DataFrame]) -> None:
    """The join assertion: every arm holds the same (time_series_id, power_fcst_init_time, valid_time) keys."""
    first, *rest = list(frames)
    ref = frames[first].select(KEYS).unique().sort(KEYS)
    for name in rest:
        other = frames[name].select(KEYS).unique().sort(KEYS)
        assert ref.equals(other), f"arms {first!r} and {name!r} hold different keys"


def restrict(
    *,
    frame: pl.DataFrame,
    group: str,
    band: tuple[float, float] | None,
    groups: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Return the rows of a Primary group and, optionally, a lead band in hours (low inclusive, high exclusive)."""
    out = frame.filter(pl.col("time_series_id").is_in(groups[group].tolist()))
    if band is not None:
        out = out.filter((pl.col("lead_hours") >= band[0]) & (pl.col("lead_hours") < band[1]))
    return out


def t_interval(*, monthly: np.ndarray, level: float) -> tuple[float, float]:
    """Return the t interval on monthly differences at `level` percent (n - 1 degrees of freedom)."""
    n = len(monthly)
    half = stats.t.ppf(1.0 - (1.0 - level / 100.0) / 2.0, n - 1) * monthly.std(ddof=1) / np.sqrt(n)
    return float(monthly.mean() - half), float(monthly.mean() + half)


def contrast(
    *, per_seed: dict[str, pl.DataFrame], averaged: dict[str, pl.DataFrame], treatment: str,
    reference: str, group: str, groups: dict[str, np.ndarray], metric: str = "crps",
    level: float = LEVEL, band: tuple[float, float] | None = HEADLINE_BAND_HOURS,
) -> dict:  # fmt: skip
    """Return one treatment-minus-reference contrast with its bootstrap interval and the three guards."""
    sel = {
        n: restrict(frame=averaged[n], group=group, band=band, groups=groups)
        for n in (treatment, reference)
    }
    loss = experiment_store.losses_for_bootstrap(
        arm_losses={n: sel[n].select(*KEYS, "seed", metric) for n in sel}
    )
    lo, hi = bootstrap.bootstrap_difference_at_level(
        losses=loss, treatment=treatment, reference=reference, metric=metric, level=level, n_resamples=N_RESAMPLES,
    )  # fmt: skip
    t_row = sel[treatment].sort(KEYS)
    r_row = sel[reference].sort(KEYS)
    diff = t_row[metric].to_numpy() - r_row[metric].to_numpy()
    months = t_row["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    monthly = np.array([diff[months == m].mean() for m in np.unique(months)])
    point = float(diff.mean())
    t_lo, t_hi = t_interval(monthly=monthly, level=level)
    sign = 1.0 if point > 0 else -1.0
    t_guard = (t_lo > 0) if sign > 0 else (t_hi < 0)
    seeds = sorted(
        set(per_seed[treatment]["seed"].unique().to_list())
        & set(per_seed[reference]["seed"].unique().to_list())
    )
    seed_diffs = []
    for s in seeds:
        a = restrict(
            frame=per_seed[treatment].filter(pl.col("seed") == s),
            group=group,
            band=band,
            groups=groups,
        ).sort(KEYS)
        b = restrict(
            frame=per_seed[reference].filter(pl.col("seed") == s),
            group=group,
            band=band,
            groups=groups,
        ).sort(KEYS)
        seed_diffs.append(float((a[metric].to_numpy() - b[metric].to_numpy()).mean()))
    same_sign = sum(1 for d in seed_diffs if d * sign > 0)
    floor = NOISE_FLOOR.get(group)
    return {
        "treatment": treatment, "reference": reference, "group": group, "metric": metric, "level": level,
        "point": point, "lower": lo, "upper": hi, "t_lower": t_lo, "t_upper": t_hi,
        "monthly": monthly.tolist(), "seed_differences": seed_diffs, "seeds_same_sign": same_sign,
        "n_seeds": len(seeds), "t_guard": bool(t_guard),
        "seed_guard": same_sign == len(seeds), "seed_guard_4_of_5": same_sign >= max(len(seeds) - 1, 1),
        "floor_guard": bool(floor is not None and abs(point) >= floor),
        "n_rows": len(diff),
    }  # fmt: skip


def group_reading(*, c: dict, group: str) -> str:
    """Return the group's reading of H1 by the first row of the table of section 1.8 that holds."""
    d, lo, hi = c["point"], c["lower"], c["upper"]
    guards = c["t_guard"] and c["seed_guard"] and c["floor_guard"]
    if lo > TAU and guards:
        return "clear loss"
    if lo > 0 and guards:
        return "loss"
    if hi < 0 and d <= -TAU and guards:
        return "gain"
    if hi < 0 and c["seed_guard_4_of_5"]:
        return "small gain"
    if lo > -DELTA[group] and hi < DELTA[group]:
        return "no detectable difference"
    return "cannot tell"


def verdict(  # noqa: PLR0911
    *, readings: dict[str, str], h2_upper: dict[str, float], guards_hold: bool = True
) -> str:
    """Return the experiment's verdict by the first row of the verdict table of section 1.8."""
    r = list(readings.values())
    if "clear loss" in r:
        return "refuted"
    if "loss" in r or all(x == "no detectable difference" for x in r):
        return "not supported"
    if all(x == "gain" for x in r):
        if all(u < 0 for u in h2_upper.values()) and guards_hold:
            return "supported"
        return "partly supported"
    if "gain" in r and "loss" not in r:
        return "partly supported"
    if "small gain" in r and "gain" not in r and "loss" not in r:
        return "small gain"
    return "inconclusive"


def screen_rules(
    *, h1: dict[str, dict], h1_months: dict[str, np.ndarray], h2: dict[str, dict]
) -> dict:
    """Return the sanity rule and the stop rule of section 1.6 from the screen contrasts of `M*`."""
    sanity_stop = all(h2[g]["point"] > 0 for g in ("six", "eight"))
    big = all(h1[g]["point"] > 0.15 for g in ("six", "eight"))
    moderate = all(
        h1[g]["point"] > 0.10 and (h1_months[g] > 0).sum() >= 3 for g in ("six", "eight")
    )
    return {"sanity_stop_and_audit": bool(sanity_stop), "stop_rule": bool(big or moderate)}


def cmd_candidate() -> int:
    """KA8: reproduce the table of section 1.5 and `exports/NRA_<month>.parquet` from the per-seed files."""
    groups = group_ids()
    truth = truth_stack()
    months = arms.ALL_FOLDS
    per_seed = add_losses(frame=load_arm(arm="C", seeds=(0, 1), months=months), truth=truth)
    averaged = add_losses(
        frame=seed_average(frame=load_arm(arm="C", seeds=(0, 1), months=months)), truth=truth
    )
    table = {}
    for g, expected in KA8_EXPECTED.items():
        a = restrict(frame=averaged, group=g, band=HEADLINE_BAND_HOURS, groups=groups)
        by_month = (
            a.group_by(a["power_fcst_init_time"].dt.strftime("%Y-%m").alias("month"))
            .agg(pl.col("crps").mean())
            .sort("month")
        )
        s = restrict(frame=per_seed, group=g, band=HEADLINE_BAND_HOURS, groups=groups)
        diff = (
            s.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
            .group_by("month", "seed")
            .agg(pl.col("crps").mean())
            .pivot(on="seed", index="month", values="crps")
            .sort("month")
        )
        d = diff["0"].to_numpy() - diff["1"].to_numpy()
        got = (
            float(a["crps"].mean()),  # ty: ignore[invalid-argument-type]
            float(by_month["crps"].mean()),  # ty: ignore[invalid-argument-type]
            float(d.mean()),
            float(d.std(ddof=1)),
            a.height,
        )
        table[g] = got
        ok = (
            all(abs(x - y) < 5e-4 for x, y in zip(got[:4], expected[:4], strict=True))
            and got[4] == expected[4]
        )
        print(
            g, "got", [round(v, 3) for v in got], "expected", expected, "OK" if ok else "MISMATCH"
        )
        assert ok, f"KA8 failed for {g}"
    # the export: the seed mean in MW equals experiment 667's exports/NRA_<month>.parquet
    for month in months:
        mine = exp.contract_rows(
            mean=exp.seed_mean(
                frames=[
                    pl.read_parquet(
                        candidate_files(month=month, seed=s), columns=[*KEYS, *exp.QCOLS]
                    )
                    for s in (0, 1)
                ]
            )
        )
        stored = pl.read_parquet(exp.ROOT / "exports" / f"NRA_{month}.parquet").select(mine.columns)
        assert mine.select(KEYS).equals(stored.select(KEYS))
        worst = float(np.abs(mine.select(QCOLS).to_numpy() - stored.select(QCOLS).to_numpy()).max())
        assert worst <= 1e-4, (month, worst)
        for s in (0, 1):
            path = candidate_files(month=month, seed=s)
            print(month, s, path.stat().st_mtime, "export max abs diff (MW)", worst)
    write(name="candidate", obj=table)
    return 0


def write(*, name: str, obj: object) -> None:
    """Write `score/<name>.json` atomically and print it."""
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    path = SCORE_DIR / f"{name}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)
    print(json.dumps(obj, indent=1, default=float))


def load_scored(
    *, arm: str, seeds: tuple[int, ...], months: tuple[str, ...], truth: np.ndarray
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return an arm's per-seed rows and its seed-averaged rows, both with losses."""
    raw = load_arm(arm=arm, seeds=seeds, months=months)
    return add_losses(frame=raw, truth=truth), add_losses(
        frame=seed_average(frame=raw), truth=truth
    )


def run_contrasts(
    *,
    star: str,
    months: tuple[str, ...],
    seeds: tuple[int, ...],
    pairs: list[tuple[str, str]],
    level: float,
) -> dict:
    """Score contrasts (treatment, reference) for the six and the eight."""
    groups, truth = group_ids(), truth_stack()
    names = sorted({n for p in pairs for n in p})
    per_seed, averaged = {}, {}
    for n in names:
        per_seed[n], averaged[n] = load_scored(
            arm="C" if n == "C" else n, seeds=seeds, months=months, truth=truth
        )
    # all arms share the scored keys: restrict each to the intersection of the keys, then assert
    common = None
    for n in names:
        k = averaged[n].select(KEYS).unique()
        common = k if common is None else common.join(k, on=list(KEYS), how="inner")
    for n in names:
        per_seed[n] = per_seed[n].join(common, on=list(KEYS), how="inner")
        averaged[n] = averaged[n].join(common, on=list(KEYS), how="inner")
    assert_same_keys(frames=averaged)
    out: dict = {
        "star": star,
        "level": level,
        "n_common_keys": (common.height if common is not None else 0),
    }
    for t, r in pairs:
        for g in ("six", "eight"):
            out[f"{t}-{r}:{g}"] = contrast(
                per_seed=per_seed,
                averaged=averaged,
                treatment=t,
                reference=r,
                group=g,
                groups=groups,
                level=level,
            )
    return out


def cmd_screen() -> int:
    """Sanity rule and stop rule on the screen folds, with fold 2025-11 left out as a second reading."""
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    res = run_contrasts(
        star=star,
        months=arms.SCREEN_FOLDS,
        seeds=arms.SCREEN_SEEDS,
        pairs=[(star, "C"), (star, "P")],
        level=95.0,
    )
    h1 = {g: res[f"{star}-C:{g}"] for g in ("six", "eight")}
    h2 = {g: res[f"{star}-P:{g}"] for g in ("six", "eight")}
    months = {g: np.array(h1[g]["monthly"]) for g in h1}
    res["rules"] = screen_rules(h1=h1, h1_months=months, h2=h2)
    res["rules_without_2025_11"] = screen_rules(
        h1={g: {**h1[g], "point": float(np.mean(h1[g]["monthly"][1:]))} for g in h1},
        h1_months={g: months[g][1:] for g in h1}, h2=h2,
    )  # fmt: skip
    write(name="screen", obj=res)
    return 0


def cmd_headline() -> int:
    """H1 and H2 at 98.75% with 5 seeds, the three guards, the readings, and the verdict."""
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    res = run_contrasts(
        star=star,
        months=arms.ALL_FOLDS,
        seeds=arms.HEADLINE_SEEDS,
        pairs=[(star, "C"), (star, "P")],
        level=LEVEL,
    )
    readings = {g: group_reading(c=res[f"{star}-C:{g}"], group=g) for g in ("six", "eight")}
    h2_upper = {g: res[f"{star}-P:{g}"]["upper"] for g in ("six", "eight")}
    res["readings"] = readings
    res["h2_upper"] = h2_upper
    res["verdict"] = verdict(readings=readings, h2_upper=h2_upper)
    res["note"] = (
        "the `supported` guards (leave-one-out, q01/q99, MAE, swap (a)) are read from the report-only tables"
    )
    write(name="headline", obj=res)
    return 0


def cmd_swaps() -> int:
    """Report-only: the history-weather swaps (a), (b), and (c) of design 4.3 against the Candidate (seeds 0 and 1).

    Swap (a) alone feeds the `supported` guard. Swap (b) minus swap (a) is the effect of the 14 B1
    columns, and swap (c) says how much the Candidate gains from ERA5 at its lag times.
    """
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    res = run_contrasts(
        star=star,
        months=arms.ALL_FOLDS,
        seeds=arms.SCREEN_SEEDS,
        pairs=[
            (f"{star}-swapa", "C"),
            (f"{star}-swapb", "C"),
            (f"{star}-swapb", f"{star}-swapa"),
            (star, f"{star}-swapa"),
            (SWAP_C, "C"),
        ],
        level=95.0,
    )
    res["guard_swap_a"] = {
        g: group_reading(c=res[f"{star}-swapa-C:{g}"], group=g) for g in ("six", "eight")
    }
    write(name="swaps", obj=res)
    return 0


def cmd_report_only() -> int:
    """The 95% report-only readings of section 1.4 on seeds 0 and 1 and the normalised MAE beside CRPS."""
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    pairs = [
        (star, "A"),
        (star, "K"),
        ("K", "N"),
        (star, "N"),
        ("E", star),
        (star, "C"),
        (star, "P"),
    ]
    out = {}
    for metric in ("crps", "abs_err"):
        groups, truth = group_ids(), truth_stack()
        names = sorted({n for p in pairs for n in p})
        per_seed, averaged = {}, {}
        for n in names:
            per_seed[n], averaged[n] = load_scored(
                arm=n, seeds=arms.SCREEN_SEEDS, months=arms.ALL_FOLDS, truth=truth
            )
        for t, r in pairs:
            for g in ("six", "eight"):
                out[f"{metric}:{t}-{r}:{g}"] = contrast(
                    per_seed=per_seed,
                    averaged=averaged,
                    treatment=t,
                    reference=r,
                    group=g,
                    groups=groups,
                    metric=metric,
                    level=95.0,
                )
    write(name="report_only", obj=out)
    return 0


def cmd_write_predictions() -> int:
    """Write each fitted arm's seed-averaged forecasts (contract shares) into the experiment store."""
    exp_dir = next(ML_EXPERIMENTS_DATA_DIR.glob("00713-*"))
    experiment_id = int(exp_dir.name.split("-")[0])
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    for arm in (star, "P", "A", "K", "N", "E"):
        seeds = arms.HEADLINE_SEEDS if arm in {star, "P"} else arms.SCREEN_SEEDS
        frame = seed_average(frame=load_arm(arm=arm, seeds=seeds, months=arms.ALL_FOLDS))
        experiment_store.write_predictions(
            experiment_id=experiment_id, name=f"{arm.lower()}-seed-mean", frame=frame, kind="quantiles", row_set_id=arms.ROW_SET_ID,
        )  # fmt: skip
    return 0


def main() -> int:
    """Dispatch the sub-command."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("candidate", "screen", "headline", "report-only", "swaps", "write-predictions"),
    )
    args = parser.parse_args()
    return {
        "candidate": cmd_candidate, "screen": cmd_screen, "headline": cmd_headline,
        "report-only": cmd_report_only, "swaps": cmd_swaps, "write-predictions": cmd_write_predictions,
    }[args.command]()  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())

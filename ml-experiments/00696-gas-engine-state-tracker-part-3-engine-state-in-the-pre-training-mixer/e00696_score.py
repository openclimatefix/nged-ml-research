# ruff: noqa: E501, PT018, C416, FURB171
r"""Score experiment 696: four planned contrasts at 98.75%, the encoder probe M1, and the verdict table.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00696-*/e00696_score.py [--resamples N]

Reads each arm's `runs/<ARM>/arms/k1_5w_sp1/work/fits/P5/<month>_d0_s<seed>.parquet` (quantiles as
fractions of effective capacity, at the network's 09:00 UTC origin; seeds 0 to 2 of arms A0 and W0 come
from experiment 695's folder), relabels the origin to the contract's 12:00 UTC, restricts every arm to
the decision window, and scores against the metered reading (D8) or the injected reading (the host).
Nothing here fits a network. Every arm's forecast is the network's own 11 quantiles, with no add-back.

**Contrasts.** C1: S1 minus A0 at D8. C2: S1 minus SP at D8. C3: WST minus W0 at the host. C4:
WS1 minus W0 at the host. Each is read at 98.75% (Bonferroni for four), and has three decision
intervals: a paired t interval on the 8 monthly differences of the seed means, a bootstrap over
whole origin months of the seed-averaged losses, and a paired t interval on the per-seed mean
differences. A contrast passes when all three lie below zero. The library interval
(`studies.bootstrap`, one seed per resample) is printed beside them. The encoder probe M1 compares
the probe's held-out error of WST with W0's over the 8 folds.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00695_score as s695
import numpy as np
import polars as pl
from e00695_common import (
    D8_CAPACITY_MVA,
    D8_SERIES,
    ENGINE_MONTHS,
    FADE_MONTHS,
    QUANTILE_COLUMNS,
    SCORED_MONTHS,
    SHARED_DIR,
    d8_truth,
)
from e00695_score import (
    KEYS,
    NATIVE_TO_CONTRACT,
    TAIL_NAMES,
    capacity_of,
    contrast,
    grid_truth,
    host_series_id,
    row_scores,
    scored_window,
)
from e00696_common import (
    ARMS,
    DATA_DIR,
    DESCRIPTIVE_LEVEL,
    DESCRIPTIVE_SEEDS,
    E695_DIR,
    FAMILY_LEVEL,
    FAMILY_RESAMPLES,
    PROBE_LEVEL,
    REUSED_SEEDS,
    ROW_SET_ID,
    SEEDS,
    THRESHOLD_POINTS,
    fits_roots,
)
from e00696_verdict import VerdictInputs, verdict
from scipy import stats
from studies.bootstrap import paired_differences
from studies.experiment_store import losses_for_bootstrap, write_predictions

SCORES_DIR: Final[Path] = DATA_DIR / "scores"
CHECKS_PATH: Final[Path] = DATA_DIR / "checks.json"
DAYS_IN_MONTH: Final[dict[str, int]] = {
    "2025-11": 30,
    "2025-12": 31,
    "2026-01": 31,
    "2026-02": 28,
    "2026-03": 31,
    "2026-04": 30,
    "2026-05": 31,
    "2026-06": 30,
}
"""Origins per fold month; the 8 months sum to 242."""
BOOTSTRAP_SEED: Final[int] = 0


@dataclass(frozen=True)
class System:
    """One scored forecast: which arm's fits and which series."""

    name: str
    series: str
    """`d8` or `host`."""


SYSTEMS: Final[tuple[System, ...]] = (
    System("A0", "d8"),
    System("S1", "d8"),
    System("SP", "d8"),
    System("S1d", "d8"),
    System("W0", "host"),
    System("WST", "host"),
    System("WS1", "host"),
)


def read_fits(*, arm: str) -> pl.DataFrame:
    """Return an arm's quantiles (fractions of capacity), every series, with the contract origin and the seed.

    Seeds 0 to 2 of arms A0 and W0 come from experiment 695's folder. The native origin is 09:00 UTC,
    relabelled to 12:00 UTC. The quantiles are sorted within the row.
    """
    frames = []
    for seed, root in fits_roots(arm=arm).items():
        for month in SCORED_MONTHS:
            path = root / f"{month}_d0_s{seed}.parquet"
            frames.append(pl.read_parquet(path).with_columns(seed=pl.lit(seed, dtype=pl.Int64)))
    frame = pl.concat(frames)
    assert (frame["power_fcst_init_time"].dt.hour() == 9).all(), "a native origin is not 09:00 UTC"
    q = np.sort(frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64), axis=1)
    return frame.with_columns(
        time_series_id=pl.col("time_series_id").cast(pl.Int64),
        power_fcst_init_time=pl.col("power_fcst_init_time") + NATIVE_TO_CONTRACT,
        **{name: pl.Series(name, q[:, i]) for i, name in enumerate(QUANTILE_COLUMNS)},
    ).select(*KEYS, "seed", *QUANTILE_COLUMNS)


def forecast_of(*, fits: pl.DataFrame, series_id: int, capacity: float) -> pl.DataFrame:
    """Return one series' forecast on the decision window, in MVA."""
    return scored_window(
        frame=fits.filter(pl.col("time_series_id") == series_id).with_columns(
            *[pl.col(q) * capacity for q in QUANTILE_COLUMNS]
        )
    )


def interval(*, values: np.ndarray, level: float) -> tuple[float, float]:
    """Return the paired t interval of the mean of `values` at `level` percent."""
    low, high = stats.t.interval(
        level / 100.0, df=len(values) - 1, loc=values.mean(), scale=stats.sem(values)
    )
    return float(low), float(high)


def month_bootstrap(
    *, month_sums: np.ndarray, month_rows: np.ndarray, level: float, resamples: int
) -> tuple[float, float]:
    """Return a percentile interval of the row-weighted mean difference over resamples of whole months.

    Each resample draws as many months as there are, with replacement, and takes the sum of their
    seed-averaged differences over their number of rows. The tails each hold `(100 - level) / 2` percent
    of the draws.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(month_sums)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=resamples)
    draws = (counts @ month_sums) / (counts @ month_rows)
    tail = (100.0 - level) / 2.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return float(low), float(high)


def decision_contrast(
    *, frame: pl.DataFrame, treatment: str, reference: str, metric: str, level: float,
    resamples: int, months: tuple[str, ...] | None = None, library: bool = True,
) -> dict:  # fmt: skip
    """Return a paired contrast with its three decision intervals and the library interval.

    The decision intervals are a paired t interval on the monthly means of the seed-averaged
    difference, a bootstrap over whole months of the seed-averaged difference, and a paired t interval
    on the per-seed mean differences. The library interval draws one seed per resample.
    """
    scope = frame if months is None else frame.filter(pl.col("month").is_in(list(months)))
    differences, month_of = paired_differences(
        losses=scope, treatment=treatment, reference=reference, metric=metric
    )
    seed_mean = differences.mean(axis=0)
    labels = sorted(set(month_of.tolist()))
    sums = np.array([seed_mean[month_of == m].sum() for m in labels])
    rows = np.array([(month_of == m).sum() for m in labels], dtype=float)
    monthly = sums / rows
    per_seed = differences.mean(axis=1)
    t_month = interval(values=monthly, level=level)
    boot = month_bootstrap(month_sums=sums, month_rows=rows, level=level, resamples=resamples)
    t_seed = interval(values=per_seed, level=level)
    result = {
        "treatment": treatment, "reference": reference, "metric": metric, "level": level,
        "estimate": float(differences.mean()),
        "t_month": t_month, "bootstrap_month": boot, "t_seed": t_seed,
        "n_months": len(labels), "n_seeds": len(per_seed),
        "monthly": monthly.tolist(), "seed_estimates": per_seed.tolist(),
    }  # fmt: skip
    if library:
        lib = contrast(
            frame=scope,
            treatment=treatment,
            reference=reference,
            metric=metric,
            level=level,
            resamples=resamples,
        )
        result["library"] = (lib["lower"], lib["upper"])
    intervals = (t_month, boot, t_seed)
    result["passes"] = bool(all(high < 0.0 for _, high in intervals))
    result["all_above_zero"] = bool(all(low > 0.0 for low, _ in intervals))
    return result


def interval_line(*, label: str, result: dict) -> str:
    """Return one contrast as a printed line: estimate, three decision intervals, library interval, seed estimates."""
    seeds = ", ".join(f"{x:+.3f}" for x in result["seed_estimates"])
    lib = result.get("library")
    library = f"; library [{lib[0]:+.3f}, {lib[1]:+.3f}]" if lib else ""
    return (
        f"{label}: {result['treatment']} minus {result['reference']} ({result['metric']}, {result['level']:.2f}% level) "
        f"estimate {result['estimate']:+.3f}; t over {result['n_months']} months [{result['t_month'][0]:+.3f}, {result['t_month'][1]:+.3f}]; "
        f"month bootstrap [{result['bootstrap_month'][0]:+.3f}, {result['bootstrap_month'][1]:+.3f}]; "
        f"t over {result['n_seeds']} seeds [{result['t_seed'][0]:+.3f}, {result['t_seed'][1]:+.3f}]{library}; "
        f"passes {result['passes']}; seeds [{seeds}]"
    )


def probe_values(*, arm: str) -> dict[tuple[str, int], float]:
    """Return the probe's held-out mean absolute error (points of capacity) of each (fold, seed) of an arm.

    Seeds 0 to 2 of W0 come from the probe-only arm W0p, which loads experiment 695's stored
    pre-trained weights.
    """
    out = {}
    for seed in SEEDS:
        folder = "W0p" if arm == "W0" and seed in REUSED_SEEDS else arm
        for month in SCORED_MONTHS:
            path = DATA_DIR / "runs" / folder / "probe" / f"{month}_s{seed}.json"
            out[(month, seed)] = float(json.loads(path.read_text())["mae_points"])
    return out


def probe_contrast() -> dict:
    """Return M1: WST's probe error minus W0's, over the 8 folds' seed means, with a 95% paired t interval."""
    treatment, reference = probe_values(arm="WST"), probe_values(arm="W0")
    per_month = np.array([
        np.mean([treatment[(m, s)] for s in SEEDS]) - np.mean([reference[(m, s)] for s in SEEDS])
        for m in SCORED_MONTHS
    ])  # fmt: skip
    low, high = interval(values=per_month, level=PROBE_LEVEL)
    return {
        "estimate": float(per_month.mean()), "lower": low, "upper": high, "monthly": per_month.tolist(),
        "w0_mean_mae": float(np.mean(list(reference.values()))), "wst_mean_mae": float(np.mean(list(treatment.values()))),
        "passes": bool(high < 0.0),
    }  # fmt: skip


def other_primaries(*, a0: pl.DataFrame, s1: pl.DataFrame, resamples: int) -> list[dict]:
    """Return S1 minus A0 at every Primary except D8, with experiment 695's "moves" rule applied.

    A Primary moves when the 95% month-block interval of the paired difference excludes zero and the
    absolute estimate exceeds that Primary's difference between experiment 677's stored sister arms.
    """
    caps = capacity_of()
    truth = pl.read_parquet(SHARED_DIR / "ground_truth_primaries.parquet").with_columns(
        time_series_id=pl.col("time_series_id").cast(pl.Int64)
    )
    sisters = s695.sister_differences()
    labels = pl.read_parquet(s695.E677_PROTOCOL_ROWS, columns=["time_series_id", "label"]).unique()
    label_of = dict(zip(labels["time_series_id"].cast(pl.Int64), labels["label"], strict=True))
    out = []
    for series_id in sorted(a0["time_series_id"].unique().to_list()):
        if label_of.get(series_id) == "D8":
            continue
        losses = {}
        t = truth.filter(pl.col("time_series_id") == series_id).select("valid_time", "power")
        for name, fits in (("A0", a0), ("S1", s1)):
            part = forecast_of(fits=fits, series_id=series_id, capacity=caps[series_id])
            losses[name] = row_scores(forecast=part, truth=t, capacity=caps[series_id])
        result = contrast(
            frame=losses_for_bootstrap(arm_losses=losses),
            treatment="S1",
            reference="A0",
            metric="crps",
            level=DESCRIPTIVE_LEVEL,
            resamples=resamples,
        )
        label = label_of.get(series_id, str(series_id))
        sister = sisters.get(label, float("nan"))
        moves = bool(
            (result["upper"] < 0.0 or result["lower"] > 0.0) and abs(result["estimate"]) > sister
        )
        out.append({
            "series": label, "estimate": result["estimate"], "lower": result["lower"], "upper": result["upper"],
            "sister_difference": sister, "moves": moves, "worse": moves and result["estimate"] > 0.0,
        })  # fmt: skip
    return out


def slice_scores(*, losses: dict[str, pl.DataFrame], names: list[str]) -> pl.DataFrame:
    """Return seed-mean scores of each D8 system on engine and fade months, cold nights, and lead bands."""
    sl = s695.slice_table(losses=losses[names[0]])
    masks: dict[str, pl.Expr] = {
        "all rows": pl.lit(value=True),
        "engine months (valid time)": pl.col("engine_months"),
        "fade months (valid time)": pl.col("fade_months"),
        "tracked on (hindsight M2 flag)": pl.col("flag").fill_null(value=False),
        "tracked off (hindsight M2 flag)": ~pl.col("flag").fill_null(value=True),
        "cold winter nights": pl.col("cold_nights").fill_null(value=False),
        **{
            f"lead {band}": (pl.col("lead_h") >= lo) & (pl.col("lead_h") < hi)
            for band, (lo, hi) in s695.LEAD_BANDS.items()
        },
    }
    rows = []
    for name in names:
        joined = losses[name].join(sl, on=list(KEYS), how="left", validate="m:1")
        n_seeds = joined["seed"].n_unique()
        for label, mask in masks.items():
            part = joined.filter(mask)
            if part.is_empty():
                continue
            row = {"system": name, "slice": label, "rows": part.height // n_seeds}
            for column in (
                "crps",
                "mae",
                *[f"pinball_{t}" for t in TAIL_NAMES],
                *[f"coverage_{t}" for t in TAIL_NAMES],
            ):
                row[column] = s695.mean_of(part[column])
            rows.append(row)
    return pl.DataFrame(rows)


def check_origin_time_weather(*, names: list[str]) -> dict:
    """Assert that every arm's inference read only origin-time weather and that its rows equal A0's.

    The fit code asserts `nwp_init_time <= power_fcst_init_time` for every row of every test origin and
    prints the count of origins in each fold's `<month>_pfit*.log` ("CHECKS PASSED"). This reads every
    log of every arm (the logs of experiment 695 for seeds 0 to 2 of A0 and W0), requires each log's
    count to equal the number of days of its month, and requires the 8 months to sum to 242. The row
    keys of every arm must equal A0's, and the native origin must be 09:00 UTC.
    """
    arms = sorted({n for n in names} | {"A0"})
    checked: dict[str, dict[str, list[int]]] = {}
    for arm in arms:
        checked[arm] = {}
        for month in SCORED_MONTHS:
            logs = sorted((DATA_DIR / "runs" / arm / "logs").glob(f"{month}_pfit_*.log"))
            if arm in {"A0", "W0"}:
                logs.append(E695_DIR / "runs" / arm / "logs" / f"{month}_pfit.log")
            assert logs, f"arm {arm}, fold {month}: no pfit log"
            counts = []
            for log in logs:
                found = re.findall(
                    r"CHECKS PASSED .*'origins_weather_checked': (\d+)", log.read_text()
                )
                assert len(found) == 1, f"{log}: no single CHECKS PASSED line"
                assert int(found[0]) == DAYS_IN_MONTH[month], (log, found)
                counts.append(int(found[0]))
            checked[arm][month] = counts
    assert all(sum(v[0] for v in months.values()) == 242 for months in checked.values())
    base = read_fits(arm="A0").select(*KEYS).unique().sort(*KEYS)
    for arm in arms:
        other = read_fits(arm=arm).select(*KEYS).unique().sort(*KEYS)
        if arm == "S1d":
            other = other.join(base, on=list(KEYS), how="inner")
        assert base.height == other.height and base.equals(other), (
            f"arm {arm}'s rows differ from A0's"
        )
    nwp = (base["power_fcst_init_time"] - NATIVE_TO_CONTRACT).dt.hour().unique().to_list()
    assert nwp == [9], nwp
    return {"origins_checked_per_arm_and_fold": checked, "rows_per_arm": base.height}


def median_minus_mean(*, forecast: pl.DataFrame, capacity: float) -> float:
    """Return the mean over rows of (mean of the 11 quantiles minus the median), in points of capacity."""
    gap = (pl.mean_horizontal(*QUANTILE_COLUMNS) - pl.col("q50")) / capacity * 100.0
    return float(forecast.select(gap.mean()).item())


def main() -> None:
    """Score every system, compute the contrasts, apply the verdict table, and print the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=FAMILY_RESAMPLES)
    args = parser.parse_args()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    import e00695_clean as clean

    host_y = clean.load_setup(arm="W0").y
    host_truth = grid_truth(y=host_y)
    host_id = host_series_id()
    caps = capacity_of()
    d8_t = d8_truth()
    all_fits = {arm: read_fits(arm=arm) for arm in ("A0", "S1", "SP", "S1d", "W0", "WST", "WS1")}
    forecasts, losses = {}, {}
    for system in SYSTEMS:
        series_id, capacity = (
            (D8_SERIES, D8_CAPACITY_MVA) if system.series == "d8" else (host_id, caps[host_id])
        )
        forecasts[system.name] = forecast_of(
            fits=all_fits[system.name], series_id=series_id, capacity=capacity
        )
        truth = d8_t if system.series == "d8" else host_truth
        losses[system.name] = row_scores(
            forecast=forecasts[system.name], truth=truth, capacity=capacity
        )
    for system in SYSTEMS:
        name = f"{system.name.lower()}-{system.series}"
        if (
            system.name not in {"S1d"} and not (DATA_DIR / "predictions" / name).exists()
        ):  # stored sets are immutable
            write_predictions(
                experiment_id=696,
                name=name,
                frame=forecasts[system.name],
                kind="quantiles",
                row_set_id=ROW_SET_ID,
            )  # ty: ignore[invalid-argument-type]
    d8 = losses_for_bootstrap(arm_losses={n: losses[n] for n in ("A0", "S1", "SP")})
    host = losses_for_bootstrap(arm_losses={n: losses[n] for n in ("W0", "WST", "WS1")})
    family = {"resamples": args.resamples, "level": FAMILY_LEVEL}
    c1 = decision_contrast(
        frame=d8,
        treatment="S1",
        reference="A0",
        metric="crps",
        level=FAMILY_LEVEL,
        resamples=args.resamples,
    )
    c2 = decision_contrast(
        frame=d8,
        treatment="S1",
        reference="SP",
        metric="crps",
        level=FAMILY_LEVEL,
        resamples=args.resamples,
    )
    c3 = decision_contrast(
        frame=host,
        treatment="WST",
        reference="W0",
        metric="crps",
        level=FAMILY_LEVEL,
        resamples=args.resamples,
    )
    c4 = decision_contrast(
        frame=host,
        treatment="WS1",
        reference="W0",
        metric="crps",
        level=FAMILY_LEVEL,
        resamples=args.resamples,
    )
    m1 = probe_contrast()
    descriptive_resamples = 4_000
    mae = decision_contrast(
        frame=d8,
        treatment="S1",
        reference="A0",
        metric="mae",
        level=DESCRIPTIVE_LEVEL,
        resamples=descriptive_resamples,
    )
    seeds3 = {
        n: losses[n].filter(pl.col("seed").is_in(list(DESCRIPTIVE_SEEDS))) for n in ("A0", "S1d")
    }
    d8_s1d = losses_for_bootstrap(arm_losses=seeds3)
    extras = {
        "SP minus A0": decision_contrast(frame=d8, treatment="SP", reference="A0", metric="crps", level=DESCRIPTIVE_LEVEL, resamples=descriptive_resamples),
        "WS1 minus WST": decision_contrast(
            frame=losses_for_bootstrap(arm_losses={n: losses[n] for n in ("WST", "WS1")}), treatment="WS1", reference="WST", metric="crps", level=DESCRIPTIVE_LEVEL, resamples=descriptive_resamples),
        "S1d minus A0 (seeds 0 to 2)": decision_contrast(frame=d8_s1d, treatment="S1d", reference="A0", metric="crps", level=DESCRIPTIVE_LEVEL, resamples=descriptive_resamples),
        "C1 in engine months": decision_contrast(frame=d8, treatment="S1", reference="A0", metric="crps", level=DESCRIPTIVE_LEVEL, resamples=descriptive_resamples, months=ENGINE_MONTHS),
        "C1 in fade months": decision_contrast(frame=d8, treatment="S1", reference="A0", metric="crps", level=DESCRIPTIVE_LEVEL, resamples=descriptive_resamples, months=FADE_MONTHS),
    }  # fmt: skip
    primaries = other_primaries(a0=all_fits["A0"], s1=all_fits["S1"], resamples=2000)
    checks = json.loads(CHECKS_PATH.read_text()) if CHECKS_PATH.exists() else {}
    inputs = VerdictInputs(
        leakage_failed=not (checks.get("leakage_passed", False) and checks.get("synthetic_passed", False)),
        gate_failed=bool(checks.get("gate_failed", True)),
        c1_passes=c1["passes"], c1_estimate=c1["estimate"], c1_all_above_zero=c1["all_above_zero"],
        c2_passes=c2["passes"], m1_passes=m1["passes"], c3_all_above_zero=c3["all_above_zero"],
        mae_rises=mae["bootstrap_month"][0] > 0.0,
        other_primary_worse=any(p["worse"] for p in primaries),
        threshold=THRESHOLD_POINTS,
    )  # fmt: skip
    branch, word, trigger_5 = verdict(v=inputs)
    d8_names = ["A0", "S1", "SP", "S1d"]
    slices = slice_scores(losses=losses, names=d8_names)
    slices.write_csv(SCORES_DIR / "d8_slices.csv")
    medians = {
        n: median_minus_mean(
            forecast=forecasts[n], capacity=D8_CAPACITY_MVA if n in d8_names else caps[host_id]
        )
        for n in forecasts
    }
    report = {
        "family": family, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "m1": m1, "mae_s1_minus_a0": mae, "extras": extras,
        "other_primaries": primaries, "inputs": inputs.__dict__, "branch": branch, "verdict": word, "audit_trigger_5": trigger_5,
        "mean_crps": {n: s695.mean_of(losses[n]["crps"]) for n in losses}, "mean_mae": {n: s695.mean_of(losses[n]["mae"]) for n in losses},
        "median_minus_mean_points": medians,
        "origin_time_assertion": check_origin_time_weather(names=[s.name for s in SYSTEMS]),
        "arms": {a: ARMS[a].kind for a in ARMS},
    }  # fmt: skip
    (SCORES_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))
    s695.climatology_reference().write_csv(SCORES_DIR / "references_d8.csv")
    print_report(report=report, slices=slices)


def print_report(*, report: dict, slices: pl.DataFrame) -> None:
    """Print every contrast, the probe, the per-Primary moves, the slices, the checks, and the verdict branch."""
    print(
        f"PLANNED CONTRASTS (CRPS in points of effective capacity; negative is a gain; {report['family']['level']:.2f}% level)"
    )
    for label in ("c1", "c2", "c3", "c4"):
        print(interval_line(label=label.upper(), result=report[label]))
    m1 = report["m1"]
    print(
        f"M1 (encoder probe, held-out MAE in points of capacity, WST minus W0, 95% t over 8 folds): estimate {m1['estimate']:+.4f} "
        f"[{m1['lower']:+.4f}, {m1['upper']:+.4f}]; W0 {m1['w0_mean_mae']:.4f}, WST {m1['wst_mean_mae']:.4f}; passes {m1['passes']}"
    )
    print(interval_line(label="MAE", result=report["mae_s1_minus_a0"]))
    print("DESCRIPTIVE CONTRASTS (95%)")
    for name, result in report["extras"].items():
        print(interval_line(label=name, result=result))
    print(f"smallest gain worth porting {THRESHOLD_POINTS:.2f} points")
    print("MEAN CRPS and MAE PER SYSTEM")
    for name, crps in report["mean_crps"].items():
        print(
            f"{name}: CRPS {crps:.3f}, MAE {report['mean_mae'][name]:.3f}, mean of quantiles minus median {report['median_minus_mean_points'][name]:+.3f}"
        )
    print("OTHER PRIMARIES, D8 excluded (S1 minus A0, 95% month-block interval)")
    for row in report["other_primaries"]:
        print(
            f"{row['series']}: {row['estimate']:+.3f} [{row['lower']:+.3f}, {row['upper']:+.3f}] sister {row['sister_difference']:.3f} moves={row['moves']} worse={row['worse']}"
        )
    print("D8 SLICES")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=60, tbl_width_chars=250):
        print(slices.with_columns(pl.col(pl.Float64).round(3)))
    print(f"ORIGIN-TIME WEATHER ASSERTION: {json.dumps(report['origin_time_assertion'])}")
    print(f"VERDICT INPUTS: {json.dumps(report['inputs'])}")
    print(
        f"VERDICT BRANCH {report['branch']}: {report['verdict']}; audit trigger 5 fires: {report['audit_trigger_5']}"
    )


if __name__ == "__main__":
    main()

"""Plan calculations for experiment 729: the split rule and the numbers behind the bars.

Read-only, one core. It reads D8's pre-engine readings (training-type data, never a test score) and
prints aggregates only. It writes the explicit month lists to the private `PRIVATE/splits.md`,
and never prints a date.
"""

import math
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[1]))
import engine_constants as ec
from studies.power import scan_power

DATA: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
EXP_DIR: Final[Path] = DATA / "00729-msa-d8-neural-untangler-synthetic-known-answers"
CAPACITY_MVA: Final[float] = 10.379
D8_ID: Final[int] = 26
BSP_ID: Final[int] = 1
MIN_DAYS_IN_MONTH: Final[int] = 20
MIN_HH_IN_DAY: Final[int] = 40
# Within each group of months, test and validation months are chosen by position in time order:
# position = remainder (mod period). The BSP group holds the months with real BSP steps.
RULE_BSP_TEST: Final[tuple[int, int]] = (3, 2)
RULE_OTHER_TEST: Final[tuple[int, int]] = (3, 2)
RULE_OTHER_VALIDATION: Final[tuple[int, int]] = (6, 1)  # the BSP group takes its last month
# The fixed lockdown window of docs/ml-experiments/data-processing.md, as first and last month.
LOCKDOWN_FIRST_MONTH: Final[str] = "2020-03"
LOCKDOWN_LAST_MONTH: Final[str] = "2021-07"
FLOOR_SIGMA: Final[float] = 0.5
QHAT_POINTS: Final[float] = 0.1 / CAPACITY_MVA * 100
RNG_SEED: Final[int] = 729


def base_days() -> pl.DataFrame:
    """Return one row per base day: date, 48 half-hours of the reading in points, BSP flag."""
    frame = pl.read_parquet(
        DATA / "00697-gas-engine-state-tracker-part-4-running-patterns/frames/frame_d8.parquet"
    ).select("start", "era")
    era_end = frame.filter(pl.col("era") == "net")["start"].min()
    cut = min(era_end, ec.engine_datetime(key="d8_engine_connected_date"))
    raw = scan_power().filter(pl.col("time_series_id").is_in([D8_ID, BSP_ID])).collect()
    raw = raw.with_columns(start=pl.col("time") - pl.duration(minutes=30))
    d8 = raw.filter(pl.col("time_series_id") == D8_ID).filter(pl.col("start") < cut)
    bsp_first = raw.filter(pl.col("time_series_id") == BSP_ID)["start"].min()
    d8 = d8.with_columns(
        date=pl.col("start").dt.date(),
        hh=pl.col("start").dt.hour() * 2 + pl.col("start").dt.minute() // 30,
    )
    days = d8.group_by("date").agg(
        n=pl.len(), hh=pl.col("hh"), r=pl.col("power"), first=pl.col("start").min()
    )
    days = days.filter(pl.col("n") >= MIN_HH_IN_DAY).sort("date")
    return days.with_columns(bsp=pl.col("first") >= bsp_first).drop("first")


def month_table(days: pl.DataFrame) -> pl.DataFrame:
    """Number the base months in time order, keeping those with enough base days."""
    months = (
        days.with_columns(month=pl.col("date").dt.strftime("%Y-%m"))
        .group_by("month")
        .agg(n_days=pl.len(), bsp_days=pl.col("bsp").sum())
        .filter(pl.col("n_days") >= MIN_DAYS_IN_MONTH)
        .sort("month")
        .with_row_index("number", offset=1)
    )
    months = months.with_columns(
        has_bsp=pl.col("bsp_days") >= MIN_DAYS_IN_MONTH,
        in_lockdown=pl.col("month").is_between(
            pl.lit(LOCKDOWN_FIRST_MONTH), pl.lit(LOCKDOWN_LAST_MONTH)
        ),
    )
    months = months.with_columns(
        position=pl.when(~pl.col("in_lockdown"))
        .then(pl.col("number").rank("ordinal").over(["has_bsp", "in_lockdown"]).cast(pl.Int64))
        .otherwise(None)
    )

    bsp = pl.col("has_bsp")
    pos = pl.col("position")
    is_test = (
        pl.when(bsp)
        .then(pos % RULE_BSP_TEST[0] == RULE_BSP_TEST[1])
        .otherwise(pos % RULE_OTHER_TEST[0] == RULE_OTHER_TEST[1])
    )
    is_validation = (
        pl.when(bsp)
        .then(pos == pos.max().over("has_bsp"))
        .otherwise(pos % RULE_OTHER_VALIDATION[0] == RULE_OTHER_VALIDATION[1])
    )

    return months.with_columns(
        split=pl.when(pl.col("in_lockdown"))
        .then(pl.lit("training"))
        .when(is_test)
        .then(pl.lit("test"))
        .when(is_validation)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("training"))
    )


def write_private_splits(months: pl.DataFrame) -> None:
    """Write the explicit month lists to the private file, never to stdout."""
    out = EXP_DIR / "PRIVATE"
    out.mkdir(exist_ok=True)
    lines = [
        "# Experiment 729 split months (private, never committed)",
        "",
        (
            f"Rule: base months with at least {MIN_DAYS_IN_MONTH} base days and no day in the "
            "fixed lockdown window (months in the window are always training), in two groups "
            "(months where the BSP meter has data, and the others), each numbered from 1 "
            "in time order. "
            "Position p is test when p mod period = remainder: "
            f"BSP group {RULE_BSP_TEST}, other group {RULE_OTHER_TEST} (period, remainder). "
            f"Validation: the last month of the BSP group, and in the other group "
            f"{RULE_OTHER_VALIDATION}. Training: every other base month."
        ),
        "",
        "| Number | Position | Month | Split | Carries BSP steps |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['number']} | {r['position']} | {r['month']} | {r['split']} | {r['has_bsp']} |"
        for r in months.iter_rows(named=True)
    )
    (out / "splits.md").write_text("\n".join(lines) + "\n")


def sin_elevation(*, doy: np.ndarray, hh: np.ndarray) -> np.ndarray:
    """Return the sine of the solar elevation at 52.5 N, with UTC hour as solar time."""
    lat = math.radians(52.5)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365)
    hour_angle = np.radians((hh + 0.5) / 2 * 15 - 180)
    return np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)


def draw_engine(*, rng: np.random.Generator, n_days: int, weekend: np.ndarray) -> np.ndarray:
    """Return engine output in points, shape (n_days, 48), from the plan's parameter table."""
    n_units = int(rng.integers(1, 5))
    sizes = rng.uniform(10, 50, n_units)
    if sizes.sum() > 120:
        sizes *= 120 / sizes.sum()
    no_run = rng.uniform(0.2, 0.7)
    p_eve, p_mor, p_mid = rng.uniform(0.3, 0.9), rng.uniform(0.1, 0.6), rng.uniform(0, 0.3)
    wk_factor = rng.uniform(0.1, 0.8)
    out = np.zeros((n_days, 48))
    for day in range(n_days):
        if rng.uniform() < no_run:
            continue
        f = wk_factor if weekend[day] else 1.0
        runs = [
            (p_eve, rng.uniform(15, 17.5), rng.uniform(2, 6)),
            (p_mor, rng.uniform(5.5, 8.5), rng.uniform(1, 8)),
            (p_mid, rng.uniform(10, 14), rng.uniform(1, 4)),
        ]
        for prob, start_h, length_h in runs:
            if rng.uniform() < prob * f:
                k = int(rng.integers(1, n_units + 1))
                level = sizes[:k].sum() * rng.uniform(0.6, 1.0)
                a, b = int(start_h * 2), int(min(48, (start_h + length_h) * 2))
                out[day, a:b] = np.maximum(out[day, a:b], level)
    for _ in range(rng.poisson(rng.uniform(0, 2) * n_days / 30)):
        start = int(rng.integers(0, n_days))
        length = int(rng.integers(2, 20))
        level = sizes.sum() * rng.uniform(0.3, 1.0)
        out[start : start + length] = np.maximum(out[start : start + length], level)
    return out


def draw_solar(*, rng: np.random.Generator, doy: np.ndarray) -> np.ndarray:
    """Return solar output in points, shape (n_days, 48): a clear-sky shape times a cloud factor."""
    n_days = len(doy)
    cap = math.exp(rng.uniform(math.log(5), math.log(150)))
    sin_e = sin_elevation(doy=doy[:, None], hh=np.arange(48)[None, :])
    clear = np.clip(1.2 * sin_e, 0, 1)
    day_k = rng.beta(2, 2, n_days) * 0.85 + 0.15
    noise = np.zeros((n_days, 48))
    eps = rng.normal(0, 0.15, (n_days, 48))
    for t in range(1, 48):
        noise[:, t] = 0.9 * noise[:, t - 1] + math.sqrt(1 - 0.81) * eps[:, t]
    return cap * clear * np.clip(day_k[:, None] + noise, 0, 1.2)


def gaussian_crps(*, err: np.ndarray, sigma: float) -> np.ndarray:
    """Return the CRPS of a Gaussian of spread sigma at an observation err away from its mean."""
    z = err / sigma
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def simulate(*, days: pl.DataFrame, months: pl.DataFrame, n_segments: int) -> dict[str, float]:
    """Draw synthetic segments over the training and test base days and return the aggregates."""
    rng = np.random.default_rng(RNG_SEED)
    base_by_month: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    keyed = days.with_columns(month=pl.col("date").dt.strftime("%Y-%m"))
    usable = set(months["month"].to_list())
    for (month,), grp in keyed.group_by("month"):
        if month not in usable:
            continue
        arr = np.full((grp.height, 48), np.nan)
        for i, (hh, r) in enumerate(zip(grp["hh"].to_list(), grp["r"].to_list(), strict=True)):
            arr[i, hh] = np.asarray(r, dtype=float) / CAPACITY_MVA * 100
        for i in range(arr.shape[0]):
            ok = ~np.isnan(arr[i])
            arr[i] = np.interp(np.arange(48), np.flatnonzero(ok), arr[i, ok])
        dates = grp["date"].to_list()
        doy = np.array([d.timetuple().tm_yday for d in dates])
        weekend = np.array([d.weekday() >= 5 for d in dates])
        base_by_month[month] = (arr, doy, weekend)
    month_keys = sorted(base_by_month)
    mix = {"none": 0.15, "engine": 0.30, "solar": 0.20, "both": 0.35}
    acc = {k: [] for k in ("X", "R", "world")}
    for _ in range(n_segments):
        month = month_keys[int(rng.integers(len(month_keys)))]
        base, doy, weekend = base_by_month[month]
        world = rng.choice(list(mix), p=list(mix.values()))
        eng = (
            draw_engine(rng=rng, n_days=len(doy), weekend=weekend)
            if world in ("engine", "both")
            else 0
        )
        sol = draw_solar(rng=rng, doy=doy) if world in ("solar", "both") else 0
        x = base - eng - sol
        q = rng.uniform(0, 0.3) / CAPACITY_MVA * 100 * rng.uniform(0.8, 1.0, x.shape)
        acc["X"].append(x.ravel())
        acc["R"].append(np.sqrt(x**2 + q**2).ravel())
        acc["world"].append(np.full(x.size, list(mix).index(world)))
    x, r, world = (np.concatenate(acc[k]) for k in ("X", "R", "world"))
    dead = np.abs(x) < 2
    rev_out = (x < -2).mean()
    xhat = np.sqrt(np.maximum(r**2 - QHAT_POINTS**2, 0))
    imp = x >= 0
    return {
        "half_hours_drawn": float(x.size),
        "reverse_share_all": float((x < 0).mean()),
        "reverse_share_outside_dead_band": float(rev_out),
        "dead_band_share": float(dead.mean()),
        "reverse_share_engine_only": float((x[world == 1] < 0).mean()),
        "reverse_share_solar_only": float((x[world == 2] < 0).mean()),
        "reverse_share_both": float((x[world == 3] < 0).mean()),
        "p0_median_abs_error_import_points": float(np.abs(xhat - x)[imp].mean()),
        "p0f_crps_import_points": float(
            gaussian_crps(err=(xhat - x)[imp], sigma=FLOOR_SIGMA).mean()
        ),
        "floor_only_lobe_coverage_all": float(
            (np.abs(np.abs(x) - xhat) <= 1.645 * FLOOR_SIGMA).mean()
        ),
        "floor_only_lobe_coverage_reverse": float(
            (np.abs(np.abs(x) - xhat) <= 1.645 * FLOOR_SIGMA)[x < 0].mean()
        ),
    }


def farm_coverage(*, days: pl.DataFrame, months: pl.DataFrame) -> None:
    """Print how many metered farms have data on each base day, and how far apart CAMS cells are."""
    frame = pl.read_parquet(
        DATA / "00039-pv-disaggregation-cams-injection/work/frame.parquet",
        columns=["time", "site", "kind", "power", "ghi_u"],
    )
    farms = frame.filter(pl.col("kind") == "farm").with_columns(date=pl.col("time").dt.date())
    per_day = (
        farms.group_by("date", "site")
        .agg(n=pl.col("power").is_not_null().sum())
        .filter(pl.col("n") >= MIN_HH_IN_DAY)
        .group_by("date")
        .agg(n_farms=pl.len())
    )
    joined = days.select("date").join(per_day, on="date", how="left").fill_null(0)
    joined = joined.with_columns(month=pl.col("date").dt.strftime("%Y-%m")).join(
        months.select("month", "split"), on="month", how="left"
    )
    print("share of base days with at least 1 farm with data:", (joined["n_farms"] >= 1).mean())
    print("share of base days with at least 3 farms with data:", (joined["n_farms"] >= 3).mean())
    print("share of base days with no farm with data, by split:")
    print(joined.group_by("split").agg(no_farm=(pl.col("n_farms") == 0).mean()).sort("split"))
    wide = frame.pivot(on="site", index="time", values="ghi_u")
    daytime = wide.filter(pl.col("D8") > 5)
    for farm in sorted(farms["site"].unique().to_list()):
        rel = ((daytime[farm] - daytime["D8"]).abs() / daytime["D8"]).median()
        same = ((daytime[farm] - daytime["D8"]).abs() < 1e-6).mean()
        print(
            f"CAMS at farm {farm} vs D8, daytime: identical share {same:.4f}, median gap {rel:.3f}"
        )


def derived_constants() -> None:
    """Print the arithmetic behind the bootstrap widening, the seed floor, and the GPU hours."""
    n_contrasts = 12
    tail = 0.05 / n_contrasts / 2
    z = stats.norm.ppf(1 - tail)
    print(f"Bonferroni level over {n_contrasts}: {100 * (1 - 2 * tail):.3f}%, z = {z:.3f}")
    for label, units in (
        ("8 date blocks (2 BSP months)", 8),
        ("16 segments (2 BSP months)", 16),
        ("33 date blocks (8 months)", 33),
    ):
        print(f"t / z widening for {label}: {stats.t.ppf(1 - tail, units - 1) / z:.3f}")
    print("P(sample sd < half the true sd), 3 seeds:", round(float(stats.chi2.cdf(0.5, 2)), 3))
    print("P(sample sd < half the true sd), 5 seeds:", round(float(stats.chi2.cdf(1.0, 4)), 3))
    n_test_rows = 8 * 30 * 8 * 48
    print("synthetic test half-hours (8 months x 30 days x 8 draws x 48):", n_test_rows)
    print(
        "reliability-term sampling noise bound, 10 bins, 64 segments x 30 days independent:",
        round(10 * 0.25 / (64 * 30), 5),
    )
    print("reliability bar 0.01 is an RMS probability gap of", math.sqrt(0.01))
    fits = 18 + 9 * 3 + 2
    for sec in (3, 10):
        print(f"{fits} fits, 200 epochs, {sec} s/epoch: {fits * 200 * sec / 3600:.2f} hours")
    print(f"{fits} fits, 70 epochs, 3 s/epoch: {fits * 70 * 3 / 3600:.2f}")
    print(
        "6 LSTM fits at 10 s/epoch, rest at 3 s/epoch, 200 epochs:",
        round(((fits - 6) * 3 + 6 * 10) * 200 / 3600, 2),
    )
    print(
        "K1 F1 power: running half-hours 200, mean spell 8 half-hours -> spells",
        200 / 8,
        "sd of a proportion",
        round(math.sqrt(0.25 / (200 / 8)), 3),
    )
    print(
        "harm limit from half of the margin floor:",
        0.125,
        "over import share ->",
        round(0.125 / (1 - 0.0507), 4),
    )


def main() -> None:
    """Print the aggregates that set the pre-registered bars."""
    days = base_days()
    months = month_table(days)
    write_private_splits(months)
    counts = months.group_by("split").agg(n=pl.len(), n_bsp=pl.col("has_bsp").sum()).sort("split")
    print("split counts (months, of which carrying BSP steps):")
    print(counts)
    sigma = FLOOR_SIGMA
    print(
        "Gaussian CRPS at zero error, sigma 0.5 points:",
        sigma * (2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)),
    )
    for key, val in simulate(days=days, months=months, n_segments=3000).items():
        print(f"{key}: {val:.4f}")
    farm_coverage(days=days, months=months)
    derived_constants()


if __name__ == "__main__":
    main()

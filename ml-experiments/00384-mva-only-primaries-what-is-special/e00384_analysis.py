"""Relate per-Primary facts and regularity measures to the stack-minus-BG2 gap, by metering group.

Extends experiment 380's measures M1 to M5 (read from its stored table, not recomputed) with
residual autocorrelations at 1 hour and 1 day, step sizes, coefficient of variation, and the
correlation of the residual with the Market Index Data price. Readings before 2025-07-01 only, with
the hand-filtered events excluded. The analysis is descriptive and writes only small tables.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00384-*/e00384_analysis.py
"""

import csv
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import numpy as np
import openpyxl
import polars as pl
from scipy.stats import spearmanr
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00384-mva-only-primaries-what-is-special"
OWN: Final = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
E380: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00380-gam-stack-ablation-and-jaggedness-at-16-primaries" / "work"
)
E378_REPORT: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00378-primary-demand-default-3-at-all-16-primaries-under-demand-v1"
    / "work"
    / "score_report.txt"
)
PRICE: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00010-elexon-price-lags-stage1a-demand"
    / "prices"
    / "market_index.parquet"
)
HIERARCHY: Final = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15"
CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
N_BINS: Final[int] = 20
N_PERMUTATIONS: Final[int] = 10_000
N_BOOTSTRAP: Final[int] = 10_000
SEED: Final[int] = 384
SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
SIGNED_MW: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
NEW_MEASURES: Final[tuple[str, ...]] = (
    "acf_1h", "acf_1d", "step_median", "step_p99", "step_rate_2", "step_rate_10", "cv",
    "price_corr", "price_change_corr", "price_corr_placebo",
)  # fmt: skip
OLD_MEASURES: Final[tuple[str, ...]] = ("m1_profile_share", "m2_temperature_share",
                                        "m3_resid_acf1", "m4_step_rate")  # fmt: skip
FACTS: Final[tuple[str, ...]] = (
    "first_reading_year", "capacity_mw", "peak_mva", "n_neighbours", "n_generators_at_bsp",
    "n_pv_at_bsp", "n_wind_at_bsp",
)  # fmt: skip


def residuals(
    *, y: np.ndarray, temperature: np.ndarray, times: np.ndarray, usable: np.ndarray
) -> np.ndarray:
    """Return the residual after a half-hour-of-week profile and 20 temperature bins (NaN gaps)."""
    slot = (
        ((times.astype("datetime64[D]").astype(np.int64) + 3) % 7) * 48
        + (times.astype("datetime64[m]").astype(np.int64) % 1440) // 30
    )  # fmt: skip
    yy, ss, tt = y[usable], slot[usable], temperature[usable]
    profile = np.bincount(ss, weights=yy, minlength=336) / np.maximum(
        np.bincount(ss, minlength=336), 1
    )
    res1 = yy - profile[ss]
    edges = np.quantile(tt, np.linspace(0, 1, N_BINS + 1)[1:-1])
    bins = np.searchsorted(edges, tt)
    by_bin = np.bincount(bins, weights=res1, minlength=N_BINS) / np.maximum(
        np.bincount(bins, minlength=N_BINS), 1
    )
    full = np.full(len(y), np.nan)
    full[usable] = res1 - by_bin[bins]
    return full


def lag_corr(*, x: np.ndarray, lag: int) -> float:
    """Return the correlation of `x` with itself `lag` steps earlier, over finite pairs."""
    a, b = x[lag:], x[:-lag]
    pair = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[pair], b[pair])[0, 1])


def pair_corr(*, x: np.ndarray, z: np.ndarray) -> float:
    """Return the Pearson correlation of `x` and `z` over finite pairs."""
    pair = np.isfinite(x) & np.isfinite(z)
    return float(np.corrcoef(x[pair], z[pair])[0, 1])


def new_measures(
    *, y: np.ndarray, temperature: np.ndarray, times: np.ndarray, usable: np.ndarray,
    price: np.ndarray, price_shuffled: np.ndarray,
) -> dict[str, float]:  # fmt: skip
    """Return the measures added by this experiment for one series."""
    res = residuals(y=y, temperature=temperature, times=times, usable=usable)
    step = np.abs(np.diff(y))
    both = usable[1:] & usable[:-1]
    steps = step[both]
    d_price = np.diff(price)
    d_y = np.diff(np.where(usable, y, np.nan))
    return {
        "acf_1h": lag_corr(x=res, lag=2),
        "acf_1d": lag_corr(x=res, lag=48),
        "step_median": float(np.median(steps)),
        "step_p99": float(np.quantile(steps, 0.99)),
        "step_rate_2": float((steps > 2.0).mean()),
        "step_rate_10": float((steps > 10.0).mean()),
        "cv": float(y[usable].std() / abs(y[usable].mean())),
        "price_corr": pair_corr(x=res, z=price),
        "price_change_corr": pair_corr(x=d_y, z=d_price),
        "price_corr_placebo": pair_corr(x=res, z=price_shuffled),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    """Return the rows of a CSV file as dictionaries."""
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def facts() -> pl.DataFrame:
    """Return per-Primary facts from the hierarchy files (counts and types only)."""
    series = adapter.series_table()
    monitor = read_csv(HIERARCHY / "Monitor Direction.csv")
    by_id = {int(r["TimeSeriesID"]): r for r in monitor if r["TimeSeriesIDVersion"] == "1"}
    bsp_of = {
        r["substation_number"]: r["bsp_substation_number"]
        for r in read_csv(HIERARCHY / "Substations.csv")
    }
    pairs = [
        (r["PrimarySubstation1"], r["PrimarySubstation2"])
        for r in read_csv(HIERARCHY / "Primary Substation Interconnections.csv")
    ]
    kinds = {r["SubstationNumber"]: r for r in monitor if r["SubstationType"] in
             ("EHV Customer", "HV Customer")}  # fmt: skip
    sheet = openpyxl.load_workbook(HIERARCHY / "Peak Loads.xlsx", read_only=True).active
    peak = {int(r[0]): float(r[4]) for i, r in enumerate(sheet.iter_rows(values_only=True))
            if i and len(r) > 4 and r[0] is not None and r[4] is not None}  # fmt: skip
    rows = []
    for tsid, label, capacity in series.select(
        "time_series_id", "label", "capacity_mw"
    ).iter_rows():
        number = by_id[tsid]["SubstationNumber"]
        bsp = bsp_of[number]
        generators = [k for k, _ in kinds.items() if bsp_of.get(k) == bsp]
        types = [kinds[k]["TimeSeriesType"] for k in generators]
        rows.append({
            "label": label, "capacity_mw": capacity, "peak_mva": peak.get(tsid),
            "units": by_id[tsid]["Units"], "bsp": bsp,
            "n_neighbours": sum(number in p for p in pairs),
            "n_generators_at_bsp": len(generators),
            "n_pv_at_bsp": types.count("PV"), "n_wind_at_bsp": types.count("Wind"),
        })  # fmt: skip
    return pl.DataFrame(rows)


def gaps() -> pl.DataFrame:
    """Return the experiment 377 stack gap and the experiment 378 default 3 gap, minus BG2."""
    stack = pl.read_parquet(E380 / "gap_stack_minus_bg2.parquet").select(
        "label", "group", gap_377=pl.col("gap")
    )
    text = E378_REPORT.read_text().split("slice lead_day_1")[0]
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("┆")] if "┆" in line else []
        match = re.match(r"│\s*(D\d+)", line)
        if match and len(cells) >= 9:
            cells = [c.strip("│ ") for c in cells]
            rows[cells[0]] = float(cells[3]) - float(cells[6])
    return stack.join(
        pl.DataFrame({"label": list(rows), "gap_378": list(rows.values())}), on="label"
    )


def correlate(*, table: pl.DataFrame, measure: str, gap: str, rng: np.random.Generator) -> dict:
    """Return Spearman rho, permutation p, and leave-one-out range of a measure against a gap."""
    t = table.drop_nulls([measure, gap])
    x, g = t[measure].to_numpy(), t[gap].to_numpy()
    rho = float(spearmanr(x, g).statistic)
    perm = np.array([spearmanr(rng.permutation(x), g).statistic for _ in range(N_PERMUTATIONS)])
    p = (float((np.abs(perm) >= abs(rho)).sum()) + 1) / (N_PERMUTATIONS + 1)
    loo = [float(spearmanr(np.delete(x, k), np.delete(g, k)).statistic) for k in range(len(x))]
    return {"measure": measure, "gap": gap, "n": len(x), "rho": rho, "p": p,
            "loo_min": min(loo), "loo_max": max(loo)}  # fmt: skip


def group_difference(*, table: pl.DataFrame, measure: str, rng: np.random.Generator) -> dict:
    """Return MVA-only minus signed-MW median difference with a 95% bootstrap over Primaries."""
    a = table.filter(pl.col("label").is_in(MVA_ONLY))[measure].drop_nulls().to_numpy()
    b = table.filter(pl.col("label").is_in(SIGNED_MW))[measure].drop_nulls().to_numpy()
    obs = float(np.median(a) - np.median(b))
    boot = [float(np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))))
            for _ in range(N_BOOTSTRAP)]  # fmt: skip
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"measure": measure, "mva_median": float(np.median(a)),
            "signed_median": float(np.median(b)), "difference": obs,
            "lower": float(lo), "upper": float(hi)}  # fmt: skip


def main() -> None:
    """Compute the measures, then print the group comparison and the correlations."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    OWN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=base.CACHE_DIR)
    times = cache.times_series().to_numpy()
    before = times < np.datetime64(CUT.replace(tzinfo=None))
    price_frame = (
        pl.read_parquet(PRICE)
        .filter(pl.col("dataProvider") == "APXMIDP", pl.col("volume") > 0)
        .select(pl.col("time").dt.replace_time_zone(None).alias("start"), "price")
        .unique(subset="start", keep="first", maintain_order=True)
        .sort("start")
    )
    start = (times - np.timedelta64(30, "m")).astype("datetime64[us]")
    lookup = pl.DataFrame({"start": start}).join(
        price_frame, on="start", how="left", maintain_order="left"
    )
    price = lookup["price"].to_numpy().astype(np.float64)
    price_shuffled = rng.permutation(price)
    rows = []
    for label in series["label"]:
        site = cache.sites[label]
        usable = (
            np.asarray(site.present)[before] & ~np.asarray(site.switching_mask)[before]
            & np.isfinite(site.y[before]) & np.isfinite(site.temperature_c[before])
        )  # fmt: skip
        measures = new_measures(
            y=site.y[before].astype(np.float64),
            temperature=site.temperature_c[before].astype(np.float64),
            times=times[before],
            usable=usable,
            price=price[before],
            price_shuffled=price_shuffled[before],
        )
        first = times[np.flatnonzero(np.asarray(site.present))[0]]
        first_year = 1970 + float(first.astype("datetime64[D]").astype(np.int64)) / 365.25
        rows.append(
            {"label": label, **measures, "first_reading_year": first_year,
             "n_usable": int(usable.sum())}
        )  # fmt: skip
    table = (
        pl.DataFrame(rows)
        .join(
            pl.read_parquet(E380 / "jaggedness_table.parquet").select("label", *OLD_MEASURES),
            on="label",
        )
        .join(facts(), on="label")
        .join(gaps().drop("group"), on="label")
        .sort("label")
    )
    table.write_parquet(OWN / "measures_table.parquet")
    print("== Per-Primary table ==")
    print(table)

    measures = [*FACTS, *OLD_MEASURES, *NEW_MEASURES]
    print("\n== MVA-only minus signed-MW median difference, 95% bootstrap over Primaries ==")
    print(pl.DataFrame([group_difference(table=table, measure=m, rng=rng) for m in measures
                        if m != "peak_mva" or table["peak_mva"].null_count() == 0]))  # fmt: skip
    for gap in ("gap_377", "gap_378"):
        for name, drop in (("n = 16", []), ("n = 14, D15 and D18 dropped", SHORT_HISTORY),
                           ("MVA-only eight", [*SIGNED_MW, *SHORT_HISTORY]),
                           ("MVA-only seven, D8 dropped", [*SIGNED_MW, *SHORT_HISTORY, "D8"]),
                           ("signed-MW six", [*MVA_ONLY, *SHORT_HISTORY])):  # fmt: skip
            sub = table.filter(~pl.col("label").is_in(list(drop)))
            print(f"\n== Spearman with {gap}, {name} ==")
            print(pl.DataFrame([correlate(table=sub, measure=m, gap=gap, rng=rng)
                                for m in measures]))  # fmt: skip


if __name__ == "__main__":
    main()

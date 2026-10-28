"""Relate each Primary's regularity, measured on training data, to the stack-minus-BG2 gap.

The measures use readings before 2025-07-01 only (the start of the scored year), excluding the
hand-filtered events. They are registered in the plan of experiment 380.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
    uv run --no-sync python ml-experiments/00380-*/e00380_jaggedness.py
"""

from datetime import UTC, datetime
from typing import Any, Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OWN: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00380-gam-stack-ablation-and-jaggedness-at-16-primaries" / "work"
)
CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
STEP_POINTS: Final[float] = 5.0
"""A step is a half-hour change above 5% of effective capacity (readings are in points)."""
N_TEMPERATURE_BINS: Final[int] = 20
N_PERMUTATIONS: Final[int] = 10_000
SEED: Final[int] = 380
MEASURES: Final[tuple[str, ...]] = ("m1_profile_share", "m2_temperature_share", "m3_resid_acf1",
                                    "m4_step_rate", "m5_reverse_flow_share")  # fmt: skip
EXPECTED_SIGN: Final[dict[str, int]] = {"m1_profile_share": -1, "m2_temperature_share": -1,
                                        "m3_resid_acf1": 1, "m4_step_rate": 1,
                                        "m5_reverse_flow_share": 1}  # fmt: skip


def measures_of(
    *,
    y: np.ndarray,
    temperature: np.ndarray,
    times: np.ndarray,
    present: np.ndarray,
    allowed: np.ndarray,
    signed: bool,
) -> dict[str, float]:
    """Return M1 to M5 for one series from its readings `y` (points) before the cut."""
    usable = present & allowed & np.isfinite(y) & np.isfinite(temperature)
    slot = (
        ((times.astype("datetime64[D]").astype(np.int64) + 3) % 7) * 48
        + (times.astype("datetime64[m]").astype(np.int64) % 1440) // 30
    )  # fmt: skip
    yy, ss, tt = y[usable], slot[usable], temperature[usable]
    total = float(yy.var())
    profile = np.bincount(ss, weights=yy, minlength=336) / np.maximum(
        np.bincount(ss, minlength=336), 1
    )
    res1 = yy - profile[ss]
    m1 = 1.0 - float(res1.var()) / total
    edges = np.quantile(tt, np.linspace(0, 1, N_TEMPERATURE_BINS + 1)[1:-1])
    bins = np.searchsorted(edges, tt)
    by_bin = np.bincount(bins, weights=res1, minlength=N_TEMPERATURE_BINS) / np.maximum(
        np.bincount(bins, minlength=N_TEMPERATURE_BINS), 1
    )
    res2 = res1 - by_bin[bins]
    m2 = 1.0 - float(res2.var()) / float(res1.var())
    full = np.where(usable, 0.0, np.nan)
    full[usable] = res2
    pair = np.isfinite(full[1:]) & np.isfinite(full[:-1])
    m3 = float(np.corrcoef(full[1:][pair], full[:-1][pair])[0, 1])
    both = usable[1:] & usable[:-1]
    m4 = float((np.abs(np.diff(y))[both] > STEP_POINTS).mean())
    m5 = float((yy < 0).mean()) if signed else float("nan")
    return {"m1_profile_share": m1, "m2_temperature_share": m2, "m3_resid_acf1": m3,
            "m4_step_rate": m4, "m5_reverse_flow_share": m5}  # fmt: skip


def main() -> None:
    """Print the measures, the correlations, and the placebos."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(220)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=base.CACHE_DIR)
    times = cache.times_series().to_numpy()
    before = times < np.datetime64(CUT.replace(tzinfo=None))
    rng = np.random.default_rng(SEED)
    rows, placebo_rows = [], []
    for _, label, _ in series.iter_rows():
        site = cache.sites[label]
        signed = label in ("D5", "D9", "D10", "D11", "D14", "D16")
        args: dict[str, Any] = {
            "y": site.y[before].astype(np.float64),
            "temperature": site.temperature_c[before].astype(np.float64),
            "times": times[before], "present": np.asarray(site.present)[before],
            "allowed": ~np.asarray(site.switching_mask)[before], "signed": signed,
        }  # fmt: skip
        rows.append({"label": label, **measures_of(**args)})
        shuffled = {**args, "y": rng.permutation(args["y"])}
        placebo_rows.append({"label": label, **measures_of(**shuffled)})  # ty: ignore[invalid-argument-type]
    measure_table = pl.DataFrame(rows)
    print("== Measures from readings before 2025-07-01 ==")
    print(measure_table.sort("label"))
    print("\n== Time-shuffled readings (placebo): M3 near zero, M1 and M2 small, M4 large ==")
    print(pl.DataFrame(placebo_rows).sort("label"))
    gap = pl.read_parquet(OWN / "gap_stack_minus_bg2.parquet")
    joined = measure_table.join(gap, on="label").sort("label")
    print("\n== Gap (stack minus BG2, days 1 to 10; positive = stack worse) with measures ==")
    print(joined)
    joined.write_parquet(OWN / "jaggedness_table.parquet")
    for subset_name, drop in (("n = 16", []), ("n = 14 (D15, D18 dropped)", ["D15", "D18"]),
                              ("n = 13 (D8 also dropped)", ["D15", "D18", "D8"])):  # fmt: skip
        sub = joined.filter(~pl.col("label").is_in(drop))
        print(f"\n== Spearman correlation of each measure with the gap, {subset_name} ==")
        for measure in MEASURES:
            data = sub.drop_nulls(measure) if measure == "m5_reverse_flow_share" else sub
            if measure == "m5_reverse_flow_share":
                data = data.filter(pl.col("label").is_in(["D5", "D9", "D10", "D11", "D14", "D16"]))
            x, g = data[measure].to_numpy(), data["gap"].to_numpy()
            observed = float(spearmanr(x, g).statistic)
            permuted = np.array(
                [spearmanr(rng.permutation(x), g).statistic for _ in range(N_PERMUTATIONS)]
            )
            p_value = (float((np.abs(permuted) >= abs(observed)).sum()) + 1) / (N_PERMUTATIONS + 1)
            loo = []
            for k in range(len(x)):
                keep = np.arange(len(x)) != k
                loo.append(float(spearmanr(x[keep], g[keep]).statistic))
            print(
                f"{measure:24s} n={len(x):2d} rho={observed:+.2f} (registered sign "
                f"{EXPECTED_SIGN[measure]:+d}) permutation p={p_value:.3f} "
                f"leave-one-out rho {min(loo):+.2f} to {max(loo):+.2f}"
            )


if __name__ == "__main__":
    main()

"""Experiment 369: relate each Primary's step rate to its stack-minus-BG2 gap.

Run from the repository root, with the folders of experiments 252 and 361 on `PYTHONPATH`, as

    DATA_PATH_INTERNAL=... PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:\
    ml-experiments/00361-d8-step-catalogue-detector-injection-and-gam-stack \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00369-*/e00369_run.py

The detector is experiment 361's, unchanged (1.5 MW, 6 hours). The stack is experiment 252 arm A0
at the eight MVA Primaries and experiment 155 arm C at the six MW Primaries.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Final

import e00252_common as common
import e00252_score as score252
import e00361_detect as detect
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00369-d8-cross-site-step-rate-against-stack-gap"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
MVA_SITES: Final[tuple[str, ...]] = common.SCORED_SITES
MW_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
SITES: Final[tuple[str, ...]] = MVA_SITES + MW_SITES
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 2)  # noqa: DTZ001  (the grid is naive UTC)
SCORED_LAST: Final[datetime] = datetime(2026, 7, 1)  # noqa: DTZ001
MONTHS_SCORED: Final[float] = (SCORED_LAST - SCORED_FIRST).days / 30.4375
N_SHUFFLES: Final[int] = 20
N_BOOT: Final[int] = 20_000
E252_SCORES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00252-gam-stack-at-eight-more-primaries"
    / "work"
    / "mva"
    / "scores_a0.json"
)
E155_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)


def step_table(*, times: np.ndarray, power: np.ndarray) -> pl.DataFrame:
    """Return every rise and fall the detector finds, with start, size and sign."""
    parts = []
    for negative in (True, False):
        seg = detect.detect(power_mw=power, threshold_mw=detect.THRESHOLD_MW, negative=negative)
        parts.append(
            pl.DataFrame(
                {
                    "start": times[seg.start],
                    "size_mw": np.abs(seg.level_mw),
                    "direction": ["fall" if negative else "rise"] * len(seg),
                    "dwell_days": (seg.end - seg.start) / detect.HALF_HOURS_PER_DAY,
                },
                schema_overrides={"start": pl.Datetime("us"), "direction": pl.String},
            )
        )
    return pl.concat(parts)


def shuffled_days(*, power: np.ndarray, seed: int) -> np.ndarray:
    """Return the series with its whole days permuted, which keeps each day's shape."""
    n_days = len(power) // detect.HALF_HOURS_PER_DAY
    body = power[: n_days * detect.HALF_HOURS_PER_DAY].reshape(n_days, detect.HALF_HOURS_PER_DAY)
    order = np.random.default_rng(seed).permutation(n_days)
    return body[order].reshape(-1)


def shuffled_within_month_and_day_type(
    *, times: np.ndarray, power: np.ndarray, seed: int
) -> np.ndarray:
    """Post hoc placebo: permute days only among days of the same month and weekday/weekend type.

    Moving a day only within its own month and day type keeps the seasonal and weekly shape the
    baseline expects, so only a persistent step (a run of days at a new level) is destroyed.
    """
    n_days = len(power) // detect.HALF_HOURS_PER_DAY
    body = power[: n_days * detect.HALF_HOURS_PER_DAY].reshape(n_days, detect.HALF_HOURS_PER_DAY)
    day = times[:: detect.HALF_HOURS_PER_DAY][:n_days].astype("datetime64[D]")
    key = day.astype("datetime64[M]").astype(np.int64) * 2 + (
        (day.astype(np.int64) + 3) % 7 >= 5
    ).astype(np.int64)
    rng = np.random.default_rng(seed)
    order = np.arange(n_days)
    for k in np.unique(key):
        members = np.flatnonzero(key == k)
        order[members] = rng.permutation(members)
    return body[order].reshape(-1)


def stack_gap_mw_sites() -> dict[str, float]:
    """Return arm C minus BG2 (quantile score, points) at the six MW Primaries."""
    gaps = {}
    for site in MW_SITES:
        frame = pl.read_parquet(E155_PREDS / f"c_scored_{site}.parquet").filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= score252.HORIZON_HOURS)
        )
        bg2 = score252.reference_frame(name="bg2", site=site)
        frame = frame.join(bg2, on=score252.KEY, how="inner", validate="1:1")
        actual = frame["actual"].to_numpy().astype(np.float64)
        s_stack = score252.score(
            actual=actual,
            quantiles=frame.select(list(score252.QUANTILES)).to_numpy().astype(np.float64),
        )
        s_bg2 = score252.score(
            actual=actual,
            quantiles=frame.select([f"bg2_{q}" for q in score252.QUANTILES])
            .to_numpy()
            .astype(np.float64),
        )
        gaps[site] = float(np.mean(s_stack - s_bg2))
        print(f"{site}: stack {np.mean(s_stack):.3f} BG2 {np.mean(s_bg2):.3f} rows {frame.height}")
    return gaps


def _rho(table: pl.DataFrame, *, keep) -> float:  # noqa: ANN001
    """Return the Spearman correlation of step rate and gap over the sites `keep` accepts."""
    part = table.filter(pl.col("site").map_elements(keep, return_dtype=pl.Boolean))
    return float(spearmanr(part["rate_per_month"], part["gap_points"]).statistic)


def main() -> None:
    """Build the step rates and gaps, correlate them, and write the tables."""
    WORK.mkdir(parents=True, exist_ok=True)
    e252 = json.loads(E252_SCORES.read_text())["absolute"]
    gap = {s: e252[s]["a0"] - e252[s]["bg2"] for s in MVA_SITES} | stack_gap_mw_sites()
    print(f"CONTROL GATE: D8 gap {gap['D8']:+.3f} (stored +3.619)")
    rows, steps_all = [], []
    for site in SITES:
        times, power = detect.load_power_mw(site=site)
        steps = step_table(times=times, power=power)
        scored = steps.filter((pl.col("start") >= SCORED_FIRST) & (pl.col("start") < SCORED_LAST))
        n_hist_months = (times[-1] - times[0]) / np.timedelta64(1, "D") / 30.4375
        placebo = [
            step_table(times=times[: len(p)], power=p).height / n_hist_months
            for p in (shuffled_days(power=power, seed=s) for s in range(N_SHUFFLES))
        ]
        placebo2 = [
            step_table(times=times[: len(p)], power=p).height / n_hist_months
            for p in (
                shuffled_within_month_and_day_type(times=times, power=power, seed=s)
                for s in range(N_SHUFFLES)
            )
        ]
        real_hist = steps.height / n_hist_months
        rows.append(
            {
                "site": site,
                "gap_points": gap[site],
                "steps_scored_year": scored.height,
                "rate_per_month": scored.height / MONTHS_SCORED,
                "rises": scored.filter(pl.col("direction") == "rise").height,
                "falls": scored.filter(pl.col("direction") == "fall").height,
                "median_size_mw": float(scored["size_mw"].median()) if scored.height else None,
                "p90_size_mw": float(scored["size_mw"].quantile(0.9)) if scored.height else None,
                "history_rate_per_month": real_hist,
                "placebo_rate_per_month": float(np.mean(placebo)),
                "placebo_to_real": float(np.mean(placebo)) / real_hist if real_hist else None,
                "placebo2_rate_per_month": float(np.mean(placebo2)),
                "placebo2_to_real": float(np.mean(placebo2)) / real_hist if real_hist else None,
            }
        )
        steps_all.append(steps.with_columns(site=pl.lit(site)))
    table = pl.DataFrame(rows).sort("rate_per_month", descending=True)
    table = table.with_columns(rank_by_rate=pl.int_range(1, pl.len() + 1))
    table.write_parquet(WORK / "site_table.parquet")
    pl.concat(steps_all).write_parquet(WORK / "all_steps.parquet")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=200):
        print(table)
    rate = table["rate_per_month"].to_numpy()
    gaps = table["gap_points"].to_numpy()
    rho = float(spearmanr(rate, gaps).statistic)
    rng = np.random.default_rng(369)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(rate), len(rate))
        if len(set(rate[idx])) > 1 and len(set(gaps[idx])) > 1:
            boot.append(float(spearmanr(rate[idx], gaps[idx]).statistic))
    low, high = np.percentile(boot, [2.5, 97.5])
    ranks = dict(zip(table["site"], table["rank_by_rate"], strict=True))
    result = {
        "spearman": rho,
        "ci95": [float(low), float(high)],
        "n_sites": len(rate),
        "rank_by_rate": {s: int(ranks[s]) for s in ("D8", "D12", "D22")},
        "p3_max_placebo_to_real": float(table["placebo_to_real"].max()),  # ty: ignore[invalid-argument-type]
        "p3_post_hoc_max_placebo2_to_real": float(table["placebo2_to_real"].max()),  # ty: ignore[invalid-argument-type]
        "p3_post_hoc_per_site": dict(zip(table["site"], table["placebo2_to_real"], strict=True)),
        "p3_per_site": dict(zip(table["site"], table["placebo_to_real"], strict=True)),
        "excluding_d8_spearman": _rho(table, keep=lambda s: s != "D8"),
        "mva_only_spearman": _rho(table, keep=lambda s: s in MVA_SITES),
    }
    print(json.dumps(result, indent=1, default=float))
    (WORK / "report.json").write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()

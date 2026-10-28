"""Score experiment 371's arms at D8, D2, and D4 against experiment 360's stored frames.

Reads each arm's refit files, picks the refit that serves each run under the monthly regime,
applies the stored recipe's spread factor 1.15 (experiment 87's `stack.stack`, no correction), and
joins the result to experiment 360's stored frame for the site, which holds the actual power and
the stored monthly XGBoost member (`only_M__q*`). D8 has the 96,986 rows of experiment 354; D2 and
D4 have all stored rows to 168 hours with a finite actual. Prints every number the write-up quotes
and writes `work/results.json` and `work/printed_report.txt`.

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
        uv run --no-sync python ml-experiments/00371-*/e00371_score.py
"""

import json
from pathlib import Path
from typing import Final

import e00087_common as common
import e00087_regimes as regimes
import e00087_stack as stack
import numpy as np
import polars as pl
from e00371_fit import ARM_DIR, ARMS, SITES, WORK_DIR

FRAME_DIR: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/"
    "00360-bg2-member-drop-one-at-d8/work"
)
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 0
BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("24-72 h", 24.0, 72.0),
    ("72-120 h", 72.0, 120.0),
    ("120-168 h", 120.0, 168.0),
)
GATE_TOLERANCE: Final[float] = 0.05
STORED_CONTROL: Final[float] = 7.694
STACK_WEEKLY: Final[float] = 8.338


def arm_quantiles(*, arm: str, site: str) -> pl.DataFrame:
    """Return an arm's spread-widened quantiles at a site for the runs of the monthly regime."""
    pieces = []
    for refit, chosen in regimes.served(name="monthly", origins=common.all_origins()).items():
        path = ARM_DIR / arm / f"{site}_s0_{refit:%Y%m%dT%H%M}.parquet"
        pieces.append(pl.read_parquet(path).filter(pl.col("power_fcst_init_time").is_in(chosen)))
    raw = pl.concat(pieces)
    return stack.stack(frame=raw, correct=False, spread=True).select(
        "power_fcst_init_time", "valid_time", *common.LEVEL_COLUMNS
    )


def row_scores(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the quantile score per row in points of capacity."""
    error = actual[:, None] - quantiles
    return 100.0 * 2.0 * np.mean(np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=1)


def month_bootstrap(*, values: np.ndarray, months: np.ndarray) -> tuple[float, float, float]:
    """Return the mean and its 95% interval, resampling whole valid months."""
    labels, index = np.unique(months, return_inverse=True)
    sums = np.bincount(index, weights=values, minlength=labels.size)
    counts = np.bincount(index, minlength=labels.size).astype(np.float64)
    draws = np.random.default_rng(SEED).integers(0, labels.size, size=(N_RESAMPLES, labels.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def summarise(*, values: np.ndarray, months: np.ndarray) -> dict[str, float]:
    """Return a mean with its month-resampled 95% interval."""
    mean, low, high = month_bootstrap(values=values, months=months)
    return {"mean": mean, "low": low, "high": high}


def score_site(*, site: str) -> dict:
    """Score every arm at one site, overall and by lead band, with paired differences."""
    base = pl.read_parquet(FRAME_DIR / f"frame_{site}.parquet").select(
        "power_fcst_init_time",
        "valid_time",
        "actual",
        "lead_hours",
        *[f"only_M__{c}" for c in common.LEVEL_COLUMNS],
    )
    actual = base["actual"].to_numpy().astype(np.float64)
    lead = base["lead_hours"].to_numpy()
    months = (
        base["valid_time"].dt.convert_time_zone("Europe/London").dt.strftime("%Y-%m").to_numpy()
    )
    stored = base.select(f"only_M__{c}" for c in common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    arms = {"stored": stored}
    for name in ARMS:
        joined = base.select("power_fcst_init_time", "valid_time").join(
            arm_quantiles(arm=name, site=site),
            on=["power_fcst_init_time", "valid_time"],
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        if joined["q50"].null_count():
            msg = f"{name} at {site} lacks {joined['q50'].null_count()} rows"
            raise ValueError(msg)
        arms[name] = joined.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    scores = {n: row_scores(quantiles=q, actual=actual) for n, q in arms.items()}
    out: dict = {"rows": base.height, "arms": {}}
    for name, q in arms.items():
        median = q[:, 5]
        entry: dict = {
            "score": summarise(values=scores[name], months=months),
            "mae_median_points": float(np.abs(actual - median).mean() * 100),
            "bias_median_points": float((median - actual).mean() * 100),
            "corr_median": float(np.corrcoef(median, actual)[0, 1]),
            "coverage_q05_q95": float(((actual >= q[:, 1]) & (actual <= q[:, 9])).mean()),
            "bands": {},
        }
        for label, low, high in BANDS:
            mask = (lead > low) & (lead <= high) if low else (lead <= high)
            entry["bands"][label] = {
                "score": summarise(values=scores[name][mask], months=months[mask]),
                "mae_median_points": float(np.abs(actual[mask] - median[mask]).mean() * 100),
                "bias_median_points": float((median[mask] - actual[mask]).mean() * 100),
                "corr_median": float(np.corrcoef(median[mask], actual[mask])[0, 1]),
            }
            if name != "control":
                diff = scores[name][mask] - scores["control"][mask]
                entry["bands"][label]["minus_control"] = summarise(values=diff, months=months[mask])
        if name != "control":
            entry["minus_control"] = summarise(
                values=scores[name] - scores["control"], months=months
            )
        out["arms"][name] = entry
    return out


def main() -> None:
    """Score the three sites, print the gate and tables, and write the results."""
    results = {site: score_site(site=site) for site in SITES}
    lines: list[str] = []
    d8 = results["D8"]["arms"]
    gap = abs(d8["control"]["score"]["mean"] - STORED_CONTROL)
    lines.append(
        f"GATE refitted control {d8['control']['score']['mean']:.3f} vs stored "
        f"{d8['stored']['score']['mean']:.3f} (brief {STORED_CONTROL}); "
        f"gap {gap:.3f}, tolerance {GATE_TOLERANCE}: {'PASS' if gap <= GATE_TOLERANCE else 'FAIL'}"
    )
    for site, res in results.items():
        lines.append(f"\n== {site}: {res['rows']:,} rows (GAM stack weekly {STACK_WEEKLY} at D8)")
        for name, e in res["arms"].items():
            s = e["score"]
            diff = e.get("minus_control")
            text = (
                f" minus control {diff['mean']:+.3f} ({diff['low']:+.3f} to {diff['high']:+.3f})"
                if diff
                else ""
            )
            lines.append(
                f"{name:18s} {s['mean']:.3f} ({s['low']:.3f} to {s['high']:.3f}){text}; "
                f"MAE {e['mae_median_points']:.3f} bias {e['bias_median_points']:+.3f} "
                f"corr {e['corr_median']:.4f} cov {e['coverage_q05_q95']:.3f}"
            )
            for label, b in e["bands"].items():
                d = b.get("minus_control")
                bt = (
                    f" minus control {d['mean']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f})"
                    if d
                    else ""
                )
                lines.append(
                    f"    {label:10s} {b['score']['mean']:.3f}{bt}; "
                    f"MAE {b['mae_median_points']:.3f} bias {b['bias_median_points']:+.3f} "
                    f"corr {b['corr_median']:.4f}"
                )
    report = "\n".join(lines)
    print(report)
    (WORK_DIR / "results.json").write_text(json.dumps(results, indent=1))
    (WORK_DIR / "printed_report.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()

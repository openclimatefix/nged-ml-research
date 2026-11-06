# ruff: noqa: E501
"""Experiment 595: summary tables of the shares of M0, D5, P5, C1, and A1, with the pre-registered checks.

Reads `scores/event_shares.parquet` (written by `e00595_score.py`), prints every table the result page
quotes, and saves the same text as `scores/report.txt`. Bootstrap blocks are designs (both Primaries of
one injected transfer share a block). Counted tests: 6 (C1 minus P5 and A1 minus P5, each for the first
4 days, day 4 to the end, and the 7 days after the end, total share at q50, day-14 refit, paired by
design). C1 minus A1 and every refit-share contrast are descriptive.
"""

from typing import Final

import e00587_common as c587  # ty: ignore[unresolved-import]
import e00595_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 2000
N_TESTS: Final[int] = 6
PHASES: Final[tuple[str, ...]] = ("before", "first_4_days", "day_4_to_end", "after_7_days")
TESTED: Final[tuple[str, ...]] = PHASES[1:]
QUANTILES: Final[tuple[str, ...]] = ("q01", "q05", "q50", "q95", "q99")
HEADLINE_FLOOR: Final[float] = 0.15
FAIL_CEILING: Final[float] = 0.10


def interval(
    *, values: np.ndarray, level: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return the mean of per-design values and the percentile bootstrap interval at two-sided `level`."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(N_DRAWS, len(values))).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, level / 2)),
        float(np.quantile(draws, 1 - level / 2)),
    )


def per_design(*, shares: pl.DataFrame, value: str) -> pl.DataFrame:
    """Return one value per design, arm, refit day, and phase (mean of its two Primaries)."""
    return (
        shares.filter(pl.col("case") != "verified")
        .group_by("case", "arm", "refit_day", "phase")
        .agg(pl.col(value).mean(), pl.col("n_half_hours").mean())
        .sort("case", "arm", "refit_day", "phase")
    )


def fmt(*, m: float, lo: float, hi: float) -> str:
    """Return `m (lo to hi)` with three decimals."""
    return f"{m:+.3f} ({lo:+.3f} to {hi:+.3f})"


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print the injected-design, contrast, verified-pair, tail, and design-cell tables."""
    shares = pl.read_parquet(common.OUT / "scores" / "event_shares.parquet")
    arms = list(common.ALL_ARMS)
    lines: list[str] = []
    for value, title in (("total_q50", "total share"), ("refit_q50", "refit share")):
        d = per_design(shares=shares, value=value)
        lines.append(
            f"\n## Injected designs: {title} at q50 (mean over designs, 95% interval; n designs)"
        )
        for refit_day in (14, 0):
            lines.append(f"\nrefit day {refit_day}")
            lines.append("| phase | " + " | ".join(arms) + " |")
            lines.append("|---|" + "---|" * len(arms))
            for phase in PHASES:
                cells = []
                for arm in arms:
                    v = d.filter(
                        (pl.col("arm") == arm)
                        & (pl.col("refit_day") == refit_day)
                        & (pl.col("phase") == phase)
                    )[value].to_numpy()
                    if len(v) == 0:
                        cells.append("-")
                    else:
                        m, lo, hi = interval(values=v)
                        cells.append(f"{fmt(m=m, lo=lo, hi=hi)}; n={len(v)}")
                lines.append(f"| {phase} | " + " | ".join(cells) + " |")
    d = per_design(shares=shares, value="total_q50")
    lines.append(
        f"\n## Contrasts of the total share at the day-14 refit, Bonferroni over {N_TESTS} tests (paired by design)"
    )
    lines.append(
        "| contrast | phase | difference (Bonferroni interval) | minimum detectable effect | designs |"
    )
    lines.append("|---|---|---|---|---|")
    for a, b, counted in (("C1", "P5", True), ("A1", "P5", True), ("A1", "C1", False)):
        for phase in TESTED:
            x = d.filter((pl.col("refit_day") == 14) & (pl.col("phase") == phase))
            j = x.filter(pl.col("arm") == a).join(
                x.filter(pl.col("arm") == b), on="case", suffix="_b"
            )
            label = f"{a} minus {b}" + ("" if counted else " (descriptive)")
            if j.height < 2:
                lines.append(f"| {label} | {phase} | - | - | {j.height} |")
                continue
            diff = (j["total_q50"] - j["total_q50_b"]).to_numpy()
            m, lo, hi = interval(values=diff, level=0.05 / N_TESTS if counted else 0.05)
            lines.append(
                f"| {label} | {phase} | {fmt(m=m, lo=lo, hi=hi)} | {(hi - lo) / 2:.3f} | {j.height} |"
            )
    lines.append(
        "\n## Pre-registered check: total share at q50, day-14 refit, 7 days after the end, mean over designs"
    )
    after = d.filter((pl.col("refit_day") == 14) & (pl.col("phase") == "after_7_days"))
    means = {}
    for arm in ("C1", "A1"):
        v = after.filter(pl.col("arm") == arm)["total_q50"].to_numpy()
        if len(v):
            means[arm] = interval(values=v)
            m, lo, hi = means[arm]
            lines.append(
                f"- {arm}: {fmt(m=m, lo=lo, hi=hi)}; median {np.median(v):+.3f}; designs {len(v)}"
            )
    if "C1" in means:
        m = means["C1"][0]
        lines.append(
            f"- C1 prediction (at least {HEADLINE_FLOOR}): {'holds' if m >= HEADLINE_FLOOR else 'does not hold'}; fails if at most {FAIL_CEILING}: {'FAILS' if m <= FAIL_CEILING else 'does not fail'}"
        )
    if "A1" in means:
        m = means["A1"][0]
        lines.append(
            f"- A1 prediction P2 (at least {HEADLINE_FLOOR}): {'holds' if m >= HEADLINE_FLOOR else 'does not hold'}; at most {FAIL_CEILING}: {m <= FAIL_CEILING}"
        )
    if len(means) == 2:
        lines.append(
            f"- both C1 and A1 at most {FAIL_CEILING}: {all(v[0] <= FAIL_CEILING for v in means.values())}"
        )
    lines.append("\n## Median over designs of the total share at the day-14 refit")
    lines.append("| arm | " + " | ".join(TESTED) + " |")
    lines.append("|---|---|---|---|")
    for arm in arms:
        cells = []
        for phase in TESTED:
            v = d.filter(
                (pl.col("arm") == arm) & (pl.col("refit_day") == 14) & (pl.col("phase") == phase)
            )["total_q50"]
            cells.append(f"{v.median():+.3f}" if len(v) else "-")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append(
        "\n## Verified D5/D16 pair (per Primary, mean over seeds; share of the verified depth; descriptive)"
    )
    lines.append(
        "| arm | Primary | fold month | refit day | phase | total q50 | refit q50 | half-hours |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    v = shares.filter(pl.col("case") == "verified").sort(
        "arm", "label", "month", "refit_day", "phase"
    )
    lines.extend(
        f"| {r['arm']} | {r['label']} | {r['month']} | {r['refit_day']} | {r['phase']} | {r['total_q50']:+.3f} | {r['refit_q50']:+.3f} | {r['n_half_hours']:.0f} |"
        for r in v.iter_rows(named=True)
    )
    lines.append(
        "\n## Tail shares at the day-14 refit, injected designs, 7 days after the end (total share)"
    )
    lines.append("| arm | " + " | ".join(QUANTILES) + " |")
    lines.append("|---|" + "---|" * len(QUANTILES))
    for arm in arms:
        x = shares.filter(
            (pl.col("case") != "verified")
            & (pl.col("arm") == arm)
            & (pl.col("refit_day") == 14)
            & (pl.col("phase") == "after_7_days")
        )
        lines.append(
            f"| {arm} | "
            + " | ".join(f"{x[f'total_{q}'].mean():+.3f}" if x.height else "-" for q in QUANTILES)
            + " |"
        )
    lines.append(
        "\n## Total share at the day-14 refit by design cell, days 4 to the end and 7 days after pooled (mean; min; max over designs)"
    )
    lines.append("| arm | length (days) | start day | designs | mean | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    dz = c587.designs().select(
        case=pl.format("inj{}", pl.col("spec_id")),
        length=pl.col("length_days"),
        start=pl.col("start_day"),
    )
    cell = (
        shares.filter(
            (pl.col("case") != "verified")
            & (pl.col("refit_day") == 14)
            & pl.col("phase").is_in(["day_4_to_end", "after_7_days"])
        )
        .group_by("case", "arm")
        .agg(pl.col("total_q50").mean())
        .join(dz, on="case")
        .group_by("arm", "length", "start")
        .agg(
            pl.len().alias("designs"),
            pl.col("total_q50").mean().alias("mean"),
            pl.col("total_q50").min().alias("lo"),
            pl.col("total_q50").max().alias("hi"),
        )
        .sort("arm", "length", "start")
    )
    lines.extend(
        f"| {r['arm']} | {r['length']} | {r['start']} | {r['designs']} | {r['mean']:+.3f} | {r['lo']:+.3f} | {r['hi']:+.3f} |"
        for r in cell.iter_rows(named=True)
    )
    text = "\n".join(lines)
    (common.OUT / "scores" / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

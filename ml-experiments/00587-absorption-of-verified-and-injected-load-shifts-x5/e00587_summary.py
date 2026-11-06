# ruff: noqa: E501
"""Experiment 587: summary tables of the shares, with bootstrap intervals and contrasts.

Reads `scores/event_shares.parquet` (written by `e00587_score.py`) and prints every table the
result page quotes; the same text is saved as `scores/report.txt`. Blocks for the bootstrap are
designs (both Primaries of one injected transfer share a block), so the interval reflects
design-to-design variation.
"""

from typing import Final

import e00587_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 2000
N_TESTS: Final[int] = 6
"""Counted tests: D5 minus M0 and P5 minus D5, each in three phases (first 4 days, day 4 to the end, 7 days after the end)."""
PHASES: Final[tuple[str, ...]] = ("before", "first_4_days", "day_4_to_end", "after_7_days")
TESTED: Final[tuple[str, ...]] = PHASES[1:]
QUANTILES: Final[tuple[str, ...]] = ("q01", "q05", "q50", "q95", "q99")


def interval(
    *, values: np.ndarray, level: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return the mean of per-block values and the percentile bootstrap interval at two-sided `level`."""
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
    """Print the injected-event, contrast, and verified-pair tables."""
    shares = pl.read_parquet(common.OUT / "scores" / "event_shares.parquet")
    lines: list[str] = []
    for value, title in (("refit_q50", "refit share"), ("total_q50", "total share")):
        d = per_design(shares=shares, value=value)
        lines.append(
            f"\n## Injected designs: {title} at q50 (mean over designs, 95% interval; n designs)"
        )
        for refit_day in (0, 14):
            lines.append(f"\nrefit day {refit_day}")
            lines.append("| phase | " + " | ".join(common.ARMS) + " |")
            lines.append("|---|" + "---|" * len(common.ARMS))
            for phase in PHASES:
                cells = []
                for arm in common.ARMS:
                    v = d.filter(
                        (pl.col("arm") == arm)
                        & (pl.col("refit_day") == refit_day)
                        & (pl.col("phase") == phase)
                    )[value].to_numpy()
                    cells.append(
                        "-"
                        if len(v) == 0
                        else f"{fmt(m=interval(values=v)[0], lo=interval(values=v)[1], hi=interval(values=v)[2])}; n={len(v)}"
                    )
                lines.append(f"| {phase} | " + " | ".join(cells) + " |")
    d = per_design(shares=shares, value="refit_q50")
    lines.append(
        f"\n## Contrasts of the refit share at the day-14 refit, Bonferroni over {N_TESTS} tests (paired by design)"
    )
    lines.append(
        "| contrast | phase | difference (Bonferroni interval) | minimum detectable effect | designs |"
    )
    lines.append("|---|---|---|---|---|")
    for a, b in (("D5", "M0"), ("P5", "D5")):
        for phase in TESTED:
            x = d.filter((pl.col("refit_day") == 14) & (pl.col("phase") == phase))
            j = x.filter(pl.col("arm") == a).join(
                x.filter(pl.col("arm") == b), on="case", suffix="_b"
            )
            if j.height < 2:
                lines.append(f"| {a} minus {b} | {phase} | - | - | {j.height} |")
                continue
            diff = (j["refit_q50"] - j["refit_q50_b"]).to_numpy()
            m, lo, hi = interval(values=diff, level=0.05 / N_TESTS)
            lines.append(
                f"| {a} minus {b} | {phase} | {fmt(m=m, lo=lo, hi=hi)} | {(hi - lo) / 2:.3f} | {j.height} |"
            )
    lines.append(
        "\n## Verified D5/D16 pair (per Primary, mean over seeds; share of the verified depth)"
    )
    lines.append(
        "| arm | Primary | fold month | refit day | phase | refit q50 | total q50 | half-hours |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    v = shares.filter(pl.col("case") == "verified").sort(
        "arm", "label", "month", "refit_day", "phase"
    )
    lines.extend(
        f"| {r['arm']} | {r['label']} | {r['month']} | {r['refit_day']} | {r['phase']} | {r['refit_q50']:+.3f} | {r['total_q50']:+.3f} | {r['n_half_hours']:.0f} |"
        for r in v.iter_rows(named=True)
    )
    lines.append(
        "\n## Tail shares at the day-14 refit, injected designs, day 4 to the end and 7 days after the end (refit share)"
    )
    lines.append("| arm | phase | " + " | ".join(QUANTILES) + " |")
    lines.append("|---|---|" + "---|" * len(QUANTILES))
    for arm in common.ARMS:
        for phase in ("day_4_to_end", "after_7_days"):
            x = shares.filter(
                (pl.col("case") != "verified")
                & (pl.col("arm") == arm)
                & (pl.col("refit_day") == 14)
                & (pl.col("phase") == phase)
            )
            lines.append(
                f"| {arm} | {phase} | "
                + " | ".join(
                    f"{x[f'refit_{q}'].mean():+.3f}" if x.height else "-" for q in QUANTILES
                )
                + " |"
            )
    lines.append(
        "\n## Refit share at the day-14 refit by design cell, days 4 to the end and 7 days after the end pooled (mean over designs; range of designs)"
    )
    lines.append("| arm | length (days) | start day | designs | mean | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    dz = common.designs().select(
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
        .agg(pl.col("refit_q50").mean())
        .join(dz, on="case")
        .group_by("arm", "length", "start")
        .agg(
            pl.len().alias("designs"),
            pl.col("refit_q50").mean().alias("mean"),
            pl.col("refit_q50").min().alias("lo"),
            pl.col("refit_q50").max().alias("hi"),
        )
        .sort("arm", "length", "start")
    )
    lines.extend(
        f"| {r['arm']} | {r['length']} | {r['start']} | {r['designs']} | {r['mean']:+.3f} | {r['lo']:+.3f} | {r['hi']:+.3f} |"
        for r in cell.iter_rows(named=True)
    )
    lines.append(
        "\n## Median over designs of the refit share at the day-14 refit (the mean is carried by three 15-day designs)"
    )
    lines.append("| arm | " + " | ".join(summ_phase for summ_phase in TESTED) + " |")
    lines.append("|---|---|---|---|")
    for arm in common.ARMS:
        cells = []
        for phase in TESTED:
            v = d.filter(
                (pl.col("arm") == arm) & (pl.col("refit_day") == 14) & (pl.col("phase") == phase)
            )["refit_q50"]
            cells.append(f"{v.median():+.3f}" if len(v) else "-")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    dz2 = common.designs().select(
        case=pl.format("inj{}", pl.col("spec_id")),
        depth=pl.col("depth_share_of_smaller_capacity"),
        length=pl.col("length_days"),
    )
    p1 = (
        shares.filter(
            (pl.col("case") != "verified")
            & (pl.col("refit_day") == 14)
            & (pl.col("phase") == "after_7_days")
        )
        .group_by("case", "arm")
        .agg(pl.col("refit_q50").mean())
        .join(dz2, on="case")
        .filter(pl.col("length").is_in([15, 30]) & pl.col("depth").is_in([0.3, 0.5]))
    )
    lines.append(
        "\n## P1 subset: 15 and 30 day events at 30% and 50% depth, day-14 refit, 7 days after the end (mean over designs; range; designs)"
    )
    for arm in common.ARMS:
        v = p1.filter(pl.col("arm") == arm)["refit_q50"]
        lines.append(
            f"- {arm}: {v.mean():+.3f} (range {v.min():+.3f} to {v.max():+.3f}); {len(v)} designs; median {v.median():+.3f}"
        )
    kn = pl.read_parquet(common.OUT / "scores" / "knot_movement.parquet").filter(
        (pl.col("case") != "verified") & (pl.col("refit_day") == 14)
    )
    kn = kn.group_by("arm", "case", "label").agg(
        pl.col("knot_movement").mean(), pl.col("depth").first()
    )
    lines.append(
        "\n## Newest-knot movement at the day-14 refit, injected (share of capacity; mean, largest, Primary-events above 0.03 in either direction)"
    )
    for arm in ("D5", "P5"):
        k = kn.filter(pl.col("arm") == arm)
        lines.append(
            f"- {arm}: mean {k['knot_movement'].mean():+.4f}, largest {k['knot_movement'].max():+.4f}, above 0.03: {int((k['knot_movement'].abs() > 0.03).sum())} of {k.height}"
        )
    absorbed = (
        shares.filter(
            (pl.col("case") != "verified")
            & (pl.col("refit_day") == 14)
            & pl.col("phase").is_in(["day_4_to_end", "after_7_days"])
        )
        .group_by("case", "arm", "label", "depth")
        .agg(pl.col("refit_q50").mean())
        .with_columns((pl.col("refit_q50") * pl.col("depth").abs()).alias("absorbed_capacity"))
    )
    lines.append(
        "\n## Absorbed amount in capacity units at the day-14 refit (refit share x |depth|, days 4 to the end and 7 days after pooled, mean over Primary-events), beside |depth|"
    )
    for arm in common.ARMS:
        a = absorbed.filter(pl.col("arm") == arm)
        lines.append(
            f"- {arm}: absorbed {a['absorbed_capacity'].mean():+.3f} of capacity; mean |depth| {a['depth'].abs().mean():.3f} of capacity; Primary-events {a.height}"
        )
    text = "\n".join(lines)
    (common.OUT / "scores" / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

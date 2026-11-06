# ruff: noqa: E501, PLR0912, PLR0915
"""Experiment 599: summary tables of the shares of E0, E1, E2w, and E3, with the pre-registered checks.

Reads `scores/event_shares.parquet` (written by `e00599_score.py`) and prints every table the result
page quotes; saves the same text as `scores/report.txt`. Bootstrap blocks are designs (both
Primaries of one injected transfer share a block). Counted tests: 3 (E1, E2w, and E3 minus E0, total
share at q50, day-14 refit, 7 days after the end, paired by design), Bonferroni level 0.05 / 3.
Everything else (other phases, refit shares, tails, the verified pair) is descriptive.
"""

from typing import Final

import e00595_summary as s595  # ty: ignore[unresolved-import]
import e00599_common as common  # ty: ignore[unresolved-import]
import polars as pl

N_TESTS: Final[int] = 3
TARGET_DROP: Final[float] = 0.04
PHASES = s595.PHASES
QUANTILES = s595.QUANTILES


def main() -> None:
    """Print the share tables, the three counted contrasts, and the P1 check."""
    shares = pl.read_parquet(common.OUT / "scores" / "event_shares.parquet")
    arms = list(common.ALL_ARMS)
    lines: list[str] = []
    for value, title in (("total_q50", "total share"), ("refit_q50", "refit share")):
        d = s595.per_design(shares=shares, value=value)
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
                        m, lo, hi = s595.interval(values=v)
                        cells.append(f"{s595.fmt(m=m, lo=lo, hi=hi)}; n={len(v)}")
                lines.append(f"| {phase} | " + " | ".join(cells) + " |")
    d = s595.per_design(shares=shares, value="total_q50")
    lines.append(
        f"\n## Contrasts of the total share at the day-14 refit, Bonferroni over {N_TESTS} tests (paired by design)"
    )
    lines.append(
        "| contrast | phase | difference (interval) | minimum detectable effect | designs | counted |"
    )
    lines.append("|---|---|---|---|---|---|")
    for a in common.NEW_ARMS:
        for phase in PHASES[1:]:
            counted = phase == "after_7_days"
            x = d.filter((pl.col("refit_day") == 14) & (pl.col("phase") == phase))
            j = x.filter(pl.col("arm") == a).join(
                x.filter(pl.col("arm") == "E0"), on="case", suffix="_b"
            )
            if j.height < 2:
                lines.append(f"| {a} minus E0 | {phase} | - | - | {j.height} | {counted} |")
                continue
            diff = (j["total_q50"] - j["total_q50_b"]).to_numpy()
            m, lo, hi = s595.interval(values=diff, level=0.05 / N_TESTS if counted else 0.05)
            lines.append(
                f"| {a} minus E0 | {phase} | {s595.fmt(m=m, lo=lo, hi=hi)} | {(hi - lo) / 2:.3f} | {j.height} | {counted} |"
            )
    lines.append(
        "\n## Pre-registered P1: total share 7 days after the end, day-14 refit, mean over designs, minus E0"
    )
    after = d.filter((pl.col("refit_day") == 14) & (pl.col("phase") == "after_7_days"))
    base = after.filter(pl.col("arm") == "E0")["total_q50"].mean()
    hit = False
    for a in common.NEW_ARMS:
        v = after.filter(pl.col("arm") == a)["total_q50"].to_numpy()
        if len(v):
            drop = base - float(v.mean())
            hit = hit or drop >= TARGET_DROP
            lines.append(
                f"- {a}: share {v.mean():+.3f}, drop from E0 ({base:+.3f}) {drop:+.3f}; drop at least {TARGET_DROP}: {drop >= TARGET_DROP}"
            )
    lines.append(
        f"- P1 (at least one arm drops the share by {TARGET_DROP}): {'holds' if hit else 'does not hold'}"
    )
    lines.append("\n## Median over designs of the total share at the day-14 refit")
    lines.append("| arm | " + " | ".join(PHASES[1:]) + " |")
    lines.append("|---|---|---|---|")
    for arm in arms:
        cells = []
        for phase in PHASES[1:]:
            v = d.filter(
                (pl.col("arm") == arm) & (pl.col("refit_day") == 14) & (pl.col("phase") == phase)
            )["total_q50"]
            cells.append(f"{v.median():+.3f}" if len(v) else "-")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append("\n## Verified D5/D16 pair (per Primary, mean over seeds; descriptive)")
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
    text = "\n".join(lines)
    (common.OUT / "scores" / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

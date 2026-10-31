# ruff: noqa: E501
"""Absolute scores, planned contrasts with month-block intervals, and tails for experiment 418.

Reads the seed-averaged forecasts of the arms under `work/<folder>/`, restricts them and the stored
references to their shared half-hours, and prints:

- CRPS at days 1 to 10 per group, with lead bands, beside BG2, `manual_heuristic`, and climatology;
- C1 to C5 with the 95% month-block bootstrap of `studies.eval_protocol`;
- q95 and q99 coverage per group;
- C6 (`--lowdata`): the pretrained arm minus the ENS-only network on the first six months of ENS,
  minus the same contrast on the full window.

```bash
source /tmp/e418_env.sh
uv run --no-sync python ml-experiments/00418-*/e00418_contrasts.py full
```
"""

import argparse
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00418_fit as fit  # ty: ignore[unresolved-import]
import e00418_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

LABELS: Final[dict[str, tuple[str, ...]]] = {
    "signed-MW six": ("D5", "D9", "D10", "D11", "D14", "D16"),
    "MVA-only eight": ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
    "MVA-only seven (no D8)": ("D2", "D3", "D4", "D12", "D19", "D21", "D22"),
    "D15 and D18": ("D15", "D18"),
}
ALL_MONTHS: Final[tuple[str, ...]] = (
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 P1 - ens-only": ("P1", "ens-only"),
    "C2 P2 - ens-only": ("P2", "ens-only"),
    "C3 P3 - ens-only": ("P3", "ens-only"),
    "C4 P3 - P1": ("P3", "P1"),
    "C5 P2 - P1": ("P2", "P1"),
}


def contrast_rows(
    *, rows: dict[str, pl.DataFrame], table: pl.DataFrame, treatment: str, reference: str
) -> None:
    """Print one contrast per group with its 95% month-block interval."""
    for group, labels in LABELS.items():
        where = pl.col("label").is_in(list(labels)) & pl.col(s397.HEADLINE)
        r = metrics.month_bootstrap_contrast(
            treatment=rows[treatment],
            reference=rows[reference],
            metric="crps",
            where=where,
            level_percent=95.0,
            n_resamples=20_000,
            slice_table=table,
        )
        print(
            f"  {group:24s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
        )


def full(*, months: list[str], arms: list[str], references: bool = True) -> None:
    """Score the full folds: arms against the ENS-only refit, with the stored 397 forecasts as an extra row."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    frames = score.arm_frames(folder="seed_preds", months=months, arms=arms, series=series)
    if references:
        stored_months = [m for m in months if m in fit.SCREEN_MONTHS or m == "2025-12"]
        frames["e397_pooled_stored"] = adapter.forecast_frame(
            stored=score.average_dir(
                folder=s397.SEED_DIR / "summary" / "plain_pooled", months=stored_months
            ),
            series=series,
            unit="fraction",
        )
        frames.update(s397.references(series=series))
    names = {fit.arm_name(arm=a): a for a in arms}
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    print("\n== CRPS, days 1 to 10, equal weight per Primary, all folds in the arms ==")
    for group, labels in LABELS.items():
        values = {}
        for name, frame in scored.items():
            sel = frame.filter(
                pl.col("metric") == "crps",
                pl.col("slice") == s397.HEADLINE,
                pl.col("label").is_in(list(labels)),
            )
            values[names.get(name, name)] = float(sel["value"].mean())
        print(
            f"{group}: "
            + ", ".join(f"{k} {v:.3f}" for k, v in sorted(values.items(), key=lambda kv: kv[1]))
        )
    inverse = {a: n for n, a in names.items()}
    print("\n== planned contrasts (negative favours the first), 95% month-block interval ==")
    for label, (t, r) in CONTRASTS.items():
        if t in inverse and r in inverse:
            print(label)
            contrast_rows(rows=rows, table=table, treatment=inverse[t], reference=inverse[r])
    print("\n== tails (nominal 0.95, 0.99) ==")
    for metric in score.NOMINAL:
        for group, labels in LABELS.items():
            vals = {}
            for name, frame in scored.items():
                sel = frame.filter(
                    pl.col("metric") == metric,
                    pl.col("slice") == s397.HEADLINE,
                    pl.col("label").is_in(list(labels)),
                )
                vals[names.get(name, name)] = float(sel["value"].mean())
            print(
                f"{metric} {group}: "
                + ", ".join(
                    f"{k} {v:.3f}" for k, v in vals.items() if k in names.values() or k == "bg2"
                )
            )
    per_fold = []
    for month in months:
        sub = {
            k: s397.only_month(frame=v, month=month)
            for k, v in frames.items()
            if k in inverse.values() or k in names
        }
        _, sc, _ = s381.score_arms(arms=sub, series=series, ctx=ctx)
        for name, frame in sc.items():
            sel = frame.filter(
                pl.col("metric") == "crps",
                pl.col("slice") == s397.HEADLINE,
                pl.col("label").is_in(list(LABELS["signed-MW six"])),
            )
            per_fold.append(
                {
                    "fold": month,
                    "arm": names.get(name, name),
                    "crps_six": float(sel["value"].mean()),
                }
            )
    print("\n== signed-MW six CRPS by fold ==")
    with pl.Config(tbl_rows=60):
        print(pl.DataFrame(per_fold).pivot(on="fold", index="arm", values="crps_six"))
    _ = np


def lowdata(*, months: list[str], arm: str) -> None:
    """Print C6: the pretrained arm minus the ENS-only network, on six months of ENS and on the full window."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    for label, folder in (("six months of ENS", "lowdata"), ("full window", "seed_preds")):
        frames = score.arm_frames(
            folder=folder, months=months, arms=["ens-only", arm], series=series
        )
        table, _, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
        print(f"\n== {arm} minus ens-only, {label} ==")
        contrast_rows(
            rows=rows,
            table=table,
            treatment=fit.arm_name(arm=arm),
            reference=fit.arm_name(arm="ens-only"),
        )


def main() -> None:
    """Run a mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("full", "lowdata"))
    parser.add_argument("--months", nargs="+", default=list(ALL_MONTHS))
    parser.add_argument("--arms", nargs="+", default=["ens-only", "P1", "P2", "P3"])
    parser.add_argument("--arm", default="P1")
    parser.add_argument(
        "--no-references", action="store_true", help="score the arms alone, on all their half-hours"
    )
    args = parser.parse_args()
    if args.mode == "full":
        full(months=args.months, arms=args.arms, references=not args.no_references)
    else:
        lowdata(months=list(fit.SCREEN_MONTHS), arm=args.arm)


if __name__ == "__main__":
    main()

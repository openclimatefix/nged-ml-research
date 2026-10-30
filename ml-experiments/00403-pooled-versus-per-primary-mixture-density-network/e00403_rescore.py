# ruff: noqa: E501
"""Re-score experiment 397's pooled and per-Primary networks on the same rows (403, no refit).

Reads experiment 397's stored per-seed forecasts (summary scheme, five screen months), averages the
seeds in quantile space, restricts every arm to the half-hours they share, and prints:

- absolute CRPS at days 1 to 10 per group (signed-MW six, MVA-only eight, D15 and D18 separately);
- the contrast pooled A0 minus the per-Primary plain network per group and per Primary, with the
  month-block bootstrap of `studies.eval_protocol` (five months, so every interval is interim);
- how the gain is spread over the Primaries, and its relation to months of history.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
uv run --no-sync python ml-experiments/00403-*/e00403_rescore.py
```
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SEED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00397-plain-and-attention-mixture-density-network-for-primary-demand"
    / "work"
    / "seed_preds"
    / "summary"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
ARMS: Final[dict[str, str]] = {
    "per_primary": "plain_own",
    "pooled_plain": "plain_pooled",
    "pooled_A0": "A0",
}
HEADLINE: Final[str] = base.HEADLINE_BAND
SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
SIGNED: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SUBSETS: Final[dict[str, tuple[str, ...]]] = {
    "signed-MW six": SIGNED,
    "MVA-only eight": MVA_ONLY,
    "MVA-only seven (no D8)": tuple(s for s in MVA_ONLY if s != "D8"),
    "D15 and D18": SHORT_HISTORY,
}
KEYS: Final[list[str]] = list(common.KEYS)


def month_origins(*, month: str) -> list[datetime]:
    """Return the daily forecast origins of a test month."""
    start = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
    stop = (start + timedelta(days=32)).replace(day=1)
    return [o for o in common.all_origins() if start <= o < stop]


def seed_average(*, arm_dir: str) -> pl.DataFrame:
    """Return an arm's forecast (fraction of capacity) averaged over seeds, all five months."""
    pieces = []
    for month in MONTHS:
        files = [
            f
            for f in sorted((SEED_DIR / arm_dir).glob(f"{month}_s*.parquet"))
            if not f.stem.endswith("_gate")
        ]
        stacked = pl.concat([pl.read_parquet(f) for f in files])
        pieces.append(stacked.group_by(KEYS).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS]))
    frame = pl.concat(pieces).sort(KEYS)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def main() -> None:
    """Print the absolute table, the contrasts, and the spread of the gain."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    frames = {
        name: adapter.forecast_frame(
            stored=seed_average(arm_dir=arm_dir), series=series, unit="fraction"
        )
        for name, arm_dir in ARMS.items()
    }
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    crps = pl.concat(
        [
            t.filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE).select(
                "label", pl.lit(n).alias("arm"), "value"
            )
            for n, t in scored.items()
        ]
    ).pivot(on="arm", index="label", values="value")
    crps = crps.with_columns(
        gain=pl.col("per_primary") - pl.col("pooled_A0"),
        gain_plain=pl.col("per_primary") - pl.col("pooled_plain"),
        history_months=pl.col("label").replace_strict(
            history_months(series=series, ctx=ctx), default=None
        ),
    ).sort(pl.col("label").str.slice(1).cast(pl.Int32))
    with pl.Config(tbl_rows=40):
        print("\n== CRPS days 1 to 10 per Primary (points); gain = per-Primary minus pooled ==")
        print(crps)
    print("\n== group means (equal weight per Primary) ==")
    for name, labels in SUBSETS.items():
        sub = crps.filter(pl.col("label").is_in(list(labels)))
        print(
            f"{name:24s} per-Primary {sub['per_primary'].mean():.3f}  "
            f"pooled plain {sub['pooled_plain'].mean():.3f}  pooled A0 {sub['pooled_A0'].mean():.3f}  "
            f"gain A0 {sub['gain'].mean():+.3f}  gain plain {sub['gain_plain'].mean():+.3f}  "
            f"Primaries won by pooled A0: {(sub['gain'] > 0).sum()} of {sub.height}"
        )
    print("\n== pooled A0 minus per-Primary plain, month-block bootstrap (interim, 5 months) ==")
    for name, labels in SUBSETS.items():
        where = pl.col("label").is_in(list(labels)) & pl.col(HEADLINE)
        r = metrics.month_bootstrap_contrast(
            treatment=rows["pooled_A0"],
            reference=rows["per_primary"],
            metric="crps",
            where=where,
            level_percent=95.0,
            n_resamples=20_000,
            slice_table=table,
        )
        print(
            f"{name:24s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) "
            f"n={r['n_half_hours']}"
        )
    print("\n== per Primary, 95% (pooled A0 minus per-Primary plain) ==")
    print(
        metrics.reported_contrast_by_primary(
            treatment=rows["pooled_A0"],
            reference=rows["per_primary"],
            metric="crps",
            where=pl.col(HEADLINE),
            slice_table=table,
        )
    )
    gains = crps.filter(~pl.col("label").is_in(list(SHORT_HISTORY)))["gain"].sort(descending=True)
    total = float(gains.sum())
    print(
        f"\nSpread of the gain over the 14 countable Primaries: the best 3 carry "
        f"{gains.head(3).sum() / total:.0%} of the summed gain; "
        f"{(gains > 0).sum()} of {gains.len()} gain at all."
    )
    countable = crps.filter(~pl.col("label").is_in(list(SHORT_HISTORY)))
    r = np.corrcoef(countable["gain"].to_numpy(), countable["history_months"].to_numpy())[0, 1]
    print(f"Correlation of the gain with months of history (14 Primaries): {r:+.2f}")


def history_months(*, series: pl.DataFrame, ctx: dict) -> dict[str, float]:
    """Return each Primary's months of readings before 2025-10-01."""
    cut = datetime(2025, 10, 1, tzinfo=UTC)
    label_of = dict(zip(series["time_series_id"].to_list(), series["label"].to_list(), strict=True))
    return {
        label_of[int(sid)]: max((cut - first).days / 30.44, 0.0)
        for sid, first in ctx["first_readings"].items()
    }


if __name__ == "__main__":
    main()

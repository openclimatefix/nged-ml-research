"""Score the arms of experiment 449 beside the references on the same half-hours (no fit).

Wraps experiment 448's `e00448_refs` (its row set, slices, tail metrics, and curtailment cost) and
redirects it to this experiment's predictions folder. Added here: experiment 425's arm A, as a
single seed (seed 0) and as the seed average of seeds 0 to 4 (quantile-space mean, as experiment 425
scores it), both widened by 1.15, as references joined on the keys like the other references.
The stored five-seed pools of experiment 448 (`S1-pool5`, `S1PT-pool5`) are linked into this
experiment's predictions folder by `e00449_score.py link`.

```bash
source /tmp/e449_env.sh
uv run --no-sync python ml-experiments/00449-*/e00449_refs.py --tag four_months
uv run --no-sync python ml-experiments/00449-*/e00449_refs.py --months 2025-12 --tag december
```
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00425_score as s425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00449_run as run  # ty: ignore[unresolved-import]
import polars as pl

ARMS = (
    "S1", "S1-pool5", "S1PT-pool5", "PT0", "PT1", "PT1-full", "PT1b", "PT2",
    "PT1-pool3", "PT1-pool5",
)  # fmt: skip
A_FOLDER = "A"
_BASE_REFERENCE_FRAMES = base.reference_frames
"""Experiment 448's reference builder, kept before `main` replaces the module attribute."""


def a425_frame(
    *, series: pl.DataFrame, seeds: tuple[int, ...], months: tuple[str, ...]
) -> pl.DataFrame:
    """Return experiment 425's arm A (seed average over `seeds`, widened) in MW."""
    arm = s425.ArmFiles(name="A", folder=A_FOLDER, seeds=seeds, months=months)
    frame = s425.seed_average(arm=arm)
    return s397.widen(frame=adapter.forecast_frame(stored=frame, series=series, unit="fraction"))


def reference_frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return experiment 448's references plus experiment 425's arm A (seed 0, seeds 0 to 4)."""
    out = _BASE_REFERENCE_FRAMES(series=series)
    months = ("2025-10", "2026-01", "2026-03", "2026-06", "2025-12")
    out["425-A-s0"] = a425_frame(series=series, seeds=(0,), months=months)
    out["425-A-avg5"] = a425_frame(series=series, seeds=(0, 1, 2, 3, 4), months=months)
    return out


def main() -> None:
    """Run experiment 448's scorer on this experiment's arms."""
    base.reference_frames = reference_frames
    base.run = run
    base.TABLE_DIR = run.WORK_DIR / "figure_tables"
    base.LADDER_ARMS = ARMS
    base.REFERENCES = (*base.REFERENCES, "425-A-s0", "425-A-avg5")
    base.SEEDS = (0, 1, 2)
    base.main()


if __name__ == "__main__":
    main()

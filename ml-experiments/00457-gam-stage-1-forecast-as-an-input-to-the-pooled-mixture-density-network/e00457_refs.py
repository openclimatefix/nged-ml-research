"""Reference frames for experiment 457: the stored arm A, the two new arms, and BG2 (no fit).

Reads experiment 425's arm A per-seed forecasts (never refitted) and this experiment's `Gin` and
`Gbase` per-seed forecasts, in the `seed_preds` layout of experiment 425. Each arm is returned as a
five-seed quantile average and as its single seeds, in MW, widened by `factor` (1.15 unless a
caller asks for another), so experiment 448's row builder can score them on the shared half-hours.
"""

from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_score as s425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00452_refs as refs452  # ty: ignore[unresolved-import]
import polars as pl

MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
NEW_ARMS: Final[tuple[str, ...]] = ("Gin", "Gbase")


def arm_frame(
    *, series: pl.DataFrame, folder: str, seeds: tuple[int, ...], root: str, factor: float
) -> pl.DataFrame:
    """Return an arm's seed-averaged forecast in MW, widened by `factor` about the median.

    `root` is `425` for the stored arm A and `457` for this experiment's arms.
    """
    saved = fit425.OUT_DIR
    if root == "457":
        import e00457_fit as fit457  # ty: ignore[unresolved-import]

        s425.fit.OUT_DIR = fit457.OUT_DIR
    try:
        arm = s425.ArmFiles(name=folder, folder=folder, seeds=seeds, months=MONTHS)
        frame = adapter.forecast_frame(
            stored=s425.seed_average(arm=arm), series=series, unit="fraction"
        )
    finally:
        s425.fit.OUT_DIR = saved
    outer = s397.OUTER
    return frame.with_columns(
        *[(pl.col("q50") + factor * (pl.col(c) - pl.col("q50"))).alias(c) for c in outer]
    )


def reference_frames(*, series: pl.DataFrame, factor: float = 1.15) -> dict[str, pl.DataFrame]:
    """Return BG2, manual heuristic, arm A, and the new arms (pools and single seeds) in MW."""
    out = {
        name: frame
        for name, frame in refs452.reference_frames(series=series).items()
        if name in ("BG2", "manual_heuristic", "climatology")
    }
    out["A-avg5"] = arm_frame(series=series, folder="A", seeds=SEEDS, root="425", factor=factor)
    for seed in SEEDS:
        out[f"A-s{seed}"] = arm_frame(
            series=series, folder="A", seeds=(seed,), root="425", factor=factor
        )
    for arm in NEW_ARMS:
        out[f"{arm}-avg5"] = arm_frame(
            series=series, folder=arm, seeds=SEEDS, root="457", factor=factor
        )
        for seed in SEEDS:
            out[f"{arm}-s{seed}"] = arm_frame(
                series=series, folder=arm, seeds=(seed,), root="457", factor=factor
            )
    _ = base
    return out

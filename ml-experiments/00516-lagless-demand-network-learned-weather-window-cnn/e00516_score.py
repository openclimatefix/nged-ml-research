# ruff: noqa: E501, PERF401
r"""Score experiment 516's arms against experiment 515's, on experiment 515's ten winter folds and rows.

Reuses experiment 515's scorer (`e00515_score`) with this experiment's arms added and the output
folder changed. The rows, slices, widening, 3-seed averaging, and fold-block bootstrap are 515's;
only the Bonferroni family changes: 8 primary tests (2 CNN variants x 2 groups x all rows and cold
nights). Contrast: arm minus reference, so negative favours the arm.

References: `W0` (the 37-column base), `C-CNN-null` (the extra-parameter control), and each of
experiment 515's hand-built arms `W-T`, `W-S`, `W-X`.

Commands: `rows`, `stats` (contrasts, per-fold signs, minimum detectable effect first), `interim`
(a short numbers file for a cold-start write-up).
"""

import argparse
from pathlib import Path
from typing import Final

import e00515_score as s515  # ty: ignore[unresolved-import]
import e00516_fit as fit
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORE_DIR: Final[Path] = fit.W / "scores"
CNN: Final[tuple[str, ...]] = ("C-CNN", "T-CNN")
CONTROL: Final[str] = "C-CNN-null"
HAND_BUILT: Final[tuple[str, ...]] = ("W-T", "W-S", "W-X")
BONFERRONI: Final[int] = 8
_ = ML_EXPERIMENTS_DATA_DIR


def _arm_folder(*, arm: str, month: str) -> Path:
    if arm in fit.ARMS:
        return fit.W / "seed_preds" / arm
    return _original_arm_folder(arm=arm, month=month)


_original_arm_folder = s515.arm_folder
s515.arm_folder = _arm_folder
s515.ARMS = ("W0", *HAND_BUILT, "W-null", *CNN, CONTROL)
s515.SCORE_DIR = SCORE_DIR
s515.BONFERRONI = BONFERRONI


def stats() -> None:
    """Write and print the contrasts."""
    frame = s515.load_rows()
    have = [a for a in (*CNN, CONTROL) if f"{a}__crps" in frame.columns]
    print(f"folds scored: {sorted(frame['month'].unique().to_list())}")
    mdes = pl.DataFrame(
        [
            s515.mde(frame=frame, group=g, slice_name=sl, arms=["W0", CONTROL])
            for g in s515.GROUPS
            for sl in ("all", "cold_night")
            if f"{CONTROL}__crps" in frame.columns
        ]
    )
    print("minimum detectable effect (3-seed mean, W0 and C-CNN-null spreads):")
    print(mdes)
    mdes.write_csv(SCORE_DIR / "mde.csv")
    records = []
    for arm in have:
        for ref in ("W0", CONTROL, *HAND_BUILT):
            if ref == arm or f"{ref}__crps" not in frame.columns:
                continue
            for g in s515.GROUPS:
                for sl in s515.SLICES:
                    records.append(s515.contrast(frame=frame, arm=arm, ref=ref, group=g, slice_name=sl))  # fmt: skip
                if ref == "W0":
                    for sl in ("all", "cold_night"):
                        for metric in ("pin_q95", "pin_q99", "cov_q95", "cov_q99"):
                            records.append(s515.contrast(frame=frame, arm=arm, ref=ref, group=g, slice_name=sl, metric=metric))  # fmt: skip
    table = pl.DataFrame(records)
    table.write_csv(SCORE_DIR / "contrasts.csv")
    with pl.Config(tbl_rows=300, tbl_cols=20):
        print(
            table.filter(
                (pl.col("metric") == "crps") & pl.col("slice").is_in(["all", "cold_night"])
            )
        )
    absolute = []
    for arm in ("W0", *HAND_BUILT, "W-null", *have):
        if f"{arm}__crps" not in frame.columns:
            continue
        for g in s515.GROUPS:
            for sl in ("all", "cold_night"):
                sums = s515.fold_sums(frame=frame, column=f"{arm}__crps", mask=s515.mask_of(frame=frame, group=g, slice_name=sl))  # fmt: skip
                absolute.append({"arm": arm, "group": g, "slice": sl, "crps": sum(v[0] for v in sums.values()) / max(sum(v[1] for v in sums.values()), 1)})  # fmt: skip
    print(pl.DataFrame(absolute))
    pl.DataFrame(absolute).write_csv(SCORE_DIR / "absolute.csv")


def interim() -> None:
    """Write `interim_numbers.md`: headline contrasts with intervals, per-fold signs, and caveats."""
    table = pl.read_csv(SCORE_DIR / "contrasts.csv")
    mde = pl.read_csv(SCORE_DIR / "mde.csv")
    head = table.filter(
        (pl.col("metric") == "crps") & pl.col("slice").is_in(["all", "cold_night"]) & pl.col("group").is_in(["eight", "six"])
    ).sort("arm", "ref", "group", "slice")  # fmt: skip
    lines = [
        "# Experiment 516 interim numbers (interim and unaudited; no code review, no write-up review)",
        "",
        "Points of CRPS, arm minus reference (negative favours the arm). Interval: fold-block bootstrap, Bonferroni level 1/(2 x 8). `negative_folds` of `folds`.",
        "",
        "## Minimum detectable effect (3-seed mean)",
        "",
        str(mde),
        "",
        "## Headline contrasts",
        "",
        str(
            head.select(
                "arm", "ref", "group", "slice", "folds", "contrast", "lo", "hi", "negative_folds"
            )
        ),
        "",
        "## Caveats",
        "",
        "- Fits and scoring only: no code review of the fit script, no results review, no write-up.",
        "- The base `W0`, `W-T`, `W-S`, `W-X`, and `W-null` forecasts are experiment 515's (W0 is experiment 513's P0 for 2025-12, 2026-01, 2026-03).",
        "- Windows are built on the half-hourly step-grid series; beyond 144 h they carry no information finer than 6 h.",
        "- ERA5 windows train the pretrained weights; scoring windows are ensemble windows built as at serving.",
        "- Placebo rows with no donor keep their own window (counted in the fit log).",
        "- Event labels are incomplete, so 'rows with no labelled event' may hold unlabelled events.",
    ]
    (SCORE_DIR / "interim_numbers.md").write_text("\n".join(lines) + "\n")
    print("wrote", SCORE_DIR / "interim_numbers.md")


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rows", "stats", "interim"))
    args = parser.parse_args()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    {"rows": s515.rows_command, "stats": stats, "interim": interim}[args.command]()


if __name__ == "__main__":
    main()

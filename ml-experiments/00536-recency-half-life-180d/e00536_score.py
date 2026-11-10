# ruff: noqa: D103, E501
"""Build arms' pooled-seed predictions, the member ensembles, and the per-row scores of experiment 534.

Reuses experiment 511's scorer (via 470's) with this experiment's work folder. Stored controls and L-1d are
linked from 511. Ensembles: `ENS-3` = uniform quantile average of R-7, R-30, R-90 (R-90 = 511's L-1d);
`ENS-5` = all five members. Commands: `build`, `rows`.
"""

import argparse
from pathlib import Path

import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = ML_EXPERIMENTS_DATA_DIR / "00536-recency-half-life-180d" / "work"
E511 = k.WORK_DIR
FIT_ARMS = ("R-180",)
ENSEMBLES: dict[str, tuple[str, ...]] = {}
ARMS = (*FIT_ARMS, "R-90", *ENSEMBLES)
CONTROLS = ("K2-S1", "A-pool")
REFERENCES = ("manual_heuristic",)
MEMBER_ARMS = (*FIT_ARMS,)


def configure() -> None:
    heads470.WORK_DIR = WORK
    score470.MONTHS = build470.MONTHS
    score470.SCORE_DIR = WORK / "scores"
    score470.CANDIDATES = ARMS
    score470.CONTROLS = CONTROLS
    score470.DESCRIPTIVE = ()
    score470.SINGLES = ()
    score470.REFERENCES = REFERENCES
    score470.MAIN_ARMS = (*ARMS, *CONTROLS)
    score470.CELLS = {}

    def masks(*, wide: pl.DataFrame) -> dict[str, pl.Series]:
        have = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in score470.MAIN_ARMS])
        return {
            "main16": wide.select(have.alias("x"))["x"],
            "gam14": wide.select((have & (pl.col("group") != "report_only")).alias("x"))["x"],
            "ref14": wide.select((have & (pl.col("group") != "report_only")).alias("x"))["x"],
        }

    score470.masks = masks  # ty: ignore[invalid-assignment]


def link() -> None:
    (WORK / "predictions").mkdir(parents=True, exist_ok=True)
    for name, src in {**{a: a for a in (*CONTROLS, *REFERENCES)}, "R-90": "L-1d"}.items():
        target = WORK / "predictions" / name
        if not target.exists():
            target.symlink_to(E511 / "predictions" / src, target_is_directory=True)


def build() -> None:
    sites = build470.site_map()
    for month in build470.MONTHS:
        _, keys = build454.load_heads(month=month)
        rows = pl.DataFrame({key: keys[key] for key in build470.KEYS})
        for arm in FIT_ARMS:
            folder = WORK / "seed_preds" / arm
            frame = build470.pool_stored(folder=folder, month=month, sites=sites, seeds=(0, 1))
            joined = build470.on_rows(rows=rows, frame=frame)
            print(
                "wrote",
                arm,
                month,
                joined.height,
                "missing",
                int(joined["q50"].null_count()),
                flush=True,
            )
            build470.write_arm(arm=arm, month=month, cell=0, frame=joined)
        for name, members in ENSEMBLES.items():
            frames = [
                pl.read_parquet(build470.prediction_path(arm=m, month=month, cell=0))
                for m in members
            ]
            for f in frames[1:]:
                assert f.select(build470.KEYS).equals(frames[0].select(build470.KEYS)), (
                    "member rows differ"
                )
            mean = np.mean(
                [f.select(list(build470.LEVELS)).to_numpy().astype(np.float64) for f in frames],
                axis=0,
            )
            out = (
                frames[0]
                .select(build470.KEYS)
                .hstack(pl.DataFrame(mean, schema=list(build470.LEVELS), orient="row"))
            )
            build470.write_arm(arm=name, month=month, cell=0, frame=out)
            print("wrote", name, month, out.height, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "rows"))
    args = parser.parse_args()
    configure()
    link()
    {"build": build, "rows": score470.rows_command}[args.command]()
    _ = Path


if __name__ == "__main__":
    main()

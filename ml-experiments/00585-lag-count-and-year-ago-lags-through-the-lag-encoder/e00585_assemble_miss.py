"""Experiment 585 (post hoc): assemble the forced-missing passes (201) into prediction tables.

For arms Y1 and Y3, the inject folder holds the forecasts that the saved weights make with every
extra lag set to missing (pass 201). This script concatenates each month's windows into
`arms/<kind>miss/work/seed_preds/V3D5/<month>_s<seed>.parquet`, the layout `e00585_score.py` reads.
"""

import e00565_design6 as d6  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = (
    ML_EXPERIMENTS_DATA_DIR / "00585-lag-count-and-year-ago-lags-through-the-lag-encoder" / "arms"
)


def main() -> None:
    """Write the assembled tables."""
    for kind in ("y1", "y3"):
        for month in d6.MONTHS:
            for seed in (0, 1):
                parts = sorted(
                    (WORK / kind / "work" / "inject").glob(f"{month}_r201_d*_s{seed}.parquet")
                )
                assert len(parts) == 2, (kind, month, seed, parts)
                out = (
                    WORK
                    / f"{kind}miss"
                    / "work"
                    / "seed_preds"
                    / "V3D5"
                    / f"{month}_s{seed}.parquet"
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(out)
                print("wrote", out.name, kind)


if __name__ == "__main__":
    main()

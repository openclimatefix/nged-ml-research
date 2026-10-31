# ruff: noqa: E501
"""Score experiment 443: each arm against the skeleton, with the pre-registered decision rule.

The skeleton is experiment 430's `g0_onepass`, read from its stored predictions for the same seeds
as the arm. Scoring machinery (seed averaging, per-row quantile loss, farm-equal month means, the
month-block bootstrap at 99%) is experiment 430's `e00430_score`, which uses experiment 310's.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations:ml-experiments/00443-solar-network-hyperparameter-and-architecture-sweep \
        uv run --no-sync python ml-experiments/00443-*/e00443_score.py MODE [ARM ...]

Modes: `gate` (the new code's skeleton against the stored forecasts, seed 0), `stage1` (arms on
seeds 0 to 2), `search` (random-search arms on seeds 0 to 1, ranked), `confirm ARM ...` (arms on
seeds 3 to 5).
"""

import sys
from pathlib import Path

import e00430_score as s430  # ty: ignore[unresolved-import]
import e00443_common as common
import numpy as np
import polars as pl

QCOLS = s430.QCOLS


def load(*, root: Path, arm: str, seeds: tuple[int, ...], name: str | None = None) -> pl.DataFrame:
    """Return an arm's predictions for the seeds, averaged over seeds quantile by quantile."""
    files = sorted(f for s in seeds for f in (root / arm).glob(f"s{s}_*.parquet"))
    if not files:
        msg = f"no predictions for {arm} seeds {seeds}"
        raise FileNotFoundError(msg)
    if len(files) != 4 * len(seeds):
        msg = f"{arm}: {len(files)} job files, expected {4 * len(seeds)} (seeds {seeds} x 4 refits)"
        raise FileNotFoundError(msg)
    return s430.average_seeds(frame=pl.concat([pl.read_parquet(f) for f in files]), arm=name or arm)


def contrast(*, arm: str, seeds: tuple[int, ...]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the contrast table of the arm against the skeleton and both arms' absolute scores."""
    frames = {
        arm: load(root=common.PRED_DIR, arm=arm, seeds=seeds),
        common.SKELETON: load(root=common.E430_PRED_DIR, arm=common.SKELETON, seeds=seeds),
    }
    table = s430.contrast_table(arms=frames, pairs=[(arm, common.SKELETON)])
    return table, s430.absolute_table(arms=frames)


def verdict(*, row: dict, margin: float) -> dict[str, bool]:
    """Return which parts of the pre-registered rule a contrast row (all bands) passes."""
    return {
        "beats_margin": row["pct_change"] <= -margin,
        "interval_excludes_zero": row["hi99"] < 0.0,
        "tails_ok": row["pin01_diff"] <= common.TAIL_HARM_LIMIT
        and row["pin99_diff"] <= common.TAIL_HARM_LIMIT,
    }


def summarise(*, arms: list[str], seeds: tuple[int, ...], out: str) -> pl.DataFrame:
    """Score each arm against the skeleton and write one summary row per arm (all bands)."""
    common.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    bands = []
    for arm in arms:
        table, absolute = contrast(arm=arm, seeds=seeds)
        bands.append(table)
        allrow = table.filter(pl.col("band") == "all").to_dicts()[0]
        ab = absolute.filter(pl.col("arm") == arm).to_dicts()[0]
        flags = verdict(row=allrow, margin=common.MARGIN_PERCENT)
        rows.append(
            {
                "arm": arm,
                "score": allrow["score_arm"],
                "skeleton_score": allrow["score_ref"],
                "rows": allrow["rows"],
                "pct_change": allrow["pct_change"],
                "lo99": allrow["lo99"],
                "hi99": allrow["hi99"],
                "pin01_diff": allrow["pin01_diff"],
                "pin99_diff": allrow["pin99_diff"],
                "cover01": ab["cover01"],
                "cover99": ab["cover99"],
                "median_minus_mean": ab["median_minus_mean"],
                **flags,
                "passes": all(flags.values()),
            }
        )
    summary = pl.DataFrame(rows)
    summary.write_csv(common.SCORE_DIR / f"{out}.csv")
    pl.concat(bands).write_csv(common.SCORE_DIR / f"{out}_bands.csv")
    return summary


def gate() -> None:
    """Compare the new code's skeleton fit (seed 0) with experiment 430's stored forecasts."""
    keys = s430.KEYS
    worst = 0.0
    for start in ("2025-09", "2025-12", "2026-03", "2026-06"):
        new = pl.read_parquet(common.PRED_DIR / "skeleton" / f"s0_{start}.parquet")
        old = pl.read_parquet(common.E430_PRED_DIR / common.SKELETON / f"s0_{start}.parquet")
        joined = new.join(old, on=keys, suffix="_old")
        assert joined.height == new.height == old.height, (joined.height, new.height, old.height)
        diff = max(float(np.abs((joined[c] - joined[f"{c}_old"]).to_numpy()).max()) for c in QCOLS)
        print(f"{start}: rows {joined.height}, max abs quantile difference {diff:.3e}")
        worst = max(worst, diff)
    print("GATE", "PASSED" if worst <= 1e-4 else "MISSED", f"(worst {worst:.3e}, tolerance 1e-4)")


def main() -> None:
    """Run the named mode."""
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    mode, names = sys.argv[1], sys.argv[2:]
    if mode == "gate":
        gate()
    elif mode == "stage1":
        arms = names or list(common.STAGE1_ARMS)
        print(summarise(arms=arms, seeds=common.SEEDS_SCREEN, out="stage1"))
        print(
            f"seed minimum detectable effect and margin: see plan; margin {common.MARGIN_PERCENT}%"
        )
    elif mode == "search":
        arms = names or sorted(p.name for p in common.PRED_DIR.glob("rs_*"))
        summary = summarise(arms=arms, seeds=common.SEEDS_SEARCH, out="search").sort("score")
        print(summary)
    elif mode == "confirm":
        print(summarise(arms=names, seeds=common.SEEDS_CONFIRM, out="confirm"))
    else:
        msg = f"unknown mode {mode}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()

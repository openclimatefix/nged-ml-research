# ruff: noqa: E501
"""Join the stored worker forecasts of every Primary into one frame per Primary (experiment 375).

Workers are BG2's five members and BG2, rebuilt from experiment 87's stored refit predictions with
experiment 360's code (no fits), and the GAM stack read from experiments 252, 155, and 355. Every
quantile is in points of the series' effective capacity. Writes `work/workers_<site>.parquet`.

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00375-*/e00375_build.py
"""

import dataclasses
from pathlib import Path
from typing import Final

import e00087_assemble as asm  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final = "00375-meta-model-over-worker-forecasts-at-all-primaries"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
STACK_252: Final = ML_EXPERIMENTS_DATA_DIR / "00252-gam-stack-at-eight-more-primaries/work"
STACK_155: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction/work/dp_preds"
)
D8_ROWS: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00355-d8-weekly-refit-and-median-band-swap/work/scores_wide.parquet"
)
LEVELS_HEAD: Final = common.LEVEL_COLUMNS
MEMBERS: Final = asm.BLEND_FIVE
KEY: Final = ["power_fcst_init_time", "valid_time"]


def stack_file(site: str) -> Path | None:
    """Return the stored GAM-stack file of a Primary, or None where no stack was stored."""
    for path in (
        STACK_252 / f"mva/arms/a1/dp_preds/a2_scored_{site}.parquet",
        STACK_252 / f"control/arms/a0/dp_preds/a2_scored_{site}.parquet",
        STACK_155 / f"d_scored_{site}.parquet",
    ):
        if path.exists():
            return path
    return None


def stack_frame(site: str) -> pl.DataFrame | None:
    """Return the stack's quantiles (points), the reading, and the event mask at a Primary."""
    path = stack_file(site)
    if site == "D8":
        wide = pl.read_parquet(D8_ROWS)
        return wide.select(
            *KEY,
            *[pl.col(f"W__{q}").alias(f"gam__{q}") for q in LEVELS_HEAD],
            pl.col("actual").alias("actual_stack"),
            pl.lit(False).alias("switching_mask"),
        )
    if path is None:
        return None
    frame = pl.read_parquet(path)
    return frame.select(
        *KEY,
        *[pl.col(q).alias(f"gam__{q}") for q in LEVELS_HEAD],
        pl.col("actual").alias("actual_stack"),
        "switching_mask",
    )


def main() -> None:
    """Rebuild the members and BG2 at every Primary, join the stack, and write one frame per site."""
    WORK.mkdir(parents=True, exist_ok=True)
    ctx = dataclasses.replace(
        asm.make_context(smoke=False, store=False), results_dir=WORK / "weights"
    )
    primary = ctx.sites_of(groups=("Primary",))
    tables = {}
    for member in MEMBERS:
        raw = asm.raw_member(ctx=ctx, member=member, offset="vintage", sites=primary)
        tables[member] = asm.finish_member(ctx=ctx, raw=raw, correct=member == "ST", spread=True)
        print(f"member {member}: {tables[member].height:,} rows", flush=True)
    keys, by_member = asm.align(tables=tables, strict=True)
    del tables
    actual = asm.actual_array(frame=keys, power_of=ctx.power_of)
    bg2, _ = asm.rolling_blend(
        keys=keys, actual=actual, members=[by_member[m] for m in MEMBERS], names=list(MEMBERS)
    )
    ids = keys["time_series_id"].to_numpy()
    site_of = {i: ctx.labels[int(i)][0] for i in np.unique(ids)}
    site_column = np.array([site_of[i] for i in ids])
    summary = {}
    for site in sorted(set(site_column), key=lambda s: int(s[1:])):
        mask = site_column == site
        base = keys.filter(pl.Series(mask)).with_columns(
            actual=pl.Series(actual[mask]) * 100.0,
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            ),
        )
        columns = {f"bg2_{c}": bg2[mask][:, k] * 100.0 for k, c in enumerate(LEVELS_HEAD)}
        for m in MEMBERS:
            columns |= {
                f"only_{m}__{c}": by_member[m][mask][:, k] * 100.0
                for k, c in enumerate(LEVELS_HEAD)
            }
        frame = base.hstack(pl.DataFrame(columns))
        stack = stack_frame(site)
        if stack is not None:
            frame = frame.join(stack, on=KEY, how="inner", validate="1:1")
            frame = frame.with_columns(actual=pl.col("actual_stack")).drop("actual_stack")
        else:
            frame = frame.with_columns(switching_mask=pl.lit(False))
        frame = frame.filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") > 0.0)
            & (pl.col("lead_hours") <= 168.0)
            & (pl.col("valid_time") < common.HOLDOUT)
        ).sort(KEY)
        frame.write_parquet(WORK / f"workers_{site}.parquet")
        summary[site] = (frame.height, stack is not None)
        print(site, frame.height, "stack" if stack is not None else "NO STACK", flush=True)


if __name__ == "__main__":
    main()

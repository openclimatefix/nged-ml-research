# ruff: noqa: E501
"""Experiment 701: assemble BG2 and its monthly-only variant for the origins 2025-07-01 to 2026-10-31.

The assembly is experiment 87's (`e00087_assemble.py`), unchanged, run through the patch of
`e00701_common`. It covers the 365 stored runs and the 123 new runs together, because the monthly
blend weights of a month read the earlier cases and the 14-day correction of the frozen member
reads the 14 days before each run. The weights of the stored months are the same as experiment 87's
(a month's weights use only cases with an earlier valid time), and the script checks that the
stored runs' arms are reproduced.

Two arms are built, both Primary-only:

- `p_s6_bg2`: BG2, the blend of the weekly, monthly, analogue, pooled, and frozen members, each at
  its offset twin at the five offset Primaries (needs the `n1h` weekly files).
- `p_a3_monthly_in_weekly_slot`: the same blend with the monthly member in the weekly member's slot
  (needs no `n1h` weekly file).

Subcommands:

- `prepare` fills `work/assemble/anchor_preds` with links to experiment 87's stored refit files and
  this experiment's new files. The frozen member's file is one file per series, so the new runs'
  rows are concatenated onto the stored rows in a new file (no key may overlap).
- `build [--arms ...] [--stored-only]` assembles the arms into `work/assemble/arms` and the
  weights into `work/assemble/results`, then compares the stored runs with experiment 87's stored
  arms. `--stored-only` assembles the 365 stored runs alone, into `arms_stored_check`, which proves
  that the patched assembly reproduces experiment 87's arms before any new fit exists.

    PYTHONPATH=<folders of experiments 87, 33, 48, 701> \
        uv run --no-sync python ml-experiments/00701-*/e00701_assemble.py prepare
    ... build --arms p_a3_monthly_in_weekly_slot
"""

import argparse
import time
from datetime import timedelta

import e00087_assemble as assemble  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import e00701_common as mine
import numpy as np
import polars as pl

ARMS = ("p_a3_monthly_in_weekly_slot", "p_s6_bg2")
FROZEN_CONFIGS = ("frozen", "frozen_o", "frozen_oheld")


def all_origins(*, stored_only: bool) -> list:
    """Return the origins of the 365 stored runs, and of the 123 new runs too unless `stored_only`."""
    runs = []
    spans = [(mine.STORED_FIRST_RUN, mine.STORED_LAST_RUN)]
    if not stored_only:
        spans.append((mine.NEW_FIRST_RUN, mine.NEW_LAST_RUN))
    for first, last in spans:
        runs += [
            common.origin_of(run=first + timedelta(days=d)) for d in range((last - first).days + 1)
        ]
    return runs


def prepare() -> None:
    """Fill `work/assemble/anchor_preds` with the stored refit files and the new ones."""
    target_root = mine.ASSEMBLE_DIR / "anchor_preds"
    stored_root = mine.E87_WORK / "anchor_preds"
    merged = linked = 0
    for config_dir in sorted(p for p in stored_root.iterdir() if p.is_dir()):
        config = config_dir.name
        out = target_root / config
        out.mkdir(parents=True, exist_ok=True)
        new_files = (
            {p.name: p for p in (mine.ANCHOR_DIR / config).glob("*.parquet")}
            if (mine.ANCHOR_DIR / config).exists()
            else {}
        )
        for path in config_dir.glob("*.parquet"):
            link = out / path.name
            if path.name in new_files:
                if link.exists() or link.is_symlink():
                    link.unlink()
                old, new = pl.read_parquet(path), pl.read_parquet(new_files[path.name])
                if config not in FROZEN_CONFIGS:
                    msg = (
                        f"{config}: the new run files share the name {path.name} with a stored file"
                    )
                    raise ValueError(msg)
                overlap = old.join(
                    new.select(*common.KEYS), on=list(common.KEYS), how="inner"
                ).height
                if overlap:
                    msg = f"{path.name}: {overlap} rows are in both the stored and the new file"
                    raise ValueError(msg)
                common.atomic_write(
                    frame=pl.concat([old, new.select(old.columns).cast(old.schema)]), target=link
                )
                merged += 1
            elif not link.exists() and not link.is_symlink():
                link.symlink_to(path)
                linked += 1
        for name, path in new_files.items():
            link = out / name
            if (
                name not in {p.name for p in config_dir.glob("*.parquet")}
                and not link.exists()
                and not link.is_symlink()
            ):
                link.symlink_to(path)
                linked += 1
    print(f"prepare: {linked} files linked, {merged} frozen files merged under {target_root}")


def make_context(*, stored_only: bool) -> assemble.Context:
    """Return experiment 87's assembly context for the Primary arms, with no generator inputs."""
    series = common.series_table()
    arrays = common.load_power_arrays(series=series)
    rows = list(series.iter_rows(named=True))
    ids = [int(r["time_series_id"]) for r in rows]
    return assemble.Context(
        series=series,
        sites=pl.DataFrame(),
        arrays=arrays,
        power_of={i: arrays[r["site"]].power for i, r in zip(ids, rows, strict=True)},
        labels={i: (r["site"], r["substation_type"]) for i, r in zip(ids, rows, strict=True)},
        capacity={i: float(r["effective_capacity_mw"]) for i, r in zip(ids, rows, strict=True)},
        origins=all_origins(stored_only=stored_only),
        demand_root=mine.ASSEMBLE_DIR / "anchor_preds",
        generator_root=mine.ASSEMBLE_DIR / "generator_preds_unused",
        arms_dir=mine.ASSEMBLE_DIR / ("arms_stored_check" if stored_only else "arms"),
        results_dir=mine.ASSEMBLE_DIR / ("results_stored_check" if stored_only else "results"),
        strict=True,
        store=False,
    )


def compare_with_stored(*, arm: str, arms_dir: str) -> None:
    """Compare the stored runs of an assembled arm with experiment 87's stored arm (all 11 levels)."""
    new = pl.read_parquet(mine.ASSEMBLE_DIR / arms_dir / f"{arm}.parquet").filter(
        pl.col("power_fcst_init_time") <= common.origin_of(run=mine.STORED_LAST_RUN)
    )
    old = pl.read_parquet(mine.E87_WORK / "arms" / f"{arm}.parquet")
    keys = list(assemble.KEY_COLUMNS)
    joined = new.join(old, on=keys, suffix="_old", how="full", coalesce=True)
    if joined.height != new.height or joined.height != old.height:
        msg = f"{arm}: new {new.height:,} rows, stored {old.height:,}, joined {joined.height:,}"
        raise SystemExit(msg)
    gap = np.abs(
        joined.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
        - joined.select([f"{c}_old" for c in common.LEVEL_COLUMNS]).to_numpy().astype(np.float64)
    )
    actual_gap = float(
        np.nanmax(np.abs(joined["actual"].to_numpy() - joined["actual_old"].to_numpy()))
    )
    print(
        f"{arm} against experiment 87's stored arm: {joined.height:,} rows on the 365 stored runs; "
        f"largest quantile difference {np.nanmax(gap):.3g}, mean {np.nanmean(gap):.3g} (fractions of capacity); "
        f"largest difference of the observed value {actual_gap:.3g}"
    )


def build(*, arms: list[str], stored_only: bool) -> None:
    """Assemble the named Primary arms over the 488 runs and write them."""
    ctx = make_context(stored_only=stored_only)
    plan = {
        "p_a3_monthly_in_weekly_slot": lambda: assemble.blend_arm(
            ctx=ctx,
            slots=assemble.BLEND_MONTHLY_IN_WEEKLY_SLOT,
            offset="vintage",
            weights_name="a3",
        ),
        "p_s6_bg2": lambda: assemble.blend_arm(
            ctx=ctx, slots=assemble.BLEND_FIVE, offset="vintage", weights_name="s6"
        ),
    }
    for name in arms:
        started = time.monotonic()
        table = plan[name]()
        assemble.emit(ctx=ctx, name=name, table=table, started=started)
        compare_with_stored(arm=name, arms_dir="arms_stored_check" if stored_only else "arms")


def main() -> None:
    """Run one subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "build"))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--stored-only",
        action="store_true",
        help="assemble only the 365 stored runs (the reproduction check; needs no new fit)",
    )
    args = parser.parse_args()
    mine.ASSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare()
    else:
        build(arms=args.arms, stored_only=args.stored_only)


if __name__ == "__main__":
    main()

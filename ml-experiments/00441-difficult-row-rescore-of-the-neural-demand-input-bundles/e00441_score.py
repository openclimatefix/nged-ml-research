# ruff: noqa: E501
r"""Rescore experiment 437's stored forecasts on difficult rows (experiment 441). No model is fitted.

Two stages. `pre` reads only the controls (`R` and `A`, seeds 0 to 2 and the second seed set 3 and
4) and prints, before any arm is scored: the row and day counts of every slice per fold and per
metering group, and the minimum detectable effect of every slice. `score` then scores the eight
bundle arms of experiment 437 against their controls on the slices, with the decision rule that
`result.md` registered.

The forecasts are those of `e00437_score.load_frames` (seeds averaged in quantile space, widened by
1.15, scored on the half-hours shared with the references), so a contrast over all rows reproduces
experiment 437's. The slice masks come from `e00441_masks.py`.

```bash
source <(printf 'export PYTHONPATH=%s\n' "$(cat ml-experiments/00441-*/pythonpath.txt)")
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data OMP_NUM_THREADS=1
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00441-*/e00441_score.py pre
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00425_score as sc  # ty: ignore[unresolved-import]
import e00437_fit as fit  # ty: ignore[unresolved-import]
import e00437_score as s437  # ty: ignore[unresolved-import]
import e00441_masks as masks_module  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

HEADLINE: Final[str] = sc.HEADLINE
SUBSETS: Final[dict[str, tuple[str, ...]]] = sc.SUBSETS
SIX: Final[str] = sc.SIX
EIGHT: Final[str] = sc.EIGHT
REPORT_ONLY: Final[str] = sc.REPORT_ONLY_GROUP
GROUPS: Final[tuple[str, ...]] = (SIX, EIGHT, REPORT_ONLY)
SCREEN_ARMS: Final[tuple[str, ...]] = s437.SCREEN_ARMS
SLICES: Final[dict[str, pl.Expr]] = {
    "cold winter nights": pl.col("cold_winter_night"),
    "cold sunny days": pl.col("cold_sunny_day"),
    "cold overcast days": pl.col("cold_overcast_day"),
    "Christmas": pl.col("christmas"),
    "other bank holidays": pl.col("other_bank_holiday"),
    "heat-wave days": pl.col("heat_wave_day"),
}
MIN_DAYS_FOR_DECISION: Final[int] = 20
"""A slice with fewer distinct local dates than this is descriptive only."""
BLOCK_DAYS: Final[int] = 7
"""Length in days of the valid-date blocks of the block bootstrap and of the seed-spread estimate."""
MIN_MONTH_BLOCKS: Final[int] = 3
"""The month-block bootstrap needs at least this many origin months holding rows of the slice."""
ROWS_DIR: Final[Path] = masks_module.WORK_DIR / "rows"


def attach_masks(*, rows: pl.DataFrame, masks: pl.DataFrame) -> pl.DataFrame:
    """Return the scored rows with the slice masks, the block labels, and the fold joined on.

    A scored row without a mask row (a lead outside the headline band) gets every mask False.

    Args:
        rows: One arm's `scored_rows`.
        masks: The table written by `e00441_masks.py`.

    Returns:
        The rows with the `MASK_NAMES` columns, `fold`, `local_date`, and `block` (a 7-day label).
    """
    joined = rows.join(
        masks.select(
            "label", "power_fcst_init_time", "valid_time", "local_date", *masks_module.MASK_NAMES
        ),
        on=["label", "power_fcst_init_time", "valid_time"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    first_day = joined["local_date"].min()
    return joined.with_columns(
        *[pl.col(m).fill_null(False) for m in masks_module.MASK_NAMES],
        fold=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        block=((pl.col("local_date") - first_day).dt.total_days() // BLOCK_DAYS),
    )


def build_rows(*, arms: list[str]) -> dict[str, pl.DataFrame]:
    """Return every arm's and reference's scored rows with the masks attached.

    Args:
        arms: The bundle arms to load besides the controls and the references.

    Returns:
        The rows keyed by the names `e00437_score.load_frames` gives (`R_w115`, `T_w115`, ...).
    """
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    frames = s437.load_frames(series=series, months=fit.SCREEN_MONTHS, arms=arms)
    table, _scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    print(f"shared half-hours across every arm and reference: {table.height}")
    masks = pl.read_parquet(masks_module.MASK_PATH)
    return {name: attach_masks(rows=frame, masks=masks) for name, frame in rows.items()}


def slice_filter(*, mask: pl.Expr, group: str | None = None) -> pl.Expr:
    """Return the filter of a slice inside the headline band, optionally for one metering group."""
    keep = pl.col(HEADLINE) & mask
    if group is not None:
        keep &= pl.col("label").is_in(list(SUBSETS[group]))
    return keep


def counts(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Print and return per slice, group, and fold the scored rows and distinct local dates.

    Args:
        rows: One arm's rows with the masks attached (every arm has the same rows).

    Returns:
        Columns `slice`, `group`, `fold`, `rows`, `days`, `primaries`.
    """
    out: list[pl.DataFrame] = []
    for name, mask in SLICES.items():
        for group in (None, *GROUPS):
            keep = slice_filter(mask=mask, group=group)
            sliced = rows.filter(keep)
            for fold in (None, *fit.SCREEN_MONTHS):
                part = sliced if fold is None else sliced.filter(pl.col("fold") == fold)
                out.append(
                    pl.DataFrame(
                        {
                            "slice": name,
                            "group": group or "all 16 Primaries",
                            "fold": fold or "all folds",
                            "rows": part.height,
                            "days": part["local_date"].n_unique() if part.height else 0,
                            "primaries": part["label"].n_unique() if part.height else 0,
                        }
                    )
                )
    table = pl.concat(out)
    with pl.Config(tbl_rows=200, tbl_cols=10, fmt_str_lengths=40):
        print(table)
    return table


def seed_spread(
    *, rows: dict[str, pl.DataFrame], control: str, mask: pl.Expr, group: str
) -> dict[str, float | int | None]:
    """Return the minimum detectable effect of a slice from the control's two seed sets.

    The estimate is the standard deviation, over 7-day valid-date blocks holding rows of the slice,
    of the difference in mean CRPS between the control's seeds 0 to 2 and its seeds 3 and 4, divided
    by the square root of two (experiment 437's convention; a lower bound). The same figure over the
    folds is returned when at least three folds hold rows.

    Args:
        rows: Scored rows keyed by name, holding `{control}_s012_w115` and `{control}_s34_w115`.
        control: `R` or `A`.
        mask: The slice's mask expression.
        group: The metering group.

    Returns:
        `mde_blocks`, `n_blocks`, `mde_folds`, `n_folds`, and `mean_difference` (the mean seed-set
        difference over the slice's rows), each None when the slice has too few blocks or folds.
    """
    keep = slice_filter(mask=mask, group=group)
    first = (
        rows[f"{control}_s012_w115"]
        .filter(keep)
        .select(*sc.KEYS, "fold", "block", first=pl.col("crps"))
    )
    second = rows[f"{control}_s34_w115"].select(*sc.KEYS, second=pl.col("crps"))
    joined = first.join(second, on=sc.KEYS, how="inner", validate="1:1").with_columns(
        gap=pl.col("first") - pl.col("second")
    )
    out: dict[str, float | int | None] = {
        "mde_blocks": None,
        "n_blocks": 0,
        "mde_folds": None,
        "n_folds": 0,
        "mean_difference": float(joined["gap"].mean()) if joined.height else None,
    }
    by_block = joined.group_by("block").agg(pl.col("gap").mean())
    out["n_blocks"] = by_block.height
    if by_block.height >= 4:
        out["mde_blocks"] = float(np.std(by_block["gap"].to_numpy(), ddof=1) / np.sqrt(2.0))
    by_fold = joined.group_by("fold").agg(pl.col("gap").mean())
    out["n_folds"] = by_fold.height
    if by_fold.height >= MIN_MONTH_BLOCKS:
        out["mde_folds"] = float(np.std(by_fold["gap"].to_numpy(), ddof=1) / np.sqrt(2.0))
    return out


def pre() -> None:
    """Print the slice counts and the minimum detectable effects from the controls alone."""
    rows = build_rows(arms=[])
    print(
        "\n== row and day counts (scored rows in days 1 to 10; days are distinct local valid dates) =="
    )
    counts(rows=rows["R_w115"])
    print(
        "\n== minimum detectable effect per slice from the control seed sets "
        "(points of capacity; a lower bound; None = too few blocks or folds) =="
    )
    for name, mask in SLICES.items():
        for control in s437.CONTROLS:
            for group in (SIX, EIGHT):
                figures = seed_spread(rows=rows, control=control, mask=mask, group=group)
                shown = {k: (None if v is None else round(float(v), 4)) for k, v in figures.items()}
                print(f"{name:22s} control {control} {group:15s} {shown}")


def main() -> None:
    """Run the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pre",))
    args = parser.parse_args()
    if args.stage == "pre":
        pre()


if __name__ == "__main__":
    main()

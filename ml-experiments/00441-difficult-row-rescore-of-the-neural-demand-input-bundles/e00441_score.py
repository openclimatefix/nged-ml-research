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
from typing import Final, cast

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
    joined = rows.drop([m for m in masks_module.MASK_NAMES if m in rows.columns]).join(
        masks.select(
            "label", "power_fcst_init_time", "valid_time", "local_date", *masks_module.MASK_NAMES
        ),
        on=["label", "power_fcst_init_time", "valid_time"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    unmasked = joined.filter(pl.col(HEADLINE) & pl.col("local_date").is_null()).height
    if unmasked:
        msg = f"{unmasked} headline-band rows have no mask row"
        raise ValueError(msg)
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
        "mean_difference": float(cast("float", joined["gap"].mean())) if joined.height else None,
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


# --- stage `score` -------------------------------------------------------------------------

N_RESAMPLES: Final[int] = 20_000
BOOTSTRAP_SEED: Final[int] = 441
N_PRIMARY_TESTS: Final[int] = 10
"""T, T_pre, F, and F_pre on cold winter nights at two groups; S and S_pre on cold sunny days at the six."""
LEVEL_PERCENT: Final[float] = 100.0 - 5.0 / N_PRIMARY_TESTS
MIN_GAIN: Final[float] = 0.05
MAX_OUTSIDE_HARM: Final[float] = 0.02
MAX_TAIL_HARM: Final[float] = 0.02
MIN_FOLDS_NEGATIVE: Final[int] = 3
PRIMARY_CELLS: Final[dict[str, tuple[tuple[str, tuple[str, ...]], ...]]] = {
    "T": (("cold winter nights", (SIX, EIGHT)),),
    "T_pre": (("cold winter nights", (SIX, EIGHT)),),
    "F": (("cold winter nights", (SIX, EIGHT)),),
    "F_pre": (("cold winter nights", (SIX, EIGHT)),),
    "S": (("cold sunny days", (SIX,)),),
    "S_pre": (("cold sunny days", (SIX,)),),
}
"""The tested arm-slice-group cells; every other cell is descriptive."""
DELTA_COLUMNS: Final[tuple[str, ...]] = (
    "crps",
    "mae_median",
    "pinball_q01",
    "pinball_q05",
    "pinball_q95",
    "pinball_q99",
)
FIGURE_TABLE_DIR: Final[Path] = masks_module.WORK_DIR / "figure_tables"
ALL_ROWS: Final[str] = "all rows"


def paired(*, rows: dict[str, pl.DataFrame], arm: str) -> pl.DataFrame:
    """Return the arm's rows with, for each metric, its difference from the control's row.

    Args:
        rows: Scored rows keyed by name, holding `{arm}_w115` and its control's.
        arm: A bundle arm.

    Returns:
        One row per shared scored half-hour with `d_<metric>` columns (arm minus control).

    Raises:
        ValueError: If the arm and its control do not hold the same rows.
    """
    treatment = rows[f"{arm}_w115"]
    reference = rows[f"{s437.control_of(arm=arm)}_w115"]
    if treatment.height != reference.height:
        msg = (
            f"{arm} and its control differ in height: {treatment.height} against {reference.height}"
        )
        raise ValueError(msg)
    joined = treatment.select(
        *sc.KEYS,
        "label",
        "fold",
        "block",
        "local_date",
        HEADLINE,
        *masks_module.MASK_NAMES,
        *DELTA_COLUMNS,
    ).join(
        reference.select(*sc.KEYS, *[pl.col(c).alias(f"{c}_control") for c in DELTA_COLUMNS]),
        on=sc.KEYS,
        how="inner",
        validate="1:1",
    )
    if joined.height != treatment.height:
        msg = f"{arm}: only {joined.height} of {treatment.height} rows match the control's"
        raise ValueError(msg)
    return joined.with_columns(
        *[(pl.col(c) - pl.col(f"{c}_control")).alias(f"d_{c}") for c in DELTA_COLUMNS]
    ).drop(*DELTA_COLUMNS, *[f"{c}_control" for c in DELTA_COLUMNS])


def block_interval(
    *, frame: pl.DataFrame, column: str, rng: np.random.Generator
) -> dict[str, object]:
    """Return the mean of a column and its block-bootstrap interval at `LEVEL_PERCENT`.

    The blocks are the origin months (the fold) when at least `MIN_MONTH_BLOCKS` of them hold rows,
    and 7-day blocks of valid dates otherwise. Each resample draws blocks with replacement and takes
    the ratio of the summed values to the summed row counts.

    Args:
        frame: Rows with `fold`, `block`, and the column.
        column: The column to average.
        rng: The random generator, seeded once per run.

    Returns:
        `mean`, `lower`, `upper`, `rows`, `days`, `blocks`, and `block_kind`; the mean and bounds
        are None when the frame is empty.
    """
    if frame.is_empty():
        return {
            "mean": None,
            "lower": None,
            "upper": None,
            "rows": 0,
            "days": 0,
            "blocks": 0,
            "block_kind": "none",
        }
    kind = "month" if frame["fold"].n_unique() >= MIN_MONTH_BLOCKS else "7-day"
    key = "fold" if kind == "month" else "block"
    grouped = frame.group_by(key).agg(total=pl.col(column).sum(), count=pl.len())
    totals = grouped["total"].to_numpy().astype(np.float64)
    counts_ = grouped["count"].to_numpy().astype(np.float64)
    draws = rng.integers(0, grouped.height, size=(N_RESAMPLES, grouped.height))
    estimates = totals[draws].sum(axis=1) / counts_[draws].sum(axis=1)
    tail = (100.0 - LEVEL_PERCENT) / 2.0
    lower, upper = np.percentile(estimates, [tail, 100.0 - tail])
    return {
        "mean": float(totals.sum() / counts_.sum()),
        "lower": float(lower),
        "upper": float(upper),
        "rows": frame.height,
        "days": frame["local_date"].n_unique(),
        "blocks": grouped.height,
        "block_kind": kind,
    }


def cell_filter(*, mask: pl.Expr | None, group: str, outside: bool) -> pl.Expr:
    """Return the filter of a slice (or its complement) inside the headline band for a group."""
    keep = pl.col(HEADLINE) & pl.col("label").is_in(list(SUBSETS[group]))
    if mask is None:
        return keep
    return keep & ~mask if outside else keep & mask


def contrast_table(*, paired_by_arm: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return every arm's contrasts on all rows, each slice, and each slice's complement.

    Args:
        paired_by_arm: The output of `paired` for each arm.

    Returns:
        One row per arm, group, slice, and side (`in`, `outside`, or `all`) with the CRPS
        difference and its interval, the fold differences, and the point differences of the median
        error and of the pinball loss at q01, q05, q95, and q99.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: list[dict[str, object]] = []
    for arm, frame in paired_by_arm.items():
        for group in GROUPS:
            cells: list[tuple[str, str, pl.Expr | None]] = [(ALL_ROWS, "all", None)]
            for name, mask in SLICES.items():
                cells.append((name, "in", mask))
                cells.append((name, "outside", mask))
            for name, side, mask in cells:
                part = frame.filter(cell_filter(mask=mask, group=group, outside=side == "outside"))
                interval = block_interval(frame=part, column="d_crps", rng=rng)
                by_fold = (
                    part.group_by("fold").agg(pl.col("d_crps").mean()).sort("fold")
                    if part.height
                    else pl.DataFrame({"fold": [], "d_crps": []})
                )
                out.append(
                    {
                        "arm": arm,
                        "control": s437.control_of(arm=arm),
                        "group": group,
                        "slice": name,
                        "side": side,
                        **interval,
                        "fold_differences": {f: round(float(v), 4) for f, v in by_fold.iter_rows()},
                        "n_negative_folds": int((by_fold["d_crps"] < 0).sum())
                        if part.height
                        else 0,
                        "n_folds": by_fold.height,
                        **{
                            f"d_{c}": (
                                float(cast("float", part[f"d_{c}"].mean())) if part.height else None
                            )
                            for c in DELTA_COLUMNS
                            if c != "crps"
                        },
                    }
                )
    return pl.DataFrame(out, infer_schema_length=None)


def decide(*, table: pl.DataFrame) -> pl.DataFrame:
    """Apply the registered decision rule to the tested cells and print each reading.

    Args:
        table: The output of `contrast_table`.

    Returns:
        One row per tested cell with the five conditions and the reading.
    """
    out: list[dict[str, object]] = []
    for arm, cells in PRIMARY_CELLS.items():
        for slice_name, groups in cells:
            for group in groups:
                pick = table.filter(
                    (pl.col("arm") == arm)
                    & (pl.col("group") == group)
                    & (pl.col("slice") == slice_name)
                )
                inside = pick.filter(pl.col("side") == "in").row(0, named=True)
                outside = pick.filter(pl.col("side") == "outside").row(0, named=True)
                decidable = (
                    inside["days"] >= MIN_DAYS_FOR_DECISION and inside["block_kind"] == "month"
                )
                folds_ok = inside["n_negative_folds"] >= MIN_FOLDS_NEGATIVE or (
                    inside["n_folds"] > 0 and inside["n_negative_folds"] == inside["n_folds"]
                )
                tail_ok = all(
                    row[f"d_pinball_{level}"] <= MAX_TAIL_HARM
                    for row in (inside, outside)
                    for level in ("q95", "q99")
                )
                conditions = {
                    "gain_at_least_0.05": inside["mean"] is not None
                    and inside["mean"] <= -MIN_GAIN,
                    "interval_below_zero": inside["upper"] is not None and inside["upper"] < 0.0,
                    "folds_negative": folds_ok,
                    "outside_not_worse": outside["upper"] is not None
                    and outside["upper"] <= MAX_OUTSIDE_HARM,
                    "tails_ok": tail_ok,
                }
                reading = (
                    "ADVANCES"
                    if decidable and all(conditions.values())
                    else ("does not advance" if decidable else "descriptive only")
                )
                out.append(
                    {
                        "arm": arm,
                        "slice": slice_name,
                        "group": group,
                        "decidable": decidable,
                        **conditions,
                        "reading": reading,
                    }
                )
    table_out = pl.DataFrame(out)
    with pl.Config(tbl_rows=60, tbl_cols=12, fmt_str_lengths=30):
        print(table_out)
    return table_out


def absolute_table(*, rows: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the mean CRPS, coverage, and pinball loss of every forecast on every slice and group.

    Args:
        rows: Scored rows keyed by name.

    Returns:
        One row per forecast, group, and slice (including all rows).
    """
    metrics_ = ("crps", "mae_median", "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99",
                "pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99")  # fmt: skip
    out: list[pl.DataFrame] = []
    for name, frame in rows.items():
        if "_s012" in name or "_s34" in name:
            continue
        for group in GROUPS:
            for slice_name, mask in {ALL_ROWS: None, **SLICES}.items():
                keep = cell_filter(mask=mask, group=group, outside=False)
                part = frame.filter(keep)
                if part.is_empty():
                    continue
                per_primary = part.group_by("label").agg(
                    *[pl.col(m).mean() for m in metrics_], rows=pl.len()
                )
                out.append(
                    per_primary.select(
                        *[pl.col(m).mean() for m in metrics_], rows=pl.col("rows").sum()
                    ).with_columns(
                        forecast=pl.lit(name.removesuffix("_w115")),
                        group=pl.lit(group),
                        slice=pl.lit(slice_name),
                        days=pl.lit(part["local_date"].n_unique()),
                    )
                )
    table = pl.concat(out)
    with pl.Config(tbl_rows=400, tbl_cols=16, fmt_str_lengths=24):
        print(table.select("slice", "group", "forecast", "rows", "days", "crps", "mae_median",
                           "coverage_q95", "coverage_q99", "pinball_q95", "pinball_q99"))  # fmt: skip
    return table


def sunny_overcast_gap(*, paired_by_arm: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return each arm's gain on cold sunny days minus its gain on cold overcast days.

    A negative gap means the arm helps more on sunny days. Both slices are resampled together in
    7-day blocks, because overcast days exist only in one fold.

    Args:
        paired_by_arm: The output of `paired` for each arm.

    Returns:
        One row per arm and group with the gap, its interval, and the two slice contrasts.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    out: list[dict[str, object]] = []
    tail = (100.0 - LEVEL_PERCENT) / 2.0
    for arm, frame in paired_by_arm.items():
        for group in (SIX, EIGHT):
            keep = pl.col(HEADLINE) & pl.col("label").is_in(list(SUBSETS[group]))
            blocks = np.sort(frame.filter(keep)["block"].unique().to_numpy())
            sums = {}
            for name, mask in (
                ("sunny", pl.col("cold_sunny_day")),
                ("overcast", pl.col("cold_overcast_day")),
            ):
                grouped = (
                    frame.filter(keep & mask)
                    .group_by("block")
                    .agg(total=pl.col("d_crps").sum(), count=pl.len())
                )
                lookup = dict(
                    zip(
                        grouped["block"].to_list(),
                        zip(grouped["total"].to_list(), grouped["count"].to_list(), strict=True),
                        strict=True,
                    )
                )
                sums[name] = np.array(
                    [lookup.get(int(b), (0.0, 0)) for b in blocks], dtype=np.float64
                )
            draws = rng.integers(0, len(blocks), size=(N_RESAMPLES, len(blocks)))
            sunny_n = sums["sunny"][:, 1][draws].sum(axis=1)
            over_n = sums["overcast"][:, 1][draws].sum(axis=1)
            valid = (sunny_n > 0) & (over_n > 0)
            estimate = (
                sums["sunny"][:, 0][draws].sum(axis=1)[valid] / sunny_n[valid]
                - sums["overcast"][:, 0][draws].sum(axis=1)[valid] / over_n[valid]
            )
            point_sunny = sums["sunny"][:, 0].sum() / sums["sunny"][:, 1].sum()
            point_over = (
                sums["overcast"][:, 0].sum() / sums["overcast"][:, 1].sum()
                if sums["overcast"][:, 1].sum()
                else float("nan")
            )
            lower, upper = (
                np.percentile(estimate, [tail, 100.0 - tail])
                if estimate.size
                else (float("nan"),) * 2
            )
            out.append(
                {
                    "arm": arm,
                    "group": group,
                    "sunny_gain": float(point_sunny),
                    "overcast_gain": float(point_over),
                    "gap": float(point_sunny - point_over),
                    "lower": float(lower),
                    "upper": float(upper),
                    "resamples_used": int(valid.sum()),
                }
            )
    table = pl.DataFrame(out)
    with pl.Config(tbl_rows=40, tbl_cols=10):
        print(table)
    return table


def gate_against_437(*, paired_by_arm: dict[str, pl.DataFrame]) -> None:
    """Print each arm's all-rows contrast beside experiment 437's published contrasts (the gate)."""
    published = {
        "T": (0.016, -0.022), "S": (0.043, 0.002), "C": (0.059, 0.038), "C_tariff": (0.047, 0.021),
        "F": (0.042, -0.000), "T_pre": (0.025, -0.029), "S_pre": (-0.016, 0.011), "F_pre": (0.026, 0.020),
    }  # fmt: skip
    print(
        "\n== gate: all-rows CRPS contrast here against experiment 437's published table (six, eight) =="
    )
    for arm, frame in paired_by_arm.items():
        here = []
        for group in (SIX, EIGHT):
            part = frame.filter(cell_filter(mask=None, group=group, outside=False))
            here.append(round(float(cast("float", part["d_crps"].mean())), 3))
        print(f"{arm:9s} here {tuple(here)}  published {published[arm]}")


def score() -> None:
    """Score the eight arms on the slices and write the figure tables."""
    rows = build_rows(arms=list(SCREEN_ARMS))
    FIGURE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [a for a in SCREEN_ARMS if f"{a}_w115" not in rows]
    if missing:
        msg = f"arms missing from the loaded forecasts: {missing}"
        raise ValueError(msg)
    paired_by_arm = {arm: paired(rows=rows, arm=arm) for arm in SCREEN_ARMS}
    gate_against_437(paired_by_arm=paired_by_arm)
    print("\n== absolute scores (equal weight over Primaries; CRPS in points of capacity) ==")
    absolute_table(rows=rows).write_parquet(FIGURE_TABLE_DIR / "absolute.parquet")
    print(
        f"\n== contrasts, arm minus control, bootstrap level {LEVEL_PERCENT:.1f}%, {N_RESAMPLES:,} resamples =="
    )
    contrasts = contrast_table(paired_by_arm=paired_by_arm)
    contrasts.write_parquet(FIGURE_TABLE_DIR / "contrasts.parquet")
    with pl.Config(tbl_rows=400, tbl_cols=20, fmt_str_lengths=60):
        print(
            contrasts.filter(pl.col("group") != REPORT_ONLY).select(
                "arm",
                "group",
                "slice",
                "side",
                "mean",
                "lower",
                "upper",
                "rows",
                "days",
                "block_kind",
                "fold_differences",
                "d_mae_median",
                "d_pinball_q95",
                "d_pinball_q99",
            )
        )
    print("\n== decision rule on the tested cells ==")
    decide(table=contrasts).write_parquet(FIGURE_TABLE_DIR / "decisions.parquet")
    print(
        "\n== sunny-minus-overcast gap of each arm's gain (negative: helps more on sunny days) =="
    )
    sunny_overcast_gap(paired_by_arm=paired_by_arm).write_parquet(FIGURE_TABLE_DIR / "gap.parquet")


def main() -> None:
    """Run the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pre", "score"))
    args = parser.parse_args()
    if args.stage == "pre":
        pre()
    else:
        score()


if __name__ == "__main__":
    main()

# ruff: noqa: E501, PLR0915
"""Experiment 459: summarise the conservation check into tables, contrasts, and the decision rule.

Reads the tables that `e00459_conservation.py` wrote, prints the placebo-based minimum detectable
effect first (so it is stated before the headline is read), then the group tables, the four
pre-registered contrasts with Bonferroni (k = 4) event-resampled intervals, the robustness checks,
the unobserved-recipient share, and the allocation results. Writes CSV tables beside `result.md`.
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00459-load-shift-conservation-check-donor-and-recipient-sum"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
DOCS: Final[Path] = Path(__file__).parents[2] / "docs" / "ml-experiments" / SLUG
LEAK_THRESHOLD: Final[float] = 0.25
BOOTSTRAPS: Final[int] = 5_000
FAMILY_ALPHA: Final[float] = 0.05
K_CONTRASTS: Final[int] = 4
SEED: Final[int] = 459
GROUPS: Final[tuple[str, ...]] = ("g1", "g2", "g3")


def population(*, rows: pl.DataFrame, metering: str, observed: bool | None) -> pl.DataFrame:
    """Return the included event rows of one metering group (`all` for every group)."""
    out = rows.filter((pl.col("kind") == "event") & (pl.col("reason") == ""))
    if metering != "all":
        out = out.filter(pl.col("group") == metering)
    if observed is not None:
        out = out.filter(pl.col("observed") == observed)
    return out


def placebo_arrays(*, rows: pl.DataFrame, events: pl.DataFrame, group: str) -> list[np.ndarray]:
    """Return, for each event, the finite placebo leak shares of its placebo windows."""
    placebo = rows.filter(pl.col("kind") == "placebo")
    out = []
    for row in events["row"].to_list():
        values = placebo.filter(pl.col("row") == row)[f"{group}_leak"].drop_nans().drop_nulls()
        out.append(values.to_numpy())
    return out


def contrasts(
    *, leak: np.ndarray, placebo: list[np.ndarray], index: np.ndarray
) -> tuple[float, float]:
    """Return the median-leak contrast and the share-at-most-0.25 contrast for the chosen events."""
    event_values = leak[index]
    placebo_values = np.concatenate([placebo[i] for i in index]) if len(index) else np.array([])
    if event_values.size == 0 or placebo_values.size == 0:
        return np.nan, np.nan
    return (
        float(np.median(event_values) - np.median(placebo_values)),
        float((event_values <= LEAK_THRESHOLD).mean() - (placebo_values <= LEAK_THRESHOLD).mean()),
    )


def bootstrap(
    *, leak: np.ndarray, placebo: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Return the bootstrap draws of the two contrasts, resampling events with their placebos."""
    n = len(leak)
    draws = np.empty((BOOTSTRAPS, 2))
    for b in range(BOOTSTRAPS):
        draws[b] = contrasts(leak=leak, placebo=placebo, index=rng.integers(0, n, n))
    return draws


def main() -> None:
    """Print every table the write-up quotes."""
    rows = pl.read_parquet(WORK / "event_and_placebo_rows.parquet")
    events = rows.filter(pl.col("kind") == "event")
    rng = np.random.default_rng(SEED)
    level = 1 - FAMILY_ALPHA / K_CONTRASTS
    pct = [100 * (1 - level) / 2, 100 * (1 + level) / 2]

    print("== Counts ==")
    print(events.group_by("group", "observed", "reason").len().sort("group", "observed", "reason"))
    unobs = events.group_by("group").agg(
        n=pl.len(),
        unobserved=(~pl.col("observed")).sum(),
        share_unobserved=(~pl.col("observed")).mean(),
    )
    print("share of unobserved recipients, all 112 events by donor group:\n", unobs)
    print("overall", float((~events["observed"]).mean()))  # ty: ignore[invalid-argument-type]

    print("\n== MDE from the placebo spread (stated before the headline) ==")
    for metering in ("six", "eight"):
        pop = population(rows=rows, metering=metering, observed=True)
        leak = pop["g1_leak"].to_numpy()
        placebo = placebo_arrays(rows=rows, events=pop, group="g1")
        medians = [
            float(
                np.median(
                    np.concatenate(
                        [placebo[i] for i in rng.integers(0, len(placebo), len(placebo))]
                    )
                )
            )
            for _ in range(1_000)
        ]
        sd = float(np.std(medians))
        print(
            metering,
            "events",
            len(leak),
            "placebo-median SD",
            round(sd, 3),
            "MDE (2.8 SD)",
            round(2.8 * sd, 3),
        )

    print("\n== Group table ==")
    table = []
    for gname in GROUPS:
        for metering in ("six", "eight", "other", "all"):
            for observed in (True, False, None):
                pop = population(rows=rows, metering=metering, observed=observed)
                if pop.height == 0:
                    continue
                leak = pop[f"{gname}_leak"].drop_nans().drop_nulls().to_numpy()
                placebo = np.concatenate(
                    placebo_arrays(rows=rows, events=pop, group=gname) or [np.array([])]
                )
                table.append(
                    {
                        "group": gname, "metering": metering,
                        "observed": "all" if observed is None else str(observed),
                        "n_events": pop.height, "median_leak": float(np.median(leak)),
                        "share_leak_at_most_0.25": float((leak <= LEAK_THRESHOLD).mean()),
                        "placebo_median": float(np.median(placebo)) if placebo.size else np.nan,
                        "placebo_share_at_most_0.25": float((placebo <= LEAK_THRESHOLD).mean()) if placebo.size else np.nan,
                        "placebo_p10": float(np.percentile(placebo, 10)) if placebo.size else np.nan,
                        "placebo_p90": float(np.percentile(placebo, 90)) if placebo.size else np.nan,
                    }
                )  # fmt: skip
    group_table = pl.DataFrame(table)
    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=200):
        print(group_table)
    group_table.write_csv(DOCS / "table_groups.csv")

    print("\n== Pre-registered contrasts (G1, observed recipient), Bonferroni k = 4 ==")
    result = []
    for metering in ("six", "eight"):
        pop = population(rows=rows, metering=metering, observed=True)
        leak = pop["g1_leak"].to_numpy()
        placebo = placebo_arrays(rows=rows, events=pop, group="g1")
        everyone = np.arange(len(leak))
        point = contrasts(leak=leak, placebo=placebo, index=everyone)
        draws = bootstrap(leak=leak, placebo=placebo, rng=rng)
        lo_hi = np.nanpercentile(draws, pct, axis=0)
        donors = pop["donor"].to_numpy()
        lopo = [
            contrasts(leak=leak, placebo=placebo, index=np.flatnonzero(donors != d))
            for d in sorted(set(donors))
        ]
        largest = int(np.argmax(np.abs(pop["donor_step_mw"].to_numpy())))
        dropped = contrasts(leak=leak, placebo=placebo, index=np.delete(everyone, largest))
        for k, name in enumerate(
            ("median leak minus placebo median", "share at most 0.25 minus placebo share")
        ):
            result.append(
                {
                    "metering": metering, "contrast": name, "n_events": len(leak), "estimate": point[k],
                    "lo": lo_hi[0][k], "hi": lo_hi[1][k],
                    "leave_one_primary_out_min": min(x[k] for x in lopo),
                    "leave_one_primary_out_max": max(x[k] for x in lopo),
                    "largest_event_dropped": dropped[k],
                }
            )  # fmt: skip
    contrast_table = pl.DataFrame(result)
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=220):
        print(contrast_table)
    contrast_table.write_csv(DOCS / "table_contrasts.csv")

    print("\n== Stop rule and predictions ==")
    six_all = population(rows=rows, metering="six", observed=None)
    six_obs = population(rows=rows, metering="six", observed=True)
    six_unobs = population(rows=rows, metering="six", observed=False)

    def clean(frame: pl.DataFrame, column: str) -> np.ndarray:
        return frame[column].drop_nans().drop_nulls().to_numpy()

    for name, frame, column in (
        ("median G2 leak at the six (all included events)", six_all, "g2_leak"),
        ("median G2 leak, unobserved events at the six", six_unobs, "g2_leak"),
        ("median G3 leak at the six", six_all, "g3_leak"),
        ("median G1 leak, observed events at the six", six_obs, "g1_leak"),
    ):
        values = clean(frame, column)
        print(name, ":", float(np.median(values)), "n valid", values.size, "of", frame.height)
    g1_obs = clean(six_obs, "g1_leak")
    print(
        "share of observed-recipient events with G1 leak <= 0.25 at the six:",
        float((g1_obs <= LEAK_THRESHOLD).mean()),
        "n valid",
        g1_obs.size,
    )

    print("\n== Placebo windows with missing readings (dead or flagged meter proxy), G1 ==")
    placebo_rows = rows.filter(pl.col("kind") == "placebo")
    print(
        placebo_rows.with_columns(gappy=pl.col("g1_missing_share") > 0.05)
        .group_by("group", "gappy")
        .agg(
            pl.len().alias("windows"),
            pl.col("g1_leak").median().alias("median_g1_leak"),
            (pl.col("g1_leak") <= LEAK_THRESHOLD).mean().alias("share_at_most_0.25"),
        )
        .sort("group", "gappy")
    )

    print("\n== Allocation ==")
    alloc = pl.read_parquet(WORK / "allocation.parquet")
    alloc.write_csv(DOCS / "table_allocation.csv")
    with pl.Config(tbl_rows=120, tbl_cols=14, tbl_width_chars=250):
        for metering in ("six", "eight", "other"):
            part = alloc.filter(pl.col("metering") == metering)
            if part.is_empty():
                continue
            ratio = part["oracle_median_abs_pts_on_r0_rows"] / part["r0_median_abs_pts"]
            print(
                metering, "pairs", part.height,
                "median oracle", float(part["oracle_median_abs_pts_on_r0_rows"].median()),  # ty: ignore[invalid-argument-type]
                "median R0", float(part["r0_median_abs_pts"].median()),  # ty: ignore[invalid-argument-type]
                "median ratio", float(ratio.median()),  # ty: ignore[invalid-argument-type]
                "share of pairs with ratio < 0.5", float((ratio < 0.5).mean()),  # ty: ignore[invalid-argument-type]
            )  # fmt: skip
            harm = part.filter(pl.col("harm_rows") > 0)
            if harm.is_empty():
                print(metering, "no harm rows")
                continue
            print(
                metering, "HARM ROWS: pairs", harm.height,
                "median oracle", float(harm["harm_oracle_median_abs_pts"].median()),  # ty: ignore[invalid-argument-type]
                "median R0", float(harm["harm_r0_median_abs_pts"].median()),  # ty: ignore[invalid-argument-type]
                "median paired gain (R0 minus oracle, points)", float(harm["harm_paired_gain_median_pts"].median()),  # ty: ignore[invalid-argument-type]
                "total harm rows", int(harm["harm_rows"].sum()),
            )  # fmt: skip
            print(
                part.group_by("member")
                .agg(
                    pl.col("oracle_median_abs_pts_on_r0_rows").median().alias("oracle"),
                    pl.col("r0_median_abs_pts").median().alias("r0"),
                    pl.col("share_err_p10_pts").median().alias("share_err_p10"),
                    pl.col("share_err_p90_pts").median().alias("share_err_p90"),
                    pl.col("harm_rows").sum().alias("harm_rows"),
                    pl.col("harm_paired_gain_median_pts").median().alias("harm_gain"),
                    pl.len().alias("groups"),
                ).sort("member")
            )  # fmt: skip


if __name__ == "__main__":
    main()

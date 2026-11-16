# ruff: noqa: E501
"""Experiment 699: print the tail coverage, the cold-night scores, and the avoidable-curtailment cost of every arm.

Usage: python e00699_extras.py

Reads what experiment 677's `e00677_tails.py` and `e00677_protocol.py` wrote under each chain's `scores/`
when they ran with `E00677_ROOT` set to `roots/clean`, `roots/early`, and `roots/stage2` for steps
`5w:1:01234567` and `5wo:1:01234567` (`5w` only for `stage2`). The protocol rows are the half-hours that
the arms of experiment 668's comparison share (13 Primaries, not the 16 of the main scores).
"""

import polars as pl
from e00699_common import ROOT

ARM_OF = {
    ("clean", "5w"): "s3",
    ("clean", "5wo"): "none",
    ("early", "5w"): "s123",
    ("early", "5wo"): "s12",
    ("stage2", "5w"): "s23",
}
ORDER = ["none", "s3", "s23", "s12", "s123"]


def tails() -> None:
    """Print the share of half-hours at or below each of the q01, q05, q95, and q99 forecasts."""
    print(
        "\n=== tail calibration: share of scored half-hours at or below the quantile (nominal level in the header)"
    )
    rows = []
    for (chain, step), arm in ARM_OF.items():
        frame = pl.read_csv(ROOT / "roots" / chain / "scores" / "tail_calibration.csv").filter(
            pl.col("step").str.starts_with(f"{step} ")
        )
        rows.append(
            {
                "arm": arm,
                **{
                    f"q{level:.2f}": share
                    for level, share in zip(frame["level"], frame["share_at_or_below"], strict=True)
                    if level in {0.01, 0.05, 0.95, 0.99}
                },
            }
        )
    print(
        pl.DataFrame(rows).sort(pl.col("arm").replace_strict({a: i for i, a in enumerate(ORDER)}))
    )


def protocol() -> None:
    """Print the cold-night scores and the avoidable-curtailment cost by arm."""
    scores = []
    costs = []
    for (chain, step), arm in ARM_OF.items():
        name = "Step 5w monthly" if step == "5w" else "Step 5wo monthly"
        base = ROOT / "roots" / chain / "scores" / "protocol_p699"
        scores.append(
            pl.read_csv(base / "scores.csv")
            .filter(pl.col("arm") == name)
            .with_columns(arm_label=pl.lit(arm))
        )
        costs.append(
            pl.read_csv(base / "curtailment_cost.csv")
            .filter(pl.col("arm") == name)
            .group_by("limit_fraction")
            .agg(
                cost_gbp=pl.col("avoidable_cost_gbp").sum(),
                statuses=pl.col("status").unique().sort(),
            )
            .with_columns(arm_label=pl.lit(arm))
        )
    table = pl.concat(scores)
    print("\n=== cold nights and all rows, by metering group (protocol rows; points of capacity)")
    for metric in ("mae_median", "crps"):
        wide = table.filter(
            pl.col("slice").is_in(["all", "cold_nights"]) & (pl.col("metric") == metric)
        ).pivot(on="arm_label", index=["group", "slice"], values="mean")
        print(metric)
        print(wide.select("group", "slice", *ORDER))
    print(
        "\n=== avoidable-curtailment cost, summed over the six signed-MW Primaries (GBP), by arm and limit fraction"
    )
    print(
        pl.concat(costs)
        .pivot(on="arm_label", index="limit_fraction", values="cost_gbp")
        .select("limit_fraction", *ORDER)
        .sort("limit_fraction")
    )
    reference = pl.read_csv(
        ROOT / "roots" / "clean" / "scores" / "protocol_p699" / "curtailment_cost.csv"
    )
    for arm in ("Manual heuristic", "GAM", "Climatology"):
        print(
            arm,
            reference.filter(pl.col("arm") == arm)
            .group_by("limit_fraction")
            .agg(pl.col("avoidable_cost_gbp").sum())
            .sort("limit_fraction")
            .to_dicts(),
        )


if __name__ == "__main__":
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    tails()
    protocol()

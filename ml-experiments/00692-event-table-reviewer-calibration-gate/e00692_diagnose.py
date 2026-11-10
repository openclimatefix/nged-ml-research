"""Experiment 692 post hoc check: why does the reviewer call so many "false" candidates a transfer?

For each calibration candidate the reviewer called a transfer, look for an injected transfer at the
donor or the recipient Primary that overlaps the candidate in time, and say how the candidate's
true/false label failed to match it (start more than 24 half-hours away, wrong partner, or none).
"""

from __future__ import annotations

import warnings

import polars as pl
from e00653_common import DAY  # ty: ignore[unresolved-import]
from e00680_run import CONFIGS, Rig  # ty: ignore[unresolved-import]
from e00692_score import OUT, load_answers  # ty: ignore[unresolved-import]

import e00660_generator as gen  # ty: ignore[unresolved-import]  # isort: skip

warnings.filterwarnings("ignore")


def main() -> None:
    """Print and store one row per called false candidate."""
    key = pl.read_parquet(OUT / "answer_key_private" / "key.parquet")
    ans = load_answers(OUT / "answers" / "calibration")
    called = (
        key.filter(pl.col("set") == "calibration")
        .join(ans, on="figure")
        .filter((pl.col("verdict") == "transfer") & (pl.col("source") == "injected"))
        .filter(~pl.col("stratum").is_in(["true"]))
    )
    rig = Rig()
    rows = []
    for seed in sorted(called["seed"].unique()):
        world = gen.inject(
            panel=rig.panel, config=CONFIGS["default"], seed=int(seed), start_times=rig.starts
        )
        ev = world.events.filter(pl.col("label") == "edge_transfer").to_dicts()
        for r in called.filter(pl.col("seed") == seed).to_dicts():
            end = r["end"] if r["end"] >= 0 else r["onset"] + 30 * DAY
            near = [
                e
                for e in ev
                if e["site"] in (r["donor"], r["recipient"])
                and e["start"] < end
                and (e["end"] if e["end"] is not None else 10**9) > r["onset"]
            ]
            same_pair = [
                e
                for e in near
                if e["role"] == "donor"
                and e["site"] == r["donor"]
                and r["recipient"] in list(e["partners"])
            ]
            rows.append(
                {
                    "figure": r["figure"],
                    "stratum": r["stratum"],
                    "n_overlapping_injected_transfer_ends": len(near),
                    "same_pair_overlaps": len(same_pair),
                    "best_start_gap_hh": min(
                        (abs(e["start"] - r["onset"]) for e in same_pair), default=None
                    ),
                }
            )
    out = pl.DataFrame(rows)
    print(
        out.group_by("stratum").agg(
            pl.len().alias("n"),
            (pl.col("same_pair_overlaps") > 0).sum().alias("same_pair_injected_overlaps"),
            (pl.col("n_overlapping_injected_transfer_ends") > 0)
            .sum()
            .alias("any_injected_overlap"),
        )
    )
    print(out.filter(pl.col("same_pair_overlaps") > 0)["best_start_gap_hh"].describe())
    out.write_parquet(OUT / "answer_key_private" / "diagnose.parquet")


if __name__ == "__main__":
    main()

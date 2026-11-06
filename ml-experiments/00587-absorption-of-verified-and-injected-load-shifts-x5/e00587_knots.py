# ruff: noqa: E402, E501
"""Experiment 587: how far the newest offset knot of the D5 and P5 networks moves toward an injected
or verified event.

For each case, event Primary, fold month, refit day, and seed, the knot (`knot_new` for D5,
`off_new` for P5; a per-Primary output offset in standardised units) of the fit in the event
world minus the same knot in the no-event world, times the Primary's target standard deviation of
that fold month (capacity share), times the event's sign. Written to
`scores/knot_movement.parquet`.
"""

import sys

sys.argv = ["e00565_fit.py", "fit", "--arm", "D5"]
import e00565_fit as e  # ty: ignore[unresolved-import]
import e00587_common as common  # ty: ignore[unresolved-import]
import polars as pl
import torch  # ty: ignore[unresolved-import]

KEYS = ("knot_new", "off_new")


def knot_of(*, arm: str, world: str, month: str, day: int, seed: int) -> torch.Tensor:
    """Return the newest-knot offset of every Primary (standardised) from a saved fit."""
    tag = common.ARMS[arm][1]
    path = (
        common.world_dir(arm=arm, world=world)
        / "work"
        / "models"
        / tag
        / f"{month}_d{day}_s{seed}.pt"
    )
    sd = torch.load(path, map_location="cpu")
    key = next(k for k in KEYS if k in sd)
    return sd[key].reshape(-1)


def main() -> None:
    """Write the knot-movement table."""
    clean = e.c.clean_stack()
    std: dict[str, torch.Tensor] = {}
    rows = []
    cases = ["verified", *[f"inj{i}" for i in common.designs()["spec_id"].to_list()]]
    for case in cases:
        ev, no = common.case_worlds(case=case)
        for month in common.case_months(case=case):
            if month not in std:
                mr = e.k.build_month(month=month, clean=clean)
                std[month] = torch.from_numpy(mr.base.target_std)
            for arm in ("D5", "P5"):
                for event in common.case_events(case=case):
                    for day in (0, 14):
                        for seed in common.SEEDS:
                            d = knot_of(
                                arm=arm, world=ev, month=month, day=day, seed=seed
                            ) - knot_of(arm=arm, world=no, month=month, day=day, seed=seed)
                            move = float(event.sign * d[event.primary] * std[month][event.primary])
                            rows.append(
                                {"arm": arm, "case": case, "label": event.label, "month": month,
                                 "refit_day": day, "seed": seed, "depth": event.depth,
                                 "knot_movement": move}
                            )  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_parquet(common.OUT / "scores" / "knot_movement.parquet")
    inj = out.filter((pl.col("case") != "verified") & (pl.col("refit_day") == 14))
    print(
        inj.group_by("arm").agg(
            pl.col("knot_movement").mean().alias("mean"),
            pl.col("knot_movement").abs().max().alias("max_abs"),
        )
    )
    print(
        out.filter(pl.col("case") == "verified")
        .group_by("arm", "label", "month", "refit_day")
        .agg(pl.col("knot_movement").mean())
        .sort("arm", "label", "month", "refit_day")
    )


if __name__ == "__main__":
    main()

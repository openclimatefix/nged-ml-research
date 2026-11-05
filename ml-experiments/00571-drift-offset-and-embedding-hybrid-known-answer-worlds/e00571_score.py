# ruff: noqa: E501
"""Experiment 571: score the arms in a world, then summarise recovery of the injected component.

`rows --world W --arms ... --months ... --seeds ...` writes one table per world,
`worlds/<W>/scores/rows.parquet`: one row per forecast (Primary, origin, target) of experiment 565's
row set (experiment 559's `rows_<month>.parquet`, rows with `main16`, lead 24 h up to but not
including 120 h, a finite actual), with the world's actual and the CRPS of every arm and seed. Each
seed is scored on its own, widened by 1.15 as every stored row table is.

`summary --worlds ...` reads those tables and writes `summary_*.csv` under the experiment folder:
absolute CRPS, recovery fractions against the base without drift and the oracle, contrasts between
arms, and the untouched Primaries' change.

Recovery = (CRPS of the base network without drift minus the arm's) / (the base's minus the oracle's),
on the injected Primaries at 24-120 h, per fold, with the seeds' CRPS averaged first.
"""

import argparse
from typing import Final

import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00571_world as world
import numpy as np
import polars as pl

KEYS: Final = ["primary", "origin", "target"]
BASE: Final[str] = "M0w"
ORACLE: Final[str] = "ORA"
GROUP_OF: Final[dict[int, str]] = {0: "eight", 2: "eight", 8: "eight", 5: "six", 7: "six", 6: "six"}
"""Group of each injected Primary: D2, D4, D12 are MVA Primaries (the eight); D9, D11, D10 are MW Primaries (the six)."""


def available_seeds(*, w: str, arm: str, months: list[str], seeds: list[int]) -> list[int]:
    """Return the seeds whose assembled forecasts exist for every month."""
    folder = world.world_dir(world=w) / "work" / "seed_preds" / arm
    return [s for s in seeds if all((folder / f"{m}_s{s}.parquet").exists() for m in months)]


def world_rows(*, w: str, arms: list[str], months: list[str], seeds: list[int]) -> pl.DataFrame:
    """Return the world's scored rows with one CRPS column per arm and seed."""
    import e00510_common as c

    world.install(w)
    actual = c.clean_stack()
    sites = build470.site_map()
    frames = []
    for month in months:
        rows = pl.read_parquet(
            d6.W559 / "scores" / f"rows_{month}.parquet",
            columns=[*KEYS, "group", "lead_hours", "cold", "main16"],
        ).filter(pl.col("main16") & (pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 120))
        prim = rows["primary"].to_numpy().astype(np.int64)
        tgt = rows["target"].to_numpy().astype(np.int64)
        rows = rows.with_columns(pl.Series("actual", actual[prim, tgt].astype(np.float64)))
        rows = rows.filter(pl.col("actual").is_finite())
        cols = {}
        for arm in arms:
            for seed in available_seeds(w=w, arm=arm, months=months, seeds=seeds):
                frame = build470.pool_stored(
                    folder=world.world_dir(world=w) / "work" / "seed_preds" / arm,
                    month=month,
                    sites=sites,
                    seeds=(seed,),
                )
                fr = rows.select(*KEYS, "actual").join(
                    frame.select(*KEYS, *d6.LEVELS), on=KEYS, how="left", maintain_order="left"
                )
                ok = fr["q50"].is_not_null().to_numpy()
                fr = fr.with_columns(*[pl.col(q).fill_null(0.0) for q in d6.LEVELS])
                fr = score470.widen_arm(arm="M0", frame=fr, factor=score470.HEADLINE)
                cols[f"{arm}_s{seed}"] = np.where(
                    ok, score470.row_metrics(frame=fr)["crps"], np.nan
                )
        frames.append(
            rows.hstack(
                pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()})
            )
            .with_columns(month=pl.lit(month))
            .drop("actual")
        )
    return pl.concat(frames)


def write_rows(*, w: str, arms: list[str] | None, months: list[str], seeds: list[int]) -> None:
    """Write a world's row table (every arm with forecasts in all of `months`)."""
    if arms is None:
        folder = world.world_dir(world=w) / "work" / "seed_preds"
        arms = sorted(
            a.name
            for a in folder.iterdir()
            if available_seeds(w=w, arm=a.name, months=months, seeds=seeds)
        )
    table = world_rows(w=w, arms=arms, months=months, seeds=seeds)
    per_arm = {a: available_seeds(w=w, arm=a, months=months, seeds=seeds) for a in arms}
    assert len({tuple(v) for v in per_arm.values()}) == 1, f"arms differ in seeds: {per_arm}"
    score_cols = [c for c in table.columns if "_s" in c and c.split("_s")[-1].isdigit()]
    assert all(table[c].is_finite().all() for c in score_cols), (
        "an arm has no forecast for some rows"
    )
    out = world.world_dir(world=w) / "scores" / "rows.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    print("wrote", out, table.height, "rows;", [c for c in table.columns if "_s" in c])


def seed_mean(*, t: pl.DataFrame, arm: str) -> pl.Expr:
    """Return the mean over the arm's seeds of its CRPS column."""
    cols = [c for c in t.columns if c.startswith(f"{arm}_s")]
    return pl.mean_horizontal([pl.col(c) for c in cols])


def summary(*, worlds: list[str]) -> None:
    """Write absolute CRPS, recovery, and contrast tables for the worlds."""
    absolute, recovery, contrasts = [], [], []
    for w in worlds:
        t = pl.read_parquet(world.world_dir(world=w) / "scores" / "rows.parquet")
        arms = sorted({c.rsplit("_s", 1)[0] for c in t.columns if "_s" in c and c[0] != "_"})
        arms = [a for a in arms if a not in {"lead", "primary"}]
        t = t.with_columns(**{f"{a}_mean": seed_mean(t=t, arm=a) for a in arms})
        injected = t.filter(pl.col("primary").is_in(list(world.INJECTED))).with_columns(
            grp=pl.col("primary").replace_strict(GROUP_OF, default=None)
        )
        untouched = t.filter(~pl.col("primary").is_in(list(world.INJECTED)))
        for scope, part in (("injected", injected), ("untouched", untouched)):
            for arm in arms:
                by = part.group_by("month").agg(pl.col(f"{arm}_mean").mean().alias("crps"))
                absolute.append(
                    {
                        "world": w, "scope": scope, "arm": arm, "n_rows": part.height,
                        **{f"crps_{r['month']}": r["crps"] for r in by.iter_rows(named=True)},
                        "crps": float(by["crps"].mean()),
                    }
                )  # fmt: skip
        for grp in ("eight", "six", None):
            part = injected if grp is None else injected.filter(pl.col("grp") == grp)
            per = {
                a: part.group_by("month")
                .agg(pl.col(f"{a}_mean").mean())
                .sort("month")[f"{a}_mean"]
                .to_numpy()
                for a in arms
            }
            if BASE in per and ORACLE in per and w != "W0":  # W0 has no gap to recover
                gap = per[BASE] - per[ORACLE]
                for arm in arms:
                    rec = (per[BASE] - per[arm]) / gap
                    recovery.append(
                        {
                            "world": w, "group": grp or "both", "arm": arm,
                            "gap_base_minus_oracle": float(gap.mean()),
                            "recovery": float(((per[BASE] - per[arm]).mean()) / gap.mean()),
                            "recovery_by_fold": [float(r) for r in rec],
                        }
                    )  # fmt: skip
            for arm in arms:
                for ref in ("D5", "A10"):
                    if arm in per and ref in per and arm != ref:
                        d = per[arm] - per[ref]
                        contrasts.append(
                            {
                                "world": w, "group": grp or "both", "arm": arm, "ref": ref,
                                "difference": float(d.mean()),
                                "by_fold": [float(x) for x in d],
                                "folds_positive": int((d > 0).sum()),
                            }
                        )  # fmt: skip
    out = world.WORK
    pl.DataFrame(absolute).write_csv(out / "summary_absolute.csv")
    pl.DataFrame(recovery).with_columns(
        pl.col("recovery_by_fold").list.eval(pl.element().round(3).cast(pl.String)).list.join(" ")
    ).write_csv(out / "summary_recovery.csv")
    pl.DataFrame(contrasts).with_columns(
        pl.col("by_fold").list.eval(pl.element().round(3).cast(pl.String)).list.join(" ")
    ).write_csv(out / "summary_contrasts.csv")
    with pl.Config(tbl_rows=120, tbl_cols=12, fmt_str_lengths=40):
        print(
            pl.DataFrame(recovery)
            .drop("recovery_by_fold")
            .with_columns(pl.col(pl.Float64).round(3))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("rows", "summary"))
    parser.add_argument("--world", default="real")
    parser.add_argument("--worlds", nargs="+", default=["W0", "Wa", "Wb1", "Wc"])
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--months", nargs="+", default=list(world.FOLD_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    if args.command == "rows":
        write_rows(w=args.world, arms=args.arms, months=args.months, seeds=args.seeds)
    else:
        summary(worlds=args.worlds)

"""Score the students on experiment 404's injected scoring worlds (200 to 202), harm_inj rows.

A harm_inj row has its target outside every injected event and a day window inside one. Arms are
read from this experiment's folder (students) and experiment 404's (A0-inj, S1, B-mix); K2-S1 has
no forecast on the injected worlds. Only the (month, seed) cells that experiment 404 stored are
used. Prints the mean score per group, as the mean over months of the mean over seeds and worlds.
"""

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00490_distill as distill  # ty: ignore[unresolved-import]
import polars as pl

MONTHS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
groups = report.group_of(sites=fit.prep.primaries()["site"].to_list())
rows = []
for arm in ("D-K2", "D-Bmix", "D-K2-corr", "A0-inj", "S1", "B-mix"):
    base = distill.PRED_DIR if arm.startswith("D-") else fit.PRED_DIR
    for month in MONTHS:
        for seed in (0, 1, 2):
            for world in score.SCORING_SEEDS:
                name = f"{month}_s{seed}_w{world}.parquet"
                if not (fit.PRED_DIR / "B-mix" / name).exists():
                    continue  # experiment 404 stored these worlds for some cells only
                f = pl.read_parquet(base / arm / name)
                f = f.filter(~pl.col("tgt_inj") & pl.col("win_inj"))
                f = f.with_columns(
                    crps=score.pinball_points(frame=f),
                    group=pl.Series(groups[f["primary"].to_numpy().astype(int)]),
                )
                for (g,), p in f.group_by("group"):
                    rows.append((arm, month, seed, world, g, p.height, p["crps"].mean()))
df = pl.DataFrame(
    rows, schema=["arm", "month", "seed", "world", "group", "n", "crps"], orient="row"
)
month = df.group_by("arm", "month", "group").agg(pl.col("crps").mean(), pl.col("n").mean())
print(
    month.group_by("arm", "group")
    .agg(pl.col("crps").mean(), pl.col("n").mean())
    .sort("group", "arm")
)
month.write_csv(distill.WORK_DIR / "scores" / "harm_inj_month.csv")

# ruff: noqa: E501
r"""Print every number the write-up of experiment 470 quotes beyond `e00470_score.py` and `e00470_tails.py`.

- The pre-registered decision rule applied to the five candidates (`decision.csv`).
- Tail calibration (coverage and pinball loss at q01, q05, q95, q99) raw, widened, and after table P.
- The checks: median minus mean forecast per Primary, mean-forecast error beside the CRPS contrasts,
  and a leave-one-Primary-out table of the headline contrasts (audit trigger 4).
- Production cost per candidate.

```bash
source .e470env.sh
uv run --no-sync python ml-experiments/00470-*/e00470_report.py
```
"""

from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00470_build as build  # ty: ignore[unresolved-import]
import e00470_score as scoring  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SCORES = scoring.SCORE_DIR
CANDIDATES: Final[tuple[str, ...]] = scoring.CANDIDATES
NETWORKS: Final[dict[str, int]] = {"A-pool": 5, "K2-S1": 4, "K2-425A": 4, "K2-425A+S1": 4, "A-gam-in": 5}
MDE: Final[dict[str, float]] = {"eight": 0.0645, "six_clean": 0.0999}
"""Experiment 428's minimum detectable effect of arm A's five single seeds (`mde.csv`), per group."""
TOLERANCE: Final[float] = 0.03
Q99_TOLERANCE: Final[float] = 0.02


def decision() -> pl.DataFrame:
    """Apply the pre-registered rule: CRPS tolerance, event and cold-night checks, q99, curtailment."""
    table = scoring.load_rows().filter(pl.col("gam14"))
    rng = np.random.default_rng(1)
    records = []
    mean_crps = {}
    for sname in ("eight", "six_clean"):
        part, _ = scoring.slice_filter(table=table, name=sname)
        mean_crps[sname] = {a: float(part[f"{a}__crps"].mean()) for a in CANDIDATES}
    best = {s: min(v, key=v.get) for s, v in mean_crps.items()}  # ty: ignore[no-matching-overload]
    q99 = {}
    for sname in ("eight", "six_clean"):
        part, _ = scoring.slice_filter(table=table, name=sname)
        q99[sname] = {a: float(part[f"{a}__pin_q99"].mean()) for a in CANDIDATES}
    cur = pl.read_csv(SCORES / "curtailment_bootstrap.csv").filter(pl.col("fraction") == 0.9)
    for arm in CANDIDATES:
        row: dict = {"arm": arm, "serving_networks": NETWORKS[arm], "gam_in_serving": arm == "A-gam-in"}
        admissible = True
        for sname in ("eight", "six_clean"):
            gap = mean_crps[sname][arm] - mean_crps[sname][best[sname]]
            row[f"crps_{sname}"] = mean_crps[sname][arm]
            row[f"gap_to_best_{sname}"] = gap
            if arm != best[sname]:
                part, _ = scoring.slice_filter(table=table, name=sname)
                diff = part.select("month", "day", d=pl.col(f"{arm}__crps") - pl.col(f"{best[sname]}__crps"))
                ci = scoring.block_bootstrap(diff=diff, rng=rng)
                conclusive = abs(ci["difference"]) > MDE[sname] and (ci["low_bonferroni"] > 0 or ci["high_bonferroni"] < 0)
                row[f"conclusive_{sname}"] = conclusive
                admissible &= (not conclusive) or gap < TOLERANCE
            row[f"q99_{sname}"] = q99[sname][arm]
            row[f"q99_excess_{sname}"] = q99[sname][arm] - min(q99[sname].values())
        for sname in ("six_nra", "cold_eight", "cold_six"):
            part, prefix = scoring.slice_filter(table=table, name=sname)
            row[f"vs_K2S1_{sname}"] = float(part[f"{arm}__{prefix}crps"].mean() - part[f"K2-S1__{prefix}crps"].mean())
        row["curtailment_f09_millions"] = float(cur.filter(pl.col("item") == arm)["estimate"][0])
        costs = cur.filter(pl.col("item").is_in(list(CANDIDATES)))
        best_cost = costs.sort("estimate").row(0, named=True)["item"]
        row["curtailment_f09_best_arm"] = best_cost
        low = high = 0.0
        if arm != best_cost:
            direct = cur.filter(pl.col("item") == f"{arm} minus {best_cost}")
            reverse = cur.filter(pl.col("item") == f"{best_cost} minus {arm}")
            if direct.height:
                low, high = float(direct["low"][0]), float(direct["high"][0])
            else:
                low, high = -float(reverse["high"][0]), -float(reverse["low"][0])
        row["curtailment_f09_vs_best_low"], row["curtailment_f09_vs_best_high"] = low, high
        row["passes_curtailment"] = low <= 0.0
        row["admissible_crps"] = admissible
        row["passes_events_cold"] = all(row[f"vs_K2S1_{s}"] <= TOLERANCE for s in ("six_nra", "cold_eight", "cold_six"))
        row["passes_q99"] = all(row[f"q99_excess_{s}"] <= Q99_TOLERANCE for s in ("eight", "six_clean"))
        records.append(row)
    out = pl.DataFrame(records)
    out.write_csv(SCORES / "decision.csv")
    return out


def tails() -> pl.DataFrame:
    """Return tail coverage and pinball loss per candidate, variant, and group (rows with no labelled event)."""
    cells = pl.read_parquet(SCORES / "tails_cells.parquet").filter(
        (pl.col("cls") == "clean") & (pl.col("basis") == "metered")
    )
    cols = [c for c in cells.columns if c.startswith(("cov_", "pin_"))]
    out = (
        cells.group_by("arm", "variant", "group")
        .agg(pl.col("n").sum(), *[((pl.col(c) * pl.col("n")).sum() / pl.col("n").sum()).alias(c) for c in cols], ((pl.col("crps") * pl.col("n")).sum() / pl.col("n").sum()).alias("crps"))
        .sort("group", "arm", "variant")
    )
    out.write_csv(SCORES / "tails_summary.csv")
    return out


def checks() -> pl.DataFrame:
    """Return the median minus the mean forecast per Primary, and the mean-forecast absolute error."""
    table = scoring.load_rows().filter(pl.col("gam14")).select("month", "primary", "origin", "target", "group", "cls")
    levels = build.LEVELS
    records = []
    for arm in CANDIDATES:
        parts = []
        for month in build.MONTHS:
            keys = table.filter(pl.col("month") == month).select(*build.KEYS, "group", "cls")
            cells = [pl.read_parquet(build.prediction_path(arm=arm, month=month, cell=c)) for c in range(scoring.arm_cells(arm=arm))]
            q = np.mean([c.select(list(levels)).to_numpy().astype(np.float64) for c in cells], axis=0)
            frame = cells[0].select(*build.KEYS).hstack(pl.DataFrame(q, schema=list(levels), orient="row"))
            actual = scoring.score454.month_base(month=month).select(*build.KEYS, "actual")
            joined = keys.join(frame, on=build.KEYS).join(actual, on=build.KEYS)
            qs = joined.select(list(levels)).to_numpy()
            mean_forecast = (qs * common.ATOM_MASSES[None, :]).sum(axis=1)
            parts.append(joined.select("primary", "group", "cls", "actual", "q50").with_columns(mean=pl.Series(mean_forecast)))
        data = pl.concat(parts).with_columns(
            gap=(pl.col("q50") - pl.col("mean")) * 100.0,
            mae_mean=(pl.col("mean") - pl.col("actual")).abs() * 100.0,
            mae_median=(pl.col("q50") - pl.col("actual")).abs() * 100.0,
        )
        for (group, primary), part in data.group_by("group", "primary"):
            records.append({"arm": arm, "group": group, "primary": primary, "median_minus_mean_points": float(part["gap"].mean()), "n": part.height})
        for sname in ("eight", "six_clean"):
            g, kind = scoring.SLICES[sname]
            part = data.filter(pl.col("group") == g)
            if kind == "clean":
                part = part.filter(pl.col("cls") == "clean")
            records.append({"arm": arm, "group": sname, "primary": -1, "mae_mean_points": float(part["mae_mean"].mean()), "mae_median_points": float(part["mae_median"].mean()), "n": part.height})
    out = pl.DataFrame(records)
    out.write_csv(SCORES / "checks.csv")
    return out


def leave_one_out() -> pl.DataFrame:
    """Return each headline contrast with every Primary left out in turn (audit trigger 4)."""
    table = scoring.load_rows().filter(pl.col("gam14"))
    records = []
    for a, b in scoring.CONTRASTS:
        for sname in ("eight", "six_clean"):
            part, _ = scoring.slice_filter(table=table, name=sname)
            full = float((part[f"{a}__crps"] - part[f"{b}__crps"]).mean())
            for primary in sorted(part["primary"].unique().to_list()):
                rest = part.filter(pl.col("primary") != primary)
                records.append({"contrast": f"{a} - {b}", "slice": sname, "left_out": primary, "full": full, "without": float((rest[f"{a}__crps"] - rest[f"{b}__crps"]).mean())})
    out = pl.DataFrame(records)
    out.write_csv(SCORES / "leave_one_out.csv")
    return out


def main() -> None:
    """Print the decision table, tails, checks, and leave-one-out summary."""
    with pl.Config(tbl_rows=200, tbl_width_chars=250, tbl_cols=40):
        print(decision().with_columns(pl.col(pl.Float64).round(3)).transpose(include_header=True, header_name="field", column_names=CANDIDATES))
        t = tails()
        print(t.filter(pl.col("variant").is_in(["raw", "w1.15", "P"])).with_columns(pl.col(pl.Float64).round(3)))
        c = checks()
        print(c.filter(pl.col("primary") == -1).with_columns(pl.col(pl.Float64).round(3)))
        print(c.filter(pl.col("primary") >= 0).group_by("arm", "group").agg(pl.col("median_minus_mean_points").min().alias("min"), pl.col("median_minus_mean_points").max().alias("max")).sort("group", "arm").with_columns(pl.col(pl.Float64).round(3)))
        loo = leave_one_out()
        flips = loo.with_columns(flip=(pl.col("full") * pl.col("without") < 0)).group_by("contrast", "slice").agg(pl.col("flip").sum().alias("flips"), pl.col("without").min().alias("min_without"), pl.col("without").max().alias("max_without"), pl.col("full").first()).sort("contrast", "slice")
        print(flips.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()

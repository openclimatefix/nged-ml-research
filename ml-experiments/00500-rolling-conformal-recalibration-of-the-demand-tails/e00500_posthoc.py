"""Experiment 500, post hoc: is the CRPS gain of rolling conformal a median (bias) correction?

Scores, at the signed-MW six on clean rows over the five folds, an arm that applies only the
buffer's median shift to all 11 quantiles (a pure bias correction, `RC28-bias`), beside raw, RC28,
and RC14, and the mean absolute error of each arm's median (points of effective capacity). Also
writes the mean shift of q50, q95, and q99 per fold and lead band for the figures.
"""

import e00483_tails as tails  # ty: ignore[unresolved-import]
import e00500_conformal as conformal  # ty: ignore[unresolved-import]
import polars as pl

LEVELS = conformal.LEVELS


def main() -> None:
    """Write `posthoc.csv` and `shift_means.csv`."""
    base = conformal.load_arm(arm="raw")
    rows = []
    for days in (14, 28):
        sh = pl.read_parquet(conformal.shifts_path(days=days)).select(
            "origin", "primary", "lead_band", "s50"
        )
        j = base.join(sh, on=["origin", "primary", "lead_band"], how="left", maintain_order="left")
        bias = j.with_columns(*[(pl.col(q) + pl.col("s50")).alias(q) for q in LEVELS])
        arms = {f"RC{days}-bias": bias}
        for name, frame in arms.items():
            conformal_frame = frame
            conformal_frame.write_parquet(conformal.WORK / "arms" / f"{name}.parquet")
    frames = {
        a: conformal.load_arm(arm=a) for a in ("raw", "RC14", "RC28", "RC14-bias", "RC28-bias")
    }
    for arm, frame in frames.items():
        for group in ("signed_six", "mva_eight"):
            part = tails.slices(frame=frame)["clean"].filter(pl.col("group") == group)
            s = tails.summarise(frame=part, by=["group"]).row(0, named=True)
            mae = 100.0 * (part["actual"] - part["q50"]).abs().mean()
            cold = tails.slices(frame=frame)["clean_cold"].filter(pl.col("group") == group)
            sc = tails.summarise(frame=cold, by=["group"]).row(0, named=True)
            rows.append(
                {
                    "arm": arm,
                    "group": group,
                    "median_mae": mae,
                    "cold_cov_q95": sc["cov_q95"],
                    "cold_cov_q99": sc["cov_q99"],
                    "cold_crps": sc["crps"],
                    **{
                        k: s[k]
                        for k in (
                            "n",
                            "crps",
                            "cov_q01",
                            "cov_q05",
                            "cov_q95",
                            "cov_q99",
                            "pin_q99",
                        )
                    },
                }
            )
    out = pl.DataFrame(rows)
    out.write_csv(conformal.WORK / "posthoc.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=220):
        print(out.with_columns(pl.col(pl.Float64).round(4)))
    means = []
    for days in conformal.BUFFER_DAYS:
        sh = pl.read_parquet(conformal.shifts_path(days=days)).with_columns(days=pl.lit(days))
        means.append(
            sh.group_by("days", "lead_band", "origin").agg(
                pl.col("s50", "s95", "s99", "s05").mean()
            )
        )
    pl.concat(means).write_csv(conformal.WORK / "shift_means.csv")


if __name__ == "__main__":
    main()

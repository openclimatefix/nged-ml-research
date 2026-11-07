# ruff: noqa: D103, E501
"""Experiment 612: absorption tables (CPU): total and refit shares of K1/K2 (experiment 599's E3 arm), K3 with D5 and E3, and the class 2 candidate.

Reads `absorb/event_shares_{k3d5,c2}.parquet` (written by `e00612_absorb_score.py`) and experiment 599's
`scores/event_shares.parquet`, whose arm `E3` is K1 and K2's refit rule (the lag network with the 7-day
embargo and half weight on the next 7 days; K1 and K2 share these fits). One value per injected design (the mean of
its two Primaries); contrasts are paired by design with a Bonferroni level over the counted contrasts
(the day-14 refit, 7 days after the end, total share at q50; two contrasts: K3 and class 2 against E3).

Writes `tables_out/t6_absorption.csv` and `t6_absorption_contrasts.csv`.
"""

import e00595_summary as s595  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00612-nra-combined-networks-three-class-roster-completion"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
R599 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00599-nra-refit-embargo-and-down-weighting-of-the-newest-rows-against-load-shift-absorption"
    / "scores"
    / "event_shares.parquet"
)
LABELS = {
    "E3": "K1 and K2 (lag network, E3)",
    "k3d5": "K3 (lag-free, D5, E3)",
    "c2": "Class 2 candidate (K2 + neighbour summaries + dropout)",
}
TOL = 0.03
N_TESTS = 2


def main() -> None:
    frames = [pl.read_parquet(R599).filter(pl.col("arm") == "E3")]
    for kind in ("k3d5", "c2"):
        f = ROOT / "absorb" / f"event_shares_{kind}.parquet"
        if f.exists():
            frames.append(pl.read_parquet(f).with_columns(arm=pl.lit(kind)))
    shares = pl.concat(frames, how="diagonal_relaxed")
    rows = []
    for value in ("total_q50", "refit_q50"):
        d = s595.per_design(shares=shares, value=value)
        for arm in d["arm"].unique(maintain_order=True).to_list():
            for refit_day in (14, 0):
                for phase in s595.PHASES[1:]:
                    v = d.filter(
                        (pl.col("arm") == arm)
                        & (pl.col("refit_day") == refit_day)
                        & (pl.col("phase") == phase)
                    )[value].to_numpy()
                    if len(v) < 2:
                        continue
                    m, lo, hi = s595.interval(values=v)
                    rows.append({"network": LABELS[arm], "share": value, "refit_day": refit_day, "phase": phase, "mean": m, "lo": lo, "hi": hi, "n_designs": len(v)})  # fmt: skip
    pl.DataFrame(rows).write_csv(ROOT / "tables_out" / "t6_absorption.csv")
    d = s595.per_design(shares=shares, value="total_q50")
    contr = []
    for arm in ("k3d5", "c2"):
        for phase in ("day_4_to_end", "after_7_days"):
            x = d.filter((pl.col("refit_day") == 14) & (pl.col("phase") == phase))
            j = x.filter(pl.col("arm") == arm).join(
                x.filter(pl.col("arm") == "E3"), on="case", suffix="_b"
            )
            if j.height < 2:
                continue
            counted = phase == "after_7_days"
            m, lo, hi = s595.interval(
                values=(j["total_q50"] - j["total_q50_b"]).to_numpy(),
                level=0.05 / N_TESTS if counted else 0.05,
            )
            verdict = (
                "better"
                if hi < -TOL
                else "worse"
                if lo > TOL
                else "equivalent"
                if lo >= -TOL and hi <= TOL
                else "inconclusive"
            )
            contr.append({"contrast": f"{LABELS[arm]} minus E3 (K1/K2)", "phase": phase, "diff": m, "lo": lo, "hi": hi, "mde": (hi - lo) / 2, "n_designs": j.height, "counted": counted, "tolerance": TOL, "verdict": verdict})  # fmt: skip
    out = pl.DataFrame(contr)
    out.write_csv(ROOT / "tables_out" / "t6_absorption_contrasts.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, fmt_str_lengths=60):
        print(pl.DataFrame(rows).filter(pl.col("refit_day") == 14))
        print(out)


if __name__ == "__main__":
    main()

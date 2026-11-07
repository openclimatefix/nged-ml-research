# ruff: noqa: D103, E501
"""Experiment 613: the leave-one-out table. Each arm minus K1, with Bonferroni intervals (CPU).

For every scored arm (folder `score/<arm>/<arm>`), three paired contrasts against K1 (`k1_d5`):

- G4 coverage of the normal running arrangement inside q05-q95, 24-120 h, injected events of 15 and 30
  days, in the phase day 4 to the end, and in the 7 days after the end (resampling events: a Primary,
  month, and pass);
- median follow in day 4 to the end;
- event-free CRPS, 24-120 h, both metering groups (resampling Primary-month cells).

Also written: each arm's event-free q05-q95 coverage and the verdict for each contrast against its
tolerance (needed, removable, or inconclusive; the rule of the plan).

Usage: python e00613_summary.py [--months ...]
"""

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00613-nra-roster-contrasts-with-intervals-five-seeds"
SCORE = ML_EXPERIMENTS_DATA_DIR / NAME / "score"
REF = "k1_d5"
ARMS = {
    "k1_d5": "K1 (reference)",
    "nosyn_d5": "no synthetic shifts",
    "nolagid_d5": "no lag ID",
    "noexp_d5": "no expected embeddings",
    "noage_d5": "no lag age",
    "c1_d5": "C1 wiring in place of A1",
    "nospread_d5": "no spread inputs",
    "k1_o2": "O2 in place of D5",
}
TOL = {"cover_days4": 0.01, "cover_after": 0.01, "follow": 0.01}
N_BOOT = 4000
RNG = np.random.default_rng(608)
GROUPS = ("MVA-only eight", "signed-MW six")


def interval(*, diffs: np.ndarray, n_counted: int) -> tuple[float, float, float]:
    """Return the mean and its Bonferroni bootstrap interval."""
    idx = RNG.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    means = diffs[idx].mean(axis=1)
    a = 0.05 / n_counted
    return (
        float(diffs.mean()),
        float(np.percentile(means, 100 * a / 2)),
        float(np.percentile(means, 100 * (1 - a / 2))),
    )


def bank(*, arm: str) -> pl.DataFrame:
    t = pl.read_csv(SCORE / arm / arm / "bank_event_phase.csv").filter(
        (pl.col("system") == "G4")
        & (pl.col("band") == "24-120 h")
        & (pl.col("length_days") >= 15)
        & pl.col("phase").is_in(["days4-end", "after0-7"])
    )
    return t.select("month", "pass", "primary", "phase", "cover90", "follow_q50")


def verdict(*, lo: float, hi: float, tol: float, worse_is_negative: bool) -> str:
    worse_lo, worse_hi = (lo, hi) if worse_is_negative else (-hi, -lo)
    if worse_hi < -tol:
        return "needed"
    if worse_lo > -tol and worse_hi < tol:
        return "removable"
    return "inconclusive"


def main() -> None:
    ref_bank = bank(arm=REF)
    ref_rows = np.load(SCORE / REF / REF / "event_free_rows.npz")
    arms = [a for a in ARMS if a != REF and (SCORE / a / a / "bank_event_phase.csv").exists()]
    n_counted = 4 * len(arms)
    out = []
    for arm in arms:
        both = bank(arm=arm).join(ref_bank, on=["month", "pass", "primary", "phase"], suffix="_ref")
        rows = np.load(SCORE / arm / arm / "event_free_rows.npz")
        rec = {"arm": ARMS[arm], "n_counted": n_counted}
        for phase, key in (("days4-end", "cover_days4"), ("after0-7", "cover_after")):
            d = both.filter(pl.col("phase") == phase)
            diffs = (d["cover90"] - d["cover90_ref"]).to_numpy()
            m, lo, hi = interval(diffs=diffs, n_counted=n_counted)
            rec[key], rec[f"{key}_lo"], rec[f"{key}_hi"] = m, lo, hi
            rec[f"{key}_verdict"] = verdict(lo=lo, hi=hi, tol=TOL[key], worse_is_negative=True)
        d = both.filter(pl.col("phase") == "days4-end")
        m, lo, hi = interval(
            diffs=(d["follow_q50"] - d["follow_q50_ref"]).to_numpy(), n_counted=n_counted
        )
        rec["follow"], rec["follow_lo"], rec["follow_hi"] = m, lo, hi
        rec["follow_verdict"] = verdict(lo=lo, hi=hi, tol=TOL["follow"], worse_is_negative=False)
        # event-free CRPS, paired on rows, cells = (Primary, month)
        a = pl.DataFrame(
            {
                k: ref_rows[k]
                for k in ("prim", "origin_idx", "idx", "band", "free", "group", "month", "crps_g4")
            }
        )
        b = pl.DataFrame({k: rows[k] for k in ("prim", "origin_idx", "idx", "crps_g4")})
        j = a.join(b, on=["prim", "origin_idx", "idx"], suffix="_arm").filter(
            pl.col("free") & (pl.col("band") == 1) & pl.col("group").is_in(GROUPS)
        )
        cells = j.group_by("prim", "month").agg(
            (pl.col("crps_g4_arm") - pl.col("crps_g4")).mean().alias("d")
        )
        m, lo, hi = interval(diffs=cells["d"].to_numpy(), n_counted=n_counted)
        rec["crps"], rec["crps_lo"], rec["crps_hi"], rec["crps_mde"] = m, lo, hi, (hi - lo) / 2
        ef = pl.read_csv(SCORE / arm / arm / "event_free.csv").filter(
            (pl.col("group") == "both groups")
            & (pl.col("band") == "24-120 h")
            & (pl.col("system") == "G4")
        )
        rec["event_free_cover90"] = float(ef["cover90"][0])
        out.append(rec)
    ref_ef = pl.read_csv(SCORE / REF / REF / "event_free.csv").filter(
        (pl.col("group") == "both groups")
        & (pl.col("band") == "24-120 h")
        & (pl.col("system") == "G4")
    )
    ref_b = ref_bank.group_by("phase").agg(pl.col("cover90").mean(), pl.col("follow_q50").mean())
    table = pl.DataFrame(out, infer_schema_length=None)
    table.write_csv(SCORE / "leave_one_out.csv")
    print("K1 reference:", ref_b.to_dicts(), "event-free cover90", float(ref_ef["cover90"][0]))
    with pl.Config(tbl_rows=30, tbl_cols=30, fmt_str_lengths=40):
        print(
            table.select(
                "arm",
                "cover_days4",
                "cover_days4_lo",
                "cover_days4_hi",
                "cover_days4_verdict",
                "cover_after",
                "cover_after_verdict",
                "follow",
                "follow_verdict",
                "crps",
                "crps_mde",
                "event_free_cover90",
            )
        )


if __name__ == "__main__":
    main()

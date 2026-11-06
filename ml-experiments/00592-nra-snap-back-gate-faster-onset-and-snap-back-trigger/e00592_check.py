# ruff: noqa: E501, D103
"""Experiment 592 check: G0 under this experiment's scorer reproduces experiment 584's C2 (A2) on C1."""

import e00592_common as g  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OLD = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6/score_c1"
KEYS = ["month", "pass", "primary", "length_days", "phase", "band"]


def main() -> None:
    new = pl.read_csv(g.WORK / "bank_event_phase.csv").filter(pl.col("arm") == "G0")
    old = pl.read_csv(OLD / "bank_event_phase.csv").filter(pl.col("arm") == "A2")
    j = new.join(old, on=KEYS, suffix="_584")
    d = (j["cover90"] - j["cover90_584"]).abs()
    print("bank rows", new.height, old.height, j.height, "max |cover90 difference|", d.max())
    new_r = pl.read_csv(g.WORK / "real_event_phase.csv").filter(pl.col("gate") == "G0")
    old_r = pl.read_csv(OLD / "real_event_phase.csv").filter(pl.col("arm") == "A2")
    jr = new_r.join(old_r, on=["event", "phase", "band"], suffix="_584")
    print(
        "real rows",
        jr.height,
        "max |cover90 difference|",
        (jr["cover90"] - jr["cover90_584"]).abs().max(),
    )
    assert d.max() < 1e-4, "G0 does not reproduce experiment 584's C2"


if __name__ == "__main__":
    main()

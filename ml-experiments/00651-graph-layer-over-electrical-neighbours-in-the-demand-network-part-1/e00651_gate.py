# ruff: noqa: E501, D103, ANN001, ANN201, S101, INP001
"""Experiment 651: the pre-registered gate to the confirmation experiment, checked by script.

Reads the tables of `e00651_score.py` for banks M, S, and X. Systems are the arms rescaled to the
Candidate-gate's (K1's) band width per month, the `-rs` systems, as experiment 613 scored its Candidate.
An arm X in {GCN-E, GAT-E, FLOW-E} passes only if all hold:

(i) follow on bank M at most C's + 0.01, and on bank S at most C's;
(ii) event-free CRPS at the six MW Primaries (24-120 h) below C's by at least 0.044 points of capacity,
     or follow on bank S below C's by at least 0.03;
(iii) X beats GAT-R on the measure that passed (ii) by at least half the minimum detectable effect
      (0.022 points of capacity in CRPS, 0.015 in follow).

Usage: python e00651_gate.py [--raw]  (--raw reads the arms at their own band width)
"""

import argparse
import json

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = (
    ML_EXPERIMENTS_DATA_DIR
    / "00651-graph-layer-over-electrical-neighbours-in-the-demand-network-part-1"
)
CANDIDATES = ("GCN-E", "GAT-E", "FLOW-E")
MDE_CRPS_SIX, MDE_FOLLOW = 0.044, 0.03
BAND, SIX = "24-120 h", "signed-MW six"


def follow(*, bank: str, system: str) -> float:
    t = pl.read_csv(ROOT / "score" / bank / "arm_summary.csv")
    return float(t.filter(pl.col("system") == system)["follow_q50_days4-end"][0])


def crps_six(*, system: str) -> float:
    t = pl.read_csv(ROOT / "score" / "M" / "event_free.csv")
    r = t.filter((pl.col("system") == system) & (pl.col("group") == SIX) & (pl.col("band") == BAND))
    return float(r["crps"][0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", action="store_true")
    suffix = "" if p.parse_args().raw else "-rs"
    rows = []
    c_fm, c_fs, c_crps = (
        follow(bank="M", system="C" + suffix),
        follow(bank="S", system="C" + suffix),
        crps_six(system="C" + suffix),
    )
    r_fs, r_crps = follow(bank="S", system="GAT-R" + suffix), crps_six(system="GAT-R" + suffix)
    for x in CANDIDATES:
        sx = x + suffix
        fm, fs, cr = follow(bank="M", system=sx), follow(bank="S", system=sx), crps_six(system=sx)
        i_ok = fm <= c_fm + 0.01 and fs <= c_fs
        ii_crps = c_crps - cr >= MDE_CRPS_SIX
        ii_follow = c_fs - fs >= MDE_FOLLOW
        iii_crps = r_crps - cr >= MDE_CRPS_SIX / 2
        iii_follow = r_fs - fs >= MDE_FOLLOW / 2
        passed = i_ok and ((ii_crps and iii_crps) or (ii_follow and iii_follow))
        rows.append({
            "arm": x, "follow_M": fm, "follow_S": fs, "ef_crps_six": cr,
            "C_follow_M": c_fm, "C_follow_S": c_fs, "C_ef_crps_six": c_crps,
            "GATR_follow_S": r_fs, "GATR_ef_crps_six": r_crps,
            "i": i_ok, "ii_crps": ii_crps, "ii_follow": ii_follow, "iii_crps": iii_crps,
            "iii_follow": iii_follow, "passes": passed,
        })  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv(ROOT / "score" / f"gate{suffix or '_raw'}.csv")
    print(out)
    result = {
        "systems": "rs" if suffix else "raw",
        "passing_arms": [r["arm"] for r in rows if r["passes"]],
    }
    (ROOT / "score" / f"gate{suffix or '_raw'}.json").write_text(json.dumps(result, indent=1))
    print("GATE", result)


if __name__ == "__main__":
    main()

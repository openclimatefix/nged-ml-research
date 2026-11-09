# ruff: noqa: D103, E501
"""Experiment 663: freeze every candidate before the final-test window is scored.

Hashes the final-fold fits and window forecasts of E0, E0d and J2d, the frozen experiment 654 J2
networks and forecasts, and the scoring code, and writes `final/freeze2.json`. `--verify` checks
the stored hashes (the window scorer calls it first).

Usage: python e00663_freeze.py [--verify]
"""

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import e00663_common as c663

OUT = c663.WORK / "final"
HERE = Path(__file__).parent


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def files() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for sd in (0, 1):
        for arm in ("E0", "E0d"):
            out[f"fits_e0/{arm}_final_s{sd}.npz"] = c663.WORK / "fits_e0" / f"{arm}_final_s{sd}.npz"
            out[f"fits_e0/{arm}_final_s{sd}.pt"] = c663.WORK / "fits_e0" / f"{arm}_final_s{sd}.pt"
        out[f"fits_j2/J2d_final_s{sd}.npz"] = c663.WORK / "fits_j2" / f"J2d_final_s{sd}.npz"
        out[f"fits_j2/J2d_final_s{sd}.pt"] = c663.WORK / "fits_j2" / f"J2d_final_s{sd}.pt"
        out[f"j2_final/forecasts/J2_s{sd}.parquet"] = (
            c663.WORK / "j2_final" / "forecasts" / f"J2_s{sd}.parquet"
        )
        out[f"654/final/models/J2_s{sd}.pt"] = c663.E654 / "final" / "models" / f"J2_s{sd}.pt"
        out[f"654/final/forecasts/J2_s{sd}.parquet"] = (
            c663.E654 / "final" / "forecasts" / f"J2_s{sd}.parquet"
        )
    out["window/real_window.npz"] = c663.WORK / "window" / "real_window.npz"
    out["det_inputs_v2/final_window.npz"] = c663.WORK / "det_inputs_v2" / "final_window.npz"
    for code in (
        "e00663_window_score.py",
        "e00663_fit_e0.py",
        "e00663_fit_j2d.py",
        "e00663_j2_final.py",
        "e00663_panel.py",
        "e00663_window_data.py",
    ):
        out[f"code/{code}"] = HERE / code
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    f = OUT / "freeze2.json"
    if a.verify:
        stored = json.loads(f.read_text())["sha256_16"]
        bad = {k: (v, h(files()[k])) for k, v in stored.items() if h(files()[k]) != v}
        if bad:
            print("hash mismatch", bad)
            sys.exit(3)
        print("freeze2 verified", len(stored), "files")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    assert not f.exists(), "freeze2.json already written"
    out = {
        "frozen_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": ["2026-07-01", "2026-10-31"],
        "rule": "Candidates chosen on development months only (2025-10 to 2026-06). The window is scored once; nothing is tuned on it.",
        "candidates": {
            "E0": "frozen K1 + R-haz of experiment 652, refitted on all five development months (fits_e0/E0_final_s0,1)",
            "E0d": "E0 plus the end detector on the real neighbour graph, final fold (fits_e0/E0d_final_s0,1)",
            "J2": "experiment 654's J2 networks, not refitted; forward pass only (j2_final), same forecasts as experiment 654 (654/final/forecasts/J2_s0,1)",
            "J2d": "J2 plus the end detector, real neighbour graph, final fold (fits_j2/J2d_final_s0,1)",
            "reused_not_rescored": ["K1", "manual_heuristic"],
            "not_frozen_for_the_window": "rewired-graph placebo arms and the no-neighbour arm are development-month contrasts only",
        },
        "scenarios": "all, outside, inside, after7: the G4 gate of experiment 620 marks event half-hours (labels and detections are empty in the window); counts are lower bounds",
        "scoring_spec": "e00663_window_score.py: lead 24-240 h; CRPS and NMAE of the median in points of capacity; departure share = median of prevailing departs from median of NRA by 3 points or more",
        "sha256_16": {k: h(p) for k, p in files().items()},
    }
    f.write_text(json.dumps(out, indent=1))
    print("wrote", f)


if __name__ == "__main__":
    main()

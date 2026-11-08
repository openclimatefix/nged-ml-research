# ruff: noqa: E501, D103, ANN001, ANN201, S101, INP001
"""Experiment 651: score the graph arms and the stored Candidate on one injected bank (no fit, no GPU).

A thin wrapper around experiment 613's scorer (`e00613_nbscore`, which this experiment imports without
change): the same systems, width factors, event-free table, and bank table. What differs is the arm
list, the data folders, and the bank: M (experiment 581's, the `inject` folders), S (shift split 60/40
over two neighbours, `inject_S`), or X (no mirror, `inject_X`).

Usage: python e00651_score.py --bank M|S|X [--arms ...] [--months ...] [--seeds ...]
"""

import argparse
from pathlib import Path

import e00597_common as k597  # ty: ignore[unresolved-import]
import e00613_nbscore as nbs  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00651-graph-layer-over-electrical-neighbours-in-the-demand-network-part-1"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
ROOT612 = ML_EXPERIMENTS_DATA_DIR / "00612-nra-combined-networks-three-class-roster-completion"
ARMS: dict[str, tuple[Path, str]] = {
    "C": (ROOT612 / "P7_D50_e3", "the stored class 2 candidate (experiment 612, seeds 0 and 1)"),
    "G0": (ROOT / "G0_D50", "own gap signals only"),
    "GCN-E": (ROOT / "G0_D50_GCN", "graph convolution on the electrical edges"),
    "GAT-E": (ROOT / "G0_D50_GAT", "attention on the electrical edges"),
    "GAT-R": (ROOT / "G0_D50_GATR", "attention on a rewired graph (placebo)"),
    "GAT-F": (ROOT / "G0_D50_GATF", "attention over all Primaries"),
    "FLOW-E": (ROOT / "G0_D50_FLOW", "antisymmetric edge flows in megawatts"),
}


class GraphArm(nbs.Arm):
    """A stored arm whose work folder and injected-bank folder are named explicitly."""

    def __init__(
        self, *, name: str, folder: Path, bank: str, months: list[str], seeds: list[int]
    ) -> None:
        super().__init__(name=name, suffix="", months=months, seeds=seeds)
        self.work = folder / "work"
        self.bank_dir = "inject" if bank == "M" else f"inject_{bank}"
        # the stored Candidate's banks S and X are injected by this experiment, into its own folder
        self.bank_work = ROOT / "P7_D50" / "work" if name == "C" and bank != "M" else self.work

    def inject(
        self, *, t, month: str, r: int, rows: np.ndarray, order, sorted_keys
    ) -> dict[int, np.ndarray]:
        out = {}
        for seed in self.seeds:
            files = [
                self.bank_work / self.bank_dir / f"{month}_r{r:02d}_d{d}_s{seed}.parquet"
                for d in (0, 14)
            ]
            full = nbs.load_rows(files=files, t=t, order=order, sorted_keys=sorted_keys)
            out[seed] = full[rows]
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", choices=("M", "S", "X"), default="M")
    p.add_argument("--months", nargs="+", default=["2025-10", "2025-12"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    args = p.parse_args()
    out = ROOT / "score" / args.bank
    out.mkdir(parents=True, exist_ok=True)
    nbs.s.N_COUNTED = nbs.N_COUNTED
    inp = k597.Inputs(arm="a1")
    t = inp.t
    kk = nbs.s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk)
    sorted_keys = kk[order]
    arms = {}
    for name in args.arms:
        arm = GraphArm(
            name=name, folder=ARMS[name][0], bank=args.bank, months=args.months, seeds=args.seeds
        )
        if arm.available():
            arm.load(t=t, order=order, sorted_keys=sorted_keys)
            arms[name] = arm
        else:
            print("arm", name, "has no complete fits under", arm.work, "- skipped", flush=True)
    sy = nbs.Systems(inp=inp, arms=arms, months=args.months, seeds=args.seeds, factor_source="bank")
    sy.factors.write_csv(out / "factors.csv")
    ef = nbs.event_free(sy=sy)
    ef.write_csv(out / "event_free.csv")
    tab, _ = nbs.score_bank(sy=sy)
    tab.write_csv(out / "bank_event_phase.csv")
    nbs.summarise_bank(tab=tab).write_csv(out / "bank_summary.csv")
    nbs.arm_summary(tab=tab, ef=ef, systems=sorted(tab["system"].unique().to_list())).write_csv(
        out / "arm_summary.csv"
    )
    print("scored", sorted(arms), "bank", args.bank, "->", out, flush=True)


if __name__ == "__main__":
    main()

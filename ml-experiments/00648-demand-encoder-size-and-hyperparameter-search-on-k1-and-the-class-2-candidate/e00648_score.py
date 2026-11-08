# ruff: noqa: E402, S102, D103
"""Experiment 648: score one screen arm with experiment 613's scorer (CPU).

Runs experiment 613's `e00613_score.py` with the arm's folder and the output folder in this
experiment's private data folder. The lag-free twin of the G4 gate and of the follow ratio is
experiment 613's `k3_d5` (seed 0), read only, and is the same for every arm. The scorer's seeds, months,
and systems are experiment 613's; only the arm folder and the output folder move.

Usage: python e00648_score.py OUT --arm NAME [--months ...] [--seeds 0]
"""

import sys
from pathlib import Path

NAME613 = "00613-nra-roster-contrasts-with-intervals-five-seeds"
NAME648 = "00648-demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate"
EXP613 = Path(__file__).resolve().parents[1] / NAME613
path = EXP613 / "e00613_score.py"
src = path.read_text()
for old, new in (
    ('OUT = ROOT / "score" / MINE.out', 'OUT = ROOT.parent / NAME648 / "score" / MINE.out'),
    ('work = ROOT / "arms" / MINE.arm / "work"', 'work = ROOT.parent / NAME648 / "arms" / MINE.arm / "work"'),
):
    assert old in src, old
    src = src.replace(old, new)
src = src.replace("import argparse", f'NAME648 = "{NAME648}"\nimport argparse', 1)
sys.argv = [path.name, *sys.argv[1:]]
if "--seeds" not in sys.argv:
    sys.argv += ["--seeds", "0"]
if "--months" not in sys.argv:
    sys.argv += ["--months", "2025-10", "2026-01", "2026-06"]
sys.path.insert(0, str(EXP613))
exec(compile(src, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})

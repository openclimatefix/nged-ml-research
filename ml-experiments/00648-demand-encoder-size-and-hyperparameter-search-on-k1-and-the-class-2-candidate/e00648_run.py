# ruff: noqa: E501, S102, ANN201
"""Experiment 648: run one arm of the one-factor screen (GPU).

A launcher around experiment 613's `e00613_arm.py` (K1) and `e00613_nbarm.py` (the class 2 candidate,
arm `P7+D50`). It patches the module constants of the network before either script imports its
modules, then runs the script's own source with its output folder moved to this experiment's private
data folder (experiment 613's folder is read, never written, except that the candidate still reads
experiment 613's, 610's, and 598's signal tables).

Usage: python e00648_run.py ARM NET COMMAND [--months ...] [--seeds ...]

- ARM: a key of `ARMS` (`ref` is the unchanged network);
- NET: `k1` (COMMAND `pretrain`, `step2`, `fit`, `inject`) or `c2` (COMMAND `fit`, `inject`); the
  candidate reads the pre-training and step-2 weights that the K1 commands wrote for the same ARM
  under `chains/d5_ARM/work`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import e00397_models as base  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00559_model as m559  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00613_nbmodel as nbmodel  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME648 = "00648-demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate"
NAME613 = "00613-nra-roster-contrasts-with-intervals-five-seeds"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME648

ARMS: dict[str, dict[str, float | int | str]] = {
    "ref": {},
    "lag32": {"lag_hidden": 32},
    "lag128": {"lag_hidden": 128},
    "lagdeep": {"lag_layers": 2},
    "nbbn8": {"bottleneck": 8, "bn": 1},
    "nbbn16": {"bottleneck": 16, "bn": 1},
    "nbbn32": {"bottleneck": 32, "bn": 1},
    "comp5": {"components": 5},
    "hid128": {"hidden": 128},
    "hid256": {"hidden": 256},
    "hid384": {"hidden": 384},
    "drop0": {"dropout": 0.0},
    "drop2": {"dropout": 0.2},
    "lrhalf": {"lr_scale": 0.5},
    "lrdouble": {"lr_scale": 2.0},
    "epoch2": {"epoch_scale": 2},
}
"""Per arm, the one setting changed (`nbbn16` is the bottleneck reference for `nbbn8` and `nbbn32`; the learning-rate scale does not reach `FRESH_LR`, the step-2 mixer's rate) (the reference setting is in experiment 613's modules)."""

_p = argparse.ArgumentParser()
_p.add_argument("arm", choices=sorted(ARMS))
_p.add_argument("net", choices=("k1", "c2"))
_p.add_argument("command", choices=("pretrain", "step2", "fit", "inject", "assemble"))
_p.add_argument("--months", nargs="+", default=["2025-10", "2026-01", "2026-06"])
_p.add_argument("--seeds", nargs="+", type=int, default=[0])
MINE = _p.parse_args()
assert max(MINE.months) <= "2026-06", "July 2026 onwards is never opened"
SET = ARMS[MINE.arm]
assert MINE.net == "k1" or MINE.command in {"fit", "inject", "assemble"}, (
    "c2 runs fit, inject, and assemble"
)
if SET.get("bn"):
    assert MINE.net == "c2", "the neighbour-summary bottleneck exists only in the candidate"

# ---- patch the constants before the scripts import their modules
for mod in (m545, m559):
    if "hidden" in SET:
        mod.ENCODER_HIDDEN = mod.MIXER_HIDDEN = int(SET["hidden"])  # ty: ignore[invalid-assignment]
    if "dropout" in SET:
        mod.DROPOUT = float(SET["dropout"])  # ty: ignore[invalid-assignment]
    if "components" in SET:
        mod.COMPONENTS = int(SET["components"])  # ty: ignore[invalid-assignment]
if "components" in SET:
    c.COMPONENTS = int(SET["components"])  # ty: ignore[invalid-assignment]
if "lag_hidden" in SET:
    m560.LAG_HIDDEN = int(SET["lag_hidden"])  # ty: ignore[invalid-assignment]
if "bottleneck" in SET:
    nbmodel.BOTTLENECK_HIDDEN = int(SET["bottleneck"])  # ty: ignore[invalid-assignment]
if "lr_scale" in SET:
    c.FINE_TUNE_LR = c.FINE_TUNE_LR * float(SET["lr_scale"])  # ty: ignore[invalid-assignment]
    c.PRETRAIN_LR = c.PRETRAIN_LR * float(SET["lr_scale"])  # ty: ignore[invalid-assignment]
if "epoch_scale" in SET:
    c.MAX_EPOCHS = c.MAX_EPOCHS * int(SET["epoch_scale"])  # ty: ignore[invalid-assignment]
    c.PATIENCE = c.PATIENCE * int(SET["epoch_scale"])  # ty: ignore[invalid-assignment]
if "lag_layers" in SET:
    _mlp = base.mlp

    def mlp(*, n_in: int, hidden: int, layers: int, n_out: int, dropout: float = 0.0):
        """Return `_mlp`, with one more hidden layer for the lag encoder (the call with the wide input)."""
        is_lag_encoder = n_in >= 60 and n_out == m560.LAG_OUT and layers == 1
        return _mlp(
            n_in=n_in,
            hidden=hidden,
            layers=int(SET["lag_layers"]) if is_lag_encoder else layers,
            n_out=n_out,
            dropout=dropout,
        )

    base.mlp = mlp

# ---- run the experiment 613 script with its output folder moved
CHAIN_ARM = "ref" if SET.get("bn") else MINE.arm  # the bottleneck arms reuse the unchanged chain
CHAIN = ROOT / "chains" / f"d5_{CHAIN_ARM}" / "work"
os.environ["CHAIN613"] = str(CHAIN)
if MINE.command in {
    "pretrain",
    "step2",
}:  # a chain folder left over from other settings would be reused silently
    CHAIN.mkdir(parents=True, exist_ok=True)
    setting = CHAIN.parent / "setting.json"
    if setting.exists():
        assert json.loads(setting.read_text()) == SET, f"{setting} holds other settings"
    else:
        setting.write_text(json.dumps(SET))
if (
    MINE.net == "c2"
):  # the signal tables are experiment 598's and 610's stock-K1 tables; never build one here
    for month in MINE.months:
        for seed in MINE.seeds:
            assert any(
                (ML_EXPERIMENTS_DATA_DIR / n / "tables" / f"T_{month}_s{seed}.npz").exists()
                for n in (
                    NAME613,
                    "00598-nra-gate-against-signals-as-inputs-at-equal-band-width",
                    "00610-nra-neighbour-summary-lag-encoder-inputs-iteration",
                    "00606-nra-lag-encoder-signal-permutations-gap-averages-and-neighbours",
                )
            ), (month, seed)
EXP613 = (
    Path(__file__).resolve().parents[1] / "00613-nra-roster-contrasts-with-intervals-five-seeds"
)
common = ["--months", *MINE.months, "--seeds", *map(str, MINE.seeds)]
if MINE.net == "k1":
    path = EXP613 / "e00613_arm.py"
    argv = [path.name, f"k1_{MINE.arm}", MINE.command, *common]
    src = path.read_text().replace(f'NAME = "{NAME613}"', f'NAME = "{NAME648}"')
    assert NAME648 in src
else:
    path = EXP613 / "e00613_nbarm.py"
    argv = [
        path.name,
        "P7+D50+BN" if SET.get("bn") else "P7+D50",
        MINE.command,
        "--work-suffix",
        f"_{MINE.arm}",
        *common,
    ]
    old = 'e.WORK = ROOT / f"{DIR}{MINE.work_suffix}"'
    assert old in path.read_text()
    src = path.read_text().replace(
        old, f'e.WORK = (ROOT.parent / "{NAME648}") / f"{{DIR}}{{MINE.work_suffix}}"'
    )
sys.argv = argv
sys.path.insert(0, str(EXP613))
exec(compile(src, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})

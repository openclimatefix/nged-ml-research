# ruff: noqa: E501
"""Shared constants and arm definitions for experiment 443 (solar network settings sweep).

Farms appear only as labels A to F. Every table written here lives under `data/ml-experiments/`.
The skeleton is experiment 430's `g0_onepass`; each arm changes the settings below.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00443-solar-network-hyperparameter-and-architecture-sweep"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
PRED_DIR: Final[Path] = DATA_DIR / "predictions"
SCORE_DIR: Final[Path] = DATA_DIR / "scores"
E430_PRED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00430-unified-forecaster-solar-and-pretraining-ablations"
    / "predictions"
)
SKELETON: Final[str] = "g0_onepass"
"""Experiment 430's arm used as the skeleton and the reference of every contrast."""
SEEDS_SCREEN: Final[tuple[int, ...]] = (0, 1, 2)
SEEDS_CONFIRM: Final[tuple[int, ...]] = (3, 4, 5)
SEEDS_SEARCH: Final[tuple[int, ...]] = (0, 1)
MARGIN_PERCENT: Final[float] = 0.5
TAIL_HARM_LIMIT: Final[float] = 0.02
"""Largest allowed rise of the 0.01 and 0.99 pinball loss over the reference, in points of capacity."""
RANDOM_SEARCH_FILE: Final[Path] = DATA_DIR / "random_search_configs.json"


@dataclass(frozen=True)
class Config:
    """One network design. The defaults are the skeleton (experiment 430's `g0_onepass`)."""

    width: int = 64
    depth: int = 2
    embedding: int = 0
    """Width of a learned farm embedding joined to the inputs; 0 is none."""
    per_farm: bool = False
    """Fit one network per farm, each on that farm's rows only."""
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    components: int = 3
    dropout: float = 0.0
    half_life_years: float | None = None
    """Half-life of the row weight by the row's age at the refit; None is equal weights."""


def _arms() -> dict[str, Config]:
    out: dict[str, Config] = {}
    for w in (32, 128, 256):
        out[f"w{w}"] = Config(width=w)
    for d in (1, 3, 4):
        out[f"d{d}"] = Config(depth=d)
    for e in (4, 8, 16):
        out[f"e{e}"] = Config(embedding=e)
    out["perfarm"] = Config(per_farm=True)
    for name, lr in (("lr0005", 5e-4), ("lr001", 1e-3), ("lr004", 4e-3)):
        out[name] = Config(learning_rate=lr)
    for name, wd in (("wd0", 0.0), ("wd001", 1e-3), ("wd01", 1e-2)):
        out[name] = Config(weight_decay=wd)
    for ep in (10, 40):
        out[f"ep{ep}"] = Config(epochs=ep)
    for k in (1, 2, 5):
        out[f"k{k}"] = Config(components=k)
    for name, p in (("do01", 0.1), ("do02", 0.2)):
        out[name] = Config(dropout=p)
    out["hl05"] = Config(half_life_years=0.5)
    out["hl1"] = Config(half_life_years=1.0)
    return out


STAGE1_ARMS: Final[dict[str, Config]] = _arms()
"""The 25 one-factor arms of stage 1, by name."""

FACTORS: Final[dict[str, tuple[str, ...]]] = {
    "width": ("w32", "w128", "w256"),
    "depth": ("d1", "d3", "d4"),
    "embedding": ("e4", "e8", "e16", "perfarm"),
    "learning_rate": ("lr0005", "lr001", "lr004"),
    "weight_decay": ("wd0", "wd001", "wd01"),
    "epochs": ("ep10", "ep40"),
    "components": ("k1", "k2", "k5"),
    "dropout": ("do01", "do02"),
    "recency": ("hl05", "hl1"),
}
"""Stage-1 arms grouped by factor."""


def all_arms() -> dict[str, Config]:
    """Return the stage-1 arms, the skeleton, and any stage-2 configurations written to disk."""
    arms = {"skeleton": Config(), **STAGE1_ARMS}
    if RANDOM_SEARCH_FILE.exists():
        for name, values in json.loads(RANDOM_SEARCH_FILE.read_text()).items():
            arms[name] = Config(
                **{f.name: values[f.name] for f in fields(Config) if f.name in values}
            )
    return arms


def config_dict(config: Config) -> dict:
    """Return the config as a plain dict for JSON."""
    return asdict(config)

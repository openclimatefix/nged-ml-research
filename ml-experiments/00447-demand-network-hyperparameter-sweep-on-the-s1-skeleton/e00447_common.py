# ruff: noqa: E501
"""Shared constants and arm definitions for experiment 447 (demand network settings sweep).

The base network is S1 of experiment 428: the pooled plain mixture-density network of experiment
397 (setting g09) that reads the three origin lags and the surprise at each lag, trained on clean
and injected worlds. Every arm changes the settings below; the reference arm is S1 itself, whose
forecasts experiments 428 (seeds 0 to 2) and 439 (seeds 3 to 5) stored.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00447-demand-network-hyperparameter-sweep-on-the-s1-skeleton"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
WORK_DIR: Final[Path] = DATA_DIR / "work"
E428_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00428-day-window-convolution-attention-encoder-replacing-lagged-power-inputs"
    / "work"
)
E439_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00439-surprise-at-the-lags-confirmation-on-seeds-three-to-five"
    / "work"
)
RANDOM_SEARCH_FILE: Final[Path] = WORK_DIR / "random_search_configs.json"
REFERENCE: Final[str] = "S1"
"""Experiment 428's arm used as the base and the reference of every contrast."""

SCREEN_FOLDS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10")
"""Three test months: summer, winter, autumn. The spring fold (2026-03) is left for the confirmation."""
CONFIRM_FOLDS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03", "2025-12")
SCREEN_SEEDS: Final[tuple[int, ...]] = (0, 1)
CONFIRM_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
STORED_MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
"""The months for which experiments 428 and 439 stored S1 forecasts; 2025-12 has none."""
CRPS_MARGIN: Final[float] = 0.03
"""Smallest CRPS gain, in points of capacity, worth adopting (the agreed demand margin)."""
TAIL_HARM_LIMIT: Final[float] = 0.02
"""Largest rise of the q95 and q99 pinball loss over S1, in points of capacity."""
RANDOM_SEARCH_SIZE: Final[int] = 20
RANDOM_SEARCH_SEED: Final[int] = 447


@dataclass(frozen=True)
class Config:
    """One network design and training recipe. The defaults are S1 (setting g09 of experiment 397)."""

    hidden: int = 64
    layers: int = 2
    embedding: int = 8
    """Width of the learned Primary embedding."""
    components: int = 3
    dropout: float = 0.1
    weight_decay: float = 1e-3
    lr: float = 5e-4
    schedule: str = "constant"
    """`constant`, `cosine` (over the 60-epoch cap, stepped per epoch), or `onecycle` (peak 2 x lr)."""
    batch: int = 8_192
    optimiser: str = "adamw"
    """`adamw`, `adam` (no weight decay), `lion`, or `sgd` (momentum 0.9, Nesterov)."""


def _arms() -> dict[str, Config]:
    out: dict[str, Config] = {}
    for h in (32, 128, 256):
        out[f"w{h}"] = Config(hidden=h)
    for d in (3, 4):
        out[f"d{d}"] = Config(layers=d)
    for e in (4, 16):
        out[f"e{e}"] = Config(embedding=e)
    for k in (2, 5):
        out[f"k{k}"] = Config(components=k)
    for name, p in (("do0", 0.0), ("do02", 0.2)):
        out[name] = Config(dropout=p)
    for name, wd in (("wd1e4", 1e-4), ("wd1e2", 1e-2)):
        out[name] = Config(weight_decay=wd)
    out["lr25e5"] = Config(lr=2.5e-4)
    out["lr1e3"] = Config(lr=1e-3)
    out["cosine"] = Config(schedule="cosine")
    out["onecycle"] = Config(schedule="onecycle")
    for b in (4_096, 16_384):
        out[f"b{b}"] = Config(batch=b)
    out["adam"] = Config(optimiser="adam", weight_decay=0.0)
    out["lion"] = Config(optimiser="lion", lr=1.5e-4, weight_decay=1e-2)
    out["sgd"] = Config(optimiser="sgd", lr=5e-2, weight_decay=1e-3)
    return out


STAGE1_ARMS: Final[dict[str, Config]] = _arms()
"""The 22 one-factor arms of stage 1, by name."""

FACTORS: Final[dict[str, tuple[str, ...]]] = {
    "hidden width": ("w32", "w128", "w256"),
    "depth": ("d3", "d4"),
    "embedding width": ("e4", "e16"),
    "mixture components": ("k2", "k5"),
    "dropout": ("do0", "do02"),
    "weight decay": ("wd1e4", "wd1e2"),
    "learning rate": ("lr25e5", "lr1e3"),
    "learning-rate schedule": ("cosine", "onecycle"),
    "batch size": ("b4096", "b16384"),
    "optimiser": ("adam", "lion", "sgd"),
}
"""Stage-1 arms grouped by factor."""

ARM_PREFIX: Final[str] = "X-"
"""Registered arm names are `X-<name>`, so they never collide with experiment 428's arms."""


def all_configs() -> dict[str, Config]:
    """Return the stage-1 arms and any stage-2 configurations written to disk, by short name."""
    arms = dict(STAGE1_ARMS)
    if RANDOM_SEARCH_FILE.exists():
        for name, values in json.loads(RANDOM_SEARCH_FILE.read_text()).items():
            arms[name] = Config(
                **{f.name: values[f.name] for f in fields(Config) if f.name in values}
            )
    return arms


def config_dict(config: Config) -> dict:
    """Return the config as a plain dict for JSON."""
    return asdict(config)

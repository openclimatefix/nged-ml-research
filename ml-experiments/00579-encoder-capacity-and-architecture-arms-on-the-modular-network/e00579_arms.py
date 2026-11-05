# ruff: noqa: E501
"""The arms of experiment 579: each is the base B (experiment 572's v3+D5) plus one change."""

from dataclasses import dataclass
from typing import Final

from e00579_model import Cfg, Part  # ty: ignore[unresolved-import]

LARGE: Final[Part] = Part(layers=2, hidden=256, out=64)
XLARGE: Final[Part] = Part(layers=3, hidden=512, out=128)


@dataclass(frozen=True)
class Arm:
    """One arm: the network, the extra column kind, whether it re-uses B's stored pre-training, whether encoders unfreeze."""

    tag: str
    cfg: Cfg
    kind: str = "none"  # extra datetime columns: "none", "F1", "F2", "P2"
    reuse_pre: bool = False  # mixer-only arms fine-tune from experiment 572's stored encoders
    unfreeze: bool = False  # arm U: encoders train in fine-tuning at 0.1 times the rate


ARMS: Final[dict[str, Arm]] = {
    a.tag: a
    for a in (
        Arm("B", Cfg(), reuse_pre=True),
        Arm("WL", Cfg(w=LARGE)),
        Arm("DL", Cfg(d=LARGE)),
        Arm("WDL", Cfg(w=LARGE, d=LARGE)),
        Arm("DXL", Cfg(d=XLARGE)),
        Arm("WXL", Cfg(w=XLARGE)),
        Arm("EW", Cfg(emb=16)),
        Arm("ML", Cfg(mixer_layers=3, mixer_hidden=256), reuse_pre=True),
        Arm("K5", Cfg(k=5), reuse_pre=True),
        Arm("K8", Cfg(k=8), reuse_pre=True),
        Arm("RB", Cfg(block="res")),
        Arm("GB", Cfg(block="geglu")),
        Arm("F1", Cfg(n_extra=14), kind="F1"),
        Arm("F2", Cfg(n_extra=15), kind="F2"),
        Arm("P1", Cfg(emb_dt=16)),
        Arm("P2", Cfg(p2=True, n_extra=1), kind="P2"),
        Arm("A1", Cfg(film=True)),
        Arm("A2", Cfg(adapter_rank=4), reuse_pre=True),
        Arm("U", Cfg(), reuse_pre=True, unfreeze=True),
    )
}

"""Experiment 554's modular network (arm M0, stage `ft`) with extra input columns on one encoder.

The extra columns follow the 37 lag-free inputs in the input row. The weather encoder (or the
datetime encoder) gains one input unit per extra column, placed after the Primary embedding and
initialised to zero, so the network's function equals the pretrained network's before fine-tuning.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_BASE: Final[int] = m.N_INPUTS


class ExtraMdn(m.ModularMdn):
    """`ModularMdn` (stage `ft`) whose weather or datetime encoder also reads `n_extra` columns."""

    def __init__(self, *, n_primaries: int, encoder: str, n_extra: int) -> None:
        """Build the network; `encoder` is `weather` or `datetime`."""
        super().__init__(n_primaries=n_primaries, stage="ft")
        assert encoder in {"weather", "datetime"}
        self.encoder, self.n_extra = encoder, n_extra
        target = self.weather if encoder == "weather" else self.datetime
        old = target[0]
        assert isinstance(old, nn.Linear)
        wide = nn.Linear(old.in_features + n_extra, old.out_features)
        with torch.no_grad():
            wide.weight.zero_()
            wide.weight[:, : old.in_features] = old.weight
            wide.bias.copy_(old.bias)
        target[0] = wide

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        extra = x[:, N_BASE : N_BASE + self.n_extra]
        weather_in = torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)
        datetime_in = torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)
        if self.encoder == "weather":
            weather_in = torch.cat([weather_in, extra], dim=-1)
        else:
            datetime_in = torch.cat([datetime_in, extra], dim=-1)
        hw = nn.functional.gelu(self.weather(weather_in))
        hd = nn.functional.gelu(self.datetime(datetime_in))
        core = torch.cat([hw, hd, emb], dim=-1)
        return self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))


def load_pretrained_padded(*, model: ExtraMdn, path: str, fresh_seed: int) -> None:
    """Load experiment 554's pretrained M0 weights, padding the widened first layer with zeros.

    The pretrained fine-tuning mixer is discarded and a fresh one is initialised with `fresh_seed`,
    exactly as in experiment 554's `load_pretrained` for arm M0.
    """
    raw = torch.load(path, map_location="cpu")
    own = model.state_dict()
    state = {}
    for name, value in raw.items():
        if name.startswith("mixer_ft."):
            continue
        if value.shape != own[name].shape:
            assert name in {"weather.0.weight", "datetime.0.weight"}, name
            padded = torch.zeros_like(own[name])
            padded[:, : value.shape[1]] = value
            state[name] = padded
        else:
            state[name] = value
    missing = model.load_state_dict(state, strict=False)
    assert all(a.startswith("mixer_ft.") for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fresh_seed)
        fresh = base.mlp(
            n_in=model.mixer_ft[0].in_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
            hidden=m.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m.COMPONENTS,
            dropout=m.DROPOUT,
        )
        base.init_head(head=fresh, components=m.COMPONENTS)
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]

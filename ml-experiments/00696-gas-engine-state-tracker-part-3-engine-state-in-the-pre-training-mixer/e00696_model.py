# ruff: noqa: E501
"""Experiment 696: experiment 591's network with a pre-training mixer that reads the engine state.

`StateAuxLagMdn` is `e00591_model.AuxLagMdn` with the first layer of `mixer_pre` widened by 3
inputs, at every stage, so that the pre-training weights load into the stage 2 and stage 3 networks strictly.

- **The widened layer starts as the original.** The layer is built at the original width under the
  original random-number stream, and 3 zero columns are concatenated after the draw, so every
  weight of the original columns and every part built after the layer is identical to
  `AuxLagMdn`'s. With zero state inputs the extra weights receive no gradient and no weight
  decay, so they stay at zero.
- **Only a pre-training pass reads the columns.** A pre-training forward pass whose input carries 3
  trailing columns strips them, and a forward-pre hook on `mixer_pre` appends them to the mixer's
  input. A pass with no such columns appends zeros. A stage 2 or stage 3 pass never calls
  `mixer_pre`, and a forecast-time pass that carries state columns raises.
"""

from typing import Final

import e00545_model as m545  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00591_model as base_module  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from e00696_common import N_STATE
from torch import nn  # ty: ignore[unresolved-import]

ORIGINAL_CLASS: Final = base_module.AuxLagMdn
"""Experiment 591's network class, captured before any patch replaces it on its module."""
PROBE_WIDTH: Final[int] = 64


def widen_first_layer(*, layer: nn.Linear, extra: int) -> None:
    """Append `extra` zero input columns to a linear layer without drawing any random number."""
    with torch.no_grad():
        wide = torch.cat([layer.weight, layer.weight.new_zeros(layer.out_features, extra)], dim=1)
    layer.weight = nn.Parameter(wide)
    layer.in_features += extra


class StateAuxLagMdn(ORIGINAL_CLASS):
    """Experiment 591's network whose pre-training mixer also reads the engine-state columns."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, kind: str,
        detector: bool = False,
    ) -> None:  # fmt: skip
        """Build the network exactly as `AuxLagMdn`, then widen `mixer_pre` and hook the state in."""
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, kind=kind,
            detector=detector,
        )  # fmt: skip
        widen_first_layer(layer=self.mixer_pre[0], extra=N_STATE)  # ty: ignore[invalid-argument-type, not-subscriptable]
        self._state: torch.Tensor | None = None
        self.mixer_pre.register_forward_pre_hook(self._append_state)  # ty: ignore[unresolved-attribute]

    def _append_state(self, module: nn.Module, args: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """Append the pass's state columns (zeros when it has none) to the mixer's input."""
        _ = module
        (core,) = args
        state = self._state if self._state is not None else core.new_zeros(len(core), N_STATE)
        return (torch.cat([core, state], dim=-1),)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs; a pre-training input with 3 extra trailing columns feeds them to `mixer_pre`."""
        self._state = None
        if x.shape[1] == self.label_col + N_STATE:
            if self.stage != "pre":
                msg = (
                    "a stage 2, stage 3, or forecast-time pass must not carry engine-state columns"
                )
                raise ValueError(msg)
            x, self._state = x[:, : self.label_col], x[:, self.label_col :]
        try:
            return super().forward(x, primary)
        finally:
            self._state = None


def encoder_features(*, model: nn.Module, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
    """Return the weather, date-time, and embedding outputs of an experiment 591 network.

    The output is (rows, 2 * encoder width + embedding width). The probe M1 reads these frozen
    features. No lag input and no state column enters.
    """
    assert x.shape[1] == model.label_col  # ty: ignore[unresolved-attribute]
    emb = model.embedding(primary)  # ty: ignore[call-non-callable]
    weather_in = [x[:, m.WEATHER_COLS], emb]
    if model.n_b1:  # ty: ignore[unresolved-attribute]
        weather_in.append(x[:, base_module.mod.b1_cols(n_b1=model.n_b1)])  # ty: ignore[unresolved-attribute]
    hw = nn.functional.gelu(model.weather(torch.cat(weather_in, dim=-1)))  # ty: ignore[call-non-callable]
    hd = nn.functional.gelu(model.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))  # ty: ignore[call-non-callable]
    return torch.cat([hw, hd, emb], dim=-1)


def probe_input_width() -> int:
    """Return the width of `encoder_features`: two encoder outputs and the Primary embedding."""
    return 2 * m545.ENCODER_OUT + m545.EMBEDDING


def install() -> None:
    """Replace the network class on experiment 591's module, which `e00591_arm.new_model` reads."""
    base_module.AuxLagMdn = StateAuxLagMdn

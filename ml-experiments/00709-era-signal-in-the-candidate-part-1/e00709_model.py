# ruff: noqa: E501
"""Experiment 709: the candidate network with an era encoder, an anchor on the output, and O-enc.

`EraPermLagMdn` is `e00613_nbmodel.PermLagMdn` (the candidate's stage 3 network) plus four parts.
Each part leaves the output unchanged at the start, so a zero era block gives exactly `PermLagMdn`'s output.

- **Era encoder.** Nine inputs (unknown flag, class weight, age, six changes) pass through
  `Linear(9, 16)`, GELU, `Linear(16, 8)`. The 8 outputs are multiplied by the `active` flag (so a
  `none` row gives a zero vector) and appended to the input of the first layer of `mixer_ft`, which
  `widen_first_layer` widens by 8 zero columns. The encoder has an ordinary random initialisation,
  so the widened columns receive a gradient from the first step while the network starts as
  `PermLagMdn` (the widened columns are zero). A `none` row has a zero vector, so no era weight
  receives a gradient from it.
- **Anchor.** The anchor column (standardised, in the units of the mixture means) is added to the
  three component means, as `OutputOffsetDrift` adds its offset. Mode 1 adds the anchor in full.
  Mode 2 (a learned arm's late-confirmed era) adds the anchor minus the change the offset already
  made from the knot before the onset to the newest knot, clipped between zero and the anchor.
- **Era-id table (arm stat only).** `Embedding(130, 8)`, zero-initialised, added to the encoder's
  output before the `active` flag multiplies it. Slot 0 is shared by every unseen id and by the
  curriculum's injected eras.
- **O-enc (arm oracle-enc only).** Two `Linear(8, hidden, bias=False)` layers, zero-initialised,
  add a function of the encoder's vector to the first-layer pre-activation of the weather and
  date-time encoders. They are registered on the model, outside `model.weather` and `model.datetime`,
  so the `frozen` filter in `e00574_fit.train` (which freezes those two modules) leaves them trainable.
  Their hooks are off while the lag side's `_expected` calls run, so a lag's expected weather reads
  a zero era vector.

The penalty keeps the candidate's first-difference penalty on the offset knots unchanged and adds an
L2 penalty on the era-id table (weight decay for the table only).

The era block follows the signal block in the row: the row is `label_col + n_signal` wide for
`PermLagMdn`, then `ERA_WIDTH` era columns. A row without the era columns reads as `none`.
"""

from typing import Final

import e00559_model as m  # ty: ignore[unresolved-import]
import e00613_nbmodel as nb  # ty: ignore[unresolved-import]
import e00696_model as m696  # ty: ignore[unresolved-import]
import e00709_signal as sig
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

ENCODER_HIDDEN: Final[int] = 16
ERA_DIM: Final[int] = 8
ID_L2: Final[float] = 1e-2
"""Weight of the L2 penalty on the era-id table."""
OWN_PREFIXES: Final[tuple[str, ...]] = ("era_enc.", "era_ids.", "oenc_weather.", "oenc_datetime.")
"""Parameters that a stage 2 state dict does not hold."""
MIXER_KEY: Final[str] = "mixer_ft.0.weight"


class EraPermLagMdn(nb.PermLagMdn):
    """`PermLagMdn` plus the era encoder, the anchor, the era-id table, and optionally O-enc."""

    def __init__(
        self,
        *,
        n_primaries: int,
        stage: str,
        n_knots: int,
        n_b1: int,
        inputs: nb.InputSet,
        id_table: bool = False,
        oenc: bool = False,
    ) -> None:
        """Build the candidate's network, then add the era parts (stage `ft` only).

        Args:
            n_primaries: Number of Primaries.
            stage: Must be `ft`.
            n_knots: Number of monthly offset knots.
            n_b1: Number of B1 columns.
            inputs: The signal columns that the lag encoder reads.
            id_table: Add the era-id table (arm stat).
            oenc: Add the O-enc layers (arm oracle-enc).
        """
        assert stage == "ft", "the era parts belong to stage 3 only"
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, inputs=inputs
        )
        self.n_signal = inputs.n_extra
        self.era_enc = nn.Sequential(
            nn.Linear(sig.ENCODER_INPUTS, ENCODER_HIDDEN),
            nn.GELU(),
            nn.Linear(ENCODER_HIDDEN, ERA_DIM),
        )
        self.era_ids: nn.Embedding | None = None
        if id_table:
            self.era_ids = nn.Embedding(sig.ID_TABLE_SIZE, ERA_DIM)
            nn.init.zeros_(self.era_ids.weight)
        m696.widen_first_layer(layer=self.mixer_ft[0], extra=ERA_DIM)  # ty: ignore[invalid-argument-type, not-subscriptable]
        self.oenc_weather: nn.Linear | None = None
        self.oenc_datetime: nn.Linear | None = None
        if oenc:
            self.oenc_weather = nn.Linear(ERA_DIM, self.weather[0].out_features, bias=False)  # ty: ignore[not-subscriptable, unresolved-attribute]
            self.oenc_datetime = nn.Linear(ERA_DIM, self.datetime[0].out_features, bias=False)  # ty: ignore[not-subscriptable, unresolved-attribute]
            nn.init.zeros_(self.oenc_weather.weight)
            nn.init.zeros_(self.oenc_datetime.weight)
            self.weather[0].register_forward_hook(self._add_weather)  # ty: ignore[not-subscriptable, unresolved-attribute]
            self.datetime[0].register_forward_hook(self._add_datetime)  # ty: ignore[not-subscriptable, unresolved-attribute]
        self._vec: torch.Tensor | None = None
        self._oenc_on = False
        self.mixer_ft.register_forward_pre_hook(self._append_vec)  # ty: ignore[unresolved-attribute]

    @property
    def base_width(self) -> int:
        """Return the width of a `PermLagMdn` row: the lag-free, lag, tau, B1 and signal columns."""
        return self.label_col + self.n_signal

    # ------------------------------------------------------------------ hooks

    def _append_vec(self, module: nn.Module, args: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """Append the era vector (zeros when the pass has none) to the first mixer layer's input."""
        _ = module
        (core,) = args
        vec = self._vec if self._vec is not None else core.new_zeros(len(core), ERA_DIM)
        return (torch.cat([core, vec], dim=-1),)

    def _add_weather(self, module: nn.Module, args: tuple, out: torch.Tensor) -> torch.Tensor:
        """Add the O-enc weather term to the weather encoder's first-layer pre-activation."""
        _ = (module, args)
        if not self._oenc_on or self._vec is None or self.oenc_weather is None:
            return out
        return out + self.oenc_weather(self._vec)

    def _add_datetime(self, module: nn.Module, args: tuple, out: torch.Tensor) -> torch.Tensor:
        """Add the O-enc date-time term to the date-time encoder's first-layer pre-activation."""
        _ = (module, args)
        if not self._oenc_on or self._vec is None or self.oenc_datetime is None:
            return out
        return out + self.oenc_datetime(self._vec)

    def _expected(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return the lag side's expected-input embeddings with O-enc switched off (a zero era vector)."""
        was = self._oenc_on
        self._oenc_on = False
        try:
            return super()._expected(x=x, primary=primary, emb=emb)
        finally:
            self._oenc_on = was

    # ------------------------------------------------------------------ era vector and anchor

    def era_vector(self, *, era: torch.Tensor) -> torch.Tensor:
        """Return the (rows, 8) era vector: encoder output plus table row, zero where not active.

        A row without an era id reads the forced-zero slot (`sig.SLOT_NONE`), whose table row is
        never added, so the shared new-era slot (`sig.SLOT_SHARED`) is the only slot an unseen id trains.
        """
        vec = self.era_enc(era[:, : sig.ENCODER_INPUTS])
        if self.era_ids is not None:
            slot = era[:, sig.COL_SLOT].long()
            looked = self.era_ids(slot)
            vec = vec + torch.where(
                (slot == sig.SLOT_NONE)[:, None], torch.zeros_like(looked), looked
            )
        return torch.where(
            era[:, sig.COL_ACTIVE : sig.COL_ACTIVE + 1] > 0.5, vec, torch.zeros_like(vec)
        )

    def anchor(self, *, era: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the (rows, 1) anchor in standardised units, with the mode rule applied.

        Args:
            era: The era columns.
            primary: Primary index of each row.

        Returns:
            Mode 0 gives 0, mode 1 the full anchor, and mode 2 the full anchor minus the offset's own
            change from the onset knot position to the newest knot, clipped between 0 and the anchor.
        """
        full = era[:, sig.COL_ANCHOR : sig.COL_ANCHOR + 1]
        mode = era[:, sig.COL_MODE : sig.COL_MODE + 1]
        with torch.no_grad():
            newest = torch.full_like(era[:, 0], float(self.n_knots - 1))
            moved = self.offset(newest, primary) - self.offset(era[:, sig.COL_ONSET_TAU], primary)
        clipped = torch.minimum(
            torch.maximum(full - moved, torch.minimum(full, torch.zeros_like(full))),
            torch.maximum(full, torch.zeros_like(full)),
        )
        return torch.where(
            mode == sig.MODE_FULL,
            full,
            torch.where(mode == sig.MODE_CLIPPED, clipped, torch.zeros_like(full)),
        )

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs, with the anchor added to the three component means."""
        assert primary is not None
        era = None
        if x.shape[1] == self.base_width + sig.ERA_WIDTH:
            era, x = x[:, self.base_width :], x[:, : self.base_width]
        elif x.shape[1] != self.base_width:
            msg = f"row width {x.shape[1]} is neither {self.base_width} nor {self.base_width + sig.ERA_WIDTH}"
            raise ValueError(msg)
        self._vec = self.era_vector(era=era) if era is not None else None
        self._oenc_on = self._vec is not None
        try:
            out = super().forward(x, primary)
        finally:
            self._vec = None
            self._oenc_on = False
        if era is None:
            return out
        k = m.COMPONENTS
        shift = self.anchor(era=era, primary=primary)
        return torch.cat([out[:, :k], out[:, k : 2 * k] + shift, out[:, 2 * k :]], dim=-1)

    # ------------------------------------------------------------------ penalty and loading

    def penalty(self) -> torch.Tensor:
        """Return the offset knots' first-difference penalty (unchanged) plus the era-id table's L2."""
        out = super().penalty()
        if self.era_ids is not None:
            out = out + ID_L2 * (self.era_ids.weight**2).mean()
        return out

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: ANN001, ANN201
        """Load a stage 2 state dict: pad the lag layer and `mixer_ft.0.weight`, allow the era parts missing.

        Args:
            state_dict: A `PermLagMdn` state dict (or one of this class).
            strict: Whether other keys must match; the era parts' keys are always allowed to be missing.
            assign: Passed to `torch.nn.Module.load_state_dict`.

        Returns:
            The load result.
        """
        w = state_dict.get(nb.LAG_KEY)
        n = self.inputs.n_encoder_extra if self.inputs.n_extra else 0
        if n and w is not None and w.shape[1] == self.lag[0][0].in_features - n:
            state_dict = {
                **state_dict,
                nb.LAG_KEY: torch.cat([w, w.new_zeros(w.shape[0], n)], dim=1),
            }
        v = state_dict.get(MIXER_KEY)
        if v is not None and v.shape[1] == self.mixer_ft[0].in_features - ERA_DIM:
            state_dict = {
                **state_dict,
                MIXER_KEY: torch.cat([v, v.new_zeros(v.shape[0], ERA_DIM)], dim=1),
            }
        result = nn.Module.load_state_dict(self, state_dict, strict=False, assign=assign)
        allowed = (*OWN_PREFIXES, "bottle.", "shift_head.")
        bad = [k for k in result.missing_keys if not k.startswith(allowed)]
        assert not bad, bad
        assert not result.unexpected_keys, result.unexpected_keys
        return result


def new_era_model(
    *, n_primaries: int, n_knots: int, n_b1: int, n_signal: int, arm: str
) -> EraPermLagMdn:
    """Build the network of an era arm.

    Args:
        n_primaries: Number of Primaries.
        n_knots: Number of monthly offset knots.
        n_b1: Number of B1 columns.
        n_signal: Number of signal columns that the lag encoder reads (never 2).
        arm: The arm; `stat` adds the era-id table and `oracle-enc` adds O-enc.

    Returns:
        The network.
    """
    return EraPermLagMdn(
        n_primaries=n_primaries, stage="ft", n_knots=n_knots, n_b1=n_b1,
        inputs=nb.InputSet(n_extra=n_signal), id_table=arm == "stat", oenc=arm == "oracle-enc",
    )  # fmt: skip

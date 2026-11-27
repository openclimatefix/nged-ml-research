# ruff: noqa: E501
"""Experiment 713: the two report-only bank read-outs that change the history rows at prediction.

`shuffle` permutes the 14 history rows of each sample with a fixed seed while every row keeps its
own content (power, visibility, weather, date-time, day type, and the row-exists flag). The age
embedding belongs to the slot, so after the shuffle a row sits in a slot of another age. A forecast
that does not move under the shuffle shows that the network ignores elapsed time.

`mask9` makes the rows of ages 9 to 15 (slots 7 to 13) absent: no power is visible and the row-exists
flag is false, so the older half of the window carries no information.

Both act on an assembled batch and touch no fitted weight. They are diagnostics: no verdict reads them.
"""

from typing import Final

import e00713_model as mdl
import numpy as np
import torch  # ty: ignore[unresolved-import]

VARIANTS: Final = ("plain", "shuffle", "mask9")
FIRST_MASKED_SLOT: Final = 7
"""Slot 0 is age 2, so the age 9 row is slot 7."""
SHUFFLE_SEED: Final = 713

_ROW_FIELDS: Final = (
    "hist_power", "hist_visible", "hist_weather", "hist_datetime", "day_type_hist", "row_exists",
)  # fmt: skip


def apply_variant(*, inputs: mdl.MsaInputs, variant: str) -> mdl.MsaInputs:
    """Return the batch with the history rows shuffled, or the older rows masked, or unchanged."""
    assert variant in VARIANTS, variant
    if variant == "plain":
        return inputs
    fields = dict(inputs.__dict__)
    if variant == "mask9":
        fields["hist_visible"] = inputs.hist_visible.clone()
        fields["row_exists"] = inputs.row_exists.clone()
        fields["hist_visible"][:, FIRST_MASKED_SLOT:] = False
        fields["row_exists"][:, FIRST_MASKED_SLOT:] = False
        return mdl.MsaInputs(**fields)
    b = len(inputs.primary)
    order = np.stack(
        [np.random.default_rng([SHUFFLE_SEED, i]).permutation(mdl.N_HISTORY) for i in range(b)]
    )
    index = torch.from_numpy(order).to(inputs.hist_power.device)
    for name in _ROW_FIELDS:
        value = getattr(inputs, name)
        shape = [b, mdl.N_HISTORY] + [1] * (value.dim() - 2)
        fields[name] = torch.gather(value, 1, index.view(*shape).expand_as(value))
    return mdl.MsaInputs(**fields)

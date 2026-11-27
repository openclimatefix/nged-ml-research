# ruff: noqa: E501
"""A small random `UntInputs` batch for the tests of the model, the training loops, and the predictor."""

from types import SimpleNamespace

import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00733_inputs as inp
import e00733_model as model
import numpy as np
import torch


def make_inputs(*, b: int = 3, seed: int = 0, channels: bool = True) -> model.UntInputs:
    """Return a random batch with the shapes of experiment 713's `MsaInputs` and three untangled channels."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    hist_ok = torch.rand(b, 14, 48, generator=g) > 0.2
    u = torch.rand(b, 14, 48, 3, generator=g)
    u[..., 1] *= 4.0
    ok = (torch.rand(b, 14, 48, generator=g) > 0.3) & hist_ok & channels
    return model.UntInputs(
        primary=torch.randint(0, 16, (b,), generator=g),  # ty: ignore[unknown-argument]
        lead_day=torch.randint(0, 14, (b,), generator=g),  # ty: ignore[unknown-argument]
        top_weather=rnd(b, 48, 21),  # ty: ignore[unknown-argument]
        top_spread=rnd(b, 48, 7).abs(),  # ty: ignore[unknown-argument]
        top_datetime=rnd(b, 48, 21),  # ty: ignore[unknown-argument]
        hist_weather=rnd(b, 14, 48, 21),  # ty: ignore[unknown-argument]
        hist_datetime=rnd(b, 14, 48, 21),  # ty: ignore[unknown-argument]
        hist_power=rnd(b, 14, 48) * hist_ok,  # ty: ignore[unknown-argument]
        hist_visible=hist_ok,  # ty: ignore[unknown-argument]
        row_exists=torch.ones(b, 14, dtype=torch.bool),  # ty: ignore[unknown-argument]
        day_type_top=torch.randint(0, 8, (b, 48), generator=g),  # ty: ignore[unknown-argument]
        day_type_hist=torch.randint(0, 8, (b, 14, 48), generator=g),  # ty: ignore[unknown-argument]
        hist_u=u * ok[..., None],
        hist_u_ok=ok,
    )


class FakeSource:
    """A 713-style source: random `MsaInputs` fields, a target with NaN past the cut, and run times."""

    def __init__(self, *, run_offset_us: int = 0) -> None:
        self.device = "cpu"
        self.ens = SimpleNamespace(
            first_day=0, run_us=None, filled=torch.ones(16, 4000, 672, dtype=torch.bool)
        )
        self.steps = torch.arange(48)
        self.power_z = torch.randn(16, 48 * 4100)
        self.power_in = self.power_z
        self.run_offset_us = run_offset_us
        self.avail_extra_us = 0
        self.other_extra_us = 0

    def assemble(
        self,
        *,
        primary: torch.Tensor,
        c_idx: torch.Tensor,
        lead_day: torch.Tensor,
        ensemble: bool,
        mask_all_power: object,
        mask_newest_row: object,
        target_end: object,
    ) -> SimpleNamespace:
        b = len(primary)
        mi = make_inputs(b=b, seed=int(c_idx[0]), channels=False)
        base = {k: v for k, v in mi.__dict__.items() if k not in {"hist_u", "hist_u_ok"}}
        base["primary"], base["lead_day"] = primary, lead_day
        from e00713_model import MsaInputs  # ty: ignore[unresolved-import]

        hist_idx = (
            c_idx[:, None, None]
            - 48 * torch.arange(2, 16)[None, :, None]
            + torch.arange(48)[None, None, :]
        )
        target_idx = c_idx[:, None] + 48 * lead_day[:, None] + torch.arange(48)[None, :]
        target = torch.randn(b, 48)
        self.ens.run_us = torch.zeros(16, 4000, dtype=torch.int64)
        init = np.zeros((16, 4000), dtype=np.int64)
        avail, other = init.copy(), init.copy()
        day = c_idx // 48
        for i in range(b):
            origin_day = int(day[i])
            run = lay.GRID_START_US + origin_day * lay.DAY_US + self.run_offset_us
            self.ens.run_us[int(primary[i]), origin_day] = run
            init[int(primary[i]), origin_day] = run
            avail[int(primary[i]), origin_day] = run + 9 * lay.HOUR_US + self.avail_extra_us
            other[int(primary[i]), origin_day] = run + self.other_extra_us
        self.ens.file_run_init_us, self.ens.file_run_avail_us = init, avail
        self.ens.file_other_valid_us = other
        return SimpleNamespace(
            inputs=MsaInputs(**base), target=target, target_idx=target_idx, hist_idx=hist_idx
        )


def make_channel_tensors(*, n_grid: int, d8: int = 3) -> inp.ChannelTensors:
    rng = np.random.default_rng(0)
    values = np.stack(
        [rng.random(n_grid), rng.random(n_grid) * 20, rng.random(n_grid)], axis=1
    ).astype(np.float32)
    return inp.build_tensors(
        dense_values=values,
        dense_ok=np.ones(n_grid, bool),
        n_primaries=16,
        d8_index=d8,
        target_std=2.0,
    )

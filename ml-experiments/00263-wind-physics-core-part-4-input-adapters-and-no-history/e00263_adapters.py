"""The input adapters, the gated hybrid, and the pooled and no-history fits of experiment 00263.

One module holds every arm that trains a small neural network jointly with the physics core:

- B10 adds an input adapter, a network that rescales and shifts the hub-height wind speed before
  the power curve, each within a stated bound.
- B11 adds the input adapter and the output calibrator of experiment 230's B9, trained together.
- B13 blends the core's expected power with a free network's, through a gate on the wind speed.
- B12 is B10 with one shared core and one shared adapter pooled over the farms, with a one-hot
  farm label as an extra adapter input.

All arms read the ERA5 weather in training, drop the idle layer (experiment 230 found it changes
the score by under 0.01 points), and take their spread from `ResidualQuantiles`.
"""

from dataclasses import dataclass
from typing import Final

import e00263_core as core_module
import e00263_neural as neural
import e00263_step0 as step0
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

SCALE_RANGE: Final[tuple[float, float]] = (0.8, 1.25)
"""The input adapter's speed factor lies in this interval."""
OFFSET_LIMIT_M_S: Final[float] = 1.5
"""The input adapter's speed offset lies within plus or minus this many m/s."""
JOINT_STEPS: Final[int] = 500
JOINT_LEARNING_RATE: Final[float] = 0.01
JOINT_WEIGHT_DECAY: Final[float] = 1e-4
ADAPTER_HIDDEN: Final[int] = 16
GATE_START_BIAS: Final[float] = -3.0
"""The gate starts nearly closed, so the blend starts as the physics core."""
N_FARMS: Final[int] = 3
CHUNK_ROWS: Final[int] = 500_000


@dataclass(frozen=True)
class Parts:
    """Which optional networks a `HybridModel` carries.

    Attributes:
        input_adapter: Whether an adapter rescales and shifts the hub-height wind speed.
        calibrator: Whether a bounded correction is added to the expected power.
        gate: Whether a free network is blended in through a gate on the wind speed.
        farm_label: Whether the adapter also reads a one-hot farm label.
    """

    input_adapter: bool = False
    calibrator: bool = False
    gate: bool = False
    farm_label: bool = False


def _two_layer(*, n_inputs: int, n_outputs: int, first_bias: float = 0.0) -> nn.Sequential:
    """Return a two-hidden-layer tanh network whose last layer starts at zero weights.

    `first_bias` is the starting bias of the first output; the other outputs start at zero.
    """
    final = nn.Linear(ADAPTER_HIDDEN, n_outputs)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    with torch.no_grad():
        final.bias[0] = first_bias
    return nn.Sequential(
        nn.Linear(n_inputs, ADAPTER_HIDDEN),
        nn.Tanh(),
        nn.Linear(ADAPTER_HIDDEN, ADAPTER_HIDDEN),
        nn.Tanh(),
        final,
    )


class HybridModel(nn.Module):
    """A physics core with optional input adapter, output calibrator, and gated free network."""

    def __init__(self, *, core: core_module.WindCore, parts: Parts) -> None:
        """Wrap `core`, and start every added network at the identity, so training starts at B8."""
        super().__init__()
        self.core = core
        self.parts = parts
        adapter_inputs = neural.INPUT_COLUMNS + (N_FARMS if parts.farm_label else 0)
        if parts.input_adapter:
            unit = (1.0 - SCALE_RANGE[0]) / (SCALE_RANGE[1] - SCALE_RANGE[0])
            self.adapter = _two_layer(
                n_inputs=adapter_inputs, n_outputs=2, first_bias=float(np.log(unit / (1 - unit)))
            )
        if parts.calibrator:
            self.calibrator = _two_layer(n_inputs=neural.INPUT_COLUMNS, n_outputs=1)
        if parts.gate:
            self.gate = _two_layer(n_inputs=1, n_outputs=1, first_bias=GATE_START_BIAS)
            self.free = nn.Sequential(
                nn.Linear(neural.RAW_INPUT_COLUMNS, ADAPTER_HIDDEN),
                nn.Tanh(),
                nn.Linear(ADAPTER_HIDDEN, 1),
            )

    def adapted_speed(
        self, *, speed: torch.Tensor, inputs: torch.Tensor, farm: torch.Tensor
    ) -> torch.Tensor:
        """Return the hub-height speed after the input adapter, or `speed` if there is none."""
        if not self.parts.input_adapter:
            return speed
        features = torch.cat([inputs, farm], dim=1) if self.parts.farm_label else inputs
        raw = self.adapter(features)
        low, high = SCALE_RANGE
        scale = low + (high - low) * torch.sigmoid(raw[:, 0])
        offset = OFFSET_LIMIT_M_S * torch.tanh(raw[:, 1])
        return (scale * speed + offset).clamp(min=0.0)

    def forward(
        self,
        *,
        weather: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        inputs: torch.Tensor,
        raw: torch.Tensor,
        farm: torch.Tensor,
    ) -> torch.Tensor:
        """Return the expected power fraction for each row."""
        speed = self.core.hub_height_speed(
            speed_10m=weather[0], speed_100m=weather[1], temperature_c=weather[2]
        )
        expected = self.core.active_power(
            speed=self.adapted_speed(speed=speed, inputs=inputs, farm=farm)
        )
        if self.parts.calibrator:
            expected = expected + neural.CALIBRATOR_CAP * torch.tanh(
                self.calibrator(inputs).squeeze(-1)
            )
        if self.parts.gate:
            weight = torch.sigmoid(self.gate((speed / 10.0)[:, None]).squeeze(-1))
            expected = (1 - weight) * expected + weight * torch.sigmoid(self.free(raw).squeeze(-1))
        return expected


@dataclass(frozen=True)
class Design:
    """The tensors of one frame that `HybridModel.forward` reads."""

    weather: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    inputs: torch.Tensor
    raw: torch.Tensor
    farm: torch.Tensor


def make_design(
    *,
    frame: pl.DataFrame,
    core: core_module.WindCore,
    farm_index: int | None,
    raw_centre: np.ndarray,
    raw_scale: np.ndarray,
) -> Design:
    """Return the design of `frame`; `farm_index` None gives an all-zero farm label."""
    base_speed = neural.hub_speed(core=core, frame=frame)
    label = np.zeros((frame.height, N_FARMS), dtype=np.float32)
    if farm_index is not None:
        label[:, farm_index] = 1.0
    return Design(
        weather=step0.weather_arrays(frame=frame).as_tensors(),
        inputs=torch.as_tensor(neural.network_inputs(frame=frame, hub_speed=base_speed)),
        raw=torch.as_tensor(neural.raw_inputs(frame=frame, centre=raw_centre, scale=raw_scale)),
        farm=torch.as_tensor(label),
    )


def concat_designs(*, designs: list[Design]) -> Design:
    """Return the designs stacked row-wise."""
    return Design(
        weather=tuple(torch.cat([d.weather[i] for d in designs]) for i in range(3)),
        inputs=torch.cat([d.inputs for d in designs]),
        raw=torch.cat([d.raw for d in designs]),
        farm=torch.cat([d.farm for d in designs]),
    )


def fit_hybrid(
    *, core: core_module.WindCore, parts: Parts, design: Design, target: np.ndarray, seed: int
) -> HybridModel:
    """Fit the core and the added networks together by full-batch Adam on the L1 loss."""
    torch.manual_seed(seed)
    torch.set_num_threads(core_module.THREADS)
    model = HybridModel(core=core, parts=parts)
    goal = torch.as_tensor(target, dtype=torch.float32)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=JOINT_LEARNING_RATE, weight_decay=JOINT_WEIGHT_DECAY
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=JOINT_STEPS)
    for _ in range(JOINT_STEPS):
        optimiser.zero_grad()
        loss = (
            (
                model(
                    weather=design.weather, inputs=design.inputs, raw=design.raw, farm=design.farm
                )
                - goal
            )
            .abs()
            .mean()
        )
        loss.backward()
        optimiser.step()
        schedule.step()
    return model.eval()


def predict_hybrid(*, model: HybridModel, design: Design) -> np.ndarray:
    """Return the model's expected power fraction for every row, in float32."""
    rows = design.farm.shape[0]
    pieces = []
    with torch.no_grad():
        for start in range(0, rows, CHUNK_ROWS):
            part = slice(start, start + CHUNK_ROWS)
            pieces.append(
                model(
                    weather=tuple(w[part] for w in design.weather),
                    inputs=design.inputs[part],
                    raw=design.raw[part],
                    farm=design.farm[part],
                ).numpy()
            )
    return np.concatenate(pieces)


def fit_core_without_idle(*, frame: pl.DataFrame, seed: int = 0) -> core_module.WindCore:
    """Return the physics core fitted on `frame`, with no idle layer."""
    return core_module.fit_core(
        weather=step0.weather_arrays(frame=frame),
        target=frame["target"].to_numpy(),
        with_idle_layer=False,
        seed=seed,
    )


def default_core() -> core_module.WindCore:
    """Return the physics core at its starting (prior-only) parameters, with no fitting."""
    return core_module.WindCore(with_idle_layer=False, idle_floor=0.0).eval()


def hybrid_member_quantiles(
    *,
    parts: Parts,
    train: pl.DataFrame,
    members: pl.DataFrame,
    seed: int,
    farm_index: int | None,
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of one farm's hybrid, fitted on `train`."""
    core = fit_core_without_idle(frame=train)
    centre, scale = neural.raw_input_scaling(frame=train)
    train_design = make_design(
        frame=train, core=core, farm_index=farm_index, raw_centre=centre, raw_scale=scale
    )
    member_design = make_design(
        frame=members, core=core, farm_index=farm_index, raw_centre=centre, raw_scale=scale
    )
    target = train["target"].to_numpy()
    model = fit_hybrid(core=core, parts=parts, design=train_design, target=target, seed=seed)
    residuals = core_module.ResidualQuantiles.fit(
        expected=predict_hybrid(model=model, design=train_design), observed=target
    )
    return residuals.quantiles(expected=predict_hybrid(model=model, design=member_design))


def pooled_member_quantiles(
    *,
    parts: Parts,
    train_by_farm: dict[int, pl.DataFrame],
    members: pl.DataFrame,
    seed: int,
    target_index: int | None,
) -> np.ndarray:
    """Return quantiles of one pooled core and adapter, fitted on `train_by_farm`'s farms.

    Args:
        parts: The networks the pooled model carries.
        train_by_farm: Training rows keyed by farm index; only these farms are fitted on.
        members: The ensemble members of the farm to forecast.
        seed: The seed of the added networks.
        target_index: The farm's index if it is among `train_by_farm` (its own residual quantiles
            and farm label are used), else None (all-zero label, residual quantiles pooled over
            the fitted farms).

    Returns:
        Shape (rows, 11) quantiles of the observed fraction for `members`.
    """
    pooled = pl.concat(list(train_by_farm.values()))
    core = fit_core_without_idle(frame=pooled)
    centre, scale = neural.raw_input_scaling(frame=pooled)
    designs = {
        index: make_design(
            frame=frame, core=core, farm_index=index, raw_centre=centre, raw_scale=scale
        )
        for index, frame in train_by_farm.items()
    }
    model = fit_hybrid(
        core=core,
        parts=parts,
        design=concat_designs(designs=list(designs.values())),
        target=pooled["target"].to_numpy(),
        seed=seed,
    )
    expected = {
        index: predict_hybrid(model=model, design=design) for index, design in designs.items()
    }
    own = [target_index] if target_index is not None else list(train_by_farm)
    residuals = core_module.ResidualQuantiles.fit(
        expected=np.concatenate([expected[i] for i in own]),
        observed=np.concatenate([train_by_farm[i]["target"].to_numpy() for i in own]),
    )
    member_design = make_design(
        frame=members, core=core, farm_index=target_index, raw_centre=centre, raw_scale=scale
    )
    return residuals.quantiles(expected=predict_hybrid(model=model, design=member_design))


def prior_core_member_quantiles(
    *, train_by_farm: dict[int, pl.DataFrame], members: pl.DataFrame
) -> np.ndarray:
    """Return quantiles of the unfitted core, with residual quantiles pooled over the farms."""
    core = default_core()
    pooled = pl.concat(list(train_by_farm.values()))
    residuals = core_module.ResidualQuantiles.fit(
        expected=core_module.predict_expected_power(
            core=core, weather=step0.weather_arrays(frame=pooled)
        ),
        observed=pooled["target"].to_numpy(),
    )
    return residuals.quantiles(
        expected=core_module.predict_expected_power(
            core=core, weather=step0.weather_arrays(frame=members)
        )
    )


def pooled_network_member_quantiles(
    *, train_by_farm: dict[int, pl.DataFrame], members: pl.DataFrame, seed: int
) -> np.ndarray:
    """Return quantiles of the free network (B6) fitted on the pooled `train_by_farm` rows."""
    pooled = pl.concat(list(train_by_farm.values()))
    centre, scale = neural.raw_input_scaling(frame=pooled)
    network = neural.fit_free_network(
        inputs=neural.raw_inputs(frame=pooled, centre=centre, scale=scale),
        target=pooled["target"].to_numpy(),
        seed=seed,
    )
    return neural.free_quantiles(
        network=network, inputs=neural.raw_inputs(frame=members, centre=centre, scale=scale)
    )

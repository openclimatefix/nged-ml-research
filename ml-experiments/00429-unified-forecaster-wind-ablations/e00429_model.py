"""The shared model of experiment 00429: one pooled mixture-density network, switches for arms.

Every arm is the same network with three switches (the design note's D1 to D3):

- `route` (member handling): `summary` feeds a fixed six-number summary of the stage-3 plant's 51
  member powers (experiment 00347's y2, the reproduction gate G0); `one_pass_ms` and `one_pass_m`
  feed the mean and standard deviation (or the mean alone) of each weather column over the 51
  members; `member` runs a small encoder on every member and pools the 51 outputs by their mean,
  standard deviation, 10th and 90th percentile.
- `plant` (physics layer, `member` route only): `none`; `frozen` adds the stage-3 plant's power for
  each member to the encoder input and centres the mixture on the plant's median power; `e2e` does
  the same with the plant held as a trainable `nn.Module` that starts from the stage-3 parameters.
- `route` `memberwise` runs the trunk on each member alone (its weather, and with `plant="frozen"`
  its own plant power, which is also the centre of its mixture), and the pooled forecast is the
  equal-weight mixture of the 51 member mixtures, trained on the pooled likelihood.
- `head`: `clipped` fits the mixture by the plain likelihood and clamps the quantiles to [0, 1]
  (experiment 00347); `censored` fits the likelihood of an observation at or below 0.001 as the
  mixture's mass below zero and one at or above 0.999 as its mass above one.

A mixture without a plant is centred on the mean fraction of the training cases. The farm enters as
a one-hot column, which the first layer turns into a learned per-farm embedding.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import e00347_common as common  # ty: ignore[unresolved-import]
import e00347_plant as plant_module  # ty: ignore[unresolved-import]
import e00347_s4_inputs as inputs  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from scipy.special import ndtr
from torch import nn  # ty: ignore[unresolved-import]

DEVICE: Final = torch.device(
    "cpu" if os.environ.get("E00429_CPU") else ("cuda" if torch.cuda.is_available() else "cpu")
)
EPOCHS: Final[int] = int(os.environ.get("E00429_EPOCHS", "40"))
BATCH: Final[int] = 2048
EVAL_BATCH: Final[int] = 8192
LEARNING_RATE: Final[float] = 2e-3
PLANT_LEARNING_RATE: Final[float] = 2e-4
WEIGHT_DECAY: Final[float] = 1e-4
COMPONENTS: Final[int] = 3
ENCODER_WIDTH: Final[int] = 16
MEMBERS: Final[int] = 51
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
FARMS: Final[tuple[str, ...]] = common.FARMS
ZERO: Final[float] = 0.001
FULL: Final[float] = 0.999
LOG_SQRT_2PI: Final[float] = float(np.log(2 * np.pi)) * 0.5


@dataclass(frozen=True)
class Arm:
    """The three switches of one arm."""

    route: str
    plant: str = "none"
    head: str = "clipped"


class Cases:
    """One case set on the device: member weather, plant power, summary, farm, lead, target."""

    def __init__(
        self,
        *,
        weather: np.ndarray,
        power: np.ndarray,
        summary: np.ndarray,
        frame: pl.DataFrame,
    ) -> None:
        """Move the arrays of one case set to the device."""
        hours = frame["lead_hours"].to_numpy().astype(np.float64)
        angle = 2 * np.pi * hours / 24.0
        farm = np.stack([(frame["site"] == f).to_numpy() for f in FARMS], axis=1).astype(np.float64)
        self.aux_np = np.concatenate(
            [np.stack([np.sin(angle), np.cos(angle), hours / 100.0], axis=1), farm], axis=1
        )
        self.farm_index = torch.tensor(farm.argmax(axis=1), device=DEVICE)
        self.weather = torch.tensor(weather, device=DEVICE)
        self.power = torch.tensor(power, device=DEVICE)
        self.summary_np = summary.astype(np.float64)
        self.summary_centre = torch.tensor(summary[:, 0], dtype=torch.float32, device=DEVICE)
        self.y = torch.tensor(frame["actual_norm"].to_numpy().astype(np.float32), device=DEVICE)
        self.size = len(frame)


class Net(nn.Module):
    """The pooled mixture-density network of one arm."""

    w_mean: torch.Tensor
    w_std: torch.Tensor
    y_mean: torch.Tensor

    def __init__(
        self,
        *,
        arm: Arm,
        stats: dict[str, torch.Tensor],
        plants: dict[str, plant_module.Plant] | None,
    ) -> None:
        """Build the encoder (member route) and the trunk."""
        super().__init__()
        self.arm = arm
        if arm.route == "memberwise" and (arm.head != "clipped" or arm.plant == "e2e"):
            msg = "member-wise arms support the clipped head and no trainable plant"
            raise ValueError(msg)
        for key, value in stats.items():
            self.register_buffer(key, value)
        self.plants = nn.ModuleDict(plants or {})
        width = ENCODER_WIDTH
        n_weather = len(stats["w_mean"])
        if arm.route == "member":
            extra = 0 if arm.plant == "none" else 1
            self.encoder = nn.Sequential(
                nn.Linear(n_weather + extra, width), nn.GELU(), nn.Linear(width, width)
            )
            n_feature = 4 * width
        elif arm.route == "memberwise":
            n_feature = n_weather + (0 if arm.plant == "none" else 1)
        else:
            n_feature = {"summary": 6, "one_pass_ms": 2 * n_weather, "one_pass_m": n_weather}[
                arm.route
            ]
        self.trunk = nn.Sequential(
            nn.Linear(n_feature + 6, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 3 * COMPONENTS),
        )
        last = self.trunk[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([0.0] * 3 + [-0.1, 0.0, 0.1] + [float(np.log(0.08))] * 3))

    def member_power(self, *, raw: torch.Tensor, farm: torch.Tensor) -> torch.Tensor:
        """Return the trainable plants' power for every member: (batch, 51)."""
        power = torch.empty(raw.shape[:2], device=raw.device)
        for k, name in enumerate(FARMS):
            rows = farm == k
            if not bool(rows.any()):
                continue
            flat = raw[rows].reshape(-1, raw.shape[2])
            weather = (
                flat[:, 1],
                flat[:, 0],
                flat[:, 6],
                torch.atan2(flat[:, 4], flat[:, 5]),
            )
            power[rows] = self.plants[name](weather=weather).reshape(-1, raw.shape[1])
        return power

    def forward(
        self, *, cases: Cases, index: torch.Tensor, aux: torch.Tensor, summary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log weights, means, standard deviations, centre) for the cases `index`."""
        route, plant = self.arm.route, self.arm.plant
        raw = cases.weather[index]
        z = (raw - self.w_mean) / self.w_std
        power = None
        if plant == "frozen":
            power = cases.power[index]
        elif plant == "e2e":
            power = self.member_power(raw=raw, farm=cases.farm_index[index])
        if route == "memberwise":
            member_in = z if power is None else torch.cat([z, power[:, :, None]], dim=2)
            tiled = aux[:, None, :].expand(-1, member_in.shape[1], -1)
            out = self.trunk(torch.cat([member_in, tiled], dim=2))
            centre = self.y_mean.expand(member_in.shape[:2]) if power is None else power
            return (
                torch.log_softmax(out[..., :COMPONENTS], dim=-1),
                out[..., COMPONENTS : 2 * COMPONENTS],
                torch.exp(out[..., 2 * COMPONENTS :].clamp(-6.0, 1.0)),
                centre,
            )
        if route == "summary":
            feature = summary
        elif route == "one_pass_ms":
            feature = torch.cat([z.mean(dim=1), z.std(dim=1)], dim=1)
        elif route == "one_pass_m":
            feature = z.mean(dim=1)
        else:
            member_in = z if power is None else torch.cat([z, power[:, :, None]], dim=2)
            encoded = self.encoder(member_in)
            low_high = torch.quantile(encoded, torch.tensor([0.1, 0.9], device=raw.device), dim=1)
            feature = torch.cat(
                [encoded.mean(dim=1), encoded.std(dim=1), low_high[0], low_high[1]], dim=1
            )
        if route == "summary":
            centre = cases.summary_centre[index]
        elif power is not None:
            centre = power.median(dim=1).values
        else:
            centre = self.y_mean.expand(len(index))
        out = self.trunk(torch.cat([feature, aux], dim=1))
        return (
            torch.log_softmax(out[:, :COMPONENTS], dim=1),
            out[:, COMPONENTS : 2 * COMPONENTS],
            torch.exp(out[:, 2 * COMPONENTS :].clamp(-6.0, 1.0)),
            centre,
        )


def nll(
    *,
    parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y: torch.Tensor,
    head: str,
) -> torch.Tensor:
    """Return the mean negative log likelihood of `y` under the mixture."""
    log_w, mean, std, centre = parts
    if log_w.dim() == 3:  # noqa: PLR2004
        if head != "clipped":
            msg = "member-wise arms use the clipped head"
            raise ValueError(msg)
        z = (y[:, None] - centre)[..., None]
        log_pdf = -0.5 * ((z - mean) / std) ** 2 - torch.log(std) - LOG_SQRT_2PI
        pooled = torch.logsumexp((log_w + log_pdf).flatten(start_dim=1), dim=1)
        return -(pooled - float(np.log(centre.shape[1]))).mean()
    z = (y - centre)[:, None]
    log_pdf = -0.5 * ((z - mean) / std) ** 2 - torch.log(std) - LOG_SQRT_2PI
    if head == "clipped":
        return -torch.logsumexp(log_w + log_pdf, dim=1).mean()
    below = torch.logsumexp(log_w + torch.special.log_ndtr((-centre[:, None] - mean) / std), dim=1)
    above = torch.logsumexp(
        log_w + torch.special.log_ndtr(-(1.0 - centre[:, None] - mean) / std), dim=1
    )
    interior = torch.logsumexp(log_w + log_pdf, dim=1)
    ll = torch.where(y <= ZERO, below, torch.where(y >= FULL, above, interior))
    return -ll.mean()


def mixture_quantiles(*, parts: tuple[torch.Tensor, ...]) -> np.ndarray:
    """Return the 11 quantiles by bisection on the mixture's CDF, clamped to [0, 1]."""
    log_w, mean, std, centre = parts
    w, mean, std = (t.detach().cpu().numpy().astype(np.float64) for t in (log_w.exp(), mean, std))
    mu = centre.detach().cpu().numpy().astype(np.float64)
    out = np.empty((len(mu), len(LEVELS)))
    for k, level in enumerate(LEVELS):
        lo, hi = np.full(len(mu), -2.0), np.full(len(mu), 2.0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cdf = (w * ndtr((mid[:, None] - mean) / std)).sum(axis=1)
            below = cdf < level
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        out[:, k] = 0.5 * (lo + hi)
    return np.minimum(np.maximum(mu[:, None] + out, 0.0), 1.0)


def memberwise_quantiles(*, parts: tuple[torch.Tensor, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return (pooled (rows, 11), per-member (rows, 51, 11)) quantiles of member-wise mixtures.

    The pooled forecast is the equal-weight mixture of the 51 member mixtures; each quantile is
    found by bisection on the mixture's cumulative distribution, clamped to [0, 1].
    """
    log_w, mean, std, centre = parts
    w = log_w.exp()
    loc = centre[..., None] + mean

    def cdf(x: torch.Tensor) -> torch.Tensor:
        return (w * 0.5 * torch.erfc(-(x[..., None] - loc) / std / float(np.sqrt(2.0)))).sum(dim=-1)

    pooled, members = [], []
    for level in LEVELS:
        lo = torch.full(centre.shape[:1], -2.0, device=centre.device)
        hi = torch.full(centre.shape[:1], 3.0, device=centre.device)
        mlo = torch.full(centre.shape, -2.0, device=centre.device)
        mhi = torch.full(centre.shape, 3.0, device=centre.device)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            below = cdf(mid[:, None].expand_as(centre)).mean(dim=1) < level
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
            mmid = 0.5 * (mlo + mhi)
            mbelow = cdf(mmid) < level
            mlo, mhi = torch.where(mbelow, mmid, mlo), torch.where(mbelow, mhi, mmid)
        pooled.append(0.5 * (lo + hi))
        members.append(0.5 * (mlo + mhi))
    pooled_np = torch.stack(pooled, dim=1).clamp(0.0, 1.0).cpu().numpy()
    members_np = torch.stack(members, dim=2).clamp(0.0, 1.0).cpu().numpy()
    return pooled_np, members_np


def build_plants() -> dict[str, plant_module.Plant]:
    """Return one trainable plant per farm, started at the stage-3 parameters."""
    priors = common.prior_scales()
    out = {}
    for farm in FARMS:
        start = plant_module.PlantParameters.from_dict(
            common.read_json(name=inputs.PLANT_FILES["y2"])[farm]
        )
        out[farm] = plant_module.Plant(
            start=start,
            prior_scale=priors[farm],
            with_direction=any(d > 0 for d in start.depths),
        )
    return out


def train_and_predict(  # noqa: PLR0915
    *,
    arm: Arm,
    hindcast: pl.DataFrame,
    train_cases: Cases,
    targets: dict[str, Cases],
    cutoff: datetime,
    seed: int,
) -> tuple[dict[str, np.ndarray], nn.Module]:
    """Train one network on the hindcast and return the 11 quantiles of each target set.

    Training rows are the unmasked hindcast cases whose run and valid time precede `cutoff`;
    early stopping reads the unmasked cases of later runs, as experiment 00347's heads do.
    """
    torch.manual_seed(seed)
    usable = (~hindcast["masked"]).to_numpy()
    init, valid = hindcast["power_fcst_init_time"], hindcast["valid_time"]
    train = usable & ((init < cutoff) & (valid < cutoff)).to_numpy()
    stop = usable & (init >= cutoff).to_numpy()
    train_i_np = np.flatnonzero(train)
    flat = train_cases.weather[torch.tensor(train_i_np[::20], device=DEVICE)].reshape(
        -1, train_cases.weather.shape[2]
    )
    aux_mean = train_cases.aux_np[train].mean(axis=0)
    aux_std = np.maximum(train_cases.aux_np[train].std(axis=0), 1e-6)
    summ_mean = train_cases.summary_np[train].mean(axis=0)
    summ_std = np.maximum(train_cases.summary_np[train].std(axis=0), 1e-6)
    stats = {
        "w_mean": flat.mean(dim=0),
        "w_std": flat.std(dim=0).clamp(min=1e-6),
        "y_mean": train_cases.y[torch.tensor(train_i_np, device=DEVICE)].mean().reshape(1),
    }
    model = Net(arm=arm, stats=stats, plants=build_plants() if arm.plant == "e2e" else None).to(
        DEVICE
    )

    def prepared(cases: Cases) -> tuple[torch.Tensor, torch.Tensor]:
        aux = torch.tensor((cases.aux_np - aux_mean) / aux_std, dtype=torch.float32, device=DEVICE)
        summ = torch.tensor(
            (cases.summary_np - summ_mean) / summ_std, dtype=torch.float32, device=DEVICE
        )
        return aux, summ

    aux_train, summ_train = prepared(train_cases)
    plant_params = [p for name, p in model.named_parameters() if name.startswith("plants.")]
    other = [p for name, p in model.named_parameters() if not name.startswith("plants.")]
    groups = [{"params": other, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}]
    if plant_params:
        groups.append({"params": plant_params, "lr": PLANT_LEARNING_RATE, "weight_decay": 0.0})
    optimiser = torch.optim.AdamW(groups)
    train_i = torch.tensor(train_i_np, device=DEVICE)
    stop_i = torch.tensor(np.flatnonzero(stop), device=DEVICE)
    generator = torch.Generator().manual_seed(seed)

    def batch_nll(
        index: torch.Tensor, *, cases: Cases, aux: torch.Tensor, summ: torch.Tensor
    ) -> torch.Tensor:
        parts = model(cases=cases, index=index, aux=aux[index], summary=summ[index])
        return nll(parts=parts, y=cases.y[index], head=arm.head)

    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        order = train_i[torch.randperm(len(train_i), generator=generator).to(DEVICE)]
        for start in range(0, len(order), BATCH):
            loss = batch_nll(
                order[start : start + BATCH], cases=train_cases, aux=aux_train, summ=summ_train
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            total = sum(
                float(
                    batch_nll(
                        stop_i[s : s + EVAL_BATCH],
                        cases=train_cases,
                        aux=aux_train,
                        summ=summ_train,
                    )
                )
                * len(stop_i[s : s + EVAL_BATCH])
                for s in range(0, len(stop_i), EVAL_BATCH)
            ) / len(stop_i)
        if total < best:
            best, best_state = total, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    out = {}
    for name, cases in targets.items():
        aux, summ = prepared(cases)
        result = np.empty((cases.size, len(LEVELS)), dtype=np.float32)
        member_result = (
            np.empty((cases.size, MEMBERS, len(LEVELS)), dtype=np.float32)
            if arm.route == "memberwise"
            else None
        )
        with torch.no_grad():
            for s in range(0, cases.size, EVAL_BATCH):
                index = torch.arange(s, min(s + EVAL_BATCH, cases.size), device=DEVICE)
                parts = model(cases=cases, index=index, aux=aux[index], summary=summ[index])
                if member_result is not None:
                    pooled, members = memberwise_quantiles(parts=parts)
                    result[s : s + len(index)] = pooled
                    member_result[s : s + len(index)] = members
                else:
                    result[s : s + len(index)] = mixture_quantiles(parts=parts)
        out[name] = np.sort(result, axis=1)
        if member_result is not None:
            out[f"{name}_members"] = np.sort(member_result, axis=2)
    return out, model

# ruff: noqa: PLR0915
"""Fit experiment 429's `member_plain` wind network with a recency weight on the training rows.

The fit is experiment 461's (`e00461_wind_fit`: one fit per seed on the 2024-04-01 to 2025-04-15
hindcast, capacity at p99.9 of earlier power), with one change: the training loss of
`train_and_predict` (experiment 429's `e00429_model`, copied here) multiplies each case's
negative log likelihood by `0.5 ** (age / half_life)`, where age runs from the case's forecast
initialisation time to the cutoff 2025-04-16, and the weights are rescaled to a mean of 1 over the
training cases. Early stopping is unweighted, as in experiment 443's solar sweep. The arm `none` has
all weights 1 and is the refitted reference (gated against experiment 461's stored fits).
One file per arm and seed, `wind_parts/member_plain_<arm>_h<seed>.npy` of shape (scored cases, 11)
in units of the new capacity, is written atomically and skipped when it exists.

    source ml-experiments/00429-*/env.sh   # then add the 00461 and 00466 folders to PYTHONPATH
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00466-*/e00466_wind_fit.py [ARM ...] [--seeds 0 1 2 3 4]
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00347_s4_data as data347  # noqa: E402  # ty: ignore[unresolved-import]
import e00429_data as data  # noqa: E402  # ty: ignore[unresolved-import]
import e00429_model as model  # noqa: E402  # ty: ignore[unresolved-import]
import e00429_run as run429  # noqa: E402  # ty: ignore[unresolved-import]
import e00461_common as common461  # noqa: E402  # ty: ignore[unresolved-import]
import e00461_wind_fit as wind461  # noqa: E402  # ty: ignore[unresolved-import]
import e00466_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]
from torch import nn  # noqa: E402  # ty: ignore[unresolved-import]

DEVICE: Final = model.DEVICE
PARTS_DIR: Final = common.DATA_DIR / "wind_parts"
MODELS_DIR: Final = common.DATA_DIR / "wind_models"
ARM: Final[str] = "member_plain"


def weighted_nll_per_row(
    *, parts: tuple[torch.Tensor, ...], y: torch.Tensor, head: str
) -> torch.Tensor:
    """Return each case's negative log likelihood under the clipped single-pass mixture."""
    log_w, mean, std, centre = parts
    assert head == "clipped"
    assert log_w.dim() == 2
    z = (y - centre)[:, None]
    log_pdf = -0.5 * ((z - mean) / std) ** 2 - torch.log(std) - model.LOG_SQRT_2PI
    return -torch.logsumexp(log_w + log_pdf, dim=1)


def train_and_predict(
    *,
    arm: model.Arm,
    hindcast: pl.DataFrame,
    train_cases: model.Cases,
    targets: dict[str, model.Cases],
    cutoff: datetime,
    seed: int,
    half_life_years: float | None,
) -> tuple[dict[str, np.ndarray], nn.Module]:
    """Experiment 429's `train_and_predict` with a recency weight on the training loss."""
    torch.manual_seed(seed)
    usable = (
        np.ones(len(hindcast), dtype=bool)
        if arm.rows == "all"
        else (~hindcast["masked"]).to_numpy()
    )
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
    net = model.Net(
        arm=arm, stats=stats, plants=model.build_plants() if arm.plant == "e2e" else None
    ).to(DEVICE)

    def prepared(cases: model.Cases) -> tuple[torch.Tensor, torch.Tensor]:
        aux = torch.tensor((cases.aux_np - aux_mean) / aux_std, dtype=torch.float32, device=DEVICE)
        summ = torch.tensor(
            (cases.summary_np - summ_mean) / summ_std, dtype=torch.float32, device=DEVICE
        )
        return aux, summ

    aux_train, summ_train = prepared(train_cases)
    plant_params = [p for name, p in net.named_parameters() if name.startswith("plants.")]
    other = [p for name, p in net.named_parameters() if not name.startswith("plants.")]
    groups = [{"params": other, "lr": model.LEARNING_RATE, "weight_decay": model.WEIGHT_DECAY}]
    if plant_params:
        groups.append(
            {"params": plant_params, "lr": model.PLANT_LEARNING_RATE, "weight_decay": 0.0}
        )
    optimiser = torch.optim.AdamW(groups)
    train_i = torch.tensor(train_i_np, device=DEVICE)
    stop_i = torch.tensor(np.flatnonzero(stop), device=DEVICE)
    generator = torch.Generator().manual_seed(seed)
    age_days = (cutoff.timestamp() - init.dt.epoch("us").to_numpy() / 1e6) / 86400.0
    raw = common.row_weights(age_days=age_days, half_life_years=half_life_years)
    weights = np.zeros(len(hindcast), dtype=np.float32)
    weights[train_i_np] = raw[train_i_np] / raw[train_i_np].mean()
    weights_t = torch.tensor(weights, device=DEVICE)

    def batch_nll(
        index: torch.Tensor,
        *,
        cases: model.Cases,
        aux: torch.Tensor,
        summ: torch.Tensor,
        weighted: bool,
    ) -> torch.Tensor:
        parts = net(cases=cases, index=index, aux=aux[index], summary=summ[index])
        per_row = weighted_nll_per_row(parts=parts, y=cases.y[index], head=arm.head)
        return (per_row * weights_t[index]).mean() if weighted else per_row.mean()

    best, best_state = float("inf"), None
    for _ in range(model.EPOCHS):
        net.train()
        order = train_i[torch.randperm(len(train_i), generator=generator).to(DEVICE)]
        for start in range(0, len(order), model.BATCH):
            loss = batch_nll(
                order[start : start + model.BATCH],
                cases=train_cases,
                aux=aux_train,
                summ=summ_train,
                weighted=True,
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        net.eval()
        with torch.no_grad():
            total = sum(
                float(
                    batch_nll(
                        stop_i[s : s + model.EVAL_BATCH],
                        cases=train_cases,
                        aux=aux_train,
                        summ=summ_train,
                        weighted=False,
                    )
                )
                * len(stop_i[s : s + model.EVAL_BATCH])
                for s in range(0, len(stop_i), model.EVAL_BATCH)
            ) / len(stop_i)
        if total < best:
            best, best_state = total, {k: v.detach().clone() for k, v in net.state_dict().items()}
    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    out = {}
    for name, cases in targets.items():
        aux, summ = prepared(cases)
        result = np.empty((cases.size, len(model.LEVELS)), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, cases.size, model.EVAL_BATCH):
                index = torch.arange(s, min(s + model.EVAL_BATCH, cases.size), device=DEVICE)
                parts = net(cases=cases, index=index, aux=aux[index], summary=summ[index])
                result[s : s + len(index)] = model.mixture_quantiles(parts=parts)
        out[name] = np.sort(result, axis=1)
    return out, net


def main() -> None:
    """Fit the requested arms and seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="*", default=list(common.WIND_HALF_LIVES_YEARS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(run429.SEEDS))
    args = parser.parse_args()
    started = time.time()
    table = {
        farm: common461.capacity(site=label, cutoff=common.WIND_CUTOFF)
        for farm, label in common461.WIND_LABEL.items()
    }
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (common.DATA_DIR / "wind_capacity.json").write_text(json.dumps(table, indent=1) + "\n")
    ratio = {farm: float(row["ratio"]) for farm, row in table.items()}
    print("capacity ratio new/old:", ratio, flush=True)
    hindcast = wind461.rescaled(frame=data347.hindcast_frame(), ratio=ratio)
    scored = wind461.rescaled(frame=data347.scored_frame(), ratio=ratio)
    weather_h, power_h = data.load(set_name="hindcast")
    weather_s, power_s = data.load(set_name="scored")
    assert len(weather_h) == len(hindcast), (len(weather_h), len(hindcast))
    assert len(weather_s) == len(scored), (len(weather_s), len(scored))
    summary_h = np.zeros((len(hindcast), 6), dtype=np.float32)
    summary_s = np.zeros((len(scored), 6), dtype=np.float32)
    train = model.Cases(weather=weather_h, power=power_h, summary=summary_h, frame=hindcast)
    target = model.Cases(weather=weather_s, power=power_s, summary=summary_s, frame=scored)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.arms:
        half_life = common.WIND_HALF_LIVES_YEARS[name]
        for seed in args.seeds:
            path = PARTS_DIR / f"{ARM}_{name}_h{seed}.npy"
            if path.exists():
                continue
            began = time.time()
            out, fitted = train_and_predict(
                arm=run429.ARMS[ARM],
                hindcast=hindcast,
                train_cases=train,
                targets={"scored": target},
                cutoff=data347.MAIN_CUTOFF,
                seed=seed,
                half_life_years=half_life,
            )
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as handle:
                np.save(handle, out["scored"].astype(np.float32))
            tmp.replace(path)
            torch.save(fitted.state_dict(), MODELS_DIR / f"{ARM}_{name}_h{seed}.pt")
            print(
                f"{ARM} {name} seed {seed} done in {time.time() - began:.0f} s "
                f"(total {time.time() - started:.0f} s)",
                flush=True,
            )


if __name__ == "__main__":
    main()

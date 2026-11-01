"""Experiment 450, step 2: fit the sweep's arms and seeds, one quantile table per fit.

Each arm is experiment 429's `member_plain` (member route, no plant, clipped head, kept rows,
hidden width 64, depth 2, encoder width 16, 3 mixture components) with the named settings
changed. The A6 arm starts from weights pretrained on ERA5 weather (every ensemble member set to
the ERA5 value) and the observed power before 2024-04-01, then fine-tunes at half the learning
rate on the ensemble hindcast, as experiment 430 did for solar. A configuration is named on the
command line either as an arm below or as `cfg:<name>`, which reads `work/configs.json`.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00450-*/env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue env OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00450-*/e00450_run.py <arm> ... [--seeds 0 1 2 3 4]
```
"""

import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00230_common as c230  # ty: ignore[unresolved-import]
import e00303_common as c303  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00429_data as data  # ty: ignore[unresolved-import]
import e00450_model as model  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00450-wind-upper-tail-check-with-table-p-and-sweep"
)
PARTS_DIR: Final[Path] = DATA_DIR / "parts"
PRETRAINED_DIR: Final[Path] = DATA_DIR / "pretrained"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
PRETRAIN_END: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""ERA5 rows before this time pretrain; the ensemble hindcast starts here."""
PRETRAIN_VALIDATION: Final[timedelta] = timedelta(days=60)
PRETRAIN_LEAD_HOURS: Final[tuple[float, float]] = (12.0, 252.0)
FINE_TUNE_RATE: Final[float] = model.LEARNING_RATE / 2
BASE: Final[model.Arm] = model.Arm(route="member")
ARMS: Final[dict[str, model.Arm]] = {
    "base": BASE,
    "w32": model.Arm(route="member", hidden=32),
    "w128": model.Arm(route="member", hidden=128),
    "d1": model.Arm(route="member", depth=1),
    "d3": model.Arm(route="member", depth=3),
    "e8": model.Arm(route="member", encoder=8),
    "e32": model.Arm(route="member", encoder=32),
    "k2": model.Arm(route="member", components=2),
    "k5": model.Arm(route="member", components=5),
    "a6": BASE,
}
SEED_OFFSET: Final[dict[str, int]] = {}
"""Seeds offset per named configuration; the confirmation uses fresh seeds through `--seeds`."""


def resolve(*, name: str) -> tuple[model.Arm, bool]:
    """Return the arm for `name` and whether it starts from the ERA5-pretrained weights."""
    if name.startswith("cfg:"):
        config = json.loads((DATA_DIR / "work" / "configs.json").read_text())[name[4:]]
        pretrain = bool(config.pop("a6", False))
        return model.Arm(route="member", **config), pretrain
    return ARMS[name], name == "a6"


def era5_cases(*, seed: int) -> tuple[model.Cases, pl.DataFrame]:
    """Return the pretraining cases: ERA5 on every member, observed power, kept rows."""
    sites = c303.site_table()
    table = c230.training_table(sites=sites, era5=c230.era5_rows(sites=sites))
    table = table.filter(pl.col("valid_time") < PRETRAIN_END).join(
        c230.idle_mask().with_columns(masked=pl.lit(value=True)),
        on=["time_series_id", "valid_time"],
        how="left",
    )
    table = table.filter(pl.col("masked").is_null() & pl.col("target").is_finite()).join(
        sites.select("time_series_id", "site"), on="time_series_id"
    )
    rng = np.random.default_rng(10_000 + seed)
    lead = rng.uniform(*PRETRAIN_LEAD_HOURS, size=table.height)
    frame = table.with_columns(
        lead_hours=pl.Series(lead),
        power_fcst_init_time=pl.col("valid_time")
        - pl.Series(lead).map_elements(
            lambda h: timedelta(hours=h), return_dtype=pl.Duration("us")
        ),
        actual_norm=pl.col("target"),
        masked=pl.lit(value=False),
    )
    weather = frame.select(list(c303.WEATHER_COLUMNS)).to_numpy().astype(np.float32)
    members = np.repeat(weather[:, None, :], model.MEMBERS, axis=1)
    cases = model.Cases(
        weather=members,
        power=np.zeros((frame.height, model.MEMBERS), dtype=np.float32),
        summary=np.zeros((frame.height, 6), dtype=np.float32),
        frame=frame,
    )
    return cases, frame


def pretrained_state(
    *, seed: int, reference: tuple[model.Cases, np.ndarray]
) -> dict[str, torch.Tensor]:
    """Return the ERA5-pretrained weights of the base network for `seed` (fitted once, stored)."""
    path = PRETRAINED_DIR / f"base_s{seed}.pt"
    if path.exists():
        return torch.load(path, map_location=model.DEVICE)
    cases, frame = era5_cases(seed=seed)
    cutoff = PRETRAIN_END - PRETRAIN_VALIDATION
    _, fitted = model.train_and_predict(
        arm=BASE, hindcast=frame, train_cases=cases, targets={}, cutoff=cutoff, seed=seed,
        stats_from=reference,
    )  # fmt: skip
    state = {k: v.detach().clone() for k, v in fitted.state_dict().items()}
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


def save(*, name: str, array: np.ndarray) -> None:
    """Write an array atomically."""
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PARTS_DIR / f"{name}.npy"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def main() -> None:
    """Fit every requested arm and seed."""
    args = sys.argv[1:]
    seeds = SEEDS
    if "--seeds" in args:
        at = args.index("--seeds")
        seeds = tuple(int(s) for s in args[at + 1 :])
        args = args[:at]
    started = time.time()
    hindcast = data347.hindcast_frame()
    scored = data347.scored_frame()
    weather_h, power_h = data.load(set_name="hindcast")
    weather_s, power_s = data.load(set_name="scored")
    assert len(weather_h) == len(hindcast), (len(weather_h), len(hindcast))
    assert len(weather_s) == len(scored), (len(weather_s), len(scored))
    zeros_h = np.zeros((len(hindcast), 6), dtype=np.float32)
    zeros_s = np.zeros((len(scored), 6), dtype=np.float32)
    train = model.Cases(weather=weather_h, power=power_h, summary=zeros_h, frame=hindcast)
    target = model.Cases(weather=weather_s, power=power_s, summary=zeros_s, frame=scored)
    kept = (~hindcast["masked"]).to_numpy()
    init, valid = hindcast["power_fcst_init_time"], hindcast["valid_time"]
    in_train = kept & ((init < data347.MAIN_CUTOFF) & (valid < data347.MAIN_CUTOFF)).to_numpy()
    for name in args:
        arm, pretrain = resolve(name=name)
        for seed in seeds:
            tag = name.replace(":", "_")
            if (PARTS_DIR / f"{tag}_h{seed}.npy").exists():
                continue
            kwargs: dict = {}
            if pretrain:
                kwargs = {
                    "init_state": pretrained_state(seed=seed, reference=(train, in_train)),
                    "learning_rate": FINE_TUNE_RATE,
                }
            out, _ = model.train_and_predict(
                arm=arm, hindcast=hindcast, train_cases=train, targets={"scored": target},
                cutoff=data347.MAIN_CUTOFF, seed=seed, **kwargs,
            )  # fmt: skip
            save(name=f"{tag}_h{seed}", array=out["scored"])
            print(f"{name} seed {seed} done {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

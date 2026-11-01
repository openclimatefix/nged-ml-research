"""Fit experiment 429's `member_plain` wind network with effective capacity at p99.9.

The fit loop is experiment 429's (`e00429_run.main`), with the one change that the target
`actual_norm` of the hindcast and scored frames, stored as power over the old capacity, is divided
by each farm's `ratio` (new capacity over old). The weather inputs are unchanged and `member_plain`
reads no plant power, so the rescaling touches the training target and the scoring actuals alone.
The new capacity is the 99.9th percentile of the farm's cleaned power before 2025-04-16.
`--identity` fits with ratio 1 as a gate against experiment 429's stored forecasts.
One file per seed, `parts/member_plain_p999_h<seed>.npy` of shape (scored cases, 11), in units of
the new capacity, is written atomically and skipped when it exists.

    source ml-experiments/00429-*/env.sh   # then add the 00461 folder to PYTHONPATH
    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00461-*/e00461_wind_fit.py [--seeds 0 1 2 3 4]
"""

import argparse
import os
import time
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00347_s4_data as data347  # noqa: E402  # ty: ignore[unresolved-import]
import e00429_data as data  # noqa: E402  # ty: ignore[unresolved-import]
import e00429_model as model  # noqa: E402  # ty: ignore[unresolved-import]
import e00429_run as run429  # noqa: E402  # ty: ignore[unresolved-import]
import e00461_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]

PARTS_DIR: Final = common.DATA_DIR / "wind_parts"
MODELS_DIR: Final = common.DATA_DIR / "wind_models"
ARM: Final[str] = "member_plain"
SEEDS: Final[tuple[int, ...]] = run429.SEEDS


def ratios() -> dict[str, float]:
    """Return each farm's (W1 to W3) new over old capacity, and store the capacity table."""
    table = {
        farm: common.capacity(site=label, cutoff=common.WIND_CUTOFF)
        for farm, label in common.WIND_LABEL.items()
    }
    common.write_capacities(name="wind_capacity.json", table=table)
    return {farm: float(row["ratio"]) for farm, row in table.items()}


def rescaled(*, frame: pl.DataFrame, ratio: dict[str, float]) -> pl.DataFrame:
    """Return `frame` with `actual_norm` divided by its farm's capacity ratio."""
    scale = pl.col("site").replace_strict(ratio, return_dtype=pl.Float64)
    return frame.with_columns(actual_norm=pl.col("actual_norm") / scale)


def save(*, name: str, array: np.ndarray) -> None:
    """Write an array atomically."""
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PARTS_DIR / f"{name}.npy"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def main() -> None:
    """Fit the requested seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--identity", action="store_true")
    args = parser.parse_args()
    seeds, identity = args.seeds, args.identity
    tag = "id" if identity else "p999"
    started = time.time()
    ratio = dict.fromkeys(common.WIND_LABEL, 1.0) if identity else ratios()
    print("capacity ratio new/old:", ratio, flush=True)
    hindcast = rescaled(frame=data347.hindcast_frame(), ratio=ratio)
    scored = rescaled(frame=data347.scored_frame(), ratio=ratio)
    weather_h, power_h = data.load(set_name="hindcast")
    weather_s, power_s = data.load(set_name="scored")
    assert len(weather_h) == len(hindcast), (len(weather_h), len(hindcast))
    assert len(weather_s) == len(scored), (len(weather_s), len(scored))
    # member_plain reads neither the plant power nor the summary, so zeros stand in for them.
    summary_h = np.zeros((len(hindcast), 6), dtype=np.float32)
    summary_s = np.zeros((len(scored), 6), dtype=np.float32)
    train = model.Cases(weather=weather_h, power=power_h, summary=summary_h, frame=hindcast)
    target = model.Cases(weather=weather_s, power=power_s, summary=summary_s, frame=scored)
    for seed in seeds:
        if (PARTS_DIR / f"{ARM}_{tag}_h{seed}.npy").exists():
            continue
        out, fitted = model.train_and_predict(
            arm=run429.ARMS[ARM],
            hindcast=hindcast,
            train_cases=train,
            targets={"scored": target},
            cutoff=data347.MAIN_CUTOFF,
            seed=seed,
        )
        save(name=f"{ARM}_{tag}_h{seed}", array=out["scored"])
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(fitted.state_dict(), MODELS_DIR / f"{ARM}_{tag}_h{seed}.pt")
        print(f"{ARM} seed {seed} done {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()

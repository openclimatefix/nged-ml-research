"""Fit the arms of experiment 00429 and store their scored quantiles.

    PYTHONPATH=<429>:<347>:<230>:<303>:<310>:<326>:<336 folders> \
    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00429-*/e00429_run.py ARM [ARM ...] \
        [--seeds 0 1 2 3 4]

One file per arm and seed, `parts/<arm>_h<seed>.npy` of shape (scored cases, 11), is written
atomically and skipped when it exists. The network's state file goes to `models/`.
"""

import os
import sys
import time
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00347_common as common  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_s4_inputs as inputs  # noqa: E402  # ty: ignore[unresolved-import]
import e00429_data as data  # noqa: E402
import e00429_model as model  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]

PARTS_DIR: Final = data.DATA_DIR / "parts"
MODELS_DIR: Final = data.DATA_DIR / "models"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
ARMS: Final[dict[str, model.Arm]] = {
    "g0": model.Arm(route="summary"),
    "one_pass_ms": model.Arm(route="one_pass_ms"),
    "one_pass_m": model.Arm(route="one_pass_m"),
    "member_plain": model.Arm(route="member"),
    "member_frozen": model.Arm(route="member", plant="frozen"),
    "member_e2e": model.Arm(route="member", plant="e2e"),
    "memberwise": model.Arm(route="memberwise"),
    "memberwise_frozen": model.Arm(route="memberwise", plant="frozen"),
    "member_plain_censored": model.Arm(route="member", head="censored"),
    "member_frozen_censored": model.Arm(route="member", plant="frozen", head="censored"),
    "member_e2e_censored": model.Arm(route="member", plant="e2e", head="censored"),
}
SUMMARY_TOLERANCE: Final[float] = 1e-3


def summaries_for(*, power: np.ndarray, frame_sites: np.ndarray) -> np.ndarray:
    """Return experiment 00347's six-number plant summary of every case."""
    flat_top = {
        farm: inputs.plant.PlantParameters.from_dict(
            common.read_json(name=inputs.PLANT_FILES["y2"])[farm]
        ).scale
        for farm in data.FARMS
    }
    out = np.empty((len(power), 6), dtype=np.float32)
    for farm in data.FARMS:
        rows = frame_sites == farm
        out[rows] = inputs.summarise(power=power[rows], flat_top=flat_top[farm])
    return out


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
    summary_h = summaries_for(power=power_h, frame_sites=hindcast["site"].to_numpy())
    summary_s = summaries_for(power=power_s, frame_sites=scored["site"].to_numpy())
    stored = data347.summaries(arm_plant="y2", set_name="hindcast")
    gap = float(np.abs(summary_h - stored).max())
    assert gap < SUMMARY_TOLERANCE, f"summaries differ from experiment 00347's by {gap}"
    print(f"summaries match experiment 00347's (max gap {gap:.2e})", flush=True)
    train = model.Cases(weather=weather_h, power=power_h, summary=summary_h, frame=hindcast)
    target = model.Cases(weather=weather_s, power=power_s, summary=summary_s, frame=scored)
    for name in args:
        arm = ARMS[name]
        for seed in seeds:
            if (PARTS_DIR / f"{name}_h{seed}.npy").exists():
                continue
            out, fitted = model.train_and_predict(
                arm=arm,
                hindcast=hindcast,
                train_cases=train,
                targets={"scored": target},
                cutoff=data347.MAIN_CUTOFF,
                seed=seed,
            )
            save(name=f"{name}_h{seed}", array=out["scored"])
            if "scored_members" in out:
                save(name=f"{name}_members_h{seed}", array=out["scored_members"])
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(fitted.state_dict(), MODELS_DIR / f"{name}_h{seed}.pt")
            print(f"{name} seed {seed} done {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()

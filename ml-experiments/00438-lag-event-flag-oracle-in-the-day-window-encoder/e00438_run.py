# ruff: noqa: E501
"""Fit and forecast the E-oracle arm: experiment 428's E3 network plus the true event flag per position.

**E-oracle** is arm E3 of experiment 428 (three day windows read by a convolution and one attention
head) with one more channel on every window position: 1 where the half-hour is inside an event, else
0, and 0 where the half-hour is unobserved. An event is an injected event of experiment 402's
generator (any non-zero code) or a half-hour of the padded real-event mask of the corrected label
table. The flag is a positive control: the network is given exact trust information, and it never
advances to production.

This module reuses experiment 428's fit loop, rows, prediction, and flags unchanged. It redirects the
modules' output folders to this experiment's own work folder and swaps in a network and an input
class that carry the extra channel. Experiment 428's own folder is only read.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
source <paths>; PYTHONPATH=$PP:ml-experiments/00438-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00438-*/e00438_run.py fit --months 2026-01 --seeds 0
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_models as models  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARM: Final[str] = "E-oracle"
WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00438-lag-event-flag-oracle-in-the-day-window-encoder" / "work"
)
FLAG_CHANNEL: Final[int] = models.CHANNELS
"""Index of the flag channel (after the five channels of experiment 428)."""
_FLAGS: dict[int, torch.Tensor] = {}
"""Event-flag stacks (worlds, primaries, grid) by number of worlds (4 to train, 7 to score)."""


def flag_stack(*, code: np.ndarray, real: np.ndarray, device: str) -> torch.Tensor:
    """Return 1.0 where a half-hour is inside an injected event or the padded real-event mask."""
    flag = (code != 0) | real[None]
    return torch.from_numpy(flag.astype(np.float32)).to(device)


class OracleInputs(fit.Inputs):
    """Experiment 428's inputs plus a flag channel on each window position."""

    def raw(self, *, arm: fit.Arm, rows: torch.Tensor, world: torch.Tensor) -> dict:
        """Return experiment 428's raw inputs, and the world of each row."""
        out = super().raw(arm=arm, rows=rows, world=world)
        out["world"] = world
        return out

    def assemble(self, *, arm: fit.Arm, raw: dict, masked: bool = False) -> tuple:
        """Return `(x, primary, windows)` with the flag as the last channel of `windows`."""
        x, primary, windows = super().assemble(arm=arm, raw=raw, masked=masked)
        flags = _FLAGS[self.power.shape[0]]
        flag = win.take(power=flags, world=raw["world"], primary=raw["primary"], index=raw["index"])
        flag = torch.nan_to_num(flag, nan=0.0) * windows[:, :, 1, :]
        return x, primary, torch.cat([windows, flag[:, :, None, :]], dim=2)


def install(*, device: str) -> None:
    """Register the arm and redirect experiment 428's modules to this experiment's folders."""
    fit.ARMS[ARM] = fit.Arm(ARM, "encoder", 3, True)
    fit.MODELS_DIR = WORK_DIR / "models"
    pr.PRED_DIR = WORK_DIR / "predictions"
    original = fit.make_model

    def make_model(*, arm: fit.Arm, n_inputs: int, n_primaries: int) -> torch.nn.Module:
        if arm.name != ARM:
            return original(arm=arm, n_inputs=n_inputs, n_primaries=n_primaries)
        return models.DayEncoderMdn(
            n_inputs=n_inputs, n_primaries=n_primaries, days=arm.days,
            hidden=fit.SETTING.hidden, layers=fit.SETTING.layers,
            components=fit.SETTING.components, dropout=fit.SETTING.dropout,
            channels=models.CHANNELS + 1,
        )  # fmt: skip

    fit.make_model = make_model
    fit.Inputs = OracleInputs  # ty: ignore[invalid-assignment]
    real = data.real_event_mask()
    for include_eval in (False, True):
        _, code = data.load_stack(include_eval=include_eval)
        _FLAGS[code.shape[0]] = flag_stack(code=code, real=real, device=device)


def main() -> None:
    """Fit or forecast the arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict"))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    install(device=args.device)
    if args.command == "fit":
        power = fit.power_tensor(device=args.device)
        stack, _ = data.load_stack(include_eval=False)
        cache: dict = {}
        for month in args.months:
            for seed in args.seeds:
                print(
                    fit.fit_one(
                        arm_name=ARM, month=month, seed=seed, power=power, stack=stack,
                        device=args.device, rows_cache=cache,
                    ),
                    flush=True,
                )  # fmt: skip
        return
    power_np, code = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(args.device)
    flags = pr.Flags(code=code, real=data.real_event_mask(), power=power_np, device=args.device)
    for month in args.months:
        for seed in args.seeds:
            for line in pr.predict_month(
                arm_name=ARM, month=month, seed=seed, device=args.device, power=power,
                flags=flags, y_clean=power_np[0],
            ):  # fmt: skip
                print(line, flush=True)


if __name__ == "__main__":
    main()

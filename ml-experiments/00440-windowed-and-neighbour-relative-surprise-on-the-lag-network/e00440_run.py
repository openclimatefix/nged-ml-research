# ruff: noqa: E501
"""Fit and forecast S2, S3, and S3-rw: the S1 lag network plus windowed and neighbour surprise.

All three arms are experiment 428's S1 network (the three lagged readings and the surprise at each
lag) with extra input columns from `e00440_features`:

- **S2:** the 8 window statistics centred on each lag (24 columns).
- **S3:** S2 plus the neighbours' window statistics, the neighbour-relative surprise, and the sign
  agreement per lag (54 columns).
- **S3-rw:** S3 with random neighbours that keep each Primary's degree (control).

Everything else (rows, injected worlds, training loop, scoring) is experiment 428's, reused
unchanged. This module redirects 428's output folders to this experiment's work folder and swaps in
an input class that adds the columns and a network with the wider input.

```bash
export DATA_PATH_INTERNAL=<path to the private data folder>
source <paths>; PYTHONPATH=$PP:ml-experiments/00440-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00440-*/e00440_run.py fit --arms S2 --months 2026-01 --seeds 0
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_models as models  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import e00440_features as feat  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00440-windowed-and-neighbour-relative-surprise-on-the-lag-network"
    / "work"
)
NEIGHBOUR_FILE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00383-load-shift-xgboost-oracle-first-part-1"
    / "work"
    / "neighbours_nged.json"
)
EXTRA_COLUMNS: Final[dict[str, int]] = {"S2": feat.N_S2, "S3": feat.N_S3, "S3-rw": feat.N_S3}
ARM_NAMES: Final[tuple[str, ...]] = tuple(EXTRA_COLUMNS)
S1_COLUMNS: Final[int] = 9 + 3
"""Lag columns (values, missing flags, ages) and the surprise at each lag."""


class WindowInputs(fit.Inputs):
    """Experiment 428's inputs plus the window and neighbour surprise columns."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Build the inputs; the stacks of each world are built when first needed."""
        super().__init__(**kwargs)
        self._stacks: dict[int, feat.WorldStack] = {}
        self._tables: tuple[np.ndarray, np.ndarray] | None = None
        context = worlds402.load_context()
        sites = data.sites()
        matrix = feat.neighbour_matrix(sites=sites, path=NEIGHBOUR_FILE)
        self._neighbours = {
            "S2": None,
            "S3": torch.from_numpy(matrix).to(self.x.device),
            "S3-rw": torch.from_numpy(feat.random_neighbours(matrix=matrix)).to(self.x.device),
        }
        self._weights = torch.tensor(
            [context.capacity[s] for s in sites], dtype=torch.float32, device=self.x.device
        )

    def raw(self, *, arm: fit.Arm, rows: torch.Tensor, world: torch.Tensor) -> dict:
        """Return experiment 428's raw inputs, the world of each row, and the unshifted lags."""
        out = super().raw(arm=arm, rows=rows, world=world)
        out["world"] = world
        out["lags_unshifted"] = win.origin_lags(
            power=self.power, world=world, primary=out["primary"], origin=out["origin"],
            target=out["target"],
        )  # fmt: skip
        return out

    def stack(self, *, world: int) -> feat.WorldStack:
        """Return the stacked surprise series of a world (built on first use)."""
        if world not in self._stacks:
            assert self.surprise is not None
            if self._tables is None:
                median, half = self.surprise
                self._tables = (median.cpu().numpy(), half.cpu().numpy())
            median, half = self._tables
            power = self.power[world].cpu().numpy()
            self._stacks[world] = feat.WorldStack(
                power=power, median=median, half=half, device=str(self.x.device)
            )
        return self._stacks[world]

    def assemble(self, *, arm: fit.Arm, raw: dict, masked: bool = False) -> tuple:
        """Return `(x, primary, None)` with the S1 columns and then the extra columns appended."""
        x_in, primary, _ = super().assemble(arm=arm, raw=raw, masked=masked)
        lags = torch.full_like(raw["lags"], float("nan")) if masked else raw["lags"]
        columns = [x_in, self.lag_surprise(raw=raw, lags=lags)]
        extra = torch.zeros(len(primary), EXTRA_COLUMNS[arm.name], device=primary.device)
        if not masked:
            lag_grid = win.lag_indices(origin=raw["origin"], target=raw["target"])
            half = win.take(
                power=self.surprise[1][None], world=torch.zeros_like(primary), primary=primary,
                index=lag_grid,
            )  # fmt: skip
            step = torch.nan_to_num(raw["lags"] - raw["lags_unshifted"], nan=0.0)
            shift = torch.nan_to_num(step / half, nan=0.0)
            for w in torch.unique(raw["world"]).tolist():
                sel = raw["world"] == w
                extra[sel] = feat.row_features(
                    stack=self.stack(world=int(w)), primary=primary[sel], lag_grid=lag_grid[sel],
                    origin=raw["origin"][sel], shift=shift[sel],
                    neighbours=self._neighbours[arm.name], weights=self._weights,
                )  # fmt: skip
        columns.append(extra)
        return torch.cat(columns, dim=1), primary, None


def install() -> None:
    """Register the arms and redirect experiment 428's modules to this experiment's folders."""
    for name in ARM_NAMES:
        fit.ARMS[name] = fit.Arm(name, "lags", 0, True, surprise=True)
    fit.MODELS_DIR = WORK_DIR / "models"
    pr.PRED_DIR = WORK_DIR / "predictions"
    original = fit.make_model

    def make_model(*, arm: fit.Arm, n_inputs: int, n_primaries: int) -> torch.nn.Module:
        if arm.name not in EXTRA_COLUMNS:
            return original(arm=arm, n_inputs=n_inputs, n_primaries=n_primaries)
        return models.LaggedMdn(
            n_inputs=n_inputs + S1_COLUMNS + EXTRA_COLUMNS[arm.name], n_primaries=n_primaries,
            hidden=fit.SETTING.hidden, layers=fit.SETTING.layers,
            components=fit.SETTING.components, dropout=fit.SETTING.dropout,
        )  # fmt: skip

    fit.make_model = make_model
    fit.Inputs = WindowInputs  # ty: ignore[invalid-assignment]


def main() -> None:
    """Fit or forecast the arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict"))
    parser.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    install()
    if args.command == "fit":
        power = fit.power_tensor(device=args.device)
        stack, _ = data.load_stack(include_eval=False)
        cache: dict = {}
        for month in args.months:
            for seed in args.seeds:
                for name in args.arms:
                    print(
                        fit.fit_one(
                            arm_name=name, month=month, seed=seed, power=power, stack=stack,
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
            for name in args.arms:
                for line in pr.predict_month(
                    arm_name=name, month=month, seed=seed, device=args.device, power=power,
                    flags=flags, y_clean=power_np[0],
                ):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()

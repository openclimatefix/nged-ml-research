# ruff: noqa: E501
r"""Fit the ladder arms of experiment 530: experiment 404's lag-free recipe moved towards experiment 511's network.

All arms are monthly-cadence (one fit at the month's first day, scored on the whole month), 2 seeds, the
5 folds of experiment 511, on experiment 511's rows (the weather of the run published at the forecast
origin, so the weather's age grows with the lead). Differences from experiment 404's stored lag-free head:

- `N1`: 404's training recipe (no dropout, no weight decay, Adam at 2e-3, 6 passes over all rows, batches of 4096,
  no early stopping, recency half-life 182.6 days) on 511's rows with the lead columns removed.
- `N2`: `N1` plus 511's two lead columns (`lead`, `lead_sqrt`).
- `N3`: `N2` trained with 511's loop from scratch (dropout 0.1, AdamW weight decay 1e-3, gradient clip 1.0, batches of
  8192, 20% of the rows per epoch, early stopping on 511's held-out weeks) at 511's pretraining rate 5e-4.
- `N4`: `N3` with 511's recency half-life of 90 days.
The endpoint (`M-90`, experiment 529) adds the ERA5, CAMS, and ENS pretraining and a fine-tune at 2e-4.

```bash
PYTHONPATH=<every folder under ml-experiments/> DATA_PATH_INTERNAL=<private data folder> OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00530-*/e00530_fit.py fit --arm N1
```
"""

import argparse
import dataclasses
import time
from dataclasses import dataclass
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final = ML_EXPERIMENTS_DATA_DIR / "00530-lag-free-gap-network-family-ladder" / "work"
SEEDS: Final = (0, 1)
HALF_LIFE_404: Final[float] = 0.5 * 365.25
LR_404: Final[float] = 2e-3
EPOCHS_404: Final[int] = 6
BATCH_404: Final[int] = 4096


@dataclass(frozen=True)
class Arm:
    """How a ladder arm trains."""

    tag: str
    loop: str  # "e404" or "e511"
    lead_columns: bool
    half_life: float


ARMS: Final[dict[str, Arm]] = {
    a.tag: a
    for a in (
        Arm("N1", "e404", False, HALF_LIFE_404),
        Arm("N2", "e404", True, HALF_LIFE_404),
        Arm("N3", "e511", True, HALF_LIFE_404),
        Arm("N4", "e511", True, c.HALF_LIFE_90),
    )
}


def lead_positions() -> list[int]:
    """Return the positions of the two lead columns among 511's 37 inputs."""
    _, names, _ = data.non_lag_layout()
    return [names.index("lead"), names.index("lead_sqrt")]


def train_404(*, model: torch.nn.Module, t: dict[str, torch.Tensor], seed: int, device: str) -> int:
    """Experiment 404's lag-free loop: 6 passes over all rows, Adam at 2e-3, weighted mixture NLL, no stopping."""
    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR_404)
    n = len(t["x"])
    model.train()
    for _ in range(EPOCHS_404):
        order = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_404):
            b = order[start : start + BATCH_404]
            nll = c.row_nll(model=model, x=t["x"][b], p=t["p"][b], z=t["z"][b], outlier=False)
            loss = (t["w"][b] * nll).sum() / t["w"][b].sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()
    return EPOCHS_404


def fit(*, arm: Arm, months: list[str], device: str, limit_seeds: int | None) -> None:
    """Fit and forecast the arm for the months and seeds, skipping forecasts that exist."""
    clean = c.clean_stack()
    drop = [] if arm.lead_columns else lead_positions()
    for month in months:
        outs = {
            s: W / "seed_preds" / arm.tag / f"{month}_s{s}.parquet" for s in SEEDS[:limit_seeds]
        }
        if all(o.exists() for o in outs.values()):
            continue
        mr = k.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        keep = [i for i in range(mr.x.shape[1]) if i not in drop]
        if drop:
            mr = dataclasses.replace(mr, x=mr.x[:, keep])
            test = dataclasses.replace(test, x=test.x[:, keep])
        g = k.GpuMonth(mr=mr, device=device)
        cut = c.month_start(month=month)
        refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=arm.half_life)
        if arm.loop == "e404":
            refit = dataclasses.replace(refit, is_val=np.zeros_like(refit.is_val))
        t = g.refit_tensors(refit=refit)
        for seed, out in outs.items():
            if out.exists():
                continue
            started = time.monotonic()
            torch.manual_seed(seed)
            dropout = 0.0 if arm.loop == "e404" else c.DROPOUT
            model = base.PlainMdn(
                n_inputs=mr.x.shape[1], hidden=c.HIDDEN, layers=c.LAYERS, components=c.COMPONENTS,
                dropout=dropout, n_primaries=len(mr.base.target_mean),
            ).to(device)  # fmt: skip
            if arm.loop == "e404":
                epochs = train_404(model=model, t=t, seed=seed, device=device)
            else:
                epochs, _ = c.train(
                    model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                    lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device,
                )  # fmt: skip
            sel = np.ones(len(test.x), dtype=bool)
            frame = c.forecast_frame(
                model=model,
                outlier=False,
                test=test,
                select=sel,
                rows=mr.base,
                widen=1.0,
                device=device,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            frame.write_parquet(tmp)
            tmp.rename(out)
            print(
                f"{arm.tag}/{month}_s{seed}: {len(refit.index):,} rows, {epochs} epochs, {frame.height:,} forecasts, {time.monotonic() - started:.0f} s",
                flush=True,
            )
            del model
        del g, t, mr, test
        torch.cuda.empty_cache()


def main() -> None:
    """Run the fit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit",))
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    a = parser.parse_args()
    fit(arm=ARMS[a.arm], months=a.months, device=a.device, limit_seeds=a.limit_seeds)
    _ = pl


if __name__ == "__main__":
    main()

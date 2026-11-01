# ruff: noqa: E501
r"""Fit the new networks of experiment 494 and forecast the stored and new networks when stale.

Commands (all use experiment 425's loop unchanged; outputs go under this experiment's `work/`):

- `refits`: arm A, seeds 0 and 1, pretrained and fine-tuned at the cuts 2025-04 and 2025-06, the two
  refits that Stale6 needs and experiment 425 did not store.
- `pretrain-once`: arm `PO`, seeds 0 and 1, fine-tuned at each of the 5 scored months from the
  pretrained arm A network of the 2025-07 cut (experiment 425's stored weights), with the scored
  month's own standardiser.
- `stale`: forecast the scored months with networks refitted `--age` months earlier (arms `Stale3`,
  `Stale6`, and `Stale11`), writing raw quantiles in experiment 425's seed-prediction layout.

```bash
source <paths>   # PYTHONPATH of every folder under ml-experiments/, DATA_PATH_INTERNAL
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00494-*/e00494_fit.py refits
```
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00494-simplify-demand-network-refit-cadence-and-pretrain-once"
    / "work"
)
E425_WORK: Final[Path] = prep425.WORK_DIR / "work"
SCORED_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1)
STALE_MONTHS: Final[dict[int, tuple[str, ...]]] = {11: ("2026-06",)}
"""Stale11 is the single fold 2026-06 from the stored 2025-07 networks."""
NEW_CUTS: Final[tuple[str, ...]] = ("2025-04", "2025-06")
FIRST_STORED_CUT: Final[str] = "2025-07"
PRETRAIN_ONCE: Final[fit425.Arm] = fit425.Arm("PO", embedding=True, pretrained=True)


def shift_month(*, month: str, by: int) -> str:
    """Return `month` (YYYY-MM) moved by `by` months (negative is earlier)."""
    year, mon = map(int, month.split("-"))
    index = year * 12 + (mon - 1) + by
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def refits(*, device: str) -> None:
    """Fit arm A (pretrain and fine-tune) at the cuts that experiment 425 did not store."""
    fit425.OUT_DIR = WORK_DIR
    original = fit397.forecast_month

    def no_forecast(**_: object) -> tuple[pl.DataFrame, None]:
        """Skip the test-month forecast: the cut months have no scored origins (only the fits matter)."""
        return pl.DataFrame({"time_series_id": []}), None

    fit397.forecast_month = no_forecast  # ty: ignore[invalid-assignment]
    try:
        fit425.run_arms(
            months=list(NEW_CUTS),
            arms=[fit425.ARMS["A"]],
            seeds=list(SEEDS),
            cut=True,
            device=device,
        )
    finally:
        fit397.forecast_month = original


def pretrain_once(*, device: str) -> None:
    """Fine-tune every scored month from the 2025-07 pretrained weights of the same seed."""
    fit425.OUT_DIR = WORK_DIR
    original = fit425.pretrained_path

    def fixed(*, arm: fit425.Arm, cut: bool, month: str, seed: int) -> Path:
        """Return experiment 425's stored 2025-07 pretrained weights, whatever the month."""
        del arm, cut, month
        path = E425_WORK / "pretrained" / "A" / f"{FIRST_STORED_CUT}_s{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    fit425.pretrained_path = fixed  # ty: ignore[invalid-assignment]
    try:
        for month in SCORED_MONTHS:
            for seed in SEEDS:
                fixed(arm=PRETRAIN_ONCE, cut=True, month=month, seed=seed)
        fit425.run_arms(
            months=list(SCORED_MONTHS),
            arms=[PRETRAIN_ONCE],
            seeds=list(SEEDS),
            cut=True,
            device=device,
        )
    finally:
        fit425.pretrained_path = original


def model_folder(*, cut_month: str, seed: int) -> Path:
    """Return the stored arm A network of a cut: experiment 425's, else this experiment's."""
    for root in (E425_WORK, WORK_DIR):
        folder = root / "models" / "A" / cut_month / f"s{seed}"
        if (folder / "meta.json").exists():
            return folder
    raise FileNotFoundError(f"no arm A network at cut {cut_month} seed {seed}")


def load_stored(*, folder: Path, device: str) -> tuple[dict, SimpleNamespace]:
    """Return the fits and a stand-in fold (standardiser and target statistics) of a saved network."""
    meta = json.loads((folder / "meta.json").read_text())
    setting = fit397.Setting(**meta["setting"])
    names, _, layout = base.feature_names(scheme="summary")
    with np.load(folder / "standardiser.npz") as stored:
        standardiser = base.Standardiser(
            mean=stored["mean"].copy(), std=stored["std"].copy(), mask=stored["mask"].copy()
        )
    fits = {}
    for key in meta["epochs"]:
        g = None if key == "None" else int(key)
        model = fit397.make_model(
            arm=meta["arm"],
            setting=setting,
            layout=layout,
            n_primaries=len(meta["target_mean"]),
            n_inputs=len(names),
        )
        model.load_state_dict(
            torch.load(folder / ("pooled.pt" if g is None else f"p{g:02d}.pt"), map_location=device)
        )
        model.to(device).eval()
        fits[g] = fit397.Fitted(
            model=model,
            setting=setting,
            arm=meta["arm"],
            seed=meta["seed"],
            epochs=meta["epochs"][key],
            val_nll=meta["val_nll"][key],
        )
    fold = SimpleNamespace(
        standardiser=standardiser,
        target_mean=np.array(meta["target_mean"]),
        target_std=np.array(meta["target_std"]),
    )
    return fits, fold


def stale(*, ages: list[int], device: str) -> None:
    """Forecast each scored month with the networks refitted `age` months earlier."""
    for age in ages:
        arm = f"Stale{age}"
        for month in STALE_MONTHS.get(age, SCORED_MONTHS):
            cut_month = shift_month(month=month, by=-age)
            for seed in SEEDS:
                target = WORK_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"
                if target.exists():
                    continue
                try:
                    folder = model_folder(cut_month=cut_month, seed=seed)
                except FileNotFoundError:
                    print(f"{arm} {month} seed {seed}: no network at cut {cut_month}", flush=True)
                    continue
                fits, fold = load_stored(folder=folder, device=device)
                q, _ = fit397.forecast_month(
                    fits=fits,
                    fold=fold,  # ty: ignore[invalid-argument-type]
                    scheme="summary",
                    month=month,
                    device=device,
                )
                common.atomic_write(frame=q, target=target)
                print(f"{arm} {month} seed {seed} from cut {cut_month}", flush=True)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("refits", "pretrain-once", "stale"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ages", nargs="+", type=int, default=[3, 6, 11])
    args = parser.parse_args()
    if args.command == "refits":
        refits(device=args.device)
    elif args.command == "pretrain-once":
        pretrain_once(device=args.device)
    else:
        stale(ages=args.ages, device=args.device)


if __name__ == "__main__":
    main()

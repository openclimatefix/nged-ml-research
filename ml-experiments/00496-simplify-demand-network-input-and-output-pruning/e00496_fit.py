# ruff: noqa: E501
r"""Fit the pruned demand networks of experiment 496, with experiment 425's loop unchanged.

Every arm is experiment 425's arm A (pooled plain network, Primary embedding, setting `g09`)
behind a fixed input transform, pretrained once at the 2025-07 cut and fine-tuned at each of the 5
scored months, as the PretrainOnce arm of experiment 494 was. Arms:

- `I`: the identity transform, started from experiment 425's stored 2025-07 weights. It is the gate:
  it must reproduce experiment 494's stored PretrainOnce forecasts.
- `a`: the 7 `std_*` columns zeroed. `c`: one day-type column. `d`: the 3 weekly-lag columns zeroed.
- `b`: one Gaussian component (`components = 1`).
- `dF`: arm `d` with fresh pretraining at every fold (the pretrain-once check).
- `X`: the arms named with `--combine`, together.

Each arm's transform is recorded in `work/variants/<arm>.json`; the saved `meta.json` files are those
of experiment 397's `save_fits` and do not carry it.

```bash
source <paths>   # PYTHONPATH of every folder under ml-experiments/, DATA_PATH_INTERNAL
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00496-*/e00496_fit.py fit a c d
```
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_models as models  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00496-simplify-demand-network-input-and-output-pruning" / "work"
)
E425_WORK: Final[Path] = prep425.WORK_DIR / "work"
SCORED_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1)
PRETRAIN_CUT: Final[str] = "2025-07"
BASE_SETTING: Final[fit397.Setting] = fit425.SETTING
ORIGINAL_PATH: Final = fit425.pretrained_path
"""Experiment 425's path function, bound once so that repeated `configure` calls cannot chain."""
NAMES: Final[list[str]] = base.feature_names(scheme="summary")[0]
SPREAD_COLUMNS: Final[tuple[str, ...]] = tuple(n for n in NAMES if n.startswith("std_"))
HOLIDAY_REMOVED: Final[tuple[str, ...]] = ("christmas_days", "christmas_tent", "bank_holiday")
WEEK_LAG_COLUMNS: Final[tuple[str, ...]] = ("lag_week", "lag_week_missing", "lag_week_age")


@dataclass(frozen=True)
class Variant:
    """One simplification: columns to zero, whether to merge the holiday columns, and the output size."""

    zero: tuple[str, ...] = ()
    merge_day_type: bool = False
    components: int = 3

    def __add__(self, other: Variant) -> Variant:
        """Return the two simplifications together."""
        return Variant(
            zero=tuple(dict.fromkeys(self.zero + other.zero)),
            merge_day_type=self.merge_day_type or other.merge_day_type,
            components=min(self.components, other.components),
        )


VARIANTS: Final[dict[str, Variant]] = {
    "I": Variant(),
    "a": Variant(zero=SPREAD_COLUMNS),
    "c": Variant(zero=HOLIDAY_REMOVED, merge_day_type=True),
    "d": Variant(zero=WEEK_LAG_COLUMNS),
    "b": Variant(components=1),
}
VARIANTS["dF"] = VARIANTS["d"]
FRESH_ARMS: Final[frozenset[str]] = frozenset({"dF"})


class PrunedMdn(models.PlainMdn):
    """The plain pooled network behind a fixed input transform, which training and prediction share.

    The transform acts on the standardised row: the day-type column (if merged) is written into the
    `christmas_inside` slot as `bank_holiday + 2 * christmas_inside`, then the zeroed columns are
    multiplied by 0. The state dict is that of `PlainMdn`, so the stored 2025-07 weights load.
    """

    def __init__(self, *, variant: Variant, **kwargs: object) -> None:
        """Build the network and the transform's column indices and keep mask."""
        super().__init__(**kwargs)
        keep = torch.ones(len(NAMES))
        for name in variant.zero:
            keep[NAMES.index(name)] = 0.0
        self.register_buffer("keep", keep, persistent=False)
        self.merge_day_type = variant.merge_day_type
        self.inside = NAMES.index("christmas_inside")
        self.bank = NAMES.index("bank_holiday")

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the raw mixture outputs for the transformed row."""
        if self.merge_day_type:
            x = x.clone()
            x[..., self.inside] = x[..., self.bank] + 2.0 * x[..., self.inside]
        return super().forward(x * self.keep, primary)


def configure(*, name: str, variant: Variant) -> fit425.Arm:
    """Point experiment 425's loop at this arm: its output folder, setting, network, and weights."""
    fit425.OUT_DIR = WORK_DIR
    fit425.SETTING = replace(BASE_SETTING, components=variant.components)
    arm = fit425.Arm(name, embedding=True, pretrained=True)

    def new_model(*, arm: fit425.Arm, fold: prep425.EnsFold) -> torch.nn.Module:
        """Return the untrained pruned network for a fold."""
        del arm
        return PrunedMdn(
            variant=variant,
            n_inputs=fold.x.shape[1],
            hidden=fit425.SETTING.hidden,
            layers=fit425.SETTING.layers,
            components=variant.components,
            dropout=fit425.SETTING.dropout,
            n_primaries=len(fold.target_mean),
        )

    def path(*, arm: fit425.Arm, cut: bool, month: str, seed: int) -> Path:
        """Return the pretrained weights: one file at the 2025-07 cut, or each fold's own for `dF`."""
        return ORIGINAL_PATH(
            arm=arm, cut=cut, month=month if name in FRESH_ARMS else PRETRAIN_CUT, seed=seed
        )

    fit425.new_model = new_model  # ty: ignore[invalid-assignment]
    fit425.pretrained_path = path  # ty: ignore[invalid-assignment]
    return arm


def pretrain_once(*, arm: fit425.Arm, device: str) -> None:
    """Pretrain the arm at the 2025-07 cut, for both seeds, unless the weights exist."""
    refit = base.refit_of(month=PRETRAIN_CUT)
    todo = [
        s
        for s in SEEDS
        if not fit425.pretrained_path(arm=arm, cut=True, month=PRETRAIN_CUT, seed=s).exists()
    ]
    if not todo:
        return
    fold = prep425.build_ens_fold(refit=refit, cut=True)
    fit425.pretrain(
        arms=[arm], fold=fold, refit=refit, month=PRETRAIN_CUT, seeds=todo, cut=True,
        device=device, log={},
    )  # fmt: skip
    del fold


def link_stored_weights(*, arm: fit425.Arm) -> None:
    """Link experiment 425's stored 2025-07 arm A weights as the identity arm's pretrained weights."""
    for seed in SEEDS:
        target = fit425.pretrained_path(arm=arm, cut=True, month=PRETRAIN_CUT, seed=seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = E425_WORK / "pretrained" / "A" / f"{PRETRAIN_CUT}_s{seed}.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        if not target.is_symlink():
            target.symlink_to(E425_WORK / "pretrained" / "A" / f"{PRETRAIN_CUT}_s{seed}.pt")


def fit_arm(*, name: str, variant: Variant, months: list[str], device: str) -> None:
    """Pretrain (once, or per fold for a fresh arm) and fine-tune one arm on the scored months."""
    arm = configure(name=name, variant=variant)
    folder = WORK_DIR / "variants"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(
        json.dumps(
            {
                "zero": list(variant.zero),
                "merge_day_type": variant.merge_day_type,
                "components": variant.components,
            }
        )
    )
    if name == "I":
        link_stored_weights(arm=arm)
    elif name not in FRESH_ARMS:
        pretrain_once(arm=arm, device=device)
    fit425.run_arms(months=months, arms=[arm], seeds=list(SEEDS), cut=True, device=device)


def main() -> None:
    """Fit the named arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit",))
    parser.add_argument("arms", nargs="+", help="arm names: I a b c d dF")
    parser.add_argument("--combine", nargs="+", help="arms to combine into arm X (fit with name X)")
    parser.add_argument("--months", nargs="+", default=list(SCORED_MONTHS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name in args.arms:
        if name == "X":
            variant = sum((VARIANTS[a] for a in args.combine), Variant())
        else:
            variant = VARIANTS[name]
        fit_arm(name=name, variant=variant, months=args.months, device=args.device)


if __name__ == "__main__":
    main()

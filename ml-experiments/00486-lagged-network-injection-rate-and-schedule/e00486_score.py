"""Forecast the scored months with the networks of experiment 486, alone and in the mixture.

Experiment 404's scoring code reads and writes under its own folders and knows only its own arm
names. This script points that code at experiment 486's folders, maps each arm name to the A0-inj
network architecture (the same network; only the training worlds differ), and writes for each arm X
two forecast tables: `X` (the network alone) and `X+lagless` (the fixed 50/50 mixture with the
cross-fitted lagless table, as experiment 404's B-mix). The stored A0-inj and B-mix forecasts and
the A0-inj networks of experiment 404 are linked into experiment 486's folders, so reports and the
encroachment sweep read all arms from one place.

```bash
PYTHONPATH=<as e00486_fit.py shows> \
    uv run --no-sync python ml-experiments/00486-*/e00486_score.py predict --arms R05 R50 \
    --month 2026-06 --seed 0 --device cuda
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00486_fit as fit486  # ty: ignore[unresolved-import]

NETWORK_ARMS: Final[frozenset[str]] = frozenset(fit486.SCHEDULES)
STORED_ARMS: Final[tuple[str, ...]] = ("A0-inj", "B-mix")
PRED_DIR: Final[Path] = fit486.WORK_DIR / "predictions"
SWEEP_DIR: Final[Path] = fit486.WORK_DIR / "sweep"
_ORIGINAL_BUILD = fit.build_network
_STORED_MODELS: Final[Path] = fit.MODEL_DIR
_STORED_PREDICTIONS: Final[Path] = fit.PRED_DIR


def _build_network(*, arm: str, **kwargs):  # noqa: ANN202, ANN003
    """Build the A0-inj architecture for any of experiment 486's arm names."""
    return _ORIGINAL_BUILD(arm="A0-inj" if arm in NETWORK_ARMS else arm, **kwargs)


def link_stored() -> None:
    """Link experiment 404's stored A0-inj networks and the stored A0-inj and B-mix forecasts."""
    for source, target in (
        (_STORED_MODELS / "A0-inj", fit486.MODEL_DIR / "A0-inj"),
        *((_STORED_PREDICTIONS / arm, PRED_DIR / arm) for arm in STORED_ARMS),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)


def point_at_486() -> None:
    """Redirect experiment 404's scoring code to experiment 486's folders and arm names."""
    fit.MODEL_DIR = fit486.MODEL_DIR
    fit.PRED_DIR = PRED_DIR
    fit.WORK_DIR = fit486.WORK_DIR
    fit.build_network = _build_network
    link_stored()


def main() -> None:
    """Forecast a scored month for each arm, alone and mixed with the lagless table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict",))
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worlds", type=int, nargs="*", default=[*score.SCORING_SEEDS, -1])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    point_at_486()
    real = score.real_event_mask()
    original_load = score.load_network
    for arm in args.arms:
        for world in args.worlds:
            alone = score.forecast(
                arm=arm, month=args.month, seed=args.seed, world=world, device=args.device,
                real=real,
            )  # fmt: skip
            score.load_network = _loader_of(original=original_load, arm=arm)
            mixed = score.forecast(
                arm="B-mix", month=args.month, seed=args.seed, world=world, device=args.device,
                real=real,
            )  # fmt: skip
            score.load_network = original_load
            for name, frame in ((arm, alone), (f"{arm}+lagless", mixed)):
                target = score.prediction_path(
                    arm=name, month=args.month, seed=args.seed, world=world
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(target)
                print(
                    f"{name} {args.month} s{args.seed} w{world}: {frame.height:,} rows", flush=True
                )


def _loader_of(*, original, arm: str):  # noqa: ANN001, ANN202
    """Return a `load_network` that loads the given arm's network whatever name it is asked for."""
    chosen = arm

    def load(*, arm: str, month: str, seed: int, device: str):  # noqa: ANN202
        return original(arm=chosen, month=month, seed=seed, device=device)

    return load


if __name__ == "__main__":
    main()

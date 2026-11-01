"""Fit experiment 404's A0-inj and S1 networks for extra fit seeds, unchanged except one mapping.

Experiment 404 trained on the injected worlds of seeds 100, 101, and 102 (the files that exist),
one per fit seed 0 to 2. Fit seeds 3 and 4 have no world file of their own, so they reuse the
training world of seed `seed % 3`. Everything else (rows, six epochs of Adam, the per-seed
cross-fitted lagless table) is experiment 404's `fit_arm`.

```bash
<PYTHONPATH as e00404_fit.py shows> DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00454-*/e00454_fit.py --seeds 3 4
```
"""

import argparse

import e00404_fit as fit  # ty: ignore[unresolved-import]

TRAINING_WORLDS: int = 3


def main() -> None:
    """Fit the arms for the given months and seeds, skipping networks that already exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["A0-inj", "S1"])
    parser.add_argument(
        "--months", nargs="+", default=["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    original = fit.world_power

    def world_power(*, seed: int | None):  # noqa: ANN202
        if seed is not None and seed >= 100 + TRAINING_WORLDS:
            seed = 100 + (seed - 100) % TRAINING_WORLDS
        return original(seed=seed)

    fit.world_power = world_power
    for month in args.months:
        for seed in args.seeds:
            for arm in args.arms:
                if (fit.MODEL_DIR / arm / f"{month}_s{seed}.pt").exists():
                    continue
                fit.fit_arm(
                    arm=arm, month=month, seed=seed, device=args.device, stride=2,
                    epochs=fit.EPOCHS,
                )  # fmt: skip


if __name__ == "__main__":
    main()

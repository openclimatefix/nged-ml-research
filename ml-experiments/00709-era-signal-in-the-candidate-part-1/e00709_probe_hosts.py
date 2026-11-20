"""Experiment 709: how often do the host-separation rules hold at each onset day?

For onset days of the tuning window (every 14th day) and the bank's month starts, print the size of
the host pool and the share of 5000 random draws that satisfy the rules of plan section 20 item 13,
for the old pass (4 era hosts and the four switching kinds) and for each of the four two-and-two
passes of `inject.SWITCH_ROTATION`. Reads only the panel; fits nothing.

Run with the folders of experiments 86, 598, 660, and 687 on `PYTHONPATH`.
"""

from typing import Final

import e00709_features as features
import e00709_inject as inject
import e00709_stage0 as stage0
import numpy as np
import polars as pl

DRAWS: Final[int] = 5000
STEP_DAYS: Final[int] = 14


def share_ok(
    *, panel: features.Panel, pool: list[int], onset_day: int, kinds: tuple, n_era: int, seed: int
) -> float:
    """Return the share of `DRAWS` random assignments that satisfy the rules."""
    rng = np.random.default_rng(seed)
    ok = sum(
        inject.try_hosts(
            panel=panel, pool=pool, onset_day=onset_day, switch_kinds=kinds, n_era=n_era, rng=rng
        )
        is not None
        for _ in range(DRAWS)
    )
    return ok / DRAWS


def main() -> None:
    """Print the feasibility table."""
    panel = features.load_panel()
    feats = features.compute_panel_features(panel=panel)
    ready = np.array([f.ready_day for f in feats])
    window = tuple(features.day_index(day=d) for d in stage0.TUNING_WINDOW)
    days = list(range(window[0] + 60, window[1] - 200, STEP_DAYS))
    days += [features.day_index(day=m) for m in stage0.SCORED_MONTHS]
    rows = []
    for day in days:
        pool = inject.host_pool(panel=panel, ready_days=ready, onset_day=day)
        row: dict[str, object] = {"day": str(features.day_of(index=day)), "pool": len(pool)}
        for name, world in (("tuning", "tuning"), ("bank", "bank")):
            old = inject.WORLD_SETTINGS[world].switch_kinds
            row[f"{name}_old_4+4"] = share_ok(
                panel=panel, pool=pool, onset_day=day, kinds=old, n_era=4, seed=day
            )
        base = inject.WORLD_SETTINGS["bank"].switch_kinds
        for r, idx in enumerate(inject.SWITCH_ROTATION):
            kinds = tuple(base[i] for i in idx)
            row[f"new_2+2_rot{r}"] = share_ok(
                panel=panel, pool=pool, onset_day=day, kinds=kinds, n_era=inject.N_ERA_PER_PASS,
                seed=day + r,
            )  # fmt: skip
        rows.append(row)
    table = pl.DataFrame(rows)
    with pl.Config(tbl_rows=400, tbl_cols=12):
        print(table)
    table.write_csv(features.DATA_DIR / "host_feasibility.csv")


if __name__ == "__main__":
    main()

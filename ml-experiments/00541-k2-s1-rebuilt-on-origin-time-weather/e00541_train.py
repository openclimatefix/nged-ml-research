# ruff: noqa: E501
r"""Experiment 541, second rebuild: lag-free heads retrained on origin-time weather.

The script refits the lag-free heads on origin-time weather and rebuilds K2-S1 with them.

Part 1 fed origin-time weather to lag-free heads trained on experiment 404's freshest-run weather.
Here each "further" module trains on the rows of experiment 397's summary tables instead: one row
per (Primary, forecast origin, valid time) with the weather of the run issued at the origin, for
origins and valid times before the refit date, leads up to 240 h, the same 35 non-lag inputs, the
same module, loss, seeds, epochs, recency weights (by the valid time's age), and short-history cut
as experiment 404. The refit date is the same as experiment 404's, so nothing from the scored month
enters a fit.

```bash
PYTHONPATH=$(ls -d ml-experiments/*/ | tr '\n' ':') DATA_PATH_INTERNAL=<internal data folder> \
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00541-*/e00541_train.py
```
"""

import argparse
import time

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_data as d404  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00454_build as b454  # ty: ignore[unresolved-import]
import e00521_refs as r521  # ty: ignore[unresolved-import]
import e00541_common as m  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00541_a531 import head

ARM = "K2-S1-origin-trained"
SEEDS = (0, 1, 2)


def origin_pool(*, end: int, start: int) -> lagless.Pool:
    """Return the origin-time rows whose origin and valid time lie in `[start, end)` grid indices."""
    power = lagless.power_arrays()
    keep_cols, names, _ = data.non_lag_layout()
    assert names[:-2] == d404.lagless_columns()
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        valid_idx = (z["valid"] - prep422.GRID_START_US) // prep422.STEP_US
        origin_idx = (z["origin"] - prep422.GRID_START_US) // prep422.STEP_US
        keep = (
            (origin_idx >= start)
            & (valid_idx < end)
            & (valid_idx >= start)
            & (z["lead"] <= data.HEADLINE_MAX_HOURS)
        )
        assert (origin_idx[keep] < end).all()
        grid = valid_idx[keep].astype(np.int64)
        parts.append(
            lagless.Pool(
                site=np.full(len(grid), row["primary"], dtype=np.int64),
                grid=grid,
                x=z["x"][keep][:, keep_cols][:, :35].astype(np.float32),
                y=power[row["site"]][grid],
            )
        )
        del z
    return lagless.Pool(
        *(np.concatenate([getattr(p, f) for p in parts]) for f in ("site", "grid", "x", "y"))
    )


def fit_origin_module(*, month: str, seed: int, device: str, cache: dict) -> tuple:
    """Fit one lag-free module on origin-time training rows; return it and its scaler."""
    refit = prep.refit_of(month=month)
    end = int(common.grid_index(times=refit))
    start = max(lagless.data.FIRST_VALID_INDEX, end - lagless.WINDOW_STEPS)
    if cache.get("month") != month:
        cache.clear()
        cache["month"], cache["train"] = month, origin_pool(end=end, start=start)
        print(f"{month}: {len(cache['train'].grid):,} origin-time training rows", flush=True)
    train = cache["train"]
    age_days = (end - train.grid) / common.STEPS_PER_DAY
    weight = (0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)).astype(np.float32)
    fit_mask = np.isfinite(train.y)
    scaler = lagless.Scaler(
        x=train.x[fit_mask], site=train.site[fit_mask], y=train.y[fit_mask], n_primaries=16
    )
    z = scaler.target(y=train.y[fit_mask], site=train.site[fit_mask])
    module = lagless.fit_module(
        x=torch.tensor(scaler.inputs(x=train.x[fit_mask]), device=device), z=torch.tensor(z, device=device),
        site=torch.tensor(train.site[fit_mask], device=device), weight=torch.tensor(weight[fit_mask], device=device), seed=seed,
    )  # fmt: skip
    module.eval()
    return module, scaler


def run(*, months: list[str], device: str) -> None:
    """Build the three cells of the trained-on-origin K2-S1 for each month."""
    levels = list(r521.LEVELS)
    cache: dict = {}
    for month in months:
        started = time.monotonic()
        raw = m.raw_test(month=month, mode="origin", dense={})
        keys = pl.DataFrame(
            {
                "primary": raw["primary"].astype(np.int16),
                "origin": raw["origin_idx"].astype(np.int32),
                "target": raw["target_idx"].astype(np.int32),
            }
        ).with_row_index("li")
        lag_free = {}
        for seed in SEEDS:
            module, scaler = fit_origin_module(month=month, seed=seed, device=device, cache=cache)
            lag_free[seed] = head(
                module=module,
                scaler=scaler,
                x_raw=raw["x"][:, :35],
                site=raw["primary"],
                device=device,
            )
            del module
            torch.cuda.empty_cache()
            print(f"{month} seed {seed} fitted, {time.monotonic() - started:.0f} s", flush=True)
        s1 = {}
        s1_keys = None
        for seed in SEEDS:
            saved = torch.load(b454.head_path(kind="S", month=month, seed=seed), weights_only=False)
            s1[seed] = saved["head"]
            s1_keys = s1_keys or saved["keys"]
        assert s1_keys is not None
        skeys = pl.DataFrame(
            {
                k: s1_keys[k].astype(dt)
                for k, dt in (("primary", np.int16), ("origin", np.int32), ("target", np.int32))
            }
        ).with_row_index("si")
        both = keys.join(skeys, on=["primary", "origin", "target"], how="inner")
        assert both.height == len(keys) <= len(skeys)
        li, si = both["li"].to_numpy().astype(np.int64), both["si"].to_numpy().astype(np.int64)
        ref = pl.read_parquet(r521.E454 / "K2-S1" / f"{month}_s0_w-1.parquet").select(
            *r521.KEYS, "actual"
        )
        pick = lambda h, idx: tuple(t[torch.from_numpy(idx)] for t in h)  # noqa: E731
        base = both.select("primary", "origin", "target")
        for cell, (i, j) in enumerate(b454.PAIRS):
            members = [
                pick(s1[i], si),
                pick(s1[j], si),
                pick(lag_free[i], li),
                pick(lag_free[j], li),
            ]
            q = b454.mix_equal(heads=members)
            frame = base.hstack(
                pl.DataFrame(q.astype(np.float32), schema=levels, orient="row")
            ).join(ref, on=r521.KEYS, how="left")
            path = m.WORK / "cells" / ARM / f"{month}_c{cell}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path)
        print(f"{month}: {len(both):,} rows, {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Run the training build."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--months", nargs="+", default=list(m.MONTHS))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(months=args.months, device=args.device)


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Experiment 531, `e00531_a.py`: kept for the record.

Part A of experiment 531: experiment 404's lag-free head scored with the weather of the run
published at the forecast origin.

Experiment 404 stored the lag-free head only as output tables, never as weights. This script
therefore refits the table's "further" module (the one that forecasts the scored month) with
404's own code and seed, checks that the refit reproduces the stored table, and then feeds it
either 404's freshest-run weather (reproduction) or the weather of 511's rows (the run published
at the forecast origin).
"""

import argparse
import time

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00454_build as b454  # ty: ignore[unresolved-import]
import e00521_refs as r521  # ty: ignore[unresolved-import]
import e00531_common as m  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

SEEDS = (0, 1, 2)


def fit_further(
    *, month: str, seed: int, device: str
) -> tuple[models.LaglessModule, lagless.Scaler, lagless.Pool]:
    """Refit the further module of 404's table.

    This copies the last fit in `crossfit_table`. It returns the module, its scaler, and the pool
    it forecasts.
    """
    refit = prep.refit_of(month=month)
    end = int(common.grid_index(times=refit))
    start = max(lagless.data.FIRST_VALID_INDEX, end - lagless.WINDOW_STEPS)
    train = lagless.build_pool(lo=start, hi=end)
    after = lagless.build_pool(lo=end, hi=end + lagless.FURTHER_STEPS)
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
    return module, scaler, after


@torch.no_grad()
def head(
    *,
    module: models.LaglessModule,
    scaler: lagless.Scaler,
    x_raw: np.ndarray,
    site: np.ndarray,
    device: str,
) -> b454.Head:
    """Return the mixture (log weights, means, standard deviations in capacity fractions) for raw inputs."""
    out = []
    for a in range(0, len(x_raw), 65_536):
        xs = torch.tensor(scaler.inputs(x=x_raw[a : a + 65_536]), device=device)
        out.append(module(xs, torch.tensor(site[a : a + 65_536], device=device)).double().cpu())
    log_w, mean, std = models.mixture_parts(out=torch.cat(out), components=models.COMPONENTS)
    y_mean = torch.tensor(scaler.y_mean)[torch.from_numpy(site)][:, None]
    y_std = torch.tensor(scaler.y_std)[torch.from_numpy(site)][:, None]
    return log_w, mean * y_std + y_mean, std * y_std


def run(*, months: list[str], device: str) -> None:
    """Fit the heads, check the reproduction, and write the pair-mixture cells for both weather feeds."""
    dense = m.dense_by_primary()
    levels = list(r521.LEVELS)
    log_lines = []
    for month in months:
        started = time.monotonic()
        ref = pl.read_parquet(r521.E454 / "K2-S1" / f"{month}_s0_w-1.parquet").select(
            *r521.KEYS, "actual"
        )
        raw = {
            mode: m.raw_test(month=month, mode=mode, dense=dense) for mode in ("origin", "fresh")
        }
        heads: dict[str, dict[int, b454.Head]] = {"origin": {}, "fresh": {}}
        for seed in SEEDS:
            module, scaler, after = fit_further(month=month, seed=seed, device=device)
            # reproduction against the stored table's raw outputs
            table = pl.read_parquet(lagless.table_path(month=month, seed=seed)).filter(
                pl.col("source") == "further"
            )
            out = []
            with torch.no_grad():
                for a in range(0, len(after.x), 65_536):
                    xs = torch.tensor(scaler.inputs(x=after.x[a : a + 65_536]), device=device)
                    out.append(
                        module(xs, torch.tensor(after.site[a : a + 65_536], device=device))
                        .cpu()
                        .numpy()
                    )
            mine = pl.DataFrame(
                {
                    "primary": after.site.astype(np.int16),
                    "grid": after.grid.astype(np.int32),
                    **{f"m{i}": np.concatenate(out)[:, i] for i in range(9)},
                }
            )
            j = table.join(mine, on=["primary", "grid"])
            diff = max(float((j[f"raw_{i}"] - j[f"m{i}"]).abs().max()) for i in range(9))  # ty: ignore[invalid-argument-type]
            log_lines.append(
                f"{month} seed {seed}: refit vs stored raw outputs on {j.height:,} of {table.height:,} further rows, max |difference| {diff:.3e}"
            )
            print(log_lines[-1], flush=True)
            for mode in ("origin", "fresh"):
                heads[mode][seed] = head(
                    module=module,
                    scaler=scaler,
                    x_raw=raw[mode]["x"][:, :35],
                    site=raw[mode]["primary"],
                    device=device,
                )
            del module
            torch.cuda.empty_cache()
        for mode in ("origin", "fresh"):
            keys = pl.DataFrame(
                {
                    "primary": raw[mode]["primary"].astype(np.int16),
                    "origin": raw[mode]["origin_idx"].astype(np.int32),
                    "target": raw[mode]["target_idx"].astype(np.int32),
                }
            )
            for cell, (i, j) in enumerate(b454.PAIRS):
                q = b454.mix_equal(heads=[heads[mode][i], heads[mode][j]])
                frame = keys.hstack(
                    pl.DataFrame(q.astype(np.float32), schema=levels, orient="row")
                ).join(ref, on=r521.KEYS, how="left")
                path = m.WORK / "cells" / f"LL-ens2-{mode}" / f"{month}_c{cell}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(path)
        log_lines.append(
            f"{month}: rows origin {len(raw['origin']['x']):,}, fresh {len(raw['fresh']['x']):,} (dropped {int(raw['fresh']['dropped'][0]):,}), {time.monotonic() - started:.0f} s"
        )
        print(log_lines[-1], flush=True)
        (m.WORK / f"a_log_{month}.txt").write_text("\n".join(log_lines) + "\n")


def main() -> None:
    """Run part A."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--months", nargs="+", default=list(m.MONTHS))
    p.add_argument("--device", default="cuda")
    run(months=p.parse_args().months, device=p.parse_args().device)


if __name__ == "__main__":
    main()

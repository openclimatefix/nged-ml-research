# ruff: noqa: E501
r"""Build experiment 541's rebuilt K2-S1: two S1 heads (experiment 454, origin-time weather) plus two lag-free heads refitted and fed origin-time weather.

Experiment 454's S1 heads read experiment 397's summary rows, whose weather is the run published at the
forecast origin; their lag inputs sit before the origin. Only the lag-free heads (experiment 404's frozen
table) read the freshest run published before each target's valid time. This script refits each lag-free
"further" module (experiment 531's code and seeds 0 to 2), feeds it the weather of the 397 rows, asserts row
by row that the weather is the summary of the run initialised at origin minus 9 h, mixes it with the stored
S1 heads of 454 in 454's three seed pairs, and writes one forecast table per month and cell.

```bash
PYTHONPATH=$(ls -d ml-experiments/*/ | tr '\n' ':') DATA_PATH_INTERNAL=<internal data folder> \
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00541-*/e00541_build.py
```
"""

import argparse
import time
from datetime import UTC, datetime, timedelta

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00454_build as b454  # ty: ignore[unresolved-import]
import e00521_refs as r521  # ty: ignore[unresolved-import]
import e00541_common as m  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

PUBLICATION_DELAY_US = int(timedelta(hours=9).total_seconds() * 1e6)
DAY_US = 86_400 * 10**6
SEEDS = (0, 1, 2)


def fit_further(*, month: str, seed: int, device: str):  # noqa: ANN201
    """Refit the lag-free "further" module of one month and seed (experiment 531's function)."""
    import e00541_a531 as a531  # ty: ignore[unresolved-import]

    return a531.fit_further(month=month, seed=seed, device=device)


def lagfree_head(*, module, scaler, raw, device: str) -> b454.Head:  # noqa: ANN001
    """Return the lag-free head's mixture on the scored rows (experiment 531's function)."""
    import e00541_a531 as a531  # ty: ignore[unresolved-import]

    return a531.head(
        module=module, scaler=scaler, x_raw=raw["x"][:, :35], site=raw["primary"], device=device
    )


def assert_origin_time_weather(*, raw: dict[str, np.ndarray]) -> str:
    """Assert, for every row, that the weather columns equal the summary of the run issued at origin - 9 h.

    Returns the printed example row.
    """
    cols = m.weather_names()
    from e00428_data import non_lag_layout  # ty: ignore[unresolved-import]

    _, nl_names, _ = non_lag_layout()
    idx = [nl_names.index(c) for c in cols]
    example = ""
    total = 0
    for row in prep.primaries().iter_rows(named=True):
        sel = raw["primary"] == row["primary"]
        if not sel.any():
            continue
        origin, valid, x = raw["origin"][sel], raw["valid"][sel], raw["x"][sel][:, idx]
        run_init = origin - PUBLICATION_DELAY_US
        assert (run_init % DAY_US == 0).all(), "a run initialisation is not 00:00 UTC"
        summ = prep.load_summary(series_id=row["time_series_id"]).with_columns(
            r=pl.col("power_fcst_init_time").dt.epoch("us"), v=pl.col("valid_time").dt.epoch("us")
        )
        key = pl.DataFrame({"r": origin, "v": valid, "i": np.arange(len(origin))})
        joined = key.join(summ, on=["r", "v"], how="left").sort("i")
        assert joined.height == len(origin)
        got = joined.select(cols).to_numpy().astype(np.float64)
        rain = [k for k, c in enumerate(cols) if c.endswith("_precipitation")]
        got[:, rain] = np.log1p(
            np.clip(got[:, rain], 0, None)
        )  # the stored matrix applies log1p to rainfall
        assert np.isfinite(got).all(), f"{row['site']}: a row has no summary of the origin-time run"
        assert np.allclose(got, x, rtol=1e-4, atol=1e-4), (
            f"{row['site']}: weather differs from the origin-time run"
        )
        total += len(origin)
        if not example:
            k = int(np.argmax(valid - origin > 100 * 3600 * 10**6))
            us = lambda v: datetime.fromtimestamp(int(v) / 1e6, tz=UTC)  # noqa: E731
            example = (
                f"example row: Primary {row['site']}, origin {us(origin[k])}, target {us(valid[k])}, run init {us(run_init[k])}, "
                f"published {us(run_init[k] + PUBLICATION_DELAY_US)}, lead {(valid[k] - origin[k]) / 3.6e9:.1f} h, mean_shortwave {x[k][0]:.3f} = run's {got[k][0]:.3f}"
            )
            assert run_init[k] + PUBLICATION_DELAY_US <= origin[k]
    return f"{total:,} rows asserted. {example}"


def run(*, months: list[str], device: str) -> None:
    """Build the three cells of the rebuilt K2-S1 for each month."""
    levels = list(r521.LEVELS)
    for month in months:
        started = time.monotonic()
        raw = m.raw_test(month=month, mode="origin", dense={})
        check = assert_origin_time_weather(raw=raw)
        print(month, check, flush=True)
        keys = pl.DataFrame(
            {
                "primary": raw["primary"].astype(np.int16),
                "origin": raw["origin_idx"].astype(np.int32),
                "target": raw["target_idx"].astype(np.int32),
            }
        ).with_row_index("li")
        lag_free = {}
        for seed in SEEDS:
            module, scaler, _ = fit_further(month=month, seed=seed, device=device)
            lag_free[seed] = lagfree_head(module=module, scaler=scaler, raw=raw, device=device)
            del module
            torch.cuda.empty_cache()
        s1 = {}
        s1_keys = None
        for seed in SEEDS:
            saved = torch.load(b454.head_path(kind="S", month=month, seed=seed), weights_only=False)
            s1[seed] = saved["head"]
            if s1_keys is None:
                s1_keys = saved["keys"]
            assert all(np.array_equal(s1_keys[k], saved["keys"][k]) for k in b454.KEY_COLUMNS)
        assert s1_keys is not None
        skeys = pl.DataFrame(
            {
                k: s1_keys[k].astype(dt)
                for k, dt in (("primary", np.int16), ("origin", np.int32), ("target", np.int32))
            }
        ).with_row_index("si")
        both = keys.join(skeys, on=["primary", "origin", "target"], how="inner")
        assert both.height == len(keys) <= len(skeys), (
            f"{month}: lag-free rows {len(keys)}, S1 rows {len(skeys)}, common {both.height}"
        )
        assert not keys.select("primary", "origin", "target").is_duplicated().any()
        li, si = both["li"].to_numpy().astype(np.int64), both["si"].to_numpy().astype(np.int64)
        ref = pl.read_parquet(r521.E454 / "K2-S1" / f"{month}_s0_w-1.parquet").select(
            *r521.KEYS, "actual"
        )
        pick = lambda head, idx: tuple(t[torch.from_numpy(idx)] for t in head)  # noqa: E731
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
            path = m.WORK / "cells" / "K2-S1-origin" / f"{month}_c{cell}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path)
        line = f"{month}: lag-free rows {len(keys):,}, S1 rows {len(skeys):,}, common {len(both):,}, {time.monotonic() - started:.0f} s"
        print(line, flush=True)
        (m.WORK / f"build_{month}.txt").write_text(check + "\n" + line + "\n")


def main() -> None:
    """Run the build."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--months", nargs="+", default=list(m.MONTHS))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    m.WORK.mkdir(parents=True, exist_ok=True)
    run(months=args.months, device=args.device)


if __name__ == "__main__":
    main()

"""Bad-data injections of experiment 465 applied to the networks of experiment 486.

Experiment 465 (unmerged when this was written; its branch `robust-lags-median-lags`) corrupts the
newest usable readings of one Primary and rebuilds every lag from the corrupted readings. This
script reuses its three definitions and experiment 404's sweep machinery (rows, networks, bands):

- `stuck`: the 48 newest usable readings repeat the reading just before them.
- `dropout`: the 12 newest usable readings are zero.
- `spike`: the 4 newest usable readings rise by 4 and by 20 times the series' median absolute change
  between a half-hour and the same half-hour a week earlier (experiment 465's stand-in for the
  stage-1 residual spread).

The response is the shift of the median forecast, and the mean change of the corrupted readings
only, both as fractions of capacity. The report sums the shift (signed to follow the
corruption) over rows and divides by the summed size of the corruption.

```bash
PYTHONPATH=<as e00486_fit.py shows> \
    uv run --no-sync python ml-experiments/00486-*/e00486_bad.py run --month 2026-06 --seed 0 \
    --arms A0-inj R05 R50 Cur-down Cur-up --device cuda
```
"""

import argparse
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00404_sweep as sweep404  # ty: ignore[unresolved-import]
import e00486_score as score486  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

NEWEST_AGE: Final[int] = 7
"""Half-hours between an origin and the newest usable reading (`newest = origin - 7` in 404)."""
WEEK: Final[int] = 336
SPIKE_MULTIPLES: Final[tuple[float, ...]] = (4.0, 20.0)
VARIANTS: Final[tuple[tuple[str, float], ...]] = (
    ("stuck", 0.0),
    ("dropout", 0.0),
    ("spike", 4.0),
    ("spike", 20.0),
)


def week_spread(*, power: np.ndarray, end: int) -> float:
    """Return the median absolute change from the same half-hour a week earlier, up to `end`."""
    start = max(WEEK, end - 8 * WEEK)
    difference = np.abs(power[start:end] - power[start - WEEK : end - WEEK])
    return float(np.nanmedian(difference[np.isfinite(difference)]))


def corrupt(
    *, power: np.ndarray, primary: int, newest: int, kind: str, multiple: float
) -> np.ndarray:
    """Return a copy of `power` with the Primary's newest usable readings corrupted."""
    out = power.copy()
    row = out[primary]
    if kind == "stuck":
        row[newest - 47 : newest + 1] = row[newest - 48]
    elif kind == "dropout":
        row[newest - 11 : newest + 1] = 0.0
    else:
        row[newest - 3 : newest + 1] += multiple * week_spread(power=power[primary], end=newest)
    return out


@torch.no_grad()
def run(*, month: str, seed: int, arms: list[str], device: str) -> pl.DataFrame:
    """Return one row per arm, variant, Primary, and origin with the shift and the corruption."""
    clean = pl.read_parquet(
        score.prediction_path(arm=arms[0], month=month, seed=seed, world=-1)
    ).with_row_index("row")
    day = clean["origin"].to_numpy().astype(np.int64) // 48
    chosen = (day % sweep404.ORIGIN_STRIDE == 0) & ~(
        clean["tgt_real"].to_numpy() | clean["tgt_post"].to_numpy() | clean["win_real"].to_numpy()
    )
    row_ids = clean["row"].to_numpy().astype(np.int64)[chosen]
    loaded = {a: score.load_network(arm=a, month=month, seed=seed, device=device) for a in arms}
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=1, saved=loaded[arms[0]][1]
    )
    sub = sweep404.subset(pool=test, idx=torch.tensor(row_ids, device=device))
    base_power, _ = fit.world_power(seed=None)
    table = pl.read_parquet(fit.lagless.table_path(month=month, seed=seed))
    neighbours = rows_mod.neighbour_matrix(
        sites=fit.prep.primaries()["site"].to_list(), path=fit._neighbour_path()
    )
    metered = torch.tensor(fit.metered_indicator(), device=device)

    def medians(*, power_np: np.ndarray, rows: torch.Tensor) -> dict[str, np.ndarray]:
        stack = rows_mod.build_stack(
            power=power_np, table=table, neighbours=neighbours, device=device
        )
        power = torch.tensor(power_np, device=device)
        pool = sweep404.subset(pool=sub, idx=rows)
        out = {}
        for arm in arms:
            scalers.trust_mean = loaded[arm][1]["trust_mean"]
            scalers.trust_std = loaded[arm][1]["trust_std"]
            out[arm] = sweep404.band(
                net=loaded[arm][0], pool=pool, power=power, stack=stack, lag_stats=lag_stats,
                scalers=scalers, metered=metered, device=device, masked=(),
            )[:, 1]  # fmt: skip
        return out

    records = []
    for origin in np.unique(sub.origin.cpu().numpy()):
        origin_rows = torch.nonzero(sub.origin == int(origin)).flatten()
        baseline = medians(power_np=base_power, rows=origin_rows)
        newest = int(origin) - NEWEST_AGE
        for p in range(16):
            rows = origin_rows[sub.primary[origin_rows] == p]
            if len(rows) == 0:
                continue
            local = torch.isin(origin_rows, rows).cpu().numpy()
            for kind, multiple in VARIANTS:
                shifted = corrupt(
                    power=base_power, primary=p, newest=newest, kind=kind, multiple=multiple
                )
                window = {"stuck": 48, "dropout": 12, "spike": 4}[kind]
                size = float(
                    np.nanmean(
                        shifted[p, newest - window + 1 : newest + 1]
                        - base_power[p, newest - window + 1 : newest + 1]
                    )
                )
                result = medians(power_np=shifted, rows=rows)
                for arm, q in result.items():
                    records.append(
                        pl.DataFrame(
                            {
                                "arm": arm,
                                "variant": f"{kind}{int(multiple) or ''}",
                                "primary": p,
                                "shift": q - baseline[arm][local],
                                "corruption": size,
                            }
                        )
                    )
    return pl.concat(records)


def main() -> None:
    """Run the injections for one cell and print the shift as a share of the corruption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    score486.point_at_486()
    frame = run(month=args.month, seed=args.seed, arms=args.arms, device=args.device)
    out = score486.fit486.WORK_DIR / "bad" / f"{args.month}_s{args.seed}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)
    frame = frame.filter(pl.col("shift").is_finite() & pl.col("corruption").is_finite())
    summary = frame.group_by("arm", "variant").agg(
        pl.len().alias("rows"),
        (
            (pl.col("shift") * pl.col("corruption").sign()).sum() / pl.col("corruption").abs().sum()
        ).alias("shift_share_of_corruption"),
    )
    print(summary.sort("variant", "arm"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

"""Stage 1 of experiment 626: fit the factors of arms B, C, and D on earlier months only.

For each test month, fits on all runs of earlier months, per family and lead band: B's single
factor, D's upper and lower factors, and C's level stretch, each chosen on a grid to bring the
q05 and q95 exceedance rates of the training rows to 5%. Writes `work/factors.parquet` and prints
the table and the stability check of the registered stop rule. It reads no score of a test month.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00626-*/env.sh
nice -n 19 uv run --no-sync python ml-experiments/00626-*/e00626_fit.py
```
"""

import e00626_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl


def rows(*, runs: list[dict]) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Return, per (family, band), the pooled training rows: actual, kind, and the 51 values."""
    band = c.band_of_step()
    out: dict[tuple[int, int], list] = {}
    for run in runs:
        v, a = run["values"], run["actual"]
        ok = ~np.isnan(a) & (v[..., -1] > v[..., 0])
        for fam in range(3):
            node = np.asarray(fam == c.KIND)
            for b in range(len(c.BANDS)):
                sel = ok & node[:, None] & (band == b)[None, :]
                if sel.any():
                    out.setdefault((fam, b), []).append((a[sel], v[sel]))
    return {
        key: {
            "actual": np.concatenate([p[0] for p in parts]),
            "values": np.concatenate([p[1] for p in parts]),
        }
        for key, parts in out.items()
    }


def exceed(*, actual: np.ndarray, values: np.ndarray, fam: int) -> tuple[float, float]:
    """Return the share of actuals below q05 and above q95 of the 51 values."""
    q = {
        k: c.weibull(values=values, level=lvl) for k, lvl in c.LEVELS.items() if k in ("q05", "q95")
    }
    lo = np.clip(q["q05"], 0, 100) if fam else q["q05"]
    hi = np.clip(q["q95"], 0, 100) if fam else q["q95"]
    return float((actual < lo).mean()), float((actual > hi).mean())


def fit_cell(*, data: dict[str, np.ndarray], fam: int) -> dict[str, float]:
    """Return the fitted factors of one family and band."""
    a, v = data["actual"], data["values"]
    best = {}
    # B and D: rescale about the median, tested on the two outer quantiles
    q05, q95 = c.weibull(values=v, level=0.05), c.weibull(values=v, level=0.95)
    m = v[:, c.MID]
    lo_e, hi_e = [], []
    for f in c.FACTOR_GRID:
        lo = m - f * (m - q05)
        hi = m + f * (q95 - m)
        if fam:
            lo, hi = np.clip(lo, 0, 100), np.clip(hi, 0, 100)
        lo_e.append((a < lo).mean())
        hi_e.append((a > hi).mean())
    lo_e, hi_e = np.array(lo_e), np.array(hi_e)
    both = (lo_e - c.TARGET) ** 2 + (hi_e - c.TARGET) ** 2
    best["b"] = float(c.FACTOR_GRID[both.argmin()])
    best["d_up"] = float(c.FACTOR_GRID[((hi_e - c.TARGET) ** 2).argmin()])
    best["d_down"] = float(c.FACTOR_GRID[((lo_e - c.TARGET) ** 2).argmin()])
    # C: re-draw at stretched levels
    obj = []
    for s in c.STRETCH_GRID:
        new = c.stretched(values=v, stretch=np.full(v.shape[0], s))
        if fam:
            new = np.clip(new, 0, 100)
        lo, hi = exceed(actual=a, values=new, fam=fam)
        obj.append((lo - c.TARGET) ** 2 + (hi - c.TARGET) ** 2)
    best["c"] = float(c.STRETCH_GRID[int(np.argmin(obj))])
    return best


def main() -> None:
    """Fit every fold and write the factor table."""
    files = sorted((c.WORK_436 / "runs").glob("*.npz"))
    runs = [c.load_run(path=f) for f in files]
    month = np.array([f"{r['init']:%Y-%m}" for r in runs])
    print(f"{len(runs)} runs; months {sorted(set(month))}", flush=True)
    table = []
    for test in c.TEST_MONTHS:
        train = [r for r, m in zip(runs, month, strict=True) if m < test]
        data = rows(runs=train)
        for (fam, b), cell in sorted(data.items()):
            for name, value in fit_cell(data=cell, fam=fam).items():
                table.append(
                    {
                        "test_month": test,
                        "family": c.FAMILIES[fam],
                        "band": list(c.BANDS)[b],
                        "param": name,
                        "value": value,
                        "n_train": len(cell["actual"]),
                        "n_runs": len(train),
                    }
                )
        print(f"{test}: fitted on {len(train)} runs", flush=True)
    df = pl.DataFrame(table)
    c.WORK_626.mkdir(parents=True, exist_ok=True)
    df.write_parquet(c.WORK_626 / "factors.parquet")
    pl.Config.set_tbl_rows(200)
    wide = df.pivot(on="test_month", index=["family", "band", "param"], values="value")
    print(wide)
    spread = (
        df.group_by("family", "band", "param")
        .agg(lo=pl.col("value").min(), hi=pl.col("value").max())
        .with_columns(ratio=pl.col("hi") / pl.col("lo"))
        .with_columns(unstable=pl.col("ratio") > 1 + c.INSTABILITY)
        .sort("family", "band", "param")
    )
    print(spread)
    spread.write_parquet(c.WORK_626 / "factor_stability.parquet")


if __name__ == "__main__":
    main()

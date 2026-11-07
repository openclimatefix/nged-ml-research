# ruff: noqa: E501
"""Stage 2 of experiment 626: build arms A to D for the test months and score them.

Reads experiment 436's run files and Schaake-shuffled scenarios (arm J3), and `work/factors.parquet`
(stage 1). For each run of a test month, builds the four arms' 51 values from that month's
factors, re-orders them with J3's ranks, and writes the marginal rows (exceedance counts and CRPS
sums per arm, family, lead band, and slice) and the path-score rows (experiment 436's energy and
variogram blocks) to `work/marginal_rows.parquet` and `work/path_rows.parquet`.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00626-*/env.sh
nice -n 19 systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00626-*/e00626_score.py
```
"""

import argparse

import e00434_score as s434  # ty: ignore[unresolved-import]
import e00626_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

ARMS: tuple[str, ...] = ("A", "B", "C", "D")
SLICES: tuple[str, ...] = ("all", "cold_night", "sunny")
SUNNY_MEDIAN: float = 50.0
"""Median of the 51 values (points of capacity) from which a solar half-hour is very sunny."""


def factor_arrays(*, factors: pl.DataFrame, month: str) -> dict[str, np.ndarray]:
    """Return the fitted factors as (25, 702) arrays per parameter; 1 beyond 240 hours."""
    band = c.band_of_step()
    names = list(c.BANDS)
    out = {p: np.ones((25, c.N_STEPS)) for p in ("b", "d_up", "d_down", "c")}
    sub = factors.filter(pl.col("test_month") == month)
    for row in sub.iter_rows(named=True):
        nodes = np.asarray(c.FAMILIES.index(row["family"]) == c.KIND)
        steps = np.asarray(band == names.index(row["band"]))
        out[row["param"]][np.ix_(nodes, steps)] = row["value"]
    return out


def arm_values(
    *, values: np.ndarray, paths: np.ndarray, fac: dict[str, np.ndarray]
) -> tuple[dict, dict]:
    """Return each arm's sorted values (25, 702, 51) and scenario paths (25, 702, 51)."""
    med = values[..., c.MID : c.MID + 1]
    rank = np.argsort(np.argsort(paths, axis=-1, kind="stable"), axis=-1, kind="stable")
    out_v, out_p = {"A": values, "B": None, "C": None, "D": None}, {"A": paths}
    out_v["B"] = c.clip_physical(
        x=c.rescaled(x=values, med=med, up=fac["b"], down=fac["b"]), kind=c.KIND
    )
    out_p["B"] = c.clip_physical(
        x=c.rescaled(x=paths, med=med, up=fac["b"], down=fac["b"]), kind=c.KIND
    )
    out_v["D"] = c.clip_physical(
        x=c.rescaled(x=values, med=med, up=fac["d_up"], down=fac["d_down"]), kind=c.KIND
    )
    out_p["D"] = c.clip_physical(
        x=c.rescaled(x=paths, med=med, up=fac["d_up"], down=fac["d_down"]), kind=c.KIND
    )
    out_v["C"] = c.clip_physical(x=c.stretched(values=values, stretch=fac["c"]), kind=c.KIND)
    out_p["C"] = np.take_along_axis(out_v["C"], rank, axis=-1)
    return out_v, out_p


def marginal_rows(*, run: dict, vals: dict, month: str) -> list[dict]:
    """Return the exceedance counts and CRPS sums of every arm, family, band, and slice."""
    init, actual, base = run["init"], run["actual"], run["values"]
    band = c.band_of_step()
    valid = np.array(c.valid_times(init=init))
    night = np.array([(t.month in (12, 1, 2)) and (t.hour >= 18 or t.hour < 6) for t in valid])
    have = ~np.isnan(actual) & (base[..., -1] > base[..., 0])
    sunny = base[..., c.MID] >= SUNNY_MEDIAN
    rows: list[dict] = []
    for arm, v in vals.items():
        q = {k: c.weibull(values=v, level=lvl) for k, lvl in c.LEVELS.items()}
        y = np.nan_to_num(actual)
        below05, below10 = y < q["q05"], y < q["q10"]
        above90, above95 = y > q["q90"], y > q["q95"]
        crps = s434.crps_of_sorted(values=v, actual=y)
        for fam, name in enumerate(c.FAMILIES):
            node = np.asarray(fam == c.KIND)[:, None]
            for b, bname in enumerate(c.BANDS):
                sel0 = have & node & (band == b)[None, :]
                slices = {"all": sel0}
                if fam == 0:
                    slices["cold_night"] = sel0 & night[None, :]
                if fam == 2:
                    slices["sunny"] = sel0 & sunny
                for sl, sel in slices.items():
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    rows.append({"month": month, "init": init, "arm": arm, "family": name, "band": bname, "slice": sl, "n": n,
                                 "crps_sum": float(crps[sel].sum()), "below_q05": int(below05[sel].sum()), "below_q10": int(below10[sel].sum()),
                                 "above_q90": int(above90[sel].sum()), "above_q95": int(above95[sel].sum())})  # fmt: skip
    return rows


def main() -> None:
    """Score every run of the test months."""
    # Imported here because `e00626_common` must install the stand-in for experiment 436's build module first.
    import e00436_score as s436  # ty: ignore[unresolved-import]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    factors = pl.read_parquet(c.WORK_626 / "factors.parquet")
    observed_dir = c.WORK_436 / "paths" / "J3"
    marg: list[dict] = []
    path_rows: list[dict] = []
    files = sorted((c.WORK_436 / "runs").glob("*.npz"))
    done = 0
    for file in files:
        run = c.load_run(path=file)
        month = f"{run['init']:%Y-%m}"
        if month not in c.TEST_MONTHS:
            continue
        if args.max_runs is not None and done >= args.max_runs:
            break
        done += 1
        j3 = (
            np.load(observed_dir / file.with_suffix(".npy").name)
            .astype(np.float64)
            .transpose(1, 2, 0)
        )  # (25, 702, 51)
        if not np.array_equal(
            np.sort(j3, axis=-1), run["values"].astype(np.float32).astype(np.float64)
        ):
            msg = f"{file.name}: J3 is not a re-ordering of the run's values"
            raise ValueError(msg)
        fac = factor_arrays(factors=factors, month=month)
        vals, paths = arm_values(values=run["values"], paths=j3, fac=fac)
        for arm in ARMS:
            if not np.allclose(np.sort(paths[arm], axis=-1), vals[arm], atol=1e-6):
                msg = f"{file.name}: arm {arm} paths are not a re-ordering of its values"
                raise ValueError(msg)
        marg.extend(marginal_rows(run=run, vals=vals, month=month))
        path_rows.extend(
            s436.rows_for_run(
                paths={a: paths[a].transpose(2, 0, 1) for a in ARMS},
                actual=run["actual"], kind=run["kind"], init=run["init"],
            )
        )  # fmt: skip
        print(f"{run['init']:%Y-%m-%d} scored", flush=True)
    pl.DataFrame(marg).write_parquet(c.WORK_626 / f"marginal_rows{args.out}.parquet")
    pl.DataFrame(path_rows).write_parquet(c.WORK_626 / f"path_rows{args.out}.parquet")


if __name__ == "__main__":
    main()

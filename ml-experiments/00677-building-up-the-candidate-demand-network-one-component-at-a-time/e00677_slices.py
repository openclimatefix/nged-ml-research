# ruff: noqa: D103, E501
"""Experiment 677: slice tables and step-to-step contrasts of the ladder (private CSV).

Usage: python e00677_slices.py SPAN NAME [NAME ...]   (NAME is a step file stem without the span, such as
`step8w`; stems map to `step<N>_<version>_sp<SPAN>` through `e00677_export.VERSION`)

Reads each step's per-row scores (`scores/<file>_s012/rows.parquet`, written by `e00677_score.py`),
keeps the rows that every listed step, the manual heuristic, and climatology can all be scored on, and writes
`scores/slices_sp<SPAN>.csv` (mean and 95% interval by slice, system, and metric) and
`scores/contrasts_sp<SPAN>.csv` (adjacent-step differences with a block bootstrap over 7-day blocks of
origin days). The event sets are those of `e00668_slices_primaries`; the long-and-deep set is empty in
the backtest year and is printed with its row count first.
"""

import sys

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]
import e00677_export as exp
import e00677_score as sc
import numpy as np
import polars as pl

N_BOOT = 2000
BLOCK_DAYS = 7
import os

SEEDS_TAG = "s012"  # steps 1 to 4 have seeds 0 to 2 only
SEEDS_TAG_LATE = os.environ.get("S677SEEDS", "s012")  # steps 5 to 8: "s01234" when seeds 3 and 4 are scored
OUT_SUFFIX = "" if SEEDS_TAG_LATE == "s012" else f"_{SEEDS_TAG_LATE}"


def load(*, span: int, names: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """Return the rows scored for every system, with one crps and mae column per system."""
    site = pl.read_csv(next(s668.R.glob("00521-*/site_map.csv")))
    out = None
    labels = []
    for name in names:
        base, _, at = name.rstrip("!").partition("@")  # "step8w@1" is step 8 with monthly refits (span 1, seeds 0 to 2)
        step = base.removeprefix("step")
        folder = f"{base}_{exp.VERSION[step]}_sp{at or span}"
        tag = "s012" if at or name.endswith("!") else (SEEDS_TAG_LATE if step[0] in "5678" else SEEDS_TAG)
        r = pl.read_parquet(sc.ROOT / "scores" / f"{folder}_{tag}" / "rows.parquet").filter(pl.col("ok"))
        keep = ["time_series_id", "valid_idx", "origin_idx", "lead_h"]
        cols = [pl.col("Model_crps").alias(f"{name}_crps"), pl.col("Model_mae").alias(f"{name}_mae")]
        if out is None:
            cols += [
                pl.col("Manual heuristic_crps").alias("heur_crps"), pl.col("Manual heuristic_mae").alias("heur_mae"),
                pl.col("Climatology_crps").alias("clim_crps"), pl.col("Climatology_mae").alias("clim_mae"),
            ]  # fmt: skip
        r = r.select(*keep, *cols)
        out = r if out is None else out.join(r, on=keep, how="inner")
        labels.append(name)
    rows = out.join(site.select("time_series_id", "label", "primary"), on="time_series_id")
    return rows, site


def block_bootstrap(*, day: np.ndarray, values: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return N_BOOT resampled means, resampling `BLOCK_DAYS`-day blocks of origin days with replacement."""
    block = day // BLOCK_DAYS
    ids, inv = np.unique(block, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(ids))
    cnt = np.bincount(inv, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    return sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)


PAIRS = [
    ("step1", "step2"), ("step2", "step3a"), ("step3a", "step3b"), ("step3b", "step4"),
    ("step4", "step5w"), ("step5w", "step6w"), ("step6w", "step7w"), ("step7w", "step8w"),
    ("step4", "step5wo"), ("step5wo", "step6wo"), ("step6wo", "step7wo"), ("step7wo", "step8wo"),
    ("step5w", "step5wo"), ("step6w", "step6wo"), ("step7w", "step7wo"), ("step8w", "step8wo"),
    ("step5w", "step8w"), ("step5wo", "step8wo"),
    ("step5w!", "step5w@1"), ("step8w!", "step8w@1"), ("step5wo!", "step5wo@1"), ("step8wo!", "step8wo@1"), ("step5w@1", "step8w@1"),
]  # fmt: skip


def pairs(*, names: list[str]) -> list[tuple[str, str]]:
    """Return each step-to-step contrast whose two steps were scored: the two chains of added components, then with against without injected events."""
    return [p for p in PAIRS if p[0] in names and p[1] in names]


def main() -> None:
    span = int(sys.argv[1])
    names = sys.argv[2:]
    rows, site = load(span=span, names=names)
    day = (rows["origin_idx"].to_numpy() // s668.DAY).astype(np.int64)
    sets = s668.event_sets(site=site)
    slices: dict[tuple[str, str], np.ndarray] = {("all rows", "all"): np.ones(rows.height, bool)}
    for lo, hi, label in ((0, 24, "days 0 to 1"), (24, 72, "days 1 to 3"), (72, 168, "days 3 to 7"), (168, 340, "days 7 to 14")):
        lead = rows["lead_h"].to_numpy()
        slices[("lead band", label)] = (lead >= lo) & (lead < hi)
    counts = []
    for name, ev in sets.items():
        counts.append({"set": name, "events": ev.height, "series": ev["label"].n_unique()})
        for k, m in s668.masks(rows=rows, events=ev).items():
            slices[(name, k)] = m
    systems = {**{n: n for n in names}, "heur": "Manual heuristic", "clim": "Climatology"}
    records = []
    for (sname, part), mask in slices.items():
        n = int(mask.sum())
        for tag, system in systems.items():
            for metric in ("crps", "mae"):
                v = rows[f"{tag}_{metric}"].to_numpy().astype(np.float64)
                if n == 0:
                    mean = lo = hi = float("nan")
                else:
                    draws = block_bootstrap(day=day[mask], values=v[mask])
                    mean, (lo, hi) = float(v[mask].mean()), (float(x) for x in np.quantile(draws, [0.025, 0.975]))
                records.append({"set": sname, "part": part, "system": system, "metric": metric, "mean": mean, "lo": lo, "hi": hi, "rows": n, "origin_days": int(len(np.unique(day[mask]))) if n else 0})  # fmt: skip
    out = sc.ROOT / "scores"
    pl.DataFrame(records).write_csv(out / f"slices_sp{span}{OUT_SUFFIX}.csv")
    contrasts = []
    for a, b in pairs(names=names):
        for (sname, part), mask in slices.items():
            if not mask.any():
                continue
            for metric in ("crps", "mae"):
                d = (rows[f"{b}_{metric}"] - rows[f"{a}_{metric}"]).to_numpy().astype(np.float64)[mask]
                draws = block_bootstrap(day=day[mask], values=d)
                contrasts.append({"set": sname, "part": part, "from": a, "to": b, "metric": metric, "diff": float(d.mean()), "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975)), "rows": int(mask.sum())})  # fmt: skip
    pl.DataFrame(contrasts).write_csv(out / f"contrasts_sp{span}{OUT_SUFFIX}.csv")
    print(pl.DataFrame(counts))
    print({k: int(v.sum()) for k, v in slices.items()})


if __name__ == "__main__":
    main()

# ruff: noqa: E501, PLR0912, PLR0915, PT018
"""Experiment 735: score the arms, run the two-level bootstrap, and map the readings to a verdict.

README. This scorer reads each arm's stored forecasts (`arms/k1_<ARM>_sp1/work/fits/P5/<month>_d0_s<seed>.parquet`
under experiment 735's data folder, keyed by the network's 09:00 origin), relabels them to the 12:00
contract through `e00667_export.contract_rows`, and scores them against the cleaned power in points of
capacity as experiment 677's scorer does. It relies on `e00667_export` (contract rows),
`e00668_score_lib` (the grid, the cleaned power, the quantile levels), `e00735_columns` (the local-date
convention and the coast distance), and `e00735_windows` (windows, weather split, groups). PYTHONPATH:
source `ml-experiments/00524*/env.sh`. Run under a memory cap on the experiment's cores, for example
`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 taskset -c 22,23 .venv/bin/python
e00735_score.py --stage all`. Usage: `--stage cells|contrasts|all`, `--smoke`, `--gate`.

**One fit is scored at a time.** The `cells` stage reads one refit month and one seed, computes
per-half-hour losses, and writes only their sums per Primary, local valid date, and lead day to
`scores/cells/<arm>/<month>_s<seed>.parquet`. Every window, group, weather split, and bootstrap reading
afterwards works on those sums, so the peak memory is one fit's rows (about 0.3 GB) and the bootstrap
needs about 1 GB. The sums are of 11-quantile CRPS (twice the mean pinball loss, in points of
capacity), the absolute error of the median, the signed error of the median (median minus actual),
the pinball loss and the coverage at q01, q05, q95, and q99, the median minus the forecast mean, and the
signed error at leads of 12 to 36 hours.

**The unit of the interval is the fit.** The two-level bootstrap draws refit months with replacement,
then the 3 seeds with replacement inside each drawn month, then the 7-day blocks of valid dates that
hold selected rows with replacement inside each drawn month (the same blocks for the 3 seeds, which
share their rows). A reading is the paired difference of CRPS, arm minus reference, on the selected
rows, pooled over all draws as a ratio of sums.

**Rules fixed by the plan.** The family has 6 readings at 99.17% (C1 and C2 at the six and at the
eight; C4 and C5 at the 4 coastal Primaries). A gain needs the interval wholly below zero, a point
estimate of at most -0.02, a negative seed-mean difference in at least 6 of the 8 held-in months, and
a negative difference in at least 18 of the 24 held-in fits. Placebo readings use 95% intervals and
fail when the interval lies wholly below zero. `verdict` applies rules 1 to 6 of the plan.
"""

import argparse
import json
import math
import os
import sys
import zlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import e00735_columns as cols  # ty: ignore[unresolved-import]
import e00735_windows as win  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1"
NAME_677: Final = "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
ROOT: Final[Path] = Path(os.environ.get("R735", ML_EXPERIMENTS_DATA_DIR / NAME))
ROOT_677: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME_677
SMOKE_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[2] / ".claude" / "scratch" / "e735_smoke" / "00735-smoke"
)
assert "00735" in str(ROOT), ROOT
STORED_FILE: Final = "step5w_with_injected_sp1"
"""Experiment 677's stored arm 9a (`k1_5w_sp1`) in its backtest file names."""
ARMS: Final = ("R0", "S", "SP", "SKW", "SKWP", "SE", "SEP")
MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
)  # fmt: skip
HELD_IN_MONTHS: Final = 8
SEEDS: Final = (0, 1, 2)
GATE_MONTH: Final = "2026-10"
GATE_TOLERANCE: Final = 5e-4
"""The replay tolerance of experiment 704, as a fraction of capacity."""
N_BOOT: Final = 20_000
FAMILY_LEVEL: Final = 1.0 - 0.05 / 6
"""99.17%: 95% with a Bonferroni correction over the 6 readings of the family."""
PLACEBO_LEVEL: Final = 0.95
MIN_GAIN: Final = 0.02
MIN_MONTHS_NEGATIVE: Final = 6
MIN_FITS_NEGATIVE: Final = 18
WHOLE_YEAR_BOUND: Final = 0.02
NOT_SUPPORTED_BAND: Final = 0.05
LEAD_FIRST_DAY, LEAD_LAST_DAY = 1, 10
"""Days 1 to 10 are leads in (0, 240] hours: lead-day buckets 1 to 10."""
HOLDOUT_CODE: Final = cols.code_of(day=date(2026, 7, 1))
LEAD_DAY_MAX: Final = 14
SUM_COLUMNS: Final = (
    "crps", "mae", "signed", "pin01", "pin05", "pin95", "pin99", "cov01", "cov05", "cov95",
    "cov99", "gap", "s1236",
)  # fmt: skip
KEY_COLUMNS: Final = ("m", "seed", "primary", "vday", "lead_day")
BLOCK_DAYS: Final = 7
CHUNK: Final = 2_000


# ----------------------------------------------------------------------------- context and cells


@dataclass
class Context:
    """The cleaned power, capacities, and Primary map that every fit's scoring reads."""

    ids: list[int]
    cap: np.ndarray
    primary_of_row: np.ndarray
    row_of_tsid: np.ndarray
    grid: np.ndarray

    def padded_grid(self, *, last: int) -> np.ndarray:
        """Return the grid, padded with NaN to at least `last + 1` half-hours."""
        if last < self.grid.shape[1]:
            return self.grid
        return np.pad(
            self.grid, ((0, 0), (0, last + 1 - self.grid.shape[1])), constant_values=np.nan
        )


def make_context() -> Context:
    """Load the cleaned power and the capacities once (about 1 GB)."""
    import e00668_score_lib as lib  # ty: ignore[unresolved-import]

    table = win.sites()
    ids = sorted(table["time_series_id"].to_list())
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    cap = (
        capacity.filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")["capacity"]
        .to_numpy()
        .astype(np.float64)
    )
    primary = dict(zip(table["time_series_id"].to_list(), table["primary"].to_list(), strict=True))
    row_of = np.full(256, -1, dtype=np.int64)
    for i, t in enumerate(ids):
        row_of[t] = i
    return Context(
        ids=ids,
        cap=cap,
        primary_of_row=np.asarray([primary[t] for t in ids], dtype=np.int64),
        row_of_tsid=row_of,
        grid=lib.cleaned_grid(ids=ids),
    )


def mean_weights() -> np.ndarray:
    """Return the midpoint-rule weights that turn the 11 quantiles into the forecast mean."""
    import e00668_score_lib as lib  # ty: ignore[unresolved-import]

    taus = np.asarray(lib.TAUS)
    edges = np.concatenate([[0.0], (taus[1:] + taus[:-1]) / 2, [1.0]])
    return np.diff(edges)


def cells_from_contract(
    *, rows: pl.DataFrame, ctx: Context, month_index: int
) -> tuple[pl.DataFrame, dict]:
    """Return the sums per Primary, valid local date, and lead day of one fit's contract rows, and its row counts.

    Args:
        rows: Contract rows (12:00 origins, 14 days each, quantiles in MW).
        ctx: The scoring context.
        month_index: Index of the refit month in `MONTHS`.

    Returns:
        The cell table and a dictionary of row counts (all, scored, without truth).
    """
    import e00668_score_lib as lib  # ty: ignore[unresolved-import]

    taus = np.asarray(lib.TAUS)
    row = ctx.row_of_tsid[rows["time_series_id"].to_numpy().astype(np.int64)]
    assert (row >= 0).all()
    vidx = lib.grid_index(times=rows["valid_time"])
    oidx = lib.grid_index(times=rows["power_fcst_init_time"])
    grid = ctx.padded_grid(last=int(vidx.max()))
    lead_h = (vidx - oidx) / 2.0
    scale = 100.0 / ctx.cap[row]
    q = rows.select(lib.QCOLS).to_numpy().astype(np.float64) * scale[:, None]
    y = grid[row, vidx] * scale
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    counts = {
        "rows": len(y),
        "scored": int(ok.sum()),
        "no_truth": int((~np.isfinite(y)).sum()),
    }
    sel = np.flatnonzero(ok)
    q, y, lead_h, row = q[sel], y[sel], lead_h[sel], row[sel]
    stamp = rows["valid_time"].dt.epoch("us").to_numpy()[sel]
    d = y[:, None] - q
    pin = np.maximum(taus * d, (taus - 1.0) * d)
    lead_day = np.clip(np.ceil(lead_h / 24.0), 0, LEAD_DAY_MAX).astype(np.int8)
    in_window = (lead_h >= 12.0) & (lead_h <= 36.0)
    signed = q[:, 5] - y
    frame = pl.DataFrame(
        {
            "primary": ctx.primary_of_row[row].astype(np.int8),
            "vday": cols.local_dates(stamp_us=stamp).code.astype(np.int32),
            "lead_day": lead_day,
            "crps": 2.0 * pin.mean(axis=1),
            "mae": np.abs(y - q[:, 5]),
            "signed": signed,
            "pin01": pin[:, 0],
            "pin05": pin[:, 1],
            "pin95": pin[:, 9],
            "pin99": pin[:, 10],
            "cov01": (y <= q[:, 0]).astype(np.float64),
            "cov05": (y <= q[:, 1]).astype(np.float64),
            "cov95": (y <= q[:, 9]).astype(np.float64),
            "cov99": (y <= q[:, 10]).astype(np.float64),
            "gap": q[:, 5] - q @ mean_weights(),
            "s1236": np.where(in_window, signed, 0.0),
            "n1236": in_window.astype(np.int32),
        }
    )
    cells = (
        frame.group_by("primary", "vday", "lead_day")
        .agg(
            pl.len().cast(pl.Int32).alias("n"),
            pl.col("n1236").sum(),
            *[pl.col(c).sum() for c in SUM_COLUMNS],
        )
        .sort("primary", "vday", "lead_day")
        .with_columns(m=pl.lit(month_index, pl.Int8))
    )
    return cells, counts


def fit_path(*, root: Path, arm: str, month: str, seed: int) -> Path:
    """Return the fits file of one arm, refit month, and seed."""
    return root / "arms" / f"k1_{arm}_sp1" / "work" / "fits" / "P5" / f"{month}_d0_s{seed}.parquet"


def cell_path(*, arm: str, month: str, seed: int) -> Path:
    """Return the cells file of one fit."""
    return ROOT / "scores" / "cells" / arm / f"{month}_s{seed}.parquet"


def contract_from_fit(*, path: Path) -> pl.DataFrame:
    """Read one fits file and return its contract rows (12:00 origins, MW)."""
    import e00667_export as x667  # ty: ignore[unresolved-import]

    raw = pl.read_parquet(path, columns=[*x667.KEYS, *x667.QCOLS])
    assert raw.height == raw.unique(list(x667.KEYS)).height, ("duplicate keys", path)
    return x667.contract_rows(mean=raw.sort(list(x667.KEYS)))


def contract_from_stored(*, month: str, seed: int) -> pl.DataFrame:
    """Return experiment 677's stored arm 9a rows of one refit month and seed (already contract rows)."""
    import e00667_export as x667  # ty: ignore[unresolved-import]

    first = date(int(month[:4]), int(month[5:]), 1)
    last = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
    frame = (
        pl.scan_parquet(ROOT_677 / "backtests" / f"{STORED_FILE}.parquet")
        .filter(pl.col("seed") == seed)
        .filter(
            (pl.col("power_fcst_init_time").dt.date() >= first)
            & (pl.col("power_fcst_init_time").dt.date() < last)
        )
        .select(*x667.KEYS, *x667.QCOLS, "nwp_init_time")
        .collect()
    )
    assert frame.height, (month, seed)
    return frame.sort("time_series_id", "power_fcst_init_time", "valid_time")


def stage_cells(*, arms: list[str], stored: bool) -> None:
    """Write the cells of every fit that exists and has none yet, one fit at a time."""
    ctx = make_context()
    count_rows = []
    for arm in arms:
        for mi, month in enumerate(MONTHS):
            for seed in SEEDS:
                out = cell_path(arm=arm, month=month, seed=seed)
                if out.exists():
                    continue
                if arm == "stored":
                    if not stored:
                        continue
                    rows = contract_from_stored(month=month, seed=seed)
                else:
                    path = fit_path(root=ROOT, arm=arm, month=month, seed=seed)
                    if not path.exists():
                        continue
                    rows = contract_from_fit(path=path)
                cells, counts = cells_from_contract(rows=rows, ctx=ctx, month_index=mi)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                cells.write_parquet(tmp)
                tmp.rename(out)
                count_rows.append({"arm": arm, "month": month, "seed": seed, **counts})
                print("cells", arm, month, seed, counts, flush=True)
    if count_rows:
        path = ROOT / "scores" / "row_counts.csv"
        new = pl.DataFrame(count_rows)
        if path.exists():
            new = pl.concat([pl.read_csv(path), new]).unique(["arm", "month", "seed"], keep="last")
        new.sort("arm", "month", "seed").write_csv(path)


def load_arm(*, arm: str, months: tuple[str, ...] = MONTHS) -> pl.DataFrame | None:
    """Return an arm's cells for the months, or None if any fit of those months lacks a file."""
    parts = []
    for mi, month in enumerate(MONTHS):
        if month not in months:
            continue
        for seed in SEEDS:
            path = cell_path(arm=arm, month=month, seed=seed)
            if not path.exists():
                return None
            parts.append(
                pl.read_parquet(path).with_columns(
                    seed=pl.lit(seed, pl.Int8), m=pl.lit(mi, pl.Int8)
                )
            )
    return pl.concat(parts).sort(*KEY_COLUMNS)


# ----------------------------------------------------------------------------- rows and selections


@dataclass
class Pair:
    """Two arms' cells joined on their keys, with the differences and the row descriptors."""

    frame: pl.DataFrame
    name: str
    reference: str
    masks: dict[str, np.ndarray]
    codes: np.ndarray
    kinds: np.ndarray
    period: np.ndarray
    lead_ok: np.ndarray


def period_of(*, m: np.ndarray, vday: np.ndarray) -> np.ndarray:
    """Return the block of each cell: `held-in`, `july to september`, or `october`.

    The block follows the refit month, except that cells of the 2026-06 refit with a valid date on
    or after 2026-07-01 belong to `july to september`.
    """
    out = np.full(len(m), "held-in", dtype=object)
    out[(m == HELD_IN_MONTHS - 1) & (vday >= HOLDOUT_CODE)] = "july to september"
    out[(m >= HELD_IN_MONTHS) & (m < len(MONTHS) - 1)] = "july to september"
    out[m == len(MONTHS) - 1] = "october"
    return out.astype(str)


def make_pair(*, a: pl.DataFrame, b: pl.DataFrame, name: str, reference: str, weather) -> Pair:  # noqa: ANN001
    """Join two arms' cells (every key must match, with equal row counts) and describe the rows."""
    keys = list(KEY_COLUMNS)
    joined = a.join(b, on=keys, how="inner", suffix="_ref", validate="1:1").sort(*keys)
    assert joined.height == a.height == b.height, ("cell keys differ", name, reference)
    assert (joined["n"] == joined["n_ref"]).all(), ("row counts differ", name, reference)
    frame = joined.select(
        *keys,
        "n",
        *[(pl.col(c) - pl.col(f"{c}_ref")).alias(f"d_{c}") for c in SUM_COLUMNS],
        *[pl.col(c).alias(f"a_{c}") for c in SUM_COLUMNS],
        *[pl.col(f"{c}_ref").alias(f"b_{c}") for c in SUM_COLUMNS],
        pl.col("n1236"),
    )
    codes = frame["vday"].to_numpy().astype(np.int64)
    kinds = (
        win.weather_split(
            weather=weather[0],
            medians=weather[1],
            primary=frame["primary"].to_numpy().astype(np.int64),
            codes=codes,
        )
        if weather is not None
        else np.full(frame.height, -1, dtype=np.int8)
    )
    lead = frame["lead_day"].to_numpy()
    return Pair(
        frame=frame,
        name=name,
        reference=reference,
        masks=win.window_masks(codes=codes),
        codes=codes,
        kinds=kinds,
        period=period_of(m=frame["m"].to_numpy().astype(np.int64), vday=codes),
        lead_ok=(lead >= LEAD_FIRST_DAY) & (lead <= LEAD_LAST_DAY),
    )


# ----------------------------------------------------------------------------- the bootstrap


def gather(
    *, pair: Pair, mask: np.ndarray, column: str = "d_crps"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (D, N, n_blocks) for a selection: sums of the column and of the row counts per (month, seed, block).

    Months with no selected row keep zeros. Blocks are the 7-day groups of valid dates that hold a
    selected row, numbered inside each month.
    """
    f = pair.frame
    m = f["m"].to_numpy().astype(np.int64)[mask]
    s = f["seed"].to_numpy().astype(np.int64)[mask]
    vday = f["vday"].to_numpy().astype(np.int64)[mask]
    n_months = len(MONTHS)
    block = np.zeros(len(m), dtype=np.int64)
    n_blocks = np.zeros(n_months, dtype=np.int64)
    for mm in np.unique(m):
        sel = m == mm
        _, inv = np.unique(vday[sel] // BLOCK_DAYS, return_inverse=True)
        block[sel] = inv
        n_blocks[mm] = inv.max() + 1
    width = int(max(n_blocks.max(), 1))
    shape = (n_months, len(SEEDS), width)
    index = (m * len(SEEDS) + s) * width + block
    d = np.bincount(
        index, weights=f[column].to_numpy()[mask], minlength=int(np.prod(shape))
    ).reshape(shape)
    n = np.bincount(
        index, weights=f["n"].to_numpy().astype(np.float64)[mask], minlength=int(np.prod(shape))
    ).reshape(shape)
    return d, n, n_blocks


def bootstrap_sums(
    *, parts: list[tuple[np.ndarray, np.ndarray]], n_blocks: np.ndarray, months: list[int], rng: np.random.Generator,
    n_draws: int = N_BOOT,
) -> list[tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """Return, for each (D, N) pair, the bootstrap draws of (sum of D, sum of N) under one set of picks.

    Args:
        parts: Arrays of shape (months, seeds, blocks) that share their block numbering.
        n_blocks: The number of blocks per month (shared by every part).
        months: The month indices that are in play (8 held-in or all 12).
        rng: The generator; every part sees the same picks.
        n_draws: The number of draws.

    Returns:
        One pair of (n_draws,) arrays per part.
    """
    width = parts[0][0].shape[2]
    months_arr = np.asarray(months)
    n_m, n_s = len(months_arr), len(SEEDS)
    out = [(np.zeros(n_draws), np.zeros(n_draws)) for _ in parts]
    for start in range(0, n_draws, CHUNK):
        k = min(CHUNK, n_draws - start)
        mk = months_arr[rng.integers(0, n_m, size=(k, n_m))]
        sk = rng.integers(0, n_s, size=(k, n_m, n_s))
        nb = n_blocks[mk]
        bk = np.minimum(
            (rng.random((k, n_m, width)) * np.maximum(nb, 1)[:, :, None]).astype(np.int64),
            width - 1,
        )
        use = np.arange(width)[None, None, :] < nb[:, :, None]
        for p, (d, n) in enumerate(parts):
            idx = (mk[:, :, None, None], sk[:, :, :, None], bk[:, :, None, :])
            mask = use[:, :, None, :]
            out[p][0][start : start + k] = (d[idx] * mask).sum(axis=(1, 2, 3))
            out[p][1][start : start + k] = (n[idx] * mask).sum(axis=(1, 2, 3))
    return out


def rng_for(*, name: str) -> np.random.Generator:
    """Return a generator seeded by 735 and the reading's name, so no reading depends on another."""
    return np.random.default_rng([735, zlib.crc32(name.encode())])


def reading(
    *, pair: Pair, mask: np.ndarray, name: str, level: float, months: list[int], column: str = "d_crps",
    one_sided: bool = False,
) -> dict:  # fmt: skip
    """Return one reading: the pooled paired difference, its interval, and the sign counts.

    Args:
        pair: The two arms' joined cells.
        mask: The selected cells.
        name: The reading's name (it seeds the bootstrap).
        level: The two-sided interval level (or the one-sided level when `one_sided`).
        months: The month indices that the reading pools.
        column: The difference column.
        one_sided: Read the upper bound only (the whole-year no-harm check).

    Returns:
        The estimate, bounds, and counts, with NaN bounds where no rows were selected.
    """
    sel = mask & np.isin(pair.frame["m"].to_numpy(), months)
    d, n, n_blocks = gather(pair=pair, mask=sel, column=column)
    keep = np.zeros(len(MONTHS), dtype=bool)
    keep[months] = True
    d, n = d * keep[:, None, None], n * keep[:, None, None]
    total = n.sum()
    out = {"name": name, "level": level, "rows": int(total), "n_months": 0, "months_negative": 0,
           "n_fits": 0, "fits_negative": 0, "estimate": math.nan, "lower": math.nan, "upper": math.nan}  # fmt: skip
    if total == 0:
        return out
    out["estimate"] = float(d.sum() / total)
    by_fit = d.sum(axis=2) / np.where(n.sum(axis=2) > 0, n.sum(axis=2), np.nan)
    out["n_fits"] = int(np.isfinite(by_fit).sum())
    out["fits_negative"] = int((by_fit < 0).sum())
    by_month = d.sum(axis=(1, 2)) / np.where(n.sum(axis=(1, 2)) > 0, n.sum(axis=(1, 2)), np.nan)
    out["n_months"] = int(np.isfinite(by_month).sum())
    out["months_negative"] = int((by_month < 0).sum())
    ((sd, sn),) = bootstrap_sums(
        parts=[(d, n)], n_blocks=n_blocks, months=months, rng=rng_for(name=name)
    )
    draws = sd[sn > 0] / sn[sn > 0]
    if one_sided:
        out["upper"] = float(np.quantile(draws, level))
    else:
        tail = (1.0 - level) / 2.0
        out["lower"], out["upper"] = (float(v) for v in np.quantile(draws, [tail, 1.0 - tail]))
    return out


def is_gain(*, r: dict) -> bool:
    """Return True when a family reading meets the plan's four gain conditions."""
    return bool(
        math.isfinite(r["upper"])
        and r["upper"] < 0
        and r["estimate"] <= -MIN_GAIN
        and r["months_negative"] >= MIN_MONTHS_NEGATIVE
        and r["fits_negative"] >= MIN_FITS_NEGATIVE
    )


def is_harm(*, r: dict) -> bool:
    """Return True when a reading's interval lies wholly above zero."""
    return bool(math.isfinite(r["lower"]) and r["lower"] > 0)


def placebo_failed(*, r: dict) -> bool:
    """Return True when a placebo reading's interval lies wholly below zero (it gains)."""
    return bool(math.isfinite(r["upper"]) and r["upper"] < 0)


def is_small_gain(*, r: dict) -> bool:
    """Return True for an interval wholly below zero with a point estimate between 0 and -0.02."""
    return bool(math.isfinite(r["upper"]) and r["upper"] < 0 and -MIN_GAIN < r["estimate"] < 0)


def inside_band(*, r: dict) -> bool:
    """Return True when the interval lies inside -0.05 to +0.05."""
    return bool(
        math.isfinite(r["lower"])
        and r["lower"] >= -NOT_SUPPORTED_BAND
        and r["upper"] <= NOT_SUPPORTED_BAND
    )


def verdict(*, r: dict[str, dict]) -> dict:
    """Map the readings to the plan's verdict, first rule that applies.

    Args:
        r: The named readings: `C1 six`, `C1 eight`, `C2 six`, `C2 eight`, `C4`, `C5`; the placebos
            `SP-R0 six`, `SP-R0 eight`, `SKWP-S`, `SEP-S`; the whole-year upper bounds `year S six`,
            `year S eight`, `year SKW six`, `year SKW eight`, `year SE six`, `year SE eight`; the
            inland placebo readings `inland C4` and `inland C5`. A reading of an arm that was
            dropped is absent, and a claim that needs it does not hold.

    Returns:
        The verdict, the school claim per group, the coast claims, and the rule that decided.
    """

    def has(name: str) -> bool:
        return name in r

    def school(group: str) -> bool:
        return (
            has(f"C1 {group}") and has(f"C2 {group}") and has(f"SP-R0 {group}") and has(f"year S {group}")
            and is_gain(r=r[f"C1 {group}"]) and is_gain(r=r[f"C2 {group}"])
            and not placebo_failed(r=r[f"SP-R0 {group}"])
            and r[f"year S {group}"]["upper"] <= WHOLE_YEAR_BOUND
        )  # fmt: skip

    def coast(reading_name: str, placebo: str, arm: str) -> bool:
        needed = [
            reading_name,
            placebo,
            f"year {arm} six",
            f"year {arm} eight",
            f"inland {reading_name}",
        ]
        return (
            all(has(n) for n in needed)
            and is_gain(r=r[reading_name])
            and not placebo_failed(r=r[placebo])
            and r[f"year {arm} six"]["upper"] <= WHOLE_YEAR_BOUND
            and r[f"year {arm} eight"]["upper"] <= WHOLE_YEAR_BOUND
            and not placebo_failed(r=r[f"inland {reading_name}"])
        )

    school_six, school_eight = school("six"), school("eight")
    coast_c4, coast_c5 = coast("C4", "SKWP-S", "SKW"), coast("C5", "SEP-S", "SE")
    coast_claim = coast_c4 or coast_c5
    family = [r[n] for n in ("C1 six", "C1 eight", "C4", "C5") if n in r]
    if any(is_harm(r=r[n]) for n in ("C1 six", "C1 eight") if n in r):
        result, rule = "refuted", 1
    elif school_six and school_eight and coast_claim:
        result, rule = "supported", 2
    elif school_six or school_eight or coast_claim:
        result, rule = "partly supported", 3
    elif any(is_small_gain(r=x) for x in family):
        result, rule = "small gain", 4
    elif all(n in r for n in ("C1 six", "C1 eight", "C4", "C5")) and all(
        inside_band(r=x) for x in family
    ):
        result, rule = "not supported", 5
    else:
        result, rule = "inconclusive", 6
    return {
        "verdict": result,
        "rule": rule,
        "school_claim_six": school_six,
        "school_claim_eight": school_eight,
        "coast_claim_C4": coast_c4,
        "coast_claim_C5": coast_c5,
    }


# ----------------------------------------------------------------------------- stages


def group_mask(*, pair: Pair, indices: list[int]) -> np.ndarray:
    """Return the cells of the Primaries in a group."""
    return np.isin(pair.frame["primary"].to_numpy(), indices)


def pair_of(*, arms: dict[str, pl.DataFrame], a: str, b: str, weather) -> Pair | None:  # noqa: ANN001
    """Return the joined pair of two arms, or None if either is missing."""
    if a not in arms or b not in arms:
        return None
    return make_pair(a=arms[a], b=arms[b], name=a, reference=b, weather=weather)


def stage_contrasts(*, r0_from_stored: bool) -> dict:
    """Compute every reading, the verdict, and the tables; write them under `scores/`."""
    out_dir = ROOT / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = win.sites()
    labels = table["d_label"].to_list()
    groups = win.group_indices(labels=labels)
    six, eight, coastal, inland = (
        groups[k] for k in ("six signed-MW", "eight MVA-only", "4 coastal", "3 inland placebo")
    )
    weather_cache = ROOT / "weather_daily.npz"
    daily = win.daily_weather(expected_dir=win.EXPECTED_DIR, cache=weather_cache)
    weather = (daily, win.monthly_medians(weather=daily))
    arms: dict[str, pl.DataFrame] = {}
    for arm in (*ARMS, "stored"):
        loaded = load_arm(arm=arm)
        if loaded is not None:
            arms[arm] = loaded
    if "R0" not in arms and r0_from_stored and "stored" in arms:
        arms["R0"] = arms["stored"]
    print("arms scored in all 12 months:", sorted(arms), flush=True)
    held = list(range(HELD_IN_MONTHS))
    every = list(range(len(MONTHS)))
    readings: dict[str, dict] = {}
    rows: list[dict] = []

    def add(r: dict, **tags: object) -> dict:
        readings[r["name"]] = r
        rows.append({**tags, **r})
        return r

    def window_rows(p: Pair, period: str = "held-in") -> np.ndarray:
        return p.masks[win.SCHOOL_WINDOW] & p.lead_ok & (p.period == period)

    specs = [
        ("C1", "S", "R0", "six", six, 0.0), ("C1", "S", "R0", "eight", eight, 0.0),
        ("C2", "S", "SP", "six", six, 0.0), ("C2", "S", "SP", "eight", eight, 0.0),
    ]  # fmt: skip
    pairs: dict[tuple[str, str], Pair | None] = {}

    def get(a: str, b: str) -> Pair | None:
        if (a, b) not in pairs:
            pairs[(a, b)] = pair_of(arms=arms, a=a, b=b, weather=weather)
        return pairs[(a, b)]

    for tag, a, b, gname, idx, _ in specs:
        p = get(a, b)
        if p is not None:
            add(
                reading(
                    pair=p,
                    mask=window_rows(p) & group_mask(pair=p, indices=idx),
                    name=f"{tag} {gname}",
                    level=FAMILY_LEVEL,
                    months=held,
                ),
                tag=tag,
                group=gname,
                level="family",
            )
    for tag, a, b in (("C4", "SKW", "SKWP"), ("C5", "SE", "SEP")):
        p = get(a, b)
        if p is None:
            continue
        add(
            reading(
                pair=p,
                mask=window_rows(p) & group_mask(pair=p, indices=coastal),
                name=tag,
                level=FAMILY_LEVEL,
                months=held,
            ),
            tag=tag,
            group="4 coastal",
            level="family",
        )
        add(
            reading(
                pair=p,
                mask=window_rows(p) & group_mask(pair=p, indices=inland),
                name=f"inland {tag}",
                level=PLACEBO_LEVEL,
                months=held,
            ),
            tag=f"inland {tag}",
            group="3 inland placebo",
            level="placebo",
        )
    for name, a, b, idx_names in (
        ("SP-R0", "SP", "R0", (("six", six), ("eight", eight))),
        ("SKWP-S", "SKWP", "S", (("coastal", coastal),)),
        ("SEP-S", "SEP", "S", (("coastal", coastal),)),
    ):
        p = get(a, b)
        if p is None:
            continue
        for gname, idx in idx_names:
            key = f"{name} {gname}" if name == "SP-R0" else name
            add(
                reading(
                    pair=p,
                    mask=window_rows(p) & group_mask(pair=p, indices=idx),
                    name=key,
                    level=PLACEBO_LEVEL,
                    months=held,
                ),
                tag=name,
                group=gname,
                level="placebo",
            )
    for arm in ARMS[1:]:
        p = get(arm, "R0")
        if p is None:
            continue
        for gname, idx in (
            ("six", six),
            ("eight", eight),
            ("all 16", list(range(16))),
            ("4 coastal", coastal),
            ("3 inland placebo", inland),
        ):
            mask = p.lead_ok & group_mask(pair=p, indices=idx)
            add(
                reading(
                    pair=p,
                    mask=mask,
                    name=f"year {arm} {gname}",
                    level=PLACEBO_LEVEL,
                    months=every,
                    one_sided=True,
                ),
                tag=f"year {arm} - R0",
                group=gname,
                level="one-sided 95",
            )
            add(
                reading(
                    pair=p,
                    mask=mask,
                    name=f"year held-in {arm} {gname}",
                    level=PLACEBO_LEVEL,
                    months=held,
                    one_sided=True,
                ),
                tag=f"year held-in {arm} - R0",
                group=gname,
                level="one-sided 95",
            )
    summary = {"readings": readings, "verdict": verdict(r=readings)}
    descriptive(
        arms=arms,
        pairs=get,
        groups=groups,
        weather=weather,
        held=held,
        add=add,
        window_rows=window_rows,
    )
    (out_dir / "readings.json").write_text(json.dumps(summary, indent=1, default=float))
    pl.DataFrame(rows, infer_schema_length=None).write_csv(out_dir / "readings.csv")
    by_fit_table(get=get, groups=groups, out=out_dir / "by_fit.csv")
    metrics_tables(arms=arms, weather=weather, groups=groups, out=out_dir)
    print(json.dumps(summary["verdict"], indent=1), flush=True)
    return summary


def descriptive(*, arms, pairs, groups, weather, held, add, window_rows) -> None:  # noqa: ANN001
    """Add the descriptive readings: SKW minus S, P4, the bank-holiday slice, finer windows, October, July to September."""
    six, eight, coastal = groups["six signed-MW"], groups["eight MVA-only"], groups["4 coastal"]
    sets = (("six", six), ("eight", eight), ("4 coastal", coastal), ("all 16", list(range(16))))
    for a, b in (
        ("SKW", "S"),
        ("SE", "S"),
        ("S", "R0"),
        ("SKW", "SKWP"),
        ("SE", "SEP"),
        ("S", "SP"),
        ("R0", "stored"),
    ):
        p = pairs(a, b)
        if p is None:
            continue
        for gname, idx in sets:
            g = group_mask(pair=p, indices=idx)
            for wname, wmask in (
                ("school window", p.masks[win.SCHOOL_WINDOW]),
                ("bank window", p.masks["bank window"]),
                ("ordinary days", p.masks[win.ORDINARY]),
                *[(k, v) for k, v in p.masks.items() if k.startswith(("school: ", "bank: "))],
            ):
                add(
                    reading(
                        pair=p,
                        mask=wmask & g & p.lead_ok & (p.period == "held-in"),
                        name=f"{a}-{b} {gname} {wname} held-in",
                        level=PLACEBO_LEVEL,
                        months=held,
                    ),
                    tag=f"{a} - {b}",
                    group=gname,
                    window=wname,
                    period="held-in",
                    level="95",
                )
            for period, months in (("october", [11]), ("july to september", [8, 9, 10, 7])):
                add(
                    reading(
                        pair=p,
                        mask=p.masks[win.SCHOOL_WINDOW] & g & p.lead_ok & (p.period == period),
                        name=f"{a}-{b} {gname} {period}",
                        level=PLACEBO_LEVEL,
                        months=months,
                    ),
                    tag=f"{a} - {b}",
                    group=gname,
                    window="school window",
                    period=period,
                    level="95",
                )
            if a in {"SKW", "SE"} and b in {"SKWP", "SEP"}:
                for k, kind in enumerate(win.WEATHER_KINDS):
                    add(
                        reading(
                            pair=p,
                            mask=p.masks[win.SCHOOL_WINDOW]
                            & g
                            & p.lead_ok
                            & (p.period == "held-in")
                            & (p.kinds == k),
                            name=f"{a}-{b} {gname} {kind}",
                            level=PLACEBO_LEVEL,
                            months=held,
                        ),
                        tag=f"{a} - {b}",
                        group=gname,
                        window=kind,
                        period="held-in",
                        level="95",
                    )
    p = pairs("SKW", "SKWP")
    if p is not None:
        add(
            p4_reading(pair=p, group=coastal, held=held),
            tag="P4",
            group="4 coastal",
            window="warm and sunny minus cold or wet",
            period="held-in",
            level="95",
        )


def p4_reading(*, pair: Pair, group: list[int], held: list[int]) -> dict:
    """Return P4: the SKW-minus-SKWP gain on warm and sunny school-window days minus that on cold or wet days.

    Both parts share one set of bootstrap picks. A gain is negative, so P4 expects a negative number.
    """
    base = (
        pair.masks[win.SCHOOL_WINDOW]
        & group_mask(pair=pair, indices=group)
        & pair.lead_ok
        & (pair.period == "held-in")
    )
    masks = [base & (pair.kinds == 0), base & (pair.kinds == 1)]
    gathered = [gather(pair=pair, mask=m) for m in masks]
    width = max(g[0].shape[2] for g in gathered)
    pad = [
        (
            np.pad(g[0], ((0, 0), (0, 0), (0, width - g[0].shape[2]))),
            np.pad(g[1], ((0, 0), (0, 0), (0, width - g[1].shape[2]))),
        )
        for g in gathered
    ]
    n_blocks = np.maximum(gathered[0][2], gathered[1][2])
    out = {"name": "P4", "level": PLACEBO_LEVEL, "rows": int(sum(g[1].sum() for g in gathered)), "estimate": math.nan, "lower": math.nan, "upper": math.nan, "n_months": 0, "months_negative": 0, "n_fits": 0, "fits_negative": 0}  # fmt: skip
    if min(g[1].sum() for g in gathered) == 0:
        return out
    out["estimate"] = float(pad[0][0].sum() / pad[0][1].sum() - pad[1][0].sum() / pad[1][1].sum())
    (d0, n0), (d1, n1) = bootstrap_sums(
        parts=pad, n_blocks=n_blocks, months=held, rng=rng_for(name="P4")
    )
    ok = (n0 > 0) & (n1 > 0)
    draws = d0[ok] / n0[ok] - d1[ok] / n1[ok]
    out["lower"], out["upper"] = (float(v) for v in np.quantile(draws, [0.025, 0.975]))
    return out


def by_fit_table(*, get, groups, out: Path) -> None:  # noqa: ANN001
    """Write each family contrast by refit month and seed (the 24 held-in fits and the others)."""
    rows = []
    for tag, a, b, gname in (
        ("C1", "S", "R0", "six signed-MW"), ("C1", "S", "R0", "eight MVA-only"),
        ("C2", "S", "SP", "six signed-MW"), ("C2", "S", "SP", "eight MVA-only"),
        ("C4", "SKW", "SKWP", "4 coastal"), ("C5", "SE", "SEP", "4 coastal"),
    ):  # fmt: skip
        p = get(a, b)
        if p is None:
            continue
        mask = p.masks[win.SCHOOL_WINDOW] & p.lead_ok & group_mask(pair=p, indices=groups[gname])
        f = p.frame.filter(pl.Series(mask)).with_columns(period=pl.Series(p.period[mask]))
        agg = (
            f.group_by("m", "seed", "period")
            .agg(pl.col("d_crps").sum(), pl.col("n").sum())
            .sort("m", "seed")
        )
        rows.append(
            agg.with_columns(
                diff=pl.col("d_crps") / pl.col("n"),
                contrast=pl.lit(tag),
                group=pl.lit(gname),
                month=pl.col("m").map_elements(lambda i: MONTHS[i], return_dtype=pl.Utf8),
            ).select("contrast", "group", "month", "seed", "period", "n", "diff")
        )
    if rows:
        pl.concat(rows).write_csv(out)


def metrics_tables(*, arms: dict[str, pl.DataFrame], weather, groups: dict, out: Path) -> None:  # noqa: ANN001
    """Write per-arm metrics by window, group, period, and lead band, and the median-minus-mean gap per Primary."""
    parts, gaps = [], []
    bands = {
        "days 1 to 10": (1, 10),
        "day 1": (1, 1),
        "day 2": (2, 2),
        "days 3 to 4": (3, 4),
        "days 5 to 7": (5, 7),
        "days 8 to 10": (8, 10),
        "days 11 to 14": (11, 14),
    }
    for arm, frame in arms.items():
        codes = frame["vday"].to_numpy().astype(np.int64)
        masks = win.window_masks(codes=codes)
        prim = frame["primary"].to_numpy().astype(np.int64)
        kinds = win.weather_split(weather=weather[0], medians=weather[1], primary=prim, codes=codes)
        period = period_of(m=frame["m"].to_numpy().astype(np.int64), vday=codes)
        lead = frame["lead_day"].to_numpy()
        gaps.append(
            frame.group_by("primary")
            .agg(pl.col("gap").sum(), pl.col("n").sum())
            .with_columns(arm=pl.lit(arm), median_minus_mean=pl.col("gap") / pl.col("n"))
            .select("arm", "primary", "n", "median_minus_mean")
        )
        for wname in (win.SCHOOL_WINDOW, "bank window", win.ORDINARY, "all rows"):
            wmask = np.ones(frame.height, dtype=bool) if wname == "all rows" else masks[wname]
            for kname, kmask in (
                ("any weather", np.ones(frame.height, dtype=bool)),
                *[(k, kinds == i) for i, k in enumerate(win.WEATHER_KINDS)],
            ):
                for bname, (lo, hi) in bands.items():
                    for pname in ("held-in", "july to september", "october"):
                        sel = wmask & kmask & (lead >= lo) & (lead <= hi) & (period == pname)
                        if not sel.any():
                            continue
                        sub = frame.filter(pl.Series(sel))
                        for gname, idx in groups.items():
                            g = sub.filter(pl.col("primary").is_in(idx))
                            if g.height == 0:
                                continue
                            n = g["n"].sum()
                            parts.append({"arm": arm, "window": wname, "weather": kname, "lead": bname, "period": pname, "group": gname, "rows": int(n),
                                          **{c: float(g[c].sum() / n) for c in SUM_COLUMNS if c != "s1236"}})  # fmt: skip
    if parts:
        pl.DataFrame(parts).write_csv(out / "metrics.csv")
    if gaps:
        pl.concat(gaps).write_csv(out / "median_minus_mean.csv")


# ----------------------------------------------------------------------------- gate and smoke


def stage_gate() -> int:
    """Compare R0 in the gate month with experiment 677's stored forecasts and print the verdict of the gate."""
    worst = 0.0
    only_r0 = only_stored = 0
    for seed in SEEDS:
        a = contract_from_fit(path=fit_path(root=ROOT, arm="R0", month=GATE_MONTH, seed=seed))
        b = contract_from_stored(month=GATE_MONTH, seed=seed)
        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
        import e00667_export as x667  # ty: ignore[unresolved-import]

        cap = x667.capacity_mw().select(pl.col("time_series_id").cast(pl.Int8), "capacity_mw")
        j = a.join(b, on=keys, how="full", suffix="_stored", coalesce=True)
        both = j.filter(pl.col("q50").is_not_null() & pl.col("q50_stored").is_not_null()).join(
            cap, on="time_series_id"
        )
        only_r0 += j.filter(pl.col("q50_stored").is_null()).height
        only_stored += j.filter(pl.col("q50").is_null()).height
        for q in x667.QCOLS:
            diff = (both[q] - both[f"{q}_stored"]).abs() / both["capacity_mw"]
            worst = max(worst, float(diff.max()))
        print(
            f"gate seed {seed}: rows common {both.height}, only R0 {only_r0}, only stored {only_stored}, worst so far {worst:.2e} of capacity",
            flush=True,
        )
    passed = worst <= GATE_TOLERANCE
    print(
        f"GATE {'PASSED' if passed else 'FAILED'}: worst quantile difference {worst:.2e} of capacity (tolerance {GATE_TOLERANCE:.0e})"
    )
    (ROOT / "scores").mkdir(parents=True, exist_ok=True)
    (ROOT / "scores" / "gate.json").write_text(json.dumps({"worst": worst, "tolerance": GATE_TOLERANCE, "passed": passed, "rows_only_r0": only_r0, "rows_only_stored": only_stored}))  # fmt: skip
    return 0 if passed else 1


def stage_smoke() -> int:
    """Score each smoke arm's stage-3 validation rows, check the quantiles, and print the time of each arm-month."""
    folder = SMOKE_ROOT / "arms"
    taus = np.asarray([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
    base: np.ndarray | None = None
    for arm in ARMS:
        path = next(
            iter((folder / f"k1_{arm}_sp1" / "work" / "valpred").glob("*_s0.parquet")), None
        )
        if path is None:
            print(arm, "no valpred file")
            continue
        f = pl.read_parquet(path)
        q = (
            f.select([f"q{i:02d}" for i in (1, 5, 10, 25, 40, 50, 60, 75, 90, 95, 99)])
            .to_numpy()
            .astype(np.float64)
        )
        y = f["y"].to_numpy().astype(np.float64)
        assert np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all(), (
            f"{arm}: quantiles not finite and sorted"
        )
        d = y[:, None] - q
        crps = float((2.0 * np.maximum(taus * d, (taus - 1) * d).mean(axis=1)).mean() * 100.0)
        mae = float(np.abs(y - q[:, 5]).mean() * 100.0)
        base = q[:, 5] if arm == "R0" else base
        diff = (
            f"{float(np.abs(q[:, 5] - base).max()):.2e}"
            if base is not None and arm != "R0" and len(base) == len(q)
            else "n/a"
        )
        print(
            f"smoke {arm}: rows {f.height}, CRPS {crps:.3f}, MAE of median {mae:.3f} (points of capacity, the stage-3 validation rows), max |median - R0 median| {diff}"
        )
    times = SMOKE_ROOT / "logs" / "phase_times.csv"
    if times.exists():
        t = pl.read_csv(times).with_columns(
            minutes=(
                pl.col("end_utc").str.to_datetime() - pl.col("start_utc").str.to_datetime()
            ).dt.total_seconds()
            / 60.0
        )
        print(t.group_by("arm").agg(pl.col("minutes").sum()).sort("arm"))
    return 0


def main() -> int:
    """Run the requested stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cells", "contrasts", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument(
        "--stored", action="store_true", help="score the stored arm 9a and let it stand in for R0"
    )
    args = parser.parse_args()
    if args.smoke:
        return stage_smoke()
    if args.gate:
        return stage_gate()
    if args.stage in {"cells", "all"}:
        stage_cells(arms=[*args.arms, *(["stored"] if args.stored else [])], stored=args.stored)
    if args.stage in {"contrasts", "all"}:
        stage_contrasts(r0_from_stored=args.stored)
    return 0


if __name__ == "__main__":
    sys.exit(main())

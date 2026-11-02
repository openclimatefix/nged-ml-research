# ruff: noqa: E501, PERF401
"""Score experiment 523's Parts 1a, 1b, and 2 and the pretrained-versus-unpretrained contrast (edit 8).

Inputs are the per-month panels of `e00523_panel.py` (raw stored quantiles of L-1d, K2-S1, LL, and
L0PT on experiment 511's scored rows). No network is fitted. Units: points of effective capacity.
Row sets and lead bands are those of experiment 521 (`eight`: MVA-only Primaries; `six`: signed-MW
Primaries, rows with no flagged class). Widths and scores use the 1.15 widening that experiment 511
applied, unless a table says `raw`. Hour blocks are four-hour blocks of the UTC target half-hour.

Part 2 estimates, for each daily cutoff, one factor per tail quantile (q01, q05, q95, q99), lead
band, and metering group from stored L-1d forecasts whose target is at least 3.5 hours (7
half-hours) before the cutoff. A factor `k` makes the share of actuals beyond `q50 +/- k * (q_t -
q50)` equal the quantile level, so it is the `t`-quantile (upper tails) or the `(1 - t)`-quantile
(lower tails) of the standardised residual `(y - q50) / (q_t - q50)` (or `(q50 - y) / (q50 - q_t)`),
floored at 0.5 and capped at 3. A band with fewer than `MIN_POOL` rows falls back to 1.15.

```bash
source /tmp/e523_env.sh
uv run --no-sync python ml-experiments/00523-*/e00523_scores.py
```
"""

import json
import math
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00521_refs as refs  # ty: ignore[unresolved-import]
import e00521_stats as st  # ty: ignore[unresolved-import]
import e00523_panel as panel  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SCORES: Final[Path] = panel.DATA / "scores"
LEVELS: Final[tuple[str, ...]] = common.LEVEL_COLUMNS
LEVEL_VALUES: Final[np.ndarray] = common.LEVELS
POINTS: Final[float] = 100.0
TAIL_NAMES: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
TAIL_INDEX: Final[list[int]] = [LEVELS.index(n) for n in TAIL_NAMES]
TAIL_LEVEL: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
MEDIAN: Final[int] = LEVELS.index("q50")
FLAT: Final[float] = 1.15
MIN_POOL: Final[int] = 2_000
FACTOR_RANGE: Final[tuple[float, float]] = (0.5, 3.0)
CUTOFF_AGE_HALF_HOURS: Final[int] = 7
FIT_MONTHS: Final[tuple[str, ...]] = ("2025-12", "2026-01", "2026-03", "2026-06")
PT_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2026-01", "2026-03", "2026-06")
SETS2: Final[tuple[str, ...]] = ("eight", "six")
CELLS: Final[dict[str, tuple[str, ...]]] = {
    "L-1d": ("L1d",),
    "K2-S1": ("K2c0", "K2c1", "K2c2"),
    "LL": ("LLc0", "LLc1", "LLc2"),
    "L0PT": ("PTs0", "PTs1", "PTs2"),
}
HOUR_BLOCKS: Final[int] = 48
"""Hour blocks are single UTC half-hours: every origin is at 09:00 UTC, so lead fixes the half-hour."""
GROUP_OF_SET: Final[dict[str, str]] = {"eight": "mva_eight", "six": "signed_six"}
GRID_ORIGIN_CHECK: Final[int] = 48


def load_panel() -> pl.DataFrame:
    """Return the five months' panels with lead band and hour block."""
    frames = [pl.read_parquet(panel.panel_path(month=m)) for m in refs.MONTHS]
    return pl.concat(frames).with_columns(
        band=st.band_expr(), block=pl.col("half_hour").cast(pl.Int8)
    )


def quantiles(*, table: pl.DataFrame, cell: str) -> np.ndarray:
    """Return a cell's raw quantiles, shape (rows, 11), as float64."""
    return table.select([f"{cell}__{q}" for q in LEVELS]).to_numpy().astype(np.float64)


def widen(*, q: np.ndarray, factor: np.ndarray | float) -> np.ndarray:
    """Scale the four tail quantiles' distance from the median by `factor` (scalar or (rows, 4)); sort."""
    out = q.copy()
    med = q[:, MEDIAN : MEDIAN + 1]
    out[:, TAIL_INDEX] = med + factor * (q[:, TAIL_INDEX] - med)
    return np.sort(out, axis=1)


def metrics(*, q: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row CRPS, tail pinball loss and coverage, and the 90% width (points of capacity)."""
    err = y[:, None] - q
    pin = np.maximum(LEVEL_VALUES[None, :] * err, (LEVEL_VALUES[None, :] - 1.0) * err)
    out = {"crps": 2.0 * POINTS * (pin * common.ATOM_MASSES[None, :]).sum(axis=1)}
    for name, i in zip(TAIL_NAMES, TAIL_INDEX, strict=True):
        out[f"pin_{name}"] = POINTS * pin[:, i]
        out[f"cov_{name}"] = (y <= q[:, i]).astype(np.float64)
    out["w90"] = POINTS * (q[:, LEVELS.index("q95")] - q[:, LEVELS.index("q05")])
    out["w98"] = POINTS * (q[:, LEVELS.index("q99")] - q[:, LEVELS.index("q01")])
    return out


def finite_rows(*, table: pl.DataFrame, cells: tuple[str, ...]) -> np.ndarray:
    """Return a mask of rows where every cell's quantiles and the actual are finite."""
    ok = np.isfinite(table["actual"].to_numpy())
    for cell in cells:
        ok &= np.isfinite(quantiles(table=table, cell=cell)).all(axis=1)
    return ok


def arm_metrics(*, table: pl.DataFrame, arm: str, factor: float = FLAT) -> pl.DataFrame:
    """Return the cell-averaged per-row metrics of an arm, `arm__metric` columns, NaN where absent."""
    y = table["actual"].to_numpy().astype(np.float64)
    per_cell = []
    for cell in CELLS[arm]:
        q = quantiles(table=table, cell=cell)
        ok = np.isfinite(q).all(axis=1) & np.isfinite(y)
        m = metrics(q=widen(q=np.where(ok[:, None], q, 0.0), factor=factor), y=np.where(ok, y, 0.0))
        per_cell.append({k: np.where(ok, v, np.nan) for k, v in m.items()})
    return pl.DataFrame(
        {
            f"{arm}__{k}": pl.Series(np.mean([c[k] for c in per_cell], axis=0), dtype=pl.Float64)
            for k in per_cell[0]
        }
    )


def pit_fractions(*, q: np.ndarray, y: np.ndarray, bins: int = 10) -> np.ndarray:
    """Return the share of rows in each of `bins` PIT bins from the 11-quantile predictive function.

    The PIT is the piecewise-linear interpolation between stored quantile levels. An actual below
    q01 gets 0.005 and above q99 gets 0.995, so the end bins hold the tail mass.
    """
    count = (q <= y[:, None]).sum(axis=1)
    k = np.clip(count, 1, len(LEVELS) - 1)
    lo, hi = q[np.arange(len(y)), k - 1], q[np.arange(len(y)), k]
    span = np.where(hi > lo, hi - lo, 1.0)
    frac = np.where(hi > lo, (y - lo) / span, 0.5)
    pit = LEVEL_VALUES[k - 1] + np.clip(frac, 0.0, 1.0) * (LEVEL_VALUES[k] - LEVEL_VALUES[k - 1])
    pit = np.where(count == 0, 0.005, np.where(count == len(LEVELS), 0.995, pit))
    hist = np.bincount(np.minimum((pit * bins).astype(int), bins - 1), minlength=bins)
    return hist / max(len(y), 1)


# --- Part 1: width, growth, PIT ---------------------------------------------------------------


def width_tables(*, table: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return mean 90% width by set, band, hour block, and arm, and the width-growth ratios."""
    records = []
    for arm in CELLS:
        scored = arm_metrics(table=table, arm=arm)
        frame = (
            table.select("group", "cls", "main16", "month", "band", "block")
            .hstack(scored.select(w90=f"{arm}__w90", crps=f"{arm}__crps"))
            .with_columns(arm=pl.lit(arm))
        )
        records.append(frame)
        if arm == "LL":
            records.append(
                frame.filter(pl.col("month").is_in(list(PT_MONTHS))).with_columns(
                    arm=pl.lit("LL (L0PT folds)")
                )
            )
    long = pl.concat(records).filter(pl.col("w90").is_finite())
    out = []
    for name in st.SETS[:2]:
        part = st.row_set(table=long, name=name)
        for arm_name, sub in part.group_by("arm"):
            by = sub.group_by("band", "block").agg(pl.len().alias("n"), pl.col("w90").mean())
            by_band = (
                sub.group_by("band")
                .agg(pl.len().alias("n"), pl.col("w90").mean())
                .with_columns(block=pl.lit(-1, dtype=pl.Int8))
            )
            out.append(
                pl.concat(
                    [
                        by.cast({"block": pl.Int8}),
                        by_band.select(by.columns).cast({"block": pl.Int8}),
                    ]
                ).with_columns(set=pl.lit(name), arm=pl.lit(arm_name[0]))
            )
    widths = pl.concat(out).sort("set", "arm", "block", "band")
    ratios = []
    for (name, arm), sub in widths.group_by("set", "arm"):
        blocks = sub.filter(pl.col("block") >= 0)
        cell = {(r["band"], r["block"]): (r["w90"], r["n"]) for r in blocks.iter_rows(named=True)}

        def mean_width(bands: list[str], k: int, cell: dict = cell) -> float | None:
            """Return the row-weighted mean width of `bands` in hour block `k`, or None if absent."""
            parts = [cell[(b, k)] for b in bands if (b, k) in cell]
            return sum(w * n for w, n in parts) / sum(n for _, n in parts) if parts else None

        for label, num, den in (
            ("120-240h over 12-24h", ["120-240h"], ["12-24h"]),
            ("120-240h over below 12h", ["120-240h"], ["0-6h", "6-12h"]),
        ):
            stratified = []
            for k in range(HOUR_BLOCKS):
                top, bottom = mean_width(num, k), mean_width(den, k)
                if top is not None and bottom is not None:
                    stratified.append(top / bottom)
            ratios.append(
                {"set": name, "arm": arm, "ratio": label,
                 "pooled": pooled_ratio(sub=sub, num=num, den=den),
                 "hour_stratified": float(np.mean(stratified)), "blocks_used": len(stratified),
                 "stratified_min": float(np.min(stratified)), "stratified_max": float(np.max(stratified))}
            )  # fmt: skip
    return widths, pl.DataFrame(ratios).sort("set", "ratio", "arm")


def pooled_ratio(*, sub: pl.DataFrame, num: list[str], den: list[str]) -> float:
    """Return the row-weighted mean width of `num` bands over that of `den` bands, all hours."""
    pooled = sub.filter(pl.col("block") < 0)

    def weighted(bands: list[str]) -> float:
        part = pooled.filter(pl.col("band").is_in(bands))
        return float((part["w90"] * part["n"]).sum() / part["n"].sum())

    return weighted(num) / weighted(den)


def pit_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the PIT histogram (raw quantiles) by set, band, and arm; cells are averaged."""
    records = []
    y = table["actual"].to_numpy().astype(np.float64)
    for arm, cells in CELLS.items():
        hists = {}
        for cell in cells:
            q = quantiles(table=table, cell=cell)
            ok = np.isfinite(q).all(axis=1) & np.isfinite(y)
            hists[cell] = (q, ok)
        for name in st.SETS[:2]:
            mask = st.row_set(table=table.with_row_index("i"), name=name)["i"].to_numpy()
            for band in (*st.BANDS, st.ALL):
                sel = np.zeros(len(table), dtype=bool)
                sel[mask] = True
                if band != st.ALL:
                    sel &= (table["band"] == band).to_numpy()
                fracs = []
                for q, ok in hists.values():
                    s = sel & ok
                    fracs.append(pit_fractions(q=q[s], y=y[s]))
                mean = np.mean(fracs, axis=0)
                for i, f in enumerate(mean):
                    records.append(
                        {"set": name, "band": band, "arm": arm, "bin": i, "share": float(f)}
                    )
    return pl.DataFrame(records)


def coverage_gate(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the Part 3 gate: L-1d minus K2-S1 q95 coverage by band and the long-lead shortfall."""
    scored = pl.concat(
        [arm_metrics(table=table, arm="L-1d"), arm_metrics(table=table, arm="K2-S1")],
        how="horizontal",
    )
    full = pl.concat(
        [table.select("group", "cls", "main16", "month", "band"), scored], how="horizontal"
    )
    records = []
    for name in SETS2:
        part = st.row_set(table=full, name=name)
        cov = {}
        for band in st.BANDS:
            sub = part.filter(pl.col("band") == band)
            for q in ("q95", "q99"):
                cov[(band, q)] = float(sub[f"L-1d__cov_{q}"].mean() - sub[f"K2-S1__cov_{q}"].mean())  # ty: ignore[unresolved-attribute]
        for q in ("q95", "q99"):
            gate = cov[("120-240h", q)] - cov[("12-24h", q)]
            records.append(
                {"set": name, "quantile": q, "diff_12_24h": cov[("12-24h", q)],
                 "diff_120_240h": cov[("120-240h", q)], "extra_at_long_lead": gate,
                 "gate_holds_minus_0.03": gate <= -0.03}
            )  # fmt: skip
    return pl.DataFrame(records)


# --- Part 2: per-band widening on held-out fits -----------------------------------------------


def residual_z(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the standardised residual of each tail, shape (rows, 4), NaN when the spread is not positive."""
    med = q[:, MEDIAN]
    z = np.full((len(y), 4), np.nan)
    for j, (name, i) in enumerate(zip(TAIL_NAMES, TAIL_INDEX, strict=True)):
        if TAIL_LEVEL[name] > 0.5:
            d, num = q[:, i] - med, y - med
        else:
            d, num = med - q[:, i], med - y
        z[:, j] = np.where(d > 1e-9, num / np.where(d > 1e-9, d, 1.0), np.nan)
    return z


def causal_factors(
    *, z: np.ndarray, target: np.ndarray, band: np.ndarray, cut_target: int, n_bands: int
) -> tuple[np.ndarray, int]:
    """Return per-band factors (bands, 4) from rows whose target is at or before `cut_target`.

    Returns the factors and how many (band, tail) cells fell back to the flat factor.
    """
    out = np.full((n_bands, 4), FLAT)
    fallbacks = 0
    pool = target <= cut_target
    for b in range(n_bands):
        rows = pool & (band == b)
        for j, name in enumerate(TAIL_NAMES):
            values = z[rows, j]
            values = values[np.isfinite(values)]
            if len(values) < MIN_POOL:
                fallbacks += 1
                continue
            level = TAIL_LEVEL[name] if TAIL_LEVEL[name] > 0.5 else 1.0 - TAIL_LEVEL[name]
            out[b, j] = float(np.clip(np.quantile(values, level), *FACTOR_RANGE))
    return out, fallbacks


def part2(*, table: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return per-row variant scores, the mean factors, and the fallback counts for Part 2."""
    band_names = list(st.BANDS)
    scores, factor_rows, fallback_rows = [], [], []
    for name in SETS2:
        part = st.row_set(table=table, name=name)
        part = part.filter(pl.col("month").is_in(list(refs.MONTHS)))
        ok = finite_rows(table=part, cells=("L1d",))
        part = part.filter(pl.Series(ok))
        q = quantiles(table=part, cell="L1d")
        y = part["actual"].to_numpy().astype(np.float64)
        band = np.array([band_names.index(b) for b in part["band"].to_list()])
        target = part["target"].to_numpy().astype(np.int64)
        origin_day = part["day"].to_numpy().astype(np.int64)
        z = residual_z(q=q, y=y)
        factor_row = np.full((len(y), 4), FLAT)
        flat_causal = np.full((len(y), 4), FLAT)
        scored = np.isin(part["month"].to_numpy(), list(FIT_MONTHS))
        for d in np.unique(origin_day[scored]):
            cut_target = int(GRID_ORIGIN_CHECK * d - CUTOFF_AGE_HALF_HOURS)
            factors, fb = causal_factors(
                z=z, target=target, band=band, cut_target=cut_target, n_bands=len(band_names)
            )
            flat_f, _ = causal_factors(
                z=z, target=target, band=np.zeros_like(band), cut_target=cut_target, n_bands=1
            )
            today = scored & (origin_day == d)
            factor_row[today] = factors[band[today]]
            flat_causal[today] = flat_f[0]
            factor_rows.append(
                pl.DataFrame(
                    {
                        "set": name,
                        "day": int(d),
                        "band": band_names,
                        **{t: factors[:, j] for j, t in enumerate(TAIL_NAMES)},
                    }
                )
            )
            fallback_rows.append({"set": name, "day": int(d), "fallbacks": fb})
        variants = {
            "raw": widen(q=q, factor=1.0),
            "flat1.15": widen(q=q, factor=FLAT),
            "per-band": widen(q=q, factor=factor_row),
            "flat-causal": widen(q=q, factor=flat_causal),
        }
        cols = {}
        for label, qv in variants.items():
            for k, v in metrics(q=qv, y=y).items():
                cols[f"{label}__{k}"] = v
        scores.append(
            part.select("month", "day", "band", "group", "cls", "main16", "lead_hours")
            .with_columns(set=pl.lit(name))
            .hstack(pl.DataFrame(cols))
        )
    return pl.concat(scores), pl.concat(factor_rows), pl.DataFrame(fallback_rows)


def part2_tables(*, scores: pl.DataFrame, factors: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return absolute, factor, and contrast tables for Part 2 (only the held-out folds)."""
    held = scores.filter(pl.col("month").is_in(list(FIT_MONTHS)))
    variants = ("raw", "flat1.15", "per-band", "flat-causal")
    cols = ("crps", *[f"{k}_{t}" for k in ("pin", "cov") for t in TAIL_NAMES], "w90")
    absolute = []
    for name in SETS2:
        part = held.filter(pl.col("set") == name)
        for band in (*st.BANDS, st.ALL):
            sub = part if band == st.ALL else part.filter(pl.col("band") == band)
            for v in variants:
                absolute.append(
                    {"set": name, "band": band, "variant": v, "n": sub.height}
                    | {m: float(sub[f"{v}__{m}"].mean()) for m in cols}  # ty: ignore[invalid-argument-type]
                )
    rng = np.random.default_rng(2)
    contrasts = []
    for name in SETS2:
        part = held.filter(pl.col("set") == name)
        for a, b in (
            ("per-band", "flat1.15"),
            ("flat-causal", "flat1.15"),
            ("per-band", "flat-causal"),
        ):
            for metric, tests in (
                ("crps", st.TESTS),
                ("pin_q99", 2 * st.TESTS),
                ("pin_q95", 2 * st.TESTS),
            ):
                for band in (*st.BANDS, st.ALL):
                    sub = part if band == st.ALL else part.filter(pl.col("band") == band)
                    diff = sub.select(
                        "month", "day", d=pl.col(f"{a}__{metric}") - pl.col(f"{b}__{metric}")
                    )
                    contrasts.append(
                        {"set": name, "contrast": f"{a} - {b}", "metric": metric, "band": band,
                         "counted": a == "per-band" and b == "flat1.15" and band != st.ALL,
                         "tests": tests, **st.block_stats(diff=diff, rng=rng, tests=tests)}
                    )  # fmt: skip
    mean_factors = (
        factors.group_by("set", "band")
        .agg([pl.col(t).mean() for t in TAIL_NAMES])
        .sort("set", "band")
    )
    return {
        "part2_absolute": pl.DataFrame(absolute),
        "part2_contrasts": pl.DataFrame(contrasts),
        "part2_mean_factors": mean_factors,
    }


# --- Edit 8: pretrained against unpretrained ---------------------------------------------------


def edit8(*, table: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return absolute scores, contrasts, and the seed spread of L0PT against LL on the shared folds."""
    both = table.filter(pl.col("month").is_in(list(PT_MONTHS)))
    ok = finite_rows(table=both, cells=("PTs0", "PTs1", "PTs2", "LLc0", "LLc1", "LLc2"))
    both = both.filter(pl.Series(ok))
    pt, ll, k2 = (arm_metrics(table=both, arm=a) for a in ("L0PT", "LL", "K2-S1"))
    scored = (
        both.select("group", "cls", "main16", "month", "day", "band")
        .hstack(pt)
        .hstack(ll)
        .hstack(k2)
    )
    cols = ("crps", *[f"{k}_{t}" for k in ("pin", "cov") for t in TAIL_NAMES], "w90")
    absolute, contrasts, mde = [], [], []
    rng = np.random.default_rng(3)
    # one-cell scores for the seed spread (mean over folds of each cell's CRPS)
    per_cell: dict[str, dict[str, np.ndarray]] = {}
    y = both["actual"].to_numpy().astype(np.float64)
    for arm in ("L0PT", "LL"):
        for cell in CELLS[arm]:
            per_cell.setdefault(arm, {})[cell] = metrics(
                q=widen(q=quantiles(table=both, cell=cell), factor=FLAT), y=y
            )["crps"]
    cellframe = both.select("group", "cls", "main16", "month", "day", "band").hstack(
        pl.DataFrame({f"{arm}__{cell}": v for arm, d in per_cell.items() for cell, v in d.items()})
    )
    for name in SETS2:
        part = st.row_set(table=scored, name=name)
        cells = st.row_set(table=cellframe, name=name)
        for band in (*st.BANDS, st.ALL):
            sub = part if band == st.ALL else part.filter(pl.col("band") == band)
            csub = cells if band == st.ALL else cells.filter(pl.col("band") == band)
            for arm in ("L0PT", "LL", "K2-S1"):
                absolute.append(
                    {"set": name, "band": band, "arm": arm, "n": sub.height}
                    | {m: float(sub[f"{arm}__{m}"].mean()) for m in cols}  # ty: ignore[invalid-argument-type]
                )
            for arm, cell_names in (("L0PT", CELLS["L0PT"]), ("LL", CELLS["LL"])):
                means = (
                    csub.group_by("month")
                    .agg([pl.col(f"{arm}__{c}").mean() for c in cell_names])
                    .sort("month")
                )
                values = means.select([f"{arm}__{c}" for c in cell_names]).to_numpy()
                resid = values - values.mean(axis=1, keepdims=True)
                sd = math.sqrt(float((resid**2).sum()) / (values.shape[0] * (values.shape[1] - 1)))
                mde.append(
                    {"set": name, "band": band, "source": arm, "seed_sd": sd,
                     "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(values.shape[1])}
                )  # fmt: skip
            for metric, tests in (
                ("crps", st.TESTS),
                ("pin_q95", 2 * st.TESTS),
                ("pin_q99", 2 * st.TESTS),
            ):
                diff = sub.select(
                    "month", "day", d=pl.col(f"L0PT__{metric}") - pl.col(f"LL__{metric}")
                )
                contrasts.append(
                    {"set": name, "contrast": "L0PT - LL", "metric": metric, "band": band,
                     "counted": band != st.ALL, "tests": tests, **st.block_stats(diff=diff, rng=rng, tests=tests)}
                )  # fmt: skip
    return {
        "edit8_absolute": pl.DataFrame(absolute),
        "edit8_contrasts": pl.DataFrame(contrasts),
        "edit8_mde": pl.DataFrame(mde),
    }


def reproduction_checks(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return checks of the panel against experiment 511's stored scores and the NaN counts."""
    records = []
    for month in refs.MONTHS:
        raw = pl.read_parquet(
            refs.E511 / "scores" / f"rows_{month}.parquet",
            columns=[*refs.KEYS, "L-1d__crps", "L-1d__pin_q05", "L-1d__cov_q05", "main16"],
        )
        mine = table.filter(pl.col("month") == month)
        scored = arm_metrics(table=mine, arm="L-1d")
        joined = mine.select(*refs.KEYS).hstack(scored).join(raw, on=refs.KEYS, how="inner")
        both = joined.filter(
            pl.col("L-1d__crps").is_finite() & pl.col("L-1d__crps_right").is_finite()
        )
        records.append(
            {
                "check": f"{month} mean |CRPS panel - CRPS 511|",
                "value": float((both["L-1d__crps"] - both["L-1d__crps_right"]).abs().mean()),
            }
        )  # ty: ignore[invalid-argument-type]
        records.append(
            {
                "check": f"{month} rows with NaN L-1d crps in 511",
                "value": float(raw["L-1d__crps"].is_nan().sum()),
            }
        )
        records.append(
            {
                "check": f"{month} rows with NaN L-1d pin_q05 in 511",
                "value": float(raw["L-1d__pin_q05"].is_nan().sum()),
            }
        )
        records.append(
            {
                "check": f"{month} rows with null L-1d pin_q05 in 511",
                "value": float(raw["L-1d__pin_q05"].is_null().sum()),
            }
        )
        records.append({"check": f"{month} rows in panel", "value": float(mine.height)})
        records.append(
            {
                "check": f"{month} L-1d rows missing in panel",
                "value": float((~finite_rows(table=mine, cells=("L1d",))).sum()),
            }
        )
    return pl.DataFrame(records)


def main() -> None:
    """Write every table to this experiment's `scores` folder."""
    SCORES.mkdir(parents=True, exist_ok=True)
    st.RESAMPLES = 100_000
    table = load_panel()
    out: dict[str, pl.DataFrame] = {"checks": reproduction_checks(table=table)}
    widths, ratios = width_tables(table=table)
    out |= {"widths_by_hour": widths, "width_ratios": ratios, "pit": pit_table(table=table)}
    out["gate"] = coverage_gate(table=table)
    scores, factors, fallbacks = part2(table=table)
    out |= part2_tables(scores=scores, factors=factors)
    out["part2_fallbacks"] = fallbacks
    out |= edit8(table=table)
    for name, frame in out.items():
        frame.write_csv(SCORES / f"{name}.csv")
    print(json.dumps({k: v.shape for k, v in out.items()}))


if __name__ == "__main__":
    main()

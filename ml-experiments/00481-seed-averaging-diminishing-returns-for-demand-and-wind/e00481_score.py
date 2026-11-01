# ruff: noqa: E501, ISC004, PERF401, PLR0915
r"""Score experiment 481: how the score falls as more fit seeds are averaged, per family.

Three families, each a pool of fit seeds whose quantile forecasts are averaged in quantile space:

- `demand`: experiment 425's arm A, seeds 0 to 9, five months, widened about the median;
- `wind`: experiment 429's `member_plain`, seeds 0 to 9, 53 Tuesday runs of three farms (named A,
  B, and C here);
- `s1`: experiment 404's S1 network as experiment 454 scores it (lagged heads, rows of experiment
  454), seeds 0 to 5.

For each K the score is the mean over 100 random subsets of K seeds (all subsets if fewer exist;
`numpy.random.default_rng(481)` per family). Every score is the mean over months of the month's
value, where a month's value is the mean over Primaries (demand) or over rows (wind, S1) of the
row losses. The minimum detectable effect and the month-block bootstrap use the same month values.

```bash
# demand (PYTHONPATH of e00425_fit.py)
uv run --no-sync python ml-experiments/00481-*/e00481_score.py demand
# wind (source ml-experiments/00429-*/env.sh)
uv run --no-sync python ml-experiments/00481-*/e00481_score.py wind
# s1 (PYTHONPATH of e00481_s1_heads.py)
uv run --no-sync python ml-experiments/00481-*/e00481_score.py s1
# the assembled report (needs only numpy and polars)
uv run --no-sync python ml-experiments/00481-*/e00481_score.py report
```
Each family writes its text to `work/report_parts/<family>.txt` and its tables to
`work/figure_tables/`; `report` joins them into `score_report.txt`, the minimum detectable effects
and the Bonferroni count first.
"""

import itertools
import json
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # ty: ignore[unresolved-import]

OWN_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00481-seed-averaging-diminishing-returns-for-demand-and-wind"
)
TABLE_DIR: Final[Path] = OWN_DIR / "work" / "figure_tables"
PART_DIR: Final[Path] = OWN_DIR / "work" / "report_parts"
REPORT_FILE: Final[Path] = OWN_DIR / "score_report.txt"
SEED_BASE: Final[int] = 481
N_DRAWS: Final[int] = 100
N_RESAMPLES: Final[int] = 20_000
N_CONTRASTS: Final[dict[str, int]] = {"demand": 4, "s1": 3, "wind": 4}
BONFERRONI_TOTAL: Final[int] = sum(N_CONTRASTS.values())
ALPHA: Final[float] = 0.05 / BONFERRONI_TOTAL
LEVEL_PERCENT: Final[float] = 100.0 * (1.0 - ALPHA)
WIDENINGS: Final[tuple[float, ...]] = (1.0, 1.05, 1.10, 1.15)
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
TAIL_INDEX: Final[dict[str, int]] = {"q01": 0, "q05": 1, "q95": 9, "q99": 10}
METRICS: Final[tuple[str, ...]] = (
    "crps",
    *(f"pin_{t}" for t in TAIL_INDEX),
    *(f"cov_{t}" for t in TAIL_INDEX),
)
GPU_BATCH: Final[int] = 65536
FIVE_PERCENT_RULE: Final[float] = 1.05
DEMAND_KS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
WIND_KS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
S1_KS: Final[tuple[int, ...]] = (1, 2, 3, 6)
MONTHS_DEMAND: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
FIRST_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SECOND_SEEDS: Final[tuple[int, ...]] = (3, 4)


# ---------------------------------------------------------------------------------------------
# Generic machinery
# ---------------------------------------------------------------------------------------------


def seed_subsets(*, pool: int, k: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Return every K-subset of the pool if at most `N_DRAWS` exist, else `N_DRAWS` distinct random ones."""
    if math.comb(pool, k) <= N_DRAWS:
        return list(itertools.combinations(range(pool), k))
    chosen: set[tuple[int, ...]] = set()
    while len(chosen) < N_DRAWS:
        chosen.add(tuple(sorted(int(s) for s in rng.choice(pool, size=k, replace=False))))
    return sorted(chosen)


def widen(*, q: np.ndarray, factor: float, bounds: tuple[float, float] | None = None) -> np.ndarray:
    """Return quantiles (rows, 11) with q01, q05, q95, and q99 moved `factor` times as far from the median.

    `bounds` clips the widened quantiles to the physical range (wind shares of capacity).
    """
    out = q.astype(np.float64, copy=True)
    median = out[:, 5].copy()
    for position in TAIL_INDEX.values():
        out[:, position] = median + factor * (q[:, position] - median)
        if bounds is not None:
            out[:, position] = np.clip(out[:, position], *bounds)
    return out


def pinball_arrays(
    *, q: np.ndarray, actual: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the per-row losses of `METRICS`; `weights` (11,) turn the 11 pinball losses into `crps`."""
    error = actual[:, None] - q
    pinball = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    out = {"crps": pinball @ weights}
    for name, position in TAIL_INDEX.items():
        out[f"pin_{name}"] = pinball[:, position]
        out[f"cov_{name}"] = (actual <= q[:, position]).astype(np.float64)
    return out


@dataclass
class Combo:
    """One (group, row mask): the rows, their cell index (label, month), and counts per cell."""

    rows: np.ndarray
    cell: np.ndarray
    counts: np.ndarray
    n_labels: int
    metrics: tuple[str, ...]


class Aggregator:
    """Month values and scores of row losses over named (group, mask) combinations."""

    def __init__(
        self,
        *,
        label: np.ndarray,
        month: np.ndarray,
        n_months: int,
        groups: dict[str, list[int]],
        masks: dict[str, np.ndarray],
        head_masks: tuple[str, ...],
    ) -> None:
        """Precompute the row selection of every combination; tails only on the `head_masks`."""
        self.n_months = n_months
        self.combos: dict[tuple[str, str], Combo] = {}
        for group, members in groups.items():
            in_group = np.isin(label, members)
            lookup = {int(m): i for i, m in enumerate(members)}
            for name, mask in masks.items():
                rows = np.flatnonzero(in_group & mask)
                if rows.size == 0:
                    continue
                local = np.array([lookup[int(v)] for v in label[rows]])
                cell = local * n_months + month[rows]
                counts = np.bincount(cell, minlength=len(members) * n_months).astype(np.float64)
                metrics = METRICS if name in head_masks else ("crps",)
                self.combos[group, name] = Combo(rows, cell, counts, len(members), metrics)

    def month_values(self, *, combo: tuple[str, str], values: np.ndarray) -> np.ndarray:
        """Return the value of each month: the mean over the group's labels of their month mean."""
        c = self.combos[combo]
        sums = np.bincount(c.cell, weights=values[c.rows], minlength=c.n_labels * self.n_months)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_cell = (sums / c.counts).reshape(c.n_labels, self.n_months)
        return np.nanmean(per_cell, axis=0)

    def score(self, *, combo: tuple[str, str], values: np.ndarray) -> float:
        """Return the score: the mean over months of the month values."""
        return float(np.nanmean(self.month_values(combo=combo, values=values)))


def month_block_interval(
    *, differences: np.ndarray, level_percent: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Return the interval of the mean of per-month differences, resampling whole months."""
    draws = rng.integers(0, len(differences), size=(N_RESAMPLES, len(differences)))
    means = differences[draws].mean(axis=1)
    tail = (100.0 - level_percent) / 2.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


@dataclass
class Family:
    """A pool of seed forecasts and everything needed to score an average of them."""

    name: str
    q: np.ndarray
    """Seed quantiles, shape (pool, rows, 11), in the scoring unit of `row_arrays`."""
    row_arrays: Callable[[np.ndarray], dict[str, np.ndarray]]
    agg: Aggregator
    primary_combo: tuple[str, str]
    """The combination that carries the decision rule (the signed-MW six, days 1 to 10)."""
    primary_widening: float
    month_names: list[str]
    ks: tuple[int, ...]
    bounds: tuple[float, float] | None = None
    forecast: Callable[[tuple[int, ...]], np.ndarray] | None = None
    """Optional override of the quantile average: the forecast (rows, 11) of a seed subset."""

    @property
    def pool(self) -> int:
        """Return the number of seeds in the pool."""
        return self.q.shape[0]

    @property
    def n_rows(self) -> int:
        """Return the number of scored rows."""
        return self.q.shape[1]

    def mean_q(self, subset: tuple[int, ...]) -> np.ndarray:
        """Return the forecast of a seed subset: the quantile average unless `forecast` is set."""
        if self.forecast is not None:
            return self.forecast(subset)
        return self.q[list(subset)].mean(axis=0, dtype=np.float64)


def run_curves(*, fam: Family, extra_combo_metrics: bool = True) -> tuple[pl.DataFrame, dict]:
    """Score every K over its subsets and widenings.

    Args:
        fam: The family of seeds to score.
        extra_combo_metrics: Whether to score the extra row combinations as well.

    Returns:
        A long table of one score per K, subset, widening, combination, and metric, and a dict of
        the per-row mean CRPS over subsets at the primary widening for each K.
    """
    rng = np.random.default_rng(SEED_BASE)
    pool = fam.pool
    records: list[tuple] = []
    row_mean: dict[int, np.ndarray] = {}
    for k in fam.ks:
        subsets = seed_subsets(pool=pool, k=k, rng=rng)
        accumulated = np.zeros(fam.n_rows)
        for draw, subset in enumerate(subsets):
            mean_q = fam.mean_q(subset)
            for factor in WIDENINGS:
                arrays = fam.row_arrays(widen(q=mean_q, factor=factor, bounds=fam.bounds))
                if factor == fam.primary_widening:
                    accumulated += arrays["crps"]
                for combo, c in fam.agg.combos.items():
                    for metric in c.metrics:
                        records.append(
                            (k, draw, factor, combo[0], combo[1], metric,
                             fam.agg.score(combo=combo, values=arrays[metric]))
                        )  # fmt: skip
        row_mean[k] = accumulated / len(subsets)
        print(f"{fam.name}: K={k}: {len(subsets)} subsets scored", flush=True)
    table = pl.DataFrame(
        records,
        schema=["k", "draw", "widening", "group", "mask", "metric", "value"],
        orient="row",
    )
    return table, row_mean


def summarise_curves(*, draws: pl.DataFrame) -> pl.DataFrame:
    """Return the mean and 5th-to-95th percentile across subsets of each score."""
    return (
        draws.group_by("k", "widening", "group", "mask", "metric")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").quantile(0.05).alias("p05"),
            pl.col("value").quantile(0.95).alias("p95"),
            pl.len().alias("n_draws"),
        )
        .sort("k", "widening", "group", "mask", "metric")
    )


def minimum_detectable_effect(*, fam: Family) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the MDE and the month values of the mean of seeds 0 to 2 and of seeds 3 and 4.

    The MDE is the standard deviation over months of the difference between those two means,
    divided by the square root of two, at the family's primary combination and widening.
    """
    values = []
    for seeds in (FIRST_SEEDS, SECOND_SEEDS):
        mean_q = fam.mean_q(seeds)
        arrays = fam.row_arrays(widen(q=mean_q, factor=fam.primary_widening, bounds=fam.bounds))
        values.append(fam.agg.month_values(combo=fam.primary_combo, values=arrays["crps"]))
    sd = float(np.std(values[0] - values[1], ddof=1))
    return sd / math.sqrt(2.0), values[0], values[1]


def curve_report(
    *, fam: Family, draws: pl.DataFrame, row_mean: dict[int, np.ndarray], mde: float, unit: str
) -> tuple[list[str], dict]:
    """Return the text lines of the K curve, gains, widening, and tails, and the summary figures."""
    lines: list[str] = []
    curve = summarise_curves(draws=draws)
    group, mask = fam.primary_combo
    primary = curve.filter(
        (pl.col("widening") == fam.primary_widening)
        & (pl.col("group") == group)
        & (pl.col("mask") == mask)
    )

    def at(metric: str, k: int, w: float | None = None, g: str = group, m: str = mask) -> float:
        row = curve.filter(
            (pl.col("k") == k)
            & (pl.col("widening") == (fam.primary_widening if w is None else w))
            & (pl.col("group") == g)
            & (pl.col("mask") == m)
            & (pl.col("metric") == metric)
        )
        return float(row["mean"][0])

    lines.append(f"\n== {fam.name}: score against K ({unit}; primary row set {group} / {mask}; "
                 f"widening {fam.primary_widening}) ==")  # fmt: skip
    lines.append("mean over subsets (5th to 95th percentile across subsets):")
    for k in fam.ks:
        row = primary.filter((pl.col("k") == k) & (pl.col("metric") == "crps"))
        lines.append(
            f"K={k:2d}: {row['mean'][0]:.4f} ({row['p05'][0]:.4f} to {row['p95'][0]:.4f}); "
            f"{row['n_draws'][0]} subsets"
        )
    other_groups = sorted({g for g, m in fam.agg.combos if m == mask and g != group})
    for g in other_groups:
        lines.append(
            f"{g} (mean over subsets): "
            + ", ".join(f"K={k}: {at('crps', k, g=g):.4f}" for k in fam.ks)
        )
    for slice_mask in sorted({m for _g, m in fam.agg.combos if m != mask}):
        lines.append(
            f"slice {slice_mask}, {group}: "
            + ", ".join(
                f"K={k}: {at('crps', k, m=slice_mask):.4f}"
                for k in fam.ks
                if (group, slice_mask) in fam.agg.combos
            )
        )

    # Marginal gains.
    rng = np.random.default_rng(SEED_BASE + 1)
    lines.append(
        f"\n-- marginal gain (score at K minus score at the next K; positive means the next K is "
        f"better), month-block bootstrap {N_RESAMPLES:,} resamples, level {LEVEL_PERCENT:.2f}% "
        f"(Bonferroni, {BONFERRONI_TOTAL} contrasts in all, {N_CONTRASTS.get(fam.name, 0)} here); "
        f"MDE {mde:.4f} --"
    )
    gains = []
    for k_now, k_next in itertools.pairwise(fam.ks):
        diff = fam.agg.month_values(combo=fam.primary_combo, values=row_mean[k_now]) - (
            fam.agg.month_values(combo=fam.primary_combo, values=row_mean[k_next])
        )
        low, high = month_block_interval(differences=diff, level_percent=LEVEL_PERCENT, rng=rng)
        gain = float(diff.mean())
        gains.append(
            {"k": k_now, "k_next": k_next, "gain": gain, "low": low, "high": high,
             "inside_mde": abs(gain) < mde, "months_better": int((diff > 0).sum()),
             "n_months": len(diff)}
        )  # fmt: skip
        lines.append(
            f"K={k_now:2d} to K={k_next:2d}: gain {gain:+.4f} ({low:+.4f} to {high:+.4f}); "
            f"{'inside' if abs(gain) < mde else 'outside'} the MDE; "
            f"next K better in {gains[-1]['months_better']} of {len(diff)} months"
        )
    stays_from = None
    for i in range(len(gains)):
        if all(g["inside_mde"] for g in gains[i:]):
            stays_from = gains[i]["k"]
            break
    lines.append(
        f"marked K (gain to the next K inside the MDE, and inside for every larger K): "
        f"{stays_from if stays_from is not None else 'none'}"
    )
    pl.DataFrame(gains).write_parquet(TABLE_DIR / f"{fam.name}_gain.parquet")

    # Widening per K.
    lines.append(f"\n-- best widening per K ({group} / {mask}) --")
    widening_rows = []
    for k in fam.ks:
        crps = {w: at("crps", k, w) for w in WIDENINGS}
        best = min(crps, key=lambda w: crps[w])
        cov95 = {w: at("cov_q95", k, w) for w in WIDENINGS}
        cov99 = {w: at("cov_q99", k, w) for w in WIDENINGS}
        cov99_best = min(cov99, key=lambda w: abs(cov99[w] - 0.99))
        lines.append(
            f"K={k:2d}: best by score {best:.2f} "
            f"(" + ", ".join(f"{w:.2f}: {crps[w]:.4f}" for w in WIDENINGS) + "); "
            "coverage q95 " + ", ".join(f"{w:.2f}: {cov95[w]:.3f}" for w in WIDENINGS)
            + "; q99 " + ", ".join(f"{w:.2f}: {cov99[w]:.4f}" for w in WIDENINGS)
            + f"; q99 coverage nearest 0.99 at {cov99_best:.2f}"
        )  # fmt: skip
        widening_rows.append(
            {"k": k, "best_widening": best, "q99_nearest_widening": cov99_best}
            | {f"crps_{w:.2f}": crps[w] for w in WIDENINGS}
            | {f"cov_q95_{w:.2f}": cov95[w] for w in WIDENINGS}
            | {f"cov_q99_{w:.2f}": cov99[w] for w in WIDENINGS}
        )
    pl.DataFrame(widening_rows).write_parquet(TABLE_DIR / f"{fam.name}_widening.parquet")

    # Tails at the primary widening.
    lines.append(
        f"\n-- tails at widening {fam.primary_widening} ({group} / {mask}); coverage nominal 0.01, 0.05, 0.95, 0.99 --"
    )
    for k in fam.ks:
        lines.append(
            f"K={k:2d}: coverage "
            + ", ".join(f"{t} {at('cov_' + t, k):.4f}" for t in TAIL_INDEX)
            + "; pinball "
            + ", ".join(f"{t} {at('pin_' + t, k):.4f}" for t in TAIL_INDEX)
        )

    # Recommendation.
    k_max = fam.ks[-1]
    ceiling = at("crps", k_max)
    ceiling_q99 = at("pin_q99", k_max)
    lines.append(
        f"\n-- recommendation rule: smallest K with score within the MDE ({mde:.4f}) of K={k_max} and q99 pinball within 5% --"
    )
    recommended = None
    for k in fam.ks:
        gap = at("crps", k) - ceiling
        ratio = at("pin_q99", k) / ceiling_q99
        ok = gap <= mde and ratio <= FIVE_PERCENT_RULE
        lines.append(
            f"K={k:2d}: score gap to K={k_max} {gap:+.4f}; q99 pinball ratio {ratio:.4f}; {'meets both' if ok else 'fails'}"
        )
        if ok and recommended is None:
            recommended = k
    lines.append(f"recommended K: {recommended}")
    summary = {
        "family": fam.name,
        "mde": mde,
        "recommended_k": recommended,
        "marked_k": stays_from,
        "gains": gains,
        "primary_widening": fam.primary_widening,
        "score_by_k": {k: at("crps", k) for k in fam.ks},
    }
    return lines, summary


def save_family(*, fam: Family, draws: pl.DataFrame, lines: list[str], summary: dict) -> None:
    """Write the family's tables, text part, and summary."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)
    draws.write_parquet(TABLE_DIR / f"{fam.name}_draws.parquet")
    summarise_curves(draws=draws).write_parquet(TABLE_DIR / f"{fam.name}_curve.parquet")
    (PART_DIR / f"{fam.name}.txt").write_text("\n".join(lines) + "\n")
    (PART_DIR / f"{fam.name}.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------
# Demand: experiment 425's arm A
# ---------------------------------------------------------------------------------------------


def run_demand() -> None:
    """Score arm A over K = 1, 2, 3, 5, and 10."""
    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    import e00377_score as s377  # ty: ignore[unresolved-import]
    import e00397_score as s397  # ty: ignore[unresolved-import]
    import e00407_score as s407  # ty: ignore[unresolved-import]
    import e00425_fit as fit  # ty: ignore[unresolved-import]
    import e00425_prep as prep  # ty: ignore[unresolved-import]
    from studies.eval_protocol import metrics

    keys = list(common.KEYS)
    levels = list(common.LEVEL_COLUMNS)
    series = adapter.series_table()
    ctx = prep.cleaned_scoring_context(series=series)

    # Load the pool: seeds 0 to 4 from experiment 425's folder, 5 to 9 from this one.
    pieces: list[pl.DataFrame] = []
    stacks: list[np.ndarray] = []
    for month in MONTHS_DEMAND:
        seed_frames = []
        for seed in range(10):
            folder = (fit.OUT_DIR if seed < 5 else OWN_DIR / "work") / "seed_preds" / "A"
            seed_frames.append(pl.read_parquet(folder / f"{month}_s{seed}.parquet").sort(keys))
        first = seed_frames[0].select(keys)
        for seed, frame in enumerate(seed_frames):
            assert frame.select(keys).equals(first), f"{month} seed {seed}: keys differ from seed 0"
        pieces.append(first.with_columns(month=pl.lit(month)))
        stacks.append(np.stack([f.select(levels).to_numpy() for f in seed_frames]))
    key_frame = pl.concat(pieces).with_row_index("pos")
    q_all = np.concatenate(stacks, axis=1)  # (10, rows, 11), fraction of capacity
    print(f"demand pool: {q_all.shape}")

    # The shared scored rows: arm A's ten-seed mean (widened 1.15) with the references.
    mean_widened = widen(q=q_all.mean(axis=0, dtype=np.float64), factor=1.15)
    stored = key_frame.select(keys).with_columns(
        *[pl.Series(c, mean_widened[:, i]).cast(pl.Float32) for i, c in enumerate(levels)]
    )
    references = s397.references(series=series)
    frames = {"A": adapter.forecast_frame(stored=stored, series=series, unit="fraction")}
    for name in ("bg2", "manual_heuristic", "climatology"):
        frames[name] = references[name]
    table = metrics.shared_slice_table(
        forecasts=frames,
        actuals=ctx["actuals"],
        context=ctx["context"],
        events=ctx["events"],
        first_readings=ctx["first_readings"],
    )
    thresholds = ctx["thresholds"]
    kept = metrics.restrict_to_slice_table(forecast=frames["A"], slice_table=table)
    rows0 = metrics.scored_rows(
        forecast=kept,
        actuals=ctx["actuals"],
        series=series,
        thresholds=thresholds,
        slice_table=table,
    )
    print(f"scored rows: {rows0.height:,} of {key_frame.height:,} stored")
    joined = rows0.select(keys).join(
        key_frame.select(*keys, "pos", "month"), on=keys, how="left", validate="1:1"
    )
    assert joined["pos"].null_count() == 0
    pos = joined["pos"].to_numpy()
    q_points = (q_all[:, pos, :] * 100.0).astype(np.float32)  # points of effective capacity
    demand_examples(q_all=q_all, key_frame=key_frame, series=series, ctx=ctx)
    del q_all
    actual = rows0["actual_points"].to_numpy().astype(np.float64)

    label_names = sorted({lab for members in s407.SUBSETS.values() for lab in members})
    label_id = {lab: i for i, lab in enumerate(label_names)}
    label = np.array([label_id[v] for v in rows0["label"].to_list()])
    month_index = {m: i for i, m in enumerate(MONTHS_DEMAND)}
    month = np.array([month_index[v] for v in joined["month"].to_list()])
    lead = rows0["lead_hours"].to_numpy()
    head = rows0["lead_days_1_10"].to_numpy().astype(bool)
    masks = {
        "days 1-10": head,
        "cold nights": head & rows0["cold_nights"].to_numpy().astype(bool),
        "12-36 h": (lead >= 12.0) & (lead < 36.0),
    }
    if "event_rows" in rows0.columns:
        event = rows0["event_rows"].to_numpy().astype(bool) | rows0[
            "event_following_7_days"
        ].to_numpy().astype(bool)
        masks["no labelled event"] = head & ~event
    else:
        print("no event slice columns in the slice table: 'rows with no labelled event' not scored")
    groups = {name: [label_id[lab] for lab in members] for name, members in s407.SUBSETS.items()}
    agg = Aggregator(
        label=label,
        month=month,
        n_months=len(MONTHS_DEMAND),
        groups=groups,
        masks=masks,
        head_masks=("days 1-10",),
    )
    weights = np.full(11, 2.0 / 11.0)

    def row_arrays(q: np.ndarray) -> dict[str, np.ndarray]:
        return pinball_arrays(q=q, actual=actual, weights=weights)

    fam = Family(
        "demand",
        q_points,
        row_arrays,
        agg,
        ("signed-MW six", "days 1-10"),
        1.15,
        list(MONTHS_DEMAND),
        DEMAND_KS,
    )

    # Gate: the ten-seed mean reproduces metrics.scored_rows.
    stored_crps = rows0["crps"].to_numpy()
    mine = row_arrays(widen(q=q_points.mean(axis=0, dtype=np.float64), factor=1.15))["crps"]
    gate_gap = float(np.max(np.abs(mine - stored_crps)))
    print(
        f"gate: max |fast CRPS - metrics.scored_rows CRPS| = {gate_gap:.2e} points (K=10, widening 1.15)"
    )
    assert gate_gap < 5e-3, gate_gap

    mde, v_first, v_second = minimum_detectable_effect(fam=fam)
    lines = [
        f"gate: max |fast CRPS - metrics.scored_rows CRPS| {gate_gap:.2e} points",
        f"scored rows {rows0.height:,} (shared with arm A, bg2, manual_heuristic, climatology); months {list(MONTHS_DEMAND)}",
        f"MDE (signed-MW six, days 1-10, widening 1.15): month values of seeds 0-2 mean {np.round(v_first, 4).tolist()}, "
        f"seeds 3-4 mean {np.round(v_second, 4).tolist()}; SD of the difference {mde * math.sqrt(2):.4f}; MDE {mde:.4f} points",
    ]
    for name in ("MVA-only eight",):
        combo = (name, "days 1-10")
        alt = Family(
            "demand", q_points, row_arrays, agg, combo, 1.15, list(MONTHS_DEMAND), DEMAND_KS
        )
        lines.append(
            f"MDE for the {name} (not used): {minimum_detectable_effect(fam=alt)[0]:.4f} points"
        )

    draws, row_mean = run_curves(fam=fam)
    report_lines, summary = curve_report(
        fam=fam, draws=draws, row_mean=row_mean, mde=mde, unit="points of effective capacity"
    )
    lines += report_lines

    # Avoidable curtailment cost at the signed-MW six.
    curtailment_lines = demand_curtailment(
        fam=fam,
        rows0=rows0,
        ctx=ctx,
        series=series,
        table=table,
        frames=frames,
        metrics=metrics,
        s377=s377,
    )
    lines += curtailment_lines
    summary["mde"] = mde
    save_family(fam=fam, draws=draws, lines=lines, summary=summary)


EXAMPLE_PRIMARY: Final[str] = "D10"
EXAMPLE_ORIGINS: Final[tuple[str, ...]] = ("2026-01-12", "2026-06-15")


def demand_examples(
    *, q_all: np.ndarray, key_frame: pl.DataFrame, series: pl.DataFrame, ctx: dict
) -> None:
    """Write the example forecasts of one Primary: K = 1 (seed 0) and K = 10, widened 1.15, in MW."""
    sid = int(series.filter(pl.col("label") == EXAMPLE_PRIMARY)["time_series_id"][0])
    capacity = float(series.filter(pl.col("label") == EXAMPLE_PRIMARY)["capacity_mw"][0])
    actuals = ctx["actuals"].filter(pl.col("time_series_id") == sid)
    pieces = []
    for origin in EXAMPLE_ORIGINS:
        picked = key_frame.filter(
            (pl.col("time_series_id") == sid)
            & (pl.col("power_fcst_init_time").dt.strftime("%Y-%m-%d") == origin)
        )
        assert picked.height > 0, f"no stored rows for {EXAMPLE_PRIMARY} at {origin}"
        positions = picked["pos"].to_numpy()
        init = picked["power_fcst_init_time"][0]
        for label, seeds in (("K=1", [0]), ("K=10", list(range(10)))):
            q = (
                widen(q=q_all[seeds][:, positions, :].mean(axis=0, dtype=np.float64), factor=1.15)
                * capacity
            )
            frame = (
                picked.select("valid_time", "power_fcst_init_time")
                .with_columns(
                    (pl.col("valid_time") - pl.col("power_fcst_init_time"))
                    .dt.total_minutes()
                    .truediv(60.0)
                    .alias("lead_hours"),
                    pl.lit(label).alias("k"),
                    pl.lit(origin).alias("origin"),
                    *[
                        pl.Series(name, q[:, i])
                        for name, i in (("q05", 1), ("q25", 3), ("q50", 5), ("q75", 7), ("q95", 9))
                    ],
                )
                .join(actuals.select("valid_time", "actual"), on="valid_time", how="left")
            )
            pieces.append(frame)
        print(f"example {EXAMPLE_PRIMARY} {origin}: init {init}, {picked.height} rows")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(pieces).write_parquet(TABLE_DIR / "demand_examples.parquet")


def demand_curtailment(
    *,
    fam: Family,
    rows0: pl.DataFrame,
    ctx: dict,
    series: pl.DataFrame,
    table: pl.DataFrame,
    frames: dict,
    metrics: Any,
    s377: Any,
) -> list[str]:
    """Return the avoidable curtailment cost (pounds) at the signed-MW six for ten subsets per K."""
    from e00407_score import SUBSETS  # ty: ignore[unresolved-import]

    signed = list(SUBSETS["signed-MW six"])
    lead = rows0["lead_hours"].to_numpy()
    wanted = np.flatnonzero(
        rows0["label"].is_in(signed).to_numpy() & (lead >= 12.0) & (lead < 36.0)
    )
    base = rows0[wanted].select(
        "time_series_id", "power_fcst_init_time", "valid_time", "capacity_mw"
    )
    actual_mw = rows0[wanted]["actual_points"].to_numpy() / 100.0 * base["capacity_mw"].to_numpy()
    base = base.with_columns(pl.Series("actual", actual_mw))
    level_names = list(metrics.QUANTILE_COLUMNS)
    flows = ctx["actuals"].rename({"actual": "flow_mw"})
    rng = np.random.default_rng(SEED_BASE + 2)
    pool = fam.pool
    arms: dict[str, pl.DataFrame] = {}
    arm_k: dict[str, int] = {}
    for k in fam.ks:
        every = seed_subsets(pool=pool, k=k, rng=rng)
        picked = rng.choice(len(every), size=min(10, len(every)), replace=False)
        for i, subset in enumerate(every[int(j)] for j in sorted(picked)):
            mean_q = widen(q=fam.mean_q(subset)[wanted], factor=1.15)
            mw = mean_q / 100.0 * base["capacity_mw"].to_numpy()[:, None]
            name = f"K{k}_{i}"
            arms[name] = base.drop("capacity_mw").with_columns(
                *[pl.Series(c, mw[:, j]) for j, c in enumerate(level_names)]
            )
            arm_k[name] = k
    heuristic = metrics.restrict_to_slice_table(
        forecast=frames["manual_heuristic"], slice_table=table
    ).join(ctx["actuals"], on=["time_series_id", "valid_time"])
    arms["manual_heuristic"] = heuristic.filter(
        pl.col("time_series_id").is_in(base["time_series_id"].unique())
    )
    lines = [
        "\n== avoidable curtailment cost, pounds over the five months, signed-MW six, 12 to 36 hour leads (10 subsets per K; K=10 has one) =="
    ]
    out_rows = []
    for fraction in (0.9, 0.75, 0.5):
        cost = metrics.avoidable_curtailment_cost(
            forecasts=arms,
            series=series,
            flows=flows,
            scored_start=s377.SCORED_FIRST,
            fraction=fraction,
        )
        total = cost.group_by("arm").agg(pl.col("avoidable_cost_gbp").sum().alias("gbp")).to_dicts()
        by_arm = {r["arm"]: r["gbp"] for r in total}
        lines.append(f"limit fraction {fraction}:")
        for k in fam.ks:
            values = np.array([by_arm[a] for a, kk in arm_k.items() if kk == k], dtype=np.float64)
            lines.append(
                f"  K={k:2d}: mean {values.mean():,.0f} (5th to 95th percentile {np.percentile(values, 5):,.0f} to {np.percentile(values, 95):,.0f}; {len(values)} subsets)"
            )
            out_rows.append(
                {
                    "fraction": fraction,
                    "k": k,
                    "mean_gbp": float(values.mean()),
                    "p05": float(np.percentile(values, 5)),
                    "p95": float(np.percentile(values, 95)),
                    "n": len(values),
                }
            )
        lines.append(f"  manual_heuristic (reference): {by_arm['manual_heuristic']:,.0f}")
        out_rows.append(
            {
                "fraction": fraction,
                "k": 0,
                "mean_gbp": float(by_arm["manual_heuristic"]),
                "p05": float("nan"),
                "p95": float("nan"),
                "n": 1,
            }
        )
    pl.DataFrame(out_rows).write_parquet(TABLE_DIR / "demand_curtailment.parquet")
    return lines


# ---------------------------------------------------------------------------------------------
# Wind: experiment 429's member_plain
# ---------------------------------------------------------------------------------------------

FARM_LETTERS: Final[dict[str, str]] = {"W1": "A", "W2": "B", "W3": "C"}


def run_wind() -> None:
    """Score `member_plain` over K = 1, 2, 3, 5, and 10 at the kept rows of true days 1 to 10."""
    import e00303_score as score303  # ty: ignore[unresolved-import]
    import e00326_score as score326  # ty: ignore[unresolved-import]
    import e00347_s4_data as data347  # ty: ignore[unresolved-import]
    import e00429_data as data  # ty: ignore[unresolved-import]

    cases = data347.scored_frame()
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    pool = []
    for seed in range(10):
        folder = data.DATA_DIR / "parts" if seed < 5 else OWN_DIR / "wind" / "parts"
        pool.append(np.load(folder / f"member_plain_h{seed}.npy"))
    q = np.stack(pool).astype(np.float32)
    print(f"wind pool: {q.shape}")
    months = sorted(cases["month"].unique().to_list())
    month = np.array([months.index(m) for m in cases["month"].to_list()])
    kept = ~cases["masked"].to_numpy()
    lead = cases["lead_hours"].to_numpy()
    sites = cases["site"].to_numpy()
    true_1_10 = kept & (lead >= 24.0) & (lead <= 240.0)
    masks = {
        "true days 1-10": true_1_10,
        "true days 3-10": kept & (lead >= 72.0) & (lead <= 240.0),
        "12-36 h": kept & (lead >= 12.0) & (lead < 36.0),
        "kept": kept,
    }
    for farm, letter in FARM_LETTERS.items():
        masks[f"farm {letter} days 1-10"] = true_1_10 & (sites == farm)
    head = tuple(n for n in masks if n == "true days 1-10" or n.startswith("farm"))
    agg = Aggregator(
        label=np.zeros(len(actual), dtype=int),
        month=month,
        n_months=len(months),
        groups={"all farms": [0]},
        masks=masks,
        head_masks=head,
    )

    def row_arrays(quantiles: np.ndarray) -> dict[str, np.ndarray]:
        batch = quantiles[None]
        base = score303.row_metrics(quantiles=batch, actual=actual)
        extra = score326.extra_metrics(quantiles=batch, actual=actual)
        return {
            "crps": base["qs"][0],
            "pin_q01": base["p01"][0],
            "pin_q05": extra["p05"][0],
            "pin_q95": extra["p95"][0],
            "pin_q99": base["p99"][0],
            "cov_q01": extra["below_q01"][0],
            "cov_q05": extra["below_q05"][0],
            "cov_q95": 1.0 - extra["above_q95"][0],
            "cov_q99": 1.0 - extra["above_q99"][0],
        }

    fam = Family(
        "wind",
        q,
        row_arrays,
        agg,
        ("all farms", "true days 1-10"),
        1.0,
        months,
        WIND_KS,
        bounds=(0.0, 1.0),
    )
    mde, v_first, v_second = minimum_detectable_effect(fam=fam)
    per_seed = [
        float(
            agg.score(combo=fam.primary_combo, values=row_arrays(q[s].astype(np.float64))["crps"])
        )
        for s in range(10)
    ]
    pooled_gate = float(
        np.mean([row_arrays(q[s].astype(np.float64))["crps"][true_1_10].mean() for s in range(5)])
    )
    stored_line = ""
    report_path = data.DATA_DIR / "report.txt"
    if report_path.exists():
        for text in report_path.read_text().splitlines():
            if text.startswith("member_plain "):
                stored_line = text
                break
    lines = [
        f"scored rows {len(actual):,}; kept {int(kept.sum()):,}; true days 1-10 {int(true_1_10.sum()):,}; months {months}",
        "farms are named by letter (A, B, C), not by the identifiers of the experiments that stored them",
        f"gate: seeds 0-4 pooled over rows (experiment 429's definition) at true days 1-10: {pooled_gate:.3f} "
        "(experiment 429 stored 10.533); the scores below weight the 12 months equally, so they sit lower",
        f"per-seed score at true days 1-10, seeds 0-9: {np.round(per_seed, 4).tolist()}; "
        f"experiment 429's stored row (kept, true_1_10 first columns, mean over seeds 0-4): {stored_line[:90]!r}",
        f"MDE (all farms pooled, true days 1-10, 12 monthly blocks, widening 1.0): month values of seeds 0-2 mean {np.round(v_first, 3).tolist()}, "
        f"seeds 3-4 mean {np.round(v_second, 3).tolist()}; SD of the difference {mde * math.sqrt(2):.4f}; MDE {mde:.4f} points of capacity "
        f"(blocks are the 12 calendar months of the 53 Tuesday runs; farms are pooled within a month)",
    ]
    draws, row_mean = run_curves(fam=fam)
    report_lines, summary = curve_report(
        fam=fam, draws=draws, row_mean=row_mean, mde=mde, unit="points of capacity"
    )
    lines += report_lines
    lines.append("\n-- tails per farm at widening 1.0 (true days 1-10) --")
    curve = summarise_curves(draws=draws)
    for letter in FARM_LETTERS.values():
        for k in WIND_KS:
            sel = curve.filter(
                (pl.col("k") == k)
                & (pl.col("widening") == 1.0)
                & (pl.col("group") == "all farms")
                & (pl.col("mask") == f"farm {letter} days 1-10")
            )
            values = {r["metric"]: r["mean"] for r in sel.iter_rows(named=True)}
            lines.append(
                f"farm {letter} K={k:2d}: crps {values['crps']:.4f}; coverage "
                + ", ".join(f"{t} {values['cov_' + t]:.4f}" for t in TAIL_INDEX)
                + "; pinball "
                + ", ".join(f"{t} {values['pin_' + t]:.4f}" for t in TAIL_INDEX)
            )
    lines.append(
        f"margin agreed for wind: 0.10 points; gain K=5 to K=10 is {next(g['gain'] for g in summary['gains'] if g['k'] == 5):+.4f}"
    )
    save_family(fam=fam, draws=draws, lines=lines, summary=summary)


# ---------------------------------------------------------------------------------------------
# S1: experiment 404's lagged network as experiment 454 scores it
# ---------------------------------------------------------------------------------------------


def run_s1() -> None:
    """Score S1 over K = 1, 2, 3, and 6, by quantile averaging and by density mixture."""
    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00404_models as models  # ty: ignore[unresolved-import]
    import e00454_build as build  # ty: ignore[unresolved-import]
    import e00454_score as scoring  # ty: ignore[unresolved-import]
    import e00481_s1_heads as new_heads  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    def mix_equal(*, heads: list) -> np.ndarray:
        """Return the sorted 11 quantiles of the equal-weight mixture of heads, solved on the GPU.

        The same bisection as `e00454_build.mix_equal`, in batches on the GPU, for speed.
        """
        k = len(heads)
        log_w = torch.cat([h[0] - float(np.log(k)) for h in heads], dim=1)
        mean = torch.cat([h[1] for h in heads], dim=1)
        std = torch.cat([h[2] for h in heads], dim=1)
        out = []
        for start in range(0, len(log_w), GPU_BATCH):
            sl = slice(start, start + GPU_BATCH)
            out.append(
                models.mixture_quantiles(
                    log_w=log_w[sl].cuda(), mean=mean[sl].cuda(), std=std[sl].cuda()
                ).cpu()
            )
        return torch.sort(torch.cat(out), dim=1).values.numpy().astype(np.float64)

    months = list(build.MONTHS)
    n_seeds = 6
    base_frames, head_sets, q_parts = [], [], []
    for month in months:
        base = scoring.month_base(month=month)
        heads = []
        for seed in range(n_seeds):
            path = (
                new_heads.new_head_path(month=month)
                if seed == 5
                else build.head_path(kind="S", month=month, seed=seed)
            )
            saved = torch.load(path, weights_only=False)
            heads.append(saved["head"])
            if seed == 0:
                keys = saved["keys"]
            else:
                for k in build.KEY_COLUMNS:
                    assert np.array_equal(keys[k], saved["keys"][k]), (
                        f"{month} seed {seed}: {k} differs"
                    )
        key_frame = pl.DataFrame({k: keys[k] for k in build.KEY_COLUMNS}).with_row_index("pos")
        joined = base.select(*build.KEY_COLUMNS).join(
            key_frame, on=list(build.KEY_COLUMNS), how="left", validate="1:1"
        )
        assert joined["pos"].null_count() == 0, f"{month}: scored rows missing from the head keys"
        pos = joined["pos"].to_numpy()
        base_frames.append(base)
        head_sets.append((heads, pos))
        q_parts.append(np.stack([mix_equal(heads=[h])[pos] for h in heads]))
        print(f"s1 {month}: {len(base)} scored rows", flush=True)
    q = (np.concatenate(q_parts, axis=1) * 100.0).astype(np.float32)
    all_base = pl.concat(base_frames.copy())
    month_of = np.concatenate([np.full(len(b), i) for i, b in enumerate(base_frames)])
    actual = all_base["actual"].to_numpy().astype(np.float64) * 100.0
    group_id = {"signed_six": 0, "mva_eight": 1, "report_only": 2}
    label = np.array([group_id[g] for g in all_base["group"].to_list()])
    cls = all_base["cls"].to_numpy()
    lead = all_base["lead_hours"].to_numpy()
    masks = {
        "clean": cls == "clean",
        "harm": cls == "harm",
        "event": cls == "event",
        "clean 12-36 h": (cls == "clean") & (lead >= 12.0) & (lead < 36.0),
    }
    agg = Aggregator(
        label=label, month=month_of, n_months=len(months),
        groups={"signed_six": [0], "mva_eight": [1]}, masks=masks, head_masks=("clean",),
    )  # fmt: skip
    weights = 2.0 * np.asarray(common.ATOM_MASSES, dtype=np.float64)

    def row_arrays(quantiles: np.ndarray) -> dict[str, np.ndarray]:
        return pinball_arrays(q=quantiles, actual=actual, weights=weights)

    fam = Family("s1", q, row_arrays, agg, ("signed_six", "clean"), 1.0, months, S1_KS)

    # Gate against experiment 454's stored cells.
    cells = pl.read_parquet(scoring.SCORE_DIR / "cells.parquet")

    def stored(
        arm: str,
        *,
        variant: str = "raw",
        group: str = "signed_six",
        cls_name: str = "clean",
        cell: int | None = None,
    ) -> float:
        part = cells.filter(
            (pl.col("arm") == arm) & (pl.col("variant") == variant) & (pl.col("basis") == "metered")
            & (pl.col("group") == group) & (pl.col("cls") == cls_name)
        )  # fmt: skip
        if cell is not None:
            part = part.filter(pl.col("cell") == cell)
        return float(part.group_by("month").agg(pl.col("crps").mean())["crps"].mean())  # ty: ignore[invalid-argument-type]

    mine_seed0 = agg.score(
        combo=fam.primary_combo, values=row_arrays(q[0].astype(np.float64))["crps"]
    )
    lines = [
        f"scored rows {len(actual):,} (leads 0-240 h, experiment 454's rows); classes: "
        + ", ".join(
            f"{n} {int(m.sum()):,}" for n, m in masks.items() if n in ("clean", "harm", "event")
        ),
        f"gate: S1 seed 0, signed_six clean raw: here {mine_seed0:.4f}; experiment 454 stored cell 0 {stored('S1', cell=0):.4f}",
    ]
    mde, v_first, v_second = minimum_detectable_effect(fam=fam)
    lines.append(
        f"MDE (signed_six, clean rows, leads 0-240 h, raw, 5 monthly folds): month values of seeds 0-2 mean {np.round(v_first, 4).tolist()}, "
        f"seeds 3-4 mean {np.round(v_second, 4).tolist()}; SD of the difference {mde * math.sqrt(2):.4f}; MDE {mde:.4f} points"
    )
    draws, row_mean = run_curves(fam=fam)
    report_lines, summary = curve_report(
        fam=fam,
        draws=draws,
        row_mean=row_mean,
        mde=mde,
        unit="points of capacity, experiment 454's quantile score",
    )
    lines += report_lines
    summary["note"] = (
        "S1 seeds 3 to 5 share training worlds with seed % 3 (experiment 454 caveat): curves past K = 3 are a lower bound on diversity"
    )

    # Density mixture of the same seeds (experiment 454's S1-ens construction), as a second reading.
    def mixture(subset: tuple[int, ...]) -> np.ndarray:
        out = []
        for heads, pos in head_sets:
            out.append(mix_equal(heads=[heads[s] for s in subset])[pos] * 100.0)
        return np.concatenate(out)

    mix_fam = Family(
        "s1mix", q, row_arrays, agg, ("signed_six", "clean"), 1.0, months, S1_KS, forecast=mixture
    )
    gate_mix = agg.score(combo=fam.primary_combo, values=row_arrays(mixture((0, 1, 2)))["crps"])
    lines.append(
        f"\ngate (mixture): S1 density mixture of seeds 0-2 {gate_mix:.4f}; experiment 454 stored S1-ens3 {stored('S1-ens3', cell=0):.4f}"
    )
    mix_draws, mix_row_mean = run_curves(fam=mix_fam)
    mix_lines, mix_summary = curve_report(
        fam=mix_fam,
        draws=mix_draws,
        row_mean=mix_row_mean,
        mde=mde,
        unit="points of capacity, density mixture of the K networks",
    )
    mix_summary["note"] = "second reading; not one of the Bonferroni contrasts"
    lines += [
        "\n######## second reading: equal-weight density mixture of the K seeds (not part of the decision rule) ########",
        *mix_lines,
    ]
    save_family(fam=fam, draws=draws, lines=lines, summary=summary)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summarise_curves(draws=mix_draws).write_parquet(TABLE_DIR / "s1mix_curve.parquet")
    (PART_DIR / "s1mix.json").write_text(json.dumps(mix_summary, indent=1, default=float))

    s1_comparison()


def s1_comparison() -> None:
    """Write the table of S1 against experiment 454's stored constructs, from stored tables.

    Experiment 454 stores pinball loss as a fraction of capacity, so its `pin_q99` is multiplied by
    100 to be in points like every score here.
    """
    cells = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00454-lag-trust-mixture-and-ensemble-constructs-for-lagged-and-lagless-heads"
        / "work"
        / "scores"
        / "cells.parquet"
    )
    ours = pl.read_parquet(TABLE_DIR / "s1_curve.parquet")
    ours_mix = pl.read_parquet(TABLE_DIR / "s1mix_curve.parquet")
    networks = {
        "A0-inj": 1,
        "S1": 1,
        "LL": 1,
        "B-mix": 2,
        "S1-mix": 2,
        "S1-ens2": 2,
        "S1-ens3": 3,
        "S1-ens5": 5,
        "K2-S1": 4,
        "K3-S1": 6,
        "K5-S1": 10,
    }
    rows = []
    for arm, n_net in networks.items():
        for variant in ("raw", "w1.15"):
            part = cells.filter(
                (pl.col("arm") == arm) & (pl.col("variant") == variant) & (pl.col("basis") == "metered")
                & (pl.col("group") == "signed_six") & (pl.col("cls") == "clean")
            )  # fmt: skip
            if part.is_empty():
                continue
            monthly = part.group_by("month").agg(
                pl.col(c).mean()
                for c in ("crps", "cov_q01", "cov_q05", "cov_q95", "cov_q99", "pin_q99")
            )
            rows.append(
                {
                    "source": "experiment 454 stored",
                    "arm": arm,
                    "variant": variant,
                    "networks": n_net,
                }
                | {
                    c: float(monthly[c].mean()) * (100.0 if c == "pin_q99" else 1.0)  # ty: ignore[invalid-argument-type]
                    for c in ("crps", "cov_q01", "cov_q05", "cov_q95", "cov_q99", "pin_q99")
                }
            )
    for source, frame in (
        ("this experiment, quantile average", ours),
        ("this experiment, density mixture", ours_mix),
    ):
        for k in S1_KS:
            for w in (1.0, 1.15):
                sel = frame.filter(
                    (pl.col("k") == k)
                    & (pl.col("widening") == w)
                    & (pl.col("group") == "signed_six")
                    & (pl.col("mask") == "clean")
                )
                values = {r["metric"]: r["mean"] for r in sel.iter_rows(named=True)}
                rows.append(
                    {
                        "source": source,
                        "arm": f"S1 K={k}",
                        "variant": "raw" if w == 1.0 else "w1.15",
                        "networks": k,
                    }
                    | {
                        c: values[c]
                        for c in ("crps", "cov_q01", "cov_q05", "cov_q95", "cov_q99", "pin_q99")
                    }
                )
    comparison = pl.DataFrame(rows).sort("networks", "source", "arm", "variant")
    comparison.write_parquet(TABLE_DIR / "s1_comparison.parquet")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200, fmt_str_lengths=40):
        text = (
            "\n== S1 against experiment 454's stored constructs at their serving cost (signed_six, clean rows; mean over months; networks served per forecast pass; lagless tables count as a network) ==\n"
            + str(comparison)
        )
    (PART_DIR / "s1_comparison.txt").write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------------------------------------
# Assembled report
# ---------------------------------------------------------------------------------------------


def log_seconds(*, pattern: str, path: Path) -> list[int]:
    """Return the integer captured by `pattern` on each matching log line."""
    return [
        int(m.group(1)) for line in path.read_text().splitlines() if (m := re.search(pattern, line))
    ]


def production_cost_table() -> list[str]:
    """Return the networks served and the fits per monthly refit for each K, with timed GPU seconds."""
    logs = OWN_DIR / "logs"
    pretrain = log_seconds(pattern=r"pretrain A .* (\d+) s$", path=logs / "demand.log")
    tune = log_seconds(
        pattern=r"A \d{4}-\d{2} seed \d+: .* (\d+) s \(fine-tune", path=logs / "demand.log"
    )
    wind_cum = log_seconds(pattern=r"seed \d+ done (\d+) s", path=logs / "wind.log")
    wind = [b - a for a, b in itertools.pairwise(wind_cum)]
    s1 = log_seconds(pattern=r"S1 .* s5: [\d,]+ rows, (\d+) s", path=logs / "s1.log")
    med = lambda x: float(np.median(x)) if x else float("nan")  # noqa: E731
    lines = [
        "\n== production cost (per monthly refit of one forecast type; timed on this workstation's GPU, one process) ==",
        f"timed medians: arm A pretrain {med(pretrain):.0f} s ({len(pretrain)} fits), arm A fine-tune and forecast {med(tune):.0f} s ({len(tune)} fits), "
        f"wind member_plain {med(wind):.0f} s per seed ({len(wind)} seeds), S1 {med(s1):.0f} s per fit ({len(s1)} fits)",
        "family   K   networks per serving pass   fits per monthly refit   GPU seconds per monthly refit",
    ]
    for k in DEMAND_KS:
        lines.append(
            f"demand  {k:2d}   {k:2d}                         {2 * k:2d} ({k} pretrain + {k} fine-tune)   {k * (med(pretrain) + med(tune)):.0f}"
        )
    for k in WIND_KS:
        lines.append(
            f"wind    {k:2d}   {k:2d}                         {k:2d}                       {k * med(wind):.0f}"
        )
    for k in S1_KS:
        lines.append(
            f"S1      {k:2d}   {k:2d} (+1 lagless table)      {k:2d} (+{k} cross-fitted lagless tables, not timed)   {k * med(s1):.0f} (S1 networks only)"
        )
    return lines


def assemble_report() -> None:
    """Join the family parts into `score_report.txt`: the MDEs and the Bonferroni count first."""
    summaries = {
        name: json.loads((PART_DIR / f"{name}.json").read_text())
        for name in ("demand", "wind", "s1")
        if (PART_DIR / f"{name}.json").exists()
    }
    head = [
        "EXPERIMENT 481: seed averaging, scorer report",
        f"BONFERRONI COUNT: {BONFERRONI_TOTAL} marginal contrasts (arm A {N_CONTRASTS['demand']}, S1 {N_CONTRASTS['s1']}, wind {N_CONTRASTS['wind']}); "
        f"alpha {ALPHA:.4f} two-sided; interval level {LEVEL_PERCENT:.2f}%",
        "MINIMUM DETECTABLE EFFECT (standard deviation over folds of the difference between the mean of seeds 0-2 and the mean of seeds 3-4 of the same arm, divided by the square root of two):",
    ]
    for name, text in (("demand", "arm A, signed-MW six, days 1-10, widened 1.15, 5 monthly folds, points of effective capacity"),
                       ("wind", "member_plain, farms pooled, kept rows of true days 1-10, 12 monthly blocks, points of capacity"),
                       ("s1", "S1, signed_six, clean rows, leads 0-240 h, raw, 5 monthly folds, points (experiment 454 quantile score)")):  # fmt: skip
        if name in summaries:
            head.append(f"  {name}: {summaries[name]['mde']:.4f}  ({text})")
        else:
            head.append(f"  {name}: not scored yet")
    head.append(
        "Score per K is the mean over 100 random draws of K seeds (all subsets when fewer exist); 5th to 95th percentiles are across draws. Random generator: numpy default_rng(481), one per family."
    )
    head.append(
        "Each score is the mean over months of the month's value (mean over Primaries for demand, over rows for wind and S1). The bootstrap resamples whole months with equal weight, so it differs slightly from experiment 425's row-pooled `month_bootstrap_contrast`."
    )
    parts = ["\n".join(head)]
    for name in ("demand", "wind", "s1"):
        path = PART_DIR / f"{name}.txt"
        if path.exists():
            parts.append(f"\n################ {name} ################\n" + path.read_text())
    comp = PART_DIR / "s1_comparison.txt"
    if comp.exists():
        parts.append(comp.read_text())
    parts.append("\n".join(production_cost_table()))
    parts.append("\n".join(prediction_checks(summaries=summaries)))
    REPORT_FILE.write_text("\n".join(parts) + "\n")
    print(f"written {REPORT_FILE}")


def prediction_checks(*, summaries: dict) -> list[str]:
    """Return the pre-registered predictions P1 to P4 with the measured numbers (verdicts are for the write-up)."""
    lines = ["\n== pre-registered predictions, measured numbers =="]
    if "demand" in summaries:
        by_k = {int(k): v for k, v in summaries["demand"]["score_by_k"].items()}
        lines.append(
            f"P1 arm A, signed-MW six: K=1 to K=2 gain {by_k[1] - by_k[2]:+.4f} (predicted 0.05 to 0.12); K=3 to K=5 gain {by_k[3] - by_k[5]:+.4f} (predicted < 0.03); K=5 to K=10 gain {by_k[5] - by_k[10]:+.4f} (predicted < 0.02)"
        )
    if "wind" in summaries:
        by_k = {int(k): v for k, v in summaries["wind"]["score_by_k"].items()}
        lines.append(
            f"P3 wind member_plain: K=5 to K=10 gain {by_k[5] - by_k[10]:+.4f} points (predicted < 0.10); K=1 to K=10 gain {by_k[1] - by_k[10]:+.4f}"
        )
    lines.append(
        "P2 (best widening against K) and P4 (S1 averaging against B-mix at equal network counts) are in the widening tables and the S1 comparison table above."
    )
    return lines


def main() -> None:
    """Run one subcommand."""
    command = sys.argv[1]
    commands = {
        "demand": run_demand,
        "wind": run_wind,
        "s1": run_s1,
        "s1compare": s1_comparison,
        "report": assemble_report,
    }
    if command not in commands:
        raise SystemExit(f"unknown command {command}; choose from {list(commands)}")
    commands[command]()


if __name__ == "__main__":
    main()

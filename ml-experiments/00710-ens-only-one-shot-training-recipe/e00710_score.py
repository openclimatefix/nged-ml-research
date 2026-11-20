r"""Score experiment 710 from its stored forecasts; read experiment 707's stored `A_F` and `R_F`.

The scorer reads `seed_preds/<arm>-<state>/<month>_s<seed>.parquet` of this experiment (written by
`e00710_fit.py`) and, for `M1-n` and `M2-n`, the stored `A_F` and `R_F` forecasts of experiment 707.
It averages the seeds that both arms of a contrast hold in quantile space, widens the outer
quantiles by 1.15 about the median, and scores the result on the half-hours that every arm and
every reference holds (`e00381_score.score_arms` selects the same half-hours).
Memory stays bounded because each variant is scored one fold at a time into sums of its row losses
by slice, month, and Primary (`Scorer`), which are cached under `work/score_cache/`; every printed
figure follows from those sums.

**Metrics, per Primary group (the signed-MW six and the MVA-only eight):** CRPS and the normalised
MAE of the median; the same on sunny half-hours (`sunny_clear_sky_index` if the slice table has it,
else `sunny_unit_shape`) and on cold nights; the slump share of the forecast into injected
switching events, in total and by days since the onset; the error against the normal running
arrangement (NRA); the same figures in the 7 days after an event; the labelled-event slices of
experiment 707's scorer; the coverage of q05 and q95. Each figure prints the number of half-hours
(or events) behind it and the number of months, and a subset under 100 half-hours is marked thin.

**Contrasts** are treatment minus reference in the `i` state, read at the Bonferroni level of their
family (`e00710_arms.FAMILY_LEVELS`: R 99.17%, W 99.375%, I 99.58%, and E, the two readings of the
pre-registered extension E1, 97.5%) with a month-block bootstrap of
100,000 resamples. Everything else prints at 95%.

**The slump share** of a row with a shifted lag is `sign(d) * (q50 with shifted lags - q50 with
unshifted lags) / abs(d)`. It is averaged within each event, then over events. The unshifted
forecast is the arm's ordinary stored forecast; the shifted forecast is its bank pass
(`bank/<arm>-<state>/<month>_s<seed>_p<pass>.parquet`).

```bash
export PYTHONPATH=<this folder and the folders of experiments 377, 381, 397, 407, 581, 707>
uv run --no-sync python ml-experiments/00710-*/e00710_score.py --h-t 0.5 --b-star M2 --o-star W-h0.5
uv run --no-sync python ml-experiments/00710-*/e00710_score.py --tuning
```
"""

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base377  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00581_bank as bank581  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import e00707_score as s707  # ty: ignore[unresolved-import]
import e00710_arms as arms  # ty: ignore[unresolved-import]
import e00710_inject as inject  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy import stats
from studies.eval_protocol import metrics

from studies import bootstrap

OUT_DIR: Final[Path] = arms.WORK_DIR / "work"
STORED_707_WORK: Final[Path] = arms.EXPERIMENT_707_DIR / "work"
STORED_FOLDER: Final[dict[str, str]] = {"M1-n": "A_F", "M2-n": "R_F"}
"""The folder of experiment 707's `seed_preds` that holds a stored arm."""
KEYS: Final[list[str]] = list(common.KEYS)
SIX: Final[str] = s707.SIX
EIGHT: Final[str] = s707.EIGHT
GROUPS: Final[tuple[str, ...]] = (SIX, EIGHT)
HEADLINE: Final[str] = s707.HEADLINE
SUBSETS: Final[dict[str, tuple[str, ...]]] = s707.SUBSETS
SCALE_METRICS: Final[tuple[str, ...]] = ("crps", "mae_median")
N_RESAMPLES: Final[int] = 100_000
DESCRIPTIVE_LEVEL: Final[float] = 95.0
ONE_SIDED_LEVEL: Final[float] = 90.0
"""A two-sided 90% interval has a one-sided 95% upper bound."""
MARGIN: Final[float] = 0.02
SLICE_MARGIN: Final[float] = 0.05
MDE_ARM: Final[str] = "M2-n"
MDE_HALVES: Final[tuple[tuple[int, ...], tuple[int, ...]]] = ((0, 1, 2), (3, 4))
HOLDOUT: Final[datetime] = common.HOLDOUT
MAX_LEAD_HOURS: Final[float] = 240.0
PER_DAY: Final[int] = bank581.PER_DAY
AFTER_DAYS: Final[int] = 7
FIRST_PHASE_DAYS: Final[int] = 4
CONVERGENCE_LIMIT: Final[float] = 0.25
"""An arm whose final stage hits its limit in more than this share of fits is `not converged`."""
VERDICT_ORDER: Final[tuple[str, ...]] = ("refuted", "not supported", "inconclusive", "supported")
"""From the worst verdict to the best."""
QUANTILES: Final[tuple[str, ...]] = tuple(common.LEVEL_COLUMNS)
ABSOLUTE_RESAMPLES: Final[int] = 20_000
"""Month-block resamples behind each interval on an absolute metric."""
ABSOLUTE_SEED: Final[int] = 710
STOP_POINTS: Final[float] = 0.10
"""The experiment stops if `M2-i` scores more than this many points worse than `M1-i` or `M2-n`."""
PP_ARMS: Final[dict[str, float | str]] = {
    "W-h0.5": 0.5,
    "W-h1": 1.0,
    "W-h2": 2.0,
    "M3": "inf",
}
"""The arms whose tuning fits give `PP` each Primary's half-life in years (`M3` is infinite)."""
SUNNY_KEY: Final[str] = "one-sided 95% bound within 0.05 on sunny_clear_sky_index"
"""The one guard that may be absent: the slice table has no clear-sky-index column until built."""
SCORE_CACHE_DIR: Final[Path] = OUT_DIR / "score_cache"
"""One parquet file of scored aggregates per variant (`Scorer.aggregate`)."""
AGGREGATE_VERSION: Final[str] = "1"
"""Change this when the columns or the meaning of the aggregates change, to invalidate the cache."""
ALL_ROWS: Final[str] = "all rows"
"""The name of the slice that holds every headline-band half-hour."""
SUMMED_METRICS: Final[tuple[str, ...]] = (
    "crps",
    "mae_median",
    "pinball_q01",
    "pinball_q99",
    "coverage_q05",
    "coverage_q95",
)
"""The row losses whose sums the aggregates hold; every printed figure reads one of them."""


@dataclass(frozen=True)
class Variant:
    """One scored forecast: an arm in a state, averaged over a set of seeds."""

    arm: str
    state: str
    seeds: tuple[int, ...]
    months: tuple[str, ...] = ()
    """The folds to read; empty means every fold that the run scores."""

    @property
    def base(self) -> str:
        """Return the arm and state, such as `W-h0.5-i`."""
        return f"{self.arm}-{self.state}"

    @property
    def name(self) -> str:
        """Return the base name with the seeds appended, such as `M1-i@012`."""
        return f"{self.base}@{''.join(str(s) for s in self.seeds)}"


def forecast_folder(*, arm: str, state: str) -> Path:
    """Return the folder of an arm's per-seed forecasts: experiment 707's for a stored arm."""
    stored = STORED_FOLDER.get(f"{arm}-{state}")
    if stored is not None:
        return STORED_707_WORK / "seed_preds" / stored
    return OUT_DIR / "seed_preds" / f"{arm}-{state}"


def available_seeds(*, arm: str, state: str, months: tuple[str, ...]) -> tuple[int, ...]:
    """Return the seeds whose forecast file exists for every month.

    Args:
        arm: The arm.
        state: `i` or `n`.
        months: The folds.

    Returns:
        The complete seeds, in order.
    """
    folder = forecast_folder(arm=arm, state=state)
    return tuple(
        s for s in range(5) if all((folder / f"{m}_s{s}.parquet").exists() for m in months)
    )


def average_files(*, paths: list[Path]) -> pl.DataFrame:
    """Return the seed average of quantile files in quantile space, sorted by key.

    Args:
        paths: One file per seed; every file holds the same keys.

    Returns:
        One row per key with the 11 quantiles, each averaged over the files and re-sorted.

    Raises:
        ValueError: If a file differs in height or keys from the first.
    """
    frames = [pl.read_parquet(p) for p in paths]
    first = frames[0].select(KEYS).sort(KEYS)
    for frame in frames:
        if frame.height != frames[0].height or not frame.select(KEYS).sort(KEYS).equals(first):
            msg = f"{paths[0].parent.name}: a seed differs in height or keys from the first"
            raise ValueError(msg)
    frame = pl.concat([f.select(*KEYS, *QUANTILES) for f in frames])
    frame = frame.group_by(KEYS).agg(*[pl.col(c).mean() for c in QUANTILES]).sort(KEYS)
    values = np.sort(frame.select(QUANTILES).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(QUANTILES)]
    )


def seed_average(*, variant: Variant, months: tuple[str, ...]) -> pl.DataFrame:
    """Return a variant's forecast (fraction of capacity) averaged over its seeds, all months.

    Args:
        variant: The arm, state, and seeds.
        months: The folds to read.

    Returns:
        The seed-averaged forecast of every fold.
    """
    folder = forecast_folder(arm=variant.arm, state=variant.state)
    pieces = [
        average_files(paths=[folder / f"{m}_s{s}.parquet" for s in variant.seeds]) for m in months
    ]
    return pl.concat(pieces).sort(KEYS)


def variant_files(*, variant: Variant, months: tuple[str, ...]) -> list[Path]:
    """Return the forecast files that a variant reads: one per seed and fold."""
    folder = forecast_folder(arm=variant.arm, state=variant.state)
    return [folder / f"{m}_s{s}.parquet" for m in variant.months or months for s in variant.seeds]


def files_digest(*, paths: list[Path]) -> str:
    """Return the SHA-256 of the folder, name, size, and modification time of each file."""
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(
            f"{path.parent.name}/{path.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def frame_digest(*, frame: pl.DataFrame) -> str:
    """Return a SHA-256 of a frame's schema, height, and the sum of its row hashes."""
    hashes = frame.hash_rows().to_numpy()
    total = int(hashes.sum(dtype=np.uint64)) if len(hashes) else 0
    return hashlib.sha256(f"{frame.schema}|{frame.height}|{total}".encode()).hexdigest()


def scored_keys(*, keys: pl.DataFrame, ctx: dict) -> pl.DataFrame:
    """Return the keys of the half-hours that `metrics.row_losses` scores."""
    return metrics._scored_keys(
        forecast=keys, actuals=ctx["actuals"], holdout_start=metrics.HOLDOUT_START
    )


def variant_scored_keys(
    *, variant: Variant, months: tuple[str, ...], series: pl.DataFrame, ctx: dict
) -> pl.DataFrame:
    """Return the keys of the half-hours that a variant's forecast has and that are scored.

    Only the key columns of the first seed's file are read. Every seed holds the same keys
    (`average_files` checks that when the variant is scored).

    Args:
        variant: The arm, state, and seeds.
        months: The folds, for a variant that names none of its own.
        series: The series table.
        ctx: The scoring context.

    Returns:
        The scored keys, sorted by key.

    Raises:
        ValueError: If a key is held twice.
    """
    folder = forecast_folder(arm=variant.arm, state=variant.state)
    pieces = [
        pl.read_parquet(folder / f"{m}_s{variant.seeds[0]}.parquet", columns=KEYS).with_columns(
            pl.col("time_series_id").cast(series.schema["time_series_id"])
        )
        for m in variant.months or months
    ]
    keys = (
        pl.concat(pieces)
        .join(series.select("time_series_id"), on="time_series_id", how="semi")
        .sort(KEYS)
    )
    if keys.is_duplicated().any():
        msg = f"{variant.name}: the stored forecast holds a key twice"
        raise ValueError(msg)
    return scored_keys(keys=keys, ctx=ctx)


def reference_scored_keys(*, series: pl.DataFrame, ctx: dict) -> dict[str, pl.DataFrame]:
    """Return the scored keys of each reference, holding one reference forecast at a time.

    Only the keys of the references matter: they enter the shared half-hours and nothing else. The
    references are `S0M_w115` (experiment 389's monthly XGBoost forecast, whose widening changes
    no key) and the references of experiment 707's scorer.

    Args:
        series: The series table.
        ctx: The scoring context.

    Returns:
        The scored keys by reference name: `S0M_w115`, `bg2`, `manual_heuristic`, `climatology`.
    """
    stored = pl.read_parquet(s397.E389_ARMS / "S0M_raw.parquet")
    frame = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    del stored
    out = {"S0M_w115": scored_keys(keys=frame.select(KEYS), ctx=ctx)}
    del frame
    references = base377.reference_frames(series=series)
    for name in s707.REFERENCE_NAMES:
        out[name] = scored_keys(keys=references.pop(name).select(KEYS), ctx=ctx)
    del references
    gc.collect()
    return out


class Scorer:
    """The shared slice table and the scored aggregates of the variants, computed on demand.

    The aggregates of a variant are the sums of its row losses and the half-hour counts, by slice,
    fold, and Primary, over the headline-band half-hours. Every printed figure follows from the
    sums, so no frame of row losses is held for more than one fold of one variant. The aggregates
    are written to `SCORE_CACHE_DIR` under the variant's name and a digest of its input files and
    of the scoring context, and a later run reads them back.
    """

    def __init__(
        self,
        *,
        series: pl.DataFrame,
        ctx: dict,
        variants: list[Variant],
        months: tuple[str, ...],
        reference_keys: dict[str, pl.DataFrame],
    ) -> None:
        """Build the shared slice table of the variants and the references.

        Args:
            series: The series table.
            ctx: The scoring context of `s707.cleaned_scoring_context`.
            variants: The variants to score.
            months: The folds, for a variant that names none of its own.
            reference_keys: The scored keys of the references (`reference_scored_keys`).
        """
        self.series = series
        self.ctx = ctx
        self.months = months
        self.variants: dict[str, Variant] = {v.name: v for v in variants}
        self.aggregates: dict[str, pl.DataFrame] = {}
        shared: pl.DataFrame | None = None
        for variant in variants:
            keys = variant_scored_keys(variant=variant, months=months, series=series, ctx=ctx)
            shared = (
                keys
                if shared is None
                else shared.join(
                    keys, on=list(metrics.KEY_COLUMNS), how="inner", maintain_order="left"
                )
            )
        for keys in reference_keys.values():
            shared = (
                keys
                if shared is None
                else shared.join(
                    keys, on=list(metrics.KEY_COLUMNS), how="inner", maintain_order="left"
                )
            )
        if shared is None:
            msg = "at least one variant or reference is needed to build a slice table"
            raise ValueError(msg)
        print(f"scored: {', '.join([*self.variants, *reference_keys])}")
        self.table = metrics.shared_slice_table(
            forecasts={"shared": shared},
            actuals=ctx["actuals"],
            context=ctx["context"],
            events=ctx["events"],
            first_readings=ctx["first_readings"],
        )
        print(f"shared half-hours: {self.table.height}")
        del shared
        gc.collect()
        self.slices: tuple[str, ...] = tuple(
            c for c in self.table.columns if c not in metrics.KEY_COLUMNS
        )
        self.context_digest = hashlib.sha256(
            "|".join(
                [
                    AGGREGATE_VERSION,
                    frame_digest(frame=self.table),
                    frame_digest(frame=ctx["actuals"]),
                    frame_digest(frame=series),
                    frame_digest(frame=ctx["thresholds"]),
                ]
            ).encode()
        ).hexdigest()

    def cache_path(self, *, variant: Variant) -> Path:
        """Return the cache file of a variant's aggregates."""
        files = files_digest(paths=variant_files(variant=variant, months=self.months))
        return SCORE_CACHE_DIR / f"{variant.name}__{files[:16]}__{self.context_digest[:16]}.parquet"

    def get(self, name: str) -> pl.DataFrame | None:
        """Return a variant's aggregates, or `None` if the name is not one of the variants."""
        if name not in self.aggregates:
            variant = self.variants.get(name)
            if variant is None:
                return None
            path = self.cache_path(variant=variant)
            if path.exists():
                self.aggregates[name] = pl.read_parquet(path)
            else:
                self.aggregates[name] = self.aggregate(variant=variant)
                common.atomic_write(frame=self.aggregates[name], target=path)
        return self.aggregates[name]

    def require(self, name: str) -> pl.DataFrame:
        """Return a variant's aggregates, raising `KeyError` for a name that is not a variant."""
        found = self.get(name)
        if found is None:
            raise KeyError(name)
        return found

    def aggregate(self, *, variant: Variant) -> pl.DataFrame:
        """Score a variant one fold at a time and return its aggregates.

        Args:
            variant: The arm, state, and seeds.

        Returns:
            One row per slice, month of the forecast origin, and Primary, with `n` (half-hours) and
            the sum of each of `SUMMED_METRICS`, over the half-hours of the headline band.

        Raises:
            ValueError: If the scored half-hours are not exactly the keys of the slice table, or a
                summed loss is null.
        """
        pieces = []
        scored = 0
        for month in variant.months or self.months:
            stored = seed_average(variant=variant, months=(month,))
            frame = adapter.forecast_frame(
                stored=s407.widen_global(frame=stored), series=self.series, unit="fraction"
            )
            del stored
            kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=self.table)
            del frame
            losses = metrics.row_losses(
                forecast=kept,
                actuals=self.ctx["actuals"],
                series=self.series,
                thresholds=self.ctx["thresholds"],
            )
            del kept
            rows = metrics._with_slice_columns(rows=losses, slice_table=self.table)
            del losses
            scored += rows.height
            pieces.append(self.month_sums(rows=rows, variant=variant))
            del rows
            gc.collect()
        if scored != self.table.height:
            msg = (
                f"{variant.name}: {scored} scored half-hours, but the slice table holds "
                f"{self.table.height}. Score every variant on the shared rows"
            )
            raise ValueError(msg)
        return (
            pl.concat(pieces)
            .group_by("slice", "month", "label")
            .agg(pl.col("n").sum(), *[pl.col(m).sum() for m in SUMMED_METRICS])
            .sort("slice", "month", "label")
        )

    def month_sums(self, *, rows: pl.DataFrame, variant: Variant) -> pl.DataFrame:
        """Return the sums of the row losses of one fold, by slice, month, and Primary."""
        headline = rows.filter(pl.col(HEADLINE)).with_columns(
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
        )
        if headline.select(SUMMED_METRICS).null_count().sum_horizontal().item():
            msg = f"{variant.name}: a headline-band row loss is null"
            raise ValueError(msg)
        pieces = []
        for name in (ALL_ROWS, *self.slices):
            part = headline if name == ALL_ROWS else headline.filter(pl.col(name))
            pieces.append(
                part.group_by("month", "label")
                .agg(pl.len().cast(pl.Int64).alias("n"), *[pl.col(m).sum() for m in SUMMED_METRICS])
                .with_columns(slice=pl.lit(name))
            )
        return pl.concat(pieces)


def resolve(*, name: str, args: argparse.Namespace) -> str:
    """Return the arm that a contrast placeholder (`B*`, `O*`, `W-h_T`) stands for.

    Args:
        name: An arm name or a placeholder.
        args: The command-line arguments that name `B*`, `O*`, and `h_T`.

    Returns:
        The arm name.
    """
    return {"B*": args.b_star, "O*": args.o_star, "W-h_T": f"W-h{args.h_t:g}"}.get(name, name)


def needed_variants(
    *, months: tuple[str, ...], args: argparse.Namespace
) -> tuple[list[Variant], dict[tuple[str, str], tuple[int, ...]]]:
    """Return the variants to score and each arm-state's complete seeds.

    The variants are every arm-state with a complete seed, the seeds that both arms of each
    contrast hold, and the seed sets that the decision rule reads separately.

    Args:
        months: The folds.
        args: The command-line arguments.

    Returns:
        The variants and the seeds by (arm, state).
    """
    states = ("i", "n")
    names = [*arms.ARMS, "M3"]
    have = {
        (a, s): available_seeds(arm=a, state=s, months=months)
        for a in dict.fromkeys(names)
        for s in states
    }
    variants: dict[str, Variant] = {}

    def add(arm: str, state: str, seeds: tuple[int, ...]) -> None:
        if seeds and all(s in have.get((arm, state), ()) for s in seeds):
            v = Variant(arm=arm, state=state, seeds=seeds)
            variants[v.name] = v

    for (arm, state), seeds in have.items():
        add(arm, state, seeds)
    pairs = [
        (resolve(name=t, args=args), "i", resolve(name=r, args=args), "i")
        for t, r in arms.CONTRAST_NAMES.values()
    ]
    pairs += [(x, "i", x, "n") for x in ("M1", "M2", args.b_star)]
    pairs += [("M1h90", "i", "M1", "i")]
    pairs += [(args.b_star, "n", "M1", "n"), (args.b_star, "i", "M2", "i")]
    for ta, ts, ra, rs in pairs:
        shared = tuple(sorted(set(have.get((ta, ts), ())) & set(have.get((ra, rs), ()))))
        add(ta, ts, shared)
        add(ra, rs, shared)
    for arm, state in (("M1", "i"), ("M2", "i"), (args.b_star, "i"), (MDE_ARM[:2], "n")):
        for half in MDE_HALVES:
            add(arm, state, half)
    return list(variants.values()), have


def month_totals(
    *,
    aggregate: pl.DataFrame,
    group: str,
    metric: str,
    slice_name: str = ALL_ROWS,
    without_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the sum of a metric and the half-hour count in each month, on a group and a slice.

    Args:
        aggregate: The aggregates of one variant (`Scorer.aggregate`).
        group: A key of `SUBSETS`.
        metric: One of `SUMMED_METRICS`.
        slice_name: `ALL_ROWS` or a column of the slice table.
        without_label: A Primary to leave out, such as `D8`.

    Returns:
        The months (sorted), the sums, and the counts; empty arrays if no half-hour is left.
    """
    keep = (pl.col("slice") == slice_name) & pl.col("label").is_in(list(SUBSETS[group]))
    if without_label is not None:
        keep &= pl.col("label") != without_label
    totals = (
        aggregate.filter(keep)
        .group_by("month")
        .agg(pl.col(metric).sum(), pl.col("n").sum())
        .sort("month")
    )
    return (
        totals["month"].to_numpy(),
        totals[metric].to_numpy().astype(np.float64),
        totals["n"].to_numpy().astype(np.float64),
    )


def month_sum_bounds(
    *, sums: np.ndarray, counts: np.ndarray, level_percent: float, n_resamples: int
) -> tuple[float, float]:
    """Return the interval of a mean from resampling whole months, from each month's sum and count.

    The random stream and the draw order are those of `bootstrap._resample_bounds` for one seed, so
    the bounds equal what the row-level resampling returns, up to floating-point summation order.

    Args:
        sums: The sum of the per-row value in each month, months in sorted order.
        counts: The number of rows in each month.
        level_percent: The interval's coverage in percent, above 1 and below 100.
        n_resamples: How many resamples to draw.

    Returns:
        The lower and upper percentiles of the resampled mean.

    Raises:
        ValueError: If the level is not above 1 and below 100.
    """
    if not 1.0 < level_percent < 100.0:
        msg = f"level is a coverage in percent, above 1 and below 100: {level_percent}"
        raise ValueError(msg)
    generator = np.random.default_rng(bootstrap.BOOTSTRAP_SEED)
    resampled = np.empty(n_resamples)
    for resample in range(n_resamples):
        generator.integers(0, 1)
        drawn = generator.integers(0, len(counts), size=len(counts))
        resampled[resample] = sums[drawn].sum() / counts[drawn].sum()
    tail = (100.0 - level_percent) / 2.0
    return float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100.0 - tail))


def contrast(
    *,
    label: str,
    group: str,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    metric: str = "crps",
    slice_name: str = ALL_ROWS,
    without_label: str | None = None,
    level_percent: float = DESCRIPTIVE_LEVEL,
    note: str = "",
) -> dict[str, float] | None:
    """Print one contrast with the bootstrap interval and return its figures, or `None`.

    The contrast is the mean over the shared half-hours of treatment minus reference. Both
    variants hold the same half-hours (the slice table's), so the month total of the difference is
    the difference of the month totals.

    Args:
        label: The contrast's name.
        group: A key of `SUBSETS`.
        treatment: The aggregates of the treatment.
        reference: The aggregates of the reference.
        metric: One of `SUMMED_METRICS`.
        slice_name: `ALL_ROWS` or a column of the slice table, such as a slice or a lead band.
        without_label: A Primary to drop.
        level_percent: The interval's coverage in percent.
        note: Text to append to the line.

    Returns:
        The difference, the interval, the half-hours, and the months, or `None` for empty rows.

    Raises:
        ValueError: If the two variants hold different months or half-hour counts.
    """
    months, sums, counts = month_totals(
        aggregate=treatment,
        group=group,
        metric=metric,
        slice_name=slice_name,
        without_label=without_label,
    )
    ref_months, ref_sums, ref_counts = month_totals(
        aggregate=reference,
        group=group,
        metric=metric,
        slice_name=slice_name,
        without_label=without_label,
    )
    if len(months) == 0 and len(ref_months) == 0:
        msg = "the two forecasts share no half-hour in the filter"
        print(f"{label:34s} {group:16s} {metric:10s} unavailable: {msg}")
        return None
    if not (np.array_equal(months, ref_months) and np.array_equal(counts, ref_counts)):
        msg = f"{label}: the two variants hold different half-hours in the filter"
        raise ValueError(msg)
    low, high = month_sum_bounds(
        sums=sums - ref_sums, counts=counts, level_percent=level_percent, n_resamples=N_RESAMPLES
    )
    r = {
        "difference": float((sums.sum() - ref_sums.sum()) / counts.sum()),
        "lower": low,
        "upper": high,
        "level_percent": level_percent,
        "n_half_hours": int(counts.sum()),
        "n_months": len(months),
    }
    thin = " THIN" if r["n_half_hours"] < metrics.THIN_HALF_HOURS else ""
    print(
        f"{label:34s} {group:16s} {metric:10s} {r['difference']:+.3f} "
        f"({r['lower']:+.3f} to {r['upper']:+.3f}) level {r['level_percent']:.2f}% "
        f"n={r['n_half_hours']} months={r['n_months']}{thin} {note}"
    )
    return {k: float(r[k]) for k in ("difference", "lower", "upper", "n_half_hours", "n_months")}


def group_value(*, aggregate: pl.DataFrame, group: str, metric: str) -> float:
    """Return the equal-weight mean over a group's Primaries of the headline-band metric."""
    per_label = (
        aggregate.filter(
            (pl.col("slice") == ALL_ROWS) & pl.col("label").is_in(list(SUBSETS[group]))
        )
        .group_by("label")
        .agg(pl.col(metric).sum(), pl.col("n").sum())
    )
    if per_label.is_empty():
        return float("nan")
    return float(np.mean((per_label[metric] / per_label["n"]).to_numpy()))


def absolute_interval(
    *, aggregate: pl.DataFrame, group: str, metric: str, slice_name: str
) -> tuple[float, float, float, int, int]:
    """Return the mean of a metric on a slice with a 95% month-block interval.

    The mean here is over half-hours, so it can differ slightly from the equal-weight mean over
    Primaries that `group_value` returns. Each resample draws calendar months with replacement
    and recomputes the half-hour mean from the months' sums and counts, which is exact.

    Args:
        aggregate: The aggregates of one variant.
        group: A key of `SUBSETS`.
        metric: One of `SUMMED_METRICS`.
        slice_name: `ALL_ROWS` or a column of the slice table.

    Returns:
        The mean, the lower and upper limits, the number of half-hours, and the number of months;
        `nan` limits if the set is empty.
    """
    months, sums, counts = month_totals(
        aggregate=aggregate, group=group, metric=metric, slice_name=slice_name
    )
    if len(months) == 0:
        nan = float("nan")
        return nan, nan, nan, 0, 0
    rng = np.random.default_rng(ABSOLUTE_SEED)
    draws = rng.integers(0, len(months), size=(ABSOLUTE_RESAMPLES, len(months)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (100.0 - DESCRIPTIVE_LEVEL) / 2.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return (
        float(sums.sum() / counts.sum()),
        float(low),
        float(high),
        int(counts.sum()),
        len(months),
    )


def slice_names(*, slices: tuple[str, ...]) -> list[str]:
    """Return the slices that the metrics are reported on, `ALL_ROWS` first.

    The sunny slice is `sunny_clear_sky_index` if the table has it (the protocol's protected
    slice) and `sunny_unit_shape` otherwise; `sunny_unit_shape` is also printed when both exist.

    Args:
        slices: The columns of the shared slice table.

    Returns:
        The slices present, in reporting order.
    """
    out = [ALL_ROWS]
    out += [
        name
        for name in ("sunny_clear_sky_index", "sunny_unit_shape", "cold_nights", "event_rows")
        if name in slices
    ]
    if "event_following_7_days" in slices:
        out.append("event_following_7_days")
    return out


def mean_coverage(*, aggregate: pl.DataFrame, group: str, metric: str) -> float:
    """Return the mean of a coverage column over a group's headline-band half-hours."""
    _months, sums, counts = month_totals(aggregate=aggregate, group=group, metric=metric)
    return float(sums.sum() / counts.sum()) if counts.sum() else float("nan")


def metric_tables(*, aggregates: dict[str, pl.DataFrame], slices: tuple[str, ...]) -> None:
    """Print each metric, per group and slice, with its interval and the subset size.

    Args:
        aggregates: The aggregates of each variant (the full-seed variants).
        slices: The columns of the shared slice table.
    """
    print("\n== CRPS and MAE of the median per group and slice (95% month-block intervals) ==")
    for slice_name in slice_names(slices=slices):
        for group in GROUPS:
            for name, aggregate in aggregates.items():
                for metric in SCALE_METRICS:
                    mean, low, high, n, months = absolute_interval(
                        aggregate=aggregate, group=group, metric=metric, slice_name=slice_name
                    )
                    thin = " THIN" if n < metrics.THIN_HALF_HOURS else ""
                    print(
                        f"{slice_name:24s} {group:16s} {name:18s} {metric:9s} {mean:.3f} "
                        f"({low:.3f} to {high:.3f}) n={n} months={months}{thin}"
                    )
    print("\n== coverage of q05 and q95 (nominal 0.05 and 0.95) ==")
    for group in GROUPS:
        for name, aggregate in aggregates.items():
            print(
                f"{group:16s} {name:18s} "
                f"q05 {mean_coverage(aggregate=aggregate, group=group, metric='coverage_q05'):.3f} "
                f"q95 {mean_coverage(aggregate=aggregate, group=group, metric='coverage_q95'):.3f}"
            )


def fold_means(*, aggregate: pl.DataFrame, members: tuple[str, ...]) -> pl.DataFrame:
    """Return, per fold, the group's CRPS: the mean over Primaries of each Primary's fold mean."""
    return (
        aggregate.filter((pl.col("slice") == ALL_ROWS) & pl.col("label").is_in(list(members)))
        .group_by("label", "month")
        .agg(pl.col("crps").sum(), pl.col("n").sum())
        .with_columns(crps=pl.col("crps") / pl.col("n"))
        .group_by("month")
        .agg(pl.col("crps").mean())
        .rename({"month": "fold"})
        .sort("fold")
    )


def minimum_detectable_effect(*, scorer: Scorer) -> float | None:
    """Print and return the minimum detectable effect at the six (a lower bound).

    The effect is the standard deviation over folds of the six's CRPS difference between the mean
    of an arm's seeds 0 to 2 and the mean of its seeds 3 and 4, divided by the square root of two.

    Args:
        scorer: The scorer that holds the variants' aggregates.

    Returns:
        The effect, or `None` if `M2-n` has no complete second seed set.
    """
    print("\n== minimum detectable effect, signed-MW six (before any contrast) ==")
    names = [Variant(arm=MDE_ARM[:2], state="n", seeds=h).name for h in MDE_HALVES]
    if not all(n in scorer.variants for n in names):
        print("not computed: the stored M2-n lacks seeds 0 to 2 and 3 and 4 on every fold")
        return None
    per_fold = [fold_means(aggregate=scorer.require(n), members=SUBSETS[SIX]) for n in names]
    gap = (
        per_fold[0]
        .join(per_fold[1], on="fold", suffix="_second")
        .with_columns(difference=pl.col("crps") - pl.col("crps_second"))
    )
    sd = float(np.std(gap["difference"].to_numpy(), ddof=1))
    mde = sd / np.sqrt(2)
    print(f"{gap.height} folds, SD of the seed-half difference {sd:.4f}, MDE {mde:.4f} points")
    return mde


def verdict_for(*, lower: float, upper: float) -> str:
    """Return the decision rule's verdict for C1 from its interval, in points of CRPS.

    The first test that holds gives the verdict: `refuted` if the lower limit is above +0.02;
    `supported` if the upper limit is below +0.02 (and beats the reference if below 0);
    `not supported` if the lower limit is above 0; otherwise `inconclusive`.

    Args:
        lower: The lower limit of the interval.
        upper: The upper limit of the interval.

    Returns:
        The verdict, with the width of the interval for `inconclusive`.
    """
    if lower > MARGIN:
        return "refuted"
    if upper < MARGIN:
        return "supported and beats the reference" if upper < 0 else "supported"
    if lower > 0:
        return "not supported"
    return f"inconclusive (width {upper - lower:.3f})"


def check_verdict_branches() -> None:
    """Check on synthetic intervals that `verdict_for` reaches each of its branches.

    Raises:
        AssertionError: If an interval gives a verdict other than its branch's.
    """
    cases = {
        "refuted": (0.03, 0.08),
        "supported and beats": (-0.06, -0.01),
        "supported": (-0.03, 0.01),
        "not supported": (0.005, 0.03),
        "inconclusive": (-0.02, 0.03),
    }
    for expected, (lower, upper) in cases.items():
        got = verdict_for(lower=lower, upper=upper)
        assert got.startswith(expected), (expected, lower, upper, got)
    print("verdict_for: every branch passes on synthetic intervals")


def combine_verdicts(*, verdicts: dict[str, str]) -> str:
    """Return the experiment's verdict from the two group verdicts.

    The experiment is `supported` when both groups are, `partly supported` when one is, and
    otherwise the worse of the two group verdicts.

    Args:
        verdicts: Each group's verdict.

    Returns:
        The experiment's verdict.
    """
    words = [next(w for w in VERDICT_ORDER if v.startswith(w)) for v in verdicts.values()]
    supported = sum(w == "supported" for w in words)
    if supported == len(words):
        return "supported"
    if supported:
        return "partly supported"
    return min(words, key=VERDICT_ORDER.index)


def c1_guards(
    *,
    group: str,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    slices: tuple[str, ...],
    seed_sets: dict[str, tuple[pl.DataFrame, pl.DataFrame] | None],
    versus_m2: tuple[pl.DataFrame, pl.DataFrame] | None,
    family_level: float,
    prefix: str = "C1",
    with_versus_m2: bool = True,
) -> dict[str, bool | None]:
    """Return whether each condition that guards a `supported` verdict holds for a group.

    Args:
        group: The group.
        treatment: The aggregates of `B*-i`.
        reference: The aggregates of `M1-i`.
        slices: The columns of the shared slice table.
        seed_sets: For `seeds_012` and `seeds_34`, the (treatment, reference) aggregates on those
            seeds.
        versus_m2: The aggregates of `B*-i` and `M2-i` on shared seeds, for the six.
        family_level: The level of family R (family E for the extension E1).
        prefix: The contrast's label in the printed lines (`E1` for the extension).
        with_versus_m2: Whether the six get the `B*` minus `M2` check (not for `E1`).

    Returns:
        Each check's name and whether it holds; `None` where the rows are not available. The
        caller treats `None` as not passed, except for the clear-sky-index slice.
    """
    out: dict[str, bool | None] = {}
    for name, pair in seed_sets.items():
        if pair is None:
            out[f"point below +0.02 on {name}"] = None
        else:
            d = group_value(aggregate=pair[0], group=group, metric="crps") - group_value(
                aggregate=pair[1], group=group, metric="crps"
            )
            out[f"point below +0.02 on {name}"] = bool(d < MARGIN)
    if group == EIGHT:
        r = contrast(
            label=f"{prefix} without D8",
            group=group,
            treatment=treatment,
            reference=reference,
            without_label="D8",
            level_percent=family_level,
        )
        out["upper below +0.02 on the eight without D8"] = (
            None if r is None else r["upper"] < MARGIN
        )
    if group == SIX and with_versus_m2:
        if versus_m2 is None:
            out["B* minus M2 below +0.02 at the six"] = None
        else:
            d = group_value(aggregate=versus_m2[0], group=group, metric="crps") - group_value(
                aggregate=versus_m2[1], group=group, metric="crps"
            )
            out["B* minus M2 below +0.02 at the six"] = bool(d < MARGIN)
    sunny = "sunny_clear_sky_index"
    checks: dict[str, tuple[str, str | None]] = {
        "tail q01": ("pinball_q01", None),
        "tail q99": ("pinball_q99", None),
        "cold nights": ("crps", "cold_nights"),
        sunny: ("crps", sunny),
        **{
            band: ("crps", band)
            for band in (
                "lead_day_1",
                "lead_day_2",
                "lead_days_3_4",
                "lead_days_5_7",
                "lead_days_8_10",
            )
        },
    }
    for name, (metric, column) in checks.items():
        key = f"one-sided 95% bound within 0.05 on {name}"
        if column is not None and column not in slices:
            out[key] = None
            continue
        r = contrast(
            label=f"{prefix} {name}",
            group=group,
            treatment=treatment,
            reference=reference,
            metric=metric,
            slice_name=ALL_ROWS if column is None else column,
            level_percent=ONE_SIDED_LEVEL,
        )
        out[key] = None if r is None else r["upper"] < SLICE_MARGIN
    return out


def family_contrasts(
    *,
    scorer: Scorer,
    have: dict[tuple[str, str], tuple[int, ...]],
    args: argparse.Namespace,
    mde: float | None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Print the tested contrasts C1 to C7 and I1 to I3 per group, and the verdict of C1.

    Args:
        scorer: The scorer that holds the variants' aggregates.
        have: The complete seeds of each (arm, state).
        args: The command-line arguments (`B*`, `O*`, `h_T`).
        mde: The minimum detectable effect, or `None`.

    Returns:
        The CRPS figures by (contrast, group).
    """
    print(
        f"\n== tested contrasts, treatment minus reference, days 1 to 10, {N_RESAMPLES:,} "
        f"resamples; minimum detectable effect at the six: {mde} =="
    )
    found: dict[tuple[str, str], dict[str, float]] = {}

    def shared(ta: str, ts: str, ra: str, rs: str) -> tuple[str, str] | None:
        seeds = tuple(sorted(set(have.get((ta, ts), ())) & set(have.get((ra, rs), ()))))
        if not seeds:
            return None
        return Variant(arm=ta, state=ts, seeds=seeds).name, Variant(
            arm=ra, state=rs, seeds=seeds
        ).name

    for label, (t, r) in arms.CONTRAST_NAMES.items():
        ta, ra = resolve(name=t, args=args), resolve(name=r, args=args)
        names = shared(ta, "i", ra, "i")
        if names is None:
            print(f"{label}: skipped, {ta}-i and {ra}-i share no complete seed")
            continue
        level = arms.FAMILY_LEVELS[arms.FAMILY_OF[label]][0]
        for group in GROUPS:
            figures = contrast(
                label=f"{label} {ta}-i minus {ra}-i",
                group=group,
                treatment=scorer.require(names[0]),
                reference=scorer.require(names[1]),
                level_percent=level,
            )
            if figures is not None:
                found[label, group] = figures
    for n, x in enumerate(("M1", "M2", args.b_star), start=1):
        names = shared(x, "i", x, "n")
        if names is None:
            print(f"I{n}: skipped, {x} has no shared seed in both states")
            continue
        for group in GROUPS:
            figures = contrast(
                label=f"I{n} {x}-i minus {x}-n",
                group=group,
                treatment=scorer.require(names[0]),
                reference=scorer.require(names[1]),
                level_percent=arms.FAMILY_LEVELS["I"][0],
            )
            if figures is not None:
                found[f"I{n}", group] = figures
    for label, (ta, ts, ra, rs) in (
        ("S1", (args.b_star, "n", "M1", "n")),
        ("S2", (args.b_star, "i", "M2", "i")),
    ):
        names = shared(ta, ts, ra, rs)
        if names is None:
            print(f"{label} (secondary): skipped, {ta}-{ts} and {ra}-{rs} share no complete seed")
            continue
        for group in GROUPS:
            figures = contrast(
                label=f"{label} {ta}-{ts} minus {ra}-{rs}",
                group=group,
                treatment=scorer.require(names[0]),
                reference=scorer.require(names[1]),
                level_percent=arms.FAMILY_LEVELS["R"][0],
                note="[secondary contrast]",
            )
            if figures is not None:
                found[label, group] = figures
                print(
                    f"{label} {group}: secondary verdict (no guards): "
                    f"{verdict_for(lower=figures['lower'], upper=figures['upper'])}"
                )
    c1_verdicts(scorer=scorer, have=have, args=args, found=found)
    return found


def e1_extras(
    *,
    aggregates: dict[str, pl.DataFrame],
    variants: list[Variant],
    slices: tuple[str, ...],
    months: tuple[str, ...],
) -> None:
    """Print the guards, the slice CRPS, and the slump share of the pre-registered extension E1.

    The guards are those of C1, read on `M1h90-i` minus `FTh90d-i` at the level of family E. The
    CRPS of `M1h90-i`, `M1-i`, and `FTh90d-i` is printed on all rows, the sunny half-hours, cold
    nights, and the 7 days after an event, and the slump share beside it.

    Args:
        aggregates: The aggregates by variant name.
        variants: `M1h90-i` and `FTh90d-i` first (the contrast), then `M1-i` if it holds the same
            folds and seeds.
        slices: The columns of the shared slice table.
        months: The folds the variants hold.
    """
    treatment, reference = variants[0], variants[1]
    print(
        "\n== pre-registered extension E1: guards (those of C1), read at the level of family E =="
    )
    for group in GROUPS:
        guards = c1_guards(
            group=group,
            treatment=aggregates[treatment.name],
            reference=aggregates[reference.name],
            slices=slices,
            seed_sets={
                f"seeds {list(treatment.seeds)}": (
                    aggregates[treatment.name],
                    aggregates[reference.name],
                )
            },
            versus_m2=None,
            family_level=arms.FAMILY_LEVELS["E"][0],
            prefix="E1",
            with_versus_m2=False,
        )
        failed = [k for k, v in guards.items() if v is False]
        missing = [k for k, v in guards.items() if v is None and k != SUNNY_KEY]
        print(f"E1 {group}: guards failed {failed}; not available {missing} (read as explanatory)")
    print("\n== pre-registered extension E1: CRPS on the slices (95% month-block intervals) ==")
    wanted = (ALL_ROWS, "sunny_clear_sky_index", "sunny_unit_shape", "cold_nights")
    present = slice_names(slices=slices)
    for slice_name in (*wanted, "event_following_7_days"):
        if slice_name not in present:
            continue
        for group in GROUPS:
            for v in variants:
                mean, low, high, n, n_months = absolute_interval(
                    aggregate=aggregates[v.name], group=group, metric="crps", slice_name=slice_name
                )
                thin = " THIN" if n < metrics.THIN_HALF_HOURS else ""
                print(
                    f"{slice_name:24s} {group:16s} {v.name:18s} crps {mean:.3f} "
                    f"({low:.3f} to {high:.3f}) n={n} months={n_months}{thin}"
                )
    slump_tables(variants=variants, months=months)


def e1_report(
    *,
    scorer: Scorer,
    have: dict[tuple[str, str], tuple[int, ...]],
    months: tuple[str, ...],
) -> None:
    """Print the guards, slice CRPS, and slump share of E1 on the seeds all three arms hold.

    Args:
        scorer: The scorer that holds the variants' aggregates.
        have: The complete seeds of each (arm, state).
        months: The folds to read.
    """
    seeds = tuple(
        sorted(
            set(have.get(("M1h90", "i"), ()))
            & set(have.get(("FTh90d", "i"), ()))
            & set(have.get(("M1", "i"), ()))
        )
    )
    if not seeds:
        print("E1 (pre-registered extension): skipped, no seed is complete for all three arms")
        return
    variants = [Variant(arm=a, state="i", seeds=seeds) for a in ("M1h90", "FTh90d", "M1")]
    if all(v.name in scorer.variants for v in variants):
        e1_extras(
            aggregates={v.name: scorer.require(v.name) for v in variants},
            variants=variants,
            slices=scorer.slices,
            months=months,
        )


def e1_variants(*, seeds: tuple[int, ...], months: tuple[str, ...]) -> list[Variant]:
    """Return the E1 variants on given seeds and folds: `M1h90-i`, `FTh90d-i`, and `M1-i` if held.

    Args:
        seeds: The seeds of the contrast.
        months: The folds of the contrast.

    Returns:
        The variants, with `M1-i` last and only if its forecasts exist on those seeds and folds.
    """
    out = [Variant(arm=a, state="i", seeds=seeds, months=months) for a in ("M1h90", "FTh90d")]
    folder = forecast_folder(arm="M1", state="i")
    if all((folder / f"{m}_s{s}.parquet").exists() for m in months for s in seeds):
        out.append(Variant(arm="M1", state="i", seeds=seeds, months=months))
    return out


def c1_verdicts(
    *,
    scorer: Scorer,
    have: dict[tuple[str, str], tuple[int, ...]],
    args: argparse.Namespace,
    found: dict[tuple[str, str], dict[str, float]],
) -> None:
    """Print C1's verdict per group, with the guards, and the experiment's verdict.

    Args:
        scorer: The scorer that holds the variants' aggregates.
        have: The complete seeds of each (arm, state).
        args: The command-line arguments.
        found: The contrast figures from `family_contrasts`.
    """
    level = arms.FAMILY_LEVELS["R"][0]
    b = args.b_star
    verdicts: dict[str, str] = {}

    def pick(a: str, s: str, seeds: tuple[int, ...]) -> pl.DataFrame | None:
        name = Variant(arm=a, state=s, seeds=seeds).name
        return scorer.get(name)

    both = tuple(sorted(set(have.get((b, "i"), ())) & set(have.get(("M1", "i"), ()))))
    with_m2 = tuple(sorted(set(have.get((b, "i"), ())) & set(have.get(("M2", "i"), ()))))
    seed_sets: dict[str, tuple[pl.DataFrame, pl.DataFrame] | None] = {}
    for name, half in (("seeds 0 to 2", (0, 1, 2)), ("seeds 3 and 4", (3, 4))):
        t, r = pick(b, "i", half), pick("M1", "i", half)
        seed_sets[name] = None if t is None or r is None else (t, r)
    t_all, r_all = pick(b, "i", both), pick("M1", "i", both)
    t_m2, m2 = pick(b, "i", with_m2), pick("M2", "i", with_m2)
    print("\n== C1 verdict (decision rule of the plan, with its guards) ==")
    for group in GROUPS:
        c1 = found.get(("C1", group))
        if c1 is None or t_all is None or r_all is None:
            print(f"{group}: C1 not available")
            continue
        verdict = verdict_for(lower=c1["lower"], upper=c1["upper"])
        guards = c1_guards(
            group=group,
            treatment=t_all,
            reference=r_all,
            slices=scorer.slices,
            seed_sets=seed_sets,
            versus_m2=None if t_m2 is None or m2 is None else (t_m2, m2),
            family_level=level,
        )
        failed = [k for k, v in guards.items() if v is False]
        missing = [k for k, v in guards.items() if v is None and k != SUNNY_KEY]
        if verdict.startswith("supported"):
            if failed:
                verdict = f"inconclusive (a guard failed: {failed})"
            elif missing:
                verdict = f"inconclusive (guard not available: {missing})"
        print(f"{group}: {verdict}; guards failed {failed}; not available {missing}")
        verdicts[group] = verdict
    if len(verdicts) == len(GROUPS):
        print(f"experiment verdict: {combine_verdicts(verdicts=verdicts)}")


def slump_share(*, shifted: np.ndarray, base: np.ndarray, signed_depth: np.ndarray) -> np.ndarray:
    """Return the slump share of each row: `sign(d) * (shifted - base) / abs(d)`.

    A forecast that keeps to the normal running arrangement scores 0, and a forecast that takes up
    the whole step scores 1.

    Args:
        shifted: The median forecast with shifted lags.
        base: The median forecast with unshifted lags.
        signed_depth: The event's signed depth as a share of capacity.

    Returns:
        The share of each row.
    """
    return np.sign(signed_depth) * (shifted - base) / np.abs(signed_depth)


def lag_shift_mask(
    *,
    shift_series: np.ndarray,
    valid_index: np.ndarray,
    origin_index: np.ndarray,
    observed: np.ndarray | None = None,
) -> np.ndarray:
    """Return whether any of a row's three lags reads a shifted half-hour.

    Args:
        shift_series: The shift added to the power at each grid index (zero outside events).
        valid_index: Grid index of each row's valid time.
        origin_index: Grid index of each row's origin.
        observed: The observed power on the grid. If given, a lag that reads a missing
            observation does not count as shifted, because the shift leaves the missing value
            missing.

    Returns:
        A boolean mask over the rows.
    """
    lags = common.origin_lags(power=shift_series, target=valid_index, origin=origin_index)
    shifted = np.nan_to_num(lags) != 0
    if observed is not None:
        read = common.origin_lags(power=observed, target=valid_index, origin=origin_index)
        shifted &= np.isfinite(read)
    return np.any(shifted, axis=1)


def bank_rows(
    *, base: pl.DataFrame, shifted: pl.DataFrame, month: str, pass_id: int
) -> pl.DataFrame:
    """Return the per-row slump measures of one arm for one fold and one bank pass.

    Both forecasts are widened by 1.15. A row is kept if its lead is at most 240 hours, its valid
    time is before the holdout, and at least one of its lags is shifted (`lag_shift_mask`). Each
    row carries the share, the shifted-minus-unshifted median error in points of capacity, and
    the CRPS against the observed power with shifted and with unshifted lags, in points.

    Args:
        base: The arm's seed-averaged forecast with unshifted lags (fraction of capacity).
        shifted: The same arm's forecast with the pass's shifted lags.
        month: The fold.
        pass_id: The bank pass.

    Returns:
        One row per kept forecast row with `primary`, `event_start`, `phase`, `days_since_onset`,
        `share`, `shift_error_pts`, `crps_shifted`, `crps_base`, and `lead_hours`.
    """
    sites = prep.primaries()
    clean = inject.observed_power()
    start = int(common.grid_index(times=prep.refit_of(month=month)))  # ty: ignore[invalid-argument-type]
    end = start + bank581.END_DAY * PER_DAY
    base = s407.widen_global(frame=base)
    shifted = s407.widen_global(frame=shifted)
    joined = base.join(shifted, on=KEYS, suffix="_s", validate="1:1")
    pieces = []
    for p, site in enumerate(sites["site"].to_list()):
        own = joined.filter(pl.col("time_series_id") == sites["time_series_id"][p])
        if own.is_empty():
            continue
        signed, length, _sign = bank581.event(primary=p, r=pass_id)
        onset = end - length * PER_DAY
        series = np.zeros(len(clean[site]), dtype=np.float32)
        series[onset:end] = signed
        valid_idx = np.asarray(common.grid_index(times=own["valid_time"]))
        origin_idx = np.asarray(common.grid_index(times=own["power_fcst_init_time"]))
        mask = lag_shift_mask(
            shift_series=series,
            valid_index=valid_idx,
            origin_index=origin_idx,
            observed=clean[site],
        )
        lead = (valid_idx - origin_idx) / 2.0
        actual = common.take(values=clean[site], index=valid_idx).astype(np.float64)
        keep = mask & (lead <= MAX_LEAD_HOURS) & (valid_idx < common.grid_index(times=HOLDOUT))  # ty: ignore[invalid-argument-type]
        if not keep.any():
            continue
        q_base = own.select(QUANTILES).to_numpy()[keep].astype(np.float64)
        q_shift = own.select([f"{c}_s" for c in QUANTILES]).to_numpy()[keep].astype(np.float64)
        share = slump_share(
            shifted=q_shift[:, common.MEDIAN],
            base=q_base[:, common.MEDIAN],
            signed_depth=np.full(int(keep.sum()), signed),
        )
        origin_k = origin_idx[keep]
        days = (origin_k - onset) / PER_DAY
        phase = np.where(origin_k < end, "in_event", "after_event")
        finite = np.isfinite(actual[keep])
        crps_s = np.full(len(share), np.nan)
        crps_b = np.full(len(share), np.nan)
        for target, q in ((crps_s, q_shift), (crps_b, q_base)):
            target[finite] = 100.0 * common.pinball_quantile_score(
                quantiles=q[finite], actual=actual[keep][finite]
            )
        pieces.append(
            pl.DataFrame(
                {
                    "primary": np.full(len(share), p),
                    "label": site,
                    "month": month,
                    "pass": pass_id,
                    "phase": phase,
                    "days_since_onset": days,
                    "share": share,
                    "shift_error_pts": 100.0
                    * np.sign(signed)
                    * (q_shift[:, common.MEDIAN] - q_base[:, common.MEDIAN]),
                    "crps_shifted": crps_s,
                    "crps_base": crps_b,
                    "lead_hours": lead[keep],
                }
            )
        )
    return pl.concat(pieces) if pieces else pl.DataFrame()


def bank_table(*, variant: Variant, months: tuple[str, ...]) -> pl.DataFrame:
    """Return the per-row slump measures of an arm over the folds, from the stored bank passes.

    Args:
        variant: The arm, state, and seeds (the seeds are averaged, as in the ordinary forecast).
        months: The folds.

    Returns:
        The rows of every fold's two passes; empty if no bank file exists.
    """
    base_folder = forecast_folder(arm=variant.arm, state=variant.state)
    bank_folder = OUT_DIR / "bank" / variant.base
    pieces = []
    for month in months:
        k = arms.SCORED_MONTHS.index(month)
        for pass_id in (k, k + len(bank581.COMBOS)):
            paths = [bank_folder / f"{month}_s{s}_p{pass_id}.parquet" for s in variant.seeds]
            if not all(p.exists() for p in paths):
                continue
            base = average_files(
                paths=[base_folder / f"{month}_s{s}.parquet" for s in variant.seeds]
            )
            pieces.append(
                bank_rows(
                    base=base, shifted=average_files(paths=paths), month=month, pass_id=pass_id
                )
            )
    return pl.concat(pieces) if pieces else pl.DataFrame()


def event_means(*, bank: pl.DataFrame) -> pl.DataFrame:
    """Return the per-event means of the slump measures, by phase.

    An event is one (fold, pass, Primary). The rows inside the event are split by days since the
    onset (`first_4_days`, `day_4_to_end`, and `total`), and the rows in the 7 days after it are
    `after_7_days`.

    Args:
        bank: The output of `bank_table`.

    Returns:
        One row per event and phase with the mean share, the mean error, the mean CRPS change, and
        the number of rows.
    """
    inside = bank.filter(pl.col("phase") == "in_event")
    pieces = [
        inside.with_columns(part=pl.lit("total")),
        inside.filter(pl.col("days_since_onset") < FIRST_PHASE_DAYS).with_columns(
            part=pl.lit("first_4_days")
        ),
        inside.filter(pl.col("days_since_onset") >= FIRST_PHASE_DAYS).with_columns(
            part=pl.lit("day_4_to_end")
        ),
        bank.filter(pl.col("phase") == "after_event").with_columns(part=pl.lit("after_7_days")),
    ]
    return (
        pl.concat(pieces)
        .group_by("month", "pass", "primary", "label", "part")
        .agg(
            share=pl.col("share").mean(),
            shift_error_pts=pl.col("shift_error_pts").mean(),
            crps_change=(pl.col("crps_shifted") - pl.col("crps_base")).fill_nan(None).mean(),
            n_rows=pl.len(),
        )
    )


def slump_tables(*, variants: list[Variant], months: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    """Print the slump share and the NRA error by group, phase, and arm; return the event means.

    Args:
        variants: The variants to read (those with bank files).
        months: The folds.

    Returns:
        The per-event table of each variant, keyed by variant name.
    """
    print("\n== slump share and NRA error in injected events (mean over events) ==")
    out: dict[str, pl.DataFrame] = {}
    for v in variants:
        bank = bank_table(variant=v, months=months)
        if bank.is_empty():
            continue
        events = event_means(bank=bank)
        out[v.name] = events
        for group in GROUPS:
            for part in ("total", "first_4_days", "day_4_to_end", "after_7_days"):
                e = events.filter(
                    pl.col("label").is_in(list(SUBSETS[group])) & (pl.col("part") == part)
                )
                if e.is_empty():
                    continue
                # A mean over no value is `None` in Polars; print it as `nan` instead of crashing.
                share, nra_error, crps_change = e.select(
                    pl.col("share", "shift_error_pts", "crps_change").mean().fill_null(float("nan"))
                ).row(0)
                print(
                    f"{v.name:18s} {group:16s} {part:13s} events={e.height:4d} "
                    f"half-hours={int(e['n_rows'].sum()):6d} share {share:+.3f} "
                    f"NRA error {nra_error:+.3f} pts CRPS change {crps_change:+.3f} pts"
                )
    return out


def bank_seeds(*, arm: str, state: str, months: tuple[str, ...]) -> tuple[int, ...]:
    """Return the seeds whose two bank passes exist for every fold.

    Args:
        arm: The arm.
        state: `i` or `n`.
        months: The folds.

    Returns:
        The seeds, in order.
    """
    folder = OUT_DIR / "bank" / f"{arm}-{state}"
    passes = [
        (month, pass_id)
        for month in months
        for pass_id in (
            arms.SCORED_MONTHS.index(month),
            arms.SCORED_MONTHS.index(month) + len(bank581.COMBOS),
        )
    ]
    return tuple(
        s for s in range(5) if all((folder / f"{m}_s{s}_p{p}.parquet").exists() for m, p in passes)
    )


def slump_contrasts(
    *,
    events: dict[str, pl.DataFrame],
    have: dict[tuple[str, str], tuple[int, ...]],
    args: argparse.Namespace,
    months: tuple[str, ...],
) -> None:
    """Print I4 to I6, the `i` minus `n` slump share of `M1`, `M2`, and `B*`, per group.

    The resampling unit is the calendar month of the event, as for the CRPS contrasts. The seeds
    are those that both states hold with a forecast and with every bank pass, so that a stored
    `-n` bank of seeds 0 to 2 pairs with the first three seeds of an `-i` arm that holds five.

    Args:
        events: The per-event tables from `slump_tables`, reused where they hold the same seeds.
        have: The complete seeds of each (arm, state).
        args: The command-line arguments.
        months: The folds.
    """
    print("\n== I4 to I6: slump share, X-i minus X-n (events paired, month-block bootstrap) ==")
    level = arms.FAMILY_LEVELS["I"][0]
    for n, x in enumerate(("M1", "M2", args.b_star), start=4):
        seeds = tuple(
            sorted(
                set(have.get((x, "i"), ()))
                & set(have.get((x, "n"), ()))
                & set(bank_seeds(arm=x, state="i", months=months))
                & set(bank_seeds(arm=x, state="n", months=months))
            )
        )
        names = [Variant(arm=x, state=s, seeds=seeds).name for s in ("i", "n")]
        if not seeds:
            print(f"I{n}: skipped, {x} has no shared seeds with bank files in both states")
            continue
        for nm, state in zip(names, ("i", "n"), strict=True):
            if nm not in events:
                bank = bank_table(variant=Variant(arm=x, state=state, seeds=seeds), months=months)
                if not bank.is_empty():
                    events[nm] = event_means(bank=bank)
        if not all(nm in events for nm in names):
            print(f"I{n}: skipped, {x} has no shared seeds with bank files in both states")
            continue
        print(f"I{n} seeds {list(seeds)}")
        for group in GROUPS:
            a, b = (
                events[nm].filter(
                    pl.col("label").is_in(list(SUBSETS[group])) & (pl.col("part") == "total")
                )
                for nm in names
            )
            joined = a.join(b, on=["month", "pass", "primary"], suffix="_n", validate="1:1")
            values = (joined["share"] - joined["share_n"]).to_numpy()
            event_months = joined["month"].to_numpy()
            low, high = bootstrap._resample_bounds(
                values=values[None, :], months=event_months, level=level, n_resamples=N_RESAMPLES
            )
            print(
                f"I{n} {x}-i minus {x}-n {group:16s} {values.mean():+.3f} ({low:+.3f} to "
                f"{high:+.3f}) level {level:.2f}% events={len(values)}; reduces the slump: "
                f"{high < 0}"
            )


def real_event_table(*, variant: Variant, months: tuple[str, ...]) -> None:
    """Print the slump share and NRA error for the verified real events, per event.

    The corrected-lag forecast (bank pass 100) is the unshifted reference. Only the rows whose
    lags read a shifted half-hour are kept (`lag_shift_mask`), as in the synthetic bank. The NRA
    actual is the observed power minus the event's depth inside the event. The line gives the
    error against the NRA (`q_obs - nra`) and, beside it, the shifted-minus-corrected error
    (`q_obs - q_cor`), which is the error of the synthetic bank, in which the forecast's own bias
    cancels. The December events of D5 and D16 are
    the real-event result; events that start in May 2026 are printed under a separate flag, because
    experiment 582 found that three of the four fail the check on their depths.

    Args:
        variant: The arm, state, and seeds.
        months: The folds.
    """
    events = inject.real_events_table()
    sites = prep.primaries()["site"].to_list()
    clean = inject.observed_power()
    folder = OUT_DIR / "bank" / variant.base
    base_folder = forecast_folder(arm=variant.arm, state=variant.state)
    printed = False
    for month in months:
        paths = [folder / f"{month}_s{s}_p{bank581.REAL_PASS}.parquet" for s in variant.seeds]
        if not all(p.exists() for p in paths):
            continue
        corrected = average_files(paths=paths)
        observed = average_files(
            paths=[base_folder / f"{month}_s{s}.parquet" for s in variant.seeds]
        )
        joined = observed.join(corrected, on=KEYS, suffix="_c", validate="1:1")
        for e in events:
            site = sites[e["primary"]]
            own = joined.filter(
                pl.col("time_series_id") == prep.primaries()["time_series_id"][e["primary"]]
            )
            origin = np.asarray(common.grid_index(times=own["power_fcst_init_time"]))
            valid = np.asarray(common.grid_index(times=own["valid_time"]))
            series = np.zeros(len(clean[site]), dtype=np.float32)
            series[e["onset_idx"] : e["end_idx"]] = e["depth"]
            lag_mask = lag_shift_mask(
                shift_series=series, valid_index=valid, origin_index=origin, observed=clean[site]
            )
            inside = (
                (origin >= e["onset_idx"])
                & (origin < e["end_idx"])
                & (valid < common.grid_index(times=HOLDOUT))  # ty: ignore[invalid-argument-type]
                & lag_mask
            )
            if not inside.any():
                continue
            q_obs = own["q50"].to_numpy()[inside].astype(np.float64)
            q_cor = own["q50_c"].to_numpy()[inside].astype(np.float64)
            depth = e["depth"]
            share = float(
                np.mean(
                    slump_share(shifted=q_obs, base=q_cor, signed_depth=np.full(len(q_obs), depth))
                )
            )
            shift_error = float(np.mean(100.0 * np.sign(depth) * (q_obs - q_cor)))
            actual = common.take(values=clean[site], index=valid[inside]).astype(np.float64)
            in_span = (valid[inside] >= e["onset_idx"]) & (valid[inside] < e["end_idx"])
            nra = actual - np.where(in_span, depth, 0.0)
            error = float(np.nanmean(100.0 * np.sign(depth) * (q_obs - nra)))
            onset_month = (common.GRID_START + e["onset_idx"] * common.STEP).strftime("%Y-%m")
            flag = "real-event result" if onset_month == "2025-12" else "FLAGGED, depths unverified"
            print(
                f"{variant.name:18s} {site} transfer {e['transfer_id']} onset {onset_month} depth "
                f"{depth:+.3f} half-hours {int(inside.sum())} share {share:+.3f} NRA error "
                f"{error:+.3f} pts, shifted-minus-corrected error {shift_error:+.3f} pts [{flag}]"
            )
            printed = True
    if not printed:
        print(f"{variant.name}: no bank pass 100 files, the real-event table is empty")


def convergence_table(*, arms_states: list[tuple[str, str]]) -> None:
    """Print, per arm and state, the mean steps, the seconds, and whether the final stage converged.

    An arm whose final stage hits its epoch limit or step cap in more than a quarter of its fits is
    labelled `not converged`. Only the final stage is read, because the stored `S_era` stage of
    `M1` stopped at its limit in 38 of 60 fits.

    Args:
        arms_states: The arms and states to read.
    """
    print("\n== steps, GPU seconds, and convergence of the final stage (from the fit logs) ==")
    for arm, state in arms_states:
        logs = [
            json.loads(p.read_text()) for p in (OUT_DIR / "logs" / f"{arm}-{state}").glob("*.json")
        ]
        if not logs:
            continue
        final = [list(log["stages"].values())[-1] for log in logs]
        trained = [s for s in final if "steps" in s]
        if not trained:
            continue
        steps = np.mean(
            [sum(s["steps"] for s in log["stages"].values() if "steps" in s) for log in logs]
        )
        limit = np.mean([bool(s["hit_limit"]) for s in trained])
        seconds = np.mean([log["seconds_total"] for log in logs])
        label = " NOT CONVERGED" if limit > CONVERGENCE_LIMIT else ""
        effective = np.mean(
            [np.mean([e["effective_rows_per_batch"] for e in s["epoch_log"]]) for s in trained]
        )
        print(
            f"{arm}-{state:1s} fits {len(logs):3d} mean steps {steps:8.0f} seconds {seconds:6.0f} "
            f"final stage at a limit {limit:.2f} effective rows per batch {effective:8.0f}{label}"
        )


def tuning_scores(*, args: argparse.Namespace) -> None:
    """Print each arm's CRPS on the tuning folds, the ranking, the choices, and the stop checks.

    The score is the equal-weight mean over a group's Primaries of the mean CRPS over leads of 1 to
    10 days, in points of capacity, after the 1.15 widening, on the cached ENS rows whose origin
    falls in the month. Seeds 0 and 1 are averaged in quantile space. Arms are ranked on the folds
    2025-03 and 2025-06; `PP`'s half-lives come from 2024-10 and 2025-01. All numbers are interim.
    Every (arm, state, month, Primary) CRPS is written to `work/tuning_per_primary.parquet`, and
    `--write-pp-half-lives` writes the JSON that `e00710_fit.py --pp-half-lives` reads.

    Args:
        args: The command-line arguments.
    """
    print("TUNING FOLDS: every number is interim")
    sites = prep.primaries()
    group_of = {label: g for g, members in SUBSETS.items() if g in GROUPS for label in members}
    scores: dict[tuple[str, str, str], dict[str, float]] = {}
    per_primary: list[pl.DataFrame] = []
    for folder in sorted((OUT_DIR / "seed_preds").glob("*-*")):
        arm, state = folder.name.rsplit("-", 1)
        for month in arms.TUNING_MONTHS:
            paths = [folder / f"{month}_s{s}.parquet" for s in arms.TUNING_SEEDS]
            if not all(p.exists() for p in paths):
                continue
            quantiles = average_files(paths=paths)
            extra = pl.read_parquet(paths[0]).select(*KEYS, "lead_hours", "y")
            frame = s407.widen_global(frame=quantiles).join(extra, on=KEYS, validate="1:1")
            score = 100.0 * common.pinball_quantile_score(
                quantiles=frame.select(QUANTILES).to_numpy().astype(np.float64),
                actual=frame["y"].to_numpy().astype(np.float64),
            )
            labelled = frame.with_columns(score=pl.Series(score)).join(
                sites.select("time_series_id", "site"), on="time_series_id"
            )
            labelled = labelled.filter(pl.col("lead_hours") <= MAX_LEAD_HOURS)
            per_site = labelled.group_by("site").agg(pl.col("score").mean())
            per_primary.append(
                per_site.with_columns(arm=pl.lit(arm), state=pl.lit(state), month=pl.lit(month))
            )
            by_group: dict[str, list[float]] = {g: [] for g in GROUPS}
            for site, value in per_site.iter_rows():
                if site in group_of:
                    by_group[group_of[site]].append(value)
            scores[arm, state, month] = {g: float(np.mean(v)) for g, v in by_group.items() if v}
    table = sorted({(a, s) for a, s, _ in scores})
    for group in GROUPS:
        print(f"\n== {group}: CRPS per tuning fold ==")
        for arm, state in table:
            cells = [scores.get((arm, state, m), {}).get(group) for m in arms.TUNING_MONTHS]
            print(
                f"{arm}-{state:1s} "
                + " ".join("   n/a" if c is None else f"{c:6.3f}" for c in cells)
            )
    ranking = rank_arms(scores=scores, table=table)
    print(
        "\n== ranking on folds 2025-03 and 2025-06 (mean of the six's and the eight's score), i =="
    )
    for arm, value in sorted(ranking.items(), key=lambda kv: kv[1]):
        print(f"{arm:8s} {value:.3f}")
    print_choices(ranking=ranking)
    stop_checks(scores=scores)
    if per_primary:
        frame = pl.concat(per_primary).rename({"site": "label", "score": "crps"})
        common.atomic_write(frame=frame, target=OUT_DIR / "tuning_per_primary.parquet")
        print_per_primary(frame=frame)
        if args.write_pp_half_lives is not None:
            write_pp_half_lives(frame=frame, target=args.write_pp_half_lives)


def rank_arms(
    *, scores: dict[tuple[str, str, str], dict[str, float]], table: list[tuple[str, str]]
) -> dict[str, float]:
    """Return each `i`-state arm's mean score over the ranking folds and both groups.

    Args:
        scores: The group scores by (arm, state, month).
        table: The (arm, state) pairs with a score.

    Returns:
        The score by arm, for arms that hold every ranking fold and both groups.
    """
    ranking: dict[str, float] = {}
    for arm, state in table:
        if state != "i":
            continue
        vals = [
            scores[arm, state, m][g]
            for m in arms.RANKING_MONTHS
            for g in GROUPS
            if (arm, state, m) in scores and g in scores[arm, state, m]
        ]
        if len(vals) == len(arms.RANKING_MONTHS) * len(GROUPS):
            ranking[arm] = float(np.mean(vals))
    return ranking


def print_choices(*, ranking: dict[str, float]) -> None:
    """Print `M1-i`'s score, the parsimony choices `B*` and `O*`, and the finalists.

    Args:
        ranking: Each arm's tuning CRPS in the `i` state.
    """
    if "M1" in ranking:
        print(f"M1-i reference tuning score (never B*): {ranking['M1']:.3f}")
    if not any(is_candidate(arm=a) for a in ranking):
        return
    b_star = parsimony_pick(ranking=ranking, one_shot_only=False)
    o_star = parsimony_pick(ranking=ranking, one_shot_only=True)
    print(f"B* (parsimony rule, band 0.01; M1, P, and arms with S_era excluded): {b_star}")
    print(f"O* (one-shot recipes only): {o_star}")
    creative = sorted(
        (v, a) for a, v in ranking.items() if a in arms.ARMS and arms.ARMS[a].scored == "finalist"
    )
    finalists = [a for _, a in creative[:2]]
    extra = [o_star] if o_star not in finalists else []
    print(
        f"finalists (the two creative arms with the lowest tuning CRPS): {finalists}; "
        f"O* outside them, fit it as well: {extra}"
    )
    half_lives = {a: float(a.removeprefix("W-h")) for a in ranking if a.startswith("W-h")}
    if not half_lives:
        print("no W-h arm is ranked yet: choices not written")
        return
    h_t = half_lives[min(half_lives, key=lambda a: ranking[a])]
    choices = {"b_star": b_star, "o_star": o_star, "h_t": h_t, "finalists": finalists}
    write_json(payload=choices, target=arms.CHOICES_FILE)
    print(f"h_T (the W arm with the lowest tuning CRPS): {h_t:g} years")
    print(f"wrote the choices that e00710_fit.py and this scorer read to {arms.CHOICES_FILE}")


def stop_checks(*, scores: dict[tuple[str, str, str], dict[str, float]]) -> None:
    """Print `STOP:` if `M2-i` scores more than 0.10 points worse than `M1-i` or `M2-n`.

    The comparison is the mean over the tuning folds and groups that both arms hold. The result of
    each check is written to `work/stop_checks.json`, which `e00710_fit.py` reads and refuses to
    start Phases S, C, and SC without (a missing check counts as not passed).

    Args:
        scores: The group scores by (arm, state, month).
    """
    results: dict[str, dict] = {}
    for other, label in (("M1", "i"), ("M2", "n")):
        gaps = [
            scores["M2", "i", m][g] - scores[other, label, m][g]
            for m in arms.TUNING_MONTHS
            for g in GROUPS
            if ("M2", "i", m) in scores
            and (other, label, m) in scores
            and g in scores["M2", "i", m]
            and g in scores[other, label, m]
        ]
        if not gaps:
            print(f"stop check M2-i minus {other}-{label}: not available yet")
            results[f"M2-i minus {other}-{label}"] = {"mean": None, "passed": False}
            continue
        mean = float(np.mean(gaps))
        flag = "STOP: " if mean > STOP_POINTS else ""
        results[f"M2-i minus {other}-{label}"] = {"mean": mean, "passed": mean <= STOP_POINTS}
        print(
            f"{flag}stop check M2-i minus {other}-{label} on the tuning folds: {mean:+.3f} points "
            f"({len(gaps)} cells; the experiment stops above +{STOP_POINTS:.2f})"
        )
    write_json(payload=results, target=arms.STOP_CHECKS_FILE)


def print_per_primary(*, frame: pl.DataFrame) -> None:
    """Print the CRPS of each Primary on the half-life folds for the arms that `PP` chooses among.

    Args:
        frame: One row per (arm, state, month, label) with `crps`.
    """
    wide = (
        frame.filter(
            (pl.col("state") == "i")
            & pl.col("arm").is_in(list(PP_ARMS))
            & pl.col("month").is_in(list(arms.HALF_LIFE_MONTHS))
        )
        .group_by("label", "arm")
        .agg(pl.col("crps").mean())
        .pivot(on="arm", index="label", values="crps")
        .sort("label")
    )
    print(f"\n== per-Primary CRPS, mean of folds {arms.HALF_LIFE_MONTHS}, i (PP's choices) ==")
    print(wide)


def write_pp_half_lives(*, frame: pl.DataFrame, target: Path) -> None:
    """Write each Primary's half-life for `PP`: the lowest mean CRPS on the half-life folds.

    The candidates are `W-h0.5-i`, `W-h1-i`, `W-h2-i`, and `M3-i` (the infinite half-life).

    Args:
        frame: One row per (arm, state, month, label) with `crps`.
        target: The JSON file to write, keyed by site label.
    """
    use = frame.filter(
        (pl.col("state") == "i")
        & pl.col("arm").is_in(list(PP_ARMS))
        & pl.col("month").is_in(list(arms.HALF_LIFE_MONTHS))
    )
    folds_held = use.group_by("arm").agg(pl.col("month").n_unique().alias("n"))
    if folds_held.height < len(PP_ARMS) or min(folds_held["n"].to_list()) < len(
        arms.HALF_LIFE_MONTHS
    ):
        print(f"not written: every one of {PP_ARMS} needs both folds {arms.HALF_LIFE_MONTHS}")
        return
    means = use.group_by("label", "arm").agg(pl.col("crps").mean())
    best = means.sort("crps").group_by("label", maintain_order=True).first()
    payload = {label: PP_ARMS[arm] for label, arm in best.select("label", "arm").iter_rows()}
    write_json(payload=payload, target=target)
    print(f"wrote PP half-lives for {len(payload)} Primaries to {target}")


def hyper_parameter_count(*, arm: arms.Arm) -> int:
    """Return a count of an arm's hyper-parameters, for the parsimony rule.

    Each weight rule counts its tunable values (a half-life, a floor, a window, a schedule, a
    half-life per Primary), a sampler counts one, averaging counts one, and a cosine schedule counts
    three. The count is a convention of this script, so the rule has a fixed order.

    Args:
        arm: The arm.

    Returns:
        The count.
    """
    per_rule = {"flat": 0, "exp": 1, "floor": 2, "schedule": 3, "per_primary": 4, "placebo": 1}
    total = 0
    for stage in arm.stages:
        total += per_rule[stage.weight.kind]
        total += int(stage.mechanism == "sampler") + int(stage.ema_decay is not None)
        total += 3 * int(stage.lr_rule.kind == "cosine") + int(stage.train_days is not None)
    return total


def write_json(*, payload: dict, target: Path) -> None:
    """Write a JSON file through a temporary name that carries this process's id.

    Args:
        payload: The content.
        target: The final path.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(target)


def is_candidate(*, arm: str) -> bool:
    """Return whether an arm may be `B*`: an ENS-only recipe, not `M1`, not the placebo.

    Args:
        arm: The arm name.

    Returns:
        `False` for the reference, the placebo, and any arm with an `S_era` stage.
    """
    if arm not in arms.ARMS:
        return False
    definition = arms.ARMS[arm]
    has_era = any(
        stage.rows == "era" or "era" in (stage.state_key or "") for stage in definition.stages
    )
    return definition.kind not in ("reference", "placebo") and not has_era


def is_one_shot(*, arm: str) -> bool:
    """Return whether an arm is a one-shot recipe: a single `ONE` or `CYC` stage from scratch.

    Args:
        arm: The arm name.

    Returns:
        `False` for `M3`, whose single stage is `S_ens`, and for every arm of two or more stages.
    """
    stages = arms.ARMS[arm].stages
    return len(stages) == 1 and stages[0].name in ("ONE", "CYC")


def parsimony_pick(*, ranking: dict[str, float], one_shot_only: bool) -> str:
    """Return the recipe within 0.01 points of the best that has the fewest stages, then parameters.

    One-shot recipes are preferred on a tie. With `one_shot_only`, only recipes of one `ONE` or
    `CYC` stage from scratch compete, so `M3` (one `S_ens` stage with no weight rule) does not.
    `M1`, the placebo, and every arm with an `S_era` stage never compete, because `B*` is an
    ENS-only recipe.

    Args:
        ranking: Each arm's tuning CRPS.
        one_shot_only: Whether to restrict to one-stage recipes.

    Returns:
        The chosen arm's name.
    """
    pool = {
        a: v
        for a, v in ranking.items()
        if is_candidate(arm=a) and (not one_shot_only or is_one_shot(arm=a))
    }
    best = min(pool.values())
    near = [a for a, v in pool.items() if v <= best + 0.01]
    return min(
        near,
        key=lambda a: (
            len(arms.ARMS[a].stages),
            hyper_parameter_count(arm=arms.ARMS[a]),
            pool[a],
        ),
    )


def t_tests(*, scorer: Scorer, names: dict[str, tuple[str, str]]) -> None:
    """Print a t test on the monthly differences of each named contrast, per group.

    Args:
        scorer: The scorer that holds the variants' aggregates.
        names: Each contrast's treatment and reference variant names.
    """
    print("\n== t tests on the monthly differences (descriptive) ==")
    for label, (t, r) in names.items():
        treatment, reference = scorer.get(t), scorer.get(r)
        if treatment is None or reference is None:
            continue
        for group in GROUPS:
            _months, sums, counts = month_totals(aggregate=treatment, group=group, metric="crps")
            _months, ref_sums, _counts = month_totals(
                aggregate=reference, group=group, metric="crps"
            )
            monthly = (sums - ref_sums) / counts
            if len(monthly) > 1:
                result = stats.ttest_1samp(monthly, 0.0)
                print(
                    f"{label:20s} {group:16s} mean {monthly.mean():+.3f} "
                    f"t {result.statistic:+.2f} p {result.pvalue:.3f} months {len(monthly)}"
                )


def held_months(*, arm: str, state: str, seed: int) -> set[str]:
    """Return the scored folds whose forecast file of an arm, state, and seed exists."""
    folder = forecast_folder(arm=arm, state=state)
    return {m for m in arms.SCORED_MONTHS if (folder / f"{m}_s{seed}.parquet").exists()}


def interim_pick(
    *, ta: str, ts: str, ra: str, rs: str
) -> tuple[tuple[int, ...], tuple[str, ...]] | None:
    """Return the seeds and folds on which an interim table scores a contrast.

    The candidates are the seeds that both arms hold on at least one fold, taken in order as a
    prefix (seed 0, seeds 0 and 1, and so on). Each candidate is scored on the folds that both
    arms hold at every seed of the prefix. The candidate with the most folds wins, and with equal
    folds the one with more seeds.

    Args:
        ta: The treatment arm.
        ts: The treatment state.
        ra: The reference arm.
        rs: The reference state.

    Returns:
        The seeds and the folds in plan order, or `None` if no fold holds both arms.
    """
    both = {
        s: held_months(arm=ta, state=ts, seed=s) & held_months(arm=ra, state=rs, seed=s)
        for s in range(5)
    }
    seeds = [s for s, months in both.items() if months]
    best: tuple[tuple[int, ...], tuple[str, ...]] | None = None
    for k in range(1, len(seeds) + 1):
        chosen = tuple(seeds[:k])
        months = tuple(m for m in arms.SCORED_MONTHS if all(m in both[s] for s in chosen))
        if months and (best is None or (len(months), k) > (len(best[1]), len(best[0]))):
            best = (chosen, months)
    return best


def interim_banner(
    *, seeds: tuple[int, ...], months: tuple[str, ...], aggregate: pl.DataFrame
) -> str:
    """Return the banner that heads every table of an interim run.

    Args:
        seeds: The seeds scored.
        months: The folds scored.
        aggregate: The aggregates of one arm of the contrast.

    Returns:
        The banner text with the half-hours behind each group.
    """
    counts = {
        g: int(month_totals(aggregate=aggregate, group=g, metric="crps")[2].sum()) for g in GROUPS
    }
    return (
        f"INTERIM: folds included [{', '.join(months)}]; seeds [{', '.join(map(str, seeds))}]; "
        f"half-hours per group six={counts[SIX]} eight={counts[EIGHT]}"
    )


def interim_contrasts(*, args: argparse.Namespace, series: pl.DataFrame, ctx: dict) -> None:
    """Print each tested contrast on the folds and seeds that both of its arms hold so far.

    Every table opens with the `INTERIM` banner, and no verdict line is printed. Each contrast is
    scored on its own set of arms and references, so that a contrast whose arms are finished is
    not held back by an arm that is not.

    Args:
        args: The command-line arguments (`B*`, `O*`, `h_T`).
        series: The series table.
        ctx: The scoring context of `s707.cleaned_scoring_context`.
    """
    reference_keys = reference_scored_keys(series=series, ctx=ctx)
    pairs = {
        label: (resolve(name=t, args=args), "i", resolve(name=r, args=args), "i")
        for label, (t, r) in arms.CONTRAST_NAMES.items()
    }
    pairs |= {f"I{n}": (x, "i", x, "n") for n, x in enumerate(("M1", "M2", args.b_star), start=1)}
    for label, (ta, ts, ra, rs) in pairs.items():
        heading = f"{label} {ta}-{ts} minus {ra}-{rs}"
        pick = interim_pick(ta=ta, ts=ts, ra=ra, rs=rs)
        if pick is None:
            print(f"\n== {heading} ==\nINTERIM: no fold holds both arms; INTERIM, no verdict")
            continue
        seeds, months = pick
        variants = [
            Variant(arm=ta, state=ts, seeds=seeds, months=months),
            Variant(arm=ra, state=rs, seeds=seeds, months=months),
        ]
        if label == "E1":
            variants = e1_variants(seeds=seeds, months=months)
        scorer = Scorer(
            series=series, ctx=ctx, variants=variants, months=months, reference_keys=reference_keys
        )
        banner = interim_banner(
            seeds=seeds, months=months, aggregate=scorer.require(variants[0].name)
        )
        print(f"\n== {heading} ==\n{banner}")
        level = arms.FAMILY_LEVELS[arms.FAMILY_OF[label]][0]
        for group in GROUPS:
            contrast(
                label=heading,
                group=group,
                treatment=scorer.require(variants[0].name),
                reference=scorer.require(variants[1].name),
                level_percent=level,
            )
        print(f"{banner}\n   CRPS of each arm, all rows (95% month-block intervals)")
        for v in variants:
            for group in GROUPS:
                mean, low, high, n, n_months = absolute_interval(
                    aggregate=scorer.require(v.name),
                    group=group,
                    metric="crps",
                    slice_name=ALL_ROWS,
                )
                print(
                    f"   {v.name:18s} {group:16s} {mean:.3f} ({low:.3f} to {high:.3f}) n={n} "
                    f"months={n_months}"
                )
        if label == "E1":
            e1_extras(
                aggregates={v.name: scorer.require(v.name) for v in variants},
                variants=variants,
                slices=scorer.slices,
                months=months,
            )
        print(f"{label}: INTERIM, no verdict")
        del scorer
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--tuning", action="store_true", help="score the four tuning folds only")
    parser.add_argument(
        "--write-pp-half-lives",
        type=Path,
        default=None,
        help="with --tuning: write PP's per-Primary half-lives to this JSON file",
    )
    parser.add_argument(
        "--interim",
        action="store_true",
        help="score each contrast on the folds and seeds that both arms hold; no verdict",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the variants and the size of one variant"
    )
    for flag, kind in (("--b-star", str), ("--o-star", str), ("--h-t", float)):
        parser.add_argument(
            flag, type=kind, default=None, help="default: the tuning choice that --tuning wrote"
        )
    return parser


def resolve_choices(*, args: argparse.Namespace) -> None:
    """Fill `b_star`, `o_star`, and `h_t` that the operator did not give from the tuning choice.

    Args:
        args: The parsed arguments, changed in place.
    """
    if None in (args.b_star, args.o_star, args.h_t):
        choices = arms.read_choices()
        args.b_star = args.b_star or choices["b_star"]
        args.o_star = args.o_star or choices["o_star"]
        args.h_t = choices["h_t"] if args.h_t is None else args.h_t
    print(f"choices: B*={args.b_star} O*={args.o_star} h_T={args.h_t:g}")


def main() -> None:
    """Print the checks, the minimum detectable effect, the tables, the contrasts, the verdict."""
    args = build_parser().parse_args()
    check_verdict_branches()
    if args.tuning:
        tuning_scores(args=args)
        return
    resolve_choices(args=args)
    months = tuple(args.months or arms.SCORED_MONTHS)
    series = adapter.series_table()
    ctx = s707.cleaned_scoring_context(series=series)
    if args.interim:
        interim_contrasts(args=args, series=series, ctx=ctx)
        print("\nINTERIM convergence table (no verdict)")
        convergence_table(
            arms_states=[
                (a, s)
                for a in arms.ARMS
                for s in ("i", "n")
                if (OUT_DIR / "logs" / f"{a}-{s}").exists()
            ]
        )
        print(f"\nINTERIM run scored at {datetime.now(UTC):%F %H:%M} UTC; no verdict")
        return
    variants, have = needed_variants(months=months, args=args)
    if args.dry_run:
        print(f"{len(variants)} variants to score: {[v.name for v in variants]}")
        if variants:
            size = seed_average(variant=variants[0], months=months).estimated_size("gb")
            print(
                f"one stored forecast ({variants[0].name}, {len(months)} folds): {size:.2f} GB; "
                f"the scorer holds one fold of one variant at a time, plus the slice table"
            )
        return
    reference_keys = reference_scored_keys(series=series, ctx=ctx)
    scorer = Scorer(
        series=series, ctx=ctx, variants=variants, months=months, reference_keys=reference_keys
    )
    del reference_keys
    s707.assert_before_holdout(name="the shared slice table", frame=scorer.table)
    print(f"shared half-hours across every arm and reference: {scorer.table.height}")
    mde = minimum_detectable_effect(scorer=scorer)
    full = [v for v in variants if v.seeds == have.get((v.arm, v.state))]
    metric_tables(aggregates={v.name: scorer.require(v.name) for v in full}, slices=scorer.slices)
    found = family_contrasts(scorer=scorer, have=have, args=args, mde=mde)
    del found
    e1_report(scorer=scorer, have=have, months=months)
    events = slump_tables(variants=full, months=months)
    slump_contrasts(events=events, have=have, args=args, months=months)
    for v in full:
        if v.state == "i" and v.arm in ("M1", "M2", args.b_star):
            real_event_table(variant=v, months=months)
    convergence_table(arms_states=[(a, s) for (a, s), seeds in have.items() if seeds])
    names: dict[str, tuple[str, str]] = {}
    for label, (t, r) in arms.CONTRAST_NAMES.items():
        ta, ra = resolve(name=t, args=args), resolve(name=r, args=args)
        seeds = tuple(sorted(set(have.get((ta, "i"), ())) & set(have.get((ra, "i"), ()))))
        if seeds:
            names[label] = (
                Variant(arm=ta, state="i", seeds=seeds).name,
                Variant(arm=ra, state="i", seeds=seeds).name,
            )
    t_tests(scorer=scorer, names=names)
    print(f"\nscored at {datetime.now(UTC):%F %H:%M} UTC")


if __name__ == "__main__":
    main()

# ruff: noqa: E501, PLR0912, PT017, PT018
"""Score the Brief B arms of experiment 446 against the stored reference forecasts (no fit).

**Arms.** The new arms (`e00446_run.ARMS446`, `U3`) are read from this experiment's predictions
folder. The references `R-inj`, `S1`, and `Bmix` (and `L0`, `R` as secondary references) are experiment
428's stored forecasts, linked into the same folder by `e00446_run.py link`; `R-inj`, `S1`, and
`L0` also have forecasts on the held-out worlds 7 to 9 from `e00446_run.py predict`.

**Row sets** (flags are experiment 428's, `win_real` and the rest, plus the new `shift_lag`,
`shift_tgt`, and `days_since` of worlds 7 to 9):

- `clean`, `real_event`, `real_harm`: world 0, as in `e00428_score`. `clean_cold` is `clean` on the
  cold winter nights of experiment 441's mask table. The mask table covers the folds 2025-10, 2025-12,
  2026-01, and 2026-03 only, so 2026-06 has no cold-night rows and the report says so.
- `real_event_nra075`, `real_event_nra100`, `real_event_nra125`: the `real_event` rows scored against
  the metered actual minus the NRA correction of experiment 435 (depth factors 0.75, 1.0, 1.25).
- `*_wo_largest`: `real_event`, `real_event_nra100`, and `real_harm` without the longest padded real
  event (and, for `real_harm`, the rows whose most recent event before the target is that event).
  Real-event and real-harm rows are bootstrapped in event clusters (one cluster per padded event),
  all other rows in blocks of one fold and one origin day.
- `inj_clean`, `inj_harm`: experiment 428's held-out injected worlds 4 to 6.
- `gen_clean`: worlds 7 to 9 (the real clean series plus genuine level changes) with no level change
  in the target or in any of the three lags. `gen_jump`: a level change in force at a lag, scored
  against the world's actual (the target moves with the jump when the jump is still in force).
  `gen_jump_d0`, `gen_jump_d1_3`, and `gen_jump_d3p` split `gen_jump` by days since the onset of the
  level change in force at the newest lag (under 1, 1 to under 3, 3 or more).

Groups: the signed-MW six, the MVA-only eight, and D15 and D18 (report-only). Lead bands are
experiment 435's (`0-12h`, `12-36h`, `36-72h`, `days4-5`, `days6-10`).

**Variants.** `raw`; `w1.15` (every spread about the median multiplied by 1.15); `P` (table P
recipe of experiment 435's provisional script: one factor per group, lead band, tail, and class,
fitted on the arm's own earlier folds; the first fold is not scored under `P`). Under `P` the CRPS
is that of the seed-averaged adjusted quantiles. The stored `table_p_factors_full_428.parquet` is
read only to cross-check the refitted clean-row factors of `R-inj`.

**Outputs** (per variant, under `work/scores/<variant>/`): absolute scores and coverage, the
clean-row scores by lead band, the 90% band-width ratio (width on a row set over width on clean
rows, per lead band, weighted by the row set's rows), the mean scale factor on clean rows, the
cold-winter-night table (rows, days, and coverage per fold), contrasts with block-bootstrap
intervals, and the minimum detectable effect from `R-inj`'s seed spread. Intervals are interim.

```bash
source <paths>; PYTHONPATH=$PP DATA_PATH_INTERNAL=... \
  uv run --no-sync python ml-experiments/00446-*/e00446_score.py score --variants raw w1.15
uv run --no-sync python ml-experiments/00446-*/e00446_score.py selftest
```
"""

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score428  # ty: ignore[unresolved-import]
import e00446_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NEW_ARMS: Final[tuple[str, ...]] = (*run.ARMS446, *run.PREDICT_ONLY)
REFERENCES: Final[tuple[str, ...]] = ("R-inj", "S1", "Bmix", "L0", "R")
MISSING_OK_ON_GEN: Final[tuple[str, ...]] = ("Bmix", "R")
"""References without forecasts on worlds 7 to 9."""
SCORE_DIR: Final[Path] = run.WORK_DIR / "scores"
MASKS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00441-difficult-row-rescore-of-the-neural-demand-input-bundles"
    / "work"
    / "masks.parquet"
)
NRA_FILE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
    / "nra_correction_share.npz"
)
TABLE_P_FILE: Final[Path] = NRA_FILE.with_name("table_p_factors_full_428.parquet")
NRA_FACTORS: Final[tuple[float, ...]] = (0.75, 1.0, 1.25)
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-12h": (0.0, 12.0),
    "12-36h": (12.0, 36.0),
    "36-72h": (36.0, 72.0),
    "days4-5": (72.0, 120.0),
    "days6-10": (120.0, 240.0),
}
"""Lead bands in hours, lower bound inclusive, upper bound exclusive (the last includes 240)."""
GROUP_OF: Final[dict[str, str]] = {
    **dict.fromkeys(SIGNED_MW_LABELS, "signed-MW six"),
    **dict.fromkeys(MVA_ONLY_LABELS, "MVA-only eight"),
    **dict.fromkeys(("D15", "D18"), "D15 and D18"),
}
TAILS: Final[tuple[str, ...]] = score428.TAILS
LEVEL_OF: Final[dict[str, float]] = score428.LEVEL_OF
POINTS: Final[float] = score428.POINTS
RESAMPLES: Final[int] = score428.RESAMPLES
COLD_FOLDS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03")
VARIANTS: Final[tuple[str, ...]] = ("raw", "w1.15", "P")
P_CLASSES: Final[dict[str, str]] = {
    "clean": "clean", "inj_clean": "clean", "gen_clean": "clean",
    "inj_harm": "inj_harm", "gen_jump": "inj_harm", "real_harm": "inj_harm",
    "real_event": "inj_harm",
}  # fmt: skip
"""Class whose table P factors a row set takes (experiment 435 gives real events the `inj_harm` ones)."""
P_FIT_SETS: Final[tuple[str, ...]] = ("clean", "inj_clean", "inj_harm")
MIN_COLD_DAYS: Final[int] = 20


@dataclass(frozen=True)
class RowSet:
    """A named set of scored rows: the worlds it reads, its rule, its actual, and its bootstrap unit."""

    name: str
    worlds: tuple[int, ...]
    rule: Callable[[], pl.Expr]
    actual: str = "actual"
    cluster: bool = False


def _f(name: str) -> pl.Expr:
    return pl.col(name).fill_null(False)


def row_sets() -> list[RowSet]:
    """Return the row sets in the order the report prints them."""
    plain = ~_f("win_real") & ~_f("tgt_post")
    sets = [
        RowSet("clean", (0,), lambda: plain),
        RowSet("clean_cold", (0,), lambda: plain & _f("cold")),
        RowSet("real_event", (0,), lambda: _f("tgt_real"), cluster=True),
        RowSet("real_harm", (0,), lambda: ~_f("tgt_post") & _f("win_real"), cluster=True),
        RowSet("inj_clean", (4, 5, 6), lambda: ~_f("win_inj") & plain & ~_f("tgt_inj")),
        RowSet("inj_harm", (4, 5, 6), lambda: _f("win_inj") & ~_f("tgt_post") & ~_f("tgt_inj")),
        RowSet("gen_clean", (7, 8, 9), lambda: ~_f("shift_lag") & ~_f("shift_tgt") & plain),
        RowSet("gen_jump", (7, 8, 9), lambda: _f("shift_lag") & plain),
    ]
    for name, low, high in (("d0", 0.0, 1.0), ("d1_3", 1.0, 3.0), ("d3p", 3.0, math.inf)):
        sets.append(
            RowSet(
                f"gen_jump_{name}",
                (7, 8, 9),
                lambda low=low, high=high: (
                    _f("shift_lag")
                    & plain
                    & (pl.col("days_since") >= low)
                    & (pl.col("days_since") < high)
                ),
            )
        )
    for f in NRA_FACTORS:
        tag = f"nra{round(f * 100):03d}"
        sets.append(
            RowSet(
                f"real_event_{tag}",
                (0,),
                lambda: _f("tgt_real"),
                actual=f"actual_{tag}",
                cluster=True,
            )
        )
    for base in ("real_event", "real_event_nra100", "real_harm"):
        parent = next(s for s in sets if s.name == base)
        sets.append(
            RowSet(
                f"{base}_wo_largest",
                parent.worlds,
                lambda parent=parent: parent.rule() & ~_f("largest"),
                actual=parent.actual,
                cluster=True,
            )
        )
    return sets


def require(path: Path, what: str) -> Path:
    """Return `path` or fail with a message that names the missing input and who produces it."""
    if not path.exists():
        msg = f"{what} is missing: {path} (read-only input from another experiment's work folder)"
        raise FileNotFoundError(msg)
    return path


def lead_band_expr() -> pl.Expr:
    """Return the lead-band name of `lead_hours`, or null outside 0 to 240 hours."""
    band = pl.lit(None, dtype=pl.String)
    lead = pl.col("lead_hours")
    for i, (name, (low, high)) in enumerate(LEAD_BANDS.items()):
        last = i == len(LEAD_BANDS) - 1
        inside = (lead >= low) & ((lead <= high) if last else (lead < high))
        band = pl.when(inside).then(pl.lit(name)).otherwise(band)
    return band


def segments(*, real: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return `(seg_at, prior, largest)` for a padded real-event mask (primaries, grid).

    `seg_at` is an event id inside an event (-1 outside); `prior` is the id of the most recent event
    at or before each half-hour (-1 before the first); `largest` is the id of the longest event.
    """
    n_p, size = real.shape
    seg_at = np.full(real.shape, -1, dtype=np.int64)
    for p in range(n_p):
        starts = real[p] & ~np.concatenate([[False], real[p, :-1]])
        number = np.cumsum(starts) - 1
        seg_at[p] = np.where(real[p], p * 100_000 + number, -1)
    prior = np.full(real.shape, -1, dtype=np.int64)
    positions = np.where(seg_at >= 0, np.arange(size)[None, :], -1)
    last = np.maximum.accumulate(positions, axis=1)
    for p in range(n_p):
        found = last[p] >= 0
        prior[p, found] = seg_at[p, last[p, found]]
    ids, counts = np.unique(seg_at[seg_at >= 0], return_counts=True)
    return seg_at, prior, int(ids[np.argmax(counts)])


def widen(*, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Scale each quantile's distance from the median by `factor`."""
    return frame.with_columns(
        *[(pl.col("q50") + factor * (pl.col(q) - pl.col("q50"))).alias(q) for q in pr.LEVEL_NAMES]
    )


def read_arm(
    *, arm: str, month: str, world: int, seeds: tuple[int, ...], factor: float
) -> pl.DataFrame | None:
    """Return the seed-averaged forecast of an arm, month, and world, or None if it has none.

    The CRPS (`score`, `score_s<seed>`) is computed per seed from the seed's widened quantiles, in
    fractions of capacity, as in experiment 428. A missing file is an error except for the
    references without forecasts on the held-out worlds 7 to 9.
    """
    files = [pr.PRED_DIR / arm / f"{month}_s{s}_w{world}.parquet" for s in seeds]
    absent = [f for f in files if not f.exists()]
    if absent:
        if world in run.GEN_WORLDS and arm in MISSING_OK_ON_GEN:
            return None
        msg = f"{absent[0]} is missing: run `e00446_run.py link` or `e00446_run.py predict`"
        raise FileNotFoundError(msg)
    frames = [pl.read_parquet(f) for f in files]
    for other in frames[1:]:
        assert other["valid_us"].equals(frames[0]["valid_us"]), "seed files differ in row order"
    sorted_q = [
        np.sort(
            score428.widened(q=f.select(list(pr.LEVEL_NAMES)).to_numpy(), factor=factor), axis=1
        )
        for f in frames
    ]
    per_seed = [
        metrics.quantile_score(actual=f["actual"].to_numpy(), quantiles=q)
        for f, q in zip(frames, sorted_q, strict=True)
    ]
    mean_q = np.mean(sorted_q, axis=0)
    first = frames[0]
    keep = [
        c
        for c in first.columns
        if c not in pr.LEVEL_NAMES and not c.startswith(("masked_", "mass_"))
    ]
    out = first.select(keep).with_columns(
        [pl.Series(n, mean_q[:, i]) for i, n in enumerate(pr.LEVEL_NAMES)],
        score=pl.Series(np.mean(per_seed, axis=0)),
        **{f"score_s{s}": pl.Series(v) for s, v in zip(seeds, per_seed, strict=True)},
        world=pl.lit(world),
    )
    if "scale_f" in first.columns:
        out = out.with_columns(
            scale_f=pl.Series(np.mean([f["scale_f"].to_numpy() for f in frames], axis=0))
        )
    return out


class Context:
    """Read-only inputs shared by every arm: event clusters, NRA corrections, cold-night keys."""

    def __init__(self, *, need_cold: bool, need_nra: bool) -> None:
        """Load the real-event segments and, if asked, the NRA corrections and the cold-night keys."""
        self.sites = data.sites()
        self.real = data.real_event_mask()
        self.seg_at, self.prior, self.largest = segments(real=self.real)
        self.nra: dict[float, np.ndarray] | None = None
        if need_nra:
            z = np.load(require(NRA_FILE, "NRA-corrected actuals (experiment 435)"))
            self.nra = {f: z[f"f{f}"] for f in NRA_FACTORS}
            assert self.nra[1.0].shape == self.real.shape, (
                "NRA correction and the dense grid differ"
            )
        self.cold: pl.DataFrame | None = None
        if need_cold:
            masks = pl.read_parquet(
                require(MASKS, "the cold-winter-night mask table (experiment 441)"),
                columns=["power_fcst_init_time", "valid_time", "label", "cold_winter_night"],
            )
            self.cold = (
                masks.filter(pl.col("cold_winter_night"))
                .select(
                    label=pl.col("label"),
                    origin_us=pl.col("power_fcst_init_time").dt.epoch("us"),
                    valid_us=pl.col("valid_time").dt.epoch("us"),
                    cold=pl.lit(True),
                )
                .unique()
            )

    def tag(self, *, frame: pl.DataFrame) -> pl.DataFrame:
        """Add label, group, lead band, cluster ids, NRA actuals, and the cold-night flag."""
        primary = frame["primary"].to_numpy().astype(np.int64)
        index = ((frame["valid_us"].to_numpy() - _grid_start_us()) // _STEP_US).astype(np.int64)
        size = self.real.shape[1]
        safe = np.clip(index, 0, size - 1)
        at, prior = self.seg_at[primary, safe], self.prior[primary, safe]
        cluster = np.where(at >= 0, at, prior)
        out = frame.with_columns(
            label=pl.col("primary").replace_strict(
                dict(enumerate(self.sites)), return_dtype=pl.String
            ),
            lead_hours=pl.col("lead").cast(pl.Float64),
            cluster=pl.Series(cluster),
            largest=pl.Series(cluster == self.largest),
        ).with_columns(group=pl.col("label").replace_strict(GROUP_OF), lead_band=lead_band_expr())
        if self.nra is not None:
            out = out.with_columns(
                **{
                    f"actual_nra{round(f * 100):03d}": pl.Series(
                        frame["actual"].to_numpy() - c[primary, safe]
                    )
                    for f, c in self.nra.items()
                }
            )
        if self.cold is not None:
            out = out.join(
                self.cold, on=["label", "origin_us", "valid_us"], how="left"
            ).with_columns(cold=pl.col("cold").fill_null(False))
        else:
            out = out.with_columns(cold=pl.lit(False))
        return out


_STEP_US: Final[int] = 1_800_000_000


def _grid_start_us() -> int:
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    return int(prep422.GRID_START_US)


def load_frames(
    *, arm: str, months: list[str], seeds: tuple[int, ...], factor: float, context: Context
) -> dict[str, pl.DataFrame]:
    """Return, per month, every world of an arm concatenated and tagged."""
    out = {}
    for month in months:
        parts = [
            f
            for w in (*pr.EVAL_WORLDS, *run.GEN_WORLDS)
            if (f := read_arm(arm=arm, month=month, world=w, seeds=seeds, factor=factor))
            is not None
        ]
        out[month] = context.tag(frame=pl.concat(parts, how="diagonal")).with_columns(
            month=pl.lit(month)
        )
    return out


def fit_factors(*, fit_rows: pl.DataFrame) -> pl.DataFrame:
    """Return one factor per (group, lead band, class, tail), as experiment 435's provisional table P.

    Upper tails use the nominal-level quantile of `(actual - q50) / (q - q50)`, lower tails the
    `(1 - nominal)`-quantile of `(q50 - actual) / (q50 - q)`. Rows where `q` equals `q50` are dropped.
    """
    records = []
    for (group, band, cls), part in fit_rows.group_by("group", "lead_band", "p_class"):
        for tail in TAILS:
            level = LEVEL_OF[tail]
            spread = (pl.col(tail) - pl.col("q50")).abs()
            if level > 0.5:
                ratio, tau = (pl.col("actual") - pl.col("q50")) / spread, level
            else:
                ratio, tau = (pl.col("q50") - pl.col("actual")) / spread, 1.0 - level
            use = part.filter(spread > 1e-6)
            if use.is_empty():
                continue
            records.append(
                {"group": group, "lead_band": band, "p_class": cls, "tail": tail,
                 "factor": float(use.select(ratio.quantile(tau)).item()), "n_fit": use.height}
            )  # fmt: skip
    return pl.DataFrame(records)


def apply_factors(*, frame: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """Scale each tail quantile's distance from the median by its fitted factor (1 if none)."""
    out = frame
    for tail in TAILS:
        f = factors.filter(pl.col("tail") == tail).select("group", "lead_band", "p_class", "factor")
        out = (
            out.join(f, on=["group", "lead_band", "p_class"], how="left")
            .with_columns(
                (
                    pl.col("q50") + pl.col("factor").fill_null(1.0) * (pl.col(tail) - pl.col("q50"))
                ).alias(tail)
            )
            .drop("factor")
        )
    return out


def with_p_class(*, frame: pl.DataFrame, sets: list[RowSet]) -> pl.DataFrame:
    """Add `p_class` ('clean' or 'inj_harm') and `p_fit` (rows that may fit factors) to a frame."""
    cls = pl.lit("inj_harm")
    fitted = pl.lit(False)
    for name, mapped in P_CLASSES.items():
        rs = next(s for s in sets if s.name == name)
        in_worlds = pl.col("world").is_in(list(rs.worlds))
        hit = in_worlds & rs.rule()
        cls = pl.when(hit).then(pl.lit(mapped)).otherwise(cls)
        if name in P_FIT_SETS:
            fitted = fitted | hit
    return frame.with_columns(p_class=cls, p_fit=fitted & pl.col("lead_band").is_not_null())


def apply_p(*, frames: dict[str, pl.DataFrame], sets: list[RowSet]) -> dict[str, pl.DataFrame]:
    """Return the frames of every month but the first with table P applied (fitted on earlier months)."""
    months = list(frames)
    tagged = {m: with_p_class(frame=f, sets=sets) for m, f in frames.items()}
    out = {}
    for i, month in enumerate(months[1:], start=1):
        earlier = pl.concat([tagged[m] for m in months[:i]], how="diagonal")
        factors = fit_factors(
            fit_rows=earlier.filter(pl.col("p_fit") & pl.col("actual").is_finite()).select(
                "group", "lead_band", "p_class", "actual", "q50", *TAILS
            )
        )
        adjusted = apply_factors(frame=tagged[month], factors=factors)
        sorted_q = np.sort(adjusted.select(list(pr.LEVEL_NAMES)).to_numpy(), axis=1)
        adjusted = adjusted.with_columns(
            [pl.Series(n, sorted_q[:, j]) for j, n in enumerate(pr.LEVEL_NAMES)],
            score=pl.Series(
                metrics.quantile_score(actual=adjusted["actual"].to_numpy(), quantiles=sorted_q)
            ),
        )
        out[month] = adjusted
    return out


def aggregate(
    *, arm: str, frames: dict[str, pl.DataFrame], sets: list[RowSet], seeds: tuple[int, ...]
) -> pl.DataFrame:
    """Return per-block sums for every row set: n, CRPS, tail pinball and coverage, widths, scale."""
    records = []
    for month, frame in frames.items():
        seed_cols = [f"score_s{s}" for s in seeds if f"score_s{s}" in frame.columns]
        day = (pl.col("origin_us") // (86_400 * 1_000_000)).cast(pl.Int64)
        for rs in sets:
            if rs.name == "clean_cold" and not bool(frame["cold"].any()):
                continue
            actual = pl.col(rs.actual)
            if rs.actual not in frame.columns:
                continue
            sub = frame.filter(
                pl.col("world").is_in(list(rs.worlds))
                & rs.rule()
                & actual.is_finite()
                & pl.col("lead_band").is_not_null()
            )
            if sub.is_empty():
                continue
            exprs = [pl.col("score").alias("crps_sum")]
            for tail in TAILS:
                err = actual - pl.col(tail)
                level = LEVEL_OF[tail]
                exprs.append(
                    pl.when(err >= 0)
                    .then(level * err)
                    .otherwise((level - 1.0) * err)
                    .alias(f"pin_{tail}")
                )
                exprs.append((actual <= pl.col(tail)).cast(pl.Float64).alias(f"cov_{tail}"))
            exprs.append((pl.col("q95") - pl.col("q05")).alias("width90"))
            exprs.append((pl.col("q99") - pl.col("q01")).alias("width98"))
            if "scale_f" in sub.columns:
                exprs.append(pl.col("scale_f").alias("scale_f"))
            block = (
                pl.format("ev{}", pl.col("cluster"))
                if rs.cluster
                else pl.format("{}:{}", pl.col("month"), day)
            )
            sub = sub.with_columns(*exprs, block=block)
            sums = [
                pl.len().alias("n"),
                pl.col("crps_sum").sum(),
                pl.col("width90").sum(),
                pl.col("width98").sum(),
            ]
            sums += [pl.col(c).sum() for c in seed_cols]
            sums += [pl.col(f"pin_{t}").sum() for t in TAILS] + [
                pl.col(f"cov_{t}").sum() for t in TAILS
            ]
            if "scale_f" in sub.columns:
                sums += [pl.col("scale_f").sum(), pl.len().alias("n_scale")]
            table = sub.group_by("group", "lead_band", "block").agg(*sums)
            records.append(
                table.with_columns(arm=pl.lit(arm), month=pl.lit(month), row_set=pl.lit(rs.name))
            )
    return pl.concat(records, how="diagonal")


def collect(
    *, arms: list[str], months: list[str], seeds: tuple[int, ...], variant: str, context: Context
) -> pl.DataFrame:
    """Return the long block table of every arm for a variant (`raw`, `w1.15`, or `P`)."""
    sets = row_sets()
    factor = {"raw": 1.0, "w1.15": score428.WIDENING, "P": 1.0}[variant]
    tables = []
    for arm in arms:
        frames = load_frames(arm=arm, months=months, seeds=seeds, factor=factor, context=context)
        if variant == "P":
            frames = apply_p(frames=frames, sets=sets)
        tables.append(aggregate(arm=arm, frames=frames, sets=sets, seeds=seeds))
        print(f"  {variant}: {arm} done", flush=True)
    return pl.concat(tables, how="diagonal")


def summarise(*, table: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return CRPS (points), tail pinball (points), coverage, and 90% width, summed over blocks."""
    sums = table.group_by(by).agg(
        pl.col("n").sum(), pl.col("crps_sum").sum(), pl.col("width90").sum(),
        pl.col("width98").sum(),
        *[pl.col(f"pin_{t}").sum() for t in TAILS], *[pl.col(f"cov_{t}").sum() for t in TAILS],
        *([pl.col("scale_f").sum(), pl.col("n_scale").sum()] if "scale_f" in table.columns else []),
        pl.col("block").n_unique().alias("blocks"),
    )  # fmt: skip
    out = sums.with_columns(
        crps=pl.col("crps_sum") / pl.col("n") * POINTS,
        width90_mean=pl.col("width90") / pl.col("n"),
        **{f"pinball_{t}": pl.col(f"pin_{t}") / pl.col("n") * POINTS for t in TAILS},
        **{f"coverage_{t}": pl.col(f"cov_{t}") / pl.col("n") for t in TAILS},
    )
    if "scale_f" in out.columns:
        out = out.with_columns(
            scale_f_mean=pl.when(pl.col("n_scale") > 0).then(pl.col("scale_f") / pl.col("n_scale"))
        ).drop("scale_f", "n_scale")
    return out.drop(
        "crps_sum", "width90", "width98", *[f"pin_{t}" for t in TAILS], *[f"cov_{t}" for t in TAILS]
    )


def width_ratios(*, table: pl.DataFrame, numerator: str, denominator: str) -> pl.DataFrame:
    """Return the 90% band-width ratio of a row set over another, per arm and group.

    The ratio is formed within each lead band (mean width on `numerator` over mean width on
    `denominator`, same arm and group) and averaged with the numerator's row counts as weights.
    """
    left = summarise(
        table=table.filter(pl.col("row_set") == numerator), by=["arm", "group", "lead_band"]
    )
    right = summarise(
        table=table.filter(pl.col("row_set") == denominator), by=["arm", "group", "lead_band"]
    )
    joined = left.join(right.select("arm", "group", "lead_band", pl.col("width90_mean").alias("w_ref")),
                       on=["arm", "group", "lead_band"])  # fmt: skip
    return (
        joined.with_columns(ratio=pl.col("width90_mean") / pl.col("w_ref"))
        .group_by("arm", "group")
        .agg(
            ((pl.col("ratio") * pl.col("n")).sum() / pl.col("n").sum()).alias("width_ratio"),
            pl.col("n").sum(),
        )
        .with_columns(row_set=pl.lit(numerator), versus=pl.lit(denominator))
    )


def bootstrap(
    *, blocks: pl.DataFrame, value: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return mean per row and its 95% block-bootstrap interval (blocks resampled with replacement)."""
    per_block = blocks.group_by("block").agg(pl.col("n").sum(), pl.col(value).sum())
    n, v = per_block["n"].to_numpy().astype(float), per_block[value].to_numpy()
    draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
    boot = v[draws].sum(1) / n[draws].sum(1)
    return (
        float(v.sum() / n.sum()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def contrasts(
    *,
    table: pl.DataFrame,
    pairs: list[tuple[str, str]],
    row_sets_: list[str],
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Return paired differences (arm minus reference) of CRPS and tail pinball, in points, with intervals."""
    records = []
    value_cols = {"crps": "crps_sum", **{f"pinball_{t}": f"pin_{t}" for t in TAILS}}
    for a, b in pairs:
        for rs in row_sets_:
            for group in GROUP_NAMES:
                keys = ["block", "lead_band", "month"]
                left = table.filter(
                    (pl.col("arm") == a) & (pl.col("row_set") == rs) & (pl.col("group") == group)
                )
                right = table.filter(
                    (pl.col("arm") == b) & (pl.col("row_set") == rs) & (pl.col("group") == group)
                )
                if left.is_empty() or right.is_empty():
                    continue
                joined = left.join(right, on=keys, suffix="_b")
                if joined.is_empty():
                    continue
                for name, column in value_cols.items():
                    diff = joined.with_columns(diff=pl.col(column) - pl.col(f"{column}_b"))
                    mean, low, high = bootstrap(
                        blocks=diff.select("block", "n", "diff"), value="diff", rng=rng
                    )
                    by_month = diff.group_by("month").agg(pl.col("n").sum(), pl.col("diff").sum())
                    records.append(
                        {
                            "contrast": f"{a} - {b}", "row_set": rs, "group": group, "metric": name,
                            "n": int(joined["n"].sum()), "difference": mean * POINTS,
                            "low": low * POINTS, "high": high * POINTS,
                            "blocks": joined["block"].n_unique(),
                            "fold_differences": {r["month"]: r["diff"] / r["n"] * POINTS for r in by_month.iter_rows(named=True)},
                        }
                    )  # fmt: skip
    return pl.DataFrame(records)


GROUP_NAMES: Final[tuple[str, ...]] = ("signed-MW six", "MVA-only eight", "D15 and D18")


def cold_table(*, table: pl.DataFrame, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return the cold-winter-night rows, days, and tail coverage per arm, group, and fold."""
    cold = table.filter(pl.col("row_set") == "clean_cold")
    if cold.is_empty():
        return cold
    out = summarise(table=cold, by=["arm", "group", "month"])
    return out.select(
        "arm", "group", "month", "n", "blocks", "coverage_q05", "coverage_q95", "coverage_q99",
        "pinball_q95", "pinball_q99", "crps",
    ).sort("group", "month", "arm")  # fmt: skip


def minimum_detectable_effect(*, table: pl.DataFrame, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return experiment 428's minimum detectable effect from R-inj's seed spread (CRPS, points)."""
    sub = table.filter(pl.col("arm") == "R-inj").rename({"crps_sum": "score"})
    renamed = sub.with_columns(
        n=pl.col("n"), **{f"score_s{s}": pl.col(f"score_s{s}") for s in seeds}
    )
    return score428.minimum_detectable_effect(table=renamed, seeds=seeds)


def checks(*, table: pl.DataFrame, label: str) -> pl.DataFrame:
    """Return the predictions P1 to P4 and the clean-row limits against S1 and R-inj, read from `table`.

    Interim: the rules that read table P wait for experiment 435's table and are read here on the
    variant at hand (`label`).
    """
    records: list[dict] = []
    width_inj = width_ratios(table=table, numerator="inj_harm", denominator="inj_clean")
    width_d0 = width_ratios(table=table, numerator="gen_jump_d0", denominator="gen_clean")
    width_d3 = width_ratios(table=table, numerator="gen_jump_d3p", denominator="gen_clean")
    clean = summarise(table=table.filter(pl.col("row_set") == "clean"), by=["arm", "group"])
    harm = summarise(table=table.filter(pl.col("row_set") == "inj_harm"), by=["arm", "group"])
    for group in GROUP_NAMES:
        ref = {
            a: clean.filter((pl.col("arm") == a) & (pl.col("group") == group))
            for a in ("R-inj", "S1")
        }
        for arm in NEW_ARMS:
            rows = clean.filter((pl.col("arm") == arm) & (pl.col("group") == group))
            if rows.is_empty():
                continue
            row = rows.row(0, named=True)
            rec: dict = {"variant": label, "arm": arm, "group": group}
            if arm == "U2-clean":
                hit = width_inj.filter((pl.col("arm") == arm) & (pl.col("group") == group))
                if not hit.is_empty():
                    rec["P1_ratio_inj_harm"] = hit["width_ratio"][0]
                    rec["P1_ok"] = abs(hit["width_ratio"][0] - 1.0) <= 0.02
            for tag, w in (("d0", width_d0), ("d3p", width_d3)):
                hit = w.filter((pl.col("arm") == arm) & (pl.col("group") == group))
                if not hit.is_empty():
                    rec[f"P2_ratio_{tag}"] = hit["width_ratio"][0]
            if "scale_f_mean" in row and row["scale_f_mean"] is not None:
                rec["P4_scale_f_clean"] = row["scale_f_mean"]
                rec["P4_ok"] = row["scale_f_mean"] <= 1.03
            for name, r in ref.items():
                if r.is_empty():
                    continue
                base = r.row(0, named=True)
                rec[f"crps_vs_{name}"] = row["crps"] - base["crps"]
                rec[f"pinball_q95_vs_{name}"] = row["pinball_q95"] - base["pinball_q95"]
                rec[f"pinball_q99_vs_{name}"] = row["pinball_q99"] - base["pinball_q99"]
                rec[f"limit_ok_vs_{name}"] = bool(
                    rec[f"crps_vs_{name}"] <= 0.03
                    and max(rec[f"pinball_q95_vs_{name}"], rec[f"pinball_q99_vs_{name}"]) <= 0.02
                )
            h = harm.filter((pl.col("arm") == arm) & (pl.col("group") == group))
            h_ref = harm.filter((pl.col("arm") == "R-inj") & (pl.col("group") == group))
            if not h.is_empty() and not h_ref.is_empty():
                rec["q99_pinball_inj_harm_vs_R-inj"] = h["pinball_q99"][0] - h_ref["pinball_q99"][0]
            records.append(rec)
    # P3: U2i's q99 coverage on corrupted rows beyond U2's.
    for group in GROUP_NAMES:
        for rs in ("inj_harm", "gen_jump"):
            part = summarise(table=table.filter(pl.col("row_set") == rs), by=["arm", "group"])
            u2 = part.filter((pl.col("arm") == "U2") & (pl.col("group") == group))
            u2i = part.filter((pl.col("arm") == "U2i") & (pl.col("group") == group))
            if not u2.is_empty() and not u2i.is_empty():
                gain = u2i["coverage_q99"][0] - u2["coverage_q99"][0]
                records.append(
                    {"variant": label, "arm": "U2i", "group": group,
                     f"P3_q99_coverage_gain_{rs}": gain, f"P3_ok_{rs}": gain < 0.01}
                )  # fmt: skip
    return pl.DataFrame(records, infer_schema_length=None)


def report(*, table: pl.DataFrame, variant: str, seeds: tuple[int, ...], months: list[str]) -> None:
    """Print and store the tables of one variant."""
    out = SCORE_DIR / variant.replace(".", "_")
    out.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out / "blocks.parquet")
    rng = np.random.default_rng(0)
    cfg = pl.Config(tbl_rows=400, tbl_cols=30, fmt_str_lengths=50, fmt_float="mixed")
    with cfg:
        print(f"===== variant {variant} (months {', '.join(months)}) =====")
        absolute = summarise(table=table, by=["arm", "row_set", "group"])
        absolute.write_parquet(out / "absolute.parquet")
        print(absolute.sort("row_set", "group", "arm"))
        by_lead = summarise(
            table=table.filter(pl.col("row_set") == "clean"), by=["arm", "group", "lead_band"]
        )
        by_lead.write_parquet(out / "clean_by_lead.parquet")
        print(by_lead.sort("group", "lead_band", "arm"))
        ratios = pl.concat(
            [
                width_ratios(table=table, numerator=n, denominator=d)
                for n, d in (
                    ("inj_harm", "inj_clean"), ("real_harm", "clean"), ("real_event", "clean"),
                    ("gen_jump", "gen_clean"), ("gen_jump_d0", "gen_clean"),
                    ("gen_jump_d1_3", "gen_clean"), ("gen_jump_d3p", "gen_clean"),
                    ("gen_clean", "clean"),
                )
                if not table.filter(pl.col("row_set") == n).is_empty()
            ]
        )  # fmt: skip
        ratios.write_parquet(out / "width_ratios.parquet")
        print(ratios.sort("row_set", "group", "arm"))
        cold = cold_table(table=table, seeds=seeds)
        if cold.is_empty():
            print("no cold-winter-night rows in the scored months")
        else:
            cold.write_parquet(out / "cold_nights.parquet")
            print(cold)
            few = (
                cold.filter(pl.col("blocks") < MIN_COLD_DAYS)
                .select("group", "month", "blocks")
                .unique()
            )
            print(f"fold-and-group cells with under {MIN_COLD_DAYS} days (not a test):")
            print(few.sort("group", "month"))
            absent = [m for m in months if m not in cold["month"].unique().to_list()]
            if absent:
                print(f"months without cold-winter-night rows: {', '.join(absent)}")
        if variant != "P":
            if table.filter(pl.col("arm") == "R-inj").is_empty() or len(seeds) < 2:
                print("no minimum detectable effect: it needs R-inj forecasts on at least 2 seeds")
            else:
                mde = minimum_detectable_effect(table=table, seeds=seeds)
                mde.write_parquet(out / "mde.parquet")
                print(mde.sort("row_set", "group"))
        pairs = [
            (a, b)
            for a in (*NEW_ARMS, "S1")
            for b in ("R-inj", "S1")
            if a != b and a in table["arm"].unique().to_list()
        ]
        sets = [
            s
            for s in (
                "clean",
                "clean_cold",
                "inj_harm",
                "gen_jump",
                "real_harm",
                "real_event",
                "real_event_nra100",
                "real_event_wo_largest",
                "real_harm_wo_largest",
            )
            if not table.filter(pl.col("row_set") == s).is_empty()
        ]
        diffs = contrasts(table=table, pairs=pairs, row_sets_=sets, rng=rng)
        if not diffs.is_empty():
            diffs.write_parquet(out / "contrasts.parquet")
            print(diffs.drop("fold_differences").sort("row_set", "group", "metric", "contrast"))
        rules = checks(table=table, label=variant)
        rules.write_parquet(out / "checks.parquet")
        print(rules)


def cross_check_table_p() -> None:
    """Print stored table P's clean-class factors of R-inj beside the refit (read-only cross-check)."""
    stored = pl.read_parquet(require(TABLE_P_FILE, "table P factors (experiment 435)"))
    print(
        stored.filter((pl.col("arm") == "R-inj") & (pl.col("cls") == "clean"))
        .sort("month", "group", "lead_band", "tail")
        .head(20)
    )


def main() -> None:
    """Score the requested variants, or run the selftest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "selftest", "link-check"))
    parser.add_argument("--variants", nargs="+", default=["raw", "w1.15"], choices=VARIANTS)
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    parser.add_argument("--arms", nargs="+", default=[*REFERENCES, *NEW_ARMS])
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="omit cold nights and NRA rows if their inputs are absent",
    )
    args = parser.parse_args()
    run.install()
    if args.command == "selftest":
        selftest()
        return
    if args.command == "link-check":
        cross_check_table_p()
        return
    need_cold, need_nra = True, True
    if args.skip_missing:
        need_cold, need_nra = MASKS.exists(), NRA_FILE.exists()
    context = Context(need_cold=need_cold, need_nra=need_nra)
    months = sorted(args.months)
    seeds = tuple(args.seeds)
    arms = [a for a in args.arms if (pr.PRED_DIR / a).exists()]
    skipped = sorted(set(args.arms) - set(arms))
    if skipped:
        print(f"arms with no forecasts yet (skipped): {', '.join(skipped)}")
    for variant in args.variants:
        used = months[1:] if variant == "P" else months
        table = collect(arms=arms, months=months, seeds=seeds, variant=variant, context=context)
        report(table=table, variant=variant, seeds=seeds, months=used)


def selftest() -> None:
    """Check the tail sums, the width ratio, the event segments, table P, and the bootstrap on toys."""
    rng = np.random.default_rng(0)
    # Tail sums on a toy frame: quantiles 0..10 for every row, actual 5.5 (between q50 and q60).
    n = 50
    q = {name: np.full(n, float(i)) for i, name in enumerate(pr.LEVEL_NAMES)}
    frame = pl.DataFrame(
        {
            "world": 0, "month": "2026-01", "origin_us": np.arange(n) * 86_400 * 1_000_000,
            "actual": np.full(n, 5.5), "score": np.zeros(n), "score_s0": np.zeros(n),
            "group": "signed-MW six", "lead_band": "0-12h", "cluster": -1, "largest": False,
            "cold": False, "win_real": False, "tgt_post": False, "tgt_real": False,
            "win_inj": False, "tgt_inj": False, **q,
        }
    )  # fmt: skip
    clean = next(s for s in row_sets() if s.name == "clean")
    table = aggregate(arm="toy", frames={"2026-01": frame}, sets=[clean], seeds=(0,))
    summary = summarise(table=table, by=["arm", "row_set", "group"]).row(0, named=True)
    # actual 5.5 is above q01 (1), q05 (2), below q95 (9), q99 (10): coverage 0, 0, 1, 1.
    assert summary["coverage_q01"] == 0.0 and summary["coverage_q95"] == 1.0, summary
    # Pinball at q01 (0): error 5.5 above, loss 0.01 * 5.5; at q99: below by 4.5, loss 0.01 * 4.5.
    assert abs(summary["pinball_q01"] - 0.01 * 5.5 * POINTS) < 1e-9
    assert abs(summary["pinball_q99"] - 0.01 * 4.5 * POINTS) < 1e-9
    assert abs(summary["pinball_q95"] - 0.05 * 3.5 * POINTS) < 1e-9
    # Width ratio: a harm row set twice as wide as the clean one gives a ratio of 2.
    harm_frame = frame.with_columns(
        win_real=pl.lit(True),
        **{c: (pl.col("q50") + 2.0 * (pl.col(c) - pl.col("q50"))) for c in pr.LEVEL_NAMES},
    )
    sets = [s for s in row_sets() if s.name in ("clean", "real_harm")]
    wide = pl.concat(
        [
            aggregate(arm="toy", frames={"2026-01": frame}, sets=sets, seeds=(0,)),
            aggregate(arm="toy", frames={"2026-01": harm_frame}, sets=sets, seeds=(0,)),
        ]
    )
    ratio = width_ratios(
        table=wide.filter(pl.col("row_set") != "x"), numerator="real_harm", denominator="clean"
    )
    assert abs(ratio["width_ratio"][0] - 2.0) < 1e-9, ratio
    # Event segments: two events on primary 0 (lengths 3 and 5), one on primary 1.
    real = np.zeros((2, 40), dtype=bool)
    real[0, 2:5] = True
    real[0, 10:15] = True
    real[1, 20:22] = True
    seg_at, prior, largest = segments(real=real)
    assert largest == int(seg_at[0, 12]) and seg_at[0, 3] != seg_at[0, 12] and seg_at[0, 6] == -1
    assert prior[0, 7] == seg_at[0, 3] and prior[0, 30] == seg_at[0, 12] and prior[0, 0] == -1
    # Table P: a tail that covers too little is widened until coverage is nominal on the fitting rows.
    m = 4_000
    actual = rng.normal(0.0, 1.0, m)
    spread_frame = pl.DataFrame(
        {"actual": actual, "q50": np.zeros(m), "group": "g", "lead_band": "0-12h", "p_class": "clean",
         **{t: np.full(m, 0.5 * z) for t, z in (("q01", -2.326), ("q05", -1.645), ("q95", 1.645), ("q99", 2.326))}}
    )  # fmt: skip
    factors = fit_factors(fit_rows=spread_frame)
    fixed = apply_factors(frame=spread_frame, factors=factors)
    covered = float((fixed["actual"].to_numpy() <= fixed["q95"].to_numpy()).mean())
    assert abs(covered - 0.95) < 0.005, covered
    assert abs(float(factors.filter(pl.col("tail") == "q95")["factor"][0]) - 2.0) < 0.15
    # Bootstrap: the interval of a mean contains the sample mean.
    blocks = pl.DataFrame(
        {
            "block": [str(i) for i in range(100)],
            "n": [10] * 100,
            "v": rng.normal(1.0, 0.5, 100) * 10,
        }
    )
    mean, low, high = bootstrap(blocks=blocks, value="v", rng=rng)
    assert low < mean < high and high - low < 0.5
    # Missing inputs fail with a message that names them.
    try:
        require(
            Path("/nonexistent/masks.parquet"), "the cold-winter-night mask table (experiment 441)"
        )
    except FileNotFoundError as error:
        assert "experiment 441" in str(error)
    else:
        msg = "require must fail on a missing input"
        raise AssertionError(msg)
    print("selftest passed")


if __name__ == "__main__":
    main()

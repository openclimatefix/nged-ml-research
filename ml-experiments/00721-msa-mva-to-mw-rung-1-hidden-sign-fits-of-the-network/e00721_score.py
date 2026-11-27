# ruff: noqa: E501
"""Experiment 721: scoring of G1 to G4, the two-stage resampler, and the sign statistics.

The resampler is written here, and nothing under `src/studies/` changes.
`studies.bootstrap.bootstrap_difference_at_level` draws flat units from one `month` column, which is
about 0.77 times as wide as the two-stage draw on the seed-pair null, so no counted reading uses it.

**The two-stage draw.** Each resample draws one seed, then draws folds with replacement, then draws
    blocks
of 10 consecutive valid dates with replacement within each drawn fold. The statistic of a resample
is the
ratio of the summed paired differences to the summed row counts of the drawn blocks. The point
estimate is
the mean over the seeds of the same ratio on all blocks. The interval describes a one-seed contrast,
and
it is widened about the point estimate by the two Kish factors of `e00721_core.widened_interval`,
using
the Kish effective number of the row counts of the blocks that the reading actually holds.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import e00721_core as core
import numpy as np
import polars as pl
from scipy import stats

TAUS: Final = (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99)
"""The 11 quantile levels of the demand protocol."""
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
FOLDED_QCOLS: Final = tuple(f"f{c}" for c in QCOLS)
"""The 11 quantile columns of the folded reading forecast (`fq01` to `fq99`)."""
KEYS: Final = ("time_series_id", "power_fcst_init_time", "valid_time", "fold", "seed")
"""The row key of a prediction file: two arms are paired on it, and a different row set is
refused."""
POINTS: Final = 100.0
"""Fractions of capacity to points of capacity."""


# --------------------------------------------------------------------------- CRPS


def crps_points(*, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 quantiles, in points of capacity, per row.

    Args:
        q: Quantiles in fractions of capacity, shape `(N, 11)`, one column per level of `TAUS`.
        actual: The truth in fractions of capacity, shape `(N,)`.

    Returns:
        The loss of each row in points of capacity, shape `(N,)`.
    """
    taus = np.asarray(TAUS)[None, :]
    err = actual[:, None] - q
    pinball = np.maximum(taus * err, (taus - 1.0) * err)
    return 2.0 * pinball.mean(axis=1) * POINTS


def mae_of_median_points(*, q50: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the absolute error of the median forecast, in points of capacity, per row."""
    return np.abs(actual - q50) * POINTS


def q01_coverage(*, q01: np.ndarray, actual: np.ndarray) -> float:
    """Return the share of truths below the 1% quantile (a reliable forecast gives 0.01)."""
    return float(np.mean(actual < q01)) if q01.size else float("nan")


def with_crps(
    *, frame: pl.DataFrame, qcols: Sequence[str] = QCOLS, truth: str = "actual"
) -> pl.DataFrame:
    """Return `frame` with `crps` (points) and `mae`.

    (points) columns added from its quantile columns.
    """
    q = frame.select(list(qcols)).to_numpy().astype(np.float64)
    y = frame[truth].to_numpy().astype(np.float64)
    return frame.with_columns(
        crps=pl.Series(crps_points(q=q, actual=y)),
        mae=pl.Series(mae_of_median_points(q50=q[:, 5], actual=y)),
    )


# --------------------------------------------------------------------------- blocks and pairing


def block_ids(*, valid_date: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Return, per row, the block of 10 consecutive valid dates within its fold.

    Counted from the fold's first valid date.

    Args:
        valid_date: `datetime64[D]` of each row's valid time.
        fold: The fold label of each row.

    Returns:
        The block number of each row.
    """
    out = np.zeros(len(valid_date), dtype=np.int64)
    for f in np.unique(fold):
        sel = fold == f
        first = valid_date[sel].min()
        out[sel] = (valid_date[sel] - first).astype("timedelta64[D]").astype(
            np.int64
        ) // core.BLOCK_DAYS
    return out


def pair_arms(
    *, a: pl.DataFrame, b: pl.DataFrame, columns: Sequence[str] = ("crps", "mae")
) -> pl.DataFrame:
    """Return the rows of `a` and `b` paired on `KEYS`.

    With each arm's `columns` suffixed `_a` and `_b`.

    Args:
        a: One arm's scored rows.
        b: The other arm's scored rows.
        columns: The score columns to keep from each arm.

    Returns:
        The paired rows, sorted on `KEYS`.

    Raises:
        ValueError: When the two arms hold different row sets or different truths, because scores on
            different rows are not comparable.
    """
    left = a.select(*KEYS, "actual", *columns).rename(
        {c: f"{c}_a" for c in columns} | {"actual": "actual_a"}
    )
    right = b.select(*KEYS, "actual", *columns).rename(
        {c: f"{c}_b" for c in columns} | {"actual": "actual_b"}
    )
    joined = left.join(right, on=list(KEYS), how="inner")
    if not (joined.height == a.height == b.height):
        msg = f"the arms hold different row sets: {a.height} and {b.height} rows, {joined.height} paired"
        raise ValueError(msg)
    drift = float(
        np.max(np.abs(joined["actual_a"].to_numpy() - joined["actual_b"].to_numpy()), initial=0.0)
    )
    if drift > 1e-6:
        msg = f"the two arms score different truths (largest gap {drift:.2e} of capacity)"
        raise ValueError(msg)
    return joined.sort(*KEYS)


@dataclass(frozen=True)
class BlockTable:
    """Sums of paired differences by (fold, block), one row per block, for each seed.

    Attributes:
        fold_of_block: Integer fold code of each block, `(nb,)`.
        counts: Rows of each block (after the reading's row restriction), `(nb,)`.
        sums: Sum of the paired differences of each block, `(n_seeds, nb)`.
        seeds: The seeds, in the order of `sums`.
    """

    fold_of_block: np.ndarray
    counts: np.ndarray
    sums: np.ndarray
    seeds: tuple[int, ...]


def block_table(
    *,
    paired: pl.DataFrame,
    seeds: Sequence[int],
    metric: str = "crps",
    restrict: pl.Expr | None = None,
) -> BlockTable:
    """Aggregate paired rows into the blocks of the resampler.

    Blocks are numbered from all of the fold's rows before `restrict` is applied, so a block with no
    restricted row stays a resampling unit with a zero count, as the page's block counts assume.

    Args:
        paired: Output of `pair_arms` (all seeds stacked).
        seeds: The seeds the reading uses.
        metric: Column pair `<metric>_a` and `<metric>_b` whose difference `a - b` is read.
        restrict: A Polars expression that selects the rows of the reading (reverse-flow rows, one
            Primary).

    Returns:
        The block table of the reading.
    """
    df = paired.with_columns(
        block=pl.Series(
            block_ids(
                valid_date=paired["valid_time"].dt.date().to_numpy().astype("datetime64[D]"),
                fold=paired["fold"].to_numpy(),
            )
        ),
        diff=pl.col(f"{metric}_a") - pl.col(f"{metric}_b"),
    )
    keep = pl.lit(True) if restrict is None else restrict
    folds = sorted(df["fold"].unique().to_list())
    unit = df.select("fold", "block").unique().sort("fold", "block")
    sums = []
    counts = None
    for s in seeds:
        g = (
            df.filter(pl.col("seed") == s)
            .filter(keep)
            .group_by("fold", "block")
            .agg(sum=pl.col("diff").sum(), n=pl.len())
        )
        j = unit.join(g, on=["fold", "block"], how="left").sort("fold", "block").fill_null(0)
        sums.append(j["sum"].to_numpy().astype(np.float64))
        if counts is None:
            counts = j["n"].to_numpy().astype(np.float64)
        elif not np.array_equal(counts, j["n"].to_numpy()):
            msg = "the seeds of one reading hold different rows"
            raise ValueError(msg)
    assert counts is not None
    code = {f: i for i, f in enumerate(folds)}
    return BlockTable(
        fold_of_block=np.array([code[f] for f in unit["fold"].to_list()], dtype=np.int64),
        counts=counts,
        sums=np.vstack(sums),
        seeds=tuple(seeds),
    )


# --------------------------------------------------------------------------- the resampler


@dataclass(frozen=True)
class ContrastReading:
    """One counted or reported contrast: point estimate and widened interval, in points.

    Attributes:
        point: Mean over the seeds of the ratio of summed differences to summed rows.
        lower: Lower end of the widened interval at `level`.
        upper: Upper end of the widened interval at `level`.
        raw_lower: Lower end before the widening.
        raw_upper: Upper end before the widening.
        kish_b: Kish effective number of blocks of the reading's rows.
        n_rows: Rows of the reading.
        n_blocks: Blocks that hold at least one row.
        n_units: Resampling units before the restriction (blocks of all rows of the folds).
        n_resamples: Resamples drawn.
        units_below_minimum: Whether the units are fewer than 20 (the widening then applies, as the
            page requires).
    """

    point: float
    lower: float
    upper: float
    raw_lower: float
    raw_upper: float
    kish_b: float
    n_rows: int
    n_blocks: int
    n_units: int
    n_resamples: int
    units_below_minimum: bool

    def as_dict(self) -> dict:
        """Return the fields as a plain dictionary (for the report and the JSON files)."""
        return dict(self.__dict__)


def two_stage_draws(*, table: BlockTable, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Return the statistic of `n_resamples` two-stage resamples.

    NaN where the drawn blocks hold no row.
    """
    fold_blocks = [np.nonzero(table.fold_of_block == f)[0] for f in np.unique(table.fold_of_block)]
    n_fold = len(fold_blocks)
    n_seed = table.sums.shape[0]
    out = np.empty(n_resamples)
    for r in range(n_resamples):
        s = int(rng.integers(0, n_seed))
        drawn_folds = rng.integers(0, n_fold, n_fold)
        picked = np.concatenate(
            [
                fold_blocks[f][rng.integers(0, len(fold_blocks[f]), len(fold_blocks[f]))]
                for f in drawn_folds
            ]
        )
        n = table.counts[picked].sum()
        out[r] = table.sums[s, picked].sum() / n if n > 0 else np.nan
    return out


def read_contrast(
    *,
    table: BlockTable,
    n_resamples: int = core.N_RESAMPLES,
    seed: int = core.GENERATOR_SEED,
    level: float = core.LEVEL,
) -> ContrastReading:
    """Return the point estimate and the widened interval of a contrast.

    The generator is `np.random.default_rng(seed)`, built inside the call, so a reading never
    depends
    on what was read before it.
    """
    rng = np.random.default_rng(seed)
    total = table.counts.sum()
    if total <= 0:
        nan = float("nan")
        return ContrastReading(
            nan, nan, nan, nan, nan, 0.0, 0, 0, len(table.counts), n_resamples, True
        )
    point = float(np.mean(table.sums.sum(axis=1) / total))
    draws = two_stage_draws(table=table, n_resamples=n_resamples, rng=rng)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.nanquantile(draws, [alpha, 1.0 - alpha])
    kish_b = core.kish_effective_number(counts=table.counts)
    n_units = len(table.counts)
    # the page: "a reading with fewer than 20 resampling units ... widens its interval"
    if n_units < core.MIN_RESAMPLING_UNITS:
        w_lo, w_hi = core.widened_interval(
            point=point, lower=float(lo), upper=float(hi), kish_b=kish_b
        )
    else:
        w_lo, w_hi = float(lo), float(hi)
    return ContrastReading(
        point=point, lower=w_lo, upper=w_hi, raw_lower=float(lo), raw_upper=float(hi), kish_b=kish_b,
        n_rows=int(total), n_blocks=int((table.counts > 0).sum()), n_units=n_units,
        n_resamples=n_resamples, units_below_minimum=n_units < core.MIN_RESAMPLING_UNITS,
    )  # fmt: skip


def contrast(
    *, a: pl.DataFrame, b: pl.DataFrame, seeds: Sequence[int], restrict: pl.Expr | None = None,
    metric: str = "crps", n_resamples: int = core.N_RESAMPLES,
) -> ContrastReading:  # fmt: skip
    """Return the reading of `a` minus `b` on the rows of `restrict`, for the given seeds.

    `a` and `b` carry the `crps` and `mae` columns of `with_crps`. A restriction on `actual_a` (the
    signed
    truth) selects reverse-flow rows: `pl.col("actual_a") < 0`.
    """
    paired = pair_arms(a=a, b=b, columns=("crps", "mae"))
    paired = paired.filter(pl.col("seed").is_in(list(seeds)))
    return read_contrast(
        table=block_table(paired=paired, seeds=seeds, metric=metric, restrict=restrict),
        n_resamples=n_resamples,
    )


# --------------------------------------------------------------------------- descriptive read-outs


def share_of_gap(*, p0: float, s: float, arm: float) -> float:
    """Return the share of the gap from `P0` to `S` that `arm` recovers: `(P0 - arm) / (P0 - S)`."""
    gap = p0 - s
    return float("nan") if abs(gap) < 1e-12 else (p0 - arm) / gap


def median_minus_mean_points(*, q50: np.ndarray, mean: np.ndarray) -> float:
    """Return the mean over rows of the median forecast minus the mean forecast, in points."""
    return float(np.mean(q50 - mean) * POINTS)


# --------------------------------------------------------------------------- sign statistics


def auc(*, score: np.ndarray, positive: np.ndarray) -> float:
    """Return the area under the receiver operating characteristic curve, with ties averaged."""
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sign_tokens(
    *, rho: np.ndarray, x_true_points: np.ndarray, u: np.ndarray, token_id: np.ndarray | None = None,
    u_min: float = core.U_MIN, dead_band: float = core.DEAD_BAND_POINTS,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Return `rho` and the reverse-flow label of the distinct history tokens that the sign.

    Read-outs score.

    A token counts when `rho` and the truth are finite, the real shortwave likelihood `u` is at
    least
    `u_min`, and the signed power lies outside the dead band. With `token_id` (a grid index per row)
    the
    first row of each distinct token is kept, because each history token appears in many samples.
    """
    ok = (
        np.isfinite(rho)
        & np.isfinite(x_true_points)
        & (u >= u_min)
        & (np.abs(x_true_points) >= dead_band)
    )
    if token_id is not None:
        _, first = np.unique(np.where(ok, token_id, -1), return_index=True)
        keep = np.zeros(len(rho), dtype=bool)
        keep[first] = True
        ok &= keep
    return rho[ok], x_true_points[ok] < 0.0


def sign_accuracy(*, rho: np.ndarray, reverse: np.ndarray, threshold: float = 0.5) -> float:
    """Return the share of tokens whose called sign.

    (`rho > threshold` is export) matches the truth.
    """
    return float(np.mean((rho > threshold) == reverse)) if rho.size else float("nan")


def false_call_rate(*, rho: np.ndarray, threshold: float = 0.5) -> float:
    """Return the share of rows called reverse flow (`rho > threshold`)."""
    return float(np.mean(rho > threshold)) if rho.size else float("nan")


def hanley_mcneil_se(*, area: float, n_pos: float, n_neg: float) -> float:
    """Return the standard error of an area by [Hanley and McNeil.

    1982)](https://doi.org/10.1148/radiology.143.1.7063747.
    """
    q1 = area / (2.0 - area)
    q2 = 2.0 * area**2 / (1.0 + area)
    var = (
        area * (1.0 - area) + (n_pos - 1.0) * (q1 - area**2) + (n_neg - 1.0) * (q2 - area**2)
    ) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def se_of_area_difference(
    *, area_1: float, area_2: float, n_pos: float, n_neg: float,
    correlation: float = core.NOISE_CHECK_CORRELATION,
) -> float:  # fmt: skip
    """Return the standard error of the difference of two areas on the same tokens.

    By [Hanley and McNeil (1983)](https://doi.org/10.1148/radiology.148.3.6878708).

    `SE² = SE₁² + SE₂² - 2 r SE₁ SE₂`. The page assumes `r = 0` before any fit, which is
    conservative for
    areas that are positively correlated.
    """
    s1 = hanley_mcneil_se(area=area_1, n_pos=n_pos, n_neg=n_neg)
    s2 = hanley_mcneil_se(area=area_2, n_pos=n_pos, n_neg=n_neg)
    return math.sqrt(max(s1**2 + s2**2 - 2.0 * correlation * s1 * s2, 0.0))


def noise_check(*, n_days_pos: float, n_days_neg: float) -> dict:
    """Return the noise check of the margin.

    0.10 must be at least 3 standard errors of the area difference.

    The class counts are the numbers of distinct days that hold an export token and an import token,
    because tokens within a day share that day's sky. The areas are 0.95 and 0.85 and the
    correlation 0.
    """
    se = se_of_area_difference(
        area_1=core.NOISE_CHECK_AREAS[0], area_2=core.NOISE_CHECK_AREAS[1], n_pos=n_days_pos, n_neg=n_days_neg,
    )  # fmt: skip
    return {
        "se": se,
        "required": core.SE_REQUIRED * se,
        "margin": core.AREA_MARGIN,
        "pass": core.SE_REQUIRED * se <= core.AREA_MARGIN,
    }


def area_evaluable(*, reverse: np.ndarray) -> bool:
    """Return whether both classes hold at least 30 tokens (otherwise the area is `not evaluable`)."""
    n_pos = int(reverse.sum())
    return n_pos >= core.MIN_CLASS_TOKENS and (reverse.size - n_pos) >= core.MIN_CLASS_TOKENS


# --------------------------------------------------------------------------- the power check of G3


def g3_joint_pass_rate(*, plain: Sequence[bool], open_: Sequence[bool]) -> float:
    """Return the share of seed assignments in which both G3 readings pass at zero harm.

    `plain` and `open_` are aligned lists of booleans, one pair per assignment of five stored seeds
    to the
    roles that the two readings share (`P0` seed 0, `P0` seed 1, `SG-A-open`, and the two `SG-A`
    seeds),
    so the rate is a direct count and not the product of the two rates.
    """
    if len(plain) != len(open_) or not plain:
        msg = "the two lists must be aligned and non-empty"
        raise ValueError(msg)
    return float(np.mean([p and o for p, o in zip(plain, open_, strict=True)]))


def null_assignments(
    *, n_stored_seeds: int = 5
) -> list[tuple[tuple[int, int], tuple[int, int], int, int]]:
    """Return every assignment of distinct stored seeds to (`SG-A` seeds 0 and 1.

    `P0` seeds 0 and 1, `SG-A-open`, shared `P0` seed 0 role).

    Each item is `((a0, a1), (p0, p1), open_seed, p0_seed0)`. The five seeds fill `SG-A` seed 0,
    `SG-A`
    seed 1, `P0` seed 0, `P0` seed 1, and `SG-A-open` in every order: 120 ordered assignments, and
    60 once
    the exchange of the two `SG-A` seeds (which reads the same pooled contrast) is removed.
    """
    import itertools

    out = []
    seen = set()
    for perm in itertools.permutations(range(n_stored_seeds), 5):
        a0, a1, p0, p1, op = perm
        key = (frozenset((a0, a1)), p0, p1, op)
        if key in seen:
            continue
        seen.add(key)
        out.append(((a0, a1), (p0, p1), op, p0))
    return out


def seed_pair_point(
    *, frames: dict[tuple[str, int], pl.DataFrame], treat: Sequence[int], ref: Sequence[int],
    time_series_ids: Sequence[int] | None = None,
) -> float:  # fmt: skip
    """Return the point estimate, in points, of the contrast `treat` minus `ref` over all folds.

    `frames[(fold, seed)]` holds the rows of one stored seed with a `crps` column. Seed `treat[i]`
    is
    paired with seed `ref[i]` on `KEYS` (the seed column is ignored), and the point estimate is the
    mean over
    the pairs of the ratio of summed differences to summed rows, which is `read_contrast`'s point
    estimate.
    """
    ratios = []
    for t, r in zip(treat, ref, strict=True):
        diff_sum, n = 0.0, 0
        for fold in sorted({f for f, _ in frames}):
            a = frames[(fold, t)].drop("seed")
            b = frames[(fold, r)].drop("seed")
            if time_series_ids is not None:
                a = a.filter(pl.col("time_series_id").is_in(list(time_series_ids)))
                b = b.filter(pl.col("time_series_id").is_in(list(time_series_ids)))
            keys = [k for k in KEYS if k != "seed"]
            j = a.select(*keys, "crps").join(b.select(*keys, "crps"), on=keys, suffix="_b")
            diff_sum += float((j["crps"] - j["crps_b"]).sum())
            n += j.height
        ratios.append(diff_sum / n)
    return float(np.mean(ratios))


def power_check_g3(
    *, frames: dict[tuple[str, int], pl.DataFrame], id_d5: int, id_d10: int, n_stored_seeds: int = 5,
    pooled_limit: float = core.G3_LIMIT_POOLED_POINTS, per_limit: float = core.G3_LIMIT_PER_PRIMARY_POINTS,
) -> dict:  # fmt: skip
    """Return the joint probability, from the seed-pair null, that all six G3 limits pass.

    Two disjoint groups of the stored seeds of one recipe stand for `SG-A` and `P0`, so the true
    harm is
    zero. Under Option A a check passes when its point estimate is within its limit, so the six
    checks
    need point estimates only. The rate is counted directly over the 60 assignments of the five
    stored
    seeds to `SG-A` seeds 0 and 1, `P0` seeds 0 and 1, and `SG-A-open`, which share `P0` seed 0
    across the
    two readings. The check requires the rate to be at least 0.80.
    """
    scopes = {"pooled": (id_d5, id_d10), "D5": (id_d5,), "D10": (id_d10,)}
    limit = {"pooled": pooled_limit, "D5": per_limit, "D10": per_limit}
    plain_ok, open_ok = [], []
    for (a0, a1), (p0, p1), op, _ in null_assignments(n_stored_seeds=n_stored_seeds):
        plain = all(
            seed_pair_point(frames=frames, treat=(a0, a1), ref=(p0, p1), time_series_ids=ids)
            <= limit[k]
            for k, ids in scopes.items()
        )
        opened = all(
            seed_pair_point(frames=frames, treat=(op,), ref=(p0,), time_series_ids=ids) <= limit[k]
            for k, ids in scopes.items()
        )
        plain_ok.append(plain)
        open_ok.append(opened)
    rate = g3_joint_pass_rate(plain=plain_ok, open_=open_ok)
    return {
        "joint_pass_rate": rate, "n_assignments": len(plain_ok), "plain_pass": int(sum(plain_ok)),
        "open_pass": int(sum(open_ok)), "target": core.G3_POWER_TARGET, "pass": rate >= core.G3_POWER_TARGET,
    }  # fmt: skip

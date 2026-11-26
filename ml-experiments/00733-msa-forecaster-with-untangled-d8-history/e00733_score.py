# ruff: noqa: E501
"""Experiment 733: the paired contrasts, their intervals, the guards, the reading words.

And the verdict.

Real folds: every contrast pairs two arms on the same rows (experiment 667's keys), averages seeds
before the bootstrap, resamples fold months with experiment 713's month-block bootstrap at the
Bonferroni level, and widens the interval by the Student t ratio of the 8 fold months (7 degrees of
freedom). A difference is reported relative to the control's mean loss.

Synthetic world: the Brier score of P(X<0) is differenced on the same half-hours and resampled with
experiment 729's two-way bootstrap (blocks of 7 origin dates within a month, and whole segments),
with
its Kish widening.

The functions take frames and arrays, so a test builds them by hand. The torch-free functions here
read no file; `real_arm_frames` and `real_losses` import experiment 713's scorer when called.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_paths as paths  # noqa: F401
import numpy as np
import polars as pl

KEYS: Final = ("time_series_id", "power_fcst_init_time", "valid_time")
LEVEL_PERCENT: Final = 100.0 * core.LEVEL
EPS: Final = 1e-12


class ScoreError(RuntimeError):
    """A scoring input is unusable (a correctness failure, verdict row 1)."""


# --------------------------------------------------------------------------- real folds


def real_losses(*, frame: pl.DataFrame, truth: np.ndarray) -> pl.DataFrame:
    """Add experiment 713's loss columns (`crps`, `abs_err`, `lead_hours`.

    Pinball) to a seed-level frame.

    Args:
        frame: Quantile rows with `KEYS`, `seed`, and the 11 quantile columns, in contract shares.
        truth: Experiment 713's cleaned-truth stack.

    Returns:
        The frame with the loss columns, unscored rows dropped.
    """
    import e00713_score as s713  # ty: ignore[unresolved-import]

    return s713.add_losses(frame=frame, truth=truth)


def seed_mean(*, frame: pl.DataFrame, seeds: Sequence[int]) -> pl.DataFrame:
    """Average the quantiles of the named seeds and sort them (experiment 713's rule), as seed 0.

    Args:
        frame: Rows with `seed` and the quantile columns.
        seeds: The seeds to keep (all must be present).

    Returns:
        One row per key.

    Raises:
        ScoreError: When a seed is missing.
    """
    import e00713_score as s713  # ty: ignore[unresolved-import]

    held = set(frame["seed"].unique().to_list())
    if not set(seeds) <= held:
        msg = f"seeds {sorted(set(seeds) - held)} are missing from an arm"
        raise ScoreError(msg)
    return s713.seed_average(frame=frame.filter(pl.col("seed").is_in(list(seeds))))


def paired_difference(
    *, treatment: pl.DataFrame, reference: pl.DataFrame, metric: str
) -> pl.DataFrame:
    """Return the per-row difference (treatment minus reference) joined on `KEYS`.

    Never on position.

    Args:
        treatment: The first arm's rows.
        reference: The second arm's rows.
        metric: The loss column.

    Returns:
        The treatment's keys and metric with the difference `d`.

    Raises:
        ScoreError: When the two frames hold different keys.
    """
    if treatment.height != reference.height:
        msg = "the two arms hold different numbers of rows"
        raise ScoreError(msg)
    joined = treatment.select(*KEYS, metric).join(
        reference.select(*KEYS, pl.col(metric).alias("_ref")), on=list(KEYS), how="inner"
    )
    if joined.height != treatment.height:
        msg = "the two arms hold different keys"
        raise ScoreError(msg)
    return joined.with_columns(d=pl.col(metric) - pl.col("_ref")).drop("_ref")


@dataclass(frozen=True)
class RealContrast:
    """One real-fold contrast, in the units of the metric and relative to the reference's mean.

    Attributes:
        point: The mean difference (treatment minus reference).
        lower: The widened lower end.
        upper: The widened upper end.
        raw_lower: The lower end before widening.
        raw_upper: The upper end before widening.
        widening: The t ratio applied.
        reference_mean: The reference arm's mean metric on the rows.
        n_rows: The paired rows.
        half_width: Half the widened interval's width, relative.
        monthly: The mean difference of each fold month.
    """

    point: float
    lower: float
    upper: float
    raw_lower: float
    raw_upper: float
    widening: float
    reference_mean: float
    n_rows: int
    half_width: float
    monthly: tuple[float, ...]

    def relative(self) -> core.Reading:
        """Return the reading relative to the reference's mean metric (the verdict's scale)."""
        s = self.reference_mean
        return core.Reading(self.point / s, self.lower / s, self.upper / s)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        rel = self.relative()
        return {
            "point": self.point, "lower": self.lower, "upper": self.upper, "raw_lower": self.raw_lower,
            "raw_upper": self.raw_upper, "widening": self.widening, "reference_mean": self.reference_mean,
            "n_rows": self.n_rows, "half_width": self.half_width, "monthly": list(self.monthly),
            "relative": {"point": rel.point, "lower": rel.lower, "upper": rel.upper},
        }  # fmt: skip


def real_contrast(
    *, treatment: pl.DataFrame, reference: pl.DataFrame, metric: str = "crps",
    n_resamples: int = core.N_RESAMPLES_REAL, level: float = LEVEL_PERCENT,
) -> RealContrast:  # fmt: skip
    """Return a treatment-minus-reference contrast on the same rows.

    The interval is the month-block bootstrap, widened by the t ratio.

    Both frames hold one row per key (seeds already averaged) with the loss column `metric`.

    Args:
        treatment: The first arm.
        reference: The second arm.
        metric: The loss column.
        n_resamples: The number of resamples.
        level: The interval's coverage in percent.

    Returns:
        The contrast. The generator seed is fixed inside experiment 713's bootstrap.
    """
    from studies import bootstrap, experiment_store

    both = paired_difference(treatment=treatment, reference=reference, metric=metric)
    seed0 = pl.lit(0, dtype=pl.Int64)
    loss = experiment_store.losses_for_bootstrap(
        arm_losses={
            "t": treatment.select(*KEYS, metric).with_columns(seed=seed0),
            "r": reference.select(*KEYS, metric).with_columns(seed=seed0),
        }
    )
    lo, hi = bootstrap.bootstrap_difference_at_level(
        losses=loss,
        treatment="t",
        reference="r",
        metric=metric,
        level=level,
        n_resamples=n_resamples,
    )
    diff = both["d"].to_numpy()
    point = float(diff.mean())
    ratio = max(1.0, core.t_widening(df=core.REAL_FOLD_FACTOR_DF, level=level / 100.0))
    lower, upper = point - ratio * (point - lo), point + ratio * (hi - point)
    months = both["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    monthly = tuple(float(diff[months == m].mean()) for m in np.unique(months))
    ref_mean = float(reference[metric].mean())  # ty: ignore[invalid-argument-type]
    return RealContrast(
        point=point, lower=lower, upper=upper, raw_lower=float(lo), raw_upper=float(hi), widening=ratio,
        reference_mean=ref_mean, n_rows=len(diff), half_width=(upper - lower) / 2.0 / abs(ref_mean),
        monthly=monthly,
    )  # fmt: skip


def leave_one_month_out_keeps_sign(*, monthly: Sequence[float], point: float) -> bool:
    """Return whether the point's sign survives leaving each month out.

    The check takes the mean of the monthly differences without each month in turn.
    """
    arr = np.asarray(monthly, dtype=np.float64)
    if len(arr) < 2 or point == 0.0:
        return False
    sign = np.sign(point)
    return all(np.sign(np.delete(arr, i).mean()) == sign for i in range(len(arr)))


def mae_moves_with_crps(*, crps_point: float, mae_point: float) -> bool:
    """Return whether the normalised mean absolute error of the median moves the same way as CRPS.

    (audit trigger 3).
    """
    return bool(crps_point * mae_point >= 0.0)


def tails_not_worse(*, q01: float, q99: float) -> bool:
    """Return whether FU is within 0.05 points of M* on the q01 and q99 pinball loss.

    The arguments are the point differences.
    """
    return bool(q01 <= core.TAIL_PINBALL_MARGIN_POINTS and q99 <= core.TAIL_PINBALL_MARGIN_POINTS)


def guards_hold(*, c1: RealContrast, mae_point: float, q01: float, q99: float) -> dict[str, bool]:
    """Return the three guards of the verdict table and whether all hold.

    Args:
        c1: The CRPS contrast of FU minus M*.
        mae_point: The mean difference of the absolute error of the median, FU minus M*.
        q01: The mean difference of the q01 pinball loss.
        q99: The mean difference of the q99 pinball loss.

    Returns:
        `leave_one_month_out`, `mae_with_crps`, `tails`, and `all`.
    """
    parts = {
        "leave_one_month_out": leave_one_month_out_keeps_sign(monthly=c1.monthly, point=c1.point),
        "mae_with_crps": mae_moves_with_crps(crps_point=c1.point, mae_point=mae_point),
        "tails": tails_not_worse(q01=q01, q99=q99),
    }
    return {**parts, "all": all(parts.values())}


def lead_band_mask(*, lead_hours: np.ndarray, days: tuple[int, int]) -> np.ndarray:
    """Return the rows of lead days `days[0]` to `days[1]` (a lead day k covers hours 24.

    (k - 1) to 24 k).
    """
    return (lead_hours >= 24.0 * (days[0] - 1)) & (lead_hours < 24.0 * days[1])


def headline_mask(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the rows from 24 h up to but not including 240 h after the origin."""
    lo, hi = core.HEADLINE_LEAD_HOURS
    return (lead_hours >= lo) & (lead_hours < hi)


# --------------------------------------------------------------------------- synthetic world


@dataclass
class SynthRows:
    """The scored half-hours of the synthetic test bank, flattened.

    In the fields that experiment 729's bootstrap reads.

    Attributes:
        month: The month id of each row's segment (the resampling stratum).
        date: The origin day of each row (blocks of 7 days are cut from it).
        seg: The segment of each row.
    """

    month: np.ndarray
    date: np.ndarray
    seg: np.ndarray


def synthetic_rows(
    *,
    seg: np.ndarray,
    origin_day: np.ndarray,
    seg_month: np.ndarray,
    width: int = core.STEPS_PER_DAY,
) -> SynthRows:
    """Flatten samples of `width` half-hours into rows.

    Args:
        seg: The segment of each sample `(N,)`.
        origin_day: The origin day of each sample.
        seg_month: The month id of each segment.
        width: The half-hours per sample.

    Returns:
        The rows `(N * width,)`.
    """
    seg = np.asarray(seg)
    return SynthRows(
        month=np.repeat(np.asarray(seg_month)[seg], width), date=np.repeat(np.asarray(origin_day), width),
        seg=np.repeat(seg, width),
    )  # fmt: skip


def brier_rows(*, p: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return the squared error of each half-hour.

    (NaN where the truth or the probability is absent).
    """
    out = (np.asarray(p, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) ** 2
    return out.ravel()


def log_loss_rows(*, p: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return the log loss of each half-hour with the probability clipped to 1e-6 to 1 - 1e-6."""
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    t = np.asarray(truth, dtype=np.float64)
    return (-(t * np.log(q) + (1.0 - t) * np.log(1.0 - q))).ravel()


def synthetic_contrast(
    *,
    rows: SynthRows,
    first: np.ndarray,
    second: np.ndarray,
    n_resamples: int = core.N_RESAMPLES_SYNTHETIC,
) -> Any:
    """Return experiment 729's two-way bootstrap reading of `first - second`.

    The reading uses the half-hours where both exist.

    Args:
        rows: The flattened rows.
        first: The first arm's per-row metric `(n_rows,)`.
        second: The second arm's per-row metric.
        n_resamples: The number of resamples (the plan's 20,000).

    Returns:
        Experiment 729's `ContrastReading` (point, interval, Kish numbers, widening). Rows where
        either metric is NaN are left out of both arms.
    """
    import e00729_score as s729  # ty: ignore[unresolved-import]

    diff = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    keep = np.isfinite(diff)
    diff = np.where(keep, diff, 0.0)
    table = s729.cell_table(rows=rows, diff=diff, keep=keep)  # ty: ignore[invalid-argument-type]
    return s729.read_contrast(
        table=table, n_resamples=n_resamples, seed=core.GENERATOR_SEED, level=core.LEVEL
    )


def reliability_term(*, p: np.ndarray, truth: np.ndarray, n_bins: int = 10) -> float:
    """Return the reliability term of the Brier decomposition: the count-weighted mean of.

    (mean p - observed share) squared over equal-width bins.
    """
    p, t = np.asarray(p, dtype=np.float64).ravel(), np.asarray(truth, dtype=np.float64).ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if not len(p):
        return float("nan")
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            total += m.sum() * (p[m].mean() - t[m].mean()) ** 2
    return float(total / len(p))


def sign_forecast_inputs(
    *, p_yu: np.ndarray, p_clim: np.ndarray, truth: np.ndarray, seeds_present: bool
) -> dict[str, Any]:
    """Return the quantities that `core.sign_forecast_word` reads.

    Args:
        p_yu: YU's probability on the scored test half-hours.
        p_clim: B-clim's probability on the same half-hours.
        truth: The indicator of X < 0.
        seeds_present: Whether every seed of YU is present.

    Returns:
        `skill`, `reliability`, `share_reverse`, `n_reverse`, `seeds_present`.
    """
    ok = np.isfinite(truth) & np.isfinite(p_yu) & np.isfinite(p_clim)
    bs = float(np.mean((p_yu[ok] - truth[ok]) ** 2))
    bc = float(np.mean((p_clim[ok] - truth[ok]) ** 2))
    return {
        "skill": 1.0 - bs / max(bc, EPS), "reliability": reliability_term(p=p_yu[ok], truth=truth[ok]),
        "share_reverse": float(np.mean(truth[ok])) if ok.any() else float("nan"),
        "n_reverse": int(np.sum(truth[ok] > 0.5)), "seeds_present": seeds_present,
    }  # fmt: skip


def reading_of(*, contrast: Any) -> core.Reading:
    """Return a `core.Reading` from a `RealContrast`.

    (relative) or from experiment 729's `ContrastReading`.
    """
    if isinstance(contrast, RealContrast):
        return contrast.relative()
    r = contrast.reading
    return core.Reading(float(r.point), float(r.lower), float(r.upper))


# --------------------------------------------------------------------------- the verdict


def coverage_note(*, coverage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the synthetic world's ensemble-coverage counts as data for the verdict file.

    (a gap is a result, never an error).
    """
    return dict(coverage or {"status": "not computed"})


def assemble(
    *,
    stopped: bool,
    invalid: bool,
    failed: bool,
    c: Mapping[str, Any],
    guards: Mapping[str, bool] | None,
    sign_inputs: Mapping[str, Any] | None,
    dropped_arms: Sequence[str] = (),
    coverage: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return `verdict.json` and `readings.json` from the counted contrasts and the flags.

    Args:
        stopped: Verdict row 0 applies.
        invalid: Verdict row 1 applies.
        failed: Verdict row 2 applies.
        c: The contrasts and readings by name (`RealContrast`, 729's `ContrastReading`, or None). C1
            to C4 are read for the verdict and the words; R1 to R7 are printed as read.
        guards: `guards_hold`'s result (None when C1 is unread).
        sign_inputs: `sign_forecast_inputs`' result (None when the synthetic arms are absent).
        dropped_arms: Arms dropped by the plan's rule or by a drop of the synthetic world; their
            readings are `void`.
        coverage: The synthetic world's ensemble-coverage counts.

    Returns:
        The two JSON payloads.
    """

    def rd(name: str) -> core.Reading | None:
        v = c.get(name)
        return None if v is None else reading_of(contrast=v)

    c1, c2, c3, c4, r5 = rd("C1"), rd("C2"), rd("C3"), rd("C4"), rd("R5")
    vi = core.VerdictInputs(
        stopped=stopped, invalid=invalid, failed=failed, c1=c1, c2=c2,
        guards_hold=bool(guards and guards["all"]),
    )  # fmt: skip
    row = core.verdict_row(v=vi)
    word = core.verdict_word(v=vi)
    si = dict(sign_inputs) if sign_inputs is not None else None
    sign_word = core.sign_forecast_word(c3=c3, **si) if si is not None else "void"
    hist_word, hist_note = core.history_word(c4=c4, r5=r5)
    if c1 is not None and c1.gain and c1.point > -core.SMALLEST_GAIN and row == 4:
        note_row4 = (
            "row 4 is reachable only if C1's interval is narrower than the prediction expected"
        )
    else:
        note_row4 = None
    verdict = {
        "row": row, "word": word, "front_matter": core.front_matter_word(word=word), "guards": dict(guards or {}),
        "sign_forecast_word": sign_word, "history_word": hist_word, "history_note": hist_note, "note_row4": note_row4,
        "dropped_arms": list(dropped_arms), "synthetic_coverage": coverage_note(coverage=coverage),
        "code_version": core.CODE_VERSION, "level": core.LEVEL,
    }  # fmt: skip
    readings: dict[str, Any] = {}
    for name, v in c.items():
        if v is None:
            readings[name] = "void"
        elif isinstance(v, RealContrast):
            readings[name] = v.as_dict()
        elif hasattr(v, "reading"):
            readings[name] = {
                "point": v.reading.point, "lower": v.reading.lower, "upper": v.reading.upper,
                "kish_blocks": v.kish_blocks, "kish_segments": v.kish_segments, "widening": v.widening,
                "n_rows": v.n_rows, "n_resamples": v.n_resamples,
            }  # fmt: skip
        else:
            readings[name] = v
    for arm in dropped_arms:
        for name, spec in core.CONTRASTS.items():
            if arm in (spec.first, spec.second):
                readings[name] = "void"
    readings["sign_inputs"] = si
    return verdict, readings


def write_results(*, root: Path, verdict: Mapping[str, Any], readings: Mapping[str, Any]) -> None:
    """Write `scores/verdict.json` and `scores/readings.json` (atomically.

    They are written again by a new score run).
    """
    import json
    import os

    for name, payload in (("verdict.json", verdict), ("readings.json", readings)):
        path = root / "scores" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=float))
        tmp.rename(path)

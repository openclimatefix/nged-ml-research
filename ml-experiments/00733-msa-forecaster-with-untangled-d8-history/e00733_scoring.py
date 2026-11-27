# ruff: noqa: E501
"""Experiment 733: scoring the real folds from the stored prediction sets (no torch).

Every counted contrast reads stored sets: M* is experiment 713's stored file of the fold and seed,
and every other arm is the set `e00733_store` names. The rows are those of D8 with lead 24 h up to
240 h after the contract origin (12:00 UTC), seeds averaged and sorted as in experiment 713.

**A validation score has no M*.** M*'s stored files hold the test months only, so a validation score
(the smoke test and the scratch runs) reads F0r in M*'s place, and F0r equals M* on the test rows by
the F0r gate. A validation score never reads a test month.

**Which folds and seeds a contrast reads.** The plan's folds and seeds when every set exists; else
the screen's four folds at seeds 0 and 1; else the folds and seeds that both arms hold (a scratch run
or a stopped run). A contrast that reads less than the plan is marked `interim`.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

import e00733_baselines as baselines
import e00733_core as core
import e00733_score as sc
import e00733_store as store
import numpy as np
import polars as pl

KEYS: Final = sc.KEYS
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
QUANTILE_LEVELS: Final = (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99)
VALIDATION_REFERENCE: Final = "F0r"
N_RESAMPLES_VALIDATION: Final = 1000
"""Resamples of a validation score: the intervals are interim, and the run is a check of the code."""
TAIL_COLUMNS: Final = ("pin_q01", "pin_q05", "pin_q95", "pin_q99")
SCENARIOS: Final = {
    "all rows": "all",
    "outside events": "outside_events",
    "inside events": "event_rows",
    "7 days after an event": "event_following_7_days",
}


@dataclass(frozen=True)
class Plan:
    """The folds and seeds that a contrast reads.

    Attributes:
        folds: The fold months.
        seeds: The seeds.
        interim: Whether the plan reads less than the plan of the experiment.
    """

    folds: tuple[str, ...]
    seeds: tuple[int, ...]
    interim: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return {"folds": list(self.folds), "seeds": list(self.seeds), "interim": self.interim}


Arm = tuple[str, str | None]
"""An arm and its variant (None, "masked", or "stored")."""


def has_job(*, root: Path, arm: Arm, fold: str, seed: int, rows: str) -> bool:
    """Return whether the stored forecast of one arm, fold, and seed exists."""
    name, variant = arm
    if name == "M*":
        return baselines.m_star_file(month=fold, seed=seed).exists()
    folder = store.set_folder(
        root=root, name=store.set_name(arm=name, fold=fold, seed=seed, variant=variant, rows=rows)
    )
    return folder.exists()


class IncompletePlanError(sc.ScoreError):
    """A counted reading lacks stored forecasts for its plan (verdict row 1 on a test score)."""


def screen_stops_rest(*, root: Path) -> bool:
    """Return whether `scores/screen.json` says that the screen stopped the rest of FU and FU-S."""
    path = root / "scores" / "screen.json"
    if not path.exists():
        return False
    return bool(json.loads(path.read_text()).get("stops_rest"))


def choose_plan(
    *, root: Path, arms: Sequence[Arm], wanted_seeds: Sequence[int], rows: str,
    exists: Callable[..., bool] | None = None, screen_ok: bool = False, partial_ok: bool = False,
    required: bool = False,
) -> Plan | None:  # fmt: skip
    """Return the plan that every arm can read, or None (or raise) when none is complete.

    The full plan reads every fold at every wanted seed. The screen (four folds, seeds 0 and 1) is
    read only when `screen_ok` is true, which is the plan's rule that a stopped screen reads C1 and
    C2 on the screen alone. The folds and seeds that happen to exist are read only when
    `partial_ok` is true (figures) or on validation rows (scratch runs).

    Args:
        root: This experiment's data folder.
        arms: The arms of the contrast.
        wanted_seeds: The seeds the plan reads for the contrast.
        rows: "test" or "validation".
        exists: `(root, arm, fold, seed, rows) -> bool` (default `has_job`; tests replace it).
        screen_ok: Whether the screen plan is a plan of this reading.
        partial_ok: Whether the folds and seeds that exist are a plan of this reading.
        required: Whether an incomplete plan raises (a counted reading) instead of returning None.

    Returns:
        The first allowed plan that is complete, or None.

    Raises:
        IncompletePlanError: When `required` and no allowed plan is complete.
    """
    check = has_job if exists is None else exists
    wanted = tuple(wanted_seeds)

    def full(folds: Sequence[str], seeds: Sequence[int]) -> bool:
        return all(
            check(root=root, arm=a, fold=f, seed=s, rows=rows)
            for a in arms
            for f in folds
            for s in seeds
        )

    if full(core.ALL_FOLDS, wanted):
        return Plan(folds=core.ALL_FOLDS, seeds=wanted, interim=False)
    screen_seeds = tuple(s for s in core.SCREEN_SEEDS if s in wanted)
    if screen_ok and screen_seeds and full(core.SCREEN_FOLDS, screen_seeds):
        return Plan(folds=core.SCREEN_FOLDS, seeds=screen_seeds, interim=True)
    plan = (
        _partial_plan(arms=arms, wanted=wanted, rows=rows, check=check, root=root)
        if (partial_ok or rows != "test")
        else None
    )
    if plan is None and required:
        msg = (
            f"no complete plan for {[a[0] for a in arms]}: the full plan lacks stored forecasts "
            f"and the screen is {'allowed' if screen_ok else 'not allowed (it did not stop the rest)'}"
        )
        raise IncompletePlanError(msg)
    return plan


def _partial_plan(
    *, arms: Sequence[Arm], wanted: tuple[int, ...], rows: str, check: Callable[..., bool], root: Path
) -> Plan | None:  # fmt: skip
    """Return the folds and seeds that every arm holds, labelled interim, or None."""
    held = {
        f: {
            s
            for s in wanted
            if all(check(root=root, arm=a, fold=f, seed=s, rows=rows) for a in arms)
        }
        for f in core.ALL_FOLDS
    }
    folds = tuple(f for f, seeds in held.items() if seeds)
    if not folds:
        return None
    common = set.intersection(*(held[f] for f in folds))
    if not common:
        return None
    return Plan(folds=folds, seeds=tuple(sorted(common)), interim=True)


def validation_truth(*, root: Path) -> np.ndarray:
    """Return the cleaned actual power over capacity of the validation rows, `(16, grid)`, NaN elsewhere.

    The shared ground-truth file holds the test months only, so a validation score reads the
    `actual` column that each validation set carries (the cleaned training-window power).

    Args:
        root: This experiment's data folder.

    Returns:
        The stack, in the layout of experiment 713's `truth_stack`.

    Raises:
        sc.ScoreError: When the folder holds no validation set (the score would rest on no row).
    """
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    files = sorted((root / "predictions").glob("*_val/predictions.parquet"))
    if not files:
        msg = "no validation set exists, so there is no validation truth"
        raise sc.ScoreError(msg)
    frame = pl.concat(
        [pl.read_parquet(f, columns=["time_series_id", "valid_time", "actual"]) for f in files]
    ).unique(subset=["time_series_id", "valid_time"])
    sites = k578.sites()
    prim_of = dict(zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True))
    idx = (
        (frame["valid_time"].dt.epoch("us").to_numpy() - prep422.GRID_START_US) // prep422.STEP_US
    ).astype(int)
    prim = np.array([prim_of[int(t)] for t in frame["time_series_id"].to_list()])
    out = np.full((16, int(idx.max()) + 1), np.nan)
    out[prim, idx] = frame["actual"].to_numpy().astype(np.float64)
    return out


def read_arm(*, root: Path, arm: Arm, plan: Plan, rows: str) -> pl.DataFrame:
    """Return an arm's per-seed rows in contract origin and capacity shares, with a `seed` column.

    Args:
        root: This experiment's data folder.
        arm: The arm and its variant.
        plan: The folds and seeds.
        rows: "test" or "validation".

    Returns:
        The rows of every fold and seed of the plan.

    Raises:
        sc.ScoreError: When a set is missing or a (key, seed) is duplicated.
    """
    import e00713_score as s713  # ty: ignore[unresolved-import]

    name, variant = arm
    parts = []
    for fold in plan.folds:
        for seed in plan.seeds:
            if not has_job(root=root, arm=arm, fold=fold, seed=seed, rows=rows):
                msg = f"the forecast of {name} {fold} seed {seed} is missing"
                raise sc.ScoreError(msg)
            if name == "M*":
                frame = pl.read_parquet(
                    baselines.m_star_file(month=fold, seed=seed), columns=[*KEYS, *QCOLS]
                )
            else:
                frame = store.read_set(
                    root=root,
                    name=store.set_name(arm=name, fold=fold, seed=seed, variant=variant, rows=rows),
                    rows=rows,
                ).select(*KEYS, *QCOLS)
            parts.append(
                s713.to_contract_shares(frame=frame.with_columns(seed=pl.lit(seed, dtype=pl.Int64)))
            )
    out = pl.concat(parts)
    if out.select(*KEYS, "seed").is_duplicated().any():
        msg = f"{name} holds a duplicated (key, seed) row"
        raise sc.ScoreError(msg)
    return out


def arm_losses(*, root: Path, arm: Arm, plan: Plan, rows: str, truth: np.ndarray) -> pl.DataFrame:
    """Return an arm's seed mean with the loss columns (one row per key).

    Args:
        root: This experiment's data folder.
        arm: The arm and its variant.
        plan: The folds and seeds.
        rows: "test" or "validation".
        truth: Experiment 713's cleaned-truth stack.

    Returns:
        The frame, rows without a cleaned actual dropped.
    """
    frame = read_arm(root=root, arm=arm, plan=plan, rows=rows)
    mean = sc.seed_mean(frame=frame, seeds=plan.seeds)
    return sc.real_losses(frame=mean, truth=truth)


def restrict(
    *, frame: pl.DataFrame, ids: Sequence[int], headline: bool = True, extra: pl.Expr | None = None
) -> pl.DataFrame:  # fmt: skip
    """Return the rows of the Primaries `ids` at the headline lead (24 h up to 240 h)."""
    out = frame.filter(pl.col("time_series_id").is_in(list(ids)))
    if headline:
        lo, hi = core.HEADLINE_LEAD_HOURS
        out = out.filter((pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi))
    return out if extra is None else out.filter(extra)


def common_keys(*, a: pl.DataFrame, b: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the two frames cut to the keys that both hold (rows without an actual drop out alike).

    Args:
        a: The first frame.
        b: The second frame.

    Returns:
        The two frames, sorted by the keys.

    Raises:
        sc.ScoreError: When the arms share no key.
    """
    keys = a.select(*KEYS).join(b.select(*KEYS), on=list(KEYS), how="inner")
    if keys.is_empty():
        msg = "the two arms share no key"
        raise sc.ScoreError(msg)
    return (
        a.join(keys, on=list(KEYS), how="semi").sort(*KEYS),
        b.join(keys, on=list(KEYS), how="semi").sort(*KEYS),
    )


def contrast(
    *, treatment: pl.DataFrame, reference: pl.DataFrame, ids: Sequence[int], n_resamples: int,
    extra: pl.Expr | None = None, metric: str = "crps",
) -> sc.RealContrast | None:  # fmt: skip
    """Return the treatment-minus-reference contrast on the rows `ids` (None when no row remains)."""
    t = restrict(frame=treatment, ids=ids, extra=extra)
    r = restrict(frame=reference, ids=ids, extra=extra)
    if t.is_empty() or r.is_empty():
        return None
    t, r = common_keys(a=t, b=r)
    return sc.real_contrast(treatment=t, reference=r, metric=metric, n_resamples=n_resamples)


@dataclass
class Readings:
    """What `real_readings` returns.

    Attributes:
        contrasts: C1, C2, R1 to R4, and R12 by name (None when unread).
        plans: The plan of each contrast.
        guards_inputs: The mean differences that `guards_hold` reads (None when C1 is unread).
        r8: The lead-band, leave-one-month-out, midday, and cold-night readings of C1.
        r9: The tail pinball losses and coverages.
        r10: The load-shift scenarios.
    """

    contrasts: dict[str, sc.RealContrast | None]
    plans: dict[str, dict[str, Any] | None]
    guards_inputs: dict[str, float] | None
    r8: dict[str, Any]
    r9: dict[str, Any]
    r10: dict[str, Any]


CONTRAST_SPECS: Final = {
    # name: (treatment, reference, seeds, where)
    "C1": (("FU", None), ("M*", None), core.HEADLINE_SEEDS, "d8"),
    "C2": (("FU", None), ("FU-S", None), core.PLACEBO_SEEDS, "d8"),
    "R1": (("FS", None), ("M*", None), core.PLACEBO_SEEDS, "d8"),
    "R2": (("FU", None), ("FS", None), core.PLACEBO_SEEDS, "d8"),
    "R3": (("FU", None), ("M*", None), core.HEADLINE_SEEDS, "others"),
    "R4": (("FU", "masked"), ("FU", None), core.HEADLINE_SEEDS, "d8"),
    "R12": (("FU", "stored"), ("FU", None), core.HEADLINE_SEEDS, "d8"),
}
"""The real-fold contrasts: the arms, the seeds the plan reads, and the rows (D8 or the other 15)."""


def swap_reference(*, arm: Arm, rows: str) -> Arm:
    """Return M*'s stand-in on validation rows (F0r), and the arm itself otherwise."""
    return (VALIDATION_REFERENCE, None) if arm[0] == "M*" and rows == "validation" else arm


COUNTED_NAMES: Final = ("C1", "C2")
"""The real-fold contrasts that the verdict table reads."""


def real_readings(
    *, root: Path, rows: str, n_resamples: int | None = None, tolerate_missing: bool = False
) -> Readings:  # fmt: skip
    """Read every real-fold contrast, the three guards' inputs, and R8 to R10 from the stored sets.

    Args:
        root: This experiment's data folder.
        rows: "test" or "validation".
        n_resamples: The bootstrap resamples (default: the plan's 100,000, or 1,000 on validation
            rows, whose intervals are interim).
        tolerate_missing: Whether a counted contrast (C1 or C2) with an incomplete plan is read as
            None. The score stage sets it only when the chain stopped, a fit failed, or a
            correctness check failed, where the verdict row does not need the reading.

    Returns:
        The readings; a report-only contrast with no complete plan is None.

    Raises:
        IncompletePlanError: When C1 or C2 has no complete plan and `tolerate_missing` is false.
    """
    import e00713_score as s713  # ty: ignore[unresolved-import]

    n_boot = (
        n_resamples
        if n_resamples is not None
        else (core.N_RESAMPLES_REAL if rows == "test" else N_RESAMPLES_VALIDATION)
    )
    truth = s713.truth_stack() if rows == "test" else validation_truth(root=root)
    d8 = baselines.d8_series_id()
    import e00578_common as k578  # ty: ignore[unresolved-import]

    all_ids = [int(x) for x in k578.sites()["time_series_id"].to_list()]
    others = [i for i in all_ids if i != d8]
    cache: dict[tuple[Arm, Plan], pl.DataFrame] = {}

    def losses(arm: Arm, plan: Plan) -> pl.DataFrame:
        key = (arm, plan)
        if key not in cache:
            cache[key] = arm_losses(root=root, arm=arm, plan=plan, rows=rows, truth=truth)
        return cache[key]

    contrasts: dict[str, sc.RealContrast | None] = {}
    stops_rest = screen_stops_rest(root=root)
    plans: dict[str, dict[str, Any] | None] = {}
    frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
    for name, (treat, reference, seeds, where) in CONTRAST_SPECS.items():
        ref = swap_reference(arm=reference, rows=rows)
        counted = name in COUNTED_NAMES
        plan = choose_plan(
            root=root, arms=[treat, ref], wanted_seeds=seeds, rows=rows, screen_ok=stops_rest,
            required=counted and not tolerate_missing,
        )  # fmt: skip
        plans[name] = None if plan is None else plan.as_dict()
        if plan is None:
            contrasts[name] = None
            continue
        t, r = losses(treat, plan), losses(ref, plan)
        frames[name] = (t, r)
        contrasts[name] = contrast(
            treatment=t, reference=r, ids=[d8] if where == "d8" else others, n_resamples=n_boot
        )
    return Readings(
        contrasts=contrasts,
        plans=plans,
        guards_inputs=_guards_inputs(frames=frames, ids=[d8]),
        r8=reading_r8(frames=frames, contrasts=contrasts, d8=d8, n_resamples=n_boot),
        r9=reading_r9(root=root, rows=rows, d8=d8, truth=truth, losses=losses),
        r10=reading_r10(root=root, rows=rows, d8=d8, truth=truth, losses=losses),
    )


def mean_difference(*, t: pl.DataFrame, r: pl.DataFrame, column: str) -> float:
    """Return the mean of `t[column] - r[column]` on the keys both hold."""
    t, r = common_keys(a=t, b=r)
    return float((t[column] - r[column]).mean())  # ty: ignore[invalid-argument-type]


def _guards_inputs(
    *, frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]], ids: Sequence[int]
) -> dict[str, float] | None:
    """Return the mean differences of C1's absolute error and of the q01 and q99 pinball loss."""
    if "C1" not in frames:
        return None
    t, r = (restrict(frame=f, ids=ids) for f in frames["C1"])
    if t.is_empty() or r.is_empty():
        return None
    return {
        "mae": mean_difference(t=t, r=r, column="abs_err"),
        "q01": mean_difference(t=t, r=r, column="pin_q01"),
        "q99": mean_difference(t=t, r=r, column="pin_q99"),
    }


# --------------------------------------------------------------------------- R8


def leave_one_month_out(*, t: pl.DataFrame, r: pl.DataFrame) -> dict[str, float]:
    """Return C1's mean difference of CRPS with each fold month left out in turn.

    The month is the month of the native origin, which names the fold.
    """
    t, r = common_keys(a=t, b=r)
    d = (t["crps"] - r["crps"]).to_numpy()
    month = (t["power_fcst_init_time"] - timedelta(hours=3)).dt.strftime("%Y-%m").to_numpy()
    return {str(m): float(d[month != m].mean()) for m in np.unique(month) if (month != m).any()}


def reading_r8(
    *, frames: dict[str, tuple[pl.DataFrame, pl.DataFrame]], contrasts: dict[str, Any], d8: int,
    n_resamples: int,
) -> dict[str, Any]:  # fmt: skip
    """Return R8: C1 by lead band, with each month left out, at midday, and on cold winter nights.

    Args:
        frames: The seed-mean loss frames of each contrast (`C1` is read).
        contrasts: The contrasts (for the point of C1).
        d8: D8's `time_series_id`.
        n_resamples: The bootstrap resamples of each sub-reading.

    Returns:
        `bands` (days 2 to 3, 4 to 6, 7 to 10), `leave_one_month_out`, `midday` (10:00 to 14:00 UTC
        in folds 2026-04 to 2026-06), and `cold_winter_nights`; each is a contrast or "void" with
        the reason.
    """
    if "C1" not in frames or contrasts.get("C1") is None:
        return {"status": "void", "reason": "C1 is unread"}
    t, r = frames["C1"]
    out: dict[str, Any] = {"bands": {}}
    for lo, hi in core.BAND_LEAD_DAYS:
        lead = pl.col("lead_hours")
        extra = (lead >= 24.0 * (lo - 1)) & (lead < 24.0 * hi)
        c = contrast(treatment=t, reference=r, ids=[d8], n_resamples=n_resamples, extra=extra)
        out["bands"][f"days {lo} to {hi}"] = "void: no row" if c is None else c.as_dict()
    t8, r8 = restrict(frame=t, ids=[d8]), restrict(frame=r, ids=[d8])
    out["leave_one_month_out"] = leave_one_month_out(t=t8, r=r8)
    fold_month = (pl.col("power_fcst_init_time") - timedelta(hours=3)).dt.strftime("%Y-%m")
    hour = pl.col("valid_time").dt.hour()
    midday = (
        fold_month.is_in(list(core.LATE_SPRING_SUMMER_FOLDS))
        & (hour >= core.MIDDAY_HOURS_UTC[0])
        & (hour < core.MIDDAY_HOURS_UTC[1])
    )
    c = contrast(treatment=t, reference=r, ids=[d8], n_resamples=n_resamples, extra=midday)
    out["midday"] = "void: no row in folds 2026-04 to 2026-06" if c is None else c.as_dict()
    out["cold_winter_nights"] = cold_winter_nights(t=t8, r=r8, n_resamples=n_resamples)
    return out


def cold_winter_nights(*, t: pl.DataFrame, r: pl.DataFrame, n_resamples: int) -> Any:
    """Return C1 on the half-hours that are both a winter night and a cold night (the protocol's slices).

    The flags come from `studies.eval_protocol.slices`, with the ERA5 temperature and the sun's
    elevation of the feature cache. A fold with no such half-hour gives "void".
    """
    flags = slice_flags(rows=t.select(*KEYS), names=("winter_nights", "cold_nights"))
    if flags is None:
        return "void: the slice flags are unavailable"
    both = flags.filter(pl.col("winter_nights") & pl.col("cold_nights")).select(*KEYS)
    if both.is_empty():
        return "void: no winter night is cold in these rows"
    keep = both.join(t.select(*KEYS), on=list(KEYS), how="semi")
    c = sc.real_contrast(
        treatment=t.join(keep, on=list(KEYS), how="semi").sort(*KEYS),
        reference=r.join(keep, on=list(KEYS), how="semi").sort(*KEYS),
        n_resamples=n_resamples,
    )
    return c.as_dict()


def slice_flags(*, rows: pl.DataFrame, names: Sequence[str]) -> pl.DataFrame | None:
    """Return the named slice flags of the rows, from experiment 713's protocol inputs (None if unavailable)."""
    return protocol_flags(rows=rows, names=names)


def protocol_flags(*, rows: pl.DataFrame, names: Sequence[str]) -> pl.DataFrame | None:
    """Return the keys with the slice columns `names` (the cache, series, and events are built once)."""
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    from studies.dp_features import default_input_paths, load_or_build_feature_cache
    from studies.eval_protocol import slices
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    series = adapter.series_table()
    cache = load_or_build_feature_cache(
        paths=default_input_paths(), cache_dir=ML_EXPERIMENTS_DATA_DIR / FEATURE_CACHE
    )
    ids = [int(x) for x in rows["time_series_id"].unique().to_list()]
    mine = series.filter(pl.col("time_series_id").is_in(ids))
    first, last = rows["valid_time"].min(), rows["valid_time"].max()
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=mine,
        first=first,
        last=last + timedelta(minutes=30),  # ty: ignore[unsupported-operator]
    )
    del actuals
    shaped = rows.join(context, on=["time_series_id", "valid_time"], how="left")
    events = adapter.events_from_masks(cache=cache, series=mine)
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in mine.iter_rows()
    }
    flagged = slices.add_slice_columns(rows=shaped, events=events, first_readings=first_readings)
    return flagged.select(*KEYS, *names)


# --------------------------------------------------------------------------- R9 and R10


FEATURE_CACHE: Final = "00106-differentiable-physics-4-all-years-drift-and-recent-level/cache"
"""The feature cache of experiment 106, under the experiments' data folder (experiment 668 reads it)."""
R9_ARMS: Final = (("FU", None), ("FS", None), ("M*", None))


def tail_table(*, frame: pl.DataFrame) -> dict[str, Any]:
    """Return the mean q01, q05, q95, and q99 pinball loss and the coverage of each, with the row count."""
    out: dict[str, Any] = {"n": frame.height}
    for col in TAIL_COLUMNS:
        out[col] = float(frame[col].mean()) if frame.height else float("nan")  # ty: ignore[invalid-argument-type]
    for q in ("q01", "q05", "q95", "q99"):
        out[f"coverage_{q}"] = (
            float((frame["actual"] <= frame[q]).mean()) if frame.height else float("nan")  # ty: ignore[invalid-argument-type]
        )
    return out


def reading_r9(
    *, root: Path, rows: str, d8: int, truth: np.ndarray,
    losses: Callable[[Arm, Plan], pl.DataFrame],
) -> dict[str, Any]:  # fmt: skip
    """Return R9: the tail pinball losses and coverages of FU, FS, and M* at D8, and the curtailment cost.

    The avoidable-curtailment cost is defined only for the signed-MW Primaries, so D8 (an MVA-only
    Primary) carries the protocol's status instead of a cost.
    """
    from studies.eval_protocol import metrics

    del truth
    arms = [swap_reference(arm=a, rows=rows) for a in R9_ARMS]
    plan = choose_plan(root=root, arms=arms, wanted_seeds=core.PLACEBO_SEEDS, rows=rows)
    if plan is None:
        return {"status": "void", "reason": "FU, FS, and M* share no fold and seed"}
    out: dict[str, Any] = {"plan": plan.as_dict(), "arms": {}}
    for arm in arms:
        out["arms"][arm[0]] = tail_table(frame=restrict(frame=losses(arm, plan), ids=[d8]))
    out["avoidable_curtailment_cost"] = {
        "status": metrics.STATUS_UNSIGNED_METER,
        "note": "D8 reads unsigned apparent power, so the protocol defines no cost for it",
    }
    return out


def reading_r10(
    *, root: Path, rows: str, d8: int, truth: np.ndarray,
    losses: Callable[[Arm, Plan], pl.DataFrame],
) -> dict[str, Any]:  # fmt: skip
    """Return R10: the four load-shift scenarios at D8 for FU, FS, and M*.

    The scenarios are all rows, rows outside logged events and at least 7 days after one, rows inside
    logged events, and the 7 days after an event ends. NGED's logs may miss load transfers, so the
    rows outside events may hold events that no label covers, and the slices inside events undercount.
    """
    del truth
    arms = [swap_reference(arm=a, rows=rows) for a in R9_ARMS]
    plan = choose_plan(root=root, arms=arms, wanted_seeds=core.PLACEBO_SEEDS, rows=rows)
    if plan is None:
        return {"status": "void", "reason": "FU, FS, and M* share no fold and seed"}
    base = restrict(frame=losses(arms[0], plan), ids=[d8])
    flags = slice_flags(rows=base.select(*KEYS), names=("event_rows", "event_following_7_days"))
    if flags is None:
        return {"status": "void", "reason": "the event flags are unavailable"}
    flags = flags.with_columns(
        all=pl.lit(True),
        outside_events=~pl.col("event_rows") & ~pl.col("event_following_7_days"),
    )
    out: dict[str, Any] = {"plan": plan.as_dict(), "scenarios": {}}
    for scenario, column in SCENARIOS.items():
        keep = flags.filter(pl.col(column)).select(*KEYS)
        entry: dict[str, Any] = {"n": keep.height}
        for arm in arms:
            part = restrict(frame=losses(arm, plan), ids=[d8]).join(keep, on=list(KEYS), how="semi")
            entry[arm[0]] = {
                "n": part.height, "crps": float(part["crps"].mean()) if part.height else float("nan"),  # ty: ignore[invalid-argument-type]
                "mae": float(part["abs_err"].mean()) if part.height else float("nan"),  # ty: ignore[invalid-argument-type]
            }  # fmt: skip
        out["scenarios"][scenario] = entry
    return out


# --------------------------------------------------------------------------- the chain's two reads


def f0r_gate(*, root: Path, require_all: bool) -> dict[str, Any]:
    """Check that every fitted F0r reproduces M*'s stored forecast at D8 to 1e-6 of capacity.

    The F0r jobs are folds 2025-11 and 2026-05 at seeds 0 and 1. The gate reads the test-row sets,
    which the chain writes after each fit.

    Args:
        root: This experiment's data folder.
        require_all: Whether all four jobs must exist (the official run). A scratch run that fitted
            one job passes `False` and still needs at least one.

    Returns:
        The rows compared and the largest difference, over the jobs.

    Raises:
        StoppedError: When a job is missing, no job exists, or F0r differs from M* (verdict row 0).
    """
    from e00733_channels import StoppedError

    jobs = [(f, s) for f in core.F0R_FOLDS for s in core.F0R_SEEDS]
    present = [
        (f, s)
        for f, s in jobs
        if has_job(root=root, arm=("F0r", None), fold=f, seed=s, rows="test")
    ]
    if not present or (require_all and len(present) != len(jobs)):
        msg = f"the F0r gate found {len(present)} of {len(jobs)} F0r forecasts (verdict row 0)"
        raise StoppedError(msg)
    worst, rows = 0.0, 0
    for fold, seed in present:
        mine = store.read_set(
            root=root, name=store.set_name(arm="F0r", fold=fold, seed=seed), rows="test"
        )
        stored = baselines.read_m_star(month=fold, seed=seed)
        got = baselines.f0r_gate(f0r=mine, stored=stored)
        worst, rows = max(worst, float(got["largest_difference"])), rows + int(got["rows"])
    return {"jobs": len(present), "rows": rows, "largest_difference": worst}


def screen_point(*, root: Path, truth: np.ndarray | None = None) -> float:
    """Return the screen's relative CRPS point of FU minus M* at D8 (interim).

    The screen reads folds 2025-11, 2026-01, 2026-05, and 2026-06 at seeds 0 and 1. A scratch run
    reads the folds and seeds that exist.

    Args:
        root: This experiment's data folder.
        truth: Experiment 713's cleaned-truth stack (default: read it).

    Returns:
        The mean CRPS difference over M*'s mean CRPS, on D8's headline rows.

    Raises:
        StoppedError: When FU and M* share no fold and seed (the screen would read nothing).
    """
    import e00713_score as s713  # ty: ignore[unresolved-import]
    from e00733_channels import StoppedError

    arms = (("FU", None), ("M*", None))
    plan = choose_plan(
        root=root, arms=arms, wanted_seeds=core.SCREEN_SEEDS, rows="test", screen_ok=True
    )
    if plan is None:
        msg = "the screen found no fold and seed that FU and M* share"
        raise StoppedError(msg)
    stack = s713.truth_stack() if truth is None else truth
    d8 = baselines.d8_series_id()
    t, r = (
        restrict(frame=arm_losses(root=root, arm=a, plan=plan, rows="test", truth=stack), ids=[d8])
        for a in arms
    )
    t, r = common_keys(a=t, b=r)
    return float((t["crps"] - r["crps"]).mean() / r["crps"].mean())  # ty: ignore[invalid-argument-type, unsupported-operator]

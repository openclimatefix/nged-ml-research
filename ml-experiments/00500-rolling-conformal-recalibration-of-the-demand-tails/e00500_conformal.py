"""Experiment 500: rolling conformal recalibration of A-pool's tails, scored on experiment 483's rows.

Commands, in order (each reads what the earlier one wrote under
`data/ml-experiments/00500-*/work/`; nothing is fitted):

- `mde`: print the minimum detectable effect of the stored single seeds, before any arm is scored.
- `history`: pool experiment 425's five arm-A seeds for all 12 months (2025-07 to 2026-06), join the
  delivered actuals, and write the residual (actual minus quantile) of every forecast row.
- `check`: confirm the pooled history reproduces experiment 483's cache on the scored rows.
- `shifts`: for every scored origin, Primary, and lead band, the empirical quantile of the
  residuals of the last 7, 14, or 28 days of delivered forecasts, at each of the 11 levels.
- `arms`: write one frame per arm (raw, widenings, table P, two constants, rolling conformal).
- `score`: write the tails, contrasts, and per-fold coverage.

An actual is delivered 6 hours after its half-hour, so the buffer at origin `o` (a grid index, one
step being 30 minutes) holds forecast rows with target index at most `o - DELAY_STEPS` and above
`o - DELAY_STEPS - 48 * days`. Residuals come from experiment 425's out-of-fold predictions for
2025-07 to 2026-06, each scored with a model fitted before its month; no in-sample forecast enters.

```bash
source /tmp/rc_env.sh   # DATA_PATH_INTERNAL and a PYTHONPATH of every ml-experiments folder
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00500-*/e00500_conformal.py history
```
"""

# ruff: noqa: E501, PLR0912, PLR0915, PT018

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00483_tails as tails  # ty: ignore[unresolved-import]
import e00492_two_constant as two  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import lead_band_expr, widen  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00500-rolling-conformal-recalibration-of-the-demand-tails" / "work"
)
E425_A: Final[Path] = heads470.E425 / "seed_preds" / "A"
ALL_MONTHS: Final[tuple[str, ...]] = (
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
FOLDS: Final[tuple[str, ...]] = tails.MONTHS
LEVELS: Final[tuple[str, ...]] = tails.LEVELS
TAUS: Final[tuple[float, ...]] = tuple(models.LEVELS)
DELAY_STEPS: Final[int] = 12
"""An actual is delivered 6 hours (12 half-hour steps) after its half-hour."""
STEPS_PER_DAY: Final[int] = 48
BUFFER_DAYS: Final[tuple[int, ...]] = (7, 14, 28)
MIN_RESIDUALS: Final[int] = 100
"""A buffer cell (Primary, lead band, origin) with fewer residuals leaves the forecast unchanged."""
FINAL_WIDENING: Final[float] = 1.05
RC_WIDENED_DAYS: Final[int] = 14
RESAMPLES: Final[int] = 20_000
BONFERRONI_COUNT: Final[int] = 8
SEED: Final[int] = 500
JULY_2026: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No scored valid time is on or after this date, which is kept for a finalist."""
CANDIDATES: Final[tuple[str, ...]] = ("RC7", "RC14", "RC28", "RC14+w1.05")
FIVE_FOLD_ARMS: Final[tuple[str, ...]] = (
    "raw", "w1.15", "P", "two-constant", "RC7", "RC14", "RC28", "RC14+w1.05",
)  # fmt: skip
POST_HOC_ARMS: Final[tuple[str, ...]] = ("RC14-bias", "RC28-bias")
"""Written by `e00500_posthoc.py`: the buffer's median shift applied to all 11 quantiles."""
FOUR_FOLD_EXTRA: Final[tuple[str, ...]] = ("P-earlier", "two-constant-earlier")
GROUPS: Final[tuple[str, ...]] = ("signed_six", "mva_eight", "report_only")


def history_path(*, month: str) -> Path:
    """Return where a month's residual history is stored."""
    return WORK / "history" / f"{month}.parquet"


def shifts_path(*, days: int) -> Path:
    """Return where the buffer shifts for a buffer length are stored."""
    return WORK / "shifts" / f"shifts_{days}d.parquet"


def arm_path(*, arm: str) -> Path:
    """Return where an arm's frame is stored."""
    return WORK / "arms" / f"{arm}.parquet"


def load_base() -> pl.DataFrame:
    """Return A-pool's cached rows from experiment 483 without the half-hours of July 2026.

    Experiment 483's 2026-06 fold holds 9 hours of valid times on 2026-07-01 (leads of 39 hours or
    more), which the rule that keeps July 2026 for a finalist excludes; every arm here drops them.
    """
    return tails.load(arm="A-pool").filter(pl.col("valid_time") < JULY_2026)


def mde() -> None:
    """Print the minimum detectable effect from experiment 470's five single seeds (clean rows)."""
    table = score470.load_rows()
    records = []
    for group in ("signed_six", "mva_eight"):
        part = table.filter(
            (pl.col("group") == group) & pl.col("gam14") & (pl.col("cls") == "clean")
        )
        for metric in ("crps", "cov_q01", "cov_q05", "cov_q95", "cov_q99", "pin_q99"):
            columns = [f"A-s{s}__{metric}" for s in build470.POOL_SEEDS]
            values = np.array(
                [[part.filter(pl.col("month") == m)[c].mean() for c in columns] for m in FOLDS]
            )
            folds, k = values.shape
            resid = values - values.mean(axis=1, keepdims=True)
            sd = math.sqrt((resid**2).sum() / (folds * (k - 1)))
            records.append(
                {"group": group, "metric": metric, "seed_sd": sd,
                 "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(k)}
            )  # fmt: skip
    out = pl.DataFrame(records)
    WORK.mkdir(parents=True, exist_ok=True)
    out.write_csv(WORK / "mde.csv")
    with pl.Config(tbl_rows=30):
        print(out.with_columns(pl.col("seed_sd", "mde").round(5)))


def history() -> None:
    """Write every forecast row's residuals (actual minus quantile, capacity fractions)."""
    sites = build470.site_map()
    ctx = prep425.cleaned_scoring_context(series=adapter.series_table())
    actuals = (
        ctx["actuals"]
        .join(sites.select("time_series_id", "capacity_mw"), on="time_series_id")
        .select(
            "time_series_id",
            "valid_time",
            actual=(pl.col("actual") / pl.col("capacity_mw")).cast(pl.Float32),
        )
    )
    (WORK / "history").mkdir(parents=True, exist_ok=True)
    for month in ALL_MONTHS:
        pooled = build470.pool_stored(folder=E425_A, month=month, sites=sites)
        frame = pooled.join(actuals, on=["time_series_id", "valid_time"], how="inner")
        frame = frame.with_columns(
            lead_hours=((pl.col("target") - pl.col("origin")) / 2.0).cast(pl.Float32)
        ).with_columns(lead_band=lead_band_expr(column="lead_hours"))
        kept = frame.filter(pl.col("lead_band").is_not_null() & pl.col("actual").is_finite())
        out = kept.select(
            "primary",
            pl.col("origin").alias("init"),
            "target",
            "lead_band",
            pl.col("actual"),
            *[(pl.col("actual").cast(pl.Float64) - pl.col(q)).cast(pl.Float32).alias(f"r{q[1:]}")
              for q in LEVELS],
            *[pl.col(q).cast(pl.Float32) for q in LEVELS],
        )  # fmt: skip
        out.write_parquet(history_path(month=month))
        print(f"{month}: {pooled.height} forecast rows, {out.height} with a delivered actual",
              flush=True)  # fmt: skip


def check() -> None:
    """Confirm the pooled history reproduces experiment 483's cache (quantiles and actuals)."""
    for month in FOLDS:
        cache = load_base().filter(pl.col("month") == month)
        hist = pl.read_parquet(history_path(month=month)).rename({"init": "origin"})
        joined = cache.join(
            hist.select("primary", "origin", "target", "actual", "q01", "q50", "q95", "q99"),
            on=["primary", "origin", "target"],
            how="inner",
            suffix="_h",
        )
        assert joined.height == cache.height, (month, joined.height, cache.height)
        dq = (joined["q50"] - joined["q50_h"].cast(pl.Float64)).abs().max()
        da = (joined["actual"].cast(pl.Float64) - joined["actual_h"].cast(pl.Float64)).abs().max()
        for q in ("q01", "q95", "q99"):
            dd = (joined[q] - joined[f"{q}_h"].cast(pl.Float64)).abs().max()
            assert dd < 1e-4, (month, q, dd)  # ty: ignore[unsupported-operator]
        assert dq < 1e-4 and da < 1e-4, (month, dq, da)  # ty: ignore[unsupported-operator]
        print(f"{month}: {cache.height} cache rows match the history (max |dq50| {dq:.2e}, "
              f"|dactual| {da:.2e})", flush=True)  # fmt: skip


def load_history() -> pl.DataFrame:
    """Return the residual history of all 12 months."""
    return pl.concat([pl.read_parquet(history_path(month=m)) for m in ALL_MONTHS])


def shifts() -> None:
    """Write the shift of each level per origin, Primary, and lead band, for each buffer length."""
    hist = load_history().sort("target")
    origins = sorted(load_base()["origin"].unique().to_list())
    (WORK / "shifts").mkdir(parents=True, exist_ok=True)
    for days in BUFFER_DAYS:
        parts = []
        for origin in origins:
            high = origin - DELAY_STEPS
            low = high - STEPS_PER_DAY * days
            window = hist.filter((pl.col("target") <= high) & (pl.col("target") > low))
            grouped = window.group_by("primary", "lead_band").agg(
                pl.len().alias("n"),
                pl.col("init").min().alias("init_min"),
                pl.col("init").max().alias("init_max"),
                *[
                    pl.col(f"r{q[1:]}").cast(pl.Float64).quantile(tau, interpolation="linear")
                    .alias(f"s{q[1:]}")
                    for q, tau in zip(LEVELS, TAUS, strict=True)
                ],
            )  # fmt: skip
            assert window["target"].max() <= high and window["init"].max() < origin, origin
            parts.append(grouped.with_columns(origin=pl.lit(origin, dtype=pl.Int32)))
        out = pl.concat(parts)
        out.write_parquet(shifts_path(days=days))
        print(f"{days} days: {out.height} buffer cells, smallest n {out['n'].min()}, "
              f"{out.filter(pl.col('n') < MIN_RESIDUALS).height} below {MIN_RESIDUALS}",
              flush=True)  # fmt: skip


def apply_shift(*, frame: pl.DataFrame, days: int) -> tuple[pl.DataFrame, dict[str, float]]:
    """Return the frame shifted by the buffer's quantiles, sorted per row, and diagnostics."""
    sh = pl.read_parquet(shifts_path(days=days)).select(
        "origin", "primary", "lead_band", "n", *[f"s{q[1:]}" for q in LEVELS]
    )
    joined = frame.join(
        sh, on=["origin", "primary", "lead_band"], how="left", maintain_order="left"
    )
    usable = (pl.col("n") >= MIN_RESIDUALS).fill_null(False)
    shifted = joined.with_columns(
        *[
            (pl.col(q) + pl.when(usable).then(pl.col(f"s{q[1:]}")).otherwise(0.0)).alias(q)
            for q in LEVELS
        ]
    )
    q = shifted.select(list(LEVELS)).to_numpy()
    crossed = float((np.diff(q, axis=1) < 0).any(axis=1).mean())
    q = np.sort(q, axis=1)
    out = shifted.with_columns(*[pl.Series(name, q[:, i]) for i, name in enumerate(LEVELS)])
    diagnostics = {
        "rows": float(out.height),
        "unchanged_rows": float(joined.select((~usable).sum()).item()),
        "sorted_share": crossed,
    }
    return out.drop("n", *[f"s{q[1:]}" for q in LEVELS]), diagnostics


def fit_p(*, base: pl.DataFrame, scheme: str) -> pl.DataFrame:
    """Return table P applied to each held fold, fitted on the other folds or on earlier ones."""
    tagged = base.with_columns(group_orig=pl.col("group"))
    months = FOLDS[1:] if scheme == "earlier" else FOLDS
    parts = []
    for month in months:
        before = FOLDS[: FOLDS.index(month)] if scheme == "earlier" else None
        fit = tagged.filter(
            (pl.col("month").is_in(list(before)) if before else (pl.col("month") != month))
            & (pl.col("cls") == "clean")
        )
        factors = tails.fit_with_fallback(fit=fit, cold_class=False)
        held = tagged.filter(pl.col("month") == month)
        parts.append(prov.apply_factors(frame=held, factors=factors))
    return pl.concat(parts).drop("group_orig")


def fit_two(*, base: pl.DataFrame, scheme: str) -> pl.DataFrame:
    """Return experiment 492's two constants applied to each held fold, on every row class."""
    months = FOLDS[1:] if scheme == "earlier" else FOLDS
    parts = []
    for month in months:
        before = FOLDS[: FOLDS.index(month)] if scheme == "earlier" else None
        fit = base.filter(
            pl.col("month").is_in(list(before)) if before else (pl.col("month") != month)
        )
        constants = two.fit_constants(fit=fit)
        held = base.filter(pl.col("month") == month)
        parts.append(two.apply_constants(frame=held, constants=constants))
    return pl.concat(parts)


def arms() -> None:
    """Write every arm's frame (keys, flags, actuals, and the 11 quantiles) on A-pool's rows."""
    base = load_base()
    (WORK / "arms").mkdir(parents=True, exist_ok=True)
    frames: dict[str, pl.DataFrame] = {
        "raw": base,
        "w1.15": widen(frame=base, factor=1.15),
        "P": fit_p(base=base, scheme="lofo"),
        "P-earlier": fit_p(base=base, scheme="earlier"),
        "two-constant": fit_two(base=base, scheme="lofo"),
        "two-constant-earlier": fit_two(base=base, scheme="earlier"),
    }
    diagnostics: dict[str, dict[str, float]] = {}
    for days in BUFFER_DAYS:
        frames[f"RC{days}"], diagnostics[f"RC{days}"] = apply_shift(frame=base, days=days)
    frames["RC14+w1.05"] = widen(frame=frames["RC14"], factor=FINAL_WIDENING)
    for days in (7, 28):
        frames[f"RC{days}+w1.05"] = widen(frame=frames[f"RC{days}"], factor=FINAL_WIDENING)
    for name, frame in frames.items():
        frame.sort(*tails.KEYS).write_parquet(arm_path(arm=name))
        print(name, frame.height, flush=True)
    (WORK / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


def load_arm(*, arm: str) -> pl.DataFrame:
    """Return an arm's frame."""
    return pl.read_parquet(arm_path(arm=arm))


def row_values(*, frame: pl.DataFrame, actual: str = "actual") -> pl.DataFrame:
    """Return per-row CRPS, coverage, and pinball loss (points) as experiment 483 scores them."""
    m = score470.row_metrics(frame=frame, actual=actual)
    return pl.DataFrame({k: v for k, v in m.items() if k == "crps" or k[:3] in ("cov", "pin")})


def interval(
    *,
    values: np.ndarray,
    base: np.ndarray | None,
    blocks: np.ndarray,
    levels: tuple[float, ...],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Return the estimate and block-bootstrap bounds of mean(values) (or its ratio to `base`)."""
    ids, inv = np.unique(blocks, return_inverse=True)
    numerator = np.bincount(inv, weights=values, minlength=len(ids))
    denominator = (
        np.bincount(inv, weights=base, minlength=len(ids))
        if base is not None
        else np.bincount(inv, minlength=len(ids)).astype(np.float64)
    )
    pick = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
    draws = numerator[pick].sum(axis=1) / denominator[pick].sum(axis=1)
    out = {"estimate": float(numerator.sum() / denominator.sum())}
    for level in levels:
        tail = (100.0 - level) / 2.0
        out[f"low_{level}"] = float(np.percentile(draws, tail))
        out[f"high_{level}"] = float(np.percentile(draws, 100.0 - tail))
    return out


def score() -> None:
    """Write absolute scores, per-fold scores, contrasts against raw, and descriptive contrasts."""
    level_bonferroni = 100.0 * (1.0 - 0.05 / BONFERRONI_COUNT)
    levels = (95.0, level_bonferroni)
    rng = np.random.default_rng(SEED)
    five = {
        a: load_arm(arm=a) for a in (*FIVE_FOLD_ARMS, "RC7+w1.05", "RC28+w1.05", *POST_HOC_ARMS)
    }
    four = {a: load_arm(arm=a) for a in FOUR_FOLD_EXTRA}
    four_months = FOLDS[1:]
    for name, frame in five.items():
        assert frame.select(tails.KEYS).equals(five["raw"].select(tails.KEYS)), name
    scores: list[pl.DataFrame] = []
    sets: dict[str, dict[str, pl.DataFrame]] = {
        "five": five,
        "four": {
            **{a: f.filter(pl.col("month").is_in(four_months)) for a, f in five.items()},
            **four,
        },
    }
    for set_name, arm_frames in sets.items():
        ref = arm_frames["raw"].select(tails.KEYS)
        for a, f in arm_frames.items():
            assert f.select(tails.KEYS).equals(ref), (set_name, a)
        for arm, frame in arm_frames.items():
            for slice_name, part in tails.slices(frame=frame).items():
                for by in (["group"], ["group", "month"], ["group", "lead_band"]):
                    s = tails.summarise(frame=part, by=by)
                    scores.append(
                        s.with_columns(
                            rowset=pl.lit(set_name), arm=pl.lit(arm), slice=pl.lit(slice_name),
                            by=pl.lit("+".join(by)),
                        )
                    )  # fmt: skip
    pl.concat(scores, how="diagonal").write_parquet(WORK / "scores.parquet")

    contrasts: list[dict] = []
    coverage: list[dict] = []
    for set_name, arm_frames in sets.items():
        clean = {a: tails.slices(frame=f)["clean"] for a, f in arm_frames.items()}
        for group in GROUPS:
            ref = clean["raw"].filter(pl.col("group") == group)
            if ref.is_empty():
                continue
            blocks = (ref["month"] + "_" + ref["day"].cast(pl.String)).to_numpy()
            raw_values = row_values(frame=ref)
            for arm, frame in clean.items():
                part = frame.filter(pl.col("group") == group)
                values = row_values(frame=part)
                for metric in values.columns:
                    if metric.startswith("cov"):
                        c = interval(values=values[metric].to_numpy(), base=None, blocks=blocks,
                                     levels=levels, rng=rng)  # fmt: skip
                        coverage.append({"rowset": set_name, "group": group, "arm": arm,
                                         "metric": metric, **c})  # fmt: skip
                if arm == "raw":
                    continue
                for metric in ("crps", "pin_q99"):
                    diff = values[metric].to_numpy() - raw_values[metric].to_numpy()
                    d = interval(values=diff, base=None, blocks=blocks, levels=levels, rng=rng)
                    rel = interval(values=diff, base=raw_values[metric].to_numpy(), blocks=blocks,
                                   levels=levels, rng=rng)  # fmt: skip
                    contrasts.append(
                        {"rowset": set_name, "group": group, "arm": arm, "against": "raw",
                         "metric": metric, "counted": arm in CANDIDATES and group == "signed_six"
                         and set_name == "five", "n": part.height,
                         "difference": d["estimate"], "low_95": d["low_95.0"],
                         "high_95": d["high_95.0"], "low_bonferroni": d[f"low_{level_bonferroni}"],
                         "high_bonferroni": d[f"high_{level_bonferroni}"],
                         "relative": rel["estimate"],
                         "rel_low_bonferroni": rel[f"low_{level_bonferroni}"],
                         "rel_high_bonferroni": rel[f"high_{level_bonferroni}"]}
                    )  # fmt: skip
                    if arm in CANDIDATES and group == "signed_six":
                        wide_blocks = (
                            part["month"] + "_" + (part["day"] // 14).cast(pl.String)
                        ).to_numpy()
                        w = interval(values=diff, base=raw_values[metric].to_numpy(),
                                     blocks=wide_blocks, levels=(level_bonferroni,), rng=rng)  # fmt: skip
                        contrasts[-1] |= {
                            "rel_low_bonferroni_14d_blocks": w[f"low_{level_bonferroni}"],
                            "rel_high_bonferroni_14d_blocks": w[f"high_{level_bonferroni}"],
                        }
        # Descriptive contrasts of the candidates against the other recalibrations, six only.
        ref_arms = ("w1.15", "P", "two-constant") + (
            ("P-earlier", "two-constant-earlier") if set_name == "four" else ()
        )
        group = "signed_six"
        base_part = clean["raw"].filter(pl.col("group") == group)
        blocks = (base_part["month"] + "_" + base_part["day"].cast(pl.String)).to_numpy()
        for arm in CANDIDATES:
            va = row_values(frame=clean[arm].filter(pl.col("group") == group))
            for other in ref_arms:
                vb = row_values(frame=clean[other].filter(pl.col("group") == group))
                for metric in ("crps", "pin_q99"):
                    diff = va[metric].to_numpy() - vb[metric].to_numpy()
                    d = interval(values=diff, base=None, blocks=blocks, levels=(95.0,), rng=rng)
                    rel = interval(values=diff, base=vb[metric].to_numpy(), blocks=blocks,
                                   levels=(95.0,), rng=rng)  # fmt: skip
                    contrasts.append(
                        {"rowset": set_name, "group": group, "arm": arm, "against": other,
                         "metric": metric, "counted": False, "n": base_part.height,
                         "difference": d["estimate"], "low_95": d["low_95.0"],
                         "high_95": d["high_95.0"], "relative": rel["estimate"],
                         "rel_low_95": rel["low_95.0"], "rel_high_95": rel["high_95.0"]}
                    )  # fmt: skip
        print("scored", set_name, flush=True)
    pl.DataFrame(contrasts, infer_schema_length=None).write_csv(WORK / "contrasts.csv")
    pl.DataFrame(coverage).write_csv(WORK / "coverage_intervals.csv")
    counted = sum(1 for c in contrasts if c["counted"])
    print(f"counted tests: {counted} (pre-registered {BONFERRONI_COUNT})")
    assert counted == BONFERRONI_COUNT, counted


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("mde", "history", "check", "shifts", "arms", "score"))
    args = parser.parse_args()
    {"mde": mde, "history": history, "check": check, "shifts": shifts, "arms": arms,
     "score": score}[args.command]()  # fmt: skip


if __name__ == "__main__":
    main()

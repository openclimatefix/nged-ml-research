# ruff: noqa: E501
"""Score experiment 391's arms under demand-v1 and print every number the page quotes.

Arms (from `e00391_assemble.py`): `d` (the stored weekly member, "weekly, no widening"), `b0`,
`n0`, `n1`, `n2`, `n2_noveto`, `n3`, `o`, `n1_era5` (our monthly refits, no widening), each raw
(`<arm>`) and widened by the fixed factor 1.15 about the median (`<arm>_w`). References `bg2`,
`manual_heuristic`, and `climatology` are experiment 87's stored arms, raw and `_w`. `bg2` already
carries the spread factor 1.15 in its members; a blend of widened members cannot be unwidened
exactly, so `bg2_u` (the stored `bg2` with its quantiles divided back by 1.15 about the median) is
an approximation and is labelled so. The scoring machinery is experiment 379's.

Primary slice: contaminated-lag rows (a lag inside a labelled event) outside the events, days 1
to 10. Prints, in order: row and event counts per slice; absolute scores (CRPS, MAE of the median,
pinball loss and coverage at q01, q05, q95, q99, 5-95 and 1-99 band coverage and width, and the
error of the mean forecast) on the primary slice, on clean rows, and on every secondary slice;
the median-minus-mean forecast per site; the seven pre-registered contrasts (C1 to C7, Bonferroni
level 99.29%) on the primary slice with event blocks and on the secondary slices, raw and
widened; the decision-rule verdicts; the O-minus-b0 gap and P1 to P4 as absolute changes in CRPS
points; per-Primary and leave-one-Primary-out contrasts; and the avoidable-curtailment cost.
Event slices bootstrap whole events as blocks; the other slices bootstrap calendar months.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\
ml-experiments/00391-xgboost-lag-surprise-second-stage \
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00391-*/e00391_score.py \
    [--smoke] [--resamples 5000]
```
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00379_score as ref379  # ty: ignore[unresolved-import]
import e00391_assemble as assemble
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import KEY_COLUMNS, SIGNED_MW_LABELS

WORK: Final[Path] = assemble.WORK_DIR
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
MAX_LEAD_HOURS: Final[float] = 336.0
HEADLINE_BAND: Final[str] = "lead_days_1_10"
REFERENCE_ARMS: Final[tuple[str, ...]] = ("bg2", "manual_heuristic", "climatology")
APPROX_UNWIDENED: Final[str] = "bg2_u"
PLACEBOS: Final[tuple[str, ...]] = ("D2", "D4")
N_CONTRASTS: Final[int] = 7
BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_CONTRASTS)
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "n0", "b0"), ("C2", "n1", "n0"), ("C3", "n2", "n1"), ("C4", "n3", "n1"),
    ("C5", "o", "n2"), ("C6", "n1_era5", "n1"), ("C7", "n2", "n2_noveto"),
)  # fmt: skip
FULL_METRICS: Final[tuple[str, ...]] = ("crps", "mae_median", "pinball_tail_up", "mean_err_abs")
SHORT_METRICS: Final[tuple[str, ...]] = ("crps", "mean_err_abs")
LEAD_BANDS_HOURS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0), ("24-72 h", 24.0, 72.0), ("72-168 h", 72.0, 168.0),
    ("168-240 h", 168.0, 240.0),
)  # fmt: skip
GROUPS: Final[tuple[str, ...]] = ("signed-MW", "MVA-only", "short-history")
HARM_TOLERANCE: Final[float] = 0.02
EQUIVALENCE_MARGIN: Final[float] = 0.03
EVENT_ATTRIBUTION_DAYS: Final[int] = 8
"""A contaminated-lag row belongs to the event whose window `[start, end + 8 days]` holds it."""
CURTAILMENT_FRACTION: Final[float] = 0.9
SUMMARY_METRICS: Final[tuple[str, ...]] = (
    "crps", "mae_median", "pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99",
    "pinball_tail_up", "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99",
    "band_90", "band_98", "width_90", "width_98", "mean_err_abs", "median_minus_mean",
)  # fmt: skip


def mean_weights() -> np.ndarray:
    """Return the weights of the 11 quantiles in the mean forecast (they sum to 1)."""
    levels = metrics.LEVELS
    weights = np.empty(len(levels))
    weights[0] = levels[0] + (levels[1] - levels[0]) / 2.0
    weights[-1] = (1.0 - levels[-1]) + (levels[-1] - levels[-2]) / 2.0
    weights[1:-1] = (levels[2:] - levels[:-2]) / 2.0
    return weights


def load_arms(*, series: pl.DataFrame, arms_dir: Path) -> tuple[dict[str, pl.DataFrame], list[str]]:
    """Return every arm's forecast frame in MW, leads to `MAX_LEAD_HOURS`, and the stored arm names.

    The own arms are the files of `arms_dir`; the references are experiment 87's stored arms, each
    also widened by 1.15 (`<arm>_w`), with `bg2_u` the approximate unwidened `bg2`.
    """
    ids = series["time_series_id"].to_list()
    out: dict[str, pl.DataFrame] = {}
    order = [a for base in (assemble.STORED_ARM, *assemble.OWN_ARMS) for a in (base, f"{base}_w")]
    for name in order:
        path = arms_dir / f"{name}.parquet"
        if not path.exists():
            continue
        stored = pl.read_parquet(path).filter(pl.col("lead_hours") <= MAX_LEAD_HOURS)
        out[name] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    references = []
    for name in REFERENCE_ARMS:
        file = ref379.STORED[name]
        stored = pl.read_parquet(ref379.STORED_87 / f"{file}.parquet").filter(
            pl.col("time_series_id").is_in(ids), pl.col("lead_hours") <= MAX_LEAD_HOURS
        )
        frame = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
        out[name] = frame
        out[f"{name}_w"] = assemble.widen(frame=frame, rows=True, factor=assemble.FIXED_K)
        references += [name, f"{name}_w"]
        if name == "bg2":
            out[APPROX_UNWIDENED] = assemble.widen(
                frame=frame, rows=True, factor=1.0 / assemble.FIXED_K
            )
            references.append(APPROX_UNWIDENED)
    return out, references


def add_band_columns(
    *, rows: pl.DataFrame, forecast: pl.DataFrame, series: pl.DataFrame
) -> pl.DataFrame:
    """Add band coverage (0 or 1), band width, tail pinball, and the mean-forecast columns (points).

    `mean_err_abs` is the absolute error of the mean forecast (the trapezoid mean of the 11
    quantiles) and `median_minus_mean` is the median forecast minus the mean forecast.
    """
    weights = mean_weights()
    scale = 100.0 / pl.col("capacity_mw")
    mean_mw = pl.sum_horizontal(
        w * pl.col(c) for w, c in zip(weights, metrics.QUANTILE_COLUMNS, strict=True)
    )
    extra = forecast.join(
        series.select("time_series_id", "capacity_mw"), on="time_series_id"
    ).select(
        *KEY_COLUMNS,
        width_90=(pl.col("q95") - pl.col("q05")) * scale,
        width_98=(pl.col("q99") - pl.col("q01")) * scale,
        mean_points=mean_mw * scale,
        median_points=pl.col("q50") * scale,
    )
    return (
        rows.join(extra, on=list(KEY_COLUMNS), how="left", validate="1:1")
        .with_columns(
            band_90=pl.col("coverage_q95") - pl.col("coverage_q05"),
            band_98=pl.col("coverage_q99") - pl.col("coverage_q01"),
            pinball_tail_up=(pl.col("pinball_q95") + pl.col("pinball_q99")) / 2.0,
            mean_err_abs=(pl.col("actual_points") - pl.col("mean_points")).abs(),
            median_minus_mean=pl.col("median_points") - pl.col("mean_points"),
        )
        .drop("mean_points", "median_points")
    )


def summarise(*, rows: dict[str, pl.DataFrame], where: pl.Expr, by: str = "group") -> pl.DataFrame:
    """Return, per `by` value and arm, the mean losses and band measures over the filtered rows."""
    pieces = [
        frame.filter(where)
        .group_by(by)
        .agg(pl.len().alias("n"), *[pl.col(m).mean().alias(m) for m in SUMMARY_METRICS])
        .with_columns(arm=pl.lit(name))
        for name, frame in rows.items()
    ]
    return pl.concat(pieces).sort(by, "arm")


def event_ids(*, table: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Return `KEY_COLUMNS` and `event` (an index into `events`, -1 if none) for the event rows.

    Candidates are the rows during an event, in the 7 days after one, or with a lag inside one.
    Each is attributed to the earliest-starting event of its own series whose window
    `[start, end + EVENT_ATTRIBUTION_DAYS]` holds its valid time.
    """
    candidates = table.filter(
        pl.col("event_rows") | pl.col("event_following_7_days") | pl.col("lag_in_event")
    ).select(*KEY_COLUMNS)
    after = pl.duration(days=EVENT_ATTRIBUTION_DAYS)
    matched = (
        candidates.join(
            events.select("time_series_id", "start", "end", "event"), on="time_series_id"
        )
        .filter(
            (pl.col("valid_time") >= pl.col("start"))
            & (pl.col("valid_time") <= pl.col("end") + after)
        )
        .sort("start")
        .unique(subset=list(KEY_COLUMNS), keep="first")
        .select(*KEY_COLUMNS, "event")
    )
    return candidates.join(matched, on=list(KEY_COLUMNS), how="left").with_columns(
        pl.col("event").fill_null(-1)
    )


def event_block_contrast(
    *,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    metric: str,
    where: pl.Expr,
    level_percent: float,
    n_resamples: int,
) -> dict[str, float]:
    """Return the paired contrast with an interval from resampling whole events as blocks.

    Rows with no event (`event == -1`) form one extra block and are not counted in `n_events`.
    """
    keys = list(KEY_COLUMNS)
    shared = (
        treatment.filter(where)
        .select(*keys, "event", metric)
        .join(
            reference.select(*keys, reference=pl.col(metric)), on=keys, how="inner", validate="1:1"
        )
    )
    if shared.is_empty():
        msg = "the two forecasts share no half-hour in the filter"
        raise ValueError(msg)
    blocks = (
        shared.group_by("event")
        .agg(total=(pl.col(metric) - pl.col("reference")).sum(), count=pl.len())
        .sort("event")
    )
    totals, counts = blocks["total"].to_numpy(), blocks["count"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(totals), size=(n_resamples, len(totals)))
    stats = totals[pick].sum(axis=1) / counts[pick].sum(axis=1)
    tail = (100.0 - level_percent) / 2.0
    lower, upper = np.percentile(stats, [tail, 100.0 - tail])
    return {
        "difference": float(totals.sum() / counts.sum()),
        "lower": float(lower),
        "upper": float(upper),
        "n_half_hours": int(counts.sum()),
        "n_months": -1,
        "n_events": int((blocks["event"] >= 0).sum()),
    }


def contrast(
    *,
    rows: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    metric: str,
    where: pl.Expr,
    blocks: str,
    level_percent: float,
    n_resamples: int,
) -> dict[str, float] | None:
    """Return a contrast by event blocks or month blocks, or `None` if the filter holds no row."""
    try:
        if blocks == "event":
            return event_block_contrast(
                treatment=rows[treatment], reference=rows[reference], metric=metric, where=where,
                level_percent=level_percent, n_resamples=n_resamples,
            )  # fmt: skip
        result = metrics.month_bootstrap_contrast(
            treatment=rows[treatment], reference=rows[reference], metric=metric, where=where,
            level_percent=level_percent, n_resamples=n_resamples,
        )  # fmt: skip
    except ValueError:
        return None
    return {**result, "n_events": -1}  # ty: ignore[invalid-return-type]


def show(*, tag: str, result: dict[str, float] | None) -> None:
    """Print one contrast line."""
    if result is None:
        print(f"{tag}: no rows")
        return
    blocks = (
        f"events={result['n_events']}"
        if result["n_events"] >= 0
        else f"months={result['n_months']}"
    )
    print(
        f"{tag}: {result['difference']:+.4f} ({result['lower']:+.4f} to {result['upper']:+.4f}) "
        f"n={result['n_half_hours']:,} {blocks}"
    )


def build_rows(
    *, arms: dict[str, pl.DataFrame], table: pl.DataFrame, flags: pl.DataFrame, ids: pl.DataFrame,
    actuals: pl.DataFrame, series: pl.DataFrame, thresholds: pl.DataFrame,
) -> dict[str, pl.DataFrame]:  # fmt: skip
    """Return every arm's scored rows with the slice columns, `lag_in_event`, and `event`."""
    keys = list(KEY_COLUMNS)
    plain = table.drop("lag_in_event")
    extra = (
        table.select(*keys, "lag_in_event")
        .join(ids, on=keys, how="left")
        .with_columns(pl.col("event").fill_null(-1))
    )
    del flags
    rows: dict[str, pl.DataFrame] = {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=plain)
        scored = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=plain
        )
        scored = add_band_columns(rows=scored, forecast=kept, series=series)
        rows[name] = scored.join(extra, on=keys, validate="1:1")
    return rows


def slice_definitions(*, columns: list[str]) -> dict[str, tuple[pl.Expr, str]]:
    """Return the named slices as `(filter, block kind)`; event slices resample whole events."""
    ev, after = pl.col("event_rows"), pl.col("event_following_7_days")
    band, lag = pl.col(HEADLINE_BAND), pl.col("lag_in_event")
    primary = lag & ~ev & band
    out: dict[str, tuple[pl.Expr, str]] = {
        "PRIMARY: contaminated-lag, outside events, days 1-10": (primary, "event"),
        "all rows, days 1-10": (band, "month"),
        "outside events and 7 days after (harm slice)": (~ev & ~after & band, "month"),
        "clean rows (no lag in event), outside events": (~lag & ~ev & band, "month"),
        "during events": (ev & band, "event"),
        "7 days after events": (after & band, "event"),
    }
    for label, low, high in LEAD_BANDS_HOURS:
        in_band = (pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high)
        out[f"primary, lead {label}"] = (lag & ~ev & in_band, "event")
        out[f"all rows, lead {label}"] = (in_band, "month")
    for group in GROUPS:
        out[f"primary, group {group}"] = (primary & (pl.col("group") == group), "event")
        out[f"all rows, group {group}"] = (band & (pl.col("group") == group), "month")
    out["primary, placebos D2 and D4"] = (primary & pl.col("label").is_in(list(PLACEBOS)), "event")
    out["all rows, placebos D2 and D4"] = (band & pl.col("label").is_in(list(PLACEBOS)), "month")
    out["cold nights, days 1-10"] = (pl.col("cold_nights") & band, "month")
    sunny = "sunny_unit_shape" if "sunny_unit_shape" in columns else "sunny_clear_sky_index"
    out["sunny half-hours, days 1-10"] = (pl.col(sunny) & band, "month")
    return out


def print_absolute_scores(
    *, rows: dict[str, pl.DataFrame], slices: dict[str, tuple[pl.Expr, str]], reference_arm: str
) -> None:
    """Print row and event counts per slice, then the absolute scores per slice, group, and arm."""
    print("\n== Rows and events per slice (first arm) ==")
    first = rows[reference_arm]
    for name, (where, _) in slices.items():
        part = first.filter(where)
        print(
            f"{name}: rows={part.height:,}, events={part.filter(pl.col('event') >= 0)['event'].n_unique()}"
        )
    for name, (where, _) in slices.items():
        print(f"\n== Absolute scores, {name}: by group and arm (points of effective capacity) ==")
        print(summarise(rows=rows, where=where))


def print_site_median_minus_mean(*, rows: dict[str, pl.DataFrame]) -> None:
    """Print the mean of median-forecast minus mean-forecast per site and arm (all rows, days 1-10)."""
    print("\n== Median minus mean forecast per site (points), days 1-10 ==")
    wide = (
        pl.concat(
            [
                f.filter(pl.col(HEADLINE_BAND))
                .group_by("label")
                .agg(pl.col("median_minus_mean").mean())
                .with_columns(arm=pl.lit(n))
                for n, f in rows.items()
            ]
        )
        .pivot(on="arm", index="label", values="median_minus_mean")
        .sort("label")
    )
    print(wide)


def print_contrasts(
    *, rows: dict[str, pl.DataFrame], slices: dict[str, tuple[pl.Expr, str]], resamples: int
) -> dict[str, dict[str, dict[str, float] | None]]:
    """Print C1 to C7 on every slice, raw and widened, and return the primary and harm results."""
    kept: dict[str, dict[str, dict[str, float] | None]] = {}
    primary_name = next(iter(slices))
    harm_name = "outside events and 7 days after (harm slice)"
    print(
        f"\n== Contrasts C1 to C7: treatment minus reference, Bonferroni level {BONFERRONI_LEVEL:.2f}% =="
    )
    for code, treatment, reference in CONTRASTS:
        if treatment not in rows or reference not in rows:
            print(f"{code} {treatment} - {reference}: arm missing, skipped")
            continue
        kept[code] = {}
        for width, suffix in (("raw", ""), ("widened 1.15", "_w")):
            t, r = f"{treatment}{suffix}", f"{reference}{suffix}"
            for slice_name, (where, blocks) in slices.items():
                important = slice_name in (primary_name, harm_name)
                names = FULL_METRICS if important else SHORT_METRICS
                if suffix and not important:
                    continue
                for metric in names:
                    result = contrast(
                        rows=rows, treatment=t, reference=r, metric=metric, where=where,
                        blocks=blocks, level_percent=BONFERRONI_LEVEL, n_resamples=resamples,
                    )  # fmt: skip
                    show(tag=f"{code} {t}-{r} [{width}] {slice_name} {metric}", result=result)
                    if metric == "crps" and slice_name in (primary_name, harm_name):
                        kept[code][f"{slice_name}|{suffix}"] = result
    return kept


def print_verdicts(*, results: dict[str, dict[str, dict[str, float] | None]]) -> None:
    """Print the decision rule's verdict per contrast, raw and widened, and flag widening dependence."""
    primary_name = "PRIMARY: contaminated-lag, outside events, days 1-10"
    harm_name = "outside events and 7 days after (harm slice)"
    print(
        "\n== Decision rule (CRPS): helps = primary interval excludes zero below and harm <= +0.02 =="
    )
    for code, entries in results.items():
        verdicts = {}
        for suffix in ("", "_w"):
            p, h = entries.get(f"{primary_name}|{suffix}"), entries.get(f"{harm_name}|{suffix}")
            if p is None or h is None:
                verdicts[suffix] = "no rows"
                continue
            helps = p["upper"] < 0 and h["difference"] <= HARM_TOLERANCE
            equivalent = p["lower"] > -EQUIVALENCE_MARGIN and p["upper"] < EQUIVALENCE_MARGIN
            verdicts[suffix] = (
                f"helps={helps}, primary {p['difference']:+.4f}, harm {h['difference']:+.4f}, "
                f"within +-{EQUIVALENCE_MARGIN} (C3 equivalence test)={equivalent}"
            )
        flag = (
            "widening-dependent"
            if verdicts.get("") != verdicts.get("_w")
            and all(v != "no rows" for v in verdicts.values())
            and verdicts[""].split(",")[0] != verdicts["_w"].split(",")[0]
            else "same helps-verdict"
        )
        print(
            f"{code} raw: {verdicts.get('')}\n{code} widened: {verdicts.get('_w')}\n{code}: {flag}"
        )


def print_prediction_checks(
    *, rows: dict[str, pl.DataFrame], labelled: list[str], slices: dict[str, tuple[pl.Expr, str]]
) -> None:
    """Print the O-minus-b0 gap and P1 to P4 as absolute changes in points of CRPS."""
    primary = slices["PRIMARY: contaminated-lag, outside events, days 1-10"][0]
    harm = slices["outside events and 7 days after (harm slice)"][0]

    def mean_crps(arm: str, where: pl.Expr) -> float | None:
        if arm not in rows:
            return None
        value = rows[arm].filter(where)["crps"].mean()
        return None if value is None else float(value)  # ty: ignore[invalid-argument-type]

    def diff(a: str, b: str, where: pl.Expr) -> float | None:
        x, y = mean_crps(a, where), mean_crps(b, where)
        return None if x is None or y is None else x - y

    print("\n== O-minus-b0 gap and P1 to P4 (points of CRPS; primary slice unless stated) ==")
    for name, where in (
        ("all labels", primary),
        ("signed-MW", primary & (pl.col("group") == "signed-MW")),
        ("MVA-only", primary & (pl.col("group") == "MVA-only")),
    ):
        gap = diff("o", "b0", where)
        print(f"[{name}] O - b0 gap: {gap}")
        for tag, a, b in (("P1 n0-b0", "n0", "b0"), ("P3 n2-n1", "n2", "n1"), ("n3-b0", "n3", "b0"),
                          ("n1-b0", "n1", "b0"), ("n2-b0", "n2", "b0")):  # fmt: skip
            change = diff(a, b, where)
            share = None if change is None or not gap else 100.0 * change / abs(gap)
            print(f"[{name}] {tag}: {change} ({share}% of |gap|)")
    transfer = primary & pl.col("label").is_in(labelled)
    change = diff("n1", "n0", transfer)
    gap = diff("o", "b0", transfer)
    print(f"[Primaries with a labelled transfer {labelled}] P2 n1-n0: {change}; O - b0 gap: {gap}")
    print("P3 also asks that n3 recovers less than n1: compare n3-b0 with n1-b0 above.")
    for arm in ("n0", "n1", "n2", "n3"):
        print(f"P4 {arm}-b0 on the harm slice: {diff(arm, 'b0', harm)} (limit +{HARM_TOLERANCE})")


def print_per_primary(
    *, rows: dict[str, pl.DataFrame], slices: dict[str, tuple[pl.Expr, str]]
) -> None:
    """Print each contrast per Primary (point estimates), the worst Primary, and leave-one-out."""
    primary = slices["PRIMARY: contaminated-lag, outside events, days 1-10"][0]
    labels = sorted(rows[next(iter(rows))]["label"].unique().to_list())
    for code, treatment, reference in CONTRASTS:
        if treatment not in rows or reference not in rows:
            continue
        joined = (
            rows[treatment].filter(primary).select(*KEY_COLUMNS, "label", "crps")
            .join(rows[reference].select(*KEY_COLUMNS, ref=pl.col("crps")), on=list(KEY_COLUMNS))
            .with_columns(delta=pl.col("crps") - pl.col("ref"))
        )  # fmt: skip
        per = joined.group_by("label").agg(pl.col("delta").mean(), n=pl.len()).sort("label")
        print(f"\n== {code} {treatment}-{reference}: per Primary, primary slice (CRPS points) ==")
        print(per)
        if per.height:
            worst = per.sort("delta", descending=True).row(0, named=True)
            print(f"worst Primary: {worst['label']} ({worst['delta']:+.4f}, n={worst['n']:,})")
        for label in labels:
            rest = joined.filter(pl.col("label") != label)
            if rest.height:
                print(f"leave out {label}: {rest['delta'].mean():+.4f} (n={rest.height:,})")


def cost_report(
    *,
    arms: dict[str, pl.DataFrame],
    series: pl.DataFrame,
    actuals: pl.DataFrame,
    table: pl.DataFrame,
) -> None:
    """Print the avoidable curtailment cost per Primary and group in three row sets (signed-MW six)."""
    flows = actuals.rename({"actual": "flow_mw"})
    signed = series.filter(pl.col("label").is_in(list(SIGNED_MW_LABELS)))
    sets = {
        "during events": pl.col("event_rows"),
        "7 days after events": pl.col("event_following_7_days"),
        "all rows": pl.lit(value=True),
    }
    for name, cond in sets.items():
        keys = table.filter(cond).select(*KEY_COLUMNS)
        forecasts = {
            n: f.join(keys, on=list(KEY_COLUMNS), how="semi").join(
                actuals, on=["time_series_id", "valid_time"]
            )
            for n, f in arms.items()
        }
        parts = []
        for label in SIGNED_MW_LABELS:
            one = signed.filter(pl.col("label") == label)
            try:
                parts.append(
                    metrics.avoidable_curtailment_cost(
                        forecasts=forecasts,
                        series=one,
                        flows=flows,
                        scored_start=SCORED_FIRST,
                        fraction=CURTAILMENT_FRACTION,
                    )
                )
            except ValueError as error:
                print(f"{label}: {name}: no cost ({error})")
        if not parts:
            continue
        cost = pl.concat(parts)
        print(
            f"\n== Avoidable curtailment cost (GBP), limit {CURTAILMENT_FRACTION} of earlier max, 12-36 h: {name}, per Primary =="
        )
        print(cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label"))
        print(f"-- {name}, signed-MW group total --")
        print(cost.group_by("arm").agg(total_gbp=pl.col("avoidable_cost_gbp").sum()).sort("arm"))


def main() -> None:
    """Print the scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="read arms_smoke, write scores_smoke")
    parser.add_argument("--resamples", type=int, default=5000, help="bootstrap resamples")
    args = parser.parse_args()
    arms_dir = assemble.SMOKE_OUT_DIR if args.smoke else assemble.OUT_DIR
    out = WORK / ("scores_smoke" if args.smoke else "scores")
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(260)
    out.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=ref379.CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache, series=series, first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )  # fmt: skip
    all_events = adapter.events_from_masks(cache=cache, series=series)
    events = (
        all_events.join(series.select("time_series_id", "label"), on="time_series_id")
        .filter(pl.col("end") >= SCORED_FIRST)
        .sort("label", "start")
        .with_row_index("event")
    )
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in series.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}), series=series, before=SCORED_FIRST
    )
    arms, references = load_arms(series=series, arms_dir=arms_dir)
    print(f"arms: {list(arms)}\nstored references (bg2 carries 1.15 already): {references}")
    table = metrics.shared_slice_table(
        forecasts=arms, actuals=actuals, context=context, events=all_events,
        first_readings=first_readings,
    )  # fmt: skip
    flags = pl.read_parquet(arms_dir / "contaminated.parquet").with_columns(
        pl.col("time_series_id").cast(series.schema["time_series_id"])
    )
    table = table.join(flags, on=list(KEY_COLUMNS), how="left", validate="1:1").with_columns(
        pl.col("lag_in_event").fill_null(value=False)
    )
    print(f"shared half-hours: {table.height:,}; events scored: {events.height}")
    ids = event_ids(table=table, events=events)
    rows = build_rows(
        arms=arms,
        table=table,
        flags=flags,
        ids=ids,
        actuals=actuals,
        series=series,
        thresholds=thresholds,
    )
    pl.concat([r.with_columns(arm=pl.lit(n)) for n, r in rows.items()]).write_parquet(
        out / "rows.parquet"
    )
    slices = slice_definitions(columns=table.columns)
    reference_arm = next(iter(rows))
    labelled = sorted(events["label"].unique().to_list())
    print(f"Primaries with a labelled event: {labelled}")
    print_absolute_scores(rows=rows, slices=slices, reference_arm=reference_arm)
    print_site_median_minus_mean(rows=rows)
    results = print_contrasts(rows=rows, slices=slices, resamples=args.resamples)
    print_verdicts(results=results)
    print_prediction_checks(rows=rows, labelled=labelled, slices=slices)
    print_per_primary(rows=rows, slices=slices)
    cost_report(arms=arms, series=series, actuals=actuals, table=table)


if __name__ == "__main__":
    main()

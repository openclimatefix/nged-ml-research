r"""Experiment 697's tables (T1 to T6) and questions (Q1 to Q8), from the stored frames.

    uv run --no-sync python $E/00697-*/e00697_stats.py

Reads the frames that `e00697_frame.py` wrote. Writes `printed/tables.txt` and `stats.json` (in the
data folder). **The printed tables hold no Gas A date and no Gas A series: Gas A appears only as
aggregates over the whole record.** D8's tracked state appears only as shares over many half-hours.

Every interval resamples whole ISO weeks of local dates within an era, 2,000 draws, percentile
method, and a spell belongs to the week in which it starts. A difference of two estimates (D8
minus a control) resamples the two terms independently. Gas A's intervals are rough and decide
nothing. A bin of fewer than 20 distinct local days is left out of every test.
"""

import json
import pickle
from typing import Final

import numpy as np
import polars as pl
from e00694_common import f1_counts, f1_of, load_panel  # ty: ignore[unresolved-import]
from e00697_analysis import (  # ty: ignore[unresolved-import]
    BinnedShare,
    add_price_thirds,
    bin_shares,
    block_shares,
    cell_bins,
    day_table,
    overall_bins,
    q1,
    q2,
    q3,
    q5,
    q6,
    q7,
    q8,
    quantile_row,
    record_edges,
    record_levels,
    spell_table,
    stable_seed,
    usable,
    with_state,
)
from e00697_common import (  # ty: ignore[unresolved-import]
    BLOCK_NAMES,
    DATA_DIR,
    E695_DIR,
    ERA_NAMES,
    EVENING_FIRST,
    EVENING_LAST,
    FALSE_ALARM_PATH,
    MIN_DAYS_PER_BIN,
    PIPELINE_OFFSET,
    PRINTED_DIR,
    PROBABILITY_PATH,
    STATS_PATH,
    VERDICT_ERAS,
    Estimate,
    day_block,
    difference_estimate,
    panel_array,
    spell_quantiles,
)
from e00697_frame import load_frame  # ty: ignore[unresolved-import]

NULL_FRAMES: Final[tuple[str, ...]] = (
    "c1_null",
    "c1s_net_null",
    "c1s_solar_null",
    "c2_d2_null",
    "c2_d3_null",
    "c2_d4_null",
)
"""Controls whose truth has no weekday pattern: the null controls that the verdict compares with."""
WEEKDAY_FRAMES: Final[tuple[str, ...]] = (
    "c1_weekday",
    "c2_d2_weekday",
    "c2_d3_weekday",
    "c2_d4_weekday",
)
"""Controls whose truth is drawn within the weekday class: they give the attenuation."""
CONTROL_LABELS: Final[tuple[str, ...]] = ("ts", "tf", "truth1", "truth2")
D8_LABELS: Final[tuple[str, ...]] = ("ts", "tf", "p_any_a")
RECORD_LABELS: Final[tuple[str, ...]] = ("r1", "r2")
QUESTION_NAMES: Final[tuple[str, ...]] = ("q1", "q2", "q2q", "q3", "q3a", "q6", "q7", "q8")
C4_EXPECTED_F1: Final[float] = 0.785
C4_EXPECTED_PRECISION: Final[float] = 0.998
C4_TOLERANCE: Final[float] = 0.01
NIGHT_SHORTFALL_MW: Final[float] = -2.5
LINES: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for `printed/tables.txt`."""
    print(text, flush=True)
    LINES.append(text)


def table(*, title: str, frame: pl.DataFrame) -> None:
    """Print a titled polars table in full."""
    say(f"\n{title}")
    with pl.Config(
        tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, float_precision=3, fmt_str_lengths=40
    ):
        say(str(frame))


def control_view(*, frame: pl.DataFrame, era: str) -> pl.DataFrame:
    """Return a control's rows for a D8 era: the pre-engine span, or the span of that era."""
    return frame.filter(pl.col("era").is_in(["pre", era]))


def questions(
    *, view: pl.DataFrame, subject: str, label: str, era: str, min_days: int, min_spells: int
) -> dict[str, object]:
    """Return every question for one view, with seeds that depend on the subject, label, and era."""

    def seed(name: str) -> int:
        return stable_seed(subject, label, era, name)

    out: dict[str, object] = {
        "q1": q1(view=view, seed=seed("q1"), min_days=min_days),
        "q2": q2(view=view, seed=seed("q2"), min_days=min_days),
        "q2q": q2(view=view, seed=seed("q2q"), min_days=min_days, by_quarter=True),
    }
    if label != "p_any_a":
        out["q3"], out["q3a"] = q3(view=view, seed=seed("q3"), min_spells=min_spells)
        out["q7"] = q7(view=view, seed=seed("q7"), min_days=min_days)
        out["q8"] = q8(view=view, seed=seed("q8"))
    if era == "net":
        out["q6"] = q6(view=view, seed=seed("q6"), min_days=min_days)
    return out


def jsonable(obj: object) -> object:
    """Convert estimates, numpy values, and nested containers to plain JSON values."""
    if isinstance(obj, Estimate):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def show(*, value: object) -> str:
    """Format an estimate, a Q8 dictionary, or a missing value on one line."""
    if value is None:
        return "not estimable (a bin holds fewer than the minimum days or spells)"
    if isinstance(value, Estimate):
        return f"{value.point:+.3f} [{value.lo:+.3f}, {value.hi:+.3f}] over {value.n_weeks} weeks"
    return str(value)


# --------------------------------------------------------------------------- the questions


def run_questions(*, frames: dict[str, pl.DataFrame]) -> dict[str, dict]:
    """Compute every question for D8, Gas A, and every control, and the differences D."""
    results: dict[str, dict] = {"d8": {}, "record": {}}
    d8 = frames["d8"]
    for era in ERA_NAMES:
        results["d8"][era] = {}
        rows = d8.filter(pl.col("era") == era)
        for label in D8_LABELS:
            view = with_state(frame=rows, column=label)
            results["d8"][era][label] = questions(
                view=view,
                subject="d8",
                label=label,
                era=era,
                min_days=MIN_DAYS_PER_BIN,
                min_spells=20,
            )
    for label in RECORD_LABELS:
        view = with_state(frame=frames["record"], column=label)
        results["record"][label] = questions(
            view=view, subject="record", label=label, era="record", min_days=0, min_spells=1
        )
    for name in (*NULL_FRAMES, *WEEKDAY_FRAMES):
        results[name] = {}
        for era in VERDICT_ERAS:
            rows = control_view(frame=frames[name], era=era)
            if rows.height == 0:
                continue
            results[name][era] = {}
            for label in CONTROL_LABELS:
                view = with_state(frame=rows, column=label)
                results[name][era][label] = questions(
                    view=view,
                    subject=name,
                    label=label,
                    era=era,
                    min_days=MIN_DAYS_PER_BIN,
                    min_spells=20,
                )
    results["cross_era"] = cross_era_questions(frames=frames)
    results["difference"] = differences(results=results)
    return results


def cross_era_questions(*, frames: dict[str, pl.DataFrame]) -> dict[str, object]:
    """Q5 (night share after the solar start minus a year earlier) for D8 and the host controls."""
    out: dict[str, object] = {}
    d8 = frames["d8"].filter(pl.col("era").is_in(list(VERDICT_ERAS)))
    for label in ("ts", "tf", "p_any_a"):
        out[f"d8_{label}"] = q5(
            view=with_state(frame=d8, column=label), seed=stable_seed("d8", label, "q5")
        )
    for name in NULL_FRAMES:
        if not name.startswith("c2"):
            continue
        for label in ("ts", "truth1"):
            view = with_state(frame=frames[name], column=label)
            out[f"{name}_{label}"] = q5(view=view, seed=stable_seed(name, label, "q5"))
    for label in RECORD_LABELS:
        out[f"record_{label}"] = None
    d8_est = out["d8_ts"]
    for name in NULL_FRAMES:
        control = out.get(f"{name}_ts")
        if isinstance(d8_est, Estimate) and isinstance(control, Estimate):
            out[f"difference_{name}"] = difference_estimate(first=d8_est, second=control)
    return out


def differences(*, results: dict[str, dict]) -> dict[str, dict]:
    """Return D, D8's estimate minus each null control's, for each question and era (Ts labels)."""
    out: dict[str, dict] = {}
    for question in ("q1", "q2", "q2q", "q3", "q3a", "q6", "q7"):
        out[question] = {}
        for era in VERDICT_ERAS:
            if question == "q6" and era != "net":
                continue
            first = results["d8"][era]["ts"].get(question)
            row = {}
            for name in NULL_FRAMES:
                second = results[name].get(era, {}).get("ts", {}).get(question)
                if isinstance(first, Estimate) and isinstance(second, Estimate):
                    row[name] = difference_estimate(first=first, second=second)
            out[question][era] = row
    return out


# --------------------------------------------------------------------------- tables


def t1_coverage(*, frames: dict[str, pl.DataFrame]) -> None:
    """T1: half-hours, days, and the share of each input missing, by series, era, and half."""
    inputs = ("system_price", "gas_price_gbp_mwh", "day_ahead_price", "national_wind_mw")
    for name, frame in frames.items():
        keys = ["era", "summer_half"]
        agg = (
            frame.group_by(keys)
            .agg(
                half_hours=pl.len(),
                days=pl.col("local_date").n_unique(),
                reading_missing=1.0 - pl.col("observed").mean(),
                **{f"{c}_missing": pl.col(c).is_null().mean() for c in inputs},
            )
            .sort(keys)
        )
        table(title=f"T1 coverage: {name} (share of half-hours missing per input)", frame=agg)


def t2_labels_and_controls(
    *, frames: dict[str, pl.DataFrame], results: dict[str, dict]
) -> dict[str, object]:
    """T2: Ts against Tf, Ta, and Tc; scores of each control; the attenuation; C3 and C4."""
    d8 = frames["d8"]
    agree = (
        d8.group_by("era")
        .agg(
            flagged_ts=pl.col("ts").mean(),
            flagged_tf=pl.col("tf").mean(),
            flagged_ta=(pl.col("p_any_a") > 0.5).mean(),
            mean_probability_ta=pl.col("p_any_a").mean(),
            agree_ts_tf=(pl.col("ts") == pl.col("tf")).mean(),
            agree_ts_ta=(pl.col("ts") == (pl.col("p_any_a") > 0.5)).mean(),
            agree_ts_tc=(pl.col("ts") == (pl.col("p_any_c") > 0.5))
            .filter(pl.col("p_any_c").is_not_null())
            .mean(),
            n_tc=pl.col("p_any_c").is_not_null().sum(),
        )
        .sort("era")
    )
    table(title="T2a D8: Ts against Tf, Ta (stored probabilities), and Tc (fixed lag)", frame=agree)
    scores = []
    for name in (*NULL_FRAMES, *WEEKDAY_FRAMES):
        for era, rows in (
            frames[name].filter(pl.col("observed")).group_by("era", maintain_order=True)
        ):
            for truth in ("truth1", "truth2"):
                tp, fp, fn, _ = f1_counts(truth=rows[truth].to_numpy(), flag=rows["ts"].to_numpy())
                precision = tp / max(tp + fp, 1)
                scores.append(
                    {
                        "control": name,
                        "span": era[0],
                        "truth_at": truth,
                        "f1": f1_of(tp=tp, fp=fp, fn=fn),
                        "precision": precision,
                        "recall": tp / max(tp + fn, 1),
                        "flagged": (tp + fp) / rows.height,
                        "truth_share": (tp + fn) / rows.height,
                    }
                )
    table(
        title="T2b controls: Ts against the injected truth, by control and span",
        frame=pl.DataFrame(scores),
    )
    ratios = []
    for name in WEEKDAY_FRAMES:
        for era in VERDICT_ERAS:
            labels = results[name].get(era, {}).get("ts", {}).get("q1")
            truth = results[name].get(era, {}).get("truth1", {}).get("q1")
            if isinstance(labels, Estimate) and isinstance(truth, Estimate):
                ratios.append(
                    {
                        "control": name,
                        "era": era,
                        "q1_labels": labels.point,
                        "q1_truth": truth.point,
                        "attenuation": labels.point / truth.point if truth.point else float("nan"),
                    }
                )
    table(
        title="T2c attenuation: Q1 of the labels over Q1 of the truth, weekday-class controls",
        frame=pl.DataFrame(ratios),
    )
    c3 = c3_false_alarms()
    say(f"\nT2d C3 (experiment 694's stored false-alarm pass, read and not rerun): {c3}")
    c4 = c4_reproduction()
    say(f"T2e C4 (experiment 695's stored host states against its day draw): {c4}")
    return {"c3": c3, "c4": c4}


def c3_false_alarms() -> dict[str, int]:
    """C3: segments, half-hours, and half-hours flagged by M2 in the stored false-alarm pass."""
    segments = pickle.loads(FALSE_ALARM_PATH.read_bytes())
    return {
        "segments": len(segments),
        "half_hours": sum(s["arms"]["M2"]["n"] for s in segments),
        "flagged": sum(s["arms"]["M2"]["flagged"] for s in segments),
    }


def c4_reproduction() -> dict[str, float | bool]:
    """C4: the host's stored causal states against experiment 695's day draw, F1 and precision.

    This repeats `run_g1` of `e00695_checks.py` without its imports of the network code. The
    observed half-hours come from the panel's reading of the host, which is the same cleaned series.
    """
    from e00695_common import FOLD_MONTHS, HOLDOUT, fold_refit, grid_index
    from e00695_engine import drive_series

    host = json.loads((E695_DIR / "host.json").read_text())["host"]
    reading = panel_array(panel=load_panel(), label=host, key="power")
    n_pipeline = len(np.load(PROBABILITY_PATH)["lmean"])
    g, _ = drive_series(n_grid=n_pipeline)
    stop = grid_index(when=HOLDOUT)
    flags, truths = [], []
    for month in FOLD_MONTHS:
        first = grid_index(when=fold_refit(month=month))
        last = min(first + 31 * 48, stop)
        pk = np.load(E695_DIR / "states" / "W1" / f"{month}.npz")["pk"][first:last].astype(
            np.float64
        )
        observed = np.isfinite(reading[first - PIPELINE_OFFSET : last - PIPELINE_OFFSET])
        flags.append((np.argmax(pk, axis=1) >= 1)[observed])
        truths.append((g[first:last] > 0.0)[observed])
    tp, fp, fn, _ = f1_counts(truth=np.concatenate(truths), flag=np.concatenate(flags))
    f1, precision = f1_of(tp=tp, fp=fp, fn=fn), tp / max(tp + fp, 1)
    return {
        "host": host,
        "f1": round(f1, 3),
        "precision": round(precision, 3),
        "reproduces_0.785_and_0.998": bool(
            abs(f1 - C4_EXPECTED_F1) <= C4_TOLERANCE
            and abs(precision - C4_EXPECTED_PRECISION) <= C4_TOLERANCE
        ),
    }


def t3_blocks(*, frames: dict[str, pl.DataFrame]) -> None:
    """T3: running share by weekday class and block of the day, and the share of running days."""
    sets: list[tuple[str, pl.DataFrame, str, str]] = []
    for era in ERA_NAMES:
        rows = frames["d8"].filter(pl.col("era") == era)
        sets += [
            (f"d8 {era} ts", rows, "ts", "d8"),
            (f"d8 {era} Ta mean probability", rows, "p_any_a", "d8"),
        ]
    sets += [
        ("gas A at 1 or more units", frames["record"], "r1", "record"),
        ("gas A at 2 or more units", frames["record"], "r2", "record"),
    ]
    sets += [
        ("control c1 null: labels", frames["c1_null"], "ts", "c1"),
        ("control c1 null: truth", frames["c1_null"], "truth1", "c1"),
    ]
    sets += [
        ("control c1 weekday: labels", frames["c1_weekday"], "ts", "c1"),
        ("control c1 weekday: truth", frames["c1_weekday"], "truth1", "c1"),
    ]
    for title, rows, label, _ in sets:
        view = with_state(frame=rows, column=label)
        by_half = block_shares(view=view)
        table(title=f"T3 running share by block of the day: {title}", frame=by_half)
        days = (
            day_table(view=view)
            .group_by("weekday")
            .agg(running_day_share=pl.col("running_day").mean(), days=pl.len())
            .sort("weekday")
        )
        table(
            title=f"T3 share of running days (4 or more running half-hours) by weekday: {title}",
            frame=days,
        )


def _block_name(half_hour: int) -> str:
    """Return the block name of a local half-hour."""
    return BLOCK_NAMES[int(day_block(half_hour=np.array([half_hour]))[0])]


def t4_spells(*, frames: dict[str, pl.DataFrame]) -> None:
    """T4: spell counts, quantiles, shares of length 1 and 1 to 2, and hazards, by label set."""
    sets: list[tuple[str, pl.DataFrame, str]] = []
    for era in ERA_NAMES:
        rows = frames["d8"].filter(pl.col("era") == era)
        sets += [(f"d8 {era} ts", rows, "ts"), (f"d8 {era} tf", rows, "tf")]
    sets += [("gas A 1+ units", frames["record"], "r1"), ("gas A 2+ units", frames["record"], "r2")]
    for name in ("c1_null", "c1_weekday"):
        sets += [
            (f"{name} labels (ts)", frames[name], "ts"),
            (f"{name} labels (tf)", frames[name], "tf"),
            (f"{name} truth", frames[name], "truth1"),
        ]
    rows_out = []
    for title, rows, label in sets:
        view = with_state(frame=rows, column=label)
        for kind in (True, False):
            spells = spell_table(view=view, kind=kind)
            lengths = spells["length"].to_numpy()
            rows_out.append(
                {
                    "set": title,
                    "state": "on" if kind else "off",
                    **spell_quantiles(lengths=lengths),
                    **quantile_row(lengths=lengths),
                }
            )
    table(
        title="T4 complete spells in half-hours (hazard: ends at n over reaches n)",
        frame=pl.DataFrame(rows_out, infer_schema_length=None),
    )
    by_block = []
    for title, rows, label in sets:
        view = with_state(frame=rows, column=label)
        spells = spell_table(view=view, kind=True)
        if spells.height == 0:
            continue
        grouped = (
            spells.with_columns(
                block=pl.Series([_block_name(h) for h in spells["start_half_hour"].to_list()])
            )
            .group_by("block")
            .agg(
                n=pl.len(),
                share_len_le2=(pl.col("length") <= 2).mean(),
                median=pl.col("length").median(),
            )
            .with_columns(set=pl.lit(title))
            .sort("block")
        )
        by_block.append(grouped)
    table(
        title="T4 on-spells by the block of the day in which the spell starts",
        frame=pl.concat(by_block),
    )


def t5_levels(*, frames: dict[str, pl.DataFrame]) -> dict[str, object]:
    """T5: Gas A's levels and edges from `power`, and the tracker's fitted block sizes by era."""
    levels = record_levels(record=frames["record"])
    edges = record_edges(record=frames["record"])
    say(f"\nT5 gas A levels (from power, not from the `units` column): {levels}")
    say(
        f"T5 gas A edges: {edges['n_spells']} complete on-spells with 3 half-hours either side; "
        f"first full level within 1 half-hour {edges['share_full_level_within_1']:.3f}, "
        f"within 2 {edges['share_full_level_within_2']:.3f}"
    )
    blocks = (
        frames["d8"]
        .filter(pl.col("k_path") > 0)
        .group_by("era", "k_path", "unit_mw")
        .agg(n=pl.len())
        .with_columns(
            block_mw=pl.col("k_path") * pl.col("unit_mw"),
            share=pl.col("n") / pl.col("n").sum().over("era"),
        )
        .sort("era", "k_path")
    )
    table(
        title="T5 the tracker's fitted block size in MW among flagged half-hours, by era",
        frame=blocks,
    )
    say("\nD8's reading does not give a unit count: the tracker's count is a block size.")
    return {
        "levels": levels,
        "edges": {k: v for k, v in edges.items() if not k.endswith("profile")},
    }


def driver_rows(*, view: pl.DataFrame, subject: str, min_days: int) -> list[BinnedShare]:
    """Return the running share by bin of each driver, for one view."""
    out: list[BinnedShare] = []
    priced = add_price_thirds(view=view)
    thirds = priced["third"].to_numpy()
    out += bin_shares(
        view=priced,
        bins=thirds,
        label=f"{subject} system price third within cell",
        min_days=min_days,
    )
    for column, n_bins, name in (
        ("system_price", 5, "system price fifth overall"),
        ("spark_spread", 5, "spark spread fifth"),
        ("gas_price_gbp_mwh", 5, "gas price fifth"),
        ("day_ahead_price", 3, "day-ahead price third"),
        ("national_wind_mw", 3, "national wind third"),
        ("price_rank_in_day", 4, "price rank in day quarter"),
    ):
        usable_view = usable(frame=view).filter(pl.col(column).is_not_null())
        if usable_view.height == 0:
            continue
        out += bin_shares(
            view=usable_view,
            bins=overall_bins(view=usable_view, column=column, n_bins=n_bins),
            label=f"{subject} {name}",
            min_days=min_days,
        )
    for column, name in (("temp", "temperature"), ("wind100", "100 m wind"), ("ghi", "irradiance")):
        if column in view.columns and view[column].null_count() < view.height:
            usable_view = usable(frame=view).filter(pl.col(column).is_not_null())
            out += bin_shares(
                view=usable_view,
                bins=cell_bins(view=usable_view, column=column, n_bins=3),
                label=f"{subject} {name} third within cell",
                min_days=min_days,
            )
    return out


def t6_drivers(*, frames: dict[str, pl.DataFrame]) -> None:
    """T6: running share by driver bin, holidays, clock-change days, drops, and night share."""
    rows: list[dict] = []
    for era in ERA_NAMES:
        for label in ("ts", "p_any_a"):
            view = with_state(frame=frames["d8"].filter(pl.col("era") == era), column=label)
            rows += [
                {
                    "set": f"d8 {era} {label}",
                    "driver": b.label,
                    "bin": b.bin,
                    "half_hours": b.n,
                    "days": b.days,
                    "share": b.share,
                }
                for b in driver_rows(view=view, subject="d8", min_days=MIN_DAYS_PER_BIN)
            ]
    for label in RECORD_LABELS:
        view = with_state(frame=frames["record"], column=label)
        rows += [
            {
                "set": f"gas A {label}",
                "driver": b.label,
                "bin": b.bin,
                "half_hours": b.n,
                "days": b.days,
                "share": b.share,
            }
            for b in driver_rows(view=view, subject="record", min_days=0)
        ]
    for name in ("c2_d3_null", "c2_d4_null"):
        for era in VERDICT_ERAS:
            view = with_state(frame=control_view(frame=frames[name], era=era), column="ts")
            rows += [
                {
                    "set": f"{name} {era} labels",
                    "driver": b.label,
                    "bin": b.bin,
                    "half_hours": b.n,
                    "days": b.days,
                    "share": b.share,
                }
                for b in driver_rows(view=view, subject=name, min_days=MIN_DAYS_PER_BIN)
            ]
    table(
        title="T6 running share by driver bin (bins under 20 days are left out)",
        frame=pl.DataFrame(rows, infer_schema_length=None),
    )
    holiday_rows = []
    for era in ERA_NAMES:
        view = with_state(frame=frames["d8"].filter(pl.col("era") == era), column="ts")
        evening = view.filter(
            pl.col("local_half_hour").is_between(EVENING_FIRST, EVENING_LAST), pl.col("observed")
        )
        weekday = evening.filter(~pl.col("weekend"))
        groups = {
            "ordinary weekday evenings": weekday.filter(
                ~pl.col("bank_holiday"), ~pl.col("christmas_week"), ~pl.col("clock_change_day")
            ),
            "bank-holiday weekday evenings (pooled)": weekday.filter(
                pl.col("bank_holiday"), ~pl.col("christmas_week")
            ),
            "christmas week evenings (24 December to 1 January)": evening.filter(
                pl.col("christmas_week")
            ),
            "six clock-change days, evenings (pooled)": evening.filter(pl.col("clock_change_day")),
            "inside experiment 159's multi-day drops, evenings": evening.filter(pl.col("in_drop")),
            "outside the drops, evenings": evening.filter(~pl.col("in_drop")),
        }
        for name, group in groups.items():
            days = group["local_date"].n_unique()
            holiday_rows.append(
                {
                    "era": era,
                    "group": name,
                    "days": days,
                    "share": group["state"].mean() if group.height else None,
                    "shown": days >= MIN_DAYS_PER_BIN or "pooled" in name,
                }
            )
    table(
        title="T6 holidays, clock changes, and drops: evening share (no drop date shown)",
        frame=pl.DataFrame(holiday_rows, infer_schema_length=None),
    )
    night = with_state(frame=frames["d8"], column="ts")
    night = usable(frame=night).filter(
        pl.col("local_half_hour") <= 9, pl.col("expected").is_not_null()
    )
    monthly = (
        night.with_columns(
            residual=pl.col("reading") - pl.col("expected"),
            year_month=pl.col("local_date").dt.strftime("%Y-%m"),
        )
        .group_by("year_month")
        .agg(
            night_flagged=pl.col("state").mean(),
            night_residual_below_minus_2_5_mw=(pl.col("residual") < NIGHT_SHORTFALL_MW).mean(),
            half_hours=pl.len(),
        )
        .sort("year_month")
    )
    with pl.Config(tbl_rows=-1, float_precision=3):
        (DATA_DIR / "night_monthly.txt").write_text(str(monthly) + "\n")
    say(
    say(f"\nT6 night share by month: written to {DATA_DIR.name}/night_monthly.txt, not printed")
    )


def main() -> None:
    """Load the frames, run every question and table, and write the report and `stats.json`."""
    frames = {
        name: load_frame(name=name) for name in ("d8", "record", *NULL_FRAMES, *WEEKDAY_FRAMES)
    }
    say("Experiment 697: tables and questions")
    results = run_questions(frames=frames)
    for question in QUESTION_NAMES:
        say(f"\n{question.upper()}")
        for era in ERA_NAMES:
            for label, value in results["d8"][era].items():
                if question in value:
                    say(f"  d8 {era} {label}: {show(value=value[question])}")
        for label, value in results["record"].items():
            if question in value:
                say(f"  gas A {label} (rough interval, descriptive): {show(value=value[question])}")
    cross = {k: show(value=v) for k, v in results["cross_era"].items()}
    say(f"\nQ5 (night share after the solar start minus a year earlier): {cross}")
    for question, by_era in results["difference"].items():
        for era, row in by_era.items():
            for name, est in row.items():
                say(f"D {question} {era} against {name}: {show(value=est)}")
    t1_coverage(frames=frames)
    checks = t2_labels_and_controls(frames=frames, results=results)
    t3_blocks(frames=frames)
    t4_spells(frames=frames)
    levels = t5_levels(frames=frames)
    t6_drivers(frames=frames)
    say("\nAbout 30 exploratory intervals are shown, at 95% and without a correction.")
    PRINTED_DIR.mkdir(parents=True, exist_ok=True)
    (PRINTED_DIR / "tables.txt").write_text("\n".join(LINES) + "\n")
    STATS_PATH.write_text(
        json.dumps(jsonable({"questions": results, "checks": checks, "t5": levels}), indent=1)
    )


if __name__ == "__main__":
    main()

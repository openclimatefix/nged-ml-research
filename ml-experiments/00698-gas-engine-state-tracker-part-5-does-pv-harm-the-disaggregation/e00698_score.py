r"""Score experiment 698's stored runs: paired contrasts, descriptive tables, and step 4's recall.

    E=ml-experiments
    export PYTHONPATH=$(ls -d $E/00698-* $E/00697-* $E/00694-* $E/00695-* | paste -sd:)
    uv run --no-sync python $E/00698-*/e00698_score.py

The script reads `gate.json` and `jobs/*.npz`, writes `scores.json` (private) and
`printed/score_report.txt` (committed, with no date and no Gas A or farm series, and with scores
only as sums over whole runs, hours of the day, halves of the year, and thirds), and prints the
paired half-widths **before** any score.

**Rows.** A half-hour is scored when D8's or D3's reading is present, the day is not a clock-change
day, and the farm series is finite (`farm_ok`). The same rows are scored in every run of a host, so
the three shapes and the solar-free twin are compared on identical rows. The criteria read the
half-hours from 10:00 to 15:00 local time. The thirds are thirds of the host's CAMS shape within a
cell of local half-hour, weekday class, and month, above 50 W/m2 of CAMS irradiance.

**Contrasts.** For each host and capacity, with 3 seeds pooled (sums of the numerators and the
denominators over the seeds), the harm is stated so that a positive number is worse:

- `level`: the CAMS smooth twin against the solar-free twin (harm from the lower level and the
  fold);
- `weather_farm`: the farm run against its own twin, the smooth twin built from the farm's cell
  means (same level and same daily and seasonal shape, no weather);
- `weather_cams`: the CAMS run against the CAMS smooth twin;
- `total_farm` and `total_cams`: the farm run, or the CAMS run, against the solar-free twin.

The three criteria are F1 (falls), the false-flag share on half-hours whose injected output is zero
(rises), and the bright-minus-dull flagged share (rises). One draw of whole ISO weeks serves every
run of a host. Every contrast is computed on the smoothed labels, with the filtered labels beside
them.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from e00694_common import load_panel  # ty: ignore[unresolved-import]
from e00697_common import (  # ty: ignore[unresolved-import]
    Estimate,
    difference_estimate,
    era_bounds,
    share_columns,
    share_difference_statistic,
    week_estimate,
)
from e00698_common import (  # ty: ignore[unresolved-import]
    BOOTSTRAP_SEED,
    CAPACITIES_MW,
    CRITERIA,
    GATE_PATH,
    HOST_KEYS,
    HOST_SERIES,
    JOBS_DIR,
    LABELS,
    MIDDAY_FIRST,
    MIDDAY_LAST,
    N_RESAMPLES,
    N_SEEDS,
    PRINTED_DIR,
    PV_FREE,
    SCORE_COLUMNS,
    SCORES_PATH,
    SHAPES,
    STEP4_ERAS,
    STEP4_SEEDS,
    UNIT_THRESHOLD_MW,
    Interval,
    bright_dull_thirds,
    capacity_key,
    criterion_value,
    farm_norm_series,
    farm_on_grid,
    fold_state,
    grid_calendar_fields,
    half_width,
    host_bounds,
    host_job_name,
    paired_contrast,
    panel_series,
    read_json,
    require_passing_audit,
    score_columns,
    week_counts,
    week_table,
    write_json,
)

LINES: list[str] = []
LABEL_KEYS: Final[dict[str, str]] = {"smooth": "flag", "filtered": "filtered"}
"""The stored array of each label type."""
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "level": ("smooth", PV_FREE),
    "weather_farm": ("farm", "smooth_farm"),
    "weather_cams": ("cams", "smooth"),
    "total_farm": ("farm", PV_FREE),
    "total_cams": ("cams", PV_FREE),
}
"""Each contrast as (the job's shape, the base shape)."""
GROUPS: Final[tuple[str, ...]] = ("hour", "season", "midday", "third", "fold")


def say(text: str = "") -> None:
    """Print a line and keep it for the printed report."""
    print(text, flush=True)
    LINES.append(text)


@dataclass(frozen=True)
class HostContext:
    """The rows of a host's span and the masks every run of the host shares."""

    host: str
    first: int
    last: int
    week: np.ndarray
    half_hour: np.ndarray
    summer_half: np.ndarray
    scorable: np.ndarray
    midday: np.ndarray
    third: np.ndarray
    weeks: np.ndarray


def host_context(*, host: str, panel: dict[int, dict], farm_ok: np.ndarray) -> HostContext:
    """Build the masks of a host from the stored panel and the farm's finite mask on the span."""
    series = HOST_SERIES[host]
    first, last = host_bounds(host=host)
    y = panel_series(panel=panel, label=series, key="power")
    calendar = grid_calendar_fields(n_grid=len(y))
    window = slice(first, last)
    scorable = np.isfinite(y[window]) & ~calendar.clock_change_day[window] & farm_ok
    half_hour = calendar.half_hour[window]
    midday = (half_hour >= MIDDAY_FIRST) & (half_hour <= MIDDAY_LAST)
    full = np.zeros(len(y), dtype=bool)
    full[window] = scorable
    third = bright_dull_thirds(
        ghi=panel_series(panel=panel, label=series, key="ghi"),
        value=panel_series(panel=panel, label=series, key="pv_shape"),
        calendar=calendar,
        rows=full,
    )[window]
    weeks = np.unique(calendar.week[window][scorable & midday])
    return HostContext(
        host=host,
        first=first,
        last=last,
        week=calendar.week[window],
        half_hour=half_hour,
        summer_half=calendar.summer_half[window],
        scorable=scorable,
        midday=midday,
        third=third,
        weeks=weeks,
    )


def check_job_farms(*, farm: str) -> None:
    """Stop if a stored host job was made with a farm other than `farm`, or records none."""
    wrong = []
    for path in sorted(JOBS_DIR.glob("*.npz")):
        if path.name.startswith("step4_"):
            continue
        with np.load(path) as data:
            if "farm" not in data.files or str(data["farm"]) != farm:
                wrong.append(path.name)
    if wrong:
        msg = f"these job files were not made with farm {farm}: {wrong}; delete them and rerun jobs"
        raise SystemExit(msg)


def load_job(*, name: str) -> dict[str, np.ndarray]:
    """Return a stored run's arrays."""
    with np.load(JOBS_DIR / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def run_columns(*, run: dict[str, np.ndarray], context: HostContext, label: str) -> np.ndarray:
    """Return the nine score columns of every span row of one run and one label type."""
    g = run["g"]
    return score_columns(
        truth1=g >= UNIT_THRESHOLD_MW,
        flag=run[LABEL_KEYS[label]].astype(bool),
        zero=g == 0.0,
        third=context.third,
    )


def run_weeks(*, run: dict[str, np.ndarray], context: HostContext, label: str) -> np.ndarray:
    """Return a run's per-week sums of the score columns over the scorable midday rows."""
    columns = run_columns(run=run, context=context, label=label)
    rows = context.scorable & context.midday
    return week_table(week=context.week[rows], columns=columns[rows], weeks=context.weeks)


def pooled_weeks(
    *,
    host: str,
    capacity_mw: float | None,
    shape: str,
    seeds: list[int | None],
    context: HostContext,
    label: str,
) -> np.ndarray | None:
    """Return the seeds' per-week sums added together, or None if a run's file is missing."""
    total = None
    for seed in seeds:
        name = host_job_name(host=host, capacity_mw=capacity_mw, shape=shape, seed=seed)
        if not (JOBS_DIR / f"{name}.npz").exists():
            return None
        table = run_weeks(run=load_job(name=name), context=context, label=label)
        total = table if total is None else total + table
    return total


def group_sums(
    *, runs: list[dict[str, np.ndarray]], context: HostContext, label: str
) -> dict[str, dict[str, list[float]]]:
    """Return the nine score sums over the runs, by hour, season, third, and fold state.

    Hour and season use every scorable half-hour. Third and fold state use the midday half-hours.
    """
    out: dict[str, dict[str, list[float]]] = {g: {} for g in GROUPS}
    hour = context.half_hour // 2
    for run in runs:
        columns = run_columns(run=run, context=context, label=label)
        scorable, midday = context.scorable, context.scorable & context.midday
        fold = fold_state(signed_mw=run["signed"])
        keys: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "hour": (hour, scorable),
            "season": (context.summer_half.astype(int), scorable),
            "midday": (np.zeros(len(hour), dtype=int), midday),
            "third": (context.third, midday & (context.third >= 0)),
            "fold": (fold, midday & (fold >= 0)),
        }
        for group, (key, rows) in keys.items():
            for value in np.unique(key[rows]):
                sel = rows & (key == value)
                total = columns[sel].sum(axis=0)
                entry = out[group].setdefault(str(int(value)), [0.0] * len(SCORE_COLUMNS))
                out[group][str(int(value))] = [a + b for a, b in zip(entry, total, strict=True)]
    return out


def spread_of_changes(*, run: dict[str, np.ndarray], context: HostContext) -> float:
    """Return the robust spread (1.4826 times the median absolute) of half-hourly changes at midday.

    Only pairs of consecutive scorable midday half-hours count.
    """
    y = run["y"]
    rows = context.scorable & context.midday
    both = rows[1:] & rows[:-1]
    change = np.abs(np.diff(y))[both]
    return float(1.4826 * np.median(change)) if len(change) else float("nan")


def metrics(*, sums: list[float]) -> dict[str, float]:
    """Return precision, recall, F1, and the false-flag share from nine score sums."""
    s = np.asarray(sums, dtype=float)
    tp, fp, fn = s[0], s[1], s[2]
    return {
        "precision": float(tp / (tp + fp)) if tp + fp > 0 else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn > 0 else float("nan"),
        "f1": float(criterion_value(sums=s, criterion="f1")),
        "false_flag": float(criterion_value(sums=s, criterion="false_flag")),
    }


# --------------------------------------------------------------------------- host scoring


def score_host(  # noqa: PLR0912
    *, host: str, gate: dict, panel: dict[int, dict], farm_ok_span: np.ndarray
) -> dict:
    """Score one host: descriptive sums, paired contrasts, and the no-engine runs."""
    context = host_context(host=host, panel=panel, farm_ok=farm_ok_span)
    host_index = HOST_KEYS.index(host)
    counts = week_counts(
        n_weeks=len(context.weeks), seed=BOOTSTRAP_SEED + host_index, n_resamples=N_RESAMPLES
    )
    seeds: list[int | None] = list(range(N_SEEDS))
    result: dict = {
        "n_weeks": len(context.weeks),
        "scored_midday_half_hours": int((context.scorable & context.midday).sum()),
        "cells": {},
        "contrasts": {},
        "no_engine_contrasts": {},
    }
    shapes_here: list[tuple[float | None, str]] = [(None, PV_FREE)]
    for cap in CAPACITIES_MW:
        key = capacity_key(capacity_mw=cap)
        shapes_here.extend((cap, s) for s in SHAPES if gate["run"][host][key][s])
    for cap, shape in shapes_here:
        cell: dict = {}
        for label in LABELS:
            runs = [
                load_job(name=host_job_name(host=host, capacity_mw=cap, shape=shape, seed=s))
                for s in seeds
                if (
                    JOBS_DIR
                    / f"{host_job_name(host=host, capacity_mw=cap, shape=shape, seed=s)}.npz"
                ).exists()
            ]
            if runs:
                cell[label] = {"groups": group_sums(runs=runs, context=context, label=label)}
        engine_runs = [
            load_job(name=host_job_name(host=host, capacity_mw=cap, shape=shape, seed=s))
            for s in seeds
            if (
                JOBS_DIR / f"{host_job_name(host=host, capacity_mw=cap, shape=shape, seed=s)}.npz"
            ).exists()
        ]
        cell["fit"] = [{"k": int(r["k_max"]), "unit_mw": float(r["unit"])} for r in engine_runs]
        none_name = host_job_name(host=host, capacity_mw=cap, shape=shape, seed=None)
        if (JOBS_DIR / f"{none_name}.npz").exists():
            none_run = load_job(name=none_name)
            cell["no_engine"] = {
                "fit": {"k": int(none_run["k_max"]), "unit_mw": float(none_run["unit"])},
                "spread_of_changes": spread_of_changes(run=none_run, context=context),
                "fold_census": fold_census(run=none_run, context=context),
            }
        if engine_runs:
            cell["fold_census"] = [fold_census(run=r, context=context) for r in engine_runs]
        key = capacity_key(capacity_mw=cap)
        result["cells"][f"{key}_{shape}"] = cell
    for cap in CAPACITIES_MW:
        key = capacity_key(capacity_mw=cap)
        result["contrasts"][key] = {}
        result["no_engine_contrasts"][key] = {}
        for name, (shape, base) in CONTRASTS.items():
            if shape not in (*SHAPES,) or not gate["run"][host][key][shape]:
                continue
            if base != PV_FREE and not gate["run"][host][key][base]:
                continue
            base_cap = None if base == PV_FREE else cap
            for label in LABELS:
                for tag, run_seeds in (("engine", seeds), ("no_engine", [None])):
                    job_w = pooled_weeks(
                        host=host,
                        capacity_mw=cap,
                        shape=shape,
                        seeds=run_seeds,
                        context=context,
                        label=label,
                    )
                    base_w = pooled_weeks(
                        host=host,
                        capacity_mw=base_cap,
                        shape=base,
                        seeds=run_seeds,
                        context=context,
                        label=label,
                    )
                    if job_w is None or base_w is None:
                        continue
                    table = result["contrasts" if tag == "engine" else "no_engine_contrasts"]
                    table[key].setdefault(name, {})[label] = {
                        criterion: paired_contrast(
                            criterion=criterion, job_weeks=job_w, base_weeks=base_w, counts=counts
                        )
                        for criterion in CRITERIA
                    }
    return result


def fold_census(*, run: dict[str, np.ndarray], context: HostContext) -> dict[str, float]:
    """Return the shares of the scorable midday half-hours in each fold state."""
    rows = context.scorable & context.midday
    state = fold_state(signed_mw=run["signed"])[rows]
    n = max(int(rows.sum()), 1)
    return {
        name: float((state == code).sum() / n)
        for code, name in enumerate(("forward", "near_zero", "reverse"))
    }


# --------------------------------------------------------------------------- step 4


def score_step4(*, panel: dict[int, dict]) -> dict:
    """Return the recall of injected running half-hours by CAMS third, and the era contrast."""
    estimates: dict[str, Estimate | None] = {}
    detail: dict[str, dict] = {}
    y = panel_series(panel=panel, label="D8", key="power")
    calendar = grid_calendar_fields(n_grid=len(y))
    ghi = panel_series(panel=panel, label="D8", key="ghi")
    pv = panel_series(panel=panel, label="D8", key="pv_shape")
    for era in STEP4_ERAS:
        first, last = era_bounds()[era]
        window = slice(first, last)
        usable = np.zeros(len(y), dtype=bool)
        usable[window] = np.isfinite(y[window]) & ~calendar.clock_change_day[window]
        third = bright_dull_thirds(ghi=ghi, value=pv, calendar=calendar, rows=usable)[window]
        weeks, columns, injected = [], [], 0
        for seed in range(STEP4_SEEDS):
            path = JOBS_DIR / f"step4_{era}_s{seed}.npz"
            if not path.exists():
                continue
            run = load_job(name=f"step4_{era}_s{seed}")
            half_hour = calendar.half_hour[window]
            midday = (half_hour >= MIDDAY_FIRST) & (half_hour <= MIDDAY_LAST)
            truth = (run["g"] >= UNIT_THRESHOLD_MW) & usable[window] & midday
            injected += int(truth.sum())
            weeks.append(calendar.week[window][truth])
            columns.append(
                share_columns(
                    state=run["flag"].astype(float)[truth],
                    mask_a=third[truth] == 2,
                    mask_b=third[truth] == 0,
                )
            )
        if not weeks:
            estimates[era] = None
            continue
        week = np.concatenate(weeks)
        estimates[era] = week_estimate(
            week=week,
            columns=np.concatenate(columns),
            statistic=share_difference_statistic,
            seed=BOOTSTRAP_SEED + 100 + STEP4_ERAS.index(era),
            n_resamples=N_RESAMPLES,
        )
        detail[era] = {
            "injected_running_midday_half_hours_all_seeds": injected,
            "bright": int(sum((c[:, 1] > 0).sum() for c in columns)),
            "dull": int(sum((c[:, 3] > 0).sum() for c in columns)),
        }
    net, solar = estimates.get("net"), estimates.get("solar")
    contrast: Interval | None = None
    if net is not None and solar is not None:
        diff = difference_estimate(first=solar, second=net)
        contrast = {"point": diff.point, "lo": diff.lo, "hi": diff.hi}
    return {
        "recall_bright_minus_dull": {
            era: (None if est is None else est.to_dict()) for era, est in estimates.items()
        },
        "contrast_solar_minus_net": contrast,
        "detail": detail,
    }


# --------------------------------------------------------------------------- printing


def twin_f1_unmasked(*, panel: dict[int, dict]) -> float | None:
    """Return the midday F1 of D8-pre's solar-free twin on every observed half-hour (3 seeds).

    This is the figure that the plan compares with experiment 697's control C1 (0.767). It uses
    every half-hour with a reading that is not on a clock-change day, whatever the farm's gaps.
    """
    first, last = host_bounds(host="d8pre")
    y = panel_series(panel=panel, label="D8", key="power")
    calendar = grid_calendar_fields(n_grid=len(y))
    window = slice(first, last)
    half_hour = calendar.half_hour[window]
    rows = (
        np.isfinite(y[window])
        & ~calendar.clock_change_day[window]
        & (half_hour >= MIDDAY_FIRST)
        & (half_hour <= MIDDAY_LAST)
    )
    tp = fp = fn = 0
    for seed in range(N_SEEDS):
        name = host_job_name(host="d8pre", capacity_mw=None, shape=PV_FREE, seed=seed)
        if not (JOBS_DIR / f"{name}.npz").exists():
            return None
        run = load_job(name=name)
        truth = run["g"] >= UNIT_THRESHOLD_MW
        flag = run["flag"].astype(bool)
        tp += int((truth & flag)[rows].sum())
        fp += int((~truth & flag)[rows].sum())
        fn += int((truth & ~flag)[rows].sum())
    return 2 * tp / (2 * tp + fp + fn)


def fmt_interval(*, interval: Interval) -> str:
    """Return an interval as `point (lo to hi)`."""
    return f"{interval['point']:+.3f} ({interval['lo']:+.3f} to {interval['hi']:+.3f})"


def print_half_widths(*, scores: dict) -> None:
    """Print the paired half-widths of every contrast, before any score is read."""
    say("## Paired half-widths of the smoothed-label contrasts (95%, weeks resampled together)")
    say("host, capacity, contrast: F1, false-flag share, bright-minus-dull flagged share")
    for host, doc in scores["hosts"].items():
        for key, table in doc["contrasts"].items():
            for name, labels in table.items():
                widths = [half_width(interval=labels["smooth"][c]) for c in CRITERIA]
                say(f"{host}, {key}, {name}: " + ", ".join(f"{w:.3f}" for w in widths))


def print_scores(*, scores: dict) -> None:  # noqa: PLR0912
    """Print the contrasts and the pooled scores by group."""
    say()
    say("## Contrasts, harm positive (F1 falls, false-flag share rises, bright-minus-dull rises)")
    say("thresholds: F1 0.05, false-flag 0.02, bright-minus-dull 0.05; smoothed labels first")
    for host, doc in scores["hosts"].items():
        say(
            f"### {host}: {doc['n_weeks']} weeks, "
            f"{doc['scored_midday_half_hours']} midday half-hours"
        )
        for key, table in doc["contrasts"].items():
            for name, labels in table.items():
                for label in LABELS:
                    parts = [f"{c} {fmt_interval(interval=labels[label][c])}" for c in CRITERIA]
                    say(f"{host}, {key}, {name}, {label}: " + "; ".join(parts))
        for key, table in doc["no_engine_contrasts"].items():
            for name, labels in table.items():
                parts = [
                    f"{c} {fmt_interval(interval=labels['smooth'][c])}"
                    for c in ("false_flag", "bmd")
                ]
                say(f"{host}, {key}, {name}, no engine, smooth: " + "; ".join(parts))
    say()
    say("## Pooled midday scores by run (3 seeds), smoothed labels")
    for host, doc in scores["hosts"].items():
        for cell, content in doc["cells"].items():
            groups = content.get("smooth", {}).get("groups")
            if groups is None:
                continue
            m = metrics(sums=groups["midday"]["0"])
            fits = ", ".join(f"K={f['k']} unit={f['unit_mw']:.2f}" for f in content["fit"])
            say(
                f"{host}, {cell}: P {m['precision']:.3f}, R {m['recall']:.3f}, F1 {m['f1']:.3f}, "
                f"false-flag {m['false_flag']:.3f}; fits: {fits}"
            )
    say()
    say("## Fold census of scorable midday half-hours (shares of signed real power)")
    for host, doc in scores["hosts"].items():
        for cell, content in doc["cells"].items():
            if "fold_census" in content:
                census = content["fold_census"][0]
                say(
                    f"{host}, {cell}, seed 0: "
                    + ", ".join(f"{k} {v:.3f}" for k, v in census.items())
                )
    say()
    say("## Spread of half-hourly changes at midday, runs with no engine (MVA)")
    for host, doc in scores["hosts"].items():
        base = doc["cells"].get(f"none_{PV_FREE}", {}).get("no_engine", {}).get("spread_of_changes")
        for cell, content in doc["cells"].items():
            value = content.get("no_engine", {}).get("spread_of_changes")
            if value is not None and base is not None:
                say(f"{host}, {cell}: {value:.3f}; less the solar-free twin's {value - base:+.3f}")


def main() -> None:
    """Score every host and step 4, then write `scores.json` and the printed report."""
    require_passing_audit()
    gate = read_json(path=GATE_PATH)
    check_job_farms(farm=str(gate["farm"]))
    panel = load_panel()

    farm = farm_on_grid(
        series=farm_norm_series(farm=str(gate["farm"])),
        n_grid=len(panel_series(panel=panel, label="D8", key="power")),
    )
    scores: dict = {"hosts": {}}
    for host in HOST_KEYS:
        first, last = host_bounds(host=host)
        if not (
            JOBS_DIR / f"{host_job_name(host=host, capacity_mw=None, shape=PV_FREE, seed=0)}.npz"
        ).exists():
            continue
        scores["hosts"][host] = score_host(
            host=host, gate=gate, panel=panel, farm_ok_span=np.isfinite(farm[first:last])
        )
    print_half_widths(scores=scores)
    scores["step4"] = score_step4(panel=panel)
    twin = twin_f1_unmasked(panel=panel)
    scores["twin_f1_d8pre"] = twin
    print_scores(scores=scores)
    say()
    say(f"D8-pre solar-free twin midday F1 on every observed half-hour: {twin}")
    step4 = scores["step4"]
    say("## Step 4: recall of injected running half-hours, bright minus dull third, 10:00 to 15:00")
    for era, est in step4["recall_bright_minus_dull"].items():
        say(f"  {era}: " + ("no runs" if est is None else fmt_interval(interval=est)))
    contrast = step4["contrast_solar_minus_net"]
    say(
        "  solar minus net: "
        + ("not available" if contrast is None else fmt_interval(interval=contrast))
    )
    write_json(path=SCORES_PATH, payload=scores)
    PRINTED_DIR.mkdir(parents=True, exist_ok=True)
    (PRINTED_DIR / "score_report.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()

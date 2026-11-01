# ruff: noqa: E501, PLC0206
"""Experiment 456, step 3: run the scan, its sensitivity grid, and the three validations.

Reads `residuals_b.parquet` (step 1), the corrected event table of experiment 400, and the 25
stored cross-fitted lagless tables of experiment 449 (method a). Writes CSV and parquet tables to
the experiment's work folder; `e00456_charts.py` draws the figures from them.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:ml-experiments/00456-primary-demand-unmirrored-level-step-scan \
    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 uv run --no-sync python \
    ml-experiments/00456-primary-demand-unmirrored-level-step-scan/e00456_run.py
```
"""

import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00456_scan as scan  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00456-primary-demand-unmirrored-level-step-scan" / "work"
)
EVENTS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored"
    / "work"
    / "events_corrected.parquet"
)
LAGLESS_ERA5: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00449-pretraining-with-lags-and-surprise-inputs-on-the-s1-network"
    / "work"
    / "lagless_era5"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEED: Final[int] = 456
N_DAYS: Final[int] = 121_248 // 48
BOUNDARY_DAYS: Final[tuple[int, ...]] = tuple(
    int(common.grid_index(times=datetime(y, m, 1, tzinfo=UTC))) // 48
    for y in range(2019, 2027)
    for m in (1, 7)
    if datetime(y, m, 1, tzinfo=UTC) > common.GRID_START
)
MATCH_DAYS: Final[int] = 2
SIZES: Final[tuple[float, ...]] = (1.0, 1.5, 2.0)
"""Injected size as a multiple of the detection threshold max(3 x spread, 2 points)."""
DURATIONS: Final[tuple[int, ...]] = (7, 14, 30, 90)
WINDOWS_PER_CELL: Final[int] = 6
N_BOOTSTRAP: Final[int] = 20
BLOCK_DAYS: Final[int] = 14
LOOSE: Final[scan.Params] = scan.Params(
    spread_multiple=2.0, floor_points=1.0, penalty_multiplier=2.0
)
BASE: Final[scan.Params] = scan.Params()
GRID: Final[dict[str, scan.Params]] = {
    "baseline": BASE,
    "spread x2": replace(BASE, spread_multiple=2.0),
    "spread x4": replace(BASE, spread_multiple=4.0),
    "floor 1": replace(BASE, floor_points=1.0),
    "floor 4": replace(BASE, floor_points=4.0),
    "segment 14 d": replace(BASE, min_segment_days=14),
    "penalty x2": replace(BASE, penalty_multiplier=2.0),
    "penalty x5": replace(BASE, penalty_multiplier=5.0),
}


def load() -> tuple[dict[str, np.ndarray], dict[str, float], list[str]]:
    """Return each Primary's daily levels and capacity, and the Primary labels."""
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    capacity = dict(zip(series["site"], series["effective_capacity_mw"], strict=True))
    table = pl.read_parquet(WORK_DIR / "residuals_b.parquet").drop_nulls("prediction")
    levels: dict[str, np.ndarray] = {}
    for site in series["site"].to_list():
        rows = table.filter(pl.col("site") == site)
        levels[site] = scan.daily_levels(
            grid=rows["grid"].to_numpy(),
            residual=(rows["power"] - rows["prediction"]).to_numpy().astype(np.float64),
            n_days=N_DAYS,
        )
    return (
        levels,
        {k: float(v) for k, v in capacity.items()},
        sorted(levels, key=lambda x: int(x[1:])),
    )


def label_events(*, sites: list[str]) -> pl.DataFrame:
    """Return the labelled event rows with their onset and end day and signed depth."""
    events = pl.read_parquet(EVENTS).filter(pl.col("label").is_in(sites))
    return events.with_columns(onset_day=pl.col("onset_idx") // 48, end_day=pl.col("end_idx") // 48)


def run_scan(
    *,
    levels: dict[str, np.ndarray],
    capacity: dict[str, float],
    params: scan.Params,
    graph: dict[str, set[str]],
    all_pairs: bool,
    sum_rule: bool = False,
    start_days: int | None = None,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Detect and classify the steps of every Primary; return the table and each spread."""
    steps: list[scan.Step] = []
    spreads: dict[str, float] = {}
    for site, lv in levels.items():
        found, spreads[site] = scan.detect(
            site=site, levels=lv, capacity_mw=capacity[site], params=params
        )
        steps.extend(found)
    classes: dict[tuple[str, int], str] = {}
    info: dict[tuple[str, int], tuple[str, float]] = {}
    for st in steps:
        neighbours = set(levels) - {st.site} if all_pairs else graph.get(st.site, set())
        cls, other, change = scan.classify(
            step=st,
            levels_of=levels,
            capacity_of=capacity,
            neighbours=neighbours,
            params=params,
            start_days=start_days,
        )
        if sum_rule and cls == "unmirrored":
            window = params.mirror_start_days if start_days is None else start_days
            total = 0.0
            for day in range(st.start_day - window, st.start_day + window + 1):
                changes = [
                    scan.flank_change_mw(levels=levels[o], day=day, capacity_mw=capacity[o])
                    for o in neighbours
                    if o in levels
                ]
                opposite = [
                    c for c in changes
                    if np.isfinite(c) and c * st.mw < 0 and abs(c) >= 0.25 * abs(st.mw)
                ]  # fmt: skip
                total = max(total, abs(sum(opposite)))
            if total > 0 and 1 / params.mirror_ratio <= total / abs(st.mw) <= params.mirror_ratio:
                cls = "mirrored"
        classes[(st.site, st.start_day)] = cls
        info[(st.site, st.start_day)] = (other, change)
    scan.mark_temporary(steps=steps, classes=classes)
    rows = [
        {
            "site": st.site,
            "start_day": st.start_day,
            "end_day": st.end_day,
            "points": st.points,
            "mw": st.mw,
            "class": classes[(st.site, st.start_day)],
            "nearest_neighbour": info[(st.site, st.start_day)][0],
            "neighbour_change_mw": info[(st.site, st.start_day)][1],
        }
        for st in steps
    ]
    schema = {
        "site": pl.String, "start_day": pl.Int64, "end_day": pl.Int64, "points": pl.Float64,
        "mw": pl.Float64, "class": pl.String, "nearest_neighbour": pl.String,
        "neighbour_change_mw": pl.Float64,
    }  # fmt: skip
    return pl.DataFrame(rows, schema=schema), spreads


def match_labels(*, table: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Add `labelled`: the step starts within 2 days of a labelled onset or end at its Primary."""
    flags = []
    for row in table.iter_rows(named=True):
        site_events = events.filter(pl.col("label") == row["site"])
        hit = site_events.filter(
            ((pl.col("onset_day") - row["start_day"]).abs() <= MATCH_DAYS)
            | ((pl.col("end_day") + 1 - row["start_day"]).abs() <= MATCH_DAYS)
        )
        flags.append(hit.height > 0)
    return table.with_columns(labelled=pl.Series(flags, dtype=pl.Boolean))


def summarise(
    *, table: pl.DataFrame, levels: dict[str, np.ndarray], name: str
) -> dict[str, object]:
    """Return counts and rates per Primary-year by class, with the unmirrored share of days."""
    valid_days = sum(int(np.isfinite(v).sum()) for v in levels.values())
    years = valid_days / 365.25
    out: dict[str, object] = {"setting": name, "primary_years": round(years, 1)}
    for cls in ("mirrored", "unmirrored", "temporary", "unassessable"):
        sub = table.filter(pl.col("class") == cls)
        out[cls] = sub.height
        out[f"{cls}_per_year"] = round(sub.height / years, 3)
    if "labelled" in table.columns:
        un = table.filter(pl.col("class") == "unmirrored")
        out["unmirrored_unlabelled"] = un.filter(~pl.col("labelled")).height
        out["unmirrored_unlabelled_per_year"] = round(
            un.filter(~pl.col("labelled")).height / years, 3
        )
        mi = table.filter(pl.col("class") == "mirrored")
        out["mirrored_unlabelled"] = mi.filter(~pl.col("labelled")).height
        out["mirrored_unlabelled_per_year"] = round(
            mi.filter(~pl.col("labelled")).height / years, 3
        )
    inside = sum(
        int(
            np.isfinite(
                levels[row["site"]][
                    row["start_day"] : min(row["end_day"], row["start_day"] + 59) + 1
                ]
            ).sum()
        )
        for row in table.filter(pl.col("class") == "unmirrored").iter_rows(named=True)
    )
    out["share_days_in_unmirrored_step"] = round(inside / valid_days, 4)
    return out


def recall_on_labels(*, table: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Return recall on labelled event rows and events, in all and for events of 7 days or more."""
    rows = []
    for row in events.iter_rows(named=True):
        hit = table.filter(
            (pl.col("site") == row["label"])
            & ((pl.col("start_day") - row["onset_day"]).abs() <= MATCH_DAYS)
            & (pl.col("points").sign() == np.sign(row["depth_mw"]))
        )
        hit_mirrored = hit.filter(pl.col("class") == "mirrored")
        hit_size = hit.filter(
            (pl.col("mw").abs() / abs(row["depth_mw"]) >= 0.5)
            & (pl.col("mw").abs() / abs(row["depth_mw"]) <= 2.0)
        )
        rows.append(
            {
                "event_id": row["event_id"], "site": row["label"],
                "onset_day": row["onset_day"], "length_days": row["length_days"],
                "found": hit.height > 0, "found_mirrored": hit_mirrored.height > 0,
                "found_size_within_2x": hit_size.height > 0,
            }
        )  # fmt: skip
    per_row = pl.DataFrame(rows)
    # An event is the set of rows whose onsets lie within 2 days of one another.
    per_row = per_row.sort("onset_day").with_columns(
        event=(pl.col("onset_day").diff().fill_null(99) > MATCH_DAYS).cum_sum()
    )
    out = []
    for name, sub in (
        ("all", per_row),
        ("7 days or more", per_row.filter(pl.col("length_days") >= 7)),
    ):
        events_found = sub.group_by("event").agg(
            pl.col("found").any(),
            pl.col("found_mirrored").any(),
            pl.col("found_size_within_2x").any(),
        )
        out.append(
            {
                "subset": name, "rows": sub.height, "rows_found": int(sub["found"].sum()),
                "rows_found_mirrored": int(sub["found_mirrored"].sum()),
                "rows_found_size_within_2x": int(sub["found_size_within_2x"].sum()),
                "events": events_found.height, "events_found": int(events_found["found"].sum()),
                "events_found_mirrored": int(events_found["found_mirrored"].sum()),
            }
        )  # fmt: skip
    return pl.DataFrame(out)


def quiet_days(
    *, levels: dict[str, np.ndarray], capacity: dict[str, float], events: pl.DataFrame
) -> dict[str, np.ndarray]:
    """Return per Primary a boolean day mask: no loose-scan step start within 30 days (the pad)."""
    quiet: dict[str, np.ndarray] = {}
    for site, lv in levels.items():
        found, _ = scan.detect(site=site, levels=lv, capacity_mw=capacity[site], params=LOOSE)
        mask = np.ones(len(lv), dtype=bool)
        for st in found:
            mask[max(0, st.start_day - 30) : st.start_day + 31] = False
            mask[max(0, st.end_day - 30) : st.end_day + 31] = False
        for ev in events.filter(pl.col("label") == site).iter_rows(named=True):
            mask[max(0, ev["onset_day"] - 30) : ev["end_day"] + 31] = False
        quiet[site] = mask
    return quiet


def window_ok(*, mask: np.ndarray, levels: np.ndarray, start: int, duration: int) -> bool:
    """Return whether [start - 30, start + duration + 30) is quiet and mostly observed."""
    lo, hi = start - 30, start + duration + 30
    return (
        lo >= 0
        and hi <= len(mask)
        and bool(mask[lo:hi].all())
        and bool(np.isfinite(levels[lo:hi]).mean() > 0.8)
    )


def injection(
    *,
    levels: dict[str, np.ndarray],
    capacity: dict[str, float],
    spreads: dict[str, float],
    graph: dict[str, set[str]],
    quiet: dict[str, np.ndarray],
    quiet_only: bool,
) -> pl.DataFrame:
    """Inject known steps (alone and mirrored) and return recall and mirror class per cell."""
    rng = np.random.default_rng(SEED)
    rows = []
    sigmas = {s: scan.noise_sigma(levels=lv) for s, lv in levels.items()}
    for site in levels:
        neighbours = [n for n in sorted(graph.get(site, set())) if n in levels]
        for size in SIZES:
            for duration in DURATIONS:
                for mirror in (False, True):
                    if mirror and not neighbours:
                        continue
                    other = neighbours[0] if mirror else None
                    if mirror:
                        for cand in neighbours:
                            if np.isfinite(levels[cand]).mean() > 0.5:
                                other = cand
                                break
                    candidates = [
                        t for t in range(40, N_DAYS - duration - 40, 5)
                        if (
                            window_ok(mask=quiet[site], levels=levels[site], start=t, duration=duration)
                            and (
                                other is None
                                or window_ok(
                                    mask=quiet[other], levels=levels[other], start=t, duration=duration
                                )
                            )
                        )
                        or (
                            not quiet_only
                            and np.isfinite(levels[site][t - 30 : t + duration + 30]).mean() > 0.8
                            and (other is None or np.isfinite(levels[other][t - 30 : t + duration + 30]).mean() > 0.8)
                        )
                    ]  # fmt: skip
                    if not candidates:
                        continue
                    picks = rng.choice(
                        candidates, size=min(WINDOWS_PER_CELL, len(candidates)), replace=False
                    )
                    for t in picks:
                        sign = float(rng.choice([-1.0, 1.0]))
                        points = (
                            sign
                            * size
                            * max(BASE.spread_multiple * spreads[site], BASE.floor_points)
                        )
                        lv = levels[site].copy()
                        lv[t : t + duration] += points
                        trial = dict(levels)
                        trial[site] = lv
                        if other is not None:
                            mw = points * capacity[site] / 100.0
                            lo = levels[other].copy()
                            lo[t : t + duration] -= mw * 100.0 / capacity[other]
                            trial[other] = lo
                        found, _ = scan.detect(
                            site=site, levels=lv, capacity_mw=capacity[site], params=BASE,
                            spread=spreads[site], sigma=sigmas[site],
                        )  # fmt: skip
                        hit = [
                            s for s in found
                            if abs(s.start_day - t) <= MATCH_DAYS and s.points * points > 0
                            and 0.5 <= abs(s.points) / abs(points) <= 1.5
                        ]  # fmt: skip
                        cls = ""
                        if hit:
                            cls, _, _ = scan.classify(
                                step=hit[0], levels_of=trial, capacity_of=capacity,
                                neighbours=set(neighbours), params=BASE,
                            )  # fmt: skip
                        rows.append(
                            {
                                "site": site, "size": size, "duration": duration,
                                "mirrored_injection": mirror, "start": int(t),
                                "detected": bool(hit), "class": cls,
                            }
                        )  # fmt: skip
    return pl.DataFrame(rows)


def phase_randomised(*, levels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a surrogate with the same power spectrum as the gap-filled series and no steps.

    The Fourier phases are replaced by random ones, so slow drift and seasonal swings keep their
    strength but abrupt level changes are spread into the noise.
    """
    days = np.arange(len(levels))
    ok = np.isfinite(levels)
    filled = np.interp(days, days[ok], levels[ok])[ok.argmax() : len(levels) - ok[::-1].argmax()]
    spectrum = np.fft.rfft(filled - filled.mean())
    phases = rng.uniform(0, 2 * np.pi, len(spectrum))
    phases[0] = 0.0
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=len(filled)) + filled.mean()


def false_alarms(*, levels: dict[str, np.ndarray], capacity: dict[str, float]) -> pl.DataFrame:
    """Return steps per Primary-year found in phase-randomised surrogates of each residual."""
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for site, lv in levels.items():
        spread, sigma = scan.spread_of(levels=lv), scan.noise_sigma(levels=lv)
        counts = []
        for _ in range(N_BOOTSTRAP):
            sur = phase_randomised(levels=lv, rng=rng)
            found, _ = scan.detect(
                site=site, levels=sur, capacity_mw=capacity[site], params=BASE,
                spread=spread, sigma=sigma,
            )  # fmt: skip
            counts.append(len(found))
        rows.append(
            {
                "site": site, "years": len(sur) / 365.25,
                "false_alarms_per_year": float(np.mean(counts)) / (len(sur) / 365.25),
            }
        )  # fmt: skip
    return pl.DataFrame(rows)


def chance_mirror(
    *,
    table: pl.DataFrame,
    levels: dict[str, np.ndarray],
    capacity: dict[str, float],
    graph: dict[str, set[str]],
) -> pl.DataFrame:
    """Return the share of steps that a neighbour 'mirrors' when the neighbours are read elsewhere.

    The step's start is moved by +-30, 45, 60, and 90 days before the mirror test, so a match
    there is chance coincidence. The neighbour sets are the table pairs and all 15 other Primaries.
    """
    rows = []
    steps = table.filter(pl.col("class").is_in(["mirrored", "unmirrored", "temporary"]))
    for name, all_pairs in (("table pairs", False), ("all 15", True)):
        for offset in (0, -90, -60, -45, -30, 30, 45, 60, 90):
            hits = 0
            for row in steps.iter_rows(named=True):
                step = scan.Step(
                    site=row["site"], start_day=row["start_day"] + offset,
                    end_day=row["end_day"], points=row["points"], mw=row["mw"],
                )  # fmt: skip
                neighbours = set(levels) - {step.site} if all_pairs else graph.get(step.site, set())
                cls, _, _ = scan.classify(
                    step=step, levels_of=levels, capacity_of=capacity, neighbours=neighbours,
                    params=BASE,
                )  # fmt: skip
                hits += cls == "mirrored"
            rows.append(
                {"neighbours": name, "start_offset_days": offset, "steps": steps.height,
                 "mirrored_share": hits / steps.height}
            )  # fmt: skip
    return pl.DataFrame(rows)


def method_a_comparison(*, levels: dict[str, np.ndarray], sites: list[str]) -> pl.DataFrame:
    """Compare daily levels of method (a) and method (b) on the five stored months."""
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    arrays = common.load_series_arrays(series=series)
    rows = []
    for month in MONTHS:
        tables = [
            pl.read_parquet(LAGLESS_ERA5 / f"month={month}" / f"seed={s}.parquet") for s in range(5)
        ]
        ref = tables[0].sort("primary", "grid")
        q50 = np.mean([t.sort("primary", "grid")["q50"].to_numpy() for t in tables], axis=0)
        ref = ref.with_columns(q50=pl.Series(q50))
        for idx, site in enumerate(sites):
            sub = ref.filter(pl.col("primary") == idx)
            grid = sub["grid"].to_numpy()
            power = arrays[site].power[grid].astype(np.float64)
            ok = np.isfinite(power)
            if ok.sum() < 48 * 10:
                continue
            level_a = scan.daily_levels(
                grid=grid[ok], residual=power[ok] - sub["q50"].to_numpy()[ok], n_days=N_DAYS
            )
            level_b = levels[site]
            both = np.isfinite(level_a) & np.isfinite(level_b)
            if both.sum() < 10:
                continue
            rows.append(
                {
                    "month": month, "site": site, "days": int(both.sum()),
                    "correlation": float(np.corrcoef(level_a[both], level_b[both])[0, 1]),
                    "mean_a_minus_b_points": float(np.mean(level_a[both] - level_b[both])),
                    "sd_a_points": float(np.std(level_a[both])),
                    "sd_b_points": float(np.std(level_b[both])),
                }
            )  # fmt: skip
    return pl.DataFrame(rows)


def main() -> None:
    """Run every part of the plan and write its tables."""
    started = time.time()
    levels, capacity, sites = load()
    graph = scan.neighbour_sets()
    events = label_events(sites=sites)
    pl.DataFrame(
        {
            "site": sites,
            "valid_days": [int(np.isfinite(levels[s]).sum()) for s in sites],
            "spread_points": [scan.spread_of(levels=levels[s]) for s in sites],
        }
    ).write_csv(WORK_DIR / "spreads.csv")
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    observed = pl.read_parquet(
        common.OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"]
    )
    nulls = []
    for row in series.iter_rows(named=True):
        sub = observed.filter(pl.col("time_series_id") == row["time_series_id"])
        valid = sub.filter(pl.col("power").is_not_null())["valid_time"]
        span = sub.filter(pl.col("valid_time").is_between(valid.min(), valid.max()))
        nulls.append(
            {
                "site": row["site"],
                "half_hours_in_span": span.height,
                "null_power": int(span["power"].null_count()),
            }
        )
    pl.DataFrame(nulls).write_csv(WORK_DIR / "null_half_hours.csv")
    t0 = time.time()
    scan.detect(site=sites[0], levels=levels[sites[0]], capacity_mw=capacity[sites[0]], params=BASE)
    print(f"one Primary's change-point step: {time.time() - t0:.2f} s", flush=True)

    summary, tables = [], {}
    for name, params in GRID.items():
        table, spreads = run_scan(
            levels=levels, capacity=capacity, params=params, graph=graph, all_pairs=False
        )
        table = match_labels(table=table, events=events)
        tables[name] = table
        summary.append(summarise(table=table, levels=levels, name=name))
    for name, kwargs in {
        "all 15 neighbours": {"all_pairs": True},
        "mirror start 3 d": {"all_pairs": False, "start_days": 3},
        "sum rule": {"all_pairs": False, "sum_rule": True},
        "sum rule 3 d all 15": {"all_pairs": True, "sum_rule": True, "start_days": 3},
    }.items():
        table, _ = run_scan(levels=levels, capacity=capacity, params=BASE, graph=graph, **kwargs)  # ty: ignore[invalid-argument-type]
        table = match_labels(table=table, events=events)
        tables[name] = table
        summary.append(summarise(table=table, levels=levels, name=name))
    pl.DataFrame(summary).write_csv(WORK_DIR / "summary_by_setting.csv")
    tables["baseline"] = tables["baseline"].with_columns(
        near_block_boundary=pl.col("start_day").map_elements(
            lambda d: min(abs(d - b) for b in BOUNDARY_DAYS) <= 3, return_dtype=pl.Boolean
        )
    )
    no_boundary = tables["baseline"].filter(~pl.col("near_block_boundary"))
    pl.DataFrame(
        [summarise(table=no_boundary, levels=levels, name="baseline without block-boundary starts")]
    ).write_csv(WORK_DIR / "summary_no_boundary.csv")
    base = tables["baseline"].with_columns(
        start_date=pl.lit(common.GRID_START).dt.offset_by(pl.col("start_day").cast(pl.String) + "d")
    )
    base.write_csv(WORK_DIR / "candidate_steps.csv")
    recall_on_labels(table=tables["baseline"], events=events).write_csv(
        WORK_DIR / "recall_labels.csv"
    )
    recall_on_labels(table=tables["spread x2"], events=events).write_csv(
        WORK_DIR / "recall_labels_spread2.csv"
    )

    spreads = {s: scan.spread_of(levels=levels[s]) for s in sites}
    quiet = quiet_days(levels=levels, capacity=capacity, events=events)
    pl.DataFrame(
        {
            "site": sites,
            "valid_half_hours": [int(np.isfinite(levels[s]).sum()) * 48 for s in sites],
            "quiet_half_hours": [
                int((quiet[s] & np.isfinite(levels[s])).sum()) * 48 for s in sites
            ],
        }
    ).write_csv(WORK_DIR / "quiet_windows.csv")
    for quiet_only, label in ((True, "quiet"), (False, "random")):
        injection(
            levels=levels, capacity=capacity, spreads=spreads, graph=graph, quiet=quiet,
            quiet_only=quiet_only,
        ).write_csv(WORK_DIR / f"injection_{label}.csv")  # fmt: skip
    false_alarms(levels=levels, capacity=capacity).write_csv(WORK_DIR / "false_alarms.csv")
    chance_mirror(
        table=tables["baseline"], levels=levels, capacity=capacity, graph=graph
    ).write_csv(WORK_DIR / "chance_mirror.csv")
    method_a_comparison(levels=levels, sites=sites).write_csv(WORK_DIR / "method_a_months.csv")
    print(f"done in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()

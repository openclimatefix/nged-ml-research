r"""Experiment 698's data audit and curve gate. It runs no tracker and scores no run.

    E=ml-experiments
    export PYTHONPATH=$(ls -d $E/00698-* $E/00697-* $E/00694-* $E/00695-* | paste -sd:)
    uv run --no-sync python $E/00698-*/e00698_audit.py

The audit prints its report to `printed/audit.txt` (committed, with no date and no Gas A or farm
series), writes its numbers to `audit.json` and the gate to `gate.json` in the private data folder,
and exits with an error if a check that the design asserts fails. The checks are:

1. Row counts, missing shares, and ranges of the reading, CAMS irradiance, and CAMS solar shape at
   D8 and D3, and of the farm series over both host spans. The farm is the one of experiment 662's
   four farms with the highest finite share over both spans.
2. The curve gate: D8's solar-era engine-free curve of median reading by fifth of the CAMS shape
   (May to September, 10:00 to 15:00 local), and each host's, capacity's, and shape's curve built
   with one set of fifth edges (D8's solar-era edges). A shape runs only if its curve lies within
   1 MVA of D8's in each of the first four fifths. The smooth twin runs when the farm or the CAMS
   shape passes.
3. The night-matched constant of D8's reading before the engine, and D8's monthly midday-minus-night
   gap by era.
4. The false-flag share of experiment 697's level-matched control under both definitions
   (`truth_mw` = 0 and "under 1 unit"), and the midday F1 of that control and of control C1.
5. The spread of half-hourly changes of D8's reading on engine-off, no-drop midday half-hours, by
   era. (The same spread for each solar run beside its solar-free twin is printed by
   `e00698_score.py`, which holds the runs.)
6. Timestamps: the correlation of the changes of the reading and of the farm series with the changes
   of the CAMS shape at lags of -4 to +4 half-hours, on half-hourly changes and on hourly sums, with
   the strongest absolute value asserted at lag 0.
7. The three label-only checks of the design and D8's statistics with and without experiment 159's
   drops, in `e00698_labels.py`'s functions.
8. Step 4's selection: the off-spells, the selected days, their calendar months, and their share
   inside a drop.
9. One line comparing CAMS irradiance with ERA5 `ghi_w_m2` at the nearest stored cell.
"""

import sys
from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl
from e00694_common import load_panel  # ty: ignore[unresolved-import]
from e00697_common import (  # ty: ignore[unresolved-import]
    ERA_NAMES,
    era_bounds,
    half_hour_starts,
)
from e00697_engine import record_days  # ty: ignore[unresolved-import]
from e00698_common import (  # ty: ignore[unresolved-import]
    AUDIT_PATH,
    CAPACITIES_MW,
    COVERAGE_MIN,
    E697_FRAME_C1_NULL,
    E697_FRAME_C1S_SOLAR,
    E697_FRAME_D8,
    FARM_LETTERS,
    FOLD_NEAR_ZERO_MW,
    GATE_PATH,
    HOST_KEYS,
    HOST_SERIES,
    MIDDAY_FIRST,
    MIDDAY_LAST,
    NIGHT_LAST,
    PRINTED_DIR,
    SHAPES,
    STEP4_MIN_SPELL,
    TWIN_F1_EXPECTED,
    UNIT_THRESHOLD_MW,
    capacity_key,
    curve_medians,
    farm_finite_share,
    farm_norm_series,
    farm_on_grid,
    fifth_edges,
    fold_reading,
    gate_passes,
    grid_calendar_fields,
    host_bounds,
    longest_gap,
    off_spell_lengths,
    off_spell_rows,
    panel_series,
    solar_curve,
    summer_midday_rows,
    whole_day_rows,
    write_json,
)
from e00698_labels import label_only_checks  # ty: ignore[unresolved-import]

LINES: list[str] = []
FAILED: list[str] = []
NUMBERS: dict[str, object] = {}
LAGS: Final[tuple[int, ...]] = tuple(range(-4, 5))
MIN_ROWS_PER_MONTH: Final[int] = 40
"""A month needs this many midday and night half-hours to give a gap."""
DESIGN_SPELLS: Final[dict[str, tuple[int, int]]] = {"net": (52, 4913), "solar": (29, 2083)}
"""The design's own counts of off-spells of 48 half-hours or more and their half-hours."""
MONTH_NAMES: Final[tuple[str, ...]] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def say(text: str = "") -> None:
    """Print a line and keep it for the printed report."""
    print(text, flush=True)
    LINES.append(text)


def check(*, name: str, ok: bool, detail: str = "") -> None:
    """Print one check, and record it as failed when it does not hold."""
    say(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def fmt(*, values: np.ndarray, digits: int = 2) -> str:
    """Return a list of numbers as text."""
    return "[" + ", ".join("nan" if not np.isfinite(v) else f"{v:.{digits}f}" for v in values) + "]"


# --------------------------------------------------------------------------- step 1: the data


def step_1_data(*, panel: dict[int, dict]) -> dict[str, np.ndarray]:
    """Print row counts, missing shares, and ranges of the host series, and return the farm."""
    say("## 1. Data: counts, missing shares, and ranges")
    say("series, span, rows, missing share, min, median, max")
    for host in HOST_KEYS:
        label = HOST_SERIES[host]
        first, last = host_bounds(host=host)
        for key in ("power", "ghi", "pv_shape"):
            values = panel_series(panel=panel, label=label, key=key)[first:last]
            finite = values[np.isfinite(values)]
            say(
                f"{label} {key} ({host}): {len(values)}, "
                f"{1 - len(finite) / len(values):.4f}, {finite.min():.3f}, "
                f"{np.median(finite):.3f}, {finite.max():.3f}"
            )
    n_grid = len(panel_series(panel=panel, label="D8", key="power"))
    farms = {
        letter: farm_on_grid(series=farm_norm_series(farm=letter), n_grid=n_grid)
        for letter in FARM_LETTERS
    }
    spans = {host: host_bounds(host=host) for host in HOST_KEYS}
    pooled = {
        letter: float(np.mean(np.concatenate([np.isfinite(norm[a:b]) for a, b in spans.values()])))
        for letter, norm in farms.items()
    }
    say("farm series: finite share of all half-hours, by span, and the longest gap (half-hours)")
    for letter, norm in farms.items():
        per_span = {h: farm_finite_share(norm=norm, first=a, last=b) for h, (a, b) in spans.items()}
        gap = {h: longest_gap(finite=np.isfinite(norm[a:b])) for h, (a, b) in spans.items()}
        say(f"farm {letter}: {per_span}, pooled {pooled[letter]:.4f}, longest gap {gap}")
    best = max(FARM_LETTERS, key=lambda letter: (pooled[letter], -ord(letter)))
    NUMBERS["farm"] = best
    NUMBERS["farm_pooled_finite"] = pooled
    first, last = spans["d8pre"]
    coverage = farm_finite_share(norm=farms[best], first=first, last=last)
    NUMBERS["farm_coverage_d8pre"] = coverage
    say(
        f"chosen farm: {best} (highest pooled finite share); "
        f"coverage of D8-pre's span {coverage:.4f}"
    )
    say(
        f"coverage at least {COVERAGE_MIN}: {coverage >= COVERAGE_MIN}"
        " (the verdict's branch 1 fires when it is under, as the design states)"
    )
    pv = panel_series(panel=panel, label="D8", key="pv_shape")
    day = (pv[first:last] > 0) & np.isfinite(farms[best][first:last])
    ratio = float(np.mean(farms[best][first:last][day]) / np.mean(pv[first:last][day]))
    NUMBERS["farm_to_cams_shape_mean_ratio_d8pre"] = ratio
    say(
        f"mean of the farm's output per MW over the mean CAMS shape, daylight half-hours at "
        f"D8-pre: {ratio:.3f}"
    )
    return farms


# --------------------------------------------------------------------------- step 2: the gate


def d8_reference(*, panel: dict[int, dict], frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return D8's solar-era engine-free curve and the fifth edges that every series uses.

    The rows are D8's solar-era half-hours that the tracker labels off and that lie outside
    experiment 159's drops, in May to September, 10:00 to 15:00 local, with a reading and not on a
    clock-change day. The edges are the quantiles of the CAMS shape over those rows.
    """
    pv = panel_series(panel=panel, label="D8", key="pv_shape")
    rows = frame.filter(
        pl.col("era") == "solar",
        ~pl.col("ts"),
        ~pl.col("in_drop"),
        pl.col("observed"),
        ~pl.col("clock_change_day"),
        pl.col("month").is_between(5, 9),
        pl.col("local_half_hour").is_between(MIDDAY_FIRST, MIDDAY_LAST),
    )
    shape = pv[rows["index"].to_numpy()]
    edges = fifth_edges(values=shape)
    medians, counts = curve_medians(
        reading=rows["reading"].to_numpy(),
        shape_value=shape,
        edges=edges,
        rows=np.ones(rows.height, dtype=bool),
    )
    return {"edges": edges, "medians": medians, "counts": counts}


def step_2_gate(
    *, panel: dict[int, dict], farms: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict:
    """Build the curve table and the gate, and return the gate document."""
    say()
    say("## 2. The curve gate (one set of fifth edges, D8's solar-era edges)")
    say(f"D8 solar-era edges of the CAMS shape: {fmt(values=reference['edges'], digits=3)}")
    say(f"D8 solar-era median reading by fifth (MVA): {fmt(values=reference['medians'])}")
    say(f"rows per fifth: {reference['counts'].tolist()}")
    n_grid = len(panel_series(panel=panel, label="D8", key="power"))
    calendar = grid_calendar_fields(n_grid=n_grid)
    midday = summer_midday_rows(calendar=calendar)
    farm = str(NUMBERS["farm"])
    gate: dict[str, dict] = {h: {} for h in HOST_KEYS}
    run: dict[str, dict] = {h: {} for h in HOST_KEYS}
    curves: dict[str, dict] = {h: {} for h in HOST_KEYS}
    say(
        "host, capacity, shape: median reading by fifth (MVA); largest gap to D8 in fifths 1 to 4; "
        "share under 1 MVA; share of reverse flow; passes"
    )
    for host in HOST_KEYS:
        series = HOST_SERIES[host]
        y = panel_series(panel=panel, label=series, key="power")
        pv_shape = panel_series(panel=panel, label=series, key="pv_shape")
        first, last = host_bounds(host=host)
        in_span = np.zeros(n_grid, dtype=bool)
        in_span[first:last] = True
        free, _ = curve_medians(
            reading=y, shape_value=pv_shape, edges=reference["edges"], rows=midday & in_span
        )
        say(f"{host}, no solar: {fmt(values=free)}")
        curves[host]["none"] = {"free": free.tolist()}
        for capacity in CAPACITIES_MW:
            key = capacity_key(capacity_mw=capacity)
            passes: dict[str, bool] = {}
            curves[host][key] = {}
            for shape in SHAPES:
                curve = solar_curve(
                    shape=shape,
                    capacity_mw=capacity,
                    pv_shape=pv_shape,
                    farm_norm=farms[farm],
                    calendar=calendar,
                    first=max(first - 48 * 28, 0),
                    last=last,
                )
                reading, signed = fold_reading(
                    y=y, total_mw=curve.pv_mw, first=max(first - 48 * 28, 0), last=last
                )
                rows = midday & in_span & np.isfinite(reading)
                if shape == "farm":
                    rows &= np.isfinite(farms[farm])
                medians, counts = curve_medians(
                    reading=reading, shape_value=pv_shape, edges=reference["edges"], rows=rows
                )
                ok = gate_passes(curve=medians, reference=reference["medians"], counts=counts)
                gap = float(np.nanmax(np.abs(medians[:4] - reference["medians"][:4])))
                under = float(np.mean(reading[rows] < 1.0))
                reverse = float(np.mean(signed[rows] < -FOLD_NEAR_ZERO_MW))
                passes[shape] = ok
                curves[host][key][shape] = {
                    "medians": medians.tolist(),
                    "counts": counts.tolist(),
                    "max_gap_first_four": gap,
                    "share_under_1_mva": under,
                    "share_reverse_flow": reverse,
                    "passes": ok,
                }
                say(
                    f"{host}, {capacity} MW, {shape}: {fmt(values=medians)}; gap {gap:.2f}; "
                    f"under 1 MVA {under:.2f}; reverse {reverse:.2f}; passes {ok}"
                )
            gate[host][key] = passes
            run[host][key] = {
                "farm": passes["farm"],
                "cams": passes["cams"],
                "smooth": passes["farm"] or passes["cams"],
            }
    document = {
        "farm": farm,
        "gate": gate,
        "run": run,
        "curves": curves,
        "reference": {k: v.tolist() for k, v in reference.items()},
    }
    d8_fail = all(not gate["d8pre"][capacity_key(capacity_mw=c)]["farm"] for c in CAPACITIES_MW)
    say(f"D8-pre fails the farm gate at both capacities (branch 1): {d8_fail}")
    return document


# --------------------------------------------------------------------------- steps 3 to 5


def night_and_gap(*, frame: pl.DataFrame, panel: dict[int, dict]) -> None:
    """Print D8-pre's night-matched constant and D8's monthly midday-minus-night gap by era."""
    say()
    say("## 3. Night level and the monthly midday-minus-night gap of D8")
    off = frame.filter(~pl.col("ts"), ~pl.col("in_drop"), pl.col("observed"))
    night = pl.col("local_half_hour") <= NIGHT_LAST
    solar_night = float(off.filter(pl.col("era") == "solar", night)["reading"].median())  # ty: ignore[invalid-argument-type]
    first, last = host_bounds(host="d8pre")
    y = panel_series(panel=panel, label="D8", key="power")[first:last]
    calendar = grid_calendar_fields(n_grid=len(panel_series(panel=panel, label="D8", key="power")))
    pre_night = float(np.nanmedian(y[(calendar.half_hour[first:last] <= NIGHT_LAST)]))
    constant = pre_night - solar_night
    NUMBERS["night_constant_d8pre_mva"] = constant
    say(
        f"night median, D8-pre {pre_night:.2f} MVA, solar era engine-off {solar_night:.2f} MVA, "
        f"night-matched constant {constant:.2f} MVA"
    )
    say("month, era: median midday reading minus median night reading (MVA), engine-off, no drop")
    gaps: dict[str, dict[str, float]] = {}
    for era in ERA_NAMES:
        part = off.filter(pl.col("era") == era)
        rows = []
        for month in range(1, 13):
            sub = part.filter(pl.col("month") == month)
            mid = sub.filter(pl.col("local_half_hour").is_between(MIDDAY_FIRST, MIDDAY_LAST))
            if mid.height < MIN_ROWS_PER_MONTH or sub.filter(night).height < MIN_ROWS_PER_MONTH:
                rows.append(f"{MONTH_NAMES[month - 1]} n/a")
                continue
            gap = float(mid["reading"].median() - sub.filter(night)["reading"].median())  # ty: ignore[unsupported-operator]
            gaps.setdefault(era, {})[MONTH_NAMES[month - 1]] = gap
            rows.append(f"{MONTH_NAMES[month - 1]} {gap:+.2f}")
        say(f"{era}: " + ", ".join(rows))
    NUMBERS["monthly_gap"] = gaps


def controls_check() -> None:
    """Reproduce the stored controls' midday F1 and the two false-flag definitions."""
    say()
    say("## 4. Stored controls of experiment 697: midday F1 and the false-flag share")
    for name, path, expected in (
        ("C1 (no constant)", E697_FRAME_C1_NULL, TWIN_F1_EXPECTED),
        ("C1s solar (level-matched)", E697_FRAME_C1S_SOLAR, 0.560),
    ):
        frame = pl.read_parquet(path).filter(
            pl.col("observed"),
            ~pl.col("clock_change_day"),
            pl.col("local_half_hour").is_between(MIDDAY_FIRST, MIDDAY_LAST),
        )
        truth = frame["truth1"].to_numpy()
        flag = frame["ts"].to_numpy()
        tp = int((truth & flag).sum())
        fp = int((~truth & flag).sum())
        fn = int((truth & ~flag).sum())
        f1 = 2 * tp / (2 * tp + fp + fn)
        check(
            name=f"{name} midday F1 is {expected}", ok=abs(f1 - expected) < 0.01, detail=f"{f1:.3f}"
        )
        zero = (frame["truth_mw"] == 0).to_numpy()
        ff_zero = float(flag[zero].mean())
        ff_under = float(flag[~truth].mean())
        say(
            f"  {name}: false-flag share with truth_mw = 0 is {ff_zero:.3f}, "
            f"with under 1 unit {ff_under:.3f}; 698 uses truth_mw = 0 throughout"
        )
        NUMBERS[f"f1_{name.split()[0].lower()}_{'solar' if 'solar' in name else 'plain'}"] = f1
        NUMBERS[f"false_flag_zero_{'solar' if 'solar' in name else 'plain'}"] = ff_zero
        if "solar" in name:
            check(
                name="false-flag share (truth_mw = 0) is about 0.020",
                ok=abs(ff_zero - 0.020) < 0.004,
            )


def spread_by_era(*, frame: pl.DataFrame) -> None:
    """Print the spread of half-hourly changes on engine-off, no-drop midday half-hours by era."""
    say()
    say("## 5. Spread of half-hourly changes of D8's reading (engine-off, no drop, midday)")
    ordered = frame.sort("index").with_columns(
        change=pl.col("reading").diff(),
        same=(pl.col("index").diff() == 1) & (pl.col("era") == pl.col("era").shift(1)),
    )
    rows = ordered.filter(
        pl.col("same"),
        ~pl.col("ts"),
        ~pl.col("ts").shift(1),
        ~pl.col("in_drop"),
        ~pl.col("in_drop").shift(1),
        pl.col("local_half_hour").is_between(MIDDAY_FIRST, MIDDAY_LAST),
        pl.col("change").is_not_null(),
    )
    spreads = {}
    for era in ("net", "solar"):
        values = rows.filter(pl.col("era") == era)["change"].abs().to_numpy()
        spreads[era] = float(1.4826 * np.median(values))
        say(f"  {era}: {len(values)} changes, robust spread {spreads[era]:.3f} MVA")
    NUMBERS["change_spread_d8"] = spreads
    say(f"  solar minus net: {spreads['solar'] - spreads['net']:+.3f} MVA")


# --------------------------------------------------------------------------- step 6: timestamps


def lag_correlations(
    *, x: np.ndarray, pv: np.ndarray, valid: np.ndarray, hourly: bool
) -> np.ndarray:
    """Return the correlation of changes of `x` at lag k with changes of `pv`, for k in `LAGS`.

    Half-hourly changes are first differences. Hourly sums are `x[t] + x[t - 1]` with a change over
    one hour (two half-hours), so a lag is still in half-hours.

    Args:
        x: The series whose changes are shifted (the reading or the farm series).
        pv: The CAMS shape.
        valid: Where both series may be used.
        hourly: Whether to use hourly sums.
    """
    xs = np.where(valid, x, np.nan)
    ps = np.where(valid, pv, np.nan)
    if hourly:
        xs = xs + np.roll(xs, 1)
        ps = ps + np.roll(ps, 1)
        step = 2
    else:
        step = 1
    dx = np.full(len(xs), np.nan)
    dp = np.full(len(xs), np.nan)
    dx[step:] = xs[step:] - xs[:-step]
    dp[step:] = ps[step:] - ps[:-step]
    out = []
    for lag in LAGS:
        shifted = np.roll(dx, -lag)
        keep = np.isfinite(shifted) & np.isfinite(dp)
        out.append(float(np.corrcoef(shifted[keep], dp[keep])[0, 1]) if keep.sum() > 50 else np.nan)
    return np.asarray(out)


def step_6_timestamps(
    *, panel: dict[int, dict], frame: pl.DataFrame, farms: dict[str, np.ndarray]
) -> None:
    """Assert that the strongest correlation of changes with the CAMS shape is at lag 0."""
    say()
    say("## 6. Timestamps: correlation of changes with the CAMS shape at lags -4 to +4 half-hours")
    say(f"lags: {list(LAGS)}")
    n_grid = len(panel_series(panel=panel, label="D8", key="power"))
    pv = panel_series(panel=panel, label="D8", key="pv_shape")
    off = frame.filter(~pl.col("ts"), ~pl.col("in_drop"), pl.col("observed"))
    mask = np.zeros(n_grid, dtype=bool)
    mask[off["index"].to_numpy()] = True
    reading = np.full(n_grid, np.nan)
    reading[frame["index"].to_numpy()] = frame["reading"].to_numpy()
    first, last = host_bounds(host="d8pre")
    in_pre = np.zeros(n_grid, dtype=bool)
    in_pre[first:last] = True
    farm = farms[str(NUMBERS["farm"])]
    cases = {
        "D8 reading, engine-off and no-drop": (reading, mask & (pv > 0)),
        "farm, D8-pre span": (farm, in_pre & np.isfinite(farm) & (pv > 0)),
    }
    for name, (x, valid) in cases.items():
        for hourly in (False, True):
            corr = lag_correlations(x=x, pv=pv, valid=valid, hourly=hourly)
            kind = "hourly sums" if hourly else "half-hourly changes"
            best = LAGS[int(np.nanargmax(np.abs(corr)))]
            say(f"  {name}, {kind}: {fmt(values=corr, digits=3)}")
            check(
                name=f"strongest |correlation| at lag 0: {name}, {kind}",
                ok=best == 0,
                detail=f"best lag {best}",
            )


# --------------------------------------------------------------------------- step 8: step 4


def step_8_step4(*, frame: pl.DataFrame, panel: dict[int, dict]) -> None:
    """Print step 4's off-spells and the months and drop share of the selected days."""
    say()
    say("## 8. Step 4: whole local days inside off-spells of 48 half-hours or more")
    n_grid = len(panel_series(panel=panel, label="D8", key="power"))
    calendar = grid_calendar_fields(n_grid=n_grid)
    for era in ("net", "solar"):
        first, last = era_bounds()[era]
        part = frame.filter(pl.col("era") == era).sort("index")
        flag = part["ts"].to_numpy()
        observed = part["observed"].to_numpy()
        lengths = off_spell_lengths(flag=flag, min_length=STEP4_MIN_SPELL)
        rows = off_spell_rows(flag=flag, min_length=STEP4_MIN_SPELL)
        days = whole_day_rows(
            rows=rows, local_date=calendar.local_date[first:last], observed=observed
        )
        n_days = len(np.unique(calendar.local_date[first:last][days]))
        months = np.bincount(calendar.month[first:last][days], minlength=13)[1:]
        drop = float(part["in_drop"].to_numpy()[days].mean()) if days.any() else float("nan")
        design = DESIGN_SPELLS[era]
        say(
            f"  {era}: {len(lengths)} off-spells with {sum(lengths)} half-hours "
            f"(the design counted {design[0]} and {design[1]}); {n_days} whole local days, "
            f"{int(days.sum())} half-hours; share inside a drop {drop:.3f}"
        )
        say(
            "    selected half-hours by calendar month: "
            + ", ".join(f"{MONTH_NAMES[m]} {int(c)}" for m, c in enumerate(months) if c)
        )
        say(
            f"    if partial days were allowed (`STEP4_WHOLE_DAYS_ONLY` False): "
            f"{int((rows & observed).sum())} half-hours"
        )
        NUMBERS[f"step4_{era}"] = {
            "spells": len(lengths),
            "spell_half_hours": sum(lengths),
            "days": n_days,
            "half_hours": int(days.sum()),
            "half_hours_if_partial_days": int((rows & observed).sum()),
            "drop_share": drop,
        }


# --------------------------------------------------------------------------- step 9: ERA5


def step_9_era5(*, panel: dict[int, dict]) -> None:
    """Print one line comparing CAMS irradiance at D8 with ERA5 `ghi_w_m2` at the nearest cell.

    The site's coordinates are used to pick the cell and are never printed. A failure to read the
    stored files is printed as such.
    """
    say()
    say("## 9. CAMS irradiance against ERA5 at the nearest stored cell")
    try:
        from studies.dp_features import default_input_paths
        from studies.pv_dataset import nearest_era5_cell, read_era5

        table = pl.read_parquet(default_input_paths().series_table).filter(pl.col("site") == "D8")
        era5 = read_era5(source="open-meteo")
        cell = nearest_era5_cell(sites=table, era5=era5)
        hourly = (
            era5.join(
                cell.select(latitude="cell_latitude", longitude="cell_longitude"),
                on=["latitude", "longitude"],
            )
            .select("time", era5=pl.col("ghi_w_m2"))
            .sort("time")
        )
        ghi = panel_series(panel=panel, label="D8", key="ghi")
        index = np.arange(len(ghi))
        end = half_hour_starts(index=index) + timedelta(minutes=30)
        cams = (
            pl.DataFrame({"time": end, "cams": ghi})
            .filter(pl.col("time").dt.minute() == 0)
            .with_columns(pl.col("time").dt.cast_time_unit(hourly["time"].dtype.time_unit))  # ty: ignore[unresolved-attribute]
        )
        both = cams.join(hourly, on="time").drop_nulls()
        corr = float(np.corrcoef(both["cams"].to_numpy(), both["era5"].to_numpy())[0, 1])
        bias = float((both["cams"] - both["era5"]).mean())  # ty: ignore[invalid-argument-type]
        say(f"  {both.height} hours; correlation {corr:.3f}; mean CAMS minus ERA5 {bias:+.1f} W/m2")
        NUMBERS["cams_vs_era5"] = {"hours": both.height, "correlation": corr, "bias": bias}
    except (FileNotFoundError, ImportError, KeyError, pl.exceptions.PolarsError) as error:
        say(f"  not computed: {type(error).__name__}: {str(error)[:120]}")


# --------------------------------------------------------------------------- Gas A (private)


def gas_a_private() -> None:
    """Store a few counts of Gas A's complete days in `audit.json` only, never in the printout."""
    days = record_days()
    running = days.output >= UNIT_THRESHOLD_MW
    NUMBERS["private_gas_a"] = {
        "complete_days": len(days.output),
        "running_days": int(running.any(axis=1).sum()),
        "midday_running_days": int(running[:, MIDDAY_FIRST : MIDDAY_LAST + 1].any(axis=1).sum()),
        "midday_running_half_hours": int(running[:, MIDDAY_FIRST : MIDDAY_LAST + 1].sum()),
    }


def main() -> None:
    """Run the audit and write the printed report, `audit.json`, and `gate.json`."""
    panel = load_panel()
    frame = pl.read_parquet(E697_FRAME_D8)
    farms = step_1_data(panel=panel)
    reference = d8_reference(panel=panel, frame=frame)
    gate = step_2_gate(panel=panel, farms=farms, reference=reference)
    night_and_gap(frame=frame, panel=panel)
    controls_check()
    spread_by_era(frame=frame)
    step_6_timestamps(panel=panel, frame=frame, farms=farms)
    say()
    say("## 7. Label-only checks of D8 (with and without experiment 159's drops)")
    NUMBERS["label_only"] = label_only_checks(frame=frame, say=say)
    step_8_step4(frame=frame, panel=panel)
    step_9_era5(panel=panel)
    gas_a_private()
    say()
    say(f"failed checks: {FAILED or 'none'}")
    PRINTED_DIR.mkdir(parents=True, exist_ok=True)
    (PRINTED_DIR / "audit.txt").write_text("\n".join(LINES) + "\n")
    write_json(path=GATE_PATH, payload=gate)
    write_json(path=AUDIT_PATH, payload={"numbers": NUMBERS, "failed": FAILED})
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()

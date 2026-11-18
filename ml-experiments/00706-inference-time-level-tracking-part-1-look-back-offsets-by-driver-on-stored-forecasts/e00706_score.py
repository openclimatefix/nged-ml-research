# ruff: noqa: E501
"""Experiment 706, stage 1: scores and contrasts C1, C2, and C5 of the look-back offset arms (CPU
only).

Run it from the repository root with the environment of `CLAUDE.md` set and the folders of
experiments 668, 683, and 688 on `PYTHONPATH`:

    PP=$(ls -d ml-experiments/00668-* ml-experiments/00683-* ml-experiments/00688-* | paste -sd:)
    PYTHONPATH=$PP uv run --no-sync python ml-experiments/00706-*/e00706_score.py --n-jobs 4

Reads the prediction sets that `e00706_arms.py` wrote, the stored normal-running and prevailing
rows, and the cleaned power. Scores are in points of capacity on rows with a lead above 0 and up to
240 hours. Event-free rows are rows whose valid time lies outside every labelled or verified event
of the same Primary (`event_spans(kind="primary")` of experiment 688).

Contrasts, each paired on (series, origin, valid time) and read at the Bonferroni level for six
contrasts (a 99.17% interval, 20,000 month-block resamples) with the t test on the monthly paired
differences:

- C1: G-lb minus the stored normal-running rows (S-NRA), event-free rows, origins 2025-12 to
  2026-06.
- C2: D minus G-lb, the same rows.
- C5: P-cp minus the stored prevailing rows (S-PRV), all rows, origins 2026-01 to 2026-06.
- G-lb minus recipe R (RR), with a 95% interval.

C1, C2, and C5 are read in the signed-MW six and the MVA-only eight separately and for the 14
pooled;
D15 and D18 are report-only. The script also prints, for every arm, the CRPS and the median error
by group, by lead band, by half of the month, on cold winter nights, on sunny April to June
half-hours, in the four load-shift scenarios, the coverage and pinball loss of q01, q05, q95, and
q99, and the rows where the prevailing arm departs from the stored normal-running median. Tables are
also written as CSV under the experiment's data folder.

References computed here from the stored rows (nothing is fitted): `R0-168` (the mean of the
day-ahead residual over the valid days 8 to 2 before the origin, decayed by exp(-lead / 168 h), as
`e00647_r0.py` does), `RR` (recipe R: 0.75 times the 14-day mean error per half-hour of day, every
lead), and `M28` (the 28-day median residual as one shift of every quantile). The climatology and
the
manual heuristic are read from the per-row scores of experiment 668.
"""

import argparse
import datetime as dt
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00706_arms as arms  # ty: ignore[unresolved-import]
import e00706_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy import stats
from studies.bootstrap import bootstrap_difference_at_level
from studies.eval_protocol.metrics import MEDIAN_INDEX
from studies.experiment_store import losses_for_bootstrap, read_predictions

BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 6.0)
"""99.17%: a 95% family level shared by six contrasts."""
N_RESAMPLES: Final[int] = 20_000
REPORTED_LEVEL: Final[float] = 95.0
REPORTED_RESAMPLES: Final[int] = 2000
SMALLEST_WORTHWHILE_GAIN: Final[float] = 0.05
VETO_POINTS: Final[float] = 0.02
DEPARTURE_POINTS: Final[float] = 3.0
"""A prevailing median this far from the stored normal-running median is a departure."""
GROUP_ORDER: Final[tuple[str, ...]] = (c.GROUP_SIGNED, c.GROUP_MVA, c.POOLED, c.GROUP_SHORT)
TAILS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
SET_LABELS: Final[dict[str, str]] = {
    "g": "G-lb",
    "g-sl": "G smallest-lead",
    "g-refit": "G-refit",
    "g-w7": "G-lb 7 d",
    "g-w28": "G-lb 28 d",
    "d": "D",
    "d-pv": "D-pv",
    "d-cold": "D-cold",
    "d-plus": "D+",
    "d-w7": "D 7 d",
    "d-w28": "D 28 d",
    "p": "P-cp",
    "p-7d": "P-7d",
    "p-cp06": "P-cp 06:00",
    "p-7d06": "P-7d 06:00",
}
"""The label of each prediction set, as the tables print it."""
DEC_ARMS: Final[tuple[str, ...]] = (
    "S-NRA",
    "G-lb",
    "G smallest-lead",
    "G-refit",
    "G-lb 7 d",
    "D",
    "D-pv",
    "D-cold",
    "D+",
    "D 7 d",
    "R0-168",
    "RR",
    "M28",
)
JAN_ARMS: Final[tuple[str, ...]] = (*DEC_ARMS, "G-lb 28 d", "D 28 d")
PREVAILING_ARM_LABELS: Final[tuple[str, ...]] = (
    "S-NRA",
    "S-PRV",
    "P-cp",
    "P-7d",
    "P-cp 06:00",
    "P-7d 06:00",
)

POINT_ESTIMATE_NOTE: Final[str] = (
    "POINT ESTIMATE: means over the selected rows with no bootstrap interval; read them as "
    "descriptions, not as tests."
)
type Losses = dict[str, dict[str, np.ndarray]]
type Events = dict[str, tuple[np.ndarray, np.ndarray]]


def mean_over(*, values: np.ndarray, rows: np.ndarray) -> float:
    """Return the mean of `values` over `rows`, NaN if no row is selected."""
    selected = values[rows]
    return float(selected.mean()) if selected.size else float("nan")


def metric_mean(*, losses: Losses, label: str, key: str, rows: np.ndarray) -> float:
    """Return the mean of one arm's loss `key` over `rows`, NaN if the arm has no such loss."""
    values = losses[label].get(key)
    return float("nan") if values is None else mean_over(values=values, rows=rows)


# --------------------------------------------------------------------------- reference arms


def r0_168_correction(*, residuals: c.Residuals) -> np.ndarray:
    """Return R0-168's correction, (K, N_ORIGINS, N_H): the 7-day day-ahead residual, decayed.

    The residual of a valid UTC day is the mean over its half-hours of the lead-band-2 residual
    (a forecast with a lead of 24 to 48 hours). The origin's value is the mean of the daily means of
    the days 8 to 2 before the origin day, so the newest reading is about 33 hours old.
    """
    day_ahead = residuals.banded[:, 1, :].reshape(-1, c.N_ORIGINS, c.PER_DAY)
    seen = np.isfinite(day_ahead).sum(axis=2)
    daily = np.where(seen > 0, np.nansum(day_ahead, axis=2) / np.maximum(seen, 1), np.nan)
    level = np.zeros((daily.shape[0], c.N_ORIGINS))
    for origin in range(c.N_ORIGINS):
        window = daily[:, max(origin - 8, 0) : max(origin - 1, 0)]
        count = np.isfinite(window).sum(axis=1)
        total = np.where(np.isfinite(window), window, 0.0).sum(axis=1)
        level[:, origin] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    decay = np.exp(-(np.arange(c.N_H) / 2.0) / 168.0)
    return level[:, :, None] * decay[None, None, :]


def recipe_r_correction(*, residuals: c.Residuals) -> np.ndarray:
    """Return recipe R's correction: 0.75 times the 14-day mean error per half-hour of day, every
    lead."""
    n_series = residuals.smallest.shape[0]
    slots = np.zeros((n_series, c.N_ORIGINS, c.PER_DAY))
    for origin in range(c.N_ORIGINS):
        last = c.ORIGIN_STAMP + c.PER_DAY * origin - c.LAG_6H
        first = last - 14 * c.PER_DAY + 1
        if first < 0:
            continue
        block = residuals.smallest[:, first : last + 1].reshape(n_series, 14, c.PER_DAY)
        seen = np.isfinite(block).sum(axis=1)
        mean = np.where(seen >= 7, np.nansum(block, axis=1) / np.maximum(seen, 1), 0.0)
        slots[:, origin, (first + np.arange(c.PER_DAY)) % c.PER_DAY] = mean
    row_slot = (c.ORIGIN_STAMP + np.arange(c.N_H)) % c.PER_DAY
    return c.FACTOR_LOOK_BACK * slots[:, :, row_slot]


def m28_correction(*, residuals: c.Residuals) -> np.ndarray:
    """Return M28's correction: the median residual of the last 28 days, shifting every quantile."""
    n_series = residuals.smallest.shape[0]
    level = np.zeros((n_series, c.N_ORIGINS))
    length = 28 * c.PER_DAY
    for origin in range(c.N_ORIGINS):
        last = c.ORIGIN_STAMP + c.PER_DAY * origin - c.LAG_6H
        first = max(last - length + 1, 0)
        block = residuals.smallest[:, first : last + 1]
        enough = np.isfinite(block).sum(axis=1) >= 0.5 * length
        for series in np.flatnonzero(enough):
            level[series, origin] = float(np.nanmedian(block[series]))
    return np.broadcast_to(level[:, :, None], (n_series, c.N_ORIGINS, c.N_H)).copy()


def stored_668_losses(*, inputs: c.Inputs) -> Losses:
    """Return the climatology and the manual heuristic, from experiment 668's per-row scores.

    Experiment 668 scored the stored normal-running rows, the manual heuristic, and the climatology
    on the half-hours where all three exist; those rows are a subset of ours. The cleaned-power
    table it read is the same table.
    """
    path = c.BACKTEST_DIR / "scores_668" / "primaries_NRA" / "rows.parquet"
    columns = [
        "time_series_id",
        "valid_idx",
        "origin_idx",
        "ok",
        "Model_crps",
        "Model_mae",
        "Manual heuristic_crps",
        "Manual heuristic_mae",
        "Climatology_crps",
        "Climatology_mae",
    ]
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("ok"), pl.col("valid_idx") < c.LIB_OFFSET + c.N_STAMPS)
        .select(columns)
        .collect()
    )
    position = {int(series): index for index, series in enumerate(inputs.ids)}
    k = (
        frame["time_series_id"]
        .cast(pl.Int64)
        .replace_strict(position, return_dtype=pl.Int64)
        .to_numpy()
    )
    origin_idx = frame["origin_idx"].to_numpy().astype(np.int64)
    origin, remainder = np.divmod(origin_idx - c.LIB_OFFSET - c.ORIGIN_STAMP, c.PER_DAY)
    row = frame["valid_idx"].to_numpy().astype(np.int64) - origin_idx
    keep = (remainder == 0) & (origin >= 0) & (origin < c.N_ORIGINS) & (row >= 0) & (row < c.N_H)
    out: Losses = {}
    for label, prefix in (
        ("Climatology", "Climatology"),
        ("Manual heuristic", "Manual heuristic"),
        ("668 stored rows", "Model"),
    ):
        out[label] = {}
        for metric in ("crps", "mae"):
            dense = np.full((len(inputs.ids), c.N_ORIGINS, c.N_H), np.nan, np.float32)
            dense[k[keep], origin[keep], row[keep]] = frame[f"{prefix}_{metric}"].to_numpy()[keep]
            out[label][metric] = dense
    return out


# --------------------------------------------------------------------------- loading arms


def load_all(
    *, inputs: c.Inputs, residuals: c.Residuals, month: str | None, set_suffix: str | None
) -> tuple[Losses, dict[str, np.ndarray]]:
    """Return the per-row losses of every arm, and the quantiles of S-NRA, S-PRV, and P-cp.

    Missing prediction sets are reported and skipped.

    Args:
        inputs: The loaded inputs.
        residuals: The look-back residuals, for the reference arms.
        month: The month of `arms --month`, whose row-set ids end in `_<month>`, or None.
        set_suffix: The suffix of the set names that `arms --month` wrote, such as `-t202603`; it
            defaults to `-t<YYYYMM>` of `month`.

    Returns:
        The per-row losses, and the quantiles of S-NRA, S-PRV, and P-cp by name.
    """
    truth = c.target_truth(inputs=inputs)
    losses: Losses = {}
    kept: dict[str, np.ndarray] = {}

    def add(*, label: str, quantiles: np.ndarray) -> None:
        losses[label] = c.loss_arrays(truth=truth, quantiles=quantiles)
        if label in ("S-NRA", "S-PRV", "P-cp"):
            kept[label] = quantiles

    add(label="S-NRA", quantiles=inputs.q_nra)
    assert inputs.q_prv is not None
    add(label="S-PRV", quantiles=inputs.q_prv)
    for label, correction in (
        ("R0-168", r0_168_correction(residuals=residuals)),
        ("RR", recipe_r_correction(residuals=residuals)),
        ("M28", m28_correction(residuals=residuals)),
    ):
        add(
            label=label,
            quantiles=c.apply_correction(
                inputs=inputs, quantiles=inputs.q_nra, correction=correction.astype(np.float32)
            ),
        )
    sets = {spec.name: spec.row_set_id for spec in (*arms.LOOK_BACK_ARMS, *arms.PREVAILING_ARMS)}
    sets |= {name: c.ROW_SET_ID_FROM_JANUARY for name, _ in arms.CHANGEPOINT_ARMS}
    if month is None and set_suffix is not None:
        raise ValueError("--set-suffix needs --month")
    if month is not None and set_suffix is None:
        set_suffix = f"-t{month.replace('-', '')}"
    for name, row_set_id in sets.items():
        stored_name = name if month is None else f"{name}{set_suffix or ''}"
        stored_row_set = row_set_id if month is None else f"{row_set_id}_{month}"
        try:
            frame = read_predictions(
                experiment_id=c.EXPERIMENT_ID, name=stored_name, row_set_id=stored_row_set
            )
        except FileNotFoundError:
            print(f"MISSING prediction set {stored_name}; its rows are skipped")
            continue
        add(
            label=SET_LABELS[name],
            quantiles=c.dense_from_frame(frame=frame, ids=inputs.ids, capacity=inputs.capacity),
        )
        print(f"LOADED {stored_name} as {SET_LABELS[name]}: {frame.height:,} rows")
    return losses, kept


# --------------------------------------------------------------------------- masks


def load_events(*, inputs: c.Inputs) -> Events:
    """Return the in-event and 7-days-after masks (K, N_STAMPS) of three event sets.

    `primary` is the labelled events and the verified transfers of experiment 688's step 1,
    `verified`
    is the verified transfers alone, and `wide` adds the fitted spans of experiment 580 and the
    candidate events of experiment 685.
    """
    inside, after, spans = c.load_event_masks(inputs=inputs, kind="primary")
    only = spans.filter(pl.col("source").str.starts_with("verified transfer"))
    wide_inside, wide_after, _ = c.load_event_masks(inputs=inputs, kind="wide")
    return {
        "primary": (inside, after),
        "verified": c.event_masks(spans=only, ids=inputs.ids),
        "wide": (wide_inside, wide_after),
    }


def build_masks(
    *, inputs: c.Inputs, events: Events, first_origin: int, month: str | None
) -> dict[str, np.ndarray]:
    """Return the row masks of the tables, each (K, N_ORIGINS, N_H).

    `scored` is power present, lead up to 240 h, and origin from `first_origin`. For each event set
    the keys `<set>: all`, `<set>: event-free`, `<set>: outside and 7 d after`, `<set>: inside`, and
    `<set>: 7 d after` are the four load-shift scenarios and the event-free rows.
    """
    truth = c.target_truth(inputs=inputs)
    origins = np.arange(c.N_ORIGINS) >= first_origin
    if month is not None:
        origins &= np.isin(np.arange(c.N_ORIGINS), c.origin_of_month(month=month))
    scored = (
        np.isfinite(truth)
        & origins[None, :, None]
        & c.lead_mask()[None, None, :]
        & np.isfinite(inputs.q_nra[..., MEDIAN_INDEX])
    )
    masks: dict[str, np.ndarray] = {"scored": scored}
    for kind, (inside, after) in events.items():
        inside_t = c.stamp_to_target(per_stamp=inside)
        after_t = c.stamp_to_target(per_stamp=after)
        masks[f"{kind}: all"] = scored
        masks[f"{kind}: event-free"] = scored & ~inside_t
        masks[f"{kind}: outside and 7 d after"] = scored & ~inside_t & ~after_t
        masks[f"{kind}: inside"] = scored & inside_t
        masks[f"{kind}: 7 d after"] = scored & after_t
    overlap = c.window_overlap(
        flags=events["primary"][0], window_half_hours=14 * c.PER_DAY, lag_half_hours=c.LAG_6H
    )
    masks["primary: clean window"] = masks["primary: event-free"] & ~overlap[:, :, None]
    return masks


def slice_masks(*, inputs: c.Inputs, scored: np.ndarray) -> dict[str, np.ndarray]:
    """Return the cold-winter-night, sunny-April-to-June, and half-of-month masks.

    Realised ERA5 weather defines them, so a slice is the same set of half-hours for every arm.
    Night is a shortwave under 1 W/m2, a cold night has 2 m temperature of 3 degrees Celsius or
    below, and a sunny half-hour is in the top third of the series' own April to June daylight
    shortwave (above 20 W/m2), an ERA5 stand-in for the protocol's unit-shape rule.
    """
    era5 = c.era5_grid(ids=inputs.ids)
    stamps = pl.datetime_range(
        c.GRID_START,
        c.GRID_START + dt.timedelta(minutes=30 * (c.N_STAMPS - 1)),
        interval="30m",
        time_zone="UTC",
        eager=True,
    ).dt.convert_time_zone("Europe/London")
    month = stamps.dt.month().to_numpy()
    winter = np.isin(month, (12, 1, 2))[None, :]
    spring = np.isin(month, (4, 5, 6))[None, :]
    night = era5[..., 0] < 1.0
    cold_night = winter & night & (era5[..., 1] <= 3.0)
    daylight = (era5[..., 0] > 20.0) & spring
    sunny = np.zeros_like(daylight)
    for series in range(len(inputs.ids)):
        lit = daylight[series]
        if lit.any():
            sunny[series] = lit & (era5[series, :, 0] >= np.quantile(era5[series, lit, 0], 2 / 3))
    day_of_month = np.array([c.origin_to_utc(origin=o).day for o in range(c.N_ORIGINS)])
    return {
        "cold winter nights": scored & c.stamp_to_target(per_stamp=cold_night),
        "sunny April to June": scored & c.stamp_to_target(per_stamp=sunny),
        "origin day 1 to 14": scored & (day_of_month <= 14)[None, :, None],
        "origin day 15 on": scored & (day_of_month >= 15)[None, :, None],
    }


# --------------------------------------------------------------------------- tables


def show(*, name: str, frame: pl.DataFrame, note: str | None = None) -> None:
    """Print a table and write it as CSV (through a temporary file) under the data folder."""
    print(f"\nTABLE {name}")
    if note is not None:
        print(note)
    with pl.Config(tbl_rows=200, tbl_cols=40, fmt_float="mixed", tbl_width_chars=250):
        print(frame)
    path = c.OUT_DIR / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    frame.write_csv(temporary)
    temporary.replace(path)


def common_rows(*, losses: Losses, labels: tuple[str, ...], base: np.ndarray) -> np.ndarray:
    """Return `base` restricted to rows where every named arm has a CRPS."""
    rows = base.copy()
    for label in labels:
        if label in losses:
            rows &= np.isfinite(losses[label]["crps"])
    return rows


def group_table(
    *, inputs: c.Inputs, losses: Losses, labels: tuple[str, ...], rows: np.ndarray, name: str
) -> None:
    """Print CRPS and median error per group, and the tail pinball and coverage for the 14
    pooled."""
    masks = c.group_masks(inputs=inputs)
    table = []
    for label in labels:
        if label not in losses:
            continue
        entry: dict[str, float | str | int] = {"arm": label, "rows": int(rows.sum())}
        for group in GROUP_ORDER:
            chosen = rows & masks[group][:, None, None]
            entry[f"crps {group}"] = mean_over(values=losses[label]["crps"], rows=chosen)
            entry[f"mae {group}"] = mean_over(values=losses[label]["mae"], rows=chosen)
        for tail in TAILS:
            chosen = rows & masks[c.POOLED][:, None, None]
            entry[f"pinball {tail}"] = metric_mean(
                losses=losses, label=label, key=f"pin_{tail}", rows=chosen
            )
            entry[f"coverage {tail}"] = metric_mean(
                losses=losses, label=label, key=f"cov_{tail}", rows=chosen
            )
        table.append(entry)
    show(name=name, frame=pl.DataFrame(table))


def band_table(
    *, inputs: c.Inputs, losses: Losses, labels: tuple[str, ...], rows: np.ndarray, name: str
) -> None:
    """Print CRPS by lead band for the 14 pooled, and the gain over S-NRA by group and band."""
    masks = c.group_masks(inputs=inputs)
    table = []
    for label in labels:
        if label not in losses:
            continue
        for group in (c.POOLED, c.GROUP_SIGNED, c.GROUP_MVA):
            entry: dict[str, float | str] = {"arm": label, "group": group}
            for band, band_name in enumerate(c.BAND_NAMES):
                in_band = np.equal(c.BAND_OF_H, band)[None, None, :]
                chosen = rows & masks[group][:, None, None] & in_band
                value = mean_over(values=losses[label]["crps"], rows=chosen)
                entry[band_name] = value
                if label != "S-NRA" and "S-NRA" in losses:
                    entry[f"{band_name} minus S-NRA"] = value - mean_over(
                        values=losses["S-NRA"]["crps"], rows=chosen
                    )
            table.append(entry)
    show(name=name, frame=pl.DataFrame(table), note=POINT_ESTIMATE_NOTE)


def slice_table(
    *,
    inputs: c.Inputs,
    losses: Losses,
    labels: tuple[str, ...],
    slices: dict[str, np.ndarray],
    base: np.ndarray,
    name: str,
) -> None:
    """Print CRPS and median error of each arm in each slice, for the 14 pooled and the signed
    six."""
    masks = c.group_masks(inputs=inputs)
    table = []
    for label in labels:
        if label not in losses:
            continue
        for slice_name, slice_rows in slices.items():
            for group in (c.POOLED, c.GROUP_SIGNED, c.GROUP_MVA):
                chosen = slice_rows & base & masks[group][:, None, None]
                table.append(
                    {
                        "arm": label,
                        "slice": slice_name,
                        "group": group,
                        "rows": int(chosen.sum()),
                        "crps": mean_over(values=losses[label]["crps"], rows=chosen),
                        "mae": mean_over(values=losses[label]["mae"], rows=chosen),
                    }
                )
    show(name=name, frame=pl.DataFrame(table), note=POINT_ESTIMATE_NOTE)


def scenario_table(
    *,
    inputs: c.Inputs,
    losses: Losses,
    labels: tuple[str, ...],
    masks: dict[str, np.ndarray],
    name: str,
) -> None:
    """Print CRPS of each arm in the four load-shift scenarios, for each event set."""
    series_masks = c.group_masks(inputs=inputs)
    table = []
    for label in labels:
        if label not in losses:
            continue
        for kind in ("primary", "verified"):
            for scenario in ("all", "outside and 7 d after", "inside", "7 d after"):
                rows = masks[f"{kind}: {scenario}"] & np.isfinite(losses[label]["crps"])
                if "S-NRA" in losses:
                    rows &= np.isfinite(losses["S-NRA"]["crps"])
                chosen = rows & series_masks[c.POOLED][:, None, None]
                table.append(
                    {
                        "arm": label,
                        "event set": "labelled and verified"
                        if kind == "primary"
                        else "verified only",
                        "scenario": scenario,
                        "rows": int(chosen.sum()),
                        "crps": mean_over(values=losses[label]["crps"], rows=chosen),
                    }
                )
    show(name=name, frame=pl.DataFrame(table))


def departure_table(
    *, inputs: c.Inputs, kept: dict[str, np.ndarray], base: np.ndarray, name: str
) -> None:
    """Print how often prevailing arms depart from S-NRA's median by 3 points, and equal-width
    scores."""
    if "S-PRV" not in kept or "P-cp" not in kept:
        return
    median = kept["S-NRA"][..., MEDIAN_INDEX]
    truth = c.target_truth(inputs=inputs)
    table = []
    for label, quantiles in (("S-PRV", kept["S-PRV"]), ("P-cp", kept["P-cp"])):
        departs = np.abs(quantiles[..., MEDIAN_INDEX] - median) >= DEPARTURE_POINTS
        for group, mask in c.group_masks(inputs=inputs).items():
            chosen = base & mask[:, None, None] & np.isfinite(quantiles[..., MEDIAN_INDEX])
            table.append(
                {
                    "arm": label,
                    "group": group,
                    "rows": int(chosen.sum()),
                    "share departing by 3 points or more": float(departs[chosen].mean())
                    if chosen.any()
                    else float("nan"),
                }
            )
    show(name=f"{name}_departures", frame=pl.DataFrame(table))
    reference_width = kept["S-PRV"][..., 9] - kept["S-PRV"][..., 1]
    arm_width = kept["P-cp"][..., 9] - kept["P-cp"][..., 1]
    rows = base & np.isfinite(reference_width) & np.isfinite(arm_width)
    scaled_table = []
    crps_p = c.loss_arrays(truth=truth, quantiles=kept["P-cp"])["crps"]
    crps_s = c.loss_arrays(truth=truth, quantiles=kept["S-PRV"])["crps"]
    for group, mask in c.group_masks(inputs=inputs).items():
        chosen = rows & mask[:, None, None]
        factor = mean_over(values=reference_width, rows=chosen) / mean_over(
            values=arm_width, rows=chosen
        )
        median_p = kept["P-cp"][..., MEDIAN_INDEX : MEDIAN_INDEX + 1]
        scaled = median_p + factor * (kept["P-cp"] - median_p)
        crps_scaled = c.loss_arrays(truth=truth, quantiles=scaled)["crps"]
        scaled_table.append(
            {
                "group": group,
                "width factor": factor,
                "crps P-cp": mean_over(values=crps_p, rows=chosen),
                "crps P-cp, equal width": mean_over(values=crps_scaled, rows=chosen),
                "crps S-PRV": mean_over(values=crps_s, rows=chosen),
            }
        )
    show(name=f"{name}_equal_width", frame=pl.DataFrame(scaled_table))


# --------------------------------------------------------------------------- contrasts


def pair_frame(
    *,
    ids: np.ndarray,
    rows: tuple[np.ndarray, np.ndarray, np.ndarray],
    values: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Return the arms' CRPS on the selected rows, built by `losses_for_bootstrap`.

    Args:
        ids: The series ids, in array order.
        rows: The series position, origin, and row index of every selected row.
        values: The CRPS of each arm on those rows.

    Returns:
        The paired frame that `studies.bootstrap` reads.
    """
    k, o, h = rows
    origin_us = c.ORIGIN0_US + o.astype(np.int64) * c.DAY_US
    keys = {
        "time_series_id": pl.Series(ids[k], dtype=pl.Int8),
        "power_fcst_init_time": c.utc_series(micros=origin_us),
        "valid_time": c.utc_series(micros=origin_us + h.astype(np.int64) * c.STEP_US),
        "seed": pl.Series(np.zeros(len(k), dtype=np.int64)),
    }
    return losses_for_bootstrap(
        arm_losses={
            arm: pl.DataFrame({**keys, "crps": crps.astype(np.float64)})
            for arm, crps in values.items()
        }
    )


def interval_job(
    *,
    ids: np.ndarray,
    rows: tuple[np.ndarray, np.ndarray, np.ndarray],
    values: dict[str, np.ndarray],
    treatment: str,
    reference: str,
    level: float,
    n_resamples: int,
) -> tuple[float, float]:
    """Return the bootstrap interval of the paired CRPS difference (runs in a worker process)."""
    return bootstrap_difference_at_level(
        losses=pair_frame(ids=ids, rows=rows, values=values),
        treatment=treatment,
        reference=reference,
        metric="crps",
        level=level,
        n_resamples=n_resamples,
    )


def monthly_t_test(
    *, difference: np.ndarray, origin_month: np.ndarray, level: float
) -> dict[str, float]:
    """Return the t test on the monthly mean paired differences.

    Args:
        difference: The paired difference of every selected row.
        origin_month: The origin month label of every selected row.
        level: The two-sided coverage in percent, for the critical value.

    Returns:
        The number of months, the mean of the monthly means, its t statistic and two-sided p value,
        the critical value at `level`, and the interval half-width.
    """
    months = sorted(set(origin_month.tolist()))
    means = np.array([difference[origin_month == month].mean() for month in months])
    n = len(means)
    if n < 2:
        print("  monthly t test: n/a with one month")
        return {
            "months": n,
            "monthly mean": float(means.mean()),
            "t": float("nan"),
            "p": float("nan"),
            "t critical": float("nan"),
            "t half-width": float("nan"),
            "months with negative difference": int((means < 0).sum()),
        }
    standard_error = float(means.std(ddof=1) / np.sqrt(n))
    t_value = float(means.mean() / standard_error) if standard_error > 0 else float("inf")
    critical = float(stats.t.ppf(1.0 - (1.0 - level / 100.0) / 2.0, df=n - 1))
    return {
        "months": n,
        "monthly mean": float(means.mean()),
        "t": t_value,
        "p": float(2.0 * stats.t.sf(abs(t_value), df=n - 1)),
        "t critical": critical,
        "t half-width": critical * standard_error,
        "months with negative difference": int((means < 0).sum()),
    }


def run_contrasts(
    *,
    inputs: c.Inputs,
    losses: Losses,
    n_jobs: int,
    n_resamples: int,
    jan_base: np.ndarray,
    dec_base: np.ndarray,
) -> None:
    """Compute and print C1, C2, C5, and G-lb minus RR, each by group."""
    series_masks = c.group_masks(inputs=inputs)
    origin_month_label = c.origin_months()
    plan = [
        ("C1", "G-lb", "S-NRA", dec_base, BONFERRONI_LEVEL, n_resamples),
        ("C2", "D", "G-lb", dec_base, BONFERRONI_LEVEL, n_resamples),
        ("C5", "P-cp", "S-PRV", jan_base, BONFERRONI_LEVEL, n_resamples),
        (
            "G-lb minus RR",
            "G-lb",
            "RR",
            dec_base,
            REPORTED_LEVEL,
            min(n_resamples, REPORTED_RESAMPLES),
        ),
    ]
    jobs = [
        (name, treatment, reference, group, base, level, resamples)
        for name, treatment, reference, base, level, resamples in plan
        for group in GROUP_ORDER
        if treatment in losses and reference in losses
    ]
    results = []
    for start in range(0, len(jobs), n_jobs):
        chunk = jobs[start : start + n_jobs]
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = []
            prepared = []
            for name, treatment, reference, group, base, level, resamples in chunk:
                selection = (
                    base
                    & series_masks[group][:, None, None]
                    & np.isfinite(losses[treatment]["crps"])
                    & np.isfinite(losses[reference]["crps"])
                )
                if not selection.any():
                    continue
                rows = np.nonzero(selection)
                prepared.append((name, treatment, reference, group, level, selection))
                futures.append(
                    pool.submit(
                        interval_job,
                        ids=inputs.ids,
                        rows=rows,
                        values={
                            treatment: losses[treatment]["crps"][selection],
                            reference: losses[reference]["crps"][selection],
                        },
                        treatment=treatment,
                        reference=reference,
                        level=level,
                        n_resamples=resamples,
                    )
                )
            for (name, treatment, reference, group, level, selection), future in zip(
                prepared, futures, strict=True
            ):
                lower, upper = future.result()
                difference = (
                    losses[treatment]["crps"][selection] - losses[reference]["crps"][selection]
                )
                origin_of_row = np.nonzero(selection)[1]
                test = monthly_t_test(
                    difference=difference,
                    origin_month=origin_month_label[origin_of_row],
                    level=level,
                )
                half = (upper - lower) / 2.0
                z = float(stats.norm.ppf(1.0 - (1.0 - level / 100.0) / 2.0))
                results.append(
                    {
                        "contrast": name,
                        "treatment": treatment,
                        "reference": reference,
                        "group": group,
                        "level": round(level, 2),
                        "rows": int(selection.sum()),
                        "difference": float(difference.mean()),
                        "lower": lower,
                        "upper": upper,
                        "half-width": half,
                        "minimum detectable effect": half * (1.0 + float(stats.norm.ppf(0.8)) / z),
                        **test,
                        "bootstrap excludes zero": bool(lower > 0 or upper < 0),
                        "t test excludes zero": bool(
                            abs(test["monthly mean"]) > test["t half-width"]
                        ),  # False when the half-width is NaN (one month)
                    }
                )
    table = pl.DataFrame(results)
    show(name="contrasts", frame=table)
    print_decisions(table=table)


def print_decisions(*, table: pl.DataFrame) -> None:
    """Print the pass or fail reading of C1, C2, and C5 under the pre-registered rule."""
    print("\nDECISIONS (both the bootstrap interval and the monthly t test must exclude zero)")
    critical = table.select("level", "months", "t critical").unique().sort("level", "months")
    for row in critical.iter_rows(named=True):
        print(
            f"  t critical value at the {row['level']}% level with {row['months']} months "
            f"(scipy, two-sided): {row['t critical']:.3f}"
        )
    for name in ("C1", "C2", "C5"):
        for group in (c.GROUP_SIGNED, c.GROUP_MVA, c.POOLED, c.GROUP_SHORT):
            rows = table.filter(pl.col("contrast") == name, pl.col("group") == group)
            if rows.is_empty():
                continue
            row = rows.row(0, named=True)
            both = row["bootstrap excludes zero"] and row["t test excludes zero"]
            line = (
                f"  {name} {group}: difference {row['difference']:+.3f} "
                f"[{row['lower']:+.3f}, {row['upper']:+.3f}], half-width {row['half-width']:.3f}, "
                f"MDE {row['minimum detectable effect']:.3f}, t "
                f"{'n/a with one month' if row['months'] < 2 else format(row['t'], '+.2f')} vs "
                f"{row['t critical']:.2f}; excludes zero in both: {both}"
            )
            if name == "C1":
                line += f"; interval wholly below -{SMALLEST_WORTHWHILE_GAIN}: {row['upper'] < -SMALLEST_WORTHWHILE_GAIN}"
            if name in ("C1", "C2"):
                line += (
                    f"; point estimate worse than +{VETO_POINTS}: {row['difference'] > VETO_POINTS}"
                )
            print(line)
    print("  C1 and C2 each pass only if the signed-MW six and the MVA-only eight both pass.")


# --------------------------------------------------------------------------- sensitivities


def sensitivity_table(*, inputs: c.Inputs, losses: Losses, masks: dict[str, np.ndarray]) -> None:
    """Print C1's point estimate under the other event masks, and without D5, with D15 and D18."""
    treatment, reference = losses["G-lb"]["crps"], losses["S-NRA"]["crps"]
    difference = treatment - reference
    pooled = c.group_masks(inputs=inputs)[c.POOLED]
    labels = np.array(inputs.labels)
    table = []
    variants = {
        "event-free (primary mask)": (masks["primary: event-free"], pooled),
        "event-free (wide mask)": (masks["wide: event-free"], pooled),
        "event-free and no event in the look-back window": (masks["primary: clean window"], pooled),
        "event-free, D5 left out": (masks["primary: event-free"], pooled & (labels != "D5")),
        "event-free, D15 and D18 added": (
            masks["primary: event-free"],
            np.ones(len(inputs.ids), dtype=bool),
        ),
        "event-free, D5 only": (masks["primary: event-free"], labels == "D5"),
        "event-free, D15 and D18 only": (
            masks["primary: event-free"],
            c.group_masks(inputs=inputs)[c.GROUP_SHORT],
        ),
    }
    for name, (base, series_mask) in variants.items():
        chosen = base & series_mask[:, None, None] & np.isfinite(difference)
        table.append(
            {
                "variant": name,
                "rows": int(chosen.sum()),
                "G-lb minus S-NRA": mean_over(values=difference, rows=chosen),
            }
        )
    show(name="c1_sensitivity", frame=pl.DataFrame(table))
    months = c.origin_months()
    monthly = []
    for group, series_mask in c.group_masks(inputs=inputs).items():
        for month in sorted(set(months[c.FIRST_ORIGIN_DEC :])):
            chosen = (
                masks["primary: event-free"]
                & series_mask[:, None, None]
                & (months == month)[None, :, None]
                & np.isfinite(difference)
            )
            monthly.append(
                {
                    "group": group,
                    "origin month": month,
                    "G-lb minus S-NRA": mean_over(values=difference, rows=chosen),
                    "D minus G-lb": mean_over(
                        values=losses["D"]["crps"] - losses["G-lb"]["crps"], rows=chosen
                    ),
                }
            )
    show(name="monthly_differences", frame=pl.DataFrame(monthly))


def main() -> None:
    """Load every arm, score it, and print the tables and contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=4, help="processes for the bootstrap")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument(
        "--month",
        default=None,
        help="smoke test: read the sets that `arms --month` wrote and score only this month, YYYY-MM",
    )
    parser.add_argument(
        "--set-suffix",
        default=None,
        help="suffix of the set names that `arms --month` wrote; defaults to -t<YYYYMM> of --month",
    )
    arguments = parser.parse_args()
    inputs = c.load_inputs(load_prevailing=True, n_jobs=arguments.n_jobs)
    residuals = c.build_residuals(inputs=inputs)
    losses, kept = load_all(
        inputs=inputs, residuals=residuals, month=arguments.month, set_suffix=arguments.set_suffix
    )
    losses |= stored_668_losses(inputs=inputs)
    print(f"LOADED peak memory {c.peak_memory_gb():.2f} GB")
    events = load_events(inputs=inputs)
    dec = build_masks(
        inputs=inputs, events=events, first_origin=c.FIRST_ORIGIN_DEC, month=arguments.month
    )
    jan = build_masks(
        inputs=inputs, events=events, first_origin=c.FIRST_ORIGIN_JAN, month=arguments.month
    )
    check = losses["668 stored rows"]["crps"]
    matched = np.isfinite(check) & np.isfinite(losses["S-NRA"]["crps"])
    if matched.any():
        worst = np.abs(check[matched] - losses["S-NRA"]["crps"][matched]).max()
        print(
            f"CHECK S-NRA CRPS against experiment 668's stored per-row score: {int(matched.sum()):,} "
            f"rows, largest absolute difference {worst:.5f} points"
        )

    dec_rows = common_rows(losses=losses, labels=DEC_ARMS, base=dec["primary: event-free"])
    jan_rows = common_rows(losses=losses, labels=JAN_ARMS, base=jan["primary: event-free"])
    prevailing_rows = common_rows(
        losses=losses, labels=PREVAILING_ARM_LABELS, base=jan["primary: all"]
    )
    print(
        f"ROWS event-free, origins from 2025-12: {int(dec_rows.sum()):,}; from 2026-01: "
        f"{int(jan_rows.sum()):,}; all rows for C5: {int(prevailing_rows.sum()):,}"
    )
    group_table(
        inputs=inputs, losses=losses, labels=DEC_ARMS, rows=dec_rows, name="event_free_from_2025_12"
    )
    group_table(
        inputs=inputs, losses=losses, labels=JAN_ARMS, rows=jan_rows, name="event_free_from_2026_01"
    )
    group_table(
        inputs=inputs,
        losses=losses,
        labels=PREVAILING_ARM_LABELS,
        rows=prevailing_rows,
        name="all_rows_from_2026_01",
    )
    context = ("S-NRA", "G-lb", "D", "Climatology", "Manual heuristic")
    context_rows = common_rows(losses=losses, labels=context, base=dec["primary: event-free"])
    group_table(
        inputs=inputs, losses=losses, labels=context, rows=context_rows, name="context_rows"
    )
    print(
        "NOTE the 'Climatology' row is experiment 668's own `lib.climatology` scored in its per-row "
        "table, not `ClimatologyForecaster` of the production service."
    )
    band_labels = ("S-NRA", "G-lb", "D", "G smallest-lead", "RR", "R0-168", "M28")
    band_table(
        inputs=inputs,
        losses=losses,
        labels=band_labels,
        rows=common_rows(losses=losses, labels=band_labels, base=dec["primary: event-free"]),
        name="lead_bands_event_free",
    )
    slices = slice_masks(inputs=inputs, scored=dec["scored"])
    slice_labels = ("S-NRA", "G-lb", "D", "D-pv", "D-cold", "RR", "M28")
    slice_table(
        inputs=inputs,
        losses=losses,
        labels=slice_labels,
        slices=slices,
        base=common_rows(losses=losses, labels=slice_labels, base=dec["primary: event-free"]),
        name="slices_event_free",
    )
    # TODO(stage 2): C5 on the re-run bank rows only (the rows where experiment 667's re-run
    # forecasts exist) belongs with the stage 2 bank scoring, because the bank's forecasts are
    # re-run at each origin and cannot be scored from the stored rows read here.
    scenario_labels = ("S-NRA", "G-lb", "D", "RR", "M28", "R0-168")
    scenario_table(
        inputs=inputs, losses=losses, labels=scenario_labels, masks=dec, name="load_shift_scenarios"
    )
    departure_table(
        inputs=inputs,
        kept=kept,
        base=prevailing_rows & jan["primary: event-free"],
        name="prevailing_event_free",
    )
    if "G-lb" in losses and "D" in losses:
        sensitivity_table(inputs=inputs, losses=losses, masks=dec)
    run_contrasts(
        inputs=inputs,
        losses=losses,
        n_jobs=arguments.n_jobs,
        n_resamples=arguments.n_resamples,
        jan_base=prevailing_rows,
        dec_base=dec_rows,
    )
    print(f"PEAK MEMORY {c.peak_memory_gb():.2f} GB")


if __name__ == "__main__":
    main()

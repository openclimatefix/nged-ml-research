# ruff: noqa: E501
"""Experiment 706, stage 2: scores, contrasts C3, C4, and C6, and the known-answer test KA3 (CPU only).

Run it after `e00706_s2_fit.py --assemble`, from the repository root with the environment of
`CLAUDE.md` set and the folders of experiments 668, 683, and 688 on `PYTHONPATH` (the event spans):

    PP=$(ls -d ml-experiments/00668-* ml-experiments/00683-* ml-experiments/00688-* | paste -sd:)
    PYTHONPATH=$PP uv run --no-sync python ml-experiments/00706-*/e00706_s2_score.py --n-jobs 4

The post hoc status of stage 2 is printed with every contrast: stage 1 missed its gate, and the
maintainer asked for stage 2 anyway. Contrasts are read at the Bonferroni level for six readings
(99.17%), and the minimum detectable effect of each is printed before the estimates. The six
readings that carry the count are listed by `FAMILY` and printed at the start: the C3 contrasts of
N-gate and N-flag with N-rule on the copied share and on event-free real CRPS (pooled 14), and the
C4 contrasts of the real with the rewired graph on event-free real CRPS (pooled 14). The other
contrasts printed (C6, and every bank CRPS or C4 reading) are secondary, at the same level and
outside the count.

Two kinds of rows are scored.

Real rows (stage 1's rows, origins 2026-01 to 2026-06, event-free rows of the 14 Primaries,
`primary: event-free` of experiment 688): the prediction sets `n-rule`, `n-flag-real`,
`n-flag-rewired`, `n-gate-real`, and `n-gate-rewired` against `S-NRA` and `G-lb`, with the paired
month-block bootstrap and the t test on monthly means of `e00706_score.py`.

Bank rows (the injected shifts of experiment 667, the held-out month of each fold, even half-hours,
rows whose shift is at least 3 points of capacity and whose lead is at most 240 hours, of events
whose depth is at least 3% of capacity): the job files hold each seed's correction, so the bootstrap
resamples the (month, Primary) clusters that hold the events and, inside each event, seeds. A second
printed line per contrast resamples single events instead.

- Copied share of an event: the sum over its rows of the sign of the shift times (the arm's median
  in the injected world minus the Candidate's median in the real world), divided by the sum of the
  absolute shifts. The arm's median in the injected world is the re-run Candidate's median plus the
  arm's correction, and the real-world median is the Candidate's own, from the bank's
  `real_<month>.npz` files (never the stored NRA median). A share of 1 means the forecast copied the
  shift in full.
- CRPS on bank rows is against clean power (the truth without the shift), as the NRA arms are scored.
- C3: N-gate minus N-rule and N-flag minus N-rule (each on the real graph), on the copied share and
  on event-free real CRPS.
- C4: real graph minus rewired graph for each of N-flag and N-gate. The verdict reading is event-free
  real CRPS. The bank readings are labelled "mechanism check, not the neighbour test": the generator
  puts the mirrored step of a transfer on the real graph's neighbours only, so a network that reads
  the real graph's neighbours can detect the transfer by construction.
- C6: N-gate minus N-flag, on the copied share and on event-free real CRPS.
- KA3: on held-out bank transfers of at least 10% of capacity with a mirrored neighbour step, the
  copied share of N-gate and of N-flag must be at most 0.25; on the held-out own-only 60-day ramps it
  must be at least 0.6; both are also read with the neighbour inputs zeroed.
- The mean gates, the switching probability, and the area under the curve of the switching
  probability on real rows (against the labelled-event mask) and on bank entries (against the
  generator's transfer flag) are printed.

Three caveats apply to the readings. Arm N-flag has no lead input, so C6 on real CRPS partly measures
the fading with lead that only N-gate can learn. The transfer flag that N-flag is trained on is 1
for every injected event, own-only events included. The lead input of N-gate, with every forecast
issued at 12:00 UTC, also encodes the hour of day of the target (allowed, but a caveat).

The bank's re-run forecasts are the Candidate's own, so the bank shares and CRPS are not comparable
with the real-row tables. The own-only ramp worlds are no-re-run worlds on the real rows, so their
shares understate how far a re-run forecast would have followed the ramp.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00706_common as c  # ty: ignore[unresolved-import]
import e00706_s2_data as d  # ty: ignore[unresolved-import]
import e00706_s2_fit as fit  # ty: ignore[unresolved-import]
import e00706_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy import stats
from studies.eval_protocol.metrics import LEVELS, MEDIAN_INDEX
from studies.experiment_store import read_predictions

KA3_TRANSFER_SHARE_MAX: Final[float] = 0.25
KA3_RAMP_SHARE_MIN: Final[float] = 0.6
KA3_MIN_DEPTH: Final[float] = 0.10
EVENT_MIN_DEPTH: Final[float] = fit.SHIFT_FLOOR_POINTS / 100.0
"""Events shallower than this share of capacity are left out of the bank readings."""
MECHANISM_LABEL: Final[str] = " (mechanism check, not the neighbour test)"
FAMILY: Final[tuple[str, ...]] = (
    "C3 copied share, N-gate minus N-rule (bank events)",
    "C3 copied share, N-flag minus N-rule (bank events)",
    f"C3 event-free real CRPS ({c.POOLED}), N-gate minus N-rule",
    f"C3 event-free real CRPS ({c.POOLED}), N-flag minus N-rule",
    f"C4 event-free real CRPS ({c.POOLED}), N-gate real minus rewired",
    f"C4 event-free real CRPS ({c.POOLED}), N-flag real minus rewired",
)
"""The six readings that carry the Bonferroni count of stage 2."""
NOTES: Final[tuple[str, ...]] = (
    "N-flag has no lead input, so C6 on real CRPS partly measures the fading with lead that only N-gate can learn.",
    "flag_y is 1 for every injected event, own-only events included, so N-flag's probability detects any injected step.",
    "N-gate's lead input encodes the hour of day of the target, because every forecast is issued at 12:00 UTC (allowed, a caveat).",
    "Bank C4 is circular (the generator puts the mirrored step on real-graph neighbours only): it is a mechanism check, not the neighbour test. The verdict reading of C4 is event-free real CRPS.",
)
EVENT_RESAMPLES_PER_CHUNK: Final[int] = 250
ARM_LABELS: Final[dict[str, str]] = {
    "n-rule": "N-rule",
    "n-flag-real": "N-flag real",
    "n-flag-rewired": "N-flag rewired",
    "n-gate-real": "N-gate real",
    "n-gate-rewired": "N-gate rewired",
}
LEARNED: Final[tuple[tuple[str, str], ...]] = (
    ("n-flag", "real"),
    ("n-flag", "rewired"),
    ("n-gate", "real"),
    ("n-gate", "rewired"),
)
GATE_NAMES: Final[tuple[str, ...]] = tuple(
    f"{int(days)} d {name}"
    for days in d.WINDOW_DAYS
    for name in ("global", "irradiance", "cold", "neighbours")
)
POST_HOC_NOTE: Final[str] = (
    "POST HOC: stage 1 missed the 0.02 gate and stage 2 runs on the maintainer's instruction; "
    "contrasts are read at the 99.17% level for six readings."
)
LEVELS32: Final = LEVELS.astype(np.float32)
type Losses = dict[str, dict[str, np.ndarray]]


# --------------------------------------------------------------------------- real rows


def load_real_losses(*, inputs: c.Inputs) -> Losses:
    """Return the per-row losses of S-NRA, G-lb, N-rule, and the four learned arms on the stage 1 grid."""
    truth = c.target_truth(inputs=inputs)
    losses: Losses = {"S-NRA": c.loss_arrays(truth=truth, quantiles=inputs.q_nra)}
    sets = {"G-lb": ("g", c.ROW_SET_ID)} | {
        label: (name, c.ROW_SET_ID_FROM_JANUARY) for name, label in ARM_LABELS.items()
    }
    for label, (name, row_set_id) in sets.items():
        frame = read_predictions(experiment_id=c.EXPERIMENT_ID, name=name, row_set_id=row_set_id)
        losses[label] = c.loss_arrays(
            truth=truth,
            quantiles=c.dense_from_frame(frame=frame, ids=inputs.ids, capacity=inputs.capacity),
        )
        print(f"LOADED {name} as {label}: {frame.height:,} rows")
    return losses


def real_contrast_rows(
    *,
    inputs: c.Inputs,
    losses: Losses,
    base: np.ndarray,
    n_jobs: int,
    n_resamples: int,
    plan: tuple[tuple[str, str, str], ...],
) -> pl.DataFrame:
    """Return the real-row contrasts of `plan` (name, treatment, reference) by group, with intervals."""
    series_masks = c.group_masks(inputs=inputs)
    origin_month_label = c.origin_months()
    jobs, prepared = [], []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for name, treatment, reference in plan:
            for group in score.GROUP_ORDER:
                selection = (
                    base
                    & series_masks[group][:, None, None]
                    & np.isfinite(losses[treatment]["crps"])
                    & np.isfinite(losses[reference]["crps"])
                )
                if not selection.any():
                    continue
                prepared.append((name, treatment, reference, group, selection))
                jobs.append(
                    pool.submit(
                        score.interval_job,
                        ids=inputs.ids,
                        rows=np.nonzero(selection),
                        values={
                            treatment: losses[treatment]["crps"][selection],
                            reference: losses[reference]["crps"][selection],
                        },
                        treatment=treatment,
                        reference=reference,
                        level=score.BONFERRONI_LEVEL,
                        n_resamples=n_resamples,
                    )
                )
        results = []
        z = float(stats.norm.ppf(1.0 - (1.0 - score.BONFERRONI_LEVEL / 100.0) / 2.0))
        for (name, treatment, reference, group, selection), future in zip(
            prepared, jobs, strict=True
        ):
            lower, upper = future.result()
            difference = losses[treatment]["crps"][selection] - losses[reference]["crps"][selection]
            test = score.monthly_t_test(
                difference=difference,
                origin_month=origin_month_label[np.nonzero(selection)[1]],
                level=score.BONFERRONI_LEVEL,
            )
            half = (upper - lower) / 2.0
            results.append(
                {
                    "contrast": name,
                    "treatment": treatment,
                    "reference": reference,
                    "group": group,
                    "rows": int(selection.sum()),
                    "minimum detectable effect": half * (1.0 + float(stats.norm.ppf(0.8)) / z),
                    "difference": float(difference.mean()),
                    "lower": lower,
                    "upper": upper,
                    **test,
                    "bootstrap excludes zero": bool(lower > 0 or upper < 0),
                    "t test excludes zero": bool(abs(test["monthly mean"]) > test["t half-width"]),
                }
            )
    return pl.DataFrame(results)


# --------------------------------------------------------------------------- the bank


class EventStats:
    """Per-event copied share and CRPS of every arm variant, with one column per event.

    Each array has shape (seeds, events); an arm with no seed has one row, and a difference of two
    arms broadcasts.
    """

    def __init__(self) -> None:
        """Start with no events."""
        self.share: dict[str, list[np.ndarray]] = {}
        self.crps: dict[str, list[np.ndarray]] = {}
        self.meta: list[pl.DataFrame] = []

    def add(self, *, name: str, share: np.ndarray, crps: np.ndarray) -> None:
        """Append one month's per-event arrays (seeds, events) of a variant."""
        self.share.setdefault(name, []).append(share)
        self.crps.setdefault(name, []).append(crps)

    def get(self, *, kind: str, name: str) -> np.ndarray:
        """Return the (seeds, all events) array of `share` or `crps` for a variant."""
        parts = (self.share if kind == "share" else self.crps)[name]
        return np.concatenate(parts, axis=1)


def _per_event(*, values: np.ndarray, event: np.ndarray, n_events: int) -> np.ndarray:
    """Return the sum of `values` per event; `values` is (rows,) or (seeds, rows)."""
    if values.ndim == 1:
        return np.bincount(event, weights=values, minlength=n_events)[None, :]
    return np.stack([np.bincount(event, weights=row, minlength=n_events) for row in values])


def _crps(*, y: np.ndarray, base: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Return the CRPS in points of rows with base quantiles (n, 11) and corrections (…, n)."""
    error = y - (base + correction[..., None])
    pinball = np.maximum(LEVELS32 * error, (LEVELS32 - 1.0) * error)
    return 2.0 * pinball.mean(axis=-1)


def _variant_stats(
    *,
    correction: np.ndarray,
    event: np.ndarray,
    n_events: int,
    base: np.ndarray,
    y: np.ndarray,
    injected_median: np.ndarray,
    real_median: np.ndarray,
    sign: np.ndarray,
    denominator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one variant's copied share and mean CRPS per event, each (seeds, events).

    Args:
        correction: The correction of every row, shape (rows,) or (seeds, rows).
        event: The event index of every row.
        n_events: The number of events.
        base: The base quantiles of every row, (rows, 11).
        y: The clean power of every row, (rows,).
        injected_median: The base median of every row.
        real_median: The Candidate's median of every row in the real world.
        sign: The sign of the injected shift of every row.
        denominator: The sum of the absolute injected shifts per event, (1, events).

    Returns:
        The share and the CRPS against clean power.
    """
    corrections = correction[None, :] if correction.ndim == 1 else correction
    count = np.maximum(np.bincount(event, minlength=n_events), 1)[None, :]
    share = (
        _per_event(
            values=sign * (injected_median + corrections - real_median),
            event=event,
            n_events=n_events,
        )
        / denominator
    )
    crps = np.concatenate(
        [
            _per_event(
                values=_crps(y=y[:, None], base=base, correction=row),
                event=event,
                n_events=n_events,
            )
            for row in corrections
        ]
    )
    return share, crps / count


def bank_event_stats(*, n_seeds: int) -> tuple[EventStats, pl.DataFrame]:
    """Return the per-event stats of every bank variant over the six fold months, and the events' table.

    A variant is S-NRA in the injected world (no correction), N-rule, and each learned arm and graph,
    each also with the neighbour inputs zeroed. Rows are those of the fold month's bank with a shift
    of at least 3 points of capacity and a lead of at most 240 hours. The table has one row per
    event, with its depth and a `cluster` number for its (month, Primary).
    """
    stats_out = EventStats()
    metas = []
    for month in d.FOLD_MONTHS:
        rows = d.load_arrays(path=d.CACHE_DIR / f"bank_{month}_rows.npz")
        table = d.load_arrays(path=d.CACHE_DIR / f"bank_{month}_entries.npz")
        force = (
            (np.abs(rows["shift"]) >= fit.SHIFT_FLOOR_POINTS)
            & (rows["h"] >= 1)
            & (rows["h"] <= c.MAX_LEAD_HALF_HOURS)
        )
        force &= np.isfinite(rows["y"]) & np.isfinite(rows["q"][:, MEDIAN_INDEX])
        force &= np.isfinite(rows["q_real"])
        h = rows["h"][force].astype(np.int64)
        real_median = rows["q_real"][force].astype(np.float64)
        q, y, shift = (
            rows["q"][force],
            rows["y"][force].astype(np.float64),
            rows["shift"][force].astype(np.float64),
        )
        key = rows["pass_"][force].astype(np.int64) * 16 + rows["k"][force].astype(np.int64)
        unique, event = np.unique(key, return_inverse=True)
        n_events = len(unique)
        events = pl.read_parquet(d.HEAD_DATA / f"events_{month}.parquet")
        meta = pl.DataFrame({"pass": unique // 16, "primary": unique % 16}).join(
            events.select("pass", "primary", "depth", "n_receivers", "kind", "length_days", "sign"),
            on=["pass", "primary"],
            how="left",
        )
        metas.append(meta.with_columns(month=pl.lit(month)))
        denominator = _per_event(values=np.abs(shift), event=event, n_events=n_events)
        injected_median = q[:, MEDIAN_INDEX].astype(np.float64)
        band = c.BAND_OF_H[h]
        entry = rows["entry"][force].astype(np.int64)

        fixed = {
            "event": event,
            "n_events": n_events,
            "base": q,
            "y": y,
            "injected_median": injected_median,
            "real_median": real_median,
            "sign": np.sign(shift),
            "denominator": denominator,
        }
        variants: dict[str, np.ndarray] = {"S-NRA re-run": np.zeros(len(y))}
        rule = table["w_rule"][entry] * table["box"][entry, band]
        variants["N-rule"] = rule.astype(np.float64)
        for arm, graph in LEARNED:
            for suffix, label in (("", ""), ("_z", " zero neighbours")):
                seeds = []
                for seed in range(n_seeds):
                    job = fit.Job(arm=arm, graph=graph, month=month, seed=seed)
                    job_arrays = d.load_arrays(path=fit.JOBS_DIR / f"{job.key}.npz")
                    seeds.append(job_arrays[f"corr_bank{suffix}"][force].astype(np.float64))
                variants[f"{ARM_LABELS[f'{arm}-{graph}']}{label}"] = np.stack(seeds)
        for name, correction in variants.items():
            share, crps = _variant_stats(correction=correction, **fixed)
            stats_out.add(name=name, share=share, crps=crps)
        print(f"  bank {month}: {n_events} events, {int(force.sum()):,} rows in force", flush=True)
    meta = pl.concat(metas)
    labels = (meta["month"] + ":" + meta["primary"].cast(pl.Utf8)).to_numpy()
    _, cluster = np.unique(labels, return_inverse=True)
    return stats_out, meta.with_columns(cluster=pl.Series(cluster.astype(np.int64)))


def deep_events(*, meta: pl.DataFrame) -> np.ndarray:
    """Return which events have a depth of at least `EVENT_MIN_DEPTH` of capacity."""
    return (meta["depth"].fill_null(0.0) >= EVENT_MIN_DEPTH - 1e-9).to_numpy()


def event_bootstrap(
    *, difference: np.ndarray, cluster: np.ndarray, level: float, n_resamples: int
) -> tuple[float, float, float]:
    """Return the interval of the mean over events of a paired difference.

    The bootstrap resamples clusters and, inside events, seeds.

    Each resample draws the clusters with replacement, takes every event of a drawn cluster, and
    replaces each event's seed values by a draw of seeds with replacement. The statistic is the mean
    over the drawn events of the seed mean. Pass one cluster per event to resample single events.

    Args:
        difference: The paired differences, shape (seeds, events), or (1, events) when neither arm
            has a seed.
        cluster: The cluster number of each event, shape (events,).
        level: The coverage in percent.
        n_resamples: The number of resamples.

    Returns:
        The lower and upper bounds, and the point estimate (the mean over events of the seed mean).
    """
    n_seeds, n_events = difference.shape
    if n_events == 0:
        return float("nan"), float("nan"), float("nan")
    if not np.isfinite(difference).all():
        raise ValueError("a paired difference is not finite; a job file holds a NaN correction")
    order = np.argsort(cluster, kind="stable")
    difference = difference[:, order]
    _, sizes = np.unique(cluster[order], return_counts=True)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    n_clusters = len(sizes)
    rng = np.random.default_rng([c.EXPERIMENT_ID, 5, n_events, n_clusters])
    columns = np.arange(n_events)[None, :, None]
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, EVENT_RESAMPLES_PER_CHUNK):
        size = min(EVENT_RESAMPLES_PER_CHUNK, n_resamples - start)
        seeds = rng.integers(n_seeds, size=(size, n_events, n_seeds))
        event_means = difference[seeds, columns].mean(axis=2)
        cluster_sums = np.add.reduceat(event_means, starts, axis=1)
        drawn = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=size)
        means[start : start + size] = (drawn * cluster_sums).sum(axis=1) / (drawn @ sizes)
    tail = (100.0 - level) / 2.0
    lower, upper = np.percentile(means, [tail, 100.0 - tail])
    return float(lower), float(upper), float(difference.mean())


def event_contrasts(
    *,
    stats_in: EventStats,
    plan: tuple[tuple[str, str, str, str], ...],
    mask: np.ndarray,
    cluster: np.ndarray,
    n_resamples: int,
) -> pl.DataFrame:
    """Return the bank contrasts of `plan` (contrast, metric, treatment, reference) over the events in `mask`.

    Each contrast has two rows: the first resamples the (month, Primary) clusters named by `cluster`
    (the reading to use), and the second resamples single events.
    """
    z = float(stats.norm.ppf(1.0 - (1.0 - score.BONFERRONI_LEVEL / 100.0) / 2.0))
    table = []
    for contrast, metric, treatment, reference in plan:
        difference = (
            stats_in.get(kind=metric, name=treatment) - stats_in.get(kind=metric, name=reference)
        )[:, mask]
        chosen = cluster[mask]
        for resampled, groups in (
            ("month-Primary clusters (reading to use)", chosen),
            ("single events (second reading)", np.arange(len(chosen))),
        ):
            lower, upper, estimate = event_bootstrap(
                difference=difference,
                cluster=groups,
                level=score.BONFERRONI_LEVEL,
                n_resamples=n_resamples,
            )
            half = (upper - lower) / 2.0
            table.append(
                {
                    "contrast": contrast,
                    "metric": "copied share" if metric == "share" else "CRPS against clean power",
                    "treatment": treatment,
                    "reference": reference,
                    "resampled over": resampled,
                    "events": int(mask.sum()),
                    "clusters": len(np.unique(chosen)),
                    "minimum detectable effect": half * (1.0 + float(stats.norm.ppf(0.8)) / z),
                    "difference": estimate,
                    "lower": lower,
                    "upper": upper,
                    "excludes zero": bool(lower > 0 or upper < 0),
                }
            )
    return pl.DataFrame(table)


def print_family() -> None:
    """Print the six readings that carry the Bonferroni count, the family size, and the level used."""
    print("\nBONFERRONI FAMILY of stage 2:")
    for index, reading in enumerate(FAMILY, start=1):
        print(f"  {index}. {reading}")
    expected = 100.0 * (1.0 - 0.05 / len(FAMILY))
    print(
        f"  family size {len(FAMILY)}, level used {score.BONFERRONI_LEVEL:.2f}% "
        f"(1 - 0.05 / {len(FAMILY)} = {expected:.2f}%); every other reading is secondary, "
        "printed at the same level"
    )
    if abs(expected - score.BONFERRONI_LEVEL) > 0.01:
        print("  WARNING: the level used does not match the family size")


def share_table(*, stats_in: EventStats, meta: pl.DataFrame) -> None:
    """Print the mean copied share and CRPS of every variant on the deep bank events.

    The events are the bank events of depth at least 3% and KA3's transfers.
    """
    big = ((meta["depth"] >= KA3_MIN_DEPTH) & (meta["n_receivers"] >= 1)).to_numpy()
    everything = deep_events(meta=meta)
    table = []
    for name in stats_in.share:
        row: dict[str, float | str | int] = {"variant": name}
        for label, mask in (
            ("events of depth at least 3%", everything),
            ("transfers of at least 10% with a mirrored step", big),
        ):
            row[f"events: {label}"] = int(mask.sum())
            row[f"share: {label}"] = float(stats_in.get(kind="share", name=name)[:, mask].mean())
            row[f"CRPS: {label}"] = float(stats_in.get(kind="crps", name=name)[:, mask].mean())
        table.append(row)
    score.show(
        name="s2_bank_share_and_crps", frame=pl.DataFrame(table), note=score.POINT_ESTIMATE_NOTE
    )


# --------------------------------------------------------------------------- ramps, KA3, gates, AUC


def ramp_shares(*, inputs: c.Inputs, n_seeds: int) -> dict[str, np.ndarray]:
    """Return each variant's copied share on the own-only ramp worlds, shape (seeds, worlds), over six folds.

    A world's share is the sum over its rows with a ramp of at least 3 points and a lead of at most
    240 hours of the sign of the ramp times the correction, divided by the sum of the absolute ramp
    (the forecast is the stored one, so the injected median minus S-NRA's median is the
    correction). The rows are those of the fold month's origins.
    """
    out: dict[str, list[np.ndarray]] = {}
    for month in d.FOLD_MONTHS:
        table = d.load_arrays(path=d.CACHE_DIR / f"ramp_test_{month}_entries.npz")
        n_entries = len(table["origin"])
        entry = np.repeat(np.arange(n_entries), c.N_H)
        h = np.tile(np.arange(c.N_H), n_entries)
        stamp = c.ORIGIN_STAMP + c.PER_DAY * table["origin"][entry].astype(np.int64) + h
        ramp = d.ramp_shift(
            stamp=stamp, start=table["ramp_s0"][entry], sign=table["ramp_sign"][entry]
        )
        k, o = table["series"][entry].astype(np.int64), table["origin"][entry].astype(np.int64)
        usable = (
            (np.abs(ramp) >= fit.SHIFT_FLOOR_POINTS)
            & (h >= 1)
            & (h <= c.MAX_LEAD_HALF_HOURS)
            & (stamp < c.N_STAMPS)
            & np.isfinite(inputs.q_nra[k, o, h, MEDIAN_INDEX])
        )
        n_origins = len(c.origin_of_month(month=month))
        world = (entry // n_origins)[usable]
        n_worlds = n_entries // n_origins
        denominator = np.bincount(world, weights=np.abs(ramp[usable]), minlength=n_worlds)[None, :]
        sign = np.sign(ramp[usable])
        rule = (table["w_rule"][entry] * table["box"][entry, c.BAND_OF_H[h]]).astype(np.float64)
        out.setdefault("N-rule", []).append(
            _per_event(values=sign * rule[usable], event=world, n_events=n_worlds) / denominator
        )
        for arm, graph in LEARNED:
            for suffix, label in (("", ""), ("_z", " zero neighbours")):
                seeds = []
                for seed in range(n_seeds):
                    job = fit.Job(arm=arm, graph=graph, month=month, seed=seed)
                    arrays = d.load_arrays(path=fit.JOBS_DIR / f"{job.key}.npz")
                    seeds.append(
                        arrays[f"corr_ramp{suffix}"].reshape(-1)[usable].astype(np.float64)
                    )
                out.setdefault(f"{ARM_LABELS[f'{arm}-{graph}']}{label}", []).append(
                    _per_event(values=sign * np.stack(seeds), event=world, n_events=n_worlds)
                    / denominator
                )
    return {name: np.concatenate(parts, axis=1) for name, parts in out.items()}


def ka3_table(*, stats_in: EventStats, meta: pl.DataFrame, ramps: dict[str, np.ndarray]) -> None:
    """Print KA3: copied share on bank transfers (at most 0.25) and on own-only ramps (at least 0.6)."""
    big = ((meta["depth"] >= KA3_MIN_DEPTH) & (meta["n_receivers"] >= 1)).to_numpy()
    table = []
    for name, ramp_values in ramps.items():
        if name == "N-rule":
            continue
        transfer = float(stats_in.get(kind="share", name=name)[:, big].mean())
        ramp = float(ramp_values.mean())
        table.append(
            {
                "variant": name,
                "transfers (events)": int(big.sum()),
                "transfer share": transfer,
                "transfer share at most 0.25": transfer <= KA3_TRANSFER_SHARE_MAX,
                "ramp worlds": ramp_values.shape[1],
                "ramp share": ramp,
                "ramp share at least 0.6": ramp >= KA3_RAMP_SHARE_MIN,
            }
        )
    score.show(
        name="s2_ka3",
        frame=pl.DataFrame(table),
        note="KA3 on held-out bank transfers and own-only ramps. A zero-neighbour row zeroes the neighbour inputs of the fitted network.",
    )


def gate_table(*, n_seeds: int) -> None:
    """Print the mean gate of every look-back offset of arm N-gate, over folds and seeds, on three row sets."""
    table = []
    for graph in d.GRAPH_NAMES:
        accumulated: dict[str, list[np.ndarray]] = {
            "real rows": [],
            "bank rows in force": [],
            "ramp rows in force": [],
        }
        for month in d.FOLD_MONTHS:
            for seed in range(n_seeds):
                job = fit.Job(arm="n-gate", graph=graph, month=month, seed=seed)
                arrays = d.load_arrays(path=fit.JOBS_DIR / f"{job.key}.npz")
                for label, key in (
                    ("real rows", "gate_real"),
                    ("bank rows in force", "gate_bank"),
                    ("ramp rows in force", "gate_ramp"),
                ):
                    accumulated[label].append(arrays[key])
        for label, parts in accumulated.items():
            mean = np.nanmean(np.stack(parts), axis=0)
            table.append(
                {"graph": graph, "rows": label, **dict(zip(GATE_NAMES, mean.tolist(), strict=True))}
            )
    score.show(
        name="s2_gate_means",
        frame=pl.DataFrame(table),
        note="Mean gate per look-back offset; the start value is 0.75 for the 14 d global offset and about 0.02 for the other 15.",
    )


def _auc(*, scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the area under the ROC curve of `scores` for the boolean `labels` (rank form, ties averaged)."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def flag_auc_table(*, inputs: c.Inputs, n_seeds: int) -> None:
    """Print the area under the ROC curve of N-flag's switching probability on real rows and on bank entries.

    Real entries are (series, origin) pairs of the fold months labelled 1 when the 28-day window ends
    inside or holds a labelled event of the primary mask (events of experiment 688). Bank entries are
    labelled by the generator's transfer flag, among entries with a covered window.
    """
    inside, _, _ = c.load_event_masks(inputs=inputs, kind="primary")
    window_flag = c.window_overlap(
        flags=inside, window_half_hours=28 * c.PER_DAY, lag_half_hours=c.LAG_6H
    )
    table = []
    for graph in d.GRAPH_NAMES:
        real_scores, real_labels, bank_scores, bank_labels = [], [], [], []
        for month in d.FOLD_MONTHS:
            origins = c.origin_of_month(month=month)
            arrays = [
                d.load_arrays(
                    path=fit.JOBS_DIR
                    / f"{fit.Job(arm='n-flag', graph=graph, month=month, seed=seed).key}.npz"
                )
                for seed in range(n_seeds)
            ]
            real_p = np.mean([a["p_real"] for a in arrays], axis=0)
            real_scores.append(real_p.reshape(-1))
            real_labels.append(window_flag[:, origins].reshape(-1))
            entries = d.load_arrays(path=d.CACHE_DIR / f"bank_{month}_entries.npz")
            bank_p = np.mean([a["p_bank"] for a in arrays], axis=0)
            covered = entries["covered"]
            bank_scores.append(bank_p[covered])
            bank_labels.append(entries["flag_y"][covered] == 1)
        scores_real, labels_real = np.concatenate(real_scores), np.concatenate(real_labels)
        scores_bank, labels_bank = np.concatenate(bank_scores), np.concatenate(bank_labels)
        table.append(
            {
                "graph": graph,
                "real entries": len(scores_real),
                "real entries with an event in the window": int(labels_real.sum()),
                "AUC on real rows": _auc(scores=scores_real, labels=labels_real),
                "mean p on real rows": float(scores_real.mean()),
                "bank entries": len(scores_bank),
                "bank entries with a transfer in the window": int(labels_bank.sum()),
                "AUC on bank entries": _auc(scores=scores_bank, labels=labels_bank),
                "mean p with a transfer": float(scores_bank[labels_bank].mean()),
                "mean p without": float(scores_bank[~labels_bank].mean()),
            }
        )
    score.show(name="s2_flag_auc", frame=pl.DataFrame(table), note=score.POINT_ESTIMATE_NOTE)


# --------------------------------------------------------------------------- main


def main() -> None:
    """Score the learned arms and print the tables, the contrasts, and the known-answer test KA3."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-jobs", type=int, default=4, help="processes for the real-row bootstrap"
    )
    parser.add_argument("--n-resamples", type=int, default=score.N_RESAMPLES)
    parser.add_argument("--n-seeds", type=int, default=fit.N_SEEDS)
    arguments = parser.parse_args()
    for note in NOTES:
        print(f"NOTE: {note}")
    print_family()
    inputs = c.load_inputs(load_prevailing=False, n_jobs=arguments.n_jobs)
    losses = load_real_losses(inputs=inputs)
    events = score.load_events(inputs=inputs)
    jan = score.build_masks(
        inputs=inputs, events=events, first_origin=c.FIRST_ORIGIN_JAN, month=None
    )
    labels = ("S-NRA", "G-lb", *ARM_LABELS.values())
    rows = score.common_rows(losses=losses, labels=labels, base=jan["primary: event-free"])
    print(f"ROWS event-free, origins from 2026-01: {int(rows.sum()):,}")
    score.group_table(
        inputs=inputs, losses=losses, labels=labels, rows=rows, name="s2_event_free_from_2026_01"
    )
    score.band_table(
        inputs=inputs, losses=losses, labels=labels, rows=rows, name="s2_lead_bands_event_free"
    )
    slices = score.slice_masks(inputs=inputs, scored=jan["scored"])
    score.slice_table(
        inputs=inputs,
        losses=losses,
        labels=labels,
        slices=slices,
        base=rows,
        name="s2_slices_event_free",
    )
    score.scenario_table(
        inputs=inputs, losses=losses, labels=labels, masks=jan, name="s2_load_shift_scenarios"
    )
    print(f"LOADED real rows, peak memory {c.peak_memory_gb():.2f} GB")

    plan = (
        ("C3 N-gate minus N-rule", "N-gate real", "N-rule"),
        ("C3 N-flag minus N-rule", "N-flag real", "N-rule"),
        ("C4 N-gate real minus rewired", "N-gate real", "N-gate rewired"),
        ("C4 N-flag real minus rewired", "N-flag real", "N-flag rewired"),
        ("C6 N-gate minus N-flag", "N-gate real", "N-flag real"),
    )
    real_frame = real_contrast_rows(
        inputs=inputs,
        losses=losses,
        base=rows,
        n_jobs=arguments.n_jobs,
        n_resamples=arguments.n_resamples,
        plan=plan,
    )
    print("MINIMUM DETECTABLE EFFECT on real rows (CRPS points), before the estimates:")
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_float="mixed"):
        print(real_frame.select("contrast", "group", "minimum detectable effect"))
    score.show(
        name="s2_contrasts_real_rows",
        frame=real_frame,
        note=POST_HOC_NOTE
        + " 'Not resolved' reads inconclusive. The C4 rows here, on event-free real rows, are the verdict reading of C4.",
    )

    stats_in, meta = bank_event_stats(n_seeds=arguments.n_seeds)
    deep = deep_events(meta=meta)
    cluster = meta["cluster"].to_numpy()
    print(
        f"BANK EVENTS {meta.height}, of depth at least {100 * EVENT_MIN_DEPTH:.0f}%: {int(deep.sum())}, "
        f"in {len(np.unique(cluster[deep]))} (month, Primary) clusters"
    )
    bank_share = event_contrasts(
        stats_in=stats_in,
        plan=(
            ("C3 N-gate minus N-rule", "share", "N-gate real", "N-rule"),
            ("C3 N-flag minus N-rule", "share", "N-flag real", "N-rule"),
            ("C6 N-gate minus N-flag", "share", "N-gate real", "N-flag real"),
        ),
        mask=deep,
        cluster=cluster,
        n_resamples=arguments.n_resamples,
    )
    bank_crps = event_contrasts(
        stats_in=stats_in,
        plan=(
            (
                f"C4 N-gate real minus rewired{MECHANISM_LABEL}",
                "crps",
                "N-gate real",
                "N-gate rewired",
            ),
            (
                f"C4 N-flag real minus rewired{MECHANISM_LABEL}",
                "crps",
                "N-flag real",
                "N-flag rewired",
            ),
            (
                f"C4 N-gate real minus rewired{MECHANISM_LABEL}",
                "share",
                "N-gate real",
                "N-gate rewired",
            ),
            (
                f"C4 N-flag real minus rewired{MECHANISM_LABEL}",
                "share",
                "N-flag real",
                "N-flag rewired",
            ),
            ("C3 N-gate minus N-rule", "crps", "N-gate real", "N-rule"),
            ("C3 N-flag minus N-rule", "crps", "N-flag real", "N-rule"),
            ("C6 N-gate minus N-flag", "crps", "N-gate real", "N-flag real"),
        ),
        mask=deep,
        cluster=cluster,
        n_resamples=arguments.n_resamples,
    )
    print("MINIMUM DETECTABLE EFFECT on bank events (C3, C4, and C6), before the estimates:")
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_float="mixed", fmt_str_lengths=100):
        print(
            pl.concat([bank_share, bank_crps]).select(
                "contrast", "metric", "resampled over", "minimum detectable effect"
            )
        )
    share_table(stats_in=stats_in, meta=meta)
    score.show(
        name="s2_contrasts_bank_share",
        frame=bank_share,
        note=POST_HOC_NOTE + " Resampled over clusters (or events), seeds inside.",
    )
    score.show(
        name="s2_contrasts_bank_crps",
        frame=bank_crps,
        note=POST_HOC_NOTE
        + " Resampled over clusters (or events), seeds inside. C4 rows here are a mechanism check, not the neighbour test; the verdict reading of C4 is s2_contrasts_real_rows.",
    )
    ramps = ramp_shares(inputs=inputs, n_seeds=arguments.n_seeds)
    ka3_table(stats_in=stats_in, meta=meta, ramps=ramps)
    gate_table(n_seeds=arguments.n_seeds)
    flag_auc_table(inputs=inputs, n_seeds=arguments.n_seeds)
    print(f"PEAK MEMORY {c.peak_memory_gb():.2f} GB")


if __name__ == "__main__":
    main()

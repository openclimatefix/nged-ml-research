"""Experiment 709, stage 0: the detector and the classifiers scored on known and injected shifts.

Stage 0 runs on CPU before any neural network is fitted. It does five things:

1. **Tunes** (on 2021-11..2024-12, Primaries other than D8, before D8 or any scored month is
   examined) the detector's alarm threshold to at most 0.5 false alarms per Primary-year on clean
   history and on month-shuffled surrogates, the rule classifier's weights, the soft classifier,
   and the hidden-Markov models. Tuning worlds use their own seeds and settings. `--part tune`
   writes the frozen models to `stage0/models/` and `--load-models` reads them back.
2. **Scores the known cases:** D8's register connections and telemetry break (read from
   `era_table.parquet`, with a measured solar telemetry date and size and a penalised
   least-squares cross-check of the detector), the register's other connections, and the
   verified switching events of experiment 580 and the labelled set of experiment 41 (hindsight
   labels used only to score).
3. **Scores the injected bank** (plan section 9): sixteen passes per scored month, each with two era
   hosts and two switching hosts, and prints confusion tables at 7, 14, 28, and 60 days after the
   onset, the detection and classification delays, and the gates with binomial intervals.
4. **Prints an example** of one detected shift in the layout of plan section 5.
5. **Runs the perturbation test of the cut:** every reading at or after 00:00 of day `D`, at the
   Primary and its neighbours, is changed, and no signal at origin `D` may change (200 random
   (Primary, origin) pairs, every arm including the hidden-Markov arm).

**Scoring rule.** Every event is matched to one episode of each arm (the first episode of the arm
that starts within `ALARM_WINDOW` days of the onset). The state of an arm at a check day is
`none` before that episode starts (the event is undetected) and again after it has closed unless
the episode was classed `era` and not reversed. While the episode is open, the state is the
episode's own state. An era that was current before the onset appears in a separate column
(`pre_state`) and never counts as the detection of the event. An undetected event has an infinite
detection delay. The hidden-Markov arm is scored only on events whose onset is on or after its
training cut, because its fit read all earlier days.

Run with the folders of experiments 86, 660, 598, and 687 on `PYTHONPATH`; run `e00709_era_table.py`
first. Option `--part` selects `tune`, `real`, `injected`, `cut`, or `all`.
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final

import e00709_bocpd as bocpd
import e00709_classify as classify
import e00709_features as features
import e00709_inject as inject
import e00709_statdetect as statdetect
import e00709_stats as stats
import numpy as np
import polars as pl
from e00709_classify import STATE_NAMES, ArmType, Models, SiteResult, SiteSignal
from e00709_hmm import STATE_ERA

ARMS: Final[tuple[ArmType, ...]] = ("all", "rule", "soft", "hmm", "stat", "stat_noshape")
CHECK_DAYS: Final[tuple[int, ...]] = (7, 14, 28, 60)
TUNING_WINDOW: Final[tuple[date, date]] = (date(2021, 11, 1), date(2024, 12, 31))
HMM_TRAIN_END: Final[date] = date(2025, 1, 1)
SCORED_MONTHS: Final[tuple[date, ...]] = tuple(
    date(2025 + (m > 12), (m - 1) % 12 + 1, 1) for m in range(11, 19)
)
"""The first day of each month from 2025-11 to 2026-06."""
TUNING_PASSES: Final[int] = 80
"""Tuning passes of two era hosts and two switching hosts (the 320 events of 40 old passes)."""
MIN_TUNING_POOL: Final[int] = 8
"""Smallest host pool for which a tuning onset day is tried."""
TUNING_ONSET_TRIES: Final[int] = 30
"""Onset days drawn for a tuning pass before the pass is skipped as infeasible."""
SURROGATE_SEEDS: Final[tuple[int, ...]] = (11, 12, 13)
ALARM_WINDOW: Final[tuple[int, int]] = (-2, 90)
"""Days relative to the onset in which an episode's start counts as the detection of the event."""
EVENT_MARGIN_BEFORE: Final[int] = 3
EVENT_MARGIN_AFTER: Final[int] = 14
STAT_MARGIN_BEFORE: Final[int] = 35
"""Days before a labelled event that the statistics tuning mask starts (a window plus 7)."""
STAT_MARGIN_AFTER: Final[int] = 56
"""Days after a labelled event that the statistics tuning mask ends (the contamination span)."""
STAT_RETURN_CHECK_DAYS: Final[int] = 14
D8: Final[str] = "D8"
D8_RELIABLE_DAYS: Final[int] = 120
"""A measured telemetry date more than this many days from the register date is not used."""
D8_DETECTION_GATE_DAYS: Final[int] = 14
D8_SOLAR_GATE_DAYS: Final[int] = 30
"""The solar gate accepts a detection within this many days of the register date or of the
telemetry date."""
D8_GAS_WINDOW_ENDS: Final[tuple[date, ...]] = (
    date(2023, 8, 5), date(2023, 8, 19), date(2023, 9, 2), date(2023, 9, 30), date(2023, 10, 28),
)  # fmt: skip
D8_SOLAR_WINDOW_ENDS: Final[tuple[date, ...]] = (
    date(2025, 3, 1), date(2025, 3, 29), date(2025, 4, 26), date(2025, 5, 24), date(2025, 6, 21),
)  # fmt: skip
ENGINE_PASSES_PER_MONTH: Final[int] = 4
"""Passes per scored month of the irregular-engine world (two engines each, stage 0 only)."""
STAT_ONSET_AGES: Final[tuple[int, ...]] = (7, 14, 21, 28, 45, 60)
GATE_ERA_SHARE: Final[float] = 0.70
GATE_SWITCH_INJECTED: Final[float] = 0.10
GATE_SWITCH_VERIFIED: Final[float] = 0.15
GATE_DETECTED_SHARE: Final[float] = 0.50
DETECTION_LIMITS: Final[dict[str, int]] = {"level_m25": 7, "level_m10": 21}
ENGINE_WITHIN_DAYS: Final[int] = 28
"""Days within which an irregular-engine era should be alarmed (4 times the level_m25 limit)."""
"""Days within which the share of an era kind's events must be detected (item 5)."""
CUT_PAIRS: Final[int] = 200
CUT_SEED: Final[int] = 709_999
STAGE0_DIR: Final[Path] = features.DATA_DIR / "stage0"
MODELS_DIR: Final[Path] = STAGE0_DIR / "models"
ERA_TABLE_PATH: Final[Path] = features.DATA_DIR / "era_table.parquet"
SEGMENT_MIN_DAYS: Final[int] = 28
SEGMENT_PENALTY: Final[float] = 3.0
"""Penalty of the least-squares cross-check, in multiples of noise variance times log(days)."""
SOLAR_WINDOW_MONTHS: Final[tuple[int, int]] = (4, 8)
FOLD_SHARE: Final[float] = 0.05
"""A midday reading below this share of capacity may be a reverse flow folded by `abs`."""
MIDDAY_HALF_HOURS: Final[slice] = slice(22, 32)
NIGHT_HALF_HOURS: Final[slice] = slice(0, 12)


@dataclass
class Context:
    """The real panel and everything derived from it once.

    Attributes:
        panel: The real panel.
        feats: Features of every site.
        weather: ERA5 summaries.
        g4_threshold: Gate G4's threshold of every site.
        g4_real: Gate G4 on the real panel, shape (sites, days).
        verified_days: Durations of the verified switching events outside the scored window.
        events: The known switching events table.
        era_table: The era table that `e00709_era_table.py` wrote.
        ready_days: First day with features of every site.
        stat_windows: The window statistics of every site on the real panel.
    """

    panel: features.Panel
    feats: list[features.SiteFeatures]
    weather: features.DailyWeather
    g4_threshold: np.ndarray
    g4_real: np.ndarray
    verified_days: np.ndarray
    events: pl.DataFrame
    era_table: pl.DataFrame
    ready_days: np.ndarray
    stat_windows: list[stats.WindowStats]


@dataclass(frozen=True)
class SolarEstimate:
    """D8's solar farm as measured in the telemetry.

    Attributes:
        day: Day index of the first day after the estimated step.
        size_mw: Size of the clear-day midday reduction in megawatts (positive).
    """

    day: int
    size_mw: float


def wilson(*, k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval of a binomial share.

    Args:
        k: Successes.
        n: Trials.
        z: Normal quantile of the interval.

    Returns:
        The lower and upper ends, NaN if there are no trials.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def share_text(*, k: int, n: int) -> str:
    """Return `k/n = share [low, high]` with the Wilson 95% interval."""
    lo, hi = wilson(k=k, n=n)
    return f"{k}/{n} = {k / max(n, 1):.2f} [{lo:.2f}, {hi:.2f}]"


def build_context() -> Context:
    """Load the real panel, features, weather, gate G4, and the known events and eras.

    Returns:
        The context. `e00709_era_table.py` must have written its two tables.
    """
    panel = features.load_panel()
    feats = features.compute_panel_features(panel=panel)
    events = pl.read_parquet(features.DATA_DIR / "known_events.parquet")
    verified = events.filter(
        (pl.col("source") == "experiment 580 verified")
        & pl.col("event_id").str.ends_with("a")
        & ~pl.col("in_scored_window")
    )
    threshold = classify.g4_thresholds(site_features=feats)
    return Context(
        panel=panel,
        feats=feats,
        weather=features.load_weather(panel=panel),
        g4_threshold=threshold,
        g4_real=classify.g4_on(
            daily=np.stack([f.daily_mean for f in feats]),
            threshold=threshold,
            ready_days=np.array([f.ready_day for f in feats]),
            neighbours=panel.neighbours,
        ),
        verified_days=verified["duration_days"].to_numpy(),
        events=events,
        era_table=pl.read_parquet(ERA_TABLE_PATH),
        ready_days=np.array([f.ready_day for f in feats]),
        stat_windows=[stats.window_statistics(power_site=row) for row in panel.power],
    )


def _event_mask(
    *, ctx: Context, site: str, before: int = EVENT_MARGIN_BEFORE, after: int = EVENT_MARGIN_AFTER
) -> np.ndarray:
    """Return the days inside a known event of the site, widened by margins.

    Args:
        ctx: The context.
        site: The site name.
        before: Days before the event onset that the mask starts.
        after: Days after the event end that the mask ends.

    Returns:
        A boolean mask over the panel's days.
    """
    mask = np.zeros(ctx.panel.n_days, dtype=bool)
    for row in ctx.events.filter(pl.col("site") == site).iter_rows(named=True):
        first = features.day_index(day=row["onset_utc"].date()) - before
        last = features.day_index(day=row["end_utc"].date()) + after
        mask[max(first, 0) : last + 1] = True
    return mask


def _report_threshold(*, label: str, choice: float, rates: dict[float, float]) -> None:
    """Print a tuned threshold, and a gate failure if no threshold on the grid met the target."""
    print(f"{label}: threshold {choice}, false alarms per Primary-year {rates}")
    if rates[choice] > bocpd.FALSE_ALARM_TARGET:
        print(
            f"GATE FAILURE: no threshold on the grid reaches {bocpd.FALSE_ALARM_TARGET} false "
            f"alarms per Primary-year on {label}; the fallback {choice} gives {rates[choice]:.2f}"
        )


def tune_detector(*, ctx: Context) -> float:
    """Tune the alarm threshold on clean real history and on month-shuffled surrogates.

    The threshold is the largest of the tuned values, so that it holds on all of them. A surrogate
    swaps each day with a random day of the same month of the year, which keeps the seasonal
    pattern and removes every level shift, so every alarm on it is false.

    Args:
        ctx: The context.

    Returns:
        The chosen threshold.
    """
    window = tuple(features.day_index(day=d) for d in TUNING_WINDOW)
    sites = [i for i, s in enumerate(ctx.panel.sites) if s != D8 and ctx.ready_days[i] < window[1]]
    real_series = [
        bocpd.run_posterior(scaled=ctx.feats[i].scaled, start_day=ctx.feats[i].ready_day)
        for i in sites
    ]
    excluded = [_event_mask(ctx=ctx, site=ctx.panel.sites[i]) for i in sites]
    scaled = [ctx.feats[i].scaled for i in sites]
    spread = [ctx.feats[i].spread for i in sites]
    starts = [ctx.feats[i].ready_day for i in sites]
    real_choice, real_rates = bocpd.tune_threshold(
        series=real_series, scaled=scaled, spread=spread, start_days=starts, excluded=excluded,
        window=(window[0], window[1]),
    )  # fmt: skip
    _report_threshold(label="real clean history", choice=real_choice, rates=real_rates)
    month_of_year = np.array([features.day_of(index=i).month for i in range(ctx.panel.n_days)])
    choices = [real_choice]
    for seed in SURROGATE_SEEDS:
        shuffled = [
            bocpd.month_shuffle(scaled=x, start_day=s, month_of_year=month_of_year, seed=seed + i)
            for i, (x, s) in enumerate(zip(scaled, starts, strict=True))
        ]
        series = [
            bocpd.run_posterior(scaled=x, start_day=s)
            for x, s in zip(shuffled, starts, strict=True)
        ]
        no_mask = [np.zeros(ctx.panel.n_days, dtype=bool)] * len(sites)
        choice, rates = bocpd.tune_threshold(
            series=series, scaled=shuffled, spread=spread, start_days=starts, excluded=no_mask,
            window=(window[0], window[1]),
        )  # fmt: skip
        _report_threshold(label=f"surrogate seed {seed}", choice=choice, rates=rates)
        choices.append(choice)
    return max(choices)


def tune_stat_detector(*, ctx: Context) -> float:
    """Tune the statistics detector's threshold on clean real history, without D8.

    A 28-day change reacts to a step for 28 days before it and 56 days after it, so the mask of
    days that do not count as false alarms starts 35 days before a labelled event and ends 56 days
    after it. The false-alarm rates with the detector's 3-day and 14-day margins are printed too.
    The window is 2021-11 to 2024-12. The function also prints the share of the statistics
    detector's episodes that close by a return within 14 days of the alarm.

    Args:
        ctx: The context.

    Returns:
        The chosen threshold.
    """
    window = tuple(features.day_index(day=d) for d in TUNING_WINDOW)
    sites = [i for i, s in enumerate(ctx.panel.sites) if s != D8 and ctx.ready_days[i] < window[1]]
    windows = [ctx.stat_windows[i] for i in sites]
    scaled = [ctx.feats[i].scaled for i in sites]
    _, old_rates = statdetect.tune_stat_threshold(
        windows=windows, scaled=scaled, window=(window[0], window[1]),
        excluded=[_event_mask(ctx=ctx, site=ctx.panel.sites[i]) for i in sites],
    )  # fmt: skip
    print(f"statistics detector, detector margins (3/14 days), per Primary-year: {old_rates}")
    choice, rates = statdetect.tune_stat_threshold(
        windows=windows, scaled=scaled, window=(window[0], window[1]),
        excluded=[
            _event_mask(
                ctx=ctx, site=ctx.panel.sites[i], before=STAT_MARGIN_BEFORE, after=STAT_MARGIN_AFTER
            )
            for i in sites
        ],
    )  # fmt: skip
    print(f"statistics detector: threshold {choice}, false alarms per Primary-year {rates}")
    n_episodes = n_returned = 0
    for w, x in zip(windows, scaled, strict=True):
        episodes = statdetect.detect_stat(windows=w, scaled=x, threshold=choice).detector.episodes
        n_episodes += len(episodes)
        n_returned += sum(
            e.ended_by == "return" and e.end_day - e.alarm_day <= STAT_RETURN_CHECK_DAYS
            for e in episodes
        )
    print(
        f"statistics episodes closed by a return within {STAT_RETURN_CHECK_DAYS} days of the "
        f"alarm: {share_text(k=n_returned, n=n_episodes)}"
    )
    if rates[choice] > bocpd.FALSE_ALARM_TARGET:
        print(
            f"GATE FAILURE: no statistics threshold on the grid reaches {bocpd.FALSE_ALARM_TARGET} "
            f"false alarms per Primary-year; the fallback {choice} gives {rates[choice]:.2f}"
        )
    return choice


def _features_of_pass(
    *, ctx: Context, injected: inject.InjectedPass
) -> list[features.SiteFeatures]:
    """Return the sites' features with the injected sites recomputed."""
    touched = {e.host for e in injected.eras} | {s.donor for s in injected.switches}
    touched |= {s.recipient for s in injected.switches if s.recipient is not None}
    out = list(ctx.feats)
    for site in touched:
        out[site] = features.compute_site_features(power_site=injected.power[site])
    return out


def process_hosts(
    *, ctx: Context, models: Models, injected: inject.InjectedPass, arms: Sequence[ArmType]
) -> dict[int, SiteResult]:
    """Run the pipeline for the hosts of an injected pass.

    A donor of the `hidden_mirror` kind has no metered recipient, so its neighbour list is empty
    for the mirror evidence, the weather evidence, and gate G4, which makes `no_neighbour` equal
    to 1 as it is for a real Primary whose partner is unmetered.

    Args:
        ctx: The context.
        models: The frozen models.
        injected: The pass.
        arms: Arms to compute.

    Returns:
        The result of each host (era hosts and switching donors), by site index.
    """
    feats = _features_of_pass(ctx=ctx, injected=injected)
    grid = classify.build_mirror_grid(
        site_features=feats, neighbours=ctx.panel.neighbours, reference_features=ctx.feats
    )
    daily = np.stack([f.daily_mean for f in feats])
    ready = np.array([f.ready_day for f in feats])
    hidden = {s.donor for s in injected.switches if s.recipient is None}
    results: dict[int, SiteResult] = {}
    for host in [e.host for e in injected.eras] + [s.donor for s in injected.switches]:
        neighbours = {**ctx.panel.neighbours, host: []} if host in hidden else ctx.panel.neighbours
        g4 = classify.g4_on(
            daily=daily,
            threshold=ctx.g4_threshold,
            ready_days=ready,
            neighbours=neighbours,
            sites=[host],
        )
        results[host] = classify.process_site(
            site=host, all_features=feats,
            mirror=classify.mirror_features(grid=grid, site=host, neighbours=neighbours),
            g4=g4[host], neighbours=neighbours, weather=ctx.weather, models=models, arms=arms,
            power=injected.power[host] if ("stat" in arms or "stat_noshape" in arms) else None,
        )  # fmt: skip
    return results


def _evidence_rows(
    *, result: SiteResult, truth: dict, onset_day: int, stat: bool = False
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the evidence rows of the episode that detects a tuning event, at ages 7 to 60 days.

    Args:
        result: The host's pipeline result.
        truth: The event's truth row.
        onset_day: Day of the onset.
        stat: Read the statistics detector's episodes, and append its two shape columns.

    Returns:
        The rows and their labels (1 for an era), or None if no episode detects the event.
    """
    detector = result.stat.detector if stat and result.stat is not None else result.detector
    evidence_of = result.stat_evidence if stat else result.evidence
    episodes = [
        e
        for e in detector.episodes
        if ALARM_WINDOW[0] <= e.alarm_day - onset_day <= ALARM_WINDOW[1]
    ]
    if not episodes:
        return None
    evidence = evidence_of[episodes[0].episode]
    shape = evidence.shape if stat else None
    pick = [np.flatnonzero(evidence.days == onset_day + a) for a in STAT_ONSET_AGES]
    rows = [
        np.concatenate([evidence.x[p[0]], shape[p[0]]]) if shape is not None else evidence.x[p[0]]
        for p in pick
        if len(p) and (shape is not None or not stat)
    ]
    if not rows:
        return None
    return np.stack(rows), np.full(len(rows), 1 if truth["truth"] == "era" else 0)


def tune_classifiers(*, ctx: Context, models: Models) -> None:
    """Fit the rule weights and the soft classifier on injected tuning worlds.

    Args:
        ctx: The context.
        models: The models; `rule_weights` and `soft` are set in place.
    """
    rng = np.random.default_rng(inject.world_seed(world="tuning", index=99_999))
    window = tuple(features.day_index(day=d) for d in TUNING_WINDOW)
    xs, ys, groups = [], [], []
    stat_xs, stat_ys = [], []
    for index in range(TUNING_PASSES):
        injected = None
        for _ in range(TUNING_ONSET_TRIES):
            onset_day = int(rng.integers(window[0] + 60, window[1] - 200))
            if (
                len(
                    inject.host_pool(
                        panel=ctx.panel, ready_days=ctx.ready_days, onset_day=onset_day
                    )
                )
                < MIN_TUNING_POOL
            ):
                continue
            try:
                injected = inject.build_pass(
                    panel=ctx.panel, weather_shortwave=ctx.weather.shortwave,
                    ready_days=ctx.ready_days, verified_days=ctx.verified_days, world="tuning",
                    pass_index=index, onset_day=onset_day,
                )  # fmt: skip
            except RuntimeError:
                continue
            break
        if injected is None:
            print(f"tuning pass {index}: no feasible onset day in {TUNING_ONSET_TRIES} draws")
            continue
        results = process_hosts(ctx=ctx, models=models, injected=injected, arms=("stat",))
        for truth in injected.truth.iter_rows(named=True):
            result = results[truth["host"]]
            rows = _evidence_rows(result=result, truth=truth, onset_day=truth["onset_day"])
            if rows is not None:
                xs.append(rows[0])
                ys.append(rows[1])
                groups.append(np.full(len(rows[1]), index))
            stat_rows = _evidence_rows(
                result=result, truth=truth, onset_day=truth["onset_day"], stat=True
            )
            if stat_rows is not None:
                stat_xs.append(stat_rows[0])
                stat_ys.append(stat_rows[1])
    x, y, group = np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)
    print(f"tuning rows: {len(y)} ({int(y.sum())} era, {int((1 - y).sum())} switch)")
    models.rule_weights = classify.fit_logistic(x=x[:, : classify.N_RULE_COLUMNS], y=y)
    print(
        "rule weights:",
        dict(
            zip(
                classify.EVIDENCE_NAMES[: classify.N_RULE_COLUMNS],
                models.rule_weights.round(3),
                strict=True,
            )
        ),
    )
    models.soft = classify.train_soft(x=x, y=y, groups=group, seed=709)
    tune_stat_weights(models=models, xs=stat_xs, ys=stat_ys)


def tune_stat_weights(*, models: Models, xs: list[np.ndarray], ys: list[np.ndarray]) -> None:
    """Fit the `stat` arm's nine weights, and print the fit without the two shape columns.

    The ablation costs no extra fit of a network: it is the logistic fit on the seven rule columns
    of the same rows, and the printed figure is each fit's class-balanced training accuracy.

    Args:
        models: The models; `stat_weights` is set in place.
        xs: Evidence rows of the statistics detector's episodes (13 columns and 2 shape columns).
        ys: The labels (1 for an era).
    """
    if not xs:
        print("statistics tuning rows: none; the stat weights stay at zero")
        return
    x, y = np.concatenate(xs), np.concatenate(ys)
    columns = [*range(classify.N_RULE_COLUMNS), *range(classify.N_EVIDENCE, x.shape[1])]
    with_shape = classify.fit_logistic(x=x[:, columns], y=y)
    without = classify.fit_logistic(x=x[:, : classify.N_RULE_COLUMNS], y=y)
    models.stat_weights = with_shape
    models.stat_noshape_weights = np.concatenate([without, np.zeros(len(columns) - len(without))])

    def balanced(*, weights: np.ndarray, cols: list[int]) -> float:
        predicted = (x[:, cols] @ weights) > 0
        return 0.5 * (float(predicted[y == 1].mean()) + float((~predicted)[y == 0].mean()))

    print(
        f"statistics tuning rows: {len(y)} ({int(y.sum())} era, {int((1 - y).sum())} switch); "
        f"class-balanced training accuracy with the shape columns "
        f"{balanced(weights=with_shape, cols=columns):.3f}, without "
        f"{balanced(weights=without, cols=columns[: classify.N_RULE_COLUMNS]):.3f}"
    )
    names = [*classify.EVIDENCE_NAMES[: classify.N_RULE_COLUMNS], *classify.SHAPE_NAMES]
    print("stat weights:", dict(zip(names, with_shape.round(3), strict=True)))


def tune(*, ctx: Context) -> Models:
    """Tune the detector, the classifiers, and the hidden-Markov models, and save them.

    The hidden-Markov models are fitted per Primary on days before `HMM_TRAIN_END`, and D8's model
    is fitted on D8's own days before that date.

    Args:
        ctx: The context.

    Returns:
        The frozen models.
    """
    train_end = features.day_index(day=HMM_TRAIN_END)
    models = Models(
        threshold=tune_detector(ctx=ctx),
        rule_weights=np.zeros(classify.N_RULE_COLUMNS),
        survival=classify.SwitchSurvival(durations=ctx.verified_days),
        train_end_day=train_end,
        hmm_parameters=classify.fit_hmm_parameters(all_features=ctx.feats, train_end_day=train_end),
        stat_threshold=tune_stat_detector(ctx=ctx),
    )
    print(f"alarm threshold {models.threshold}; verified durations n={len(ctx.verified_days)}")
    tune_classifiers(ctx=ctx, models=models)
    classify.save_models(models=models, folder=MODELS_DIR)
    print(f"frozen models written to {MODELS_DIR}")
    return models


def real_results(*, ctx: Context, models: Models) -> dict[int, SiteResult]:
    """Run the pipeline for every site on the real panel.

    Args:
        ctx: The context.
        models: The frozen models.

    Returns:
        The result of each site, by site index.
    """
    grid = classify.build_mirror_grid(site_features=ctx.feats, neighbours=ctx.panel.neighbours)
    return {
        site: classify.process_site(
            site=site, all_features=ctx.feats,
            mirror=classify.mirror_features(grid=grid, site=site), g4=ctx.g4_real[site],
            neighbours=ctx.panel.neighbours, weather=ctx.weather, models=models,
            power=ctx.panel.power[site], windows=ctx.stat_windows[site],
        )
        for site in range(len(ctx.panel.sites))
    }  # fmt: skip


def _matched_episode(*, signal: SiteSignal, onset_day: int) -> int:
    """Return the first episode of an arm that starts within `ALARM_WINDOW` of the onset, or -1."""
    for k, (start, _end) in enumerate(signal.episode_bounds):
        if ALARM_WINDOW[0] <= start - onset_day <= ALARM_WINDOW[1]:
            return k
    return -1


def _matched_state(*, signal: SiteSignal, episode: int, n_days: int) -> np.ndarray:
    """Return the state of one matched episode on every day (item 1 of the review).

    The state is `none` before the episode starts, the episode's own state while it is open, `era`
    after it has closed if it was classed `era` and has not been reversed, and `none` otherwise.
    """
    if episode < 0:
        return np.zeros(n_days, dtype=np.int8)
    days = np.arange(n_days)
    after = np.where(signal.persistent_episode == episode, STATE_ERA, 0)
    own = np.where(signal.episode_id == episode, signal.state, after)
    return np.where(days < signal.episode_bounds[episode][0], 0, own).astype(np.int8)


def score_event(
    *,
    result: SiteResult,
    onset_day: int,
    end_day: int,
    truth: str,
    n_days: int,
    train_end_day: int,
) -> dict[str, object]:
    """Score one event on one Primary, with each arm matched to one episode of its own.

    Args:
        result: The Primary's pipeline result.
        onset_day: Day of the event's onset.
        end_day: Last day of a switching event (-1 for an era).
        truth: `era` or `switch`.
        n_days: Number of days on the grid.
        train_end_day: First day the hidden-Markov fit did not read; the arm is not scored on an
            earlier onset.

    Returns:
        The detector's alarm day and detection delay, and per arm the detection delay (None if the
        arm never started an episode in the window, which counts as an infinite delay), the state
        of the matched episode at each check day (-1 beyond the grid), the state of the Primary
        the day before the onset (`pre_state`), the classification delay after the arm's detection,
        and whether and for how many days a switching event was carried as an era from its onset to
        7 days after its end.
    """
    episodes = [
        e
        for e in result.detector.episodes
        if ALARM_WINDOW[0] <= e.alarm_day - onset_day <= ALARM_WINDOW[1]
    ]
    row: dict[str, object] = {
        "alarm_day": episodes[0].alarm_day if episodes else -1,
        "detection_delay": episodes[0].alarm_day - onset_day if episodes else None,
    }
    last = min(end_day + 7, n_days - 1) if truth == "switch" else n_days - 1
    for arm, signal in result.signals.items():
        if arm == "hmm" and onset_day < train_end_day:
            continue
        if arm in ("stat", "stat_noshape") and (
            result.stat is None or onset_day < result.stat.start_day
        ):
            continue
        episode = _matched_episode(signal=signal, onset_day=onset_day)
        matched = _matched_state(signal=signal, episode=episode, n_days=n_days)
        start = signal.episode_bounds[episode][0] if episode >= 0 else -1
        row[f"{arm}_detection_delay"] = start - onset_day if episode >= 0 else None
        row[f"{arm}_pre_state"] = int(signal.state[onset_day - 1]) if onset_day >= 1 else 0
        for k in CHECK_DAYS:
            index = onset_day + k - 1
            row[f"{arm}_{k}"] = int(matched[index]) if index < n_days else -1
        decided = np.flatnonzero(np.isin(matched[start:], (2, 3))) if episode >= 0 else []
        row[f"{arm}_class_delay"] = int(decided[0]) if len(decided) else None
        carried = matched[onset_day : last + 1] == STATE_ERA
        row[f"{arm}_ever_era"] = bool(carried.any())
        row[f"{arm}_days_carried"] = int(carried.sum())
    return row


def injected_events(
    *,
    ctx: Context,
    models: Models,
    passes_per_month: int,
    world: inject.WorldNameType = "bank",
) -> pl.DataFrame:
    """Score an injected world: sixteen passes per scored month, two eras and two switches each.

    Args:
        ctx: The context.
        models: The frozen models.
        passes_per_month: Passes per scored month.
        world: `bank`, or `engine` for the irregular-engine eras of stage 0.

    Returns:
        One row per injected event with the scores of `score_event`.
    """
    rows = []
    for month_index, month in enumerate(SCORED_MONTHS):
        onset_day = features.day_index(day=month)
        for j in range(passes_per_month):
            pass_index = month_index * passes_per_month + j
            injected = inject.build_pass(
                panel=ctx.panel, weather_shortwave=ctx.weather.shortwave,
                ready_days=ctx.ready_days, verified_days=ctx.verified_days, world=world,
                pass_index=pass_index, onset_day=onset_day,
            )  # fmt: skip
            results = process_hosts(ctx=ctx, models=models, injected=injected, arms=ARMS)
            for truth in injected.truth.iter_rows(named=True):
                score = score_event(
                    result=results[truth["host"]], onset_day=truth["onset_day"],
                    end_day=truth["end_day"], truth=truth["truth"], n_days=ctx.panel.n_days,
                    train_end_day=models.train_end_day,
                )  # fmt: skip
                rows.append({**truth, "month": str(month), "pass_index": pass_index, **score})
    return pl.DataFrame(rows, infer_schema_length=None)


def confusion_tables(*, events: pl.DataFrame) -> pl.DataFrame:
    """Return the confusion counts at each check day: era, switch, unknown, and none.

    The state `none` counts the events that the arm has not detected, or whose episode has closed
    without an `era` class.

    Args:
        events: The output of `injected_events` or `known_events_scores`.

    Returns:
        One row per arm, check day, and truth, with a count column per state.
    """
    pieces = []
    for arm in ARMS:
        for k in CHECK_DAYS:
            column = f"{arm}_{k}"
            if column not in events.columns:
                continue
            counts = (
                events.filter(pl.col(column) >= 0)
                .with_columns(
                    state=pl.col(column).replace_strict(
                        dict(enumerate(STATE_NAMES)), return_dtype=pl.String
                    )
                )
                .group_by("truth", "state")
                .len()
                .pivot(on="state", index="truth", values="len")
                .fill_null(0)
            )
            for name in STATE_NAMES:
                if name not in counts.columns:
                    counts = counts.with_columns(pl.lit(0, dtype=pl.UInt32).alias(name))
            pieces.append(
                counts.with_columns(arm=pl.lit(arm), days_after_onset=pl.lit(k)).select(
                    "arm", "days_after_onset", "truth", *STATE_NAMES
                )
            )
    return pl.concat(pieces).sort("arm", "days_after_onset", "truth")


def _quantiles(*, values: pl.Series) -> dict[str, float]:
    """Return the count, quartiles, and maximum of the non-null values."""
    array = values.drop_nulls().to_numpy().astype(float)
    if array.size == 0:
        return {"n": 0.0}
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {"n": float(array.size), "p25": q25, "median": median, "p75": q75, "max": array.max()}


def _detection_line(*, delays: pl.Series, limit: int) -> tuple[str, bool]:
    """Return the printout and the gate verdict for the share detected within `limit` days.

    A null delay is an event the arm never detected, which counts as infinite.
    """
    n = delays.len()
    alarmed = delays.drop_nulls().len()
    within = int((delays.drop_nulls() <= limit).sum())
    ok = n > 0 and within / n >= GATE_DETECTED_SHARE
    text = (
        f"alarmed {alarmed}/{n}; detected within {limit} days {share_text(k=within, n=n)} "
        f"(gate at least {GATE_DETECTED_SHARE}): {ok}"
    )
    return text, ok


def injected_report(*, events: pl.DataFrame) -> None:
    """Print the confusion tables, delays, duration bands, and the gates of the injected bank.

    Args:
        events: The scored injected events.
    """
    STAGE0_DIR.mkdir(parents=True, exist_ok=True)
    events.write_parquet(STAGE0_DIR / "injected_events.parquet")
    eras = events.filter(pl.col("truth") == "era")
    switches = events.filter(pl.col("truth") == "switch")
    with pl.Config(tbl_rows=100, tbl_cols=12):
        table = confusion_tables(events=events)
        table.write_csv(STAGE0_DIR / "injected_confusion.csv")
        print("Confusion tables of the injected bank (counts of events by state):")
        print(table)
        print("Detection delay in days by era kind (null delay = never detected):")
        print(
            eras.group_by("kind")
            .agg(
                n=pl.len(), alarmed=pl.col("detection_delay").is_not_null().sum(),
                median_of_alarmed=pl.col("detection_delay").median(),
                p75_of_alarmed=pl.col("detection_delay").quantile(0.75),
            )
            .sort("kind")
        )  # fmt: skip
        for arm in ARMS:
            if f"{arm}_class_delay" in events.columns:
                delays = events.filter(pl.col(f"{arm}_detection_delay").is_not_null())
                print(
                    f"classification delay (days after the detection), arm {arm}:",
                    _quantiles(values=delays[f"{arm}_class_delay"]),
                )
    print("Share of switching events ever classed era, by arm and duration band (Wilson 95%):")
    for arm in ARMS:
        for band in ("<60", "60-120", ">120", "all"):
            part = switches if band == "all" else switches.filter(pl.col("band") == band)
            k = int(part[f"{arm}_ever_era"].sum()) if part.height else 0
            carried = (
                part.filter(pl.col(f"{arm}_ever_era"))[f"{arm}_days_carried"].median()
                if k
                else None
            )
            print(
                f"  {arm:5s} {band:7s} ever era {share_text(k=k, n=part.height)}"
                f" median days carried {carried}"
            )
    print("Stage 0 gates of the injected bank:")
    for arm in ARMS:
        print(f"  arm {arm}:")
        for kind, limit in DETECTION_LIMITS.items():
            delays = eras.filter(pl.col("kind") == kind)[f"{arm}_detection_delay"]
            text, _ = _detection_line(delays=delays, limit=limit)
            print(f"    detection, {kind}: {text}")
        for kind in ("level_m25", "solar_25", "level_m10", "level_p15"):
            part = eras.filter(pl.col("kind") == kind)
            k28 = int((part[f"{arm}_28"] == STATE_ERA).sum())
            gated = (
                f" (gate at least {GATE_ERA_SHARE}): {k28 / max(part.height, 1) >= GATE_ERA_SHARE}"
            )
            print(
                f"    {kind} classed era by day 28: {share_text(k=k28, n=part.height)}"
                f"{gated if kind == 'level_m25' else ' (reported, not gated)'}"
            )
        ever = int(switches[f"{arm}_ever_era"].sum())
        ok = switches.height > 0 and ever / switches.height <= GATE_SWITCH_INJECTED
        print(
            f"    switching events ever era: {share_text(k=ever, n=switches.height)}"
            f" (gate at most {GATE_SWITCH_INJECTED}): {ok}"
        )


def engine_report(*, events: pl.DataFrame) -> None:
    """Print how each arm handles the irregular-engine eras (stage 0 only, not in the bank).

    Args:
        events: The scored events of the `engine` world.
    """
    eras = events.filter(pl.col("truth") == "era")
    print(f"Irregular-engine eras (-30% of capacity, runs on half the days), n={eras.height}:")
    for arm in ARMS:
        if f"{arm}_detection_delay" not in eras.columns:
            continue
        delays = eras[f"{arm}_detection_delay"]
        within = int((delays.drop_nulls() <= ENGINE_WITHIN_DAYS).sum())
        k28 = int((eras[f"{arm}_28"] == STATE_ERA).sum())
        print(
            f"  arm {arm}: alarmed {delays.drop_nulls().len()}/{eras.height}; within 28 days "
            f"{share_text(k=within, n=eras.height)}; classed era by day 28 "
            f"{share_text(k=k28, n=eras.height)}; median delay of alarmed "
            f"{delays.drop_nulls().median()}"
        )


def _within(*, value: object, limit: int) -> bool:
    """Return True if a detection delay is an integer of at most `limit` days."""
    return isinstance(value, int) and value <= limit


def _state_name(value: object) -> str:
    """Return the name of a state code that a score row holds."""
    return STATE_NAMES[value] if isinstance(value, int) else "unscored"


def _day(*, when: datetime) -> int:
    return features.day_index(day=when.date())


def known_events_scores(
    *, ctx: Context, results: dict[int, SiteResult], train_end_day: int
) -> pl.DataFrame:
    """Score the verified and the labelled switching events on the real panel.

    Args:
        ctx: The context.
        results: The real-panel results.
        train_end_day: First day the hidden-Markov fit did not read.

    Returns:
        One row per event and side that the detector could read (ready by the onset).
    """
    rows = []
    for event in ctx.events.iter_rows(named=True):
        site = ctx.panel.sites.index(event["site"]) if event["site"] in ctx.panel.sites else -1
        onset, end = _day(when=event["onset_utc"]), _day(when=event["end_utc"])
        if site < 0 or onset < ctx.ready_days[site] or onset >= ctx.panel.n_days:
            continue
        score = score_event(
            result=results[site], onset_day=onset, end_day=end, truth="switch",
            n_days=ctx.panel.n_days, train_end_day=train_end_day,
        )  # fmt: skip
        rows.append(
            {
                **event,
                "truth": "switch",
                "band": inject.duration_band(days=event["duration_days"]),
                "onset_day": onset,
                **score,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def verified_gate(*, scored: pl.DataFrame) -> None:
    """Print the verified-set gate per transfer, with Wilson intervals.

    A transfer counts as carried as an era if either of its sides was ever classed `era`. The 23
    transfers whose onset is outside the scored window carry the gate; the transfers inside it are
    reported apart because there are too few of them to gate.

    Args:
        scored: The output of `known_events_scores`.
    """
    verified = scored.filter(pl.col("source") == "experiment 580 verified").with_columns(
        transfer=pl.col("event_id").str.replace(r"[ab]$", "")
    )
    for arm in ARMS:
        column = f"{arm}_ever_era"
        if column not in verified.columns:
            continue
        per_transfer = (
            verified.filter(pl.col(column).is_not_null())
            .group_by("transfer")
            .agg(pl.col(column).any(), pl.col("in_scored_window").first())
        )
        for label, part in (
            ("outside the scored window", per_transfer.filter(~pl.col("in_scored_window"))),
            ("inside the scored window (reported apart)", per_transfer.filter("in_scored_window")),
        ):
            k = int(part[column].sum())
            gate = ""
            if label.startswith("outside"):
                ok = k / max(part.height, 1) <= GATE_SWITCH_VERIFIED
                gate = f" (gate at most {GATE_SWITCH_VERIFIED}): {ok}"
            print(f"  arm {arm}: transfers {label} ever era {share_text(k=k, n=part.height)}{gate}")


def g4_report(*, ctx: Context, scored: pl.DataFrame) -> None:
    """Print the ported gate G4's false triggers per Primary-year and its onset delay.

    A false trigger is a rise of the gate outside every known event (widened by the detector's
    margins). The onset delay is the first day of the gate inside the event window, on the
    verified events, in days after the onset.

    Args:
        ctx: The context.
        scored: The output of `known_events_scores`.
    """
    triggers, years = 0, 0.0
    for site, name in enumerate(ctx.panel.sites):
        if name == D8 or ctx.ready_days[site] >= ctx.panel.n_days:
            continue
        clear = ~_event_mask(ctx=ctx, site=name)
        on = ctx.g4_real[site]
        rising = on & ~np.concatenate([[False], on[:-1]])
        days = np.arange(ctx.panel.n_days) >= ctx.ready_days[site]
        triggers += int((rising & clear & days).sum())
        years += float((clear & days).sum()) / 365.25
    print(
        f"gate G4 (ported): {triggers} false triggers in {years:.1f} Primary-years clear of events"
    )
    print(f"  = {triggers / max(years, 1e-9):.2f} per Primary-year")
    delays = []
    verified = scored.filter(pl.col("source") == "experiment 580 verified")
    for row in verified.iter_rows(named=True):
        site = ctx.panel.sites.index(row["site"])
        first = row["onset_day"] - EVENT_MARGIN_BEFORE
        window = ctx.g4_real[site, first : row["onset_day"] + ALARM_WINDOW[1] + 1]
        hit = np.flatnonzero(window)
        delays.append(int(hit[0]) - EVENT_MARGIN_BEFORE if len(hit) else None)
    found = [d for d in delays if d is not None]
    median = float(np.median(found)) if found else None
    print(
        f"gate G4 on verified events: triggered {len(found)}/{len(delays)} sides, median onset "
        f"delay {median} days"
    )


def known_report(*, ctx: Context, scored: pl.DataFrame) -> None:
    """Print the known switching events' scores, the verified-set gate, and the G4 figures.

    Args:
        ctx: The context.
        scored: The output of `known_events_scores`.
    """
    for source in ("experiment 580 verified", "experiment 41 labelled set"):
        part = scored.filter(pl.col("source") == source) if scored.height else scored
        print(f"{source}: n={part.height} events read by the detector (each side counted)")
        if part.height == 0:
            continue
        print(confusion_tables(events=part))
        if source == "experiment 41 labelled set":
            for arm in ARMS:
                column = f"{arm}_ever_era"
                if column in part.columns:
                    scored_part = part.filter(pl.col(column).is_not_null())
                    ever = int(scored_part[column].sum())
                    print(
                        f"  arm {arm}: ever classed era "
                        f"{share_text(k=ever, n=scored_part.height)} (reported, not gated)"
                    )
    print("Verified switching events, per transfer (both sides read):")
    verified_gate(scored=scored)
    g4_report(ctx=ctx, scored=scored)
    print(
        "The verified events were selected for a mirrored shift, so the mirror evidence is circular"
        " on that set and the score is optimistic; experiment 41's labels are partly unconfirmed."
    )


def _clear_midday_power(*, ctx: Context, site: int, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Return, for the clear days of April to August of a year, the midday and night levels.

    Levels are in megawatts. A day is clear if its ERA5 midday shortwave is in the top 40% of the
    window's days.
    """
    first = features.day_index(day=date(year, SOLAR_WINDOW_MONTHS[0], 1))
    last = features.day_index(day=date(year, SOLAR_WINDOW_MONTHS[1], 28))
    grid = ctx.panel.power[site].reshape(ctx.panel.n_days, features.HALF_HOURS_PER_DAY)
    capacity = ctx.panel.capacity_mw[site]
    midday = np.nanmean(grid[first : last + 1, MIDDAY_HALF_HOURS], axis=1) * capacity
    night = np.nanmean(grid[first : last + 1, NIGHT_HALF_HOURS], axis=1) * capacity
    sun = ctx.weather.shortwave_midday[site, first : last + 1]
    clear = np.isfinite(midday) & np.isfinite(night) & (sun >= np.nanpercentile(sun, 60))
    return midday[clear], night[clear]


def _step_in_series(*, level: np.ndarray, min_each_side: int) -> tuple[int, float]:
    """Return the split of a series that minimises the squared error of two means, and the step."""
    best, best_cost = min_each_side, np.inf
    for k in range(min_each_side, level.size - min_each_side + 1):
        cost = ((level[:k] - level[:k].mean()) ** 2).sum() + (
            (level[k:] - level[k:].mean()) ** 2
        ).sum()
        if cost < best_cost:
            best, best_cost = k, float(cost)
    return best, float(level[best:].mean() - level[:best].mean())


def d8_solar_estimate(*, ctx: Context, register_day: int) -> SolarEstimate | None:
    """Estimate the telemetry date and size of D8's solar farm, and print the evidence.

    The date is the least-squares step of the clear-day midday-minus-night level between 2025-02-01
    and 2025-08-31, and the same step search on 2024 is printed as a control. The size is the fall
    of the clear-day midday median from April to August 2024 to the same months of 2025. D8 is a
    megavolt-ampere-only Primary, so a reverse flow folds under `abs` and the size is then a lower
    bound; the count of folded-looking days is printed.

    Args:
        ctx: The context.
        register_day: Day index of the register's connection date.

    Returns:
        The estimate, or None if the date is further than `D8_RELIABLE_DAYS` from the register date.
    """
    site = ctx.panel.sites.index(D8)
    grid = ctx.panel.power[site].reshape(ctx.panel.n_days, features.HALF_HOURS_PER_DAY)
    capacity = float(ctx.panel.capacity_mw[site])
    spans = {}
    for year in (2024, 2025):
        first = features.day_index(day=date(year, 2, 1))
        last = features.day_index(day=date(year, 8, 31))
        midday = np.nanmean(grid[first : last + 1, MIDDAY_HALF_HOURS], axis=1) * capacity
        night = np.nanmean(grid[first : last + 1, NIGHT_HALF_HOURS], axis=1) * capacity
        sun = ctx.weather.shortwave_midday[site, first : last + 1]
        keep = np.isfinite(midday) & np.isfinite(night) & (sun >= np.nanpercentile(sun, 60))
        days = np.arange(first, last + 1)[keep]
        split, step = _step_in_series(level=(midday - night)[keep], min_each_side=15)
        spans[year] = (int(days[split]), step)
    midday_2024, _ = _clear_midday_power(ctx=ctx, site=site, year=2024)
    midday_2025, _ = _clear_midday_power(ctx=ctx, site=site, year=2025)
    size = float(np.median(midday_2024) - np.median(midday_2025))
    folded = int((midday_2025 < FOLD_SHARE * capacity).sum())
    estimate_day, step = spans[2025]
    print(
        f"D8 solar telemetry: step in clear-day midday-minus-night level on "
        f"{features.day_of(index=estimate_day)} of {step:+.2f} MW (register date "
        f"{features.day_of(index=register_day)}); control search on 2024 finds "
        f"{features.day_of(index=spans[2024][0])} with {spans[2024][1]:+.2f} MW"
    )
    print(
        f"  clear-day midday median April-August: {np.median(midday_2024):.2f} MW in 2024, "
        f"{np.median(midday_2025):.2f} MW in 2025; fall {size:.2f} MW (register: 8.91 MW; "
        f"recollection: 0.7 MW); is_mva={bool(ctx.panel.is_mva[site])}; "
        f"{folded}/{midday_2025.size} clear 2025 midday days under {FOLD_SHARE:.0%} of capacity "
        f"(a reverse flow folded by abs would make the fall a lower bound)"
    )
    if abs(estimate_day - register_day) > D8_RELIABLE_DAYS:
        print(f"  the step is more than {D8_RELIABLE_DAYS} days from the register date: not used")
        return None
    return SolarEstimate(day=estimate_day, size_mw=size)


def segmentation_cross_check(*, ctx: Context, results: dict[int, SiteResult]) -> list[int]:
    """Print D8's penalised least-squares segmentation against the detector's episodes.

    The series is D8's scaled midday feature. The penalty is `SEGMENT_PENALTY` times the noise
    variance (from the robust spread of first differences) times the log of the series length.

    Args:
        ctx: The context.
        results: The real-panel results.

    Returns:
        The day index of each least-squares change point.
    """
    site = ctx.panel.sites.index(D8)
    feat = ctx.feats[site]
    days = np.flatnonzero(
        np.isfinite(feat.scaled[:, 2]) & (np.arange(feat.scaled.shape[0]) >= feat.ready_day)
    )
    x = feat.scaled[days, 2]
    diff = np.diff(x)
    sigma2 = (1.4826 * np.median(np.abs(diff - np.median(diff))) / math.sqrt(2)) ** 2
    penalty = SEGMENT_PENALTY * sigma2 * math.log(x.size)
    cuts = [
        int(days[c])
        for c in bocpd.penalised_segmentation(x=x, penalty=penalty, min_size=SEGMENT_MIN_DAYS)
    ]
    onsets = [e.onset_day for e in results[site].detector.episodes]
    print(
        f"D8 least-squares segmentation (midday feature): {len(cuts)} change points "
        f"{[str(features.day_of(index=c)) for c in cuts]}"
    )
    near = sum(any(abs(c - o) <= 30 for o in onsets) for c in cuts)
    print(
        f"  {near}/{len(cuts)} change points within 30 days of a detector onset; detector onsets "
        f"{[str(features.day_of(index=o)) for o in onsets]}"
    )
    return cuts


def d8_report(*, ctx: Context, results: dict[int, SiteResult], models: Models) -> None:
    """Print D8's known eras, read from the era table, with pass/fail gates.

    Era 1 is scored from its telemetry date. Era 2 (the solar farm) is scored from both the
    register date and the telemetry date of the era table (the year-on-year estimate; the clear-day
    estimate if the table has none), and a detection within `D8_SOLAR_GATE_DAYS` of either date
    passes, with the passing date printed. A pass on the solar telemetry date rests on a date
    fitted to the same telemetry, so the printed label says so; only the register-date pass is
    independent. A table date more than `D8_RELIABLE_DAYS` from the register date is not used. The
    2025-04-26 alarm of the design is 33 days after the register date and before the stage 0
    clear-day estimate of 2025-06-10, which the table's estimate replaces. A pass also needs the
    arm's state to be `era` 28 days after the date.

    Args:
        ctx: The context.
        results: The real-panel results.
        models: The frozen models.
    """
    d8 = ctx.panel.sites.index(D8)
    eras = ctx.era_table.filter((pl.col("primary") == D8) & pl.col("era_index").is_not_null())
    cuts = segmentation_cross_check(ctx=ctx, results=results)
    d8_stat_table(ctx=ctx)
    for row in eras.sort("era_index").iter_rows(named=True):
        register = features.day_index(day=row["register_date"])
        solar = row["technology"] == "solar farm"
        telemetry = row["telemetry_date"]
        estimate = d8_solar_estimate(ctx=ctx, register_day=register) if solar else None
        if telemetry is not None and solar:
            telemetry_day = features.day_index(day=telemetry)
            if abs(telemetry_day - register) > D8_RELIABLE_DAYS:
                print(
                    f"  the table's telemetry date {telemetry} is more than {D8_RELIABLE_DAYS} "
                    "days from the register date: not used"
                )
                dates = []
            else:
                dates = [("telemetry date (fitted on the telemetry itself)", telemetry_day)]
        elif telemetry is not None:
            dates = [("telemetry date", features.day_index(day=telemetry))]
        elif estimate is not None:
            dates = [("clear-day estimate", estimate.day)]
        else:
            dates = []
        if solar:
            dates.insert(0, ("register date", register))
        elif not dates:
            dates = [("register date", register)]
        gate_days = D8_SOLAR_GATE_DAYS if solar else D8_DETECTION_GATE_DAYS
        scores = {
            label: score_event(
                result=results[d8], onset_day=day, end_day=-1, truth="era",
                n_days=ctx.panel.n_days, train_end_day=models.train_end_day,
            )
            for label, day in dates
        }  # fmt: skip
        print(
            f"D8 era {row['era_index']} ({row['technology']}): scored from "
            + ", ".join(f"the {label} {features.day_of(index=day)}" for label, day in dates)
            + f"; least-squares change within 30 days of any: "
            f"{any(abs(c - day) <= 30 for c in cuts for _, day in dates)}"
        )
        for arm in ARMS:
            passing = [
                label
                for label, score in scores.items()
                if _within(value=score.get(f"{arm}_detection_delay"), limit=gate_days)
            ]
            reported = scores[passing[0]] if passing else next(iter(scores.values()))
            if f"{arm}_28" not in reported:
                print(f"    arm {arm}: not scored (onset before the arm's first readable day)")
                continue
            delays = {label: sc.get(f"{arm}_detection_delay") for label, sc in scores.items()}
            classed = reported[f"{arm}_28"] == STATE_ERA
            print(
                f"    arm {arm}: detection delays {delays} (gate at most {gate_days} days from "
                f"{'either date' if solar else 'the date'}): "
                f"{'PASS on the ' + ' and '.join(passing) if passing else 'FAIL'}; state at day 28 "
                f"{_state_name(reported[f'{arm}_28'])} (gate era): "
                f"{'PASS' if classed else 'FAIL'}; "
                f"classification delay {reported[f'{arm}_class_delay']}; state before the onset "
                f"{_state_name(reported[f'{arm}_pre_state'])}"
            )


def d8_stat_table(*, ctx: Context) -> None:
    """Print D8's window statistics around the two known changes, as in the L-stat design.

    Args:
        ctx: The context.
    """
    site = ctx.panel.sites.index(D8)
    windows = ctx.stat_windows[site]
    print("D8 window statistics, 28-day windows (share of capacity):")
    print("  window ends   p50      p05      residual p05  residual IQR")
    for end in D8_GAS_WINDOW_ENDS:
        row = windows.stat[features.day_index(day=end)]
        print(
            f"  {end}   {row[stats.P50]:.3f}   {row[stats.P05]:.3f}   "
            f"{row[stats.RESIDUAL_SHAPE.start]:+.3f}        "
            f"{row[stats.RESIDUAL_SHAPE.start + 2]:.3f}"
        )
    print("D8 midday minus night, 28-day windows, against one year earlier:")
    level = windows.stat[:, stats.MIDDAY] - windows.stat[:, stats.NIGHT]
    for end in D8_SOLAR_WINDOW_ENDS:
        day = features.day_index(day=end)
        print(
            f"  {end}   {level[day]:+.3f} vs {level[day - stats.YEAR_DAYS]:+.3f}   "
            f"difference {level[day] - level[day - stats.YEAR_DAYS]:+.3f}"
        )


def register_rows_report(*, ctx: Context, results: dict[int, SiteResult], models: Models) -> None:
    """Score the era table's other register connections (dates are unconfirmed by telemetry).

    Args:
        ctx: The context.
        results: The real-panel results.
        models: The frozen models.
    """
    rows = ctx.era_table.filter(
        (pl.col("primary") != D8) & pl.col("register_date").is_not_null()
        & (pl.col("confidence") == "register only")
    )  # fmt: skip
    scored = []
    for row in rows.iter_rows(named=True):
        site = ctx.panel.sites.index(row["primary"]) if row["primary"] in ctx.panel.sites else -1
        onset = features.day_index(day=row["register_date"])
        if site < 0 or onset < ctx.ready_days[site] or onset >= ctx.panel.n_days:
            continue
        score = score_event(
            result=results[site], onset_day=onset, end_day=-1, truth="era",
            n_days=ctx.panel.n_days, train_end_day=models.train_end_day,
        )  # fmt: skip
        scored.append({**row, **score})
    print(f"Other register connections of 1 MW or more: {rows.height} rows, {len(scored)} readable")
    if not scored:
        return
    frame = pl.DataFrame(scored, infer_schema_length=None)
    with pl.Config(tbl_rows=60, tbl_cols=12):
        print(
            frame.select(
                "primary",
                "register_date",
                "size_mw_register",
                "technology",
                "all_detection_delay",
                "all_28",
                "all_class_delay",
            )
        )
    detected = int(frame["all_detection_delay"].is_between(-2, 21).sum())
    era = int((frame["all_28"] == STATE_ERA).sum())
    print(
        f"  detected within 21 days of the register date {share_text(k=detected, n=frame.height)}; "
        f"classed era at day 28 {share_text(k=era, n=frame.height)} (descriptive: a register "
        f"date is not the telemetry date)"
    )


def print_example(
    *, ctx: Context, result: SiteResult, site: str, onset_day: int, truth: str, arm: ArmType = "all"
) -> None:
    """Print the daily rows of the episode that matches an onset, in the layout of plan section 5.

    Args:
        ctx: The context.
        result: The Primary's pipeline result.
        site: The Primary's name.
        onset_day: Day index of the onset to match.
        truth: A description of the truth behind the onset.
        arm: The arm to print.
    """
    if arm not in result.signals:
        print("example: arm not computed")
        return
    signal = result.signals[arm]
    episode = _matched_episode(signal=signal, onset_day=onset_day)
    if episode < 0:
        print(f"example: no episode of {site} starts within the window of {onset_day}")
        return
    detected = result.detector.episodes[episode]
    print(f"primary={site} onset={features.day_of(index=onset_day)} truth={truth} arm={arm}")
    print("day_processed  p(run<28d)  alarm  log_odds_era  state     first_origin_with_this_state")
    previous = -1
    spans: list[tuple[str, int]] = []
    for day in range(detected.alarm_day - 2, min(detected.alarm_day + 90, ctx.panel.n_days)):
        state = int(signal.state[day])
        changed = state != previous
        if changed:
            spans.append((STATE_NAMES[state], day + 1))
        log_odds = "-" if np.isnan(signal.log_odds[day]) else f"{signal.log_odds[day]:+.1f}"
        if changed or day == detected.alarm_day:
            origin = f"{features.day_of(index=day + 1)} 09:00 UTC" if changed else "-"
            alarm = "yes" if day >= detected.alarm_day else "no"
            print(
                f"{features.day_of(index=day)}     {result.detector.p_short[day]:.2f}        "
                f"{alarm:5s}  {log_odds:12s}  {STATE_NAMES[state]:8s}  {origin}"
            )
        previous = state
    classed = [s for s in spans if s[0] in ("era", "switch")]
    class_day = classed[0][1] - 1 if classed else None
    print(
        f"detection_delay_days={detected.alarm_day - onset_day}  classification_delay_days="
        f"{None if class_day is None else class_day - detected.alarm_day}"
    )
    print(" | ".join(f"origins from {features.day_of(index=d)} {name}" for name, d in spans))


def example_report(*, ctx: Context, results: dict[int, SiteResult], models: Models) -> None:
    """Print the example for D8's 2023-08-15 telemetry break, or for a bank event if none matches.

    Args:
        ctx: The context.
        results: The real-panel results.
        models: The frozen models.
    """
    d8 = ctx.panel.sites.index(D8)
    row = ctx.era_table.filter((pl.col("primary") == D8) & (pl.col("era_index") == 1)).row(
        0, named=True
    )
    onset = features.day_index(day=row["telemetry_date"])
    if _matched_episode(signal=results[d8].signals["all"], onset_day=onset) >= 0:
        print_example(
            ctx=ctx, result=results[d8], site=D8, onset_day=onset,
            truth=f"era (telemetry date {row['telemetry_date']}, register {row['register_date']})",
        )  # fmt: skip
        return
    print("example: no D8 episode matches the 2023-08-15 telemetry date; using a bank event")
    injected = inject.build_pass(
        panel=ctx.panel, weather_shortwave=ctx.weather.shortwave, ready_days=ctx.ready_days,
        verified_days=ctx.verified_days, world="bank", pass_index=0,
        onset_day=features.day_index(day=SCORED_MONTHS[0]),
    )  # fmt: skip
    results_bank = process_hosts(ctx=ctx, models=models, injected=injected, arms=("all",))
    truth = injected.truth.row(0, named=True)
    print_example(
        ctx=ctx, result=results_bank[truth["host"]], site=truth["site"],
        onset_day=truth["onset_day"],
        truth=f"{truth['truth']} kind={truth['kind']} size={truth['size']:+.2f}",
    )  # fmt: skip


def _perturb(
    *, panel: features.Panel, site: int, origin_day: int, rng: np.random.Generator
) -> features.Panel:
    """Return the panel cut after day `origin_day`, with the readings of day D changed.

    Every reading at or after 00:00 of the origin day changes, at the site and at its neighbours.
    The panel stops at the end of the origin day because nothing later can reach a signal that
    reads day `D - 1` at the latest.
    """
    cut = origin_day * features.HALF_HOURS_PER_DAY
    power = panel.power[:, : cut + features.HALF_HOURS_PER_DAY].copy()
    for row in [site, *panel.neighbours[site]]:
        power[row, cut:] = power[row, cut:] * rng.uniform(1.5, 2.5) + rng.uniform(0.2, 0.6)
    return panel.with_power(power=power)


def cut_test(*, ctx: Context, models: Models, results: dict[int, SiteResult], pairs: int) -> None:
    """Assert that readings at or after 00:00 of day D never change a signal at origin D.

    Every arm is checked, the hidden-Markov arm included.

    Args:
        ctx: The context.
        models: The frozen models.
        results: The real-panel results, the baseline.
        pairs: Number of random (Primary, origin) pairs.

    Raises:
        AssertionError: If any signal at the origin differs after the perturbation.
    """
    rng = np.random.default_rng(CUT_SEED)
    eligible = [
        i for i in range(len(ctx.panel.sites)) if ctx.ready_days[i] + 400 < ctx.panel.n_days - 2
    ]
    worst = 0.0
    for n in range(pairs):
        site = int(rng.choice(eligible))
        origin_day = int(rng.integers(ctx.ready_days[site] + 400, ctx.panel.n_days - 1))
        panel = _perturb(panel=ctx.panel, site=site, origin_day=origin_day, rng=rng)
        feats = [
            features.compute_site_features(power_site=panel.power[i])
            if i in (site, *ctx.panel.neighbours[site])
            else _truncate(feat=ctx.feats[i], n_days=panel.n_days)
            for i in range(len(panel.sites))
        ]
        grid = classify.build_mirror_grid(site_features=feats, neighbours=panel.neighbours)
        g4 = classify.g4_on(
            daily=np.stack([f.daily_mean for f in feats]), threshold=ctx.g4_threshold,
            ready_days=np.array([f.ready_day for f in feats]), neighbours=panel.neighbours,
            sites=[site],
        )  # fmt: skip
        weather = features.DailyWeather(
            shortwave=ctx.weather.shortwave[:, : panel.power.shape[1]],
            shortwave_midday=ctx.weather.shortwave_midday[:, : panel.n_days],
            temperature=ctx.weather.temperature[:, : panel.n_days],
        )
        redone = classify.process_site(
            site=site, all_features=feats, mirror=classify.mirror_features(grid=grid, site=site),
            g4=g4[site], neighbours=panel.neighbours, weather=weather, models=models,
            power=panel.power[site],
        )  # fmt: skip
        for arm, signal in redone.signals.items():
            before = results[site].signals[arm].at_origin(origin_day=origin_day)
            after = signal.at_origin(origin_day=origin_day)
            diff = float(np.nanmax(np.abs(before - after))) if np.isfinite(before).any() else 0.0
            worst = max(worst, diff)
            if not np.allclose(before, after, equal_nan=True, atol=1e-9):
                msg = (
                    f"pair {n}: signal of arm {arm} at origin day {origin_day} of "
                    f"{panel.sites[site]} changed by {diff}"
                )
                raise AssertionError(msg)
    print(f"perturbation test passed for {pairs} (Primary, origin) pairs; largest change {worst}")


def _truncate(*, feat: features.SiteFeatures, n_days: int) -> features.SiteFeatures:
    """Return the features of an untouched site cut to the first `n_days` days."""
    return features.SiteFeatures(
        residual=feat.residual[: n_days * features.HALF_HOURS_PER_DAY],
        exists=feat.exists[:n_days], raw=feat.raw[:n_days], scaled=feat.scaled[:n_days],
        spread=feat.spread, daily_mean=feat.daily_mean[:n_days], first_day=feat.first_day,
        ready_day=feat.ready_day,
    )  # fmt: skip


def main() -> None:
    """Run stage 0 from the command line."""
    parser = argparse.ArgumentParser(description="Experiment 709 stage 0 report")
    parser.add_argument(
        "--part", choices=("all", "tune", "real", "injected", "engine", "cut"), default="all"
    )
    parser.add_argument("--passes-per-month", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=CUT_PAIRS)
    parser.add_argument(
        "--load-models", action="store_true", help="Read the frozen models instead of tuning"
    )
    args = parser.parse_args()
    STAGE0_DIR.mkdir(parents=True, exist_ok=True)
    ctx = build_context()
    models = classify.load_models(folder=MODELS_DIR) if args.load_models else tune(ctx=ctx)
    if args.part == "tune":
        return
    if not np.any(models.stat_weights) or not np.any(models.stat_noshape_weights):
        msg = "the stat arms' weights are missing or zero in the loaded models: run --part tune"
        raise RuntimeError(msg)
    results = real_results(ctx=ctx, models=models)
    if args.part in ("all", "real"):
        scored = known_events_scores(ctx=ctx, results=results, train_end_day=models.train_end_day)
        d8_report(ctx=ctx, results=results, models=models)
        register_rows_report(ctx=ctx, results=results, models=models)
        known_report(ctx=ctx, scored=scored)
        example_report(ctx=ctx, results=results, models=models)
    if args.part in ("all", "injected"):
        events = injected_events(ctx=ctx, models=models, passes_per_month=args.passes_per_month)
        injected_report(events=events)
    if args.part in ("all", "engine"):
        engine = injected_events(
            ctx=ctx, models=models, passes_per_month=ENGINE_PASSES_PER_MONTH, world="engine"
        )
        engine_report(events=engine)
    if args.part in ("all", "cut"):
        cut_test(ctx=ctx, models=models, results=results, pairs=args.pairs)


if __name__ == "__main__":
    main()

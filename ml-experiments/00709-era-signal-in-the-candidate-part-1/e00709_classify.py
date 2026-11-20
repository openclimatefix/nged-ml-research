"""Experiment 709: era-or-switch evidence, the rule classifier, and the small soft classifier.

A detected shift (an episode of `e00709_bocpd`) is either an *era* (a permanent change in the
Primary's own demand) or a *switching event* (demand moved to or from a neighbour, usually for days
to weeks). The classifier reads four kinds of evidence, every one computed at the end of day `D - 1`
for the origin of day `D`, so the perturbation test of plan section 5 holds for all of them:

- **The neighbours' mirror image.** The causal mirror statistic of experiment 687
  (`e00687_features`, `mir_12` and `mir_48`), taken at the cut of each day and kept as a running
  maximum since the day before the onset.
- **Duration.** The log-odds that an event of this age is not a switching event, from the empirical
  survival of the labelled switching durations (`SwitchSurvival`).
- **The weather response.** The slope of the midday-band shift on midday shortwave radiation, and of
  the evening-band shift on temperature, from ERA5 days at least 7 days before the origin.
- **Gate G4 of experiment 592.** The gate turns on when the day's mean residual passes the
  Primary's threshold and a neighbour shows the opposite sign (`g4_on`, a port of `gate_fresh`
  that reads the daily climatology residual in place of experiment 565's stored forecast).

**Classifiers.** The `rule` arm sums seven evidence columns with weights fitted by logistic
regression on injected shifts, and holds three states (`unknown` until the log-odds pass 2.2 in
either direction, then `era` or `switch` for good, unless the level returns within 120 days and the
era is reversed). The `soft` arm is a network with one hidden layer of 16 units over 13 columns,
calibrated by Platt scaling, whose probability of `era` is refreshed daily. The `all` arm treats
every alarm as an era. Hindsight labels are never read as inputs.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import e00687_features as f687  # ty: ignore[unresolved-import]
import e00709_bocpd as bocpd
import e00709_features as features
import e00709_hmm as hmm
import e00709_statdetect as statdetect
import e00709_stats as stats
import numpy as np
from e00709_hmm import STATE_ERA, STATE_SWITCH, STATE_UNKNOWN

ArmType = Literal["all", "rule", "soft", "hmm", "stat", "stat_noshape"]
STATE_NAMES: Final[tuple[str, ...]] = ("none", "unknown", "era", "switch")

EVIDENCE_NAMES: Final[tuple[str, ...]] = (
    "bias", "mirror_48", "no_neighbour", "g4_fraction", "duration", "solar_t", "temp_t",
    "mirror_12", "cu_ratio", "cu_ret", "rate", "age_log", "change_share",
)  # fmt: skip
N_RULE_COLUMNS: Final[int] = 7
"""The first seven evidence columns feed the rule classifier."""
N_EVIDENCE: Final[int] = len(EVIDENCE_NAMES)
SHAPE_NAMES: Final[tuple[str, ...]] = ("profile_misfit", "residual_shape_change")
N_STAT_COLUMNS: Final[int] = N_RULE_COLUMNS + len(SHAPE_NAMES)
"""The `stat` arm's log-odds reads the seven rule columns and the two shape-evidence columns."""
STAT_DWELL_DAYS: Final[int] = 14
"""Age from the estimated onset before the `stat` arm may declare `era`; `switch` may be earlier."""
SHAPE_MIN_DAYS: Final[int] = 7
SHAPE_CLIP: Final[float] = 8.0
LOG_ODDS_THRESHOLD: Final[float] = float(np.log(9.0))
SOFT_LOW: Final[float] = 0.2
SOFT_HIGH: Final[float] = 0.8
REVERSAL_DAYS: Final[int] = 120
SETTLING_DAYS: Final[int] = 28
MAX_EVIDENCE_DAYS: Final[int] = 240
MIRROR_CLIP: Final[float] = 8.0
T_CLIP: Final[float] = 6.0
WEATHER_LAG_DAYS: Final[int] = 6
"""ERA5 days at most this many days before the last processed day: at least 7 days before origin."""
WEATHER_MIN_DAYS: Final[int] = 6
NEIGHBOUR_LOOKBACK_DAYS: Final[int] = 14
G4_SD_MULTIPLE: Final[float] = 3.0
G4_MIN_SHARE: Final[float] = 0.05
G4_MIN_THRESHOLD_DAYS: Final[int] = 100
SURVIVAL_FLOOR: Final[float] = 0.01
SOFT_HIDDEN: Final[int] = 16


# ---------------------------------------------------------------- evidence inputs


@dataclass(frozen=True)
class MirrorGrid:
    """Experiment 687's block means of the panel's residual, ready for the causal features.

    Attributes:
        grids: The `Grids` of `e00687_features.build_grids`.
        neighbours: Electrical neighbours as indices into the panel's sites.
    """

    grids: f687.Grids
    neighbours: dict[int, list[int]]


def _known_residual(*, feat: features.SiteFeatures) -> np.ndarray:
    """Return a site's residual with every day before its noise unit exists set to NaN.

    The noise unit of a site is measured on its first 365 days of features, so it is known only
    from `ready_day`. A site whose window is incomplete (`ready_day` equal to the number of days)
    contributes nothing, which keeps a unit that is measured on later days from scaling the
    readings of earlier ones.
    """
    out = feat.residual.copy()
    out[: feat.ready_day * features.HALF_HOURS_PER_DAY] = np.nan
    return out


def build_mirror_grid(
    *,
    site_features: Sequence[features.SiteFeatures],
    neighbours: dict[int, list[int]],
    reference_features: Sequence[features.SiteFeatures] | None = None,
) -> MirrorGrid:
    """Build experiment 687's grids from the sites' climatology residuals.

    The noise unit of each block width is measured on each Primary's first 365 days of features,
    and a Primary's readings count only from the day that window is complete (`_known_residual`).
    The all-Primaries median that removes common weather is taken over `reference_features`, which
    is the uninjected panel when shifts are injected, as experiment 687 does.

    Args:
        site_features: Features of every site (injected or real).
        neighbours: Electrical neighbours of each site.
        reference_features: Features of the uninjected panel; defaults to `site_features`.

    Returns:
        The grids and the neighbour map.
    """
    gap = np.stack([_known_residual(feat=f) for f in site_features])
    reference = (
        gap
        if reference_features is None
        else np.stack([_known_residual(feat=f) for f in reference_features])
    )
    development = np.full_like(gap, np.nan)
    for row, feat in enumerate(site_features):
        if feat.first_day >= 0 and feat.ready_day < feat.raw.shape[0]:
            lo = feat.first_day * features.HALF_HOURS_PER_DAY
            hi = lo + features.SCALE_WINDOW_DAYS * features.HALF_HOURS_PER_DAY
            development[row, lo:hi] = feat.residual[lo:hi]
    units = {w: f687.block_unit(gaps=[development], w=w) for w in f687.W}
    grids = f687.build_grids(gap=gap, gap_sig=gap, gap_ref=reference, units=units, start=0)
    return MirrorGrid(grids=grids, neighbours=neighbours)


def mirror_features(
    *, grid: MirrorGrid, site: int, neighbours: dict[int, list[int]] | None = None
) -> np.ndarray:
    """Return experiment 687's causal features of one site at the end of every day.

    Args:
        grid: The grids.
        site: Index of the site.
        neighbours: Neighbour map to use instead of the grid's, for a host whose mirror partner is
            unmetered (an empty list for the host).

    Returns:
        An array (days, 687 features), the features read at the last half-hour of each day.
    """
    n_days = grid.grids.lvl[f687.W[0]].shape[1] // features.HALF_HOURS_PER_DAY
    days = np.arange(n_days)
    cut = (days + 1) * features.HALF_HOURS_PER_DAY - 1
    prim = np.full(n_days, site)
    out, _ = f687.features_at(
        g=grid.grids, prim=prim, t=cut, neighbours=neighbours or grid.neighbours
    )
    return out


def g4_on(
    *,
    daily: np.ndarray,
    threshold: np.ndarray,
    ready_days: np.ndarray,
    neighbours: dict[int, list[int]],
    sites: Sequence[int] | None = None,
) -> np.ndarray:
    """Return gate G4 of experiment 592 for every site and day, read at the end of the day.

    The gate turns on when one daily reading passes the site's threshold and an electrical
    neighbour has a reading of the opposite sign above its own threshold on the same day. It turns
    off when the reading falls below half the threshold. A missing reading leaves the state.

    Args:
        daily: Daily mean residual of every site, shape (sites, days).
        threshold: Per-site threshold, `max(3 sd, 5% of capacity)`, measured on the site's first 365
            days of features.
        ready_days: First day on which each site's threshold exists (the end of that 365-day
            window). Before it the site shows no reading above its threshold, so a threshold
            measured on later days never acts on earlier ones.
        neighbours: Electrical neighbours of each site.
        sites: Sites to compute (the others stay off); all sites if omitted.

    Returns:
        A boolean array (sites, days).
    """
    n_sites, n_days = daily.shape
    known = np.arange(n_days)[None, :] >= ready_days[:, None]
    big = known & np.isfinite(daily) & (np.abs(daily) > threshold[:, None])
    sign = np.where(big, np.sign(daily), 0.0)
    on = np.zeros((n_sites, n_days), dtype=bool)
    for site in range(n_sites) if sites is None else sites:
        state = False
        for day in range(n_days):
            reading = daily[site, day]
            if not state and sign[site, day] != 0:
                state = any(sign[q, day] == -sign[site, day] for q in neighbours.get(site, []))
            elif state and np.isfinite(reading) and abs(reading) < 0.5 * threshold[site]:
                state = False
            on[site, day] = state
    return on


def g4_thresholds(*, site_features: Sequence[features.SiteFeatures]) -> np.ndarray:
    """Return each site's G4 threshold, from its first 365 days of features.

    Args:
        site_features: Features of every site.

    Returns:
        `max(3 sd, 5% of capacity)` per site.
    """
    out = np.full(len(site_features), G4_MIN_SHARE)
    for row, feat in enumerate(site_features):
        if feat.first_day < 0:
            continue
        window = feat.daily_mean[feat.first_day : feat.first_day + features.SCALE_WINDOW_DAYS]
        if np.isfinite(window).sum() >= G4_MIN_THRESHOLD_DAYS:
            out[row] = max(G4_SD_MULTIPLE * float(np.nanstd(window)), G4_MIN_SHARE)
    return out


@dataclass(frozen=True)
class SwitchSurvival:
    """Empirical survival of the labelled switching events' durations.

    Attributes:
        durations: Durations in days of the labelled switching events (the verified events of
            experiment 580 without the ones in the scored window).
    """

    durations: np.ndarray

    def log_odds_not_switch(self, *, age: np.ndarray) -> np.ndarray:
        """Return the log-odds that a shift of this age is not a switching event.

        Args:
            age: Age of the shift in days.

        Returns:
            Minus the log of the (smoothed, floored) share of labelled events that last this long.
        """
        n = len(self.durations)
        alive = (self.durations[None, :] >= np.asarray(age, dtype=float)[:, None]).sum(axis=1)
        return -np.log(np.maximum((alive + 0.5) / (n + 1.0), SURVIVAL_FLOOR))


def _slope_t(*, x: np.ndarray, y: np.ndarray) -> float:
    """Return the absolute t-statistic of the slope of y on x, 0 if it is undefined."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < WEATHER_MIN_DAYS:
        return 0.0
    x, y = x[ok], y[ok]
    centred = x - x.mean()
    sxx = float((centred**2).sum())
    if sxx <= 0:
        return 0.0
    slope = float((centred * (y - y.mean())).sum() / sxx)
    resid = y - y.mean() - slope * centred
    variance = float((resid**2).sum() / max(len(x) - 2, 1))
    if variance <= 0:
        return T_CLIP
    return min(abs(slope) / np.sqrt(variance / sxx), T_CLIP)


def weather_evidence(
    *,
    feat: features.SiteFeatures,
    previous_mean: np.ndarray,
    weather: features.DailyWeather,
    site: int,
    onset: int,
    day: int,
) -> tuple[float, float]:
    """Return the solar and the temperature response of a shift, from ERA5 days that are old enough.

    The solar response is the t-statistic of the midday-band shift on midday shortwave radiation,
    counted only when the night-band shift is under half the midday shift (a solar farm adds
    nothing at night). The temperature response is the t-statistic of the evening-band shift on
    temperature (a transferred load follows temperature).

    Args:
        feat: The site's features.
        previous_mean: The previous segment's mean scaled features.
        weather: ERA5 summaries.
        site: Index of the site.
        onset: Estimated onset day of the shift.
        day: Last processed day; ERA5 days up to `day - 6` are read.

    Returns:
        The solar and the temperature t-statistics, each 0 to 6.
    """
    last = day - WEATHER_LAG_DAYS
    if last - onset + 1 < WEATHER_MIN_DAYS:
        return 0.0, 0.0
    span = slice(onset, last + 1)
    shift = feat.raw[span] - (previous_mean * feat.spread)[None, :]
    solar = _slope_t(x=weather.shortwave_midday[site, span], y=shift[:, 2])
    midday = np.nanmean(shift[:, 2])
    night = np.nanmean(shift[:, 0])
    if not np.isfinite(midday) or abs(night) >= 0.5 * abs(midday):
        solar = 0.0
    temperature = _slope_t(x=weather.temperature[site, span], y=shift[:, 3])
    return float(solar), float(temperature)


# ---------------------------------------------------------------- the evidence of one episode


@dataclass(frozen=True)
class EpisodeEvidence:
    """The evidence columns of one episode, from its alarm day on.

    Attributes:
        days: Days from the alarm day to the last day with evidence.
        x: Evidence matrix, shape (days, 13), columns in `EVIDENCE_NAMES`.
        shape: The two shape-evidence columns (`SHAPE_NAMES`) of an episode of the `stat` arm,
            shape (days, 2), or None.
    """

    days: np.ndarray
    x: np.ndarray
    shape: np.ndarray | None = None


def episode_evidence(
    *,
    episode: bocpd.Episode,
    detector: bocpd.DetectorOutput,
    feat: features.SiteFeatures,
    mirror: np.ndarray,
    g4: np.ndarray,
    site: int,
    neighbours: dict[int, list[int]],
    all_features: Sequence[features.SiteFeatures],
    weather: features.DailyWeather,
    survival: SwitchSurvival,
) -> EpisodeEvidence:
    """Compute the evidence of an episode on each day from its alarm, with no later reading.

    Args:
        episode: The detected episode.
        detector: The Primary's detector output.
        feat: The Primary's features.
        mirror: The Primary's experiment 687 features at the end of each day, shape (days, 687).
        g4: Gate G4 of this Primary on each day.
        site: Index of the Primary.
        neighbours: Electrical neighbours of each site.
        all_features: Features of every site, to find the metered neighbours.
        weather: ERA5 summaries.
        survival: Survival of the labelled switching durations.

    Returns:
        The evidence matrix.
    """
    n_days = feat.raw.shape[0]
    stop = min(
        episode.end_day if episode.end_day >= 0 else n_days - 1,
        episode.alarm_day + MAX_EVIDENCE_DAYS - 1,
    )
    days = np.arange(episode.alarm_day, stop + 1)
    first = max(episode.onset_day - 1, 0)
    names = f687.FEATURE_NAMES
    run = np.arange(first, stop + 1)
    mir48 = np.maximum.accumulate(mirror[run, names.index("mir_48")])
    mir12 = np.maximum.accumulate(mirror[run, names.index("mir_12")])
    g4_frac = np.cumsum(g4[run]) / np.arange(1, len(run) + 1)
    pick = days - first
    lookback = slice(max(episode.onset_day - NEIGHBOUR_LOOKBACK_DAYS, 0), max(episode.onset_day, 1))
    metered = any(
        np.isfinite(all_features[j].daily_mean[lookback]).any() for j in neighbours.get(site, [])
    )
    age = days - episode.onset_day + 1
    x = np.zeros((len(days), N_EVIDENCE))
    x[:, 0] = 1.0
    x[:, 1] = np.clip(mir48[pick], 0.0, MIRROR_CLIP) if metered else 0.0
    x[:, 2] = 0.0 if metered else 1.0
    x[:, 3] = g4_frac[pick]
    x[:, 4] = survival.log_odds_not_switch(age=age)
    for i, day in enumerate(days):
        solar, temperature = weather_evidence(
            feat=feat,
            previous_mean=episode.previous_mean,
            weather=weather,
            site=site,
            onset=episode.onset_day,
            day=int(day),
        )
        x[i, 5], x[i, 6] = solar, temperature
    x[:, 7] = np.clip(mir12[pick], 0.0, MIRROR_CLIP) if metered else 0.0
    x[:, 8] = mirror[days, names.index("cu_ratio")]
    x[:, 9] = mirror[days, names.index("cu_ret")]
    x[:, 10] = mirror[days, names.index("rate")]
    x[:, 11] = np.log1p(np.minimum(age, 365))
    delta_share = np.abs(detector.delta_scaled[days]) * feat.spread[None, :]
    x[:, 12] = np.nan_to_num(np.nanmax(delta_share, axis=1))
    return EpisodeEvidence(days=days, x=x)


def _valid_days(*, block: np.ndarray) -> int:
    """Return the number of days of a (days, 48) block with at least 24 readings."""
    return int((np.isfinite(block).sum(axis=1) >= features.MIN_READINGS_PER_DAY).sum())


def shape_evidence(
    *, power_site: np.ndarray, episode: bocpd.Episode, days: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Return the two shape-evidence columns of an episode on each of its days.

    Both columns compare the post-onset days (the first 28 days after the onset, or fewer up to the
    day, and at least 7) with the 28 days before the onset, so no window straddles the step.

    - *Profile misfit.* Fit the post-onset median half-hour profile as `a * pre-onset profile + b`.
      A transfer scales or shifts the profile, so its misfit is small. An engine or a solar farm
      adds a shape that the fit cannot reproduce. The root-mean-square misfit is divided by the
      scale of the 50th-percentile statistic.
    - *Residual-shape change.* The largest absolute change, over the 5th percentile, the 95th
      percentile, and the interquartile range of the residual around each window's own median
      profile, divided by that statistic's scale.

    Args:
        power_site: The Primary's power over capacity.
        episode: The episode.
        days: The days to evaluate (the evidence days of the episode).
        scale: The statistics detector's scale on the alarm day, shape (12,).

    Returns:
        An array (days, 2), zero before 7 days after the onset or where the pre-onset window has
        fewer than 17 valid days.
    """
    grid = power_site.reshape(-1, features.HALF_HOURS_PER_DAY)
    onset = episode.onset_day
    out = np.zeros((len(days), len(SHAPE_NAMES)))
    pre = grid[max(onset - stats.WINDOW_DAYS, 0) : onset]
    if pre.shape[0] == 0 or _valid_days(block=pre) < stats.MIN_VALID_DAYS:
        return out
    floor = stats.SCALE_FLOOR
    level_scale = float(scale[stats.P50]) if np.isfinite(scale[stats.P50]) else floor
    shape_scale = np.where(
        np.isfinite(scale[stats.RESIDUAL_SHAPE]), scale[stats.RESIDUAL_SHAPE], floor
    )
    pre_profile = np.nanmedian(pre, axis=0)
    pre_shape = np.array(stats.residual_shape(block=pre))
    cache: dict[int, tuple[float, float]] = {}
    for i, day in enumerate(days):
        end = min(int(day), onset + stats.WINDOW_DAYS - 1)
        if end - onset + 1 < SHAPE_MIN_DAYS:
            continue
        if end not in cache:
            post = grid[onset : end + 1]
            post_profile = np.nanmedian(post, axis=0)
            ok = np.isfinite(pre_profile) & np.isfinite(post_profile)
            misfit = shape_change = 0.0
            if ok.sum() >= features.HALF_HOURS_PER_DAY // 2:
                design = np.stack([pre_profile[ok], np.ones(int(ok.sum()))], axis=1)
                coefficients, *_ = np.linalg.lstsq(design, post_profile[ok], rcond=None)
                residual = post_profile[ok] - design @ coefficients
                misfit = float(np.sqrt(np.mean(residual**2)) / level_scale)
                post_shape = np.array(stats.residual_shape(block=post))
                change = np.abs(post_shape - pre_shape) / shape_scale
                shape_change = float(np.nanmax(change)) if np.isfinite(change).any() else 0.0
            cache[end] = (min(misfit, SHAPE_CLIP), min(shape_change, SHAPE_CLIP))
        out[i] = cache[end]
    return out


def stat_episode_evidence(
    *,
    output: statdetect.StatOutput,
    power_site: np.ndarray,
    feat: features.SiteFeatures,
    mirror: np.ndarray,
    g4: np.ndarray,
    site: int,
    neighbours: dict[int, list[int]],
    all_features: Sequence[features.SiteFeatures],
    weather: features.DailyWeather,
    survival: SwitchSurvival,
) -> dict[int, EpisodeEvidence]:
    """Compute the evidence of every episode of the statistics detector, shape terms included.

    Args:
        output: The statistics detector's output.
        power_site: The Primary's power over capacity.
        feat: The Primary's features.
        mirror: The Primary's experiment 687 features at the end of each day.
        g4: Gate G4 of this Primary on each day.
        site: Index of the Primary.
        neighbours: Electrical neighbours of each site.
        all_features: Features of every site.
        weather: ERA5 summaries.
        survival: Survival of the labelled switching durations.

    Returns:
        The evidence of each episode, with the `shape` columns set.
    """
    out = {}
    for episode in output.detector.episodes:
        base = episode_evidence(
            episode=episode, detector=output.detector, feat=feat, mirror=mirror, g4=g4, site=site,
            neighbours=neighbours, all_features=all_features, weather=weather, survival=survival,
        )  # fmt: skip
        shape = shape_evidence(
            power_site=power_site, episode=episode, days=base.days,
            scale=output.scale[episode.alarm_day],
        )  # fmt: skip
        out[episode.episode] = EpisodeEvidence(days=base.days, x=base.x, shape=shape)
    return out


# ---------------------------------------------------------------- the fitted classifiers


def fit_logistic(*, x: np.ndarray, y: np.ndarray, l2: float = 0.1) -> np.ndarray:
    """Fit a class-balanced, L2-penalised logistic regression by Newton's method.

    Args:
        x: Rows of the first `N_RULE_COLUMNS` evidence columns (the bias is column 0).
        y: 1 for an era, 0 for a switching event.
        l2: Penalty on every weight except the bias.

    Returns:
        The weights; the log-odds of `era` is `x @ w`.
    """
    positive = max(float(y.sum()), 1.0)
    negative = max(float((1 - y).sum()), 1.0)
    sample_weight = np.where(y == 1, 0.5 / positive, 0.5 / negative) * len(y)
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.0
    w = np.zeros(x.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        gradient = x.T @ (sample_weight * (p - y)) + penalty @ w
        hessian = (
            (x * (sample_weight * p * (1 - p))[:, None]).T @ x + penalty + 1e-6 * np.eye(len(w))
        )
        step = np.linalg.solve(hessian, gradient)
        w = w - step
        if float(np.abs(step).max()) < 1e-8:
            break
    return w


@dataclass(frozen=True)
class SoftClassifier:
    """A network with one hidden layer of 16 units that outputs a calibrated probability of `era`.

    Attributes:
        mean: Column means of the training inputs.
        scale: Column standard deviations of the training inputs.
        w1: Input-to-hidden weights, shape (inputs, 16).
        b1: Hidden biases.
        w2: Hidden-to-output weights.
        b2: Output bias.
        platt: Slope and intercept of the Platt scaling of the output logit.
    """

    mean: np.ndarray
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    platt: tuple[float, float]

    def logit(self, *, x: np.ndarray) -> np.ndarray:
        """Return the raw output logit of each row.

        Args:
            x: Evidence rows, shape (rows, 13).

        Returns:
            The logit before calibration.
        """
        hidden = np.tanh(((x - self.mean) / self.scale) @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def probability(self, *, x: np.ndarray) -> np.ndarray:
        """Return the calibrated probability of `era` of each row.

        Args:
            x: Evidence rows, shape (rows, 13).

        Returns:
            Probabilities between 0 and 1.
        """
        z = self.platt[0] * self.logit(x=x) + self.platt[1]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_soft(
    *, x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, epochs: int = 600
) -> SoftClassifier:
    """Train the soft classifier on injected shifts and calibrate it on held-out injection passes.

    Args:
        x: Evidence rows, shape (rows, 13).
        y: 1 for an era, 0 for a switching event.
        groups: Injection pass of each row; whole passes are held out for the calibration.
        seed: Random seed.
        epochs: Full-batch Adam steps.

    Returns:
        The trained and calibrated classifier.
    """
    rng = np.random.default_rng(seed)
    passes = np.unique(groups)
    held = set(rng.choice(passes, size=max(len(passes) // 5, 1), replace=False).tolist())
    holdout = np.isin(groups, list(held))
    mean = x[~holdout].mean(axis=0)
    scale = np.maximum(x[~holdout].std(axis=0), 1e-6)
    z = (x[~holdout] - mean) / scale
    target = y[~holdout].astype(float)
    w1 = rng.normal(0, 0.3, size=(x.shape[1], SOFT_HIDDEN))
    b1 = np.zeros(SOFT_HIDDEN)
    w2 = rng.normal(0, 0.3, size=SOFT_HIDDEN)
    b2 = 0.0
    params = [w1, b1, w2, np.array([b2])]
    moment1 = [np.zeros_like(p) for p in params]
    moment2 = [np.zeros_like(p) for p in params]
    for step in range(1, epochs + 1):
        hidden = np.tanh(z @ params[0] + params[1])
        p = 1.0 / (1.0 + np.exp(-(hidden @ params[2] + params[3][0])))
        err = (p - target) / len(target)
        back = np.outer(err, params[2]) * (1 - hidden**2)
        grads = [z.T @ back + 1e-3 * params[0], back.sum(axis=0), hidden.T @ err + 1e-3 * params[2],
                 np.array([err.sum()])]  # fmt: skip
        for k, g in enumerate(grads):
            moment1[k] = 0.9 * moment1[k] + 0.1 * g
            moment2[k] = 0.999 * moment2[k] + 0.001 * g**2
            params[k] = params[k] - 0.01 * (moment1[k] / (1 - 0.9**step)) / (
                np.sqrt(moment2[k] / (1 - 0.999**step)) + 1e-8
            )
    raw = SoftClassifier(mean=mean, scale=scale, w1=params[0], b1=params[1], w2=params[2],
                         b2=float(params[3][0]), platt=(1.0, 0.0))  # fmt: skip
    logit = raw.logit(x=x[holdout]) if holdout.any() else raw.logit(x=x)
    truth = y[holdout] if holdout.any() else y
    slope_intercept = fit_logistic(
        x=np.column_stack([np.ones_like(logit), logit]), y=truth, l2=1e-3
    )
    platt = (float(slope_intercept[1]), float(slope_intercept[0]))
    return SoftClassifier(
        mean=mean,
        scale=scale,
        w1=params[0],
        b1=params[1],
        w2=params[2],
        b2=float(params[3][0]),
        platt=platt,
    )


@dataclass
class Models:
    """Everything that stage 0 fits once, on tuning data, and then freezes.

    Attributes:
        threshold: The detector's alarm threshold.
        rule_weights: Weights of the rule classifier's seven evidence columns.
        survival: Survival of the labelled switching durations.
        train_end_day: First day that the hidden-Markov fits did not read.
        soft: The soft classifier, or None before it is trained.
        hmm_parameters: The sticky hidden-Markov model of each site, by site index.
        stat_threshold: The statistics detector's alarm threshold on the largest absolute scaled
            change.
        stat_weights: Weights of the `stat` arm's nine evidence columns (the seven rule columns and
            the two shape-evidence columns).
        stat_noshape_weights: Weights of the `stat_noshape` arm, which uses the statistics detector
            and the seven rule columns only, with the same 9-column layout (the shape weights are
            zero).
    """

    threshold: float
    rule_weights: np.ndarray
    survival: SwitchSurvival
    train_end_day: int
    soft: SoftClassifier | None = None
    hmm_parameters: dict[int, hmm.HmmParameters] = field(default_factory=dict)
    stat_threshold: float = statdetect.DEFAULT_STAT_THRESHOLD
    stat_weights: np.ndarray = field(default_factory=lambda: np.zeros(N_STAT_COLUMNS))
    stat_noshape_weights: np.ndarray = field(default_factory=lambda: np.zeros(N_STAT_COLUMNS))


# ---------------------------------------------------------------- classes per episode and site


@dataclass(frozen=True)
class EpisodeClass:
    """The class of an episode on each day from its alarm.

    Attributes:
        state: Class code of each day (unknown, era, or switch).
        p_era: Probability of `era` (0 or 1 for the arms without a probability).
        log_odds: The rule's log-odds, NaN for the arms without one.
        class_day: First day with class `era`, or -1 if the episode was never classed `era`.
    """

    state: np.ndarray
    p_era: np.ndarray
    log_odds: np.ndarray
    class_day: int


def classify_episode(  # noqa: PLR0912 - one branch per arm
    *, evidence: EpisodeEvidence, episode: bocpd.Episode, arm: ArmType, models: Models
) -> EpisodeClass:
    """Classify an episode day by day with one of the three detector-fed arms.

    A class `era` is reversed to `switch` if the level returns inside 120 days of the onset
    (arms `rule` and `soft`; the `all` arm never reverses).

    Args:
        evidence: The episode's evidence.
        episode: The episode.
        arm: `all`, `rule`, `soft`, or `stat`. The `stat` arm reads the nine columns of
            `N_STAT_COLUMNS` and may declare `era` only from `STAT_DWELL_DAYS` of age after the
            estimated onset (`switch` may be declared earlier).
        models: The frozen weights and thresholds.

    Returns:
        The class of the episode on each day.

    Raises:
        ValueError: If the arm is `soft` and no soft classifier is set, or the arm is `hmm`.
    """
    m = len(evidence.days)
    log_odds = np.full(m, np.nan)
    if arm == "all":
        p_era, state = np.ones(m), np.full(m, STATE_ERA, dtype=np.int8)
    elif arm == "rule":
        log_odds = evidence.x[:, :N_RULE_COLUMNS] @ models.rule_weights
        state = np.full(m, STATE_UNKNOWN, dtype=np.int8)
        locked = STATE_UNKNOWN
        for i in range(m):
            if locked == STATE_UNKNOWN and abs(log_odds[i]) >= LOG_ODDS_THRESHOLD:
                locked = STATE_ERA if log_odds[i] > 0 else STATE_SWITCH
            state[i] = locked
        p_era = np.where(state == STATE_ERA, 1.0, np.where(state == STATE_UNKNOWN, 0.5, 0.0))
    elif arm in ("stat", "stat_noshape"):
        shape = evidence.shape if evidence.shape is not None else np.zeros((m, len(SHAPE_NAMES)))
        weights = models.stat_weights if arm == "stat" else models.stat_noshape_weights
        log_odds = np.hstack([evidence.x[:, :N_RULE_COLUMNS], shape]) @ weights
        age = evidence.days - episode.onset_day + 1
        state = np.full(m, STATE_UNKNOWN, dtype=np.int8)
        locked = STATE_UNKNOWN
        for i in range(m):
            if locked == STATE_UNKNOWN:
                if log_odds[i] <= -LOG_ODDS_THRESHOLD:
                    locked = STATE_SWITCH
                elif log_odds[i] >= LOG_ODDS_THRESHOLD and age[i] >= STAT_DWELL_DAYS:
                    locked = STATE_ERA
            state[i] = locked
        p_era = np.where(state == STATE_ERA, 1.0, np.where(state == STATE_UNKNOWN, 0.5, 0.0))
    elif arm == "soft":
        if models.soft is None:
            msg = "the soft classifier is not trained"
            raise ValueError(msg)
        p_era = models.soft.probability(x=evidence.x)
        state = np.where(
            p_era >= SOFT_HIGH, STATE_ERA, np.where(p_era <= SOFT_LOW, STATE_SWITCH, STATE_UNKNOWN)
        ).astype(np.int8)
    else:
        msg = "the hmm arm is not classified from episodes"
        raise ValueError(msg)
    reversed_at = _reversal_index(evidence=evidence, episode=episode) if arm != "all" else None
    if reversed_at is not None:
        state[reversed_at:] = STATE_SWITCH
        p_era[reversed_at:] = 0.0
    era_days = np.flatnonzero(state == STATE_ERA)
    class_day = int(evidence.days[era_days[0]]) if len(era_days) else -1
    return EpisodeClass(state=state, p_era=p_era, log_odds=log_odds, class_day=class_day)


def _reversal_index(*, evidence: EpisodeEvidence, episode: bocpd.Episode) -> int | None:
    """Return the index from which a returned level re-classes the episode as a switch."""
    if episode.ended_by != "return" or episode.end_day - episode.onset_day + 1 > REVERSAL_DAYS:
        return None
    hits = np.flatnonzero(evidence.days >= episode.end_day)
    return int(hits[0]) if len(hits) else None


@dataclass(frozen=True)
class SiteSignal:
    """The per-day signal of one Primary for one arm, as of the end of each day.

    The signal at the origin of day `D` is the entry of day `D - 1` (`at_origin`).

    Attributes:
        state: Class code per day: 0 none, 1 unknown, 2 era, 3 switch.
        p_era: Probability of `era`.
        log_odds: The rule's log-odds (NaN where not defined).
        age_days: Age of the current shift or era in days.
        era_index: Number of eras adopted and not reversed by each day.
        change_vector: Scaled change of the six features, frozen 28 days after an era class.
        anchor: The era anchor, the day-mean of the change in capacity share (zero unless `era`).
        depth: Estimated depth of a switching event or of an unknown shift, a share of capacity.
        episode_id: The open episode (or hidden-Markov excursion) of each day, or -1.
        persistent_episode: The episode whose era is current on each day, or -1.
        withdrawn: True on a day when an era was withdrawn (reversed, or its level returned), so
            the anchor falls back to the previous era's.
        episode_bounds: First day and last day (-1 while open) of each episode or excursion.
        era_id: The id of the current era of the `stat` arm (0 before any era is confirmed, and for
            every other arm). An id confirmed on day `T` appears from the entry of day `T`.
    """

    state: np.ndarray
    p_era: np.ndarray
    log_odds: np.ndarray
    age_days: np.ndarray
    era_index: np.ndarray
    change_vector: np.ndarray
    anchor: np.ndarray
    depth: np.ndarray
    episode_id: np.ndarray
    persistent_episode: np.ndarray
    withdrawn: np.ndarray
    episode_bounds: list[tuple[int, int]]
    era_id: np.ndarray

    def at_origin(self, *, origin_day: int) -> np.ndarray:
        """Return every column at the origin of a day, as one flat vector for comparison.

        Args:
            origin_day: The origin day `D`; the values are those after day `D - 1`.

        Returns:
            State, probability, log-odds, age, era index, anchor, depth, withdrawal flag,
            persistent episode, era id, and the change vector.
        """
        d = origin_day - 1
        return np.concatenate(
            [
                [self.state[d], self.p_era[d], self.log_odds[d], self.age_days[d]],
                [self.era_index[d], self.anchor[d], self.depth[d], self.withdrawn[d]],
                [self.persistent_episode[d], self.era_id[d]],
                self.change_vector[d],
            ]
        )


def _frozen_change(
    *, detector: bocpd.DetectorOutput, episode: bocpd.Episode, class_day: int, day: int
) -> np.ndarray:
    """Return the change vector at `day`: live until 28 days after the era class, then frozen."""
    n_days = detector.delta_scaled.shape[0]
    last_open = episode.end_day if episode.end_day >= 0 else n_days - 1
    freeze = min(class_day + SETTLING_DAYS, last_open) if class_day >= 0 else last_open
    return np.nan_to_num(detector.delta_scaled[min(day, freeze)])


def _level_returned(*, episode: bocpd.Episode, day: int) -> bool:
    """Return True once an episode ended because its level came back, on the day after."""
    return episode.ended_by == "return" and episode.end_day < day


def compose_site_signal(
    *,
    detector: bocpd.DetectorOutput,
    classes: dict[int, EpisodeClass],
    spread: np.ndarray,
    n_days: int,
    vectors: dict[int, tuple[int, np.ndarray]] | None = None,
) -> SiteSignal:
    """Combine the episodes' classes into one per-day signal of a Primary.

    While an episode is open, its own class is the state. After a switch or an unknown episode
    closes, the state returns to the earlier era or to `none`. An episode that was classed `era`
    stays `era` (with its frozen change vector) until a later era is adopted or the class is
    reversed. An era whose level returns after more than 120 days is withdrawn: the signal reverts
    to the previous era's change vector and anchor and `withdrawn` is set on that day. The change
    vector of a later era is relative to the level of the era before it.

    Args:
        detector: The Primary's detector output.
        classes: The class of each episode.
        spread: The robust spread of each feature, to express the change as a share of capacity.
        n_days: Number of days.
        vectors: For the `stat` arm, each episode's change vector and the day it can be read
            (`StatOutput.vectors`), from which the era ids are drawn. None for the other arms.

    Returns:
        The signal of the arm.
    """
    state = np.zeros(n_days, dtype=np.int8)
    p_era = np.zeros(n_days)
    log_odds = np.full(n_days, np.nan)
    age = np.zeros(n_days)
    era_index = np.zeros(n_days, dtype=np.int64)
    change = np.zeros((n_days, features.N_FEATURES))
    anchor = np.zeros(n_days)
    depth = np.zeros(n_days)
    withdrawn = np.zeros(n_days, dtype=bool)
    persistent_episode = np.full(n_days, -1, dtype=np.int64)
    persistent: list[int] = []
    stacks: list[tuple[int, ...]] = []
    registered: set[int] = set()
    weights = features.BAND_WEIGHTS * spread
    for day in range(n_days):
        for j in [j for j in persistent if _level_returned(episode=detector.episodes[j], day=day)]:
            persistent.remove(j)
            withdrawn[day] = True
        k = int(detector.episode_id[day])
        if k >= 0 and k in classes:
            ep, ec = detector.episodes[k], classes[k]
            i = min(day - ep.alarm_day, len(ec.state) - 1)
            st = int(ec.state[i])
            if st == STATE_ERA and k not in registered:
                registered.add(k)
                persistent.append(k)
            if st == STATE_SWITCH and k in persistent:
                persistent.remove(k)
                withdrawn[day] = True
            state[day], p_era[day], log_odds[day] = st, ec.p_era[i], ec.log_odds[i]
            age[day] = day - ep.onset_day + 1
            vector = _frozen_change(detector=detector, episode=ep, class_day=ec.class_day, day=day)
            change[day] = vector
            if st == STATE_ERA:
                anchor[day] = float(weights @ vector)
            else:
                depth[day] = float(weights @ vector)
        elif persistent:
            ep, ec = detector.episodes[persistent[-1]], classes[persistent[-1]]
            state[day], p_era[day] = STATE_ERA, 1.0
            age[day] = min(day - ep.onset_day + 1, 365)
            vector = _frozen_change(detector=detector, episode=ep, class_day=ec.class_day, day=day)
            change[day], anchor[day] = vector, float(weights @ vector)
        era_index[day] = len(persistent)
        persistent_episode[day] = persistent[-1] if persistent else -1
        stacks.append(tuple(persistent))
    era_id = _era_ids(stacks=stacks, vectors=vectors, n_days=n_days)
    return SiteSignal(
        state=state,
        p_era=p_era,
        log_odds=log_odds,
        age_days=age,
        era_index=era_index,
        change_vector=change,
        anchor=anchor,
        depth=depth,
        episode_id=detector.episode_id,
        persistent_episode=persistent_episode,
        withdrawn=withdrawn,
        episode_bounds=[(e.alarm_day, e.end_day) for e in detector.episodes],
        era_id=era_id,
    )


def _era_ids(
    *,
    stacks: list[tuple[int, ...]],
    vectors: dict[int, tuple[int, np.ndarray]] | None,
    n_days: int,
) -> np.ndarray:
    """Return the era id on each day (zero unless `vectors` is given).

    An era is confirmed on the later of the day it first stands as an era and the day its change
    vector can be read. Days before that keep the earlier era's id and are never relabelled.
    """
    out = np.zeros(n_days, dtype=np.int64)
    if vectors is None:
        return out
    first_seen: dict[int, int] = {}
    for day, stack in enumerate(stacks):
        for episode in stack:
            first_seen.setdefault(episode, day)
    confirmed_on = {
        k: max(day, vectors[k][0]) for k, day in first_seen.items() if vectors[k][0] < n_days
    }
    ids = statdetect.assign_era_ids(
        confirmed=[(k, day, vectors[k][1]) for k, day in confirmed_on.items()]
    )
    for day, stack in enumerate(stacks):
        for episode in reversed(stack):
            if episode in confirmed_on and confirmed_on[episode] <= day:
                out[day] = ids[episode]
                break
    return out


def hmm_signal(*, signal: hmm.HmmSignal, spread: np.ndarray) -> SiteSignal:
    """Express the hidden-Markov arm's output in the same columns as the other arms.

    Args:
        signal: The output of `e00709_hmm.classify_states`.
        spread: The Primary's robust spread.

    Returns:
        A `SiteSignal` with `p_era` set to 1 on `era` days and 0 otherwise.
    """
    n_days = signal.state.shape[0]
    weights = features.BAND_WEIGHTS * spread
    share = signal.change_vector @ weights
    return SiteSignal(
        state=signal.state,
        p_era=(signal.state == STATE_ERA).astype(float),
        log_odds=np.full(n_days, np.nan),
        age_days=signal.dwell_days.astype(float),
        era_index=signal.era_index,
        change_vector=signal.change_vector,
        anchor=np.where(signal.state == STATE_ERA, share, 0.0),
        depth=np.where(signal.state == STATE_SWITCH, share, 0.0),
        episode_id=signal.active_excursion,
        persistent_episode=signal.persistent_excursion,
        withdrawn=np.zeros(n_days, dtype=bool),
        episode_bounds=[(a, b) for a, b, _ in signal.excursions],
        era_id=np.zeros(n_days, dtype=np.int64),
    )


@dataclass(frozen=True)
class SiteResult:
    """Everything the pipeline produces for one Primary.

    Attributes:
        detector: The detector output.
        evidence: The evidence of each episode.
        signals: The per-day signal of each arm.
        stat: The statistics detector's output, or None if the `stat` arm was not computed.
        stat_evidence: The evidence of each statistics-detector episode, by episode number.
    """

    detector: bocpd.DetectorOutput
    evidence: dict[int, EpisodeEvidence]
    signals: dict[ArmType, SiteSignal]
    stat: statdetect.StatOutput | None = None
    stat_evidence: dict[int, EpisodeEvidence] = field(default_factory=dict)


def process_site(
    *,
    site: int,
    all_features: Sequence[features.SiteFeatures],
    mirror: np.ndarray,
    g4: np.ndarray,
    neighbours: dict[int, list[int]],
    weather: features.DailyWeather,
    models: Models,
    arms: Sequence[ArmType] = ("all", "rule", "soft", "hmm", "stat"),
    power: np.ndarray | None = None,
    windows: stats.WindowStats | None = None,
) -> SiteResult:
    """Run the detector, the evidence, and the arms for one Primary.

    A site whose features never become ready returns state `none` on every day.

    Args:
        site: Index of the Primary.
        all_features: Features of every site (the neighbours' readings are needed).
        mirror: This site's experiment 687 features at the end of every day.
        g4: Gate G4 of this site on every day.
        neighbours: Electrical neighbours of each site.
        weather: ERA5 summaries.
        models: The frozen detector threshold, weights, and classifiers.
        arms: The arms to compute.
        power: This site's power over capacity, needed by the `stat` arm.
        windows: This site's window statistics (computed from `power` if omitted).

    Returns:
        The site's result.

    Raises:
        ValueError: If the `stat` arm is requested without `power`.
    """
    feat = all_features[site]
    n_days = feat.raw.shape[0]
    detector = bocpd.detect(
        scaled=feat.scaled, spread=feat.spread, start_day=feat.ready_day, threshold=models.threshold
    )
    evidence = {
        e.episode: episode_evidence(
            episode=e,
            detector=detector,
            feat=feat,
            mirror=mirror,
            g4=g4,
            site=site,
            neighbours=neighbours,
            all_features=all_features,
            weather=weather,
            survival=models.survival,
        )
        for e in detector.episodes
    }
    signals: dict[ArmType, SiteSignal] = {}
    stat_output: statdetect.StatOutput | None = None
    stat_evidence: dict[int, EpisodeEvidence] = {}
    if ("stat" in arms or "stat_noshape" in arms) and feat.ready_day < n_days:
        if power is None:
            msg = "the stat arm needs the site's power"
            raise ValueError(msg)
        stat_output = statdetect.detect_stat(
            windows=windows if windows is not None else stats.window_statistics(power_site=power),
            scaled=feat.scaled,
            threshold=models.stat_threshold,
        )
        stat_evidence = stat_episode_evidence(
            output=stat_output, power_site=power, feat=feat, mirror=mirror, g4=g4, site=site,
            neighbours=neighbours, all_features=all_features, weather=weather,
            survival=models.survival,
        )  # fmt: skip
    for arm in arms:
        if arm in ("stat", "stat_noshape"):
            if stat_output is not None:
                classes = {
                    e.episode: classify_episode(
                        evidence=stat_evidence[e.episode], episode=e, arm=arm, models=models
                    )
                    for e in stat_output.detector.episodes
                }
                signals[arm] = compose_site_signal(
                    detector=stat_output.detector, classes=classes, spread=feat.spread,
                    n_days=n_days, vectors=stat_output.vectors,
                )  # fmt: skip
            continue
        if arm == "hmm":
            params = models.hmm_parameters.get(site)
            if params is None or feat.ready_day >= n_days:
                continue
            probs = hmm.forward_filter(scaled=feat.scaled, params=params, start_day=feat.ready_day)
            states = hmm.classify_states(
                probabilities=probs,
                params=params,
                train_end_day=models.train_end_day,
                start_day=feat.ready_day,
            )
            signals["hmm"] = hmm_signal(signal=states, spread=feat.spread)
            continue
        classes = {
            e.episode: classify_episode(
                evidence=evidence[e.episode], episode=e, arm=arm, models=models
            )
            for e in detector.episodes
        }
        signals[arm] = compose_site_signal(
            detector=detector, classes=classes, spread=feat.spread, n_days=n_days
        )
    return SiteResult(
        detector=detector,
        evidence=evidence,
        signals=signals,
        stat=stat_output,
        stat_evidence=stat_evidence,
    )


def fit_hmm_parameters(
    *, all_features: Sequence[features.SiteFeatures], train_end_day: int
) -> dict[int, hmm.HmmParameters]:
    """Fit the sticky hidden-Markov model of every site that has enough clean days.

    Args:
        all_features: Features of every site (real, uninjected).
        train_end_day: First day that the fit must not read.

    Returns:
        The parameters of each site whose detector is ready by 365 days before the cut.
    """
    out = {}
    for site, feat in enumerate(all_features):
        if feat.ready_day + features.SCALE_WINDOW_DAYS < train_end_day:
            out[site] = hmm.fit_hmm(
                scaled=feat.scaled, train_end_day=train_end_day, start_day=feat.ready_day
            )
    return out


def save_models(*, models: Models, folder: Path) -> None:
    """Write the frozen models to `models.npz` (arrays) and `models.json` (scalars).

    Phase B reads these files instead of tuning again.

    Args:
        models: The tuned models.
        folder: Folder that receives the two files.
    """
    folder.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "rule_weights": models.rule_weights,
        "survival_durations": models.survival.durations,
        "stat_weights": models.stat_weights,
        "stat_noshape_weights": models.stat_noshape_weights,
    }
    scalars: dict[str, object] = {
        "threshold": models.threshold,
        "train_end_day": models.train_end_day,
        "hmm_sites": sorted(models.hmm_parameters),
        "soft": models.soft is not None,
        "stat_threshold": models.stat_threshold,
    }
    if models.soft is not None:
        soft = models.soft
        arrays |= {
            "soft_mean": soft.mean, "soft_scale": soft.scale, "soft_w1": soft.w1,
            "soft_b1": soft.b1, "soft_w2": soft.w2,
        }  # fmt: skip
        scalars |= {"soft_b2": soft.b2, "soft_platt": list(soft.platt)}
    for site, params in models.hmm_parameters.items():
        arrays[f"hmm_{site}_means"] = params.means
        arrays[f"hmm_{site}_variances"] = params.variances
        arrays[f"hmm_{site}_initial"] = params.initial
    np.savez(folder / "models.npz", **arrays)  # ty: ignore[invalid-argument-type]
    (folder / "models.json").write_text(json.dumps(scalars, indent=2))


def load_models(*, folder: Path) -> Models:
    """Read the frozen models that `save_models` wrote.

    Args:
        folder: Folder that holds `models.npz` and `models.json`.

    Returns:
        The models.
    """
    scalars = json.loads((folder / "models.json").read_text())
    with np.load(folder / "models.npz") as arrays:
        soft = None
        if scalars["soft"]:
            soft = SoftClassifier(
                mean=arrays["soft_mean"], scale=arrays["soft_scale"], w1=arrays["soft_w1"],
                b1=arrays["soft_b1"], w2=arrays["soft_w2"], b2=float(scalars["soft_b2"]),
                platt=(float(scalars["soft_platt"][0]), float(scalars["soft_platt"][1])),
            )  # fmt: skip
        return Models(
            threshold=float(scalars["threshold"]),
            rule_weights=arrays["rule_weights"],
            survival=SwitchSurvival(durations=arrays["survival_durations"]),
            train_end_day=int(scalars["train_end_day"]),
            soft=soft,
            stat_threshold=float(scalars.get("stat_threshold", statdetect.DEFAULT_STAT_THRESHOLD)),
            stat_weights=(
                arrays["stat_weights"]
                if "stat_weights" in arrays.files
                else np.zeros(N_STAT_COLUMNS)
            ),
            stat_noshape_weights=(
                arrays["stat_noshape_weights"]
                if "stat_noshape_weights" in arrays.files
                else np.zeros(N_STAT_COLUMNS)
            ),
            hmm_parameters={
                int(site): hmm.HmmParameters(
                    means=arrays[f"hmm_{site}_means"],
                    variances=arrays[f"hmm_{site}_variances"],
                    initial=arrays[f"hmm_{site}_initial"],
                )
                for site in scalars["hmm_sites"]
            },
        )

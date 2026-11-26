# ruff: noqa: E501
"""Experiment 732: the baselines B-PV0, B-PV1, B-PV2 (solar output without a solar head), and B-T and B-TN (step rules).

This module imports no torch at the top. B-PV1 and B-PV2 read stored probabilities and demand
outputs of experiment 730 and, on the injection overlay, the outputs of experiment 730's stored
networks (`infer_730`, which imports torch inside the function and only reads the state files).
B-PV0 is a least-squares fit on the validation bank. B-T and B-TN are logistic regressions fitted
on the validation overlay with Newton steps. Their inputs are step statistics in noise units: the
mean of a series over the next 2 hours minus its mean over the previous 2 hours, after the series'
daily shape is removed, divided by the series' noise unit (`e00732_neighbours`). Nothing here
reads a test score.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import e00732_arms as arms
import e00732_core as core
import e00732_neighbours as nb
import e00732_overlays as ov
import numpy as np

# --------------------------------------------------------------------------- B-PV1 and B-PV2


def signed_reading_estimate(*, p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return X-hat: -M where P(X<0) is above 0.5 and +M otherwise, with M = sqrt(max(R^2 - Q-hat^2, 0)).

    Args:
        p: The seed-mean P(X<0) of each row.
        r: The reading R of each row, in points.

    Returns:
        The signed reading estimate, in points.
    """
    m = np.sqrt(np.maximum(r**2 - core.Q_HAT_POINTS**2, 0.0))
    return np.where(p > 0.5, -m, m)


def solar_from_demand(
    *, demand: np.ndarray, p: np.ndarray, r: np.ndarray, cosz: np.ndarray
) -> np.ndarray:
    """Return max(0, demand - X-hat) on sun-up half-hours and 0 elsewhere, for one seed (or a seed mean).

    Args:
        demand: A demand estimate, in points (experiment 730's demand output, or D8's climatology median).
        p: P(X<0) of the same rows.
        r: The reading R.
        cosz: The cosine of the solar zenith angle.

    Returns:
        The solar estimate in points; zero where the sun is not up (night, dawn, and dusk).
    """
    est = np.maximum(demand - signed_reading_estimate(p=p, r=r), 0.0)
    return np.where(cosz > core.SUN_UP_COSZ, est, 0.0)


def b_pv2(
    *,
    demand_by_seed: Sequence[np.ndarray],
    p_by_seed: Sequence[np.ndarray],
    r: np.ndarray,
    cosz: np.ndarray,
) -> np.ndarray:
    """Return B-PV2: the rule per seed of SD*, averaged over seeds 0 to 2 (it fails on a seed count that is not 3)."""
    if len(demand_by_seed) != len(core.SEEDS) or len(p_by_seed) != len(core.SEEDS):
        msg = "B-PV2 needs the three seeds of SD* (a missing seed makes the baseline void)"
        raise ValueError(msg)
    return np.mean(
        [
            solar_from_demand(demand=d, p=p, r=r, cosz=cosz)
            for d, p in zip(demand_by_seed, p_by_seed, strict=True)
        ],
        axis=0,
    )


def b_pv1(
    *, demand_median: np.ndarray, p_by_seed: Sequence[np.ndarray], r: np.ndarray, cosz: np.ndarray
) -> np.ndarray:
    """Return B-PV1: the same rule with the climatology median in place of the demand output, on S's P(X<0)."""
    if len(p_by_seed) != len(core.SEEDS):
        msg = "B-PV1 needs the three seeds of S"
        raise ValueError(msg)
    return np.mean(
        [solar_from_demand(demand=demand_median, p=p, r=r, cosz=cosz) for p in p_by_seed], axis=0
    )


@dataclass(frozen=True)
class Outputs730:
    """The outputs of experiment 730's stored networks on one member of the windows, per seed.

    Attributes:
        p: P(X<0) of each seed, flat on the valid half-hours.
        demand: The demand output of each seed (None for arm S, which has none).
        r: The member's reading in points, flat on the same half-hours.
        cosz: The cosine of the zenith angle on the same half-hours.
    """

    p: Sequence[np.ndarray]
    demand: Sequence[np.ndarray] | None
    r: np.ndarray
    cosz: np.ndarray


def b_pv_member(*, outputs: Outputs730, demand_median: np.ndarray | None = None) -> np.ndarray:
    """Return B-PV2 (when `outputs` holds a demand output) or B-PV1 (with `demand_median`) on one member of the windows.

    The plain member and the injected member each give one estimate, so the recovery of a rule on the
    injection overlay is the difference of the two estimates against delta, exactly as for a head.
    """
    if outputs.demand is not None:
        return b_pv2(
            demand_by_seed=outputs.demand, p_by_seed=outputs.p, r=outputs.r, cosz=outputs.cosz
        )
    if demand_median is None:
        msg = "B-PV1 needs the demand climatology's median"
        raise ValueError(msg)
    return b_pv1(demand_median=demand_median, p_by_seed=outputs.p, r=outputs.r, cosz=outputs.cosz)


def infer_730(
    *,
    state_path: Path,
    demand: bool,
    prepared: Sequence[arms.Prepared],
    index: ov.PairIndex | None = None,
    amount: float | None = None,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict[str, object]:
    """Run one stored network of experiment 730 on this experiment's windows (inference only, read only).

    Experiment 730's networks read the 16 base channels of the window and no neighbour channel, so the
    windows are built without neighbours. With `amount`, the injected member is built as well and
    predicted by the same network.

    Args:
        state_path: The state file of experiment 730's network (`networks/<name>.pt`), never written.
        demand: Whether the network is a demand-output arm (SD*), which has three output channels.
        prepared: The prepared segments of the bank or overlay.
        index: The pair index of the bank (needed with `amount`).
        amount: The injected amount of the test pair, or None for the plain member only.
        device: "cuda" or "cpu".
        batch_size: Windows per forward pass.

    Returns:
        "plain" and "inj" (None without `amount`): the decoded arrays `(P, 48)` (`p`, and
        `demand_mu` for a demand arm); "t" and "ti": the targets; "pair" and "delta".
    """
    import e00730_arms as arms730
    import e00730_nets as nets730
    import torch

    spec730 = arms730.ArmSpec("SD-R" if demand else "S", "d", demand=demand, style=core.STYLE)
    net = nets730.build_net(arch="unet", spec=spec730)
    net.load_state_dict(torch.load(state_path, map_location=device))
    net.to(device).eval()
    spec = arms.eval_spec(inject=amount is not None, neighbours=False)
    picks = arms.all_picks(prepared=prepared)
    outs: dict[str, list] = {k: [] for k in ("plain", "inj", "t", "ti", "pair", "delta")}
    middle = slice(arms.MID, arms.MID + 48)
    with torch.no_grad():
        for i in range(0, len(picks), batch_size):
            batch = arms.make_eval_batch(
                prepared=prepared,
                picks=picks[i : i + batch_size],
                spec=spec,
                index=index,
                amount=amount,
            )
            for member, key in ((batch.x, "plain"), (batch.xi, "inj")):
                if member is None:
                    continue
                raw = net(torch.from_numpy(member).to(device))[:, :, middle]
                decoded = nets730.decode(raw=raw)
                outs[key].append({k: v.float().cpu().numpy() for k, v in decoded.items()})
            outs["t"].append(batch.t)
            if batch.ti is not None:
                outs["ti"].append(batch.ti)
            outs["pair"].append(batch.pair)
            outs["delta"].append(batch.delta)

    def join(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray] | None:
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None

    return {
        "plain": join(outs["plain"]),
        "inj": join(outs["inj"]),
        "t": np.concatenate(outs["t"]),
        "ti": np.concatenate(outs["ti"]) if outs["ti"] else None,
        "pair": np.concatenate(outs["pair"]),
        "delta": np.concatenate(outs["delta"]),
    }


# --------------------------------------------------------------------------- B-PV0


PV0_NAMES: Final = (
    "csi_x_clear",
    "clear",
    "cosz",
    "csi_x_clear_x_clear",
    "csi_x_clear_x_cosz",
    "clear_x_cosz",
)
"""B-PV0's six inputs: the three sky quantities and their three pairwise products."""


def pv0_features(*, csi: np.ndarray, clear: np.ndarray, cosz: np.ndarray) -> np.ndarray:
    """Return B-PV0's inputs `(..., 6)`: the CAMS index times the clear-sky irradiance, the irradiance, the cosine of the zenith angle, and their pairwise products."""
    a = csi * clear
    return np.stack([a, clear, cosz, a * clear, a * cosz, clear * cosz], axis=-1)


def fit_pv0(
    *, csi: np.ndarray, clear: np.ndarray, cosz: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Return B-PV0's weights (the intercept first): a least-squares fit of the solar target on the sky alone.

    Fit it on the validation bank's valid half-hours, all world types. B-PV0 reads no reading, so it
    cannot know how much solar capacity a window holds: it gives the average solar output of the bank
    for the window's sky.

    Args:
        csi: The CAMS clear-sky index (clipped as the input channel is), flat.
        clear: The clear-sky irradiance over 1,000, flat.
        cosz: The cosine of the zenith angle, flat.
        target: The solar target in points, flat.

    Returns:
        The seven weights.
    """
    keep = np.isfinite(target) & np.isfinite(csi) & np.isfinite(clear) & np.isfinite(cosz)
    if not keep.any() or not np.any(target[keep] > 0.0):
        msg = "B-PV0 needs validation half-hours with solar output"
        raise ValueError(msg)
    z = np.column_stack(
        [np.ones(int(keep.sum())), pv0_features(csi=csi[keep], clear=clear[keep], cosz=cosz[keep])]
    )
    return np.linalg.lstsq(z, target[keep], rcond=None)[0]


def b_pv0(
    *, weights: np.ndarray, csi: np.ndarray, clear: np.ndarray, cosz: np.ndarray
) -> np.ndarray:
    """Return B-PV0's solar estimate in points: the fitted rule clipped at zero, and zero at night."""
    est = weights[0] + pv0_features(csi=csi, clear=clear, cosz=cosz) @ weights[1:]
    return np.where(cosz > core.NIGHT_COSZ, np.maximum(est, 0.0), 0.0)


def sky_of_windows(*, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the CAMS index, the clear-sky irradiance, and the cosine of the zenith angle of the scored day of plain windows.

    Args:
        x: Plain windows `(B, C, 96)` (the sky channels are those of experiment 730's stack).

    Returns:
        Three `(B, 48)` arrays.
    """
    s = slice(arms.MID, arms.MID + 48)
    return x[:, arms.ROW["csi"], s], x[:, arms.ROW["clear"], s], x[:, arms.ROW["cosz"], s]


# --------------------------------------------------------------------------- step statistics


def weekend_flags(*, dates: np.ndarray, holidays: np.ndarray) -> np.ndarray:
    """Return whether each day is a weekend day or a bank holiday (the two daily shapes of the step statistic)."""
    days = dates.astype("datetime64[D]")
    weekday = (days.astype(np.int64) + 3) % 7  # 1970-01-01 was a Thursday
    return (weekday >= 5) | np.isin(days, holidays.astype("datetime64[D]"))


@dataclass(frozen=True)
class StepScales:
    """The daily shapes and the noise units that turn a series into step statistics in noise units.

    Attributes:
        shapes: The daily shape `(2, 48)` of each series (weekdays, then weekends), from the training
            base days alone: "D8", "BSP", "D3", "D5", and "D14".
        units: The noise unit in points of D8, D3, D5, and D14 (`limits.json` holds them). The BSP
            meter's step is read in D8's noise unit, because both are in points.
    """

    shapes: Mapping[str, np.ndarray]
    units: Mapping[str, float]


@dataclass(frozen=True)
class StepSeries:
    """The series of one segment (or one real month) on its dense axis, for the step statistics.

    Attributes:
        reading: D8's reading in points `(n, 48)`, NaN where absent.
        bsp: The BSP meter in points, NaN where the meter has no data.
        cosz: The cosine of the zenith angle.
        neighbours: Each neighbour's folded reading in points, NaN where absent.
        weekend: Whether each day is a weekend day or a bank holiday `(n,)`.
        dates: The days, `datetime64[D]`, consecutive.
    """

    reading: np.ndarray
    bsp: np.ndarray
    cosz: np.ndarray
    neighbours: Mapping[str, np.ndarray]
    weekend: np.ndarray
    dates: np.ndarray


def compute_scales(
    *,
    reading: np.ndarray,
    bsp: np.ndarray,
    neighbours: Mapping[str, np.ndarray],
    dates: np.ndarray,
    weekend: np.ndarray,
    train_rows: np.ndarray,
) -> StepScales:
    """Compute the daily shapes and the four noise units from the training base days alone.

    Args:
        reading: D8's real reading on the base calendar `(n, 48)`.
        bsp: The real BSP meter on the same calendar (NaN without data).
        neighbours: D3, D5, and D14 on the same calendar in points (D5 and D14 signed, and read here
            as magnitudes, because every neighbour enters as a magnitude).
        dates: The base days, sorted.
        weekend: Weekend or bank-holiday flags `(n,)`.
        train_rows: The training base days `(n,)` bool.

    Returns:
        The scales. A BSP meter with no training value for a half-hour of a daily shape gets a zero
        shape (the meter is absent in most months).
    """
    series = {"D8": reading, **{n: np.abs(neighbours[n]) for n in nb.NEIGHBOURS}}
    shapes: dict[str, np.ndarray] = {}
    units: dict[str, float] = {}
    for name, values in series.items():
        shape = nb.daily_shape(values=values, weekend=weekend, train_rows=train_rows)
        shapes[name] = shape
        stat = nb.step_statistic(values=values, dates=dates, weekend=weekend, shape=shape)
        units[name] = nb.noise_unit(statistic=stat, train_rows=train_rows)
    try:
        shapes["BSP"] = nb.daily_shape(values=bsp, weekend=weekend, train_rows=train_rows)
    except nb.NeighbourError:
        shapes["BSP"] = np.zeros((2, nb.HALF_HOURS))
    return StepScales(shapes=shapes, units=units)


def series_of_segment(
    *,
    seg: object,
    neighbours: Mapping[str, np.ndarray],
    floors: Mapping[str, float],
    holidays: np.ndarray,
) -> StepSeries:
    """Return the step series of a segment (or a real pseudo-segment) with its neighbours' series.

    Args:
        seg: The segment, after any transfer overlay: its reading, BSP meter, and sky are used.
        neighbours: Each neighbour's series on the segment's dense axis in points, signed for D5 and
            D14 and after the mirrors of the transfers (`TransferOverlay.neighbours`); an empty
            mapping means no neighbour holds a reading.
        floors: The neighbours' fold floors in points (`TransferOverlay.floors`).
        holidays: Bank-holiday dates.

    Returns:
        The series. Each neighbour is folded with its floor, so it enters as a magnitude.
    """
    reading = np.asarray(seg.r, dtype=np.float64)  # ty: ignore[unresolved-attribute]
    folded = {
        n: nb.fold(signed=np.asarray(neighbours[n], dtype=np.float64), floor=floors.get(n, 0.0))
        if n in neighbours
        else np.full(reading.shape, np.nan)
        for n in nb.NEIGHBOURS
    }
    return StepSeries(
        reading=reading,
        bsp=np.asarray(seg.bsp, dtype=np.float64),  # ty: ignore[unresolved-attribute]
        cosz=np.asarray(seg.cosz, dtype=np.float64),  # ty: ignore[unresolved-attribute]
        neighbours=folded,
        weekend=weekend_flags(dates=seg.dates, holidays=holidays),  # ty: ignore[unresolved-attribute]
        dates=seg.dates,  # ty: ignore[unresolved-attribute]
    )


def series_of_overlay(*, overlay: ov.TransferOverlay, holidays: np.ndarray) -> StepSeries:
    """Return the step series of a transfer overlay (the mirrors already applied to the neighbours)."""
    return series_of_segment(
        seg=overlay.seg, neighbours=overlay.neighbours, floors=overlay.floors, holidays=holidays
    )


@dataclass(frozen=True)
class StepStatistics:
    """The step statistics of one series set, in noise units (NaN where a window lacks data).

    Attributes:
        d8: D8's reading `(n, 48)`.
        bsp: The BSP meter, in D8's noise unit.
        neighbours: Each neighbour, in its own noise unit.
    """

    d8: np.ndarray
    bsp: np.ndarray
    neighbours: Mapping[str, np.ndarray]


def step_statistics(*, series: StepSeries, scales: StepScales) -> StepStatistics:
    """Return the step statistics in noise units of one series set, the daily shapes removed."""

    def one(values: np.ndarray, name: str, unit: str) -> np.ndarray:
        stat = nb.step_statistic(
            values=values, dates=series.dates, weekend=series.weekend, shape=scales.shapes[name]
        )
        return stat / scales.units[unit]

    return StepStatistics(
        d8=one(series.reading, "D8", "D8"),
        bsp=one(series.bsp, "BSP", "D8"),
        neighbours={n: one(series.neighbours[n], n, n) for n in nb.NEIGHBOURS},
    )


BT_FEATURES: Final = ("abs_step", "abs_bsp_step", "abs_step_minus_bsp_step", "cosz")
"""B-T's four inputs: D8's step, the BSP meter's step, their difference (all absolute, in D8's noise units), and the cosine of the zenith angle."""
BTN_EXTRA_FEATURES: Final = (
    "aligned_D3",
    "aligned_D5",
    "aligned_D14",
    "largest_opposite_step",
    "same_sign_count",
)
"""B-TN's five extra inputs: each neighbour's step times the sign of D8's step, the largest opposite-sign step, and the same-sign count."""
BTN_FEATURES: Final = (*BT_FEATURES, *BTN_EXTRA_FEATURES)
SAME_SIGN_NOISE_UNITS: Final = 2.0
"""A neighbour counts as stepping with D8 when its step is at least this many noise units and has D8's sign (experiment 70)."""


def step_features(*, stats: StepStatistics, cosz: np.ndarray, neighbours: bool) -> np.ndarray:
    """Return B-T's (or B-TN's) inputs `(n, 48, F)` from step statistics in noise units.

    A missing statistic (NaN) gives 0 in its inputs. Where the BSP meter has no data, the two BSP
    inputs are 0.

    Args:
        stats: The step statistics.
        cosz: The cosine of the zenith angle `(n, 48)`.
        neighbours: Whether to add B-TN's neighbour inputs.

    Returns:
        The inputs, in the order of `BT_FEATURES` and then `BTN_EXTRA_FEATURES`.
    """
    s8 = np.nan_to_num(stats.d8, nan=0.0)
    sb = np.nan_to_num(stats.bsp, nan=0.0)
    columns = [np.abs(s8), np.abs(sb), np.abs(s8 - sb), cosz]
    if neighbours:
        sign = np.sign(s8)
        aligned = np.stack(
            [np.nan_to_num(stats.neighbours[n], nan=0.0) * sign for n in nb.NEIGHBOURS]
        )
        columns += list(aligned)
        columns.append(np.maximum(0.0, -aligned.min(axis=0)))
        columns.append((aligned >= SAME_SIGN_NOISE_UNITS).sum(axis=0).astype(np.float64))
    return np.stack(columns, axis=-1)


def step_feature_rows(
    *,
    series: Sequence[StepSeries],
    prepared: Sequence[arms.Prepared],
    scales: StepScales,
    neighbours: bool,
) -> np.ndarray:
    """Return the rule's inputs `(P, 48, F)` for every scored day, in the order of `arms.all_picks`.

    Args:
        series: The step series of the bank's overlays, in the order of the bank's segments.
        prepared: The prepared segments (`source_index` gives the segment's position in `series`).
        scales: The shapes and noise units from the training base days.
        neighbours: B-TN (True) or B-T (False).

    Returns:
        One row of 48 half-hours per scored day.
    """
    cache: dict[int, np.ndarray] = {}
    rows = []
    for p in prepared:
        k = p.source_index
        if k not in cache:
            cache[k] = step_features(
                stats=step_statistics(series=series[k], scales=scales),
                cosz=series[k].cosz,
                neighbours=neighbours,
            )
        rows.extend(cache[k][d] for d in p.days)
    width = len(BTN_FEATURES) if neighbours else len(BT_FEATURES)
    return np.stack(rows) if rows else np.zeros((0, 48, width))


# --------------------------------------------------------------------------- the logistic rule


def fit_logistic(
    *, features: np.ndarray, y: np.ndarray, ridge: float = 1e-3, steps: int = 50
) -> np.ndarray:
    """Return the weights (bias first) of a ridge-penalised logistic regression fitted with Newton steps.

    Args:
        features: `(N, F)`.
        y: 0 or 1 labels `(N,)`.
        ridge: The penalty on the weights (not the bias).
        steps: The Newton steps.

    Returns:
        The `F + 1` weights.
    """
    if features.shape[0] == 0 or not (0.0 < float(np.mean(y)) < 1.0):
        msg = "the step rule needs rows with and without a step"
        raise ValueError(msg)
    mu, sd = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-9)
    z = np.column_stack([np.ones(len(y)), (features - mu) / sd])
    w = np.zeros(z.shape[1])
    pen = np.full(z.shape[1], ridge)
    pen[0] = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))
        grad = z.T @ (p - y) + pen * w
        hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.abs(step).max()) < 1e-8:
            break
    return np.concatenate([w[:1] - (w[1:] * mu / sd).sum(), w[1:] / sd])


def predict_logistic(*, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the logistic probability of each row of `features`."""
    return 1.0 / (1.0 + np.exp(-np.clip(weights[0] + features @ weights[1:], -30, 30)))


def fit_step_rule(*, features: np.ndarray, event: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fit B-T or B-TN on the validation overlay: the step event of each valid half-hour on its inputs.

    Args:
        features: `(P, 48, F)` from `step_feature_rows` on the validation overlay.
        event: The step event `(P, 48)`.
        valid: Valid half-hours `(P, 48)`.

    Returns:
        The weights. The fit fails when the validation overlay holds no step or no half-hour without one.
    """
    return fit_logistic(features=features[valid], y=(event[valid] > 0.5).astype(np.float64))


def predict_step_rule(*, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the rule's step probability `(P, 48)`."""
    return predict_logistic(features=features, weights=weights)

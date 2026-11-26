# ruff: noqa: E501
"""Experiment 733: the re-run of the untangling network on days D - 2 and D - 1 of each origin.

The stored outputs of experiment 732 cover every UTC day whose 48-hour window ends before the
forecaster's visibility limit. For an origin on day D, the windows of UTC days D - 2 and D - 1 end
after the limit (09:00 UTC on day D - 1, which is 27 hours before the 12:00 UTC contract origin), so
the network runs again on those two windows with every input after the limit hidden:

- D8's reading and its presence mask, the three neighbours (hidden wherever D8 is), and the BSP step
  and its mask are set to zero, which is the hidden-span input that experiment 732 trained on;
- the CAMS clear-sky index and its presence mask are set to zero (the geometry columns stay);
- the ERA5 temperature, wind speed, and precipitation are replaced by the mean of the ensemble
  forecast: for a valid time up to the native origin (09:00 UTC on day D) the 00Z run of day D - 1,
  and after the native origin the 00Z run of day D. Each is the latest run that is available at the
  native origin and whose first stored valid time lies before the valid time it fills.

The pure functions in this module need no torch and no experiment 732. The functions of the section
"Real adapters" import experiment 732's code and torch when they are called.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_paths as paths  # noqa: F401
import numpy as np
from e00733_channels import GRID_START_EPOCH_DAY

STEPS: Final = core.WINDOW_STEPS
MID: Final = core.WINDOW_MIDDLE_START
RERUN_OFFSETS: Final = (2, 1)
"""The re-run windows of an origin on day D are those of UTC days D - 2 and D - 1."""
LAST_VISIBLE_SLOT_OF_D_MINUS_1: Final = 17
"""The last slot of UTC day D - 1 whose reading is visible: slot 17 (08:30 to 09:00 UTC, stamped 09:00)."""
MM_PER_HOUR_PER_KG_M2_S: Final = 3600.0
"""The ensemble's precipitation is in kg m-2 s-1; the network's ERA5 channel is in millimetres per hour."""
PRECIP_SCALE: Final = 0.5
"""The precipitation channel is `log1p(millimetres per hour) / PRECIP_SCALE`, as experiment 729 builds it."""
US_PER_HOUR: Final = 3_600 * 1_000_000
US_PER_DAY: Final = 24 * US_PER_HOUR
US_PER_STEP: Final = US_PER_HOUR // 2
ENS_COLUMNS: Final = {
    "temperature": "mean_temperature_2m", "wind": "mean_wind_speed_10m", "precip": "mean_precipitation",
}  # fmt: skip
"""The columns of an ensemble run summary that replace the three ERA5 channels."""
POISON_VALUES: Final = (float("nan"), 1e6)
HIDDEN_ROWS: Final = ("r", "r_mask", "csi", "cams_mask", "d_bsp", "bsp_mask")
"""The base channels that are zero at a hidden position."""
ERA5_ROWS: Final = ("t2m", "wind10", "precip")


class RerunError(RuntimeError):
    """A re-run check failed (a correctness failure, verdict row 1)."""


# --------------------------------------------------------------------------- the visibility limit on a window


def last_visible_abs_slot(*, origin_day: int) -> int:
    """Return the last visible half-hour of an origin on grid day `origin_day`.

    As a slot counted from midnight of day 0.

    The newest visible stamp is `c - 54` with `c = 48 D + 24`, and slot `a` carries stamp `a + 1`, so
    slot `48 (D - 1) + 17` is the last visible one.
    """
    c = origin_day * core.STEPS_PER_DAY + 24
    return c - core.NEWEST_VISIBLE_BEFORE_CONTRACT - 1


def window_abs_slots(*, day: int) -> np.ndarray:
    """Return the absolute slot of each of the 96 columns of the window of UTC day `day`.

    Slots are counted from midnight of day 0.

    A window runs from 12:00 UTC on the day before `day` to 12:00 UTC on the day after: column `k`
    is slot `48 (day - 1) + 24 + k`.
    """
    return (day - 1) * core.STEPS_PER_DAY + MID + np.arange(STEPS)


def window_hidden(*, origin_day: int, day: int) -> np.ndarray:
    """Return which of the 96 columns of the window of UTC day `day` lie after the visibility limit of `origin_day`."""
    return window_abs_slots(day=day) > last_visible_abs_slot(origin_day=origin_day)


def visible_middle(*, origin_day: int, day: int) -> np.ndarray:
    """Return which of the 48 scored half-hours of UTC day `day` are visible to an origin on `origin_day`."""
    return ~window_hidden(origin_day=origin_day, day=day)[MID : MID + core.STEPS_PER_DAY]


def assert_limit_matches_layout(*, origin_day: int) -> None:
    """Check that the limit of this module is experiment 713's `c - 54` rule, on one origin.

    Args:
        origin_day: The grid day number of the origin.

    Raises:
        RerunError: When a slot is visible here and not under the rule, or the reverse.
    """
    c = origin_day * core.STEPS_PER_DAY + 24
    slots = np.arange(
        origin_day * core.STEPS_PER_DAY - 5 * core.STEPS_PER_DAY,
        origin_day * core.STEPS_PER_DAY + 1,
    )
    by_rule = (slots + 1) <= c - core.NEWEST_VISIBLE_BEFORE_CONTRACT
    here = slots <= last_visible_abs_slot(origin_day=origin_day)
    if not np.array_equal(by_rule, here) or not by_rule.any() or by_rule.all():
        msg = "the visibility limit of the re-run differs from experiment 713's rule"
        raise RerunError(msg)


# --------------------------------------------------------------------------- hiding


def hide_window(
    *, window: np.ndarray, rows: Mapping[str, int], hidden: np.ndarray,
    era5: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:  # fmt: skip
    """Hide every input after the limit in a base-channel window, in place.

    The reading, the CAMS clear-sky index, the BSP step, and their masks are set to zero at the
    hidden columns. The ERA5 rows at the hidden columns are replaced by `era5` (zero where `era5`
    holds NaN, which is also the value of weather that ERA5 lacks). The neighbour channels are not in
    this window; the caller builds them from the hidden window and the same `hidden`.

    Args:
        window: The base channels `(C, 96)`.
        rows: The row index of each named channel.
        hidden: The hidden columns `(96,)`.
        era5: The replacement of `t2m`, `wind10`, and `precip`, each `(96,)` already in the units
            of the channel, or None to zero them.

    Returns:
        The same array.
    """
    if not hidden.any():
        return window
    for name in HIDDEN_ROWS:
        window[rows[name], hidden] = 0.0
    for name in ERA5_ROWS:
        values = (
            np.zeros(STEPS)
            if era5 is None
            else np.nan_to_num(np.asarray(era5[name], dtype=np.float64), nan=0.0)
        )
        window[rows[name], hidden] = values[hidden]
    return window


def standardise_ens(
    *, temperature: np.ndarray, wind: np.ndarray, precip_kg_m2_s: np.ndarray, t_mean: float, t_sd: float, w_mean: float, w_sd: float
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return the ensemble means in the units of the network's ERA5 channels.

    Temperature and wind speed are standardised with the training days' mean and standard deviation
    (experiment 729's `InputStats`); precipitation is `log1p(mm per hour) / 0.5`.
    """
    mm_per_hour = np.asarray(precip_kg_m2_s, dtype=np.float64) * MM_PER_HOUR_PER_KG_M2_S
    return {
        "t2m": (np.asarray(temperature, dtype=np.float64) - t_mean) / t_sd,
        "wind10": (np.asarray(wind, dtype=np.float64) - w_mean) / w_sd,
        "precip": np.log1p(np.clip(mm_per_hour, 0.0, None)) / PRECIP_SCALE,
    }


# --------------------------------------------------------------------------- ensemble runs


@dataclass(frozen=True)
class EnsRun:
    """The ensemble-mean weather of one 00Z run at D8's cell, on the half-hourly valid times.

    Attributes:
        init_day: The grid day number of the run's 00:00 UTC initialisation.
        valid_us: The valid-time stamps (period ends) in microseconds since the epoch, sorted.
        temperature: The 2 m temperature in degrees Celsius.
        wind: The 10 m wind speed in metres per second.
        precip: The precipitation in kg m-2 s-1.
    """

    init_day: int
    valid_us: np.ndarray
    temperature: np.ndarray
    wind: np.ndarray
    precip: np.ndarray


def read_ens_run(*, summary_dir: Path, run_epoch_day: int, series_id: int) -> EnsRun:
    """Read one run's summary file for one series.

    Args:
        summary_dir: The folder of `init=<date>.parquet` files.
        run_epoch_day: The epoch day of the run's 00:00 UTC initialisation.
        series_id: The `time_series_id` of D8.

    Returns:
        The run.

    Raises:
        RerunError: When the file is missing or holds no row of the series.
    """
    import polars as pl

    day = date(1970, 1, 1) + timedelta(days=run_epoch_day)
    path = summary_dir / f"init={day.isoformat()}.parquet"
    if not path.exists():
        msg = "an ensemble run summary that the re-run needs is missing"
        raise RerunError(msg)
    frame = (
        pl.read_parquet(path, columns=["time_series_id", "valid_time", *ENS_COLUMNS.values()])
        .filter(pl.col("time_series_id") == series_id)
        .sort("valid_time")
    )
    if frame.is_empty():
        msg = "an ensemble run summary holds no row of D8"
        raise RerunError(msg)
    return EnsRun(
        init_day=run_epoch_day - GRID_START_EPOCH_DAY,
        valid_us=frame["valid_time"].dt.epoch("us").to_numpy().astype(np.int64),
        temperature=frame[ENS_COLUMNS["temperature"]].to_numpy().astype(np.float64),
        wind=frame[ENS_COLUMNS["wind"]].to_numpy().astype(np.float64),
        precip=frame[ENS_COLUMNS["precip"]].to_numpy().astype(np.float64),
    )


def ens_for_window(
    *, previous: EnsRun, current: EnsRun, origin_epoch_day: int, window_stamps_us: np.ndarray
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return the ensemble means at the stamps of a window.

    The day D - 1 run up to the native origin, the day D run after.

    Args:
        previous: The 00Z run of day D - 1.
        current: The 00Z run of day D.
        origin_epoch_day: The epoch day of D.
        window_stamps_us: The period-end stamp of each of the 96 columns, in microseconds.

    Returns:
        `temperature`, `wind`, and `precip`, each `(96,)` with NaN where the chosen run has no stamp.

    Raises:
        RerunError: When the runs are not those of days D - 1 and D.
    """
    epoch_grid = GRID_START_EPOCH_DAY
    if (
        previous.init_day != origin_epoch_day - 1 - epoch_grid
        or current.init_day != origin_epoch_day - epoch_grid
    ):
        msg = "the ensemble runs are not the 00Z runs of days D - 1 and D"
        raise RerunError(msg)
    native_us = origin_epoch_day * US_PER_DAY + 9 * US_PER_HOUR
    out = {k: np.full(len(window_stamps_us), np.nan) for k in ("temperature", "wind", "precip")}
    for run, chosen in (
        (previous, window_stamps_us <= native_us),
        (current, window_stamps_us > native_us),
    ):
        pos = np.searchsorted(run.valid_us, window_stamps_us)
        pos = np.clip(pos, 0, len(run.valid_us) - 1)
        found = chosen & (run.valid_us[pos] == window_stamps_us)
        for key, column in out.items():
            column[found] = getattr(run, key)[pos[found]]
    return out


def window_stamps_us(*, day_epoch: int) -> np.ndarray:
    """Return the period-end stamp (microseconds) of each of the 96 columns of the window of the UTC day `day_epoch`."""
    first_slot = (day_epoch - 1) * core.STEPS_PER_DAY + MID
    return (first_slot + 1 + np.arange(STEPS)) * US_PER_STEP


# --------------------------------------------------------------------------- the orchestration


@dataclass(frozen=True)
class RerunOutputs:
    """The re-run outputs of every origin, averaged over seeds.

    Attributes:
        origin_days: The grid day number of each origin `(n,)`.
        values: `(n, 2, 48, 3)` float32: P(X<0), the solar output in points, and the step probability
            for UTC days D - 2 and D - 1.
        ok: `(n, 2, 48)`: whether the half-hour is visible and D8 holds a reading there.
    """

    origin_days: np.ndarray
    values: np.ndarray
    ok: np.ndarray


def rerun_origins(
    *,
    origin_days: Sequence[int],
    build_window: Callable[[int, int, np.ndarray], tuple[np.ndarray, np.ndarray]],
    predict: Callable[[np.ndarray], np.ndarray],
    n_seeds: int,
    batch: int = 256,
) -> RerunOutputs:  # fmt: skip
    """Re-run the network on days D - 2 and D - 1 of every origin and average the outputs over seeds.

    Args:
        origin_days: The grid day number D of each origin.
        build_window: `(origin_day, day, hidden)` to the input window `(C, 96)` with the inputs after
            the limit hidden, and the middle's reading mask `(48,)` before hiding.
        predict: The windows `(n, C, 96)` to the decoded outputs `(seeds, n, 3, 48)`.
        n_seeds: The number of seeds `predict` returns.
        batch: Windows per call of `predict`.

    Returns:
        The outputs.

    Raises:
        RerunError: When `predict` returns a shape other than `(n_seeds, n, 3, 48)` or a non-finite value.
    """
    jobs = [(d, d - off) for d in origin_days for off in RERUN_OFFSETS]
    windows, readings = [], []
    for origin_day, day in jobs:
        hidden = window_hidden(origin_day=origin_day, day=day)
        w, reading_ok = build_window(origin_day, day, hidden)
        windows.append(w)
        readings.append(reading_ok)
    stacked = np.stack(windows).astype(np.float32)
    outs = []
    for i in range(0, len(stacked), batch):
        got = np.asarray(predict(stacked[i : i + batch]))
        want = (n_seeds, len(stacked[i : i + batch]), core.N_CHANNELS, core.STEPS_PER_DAY)
        if got.shape != want:
            msg = f"the network's outputs have shape {got.shape}, not {want}"
            raise RerunError(msg)
        outs.append(got.mean(axis=0))
    values = np.concatenate(outs).transpose(0, 2, 1)  # (n, 48, 3)
    if not np.isfinite(values).all():
        msg = "the re-run produced a non-finite output"
        raise RerunError(msg)
    n = len(origin_days)
    visible = np.stack([visible_middle(origin_day=d, day=day) for d, day in jobs])
    ok = (visible & np.stack(readings)).reshape(n, len(RERUN_OFFSETS), core.STEPS_PER_DAY)
    return RerunOutputs(
        origin_days=np.asarray(origin_days, dtype=np.int64),
        values=values.reshape(n, len(RERUN_OFFSETS), core.STEPS_PER_DAY, core.N_CHANNELS).astype(
            np.float32
        ),
        ok=ok,
    )


def write_rerun(*, outputs: RerunOutputs, path: Path) -> None:
    """Write the re-run outputs as one compressed archive, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, origin_days=outputs.origin_days, values=outputs.values, ok=outputs.ok)
    tmp.replace(path)


def read_rerun(*, path: Path) -> RerunOutputs:
    """Read the archive written by `write_rerun`."""
    with np.load(path) as z:
        return RerunOutputs(origin_days=z["origin_days"], values=z["values"], ok=z["ok"])


def assert_rerun_ignores_inputs_after_limit(
    *, run: Callable[[float | None], np.ndarray], poison_values: Sequence[float] = POISON_VALUES, tol: float = core.TOLERANCE_POISON,
) -> int:  # fmt: skip
    """Check that poisoning every input after the limit leaves the re-run's outputs finite and unchanged.

    Args:
        run: Returns the re-run's outputs with the inputs after the limit replaced by the given
            value (None leaves them as they are).
        poison_values: The poison values to try (NaN and a large finite value).
        tol: The largest allowed absolute change.

    Returns:
        The number of poison values checked.

    Raises:
        RerunError: When an output is non-finite or changes, or when `poison_values` is empty.
    """
    if not len(poison_values):
        msg = "no poison value was given, so the poisoned re-run checked nothing"
        raise RerunError(msg)
    clean = run(None)
    if clean.size == 0 or not np.isfinite(clean).all():
        msg = "the clean re-run is empty or non-finite"
        raise RerunError(msg)
    for value in poison_values:
        poisoned = run(value)
        if not np.isfinite(poisoned).all():
            msg = "a poisoned re-run produced a non-finite output"
            raise RerunError(msg)
        if np.abs(poisoned - clean).max() > tol:
            msg = "a re-run output depends on an input after the visibility limit"
            raise RerunError(msg)
    return len(poison_values)


# --------------------------------------------------------------------------- real adapters


def span_segment(
    *, cal: Any, first_epoch_day: int, last_epoch_day: int, reading: np.ndarray, truth: np.ndarray, use: np.ndarray,
) -> Any:  # fmt: skip
    """Return one pseudo-segment of real days that spans every calendar day from `first_epoch_day` to `last_epoch_day`.

    The segment is experiment 729's `real_segment` generalised from one month to a range of days, so
    a window near a month edge reads the real days on both sides. The first and last dense days are
    padding days with no reading.

    Args:
        cal: Experiment 729's `Calendar` of the whole day table.
        first_epoch_day: The first real day, in days since 1970-01-01.
        last_epoch_day: The last real day.
        reading: D8's reading on the calendar axis `(n, 48)`, in points.
        truth: The answer on the calendar axis (the reading itself on real days).
        use: Which calendar rows may be used.

    Returns:
        The segment (`e00729_sim.Segment`).
    """
    import e00729_core as core729
    import e00729_sim as sim

    epoch = np.asarray(cal.dates).astype("datetime64[D]").astype(np.int64)
    dates = (np.arange(first_epoch_day - 1, last_epoch_day + 2)).astype("datetime64[D]")
    pos = np.clip(np.searchsorted(epoch, dates.astype(np.int64)), 0, len(epoch) - 1)
    present = (epoch[pos] == dates.astype(np.int64)) & use[pos]
    present[0] = present[-1] = False
    cal_index = np.where(present, pos, -1)
    r = sim.to_dense(calendar_values=reading, cal_index=cal_index)
    x = sim.to_dense(calendar_values=truth, cal_index=cal_index)
    shape = (len(dates), core.STEPS_PER_DAY)
    bsp = sim.to_dense(calendar_values=cal.bsp, cal_index=cal_index)
    return sim.Segment(
        month="span", world="none", source="none", dates=dates, cal_index=cal_index,
        base=r, valid=np.isfinite(r) & np.isfinite(x), engine=np.zeros(shape, np.float32),
        solar=np.zeros(shape, np.float32), x=x, r=r, bsp=bsp,
        csi=sim.to_dense(calendar_values=cal.csi, cal_index=cal_index),
        clear=sim.to_dense(calendar_values=cal.clear, cal_index=cal_index),
        cosz=sim.to_dense(calendar_values=cal.cosz, cal_index=cal_index),
        t2m=sim.to_dense(calendar_values=cal.t2m, cal_index=cal_index),
        wind10=sim.to_dense(calendar_values=cal.wind10, cal_index=cal_index),
        precip=sim.to_dense(calendar_values=cal.precip, cal_index=cal_index),
        s1_fallback=np.zeros(len(dates), dtype=bool),
        has_bsp=int(np.isfinite(bsp).any(axis=1).sum()) >= core729.MIN_DAYS_IN_MONTH,
        params={},
    )  # fmt: skip


def poisoned_calendar(*, cal: Any, origin_epoch_day: int, value: float) -> Any:
    """Return a copy of the calendar with every input after the visibility limit of one origin set to `value`.

    The inputs are D8's reading, the BSP meter, the CAMS clear-sky index, and the ERA5 temperature,
    wind speed, and precipitation, at every half-hour after the last visible slot of the origin.
    The neighbours are poisoned by `poisoned_neighbours`.
    """
    import copy

    out = copy.copy(cal)
    epoch = np.asarray(cal.dates).astype("datetime64[D]").astype(np.int64)
    limit = last_visible_abs_slot(origin_day=origin_epoch_day - GRID_START_EPOCH_DAY)
    slot = (epoch[:, None] - GRID_START_EPOCH_DAY) * core.STEPS_PER_DAY + np.arange(
        core.STEPS_PER_DAY
    )[None, :]
    after = slot > limit
    for name in ("reading", "bsp", "csi", "t2m", "wind10", "precip"):
        arr = np.array(getattr(cal, name), dtype=np.float32, copy=True)
        arr[after] = value
        setattr(out, name, arr)
    return out


def poisoned_neighbours(
    *, neighbours: Mapping[str, np.ndarray], dates: np.ndarray, origin_epoch_day: int, value: float
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return the neighbours' series (on the calendar axis) with every half-hour after the limit set to `value`."""
    epoch = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    limit = last_visible_abs_slot(origin_day=origin_epoch_day - GRID_START_EPOCH_DAY)
    slot = (epoch[:, None] - GRID_START_EPOCH_DAY) * core.STEPS_PER_DAY + np.arange(
        core.STEPS_PER_DAY
    )[None, :]
    after = slot > limit
    out = {}
    for name, arr in neighbours.items():
        a = np.array(arr, dtype=np.float32, copy=True)
        a[after] = value
        out[name] = a
    return out


def load_network(
    *, source: str, seed: int, root_732: Path, root_730: Path, device: str = "cpu"
) -> Any:
    """Load one stored untangling network.

    Args:
        source: "U*" (experiment 732) or "S" (experiment 730).
        seed: The seed.
        root_732: Experiment 732's data folder.
        root_730: Experiment 730's data folder.
        device: "cpu" (the plan's CPU steps never use the GPU).

    Returns:
        The network in evaluation mode. `U*` reads the stack and the 12 neighbour channels; `S` reads
        the stack alone.
    """
    import e00730_arms as arms730
    import e00730_nets as nets730
    import e00732_arms as arms
    import e00732_core as core732
    import e00732_nets as nets732
    import torch

    if source == core.SOURCE_732:
        spec = core732.ARMS["U*"]
        net = nets732.build_net(spec=spec, c_in=arms.n_channels(spec=spec))
        state = root_732 / "networks" / f"Ustar_unet_s{seed}.pt"
    else:
        net = nets730.build_net(
            arch="unet", spec=arms730.ArmSpec("S", "d", demand=False, style=core732.STYLE)
        )
        state = root_730 / "networks" / f"S_unet_s{seed}.pt"
    net.load_state_dict(torch.load(state, map_location=device))
    return net.to(device).eval()


def predict_windows(
    *, nets: Sequence[Any], windows: np.ndarray, source: str, n_base: int, device: str = "cpu"
) -> np.ndarray:
    """Run the networks on windows `(n, C, 96)` and return `(seeds, n, 3, 48)`.

    P(X<0), the solar output, and the step probability.

    The scored middle of each output is the 48 steps from step 24. A network without the extra heads
    (`S`) returns zeros for the solar output and the step probability.
    """
    import e00732_nets as nets732
    import torch

    x = torch.from_numpy(
        np.ascontiguousarray(
            windows[:, :n_base] if source != core.SOURCE_732 else windows, dtype=np.float32
        )
    ).to(device)
    out = []
    with torch.no_grad():
        for net in nets:
            raw = net(x)[:, :, MID : MID + core.STEPS_PER_DAY]
            if raw.shape[1] == 1:
                p = torch.sigmoid(raw[:, 0])
                dec = {"p": p, "pv": torch.zeros_like(p), "step_p": torch.zeros_like(p)}
            else:
                dec = nets732.decode(raw=raw)
            out.append(
                torch.stack([dec["p"], dec["pv"], dec["step_p"]], dim=1).float().cpu().numpy()
            )
    return np.stack(out)

# ruff: noqa: D205, E501, PT008
"""Experiment 725: checks KL3, KL3n, and KL3r, the known-answer test that no history input uses
weather unavailable at the origin.

**KL3.** For one sample (a Primary and a contract origin `c`), every stored source array is copied, and
every value that the live rule forbids is set to 1e6 and then to NaN, before any lookup:

- ERA5 values (columns 1 to 14 of the dense array) stamped after 09:00 UTC on D - 5;
- half-hourly CAMS values (columns 0 and 15 to 18) whose CAMS source stamp is after 00:00 UTC on
  D - 2;
- CAMS day summaries of UTC days after D - 3;
- the `ft` values and stored B1 values whose run has a forecast origin at or after the native origin
  (this includes the run of day D, whose forecast origin is the native origin).

The assembled history weather must be bit-identical to the unpoisoned batch, except in the counted
fallback cells of a training sample (where an ENS cell reads the analysis, which may be poisoned).

**KL3n.** The same poisons moved one step earlier (ERA5 after 09:00 UTC on D - 6, CAMS after 00:00
UTC on D - 3, day summaries after D - 4, runs with a forecast origin at or after 09:00 UTC on D - 1)
must each change at least one input of at least one sample.

**KL3r.** The 12 analysis window columns and the CAMS day summary come from `e00632_data.dense_b1()`,
which loads arrays built once from the clean data, so a poison of the raw arrays never reaches them.
KL3r rebuilds the window columns and the day summary of one Primary from the poisoned weather
(`e00437_features.dense_arrays` with `e00418_prep.dense_weather` patched, and
`e00437_features.cams_previous_day` with the hourly CAMS table poisoned) and assembles the history from
the rebuilt columns. The rebuild takes about 1 minute per Primary, on the CPU.
"""

import copy
import unittest.mock
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00725_batch as lb
import e00725_rule as rule
import numpy as np
import torch  # ty: ignore[unresolved-import]

POISON_VALUES: Final = (1e6, float("nan"))
KINDS: Final = ("era5", "cams", "day", "ens")
ERA5_COLS: Final = list(rule.ERA5_COLUMNS)
CAMS_COLS: Final = list(rule.CAMS_COLUMNS)


@dataclass
class OneSample:
    """One (Primary, contract origin) sample and a lead day for the assembly."""

    primary: int
    c_idx: int
    lead_day: int = 0


def assemble_one(*, source: lb.LiveSource, sample: OneSample) -> bt.Assembled:
    """Assemble one sample with a source (the stage 2 and prediction path: ensemble target row)."""
    dev = source.device
    return source.assemble(
        primary=torch.tensor([sample.primary], device=dev),
        c_idx=torch.tensor([sample.c_idx], device=dev),
        lead_day=torch.tensor([sample.lead_day], device=dev),
        ensemble=True,
    )


def poisoned_source(
    *, live: lb.LiveSource, sample: OneSample, value: float, kinds: Sequence[str] = KINDS, shift_days: int = 0
) -> lb.LiveSource:  # fmt: skip
    """Return a copy of `live` whose forbidden values (of the given `kinds`) for `sample` are `value`.

    Args:
        live: The source.
        sample: The Primary and the contract origin.
        value: 1e6 or NaN.
        kinds: Which of `era5`, `cams`, `day`, and `ens` to poison.
        shift_days: 0 for the exact limits of the rule; 1 moves every limit one day earlier (the
            negative controls of KL3n).

    Returns:
        A copy of `live` with the forbidden values replaced.
    """
    assert shift_days in {0, 1}
    new = copy.copy(live)
    p, c = sample.primary, sample.c_idx
    native = c - lay.NATIVE_OFFSET_STEPS
    day = c // lay.STEPS_PER_DAY
    back = lay.STEPS_PER_DAY * shift_days
    if "era5" in kinds or "cams" in kinds:
        dense = live.dense_weather.clone()
        if "era5" in kinds:
            dense[p, native - rule.ERA5_LAG_HOURS * 2 - back + 1 :, ERA5_COLS] = value
        if "cams" in kinds:
            cams_last = lay.STEPS_PER_DAY * (day - 2) - back
            dense[p, cams_last + 1 :, CAMS_COLS] = value
        new.dense_weather = dense
    if "day" in kinds:
        days = live.cams_days.clone()
        last_slot = live.grid_day0 + day - rule.CAMS_LAST_DAY_BEFORE - shift_days - live.day0
        days[p, last_slot + 1 :] = value
        new.cams_days = days
    if "ens" in kinds:
        ens = live.ens_z.clone()
        late = live.ens_origin[p] >= native - back
        ens[p, late] = value
        new.ens_z = ens
    return new


def exempt_mask(*, plan_fallback: torch.Tensor, phase: str) -> torch.Tensor:
    """Return the (1, 14, 48, 21) bool mask of the cells that KL3 exempts: the counted training fallback cells."""
    mask = torch.zeros(
        (*plan_fallback.shape[:-1], rule.N_WEATHER_COLUMNS),
        dtype=torch.bool,
        device=plan_fallback.device,
    )
    if phase == "train":
        mask[..., : rule.N_ENS_COLUMNS] = plan_fallback
    return mask


def equal_outside(*, a: torch.Tensor, b: torch.Tensor, exempt: torch.Tensor) -> bool:
    """Return whether two history-weather tensors are bit-identical outside the exempt cells (NaN
    counts as different).
    """
    zero = torch.zeros_like(a)
    return torch.equal(torch.where(exempt, zero, a), torch.where(exempt, zero, b))


def fallback_count_reference(
    *, found: np.ndarray, rule_name: str, primary: np.ndarray, c_idx: np.ndarray
) -> int:  # fmt: skip
    """Return the number of fallback cells of the samples from the cache's `found` mask alone (the KL7 count).

    A fallback cell is a (token, column) pair that the rule reads from the ENS value and whose
    half-hour has no row.
    """
    per_token = rule.ens_mask(rule=rule_name).sum(axis=-1)  # ty: ignore[invalid-argument-type]
    contract = rule.STEPS_PER_DAY * rule.REFERENCE_DAY + rule.STEPS_PER_DAY // 2
    offsets = rule.token_grid_index(day=rule.REFERENCE_DAY) - contract  # (14, 48), token minus c
    total = 0
    for p, c in zip(primary, c_idx, strict=True):
        missing = ~np.asarray(found[int(p)][int(c) + offsets])
        total += int((per_token * missing).sum())
    return total


def kl3(
    *, live: lb.LiveSource, samples: Sequence[OneSample], phase: str, with_controls: bool = True,
) -> dict:  # fmt: skip
    """Run KL3 (and the KL3n controls) on `samples` and return the result.

    Args:
        live: The source.
        samples: The Primaries and contract origins to check.
        phase: `train` or `predict`.
        with_controls: Whether to run the KL3n negative controls as well.

    Returns:
        A dict with `pass`, the number of samples, the number of exempt cells, and, for each control kind,
        whether any sample changed.

    Raises:
        AssertionError: If a poisoned batch differs from the clean batch outside the exempt cells.
    """
    n_exempt = 0
    changed = dict.fromkeys(KINDS, False)
    for sample in samples:
        clean = assemble_one(source=live, sample=sample)
        plan = live.plan(primary=clean.inputs.primary, hist_idx=clean.hist_idx)
        exempt = exempt_mask(plan_fallback=plan.fallback, phase=phase)
        n_exempt += int(exempt.sum())
        for value in POISON_VALUES:
            bad = assemble_one(
                source=poisoned_source(live=live, sample=sample, value=value), sample=sample
            )
            assert equal_outside(
                a=bad.inputs.hist_weather, b=clean.inputs.hist_weather, exempt=exempt
            ), f"KL3: a poisoned source ({value}) changed the history weather of sample {sample}"
        if not with_controls:
            continue
        for kind in KINDS:
            moved = assemble_one(
                source=poisoned_source(live=live, sample=sample, value=1e6, kinds=(kind,), shift_days=1),
                sample=sample,
            )  # fmt: skip
            if not equal_outside(
                a=moved.inputs.hist_weather, b=clean.inputs.hist_weather, exempt=exempt
            ):
                changed[kind] = True
    ok_controls = all(changed.values()) if with_controls else None
    return {
        "pass": bool(ok_controls) if with_controls else True, "n_samples": len(samples),
        "n_exempt_cells": n_exempt, "controls_changed": changed,
    }  # fmt: skip


# ----------------------------------------------------------------------------------------- KL3r


def _poison_weather(
    *,
    weather: np.ndarray,
    c_idx: int,
    value: float,
    shift_days: int,
    kinds: Sequence[str] = ("era5", "cams"),
) -> np.ndarray:
    """Return a copy of the raw (G, 7) weather of a Primary with the forbidden half-hours set to `value`.

    `kinds` selects the ERA5 columns (1 to 6, which feed the trailing temperature columns 7 to 14) and the
    CAMS shortwave column (0, which feeds the trailing irradiance columns 15 to 18).
    """
    out = np.array(weather, copy=True)
    native = c_idx - lay.NATIVE_OFFSET_STEPS
    day = c_idx // lay.STEPS_PER_DAY
    back = lay.STEPS_PER_DAY * shift_days
    if "era5" in kinds:
        out[native - rule.ERA5_LAG_HOURS * 2 - back + 1 :, 1:] = value
    if "cams" in kinds:
        out[lay.STEPS_PER_DAY * (day - 2) - back + 1 :, 0] = value
    return out


def _poison_cams_table(*, path: Path, out: Path, c_idx: int, value: float, shift_days: int) -> Path:
    """Write a copy of the hourly CAMS table with the global irradiance after 00:00 UTC on D - 2
    (less `shift_days`) set to `value`.
    """
    import polars as pl

    grid_start = datetime(2019, 9, 1, tzinfo=UTC)
    day = c_idx // lay.STEPS_PER_DAY
    limit = grid_start + timedelta(days=day - 2 - shift_days)
    table = pl.read_parquet(path)
    table = table.with_columns(
        ghi_w_m2=pl.when(pl.col("time") > limit).then(pl.lit(value)).otherwise(pl.col("ghi_w_m2"))
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    return out


def rebuilt_primary_arrays(
    *, lw: object, row: dict, weather: np.ndarray, cams_path: Path, local_date: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Rebuild one Primary's standardised weather (7 columns), window columns (12), and CAMS day
    summaries from raw inputs.

    The weather is built as `e00632_data.dense_raw` builds it (copied for one Primary), the window columns
    through `e00437_features.dense_arrays`, and the day summaries through `e00437_features.cams_previous_day`;
    `e00418_prep.dense_weather` and `e00418_prep.CAMS_PATH` are patched to return the given inputs.

    Args:
        lw: The loaded world, whose `stats` hold the fold's statistics.
        row: The Primary's row of the metadata table.
        weather: The raw dense weather of the Primary.
        cams_path: The CAMS table that the patched `CAMS_PATH` points to.
        local_date: The local date of every half-hour of the grid.

    Returns:
        (7-column weather (G, 7) float16, window columns (G, 12) float16, day summaries (N_DAYS, 2) float32),
        all standardised with the fold's statistics in `lw.stats`.
    """
    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00418_prep as prep418  # ty: ignore[unresolved-import]
    import e00437_features as f437  # ty: ignore[unresolved-import]
    import e00632_data as d632  # ty: ignore[unresolved-import]
    import polars as pl

    stats = lw.stats  # ty: ignore[unresolved-attribute]
    with (
        unittest.mock.patch.object(prep418, "dense_weather", lambda **_: weather),
        unittest.mock.patch.object(prep418, "CAMS_PATH", cams_path),
    ):
        grid = common.grid_times()
        derived = common.derived_columns(
            valid_time=grid, latitude=row["latitude"], longitude=row["longitude"]
        )
        columns: dict[str, np.ndarray] = {}
        for k, name in enumerate(prep.WEATHER):
            columns[f"mean_{name}"] = np.nan_to_num(weather[:, k], nan=0.0)
            columns[f"std_{name}"] = np.zeros(grid.len(), dtype=np.float32)
        for name in common.DERIVED:
            columns[name] = derived[name].to_numpy()
        for name in common.LAG_COLUMNS:
            columns[name] = np.zeros(grid.len(), dtype=np.float32)
        columns["lead_hours"] = np.zeros(grid.len(), dtype=np.float32)
        frame = pl.DataFrame(columns).with_columns(valid_time=grid, power_fcst_init_time=grid)
        x = prep.feature_matrix(frame=frame, scheme="summary")
        raw7 = np.where(np.isfinite(weather), x[:, d632.WEATHER_IN_46], np.nan).astype(np.float32)
        z7 = np.clip(
            (raw7 - stats["weather_mean"]) / stats["weather_std"], -d632.CLIP_SD, d632.CLIP_SD
        )
        dense = f437.dense_arrays(row=row, local_date=local_date)
        grid12 = np.stack([dense[name] for name in d632.B1_GRID_NAMES], axis=1).astype(np.float32)
        cams = np.full((d632.N_DAYS, 2), np.nan, dtype=np.float32)
        for day, (mean, csi) in f437.cams_previous_day(site=row["site"]).items():
            if 0 <= day - d632.DAY0 < d632.N_DAYS:
                cams[day - d632.DAY0] = (mean, csi)
        z12, zc = d632.b1_scale(
            grid=grid12, cams=cams, mean=stats["b1_mean"], std=stats["b1_std"], device="cpu"
        )
    return (
        np.nan_to_num(z7, nan=0.0).astype(np.float16), z12.numpy(), zc.numpy(),
    )  # fmt: skip


def source_with_rebuilt(
    *, live: lb.LiveSource, primary: int, z7: np.ndarray, z12: np.ndarray, zc: np.ndarray
) -> lb.LiveSource:  # fmt: skip
    """Return a copy of `live` whose analysis arrays for `primary` are the rebuilt ones."""
    new = copy.copy(live)
    dense = live.dense_weather.clone()
    dev = dense.device
    dense[primary, :, :7] = torch.from_numpy(z7).to(dev)
    dense[primary, :, 7:19] = torch.from_numpy(z12).to(dev)
    days = live.cams_days.clone()
    days[primary] = torch.from_numpy(zc).to(dev)
    new.dense_weather, new.cams_days = dense, days
    return new


class Control(NamedTuple):
    """One negative control: the poisoned inputs, the history row to look at, and the columns that must change."""

    kinds: tuple[str, ...]
    slot: int
    columns: slice


CONTROLS: Final = {
    "era5": Control(kinds=("era5",), slot=7 - rule.FIRST_AGE, columns=slice(7, 15)),
    "cams": Control(kinds=("cams",), slot=3 - rule.FIRST_AGE, columns=slice(15, 19)),
    "day_summary": Control(kinds=(), slot=-1, columns=slice(19, 21)),
}
"""The three negative controls of KL3r: the poison moved one day earlier in one input, and the columns that must change.

The ERA5 control poisons raw weather columns 1 to 6 and must change trailing columns 7 to 14 at age 7; the CAMS
control poisons raw column 0 and must change trailing columns 15 to 18 at age 3; the day-summary control poisons
the hourly CAMS table alone and must change columns 19 and 20 at the tokens that read the summary of D - 3.
"""


def kl3r(
    *, lw: object, samples: Sequence[OneSample], phase: str, scratch: Path, with_controls: bool = True,
) -> dict:  # fmt: skip
    """Run KL3r on `samples` (one Primary and origin each).

    For each sample the analysis columns are rebuilt from clean weather and from poisoned weather
    (1e6 and NaN); the history assembled from the poisoned rebuild must equal the history assembled
    from the clean rebuild, outside the counted fallback cells. The clean rebuild must also equal
    the cached batch. With `with_controls`, each of the three controls of `CONTROLS` (the poison
    moved one day earlier) must change its own columns in at least one sample. The clean rebuild is
    cached for each Primary, because it does not depend on the origin.
    """
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00418_prep as prep418  # ty: ignore[unresolved-import]
    import e00437_features as f437  # ty: ignore[unresolved-import]

    live: lb.LiveSource = lw.live  # ty: ignore[unresolved-attribute]
    rows = {int(r["primary"]): r for r in prep.primaries().iter_rows(named=True)}
    local_date = f437.local_date_codes()
    changed = dict.fromkeys(CONTROLS, False)
    n_cached_mismatch = 0
    clean_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    weather_cache: dict[int, np.ndarray] = {}
    for sample in samples:
        row = rows[sample.primary]
        if sample.primary not in clean_cache:
            weather_cache[sample.primary] = prep418.dense_weather(
                series_id=int(row["time_series_id"]), site=row["site"]
            )
            clean_cache[sample.primary] = rebuilt_primary_arrays(
                lw=lw, row=row, weather=weather_cache[sample.primary], cams_path=prep418.CAMS_PATH,
                local_date=local_date,
            )  # fmt: skip
        weather = weather_cache[sample.primary]
        clean_arrays = clean_cache[sample.primary]
        clean_src = source_with_rebuilt(
            live=live, primary=sample.primary, z7=clean_arrays[0], z12=clean_arrays[1], zc=clean_arrays[2]
        )  # fmt: skip
        reference = assemble_one(source=clean_src, sample=sample)
        cached = assemble_one(source=live, sample=sample)
        plan = live.plan(primary=reference.inputs.primary, hist_idx=reference.hist_idx)
        exempt = exempt_mask(plan_fallback=plan.fallback, phase=phase)
        if not equal_outside(
            a=reference.inputs.hist_weather, b=cached.inputs.hist_weather, exempt=exempt
        ):
            n_cached_mismatch += 1
        for value in POISON_VALUES:
            arrays = _rebuild_poisoned(
                lw=lw, row=row, weather=weather, sample=sample, value=value, shift_days=0,
                scratch=scratch, local_date=local_date,
            )  # fmt: skip
            src = source_with_rebuilt(
                live=live, primary=sample.primary, z7=arrays[0], z12=arrays[1], zc=arrays[2]
            )
            src = poisoned_source(live=src, sample=sample, value=value, kinds=("ens",))
            bad = assemble_one(source=src, sample=sample)
            assert equal_outside(
                a=bad.inputs.hist_weather, b=reference.inputs.hist_weather, exempt=exempt
            ), f"KL3r: a poisoned rebuild ({value}) changed the history weather of sample {sample}"
        if not with_controls:
            continue
        reads_d3 = rule.cell_sources(rule=live.rule_name) == rule.SRC_DAY_D3  # ty: ignore[invalid-argument-type]
        for name, control in CONTROLS.items():
            if changed[name]:
                continue
            arrays = _rebuild_poisoned(
                lw=lw, row=row, weather=weather, sample=sample, value=1e6, shift_days=1,
                scratch=scratch, local_date=local_date, kinds=control.kinds, cams_table=name == "day_summary",
            )  # fmt: skip
            src = source_with_rebuilt(
                live=live, primary=sample.primary, z7=arrays[0], z12=arrays[1], zc=arrays[2]
            )
            moved = assemble_one(source=src, sample=sample)
            diff = (moved.inputs.hist_weather != reference.inputs.hist_weather)[0].cpu().numpy()
            columns = control.columns
            if name == "day_summary":
                changed[name] = bool((diff[..., columns] & reads_d3[..., columns]).any())
            else:
                changed[name] = bool(diff[control.slot, :, columns].any())
    ok_controls = all(changed.values()) if with_controls else True
    return {
        "pass": bool(ok_controls) and n_cached_mismatch == 0, "n_samples": len(samples),
        "controls_changed": changed, "clean_rebuild_differs_from_cache": n_cached_mismatch,
    }  # fmt: skip


def _rebuild_poisoned(
    *, lw: object, row: dict, weather: np.ndarray, sample: OneSample, value: float, shift_days: int,
    scratch: Path, local_date: np.ndarray, kinds: Sequence[str] = ("era5", "cams"), cams_table: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Rebuild one Primary's analysis arrays from the poisoned raw weather (`kinds`) and, if
    `cams_table`, the poisoned hourly CAMS table.
    """
    import e00418_prep as prep418  # ty: ignore[unresolved-import]

    bad_weather = _poison_weather(
        weather=weather, c_idx=sample.c_idx, value=value, shift_days=shift_days, kinds=kinds
    )
    if cams_table:
        cams_path = _poison_cams_table(
            path=prep418.CAMS_PATH, out=scratch / f"cams_poison_{sample.primary}_{sample.c_idx}.parquet",
            c_idx=sample.c_idx, value=value, shift_days=shift_days,
        )  # fmt: skip
    else:
        cams_path = prep418.CAMS_PATH
    return rebuilt_primary_arrays(
        lw=lw, row=row, weather=bad_weather, cams_path=cams_path, local_date=local_date
    )

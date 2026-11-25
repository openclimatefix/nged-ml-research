# ruff: noqa: E501
"""Experiment 721: the loader of real folds, the probe provider, and the PV tables.

**This module reads private data and has been run only through `tests/smoke_real_files.py`'s
read-only checks.** Its functions mirror experiment 716's `e00716_fit.build_pv_tables`, which builds
the same
arrays for a real fold, with these differences:

1. The world's power (the injection at the hidden pair and the four visible Primaries, the folding, the
   gap-row drop) comes from `e00721_worlds.build_world_arrays` on experiment 720's panel.
2. The capacity of the hidden pair (D5 and D10) comes from a file-based probe on folded readings
   (`ProbeProvider`), never from the signed readings. Every other Primary reads experiment 716's
   `capacity_table` (experiment 649's table for the six signed-MW Primaries, and experiment 662's
   for the
   eight MVA-only Primaries).
3. The oracle injected capacity is added to the share scale of each injected Primary (`increment / 100`
   times the scale of 1, 0.7, or 1.3).
4. Stage 1 reads the dense table (`PvSource` in world mode); stages 2 and 3 read the ensemble table.
5. `SG-A-S` shifts the shortwave, temperature, and `u` by 28 days in the history rows and the target
row.
"""

import contextlib
import datetime
import json
import os
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00721_paths  # noqa: F401

# isort: split
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00716_batch as pvb  # ty: ignore[unresolved-import]
import e00721_batch as sgb
import e00721_core as core
import e00721_fit as fit
import e00721_jobs as jobs
import e00721_train as trn
import e00721_worlds as wd
import numpy as np
import torch  # ty: ignore[unresolved-import]

PROBE_FOLDER: Final = "probes"
"""Folder, under the experiment's output root, of the probe files: `<world>_<fold>.json` holding `{"D5": mw, "D10": mw}`."""
SUPERVISED_SITES: Final = core.VISIBLE_SIGNED
"""The Primaries whose true sign trains the head: the four visible signed ones. D5, D10, and D15 are excluded."""


class ProbeFileMissingError(FileNotFoundError):
    """Raised when the capacity probe of a world and fold has not been written."""


@dataclass(frozen=True)
class ProbeProvider:
    """File-based capacity probe of the hidden pair on the folded readings (experiment 662's probe).

    The probe runs outside this module (a CPU job on the folded readings with data before the fold's
    start only) and writes `<root>/probes/<world>_<fold>.json`. This provider reads the file and never
    fabricates a value.
    """

    root: Path

    def path(self, *, world: str, fold: str) -> Path:
        """Return the probe file of a world and fold."""
        return self.root / PROBE_FOLDER / f"{world}_{fold}.json"

    def capacity_mw(self, *, world: str, fold: str) -> dict[str, float]:
        """Return the probe's capacity in MW at D5 and D10, or raise naming the missing file."""
        path = self.path(world=world, fold=fold)
        if not path.exists():
            msg = (
                f"the capacity probe of the hidden pair for world {world}, fold {fold} is missing: {path}. "
                "Run experiment 662's probe on the folded readings of that world with data before the fold's start, "
                "and write {'D5': mw, 'D10': mw} there."
            )
            raise ProbeFileMissingError(msg)
        body = json.loads(path.read_text())
        missing = [s for s in core.HIDDEN_PAIR if s not in body or not float(body[s]) > 0.0]
        if missing:
            msg = f"{path} has no positive capacity for {missing}"
            raise ValueError(msg)
        return {s: float(body[s]) for s in core.HIDDEN_PAIR}


SYNTH_WORLD_FILE: Final = "synthetic_world.json"
"""File, under the output root, that Stage 0 writes: `{"day_offset_points": 0.0}` or `5.0`, and the checks."""
GRID_END_87: Final = datetime.datetime(2026, 7, 31, 23, 30, tzinfo=datetime.UTC)
"""The last stamp of experiment 87's grid (121,248 half-hours), which experiment 667 moves to 2026-11-20."""
N_GRID_87: Final = 121_248


def sky_columns() -> dict[str, tuple[int, ...]]:
    """Return the sky column indices that `blank_sky` zeroes, from `e00721_arms` (tested)."""
    import e00721_arms as arms

    return {
        "weather": tuple(arms.SKY_WEATHER_COLUMNS), "datetime": tuple(arms.SKY_DATETIME_COLUMNS),
        "spread": tuple(arms.SKY_SPREAD_COLUMNS),
    }  # fmt: skip


def capacities(
    *, fold: str, sites: list[str], world: str, probe: ProbeProvider, power_cap_mw: np.ndarray
) -> np.ndarray:
    """Return `(16,)` as-of AC capacity in MW: experiment 716's table, with D5 and D10 from the probe."""
    import e00716_pv as pv  # ty: ignore[unresolved-import]

    caps = np.array([c.mw for c in pv.capacity_table(fold=fold, sites=sites)])
    probed = probe.capacity_mw(world=world, fold=fold)
    for site, mw in probed.items():
        caps[sites.index(site)] = mw
    del power_cap_mw
    return caps


def share_scale_with_injection(
    *, caps_mw: np.ndarray, power_cap_mw: np.ndarray, increment_pct: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    """Return `(16,)` the share of capacity that `p` assumes.

    The share is the as-of capacity's share plus the oracle injected size times `scale`.
    """
    base = np.where(power_cap_mw > 0, caps_mw / power_cap_mw, 0.0)
    return np.where(base > 0, base, 0.0) + scale * increment_pct / 100.0


def target_row_u_gate(
    *, u_grid: np.ndarray, origin_idx: np.ndarray, native_offset: int = lay.NATIVE_OFFSET_STEPS
) -> np.ndarray:
    """Return `(n,)` the newest `u` that an origin may read.

    That is the `u` at the native origin or earlier, never the target row's own.

    The sign head reads history tokens only, whose `u` is recorded sky at stamps before the native
    origin minus the power cap. This function returns `u_grid` at the native origin so that a test can
    check that no `u` at or after the origin is read.
    """
    return u_grid[:, origin_idx - native_offset]


@contextlib.contextmanager
def grid_of_experiment_87() -> Generator[None]:
    """Put experiment 87's grid end back while the panel and the farm frame are read.

    Loading experiment 713's pipeline moves `e00087_common.GRID_END` to 2026-11-20 (experiment 667's
    extension), after which the power and the farm series have 126,624 half-hours and the CAMS arrays of
    experiment 412 have 121,248, so `load_panel` raises. Both grids start on 2019-09-01 00:00, so the
    first 121,248 stamps are the same half-hours (checked against the real files).
    """
    import e00087_common as common  # ty: ignore[unresolved-import]

    old = common.GRID_END
    common.GRID_END = GRID_END_87
    try:
        yield
    finally:
        common.GRID_END = old


@dataclass
class GridInputs:
    """What a world builder reads from the real files, before any arm is built (CPU only).

    Attributes:
        pipe: Experiment 713's pipeline of the fold.
        panel: Experiment 720's panel on experiment 87's grid.
        farms: The farm series of the six injected sites on experiment 87's grid, NaN from 2026-07-01 on.
        n_grid: Experiment 713's grid length (126,624), longer than the panel's.
        stamp_us: `(n_grid,)` the stamps of experiment 713's grid.
    """

    pipe: Any
    panel: Any
    farms: dict[str, np.ndarray]
    n_grid: int
    stamp_us: np.ndarray


def clip_farm(*, farm: np.ndarray, stamp_us: np.ndarray) -> np.ndarray:
    """Return a farm series with every half-hour on or after 2026-07-01 set to NaN (the farm frame runs to 2026-09-10)."""
    return wd.clip_to_holdout(farm_at=lambda s: farm[: len(s)], stamp_us=stamp_us[: len(farm)])


def load_grid_inputs(*, fold: str, seed: int, phase: str, with_farms: bool = True) -> GridInputs:
    """Load the pipeline, the panel, and the clipped farm series of a fold.

    The loading order keeps the grids equal.
    """
    import e00422_prep as p422  # ty: ignore[unresolved-import]
    import e00506_data as d506  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00720_real as real  # ty: ignore[unresolved-import]

    pipe = inp.load_pipeline(month=fold, phase=phase, chain_root=inp.chain_root_of_seed(seed=seed))
    n713 = int(pipe.d632.dense_raw().shape[1])
    with grid_of_experiment_87():
        panel = real.load_panel(dense_raw_path=Path(pipe.d632.CACHE) / "dense_raw.npy")
        assert len(panel.stamp_us) == N_GRID_87, len(panel.stamp_us)
        farms = {}
        if with_farms:
            for site in (*core.HIDDEN_PAIR, *core.VISIBLE_SIGNED):
                raw = np.asarray(d506.farm_norm(farm=d506.FARM_OF_SITE[site]), dtype=np.float64)
                assert len(raw) == len(panel.stamp_us), (site, len(raw))
                farms[site] = clip_farm(farm=raw, stamp_us=panel.stamp_us)
    stamp = p422.GRID_START_US + np.arange(n713, dtype=np.int64) * p422.STEP_US
    assert bool((stamp[: len(panel.stamp_us)] == panel.stamp_us).all()), (
        "the two grids do not share their first stamps"
    )
    return GridInputs(pipe=pipe, panel=panel, farms=farms, n_grid=n713, stamp_us=stamp)


def build_fold_world_arrays(
    *, panel: Any, farm_for_site: dict[str, np.ndarray], world: str, cutoff_idx: int | None
) -> wd.WorldArrays:
    """Return the world's arrays for a fold, after the holdout guards.

    In the train phase, every value at or after the cutoff is NaN. The guards cover the panel's power
    and every farm series, because an injected value on or after 2026-07-01 would not be seen by a guard
    that reads the power alone.
    """
    wd.holdout_guard(valid_us=panel.stamp_us[np.isfinite(panel.power_pct).any(axis=0)])
    for farm in farm_for_site.values():
        wd.holdout_guard(valid_us=panel.stamp_us[np.isfinite(farm)])
    arrays = wd.build_world_arrays(
        world=world, sites=panel.sites, power_pct=panel.power_pct, k=panel.k, stamp_us=panel.stamp_us, farm_for_site=farm_for_site,
    )  # fmt: skip
    if cutoff_idx is not None:
        for name in ("signed", "fold_view", "signed_view"):
            getattr(arrays, name)[:, cutoff_idx:] = np.nan
    return arrays


def folding_roles(*, sites: list[str], spec: core.ArmSpec) -> dict[str, np.ndarray]:
    """Return the `(16,)` boolean roles of the Primaries in an arm.

    - `visible`: the four visible signed Primaries (signed target; the sign head's labels).
    - `hidden`: D5 and D10.
    - `unfolded`: D15 and D18, whose stored readings are signed. The page names no folding for them, so
      they train on the readings as stored with the plain likelihood (no gate, no labels).
    - `folded_reading`: the Primaries whose reading is a folded magnitude: the eight MVA-only Primaries
      and the hidden pair (except in arm `S`, which reads the signed truth there).
    - `supervised`: the Primaries whose true sign trains the head (the four visible signed ones).
    """
    arr = lambda names: np.array([s in names for s in sites])  # noqa: E731
    visible, hidden, unfolded = (
        arr(core.VISIBLE_SIGNED),
        arr(core.HIDDEN_PAIR),
        arr(core.UNFOLDED_SIGNED),
    )
    hidden_signed = hidden & (spec.readings == "signed")
    return {
        "visible": visible, "hidden": hidden, "unfolded": unfolded, "supervised": visible,
        "folded_reading": ~visible & ~unfolded & ~hidden_signed,
    }  # fmt: skip


def dense_tables(*, hist_p: np.ndarray, hist_ok: np.ndarray, device: str) -> pvb.PvTables:
    """Return the stage-1 `PvTables`: the dense `p` read at the contract origin of every day."""
    n_grid = hist_p.shape[1]
    n_days = n_grid // lay.STEPS_PER_DAY
    rel = np.arange(lay.STEPS_PER_DAY * lay.N_LEAD_DAYS)
    idx = (np.arange(n_days) * lay.STEPS_PER_DAY + 24)[:, None] + rel[None, :]
    valid = idx < n_grid
    top = np.zeros((hist_p.shape[0], n_days, rel.size, 2), dtype=np.float32)
    top[..., 0] = np.where(valid[None], hist_p[:, np.clip(idx, 0, n_grid - 1)], 0.0)
    filled = (
        np.broadcast_to(valid, (hist_p.shape[0], n_days, rel.size)).copy()
        & hist_ok[:, np.clip(idx, 0, n_grid - 1)]
    )
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return pvb.PvTables(
        hist_p=to(hist_p), hist_ok=to(hist_ok), top_p0=to(top), top_filled=to(filled), first_day=0,
        run_us=to(np.zeros((hist_p.shape[0], n_days), dtype=np.int64)),
    )  # fmt: skip


def pv_tables(
    *, pipe, data, share_scale: np.ndarray, caps_mw: np.ndarray, power_cap_mw: np.ndarray, increment_pct: np.ndarray, std: np.ndarray,  # noqa: ANN001
    stage: int, shifted: bool, scale: float, device: str,
) -> pvb.PvTables:  # fmt: skip
    """Return the `PvTables` of a stage, built with experiment 716's functions.

    Stage 1 reads the dense `p`, and stages 2 and 3 read the ensemble table.
    """
    import e00716_pv as pv  # ty: ignore[unresolved-import]
    from e00422_prep import GRID_START_US, STEP_US  # ty: ignore[unresolved-import]

    raw = pipe.d632.dense_raw()
    n_grid = raw.shape[1]
    doy = pv.day_of_year(grid_start_us=pv.GRID_START_US, n_grid=n_grid)
    caps_total = caps_mw + scale * increment_pct / 100.0 * power_cap_mw
    ghi = temp = None
    if shifted:
        ghi, temp, _, _ = pv.shifted_sky(raw=raw)
    est = pv.dense_share(
        raw=raw,
        doy=doy,
        capacities_mw=caps_total,
        power_capacity_mw=power_cap_mw,
        ghi_override=ghi,
        temperature_override=temp,
    )
    hist_p, hist_ok = pv.p_dense(estimate=est, target_std=std)
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    if stage == 1:
        return dense_tables(hist_p=hist_p, hist_ok=hist_ok, device=device)
    mr, test = data.rows
    standardiser = mr.base.standardiser
    first_day, n_days = data.source.ens.first_day, data.source.ens.filled.shape[1]
    origin_idx = ((mr.origin_us - GRID_START_US) // STEP_US).astype(np.int64)
    raw_t = pv.target_raw_from_rows(
        x=mr.x, primary=mr.primary, origin_idx=origin_idx, target_idx=mr.target_idx, origin_us=mr.origin_us,
        mean=standardiser.mean, std=standardiser.std, first_day=first_day, n_days=n_days,
    )  # fmt: skip
    if test is not None:
        scored = pv.target_raw_from_rows(
            x=test.x, primary=test.primary, origin_idx=test.origin_idx, target_idx=test.target_idx, origin_us=test.origin,
            mean=standardiser.mean, std=standardiser.std, first_day=first_day, n_days=n_days,
        )  # fmt: skip
        raw_t = pv.merge_target_raw(base=raw_t, other=scored)
    donor = None
    run_us = raw_t.run_us
    if shifted:
        filled_days = np.nonzero(raw_t.filled.any(axis=(0, 2)))[0]
        first_ens = int(first_day + filled_days.min())
        abs_day = first_day + np.arange(n_days)
        donor_abs, _ = pv.donor_origin_day(origin_day=abs_day, first_ens_day=first_ens)
        donor = donor_abs - first_day
        run_us = np.where(
            (donor[None, :] >= 0) & (donor[None, :] < n_days),
            raw_t.run_us[:, np.clip(donor, 0, n_days - 1)],
            0,
        )
    top = pv.target_estimate(
        raw=raw_t, doy_of_day=pv.target_doy(first_day=first_day, n_days=n_days), share_scale=share_scale, target_std=std, donor_day=donor,
    )  # fmt: skip
    return pvb.PvTables(
        hist_p=to(hist_p), hist_ok=to(hist_ok), top_p0=to(top.p0), top_filled=to(top.filled), first_day=first_day,
        run_us=to(run_us.astype(np.int64)), exception_days=frozenset(),
    )  # fmt: skip


def series_id_array(*, sites: list[str]) -> np.ndarray:
    """Return `(16,)` experiment 87's `time_series_id` of each Primary in panel order."""
    import e00087_common as common  # ty: ignore[unresolved-import]

    ids = {
        r["site"]: int(r["time_series_id"])
        for r in common.series_table().to_dicts()
        if r["substation_type"] == "Primary"
    }
    return np.array([ids[s] for s in sites], dtype=np.int64)


def on_grid_713(*, arr: np.ndarray, n_grid: int) -> np.ndarray:
    """Return `arr` (on experiment 87's grid) on experiment 713's grid, NaN beyond experiment 87's end."""
    return sgb.pad_to_grid(a=arr, n_grid=n_grid)


def build_real_arm_world(
    *, job: fit.Job, data_dir: Path, device: str, phase: str, scale: float = 1.0
) -> trn.ArmWorld:
    """Return the `ArmWorld` of a real-fold job.

    The world (the injection, the folding, the gap rows) comes from experiment 720's panel and this
    experiment's worlds. Experiment 713's source supplies the weather, the date-time columns, the day
    types, and the ensemble table, and its power is replaced by the world's standardised target, so that
    the target, the history, the shift, and `q_v` all read the same series.

    `phase` is `train` or `predict`: in the train phase every value at or after the cutoff is NaN.
    """
    import e00397_prep as prep397  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    spec = core.ARM_SPECS[job.arm]
    gi = load_grid_inputs(fold=job.fold, seed=job.seed, phase=phase)
    pipe, panel, n_grid = gi.pipe, gi.panel, gi.n_grid
    data = inp.build_fold_data(pipe=pipe, device=device, keep_rows=True)
    world = "W0" if job.world == "W0" else "W2"
    cutoff = data.cutoff_idx
    w = build_fold_world_arrays(
        panel=panel,
        farm_for_site=gi.farms,
        world=world,
        cutoff_idx=cutoff if phase == "train" else None,
    )
    sites = list(panel.sites)
    roles = folding_roles(sites=sites, spec=spec)
    a = {
        k: on_grid_713(arr=v, n_grid=n_grid)
        for k, v in jobs.arm_arrays(
            spec=spec, signed=w.signed, fold_view=w.fold_view, signed_view=w.signed_view
        ).items()
    }
    signed = on_grid_713(arr=w.signed, n_grid=n_grid)
    reading = on_grid_713(arr=w.fold_view, n_grid=n_grid)
    u_real = on_grid_713(arr=np.nan_to_num(panel.u), n_grid=n_grid)
    k_grid = on_grid_713(arr=np.nan_to_num(panel.k), n_grid=n_grid)
    pre = np.arange(n_grid) < cutoff
    mean = np.nanmean(np.where(pre[None, :], a["target"], np.nan), axis=1)
    std = np.nanstd(np.where(pre[None, :], a["target"], np.nan), axis=1)
    power_cap = np.array(
        [float(r["effective_capacity_mw"]) for r in prep397.primaries().iter_rows(named=True)]
    )
    probe = ProbeProvider(root=fit.data_root(data_dir=data_dir))
    caps = capacities(fold=job.fold, sites=sites, world=world, probe=probe, power_cap_mw=power_cap)
    share_scale = share_scale_with_injection(
        caps_mw=caps, power_cap_mw=power_cap, increment_pct=w.increment_pct, scale=scale
    )
    shifted = spec.shift_days > 0

    def z(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((arr - mean[:, None]) / std[:, None]).astype(np.float32)).to(
            device
        )

    world_source = sgb.with_world_target(source=data.source, power_z=z(a["target"]))
    supervised, folded_reading = roles["supervised"], roles["folded_reading"]
    # the page: the likelihood gate reads the real sky in the placebos, so `u_grid` is the real `u` in every arm; only `p` and `p_0` are shifted.
    sign = np.where(
        np.isfinite(signed) & supervised[:, None], (signed < 0).astype(np.float32), np.nan
    )
    sources: dict[int, sgb.SgSource] = {}
    for stage in (1, 2):
        tables = pv_tables(
            pipe=pipe, data=data, share_scale=share_scale, caps_mw=caps, power_cap_mw=power_cap, increment_pct=w.increment_pct, std=std,
            stage=stage, shifted=shifted, scale=scale, device=device,
        )  # fmt: skip
        src = pvb.PvSource(source=world_source, tables=tables, world=stage == 1)
        sources[stage] = sgb.SgSource(
            pv=src, plain_z=z(a["plain"]), fold_z=z(a["fold"]), u_grid=torch.from_numpy(u_real.astype(np.float32)).to(device),
            sign_grid=torch.from_numpy(sign.astype(np.float32)).to(device), always_folded=folded_reading, supervised=supervised,
        )  # fmt: skip
    sources[3] = sources[2]
    gate = np.array(
        [
            float(wd.primary_gate_open(reading_points=w.fold_view[i, pre[: w.fold_view.shape[1]]] * 100.0, u=panel.u[i, pre[: w.fold_view.shape[1]]], k=panel.k[i, pre[: w.fold_view.shape[1]]], valid_us=panel.stamp_us[pre[: w.fold_view.shape[1]]])["open"])
            for i in range(16)
        ]
    )  # fmt: skip
    gate[roles["unfolded"]] = 0.0
    if spec.sign_head:
        shut = [sites[i] for i in np.nonzero(supervised & (gate < 0.5))[0]]
        assert not shut, (
            f"the export gate is shut at the supervised Primaries {shut}: their sign labels would be discarded"
        )
    tworld = jobs.tr.World(
        source=sources[1], cutoff_idx=cutoff, clean=a["target"], target_mean=mean, target_std=std, month=job.fold, placebo_pool=None,
    )  # fmt: skip
    hidden_ids = tuple(sites.index(s) for s in core.HIDDEN_PAIR)
    mva_ids = tuple(sites.index(s) for s in core.MVA_ONLY)
    score_c = (
        np.asarray(data.test_origin_idx, dtype=np.int64) + lay.NATIVE_OFFSET_STEPS
        if data.test_origin_idx is not None
        else None
    )
    return trn.ArmWorld(
        spec=spec, world=tworld, sources=sources, clean_plain=a["plain"], clean_fold=a["fold"],
        folded_target=folded_reading if spec.loss == "folded" else np.zeros(16, dtype=bool), shift=mean / std,
        q_v=np.full(16, core.REACTIVE_FLOOR_POINTS / 100.0) / std, gate=gate, val_primaries=hidden_ids, seed=job.seed, fold=job.fold,
        signed_truth=signed, reading=reading, predict_stage=2, score_c_idx=score_c, score_primaries=(*hidden_ids, *mva_ids),
        train_primaries=None, stamp_us=gi.stamp_us, k_grid=k_grid, series_ids=series_id_array(sites=sites),
    )  # fmt: skip


def synthetic_day_offset_points(*, root: Path) -> float:
    """Return the synthetic world's day offset in points.

    The offset is 0 unless Stage 0 wrote `SYNTH_WORLD_FILE` with another value.
    """
    path = root / SYNTH_WORLD_FILE
    return float(json.loads(path.read_text())["day_offset_points"]) if path.exists() else 0.0


def synthetic_inputs(*, gi: GridInputs, seed: int, day_offset_points: float) -> dict[str, Any]:
    """Return the synthetic world's arrays on the real grid (pure given `gi`, and used by Stage 0 as well).

    The gross demand is `known_demand` of the real calendar (bank holidays included) and the real 2 m
    temperature; the PV term is `C u` with the real shortwave likelihood `u`.
    """
    import e00713_known_answer as ka  # ty: ignore[unresolved-import]
    from e00422_prep import GRID_START_US  # ty: ignore[unresolved-import]

    panel, n_grid = gi.panel, len(gi.panel.stamp_us)
    temperature = np.where(
        np.isfinite(panel.temperature_c), panel.temperature_c, np.nanmedian(panel.temperature_c)
    )
    holidays = lay.england_bank_holidays(years=range(2019, 2029))
    day_type = lay.day_types_for_grid(grid_start_us=GRID_START_US, n_grid=n_grid, holidays=holidays)
    half_hour = lay.local_half_hours_for_grid(grid_start_us=GRID_START_US, n_grid=n_grid)
    sites = list(panel.sites)
    exporters = tuple(sites.index(s) for s in core.SYNTH_EXPORTERS)
    non_exporters = tuple(sites.index(s) for s in core.SYNTH_NON_EXPORTERS)
    four = (*exporters, *non_exporters)
    u = np.nan_to_num(panel.u)
    syn, signed, reading = wd.synthetic_on_real_grid(
        temperature_c=temperature, u=u, stamp_us=panel.stamp_us, day_type=day_type, half_hour=half_hour, table=ka.draw_table(),
        primaries=four, exporters=exporters, seed=seed, day_offset_points=day_offset_points,
    )  # fmt: skip
    return {
        "syn": syn, "signed": signed, "reading": reading, "u": u, "k": np.nan_to_num(panel.k), "exporters": exporters,
        "non_exporters": non_exporters, "four": four, "day_type": day_type, "half_hour": half_hour,
    }  # fmt: skip


def synthetic_score_origins(*, n_grid: int) -> np.ndarray:
    """Return the contract origins (12:00 UTC, every day) of the synthetic fit's scored months.

    The days run from 2026-03-01 until the last origin whose 10 days of targets end before the
    2026-07-01 holdout. The synthetic arrays have no real label, so any origin in the range is scored.
    """
    import e00721_stage0 as stage0

    start, stop = stage0.synthetic_scored_span(n_grid=n_grid)
    holdout = int(
        (np.datetime64(core.HOLDOUT_START_ISO, "us").astype(np.int64) - lay.GRID_START_US)
        // lay.STEP_US
    )
    last = min(stop, holdout) - lay.STEPS_PER_DAY * lay.N_LEAD_DAYS
    days = np.arange(start // lay.STEPS_PER_DAY, last // lay.STEPS_PER_DAY)
    return (days * lay.STEPS_PER_DAY + 24).astype(np.int64)


def build_synthetic_arm_world(
    *, job: fit.Job, data_dir: Path, device: str, phase: str
) -> trn.ArmWorld:
    """Return the `ArmWorld` of a synthetic-world job: four synthetic Primaries on the real grid.

    The network reads the real weather of experiment 713's pipeline, and the target is the synthetic
    reading (experiment 713's own power is replaced). All four Primaries train on the folded reading, and
    the true sign of all four trains the sign head (the page does not say which labels the synthetic
    world uses). Stage 1 only, so the world mode of `PvSource` reads the dense `p`.
    """
    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    spec = core.ARM_SPECS[job.arm]
    gi = load_grid_inputs(
        fold=core.SYNTH_PIPELINE_FOLD, seed=job.seed, phase=phase, with_farms=False
    )
    data = inp.build_fold_data(pipe=gi.pipe, device=device, keep_rows=True)
    n_grid, cutoff = gi.n_grid, data.cutoff_idx
    parts = synthetic_inputs(
        gi=gi,
        seed=job.seed,
        day_offset_points=synthetic_day_offset_points(root=fit.data_root(data_dir=data_dir)),
    )
    syn, four, exporters, non_exporters = (
        parts["syn"],
        parts["four"],
        parts["exporters"],
        parts["non_exporters"],
    )
    signed, reading = (
        on_grid_713(arr=parts["signed"], n_grid=n_grid),
        on_grid_713(arr=parts["reading"], n_grid=n_grid),
    )
    u_real, k_grid = (
        on_grid_713(arr=parts["u"], n_grid=n_grid),
        on_grid_713(arr=parts["k"], n_grid=n_grid),
    )
    cap = np.zeros(16)
    cap[list(four)] = syn.capacity_share
    if phase == "train":
        signed[:, cutoff:] = np.nan
        reading[:, cutoff:] = np.nan
    pre = np.arange(n_grid) < cutoff
    mean, std = np.zeros(16), np.ones(16)
    for i in four:
        mean[i], std[i] = np.nanmean(reading[i, pre]), np.nanstd(reading[i, pre])
    supervised = np.zeros(16, dtype=bool)
    supervised[list(four)] = True
    u_for_p = (
        np.nan_to_num(wd.shifted_sky(a=u_real, days=spec.shift_days))
        if spec.shift_days > 0
        else u_real
    )

    def z(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((arr - mean[:, None]) / std[:, None]).astype(np.float32)).to(
            device
        )

    p_all = np.clip(cap[:, None] * u_for_p / std[:, None], 0.0, 10.0).astype(np.float32)
    tables = dense_tables(hist_p=p_all, hist_ok=np.isfinite(reading) | ~pre[None, :], device=device)
    sign = np.where(
        np.isfinite(signed) & supervised[:, None], (signed < 0).astype(np.float32), np.nan
    )
    world_source = sgb.with_world_target(source=data.source, power_z=z(reading))
    source = sgb.SgSource(
        pv=pvb.PvSource(source=world_source, tables=tables, world=True), plain_z=z(reading), fold_z=z(reading),
        u_grid=torch.from_numpy(u_real.astype(np.float32)).to(device), sign_grid=torch.from_numpy(sign.astype(np.float32)).to(device),
        always_folded=supervised, supervised=supervised,
    )  # fmt: skip
    gate = np.zeros(16)
    gate[list(exporters)] = 1.0
    tworld = jobs.tr.World(
        source=source, cutoff_idx=cutoff, clean=reading, target_mean=mean, target_std=std, month=job.fold, placebo_pool=None,
    )  # fmt: skip
    score_c = synthetic_score_origins(n_grid=n_grid) if phase == "predict" else None
    return trn.ArmWorld(
        spec=spec, world=tworld, sources={1: source}, clean_plain=reading, clean_fold=reading, folded_target=supervised,
        shift=mean / std, q_v=np.full(16, core.REACTIVE_FLOOR_POINTS / 100.0) / std, gate=gate, val_primaries=four, seed=job.seed,
        fold=job.fold, signed_truth=signed, reading=reading, predict_stage=1, score_c_idx=score_c, score_primaries=four,
        train_primaries=four, stamp_us=gi.stamp_us, k_grid=k_grid, series_ids=series_id_array(sites=list(gi.panel.sites)), exporters=exporters, non_exporters=non_exporters,
    )  # fmt: skip


def make_world_builder(*, data_dir: Path, device: str, phase: str):  # noqa: ANN201
    """Return a builder of real `ArmWorld`s for `WorldInputs`.

    A real fold is built in one phase per process (`E00667_PHASE`): the fit process builds with
    `phase="train"`, and the predict process with `phase="predict"`.
    """
    assert os.environ.get("E00667_PHASE", phase) == phase

    def make(job: fit.Job) -> trn.ArmWorld:
        if job.world == "synthetic":
            return build_synthetic_arm_world(job=job, data_dir=data_dir, device=device, phase=phase)
        return build_real_arm_world(
            job=job, data_dir=data_dir, device=device, phase=phase, scale=job.capacity_scale
        )

    return make

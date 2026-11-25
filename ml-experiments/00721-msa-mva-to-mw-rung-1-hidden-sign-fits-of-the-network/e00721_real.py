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

import json
import os
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


def build_fold_world_arrays(
    *, panel: Any, farm_for_site: dict[str, np.ndarray], world: str, cutoff_idx: int | None
) -> wd.WorldArrays:
    """Return the world's arrays for a fold, after the holdout guard.

    In the train phase, every value at or after the cutoff is NaN.
    """
    wd.holdout_guard(valid_us=panel.stamp_us[np.isfinite(panel.power_pct).any(axis=0)])
    arrays = wd.build_world_arrays(
        world=world, sites=panel.sites, power_pct=panel.power_pct, k=panel.k, stamp_us=panel.stamp_us, farm_for_site=farm_for_site,
    )  # fmt: skip
    if cutoff_idx is not None:
        for name in ("signed", "fold_view", "signed_view"):
            getattr(arrays, name)[:, cutoff_idx:] = np.nan
    return arrays


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
        n_days = n_grid // lay.STEPS_PER_DAY
        rel = np.arange(lay.STEPS_PER_DAY * lay.N_LEAD_DAYS)
        idx = (np.arange(n_days) * lay.STEPS_PER_DAY + 24)[:, None] + rel[None, :]
        valid = idx < n_grid
        top = np.zeros((16, n_days, rel.size, 2), dtype=np.float32)
        top[..., 0] = np.where(valid[None], hist_p[:, np.clip(idx, 0, n_grid - 1)], 0.0)
        filled = (
            np.broadcast_to(valid, (16, n_days, rel.size)).copy()
            & hist_ok[:, np.clip(idx, 0, n_grid - 1)]
        )
        return pvb.PvTables(
            hist_p=to(hist_p), hist_ok=to(hist_ok), top_p0=to(top), top_filled=to(filled), first_day=0,
            run_us=to(np.zeros((16, n_days), dtype=np.int64)),
        )  # fmt: skip
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


def build_real_arm_world(
    *, job: fit.Job, data_dir: Path, device: str, phase: str, scale: float = 1.0
) -> trn.ArmWorld:
    """Return the `ArmWorld` of a real-fold job.

    The world comes from experiment 713's pipeline, experiment 720's panel, and the worlds of this
    experiment.

    `phase` is `train` or `predict` and is passed to experiment 713's pipeline, which masks everything at
    or after the cutoff in the train phase (this module also sets the world's power to NaN there).
    """
    import e00397_prep as prep397  # ty: ignore[unresolved-import]
    import e00506_data as d506  # ty: ignore[unresolved-import]
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00720_real as real  # ty: ignore[unresolved-import]

    spec = core.ARM_SPECS[job.arm]
    pipe = inp.load_pipeline(
        month=job.fold, phase=phase, chain_root=inp.chain_root_of_seed(seed=job.seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device, keep_rows=True)
    panel = real.load_panel(dense_raw_path=Path(pipe.d632.CACHE) / "dense_raw.npy")
    n_grid = len(panel.stamp_us)
    farms = {
        s: d506.farm_norm(farm=d506.FARM_OF_SITE[s])
        for s in (*core.HIDDEN_PAIR, *core.VISIBLE_SIGNED)
    }
    world = "W0" if job.world == "W0" else "W2"
    cutoff = data.cutoff_idx if phase == "train" else None
    w = build_fold_world_arrays(panel=panel, farm_for_site=farms, world=world, cutoff_idx=cutoff)
    a = jobs.arm_arrays(
        spec=spec, signed=w.signed, fold_view=w.fold_view, signed_view=w.signed_view
    )
    pre = np.arange(n_grid) < data.cutoff_idx
    mean = np.nanmean(np.where(pre[None, :], a["target"], np.nan), axis=1)
    std = np.nanstd(np.where(pre[None, :], a["target"], np.nan), axis=1)
    power_cap = np.array(
        [float(r["effective_capacity_mw"]) for r in prep397.primaries().iter_rows(named=True)]
    )
    probe = ProbeProvider(root=fit.data_root(data_dir=data_dir))
    caps = capacities(
        fold=job.fold, sites=panel.sites, world=world, probe=probe, power_cap_mw=power_cap
    )
    share_scale = share_scale_with_injection(
        caps_mw=caps, power_cap_mw=power_cap, increment_pct=w.increment_pct, scale=scale
    )
    shifted = spec.shift_days > 0
    u = (
        np.nan_to_num(wd.shifted_sky(a=panel.u, days=spec.shift_days))
        if shifted
        else np.nan_to_num(panel.u)
    )

    def z(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((arr - mean[:, None]) / std[:, None]).astype(np.float32)).to(
            device
        )

    supervised = np.array([s in SUPERVISED_SITES for s in panel.sites])
    hidden = np.array([s in core.HIDDEN_PAIR for s in panel.sites])
    always_folded = ~np.array([s in core.VISIBLE_SIGNED for s in panel.sites]) & ~(
        hidden & (spec.readings == "signed")
    )
    sign = np.where(
        np.isfinite(w.signed) & supervised[:, None], (w.signed < 0).astype(np.float32), np.nan
    )
    sources: dict[int, sgb.SgSource] = {}
    for stage in (1, 2):
        tables = pv_tables(
            pipe=pipe, data=data, share_scale=share_scale, caps_mw=caps, power_cap_mw=power_cap, increment_pct=w.increment_pct, std=std,
            stage=stage, shifted=shifted, scale=scale, device=device,
        )  # fmt: skip
        src = pvb.PvSource(source=data.source, tables=tables, world=stage == 1)
        sources[stage] = sgb.SgSource(
            pv=src, plain_z=z(a["plain"]), fold_z=z(a["fold"]), u_grid=torch.from_numpy(u.astype(np.float32)).to(device),
            sign_grid=torch.from_numpy(sign.astype(np.float32)).to(device), always_folded=always_folded, supervised=supervised,
        )  # fmt: skip
    sources[3] = sources[2]
    gate = np.array(
        [
            float(wd.primary_gate_open(reading_points=w.fold_view[i, pre] * 100.0, u=u[i, pre], k=panel.k[i, pre], valid_us=panel.stamp_us[pre])["open"])
            for i in range(16)
        ]
    )  # fmt: skip
    tworld = jobs.tr.World(
        source=sources[1], cutoff_idx=data.cutoff_idx, clean=a["target"], target_mean=mean, target_std=std, month=job.fold, placebo_pool=None,
    )  # fmt: skip
    hidden_ids = tuple(panel.sites.index(s) for s in core.HIDDEN_PAIR)
    mva_ids = tuple(panel.sites.index(s) for s in core.MVA_ONLY)
    score_c = (
        np.asarray(data.test_origin_idx, dtype=np.int64) + lay.NATIVE_OFFSET_STEPS
        if data.test_origin_idx is not None
        else None
    )
    return trn.ArmWorld(
        spec=spec, world=tworld, sources=sources, clean_plain=a["plain"], clean_fold=a["fold"],
        folded_target=always_folded if spec.loss == "folded" else np.zeros(16, dtype=bool), shift=mean / std,
        q_v=np.full(16, core.REACTIVE_FLOOR_POINTS / 100.0) / std, gate=gate, val_primaries=hidden_ids, seed=job.seed, fold=job.fold,
        signed_truth=w.signed, reading=w.fold_view, predict_stage=2, score_c_idx=score_c, score_primaries=(*hidden_ids, *mva_ids),
        train_primaries=None, stamp_us=panel.stamp_us,
    )  # fmt: skip


def make_world_builder(*, data_dir: Path, device: str, phase: str):  # noqa: ANN201
    """Return a builder of real `ArmWorld`s for `WorldInputs`.

    A real fold is built in one phase per process (`E00667_PHASE`): the fit process builds with
    `phase="train"`, and the predict process with `phase="predict"`.
    """
    assert os.environ.get("E00667_PHASE", phase) == phase

    def make(job: fit.Job) -> trn.ArmWorld:
        return build_real_arm_world(job=job, data_dir=data_dir, device=device, phase=phase)

    return make

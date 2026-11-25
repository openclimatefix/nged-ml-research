# ruff: noqa: E501, PLR0915
"""Experiment 721: the code-works pass on a synthetic mini world (no real data, no rung 1 fit).

The mini world is random weather on a grid of 105 days with a signed demand-minus-PV process that
the real synthetic world also uses (`e00721_worlds.build_synthetic`). It runs the whole chain: the
batch builder (`SgSource` over experiment 716's `PvSource` over experiment 713's `Source`), the fit
loop with its stages, the stored states, the predict phase, the scoring driver, and the report. The pass
count is cut to a few with `E00721_E2E_PASSES`, and every output goes under a scratch path that the
caller names. It proves that the code runs end to end and says nothing about any forecast.

Usage: run.sh --go e00721_e2e.py SCRATCH_DIR [--passes 2]
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Final

import e00721_paths

# isort: split
import e00713_batch as bt  # ty: ignore[unresolved-import]
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

N_DAYS: Final = 105
CUTOFF_DAY: Final = 85
ENS_FIRST_DAY: Final = 20
ENS_DAYS: Final = 80
SITES: Final = (
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D19", "D21", "D22", "D1",
)  # fmt: skip
"""16 labels in Primary order: the 14 named in `core`, D15, and one more."""
SYNTH_PRIMARIES: Final = (0, 1, 2, 3)
"""The synthetic world's four Primaries; the first two export."""


def site_index(*names: str) -> tuple[int, ...]:
    """Return the Primary indices of the sites `names`."""
    return tuple(SITES.index(n) for n in names)


class MiniWorld:
    """The arrays of one mini world, in fractions of capacity, built once per seed and exporter set."""

    def __init__(self, *, seed: int, world: str) -> None:
        """Build the calendar, sky, demand, and the signed and folded power of 16 Primaries."""
        rng = np.random.default_rng([721, seed])
        n_grid = lay.STEPS_PER_DAY * N_DAYS
        self.n_grid, self.world, self.seed = n_grid, world, seed
        hod = (np.arange(n_grid) % 48) / 2.0
        day = np.arange(n_grid) // 48
        cloud = rng.uniform(0.3, 1.0, size=(16, N_DAYS))
        shape = np.clip(np.sin(np.pi * (hod - 5.0) / 14.0), 0.0, 1.0)
        self.u = shape[None, :] * cloud[:, day]
        temp = 8.0 + 4.0 * rng.standard_normal((16, n_grid)).cumsum(axis=1) / np.sqrt(
            np.arange(1, n_grid + 1)
        )
        table = 0.05 * np.sin(2 * np.pi * np.arange(48) / 48)[None, :] * np.ones((8, 1))
        demand_mean = wd.known_demand(
            temperature_c=temp,
            day_type=(day % 7).astype(np.int64),
            half_hour=(np.arange(n_grid) % 48),
            table=table,
        )
        month = np.full(n_grid, 6)
        exporter = np.zeros(16, dtype=bool)
        exporter[list(site_index(*core.VISIBLE_SIGNED))] = True
        if world == "W2":
            exporter[list(site_index(*core.HIDDEN_PAIR))] = True
        self.syn = wd.build_synthetic(
            demand_mean=demand_mean,
            u=self.u,
            month=month,
            exporter=exporter,
            seed=seed,
            day_offset_points=5.0,
        )
        visible = np.zeros(16, dtype=bool)
        visible[list(site_index(*core.VISIBLE_SIGNED))] = True
        hidden = np.zeros(16, dtype=bool)
        hidden[list(site_index(*core.HIDDEN_PAIR))] = True
        self.visible, self.hidden = visible, hidden
        self.signed = np.where((visible | hidden)[:, None], self.syn.x, np.nan)
        self.fold_view = self.syn.reading
        self.signed_view = np.where(visible[:, None], self.syn.x, self.syn.reading)
        self.weather = rng.standard_normal((16, n_grid, 21)).astype(np.float16)
        self.weather[..., 0] = (self.u - 0.3).astype(np.float16)
        self.datetime = rng.standard_normal((16, n_grid, 21)).astype(np.float16)
        self.day_type = torch.from_numpy((day % 7).astype(np.int64))
        self.ens_weather = rng.standard_normal((16, ENS_DAYS, 672, 21)).astype(np.float16)
        self.ens_spread = rng.random((16, ENS_DAYS, 672, 7)).astype(np.float16)
        idx = ((ENS_FIRST_DAY + np.arange(ENS_DAYS)) * 48 + 24)[:, None] + np.arange(672)[None, :]
        self.ens_u = np.where(idx < n_grid, self.u[:, np.clip(idx, 0, n_grid - 1)], 0.0)
        self.ens_weather[..., 0] = (self.ens_u - 0.3).astype(np.float16)
        self.stamp_us = lay.GRID_START_US + np.arange(n_grid, dtype=np.int64) * lay.STEP_US


def build_arm_world(*, job: fit.Job, mini: MiniWorld, device: str = "cpu") -> trn.ArmWorld:
    """Return the `ArmWorld` of `job` on `mini`: the sources of each stage, the arrays, and the scored rows."""
    spec = core.ARM_SPECS[job.arm]
    synthetic = job.world == "synthetic"
    exporters = site_index("D9", "D11") if synthetic else ()
    non_exporters = site_index("D2", "D3") if synthetic else ()
    four = np.zeros(16, dtype=bool)
    four[[*exporters, *non_exporters]] = True
    supervised = four if synthetic else mini.visible
    a = jobs.arm_arrays(
        spec=spec, signed=mini.signed, fold_view=mini.fold_view, signed_view=mini.signed_view
    )
    n_grid = mini.n_grid
    target = a["target"]
    mean = np.nanmean(target, axis=1)
    std = np.nanstd(target, axis=1) + 1e-6

    def z(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((arr - mean[:, None]) / std[:, None]).astype(np.float32)).to(
            device
        )

    shifted = spec.shift_days > 0
    u = np.nan_to_num(wd.shifted_sky(a=mini.u, days=spec.shift_days)) if shifted else mini.u
    weather = mini.weather.copy()
    ens_weather = mini.ens_weather.copy()
    if shifted:
        weather[..., 0] = (u - 0.3).astype(np.float16)
        sh = np.nan_to_num(
            wd.shifted_sky(a=mini.ens_u.reshape(16, -1), days=spec.shift_days)
        ).reshape(mini.ens_u.shape)
        ens_weather[..., 0] = (sh - 0.3).astype(np.float16)
    cap = mini.syn.capacity_share
    p_all = np.clip(cap[:, None] * u / std[:, None], 0.0, 10.0).astype(np.float32)
    run = torch.zeros(16, ENS_DAYS, dtype=torch.int64)
    c = (ENS_FIRST_DAY + torch.arange(ENS_DAYS)) * 48 + 24
    run[:] = (
        lay.GRID_START_US + (c - lay.NATIVE_OFFSET_STEPS) * lay.STEP_US - lay.ENSEMBLE_LEAD_US
    )[None, :]
    ens = bt.EnsTensors(
        weather=torch.from_numpy(ens_weather).to(device), spread=torch.from_numpy(mini.ens_spread).to(device),
        filled=torch.ones(16, ENS_DAYS, 672, dtype=torch.bool, device=device), first_day=ENS_FIRST_DAY, run_us=run.to(device),
    )  # fmt: skip
    power_z = z(target)

    def top_table(*, first_day: int, n_days: int) -> tuple[torch.Tensor, torch.Tensor]:
        rel = np.arange(672)
        idx = ((first_day + np.arange(n_days)) * 48 + 24)[:, None] + rel[None, :]
        valid = idx < n_grid
        p0 = np.zeros((16, n_days, 672, 2), dtype=np.float32)
        p0[..., 0] = np.where(valid[None], p_all[:, np.clip(idx, 0, n_grid - 1)], 0.0)
        return torch.from_numpy(p0).to(device), torch.from_numpy(
            np.broadcast_to(valid, (16, n_days, 672)).copy()
        ).to(device)

    def source(*, stage: int) -> sgb.SgSource:
        base = bt.Source(
            power_z=power_z, dense_weather=torch.from_numpy(weather).to(device), dense_datetime=torch.from_numpy(mini.datetime).to(device),
            day_type=mini.day_type.to(device), ens=ens if stage >= 2 else None,
        )  # fmt: skip
        first, n_days = (ENS_FIRST_DAY, ENS_DAYS) if stage >= 2 else (0, N_DAYS)
        p0, filled = top_table(first_day=first, n_days=n_days)
        tables = pvb.PvTables(
            hist_p=torch.from_numpy(p_all).to(device), hist_ok=torch.ones(16, n_grid, dtype=torch.bool, device=device),
            top_p0=p0, top_filled=filled, first_day=first,
            run_us=run.to(device) if stage >= 2 else torch.zeros(16, N_DAYS, dtype=torch.int64, device=device),
        )  # fmt: skip
        pv = pvb.PvSource(source=base, tables=tables, world=stage == 1)
        always_folded = (
            four if synthetic else ~mini.visible & ~(mini.hidden & (spec.readings == "signed"))
        )
        sign = np.where(np.isfinite(mini.signed), (mini.signed < 0).astype(np.float32), np.nan)
        return sgb.SgSource(
            pv=pv, plain_z=z(a["plain"]), fold_z=z(a["fold"]), u_grid=torch.from_numpy(mini.u.astype(np.float32)).to(device),
            sign_grid=torch.from_numpy(sign.astype(np.float32)).to(device), always_folded=always_folded, supervised=supervised,
        )  # fmt: skip

    cutoff = CUTOFF_DAY * 48
    sources = {1: source(stage=1), 2: source(stage=2), 3: source(stage=2)}
    gate = (
        (mini.syn.exporter & ~mini.hidden).astype(float)
        if job.world == "W0"
        else mini.syn.exporter.astype(float)
    )
    if synthetic:
        gate = np.zeros(16)
        gate[list(exporters)] = 1.0
    world = jobs.tr.World(
        source=sources[1], cutoff_idx=cutoff, clean=target, target_mean=mean, target_std=std,
        month="2025-12", placebo_pool=None,
    )  # fmt: skip
    hidden_ids = site_index(*core.HIDDEN_PAIR)
    primaries_train = (
        (*exporters, *non_exporters)
        if synthetic
        else (*hidden_ids, *site_index(*core.VISIBLE_SIGNED), *site_index(*core.MVA_ONLY[:2]))
    )
    score_c = (np.arange(CUTOFF_DAY, CUTOFF_DAY + 5) * 48 + 24).astype(np.int64)
    return trn.ArmWorld(
        spec=spec, world=world, sources=sources, clean_plain=a["plain"], clean_fold=a["fold"],
        folded_target=~mini.visible & ~(mini.hidden & (spec.readings == "signed")) if spec.loss == "folded" else np.zeros(16, dtype=bool),
        shift=mean / std, q_v=np.full(16, core.REACTIVE_FLOOR_POINTS / 100.0) / std, gate=gate, val_primaries=(*exporters, *non_exporters) if synthetic else hidden_ids,
        seed=job.seed, fold=job.fold, signed_truth=mini.signed, reading=mini.fold_view, batch_size=64,
        predict_stage=1 if synthetic else 2, score_c_idx=score_c,
        score_primaries=(*exporters, *non_exporters) if synthetic else (*hidden_ids, *site_index(*core.MVA_ONLY[:2])),
        train_primaries=primaries_train, stamp_us=mini.stamp_us, k_grid=mini.u, exporters=exporters, non_exporters=non_exporters,
    )  # fmt: skip


def make_world_builder(*, device: str = "cpu"):  # noqa: ANN201
    """Return a builder of `ArmWorld`s over cached mini worlds."""
    cache: dict[tuple[int, str], MiniWorld] = {}

    def make(job: fit.Job) -> trn.ArmWorld:
        world = "W2" if job.world in {"W2", "synthetic"} else "W0"
        key = (job.seed, world)
        if key not in cache:
            cache[key] = MiniWorld(seed=job.seed, world=world)
        return build_arm_world(job=job, mini=cache[key], device=device)

    return make


E2E_JOBS: Final = (
    ("P0", "W2"), ("SG-0", "W2"), ("SG-A", "W2"), ("SG-A-S", "W2"), ("P0", "W0"), ("SG-A", "W0"), ("SG-A-open", "W0"),
)  # fmt: skip
"""The arms of the end-to-end pass: the arms of G1, G2, and G3 in one fold at one seed (`S` is left out to save time)."""


def wait_for_resources(*, max_wait_s: float = 3600.0) -> None:
    """Wait until the load and the free memory satisfy the compute rules (one check every 30 s)."""
    waited = 0.0
    while fit.resource_problems(load=fit.read_uptime_load(), free_gb=fit.read_free_gb()):
        if waited >= max_wait_s:
            msg = "the workstation stayed above the compute rules' load limit"
            raise RuntimeError(msg)
        time.sleep(30.0)
        waited += 30.0


def run(*, scratch: Path, passes: int, seeds: tuple[int, ...] = (0,)) -> dict:
    """Run every arm of both mini worlds for `seeds` and return the timings and the peak memory."""
    vars(core).update(
        REAL_FOLD_CAP_FIRST=passes, REAL_FOLD_CAP_RERUN=passes + 1, CONTINUATION_CAP_PASSES=passes,
        SYNTH_MAX_PASSES=passes,
    )  # fmt: skip
    inputs = jobs.WorldInputs(make_world=make_world_builder())
    budget = fit.Budget()
    timings: dict[str, dict] = {}
    all_jobs = [
        fit.Job(arm=arm, world=world, fold="2025-12", seed=s)
        for s in seeds
        for arm, world in E2E_JOBS
    ]
    all_jobs += [
        fit.Job(arm=arm, world="synthetic", fold="synthetic", seed=s)
        for s in seeds
        for arm in ("SG-A", "SG-A-S")
    ]
    for job in all_jobs:
        wait_for_resources()
        t0 = time.monotonic()
        curve = fit.run_job(
            job=job, inputs=inputs, data_dir=scratch, budget=budget, continuation=True
        )
        passes_run = sum(r["passes"] for r in curve["stages"])
        elapsed = time.monotonic() - t0
        timings[job.key] = {
            "seconds": round(elapsed, 2),
            "passes": passes_run,
            "seconds_per_pass": round(elapsed / max(passes_run, 1), 3),
        }
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {"timings": timings, "peak_host_mb": round(peak_mb), "jobs": [j.key for j in all_jobs]}


def main() -> int:
    """Run the pass and write `e2e_run.json` under the scratch path."""
    e00721_paths.pin_threads()
    parser = argparse.ArgumentParser()
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()
    assert str(args.scratch).startswith(str(Path.home())), "the scratch path must sit under $HOME"
    torch.set_num_threads(1)
    out = run(scratch=args.scratch, passes=args.passes)
    import e00721_driver as drv
    import e00721_report as rep

    hidden = dict(zip(core.HIDDEN_PAIR, site_index(*core.HIDDEN_PAIR), strict=True))
    mde_path = args.scratch / "mde.json"
    summary = drv.score_all(
        data_dir=args.scratch, folds=("2025-12",), seeds=(0,), hidden_ids=hidden, mva_ids=site_index(*core.MVA_ONLY[:2]),
        n_resamples=200, mde_path=mde_path, seed_difference=0.0,
    )  # fmt: skip
    drv.write_summary(summary=summary, path=args.scratch / "summary.json")
    out["report"] = rep.format_report(
        summary=summary, mde=core.read_mde_file(path=mde_path)
    ).splitlines()[:2]
    out["summary_keys"] = sorted(summary)
    (args.scratch / "e2e_run.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                "peak_host_mb": out["peak_host_mb"],
                "seconds_per_pass": {k: v["seconds_per_pass"] for k, v in out["timings"].items()},
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stages of experiment 734, one command per stage.

    e00734_run.py stage0
    e00734_run.py gallery
    e00734_run.py gates
    e00734_run.py tune --phase {own,own_ext,own_ext2,nbr} --worker i --of n
    e00734_run.py tune --extend-pelt|--extend-kappa|--extend-tau --worker i --of n
        (each explicit and once; --extend-tau also needs the H-nbr re-decode it starts)
    e00734_run.py choose --phase {own,final,g3b}
    e00734_run.py score --set <set> --worker i --of n
        (set: default, cascade, cold, long_heavy, short_heavy, constant_mw_square, solar, bias,
         uninjected; run `uninjected` before `solar` and `bias`, and before `report`)
    e00734_run.py real
    e00734_run.py report
    e00734_run.py figures --which {scored,real}

`run.sh` sets the environment and the memory-capped scope. Every output goes under `OUT`, never
into another experiment's folder, and every stage that reads a scored world refuses to start
without the review markers.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Final

import e00734_bank as bank
import e00734_decode as dec
import e00734_edge as edge
import e00734_events as ev
import e00734_hmm as hmm
import e00734_panel as pan
import e00734_store as store
import e00734_tasks as tasks
import e00734_tuning as tuning
import numpy as np
import polars as pl
from e00653_common import GRID_LENGTH, SCORED_FROM  # ty: ignore[unresolved-import]

OUT: Final[Path] = tasks.OUT
PARAMS: Final[Path] = OUT / "params.json"
PARAMS_SHA: Final[Path] = OUT / "params.sha256"
CODE_SHA: Final[Path] = OUT / "code.sha256"
"""Hash of the scoring code, written by `freeze-code` before stage 3."""
EXIT_FAILED: Final[int] = 3
"""Exit code of a failed world (the launcher's busy code is different)."""
STOP_CPU_HOURS: Final[float] = 90.0
SETS: Final[tuple[str, ...]] = (
    "default",
    "cascade",
    "cold",
    "long_heavy",
    "short_heavy",
    "constant_mw_square",
    "solar",
    "bias",
    "uninjected",
)
SEEDS: Final[dict[str, tuple[int, ...]]] = {
    "default": bank.SCORED_DEFAULT_SEEDS,
    "cascade": bank.CASCADE_SEEDS,
    "cold": bank.COLD_SEEDS,
    "long_heavy": bank.SENSITIVITY_SEEDS,
    "short_heavy": bank.SENSITIVITY_SEEDS,
    "constant_mw_square": bank.SENSITIVITY_SEEDS,
}
HASATTR: Final[dict[str, tuple[str, ...]]] = {
    "e00653_common": (
        "GRID_LENGTH",
        "BLOCK",
        "N_BLOCKS",
        "SCORED_FROM",
        "block_means",
        "Panel",
        "WORK",
        "E580",
    ),
    "e00653_core": ("freeze", "decode_world", "match"),
    "e00653_slow": (
        "fit_slow",
        "to_half_hours",
        "noise_unit",
        "rolling_mean",
        "MASK_UNITS",
        "MASK_WIDEN",
        "HUBER_K",
    ),
    "e00653_segment": ("Event", "Params", "classify", "refine"),
    "e00653_inject": ("GenParams", "inject", "Truth", "OFFSET_2019"),
    "e00653_run": ("setup", "_overlap"),
    "e00660_ev34_detector": ("Rig", "truths"),
    "e00660_generator": (
        "inject",
        "apply_events",
        "shuffled_placebo",
        "event_windows",
        "GeneratorConfig",
        "event_delta",
        "default_config",
        "PowerPanel",
    ),
    "e00665_score": ("cell_of", "with_cells"),
    "e00670_run": (
        "params653",
        "CONFIGS",
        "CELLS",
        "event_free",
        "logged_shifts",
        "matched",
        "verified_d5_d16",
        "record_end_of",
        "events_table",
    ),
    "e00680_run": ("decode_variants",),
    "e00680_search": ("search_both_ends",),
    "e00686_false_events": ("seed_of", "N_WORLDS"),
}


def pin_blas() -> None:
    """Set one thread per numerical library before any worker pool starts."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[k] = "1"


def sha256_of(*, path: Path) -> str:
    """Hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_review(*, marker: str) -> None:
    """Stop unless a review marker exists."""
    if not (OUT / marker).exists():
        sys.exit(f"refusing to run: {OUT / marker} does not exist (the review has not passed)")


def code_hash() -> str:
    """SHA-256 over the names and bytes of every non-test module here, and the C kernel."""
    here = Path(__file__).parent
    files = sorted([*here.glob("e00734_*.py"), *here.glob("e00734_*.c")])
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def require_code_frozen() -> None:
    """Stop unless the code hash equals the one frozen with the stage 3 review."""
    if not CODE_SHA.exists() or code_hash() != CODE_SHA.read_text().split()[0]:
        sys.exit("refusing to run: the scoring code differs from code.sha256 (run freeze-code)")


def load_params() -> dict:
    """The tuned values, after checking the code hash, their hash, and the stage 3 marker."""
    require_review(marker="review_stage3.ok")
    require_code_frozen()
    if not PARAMS.exists() or sha256_of(path=PARAMS) != PARAMS_SHA.read_text().split()[0]:
        sys.exit("refusing to run: params.json is missing or its hash does not match params.sha256")
    return json.loads(PARAMS.read_text())


def shard(*, items: list, worker: int, of: int) -> list:
    """The items of worker `worker` out of `of` workers."""
    return items[worker::of]


def run_guarded(*, name: str, fn: Callable[..., None], **kwargs: object) -> None:
    """Run one task; on failure write the message and traceback, and exit with code 3."""
    if not run_task(name=name, fn=fn, **kwargs):
        sys.exit(EXIT_FAILED)


def run_task(*, name: str, fn: Callable[..., None], **kwargs: object) -> bool:
    """Run one task unless its done marker exists; record a failure and return False on error.

    A finished task writes `OUT/done/<name>.ok`, holding the time and the Viterbi kernels this
    process has decoded with so far (`hmm.KERNELS_USED`), so a re-run after a crash skips every
    task that already finished and repeats only the ones that did not.

    Args:
        name: The task's name, unique within the experiment.
        fn: The task.
        **kwargs: The task's keyword arguments.

    Returns:
        True when the task finished now or earlier, False when it failed.
    """
    done = OUT / "done" / f"{name}.ok"
    if done.exists():
        print(f"skip {name}: already done", flush=True)
        return True
    try:
        fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - the failure record is the point of this boundary
        tasks.atomic_json(
            obj={"task": name, "message": str(exc), "traceback": traceback.format_exc()},
            path=OUT / "failed" / f"{name}.json",
        )
        print(f"FAILED {name}: {exc}", file=sys.stderr, flush=True)
        return False
    done.parent.mkdir(parents=True, exist_ok=True)
    kernels = ",".join(sorted(hmm.KERNELS_USED)) or "none"
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    done.write_text(
        f"{time.strftime('%FT%TZ', time.gmtime())} viterbi_kernels={kernels} "
        f"peak_rss_gb={peak_gb:.2f}\n"
    )
    print(f"peak resident memory of this worker so far: {peak_gb:.2f} GB", flush=True)
    return True


def run_shard(*, names_and_kwargs: list[tuple[str, dict]], fn: Callable[..., None]) -> None:
    """Run a worker's tasks in turn, stopping at the CPU budget; exit with code 3 on any failure."""
    failed = 0
    for name, kwargs in names_and_kwargs:
        require_budget()
        failed += not run_task(name=name, fn=fn, **kwargs)
    if failed:
        sys.exit(EXIT_FAILED)


BUDGET_MARKER: Final[str] = "cpu_budget_extended.ok"
"""Written by the manager to allow a run past `STOP_CPU_HOURS`."""


def spent_cpu_hours() -> float:
    """CPU-hours in the per-phase time log (one thread per task)."""
    log = OUT / "time_log.csv"
    if not log.exists():
        return 0.0
    total = 0.0
    for line in log.read_text().splitlines():
        parts = line.split(",")
        if len(parts) == 5:
            a, b = (calendar.timegm(time.strptime(v, "%Y-%m-%dT%H:%M:%S")) for v in parts[3:5])
            total += (b - a) / 3600.0
    return total


def require_budget() -> None:
    """Stop at the CPU-hour line, projected or spent; the marker only allows the line to be 55."""
    if STOP_CPU_HOURS > 40.0 and not (OUT / BUDGET_MARKER).exists():
        sys.exit(f"refusing to run: {OUT / BUDGET_MARKER} does not exist (the stop is above 40)")
    gates_file = OUT / "gates.json"
    if not gates_file.exists():
        sys.exit("refusing to run: gates.json does not exist (run and pass the gates first)")
    projected = json.loads(gates_file.read_text())["timed_world"]["projection_cpu_hours"]["total"]
    if projected > STOP_CPU_HOURS:
        sys.exit(
            f"refusing to run: the timed world projects {projected:.1f} CPU-hours, over the "
            f"{STOP_CPU_HOURS:.0f} CPU-hour stop; ask the manager (marker {BUDGET_MARKER})"
        )
    spent = spent_cpu_hours()
    if spent > STOP_CPU_HOURS:
        sys.exit(
            f"refusing to run: {spent:.1f} CPU-hours spent, over the {STOP_CPU_HOURS:.0f} stop"
        )


# stage 0 -----------------------------------------------------------------------------------


def _mad(x: np.ndarray) -> float:
    v = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else float("nan")


def stage0() -> None:
    """Pre-checks on the uninjected real record of 2022 and 2023, in event-free blocks only.

    Per Primary: the share of block variance explained by the shared term, the noise unit with
    and without it, the on/off switch, and the noise unit and 30-day drift of three baselines.
    Nothing here changes stage 1.
    """
    ctx = dec.load_context()
    rig = ctx.rig
    busy = dec.event_free_masks(rig=rig, mask_tuning_years=False)
    power = rig.panel.power
    sh = pan.build_blocks(
        rig=rig,
        power=power,
        arm_input="shared",
        loading=None,
        switch=None,
        sigma_from="tune",
        calibration=ctx.cal_sh,
        n_blocks=pan.TUNE_BLOCKS,
    )
    cm = pan.build_blocks(
        rig=rig,
        power=power,
        arm_input="cm",
        loading=None,
        switch=None,
        sigma_from="tune",
        calibration=ctx.cal_cm,
        n_blocks=pan.TUNE_BLOCKS,
    )
    out: dict[str, dict] = {}
    for s in sh.sites:
        blocks_busy = (
            np.add.reduceat(
                busy[s][: pan.TUNE_BLOCKS * pan.BLOCK].astype(int),
                np.arange(0, pan.TUNE_BLOCKS * pan.BLOCK, pan.BLOCK),
            )
            > 0
        )
        ok = ~blocks_busy & np.isfinite(sh.y[s])
        r = (sh.y[s] - sh.slow[s])[ok]
        r_term = (
            sh.y[s]
            - sh.slow[s]
            - ctx.cal_sh.loading[s] * ctx.cal_sh.sigma_raw[s] * _term(sh=sh, ctx=ctx, s=s)
        )[ok]
        base = {
            "arm C forecast": _baseline_stats(r=np.where(ok, sh.y[s] - sh.slow[s], np.nan)),
            "weekly profile": _baseline_stats(
                r=np.where(ok, _weekly(x=pan.block_means(x=power[s])[: pan.TUNE_BLOCKS]), np.nan)
            ),
            "neighbourhood share": _baseline_stats(
                r=np.where(ok, _share(power=power, s=s, neighbours=ctx.neighbours), np.nan)
            ),
        }
        out[s] = {
            "n_event_free_blocks": int(ok.sum()),
            "explained_variance": float(1 - np.nanvar(r_term) / np.nanvar(r)),
            "noise_unit_without": _mad(r),
            "noise_unit_with": _mad(r_term),
            "shared_switch": bool(ctx.cal_sh.switch[s]),
            "loading": float(ctx.cal_sh.loading[s]),
            "sigma_cm": float(cm.sigma.get(s, float("nan"))),
            "baselines": base,
            "candidates_for_part_2": [
                name
                for name, v in base.items()
                if name != "arm C forecast"
                and v["noise_unit"] <= (1 - BASELINE_GAIN) * base["arm C forecast"]["noise_unit"]
            ],
        }
    tasks.atomic_json(obj=out, path=OUT / "stage0.json")
    print(json.dumps(out, indent=1, default=float))


BASELINE_GAIN: Final[float] = 0.10
"""A baseline that lowers the noise unit by this share or more is a candidate for part 2."""


def _share(
    *, power: dict[str, np.ndarray], s: str, neighbours: dict[str, tuple[str, ...]]
) -> np.ndarray:
    """Residual of the neighbourhood-share baseline on the tuning blocks of Primary `s`."""
    n = pan.TUNE_BLOCKS
    others = [pan.block_means(x=power[o])[:n] for o in neighbours.get(s, ()) if o in power]
    return store.neighbourhood_share_residual(own=pan.block_means(x=power[s])[:n], others=others)


def _term(*, sh: pan.BlockPanel, ctx: dec.Context, s: str) -> np.ndarray:
    """The shared term in noise units, recovered from the block panel (zeros when off)."""
    scale = ctx.cal_sh.loading[s] * ctx.cal_sh.sigma_raw[s]
    return sh.shared[s] / scale if scale != 0 else np.zeros_like(sh.shared[s])


def _weekly(*, x: np.ndarray) -> np.ndarray:
    """Block series minus the median of the same block of the week over 4 weeks either side."""
    week = 28
    out = np.full(len(x), np.nan)
    for t in range(4 * week, len(x) - 4 * week):
        ref = [x[t + k * week] for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        out[t] = x[t] - np.nanmedian(ref) if np.isfinite(ref).any() else np.nan
    return out


def _baseline_stats(*, r: np.ndarray) -> dict[str, float]:
    """Noise unit and 30-day drift (spread of the 30-day mean) of a residual series."""
    k = 120
    ok = np.isfinite(r)
    sm = np.convolve(np.where(ok, r, 0.0), np.ones(k), mode="valid") / np.maximum(
        np.convolve(ok.astype(float), np.ones(k), mode="valid"), 1
    )
    return {"noise_unit": _mad(r), "drift_30d": _mad(sm)}


# gates -------------------------------------------------------------------------------------


def gates() -> None:
    """Gates G1, G2, G4, G5, the attribute checks, and the timed world; writes `gates.json`."""
    require_review(marker="review_stage1.ok")
    import importlib

    out: dict = {}
    missing = [
        f"{m}.{a}"
        for m, names in HASATTR.items()
        for a in names
        if not hasattr(importlib.import_module(m), a)
    ]
    out["attributes_missing"] = missing
    assert not missing, missing
    ctx = dec.load_context()
    rig = ctx.rig
    out["G5"] = _g5(rig=rig)
    out["cm_units"] = _cm_units(ctx=ctx)
    out["G1"] = _g1(ctx=ctx)
    out["G2"] = _g2(ctx=ctx)
    out["G4"] = _g4(ctx=ctx)
    out["timed_world"] = _timed(ctx=ctx)
    tasks.atomic_json(obj=out, path=OUT / "gates.json")
    print(json.dumps(out, indent=1, default=float))


def _g5(*, rig: object) -> dict:
    lengths = {s: len(v) for s, v in rig.panel.power.items()}  # ty: ignore[unresolved-attribute]
    lengths |= {f"forecast_{s}": len(v) for s, v in rig.forecast.items()}  # ty: ignore[unresolved-attribute]
    assert set(lengths.values()) == {GRID_LENGTH}, lengths
    assert all(not np.isfinite(v[SCORED_FROM:]).any() for v in rig.tune_panel.power.values())  # ty: ignore[unresolved-attribute]
    assert pan.TUNE_BLOCKS * pan.BLOCK == SCORED_FROM
    return {"length": GRID_LENGTH, "tune_blocks": pan.TUNE_BLOCKS, "passed": True}


def _cm_units(*, ctx: dec.Context) -> dict:
    """The `cm` input and its noise units equal experiment 660's `real_residual` and `frozen`."""
    rig = ctx.rig
    resid = pan.common_mode_residual(rig=rig, power=rig.panel.power)
    gap = max(float(np.nanmax(np.abs(resid[s] - rig.real_residual[s]), initial=0.0)) for s in resid)
    sigma_gap = max(
        abs(ctx.cal_cm.sigma[s] - rig.frozen.sigma[s]) / rig.frozen.sigma[s]
        for s in ctx.cal_cm.sigma
    )
    assert gap < 1e-6, gap
    assert sigma_gap < 1e-9, sigma_gap
    return {"residual_max_abs_mw": gap, "sigma_max_rel": sigma_gap, "passed": True}


def _g1(*, ctx: dec.Context) -> dict:
    """Experiment 680's `decode_variants` on worlds 900 to 903 reproduces the stored rows."""
    import e00660_generator as gen  # ty: ignore[unresolved-import]
    from e00653_core import match  # ty: ignore[unresolved-import]
    from e00660_ev34_detector import truths  # ty: ignore[unresolved-import]
    from e00670_run import CONFIGS, params653  # ty: ignore[unresolved-import]
    from e00680_run import OUT as OUT680  # ty: ignore[unresolved-import]
    from e00680_run import decode_variants  # ty: ignore[unresolved-import]

    rig = ctx.rig
    stored = pl.read_parquet(OUT680 / "injected_default_900.parquet").filter(
        pl.col("method") == "E1-J-sym"
    )
    first_mismatch = None
    for seed in range(900, 904):
        world = gen.inject(
            panel=rig.panel, config=CONFIGS["default"], seed=seed, start_times=rig.starts
        )
        variants, _ = decode_variants(
            rig=rig,
            residual=rig.residual_of(world=world),
            frozen=rig.frozen,
            params=params653(),
            rng=np.random.default_rng(seed),
        )
        rows = pl.DataFrame(
            match(
                truth=truths(events=world.events, lo=SCORED_FROM, hi=GRID_LENGTH),
                events=variants["E1-J-sym"],
                world_end=GRID_LENGTH,
            )
        )
        keys = ["site", "start", "group"]
        mine = rows.sort(keys).select(*keys, "found", "right_class")
        theirs = (
            stored.filter(pl.col("world") == seed).sort(keys).select(*keys, "found", "right_class")
        )
        if not mine.equals(theirs):
            first_mismatch = {"world": seed, "rows": [mine.height, theirs.height]}
            break
    assert first_mismatch is None, first_mismatch
    return {"passed": True}


def _g2(*, ctx: dec.Context) -> dict:
    from e00653_core import freeze  # ty: ignore[unresolved-import]
    from e00670_run import params653  # ty: ignore[unresolved-import]
    from e00680_run import OUT as OUT680  # ty: ignore[unresolved-import]
    from e00680_run import decode_variants  # ty: ignore[unresolved-import]

    rig = ctx.rig
    frozen = freeze(panel=rig.p653, residual=rig.real_residual)
    variants, _ = decode_variants(
        rig=rig,
        residual=rig.real_residual,
        frozen=frozen,
        params=params653(),
        rng=np.random.default_rng(680),
    )
    mine = [
        (e.primary, int(e.start), -1 if e.end is None else int(e.end))
        for e in variants["E1-J-sym"]
        if e.start >= SCORED_FROM
    ]
    stored = pl.read_parquet(OUT680 / "real_events_E1-J-sym.parquet")
    cmp = store.event_key_mismatches(mine=mine, stored=stored)
    assert cmp["equal"], cmp
    assert cmp["n_mine"] == cmp["n_stored"] == 249, cmp
    return {"events": cmp["n_mine"], "stored": cmp["n_stored"], "passed": True}


def _g4(*, ctx: dec.Context) -> dict:
    """Layers are nonzero where they should be and zero elsewhere (2 cascade, 2 cold worlds)."""
    import e00734_layers as layers
    import polars as pl_

    rig = ctx.rig
    res: dict = {"cascade": [], "cold": []}
    for seed in bank.CASCADE_SEEDS[:2]:
        w = tasks.world_for(set_name="cascade", seed=seed, ctx=ctx)
        table = w["table"]
        eligible = len(layers.eligible_groups(events=w["world"].events))
        n_cascaded = table.filter(pl_.col("cascade_id").is_not_null())["group_id"].n_unique()
        assert abs(n_cascaded - layers.CASCADE_SHARE * eligible) <= 5, (n_cascaded, eligible)
        plain = w["world"].power
        for s, v in w["power"].items():
            changed = np.flatnonzero(np.nan_to_num(v) != np.nan_to_num(plain[s]))
            rows = table.filter((pl_.col("site") == s) & pl_.col("cascade_id").is_not_null())
            if rows.height:
                assert len(changed) > 0
                assert changed.min() >= rows["start"].min(), (s, changed.min())
            else:
                assert len(changed) == 0
        res["cascade"].append({"seed": seed, "cascaded": n_cascaded, "eligible": eligible})
    for seed in bank.COLD_SEEDS[:2]:
        w = tasks.world_for(set_name="cold", seed=seed, ctx=ctx)
        inside = tasks._spell_windows(spells=w["spells"], sites=list(rig.panel.sites))
        plain = w["world"].power
        for s, v in w["power"].items():
            moved = np.nan_to_num(v) != np.nan_to_num(plain[s])
            assert not moved[~inside[s]].any(), s
        res["cold"].append({"seed": seed, "spells": w["spells"].height})
    res["passed"] = True
    return res


def _timed(*, ctx: dec.Context) -> dict:
    """Time the Viterbi pass on one Primary first, then the passes of tuning world 734000."""
    import e00660_generator as gen  # ty: ignore[unresolved-import]

    rig = ctx.rig
    world = gen.inject(
        panel=rig.tune_panel,
        config=gen.default_config(),
        seed=bank.TIMED_SEED,
        start_times=rig.starts,
    )
    d = dec.WorldDecoder(
        ctx=ctx, power=world.power, cal_cm=ctx.cal_cm, cal_sh=ctx.cal_sh, n_blocks=pan.TUNE_BLOCKS
    )
    p = hmm.Priors(rho_on=3e-4, rho_off=1 / 240, rho_cont=0.2, kappa=5.0)
    panel = d.panel(kind="cm")
    s0 = panel.sites[0]
    t = time.time()
    hmm.viterbi(
        x=panel.observation(site=s0),
        f=panel.forecast[s0],
        sigma=panel.sigma[s0],
        scale=panel.scale[s0],
        p=p,
    )
    vit_first = time.time() - t
    print(f"Viterbi on {s0}: {vit_first:.1f} s for {pan.TUNE_BLOCKS} blocks", flush=True)
    t = time.time()
    n_sites = len(panel.sites)
    for s in panel.sites:
        hmm.forward_backward(
            x=panel.observation(site=s),
            f=panel.forecast[s],
            sigma=panel.sigma[s],
            scale=panel.scale[s],
            p=p,
        )
    fb = (time.time() - t) / n_sites
    t = time.time()
    for s in panel.sites:
        hmm.viterbi(
            x=panel.observation(site=s),
            f=panel.forecast[s],
            sigma=panel.sigma[s],
            scale=panel.scale[s],
            p=p,
        )
    vit = (time.time() - t) / n_sites
    per_block = {"fb": fb / pan.TUNE_BLOCKS, "vit": vit / pan.TUNE_BLOCKS}
    return {
        "peak_rss_gb_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "viterbi_first_primary_s": vit_first,
        "fb_s_per_primary": fb,
        "viterbi_s_per_primary": vit,
        "viterbi_kernels": sorted(hmm.KERNELS_USED),
        "tune_blocks": pan.TUNE_BLOCKS,
        "projection_cpu_hours": project(per_block=per_block, n_sites=n_sites),
    }


def project(*, per_block: dict[str, float], n_sites: int) -> dict[str, float]:
    """Projected CPU-hours of stages 2 to 4 from the measured seconds per block and pass."""
    unit = per_block["fb"] + per_block["vit"]
    full, tune = pan.N_BLOCKS, pan.TUNE_BLOCKS
    h = 3600.0
    stage2 = 31 * (12 * unit * tune + (per_block["fb"] + 3 * unit) * tune) * n_sites / h
    per_world = (unit + (per_block["fb"] + unit) * 2 + unit) * full * n_sites
    stage3 = 80 * per_world / h
    stage4 = (30 * 2 * unit + 26 * 2 * unit + 5 * unit) * full * n_sites / h
    total = stage2 + stage3 + stage4
    return {
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
        "total": total,
        "stop_at": STOP_CPU_HOURS,
    }


# choose ------------------------------------------------------------------------------------


def _read_setting(*, name: str, seeds: tuple[int, ...], with_tau_ext: bool = False) -> pl.DataFrame:
    parts = [OUT / "tune" / f"{name}_{s}.parquet" for s in seeds]
    if with_tau_ext:
        parts += [OUT / "tune" / f"{name}_tau_{s}.parquet" for s in seeds]
    return pl.concat([pl.read_parquet(path) for path in parts])


def _evaluate(
    *,
    name: str,
    rates: dict,
    years: float,
    sigma653: dict[str, float],
    uses_tau: bool,
    extra_taus: tuple[float, ...] = (),
) -> dict:
    events = pl.read_parquet(OUT / "tune" / f"{name}_real.parquet")
    rows = _read_setting(name=name, seeds=bank.TUNING_SEEDS, with_tau_ext=bool(extra_taus))
    if uses_tau:
        tau = tasks.pick_tau(
            events=events,
            rates=rates,
            years=years,
            sigma653=sigma653,
            taus=(*tasks.TAUS, *extra_taus),
        )
    else:
        big = events.filter(
            pl.col("amp_mw").abs()
            >= 2.0 * pl.col("primary").replace_strict(sigma653, return_dtype=pl.Float64)
        )
        tau = (
            1.0
            if events.height / years <= rates["all"] and big.height / years <= rates["big"]
            else None
        )
    kept = events if tau is None else events.filter(pl.col("confidence") >= tau)
    all_rate = kept.height / years
    big_kept = kept.filter(
        pl.col("amp_mw").abs()
        >= 2.0 * pl.col("primary").replace_strict(sigma653, return_dtype=pl.Float64)
    ).height
    return {
        "setting": name,
        "tau": tau,
        "eligible": tau is not None,
        "all_rate_at_099": events.filter(pl.col("confidence") >= 0.99).height / years,
        "all_rate": all_rate,
        "big_rate": big_kept / years,
        "objective": float("nan") if tau is None else tasks.objective(rows=rows, tau=tau),
    }


def _best(*, table: list[dict], key: Callable[[dict], tuple]) -> tuple[dict, bool]:
    """The eligible setting with the best objective (ties: larger rho_off, then smaller rho_on)."""
    eligible = [r for r in table if r["eligible"]]
    if eligible:
        return max(eligible, key=key), False
    return min(table, key=lambda r: r["all_rate_at_099"]), True


def _tuning_keys() -> list[str]:
    return [*map(str, bank.TUNING_SEEDS), "real"]


def _sigma653() -> dict[str, float]:
    return json.loads(
        (
            OUT.parent / "00653-edge-state-event-table-penalised-segmentation" / "noise_units.json"
        ).read_text()
    )["sigma_block_mw"]


def edge_check(
    *,
    own: list[dict],
    pelt: list[dict],
    nbr: list[dict] | None,
    chosen: dict,
    rates: dict,
    final_extension: bool = False,
    nbr_taus: tuple[float, ...] = tasks.TAUS,
) -> dict:
    """Edge check of every arm's chosen values, written to `edge_check.json` and printed.

    Every arm gets the same check: a chosen value on the lowest or highest value of its grid is a
    hit when the four-cell objective was still better there than at the next value inward by
    `edge.MIN_GAIN`, and a tau on its grid floor is a hit unless a called rate binds there.
    """
    checks: list[dict] = []
    h = chosen["h_own"]
    checks.extend(
        edge.parameter_edge(
            arm="H-own",
            name=key,
            grid=[r[key] for r in own],
            chosen=h[key],
            objective_by_value=edge.profile(table=own, key=key),
        )
        for key in tuning.RHO_KEYS
    )
    row = next(
        (r for r in own if all(r[k] == h[k] for k in tuning.RHO_KEYS)),
        {"big_rate": float("nan"), "all_rate": float("nan")},
    )
    checks.append(
        edge.tau_edge(
            arm="H-own",
            grid=tasks.TAUS,
            chosen=h["tau"],
            big_rate=row["big_rate"],
            big_limit=rates["big"],
            all_rate=row["all_rate"],
            all_limit=rates["all"],
        )
    )
    checks.append(
        edge.parameter_edge(
            arm="PELT",
            name="penalty",
            grid=[r["penalty"] for r in pelt],
            chosen=chosen["pelt"]["penalty"],
            objective_by_value=edge.profile(table=pelt, key="penalty"),
        )
    )
    if nbr is not None and "h_nbr" in chosen:
        k = chosen["h_nbr"]
        checks.append(
            edge.parameter_edge(
                arm="H-nbr",
                name="kappa",
                grid=[r["kappa"] for r in nbr],
                chosen=k["kappa"],
                objective_by_value=edge.profile(table=nbr, key="kappa"),
            )
        )
        nrow = next(r for r in nbr if r["kappa"] == k["kappa"])
        checks.append(
            edge.tau_edge(
                arm="H-nbr",
                grid=nbr_taus,
                chosen=k["tau"],
                big_rate=nrow["big_rate"],
                big_limit=rates["big"],
                all_rate=nrow["all_rate"],
                all_limit=rates["all"],
            )
        )
    for c in checks:
        c["final_extension"] = bool(final_extension and c["arm"] == "H-own")
    hits = [f"{c['arm']} {c['parameter']}" for c in checks if c["hit"]]
    state = tuning.read_state(out=OUT)
    out = {
        "min_gain": edge.MIN_GAIN,
        "extensions": state,
        "equal_effort": {
            f"{c['arm']} {c['parameter']}": c["classification"]
            for c in checks
            if c["arm"] in ("PELT", "H-nbr")
        },
        "checks": checks,
        "hits": hits,
        "any_hit": bool(hits),
        "grid_h_own_rho_on": sorted({r["rho_on"] for r in own}),
    }
    tasks.atomic_json(obj=out, path=OUT / "edge_check.json")
    print("\n".join(edge.summary_lines(checks=checks)))
    print(f"edge hits: {hits or 'none'}")
    return out


def _own_table(
    *, rates: dict, years: float, sigma653: dict[str, float], extended: bool
) -> list[dict]:
    settings = tuning.ALL_SETTINGS if extended else tuning.SETTINGS
    return [
        _evaluate(name=f"own{i:02d}", rates=rates, years=years, sigma653=sigma653, uses_tau=True)
        | dict(zip(tuning.RHO_KEYS, s, strict=True))
        for i, s in enumerate(settings)
    ]


def _pelt_grid(*, keys: list[str]) -> tuple[float, ...]:
    """The PELT penalty grid, with the extension's values once their files all exist."""
    grid = tuple(__import__("e00734_pelt").PENALTY_GRID)
    extra = tuning.read_state(out=OUT)["pelt"]
    if extra is None:
        return grid
    names = [f"pelt{len(grid) + i:02d}" for i in range(len(extra["values"]))]
    if tuning.extension_complete(out=OUT, keys=keys, names=names):
        return grid + tuple(extra["values"])
    return grid


def _pelt_table(
    *, rates: dict, years: float, sigma653: dict[str, float], grid: tuple[float, ...]
) -> list[dict]:
    return [
        _evaluate(name=f"pelt{i:02d}", rates=rates, years=years, sigma653=sigma653, uses_tau=False)
        | {"penalty": p}
        for i, p in enumerate(grid)
    ]


def _choose_own(*, rates: dict, years: float, sigma653: dict[str, float]) -> None:
    """Choose H-own and PELT from the largest complete grid, backing up what it overwrites.

    The grid has a level: `first` (12 settings), `extended` (20), or `extended2` (34). A choice
    from a larger grid replaces one from a smaller grid, after a copy of the old files to `*_v1`
    (from the first grid) or `*_v2` (from the first extension). A choice is never replaced by one
    from a grid of the same level.
    """
    target = OUT / "choice_own.json"
    keys = _tuning_keys()
    level = tuning.grid_level(out=OUT, keys=keys)
    if target.exists():
        old = json.loads(target.read_text()).get("grid", "first")
        if tuning.GRID_LEVELS.index(old) >= level:
            sys.exit(f"refusing to overwrite {target}")
        copied = tuning.backup_v1(out=OUT) if old == "first" else tuning.backup_v2(out=OUT)
        print("backed up:", copied)
    parts = [tuning.SETTINGS, tuning.SETTINGS_EXT, tuning.SETTINGS_EXT2][: level + 1]
    settings = tuple(s for part in parts for s in part)
    own = [
        _evaluate(name=f"own{i:02d}", rates=rates, years=years, sigma653=sigma653, uses_tau=True)
        | dict(zip(tuning.RHO_KEYS, s, strict=True))
        for i, s in enumerate(settings)
    ]
    pelt = _pelt_table(rates=rates, years=years, sigma653=sigma653, grid=_pelt_grid(keys=keys))
    best, fb_own = _best(table=own, key=lambda r: (r["objective"], r["rho_off"], -r["rho_on"]))
    best_p, fb_pelt = _best(table=pelt, key=lambda r: (r["objective"], r["penalty"]))
    if fb_pelt:
        best_p = max(pelt, key=lambda r: r["penalty"])
    pl.DataFrame(own + pelt, strict=False).write_parquet(OUT / "tuning_own.parquet")
    chosen = {
        "h_own": {k: best[k] for k in (*tuning.RHO_KEYS, "tau")}
        | {"fallback": fb_own, "tau": best["tau"] if best["tau"] is not None else 0.99},
        "pelt": {"penalty": best_p["penalty"], "fallback": fb_pelt},
        "grid": tuning.GRID_LEVELS[level],
    }
    tasks.atomic_json(obj=chosen, path=target)
    print(json.loads(target.read_text()))
    edge_check(own=own, pelt=pelt, nbr=None, chosen=chosen, rates=rates, final_extension=level == 2)


def choose(*, phase: str) -> None:
    """Choose the tuned values from the tuning parquet files and refuse to overwrite a choice.

    `own` chooses H-own and PELT. After the extension of the H-own grid it chooses once more from
    the extended grid, after copying the first outputs to `*_v1`. `final` chooses kappa and tau of
    H-nbr, writes `params.json`, and runs gate G3b. `g3b` only re-runs gate G3b on `params.json`.
    Every choice ends with the edge check of all the arms chosen so far.
    """
    if phase == "g3b":
        g3b(tuned=json.loads(PARAMS.read_text()))
        return
    sigma653 = _sigma653()
    rates = json.loads((OUT / "tune" / "e1jsym_rates.json").read_text())
    years = rates["primary_years"]
    if phase == "own":
        _choose_own(rates=rates, years=years, sigma653=sigma653)
        return
    if PARAMS.exists():
        stale = [
            OUT / tuning.versioned_name(name="params.json", version=v)
            for v in ("v2", "v1")
            if (OUT / tuning.versioned_name(name="params.json", version=v)).exists()
        ]
        if not any(sha256_of(path=b) == sha256_of(path=PARAMS) for b in stale):
            sys.exit(f"refusing to overwrite {PARAMS}")
    chosen = json.loads((OUT / "choice_own.json").read_text())
    tag = tuning.nbr_tag(out=OUT)
    kappas = list(tasks.KAPPAS)
    extra = tuning.read_state(out=OUT)["kappa"]
    if extra is not None and all(
        (OUT / "tune" / f"{tag}_k{k:g}_{key}.parquet").exists()
        for k in extra["values"]
        for key in _tuning_keys()
    ):
        kappas += list(extra["values"])
    tau_state = tuning.read_state(out=OUT)["tau"]
    extra_taus: tuple[float, ...] = ()
    if tau_state is not None and all(
        (OUT / "tune" / f"{tag}_k{k:g}_tau_{key}.parquet").exists()
        for k in kappas
        for key in map(str, bank.TUNING_SEEDS)
    ):
        extra_taus = tuple(tau_state["values"])
    nbr_taus = tuple(sorted((*tasks.TAUS, *extra_taus)))
    nbr = [
        _evaluate(
            name=f"{tag}_k{k:g}",
            rates=rates,
            years=years,
            sigma653=sigma653,
            uses_tau=True,
            extra_taus=extra_taus,
        )
        | {"kappa": k}
        for k in kappas
    ]
    best_n, fb_nbr = _best(table=nbr, key=lambda r: (r["objective"], -r["kappa"]))
    tuned = {
        **chosen,
        "h_nbr": {
            "kappa": float(best_n["kappa"]),
            "tau": best_n["tau"] if best_n["tau"] is not None else 0.99,
            "fallback": fb_nbr,
        },
        "nbr_outputs": tag,
    }
    own_table = pl.read_parquet(OUT / "tuning_own.parquet")
    pl.concat([own_table, pl.DataFrame(nbr, strict=False)], how="diagonal").write_parquet(
        OUT / "tuning.parquet"
    )
    tasks.atomic_json(obj=tuned, path=PARAMS)
    PARAMS_SHA.write_text(f"{sha256_of(path=PARAMS)}  params.json\n")
    print(json.dumps(tuned, indent=1))
    own_rows = own_table.filter(pl.col("setting").str.starts_with("own")).to_dicts()
    pelt_rows = own_table.filter(pl.col("setting").str.starts_with("pelt")).to_dicts()
    edge_check(
        own=own_rows,
        pelt=pelt_rows,
        nbr=nbr,
        chosen=tuned,
        rates=rates,
        final_extension=chosen.get("grid") == "extended2",
        nbr_taus=nbr_taus,
    )
    g3b(tuned=tuned)


def g3b(*, tuned: dict) -> None:
    """Gate G3b at both taus, and the ramp diagnostic (printed, never a gate)."""
    tuning.backup_v1(out=OUT, names=("g3b.json",))
    by_arm = tuning.g3b_rates(
        arms={
            arm: (
                dec.arm_params(tuned=tuned, arm=arm).priors,
                dec.arm_params(tuned=tuned, arm=arm).tau,
            )
            for arm in ("H-own", "H-nbr")
        }
    )
    ramp = ev.ramp_diagnostic(
        priors={
            "H-nbr": dec.arm_params(tuned=tuned, arm="H-nbr").priors,
            "H-nbr-noramp": dec.arm_params(tuned=tuned, arm="H-nbr-noramp").priors,
        },
        tau=dec.arm_params(tuned=tuned, arm="H-nbr").tau,
    )
    out = {
        "G3b_called_per_primary_year": by_arm["H-own"]["called_per_primary_year"],
        "G3b_passed": all(v["passed"] for v in by_arm.values()),
        "by_arm": by_arm,
        "ramp_diagnostic": ramp,
    }
    tasks.atomic_json(obj=out, path=OUT / "g3b.json")
    print(json.dumps(out, indent=1, default=float))


# real record -------------------------------------------------------------------------------


def real() -> None:
    """Decode the whole real record with every arm and write tables, rates, and counts."""
    import e00670_run as r670  # ty: ignore[unresolved-import]
    from e00653_common import DAY  # ty: ignore[unresolved-import]

    tuned = load_params()
    ctx = dec.load_context(sigma_from="whole")
    rig = ctx.rig
    d = dec.WorldDecoder(ctx=ctx, power=rig.panel.power, cal_cm=ctx.cal_cm, cal_sh=ctx.cal_sh)
    years_scored = tasks.primary_years(power=rig.panel.power, lo=SCORED_FROM, hi=GRID_LENGTH)
    busy = dec.event_free_masks(rig=rig)
    free_years = sum(int((~busy[s] & np.isfinite(rig.panel.power[s])).sum()) for s in busy) / (
        DAY * 365.25
    )
    logged = r670.logged_shifts(rig=rig, resid=rig.real_residual)
    summary: dict = {
        "scored_primary_years": years_scored,
        "event_free_primary_years": free_years,
        "logged_total": logged.height,
    }
    stored = pl.read_parquet(
        tasks.ML_EXPERIMENTS_DATA_DIR
        / "00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search"
        / "real_events_E1-J-sym.parquet"
    )
    tables: dict[str, pl.DataFrame] = {}
    for arm in dec.ARMS:
        ap = dec.arm_params(tuned=tuned, arm=arm)
        res = d.run_arm(arm=arm, params=ap)
        called = d.final_events(result=res, tau=ap.tau)
        table = dec.events_frame(events=called).with_columns(primary_=pl.col("primary"))
        tables[arm] = table
        tasks.atomic_parquet(frame=table, path=OUT / "real" / f"events_{arm}.parquet")
        if arm != "PELT":
            tasks.atomic_parquet(
                frame=_block_table(res=res), path=OUT / "real" / f"blocks_{arm}.parquet"
            )
        summary[arm] = _real_summary(
            called=called,
            ctx=ctx,
            busy=busy,
            years_scored=years_scored,
            free_years=free_years,
            logged=logged,
            r670=r670,
        )
    summary["E1-J-sym"] = _stored_summary(
        stored=stored, ctx=ctx, busy=busy, years_scored=years_scored, free_years=free_years
    )
    for arm in dec.ARMS:
        summary[arm]["ratio_scored"] = _ratios(
            summary[arm]["rates_scored"], summary["E1-J-sym"]["rates_scored"]
        )
        summary[arm]["ratio_event_free"] = _ratios(
            summary[arm]["rates_event_free"], summary["E1-J-sym"]["rates_event_free"]
        )
    tasks.atomic_json(obj=summary, path=OUT / "real" / "summary.json")
    print(json.dumps(summary, indent=1, default=float))


def _ratios(a: list[float], b: list[float]) -> list[float]:
    return [x / y if y else float("nan") for x, y in zip(a, b, strict=True)]


def _block_table(*, res: dec.ArmResult) -> pl.DataFrame:
    frames = []
    for s, post in res.posts.items():
        q = ev.block_quantiles(post=post, sigma=res.blocks.sigma[s])
        frames.append(
            pl.DataFrame(
                {
                    "primary": [s] * len(q),
                    "block": np.arange(len(q)),
                    **{f"q{int(v * 100):02d}": q[:, j] for j, v in enumerate(ev.QUANTILES)},
                    "p_zero": post.o_marg[:, hmm.FULL_GRID.zero],
                    "unknown": res.blocks.unknown[s],
                }
            )
        )
    return pl.concat(frames)


def _rates(
    *,
    table: pl.DataFrame,
    sigma653: dict[str, float],
    years: float,
    mask: dict[str, np.ndarray] | None,
) -> list[float]:
    """Called events per Primary-year, all and of 2 or more noise units, from the scored years."""
    t = table.filter(pl.col("start") >= SCORED_FROM)
    if mask is not None:
        keep = []
        for r in t.iter_rows(named=True):
            end = GRID_LENGTH if r["end"] < 0 else r["end"]
            keep.append(not mask[r["primary"]][r["start"] : max(end, r["start"] + 1)].any())
        t = t.filter(pl.Series(keep, dtype=pl.Boolean)) if keep else t
    big = t.filter(
        pl.col("amp_mw").abs()
        >= 2.0 * pl.col("primary").replace_strict(sigma653, return_dtype=pl.Float64)
    )
    return [t.height / years, big.height / years]


def _real_summary(
    *,
    called: list,
    ctx: dec.Context,
    busy: dict,
    years_scored: float,
    free_years: float,
    logged: pl.DataFrame,
    r670: ModuleType,
) -> dict:
    table = dec.events_frame(events=called)
    scored = table.filter(pl.col("start") >= SCORED_FROM)
    return {
        "events_scored_years": scored.height,
        "rates_scored": _rates(table=table, sigma653=ctx.sigma653, years=years_scored, mask=None),
        "rates_event_free": _rates(table=table, sigma653=ctx.sigma653, years=free_years, mask=busy),
        "grades": {g: scored.filter(pl.col("grade") == g).height for g in ev.GRADES},
        "logged_matched": r670.matched(table=table.rename({"primary": "primary"}), logged=logged),
        "d5_d16": r670.verified_d5_d16(events=called),
        "d8_events": scored.filter(pl.col("primary") == "D8").height,
        "christmas": _christmas(table=scored),
        "refit_boundary": _refit(table=scored),
    }


def _stored_summary(
    *, stored: pl.DataFrame, ctx: dec.Context, busy: dict, years_scored: float, free_years: float
) -> dict:
    table = stored.rename({"primary": "primary"}).select("primary", "start", "end", "amp_mw")
    return {
        "rates_scored": _rates(table=table, sigma653=ctx.sigma653, years=years_scored, mask=None),
        "rates_event_free": _rates(table=table, sigma653=ctx.sigma653, years=free_years, mask=busy),
    }


def _christmas(*, table: pl.DataFrame) -> int:
    """Called events that start from 20 December to 6 January."""
    from datetime import UTC, datetime, timedelta

    n = 0
    for s in table["start"].to_list():
        t = datetime(2022, 1, 1, tzinfo=UTC) + timedelta(minutes=30 * int(s))
        n += (t.month == 12 and t.day >= 20) or (t.month == 1 and t.day <= 6)
    return n


def _refit(*, table: pl.DataFrame) -> int:
    """Called events that start within one block of the first of a month (a monthly refit)."""
    from datetime import UTC, datetime, timedelta

    n = 0
    for s in table["start"].to_list():
        t = datetime(2022, 1, 1, tzinfo=UTC) + timedelta(minutes=30 * int(s))
        first = t.replace(day=1, hour=0, minute=0)
        n += (
            abs((t - first).total_seconds()) <= 6 * 3600
            or abs((t - (first + timedelta(days=32)).replace(day=1)).total_seconds()) <= 6 * 3600
        )
    return n


# main --------------------------------------------------------------------------------------


def _register_tau_extension() -> str:
    """Check and record the extension of H-nbr's tau grid below its floor; return the nbr tag.

    The called rates at a lower tau come from the stored events of the real record, so the check
    needs no decoding. The truth rows of the tuning worlds at the new taus need H-nbr decoded once
    more (about 2.3 CPU-hours for 3 kappa values over the 30 worlds).

    Raises:
        ValueError: If the choice is not the floor, or a call rate binds so that no lower tau is
            eligible, or a different extension was registered before.
    """
    params = json.loads(PARAMS.read_text())
    kappa, tau = params["h_nbr"]["kappa"], params["h_nbr"]["tau"]
    tag = tuning.nbr_tag(out=OUT)
    rates = json.loads((OUT / "tune" / "e1jsym_rates.json").read_text())
    sigma653 = _sigma653()
    events = pl.read_parquet(OUT / "tune" / f"{tag}_k{kappa:g}_real.parquet")

    def within(t: float) -> bool:
        kept = events.filter(pl.col("confidence") >= t)
        big = kept.filter(
            pl.col("amp_mw").abs()
            >= 2.0 * pl.col("primary").replace_strict(sigma653, return_dtype=pl.Float64)
        )
        return (
            kept.height / rates["primary_years"] <= rates["all"]
            and big.height / rates["primary_years"] <= rates["big"]
        )

    values = tuning.lower_taus(grid=tasks.TAUS, chosen=tau, within_limits=within)
    tuning.register_other_extension(out=OUT, arm="tau", side="low", values=values)
    print(f"tau extension (low): {values}; about 2.3 CPU-hours for the H-nbr re-decode")
    return tag


def tune_plan(
    *, phase: str, extend_pelt: bool, extend_kappa: bool, extend_tau: bool = False
) -> tuple[str, str]:
    """The phase to run and the name of its done markers, after the extension rules.

    The H-own grid is extended twice and never again (`own_ext`, `own_ext2`). The PELT and kappa
    grids are extended only by an explicit flag, once each, with at most 3 new values on the edge
    where the choice sits. Any refusal stops the run before a task starts.
    """
    try:
        if extend_pelt:
            grid = tuple(__import__("e00734_pelt").PENALTY_GRID)
            penalty = json.loads((OUT / "choice_own.json").read_text())["pelt"]["penalty"]
            side, values = tuning.extension_values(grid=grid, chosen=penalty)
            tuning.register_other_extension(out=OUT, arm="pelt", side=side, values=values)
            hours = tuning.projected_cpu_hours(n_settings=len(values))
            print(f"PELT extension ({side}): {values}; about {hours:.1f} CPU-hours")
            return "pelt_ext", "pelt_ext"
        if extend_kappa:
            if not PARAMS.exists():
                sys.exit("--extend-kappa needs params.json (run choose --phase final first)")
            kappa = float(json.loads(PARAMS.read_text())["h_nbr"]["kappa"])
            side, values = tuning.extension_values(
                grid=tuple(float(k) for k in tasks.KAPPAS), chosen=kappa, floor=0.0
            )
            tuning.register_other_extension(out=OUT, arm="kappa", side=side, values=values)
            hours = tuning.projected_cpu_hours(n_settings=len(values))
            print(f"kappa extension ({side}): {values}; about {hours:.1f} CPU-hours")
            return "nbr_ext", f"{tuning.nbr_tag(out=OUT)}_ext"
        if extend_tau:
            return "nbr_tau", f"{_register_tau_extension()}_tau"
        if phase in ("own_ext", "own_ext2"):
            tuning.register_h_own_extension(out=OUT, number=1 if phase == "own_ext" else 2)
            return phase, phase
    except ValueError as exc:
        sys.exit(f"refusing to extend: {exc}")
    return phase, tuning.nbr_tag(out=OUT) if phase == "nbr" else phase


def main() -> None:  # noqa: PLR0912, PLR0915 - one branch per stage
    """Parse the command line and run one stage."""
    pin_blas()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage",
        choices=(
            "stage0",
            "gallery",
            "gates",
            "tune",
            "choose",
            "score",
            "real",
            "report",
            "figures",
            "freeze-code",
        ),
    )
    ap.add_argument("--which", default="scored", choices=("scored", "real"))
    ap.add_argument("--phase", default="own")
    ap.add_argument(
        "--extend-pelt",
        action="store_true",
        help="tune: add up to 3 penalties beyond the PELT grid edge where the choice sits (once)",
    )
    ap.add_argument(
        "--extend-tau",
        action="store_true",
        help="tune: add up to 3 tau values below H-nbr's floor of 0.5 where rates allow (once)",
    )
    ap.add_argument(
        "--extend-kappa",
        action="store_true",
        help="tune: add up to 3 values beyond the kappa grid edge where the choice sits (once)",
    )
    ap.add_argument("--set", default="default", choices=SETS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "stage0":
        stage0()
    elif args.stage == "gallery":
        import e00734_figures as figs

        figs.gallery(out=OUT)
    elif args.stage == "gates":
        run_guarded(name="gates", fn=gates)
    elif args.stage == "tune":
        require_review(marker="review_stage1.ok")
        keys = [
            *map(str, bank.guard_tuning(seeds=bank.TUNING_SEEDS, extra=bank.solar_seeds())),
            "real",
        ]
        phase, marker = tune_plan(
            phase=args.phase,
            extend_pelt=args.extend_pelt,
            extend_kappa=args.extend_kappa,
            extend_tau=args.extend_tau,
        )
        if phase == "own":
            keys.append("e1jsym")
        run_shard(
            names_and_kwargs=[
                (f"tune_{marker}_{key}", {"phase": phase, "key": key})
                for key in shard(items=keys, worker=args.worker, of=args.of)
            ],
            fn=tasks.tune_task,
        )
    elif args.stage == "freeze-code":
        CODE_SHA.write_text(f"{code_hash()}  scoring code\n")
        print(CODE_SHA.read_text(), end="")
    elif args.stage == "report":
        require_review(marker="review_stage3.ok")
        require_code_frozen()
        import e00734_score as score_

        score_.report(out=OUT)
    elif args.stage == "figures":
        require_review(marker="review_stage3.ok")
        require_code_frozen()
        import e00734_figures as figs

        (figs.scored_figures if args.which == "scored" else figs.real_figures)(out=OUT)
    elif args.stage == "choose":
        require_review(marker="review_stage1.ok")
        choose(phase=args.phase)
    elif args.stage == "score":
        tuned = load_params()
        if args.set == "uninjected":
            run_shard(
                names_and_kwargs=[
                    (f"uninjected_{arm}", {"arm": arm, "tuned": tuned})
                    for arm in shard(items=list(dec.H_ARMS), worker=args.worker, of=args.of)
                ],
                fn=tasks.uninjected_task,
            )
        elif args.set in ("solar", "bias"):
            kind, n = ("pv_growth", 24) if args.set == "solar" else ("seasonal_bias", 2)
            run_shard(
                names_and_kwargs=[
                    (f"{args.set}_{k}", {"kind": kind, "k": k, "tuned": tuned})
                    for k in shard(items=list(range(n)), worker=args.worker, of=args.of)
                ],
                fn=tasks.solar_bias_task,
            )
        else:
            run_shard(
                names_and_kwargs=[
                    (f"{args.set}_{seed}", {"set_name": args.set, "seed": seed, "tuned": tuned})
                    for seed in shard(items=list(SEEDS[args.set]), worker=args.worker, of=args.of)
                ],
                fn=tasks.score_task,
            )
    else:
        run_guarded(name="real", fn=real)


if __name__ == "__main__":
    main()

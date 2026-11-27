"""Tuning and scoring tasks of experiment 734, one world per task.

Every task writes its own files under `OUT`, atomically, so a crashed worker is re-run without
touching the others and a parallel run gives the numbers a serial run would.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import e00734_bank as bank
import e00734_decode as dec
import e00734_events as ev
import e00734_hmm as hmm
import e00734_layers as layers
import e00734_panel as pan
import e00734_score as score
import e00734_store as store
import e00734_tuning as tuning
import numpy as np
import polars as pl
from e00653_common import GRID_LENGTH, SCORED_FROM  # ty: ignore[unresolved-import]
from e00653_core import decode_world, match  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

if TYPE_CHECKING:
    from e00660_generator import InjectedWorld  # ty: ignore[unresolved-import]

OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00734-switching-offset-model-part-1"
TAUS: Final[tuple[float, ...]] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SETTINGS: Final = tuning.SETTINGS
"""The 12 (rho_on, rho_off, rho_cont) settings of H-own in the first tuning."""
KAPPAS: Final[tuple[int, ...]] = (0, 5, 20)
SEED_OFFSET: Final[int] = 7_340_000
"""Added to a world's seed to seed its layer, so the layer's draws differ from the world's."""
SENSITIVITY: Final[tuple[str, ...]] = ("long_heavy", "short_heavy", "constant_mw_square")
DAY: Final[int] = 48
POSITIVE: Final[tuple[str, ...]] = ("edge_transfer", "shift")


def atomic_parquet(*, frame: pl.DataFrame, path: Path) -> None:
    """Write a parquet file under a temporary name and rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.rename(path)


def atomic_json(*, obj: object, path: Path) -> None:
    """Write a JSON file under a temporary name and rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)


def time_log(*, stage: str, task: str, start: float) -> None:
    """Append one line of the per-phase time log (device, start, end)."""
    stamp = "%FT%T"
    begin, end = (time.strftime(stamp, time.gmtime(v)) for v in (start, time.time()))
    line = f"{stage},{task},cpu,{begin},{end}\n"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "time_log.csv").open("a") as f:
        f.write(line)


def tune_settings(*, phase: str) -> list[tuple[str, str, dec.ArmParams]]:
    """The `(setting name, arm, params)` triples of a tuning phase.

    Phase `own` holds the 12 settings of H-own and the 12 PELT penalties. Phases `own_ext` and
    `own_ext2` hold the 8 and the 14 settings that extend the H-own grid. Phases `pelt_ext` and
    `nbr_ext` hold the values that an explicit flag added to the PELT penalty grid and the kappa
    grid. Phase `nbr_tau` re-decodes H-nbr to write the truth rows at the lower tau values that
    `--extend-tau` added (files `<name>_tau_<seed>.parquet`). Phase `nbr` holds the three values
    of kappa, with the rho values that `choose --phase own` stored, under the name that
    `tuning.nbr_tag` gives (`nbr` while the first outputs stay valid, `nbr2` after they do not).
    """
    if phase in ("own_ext", "own_ext2"):
        names, settings = (
            (tuning.ext_names(), tuning.SETTINGS_EXT)
            if phase == "own_ext"
            else (tuning.ext2_names(), tuning.SETTINGS_EXT2)
        )
        return [
            (name, "H-own", dec.ArmParams(priors=hmm.Priors(*s), tau=1.0))
            for name, s in zip(names, settings, strict=True)
        ]
    if phase == "pelt_ext":
        from e00734_pelt import PENALTY_GRID

        extra = tuning.read_state(out=OUT)["pelt"]["values"]
        return [
            (
                f"pelt{len(PENALTY_GRID) + i:02d}",
                "PELT",
                dec.ArmParams(priors=hmm.Priors(*SETTINGS[0]), tau=1.0, penalty=p),
            )
            for i, p in enumerate(extra)
        ]
    if phase == "own":
        own = [
            (f"own{i:02d}", "H-own", dec.ArmParams(priors=hmm.Priors(*s), tau=1.0))
            for i, s in enumerate(SETTINGS)
        ]
        from e00734_pelt import PENALTY_GRID

        pelt = [
            (
                f"pelt{i:02d}",
                "PELT",
                dec.ArmParams(priors=hmm.Priors(*SETTINGS[0]), tau=1.0, penalty=p),
            )
            for i, p in enumerate(PENALTY_GRID)
        ]
        return own + pelt
    chosen = json.loads((OUT / "choice_own.json").read_text())["h_own"]
    base = {k: chosen[k] for k in tuning.RHO_KEYS}
    tag = tuning.nbr_tag(out=OUT)
    kappas = tuple(tuning.read_state(out=OUT)["kappa"]["values"]) if phase == "nbr_ext" else KAPPAS
    if phase == "nbr_tau":
        extra_k = tuning.read_state(out=OUT)["kappa"]
        kappas = KAPPAS + (tuple(extra_k["values"]) if extra_k else ())
    return [
        (
            f"{tag}_k{k:g}",
            "H-nbr",
            dec.ArmParams(priors=hmm.Priors(**base, kappa=float(k)), tau=1.0),
        )
        for k in kappas
    ]


def tune_task(*, phase: str, key: str) -> None:
    """Decode one tuning world (or the real record before 2024, or experiment 680's table).

    A tuning world is the default generator on the tuning panel (power hidden from 2024-01-01),
    cut at 2024-01-01 before any pass runs. For each setting the task writes the truth rows at
    every threshold, or the table of events for the real record.
    """
    import e00660_generator as gen  # ty: ignore[unresolved-import]
    from e00660_ev34_detector import truths  # ty: ignore[unresolved-import]

    start = time.time()
    ctx = dec.load_context()
    rig = ctx.rig
    if key == "e1jsym":
        _e1jsym_rates(ctx=ctx)
        return
    if key == "real":
        power = rig.tune_panel.power
        world = None
    else:
        seed = bank.guard_tuning(seeds=[int(key)], extra=bank.solar_seeds())[0]
        world = gen.inject(
            panel=rig.tune_panel, config=gen.default_config(), seed=seed, start_times=rig.starts
        )
        power = world.power
    if phase == "nbr_tau" and world is None:
        return  # the real record's events hold every confidence, so its rates need no new decode
    assert all(not np.isfinite(v[SCORED_FROM:]).any() for v in power.values())
    d = dec.WorldDecoder(
        ctx=ctx, power=power, cal_cm=ctx.cal_cm, cal_sh=ctx.cal_sh, n_blocks=pan.TUNE_BLOCKS
    )
    assert d.n_blocks * dec.BLOCK == SCORED_FROM
    for name, arm, ap in tune_settings(phase=phase):
        res = d.run_arm(arm=arm, params=ap)
        if world is None:
            atomic_parquet(
                frame=dec.events_frame(events=res.events),
                path=OUT / "tune" / f"{name}_real.parquet",
            )
            continue
        truth = truths(events=world.events, lo=0, hi=SCORED_FROM)
        rows = []
        new_taus = tuning.read_state(out=OUT)["tau"] if phase == "nbr_tau" else None
        grid = tuple(new_taus["values"]) if new_taus else TAUS
        for tau in grid if arm != "PELT" else (1.0,):
            matched = ev.match_called(
                truth=truth, events=res.events, tau=tau, world_end=SCORED_FROM
            )
            rows.extend({**r, "tau": tau, "world": int(key)} for r in matched)
        suffix = "_tau" if phase == "nbr_tau" else ""
        atomic_parquet(
            frame=pl.DataFrame(rows), path=OUT / "tune" / f"{name}{suffix}_{key}.parquet"
        )
    time_log(stage=f"tune-{phase}", task=key, start=start)


def _e1jsym_rates(*, ctx: dec.Context) -> None:
    """E1-J-sym's two called rates on the real record of 2022 and 2023 (experiment 680's code)."""
    from e00670_run import params653  # ty: ignore[unresolved-import]
    from e00680_run import decode_variants  # ty: ignore[unresolved-import]

    rig = ctx.rig
    variants, _ = decode_variants(
        rig=rig,
        residual=rig.real_tune,
        frozen=rig.frozen,
        params=params653(),
        rng=np.random.default_rng(680),
    )
    years = primary_years(power=rig.tune_panel.power, lo=0, hi=SCORED_FROM)
    events = [e for e in variants["E1-J-sym"] if e.start < SCORED_FROM]
    rates = ev.called_rates(
        events=events, sigma653=ctx.sigma653, primary_years=years, lo=0, hi=SCORED_FROM
    )
    atomic_json(
        obj={"primary_years": years, "all": rates[0], "big": rates[1], "n_events": len(events)},
        path=OUT / "tune" / "e1jsym_rates.json",
    )


def primary_years(*, power: dict[str, np.ndarray], lo: int, hi: int) -> float:
    """Primary-years with a reading in `[lo, hi)`."""
    return sum(int(np.isfinite(v[lo:hi]).sum()) for v in power.values()) / (DAY * 365.25)


def pick_tau(
    *,
    events: pl.DataFrame,
    rates: dict,
    years: float,
    sigma653: dict[str, float],
    taus: tuple[float, ...] = TAUS,
) -> float | None:
    """Lowest threshold at which both called rates are at or below E1-J-sym's, else None."""
    for tau in sorted(taus):
        kept = events.filter(pl.col("confidence") >= tau)
        big = kept.filter(
            pl.col("amp_mw").abs()
            >= 2.0 * pl.col("primary").replace_strict(sigma653, return_dtype=pl.Float64)
        )
        if kept.height / years <= rates["all"] and big.height / years <= rates["big"]:
            return tau
    return None


def objective(*, rows: pl.DataFrame, tau: float) -> float:
    """MW-weighted found share over the four long and medium cells, at one threshold."""
    cells = score.add_cells(rows=rows.filter(pl.col("tau") == tau))
    four = cells.filter(pl.col("cell").str.contains("^(long|medium)"))
    return score.found_share(rows=four)


def world_for(*, set_name: str, seed: int, ctx: dec.Context) -> dict:
    """Build the world of a scoring set: power, truth, hard-negative windows, and overrides."""
    import e00660_generator as gen  # ty: ignore[unresolved-import]
    from e00660_ev34_detector import truths  # ty: ignore[unresolved-import]
    from e00670_run import CONFIGS  # ty: ignore[unresolved-import]

    rig = ctx.rig
    config = CONFIGS["default" if set_name in ("default", "cascade", "cold") else set_name]
    world = gen.inject(panel=rig.panel, config=config, seed=seed, start_times=rig.starts)
    table, power, spells = world.events, world.power, None
    if set_name == "cascade":
        table = layers.cascade_events(
            events=world.events, rng=np.random.default_rng(seed + SEED_OFFSET)
        )
        power = gen.apply_events(panel=rig.panel, events=table.drop("cascade_id", "cascade_end"))
        truth = layers.cascade_truths(events=table, lo=SCORED_FROM, hi=GRID_LENGTH)
    else:
        if set_name == "cold":
            add, spells = layers.cold_spells(rig=rig, rng=np.random.default_rng(seed + SEED_OFFSET))
            power = layers.apply_additions(
                power=world.power, additions=add, is_mva=rig.panel.is_mva
            )
        truth = truths(events=world.events, lo=SCORED_FROM, hi=GRID_LENGTH)
    hard = world.events.filter(pl.col("label") == "hard_negative")
    return {
        "world": world,
        "table": table,
        "power": power,
        "truth": truth,
        "hard": hard,
        "spells": spells,
        "config": config,
    }


def _stored_tables(*, ctx: dec.Context, resid: dict) -> dict[str, list]:
    """E1 and E1-J-sym events from experiment 653's and 680's code on a residual."""
    from e00670_run import params653  # ty: ignore[unresolved-import]
    from e00680_search import search_both_ends  # ty: ignore[unresolved-import]

    rig = ctx.rig
    events, slows = decode_world(
        panel=rig.p653, residual=resid, frozen=rig.frozen, arm="masked", params=params653()
    )
    xh = {s: resid[s] - store.blocks_to_half_hours(level=slows[s]) for s in slows}
    both = search_both_ends(
        events=copy.deepcopy(events), xh=xh, sigma=rig.frozen.sigma, neighbours=rig.p653.neighbours
    )
    return {"E1": events, "E1-J-sym": both}


def hard_overlap(*, events: list, hard: pl.DataFrame, length: int, windows: dict) -> list[bool]:
    """Whether each event overlaps a hard-negative window of its Primary by half its length."""
    out = []
    for e in events:
        end = length if e.end is None else e.end
        w = windows.get(e.primary)
        out.append(
            w is not None and score.event_overlap_share(start=e.start, end=end, windows=w) >= 0.5
        )
    return out


def matched_flags(*, events: list, truth: list, world_end: int) -> list[bool]:
    """Whether each called event matches some truth event under experiment 653's rule."""
    out = []
    for e in events:
        hit = False
        for t in truth:
            t_end = world_end if t.end is None else t.end
            e_end = world_end if e.end is None else e.end
            lo, hi = max(t.start, e.start), min(t_end, e_end)
            inter = max(0, hi - lo)
            union = (t_end - t.start) + (e_end - e.start) - inter
            if (
                t.site == e.primary
                and np.sign(t.amp_mw) == np.sign(e.amp_mw)
                and abs(t.start - e.start) <= 24
                and union > 0
                and inter / union >= 0.5
            ):
                hit = True
                break
        out.append(hit)
    return out


def uninjected_task(*, arm: str, tuned: dict) -> None:
    """Decode the uninjected real record once with one H arm and cache its summaries.

    Every scored world compares its posterior with this one (the calibration of the predicted
    shift), and the solar-growth and bias worlds read its called events, so the decode runs once
    and not once per world. The record uses the tuning-year noise units, as every world does.
    """
    start = time.time()
    ctx = dec.load_context()
    rig = ctx.rig
    d = dec.WorldDecoder(ctx=ctx, power=rig.panel.power, cal_cm=ctx.cal_cm, cal_sh=ctx.cal_sh)
    ap = dec.arm_params(tuned=tuned, arm=arm)
    res = d.run_arm(arm=arm, params=ap)
    blocks = {s: store.site_blocks(post=p, sigma=res.blocks.sigma[s]) for s, p in res.posts.items()}
    folder = OUT / "uninjected"
    atomic_parquet(
        frame=store.uninjected_table(blocks=blocks), path=folder / f"{arm}_blocks.parquet"
    )
    called = d.final_events(result=res, tau=ap.tau)
    atomic_parquet(frame=dec.events_frame(events=called), path=folder / f"{arm}_events.parquet")
    time_log(stage="uninjected", task=arm, start=start)


def truth_windows(*, w: dict, panel: object) -> list[dict]:
    """Each positive truth event with the injected megawatts at every block of its span.

    The injected change of a truth event is the sum, over the table rows of its group at its
    Primary, of the change each row makes to the series (`gen.event_delta`), averaged over each
    6-hour block. A cascaded truth event sums the rows of all its steps.

    Args:
        w: The world dict of `world_for`.
        panel: The generator's `PowerPanel` of the world.

    Returns:
        One dict per truth event with `site`, `start`, `end`, `kind`, and `truth_mw`.
    """
    import e00660_generator as gen  # ty: ignore[unresolved-import]

    rows = w["table"].to_dicts()
    out = []
    for t in w["truth"]:
        mine = [r for r in rows if r["site"] == t.site and r["group_id"] == t.group]
        delta = np.zeros(GRID_LENGTH)
        for r in mine:
            delta += gen.event_delta(row=r, panel=panel)  # ty: ignore[invalid-argument-type]
        win = store.block_slice(start=t.start, end=t.end, n_blocks=pan.N_BLOCKS)
        out.append(
            {
                "site": t.site,
                "start": int(t.start),
                "end": None if t.end is None else int(t.end),
                "kind": t.kind,
                "truth_mw": store.block_mean_of(delta=delta, window=win),
            }
        )
    return out


def _world_set(
    *,
    ctx: dec.Context,
    d: dec.WorldDecoder,
    w: dict,
    set_name: str,
    folder: Path,
    seed: int,
    tuned: dict,
    arms: tuple[str, ...],
    windows: dict,
    stored_tables: bool,
) -> None:
    """Decode a world with each arm and write matches, calls, and the posterior files."""
    rig = ctx.rig
    props = {(r["site"], r["start"]): r["prop_share"] for r in w["table"].iter_rows(named=True)}
    cascaded = store.cascaded_keys(table=w["table"])
    truth_win = truth_windows(w=w, panel=rig.panel) if any(a in dec.H_ARMS for a in arms) else []
    hard_rows = [
        {
            "site": r["site"],
            "start": int(r["start"]),
            "end": None if r["end"] is None else int(r["end"]),
            "event_class": r["event_class"],
        }
        for r in w["hard"].iter_rows(named=True)
    ]
    for arm in arms:
        ap = dec.arm_params(tuned=tuned, arm=arm)
        res = d.run_arm(arm=arm, params=ap)
        called = d.final_events(result=res, tau=ap.tau)
        extra: dict[str, list] = {}
        if res.posts:
            blocks = {
                s: store.site_blocks(post=p, sigma=res.blocks.sigma[s])
                for s, p in res.posts.items()
            }
            extra = store.call_shift_columns(called=called, blocks=blocks, is_mva=rig.panel.is_mva)
            _write_posteriors(
                folder=folder,
                arm=arm,
                seed=seed,
                blocks=blocks,
                truth_win=truth_win,
                hard_rows=hard_rows,
            )
            _write_scenarios(
                folder=folder,
                arm=arm,
                seed=seed,
                ctx=ctx,
                power=w["power"],
                blocks=blocks,
                called=called,
            )
        _write_arm(
            folder=folder,
            arm=arm,
            seed=seed,
            set_name=set_name,
            truth=w["truth"],
            called=called,
            windows=windows,
            hard=w["hard"],
            props=props,
            cascaded=cascaded,
            extra=extra,
        )
    if stored_tables:
        resid = rig.residual_of(world=cast("InjectedWorld", SimpleNamespace(power=w["power"])))
        for arm, events in _stored_tables(ctx=ctx, resid=resid).items():
            _write_arm(
                folder=folder,
                arm=arm,
                seed=seed,
                set_name=set_name,
                truth=w["truth"],
                called=events,
                windows=windows,
                hard=w["hard"],
                props=props,
                cascaded=cascaded,
                extra={},
            )


def placebo_world(*, w: dict, seed: int, ctx: dec.Context) -> dict:
    """The world of a default seed with its events moved to random times (`shuffled_placebo`).

    Sizes, durations, donors, and recipients are unchanged, and the link between an event and the
    real power and weather around it is broken, so a method that finds shuffled events as well as
    real ones is reading something other than the real record.
    """
    import e00660_generator as gen  # ty: ignore[unresolved-import]
    from e00660_ev34_detector import truths  # ty: ignore[unresolved-import]

    rig = ctx.rig
    table = gen.shuffled_placebo(panel=rig.panel, events=w["world"].events, seed=seed + SEED_OFFSET)
    return {
        "world": w["world"],
        "table": table,
        "power": gen.apply_events(panel=rig.panel, events=table),
        "truth": truths(events=table, lo=SCORED_FROM, hi=GRID_LENGTH),
        "hard": table.filter(pl.col("label") == "hard_negative"),
        "spells": None,
    }


PLACEBO_ARMS: Final[tuple[str, ...]] = ("H-nbr", "PELT")
"""Arms decoded on the shuffled-placebo table of each default world (E1-J-sym is added)."""


def score_task(*, set_name: str, seed: int, tuned: dict) -> None:
    """Decode one scored world with every arm of the set and write what scoring reads.

    Per world and arm the task writes the match rows and the call table, and for each H arm also
    the posterior quantiles inside every truth window, the posterior mean shift over every hard
    negative, and the four scenarios of the lag-free forecast's error. A default world also gets a
    shuffled-placebo decode of `PLACEBO_ARMS` and E1-J-sym, written under `scored/placebo`.
    """
    import e00660_generator as gen  # ty: ignore[unresolved-import]

    start = time.time()
    ctx = dec.load_context()
    rig = ctx.rig
    w = world_for(set_name=set_name, seed=seed, ctx=ctx)
    d = dec.WorldDecoder(ctx=ctx, power=w["power"], cal_cm=ctx.cal_cm, cal_sh=ctx.cal_sh)
    arms = ("H-own", "H-nbr") if set_name in SENSITIVITY else dec.ARMS
    windows = gen.event_windows(events=w["hard"], length=GRID_LENGTH)
    if set_name == "cold":
        windows = _spell_windows(spells=w["spells"], sites=list(rig.panel.sites))
    _world_set(
        ctx=ctx,
        d=d,
        w=w,
        set_name=set_name,
        folder=OUT / "scored" / set_name,
        seed=seed,
        tuned=tuned,
        arms=arms,
        windows=windows,
        stored_tables=set_name in ("default", "cascade", "cold"),
    )
    if set_name == "default":
        pw = placebo_world(w=w, seed=seed, ctx=ctx)
        pd_ = dec.WorldDecoder(ctx=ctx, power=pw["power"], cal_cm=ctx.cal_cm, cal_sh=ctx.cal_sh)
        _world_set(
            ctx=ctx,
            d=pd_,
            w=pw,
            set_name="placebo",
            folder=OUT / "scored" / "placebo",
            seed=seed,
            tuned=tuned,
            arms=PLACEBO_ARMS,
            windows=gen.event_windows(events=pw["hard"], length=GRID_LENGTH),
            stored_tables=True,
        )
    time_log(stage=f"score-{set_name}", task=str(seed), start=start)


def _write_posteriors(
    *,
    folder: Path,
    arm: str,
    seed: int,
    blocks: dict[str, store.SiteBlocks],
    truth_win: list[dict],
    hard_rows: list[dict],
) -> None:
    """Write the posterior quantiles in the truth windows and the shift over hard negatives."""
    atomic_parquet(
        frame=store.window_table(world=seed, arm=arm, truths=truth_win, blocks=blocks),
        path=folder / f"post_{arm}_{seed}.parquet",
    )
    if arm == "H-nbr":
        atomic_parquet(
            frame=store.uninjected_table(blocks=blocks),
            path=folder / f"blocks_{arm}_{seed}.parquet",
        )
    shifts = store.window_mean_shift(windows=hard_rows, blocks=blocks)
    if shifts.height:
        shifts = shifts.with_columns(world=pl.lit(seed), arm=pl.lit(arm))
    atomic_parquet(frame=shifts, path=folder / f"hard_{arm}_{seed}.parquet")


def _write_scenarios(
    *,
    folder: Path,
    arm: str,
    seed: int,
    ctx: dec.Context,
    power: dict[str, np.ndarray],
    blocks: dict[str, store.SiteBlocks],
    called: list,
) -> None:
    """Write the four scenarios of the lag-free forecast's error, before and after the shift."""
    rig = ctx.rig
    by_site: dict[str, list[tuple[int, int | None]]] = {}
    for e in called:
        by_site.setdefault(e.primary, []).append((e.start, e.end))
    shift = {s: np.repeat(b.mean_mw, store.BLOCK) for s, b in blocks.items() if s in power}
    frame = store.scenario_rows(
        world=seed,
        arm=arm,
        power={s: power[s] for s in shift},
        forecast={s: rig.forecast[s] for s in shift},
        capacity=rig.p653.capacity,
        shift_mw=shift,
        called=by_site,
        lo=SCORED_FROM,
    )
    atomic_parquet(frame=frame, path=folder / f"scen_{arm}_{seed}.parquet")


def _spell_windows(*, spells: pl.DataFrame, sites: list[str]) -> dict[str, np.ndarray]:
    """Boolean windows of the cold spells, the same at every Primary."""
    mask = np.zeros(GRID_LENGTH, dtype=bool)
    for r in spells.iter_rows(named=True):
        mask[r["start"] : r["end"]] = True
    return dict.fromkeys(sites, mask)


def _write_arm(
    *,
    folder: Path,
    arm: str,
    seed: int,
    set_name: str,
    truth: list,
    called: list,
    windows: dict,
    hard: pl.DataFrame,
    props: dict,
    cascaded: set[tuple[str, int]],
    extra: dict[str, list],
) -> None:
    rows = match(truth=truth, events=called, world_end=GRID_LENGTH)
    frame = pl.DataFrame(rows) if rows else pl.DataFrame()
    if frame.height:
        frame = frame.with_columns(
            world=pl.lit(seed),
            method=pl.lit(arm),
            set=pl.lit(set_name),
            prop_share=pl.Series([props.get((r["site"], r["start"]), float("nan")) for r in rows]),
            cascaded=pl.Series(
                [(r["site"], r["start"]) in cascaded for r in rows], dtype=pl.Boolean
            ),
        )
    atomic_parquet(frame=frame, path=folder / f"{arm}_{seed}.parquet")
    calls = dec.events_frame(events=called).with_columns(
        world=pl.lit(seed),
        matched=pl.Series(
            matched_flags(events=called, truth=truth, world_end=GRID_LENGTH), dtype=pl.Boolean
        ),
        on_hard_negative=pl.Series(
            hard_overlap(events=called, hard=hard, length=GRID_LENGTH, windows=windows),
            dtype=pl.Boolean,
        ),
    )
    for name, dtype in (
        ("post_mean_mw", pl.Float64),
        ("partner_post_mean_mw", pl.Float64),
        ("signed_pair", pl.Boolean),
    ):
        calls = calls.with_columns(
            pl.Series(name, extra.get(name, [None] * calls.height), dtype=dtype)
        )
    atomic_parquet(frame=calls, path=folder / f"{arm}_{seed}_calls.parquet")


def solar_bias_task(*, kind: str, k: int, tuned: dict) -> None:
    """Count false events of H-own and H-nbr in one solar-growth or seasonal-bias world.

    The world is experiment 653's injector on the common-mode residual; the injected change is
    added to the raw residual for the shared input. A false event is a called event at a Primary
    that received growth or bias, starting in the scored years, that overlaps no event of the same
    arm on the uninjected record by an intersection over union of 0.2 or more. The events of the
    uninjected record come from the cache that `uninjected_task` wrote.
    """
    from e00653_inject import GenParams, inject  # ty: ignore[unresolved-import]
    from e00653_run import setup  # ty: ignore[unresolved-import]
    from e00686_false_events import seed_of  # ty: ignore[unresolved-import]

    start = time.time()
    ctx = dec.load_context()
    rig = ctx.rig
    panel, res, fz = setup()
    seed = seed_of(k=k) if kind == "pv_growth" else 1000 + 17 * k + len("seasonal_bias")
    world, _ = inject(
        panel=panel,
        residual=res,
        sigma=fz.sigma,
        rng=np.random.default_rng(seed),
        lo=SCORED_FROM,
        hi=GRID_LENGTH,
        gen=GenParams(),
        kinds=(kind,),
    )
    change = {s: world[s] - res[s] for s in world}
    grown = [s for s in change if np.nanmax(np.abs(change[s])) > 0]
    raw = pan.raw_residual(rig=rig, power=rig.panel.power)
    raw_world = {s: raw[s] + np.nan_to_num(change[s]) for s in raw}
    sick = dec.WorldDecoder(
        ctx=ctx,
        power=rig.panel.power,
        cal_cm=ctx.cal_cm,
        cal_sh=ctx.cal_sh,
        resid_cm=world,
        resid_raw=raw_world,
    )
    rows = []
    for arm in ("H-own", "H-nbr"):
        cache = OUT / "uninjected" / f"{arm}_events.parquet"
        if not cache.exists():
            raise FileNotFoundError(f"{cache} is missing: run `score --set uninjected` first")
        before = pl.read_parquet(cache)
        ap = dec.arm_params(tuned=tuned, arm=arm)
        after = sick.final_events(result=sick.run_arm(arm=arm, params=ap), tau=ap.tau)
        false = store.false_events(
            after=after, before=before, grown=grown, lo=SCORED_FROM, length=GRID_LENGTH
        )
        years = len(grown) * (GRID_LENGTH - SCORED_FROM) / (DAY * 365.25)
        rows.append(
            {
                "kind": kind,
                "k": k,
                "arm": arm,
                "false_events": len(false),
                "grown_primaries": len(grown),
                "per_primary_year": len(false) / years,
            }
        )
    atomic_parquet(frame=pl.DataFrame(rows), path=OUT / "scored" / kind / f"false_{k}.parquet")
    time_log(stage=f"score-{kind}", task=str(k), start=start)


def with_no_boost(*, p: hmm.Priors) -> hmm.Priors:
    """The same priors with kappa 0."""
    return replace(p, kappa=0.0)

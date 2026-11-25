# ruff: noqa: E501, PT018, PLR0912, PLR0915
"""Experiment 725: the wiring checks KL1 to KL10 (design section 5).

Usage:
    e00725_checks.py pure                      # KL2's table checks and the count of changed values (no data)
    e00725_checks.py fold FOLD PHASE [--device cuda] [--only KL1,KL3,...] [--kl3r-primaries 2]
    e00725_checks.py kl5 PHASE                 # KL5: stored B1 against the `ft` series (CPU, fold-free)
    e00725_checks.py kl8                       # KL8: the look-ahead statements and the data check
    e00725_checks.py collect [--folds 2025-11,...]   # merge the results into wiring_checks_passed.json

Every check is a wiring check that cannot fail on a correct build. `fold FOLD PHASE` builds one fold's
world for one phase (one process per fold and phase, because experiment 667's fold module reads the fold
month once, at import) and writes `work/checks/<FOLD>_<PHASE>.json`. `collect` writes
`work/wiring_checks_passed.json`, which `e00725_fit.py` and `e00725_predict.py` require, only when every
check of every requested fold and phase passed. A failed check stops the experiment and goes to the manager.

Results go into the JSON file of the fold and phase; the thresholds are the design's and are fixed:

- KL1: bit-identical inputs with every cell on the analysis (64 samples per fold, both phases); the
  stored `M*` state predicting through F8's source equals the stored forecast file within 1e-6 of capacity.
- KL2: the source of every history value equals the `datetime` table; 3,763 changed values.
- KL3 and KL3n: poisoned sources change no history input outside the counted fallback cells; every
  negative control changes an input; the exempt count equals the independent KL7 count.
- KL3r: the poisoned rebuild of the analysis columns changes no input.
- KL4: no per-batch availability assertion raises, and the assertion fires on a tampered forecast origin.
- KL5: stored trailing columns equal the columns recomputed from the `ft` series within 1e-4.
- KL6: experiment 713's poisoned-power check (KA5) and next-day-run canary (KA6) pass through the source.
- KL7: under 2% of training cells fall back; under 1% of scored cells read zero; no scored cell reads an analysis.
- KL8: the look-ahead statements have no negative margin, and the stored run frame passes.
- KL9: `assemble` is called once per batch in stage 2, in stage 3, and at prediction; the stage 3 source is a
  `LiveSource`; the history inputs of `L` and `Z` differ only in the changed cells.
- KL10: with the ENS arrays replaced by the analysis arrays, `L` equals `Z` except in the day summary;
  50 random ENS cells equal an independent lookup of the raw summary row.
"""

import argparse
import json
import os
import sys
import unittest.mock
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00725_common as cm
import e00725_rule as rule
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

TRAIN_SAMPLES: Final = 64
FALLBACK_TRAIN_LIMIT: Final = 0.02
ZERO_SCORED_LIMIT: Final = 0.01
FORECAST_TOLERANCE: Final = 1e-6
KL5_TOLERANCE: Final = 1e-4
KL10_RAW_TOLERANCE: Final = 1e-4
N_KL10_CELLS: Final = 50
GRID_START: Final = datetime(2019, 9, 1, tzinfo=UTC)
CLOCK_CHANGES: Final = (date(2025, 10, 26), date(2026, 3, 29))
"""The two UK clock changes that fall in the history of the folds."""
KL5_COLUMNS: Final = (0, 1, 2, 3, 4, 8, 9, 10, 11)
"""Indices in the 12 trailing columns of the nine that are recomputed from the `ft` series (not t_min_day, t_max_day, t_cold_snap_days)."""


def checks_dir(*, data_dir: Path) -> Path:
    """Return the folder of the per-fold check results."""
    return cm.work(data_dir=data_dir) / "checks"


def contract_index(*, day: date) -> int:
    """Return the grid index of 12:00 UTC on `day` from `datetime` arithmetic."""
    stamp = datetime(day.year, day.month, day.day, 12, tzinfo=UTC)
    return int((stamp - GRID_START).total_seconds() // 1800)


def contract_datetime(*, c_idx: int) -> datetime:
    """Return the 12:00 UTC contract origin of a grid index as a datetime."""
    return GRID_START + timedelta(minutes=30 * int(c_idx))


def pick_origins(*, available: np.ndarray) -> list[int]:
    """Return 3 contract origins from the sorted `available` ones, plus one near each clock change (if any).

    The three are the first, the middle, and the last; a clock-change origin is the available origin
    nearest to two days after the change, if one lies within 7 days of it.
    """
    available = np.unique(np.asarray(available, dtype=np.int64))
    picks = {int(available[0]), int(available[len(available) // 2]), int(available[-1])}
    for change in CLOCK_CHANGES:
        target = contract_index(day=change + timedelta(days=2))
        nearest = int(available[np.argmin(np.abs(available - target))])
        if abs(nearest - target) <= 7 * rule.STEPS_PER_DAY:
            picks.add(nearest)
    return sorted(picks)


# ------------------------------------------------------------------------------------------ KL2 pure


def kl2_pure(*, contracts: Sequence[datetime]) -> dict:
    """Check KL2's table: the grid table equals the `datetime` table, and 3,763 values change."""
    import e00725_rule as r

    for contract in contracts:
        for name in r.RULES:
            grid = r.cell_sources(rule=name)  # ty: ignore[invalid-argument-type]
            reference = r.reference_table(contract=contract, rule=name)  # ty: ignore[invalid-argument-type]
            assert np.array_equal(grid, reference), (
                f"KL2: the grid table of rule {name} differs from the datetime table at {contract}"
            )
    changed = int(r.changed_cells(rule="L").sum())
    assert changed == r.CHANGED_VALUES_PER_SAMPLE, f"KL2: {changed} changed values, not 3763"
    return {"pass": True, "n_contracts": len(contracts), "changed_values_L": changed,
            "changed_values_E": int(r.changed_cells(rule="E").sum())}  # fmt: skip


def summary_slot_reference(*, day_read: date, day0: int) -> int:
    """Return the slot of the CAMS day summary of UTC day `day_read` from `datetime` arithmetic."""
    return (day_read - date(1970, 1, 1)).days - day0


# ---------------------------------------------------------------------------------- the fold's checks


def sentinel_source(*, live, day0: int):  # noqa: ANN001, ANN201
    """Return a copy of `live` whose arrays hold source sentinels, so that a value identifies its source.

    The analysis CAMS columns hold 100, the analysis ERA5 columns 200, the analysis day summary 500, and
    the ENS arrays 400 (every cell found, with a forecast origin equal to the token's own stamp). The CAMS
    day summaries hold the slot modulo 1024 in column 19 and 1024 plus that in column 20.
    """
    import copy

    import torch  # ty: ignore[unresolved-import]

    new = copy.copy(live)
    n_primaries, n_grid, _ = live.dense_weather.shape
    dev = live.dense_weather.device
    dense = torch.zeros(
        (n_primaries, n_grid, rule.N_WEATHER_COLUMNS), dtype=torch.float16, device=dev
    )
    dense[..., list(rule.CAMS_COLUMNS)] = 100.0
    dense[..., list(rule.ERA5_COLUMNS)] = 200.0
    dense[..., list(rule.DAY_COLUMNS)] = 500.0
    new.dense_weather = dense
    new.ens_z = torch.full(
        (n_primaries, n_grid, rule.N_ENS_COLUMNS), 400.0, dtype=torch.float16, device=dev
    )
    new.ens_found = torch.ones((n_primaries, n_grid), dtype=torch.bool, device=dev)
    new.ens_origin = torch.arange(n_grid, device=dev)[None, :].expand(n_primaries, -1).clone()
    slots = torch.arange(live.cams_days.shape[1], device=dev) % 1024
    days = torch.zeros_like(live.cams_days)
    days[..., 0] = slots[None, :].float()
    days[..., 1] = (slots + 1024)[None, :].float()
    new.cams_days = days
    new.counters = dict.fromkeys(live.counters, 0)
    del day0
    return new


def kl2_batch(*, live, samples, day0: int) -> dict:  # noqa: ANN001
    """Check KL2 on batches: the source of every history value equals the `datetime` table of its origin."""
    import e00725_poison as pz

    sent = sentinel_source(live=live, day0=day0)
    for sample in samples:
        out = pz.assemble_one(source=sent, sample=sample)
        hist = out.inputs.hist_weather[0].float().cpu().numpy()
        contract = contract_datetime(c_idx=sample.c_idx)
        table = rule.reference_table(contract=contract, rule=live.rule_name)
        expected = np.zeros_like(hist)
        expected[table == rule.SRC_CAMS] = 100.0
        expected[table == rule.SRC_ERA5] = 200.0
        expected[table == rule.SRC_ENS] = 400.0
        for slot in range(rule.N_ROWS):
            age = rule.FIRST_AGE + slot
            for j in range(rule.STEPS_PER_DAY):
                stamp = contract - timedelta(days=age) + timedelta(minutes=30 * j)
                for col in rule.DAY_COLUMNS:
                    code = table[slot, j, col]
                    if live.rule_name == "Z" or (code == rule.SRC_DAY and not live.live_summary):
                        expected[slot, j, col] = 500.0
                        continue
                    read = (
                        stamp.date() - timedelta(days=1)
                        if code == rule.SRC_DAY
                        else contract.date() - timedelta(days=rule.CAMS_LAST_DAY_BEFORE)
                    )
                    value = summary_slot_reference(day_read=read, day0=day0) % 1024
                    expected[slot, j, col] = value + (1024 if col == rule.DAY_COLUMNS[1] else 0)
        wrong = np.argwhere(hist != expected)
        assert len(wrong) == 0, (
            f"KL2: {len(wrong)} history values read another source than the datetime table "
            f"(first at {wrong[0].tolist()}, sample {sample})"
        )
    return {"pass": True, "n_samples": len(samples)}


def assemble_equal(*, a, b) -> bool:  # noqa: ANN001
    """Return whether two assembled batches have bit-identical inputs, targets, and indices."""
    import torch  # ty: ignore[unresolved-import]

    same = torch.equal(a.target.nan_to_num(), b.target.nan_to_num())
    same &= torch.equal(a.target_idx, b.target_idx) and torch.equal(a.hist_idx, b.hist_idx)
    for name in a.inputs.__dict__:
        same &= torch.equal(getattr(a.inputs, name), getattr(b.inputs, name))
    return bool(same)


def kl1_batch(*, live, picks) -> dict:  # noqa: ANN001
    """Check KL1 on batches: with every cell on the analysis, F8's batch equals experiment 713's `Source.assemble`."""
    import e00713_batch as bt  # ty: ignore[unresolved-import]
    import e00725_poison as pz

    off = live.analysis_copy()
    for sample in picks:
        ours = pz.assemble_one(source=off, sample=sample)
        dev = live.device
        import torch  # ty: ignore[unresolved-import]

        theirs = bt.Source.assemble(
            live, primary=torch.tensor([sample.primary], device=dev),
            c_idx=torch.tensor([sample.c_idx], device=dev),
            lead_day=torch.tensor([sample.lead_day], device=dev), ensemble=True,
        )  # fmt: skip
        assert assemble_equal(a=ours, b=theirs), (
            f"KL1: the rule-off batch differs from Source.assemble at {sample}"
        )
    return {"pass": True, "n_samples": len(picks)}


def kl1_forecast(*, lw, data_dir: Path, month: str, seed: int = 0) -> dict:  # noqa: ANN001
    """Check KL1's forecast: the stored `M*` state, predicting through F8's source with the rule off, equals the stored file."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_arms as arms713  # ty: ignore[unresolved-import]
    import e00713_model as mdl  # ty: ignore[unresolved-import]
    import e00713_predict as pred713  # ty: ignore[unresolved-import]
    import e00725_frozen as fz
    import polars as pl
    import torch  # ty: ignore[unresolved-import]

    device = lw.live.device
    spec = arms713.resolve(arm=cm.M_STAR, data_dir=data_dir)
    z_live = (
        lw.live.analysis_copy()
    )  # exercises the rewrite path of `LiveSource.assemble` with the rule off
    lw.data.source, lw.world.source = z_live, z_live
    model = mdl.MsaNet(switches=spec.switches).to(device)
    state = arms713.model_file(data_dir=data_dir, arm=cm.M_STAR, month=month, seed=seed, stage=3)
    model.load_state_dict(torch.load(state, map_location=device))
    sites = k578.sites().sort("primary")
    frame = pred713.forecast_frame(
        model=model, data=lw.data, spec=spec, world=lw.world, seed=seed, mask_history=False,
        grid_start_us=prep422.GRID_START_US, time_series_ids=sites["time_series_id"].to_numpy(),
    )  # fmt: skip
    stored = pl.read_parquet(
        arms713.fit_file(data_dir=data_dir, arm=cm.M_STAR, month=month, seed=seed)
    )
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    q = list(pred713.QCOLS)
    joined = frame.select(*keys, *q).join(
        stored.select(*keys, *q), on=keys, how="inner", suffix="_stored"
    )
    assert joined.height == stored.height == frame.height, (
        joined.height,
        stored.height,
        frame.height,
    )
    worst = max(float((joined[c] - joined[f"{c}_stored"]).abs().max()) for c in q)  # ty: ignore[invalid-argument-type]
    assert worst <= FORECAST_TOLERANCE, (
        f"KL1: the forecast differs from the stored file by {worst:.2e} of capacity"
    )
    del fz
    return {"pass": True, "max_abs_difference": worst, "rows": joined.height}


def t3_known_answer(*, live, picks) -> dict:  # noqa: ANN001
    """Check M6: with the target-row summary moved back by one day (the parent's own day), the batch equals the unshifted batch bit for bit."""
    import copy

    import e00725_poison as pz

    moved = copy.copy(live)
    moved.target_summary_back_days = 1
    for sample in picks:
        a = pz.assemble_one(source=moved, sample=sample)
        b = pz.assemble_one(source=live, sample=sample)
        assert assemble_equal(a=a, b=b), f"T3: a shift of one day changes the batch at {sample}"
    return {"pass": True, "n_samples": len(picks)}


def kl4(*, live, picks) -> dict:  # noqa: ANN001
    """Check KL4: no availability assertion raises on the picks, and one raises on a tampered forecast origin."""
    import e00725_poison as pz

    for sample in picks:
        pz.assemble_one(source=live, sample=sample)
    tampered = live.with_rule(rule_name=live.rule_name)
    tampered.ens_origin = live.ens_origin + 10**6
    raised = False
    try:
        pz.assemble_one(source=tampered, sample=picks[0])
    except AssertionError:
        raised = True
    assert raised, "KL4: a forecast origin after the token's stamp did not raise"
    return {"pass": True, "n_samples": len(picks), "tamper_raised": raised}


def kl6(*, live, picks) -> dict:  # noqa: ANN001
    """Check KL6: experiment 713's KA5 (poisoned power) and KA6 (next-day run) pass through F8's source."""
    import e00713_batch as bt  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    dev = live.device
    primary = torch.tensor([s.primary for s in picks], device=dev)
    c_idx = torch.tensor([s.c_idx for s in picks], device=dev)
    lead = torch.tensor([s.lead_day for s in picks], device=dev)
    batch = live.assemble(primary=primary, c_idx=c_idx, lead_day=lead, ensemble=True)
    ka5 = bt.poison_check(
        source=live, batch=batch, ages=None, age_exists=None, ensemble=True, lead_day=lead, limit=len(picks)
    )  # fmt: skip
    ka6 = bt.ka6_canary(source=live, primary=primary, c_idx=c_idx, lead_day=lead)
    return {"pass": True, "KA5": ka5, "KA6": ka6}


def kl7(*, cache, picks_primary: np.ndarray, picks_c: np.ndarray, rule_name: str) -> dict:  # noqa: ANN001
    """Count the fallback cells of the samples (KL7): the cells the rule reads from ENS whose half-hour has no row."""
    import e00725_poison as pz

    per_token = rule.ens_mask(rule=rule_name).sum()  # ty: ignore[invalid-argument-type]
    wanted = int(per_token) * len(picks_c)
    found = np.asarray(cache.found)
    fallback = pz.fallback_count_reference(
        found=found, rule_name=rule_name, primary=picks_primary, c_idx=picks_c
    )  # fmt: skip
    per_primary = {}
    for p in range(found.shape[0]):
        sel = picks_primary == p
        if sel.any():
            per_primary[p] = pz.fallback_count_reference(
                found=found, rule_name=rule_name, primary=picks_primary[sel], c_idx=picks_c[sel]
            ) / (int(per_token) * int(sel.sum()))
    return {
        "rule": rule_name, "cells": wanted, "fallback": fallback, "share": fallback / max(wanted, 1),
        "share_per_primary": {str(k): round(v, 5) for k, v in per_primary.items()},
    }  # fmt: skip


def kl7_fold(*, lw, phase: str) -> dict:  # noqa: ANN001
    """Run KL7 for arms `L` and `E` on the fold's training pairs or scored origins and apply the limits.

    The shares (per arm, per Primary) are printed and stored before any limit is evaluated.
    """
    import e00713_train as tr  # ty: ignore[unresolved-import]
    import e00725_poison as pz

    if phase == "train":
        pairs = [tr.make_samples(world=lw.world, stage=s, placebo=False) for s in (2, 3)]
        primary = np.concatenate([p.primary for p in pairs])
        c_idx = np.concatenate([p.c_idx for p in pairs])
        limit = FALLBACK_TRAIN_LIMIT
    else:
        origins = lw.data.test_origin_idx + rule.NATIVE_OFFSET_STEPS
        primary, c_idx = (a.ravel() for a in np.meshgrid(np.arange(16), origins, indexing="ij"))
        limit = ZERO_SCORED_LIMIT
    out: dict = {"n_samples": len(c_idx), "limit": limit}
    for name in ("L", "E"):
        result = kl7(cache=lw.cache, picks_primary=primary, picks_c=c_idx, rule_name=name)
        result["limit"] = limit
        result["pass"] = bool(result["share"] < limit)
        out[name] = result
    print("KL7 shares:", json.dumps(out, default=str), flush=True)
    if phase == "predict":
        # no scored cell reads an analysis fallback: a fallback cell of a scored batch reads exactly 0
        n_fallback = 0
        for p, c in zip(primary.tolist(), c_idx.tolist(), strict=True):
            batch = pz.assemble_one(source=lw.live, sample=pz.OneSample(primary=p, c_idx=c))
            plan = lw.live.plan(primary=batch.inputs.primary, hist_idx=batch.hist_idx)
            values = batch.inputs.hist_weather[..., : rule.N_ENS_COLUMNS][plan.fallback]
            n_fallback += int(values.numel())
            assert bool((values == 0).all()), "KL7: a scored fallback cell reads a nonzero value"
        out["scored_fallback_cells"] = n_fallback
        out["scored_fallback_reads_zero"] = True
    out["pass"] = bool(all(out[n]["pass"] for n in ("L", "E")))
    assert out["pass"], (
        f"KL7: the share of ENS cells with no row is above {limit} ({phase} phase); "
        "a breach goes to the manager as a coverage finding"
    )
    return out


def kl9(*, lw, month: str, seed: int, data_dir: Path, device: str) -> dict:  # noqa: ANN001
    """Check KL9: `assemble` runs once per batch in stage 2, in stage 3, and at prediction; stage 3 keeps the type; L and Z differ only in the changed cells."""
    import e00713_train as tr  # ty: ignore[unresolved-import]
    import e00725_batch as lb
    import e00725_fit as fit_mod

    live = lw.live
    out: dict = {}
    batches = {"n": 0}
    original = tr.batch_for

    def counting(**kwargs):  # noqa: ANN003, ANN202
        batches["n"] += 1
        return original(**kwargs)

    for stage in cm.STAGES:
        samples = tr.make_samples(world=lw.world, stage=stage, placebo=False)
        train_idx = np.nonzero(~samples.is_val)[0][:768]
        val_idx = np.nonzero(samples.is_val)[0][:256]
        keep = np.concatenate([train_idx, val_idx])
        sub = tr.SampleSet(
            primary=samples.primary[keep], c_idx=samples.c_idx[keep], is_val=samples.is_val[keep]
        )  # fmt: skip
        model = fit_mod.new_model(seed=seed, month=month, device=device, data_dir=data_dir)
        before_calls = live.counters["assemble_calls"]
        batches["n"] = 0
        with unittest.mock.patch.object(tr, "batch_for", counting):
            tr.train_stage(
                model=model, world=lw.world, stage=stage, samples=sub, seed=seed, encoders="frozen",
                max_passes=1,
            )  # fmt: skip
        calls = live.counters["assemble_calls"] - before_calls
        assert calls == batches["n"] > 0, (
            f"KL9: stage {stage} made {calls} assemble calls for {batches['n']} batches"
        )
        out[f"stage{stage}_calls"] = calls
        if stage == 3:
            source3 = tr.synthetic_source(world=lw.world, samples=sub)
            assert type(source3) is lb.LiveSource, f"KL9: stage 3 reads a {type(source3).__name__}"
            assert source3.counters is live.counters, (
                "KL9: the stage 3 source does not share the counters"
            )
            z3 = tr.synthetic_source(
                world=_with_source(lw.world, live.with_rule(rule_name="Z")), samples=sub
            )
            idx = np.arange(min(256, len(sub)))
            kw = {
                "samples": sub,
                "index": idx,
                "lead": np.zeros(len(idx), dtype=np.int64),
                "ensemble": True,
                "cut": tr.target_end(world=lw.world, stage=3),
                "dropout": None,
            }
            a = tr.batch_for(source=source3, **kw)
            b = tr.batch_for(source=z3, **kw)
            diff = (a.inputs.hist_weather != b.inputs.hist_weather).any(dim=0).cpu().numpy()
            changed = rule.changed_cells(rule="L")
            assert not bool((diff & ~changed).any()), (
                "KL9: L and Z differ outside the changed cells"
            )
            sources = rule.cell_sources(rule="L")
            ens = sources == rule.SRC_ENS
            cams_cols = [0, *range(15, 19)]
            blocks = {
                "cams_columns_young_tokens": (diff[..., cams_cols] & ens[..., cams_cols]).any(),
                "day_summary": (diff[..., 19:21] & (sources[..., 19:21] == rule.SRC_DAY_D3)).any(),
                "columns_1_to_14_where_column_0_reads_cams": (
                    diff[..., 1:15] & ens[..., 1:15] & (sources[..., :1] == rule.SRC_CAMS)
                ).any(),
            }
            for name, hit in blocks.items():
                assert bool(hit), (
                    f"KL9: no value of the changed block {name} differs between L and Z"
                )
                out[f"block_{name}_differs"] = True
    out["pass"] = True
    return out


def kl9_predict(*, lw, month: str, data_dir: Path, device: str) -> dict:  # noqa: ANN001
    """Check KL9 at prediction: `assemble` runs once per batch of `forecast_frame`, and the data source is a `LiveSource`."""
    import copy

    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_arms as arms713  # ty: ignore[unresolved-import]
    import e00713_predict as pred713  # ty: ignore[unresolved-import]
    import e00713_train as tr  # ty: ignore[unresolved-import]
    import e00725_batch as lb
    import e00725_fit as fit_mod

    assert type(lw.data.source) is lb.LiveSource, (
        f"KL9: the data source is a {type(lw.data.source).__name__}"
    )
    spec = arms713.resolve(arm=cm.M_STAR, data_dir=data_dir)
    model = fit_mod.new_model(seed=0, month=month, device=device, data_dir=data_dir)
    data = copy.copy(lw.data)
    data.test_origin_idx = lw.data.test_origin_idx[:1]
    n_pairs = 16 * 14
    n_batches = -(-n_pairs // pred713.PREDICT_BATCH)
    batches = {"n": 0}
    original = tr.batch_for

    def counting(**kwargs):  # noqa: ANN003, ANN202
        batches["n"] += 1
        return original(**kwargs)

    before = lw.live.counters["assemble_calls"]
    sites = k578.sites().sort("primary")
    with unittest.mock.patch.object(tr, "batch_for", counting):
        pred713.forecast_frame(
            model=model, data=data, spec=spec, world=lw.world, seed=0, mask_history=False,
            grid_start_us=prep422.GRID_START_US, time_series_ids=sites["time_series_id"].to_numpy(),
        )  # fmt: skip
    calls = lw.live.counters["assemble_calls"] - before
    assert batches["n"] == n_batches and calls == n_batches, (
        f"KL9: {calls} assemble calls and {batches['n']} batches at prediction, {n_batches} expected"
    )
    return {"pass": True, "prediction_batches": n_batches, "assemble_calls": calls}


def _with_source(world, source):  # noqa: ANN001, ANN202
    """Return a copy of an experiment 713 `World` that reads `source`."""
    import dataclasses

    return dataclasses.replace(world, source=source)


def kl10(*, lw, picks, data_dir: Path, phase: str) -> dict:  # noqa: ANN001
    """Check KL10: ENS arrays replaced by the analysis arrays give the rule-off batch, and 50 ENS cells match the raw rows."""
    import copy

    import e00725_poison as pz
    import torch  # ty: ignore[unresolved-import]

    live = lw.live
    n_grid = live.dense_weather.shape[1]
    sub = copy.copy(live)
    sub.ens_z = live.dense_weather[..., : rule.N_ENS_COLUMNS].clone()
    sub.ens_found = torch.ones_like(live.ens_found)
    sub.ens_origin = (
        torch.arange(n_grid, device=live.device)[None, :]
        .expand(live.dense_weather.shape[0], -1)
        .clone()
    )
    sub.counters = dict.fromkeys(live.counters, 0)
    z = live.with_rule(rule_name="Z")
    expected_changed = rule.cell_sources(rule="L") == rule.SRC_DAY_D3
    for sample in picks:
        a = pz.assemble_one(source=sub, sample=sample)
        b = pz.assemble_one(source=z, sample=sample)
        diff = (a.inputs.hist_weather != b.inputs.hist_weather)[0].cpu().numpy()
        assert not bool((diff & ~expected_changed).any()), (
            f"KL10: a value other than the day summary differs at {sample}"
        )
        assert expected_changed.sum() == 48
        tokens = np.argwhere(expected_changed[..., 19])
        slots = live.summary_slots(
            c_idx=torch.tensor([sample.c_idx], device=live.device), hist_idx=a.hist_idx, live=True
        )[0].cpu().numpy()  # fmt: skip
        want = (
            live.cams_days[sample.primary, torch.as_tensor(slots, device=live.device)]
            .half()
            .float()
            .cpu()
            .numpy()
        )
        got = a.inputs.hist_weather[0, ..., 19:21].cpu().numpy()
        for r, j in tokens:
            assert np.array_equal(got[r, j], want[r, j]), (
                f"KL10: the day summary of D - 3 is wrong at token {(r, j)}"
            )
    cells = _kl10_raw_cells(lw=lw, picks=picks, phase=phase, n=N_KL10_CELLS)
    return {"pass": True, "n_samples": len(picks), "raw_cells_checked": cells}


def _kl10_raw_cells(*, lw, picks, phase: str, n: int) -> dict:  # noqa: ANN001
    """Compare `n` random ENS weather cells, spread over the 7 weather columns, with an independent Polars lookup of the raw summary rows.

    The raw folder is the one the phase's cache was built from (`meta.json` records it). Each cell is
    compared twice: the raw value in `cache.raw` within 1e-4, and the standardised float16 value exactly.
    """
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00428_data as data428  # ty: ignore[unresolved-import]
    import e00725_poison as pz
    import e00725_sources as src
    import polars as pl

    rng = np.random.default_rng(0)
    live = lw.live
    raw_dir = Path(src.read_meta(phase=phase, data_dir=ML_EXPERIMENTS_DATA_DIR)["raw_dir"])
    sites = {int(r["primary"]): r["site"] for r in prep.primaries().iter_rows(named=True)}
    keep37, _, _ = data428.non_lag_layout()
    mean, std = lw.stats["weather_mean"], lw.stats["weather_std"]
    done, n_zero, worst = 0, 0, 0.0
    while done < n:
        col = done % src.N_WEATHER
        sample = picks[int(rng.integers(len(picks)))]
        out = pz.assemble_one(source=live, sample=sample)
        plan = live.plan(primary=out.inputs.primary, hist_idx=out.hist_idx)
        cells = np.argwhere(plan.used[0, ..., col].cpu().numpy())
        if len(cells) == 0:
            continue
        r, j = cells[int(rng.integers(len(cells)))]
        g = int(out.hist_idx[0, r, j])
        valid_us = prep422.GRID_START_US + g * prep422.STEP_US
        z = np.load(raw_dir / "summary" / f"{sites[sample.primary]}.npz")
        frame = pl.DataFrame(
            {"valid": z["valid"], "lead": z["lead"], "w": z["x"][:, keep37][:, col]}
        )
        row = frame.filter(pl.col("valid") == valid_us).sort("lead").head(1)
        assert row.height == 1, f"KL10: no raw summary row at valid time {valid_us}"
        raw_value = float(row["w"][0])
        n_zero += int(raw_value == 0.0)
        cached = float(lw.cache.raw[sample.primary, g, col])
        worst = max(worst, abs(cached - raw_value))
        assert abs(cached - raw_value) <= KL10_RAW_TOLERANCE, (
            f"KL10: the cached raw value of cell {(r, j, col)} is {cached}, the raw row gives {raw_value}"
        )
        expected = np.float16(np.clip((raw_value - mean[col]) / std[col], -5.0, 5.0))
        got = np.float16(float(out.inputs.hist_weather[0, r, j, col]))
        assert expected == got, (
            f"KL10: the ENS cell {(r, j, col)} reads {got}, the raw row gives {expected} ({phase})"
        )
        done += 1
    return {
        "cells": done,
        "exact_zero_cells": n_zero,
        "worst_raw_difference": worst,
        "raw_dir": str(raw_dir),
    }


def run_fold(*, month: str, phase: str, device: str, data_dir: Path, only: set[str]) -> dict:
    """Run the checks of one fold and one phase and write the result file."""
    os.environ["E00667_PHASE"] = phase
    import e00713_train as tr  # ty: ignore[unresolved-import]
    import e00725_frozen as fz
    import e00725_poison as pz
    import e00725_world as world_mod

    fz.assert_frozen_checkout()
    fz.limit_gpu_memory(device=device)
    lw = world_mod.load_world(
        month=month, seed=0, rule_name="L", phase=phase, device=device, data_dir=data_dir
    )  # fmt: skip
    day0 = lw.live.day0
    results: dict = {"month": month, "phase": phase}
    rng = np.random.default_rng(0)
    if phase == "train":
        samples = tr.make_samples(world=lw.world, stage=2, placebo=False)
        keep = np.nonzero(~samples.is_val)[0]
        early = keep[np.argsort(samples.c_idx[keep])[:8]]
        chosen = np.concatenate([early, rng.choice(keep, size=TRAIN_SAMPLES - 8, replace=False)])
        picks = [
            pz.OneSample(primary=int(samples.primary[i]), c_idx=int(samples.c_idx[i]))
            for i in chosen
        ]
        origins = pick_origins(available=np.unique(samples.c_idx[keep]))
    else:
        origins_all = lw.data.test_origin_idx + lay_native_offset()
        picks = [pz.OneSample(primary=int(p), c_idx=int(c)) for c in origins_all for p in range(16)]
        origins = pick_origins(available=origins_all)
    two_origin_picks = [pz.OneSample(primary=p, c_idx=c) for c in origins for p in (0, 5)]
    run = lambda name: not only or name in only  # noqa: E731
    results["origins"] = origins
    if run("KL1"):
        results["KL1"] = kl1_batch(live=lw.live, picks=picks[:TRAIN_SAMPLES])
    if run("KL2"):
        results["KL2"] = {
            "table": kl2_pure(contracts=[contract_datetime(c_idx=c) for c in origins]),
            **{name: kl2_batch(live=lw.live.with_rule(rule_name=name), samples=two_origin_picks, day0=day0) for name in rule.RULES},
        }  # fmt: skip
    if run("KL3"):
        results["KL3"] = pz.kl3(live=lw.live, samples=picks, phase=phase, with_controls=True)
        ref = pz.fallback_count_reference(
            found=np.asarray(lw.cache.found), rule_name="L",
            primary=np.array([s.primary for s in picks]), c_idx=np.array([s.c_idx for s in picks]),
        )  # fmt: skip
        results["KL3"]["fallback_reference"] = ref
        if phase == "train":
            assert results["KL3"]["n_exempt_cells"] == ref, (
                f"KL3: {results['KL3']['n_exempt_cells']} exempt cells, KL7 counts {ref}"
            )
        else:
            assert results["KL3"]["n_exempt_cells"] == 0, "KL3: a scored origin has exempt cells"
    if run("T3"):
        try:
            results["T3_known_answer"] = t3_known_answer(live=lw.live, picks=two_origin_picks)
        except AssertionError as error:
            # T3 is report-only: the failure is recorded and gates phase 4, not the fold check
            results["T3_known_answer"] = {"pass": False, "error": str(error)}
    if run("KL4"):
        results["KL4"] = kl4(live=lw.live, picks=picks[:TRAIN_SAMPLES])
    if run("KL6") and phase == "train":
        results["KL6"] = kl6(live=lw.live, picks=picks[:TRAIN_SAMPLES])
    if run("KL7"):
        results["KL7"] = kl7_fold(lw=lw, phase=phase)
    if run("KL9") and phase == "train" and month == cm.ALL_FOLDS[0]:
        results["KL9"] = kl9(lw=lw, month=month, seed=0, data_dir=data_dir, device=device)
    if run("KL10"):
        results["KL10"] = kl10(lw=lw, picks=two_origin_picks, data_dir=data_dir, phase=phase)
    if phase == "predict" and month == cm.ALL_FOLDS[0]:
        # the two checks below swap the sources or run the network, so they come last
        if run("KL9"):
            results["KL9_predict"] = kl9_predict(
                lw=lw, month=month, data_dir=data_dir, device=device
            )
        if run("KL1"):
            results["KL1_forecast"] = kl1_forecast(lw=lw, data_dir=data_dir, month=month)
    results["pass"] = all(
        v.get("pass", True)
        for k, v in results.items()
        if isinstance(v, dict) and k != "T3_known_answer"
    )
    path = checks_dir(data_dir=data_dir) / f"{month}_{phase}.json"
    cm.write_check_json(obj=results, path=path)
    print(json.dumps(results, indent=1, default=str)[:4000], flush=True)
    return results


def kl3r_fold(*, month: str, data_dir: Path, n_primaries: int) -> dict:
    """Run KL3r for one fold on the CPU, outside the GPU slot (it rebuilds the analysis arrays from poisoned inputs).

    The rebuild takes about 1 minute per Primary and poisoned copy, so the job runs with `--device cpu`
    semantics: the world is built on the CPU and no GPU memory is touched.
    """
    os.environ["E00667_PHASE"] = "train"
    import e00713_train as tr  # ty: ignore[unresolved-import]
    import e00725_frozen as fz
    import e00725_poison as pz
    import e00725_world as world_mod

    fz.assert_frozen_checkout()
    lw = world_mod.load_world(
        month=month, seed=0, rule_name="L", phase="train", device="cpu", data_dir=data_dir
    )
    samples = tr.make_samples(world=lw.world, stage=2, placebo=False)
    origins = pick_origins(available=np.unique(samples.c_idx[~samples.is_val]))
    chosen = [pz.OneSample(primary=p, c_idx=c) for c in origins for p in range(n_primaries)]
    scratch = cm.work(data_dir=data_dir) / "scratch"
    result = {
        "month": month,
        "phase": "train",
        "origins": origins,
        **pz.kl3r(lw=lw, samples=chosen, phase="train", scratch=scratch),
    }
    cm.write_check_json(obj=result, path=checks_dir(data_dir=data_dir) / f"{month}_kl3r.json")
    print(json.dumps(result, indent=1), flush=True)
    return result


def lay_native_offset() -> int:
    """Return the grid steps from the native to the contract origin."""
    return rule.NATIVE_OFFSET_STEPS


# ----------------------------------------------------------------------------------------- KL5, KL8


def common_grid() -> np.ndarray:
    """Return the half-hourly grid that experiment 437's `dense_arrays` builds on (after any `redirect`)."""
    import e00087_common as common  # ty: ignore[unresolved-import]

    return common.grid_times().to_numpy()


def kl5(*, phase: str, data_dir: Path) -> dict:
    """Check KL5: the nine trailing columns recomputed from the `ft` series equal experiment 437's stored values.

    For each Primary, the `ft` table's weather is passed through `e00437_features.dense_arrays` (with
    `e00418_prep.dense_weather` patched to return it), and the nine columns are compared at every ENS
    cell of 3 origins per fold wherever the `ft` series has no gap in the 61 half-hours up to the cell.
    """
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00418_prep as prep418  # ty: ignore[unresolved-import]
    import e00437_features as f437  # ty: ignore[unresolved-import]
    import e00632_data as d632  # ty: ignore[unresolved-import]
    import e00667_extend as ext  # ty: ignore[unresolved-import]
    import e00725_sources as src

    # The cache, and experiment 713's world in both phases, sit on experiment 667's extended grid
    # (126,624 half-hours to 2026-11-20). `dense_arrays` builds its clear-sky array on the grid of
    # `e00087_common.GRID_END`, which is 2026-07-31 (121,248 half-hours) until `redirect` extends it.
    ext.redirect()
    cache = src.load_cache(phase=phase, data_dir=data_dir)
    assert len(common_grid()) == cache.raw.shape[1], (len(common_grid()), cache.raw.shape)
    shortwave = prep418.WEATHER.index("shortwave")
    temperature = prep418.WEATHER.index("temperature_2m")
    local_date = f437.local_date_codes()
    names = [d632.B1_GRID_NAMES[k] for k in KL5_COLUMNS]
    offsets = rule.token_grid_index(day=rule.REFERENCE_DAY) - (
        rule.STEPS_PER_DAY * rule.REFERENCE_DAY + 24
    )
    ens = rule.ens_mask(rule="L")[..., 7:19].any(axis=-1)  # ty: ignore[invalid-argument-type]
    contracts = []
    for month in cm.ALL_FOLDS:
        year, mon = (int(x) for x in month.split("-"))
        first = date(year, mon, 1)
        deltas = (8, 15, 22) if phase == "predict" else (-24, -15, -5)
        contracts += [contract_index(day=first + timedelta(days=d)) for d in deltas]
    compared = dict.fromkeys(names, 0)
    in_scope = dict.fromkeys(names, 0)
    one_sided_nan = dict.fromkeys(names, 0)
    worst = dict.fromkeys(names, 0.0)
    for row in prep.primaries().iter_rows(named=True):
        p = int(row["primary"])
        weather = np.asarray(cache.raw[p, :, : src.N_WEATHER], dtype=np.float32)
        with unittest.mock.patch.object(prep418, "dense_weather", lambda *, w=weather, **_: w):
            dense = f437.dense_arrays(row=row, local_date=local_date)
        finite = np.isfinite(weather[:, shortwave]) & np.isfinite(weather[:, temperature])
        csum = np.concatenate([[0], np.cumsum(finite)])
        for c in contracts:
            idx = c + offsets[ens]
            window_ok = (csum[idx + 1] - csum[np.maximum(idx - 60, 0)]) == (
                idx + 1 - np.maximum(idx - 60, 0)
            )
            use = idx[window_ok & np.asarray(cache.found[p, idx])]
            for k, name in zip(KL5_COLUMNS, names, strict=True):
                stored = np.asarray(cache.raw[p, use, src.N_WEATHER + k], dtype=np.float64)
                diff = np.abs(stored - dense[name][use])
                ref = dense[name][use]
                one_sided_nan[name] += int((np.isnan(stored) != np.isnan(ref)).sum())
                in_scope[name] += len(idx)
                compared[name] += len(use)
                worst[name] = max(worst[name], float(np.nanmax(diff)) if len(diff) else 0.0)
    result = {
        "pass": all(w <= KL5_TOLERANCE for w in worst.values()) and not any(one_sided_nan.values()),
        "phase": phase, "cells_compared": compared, "share_of_cells_compared": {
            k: compared[k] / max(in_scope[k], 1) for k in names
        },
        "one_sided_nan_cells": one_sided_nan, "worst_abs_difference": worst,
    }  # fmt: skip
    cm.write_check_json(obj=result, path=checks_dir(data_dir=data_dir) / f"kl5_{phase}.json")
    print(json.dumps(result, indent=1), flush=True)
    assert result["pass"], f"KL5: stored B1 columns differ from the ft series: {worst}"
    return result


def kl8(*, data_dir: Path) -> dict:
    """Check KL8: print the look-ahead statements for `L`, `Z`, and `E` and require no negative margin and a PASS on the stored run frame."""
    import e00725_weather_lookahead as wl

    contract = datetime(2026, 3, 10, 12, tzinfo=UTC)
    out: dict = {"pass": True}
    for name in rule.RULES:
        print("\n".join(wl.statement(contract=contract, rule_name=name)), flush=True)  # ty: ignore[invalid-argument-type]
        negative = [
            x.text for x in wl.lines_for(contract=contract, rule_name=name) if x.margin_hours < 0
        ]  # ty: ignore[invalid-argument-type]
        out[name] = {"negative_margins": negative}
        out["pass"] &= not negative
    ok, text = wl.check_data(contract=contract)
    print("\n".join(text), flush=True)
    out["stored_run_frame"] = {"pass": ok, "text": text}
    out["pass"] &= ok
    cm.write_check_json(obj=out, path=checks_dir(data_dir=data_dir) / "kl8.json")
    assert out["pass"], "KL8: a negative margin or a failed run-frame check"
    return out


def pure(*, data_dir: Path) -> dict:
    """Run the checks that need no data: KL2's table over the clock-change origins, and the count of changed values."""
    contracts = [contract_datetime(c_idx=contract_index(day=d)) for d in (
        date(2025, 10, 28), date(2025, 11, 8), date(2026, 3, 31), date(2026, 6, 20),
    )]  # fmt: skip
    result = kl2_pure(contracts=contracts)
    cm.write_check_json(obj=result, path=checks_dir(data_dir=data_dir) / "pure.json")
    return result


def collect(*, data_dir: Path, folds: Sequence[str]) -> dict:
    """Merge the results of the checks into `wiring_checks_passed.json`, only if every one passed."""
    import e00725_frozen as fz

    folder = checks_dir(data_dir=data_dir)
    needed = ["pure.json", "kl5_train.json", "kl5_predict.json", "kl8.json"]
    needed += [f"{m}_{ph}.json" for m in folds for ph in ("train", "predict")]
    needed += [f"{m}_kl3r.json" for m in folds]
    record: dict = {"code": fz.code_record(), "checks_hash": cm.checks_hash(), "folds": list(folds)}
    for name in needed:
        path = folder / name
        assert path.exists(), f"{path} is missing: run the check first"
        content = json.loads(path.read_text())
        assert content.get("pass"), f"{name} did not pass"
        assert content.get("checks_hash") == cm.checks_hash(), (
            f"{name} was written by other check code than the code that is running: rerun it"
        )
        record[name] = content
    record["KL1_forecast_present"] = any(
        "KL1_forecast" in record.get(f"{m}_predict.json", {}) for m in folds
    )
    cm.write_json_atomic(obj=record, path=cm.wiring_file(data_dir=data_dir))
    print(f"wrote {cm.wiring_file(data_dir=data_dir)}", flush=True)
    return record


def main() -> int:
    """Parse the command line and run a check command."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pure")
    p_fold = sub.add_parser("fold")
    p_fold.add_argument("month")
    p_fold.add_argument("phase", choices=("train", "predict"))
    p_fold.add_argument("--device", default="cuda")
    p_fold.add_argument("--only", default="")
    p3r = sub.add_parser("kl3r")
    p3r.add_argument("month")
    p3r.add_argument("--primaries", type=int, default=2)
    p5 = sub.add_parser("kl5")
    p5.add_argument("phase", choices=("train", "predict"))
    sub.add_parser("kl8")
    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--folds", default=",".join(cm.ALL_FOLDS))
    args = parser.parse_args()
    if args.command in {"fold", "kl3r"}:
        import torch  # ty: ignore[unresolved-import]

        torch.set_num_threads(1)
    data_dir = ML_EXPERIMENTS_DATA_DIR
    if args.command == "pure":
        print(pure(data_dir=data_dir))
    elif args.command == "fold":
        only = {x for x in args.only.split(",") if x}
        result = run_fold(
            month=args.month, phase=args.phase, device=args.device, data_dir=data_dir, only=only,
        )  # fmt: skip
        return 0 if result["pass"] else 1
    elif args.command == "kl3r":
        result = kl3r_fold(month=args.month, data_dir=data_dir, n_primaries=args.primaries)
        return 0 if result["pass"] else 1
    elif args.command == "kl5":
        kl5(phase=args.phase, data_dir=data_dir)
    elif args.command == "kl8":
        kl8(data_dir=data_dir)
    else:
        collect(data_dir=data_dir, folds=args.folds.split(","))
    return 0


if __name__ == "__main__":
    sys.exit(main())

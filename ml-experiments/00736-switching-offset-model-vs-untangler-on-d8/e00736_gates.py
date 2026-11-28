"""Gates G1 to G4 of experiment 736, and the attribute check of the modules it imports.

G1: the head's stored rows reproduce. G2: the injection is exact, including the +1 index. G3: the
decode of the real record reproduces experiment 734's stored event table (keys, amplitudes,
confidences, and grades). G4: the regenerated overlays reproduce experiment 732's digest. Only G3
decodes, and it decodes the un-injected record once.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Final

import e00736_decode as decode
import e00736_heads as heads
import numpy as np
import polars as pl
from e00736_common import (
    HALF_HOURS_PER_DAY,
    INDEX_OFFSET,
    OVERLAY_DIGEST_732,
    POINT_MW,
    time_log,
)
from e00736_world import (
    D8,
    DESIGN_FULL,
    DESIGN_TRANSFERS_ONLY,
    GRID_LENGTH,
    READ_PARTNERS,
    TOLERANCE_G2,
    Env,
    build_world_power,
    grid_day_of,
    months_and_draws,
    scored_mask,
    transfer_table,
    world_additions,
)

HASATTR: Final[dict[str, tuple[str, ...]]] = {
    "e00653_common": ("GRID_LENGTH",),
    "e00729_banks": ("load_bank",),
    "e00730_banks": ("bank_path",),
    "e00732_cli": ("real_context", "real_base", "real_env"),
    "e00732_overlays": (
        "overlay_segments",
        "overlay_digest",
        "transfer_arrays",
        "reactive_floor",
        "refold",
        "Transfer",
    ),
    "e00732_paths": ("data_root", "root_730"),
    "e00734_decode": ("WorldDecoder", "load_context", "arm_params", "events_frame"),
    "e00734_hmm": ("KERNELS_USED",),
    "e00734_layers": ("apply_additions",),
    "e00734_store": ("event_key_mismatches", "site_blocks"),
}
"""Every attribute taken from an imported module of an earlier experiment."""
HASATTR_CLASS: Final[dict[tuple[str, str], tuple[str, ...]]] = {
    ("e00734_decode", "WorldDecoder"): ("run_arm", "final_events"),
}
"""Every method taken from a class of an imported module, keyed by module and class."""
EXPECTED_TRANSFERS_ALL: Final[int] = 384
EXPECTED_EDGE_ROWS: Final[int] = 768
EXPECTED_TRANSFERS_INSIDE: Final[int] = 288
"""Transfers in the 6 months that lie inside experiment 653's record."""
EXPECTED_RECAL_CALLS: Final[int] = 304
EXPECTED_RECAL_RECALL: Final[float] = 0.490
EXPECTED_MERGED_CALLS: Final[int] = 244
EXPECTED_COUNTED_FOUND: Final[int] = 12
EXPECTED_COUNTED: Final[int] = 21
EXPECTED_UNINJECTED_EVENTS: Final[int] = 131
EXPECTED_UNINJECTED_D8: Final[int] = 28
PLUS_ONE_TOLERANCE_MW: Final[float] = 1e-3
"""Largest gap between D8's panel value and the bank's base reading in megawatts at the +1 index."""


def missing_attributes() -> list[str]:
    """Return every `module.attribute` of `HASATTR` that an imported module lacks.

    Returns:
        The missing names (empty when all exist).
    """
    out: list[str] = []
    importlib.import_module("e00732_paths")  # puts experiments 729 and 730 on `sys.path`
    for module, names in HASATTR.items():
        mod = importlib.import_module(module)
        out += [f"{module}.{a}" for a in names if not hasattr(mod, a)]
    for (module, cls), names in HASATTR_CLASS.items():
        klass = getattr(importlib.import_module(module), cls, None)
        out += [f"{module}.{cls}.{a}" for a in names if not hasattr(klass, a)]
    return out


def g4(*, env: Env) -> dict[str, Any]:
    """Gate G4: the regenerated test overlays reproduce experiment 732's digest and counts.

    Args:
        env: The regenerated bank and overlays.

    Returns:
        The record with `passed`.
    """
    n_transfers = sum(len(o.transfers) for o in env.overlays)
    return {
        "digest": env.digest,
        "digest_ok": env.digest == OVERLAY_DIGEST_732,
        "n_transfers": n_transfers,
        "n_edge_rows": 2 * n_transfers,
        "passed": env.digest == OVERLAY_DIGEST_732
        and n_transfers == EXPECTED_TRANSFERS_ALL
        and 2 * n_transfers == EXPECTED_EDGE_ROWS,
    }


def g1(*, env: Env) -> dict[str, Any]:
    """Gate G1: the head's rows reproduce, under the report's match and under this plan's rule.

    Args:
        env: The regenerated bank and overlays.

    Returns:
        The record with `passed`.
    """
    frozen = heads.frozen_thresholds()
    digests = heads.check_stored_digests()
    recal = heads.reproduce_recalibration(arm=heads.HEAD_ARM)
    out = heads.load_outputs(arm=heads.HEAD_ARM)
    first = {
        i: int(np.asarray(s.dates).astype("datetime64[D]")[0].astype(int))
        for i, s in enumerate(env.segments)
    }
    calls = heads.merged_calls(out=out, frozen=frozen, first_day=first)
    table = transfer_table(segments=env.segments, overlays=env.overlays, noise_d8=env.noise_d8)
    found = heads.found_by_head(truth=table, calls=calls, tolerance=12)
    counted = table["counted"].to_numpy()
    n_found = int((found & counted).sum())
    ok = (
        all(digests.values())
        and int(recal["calls"]) == EXPECTED_RECAL_CALLS
        and round(recal["recall_4_and_more"], 3) == EXPECTED_RECAL_RECALL
        and calls.height == EXPECTED_MERGED_CALLS
        and int(counted.sum()) == EXPECTED_COUNTED
        and n_found == EXPECTED_COUNTED_FOUND
    )
    return {
        "digests_ok": digests,
        "recalibration": recal,
        "merged_calls": calls.height,
        "counted": int(counted.sum()),
        "counted_found": n_found,
        "passed": bool(ok),
    }


def _expected_additions(*, env: Env, draw: int, sites: list[str]) -> dict[str, np.ndarray]:
    """Build a world's transfer-only additions again from `Transfer.mean_levels()` in float64."""
    exp = {s: np.zeros(GRID_LENGTH) for s in sites}
    layout = months_and_draws(segments=env.segments).filter(pl.col("draw") == draw)
    for r in layout.iter_rows(named=True):
        seg, o = env.segments[r["seg"]], env.overlays[r["seg"]]
        day0 = grid_day_of(seg=seg)
        for t in o.transfers:
            lo = day0 * HALF_HOURS_PER_DAY + t.start_half_hour + INDEX_OFFSET
            level = t.mean_levels()
            exp[D8][lo : lo + len(level)] += level * POINT_MW
            if t.mirrored:
                exp[t.partner][lo : lo + len(level)] -= t.jitter * level * POINT_MW
    return exp


def g2(*, env: Env, power: dict[str, np.ndarray], is_mva: dict[str, bool]) -> dict[str, Any]:
    """Gate G2: the injection is exact, the fold is `|p + add|`, and the +1 index holds.

    The check builds each world's additions of transfers alone and compares them half-hour by
    half-hour with `Transfer.mean_levels()` times 0.10379 MW per point in 64-bit floats, to a
    relative tolerance of 1e-6. It checks that D8, an MVA Primary, takes `|p + add|`, that every
    mirrored partner moves by minus the jitter factor times the transfer and is nonzero in every
    half-hour of the transfer, that D8 and every mirrored partner have no addition outside the
    injected half-hours in the transfers-only worlds, that the head's reading is experiment 732's
    refold with the segment's reactive floor, and that D8's panel value equals the bank's base
    reading times 0.10379 MW at the bank half-hour plus 1 (and at no other offset) in the segments
    with neither engine nor solar output.

    Args:
        env: The regenerated bank and overlays.
        power: The real power of every Primary of the panel.
        is_mva: Whether each Primary is an MVA meter.

    Returns:
        The record with `passed`.
    """
    import e00732_overlays as ov  # ty: ignore[unresolved-import]

    sites = list(power)
    worst = {"transfer": 0.0, "mirror": 0.0, "fold": 0.0, "engine_solar": 0.0, "refold": 0.0}
    n_half_hours = 0
    n_zero = 0
    n_mirror_zero = 0
    n_stray = 0
    for draw in range(8):
        only = world_additions(
            segments=env.segments,
            overlays=env.overlays,
            draw=draw,
            design=DESIGN_TRANSFERS_ONLY,
            sites=sites,
        )
        full = world_additions(
            segments=env.segments, overlays=env.overlays, draw=draw, design=DESIGN_FULL, sites=sites
        )
        exp = _expected_additions(env=env, draw=draw, sites=sites)
        inj = exp[D8] != 0.0
        n_half_hours += int(inj.sum())
        n_zero += int((only[D8][inj] == 0.0).sum())
        n_stray += int(np.any(only[D8][~inj] != 0.0))
        worst["transfer"] = max(
            worst["transfer"],
            float(np.max(np.abs(only[D8][inj] - exp[D8][inj]) / np.abs(exp[D8][inj]), initial=0.0)),
        )
        for p in READ_PARTNERS:
            m = exp[p] != 0.0
            n_mirror_zero += int((only[p][m] == 0.0).sum())
            worst["mirror"] = max(
                worst["mirror"],
                float(np.max(np.abs(only[p][m] - exp[p][m]) / np.abs(exp[p][m]), initial=0.0)),
            )
            if np.any(only[p][~m] != 0.0):
                worst["mirror"] = float("inf")
        worst["engine_solar"] = max(
            worst["engine_solar"],
            float(np.max(np.abs((full[D8] - only[D8]) - _arrays(env=env, draw=draw)))),
        )
        world = build_world_power(power=power, additions=full, is_mva=is_mva)
        for s in sites:
            want = np.where(
                (full[s] != 0) & np.isfinite(power[s]),
                np.abs(power[s] + full[s]) if is_mva[s] else power[s] + full[s],
                power[s],
            )
            worst["fold"] = max(
                worst["fold"], float(np.nanmax(np.abs(world[s] - want), initial=0.0))
            )
    for i in months_and_draws(segments=env.segments)["seg"].to_list():
        seg, o = env.segments[i], env.overlays[i]
        q = ov.reactive_floor(seg=seg)
        want = np.where(
            np.isfinite(seg.r), ov.refold(x=np.nan_to_num(seg.x + o.t, nan=0.0), q=q), np.nan
        )
        got = np.asarray(o.seg.r)
        ok = np.isfinite(want) & np.isfinite(got)
        worst["refold"] = max(
            worst["refold"], float(np.max(np.abs(want[ok] - got[ok]), initial=0.0))
        )
    plus_one = _plus_one(env=env, d8=power[D8])
    table = transfer_table(segments=env.segments, overlays=env.overlays, noise_d8=env.noise_d8)
    ok = (
        is_mva[D8]
        and worst["transfer"] <= TOLERANCE_G2
        and worst["mirror"] <= TOLERANCE_G2
        and worst["engine_solar"] <= 1e-12
        and worst["fold"] <= 1e-12
        and worst["refold"] <= 1e-4
        and n_zero == 0
        and n_mirror_zero == 0
        and n_stray == 0
        and plus_one["passed"]
        and table.height == EXPECTED_TRANSFERS_INSIDE
    )
    return {
        "worst_relative_or_absolute_error": worst,
        "injected_half_hours": n_half_hours,
        "injected_half_hours_with_zero_addition": n_zero,
        "mirror_half_hours_with_zero_addition": n_mirror_zero,
        "worlds_with_a_stray_addition_at_d8": n_stray,
        "n_transfers": table.height,
        "n_start_on_scored_half_hour": int(table["start_scored"].sum()),
        "n_end_on_scored_half_hour": int(table["end_scored"].sum()),
        "d8_is_mva": bool(is_mva[D8]),
        "plus_one_index": plus_one,
        "passed": bool(ok),
    }


def _arrays(*, env: Env, draw: int) -> np.ndarray:
    """Return the engine and solar output, in megawatts, of a draw's segments on the panel."""
    out = np.zeros(GRID_LENGTH)
    layout = months_and_draws(segments=env.segments).filter(pl.col("draw") == draw)
    for r in layout.iter_rows(named=True):
        seg = env.segments[r["seg"]]
        idx = grid_day_of(seg=seg) * HALF_HOURS_PER_DAY + np.arange(seg.x.size) + INDEX_OFFSET
        keep = (idx >= 0) & (idx < GRID_LENGTH)
        both = np.asarray(seg.engine, dtype=np.float64).reshape(-1) + np.asarray(
            seg.solar, dtype=np.float64
        ).reshape(-1)
        out[idx[keep]] -= both[keep] * POINT_MW
    return out


def _plus_one(*, env: Env, d8: np.ndarray) -> dict[str, Any]:
    """Check the +1 index on the valid base half-hours of segments with no engine and no solar."""
    gaps = {-1: 0.0, 0: 0.0, 1: 0.0, 2: 0.0}
    n_seg = 0
    for i in months_and_draws(segments=env.segments)["seg"].to_list():
        seg = env.segments[i]
        if (np.asarray(seg.engine) != 0).any() or (np.asarray(seg.solar) != 0).any():
            continue
        n_seg += 1
        base = np.asarray(seg.base, dtype=np.float64).reshape(-1) * POINT_MW
        use = scored_mask(seg=seg)
        for off in list(gaps):
            idx = grid_day_of(seg=seg) * HALF_HOURS_PER_DAY + np.arange(seg.x.size) + off
            ok = use & (idx >= 0) & (idx < GRID_LENGTH)
            ok &= np.isfinite(d8[np.clip(idx, 0, GRID_LENGTH - 1)])
            gaps[off] = max(gaps[off], float(np.max(np.abs(d8[idx[ok]] - base[ok]), initial=0.0)))
    return {
        "segments_without_engine_or_solar": n_seg,
        "max_abs_gap_mw_by_offset": {str(k): v for k, v in gaps.items()},
        "passed": bool(
            n_seg > 0
            and gaps[1] <= PLUS_ONE_TOLERANCE_MW
            and all(gaps[k] > 10 * PLUS_ONE_TOLERANCE_MW for k in (-1, 0, 2))
        ),
    }


def g3() -> dict[str, Any]:
    """Gate G3: the un-injected decode reproduces experiment 734's stored H-nbr event table.

    Returns:
        The record with `passed` (this decodes the real record once with H-nbr).
    """
    import e00734_store as store  # ty: ignore[unresolved-import]

    start = time.time()
    mine_frame = decode.uninjected_frame(arm="H-nbr")
    stored = decode.stored_uninjected(arm="H-nbr")
    cmp = store.event_key_mismatches(mine=decode.event_keys(frame=mine_frame), stored=stored)
    values = decode.compare_event_values(mine=mine_frame, stored=stored)
    n_d8 = int((stored["primary"] == D8).sum())
    n_primaries = stored["primary"].n_unique()
    time_log(stage="gate3", task="uninjected_H-nbr", start=start)
    return {
        "n_mine": cmp["n_mine"],
        "n_stored": cmp["n_stored"],
        "n_primaries_in_stored_table": n_primaries,
        "n_stored_at_d8": n_d8,
        "only_mine": cmp["only_mine"],
        "only_stored": cmp["only_stored"],
        "values": values,
        "passed": bool(
            cmp["equal"]
            and values["equal"]
            and cmp["n_stored"] == EXPECTED_UNINJECTED_EVENTS
            and n_d8 == EXPECTED_UNINJECTED_D8
        ),
    }

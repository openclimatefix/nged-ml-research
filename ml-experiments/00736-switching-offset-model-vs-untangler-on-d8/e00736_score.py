"""Scoring of experiment 736: the match, paired table, sign test, placebo, and report.

Both detectors are scored on the same injected transfers. All times are on the bank's dense axis of
the transfer's segment: a call of the offset model at panel index `p` sits at `p - 1 - 48 * day0`
of its segment, where `day0` is the segment's first dense day on the panel.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Any, Final

import e00736_decode as decode
import e00736_heads as heads
import numpy as np
import polars as pl
from e00736_common import (
    ARMS,
    HALF_HOURS_PER_DAY,
    INDEX_OFFSET,
    N_DRAWS,
    OUT,
    PLACEBO_SHIFT_HH,
    TOLERANCE_HH,
    atomic_json,
)
from e00736_match import (
    PORTING_THRESHOLD,
    false_calls,
    group_interval,
    match_by_segment,
    months_without_discordant,
    paired_counts,
    sign_test_p,
    v1_passes,
    verdict,
)
from e00736_world import (
    D8,
    grid_day_of,
    months_and_draws,
    scored_mask,
    transfer_table,
    with_gaps,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

HEAD_ARM: Final[str] = "U-noBSP"
HEAD_REFERENCE: Final[str] = "U*"
OFFSET_ARM: Final[str] = "H-nbr"
OFFSET_REFERENCE: Final[str] = "H-own"
POST_HOC_TOLERANCE_HH: Final[int] = 24
"""The 12-hour reading of experiment 734's tolerance, reported as a post hoc row."""
REFERENCE_TOLERANCE_HH: Final[int] = 1
"""The recalibration report's tolerance for the head."""
SCORES_DIR: Final[str] = "scores"


def offset_calls(
    *, events: pl.DataFrame, draw: int, segments: Sequence[Any], layout: pl.DataFrame
) -> pl.DataFrame:
    """Return D8's decoded events of one world on each segment's dense axis.

    An event belongs to the segment whose dense days hold its start. The sign is the sign of the
    event's `amp_mw` (positive when D8 gains load). `end` is -1 for an event with no end.

    Args:
        events: The world's decoded events (all Primaries).
        draw: The world's draw index.
        segments: The test bank's segments.
        layout: The months and draws of the segments inside the record.

    Returns:
        One row per D8 event with `seg`, `time`, `sign`, `end`, and `amp_mw`.
    """
    d8 = events.filter(pl.col("primary") == D8)
    rows: list[dict[str, Any]] = []
    for r in layout.filter(pl.col("draw") == draw).iter_rows(named=True):
        seg = segments[r["seg"]]
        shift = grid_day_of(seg=seg) * HALF_HOURS_PER_DAY + INDEX_OFFSET
        n = int(seg.x.size)
        for e in d8.iter_rows(named=True):
            t = int(e["start"]) - shift
            if 0 <= t < n:
                rows.append(
                    {
                        "seg": r["seg"],
                        "time": t,
                        "sign": 1 if e["amp_mw"] >= 0 else -1,
                        "end": -1 if e["end"] < 0 else int(e["end"]) - shift,
                        "amp_mw": float(e["amp_mw"]),
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "seg": pl.Int64,
            "time": pl.Int64,
            "sign": pl.Int64,
            "end": pl.Int64,
            "amp_mw": pl.Float64,
        },
    )


def offset_found(
    *, truth: pl.DataFrame, calls: pl.DataFrame, tolerance: int, time_col: str = "start_hh"
) -> tuple[np.ndarray, np.ndarray]:
    """Return whether the offset model found each transfer's start, and found both ends.

    A transfer is found when a call of its sign starts within the tolerance of the injected start.
    It is found with both ends when the matched event also ends within the tolerance of the injected
    end.

    Args:
        truth: The transfer table.
        calls: The offset model's calls on the dense axis.
        tolerance: The tolerance in half-hours.
        time_col: The truth column of the start edge.

    Returns:
        `(found, found_both_ends)`, each in the order of `truth`.
    """
    matched = match_by_segment(truth=truth, calls=calls, tolerance=tolerance, time_col=time_col)
    found = matched >= 0
    if len(calls) == 0:
        return found, found
    ends = calls["end"].to_numpy()
    got = np.where(found, ends[np.maximum(matched, 0)], -1)
    both = found & (got >= 0) & (np.abs(got - truth["end_hh"].to_numpy()) <= tolerance)
    return found, both


def with_placebo(
    *, table: pl.DataFrame, segments: Sequence[Any], d8_power: np.ndarray
) -> pl.DataFrame:
    """Add the placebo's moved start and whether the placebo keeps the transfer.

    The placebo moves each injected start 7 days later. It drops a start whose moved time lands
    inside a transfer of the segment, within 6 hours of any real start or end edge of the segment
    (a dropped transfer's edges count), off a scored half-hour, outside the segment's own days (so
    across a month end), or in a block where D8 has no reading.

    Args:
        table: The transfer table.
        segments: The test bank's segments.
        d8_power: D8's power on the panel, NaN where missing.

    Returns:
        The table with `placebo_hh` and `placebo_ok`.
    """
    moved = table["start_hh"].to_numpy() + PLACEBO_SHIFT_HH
    ok = np.zeros(table.height, dtype=bool)
    seg_of = table["seg"].to_numpy()
    start, end = table["start_hh"].to_numpy(), table["end_hh"].to_numpy()
    panel = table["start_panel"].to_numpy() + PLACEBO_SHIFT_HH
    for i in range(table.height):
        seg = segments[int(seg_of[i])]
        n = int(seg.x.size)
        m = int(moved[i])
        inside_days = HALF_HOURS_PER_DAY <= m < n - HALF_HOURS_PER_DAY
        if not inside_days or not bool(scored_mask(seg=seg)[m]):
            continue
        same = np.flatnonzero(seg_of == seg_of[i])
        if any(
            (start[j] <= m <= end[j])
            or abs(m - start[j]) <= TOLERANCE_HH
            or abs(m - end[j]) <= TOLERANCE_HH
            for j in same
        ):
            continue
        p = int(panel[i])
        blk = (p // 12) * 12
        if not (
            blk >= 0 and blk + 12 <= len(d8_power) and np.isfinite(d8_power[blk : blk + 12]).any()
        ):
            continue
        ok[i] = True
    return table.with_columns(placebo_hh=pl.Series(moved), placebo_ok=pl.Series(ok))


def false_rate(
    *, calls: pl.DataFrame, truth: pl.DataFrame, segments: Sequence[Any], layout: pl.DataFrame
) -> dict[str, float]:
    """Count a detector's false edge calls on scored base days, and give the rate per 100 days.

    A call is false when it matches no injected start (same sign) or end (opposite sign) within
    6 hours. Only calls on a base day of the segment count, and the day count is the number of
    base days of the segments in `layout`.

    Args:
        calls: The detector's calls on the dense axis, for every segment of `layout`.
        truth: The transfer table (every injected transfer, gaps included).
        segments: The test bank's segments.
        layout: The segments to score.

    Returns:
        `false`, `calls_on_base_days`, `base_days`, and `per_100_days`.
    """
    n_false = n_calls = days = 0
    for seg_i in layout["seg"].to_list():
        seg = segments[seg_i]
        base = np.asarray(seg.cal_index) >= 0
        days += int(base.sum())
        c = calls.filter(pl.col("seg") == seg_i)
        t = c["time"].to_numpy()
        on_base = (
            (t >= 0) & (t < seg.x.size) & base[np.clip(t // HALF_HOURS_PER_DAY, 0, len(base) - 1)]
        )
        c = c.filter(pl.Series(on_base))
        tr = truth.filter(pl.col("seg") == seg_i)
        flags = false_calls(
            call_times=c["time"].to_numpy(),
            call_signs=c["sign"].to_numpy(),
            start_times=tr["start_hh"].to_numpy(),
            start_signs=tr["sign"].to_numpy(),
            end_times=tr["end_hh"].to_numpy(),
            end_signs=-tr["sign"].to_numpy(),
        )
        n_false += int(flags.sum())
        n_calls += c.height
    return {
        "false": float(n_false),
        "calls_on_base_days": float(n_calls),
        "base_days": float(days),
        "per_100_days": 100.0 * n_false / days if days else float("nan"),
    }


def share(*, found: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Return the count and share found among the transfers of a mask.

    Args:
        found: Whether each transfer was found.
        mask: Which transfers to count.

    Returns:
        `found`, `n`, and `share` (NaN when the mask is empty).
    """
    n = int(mask.sum())
    k = int((found & mask).sum())
    return {"found": float(k), "n": float(n), "share": k / n if n else float("nan")}


def score_all(
    *,
    table: pl.DataFrame,
    head_calls: dict[str, pl.DataFrame],
    offset: dict[str, pl.DataFrame],
    segments: Sequence[Any],
    layout: pl.DataFrame,
    gates_passed: bool,
) -> dict[str, Any]:
    """Score every row of the plan from the transfer table and the detectors' call tables.

    Args:
        table: The transfer table with `gap`, `placebo_hh`, and `placebo_ok`. Its gap rows stay in
            the truth of the false-call count, so a call near a dropped transfer's edge is not
            false.
        head_calls: Head arm to its merged calls on the dense axis.
        offset: Offset-model arm to its calls on the dense axis (all worlds together).
        segments: The test bank's segments.
        layout: The segments inside the record, with month and draw.
        gates_passed: Whether gates G1 to G4 passed.

    Returns:
        The scores, JSON-serialisable, and `table` extended with each detector's `found_*` columns
        is returned under the key `table`.
    """
    scored = table.filter(~pl.col("gap"))
    cols: dict[str, np.ndarray] = {}
    for arm, calls in head_calls.items():
        for label, tol in (
            ("", TOLERANCE_HH),
            ("_1hh", REFERENCE_TOLERANCE_HH),
            ("_12h", POST_HOC_TOLERANCE_HH),
        ):
            cols[f"found_{arm}{label}"] = heads.found_by_head(
                truth=scored, calls=calls, tolerance=tol
            )
        cols[f"found_{arm}_end"] = heads.found_by_head(
            truth=scored, calls=calls, tolerance=TOLERANCE_HH, time_col="end_hh", end=True
        )
        cols[f"found_{arm}_both"] = cols[f"found_{arm}"] & cols[f"found_{arm}_end"]
        cols[f"placebo_{arm}"] = heads.found_by_head(
            truth=scored, calls=calls, tolerance=TOLERANCE_HH, time_col="placebo_hh"
        )
    for arm, calls in offset.items():
        for label, tol in (("", TOLERANCE_HH), ("_12h", POST_HOC_TOLERANCE_HH)):
            f, both = offset_found(truth=scored, calls=calls, tolerance=tol)
            cols[f"found_{arm}{label}"] = f
            if label == "":
                cols[f"found_{arm}_both"] = both
        cols[f"placebo_{arm}"] = offset_found(
            truth=scored, calls=calls, tolerance=TOLERANCE_HH, time_col="placebo_hh"
        )[0]
    wide = scored.with_columns(**{k: pl.Series(v) for k, v in cols.items()})
    out = _summaries(
        wide=wide,
        table=table,
        head_calls=head_calls,
        offset=offset,
        segments=segments,
        layout=layout,
        gates_passed=gates_passed,
    )
    out["table"] = wide
    return out


def _summaries(
    *,
    wide: pl.DataFrame,
    table: pl.DataFrame,
    head_calls: dict[str, pl.DataFrame],
    offset: dict[str, pl.DataFrame],
    segments: Sequence[Any],
    layout: pl.DataFrame,
    gates_passed: bool,
) -> dict[str, Any]:
    counted = wide["counted"].to_numpy()
    big = wide["big"].to_numpy()
    f = {c: wide[c].to_numpy() for c in wide.columns if c.startswith(("found_", "placebo_"))}
    pc = paired_counts(
        found_head=f[f"found_{HEAD_ARM}"][counted], found_offset=f[f"found_{OFFSET_ARM}"][counted]
    )
    months = wide["month_idx"].to_numpy()[counted]
    draws = wide["draw"].to_numpy()[counted]
    all_months = np.unique(layout["month_idx"].to_numpy())
    fh, fo = f[f"found_{HEAD_ARM}"][counted], f[f"found_{OFFSET_ARM}"][counted]
    iv_world = group_interval(found_head=fh, found_offset=fo, groups=draws)
    iv_month = group_interval(found_head=fh, found_offset=fo, groups=months, seed=737)
    empty = months_without_discordant(
        found_head=fh, found_offset=fo, months=months, all_months=all_months
    )
    fr = {
        arm: false_rate(calls=c, truth=table, segments=segments, layout=layout)
        for arm, c in {HEAD_ARM: head_calls[HEAD_ARM], OFFSET_ARM: offset[OFFSET_ARM]}.items()
    }
    v1 = v1_passes(
        offset_false_per_100=fr[OFFSET_ARM]["per_100_days"],
        head_false_per_100=fr[HEAD_ARM]["per_100_days"],
    )
    decision = verdict(
        gates_passed=gates_passed,
        counts=pc,
        interval_world=iv_world,
        interval_month=iv_month,
        empty_months=empty,
        v1_passes=v1,
    )
    bands = {}
    for arm in (*head_calls, *offset):
        key = f"found_{arm}"
        bands[arm] = {
            band: share(found=f[key], mask=big & (wide["band"].to_numpy() == band))
            for band in wide["band"].unique().sort().to_list()
        }
    mirrored = wide["mirrored"].to_numpy()
    plain = ~wide["has_arrays"].to_numpy()
    paired_secondary = {}
    for name, mask in (
        ("counted_mirrored", counted & mirrored),
        ("counted_not_mirrored", counted & ~mirrored),
        ("counted_no_engine_or_solar", counted & plain),
        ("counted_with_engine_or_solar", counted & ~plain),
    ):
        paired_secondary[name] = paired_counts(
            found_head=f[f"found_{HEAD_ARM}"][mask], found_offset=f[f"found_{OFFSET_ARM}"][mask]
        )._asdict()
    p5 = {
        name: {
            "n": int(mask.sum()),
            "H-nbr_found": int((f[f"found_{OFFSET_ARM}"] & mask).sum()),
            "H-own_found": int((f[f"found_{OFFSET_REFERENCE}"] & mask).sum()),
        }
        for name, mask in (("mirrored", counted & mirrored), ("not_mirrored", counted & ~mirrored))
    }
    both = {
        arm: share(found=f[f"found_{arm}_both"], mask=counted) for arm in (*head_calls, *offset)
    }
    keep = wide["placebo_ok"].to_numpy()
    placebo = {
        arm: {
            "counted": share(found=f[f"placebo_{arm}"], mask=counted & keep),
            "big": share(found=f[f"placebo_{arm}"], mask=big & keep),
        }
        for arm in (*head_calls, *offset)
    }
    post_hoc = paired_counts(
        found_head=f[f"found_{HEAD_ARM}_12h"][counted],
        found_offset=f[f"found_{OFFSET_ARM}_12h"][counted],
    )._asdict()
    discordant = (
        wide.filter(pl.col("counted"))
        .with_columns(only_head=pl.Series(fh & ~fo), only_offset=pl.Series(~fh & fo))
        .group_by("month_idx")
        .agg(
            only_head=pl.col("only_head").sum(), only_offset=pl.col("only_offset").sum(), n=pl.len()
        )
        .sort("month_idx")
        .to_dicts()
    )
    return {
        "verdict": decision,
        "gates_passed": gates_passed,
        "counted_cell": {
            "n": pc.n,
            "paired": pc._asdict(),
            "c1": pc.c1,
            "sign_test_p": sign_test_p(b=pc.only_head, c=pc.only_offset),
            "interval_world": iv_world,
            "interval_month": iv_month,
            "months_without_discordant": empty,
            "porting_threshold": PORTING_THRESHOLD,
        },
        "veto_v1": {"passes": v1, "false_calls": fr},
        "shares_by_band_big": bands,
        "both_ends_counted": both,
        "paired_secondary": paired_secondary,
        "p5_mirrored_split": p5,
        "placebo": placebo,
        "placebo_starts_kept": int(keep.sum()),
        "placebo_starts_scored": int(keep.size),
        "post_hoc_12_hour_tolerance_counted": post_hoc,
        "discordant_by_month": discordant,
        "n_transfers": wide.height,
        "n_dropped_for_gaps": 0,
    }


def run_score() -> dict[str, Any]:
    """Score the experiment from stored tables and decoded events, and write the score files.

    Returns:
        The scores (without the per-transfer table, which goes to `scores/transfer_scores.parquet`).
    """
    import e00734_decode as dec  # ty: ignore[unresolved-import]
    from e00736_world import load_env

    env = load_env()
    ctx = dec.load_context()
    d8 = ctx.rig.panel.power[D8]
    layout = months_and_draws(segments=env.segments)
    table = transfer_table(segments=env.segments, overlays=env.overlays, noise_d8=env.noise_d8)
    table = with_placebo(
        table=with_gaps(table=table, d8_power=d8), segments=env.segments, d8_power=d8
    )
    first_day = {
        i: int(np.asarray(s.dates).astype("datetime64[D]")[0].astype(int))
        for i, s in enumerate(env.segments)
    }
    frozen = heads.frozen_thresholds()
    head_calls = {
        arm: heads.merged_calls(out=heads.load_outputs(arm=arm), frozen=frozen, first_day=first_day)
        for arm in heads.ARMS
    }
    offset = {
        arm: pl.concat(
            [
                offset_calls(
                    events=decode.read_events(draw=k, arm=arm),
                    draw=k,
                    segments=env.segments,
                    layout=layout,
                )
                for k in range(N_DRAWS)
            ]
        )
        for arm in ARMS
    }
    gates = json.loads((OUT / "gates.json").read_text())
    passed = all(gates.get(g, {}).get("passed") is True for g in ("G1", "G2", "G3", "G4"))
    result = score_all(
        table=table,
        head_calls=head_calls,
        offset=offset,
        segments=env.segments,
        layout=layout,
        gates_passed=passed,
    )
    wide = result.pop("table")
    result["n_dropped_for_gaps"] = table.height - wide.height
    result["out_folder"] = str(OUT)
    (OUT / SCORES_DIR).mkdir(parents=True, exist_ok=True)
    wide.write_parquet(OUT / SCORES_DIR / "transfer_scores.parquet")
    atomic_json(obj=result, path=OUT / SCORES_DIR / "scores.json")
    return result


def report() -> None:
    """Print the report from the stored scores, and store the output as `report_output.txt`."""
    scores = json.loads((OUT / SCORES_DIR / "scores.json").read_text())
    design = json.loads((OUT / "world_design.json").read_text())
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"world design that ran: {design['design']}")
        print(f"verdict: {scores['verdict']} (gates passed: {scores['gates_passed']})")
        cell = scores["counted_cell"]
        pc = cell["paired"]
        print(
            f"counted cell: {cell['n']} transfers of 24 to 96 hours and 4 step noise units or more "
            f"({scores['n_dropped_for_gaps']} dropped for gaps in all bands)"
        )
        print(
            f"paired counts: both {pc['both']}, only {HEAD_ARM} {pc['only_head']}, "
            f"only {OFFSET_ARM} {pc['only_offset']}, neither {pc['neither']}"
        )
        print(f"C1 = {cell['c1']:.3f}, exact sign test p = {cell['sign_test_p']:.4f}")
        print(
            f"interval over worlds: {cell['interval_world']}; over months: {cell['interval_month']}"
        )
        print(f"months without a discordant transfer: {cell['months_without_discordant']}")
        print(f"veto V1 passes: {scores['veto_v1']['passes']}")
        print(json.dumps(scores, indent=1, default=float))
    text = buf.getvalue()
    print(text, end="")
    (OUT / "report_output.txt").write_text(text)


__all__ = ["report", "run_score", "score_all"]

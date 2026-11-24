# ruff: noqa: E501, PERF401, PLR0915
"""Experiment 723, job W2: the concordance of the reading's fast change with a sky sensor, in injected worlds.

Run it only through `run.sh --go e00723_w2.py`, after the Opus code review, after job W0, and after
the manager has replied to the message "fits starting".

**The farm sensor binds.** Each qualifying pair of a host and a metered farm outside the host's
polygon is injected at levels 1.0 and 2.0 into the host's signed series, folded with a reactive
floor of 2 points. D10 folded, D15 folded, and the six signed-MW Primaries un-injected complete
the worlds. The injected farm never enters its own world's sensor. The meter sensor and the CAMS
sensor are scored with the same statistics and carry no verdict.

**The order is fixed.** The verdict file is written first. Every read-out that reads the
population of a bar (the pooled area without D14, the powered sub-pool, the meter and CAMS areas,
the unmasked statistics, the fold ratios, the dark-hour closure slopes, and the two D15 fits) runs
after the verdict file is written, and changes no verdict. A read-out that did not withhold may
not later be cited as evidence for a hint.

Outputs: `w2_results.csv`, `w2_verdict.json`, `w2_readouts.csv`, `w2_d15_fits.csv`,
`run_log_w2.json`. Stdout prints counts, shares, areas, and intervals only.
"""

import time
from pathlib import Path
from typing import Any, Final

import e00722_common as common722  # ty: ignore[unresolved-import]
import e00722_data as data  # ty: ignore[unresolved-import]
import e00722_sky as sky  # ty: ignore[unresolved-import]
import e00723_common as common
import e00723_core as core
import e00723_guard as guard
import e00723_score as score
import e00723_world as world
import numpy as np
import polars as pl

FOLDER: Final[Path] = guard.FOLDER
"""This experiment's code folder."""

OUTPUTS: Final[tuple[str, ...]] = (
    "w2_results.csv", common.W2_VERDICT_NAME, "w2_readouts.csv", "w2_d15_fits.csv",
)  # fmt: skip
"""Every file the job writes outside `PRIVATE/`; a run deletes them first so that a crash leaves none stale."""


def read_membership(*, out_dir: Path = guard.OUT_DIR) -> dict[str, list[str]]:
    """Return each host's sensor farms from job W0's membership table."""
    frame = pl.read_csv(out_dir / common.W0_MEMBERSHIP_NAME)
    out: dict[str, list[str]] = {}
    for host, farm in zip(frame["host"].to_list(), frame["farm"].to_list(), strict=True):
        out.setdefault(host, []).append(farm)
    return out


def read_windows(
    *, mask_available: bool, private_dir: Path = guard.PRIVATE_DIR
) -> dict[str, list[tuple[int, int]]]:
    """Return each host's switching-log windows from job W0, or none when the logs gave no mask."""
    if not mask_available:
        return {}
    frame = pl.read_parquet(private_dir / common.W0_WINDOWS_NAME)
    out: dict[str, list[tuple[int, int]]] = {}
    for label, a, b in zip(
        frame["label"].to_list(),
        frame["first_row"].to_list(),
        frame["last_row"].to_list(),
        strict=True,
    ):
        out.setdefault(label, []).append((a, b))
    return out


def result_row(
    *,
    measure: str,
    scope: str,
    outcome: str,
    stats: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one row of a results table, with R1's `n_rows` (usable pairs) and `n_days`."""
    return {
        "measure": measure,
        "scope": scope,
        "outcome": outcome,
        "value": stats.get("area", stats.get("share", float("nan"))),
        "se": stats.get("se", float("nan")),
        "lo": stats.get("lo", float("nan")),
        "hi": stats.get("hi", float("nan")),
        "n_stretches": int(stats.get("n_export", 0) + stats.get("n_import", 0))
        if "n_export" in stats
        else int(stats.get("n", 0)),
        "n_rows": int(stats.get("n_pairs", 0)),
        "n_days": int(stats.get("n_days", 0)),
        **(extra or {}),
    }


def evaluate_bars(
    *, scores: list[score.WorldScore], weights: np.ndarray, d15_binds: bool
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Return the outcome of each bar, the results rows, and the details the verdict file keeps.

    The bars: the pooled area (farm sensor), the placebo (the sensor shifted by 28 days), false
    export evidence at un-injected D5 and D10, the transfer check at un-injected D9, D11, D14, and
    D16, D15's dead band, recall on D10's real export stretches, and calibration (void by construction:
    the likelihood ratio has no definition on the plan page).
    """
    rows: list[dict[str, Any]] = []
    inj = score.tagged(scores=scores, sensor="farm", roles=("injected",))
    pooled = score.area_of(rows=inj, weights=weights)
    level_1 = score.area_of(rows=inj.filter(pl.col("level") == 1.0), weights=None)
    level_2 = score.area_of(rows=inj.filter(pl.col("level") == 2.0), weights=None)
    level_1_readable = (
        level_1["n_export"] >= core.MIN_STRETCHES and level_1["n_import"] >= core.MIN_STRETCHES
    )
    # RULED: the placebo is read on exactly the area's stretches and pairs (the farm sensor's
    # scorable stretches, the real index change), so it has no void state of its own.
    shifted_rows = score.tagged(
        scores=scores, sensor="shifted", roles=("injected",), population_sensor="farm"
    )
    shifted = score.area_of(rows=shifted_rows, weights=None)
    if shifted_rows.height != inj.height:
        msg = "the placebo must be read on the same stretches as the area"
        raise RuntimeError(msg)
    outcomes: dict[str, str] = {}
    outcomes["area"] = core.area_bar(
        point=pooled["area"],
        lo=pooled["lo"],
        point_level_1=level_1["area"] if level_1_readable else float("nan"),
    )
    outcomes["placebo"] = core.placebo_bar(real_area=pooled["area"], shifted_area=shifted["area"])
    if outcomes["placebo"] == "void" and np.isfinite(pooled["area"]):
        msg = "the placebo has no void state: its population is the area's"
        raise RuntimeError(msg)
    rows += [
        result_row(
            measure="pooled_area_farm_sensor",
            scope="all pairs, both levels",
            outcome=outcomes["area"],
            stats=pooled,
        ),
        result_row(
            measure="area_level_1.0",
            scope="all pairs",
            outcome="report",
            stats=level_1,
            extra={"readable": level_1_readable},
        ),
        result_row(measure="area_level_2.0", scope="all pairs", outcome="report", stats=level_2),
        result_row(
            measure="area_shifted_28_days",
            scope="all pairs, both levels",
            outcome=outcomes["placebo"],
            stats=shifted,
            extra={
                "gap": pooled["area"] - shifted["area"],
                "n_stretches_without_a_shifted_pair": int((shifted_rows["n_pairs"] == 0).sum()),
            },
        ),
    ]
    # False export evidence at un-injected D5 and D10 (pooled over the two hosts).
    un = score.tagged(scores=scores, sensor="farm", roles=("uninjected",))
    imp = un.filter(
        pl.col("host").is_in(score.UNINJECTED_FALSE_EXPORT_HOSTS) & ~pl.col("is_export")
    )
    ev = score.evidence_counts(rows=imp)
    outcomes["false_export"] = core.false_export_bar(
        n_export_evidence=ev["n_export_evidence"], n_scorable_import=ev["n"]
    )
    rows.append(
        result_row(
            measure="false_export_evidence",
            scope="un-injected D5 and D10, true import stretches",
            outcome=outcomes["false_export"],
            stats={
                **ev,
                "share": ev["n_export_evidence"] / ev["n"] if ev["n"] else float("nan"),
                "n_days": imp["day"].n_unique() if imp.height else 0,
                "n_pairs": int(imp["n_pairs"].sum()) if imp.height else 0,
            },
            extra={"allowance": core.floor_allowance(n=ev["n"])},
        )
    )
    # Each host beside the pooled bar, with no bar of its own (RULED: the pooled bar stays).
    for host in score.UNINJECTED_FALSE_EXPORT_HOSTS:
        host_ev = score.evidence_counts(rows=imp.filter(pl.col("host") == host))
        rows.append(
            result_row(
                measure="false_export_evidence_by_host",
                scope=f"un-injected {host}, true import stretches",
                outcome="report",
                stats={
                    **host_ev,
                    "share": host_ev["n_export_evidence"] / host_ev["n"]
                    if host_ev["n"]
                    else float("nan"),
                },
                extra={"allowance_if_alone": core.floor_allowance(n=host_ev["n"])},
            )
        )
    # Transfer check at un-injected D9, D11, D14, D16.
    tr = un.filter(pl.col("host").is_in(score.TRANSFER_HOSTS) & ~pl.col("is_export"))
    hit = (tr["c"].to_numpy() < -core.DEAD_BAND) if tr.height else np.array([], dtype=bool)
    share, se = core.bootstrap_share(
        hit=hit, day=tr["day"].to_numpy() if tr.height else np.array([], dtype=int), weights=weights
    )
    outcomes["transfer"] = core.transfer_bar(
        share_import_evidence=share, n_stretches=tr.height, se=se
    )
    rows.append(
        result_row(
            measure="transfer_import_evidence",
            scope="un-injected D9, D11, D14, D16, true import stretches",
            outcome=outcomes["transfer"],
            stats={
                "share": share,
                "se": se,
                "n": tr.height,
                "n_days": tr["day"].n_unique() if tr.height else 0,
                "n_pairs": int(tr["n_pairs"].sum()) if tr.height else 0,
            },
        )
    )
    # D15's dead band (every stretch is the denominator).
    dp = score.dead_band_population(scores=scores)
    outcomes["d15_dead_band"] = core.dead_band_bar(
        n_no_evidence=dp["n_no_evidence"], n_stretches=dp["n_stretches"], binds=d15_binds
    )
    rows.append(
        result_row(
            measure="d15_dead_band",
            scope="D15 folded, every post-bounce stretch on a one-stretch day",
            outcome=outcomes["d15_dead_band"],
            stats={
                "share": dp["n_no_evidence"] / dp["n_stretches"]
                if dp["n_stretches"]
                else float("nan"),
                "n": dp["n_stretches"],
            },
            extra={
                "share_among_scorable": dp["n_scorable_no_evidence"] / dp["n_scorable"]
                if dp["n_scorable"]
                else float("nan"),
                "n_scorable": dp["n_scorable"],
                "binds": d15_binds,
            },
        )
    )
    # Recall on D10's real export stretches.
    d10 = score.tagged(scores=scores, sensor="farm", roles=("d10_folded",)).filter(
        pl.col("is_export")
    )
    n_hit = int((d10["c"].to_numpy() > core.DEAD_BAND).sum()) if d10.height else 0
    outcomes["recall"] = core.recall_bar(n_export_evidence=n_hit, n_scorable_export=d10.height)
    rows.append(
        result_row(
            measure="recall_d10_export",
            scope="D10 folded, true export stretches",
            outcome=outcomes["recall"],
            stats={
                "share": n_hit / d10.height if d10.height else float("nan"),
                "n": d10.height,
                "n_days": d10["day"].n_unique() if d10.height else 0,
                "n_pairs": int(d10["n_pairs"].sum()) if d10.height else 0,
            },
            extra={
                "note": "no bar on real export was read" if outcomes["recall"] == "void" else ""
            },
        )
    )
    outcomes["calibration"] = "void"
    rows.append(
        result_row(
            measure="calibration_likelihood_ratio",
            scope="not defined",
            outcome="void",
            stats={},
            extra={
                "note": "void by construction: the plan page defines no likelihood ratio, so no run reads this bar; neutral"
            },
        )
    )
    details = {
        "pooled": pooled, "level_1": level_1, "level_2": level_2, "shifted": shifted,
        "level_1_readable": level_1_readable,
    }  # fmt: skip
    return outcomes, rows, details


def time_closure_draw(
    *,
    ctx: common.Context,
    windows: dict[str, list[tuple[int, int]]],
    weights: np.ndarray,
    n: int = 10,
) -> float:
    """Return the core-seconds per bootstrap draw of one closure fit (D10's dark hours), timed on `n` draws.

    The figure includes the fit's setup, so it over-states a draw; the timing gate multiplies it by
    the 1,000 draws and the 7 closure fits of the read-outs.
    """
    quiet = world.core722.quiet_mask(engine_change=ctx.eng.change, calendar_ok=ctx.eng.has_row)
    day = sky.day_of(n=core.N_GRID) - ctx.start_row // core.HALF_HOURS_PER_DAY
    rows = world.closure_rows(
        zenith=ctx.sky_of["D10"].zenith,
        masked=score.windows_mask(windows=windows, host="D10"),
        start=ctx.start_row,
        kind="dark",
    )
    t0 = time.process_time()
    world.closure_fit(
        response=ctx.residual(host="D10"),
        regressors={"d10": ctx.series["D10"]},
        rows=rows,
        key=ctx.eng.key,
        quiet=quiet,
        day=day,
        weights=weights[:n],
    )
    return (time.process_time() - t0) / max(n, 1)


def run(*, gate: guard.TimingGate, inputs: list[Path], info: dict[str, Any]) -> None:
    """Run job W2; `inputs` and `info` fill the run log as the job reads."""
    common.assert_launch()
    w0 = common.require_w0()
    info["attempt"] = guard.begin_attempt(job="w2", names=(*OUTPUTS, "run_log_w2.json"))
    tests = guard.known_answer_tests()
    info["known_answer_tests"] = tests
    guard.say("known-answer tests passed:", tests["passed"], "| tests:", tests["n_tests"])
    checks722 = common722.read_checks()
    log = guard.ReadLog()
    ctx = common.load_context(log=log, inputs=inputs)
    info["power_table_delta_version"] = data.power_table_version()
    info["read_order"] = log.as_list()
    info["holdout"] = log.holdout
    correctness_ok = bool(
        tests["passed"]
        and common.assert_later_span(n_days=ctx.n_days)
        and log.holdout_clean()
        and w0["later_span_ok"]
        and w0["holdout_clean"]
    )
    timestamp_defect = not checks722.get("stamp_alignment_ok", True)
    pairs = common.read_pairs()
    qual, n_farm_reads_negative = common.qualifying_pairs(pairs=pairs)
    counts = common.pair_counts(pairs=pairs)
    n_keep = counts["n_qualifying_pairs"]
    n_powered = counts["n_powered_pairs_level_2"]
    n_hosts = counts["n_powered_hosts"]
    if n_farm_reads_negative:
        guard.say(
            "pairs dropped because the farm's generation reads negative:", n_farm_reads_negative
        )
    masked_label = "masked" if w0["mask_available"] else "transfers not masked"
    arc_void = bool(w0["arc_term_void"])
    base = {
        "mask": masked_label, "n_pairs_keeping_sensor": n_keep, "n_powered_pairs_level_2": n_powered,
        "n_powered_hosts": n_hosts, "correctness_ok": correctness_ok, "arc_term_void": arc_void,
        "n_pairs_dropped_farm_reads_negative": n_farm_reads_negative,
        "known_answer_tests": tests, "void_by_construction": list(core.VOID_BY_CONSTRUCTION),
    }  # fmt: skip
    if correctness_ok and core.void_before_run(
        n_pairs_keeping_sensor=n_keep,
        n_powered_pairs_level_2=n_powered,
        n_powered_hosts=n_hosts,
        arc_term_void_flag=arc_void,
    ):
        guard.write_public_json(
            payload={
                "verdict": "inconclusive",
                "row": "row_2_void_before_run",
                "worlds_built": 0,
                **base,
            },
            name=common.W2_VERDICT_NAME,
        )
        guard.say("VERDICT: inconclusive (void before run); no world built")
        return
    membership = read_membership()
    windows = read_windows(mask_available=bool(w0["mask_available"]))
    qualifying = sorted(zip(qual["host"].to_list(), qual["farm"].to_list(), strict=True))
    specs = score.world_specs(qualifying=qualifying)
    weights = core.day_weights(n_days=ctx.n_days)
    expected = [sp for sp in specs if score.expected_to_score(spec=sp, membership=membership)]
    n_injected = sum(1 for sp in specs if sp.role == "injected")
    guard.say("FITS STARTING: job W2;", len(expected), "worlds of", len(specs), "planned")
    scores: list[score.WorldScore] = []
    skips: list[score.SkippedWorld] = []
    n_worlds_planned = len(expected)
    gate.mark()
    stopped: str | None = None
    per_draw = 0.0
    per_ols_draw = 0.0
    for i, spec in enumerate(specs):
        s = score.score_world(
            ctx=ctx, spec=spec, membership=membership, windows=windows, skips=skips
        )
        if s is not None:
            scores.append(s)
        if len(scores) == 1 and s is not None:
            # The pooled population holds about `n_injected` times the first world's stretches.
            per_draw = score.time_draws(
                rows=score.tagged(
                    scores=scores,
                    sensor="farm",
                    roles=("injected", "uninjected", "d10_folded", "d15_folded"),
                ),
                weights=weights,
                replicate=max(n_injected, 1),
            )
            per_ols_draw = time_closure_draw(ctx=ctx, windows=windows, weights=weights)
            projected = core.project_w2_seconds(
                seconds_per_world=s.seconds,
                seconds_per_draw=per_draw,
                n_worlds=n_worlds_planned,
                rescoring=bool(windows),
                seconds_per_ols_draw=per_ols_draw,
            )
            info["projection_core_seconds"] = projected
            if core.over_cap(projected_seconds=projected, cap_seconds=gate.cap_core_seconds):
                msg = f"projected {projected:.0f} core-seconds exceeds the cap; report to the manager; no verdict is written"
                raise guard.TimingGateError(msg)
        if i >= 1 and gate.elapsed() > gate.cap_core_seconds:
            stopped = "the timing gate stopped the job after it had started"
            break
    n_after_verdict = (
        per_draw * core.N_BOOT * core.N_POOLED_BOOTSTRAPS_AFTER_VERDICT
        + per_ols_draw * core.N_BOOT * core.N_CLOSURE_FITS
        + (sum(s.seconds for s in scores) if windows else 0.0)
    )
    if stopped is None and gate.elapsed() + n_after_verdict > gate.cap_core_seconds:
        stopped = "the timing gate stopped the job before the pooled bootstrap"
    scored_names = [s.spec.name for s in scores]
    if len(set(scored_names)) != len(scored_names):
        msg = "a world was scored twice"
        raise RuntimeError(msg)
    info["worlds_scored"] = len(scores)
    info["worlds_planned"] = n_worlds_planned
    info["worlds_skipped"] = [
        {"name": k.name, "reason": k.reason, "n_reference_rows": k.n_reference_rows} for k in skips
    ]
    worlds_equal = bool(
        stopped is not None
        or (len(scores) == n_worlds_planned and set(scored_names) == {e.name for e in expected})
    )
    info["worlds_scored_equal_planned"] = worlds_equal
    if stopped is not None:
        guard.write_public_json(
            payload={
                "verdict": "inconclusive",
                "row": "timing_gate_stop",
                "reason": stopped,
                **base,
            },
            name=common.W2_VERDICT_NAME,
        )
        return
    outcomes, rows, details = evaluate_bars(
        scores=scores, weights=weights, d15_binds=bool(w0["d15_dead_band_binds"])
    )
    align = score.alignment(scores=scores)
    x = core.W2Inputs(
        correctness_ok=correctness_ok,
        alignment_ok=bool(align["ok"]),
        timestamp_defect=timestamp_defect,
        n_pairs_keeping_sensor=n_keep,
        n_powered_pairs_level_2=n_powered,
        n_powered_hosts=n_hosts,
        pooled_se=details["pooled"]["se"],
        n_export_stretches=details["pooled"]["n_export"],
        n_import_stretches=details["pooled"]["n_import"],
        level_1_readable=details["level_1_readable"],
        outcomes=outcomes,
        arc_term_void=arc_void,
    )
    verdict, row = core.verdict_w2(x=x)
    counts_by_world = {s.spec.name: s.counts for s in scores}
    info["threshold_diagnostics"] = {s.spec.name: s.diagnostics for s in scores}
    void_bars = sorted(k for k, v in outcomes.items() if v == "void")
    guard.write_public_csv(
        frame=pl.DataFrame(rows, infer_schema_length=None), name="w2_results.csv"
    )
    payload = {
        "verdict": verdict, "row": row, "outcomes": outcomes, "void_bars": void_bars,
        "alignment": align, "worlds_scored": len(scores), "worlds_planned": n_worlds_planned,
        "worlds_scored_equal_planned": worlds_equal, "worlds_skipped": info["worlds_skipped"],
        "n_worlds_non_finite_injection_scale": sum(
            1 for k in skips if k.reason == "injection_scale_not_finite"
        ),
        "placebo": details["shifted"],
        "pooled": details["pooled"], "level_1": details["level_1"],
        "timestamp_defect": timestamp_defect, "counts_by_world": counts_by_world, **base,
    }  # fmt: skip
    guard.write_public_json(payload=payload, name=common.W2_VERDICT_NAME)
    guard.say(
        "VERDICT:",
        verdict,
        "|",
        row,
        "| void bars:",
        void_bars,
        "| pooled area:",
        round(details["pooled"]["area"], 3),
    )
    # ---- Read-outs: they run only now, after the verdict file is written, and change no verdict. ----
    readouts(
        ctx=ctx,
        scores=scores,
        specs=specs,
        membership=membership,
        windows=windows,
        weights=weights,
        pairs=qual,
        gate=gate,
    )
    guard.say("LAST FIT ENDED: job W2; core-seconds", round(gate.elapsed(), 1))


def readouts(
    *,
    ctx: common.Context,
    scores: list[score.WorldScore],
    specs: list[score.WorldSpec],
    membership: dict[str, list[str]],
    windows: dict[str, list[tuple[int, int]]],
    weights: np.ndarray,
    pairs: pl.DataFrame,
    gate: guard.TimingGate,
) -> None:
    """Run the read-outs after the verdict file: areas by sensor and sub-pool, unmasked statistics, fold ratios, closure slopes, and the two D15 fits."""
    out: list[dict[str, Any]] = []
    inj_farm = score.tagged(scores=scores, sensor="farm", roles=("injected",))
    out.append(
        {
            "readout": "area_without_D14",
            **score.area_of(rows=inj_farm.filter(pl.col("host") != "D14"), weights=weights),
        }
    )
    powered_keys = {
        (h, f, lv)
        for h, f, p1, p2 in zip(
            pairs["host"],
            pairs["farm"],
            pairs["powered_level_1.0"],
            pairs["powered_level_2.0"],
            strict=True,
        )
        for lv, p in ((1.0, p1), (2.0, p2))
        if p
    }
    keep = [
        (h, f, lv) in powered_keys
        for h, f, lv in zip(inj_farm["host"], inj_farm["farm"], inj_farm["level"], strict=True)
    ]
    out.append(
        {
            "readout": "area_powered_sub_pool",
            **score.area_of(
                rows=inj_farm.filter(pl.Series(keep, dtype=pl.Boolean))
                if inj_farm.height
                else inj_farm,
                weights=weights,
            ),
        }
    )
    for sensor in ("meter", "cams"):
        out.append(
            {
                "readout": f"area_{sensor}_sensor",
                **score.area_of(
                    rows=score.tagged(scores=scores, sensor=sensor, roles=("injected",)),
                    weights=weights,
                ),
            }
        )
    blind = score.area_of(
        rows=score.tagged(scores=scores, sensor="meter", roles=("d10_folded",)), weights=weights
    )
    out.append(
        {
            "readout": "blind_meter_sensor_area_folded_d10",
            **blind,
            "contradicts_identity": core.meter_blind_contradicts(area=blind["area"]),
        }
    )
    # Unmasked statistics: the same worlds scored with no switching-log mask.
    if windows:
        unmasked = [
            s
            for spec in specs
            if (s := score.score_world(ctx=ctx, spec=spec, membership=membership, windows={}))
            is not None
        ]
        out.append(
            {
                "readout": "area_farm_sensor_unmasked",
                **score.area_of(
                    rows=score.tagged(scores=unmasked, sensor="farm", roles=("injected",)),
                    weights=weights,
                ),
            }
        )
        gate.check(done=1, total=1)
    guard.write_public_csv(
        frame=pl.DataFrame(out, infer_schema_length=None), name="w2_readouts.csv"
    )
    fits = closure_readouts(ctx=ctx, membership=membership, windows=windows, weights=weights)
    guard.write_public_csv(
        frame=pl.DataFrame(fits, infer_schema_length=None), name="w2_d15_fits.csv"
    )


def closure_readouts(
    *,
    ctx: common.Context,
    membership: dict[str, list[str]],
    windows: dict[str, list[tuple[int, int]]],
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    """Return the D10 fold ratios, the dark-hour closure slopes, and the two D15 fits, as rows with intervals.

    Slopes are the closure slopes of the hourly change of the leave-out residual on the host's
    hourly change (signed for the dark-hour slopes and the D15 fits, folded for D10's fold
    ratios), through the origin, with a whole-day bootstrap. Every row is a read-out with no bar.
    """
    start = ctx.start_row
    key, quiet = (
        ctx.eng.key,
        world.core722.quiet_mask(engine_change=ctx.eng.change, calendar_ok=ctx.eng.has_row),
    )
    day = sky.day_of(n=core.N_GRID) - start // core.HALF_HOURS_PER_DAY
    rows: list[dict[str, Any]] = []

    def fit(
        *,
        name: str,
        host: str,
        kind: str,
        regressors: dict[str, np.ndarray],
        extra: np.ndarray | None = None,
        min_elev: float | None = None,
        contradicts_if_lower_end_above_zero: bool = False,
    ) -> world.core722.Fit:
        masked = score.windows_mask(windows=windows, host=host)
        sel = world.closure_rows(
            zenith=ctx.sky_of[host].zenith,
            masked=masked,
            start=start,
            kind=kind,
            min_elevation_deg=min_elev,
        )
        if extra is not None:
            sel &= extra
        f = world.closure_fit(
            response=ctx.residual(host=host),
            regressors=regressors,
            rows=sel,
            key=key,
            quiet=quiet,
            day=day,
            weights=weights,
        )
        for i, n in enumerate(f.names):
            rows.append(
                {
                    "slope": name,
                    "regressor": n,
                    "host": host,
                    "coef": float(f.coef[i]),
                    "se": float(f.se[i]),
                    "lo": float(f.lo[i]),
                    "hi": float(f.hi[i]),
                    "n_rows": f.n_rows,
                    "n_days": f.n_days,
                    **(
                        {"contradicts_identity": bool(f.lo[i] > 0)}
                        if contradicts_if_lower_end_above_zero
                        else {}
                    ),
                }
            )
        return f

    d15 = ctx.series["D15"]
    d10 = ctx.series["D10"]
    d10_folded = world.fold(signed_mw=d10, capacity_mw=ctx.capacity(host="D10"))
    fit(name="d15_dark_closure", host="D15", kind="dark", regressors={"d15": d15})
    fit(name="d10_dark_closure", host="D10", kind="dark", regressors={"d10": d10})
    export_h = np.zeros(core.N_GRID, dtype=bool)
    import_h = np.zeros(core.N_GRID, dtype=bool)
    export_h[1:] = (d10[1:] < 0) & (d10[:-1] < 0)
    import_h[1:] = (d10[1:] > 0) & (d10[:-1] > 0)
    fit(
        name="d10_fold_ratio_export_hours",
        host="D10",
        kind="lit",
        regressors={"d10_folded": d10_folded},
        extra=export_h,
        min_elev=30.0,
        contradicts_if_lower_end_above_zero=True,
    )
    fit(
        name="d10_fold_ratio_import_hours",
        host="D10",
        kind="lit",
        regressors={"d10_folded": d10_folded},
        extra=import_h,
        min_elev=30.0,
    )
    # The two D15 fits, last: D15 enters the dead-band bar.
    d15_farms = membership.get("D15", [])
    if len(d15_farms) >= 2:
        sensor_level = ctx.farm_mean_share(farms=d15_farms)
        fa = fit(
            name="d15_fit_a_lit",
            host="D15",
            kind="lit",
            regressors={"d15": d15, "farm_sensor": sensor_level},
        )
        lo, hi = float(fa.lo[0]), float(fa.hi[0])
        rows.append(
            {
                "slope": "d15_fit_a_lit",
                "regressor": "reading",
                "host": "D15",
                "reading": core.fit_a_reading(lo=lo, hi=hi),
            }
        )
    inside = ctx.farms_inside(host="D15")
    if inside:
        fb = fit(
            name="d15_fit_b_lit",
            host="D15",
            kind="lit",
            regressors={"d15": d15, "farm_inside_mw": ctx.farm_output(farm=inside[0])},
        )
        rows.append(
            {
                "slope": "d15_fit_b_lit",
                "regressor": "reading",
                "host": "D15",
                "reading": "supports an add-back"
                if core.fit_b_supports_addback(
                    slope=float(fb.coef[0]), farm_coefficient=float(fb.coef[1])
                )
                else "does not support an add-back",
            }
        )
    return rows


def main() -> None:
    """Run job W2 and write its run log even when the job crashes."""
    gate = guard.TimingGate(cap_core_seconds=core.CAP_CORE_SECONDS)
    inputs: list[Path] = []
    info: dict[str, Any] = {}
    with guard.run_log(job="w2", gate=gate, inputs=inputs, info=info):
        run(gate=gate, inputs=inputs, info=info)


if __name__ == "__main__":
    main()

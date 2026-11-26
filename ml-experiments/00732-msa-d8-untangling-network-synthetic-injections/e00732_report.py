# ruff: noqa: E501
"""Experiment 732: the real-day stage, the scoring stage, and the report stage (numpy only, CPU).

`stage_score` rebuilds the test rows and the truth of every overlay, reads the stored predictions of
every fitted arm (seed means) and of experiment 730's stored arms, builds the baselines that need no
network (B-PV2 from the stored SD* outputs, B-T, and B-TN), and writes `scores/readings.json`. An arm
or baseline that is missing gives a void reading, never a zero. `stage_real` reads the real-day sets
(H2, P1, P2, the night bar). `stage_report` combines the readings, the harm checks, and the stop
files into `scores/verdict.json` through `score.build_verdict`. Nothing here prints a value, a date,
or a name; the files hold aggregates only.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import e00732_paths  # noqa: F401

# isort: split
import e00729_banks as banks729
import e00729_score as score729
import e00730_banks as banks730
import e00730_score as score730
import e00732_arms as arms
import e00732_baselines as baselines
import e00732_core as core
import e00732_overlays as ov
import e00732_real as real
import e00732_score as score
import e00732_series_figures as sf
import e00732_stages as stages
import e00732_truth as truth
import numpy as np

SCORES = "scores"
READINGS_FILE = f"{SCORES}/readings.json"
REAL_FILE = f"{SCORES}/real.json"
VERDICT_FILE = f"{SCORES}/verdict.json"
FITTED_ARMS = ("U*", "U-NC", "U-NCw", "S+", "U-SG", "U*-L", "U-N", "U-noBSP")


class ScoreError(RuntimeError):
    """Raised when a stored array cannot be lined up with the rows (a correctness failure, verdict row 1)."""


def read_set(*, folder: Path, name: str) -> dict[str, np.ndarray] | None:
    """Read one stored prediction file, or None when it does not exist."""
    path = folder / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def mean_over_seeds(
    *, arrays: Sequence[Mapping[str, np.ndarray] | None], key: str
) -> np.ndarray | None:
    """Return the mean over seeds of one stored output, or None when a seed, or the output, is missing."""
    if not arrays or any(a is None or key not in a for a in arrays):
        return None
    return np.mean([a[key].astype(np.float64) for a in arrays if a is not None], axis=0)


def arm_scores(*, root: Path, arm: str, n_rows: int) -> score.ArmScores:
    """Return one fitted arm's seed-averaged outputs on the test rows, with None for an output that was not stored.

    Raises:
        ScoreError: When a stored array does not have `n_rows` rows.
    """
    spec = core.ARMS[arm]
    folders = [
        root / "predictions" / core.job_name(arm=arm, arch=spec.arch, seed=s) for s in core.SEEDS
    ]

    def seeds(name: str) -> list[dict[str, np.ndarray] | None]:
        return [read_set(folder=f, name=name) for f in folders if f.exists()]

    def get(name: str, key: str) -> np.ndarray | None:
        a = seeds(name)
        out = mean_over_seeds(arrays=a, key=key) if len(a) == len(core.SEEDS) else None
        if out is not None and out.shape[0] != n_rows:
            msg = f"a stored array of {arm} has {out.shape[0]} rows, not {n_rows}"
            raise ScoreError(msg)
        return out

    return score.ArmScores(
        p=get("test", "p"), p_tr=get("test_transfer", "p"),
        pv=get("test", "pv") if spec.heads else None,
        pv_tr=get("test_transfer", "pv") if spec.heads else None,
        pv_inj={a: v for a in core.TEST_INJECTION_POINTS if spec.heads and (v := get(f"test_inj_{a:g}", "pv")) is not None},
        pv_raised=get("test_raised", "pv") if spec.heads else None,
        step=get("test_transfer", "step_p") if spec.heads else None,
        size_hat=get("test_transfer", "size") if spec.heads else None,
    )  # fmt: skip


def stored_arm_scores(*, root730: Path, arm: str) -> score.ArmScores:
    """Return experiment 730's stored arm S or S-L (its P(X<0) on the plain test rows)."""
    folder, arch = stages.CONTROL_FOLDERS[arm]
    arrays = stages.load_stored(root730=root730, arm=folder, arch=arch)
    return score.ArmScores(p=stages.seed_mean_p(arrays=arrays))


def build_truth(
    *, root730: Path, env: stages.Env, noise_d8: float
) -> tuple[score.Truth, list[arms.Prepared], np.ndarray]:
    """Rebuild the test rows, the transfer overlay's truth, and the injection deltas, all flat on the valid rows.

    Returns:
        The truth, the prepared transfer overlay, and the plain window's `valid` mask `(P, 48)`.

    Raises:
        ScoreError: When the plain and transfer overlays disagree on which half-hours are valid.
    """
    te = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    common = {"holidays": env.holidays, "stats": env.stats, "medians": env.medians}
    plain = arms.prepare_plain(segments=te.segments, neighbours=env.neighbours, **common)
    overlay = arms.prepare(
        overlays=ov.overlay_segments(segments=te.segments, kind="test", neighbours=env.neighbours), **common
    )  # fmt: skip
    index = ov.build_pair_index(segments=te.segments, sources=core.TEST_SIBLING_SOURCES)
    t_plain, t_tr = truth.truth(prepared=plain), truth.truth(prepared=overlay)
    if not np.array_equal(t_plain.valid(), t_tr.valid()):
        msg = "the plain and transfer overlays differ in their valid half-hours"
        raise ScoreError(msg)
    t_raised = truth.truth(prepared=plain, eps=core.TEST_EPS)
    valid = t_plain.valid()
    delta, pair = {}, {}
    for a in core.TEST_INJECTION_POINTS:
        ti = truth.truth(prepared=plain, index=index, amount=a)
        delta[a] = ti.flat(values=ti.delta)
        pair[a] = ti.flat(values=np.broadcast_to(ti.pair[:, None], valid.shape))
    rows = score729.rows_from_segments(segments=te.segments, holidays=env.holidays)
    tr = score.transfer_rows(
        transfers=[p.transfers for p in overlay], picks=arms.all_picks(prepared=overlay),
        valid=valid, noise_d8=noise_d8,
    )  # fmt: skip
    return (
        score.Truth(
            rows=rows, x_tr=t_tr.flat(values=t_tr.row(name="x")), event=t_tr.flat(values=t_tr.row(name="event")),
            size=t_tr.flat(values=t_tr.row(name="size")), cosz=t_plain.flat(values=t_plain.cosz),
            raisable=t_raised.flat(values=t_raised.keep_raisable), delta=delta, pair=pair, tr=tr,
        ),
        overlay,
        valid,
    )  # fmt: skip


def step_rule_scores(
    *, root730: Path, env: stages.Env, base: stages.BaseDays, test_overlay: Sequence[arms.Prepared],
    test_valid: np.ndarray,
) -> dict[str, score.ArmScores]:  # fmt: skip
    """Fit B-T and B-TN on the validation overlay and return their step probabilities on the test overlay (flat on the valid rows)."""
    va = banks729.load_bank(path=banks730.bank_path(root=root730, kind="validation"))
    te = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    common = {"holidays": env.holidays, "stats": env.stats, "medians": env.medians}
    va_ov = ov.overlay_segments(segments=va.segments, kind="validation", neighbours=env.neighbours)
    te_ov = ov.overlay_segments(segments=te.segments, kind="test", neighbours=env.neighbours)
    va_prep = arms.prepare(overlays=va_ov, **common)
    scales = baselines.compute_scales(
        reading=base.reading, bsp=base.bsp, neighbours=env.neighbours, dates=base.dates,
        weekend=base.weekend, train_rows=base.train_rows,
    )  # fmt: skip
    series_va = [baselines.series_of_overlay(overlay=o, holidays=env.holidays) for o in va_ov]
    series_te = [baselines.series_of_overlay(overlay=o, holidays=env.holidays) for o in te_ov]
    t_va = truth.truth(prepared=va_prep)
    out = {}
    for name, with_neighbours in (("B-TN", True), ("B-T", False)):
        weights = baselines.fit_step_rule(
            features=baselines.step_feature_rows(series=series_va, prepared=va_prep, scales=scales, neighbours=with_neighbours),
            event=t_va.row(name="event"), valid=t_va.valid(),
        )  # fmt: skip
        feats = baselines.step_feature_rows(
            series=series_te, prepared=test_overlay, scales=scales, neighbours=with_neighbours
        )
        out[name] = score.ArmScores(
            step=baselines.predict_step_rule(features=feats, weights=weights)[test_valid]
        )
    return out


def pv2_scores(*, root730: Path, truth_: score.Truth) -> score.ArmScores:
    """Return B-PV2 on the plain test rows from the stored SD* outputs (void when a seed is missing)."""
    folder, arch = stages.CONTROL_FOLDERS["SD*"]
    arrays = stages.load_stored(root730=root730, arm=folder, arch=arch)
    pv = baselines.b_pv2(
        demand_by_seed=[a["demand_mu"].astype(np.float64) for a in arrays],
        p_by_seed=[a["p"].astype(np.float64) for a in arrays],
        r=truth_.rows.r, cosz=truth_.cosz,
    )  # fmt: skip
    return score.ArmScores(pv=pv)


def reading_json(*, readings: Mapping[str, object]) -> dict[str, Any]:
    """Return the readings as plain numbers (a void reading carries `void` true and NaN numbers)."""
    out: dict[str, Any] = {}
    for group, items in readings.items():
        if isinstance(items, Mapping):
            out[group] = {k: score.reading_dict(reading=v) for k, v in items.items()}  # ty: ignore[invalid-argument-type]
        else:
            out[group] = score.reading_dict(reading=items)  # ty: ignore[invalid-argument-type]
    return out


def solar_bar_flags(
    *, truth_: score.Truth, a: score.ArmScores, readings: Mapping[str, object]
) -> dict[str, bool]:
    """Return the eight solar bars of U* as pass flags (a bar that cannot be read fails)."""
    rows, up = truth_.rows, score.sun_up(cosz=truth_.cosz)
    del up
    keep = {
        amt: score.injection_keep(delta=truth_.delta[amt], amount=amt, cosz=truth_.cosz, pair=truth_.pair[amt].astype(float))
        for amt in core.TEST_INJECTION_POINTS
    }  # fmt: skip
    reverse = rows.x < 0.0
    have_inj = a.pv is not None and set(a.pv_inj) == set(core.TEST_INJECTION_POINTS)
    recovery = (
        {
            amt: score.recovery_bar(pv_inj=a.pv_inj[amt], pv_plain=a.pv, delta=truth_.delta[amt], amount=amt, keep=keep[amt])  # ty: ignore[invalid-argument-type]
            for amt in core.TEST_INJECTION_POINTS
        }
        if have_inj else None
    )  # fmt: skip
    reverse_bar = (
        score.recovery_reverse_bar(pv_inj=a.pv_inj, pv_plain=a.pv, delta=truth_.delta, keep=keep, reverse=reverse)  # ty: ignore[invalid-argument-type]
        if have_inj else None
    )  # fmt: skip
    counted, report = readings["counted"], readings["report"]
    return score.solar_bars(
        recovery=recovery,
        recovery_reverse=reverse_bar,
        absolute=None if a.pv is None else score.absolute_bar(pv=a.pv, solar=rows.solar, cosz=truth_.cosz, world_is_solar=score.solar_world(rows=rows)),
        sky=score.gain_bar(reading=report["R23"]),  # ty: ignore[invalid-argument-type, not-subscriptable]
        night=None if a.pv is None else score.night_bar(pv=a.pv, cosz=truth_.cosz, valid=np.ones(rows.n, dtype=bool)),
        monotone=None if a.pv is None or a.pv_raised is None else score.monotone_bar(pv=a.pv, pv_raised=a.pv_raised, raisable=truth_.raisable),
        response=None if a.pv is None or a.pv_raised is None else score.response_bar(pv=a.pv, pv_raised=a.pv_raised, raisable=truth_.raisable, solar_target=rows.solar),
        demand=score.gain_bar(reading=counted["C4"]),  # ty: ignore[invalid-argument-type, not-subscriptable]
    )  # fmt: skip


def stage_score(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    base: stages.BaseDays,
    log: stages.Log,
    ctx: object = None,
) -> dict:
    """Score every fitted arm and baseline on the test bank and write `scores/readings.json` (aggregates only).

    Raises:
        StoppedError: When experiment 730's stored predictions do not line up with the rows (correctness failure).
    """
    del ctx
    stages.guard_run(root=root, root730=root730, stage="score")
    limits = stages.read_json(path=root / "limits.json")
    truth_, overlay, valid = build_truth(
        root730=root730, env=env, noise_d8=float(limits["noise_units"]["D8"])
    )
    check = stages.read_json(path=root / stages.BANKS_CHECK)
    stored = {
        arm: stages.load_stored(root730=root730, arm=stages.CONTROL_FOLDERS[arm][0], arch=stages.CONTROL_FOLDERS[arm][1])
        for arm in ("S", "S-L")
    }  # fmt: skip
    problems = score.stored_prediction_problems(
        arrays=stored, rows=truth_.rows, recorded_digest=check.get("row_key_digest")
    )
    if problems:
        msg = "stored predictions do not line up with the rows: " + " | ".join(problems)
        raise stages.leak.CorrectnessFailureError(msg)
    n = truth_.rows.n
    arm_map: dict[str, score.ArmScores] = {
        a: arm_scores(root=root, arm=a, n_rows=n) for a in FITTED_ARMS
    }
    arm_map["S"], arm_map["S-L"] = (stored_arm_scores(root730=root730, arm=a) for a in ("S", "S-L"))
    arm_map["B-PV2"] = pv2_scores(root730=root730, truth_=truth_)
    arm_map.update(
        step_rule_scores(
            root730=root730, env=env, base=base, test_overlay=overlay, test_valid=valid
        )
    )
    readings = score.compute_readings(truth=truth_, arms=arm_map)
    h1 = (
        score.harm_by_band(rows=truth_.rows, p=arm_map["U*"].p, limits=core.read_limits_file(path=root / "limits.json"))  # ty: ignore[invalid-argument-type]
        if arm_map["U*"].p is not None else {}
    )  # fmt: skip
    bars = solar_bar_flags(truth_=truth_, a=arm_map["U*"], readings=readings)
    f1_big = 0.0
    if arm_map["U*"].step is not None:
        found = score.step_f1(
            step_p=arm_map["U*"].step[None, :],
            event=truth_.event[None, :],
            size=truth_.size[None, :],
            valid=np.ones((1, n), dtype=bool),
        )
        f1_big = float(found["f1"])
    record = {
        "readings": reading_json(readings=readings), "h1": h1, "solar_bars": bars, "f1_big": f1_big,
        "n_rows": n, "n_transfer_steps": int(truth_.event.sum()),
        "windows_by_amount": {str(a): int(truth_.pair[a].sum()) for a in core.TEST_INJECTION_POINTS},
        "n_bsp_reverse": int(((truth_.rows.x < 0.0) & np.asarray(truth_.rows.has_bsp, dtype=bool)).sum()),
        "arms_present": sorted(k for k, v in arm_map.items() if v.p is not None or v.pv is not None or v.step is not None),
    }  # fmt: skip
    stages.write_json(path=root / READINGS_FILE, payload=record)
    log(f"scored {n} rows; arms present: {record['arms_present']}")
    return record


def stage_real(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    base: stages.BaseDays,
    log: stages.Log,
    ctx: object = None,
) -> dict:
    """Read the real-day sets of U*: H2 on the held-out pre-engine days and the night bar P1 on every era (aggregates only)."""
    del env, base
    stages.guard_run(root=root, root730=root730, stage="real")
    folders = [
        root / "predictions" / core.job_name(arm="U*", arch=core.ARMS["U*"].arch, seed=s)
        for s in core.SEEDS
    ]
    limits = core.read_limits_file(path=root / "limits.json")
    out: dict[str, Any] = {"h2": None, "p1": None}
    r1 = mean_over_seeds(arrays=[read_set(folder=f, name="real_r1") for f in folders], key="p")
    if r1 is not None:
        out["h2"] = score730.real_harm_check(p=r1, limit=limits["pooled"].limit)
    eras = [read_set(folder=f, name="real_eras") for f in folders]
    pv = mean_over_seeds(arrays=eras, key="pv")
    if pv is not None and ctx is not None:
        import e00729_stages as stages729

        segs = stages729.real_sets(ctx=ctx)["eras"]  # ty: ignore[invalid-argument-type]
        rows = score729.rows_from_segments(segments=segs, holidays=ctx.cal.holidays)  # ty: ignore[unresolved-attribute]
        cosz = np.concatenate([s.cosz[s.cal_index >= 0].reshape(-1) for s in segs])
        if cosz.shape[0] == pv.shape[0] == rows.n:
            out["p1"] = real.p1_night(pv=pv, cosz=cosz, valid=np.ones(rows.n, dtype=bool))
    stages.write_json(path=root / REAL_FILE, payload=out)
    log(f"real: H2 read {out['h2'] is not None}, P1 read {out['p1'] is not None}")
    return out


def stage_report(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    base: stages.BaseDays,
    log: stages.Log,
    ctx: object = None,
) -> dict:
    """Write `scores/verdict.json` from the readings, the harm checks, and the stop and invalid files."""
    del env, base, ctx
    stages.guard_run(root=root, root730=root730, stage="report")
    rec = stages.read_json(path=root / READINGS_FILE)
    real_rec = (
        stages.read_json(path=root / REAL_FILE) if (root / REAL_FILE).exists() else {"h2": None}
    )
    invalid = root / stages.INVALID_FILE
    readings = rec["readings"]

    def as_reading(item: Mapping[str, float | bool] | None) -> score729.ContrastReading | None:
        if item is None or item.get("void"):
            return None
        return score729.ContrastReading(
            reading=core.core729.Reading(float(item["point"]), float(item["lower"]), float(item["upper"])),
            raw_lower=float(item["lower"]), raw_upper=float(item["upper"]),
            kish_blocks=float(item["kish_blocks"]), kish_segments=float(item["kish_segments"]),
            widening=float(item["widening"]), n_rows=int(item["n_rows"]), n_resamples=core.N_RESAMPLES,
        )  # fmt: skip

    margin = float(stages.read_json(path=root730 / "margin.json")["margin"])
    reach = score.overlay_reachability(
        n_transfer_steps=int(rec["n_transfer_steps"]),
        windows_by_amount={float(k): v for k, v in rec["windows_by_amount"].items()},
    )  # fmt: skip
    facts = score.VerdictFacts(
        margin=margin, h1=rec["h1"], h2=real_rec.get("h2"),
        hb=as_reading(readings["hb"]),
        c={k: as_reading(v) for k, v in readings["counted"].items()},
        r4=as_reading(readings["report"].get("R4")), r19=as_reading(readings["report"].get("R19")),
        solar_bars=rec["solar_bars"], injection_reachable=reach["injection_reachable"],
        transfer_reachable=reach["transfer_reachable"], f1_big=float(rec["f1_big"]),
        n_bsp_reverse=int(rec["n_bsp_reverse"]),
        invalid=stages.read_json(path=invalid).get("rule") if invalid.exists() else None,
    )  # fmt: skip
    verdict = score.build_verdict(facts=facts)
    stages.write_json(path=root / VERDICT_FILE, payload=verdict)
    log(f"verdict row {verdict['row']}: {verdict['verdict']}")
    return verdict


# --------------------------------------------------------------------------- the time series windows and the result figures


def unflatten(*, flat: np.ndarray | None, valid: np.ndarray) -> np.ndarray | None:
    """Scatter flat per-row values back onto `(P, 48)` windows (NaN where the half-hour is not valid)."""
    if flat is None:
        return None
    out = np.full(valid.shape, np.nan)
    out[valid] = flat
    return out


def window_table(
    *,
    truth_: score.Truth,
    valid: np.ndarray,
    plain: Sequence[arms.Prepared],
    overlay: Sequence[arms.Prepared],
    sources: Sequence[str],
    arm_map: Mapping[str, score.ArmScores],
    t_plain: truth.Truth,
    noise_d8: float,
) -> sf.WindowTable:
    """Build the per-window table of the test bank (one window per scored day), from the flat rows and the stored outputs."""
    picks = arms.all_picks(prepared=plain)
    p_n = len(picks)
    nan = np.full((p_n, 48), np.nan)
    ustar, s_arm = arm_map["U*"], arm_map["S"]
    p = unflatten(flat=ustar.p, valid=valid)
    p = nan if p is None else p
    p_s = unflatten(flat=s_arm.p, valid=valid)
    x = t_plain.row(name="x")
    xv = np.where(valid, x, np.nan)
    solar, engine = t_plain.row(name="solar"), t_plain.row(name="engine")
    reading = np.where(valid, t_plain.row(name="r"), np.nan)
    pv = unflatten(flat=ustar.pv, valid=valid)
    pv = nan if pv is None else pv
    up = t_plain.cosz > core.SUN_UP_COSZ
    brier = lambda q: np.nanmean((q - (xv < 0.0)) ** 2, axis=1)  # noqa: E731
    first_engine = np.where((engine > 0).any(axis=1), np.argmax(engine > 0, axis=1), 0)
    tr = truth_.tr
    edge = unflatten(flat=tr.edge.astype(float), valid=valid) > 0
    direction = unflatten(flat=tr.edge_direction, valid=valid)
    step_p, step_un = (
        unflatten(flat=ustar.step, valid=valid),
        unflatten(flat=arm_map.get("U-N", score.ArmScores()).step, valid=valid),
    )
    event = unflatten(flat=truth_.event, valid=valid)
    sq = lambda q: (q - np.where(np.isnan(event), 0, event)) ** 2  # noqa: E731
    noise = unflatten(flat=tr.edge_noise, valid=valid)
    shape = unflatten(flat=tr.edge_shape.astype(float), valid=valid)
    case = unflatten(flat=tr.edge_case.astype(float), valid=valid)
    inside = unflatten(flat=tr.inside.astype(float), valid=valid) > 0
    missed = (
        np.where(edge & (step_p < 0.5), noise, 0.0) if step_p is not None else np.zeros((p_n, 48))
    )
    with np.errstate(all="ignore"):
        return sf.WindowTable(
            world=np.array([plain[sg].world for sg, _ in picks]),
            engine_run=(engine > 0).any(axis=1), solar_out=(solar > 0).any(axis=1),
            source=np.array([sources[plain[sg].source_index] for sg, _ in picks]),
            depth=np.nanmax(np.where(valid, -x, -np.inf), axis=1), evening=first_engine >= 28,
            near_zero=(np.abs(reading) < sf.NEAR_ZERO_POINTS).sum(axis=1), mean_p=np.nanmean(p, axis=1), max_p=np.nanmax(np.where(valid, p, -1), axis=1),
            gain_vs_s=brier(p_s) - brier(p) if p_s is not None else np.zeros(p_n),
            solar_error=np.nanmean(np.where(up & valid, np.abs(pv - solar), np.nan), axis=1), pv_max=np.nanmax(np.where(valid, pv, -np.inf), axis=1),
            adds=(edge & (direction > 0)).any(axis=1), removes=(edge & (direction < 0)).any(axis=1),
            in_solar=(inside & (solar > 0)).any(axis=1) & edge.any(axis=1),
            other_bsp=(edge & (case == 3)).any(axis=1), solar_shape=(edge & (shape == 2)).any(axis=1),
            mirrored=(edge & np.isin(case, (0, 1))).any(axis=1),
            step_gain_vs_un=np.nanmean(sq(step_un), axis=1) - np.nanmean(sq(step_p), axis=1) if step_un is not None and step_p is not None else np.zeros(p_n),
            missed_noise_units=missed.max(axis=1), real=np.zeros(p_n, dtype=bool),
        )  # fmt: skip


def stage_series(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    base: stages.BaseDays,
    log: stages.Log,
    ctx: object = None,
) -> dict:
    """Draw the windows W1 to W23 of the test bank to `figures/series/` (windows with S1 solar output go to `PRIVATE/series/` only).

    Each rule is applied to the published table (S2 and engine-only windows) and to the full table; a
    figure is written only when a window fits. W9 to W11 draw one injection pair each, and W24 and W25 need real-day predictions
    (a rule without a window is listed as not drawn). Only counts are returned.
    """
    del ctx
    import e00732_figures as figures

    stages.guard_run(root=root, root730=root730, stage="series")
    limits = stages.read_json(path=root / "limits.json")
    noise = float(limits["noise_units"]["D8"])
    truth_, overlay, valid = build_truth(root730=root730, env=env, noise_d8=noise)
    te = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    plain = arms.prepare_plain(
        segments=te.segments,
        neighbours=env.neighbours,
        holidays=env.holidays,
        stats=env.stats,
        medians=env.medians,
    )
    t_plain = truth.truth(prepared=plain)
    arm_map = {a: arm_scores(root=root, arm=a, n_rows=truth_.rows.n) for a in ("U*", "U-N")}
    arm_map["S"] = stored_arm_scores(root730=root730, arm="S")
    sources = [s.source for s in te.segments]
    table = window_table(
        truth_=truth_,
        valid=valid,
        plain=plain,
        overlay=overlay,
        sources=sources,
        arm_map=arm_map,
        t_plain=t_plain,
        noise_d8=noise,
    )
    t_tr = truth.truth(prepared=overlay)
    drawn, missing = [], []
    for published in (True, False):
        plan = sf.plan_windows(t=table, published=published)
        folder = root / "figures" / "series" if published else root / "PRIVATE" / "series"
        for name, i in plan.items():
            if i is None or (not published and name in drawn):
                if i is None and published:
                    missing.append(name)
                continue
            if not published and sources[plain[i].source_index] != "S1" and table.source[i] != "S1":
                continue
            win = {
                "truth": t_plain.row(name="x")[i], "reading": t_plain.row(name="r")[i], "solar": t_plain.row(name="solar")[i],
                "pv": np.nan_to_num((unflatten(flat=arm_map["U*"].pv, valid=valid) if arm_map["U*"].pv is not None else np.zeros(valid.shape))[i]),
                "step_event": t_tr.row(name="event")[i],
                "step_p": np.nan_to_num((unflatten(flat=arm_map["U*"].step, valid=valid) if arm_map["U*"].step is not None else np.zeros(valid.shape))[i]),
                "p": np.nan_to_num((unflatten(flat=arm_map["U*"].p, valid=valid) if arm_map["U*"].p is not None else np.zeros(valid.shape))[i]),
            }  # fmt: skip
            figures.save_svg(
                chart=sf.window_figure(window=win, title=f"{name}: {core.SERIES_WINDOWS[name]}"),
                path=folder / f"series_{name}.svg",
            )
            drawn.append(name)
    log(f"series: drawn {len(drawn)} windows; no window fits {sorted(set(missing))}")
    return {"drawn": sorted(set(drawn)), "no_window": sorted(set(missing))}


def stage_figures(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    base: stages.BaseDays,
    log: stages.Log,
    ctx: object = None,
) -> dict:
    """Draw the contrast figure (C1 to C7 with their intervals) from `scores/readings.json` (the loss curves have their own stage)."""
    del root730, env, base, ctx
    import e00732_figures as figures

    rec = stages.read_json(path=root / READINGS_FILE)
    path = root / "figures" / "contrasts.svg"
    figures.save_svg(chart=figures.contrast_chart(readings=rec["readings"]["counted"]), path=path)
    log("figures: contrasts written")
    return {"contrasts": str(path.name)}

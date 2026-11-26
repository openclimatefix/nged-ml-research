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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import e00732_paths

# isort: split
import e00729_banks as banks729
import e00729_score as score729
import e00730_banks as banks730
import e00730_score as score730
import e00732_arms as arms
import e00732_baselines as baselines
import e00732_core as core
import e00732_leakage as leak
import e00732_overlays as ov
import e00732_real as real
import e00732_result_figures as rf
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


FiguresError = rf.FiguresError


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


def check_stored_keys(
    *, arrays: Sequence[Mapping[str, np.ndarray] | None], keys: np.ndarray, what: str
) -> None:
    """Check that every stored array carries the row keys of the test rows, and that they match.

    Args:
        arrays: The stored files of one set, one for each seed.
        keys: The (segment, date, half-hour) keys of the test rows `(3, N)`.
        what: The arm and set, for the message.

    Raises:
        CorrectnessFailureError: When a file lacks the keys or they differ from the rows' (a stored
            output that is not on the rows it is scored against, verdict row 1).
    """
    for a in arrays:
        if a is None:
            continue
        if any(k not in a for k in truth.ROW_KEY_NAMES):
            msg = f"{what}: a stored file holds no row keys"
            raise stages.leak.CorrectnessFailureError(msg)
        stored = np.stack([a[k].astype(np.int64) for k in truth.ROW_KEY_NAMES])
        if stored.shape != keys.shape or not np.array_equal(stored, keys):
            msg = f"{what}: the stored row keys differ from the rows of the test bank"
            raise stages.leak.CorrectnessFailureError(msg)


def arm_scores(
    *, root: Path, arm: str, n_rows: int, keys: np.ndarray | None = None
) -> score.ArmScores:
    """Return one fitted arm's seed-averaged outputs on the test rows.

    An output that was not stored is returned as None.

    Args:
        root: The data folder of this experiment.
        arm: The arm's name.
        n_rows: The number of scored rows of the test bank.
        keys: The (segment, date, half-hour) keys of the test rows `(3, N)`; every stored file must
            hold the same keys (not checked when None).

    Returns:
        The arm's outputs (None where an output was not stored).

    Raises:
        ScoreError: When a stored array does not have `n_rows` rows.
        CorrectnessFailureError: When a stored file's row keys are missing or differ.
    """
    spec = core.ARMS[arm]
    folders = [
        root / "predictions" / core.job_name(arm=arm, arch=spec.arch, seed=s) for s in core.SEEDS
    ]

    def seeds(name: str) -> list[dict[str, np.ndarray] | None]:
        return [read_set(folder=f, name=name) for f in folders if f.exists()]

    def get(name: str, key: str) -> np.ndarray | None:
        a = seeds(name)
        if keys is not None:
            check_stored_keys(arrays=a, keys=keys, what=f"{arm} {name}")
        out = mean_over_seeds(arrays=a, key=key) if len(a) == len(core.SEEDS) else None
        if out is not None and out.shape[0] != n_rows:
            msg = f"a stored array of {arm} has {out.shape[0]} rows, not {n_rows}"
            raise ScoreError(msg)
        return out

    return score.ArmScores(
        p=get("test", "p"), p_tr=get("test_transfer", "p"), p_nobsp=get("test_nobsp", "p"),
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


def row_keys_of(*, rows: score729.Rows) -> np.ndarray:
    """Return the (segment, date, half-hour) keys of the rows `(3, N)`."""
    return np.stack([np.asarray(a, dtype=np.int64) for a in (rows.seg, rows.date, rows.hh)])


def check_truth_matches_rows(
    *, truth_: truth.Truth, prepared: Sequence[arms.Prepared], rows: score729.Rows
) -> None:
    """Check that the rebuilt windows hold the same rows as experiment 729's `Rows` of the test bank.

    The (segment, date, half-hour) keys, the signed truth, the reading, and the BSP flag of the flat
    valid half-hours must equal `rows.seg`, `rows.date`, `rows.hh`, `rows.x`, `rows.r`, and
    `rows.has_bsp`.

    Args:
        truth_: The truth of the plain windows.
        prepared: The prepared plain segments the truth was built from.
        rows: The test bank's rows from `rows_from_segments`.

    Raises:
        CorrectnessFailureError: When a key or a value differs (verdict row 1).
    """
    keys = truth_.flat_keys()
    wanted = row_keys_of(rows=rows)
    if keys.shape != wanted.shape or not np.array_equal(keys, wanted):
        msg = "the rebuilt windows hold other (segment, date, half-hour) rows than the test bank's rows"
        raise stages.leak.CorrectnessFailureError(msg)
    if not np.allclose(truth_.flat(values=truth_.row(name="x")), rows.x, atol=1e-3):
        msg = "the rebuilt windows hold another signed truth than the test bank's rows"
        raise stages.leak.CorrectnessFailureError(msg)
    r = truth_.flat(values=truth_.r_points)  # ty: ignore[invalid-argument-type]
    shown = np.isfinite(r)
    if not np.allclose(r[shown], np.asarray(rows.r)[shown], atol=1e-2):
        msg = "the rebuilt windows hold another reading than the test bank's rows"
        raise stages.leak.CorrectnessFailureError(msg)
    has_bsp = np.asarray([p.has_bsp for p in prepared])
    per_window = np.broadcast_to(
        has_bsp[[i for i, _ in truth_.picks]][:, None], truth_.valid().shape
    )
    if not np.array_equal(truth_.flat(values=per_window), np.asarray(rows.has_bsp, dtype=bool)):
        msg = "the rebuilt windows hold another BSP flag than the test bank's rows"
        raise stages.leak.CorrectnessFailureError(msg)


def build_truth(
    *, root730: Path, env: stages.Env, noise_d8: float
) -> tuple[score.Truth, list[arms.Prepared], np.ndarray]:
    """Rebuild the test rows, the transfer overlay's truth, and the injection deltas, all flat.

    The arrays cover the valid rows only.

    Args:
        root730: Experiment 730's data folder (read only).
        env: The stages' environment.
        noise_d8: D8's noise unit in points.

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
    check_truth_matches_rows(truth_=t_plain, prepared=plain, rows=rows)
    tr = score.transfer_rows(
        transfers=[p.transfers for p in overlay], picks=arms.all_picks(prepared=overlay),
        valid=valid, noise_d8=noise_d8,
    )  # fmt: skip
    return (
        score.Truth(
            rows=rows, x_tr=t_tr.flat(values=t_tr.row(name="x")), event=t_tr.flat(values=t_tr.row(name="event")),
            size=t_tr.flat(values=t_tr.row(name="size")), cosz=t_plain.flat(values=t_plain.cosz),
            raisable=t_raised.flat(values=t_raised.keep_raisable), delta=delta, pair=pair, tr=tr,
            unreliable=(
                None if t_plain.unreliable is None else t_plain.flat(values=t_plain.unreliable)
            ),
        ),
        overlay,
        valid,
    )  # fmt: skip


def step_rule_scores(
    *, root730: Path, env: stages.Env, base: stages.BaseDays, test_overlay: Sequence[arms.Prepared],
    test_valid: np.ndarray, root: Path | None = None,
) -> dict[str, score.ArmScores]:  # fmt: skip
    """Fit B-T and B-TN on the validation overlay and return their step probabilities.

    The probabilities are for the test overlay (flat on the valid rows). With `root`, the function
    writes `scores/bt_fit.json` once (the number of weights of each rule and the split it was fitted
    on), so that the order of the choices and the first test score can be checked.
    """
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
    fitted: dict[str, dict[str, object]] = {}
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
        fitted[name] = {"n_weights": int(np.size(weights)), "fitted_on": "validation"}
    if root is not None and not (root / stages.BT_RECORD).exists():
        stages.write_once_json(path=root / stages.BT_RECORD, payload=fitted)
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


def sky_rows(*, prepared: Sequence[arms.Prepared]) -> dict[str, np.ndarray]:
    """Return the sky and the solar target of plain windows, flat on the valid half-hours.

    The arrays are the CAMS index, the clear-sky irradiance, the cosine of the zenith angle, and
    the solar target.

    Args:
        prepared: Plain prepared segments.

    Returns:
        Four flat arrays `csi`, `clear`, `cosz`, and `solar`, in the order of the stored predictions.
    """
    picks = arms.all_picks(prepared=prepared)
    spec = arms.eval_spec(inject=False, neighbours=False)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("csi", "clear", "cosz", "solar", "valid")}
    for i in range(0, len(picks), truth.CHUNK):
        batch = arms.make_eval_batch(prepared=prepared, picks=picks[i : i + truth.CHUNK], spec=spec)
        csi, clear, cosz = baselines.sky_of_windows(x=batch.x)
        parts["csi"].append(csi)
        parts["clear"].append(clear)
        parts["cosz"].append(cosz)
        parts["solar"].append(batch.t[:, arms.T["solar"]])
        parts["valid"].append(batch.t[:, arms.T["valid"]] > 0.5)
    valid = np.concatenate(parts["valid"])
    return {k: np.concatenate(v)[valid] for k, v in parts.items() if k != "valid"}


def pv0_scores(*, root730: Path, env: stages.Env) -> score.ArmScores:
    """Return B-PV0 on the plain test rows: the sky-only rule fitted on the plain validation bank.

    Args:
        root730: Experiment 730's data folder (read only).
        env: The stages' environment.

    Returns:
        The rule's solar estimate on the test rows.
    """
    common = {"holidays": env.holidays, "stats": env.stats, "medians": env.medians}
    va = banks729.load_bank(path=banks730.bank_path(root=root730, kind="validation"))
    te = banks729.load_bank(path=banks730.bank_path(root=root730, kind="test"))
    fit_rows = sky_rows(
        prepared=arms.prepare_plain(segments=va.segments, neighbours=env.neighbours, **common)
    )
    try:
        weights = baselines.fit_pv0(
            csi=fit_rows["csi"],
            clear=fit_rows["clear"],
            cosz=fit_rows["cosz"],
            target=fit_rows["solar"],
        )
    except ValueError:
        return score.ArmScores()  # no validation half-hour holds solar output: B-PV0 is void
    t = sky_rows(
        prepared=arms.prepare_plain(segments=te.segments, neighbours=env.neighbours, **common)
    )
    return score.ArmScores(pv=baselines.b_pv0(weights=weights, csi=t["csi"], clear=t["clear"], cosz=t["cosz"]))  # fmt: skip


def pv1_scores(*, root730: Path, truth_: score.Truth, ctx: object) -> score.ArmScores:
    """Return B-PV1 on the plain test rows.

    The rule uses the climatology's median demand in place of SD*'s demand, with S's P(X<0).

    Args:
        root730: Experiment 730's data folder (read only).
        truth_: The test rows' truth.
        ctx: Experiment 729's context (the climatology is fitted on its training base days).

    Returns:
        The rule's solar estimate (void, with no output, when a seed of S is missing).
    """
    import e00729_stages as stages729
    import e00730_signed as signed

    folder, arch = stages.CONTROL_FOLDERS["S"]
    arrays = stages.load_stored(root730=root730, arm=folder, arch=arch)
    climatology, _ = stages729.fit_baselines(ctx=ctx)  # ty: ignore[invalid-argument-type]
    median = signed.demand_climatology(climatology=climatology, features=truth_.rows.features)
    pv = baselines.b_pv1(
        demand_median=median, p_by_seed=[a["p"].astype(np.float64) for a in arrays],
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
        amt: score.injection_keep(delta=truth_.delta[amt], amount=amt, cosz=truth_.cosz, pair=truth_.pair[amt].astype(float), exclude=truth_.exclude)
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
        absolute=None if a.pv is None else score.absolute_bar(pv=a.pv, solar=rows.solar, cosz=truth_.cosz, world_is_solar=score.solar_world(rows=rows), exclude=truth_.exclude),
        sky=score.gain_bar(reading=report["R23"]),  # ty: ignore[invalid-argument-type, not-subscriptable]
        night=None if a.pv is None else score.night_bar(pv=a.pv, cosz=truth_.cosz, valid=np.ones(rows.n, dtype=bool)),
        monotone=None if a.pv is None or a.pv_raised is None else score.monotone_bar(pv=a.pv, pv_raised=a.pv_raised, raisable=truth_.raisable),
        response=None if a.pv is None or a.pv_raised is None else score.response_bar(pv=a.pv, pv_raised=a.pv_raised, raisable=truth_.raisable, solar_target=rows.solar),
        demand=score.gain_bar(reading=counted["C4"]),  # ty: ignore[invalid-argument-type, not-subscriptable]
    )  # fmt: skip


def assemble_arms(
    *, root: Path, root730: Path, env: stages.Env, base: stages.BaseDays, truth_: score.Truth,
    ctx: object = None, test_overlay: Sequence[arms.Prepared] | None = None, test_valid: np.ndarray | None = None,
) -> dict[str, score.ArmScores]:  # fmt: skip
    """Return every arm's and baseline's seed-averaged outputs on the test rows.

    An output that does not exist is None.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        env: The stages' environment.
        base: D8's base calendar.
        truth_: The test rows' truth.
        ctx: Experiment 729's context (B-PV1's climatology); without it B-PV1 stays void.
        test_overlay: The prepared transfer overlay (rebuilt when None).
        test_valid: The plain window's valid mask (rebuilt when None).

    Returns:
        The outputs by arm name.
    """
    if test_overlay is None or test_valid is None:
        _, test_overlay, test_valid = build_truth(root730=root730, env=env, noise_d8=1.0)
    n = truth_.rows.n
    arm_map: dict[str, score.ArmScores] = {
        a: arm_scores(root=root, arm=a, n_rows=n, keys=row_keys_of(rows=truth_.rows))
        for a in FITTED_ARMS
    }
    arm_map["S"], arm_map["S-L"] = (stored_arm_scores(root730=root730, arm=a) for a in ("S", "S-L"))
    arm_map["B-PV2"] = pv2_scores(root730=root730, truth_=truth_)
    arm_map["B-PV0"] = pv0_scores(root730=root730, env=env)
    if ctx is not None:
        arm_map["B-PV1"] = pv1_scores(root730=root730, truth_=truth_, ctx=ctx)
    arm_map.update(
        step_rule_scores(
            root730=root730,
            env=env,
            base=base,
            test_overlay=test_overlay,
            test_valid=test_valid,
            root=root,
        )
    )
    return arm_map


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

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        env: The stages' environment.
        base: D8's base calendar.
        log: Receives progress lines.
        ctx: Experiment 729's context (B-PV1's climatology); without it B-PV1 stays void.

    Returns:
        The record written to `scores/readings.json`.

    Raises:
        StoppedError: When experiment 730's stored predictions do not line up with the rows
            (correctness failure).
    """
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
    arm_map = assemble_arms(
        root=root,
        root730=root730,
        env=env,
        base=base,
        truth_=truth_,
        ctx=ctx,
        test_overlay=overlay,
        test_valid=valid,
    )
    leak.assert_limits_before_fits(
        limits=root / "limits.json", fit_records=stages.prediction_files(root=root)
    )
    leak.assert_no_test_choice(events=stages.choice_events(root=root))
    stages.write_marker(root=root)
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
        "n_rows": n, "n_unreliable_excluded": int(truth_.exclude.sum()), "n_transfer_steps": int(((truth_.event > 0) & (np.abs(truth_.size) >= core.MIN_STEP_POINTS_REACHABLE)).sum()),
        "windows_by_amount": {str(a): int(truth_.pair[a].sum()) for a in core.TEST_INJECTION_POINTS},
        "n_bsp_reverse": int(((truth_.rows.x < 0.0) & np.asarray(truth_.rows.has_bsp, dtype=bool)).sum()),
        "arms_present": sorted(k for k, v in arm_map.items() if v.p is not None or v.pv is not None or v.step is not None),
    }  # fmt: skip
    stages.write_json(path=root / READINGS_FILE, payload=record)
    log(f"scored {n} rows; arms present: {record['arms_present']}")
    return record


@dataclass(frozen=True)
class RealRows:
    """The valid half-hours of real-day pseudo-segments, flat in the order of the stored predictions.

    Attributes:
        cosz: The cosine of the solar zenith angle.
        csi: The CAMS clear-sky index.
        hh: The UTC half-hour of the day, 0 to 47.
        day: The calendar row of each half-hour's day.
    """

    cosz: np.ndarray
    csi: np.ndarray
    hh: np.ndarray
    day: np.ndarray


def real_rows_of(*, segments: Sequence[Any]) -> RealRows:
    """Return the real-day half-hours of pseudo-segments (the rows that `rows_from_segments` scores)."""
    cosz, csi, hh, day = [], [], [], []
    for seg in segments:
        for d in np.flatnonzero(seg.cal_index >= 0):
            ok = seg.valid[d] & np.isfinite(seg.x[d])
            cosz.append(seg.cosz[d][ok])
            csi.append(seg.csi[d][ok])
            hh.append(np.arange(48)[ok])
            day.append(np.full(int(ok.sum()), int(seg.cal_index[d])))
    return RealRows(
        cosz=np.concatenate(cosz).astype(np.float64), csi=np.concatenate(csi).astype(np.float64),
        hh=np.concatenate(hh), day=np.concatenate(day),
    )  # fmt: skip


def to_days(*, rows: RealRows, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter flat per-half-hour values onto whole days.

    Args:
        rows: The half-hours of the values.
        values: One value per half-hour.

    Returns:
        The values `(D, 48)` (NaN where the half-hour has no row), the valid mask `(D, 48)`, and the
        calendar row of each day.
    """
    days = np.unique(rows.day)
    where = np.searchsorted(days, rows.day)
    out = np.full((len(days), 48), np.nan)
    valid = np.zeros((len(days), 48), dtype=bool)
    out[where, rows.hh] = values
    valid[where, rows.hh] = True
    return out, valid, days


def real_day_checks(
    *,
    r1: RealRows | None,
    eras: RealRows | None,
    k1: RealRows | None,
    outputs: Mapping[str, Mapping[str, np.ndarray | None]],
    day_eras: np.ndarray,
    running_k1: np.ndarray | None,
    edges: Sequence[real.Edge],
    closure_days: np.ndarray | None,
    dates: np.ndarray,
    private_log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return P2, P3, P4, and K3 from the stored real-day outputs, as aggregates only.

    A missing input leaves its check None.

    Args:
        r1: The half-hours of the held-out pre-engine days.
        eras: The half-hours of the net, solar, and fade era days.
        k1: The half-hours of the days with a metered engine record.
        outputs: The seed-averaged outputs by set (`r1`, `eras`, `k1`) and key (`pv`, `step_p`), and
            for the rules K3 reads, `u_step` and `un_step` on `r1`.
        day_eras: The era name of each calendar row.
        running_k1: Whether the metered engine is running, on each half-hour of `k1`.
        edges: The logged closure edges of D8.
        closure_days: Whether each calendar row lies within a logged closure.
        dates: The date of each calendar row (`datetime64[D]`).
        private_log: Receives P4's count of running half-hours (it never enters the result).

    Returns:
        "p2", "p3", "p4", and "k3" (each None when its inputs are missing).
    """
    out: dict[str, Any] = {"p2": None, "p3": None, "p4": None, "k3": None}
    pv_r1 = outputs.get("r1", {}).get("pv")
    if r1 is not None and pv_r1 is not None:
        out["p2"] = real.p2_no_farm(pv=pv_r1, cosz=r1.cosz, valid=np.ones(len(pv_r1), dtype=bool))
    pv_e = outputs.get("eras", {}).get("pv")
    if eras is not None and pv_e is not None:
        pv_d, valid, days = to_days(rows=eras, values=pv_e)
        csi_d, _, _ = to_days(rows=eras, values=eras.csi)
        cosz_d, _, _ = to_days(rows=eras, values=eras.cosz)
        solar_days = np.isin(day_eras[days], ["solar", "fade"])
        out["p3"] = real.p3_follows_sky(
            pv=np.nan_to_num(pv_d[solar_days]), csi=np.nan_to_num(csi_d[solar_days]),
            cosz=np.nan_to_num(cosz_d[solar_days]), valid=valid[solar_days],
        )  # fmt: skip
    pv_k = outputs.get("k1", {}).get("pv")
    if k1 is not None and pv_k is not None and running_k1 is not None:
        out["p4"] = real.p4_absorb(
            pv=pv_k, running=np.asarray(running_k1, dtype=bool), hh=k1.hh,
            valid=np.ones(len(pv_k), dtype=bool), private_log=private_log,
        )  # fmt: skip
    steps = {
        n: outputs["r1"].get(k)
        for n, k in (("U*", "u_step"), ("U-N", "un_step"))
        if "r1" in outputs
    }
    if r1 is not None and closure_days is not None and any(v is not None for v in steps.values()):
        shaped = {}
        valid_d = days_r1 = None
        for name, flat in steps.items():
            if flat is None:
                continue
            shaped[name], valid_d, days_r1 = to_days(rows=r1, values=flat)
        assert valid_d is not None
        assert days_r1 is not None
        day_index = {dates[d].astype(object): i for i, d in enumerate(days_r1)}
        out["k3"] = real.k3_rules(
            edges=edges, steps={k: np.nan_to_num(v) for k, v in shaped.items()}, day_index=day_index,
            valid=valid_d, closure_days=np.asarray(closure_days)[days_r1],
        )  # fmt: skip
    return out


def stage_real(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    base: stages.BaseDays,
    log: stages.Log,
    ctx: object = None,
) -> dict:
    """Read the real-day sets of U*: H2, P1, P2, P3, P4, and K3 for U* and U-N.

    The function reads every era and writes aggregates only. K3 for B-TN stays unread, and K3 is
    void unless the switching logs give at least 10 closure edges of D8 on days with a reading.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        env: Unused.
        base: Unused.
        log: Receives progress lines.
        ctx: The context (the real-day sets are rebuilt from it).

    Returns:
        The record written to `scores/real.json`.
    """
    del env, base
    stages.guard_run(root=root, root730=root730, stage="real")

    def folders_of(arm: str) -> list[Path]:
        spec = core.ARMS[arm]
        return [
            root / "predictions" / core.job_name(arm=arm, arch=spec.arch, seed=s)
            for s in core.SEEDS
        ]

    def private_log(line: str) -> None:
        with (e00732_paths.private_dir(root=root) / "real_private.log").open("a") as f:
            f.write(line + "\n")

    folders = folders_of("U*")
    limits = core.read_limits_file(path=root / "limits.json")
    out: dict[str, Any] = {"h2": None, "p1": None, "p2": None, "p3": None, "p4": None, "k3": None}
    r1 = mean_over_seeds(arrays=[read_set(folder=f, name="real_r1") for f in folders], key="p")
    if r1 is not None:
        out["h2"] = score730.real_harm_check(p=r1, limit=limits["pooled"].limit)
    eras = [read_set(folder=f, name="real_eras") for f in folders]
    pv = mean_over_seeds(arrays=eras, key="pv")
    if ctx is not None:
        import e00729_data as data729
        import e00729_stages as stages729

        sets = stages729.real_sets(ctx=ctx)  # ty: ignore[invalid-argument-type]
        cal = ctx.table.calendar  # ty: ignore[unresolved-attribute]
        rows = {k: real_rows_of(segments=v) for k, v in sets.items()}
        if pv is not None and rows["eras"].cosz.shape[0] == pv.shape[0]:
            out["p1"] = real.p1_night(
                pv=pv, cosz=rows["eras"].cosz, valid=np.ones(len(pv), dtype=bool)
            )
        outputs: dict[str, dict[str, np.ndarray | None]] = {
            "r1": {"pv": mean_over_seeds(arrays=[read_set(folder=f, name="real_r1") for f in folders], key="pv"),
                   "u_step": mean_over_seeds(arrays=[read_set(folder=f, name="real_r1") for f in folders], key="step_p"),
                   "un_step": mean_over_seeds(arrays=[read_set(folder=f, name="real_r1") for f in folders_of("U-N")], key="step_p")},
            "eras": {"pv": pv},
            "k1": {"pv": mean_over_seeds(arrays=[read_set(folder=f, name="real_k1") for f in folders], key="pv")},
        }  # fmt: skip
        units = data729.record_by_day(
            record=data729.metered_record(), days=cal.dates, column="units"
        )
        running = (
            np.nan_to_num(stages729.record_values_for_rows(segments=sets["k1"], record=units)) >= 1
        )
        edges, status = real.load_closure_edges()
        closure, _ = data729.load_closure_days()
        closure_days = np.array(
            [d.astype(object) in closure for d in cal.dates.astype("datetime64[D]")]
        )
        out.update(
            real_day_checks(
                r1=rows["r1"], eras=rows["eras"], k1=rows["k1"], outputs=outputs,
                day_eras=np.asarray(ctx.table.era), running_k1=running, edges=edges,  # ty: ignore[unresolved-attribute]
                closure_days=closure_days, dates=cal.dates.astype("datetime64[D]"),
                private_log=private_log,
            )
        )  # fmt: skip
        out["k3_status"] = status
        out["k3_note"] = "B-TN is not read on real days"
    stages.write_json(path=root / REAL_FILE, payload=out)
    log(
        "real: "
        + ", ".join(f"{k} read {out[k] is not None}" for k in ("h2", "p1", "p2", "p3", "p4", "k3"))
    )
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
    """Write `scores/verdict.json` from the readings, the harm checks, and the stop and invalid files.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        env: Unused.
        base: Unused.
        log: Receives progress lines.
        ctx: Unused.

    Returns:
        The verdict record.
    """
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
    """Build the per-window table of the test bank (one window per scored day) from the flat rows.

    The table also holds the stored outputs.
    """
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
    edge = unflatten(flat=tr.edge.astype(float), valid=valid) > 0  # ty: ignore[unsupported-operator]
    direction = unflatten(flat=tr.edge_direction, valid=valid)
    step_p, step_un = (
        unflatten(flat=ustar.step, valid=valid),
        unflatten(flat=arm_map.get("U-N", score.ArmScores()).step, valid=valid),
    )
    event = unflatten(flat=truth_.event, valid=valid)
    sq = lambda q: (q - np.where(np.isnan(event), 0, event)) ** 2  # noqa: E731  # ty: ignore[no-matching-overload]
    noise = unflatten(flat=tr.edge_noise, valid=valid)
    shape = unflatten(flat=tr.edge_shape.astype(float), valid=valid)
    case = unflatten(flat=tr.edge_case.astype(float), valid=valid)
    inside = unflatten(flat=tr.inside.astype(float), valid=valid) > 0  # ty: ignore[unsupported-operator]
    missed = (
        np.where(edge & (step_p < 0.5), noise, 0.0) if step_p is not None else np.zeros((p_n, 48))  # ty: ignore[no-matching-overload]
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
            adds=(edge & (direction > 0)).any(axis=1), removes=(edge & (direction < 0)).any(axis=1),  # ty: ignore[unsupported-operator]
            in_solar=(inside & (solar > 0)).any(axis=1) & edge.any(axis=1),
            other_bsp=(edge & (case == 3)).any(axis=1), solar_shape=(edge & (shape == 2)).any(axis=1),
            mirrored=(edge & np.isin(case, (0, 1))).any(axis=1),  # ty: ignore[invalid-argument-type]
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
    """Draw the windows W1 to W23 of the test bank to `figures/series/`.

    Windows with S1 solar output go to `PRIVATE/series/` only.

    Each rule is applied to the published table (S2 and engine-only windows) and to the full table; a
    figure is written only when a window fits. W9 to W11 draw one injection pair each, and W24 and
    W25 need real-day predictions
    (a rule without a window is listed as not drawn). Only counts are returned.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        env: The stages' environment.
        base: Unused.
        log: Receives progress lines.
        ctx: Unused.

    Returns:
        The windows drawn and the rules that found no window.
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
    if arm_map["U*"].p is None or arm_map["U*"].pv is None or arm_map["U*"].step is None:
        msg = "the series figures need the stored outputs of the three seeds of U* (no empty plot is drawn)"
        raise FiguresError(msg)
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
                "pv": np.nan_to_num((unflatten(flat=arm_map["U*"].pv, valid=valid) if arm_map["U*"].pv is not None else np.zeros(valid.shape))[i]),  # ty: ignore[not-subscriptable]
                "step_event": t_tr.row(name="event")[i],
                "step_p": np.nan_to_num((unflatten(flat=arm_map["U*"].step, valid=valid) if arm_map["U*"].step is not None else np.zeros(valid.shape))[i]),  # ty: ignore[not-subscriptable]
                "p": np.nan_to_num((unflatten(flat=arm_map["U*"].p, valid=valid) if arm_map["U*"].p is not None else np.zeros(valid.shape))[i]),  # ty: ignore[not-subscriptable]
            }  # fmt: skip
            figures.save_svg(
                chart=sf.window_figure(window=win, title=f"{name}: {core.SERIES_WINDOWS[name]}"),
                path=folder / f"series_{name}.svg",
            )
            drawn.append(name)
    pair_index = ov.build_pair_index(segments=te.segments, sources=core.TEST_SIBLING_SOURCES)
    for name, amount in zip(("W9", "W10", "W11"), core.TEST_INJECTION_POINTS, strict=True):
        drawn_pair = draw_injection_window(
            root=root, name=name, amount=amount, plain=plain, pair_index=pair_index, valid=valid,
            arm=arm_map["U*"], sources=sources,
        )  # fmt: skip
        drawn += drawn_pair
        if not drawn_pair:
            missing.append(name)
    log(f"series: drawn {len(drawn)} windows; no window fits {sorted(set(missing))}")
    return {"drawn": sorted(set(drawn)), "no_window": sorted(set(missing))}


def draw_injection_window(
    *, root: Path, name: str, amount: float, plain: Sequence[arms.Prepared], pair_index: ov.PairIndex,
    valid: np.ndarray, arm: score.ArmScores, sources: Sequence[str],
) -> list[str]:  # fmt: skip
    """Draw the injection pair of one amount whose total delta is the median of its pairs.

    Windows whose solar source is S1 go to the private folder, the others to the published one.

    Args:
        root: The data folder of this experiment.
        name: The window's name (W9, W10, or W11).
        amount: The injected amount in points.
        plain: The prepared plain test segments.
        pair_index: The pair index of the test bank.
        valid: The valid half-hours `(P, 48)`.
        arm: U*'s seed-averaged outputs.
        sources: The solar source of each segment.

    Returns:
        The name when a window was drawn, else an empty list.

    Raises:
        FiguresError: When U*'s solar output of the injected member is missing.
    """
    import e00732_figures as figures

    pv_inj = arm.pv_inj.get(amount)
    if arm.pv is None or pv_inj is None:
        msg = f"{name} needs U*'s solar output on the injected member (amount {amount:g})"
        raise FiguresError(msg)
    ti = truth.truth(prepared=plain, index=pair_index, amount=amount)
    totals = np.nansum(ti.delta, axis=1)
    seg_source = np.array([sources[seg_i] for seg_i, _ in ti.picks])
    drawn: list[str] = []
    for published in (True, False):
        mask = (ti.pair > 0.5) & (totals > 0)
        mask &= (seg_source != "S1") if published else (seg_source == "S1")
        i = sf.series730.median_depth_index(mask=mask, depth=totals)
        if i is None:
            continue
        window = {
            "pv": np.nan_to_num(unflatten(flat=arm.pv, valid=valid)[i]),  # ty: ignore[not-subscriptable]
            "pv_inj": np.nan_to_num(unflatten(flat=pv_inj, valid=valid)[i]),  # ty: ignore[not-subscriptable]
            "delta": ti.delta[i],
        }
        folder = root / "figures" / "series" if published else root / "PRIVATE" / "series"
        figures.save_svg(
            chart=sf.injection_figure(window=window, title=f"{name}: {core.SERIES_WINDOWS[name]}"),
            path=folder / f"series_{name}.svg",
        )
        drawn.append(name)
    return drawn


def stage_figures(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    base: stages.BaseDays,
    log: stages.Log,
    ctx: object = None,
) -> dict:
    """Draw the contrast figure (C1 to C7 with their intervals).

    The figure reads `scores/readings.json`. The loss curves have their own stage.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        env: Unused.
        base: Unused.
        log: Receives progress lines.
        ctx: Unused.

    Returns:
        The file names written.
    """
    import e00732_figures as figures

    stages.guard_run(root=root, root730=root730, stage="figures")
    rec = stages.read_json(path=root / READINGS_FILE)
    if "U*" not in rec["arms_present"]:
        msg = "the figures need the stored outputs of U* (no empty plot is drawn)"
        raise FiguresError(msg)
    written = {}
    noise = float(stages.read_json(path=root / "limits.json")["noise_units"]["D8"])
    truth_, overlay, valid = build_truth(root730=root730, env=env, noise_d8=noise)
    arm_map = assemble_arms(
        root=root, root730=root730, env=env, base=base, truth_=truth_, ctx=ctx,
        test_overlay=overlay, test_valid=valid,
    )  # fmt: skip
    for name, chart in rf.all_figures(
        truth=truth_, arms=arm_map, limits=core.read_limits_file(path=root / "limits.json")
    ).items():
        figures.save_svg(chart=chart, path=root / "figures" / f"{name}.svg")
        written[name] = f"{name}.svg"
    path = root / "figures" / "contrasts.svg"
    figures.save_svg(chart=figures.contrast_chart(readings=rec["readings"]["counted"]), path=path)
    written["contrasts"] = path.name
    bsp = {
        k: v for k, v in rec["readings"]["report"].items() if k.split("_")[0] in ("R19", "R20", "R21", "R22")
    }  # fmt: skip
    path = root / "figures" / "bsp_input.svg"
    try:
        chart = figures.contrast_chart(
            readings=bsp, title="The BSP input: R19 to R22 with their intervals"
        )
    except figures.HistoryError:
        written["bsp_input"] = "every BSP reading is void: not drawn"
    else:
        figures.save_svg(chart=chart, path=path)
        written["bsp_input"] = path.name
    if (root / REAL_FILE).exists():
        path = root / "figures" / "real_checks.svg"
        figures.save_svg(
            chart=figures.real_checks_chart(real=stages.read_json(path=root / REAL_FILE)), path=path
        )
        written["real_checks"] = path.name
    log("figures: " + ", ".join(sorted(written)))
    return written

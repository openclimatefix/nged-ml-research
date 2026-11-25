# ruff: noqa: E501
"""Experiment 721: score the stored fits and write the summary JSON that `e00721_report` reads.

The driver reads only files that the fit and predict phases wrote (`predictions/*.parquet`,
`fits/*.curve.json`, `predictions/*.tokens.npz`). It writes the counted readings G1 to G3, the six
limit checks of G3, the flags of the `SG-A` fits in `W2`, the synthetic world's gate, the Stage 0 table,
and the reports of G4. The summary holds contrasts in points, areas, and booleans, and no generator
series, name, or dated value. The MDE file is written first and refuses to be overwritten.
"""

import datetime
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import e00721_core as core
import e00721_fit as fit
import e00721_score as score
import numpy as np
import polars as pl

HOLDOUT_START = datetime.datetime.fromisoformat(core.HOLDOUT_START_ISO).replace(tzinfo=datetime.UTC)
"""The first instant of the holdout: a stored forecast at or after it is never scored."""


def load_frame(*, data_dir: Path, job: fit.Job) -> pl.DataFrame | None:
    """Return a job's scored rows with `crps`, `mae`, and `fcrps` columns, or `None` when the job has no rows."""
    path = fit.job_paths(data_dir=data_dir, job=job)["predictions"]
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    df = score.with_crps(frame=df)
    folded = score.with_crps(
        frame=df.select(*score.KEYS, "actual", "reading", *score.FOLDED_QCOLS),
        qcols=score.FOLDED_QCOLS,
        truth="reading",
    )
    return df.with_columns(fcrps=folded["crps"])


def contrast_dict(
    *,
    a: pl.DataFrame,
    b: pl.DataFrame,
    seeds: Sequence[int],
    restrict: pl.Expr | None,
    metric: str = "crps",
    n_resamples: int,
) -> dict:
    """Return `a` minus `b` as a dictionary (`point`, `lower`, `upper`, the counts, and the Kish number)."""
    a2, b2 = (x.with_columns(crps=pl.col(metric)) for x in (a, b)) if metric != "crps" else (a, b)
    reading = score.contrast(a=a2, b=b2, seeds=seeds, restrict=restrict, n_resamples=n_resamples)
    return reading.as_dict()


def frames(
    *,
    data_dir: Path,
    arm: str,
    world: str,
    folds: Sequence[str],
    seeds: Sequence[int],
    ids: Sequence[int] | None,
) -> pl.DataFrame | None:
    """Return the stacked scored rows of an arm, or `None` if any job is missing.

    The rows cover the folds and seeds, restricted to the Primaries `ids`.
    """
    parts = []
    for fold in folds:
        for seed in seeds:
            df = load_frame(
                data_dir=data_dir, job=fit.Job(arm=arm, world=world, fold=fold, seed=seed)
            )
            if df is None:
                return None
            parts.append(
                df if ids is None else df.filter(pl.col("time_series_id").is_in(list(ids)))
            )
    return pl.concat(parts)


def flags_of(*, data_dir: Path, folds: Sequence[str], seeds: Sequence[int]) -> list[dict]:
    """Return the flag of every counted stage of the `SG-A` fits in `W2`.

    Under Branch A only stage 1 of an `SG-A` fit is flagged: stages 2 and 3 are read by no verdict row
    ("Stages 2 and 3 are not read by any verdict row"), so each fit contributes one entry. A fit run
    without its continuation counts as flagged ("a fit run without its continuation ... counts as
    flagged in rows 7 to 9, because its stage cannot be shown to have converged"). A missing curve adds
    an unflagged entry, because the missing fit is routed to row 3 by `fit_failed`.
    """
    out = []
    for fold in folds:
        for seed in seeds:
            path = fit.job_paths(
                data_dir=data_dir, job=fit.Job(arm="SG-A", world="W2", fold=fold, seed=seed)
            )["curve"]
            if not path.exists():
                out.append({"evaluated": False, "flagged": False, "reason": "missing curve"})
                continue
            stage1 = json.loads(path.read_text())["stages"][0]
            skipped = {
                "evaluated": False,
                "flagged": True,
                "reason": "no continuation: stage cannot be shown to have converged",
            }
            out.append(stage1["flag"] or skipped)
    return out


def synthetic_cap_hit(*, data_dir: Path, seeds: Sequence[int]) -> bool:
    """Return whether any stored `SG-A` synthetic fit ran to the 400-pass cap (row 2 then reads `inconclusive`)."""
    for seed in seeds:
        path = fit.job_paths(
            data_dir=data_dir,
            job=fit.Job(arm="SG-A", world="synthetic", fold="synthetic", seed=seed),
        )["curve"]
        if path.exists() and any(
            len(st["val_history"]) >= core.SYNTH_MAX_PASSES
            for st in json.loads(path.read_text())["stages"]
        ):
            return True
    return False


def _load_tokens(*, data_dir: Path, arm: str, seed: int) -> dict | None:
    path = fit.job_paths(
        data_dir=data_dir, job=fit.Job(arm=arm, world="synthetic", fold="synthetic", seed=seed)
    )["predictions"].with_suffix(".tokens.npz")
    return dict(np.load(path)) if path.exists() else None


def synthetic_gate(*, data_dir: Path, seeds: Sequence[int]) -> dict:
    """Return the synthetic world's bars per seed.

    The bars are the page's: the sign accuracy outside the 2-point dead band at the two exporting
    Primaries (no restriction on `u`, because the page states none), the false-call rate on the sunny
    midday half-hours of the two non-exporting Primaries with the gate forced open, and the area margin
    over `SG-A-S` on the distinct tokens of the exporters whose `u` is at least 0.3 and which lie outside
    the dead band. Seed 1 is read only when seed 0 passed ("seed 0 first, then seed 1 only if seed 0
    passed"). Every token was scored by a network that no training batch read it through.

    Each seed's entry carries `status`: `missing` (a needed fit or token file is absent), `not_evaluable`
    (an area on fewer than 30 tokens of either class), `pass`, or `fail`.
    """
    per_seed = []
    for seed in seeds:
        sg, placebo = (
            _load_tokens(data_dir=data_dir, arm=arm, seed=seed) for arm in ("SG-A", "SG-A-S")
        )
        if sg is None or placebo is None:
            per_seed.append({"seed": seed, "status": "missing", "evaluable": False, "pass": False})
            break
        acc_rho, acc_rev = score.sign_tokens(
            rho=sg["exporters_rho"], x_true_points=sg["exporters_x_points"], u=sg["exporters_u"],
            token_id=sg["exporters_token_id"], u_min=-np.inf,
        )  # fmt: skip
        accuracy = score.sign_accuracy(rho=acc_rho, reverse=acc_rev)
        day_rho, day_rev = score.sign_tokens(
            rho=sg["exporters_rho"], x_true_points=sg["exporters_x_points"], u=sg["exporters_u"],
            token_id=sg["exporters_token_id"], u_min=core.U_MIN,
        )  # fmt: skip
        accuracy_u03 = score.sign_accuracy(rho=day_rho, reverse=day_rev)
        sunny = sg["open_sunny"].astype(bool) & np.isfinite(sg["open_rho"])
        false_call = score.false_call_rate(rho=sg["open_rho"][sunny])
        areas = []
        for tok in (sg, placebo):
            rho, rev = score.sign_tokens(
                rho=tok["exporters_rho"], x_true_points=tok["exporters_x_points"], u=tok["exporters_u"], token_id=tok["exporters_token_id"],
            )  # fmt: skip
            areas.append((score.auc(score=rho, positive=rev), rev))
        (area, rev), (area_s, _) = areas
        evaluable = score.area_evaluable(reverse=rev)
        ok = bool(
            accuracy >= core.SIGN_ACCURACY_BAR and false_call <= core.FALSE_CALL_RATE_MAX
            and evaluable and area - area_s >= core.AREA_MARGIN
        )  # fmt: skip
        status = "not_evaluable" if not evaluable else ("pass" if ok else "fail")
        per_seed.append({
            "seed": seed, "status": status, "evaluable": bool(evaluable), "sign_accuracy": accuracy, "sign_accuracy_u03": accuracy_u03, "false_call_rate": false_call,
            "area": area, "area_placebo": area_s, "n_sunny_midday": int(sunny.sum()), "pass": status == "pass",
        })  # fmt: skip
        if status != "pass":
            break
    return {
        "per_seed": per_seed,
        "pass": bool(len(per_seed) == len(seeds) and all(s["pass"] for s in per_seed)),
        "missing": any(s["status"] == "missing" for s in per_seed),
        "not_evaluable": any(s["status"] == "not_evaluable" for s in per_seed),
        "failed": any(s["status"] == "fail" for s in per_seed),
    }


def seed_difference_points(*, data_dir: Path, hidden_ids: dict[str, int]) -> float:
    """Return the difference between the two seeds of `SG-0` on fold 2026-05, in points.

    The reading is the signed CRPS on the reverse-flow rows of the hidden pair, pooled over both
    Primaries, on the rows that the two seeds share.
    """
    ids = list(hidden_ids.values())
    frames_ = []
    for seed in (0, 1):
        df = load_frame(
            data_dir=data_dir, job=fit.Job(arm="SG-0", world="W2", fold=core.MDE_FOLD, seed=seed)
        )
        if df is None:
            msg = f"the MDE needs SG-0 in W2, fold {core.MDE_FOLD}, seeds 0 and 1: seed {seed} is missing"
            raise FileNotFoundError(msg)
        frames_.append(df.filter(pl.col("time_series_id").is_in(ids)))
    keys = [k for k in score.KEYS if k != "seed"]
    joined = (
        frames_[0].select(*keys, "actual", "crps")
        .join(frames_[1].select(*keys, "crps"), on=keys, suffix="_b")
        .filter(pl.col("actual") < 0.0)
    )  # fmt: skip
    return float(joined.select((pl.col("crps") - pl.col("crps_b")).mean()).item())


def power_check(*, m1_dir: Path, folds: Sequence[str], series_ids: dict[str, int]) -> dict:
    """Return the power check of G3 from experiment 713's stored `M1` seeds at D5 and D10.

    The five stored seeds of each fold are scored with this page's recipe (leads of 24 up to but not
    including 240 hours, the signed CRPS), and `score.power_check_g3` counts the joint pass rate over
    the 60 assignments of the seeds.
    """
    frames_: dict[tuple[str, int], pl.DataFrame] = {}
    for fold in folds:
        for seed in range(5):
            path = m1_dir / f"{fold}_s{seed}.parquet"
            if not path.exists():
                msg = f"the power check needs {path}"
                raise FileNotFoundError(msg)
            df = pl.read_parquet(path).filter(
                pl.col("time_series_id").is_in(list(series_ids.values())),
                pl.col("lead_hours") >= core.LEAD_MIN_HOURS,
                pl.col("lead_hours") < core.LEAD_MAX_HOURS_EXCLUSIVE,
                pl.col("valid_time").dt.replace_time_zone(None)
                < HOLDOUT_START.replace(tzinfo=None),
            )
            frames_[(fold, seed)] = score.with_crps(frame=df.with_columns(fold=pl.lit(fold)))
    return score.power_check_g3(frames=frames_, id_d5=series_ids["D5"], id_d10=series_ids["D10"])


def build_stops(
    *, data_dir: Path, stage0: dict | None, power: dict | None, synth: dict
) -> core.Stops:
    """Return the stops of verdict row 0 from the stored files.

    Stage 0's stops come from `stage0.json`, the manager's halt from the quick look's harm stop in
    `quicklook.json`, and the stop of an area that is `not evaluable` from the synthetic gate.
    """
    stops = {
        k: bool(v)
        for k, v in ((stage0 or {}).get("stops", {})).items()
        if k in core.Stops.__dataclass_fields__
    }
    quick = fit.data_root(data_dir=data_dir) / "quicklook.json"
    stops["manager_halt"] = bool(
        quick.exists() and json.loads(quick.read_text()).get("harm_stop", False)
    )
    stops["power_check_failed"] = bool(power is not None and not power["pass"])
    stops["area_not_evaluable"] = bool(synth["not_evaluable"])
    return core.Stops(**stops)


def score_all(
    *, data_dir: Path, folds: Sequence[str], seeds: Sequence[int], hidden_ids: dict[str, int], mva_ids: Sequence[int],
    n_resamples: int = core.N_RESAMPLES, stage0: dict | None = None, mde_path: Path | None = None,
    seed_difference: float | None = None, power: dict | None = None,
) -> dict:  # fmt: skip
    """Return the summary of the stored fits.

    Args:
        data_dir: The data folder that holds the experiment's output.
        folds: The folds of the counted readings.
        seeds: The seeds of the counted readings (0 and 1).
        hidden_ids: `{"D5": index, "D10": index}`.
        mva_ids: The Primary indices of the eight MVA-only Primaries (for G4).
        n_resamples: Resamples of each reading.
        stage0: The Stage 0 table (the flag test and the stored curves), copied into the summary.
        mde_path: Where the MDE file is written, which refuses to be overwritten.
        seed_difference: The seed difference of `SG-0` on fold 2026-05, for the MDE (computed when `None`).
        power: The result of `power_check`, copied into the summary.

    Returns:
        The summary: `g1`, `g2`, `g3_checks`, `flags`, `synthetic`, `g4`, `read_outs`, `stage0`, and
        the flags of failure.
    """
    ids = tuple(hidden_ids.values())
    reverse = pl.col("actual_a") < 0.0
    if mde_path is not None and not mde_path.exists():
        diff = (
            seed_difference
            if seed_difference is not None
            else seed_difference_points(data_dir=data_dir, hidden_ids=hidden_ids)
        )
        core.write_mde_file(path=mde_path, seed_difference_points=diff)
    summary: dict = {
        "stage0": stage0 or {},
        "folds": list(folds),
        "seeds": list(seeds),
        "power_check": power or {},
    }
    synth = synthetic_gate(data_dir=data_dir, seeds=seeds)
    summary["synthetic"] = synth
    summary["synthetic_failed"] = bool(synth["failed"])
    summary["synthetic_missing"] = bool(synth["missing"])
    summary["synthetic_not_evaluable"] = bool(synth["not_evaluable"])
    sg_a = frames(data_dir=data_dir, arm="SG-A", world="W2", folds=folds, seeds=seeds, ids=ids)
    sg_0 = frames(data_dir=data_dir, arm="SG-0", world="W2", folds=folds, seeds=seeds, ids=ids)
    sg_s = frames(data_dir=data_dir, arm="SG-A-S", world="W2", folds=folds, seeds=seeds, ids=ids)
    # `P0` in `W2` is fitted at seed 0 only (page, wave table): a read-out and G4 use it, and no counted reading does
    p0_w2 = frames(data_dir=data_dir, arm="P0", world="W2", folds=folds, seeds=(0,), ids=ids)
    missing = [n for n, f in (("SG-A", sg_a), ("SG-0", sg_0), ("SG-A-S", sg_s)) if f is None]
    # a fit that a stop of row 0 or row 2 leaves unrun never falls in row 3
    stopped_earlier = bool(synth["failed"] or synth["not_evaluable"])
    summary["fit_failed"] = bool((missing or synth["missing"]) and not stopped_earlier)
    summary["missing"] = missing
    nan = {"point": float("nan"), "lower": float("nan"), "upper": float("nan")}
    summary["g1"] = (
        contrast_dict(a=sg_a, b=sg_0, seeds=seeds, restrict=reverse, n_resamples=n_resamples)
        if sg_a is not None and sg_0 is not None
        else nan
    )
    summary["g2"] = (
        contrast_dict(a=sg_a, b=sg_s, seeds=seeds, restrict=reverse, n_resamples=n_resamples)
        if sg_a is not None and sg_s is not None
        else nan
    )
    summary["read_outs"] = {}
    if sg_a is not None and p0_w2 is not None:
        summary["read_outs"]["sg_a_minus_p0_reverse"] = contrast_dict(
            a=sg_a.filter(pl.col("seed") == 0),
            b=p0_w2,
            seeds=(0,),
            restrict=reverse,
            n_resamples=n_resamples,
        )
    # G3: SG-A minus P0 in W0 on the plain fits (both seeds) and on SG-A-open (seed 0).
    checks = []
    for reading, arm, rseeds in (("plain", "SG-A", seeds), ("open", "SG-A-open", (0,))):
        for scope in ("pooled", *hidden_ids):
            sel = ids if scope == "pooled" else (hidden_ids[scope],)
            a = frames(data_dir=data_dir, arm=arm, world="W0", folds=folds, seeds=rseeds, ids=sel)
            b = frames(data_dir=data_dir, arm="P0", world="W0", folds=folds, seeds=rseeds, ids=sel)
            if a is None or b is None:
                continue
            r = contrast_dict(a=a, b=b, seeds=rseeds, restrict=None, n_resamples=n_resamples)
            checks.append(
                {
                    "reading": reading,
                    "scope": scope,
                    "point": r["point"],
                    "lower": r["lower"],
                    "limit": core.g3_limit(scope=scope),
                    "upper": r["upper"],
                }
            )
    summary["g3_checks"] = [{k: v for k, v in c.items() if k != "upper"} for c in checks]
    summary["g3_intervals"] = checks
    summary["g3_complete"] = len(checks) == 6
    if len(checks) != 6 and not stopped_earlier:
        summary["fit_failed"] = True
    summary["flags"] = flags_of(data_dir=data_dir, folds=folds, seeds=seeds)
    # G4: the folded CRPS of each arm at the eight MVA-only Primaries, seed 0, each minus P0.
    g4 = {}
    base = frames(data_dir=data_dir, arm="P0", world="W2", folds=folds, seeds=(0,), ids=mva_ids)
    for arm in ("S", "SG-0", "SG-A"):
        a = frames(data_dir=data_dir, arm=arm, world="W2", folds=folds, seeds=(0,), ids=mva_ids)
        if a is not None and base is not None:
            g4[arm] = contrast_dict(
                a=a, b=base, seeds=(0,), restrict=None, metric="fcrps", n_resamples=n_resamples
            )
    summary["g4"] = g4
    summary["stops"] = asdict(
        build_stops(data_dir=data_dir, stage0=stage0, power=power, synth=synth)
    )
    summary["correctness_failed"] = bool(read_correctness(data_dir=data_dir))
    summary["correctness_failures"] = read_correctness(data_dir=data_dir)
    summary["synthetic_cap_hit"] = synthetic_cap_hit(data_dir=data_dir, seeds=seeds)
    return summary


def correctness_path(*, data_dir: Path) -> Path:
    """Return the file that records correctness failures.

    The failures are the holdout guard, the fold point of the loss, and the other checks of
    verdict row 1.
    """
    return fit.data_root(data_dir=data_dir) / "correctness.json"


def record_correctness_failure(*, data_dir: Path, name: str, detail: str) -> None:
    """Append a correctness failure to `correctness.json` (verdict row 1 reads it)."""
    path = correctness_path(data_dir=data_dir)
    old = read_correctness(data_dir=data_dir)
    fit.atomic_write_json(path=path, body={"failures": [*old, {"check": name, "detail": detail}]})


def read_correctness(*, data_dir: Path) -> list[dict]:
    """Return the stored correctness failures (empty when there are none)."""
    path = correctness_path(data_dir=data_dir)
    return json.loads(path.read_text())["failures"] if path.exists() else []


def write_summary(*, summary: dict, path: Path) -> None:
    """Write the summary JSON atomically."""
    fit.atomic_write_json(path=path, body=json.loads(json.dumps(summary, default=float)))

# ruff: noqa: E501
"""Experiment 721: score the stored fits and write the summary JSON that `e00721_report` reads.

The driver reads only files that the fit and predict phases wrote (`predictions/*.parquet`,
`fits/*.curve.json`, `predictions/*.tokens.npz`). It writes the counted readings G1 to G3, the six
limit checks of G3, the flags of the `SG-A` fits in `W2`, the synthetic world's gate, the Stage 0 table,
and the reports of G4. The summary holds contrasts in points, areas, and booleans, and no generator
series, name, or dated value. The MDE file is written first and refuses to be overwritten.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import e00721_core as core
import e00721_fit as fit
import e00721_score as score
import numpy as np
import polars as pl


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
    """Return the flag of every counted stage of the `SG-A` fits in `W2` (a missing curve counts as not evaluated)."""
    out = []
    for fold in folds:
        for seed in seeds:
            path = fit.job_paths(
                data_dir=data_dir, job=fit.Job(arm="SG-A", world="W2", fold=fold, seed=seed)
            )["curve"]
            if not path.exists():
                out.append({"evaluated": False, "flagged": False, "reason": "missing curve"})
                continue
            no_flag = {"evaluated": False, "flagged": False, "reason": "no continuation requested"}
            out.extend(stage["flag"] or no_flag for stage in json.loads(path.read_text())["stages"])
    return out


def synthetic_gate(*, data_dir: Path, seeds: Sequence[int]) -> dict:
    """Return the synthetic world's bars per seed.

    The bars are the sign accuracy, the false-call rate, and the area margin over `SG-A-S`.
    """
    per_seed = []
    for seed in seeds:
        toks = {}
        for arm in ("SG-A", "SG-A-S"):
            path = fit.job_paths(
                data_dir=data_dir,
                job=fit.Job(arm=arm, world="synthetic", fold="synthetic", seed=seed),
            )["predictions"].with_suffix(".tokens.npz")
            if not path.exists():
                toks[arm] = None
                continue
            toks[arm] = dict(np.load(path))
        if toks["SG-A"] is None or toks["SG-A-S"] is None:
            per_seed.append({"seed": seed, "evaluable": False, "pass": False})
            continue
        sg = toks["SG-A"]
        rho, reverse = score.sign_tokens(
            rho=sg["exporters_rho"],
            x_true_points=sg["exporters_x_points"],
            u=sg["exporters_u"],
            token_id=sg["exporters_token_id"],
        )
        acc = score.sign_accuracy(rho=rho, reverse=reverse)
        open_rho = sg["open_rho"]
        sunny = sg["open_u"] >= core.U_MIN
        false_call = score.false_call_rate(rho=open_rho[sunny]) if sunny.any() else float("nan")
        pl_ = toks["SG-A-S"]
        rho_s, reverse_s = score.sign_tokens(
            rho=pl_["exporters_rho"],
            x_true_points=pl_["exporters_x_points"],
            u=pl_["exporters_u"],
            token_id=pl_["exporters_token_id"],
        )
        area, area_s = (
            score.auc(score=rho, positive=reverse),
            score.auc(score=rho_s, positive=reverse_s),
        )
        evaluable = score.area_evaluable(reverse=reverse)
        margin_ok = bool(evaluable and area - area_s >= core.AREA_MARGIN)
        per_seed.append({
            "seed": seed, "evaluable": bool(evaluable), "sign_accuracy": acc, "false_call_rate": false_call,
            "area": area, "area_placebo": area_s,
            "pass": bool(acc >= core.SIGN_ACCURACY_BAR and false_call <= core.FALSE_CALL_RATE_MAX and margin_ok),
        })  # fmt: skip
    return {
        "per_seed": per_seed,
        "pass": bool(per_seed and all(s["pass"] for s in per_seed)),
        "evaluable": all(s["evaluable"] for s in per_seed),
    }


def score_all(
    *, data_dir: Path, folds: Sequence[str], seeds: Sequence[int], hidden_ids: dict[str, int], mva_ids: Sequence[int],
    n_resamples: int = core.N_RESAMPLES, stage0: dict | None = None, mde_path: Path | None = None,
    seed_difference_points: float | None = None,
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
        seed_difference_points: The seed difference of `SG-0` on fold 2026-05, for the MDE.

    Returns:
        The summary: `g1`, `g2`, `g3_checks`, `flags`, `synthetic`, `g4`, `read_outs`, `stage0`, and
        the flags of failure.
    """
    ids = tuple(hidden_ids.values())
    reverse = pl.col("actual_a") < 0.0
    if mde_path is not None and seed_difference_points is not None:
        core.write_mde_file(path=mde_path, seed_difference_points=seed_difference_points)
    summary: dict = {"stage0": stage0 or {}, "folds": list(folds), "seeds": list(seeds)}
    synth = synthetic_gate(data_dir=data_dir, seeds=seeds)
    summary["synthetic"] = synth
    summary["synthetic_failed"] = not synth["pass"]
    sg_a = frames(data_dir=data_dir, arm="SG-A", world="W2", folds=folds, seeds=seeds, ids=ids)
    sg_0 = frames(data_dir=data_dir, arm="SG-0", world="W2", folds=folds, seeds=seeds, ids=ids)
    sg_s = frames(data_dir=data_dir, arm="SG-A-S", world="W2", folds=folds, seeds=seeds, ids=ids)
    p0_w2 = frames(data_dir=data_dir, arm="P0", world="W2", folds=folds, seeds=seeds, ids=ids)
    missing = [
        n for n, f in (("SG-A", sg_a), ("SG-0", sg_0), ("SG-A-S", sg_s), ("P0", p0_w2)) if f is None
    ]
    summary["fit_failed"] = bool(missing)
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
            a=sg_a, b=p0_w2, seeds=seeds, restrict=reverse, n_resamples=n_resamples
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
    if len(checks) != 6:
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
    summary["stops"] = asdict(core.Stops())
    summary["correctness_failed"] = False
    summary["synthetic_cap_hit"] = False
    return summary


def write_summary(*, summary: dict, path: Path) -> None:
    """Write the summary JSON atomically."""
    fit.atomic_write_json(path=path, body=json.loads(json.dumps(summary, default=float)))

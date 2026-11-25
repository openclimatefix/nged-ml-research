# ruff: noqa: E501, PT018, PERF401
"""Experiment 725: the counted contrasts L1 and L2, the guards, the decision rules, and the report-only readings.

Usage:
    e00725_score.py quick          # quick look, folds 2025-11, 2026-01, 2026-03, 2026-06, seeds 0 and 1 (interim) + the audit rule
    e00725_score.py zrule          # the rule on `Z` (design 6.1) on the quick-look folds, seeds 0 and 1
    e00725_score.py two            # counted L1 and L2, 8 folds, seeds 0 and 1, and the futility rule
    e00725_score.py five           # counted L1 and L2, 8 folds, seeds 0 to 4, and the verdict
    e00725_score.py report-only    # the 95% readings of section 6.4 that the stored arms allow
    e00725_score.py probe-flag     # the flag of section 8 that decides whether the budget probe runs
    e00725_score.py probe-read     # the probe's gain on validation loss (report-only)
    e00725_score.py stopping       # the stopping-pass table: each F8 fit beside the stored `M*` fit
    e00725_score.py repro          # the reproduction check of `Z`, fold 2025-11, seed 0 (phase 0)

**The scorer is experiment 713's.** The functions that load a forecast file, convert it to the contract
origin, compute the losses, seed-average, pair, bootstrap, and apply the guards are imported from the
frozen checkout (`e00713_score`), which must reproduce code hash `e85b84e2ea04b3cb`; no experiment 713 file
is edited. F8 supplies the loader for its own arms, the two-seed and five-seed floors and deltas of design
section 6.1, the group readings and verdict of section 6.3, the L2 sentences, the futility rule, the rule on
`Z`, the audit rule of the quick look, and the probe flag.

Every result is written to `score/<command>.json` in F8's data folder. All losses are in points of capacity.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import e00725_common as cm
import e00725_frozen as fz
import e00725_t3_candidate as t3c
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

fz.assert_frozen_checkout()

import e00713_score as sc713  # ty: ignore[unresolved-import]  # noqa: E402

STAR: Final = "M1"
"""The name of `M*` in experiment 713's loader."""
TAU: Final = sc713.TAU
LEVEL: Final = sc713.LEVEL
REPORT_LEVEL: Final = 95.0
NOISE_FLOOR: Final = {2: {"six": 0.054, "eight": 0.142}, 5: {"six": 0.034, "eight": 0.090}}
DELTA: Final = {2: {"six": 0.094, "eight": 0.25}, 5: {"six": 0.06, "eight": 0.16}}
F8_ARMS: Final = ("L", "Z", "E", "S", "T3-M", "T3-L", "T3-C", "L-P", "Z-P")
AUDIT_BIG: Final = 0.15
AUDIT_MODERATE: Final = 0.10
LS_BAND: Final = 0.02
PROBE_THRESHOLD_NATS: Final = 0.01
SCORE_DIR: Final[Path] = cm.ROOT / "score"
DECISIONS_DIR: Final[Path] = cm.ROOT / "decisions"
GROUPS: Final = ("six", "eight")


# ----------------------------------------------------------------------------------------- loading


def arm_file(*, arm: str, month: str, seed: int) -> Path:
    """Return the per-seed forecast file of an F8 arm; `T3-C` is the stored Candidate's prediction on moved inputs."""
    if arm == t3c.ARM:
        return t3c.fit_file(month=month, seed=seed)
    return cm.fit_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed)


def load_arm(*, arm: str, seeds: tuple[int, ...], months: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm's per-seed rows in contract origin and capacity shares, with a `seed` column.

    `C` and `M1` come from experiment 713's loader; an F8 arm reads `work/fits/<arm>/<month>_s<seed>.parquet`.
    """
    if arm not in F8_ARMS:
        return sc713.load_arm(arm=arm, seeds=seeds, months=months)
    parts = []
    for month in months:
        for seed in seeds:
            path = arm_file(arm=arm, month=month, seed=seed)
            frame = pl.read_parquet(path, columns=[*sc713.KEYS, *sc713.QCOLS]).with_columns(
                seed=pl.lit(seed, dtype=pl.Int64)
            )
            parts.append(sc713.to_contract_shares(frame=frame))
    frame = pl.concat(parts)
    assert frame.select(*sc713.KEYS, "seed").is_duplicated().sum() == 0, f"{arm}: duplicated row"
    assert frame["seed"].n_unique() == len(seeds), f"{arm} does not hold every seed of {seeds}"
    return frame


def load_all(
    *, names: list[str], months: tuple[str, ...], seeds: tuple[int, ...]
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, np.ndarray]]:
    """Return every arm's per-seed and seed-averaged scored rows and the Primary groups; assert equal keys."""
    groups, truth = sc713.group_ids(), sc713.truth_stack()
    per_seed, averaged = {}, {}
    for name in names:
        raw = load_arm(arm=name, seeds=seeds, months=months)
        per_seed[name] = sc713.add_losses(frame=raw, truth=truth)
        averaged[name] = sc713.add_losses(frame=sc713.seed_average(frame=raw), truth=truth)
    sc713.assert_same_keys(frames=averaged)
    return per_seed, averaged, groups


def write(*, name: str, obj: object, folder: Path = SCORE_DIR) -> None:
    """Write `<folder>/<name>.json` atomically and print it."""
    cm.write_json_atomic(obj=obj, path=folder / f"{name}.json")
    print(json.dumps(obj, indent=1, default=float), flush=True)


# ------------------------------------------------------------------------------- readings and verdict


def group_reading(*, c: dict, group: str, n_seeds: int) -> str:
    """Return the group's reading by the first row of the table of design section 6.3 that holds.

    The floor guard uses the floor of the number of seeds, and the `no detectable difference` row uses
    the matching `delta`. The seed guard of "small gain" asks for 4 of 5 seeds, or both seeds of two.
    """
    d, lo, hi = c["point"], c["lower"], c["upper"]
    floor_guard = abs(d) >= NOISE_FLOOR[n_seeds][group]
    guards = bool(c["t_guard"] and c["seed_guard"] and floor_guard)
    delta = DELTA[n_seeds][group]
    seed_ok = c["seeds_same_sign"] >= (4 if n_seeds == 5 else n_seeds)
    if lo > TAU and guards:
        return "clear loss"
    if lo > 0 and guards:
        return "loss"
    if hi < 0 and d <= -TAU and guards:
        return "gain"
    if hi < 0 and seed_ok:
        return "small gain"
    if lo > -delta and hi < delta:
        return "no detectable difference"
    return "cannot tell"


def verdict_l1(*, readings: dict[str, str], guards_hold: bool) -> str:
    """Return the verdict word by the table of design section 6.3, read from L1 alone."""
    r = list(readings.values())
    if "clear loss" in r:
        return "refuted"
    if "loss" in r or all(x == "no detectable difference" for x in r):
        return "not supported"
    if all(x == "gain" for x in r):
        return "supported" if guards_hold else "partly supported"
    if "gain" in r:
        return "partly supported"
    if "small gain" in r:
        return "small gain"
    return "inconclusive"


def l2_sentence(*, reading: str) -> str:
    """Return the L2 sentence of design section 6.3 for one group's reading of L2."""
    if reading in {"loss", "clear loss"}:
        return (
            "The lab shortcut accounted for a detectable part of the score of `M*` in this group."
        )
    if reading == "no detectable difference":
        return (
            "Removing the lab shortcut changed the score of `M*` by less than delta in this group."
        )
    if reading in {"gain", "small gain"}:
        return (
            "`L` scored better than `M*` in this group, which the design did not predict; audit "
            "trigger 5 of experiment 713 fires."
        )
    return "The interval of L2 is too wide to say."


def contrasts(
    *, pairs: list[tuple[str, str]], months: tuple[str, ...], seeds: tuple[int, ...], level: float,
    loaded: tuple | None = None,
) -> dict:  # fmt: skip
    """Return each (treatment, reference) contrast for both groups, with the guards of experiment 713."""
    names = sorted({n for p in pairs for n in p})
    per_seed, averaged, groups = loaded or load_all(names=names, months=months, seeds=seeds)
    out: dict = {
        "level": level,
        "n_seeds": len(seeds),
        "months": list(months),
        "seeds": list(seeds),
    }
    for t, r in pairs:
        for g in GROUPS:
            out[f"{t}-{r}:{g}"] = sc713.contrast(
                per_seed=per_seed, averaged=averaged, treatment=t, reference=r, group=g,
                groups=groups, level=level,
            )  # fmt: skip
    return out


def counted(*, months: tuple[str, ...], seeds: tuple[int, ...]) -> dict:
    """Return L1 and L2 at 98.75%, their readings, the L2 sentences, the guards, and (L1) the verdict."""
    n = len(seeds)
    loaded = load_all(names=["L", "C", STAR], months=months, seeds=seeds)
    res = contrasts(
        pairs=[("L", "C"), ("L", STAR)], months=months, seeds=seeds, level=LEVEL, loaded=loaded
    )
    read = {
        "L1": {g: group_reading(c=res[f"L-C:{g}"], group=g, n_seeds=n) for g in GROUPS},
        "L2": {g: group_reading(c=res[f"L-{STAR}:{g}"], group=g, n_seeds=n) for g in GROUPS},
    }
    res["readings"] = read
    res["l2_sentences"] = {g: l2_sentence(reading=read["L2"][g]) for g in GROUPS}
    guards = sc713.supported_guards(
        star="L", averaged=loaded[1], groups=loaded[2], swap_a_points={"six": -1.0, "eight": -1.0}
    )  # the swap guard belongs to experiment 713; F8 has no swap, so it is neutralised and ignored
    for g in GROUPS:
        part = guards[g]
        part["all_hold_f8"] = bool(
            part["leave_one_primary_out"] and part["leave_one_month_out"]
            and all(t["ok"] for t in part["tails"].values()) and part["mae_moves_with_crps"]
        )  # fmt: skip
    guards["all_hold_f8"] = bool(all(guards[g]["all_hold_f8"] for g in GROUPS))
    res["supported_guards"] = guards
    res["verdict_from_L1"] = verdict_l1(readings=read["L1"], guards_hold=guards["all_hold_f8"])
    res["L1_point_below_zero_in_a_group"] = bool(any(res[f"L-C:{g}"]["point"] < 0 for g in GROUPS))
    return res


# ------------------------------------------------------------------------------------- the commands


def cmd_quick() -> int:
    """Quick look (interim, 95%) on the quick-look folds, seeds 0 and 1, and the audit rule on the harm side."""
    res = contrasts(
        pairs=[("L", "C"), ("L", STAR), ("S", "C"), ("S", STAR)],
        months=cm.QUICK_FOLDS, seeds=cm.TWO_SEEDS, level=REPORT_LEVEL,
    )  # fmt: skip
    months = {g: np.array(res[f"L-{STAR}:{g}"]["monthly"]) for g in GROUPS}
    points = {g: res[f"L-{STAR}:{g}"]["point"] for g in GROUPS}
    big = all(points[g] > AUDIT_BIG for g in GROUPS)
    moderate = all(points[g] > AUDIT_MODERATE and int((months[g] > 0).sum()) >= 3 for g in GROUPS)
    res["label"] = "interim: 4 folds, 2 seeds, 95% intervals"
    res["audit_rule"] = {
        "fires": bool(big or moderate),
        "big": bool(big),
        "moderate": bool(moderate),
        "points_L_minus_Mstar": points,
    }
    write(name="quick", obj=res)
    write(name="quick_audit", obj=res["audit_rule"], folder=DECISIONS_DIR)
    if res["audit_rule"]["fires"]:
        print(
            "AUDIT RULE FIRED: stop and ask the manager for a wiring audit before phase 2.",
            flush=True,
        )
    return 0


def cmd_zrule() -> int:
    """The rule on `Z`: audit trigger 1 fires if |Z - M*| exceeds the two-seed floor in either group."""
    res = contrasts(
        pairs=[("Z", STAR)], months=cm.QUICK_FOLDS, seeds=cm.TWO_SEEDS, level=REPORT_LEVEL
    )  # fmt: skip
    fires = {g: bool(abs(res[f"Z-{STAR}:{g}"]["point"]) > NOISE_FLOOR[2][g]) for g in GROUPS}
    res["audit_trigger_1_fires"] = bool(any(fires.values()))
    res["per_group"] = fires
    write(name="zrule", obj=res)
    write(name="zrule", obj=res, folder=DECISIONS_DIR)
    return 0


def read_cut_arms() -> list[str]:
    """Return the arms that the phase 0 projection cut."""
    path = cm.work(data_dir=ML_EXPERIMENTS_DATA_DIR) / "projection.json"
    return json.loads(path.read_text()).get("cut_arms", []) if path.exists() else []


def cmd_two() -> int:
    """The counted two-seed reading and the futility rule (design section 7)."""
    assert (DECISIONS_DIR / "zrule.json").exists() or "Z" in read_cut_arms(), (
        "run `e00725_score.py zrule` first: audit trigger 1 is read before the counted two-seed L2"
    )
    res = counted(months=cm.ALL_FOLDS, seeds=cm.TWO_SEEDS)
    futility_fired = not res["L1_point_below_zero_in_a_group"]
    res["futility_rule_fired"] = bool(futility_fired)
    write(name="two_seed", obj=res)
    write(
        name="futility",
        obj={"fired": bool(futility_fired), "L1_points": {g: res[f"L-C:{g}"]["point"] for g in GROUPS}},
        folder=DECISIONS_DIR,
    )  # fmt: skip
    if futility_fired:
        print("FUTILITY RULE FIRED: write up at 2 seeds; phase 6 does not run.", flush=True)
    return 0


def cmd_five() -> int:
    """The counted five-seed reading and the verdict."""
    write(name="five_seed", obj=counted(months=cm.ALL_FOLDS, seeds=cm.ALL_SEEDS))
    return 0


def available(*, arm: str, months: tuple[str, ...], seeds: tuple[int, ...]) -> bool:
    """Return whether every forecast file of an F8 arm exists (a stored arm always does)."""
    if arm not in F8_ARMS:
        return True
    return all(
        arm_file(arm=arm, month=m, seed=s).exists() for m in months for s in seeds
    )  # fmt: skip


def cmd_report_only() -> int:
    """The report-only readings of design section 6.4 at 95%, for each reading whose arms exist."""
    plan = [
        ("S", "C", cm.ALL_FOLDS), ("S", STAR, cm.ALL_FOLDS), ("L", "S", cm.ALL_FOLDS),
        ("L", "Z", cm.QUICK_FOLDS), ("Z", STAR, cm.QUICK_FOLDS), ("L", "E", cm.QUICK_FOLDS),
        ("E", "C", cm.QUICK_FOLDS), ("T3-L", "T3-M", cm.ALL_FOLDS), ("T3-L", STAR, cm.ALL_FOLDS),
        ("T3-M", STAR, cm.ALL_FOLDS), ("T3-L", "T3-C", cm.ALL_FOLDS), ("T3-C", "C", cm.ALL_FOLDS),
    ]  # fmt: skip
    out: dict = {"level": REPORT_LEVEL, "skipped": []}
    for t, r, months in plan:
        if not all(available(arm=a, months=months, seeds=cm.TWO_SEEDS) for a in (t, r)):
            out["skipped"].append(f"{t}-{r}: an arm is missing")
            continue
        for metric in ("crps", "abs_err"):
            res = contrasts(pairs=[(t, r)], months=months, seeds=cm.TWO_SEEDS, level=REPORT_LEVEL)
            if metric == "crps":
                out[f"{t}-{r}"] = {g: res[f"{t}-{r}:{g}"] for g in GROUPS}
            else:
                loaded = load_all(names=sorted({t, r}), months=months, seeds=cm.TWO_SEEDS)
                out[f"{t}-{r}:abs_err"] = {
                    g: sc713.contrast(
                        per_seed=loaded[0], averaged=loaded[1], treatment=t, reference=r, group=g,
                        groups=loaded[2], level=REPORT_LEVEL, metric="abs_err",
                    )["point"]
                    for g in GROUPS
                }  # fmt: skip
    if "L-S" in out:
        out["L_minus_S_within_0.02_in_both_groups"] = bool(
            all(abs(out["L-S"][g]["point"]) <= LS_BAND for g in GROUPS)
        )
    out["note_T3"] = (
        "T3: the target row's CAMS day summary of UTC day D - 3 in all three arms (T3-C is the stored Candidate); "
        "no arm was trained on the moved summary."
    )
    write(name="report_only", obj=out)
    return 0


# ---------------------------------------------------------------- stopping passes and the probe flag


def stored_entry(*, log: dict) -> dict:
    """Return experiment 713's stored stage log in the shape that `stopping_record` reads."""
    if "first" in log:
        return log
    return {"stage": log.get("stage", 3), "pairs": log["pairs"], "first": log, "rerun": None}


def stage_logs(*, arm: str, month: str, seed: int, stage: int) -> tuple[dict | None, dict | None]:
    """Return the stage log of an F8 arm and of the stored `M*` fit of the same fold and seed."""
    own = cm.stage_log_file(
        data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed, stage=stage
    )
    path713 = fz.stage_log_713(
        data_dir=ML_EXPERIMENTS_DATA_DIR, month=month, seed=seed, stage=stage
    )
    return (
        json.loads(own.read_text()) if own.exists() else None,
        json.loads(path713.read_text()) if path713.exists() else None,
    )


def stopping_table(*, arm: str, months: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict]:
    """Return one row per (fold, seed, stage): the F8 stopping record beside the stored `M*` record."""
    rows = []
    for month in months:
        for seed in seeds:
            for stage in cm.STAGES:
                own, stored = stage_logs(arm=arm, month=month, seed=seed, stage=stage)
                if own is None or stored is None:
                    continue
                a = cm.stopping_record(entry=own)
                b = cm.stopping_record(entry=stored_entry(log=stored), stored=True)
                rows.append({
                    "arm": arm, "month": month, "seed": seed, "stage": stage, "f8": a, "stored_M*": b,
                    "flag_best_pass_under_half": bool(stage == 3 and a["best_pass"] < 0.5 * b["best_pass"]),
                })  # fmt: skip
    return rows


def cmd_stopping() -> int:
    """Print and write the stopping-pass table of every F8 arm that has fits."""
    out = {
        arm: stopping_table(arm=arm, months=cm.ALL_FOLDS, seeds=cm.ALL_SEEDS)
        for arm in ("L", "Z", "E", "L-P", "Z-P")
    }
    write(name="stopping", obj=out)
    return 0


def cmd_probe_flag() -> int:
    """The flag of design section 8: L2 above the two-seed floor in either group, or more than 4 of 16 flagged `L` fits."""
    res = contrasts(
        pairs=[("L", STAR)], months=cm.ALL_FOLDS, seeds=cm.TWO_SEEDS, level=REPORT_LEVEL
    )  # fmt: skip
    above = {g: bool(res[f"L-{STAR}:{g}"]["point"] > NOISE_FLOOR[2][g]) for g in GROUPS}
    rows = [
        r
        for r in stopping_table(arm="L", months=cm.ALL_FOLDS, seeds=cm.TWO_SEEDS)
        if r["stage"] == 3
    ]
    flagged = int(sum(r["flag_best_pass_under_half"] for r in rows))
    out = {
        "L2_above_floor": above, "flagged_fits": flagged, "n_fits_checked": len(rows),
        "flag": bool(any(above.values()) or flagged > 4),
        "values": {"L2_points": {g: res[f"L-{STAR}:{g}"]["point"] for g in GROUPS}, "flagged": flagged},
    }  # fmt: skip
    write(name="probe_flag", obj=out, folder=DECISIONS_DIR)
    write(name="probe_flag", obj=out)
    return 0


def _simulate_patience(*, history: list[float], patience: int) -> int:
    """Return the 1-based best pass that a run with `patience` would pick from a validation history."""
    best, best_pass = float("inf"), 0
    for i, v in enumerate(history, start=1):
        if v < best - 1e-5:
            best, best_pass = v, i
        elif i - best_pass >= patience:
            break
    return best_pass


def _median_window(*, history: list[float], end_pass: int) -> tuple[float, float]:
    """Return the median and the spread (max minus min) of the 5 passes that end at `end_pass`."""
    window = history[max(end_pass - 5, 0) : end_pass]
    return float(np.median(window)), float(max(window) - min(window))


def cmd_probe_read() -> int:
    """The probe's gain on validation loss for `L-P` and `Z-P` (report-only), stage 3 first and stage 2 beside it."""
    out: dict = {"threshold_nats": PROBE_THRESHOLD_NATS, "folds": {}}
    for month in ("2025-11", "2026-06"):
        fold: dict = {}
        for arm in ("L-P", "Z-P"):
            for stage in cm.STAGES:
                own, _ = stage_logs(arm=arm, month=month, seed=0, stage=stage)
                if own is None:
                    continue
                hist = list(own["first"]["val_history"])
                best_pass = cm.stopping_record(
                    entry={
                        "stage": stage,
                        "pairs": 1,
                        "first": {**own["first"], "val_history": hist},
                        "rerun": None,
                    }
                )["best_pass"]
                p8 = _simulate_patience(history=hist, patience=8)
                med_best, spread_best = _median_window(history=hist, end_pass=best_pass)
                med_p8, spread_p8 = _median_window(history=hist, end_pass=p8)
                fold[f"{arm}:stage{stage}"] = {
                    "global_best_pass": best_pass, "patience8_pass": p8, "gain_nats": med_p8 - med_best,
                    "spread_at_best": spread_best, "spread_at_p8": spread_p8,
                }  # fmt: skip
        out["folds"][month] = fold
    margins = {}
    for stage in cm.STAGES:
        diffs = []
        for fold in out["folds"].values():
            lp, zp = fold.get(f"L-P:stage{stage}"), fold.get(f"Z-P:stage{stage}")
            if lp and zp:
                diffs.append(lp["gain_nats"] - zp["gain_nats"])
        margins[f"stage{stage}"] = {
            "n_folds_run": len(diffs),
            "gain_L_minus_gain_Z_per_fold": diffs,
            "exceeds_threshold_on_all_folds_run": bool(diffs and all(d > PROBE_THRESHOLD_NATS for d in diffs)),
        }  # fmt: skip
    out["margins"] = margins
    out["page_statement_patience_8_may_undertrain_L"] = margins["stage3"][
        "exceeds_threshold_on_all_folds_run"
    ]
    if all(available(arm=a, months=("2025-11", "2026-06"), seeds=(0,)) for a in ("L", "L-P")):
        res = contrasts(
            pairs=[("L-P", "L")], months=("2025-11", "2026-06"), seeds=(0,), level=REPORT_LEVEL
        )
        out["L-P_minus_L"] = {g: res[f"L-P-L:{g}"]["point"] for g in GROUPS}
    write(name="probe_read", obj=out)
    return 0


# ---------------------------------------------------------------------------------- reproduction check


def headline_crps(*, arm: str, month: str, seed: int) -> dict[str, float]:
    """Return the headline-band CRPS (24 to 240 h) of one arm, fold, and seed, per group."""
    groups, truth = sc713.group_ids(), sc713.truth_stack()
    raw = load_arm(arm=arm, seeds=(seed,), months=(month,))
    scored = sc713.add_losses(frame=raw, truth=truth)
    return {
        g: float(
            sc713.restrict(frame=scored, group=g, band=sc713.HEADLINE_BAND_HOURS, groups=groups)[
                "crps"
            ].mean()
        )  # ty: ignore[invalid-argument-type]
        for g in GROUPS
    }


def seed_spread(*, month: str = "2025-11") -> dict:
    """Return the across-seed standard deviation of `best_val` per stage and of the headline CRPS of `M*`."""
    out: dict = {"best_val": {}, "crps": {}}
    for stage in cm.STAGES:
        fold, pooled = [], []
        for m in cm.ALL_FOLDS:
            vals = []
            for s in cm.ALL_SEEDS:
                _, stored = stage_logs(arm="L", month=m, seed=s, stage=stage)
                if stored is not None:
                    vals.append(float(stored_entry(log=stored)["first"]["best_val"]))
            if len(vals) >= 2:
                pooled.append(float(np.std(vals, ddof=1)))
                if m == month:
                    fold = vals
        out["best_val"][f"stage{stage}"] = {
            "sd_fold": float(np.std(fold, ddof=1)) if len(fold) >= 2 else None,
            "sd_pooled_mean_of_fold_sds": float(np.mean(pooled)) if pooled else None,
        }
    per_seed = [headline_crps(arm=STAR, month=month, seed=s) for s in cm.ALL_SEEDS]
    out["crps"] = {g: float(np.std([x[g] for x in per_seed], ddof=1)) for g in GROUPS}
    return out


def cmd_repro() -> int:
    """Compare the `Z` fit of fold 2025-11, seed 0, with the stored `M*` fit (design section 5), and write the verdict."""
    month, seed = "2025-11", 0
    spread = seed_spread(month=month)
    rows, miss_nats, miss_points = [], [], []
    for stage in cm.STAGES:
        own, stored = stage_logs(arm="Z", month=month, seed=seed, stage=stage)
        assert own is not None and stored is not None, (
            f"the Z fit or the stored M* log of stage {stage} is missing"
        )
        st = stored_entry(log=stored)["first"]
        diff = abs(float(own["best_val"]) - float(st["best_val"]))
        miss_nats.append((stage, diff))
        rows.append({
            "stage": stage, "best_val_Z": own["best_val"], "best_val_stored": st["best_val"],
            "abs_difference": diff, "passes_Z": own["passes"], "passes_stored": st["passes"],
            "best_pass_Z": own["stopping"]["best_pass"],
            "best_pass_stored": cm.stopping_record(entry=stored_entry(log=stored), stored=True)["best_pass"],
        })  # fmt: skip
    z, m = (
        headline_crps(arm="Z", month=month, seed=seed),
        headline_crps(arm=STAR, month=month, seed=seed),
    )
    for g in GROUPS:
        miss_points.append((g, abs(z[g] - m[g])))
    sd_val = spread["best_val"]
    outcome = "pass"
    notes = []
    for stage, diff in miss_nats:
        if diff > 0.01:
            sd = (
                sd_val[f"stage{stage}"]["sd_fold"]
                or sd_val[f"stage{stage}"]["sd_pooled_mean_of_fold_sds"]
            )
            outcome = (
                "stop"
                if sd is None or diff > sd
                else (outcome if outcome == "stop" else "manager_gpu_nondeterminism")
            )
            notes.append(
                f"stage {stage} best_val differs by {diff:.4f} nats (sd across seeds {sd})"
            )
    for g, diff in miss_points:
        if diff > 0.02:
            sd = spread["crps"][g]
            outcome = (
                "stop"
                if diff > sd
                else (outcome if outcome == "stop" else "manager_gpu_nondeterminism")
            )
            notes.append(
                f"{g}: headline CRPS differs by {diff:.4f} points (sd across seeds {sd:.4f})"
            )
    res = {
        "fold": month, "seed": seed, "rows": rows, "crps_Z": z, "crps_M*": m, "spread": spread,
        "outcome": outcome, "notes": notes,
        "tolerances": {"nats": 0.01, "points": 0.02},
    }  # fmt: skip
    write(name="repro", obj=res)
    write(name="repro", obj=res, folder=DECISIONS_DIR)
    return 0 if outcome == "pass" else 1


COMMANDS: Final = {
    "quick": cmd_quick, "zrule": cmd_zrule, "two": cmd_two, "five": cmd_five,
    "report-only": cmd_report_only, "probe-flag": cmd_probe_flag, "probe-read": cmd_probe_read,
    "stopping": cmd_stopping, "repro": cmd_repro,
}  # fmt: skip


def main() -> int:
    """Run one scoring command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args()
    return COMMANDS[args.command]()


if __name__ == "__main__":
    sys.exit(main())

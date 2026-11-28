# ruff: noqa: E501
"""Print every number the page quotes: the contrasts, the net effects, the verdict, and the checks.

`compute_report` reads the tables of sums that `e00738_score.score_fold_seed` wrote (one per fold
and seed) and returns every reading as plain data. Every contrast is the first arm minus the second,
in unscaled CRPS unless it says otherwise. `render` turns the result into text, which the report
stage prints and saves.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import e00738_common as common
import e00738_score as score
import numpy as np

Tables = Mapping[tuple[str, int], dict]


def reading_dict(*, r: score.Reading) -> dict[str, object]:
    """Return a reading as plain data."""
    return {
        "point": r.point,
        "t": [r.t_lower, r.t_upper],
        "boot": [r.boot_lower, r.boot_upper],
        "seed_points": list(r.seed_points),
        "gain": r.gain,
        "seeds_agree": r.seeds_agree,
    }


class CommonArgs(TypedDict):
    """The keyword arguments that every contrast and net-effect call of one report shares."""

    tables: Tables
    months: tuple[str, ...]
    seeds: tuple[int, ...]
    labels: list[str]
    n_resamples: int


def with_months(*, kw: CommonArgs, months: tuple[str, ...]) -> CommonArgs:
    """Return the shared arguments with a different set of months."""
    return CommonArgs(
        tables=kw["tables"],
        months=months,
        seeds=kw["seeds"],
        labels=kw["labels"],
        n_resamples=kw["n_resamples"],
    )


def contrast(
    *,
    tables: Tables,
    months: tuple[str, ...],
    seeds: tuple[int, ...],
    labels: list[str],
    a: str,
    b: str,
    wld: str,
    slc: str,
    group: str = "all14",
    metric: str = "crps",
    level: float = common.LEVEL_BONFERRONI,
    drop: tuple[str, ...] = (),
    n_resamples: int = common.N_RESAMPLES,
) -> score.Reading:
    """Return one contrast `a - b` as a reading.

    Args:
        tables: The sums of each (month, seed).
        months: The fold months used.
        seeds: The seeds used.
        labels: The 16 Primaries' labels.
        a: The first arm.
        b: The second arm.
        wld: The world.
        slc: The slice.
        group: The Primary group.
        metric: The metric.
        level: The coverage in percent.
        drop: Primary labels left out (leave-one-out).
        n_resamples: Bootstrap resamples.

    Returns:
        The reading.
    """
    prims = score.primary_indices(group=group, labels=labels, drop=drop)
    sums, counts = score.month_arrays(
        tables=tables,
        months=months,
        seeds=seeds,
        arm_a=a,
        arm_b=b,
        wld=wld,
        slc=slc,
        metric=metric,
        prims=prims,
    )
    return score.reading_from_month_arrays(
        sums=sums, counts=counts, level=level, n_resamples=n_resamples
    )


def net(
    *,
    tables: Tables,
    months: tuple[str, ...],
    seeds: tuple[int, ...],
    labels: list[str],
    a: str,
    b: str,
    q: float,
    wld: str = "train_family",
    group: str = "all14",
    level: float = common.LEVEL_BONFERRONI,
    drop: tuple[str, ...] = (),
    n_resamples: int = common.N_RESAMPLES,
) -> score.Reading:
    """Return the net all-rows effect q C + (1 - q) V of `a` against `b`, from in-force and event-free rows."""
    prims = score.primary_indices(group=group, labels=labels, drop=drop)
    c = score.month_arrays(
        tables=tables,
        months=months,
        seeds=seeds,
        arm_a=a,
        arm_b=b,
        wld=wld,
        slc="in_force",
        metric="crps",
        prims=prims,
    )
    v = score.month_arrays(
        tables=tables,
        months=months,
        seeds=seeds,
        arm_a=a,
        arm_b=b,
        wld=wld,
        slc="event_free",
        metric="crps",
        prims=prims,
    )
    return score.net_reading(c=c, v=v, q=q, level=level, n_resamples=n_resamples)


def screen_in_force_scaled(
    *, tables: Tables, months: tuple[str, ...], labels: list[str]
) -> dict[str, float]:
    """Return the in-force scaled CRPS of LD, LM, and LR on the screen folds at seed 0 (interim)."""
    prims = score.primary_indices(group="all14", labels=labels)
    out = {}
    for arm in common.SURVIVAL_ARMS:
        s, n = score.month_arrays(
            tables=tables,
            months=months,
            seeds=(common.SCREEN_SEED,),
            arm_a=arm,
            arm_b=None,
            wld="train_family",
            slc="in_force",
            metric="crps_scaled",
            prims=prims,
        )
        out[arm] = float((s[0] / n).mean())
    return out


def compute_report(
    *,
    tables: Tables,
    labels: list[str],
    s_star: str,
    months: tuple[str, ...] = common.FOLDS,
    seeds: tuple[int, ...] = common.SEEDS,
    invalid: bool = False,
    n_resamples: int = common.N_RESAMPLES,
) -> dict:
    """Return every reading of the page.

    Args:
        tables: The sums of each (month, seed).
        labels: The 16 Primaries' labels in index order.
        s_star: The chosen survival arm.
        months: The scored folds.
        seeds: The seeds.
        invalid: Whether a leakage test, the oracle, the reproduction, or the gate failed.
        n_resamples: Bootstrap resamples (tests lower it).

    Returns:
        A dict of readings, the verdict state, and the verdict.
    """
    kw: CommonArgs = {
        "tables": tables,
        "months": months,
        "seeds": seeds,
        "labels": labels,
        "n_resamples": n_resamples,
    }
    c1 = contrast(**kw, a=s_star, b=common.ARM_M1, wld="train_family", slc="in_force")
    c2 = contrast(**kw, a=s_star, b="LP", wld="train_family", slc="in_force")
    c3 = contrast(**kw, a=s_star, b=common.ARM_W0, wld="train_family", slc="in_force")
    v1 = contrast(
        **kw,
        a=s_star,
        b=common.ARM_M1,
        wld="train_family",
        slc="event_free",
        level=common.LEVEL_PLAIN,
    )
    v2 = contrast(**kw, a=s_star, b=common.ARM_M1, wld="real", slc="all", level=common.LEVEL_PLAIN)
    n1 = net(**kw, a=s_star, b=common.ARM_M1, q=common.Q_MAIN)
    n3 = net(**kw, a=s_star, b=common.ARM_W0, q=common.Q_MAIN)
    held = {
        name: contrast(**kw, a=s_star, b=other, wld="held_out", slc="in_force")
        for name, other in (("C2", "LP"), ("C3", common.ARM_W0))
    }
    held_agrees = all(
        np.sign(held[k].point) == np.sign(c.point) for k, c in (("C2", c2), ("C3", c3))
    )
    state = score.VerdictState(
        invalid=invalid,
        n1_lower_above_zero=n1.t_lower > 0.0,
        v1_fails=v1.point > common.V1_FAIL,
        n1_inside_band=n1.t_lower >= -common.NO_SHIFT_BAND and n1.t_upper <= common.NO_SHIFT_BAND,
        n1_gain=n1.gain,
        n1_at_threshold=n1.point <= common.N1_THRESHOLD,
        n1_seeds=n1.seeds_agree,
        n3_gain=n3.gain,
        n3_at_threshold=n3.point <= common.N3_THRESHOLD,
        n3_seeds=n3.seeds_agree,
        c2_gain=c2.gain,
        c2_seeds=c2.seeds_agree,
        held_out_agrees=bool(held_agrees),
    )
    row, _ = score.verdict(state=state)
    unscreened = tuple(m for m in months if m not in common.SCREEN_FOLDS)
    unscreened_c1 = contrast(
        **with_months(kw=kw, months=unscreened),
        a=s_star,
        b=common.ARM_M1,
        wld="train_family",
        slc="in_force",
    )
    report: dict = {
        "s_star": s_star,
        "C1": reading_dict(r=c1),
        "C2": reading_dict(r=c2),
        "C3": reading_dict(r=c3),
        "V1": reading_dict(r=v1),
        "V2": reading_dict(r=v2),
        "N1": reading_dict(r=n1),
        "N3": reading_dict(r=n3),
        "held_out": {k: reading_dict(r=r) for k, r in held.items()},
        "V2_flag": v2.point > common.V2_FLAG,
        "verdict_state": state.__dict__,
        "verdict_row": row,
        "verdict": score.label_of(state=state),
        "unscreened_folds": {
            "months": list(unscreened),
            "C1": reading_dict(r=unscreened_c1),
            "same_sign_as_8_fold": bool(np.sign(unscreened_c1.point) == np.sign(c1.point)),
        },
    }
    report["net_by_q"] = {
        str(q): {
            "N1": reading_dict(r=net(**kw, a=s_star, b=common.ARM_M1, q=q)),
            "N3": reading_dict(r=net(**kw, a=s_star, b=common.ARM_W0, q=q)),
        }
        for q in common.Q_VALUES
    }
    report["report_only"] = {
        "W0_minus_M1": reading_dict(
            r=contrast(**kw, a=common.ARM_W0, b=common.ARM_M1, wld="train_family", slc="in_force")
        ),
        "LR_minus_LM": reading_dict(
            r=contrast(**kw, a="LR", b="LM", wld="train_family", slc="in_force")
        ),
        "LR_minus_LM_held_out": reading_dict(
            r=contrast(**kw, a="LR", b="LM", wld="held_out", slc="in_force")
        ),
        "LM_minus_LD": reading_dict(
            r=contrast(**kw, a="LM", b="LD", wld="train_family", slc="in_force")
        ),
        "C1_scaled": reading_dict(
            r=contrast(
                **kw,
                a=s_star,
                b=common.ARM_M1,
                wld="train_family",
                slc="in_force",
                metric="crps_scaled",
            )
        ),
        "C1_nmae": reading_dict(
            r=contrast(
                **kw, a=s_star, b=common.ARM_M1, wld="train_family", slc="in_force", metric="nmae"
            )
        ),
        "C3_nmae": reading_dict(
            r=contrast(
                **kw, a=s_star, b=common.ARM_W0, wld="train_family", slc="in_force", metric="nmae"
            )
        ),
    }
    for sel in ("LD", "LM", "LR"):
        report["report_only"][f"{sel}_C1"] = reading_dict(
            r=contrast(**kw, a=sel, b=common.ARM_M1, wld="train_family", slc="in_force")
        )
    report["by_group"] = {
        g: {
            k: reading_dict(
                r=contrast(**kw, a=s_star, b=b, wld="train_family", slc="in_force", group=g)
            )
            for k, b in (("C1", common.ARM_M1), ("C2", "LP"), ("C3", common.ARM_W0))
        }
        for g in common.GROUP_OF_PRIMARY
    }
    report["by_age_band"] = {
        name: {
            k: reading_dict(
                r=contrast(**kw, a=s_star, b=b, wld="train_family", slc=f"in_force_{name}")
            )
            for k, b in (("C1", common.ARM_M1), ("C2", "LP"), ("C3", common.ARM_W0))
        }
        for name, _, _ in common.AGE_BANDS
    }
    report["by_stratum"] = {
        s: {
            arm: contrast(**kw, a=arm, b=common.ARM_M1, wld="train_family", slc=s).point
            for arm in common.FIT_ARMS
        }
        for s in ("far", "middle", "near", "ended_24h", "ended_1_7", "event_free", "onset_ahead")
    }
    report["leave_one_primary_out"] = {
        lab: {
            k: contrast(**kw, a=s_star, b=b, wld="train_family", slc="in_force", drop=(lab,)).point
            for k, b in (("C1", common.ARM_M1), ("C2", "LP"), ("C3", common.ARM_W0))
        }
        for lab in common.GROUP_OF_PRIMARY["all14"]
    }
    report["leave_one_month_out"] = {
        m: {
            k: contrast(
                **with_months(kw=kw, months=tuple(x for x in months if x != m)),
                a=s_star,
                b=b,
                wld="train_family",
                slc="in_force",
            ).point
            for k, b in (("C1", common.ARM_M1), ("C2", "LP"), ("C3", common.ARM_W0))
        }
        for m in months
    }
    report["without_d5"] = {
        k: contrast(**kw, a=s_star, b=b, wld="train_family", slc="in_force", drop=("D5",)).point
        for k, b in (("C1", common.ARM_M1), ("C2", "LP"), ("C3", common.ARM_W0))
    }
    report["eight_with_and_without_d8"] = {
        g: contrast(
            **kw, a=s_star, b=common.ARM_M1, wld="train_family", slc="in_force", group=g
        ).point
        for g in ("eight", "eight_no_d8")
    }
    report["trigger_flips"] = {
        "sign_flips": [
            lab
            for lab, v in report["leave_one_primary_out"].items()
            if any(np.sign(v[k]) != np.sign(report[k]["point"]) for k in ("C1", "C2", "C3"))
        ],
    }
    prims = score.primary_indices(group="all14", labels=labels)
    first = months[0]
    report["scale_factors"] = {
        arm: float(np.mean([tables[(m, s)][arm]["scale_factor"] for m in months for s in seeds]))
        for arm in common.ALL_ARMS
    }
    report["follow_share"] = {}
    for arm in common.FIT_ARMS:
        num, den = (
            sum(
                tables[(m, s)][arm]["train_family"]["far"][key][prims].sum()
                for m in months
                for s in seeds
            )
            for key in ("follow_num", "follow_den")
        )
        report["follow_share"][arm] = float(num / den) if den > 0 else None
    report["departure_share_real"] = {
        arm: float(
            sum(
                tables[(m, s)][arm]["real"]["all"]["departs"][prims].sum()
                for m in months
                for s in seeds
            )
            / sum(
                tables[(m, s)][arm]["real"]["all"]["n"][prims].sum() for m in months for s in seeds
            )
        )
        for arm in common.FIT_ARMS
    }
    report["event_free_with_ended_history_share"] = float(
        sum(
            tables[(m, s)][s_star]["train_family"]["event_free_ended_8_15"]["n"][prims].sum()
            for m in months
            for s in seeds
        )
        / max(
            sum(
                tables[(m, s)][s_star]["train_family"]["event_free"]["n"][prims].sum()
                for m in months
                for s in seeds
            ),
            1,
        )
    )
    del first
    return report


def render(*, report: dict) -> str:
    """Return the report as text: the verdict first, with the V2 flag in its first sentence."""
    lines = []
    flag = " V2 is flagged (its point estimate is above +0.10)." if report["V2_flag"] else ""
    lines.append(f"VERDICT: {report['verdict']} (table row {report['verdict_row']}).{flag}")
    lines.append(f"S* = {report['s_star']}")
    for key in ("C1", "C2", "C3", "V1", "V2", "N1", "N3"):
        r = report[key]
        lines.append(
            f"{key}: point {r['point']:+.3f}, t [{r['t'][0]:+.3f}, {r['t'][1]:+.3f}], "
            f"bootstrap [{r['boot'][0]:+.3f}, {r['boot'][1]:+.3f}], seeds {[round(x, 3) for x in r['seed_points']]}"
        )
    for q, d in report["net_by_q"].items():
        lines.append(f"q={q}: N1 {d['N1']['point']:+.3f}, N3 {d['N3']['point']:+.3f}")
    lines.append(
        "held-out C2, C3: "
        + ", ".join(f"{k} {v['point']:+.3f}" for k, v in report["held_out"].items())
    )
    lines.append(
        "scale factors: " + ", ".join(f"{k} {v:.2f}" for k, v in report["scale_factors"].items())
    )
    lines.append(
        "follow share (far stratum): "
        + ", ".join(f"{k} {v}" for k, v in report["follow_share"].items())
    )
    lines.append(
        "share of real rows where the median departs from M1's by more than 0.5 points: "
        + ", ".join(f"{k} {v:.3f}" for k, v in report["departure_share_real"].items())
    )
    lines.append(
        f"unscreened folds C1 same sign as the 8-fold reading: {report['unscreened_folds']['same_sign_as_8_fold']}"
    )
    lines.append(
        f"share of event-free rows whose history holds an ended shift: {report['event_free_with_ended_history_share']:.3f}"
    )
    lines.append("leave-one-Primary-out sign flips: " + str(report["trigger_flips"]["sign_flips"]))
    return "\n".join(lines)


def load_tables(
    *, months: tuple[str, ...], seeds: tuple[int, ...], out: Path | None = None
) -> dict[tuple[str, int], dict]:
    """Read the tables of sums of every (month, seed)."""
    return {(m, s): score.read_sums(month=m, seed=s, out=out) for m in months for s in seeds}


def write_report(*, report: dict, out: Path | None = None) -> None:
    """Save the report as JSON and text under `scores/`."""
    root = (out if out is not None else common.OUT) / "scores"
    common.atomic_json(obj=report, path=root / "report.json")
    (root / "report.txt").write_text(render(report=report) + "\n")


def audit_triggers(*, report: dict) -> dict[str, bool]:
    """Return which audit triggers of the `ml-experiment` skill the report fires on its face.

    Args:
        report: The output of `compute_report`.

    Returns:
        Trigger 3 (CRPS and median-error contrasts of C1 point in different directions), trigger 4
        (leaving a Primary out flips a counted contrast's sign), and trigger 5 (W0 beats S* in force
        while S* contains no W0, which is not nested, so only the control check applies).
    """
    ro = report["report_only"]
    return {
        "trigger_3_c1": bool(np.sign(report["C1"]["point"]) != np.sign(ro["C1_nmae"]["point"])),
        "trigger_3_c3": bool(np.sign(report["C3"]["point"]) != np.sign(ro["C3_nmae"]["point"])),
        "trigger_4": bool(report["trigger_flips"]["sign_flips"]),
        "trigger_2_prediction_misses": False,
    }


def read_report(*, out: Path | None = None) -> dict:
    """Read the saved report."""
    return json.loads(
        ((out if out is not None else common.OUT) / "scores" / "report.json").read_text()
    )

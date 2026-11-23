# ruff: noqa: E501, PT018
"""Experiment 718, WP7b: the counted contrasts, the report-only readings, the screen rules, and the verdict of each part.

Usage (each command reads stored forecasts, so none is run before the fits exist):
    e00718_contrasts.py screen PART      # the 4 screen folds, seeds 0 and 1, at 95%: stop rule, audit trigger 1, the rule on Z
    e00718_contrasts.py headline PART    # the 8 folds: the two counted contrasts at 98.75%, guards, verdict, sensitivity, clean-fold reading
    e00718_contrasts.py report-only PART # the report-only contrasts of section 4 and 7.3 at 95%, by group, lead band, and Primary
    e00718_contrasts.py self-test        # the whole chain on synthetic forecasts

PART is 1 (A1 = R - M*, A2 = R - R-A) or 2 (B1 = U - M*, B2 = U - U-A). Every contrast is experiment 713's
`contrast` (bootstrap, t guard, seed guard), imported unchanged; the group reading and the verdict are
`e00718_decision`'s. The contrasts are treatment minus reference in points of CRPS, on the headline band.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00713_arms as a713  # ty: ignore[unresolved-import]
import e00713_score as s713  # ty: ignore[unresolved-import]
import e00718_arms as arms
import e00718_decision as dec
import e00718_score as sc
import numpy as np
import polars as pl
from scipy import stats
from studies.sources import ML_EXPERIMENTS_DATA_DIR

KEYS: Final = s713.KEYS
BAND: Final = s713.HEADLINE_BAND_HOURS
COUNTED: Final = {
    1: {"first": ("A1", "R", "STAR"), "second": ("A2", "R", "R-A"), "placebo_arm": "R-A"},
    2: {"first": ("B1", "U", "STAR"), "second": ("B2", "U", "U-A"), "placebo_arm": "U-A"},
}
REPORT_ONLY_PAIRS: Final = {
    1: [
        ("U", "P"),
        ("R", "U"),
        ("R", "P"),
        ("P90", "P"),
        ("R", "P90"),
        ("R-A", "P90"),
        ("R-A", "STAR"),
        ("Z", "STAR"),
        ("R", "Z"),
        ("R", "C"),
        ("STAR", "C"),
    ],
    2: [
        ("U", "P"),
        ("U27", "STAR"),
        ("U27", "U27-A"),
        ("U27", "U"),
        ("UX", "U"),
        ("UX", "STAR"),
        ("U-A", "STAR"),
        ("Z", "STAR"),
        ("U", "Z"),
        ("U", "C"),
        ("STAR", "C"),
    ],
}
BANDS: Final = {
    "24-72 h": (24.0, 72.0),
    "72-168 h": (72.0, 168.0),
    "168-240 h": (168.0, 240.0),
    "headline 24-240 h": BAND,
}


def resolve(*, name: str, star: str) -> str:
    """Return the arm name for `STAR`."""
    return star if name == "STAR" else name


def with_all_group(*, groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the Primary groups plus `all`, the 16 Primaries pooled, and `short`, D15 with D18."""
    out = dict(groups)
    out["all"] = np.concatenate([groups["six"], groups["eight"], groups["other"]])
    out["short"] = groups["other"]
    return out


def counted_contrast(
    *, per_seed: dict, averaged: dict, groups: dict, treatment: str, reference: str, n_seeds: int,
    level: float, band: tuple[float, float] | None = BAND, metric: str = "crps",
) -> dict[str, dict]:  # fmt: skip
    """Return the contrast of each of the six and the eight, with the group reading of `e00718_decision`."""
    stored = per_seed[treatment]["seed"].n_unique()
    assert stored == n_seeds, (
        f"{treatment} holds {stored} seeds, and the reading was asked for {n_seeds}"
    )
    out = {}
    for g in dec.GROUPS:
        c = s713.contrast(
            per_seed=per_seed,
            averaged=averaged,
            treatment=treatment,
            reference=reference,
            group=g,
            groups=groups,
            level=level,
            band=band,
            metric=metric,
        )
        c["floor_guard"] = dec.floor_guard(c=c, group=g, n_seeds=n_seeds)
        c["reading"] = dec.group_reading(c=c, group=g, n_seeds=n_seeds)
        out[g] = c
    return out


def monthly_differences(
    *,
    averaged: dict,
    groups: dict,
    treatment: str,
    reference: str,
    group: str,
    metric: str = "crps",
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the paired differences of a group in the headline band, and their monthly means."""
    sel = {
        n: s713.restrict(frame=averaged[n], group=group, band=BAND, groups=groups)
        for n in (treatment, reference)
    }
    both = s713.paired(treatment=sel[treatment], reference=sel[reference], metric=metric)
    month = both["power_fcst_init_time"].dt.strftime("%Y-%m")
    by_month = (
        both.with_columns(m=month).group_by("m").agg(pl.col("d").mean()).sort("m")["d"].to_numpy()
    )
    return both, by_month


def guards(
    *, averaged: dict, groups: dict, treatment: str, star: str, cold_night_difference: dict[str, float] | None,
    swap_points: dict[str, float] | None,
) -> dict:  # fmt: skip
    """Return the guards of `supported` for the treatment against `M*` in each group (section 7.3).

    A guard that needs an input that has not been read (the cold-night slice, the selection swap) fails.
    """
    out: dict = {}
    for g in dec.GROUPS:
        both, by_month = monthly_differences(
            averaged=averaged, groups=groups, treatment=treatment, reference=star, group=g
        )
        by_id = both.group_by("time_series_id").agg(
            pl.col("d").sum().alias("s"), pl.len().alias("n")
        )
        total, count = float(by_id["s"].sum()), int(by_id["n"].sum())
        full = np.sign(total / count)
        loo_primary = [
            np.sign((total - s_) / (count - n_)) == full
            for s_, n_ in zip(by_id["s"], by_id["n"], strict=True)
        ]
        loo_month = [
            np.sign((by_month.sum() - x) / (len(by_month) - 1)) == np.sign(by_month.mean())
            for x in by_month
        ]
        tails_ok = True
        sel = {
            n: s713.restrict(frame=averaged[n], group=g, band=BAND, groups=groups)
            for n in (treatment, star)
        }
        for col in ("pin_q01", "pin_q99"):
            d = monthly_differences(
                averaged=averaged,
                groups=groups,
                treatment=treatment,
                reference=star,
                group=g,
                metric=col,
            )[1]
            upper = d.mean() + stats.t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            tails_ok &= bool(upper <= dec.TAIL_LIMIT)
        mae = float(
            s713.paired(treatment=sel[treatment], reference=sel[star], metric="abs_err")["d"].mean()
        )  # ty: ignore[invalid-argument-type]
        point = float(both["d"].mean())  # ty: ignore[invalid-argument-type]
        out[g] = dec.supported_guards(
            sign_holds_leave_primary=bool(all(loo_primary)), sign_holds_leave_month=bool(all(loo_month)), tails_ok=tails_ok,
            mae_moves_with_crps=bool(np.sign(mae) == np.sign(point)),
            cold_night_difference=float("inf") if cold_night_difference is None else cold_night_difference[g],
            swap_keeps_sign=None if swap_points is None else bool(np.sign(swap_points[g]) == np.sign(point)),
        )  # fmt: skip
    out["all_hold"] = all(out[g]["all_hold"] for g in dec.GROUPS)
    return out


def headline(
    *, part: int, per_seed: dict, averaged: dict, groups: dict, star: str, n_seeds: int,
    cold_night_difference: dict[str, float] | None = None, swap_points: dict[str, float] | None = None,
) -> dict:  # fmt: skip
    """Return the counted contrasts at 98.75%, the readings, the guards, and the verdict of one part."""
    spec = COUNTED[part]
    (n1, t1, r1), (n2, t2, r2) = spec["first"], spec["second"]
    r1 = resolve(name=r1, star=star)
    placebo = spec["placebo_arm"]
    args = {"per_seed": per_seed, "averaged": averaged, "groups": groups, "n_seeds": n_seeds}
    first = counted_contrast(treatment=t1, reference=r1, level=dec.LEVEL, **args)  # ty: ignore[invalid-argument-type]
    second = counted_contrast(treatment=t2, reference=r2, level=dec.LEVEL, **args)  # ty: ignore[invalid-argument-type]
    placebo_vs_star = counted_contrast(treatment=placebo, reference=star, level=dec.LEVEL, **args)  # ty: ignore[invalid-argument-type]
    g = guards(
        averaged=averaged,
        groups=groups,
        treatment=t1,
        star=star,
        cold_night_difference=cold_night_difference,
        swap_points=swap_points,
    )
    readings = {k: first[k]["reading"] for k in dec.GROUPS}
    word = dec.verdict(
        readings=readings, placebo_upper={k: second[k]["upper"] for k in dec.GROUPS}, placebo_point={k: second[k]["point"] for k in dec.GROUPS},
        placebo_vs_star={k: placebo_vs_star[k]["reading"] for k in dec.GROUPS}, guards_hold=bool(g["all_hold"]),
    )  # fmt: skip
    sens = {
        n: counted_contrast(treatment=t, reference=r, level=dec.SENSITIVITY_LEVEL, **args)  # ty: ignore[invalid-argument-type]
        for n, t, r in ((n1, t1, r1), (n2, t2, r2))
    }
    return {
        "part": part, "n_seeds": n_seeds, "level": dec.LEVEL, n1: first, n2: second, f"{placebo}-STAR": placebo_vs_star,
        "readings": readings, "guards": g, "verdict": word, "unread_guards": [k for k, v in (("cold_nights", cold_night_difference), ("selection_swap", swap_points)) if v is None],
        "sensitivity_99.375": {n: {k: {"point": c[k]["point"], "lower": c[k]["lower"], "upper": c[k]["upper"]} for k in dec.GROUPS} for n, c in sens.items()},
        "predictions": {n: dec.PREDICTIONS[n] for n in (n1,) if n in dec.PREDICTIONS},
    }  # fmt: skip


def screen(
    *, part: int, per_seed: dict, averaged: dict, groups: dict, star: str, z_present: bool = True,
) -> dict:  # fmt: skip
    """Return the screen's stop rule, audit trigger 1, and the rule on `Z`, from the first and second contrasts at 95%."""
    spec = COUNTED[part]
    (n1, t1, r1), (n2, t2, r2) = spec["first"], spec["second"]
    args = {
        "per_seed": per_seed,
        "averaged": averaged,
        "groups": groups,
        "n_seeds": 2,
        "level": dec.REPORT_LEVEL,
    }
    first = counted_contrast(treatment=t1, reference=resolve(name=r1, star=star), **args)  # ty: ignore[invalid-argument-type]
    second = counted_contrast(treatment=t2, reference=r2, **args)  # ty: ignore[invalid-argument-type]
    months = {
        g: monthly_differences(
            averaged=averaged,
            groups=groups,
            treatment=t1,
            reference=resolve(name=r1, star=star),
            group=g,
        )[1]
        for g in dec.GROUPS
    }
    out = {
        "part": part, "interim": True, n1: first, n2: second,
        "stop_rule": dec.stop_rule(first_contrast={g: first[g]["point"] for g in dec.GROUPS}, monthly=months),
        "audit_trigger_1_placebo": dec.audit_trigger_1(placebo_point={g: second[g]["point"] for g in dec.GROUPS}),
    }  # fmt: skip
    out.update(
        nested_and_u27(averaged=averaged, per_seed=per_seed, groups=groups, star=star, args=args)
    )
    if z_present and "Z" in averaged:
        z = counted_contrast(treatment="Z", reference=star, **args)  # ty: ignore[invalid-argument-type]
        out["Z-STAR"] = z
        out["audit_trigger_1_z_rule"] = dec.z_rule_fires(
            z_minus_star={g: z[g]["point"] for g in dec.GROUPS}
        )
    return out


def nested_and_u27(*, averaged: dict, per_seed: dict, groups: dict, star: str, args: dict) -> dict:
    """Return audit trigger 5 (a nested arm beats its container) and the `U27` advance rule, for the arms that are stored."""
    out: dict = {}

    def points(treatment: str, reference: str) -> dict[str, float] | None:
        if treatment not in averaged or reference not in averaged:
            return None
        c = counted_contrast(treatment=treatment, reference=reference, **args)
        return {g: c[g]["point"] for g in dec.GROUPS}

    triggers = {}
    for nested, container in ((star, "U27"), ("R-A", "R"), ("U-A", "U"), ("U27-A", "U27")):
        d = points(nested, container)
        if d is not None:
            triggers[f"{nested}-{container}"] = dec.nested_beats_container(nested_minus_container=d)
    if triggers:
        out["audit_trigger_5"] = triggers
        out["audit_trigger_5_fires"] = any(triggers.values())
    u27_a, u27_star = points("U27", "U27-A"), points("U27", star)
    if u27_a is not None and u27_star is not None:
        out["u27_advances"] = dec.u27_advances(u27_minus_u27a=u27_a, u27_minus_star=u27_star)
    return out


def u27_exception(*, per_seed: dict, averaged: dict, groups: dict) -> bool | None:
    """Return whether `U27` minus `U27-A` is below zero in both groups with its 95% interval; None where `U27-A` is not stored."""
    if "U27" not in averaged or "U27-A" not in averaged:
        return None
    c = counted_contrast(
        per_seed=per_seed, averaged=averaged, groups=groups, treatment="U27", reference="U27-A",
        n_seeds=2, level=dec.REPORT_LEVEL,
    )  # fmt: skip
    return all(c[g]["upper"] < 0 for g in dec.GROUPS)


CHRISTMAS_WINDOWS: Final = {
    "20 December to 5 January": ((12, 20), (1, 5)),
    "24 to 26 December": ((12, 24), (12, 26)),
}


def christmas_reading(*, averaged: dict, star: str, boot: int = 5000) -> dict:
    """Return the report-only Christmas readings: `UX` minus `U` and `UX` minus `M*` on the windows' targets, all 16 Primaries, with a day-block 95% interval.

    One winter: a pointer for a follow-up, not a decision (design section 8).
    """
    out: dict = {
        "label": "report-only, one winter, targets by valid day; an interval from resampling days"
    }
    rng = np.random.default_rng(0)
    for t_raw, r_raw in (("UX", "U"), ("UX", "STAR")):
        t, r = t_raw, resolve(name=r_raw, star=star)
        if t not in averaged or r not in averaged:
            out[f"{t}-{r}"] = "not stored"
            continue
        both = s713.paired(treatment=averaged[t], reference=averaged[r], metric="crps")
        both = (
            both.join(averaged[t].select(*KEYS, "valid_time"), on=list(KEYS))
            if "valid_time" not in both.columns
            else both
        )
        md = pl.col("valid_time").dt.month().cast(pl.Int32) * 100 + pl.col(
            "valid_time"
        ).dt.day().cast(pl.Int32)
        for name, ((m0, d0), (m1, d1)) in CHRISTMAS_WINDOWS.items():
            lo, hi = m0 * 100 + d0, m1 * 100 + d1
            sel = both.filter((md >= lo) | (md <= hi) if lo > hi else (md >= lo) & (md <= hi))
            if sel.height == 0:
                out[f"{t}-{r}, {name}"] = "no targets in the stored folds"
                continue
            days = (
                sel.with_columns(day=pl.col("valid_time").dt.date())
                .group_by("day")
                .agg(pl.col("d").mean())["d"]
                .to_numpy()
            )
            means = rng.choice(days, size=(boot, len(days))).mean(axis=1)
            out[f"{t}-{r}, {name}"] = {
                "point": float(days.mean()), "lower": float(np.quantile(means, 0.025)),
                "upper": float(np.quantile(means, 0.975)), "days": len(days),
            }  # fmt: skip
    return out


def clean_fold_reading(
    *, part: int, per_seed: dict, averaged: dict, groups: dict, star: str
) -> dict:
    """Return the first and second contrasts on the four folds clean of the exploration's selection, at 95%, labelled report-only."""

    def keep(f: pl.DataFrame) -> pl.DataFrame:
        return f.filter(pl.col("power_fcst_init_time").dt.strftime("%Y-%m").is_in(a713.OTHER_FOLDS))

    ps = {k: keep(v) for k, v in per_seed.items()}
    av = {k: keep(v) for k, v in averaged.items()}
    spec = COUNTED[part]
    out = {
        "label": "clean of the exploration's selection (folds 2025-12, 2026-02, 2026-04, 2026-05); report-only"
    }
    for n, t, r in (spec["first"], spec["second"]):
        c = counted_contrast(
            per_seed=ps,
            averaged=av,
            groups=groups,
            treatment=t,
            reference=resolve(name=r, star=star),
            n_seeds=2,
            level=dec.REPORT_LEVEL,
        )
        out[n] = {g: {k: c[g][k] for k in ("point", "lower", "upper")} for g in dec.GROUPS}
    return out


def report_only(
    *, part: int, per_seed: dict, averaged: dict, groups: dict, star: str, labels: dict[int, str] | None = None,
) -> dict:  # fmt: skip
    """Return the report-only contrasts at 95%: by group (six, eight, all 16, D15 with D18), by lead band, and by Primary."""
    wide = with_all_group(groups=groups)
    out: dict = {"level": dec.REPORT_LEVEL, "contrasts": {}, "bands": {}, "per_primary": None}
    for t_raw, r_raw in REPORT_ONLY_PAIRS[part]:
        t, r = resolve(name=t_raw, star=star), resolve(name=r_raw, star=star)
        if t not in averaged or r not in averaged:
            out["contrasts"][f"{t}-{r}"] = "not stored"
            continue
        row = {}
        for g in ("six", "eight", "all", "short"):
            c = s713.contrast(
                per_seed=per_seed,
                averaged=averaged,
                treatment=t,
                reference=r,
                group=g,
                groups=wide,
                level=dec.REPORT_LEVEL,
            )
            mae = s713.contrast(
                per_seed=per_seed,
                averaged=averaged,
                treatment=t,
                reference=r,
                group=g,
                groups=wide,
                level=dec.REPORT_LEVEL,
                metric="abs_err",
            )
            row[g] = {"crps": [c["lower"], c["point"], c["upper"]], "mae_point": mae["point"]}
        out["contrasts"][f"{t}-{r}"] = row
    first_t, first_r = COUNTED[part]["first"][1], resolve(name=COUNTED[part]["first"][2], star=star)
    for name, band in BANDS.items():
        out["bands"][name] = {
            g: counted_contrast(
                per_seed=per_seed,
                averaged=averaged,
                groups=groups,
                treatment=first_t,
                reference=first_r,
                n_seeds=2,
                level=dec.REPORT_LEVEL,
                band=band,
            )[g]["point"]
            for g in dec.GROUPS
        }
    if labels is not None:
        scored = {
            n: averaged[n]
            for n in (first_t, first_r, COUNTED[part]["placebo_arm"])
            if n in averaged
        }
        out["per_primary"] = sc.per_primary_table(
            scored=scored,  # ty: ignore[invalid-argument-type]
            reference=first_r,
            labels=labels,  # ty: ignore[invalid-argument-type]
        ).to_dicts()
    return out


# ------------------------------------------------------------------------------------ commands


def _score_dir(*, part: int) -> Path:
    return arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR, part=part) / "score"


def _write(*, part: int, name: str, obj: object) -> None:
    path = _score_dir(part=part) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)
    print(json.dumps(obj, indent=1, default=float))


def _available(*, arm: str, months: Sequence[str], seeds: Sequence[int]) -> bool:
    """Return whether every forecast file of `arm` for the folds and seeds exists."""
    return all(
        sc.fit_path(arm=arm, month=m, seed=s, data_dir=ML_EXPERIMENTS_DATA_DIR).exists()
        for m in months
        for s in seeds
    )


def _load(
    *, part: int, months: Sequence[str], seeds: Sequence[int], extra: Sequence[str] = (), optional: Sequence[str] = ()
) -> tuple[dict, dict, dict, str]:  # fmt: skip
    """Load the counted arms and `extra` (all required) and those of `optional` that are stored."""
    star = a713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    spec = COUNTED[part]
    names = {star, "P", "C", spec["first"][1], spec["second"][2], spec["placebo_arm"], *extra}
    names |= {a for a in optional if _available(arm=a, months=months, seeds=seeds)}
    per_seed, averaged, groups = sc.load_all(names=sorted(names), months=months, seeds=seeds)  # ty: ignore[invalid-argument-type]
    return per_seed, averaged, groups, star


LAST: dict = {}
"""The dicts of the last self-test, for the report's own self-test."""


def self_test() -> int:  # noqa: PLR0915
    """Run the screen, the headline, the clean-fold reading, and the report-only readings on synthetic scored forecasts."""
    rng = np.random.default_rng(0)
    months = [f"2025-{m:02d}" for m in (11, 12)] + [f"2026-{m:02d}" for m in range(1, 7)]
    ids = {"six": np.arange(0, 6), "eight": np.arange(6, 14), "other": np.arange(14, 16)}
    s713.N_RESAMPLES = 2000
    rows = []
    for mi, month in enumerate(months):
        for tid in range(16):
            for day in (2, 3, 4, 5, 6, 7, 24):
                for hr in (30, 100, 200):
                    t0 = datetime.fromisoformat(f"{month}-{day:02d}T12:00:00")
                    rows.append(
                        (tid, t0, t0 + timedelta(hours=hr), float(hr), mi, float(rng.normal()))
                    )
    base = pl.DataFrame(
        rows,
        schema=[
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            "mi",
            "noise",
        ],
        orient="row",
    )
    base = base.with_columns(
        pl.col("time_series_id").cast(pl.Int64),
        pl.col("power_fcst_init_time").cast(pl.Datetime("us")),
        pl.col("valid_time").cast(pl.Datetime("us")),
    )
    effect = {
        "STAR": 0.0,
        "P": 0.3,
        "C": 0.5,
        "R": -0.25,
        "R-A": -0.02,
        "Z": 0.0,
        "P90": 0.1,
        "U": -0.1,
        "U-A": 0.0,
        "U27": -0.1,
        "U27-A": 0.0,
        "UX": -0.3,
    }
    star = "M1"
    effect[star] = 0.0

    def make(arm: str, seeds: Sequence[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
        parts = []
        for s in seeds:
            sd = np.random.default_rng(100 * s + hash(arm) % 97).normal(size=base.height) * 0.01
            crps = 2.0 + effect.get(arm, 0.0) + 0.0 * base["noise"].to_numpy() + sd
            parts.append(
                base.with_columns(
                    seed=pl.lit(s, dtype=pl.Int64),
                    crps=pl.Series(crps),
                    abs_err=pl.Series(crps * 1.5),
                    pin_q01=pl.Series(crps * 0.1),
                    pin_q99=pl.Series(crps * 0.1),
                )
            )
        per = pl.concat(parts)
        avg = (
            per.group_by(list(KEYS))
            .agg(
                pl.col("lead_hours").first(),
                pl.col("crps").mean(),
                pl.col("abs_err").mean(),
                pl.col("pin_q01").mean(),
                pl.col("pin_q99").mean(),
            )
            .with_columns(seed=pl.lit(0, dtype=pl.Int64))
        )
        return per, avg

    names = [star, "P", "C", "R", "R-A", "Z", "P90", "U", "U-A", "U27", "U27-A", "UX"]
    per_seed, averaged = {}, {}
    for n in names:
        per_seed[n], averaged[n] = make(n, (0, 1))
    sc_out = screen(part=1, per_seed=per_seed, averaged=averaged, groups=ids, star=star)
    assert sc_out["stop_rule"] is False and sc_out["audit_trigger_1_placebo"] is False
    assert sc_out["audit_trigger_5_fires"] is False and sc_out["u27_advances"] is True, sc_out
    assert set(sc_out["audit_trigger_5"]) == {"M1-U27", "R-A-R", "U-A-U", "U27-A-U27"}
    assert u27_exception(per_seed=per_seed, averaged=averaged, groups=ids) is True
    xmas = christmas_reading(averaged=averaged, star=star)
    key = "UX-U, 20 December to 5 January"
    assert xmas[key]["point"] < -0.15 and xmas[key]["upper"] < 0 and xmas[key]["days"] >= 6, xmas
    assert xmas["UX-M1, 24 to 26 December"]["days"] >= 1 and xmas[key]["days"] <= 20, xmas
    closed = dec.line_closure(
        verdicts={"part1": "not supported", "part2": "refuted"},
        first_points={q: dict.fromkeys(dec.GROUPS, 0.1) for q in ("part1", "part2")},
        u27_exception=u27_exception(per_seed=per_seed, averaged=averaged, groups=ids),
    )
    assert closed["closed"] is False, "a U27 exception keeps the line open"
    assert sc_out["audit_trigger_1_z_rule"] is False
    h = headline(
        part=1,
        per_seed=per_seed,
        averaged=averaged,
        groups=ids,
        star=star,
        n_seeds=2,
        cold_night_difference={"six": 0.0, "eight": 0.0},
        swap_points={"six": -0.2, "eight": -0.2},
    )
    assert h["A1"]["six"]["point"] < -0.2, h["A1"]["six"]["point"]
    assert h["readings"]["six"] in {"gain", "small gain"}, h["readings"]
    assert h["verdict"] in {"supported", "partly supported", "small gain"}, h["verdict"]
    h2 = headline(part=1, per_seed=per_seed, averaged=averaged, groups=ids, star=star, n_seeds=2)
    assert h2["verdict"] != "supported", (
        "a verdict of `supported` without the cold-night and swap guards"
    )
    assert set(h2["unread_guards"]) == {"cold_nights", "selection_swap"}
    clean = clean_fold_reading(part=1, per_seed=per_seed, averaged=averaged, groups=ids, star=star)
    assert "clean of the exploration" in clean["label"]
    ro = report_only(
        part=1,
        per_seed=per_seed,
        averaged=averaged,
        groups=ids,
        star=star,
        labels={i: f"D{i}" for i in range(16)},
    )
    assert "R-STAR" not in ro["contrasts"] and ro["contrasts"]["R-P90"]["all"]["crps"][1] < 0
    assert len(ro["per_primary"]) == 3 * 16
    LAST.update(headline=h, screen=sc_out, report_only=ro, clean=clean)
    h_two = headline(part=2, per_seed=per_seed, averaged=averaged, groups=ids, star=star, n_seeds=2)
    assert h_two["B1"]["six"]["point"] < 0 and "U-A-STAR" in h_two, (
        "part 2 reads B1 = U - M* and the placebo U-A - M*"
    )
    print(
        f"synthetic part 1: A1 six {h['A1']['six']['point']:+.2f} reads {h['readings']['six']}, verdict {h['verdict']}; without the cold-night and swap guards: {h2['verdict']}"
    )
    print(
        "e00718_contrasts: all checks passed (screen, headline, guards, clean-fold reading, report-only readings)"
    )
    return 0


def closure(*, part: int) -> None:
    """Write the line closure (design section 7.4) from the two parts' headline records and the `U27` exception."""
    heads = {
        f"part{q}": json.loads((_score_dir(part=q) / "headline.json").read_text()) for q in (1, 2)
    }
    firsts = {
        f"part{q}": {g: heads[f"part{q}"][COUNTED[q]["first"][0]][g]["point"] for g in dec.GROUPS}
        for q in (1, 2)
    }
    months = a713.ALL_FOLDS
    if not _available(arm="U27-A", months=months, seeds=arms.SCREEN_SEEDS):
        months = a713.SCREEN_FOLDS
    per_seed, averaged, groups, _ = _load(
        part=2, months=months, seeds=arms.SCREEN_SEEDS, optional=("U27", "U27-A")
    )
    exception = u27_exception(per_seed=per_seed, averaged=averaged, groups=groups)
    _write(
        part=part,
        name="closure",
        obj=dec.line_closure(
            verdicts={k: v["verdict"] for k, v in heads.items()},
            first_points=firsts,
            u27_exception=exception,
        ),
    )


def main() -> int:
    """Run a command."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["screen", "headline", "report-only", "closure", "christmas", "self-test"],
    )
    parser.add_argument("part", nargs="?", type=int, choices=[1, 2])
    parser.add_argument(
        "--seeds", type=int, default=2, help="seeds per counted arm of the headline: 2 or 5"
    )
    args = parser.parse_args()
    if args.command == "self-test":
        return self_test()
    assert args.part is not None
    part = args.part
    if args.command == "screen":
        per_seed, averaged, groups, star = _load(
            part=part, months=a713.SCREEN_FOLDS, seeds=arms.SCREEN_SEEDS, extra=("Z",),
            optional=("R", "R-A", "U", "U-A", "U27", "U27-A"),
        )  # fmt: skip
        _write(
            part=part,
            name="screen",
            obj=screen(part=part, per_seed=per_seed, averaged=averaged, groups=groups, star=star),
        )
    elif args.command == "headline":
        seeds = tuple(range(args.seeds))
        per_seed, averaged, groups, star = _load(part=part, months=a713.ALL_FOLDS, seeds=seeds)
        _write(
            part=part,
            name="headline",
            obj=headline(
                part=part,
                per_seed=per_seed,
                averaged=averaged,
                groups=groups,
                star=star,
                n_seeds=args.seeds,
            ),
        )
        _write(
            part=part,
            name="clean_folds",
            obj=clean_fold_reading(
                part=part, per_seed=per_seed, averaged=averaged, groups=groups, star=star
            ),
        )
    elif args.command == "report-only":
        named = {n for pair in REPORT_ONLY_PAIRS[part] for n in pair} - {"STAR"}
        per_seed, averaged, groups, star = _load(
            part=part, months=a713.ALL_FOLDS, seeds=arms.SCREEN_SEEDS, optional=sorted(named)
        )
        sites = s713.k578.sites()  # ty: ignore[unresolved-attribute]
        labels = dict(zip(sites["time_series_id"].to_list(), sites["label"].to_list(), strict=True))
        _write(
            part=part,
            name="report_only",
            obj=report_only(
                part=part, per_seed=per_seed, averaged=averaged, groups=groups, star=star,
                labels=labels,
            ),
        )  # fmt: skip
    elif args.command == "christmas":
        months = ("2025-12", "2026-01")
        _, averaged, _, star = _load(
            part=2, months=months, seeds=arms.SCREEN_SEEDS, optional=("U", "UX")
        )
        _write(part=2, name="christmas", obj=christmas_reading(averaged=averaged, star=star))
    else:
        closure(part=part)
    return 0


if __name__ == "__main__":
    sys.exit(main())

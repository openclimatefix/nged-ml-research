# ruff: noqa: E501, PLR0911
"""Experiment 715: the counted contrasts N1 and N2, the guards, the decision rule, and the report-only readings.

Usage:
    e00715_score.py selftest       # the decision rule on constructed readings (no data)
    e00715_score.py first          # first run on the stored `M*` and `C` alone: the table of experiment 713's section 1.5
    e00715_score.py headline       # N1 and N2 at 98.75%, the guards, the readings, the verdict, the noise F4 actually has
    e00715_score.py report-only    # the 95% report-only readings of section 7.1 and the KN8 branch of section 8.2

Every table is printed and written to `score/<command>.json` in the experiment's data folder. All losses
are in points of capacity, as in experiment 713's scorer, which this script imports: `contrast`,
`load_arm`'s conversion to the contract origin, `add_losses`, `seed_average`, and `paired`. **Only the
constants and the two rule functions differ from experiment 713's:** the noise floor is 0.054 at the six
and 0.142 at the eight (2 seeds), `delta` is 0.094 and 0.25, `tau` is 0.03, and row 5 of the group table
requires `L <= 0 <= U`. The module's constants in experiment 713's scorer are replaced at import time,
so `contrast` reads F4's.

**Contrasts, treatment minus reference.** N1 is `F4` minus `M*`, N2 is `F4` minus `F4-R`, each on experiment
713's headline band (24 h up to, but not including, 240 h), the 8 folds, seeds 0 and 1, for the six and the
eight, at 98.75%. Every contrast needs the arms to hold identical keys, which `assert_same_keys` asserts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import e00713_arms as a713  # ty: ignore[unresolved-import]
import e00713_score as sc  # ty: ignore[unresolved-import]
import e00715_arms as arms
import e00715_neighbours as nb
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

TAU: Final = 0.03
DELTA: Final = {"six": 0.094, "eight": 0.25}
NOISE_FLOOR: Final = {"six": 0.054, "eight": 0.142}
"""Twice the seed-noise standard error at 2 seeds (design section 7.1)."""
LEVEL: Final = sc.LEVEL
SEEDS: Final = (0, 1)
FIVE_SEEDS: Final = (0, 1, 2, 3, 4)
SCORE_DIR: Final[Path] = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR) / "score"
BANDS: Final = {"24-72 h": (24.0, 72.0), "72-144 h": (72.0, 144.0), "144-240 h": (144.0, 240.0)}
PINBALL_TOLERANCE: Final = 0.05
DONOR_HARM_LIMIT: Final = 0.10
KN8_TOLERANCE: Final = 1e-3

sc.NOISE_FLOOR.clear()
sc.NOISE_FLOOR.update(NOISE_FLOOR)


def group_ids() -> dict[str, np.ndarray]:
    """Return the `time_series_id`s of the six, the eight, the other two, and all 16 Primaries."""
    groups = sc.group_ids()
    groups["all16"] = np.concatenate(list(groups.values()))
    return groups


def forecast_path(*, arm: str, month: str, seed: int) -> Path:
    """Return the per-seed forecast file of an arm: F4's own, the Candidate's (`C`), or experiment 713's."""
    if arm == "C":
        return sc.candidate_files(month=month, seed=seed)
    if arm in arms.ARMS:
        return arms.fit_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed)
    return a713.fit_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed)


def load_arm(*, arm: str, seeds: tuple[int, ...], months: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm's per-seed rows in contract origin and capacity shares, with a `seed` column."""
    parts = []
    for month in months:
        for seed in seeds:
            path = forecast_path(arm=arm, month=month, seed=seed)
            assert path.exists(), (
                f"{path} is missing: arm {arm} has no forecast for {month} s{seed}"
            )
            frame = pl.read_parquet(path, columns=[*sc.KEYS, *sc.QCOLS]).with_columns(
                seed=pl.lit(seed, dtype=pl.Int64)
            )
            parts.append(sc.to_contract_shares(frame=frame))
    frame = pl.concat(parts)
    assert frame.select(*sc.KEYS, "seed").is_duplicated().sum() == 0, (
        f"arm {arm!r} holds a duplicated (key, seed) row"
    )
    assert frame["seed"].n_unique() == len(seeds), (
        f"arm {arm!r} does not hold every seed of {seeds}"
    )
    return frame


def star() -> str:
    """Return the selected screen arm of experiment 713."""
    return a713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)


def group_reading(*, c: dict, group: str) -> str:
    """Return the group's reading of N1 by the first row of the table of design section 7.6 that holds."""
    d, lo, hi = c["point"], c["lower"], c["upper"]
    guards = bool(c["t_guard"] and c["seed_guard"] and c["floor_guard"])
    if lo > TAU and guards:
        return "clear loss"
    if lo > 0 and guards:
        return "loss"
    if hi < 0 and d <= -TAU and guards:
        return "gain"
    if hi < 0 and c["seed_guard"]:
        return "small gain"
    if lo <= 0 <= hi and lo > -DELTA[group] and hi < DELTA[group]:
        return "no detectable difference"
    return "cannot tell"


def verdict(*, readings: dict[str, str], n2_upper: dict[str, float], guards_hold: bool) -> str:
    """Return the experiment's verdict by the first row of the verdict table of design section 7.6."""
    r = list(readings.values())
    if "clear loss" in r:
        return "refuted"
    if "loss" in r or all(x == "no detectable difference" for x in r):
        return "not supported"
    if all(x == "gain" for x in r):
        if all(u < 0 for u in n2_upper.values()) and guards_hold:
            return "supported"
        return "partly supported"
    if "gain" in r and "loss" not in r:
        return "partly supported"
    if "small gain" in r and "gain" not in r and "loss" not in r:
        return "small gain"
    return "inconclusive"


def pair(
    *,
    treatment: str,
    reference: str,
    months: tuple[str, ...],
    seeds_t: tuple[int, ...] = SEEDS,
    seeds_r: tuple[int, ...] = SEEDS,
) -> tuple[dict, dict]:
    """Return the per-seed and seed-averaged scored rows of two arms on `months`, with the keys asserted equal."""
    truth = sc.truth_stack()
    per_seed, averaged = {}, {}
    for name, seeds in ((treatment, seeds_t), (reference, seeds_r)):
        raw = load_arm(arm=name, seeds=seeds, months=months)
        per_seed[name] = sc.add_losses(frame=raw, truth=truth)
        averaged[name] = sc.add_losses(frame=sc.seed_average(frame=raw), truth=truth)
    sc.assert_same_keys(frames=averaged)
    return per_seed, averaged


def read_pair(
    *,
    treatment: str,
    reference: str,
    months: tuple[str, ...],
    level: float,
    groups: dict[str, np.ndarray],
    group_names: tuple[str, ...] = ("six", "eight"),
    seeds_t: tuple[int, ...] = SEEDS,
    seeds_r: tuple[int, ...] = SEEDS,
    band: tuple[float, float] | None = sc.HEADLINE_BAND_HOURS,
) -> dict:
    """Return the contrasts of one pair for the groups in `group_names`."""
    per_seed, averaged = pair(
        treatment=treatment, reference=reference, months=months, seeds_t=seeds_t, seeds_r=seeds_r
    )
    out = {}
    for g in group_names:
        out[g] = sc.contrast(
            per_seed=per_seed,
            averaged=averaged,
            treatment=treatment,
            reference=reference,
            group=g,
            groups=groups,
            level=level,
            band=band,
        )
        for metric in ["abs_err"]:
            out[f"{g}:{metric}"] = sc.contrast(
                per_seed=per_seed,
                averaged=averaged,
                treatment=treatment,
                reference=reference,
                group=g,
                groups=groups,
                level=level,
                band=band,
                metric=metric,
            )["point"]
    out["_frames"] = (per_seed, averaged)  # ty: ignore[invalid-assignment]
    return out


def leave_one_out(
    *, averaged: dict, treatment: str, reference: str, group: str, groups: dict[str, np.ndarray]
) -> dict:
    """Return whether the sign of the contrast holds with each Primary, and each month, left out in turn."""
    sel = {
        n: sc.restrict(frame=averaged[n], group=group, band=sc.HEADLINE_BAND_HOURS, groups=groups)
        for n in (treatment, reference)
    }
    both = sc.paired(treatment=sel[treatment], reference=sel[reference], metric="crps")
    point = float(both["d"].mean())
    months = both["power_fcst_init_time"].dt.strftime("%Y-%m")
    by_month = {
        m: float(both.filter(months != m)["d"].mean()) for m in sorted(months.unique().to_list())
    }
    by_primary = {
        int(t): float(both.filter(pl.col("time_series_id") != t)["d"].mean()) for t in groups[group]
    }
    sign = np.sign(point)
    return {
        "point": point, "by_month": by_month, "by_primary": by_primary,
        "sign_holds_without_each_month": all(np.sign(v) == sign for v in by_month.values()),
        "sign_holds_without_each_primary": all(np.sign(v) == sign for v in by_primary.values()),
    }  # fmt: skip


def tail_pinball(
    *, averaged: dict, treatment: str, reference: str, group: str, groups: dict[str, np.ndarray]
) -> dict:
    """Return the difference in mean q01 and q99 pinball loss, treatment minus reference."""
    out = {}
    for col in ("pin_q01", "pin_q99"):
        sel = {
            n: sc.restrict(
                frame=averaged[n], group=group, band=sc.HEADLINE_BAND_HOURS, groups=groups
            )
            for n in (treatment, reference)
        }
        out[col] = float(
            sc.paired(treatment=sel[treatment], reference=sel[reference], metric=col)["d"].mean()
        )
    out["within_tolerance"] = all(abs(v) <= PINBALL_TOLERANCE for v in out.values())
    return out


def seed_noise(
    *, per_seed: dict, groups: dict[str, np.ndarray], treatment: str, reference: str
) -> dict:
    """Return the monthly standard deviation of seed 0 minus seed 1 for each arm and for the paired contrast (design section 7.2)."""
    out: dict = {}
    for g in ("six", "eight"):
        monthly = {}
        for name in (treatment, reference):
            sel = sc.restrict(
                frame=per_seed[name], group=g, band=sc.HEADLINE_BAND_HOURS, groups=groups
            )
            m = (
                sel.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
                .group_by("month", "seed")
                .agg(pl.col("crps").mean())
                .pivot(on="seed", index="month", values="crps")
                .sort("month")
            )
            monthly[name] = m["0"].to_numpy() - m["1"].to_numpy()
        out[g] = {name: float(np.std(v, ddof=1)) for name, v in monthly.items()}
        out[g]["contrast_seed0_minus_seed1"] = float(
            np.std(monthly[treatment] - monthly[reference], ddof=1)
        )
    return out


def cmd_headline() -> int:
    """N1 and N2 at 98.75%, the guards, the readings, the verdict, and what F4 actually has for seed noise."""
    groups = group_ids()
    m_star = star()
    n1 = read_pair(
        treatment="F4", reference=m_star, months=arms.ALL_FOLDS, level=LEVEL, groups=groups
    )
    n2 = read_pair(
        treatment="F4", reference="F4-R", months=arms.ALL_FOLDS, level=LEVEL, groups=groups
    )
    readings = {g: group_reading(c=n1[g], group=g) for g in ("six", "eight")}
    n2_upper = {g: n2[g]["upper"] for g in ("six", "eight")}
    averaged = n1["_frames"][1]  # ty: ignore[not-subscriptable]
    guards = {
        g: {
            "leave_one_out": leave_one_out(averaged=averaged, treatment="F4", reference=m_star, group=g, groups=groups),
            "tails": tail_pinball(averaged=averaged, treatment="F4", reference=m_star, group=g, groups=groups),
            "mae_moves_with_crps": bool(np.sign(n1[f"{g}:abs_err"]) == np.sign(n1[g]["point"])),
        }
        for g in ("six", "eight")
    }  # fmt: skip
    s6_path = SCORE_DIR / "s6_donor_harm.json"
    donor = json.loads(s6_path.read_text())["f4_minus_mstar"] if s6_path.exists() else None
    guards["donor_after_temporary_moves"] = {  # ty: ignore[invalid-assignment]
        "s6_f4_minus_mstar": donor,
        "within_limit": None if donor is None else donor <= DONOR_HARM_LIMIT,
    }
    guards_hold = bool(
        all(g["leave_one_out"]["sign_holds_without_each_month"] and g["leave_one_out"]["sign_holds_without_each_primary"] and g["tails"]["within_tolerance"] and g["mae_moves_with_crps"] for g in (guards["six"], guards["eight"]))  # ty: ignore[not-subscriptable]
        and guards["donor_after_temporary_moves"]["within_limit"]
    )  # fmt: skip
    noise = seed_noise(per_seed=n1["_frames"][0], groups=groups, treatment="F4", reference=m_star)  # ty: ignore[not-subscriptable]
    result = {
        "star": m_star, "level": LEVEL, "delta": DELTA, "tau": TAU, "noise_floor": NOISE_FLOOR,
        "N1": {g: dict(n1[g]) for g in ("six", "eight")},
        "N2": {g: dict(n2[g]) for g in ("six", "eight")},
        "readings": readings, "n2_upper": n2_upper, "guards": guards, "supported_guards_hold": guards_hold,
        "verdict": verdict(readings=readings, n2_upper=n2_upper, guards_hold=guards_hold), "seed_noise": noise,
        "note": "the donor guard is pending until `e00715_era.py` has written score/s6_donor_harm.json" if donor is None else "",
        "stop_rule": "a verdict of `not supported` or `refuted` closes neighbour rows in the axial-attention network on the 16-Primary panel; a retest needs the fleet-scale adjacency",
    }  # fmt: skip
    print_shares(result=result)
    write(name="headline", obj=result)
    return 0


def print_shares(*, result: dict) -> None:
    """Print the visible share of the neighbour tokens per fold and the two-link share of `F4-R` (design sections 2 and 9)."""
    table = nb.load_table()
    rows = {}
    for month in arms.ALL_FOLDS:
        shares = []
        for seed in SEEDS:
            path = arms.log_file(
                data_dir=ML_EXPERIMENTS_DATA_DIR, arm="F4", month=month, seed=seed, kind="predict"
            )
            if path.exists():
                shares.append(json.loads(path.read_text()).get("visible_share"))
        rows[month] = shares
    result["visible_neighbour_share_by_fold"] = rows
    two = {}
    for seed in SEEDS:
        per_host = [
            nb.two_link_share(
                table=table,
                host=h,
                slots=nb.slots_for(table=table, host=h, mode="rewired", seed=seed),
            )
            for h in range(nb.N_PRIMARIES)
        ]
        two[seed] = float(np.nanmean(per_host))
    result["two_link_share_of_the_rewired_slots"] = two
    print("visible share of the neighbour tokens by fold:", rows)
    print("share of the rewired slots that are two links from the host, by seed:", two)


def cmd_report_only() -> int:
    """The 95% report-only readings of section 7.1 and the KN8 branch of section 8.2."""
    groups = group_ids()
    m_star = star()
    out: dict = {}
    jobs = [
        ("F4-minus-Mstar_5seeds", "F4", m_star, arms.ALL_FOLDS, SEEDS, FIVE_SEEDS),
        ("F4-minus-F4-0", "F4", "F4-0", arms.SCREEN5_FOLDS, SEEDS, SEEDS),
        ("F4-0-minus-Mstar", "F4-0", m_star, arms.SCREEN5_FOLDS, SEEDS, SEEDS),
        ("F4-R-minus-Mstar", "F4-R", m_star, arms.ALL_FOLDS, SEEDS, SEEDS),
        ("F4-minus-F4-T", "F4", "F4-T", arms.SCREEN5_FOLDS, SEEDS, SEEDS),
        ("F4E-minus-F4", "F4E", "F4", arms.SCREEN5_FOLDS, SEEDS, SEEDS),
        ("F4-minus-C", "F4", "C", arms.ALL_FOLDS, SEEDS, SEEDS),
    ]
    for name, t, r, months, st, sr in jobs:
        try:
            res = read_pair(
                treatment=t,
                reference=r,
                months=months,
                level=95.0,
                groups=groups,
                group_names=("six", "eight", "other", "all16"),
                seeds_t=st,
                seeds_r=sr,
            )
        except AssertionError as err:
            out[name] = {"missing": str(err)}
            print(name, "missing:", err)
            continue
        out[name] = {
            g: {k: v for k, v in res[g].items() if k != "monthly"}
            for g in ("six", "eight", "other", "all16")
        }
        if name == "F4-minus-Mstar_5seeds":
            per_seed, averaged = res["_frames"]  # ty: ignore[not-subscriptable]
            out[name]["by_lead_band"] = {
                b: {
                    g: sc.contrast(
                        per_seed=per_seed,
                        averaged=averaged,
                        treatment=t,
                        reference=r,
                        group=g,
                        groups=groups,
                        level=95.0,
                        band=band,
                    )["point"]
                    for g in ("six", "eight")
                }
                for b, band in BANDS.items()
            }
            out[name]["by_primary"] = {
                nb.LABELS[p]: _primary_point(
                    averaged=averaged,
                    treatment=t,
                    reference=r,
                    time_series_id=int(groups["all16"][p]),
                )
                for p in range(nb.N_PRIMARIES)
            }
    kn8 = SCORE_DIR / "kn8.json"
    out["kn8"] = json.loads(kn8.read_text()) if kn8.exists() else "KN8 has not been run"
    if (
        "F4-0-minus-Mstar" in out
        and "missing" not in out["F4-0-minus-Mstar"]
        and "F4-minus-Mstar_5seeds" in out
    ):
        for g in ("six", "eight"):
            ratio = abs(out["F4-0-minus-Mstar"][g]["point"]) / max(
                abs(out["F4-minus-Mstar_5seeds"][g]["point"]), 1e-9
            )
            out[f"control_noise_over_n1_{g}"] = ratio
            out[f"control_is_f4_0_{g}"] = bool(ratio > 0.5)
    write(name="report_only", obj=out)
    return 0


def _primary_point(*, averaged: dict, treatment: str, reference: str, time_series_id: int) -> float:
    """Return the mean headline-band CRPS difference of one Primary."""
    sel = {
        n: averaged[n].filter(
            (pl.col("time_series_id") == time_series_id)
            & (pl.col("lead_hours") >= sc.HEADLINE_BAND_HOURS[0])
            & (pl.col("lead_hours") < sc.HEADLINE_BAND_HOURS[1])
        )
        for n in (treatment, reference)
    }
    return float(
        sc.paired(treatment=sel[treatment], reference=sel[reference], metric="crps")["d"].mean()
    )


def cmd_first() -> int:
    """First run, on the stored `M*` and `C` alone: the pooled CRPS of each over the headline band, which sets the scale of every contrast."""
    groups = group_ids()
    m_star = star()
    truth = sc.truth_stack()
    out = {}
    for name in (m_star, "C"):
        raw = load_arm(arm=name, seeds=SEEDS, months=arms.ALL_FOLDS)
        scored = sc.add_losses(frame=sc.seed_average(frame=raw), truth=truth)
        out[name] = {
            g: float(
                sc.restrict(frame=scored, group=g, band=sc.HEADLINE_BAND_HOURS, groups=groups)[
                    "crps"
                ].mean()
            )
            for g in ("six", "eight", "other")
        }
    write(name="first", obj=out)
    return 0


def write(*, name: str, obj: object) -> None:
    """Write `score/<name>.json` atomically and print it."""
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    path = SCORE_DIR / f"{name}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)
    print(json.dumps(obj, indent=1, default=float))


def selftest() -> int:
    """Check the decision rule on constructed readings."""

    def c(
        *,
        point: float,
        lower: float,
        upper: float,
        guards: bool = True,
        seed_guard: bool | None = None,
    ) -> dict:
        return {
            "point": point,
            "lower": lower,
            "upper": upper,
            "t_guard": guards,
            "seed_guard": guards if seed_guard is None else seed_guard,
            "floor_guard": guards,
        }

    assert group_reading(c=c(point=0.2, lower=0.1, upper=0.3), group="six") == "clear loss"
    assert group_reading(c=c(point=0.05, lower=0.01, upper=0.1), group="six") == "loss"
    assert group_reading(c=c(point=-0.2, lower=-0.3, upper=-0.1), group="six") == "gain"
    assert (
        group_reading(
            c=c(point=-0.02, lower=-0.05, upper=-0.01, guards=False, seed_guard=True), group="six"
        )
        == "small gain"
    )
    assert (
        group_reading(c=c(point=0.0, lower=-0.05, upper=0.05, guards=False), group="six")
        == "no detectable difference"
    )
    assert (
        group_reading(c=c(point=0.0, lower=-0.1, upper=0.05, guards=False), group="six")
        == "cannot tell"
    )  # L <= -delta
    assert (
        group_reading(c=c(point=0.0, lower=-0.05, upper=0.2, guards=False), group="six")
        == "cannot tell"
    )  # U >= delta
    assert (
        group_reading(c=c(point=0.1, lower=0.02, upper=0.2, guards=False), group="six")
        == "cannot tell"
    )  # excludes zero, a guard fails
    assert (
        group_reading(c=c(point=0.0, lower=0.0, upper=0.05, guards=False), group="six")
        == "no detectable difference"
    )  # L = 0 is row 5
    nd = "no detectable difference"
    assert (
        verdict(
            readings={"six": "clear loss", "eight": "gain"},
            n2_upper={"six": -1.0, "eight": -1.0},
            guards_hold=True,
        )
        == "refuted"
    )
    assert (
        verdict(
            readings={"six": nd, "eight": nd}, n2_upper={"six": 0.0, "eight": 0.0}, guards_hold=True
        )
        == "not supported"
    )
    assert (
        verdict(
            readings={"six": "loss", "eight": "gain"},
            n2_upper={"six": 0.0, "eight": 0.0},
            guards_hold=True,
        )
        == "not supported"
    )
    assert (
        verdict(
            readings={"six": "gain", "eight": "gain"},
            n2_upper={"six": -0.1, "eight": -0.1},
            guards_hold=True,
        )
        == "supported"
    )
    assert (
        verdict(
            readings={"six": "gain", "eight": "gain"},
            n2_upper={"six": 0.1, "eight": -0.1},
            guards_hold=True,
        )
        == "partly supported"
    )
    assert (
        verdict(
            readings={"six": "gain", "eight": "gain"},
            n2_upper={"six": -0.1, "eight": -0.1},
            guards_hold=False,
        )
        == "partly supported"
    )
    assert (
        verdict(
            readings={"six": "gain", "eight": nd},
            n2_upper={"six": 0.0, "eight": 0.0},
            guards_hold=True,
        )
        == "partly supported"
    )
    assert (
        verdict(
            readings={"six": "small gain", "eight": nd},
            n2_upper={"six": 0.0, "eight": 0.0},
            guards_hold=True,
        )
        == "small gain"
    )
    assert (
        verdict(
            readings={"six": "cannot tell", "eight": nd},
            n2_upper={"six": 0.0, "eight": 0.0},
            guards_hold=True,
        )
        == "inconclusive"
    )
    assert (
        verdict(
            readings={"six": "small gain", "eight": "gain"},
            n2_upper={"six": 0.0, "eight": 0.0},
            guards_hold=True,
        )
        == "partly supported"
    )
    assert sc.NOISE_FLOOR == NOISE_FLOOR, sc.NOISE_FLOOR
    print("score self-test passed (decision rule on constructed readings)")
    return 0


def main() -> int:
    """Run one command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["selftest", "first", "headline", "report-only"])
    args = parser.parse_args()
    return {
        "selftest": selftest,
        "first": cmd_first,
        "headline": cmd_headline,
        "report-only": cmd_report_only,
    }[args.command]()


if __name__ == "__main__":
    sys.exit(main())

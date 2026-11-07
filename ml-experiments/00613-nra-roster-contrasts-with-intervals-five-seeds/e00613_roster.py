# ruff: noqa: D103, E501, PERF401
"""Experiment 613: the three families of roster contrasts, each with a Bonferroni interval (CPU).

Families (every contrast is a paired difference over matched cells, resampled with experiment 612's
bootstrap; `mde` is half the interval width; a contrast is "equivalent" only if its whole interval
lies inside the tolerance):

- `class2`: the class 2 candidate at K1's band width against K1 (coverage, follow, safe-side share,
  injected phases; event-free coverage) and against K2 (event-free CRPS), five months, plus the
  absorption contrast against K1 and K2 (experiment 599's E3 arm);
- `o2`: O2 against D5 in K1, K2, and K3 with tail constants on both, four months with constants;
- `e3`: the 3-day embargo against E3 (experiment 599's E0 against E3 for absorption), five months.

Usage: python e00613_roster.py [--suffix _s01|_s234] (the suffix selects a score folder of a seed subset)
Writes `tables_out/roster_<family><suffix>.csv` in this experiment's private folder.
"""

import argparse

import e00595_summary as s595  # ty: ignore[unresolved-import]
import e00613_tables as t
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

_p = argparse.ArgumentParser()
_p.add_argument("--suffix", default="")
MINE = _p.parse_args()
SFX = MINE.suffix
OUTDIR = t.ROOT / "tables_out"
OUTDIR.mkdir(exist_ok=True)
ALL5 = None
TAIL = ["2025-12", "2026-01", "2026-03", "2026-06"]
SIDE_TOL = 0.01
R599 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00599-nra-refit-embargo-and-down-weighting-of-the-newest-rows-against-load-shift-absorption"
    / "scores"
    / "event_shares.parquet"
)
R612_C2 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00612-nra-combined-networks-three-class-roster-completion"
    / "absorb"
    / "event_shares_c2.parquet"
)
BANDS = ("24-120 h", "120-240 h")


def side(*, lo: float, hi: float, tol: float) -> str:
    if -tol <= lo and hi <= tol:
        return "inside the tolerance"
    if lo > tol:
        return "above the tolerance"
    if hi < -tol:
        return "below the tolerance"
    return "crosses the tolerance"


def a(arm: str) -> str:
    return arm + SFX


def run(*, family: str, specs: list[tuple], months: list[str] | None) -> pl.DataFrame:
    """Evaluate the specs at the family's Bonferroni level.

    Each spec is (label, kind, A, B, band, x, metric, tol, higher_is_better).
    """
    n = len(specs)
    alpha = 0.05 / n
    rows = []
    for label, kind, A, B, band, x, metric, tol, hib in specs:
        if kind == "bank":
            rec = t.bank_contrast(
                a=A, b=B, band=band, phase=x, metric=metric, alpha=alpha, months=months
            )
        else:
            rec = t.free_contrast(
                a=A, b=B, band=band, group=x, metric=metric, alpha=alpha, months=months
            )
        if rec is None:
            print("MISSING", family, label)
            continue
        rows.append({"family": family, "contrast": label, **rec, "tolerance": tol, "n_counted": n,
                     "verdict": t.verdict(lo=rec["lo"], hi=rec["hi"], tol=tol, higher_is_better=hib),
                     "side": side(lo=rec["lo"], hi=rec["hi"], tol=tol)})  # fmt: skip
    return pl.DataFrame(rows, infer_schema_length=None)


def grp(x: str) -> str:
    return "eight MVA" if x == t.EIGHT else "six MW"


def absorption(*, a_arm: str, b_arm: str, sources: dict[str, tuple]) -> list[dict]:
    """Return counted absorption contrasts a_arm minus b_arm at the day-14 refit, per design."""
    frames = []
    for arm, (path, name) in sources.items():
        f = pl.read_parquet(path)
        frames.append(
            (f if name is None else f.filter(pl.col("arm") == name)).with_columns(arm=pl.lit(arm))
        )
    d = s595.per_design(shares=pl.concat(frames, how="diagonal_relaxed"), value="total_q50")
    out = []
    for phase in ("day_4_to_end", "after_7_days"):
        x = d.filter((pl.col("refit_day") == 14) & (pl.col("phase") == phase))
        j = x.filter(pl.col("arm") == a_arm).join(
            x.filter(pl.col("arm") == b_arm), on="case", suffix="_b"
        )
        if j.height < 2:
            continue
        m, lo, hi = s595.interval(
            values=(j["total_q50"] - j["total_q50_b"]).to_numpy(), level=0.05 / 2
        )
        out.append({"contrast": f"absorption {phase}: {a_arm} minus {b_arm}", "n_cells": j.height, "diff": m,
                    "lo": lo, "hi": hi, "mde": (hi - lo) / 2, "level_a": float(j["total_q50"].mean()),
                    "level_b": float(j["total_q50_b"].mean()), "tolerance": t.TOL["absorb"], "n_counted": 2,
                    "verdict": t.verdict(lo=lo, hi=hi, tol=t.TOL["absorb"], higher_is_better=False),
                    "side": side(lo=lo, hi=hi, tol=t.TOL["absorb"])})  # fmt: skip
    return out


def class2(*, cand_arm: str, label: str) -> pl.DataFrame:
    cand = (a(cand_arm), "K2w")
    k1, k2 = (a("k1_d5"), "K1"), (a("k1_d5"), "K2")
    specs = []
    for phase in ("days4-end", "after0-7"):
        specs.append(
            (
                f"injected cover90 {phase}",
                "bank",
                cand,
                k1,
                "24-120 h",
                phase,
                "cover90",
                t.TOL["cover"],
                True,
            )
        )
    specs.append(
        (
            "injected follow_q50 days4-end",
            "bank",
            cand,
            k1,
            "24-120 h",
            "days4-end",
            "follow_q50",
            t.TOL["follow"],
            False,
        )
    )
    specs.append(
        (
            "injected safe-side share (NRA below q95/above q05) days4-end",
            "bank",
            cand,
            k1,
            "24-120 h",
            "days4-end",
            "nra_side95",
            SIDE_TOL,
            True,
        )
    )
    for band in BANDS:
        for g in t.GROUPS:
            specs.append(
                (
                    f"event-free cover90 {band} {grp(g)}",
                    "free",
                    cand,
                    k1,
                    band,
                    g,
                    "cover90",
                    t.TOL["cover"],
                    True,
                )
            )
            specs.append(
                (
                    f"event-free CRPS {band} {grp(g)} (minus K2)",
                    "free",
                    cand,
                    k2,
                    band,
                    g,
                    "crps",
                    t.TOL["crps"],
                    False,
                )
            )
    out = run(family=f"class2:{label}", specs=specs, months=None)
    if not SFX:
        ab = absorption(a_arm="c2", b_arm="E3", sources={"c2": (R612_C2, None), "E3": (R599, "E3")})
        if ab:
            out = pl.concat(
                [
                    out,
                    pl.DataFrame(ab, infer_schema_length=None).with_columns(
                        family=pl.lit(f"class2:{label}")
                    ),
                ],
                how="diagonal_relaxed",
            )
    return out


def o2() -> pl.DataFrame:
    nets = {"K1": ((a("k1_o2"), "K1+t"), (a("k1_d5"), "K1+t")),
            "K2": ((a("k1_o2"), "K2+t"), (a("k1_d5"), "K2+t")),
            "K3": ((a("k3_o2"), "K2+t"), (a("k3_d5b"), "K2+t"))}  # fmt: skip
    specs = []
    for net, (A, B) in nets.items():
        for band in BANDS:
            for g in t.GROUPS:
                specs.append(
                    (
                        f"{net}: O2 minus D5, event-free cover90 {band} {grp(g)}",
                        "free",
                        A,
                        B,
                        band,
                        g,
                        "cover90",
                        t.TOL["cover"],
                        True,
                    )
                )
                specs.append(
                    (
                        f"{net}: O2 minus D5, event-free CRPS {band} {grp(g)}",
                        "free",
                        A,
                        B,
                        band,
                        g,
                        "crps",
                        t.TOL["crps"],
                        False,
                    )
                )
        if net != "K3":
            for phase in ("days4-end", "after0-7"):
                specs.append(
                    (
                        f"{net}: O2 minus D5, injected cover90 {phase}",
                        "bank",
                        A,
                        B,
                        "24-120 h",
                        phase,
                        "cover90",
                        t.TOL["cover"],
                        True,
                    )
                )
            specs.append(
                (
                    f"{net}: O2 minus D5, injected follow_q50 days4-end",
                    "bank",
                    A,
                    B,
                    "24-120 h",
                    "days4-end",
                    "follow_q50",
                    t.TOL["follow"],
                    False,
                )
            )
    return run(family="o2", specs=specs, months=TAIL)


def e3() -> pl.DataFrame:
    A1, B1 = (a("noe3_d5"), "K1"), (a("k1_d5"), "K1")
    A2, B2 = (a("noe3_d5"), "K2"), (a("k1_d5"), "K2")
    specs = []
    for phase in ("days4-end", "after0-7"):
        specs.append(
            (
                f"3-day minus E3, injected cover90 {phase}",
                "bank",
                A1,
                B1,
                "24-120 h",
                phase,
                "cover90",
                t.TOL["cover"],
                True,
            )
        )
    specs.append(
        (
            "3-day minus E3, injected follow_q50 days4-end",
            "bank",
            A1,
            B1,
            "24-120 h",
            "days4-end",
            "follow_q50",
            t.TOL["follow"],
            False,
        )
    )
    specs.append(
        (
            "3-day minus E3, injected safe-side share days4-end",
            "bank",
            A1,
            B1,
            "24-120 h",
            "days4-end",
            "nra_side95",
            SIDE_TOL,
            True,
        )
    )
    for band in BANDS:
        for g in t.GROUPS:
            specs.append(
                (
                    f"3-day minus E3, event-free cover90 {band} {grp(g)}",
                    "free",
                    A2,
                    B2,
                    band,
                    g,
                    "cover90",
                    t.TOL["cover"],
                    True,
                )
            )
            specs.append(
                (
                    f"3-day minus E3, event-free CRPS {band} {grp(g)}",
                    "free",
                    A2,
                    B2,
                    band,
                    g,
                    "crps",
                    t.TOL["crps"],
                    False,
                )
            )
    out = run(family="e3", specs=specs, months=None)
    if not SFX:
        ab = absorption(a_arm="E0", b_arm="E3", sources={"E0": (R599, "E0"), "E3": (R599, "E3")})
        if ab:
            out = pl.concat(
                [out, pl.DataFrame(ab, infer_schema_length=None).with_columns(family=pl.lit("e3"))],
                how="diagonal_relaxed",
            )
    return out


def main() -> None:
    cols = [
        "family",
        "contrast",
        "diff",
        "lo",
        "hi",
        "mde",
        "n_cells",
        "n_events",
        "verdict",
        "side",
        "n_counted",
    ]
    for name, frame in (("class2", pl.concat([class2(cand_arm="c2_e3w", label="in-sample width"),
                                              class2(cand_arm="c2_e3wo", label="leave-month-out width")], how="diagonal_relaxed")),
                        ("o2", o2()), ("e3", e3())):  # fmt: skip
        frame.write_csv(OUTDIR / f"roster_{name}{SFX}.csv")
        with pl.Config(tbl_rows=100, tbl_width_chars=250, fmt_str_lengths=70):
            print(frame.select([c for c in cols if c in frame.columns]))


if __name__ == "__main__":
    main()

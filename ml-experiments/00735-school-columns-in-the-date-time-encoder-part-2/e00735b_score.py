# ruff: noqa: E501, PT018
"""Experiment 735 part 2: score the arms R1, SKE, SKE-S21, and SKE-CP and map the readings to a verdict.

README. This script imports part 1's scorer (`e00735_score`: the cells, the pairing on row keys, the
two-level bootstrap, the gain rules, the replay gate, the metric tables) and changes only what the new
arm names and the 8 refit months need. It sets `R735` before the import (part 1's scorer reads its data
folder from that variable), then replaces four things on the imported module:

- `MONTHS` is the 8 held-in refit months (2025-11 to 2026-06), so no cell of a later month is read;
- `period_of` labels cells of the 2026-06 refit with a valid date on or after 2026-07-01 as `july to
  september` (excluded from every reading) and every other cell `held-in`; part 1's version would label
  the eighth month `october`;
- `GATE_MONTH` is 2026-03 (the replay gate runs R0 there);
- `FAMILY_LEVEL` is 99%: 95% with a Bonferroni correction over this plan's 5 readings.

The cells stage, the stamps, the bootstrap, the gate, and the metric tables are part 1's code, run on
this experiment's folder. The contrasts and the verdict are this file's, because the arm names and
the plan's rules differ. A reading is the paired difference of CRPS (points of capacity), arm minus
reference, on the rows of the school windows at days 1 to 10 of the 8 held-in months, pooled over 3
seeds (the same rows and the same bootstrap as part 1: refit months, then seeds, then 7-day blocks of
valid dates).

Readings (the plan's decision rule):

- `C1 six` and `C1 eight`: SKE minus R1; `C2 six` and `C2 eight`: SKE minus SKE-S21; `C3`: SKE minus
  SKE-CP at the 4 coastal Primaries, all at 99%. `inland C3` is the same contrast at the 3 inland
  Primaries at 95%.
- Placebos at 95%: `S21-R1 six` and `S21-R1 eight` (SKE-S21 minus R1 on the school windows), and
  `CP-R1` (SKE-CP minus R1 at the 4 coastal Primaries).
- No-harm, one-sided 95% upper bound on all days-1-to-10 rows of the 8 held-in months, SKE minus R1:
  `year held-in SKE six`, `... eight`, and `... all 16`.
- Descriptive: SKE minus R0, R1 minus R0 (R0 is experiment 677's stored control), and the manual
  heuristic and climatology.

Usage: `--stage cells|contrasts|all`, `--stored` (score experiment 677's stored rows and let them stand
in for R0), `--gate`, `--smoke`. Run under a memory cap on the experiment's cores, with PYTHONPATH from
`ml-experiments/00524*/env.sh`; peak memory is part 1's scoring peak (under 8 GB).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00735-school-columns-in-the-date-time-encoder-part-2"
os.environ.setdefault("R735", str(ML_EXPERIMENTS_DATA_DIR / NAME))
_R735 = os.environ["R735"]
assert "school-columns-in-the-date-time-encoder-part-2" in _R735 or "00735-smoke" in _R735, _R735
assert "coast-and-school" not in _R735 and "00677" not in _R735, _R735

import e00735_score as s1  # ty: ignore[unresolved-import]  # noqa: E402
import e00735_windows as win  # ty: ignore[unresolved-import]  # noqa: E402

NEW_ARMS: Final = ("R1", "SKE", "SKE-S21", "SKE-CP")
MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
HELD_IN_MONTHS: Final = len(MONTHS)
GATE_MONTH: Final = "2026-03"
FAMILY_LEVEL: Final = 1.0 - 0.05 / 5
PLACEBO_LEVEL: Final = 0.95
SMOKE_ROOT: Final = (
    Path(__file__).resolve().parents[2] / ".claude" / "scratch" / "e735b_smoke" / "00735-smoke"
)
ROOT: Final[Path] = s1.ROOT
ROOT_PART_1: Final[Path] = ML_EXPERIMENTS_DATA_DIR / s1.NAME

s1.MONTHS = MONTHS
s1.HELD_IN_MONTHS = HELD_IN_MONTHS
s1.GATE_MONTH = GATE_MONTH
s1.FAMILY_LEVEL = FAMILY_LEVEL
s1.ARMS = ("R0", *NEW_ARMS)


def period_of(*, m: np.ndarray, vday: np.ndarray) -> np.ndarray:
    """Return `held-in` for every cell, except `july to september` for cells with a valid date on or after 2026-07-01."""
    out = np.full(len(m), "held-in", dtype=object)
    out[vday >= s1.HOLDOUT_CODE] = "july to september"
    return out.astype(str)


s1.period_of = period_of

REQUIRED_READINGS: Final[dict[str, tuple[str, ...]]] = {
    **dict.fromkeys(("C1 six", "C1 eight"), ("SKE", "R1")),
    **dict.fromkeys(("C2 six", "C2 eight"), ("SKE", "SKE-S21")),
    "C3": ("SKE", "SKE-CP"),
    "inland C3": ("SKE", "SKE-CP"),
    **dict.fromkeys(("S21-R1 six", "S21-R1 eight"), ("SKE-S21", "R1")),
    "CP-R1": ("SKE-CP", "R1"),
    **{f"year held-in SKE {g}": ("SKE", "R1") for g in ("six", "eight")},
}
"""Every reading that the verdict reads, with the arms it needs."""


def verdict(*, r: dict[str, dict]) -> dict:
    """Map the readings to the plan's verdict, first rule that applies.

    Args:
        r: The named readings (the keys of `REQUIRED_READINGS`).

    Returns:
        The verdict, the school claim per group, the coast claim, and the rule that decided.

    Raises:
        ValueError: If a required reading is missing, because a missing reading would otherwise
            turn into `inconclusive` without a word.
    """
    absent = [name for name in REQUIRED_READINGS if name not in r]
    if absent:
        raise ValueError(f"readings missing: {absent}")

    def school(group: str) -> bool:
        return bool(
            s1.is_gain(r=r[f"C1 {group}"])
            and s1.is_gain(r=r[f"C2 {group}"])
            and not s1.placebo_failed(r=r[f"S21-R1 {group}"])
            and r[f"year held-in SKE {group}"]["upper"] <= s1.WHOLE_YEAR_BOUND
        )

    school_six, school_eight = school("six"), school("eight")
    coast = bool(
        s1.is_gain(r=r["C3"])
        and not s1.placebo_failed(r=r["CP-R1"])
        and r["year held-in SKE six"]["upper"] <= s1.WHOLE_YEAR_BOUND
        and r["year held-in SKE eight"]["upper"] <= s1.WHOLE_YEAR_BOUND
        and not s1.placebo_failed(r=r["inland C3"])
    )
    family_gains = [r[n] for n in ("C1 six", "C1 eight", "C2 six", "C2 eight", "C3")]
    if any(s1.is_harm(r=r[f"C1 {g}"]) and s1.is_harm(r=r[f"C2 {g}"]) for g in ("six", "eight")):
        result, rule = "refuted", 1
    elif school_six and school_eight and coast:
        result, rule = "supported", 2
    elif school_six or school_eight or coast:
        result, rule = "partly supported", 3
    elif any(s1.is_small_gain(r=r[n]) for n in ("C1 six", "C1 eight", "C3")):
        result, rule = "small gain", 4
    elif all(s1.inside_band(r=x) for x in family_gains):
        result, rule = "not supported", 5
    else:
        result, rule = "inconclusive", 6
    return {
        "verdict": result,
        "rule": rule,
        "school_claim_six": school_six,
        "school_claim_eight": school_eight,
        "coast_claim": coast,
    }


def stage_contrasts(*, r0_from_stored: bool) -> dict:
    """Compute every reading, the verdict, and the tables; write them under `scores/`."""
    out_dir = ROOT / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = win.sites()
    groups = win.group_indices(labels=table["d_label"].to_list())
    six, eight, coastal, inland = (
        groups[k] for k in ("six signed-MW", "eight MVA-only", "4 coastal", "3 inland placebo")
    )
    part_1_cache = ROOT_PART_1 / win.WEATHER_CACHE_NAME
    daily = win.daily_weather(
        expected_dir=win.EXPECTED_DIR,
        cache=part_1_cache if part_1_cache.exists() else ROOT / win.WEATHER_CACHE_NAME,
    )
    weather = (daily, win.monthly_medians(weather=daily))
    arms: dict[str, pl.DataFrame] = {}
    for arm in ("R0", *NEW_ARMS, "stored", *s1.BASELINES):
        loaded = s1.load_arm(arm=arm, months=MONTHS)
        if loaded is not None:
            arms[arm] = loaded
    if "R0" not in arms and r0_from_stored and "stored" in arms:
        arms["R0"] = arms["stored"]
    print("arms scored in all 8 months:", sorted(arms), flush=True)
    held = list(range(HELD_IN_MONTHS))
    readings: dict[str, dict] = {}
    rows: list[dict] = []

    def add(r: dict, **tags: object) -> dict:
        readings[r["name"]] = r
        rows.append({**tags, **r})
        return r

    pairs: dict[tuple[str, str], s1.Pair | None] = {}

    def get(a: str, b: str) -> s1.Pair | None:
        if (a, b) not in pairs:
            pairs[(a, b)] = s1.pair_of(arms=arms, a=a, b=b, weather=weather)
        return pairs[(a, b)]

    def window_rows(p: s1.Pair) -> np.ndarray:
        return p.masks[win.SCHOOL_WINDOW] & p.lead_ok & (p.period == "held-in")

    for name, a, b, idx, level, tag in (
        ("C1 six", "SKE", "R1", six, FAMILY_LEVEL, "C1"),
        ("C1 eight", "SKE", "R1", eight, FAMILY_LEVEL, "C1"),
        ("C2 six", "SKE", "SKE-S21", six, FAMILY_LEVEL, "C2"),
        ("C2 eight", "SKE", "SKE-S21", eight, FAMILY_LEVEL, "C2"),
        ("C3", "SKE", "SKE-CP", coastal, FAMILY_LEVEL, "C3"),
        ("inland C3", "SKE", "SKE-CP", inland, PLACEBO_LEVEL, "inland C3"),
        ("S21-R1 six", "SKE-S21", "R1", six, PLACEBO_LEVEL, "S21-R1"),
        ("S21-R1 eight", "SKE-S21", "R1", eight, PLACEBO_LEVEL, "S21-R1"),
        ("CP-R1", "SKE-CP", "R1", coastal, PLACEBO_LEVEL, "CP-R1"),
    ):
        p = get(a, b)
        if p is None:
            continue
        add(
            s1.reading(
                pair=p,
                mask=window_rows(p) & s1.group_mask(pair=p, indices=idx),
                name=name,
                level=level,
                months=held,
            ),
            tag=tag,
            group=name,
            level="family 99" if level == FAMILY_LEVEL else "95",
        )
    p = get("SKE", "R1")
    if p is not None:
        for gname, idx in (("six", six), ("eight", eight), ("all 16", list(range(16)))):
            mask = p.lead_ok & s1.group_mask(pair=p, indices=idx) & (p.period == "held-in")
            add(
                s1.reading(
                    pair=p,
                    mask=mask,
                    name=f"year held-in SKE {gname}",
                    level=PLACEBO_LEVEL,
                    months=held,
                    one_sided=True,
                ),
                tag="year held-in SKE - R1",
                group=gname,
                level="one-sided 95",
            )
    sets = (("six", six), ("eight", eight), ("4 coastal", coastal), ("all 16", list(range(16))))
    for a, b in (
        ("SKE", "R0"), ("R1", "R0"), ("SKE", "R1"), ("SKE", "SKE-S21"), ("SKE", "SKE-CP"),
        ("SKE", "climatology"), ("SKE", "heuristic"),
    ):  # fmt: skip
        p = get(a, b)
        if p is None:
            continue
        for gname, idx in sets:
            g = s1.group_mask(pair=p, indices=idx)
            for wname, wmask in (
                ("school window", p.masks[win.SCHOOL_WINDOW]),
                ("ordinary days", p.masks[win.ORDINARY]),
                *[(k, v) for k, v in p.masks.items() if k.startswith("school: ")],
            ):
                add(
                    s1.reading(
                        pair=p,
                        mask=wmask & g & p.lead_ok & (p.period == "held-in"),
                        name=f"{a}-{b} {gname} {wname} held-in",
                        level=PLACEBO_LEVEL,
                        months=held,
                    ),
                    tag=f"{a} - {b}",
                    group=gname,
                    window=wname,
                    level="95 (descriptive)",
                )
    summary = {"readings": readings, "verdict": verdict(r=readings)}
    summary["unpaired_rows"] = {
        f"{p.name} - {p.reference}": p.unpaired for p in pairs.values() if p is not None
    }
    (out_dir / "readings.json").write_text(json.dumps(summary, indent=1, default=float))
    pl.DataFrame(rows, infer_schema_length=None).write_csv(out_dir / "readings.csv")
    by_fit_table(get=get, groups=groups, out=out_dir / "by_fit.csv")
    s1.metrics_tables(arms=arms, weather=weather, groups=groups, out=out_dir)
    print(json.dumps(summary["verdict"], indent=1), flush=True)
    return summary


def by_fit_table(*, get, groups: dict, out: Path) -> None:  # noqa: ANN001
    """Write each family contrast by refit month and seed (the 24 held-in fits)."""
    rows = []
    for tag, a, b, gname in (
        ("C1", "SKE", "R1", "six signed-MW"), ("C1", "SKE", "R1", "eight MVA-only"),
        ("C2", "SKE", "SKE-S21", "six signed-MW"), ("C2", "SKE", "SKE-S21", "eight MVA-only"),
        ("C3", "SKE", "SKE-CP", "4 coastal"),
    ):  # fmt: skip
        p = get(a, b)
        if p is None:
            continue
        mask = (
            p.masks[win.SCHOOL_WINDOW]
            & p.lead_ok
            & (p.period == "held-in")
            & s1.group_mask(pair=p, indices=groups[gname])
        )
        f = p.frame.filter(pl.Series(mask))
        agg = (
            f.group_by("m", "seed").agg(pl.col("d_crps").sum(), pl.col("n").sum()).sort("m", "seed")
        )
        rows.append(
            agg.with_columns(
                diff=pl.col("d_crps") / pl.col("n"),
                contrast=pl.lit(tag),
                group=pl.lit(gname),
                month=pl.col("m").map_elements(lambda i: MONTHS[i], return_dtype=pl.Utf8),
            ).select("contrast", "group", "month", "seed", "n", "diff")
        )
    if rows:
        pl.concat(rows).write_csv(out)


def stage_smoke() -> int:
    """Score each smoke arm's stage-3 validation rows, check the quantiles, and print the time of each arm-month."""
    folder = SMOKE_ROOT / "arms"
    taus = np.asarray([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
    base: np.ndarray | None = None
    for arm in NEW_ARMS:
        path = next(
            iter((folder / f"k1_{arm}_sp1" / "work" / "valpred").glob("*_s0.parquet")), None
        )
        if path is None:
            print(arm, "no valpred file")
            continue
        f = pl.read_parquet(path)
        qcols = [f"q{i:02d}" for i in (1, 5, 10, 25, 40, 50, 60, 75, 90, 95, 99)]
        q = f.select(qcols).to_numpy().astype(np.float64)
        y = f["y"].to_numpy().astype(np.float64)
        assert np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all(), (
            f"{arm}: quantiles not finite and sorted"
        )
        d = y[:, None] - q
        crps = float((2.0 * np.maximum(taus * d, (taus - 1) * d).mean(axis=1)).mean() * 100.0)
        mae = float(np.abs(y - q[:, 5]).mean() * 100.0)
        base = q[:, 5] if arm == "R1" else base
        diff = (
            f"{float(np.abs(q[:, 5] - base).max()):.2e}"
            if base is not None and arm != "R1" and len(base) == len(q)
            else "n/a"
        )
        print(
            f"smoke {arm}: rows {f.height}, CRPS {crps:.3f}, MAE of median {mae:.3f} (points of capacity, the stage-3 validation rows), max |median - R1 median| {diff}"
        )
    times = SMOKE_ROOT / "logs" / "phase_times.csv"
    if times.exists():
        t = pl.read_csv(times).with_columns(
            minutes=(
                pl.col("end_utc").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ")
                - pl.col("start_utc").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ")
            ).dt.total_seconds()
            / 60.0
        )
        print(t.group_by("arm", "phase").agg(pl.col("minutes").sum()).sort("arm", "phase"))
    return 0


def main() -> int:
    """Run the requested stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cells", "contrasts", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--arms", nargs="+", default=list(NEW_ARMS))
    parser.add_argument(
        "--stored", action="store_true", help="score the stored arm 9a and let it stand in for R0"
    )
    args = parser.parse_args()
    if args.smoke:
        return stage_smoke()
    if args.gate:
        return s1.stage_gate()
    if args.stage in {"cells", "all"}:
        s1.stage_cells(
            arms=[*args.arms, *(["stored", *s1.BASELINES] if args.stored else [])],
            stored=args.stored,
        )
    if args.stage in {"contrasts", "all"}:
        stage_contrasts(r0_from_stored=args.stored)
    return 0


if __name__ == "__main__":
    sys.exit(main())

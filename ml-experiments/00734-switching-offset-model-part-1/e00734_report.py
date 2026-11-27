"""Stage 5 of experiment 734: every number the page quotes, read from the files under `OUT`.

The report reads the scored files of `e00734_tasks.py`, the real-record files of `e00734_run.py`,
and the stored rows of experiments 680 and 670 (read-only). It decodes nothing. Every pre-registered
prediction (P1 to P10), both vetoes, the verdict, the leave-one-out table, and the audit triggers
are computed here, with the plan's rules and no others.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import e00734_score as score
import e00734_store as store
import numpy as np
import polars as pl

E680: Final[str] = "00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search"
E670: Final[str] = "00670-event-table-recipient-search-on-six-cell-grid"
H_ARMS: Final[tuple[str, ...]] = ("H-own", "H-nbr", "H-nbr-noramp", "H-nbr-noshared")
BENCH: Final[str] = "E1-J-sym"
SENSITIVITY: Final[tuple[str, ...]] = ("long_heavy", "short_heavy", "constant_mw_square")
TEMPORARY: Final[tuple[str, ...]] = ("edge", "solo")
GATES: Final[tuple[str, ...]] = ("G1", "G2", "G4", "G5")
"""Gates recorded in `gates.json`; G3 is the pytest file and G3b is `g3b.json`."""
STORED_FALSE_SOLAR: Final[int] = 333
"""False events of experiment 653's masked arm on experiment 686's 24 solar-growth worlds."""
STORED_BIAS_RATE: Final[float] = 2.2
"""False events per injected Primary-year of experiment 653's masked arm, seasonal-bias worlds."""
P3_RANGE: Final[tuple[float, float]] = (0.03, 0.08)
P4_MIN: Final[float] = 0.02
P5_MIN: Final[float] = 0.05
P2_MIN: Final[float] = 0.15
P6_BAND: Final[float] = 0.03
P7_RANGE: Final[tuple[float, float]] = (0.70, 0.90)
P8_RAMP_MIN: Final[float] = 0.05
P9_MAX: Final[float] = 0.5
P10_RANGE: Final[tuple[float, float]] = (0.90, 1.10)
NEITHER_MAX: Final[float] = 0.5
"""Share of found injected temporary events graded "neither" above which the design is wrong."""
SHORT_CELLS: Final[tuple[str, ...]] = ("short, shallow", "short, deep")


@dataclass(frozen=True)
class Stored:
    """Stored rows of experiments 680 and 670 (read only)."""

    e1jsym: pl.DataFrame
    """E1-J-sym's rows on the 40 default worlds."""
    e1: pl.DataFrame
    """E1's rows on the 40 default worlds."""
    sensitivity: dict[str, pl.DataFrame] = field(default_factory=dict)
    """E1-J-sym's rows on each sensitivity configuration."""


def load_stored(*, root: Path | None = None) -> Stored:
    """Read the stored rows of experiments 680 and 670 for the default and sensitivity worlds."""
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    base = ML_EXPERIMENTS_DATA_DIR if root is None else root

    def rows(*, folder: str, config: str, method: str, worlds: range) -> pl.DataFrame:
        # Every `injected_<config>_<first seed>.parquet`, as experiments 670 and 680 read them.
        paths = sorted((base / folder).glob(f"injected_{config}_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no stored rows injected_{config}_*.parquet in {folder}")
        frames = [pl.read_parquet(p) for p in paths]
        out = pl.concat(frames, how="diagonal_relaxed").filter(
            (pl.col("method") == method) & pl.col("world").is_in(list(worlds))
        )
        got = set(out["world"].unique().to_list())
        if got != set(worlds):
            raise ValueError(f"{folder} {config} {method}: worlds {sorted(got)}, not {worlds}")
        return out

    return Stored(
        e1jsym=rows(folder=E680, config="default", method=BENCH, worlds=range(900, 940)),
        e1=rows(folder=E670, config="default", method="E1", worlds=range(900, 940)),
        sensitivity={
            c: rows(folder=E680, config=c, method=BENCH, worlds=range(1000, 1010))
            for c in SENSITIVITY
        },
    )


def read_files(*, folder: Path, arm: str, prefix: str = "", suffix: str = "") -> pl.DataFrame:
    """Concatenate `<prefix><arm>_<seed><suffix>.parquet` files of a folder, and only those.

    The match rows of an arm are `H-nbr_900.parquet`. The call tables `H-nbr_900_calls.parquet`
    and the posterior files `post_H-nbr_900.parquet` have the same arm name and must not mix in,
    and `H-nbr` must not pick up `H-nbr-noramp_900.parquet`.

    Args:
        folder: The scored folder of one set.
        arm: The arm's name.
        prefix: The file-name prefix of the kind of file (`post_`, `hard_`, `scen_`, or none).
        suffix: The file-name suffix before the extension (`_calls` or none).

    Returns:
        The rows of all matching files, or an empty frame if there are none.
    """
    pattern = re.compile(
        rf"^{re.escape(prefix)}{re.escape(arm)}_(\d+){re.escape(suffix)}\.parquet$"
    )
    files = sorted(f for f in folder.glob("*.parquet") if pattern.match(f.name))
    frames = [pl.read_parquet(f) for f in files]
    frames = [f for f in frames if f.height]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def cell_rows(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Injected rows of 2 noise units or more with their cell and month (empty stays empty)."""
    return score.add_cells(rows=rows) if rows.height else rows


def share_table(*, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """MW-weighted share found and share paired, per cell and arm."""
    cells = sorted({c for t in tables.values() if t.height for c in t["cell"].to_list()})
    out = []
    for cell in cells:
        for arm, t in tables.items():
            if not t.height:
                continue
            sub = t.filter(pl.col("cell") == cell)
            edge = sub.filter(pl.col("edge_truth")) if "edge_truth" in sub.columns else sub
            out.append(
                {
                    "cell": cell,
                    "arm": arm,
                    "n_events": sub.height,
                    "found": score.found_share(rows=t, cell=cell),
                    "paired": score.found_share(rows=edge, col="right_class"),
                }
            )
    return pl.DataFrame(out)


def calibration_by_cell(
    *, post: pl.DataFrame, uninjected: pl.DataFrame, rows: pl.DataFrame
) -> pl.DataFrame:
    """Pinball loss and coverage of the predicted switched megawatts, by cell and found flag.

    The predicted shift at a block is the injected world's posterior quantile minus the uninjected
    world's posterior median at the same block. Edge and solo events use the offset's quantiles,
    and permanent shifts use the quantiles of the offset plus the permanent level and the
    uninjected median of the same quantity, because the model is meant to put a permanent shift in
    the level.

    Args:
        post: The posterior-window rows of one arm over the worlds (`store.window_table`).
        uninjected: The arm's uninjected block table (`store.uninjected_table`).
        rows: The arm's match rows with `cell` (from `score.add_cells`) and `found`.

    Returns:
        One row per cell, kind group, and found flag, with the number of events and blocks, the
        coverage of the 80% interval pooled over blocks (`coverage_80`) and averaged over events
        (`event_coverage_80`), and the pinball losses.
    """
    if not post.height or not rows.height:
        return pl.DataFrame()
    keys = rows.select("world", "site", "start", "cell", "found")
    j = post.join(keys, on=["world", "site", "start"], how="inner").join(
        uninjected.select("site", "block", u50="q50", us50="s50"), on=["site", "block"]
    )
    out = []
    is_permanent = pl.col("kind").is_in(list(store.PERMANENT_KINDS))
    for group, cols, base in (
        ("temporary", store.Q_COLS, "u50"),
        ("permanent", store.S_COLS, "us50"),
    ):
        sel = j.filter(~is_permanent if group == "temporary" else is_permanent)
        for (cell, found), g in sel.group_by("cell", "found"):
            quant = np.stack([(g[c] - g[base]).to_numpy() for c in cols], axis=1)
            truth = g["truth_mw"].to_numpy()
            cal = score.calibration(truth=truth, quantiles=quant)
            covered = (truth >= quant[:, 1]) & (truth <= quant[:, 5])
            per_event = (
                g.with_columns(covered=pl.Series(covered))
                .group_by("world", "site", "start")
                .agg(c=pl.col("covered").mean())
            )
            out.append(
                {
                    "cell": cell,
                    "group": group,
                    "found": found,
                    "n_events": per_event.height,
                    "n_blocks": g.height,
                    "event_coverage_80": float(per_event["c"].mean()),  # ty: ignore[invalid-argument-type]
                    **cal,
                }
            )
    return pl.DataFrame(out).sort("group", "cell", "found") if out else pl.DataFrame()


def conservation_residual(*, calls: pl.DataFrame) -> dict[str, float]:
    """Sum of the two posterior mean shifts over a mirrored event, as a share of the donor's shift.

    Only mirrored events between signed-megawatt Primaries count. The donor is the event with the
    negative amplitude, so each pair counts once.

    Args:
        calls: H-nbr's call table over the worlds.

    Returns:
        The number of pairs and the median, quartiles, and mean absolute value of the share.
    """
    if not calls.height:
        return {"n": 0}
    sel = calls.filter(
        pl.col("grade").str.starts_with("mirrored")
        & pl.col("signed_pair").fill_null(False)
        & (pl.col("amp_mw") < 0)
        & pl.col("post_mean_mw").is_not_null()
        & pl.col("partner_post_mean_mw").is_not_null()
        & (pl.col("post_mean_mw").abs() > 0)
    )
    if not sel.height:
        return {"n": 0}
    share = (
        (sel["post_mean_mw"] + sel["partner_post_mean_mw"]) / sel["post_mean_mw"].abs()
    ).to_numpy()
    q1, med, q3 = np.percentile(share, [25, 50, 75])
    return {
        "n": int(sel.height),
        "median": float(med),
        "q25": float(q1),
        "q75": float(q3),
        "mean_abs": float(np.abs(share).mean()),
    }


def timing_error(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Median and 90th percentile of the start and end errors in hours, by cell (found events)."""
    if not rows.height or "start_err_hh" not in rows.columns:
        return pl.DataFrame()
    out = []
    for cell in sorted(set(rows["cell"].to_list())):
        sub = rows.filter((pl.col("cell") == cell) & pl.col("found"))
        row: dict = {
            "cell": cell,
            "n": sub.height,
            "missed": rows.filter(pl.col("cell") == cell).height - sub.height,
        }
        for name in ("start_err_hh", "end_err_hh"):
            v = sub[name].drop_nulls().abs().to_numpy() / 2.0
            row[f"{name[:-3]}_median_h"] = float(np.median(v)) if len(v) else float("nan")
            row[f"{name[:-3]}_p90_h"] = float(np.percentile(v, 90)) if len(v) else float("nan")
        out.append(row)
    return pl.DataFrame(out)


def calls_on_hard(*, calls: pl.DataFrame, hard: pl.DataFrame) -> dict[str, float]:
    """Called events on hard negatives per hard negative, and the mean posterior shift over them."""
    n_hard = hard.height
    n_calls = int(calls["on_hard_negative"].sum()) if calls.height else 0
    return {
        "hard_negatives": n_hard,
        "calls_on_hard": n_calls,
        "calls_per_hard": n_calls / n_hard if n_hard else float("nan"),
        "mean_abs_shift_mw": float(hard["mean_abs_shift_mw"].mean()) if n_hard else float("nan"),  # ty: ignore[invalid-argument-type]
    }


def false_calls(*, calls: pl.DataFrame) -> pl.DataFrame:
    """Calls that match no truth and overlap no hard negative, minus the real-record events."""
    if not calls.height:
        return calls
    cand = calls.filter(~pl.col("matched") & ~pl.col("on_hard_negative"))
    return score.drop_recurring(candidates=cand)


def leave_one_out_flags(*, loo: pl.DataFrame, point: float) -> pl.DataFrame:
    """Mark the leave-one-out rows whose point estimate crosses zero or the +0.10 bar."""
    side = (point >= score.C1_BAR, point > 0)
    return loo.with_columns(
        flips=(pl.col("point") >= score.C1_BAR) != side[0],
        crosses_zero=(pl.col("point") > 0) != side[1],
    )


def _ok(*, value: float | None, test: Callable[[float], bool]) -> bool | None:
    """Whether a number passes a test, or None when the number is missing or not a number."""
    return None if value is None or math.isnan(value) else test(value)


def _all(*, values: dict | None, test: Callable[[float], bool]) -> bool | None:
    """Whether every number of a dict passes a test, or None when there are none."""
    if not values:
        return None
    flags = [_ok(value=v, test=test) for v in values.values()]
    return None if any(f is None for f in flags) else all(flags)


def evaluate_predictions(*, d: dict) -> list[dict]:
    """Each pre-registered prediction with its number and whether it held.

    Args:
        d: The numbers of the report: `c1`, `c2` (contrasts), `p3`, `p4`, `p5` (points),
            `p6` (points by short cell), `p7` (coverage by long and medium cell), `p8` (two
            points), `p9` (ratio), and `p10` (two ratios). A missing key gives `held: None`.

    Returns:
        One dict per prediction.
    """
    p1 = getattr(d.get("c1"), "point", None)
    p2 = getattr(d.get("c2"), "point", None)
    p8, p10 = d.get("p8"), d.get("p10")
    return [
        {
            "id": "P1",
            "value": p1,
            "rule": ">= 0.10",
            "held": _ok(value=p1, test=lambda v: v >= score.C1_BAR),
        },
        {
            "id": "P2",
            "value": p2,
            "rule": f">= {P2_MIN}",
            "held": _ok(value=p2, test=lambda v: v >= P2_MIN),
        },
        {
            "id": "P3",
            "value": d.get("p3"),
            "rule": f"{P3_RANGE[0]} to {P3_RANGE[1]}",
            "held": _ok(value=d.get("p3"), test=lambda v: P3_RANGE[0] <= v <= P3_RANGE[1]),
        },
        {
            "id": "P4",
            "value": d.get("p4"),
            "rule": f">= {P4_MIN}",
            "held": _ok(value=d.get("p4"), test=lambda v: v >= P4_MIN),
        },
        {
            "id": "P5",
            "value": d.get("p5"),
            "rule": f">= {P5_MIN}",
            "held": _ok(value=d.get("p5"), test=lambda v: v >= P5_MIN),
        },
        {
            "id": "P6",
            "value": d.get("p6"),
            "rule": f"within +-{P6_BAND} in each short cell",
            "held": _all(values=d.get("p6"), test=lambda v: abs(v) <= P6_BAND),
        },
        {
            "id": "P7",
            "value": d.get("p7"),
            "rule": f"{P7_RANGE[0]} to {P7_RANGE[1]} in each long and medium cell",
            "held": _all(values=d.get("p7"), test=lambda v: P7_RANGE[0] <= v <= P7_RANGE[1]),
        },
        {
            "id": "P8",
            "value": p8,
            "rule": f"H-nbr minus E1-J-sym >= 0; H-nbr minus H-nbr-noramp >= {P8_RAMP_MIN}",
            "held": None if not p8 else (p8[0] >= 0 and p8[1] >= P8_RAMP_MIN),
        },
        {
            "id": "P9",
            "value": d.get("p9"),
            "rule": f"<= {P9_MAX}",
            "held": _ok(value=d.get("p9"), test=lambda v: v <= P9_MAX),
        },
        {
            "id": "P10",
            "value": p10,
            "rule": f"each {P10_RANGE[0]} to {P10_RANGE[1]}",
            "held": None if not p10 else all(P10_RANGE[0] <= x <= P10_RANGE[1] for x in p10),
        },
    ]


def gate_status(*, out: Path) -> dict[str, bool | None]:
    """Whether each recorded gate passed (None when its file is missing)."""
    status: dict[str, bool | None] = {}
    g = out / "gates.json"
    data = json.loads(g.read_text()) if g.exists() else {}
    for k in GATES:
        status[k] = None if k not in data else bool(data[k].get("passed", False))
    b = out / "g3b.json"
    status["G3b"] = bool(json.loads(b.read_text())["G3b_passed"]) if b.exists() else None
    return status


def scenario_table(*, scen: pl.DataFrame) -> pl.DataFrame:
    """Normalised mean absolute error of the lag-free forecast per arm and scenario, pooled."""
    if not scen.height:
        return scen
    return (
        scen.group_by("arm", "scenario")
        .agg(n=pl.col("n").sum(), b=pl.col("sum_before").sum(), a=pl.col("sum_after").sum())
        .with_columns(nmae_before=pl.col("b") / pl.col("n"), nmae_after=pl.col("a") / pl.col("n"))
        .drop("b", "a")
        .sort("arm", "scenario")
    )


def _has_cell(*, table: pl.DataFrame, cell: str) -> bool:
    return bool(table.height) and cell in set(table["cell"].to_list())


def _diff(*, a: pl.DataFrame, b: pl.DataFrame, cell: str) -> float | None:
    if not a.height or not b.height:
        return None
    return score.found_share(rows=a, cell=cell) - score.found_share(rows=b, cell=cell)


def _load_set(*, folder: Path, arms: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    return {a: cell_rows(rows=read_files(folder=folder, arm=a)) for a in arms}


def compute(*, out: Path, stored: Stored, n_boot_seed: int = 734) -> dict:  # noqa: PLR0912, PLR0915
    """Compute every number of the report from the files under `out` and the stored rows.

    Args:
        out: The experiment's data folder.
        stored: The stored rows of experiments 680 and 670.
        n_boot_seed: The seed of the bootstrap generator.

    Returns:
        A dict of tables and numbers, printed by `report`.
    """
    rng = np.random.default_rng(n_boot_seed)
    res: dict = {}
    scored = out / "scored"
    default = _load_set(folder=scored / "default", arms=(*H_ARMS, "PELT"))
    default[BENCH] = cell_rows(rows=stored.e1jsym)
    default["E1"] = cell_rows(rows=stored.e1)
    res["shares"] = share_table(tables=default)
    cell = score.COUNTED_CELL
    hn, hown, pelt, bench = default["H-nbr"], default["H-own"], default["PELT"], default[BENCH]
    d: dict = {}
    if hn.height and bench.height:
        d["c1"] = score.contrast(a=hn, b=bench, cell=cell, rng=rng)
    if hn.height and pelt.height:
        d["c2"] = score.contrast(a=hn, b=pelt, cell=cell, rng=rng)
    d["p3"] = _diff(a=hown, b=bench, cell=cell)
    d["p4"] = _diff(a=hn, b=hown, cell=cell)
    d["p5"] = _diff(a=hn, b=bench, cell="long, deep")
    d["p6"] = {c: _diff(a=hn, b=bench, cell=c) for c in SHORT_CELLS}
    res["intervals"] = {
        name: score.contrast(a=default[a], b=default[b], cell=c, rng=rng)
        for name, a, b, c in (
            ("P3 H-own - E1-J-sym", "H-own", BENCH, cell),
            ("P4 H-nbr - H-own", "H-nbr", "H-own", cell),
            ("P5 H-nbr - E1-J-sym, long deep", "H-nbr", BENCH, "long, deep"),
        )
        if _has_cell(table=default[a], cell=c) and _has_cell(table=default[b], cell=c)
    }
    # calibration (P7) and the posterior files
    calib = []
    for arm in H_ARMS:
        post = read_files(folder=scored / "default", arm=arm, prefix="post_")
        uninj_file = out / "uninjected" / f"{arm}_blocks.parquet"
        if post.height and uninj_file.exists() and default[arm].height:
            calib.append(
                calibration_by_cell(
                    post=post, uninjected=pl.read_parquet(uninj_file), rows=default[arm]
                ).with_columns(arm=pl.lit(arm))
            )
    res["calibration"] = pl.concat(calib, how="diagonal_relaxed") if calib else pl.DataFrame()
    cal = res["calibration"]
    if cal.height:
        sel = cal.filter(
            (pl.col("arm") == "H-nbr")
            & pl.col("found")
            & (pl.col("group") == "temporary")
            & pl.col("cell").str.contains("^(long|medium)")
        )
        d["p7"] = dict(zip(sel["cell"].to_list(), sel["coverage_80"].to_list(), strict=True))
    # cascade (P8) and cold spells (P9)
    casc = _load_set(folder=scored / "cascade", arms=(*H_ARMS, "PELT", BENCH, "E1"))
    only = {a: (t.filter(pl.col("cascaded")) if t.height else t) for a, t in casc.items()}
    res["cascade"] = {
        a: score.found_share(rows=t) if t.height else float("nan") for a, t in only.items()
    }
    if only["H-nbr"].height and only[BENCH].height and only["H-nbr-noramp"].height:
        d["p8"] = (
            score.found_share(rows=only["H-nbr"]) - score.found_share(rows=only[BENCH]),
            score.found_share(rows=only["H-nbr"]) - score.found_share(rows=only["H-nbr-noramp"]),
        )
    cold = {
        a: read_files(folder=scored / "cold", arm=a, suffix="_calls")
        for a in ("H-nbr", "H-nbr-noshared", BENCH)
    }
    n_cold = {a: (int(t["on_hard_negative"].sum()) if t.height else None) for a, t in cold.items()}
    res["cold_calls_in_spells"] = n_cold
    if n_cold["H-nbr"] is not None and n_cold["H-nbr-noshared"]:
        d["p9"] = n_cold["H-nbr"] / n_cold["H-nbr-noshared"]
    # real record: vetoes and P10
    summary_file = out / "real" / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    res["real"] = summary
    v1 = v2 = False
    if "H-nbr" in summary and BENCH in summary:
        h, b = summary["H-nbr"], summary[BENCH]
        r1, v1 = score.veto_ratios(
            h_rates=tuple(h["rates_scored"]), base_rates=tuple(b["rates_scored"])
        )
        r2, v2 = score.veto_ratios(
            h_rates=tuple(h["rates_event_free"]), base_rates=tuple(b["rates_event_free"])
        )
        res["vetoes"] = {"V1": {"ratios": r1, "passes": v1}, "V2": {"ratios": r2, "passes": v2}}
        d["p10"] = tuple(r2)
    res["predictions"] = evaluate_predictions(d=d)
    res["d"] = d
    # hard negatives, placebo, conservation, scenarios, false calls
    hard = {}
    for folder in ("default", "placebo"):
        for arm in (*H_ARMS, "PELT", BENCH):
            calls = read_files(folder=scored / folder, arm=arm, suffix="_calls")
            h_files = read_files(folder=scored / folder, arm="H-nbr", prefix="hard_")
            if calls.height:
                hard[f"{folder}/{arm}"] = calls_on_hard(calls=calls, hard=h_files)
    res["hard_negatives"] = hard
    placebo = _load_set(folder=scored / "placebo", arms=("H-nbr", "PELT", BENCH))
    res["placebo_shares"] = share_table(tables=placebo)
    if all(placebo[a].height for a in ("H-nbr", BENCH)):
        res["placebo_c1"] = score.contrast(a=placebo["H-nbr"], b=placebo[BENCH], cell=cell, rng=rng)
    if all(placebo[a].height for a in ("H-nbr", "PELT")):
        res["placebo_c2"] = score.contrast(
            a=placebo["H-nbr"], b=placebo["PELT"], cell=cell, rng=rng
        )
    res["conservation"] = conservation_residual(
        calls=read_files(folder=scored / "default", arm="H-nbr", suffix="_calls")
    )
    scen = pl.concat(
        [read_files(folder=scored / "default", arm=a, prefix="scen_") for a in H_ARMS],
        how="diagonal_relaxed",
    )
    res["scenarios"] = scenario_table(scen=scen)
    res["false_calls"] = {
        a: false_calls(calls=read_files(folder=scored / "default", arm=a, suffix="_calls")).height
        for a in (*H_ARMS, "PELT")
    }
    # leave one out
    if hn.height and bench.height:
        res["loo_world"] = leave_one_out_flags(
            loo=score.leave_one_out(a=hn, b=bench, cell=cell, by="world"), point=d["c1"].point
        )
        res["loo_primary"] = leave_one_out_flags(
            loo=score.leave_one_out(a=hn, b=bench, cell=cell, by="site"), point=d["c1"].point
        )
    # timing, proportional shapes, grades
    res["timing"] = timing_error(rows=hn)
    if hn.height:
        res["prop_share"] = _by_prop_share(hn=hn, bench=bench)
        temp = hn.filter(
            pl.col("kind").is_in(list(TEMPORARY)) & pl.col("found") & (pl.col("days") >= 0)
        )
        res["neither_share"] = (
            float((temp["decoded_kind"] == "neither").mean())  # ty: ignore[invalid-argument-type]
            if temp.height and "decoded_kind" in temp.columns
            else None
        )
    # robustness, solar growth, seasonal bias
    sens = {}
    for c in SENSITIVITY:
        tabs = {a: cell_rows(rows=read_files(folder=scored / c, arm=a)) for a in ("H-own", "H-nbr")}
        tabs[BENCH] = cell_rows(rows=stored.sensitivity.get(c, pl.DataFrame()))
        sens[c] = share_table(tables=tabs)
    res["robustness"] = sens
    res["solar"] = _false_event_totals(folder=scored / "pv_growth")
    res["bias"] = _false_event_totals(folder=scored / "seasonal_bias")
    # verdict and audit triggers
    gates = gate_status(out=out)
    res["gates"] = gates
    c1 = d.get("c1")
    if c1 is None:
        res["verdict"] = "no C1 (the scored files are missing)"
    elif any(v is not True for k, v in gates.items() if k != "G3b") or gates.get("G3b") is False:
        res["verdict"] = "invalid (a gate did not pass or is missing)"
    else:
        below = c1.world_hi < 0 and c1.month_hi < 0
        res["verdict"] = score.verdict(c1=c1, v1=v1, v2=v2, c1_upper_below_zero=below)
    res["triggers"] = _triggers(res=res, d=d)
    return res


def _by_prop_share(*, hn: pl.DataFrame, bench: pl.DataFrame) -> dict[str, float]:
    """Found share of H-nbr and E1-J-sym by the generator's donor-profile share."""
    keys = hn.select("world", "site", "start", "prop_share").unique(
        subset=["world", "site", "start"]
    )
    other = bench.join(keys, on=["world", "site", "start"], how="left") if bench.height else bench
    out: dict[str, float] = {}
    for name, t in (("H-nbr", hn), (BENCH, other)):
        if not t.height or "prop_share" not in t.columns:
            continue
        for label, cond in (
            ("below 0.5", pl.col("prop_share") < 0.5),
            ("0.5 or more", pl.col("prop_share") >= 0.5),
        ):
            out[f"{name}, prop_share {label}"] = score.found_share(rows=t.filter(cond))
    return out


def _false_event_totals(*, folder: Path) -> dict:
    files = sorted(folder.glob("false_*.parquet")) if folder.exists() else []
    if not files:
        return {}
    t = pl.concat([pl.read_parquet(f) for f in files])
    g = t.group_by("arm").agg(
        false_events=pl.col("false_events").sum(),
        worlds=pl.len(),
        grown_primaries=pl.col("grown_primaries").sum(),
        mean_per_primary_year=pl.col("per_primary_year").mean(),
    )
    return {r["arm"]: r for r in g.iter_rows(named=True)}


def _triggers(*, res: dict, d: dict) -> dict[str, str]:
    """The audit triggers of the plan that the stored numbers can decide."""
    out: dict[str, str] = {}
    gates = res["gates"]
    out["1 gate failed"] = str(any(v is False for v in gates.values()))
    out["2 counted prediction failed (P1 or P2)"] = str(
        any(p["held"] is False for p in res["predictions"] if p["id"] in ("P1", "P2"))
    )
    flips = 0
    for key in ("loo_world", "loo_primary"):
        t = res.get(key)
        if t is not None and t.height:
            flips += int(t["flips"].sum()) + int(t["crosses_zero"].sum())
    out["4 leaving one world or Primary out crosses 0 or the +0.10 bar"] = str(flips > 0)
    c1 = d.get("c1")
    if c1 is not None and d.get("p4") is not None:
        half = (c1.world_hi - c1.world_lo) / 2
        out["5 H-own beats H-nbr by more than C1's half-width"] = str(-d["p4"] > half)
        out["6 C1 world interval width (compare with experiment 680's)"] = (
            f"{c1.world_hi - c1.world_lo:.3f}"
        )
    return out


def report(*, out: Path, stored: Stored | None = None) -> None:
    """Print every number the page quotes from the files under `out`."""
    res = compute(out=out, stored=load_stored() if stored is None else stored)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    for key, value in res.items():
        print(f"\n== {key}")
        if isinstance(value, pl.DataFrame):
            print(value)
        elif isinstance(value, dict) and any(isinstance(v, pl.DataFrame) for v in value.values()):
            for k, v in value.items():
                print(f"-- {k}")
                print(v)
        else:
            print(value)

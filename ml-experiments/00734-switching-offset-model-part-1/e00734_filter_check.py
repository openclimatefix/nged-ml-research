"""Read-only diagnostic: would a more restrictive switching model find more at equal call rates?

This is a read-only tuning-world diagnostic. It is not a counted contrast, and nothing in it may be
tuned on the scored worlds. It reads only the stored tuning outputs of experiment 734 (the 30
tuning worlds, seeds 734000 to 734029, and the real record before 2024-01-01), checks every seed
with `bank.guard_tuning`, decodes nothing, and writes under `filter_check/` of the experiment's
data folder.

Six filters are applied to the calls of each arm at its final tuned setting (read from
`params.json` at run time) and again at tau 0.5, where the arm has a tau:

- F1: the call lasts at least 1 day.
- F2: the call lasts at least 2 days.
- F3: the call is mirrored, meaning a neighbouring Primary has a call of the opposite sign that
  starts within 3 blocks (36 half-hours). This is the rule of `events.grade`, so the call would be
  graded "mirrored and returned" or "mirrored, not yet returned". The stored tables hold no
  grade (see below), so the diagnostic recomputes it from the stored calls and the neighbour list.
- F4: no other call at the same Primary overlaps the call; of two overlapping calls, the one with
  the higher confidence stays.
- F5: the call returns to baseline within the record (it has an end).
- F6: F1, F3, F4 and F5 together (F1, F3, F5 first, then F4 among the survivors).

What the stored outputs allow:

- The real record (`tune/<setting>_real.parquet`) holds every event with its start, end, amplitude,
  confidence, and class, but its `grade` and `partners` columns are the ungraded defaults (grading
  happens on the called set only). All six filters are therefore exact on the real record, with F3
  recomputed from the neighbour list of experiment 653.
- A tuning world (`tune/<setting>_<seed>.parquet`) holds one row per injected event, with its
  `found` flag at each tau, the start and end errors of the matching call, and the event's
  `edge_truth` flag. It holds no call table. F1, F2, and F5 are exact for matched calls (the call
  duration is the injected duration plus the end error minus the start error, and a matched call
  with no end error is open). F3 and F4 cannot be recomputed, so they are upper bounds: F3 keeps a
  found call when the injected event is an edge transfer (the call would have to be graded
  mirrored), and F4 removes a found call only when another found call at the same Primary overlaps
  it (unmatched calls are not stored). F6 inherits both bounds. Where an injected event is open
  ended, F5 drops its call, because the stored rows cannot show whether the call closed.
- The recall ceiling of a filter is the share of injected megawatts in a cell that the filter
  could ever find, given the injected events alone.

Run (read-only; creates `filter_check/`):

    uv run --no-sync python e00734_filter_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import e00734_bank as bank
import e00734_score as score
import e00734_tasks as tasks
import e00734_tuning as tuning

DAY: Final[int] = 48
MIRROR_WINDOW: Final[int] = 36
"""Half-hours: 3 blocks of 12, the window of `events.grade`."""
LEVEL_CLS: Final[str] = "level_change"
FILTERS: Final[tuple[str, ...]] = ("F0", "F1", "F2", "F3", "F4", "F5", "F6")
"""F0 is the unfiltered calls, for reference."""
EXACT_IN_WORLDS: Final[dict[str, bool]] = {
    "F0": True,
    "F1": True,
    "F2": True,
    "F3": False,
    "F4": False,
    "F5": True,
    "F6": False,
}
CELLS: Final[tuple[str, ...]] = (
    "long, shallow",
    "long, deep",
    "medium, shallow",
    "medium, deep",
    "short, shallow",
    "short, deep",
)
OUT_DIR: Final[Path] = tasks.OUT / "filter_check"


def call_days(*, calls: pl.DataFrame) -> pl.Series:
    """Duration of each call in days, infinite when the call has no end (`end` is -1)."""
    return (
        pl.select(
            pl.when(calls["end"] < 0)
            .then(pl.lit(float("inf")))
            .otherwise((calls["end"] - calls["start"]) / DAY)
        )
        .to_series()
        .alias("days")
    )


def mirror_flags(
    *, calls: pl.DataFrame, neighbours: dict[str, tuple[str, ...]], window: int = MIRROR_WINDOW
) -> list[bool]:
    """Whether each call has a mirror partner, by the rule of `events.grade`.

    A call is mirrored when a call that is not a level change, at a neighbouring Primary, has the
    opposite sign and starts within `window` half-hours. A level change is never mirrored.

    Args:
        calls: The called events, with `primary`, `start`, `amp_mw`, and `cls`.
        neighbours: The neighbours of every Primary.
        window: The half-width of the mirror window in half-hours.

    Returns:
        One flag per call, in row order.
    """
    rows = list(calls.iter_rows(named=True))
    flags = []
    for e in rows:
        if e["cls"] == LEVEL_CLS:
            flags.append(False)
            continue
        flags.append(
            any(
                n["cls"] != LEVEL_CLS
                and n["primary"] in neighbours.get(e["primary"], ())
                and np.sign(n["amp_mw"]) == -np.sign(e["amp_mw"])
                and abs(n["start"] - e["start"]) <= window
                for n in rows
            )
        )
    return flags


def non_overlapping(*, calls: pl.DataFrame) -> pl.DataFrame:
    """Keep one call at a time per Primary: of overlapping calls, the higher confidence stays.

    Calls are taken in order of falling confidence (ties by earlier start), and a call stays only
    if it overlaps no call that stayed. An open call (`end` is -1) runs to the end of the record.
    """
    kept: list[dict] = []
    for e in calls.sort(["confidence", "start"], descending=[True, False]).iter_rows(named=True):
        end = float("inf") if e["end"] < 0 else e["end"]
        clash = any(
            k["primary"] == e["primary"]
            and e["start"] < (float("inf") if k["end"] < 0 else k["end"])
            and k["start"] < end
            for k in kept
        )
        if not clash:
            kept.append(e)
    return pl.DataFrame(kept, schema=calls.schema) if kept else calls.clear()


def filtered_calls(
    *, calls: pl.DataFrame, neighbours: dict[str, tuple[str, ...]]
) -> dict[str, pl.DataFrame]:
    """The surviving calls of each filter F0 to F6 on a table of called events (exact)."""
    days = call_days(calls=calls)
    graded = calls.with_columns(
        days=days,
        mirrored=pl.Series(mirror_flags(calls=calls, neighbours=neighbours), dtype=pl.Boolean),
    ).with_columns(returned=pl.col("end") >= 0)
    f6 = non_overlapping(
        calls=graded.filter((pl.col("days") >= 1.0) & pl.col("mirrored") & pl.col("returned"))
    )
    return {
        "F0": graded,
        "F1": graded.filter(pl.col("days") >= 1.0),
        "F2": graded.filter(pl.col("days") >= 2.0),
        "F3": graded.filter(pl.col("mirrored")),
        "F4": non_overlapping(calls=graded),
        "F5": graded.filter(pl.col("returned")),
        "F6": f6,
    }


def world_flags(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Add the columns `keep_<F>` and `elig_<F>` to truth rows.

    `keep_<F>` says whether the matched call survives the filter, and `elig_<F>` says whether the
    injected event could ever be found under the filter.

    `keep_F3`, `keep_F4`, and `keep_F6` are upper bounds, as the module docstring says.
    """
    closed = pl.col("end") >= 0
    open_call = pl.col("end_err_hh").is_nan() | pl.col("end_err_hh").is_null()
    call_dur = pl.when(closed & ~open_call).then(
        pl.col("days") + (pl.col("end_err_hh") - pl.col("start_err_hh")) / DAY
    )
    call_days_expr = pl.when(closed & ~open_call).then(call_dur).otherwise(float("inf"))
    d = rows.with_columns(
        call_days=call_days_expr,
        call_returned=closed & ~open_call,
        call_start=pl.col("start") + pl.col("start_err_hh").fill_nan(0.0),
    ).with_columns(
        call_end=pl.when(pl.col("call_returned"))
        .then(pl.col("end") + pl.col("end_err_hh"))
        .otherwise(float("inf"))
    )
    d = d.with_columns(overlapped=_overlapped(rows=d))
    keep = {
        "F0": pl.lit(True),
        "F1": pl.col("call_days") >= 1.0,
        "F2": pl.col("call_days") >= 2.0,
        "F3": pl.col("edge_truth"),
        "F4": ~pl.col("overlapped"),
        "F5": pl.col("call_returned"),
        "F6": (pl.col("call_days") >= 1.0)
        & pl.col("edge_truth")
        & pl.col("call_returned")
        & ~pl.col("overlapped"),
    }
    open_truth = pl.col("days") < 0
    elig = {
        "F0": pl.lit(True),
        "F1": open_truth | (pl.col("days") >= 1.0),
        "F2": open_truth | (pl.col("days") >= 2.0),
        "F3": pl.col("edge_truth"),
        "F4": pl.lit(True),
        "F5": ~open_truth,
        "F6": (pl.col("days") >= 1.0) & pl.col("edge_truth") & ~open_truth,
    }
    return d.with_columns(
        *[keep[f].alias(f"keep_{f}") for f in FILTERS],
        *[elig[f].alias(f"elig_{f}") for f in FILTERS],
    )


def _overlapped(*, rows: pl.DataFrame) -> pl.Series:
    """Whether a found call overlaps a found call of larger injected size at the same Primary.

    The comparison is within one world and one tau. Larger injected size stands in for the higher
    confidence of F4, which the stored rows do not hold.
    """
    out = np.zeros(rows.height, dtype=bool)
    idx = rows.with_row_index("i")
    for _, g in idx.filter(pl.col("found")).group_by(["world", "tau", "site"], maintain_order=True):
        order = g.sort(g["amp_mw"].abs(), descending=True)
        kept: list[tuple[float, float]] = []
        for r in order.iter_rows(named=True):
            lo, hi = r["call_start"], r["call_end"]
            if any(lo < k_hi and k_lo < hi for k_lo, k_hi in kept):
                out[r["i"]] = True
            else:
                kept.append((lo, hi))
    return pl.Series("overlapped", out)


def found_by_filter(*, rows: pl.DataFrame, tau: float) -> list[dict]:
    """Found share and recall ceiling of each filter in each cell at one tau (MW-weighted)."""
    cells = score.add_cells(rows=world_flags(rows=rows.filter(pl.col("tau") == tau)))
    out = []
    for f in FILTERS:
        d = cells.with_columns(found_f=pl.col("found") & pl.col(f"keep_{f}"))
        for cell in (*CELLS, "long+medium"):
            sub = (
                d.filter(pl.col("cell").str.contains("^(long|medium)"))
                if cell == "long+medium"
                else d.filter(pl.col("cell") == cell)
            )
            w = sub["amp_mw"].abs()
            total = float(w.sum())
            out.append(
                {
                    "filter": f,
                    "cell": cell,
                    "tau": tau,
                    "n_injected": sub.height,
                    "found_share": score.found_share(rows=sub, col="found_f"),
                    "recall_ceiling": float((sub[f"elig_{f}"].cast(pl.Float64) * w).sum()) / total
                    if total > 0
                    else float("nan"),
                    "exact_in_worlds": EXACT_IN_WORLDS[f],
                }
            )
    return out


def real_rates(
    *,
    calls: pl.DataFrame,
    tau: float,
    years: float,
    sigma653: dict[str, float],
    neighbours: dict[str, tuple[str, ...]],
) -> list[dict]:
    """The number of calls and both call rates of each filter on the real record at one tau."""
    kept = calls.filter(pl.col("confidence") >= tau)
    out = []
    for f, c in filtered_calls(calls=kept, neighbours=neighbours).items():
        big = c.filter(
            pl.col("amp_mw").abs()
            >= 2.0 * pl.col("primary").replace_strict(sigma653, return_dtype=pl.Float64)
        )
        out.append(
            {
                "filter": f,
                "tau": tau,
                "n_calls": c.height,
                "all_rate": c.height / years,
                "big_rate": big.height / years,
            }
        )
    return out


def equal_rate_tau(
    *, by_tau: dict[float, dict[str, dict]], rates: dict, filter_name: str
) -> float | None:
    """The lowest tau at which the filtered call rates are at or below E1-J-sym's, else None.

    Args:
        by_tau: For each tau, each filter's row from `real_rates`.
        rates: E1-J-sym's `all` and `big` rates.
        filter_name: The filter to look up.

    Returns:
        The lowest tau that meets both rates, or None when no tau does.
    """
    for tau in sorted(by_tau):
        r = by_tau[tau][filter_name]
        if r["all_rate"] <= rates["all"] and r["big_rate"] <= rates["big"]:
            return tau
    return None


def load_neighbours(*, sites: set[str]) -> dict[str, tuple[str, ...]]:
    """The neighbour lists of experiment 653 (symmetrised, restricted to `sites`)."""
    from e00653_common import NEIGHBOURS  # ty: ignore[unresolved-import]

    raw = json.loads(NEIGHBOURS.read_text())
    nb = {a: {b for b in raw.get(a, []) if b in sites} for a in sites}
    for a in sites:
        for b in nb[a]:
            nb[b].add(a)
    return {a: tuple(sorted(v)) for a, v in nb.items()}


def final_settings(*, params: dict) -> list[dict]:
    """The setting file name, arm, and tuned tau of each arm, from `params.json`.

    The H-own file is the entry of the grid whose three rho values equal the chosen ones, and the
    PELT file is the position of the chosen penalty in the grid and its extension.
    """
    from e00734_pelt import PENALTY_GRID

    own = params["h_own"]
    idx = next(
        i
        for i, s in enumerate(tuning.ALL_SETTINGS)
        if all(
            np.isclose(v, own[k], rtol=1e-12, atol=0.0)
            for k, v in zip(tuning.RHO_KEYS, s, strict=True)
        )
    )
    extra = tuning.read_state(out=tasks.OUT)["pelt"]
    penalties = (*PENALTY_GRID, *(extra["values"] if extra else ()))
    p_idx = penalties.index(params["pelt"]["penalty"])
    tag = tuning.nbr_tag(out=tasks.OUT)
    return [
        {
            "arm": "H-nbr",
            "name": f"{tag}_k{params['h_nbr']['kappa']:g}",
            "tau": params["h_nbr"]["tau"],
        },
        {"arm": "H-own", "name": f"own{idx:02d}", "tau": own["tau"]},
        {"arm": "PELT", "name": f"pelt{p_idx:02d}", "tau": 1.0},
    ]


def _world_rows(*, name: str) -> pl.DataFrame:
    seeds = bank.guard_tuning(seeds=bank.TUNING_SEEDS, extra=bank.solar_seeds())
    paths = [tasks.OUT / "tune" / f"{name}_{s}.parquet" for s in seeds]
    paths += [p for s in seeds if (p := tasks.OUT / "tune" / f"{name}_tau_{s}.parquet").exists()]
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal")


def run() -> dict:
    """Run every arm at its final setting and write `filter_check/filter_check.json`."""
    params = json.loads((tasks.OUT / "params.json").read_text())
    rates = json.loads((tasks.OUT / "tune" / "e1jsym_rates.json").read_text())
    years = rates["primary_years"]
    sigma = json.loads(
        (
            tasks.OUT.parent
            / "00653-edge-state-event-table-penalised-segmentation"
            / "noise_units.json"
        ).read_text()
    )["sigma_block_mw"]
    result: dict[str, list] = {"worlds": [], "real": [], "equal_rate": []}
    for s in final_settings(params=params):
        calls = pl.read_parquet(tasks.OUT / "tune" / f"{s['name']}_real.parquet").filter(
            pl.col("start") < tasks.SCORED_FROM
        )
        nb = load_neighbours(sites=set(calls["primary"].unique().to_list()))
        rows = _world_rows(name=s["name"])
        taus_in_worlds = sorted(rows["tau"].unique().to_list())
        modes = {"tuned": s["tau"]} | ({"0.5": 0.5} if s["arm"] != "PELT" else {})
        for mode, tau in modes.items():
            for r in found_by_filter(rows=rows, tau=tau):
                result["worlds"].append({"arm": s["arm"], "setting": s["name"], "mode": mode} | r)
            for r in real_rates(calls=calls, tau=tau, years=years, sigma653=sigma, neighbours=nb):
                result["real"].append({"arm": s["arm"], "setting": s["name"], "mode": mode} | r)
        by_tau = {
            t: {
                r["filter"]: r
                for r in real_rates(calls=calls, tau=t, years=years, sigma653=sigma, neighbours=nb)
            }
            for t in taus_in_worlds
        }
        for f in FILTERS:
            t = equal_rate_tau(by_tau=by_tau, rates=rates, filter_name=f)
            row = {"arm": s["arm"], "setting": s["name"], "filter": f, "equal_rate_tau": t}
            if t is not None:
                table = {
                    r["cell"]: r["found_share"]
                    for r in found_by_filter(rows=rows, tau=t)
                    if r["filter"] == f
                }
                row |= {c: table[c] for c in ("long, shallow", "long, deep", "long+medium")}
            result["equal_rate"].append(row)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "filter_check.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def print_tables(*, result: dict) -> None:
    """Print the compact tables: found share, call rate, and equal-rate found share."""
    w = pl.DataFrame(result["worlds"])
    key = w.filter(pl.col("cell").is_in(["long, shallow", "long, deep", "long+medium"]))
    print("FOUND SHARE (tuning worlds) and recall ceiling; F3, F4, F6 are upper bounds")
    print(
        key.select(
            "arm",
            "mode",
            "filter",
            "cell",
            pl.col("found_share").round(3),
            pl.col("recall_ceiling").round(3),
        ).sort(["arm", "mode", "filter", "cell"])
    )
    print("CALLS ON THE REAL TUNING RECORD (rates per Primary-year)")
    print(pl.DataFrame(result["real"]).with_columns(pl.col("all_rate", "big_rate").round(3)))
    print("AT E1-J-SYM'S CALL RATES")
    print(pl.DataFrame(result["equal_rate"]))


if __name__ == "__main__":
    print_tables(result=run())

# ruff: noqa: E501, PERF401, PLR0915
"""Experiment 723, job W3: the two controls at D12 and D3, and the read-outs at D12.

Run it only through `run.sh --go e00723_w3.py`, after job W2 has returned `supported` (the job
refuses to start otherwise), and after the manager has replied to the message "fits starting".
Job W3 is part of the same experiment and is not a further attempt.

**Controls, with bars.** D12's anchor stretches, where night import fixes the sign, and D3's
post-bounce stretches. A control passes when at most `floor(0.02 * n)` of its `n` scorable
stretches give export evidence, and is void under 30 scorable stretches. The farm sensor binds, and
a switching-log mask applies when job W0 built one.

**Read-outs, no bar, written after the verdict file.** The shares of D12's one-stretch post-bounce
days from April to September on which `C` is above +0.25, below -0.25, and in the dead band, for the
farm sensor and the meter sensor; the pooled fold ratio of D12 on the same days with and without
the farm sensor; and D12's dark-hour fold ratio. **The pooled fold ratio at D12 is never a label for
a day, a stretch, or a half-hour, and never becomes a prior.**

Outputs: `w3_results.csv`, `w3_verdict.json`, `w3_readouts.csv`, `run_log_w3.json`.
"""

from pathlib import Path
from typing import Any, Final

import e00722_data as data  # ty: ignore[unresolved-import]
import e00722_sky as sky  # ty: ignore[unresolved-import]
import e00723_common as common
import e00723_core as core
import e00723_guard as guard
import e00723_score as score
import e00723_w2 as w2
import e00723_world as world
import numpy as np
import polars as pl

FOLDER: Final[Path] = guard.FOLDER
"""This experiment's code folder."""

OUTPUTS: Final[tuple[str, ...]] = ("w3_results.csv", common.W3_VERDICT_NAME, "w3_readouts.csv")
"""Every file the job writes outside `PRIVATE/`."""


def control_population(*, table: pl.DataFrame, which: str) -> pl.DataFrame:
    """Return the scorable stretches of a control under the farm sensor.

    `anchor` is D12's anchor stretches (the first and last stretch of a day) with at least 8
    usable pairs, on every day. `post_bounce` is D3's post-bounce stretches with at least 8 usable
    pairs on days with exactly one post-bounce stretch, the population that job W2 calibrates and
    that the read-outs at D12 use. RULED (design lead).
    """
    enough = pl.col("n_farm") >= core.MIN_USABLE_PAIRS
    if which == "anchor":
        return table.filter(~pl.col("is_post_bounce") & enough)
    return table.filter(pl.col("is_post_bounce") & (pl.col("n_post_bounce_in_day") == 1) & enough)


def control_outcome(*, rows: pl.DataFrame) -> tuple[str, dict[str, int]]:
    """Return a control's outcome and counts.

    The control passes when at most `floor(0.02 * n)` of `n` scorable stretches give export evidence.
    """
    c = rows["c_farm"].to_numpy() if rows.height else np.array([])
    hits = int((c > core.DEAD_BAND).sum())
    return core.false_export_bar(n_export_evidence=hits, n_scorable_import=rows.height), {
        "n": rows.height,
        "n_export_evidence": hits,
        "allowance": core.floor_allowance(n=rows.height),
    }


def one_stretch_days(*, table: pl.DataFrame, month_of_day: np.ndarray) -> pl.DataFrame:
    """Return the post-bounce stretches on days with exactly one such stretch, from April to September."""
    in_season = set(np.flatnonzero(np.isin(month_of_day, core.TEST_MONTHS)).tolist())
    return table.filter(
        pl.col("is_post_bounce")
        & (pl.col("n_post_bounce_in_day") == 1)
        & pl.col("day").is_in(in_season)
    )


def shares_of(*, rows: pl.DataFrame, sensor: str) -> dict[str, float]:
    """Return the shares of stretches with `C` above +0.25, below -0.25, and in the dead band.

    A stretch with under 8 usable pairs is in the dead band.
    """
    n = rows.height
    ev = [
        core.evidence_of(c=c, n_usable=k)
        for c, k in zip(rows[f"c_{sensor}"].to_list(), rows[f"n_{sensor}"].to_list(), strict=True)
    ]
    n_exp, n_imp = ev.count("export"), ev.count("import")
    return {
        "n_days": n,
        "share_export": n_exp / n if n else float("nan"),
        "share_import": n_imp / n if n else float("nan"),
        "share_dead_band": (n - n_exp - n_imp) / n if n else float("nan"),
        "share_export_of_evidence": n_exp / (n_exp + n_imp) if (n_exp + n_imp) else float("nan"),
    }


def run(*, gate: guard.TimingGate, inputs: list[Path], info: dict[str, Any]) -> None:
    """Run job W3; `inputs` and `info` fill the run log as the job reads."""
    common.assert_launch()
    w0 = common.require_w0()
    common.require_w2_supported()
    info["attempt"] = guard.begin_attempt(job="w3", names=(*OUTPUTS, "run_log_w3.json"))
    log = guard.ReadLog()
    ctx = common.load_context(log=log, inputs=inputs)
    info["power_table_delta_version"] = data.power_table_version()
    info["read_order"] = log.as_list()
    info["holdout"] = log.holdout
    membership = w2.read_membership()
    mask_available = bool(w0["mask_available"])
    windows = w2.read_windows(mask_available=mask_available)
    guard.say("FITS STARTING: job W3 (two controls)")
    gate.mark()
    tables: dict[str, pl.DataFrame] = {}
    for host in ("D12", "D3"):
        s = score.score_world(
            ctx=ctx,
            spec=score.WorldSpec(f"real_{host}", "real", host, None, None),
            membership=membership,
            windows=windows,
        )
        if s is not None:
            tables[host] = s.table
    gate.check(done=1, total=1)
    outcomes: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for name, host, which in (
        ("d12_anchor_stretches", "D12", "anchor"),
        ("d3_post_bounce_stretches", "D3", "post_bounce"),
    ):
        if host not in tables:
            outcomes[name], counts[name] = "void", {"n": 0, "n_export_evidence": 0, "allowance": 0}
            continue
        outcomes[name], counts[name] = control_outcome(
            rows=control_population(table=tables[host], which=which)
        )
    verdict, row = core.verdict_w3(control_outcomes=outcomes, masked=mask_available)
    results = [
        {
            "control": k,
            "outcome": v,
            **counts[k],
            "mask": "masked" if mask_available else "transfers not masked",
        }
        for k, v in outcomes.items()
    ]
    guard.write_public_csv(frame=pl.DataFrame(results), name="w3_results.csv")
    guard.write_public_json(
        payload={
            "verdict": verdict,
            "row": row,
            "outcomes": outcomes,
            "counts": counts,
            "mask_available": mask_available,
        },
        name=common.W3_VERDICT_NAME,
    )
    guard.say("VERDICT:", verdict, "| controls:", outcomes)
    # ---- Read-outs: after the verdict file, no bar. ----
    out: list[dict[str, Any]] = []
    start = ctx.start_row
    month_of_day = world.month_of_rows(n=core.N_GRID)[start :: core.HALF_HOURS_PER_DAY]
    d12_table = tables.get("D12")
    sel_days: set[int] = set()
    if d12_table is not None:
        sel = one_stretch_days(table=d12_table, month_of_day=month_of_day)
        sel_days = set(sel["day"].to_list())
        for sensor in ("farm", "meter"):
            out.append(
                {"readout": f"d12_shares_{sensor}_sensor", **shares_of(rows=sel, sensor=sensor)}
            )
    key = ctx.eng.key
    quiet = world.core722.quiet_mask(engine_change=ctx.eng.change, calendar_ok=ctx.eng.has_row)
    day = sky.day_of(n=core.N_GRID) - start // core.HALF_HOURS_PER_DAY
    masked = score.windows_mask(windows=windows, host="D12")
    weights = core.day_weights(n_days=ctx.n_days)
    d12 = ctx.series["D12"]
    farms = membership.get("D12", [])
    on_days = np.isin(day, list(sel_days)) if sel_days else np.zeros(core.N_GRID, dtype=bool)
    lit_rows = (
        world.closure_rows(zenith=ctx.sky_of["D12"].zenith, masked=masked, start=start, kind="lit")
        & on_days
    )
    dark_rows = world.closure_rows(
        zenith=ctx.sky_of["D12"].zenith, masked=masked, start=start, kind="dark"
    )
    resid = ctx.residual(host="D12")
    fits = [
        ("d12_pooled_fold_ratio", lit_rows, {"d12": d12}),
        ("d12_dark_hour_fold_ratio", dark_rows, {"d12": d12}),
    ]
    if len(farms) >= 2:
        fits.append(
            (
                "d12_pooled_fold_ratio_with_farm_sensor",
                lit_rows,
                {"d12": d12, "farm_sensor": ctx.farm_mean_share(farms=farms)},
            )
        )
    for name, rows, regs in fits:
        f = world.closure_fit(
            response=resid,
            regressors=regs,
            rows=rows,
            key=key,
            quiet=quiet,
            day=day,
            weights=weights,
        )
        out.append(
            {
                "readout": name,
                "coef": float(f.coef[0]),
                "se": float(f.se[0]),
                "lo": float(f.lo[0]),
                "hi": float(f.hi[0]),
                "n_rows": f.n_rows,
                "n_days": f.n_days,
            }
        )
    guard.write_public_csv(
        frame=pl.DataFrame(out, infer_schema_length=None), name="w3_readouts.csv"
    )
    guard.say("JOB W3 DONE; core-seconds", round(gate.elapsed(), 1))


def main() -> None:
    """Run job W3 and write its run log even when the job crashes."""
    gate = guard.TimingGate(cap_core_seconds=core.CAP_CORE_SECONDS)
    inputs: list[Path] = []
    info: dict[str, Any] = {}
    with guard.run_log(job="w3", gate=gate, inputs=inputs, info=info):
        run(gate=gate, inputs=inputs, info=info)


if __name__ == "__main__":
    main()

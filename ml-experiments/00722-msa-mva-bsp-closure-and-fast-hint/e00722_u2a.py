"""Experiment 722, job 2 (rung U2a): the BSP meter against the engine record and folded D15.

Run it only through `run.sh --go e00722_u2a.py`, after the Opus code review and after job 1. The
bars and the decision rule are the named constants and functions of `e00722_core.py`.

**The leave-out residual never subtracts the scored child's own series.** For the engine test the
residual leaves D8 out. For the D15 test the residual leaves D15 out, and D15's signed series
enters only as the folded regressor `m = |X|` and as the truth. Every result is also reported in
the all-children form, which subtracts the scored reading as well and differs by exactly that
reading (so the ratio is the leave-out ratio minus 1).

Outputs: `u2a_results.csv`, `u2a_verdict.json`, `run_log_u2a.json`. Stdout prints counts, shares,
ratios, and intervals only.
"""

import os
from pathlib import Path
from typing import Any, Final

import e00722_common as common
import e00722_core as core
import e00722_data as data
import e00722_guard as guard
import e00722_sky as sky
import numpy as np
import polars as pl

FOLDER: Final[Path] = Path(__file__).resolve().parent
"""This experiment's code folder."""


def hour_end_rows(*, n: int) -> np.ndarray:
    """Return True on grid rows whose end stamp falls on the hour (the non-overlapping hours).

    The first half-hour ends at 00:30, so the odd rows end on the hour.
    """
    return np.arange(n) % 2 == 1


def masked(*, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return `x` with NaN outside `rows`."""
    return np.where(rows, x, np.nan)


def folded_child_test(
    *,
    flow: np.ndarray,
    leave_out_residual: np.ndarray,
    key: np.ndarray,
    quiet: np.ndarray,
    day: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    """Score one child whose signed series is folded to a magnitude, against its hidden sign.

    Args:
        flow: The child's true signed series `X` (positive for import), on the grid.
        leave_out_residual: `R(-i)` for that child, on the grid (NaN where unavailable).
        key: The matched key of each half-hour.
        quiet: True at quiet half-hours.
        day: The day of each row.
        weights: The master draw of whole days.

    Returns:
        The closure fits (de-meaned, raw, and all-children), the area under the curve with and
        without the crossing exclusion, the raw-change area (reported with no bar), the placebo
        area, the reach, and the counts. The de-meaned closure slope and area bind (the ruling on
        B9); the raw forms are reported only.
    """
    rows = hour_end_rows(n=len(flow))
    lag = core.HOURLY_LAG_HALF_HOURS
    folded = np.abs(flow)

    def dm(x: np.ndarray) -> np.ndarray:
        return core.demean_matched(x=core.change(x=x, lag=lag), key=key, quiet=quiet)

    d_r, d_x, d_m = dm(leave_out_residual), dm(flow), dm(folded)
    d_all = dm(core.all_children_residual(leave_out=leave_out_residual, reading=folded))
    d_r_raw = core.change(x=leave_out_residual, lag=lag)
    d_x_raw = core.change(x=flow, lag=lag)
    closure = core.bootstrap_ols(
        y=masked(x=d_r, rows=rows),
        x=masked(x=d_x, rows=rows),
        day=day,
        weights=weights,
        names=["slope"],
    )
    raw = core.bootstrap_ols(
        y=masked(x=d_r_raw, rows=rows),
        x=masked(x=d_x_raw, rows=rows),
        day=day,
        weights=weights,
        names=["slope"],
    )
    all_children = core.bootstrap_ols(
        y=masked(x=d_all, rows=rows),
        x=masked(x=d_x, rows=rows),
        day=day,
        weights=weights,
        names=["slope"],
    )
    table = core.hourly_sign_rows(d_residual=d_r, d_folded=d_m, flow=flow, hour_end=rows)
    scored = table["scorable"]
    auc = core.bootstrap_auc(
        score=np.where(scored, table["ratio"], np.nan),
        label=np.where(scored, table["label"], np.nan),
        day=day,
        weights=weights,
    )
    d_m_raw = core.change(x=folded, lag=lag)
    table_raw = core.hourly_sign_rows(
        d_residual=d_r_raw, d_folded=d_m_raw, flow=flow, hour_end=rows
    )
    scored_raw = table_raw["scorable"]
    auc_raw = core.bootstrap_auc(
        score=np.where(scored_raw, table_raw["ratio"], np.nan),
        label=np.where(scored_raw, table_raw["label"], np.nan),
        day=day,
        weights=weights,
    )
    with_crossing = table["size_ok"]
    sign_at_end = np.where(flow > 0, 1.0, 0.0)
    auc_all = core.bootstrap_auc(
        score=np.where(with_crossing, table["ratio"], np.nan),
        label=np.where(with_crossing, sign_at_end, np.nan),
        day=day,
        weights=weights,
    )
    shifted_m = core.shift_rows(x=d_m, rows=core.PLACEBO_SHIFT_DAYS * core.HALF_HOURS_PER_DAY)
    table_p = core.hourly_sign_rows(d_residual=d_r, d_folded=shifted_m, flow=flow, hour_end=rows)
    scored_p = table_p["scorable"]
    auc_p = core.bootstrap_auc(
        score=np.where(scored_p, table_p["ratio"], np.nan),
        label=np.where(scored_p, table_p["label"], np.nan),
        day=day,
        weights=weights,
    )
    n_finite = int(table["finite"].sum())
    return {
        "closure": closure,
        "closure_raw": raw,
        "closure_all_children": all_children,
        "auc": auc,
        "auc_with_crossing": auc_all,
        "auc_raw": auc_raw,
        "n_scorable_raw": int(scored_raw.sum()),
        "auc_placebo": auc_p,
        "n_scorable": int(scored.sum()),
        "n_crossing_excluded": int(table["crossing"].sum()),
        "n_scorable_placebo": int(scored_p.sum()),
        "n_hours": n_finite,
        "reach": float(table["size_ok"].sum() / n_finite) if n_finite else float("nan"),
    }


def engine_test(
    *,
    residual: np.ndarray,
    d8_reading: np.ndarray,
    record: np.ndarray,
    key: np.ndarray,
    quiet: np.ndarray,
    day: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    """Regress the 2-hour change of `R(-D8)`, less its matched quiet-time mean, on the record's.

    The placebos roll the record's 2-hour change by 7 and by 14 days inside the record's own
    span, so that every row is kept.
    """
    lag = core.ENGINE_LAG_HALF_HOURS
    x = core.change(x=record, lag=lag)
    y = core.demean_matched(x=core.change(x=residual, lag=lag), key=key, quiet=quiet)
    y_all = core.demean_matched(
        x=core.change(
            x=core.all_children_residual(leave_out=residual, reading=d8_reading), lag=lag
        ),
        key=key,
        quiet=quiet,
    )
    main = core.bootstrap_ols(y=y, x=x, day=day, weights=weights, names=["slope"])
    all_children = core.bootstrap_ols(y=y_all, x=x, day=day, weights=weights, names=["slope"])
    span = np.flatnonzero(np.isfinite(record))
    placebos = {}
    for days in core.ENGINE_PLACEBO_SHIFT_DAYS:
        shifted = x.copy()
        if span.size:
            a, b = int(span[0]), int(span[-1]) + 1
            shifted[a:b] = core.circular_shift(x=x[a:b], rows=days * core.HALF_HOURS_PER_DAY)
        placebos[days] = core.bootstrap_ols(
            y=y, x=shifted, day=day, weights=weights, names=["slope"]
        )
    return {"main": main, "all_children": all_children, "placebos": placebos}


def fit_row(*, variant: str, test: str, form: str, fit: core.Fit, outcome: str) -> dict[str, Any]:
    """Return one line of `u2a_results.csv` from a one-regressor fit."""
    c, se, lo, hi = fit.get(fit.names[0])
    return {
        "variant": variant,
        "test": test,
        "residual_form": form,
        "estimate": c,
        "se": se,
        "lo": lo,
        "hi": hi,
        "n_rows": fit.n_rows,
        "n_days_used": fit.n_days,
        "outcome": outcome,
    }


def auc_row(
    *,
    variant: str,
    test: str,
    form: str,
    auc: tuple[float, float, float, float],
    n: int,
    outcome: str,
) -> dict[str, Any]:
    """Return one line of `u2a_results.csv` for an area under the curve."""
    return {
        "variant": variant,
        "test": test,
        "residual_form": form,
        "estimate": auc[0],
        "se": auc[1],
        "lo": auc[2],
        "hi": auc[3],
        "n_rows": n,
        "n_days_used": None,
        "outcome": outcome,
    }


def run(*, gate: guard.TimingGate, inputs: list[Path], info: dict[str, Any]) -> None:  # noqa: PLR0915
    """Run job 2; `inputs` and `info` fill the run log as the job reads."""
    common.assert_environment(env=dict(os.environ))
    checks = common.read_checks()
    common.require_stamps(checks=checks)
    log = guard.ReadLog()
    inputs += [data.SITE_MAP_PATH, data.METADATA_PATH, data.HIERARCHY_PATH, data.ENGINE_FRAME_PATH]
    info["power_table_delta_version"] = data.power_table_version()
    topology = common.load_topology()
    classes = common.read_classes()
    labels_a = core.BSP_A_PRIMARIES
    labels_b = core.BSP_B_PRIMARIES
    ids = [
        *topology.ids_of(labels=(*labels_a, *labels_b)).values(),
        *topology.bsp_meter.values(),
        *classes["generator_id"].unique().to_list(),
    ]
    power = data.load_power(ids=ids, log=log, name="primaries_bsp_meters_generators")
    series = common.label_arrays(topology=topology, power=power)
    eng = data.load_engine_frame(log=log)
    record, record_path, _ = data.load_engine_record(log=log)
    inputs += [record_path]
    quiet = core.quiet_mask(engine_change=eng.change, calendar_ok=eng.has_row)
    day = sky.day_of(n=core.N_GRID)
    n_days = int(day.max()) + 1
    weights = core.day_weights(
        n_days=n_days, n_boot=core.N_BOOT, rng=np.random.default_rng(core.SEED)
    )
    start_a = common.both_start(series=series, transformers=core.TWO_TRANSFORMERS_2025)
    guard.say("FITS STARTING: job 2 (U2a)")
    rows: list[dict[str, Any]] = []
    verdicts: dict[str, Any] = {}
    units_per_variant = 3
    total_units = units_per_variant * len(common.VARIANTS)
    units_done = 0
    gate.mark()
    for variant in common.VARIANTS:
        gens = {
            b: [
                power[g]
                for g in common.generator_ids_to_subtract(classes=classes, bsp=b, variant=variant)
            ]
            for b in ("A", "B")
        }
        res_d8 = common.spanned_residual(
            bsp=power[topology.bsp_meter["A"]],
            series=series,
            children=labels_a,
            transformers=core.TWO_TRANSFORMERS_2025,
            leave_out="D8",
            generators=gens["A"],
            start=start_a,
        )
        res_d15 = common.spanned_residual(
            bsp=power[topology.bsp_meter["A"]],
            series=series,
            children=labels_a,
            transformers=core.TWO_TRANSFORMERS_2025,
            leave_out="D15",
            generators=gens["A"],
            start=start_a,
        )
        res_d15 = np.where(np.arange(core.N_GRID) >= start_a, res_d15, np.nan)
        res_d10 = common.spanned_residual(
            bsp=power[topology.bsp_meter["B"]],
            series=series,
            children=labels_b,
            transformers=(),
            leave_out="D10",
            generators=gens["B"],
            start=0,
        )
        eng_res = engine_test(
            residual=res_d8,
            d8_reading=series["D8"],
            record=record,
            key=eng.key,
            quiet=quiet,
            day=day,
            weights=weights,
        )
        units_done += 1
        gate.check(done=units_done, total=total_units)
        m = eng_res["main"]
        placebo_ci = {d: (f.lo[0], f.hi[0]) for d, f in eng_res["placebos"].items()}
        engine_outcome = (
            "void"
            if checks["engine_test_void"]
            else core.engine_bar(
                lo=float(m.lo[0]), hi=float(m.hi[0]), se=float(m.se[0]), shifted=placebo_ci
            )
        )
        rows.append(
            fit_row(variant=variant, test="engine", form="leave_out", fit=m, outcome=engine_outcome)
        )
        rows.append(
            fit_row(
                variant=variant,
                test="engine",
                form="all_children",
                fit=eng_res["all_children"],
                outcome="report",
            )
        )
        rows += [
            fit_row(
                variant=variant,
                test=f"engine_shift_{d}_days",
                form="leave_out",
                fit=f,
                outcome="report",
            )
            for d, f in eng_res["placebos"].items()
        ]
        d15 = folded_child_test(
            flow=series["D15"],
            leave_out_residual=res_d15,
            key=eng.key,
            quiet=quiet,
            day=day,
            weights=weights,
        )
        units_done += 1
        gate.check(done=units_done, total=total_units)
        cl = d15["closure"]
        closure_outcome = core.d15_closure_bar(
            lo=float(cl.lo[0]), hi=float(cl.hi[0]), se=float(cl.se[0])
        )
        sign_outcome = core.d15_sign_bar(auc=d15["auc"][0], n_hours=d15["n_scorable"])
        placebo_outcome = core.d15_placebo_bar(
            sign_outcome=sign_outcome,
            lo=d15["auc_placebo"][2],
            hi=d15["auc_placebo"][3],
            n_hours=d15["n_scorable_placebo"],
        )
        rows.append(
            fit_row(
                variant=variant,
                test="d15_closure",
                form="leave_out",
                fit=cl,
                outcome=closure_outcome,
            )
        )
        rows.append(
            fit_row(
                variant=variant,
                test="d15_closure_raw",
                form="leave_out",
                fit=d15["closure_raw"],
                outcome="report",
            )
        )
        rows.append(
            fit_row(
                variant=variant,
                test="d15_closure",
                form="all_children",
                fit=d15["closure_all_children"],
                outcome="report",
            )
        )
        rows.append(
            auc_row(
                variant=variant,
                test="d15_sign",
                form="leave_out",
                auc=d15["auc"],
                n=d15["n_scorable"],
                outcome=sign_outcome,
            )
        )
        rows.append(
            auc_row(
                variant=variant,
                test="d15_sign_with_crossing",
                form="leave_out",
                auc=d15["auc_with_crossing"],
                n=d15["n_hours"],
                outcome="report",
            )
        )
        rows.append(
            auc_row(
                variant=variant,
                test="d15_sign_raw",
                form="leave_out",
                auc=d15["auc_raw"],
                n=d15["n_scorable_raw"],
                outcome="report",
            )
        )
        rows.append(
            auc_row(
                variant=variant,
                test="d15_placebo",
                form="leave_out",
                auc=d15["auc_placebo"],
                n=d15["n_scorable_placebo"],
                outcome=placebo_outcome,
            )
        )
        rows.append(
            {
                "variant": variant,
                "test": "d15_reach",
                "residual_form": "leave_out",
                "estimate": d15["reach"],
                "se": None,
                "lo": None,
                "hi": None,
                "n_rows": d15["n_hours"],
                "n_days_used": None,
                "outcome": "report",
            }
        )
        d10 = folded_child_test(
            flow=series["D10"],
            leave_out_residual=res_d10,
            key=eng.key,
            quiet=quiet,
            day=day,
            weights=weights,
        )
        units_done += 1
        gate.check(done=units_done, total=total_units)
        rows.append(
            fit_row(
                variant=variant,
                test="d10_closure",
                form="leave_out",
                fit=d10["closure"],
                outcome="report",
            )
        )
        rows.append(
            auc_row(
                variant=variant,
                test="d10_sign",
                form="leave_out",
                auc=d10["auc"],
                n=d10["n_scorable"],
                outcome="report",
            )
        )
        outcomes = {
            "engine": engine_outcome,
            "d15_closure": closure_outcome,
            "d15_sign": sign_outcome,
            "d15_placebo": placebo_outcome,
        }
        verdicts[variant] = {
            "verdict": core.verdict_u2a(outcomes=outcomes),
            "outcomes": outcomes,
            "d15_n_scorable": d15["n_scorable"],
            "d15_n_crossing_excluded": d15["n_crossing_excluded"],
            "d15_reach": d15["reach"],
            "d10_n_scorable": d10["n_scorable"],
            "engine_rows": m.n_rows,
        }
        guard.say(
            variant,
            "engine slope",
            round(float(m.coef[0]), 3),
            "interval",
            round(float(m.lo[0]), 3),
            round(float(m.hi[0]), 3),
            "outcome",
            engine_outcome,
            "| D15 closure",
            round(float(cl.coef[0]), 3),
            "outcome",
            closure_outcome,
            "| D15 area",
            round(d15["auc"][0], 3),
            "hours",
            d15["n_scorable"],
            "outcome",
            sign_outcome,
            "placebo",
            placebo_outcome,
            "| verdict",
            verdicts[variant]["verdict"],
        )
    frame = pl.DataFrame(rows, infer_schema_length=None)
    guard.write_public_csv(frame=frame, name="u2a_results.csv")
    payload = {
        "verdict": verdicts["primary"]["verdict"],
        "stops_u2b": verdicts["primary"]["verdict"] == "not supported",
        "variants": verdicts,
        "binding_variant": "primary",
    }
    guard.write_public_json(payload=payload, name=common.U2A_VERDICT_NAME)
    guard.say(
        "LAST FIT ENDED: job 2 (U2a); verdict",
        payload["verdict"],
        "core-seconds",
        round(gate.elapsed(), 1),
    )


def main() -> None:
    """Run job 2 and write its run log even when the job crashes."""
    gate = guard.TimingGate()
    inputs: list[Path] = []
    info: dict[str, Any] = {}
    with guard.run_log(job="u2a", folder=FOLDER, gate=gate, inputs=inputs, info=info):
        run(gate=gate, inputs=inputs, info=info)


if __name__ == "__main__":
    main()

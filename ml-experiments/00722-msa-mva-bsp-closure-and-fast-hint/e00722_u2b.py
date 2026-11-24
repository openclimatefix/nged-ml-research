# ruff: noqa: E501
"""Experiment 722, job 3 (rung U2b): one joint regression at BSP A, with a sky term.

Run it only through `run.sh --go e00722_u2b.py`, after the Opus code review, job 1, and a job 2
verdict that is not `not supported`.

**The response is the half-hour change of the BSP meter minus the signed children's physical
flows, plus the outside generators.** That residual is `U + sum of the four MVA children's flows`.
The regressors are the half-hour changes of the readings of D3, D8, D12, and D22, and the fast sky
change at D12. Every series is de-meaned at matched quiet times first (F3), and rows within two
half-hours of a change in the engine state are dropped and counted (S4).

The calibration fit uses dark half-hours and the half-hours of overcast days. The test fit uses
the post-bounce stretches of intermittent days from April to September, and the bars bind on days
with exactly one such stretch (F4). The later span, where both transformers of 2025 are
subtracted, binds, and the earlier span is reported apart.

Outputs: `u2b_results.csv`, `u2b_verdict.json`, `u2b_d8_exploratory.csv`, `run_log_u2b.json`.
Stdout prints counts, shares, ratios, and intervals only.
"""

import json
import os
from pathlib import Path
from typing import Any, Final, cast

import e00722_common as common
import e00722_core as core
import e00722_d8core as d8core
import e00722_data as data
import e00722_guard as guard
import e00722_sky as sky
import numpy as np
import polars as pl

FOLDER: Final[Path] = Path(__file__).resolve().parent
"""This experiment's code folder."""

REGRESSORS_A: Final[tuple[str, ...]] = ("D3", "D8", "D12", "D22", "sky")
"""The regressors of BSP A's joint regression."""

REGRESSORS_B: Final[tuple[str, ...]] = ("D2", "D4", "D19", "D21", "sky")
"""The regressors of BSP B's control regression."""

SKY_REFERENCE: Final[dict[str, str]] = {"A": "D12", "B": "D2"}
"""The Primary whose CAMS point gives each BSP's sky rows and its fast sky term."""

TERCILES: Final[tuple[str, ...]] = ("lower", "middle", "upper")
"""The thirds of the lower of the two readings around an engine event."""

ERAS: Final[tuple[str, ...]] = ("net", "solar", "fade")
"""The three eras of D8's series."""

M_BOOT_EVENTS: Final[int] = 1000
"""Bootstrap replicates over events in the exploratory table."""


# ---------------------------------------------------------------------------------------------
# Pure helpers (tested on synthetic arrays)
# ---------------------------------------------------------------------------------------------
fill_sky_change = sky.fill_sky_change
"""The fast sky change with the dark half-hours set to 0 (see `e00722_sky.fill_sky_change`)."""


def calibration_rows(*, zenith: np.ndarray, class_of_row: np.ndarray) -> np.ndarray:
    """Return dark half-hours and the half-hours of overcast days (the calibration fit's rows)."""
    has_sky = np.isfinite(zenith)
    dark = has_sky & ((90.0 - zenith) <= core.DARK_ELEVATION_DEG)
    overcast = has_sky & (class_of_row == sky.CLASS_OVERCAST)
    return dark | overcast


def stretch_rows(
    *,
    reading: np.ndarray,
    lit: np.ndarray,
    classes: np.ndarray,
    month: np.ndarray,
    capacity_mw: float,
    want: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the test fit's rows and the number of post-bounce stretches of every day.

    Args:
        reading: The Primary's raw reading on the grid.
        lit: Daylight half-hours at the Primary's CAMS point.
        classes: The cloud class of each day.
        month: The calendar month of each day.
        capacity_mw: The Primary's capacity.
        want: `one` for days with exactly one post-bounce stretch, `many` for days with 3 or
            more, and `all` for every post-bounce stretch.

    Returns:
        The rows of the post-bounce stretches of intermittent days in the test months, and the
        count of post-bounce stretches per day (-1 for a day that cannot be counted).
    """
    level = sky.touch_level(reading=reading, lit=lit, capacity_mw=capacity_mw)
    touch = sky.touches(reading=reading, lit=lit, level=level)
    spans = sky.day_stretches(touch=touch, lit=lit, usable=np.isfinite(reading))
    n_days = len(classes)
    counts = sky.post_bounce_counts(stretches=spans, n_days=n_days)
    eligible = (classes == sky.CLASS_INTERMITTENT) & np.isin(month, core.TEST_MONTHS)
    pick = {
        "one": counts == core.ONE_STRETCH,
        "many": counts >= core.MANY_STRETCHES,
        "all": counts >= 1,
    }[want]
    out = np.zeros(len(reading), dtype=bool)
    for d in np.flatnonzero(eligible & pick):
        for lo, hi, post in spans.get(int(d), []):
            if post:
                out[lo : hi + 1] = True
    return out, counts


def day_count_shares(*, counts: np.ndarray, eligible: np.ndarray) -> dict[str, float]:
    """Return the share of eligible days with 0, 1, 2, and 3 or more post-bounce stretches."""
    c = counts[eligible & (counts >= 0)]
    if c.size == 0:
        return {"0": float("nan"), "1": float("nan"), "2": float("nan"), "3_or_more": float("nan")}
    return {
        "0": float((c == 0).mean()),
        "1": float((c == 1).mean()),
        "2": float((c == 2).mean()),
        "3_or_more": float((c >= core.MANY_STRETCHES).mean()),
    }


def excess_steps(
    *, step: np.ndarray, events: np.ndarray, key: np.ndarray, cand: dict[int, np.ndarray]
) -> np.ndarray:
    """Return each event's step less the mean step of its matched quiet pool (NaN when the pool is small)."""
    out = np.full(len(events), np.nan)
    for j, h in enumerate(events):
        pool = cand.get(int(key[h]))
        if pool is None or not np.isfinite(step[h]):
            continue
        pool = pool[np.isfinite(step[pool])]
        if len(pool) >= d8core.MIN_POOL:
            out[j] = step[h] - step[pool].mean()
    return out


def ratio_interval(
    *, num: np.ndarray, den: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float, int]:
    """Return the through-origin ratio `sum(num * den) / sum(den ** 2)`, its 95% interval, and n.

    The interval resamples events. It leaves out the sampling error of the matched pools.
    """
    ok = np.isfinite(num) & np.isfinite(den)
    a, b = num[ok], den[ok]
    n = int(ok.sum())
    if n < d8core.MIN_POOL or (b * b).sum() == 0:
        return float("nan"), float("nan"), float("nan"), n
    est = float((a * b).sum() / (b * b).sum())
    ix = rng.integers(0, n, size=(n_boot, n))
    boots = (a[ix] * b[ix]).sum(1) / np.maximum((b[ix] ** 2).sum(1), 1e-300)
    lo, hi = np.percentile(boots, core.INTERVAL_PERCENTILES)
    return est, float(lo), float(hi), n


def lower_reading_third(*, reading: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Return 0, 1, or 2 for the third of the lower of the two readings around each event.

    The two readings are the means of the 4 half-hours before and the 4 from the event on. The
    thirds are taken across the events handed in, so call the function once per era.
    """
    before = np.array([reading[h - d8core.PRE : h].mean() for h in events])
    after = np.array([reading[h : h + d8core.POST].mean() for h in events])
    lower = np.minimum(before, after)
    ok = np.isfinite(lower)
    out = np.full(len(events), -1)
    if ok.sum() >= 3:
        cuts = np.quantile(lower[ok], [1 / 3, 2 / 3])
        out[ok] = np.digitize(lower[ok], cuts)
    return out


def d8_exploratory(
    *,
    eng: data.EngineFrame,
    d8_reading: np.ndarray,
    residual: np.ndarray,
    type_sums: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Return `dR(-D8) / dm_D8` by era and by the lower reading's third, and each generator type's co-step.

    There is no bar. The step of each series is its 2-hour change at the event less the mean step
    at matched quiet half-hours, and each ratio is over D8's own excess step in the same stratum.
    """
    h, _dk, era, cand_ok = d8core.isolated_events(
        change=eng.change, k_path=eng.k_path, era=eng.era, has_row=eng.has_row, in_drop=eng.in_drop
    )
    cand = d8core.candidates_by_key(key=eng.key, ok=cand_ok)
    series = {"R_leave_out_D8": residual, **type_sums}
    steps = {name: d8core.step_vec(x=x) for name, x in series.items()}
    step_m = d8core.step_vec(x=d8_reading)
    m_ex = excess_steps(step=step_m, events=h, key=eng.key, cand=cand)
    excess = {
        name: excess_steps(step=s, events=h, key=eng.key, cand=cand) for name, s in steps.items()
    }
    rows = []
    for e in ERAS:
        in_era = era == e
        third = np.full(len(h), -1)
        idx = np.flatnonzero(in_era)
        if idx.size:
            third[idx] = lower_reading_third(reading=d8_reading, events=h[idx])
        for t_name, t_val in [("all", None), *zip(TERCILES, range(3), strict=True)]:
            sel = in_era & ((third == t_val) if t_val is not None else True)
            for name, ex in excess.items():
                est, lo, hi, n = ratio_interval(
                    num=ex[sel], den=m_ex[sel], rng=rng, n_boot=M_BOOT_EVENTS
                )
                rows.append(
                    {
                        "era": e,
                        "third": t_name,
                        "series": name,
                        "ratio": est,
                        "lo": lo,
                        "hi": hi,
                        "n_events": n,
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "era": pl.String,
            "third": pl.String,
            "series": pl.String,
            "ratio": pl.Float64,
            "lo": pl.Float64,
            "hi": pl.Float64,
            "n_events": pl.Int64,
        },
    )


def outcome_of_fit(*, fit: core.Fit, name: str, kind: str) -> str:
    """Return the bar outcome of one coefficient of a fit for the named kind of bar."""
    _, se, lo, hi = fit.get(name)
    if kind == "calibration":
        return core.calibration_bar(lo=lo, se=se)
    if kind == "positive":
        return core.positive_control_bar(lo=lo, se=se)
    if kind == "shifted":
        return core.shifted_date_bar(lo=lo, hi=hi, se=se)
    if kind == "test":
        return core.reverse_flow_test(lo=lo, hi=hi, se=se)
    msg = f"unknown bar kind {kind}"
    raise ValueError(msg)


def fit_rows(
    *,
    variant: str,
    span: str,
    fit_name: str,
    fit: core.Fit,
    bars: dict[str, str],
    addback_form: str,
) -> list[dict[str, Any]]:
    """Return one line per regressor of a fit for `u2b_results.csv`."""
    out = []
    for name in fit.names:
        c, se, lo, hi = fit.get(name)
        out.append(
            {
                "variant": variant,
                "addback_form": addback_form,
                "span": span,
                "fit": fit_name,
                "regressor": name,
                "estimate": c,
                "se": se,
                "lo": lo,
                "hi": hi,
                "n_rows": fit.n_rows,
                "n_days_used": fit.n_days,
                "outcome": bars.get(name, "report"),
            }
        )
    return out


# ---------------------------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------------------------
def run(*, gate: guard.TimingGate, inputs: list[Path], info: dict[str, Any]) -> None:  # noqa: PLR0912, PLR0915
    """Run job 3; `inputs` and `info` fill the run log as the job reads."""
    common.assert_environment(env=dict(os.environ))
    checks = common.read_checks()
    common.require_class_audit(checks=checks)
    common.require_stamps(checks=checks)
    u2a = json.loads((guard.OUT_DIR / common.U2A_VERDICT_NAME).read_text())
    if u2a["stops_u2b"]:
        msg = "job 2 returned `not supported`, which stops job 3"
        raise common.JobRefusedError(msg)
    log = guard.ReadLog()
    inputs += [
        data.SITE_MAP_PATH,
        data.METADATA_PATH,
        data.HIERARCHY_PATH,
        data.FRAME39_PATH,
        data.SITES39_PATH,
        data.ENGINE_FRAME_PATH,
    ]
    info["power_table_delta_version"] = data.power_table_version()
    info["u2a_verdict"] = u2a["verdict"]
    topology = common.load_topology()
    classes = common.read_classes()
    primaries_sky, _, _, _, _ = data.load_sky(log=log)
    label_of = dict(
        zip(topology.primaries["time_series_id"], topology.primaries["label"], strict=True)
    )
    sites = {label_of[ts]: s for ts, s in primaries_sky.items() if ts in label_of}
    ids = [
        *topology.primaries["time_series_id"].to_list(),
        *topology.bsp_meter.values(),
        *classes["generator_id"].unique().to_list(),
    ]
    power = data.load_power(ids=ids, log=log, name="primaries_bsp_meters_generators")
    series = common.label_arrays(topology=topology, power=power)
    eng = data.load_engine_frame(log=log)
    n_days = (core.N_GRID + core.HALF_HOURS_PER_DAY - 1) // core.HALF_HOURS_PER_DAY
    day = sky.day_of(n=core.N_GRID)
    month = sky.month_of_day(n_days=n_days)
    quiet = core.quiet_mask(engine_change=eng.change, calendar_ok=eng.has_row)
    near_step = core.within_distance(flag=eng.change, distance=core.ENGINE_STEP_DROP_HALF_HOURS)
    weights = core.day_weights(
        n_days=n_days, n_boot=core.N_BOOT, rng=np.random.default_rng(core.SEED)
    )
    start_a = common.both_start(series=series, transformers=core.TWO_TRANSFORMERS_2025)
    rows_idx = np.arange(core.N_GRID)
    binding_form = checks["binding_addback_form"]
    forms: tuple[core.AddbackForm, ...] = (
        core.ADDBACK_FORMS if binding_form == "unresolved" else ("abs_of_sum",)
    )
    added_back: dict[str, np.ndarray] = {}
    for c in core.BSP_A_UNSIGNED:
        cand = common.addback_candidates(classes=classes, primary=c)
        added_back[c] = np.sum([power[g] for g in cand], axis=0) if cand else np.zeros(core.N_GRID)
    total_fits = 2 * len(common.VARIANTS) * len(forms) * 6

    def dm(x: np.ndarray) -> np.ndarray:
        return core.demean_matched(x=core.change(x=x, lag=1), key=eng.key, quiet=quiet)

    def sky_inputs(label: str) -> dict[str, Any]:
        s = sites[label]
        k = sky.clear_sky_index(ghi=s.ghi, ghi_clear=s.ghi_clear)
        lit = sky.daylight(zenith_deg=s.zenith, min_elevation_deg=core.DAYLIGHT_ELEVATION_DEG)
        cls = sky.classify_days(k=k, lit=lit, n_days=n_days)
        du = sky.fast_irradiance_change(u=s.u, u_clear=s.u_clear, k=k, lit=lit, n_days=n_days)
        return {
            "lit": lit,
            "classes": cls,
            "du": fill_sky_change(du=du, zenith=s.zenith),
            "zenith": s.zenith,
        }

    results: list[dict[str, Any]] = []
    verdicts: dict[str, Any] = {}
    guard.say("FITS STARTING: job 3 (U2b)")
    gate.mark()
    for variant in common.VARIANTS:
        gens = {
            b: [
                power[g]
                for g in common.generator_ids_to_subtract(classes=classes, bsp=b, variant=variant)
            ]
            for b in ("A", "B")
        }
        per_bsp: dict[str, dict[str, Any]] = {}
        for bsp, unsigned, signed, transformers, start in (
            ("A", core.BSP_A_UNSIGNED, core.BSP_A_SIGNED, core.TWO_TRANSFORMERS_2025, start_a),
            ("B", core.BSP_B_UNSIGNED, core.BSP_B_SIGNED, (), 0),
        ):
            ref = SKY_REFERENCE[bsp]
            si = sky_inputs(ref)
            response = common.spanned_residual(
                bsp=power[topology.bsp_meter[bsp]],
                series=series,
                children=signed,
                transformers=transformers,
                leave_out=None,
                generators=gens[bsp],
                start=start,
            )
            y = dm(response)
            x = np.column_stack(
                [dm(series[c]) for c in unsigned]
                + [core.demean_matched(x=si["du"], key=eng.key, quiet=quiet)]
            )
            class_of_row = si["classes"][day]
            cal = calibration_rows(zenith=si["zenith"], class_of_row=class_of_row)
            usable = eng.has_row & ~near_step
            dropped = int((np.isfinite(y) & eng.has_row & near_step).sum())
            per_bsp[bsp] = {
                "y": y,
                "x": x,
                "cal": cal,
                "usable": usable,
                "si": si,
                "dropped": dropped,
                "names": (*unsigned, "sky"),
            }
        # ---- BSP B: the four settled importers (their controls bind on the calibration fit) ----
        b = per_bsp["B"]
        b_fits = {}
        for fit_name, sel in (
            ("calibration", b["cal"]),
            ("all_rows", np.ones(core.N_GRID, dtype=bool)),
        ):
            use = b["usable"] & sel
            b_fits[fit_name] = core.bootstrap_ols(
                y=np.where(use, b["y"], np.nan),
                x=np.where(use[:, None], b["x"], np.nan),
                day=day,
                weights=weights,
                names=b["names"],
            )
        bsp_b_controls = {
            f"bsp_b_{label}": outcome_of_fit(fit=b_fits["calibration"], name=label, kind="positive")
            for label in core.BSP_B_UNSIGNED
        }
        for fit_name, fit in b_fits.items():
            bars = (
                {c: bsp_b_controls[f"bsp_b_{c}"] for c in core.BSP_B_UNSIGNED}
                if fit_name == "calibration"
                else {}
            )
            results += fit_rows(
                variant=variant,
                span="all",
                fit_name=f"bsp_b_{fit_name}",
                fit=fit,
                bars=bars,
                addback_form="not_applicable",
            )
        form_verdicts: dict[str, Any] = {}
        for form in forms:
            form_label = form if binding_form == "unresolved" else binding_form
            x_form = np.column_stack(
                [
                    dm(
                        core.addback_regressor(
                            reading=series[c], added_back=added_back[c], form=form
                        )
                    )
                    for c in core.BSP_A_UNSIGNED
                ]
                + [core.demean_matched(x=per_bsp["A"]["si"]["du"], key=eng.key, quiet=quiet)]
            )
            # ---- BSP A: joint regression, by span ----
            a = per_bsp["A"]
            d12_reading = series["D12"]
            t_one, counts = stretch_rows(
                reading=d12_reading,
                lit=a["si"]["lit"],
                classes=a["si"]["classes"],
                month=month,
                capacity_mw=topology.capacity(label="D12"),
                want="one",
            )
            t_all, _ = stretch_rows(
                reading=d12_reading,
                lit=a["si"]["lit"],
                classes=a["si"]["classes"],
                month=month,
                capacity_mw=topology.capacity(label="D12"),
                want="all",
            )
            t_many, _ = stretch_rows(
                reading=d12_reading,
                lit=a["si"]["lit"],
                classes=a["si"]["classes"],
                month=month,
                capacity_mw=topology.capacity(label="D12"),
                want="many",
            )
            eligible = (a["si"]["classes"] == sky.CLASS_INTERMITTENT) & np.isin(
                month, core.TEST_MONTHS
            )
            shares = day_count_shares(counts=counts, eligible=eligible)
            spans = {"later": rows_idx >= start_a, "earlier": rows_idx < start_a}
            fits: dict[tuple[str, str], core.Fit] = {}
            for span, in_span in spans.items():
                base = a["usable"] & in_span
                for fit_name, sel in (
                    ("calibration", a["cal"]),
                    ("test_one_stretch", t_one),
                    ("test_all_stretches", t_all),
                    ("test_three_or_more", t_many),
                ):
                    use = base & sel
                    fit = core.bootstrap_ols(
                        y=np.where(use, a["y"], np.nan),
                        x=np.where(use[:, None], x_form, np.nan),
                        day=day,
                        weights=weights,
                        names=a["names"],
                    )
                    if not fits:
                        gate.check(done=1, total=total_fits)
                    fits[(span, fit_name)] = fit
                shifted_x = x_form.copy()
                shifted_x[:, 2] = core.shift_rows(
                    x=x_form[:, 2], rows=core.PLACEBO_SHIFT_DAYS * core.HALF_HOURS_PER_DAY
                )
                for fit_name, sel in (("calibration_shifted", a["cal"]), ("test_shifted", t_one)):
                    use = base & sel
                    fits[(span, fit_name)] = core.bootstrap_ols(
                        y=np.where(use, a["y"], np.nan),
                        x=np.where(use[:, None], shifted_x, np.nan),
                        day=day,
                        weights=weights,
                        names=a["names"],
                    )
            cal_bar = outcome_of_fit(
                fit=fits[("later", "calibration")], name="D12", kind="calibration"
            )
            test_bar = outcome_of_fit(
                fit=fits[("later", "test_one_stretch")], name="D12", kind="test"
            )
            controls: dict[str, str] = {
                "d3_calibration": outcome_of_fit(
                    fit=fits[("later", "calibration")], name="D3", kind="positive"
                ),
                "d3_test": outcome_of_fit(
                    fit=fits[("later", "test_one_stretch")], name="D3", kind="positive"
                ),
                "shifted_date": outcome_of_fit(
                    fit=fits[("later", "test_shifted")], name="D12", kind="shifted"
                ),
            }
            controls.update(bsp_b_controls)
            verdict = core.verdict_u2b(
                calibration=cast(core.Outcome, cal_bar),
                test=cast(core.TestOutcome, test_bar),
                controls=cast(dict[str, core.Outcome], controls),
            )
            # ---- rows ----
            for (span, fit_name), fit in fits.items():
                bars: dict[str, str] = {}
                if span == "later":
                    if fit_name == "calibration":
                        bars = {"D12": cal_bar, "D3": controls["d3_calibration"]}
                    elif fit_name == "test_one_stretch":
                        bars = {"D12": test_bar, "D3": controls["d3_test"]}
                    elif fit_name == "test_shifted":
                        bars = {"D12": controls["shifted_date"]}
                results += fit_rows(
                    variant=variant,
                    span=span,
                    fit_name=fit_name,
                    fit=fit,
                    bars=bars,
                    addback_form=form_label,
                )
            cal_fit = fits[("later", "calibration")]
            form_verdicts[form_label] = {
                "verdict": verdict,
                "calibration": cal_bar,
                "test": test_bar,
                "controls": controls,
                "common_mode_inflation_d12": float(cal_fit.get("D12")[0]),
                "detection_limit_d12_test": core.detection_limit(
                    se=fits[("later", "test_one_stretch")].get("D12")[1]
                ),
                "day_shares_by_post_bounce_count": shares,
                "rows_dropped_next_to_engine_steps": {
                    "A": per_bsp["A"]["dropped"],
                    "B": per_bsp["B"]["dropped"],
                },
                "later_span_starts_after_grid_start": start_a > 0,
            }
            guard.say(
                variant,
                form_label,
                "D12 calibration",
                round(float(cal_fit.get("D12")[0]), 3),
                "lower end",
                round(float(cal_fit.get("D12")[2]), 3),
                cal_bar,
                "| D12 test (one stretch)",
                [round(v, 3) for v in fits[("later", "test_one_stretch")].get("D12")],
                test_bar,
                "| controls",
                controls,
                "| day shares",
                {k: round(v, 3) for k, v in shares.items()},
                "| rows dropped next to engine steps",
                form_verdicts[form_label]["rows_dropped_next_to_engine_steps"],
                "| verdict",
                verdict,
            )
        distinct = {v["verdict"] for v in form_verdicts.values()}
        verdicts[variant] = {
            "verdict": distinct.pop() if len(distinct) == 1 else "inconclusive",
            "binding_addback_form": binding_form,
            "form_unresolved": binding_form == "unresolved",
            "forms": form_verdicts,
        }
        if variant == "primary":
            by_type: dict[str, list[np.ndarray]] = {}
            for g in (
                classes.filter(
                    (pl.col("bsp") == "A")
                    & (pl.col("units") == "MW")
                    & ~pl.col("is_engine")
                    & (pl.col("coverage") >= core.MIN_GENERATOR_COVERAGE)
                )
                .unique(subset=["generator_id"])
                .iter_rows(named=True)
            ):
                by_type.setdefault(g["generator_type"], []).append(power[g["generator_id"]])
            r_d8 = common.spanned_residual(
                bsp=power[topology.bsp_meter["A"]],
                series=series,
                children=core.BSP_A_PRIMARIES,
                transformers=core.TWO_TRANSFORMERS_2025,
                leave_out="D8",
                generators=gens["A"],
                start=start_a,
            )
            exploratory = d8_exploratory(
                eng=eng,
                d8_reading=series["D8"],
                residual=r_d8,
                type_sums={f"type:{t}": np.sum(v, axis=0) for t, v in sorted(by_type.items())},
                rng=np.random.default_rng(core.SEED + 2),
            )
            guard.write_public_csv(frame=exploratory, name="u2b_d8_exploratory.csv")
            guard.say("D8 exploratory table rows:", exploratory.height)
    guard.write_public_csv(
        frame=pl.DataFrame(results, infer_schema_length=None), name="u2b_results.csv"
    )
    payload = {
        "verdict": verdicts["primary"]["verdict"],
        "binding_variant": "primary",
        "variants": verdicts,
        "u2a_verdict": u2a["verdict"],
    }
    guard.write_public_json(payload=payload, name="u2b_verdict.json")
    guard.say(
        "LAST FIT ENDED: job 3 (U2b); verdict",
        payload["verdict"],
        "u2a verdict",
        u2a["verdict"],
        "core-seconds",
        round(gate.elapsed(), 1),
    )


def main() -> None:
    """Run job 3 and write its run log even when the job crashes."""
    gate = guard.TimingGate()
    inputs: list[Path] = []
    info: dict[str, Any] = {}
    with guard.run_log(job="u2b", folder=FOLDER, gate=gate, inputs=inputs, info=info):
        run(gate=gate, inputs=inputs, info=info)


if __name__ == "__main__":
    main()

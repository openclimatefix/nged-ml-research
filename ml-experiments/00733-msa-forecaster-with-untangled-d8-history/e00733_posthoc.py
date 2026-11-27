# ruff: noqa: E501
"""Experiment 733: the post hoc readings P1 to P4 of the arms FS, Y0b, YUb, and YOb (no torch).

The maintainer asked for the fits of FS after seeing the verdict. The finding that all six
synthetic fits of the main run stored pass 1 led the manager to add one pre-registered check on the
synthetic world, set before any post hoc fit: Y0b, YUb, and YOb are the forecasters Y0, YU, and the
oracle (the true sign and solar output of the history half-hours before the origin), each fitted
once, at seed 0, and stopped on the sign head's own validation binary cross-entropy. Every
reading here is post hoc and report-only. FS, Y0b, YUb, and YOb are read from the post hoc data
folder (`posthoc/<name of the data folder>`); FU, M*, Y0, and YU are read from the main data
folder, which this module never writes.

- P1: FS minus FU, CRPS of D8 on the real folds. The month-block bootstrap, widened by a t ratio,
  gives its interval, at the level of the counted contrast C1.
- P2: FS minus M*, CRPS of D8 on the real folds, read as P1.
- P3: YOb minus Y0b, Brier score of the sign on the synthetic world, single seed.
- P4: YUb minus Y0b, Brier score of the sign on the synthetic world, single seed.

P3 and P4 are point differences only: one seed gives no interval.
Beside them, the report lists each synthetic arm's stored pass, its validation sign cross-entropy at
that pass, and its Brier score on the scored test half-hours, for Y0b, YUb, and YOb and for seed 0 of Y0
and YU of the main run.

The stage `posthoc_score` writes `posthoc/readings.json`, `posthoc/report.md`, and
`posthoc/posthoc_contrasts.svg`, and nothing else.
"""

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import e00733_baselines as baselines
import e00733_core as core
import e00733_fit as fit
import e00733_scoring as scoring
import e00733_scoring_synth as ss
import e00733_stages as st
import e00733_synth as sy
import e00733_world as world
import numpy as np
import polars as pl

READINGS_FILE: Final = "readings.json"
REPORT_FILE: Final = "report.md"
FIGURE_FILE: Final = "posthoc_contrasts.svg"
REAL_SPECS: Final = {
    "P1": (("FS", None), ("FU", None)),
    "P2": (("FS", None), ("M*", None)),
}
"""The real-fold contrasts: the arm and its reference, each as (arm, variant)."""
SYNTHETIC_SPECS: Final = {"P3": ("YOb", "Y0b"), "P4": ("YUb", "Y0b")}
"""The synthetic contrasts: the arm and its reference, both from the post hoc folder."""
SYNTHETIC_TABLE_ARMS: Final = (
    ("Y0", "main"), ("YU", "main"), ("Y0b", "posthoc"), ("YUb", "posthoc"), ("YOb", "posthoc"),
)  # fmt: skip
"""The synthetic arms of the stored-pass table, each with the data folder that holds it."""
SINGLE_SEED_NOTE: Final = "single seed, post hoc, no interval claim"
PERCENT: Final = 100.0
MAX_REPORT_LINES: Final = 30


def contrast_label(*, name: str) -> str:
    """Return the label of a contrast, such as "P1 post hoc: FS minus FU"."""
    if name in REAL_SPECS:
        (a, _), (b, _) = REAL_SPECS[name]
        return f"{name} post hoc: {a} minus {b}"
    a, b = SYNTHETIC_SPECS[name]
    return f"{name} post hoc: {a} minus {b}"


def unread(*, name: str, reason: str) -> dict[str, Any]:
    """Return the entry of a contrast that could not be read: status `void`, with the reason."""
    return {
        "post_hoc": True,
        "label": contrast_label(name=name),
        "status": "void",
        "reason": reason,
    }


def real_contrasts(
    *, root: Path, fit_root: Path, rows: str, n_resamples: int | None = None
) -> dict[str, dict[str, Any]]:
    """Read P1 and P2: FS (post hoc folder) against FU and M* (main folder), CRPS of D8.

    The plan is every fold at every seed of FS. A validation score (a scratch run) reads the folds
    and seeds that the arms hold, and puts F0r in the place of M*, as the main score does. Both
    contrasts also carry the mean absolute error of the median, as the difference and relative to
    the reference's mean.

    Args:
        root: The data folder of the main run (read only).
        fit_root: The data folder of the post hoc fits (read only here).
        rows: "test" or "validation".
        n_resamples: The bootstrap resamples (default the plan's, or 1,000 on validation rows).

    Returns:
        P1 and P2, each labelled post hoc; a contrast without a complete plan is `void`.
    """
    import e00713_score as s713  # ty: ignore[unresolved-import]

    n_boot = (
        n_resamples
        if n_resamples is not None
        else (core.N_RESAMPLES_REAL if rows == "test" else scoring.N_RESAMPLES_VALIDATION)
    )
    d8 = baselines.d8_series_id()
    truth = s713.truth_stack() if rows == "test" else scoring.validation_truth(root=root)
    root_of = {"FS": fit_root}

    def where(arm: scoring.Arm) -> Path:
        return root_of.get(arm[0], root)

    def exists(*, root: Path, arm: scoring.Arm, fold: str, seed: int, rows: str) -> bool:
        return scoring.has_job(root=where(arm), arm=arm, fold=fold, seed=seed, rows=rows)

    out: dict[str, dict[str, Any]] = {}
    for name, (treat, reference) in REAL_SPECS.items():
        ref = scoring.swap_reference(arm=reference, rows=rows)
        plan = scoring.choose_plan(
            root=root, arms=[treat, ref], wanted_seeds=core.PLACEBO_SEEDS, rows=rows, exists=exists
        )
        if plan is None:
            out[name] = unread(name=name, reason="a stored forecast of the plan is missing")
            continue
        t = scoring.arm_losses(root=where(treat), arm=treat, plan=plan, rows=rows, truth=truth)
        r = scoring.arm_losses(root=where(ref), arm=ref, plan=plan, rows=rows, truth=truth)
        result = scoring.contrast(treatment=t, reference=r, ids=[d8], n_resamples=n_boot)
        if result is None:
            out[name] = unread(name=name, reason="no row of D8 is scored")
            continue
        tr, rr = (scoring.restrict(frame=f, ids=[d8]) for f in (t, r))
        mae = scoring.mean_difference(t=tr, r=rr, column="abs_err")
        mae_ref = float(rr["abs_err"].mean())  # ty: ignore[invalid-argument-type]
        out[name] = {
            "post_hoc": True, "label": contrast_label(name=name), "status": "read",
            "treatment": treat[0], "reference": ref[0], "metric": "crps", "plan": plan.as_dict(),
            "reading": result.as_dict(), "mae_of_median_difference": mae,
            "mae_of_median_relative": mae / mae_ref, "interim": plan.interim or rows != "test",
        }  # fmt: skip
    return out


def _scored_arrays(
    *, root: Path, root_of: Mapping[str, Path], arms: Sequence[str], rows: str
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    """Return the seed 0 archives of the arms that exist, the scored-row mask, and the true sign.

    The world (the bank's segments) is read from `root`, the data folder of the main run.

    Args:
        root: The data folder of the main run.
        root_of: The folder that holds each arm's archives.
        arms: The arms to read.
        rows: The row set to read, `test` or `validation`.

    Returns:
        The seed 0 archives by arm, the scored-row mask, and the true sign.

    Raises:
        sc.ScoreError: When an archive holds another number of samples than the bank, or two arms
            disagree on the true sign.
    """
    present = {
        a: arrays
        for a in arms
        if (arrays := ss.read_arm(
            root=root_of[a], arm=a, seeds=core.POSTHOC_SYNTHETIC_SEEDS, rows=rows
        )) is not None
    }  # fmt: skip
    if not present:
        return {}, np.empty(0), np.empty(0)
    segments, _ = world.read_world(root=root, kind="validation" if rows == "validation" else "test")
    index = sy.enumerate_samples(segments=segments)
    for arrays in present.values():
        if len(arrays["seg"]) != len(index):
            msg = "a stored synthetic archive holds another number of samples than the bank"
            raise ss.sc.ScoreError(msg)
    truth = next(iter(present.values()))["sign_truth"]
    for arrays in present.values():
        if not np.array_equal(arrays["sign_truth"], truth, equal_nan=True):
            msg = "two arms hold different true signs for the same samples"
            raise ss.sc.ScoreError(msg)
    keep = ss.scored_mask(lead_index=index.lead_index)[:, None] & np.isfinite(truth)
    return present, keep, truth


def _brier(*, p: np.ndarray, truth: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Return the Brier score of each scored half-hour (NaN where it is not scored)."""
    return ss.masked(values=(np.asarray(p) - truth) ** 2, keep=keep & np.isfinite(p))


def synthetic_contrasts(*, root: Path, fit_root: Path, rows: str) -> dict[str, dict[str, Any]]:
    """Read P3 and P4: YOb and YUb against Y0b (all in the post hoc folder), sign Brier score.

    The scored half-hours are those of the counted contrast C4: lead days 2 to 10 of the bank that
    hold a sign. Each contrast is the difference of the mean Brier score over the half-hours where
    both arms are scored, at seed 0 alone. It has no interval, because one seed gives no estimate
    of the variance between fits, and the entry says so. A contrast whose arm lacks the archive of
    seed 0 is `void`, and the reason names the arm.

    Args:
        root: The data folder of the main run (read only; it holds the world).
        fit_root: The data folder of the post hoc fits (read only here).
        rows: "test" or "validation".

    Returns:
        P3 and P4, each labelled post hoc.

    Raises:
        sc.ScoreError: When an archive holds another number of samples than the bank, or the arms
            disagree on the true sign.
    """
    names = sorted({a for pair in SYNTHETIC_SPECS.values() for a in pair})
    present, keep, truth = _scored_arrays(
        root=root, root_of=dict.fromkeys(names, fit_root), arms=names, rows=rows
    )
    out: dict[str, dict[str, Any]] = {}
    for name, (first, second) in SYNTHETIC_SPECS.items():
        missing = [a for a in (first, second) if a not in present]
        if missing:
            reason = "; ".join(f"{a} holds no archive of seed 0 on {rows} rows" for a in missing)
            out[name] = unread(name=name, reason=reason)
            continue
        diff = _brier(p=present[first]["p_neg"], truth=truth, keep=keep) - _brier(
            p=present[second]["p_neg"], truth=truth, keep=keep
        )
        ok = np.isfinite(diff)
        out[name] = {
            "post_hoc": True, "label": contrast_label(name=name), "status": "read",
            "treatment": first, "reference": second, "metric": "brier",
            "point": float(diff[ok].mean()) if ok.any() else float("nan"),
            "n_rows": int(ok.sum()), "seeds": list(core.POSTHOC_SYNTHETIC_SEEDS),
            "note": SINGLE_SEED_NOTE, "interim": rows != "test",
        }  # fmt: skip
    return out


def synthetic_arm_table(*, root: Path, fit_root: Path, rows: str) -> dict[str, dict[str, Any]]:
    """Return, for each arm of `SYNTHETIC_TABLE_ARMS`, how many passes were stored and how it scores.

    The entry holds `stop_on` (the loss that picked the stored pass), `stored_pass`, `passes` (the
    passes the fit ran), `val_sign_bce` (the validation sign cross-entropy at the stored pass),
    and `brier` (the mean Brier score of the sign over the scored half-hours at seed 0). An arm
    whose fit record or archive does not exist is left out.

    Args:
        root: The data folder of the main run (read only).
        fit_root: The data folder of the post hoc fits (read only here).
        rows: "test" or "validation".

    Returns:
        The table, by arm name.
    """
    root_of = {a: (fit_root if where == "posthoc" else root) for a, where in SYNTHETIC_TABLE_ARMS}
    present, keep, truth = _scored_arrays(root=root, root_of=root_of, arms=list(root_of), rows=rows)
    out: dict[str, dict[str, Any]] = {}
    for arm, arrays in present.items():
        path = fit.record_path(root=root_of[arm], arm=arm, fold=None, seed=0)
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        history = record["history"]
        stored = int(record["stored_pass"])
        brier = _brier(p=arrays["p_neg"], truth=truth, keep=keep)
        out[arm] = {
            "post_hoc": True, "stop_on": record.get("stop_on", core.STOP_ON_TOTAL),
            "stored_pass": stored, "passes": len(history["val"]),
            "val_sign_bce": float(history["val_bce"][stored - 1]),
            "brier": float(np.nanmean(brier)), "interim": rows != "test",
        }  # fmt: skip
    return out


def posthoc_readings(
    *, root: Path, rows: str = "test", n_resamples_real: int | None = None
) -> dict[str, Any]:
    """Return the four post hoc contrasts and the synthetic arms' stored passes, all labelled post hoc.

    The level (`core.LEVEL`) is that of the two real-fold contrasts' intervals.

    A contrast whose arms lack the required forecasts is `void` with the reason, so a run that
    stopped at the cost cap is read as far as it went. `post_hoc_run` carries the state of the run
    and its skipped and failed jobs.
    """
    fit_root = st.posthoc_fit_root(root=root)
    status_path = root / st.POSTHOC_STATUS
    contrasts = {
        **real_contrasts(root=root, fit_root=fit_root, rows=rows, n_resamples=n_resamples_real),
        **synthetic_contrasts(root=root, fit_root=fit_root, rows=rows),
    }
    return {
        "post_hoc": True, "rows": rows, "level": core.LEVEL, "code_version": core.CODE_VERSION,
        "cost": st.posthoc_cost(root=root),
        "post_hoc_run": (
            {k: st.read_json(path=status_path).get(k) for k in ("state", "reason", "skipped", "failed")}
            if status_path.exists()
            else None
        ),
        "contrasts": {n: contrasts[n] for n in (*REAL_SPECS, *SYNTHETIC_SPECS)},
        "synthetic_arms": synthetic_arm_table(root=root, fit_root=fit_root, rows=rows),
    }  # fmt: skip


def read_line(*, name: str, entry: Mapping[str, Any]) -> str:
    """Return the report sentence of one contrast; every number in it is labelled post hoc."""
    label = f"{name} (post hoc)"
    if entry.get("status") != "read":
        return f"{label} was not read: {entry.get('reason')}."
    if "reading" in entry:
        rel = entry["reading"]["relative"]
        return (
            f"{label}, {entry['treatment']} minus {entry['reference']}, CRPS at D8: "
            f"{PERCENT * rel['point']:+.2f}% of {entry['reference']}'s mean CRPS "
            f"(interval {PERCENT * rel['lower']:+.2f}% to {PERCENT * rel['upper']:+.2f}%, "
            f"{entry['reading']['n_rows']} rows); the mean absolute error of the median differs by "
            f"{PERCENT * entry['mae_of_median_relative']:+.2f}% (post hoc)."
        )
    return (
        f"{label}, {entry['treatment']} minus {entry['reference']}, Brier score of the sign: "
        f"{entry['point']:+.5f} ({SINGLE_SEED_NOTE}; {entry['n_rows']} half-hours)."
    )


def arm_line(*, arm: str, entry: Mapping[str, Any]) -> str:
    """Return the report sentence on one synthetic arm's stored pass, labelled post hoc."""
    rule = (
        "the sign cross-entropy alone"
        if entry["stop_on"] == core.STOP_ON_SIGN_BCE
        else "the total loss"
    )
    return (
        f"{arm} (post hoc table): stopped on {rule}; stored pass {entry['stored_pass']} of "
        f"{entry['passes']} passes; validation sign cross-entropy {entry['val_sign_bce']:.4f}; "
        f"Brier score of the sign {entry['brier']:.5f} (single seed, post hoc)."
    )


def cost_line(*, cost: Mapping[str, Any] | None) -> str:
    """Return the sentence on the measured cost of the post hoc fits (GPU-process-hours, post hoc)."""
    if not cost or not cost.get("per_arm"):
        return "The cost of the post hoc fits (post hoc) was not measured: no job record exists."
    arms = ", ".join(f"{a} {h:.2f}" for a, h in cost["per_arm"].items())
    return (
        f"The post hoc fits cost {cost['posthoc']:.2f} GPU-process-hours (post hoc; {arms}). With the "
        f"main chain's {cost['main_chain']:.2f} the running total is {cost['running_total']:.2f}, "
        f"against the stop line of {core.STOP_LINE_HOURS:g}."
    )


def render_report(*, readings: Mapping[str, Any], verdict_row: int | None) -> str:
    """Return the text of `report.md`, at most `MAX_REPORT_LINES` lines of plain prose."""
    row = "its recorded row" if verdict_row is None else f"row {verdict_row}"
    interim = readings.get("rows") != "test"
    tag = " Every number is interim." if interim else ""
    lines = [
        "# Post hoc readings of arms FS, Y0b, YUb, and YOb, experiment 733",
        "",
        "The maintainer asked for these runs after seeing the verdict, so every number below is post hoc.",
        "The timing gate had dropped FS (the forecaster given P(reverse flow) alone) on cost.",
        "All six synthetic fits of the main run stored pass 1, so before any post hoc fit the manager set one check.",
        "Y0b, YUb, and YOb (post hoc) are Y0, YU, and the oracle (the true sign and solar output of the history),",
        "each fitted once, at seed 0, and stopped on the sign head's own validation cross-entropy (post hoc).",
        f"The verdict stays {row} whatever these readings show.",
        "A gain here is a lead for a follow-up, not a finding." + tag,
        "",
    ]
    lines += [read_line(name=n, entry=e) for n, e in readings["contrasts"].items()]
    table = readings.get("synthetic_arms") or {}
    if table:
        lines += [""] + [arm_line(arm=a, entry=e) for a, e in table.items()]
    lines += ["", cost_line(cost=readings.get("cost"))]
    run = readings.get("post_hoc_run") or {}
    if run.get("state") in ("stopped", "failed") or run.get("failed"):
        lines += [
            (
                f"The post hoc run ended {run.get('state')} ({run.get('reason')}): "
                f"{len(run.get('skipped') or [])} fits were skipped and {len(run.get('failed') or [])} failed, "
                "so each contrast above uses only the forecasts that exist, and a void contrast names what is missing (post hoc)."
            )
        ]
    lines += [
        "",
        (
            "Each interval (post hoc) of P1 and P2 uses the month-block bootstrap with a t widening, at "
            f"the level {PERCENT * readings['level']:.2f}%. P3 and P4 have no interval: they are single-seed "
            "point differences (post hoc)."
        ),
    ]
    if len(lines) > MAX_REPORT_LINES:
        msg = f"the report holds {len(lines)} lines, above the limit of {MAX_REPORT_LINES}"
        raise ValueError(msg)
    return "\n".join(lines) + "\n"


def figure_rows(*, readings: Mapping[str, Any]) -> pl.DataFrame:
    """Return the plotted rows: panel, label, point, lower, upper (real in percent of the reference).

    The synthetic rows have no interval, so their `lower` and `upper` are null.
    """
    rows = []
    for entry in readings["contrasts"].values():
        if entry.get("status") != "read":
            continue
        if "reading" in entry:
            rel = entry["reading"]["relative"]
            vals = [PERCENT * rel[k] for k in ("point", "lower", "upper")]
            panel = "real"
        else:
            vals = [entry["point"], None, None]
            panel = "synthetic"
        rows.append(
            {
                "panel": panel, "label": entry["label"], "arm": entry["treatment"],
                "point": vals[0], "lower": vals[1], "upper": vals[2],
            }
        )  # fmt: skip
    return pl.DataFrame(
        rows,
        schema={
            "panel": pl.Utf8,
            "label": pl.Utf8,
            "arm": pl.Utf8,
            "point": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
        },
    )


def contrast_figure(*, readings: Mapping[str, Any]) -> Any:
    """Return the figure of the four contrasts; every label says post hoc.

    The real contrasts carry intervals; the single-seed synthetic ones are dots without an interval.

    Args:
        readings: The post hoc readings P1 to P4.

    Returns:
        The Altair chart.

    Raises:
        ValueError: When no contrast has a reading to draw.
    """
    import altair as alt
    import e00729_figures as figures729

    data = figure_rows(readings=readings)
    if data.is_empty():
        msg = "no post hoc contrast has a reading to draw"
        raise ValueError(msg)
    level = f"{PERCENT * readings['level']:.2f}%"
    colours = {"FS": figures729.GREEN, "YUb": figures729.ORANGE, "YOb": figures729.PURPLE}
    spec = {
        "real": "Difference in CRPS at D8, post hoc (% of the reference arm's CRPS; negative: FS better)",
        "synthetic": "Difference in sign Brier score, post hoc, single seed (negative: first arm better)",
    }
    charts = []
    for panel, x_title in spec.items():
        part = data.filter(pl.col("panel") == panel)
        if part.is_empty():
            continue
        order = part["label"].to_list()
        arms = list(dict.fromkeys(part["arm"].to_list()))
        colour = alt.Color(
            "arm:N", legend=None,
            scale=alt.Scale(domain=arms, range=[colours[a] for a in arms]),
        )  # fmt: skip
        base = alt.Chart(part).encode(y=alt.Y("label:N", sort=order, title=None))  # ty: ignore[unresolved-attribute]
        dots = base.mark_circle(size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("point:Q", title=x_title), color=colour
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#555555")
            .encode(x="x:Q")
        )
        layers = dots + zero
        if panel == "real":
            bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
                x="lower:Q", x2="upper:Q", color=colour
            )
            layers = bars + layers
        charts.append(layers.properties(width=figures729.WIDTH_PX - 170, height=30 * len(order)))
    title = [
        "Post hoc contrasts P1 to P4 of FS, Y0b, YUb, and YOb, set after the verdict; report only.",
        f"Dot: estimate. Line: {level} interval (real folds only; single-seed synthetic: no interval).",
        "Dashed rule: no difference.",
    ]
    return alt.vconcat(*charts).properties(title=alt.TitleParams(title, anchor="start"))


def write_outputs(*, root: Path, readings: Mapping[str, Any]) -> dict[str, Path]:
    """Write `readings.json`, `report.md`, and the figure under `<root>/posthoc/` and nothing else."""
    import e00729_figures as figures729

    folder = st.posthoc_dir(root=root)
    folder.mkdir(parents=True, exist_ok=True)
    verdict_path = root / "scores" / "verdict.json"
    row = json.loads(verdict_path.read_text()).get("row") if verdict_path.exists() else None
    written = {
        "readings": folder / READINGS_FILE, "report": folder / REPORT_FILE,
        "figure": folder / FIGURE_FILE,
    }  # fmt: skip
    tmp = written["readings"].with_name(READINGS_FILE + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(readings, indent=1, default=float))
    tmp.rename(written["readings"])
    written["report"].write_text(render_report(readings=readings, verdict_row=row))
    figures729.save_svg(chart=contrast_figure(readings=readings), path=written["figure"])
    return written


def stage_posthoc_score(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Run the `posthoc_score` stage: read P1 to P4 and write the three files under `posthoc/`.

    The stage runs after a post hoc run that ended complete, failed, or stopped (at the cost cap or
    by Ctrl-C). It reads what exists: a contrast whose arms lack all required seeds on the test
    rows is `void` with a reason, and the others are computed. The report says which arms or seeds
    are missing.

    Args:
        args: The parsed arguments (`--validation-only` on a scratch root reads validation rows).
        root: The data folder of the main run (read only).

    Returns:
        The readings.

    Raises:
        RuntimeError: When `--validation-only` is given without a scratch data root.
    """
    validation = bool(getattr(args, "validation_only", False))
    if validation and getattr(args, "data_root", None) is None:
        msg = "--validation-only is for a scratch data root"
        raise RuntimeError(msg)
    readings = posthoc_readings(root=root, rows="validation" if validation else "test")
    write_outputs(root=root, readings=readings)
    for name, entry in readings["contrasts"].items():
        print(read_line(name=name, entry=entry))
    for arm, entry in readings["synthetic_arms"].items():
        print(arm_line(arm=arm, entry=entry))
    return readings

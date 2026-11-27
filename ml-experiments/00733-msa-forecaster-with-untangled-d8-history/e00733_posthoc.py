# ruff: noqa: E501
"""Experiment 733: the post hoc readings P1 to P6 of the arms FS, YS, and YO (no torch).

The maintainer asked for the fits of FS and YS after seeing the verdict, and the manager added the
oracle YO, so every reading here is post hoc and report-only. FS, YS, and YO are read from the post
hoc data folder
(`posthoc/<name of the data folder>`); FU, M*, Y0, and YU are read from the main data folder, which
this module never writes. The four contrasts use the machinery of the counted contrasts C1 and C4:
the month-block bootstrap widened by a t ratio on the real folds, the two-way bootstrap on the
synthetic world, and the same level (`core.LEVEL`).

- P1: FS minus FU, CRPS of D8 on the real folds.
- P2: FS minus M*, CRPS of D8 on the real folds.
- P3: YS minus Y0, Brier score of the sign on the synthetic world.
- P4: YS minus YU, Brier score of the sign on the synthetic world.
- P5: YO minus Y0, Brier score of the sign on the synthetic world.
- P6: YO minus YU, Brier score of the sign on the synthetic world.

The stage `posthoc_score` writes `posthoc/readings.json`, `posthoc/report.md`, and
`posthoc/posthoc_contrasts.svg`, and nothing else.
"""

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import e00733_baselines as baselines
import e00733_core as core
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
SYNTHETIC_SPECS: Final = {
    "P3": ("YS", "Y0"), "P4": ("YS", "YU"), "P5": ("YO", "Y0"), "P6": ("YO", "YU"),
}  # fmt: skip
"""The synthetic contrasts: the arm and its reference."""
PERCENT: Final = 100.0
MAX_REPORT_LINES: Final = 25


def contrast_label(*, name: str) -> str:
    """Return the label of a contrast, such as "P1 post hoc: FS minus FU"."""
    if name in REAL_SPECS:
        (a, _), (b, _) = REAL_SPECS[name]
        return f"{name} post hoc: {a} minus {b}"
    a, b = SYNTHETIC_SPECS[name]
    return f"{name} post hoc: {a} minus {b}"


def unread(*, name: str, reason: str) -> dict[str, Any]:
    """Return the entry of a contrast that could not be read."""
    return {
        "post_hoc": True,
        "label": contrast_label(name=name),
        "status": "unread",
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
        P1 and P2, each labelled post hoc; a contrast without a complete plan is `unread`.
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


def synthetic_contrasts(
    *, root: Path, fit_root: Path, rows: str, n_resamples: int | None = None
) -> dict[str, dict[str, Any]]:
    """Read P3 to P6: YS and YO (post hoc folder) against Y0 and YU (main folder), sign Brier score.

    The scored half-hours are those of the counted contrast C4: lead days 2 to 10 of the test bank
    that hold a sign. A contrast whose arm has a seed missing is `unread` with the reason.

    Args:
        root: The data folder of the main run (read only).
        fit_root: The data folder of the post hoc fits (read only here).
        rows: "test" or "validation".
        n_resamples: The bootstrap resamples (default the plan's, or 1,000 on validation rows).

    Returns:
        P3 to P6, each labelled post hoc.

    Raises:
        sc.ScoreError: When an archive holds another number of samples than the bank, or the arms
            disagree on the true sign.
    """
    n_boot = (
        n_resamples
        if n_resamples is not None
        else (core.N_RESAMPLES_SYNTHETIC if rows == "test" else ss.N_RESAMPLES_VALIDATION)
    )
    roots = {"YS": fit_root, "YO": fit_root, "Y0": root, "YU": root}
    held = {a: ss.seeds_held(root=r, arm=a, rows=rows) for a, r in roots.items()}
    arms = {
        a: ss.read_arm(root=roots[a], arm=a, seeds=held[a], rows=rows) if held[a] else None
        for a in roots
    }
    readable = {
        n: pair for n, pair in SYNTHETIC_SPECS.items() if all(arms[a] is not None for a in pair)
    }
    out: dict[str, dict[str, Any]] = {
        n: unread(
            name=n,
            reason=f"the archives of {', '.join(a for a in pair if arms[a] is None)} are missing",
        )
        for n, pair in SYNTHETIC_SPECS.items()
        if n not in readable
    }
    if not readable:
        return out
    segments, _ = world.read_world(root=root, kind="validation" if rows == "validation" else "test")
    index = sy.enumerate_samples(segments=segments)
    present = {a: v for a, v in arms.items() if v is not None}
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
    srows = ss.sc.synthetic_rows(
        seg=index.seg, origin_day=index.origin_day, seg_month=ss.segment_months(segments=segments)
    )

    def brier(p: np.ndarray) -> np.ndarray:
        return ss.masked(values=(np.asarray(p) - truth) ** 2, keep=keep & np.isfinite(p))

    for name, (first, second) in readable.items():
        result = ss.sc.synthetic_contrast(
            rows=srows, first=brier(present[first]["p_neg"]), second=brier(present[second]["p_neg"]),
            n_resamples=n_boot,
        )  # fmt: skip
        out[name] = {
            "post_hoc": True, "label": contrast_label(name=name), "status": "read",
            "treatment": first, "reference": second, "metric": "brier",
            "point": result.reading.point, "lower": result.reading.lower,
            "upper": result.reading.upper, "kish_blocks": result.kish_blocks,
            "kish_segments": result.kish_segments, "widening": result.widening,
            "n_rows": result.n_rows, "n_resamples": result.n_resamples,
            "seeds": {a_: list(held[a_]) for a_ in (first, second)}, "interim": rows != "test",
        }  # fmt: skip
    return {n: out[n] for n in SYNTHETIC_SPECS}


def posthoc_readings(
    *, root: Path, rows: str = "test", n_resamples_real: int | None = None,
    n_resamples_synthetic: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Return the four post hoc contrasts, each labelled post hoc, with the level they share."""
    fit_root = st.posthoc_fit_root(root=root)
    contrasts = {
        **real_contrasts(root=root, fit_root=fit_root, rows=rows, n_resamples=n_resamples_real),
        **synthetic_contrasts(
            root=root, fit_root=fit_root, rows=rows, n_resamples=n_resamples_synthetic
        ),
    }
    return {
        "post_hoc": True, "rows": rows, "level": core.LEVEL, "code_version": core.CODE_VERSION,
        "cost": st.posthoc_cost(root=root),
        "contrasts": {n: contrasts[n] for n in (*REAL_SPECS, *SYNTHETIC_SPECS)},
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
        f"{entry['point']:+.5f} (interval {entry['lower']:+.5f} to {entry['upper']:+.5f}, "
        f"{entry['n_rows']} half-hours)."
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
        "# Post hoc readings of arms FS, YS, and YO, experiment 733",
        "",
        "The maintainer asked for these runs after seeing the verdict, so every number below is post hoc.",
        "The timing gate had dropped FS (the forecaster given P(reverse flow) alone) and YT on cost.",
        "FS and two new synthetic arms were fitted afterwards: YS (P(reverse flow) alone) and YO (the",
        "true sign and solar output of the history, as an oracle). None is part of the verdict.",
        f"The verdict stays {row} whatever these readings show.",
        "A gain here is a lead for a follow-up, not a finding." + tag,
        "",
    ]
    lines += [read_line(name=n, entry=e) for n, e in readings["contrasts"].items()]
    lines += ["", cost_line(cost=readings.get("cost"))]
    lines += [
        "",
        (
            "Each interval (post hoc) uses the method of the counted contrast of the same world: the "
            "month-block bootstrap with a t widening on the real folds, and the two-way bootstrap on "
            f"the synthetic world, at the level {PERCENT * readings['level']:.2f}%."
        ),
    ]
    if len(lines) > MAX_REPORT_LINES:
        msg = f"the report holds {len(lines)} lines, above the limit of {MAX_REPORT_LINES}"
        raise ValueError(msg)
    return "\n".join(lines) + "\n"


def figure_rows(*, readings: Mapping[str, Any]) -> pl.DataFrame:
    """Return the plotted rows: panel, label, point, lower, upper (real in percent of the reference)."""
    rows = []
    for entry in readings["contrasts"].values():
        if entry.get("status") != "read":
            continue
        if "reading" in entry:
            rel = entry["reading"]["relative"]
            vals = [PERCENT * rel[k] for k in ("point", "lower", "upper")]
            panel = "real"
        else:
            vals = [entry[k] for k in ("point", "lower", "upper")]
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
    """Return the figure of the four contrasts with their intervals; every label says post hoc.

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
    colours = {"FS": figures729.GREEN, "YS": figures729.ORANGE, "YO": figures729.PURPLE}
    spec = {
        "real": "Difference in CRPS at D8, post hoc (% of the reference arm's CRPS; negative: FS better)",
        "synthetic": "Difference in sign Brier score, post hoc (negative: first arm better)",
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
        bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lower:Q", title=x_title), x2="upper:Q", color=colour
        )
        dots = base.mark_circle(size=70).encode(x="point:Q", color=colour)  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#555555")
            .encode(x="x:Q")
        )
        charts.append(
            (bars + dots + zero).properties(width=figures729.WIDTH_PX - 170, height=30 * len(order))
        )
    title = [
        "Post hoc contrasts P1 to P6 of FS, YS, and YO, set after the verdict; report only.",
        f"Dot: estimate. Line: {level} interval. Dashed rule: no difference.",
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
    return readings

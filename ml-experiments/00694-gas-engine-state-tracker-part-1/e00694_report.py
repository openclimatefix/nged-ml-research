# ruff: noqa: E501
r"""Score the stored runs, print every number the write-up quotes, and draw the result figures.

    OMP_NUM_THREADS=1 PYTHONPATH=src:ml-experiments/00694-gas-engine-state-tracker-part-1 \
        uv run --no-sync python ml-experiments/00694-*/e00694_report.py

Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.
Gas A is drawn on a day axis with no calendar dates. D8 is a substation.
"""

import os
import pickle
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00694_common import DATA_DIR, DOCS_DIR, f1_of

ARMS: Final[tuple[str, ...]] = ("M0", "M1", "M2", "M2-dwell")
PRIMARY: Final[tuple[str, ...]] = ("record", "synth0", "synth1", "synth2", "synth3")
RULES: Final[tuple[str, ...]] = ("folded", "signed")
N_BOOT: Final[int] = 10_000
ALPHA_BONF: Final[float] = 0.05 / 3
WIDTH: Final[int] = 560
LINES: list[str] = []
REPORT_PATH: Final[Path] = Path(__file__).resolve().parent / "printed_report.txt"
ARM_COLOURS: Final[list[str]] = [ocf.DATA_PURPLE, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]


def say(text: str = "") -> None:
    """Print a line and keep it for the printed report."""
    print(text)
    LINES.append(text)


def load(name: str) -> Any:
    """Load one stored stage output."""
    return pickle.loads((DATA_DIR / f"{name}.pkl").read_bytes())


def wrap(text: str, width: int = 95) -> list[str]:
    """Wrap a title or subtitle to the plot width."""
    return textwrap.wrap(text, width)


def save(*, chart: alt.Chart | alt.VConcatChart, name: str) -> None:
    """Write an SVG beside the write-up (and a PNG when PNG_DIR is set)."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    if png_dir := os.environ.get("PNG_DIR"):
        chart.save(Path(png_dir) / f"{name}.png", scale_factor=1.0)


# ------------------------------------------------------------------ aggregation


def tensor(*, runs: list[dict], rule: str, scheds: tuple[str, ...]) -> tuple[np.ndarray, list]:
    """Counts (tp, fp, fn) as an array (cluster, segment, arm, 3); clusters are schedule-weeks."""
    segs = sorted({(r["host"], r["period"], r["index"]) for r in runs})
    lookup = {(r["sched"], r["host"], r["period"], r["index"], r["rule"]): r for r in runs}
    x = np.zeros((len(scheds) * 7, len(segs), len(ARMS), 3))
    for si, sched in enumerate(scheds):
        for gi, (host, period, index) in enumerate(segs):
            r = lookup[(sched, host, period, index, rule)]
            for ai, arm in enumerate(ARMS):
                x[si * 7 : si * 7 + 7, gi, ai, :] = r["arms"][arm]["counts"][:, :3]
    return x, segs


def f1_arr(*, s: np.ndarray) -> np.ndarray:
    """F1 from summed (tp, fp, fn) along the last axis."""
    tp, fp, fn = s[..., 0], s[..., 1], s[..., 2]
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)


def cluster_bootstrap(*, x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Resample clusters, then segments within each drawn cluster; return summed counts (B, A, 3)."""
    rng = np.random.default_rng(seed)
    n_c, n_s = x.shape[:2]
    out = np.empty((N_BOOT, len(ARMS), 3))
    chunk = 250
    for lo in range(0, N_BOOT, chunk):
        ci = rng.integers(n_c, size=(chunk, n_c))
        si = rng.integers(n_s, size=(chunk, n_c, n_s))
        out[lo : lo + chunk] = x[ci[:, :, None], si].sum(axis=(1, 2))
    return out


def interval(*, d: np.ndarray, level: float) -> tuple[float, float]:
    """Percentile interval at a two-sided level (e.g. 0.95)."""
    return (
        float(np.percentile(d, 100 * (1 - level) / 2)),
        float(np.percentile(d, 100 * (1 + level) / 2)),
    )


def pooled_metrics(*, x: np.ndarray) -> dict[str, np.ndarray]:
    """Pooled F1, precision, and recall per arm from a counts tensor."""
    s = x.sum(axis=(0, 1))
    tp, fp, fn = s[:, 0], s[:, 1], s[:, 2]
    return {
        "f1": f1_arr(s=s),
        "precision": tp / np.maximum(tp + fp, 1),
        "recall": tp / np.maximum(tp + fn, 1),
    }


def unit_recall(*, runs: list[dict], rule: str, scheds: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Recall by the number of units truly running (1, 2, 3), pooled, per arm."""
    out = {}
    for arm in ARMS:
        hit, tot = np.zeros(4), np.zeros(4)
        for r in runs:
            if r["rule"] == rule and r["sched"] in scheds:
                hit += r["arms"][arm]["unit_hit"]
                tot += r["arms"][arm]["unit_tot"]
        out[arm] = hit[1:] / np.maximum(tot[1:], 1)
    return out


def switch_stats(*, runs: list[dict], rule: str, scheds: tuple[str, ...]) -> dict[str, tuple]:
    """Median switch-time error (half-hours) and the share of true switches matched, per arm."""
    out = {}
    for arm in ARMS:
        errs, n_sw = [], 0
        for r in runs:
            if r["rule"] == rule and r["sched"] in scheds:
                errs += r["arms"][arm]["sw_err"]
                n_sw += r["arms"][arm]["n_sw"]
        out[arm] = (float(np.median(errs)) if errs else float("nan"), len(errs) / max(n_sw, 1))
    return out


# ------------------------------------------------------------------ checks and verdict


def p2_holds(*, m: dict, rec: np.ndarray, sw: tuple, arm_i: int) -> tuple[bool, bool]:
    """Return (all P2 thresholds hold, the core ones hold) for one arm and rule.

    The core thresholds are recall at 2 and 3 units and precision.
    """
    core = bool(m["precision"][arm_i] >= 0.80 and rec[1] >= 0.80 and rec[2] >= 0.80)
    rest = bool(m["f1"][arm_i] >= 0.80 and rec[0] >= 0.50 and sw[0] <= 1.0)
    return core and rest, core


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print the results and draw the figures."""
    fa, inj = load("fa"), load("inject")
    k1, d8, p5 = load("k1"), load("d8"), load("p5")
    example = load("example")

    # ---- false alarms on un-injected segments (stop rule 2, P3)
    say("== False alarms on un-injected segments (120 tracked days each) ==")
    hosts = sorted({r["host"] for r in fa})
    fa_rows: list[dict[str, Any]] = []
    for host in hosts:
        for arm in ARMS:
            sel = [r["arms"][arm] for r in fa if r["host"] == host]
            flagged, n = sum(a["flagged"] for a in sel), sum(a["n"] for a in sel)
            act, days = sum(a["active"] for a in sel), sum(a["days"] for a in sel)
            ks = [a["k"] for a in sel]
            fa_rows.append(
                {
                    "host": host,
                    "arm": arm,
                    "share": flagged / n,
                    "active_per_month": 30 * act / max(days, 1),
                    "k_selected": str(ks) if arm != "M0" else "",
                }
            )
            for period in ("pre-solar", "solar"):
                sub = [r["arms"][arm] for r in fa if r["host"] == host and r["period"] == period]
                if sub:
                    fa_rows[-1].setdefault("by_period", {})[period] = sum(
                        a["flagged"] for a in sub
                    ) / sum(a["n"] for a in sub)
    for row in fa_rows:
        say(
            f"{row['host']:>4} {row['arm']:<9} flagged {100 * row['share']:5.2f}%  "
            f"active days/month {row['active_per_month']:.2f}  K per segment {row['k_selected']}  "
            f"by period {row.get('by_period')}"
        )
    m2_fail = [r["host"] for r in fa_rows if r["arm"] == "M2" and r["share"] > 0.03]
    stop2 = len(m2_fail) >= 3
    say(
        f"Stop rule 2 (M2 flags over 3% on 3 or more of {len(hosts)} hosts): {stop2}; M2 over 3%: {m2_fail}"
    )
    p3_fail = [
        r["host"]
        for r in fa_rows
        if r["arm"] == "M2" and (r["share"] > 0.03 or r["active_per_month"] > 1.0)
    ]
    say(f"P3 entries failing for M2: {p3_fail}")
    m0_p3 = not any(
        r["share"] > 0.03 or r["active_per_month"] > 1.0 for r in fa_rows if r["arm"] == "M0"
    )

    # ---- known answers, primary schedules
    verdict_bits: dict[str, Any] = {}
    boots = {}
    for rule in RULES:
        x, segs = tensor(runs=inj, rule=rule, scheds=PRIMARY)
        b = cluster_bootstrap(x=x, seed=1 if rule == "folded" else 2)
        boots[rule] = (x, b, segs)
        m = pooled_metrics(x=x)
        rec = unit_recall(runs=inj, rule=rule, scheds=PRIMARY)
        sw = switch_stats(runs=inj, rule=rule, scheds=PRIMARY)
        say(
            f"\n== Known answers, {rule} rule, primary schedules (record + 4 synthetic), {x.shape[1]} segments =="
        )
        for ai, arm in enumerate(ARMS):
            lo, hi = interval(d=f1_arr(s=b[:, ai, :]), level=0.95)
            say(
                f"{arm:<9} F1 {m['f1'][ai]:.3f} [{lo:.3f}, {hi:.3f}]  precision {m['precision'][ai]:.3f}  "
                f"recall {m['recall'][ai]:.3f}  recall at 1/2/3 units "
                f"{rec[arm][0]:.2f}/{rec[arm][1]:.2f}/{rec[arm][2]:.2f}  "
                f"switch error median {sw[arm][0]:.1f} half-hours, matched {sw[arm][1]:.2f}"
            )
        # contrasts
        f1b = f1_arr(s=b)
        for name, i, j in (("C1 M2-M0", 2, 0), ("C2 M2-M1", 2, 1), ("C3 M2dwell-M2", 3, 2)):
            d = f1b[:, i] - f1b[:, j]
            lo, hi = interval(d=d, level=1 - ALPHA_BONF)
            pt = float(m["f1"][i] - m["f1"][j])
            say(f"{name}: {pt:+.3f}  98.3% interval [{lo:+.3f}, {hi:+.3f}]")
            verdict_bits[f"{rule}_{name[:2]}"] = (pt, lo, hi)
        ok, core = p2_holds(m=m, rec=rec["M2"], sw=sw["M2"], arm_i=2)
        ok0, _ = p2_holds(m=m, rec=rec["M0"], sw=sw["M0"], arm_i=0)
        verdict_bits[f"{rule}_p2"] = (ok, core)
        verdict_bits[f"{rule}_m0p2"] = ok0
        say(f"P2 for M2 under {rule}: all {ok}, core (precision, recall at 2 and 3 units) {core}")
        # leave one host out (audit trigger 4)
        for host in sorted({s[0] for s in segs}):
            keep = [i for i, s in enumerate(segs) if s[0] != host]
            mm = pooled_metrics(x=x[:, keep])
            say(
                f"  without {host}: C1 {mm['f1'][2] - mm['f1'][0]:+.3f}  C2 {mm['f1'][2] - mm['f1'][1]:+.3f}"
            )
        # all schedules, including the reported-only ones
        x_all, _ = tensor(runs=inj, rule=rule, scheds=tuple(sorted({r["sched"] for r in inj})))
        ma = pooled_metrics(x=x_all)
        say(f"  all 9 schedules, pooled F1 by arm: {np.round(ma['f1'], 3).tolist()}")

    # ---- every schedule on its own (the reported-only ones do not decide)
    for rule in RULES:
        say(f"\nPooled F1 by schedule, {rule} rule (M0, M1, M2, M2-dwell):")
        for sched in sorted({r["sched"] for r in inj}):
            xs, _ = tensor(runs=inj, rule=rule, scheds=(sched,))
            ms = pooled_metrics(x=xs)
            recs = unit_recall(runs=inj, rule=rule, scheds=(sched,))
            say(
                f"  {sched:<7} F1 {np.round(ms['f1'], 3).tolist()}  precision (M2) {ms['precision'][2]:.2f}"
                f"  recall at 1/2/3 units (M2) {np.round(recs['M2'], 2).tolist()}"
            )

    # ---- real record only (cluster = week of the one schedule)
    for rule in RULES:
        x, _ = tensor(runs=inj, rule=rule, scheds=("record",))
        m = pooled_metrics(x=x)
        say(f"Record only, {rule}: F1 by arm {np.round(m['f1'], 3).tolist()}")

    # ---- recovery of K and u
    say("\n== Fitted K and u by schedule (M2, folded rule; M1 in brackets) ==")
    for sched in sorted({r["sched"] for r in inj}):
        sel = [r for r in inj if r["sched"] == sched and r["rule"] == "folded"]
        ks = np.array([r["arms"]["M2"]["k"] for r in sel])
        us = np.array([r["arms"]["M2"]["u"] for r in sel])
        k1s = np.array([r["arms"]["M1"]["k"] for r in sel])
        u1s = np.array([r["arms"]["M1"]["u"] for r in sel])
        say(
            f"{sched}: M2 K counts {np.bincount(ks, minlength=4).tolist()} median u {np.median(us[ks > 0]) if (ks > 0).any() else float('nan'):.2f}"
            f"  (M1 K counts {np.bincount(k1s, minlength=4).tolist()} median u {np.median(u1s[k1s > 0]) if (k1s > 0).any() else float('nan'):.2f})"
        )

    # ---- K1
    say("\n== K1: D8's own reading with the record subtracted (does not decide) ==")
    for r in k1:
        for arm, v in r["arms"].items():
            f = f1_of(tp=v["tp"], fp=v["fp"], fn=v["fn"])
            say(
                f"{r['rule']:<7} {arm:<9} F1 {f:.3f} tp {v['tp']} fp {v['fp']} fn {v['fn']} K {v['k']} u {v['u']}"
            )

    # ---- real D8
    say("\n== Real D8, fitted without Gas A ==")
    eras = {e["label"]: e for e in d8}
    for label, e in eras.items():
        say(
            f"{label}: M2 K {e['M2']['k']} u {e['M2']['u']:.2f}, share flagged {e['M2']['flag'].mean():.3f}; "
            f"M1 K {e['M1']['k']} u {e['M1']['u']:.2f}, share {e['M1']['flag'].mean():.3f}; "
            f"M2 without the fold K {e['M2s']['k']} u {e['M2s']['u']:.2f}, share {e['M2s']['flag'].mean():.3f}"
        )
        say(
            f"   M2 transition matrix {np.round(e['M2'].get('transition', np.zeros((1, 1))), 3).tolist()}"
        )
    p1 = eras["net"]["M2"]["k"] == 3 and 1.7 <= eras["net"]["M2"]["u"] <= 2.3
    drift = abs(eras["net"]["M2"]["u"] - eras["solar"]["M2"]["u"])
    say(f"P1 (K = 3 and u in [1.7, 2.3] on the net era): {p1}; u drift net to solar {drift:.2f} MW")

    # ---- P5
    say("\n== P5: held-out predictive log-likelihood on even months of the net era ==")
    months = sorted(p5["fold"]["per_month"])
    diffs = []
    for m_ in months:
        lf, n = p5["fold"]["per_month"][m_]
        ls, _ = p5["signed"]["per_month"][m_]
        diffs.append(lf - ls)
        say(f"{m_}: n {n}, fold {lf:.1f}, signed {ls:.1f}, difference {lf - ls:+.1f}")
    wins = int(np.sum(np.array(diffs) > 0))
    say(
        f"fold fits K {p5['fold']['k']} u {p5['fold']['u']:.2f}; signed fits K {p5['signed']['k']} "
        f"u {p5['signed']['u']:.2f}; total difference {sum(diffs):+.1f}; fold better in {wins} of {len(diffs)} months"
    )
    p5_ok = sum(diffs) > 0 and wins >= 2 * len(diffs) / 3
    say(f"P5 (total above zero and fold better in at least two thirds of months): {p5_ok}")

    # ---- verdict
    c1 = [verdict_bits[f"{r}_C1"] for r in RULES]
    c1_clears = all(pt >= 0.10 and lo > 0 for pt, lo, _ in c1)
    p2_all = all(verdict_bits[f"{r}_p2"][0] for r in RULES)
    p2_core = all(verdict_bits[f"{r}_p2"][1] for r in RULES)
    m0_alone = all(verdict_bits[f"{r}_m0p2"] for r in RULES) and m0_p3
    p3_ok = len(p3_fail) == 0
    if stop2 or not p2_core or len(p3_fail) >= 2:
        verdict = "not supported"
    elif p2_all and p3_ok and p1 and p5_ok and c1_clears:
        verdict = "supported"
    elif p2_all and p3_ok:
        verdict = "partly supported"
    elif m0_alone or drift > 0.5:
        verdict = "inconclusive"
    else:
        verdict = "not supported"
    if m0_alone and p2_all and p3_ok:
        verdict = "inconclusive"
    if drift > 0.5 and verdict in {"supported", "partly supported"}:
        verdict = "inconclusive"
    say(
        f"\nVERDICT by the pre-registered table: {verdict} "
        f"(stop2 {stop2}, P1 {p1}, P2 all {p2_all}, P2 core {p2_core}, P3 failures {p3_fail}, "
        f"P4/C1 clears {c1_clears}, P5 {p5_ok}, M0 alone {m0_alone}, u drift {drift:.2f})"
    )

    figures(fa_rows=fa_rows, boots=boots, example=example, eras=eras)
    REPORT_PATH.write_text("\n".join(LINES) + "\n")


# ------------------------------------------------------------------ figures


def figures(*, fa_rows: list[dict], boots: dict, example: dict, eras: dict) -> None:
    """Draw figures 6 to 9."""
    # Figure 6: one injected segment, raw reading first, then each arm's flags
    start = 30 * 48
    n = 56 * 48
    day = np.arange(n) / 48.0
    truth = example["truth"][start : start + n]
    rows = []
    rows += [
        {"day": d, "value": v, "panel": "Reading before injection (MVA)"}
        for d, v in zip(day, example["clean"][start : start + n], strict=True)
    ]
    rows += [
        {"day": d, "value": v, "panel": "Reading after injection (MVA)"}
        for d, v in zip(day, example["y"][start : start + n], strict=True)
    ]
    base = alt.Chart(pl.DataFrame(rows)).mark_line(strokeWidth=1, clip=True, color=ocf.DATA_BLUE)
    top = (
        base.encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:Q", title=None, axis=alt.Axis(values=list(range(0, 57, 7)))),
            y=alt.Y("value:Q", title="MVA"),
        )
        .properties(width=WIDTH, height=90)
        .facet(
            row=alt.Row(
                "panel:N",
                title=None,
                sort=["Reading before injection (MVA)", "Reading after injection (MVA)"],
                header=alt.Header(labelOrient="top", labelAngle=0, labelFontSize=9),
            ),
            spacing=4,
        )
    )
    state_rows = []
    for d, t in zip(day, truth, strict=True):
        state_rows.append({"day": d, "who": "Truth (units)", "units": int(t)})
    for arm in ("M2", "M1", "M0"):
        flag = example["arms"][arm]["flag"][start : start + n]
        for d, f in zip(day, flag, strict=True):
            state_rows.append({"day": d, "who": f"{arm} flag", "units": int(f)})
    states = (
        alt.Chart(pl.DataFrame(state_rows))  # ty: ignore[unresolved-attribute]
        .mark_line(interpolate="step-after", strokeWidth=1.2, clip=True)
        .encode(
            x=alt.X(
                "day:Q",
                title="Days from the start of the 56-day window (no calendar dates)",
                axis=alt.Axis(values=list(range(0, 57, 7))),
            ),
            y=alt.Y("units:Q", title=None, axis=alt.Axis(values=[0, 1, 2, 3])),
            color=alt.Color(
                "who:N",
                scale=alt.Scale(
                    domain=["Truth (units)", "M2 flag", "M1 flag", "M0 flag"],
                    range=[ocf.BLACK_2, ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH, height=70)
        .facet(
            row=alt.Row(
                "who:N",
                title=None,
                header=alt.Header(labelOrient="top", labelAngle=0, labelFontSize=9),
            ),
            spacing=2,
        )
    )
    fig6 = alt.vconcat(top, states).properties(
        title=alt.TitleParams(
            wrap(
                "One injected example (a host Primary, Gas A's 44-day record, folded rule). Top: the host's reading before and after injection. Bottom: the true number of units running and each tracker's flag."
            ),
            anchor="start",
        )
    )
    save(chart=fig6, name="figure_6_injected_example")

    # Figure 7: F1, precision, recall per arm, both rules
    rows = []
    for rule, (x, b, _) in boots.items():
        m = pooled_metrics(x=x)
        for ai, arm in enumerate(ARMS):
            s = b[:, ai, :]
            tp, fp, fn = s[:, 0], s[:, 1], s[:, 2]
            for name, dist, pt in (
                ("F1", f1_arr(s=s), m["f1"][ai]),
                ("Precision", tp / np.maximum(tp + fp, 1), m["precision"][ai]),
                ("Recall", tp / np.maximum(tp + fn, 1), m["recall"][ai]),
            ):
                lo, hi = interval(d=dist, level=0.95)
                rows.append(
                    {
                        "rule": f"{rule} rule",
                        "arm": arm,
                        "metric": name,
                        "point": float(pt),
                        "lo": lo,
                        "hi": hi,
                    }
                )
    frame = pl.DataFrame(rows)
    pts = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X(
                "point:Q",
                title="Score on half-hours with a unit running (higher is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("arm:N", title=None, sort=list(ARMS)),
            color=alt.Color(
                "arm:N", scale=alt.Scale(domain=list(ARMS), range=ARM_COLOURS), legend=None
            ),
        )
    )
    bars = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2)
        .encode(
            x="lo:Q",
            x2="hi:Q",
            y=alt.Y("arm:N", sort=list(ARMS)),
            color=alt.Color(
                "arm:N", scale=alt.Scale(domain=list(ARMS), range=ARM_COLOURS), legend=None
            ),
        )
    )
    fig7 = (
        (bars + pts)
        .properties(width=WIDTH - 60, height=90)
        .facet(
            row=alt.Row("metric:N", title=None), column=alt.Column("rule:N", title=None), spacing=6
        )
        .properties(
            title=alt.TitleParams(
                wrap(
                    "Known-answer scores of each tracker, pooled over the engine record and 4 synthetic schedules injected into host series. Dot: pooled score. Line: 95% interval from resampling schedule-weeks and then host segments."
                ),
                anchor="start",
            )
        )
    )
    save(chart=fig7, name="figure_7_known_answer_scores")

    # Figure 8: false alarms
    frame = pl.DataFrame(
        [
            {k: v for k, v in r.items() if k in {"host", "arm", "share"}}
            | {"share_pct": 100 * r["share"]}
            for r in fa_rows
        ]
    )
    bars = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("host:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("arm:N", sort=list(ARMS)),
            y=alt.Y(
                "share_pct:Q",
                title="Half-hours flagged (%; smaller is better)",
                scale=alt.Scale(type="symlog", constant=1),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARMS), range=ARM_COLOURS),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    rule3 = (
        alt.Chart(pl.DataFrame({"y": [3.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=ocf.BLACK_2)
        .encode(y="y:Q")
    )
    fig8 = (bars + rule3).properties(
        width=WIDTH,
        height=200,
        title=alt.TitleParams(
            wrap(
                "Share of half-hours each tracker flags as engine running on un-injected history, by host Primary (D8 is before the engine). Dashed line: the 3% limit."
            ),
            anchor="start",
        ),
    )
    save(chart=fig8, name="figure_8_false_alarms")

    # Figure 9: D8 monthly running share
    rows = []
    for label, e in eras.items():
        for arm in ("M2", "M1"):
            flags = e[arm]["flag"]
            months = e["month"]
            for m_ in np.unique(months):
                sel = months == m_
                if sel.sum() >= 600:
                    rows.append(
                        {
                            "month": str(m_),
                            "arm": arm,
                            "share": float(flags[sel].mean()),
                            "era": label,
                        }
                    )
    frame = pl.DataFrame(rows).with_columns(pl.col("month").str.to_date("%Y-%m"))
    fig9 = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title="Month", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("share:Q", title="Share of half-hours flagged as running"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=["M2", "M1"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top", title=None),
            ),
            detail="era:N",
        )
        .properties(
            width=WIDTH,
            height=170,
            title=alt.TitleParams(
                wrap(
                    "Share of D8's half-hours flagged as engine running, by month, fitted separately on each of three eras with no engine record. Months with fewer than 600 tracked half-hours are left out."
                ),
                anchor="start",
            ),
        )
    )
    save(chart=fig9, name="figure_9_d8_running_share")


if __name__ == "__main__":
    main()

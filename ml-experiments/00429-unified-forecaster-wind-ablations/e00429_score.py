# ruff: noqa: E501
"""Score the arms of experiment 00429 on the 53 scored Tuesday runs and print the report.

    source ml-experiments/00429-*/env.sh
    uv run --no-sync python ml-experiments/00429-*/e00429_score.py [--g0]

`--g0` prints only the reproduction gate. Otherwise the report holds absolute scores, tails, the
median-minus-mean check, the five pre-registered contrasts (99.8% month-block intervals, Bonferroni
over five), reference contrasts, and each contrast's minimum detectable effect. Per-seed scores
are averaged within each row before resampling, as experiment 00347 does.
"""

import json
import os
import sys
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_score as score326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_score as score347  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00429_data as data  # noqa: E402
import e00429_run as run  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

G0_TOLERANCE: Final[float] = 0.05
G0_EXPECTED: Final[dict[str, float]] = {"kept": 11.081, "true_1_10": 10.518}
"""Experiment 00347's y2, from its published stage-4 report."""
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 member_plain - one_pass_ms (A1)": ("member_plain", "one_pass_ms"),
    "C2 member_plain - one_pass_m (A1)": ("member_plain", "one_pass_m"),
    "C3 member_frozen - member_plain (A4)": ("member_frozen", "member_plain"),
    "C4 member_e2e - member_frozen (A4)": ("member_e2e", "member_frozen"),
    "C6 memberwise pooled mixture - one_pass_ms (A1)": ("memberwise", "one_pass_ms"),
}
CENSORED_CONTRAST: Final[str] = "C5 {arm}_censored - {arm} (A7)"
LEVEL: Final[float] = 1 - 0.01 / 6
SEEDS_EXPECTED: Final[int] = 5
MARGIN: Final[float] = 0.10
"""The maintainer agreed this margin for wind on 2026-10-31."""
UPPER_TAIL_LIMIT: Final[float] = 0.015
"""Share of observations above q99 at true days 1 to 10; an arm above it fails the upper tail."""
TAIL_LIMIT: Final[float] = 0.02
SUBSETS: Final[tuple[str, ...]] = (
    "kept",
    "true_1_10",
    "true_3_10",
    "h12_36",
    "true_1_14",
    "all",
    "days_1_2",
    "days_3_10",
)
REFERENCES: Final[tuple[str, ...]] = ("ref_347_y2", "ref_default", "ref_clim")


class Intervals(score347.Intervals):
    """The month resampler read at 95% and at the Bonferroni level for five contrasts."""

    def read(self, *, values: np.ndarray, keep: np.ndarray) -> dict:
        """Return the mean and the 95% and 99.8% intervals."""
        picked, totals = self.draws(values=values, keep=keep)
        means = picked / np.maximum(totals, 1)
        out: dict = {"mean": float(values[:, keep].mean())}
        for level in (0.95, LEVEL):
            low, high = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
            out[f"{level:.4f}"] = [float(low), float(high)]
        return out


def fmt(entry: dict) -> str:
    """Return 'mean (95% low to high; 99.8% low to high)'."""
    low, high = entry["0.9500"]
    wide_low, wide_high = entry[f"{LEVEL:.4f}"]
    return (
        f"{entry['mean']:+.3f} (95% {low:+.3f} to {high:+.3f}; "
        f"99.8% {wide_low:+.3f} to {wide_high:+.3f})"
    )


def load_parts(*, name: str) -> np.ndarray | None:
    """Return the stored quantiles of `name` for every seed on disk: (seeds, rows, 11)."""
    paths = [run.PARTS_DIR / f"{name}_h{s}.npy" for s in run.SEEDS]
    present = [np.load(p) for p in paths if p.exists()]
    if present and len(present) != SEEDS_EXPECTED:
        msg = f"{name}: {len(present)} seeds on disk, expected {SEEDS_EXPECTED}"
        raise ValueError(msg)
    return np.stack(present) if present else None


def load_arms() -> dict[str, np.ndarray]:
    """Return every arm and reference with stored rows."""
    arms = {n: p for n in run.ARMS if (p := load_parts(name=n)) is not None}
    arms["ref_347_y2"] = np.stack(
        [np.load(data347.PARTS_DIR / f"y2_h{s}.npy") for s in score347.SEEDS]
    )
    arms["ref_default"] = np.load(data347.E339_FINAL / "default.npy")
    arms["ref_clim"] = np.load(data347.E324_FINAL / "W0.npy")
    return arms


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print the report."""
    cases = data347.scored_frame().with_columns(lead_day=pl.col("lead_day").clip(0, 10))
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    arms = load_arms()
    flags = {
        "all": np.ones(cases.height, dtype=bool),
        "kept": ~cases["masked"].to_numpy(),
        "days_1_2": cases["lead_day"].is_between(1, 2).to_numpy(),
        "days_3_10": cases["lead_day"].is_between(3, 10).to_numpy(),
        "true_1_10": (~cases["masked"] & cases["lead_hours"].is_between(24, 240)).to_numpy(),
        "true_3_10": (~cases["masked"] & cases["lead_hours"].is_between(72, 240)).to_numpy(),
        "h12_36": (
            ~cases["masked"] & (cases["lead_hours"] >= 12) & (cases["lead_hours"] < 36)
        ).to_numpy(),
        "true_1_14": (~cases["masked"] & cases["lead_hours"].is_between(24, 336)).to_numpy(),
    }
    sites = cases["site"].to_numpy()
    resampler = Intervals(month=cases["month"].to_numpy(), seeds=1)
    metrics: dict[str, dict[str, np.ndarray]] = {}
    seed_sd: dict[str, dict[str, float]] = {}
    for name, q in arms.items():
        raw = {
            **score303.row_metrics(quantiles=q, actual=actual),
            **score326.extra_metrics(quantiles=q, actual=actual),
        }
        metrics[name] = {k: v.mean(axis=0, keepdims=True) for k, v in raw.items()}
        seed_sd[name] = {
            s: float(np.std(raw["qs"][:, flags[s]].mean(axis=1), ddof=1)) if len(q) > 1 else np.nan
            for s in ("kept", "true_1_10")
        }
    qs = {n: m["qs"] for n, m in metrics.items()}
    lines = [f"EXPERIMENT 429: scored cases {cases.height} ({int(flags['kept'].sum())} kept)"]

    lines.append("-- reproduction gate G0 (experiment 00347's y2 against the shared code) --")
    if "g0" in qs:
        for subset, expected in G0_EXPECTED.items():
            got = float(qs["g0"][0, flags[subset]].mean())
            stored = float(qs["ref_347_y2"][0, flags[subset]].mean())
            lines.append(
                f"G0 {subset}: {got:.3f}; stored y2 {stored:.3f}; published {expected:.3f}; "
                f"gap to published {got - expected:+.3f}; "
                f"{'PASS' if abs(got - expected) <= G0_TOLERANCE else 'MISS'} (tolerance {G0_TOLERANCE})"
            )
    else:
        lines.append("G0 has no stored rows")
    if "--g0" in sys.argv:
        print("\n".join(lines))
        return

    lines.append("-- absolute quantile scores (points of capacity; mean over seeds) --")
    lines.append("arm".ljust(28) + "".join(s.rjust(11) for s in SUBSETS) + "  seed sd kept / 1-10")
    for name in qs:
        row = "".join(f"{qs[name][0, flags[s]].mean():11.3f}" for s in SUBSETS)
        lines.append(
            f"{name.ljust(28)}{row}  {seed_sd[name]['kept']:.3f} / {seed_sd[name]['true_1_10']:.3f}"
        )

    lines.append(
        "-- tails, kept rows at true days 1 to 10: pinball at q01, q05, q95, q99; share beyond q01, q99 --"
    )
    for name, m in metrics.items():
        keep = flags["true_1_10"]
        lines.append(
            f"{name.ljust(28)}"
            + "  ".join(f"{k} {m[k][0, keep].mean():.3f}" for k in ("p01", "p05", "p95", "p99"))
            + f"  below_q01 {m['below_q01'][0, keep].mean():.4f}  above_q99 {m['above_q99'][0, keep].mean():.4f}"
        )

    lines.append("-- tails per farm, true_1_10 --")
    for name, m in metrics.items():
        for farm in data.FARMS:
            keep = flags["true_1_10"] & (sites == farm)
            lines.append(
                f"{name.ljust(28)}{farm}  "
                + "  ".join(f"{k} {m[k][0, keep].mean():.3f}" for k in ("p01", "p05", "p95", "p99"))
                + f"  below_q01 {m['below_q01'][0, keep].mean():.4f}"
                f"  above_q99 {m['above_q99'][0, keep].mean():.4f}"
            )
    lines.append("-- check: median minus mean forecast per farm, kept rows (points) --")
    for name, m in metrics.items():
        lines.append(
            name.ljust(28)
            + "  ".join(
                f"{f} {m['median_minus_mean'][0, flags['kept'] & (sites == f)].mean():+.3f}"
                for f in data.FARMS
            )
        )
    lines.append("-- check: error of the mean forecast, root mean square in points, kept rows --")
    for name, m in metrics.items():
        lines.append(f"{name.ljust(28)}{np.sqrt(m['mean_sq_error'][0, flags['kept']].mean()):.3f}")

    member_arms = [n for n in run.ARMS if (run.PARTS_DIR / f"{n}_members_h0.npy").exists()]
    if member_arms:
        lines.append(
            "-- per-member quantile forecasts: score of each member's 11 quantiles against the "
            "observation, averaged over the 51 members; share of observations beyond a member's q01 "
            "and q99, averaged over members --"
        )
    member_scores: dict[str, np.ndarray] = {}
    for name in member_arms:
        per_seed, below, above = [], [], []
        for s in run.SEEDS:
            mq = np.load(run.PARTS_DIR / f"{name}_members_h{s}.npy")
            flat = mq.reshape(-1, mq.shape[2]).astype(np.float64)
            rows_score = c303.quantile_score(quantiles=flat, actual=np.repeat(actual, mq.shape[1]))
            per_seed.append(
                rows_score.reshape(len(actual), -1).mean(axis=1) * c303.POINTS_PER_FRACTION
            )
            below.append((actual[:, None] < mq[:, :, 0]).mean(axis=1))
            above.append((actual[:, None] > mq[:, :, 10]).mean(axis=1))
        member_scores[name] = np.mean(per_seed, axis=0)
        qs[f"{name}_member_mean"] = member_scores[name][None]
        pooled = qs[name][0]
        for label, keep in (
            ("true_1_10", flags["true_1_10"]),
            ("h12_36", flags["h12_36"]),
            ("true_1_14", flags["true_1_14"]),
            ("kept", flags["kept"]),
        ):
            lines.append(
                f"{name} {label}: mean member score {member_scores[name][keep].mean():.3f}; "
                f"pooled mixture {pooled[keep].mean():.3f}; share below member q01 "
                f"{np.mean(below, axis=0)[keep].mean():.4f}; above member q99 "
                f"{np.mean(above, axis=0)[keep].mean():.4f}"
            )

    contrasts = dict(CONTRASTS)
    for base in ("one_pass_ms", "member_plain", "member_frozen", "member_e2e"):
        if f"{base}_censored" in qs:
            contrasts[CENSORED_CONTRAST.format(arm=base)] = (f"{base}_censored", base)
    report: dict = {"contrasts": {}}
    lines.append(
        f"-- pre-registered contrasts (negative favours the first arm); 99% Bonferroni level "
        f"{LEVEL:.4f}; margin {MARGIN} --"
    )
    for label, (a, b) in contrasts.items():
        if a not in qs or b not in qs:
            lines.append(f"{label}: arm missing")
            continue
        entries = {}
        for subset in ("true_1_10", "true_3_10", "kept"):
            entries[subset] = resampler.read(values=qs[a] - qs[b], keep=flags[subset])
            report["contrasts"][f"{label} | {subset}"] = entries[subset]
            lines.append(f"{label} | {subset}: {fmt(entries[subset])}")
        keep = flags["true_1_10"]
        tail = {
            k: float((metrics[a][k][0, keep] - metrics[b][k][0, keep]).mean())
            for k in ("p01", "p99")
        }
        relative = tail["p99"] / float(metrics[b]["p99"][0, keep].mean())
        lines.append(
            f"{label} | tail change at true_1_10: q01 {tail['p01']:+.3f}, q99 {tail['p99']:+.3f}"
            f" (limit {TAIL_LIMIT}); q99 pinball relative change {relative:+.1%}"
        )
        low, high = entries["true_1_10"][f"{LEVEL:.4f}"]
        half_width = (high - low) / 2
        sds = [seed_sd[a]["true_1_10"], seed_sd[b]["true_1_10"]]
        lines.append(
            f"{label} | minimum detectable effect at true_1_10: interval half-width {half_width:.3f}"
            f"; seed sd of the two arms {sds[0]:.3f}, {sds[1]:.3f}"
        )
        decisive = "true_3_10" if label.startswith("C1") else "true_1_10"
        d_low, d_high = entries[decisive][f"{LEVEL:.4f}"]
        point = entries[decisive]["mean"]
        harm = max(tail.values()) > TAIL_LIMIT
        upper = float(metrics[a]["above_q99"][0, keep].mean())
        if upper > UPPER_TAIL_LIMIT:
            lines.append(
                f"{label} | UPPER-TAIL FAIL: {a} has {upper:.2%} of observations above q99 "
                f"(limit {UPPER_TAIL_LIMIT:.1%}); it does not advance"
            )
            harm = True
        if d_high < 0 and abs(point) >= MARGIN and not harm:
            verdict = "KEPT for the first arm: interval below zero, beats the margin, no tail harm"
        elif upper > UPPER_TAIL_LIMIT:
            verdict = "BLOCKED: the first arm fails the upper-tail rule, whatever its score"
        elif d_low > 0 or harm:
            verdict = "first arm is worse, or tail harm blocks it"
        elif (d_high - d_low) / 2 > MARGIN:
            verdict = "INCONCLUSIVE: interval half-width exceeds the margin"
        else:
            verdict = "NOT KEPT: no gain beyond the margin, and the interval is narrow"
        lines.append(f"{label} | verdict on {decisive}: {verdict}")
    lines.append(
        "-- reference contrasts, first arm minus reference, true_1_10 and kept (95% / 99.8%) --"
    )
    for name in qs:  # noqa: PLC0206
        if name in REFERENCES:
            continue
        for ref in ("ref_347_y2", "ref_default"):
            for subset in ("true_1_10", "kept"):
                entry = resampler.read(values=qs[name] - qs[ref], keep=flags[subset])
                lines.append(f"{name} minus {ref} | {subset}: {fmt(entry)}")
    text = "\n".join(lines)
    print(text)
    (data.DATA_DIR / "report.txt").write_text(text + "\n")
    (data.DATA_DIR / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

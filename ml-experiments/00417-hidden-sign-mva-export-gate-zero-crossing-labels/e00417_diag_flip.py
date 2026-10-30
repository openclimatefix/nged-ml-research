# ruff: noqa: E501
"""Zero-fit diagnostic for design I (maintainer's addendum): a whole-day slope flip.

For each day, take the half-hours with CAMS clear-sky index `k >= 0.7` and scaled unit shape `u >=
0.05` (CAMS irradiance enters through `u`, which rises monotonically with it). Remove the hour-by-week
demand profile (the mean reading at each half-hour of the day and ISO week over the Primary's own
non-sunny half-hours, `k < 0.7`, in the same window). Order the day's sunny half-hours by `u`, fit the
slope of the residual on `u` over the lower half and the upper half, and score the day as
`min(-slope_lower, slope_upper)`: positive when the slope is negative on the lower half and positive
on the upper half (the flip). A day is flagged when its score is positive. A second flag requires the
score to exceed the 95th percentile of the score over the no-injection days of the visible signed
Primaries (the same rule is fitted on visible Primaries only, never the hidden pair).

```bash
. env.sh && uv run --no-sync python ml-experiments/00417-*/e00417_diag_flip.py run
```
"""

import argparse
import json
from typing import Final

import e00417_diagnostics as diag
import e00417_inputs as inputs
import numpy as np
import polars as pl

MIN_SUNNY_HALF_HOURS: Final[int] = 12
MIN_PER_HALF: Final[int] = 6
MIN_U: Final[float] = 0.05
CLEAR_K: Final[float] = 0.7
CALIBRATION_PERCENTILE: Final[float] = 95.0


def _keys(*, t: diag.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the (ISO week, half-hour of day) key, the month, and the UTC day of each row."""
    start = pl.Series(t.start_us).cast(pl.Datetime("us"))
    halfhour = ((t.start_us % inputs.US_PER_DAY) // inputs.US_PER_HALF_HOUR).astype(int)
    return (
        start.dt.week().to_numpy() * 100 + halfhour,
        start.dt.month().to_numpy(),
        t.start_us // inputs.US_PER_DAY,
    )


def build_profile(*, t: diag.Table) -> dict[int, float]:
    """Return the mean reading at each (ISO week, half-hour) over the table's non-sunny rows."""
    key, _month, _day = _keys(t=t)
    sunny = (t.k >= CLEAR_K) & (t.u >= MIN_U)
    frame = (
        pl.DataFrame({"g": key[~sunny], "r": t.r[~sunny]})
        .group_by("g")
        .agg(pl.col("r").mean().alias("p"), pl.len().alias("n"))
        .filter(pl.col("n") >= 5)
    )
    return dict(zip(frame["g"].to_list(), frame["p"].to_list(), strict=True))


def day_scores(*, t: diag.Table, profile: dict[int, float] | None = None) -> dict[str, np.ndarray]:
    """Return each day's flip score, slopes, and flip point (`u` at the split) for summer days.

    The demand profile comes from `profile` (built on a training window) or, if absent, from the
    table itself.
    """
    key, month, day = _keys(t=t)
    sunny = (t.k >= CLEAR_K) & (t.u >= MIN_U)
    lookup = profile if profile is not None else build_profile(t=t)
    resid = np.array([t.r[i] - lookup.get(key[i], np.nan) for i in range(len(key))])
    out: dict[str, list[float]] = {"day": [], "score": [], "lower": [], "upper": [], "flip_u": []}
    summer = (month >= 4) & (month <= 9)
    for d in np.unique(day[sunny & summer]):
        sel = sunny & (day == d) & np.isfinite(resid)
        if sel.sum() < MIN_SUNNY_HALF_HOURS:
            continue
        u, r = t.u[sel], resid[sel]
        order = np.argsort(u)
        u, r = u[order], r[order]
        half = len(u) // 2
        if half < MIN_PER_HALF:
            continue
        lower = float(np.polyfit(u[:half], r[:half], 1)[0])
        upper = float(np.polyfit(u[half:], r[half:], 1)[0])
        out["day"].append(float(d))
        out["score"].append(min(-lower, upper))
        out["lower"].append(lower)
        out["upper"].append(upper)
        out["flip_u"].append(float(u[half]))
    return {k: np.array(v) for k, v in out.items()}


def run() -> dict:
    """Score every setting and write the table."""
    tables = {
        (name, s): diag.load_table(site=s, setting=setting)
        for name, setting in diag.SETTINGS.items()
        for s in inputs.SIGNED_SIX
    }
    scores = {key: day_scores(t=t) for key, t in tables.items()}
    results: dict = {}
    for name, placebo_name in diag.PLACEBO_OF.items():
        visible = np.concatenate([scores[(placebo_name, s)]["score"] for s in diag.VISIBLE_FOR_R1])
        threshold = float(np.percentile(visible, CALIBRATION_PERCENTILE))
        inj = {s: scores[(name, s)]["score"] for s in inputs.SIGNED_SIX}
        pla = {s: scores[(placebo_name, s)]["score"] for s in diag.PLACEBO_SITES}
        flags = {
            tag: {
                "injected_recall": {s: float(np.mean(v > cut)) for s, v in inj.items()},
                "placebo_false_flag": {s: float(np.mean(v > cut)) for s, v in pla.items()},
                "pooled_recall": float(np.mean(np.concatenate(list(inj.values())) > cut)),
                "pooled_false_flag": float(np.mean(np.concatenate(list(pla.values())) > cut)),
            }
            for tag, cut in (("sign_only", 0.0), ("calibrated", threshold))
        }
        results[name] = {
            "pooled": diag.pooled_auc_with_interval(injected=inj, placebo=pla),
            "per_primary": {
                s: diag.pooled_auc_with_interval(
                    injected={s: inj[s]}, placebo={s: scores[(placebo_name, s)]["score"]}
                )
                for s in inputs.SIGNED_SIX
            },
            "calibrated_threshold": threshold,
            "n_days_injected": {s: len(v) for s, v in inj.items()},
            "flags": flags,
        }
    (diag.OUT_DIR / "diag_flip.json").write_text(json.dumps(results, indent=1))
    return results


def main() -> None:
    """Run and print."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run",))
    parser.parse_args()
    for name, e in run().items():
        p = e["pooled"]
        print(
            f"\n== {name}: I AUC {p['auc']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}] (n {p['n_injected']} vs {p['n_placebo']}); per Primary { {s: round(v['auc'], 2) for s, v in e['per_primary'].items()} }"
        )
        for tag, f in e["flags"].items():
            print(
                f"  {tag}: recall {f['pooled_recall']:.2f}, false flag {f['pooled_false_flag']:.3f}; D5 recall {f['injected_recall']['D5']:.2f}, D5 placebo flag {f['placebo_false_flag']['D5']:.3f}"
            )


if __name__ == "__main__":
    main()

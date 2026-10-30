# ruff: noqa: E501
"""Refined day-level zero-fit diagnostic: design I merged with design C (the Opus review of 2026-10-30).

Each April-to-September day of a folded Primary is scored by how much better a V fits the
demand-removed reading than a non-increasing curve does, with the flip required on the morning and
the afternoon limbs separately:

- **Hourly means.** The reading, `u`, `k`, and ERA5 temperature are averaged over the two half-hours
  of each UTC hour (CAMS is hourly).
- **Demand profile.** `D(h)` is the median reading at hour `h` over overcast days (daily mean `k` of
  the hours with `u > 0.01` below 0.3) of the same 28-day block and day type (weekday, or Saturday,
  Sunday, and bank holiday), widened to all day types and then to the neighbouring blocks when fewer
  than 3 overcast hours exist, plus a linear ERA5-temperature term fitted on the overcast hours.
- **Residual.** `e = r^2 - D^2` on hours with `0.1 <= u < 0.95`.
- **Limb score.** On each limb (hours before and after solar noon), the drop in residual sum of
  squares from the best non-increasing fit of `e` on `u` (pool adjacent violators) to the best
  continuous two-piece line with a positive upper slope, divided by the two-piece sum of squares;
  with a negative lower slope as well (a V) and at least 3 hours either side of the breakpoint. The day's score is the smaller of the two limbs
  and its breakpoint is their mean. A day with a daily maximum `u` below 0.6 is ineligible.
- **Threshold.** The 98th percentile of the score over the no-injection days of the other signed
  Primaries (leave one Primary out in the diagnostic; the visible four in a fit).

```bash
. env.sh && uv run --no-sync python ml-experiments/00417-*/e00417_day_flip.py run
```
"""

import argparse
import json
from dataclasses import dataclass
from typing import Final

import e00417_diagnostics as diag
import e00417_inputs as inputs
import numpy as np
import polars as pl

US_PER_HOUR: Final[int] = 3_600_000_000
U_LOW: Final[float] = 0.1
U_HIGH: Final[float] = 0.95
MIN_DAY_MAX_U: Final[float] = 0.6
OVERCAST_K: Final[float] = 0.3
MIN_OVERCAST_HOURS: Final[int] = 3
MIN_SIDE: Final[int] = 3
THRESHOLD_PERCENTILE: Final[float] = 98.0
BLOCK_DAYS: Final[int] = 28
MAX_EARLIER_BLOCKS: Final[int] = 6
SUMMER: Final[tuple[int, int]] = (4, 9)
# Mean absolute second difference of hourly u above which a day counts as broken cloud.
BROKEN_ROUGHNESS: Final[float] = 0.05


@dataclass
class Hourly:
    """A Primary's hourly means over a window."""

    hour: np.ndarray
    day: np.ndarray
    hod: np.ndarray
    r: np.ndarray
    u: np.ndarray
    k: np.ndarray
    temp: np.ndarray
    weekend: np.ndarray
    month: np.ndarray
    noon_hod: float


def hourly(*, t: diag.Table, u: np.ndarray | None = None) -> Hourly:
    """Average a table's half-hours into UTC hours (`u` may replace the table's, as in the shuffle)."""
    u = t.u if u is None else u
    hour = t.start_us // US_PER_HOUR
    frame = (
        pl.DataFrame(
            {
                "h": hour,
                "r": t.r,
                "u": u,
                "k": t.k,
                "t": np.nan_to_num(t.temp, nan=float(np.nanmean(t.temp))),
                "w": (t.dow >= 6) | t.bank_holiday,
            }
        )
        .group_by("h")
        .agg(
            pl.col("r").mean(),
            pl.col("u").mean(),
            pl.col("k").mean(),
            pl.col("t").mean(),
            pl.col("w").max(),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") == 2)
        .sort("h")
    )
    h = frame["h"].to_numpy()
    month = pl.Series(h * US_PER_HOUR).cast(pl.Datetime("us")).dt.month().to_numpy()
    return Hourly(
        hour=h,
        day=h // 24,
        hod=h % 24,
        r=frame["r"].to_numpy(),
        u=frame["u"].to_numpy(),
        k=frame["k"].to_numpy(),
        temp=frame["t"].to_numpy(),
        weekend=frame["w"].to_numpy().astype(bool),
        month=month,
        noon_hod=t.noon_utc_hour,
    )


@dataclass
class Profile:
    """Overcast-day demand medians by (block, day type, hour of day) and the temperature slopes."""

    median_r: dict[tuple[int, int, int], float]
    median_t: dict[tuple[int, int, int], float]
    slope: dict[int, float]


def build_profile(*, h: Hourly) -> Profile:
    """Return the overcast-day profile of a window."""
    daylight = h.u > 0.01
    frame = pl.DataFrame({"d": h.day, "k": np.where(daylight, h.k, np.nan), "u": h.u}).with_columns(
        pl.col("k").fill_nan(None)
    )
    daily = frame.group_by("d").agg(
        pl.col("k").mean().alias("mk"), (pl.col("u") > 0.01).sum().alias("nd")
    )
    overcast_days = {
        int(d) for d, mk, nd in daily.iter_rows() if nd >= 6 and mk is not None and mk < OVERCAST_K
    }
    sel = np.isin(h.day, list(overcast_days))
    block = h.day // BLOCK_DAYS
    key = np.stack([block, h.weekend.astype(int), h.hod], axis=1)
    frame = pl.DataFrame(
        {"b": key[sel, 0], "w": key[sel, 1], "h": key[sel, 2], "r": h.r[sel], "t": h.temp[sel]}
    )
    grouped = frame.group_by("b", "w", "h").agg(
        pl.col("r").median().alias("mr"), pl.col("t").median().alias("mt"), pl.len().alias("n")
    )
    median_r, median_t = {}, {}
    for b, w, hh, mr, mt, n in grouped.iter_rows():
        if n >= MIN_OVERCAST_HOURS:
            median_r[(b, w, hh)], median_t[(b, w, hh)] = mr, mt
    slope = {}
    for hh in range(24):
        xs, ys = [], []
        for b, w, hh2, r, tt in frame.iter_rows():
            if hh2 == hh and (b, w, hh) in median_r:
                xs.append(tt - median_t[(b, w, hh)])
                ys.append(r - median_r[(b, w, hh)])
        slope[hh] = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 10 and np.ptp(xs) > 0 else 0.0
    return Profile(median_r=median_r, median_t=median_t, slope=slope)


def expected(*, p: Profile, b: int, w: int, hh: int, temp: float) -> float:
    """Return the expected demand reading, widening day type and then block when a cell is empty."""
    earlier = [(b - k, ww) for k in range(1, MAX_EARLIER_BLOCKS + 1) for ww in (w, 1 - w)]
    for bb, ww in ((b, w), (b, 1 - w), *earlier, (b + 1, w), (b + 1, 1 - w)):
        if (bb, ww, hh) in p.median_r:
            return p.median_r[(bb, ww, hh)] + p.slope[hh] * (temp - p.median_t[(bb, ww, hh)])
    return float("nan")


def pav_decreasing_rss(*, y: np.ndarray) -> float:
    """Return the residual sum of squares of the best non-increasing fit to `y` (in order of `u`)."""
    blocks: list[list[float]] = []
    for v in y:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, n2 = blocks.pop()
            m1, n1 = blocks.pop()
            blocks.append([(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2])
    fit = np.concatenate([[m] * int(n) for m, n in blocks])
    return float(np.sum((y - fit) ** 2))


def limb_score(*, u: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """Return (score, breakpoint) of one limb; score 0 and NaN when no feasible V exists."""
    order = np.argsort(u)
    u, e = u[order], e[order]
    n = len(u)
    if n < 2 * MIN_SIDE:
        return 0.0, float("nan")
    iso = pav_decreasing_rss(y=e)
    best_rss, best_bp = np.inf, float("nan")
    for k in range(MIN_SIDE, n - MIN_SIDE):
        bp = u[k]
        design = np.stack([np.ones(n), u, np.maximum(u - bp, 0.0)], axis=1)
        coef, *_ = np.linalg.lstsq(design, e, rcond=None)
        if coef[1] >= 0 or coef[1] + coef[2] <= 0:
            continue
        rss = float(np.sum((e - design @ coef) ** 2))
        if rss < best_rss:
            best_rss, best_bp = rss, float(bp)
    if not np.isfinite(best_rss) or best_rss <= 0:
        return 0.0, float("nan")
    return max((iso - best_rss) / best_rss, 0.0), best_bp


@dataclass
class DayResult:
    """Scores, breakpoints, and mean midday residuals of a Primary's eligible days."""

    day: np.ndarray
    score: np.ndarray
    breakpoint: np.ndarray
    level: np.ndarray
    broken: np.ndarray


def score_days(*, h: Hourly, profile: Profile) -> DayResult:
    """Score every eligible summer day."""
    days, scores, bps, levels, broken_flags = [], [], [], [], []
    daily_max = (
        pl.DataFrame({"d": h.day, "u": h.u}).group_by("d").agg(pl.col("u").max()).iter_rows()
    )
    for d, umax in daily_max:
        if umax < MIN_DAY_MAX_U:
            continue
        sel = np.flatnonzero(h.day == d)
        if not (SUMMER[0] <= h.month[sel[0]] <= SUMMER[1]):
            continue
        b = int(d) // BLOCK_DAYS
        e_all = np.array(
            [
                h.r[i] ** 2
                - expected(
                    p=profile, b=b, w=int(h.weekend[i]), hh=int(h.hod[i]), temp=float(h.temp[i])
                )
                ** 2
                for i in sel
            ]
        )
        use = (h.u[sel] >= U_LOW) & (h.u[sel] < U_HIGH) & np.isfinite(e_all)
        u_day = h.u[sel]
        daylight = u_day >= U_LOW
        u_by_time = u_day[daylight][np.argsort(h.hod[sel][daylight])]
        roughness = float(np.mean(np.abs(np.diff(u_by_time, n=2)))) if len(u_by_time) > 2 else 0.0
        broken = roughness > BROKEN_ROUGHNESS
        if broken:
            # Broken cloud: a V in u over the whole day, with at least MIN_SIDE hours (6 half-hours)
            # on each arm, whatever the time of day. Several crossings are repeat visits to one u*.
            if use.sum() < 2 * MIN_SIDE + 1:
                continue
            score, bp = limb_score(u=u_day[use], e=e_all[use])
        else:
            morning = use & (h.hod[sel] + 0.5 < h.noon_hod)
            afternoon = use & (h.hod[sel] + 0.5 >= h.noon_hod)
            if morning.sum() < 2 * MIN_SIDE or afternoon.sum() < 2 * MIN_SIDE:
                continue
            s_m, bp_m = limb_score(u=u_day[morning], e=e_all[morning])
            s_a, bp_a = limb_score(u=u_day[afternoon], e=e_all[afternoon])
            score = min(s_m, s_a)
            bp = (
                float(np.nanmean([bp_m, bp_a]))
                if np.isfinite(bp_m) and np.isfinite(bp_a)
                else float("nan")
            )
        midday = (h.hod[sel] >= 10) & (h.hod[sel] < 14) & np.isfinite(e_all)
        days.append(int(d))
        scores.append(score)
        bps.append(bp)
        broken_flags.append(broken)
        levels.append(float(e_all[midday].mean()) if midday.any() else float("nan"))
    return DayResult(
        np.array(days),
        np.array(scores),
        np.array(bps),
        np.array(levels),
        np.array(broken_flags, dtype=bool),
    )


# --- the known-answer run ---------------------------------------------------------------------

SETTINGS: Final[dict[str, inputs.Setting]] = {
    "L2_Q2": inputs.Setting(2.0, 2.0),
    "L2_Q4": inputs.Setting(2.0, 4.0),
    "L1_Q2": inputs.Setting(1.0, 2.0),
    "L2_PF0.97": inputs.Setting(2.0, 2.0, 0.97),
    "L2_CAP0.7": inputs.Setting(2.0, 2.0, 1.0, 0.7),
    "L0_Q2": inputs.Setting(0.0, 2.0),
    "L0_Q4": inputs.Setting(0.0, 4.0),
    "L0_PF0.97": inputs.Setting(0.0, 2.0, 0.97),
}
PLACEBO_OF: Final[dict[str, str]] = {
    "L2_Q2": "L0_Q2",
    "L2_Q4": "L0_Q4",
    "L1_Q2": "L0_Q2",
    "L2_PF0.97": "L0_PF0.97",
    "L2_CAP0.7": "L0_Q2",
    "L2_Q2_shuffle": "L0_Q2",
}


def auc(*, positive: np.ndarray, negative: np.ndarray) -> float:
    """Return the area under the ROC curve (ties count half)."""
    return diag.auc(
        positive=positive[np.isfinite(positive)], negative=negative[np.isfinite(negative)]
    )


def run() -> dict:  # noqa: PLR0915
    """Score the days of every setting, and read the bars."""
    sites = inputs.SIGNED_SIX
    tables = {
        (n, s): diag.load_table(site=s, setting=st) for n, st in SETTINGS.items() for s in sites
    }
    results: dict = {}
    hours = {key: hourly(t=t) for key, t in tables.items()}
    profiles = {key: build_profile(h=h) for key, h in hours.items()}
    days = {key: score_days(h=hours[key], profile=profiles[key]) for key in tables}
    # The shuffled-sky arm: u permuted across days within (year, month, half-hour) of each Primary.
    for s in sites:
        t = tables[("L2_Q2", s)]
        u_shuffled = inputs.shuffled_u(site=s, valid_us=t.start_us + inputs.US_PER_HALF_HOUR)
        hs = hourly(t=t, u=np.nan_to_num(u_shuffled).astype(np.float64))
        hours[("L2_Q2_shuffle", s)] = hs
        profiles[("L2_Q2_shuffle", s)] = build_profile(h=hs)
        days[("L2_Q2_shuffle", s)] = score_days(h=hs, profile=profiles[("L2_Q2_shuffle", s)])
        tables[("L2_Q2_shuffle", s)] = t
    for name, placebo_name in PLACEBO_OF.items():
        entry: dict = {"per_primary": {}}
        pos_scores, neg_scores, pos_level, neg_level = [], [], [], []
        paired_inj, paired_pla = [], []
        reg_pos: dict[str, list] = {"smooth": [], "broken": []}
        reg_neg: dict[str, list] = {"smooth": [], "broken": []}
        recall_num = recall_den = total_positive = 0
        for s in sites:
            d_inj, d_pla = days[(name, s)], days[(placebo_name, s)]
            t = tables[(name, s)]
            # A day is positive if at least 2 of its half-hours have signed P' < 0.
            start_day = t.start_us // inputs.US_PER_DAY
            neg_count = {
                int(d): int(c)
                for d, c in pl.DataFrame({"d": start_day[t.truth < 0]})
                .group_by("d")
                .len()
                .iter_rows()
            }
            positive = np.array([neg_count.get(int(d), 0) >= 2 for d in d_inj.day])
            # Leave one Primary out: the threshold is the 98th percentile of the others' placebo scores.
            thr = {
                b: float(
                    np.percentile(
                        np.concatenate(
                            [
                                days[(placebo_name, o)].score[days[(placebo_name, o)].broken == b]
                                for o in sites
                                if o != s
                            ]
                        ),
                        THRESHOLD_PERCENTILE,
                    )
                )
                for b in (False, True)
            }
            flagged_pla = d_pla.score > np.where(d_pla.broken, thr[True], thr[False])
            flagged_inj = d_inj.score > np.where(d_inj.broken, thr[True], thr[False])
            common_days = np.intersect1d(d_inj.day, d_pla.day)
            paired_inj.append(d_inj.score[np.isin(d_inj.day, common_days)])
            paired_pla.append(d_pla.score[np.isin(d_pla.day, common_days)])
            reg_pos["smooth"].append(d_inj.score[positive & ~d_inj.broken])
            reg_neg["smooth"].append(d_inj.score[~positive & ~d_inj.broken])
            reg_pos["broken"].append(d_inj.score[positive & d_inj.broken])
            reg_neg["broken"].append(d_inj.score[~positive & d_inj.broken])
            pos_scores.append(d_inj.score[positive])
            neg_scores.append(d_inj.score[~positive])
            pos_level.append(d_inj.level[positive])
            neg_level.append(d_inj.level[~positive])
            total_positive += int(sum(1 for c in neg_count.values() if c >= 2))
            recall_num += int((flagged_inj & positive).sum())
            recall_den += int(positive.sum())
            entry["per_primary"][s] = {
                "false_flag_placebo_smooth": float(flagged_pla[~d_pla.broken].mean())
                if (~d_pla.broken).any()
                else float("nan"),
                "false_flag_placebo_broken": float(flagged_pla[d_pla.broken].mean())
                if d_pla.broken.any()
                else float("nan"),
                "n_broken_injected": int(d_inj.broken.sum()),
                "n_eligible_injected": len(d_inj.day),
                "n_positive": int(positive.sum()),
                "recall_at_threshold": float(flagged_inj[positive].mean())
                if positive.any()
                else float("nan"),
                "false_flag_placebo": float(flagged_pla.mean())
                if len(flagged_pla)
                else float("nan"),
                "n_eligible_placebo": len(d_pla.day),
                "threshold_smooth": thr[False],
                "threshold_broken": thr[True],
                "paired_auc": auc(
                    positive=d_inj.score[np.isin(d_inj.day, common_days)],
                    negative=d_pla.score[np.isin(d_pla.day, common_days)],
                ),
            }
        entry["auc_positive_vs_negative_days"] = auc(
            positive=np.concatenate(pos_scores), negative=np.concatenate(neg_scores)
        )
        for regime in ("smooth", "broken"):
            entry[f"auc_{regime}_days"] = auc(
                positive=np.concatenate(reg_pos[regime]), negative=np.concatenate(reg_neg[regime])
            )
            entry[f"n_positive_{regime}"] = int(sum(len(x) for x in reg_pos[regime]))
        entry["auc_level_baseline"] = auc(
            positive=np.concatenate(pos_level), negative=np.concatenate(neg_level)
        )
        entry["paired_auc_pooled"] = auc(
            positive=np.concatenate(paired_inj), negative=np.concatenate(paired_pla)
        )
        entry["pooled_recall"] = recall_num / max(recall_den, 1)
        entry["recall_all_positive"] = recall_num / max(total_positive, 1)
        entry["n_positive_total"] = total_positive
        entry["auc_per_primary"] = {
            s: auc(positive=pos_scores[i], negative=neg_scores[i]) for i, s in enumerate(sites)
        }
        entry["n_positive_days"] = recall_den
        results[name] = entry
    (diag.OUT_DIR / "day_flip.json").write_text(json.dumps(results, indent=1))
    return results


def main() -> None:
    """Run and print the headline numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run",))
    parser.parse_args()
    for name, e in run().items():
        print(
            f"\n== {name}: AUC {e['auc_positive_vs_negative_days']:.3f} (level baseline "
            f"{e['auc_level_baseline']:.3f}); paired AUC {e['paired_auc_pooled']:.3f}; recall "
            f"{e['pooled_recall']:.2f} of {e['n_positive_days']} positive days; AUC smooth "
            f"{e['auc_smooth_days']:.3f} (n_pos {e['n_positive_smooth']}), broken "
            f"{e['auc_broken_days']:.3f} (n_pos {e['n_positive_broken']})"
        )
        for s, p in e["per_primary"].items():
            print(
                f"  {s}: recall {p['recall_at_threshold']:.2f} (n_pos {p['n_positive']}), false flag "
                f"{p['false_flag_placebo']:.3f} (smooth {p['false_flag_placebo_smooth']:.3f}, broken "
                f"{p['false_flag_placebo_broken']:.3f}; n {p['n_eligible_placebo']}, broken "
                f"{p['n_broken_injected']} of {p['n_eligible_injected']} injected), paired AUC {p['paired_auc']:.2f}"
            )


if __name__ == "__main__":
    main()

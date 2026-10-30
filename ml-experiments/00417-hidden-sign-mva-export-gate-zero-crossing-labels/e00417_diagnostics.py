# ruff: noqa: E501
"""Zero-fit diagnostics for designs C, D, and E, and the zero-crossing labeller of design C.

No network is trained. The folded readings of the six signed-MW Primaries (experiment 412's
harness: real signed power plus the injected farm, folded as an MVA meter reads it) are examined
before 2025-07-01, and the area under the receiver operating characteristic curve measures how well
each statistic separates injected from no-injection days, pairs, or Primary-months.

- **C, W-shape score** (per clear day): the smaller of (centre maximum - left minimum) and (centre
  maximum - right minimum) of the reading around solar noon `t0`, with the minima in `t0 -+ 5 h` to
  `t0 -+ 1 h` and the maximum in `t0 +- 1 h`. A clear day has `k >= 0.7` throughout `t0 +- 5 h`.
- **D, Sunday-and-holiday contrast** (per clear Sunday or bank holiday): the midday reading minus
  the mean midday reading of clear Tuesdays to Thursdays within 14 days, with a per-Primary linear
  ERA5-temperature term removed.
- **E, sunny-versus-cloudy slope** (per Primary-month): slope of the midday reading on `k` after
  removing a mean for each half-hour of the day and ISO week, with ERA5 temperature as covariate.

```bash
. env.sh   # PYTHONPATH and DATA_PATH_INTERNAL, as e00417_fit.py shows
uv run --no-sync python ml-experiments/00417-*/e00417_diagnostics.py run
```
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00405_prep as prep405  # ty: ignore[unresolved-import]
import e00417_inputs as inputs
import numpy as np
import polars as pl

OUT_DIR: Final[Path] = inputs.WORK_DIR / "diagnostics"
WINDOW_END_US: Final[int] = int(datetime(2025, 7, 1, tzinfo=UTC).timestamp() * 1_000_000)
VISIBLE_FOR_R1: Final[tuple[str, ...]] = ("D9", "D11", "D14", "D16")
PLACEBO_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D11", "D14", "D16")
SETTINGS: Final[dict[str, inputs.Setting]] = {
    "L2": inputs.Setting(2.0, 2.0),
    "L1": inputs.Setting(1.0, 2.0),
    "L2_PF0.97": inputs.Setting(2.0, 2.0, 0.97),
    "L0": inputs.Setting(0.0, 2.0),
    "L0_PF0.97": inputs.Setting(0.0, 2.0, 0.97),
}
PLACEBO_OF: Final[dict[str, str]] = {
    "L2": "L0",
    "L1": "L0",
    "L2_PF0.97": "L0_PF0.97",
}
HALF_HOURS_PER_SIDE: Final[int] = 10
"""`t0 -+ 5 h` is 10 half-hours either side."""
CLEAR_K: Final[float] = 0.7
MIN_REFERENCE_DAYS: Final[int] = 3
N_BOOTSTRAP: Final[int] = 500
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)


@dataclass
class Table:
    """One Primary's unique half-hours before 2025-07-01 at one setting."""

    site: str
    start_us: np.ndarray
    r: np.ndarray
    truth: np.ndarray
    u: np.ndarray
    k: np.ndarray
    temp: np.ndarray
    dow: np.ndarray
    bank_holiday: np.ndarray
    noon_utc_hour: float


def solar_noon_hour(*, site: str) -> float:
    """Return the UTC hour of solar noon (longitude only; the equation of time is under 16 minutes)."""
    row = next(r for r in common.series_table().iter_rows(named=True) if r["site"] == site)
    return 12.0 - float(row["longitude"]) / 15.0


def load_table(
    *, site: str, setting: inputs.Setting, start_us: int = 0, end_us: int = WINDOW_END_US
) -> Table:
    """Return the Primary's unique half-hours in `[start_us, end_us)` with the reading in points."""
    arr = inputs.site_arrays(site=site, fold=True, setting=setting)
    keep = (
        (arr.valid >= start_us)
        & (arr.valid < end_us)
        & np.isfinite(arr.y)
        & np.isfinite(arr.y_truth)
    )
    valid, first = np.unique(arr.valid[keep], return_index=True)
    index = np.flatnonzero(keep)[first]
    x = arr.x[index]
    u, k = inputs.cams_for_rows(site=site, valid_us=valid)
    dow = (
        np.argmax(
            np.stack([x[:, prep405.col(name=f"dow_{d}")] for d in range(1, 8)], axis=1), axis=1
        )
        + 1
    )
    return Table(
        site=site,
        start_us=valid - inputs.US_PER_HALF_HOUR,
        r=arr.y[index].astype(np.float64) * 100.0,
        truth=arr.y_truth[index].astype(np.float64) * 100.0,
        u=np.nan_to_num(u).astype(np.float64),
        k=np.nan_to_num(k).astype(np.float64),
        temp=inputs.era5_for_rows(site=site, valid_us=valid).astype(np.float64),
        dow=dow,
        bank_holiday=x[:, prep405.col(name="bank_holiday")] > 0.5,
        noon_utc_hour=solar_noon_hour(site=site),
    )


# --- C: the W-shape score ---------------------------------------------------------------------


@dataclass
class DayScores:
    """The W-shape score of each clear day, and where the two minima fall."""

    day: np.ndarray
    score: np.ndarray
    left: np.ndarray
    right: np.ndarray
    """Table indices of the left and right minima."""
    noon_index: np.ndarray


def clear_days(*, t: Table) -> DayScores:
    """Return the W-shape score of every clear day of the table."""
    day_us = t.start_us // inputs.US_PER_DAY
    unique_days = np.unique(day_us).tolist()
    days, scores, lefts, rights, noons = [], [], [], [], []
    start_of_noon = round(t.noon_utc_hour * 2.0)
    for d in unique_days:
        base_us = d * inputs.US_PER_DAY
        want = (
            base_us
            + (start_of_noon + np.arange(-HALF_HOURS_PER_SIDE, HALF_HOURS_PER_SIDE + 1))
            * inputs.US_PER_HALF_HOUR
        )
        pos = np.searchsorted(t.start_us, want)
        pos = np.clip(pos, 0, len(t.start_us) - 1)
        ok = t.start_us[pos] == want
        if not ok.all():
            continue
        if t.k[pos].min() < CLEAR_K:
            continue
        r = t.r[pos]
        centre = HALF_HOURS_PER_SIDE
        left_slice = slice(0, centre - 1)
        right_slice = slice(centre + 2, 2 * centre + 1)
        centre_slice = slice(centre - 1, centre + 2)
        li = int(np.argmin(r[left_slice]))
        ri = int(np.argmin(r[right_slice])) + centre + 2
        cmax = float(r[centre_slice].max())
        score = min(cmax - r[li], cmax - r[ri])
        days.append(d)
        scores.append(score)
        lefts.append(int(pos[li]))
        rights.append(int(pos[ri]))
        noons.append(int(pos[centre]))
    return DayScores(
        day=np.array(days),
        score=np.array(scores),
        left=np.array(lefts),
        right=np.array(rights),
        noon_index=np.array(noons),
    )


def label_reverse(*, t: Table, scores: DayScores, threshold: float) -> np.ndarray:
    """Return a boolean per table row: reverse flow between the two minima on W days."""
    label = np.zeros(len(t.r), dtype=bool)
    for left, right, score in zip(scores.left, scores.right, scores.score, strict=True):
        if score > threshold:
            label[left : right + 1] = True
    return label


# --- areas under the curve --------------------------------------------------------------------


def auc(*, positive: np.ndarray, negative: np.ndarray) -> float:
    """Return P(positive > negative) + half the ties, the area under the ROC curve."""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    values = np.concatenate([positive, negative])
    ranks = pl.Series(values).rank("average").to_numpy()
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pooled_auc_with_interval(
    *, injected: dict[str, np.ndarray], placebo: dict[str, np.ndarray], seed: int = 417
) -> dict[str, float | int]:
    """Return the pooled area and a 95% interval from resampling units within each Primary."""
    pos = np.concatenate(list(injected.values())) if injected else np.array([])
    neg = np.concatenate(list(placebo.values())) if placebo else np.array([])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOTSTRAP):
        p = [v[rng.integers(0, len(v), len(v))] for v in injected.values() if len(v)]
        n = [v[rng.integers(0, len(v), len(v))] for v in placebo.values() if len(v)]
        if p and n:
            draws.append(auc(positive=np.concatenate(p), negative=np.concatenate(n)))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5]) if draws else (float("nan"),) * 2
    return {
        "auc": auc(positive=pos, negative=neg),
        "lo": float(lo),
        "hi": float(hi),
        "n_injected": len(pos),
        "n_placebo": len(neg),
    }


# --- D and E ----------------------------------------------------------------------------------


def midday_mask(*, t: Table, lo: float = 10.0, hi: float = 14.0) -> np.ndarray:
    """Return rows whose start is in the UTC hours [lo, hi)."""
    hour = ((t.start_us % inputs.US_PER_DAY) / inputs.US_PER_HALF_HOUR) / 2.0
    return (hour >= lo) & (hour < hi)


def day_means(*, t: Table, mask: np.ndarray, values: np.ndarray) -> dict[int, float]:
    """Return the mean of `values` over the masked rows of each day."""
    day = t.start_us // inputs.US_PER_DAY
    frame = (
        pl.DataFrame({"d": day[mask], "v": values[mask]})
        .group_by("d")
        .agg(pl.col("v").mean(), pl.len().alias("n"))
    )
    return {int(d): float(v) for d, v, n in frame.iter_rows() if n >= 6}


def sunday_contrast(*, t: Table) -> dict[str, np.ndarray]:
    """Return the temperature-adjusted Sunday-or-holiday minus weekday midday contrast per pair."""
    mid = midday_mask(t=t)
    day = t.start_us // inputs.US_PER_DAY
    clear_mid = pl.DataFrame({"d": day[mid], "k": t.k[mid]}).group_by("d").agg(pl.col("k").min())
    clear = {int(d) for d, k in clear_mid.iter_rows() if k >= CLEAR_K}
    r_day = day_means(t=t, mask=mid, values=t.r)
    t_day = day_means(t=t, mask=mid, values=np.nan_to_num(t.temp, nan=np.nanmean(t.temp)))
    night = midday_mask(t=t, lo=0.0, hi=4.0)
    n_day = day_means(t=t, mask=night, values=t.r)
    dow_of_day = {int(d): int(w) for d, w in zip(day[mid], t.dow[mid], strict=True)}
    holiday_of_day = {int(d): bool(h) for d, h in zip(day[mid], t.bank_holiday[mid], strict=True)}
    weekdays = [
        d
        for d in clear
        if d in r_day and d in t_day and dow_of_day.get(d) in (2, 3, 4) and not holiday_of_day[d]
    ]
    if len(weekdays) < MIN_REFERENCE_DAYS:
        return {"diff": np.array([]), "night": np.array([])}
    slope = float(np.polyfit([t_day[d] for d in weekdays], [r_day[d] for d in weekdays], 1)[0])
    diffs, nights = [], []
    for d in sorted(clear):
        if d not in r_day or d not in t_day:
            continue
        if not (dow_of_day.get(d) == 7 or holiday_of_day.get(d, False)):
            continue
        refs = [w for w in weekdays if abs(w - d) <= 14]
        if len(refs) < MIN_REFERENCE_DAYS:
            continue
        r_ref = float(np.mean([r_day[w] for w in refs]))
        t_ref = float(np.mean([t_day[w] for w in refs]))
        diffs.append(r_day[d] - r_ref - slope * (t_day[d] - t_ref))
        night_refs = [n_day[w] for w in refs if w in n_day]
        if d in n_day and night_refs:
            nights.append(n_day[d] - float(np.mean(night_refs)))
    return {"diff": np.array(diffs), "night": np.array(nights)}


def cloud_slopes(*, t: Table) -> dict[tuple[int, int], float]:
    """Return the slope of the midday reading on `k` for each (year, month) of the table."""
    mid = midday_mask(t=t)
    start = pl.Series(t.start_us).cast(pl.Datetime("us"))
    year, month = start.dt.year().to_numpy(), start.dt.month().to_numpy()
    week = start.dt.week().to_numpy()
    halfhour = ((t.start_us % inputs.US_PER_DAY) // inputs.US_PER_HALF_HOUR).astype(int)
    temp = np.nan_to_num(t.temp, nan=float(np.nanmean(t.temp)))
    out = {}
    for y in np.unique(year):
        for m in SUMMER_MONTHS:
            sel = mid & (year == y) & (month == m)
            if sel.sum() < 200:
                continue
            key = week[sel] * 100 + halfhour[sel]
            frame = pl.DataFrame({"g": key, "r": t.r[sel], "k": t.k[sel], "t": temp[sel]})
            frame = frame.with_columns(
                *[(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in ("r", "k", "t")]
            )
            design = frame.select("k", "t").to_numpy()
            coef = np.linalg.lstsq(design, frame["r"].to_numpy(), rcond=None)[0]
            out[(int(y), m)] = float(coef[0])
    return out


# --- the run ----------------------------------------------------------------------------------


def run() -> dict:
    """Compute every diagnostic and write the table of areas."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        (name, site): load_table(site=site, setting=setting)
        for name, setting in SETTINGS.items()
        for site in inputs.SIGNED_SIX
    }
    results: dict = {}
    scores = {key: clear_days(t=t) for key, t in tables.items()}
    for name, placebo_name in PLACEBO_OF.items():
        entry: dict = {}
        # Threshold of the labeller: 99th percentile over the no-injection days of the visible four.
        visible = np.concatenate([scores[(placebo_name, s)].score for s in VISIBLE_FOR_R1])
        threshold = float(np.percentile(visible, 99))
        entry["c_threshold_points"] = threshold
        entry["c_n_visible_days"] = len(visible)
        c_inj = {s: scores[(name, s)].score for s in inputs.SIGNED_SIX}
        c_pla = {s: scores[(placebo_name, s)].score for s in PLACEBO_SITES}
        entry["C"] = {
            "pooled": pooled_auc_with_interval(injected=c_inj, placebo=c_pla),
            "per_primary": {
                s: pooled_auc_with_interval(
                    injected={s: c_inj[s]}, placebo={s: scores[(placebo_name, s)].score}
                )
                for s in inputs.SIGNED_SIX
            },
        }
        d_inj = {s: sunday_contrast(t=tables[(name, s)]) for s in inputs.SIGNED_SIX}
        d_pla = {s: sunday_contrast(t=tables[(placebo_name, s)]) for s in PLACEBO_SITES}
        entry["D"] = {
            "pooled": pooled_auc_with_interval(
                injected={s: v["diff"] for s, v in d_inj.items()},
                placebo={s: v["diff"] for s, v in d_pla.items()},
            ),
            "night_positive_share_injected": {
                s: float(np.mean(v["night"] > 0)) if len(v["night"]) else float("nan")
                for s, v in d_inj.items()
            },
            "midday_positive_share": {
                "injected": {
                    s: float(np.mean(v["diff"] > 0)) if len(v["diff"]) else float("nan")
                    for s, v in d_inj.items()
                },
                "placebo": {
                    s: float(np.mean(v["diff"] > 0)) if len(v["diff"]) else float("nan")
                    for s, v in d_pla.items()
                },
            },
            "per_primary": {
                s: pooled_auc_with_interval(
                    injected={s: d_inj[s]["diff"]},
                    placebo={s: sunday_contrast(t=tables[(placebo_name, s)])["diff"]},
                )
                for s in inputs.SIGNED_SIX
            },
        }
        e_inj = {
            s: np.array(list(cloud_slopes(t=tables[(name, s)]).values())) for s in inputs.SIGNED_SIX
        }
        e_pla = {
            s: np.array(list(cloud_slopes(t=tables[(placebo_name, s)]).values()))
            for s in PLACEBO_SITES
        }
        entry["E"] = {
            "pooled": pooled_auc_with_interval(injected=e_inj, placebo=e_pla),
            "mean_slope_injected": {s: float(v.mean()) for s, v in e_inj.items()},
            "mean_slope_placebo": {s: float(v.mean()) for s, v in e_pla.items()},
            "per_primary": {
                s: pooled_auc_with_interval(
                    injected={s: e_inj[s]},
                    placebo={s: np.array(list(cloud_slopes(t=tables[(placebo_name, s)]).values()))},
                )
                for s in inputs.SIGNED_SIX
            },
        }
        # The labeller at its operating threshold, at the r1 pair.
        label = {}
        for s in ("D5", "D10"):
            for tag, nm in (("injected", name), ("placebo", placebo_name)):
                t = tables[(nm, s)]
                sc = scores[(nm, s)]
                mask = label_reverse(t=t, scores=sc, threshold=threshold)
                on_clear = np.zeros(len(t.r), dtype=bool)
                for lo, hi in zip(sc.left, sc.right, strict=True):
                    on_clear[max(lo - 6, 0) : hi + 7] = True
                truth = t.truth < 0
                label[f"{s}_{tag}"] = {
                    "day_flag_rate": float(np.mean(sc.score > threshold)),
                    "n_clear_days": len(sc.score),
                    "recall": float(mask[truth & on_clear].mean())
                    if (truth & on_clear).any()
                    else float("nan"),
                    "n_reverse_half_hours": int((truth & on_clear).sum()),
                    "false_reverse": float(mask[~truth & on_clear].mean()),
                }
        entry["C_labeller"] = label
        results[name] = entry
    # CAMS against ERA5 shortwave, a cross-check of the CAMS arrays.
    cross = {}
    for s in inputs.SIGNED_SIX:
        t = tables[("L0", s)]
        sw = inputs.era5_for_rows(
            site=s, valid_us=t.start_us + inputs.US_PER_HALF_HOUR, name="shortwave"
        )
        ok = (t.u > 0.01) & np.isfinite(sw)
        cross[s] = float(np.corrcoef(t.u[ok], sw[ok])[0, 1])
    results["cams_vs_era5_shortwave_correlation"] = cross
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(results, indent=1))
    return results


def main() -> None:
    """Run the diagnostics and print the pooled areas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run",))
    parser.parse_args()
    results = run()
    for name, entry in results.items():
        if name.startswith("cams"):
            print(name, {k: round(v, 3) for k, v in entry.items()})
            continue
        print(f"\n== {name}: labeller threshold {entry['c_threshold_points']:.1f} points")
        for design in ("C", "D", "E"):
            p = entry[design]["pooled"]
            per = {s: round(v["auc"], 2) for s, v in entry[design]["per_primary"].items()}
            print(
                f"{design}: AUC {p['auc']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}] "
                f"(n {p['n_injected']} vs {p['n_placebo']}); per Primary {per}"
            )
        for k, v in entry["C_labeller"].items():
            print(f"  labeller {k}: {v}")


if __name__ == "__main__":
    main()

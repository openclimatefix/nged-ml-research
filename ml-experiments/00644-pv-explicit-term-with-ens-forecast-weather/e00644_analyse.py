# ruff: noqa: E501, ANN001, PERF401
"""Experiment 644: score every arm and print the planned contrasts.

CRPS, the day-block bootstrap, and the seed-split null are experiment 639's (`e00639_analyse`).
A contrast is the control's CRPS minus the explicit network's (`b`), so a positive value favours `b`.
Arms: `a` free; `b` explicit (calibrated); `c` explicit (raw); `f` explicit (R4 table); `p` capacity
placebo; `s` shuffled-day placebo; `d` the 14-day tracker (arm `a` shifted); `e` arm `a` shifted by a
constant fitted on held-out fine-tuning rows.

Usage: python e00644_analyse.py contrasts [--seeds 0 1 2 3 4] [--tag stage1]
"""

import argparse
import itertools
import json
from typing import Final

import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00644_fit as fit
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final = fit.WORK / "analysis"
BANDS: Final[dict[str, tuple[int, int]]] = {"d1-2": (1, 2), "d3-5": (3, 5), "d6-10": (6, 10)}
TRAIL_DAYS: Final = 14
MIN_HISTORY_DAYS: Final = 3
DAY_US: Final = 86_400_000_000
N_CONTRASTS: Final = 19
LEVEL: Final = 1.0 - a639.ALPHA / N_CONTRASTS
FIT_ARMS: Final = ("a", "b", "c", "f", "p", "s")


def realised_kt() -> pl.DataFrame:
    """Return experiment 639's realised clear-sky index by Primary and valid time (the same in both worlds)."""
    parts = []
    for cutoff in k639.CUTOFFS:
        z = np.load(k639.test_rows_path(cutoff=cutoff, world="real"))
        parts.append(pl.DataFrame({"primary": z["primary"], "t_us": z["t_us"], "kt": z["kt"]}))
    return pl.concat(parts).unique(subset=["primary", "t_us"])


def load(*, world: str) -> dict[str, np.ndarray]:
    """Return the test rows of every cutoff's months, the six Primaries, with masks and keys."""
    kt = realised_kt()
    parts = []
    for cutoff in fit.CUTOFFS:
        for month in fit.MONTHS[cutoff]:
            z = np.load(fit.rows_path(month=month, world=world))
            n = len(z["y"])
            parts.append(
                pl.DataFrame(
                    {
                        "cutoff": np.full(n, fit.CUTOFFS.index(cutoff)), "primary": z["primary"],
                        "y": z["y"], "t_us": z["t_us"], "origin_us": z["origin_us"], "lead": z["lead"],
                        "kt_forecast": z["kt_forecast"], "row_in_file": np.arange(n),
                        "file": np.full(n, len(parts)),
                    }
                )
            )
    frame = pl.concat(parts).join(kt, on=["primary", "t_us"], how="left")
    hour = ((frame["t_us"] // 3_600_000_000) % 24).to_numpy()
    out = {n: frame[n].to_numpy() for n in ("cutoff", "primary", "y", "t_us", "origin_us", "lead", "kt_forecast", "file", "row_in_file")}
    out["kt"] = frame["kt"].fill_null(-1.0).to_numpy()
    out["hour"] = hour
    out["midday"] = (hour >= k639.SUNNY_HOURS[0]) & (hour <= k639.SUNNY_HOURS[1])
    out["sunny"] = (out["kt"] >= k639.SUNNY_KT) & out["midday"]
    out["sunny_forecast"] = (out["kt_forecast"] >= k639.SUNNY_KT) & out["midday"]
    out["lead_day"] = (out["lead"] // 24).astype(int) + 1
    out["day"] = out["cutoff"] * 100_000 + out["t_us"] // DAY_US % 100_000
    out["valid_date"] = out["t_us"] // DAY_US
    out["origin_date"] = out["origin_us"] // DAY_US
    out["files"] = np.array([(c, m) for c in fit.CUTOFFS for m in fit.MONTHS[c]])
    return out


def _quantiles(*, world: str, arm: str, seed: int, rows: dict) -> np.ndarray:
    """Return the stored quantiles (rows, 11) of a fit arm in the order of `rows`."""
    out = np.empty((len(rows["y"]), 11), dtype=np.float32)
    for i, (cutoff, month) in enumerate(rows["files"]):
        sel = rows["file"] == i
        path = fit.fits_path(cutoff=cutoff, month=month, world=world, arm=arm, seed=seed).with_suffix(".npz")
        out[sel] = np.load(path)["q"]
    return out


def _tracker(*, q: np.ndarray, rows: dict) -> np.ndarray:
    """Arm `d`: arm `a` plus the trailing 14-day mean error of its own day-1 forecasts.

    The history of a row with origin date O holds the sunny-midday half-hours of dates O-14 to O-1
    whose lead is below 24 hours (the last run issued before each half-hour), per Primary. Dates
    with fewer than `MIN_HISTORY_DAYS` days of history get no shift. Applied on sunny middays.
    """
    out = q.copy()
    hist = rows["sunny"] & (rows["lead"] < 24)
    err = q[:, 5] - rows["y"]
    for p in np.unique(rows["primary"]):
        m = rows["primary"] == p
        h = m & hist
        dates, inv = np.unique(rows["valid_date"][h], return_inverse=True)
        s, n = np.bincount(inv, weights=err[h]), np.bincount(inv).astype(float)
        target = m & rows["sunny"]
        for od in np.unique(rows["origin_date"][target]):
            win = (dates >= od - TRAIL_DAYS) & (dates < od)
            if win.sum() >= MIN_HISTORY_DAYS:
                out[target & (rows["origin_date"] == od)] -= s[win].sum() / n[win].sum()
    return out


def _shifted(*, world: str, seed: int, rows: dict, q: np.ndarray) -> np.ndarray:
    """Arm `e`: arm `a` minus the per-Primary constant stored by the fit, on sunny middays."""
    shift = np.zeros(len(rows["y"]))
    for i, (cutoff, month) in enumerate(rows["files"]):
        path = fit.fits_path(cutoff=cutoff, month=month, world=world, arm="a", seed=seed).with_suffix(".npz")
        sel = rows["file"] == i
        shift[sel] = np.load(path)["shift"][rows["primary"][sel]]
    out = q.copy()
    out[rows["sunny"]] -= shift[rows["sunny"]][:, None]
    return out


def arm_quantiles(*, world: str, arm: str, seed: int, rows: dict) -> np.ndarray:
    """Return the quantiles of any arm, `d` and `e` included."""
    if arm in FIT_ARMS:
        return _quantiles(world=world, arm=arm, seed=seed, rows=rows)
    base = _quantiles(world=world, arm="a", seed=seed, rows=rows)
    return _tracker(q=base, rows=rows) if arm == "d" else _shifted(world=world, seed=seed, rows=rows, q=base)


def arm_crps(*, world: str, arm: str, rows: dict, seeds: tuple[int, ...]) -> np.ndarray:
    """Return CRPS (points of capacity) per seed and row: shape (seeds, rows)."""
    return np.stack(
        [100.0 * a639.crps(actual=rows["y"].astype(np.float64), q=arm_quantiles(world=world, arm=arm, seed=s, rows=rows).astype(np.float64)) for s in seeds]
    )


def _masks(*, rows: dict) -> dict[str, np.ndarray]:
    """Return the scoring slices: sunny middays by lead band, and all half-hours."""
    out = {"all": np.ones(len(rows["y"]), dtype=bool)}
    for band, (lo, hi) in BANDS.items():
        inb = (rows["lead_day"] >= lo) & (rows["lead_day"] <= hi)
        out[f"sunny_{band}"] = rows["sunny"] & inb
        out[f"allhh_{band}"] = inb
    out["sunny_all"] = rows["sunny"]
    out["midday_all"] = rows["midday"]
    out["sunny_fc"] = rows["sunny_forecast"]
    return out


def _contrast(*, rows, control: np.ndarray, expl: np.ndarray, mask: np.ndarray) -> dict:
    diff = (control - expl).mean(axis=0)
    mean, lo, hi = a639.bootstrap(diff=diff[mask], day=rows["day"][mask], level=LEVEL)
    per_seed = (control - expl)[:, mask].mean(axis=1)
    return {"diff": mean, "lo": lo, "hi": hi, "n_rows": int(mask.sum()), "per_seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0}


def null_bar(*, scores_a: np.ndarray, rows: dict, mask: np.ndarray) -> float:
    """Return the mean half-width of the seed-split null on arm `a` (experiment 639's rule) for a slice."""
    n = scores_a.shape[0]
    if n < 4:
        return float("nan")
    widths = []
    for chosen in itertools.combinations(range(n), n // 2 + 1):
        rest = [s for s in range(n) if s not in chosen]
        diff = scores_a[list(chosen)].mean(axis=0) - scores_a[rest].mean(axis=0)
        _, lo, hi = a639.bootstrap(diff=diff[mask], day=rows["day"][mask], level=LEVEL)
        widths.append((hi - lo) / 2)
    return float(np.mean(widths))


def contrasts(*, seeds: tuple[int, ...]) -> dict:
    """Return every planned contrast, with leave-one-Primary-out, for both worlds."""
    out: dict = {"seeds": list(seeds), "level": LEVEL, "n_contrasts": N_CONTRASTS}
    for world in k639.WORLDS:
        rows = load(world=world)
        masks = _masks(rows=rows)
        arms = ("a", "b", "c", "d", "e", "f", "p", "s")
        scores = {arm: arm_crps(world=world, arm=arm, rows=rows, seeds=seeds) for arm in arms}
        res: dict = {"scores": {}, "contrasts": {}, "bars": {}}
        for name, mask in masks.items():
            res["scores"][name] = {arm: float(scores[arm][:, mask].mean()) for arm in arms}
            res["bars"][name] = null_bar(scores_a=scores["a"], rows=rows, mask=mask) if name.startswith("sunny_d") else None
        for band in BANDS:
            name = f"sunny_{band}"
            best = np.where(
                scores["d"][:, masks[name]].mean() <= scores["e"][:, masks[name]].mean(), 0, 1
            )
            ctrl_best = scores["d"] if best == 0 else scores["e"]
            res["contrasts"][name] = {
                "vs_a": _contrast(rows=rows, control=scores["a"], expl=scores["b"], mask=masks[name]),
                "vs_best_de": {**_contrast(rows=rows, control=ctrl_best, expl=scores["b"], mask=masks[name]), "control": "d" if best == 0 else "e"},
                "vs_d": _contrast(rows=rows, control=scores["d"], expl=scores["b"], mask=masks[name]),
                "vs_e": _contrast(rows=rows, control=scores["e"], expl=scores["b"], mask=masks[name]),
                "placebo_p": _contrast(rows=rows, control=scores["p"], expl=scores["b"], mask=masks[name]),
                "placebo_s": _contrast(rows=rows, control=scores["s"], expl=scores["b"], mask=masks[name]),
                "lopo": {},
            }
            for p in np.unique(rows["primary"]):
                keep = masks[name] & (rows["primary"] != p)
                res["contrasts"][name]["lopo"][int(p)] = _contrast(rows=rows, control=scores["a"], expl=scores["b"], mask=keep)["diff"]
        res["contrasts"]["all_half_hours"] = {"vs_a": _contrast(rows=rows, control=scores["a"], expl=scores["b"], mask=masks["all"])}
        res["by_lead_day"] = {
            str(dday): {arm: float(scores[arm][:, rows["sunny"] & (rows["lead_day"] == dday)].mean()) for arm in arms}
            for dday in range(1, 11)
            if (rows["sunny"] & (rows["lead_day"] == dday)).any()
        }
        q50 = {arm: np.stack([arm_quantiles(world=world, arm=arm, seed=s, rows=rows)[:, 5] for s in seeds]).mean(axis=0) for arm in ("a", "b")}
        res["mae_sunny"] = {arm: float(np.abs(q50[arm] - rows["y"])[rows["sunny"]].mean() * 100) for arm in q50}
        out[world] = res
    return out


def main() -> None:
    """Write and print the contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("contrasts",))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k639.SEEDS))
    parser.add_argument("--tag", default="stage1")
    a = parser.parse_args()
    result = contrasts(seeds=tuple(a.seeds))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"contrasts_{a.tag}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

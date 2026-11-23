# ruff: noqa: E501, T201, PLR0913, PLR0915, PT018
"""Experiment 720, rung 0e: V4 and the analyses of section 3 of the addendum, as pure functions.

Every function takes arrays and returns arrays or small tables with no dates and no generator values.
The real inputs (day classes of the generator Primary, the Primary's temperature, stored forecasts of
experiment 713) are read by `run_section3`, which writes only under the private folder, and publishes
pooled counts and bands through `e00720_guard.public_aggregate` alone. Not wired: the system-price
fifths (the price files of experiments 185 and 570 are not read here), and the two cleaner series in
the MSA network (they need a network fit, which this rung does not run).
"""

import sys
from typing import Any, Final

import numpy as np

RUNNING_MIN_HALF_HOURS: Final[int] = 4
IDLE_MAX_P: Final[float] = 0.05
RUNNING_P: Final[float] = 0.5
EXPORT_BANDS: Final[dict[str, tuple[float, float]]] = {
    "march_to_september": (0.70, 0.93),
    "october_to_february": (0.03, 0.25),
}
"""The share of export days that `reverse_flow_analysis.md` reports at the exporting Primary."""
TEMP_EDGES: Final[tuple[float, ...]] = (5.0, 10.0, 15.0)


def classify_days(*, p_any: np.ndarray) -> np.ndarray:
    """Return 1 for a running day (at least 4 half-hours above 0.5), 0 for an idle day (none above 0.05), else -1.

    Args:
        p_any: `P(n >= 1)` per half-hour, `(days, 48)`.
    """
    running = (p_any > RUNNING_P).sum(axis=1) >= RUNNING_MIN_HALF_HOURS
    idle = ~(p_any > IDLE_MAX_P).any(axis=1)
    return np.where(running, 1, np.where(idle, 0, -1))


def temperature_band(
    *, temperature: np.ndarray, edges: tuple[float, ...] = TEMP_EDGES
) -> np.ndarray:
    """Return the index of the temperature band of each day (0 below the first edge)."""
    return np.searchsorted(np.asarray(edges), temperature, side="right")


def matched_profile_difference(
    *,
    profile: np.ndarray,
    cls: np.ndarray,
    day_type: np.ndarray,
    month: np.ndarray,
    band: np.ndarray,
    min_per_class: int = 3,
) -> dict[str, Any]:
    """Return the mean running-minus-idle half-hourly profile, matched on day type, month, and temperature band.

    Each cell (day type, month, band) that holds at least `min_per_class` running and `min_per_class`
    idle days contributes its difference of means, weighted by the smaller of the two counts. Days of
    other classes and days with a NaN profile are left out.

    Args:
        profile: `(days, 48)` the reading, or recovered gross demand.
        cls: Day class from `classify_days`.
        day_type: 0 weekday, 1 Saturday, 2 Sunday.
        month: Calendar month.
        band: Temperature band.
        min_per_class: Smallest count of each class in a cell.

    Returns:
        `difference` `(48,)`, `n_cells`, `n_running`, `n_idle` (the days used), and the unmatched
        difference `raw_difference` for comparison.
    """
    ok = np.isfinite(profile).all(axis=1)
    diff = np.zeros(profile.shape[1])
    wsum = 0.0
    n_cells = n_run = n_idle = 0
    key = (day_type * 13 + month) * 8 + band
    for c in np.unique(key[ok]):
        sel = ok & (key == c)
        r, i = sel & (cls == 1), sel & (cls == 0)
        if r.sum() >= min_per_class and i.sum() >= min_per_class:
            w = float(min(r.sum(), i.sum()))
            diff += w * (profile[r].mean(axis=0) - profile[i].mean(axis=0))
            wsum += w
            n_cells += 1
            n_run += int(r.sum())
            n_idle += int(i.sum())
    r_all, i_all = ok & (cls == 1), ok & (cls == 0)
    raw = (
        profile[r_all].mean(axis=0) - profile[i_all].mean(axis=0)
        if r_all.any() and i_all.any()
        else np.full(profile.shape[1], np.nan)
    )
    return {
        "difference": diff / wsum if wsum else np.full(profile.shape[1], np.nan),
        "n_cells": n_cells, "n_running": n_run, "n_idle": n_idle, "raw_difference": raw,
    }  # fmt: skip


def running_share(*, running: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Return the share of half-hours with `running` true in each group `0..n_groups-1` (NaN where empty)."""
    out = np.full(n_groups, np.nan)
    for g in range(n_groups):
        sel = group == g
        if sel.any():
            out[g] = float(running[sel].mean())
    return out


def fifths(*, price: np.ndarray) -> np.ndarray:
    """Return the fifth (0 cheapest to 4 dearest) of each finite price, -1 for NaN."""
    out = np.full(price.shape, -1, dtype=int)
    ok = np.isfinite(price)
    edges = np.nanquantile(price[ok], [0.2, 0.4, 0.6, 0.8])
    out[ok] = np.searchsorted(edges, price[ok], side="right")
    return out


def d12_export_summary(*, p_neg: np.ndarray, u: np.ndarray, month: np.ndarray) -> dict[str, Any]:
    """Return V4's D12 read-outs: `P(X < 0)` in the dark, and the export-day share by season against its band.

    An export day has at least 2 half-hours with `u >= 0.3` and `P(X < 0) > 0.5`. Months are pooled into
    the two seasons of the bands, so no per-month series is returned.
    """
    dark = (u < 0.01) & np.isfinite(p_neg)
    gated = (u >= 0.3) & np.isfinite(p_neg)
    day_export = ((p_neg > 0.5) & gated).sum(axis=1) >= 2
    has_day = gated.sum(axis=1) >= 2
    out: dict[str, Any] = {
        "dark_mean_p_negative": float(p_neg[dark].mean()) if dark.any() else float("nan"),
        "dark_share_over_half": float((p_neg[dark] > 0.5).mean()) if dark.any() else float("nan"),
    }
    summer = np.isin(month, range(3, 10))
    for name, sel in (("march_to_september", summer), ("october_to_february", ~summer)):
        d = has_day & sel
        share = float(day_export[d].mean()) if d.any() else float("nan")
        lo, hi = EXPORT_BANDS[name]
        out[f"export_day_share_{name}"] = share
        out[f"within_band_{name}"] = bool(lo <= share <= hi) if np.isfinite(share) else False
        out[f"n_days_{name}"] = int(d.sum())
    return out


def recut_by_class(
    *,
    abs_error: np.ndarray,
    hour: np.ndarray,
    target_cls: np.ndarray,
    history_cls: np.ndarray,
    solar_block: tuple[int, int] = (10, 15),
) -> dict[str, Any]:
    """Re-score stored forecasts by the target day's class and the history majority's class, outside the solar block.

    Args:
        abs_error: Absolute error of the median forecast, one value per forecast row.
        hour: Local hour of the row's valid time.
        target_cls: Class (1 running, 0 idle) of the row's target day.
        history_cls: Class of most history days (1 running, 0 idle).
        solar_block: Hours of the day excluded.

    Returns:
        `cells`: the mean error and the row count in each of the four (target, history) cells, and
        `different`, the mean error of the rows whose two classes differ against those that agree.
    """
    outside = (hour < solar_block[0]) | (hour >= solar_block[1])
    ok = (
        outside
        & np.isfinite(abs_error)
        & np.isin(target_cls, (0, 1))
        & np.isin(history_cls, (0, 1))
    )
    cells = {}
    for t in (0, 1):
        for h in (0, 1):
            sel = ok & (target_cls == t) & (history_cls == h)
            cells[(t, h)] = (
                float(abs_error[sel].mean()) if sel.any() else float("nan"),
                int(sel.sum()),
            )
    diff, same = ok & (target_cls != history_cls), ok & (target_cls == history_cls)
    return {
        "cells": cells,
        "different": float(abs_error[diff].mean()) if diff.any() else float("nan"),
        "same": float(abs_error[same].mean()) if same.any() else float("nan"),
        "n_different": int(diff.sum()), "n_same": int(same.sum()),
    }  # fmt: skip


def history_majority(*, day_cls: np.ndarray, n_days: int) -> np.ndarray:
    """Return, per day, the class held by most of the `n_days` days before it (-1 when none is running or idle, or on a tie)."""
    out = np.full(len(day_cls), -1)
    for d in range(n_days, len(day_cls)):
        h = day_cls[d - n_days : d]
        r, i = int((h == 1).sum()), int((h == 0).sum())
        out[d] = 1 if r > i else 0 if i > r else -1
    return out


def run_section3(
    *, dense_raw: Any = None, cap_hours: float = 15.0, fit_kwargs: dict[str, Any] | None = None
) -> int:  # pragma: no cover - needs the stored panel
    """Run the D8 analyses of section 3 and the V4 read-outs on stored assets (waits for the code review).

    Reads experiment 694's panel (through its loaders), the private era file `era_dates.json` (keys
    `d8_era_start` and `d8_era_end`, ISO dates, never committed), the D8 temperature from experiment 713's
    dense array, and experiment 713's stored `M1` forecasts. Every output goes under the private folder.
    Reports of D8 are pooled over the era: no month or date labels leave this function, and the running
    months are reduced to the hour bands of `e00720_guard.pool_bands`.
    """
    import json
    from datetime import UTC, datetime

    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00694_run as r694  # ty: ignore[unresolved-import]
    import e00720_guard as guard
    import e00720_hmm as hmm
    import e00720_v2 as v2
    import polars as pl

    era = json.loads(guard.private_path(name="era_dates.json").read_text())
    start, end = (
        datetime.fromisoformat(era[k]).replace(tzinfo=UTC) for k in ("d8_era_start", "d8_era_end")
    )
    r694.init()
    a, b = c694.stamp(when=start), c694.stamp(when=end)
    y = r694.SERIES["D8"][a - r694.PRE_ROLL : b]
    gate = hmm.CostGate(cap_hours=cap_hours)
    hf = v2.hindsight_fit(
        y=y,
        first_index=a - r694.PRE_ROLL,
        pre_roll=r694.PRE_ROLL,
        gate=gate,
        projected_fits=1.0,
        **(fit_kwargs or {}),
    )
    assert hf.p_any is not None
    assert hf.e_n is not None
    p_days, _ = v2.to_days(a=hf.p_any, first_index=a)
    cls = classify_days(p_any=np.nan_to_num(p_days, nan=0.5))
    q = c694.Q_FLOOR
    real_mw = np.sqrt(np.maximum(y[r694.PRE_ROLL :] ** 2 - q**2, 0.0))
    gross_mw = real_mw + hf.unit_size * hf.e_n
    read_d, _ = v2.to_days(a=real_mw, first_index=a)
    gross_d, _ = v2.to_days(a=gross_mw, first_index=a)
    day_idx = a // 48 + np.arange(len(cls))
    weekday = (v2.START_DOW_GRID + day_idx) % 7
    day_type = np.where(weekday == 5, 1, np.where(weekday == 6, 2, 0))
    base = np.datetime64(c694.GRID_START.replace(tzinfo=None), "D")
    month = ((base + day_idx).astype("datetime64[M]").astype(int)) % 12 + 1
    rows: list[dict[str, Any]] = []
    if dense_raw is not None:
        import e00506_data as data  # ty: ignore[unresolved-import]
        import e00720_real as real

        sites = sorted(
            data.series_table().filter(pl.col("substation_type") == "Primary")["site"].to_list(),
            key=lambda x: int(x[1:]),
        )
        raw = np.load(dense_raw, mmap_mode="r")
        e87_start = datetime(2019, 9, 1, tzinfo=UTC)
        offset = int((c694.GRID_START - e87_start).total_seconds() // 1800) + a
        t = np.asarray(
            raw[sites.index("D8"), offset : offset + len(y) - r694.PRE_ROLL, 2], dtype=float
        )
        temp = np.nanmean(v2.to_days(a=t, first_index=a)[0], axis=1)
        band = temperature_band(temperature=np.nan_to_num(temp, nan=10.0))
        del real
    else:
        band = np.zeros(len(cls), dtype=int)
    for name, prof in (("reading", read_d), ("gross", gross_d)):
        m = matched_profile_difference(
            profile=prof, cls=cls, day_type=day_type, month=month, band=band
        )
        rows += [
            {
                "profile": name,
                "half_hour": h,
                "matched": float(m["difference"][h]),
                "unmatched": float(m["raw_difference"][h]),
                "n_cells": m["n_cells"],
            }
            for h in range(48)
        ]
    pl.DataFrame(rows).write_csv(guard.private_path(name="d8_running_vs_idle_profiles.csv"))
    run_hh = (p_days > RUNNING_P).reshape(-1)
    share_hour = running_share(running=run_hh, group=np.tile(np.arange(48), len(cls)), n_groups=48)
    share_band = running_share(
        running=(p_days > RUNNING_P).any(axis=1), group=band, n_groups=len(TEMP_EDGES) + 1
    )
    pl.DataFrame({"hour_slot": range(48), "running_share": share_hour}).write_csv(
        guard.private_path(name="d8_running_share_by_hour.csv")
    )
    pl.DataFrame(
        {"temperature_band": range(len(share_band)), "running_day_share": share_band}
    ).write_csv(guard.private_path(name="d8_running_share_by_temperature.csv"))
    monthly_hours = np.array(
        [float((p_days[month == m] > RUNNING_P).sum()) / 2.0 for m in range(1, 13)]
    )
    pl.DataFrame(
        {
            "band": guard.pool_bands(monthly_hours=monthly_hours)["band"],
            "n_months": guard.pool_bands(monthly_hours=monthly_hours)["n_months"],
        }
    ).write_csv(guard.private_path(name="d8_running_hours_bands.csv"))
    print("wrote the D8 section-3 tables under the private folder")
    return 0


def run_m1_recut(
    *, cls_by_day: np.ndarray, first_day_us: int, ts_id: int, history_days: int = 7
) -> dict[str, Any]:  # pragma: no cover - needs the stored forecasts
    """Re-score experiment 713's stored `M1` forecasts of one Primary by day class (cells pooled across folds).

    `cls_by_day[d]` is the class (1 running, 0 idle, -1 other) of UTC day `first_day_us // US_PER_DAY + d`.
    The history majority is the class held by most of the `history_days` days before the target day, an
    assumption the page states. Reads `work/fits/M1/*.parquet` of experiment 713; nothing is written.
    """
    import e00720_real as real
    import polars as pl
    from ml_core.paths import ML_EXPERIMENTS_DATA_DIR  # ty: ignore[unresolved-import]

    root = (
        ML_EXPERIMENTS_DATA_DIR
        / "00713-msa-axial-attention-demand-network-part-1"
        / "work"
        / "fits"
        / "M1"
    )
    hist = history_majority(day_cls=cls_by_day, n_days=history_days)
    frames = [
        pl.read_parquet(p).filter(pl.col("time_series_id") == ts_id)
        for p in sorted(root.glob("*_s0.parquet"))
    ]
    f = pl.concat(frames)
    vt = f["valid_time"].dt.epoch("us").to_numpy()
    day = (vt // real.US_PER_DAY) - (first_day_us // real.US_PER_DAY)
    inside = (day >= 0) & (day < len(cls_by_day))
    err = np.abs(f["q50"].to_numpy() - f["actual"].to_numpy())[inside]
    hour = ((vt % real.US_PER_DAY) // 3_600_000_000)[inside]
    return recut_by_class(
        abs_error=err, hour=hour, target_cls=cls_by_day[day[inside]], history_cls=hist[day[inside]]
    )


def selftest() -> int:
    """Check each function against a known answer on synthetic data."""
    rng = np.random.default_rng(2)
    nd = 400
    dtype = (np.arange(nd) % 7 >= 5).astype(int) * 1
    month = (np.arange(nd) // 30) % 12 + 1
    temp = 8.0 + 8.0 * np.sin(np.arange(nd) / 58.0) + rng.normal(0, 1, nd)
    band = temperature_band(temperature=temp)
    # the engine runs more on cold weekdays, so unmatched profiles are confounded
    p_run = np.clip(0.25 + 0.5 * (temp < 8) * (dtype == 0), 0, 1)
    running_day = rng.random(nd) < p_run
    p_any = np.where(running_day[:, None] & (np.arange(48) >= 36)[None, :], 0.9, 0.01)
    cls = classify_days(p_any=p_any)
    assert (cls[running_day] == 1).all() and (cls[~running_day] == 0).all()
    gross = (
        5.0 + 2.0 * (temp[:, None] < 8) * np.ones(48) + rng.normal(0, 0.1, (nd, 48))
    )  # gross demand depends on cold only
    reading = gross - 1.5 * running_day[:, None] * (np.arange(48) >= 36)[None, :]
    mg = matched_profile_difference(profile=gross, cls=cls, day_type=dtype, month=month, band=band)
    mr = matched_profile_difference(
        profile=reading, cls=cls, day_type=dtype, month=month, band=band
    )
    assert mg["n_cells"] > 5
    assert np.abs(mg["difference"]).max() < 0.35, np.abs(
        mg["difference"]
    ).max()  # matched: gross has the same profile
    assert abs(mr["difference"][40] + 1.5) < 0.4 and abs(mr["difference"][5]) < 0.35
    assert (
        np.abs(mg["raw_difference"]).max() > np.abs(mg["difference"]).max()
    )  # unmatched is confounded
    # running share
    grp = np.tile(np.arange(48), nd)
    sh = running_share(running=(p_any > 0.5).reshape(-1), group=grp, n_groups=48)
    assert sh[40] > 0.2 and sh[3] == 0.0
    f = fifths(price=np.arange(100.0))
    assert np.bincount(f).tolist() == [20] * 5
    # D12 summary: exports on 80% of summer days, none in the dark
    u = np.tile(np.clip(np.sin(np.pi * (np.arange(48) - 10) / 28.0), 0, 1), (nd, 1))
    exp = rng.random(nd) < np.where(np.isin(month, range(3, 10)), 0.8, 0.1)
    pn = np.where(exp[:, None] & (u > 0.5), 0.95, 0.02)
    s = d12_export_summary(p_neg=pn, u=u, month=month)
    assert (
        s["within_band_march_to_september"]
        and s["within_band_october_to_february"]
        and s["dark_mean_p_negative"] < 0.05
    ), s
    # re-cut: error is larger when the target and the history classes differ
    t_cls = rng.integers(0, 2, 5000)
    h_cls = rng.integers(0, 2, 5000)
    hr = rng.integers(0, 24, 5000)
    err = 0.3 + 0.6 * (t_cls != h_cls) + rng.normal(0, 0.05, 5000)
    rc = recut_by_class(abs_error=err, hour=hr, target_cls=t_cls, history_cls=h_cls)
    assert abs(rc["different"] - 0.9) < 0.05 and abs(rc["same"] - 0.3) < 0.05
    maj = history_majority(day_cls=np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1]), n_days=3)
    assert maj.tolist()[3] == 1 and maj.tolist()[6] == 0
    print(
        "PASS section-3 analyses and V4 read-outs: matched profiles, running share, fifths, D12 export bands, class re-cut"
    )
    return 0


if __name__ == "__main__":
    sys.exit(selftest())

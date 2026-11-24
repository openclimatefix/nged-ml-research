# ruff: noqa: E501, PLR0915, PT018
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


def history_class_before_origin(
    *, cls_by_day: np.ndarray, origin_day: np.ndarray, first: int = 2, last: int = 15
) -> np.ndarray:
    """Return the class held by most of the UTC days `first` to `last` days before each origin's date (-1 on a tie or none).

    This is the history of `reverse_flow_analysis.md` section 4: the 14 UTC days 2 to 15 days before the
    origin's date. `origin_day` holds each origin's day index on the same axis as `cls_by_day`.
    """
    out = np.full(len(origin_day), -1)
    for i, d in enumerate(origin_day):
        lo, hi = int(d) - last, int(d) - first + 1
        if lo < 0 or hi > len(cls_by_day):
            continue
        h = cls_by_day[lo:hi]
        r, n_idle = int((h == 1).sum()), int((h == 0).sum())
        out[i] = 1 if r > n_idle else 0 if n_idle > r else -1
    return out


def monthly_running_hours(*, p_days: np.ndarray, day_month_index: np.ndarray) -> np.ndarray:
    """Return the hours with `P(n >= 1) > 0.5` in each distinct calendar month (year and month), not summed across years."""
    return np.array(
        [
            float((p_days[day_month_index == m] > RUNNING_P).sum()) / 2.0
            for m in np.unique(day_month_index)
        ]
    )


def v2_k1_gate_passed(*, private: Any) -> bool:
    """Return whether the private V2 and K1 outputs both exist and both bars passed (D8's read-outs are gated on them)."""
    import polars as pl

    v2p, k1p = private / "v2_bar.csv", private / "k1_v3.csv"
    if not (v2p.exists() and k1p.exists()):
        return False
    return bool(pl.read_csv(v2p)["bar_passed"][0]) and bool(pl.read_csv(k1p)["k1_passed"][0])


def run_section3(
    *,
    dense_raw: Any = None,
    cap_hours: float = 15.0,
    fit_kwargs: dict[str, Any] | None = None,
    require_gate: bool = True,
) -> int:  # pragma: no cover - needs the stored panel
    """Run the D8 analyses of section 3 and the V4 read-outs on stored assets (waits for the code review).

    Refuses to run unless the private V2 and K1 results passed. Reads experiment 694's panel (through its
    loaders), the private era file `era_dates.json` (keys `d8_era_start`, `d8_era_end`, and `solar_start`,
    ISO dates, never committed; the era must end before the farm joins the series, so the reading is
    `X >= 0` and the recovered gross demand is the reading plus the engine's expected output), the D8 temperature from experiment 713's
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

    if require_gate and not v2_k1_gate_passed(private=guard.PRIVATE_DIR):
        msg = (
            "D8's read-outs run only after V2 and K1 both passed (private v2_bar.csv and k1_v3.csv)"
        )
        raise SystemExit(msg)
    era = json.loads(guard.private_path(name="era_dates.json").read_text())
    start, end, solar = (
        datetime.fromisoformat(era[k]).replace(tzinfo=UTC)
        for k in ("d8_era_start", "d8_era_end", "solar_start")
    )
    if end > solar:
        msg = "the era must end before the farm joins the series: until the farm-and-engine variant passes, no signed-power read-out runs there"
        raise SystemExit(msg)
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
    month_abs = (base + day_idx).astype("datetime64[M]").astype(int)
    month = month_abs % 12 + 1
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
    monthly_hours = monthly_running_hours(p_days=p_days, day_month_index=month_abs)
    pl.DataFrame(
        {
            "band": guard.pool_bands(monthly_hours=monthly_hours)["band"],
            "n_months": guard.pool_bands(monthly_hours=monthly_hours)["n_months"],
        }
    ).write_csv(guard.private_path(name="d8_running_hours_bands.csv"))
    print("wrote the D8 section-3 tables under the private folder")
    return 0


def run_m1_recut(
    *, cls_by_day: np.ndarray, first_day_us: int, ts_id: int
) -> dict[str, Any]:  # pragma: no cover - needs the stored forecasts
    """Re-score experiment 713's stored `M1` minus Candidate forecasts of one Primary by day class, cells pooled across folds.

    A cell is one origin and one valid day of the headline band (leads of 24 h up to but not including
    240 h), as in `reverse_flow_analysis.md` section 4. `cls_by_day[d]` is the class (1 running, 0 idle,
    -1 other) of UTC day `first_day_us // US_PER_DAY + d`. The history class of a cell is the majority
    class of the UTC days 2 to 15 days before the origin's date. The measure is the absolute error of
    `M1`'s median minus the Candidate's, seed 0 of each. Reads `work/fits/M1/*_s0.parquet` of experiment 713
    and the Candidate's stored seed-0 files of experiment 667; nothing is written.
    """
    import e00667_export as exp  # ty: ignore[unresolved-import]
    import e00720_real as real
    import polars as pl
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    root = (
        ML_EXPERIMENTS_DATA_DIR
        / "00713-msa-axial-attention-demand-network-part-1"
        / "work"
        / "fits"
        / "M1"
    )
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    parts = []
    for path in sorted(root.glob("*_s0.parquet")):
        month = path.name.split("_")[0]
        m1 = (
            pl.read_parquet(path)
            .filter(pl.col("time_series_id") == ts_id)
            .select(*keys, "lead_hours", m1="q50", actual="actual")
        )
        cand = exp.ROOT / f"P7_D50_{month}" / "work" / "fits" / "P5" / f"{month}_d0_s0.parquet"
        c = pl.read_parquet(cand).filter(pl.col("time_series_id") == ts_id).select(*keys, c="q50")
        parts.append(m1.join(c, on=keys, how="inner"))
    f = pl.concat(parts).filter((pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 240))
    vt = f["valid_time"].dt.epoch("us").to_numpy()
    origin = f["power_fcst_init_time"].dt.epoch("us").to_numpy()
    day0 = first_day_us // real.US_PER_DAY
    day = vt // real.US_PER_DAY - day0
    origin_day = origin // real.US_PER_DAY - day0
    inside = (day >= 0) & (day < len(cls_by_day))
    diff = (
        np.abs(f["m1"].to_numpy() - f["actual"].to_numpy())
        - np.abs(f["c"].to_numpy() - f["actual"].to_numpy())
    )[inside]
    hour = ((vt % real.US_PER_DAY) // 3_600_000_000)[inside]
    hist = history_class_before_origin(cls_by_day=cls_by_day, origin_day=origin_day[inside])
    return recut_by_class(
        abs_error=diff, hour=hour, target_cls=cls_by_day[day[inside]], history_cls=hist
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
    day_cls = np.array([1] * 6 + [0] * 8 + [1] * 10 + [0] * 10)
    h = history_class_before_origin(cls_by_day=day_cls, origin_day=np.array([16, 21, 5, 22, 30]))
    assert h.tolist() == [0, 0, -1, -1, 1], (
        h.tolist()
    )  # origin 5 has too little history; origin 22 is a tie
    mw = monthly_running_hours(
        p_days=np.vstack([np.full((30, 48), 0.9), np.zeros((30, 48))]),
        day_month_index=np.array([0] * 30 + [12] * 30),
    )
    assert mw.tolist() == [720.0, 0.0]  # two Januaries a year apart are two months, not one
    print(
        "PASS section-3 analyses and V4 read-outs: matched profiles, running share, fifths, D12 export bands, class re-cut"
    )
    return 0


if __name__ == "__main__":
    sys.exit(selftest())

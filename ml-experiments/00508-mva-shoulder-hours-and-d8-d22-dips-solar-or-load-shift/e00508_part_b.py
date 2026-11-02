"""Experiment 508, Part B: are the D8 and D22 midday dips solar, load transfers, or the engine?

    PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-*:ml-experiments/00508-* \
        uv run --no-sync python ml-experiments/00508-*/e00508_part_b.py

Every test has a placebo or a known answer. The shape test is a labelled diagnostic only.
"""

import json
from datetime import UTC, date, datetime
from typing import Final

import e00506_data as data  # ty: ignore[unresolved-import]
import e00508_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00508_geometry import Mats  # ty: ignore[unresolved-import]

DIP_POINTS: Final[float] = 6.9
"""Experiment 506's dip threshold (points of capacity)."""
N_BASE: Final[int] = 14
SLOTS: Final[slice] = slice(16, 35)
"""UK-clock half-hours 08:00 to 17:00 of a day."""
MID: Final[slice] = slice(20, 30)
ENGINE_FLAG: Final[float] = 0.5
ENGINE_START: Final[date] = date(2023, 8, 15)
N_BOOT: Final[int] = 20_000
N_INJECT: Final[int] = 300
SHAPE_ACCURACY: Final[float] = 0.8
SEED: Final[int] = 5081
SIBLINGS: Final[dict[str, tuple[str, ...]]] = {
    "D8": ("D3", "D12", "D22"),
    "D22": ("D3", "D8", "D12"),
}
ENGINE_DIR: Final = data.ML_EXPERIMENTS_DATA_DIR / "00291-d8-drops-engine-spells-load-shift"
SPREAD_TABLE: Final = (
    data.ML_EXPERIMENTS_DATA_DIR
    / "00186-switching-d8-gas-peaker-spark-spread-and-triads"
    / "work"
    / "d8_day_table.parquet"
)
LABELLED: Final = (
    data.ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
)


def engine_on_grid(*, label: str) -> np.ndarray:
    """Return experiment 291's engine-running probability aligned to the grid (NaN outside)."""
    run = np.load(ENGINE_DIR / f"p_run_{label}_v1.npy")
    start = (data.grid().dt.offset_by("-30m")).to_numpy().astype("datetime64[m]").astype(np.int64)
    origin = np.datetime64("2022-01-01T00:00").astype("datetime64[m]").astype(np.int64)
    idx = (start - origin) // 30
    out = np.full(len(idx), np.nan)
    ok = (idx >= 0) & (idx < len(run))
    out[ok] = run[idx[ok]]
    return out


def baseline(*, mats: Mats, which: str, i: int, pool: np.ndarray, window: slice) -> np.ndarray:
    """Return the median of the same half-hours on the 14 nearest same-day-type pool days."""
    arr = getattr(mats, which) if which in {"p", "u", "k"} else mats.extra[which]
    cand = np.where(pool & (mats.weekend == mats.weekend[i]))[0]
    cand = cand[cand != i]
    cand = cand[np.isfinite(arr[cand][:, window]).all(axis=1)]
    near = cand[np.argsort(np.abs(mats.ordinal[cand] - mats.ordinal[i]), kind="stable")[:N_BASE]]
    return np.median(arr[near][:, window], axis=0)


def classify(*, drop: np.ndarray, shape: np.ndarray) -> dict[str, float]:
    """Compare a solar-shaped drop with a boxcar of the same complexity (one amplitude each).

    `drop` is the baseline minus the reading over the day's slots, and `shape` the half-hourly
    fleet shape `u`. The boxcar covers the contiguous run around the deepest slot where the drop is
    at least half its peak. Each shape gets its least-squares amplitude (at least zero). The label
    is solar if the solar shape leaves the smaller sum of squares.
    """
    total = float((drop**2).sum())
    peak = int(np.argmax(drop))
    if drop[peak] <= 0:
        return {"r2_solar": float("nan"), "r2_box": float("nan"), "label_solar": float("nan")}
    lo = hi = peak
    while lo > 0 and drop[lo - 1] >= 0.5 * drop[peak]:
        lo -= 1
    while hi < len(drop) - 1 and drop[hi + 1] >= 0.5 * drop[peak]:
        hi += 1
    box = np.zeros(len(drop))
    box[lo : hi + 1] = 1.0
    sse = []
    for basis in (shape, box):
        amp = max(0.0, float((drop * basis).sum() / max((basis**2).sum(), 1e-9)))
        sse.append(float(((drop - amp * basis) ** 2).sum()))
    return {
        "r2_solar": 1 - sse[0] / total,
        "r2_box": 1 - sse[1] / total,
        "label_solar": float(sse[0] < sse[1]),
    }


def wilson(*, k: int, n: int) -> tuple[float, float]:
    """Return the 95% Wilson interval of a share."""
    if n == 0:
        return float("nan"), float("nan")
    z, ph = 1.96, k / n
    centre = (ph + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def dip_days(*, site: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (all sunny and cloudy midday day rows, dip rows) of a site with the dip flag."""
    table, _ = common.day_tables(site=site, level=0.0)
    midday = table.filter(pl.col("window") == "midday").with_columns(dip=pl.col("Q0") < DIP_POINTS)
    return midday, midday.filter(pl.col("dip"), pl.col("sky") == "sunny")


def shape_test(*, site: str, rng: np.random.Generator) -> pl.DataFrame:
    """Classify every dip day, injected boxcars and solar-shaped drops, and D10's real days."""
    midday, dips = dip_days(site=site)
    mats = Mats(site=site)
    where = {int(d): i for i, d in enumerate(mats.dates)}
    dip_set = set(dips["date"].to_list())
    pool = np.array([int(d) not in dip_set for d in mats.dates])
    rows = []
    for d in dips["date"].to_list():
        i = where.get(int(d))
        if i is None:
            continue
        base = baseline(mats=mats, which="p", i=i, pool=pool, window=SLOTS)
        drop = base - np.abs(mats.p[i, SLOTS])
        if not np.isfinite(drop).all():
            continue
        rows.append(
            {"site": site, "kind": "dip", "date": int(d), "year": int(mats.year[i])}
            | classify(drop=drop, shape=mats.u[i, SLOTS])
        )
    if True:
        early = midday.filter(pl.col("sky") == "sunny", pl.col("year") <= 2022, ~pl.col("dip"))[
            "date"
        ].to_numpy()
        chosen = rng.choice(early, size=min(N_INJECT, len(early)), replace=False)
        for kind in ("boxcar", "solar", "none"):
            for d in chosen:
                i = where.get(int(d))
                if i is None:
                    continue
                base = baseline(mats=mats, which="p", i=i, pool=pool, window=SLOTS)
                reading = np.abs(mats.p[i, SLOTS]).copy()
                amp = rng.uniform(40.0, 50.0)
                if kind == "boxcar":
                    length = int(rng.integers(3, 9))
                    start = int(
                        rng.integers(MID.start - SLOTS.start, MID.stop - SLOTS.start - length + 1)
                    )
                    reading[start : start + length] -= amp
                elif kind == "solar":
                    shape = mats.u[i, SLOTS]
                    reading -= amp * shape / max(shape.max(), 1e-9)
                reading = np.abs(reading)
                drop = base - reading
                if not np.isfinite(drop).all():
                    continue
                rows.append(
                    {
                        "site": site,
                        "kind": f"injected_{kind}",
                        "date": int(d),
                        "year": int(mats.year[i]),
                    }
                    | classify(drop=drop, shape=mats.u[i, SLOTS])
                )
    return pl.DataFrame(rows)


def d10_calibration() -> pl.DataFrame:
    """Classify D10's real reverse-flow days (signed midday minimum below zero)."""
    table, _ = common.day_tables(site="D10", level=0.0)
    real = table.filter(
        pl.col("window") == "midday", pl.col("sky") == "sunny", pl.col("signed_min") < 0
    )["date"].to_list()
    mats = Mats(site="D10")
    where = {int(d): i for i, d in enumerate(mats.dates)}
    pool = np.array([int(d) not in set(real) for d in mats.dates])
    rows = []
    for d in real:
        i = where.get(int(d))
        if i is None:
            continue
        base = baseline(mats=mats, which="p", i=i, pool=pool, window=SLOTS)
        drop = base - mats.p[i, SLOTS]
        if np.isfinite(drop).all():
            rows.append(
                {"site": "D10", "kind": "real_reverse", "date": int(d), "year": int(mats.year[i])}
                | classify(drop=drop, shape=mats.u[i, SLOTS])
            )
    return pl.DataFrame(rows)


def sky_placebo(*, site: str, rng: np.random.Generator) -> pl.DataFrame:
    """Return dip rates on sunny and cloudy middays of March to October, by year and pooled."""
    midday, _ = dip_days(site=site)
    sub = midday.filter(pl.col("month").is_between(3, 10))
    out = []
    for year in [*sorted(sub["year"].unique().to_list()), "pooled"]:
        part = sub if year == "pooled" else sub.filter(pl.col("year") == year)
        row = {"site": site, "year": str(year)}
        for sky in ("sunny", "cloudy"):
            q = part.filter(pl.col("sky") == sky)
            k, n = int(q["dip"].sum()), q.height
            lo, hi = wilson(k=k, n=n)
            row |= {
                f"{sky}_dips": k,
                f"{sky}_days": n,
                f"{sky}_rate": k / n if n else float("nan"),
                f"{sky}_lo": lo,
                f"{sky}_hi": hi,
            }
        out.append(row)
    # Month-block bootstrap of the pooled cloudy-to-sunny ratio of rates.
    blocks = sub.with_columns(block=pl.col("year") * 100 + pl.col("month"))
    ids = np.sort(blocks["block"].unique().to_numpy())
    stats = {
        sky: blocks.filter(pl.col("sky") == sky)
        .group_by("block")
        .agg(k=pl.col("dip").sum(), n=pl.len())
        .join(pl.DataFrame({"block": ids}), on="block", how="right")
        .sort("block")
        .fill_null(0)
        for sky in ("sunny", "cloudy")
    }
    ks, ns = stats["sunny"]["k"].to_numpy(), stats["sunny"]["n"].to_numpy()
    kc, nc = stats["cloudy"]["k"].to_numpy(), stats["cloudy"]["n"].to_numpy()

    def mantel_haenszel(w: np.ndarray) -> float:
        """Cloudy-to-sunny risk ratio over year-month strata, with 0.5 added to each sum."""
        tot = np.maximum(ns + nc, 1)
        num = (w * kc * ns / tot).sum() + 0.5
        den = (w * ks * nc / tot).sum() + 0.5
        return float(num / den)

    ratios = np.array(
        [
            mantel_haenszel(np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)))
            for _ in range(N_BOOT)
        ]
    )
    out[-1] |= {
        "mh_cloudy_to_sunny": mantel_haenszel(np.ones(len(ids))),
        "mh_lo": float(np.percentile(ratios, 2.5)),
        "mh_hi": float(np.percentile(ratios, 97.5)),
        "n_strata": len(ids),
    }
    return pl.DataFrame(out, infer_schema_length=None)


def engine_test(*, rng: np.random.Generator) -> tuple[pl.DataFrame, dict]:
    """Return per-dip-day engine overlap at D8 against matched placebo days, and summary stats."""
    extra = {f"run_{s}": engine_on_grid(label=s) for s in ("D8", "D5", "D10", "D11")}
    mats = Mats(site="D8", extra=extra)
    midday, dips = dip_days(site="D8")
    dip_set = set(dips["date"].to_list())
    where = {int(d): i for i, d in enumerate(mats.dates)}
    pool = np.array([int(d) not in dip_set for d in mats.dates])
    rows = []
    for d in dips["date"].to_list():
        i = where.get(int(d))
        if i is None or date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100) < ENGINE_START:
            continue
        low = np.abs(mats.p[i, MID]) < DIP_POINTS
        if not low.any():
            continue
        slots = np.arange(MID.start, MID.stop)[low]
        run = mats.extra["run_D8"][i, slots]
        # Placebo: the same slots on the 14 nearest same-day-type non-dip days after the start.
        cand = np.where(
            pool & (mats.weekend == mats.weekend[i]) & (mats.ordinal >= ENGINE_START.toordinal())
        )[0]
        cand = cand[np.isfinite(mats.extra["run_D8"][cand][:, slots]).all(axis=1)]
        near = cand[
            np.argsort(np.abs(mats.ordinal[cand] - mats.ordinal[i]), kind="stable")[:N_BASE]
        ]
        placebo = mats.extra["run_D8"][near][:, slots].mean()
        rows.append(
            {
                "date": int(d),
                "n_dip_half_hours": int(low.sum()),
                "engine_share": float(np.nanmean(run > ENGINE_FLAG)),
                "placebo_share": float(placebo > ENGINE_FLAG)
                if False
                else float((mats.extra["run_D8"][near][:, slots] > ENGINE_FLAG).mean()),
                "control_share": float(
                    np.nanmean(
                        [
                            (mats.extra[f"run_{c}"][i, slots] > ENGINE_FLAG).mean()
                            for c in ("D5", "D10", "D11")
                        ]
                    )
                ),
            }
        )
    table = pl.DataFrame(rows).with_columns(coincident=pl.col("engine_share") >= 0.5)
    spread = pl.read_parquet(SPREAD_TABLE).select("date", "spread_40")
    sunny = midday.filter(pl.col("sky") == "sunny").with_columns(
        pl.col("date")
        .map_elements(lambda x: date(x // 10000, x // 100 % 100, x % 100), return_dtype=pl.Date)
        .alias("date_d")
    )
    joined = sunny.join(spread, left_on="date_d", right_on="date", how="inner").filter(
        pl.col("date_d") >= ENGINE_START
    )
    d_pos = joined.filter(pl.col("dip"))["spread_40"].to_numpy() > 0
    n_pos = joined.filter(~pl.col("dip"))["spread_40"].to_numpy() > 0
    boots = [
        rng.choice(d_pos, len(d_pos)).mean() - rng.choice(n_pos, len(n_pos)).mean()
        for _ in range(5000)
    ]
    diffs = (table["engine_share"] - table["placebo_share"]).to_numpy()
    boot_diff = [rng.choice(diffs, len(diffs)).mean() for _ in range(5000)]
    summary = {
        "n_dip_days_with_engine_gate": table.height,
        "n_coincident": int(table["coincident"].sum()),
        "coincident_wilson": wilson(k=int(table["coincident"].sum()), n=table.height),
        "mean_engine_share_dip": float(table["engine_share"].mean()),
        "mean_engine_share_placebo": float(table["placebo_share"].mean()),
        "mean_engine_share_controls": float(table["control_share"].mean()),
        "dip_minus_placebo": float((table["engine_share"] - table["placebo_share"]).mean()),
        "dip_minus_placebo_ci": [
            float(np.percentile(boot_diff, 2.5)),
            float(np.percentile(boot_diff, 97.5)),
        ],
        "spread_positive_dip_days": float(d_pos.mean()),
        "spread_positive_other_sunny_days": float(n_pos.mean()),
        "spread_diff": float(d_pos.mean() - n_pos.mean()),
        "spread_diff_lo": float(np.percentile(boots, 2.5)),
        "spread_diff_hi": float(np.percentile(boots, 97.5)),
        "n_spread_days": [len(d_pos), len(n_pos)],
    }
    return table, summary


def recipients(*, site: str, rng: np.random.Generator) -> pl.DataFrame:
    """Return siblings' mean reading change over the dip half-hours against a baseline."""
    mats = {s: Mats(site=s) for s in (site, *SIBLINGS[site])}
    _, dips = dip_days(site=site)
    dip_set = set(dips["date"].to_list())
    main = mats[site]
    where = {int(d): i for i, d in enumerate(main.dates)}
    pool = np.array([int(d) not in dip_set for d in main.dates])
    out = []
    for sib in SIBLINGS[site]:
        m = mats[sib]
        sib_where = {int(d): i for i, d in enumerate(m.dates)}
        changes, placebo_means = [], []
        for d in dips["date"].to_list():
            i, j = where.get(int(d)), sib_where.get(int(d))
            if i is None or j is None:
                continue
            low = np.abs(main.p[i, MID]) < DIP_POINTS
            if not low.any():
                continue
            slots = np.arange(MID.start, MID.stop)[low]
            cand = np.where(pool & (main.weekend == main.weekend[i]))[0]
            cand = cand[cand != i]
            cand_dates = [int(main.dates[c]) for c in cand]
            keep = [c for c, dd in zip(cand, cand_dates, strict=True) if dd in sib_where]
            keep = np.array(keep)
            near = keep[
                np.argsort(np.abs(main.ordinal[keep] - main.ordinal[i]), kind="stable")[:N_BASE]
            ]
            base = np.median(
                np.stack([np.abs(m.p[sib_where[int(main.dates[c])], slots]) for c in near]), axis=0
            )
            change = float(np.nanmean(np.abs(m.p[j, slots]) - base))
            changes.append(change)
            # Placebo: the same statistic on one random non-dip day with the same slots.
            pc = int(rng.choice(near))
            placebo_means.append(
                float(np.nanmean(np.abs(m.p[sib_where[int(main.dates[pc])], slots]) - base))
            )
        arr = np.array(changes)
        arr = arr[np.isfinite(arr)]
        boots = [rng.choice(arr, len(arr)).mean() for _ in range(5000)]
        out.append(
            {
                "primary": site,
                "sibling": sib,
                "n_dip_days": len(arr),
                "mean_change_points": float(arr.mean()),
                "lo": float(np.percentile(boots, 2.5)),
                "hi": float(np.percentile(boots, 97.5)),
                "placebo_mean_change": float(np.nanmean(placebo_means)),
            }
        )
    return pl.DataFrame(out)


def labelled_overlap(*, site: str, dip_dates: list[int]) -> dict:
    """Count dip days outside every labelled real event (onset to end) that names the Primary."""
    series_id = int(data.series_table().filter(pl.col("site") == site)["time_series_id"].item())
    labels = pl.read_parquet(LABELLED).filter(
        pl.col("kind").str.starts_with("real"),
        pl.col("onset_utc").is_not_null(),
        (pl.col("side_a") == series_id) | (pl.col("side_b") == series_id),
    )
    inside = 0
    for d in dip_dates:
        start = datetime(int(d) // 10000, int(d) // 100 % 100, int(d) % 100, tzinfo=UTC)
        end = start.replace(hour=23, minute=59)
        if labels.filter((pl.col("onset_utc") <= end) & (pl.col("end_utc") >= start)).height:
            inside += 1
    return {
        "site": site,
        "dip_days": len(dip_dates),
        "inside_labelled_event": inside,
        "outside": len(dip_dates) - inside,
        "labelled_events_with_times": labels.height,
        "last_labelled_end": str(labels["end_utc"].max()) if labels.height else None,
    }


def main() -> None:
    """Run every Part B test and print the tables."""
    common.WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(30)
    out = {}
    expected = {"D8": 71, "D22": 13, "D12": 13}
    for site, n_expected in expected.items():
        n_found = dip_days(site=site)[1].height
        print(f"dip days {site}: found {n_found}, experiment 506 {n_expected}")
        if n_found != n_expected:
            msg = (
                f"{site}: dip-day count {n_found} does not reproduce experiment 506's {n_expected}"
            )
            raise SystemExit(msg)
    for site in ("D8", "D22"):
        midday, dips = dip_days(site=site)
        year_share = (
            midday.filter(pl.col("sky") == "sunny")
            .group_by("year")
            .agg(days=pl.len(), dips=pl.col("dip").sum(), share=pl.col("dip").mean())
            .sort("year")
        )
        print(f"== {site}: sunny middays, dip days {dips.height}")
        print(year_share)
        out[f"{site}_labelled"] = labelled_overlap(site=site, dip_dates=dips["date"].to_list())
    print(json.dumps(out, indent=1))

    shape = pl.concat(
        [shape_test(site="D8", rng=rng), shape_test(site="D22", rng=rng), d10_calibration()]
    )
    shape.write_parquet(common.WORK / "partB_shape.parquet")
    print("== shape test: share of days the solar shape beats a boxcar (BIC)")
    print(
        shape.group_by("site", "kind")
        .agg(
            n=pl.len(),
            solar_better=pl.col("label_solar").sum(),
            share_solar=pl.col("label_solar").mean(),
            r2_solar=pl.col("r2_solar").median(),
            r2_box=pl.col("r2_box").median(),
        )
        .sort("site", "kind")
    )
    placebo = pl.concat([sky_placebo(site=s, rng=rng) for s in ("D8", "D22")], how="diagonal")
    placebo.write_parquet(common.WORK / "partB_sky_placebo.parquet")
    print("== March to October: dip rate on sunny and cloudy middays")
    print(placebo)
    table, summary = engine_test(rng=rng)
    table.write_parquet(common.WORK / "partB_engine_d8.parquet")
    print("== D8 engine states and spark spread")
    print(json.dumps(summary, indent=1))
    for site in ("D8", "D22"):
        rec = recipients(site=site, rng=rng)
        rec.write_parquet(common.WORK / f"partB_recipients_{site}.parquet")
        print(f"== recipients of {site} (low power)")
        print(rec)
    inj = shape.filter(pl.col("kind").str.starts_with("injected")).drop_nulls("label_solar")
    correct = {
        kind: float(
            inj.filter(pl.col("kind") == f"injected_{kind}")["label_solar"].mean()
            if kind == "solar"
            else 1 - inj.filter(pl.col("kind") == f"injected_{kind}")["label_solar"].mean()
        )
        for kind in ("solar", "boxcar")
    }
    shape_ok = min(correct.values()) >= SHAPE_ACCURACY
    print(f"== shape calibration: share correct {correct}; usable for classification: {shape_ok}")
    d8_cause = table.with_columns(
        engine=pl.col("coincident") & (pl.col("engine_share") > pl.col("placebo_share"))
    ).join(
        shape.filter(pl.col("site") == "D8", pl.col("kind") == "dip").select("date", "label_solar"),
        on="date",
        how="left",
    )
    for site in ("D8", "D22"):
        if site == "D8":
            frame = d8_cause
        else:
            frame = (
                shape.filter(pl.col("site") == "D22", pl.col("kind") == "dip")
                .select("date", "label_solar")
                .with_columns(engine=pl.lit(False))
            )
        cause = frame.with_columns(
            cause=pl.when(pl.col("engine"))
            .then(pl.lit("engine running"))
            .when(~pl.lit(shape_ok))
            .then(pl.lit("shape cannot separate solar from load transfer"))
            .when(pl.col("label_solar") == 1)
            .then(pl.lit("solar-consistent, no engine flagged"))
            .when(pl.col("label_solar") == 0)
            .then(pl.lit("load-transfer-consistent, no engine flagged"))
            .otherwise(pl.lit("unclassified"))
        )["cause"].value_counts()
        print(f"== {site} dip days by cause class (D22: the engine class is not assessable)")
        print(cause)


if __name__ == "__main__":
    main()

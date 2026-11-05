# ruff: noqa: E501, D103, B023, PERF401
"""Experiment 581: tail metrics by phase on the lag-only injected bank.

For each arm, month, pass, and Primary (one injected event), read the forecast made with the
contaminated lags (`inject/`), the same weights' forecast with clean lags (`fits/V3D5/`, the
"uninjected twin"), and the cleaned power at the target time (the NRA reference: the injected shift
is added to the lag inputs only). Quantiles are the mean over the two seeds. Per event, phase, and
lead band the metrics are:

- follow ratio at tau = mean over rows of (q_tau injected - q_tau twin) x sign / depth (positive means
  moved towards the event); the "free" version uses the lag-free D5 network (experiment 565) as the
  reference instead of the twin;
- coverage of the NRA power inside q05-q95 and q01-q99; the one-sided share on the NRA side (y >= q05
  when the event raised load, y <= q95 when it lowered it; the same at q01 and q99);
- pinball loss at tau in 0.01, 0.05, 0.95, 0.99; mean width q95 - q05 and q99 - q01.

Phases are by forecast origin relative to the event: `before` (7 days to onset), `d0-3`, `d4-end`,
`after0-7`, `after7-14`. Capacity fractions throughout.
"""

import sys
from pathlib import Path

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_bank as bank
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00581-robust-multiweek-lag-encoder-x3"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
W572 = ML_EXPERIMENTS_DATA_DIR / "00572-two-offset-part-1-lags-versus-the-slow-output-offset/work"
W565 = k578.d6.W
MONTHS = k578.MONTHS
EXTRA_MONTHS = ("2026-05",)
"""Extra fold months covering the verified transfers (fitted for v3d5 and x3med only)."""
SEEDS = (0, 1)
TAUS = {"q01": 0.01, "q05": 0.05, "q50": 0.5, "q95": 0.95, "q99": 0.99}
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
PHASES = ("before", "d0-3", "d4-end", "after0-7", "after7-14")
BANDS = {"0-24h": (0.0, 24.0), "24-120h": (24.0, 120.0)}


W582 = ML_EXPERIMENTS_DATA_DIR / "00582-nra-tails-on-verified-load-shift-events/work"


def arm_root(arm: str, month: str = "") -> Path:
    """Return the work folder of an arm: `v3d5` is experiment 572's, or experiment 582's for the extra months."""
    if arm == "v3d5":
        return W582 if month in EXTRA_MONTHS else W572
    return WORK / "arms" / arm / "work"


def inject_dir(arm: str) -> Path:
    return (
        (WORK / "arms" / arm / "work" / "inject")
        if arm != "v3d5"
        else WORK / "arms" / "v3d5" / "work" / "inject"
    )


def seed_mean(*, paths: list[Path]) -> pl.DataFrame:
    """Return the per-row mean of the quantile columns over the seeds' files (keys joined)."""
    frames = [pl.read_parquet(p, columns=[*KEYS, *TAUS]) for p in paths]
    return pl.concat(frames).group_by(KEYS).agg([pl.col(q).mean() for q in TAUS])


def pinball(*, y: pl.Expr, q: pl.Expr, tau: float) -> pl.Expr:
    return pl.max_horizontal(tau * (y - q), (tau - 1.0) * (y - q))


def event_table(
    *,
    arm: str,
    month: str,
    r: int,
    sites: pl.DataFrame,
    clean: np.ndarray,
    free: pl.DataFrame | None,
) -> pl.DataFrame:
    """Return per-(Primary, phase, band) metrics of pass `r` of `month` for the arm."""
    root = arm_root(arm)
    cuts = ("d0", "d14")
    inj = seed_mean(
        paths=[
            inject_dir(arm) / f"{month}_r{r:02d}_{d}_s{s}.parquet"
            for d in cuts
            for s in SEEDS
            if (inject_dir(arm) / f"{month}_r{r:02d}_{d}_s{s}.parquet").exists()
        ]
    )
    twin = seed_mean(
        paths=[root / "fits" / "V3D5" / f"{month}_{d}_s{s}.parquet" for d in cuts for s in SEEDS]
    )
    j = inj.join(twin, on=KEYS, suffix="_t")
    if free is not None:
        j = j.join(free.rename({q: f"{q}_f" for q in TAUS}), on=KEYS, how="left")
    prim_of = dict(zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True))
    start = k578.c.month_start(month=month)
    start_idx = (k578.c.us(t=start) - k578.prep.GRID_START_US) // k578.prep.STEP_US
    end_idx = start_idx + bank.END_DAY * bank.PER_DAY
    prim = np.array([prim_of[t] for t in j["time_series_id"].to_list()])
    v_idx = (
        (j["valid_time"].dt.epoch("us").to_numpy() - k578.prep.GRID_START_US) // k578.prep.STEP_US
    ).astype(np.int64)
    o_idx = (
        (j["power_fcst_init_time"].dt.epoch("us").to_numpy() - k578.prep.GRID_START_US)
        // k578.prep.STEP_US
    ).astype(np.int64)
    ev = np.array([bank.event(primary=p, r=r) for p in range(clean.shape[0])])
    depth, length, sign = ev[prim, 0], ev[prim, 1], ev[prim, 2]
    onset = end_idx - length * bank.PER_DAY
    d_on = (o_idx - onset) / bank.PER_DAY
    d_end = (o_idx - end_idx) / bank.PER_DAY
    phase = np.full(len(j), "", dtype=object)
    phase[(d_on >= -7) & (d_on < 0)] = "before"
    phase[(d_on >= 0) & (d_on < 4) & (d_end < 0)] = "d0-3"
    phase[(d_on >= 4) & (d_end < 0)] = "d4-end"
    phase[(d_end >= 0) & (d_end < 7)] = "after0-7"
    phase[(d_end >= 7) & (d_end < 14)] = "after7-14"
    y = clean[prim, v_idx]
    lead = (j["valid_time"] - j["power_fcst_init_time"]).dt.total_hours().to_numpy()
    df = j.with_columns(
        primary=pl.Series(prim), phase=pl.Series(phase), y=pl.Series(y), lead=pl.Series(lead),
        sgn=pl.Series(sign.astype(np.float64)), depth=pl.Series(np.abs(depth)),
        length_days=pl.Series(length), pass_r=pl.lit(r),
    ).filter((pl.col("phase") != "") & pl.col("y").is_finite())  # fmt: skip
    return _aggregate(df=df, free=free is not None).with_columns(
        month=pl.lit(month), pass_r=pl.lit(r), arm=pl.lit(arm)
    )


def _aggregate(*, df: pl.DataFrame, free: bool) -> pl.DataFrame:
    """Return per-(Primary, phase, band) means of the row metrics."""
    one_lo = pl.when(pl.col("sgn") > 0).then(pl.col("y") >= pl.col("q05")).otherwise(True)
    one_hi = pl.when(pl.col("sgn") < 0).then(pl.col("y") <= pl.col("q95")).otherwise(True)
    one_lo99 = pl.when(pl.col("sgn") > 0).then(pl.col("y") >= pl.col("q01")).otherwise(True)
    one_hi99 = pl.when(pl.col("sgn") < 0).then(pl.col("y") <= pl.col("q99")).otherwise(True)
    exprs = [
        ((pl.col("y") >= pl.col("q05")) & (pl.col("y") <= pl.col("q95")))
        .cast(pl.Float64)
        .alias("cov90"),
        ((pl.col("y") >= pl.col("q01")) & (pl.col("y") <= pl.col("q99")))
        .cast(pl.Float64)
        .alias("cov98"),
        (one_lo & one_hi).cast(pl.Float64).alias("nra_side95"),
        (one_lo99 & one_hi99).cast(pl.Float64).alias("nra_side99"),
        (pl.col("q95") - pl.col("q05")).alias("width90"),
        (pl.col("q99") - pl.col("q01")).alias("width98"),
        (pl.col("q95_t") - pl.col("q05_t")).alias("width90_twin"),
        ((pl.col("y") >= pl.col("q05_t")) & (pl.col("y") <= pl.col("q95_t")))
        .cast(pl.Float64)
        .alias("cov90_twin"),
        ((pl.col("q50_t") >= pl.col("q05")) & (pl.col("q50_t") <= pl.col("q95")))
        .cast(pl.Float64)
        .alias("twin_median_inside"),
    ]
    for name, tau in TAUS.items():
        exprs.append(
            (((pl.col(name) - pl.col(f"{name}_t")) * pl.col("sgn")) / pl.col("depth")).alias(
                f"follow_{name}"
            )
        )
        if free:
            exprs.append(
                (((pl.col(name) - pl.col(f"{name}_f")) * pl.col("sgn")) / pl.col("depth")).alias(
                    f"follow_free_{name}"
                )
            )
        if name != "q50":
            exprs.append(pinball(y=pl.col("y"), q=pl.col(name), tau=tau).alias(f"pin_{name}"))
            exprs.append(
                pinball(y=pl.col("y"), q=pl.col(f"{name}_t"), tau=tau).alias(f"pin_{name}_twin")
            )
    df = df.with_columns(*exprs)
    out = []
    for band, (lo, hi) in BANDS.items():
        sub = df.filter((pl.col("lead") >= lo) & (pl.col("lead") < hi))
        agg = (
            sub.group_by("primary", "phase")
            .agg(
                pl.len().alias("n"),
                pl.first("depth"),
                pl.first("length_days"),
                pl.first("sgn"),
                *[
                    pl.col(c).mean()
                    for c in df.columns
                    if c.startswith(("follow_", "pin_", "cov", "nra_side", "width", "twin_median"))
                ],
            )
            .with_columns(band=pl.lit(band))
        )
        out.append(agg)
    return pl.concat(out)


def real_table(
    *,
    arm: str,
    month: str,
    sites: pl.DataFrame,
    clean: np.ndarray,
    events: list[dict],
    free: pl.DataFrame,
) -> pl.DataFrame:
    """Return per-(event, phase, band) metrics of the verified real events for the arm in `month`.

    The forecast uses the observed lags; the twin is the same weights with the lags taken from the
    power with every verified event removed (pass `REAL_PASS`); the NRA reference is that corrected
    power (observed minus the fitted depth).
    """
    root = arm_root(arm, month)
    inj_dir = inject_dir(arm)
    fixed = bank.corrected_clean(clean=clean, events=events)
    obs = seed_mean(
        paths=[
            root / "fits" / "V3D5" / f"{month}_{d}_s{s}.parquet"
            for d in ("d0", "d14")
            for s in SEEDS
        ]
    )
    tw_paths = [
        inj_dir / f"{month}_r{bank.REAL_PASS}_{d}_s{s}.parquet"
        for d in ("d0", "d14")
        for s in SEEDS
    ]
    twin = seed_mean(paths=tw_paths)
    j = obs.join(twin, on=KEYS, suffix="_t").join(
        free.rename({q: f"{q}_f" for q in TAUS}), on=KEYS, how="left"
    )
    prim_of = dict(zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True))
    prim = np.array([prim_of[t] for t in j["time_series_id"].to_list()])
    v_idx = (
        (j["valid_time"].dt.epoch("us").to_numpy() - k578.prep.GRID_START_US) // k578.prep.STEP_US
    ).astype(np.int64)
    o_idx = (
        (j["power_fcst_init_time"].dt.epoch("us").to_numpy() - k578.prep.GRID_START_US)
        // k578.prep.STEP_US
    ).astype(np.int64)
    lead = (j["valid_time"] - j["power_fcst_init_time"]).dt.total_hours().to_numpy()
    tabs = []
    for q in events:
        sel = prim == q["primary"]
        if not sel.any():
            continue
        d_on = (o_idx[sel] - q["onset_idx"]) / bank.PER_DAY
        d_end = (o_idx[sel] - q["end_idx"]) / bank.PER_DAY
        phase = np.full(sel.sum(), "", dtype=object)
        phase[(d_on >= -7) & (d_on < 0)] = "before"
        phase[(d_on >= 0) & (d_on < 4) & (d_end < 0)] = "d0-3"
        phase[(d_on >= 4) & (d_end < 0)] = "d4-end"
        phase[(d_end >= 0) & (d_end < 7)] = "after0-7"
        phase[(d_end >= 7) & (d_end < 14)] = "after7-14"
        if not (phase != "").any():
            continue
        df = j.filter(pl.Series(sel)).with_columns(
            primary=pl.lit(q["primary"]), phase=pl.Series(phase), y=pl.Series(fixed[q["primary"], v_idx[sel]]),
            lead=pl.Series(lead[sel]), sgn=pl.lit(float(np.sign(q["depth"]))), depth=pl.lit(abs(q["depth"])),
            length_days=pl.lit(float(q["length_days"])),
        ).filter((pl.col("phase") != "") & pl.col("y").is_finite())  # fmt: skip
        if df.height:
            tabs.append(
                _aggregate(df=df, free=True).with_columns(
                    transfer_id=pl.lit(q["transfer_id"]),
                    side=pl.lit(q["side"]),
                    label=pl.lit(q["label"]),
                    month=pl.lit(month),
                    arm=pl.lit(arm),
                )
            )
    return pl.concat(tabs) if tabs else pl.DataFrame()


def event_free(
    *, arm: str, sites: pl.DataFrame, clean: np.ndarray, events: list[dict]
) -> pl.DataFrame:
    """Return calibration, pinball, and width on rows with no verified or labelled event within 7 days.

    Rows are forecasts with the observed lags at 0-24 h and 24-120 h; a row is excluded when its origin
    or valid time lies from 7 days before onset to 14 days after the end of any verified event, or from
    7 days before onset to 7 days after the end of a labelled event, at that Primary.
    """
    lab = pl.read_parquet(k578.EVENTS)
    prim_of = dict(zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True))
    spans: dict[int, list[tuple[int, int]]] = {}
    for q in events:
        spans.setdefault(q["primary"], []).append((q["onset_idx"] - 7 * 48, q["end_idx"] + 14 * 48))
    for q in lab.iter_rows(named=True):
        spans.setdefault(prim_of[q["time_series_id"]], []).append(
            (q["onset_idx"] - 7 * 48, q["end_idx"] + 7 * 48)
        )
    tabs = []
    for m in MONTHS:
        df = seed_mean(
            paths=[
                arm_root(arm, m) / "fits" / "V3D5" / f"{m}_{d}_s{s}.parquet"
                for d in ("d0", "d14")
                for s in SEEDS
            ]
        )
        prim = np.array([prim_of[t] for t in df["time_series_id"].to_list()])
        v = (
            (df["valid_time"].dt.epoch("us").to_numpy() - k578.prep.GRID_START_US)
            // k578.prep.STEP_US
        ).astype(np.int64)
        o = (
            (df["power_fcst_init_time"].dt.epoch("us").to_numpy() - k578.prep.GRID_START_US)
            // k578.prep.STEP_US
        ).astype(np.int64)
        bad = np.zeros(len(df), dtype=bool)
        for p, sp in spans.items():
            for lo, hi in sp:
                bad |= (prim == p) & (((o >= lo) & (o < hi)) | ((v >= lo) & (v < hi)))
        y = clean[prim, v]
        lead = (df["valid_time"] - df["power_fcst_init_time"]).dt.total_hours().to_numpy()
        d = df.with_columns(
            primary=pl.Series(prim),
            y=pl.Series(y),
            lead=pl.Series(lead),
            bad=pl.Series(bad),
            month=pl.lit(m),
        ).filter(~pl.col("bad") & pl.col("y").is_finite())
        tabs.append(d.drop("bad"))
    d = pl.concat(tabs).join(sites.select("primary", "group"), on="primary")
    out = []
    for band, (lo, hi) in BANDS.items():
        sub = d.filter((pl.col("lead") >= lo) & (pl.col("lead") < hi))
        for grp, g in [("all", sub), *[(k[0], x) for k, x in sub.group_by("group")]]:
            row = {"arm": arm, "band": band, "group": grp, "n": g.height,
                   "cov90": float(((g["y"] >= g["q05"]) & (g["y"] <= g["q95"])).mean()),
                   "cov98": float(((g["y"] >= g["q01"]) & (g["y"] <= g["q99"])).mean()),
                   "width90": float((g["q95"] - g["q05"]).mean()), "width98": float((g["q99"] - g["q01"]).mean())}  # fmt: skip
            for name, tau in TAUS.items():
                if name != "q50":
                    row[f"below_{name}"] = float((g["y"] <= g[name]).mean())
                    row[f"pin_{name}"] = float(
                        g.select(pinball(y=pl.col("y"), q=pl.col(name), tau=tau)).to_series().mean()
                    )
            out.append(row)
    return pl.DataFrame(out)


def main() -> None:
    arms = sys.argv[1:] or ["v3d5", "x3med"]
    sites = k578.sites()
    clean = k578.clean()
    out_dir = WORK / "tails"
    out_dir.mkdir(parents=True, exist_ok=True)
    free_all = {}
    for m in (*MONTHS, *EXTRA_MONTHS):
        if m in EXTRA_MONTHS:
            free_all[m] = pl.DataFrame(
                schema={
                    "time_series_id": pl.Int32,
                    "power_fcst_init_time": pl.Datetime("us", "UTC"),
                    "valid_time": pl.Datetime("us", "UTC"),
                    **dict.fromkeys(TAUS, pl.Float32),
                }
            )
            continue
        parts = [
            pl.read_parquet(W565 / "fits" / "D5" / f"{m}_{d}_s{s}.parquet", columns=[*KEYS, *TAUS])
            for d in ("d0", "d14")
            for s in SEEDS
        ]
        free_all[m] = pl.concat(parts).group_by(KEYS).agg([pl.col(q).mean() for q in TAUS])
    sm = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    verified = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00580-verified-load-shift-event-set-and-label-audit/work/events_verified.parquet"
    )
    events = bank.verified_events(
        capacity_mw=dict(zip(sm["label"].to_list(), sm["capacity_mw"].to_list(), strict=True)),
        label_to_primary=dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True)),
        table=verified,
    )
    for arm in arms:
        for name, fn in (
            (
                "real",
                lambda: pl.concat(
                    [
                        t
                        for t in (
                            real_table(
                                arm=arm,
                                month=m,
                                sites=sites,
                                clean=clean,
                                events=events,
                                free=free_all[m],
                            )
                            for m in (
                                *MONTHS,
                                *[
                                    x
                                    for x in EXTRA_MONTHS
                                    if (
                                        arm_root(arm, x) / "fits" / "V3D5" / f"{x}_d14_s1.parquet"
                                    ).exists()
                                    and (
                                        inject_dir(arm) / f"{x}_r{bank.REAL_PASS}_d14_s1.parquet"
                                    ).exists()
                                ],
                            )
                        )
                        if t.height
                    ]
                ),
            ),
            ("free", lambda: event_free(arm=arm, sites=sites, clean=clean, events=events)),
        ):
            tgt = out_dir / f"{name}_{arm}.parquet"
            if not tgt.exists():
                tmp = tgt.with_suffix(".tmp")
                fn().write_parquet(tmp)
                tmp.rename(tgt)
    for arm in arms:
        target = out_dir / f"events_{arm}.parquet"
        if target.exists():
            continue
        tabs = []
        for m in MONTHS:
            for r in range(bank.N_PASSES):
                tabs.append(
                    event_table(arm=arm, month=m, r=r, sites=sites, clean=clean, free=free_all[m])
                )
            print(arm, m, "done", flush=True)
        t = pl.concat(tabs)
        tmp = target.with_suffix(".tmp")
        t.write_parquet(tmp)
        tmp.rename(target)


if __name__ == "__main__":
    main()

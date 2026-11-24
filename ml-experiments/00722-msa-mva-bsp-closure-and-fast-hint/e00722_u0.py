# ruff: noqa: E501
"""Experiment 722, job 1 (rung U0): topology, generator classes, add-back checks, class audit.

Run it only through `run.sh --go e00722_u0.py`, after the Opus code review. The order of reads is
fixed and logged: CAMS and the farms' output first, then the class audit and the same-sky pairs,
and only then any Primary's reading, the BSP meters, or a generator. `u0_checks.json` records the
read order, and `ReadLog.assert_audit_before_primary_reads` raises if a Primary was read earlier.

Outputs under `data/ml-experiments/00722-.../`: `u0_topology.parquet`,
`u0_generator_classes.parquet`, `u0_checks.json`, `u0_era_split.csv`, `u0_same_sky_pairs.csv`,
`u0_class_audit.csv`, and `run_log_u0.json`; dated detail goes under `PRIVATE/`. Stdout prints
counts, shares, ratios, and intervals only.
"""

import os
from pathlib import Path
from typing import Any, Final

import e00722_common as common
import e00722_core as core
import e00722_d8core as d8core
import e00722_data as data
import e00722_guard as guard
import e00722_sky as sky
import numpy as np
import polars as pl

FOLDER: Final[Path] = Path(__file__).resolve().parent
"""This experiment's code folder."""

SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The hosts of the same-sky pairs: the six signed-MW Primaries."""

ERA_GROUPS: Final[tuple[str, ...]] = ("net", "solar", "fade")
"""The three eras of D8's series in the stored BSP run."""

ERA_SPLIT_SERIES: Final[tuple[str, ...]] = ("meter", "r_noD8_6others_noWrangle")
"""The two rows of the era table of section 5.3 (raw meter, and meter minus the six others)."""

M_NULL: Final[int] = 20
"""Null draws per event inside each bootstrap replicate (as in the stored BSP run)."""


# ---------------------------------------------------------------------------------------------
# Pure helpers (tested on synthetic arrays)
# ---------------------------------------------------------------------------------------------
def same_sky_pair_table(
    *,
    k_of: dict[str, np.ndarray],
    lit_of: dict[str, np.ndarray],
    hosts: tuple[str, ...],
    farms: list[str],
    metered_farms: list[str],
) -> pl.DataFrame:
    """Return the correlation of fast clear-sky-index changes for every host and farm pair.

    Every pair is computed. The column `farm_metered_under_a_bsp` flags the pairs whose farm is a
    metered generator under BSP A or BSP B, so that the pair rule can count the unmetered farms
    alone and report the metered pairs apart.
    """
    pair_rows = []
    for host in hosts:
        for farm in farms:
            r, n = sky.correlation_of_changes(
                a=sky.fast_k_change(k=k_of[host], lit=lit_of[host]),
                b=sky.fast_k_change(k=k_of[farm], lit=lit_of[farm]),
            )
            pair_rows.append(
                {
                    "host": host,
                    "farm": farm,
                    "r": r,
                    "n_rows": n,
                    "same_sky": bool(r >= core.SAME_SKY_CORRELATION),
                    "farm_metered_under_a_bsp": farm in metered_farms,
                }
            )
    return pl.DataFrame(
        pair_rows,
        schema={
            "host": pl.String,
            "farm": pl.String,
            "r": pl.Float64,
            "n_rows": pl.Int64,
            "same_sky": pl.Boolean,
            "farm_metered_under_a_bsp": pl.Boolean,
        },
    )


def long_gaps(*, x: np.ndarray, min_len: int = core.LONG_GAP_HALF_HOURS) -> int:
    """Count runs of missing half-hours longer than `min_len` between the first and last reading."""
    finite = np.isfinite(x)
    if not finite.any():
        return 0
    idx = np.flatnonzero(finite)
    gaps = np.diff(idx) - 1
    return int((gaps > min_len).sum())


def generator_coverage(*, x: np.ndarray, span: np.ndarray) -> float:
    """Return the share of the rows of `span` where the series has a value."""
    return float(np.isfinite(x[span]).mean()) if span.any() else float("nan")


def identify_engine_series(*, record: np.ndarray, gens: dict[int, np.ndarray]) -> int | None:
    """Return the id of the generator series that equals the engine record, or None.

    A generator is the engine's own series when its readings correlate with the record at
    `ENGINE_IDENTITY_CORR` or more on the record's half-hours. Subtracting that series would
    cancel the engine signal that job 2 looks for.
    """
    best: tuple[float, int] | None = None
    for ts, x in gens.items():
        ok = np.isfinite(record) & np.isfinite(x)
        if ok.sum() < core.MIN_PAIR_ROWS or record[ok].std() == 0 or x[ok].std() == 0:
            continue
        c = float(np.corrcoef(record[ok], x[ok])[0, 1])
        if c >= core.ENGINE_IDENTITY_CORR and (best is None or c > best[0]):
            best = (c, ts)
    return None if best is None else best[1]


def era_split_from_results(*, results: pl.DataFrame) -> pl.DataFrame:
    """Recompute the era table of section 5.3 from the stored event-study CSV.

    The ratio is each series' excess step over matched quiet times divided by D8's own excess step
    in the same event group, with the series' 95% interval divided by the same figure. D8's own
    uncertainty is not propagated.
    """
    d8 = results.filter(pl.col("series") == "d8_reading").select(
        "events", d8_excess=pl.col("excess")
    )
    keep = [f"{kind}_iso_{era}" for kind in ("on", "off") for era in ERA_GROUPS]
    keep += ["all_iso_per_unit", "on_iso", "off_iso"]
    return (
        results.filter(pl.col("series").is_in(ERA_SPLIT_SERIES) & pl.col("events").is_in(keep))
        .join(d8, on="events", how="inner")
        .with_columns(
            ratio=pl.col("excess") / pl.col("d8_excess"),
            a=pl.col("excess_ci_lo") / pl.col("d8_excess"),
            b=pl.col("excess_ci_hi") / pl.col("d8_excess"),
        )
        .select(
            "series",
            "events",
            "n_events",
            "ratio",
            ratio_lo=pl.min_horizontal("a", "b"),
            ratio_hi=pl.max_horizontal("a", "b"),
        )
        .sort("series", "events")
    )


def switching_windows_per_primary(
    *, sheets: dict[str, pl.DataFrame], numbers: dict[str, int]
) -> pl.DataFrame:
    """Count the switching-log windows of each Primary from whatever sheets the workbook holds.

    A sheet is used when it has a column whose name holds "substation" and a column whose name
    holds "start", "from", or "begin". A window is a row of such a sheet whose substation number
    is the Primary's. A workbook with no such sheet gives zero windows and `readable` False.
    """
    counts = dict.fromkeys(numbers, 0)
    readable = False
    for frame in sheets.values():
        names = {c.lower(): c for c in frame.columns}
        sub = next((v for k, v in names.items() if "substation" in k and "name" not in k), None)
        beg = next(
            (v for k, v in names.items() if any(p in k for p in ("start", "from", "begin"))), None
        )
        if sub is None or beg is None:
            continue
        readable = True
        for label, number in numbers.items():
            counts[label] += int((frame[sub].cast(pl.String, strict=False) == str(number)).sum())
    return pl.DataFrame(
        {
            "label": list(counts),
            "n_windows": list(counts.values()),
            "readable": [readable] * len(counts),
        }
    )


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------
def sky_stage(*, topology: data.Topology, log: guard.ReadLog) -> dict[str, Any]:
    """Run the class audit and the same-sky pairs from CAMS and the farms' output alone."""
    primaries_sky, farm_sky, farm_cap, farm_id, n_after = data.load_sky(log=log)
    n_days = (core.N_GRID + core.HALF_HOURS_PER_DAY - 1) // core.HALF_HOURS_PER_DAY
    month = sky.month_of_day(n_days=n_days)
    label_of = dict(
        zip(topology.primaries["time_series_id"], topology.primaries["label"], strict=True)
    )
    sites: dict[str, data.Sky] = {
        label_of[ts]: s for ts, s in primaries_sky.items() if ts in label_of
    }
    sites.update(farm_sky)
    classes10: dict[str, np.ndarray] = {}
    classes15: dict[str, np.ndarray] = {}
    has_data: dict[str, np.ndarray] = {}
    k_of: dict[str, np.ndarray] = {}
    lit_of: dict[str, np.ndarray] = {}
    for name, s in sites.items():
        k = sky.clear_sky_index(ghi=s.ghi, ghi_clear=s.ghi_clear)
        lit10 = sky.daylight(zenith_deg=s.zenith, min_elevation_deg=core.DAYLIGHT_ELEVATION_DEG)
        lit15 = sky.daylight(zenith_deg=s.zenith, min_elevation_deg=core.ELEVATION_SENSITIVITY_DEG)
        classes10[name] = sky.classify_days(k=k, lit=lit10, n_days=n_days)
        classes15[name] = sky.classify_days(k=k, lit=lit15, n_days=n_days)
        has_data[name] = sky.days_with_data(k=k, lit=lit10, n_days=n_days)
        k_of[name], lit_of[name] = k, lit10
    rows: list[sky.AuditRow] = []
    for label in topology.primaries["label"].to_list():
        if label in sites:
            rows.append(
                sky.audit_season_share(
                    classes=classes10[label], month=month, has_data=has_data[label], scope=label
                )
            )
        else:
            rows.append(
                sky.AuditRow(3, label, float("nan"), "CAMS at every Primary", False, True, 0)
            )
    for name in farm_sky:
        power = farm_sky[name].power
        assert power is not None
        n_big, coverage = sky.count_big_changes(
            output_mw=power, capacity_mw=farm_cap[name], lit=lit_of[name], n_days=n_days
        )
        rows += sky.audit_farm(classes=classes10[name], n_big=n_big, coverage=coverage, scope=name)
    all_has = np.concatenate([has_data[n] for n in sites])
    rows.append(
        sky.audit_elevation_sensitivity(
            classes_10=np.concatenate([classes10[n] for n in sites]),
            classes_15=np.concatenate([classes15[n] for n in sites]),
            has_data=all_has,
            scope="all_sites",
        )
    )
    per_site = [
        sky.audit_elevation_sensitivity(
            classes_10=classes10[n], classes_15=classes15[n], has_data=has_data[n], scope=n
        )
        for n in sites
    ]
    per_site = [
        sky.AuditRow(41, r.scope, r.value, r.threshold, r.passed, r.readable, r.n_days)
        for r in per_site
    ]
    audit = pl.DataFrame([r.__dict__ for r in [*rows, *per_site]]).rename({"n_days": "n_days_used"})
    audit_ok = sky.audit_passes(rows=rows)
    # Same-sky pairs: every host and farm pair, with the farms that are metered under a BSP flagged.
    metered_ids = set(topology.generators["time_series_id"].to_list())
    metered_farms = sorted(name for name, ts in farm_id.items() if ts in metered_ids)
    pairs = same_sky_pair_table(
        k_of=k_of,
        lit_of=lit_of,
        hosts=SIGNED_SIX,
        farms=sorted(farm_sky),
        metered_farms=metered_farms,
    )
    du_of = {
        label: sky.fill_sky_change(
            du=sky.fast_irradiance_change(
                u=s.u, u_clear=s.u_clear, k=k_of[label], lit=lit_of[label], n_days=n_days
            ),
            zenith=s.zenith,
        )
        for label, s in sites.items()
        if label in set(topology.primaries["label"].to_list())
    }
    return {
        "audit": audit,
        "audit_ok": audit_ok,
        "pairs": pairs,
        "n_pairs": int(pairs["same_sky"].sum()),
        "n_pairs_unmetered": int(
            pairs.filter(~pl.col("farm_metered_under_a_bsp"))["same_sky"].sum()
        ),
        "du_of": du_of,
        "metered_farms": metered_farms,
        "farm_id": farm_id,
        "n_after_cams": n_after,
        "sites": sites,
        "k": k_of,
        "lit": lit_of,
        "n_days": n_days,
        "n_audit_rows_failed": int(sum(1 for r in rows if r.readable and not r.passed)),
    }


def addback_stage(
    *,
    topology: data.Topology,
    eng: data.EngineFrame,
    series: dict[str, np.ndarray],
    bsp_series: dict[str, np.ndarray],
    gens: dict[int, np.ndarray],
    record: np.ndarray,
    du_of: dict[str, np.ndarray],
    engine_id: int | None,
    labels: dict[int, str],
    n_days: int,
    gate: guard.TimingGate,
) -> pl.DataFrame:
    """Regress each Primary's half-hour change on each generator's change, by era (F1).

    Both changes are de-meaned at matched quiet times, and rows within two half-hours of a change
    in the engine state are dropped. The regression goes through the origin with a whole-day
    bootstrap that shares its resampled days across every pair. D8's three eras use the engine
    frame's rows; every other Primary uses the whole grid, because its regression does not need
    the engine's state.

    Each row also carries report-only columns: a second fit of the same Primary on every
    well-covered generator of its BSP and on the Primary's own fast sky change together (B2),
    the sign of each generator's change against the change of its BSP meter minus every Primary
    (B5), and whether the generator could be the engine's own series (B4).
    """
    rng = np.random.default_rng(core.SEED)
    weights = core.day_weights(n_days=n_days, n_boot=core.N_BOOT, rng=rng)
    day = sky.day_of(n=core.N_GRID)
    quiet = core.quiet_mask(engine_change=eng.change, calendar_ok=eng.has_row)
    drop = core.within_distance(flag=eng.change, distance=core.ENGINE_STEP_DROP_HALF_HOURS)

    def demeaned(x: np.ndarray) -> np.ndarray:
        return core.demean_matched(x=core.change(x=x, lag=1), key=eng.key, quiet=quiet)

    def rows_of(*, label: str, era: str) -> np.ndarray:
        if label == "D8" and era != "all":
            return eng.has_row & ~drop & (eng.era == era)
        return ~drop

    d_prim = {label: demeaned(x) for label, x in series.items()}
    d_gen = {ts: demeaned(x) for ts, x in gens.items()}
    d_sky = {
        label: core.demean_matched(x=du, key=eng.key, quiet=quiet) for label, du in du_of.items()
    }
    cov = {ts: generator_coverage(x=x, span=eng.has_row) for ts, x in gens.items()}
    meta = topology.generators
    bsp_of = dict(zip(meta["time_series_id"], meta["bsp"], strict=True))
    # B4 and B5: per-generator checks that do not depend on a Primary.
    engine_corr: dict[int, tuple[float, int]] = {}
    sign_corr: dict[int, float] = {}
    for ts, x in gens.items():
        engine_corr[ts] = sky.correlation_of_changes(
            a=core.change(x=x, lag=1), b=core.change(x=record, lag=1)
        )
        members = [
            series[p] for p in topology.primaries.filter(pl.col("bsp") == bsp_of[ts])["label"]
        ]
        rest = bsp_series[bsp_of[ts]] - np.sum(np.stack(members), axis=0)
        sign_corr[ts] = sky.correlation_of_changes(
            a=core.change(x=x, lag=1), b=core.change(x=rest, lag=1)
        )[0]
    # B2: the joint fit of each Primary, once per Primary and era.
    joint: dict[tuple[str, str], core.Fit] = {}
    rows: list[dict[str, Any]] = []
    n_done = 0
    n_total = meta.height * topology.primaries.height
    gate.mark()
    for g in meta.iter_rows(named=True):
        ts, bsp = g["time_series_id"], g["bsp"]
        for p in topology.primaries.filter(pl.col("bsp") == bsp).iter_rows(named=True):
            label = p["label"]
            eras = [e for e in ERA_GROUPS if (eng.era == e).any()] if label == "D8" else ["all"]
            for era in eras:
                use = rows_of(label=label, era=era)
                fit = core.bootstrap_ols(
                    y=np.where(use, d_prim[label], np.nan),
                    x=np.where(use, d_gen[ts], np.nan),
                    day=day,
                    weights=weights,
                    names=["coef"],
                )
                c, _, lo, hi = fit.get("coef")
                if (label, era) not in joint:
                    members_ts = [
                        t
                        for t in gens
                        if bsp_of[t] == bsp and cov[t] >= core.MIN_GENERATOR_COVERAGE
                    ]
                    joint[(label, era)] = core.bootstrap_ols(
                        y=np.where(use, d_prim[label], np.nan),
                        x=np.column_stack(
                            [np.where(use, d_gen[t], np.nan) for t in members_ts]
                            + [np.where(use, d_sky[label], np.nan)]
                        ),
                        day=day,
                        weights=weights,
                        names=[labels[t] for t in members_ts] + ["sky"],
                    )
                jc, jlo, jhi = float("nan"), float("nan"), float("nan")
                if labels[ts] in joint[(label, era)].names:
                    jc, _, jlo, jhi = joint[(label, era)].get(labels[ts])
                rows.append(
                    {
                        "bsp": bsp,
                        "generator_id": ts,
                        "generator": labels[ts],
                        "generator_type": g["type"],
                        "units": g["units"],
                        "primary": label,
                        "primary_units": p["units"],
                        "era": era,
                        "coef": c,
                        "lo": lo,
                        "hi": hi,
                        "n_rows": fit.n_rows,
                        "label": core.classify_addback(lo=lo, hi=hi),
                        "joint_coef": jc,
                        "joint_lo": jlo,
                        "joint_hi": jhi,
                        "joint_label": core.classify_addback(lo=jlo, hi=jhi)
                        if np.isfinite(jlo + jhi)
                        else "not_fitted",
                        "is_engine": ts == engine_id,
                        "engine_corr": engine_corr[ts][0],
                        "engine_overlap_rows": engine_corr[ts][1],
                        "engine_like": core.engine_like(
                            corr=engine_corr[ts][0],
                            n_overlap=engine_corr[ts][1],
                            generator_type=g["type"],
                        ),
                        "sign_corr": sign_corr[ts],
                        "sign_ok": core.generation_sign_ok(corr=sign_corr[ts]),
                    }
                )
            n_done += 1
            if n_done == 1:
                gate.check(done=1, total=n_total)
    frame = pl.DataFrame(rows)
    classes = {
        ts: core.generator_class(labels=sub["label"].to_list())
        for (ts,), sub in frame.group_by("generator_id")
    }
    joint_classes = {
        ts: core.generator_class(
            labels=[x for x in sub["joint_label"].to_list() if x != "not_fitted"]
        )
        for (ts,), sub in frame.group_by("generator_id")
    }
    out = frame.with_columns(
        generator_class=pl.col("generator_id").replace_strict(classes, return_dtype=pl.String),
        joint_generator_class=pl.col("generator_id").replace_strict(
            joint_classes, return_dtype=pl.String
        ),
        coverage=pl.col("generator_id").replace_strict(cov, return_dtype=pl.Float64),
    ).with_columns(class_differs_joint=pl.col("generator_class") != pl.col("joint_generator_class"))
    subtract = [
        core.subtract_generator(
            cls=r["generator_class"],
            units=r["units"],
            coverage=r["coverage"],
            is_engine=r["is_engine"],
            engine_like=r["engine_like"],
            sign_ok=r["sign_ok"],
        )
        for r in out.iter_rows(named=True)
    ]
    return out.with_columns(subtract=pl.Series(subtract, dtype=pl.Boolean))


def co_step_by_type(
    *,
    topology: data.Topology,
    eng: data.EngineFrame,
    d8_reading: np.ndarray,
    gens: dict[int, np.ndarray],
    classes: pl.DataFrame,
    gate: guard.TimingGate,
) -> pl.DataFrame:
    """Return each generator type's excess step at D8's engine events, over D8's own excess step.

    The groups are every isolated event per unit, and the on and off events of each era. A type's
    series is the sum of its MW generators under BSP A that cover the span and are not the
    engine's own series, and it is NaN where any member is missing. Each ratio and its 95%
    interval divide the type's excess step (over matched quiet times) by D8's own excess step in
    the same group.
    """
    rng = np.random.default_rng(core.SEED + 1)
    h, dk, era, cand_ok = d8core.isolated_events(
        change=eng.change, k_path=eng.k_path, era=eng.era, has_row=eng.has_row, in_drop=eng.in_drop
    )
    cand = d8core.candidates_by_key(key=eng.key, ok=cand_ok)
    by_type: dict[str, list[np.ndarray]] = {}
    ok_rows = classes.filter(
        (pl.col("bsp") == "A")
        & (pl.col("units") == "MW")
        & ~pl.col("is_engine")
        & (pl.col("coverage") >= core.MIN_GENERATOR_COVERAGE)
    ).unique(subset=["generator_id"])
    for g in ok_rows.iter_rows(named=True):
        by_type.setdefault(g["generator_type"], []).append(gens[g["generator_id"]])
    series = {"d8_reading": d8_reading} | {
        f"type:{t}": np.sum(arrs, axis=0) for t, arrs in sorted(by_type.items())
    }
    steps = {name: d8core.step_vec(x=x) for name, x in series.items()}
    groups: dict[str, np.ndarray] = {"all_iso_per_unit": np.ones(len(h), dtype=bool)}
    for e in ERA_GROUPS:
        groups[f"on_iso_{e}"] = (dk > 0) & (era == e)
        groups[f"off_iso_{e}"] = (dk < 0) & (era == e)
    rows: list[dict[str, Any]] = []
    done = 0
    gate.mark()
    for gname, mask in groups.items():
        scale = gname == "all_iso_per_unit"
        res = {
            name: d8core.study_core(
                step=step,
                events=h[mask],
                sign=dk[mask],
                scale=scale,
                key_of=eng.key,
                cand_by_key=cand,
                rng=rng,
                n_boot=core.N_BOOT,
                m_null=M_NULL,
            )
            for name, step in steps.items()
        }
        done += 1
        gate.check(done=done, total=len(groups))
        base = res.get("d8_reading")
        if base is None or base["excess"] == 0:
            continue
        for name, r in res.items():
            if r is None or name == "d8_reading":
                continue
            a, b = r["excess_lo"] / base["excess"], r["excess_hi"] / base["excess"]
            rows.append(
                {
                    "series": name,
                    "events": gname,
                    "n_events": r["n_events"],
                    "ratio": r["excess"] / base["excess"],
                    "ratio_lo": min(a, b),
                    "ratio_hi": max(a, b),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "series": pl.String,
            "events": pl.String,
            "n_events": pl.Int64,
            "ratio": pl.Float64,
            "ratio_lo": pl.Float64,
            "ratio_hi": pl.Float64,
        },
    )


def co_step_schema() -> dict[str, Any]:
    """Return the columns of the era and co-step tables."""
    return {
        "series": pl.String,
        "events": pl.String,
        "n_events": pl.Int64,
        "ratio": pl.Float64,
        "ratio_lo": pl.Float64,
        "ratio_hi": pl.Float64,
    }


def stamp_stage(
    *,
    topology: data.Topology,
    series: dict[str, np.ndarray],
    bsp_series: dict[str, np.ndarray],
    stage: dict[str, Any],
) -> dict[str, Any]:
    """Check 6: the lag of the peak cross-correlation of daylight changes, per series, must be 0."""
    du: dict[str, np.ndarray] = {}
    for label, s in stage["sites"].items():
        if label in series:
            du[label] = sky.fast_irradiance_change(
                u=s.u,
                u_clear=s.u_clear,
                k=stage["k"][label],
                lit=stage["lit"][label],
                n_days=stage["n_days"],
            )
    lags: dict[str, int] = {}
    n_signal = n_zero = 0
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = [
        (label, series[label], du[label]) for label in du
    ]
    for letter in ("A", "B"):
        members = [
            du[p]
            for p in topology.primaries.filter(pl.col("bsp") == letter)["label"].to_list()
            if p in du
        ]
        if members:
            with np.errstate(all="ignore"):
                pairs.append(
                    (f"BSP_{letter}", bsp_series[letter], np.nanmean(np.stack(members), axis=0))
                )
    for name, x, y in pairs:
        dx = np.where(np.isfinite(y), core.change(x=x, lag=1), np.nan)
        lag, corr = sky.peak_lag(x=dx, y=y)
        if np.isfinite(corr) and abs(corr) >= core.STAMP_MIN_CORR:
            n_signal += 1
            n_zero += lag == 0
            lags[name] = lag
    return {
        "n_series_with_signal": n_signal,
        "n_series_at_lag_zero": int(n_zero),
        "nonzero_lags": {k: v for k, v in lags.items() if v != 0},
        "ok": n_signal > 0 and n_zero == n_signal,
    }


# ---------------------------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------------------------
def run(*, gate: guard.TimingGate, inputs: list[Path], info: dict[str, Any]) -> None:  # noqa: PLR0915
    """Run job 1; `inputs` and `info` fill the run log as the job reads."""
    common.assert_environment(env=dict(os.environ))
    log = guard.ReadLog()
    inputs += [data.SITE_MAP_PATH, data.METADATA_PATH, data.HIERARCHY_PATH]
    info["power_table_delta_version"] = data.power_table_version()
    topology = common.load_topology()
    counts_ok = data.assert_topology(topology=topology)
    guard.say("check 1: topology asserted; generator counts as designed:", counts_ok)
    inputs += [data.FRAME39_PATH, data.SITES39_PATH]
    stage = sky_stage(topology=topology, log=log)
    guard.write_public_csv(frame=stage["audit"], name="u0_class_audit.csv")
    guard.write_public_csv(frame=stage["pairs"], name="u0_same_sky_pairs.csv")
    log.mark_audit_done()
    guard.say(
        "class audit passed:",
        stage["audit_ok"],
        "rows failed:",
        stage["n_audit_rows_failed"],
        "same-sky pairs:",
        stage["n_pairs"],
        "of",
        stage["pairs"].height,
        "| unmetered farms only:",
        stage["n_pairs_unmetered"],
    )
    # ---- from here on, Primary readings, the BSP meters, and generators are read ----
    ids = [
        *topology.primaries["time_series_id"].to_list(),
        *topology.bsp_meter.values(),
        *topology.generators["time_series_id"].to_list(),
    ]
    power = data.load_power(ids=ids, log=log, name="primaries_bsp_meters_generators")
    log.assert_audit_before_primary_reads()
    series = common.label_arrays(topology=topology, power=power)
    bsp_series = {k: power[v] for k, v in topology.bsp_meter.items()}
    gens = {ts: power[ts] for ts in topology.generators["time_series_id"].to_list()}
    inputs += [data.ENGINE_FRAME_PATH]
    eng = data.load_engine_frame(log=log)
    record, record_path, _ = data.load_engine_record(log=log)
    inputs += [record_path]
    bsp_of_gen = dict(
        zip(topology.generators["time_series_id"], topology.generators["bsp"], strict=True)
    )
    engine_id = identify_engine_series(
        record=record, gens={ts: x for ts, x in gens.items() if bsp_of_gen[ts] == "A"}
    )
    guard.say("engine series identified among the generators under BSP A:", engine_id is not None)
    labels = core.generator_labels(
        bsp=topology.generators["bsp"].to_list(),
        ids=topology.generators["time_series_id"].to_list(),
    )
    guard.write_private_parquet(
        frame=pl.DataFrame({"generator": list(labels.values()), "generator_id": list(labels)}),
        name=common.LABELS_NAME,
    )
    n_days = stage["n_days"]
    classes = addback_stage(
        topology=topology,
        eng=eng,
        series=series,
        bsp_series=bsp_series,
        gens=gens,
        record=record,
        du_of=stage["du_of"],
        engine_id=engine_id,
        labels=labels,
        n_days=n_days,
        gate=gate,
    )
    guard.write_public_parquet(frame=classes.drop("generator_id"), name=common.CLASSES_NAME)
    per_generator = classes.unique(subset=["generator_id"])
    cls_summary = per_generator.group_by("generator_class").len()
    guard.say(
        "generator classes:",
        dict(zip(cls_summary["generator_class"], cls_summary["len"], strict=True)),
        "| class differs under the joint fit:",
        int(per_generator["class_differs_joint"].sum()),
        "| not positive for generation:",
        int((~per_generator["sign_ok"]).sum()),
        "| engine-like:",
        int(per_generator["engine_like"].sum()),
    )
    n_mva_gens = int((topology.generators["units"] == "MVA").sum())
    guard.say("generator series metered in MVA:", n_mva_gens)
    # check 3 and check 5 on the engine record
    cap_d8 = topology.capacity(label="D8")
    below = core.below_count(
        reading_mw=series["D8"][np.isfinite(record)],
        engine_mw=record[np.isfinite(record)],
        capacity_mw=cap_d8,
    )
    guard.say(
        "check 3: half-hours with reading minus record below -2 points:",
        below.n_below,
        "of",
        below.n_total,
        "share",
        round(below.share, 4),
        below.form_hint,
    )
    candidates = {
        p: len(common.addback_candidates(classes=classes, primary=p)) for p in core.BSP_A_UNSIGNED
    }
    addback_form = core.binding_addback_form(
        n_candidates=sum(candidates.values()), n_below=below.n_below
    )
    guard.say(
        "add-back candidates by unsigned Primary:", candidates, "| binding form:", addback_form
    )
    # check 5: the BSP meter's coverage, and the readable share of the de-meaned residual change
    rec = np.isfinite(record)
    coverage = float(np.isfinite(bsp_series["A"][rec]).mean()) if rec.any() else 0.0
    start_a = common.both_start(series=series, transformers=core.TWO_TRANSFORMERS_2025)
    res_d8 = common.spanned_residual(
        bsp=bsp_series["A"],
        series=series,
        children=core.BSP_A_PRIMARIES,
        transformers=core.TWO_TRANSFORMERS_2025,
        leave_out="D8",
        generators=[
            gens[g]
            for g in common.generator_ids_to_subtract(classes=classes, bsp="A", variant="primary")
        ],
        start=start_a,
    )
    quiet = core.quiet_mask(engine_change=eng.change, calendar_ok=eng.has_row)
    y_engine = core.demean_matched(
        x=core.change(x=res_d8, lag=core.ENGINE_LAG_HALF_HOURS), key=eng.key, quiet=quiet
    )
    x_engine = core.change(x=record, lag=core.ENGINE_LAG_HALF_HOURS)
    rows_engine = np.isfinite(x_engine)
    readable = float(np.isfinite(y_engine[rows_engine]).mean()) if rows_engine.any() else 0.0
    engine_void = coverage < core.MIN_ENGINE_COVERAGE or readable < core.MIN_ENGINE_COVERAGE
    key_agreement = float((eng.key[eng.has_row] == eng.frame_key[eng.has_row]).mean())
    guard.say(
        "check 5: BSP A meter coverage of the engine record:",
        round(coverage, 3),
        "| record changes with a finite de-meaned residual change:",
        round(readable, 3),
        "| calendar key agrees with the frame's key on frame rows:",
        round(key_agreement, 4),
        "| engine test void:",
        engine_void,
    )
    # check 4
    guard.say(
        "check 4: farms of the frame that are metered under BSP A or B:",
        len(stage["metered_farms"]),
    )
    # check 6
    stamps = stamp_stage(topology=topology, series=series, bsp_series=bsp_series, stage=stage)
    guard.say(
        "check 6: series at lag 0:",
        stamps["n_series_at_lag_zero"],
        "of",
        stamps["n_series_with_signal"],
        "ok:",
        stamps["ok"],
    )
    # check 7
    switching: dict[str, Any] = {"readable": False}
    try:
        inputs += [data.SWITCHING_LOG_PATH]
        sheets = pl.read_excel(data.SWITCHING_LOG_PATH, sheet_id=0)
        numbers = {
            r["label"]: int(r["substation_number"])
            for r in topology.primaries.join(
                pl.read_parquet(data.METADATA_PATH).select("time_series_id", "substation_number"),
                on="time_series_id",
            ).iter_rows(named=True)
        }
        table = switching_windows_per_primary(sheets=sheets, numbers=numbers)
        switching = {
            "readable": bool(table["readable"][0]),
            "n_windows_total": int(table["n_windows"].sum()),
            "n_windows_by_primary": dict(zip(table["label"], table["n_windows"], strict=True)),
        }
    except Exception as exc:  # noqa: BLE001
        switching = {"readable": False, "reason": type(exc).__name__}
    guard.say(
        "check 7: switching-log windows readable:",
        switching["readable"],
        "total:",
        switching.get("n_windows_total"),
        "by Primary:",
        switching.get("n_windows_by_primary"),
    )
    # check 8
    gap_counts = {p: long_gaps(x=series[p]) for p in ("D2", "D3", "D4", "D19", "D21")}
    guard.say("check 8: gaps longer than one day:", gap_counts)
    # check 9: the share of the largest embedded PV farm at D12 in D12's PV capacity
    d12 = classes.filter(
        (pl.col("primary") == "D12")
        & (pl.col("label") == "embedded_not_added_back")
        & (pl.col("generator_type") == "PV")
    ).unique(subset=["generator_id"])
    capacities = data.load_effective_capacity(ids=d12["generator_id"].to_list())
    register_mw = data.register_pv_capacity_mw(primary="D12")
    band = core.dominance_band(
        largest_farm_mw=max(capacities.values(), default=0.0), total_pv_mw=register_mw
    )
    guard.say(
        "check 9: metered PV generators embedded at D12:",
        d12.height,
        "dominance band:",
        band,
        "register table found:",
        register_mw is not None,
    )
    # check 10
    results_path = data.D8_BSP_STUDY_DIR / "event_study_results.csv"
    inputs += [results_path]
    era_table = (
        era_split_from_results(results=pl.read_csv(results_path))
        if results_path.is_file()
        else pl.DataFrame(schema=co_step_schema())
    )
    co_step = co_step_by_type(
        topology=topology, eng=eng, d8_reading=series["D8"], gens=gens, classes=classes, gate=gate
    )
    era_all = pl.concat(
        [
            era_table.with_columns(source=pl.lit("stored_csv")),
            co_step.with_columns(source=pl.lit("recomputed")),
        ],
        how="diagonal",
    )
    guard.write_public_csv(frame=era_all, name="u0_era_split.csv")
    guard.say(
        "check 10: era-split rows:", era_table.height, "co-step rows by type:", co_step.height
    )
    # check 11: rows on or after the holdout, per input actually used
    ens_leads, n_ens_after = data.ens_temperature_leads()
    inputs += [data.ENS_MEMBERS_PATH]
    log.record_holdout(name="ens_table", read_after=0, dropped=n_ens_after)
    guard.say(
        "check 11: rows on or after the holdout by input:", log.holdout, "ENS leads:", ens_leads
    )
    # topology output: generators carry their anonymous label and no time series id
    later = np.arange(core.N_GRID) >= start_a
    prim_out = topology.primaries.with_columns(
        kind=pl.lit("primary"),
        generator_type=pl.lit(None, dtype=pl.String),
        coverage_grid=pl.Series(
            [float(np.isfinite(series[lb]).mean()) for lb in topology.primaries["label"]]
        ),
        coverage_later_span=pl.Series(
            [float(np.isfinite(series[lb][later]).mean()) for lb in topology.primaries["label"]]
        ),
    )
    gen_out = (
        topology.generators.with_columns(
            kind=pl.lit("generator"),
            label=pl.col("time_series_id").replace_strict(labels, return_dtype=pl.String),
            time_series_id=pl.lit(None, dtype=pl.Int32),
            capacity_mw=pl.lit(None, dtype=pl.Float64),
            coverage_grid=pl.Series(
                [float(np.isfinite(gens[t]).mean()) for t in topology.generators["time_series_id"]]
            ),
            coverage_later_span=pl.Series(
                [
                    float(np.isfinite(gens[t][later]).mean())
                    for t in topology.generators["time_series_id"]
                ]
            ),
        )
        .rename({"type": "generator_type"})
        .select(
            "time_series_id",
            "units",
            "bsp",
            "kind",
            "generator_type",
            "label",
            "capacity_mw",
            "coverage_grid",
            "coverage_later_span",
        )
    )
    topo_out = pl.concat([prim_out.select(gen_out.columns), gen_out], how="vertical_relaxed")
    guard.write_public_parquet(frame=topo_out, name="u0_topology.parquet")
    guard.write_private_parquet(
        frame=pl.DataFrame(
            {"farm": list(stage["farm_id"]), "time_series_id": list(stage["farm_id"].values())}
        ),
        name="u0_farm_labels.parquet",
    )
    best_corr = max(
        (v for v in per_generator["engine_corr"].to_list() if v is not None and np.isfinite(v)),
        default=float("nan"),
    )
    pair_rule_enough = stage["n_pairs_unmetered"] >= core.MIN_SAME_SKY_PAIRS
    checks = {
        "class_audit_ok": stage["audit_ok"],
        "n_same_sky_pairs": stage["n_pairs"],
        "n_same_sky_pairs_unmetered_farms": stage["n_pairs_unmetered"],
        "u1_pair_rule": "unmetered_pairs_enough"
        if pair_rule_enough
        else "metered_pairs_reported_separately_u1_inconclusive",
        "u1_inconclusive_by_pair_rule": not pair_rule_enough,
        "skip_jobs_4_and_5": stage["n_pairs"] < core.MIN_SAME_SKY_PAIRS,
        "engine_test_void": engine_void,
        "engine_record_coverage": coverage,
        "engine_record_readable_share": readable,
        "calendar_key_agrees_with_frame_key": key_agreement,
        "stamp_alignment_ok": stamps["ok"],
        "stamp_alignment": stamps,
        "generator_counts_as_designed": counts_ok,
        "engine_series_found": engine_id is not None,
        "engine_best_generator_change_correlation": best_corr,
        "n_engine_like_generators": int(per_generator["engine_like"].sum()),
        "n_generators_not_positive_for_generation": int((~per_generator["sign_ok"]).sum()),
        "n_generators_class_differs_under_joint_fit": int(
            per_generator["class_differs_joint"].sum()
        ),
        "n_mva_generator_series": n_mva_gens,
        "below_minus_2_points": {
            "n": below.n_below,
            "total": below.n_total,
            "share": below.share,
            "hint": below.form_hint,
        },
        "addback_candidates_by_unsigned_primary": candidates,
        "binding_addback_form": addback_form,
        "regression_span": "whole grid for every Primary except D8, whose eras use the engine frame",
        "farms_metered_under_a_bsp": len(stage["metered_farms"]),
        "switching_logs": switching,
        "long_gaps": gap_counts,
        "d12_dominance_band": band,
        "d12_register_table_found": register_mw is not None,
        "rows_on_or_after_holdout": log.holdout,
        "ens_temperature_leads": ens_leads,
        "all_holdout_counts_zero": log.holdout_clean(),
        "read_order": log.as_list(),
    }
    guard.say("job 1 done; core-seconds:", round(gate.elapsed(), 1))
    guard.write_public_json(payload=checks, name=common.CHECKS_NAME)
    if not stage["audit_ok"] or not stamps["ok"]:
        raise SystemExit(2)


def main() -> None:
    """Run job 1 and write its run log even when the job crashes."""
    gate = guard.TimingGate()
    inputs: list[Path] = []
    info: dict[str, Any] = {}
    with guard.run_log(job="u0", folder=FOLDER, gate=gate, inputs=inputs, info=info):
        run(gate=gate, inputs=inputs, info=info)


if __name__ == "__main__":
    main()

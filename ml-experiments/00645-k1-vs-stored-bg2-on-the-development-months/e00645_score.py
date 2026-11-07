# ruff: noqa: D103, E501, PERF401, PLR0912
"""Experiment 645: score K1, the Candidate, K3, BG2, and two comparators on the development months.

Reads the roster export (K1, Candidate, K3; experiments 612 and 613) and experiment 87's stored BG2
(`p_s6_bg2`), joins them on (time_series_id, power_fcst_init_time, valid_time), and scores every
system on the common rows with the formulas of experiment 620's `e00620_score.py` (imported, not
edited). The comparators are regenerated from the cleaned power as in experiment 620. No refit and
no day on or after 2026-07-01 is read. The event-free cut applies experiment 637's frozen detector
to development days only (the residuals are sliced at 2026-07-01 before the detector runs).

Usage: python e00645_score.py OUT_DIR
"""

import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import e00583_common as c  # ty: ignore[unresolved-import]
import e00620_score as s  # ty: ignore[unresolved-import]
import e00637_detector as d  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

MONTHS: Final = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
EXPORT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
BG2_PATH: Final[Path] = (
    next(ML_EXPERIMENTS_DATA_DIR.glob("00087-*"))
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)
RULE_PATH: Final[Path] = (
    Path(d.__file__).parent.parent
    / "00637-ara-detector-rescore-of-the-bakeoff-final-test"
    / "ara_rule.json"
)
RULE_SHA256: Final[str] = "e1c8df968b2da090b7c959077e38894d621a437796058e7cac4f0b33272280e5"
SYSTEMS: Final = ("K1", "Candidate", "K3", "BG2", "Manual heuristic", "Climatology")
VS_BG2: Final = ("K1", "Candidate", "K3", "Manual heuristic")
BANDS: Final = {
    "days 1-10": (0, 240),
    "day 1": (0, 24),
    "day 2": (24, 48),
    "days 3-5": (48, 120),
    "days 6-10": (120, 240),
}
KEYS: Final = s.KEYS
QCOLS: Final = s.QCOLS
WINDOW_START: Final = datetime(2026, 7, 1, tzinfo=UTC)
N_BOOT: Final[int] = 4000
MEASURES: Final = (
    "crps", "mae", "twcrps", "cover90", "cover98", "pin01", "pin05", "pin95", "pin99", "ge05", "le95",
)  # fmt: skip


def month_boot(
    *, block: np.ndarray, vals: np.ndarray, seed: int = 645
) -> tuple[float, float, float]:
    """Return the mean and the 95% month-block bootstrap interval (origin months resampled)."""
    blocks, inv = np.unique(block, return_inverse=True)
    tot, cnt = np.bincount(inv, weights=vals), np.bincount(inv).astype(float)
    pick = np.random.default_rng(seed).integers(0, len(blocks), size=(N_BOOT, len(blocks)))
    draws = tot[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    return float(vals.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load() -> dict:
    """Return the joined rows of all five months as arrays, and the row-count audit."""
    sub = pl.read_csv(EXPORT / "substations.csv")
    cap = dict(zip(sub["time_series_id"].to_list(), sub["capacity"].to_list(), strict=True))
    name = dict(zip(sub["time_series_id"].to_list(), sub["primary_name"].to_list(), strict=True))
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    clean = c.k578.clean()
    bg2_all = pl.read_parquet(BG2_PATH).filter(pl.col("lead_hours") < 240)
    audit, parts = [], []
    thr_cut = int((datetime(2025, 10, 1, tzinfo=UTC) - s.START_US_DT) / s.STEP_DT)
    for mi, m in enumerate(MONTHS):
        fr = {n: pl.read_parquet(EXPORT / f"{n}_{m}.parquet") for n in s.NETWORKS}
        base = fr["K1"]
        for n in s.NETWORKS:
            assert fr[n].select(KEYS).equals(base.select(KEYS)), n
        assert base["valid_time"].max() < WINDOW_START  # ty: ignore[unsupported-operator]
        bg2 = bg2_all.select(*KEYS, "capacity_mw", *QCOLS).rename({q: f"bg2_{q}" for q in QCOLS})
        j = base.join(bg2, on=KEYS, how="left", validate="1:1", maintain_order="left")
        assert j.height == base.height
        have = j["bg2_q50"].is_not_null().to_numpy()
        tid = j["time_series_id"].to_numpy()
        capv = np.array([cap[int(t)] for t in tid])
        capv_bg2 = j["capacity_mw"].to_numpy().astype(float)
        both = have & np.isfinite(capv_bg2)
        assert np.allclose(capv_bg2[both], capv[both], rtol=1e-4), (
            "BG2 capacity differs from the export's"
        )
        prim = np.array([prim_of[int(t)] for t in tid])
        idx = (
            ((j["valid_time"].dt.epoch("us") - s.START_US) // s.STEP_US).to_numpy().astype(np.int64)
        )
        oidx = (
            ((j["power_fcst_init_time"].dt.epoch("us") - s.START_US) // s.STEP_US)
            .to_numpy()
            .astype(np.int64)
        )
        lead = j["lead_hours"].to_numpy().astype(np.float64)
        y = j["observed_power"].to_numpy().astype(np.float64) / capv
        q = {
            n: np.sort(fr[n].select(QCOLS).to_numpy().astype(np.float64) / capv[:, None], axis=1)
            for n in s.NETWORKS
        }
        # BG2's stored quantiles are shares of capacity, as experiment 633 uses them.
        q["BG2"] = np.sort(
            j.select([f"bg2_{x}" for x in QCOLS]).to_numpy().astype(np.float64), axis=1
        )
        q["Manual heuristic"] = s.manual_heuristic(clean=clean, prim=prim, idx=idx, lead_h=lead)
        cutoff = int(
            (datetime.fromisoformat(m + "-01").replace(tzinfo=UTC) - s.START_US_DT) / s.STEP_DT
        )
        q["Climatology"] = s.climatology(clean=clean, prim=prim, idx=idx, cutoff_idx=cutoff)
        ok = np.isfinite(y)
        rec = {
            "month": m,
            "export_rows": len(y),
            "no_observed": int((~ok).sum()),
            "no_bg2": int((ok & ~have).sum()),
        }
        ok &= have
        for n in SYSTEMS:
            miss = ok & ~np.isfinite(q[n]).all(axis=1)
            rec[f"no_{n}"] = int(miss.sum())
            ok &= ~miss
        rec["common_rows"] = int(ok.sum())
        audit.append(rec)
        sel = np.flatnonzero(ok)
        parts.append(
            {
                "month": np.full(len(sel), mi),
                "prim": prim[sel],
                "tid": tid[sel],
                "lead": lead[sel],
                "idx": idx[sel],
                "oidx": oidx[sel],
                "y": y[sel],
                "event_free_export": j["event_free"].to_numpy()[sel],
                "q": {n: v[sel] for n, v in q.items()},
            }
        )
    keys = ("month", "prim", "tid", "lead", "idx", "oidx", "y", "event_free_export")
    out: dict[str, Any] = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["q"] = {n: np.concatenate([p["q"][n] for p in parts]) for n in SYSTEMS}
    out["audit"] = pl.DataFrame(audit)
    out["thr_p"] = np.array([np.nanquantile(clean[p, :thr_cut], 0.9) for p in range(16)])
    out["clean"] = clean
    out["name"] = {prim_of[t]: name[t] for t in prim_of if t in name}
    return out


def ara_flags(*, clean: np.ndarray) -> np.ndarray:
    """Return the (16, n_days) mask of days that experiment 637's frozen rule flags, development days only."""
    assert hashlib.sha256(RULE_PATH.read_bytes()).hexdigest() == RULE_SHA256, (
        "ara_rule.json changed"
    )
    rule = d.load_rule(RULE_PATH)
    dev = clean[
        :, : d.day_of(d.DEV_END) * d.PER_DAY
    ]  # nothing from 2026-07-01 on enters the residuals
    resid = d.combine(
        daily=d.daily_residuals(clean=dev, ref_clim=d.climatology_reference(clean=dev))
    )
    op = rule["operating"]
    return d.flag_days(
        resid=resid,
        centre=np.array(rule["centre"]),
        mad=np.array(rule["mad"]),
        k=op["k"],
        n=op["n"],
    )


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = load()
    r["audit"].write_csv(out / "row_audit.csv")
    print(r["audit"], flush=True)
    prim, lead, y, month, oidx, idx = r["prim"], r["lead"], r["y"], r["month"], r["oidx"], r["idx"]
    keep = ara_flags(clean=r["clean"])
    flagged = keep[prim, idx // d.PER_DAY]
    names = r["name"]
    label = np.array([names[int(p)] for p in prim])
    short = np.isin(label, ["D15", "D18"])
    transfer = np.isin(label, ["D14", "D15", "D18"])
    q = r["q"]
    thr = r["thr_p"][prim]
    stats = {}
    for n in SYSTEMS:
        pin = s.pinball(y=y, q=q[n])
        stats[n] = {
            "crps": 2.0 * pin.mean(axis=1) * 100,
            "mae": np.abs(y - q[n][:, 5]) * 100,
            "twcrps": s.twcrps(y=y, q=q[n], thr=thr) * 100,
            "cover90": ((q[n][:, 1] <= y) & (y <= q[n][:, 9])).astype(float),
            "cover98": ((q[n][:, 0] <= y) & (y <= q[n][:, 10])).astype(float),
            "ge05": (y >= q[n][:, 1]).astype(float),
            "le95": (y <= q[n][:, 9]).astype(float),
            "pin01": pin[:, 0] * 100,
            "pin05": pin[:, 1] * 100,
            "pin95": pin[:, 9] * 100,
            "pin99": pin[:, 10] * 100,
        }
    cuts = {
        "16 Primaries": np.ones(len(y), bool),
        "14 (no D15, D18)": ~short,
        "13 (no D14, D15, D18)": ~transfer,
    }
    readings = {
        "all rows": np.ones(len(y), bool),
        "event-free (637 detector)": ~flagged,
        "event-free (export flag)": r["event_free_export"].astype(bool),
    }
    day = oidx // 48
    summary, contrasts = [], []
    for cut, cm in cuts.items():
        for reading, rm in readings.items():
            for band, (lo_h, hi_h) in BANDS.items():
                m_ = cm & rm & (lead >= lo_h) & (lead < hi_h)
                if m_.sum() < 100:
                    continue
                for sysname in SYSTEMS:
                    for meas in MEASURES:
                        v = stats[sysname][meas][m_]
                        mean, mlo, mhi = month_boot(block=month[m_], vals=v)
                        _, dlo, dhi, _ = s.day_boot(day=day[m_], vals=v, seed=645)
                        summary.append(
                            {"cut": cut, "reading": reading, "band": band, "system": sysname,
                             "measure": meas, "mean": mean, "month_lo": mlo, "month_hi": mhi,
                             "day_lo": dlo, "day_hi": dhi, "rows": int(m_.sum())}
                        )  # fmt: skip
                for sysname in VS_BG2:
                    for meas in ("crps", "mae", "twcrps", "pin95", "pin99"):
                        v = stats[sysname][meas][m_] - stats["BG2"][meas][m_]
                        mean, mlo, mhi = month_boot(block=month[m_], vals=v)
                        _, dlo, dhi, _ = s.day_boot(day=day[m_], vals=v, seed=645)
                        contrasts.append(
                            {"cut": cut, "reading": reading, "band": band, "a": sysname,
                             "measure": meas, "a_minus_bg2": mean, "month_lo": mlo, "month_hi": mhi,
                             "day_lo": dlo, "day_hi": dhi, "rows": int(m_.sum())}
                        )  # fmt: skip
    pl.DataFrame(summary).write_csv(out / "summary.csv")
    pl.DataFrame(contrasts).write_csv(out / "contrasts.csv")
    rows = []
    for reading, rm in readings.items():
        for p in range(16):
            m_ = rm & (prim == p)
            if m_.sum() < 100:
                continue
            for sysname in SYSTEMS:
                rows.append(
                    {"by": "primary", "key": names[p], "reading": reading, "system": sysname,
                     "rows": int(m_.sum()), **{k: float(stats[sysname][k][m_].mean()) for k in MEASURES}}
                )  # fmt: skip
        for mi, mname in enumerate(MONTHS):
            m_ = rm & (month == mi) & ~transfer
            for sysname in SYSTEMS:
                rows.append(
                    {"by": "month (13 Primaries)", "key": mname, "reading": reading,
                     "system": sysname, "rows": int(m_.sum()),
                     **{k: float(stats[sysname][k][m_].mean()) for k in MEASURES}}
                )  # fmt: skip
    first_day = d.day_of(datetime(2025, 10, 1, tzinfo=UTC))
    pl.DataFrame(
        {
            "flagged_rows": [int(flagged.sum())],
            "scored_rows": [len(y)],
            "flagged_primary_days_from_2025_10_01": [
                int(keep[:, first_day : d.day_of(d.DEV_END)].sum())
            ],
        }
    ).write_csv(out / "detector_census.csv")
    pl.DataFrame(rows).write_csv(out / "by_primary_and_month.csv")
    np.savez_compressed(
        out / "rows.npz",
        prim=prim, lead=lead, day=day, month=month, idx=idx, y=y, flagged=flagged,
        **{f"crps_{n}": stats[n]["crps"] for n in SYSTEMS},
        **{f"q50_{n}": q[n][:, 5] for n in SYSTEMS},
    )  # fmt: skip
    print("done", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

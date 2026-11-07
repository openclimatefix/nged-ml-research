# ruff: noqa: D103, E501
"""Experiment 646: score arms A, B, and C on the development months (CPU, no refit).

Rows are the roster export's rows (experiment 614; the intersection of K1, Candidate, and K3), so
every arm is scored on the rows experiments 612, 613, and 645 scored. Each arm is a set of fit
folders (`fits/TAG/MONTH_dDAY_sSEED.parquet`, capacity fractions) averaged over its seeds quantile
by quantile. Event-free rows are experiment 637's frozen detector, as in experiment 645.

Width. Each system is also scored after one width factor per metering group and lead band, about
the median, chosen on the event-free rows of earlier months only so that the q05 to q95 coverage is
closest to 0.90 (the first month borrows the second month's factor, flagged as lookahead). Every
system gets the same treatment, so the comparison has equal calibration freedom.

Usage: python e00646_score.py OUT_DIR LABEL=SPEC [LABEL=SPEC ...]
SPEC is `TAG:SEEDS:DIR[,DIR...]` where SEEDS is `0` or `0-2`, and each DIR holds `fits/`. A seed's
files are looked up in the DIRs in order.
"""

import sys
from pathlib import Path
from typing import Any, Final

import e00583_common as c  # ty: ignore[unresolved-import]
import e00620_score as s  # ty: ignore[unresolved-import]
import e00637_detector as d  # ty: ignore[unresolved-import]
import e00645_score as s645  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MONTHS: Final = s645.MONTHS
EXPORT: Final = s645.EXPORT
BANDS: Final = {k: v for k, v in s645.BANDS.items() if k != "days 1-10"}
GRID: Final = np.linspace(0.5, 2.5, 81)
GROUPS: Final = ("six MW Primaries", "eight MVA Primaries")
FOLD_GROUPS: Final = {"2025-10 + 2025-12": (0, 1), "2026-01 + 2026-03": (2, 3), "2026-06": (4,)}


def arm_quantiles(*, spec: str, month: str, keys: pl.DataFrame) -> np.ndarray:
    """Return the arm's quantiles (rows, 11) as capacity fractions on `keys`, NaN where missing."""
    tag, seeds_txt, dirs_txt = spec.split(":", 2)
    lo, _, hi = seeds_txt.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    dirs = [Path(x) for x in dirs_txt.split(",")]
    acc = []
    for seed in seeds:
        files = []
        for dd in dirs:
            files = sorted((dd / "fits" / tag).glob(f"{month}_d*_s{seed}.parquet"))
            if files:
                break
        assert files, f"{spec}: no fits for {month} seed {seed}"
        fr = pl.concat([pl.read_parquet(f) for f in files])
        j = keys.join(fr, on=s.KEYS, how="left", validate="1:1", maintain_order="left")
        acc.append(j.select(s.QCOLS).to_numpy().astype(np.float64))
    return np.sort(np.mean(acc, axis=0), axis=1)


def load(*, specs: dict[str, str]) -> dict:
    sub = pl.read_csv(EXPORT / "substations.csv")
    cap = dict(zip(sub["time_series_id"].to_list(), sub["capacity"].to_list(), strict=True))
    group = dict(zip(sub["time_series_id"].to_list(), sub["group"].to_list(), strict=True))
    name = dict(zip(sub["time_series_id"].to_list(), sub["primary_name"].to_list(), strict=True))
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    clean = c.k578.clean()
    thr_cut = int((s645.datetime(2025, 10, 1, tzinfo=s645.UTC) - s.START_US_DT) / s.STEP_DT)
    parts, audit = [], []
    for mi, m in enumerate(MONTHS):
        base = pl.read_parquet(EXPORT / f"K3_{m}.parquet")
        keys = base.select(s.KEYS)
        tid = base["time_series_id"].to_numpy()
        capv = np.array([cap[int(t)] for t in tid])
        q = {n: arm_quantiles(spec=sp, month=m, keys=keys) for n, sp in specs.items()}
        y = base["observed_power"].to_numpy().astype(np.float64) / capv
        ok = np.isfinite(y)
        for values in q.values():
            ok &= np.isfinite(values).all(axis=1)
        audit.append({"month": m, "export_rows": len(y), "common_rows": int(ok.sum())})
        sel = np.flatnonzero(ok)
        idx = (
            ((base["valid_time"].dt.epoch("us") - s.START_US) // s.STEP_US)
            .to_numpy()
            .astype(np.int64)
        )
        oidx = (
            ((base["power_fcst_init_time"].dt.epoch("us") - s.START_US) // s.STEP_US)
            .to_numpy()
            .astype(np.int64)
        )
        parts.append(
            {
                "month": np.full(len(sel), mi),
                "prim": np.array([prim_of[int(t)] for t in tid])[sel],
                "label": np.array([name[int(t)] for t in tid])[sel],
                "group": np.array([group[int(t)] for t in tid])[sel],
                "lead": base["lead_hours"].to_numpy().astype(np.float64)[sel],
                "idx": idx[sel],
                "oidx": oidx[sel],
                "y": y[sel],
                "q": {n: v[sel] for n, v in q.items()},
            }
        )
    out: dict[str, Any] = {
        k: np.concatenate([p[k] for p in parts])
        for k in ("month", "prim", "label", "group", "lead", "idx", "oidx", "y")
    }
    out["q"] = {n: np.concatenate([p["q"][n] for p in parts]) for n in specs}
    out["audit"] = pl.DataFrame(audit)
    out["thr_p"] = np.array([np.nanquantile(clean[p, :thr_cut], 0.9) for p in range(16)])
    out["clean"] = clean
    return out


def band_of(lead: np.ndarray) -> np.ndarray:
    out = np.full(len(lead), "", dtype=object)
    for name, (lo, hi) in BANDS.items():
        out[(lead >= lo) & (lead < hi)] = name
    return out


def width_factors(
    *,
    q: np.ndarray,
    y: np.ndarray,
    month: np.ndarray,
    grp: np.ndarray,
    band: np.ndarray,
    ef: np.ndarray,
) -> np.ndarray:
    """Return the per-row factor: chosen on earlier months' event-free rows of the row's group and band."""
    fac = np.ones(len(y))
    med = q[:, 5:6]
    for g in GROUPS:
        for b in BANDS:
            cell = (grp == g) & (band == b) & ef
            chosen = {}
            for mi in range(len(MONTHS)):
                prev = cell & (month < mi) if mi > 0 else cell & (month == 1)
                if prev.sum() < 200:
                    chosen[mi] = 1.0
                    continue
                cov = [
                    float(
                        (
                            (med[prev, 0] + f * (q[prev, 1] - med[prev, 0]) <= y[prev])
                            & (y[prev] <= med[prev, 0] + f * (q[prev, 9] - med[prev, 0]))
                        ).mean()
                    )
                    for f in GRID
                ]
                chosen[mi] = float(GRID[int(np.argmin(np.abs(np.array(cov) - 0.9)))])
            for mi, f in chosen.items():
                fac[(grp == g) & (band == b) & (month == mi)] = f
    return fac


def row_stats(*, y: np.ndarray, q: np.ndarray, thr: np.ndarray) -> dict[str, np.ndarray]:
    pin = s.pinball(y=y, q=q)
    return {
        "crps": 2.0 * pin.mean(axis=1) * 100,
        "mae": np.abs(y - q[:, 5]) * 100,
        "twcrps": s.twcrps(y=y, q=q, thr=thr) * 100,
        "cover90": ((q[:, 1] <= y) & (y <= q[:, 9])).astype(float),
        "cover98": ((q[:, 0] <= y) & (y <= q[:, 10])).astype(float),
        "ge05": (y >= q[:, 1]).astype(float),
        "le95": (y <= q[:, 9]).astype(float),
        "pin01": pin[:, 0] * 100,
        "pin05": pin[:, 1] * 100,
        "pin95": pin[:, 9] * 100,
        "pin99": pin[:, 10] * 100,
    }


def main(out: Path, specs: dict[str, str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = load(specs=specs)
    r["audit"].write_csv(out / "row_audit.csv")
    print(r["audit"], flush=True)
    prim, lead, y, month, oidx, idx = r["prim"], r["lead"], r["y"], r["month"], r["oidx"], r["idx"]
    flagged = s645.ara_flags(clean=r["clean"])[prim, idx // d.PER_DAY]
    ef = ~flagged
    band = band_of(lead)
    label, grp = r["label"], r["group"]
    thr = r["thr_p"][prim]
    short = np.isin(label, ["D15", "D18"])
    cuts = {"14 (no D15, D18)": ~short, "16 Primaries": np.ones(len(y), bool)}
    rows, per_row = [], {}
    for n, q in r["q"].items():
        fac = width_factors(q=q, y=y, month=month, grp=grp, band=band, ef=ef)
        med = q[:, 5:6]
        qs = med + fac[:, None] * (q - med)
        for variant, qq in (("unscaled", q), ("scaled", qs)):
            per_row[(n, variant)] = row_stats(y=y, q=qq, thr=thr)
    day = oidx // 48
    for (n, variant), st in per_row.items():
        for cut, cm in cuts.items():
            for reading, rm in (("event-free", ef), ("all rows", np.ones(len(y), bool))):
                for b, (lo, hi) in {**BANDS, "days 1-10": (0, 240)}.items():
                    m_ = cm & rm & (lead >= lo) & (lead < hi)
                    for meas, v in st.items():
                        mean, mlo, mhi = s645.month_boot(block=month[m_], vals=v[m_], seed=646)
                        rows.append(
                            {
                                "system": n,
                                "variant": variant,
                                "cut": cut,
                                "reading": reading,
                                "band": b,
                                "measure": meas,
                                "mean": mean,
                                "month_lo": mlo,
                                "month_hi": mhi,
                                "rows": int(m_.sum()),
                            }
                        )
    pl.DataFrame(rows).write_csv(out / "summary.csv")
    # fold groups and per-Primary event-free CRPS, days 1-10
    fg = []
    for (n, variant), st in per_row.items():
        for cut, cm in cuts.items():
            for fname, months in FOLD_GROUPS.items():
                m_ = cm & ef & np.isin(month, months) & (lead < 240)
                fg.append(
                    {
                        "system": n,
                        "variant": variant,
                        "cut": cut,
                        "fold_group": fname,
                        "crps": float(st["crps"][m_].mean()),
                        "rows": int(m_.sum()),
                    }
                )
    pl.DataFrame(fg).write_csv(out / "fold_groups.csv")
    pp = []
    for (n, variant), st in per_row.items():
        for lab in np.unique(label):
            m_ = ef & (label == lab) & (lead < 240)
            pp.append(
                {
                    "system": n,
                    "variant": variant,
                    "primary": lab,
                    **{
                        k: float(st[k][m_].mean())
                        for k in (
                            "crps",
                            "mae",
                            "twcrps",
                            "cover90",
                            "ge05",
                            "le95",
                            "pin95",
                            "pin99",
                        )
                    },
                }
            )
    pl.DataFrame(pp).write_csv(out / "by_primary.csv")
    np.savez_compressed(out / "rows.npz", prim=prim, short=short, lead=lead, day=day, month=month, idx=idx, y=y, flagged=flagged,
                        **{f"crps_{n}_{v}": st["crps"] for (n, v), st in per_row.items()})  # fmt: skip
    print("done", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), dict(a.split("=", 1) for a in sys.argv[2:]))

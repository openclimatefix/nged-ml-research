# ruff: noqa: D103, PLR0915, PERF401
"""Experiment 637: apply the frozen ARA day detector once to the window and re-score (CPU).

Reads experiment 620's `rows.npz` (per-row CRPS and median of the five systems), the cleaned power,
and the frozen `ara_rule.json`. Re-runs no network. Descriptive second reading; nothing is ranked.

Usage: python e00637_window.py RULE_JSON SCORE_DIR_620 OUT_DIR
"""

import hashlib
import sys
from pathlib import Path

import e00583_common as c  # ty: ignore[unresolved-import]
import e00620_extend as ext  # ty: ignore[unresolved-import]
import e00620_score as s  # ty: ignore[unresolved-import]
import e00620_second_reading as r1  # ty: ignore[unresolved-import]
import e00637_detector as d  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SYSTEMS = s.SYSTEMS


def main(rule_path: Path, score: Path, out: Path) -> None:
    ext.redirect()
    rule = d.load_rule(rule_path)
    print("rule SHA-256", hashlib.sha256(rule_path.read_bytes()).hexdigest(), flush=True)
    clean = c.k578.clean()
    resid = d.combine(
        daily=d.daily_residuals(clean=clean, ref_clim=d.climatology_reference(clean=clean))
    )
    centre, mad = np.array(rule["centre"]), np.array(rule["mad"])
    start, end = d.day_of(d.DEV_END), d.day_of(d.WINDOW_END)
    z = (resid - centre[:, None]) / mad[:, None]
    z0 = start - 30  # flags use 30 earlier days only to start runs and bridges
    z_ = np.load(score / "rows.npz")
    prim, idx, y, day = z_["prim"], z_["idx"], z_["y"], z_["day"]
    labels = pl.read_csv(score / "primaries.csv")["label"].to_list()
    vday = idx // d.PER_DAY
    flag_end = int(vday.max()) + 1
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {
                "primary": labels[p],
                "day": str((s.START_US_DT + dd * s.STEP_DT * d.PER_DAY).date()),
                "z": float(z[p, dd]),
            }
            for p in range(16)
            for dd in range(start, end)
        ]
    ).write_csv(out / "daily_z.csv")
    # rule R1 (experiment 620) for comparison
    r1_resid = r1.residual_days(
        clean=clean,
        cutoff_idx=int(
            (r1.datetime.fromisoformat(r1.CUTOFF).replace(tzinfo=r1.UTC) - s.START_US_DT)
            / s.STEP_DT
        ),
    )
    _, r1_keep = r1.kept_runs(resid=r1_resid)
    r1_d0 = r1.day_index(r1.FIRST_DAY)
    levels = {
        "operating": rule["operating"],
        "stricter": rule["stricter"],
        "looser": rule["looser"],
    }
    readings: dict[str, np.ndarray] = {"all rows": np.ones(len(prim), bool)}
    flags = {}
    for name, sel in levels.items():
        keep = d.flag_days(
            resid=resid[:, z0:flag_end], centre=centre, mad=mad, k=sel["k"], n=sel["n"]
        )
        full = np.zeros(resid.shape, bool)
        full[:, z0:flag_end] = keep
        flags[name] = full
        readings[f"kept, {name} (k={sel['k']})"] = ~full[prim, vday]
    full_op = flags["operating"]
    # flag census
    census = []
    for name, full in flags.items():
        removed = full[prim, vday]
        census.append(
            {
                "level": name,
                "rows_removed": int(removed.sum()),
                "rows": len(removed),
                "share_rows": float(removed.mean()),
                "primary_days": int(full[:, start:end].sum()),
                "share_primary_days": float(full[:, start:end].mean()),
            }
        )
    pl.DataFrame(census).write_csv(out / "census.csv")
    print(pl.DataFrame(census), flush=True)
    months = []
    for dd in range(start, end):
        stamp = s.START_US_DT + dd * s.STEP_DT * d.PER_DAY
        months.append(stamp.strftime("%Y-%m"))
    months = np.array(months)
    pl.DataFrame(
        [
            {
                "month": m,
                "primary_days": int((months == m).sum() * 16),
                "flagged": int(full_op[:, start:end][:, months == m].sum()),
                "flag_rate": float(full_op[:, start:end][:, months == m].mean()),
            }
            for m in sorted(set(months))
        ]
    ).write_csv(out / "flag_rate_by_month.csv")
    per_p = []
    for p in range(16):
        per_p.append(
            {
                "primary": labels[p],
                "flagged_days": int(full_op[p, start:end].sum()),
                "of_days": end - start,
            }
        )
    pl.DataFrame(per_p).write_csv(out / "flagged_days_by_primary.csv")
    print(pl.DataFrame(per_p), flush=True)
    # D14 / D18 episode: 2026-07-07 to 2026-10-26
    ep0, ep1 = (
        d.day_of(d.DEV_END.replace(month=7, day=7)),
        d.day_of(d.DEV_END.replace(month=10, day=26)) + 1,
    )
    ep = [
        {
            "primary": lab,
            "episode_days": ep1 - ep0,
            "flagged": int(full_op[labels.index(lab), ep0:ep1].sum()),
        }
        for lab in ("D14", "D18")
    ]
    pl.DataFrame(ep).write_csv(out / "episode.csv")
    print(pl.DataFrame(ep), flush=True)
    # overlap with R1
    r1_full = np.zeros(resid.shape, bool)
    r1_full[:, r1_d0 : r1_d0 + r1_keep.shape[1]] = r1_keep
    a, b = full_op[:, start:end], r1_full[:, start:end]
    pl.DataFrame(
        [{"detector_days": int(a.sum()), "r1_days": int(b.sum()), "both": int((a & b).sum())}]
    ).write_csv(out / "overlap_r1.csv")
    print("overlap with R1", int(a.sum()), int(b.sum()), int((a & b).sum()), flush=True)
    # scores
    rows, wins, per_rows = [], [], []
    for name, m in readings.items():
        per = {
            n: np.array(
                [
                    z_[f"crps_{n}"][m & (prim == p)].mean() if (m & (prim == p)).any() else np.nan
                    for p in range(16)
                ]
            )
            for n in SYSTEMS
        }
        for n in SYSTEMS:
            crps_ = z_[f"crps_{n}"]
            mae = np.abs(y - z_[f"q50_{n}"]) * 100
            mc, lc, hc, _ = s.day_boot(day=day[m], vals=crps_[m])
            mm, lm, hm, _ = s.day_boot(day=day[m], vals=mae[m])
            rec = {
                "reading": name,
                "system": n,
                "rows": int(m.sum()),
                "crps": mc,
                "crps_lo": lc,
                "crps_hi": hc,
                "mae": mm,
                "mae_lo": lm,
                "mae_hi": hm,
            }
            if n != "Manual heuristic":
                dd_ = crps_[m] - z_["crps_Manual heuristic"][m]
                dm, dl, dh, _ = s.day_boot(day=day[m], vals=dd_)
                rec |= {"crps_minus_heuristic": dm, "diff_lo": dl, "diff_hi": dh}
                ok = np.isfinite(per[n]) & np.isfinite(per["Manual heuristic"])
                wins.append(
                    {
                        "reading": name,
                        "system": n,
                        "primaries_scored": int(ok.sum()),
                        "lower_crps_than_heuristic": int(
                            (per[n][ok] < per["Manual heuristic"][ok]).sum()
                        ),
                    }
                )
            rows.append(rec)
        per_rows.extend(
            {"reading": name, "primary": labels[p], **{n: float(per[n][p]) for n in SYSTEMS}}
            for p in range(16)
        )
    pl.DataFrame(per_rows).write_csv(out / "per_primary.csv")
    pl.DataFrame(rows).write_csv(out / "scores.csv")
    pl.DataFrame(wins).write_csv(out / "wins.csv")
    print(
        pl.DataFrame(rows).select(
            "reading",
            "system",
            "rows",
            "crps",
            "crps_lo",
            "crps_hi",
            "mae",
            "crps_minus_heuristic",
            "diff_lo",
            "diff_hi",
        ),
        flush=True,
    )
    print(pl.DataFrame(wins), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

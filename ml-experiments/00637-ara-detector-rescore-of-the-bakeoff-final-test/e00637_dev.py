"""Experiment 637: choose and freeze the ARA day detector on development data only (to 2026-06-30).

Truth on development days: the verified hand-filtered real load transfers lasting over 1 day (from
experiment 41's labelled set, the same set as experiments 101 and 130), plus known-answer steps
injected into the cleaned power. Reads no day from 2026-07-01 on to choose anything. Writes
`ara_rule.json` (rule, thresholds, development precision and recall) and prints its SHA-256.

Usage: python e00637_dev.py OUT_DIR
"""

import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00583_common as c  # ty: ignore[unresolved-import]
import e00620_extend as ext  # ty: ignore[unresolved-import]
import e00637_detector as d  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00101_common import LABELLED_PATH  # ty: ignore[unresolved-import]

KS: Final = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)
NS: Final = (2, 3, 5, 7)
SIZES_MW: Final = (0.5, 1.0, 2.0, 4.0, 8.0)
DURATIONS: Final = (2, 3, 5, 7, 14, 28)
PAD: Final[int] = 3
"""Days either side of a labelled event within which a flag counts as probably explained."""
N_WORLDS: Final[int] = 4
EVENTS_PER_WORLD: Final[int] = 30


def real_event_days(*, series_to_prim: dict[int, int], n_days: int) -> np.ndarray:
    """Return the (16, n_days) mask of days inside verified real transfers longer than 1 day."""
    lab = pl.read_parquet(LABELLED_PATH).filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility").is_in(["conserved", "one_sided"]),
    )
    mask = np.zeros((16, n_days), bool)
    for row in lab.iter_rows(named=True):
        on, end = row["onset_utc"], row["end_utc"]
        if on is None or end is None or (end - on) <= timedelta(days=1):
            continue
        ids = {row["donor"], row["side_a"], row["side_b"]}
        ids |= {int(r["series"]) for r in json.loads(row["receivers"])}
        for sid in ids:
            if sid is not None and int(sid) in series_to_prim:
                mask[
                    series_to_prim[int(sid)],
                    d.day_of(on) : d.day_of(end - timedelta(seconds=1)) + 1,
                ] = True
    return mask


def inject(*, clean: np.ndarray, cap_mw: np.ndarray, seed: int):
    """Return the injected copy, the (16, n_days) event-day mask, and a list of event records."""
    rng = np.random.default_rng(seed)
    out = clean.copy()
    mask = np.zeros((16, clean.shape[1] // d.PER_DAY), bool)
    events = []
    d0, d1 = d.day_of(d.DEV_START) + 10, d.day_of(d.DEV_END) - 35
    busy = np.zeros_like(mask)
    tries = 0
    while len(events) < EVENTS_PER_WORLD and tries < 2000:
        tries += 1
        p = int(rng.integers(16))
        day = int(rng.integers(d0, d1))
        dur = int(rng.choice(DURATIONS))
        if (
            busy[:, day - 10 : day + dur + 10].any()
            or np.isfinite(clean[p, day * 48 : (day + dur) * 48]).mean() < 0.95
        ):
            continue
        mw = float(rng.choice(SIZES_MW)) * rng.choice([-1, 1])
        out[p, day * 48 : (day + dur) * 48] += mw / cap_mw[p]
        mask[p, day : day + dur] = True
        busy[:, day - 10 : day + dur + 10] = True
        events.append({"prim": p, "day": day, "dur": dur, "mw": abs(mw)})
    return out, mask, events


def main(out_dir: Path) -> None:
    ext.redirect()
    clean = c.k578.clean()
    sm = c.k578.sites()
    cap = sm["capacity_mw"].to_numpy()
    series_to_prim = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    n_days = clean.shape[1] // d.PER_DAY
    dev = np.zeros(n_days, bool)
    dev[d.day_of(d.DEV_START) : d.day_of(d.DEV_END)] = True
    ref_clim = d.climatology_reference(clean=clean)
    real_all = real_event_days(series_to_prim=series_to_prim, n_days=n_days)
    near = real_all.copy()
    for sft in range(1, PAD + 1):
        near |= np.roll(real_all, sft, axis=1) | np.roll(real_all, -sft, axis=1)
    real, near = real_all & dev, near & dev
    worlds = [("real", clean, np.zeros_like(real), [])]
    for seed in range(N_WORLDS):
        inj, mask, ev = inject(clean=clean, cap_mw=cap, seed=seed)
        worlds.append((f"inj{seed}", inj, mask, ev))
    resid = {}
    for name, grid, _, _ in worlds:
        resid[name] = d.combine(daily=d.daily_residuals(clean=grid, ref_clim=ref_clim))
    centre, mad = d.scale(resid=resid["real"])
    truth = {name: (real | m) for name, _, m, _ in worlds}
    rows = []
    for k in KS:
        for n in NS:
            tp = fn = fp = 0
            tp_r = fn_r = fp_r = 0
            for name, _, m, _ in worlds:
                keep = d.flag_days(
                    resid=resid[name][:, : d.day_of(d.DEV_END)], centre=centre, mad=mad, k=k, n=n
                )
                keep = np.pad(keep, ((0, 0), (0, n_days - keep.shape[1]))) & dev
                if name == "real":
                    tp_r = int((keep & real).sum())
                    fn_r = int((~keep & real).sum())
                    fp_r = int((keep & ~near).sum())
                    continue
                tp += int((keep & m).sum())
                fn += int((~keep & m).sum())
                fp += int((keep & ~(m | near)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            rows.append(
                {
                    "k": k,
                    "n": n,
                    "precision": prec,
                    "recall": rec,
                    "f1_injected": 2 * prec * rec / max(prec + rec, 1e-9),
                    "f1_real": 2
                    * (tp_r / max(tp_r + fp_r, 1))
                    * (tp_r / max(tp_r + fn_r, 1))
                    / max(tp_r / max(tp_r + fp_r, 1) + tp_r / max(tp_r + fn_r, 1), 1e-9),
                    "real_recall": tp_r / max(tp_r + fn_r, 1),
                    "real_flagged_unexplained": fp_r,
                }
            )
    grid_df = (
        pl.DataFrame(rows)
        .with_columns(f1=(pl.col("f1_injected") + pl.col("f1_real")) / 2)
        .sort(["f1", "k", "n"], descending=[True, False, False])
    )
    print(grid_df)
    best = grid_df.row(0, named=True)
    ks = sorted(KS)
    i = ks.index(best["k"])
    rule = {
        "operating": {"k": best["k"], "n": best["n"]},
        "stricter": {"k": ks[min(i + 1, len(ks) - 1)], "n": best["n"]},
        "looser": {"k": ks[max(i - 1, 0)], "n": best["n"]},
        "bridge_gap_days": 1,
        "dev_start": str(d.DEV_START.date()),
        "dev_end": str(d.DEV_END.date()),
        "climatology_cutoff": str(d.CLIM_CUTOFF.date()),
        "centre": centre.tolist(),
        "mad": mad.tolist(),
        "development": {},
        "grid": grid_df.to_dicts(),
    }
    for label in ("stricter", "operating", "looser"):
        sel = rule[label]
        rec = next(r for r in rows if r["k"] == sel["k"] and r["n"] == sel["n"])
        sel["dev_precision"], sel["dev_recall"] = rec["precision"], rec["recall"]
        sel["dev_real_recall"] = rec["real_recall"]
    # recall by injected size at the operating point
    by_size = {}
    for name, _, m, events in worlds[1:]:
        keep = d.flag_days(
            resid=resid[name][:, : d.day_of(d.DEV_END)],
            centre=centre,
            mad=mad,
            k=best["k"],
            n=best["n"],
        )
        for e in events:
            sl = slice(e["day"], e["day"] + e["dur"])
            by_size.setdefault(e["mw"], []).append(float(keep[e["prim"], sl].mean()))
    rule["development"]["injected_recall_by_mw"] = {
        str(mw): float(np.mean(v)) for mw, v in sorted(by_size.items())
    }
    print(rule["operating"], rule["stricter"], rule["looser"], rule["development"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / d.RULE_NAME
    path.write_text(json.dumps(rule, indent=2, sort_keys=True))
    print("SHA-256", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(Path(sys.argv[1]))

# ruff: noqa: D103, E501
"""Experiment 647, step 1: the input inventory (CPU, read only).

For each input the plan names, this prints and writes (`inventory.md`, `inventory_*.csv` in the data
folder) its source file, its first and last available UTC date, and the share of the scored rows for
which it can be formed. Nothing is fitted, and no day on or after 2026-07-01 is read.

Usage: python e00647_inventory.py
"""

from datetime import timedelta

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00582_common as k582  # ty: ignore[unresolved-import]
import e00606_signals as s606  # ty: ignore[unresolved-import]
import e00645_score as k645  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

D = ML_EXPERIMENTS_DATA_DIR
GS = k578.GRID_START
PER = 48
OUT = c647.WORK
OUT.mkdir(parents=True, exist_ok=True)
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def date(i: int) -> str:
    return (GS + timedelta(minutes=30 * int(i))).strftime("%Y-%m-%d")


def span(mask_row: np.ndarray) -> str:
    ix = np.flatnonzero(mask_row)
    return f"{date(ix.min())} to {date(ix.max())}" if len(ix) else "none"


T_DIRS = (
    "00598-nra-gate-against-signals-as-inputs-at-equal-band-width",
    "00610-nra-neighbour-summary-lag-encoder-inputs-iteration",
    "00613-nra-roster-contrasts-with-intervals-five-seeds",
)
CUT = int((c647.WINDOW_START - GS) / timedelta(minutes=30))
"""First grid index of 2026-07-01: nothing from here on is read."""
sm = k578.sites()
label = sm["label"].to_list()
clean = k578.clean()[:, :CUT]
exp = c647.load_export()
exp_k1 = exp.filter(pl.col("lead_hours") >= 24)
origins = exp.select("time_series_id", "month", "oi", "od").unique()
prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))

say("# Experiment 647: input inventory\n")

# 1. stored K1 export
say("## 1. K1 forecasts: the stored roster export (mean of five seeds, G4 band, no tail constants)")
say(f"Source: `{c647.EXPORT}/K1_<month>.parquet` (experiments 612 and 613).\n")
rows = []
for m in c647.MONTHS:
    e = exp.filter(pl.col("month") == m)
    rows.append(
        {"month": m, "rows": e.height, "primaries": e["time_series_id"].n_unique(),
         "first_origin": str(e["power_fcst_init_time"].min())[:10],
         "last_origin": str(e["power_fcst_init_time"].max())[:10],
         "first_valid": str(e["valid_time"].min())[:10], "last_valid": str(e["valid_time"].max())[:10],
         "observed_share": float(e["observed_power"].is_not_null().mean()),
         "event_free_share": float(e["event_free"].mean()),
         "verified_rows": int(e["in_verified_event"].sum())}
    )  # fmt: skip
tab = pl.DataFrame(rows)
tab.write_csv(OUT / "inventory_export.csv")
say(tab.to_pandas().to_markdown(index=False) if False else str(tab))

# 2. cleaned observed power
say("\n## 2. Cleaned observed power (also the injected bank's un-shifted power)")
say(
    "Source: `e00578_common.clean()` (the cleaned stack of experiment 87; D15 and D18 cut before 2025-03-01).\n"
)
fin = np.isfinite(clean)
for p in range(16):
    say(
        f"- {label[p]}: {span(fin[p])}, finite share {fin[p][fin[p].argmax() :].mean():.3f} from first finite value"
    )

# 3. lag-free table T
say(
    "\n## 3. Lag-free table T (seeds 0 and 1 from experiments 598 and 610, seeds 2 to 4 from experiment 613)"
)
say(
    "Source: `00598-*/tables`, `00610-*/tables`, and `00613-*/tables`, files `T_<month>_s<seed>.npz`; each holds the lags-masked median of the freshest origin before each grid index (lead at most 24 h), over the fold's training and test origins.\n"
)
trows = []
t_by_month: dict[str, np.ndarray] = {}
for m in c647.MONTHS:
    for seed in range(5):
        f = next(p for d_ in T_DIRS if (p := D / d_ / "tables" / f"T_{m}_s{seed}.npz").exists())
        z = np.load(f)
        med = z["median"][:, :CUT]
        filled = np.isfinite(med).any(axis=0)
        if seed == 0:
            t_by_month[m] = med
        first = np.flatnonzero(filled)
        trows.append(
            {"month": m, "seed": seed, "file": f.parent.parent.name[:5], "filled": int(np.isfinite(med).sum()),
             "first": date(first.min()), "last": date(first.max()),
             "first_origin": date(int(z["origin_idx"][z["origin_idx"] >= 0].min()))}
        )  # fmt: skip
ttab = pl.DataFrame(trows)
ttab.write_csv(OUT / "inventory_T.csv")
say(str(ttab))
say(
    "\nShare of T available over the 14 days before each scored origin day (all Primaries, seed 0):"
)
for m in c647.MONTHS:
    o = origins.filter(pl.col("month") == m)
    first_o = int(o["oi"].min())
    med = t_by_month[m]
    pre = np.isfinite(med[:, first_o - 14 * PER : first_o]).mean()
    pre7 = np.isfinite(med[:, first_o - 7 * PER : first_o]).mean()
    pre1 = np.isfinite(med[:, first_o - 1 * PER : first_o]).mean()
    say(
        f"- {m}: first origin {date(first_o)}; T share in the 14 days before {pre:.3f}, 7 days {pre7:.3f}, 1 day {pre1:.3f}"
    )

# 4. signals of experiment 606
say(
    "\n## 4. Gap signals (experiment 606: `gap`, `ew1`, `ew3`, `ew7`; formed from cleaned power minus T)"
)
say(
    "Source: `e00606_signals.quantity_arrays`; the neighbour summaries (`nb`) need `NB.of`, the electrical neighbours of experiment 578's site map. Share of scored (Primary, origin) pairs where each value is finite at the newest allowed lag (origin minus 24 h), seed 0:\n"
)
srows = []
for m in c647.MONTHS:
    o = origins.filter(pl.col("month") == m)
    prim = np.array([prim_of[int(t)] for t in o["time_series_id"].to_list()])
    oi = o["oi"].to_numpy()
    res = (clean - t_by_month[m]).astype(np.float32)
    arrs = s606.quantity_arrays(res=res)
    rec = {"month": m, "pairs": len(oi)}
    for name, arr in arrs.items():
        rec[name] = float(np.isfinite(arr[prim, oi - 2 * PER]).mean())
    d1 = [m_ for m_ in range(len(oi)) if oi[m_] == oi.min()]
    rec["ew7_first_origin_day"] = float(np.isfinite(arrs["ew7"][prim[d1], oi[d1] - 2 * PER]).mean())
    srows.append(rec)
stab = pl.DataFrame(srows)
stab.write_csv(OUT / "inventory_signals.csv")
say(str(stab))
nb = k578.sites()
say(f"\nElectrical neighbours (experiment 578 site map): see `sites()`; columns {nb.columns}")

# 5. G4 gate
say("\n## 5. G4 gate state")
z = np.load(D / "00592-nra-snap-back-gate-faster-onset-and-snap-back-trigger/gate_state.npz")
on = z["G4_on"]
say(
    f"Source: `00592-*/gate_state.npz` (G4_on, G4_snap; shape {on.shape} Primaries x origin days); experiment 613's scorer rebuilds it from the fits and asserts equality with this file."
)
g4 = (
    exp.filter(pl.col("time_series_id").is_not_null())
    .group_by("month")
    .agg(gate_on_share=pl.col("gate_on").mean())
)
say(f"Origin days with G4 on or held in the export (K1 `gate_on`), by month:\n{g4.sort('month')}")

# 6. detector flags
say("\n## 6. Experiment 637's frozen detector (ARA rule) flags on development days")
flags = k645.ara_flags(clean=clean)
say(
    f"Source: `00637-*/ara_rule.json` (sha256 checked by `e00645_score.ara_flags`); flags shape {flags.shape}, development days only (to 2026-06-30)."
)
d_lo = int((c647.DEV_FIRST_DAY - GS).days)
fl = np.flatnonzero(flags[:, d_lo:].any(axis=0)) + d_lo
say(
    f"Flagged Primary-days from {c647.DEV_FIRST_DAY:%Y-%m-%d} to 2026-06-30: {int(flags[:, d_lo:].sum())}; first flagged day {date(fl.min() * PER)}, last {date(fl.max() * PER)}"
)
fl_rows = exp_k1.with_columns(day=pl.col("vi") // PER)
pr = np.array([prim_of[int(t)] for t in fl_rows["time_series_id"].to_list()])
fr = flags[pr, fl_rows["day"].to_numpy()]
say(
    f"Share of scored export rows (leads 24 to 240 h) on a flagged day: {fr.mean():.4f} ({int(fr.sum())} rows)"
)

# 7. verified events
say("\n## 7. Verified D5 and D16 transfer (experiment 580)")
ve = k582.verified_events().filter(
    pl.col("onset_idx") >= int((c647.DEV_FIRST_DAY - GS) / timedelta(minutes=30))
)
say(f"Source: `e00582_common.verified_events()`; {ve.height} Primary-events.")
for r in ve.iter_rows(named=True):
    say(
        f"- {r['label']}: onset {date(r['onset_idx'])}, end {date(r['end_idx'])}, depth share {r['depth_share']:+.3f}"
    )
vr = exp_k1.filter(pl.col("in_verified_event"))
say(
    f"Export rows in the verified events with lead 24 to 240 h: {vr.height} (design answer: 13,068)."
)

# 8. injection bank
say("\n## 8. Injected lag bank (experiment 581)")
bank = pl.read_parquet(
    next((D / "00581-robust-multiweek-lag-encoder-x3").glob("bank_events_x3syn_*.parquet"))
)
say(
    f"Source: `e00581_bank.contaminate` and `00581-*/bank_events_x3syn_*.parquet` ({bank.height} events = 5 months x 24 passes x 16 Primaries)."
)
first_origin = {m: int(origins.filter(pl.col("month") == m)["oi"].min()) for m in c647.MONTHS}
brows = []
for m in c647.MONTHS:
    b = bank.filter(pl.col("month") == m)
    fo = first_origin[m]
    brows.append(
        {"month": m, "events": b.height, "earliest_onset": date(int(b["onset_idx"].min())),
         "end": date(int(b["end_idx"].max())),
         "starts_before_first_origin": int((b["onset_idx"] < fo).sum()),
         "starts_in_previous_month": int((b["onset_idx"] < fo - fo % PER).sum())}
    )  # fmt: skip
btab = pl.DataFrame(brows)
btab.write_csv(OUT / "inventory_bank.csv")
say(str(btab))

# 9. stored forecasts on the bank
say("\n## 9. Stored K1 forecasts on the injected bank (experiment 613 `inject` files)")
inj = D / "00613-nra-roster-contrasts-with-intervals-five-seeds/arms/k1_d5/work/inject"
files = sorted(inj.glob("*.parquet"))
say(
    f"Source: `{inj}`; {len(files)} files (expected 1200 = 5 months x 24 passes x 2 refits x 5 seeds); lags-masked twin from `.../fits/P5-removed`."
)
irows = []
for m in c647.MONTHS:
    for refit in ("d0", "d14"):
        f = inj / f"{m}_r00_{refit}_s0.parquet"
        e = pl.read_parquet(f, columns=["power_fcst_init_time", "valid_time"])
        irows.append(
            {"month": m, "refit": refit, "first_origin": str(e["power_fcst_init_time"].min())[:10],
             "last_origin": str(e["power_fcst_init_time"].max())[:10],
             "last_valid": str(e["valid_time"].max())[:10], "rows": e.height}
        )  # fmt: skip
itab = pl.DataFrame(irows)
itab.write_csv(OUT / "inventory_inject.csv")
say(str(itab))

(OUT / "inventory.md").write_text("\n".join(lines) + "\n")
say(f"\nwritten {OUT / 'inventory.md'}")

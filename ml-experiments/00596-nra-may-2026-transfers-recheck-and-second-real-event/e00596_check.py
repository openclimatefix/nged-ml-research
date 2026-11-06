"""Experiment 596: recheck the May 2026 load-shift events against the lag-free forecast.

Implements the pass rule fixed in `result.md` before any gap was computed. Writes
`check_table.csv`, `placebo.csv`, and `gap_series.npz` to this experiment's data folder.
"""

from pathlib import Path
from typing import Final

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00596-nra-may-2026-transfers-recheck-and-second-real-event"
)
DAY: Final[int] = 48
MAY_START: Final[str] = "2026-05"
Q50: Final[int] = a.IQ[0.5]


def gap_series(*, t: a.Table) -> np.ndarray:
    """Return (Primary, grid) mean of observed minus the lag-free median over 24-120 h leads."""
    keep = (t.lead >= 24) & np.isfinite(t.arms["E"][:, Q50])
    g = (t.obs - t.arms["E"][:, Q50])[keep]
    p, i = t.prim[keep], t.idx[keep]
    shape = t.clean.shape
    tot = np.zeros(shape)
    cnt = np.zeros(shape)
    np.add.at(tot, (p, i), g)
    np.add.at(cnt, (p, i), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / cnt, np.nan)


def signal(*, gap: np.ndarray, p: int, onset: int, end: int) -> float:
    """Return the signal S of an event.

    S is the mean gap inside the span (1 day in from each end) minus the mean gap in the 7 days
    that start 1 day after the end.
    """
    inside = gap[p, onset + DAY : end - DAY]
    after = gap[p, end + DAY : end + 8 * DAY]
    return float(np.nanmean(inside) - np.nanmean(after))


def merged_events(*, ev: pl.DataFrame) -> list[dict]:
    """Return one dict per (Primary, merged span)."""
    out: list[dict] = []
    for p, g in ev.sort("primary", "onset_idx").group_by("primary", maintain_order=True):
        cur = None
        for r in g.iter_rows(named=True):
            if cur is not None and r["onset_idx"] <= cur["end_idx"]:
                cur["end_idx"] = max(cur["end_idx"], r["end_idx"])
                cur["transfers"].append(r["transfer_id"])
            else:
                if cur is not None:
                    out.append(cur)
                cur = {
                    "primary": p[0],
                    "label": r["label"],
                    "onset_idx": r["onset_idx"],
                    "end_idx": r["end_idx"],
                    "transfers": [r["transfer_id"]],
                }
        if cur is not None:
            out.append(cur)
    return out


def main() -> None:
    """Run the pass rule over the May 2026 Primary-events and write the check table."""
    OUT.mkdir(parents=True, exist_ok=True)
    t = a.Table(with_weekmask=False)
    gap = gap_series(t=t)
    ev = k.verified_events()
    # merged spans: the depth of a merged span is the mean over the span, as in experiment 582
    merged = merged_events(ev=ev)
    for m in merged:
        length = m["end_idx"] - m["onset_idx"]
        # mean depth over the merged span: each transfer contributes depth * its length / merged
        # length (transfers are summed where they overlap)
        tot = 0.0
        for r in ev.filter(
            pl.col("transfer_id").is_in(m["transfers"]) & (pl.col("primary") == m["primary"])
        ).iter_rows(named=True):
            tot += r["depth_share"] * (r["end_idx"] - r["onset_idx"]) / length
        m["mean_depth_share"] = tot
    # near mask: within 7 days of any verified span at the Primary
    near = np.zeros(t.clean.shape, bool)
    for r in ev.iter_rows(named=True):
        near[r["primary"], max(r["onset_idx"] - 7 * DAY, 0) : r["end_idx"] + 7 * DAY] = True
    first_may = (np.datetime64("2026-05-01") - np.datetime64("2019-09-01")).astype(int) * DAY
    rows = []
    for m in merged:
        if m["onset_idx"] < first_may:
            continue
        p, on, en = m["primary"], m["onset_idx"], m["end_idx"]
        s = signal(gap=gap, p=p, onset=on, end=en)
        d = m["mean_depth_share"]
        # placebo windows of the same length, slid by one day, in months with forecasts
        length = en - on
        ps = []
        for s0 in range(0, gap.shape[1] - length - 9 * DAY, DAY):
            if near[p, s0 : s0 + length + 8 * DAY].any():
                continue
            seg = gap[p, s0 + DAY : s0 + length - DAY]
            aft = gap[p, s0 + length + DAY : s0 + length + 8 * DAY]
            if np.isfinite(seg).mean() < 0.8 or np.isfinite(aft).mean() < 0.8:
                continue
            ps.append(float(np.nanmean(seg) - np.nanmean(aft)))
        ps_a = np.abs(np.array(ps))
        p95 = float(np.quantile(ps_a, 0.95)) if len(ps_a) else float("nan")
        rows.append(
            {
                "label": m["label"],
                "primary": p,
                "transfers": ",".join(map(str, m["transfers"])),
                "onset": k.date_of(on),
                "end": k.date_of(en),
                "length_days": length / DAY,
                "depth_share": d,
                "depth_mw": d * float(t.capacity[p]),
                "S": s,
                "S_over_depth": s / d,
                "placebo_n": len(ps_a),
                "placebo_p95": p95,
                "sign_ok": bool(np.sign(s) == np.sign(d)),
                "size_ok": bool(0.5 <= s / d <= 1.5),
                "placebo_ok": bool(abs(s) > p95),
            }
        )
    tab = pl.DataFrame(rows).with_columns(
        passes=pl.col("sign_ok") & pl.col("size_ok") & pl.col("placebo_ok")
    )
    tab.write_csv(OUT / "check_table.csv")
    np.savez_compressed(OUT / "gap_series.npz", gap=gap)
    with pl.Config(tbl_cols=20, tbl_width_chars=250):
        print(tab)


if __name__ == "__main__":
    main()

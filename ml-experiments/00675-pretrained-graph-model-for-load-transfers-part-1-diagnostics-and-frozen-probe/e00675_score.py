"""Score the probe arms of experiment 675 against E1-J at E1-J's false-donor rate.

    uv run --no-sync python e00675_score.py [probe|probe_smoke]

The false-donor rate is E1-J's on the 40 scored worlds: decoded downward events of 4 noise units or
more that match no injected downward event (any kind) within 24 half-hours at the same Primary,
per Primary-year. A probe alarm episode is a run of hours with a score above the threshold (runs
within 24 hours merge); it is false if it overlaps no injected downward event's window from 12
hours before to 24 hours after its start. The threshold is the one whose false-episode rate equals
E1-J's. A donor is found when an episode overlaps its window. Shares are weighted by |MW| and
intervals resample whole worlds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM
from e00665_score import cell_of  # ty: ignore[unresolved-import]
from e00675_pretrain import ROOT

CELLS = ["long, deep", "long, shallow", "medium, deep", "medium, shallow", "short, deep", "short, shallow"]
import os
WORLDS = list(range(900, 903 if os.environ.get("E675_SMOKE") else 940))
SCORED_H = SCORED_FROM // 2
YEARS = (GRID_LENGTH - SCORED_FROM) / (365.25 * DAY)


def sites_of() -> list[str]:
    """The 15 Primaries in panel order."""
    from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]

    return list(Rig().p653.sites)


def cell(row: dict) -> str:
    """Cell of an injected event."""
    days = None if row["end"] < 0 else (row["end"] - row["start"]) / DAY
    return cell_of(days=days, units=row["units"]) if row["units"] >= 2.0 else "under 2"


def truth_tables(*, worlds: list[int]) -> dict[int, pl.DataFrame]:
    """Truth tables of the scored worlds."""
    return {w: pl.read_parquet(ROOT / "worlds" / f"test_{w}_truth.parquet") for w in worlds}


def e1j_numbers(*, worlds: list[int], truth: dict[int, pl.DataFrame], sites: list[str]) -> tuple[float, pl.DataFrame]:
    """E1-J's false-donor rate per Primary-year and its donor-found rows.

    E1-J's alarms are its decoded downward events of 4 noise units or more, one hour each, merged
    into episodes and compared with the injected downward windows exactly as the probe's are.
    """
    eps: dict = {}
    for w in worlds:
        ev = pl.read_parquet(ROOT / "worlds" / f"test_{w}_e1j.parquet").filter(
            (pl.col("amp_mw") < 0) & (pl.col("start") >= SCORED_FROM) & (pl.col("units") >= 4)
        )
        for i, site in enumerate(sites):
            hot = np.zeros(GRID_LENGTH // 2 - SCORED_H + 48, bool)
            for st in ev.filter(pl.col("primary") == site)["start"].to_list():
                hot[st // 2 - SCORED_H] = True
            eps[(w, i)] = episodes(hot=hot)
    wins = windows_of(truth=truth, sites=sites)
    rate = false_rate(eps=eps, wins=wins) / (len(worlds) * 15 * YEARS)
    rows = []
    for w in worlds:
        for d in truth[w].filter((pl.col("amp_mw") < 0) & (pl.col("kind") == "edge") & (pl.col("partners") != "")).iter_rows(named=True):
            h0 = d["start"] // 2 - SCORED_H
            hit = any(a <= h0 + 24 and b >= h0 - 12 for a, b in eps[(w, sites.index(d["site"]))])
            rows.append({"world": w, "cell": cell(d), "amp": abs(d["amp_mw"]), "found": float(hit)})
    return rate, pl.DataFrame(rows)


def episodes(*, hot: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True in a boolean vector, merging runs less than 24 apart."""
    idx = np.flatnonzero(hot)
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > 24:
            out.append((int(start), int(prev)))
            start = i
        prev = i
    out.append((int(start), int(prev)))
    return out


def probe_episodes(*, scores: dict[int, np.ndarray], thr: float) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Alarm episodes per (world, node) among the scored hours."""
    return {
        (w, i): episodes(hot=sc[i, SCORED_H:] > thr) for w, sc in scores.items() for i in range(sc.shape[0])
    }


def windows_of(*, truth: dict[int, pl.DataFrame], sites: list[str]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Windows (hours after SCORED_H) of every injected downward event, per (world, node)."""
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for w, t in truth.items():
        for r in t.filter(pl.col("amp_mw") < 0).iter_rows(named=True):
            h0 = r["start"] // 2 - SCORED_H
            out.setdefault((w, sites.index(r["site"])), []).append((h0 - 12, h0 + 24))
    return out


def false_rate(*, eps: dict, wins: dict) -> int:
    """Number of episodes that overlap no injected downward window."""
    n = 0
    for key, lst in eps.items():
        for a, b in lst:
            n += not any(a <= wb and b >= wa for wa, wb in wins.get(key, []))
    return n


def auroc(*, y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve by ranks (NaN when one class is missing)."""
    pos = y == 1
    if pos.all() or (~pos).all():
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def score_arm(*, arm: str, seed: int, probe: Path, truth: dict, sites: list[str], target: float) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    """Donor-found rows at E1-J's false rate, and the recipient pairs, of one arm and seed."""
    scores = {w: np.load(probe / f"score_{arm}_seed{seed}_world{w}.npy") for w in WORLDS}
    wins = windows_of(truth=truth, sites=sites)
    lo, hi = -6.0, 6.0
    for _ in range(25):
        mid = (lo + hi) / 2
        rate = false_rate(eps=probe_episodes(scores=scores, thr=mid), wins=wins) / (len(WORLDS) * 15 * YEARS)
        lo, hi = (mid, hi) if rate > target else (lo, mid)
    thr = hi
    achieved = false_rate(eps=probe_episodes(scores=scores, thr=thr), wins=wins) / (len(WORLDS) * 15 * YEARS)
    print(f"{arm} seed {seed}: threshold {thr:.2f}, false-episode rate {achieved:.3f} (target {target:.3f})")
    eps = probe_episodes(scores=scores, thr=thr)
    rows = []
    for w in WORLDS:
        for d in truth[w].filter((pl.col("amp_mw") < 0) & (pl.col("kind") == "edge") & (pl.col("partners") != "")).iter_rows(named=True):
            h0 = d["start"] // 2 - SCORED_H
            hit = any(a <= h0 + 24 and b >= h0 - 12 for a, b in eps[(w, sites.index(d["site"]))])
            rows.append({"world": w, "cell": cell(d), "amp": abs(d["amp_mw"]), "found": float(hit)})
    rec = pl.read_parquet(probe / f"{arm}_seed{seed}_recipient.parquet")
    return pl.DataFrame(rows), rec, thr


def share(*, df: pl.DataFrame, cell_name: str, worlds: np.ndarray) -> float:
    """Weighted found share of a cell over a multiset of worlds."""
    sub = df.filter(pl.col("cell") == cell_name)
    g = sub.group_by("world").agg((pl.col("amp") * pl.col("found")).sum().alias("f"), pl.col("amp").sum().alias("t"))
    f = {r["world"]: (r["f"], r["t"]) for r in g.iter_rows(named=True)}
    num = sum(f.get(int(w), (0, 0))[0] for w in worlds)
    den = sum(f.get(int(w), (0, 0))[1] for w in worlds)
    return num / den if den else float("nan")


def main() -> None:
    """Score every arm and seed that has run, and print the contrasts."""
    probe = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "probe")
    sites = sites_of()
    truth = truth_tables(worlds=WORLDS)
    target, e1j = e1j_numbers(worlds=WORLDS, truth=truth, sites=sites)
    print(f"E1-J false-donor rate {target:.3f} per Primary-year")
    arms = {}
    for arm in ("P-M", "P-C", "R", "F"):
        for seed in (0, 1, 2):
            if (probe / f"{arm}_seed{seed}_donors.parquet").exists():
                arms[(arm, seed)] = score_arm(arm=arm, seed=seed, probe=probe, truth=truth, sites=sites, target=target)
    rng = np.random.default_rng(0)
    boots = [rng.choice(WORLDS, len(WORLDS)) for _ in range(1000)]
    out: dict = {"e1j_false_rate": target, "cells": {}}
    for c in CELLS:
        out["cells"][c] = {"E1-J": share(df=e1j, cell_name=c, worlds=np.array(WORLDS))}
        for (arm, seed), (df, rec, thr) in arms.items():
            sub = rec.filter(pl.col("cell") == c)
            au = auroc(y=sub["label"].to_numpy(), s=sub["score"].to_numpy()) if sub.height else float("nan")
            out["cells"][c][f"{arm}/{seed}"] = {
                "donor_found": share(df=df, cell_name=c, worlds=np.array(WORLDS)),
                "donor_found_ci": [float(x) for x in np.nanpercentile([share(df=df, cell_name=c, worlds=b) for b in boots[:300]], [2.5, 97.5])],
                "recipient_auroc": au, "threshold": thr,
                "n_donors": df.filter(pl.col("cell") == c).height, "n_pairs": sub.height,
            }  # fmt: skip
    (probe / "summary.json").write_text(json.dumps(out, indent=1))
    for c in CELLS:
        print(c, json.dumps({k: (round(v["donor_found"], 3), round(v["recipient_auroc"], 3)) if isinstance(v, dict) else round(v, 3) for k, v in out["cells"][c].items()}))


if __name__ == "__main__":
    main()

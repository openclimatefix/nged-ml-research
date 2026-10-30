# ruff: noqa: E501
"""Tune and check the detector of experiment 391 before any forecast score is read.

For each stage 1 version (`ens`, `era5`) the script builds the 6-hourly statistics, picks the
global threshold k as the smallest value on the grid 1.5 to 6.0 (step 0.05) that gives at most
2.0 false alarms per Primary-year after the neighbour veto over the tuning months, and writes
`work/detector/k.json`. It then prints the checks the plan fixes: recall on the logged transfers
that fall in the tuning months, the veto's two numbers (false alarms removed, labelled transfers
lost), false alarms per Primary-year by Primary, and the marked half-hours against the labelled
count. `--gate` also runs the known-answer gate on injected transfers (the ENS version only).

The tuning months are `TUNE_START` to `TUNE_END`: the stage 1 tables begin in 2024-10, and the
noise units need 30 days of history, so the 2024-04 start of the distillation design is not
available. A false alarm is an alarm that opens outside every labelled span (any source) that
touches the Primary or a listed neighbour, and the 72 hours after it.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00391-*/e00391_detector.py --gate
```
"""

import argparse
import csv
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00391_features as feats
import e00391_stage1 as stage1
import numpy as np
import polars as pl

LABELS: Final[Path] = Path(
    "/home/jack/dev/nged-substation-forecast/data/ml-experiments/briefs/sw_block_detector_label_table.csv"
)
OUT_DIR: Final[Path] = stage1.WORK_DIR / "detector"
TUNE_START: Final[datetime] = datetime(2024, 11, 1, tzinfo=UTC)
TUNE_END: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
K_GRID: Final[np.ndarray] = np.round(np.arange(1.5, 6.0 + 1e-9, 0.05), 2)
TARGET_RATE: Final[float] = 2.0
AFTER: Final[timedelta] = timedelta(hours=72)
SEED: Final[int] = 391
DURATIONS_DAYS: Final[tuple[float, ...]] = (0.5, 1, 2, 3, 7, 14)
SHARES: Final[tuple[float, ...]] = (0.10, 0.25)
SHARES_MW: Final[tuple[float, ...]] = (0.71, 1.1, 2.3)
"""Post hoc depths in MW: the quartiles of the label table's logged transfers."""
DEPTH_MW: list[bool] = [False]
"""Set by `--gate-mw`: `share` then means a depth in MW, not a share of the donor's level."""
RECEIVERS: Final[tuple[int, ...]] = (1, 2, 3)
GATE_RECALL: Final[float] = 0.55
GATE_VETO_KEEP: Final[float] = 0.95
GATE_PLACEBO_DROP: Final[float] = 0.5


def load_labels() -> list[dict]:
    """Return the label table rows with grid-index spans and the site codes in `pair`."""
    out = []
    for r in csv.DictReader(LABELS.open()):
        start = datetime.strptime(r["onset_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        end = (
            datetime.strptime(r["end_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            if r["end_utc"]
            else start + timedelta(hours=float(r["duration_hours"] or 12))
        )
        out.append(
            {
                "start": int(common.grid_index(times=start)),
                "end": int(common.grid_index(times=end)),
                "sites": set(re.findall(r"D\d+", r["pair"])),
                "source": r["source"],
                "drop": r["source"].startswith("D8 multi-day"),
                "transfer": r["source"].startswith("logged"),
            }
        )
    return out


def load_stage1(*, version: str) -> tuple[dict, dict, dict]:
    """Return per-site dense `u`, residual `r`, and the stage 1 table for a version."""
    size = common.grid_times().len()
    sites = list(feats.ADJACENCY)
    u, r, tables = {}, {}, {}
    for site in sites:
        table = pl.read_parquet(stage1.STAGE1_DIR / version / f"{site}.parquet")
        index = np.asarray(common.grid_index(times=table["valid_time"]))
        u[site], r[site] = np.full(size, np.nan, np.float32), np.full(size, np.nan, np.float32)
        u[site][index], r[site][index] = table["u"].to_numpy(), table["r_mw"].to_numpy()
        tables[site] = (index, table)
    return u, r, tables


def statistics(*, u: dict, r: dict, sites: list[str] | None = None) -> dict[str, feats.Origins]:
    """Return `Origins` for each site."""
    return {s: feats.origin_statistics(u=u[s], r=r[s]) for s in (sites or list(u))}


def openings(*, state: dict[str, np.ndarray]) -> np.ndarray:
    """Return the origin positions where an alarm opens (tag set) at a site."""
    return np.flatnonzero(state["tag"] > 0)


def false_alarm_counts(
    *, alarms: dict, labels: list[dict], first: int, last: int
) -> tuple[dict[str, int], float]:
    """Return false alarms per site over origins in `[first, last]` (grid index) and the years."""
    counts = {}
    for site, state in alarms.items():
        near = {site, *feats.ADJACENCY[site]}
        spans = [
            (lab["start"], lab["end"] + int(AFTER / common.STEP))
            for lab in labels
            if lab["sites"] & near
        ]
        n = 0
        for j in openings(state=state):
            at = j * feats.STEP_PER_ORIGIN
            if first <= at <= last and not any(a <= at <= b for a, b in spans):
                n += 1
        counts[site] = n
    return counts, (last - first) / (common.STEPS_PER_DAY * 365.25)


def tune(*, stats: dict, labels: list[dict], veto: bool, first: int, last: int) -> float:
    """Return the smallest k on the grid with at most `TARGET_RATE` false alarms per site-year."""
    for k in K_GRID:
        alarms = feats.alarm_state(stats=stats, k=float(k), veto=veto)
        counts, years = false_alarm_counts(alarms=alarms, labels=labels, first=first, last=last)
        if sum(counts.values()) / (len(counts) * years) <= TARGET_RATE:
            return float(k)
    return float(K_GRID[-1])


def recall(*, alarms: dict, labels: list[dict], first: int, last: int) -> tuple[int, int]:
    """Return logged transfers in `[first, last]` that an alarm at either Primary overlaps."""
    found = total = 0
    for lab in labels:
        if not lab["transfer"] or not (first <= lab["start"] <= last):
            continue
        total += 1
        hit = False
        for site in lab["sites"] & set(alarms):
            for j in openings(state=alarms[site]):
                at = j * feats.STEP_PER_ORIGIN
                if lab["start"] <= at <= lab["end"] + common.STEPS_PER_DAY:
                    hit = True
        found += hit
    return found, total


def marked_half_hours(*, alarms: dict, first: int, last: int) -> int:
    """Return half-hours inside open alarms, over all sites, between `first` and `last`."""
    return int(
        sum(
            int(state["open"][first // 12 : last // 12].sum()) * feats.STEP_PER_ORIGIN
            for state in alarms.values()
        )
    )


def inject(
    *,
    tables: dict,
    u: dict,
    r: dict,
    donor: str,
    receivers: list[str],
    onset: int,
    steps: int,
    share: float,
    placebo: bool,
) -> tuple[dict, dict]:
    """Return copies of `u` and `r` with a transfer (or a common-mode placebo) injected.

    The donor loses `share` of its typical level (the median of the absolute stage 1 median) for
    `steps` half-hours from `onset`, and the receivers split the load equally. The placebo gives
    the donor and every receiver the same-sign step of size `share` of their own level. The normal
    score is recomputed from the stored quantiles.
    """
    u2, r2 = dict(u), dict(r)
    sites = [donor, *receivers]
    donor_level = (
        1.0
        if DEPTH_MW[0]
        else float(np.nanmedian(np.abs(tables[donor][1]["median_mw"].to_numpy())))
    )
    for site in sites:
        index, table = tables[site]
        level = 1.0 if DEPTH_MW[0] else float(np.nanmedian(np.abs(table["median_mw"].to_numpy())))
        if placebo:
            delta = -share * level
        elif site == donor:
            delta = -share * donor_level
        else:
            delta = share * donor_level / len(receivers)
        inside = (index >= onset) & (index < onset + steps)
        quantiles = table.select(common.LEVEL_COLUMNS).to_numpy()
        actual = table["median_mw"].to_numpy() + table["r_mw"].to_numpy()
        actual = np.where(inside, actual + delta, actual)
        new_u = stage1.pit_normal_score(quantiles=quantiles, actual=actual)
        new_r = actual - table["median_mw"].to_numpy()
        u2[site], r2[site] = u[site].copy(), r[site].copy()
        u2[site][index], r2[site][index] = new_u, new_r
    return u2, r2


def gate(*, k: float, u: dict, r: dict, tables: dict, first: int, last: int) -> None:
    """Run the known-answer gate on injected transfers and print recall and the placebo."""
    rng = np.random.default_rng(SEED)
    donors = [s for s in feats.ADJACENCY if len(feats.ADJACENCY[s]) >= 3][:10]
    shares = SHARES_MW if DEPTH_MW[0] else SHARES
    rows = []
    for donor in donors:
        for days in DURATIONS_DAYS:
            for share in shares:
                for n in RECEIVERS:
                    steps = int(days * common.STEPS_PER_DAY)
                    onset = int(
                        rng.integers(
                            first + 30 * common.STEPS_PER_DAY,
                            last - steps - 2 * common.STEPS_PER_DAY,
                        )
                    )
                    receivers = list(feats.ADJACENCY[donor][:n])
                    for placebo in (
                        (False, True) if n == 2 and share == shares[-1] and days >= 2 else (False,)
                    ):
                        u2, r2 = inject(
                            tables=tables,
                            u=u,
                            r=r,
                            donor=donor,
                            receivers=receivers,
                            onset=onset,
                            steps=steps,
                            share=share,
                            placebo=placebo,
                        )
                        sites = [donor, *receivers, *feats.ADJACENCY[donor]]
                        stats = {
                            s: feats.origin_statistics(u=u2[s], r=r2[s])
                            if s in (donor, *receivers)
                            else base_stats[s]
                            for s in dict.fromkeys(sites)
                        }
                        hits = {}
                        for veto in (False, True):
                            state = feats.alarm_state(stats=stats, k=k, veto=veto)[donor]
                            at = openings(state=state) * feats.STEP_PER_ORIGIN
                            hits[veto] = bool(
                                ((at >= onset) & (at <= onset + steps + common.STEPS_PER_DAY)).any()
                            )
                        rows.append((donor, days, share, n, placebo, hits[False], hits[True]))
    frame = pl.DataFrame(
        rows,
        schema=["donor", "days", "share", "receivers", "placebo", "plain", "veto"],
        orient="row",
    )
    real = frame.filter(~pl.col("placebo"))
    shares = SHARES_MW if DEPTH_MW[0] else SHARES
    long25 = real.filter((pl.col("share") == shares[-1]) & (pl.col("days") >= 2))
    print("\n== Known-answer gate ==")
    print(
        real.group_by("days", "share")
        .agg(pl.col("plain").mean(), pl.col("veto").mean())
        .sort("days", "share")
    )
    print(
        real.group_by("receivers")
        .agg(pl.col("plain").mean(), pl.col("veto").mean())
        .sort("receivers")
    )
    placebo = frame.filter(pl.col("placebo"))
    split = real.filter(pl.col("receivers") >= 2)
    keep = float(split["veto"].sum()) / max(1, float(split["plain"].sum()))
    drop = 1 - float(placebo["veto"].sum()) / max(1, float(placebo["plain"].sum()))
    recall_long = float(long25["plain"].mean())  # ty: ignore[invalid-argument-type]
    print(f"recall (plain) at 25% and 2 days or more: {recall_long:.2f} (gate >= {GATE_RECALL})")
    print(f"veto keeps {keep:.2f} of plain hits on 2 and 3 receivers (gate >= {GATE_VETO_KEEP})")
    print(
        f"veto drops {drop:.2f} of placebo alarms (gate >= {GATE_PLACEBO_DROP}); n placebo {placebo.height}"
    )
    passed = recall_long >= GATE_RECALL and keep >= GATE_VETO_KEEP and drop >= GATE_PLACEBO_DROP
    print("GATE PASSED" if passed else "GATE FAILED")


base_stats: dict[str, feats.Origins] = {}


def main() -> None:
    """Tune k per version, print the detector checks, and optionally run the gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument(
        "--gate-mw", action="store_true", help="post hoc gate with depths in MW; reads k.json"
    )
    args = parser.parse_args()
    labels = load_labels()
    first = int(common.grid_index(times=TUNE_START))
    last = int(common.grid_index(times=TUNE_END))
    if args.gate_mw:
        k = float(json.loads((OUT_DIR / "k.json").read_text())["ens"])
        u, r, tables = load_stage1(version="ens")
        base_stats.update(statistics(u=u, r=r))
        sigma = {s: float(np.nanmedian(base_stats[s].r_sigma)) for s in base_stats}
        print(
            "noise scale (median r_sigma, MW) per Primary:",
            {s: round(v, 2) for s, v in sigma.items()},
        )
        print("injected depths (MW):", SHARES_MW)
        DEPTH_MW[0] = True
        gate(k=k, u=u, r=r, tables=tables, first=first, last=last)
        return
    ks: dict[str, float] = {}
    for version in stage1.VERSIONS:
        u, r, tables = load_stage1(version=version)
        stats = statistics(u=u, r=r)
        base_stats.update(stats)
        k_veto = tune(stats=stats, labels=labels, veto=True, first=first, last=last)
        k_plain = tune(stats=stats, labels=labels, veto=False, first=first, last=last)
        ks[version] = k_veto
        print(f"\n== {version}: k with veto {k_veto}, k without veto {k_plain} ==")
        for name, veto, k in (
            ("veto", True, k_veto),
            ("plain, same k", False, k_veto),
            ("plain, own k", False, k_plain),
        ):
            alarms = feats.alarm_state(stats=stats, k=k, veto=veto)
            counts, years = false_alarm_counts(alarms=alarms, labels=labels, first=first, last=last)
            found, total = recall(alarms=alarms, labels=labels, first=first, last=last)
            marked = marked_half_hours(alarms=alarms, first=first, last=last)
            print(
                f"{name}: false alarms {sum(counts.values())} ({sum(counts.values()) / (len(counts) * years):.2f} per Primary-year), transfers found {found} of {total}, marked half-hours {marked:,}"
            )
            if name == "veto":
                print("by Primary:", counts)
        if version == "ens" and args.gate:
            gate(k=k_veto, u=u, r=r, tables=tables, first=first, last=last)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "k.json").write_text(json.dumps(ks))
    print("wrote", OUT_DIR / "k.json", ks)


if __name__ == "__main__":
    main()

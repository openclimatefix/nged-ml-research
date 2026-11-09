"""Evidence 3 and 4: cross-injector recall and recall by size, with experiment 653's detector.

    DATA_PATH_INTERNAL=... PYTHONPATH=<this folder> uv run --no-sync python e00660_ev34_detector.py

The detector is experiment 653's penalised segmentation of the Primaries' residuals (imported by
path from that experiment's folder; its code is not changed). "Training" the detector means
choosing its two penalties (`on_cost`, `stay_cost`) on injected worlds of 2022 and 2023, exactly as
experiment 653 did, under the same cap on the alarm rate on the real tuning years.

- Evidence 3: tune on configuration A, test on configuration B, and the reverse, against tuning on
  the test configuration itself. A is constant-MW square steps; B is shaped transfers with a
  different duration law (short-heavy). The recall drop is in-configuration minus cross.
- Evidence 4: recall by size in noise units on default-configuration worlds, beside the same
  detector's recall on NGED's logged events of at least a day (scored years only).
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import e00660_data as data
import e00660_generator as gen
import numpy as np
import polars as pl

sys.path.insert(
    0, str(Path(__file__).parents[1] / "00653-edge-state-event-table-penalised-segmentation")
)
import e00653_common as c653  # ty: ignore[unresolved-import]
from e00653_core import Frozen, decode_world, freeze, match  # ty: ignore[unresolved-import]
from e00653_inject import Truth  # ty: ignore[unresolved-import]
from e00653_segment import Params  # ty: ignore[unresolved-import]

SCORED_FROM = c653.SCORED_FROM
GRID_LENGTH = c653.GRID_LENGTH
TUNE_ON = (4.0, 6.0, 8.0)
TUNE_STAY = (0.02, 0.05, 0.1)
MAX_REAL_RATE = 6.0
N_TUNE_WORLDS = 4
N_TEST_WORLDS = 8
SIZE_BINS = (1.0, 2.0, 4.0, 6.0, 10.0, 20.01)
POSITIVE = ("edge_transfer", "shift")


def truths(*, events: pl.DataFrame, lo: int, hi: int) -> list[Truth]:
    """Positive rows starting in [lo, hi) as experiment 653's `Truth` objects."""
    out = []
    for r in events.filter(pl.col("label").is_in(POSITIVE)).to_dicts():
        if not lo <= r["start"] < hi:
            continue
        if (r["role"] == "donor" and r["partners"]) or (r["role"] == "recipient" and r["partners"]):
            kind = "edge"
        elif r["event_class"] == "permanent_shift":
            kind = "permanent_shift"
        else:
            kind = "solo"
        partner_amp = {}
        if r["role"] == "donor":
            for p in r["partners"]:
                rec = events.filter(
                    (pl.col("group_id") == r["group_id"])
                    & (pl.col("site") == p)
                    & (pl.col("role") == "recipient")
                )
                partner_amp[p] = float(rec["delta"][0]) if rec.height else 0.0
        out.append(
            Truth(
                site=r["site"],
                start=int(r["start"]),
                end=None if r["end"] is None else int(r["end"]),
                amp_mw=float(r["delta"]),
                units=float(r["units"]),
                kind=kind,
                group=int(r["group_id"]),
                partners=list(r["partners"]),
                shape=r["shape"],
                partner_amp=partner_amp,
            )
        )
    return out


class Rig:
    """Panels, forecasts, and frozen noise units shared by every world."""

    def __init__(self) -> None:
        """Load the panel and the experiment 653 inputs."""
        self.panel = data.load_panel()
        self.p653 = c653.Panel()
        self.forecast = {s: data.load_forecast(site=s) for s in self.panel.sites}
        self.starts = data.logged_start_times()
        real = self.p653.mask_common_mode(residual=self.p653.residual)
        self.real_residual = real
        tune = {
            s: np.where(np.arange(GRID_LENGTH) < SCORED_FROM, v, np.nan) for s, v in real.items()
        }
        self.real_tune = tune
        self.frozen: Frozen = freeze(panel=self.p653, residual=tune)
        self.tune_panel = gen.PowerPanel(
            power={
                s: np.where(np.arange(GRID_LENGTH) < SCORED_FROM, v, np.nan)
                for s, v in self.panel.power.items()
            },
            is_mva=self.panel.is_mva,
            noise=self.panel.noise,
            neighbours=self.panel.neighbours,
            origin=self.panel.origin,
        )

    def residual_of(self, *, world: gen.InjectedWorld) -> dict[str, np.ndarray]:
        """The detector's input: injected power minus the lag-free forecast, common mode removed."""
        raw = {s: world.power[s] - self.forecast[s] for s in self.panel.sites}
        return self.p653.mask_common_mode(residual=raw)

    def decode(self, *, residual: dict[str, np.ndarray], params: Params) -> list:
        """Decode a residual panel with the masked frozen slow level."""
        events, _ = decode_world(
            panel=self.p653, residual=residual, frozen=self.frozen, arm="masked", params=params
        )
        return events


def mw_recall(rows: pl.DataFrame, *, big: bool) -> float:
    """MW-weighted share found; `big` keeps events of 4 noise units or more lasting a day."""
    if big:
        rows = rows.filter((pl.col("units") >= 4) & ((pl.col("days") >= 1) | (pl.col("days") < 0)))
    w = rows["amp_mw"].abs()
    if rows.height == 0 or float(w.sum()) == 0:
        return float("nan")
    return float((rows["found"].cast(pl.Float64) * w).sum() / w.sum())  # ty: ignore[unsupported-operator]


def world_rows(
    *, rig: Rig, world: gen.InjectedWorld, params: Params, lo: int, hi: int
) -> pl.DataFrame:
    """Score a world's positive events starting in [lo, hi)."""
    residual = rig.residual_of(world=world)
    events = rig.decode(residual=residual, params=params)
    rows = match(truth=truths(events=world.events, lo=lo, hi=hi), events=events, world_end=hi)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def make_worlds(
    *, rig: Rig, config: gen.GeneratorConfig, seeds: list[int], tune: bool
) -> list[gen.InjectedWorld]:
    """Inject `config` into the tuning years (power after 2023 removed) or the full record."""
    panel = rig.tune_panel if tune else rig.panel
    return [gen.inject(panel=panel, config=config, seed=s, start_times=rig.starts) for s in seeds]


def tune_detector(*, rig: Rig, worlds: list[gen.InjectedWorld]) -> tuple[Params, pl.DataFrame]:
    """Pick the penalties with the best recall under the real-alarm-rate cap (as experiment 653)."""
    n_years = SCORED_FROM / (365.25 * gen.DAY)
    out = []
    for on, stay in itertools.product(TUNE_ON, TUNE_STAY):
        p = Params(on_cost=on, off_cost=on, stay_cost=stay)
        rate = len(rig.decode(residual=rig.real_tune, params=p)) / (len(rig.panel.sites) * n_years)
        recalls = []
        for w in worlds:
            rows = world_rows(rig=rig, world=w, params=p, lo=0, hi=SCORED_FROM)
            recalls.append(mw_recall(rows, big=True))
        out.append(
            {"on": on, "stay": stay, "real_rate": rate, "recall": float(np.nanmean(recalls))}
        )
    table = pl.DataFrame(out)
    ok = table.filter(pl.col("real_rate") <= MAX_REAL_RATE)
    pick = (ok if ok.height else table).sort("recall", descending=True).row(0, named=True)
    return Params(on_cost=pick["on"], off_cost=pick["on"], stay_cost=pick["stay"]), table


def bootstrap_worlds(values: list[float], rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean and 95% interval over worlds (resampling whole worlds)."""
    v = np.array([x for x in values if np.isfinite(x)])
    boots = [rng.choice(v, size=len(v)).mean() for _ in range(2000)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cross_injector(*, rig: Rig) -> dict:
    """Evidence 3: recall of detectors tuned on A or B, tested on A and B."""
    rng = np.random.default_rng(660)
    configs = {
        "A_constant_square": gen.constant_mw_square(),
        "B_shaped_short_heavy": gen.shaped_transfers(),
    }
    tune_worlds = {
        k: make_worlds(rig=rig, config=cfg, seeds=list(range(100, 100 + N_TUNE_WORLDS)), tune=True)
        for k, cfg in configs.items()
    }
    test_worlds = {
        k: make_worlds(rig=rig, config=cfg, seeds=list(range(500, 500 + N_TEST_WORLDS)), tune=False)
        for k, cfg in configs.items()
    }
    params, tables = {}, {}
    for k in configs:
        params[k], tables[k] = tune_detector(rig=rig, worlds=tune_worlds[k])
    cells = []
    for tuned, tested in itertools.product(configs, configs):
        big, allsz = [], []
        for w in test_worlds[tested]:
            rows = world_rows(
                rig=rig, world=w, params=params[tuned], lo=SCORED_FROM, hi=GRID_LENGTH
            )
            big.append(mw_recall(rows, big=True))
            allsz.append(mw_recall(rows, big=False))
        cells.append(
            {
                "tuned_on": tuned,
                "tested_on": tested,
                "recall_big": bootstrap_worlds(big, rng),
                "recall_all": bootstrap_worlds(allsz, rng),
                "per_world_big": big,
                "params": [params[tuned].on_cost, params[tuned].stay_cost],
            }
        )
    drops = []
    for tested in configs:
        own = next(c for c in cells if c["tuned_on"] == tested and c["tested_on"] == tested)
        for tuned in configs:
            if tuned == tested:
                continue
            cross = next(c for c in cells if c["tuned_on"] == tuned and c["tested_on"] == tested)
            d = np.array(own["per_world_big"]) - np.array(cross["per_world_big"])
            drops.append(
                {
                    "tuned_on": tuned,
                    "tested_on": tested,
                    "drop": bootstrap_worlds(list(d), rng),
                }
            )
    return {"cells": cells, "drops": drops, "tuning": {k: t.to_dicts() for k, t in tables.items()}}


def real_events_scored() -> pl.DataFrame:
    """Real events visible in the telemetry, of at least a day, starting in the scored years.

    These are the transfers that experiment 580 found in the telemetry alone (both Primaries of
    each pair), plus the labelled events it classed as an unpaired step. Events logged by NGED
    but invisible in the telemetry are left out, because a detector cannot be asked to find them.
    The set is chosen by a step detector, which flatters the recall of any step detector.
    """
    sigma = json.loads((c653.WORK / "noise_units.json").read_text())["sigma_block_mw"]
    offset = int((data.GRID_START - data.LABEL_ORIGIN).total_seconds() // 1800)
    rows = []

    def add(site: str, on: object, end: object, depth: object, source: str) -> None:
        if site not in sigma or on is None or end is None or depth is None:
            return
        start, stop = int(on) - offset, int(end) - offset  # ty: ignore[invalid-argument-type]
        if SCORED_FROM <= start < GRID_LENGTH and stop - start >= gen.DAY:
            rows.append(
                {
                    "site": site,
                    "start": start,
                    "end": stop,
                    "amp_mw": float(depth),  # ty: ignore[invalid-argument-type]
                    "units": abs(float(depth)) / sigma[site],  # ty: ignore[invalid-argument-type]
                    "source": source,
                }
            )

    for r in pl.read_parquet(data.E580 / "events_verified.parquet").to_dicts():
        add(r["primary_a"], r["onset_a_idx"], r["end_a_idx"], r["depth_a_mw"], "verified pair")
        add(r["primary_b"], r["onset_b_idx"], r["end_b_idx"], r["depth_b_mw"], "verified pair")
    for r in pl.read_parquet(data.E580 / "events_labelled_status.parquet").to_dicts():
        if str(r["final_class"]).startswith("unpaired step"):
            add(
                r["label"], r["fit_onset_idx"], r["fit_end_idx"], r["fit_depth_mw"], "unpaired step"
            )
    return pl.DataFrame(rows)


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval of a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z, p = 1.96, k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def recall_by_size(*, rig: Rig, params: Params) -> dict:
    """Evidence 4: recall by size on injected default worlds and on real events."""
    rng = np.random.default_rng(661)
    worlds = make_worlds(
        rig=rig,
        config=gen.default_config(),
        seeds=list(range(900, 900 + N_TEST_WORLDS)),
        tune=False,
    )
    inj = []
    for i, w in enumerate(worlds):
        rows = world_rows(rig=rig, world=w, params=params, lo=SCORED_FROM, hi=GRID_LENGTH)
        inj.append(rows.with_columns(world=pl.lit(i)))
    inj_all = pl.concat(inj).filter((pl.col("days") >= 1) | (pl.col("days") < 0))
    real = real_events_scored()
    truth = [
        Truth(
            site=r["site"],
            start=r["start"],
            end=r["end"],
            amp_mw=r["amp_mw"],
            units=r["units"],
            kind="solo",
            group=i,
        )
        for i, r in enumerate(real.to_dicts())
    ]
    decoded = rig.decode(residual=rig.real_residual, params=params)
    real_rows = pl.DataFrame(match(truth=truth, events=decoded, world_end=GRID_LENGTH))
    out = []
    for lo, hi in itertools.pairwise(SIZE_BINS):
        ib = inj_all.filter((pl.col("units") >= lo) & (pl.col("units") < hi))
        per_world = [float(g["found"].mean()) for _, g in ib.group_by("world") if g.height >= 1]  # ty: ignore[invalid-argument-type]
        rb = real_rows.filter((pl.col("units") >= lo) & (pl.col("units") < hi))
        k, n = int(rb["found"].sum()) if rb.height else 0, rb.height
        out.append(
            {
                "bin": f"{lo:g} to {hi:g}",
                "injected_n": ib.height,
                "injected_recall": bootstrap_worlds(per_world, rng) if per_world else None,
                "real_n": n,
                "real_found": k,
                "real_recall": (k / n if n else float("nan")),
                "real_interval": wilson(k, n),
            }
        )
    return {"bins": out, "real_events": real.height, "real_found": int(real_rows["found"].sum())}


if __name__ == "__main__":
    rig = Rig()
    data.WORK.mkdir(parents=True, exist_ok=True)
    if "cross" in sys.argv:
        res3 = cross_injector(rig=rig)
        (data.WORK / "ev3_cross_injector.json").write_text(
            json.dumps(res3, indent=1, default=float)
        )
        print(
            json.dumps(
                {
                    "drops": res3["drops"],
                    "cells": [
                        {k: v for k, v in c.items() if k != "per_world_big"} for c in res3["cells"]
                    ],
                },
                indent=1,
                default=float,
            )
        )
    if "size" in sys.argv:
        p = replace(Params(), **json.loads((c653.WORK / "params.json").read_text()))
        res4 = recall_by_size(rig=rig, params=p)
        (data.WORK / "ev4_recall_by_size.json").write_text(
            json.dumps(res4, indent=1, default=float)
        )
        print(json.dumps(res4, indent=1, default=float))

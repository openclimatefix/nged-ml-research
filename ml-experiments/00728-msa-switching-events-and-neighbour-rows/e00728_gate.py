# ruff: noqa: E501
"""Experiment 728: the harness gate G-H, checks GH1, GH2, and GH4 to GH9 (design section 5.1). CPU only.

Usage: `run.sh e00728_gate.py gh [--only GH4 ...]`

**No check reads a network, a GPU, or a stored forecast.** GH3 (reproduction of the stored `M*`
forecasts) needs the new prediction path and runs in phase 0. Every check prints its numbers, writes
them into `work/gate_h.json` in this experiment's data folder, and fails with an assertion message
that names the number. **A failed check stops the experiment: no waiver, and no rule is changed after
a count is seen.**

**GH4 runs first and stops the line.** It prints the count at every step of the filter: all temporary
moves of `g2`, with a metered mirror, with a donor step of at least 0.10 of capacity (the design
expects 81), passing the full visible-depth definition, and lasting 7 days or more (43 expected of
the 81), then the counted moves per fold month, mirror state, and duration band, and the rows per
move at the scored origins under both cuts of the counted slice. **Decision (a) of the maintainer:
the counted set is the design's visible-depth set. If its pooled count is below 60, or a fold month
holds fewer than 8, GH4 prints the floor-only count beside it and stops. The gate never falls back to
the floor-only set.**

**What the harness approximates.** The training draws of GH5 and GH7 use the 709 panel's power as
the cleaned power (`ctx.panel.power`) truncated at the first half-hour of the fold month (the
training world reads nothing later) and a training span of the 730 days that end 28 days before
the fold month. The fits use experiment 713's cleaned power and its own span, so the draws of the fits
differ in their NaN pattern and in their onsets. The checks test the generator, not the fits' arrays.

**GH5's rule, written before any real number was seen.** The draws are 50 per fold month, 400 in
all. The 50 draws of each of the 8 fold months are pooled into one table, and the gate runs 5 tests
per pooled fold (kind by sign, kind by Primary, kind by sign within each of the two groups, and the
balance of unmirrored rises and falls), two z-tests on the fall-share gap (one per group, over all
400 draws), and two pooled chi-squared tests (kind by group, and kind by Primary, over all 400
draws). That is 44 tests, read at the level 0.01 / 44 (a Bonferroni correction). A coverage stop
reads the counts: in each group and kind, at least 600 events over all the draws and at least 80 in
each fold month (the floors of GH4, scaled by the 10-fold increase of the draws), and at least one
event of each kind at every Primary. Any shortfall stops the line, with no waiver and no fallback,
and the maintainer is told. The training world is built on the cleaned power before the first
half-hour of the fold month and reads nothing later.

**The thresholds of GH1 and GH2, written before any number was seen.**

- GH1: the truth oracle's largest absolute row score is at most 1e-9 points of capacity; its mean
  follow share after a temporary move in force at the origin is within 1e-9 of 0; its mean follow
  share after a `flat` permanent change is within 0.05 of 1.
- GH2: the persistence oracle's mean follow share after a temporary move in force at the origin, and
  after a `flat` permanent change, are each within 0.15 of 1; its mean counted CRPS is above the truth
  oracle's (which is 0).
- The follow share after the other three shapes of permanent change is printed and is not a check:
  the three shapes (`profile_prop`, `daylight`, `quantised`) have a daily shape, so a mean over the
  visible half-hours does not equal the shift at one target half-hour.
"""

import argparse
import dataclasses
import hashlib
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Final

import e00714_common as common714  # ty: ignore[unresolved-import]
import e00714_generator2 as g2gen  # ty: ignore[unresolved-import]
import e00715_neighbours as nb  # ty: ignore[unresolved-import]
import e00728_score as sc
import e00728_worlds as wd
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00728-msa-switching-events-and-neighbour-rows"
GATE_FILE: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work" / "gate_h.json"
FLOOR_POOLED: Final = 60
FLOOR_PER_MONTH: Final = 8
EXPECTED_FLOOR_ONLY: Final = 81
EXPECTED_LASTING: Final = 43
N_DRAWS_PER_FOLD: Final = 50
"""Training draws of each of the 8 fold months in GH5 and GH7: 400 draws in all."""
TRAIN_SEEDS: Final = tuple(range(N_DRAWS_PER_FOLD))
DRAW_SCALE: Final = N_DRAWS_PER_FOLD // 5
"""The floors of GH4 were set for a world of 5 draws per fold month; GH5 scales them by this."""
SPAN_DAYS: Final = 730
SPAN_LAG_DAYS: Final = 28
ORACLE_TOLERANCE: Final = 1e-9
FLAT_FOLLOW_TOLERANCE: Final = 0.05
PERSISTENCE_TOLERANCE: Final = 0.15
CHECK_ORDER: Final = ("GH4", "GH6", "GH7", "GH5", "GH8", "GH9", "GH1", "GH2")
N_TESTS: Final = wd.N_INDEPENDENCE_TESTS * len(wd.FOLD_MONTHS) + 4
"""The tests of GH5: five on each of the 8 pooled folds, two z-tests on the fall-share gap (one per
group), and two pooled chi-squared tests (kind by group, and kind by Primary)."""
FAMILY_LEVEL: Final = wd.INDEPENDENCE_LEVEL / N_TESTS
SCRIPTS: Final = ("e00728_worlds.py", "e00728_score.py", "e00728_gate.py")


class Env:
    """The real panel, the weather, and the arrays of the 16 Primaries in `nb.LABELS` order."""

    def __init__(self) -> None:
        """Load the context and check that the panel's sites follow the order of the table."""
        self.ctx = common714.load_context()
        panel = self.ctx.panel
        assert tuple(panel.sites) == tuple(nb.LABELS), (panel.sites, nb.LABELS)
        self.capacity_mw = np.asarray(panel.capacity_mw, dtype=np.float64)
        self.mva_only = np.asarray(panel.is_mva, dtype=bool)
        floor = float(np.nanmin(panel.power[self.mva_only]))
        assert floor >= -wd.FLOOR_TOLERANCE, (
            f"the cleaned power of an MVA-only Primary goes below zero ({floor:.3f}): the floor rule of the training world would reject every event there"
        )
        self.table = nb.load_table()
        self.vis: dict[str, np.ndarray] = {}
        self.errors: dict[str, np.ndarray] = {}

    def visible(self, *, month: str) -> np.ndarray:
        """Return (and cache) the visible depth of each Primary for a fold month."""
        if month not in self.vis:
            self.errors[month] = wd.reference_error(ctx=self.ctx, month=month)
            self.vis[month] = wd.visible_depth(reference_error_share=self.errors[month])
        return self.vis[month]

    def span(self, *, month: str) -> tuple[int, int]:
        """Return the harness's training span `[first, last)` of a fold month, in grid half-hours."""
        last = (common714.month_start_day(month=month) - SPAN_LAG_DAYS) * common714.DAY
        return last - SPAN_DAYS * common714.DAY, last

    def end(self, *, month: str) -> int:
        """Return the first grid half-hour of a fold month, where the training world must stop."""
        return common714.month_start_day(month=month) * common714.DAY

    def clean(self, *, month: str) -> np.ndarray:
        """Return the cleaned power that a training draw of the fold month reads."""
        return self.ctx.panel.power[:, : self.end(month=month)]

    def draw(self, *, month: str, seed: int) -> wd.TrainingWorld:
        """Return one training draw of a fold month and a training seed."""
        first, last = self.span(month=month)
        return wd.training_world(
            clean=self.ctx.panel.power, end=self.end(month=month), first=first, last=last,
            world_seed=wd.train_seed(month=month, seed=seed), table=self.table,
            capacity_mw=self.capacity_mw, mva_only=self.mva_only, vis=self.visible(month=month),
        )  # fmt: skip


def print_frame(*, title: str, frame: object) -> None:
    """Print a titled table."""
    assert isinstance(frame, pl.DataFrame), type(frame)
    print(f"\n{title}")
    with pl.Config(tbl_rows=200, tbl_cols=40):
        print(frame)


def g2_event_table(*, env: Env) -> pl.DataFrame:
    """Return `g2`'s events with the visible-depth columns, for the four injected fold months."""
    vis_by_month = {m: env.visible(month=m) for m in wd.INJECTED_MONTHS}
    return wd.g2_events(capacity_mw=env.capacity_mw, vis_by_month=vis_by_month)


def scored_sites(*, row: dict) -> list[int]:
    """Return the Primaries at which a counted move is scored: the host, and a metered recipient."""
    sites = [int(row["host"])]
    if (
        row["mirror_state"] in {"full", "partial"}
        and row["recipient"] is not None
        and row["recipient"] >= 0
    ):
        sites.append(int(row["recipient"]))
    return sites


def gh4(*, env: Env) -> dict:
    """GH4: print every count of the counted event set, and apply the two floors."""
    for month in wd.FOLD_MONTHS:
        env.visible(month=month)
    errors = pl.DataFrame(
        {"primary": list(nb.LABELS), **{m: env.errors[m] * 100 for m in wd.FOLD_MONTHS}}
    )
    print_frame(
        title="Daily-mean reference error, points of capacity (NaN: no reference)", frame=errors
    )
    visible = pl.DataFrame(
        {"primary": list(nb.LABELS), **{m: env.vis[m] * 100 for m in wd.FOLD_MONTHS}}
    )
    print_frame(title="Visible depth, points of capacity", frame=visible)
    events = g2_event_table(env=env)
    report = wd.count_report(events=events)
    permanent = events.filter(pl.col("truth") == "era")
    visible_permanent = permanent.filter(pl.col("visible"))
    steps = {
        "events of g2": events.height,
        "permanent changes": permanent.height,
        "permanent changes of visible depth (the guard P set)": visible_permanent.height,
        "all temporary moves": report.all_temporary,
        "with a metered mirror": report.metered,
        "donor step at least 0.10 of capacity (design: 81)": report.floor_only,
        "  of those, lasting 7 days or more (design: 43 of the 81)": report.lasting_floor_only,
        "passing the visible depth (the counted set)": report.visible_depth,
        "  of those, lasting 7 days or more": report.lasting,
    }
    for name, value in steps.items():
        print(f"GH4 count: {name}: {value}")
    print_frame(title="Counted moves per fold month", frame=report.per_month)
    print_frame(title="Counted moves per mirror state", frame=report.per_mirror_state)
    print_frame(title="Counted moves per duration band", frame=report.per_band)
    print_frame(
        title="Permanent changes of visible depth, per shape and sign",
        frame=visible_permanent.with_columns(sign=pl.col("size").sign())
        .group_by("kind", "sign")
        .len()
        .sort("kind", "sign"),
    )
    counted = wd.counted_moves(events=events)
    counted = counted.with_columns(
        group=pl.col("host").map_elements(
            lambda h: "mva_only" if env.mva_only[h] else "signed_mw", return_dtype=pl.String
        )
    )
    print_frame(
        title="Counted moves per group of the scored host",
        frame=counted.group_by("group").len().sort("group"),
    )
    rows = []
    for row in counted.iter_rows(named=True):
        per_site = [
            sc.slice_rows(
                clean=env.ctx.panel.power[site], onset_day=row["onset_day"], end_day=row["end_day"]
            )
            for site in scored_sites(row=row)
        ]
        rows.append(
            {
                "month": row["month"], "pass_index": row["pass_index"], "host": row["host"],
                "mirror_state": row["mirror_state"], "onset_day": row["onset_day"],
                "duration_days": row["duration_days"], "n_sites": len(per_site),
                "nominal": per_site[0].nominal, "by_origin": per_site[0].by_origin,
                "past_grid": sum(r.past_grid for r in per_site),
                "no_truth": sum(r.no_truth for r in per_site),
                "scorable": sum(r.scorable for r in per_site), "slice_1": sum(r.slice_1 for r in per_site),
                "slice_2": sum(r.slice_2 for r in per_site),
                "min_scorable_per_site": min(r.scorable for r in per_site),
            }
        )  # fmt: skip
    per_move = pl.DataFrame(rows)
    print_frame(
        title="Counted moves: rows in the counted slice, summed over the scored Primaries (nominal is per Primary)",
        frame=per_move,
    )
    totals = {
        "scorable rows, slice 1 (move in force)": int(per_move["slice_1"].sum()),
        "scorable rows, slice 2 (7 days after the end)": int(per_move["slice_2"].sum()),
        "rows past the end of the grid": int(per_move["past_grid"].sum()),
        "rows inside the grid with a NaN truth": int(per_move["no_truth"].sum()),
    }
    for name, value in totals.items():
        print(f"GH4 rows: {name}: {value}")
    no_rows = per_move.filter(pl.col("scorable") == 0)
    one_site_empty = per_move.filter(pl.col("min_scorable_per_site") == 0)
    print(
        f"GH4: counted moves with no scorable row at any scored Primary: {no_rows.height}; with no scorable row at one scored Primary: {one_site_empty.height}"
    )
    ok = wd.pooled_floor_ok(report=report, pooled=FLOOR_POOLED, per_month=FLOOR_PER_MONTH)
    assert ok, (
        f"GH4 FAILS: the counted set holds {report.visible_depth} moves (floor {FLOOR_POOLED}) and "
        f"{report.per_month.to_dicts()} per fold month (floor {FLOOR_PER_MONTH} each); the floor-only "
        f"count is {report.floor_only}. Stop before any GPU job and tell the manager. The gate does not fall back."
    )
    assert no_rows.height == 0, (
        f"GH4 FAILS: {no_rows.height} counted moves have no scorable row in the slice"
    )
    return {
        **dict(steps), **totals, "per_month": report.per_month.to_dicts(),
        "no_scorable_rows": no_rows.height, "one_scored_primary_empty": one_site_empty.height,
    }  # fmt: skip


def gh6(*, env: Env) -> dict:
    """GH6: the training, `g2`, and held-out world seeds are pairwise disjoint."""
    train = [wd.train_seed(month=m, seed=s) for m in wd.FOLD_MONTHS for s in TRAIN_SEEDS]
    per_month = {m: g2gen.stored_indices(world="g2", month=m) for m in wd.INJECTED_MONTHS}
    g2_list = wd.g2_seeds(passes_per_month=per_month)
    held = [wd.held_out_seed(month=m) for m in wd.INJECTED_MONTHS]
    wd.assert_disjoint(train=train, g2=g2_list, held_out=held)
    wd.assert_train_seed_allowed(world_seed=train[0], test_seeds=[*g2_list, *held])
    print(
        f"GH6: {len(train)} training seeds, {len(g2_list)} g2 seeds, {len(held)} held-out seeds, pairwise disjoint"
    )
    print(
        f"GH6: training seeds {train[:3]}..., held-out seeds {held}, g2 seeds {min(g2_list)} to {max(g2_list)}"
    )
    return {"n_train": len(train), "n_g2": len(g2_list), "held_out": held}


def gh7(*, env: Env) -> dict:
    """GH7: clean targets and the cap on stacked permanent changes, in every training draw."""
    rows = []
    for month in wd.FOLD_MONTHS:
        first, last = env.span(month=month)
        for seed in TRAIN_SEEDS:
            world = env.draw(month=month, seed=seed)
            stats = wd.check_targets_and_limits(
                world=world, clean=env.clean(month=month), mva_only=env.mva_only
            )
            cover = wd.realised_cover(
                world=world, clean=env.clean(month=month), first=first, last=last
            )
            rows.append({"month": month, "seed": seed, **stats, "cover": cover})
    table = pl.DataFrame(rows)
    print_frame(
        title="GH7: every training draw (permanent_share is the share of events that are permanent changes)",
        frame=table,
    )
    return {"draws": rows}


def slim(world: wd.TrainingWorld) -> wd.TrainingWorld:
    """Return the draw without its two arrays, which GH5 does not read."""
    empty = np.zeros((1, 1))
    return dataclasses.replace(world, history=empty, target=empty)


def gh5(*, env: Env) -> dict:
    """GH5: balance of sign and site across the two kinds, in `g2` and in every training draw.

    The 50 draws of one fold month are pooled into one table, and every test is read at the family
    level: `INDEPENDENCE_LEVEL` divided by the number of tests of the gate (a Bonferroni
    correction). Only the events of a draw are kept, because the arrays of 400 draws would not fit
    in memory.
    """
    events = g2_event_table(env=env)
    signed = events.with_columns(sign=pl.col("size").sign())
    print_frame(
        title="GH5 g2: kind by sign of the scored step",
        frame=signed.group_by("truth", "sign").len().sort("truth", "sign"),
    )
    print_frame(
        title="GH5 g2: kind by Primary",
        frame=events.group_by("truth", "host")
        .len()
        .pivot(on="truth", index="host", values="len")
        .sort("host"),
    )
    print_frame(
        title="GH5 g2: kind by shape",
        frame=events.group_by("truth", "kind").len().sort("truth", "kind"),
    )
    print_frame(
        title="GH5 g2: permanent changes by sign and size",
        frame=events.filter(pl.col("truth") == "era").group_by("size").len().sort("size"),
    )
    level = FAMILY_LEVEL
    print(f"GH5: {N_TESTS} tests, family level {wd.INDEPENDENCE_LEVEL}, per-test level {level:.2e}")
    rows = []
    all_worlds: list[wd.TrainingWorld] = []
    worlds_by_month: dict[str, list[wd.TrainingWorld]] = {}
    for month in wd.FOLD_MONTHS:
        worlds = [slim(env.draw(month=month, seed=seed)) for seed in TRAIN_SEEDS]
        all_worlds.extend(worlds)
        worlds_by_month[month] = worlds
        tables = wd.independence_tables(worlds=worlds, mva_only=env.mva_only, level=level)
        for notice in tables["notices"]:  # ty: ignore[not-iterable]
            print(f"{month}: {notice}")
        unmirrored = tables["unmirrored_temporary"]
        assert isinstance(unmirrored, dict)
        print_frame(
            title=f"GH5 {month}: kind by sign within the MVA-only Primaries",
            frame=tables["kind_by_sign_mva_only"],
        )
        print_frame(
            title=f"GH5 {month}: kind by sign within the signed-MW Primaries",
            frame=tables["kind_by_sign_signed_mw"],
        )
        rows.append(
            {
                "month": month, "n_events": tables["n_events"], "p_kind_by_sign": tables["p_kind_by_sign"],
                "p_kind_by_host": tables["p_kind_by_host"], "p_sign_mva_only": tables["p_kind_by_sign_mva_only"],
                "p_sign_signed_mw": tables["p_kind_by_sign_signed_mw"], "unmirrored_n": unmirrored["n"],
                "unmirrored_rise_share": unmirrored["rises"] / max(unmirrored["n"], 1), "unmirrored_z": unmirrored["z"],
            }
        )  # fmt: skip
    print_frame(
        title=f"GH5 training draws, pooled per fold: every p-value must be at least {level:.2e}",
        frame=pl.DataFrame(rows),
    )
    pooled_tests = wd.kind_independence_pooled(
        worlds=all_worlds, mva_only=env.mva_only, level=level
    )
    print_frame(
        title="GH5 kind by group, over all the draws",
        frame=pooled_tests["table_group"],  # ty: ignore[invalid-argument-type]
    )
    print_frame(
        title="GH5 events per Primary and kind, over all the draws",
        frame=pooled_tests["per_primary"].pivot(on="kind", index="host", values="len").sort("host"),  # ty: ignore[unresolved-attribute]
    )
    print(
        f"GH5 pooled chi-squared p-values: kind by group {pooled_tests['p_kind_by_group']:.2e}, "
        f"kind by Primary {pooled_tests['p_kind_by_host_pooled']:.2e}, level {level:.2e}"
    )
    problems = wd.coverage_problems(
        worlds_by_month=worlds_by_month, mva_only=env.mva_only,
        pooled=FLOOR_POOLED * DRAW_SCALE, per_month=FLOOR_PER_MONTH * DRAW_SCALE,
    )  # fmt: skip
    assert not problems, (
        f"GH5 FAILS the coverage floors of GH4 in the training draws: {problems}. Stop before "
        "any GPU job and tell the manager. The gate does not fall back."
    )
    gaps = wd.fall_share_gap(worlds=all_worlds, mva_only=env.mva_only, level=level)
    print_frame(
        title=(
            f"GH5 over all {len(all_worlds)} draws: fall share of permanent changes minus that of "
            f"temporary moves, z-test at level {level:.2e}"
        ),
        frame=pl.DataFrame([{"group": g, **r} for g, r in gaps.items()]),
    )
    first_try: dict[str, list[int]] = {}
    for world in all_worlds:
        for key, (n, ok) in world.first_try.items():
            total = first_try.setdefault(key, [0, 0])
            total[0] += n
            total[1] += ok
    print_frame(
        title="GH5 first-try acceptance by kind, group and sign (floor test and limits on stacking)",
        frame=pl.DataFrame(
            [
                {"kind|group|sign": k, "n": n, "accepted": ok, "share": ok / max(n, 1)}
                for k, (n, ok) in sorted(first_try.items())
            ]
        ),
    )
    assert pooled_tests["passes"], (
        f"GH5 FAILS: kind is not independent of the group or of the Primary over all the draws "
        f"(p {pooled_tests['p_kind_by_group']:.2e} and {pooled_tests['p_kind_by_host_pooled']:.2e}, level {level:.2e})"
    )
    failed = {g: (r["gap"], r["p"]) for g, r in gaps.items() if not r["passes"]}
    assert not failed, f"GH5 FAILS the z-test on the fall-share gap (gap, p) per group: {failed}"
    return {"per_test_level": level, "folds": rows, "fall_gap": gaps}


def gh8(*, env: Env) -> dict:
    """GH8: no permanent change of `g2` ends inside the scored month (it still shows in the last 7 days)."""
    events = g2_event_table(env=env)
    era = events.filter(pl.col("truth") == "era")
    ended = era.filter(pl.col("end_day").is_not_null() & (pl.col("end_day") >= 0))
    assert ended.height == 0, f"GH8 FAILS: {ended.height} permanent changes of g2 have an end day"
    missing = []
    for month in wd.INJECTED_MONTHS:
        rows = era.filter(pl.col("month") == month)
        for index in sorted(set(rows["pass_index"].to_list())):
            gen = g2gen.read_pass(world="g2", month=month, index=index)
            hi = gen.power.start_hh + gen.power.power.shape[1]
            for ev in rows.filter(pl.col("pass_index") == index).iter_rows(named=True):
                injected = gen.power.get(site=ev["host"], lo=hi - 7 * common714.DAY, hi=hi)
                clean = env.ctx.panel.power[ev["host"], hi - 7 * common714.DAY : hi]
                diff = injected - clean
                if not (np.isfinite(diff) & (np.abs(diff) > 1e-9)).any():
                    missing.append((month, index, ev["host"], ev["kind"]))
    print(
        f"GH8: {era.height} permanent changes, none ends; {len(missing)} show no change in the last 7 days of the stored window"
    )
    assert not missing, (
        f"GH8 FAILS: a permanent change shows no change in the last 7 days: {missing[:5]}"
    )
    return {"n_permanent": era.height}


def gh9(*, env: Env) -> dict:
    """GH9: the host arrays that `Rb` and `Rbn` read for one fold and seed are identical.

    Each arm's draw is made independently from the seed, and the hashes of its history and target
    arrays are compared. A fit records the same hash (`wd.world_hash`) in its log, and the counted
    reading compares the two arms' hashes.
    """
    month, seed = wd.FOLD_MONTHS[0], 0
    draws = {"Rb": env.draw(month=month, seed=seed), "Rbn": env.draw(month=month, seed=seed)}
    hashes = {arm: wd.world_hash(world=w) for arm, w in draws.items()}
    wd.check_shared_draw(host_a=draws["Rb"].history, host_b=draws["Rbn"].history)
    wd.check_shared_draw(host_a=draws["Rb"].target, host_b=draws["Rbn"].target)
    assert hashes["Rb"] == hashes["Rbn"], f"Rb and Rbn hash differently: {hashes}"
    assert draws["Rb"].events == draws["Rbn"].events, (
        "two draws of one seed give different event tables"
    )
    print(
        f"GH9: {month} seed {seed}: Rb and Rbn arrays hash to {hashes['Rb']} ({len(draws['Rb'].events)} events)"
    )
    return {"month": month, "seed": seed, "hashes": hashes}


def oracle_readings(
    *, env: Env, oracle: Callable[..., pl.DataFrame]
) -> dict[str, sc.OracleReading]:
    """Score an oracle on `g2`, fold month by fold month, and return one reading per month."""
    out = {}
    for month in wd.INJECTED_MONTHS:
        injected_frames, real_frames = [], []
        for index in g2gen.stored_indices(world="g2", month=month):
            gen = g2gen.read_pass(world="g2", month=month, index=index)
            for row, ev in enumerate(gen.truth.iter_rows(named=True)):
                sites = [ev["host"]]
                if (
                    ev["truth"] == "switch"
                    and ev["mirror_state"] in {"full", "partial"}
                    and ev["recipient"] >= 0
                ):
                    sites.append(ev["recipient"])
                for site in sites:
                    inj = gen.power.get(site=site, lo=0, hi=common714.N_HH)
                    rows = sc.rows_for_event(
                        event_id=f"{month}_p{index}_e{row}", primary=site, event=ev, injected=inj,
                        clean=env.ctx.panel.power[site],
                    )  # fmt: skip
                    injected_frames.append(oracle(rows=rows, world="injected"))
                    real_frames.append(oracle(rows=rows, world="real"))
        out[month] = sc.read_oracle(
            injected=pl.concat(injected_frames), real=pl.concat(real_frames)
        )
    return out


def pool(*, readings: dict[str, sc.OracleReading]) -> dict[str, float]:
    """Pool the monthly readings of an oracle: the largest row score, and the mean of the monthly means.

    The mean of the monthly means gives each of the four injected fold months equal weight, as the
    bootstrap of the counted contrasts does at its top level.
    """

    def mean_of(field: str) -> float:
        values = {m: getattr(r, field) for m, r in readings.items()}
        bad = [m for m, v in values.items() if not np.isfinite(v)]
        assert not bad, f"the reading `{field}` is not finite in fold month(s) {bad}: {values}"
        return float(np.mean(list(values.values())))

    return {
        "max_abs_score": max(r.max_abs_score for r in readings.values()),
        "mean_counted_crps": mean_of("mean_counted_crps"),
        "follow_permanent_flat": mean_of("follow_permanent"),
        "follow_permanent_all": mean_of("follow_permanent_all"),
        "follow_temporary_in_force": mean_of("follow_temporary"),
        "n_temporary": sum(r.n_temporary for r in readings.values()),
        "n_permanent": sum(r.n_permanent for r in readings.values()),
    }


def gh1(*, env: Env) -> dict:
    """GH1: the truth oracle scores zero and follows permanent changes alone."""
    readings = oracle_readings(env=env, oracle=sc.truth_oracle)
    pooled = pool(readings=readings)
    print(f"GH1 truth oracle: {pooled}")
    assert pooled["max_abs_score"] <= ORACLE_TOLERANCE, (
        f"GH1 FAILS: largest row score {pooled['max_abs_score']}"
    )
    assert abs(pooled["follow_temporary_in_force"]) <= ORACLE_TOLERANCE, pooled
    assert abs(pooled["follow_permanent_flat"] - 1.0) <= FLAT_FOLLOW_TOLERANCE, pooled
    return pooled


def gh2(*, env: Env) -> dict:
    """GH2: the persistence oracle follows both kinds and scores worse than the truth oracle."""
    readings = oracle_readings(env=env, oracle=sc.persistence_oracle)
    pooled = pool(readings=readings)
    print(f"GH2 persistence oracle: {pooled}")
    assert abs(pooled["follow_temporary_in_force"] - 1.0) <= PERSISTENCE_TOLERANCE, pooled
    assert abs(pooled["follow_permanent_flat"] - 1.0) <= PERSISTENCE_TOLERANCE, pooled
    assert pooled["mean_counted_crps"] > ORACLE_TOLERANCE, pooled
    return pooled


CHECKS: Final[dict[str, Callable[..., dict]]] = {
    "GH1": gh1, "GH2": gh2, "GH4": gh4, "GH5": gh5, "GH6": gh6, "GH7": gh7, "GH8": gh8, "GH9": gh9,
}  # fmt: skip


def file_hashes() -> dict[str, str]:
    """Return the sha256 of the scripts that the gate ran."""
    here = Path(__file__).parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in SCRIPTS}


def run(*, only: list[str]) -> int:
    """Run the checks in `CHECK_ORDER`, record each result, and return 0 if all pass.

    The record `gate_h.json` is always written, also when a check raises, and holds the failure.
    """
    results: dict[str, object] = {"scripts": file_hashes()}
    failed: list[str] = []
    try:
        env = Env()
        for name in CHECK_ORDER:
            if only and name not in only:
                continue
            print(f"\n===== {name} =====", flush=True)
            try:
                results[name] = {"pass": True, **CHECKS[name](env=env)}
            except Exception as exc:  # noqa: BLE001  (a check may fail in any way; the record keeps it)
                results[name] = {
                    "pass": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failed.append(name)
                print(f"{name} FAILED: {type(exc).__name__}: {exc}", flush=True)
                if name == "GH4":
                    break
    except Exception as exc:  # noqa: BLE001  (the environment itself failed to load)
        results["environment"] = {
            "pass": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        failed.append("environment")
        print(f"the environment failed to load: {type(exc).__name__}: {exc}", flush=True)
    finally:
        GATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = GATE_FILE.with_name(GATE_FILE.name + ".tmp")
        tmp.write_text(json.dumps(results, indent=1, default=str))
        tmp.rename(GATE_FILE)
    print(
        f"\nG-H: {'PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}; record {GATE_FILE}"
    )
    return 1 if failed else 0


def main() -> int:
    """Parse the command line and run G-H."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["gh"])
    parser.add_argument("--only", nargs="*", default=[], choices=sorted(CHECKS))
    args = parser.parse_args()
    return run(only=args.only)


if __name__ == "__main__":
    sys.exit(main())

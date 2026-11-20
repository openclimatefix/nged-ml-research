# ruff: noqa: E501, PT018, PLW3301, D205, D209, PLR0912, PLR0915, PERF401
"""Experiment 715: the injected worlds (design section 7.3): the ceiling table, KN11, the
predictions, and S1 to S7.

Usage:
    e00715_era.py ceiling                         # print the ceiling table; assert it equals the
    design's to 3 decimals
    e00715_era.py kn11a MONTH                     # KN11(a), before any injected-world score is read
    e00715_era.py predict ARM WORLD MONTH SEED    # prediction only: ARM on the injected passes of
    WORLD (`g2` or `bank`)
    e00715_era.py follow WORLD                    # follow shares of every stored arm; S1 to S7 with
    their intervals

**Prediction only.** The onset lies after every fit's cutoff, so every fitted state was trained on
the real
power. `predict` loads the stored stage 3 state of an arm, reads the injected panel as the host's history
and as the neighbour power (mode `predict_injected`: the real neighbours of an injected host carry their
own injected power too), and forecasts each event's host at the origins 7, 10, 14, and 21 days after
the
onset at all 14 lead days. **Every origin day is bounded by the days `ens.run_us` holds**, checked by
`e00715_batch.check_origin_days`, and an origin outside them is dropped and counted in the log.

**Two worlds.** `g2` is experiment 714's second generator (`00714-*/worlds/g2/<month>_p<n>.npz` and the
truth parquet). `bank` is experiment 709's bank, read through experiment 713's adapter. A permanent
change (`truth == "era"`) is scored against the injected power and a temporary move against the power
without the move, which is the cleaned power of the host, as experiment 709 scored it.
"""

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

os.environ.setdefault("E00667_PHASE", "predict")

import e00713_era as era713  # ty: ignore[unresolved-import]
import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00715_arms as arms
import e00715_batch as fb
import e00715_neighbours as nb
import e00715_predict as pr
import e00715_train as tr
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_714: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00714-*"))
ORIGIN_DAYS: Final = (7, 10, 14, 21)
FOLLOW_LEAD_DAYS: Final = range(2, 11)
MIN_FULL_ROWS: Final = 4
DRAWS: Final = 10_000
PUBLISHED_CEILING: Final[dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]]] = {
    ("all", "known_answer"): ((0.916, 0.924), (0.72, 0.77)),
    ("all", "host"): ((0.630, 0.637), (0.23, 0.22)),
    ("all", "host_nbr"): ((0.666, 0.659), (0.24, 0.28)),
    ("metered", "known_answer"): ((1.000, 1.000), (1.00, 1.00)),
    ("metered", "host"): ((0.622, 0.639), (0.20, 0.23)),
    ("metered", "host_nbr"): ((0.699, 0.690), (0.32, 0.33)),
    ("metered", "host_nbr_rewired"): ((0.612, 0.640), (0.19, 0.22)),
    ("hidden", "known_answer"): ((0.776, 0.796), (0.43, 0.43)),
    ("hidden", "host"): ((0.644, 0.633), (0.29, 0.22)),
    ("hidden", "host_nbr"): ((0.612, 0.606), (0.20, 0.19)),
}  # fmt: skip
SUBSETS: Final = {
    "all": None,
    "metered": ["full", "partial", "none"],
    "hidden": ["hidden", "none"],
}


def auroc(*, label: np.ndarray, score: np.ndarray) -> float:
    """Return the area under the ROC curve (ties share their average rank)."""
    from scipy.stats import rankdata

    label = np.asarray(label).astype(int)
    n1, n0 = int(label.sum()), int((1 - label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    rank = rankdata(score)
    return float((rank[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gap(*, label: np.ndarray, score: np.ndarray) -> float:
    """Return the largest true-positive rate minus false-positive rate along the score."""
    label = np.asarray(label).astype(int)
    order = np.argsort(-np.asarray(score), kind="stable")
    y = label[order]
    tpr, fpr = np.cumsum(y) / y.sum(), np.cumsum(1 - y) / (1 - y).sum()
    return float(max(0.0, (tpr - fpr).max()))


def ceiling_table(*, check: bool = True) -> pl.DataFrame:
    """Print the ceiling table from experiment 714's out-of-fold scores, and assert it equals the
    design's to 3 decimals."""
    rows = []
    for (subset, column), (aurocs, gaps) in PUBLISHED_CEILING.items():
        mine_a, mine_g = [], []
        for k in (7, 14):
            d = pl.read_parquet(DATA_714 / "results" / f"oof_g2_k{k}.parquet")
            states = SUBSETS[subset]
            if states is not None:
                d = d.filter(pl.col("mirror_state").is_in(states))
            label, score = d["label"].to_numpy(), d[column].to_numpy()
            mine_a.append(round(auroc(label=label, score=score), 3))
            mine_g.append(round(gap(label=label, score=score), 2))
        rows.append(
            {
                "events": subset,
                "classifier": column,
                "n": d.height,
                "auroc_7": mine_a[0],
                "auroc_14": mine_a[1],
                "gap_7": mine_g[0],
                "gap_14": mine_g[1],
            }
        )
        if check:
            assert tuple(mine_a) == aurocs and tuple(mine_g) == gaps, (
                subset,
                column,
                mine_a,
                mine_g,
                aurocs,
                gaps,
            )
    table = pl.DataFrame(rows)
    print(
        "Ceiling table (experiment 714's out-of-fold scores, 7 and 14 days after the onset); reproduced to 3 decimals"
    )
    print(table)
    return table


def g2_panel(
    *, month: str, index: int, clean: np.ndarray
) -> tuple[np.ndarray, pl.DataFrame, list[int]]:
    """Return the injected panel of a second-generator pass on the adapter's grid, its truth table,
    and its touched Primaries."""
    import e00714_generator2 as g2  # ty: ignore[unresolved-import]

    gen = g2.read_pass(world="g2", month=month, index=index)
    out = clean.astype(np.float64).copy()
    lo, n = gen.power.start_hh, gen.power.power.shape[1]
    assert gen.power.power.shape[0] == clean.shape[0] and lo + n <= clean.shape[1], (
        lo,
        n,
        clean.shape,
    )
    out[:, lo : lo + n] = gen.power.power
    return out, gen.truth, list(gen.touched)


def bank_panel(
    *, month: str, index: int, clean: np.ndarray
) -> tuple[np.ndarray, pl.DataFrame, list[int]]:
    """Return the injected panel of an experiment 709 bank pass through experiment 713's adapter."""
    ipass = era713.load_pass(month=month, index=index)
    return (
        era713.full_panel(injected=ipass["power"], clean=clean),
        ipass["truth_table"],
        sorted(ipass["touched"]),
    )


def passes_of(*, world: str, month: str) -> list[int]:
    """Return the stored pass indices of `world` for `month`."""
    if world == "g2":
        import e00714_generator2 as g2  # ty: ignore[unresolved-import]

        return g2.stored_indices(world="g2", month=month)
    return list(era713.PASSES)


def panel_of(
    *, world: str, month: str, index: int, clean: np.ndarray
) -> tuple[np.ndarray, pl.DataFrame, list[int]]:
    """Return (panel, truth table, touched Primaries) of one pass of `world`."""
    return (g2_panel if world == "g2" else bank_panel)(month=month, index=index, clean=clean)


def slot_flags(
    *, truth: pl.DataFrame, touched: list[int], table: nb.NeighbourTable, world: str
) -> pl.DataFrame:
    """Add the two leakage flags of design section 9 (rules 3 and 4) to a pass's truth table.

    `slot_other_event` is true where the real slots of an event's host hold a site of another event
    of the pass (a host, or the recipient of a transfer). `rewired_touched` is true where the
    rewired slots of the host, for any seed of the fits, hold a Primary that carries an injected
    change in the pass. On the second generator, rule 3 is asserted: no host has such a slot.
    """
    hosts = truth["host"].to_list()
    recipients = (
        truth["recipient"].to_list() if "recipient" in truth.columns else [None] * len(hosts)
    )
    sites = [
        {int(h)} | ({int(r)} if r is not None and int(r) >= 0 else set())
        for h, r in zip(hosts, recipients, strict=True)
    ]
    other, rewired = [], []
    for i, h in enumerate(hosts):
        mine = set(nb.real_slots(table=table, host=int(h))) - {-1}
        elsewhere: set[int] = set().union(*[x for j, x in enumerate(sites) if j != i])
        other.append(bool(mine & elsewhere))
        rew: set[int] = set()
        for seed in arms.SEEDS:
            rew |= set(nb.rewired_slots(table=table, host=int(h), seed=seed)) - {-1}
        rewired.append(bool(rew & {int(t) for t in touched}))
    if world == "g2":
        assert not any(other), (
            f"design section 9 rule 3: a slot of the second generator holds another event's site, events {[i for i, o in enumerate(other) if o]}"
        )
    return truth.with_columns(
        slot_other_event=pl.Series(other, dtype=pl.Boolean),
        rewired_touched=pl.Series(rewired, dtype=pl.Boolean),
    )


def visible_change(
    *, panel: np.ndarray, clean: np.ndarray, host: int, onset_day: int, c_idx: int
) -> float:
    """Return the mean injected change of the host's visible half-hours after the onset (NaN if none).

    A half-hour is visible when its grid index is at most the native origin's index minus 48.
    """
    newest = c_idx - pr.lay.NATIVE_OFFSET_STEPS - 48
    lo = onset_day * 48
    if newest < lo:
        return float("nan")
    d = panel[host, lo : newest + 1] - clean[host, lo : newest + 1]
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else float("nan")


def window_of(*, truth: pl.DataFrame, n_grid: int) -> tuple[int, int]:
    """Return the grid half-hour range `[lo, hi)` that holds every event of a pass."""
    starts = truth["onset_day"].to_numpy() * 48
    ends = np.where(truth["end_day"].to_numpy() < 0, n_grid, (truth["end_day"].to_numpy() + 1) * 48)
    return int(starts.min()), int(ends.max())


def kn11a(*, month: str, device: str = "cpu") -> dict:
    """KN11(a): the injected panels of both worlds equal the cleaned power outside the touched
    Primaries and windows."""
    pipe = inp.load_pipeline(month=month, phase="predict")
    data = inp.build_fold_data(pipe=pipe, device=device)
    clean = data.clean
    checked = 0
    for world in ("g2", "bank"):
        for index in passes_of(world=world, month=month):
            panel, truth, touched = panel_of(world=world, month=month, index=index, clean=clean)
            deviates = np.isfinite(panel) & np.isfinite(clean) & (np.abs(panel - clean) > 1e-6)
            lo, hi = window_of(truth=truth, n_grid=clean.shape[1])
            allowed = np.zeros(clean.shape, dtype=bool)
            for p in touched:
                allowed[p, lo:hi] = True
            outside = deviates & ~allowed
            assert not outside.any(), (
                f"{world} pass {index}: {int(outside.sum())} half-hours deviate outside the touched windows, Primaries {sorted(set(np.nonzero(outside)[0].tolist()))}"
            )
            hosts = set(truth["host"].to_list())
            assert hosts <= set(touched), f"{world} pass {index}: a host is not a touched Primary"
            checked += 1
    print(
        f"KN11(a) {month}: {checked} passes of the second generator and the bank equal the cleaned power outside the touched windows"
    )
    return {"pass": True, "passes": checked, "month": month}


def kn11b(*, month: str, seed: int) -> dict:
    """KN11(b): F4's prediction path with a stored state of `M*` reproduces experiment 713's bank
    forecasts of `M*` (1e-4 of capacity)."""
    star = arms.a713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    folder = era713.BANK713 / f"{star}_{month}_s{seed}"
    stored = sorted(folder.glob("pass[0-9][0-9].parquet")) if folder.exists() else []
    if not stored:
        return {
            "skipped": f"experiment 713 has not stored bank forecasts of {star} for {month} s{seed}"
        }
    out = injected_forecasts(
        arm=star,
        world="bank",
        month=month,
        seed=seed,
        data_dir=ML_EXPERIMENTS_DATA_DIR,
        origins="713",
        device="cpu",
    )
    worst = 0.0
    for path in stored:
        mine = pl.read_parquet(out / path.name)
        theirs = pl.read_parquet(path)
        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
        j = mine.join(theirs, on=keys, suffix="_s")
        assert j.height > 0, f"{path.name}: no common rows"
        worst = max(worst, max(float((j[q] - j[f"{q}_s"]).abs().max()) for q in pr.QCOLS))  # ty: ignore[invalid-argument-type]
    return {"max_abs_difference": worst, "pass": worst <= 1e-4}


def event_origins(*, truth: pl.DataFrame, ens_days: tuple[int, int]) -> list[tuple[int, int, int]]:
    """Return (event index, days after the onset, contract origin index) inside the days `ens_days =
    (first, last)` of the ensemble table."""
    out = []
    for i, ev in enumerate(truth.iter_rows(named=True)):
        for k in ORIGIN_DAYS:
            day = ev["onset_day"] + k
            if ens_days[0] <= day <= ens_days[1]:
                out.append((i, k, day * 48 + 24))
    return out


def injected_forecasts(
    *, arm: str, world: str, month: str, seed: int, data_dir: Path, origins: str, device: str
) -> Path:
    """Forecast each event's host of every pass at the sparse origins; write `pass<NN>.parquet`;
    return the folder."""
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]

    folder = arms.root(data_dir=data_dir) / "work" / "injected" / world / f"{arm}_{month}_s{seed}"
    arms.install_own_cache()
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    log: dict = {"arm": arm, "world": world, "month": month, "seed": seed, **arms.code_record()}
    model, neighbours, spec = pr.load_network(
        arm=arm, month=month, seed=seed, device=device, data_dir=data_dir, accept=[], record=log
    )
    table = nb.load_table()
    base = arms.f4_source(
        data=data, table=table, mode=spec.slots if spec else "none", rewire_seed=seed
    )
    ens = data.source.ens
    assert ens is not None
    ens_days = (int(ens.first_day), int(ens.first_day + ens.filled.shape[1] - 1))
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    dropped = 0
    for index in passes_of(world=world, month=month):
        out = folder / f"pass{index:02d}.parquet"
        if out.exists():
            continue
        panel, truth, touched = panel_of(world=world, month=month, index=index, clean=data.clean)
        truth = slot_flags(truth=truth, touched=touched, table=table, world=world)
        z = torch.from_numpy(
            ((panel - data.target_mean[:, None]) / data.target_std[:, None]).astype(np.float32)
        ).to(device)
        named: dict[fb.ArrayName, torch.Tensor] = {"injected": z}
        host_arr, nbr_arr, label = fb.arrays_for_mode(mode="predict_injected", named=named)
        src = base.with_arrays(host_power_in=host_arr, neighbour_power=nbr_arr, label=label)
        pairs = event_origins(truth=truth, ens_days=ens_days)
        dropped += len(truth) * len(ORIGIN_DAYS) - len(pairs)
        hosts = np.array([truth["host"][i] for i, _, _ in pairs])
        c = np.array([o for _, _, o in pairs])
        fb.check_origin_days(ens=ens, c_idx=c, what=f"{world} pass {index}")
        changes = np.array(
            [
                visible_change(
                    panel=panel, clean=data.clean, host=int(truth["host"][i]),
                    onset_day=int(truth["onset_day"][i]), c_idx=int(o),
                )
                for i, _, o in pairs
            ]
        )  # fmt: skip
        samples = tr.F4Samples(primary=hosts, c_idx=c, is_val=np.zeros(len(c), bool))  # ty: ignore[unknown-argument]
        frames = []
        with torch.no_grad():
            for lead in range(14):
                for start in range(0, len(c), 512):
                    sel = np.arange(start, min(start + 512, len(c)))
                    b = tr.batch_for(
                        source=src,
                        samples=samples,
                        index=sel,
                        lead=np.full(len(sel), lead),
                        ensemble=True,
                        cut=10**9,
                        dropout=None,
                    )
                    out_t = tr.run_model(model=model, batch=b, neighbours=neighbours)
                    q = pr.mdl.standardised_quantiles(out=out_t)
                    p = hosts[sel]
                    q = (
                        torch.sort(
                            q * torch.from_numpy(data.target_std).to(q.device)[p][:, None, None]
                            + torch.from_numpy(data.target_mean).to(q.device)[p][:, None, None],
                            dim=-1,
                        )
                        .values.cpu()
                        .numpy()
                    )
                    tidx = b.host.target_idx.cpu().numpy()
                    frames.append(
                        pl.DataFrame(
                            {
                                "time_series_id": np.repeat(ids[p], 48).astype(np.int64),
                                "event": np.repeat(np.array([pairs[s][0] for s in sel]), 48),
                                "days_after_onset": np.repeat(
                                    np.array([pairs[s][1] for s in sel]), 48
                                ),
                                "origin_us": np.repeat(
                                    prep422.GRID_START_US
                                    + (c[sel] - pr.lay.NATIVE_OFFSET_STEPS) * prep422.STEP_US,
                                    48,
                                ),
                                "valid_us": prep422.GRID_START_US
                                + tidx.reshape(-1) * prep422.STEP_US,
                                "lead_day": np.full(len(sel) * 48, lead + 1),
                                "change": np.repeat(changes[sel], 48),
                                "truth_share": panel[p[:, None], tidx].reshape(-1),
                                "clean_share": data.clean[p[:, None], tidx].reshape(-1),
                                **{
                                    n: q[:, :, i].reshape(-1).astype(np.float32)
                                    for i, n in enumerate(pr.QCOLS)
                                },
                            }
                        )
                    )
        frame = (
            pl.concat(frames)
            .with_columns(
                power_fcst_init_time=pl.col("origin_us")
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("UTC"),
                valid_time=pl.col("valid_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            )
            .drop("origin_us", "valid_us")
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        truth_out = out.with_name(f"pass{index:02d}_truth.parquet")
        truth.write_parquet(truth_out)
        tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
        frame.write_parquet(tmp)
        tmp.rename(out)
        print(arm, world, month, seed, "pass", index, frame.height, "rows", flush=True)
    (folder / "log.json").write_text(
        json.dumps(
            {**log, "origins_dropped_outside_the_ensemble_table": dropped}, indent=1, default=str
        )
    )
    print(
        f"{arm} {world} {month} s{seed}: {dropped} event-origins fell outside the days the ensemble table holds and were dropped",
        flush=True,
    )
    return folder


def follow_share(*, injected_q50: np.ndarray, real_q50: np.ndarray, change: float) -> float:
    """Return the follow share of one event at one origin: the mean median difference over the
    target rows, over the mean injected change."""
    assert change != 0.0
    return float(np.mean(injected_q50 - real_q50) / change)


def bootstrap_contrast(
    *,
    table: pl.DataFrame,
    arm_a: str,
    arm_b: str,
    subset: list[str] | None,
    k: int,
    draws: int = DRAWS,
    seed: int = 0,
) -> dict:
    """Return the AUROC of the follow share of `arm_a` minus that of `arm_b` at `k` days, with a 95%
    interval that resamples whole passes within months, then months."""
    d = table.filter(pl.col("days_after_onset") == k)
    if subset is not None:
        d = d.filter(pl.col("mirror_state").is_in(subset))
    keys = ["month", "pass_id", "event"]
    wide = d.pivot(on="arm", index=[*keys, "label"], values="follow").drop_nulls([arm_a, arm_b])
    point = auroc(label=wide["label"].to_numpy(), score=wide[arm_a].to_numpy()) - auroc(
        label=wide["label"].to_numpy(), score=wide[arm_b].to_numpy()
    )
    rng = np.random.default_rng(seed)
    months = sorted(wide["month"].unique().to_list())
    by_month = {
        m: [g for _, g in wide.filter(pl.col("month") == m).group_by("pass_id")] for m in months
    }
    est = []
    for _ in range(draws):
        picked = []
        for m in rng.choice(months, size=len(months)):
            groups = by_month[m]
            picked += [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        s = pl.concat(picked)
        est.append(
            auroc(label=s["label"].to_numpy(), score=s[arm_a].to_numpy())
            - auroc(label=s["label"].to_numpy(), score=s[arm_b].to_numpy())
        )
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return {"diff": point, "low": float(lo), "high": float(hi), "n_events": wide.height}


SEEDS: Final = arms.SEEDS
KEYS: Final = ("time_series_id", "power_fcst_init_time", "valid_time")
UNDERPOWERED_BELOW: Final = 60
NEVER_ENDS: Final = 10**6
"""The `end_day` stand-in of a temporary move that does not end inside the grid."""
SLICES_SWITCH: Final = ("in force", "after the end, 7 days", "switch to end+7 d")
SLICE_ERA: Final = "era 7-28 d"


def injected_folder(*, world: str, arm: str, month: str, seed: int) -> Path:
    """Return the folder of an arm's injected-world forecasts (`injected_forecasts` writes it)."""
    return (
        arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR)
        / "work"
        / "injected"
        / world
        / f"{arm}_{month}_s{seed}"
    )


def pass_files(*, folder: Path) -> list[Path]:
    """Return the forecast files `pass<NN>.parquet` of a folder, in order."""
    return sorted(folder.glob("pass[0-9][0-9].parquet"))


def mean_gap(*, label: np.ndarray, score: np.ndarray) -> float:
    """Return the mean follow share after permanent changes minus that after temporary moves (S4)."""
    ok = np.isfinite(score)
    pos, neg = score[ok & (label == 1)], score[ok & (label == 0)]
    return float(pos.mean() - neg.mean()) if pos.size and neg.size else float("nan")


def event_crps(*, forecast: pl.DataFrame, events: pl.DataFrame, ids: np.ndarray) -> list[dict]:
    """Return one row per (event, slice) with the mean CRPS, in points of capacity, of the host's rows.

    A permanent change is scored on its injected power over the origins 7 to 28 days after the onset.
    A temporary move is scored on the power without the move, in three slices: while it is in
    force, the 7 days after it ends (none when it does not end), and experiment 713's combined slice.
    """
    import e00709_features as features  # ty: ignore[unresolved-import]
    from studies.eval_protocol.metrics import quantile_score

    origin_day = (
        (
            forecast["power_fcst_init_time"].dt.replace_time_zone(None)
            - features.GRID_START.replace(tzinfo=None)
        )
        .dt.total_days()
        .to_numpy()
    )
    site = forecast["time_series_id"].to_numpy()
    q = forecast.select(list(pr.QCOLS)).to_numpy().astype(np.float64)
    out = []
    for index, event in enumerate(events.iter_rows(named=True)):
        host_rows = site == ids[event["host"]]
        onset = event["onset_day"]
        if event["truth"] == "era":
            slices = {SLICE_ERA: (onset + 7, onset + 28)}
            column = "truth_share"
        else:
            end = (
                event["end_day"]
                if event["end_day"] is not None and event["end_day"] >= 0
                else NEVER_ENDS
            )
            slices = {"in force": (onset, end), "switch to end+7 d": (onset, end + 7)}
            if end != NEVER_ENDS:
                slices["after the end, 7 days"] = (end, end + 7)
            column = "clean_share"
        actual = forecast[column].to_numpy().astype(np.float64)
        for label, (lo, hi) in slices.items():
            sel = host_rows & (origin_day >= lo) & (origin_day < hi) & np.isfinite(actual)
            if sel.any():
                crps = float(
                    np.mean(quantile_score(actual=actual[sel] * 100.0, quantiles=q[sel] * 100.0))
                )
                out.append({"event": index, "slice": label, "crps": crps, "n_rows": int(sel.sum())})
    return out


def read_arm_world(*, world: str, arm: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (follow rows, CRPS rows) of an arm on a world, over the bank folds and both seeds.

    The follow rows hold one row per (month, seed, pass, event, origin day): the follow share and
    the event's truth fields and leakage flags. A missing folder is skipped and printed.
    """
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00715_score as s715

    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    follow_rows: list[dict] = []
    crps_rows: list[dict] = []
    for month in arms.BANK_FOLDS:
        for seed in SEEDS:
            folder = injected_folder(world=world, arm=arm, month=month, seed=seed)
            files = pass_files(folder=folder)
            if not files:
                print(f"missing: {arm} {world} {month} s{seed} has no injected forecasts")
                continue
            real = (
                s715.load_arm(arm=arm, seeds=(seed,), months=(month,))
                .select(*KEYS, pl.col("q50").alias("q50_real"))
                .with_columns(
                    power_fcst_init_time=pl.col("power_fcst_init_time") - timedelta(hours=3)
                )
            )  # the stored real forecasts carry the contract origin; the injected ones the native origin
            for path in files:
                forecast = pl.read_parquet(path)
                events = pl.read_parquet(path.with_name(path.stem + "_truth.parquet"))
                tag = {
                    "arm": arm,
                    "world": world,
                    "month": month,
                    "seed": seed,
                    "pass_id": path.stem,
                }
                crps_rows += [
                    {**tag, **r} for r in event_crps(forecast=forecast, events=events, ids=ids)
                ]
                window = forecast.filter(pl.col("lead_day").is_between(*FOLLOW_LEAD_DAYS_RANGE))
                joined = window.join(real, on=list(KEYS), how="inner")
                assert joined.height == window.height, (
                    f"{arm} {world} {month} s{seed} {path.stem}: {window.height - joined.height} injected rows have no real-world twin"
                )
                per = (
                    joined.group_by("event", "days_after_onset")
                    .agg(
                        (pl.col("q50") - pl.col("q50_real")).mean().alias("shift"),
                        pl.col("change").first(),
                    )
                    .with_columns(
                        follow=pl.when(
                            pl.col("change").is_finite() & (pl.col("change").abs() > 1e-6)
                        )
                        .then(pl.col("shift") / pl.col("change"))
                        .otherwise(None)
                    )
                )
                flags = events.with_row_index("event").select(
                    "event", "truth", "slot_other_event", "rewired_touched",
                    (pl.col("mirror_state") if "mirror_state" in events.columns else pl.lit("bank")).alias("mirror_state"),
                )  # fmt: skip
                for r in per.join(flags, on="event").iter_rows(named=True):
                    follow_rows.append({**tag, **r})
    return pl.DataFrame(follow_rows), pl.DataFrame(crps_rows)


FOLLOW_LEAD_DAYS_RANGE: Final = (FOLLOW_LEAD_DAYS.start, FOLLOW_LEAD_DAYS.stop - 1)


def with_keep(*, frame: pl.DataFrame, world: str, readings: str) -> pl.DataFrame:
    """Drop the events that design section 9 excludes, for `readings` `s1_s3` or `s4_s7`; return the rest.

    Rule 4 drops an event from S1 to S3 when its rewired slots hold a touched Primary. Rule 3 drops,
    on the bank, a temporary move whose donor has another event's site in a slot, from every reading.
    """
    keep = pl.lit(True)
    if readings == "s1_s3":
        keep = keep & ~pl.col("rewired_touched")
    if world == "bank":
        keep = keep & ~(pl.col("slot_other_event") & (pl.col("truth") == "switch"))
    out = frame.filter(keep)
    print(
        f"{world} {readings}: dropped {frame.height - out.height} of {frame.height} rows by the leakage rules (design section 9)"
    )
    return out


def pass_bootstrap_ci(
    *, frame: pl.DataFrame, column: str, draws: int = DRAWS, seed: int = 0
) -> dict:
    """Return the mean of `column` over events and its 95% interval, resampling passes within months, then months."""
    d = frame.drop_nulls(column)
    months = sorted(d["month"].unique().to_list())
    by_month = {
        m: [g[column].to_numpy() for _, g in d.filter(pl.col("month") == m).group_by("pass_id")]
        for m in months
    }
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(draws):
        picked: list[np.ndarray] = []
        for m in rng.choice(months, size=len(months)):
            groups = by_month[m]
            picked += [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        est.append(float(np.concatenate(picked).mean()))
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {
        "mean": float(d[column].to_numpy().mean()),
        "low": float(lo),
        "high": float(hi),
        "n_events": d.height,
    }


def crps_contrast(*, crps: pl.DataFrame, treatment: str, reference: str, slice_name: str) -> dict:
    """Return the mean CRPS of `treatment` minus `reference` over the events of a slice, in points of capacity."""
    per = (
        crps.filter(pl.col("slice") == slice_name)
        .group_by("arm", "month", "pass_id", "event")
        .agg(pl.col("crps").mean())
    )
    keys = ["month", "pass_id", "event"]
    a = per.filter(pl.col("arm") == treatment).drop("arm")
    b = per.filter(pl.col("arm") == reference).drop("arm")
    both = a.join(b, on=keys, suffix="_ref").with_columns(d=pl.col("crps") - pl.col("crps_ref"))
    if both.is_empty():
        return {"n_events": 0}
    return pass_bootstrap_ci(frame=both, column="d")


def cmd_follow(*, world: str) -> dict:
    """Print and return S1 to S7 for `world` (`g2` or `bank`), at 95%.

    Every reading is secondary, and the count of readings is printed beside them (design 7.3).
    """
    import e00713_arms as a713  # ty: ignore[unresolved-import]
    import e00715_score as s715

    star = a713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    names = [star, "F4", "F4-R", "F4-W", "F4-WR"]
    follow_parts, crps_parts = [], []
    for arm in names:
        f, c = read_arm_world(world=world, arm=arm)
        if f.is_empty():
            continue
        follow_parts.append(f)
        crps_parts.append(c)
    assert follow_parts, f"no injected forecasts of any arm on world {world}: run `predict` first"
    follow = pl.concat(follow_parts, how="diagonal")
    crps = pl.concat(crps_parts, how="diagonal")
    result: dict = {"world": world, "star": star}
    # per event, the seeds are averaged first, as the other readings do
    base = follow.group_by(
        "arm",
        "month",
        "pass_id",
        "event",
        "days_after_onset",
        "truth",
        "mirror_state",
        "slot_other_event",
        "rewired_touched",
    ).agg(pl.col("follow").mean())
    base = base.with_columns(label=(pl.col("truth") == "era").cast(pl.Int64))
    result["ceiling"] = "printed by `e00715_era.py ceiling`"

    def reading(*, a: str, b: str, subset: str | None, k: int, readings: str) -> dict:
        tab = with_keep(frame=base, world=world, readings=readings)
        sub = SUBSETS[subset] if subset else None
        return bootstrap_contrast(table=tab, arm_a=a, arm_b=b, subset=sub, k=k)

    s123 = {}
    if world == "g2":
        for label, a, b, subset in (
            ("S1", "F4", "F4-R", "metered"), ("S2", "F4-W", "F4-WR", "metered"),
            ("S3 (F4)", "F4", "F4-R", "hidden"), ("S3 (F4-W)", "F4-W", "F4-WR", "hidden"),
        ):  # fmt: skip
            for k in (7, 14):
                if {a, b} <= set(base["arm"].unique().to_list()):
                    s123[f"{label} at {k} d"] = reading(
                        a=a, b=b, subset=subset, k=k, readings="s1_s3"
                    )
    else:
        for k in (7, 14):
            if {"F4", "F4-R"} <= set(base["arm"].unique().to_list()):
                s123[f"S7: S1 at {k} d"] = reading(
                    a="F4", b="F4-R", subset=None, k=k, readings="s4_s7"
                )
    result["S1_S3_or_S7"] = s123
    kept = with_keep(frame=base, world=world, readings="s4_s7")
    result["S4"] = {
        f"{arm} at {k} d": {
            "mean_gap": mean_gap(label=d["label"].to_numpy(), score=d["follow"].to_numpy()),
            "roc_gap": gap(label=d["label"].to_numpy(), score=d["follow"].drop_nulls().to_numpy()) if d["follow"].null_count() == 0 else None,
        }
        for arm in names
        for k in (7, 14)
        if not (d := kept.filter((pl.col("arm") == arm) & (pl.col("days_after_onset") == k))).is_empty()
    }  # fmt: skip
    # events dropped by rule 3 are dropped from the CRPS rows too
    drop_ids = (
        follow.filter(pl.col("slot_other_event") & (pl.col("truth") == "switch"))
        .select("month", "pass_id", "event")
        .unique()
        if world == "bank"
        else None
    )
    if drop_ids is not None:
        crps = crps.join(drop_ids, on=["month", "pass_id", "event"], how="anti")
    counts = (
        crps.filter(pl.col("arm") == "F4")
        .unique(["month", "pass_id", "event", "slice"])
        .group_by("slice")
        .agg(pl.len().alias("n_events"))
    )
    result["events_per_slice"] = {r["slice"]: r["n_events"] for r in counts.iter_rows(named=True)}
    result["underpowered_slices"] = [
        s for s, n in result["events_per_slice"].items() if n < UNDERPOWERED_BELOW
    ]
    s56 = {}
    pairs = [(star, "F4"), ("F4-R", "F4"), ("F4-WR", "F4-W")]
    for ref, t in pairs:
        for slice_name in (SLICE_ERA, *SLICES_SWITCH):
            c = crps_contrast(crps=crps, treatment=t, reference=ref, slice_name=slice_name)
            if c.get("n_events"):
                s56[f"{t} minus {ref}: {slice_name}"] = c
    if world == "bank":
        for t in ("F4", "F4-W", star):
            for slice_name in SLICES_SWITCH:
                c = crps_contrast(
                    crps=_with_candidate(crps=crps, arm=t),
                    treatment=t,
                    reference="C",
                    slice_name=slice_name,
                )
                if c.get("n_events"):
                    s56[f"{t} minus C: {slice_name}"] = c
    result["S5_S6"] = s56
    result["count_of_readings"] = len(s123) + len(result["S4"]) + len(s56)
    result["note"] = (
        "all readings are secondary, at 95%, and change no verdict; quote none without the count of readings"
    )
    if world == "bank":
        result["note"] += (
            "; bank readings carry the bank's other cues (the leakage section, rule 3)"
        )
    print(json.dumps(result, indent=1, default=float))
    s715.write(name=f"era_{world}", obj=result)
    if world == "g2":
        donor = s56.get(f"F4 minus {star}: switch to end+7 d")
        if donor:
            s715.write(
                name="s6_donor_harm",
                obj={
                    "f4_minus_mstar": donor["mean"],
                    "interval": [donor["low"], donor["high"]],
                    "n_events": donor["n_events"],
                    "slice": "switch to end+7 d",
                    "world": "g2",
                },
            )
    return result


def _with_candidate(*, crps: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Return `crps` plus the Candidate's (`C`) CRPS rows of the events and keys that `arm` holds, bank only."""
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00709_features as features  # noqa: F401  # ty: ignore[unresolved-import]

    ids = k578.sites()["time_series_id"].to_numpy()
    d709 = era713.D709
    rows = []
    for month in arms.BANK_FOLDS:
        for seed in SEEDS:
            folder = injected_folder(world="bank", arm=arm, month=month, seed=seed)
            for path in pass_files(folder=folder):
                events = pl.read_parquet(path.with_name(path.stem + "_truth.parquet"))
                mine = pl.read_parquet(path, columns=list(KEYS))
                ref = pl.read_parquet(d709 / "bank" / f"none_{month}_s{seed}" / path.name).join(
                    mine, on=list(KEYS), how="inner"
                )
                rows += [
                    {
                        "arm": "C",
                        "world": "bank",
                        "month": month,
                        "seed": seed,
                        "pass_id": path.stem,
                        **r,
                    }
                    for r in event_crps(forecast=ref, events=events, ids=ids)
                ]
    return pl.concat([crps, pl.DataFrame(rows)], how="diagonal") if rows else crps


def selftest() -> int:
    """CPU checks of the arithmetic with no data: AUROC, gap, follow share, origin selection."""
    y = np.array([0, 0, 1, 1])
    assert auroc(label=y, score=np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert auroc(label=y, score=np.array([0.4, 0.3, 0.2, 0.1])) == 0.0
    assert auroc(label=y, score=np.zeros(4)) == 0.5
    assert gap(label=y, score=np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert (
        abs(
            follow_share(injected_q50=np.full(48, 0.5), real_q50=np.full(48, 0.2), change=0.3) - 1.0
        )
        < 1e-12
    )
    truth = pl.DataFrame({"host": [3], "onset_day": [100]})
    got = event_origins(truth=truth, ens_days=(100, 114))
    assert [k for _, k, _ in got] == [7, 10, 14], got
    assert [o for _, _, o in got] == [107 * 48 + 24, 110 * 48 + 24, 114 * 48 + 24]
    # a perfect score separates the classes at both arms; arm B is random
    rng = np.random.default_rng(0)
    n = 120
    label = rng.integers(0, 2, n)
    rows = []
    for arm, score in (("A", label + 0.1 * rng.normal(size=n)), ("B", rng.normal(size=n))):
        rows += [
            {
                "month": f"m{i % 2}",
                "pass_id": f"p{i % 6}",
                "event": i,
                "days_after_onset": 14,
                "mirror_state": "none",
                "arm": arm,
                "follow": float(score[i]),
                "label": int(label[i]),
            }
            for i in range(n)
        ]
    res = bootstrap_contrast(
        table=pl.DataFrame(rows), arm_a="A", arm_b="B", subset=None, k=14, draws=200
    )
    assert res["diff"] > 0.3 and res["low"] > 0.0, res
    # slot flags: two events whose hosts are neighbours flag each other; the same set far apart does not
    table = nb.load_table()
    h0 = 0
    near = next(q for q in nb.real_slots(table=table, host=h0) if q >= 0)
    far = next(
        h
        for h in range(nb.N_PRIMARIES)
        if h not in {h0, near, *nb.real_slots(table=table, host=h0)}
        and h not in set(nb.real_slots(table=table, host=near))
    )
    two = pl.DataFrame({"host": [h0, near], "truth": ["era", "era"]})
    flagged = slot_flags(truth=two, touched=[h0, near], table=table, world="bank")
    assert flagged["slot_other_event"].to_list() == [True, True], flagged
    apart = slot_flags(
        truth=pl.DataFrame({"host": [h0, far], "truth": ["era", "era"]}),
        touched=[h0, far],
        table=table,
        world="bank",
    )
    assert not any(apart["slot_other_event"]), "far apart hosts should not flag each other"
    try:
        slot_flags(truth=two, touched=[h0, near], table=table, world="g2")
    except AssertionError:
        pass
    else:
        raise AssertionError("rule 3 did not fire on the second generator")
    kept = with_keep(
        frame=pl.DataFrame(
            {
                "truth": ["switch", "era", "switch"],
                "slot_other_event": [True, True, False],
                "rewired_touched": [False, False, True],
            }
        ),
        world="bank",
        readings="s4_s7",
    )
    assert kept.height == 2
    kept = with_keep(
        frame=pl.DataFrame(
            {
                "truth": ["switch", "era", "switch"],
                "slot_other_event": [True, True, False],
                "rewired_touched": [False, False, True],
            }
        ),
        world="g2",
        readings="s1_s3",
    )
    assert kept.height == 2
    # visible change: a step of 0.1 from half-hour 480 is visible only up to the origin's newest visible half-hour
    clean = np.zeros((16, 2000))
    panel = clean.copy()
    panel[3, 480:] = 0.1
    c = 480 + 7 * 48 + 24
    assert (
        abs(visible_change(panel=panel, clean=clean, host=3, onset_day=10, c_idx=c) - 0.1) < 1e-12
    )
    assert np.isnan(
        visible_change(
            panel=panel,
            clean=clean,
            host=3,
            onset_day=10,
            c_idx=10 * 48 + pr.lay.NATIVE_OFFSET_STEPS + 40,
        )
    )
    assert (
        abs(mean_gap(label=np.array([1, 1, 0, 0]), score=np.array([1.0, 0.8, 0.1, 0.3])) - 0.7)
        < 1e-12
    )
    ci = pass_bootstrap_ci(
        frame=pl.DataFrame(
            {
                "month": ["a"] * 4 + ["b"] * 4,
                "pass_id": ["p0", "p0", "p1", "p1"] * 2,
                "d": [-1.0] * 8,
            }
        ),
        column="d",
        draws=50,
    )
    assert ci["mean"] == -1.0 and ci["high"] == -1.0
    print("injected-world self-test passed (no data, no fit)")
    return 0


def main() -> int:
    """Run one command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ceiling", "kn11a", "predict", "follow", "selftest"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    if args.command == "ceiling":
        ceiling_table()
        return 0
    if args.command == "follow":
        cmd_follow(world=args.args[0])
        return 0
    if args.command == "kn11a":
        kn11a(month=args.args[0], device="cpu")
        return 0
    arm, world, month, seed = args.args[0], args.args[1], args.args[2], int(args.args[3])
    ceiling_table()
    injected_forecasts(
        arm=arm,
        world=world,
        month=month,
        seed=seed,
        data_dir=ML_EXPERIMENTS_DATA_DIR,
        origins="sparse",
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ruff: noqa: E501, PT018, PLW3301
"""Experiment 715: the injected worlds (design section 7.3): the ceiling table, KN11, the predictions, and S1 to S7.

Usage:
    e00715_era.py ceiling                         # print the ceiling table; assert it equals the design's to 3 decimals
    e00715_era.py kn11a MONTH                     # KN11(a), before any injected-world score is read
    e00715_era.py predict ARM WORLD MONTH SEED    # prediction only: ARM on the injected passes of WORLD (`g2` or `bank`)
    e00715_era.py follow WORLD                    # follow shares of every stored arm; S1 to S7 with their intervals

**Prediction only.** The onset lies after every fit's cutoff, so every fitted state was trained on the real
power. `predict` loads the stored stage 3 state of an arm, reads the injected panel as the host's history
and as the neighbour power (mode `predict_injected`: the real neighbours of an injected host carry their
own injected power too), and forecasts each event's host at the origins 7, 10, 14, and 21 days after the
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
from pathlib import Path
from typing import Final

os.environ.setdefault("E00667_PHASE", "predict")

import e00713_era as era713
import e00713_inputs as inp
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
    """Print the ceiling table from experiment 714's out-of-fold scores, and assert it equals the design's to 3 decimals."""
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
    """Return the injected panel of a second-generator pass on the adapter's grid, its truth table, and its touched Primaries."""
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


def window_of(*, truth: pl.DataFrame, n_grid: int) -> tuple[int, int]:
    """Return the grid half-hour range `[lo, hi)` that holds every event of a pass."""
    starts = truth["onset_day"].to_numpy() * 48
    ends = np.where(truth["end_day"].to_numpy() < 0, n_grid, (truth["end_day"].to_numpy() + 1) * 48)
    return int(starts.min()), int(ends.max())


def kn11a(*, month: str, device: str = "cpu") -> dict:
    """KN11(a): the injected panels of both worlds equal the cleaned power outside the touched Primaries and windows."""
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
    """KN11(b): F4's prediction path with a stored state of `M*` reproduces experiment 713's bank forecasts of `M*` (1e-4 of capacity)."""
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
        worst = max(worst, max(float((j[q] - j[f"{q}_s"]).abs().max()) for q in pr.QCOLS))
    return {"max_abs_difference": worst, "pass": worst <= 1e-4}


def event_origins(*, truth: pl.DataFrame, ens_days: tuple[int, int]) -> list[tuple[int, int, int]]:
    """Return (event index, days after the onset, contract origin index) inside the days `ens_days = (first, last)` of the ensemble table."""
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
    """Forecast each event's host of every pass at the sparse origins; write `pass<NN>.parquet`; return the folder."""
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
        panel, truth, _ = panel_of(world=world, month=month, index=index, clean=data.clean)
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
        samples = tr.F4Samples(primary=hosts, c_idx=c, is_val=np.zeros(len(c), bool))
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
    """Return the follow share of one event at one origin: the mean median difference over the target rows, over the mean injected change."""
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
    """Return the AUROC of the follow share of `arm_a` minus that of `arm_b` at `k` days, with a 95% interval that resamples whole passes within months, then months."""
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
    print("injected-world self-test passed (no data, no fit)")
    return 0


def main() -> int:
    """Run one command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ceiling", "kn11a", "predict", "selftest"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    if args.command == "ceiling":
        ceiling_table()
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

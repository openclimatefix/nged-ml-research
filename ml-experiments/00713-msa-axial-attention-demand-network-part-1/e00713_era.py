# ruff: noqa: E501
"""Experiment 713: the secondary era set (design 1.10): KA11, the bank predictions, and R1 to R3.

Usage:
    e00713_era.py ka11 MONTH                 # check KA11 before any bank score is read
    e00713_era.py bank ARM MONTH SEED        # prediction only: ARM on 709's 16 passes of MONTH
    e00713_era.py score                      # R2 and R3 contrasts from the stored bank forecasts
    e00713_era.py r1                         # R1: D8 alone on the real scored rows

**Prediction only.** The onset is the fold's cutoff day, so every fitted state was trained on the real
power. `bank` loads the stored stage 3 state of an arm, replaces the history power with
`InjectedPass.power` (`00709-*/worlds/bank_<month>_p<pass>.pkl`), and forecasts the host
Primaries of the
pass's events for every origin of the month. Nothing is refitted. The scored slices are those of
`e00709_bank_score.event_losses`: the event's CRPS is the mean over its rows, against
`truth_share` for a
permanent change (origins 0 to 7 and 7 to 28 days after the onset) and `clean_share` for a temporary
move (from the onset to 7 days after the end). The reference `C` is experiment 709's arm `none` on the
same rows. The bank forecasts are written to `bank713/<arm>_<month>_s<seed>/passNN.parquet`.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Final

os.environ.setdefault("E00667_PHASE", "predict")

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00709_bank_score as bank_score  # ty: ignore[unresolved-import]
import e00713_arms as arms
import e00713_bank as bank
import e00713_inputs as inp
import e00713_layout as lay
import e00713_model as mdl
import e00713_score as score
import e00713_train as tr
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.eval_protocol.metrics import quantile_score
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
D709: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00709-*"))
BANK713: Final[Path] = ML_EXPERIMENTS_DATA_DIR / arms.NAME / "bank713"
PASSES: Final = range(16)
QCOLS: Final = score.QCOLS
LOSS_TOLERANCE: Final = 5e-4
"""Three decimals."""
THRESHOLDS: Final = {
    "reads_era": -1.0, "stem_adds": -0.30, "harm": 0.10, "harm_upper": 0.20, "stem_worsens": 0.10,
    "gain_d8": -0.25, "loss_d8": 0.25,
}  # fmt: skip


def load_pass(*, month: str, index: int):  # noqa: ANN201
    """Return experiment 709's injected pass (`InjectedPass`) for a fold month."""
    path = D709 / "worlds" / f"bank_{month}_p{index}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)  # our own stored bank


def ka11(*, month: str, device: str) -> None:
    """KA11: the injected power equals the cleaned power outside the injected windows; `none` is reproduced."""
    pipe = inp.load_pipeline(month=month, phase="predict")
    data = inp.build_fold_data(pipe=pipe, device=device)
    clean = data.clean
    for index in PASSES:
        ipass = load_pass(month=month, index=index)
        power = np.asarray(ipass.power, dtype=np.float64)
        assert power.shape == clean.shape, (
            f"the injected panel {power.shape} and the cleaned power {clean.shape} differ in shape"
        )
        deviates = np.isfinite(power) & np.isfinite(clean) & (np.abs(power - clean) > 1e-6)
        truth = ipass.truth
        window = np.zeros(clean.shape[1], dtype=bool)
        for event in truth.iter_rows(named=True):
            end = event["end_day"] if event["end_day"] >= 0 else clean.shape[1] // 48
            window[event["onset_day"] * 48 : (end + 1) * 48] = True
        outside = deviates & ~window[None, :]
        assert not outside.any(), (
            f"pass {index}: {int(outside.sum())} half-hours deviate outside the windows"
        )
        hosts = set(truth["host"].to_list())
        movers = set(np.nonzero(deviates.any(axis=1))[0].tolist())
        print(f"pass {index}: hosts {sorted(hosts)}, Primaries that deviate {sorted(movers)}")
    ids = k578.sites()["time_series_id"].to_numpy()
    for seed in arms.SCREEN_SEEDS:
        mine = bank_score.event_losses(arm="none", month=month, seed=seed, ids=ids)
        again = event_losses_from_files(
            folder=D709 / "bank" / f"none_{month}_s{seed}", ids=ids, arm_column="q"
        )
        merged = mine.join(again, on=["pass", "event", "slice"], suffix="_copy")
        worst = float((merged["crps"] - merged["crps_copy"]).abs().max())
        assert worst <= LOSS_TOLERANCE, worst
        print(
            f"KA11 {month} s{seed}: experiment 709's `none` event losses reproduced, max diff {worst:.1e}"
        )


def event_losses_from_files(*, folder: Path, ids: np.ndarray, arm_column: str) -> pl.DataFrame:
    """Recompute event losses from a bank folder (a copy of the slices of `e00709_bank_score.event_losses`)."""
    rows = []
    for forecast_path in sorted(folder.glob("pass[0-9][0-9].parquet")):
        events = pl.read_parquet(forecast_path.with_name(forecast_path.stem + "_truth.parquet"))
        forecast = pl.read_parquet(forecast_path)
        rows.extend(
            event_rows(
                forecast=forecast, events=events, ids=ids, name=forecast_path.stem, fold="", seed=0
            )
        )
    return pl.DataFrame(rows).select("pass", "event", "slice", "crps")


def event_rows(
    *,
    forecast: pl.DataFrame,
    events: pl.DataFrame,
    ids: np.ndarray,
    name: str,
    fold: str,
    seed: int,
) -> list[dict]:
    """Return one row per (event, slice) with the mean CRPS of the host's rows in the slice."""
    import e00709_features as features  # ty: ignore[unresolved-import]

    origin_day = (
        (
            forecast["power_fcst_init_time"].dt.replace_time_zone(None)
            - pl.Series([features.GRID_START.replace(tzinfo=None)] * forecast.height)
        )
        .dt.total_days()
        .to_numpy()
    )
    site = forecast["time_series_id"].to_numpy()
    q = forecast.select(list(QCOLS)).to_numpy().astype(np.float64)
    out = []
    for index, event in enumerate(events.iter_rows(named=True)):
        host_rows = site == ids[event["host"]]
        onset = event["onset_day"]
        if event["truth"] == "era":
            slices = {"era 0-7 d": (onset, onset + 7), "era 7-28 d": (onset + 7, onset + 28)}
            column = "truth_share"
        else:
            slices = {"switch to end+7 d": (onset, event["end_day"] + 7)}
            column = "clean_share"
        truth = forecast[column].to_numpy().astype(np.float64)
        for label, (lo, hi) in slices.items():
            sel = host_rows & (origin_day >= lo) & (origin_day < hi) & np.isfinite(truth)
            if sel.any():
                crps = float(
                    np.mean(quantile_score(actual=truth[sel] * 100.0, quantiles=q[sel] * 100.0))
                )
                out.append(
                    {
                        "fold": fold,
                        "pass": name,
                        "event": index,
                        "slice": label,
                        "seed": seed,
                        "crps": crps,
                        "n_rows": int(sel.sum()),
                    }
                )
    return out


def cmd_bank(*, arm: str, month: str, seed: int, device: str) -> None:
    """Forecast the host Primaries of every pass with the injected history power."""
    data_dir = ML_EXPERIMENTS_DATA_DIR
    folder = BANK713 / f"{arm}_{month}_s{seed}"
    spec = arms.resolve(arm=arm, data_dir=data_dir)
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    model = bank.load_model(
        spec=spec, arm=arm, month=month, seed=seed, device=device, data_dir=data_dir
    )
    pool = tr.block_pool(clean=data.clean)
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    for index in PASSES:
        out = folder / f"pass{index:02d}.parquet"
        if out.exists():
            continue
        ipass = load_pass(month=month, index=index)
        reference = pl.read_parquet(D709 / "bank" / f"none_{month}_s0" / f"pass{index:02d}.parquet")
        events = ipass.truth
        hosts = sorted(set(events["host"].to_list()))
        z = (np.asarray(ipass.power) - data.target_mean[:, None]) / data.target_std[:, None]
        src = data.source.with_power_in(power_in=torch.from_numpy(z.astype(np.float32)).to(device))
        origins = np.sort(reference["power_fcst_init_time"].unique().to_numpy())
        import e00422_prep as prep422  # ty: ignore[unresolved-import]

        native = (
            (origins.astype("datetime64[us]").astype(np.int64) - prep422.GRID_START_US)
            // prep422.STEP_US
        ).astype(np.int64)
        c_all = native + lay.NATIVE_OFFSET_STEPS
        prim, cc = np.meshgrid(np.array(hosts), c_all, indexing="ij")
        frames = []
        for lead in range(14):
            q = []
            for start in range(0, prim.size, 512):
                sel = np.arange(start, min(start + 512, prim.size))
                samples = tr.SampleSet(prim.ravel(), cc.ravel(), np.zeros(prim.size, bool))
                if spec.placebo:
                    samples.ages = tr.placebo_ages_for(
                        primary=prim.ravel(), c_idx=cc.ravel(), pool=pool
                    )
                b = tr.batch_for(
                    source=src,
                    samples=samples,
                    index=sel,
                    lead=np.full(len(sel), lead),
                    ensemble=True,
                    cut=10**9,
                    dropout=None,
                )
                with torch.no_grad():
                    qq = mdl.standardised_quantiles(out=model(b.inputs).out)
                p = prim.ravel()[sel]
                qq = (
                    qq.cpu().numpy() * data.target_std[p][:, None, None]
                    + data.target_mean[p][:, None, None]
                )
                q.append((np.sort(qq, axis=-1), b.target_idx.cpu().numpy(), p, cc.ravel()[sel]))
            for qq, tidx, p, c in q:
                frames.append(
                    pl.DataFrame(
                        {
                            "time_series_id": np.repeat(ids[p], 48),
                            "origin_us": np.repeat(
                                prep422.GRID_START_US
                                + (c - lay.NATIVE_OFFSET_STEPS) * prep422.STEP_US,
                                48,
                            ),
                            "valid_us": prep422.GRID_START_US + tidx.reshape(-1) * prep422.STEP_US,
                            **{
                                n: qq[:, :, i].reshape(-1).astype(np.float32)
                                for i, n in enumerate(QCOLS)
                            },
                        }
                    )
                )
        mine = (
            pl.concat(frames)
            .with_columns(
                power_fcst_init_time=pl.col("origin_us")
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("UTC"),
                valid_time=pl.col("valid_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
                time_series_id=pl.col("time_series_id").cast(pl.Int64),
            )
            .drop("origin_us", "valid_us")
        )
        truth_cols = reference.select(
            "time_series_id", "power_fcst_init_time", "valid_time", "truth_share", "clean_share"
        )
        joined = mine.join(
            truth_cols, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="inner"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        joined.write_parquet(tmp)
        tmp.rename(out)
        events.write_parquet(out.with_name(f"pass{index:02d}_truth.parquet"))
        print(arm, month, seed, "pass", index, joined.height, "rows", flush=True)


def pass_bootstrap(
    *, diffs: np.ndarray, n: int = 10_000, level: float = 95.0
) -> tuple[float, float]:
    """Return a percentile bootstrap interval of the mean, resampling whole passes."""
    rng = np.random.default_rng(0)
    means = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n)])
    tail = (100.0 - level) / 2.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def cmd_score() -> None:
    """R2 and R3: event losses of each arm and of `none` on the same rows, and the contrasts."""
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    ids = k578.sites()["time_series_id"].to_numpy()
    losses = []
    for arm in (star, "E", "P", "K"):
        for month in arms.BANK_FOLDS:
            for seed in arms.SCREEN_SEEDS:
                folder = BANK713 / f"{arm}_{month}_s{seed}"
                for path in sorted(folder.glob("pass[0-9][0-9].parquet")):
                    forecast = pl.read_parquet(path)
                    events = pl.read_parquet(path.with_name(path.stem + "_truth.parquet"))
                    for r in event_rows(
                        forecast=forecast,
                        events=events,
                        ids=ids,
                        name=path.stem,
                        fold=month,
                        seed=seed,
                    ):
                        losses.append({"arm": arm, **r})  # noqa: PERF401
                    ref = pl.read_parquet(D709 / "bank" / f"none_{month}_s{seed}" / path.name).join(
                        forecast.select("time_series_id", "power_fcst_init_time", "valid_time"),
                        on=["time_series_id", "power_fcst_init_time", "valid_time"], how="inner",
                    )  # fmt: skip
                    for r in event_rows(
                        forecast=ref, events=events, ids=ids, name=path.stem, fold=month, seed=seed
                    ):
                        losses.append({"arm": "C", **r})  # noqa: PERF401
    frame = pl.DataFrame(losses)
    per_event = frame.group_by("arm", "fold", "pass", "event", "slice").agg(pl.col("crps").mean())
    out = {}
    for t, r in ((star, "C"), (star, "P"), ("E", star), ("E", "P"), ("E", "C"), ("K", "C")):
        for label in ("era 0-7 d", "era 7-28 d", "switch to end+7 d"):
            a = per_event.filter((pl.col("arm") == t) & (pl.col("slice") == label))
            b = per_event.filter((pl.col("arm") == r) & (pl.col("slice") == label))
            m = a.join(b, on=["fold", "pass", "event"], suffix="_ref")
            if m.is_empty():
                continue
            d = (m["crps"] - m["crps_ref"]).to_numpy()
            pass_id = (m["fold"] + m["pass"]).to_numpy()
            by_pass = np.array([d[pass_id == u].mean() for u in np.unique(pass_id)])
            lo, hi = pass_bootstrap(diffs=by_pass)
            out[f"{t}-{r}:{label}"] = {
                "mean": float(by_pass.mean()),
                "lower": lo,
                "upper": hi,
                "n_passes": len(by_pass),
            }
    out["thresholds"] = THRESHOLDS
    score.write(name="era_r2_r3", obj=out)


def cmd_r1() -> None:
    """R1: D8 alone, real rows, headline band, 8 folds, seeds 0 and 1: `E`-`M*`, `E`-`C`, `M*`-`C`."""
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    sites = k578.sites()
    groups = {"d8": sites.filter(pl.col("label") == "D8")["time_series_id"].to_numpy()}
    truth = score.truth_stack()
    names = [star, "E", "C"]
    per_seed, averaged = {}, {}
    for n in names:
        per_seed[n], averaged[n] = score.load_scored(
            arm=n, seeds=arms.SCREEN_SEEDS, months=arms.ALL_FOLDS, truth=truth
        )
    out = {}
    for t, r in (("E", star), ("E", "C"), (star, "C")):
        c = score.contrast(
            per_seed=per_seed,
            averaged=averaged,
            treatment=t,
            reference=r,
            group="d8",
            groups=groups,
            level=95.0,
        )
        out[f"{t}-{r}"] = c
    score.write(name="era_r1", obj=out)


def main() -> int:
    """Dispatch the sub-command."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    k = sub.add_parser("ka11")
    k.add_argument("month")
    b = sub.add_parser("bank")
    b.add_argument("arm")
    b.add_argument("month")
    b.add_argument("seed", type=int)
    sub.add_parser("score")
    sub.add_parser("r1")
    for p in (k, b):
        p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "ka11":
        ka11(month=args.month, device=args.device)
    elif args.command == "bank":
        cmd_bank(arm=args.arm, month=args.month, seed=args.seed, device=args.device)
    elif args.command == "score":
        cmd_score()
    else:
        cmd_r1()
    return 0


if __name__ == "__main__":
    sys.exit(main())

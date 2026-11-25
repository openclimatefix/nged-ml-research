# ruff: noqa: ANN001, D103, D205, D209, E402, E501, I001, PT018
"""Experiment 726: run injected panels through the stored stage 3 state of an MSA arm (GPU).

**Source.** The loop is a copy of `cmd_bank` in experiment 713's `e00713_era.py` (frozen checkout
`7c29114a8`), which loads the stored state, swaps the injected panel into the source as the
history power (`with_power_in`), and forecasts in batches of 512 samples. `load_model`,
`batch_for`, and `standardised_quantiles` are imported from the frozen checkout. Changes:

1. **Panel.** The panel is experiment 581's (`e00581_bank.contaminate`), with every Primary
   shifted (`all`), none (`zero`), or only the members of one host-only set (`host`). It is
   standardised with the fold's own mean and standard deviation, as `cmd_shifts` does.
2. **Rows.** All 16 Primaries (the members only, for `host`), the origins of the Candidate's stored
   forecast with origin day 0 to 20, and the lead indices that reach into the 24 to 240 hour band
   (`lead_indices`, derived from the layout).
3. **Paths.** Output goes to `msa/<world>/<arm>_<month>_s<seed>/pass<NN>.parquet` under this
   experiment's folder, through `write_parquet_atomic`. The frozen checkout is read only.
4. **Gates.** Before any state loads, the command asserts experiment 713's code hash, the HEAD of the
   checkout, and a clean checkout, and prints experiment 713's weather look-ahead statement (check KW).
5. **Read-outs.** The command prints the share of shifted history half-hours beyond the clip, and
   with `--check-km3` it checks that the shifted panel reaches the history tokens.

Usage: python e00726_msa_panel.py WORLD ARM MONTH SEED [--passes R ...] [--sets K ...] [--mask-history]
[--check-km3] [--device cuda]

`ARM` is `M1`, `K`, `E`, or `N`. `--mask-history` forecasts with every history token masked and
writes under the arm name `<ARM>-nohist` (check KM2). Run it through `run.sh` with
`EXTRA_PYTHONPATH` set to the frozen checkout's experiment 713 folder.
"""

import argparse
import os
import sys
import time

import numpy as np
import polars as pl

import e00726_common as com

ARGV = sys.argv[1:]
_p = argparse.ArgumentParser()
_p.add_argument("world", choices=com.MSA_WORLDS)
_p.add_argument("arm", choices=("M1", "K", "E", "N"))
_p.add_argument("month", choices=com.FOLD_MONTHS)
_p.add_argument("seed", type=int)
_p.add_argument("--passes", nargs="+", type=int, default=None)
_p.add_argument("--sets", nargs="+", type=int, default=None)
_p.add_argument("--mask-history", action="store_true")
_p.add_argument("--check-km3", action="store_true")
_p.add_argument("--device", default="cuda")
MINE = _p.parse_args()
com.require(os.environ.get("E00667_PHASE") == "predict", "set E00667_PHASE=predict (run.sh does)")
if MINE.world == "host":
    assert MINE.sets is not None and len(MINE.sets) == 1, "world host needs one --sets value"
passes = (
    MINE.passes
    if MINE.passes is not None
    else ([0] if MINE.world == "zero" else list(range(com.N_PASSES)))
)
assert MINE.world != "host" or MINE.passes is not None, "world host needs --passes"
RECORD_713 = com.assert_frozen_713()  # before any module of the checkout loads a state
e = com.prime_pipeline(month=MINE.month)  # the stub frames of the pipeline are redirected

import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00713_arms as arms  # ty: ignore[unresolved-import]
import e00713_bank as bank713  # ty: ignore[unresolved-import]
import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_predict as pred  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00713_weather_lookahead as wl  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from datetime import UTC, datetime
from studies.sources import ML_EXPERIMENTS_DATA_DIR

torch.set_num_threads(1)
BATCH = 512
ARM_NAME = f"{MINE.arm}-nohist" if MINE.mask_history else MINE.arm


def origins_of_candidate(*, month: str) -> np.ndarray:
    """Return the native origin indices (grid steps) of origin day 0 to 20 in the Candidate's stored forecast.

    The stored forecast of seed 0 on the unshifted panel is the reference of the same-origin rule.
    """
    stored = com.read_forecast(
        path=com.root_of_seed(seed=0).stored_fit(month=month, seed=0), columns=("q50",)
    )
    origin = stored["power_fcst_init_time"].unique().dt.epoch("us").to_numpy()
    day = com.origin_day_of(origin_us=origin, month=month)
    keep = np.sort(origin[(day >= 0) & (day <= com.MAX_ORIGIN_DAY)])
    assert len(keep) > 0, "the Candidate's stored forecast has no origin on days 0 to 20"
    return ((keep - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)


def weather_statement(*, month: str) -> None:
    """Check KW: print experiment 713's look-ahead statement and its data check for the first origin
    of the month."""
    y, m = (int(v) for v in month.split("-"))
    contract = datetime(y, m, 10, 12, tzinfo=UTC)
    print("\n".join(wl.statement(contract=contract, primary="D5")), flush=True)
    ok, lines = wl.check_data(contract=contract)
    print("\n".join(lines), flush=True)
    assert ok, "KW: the stored run frame is not issued 9 hours before the native origin"


def forecast_pass(
    *, model, src, data, primaries: np.ndarray, native: np.ndarray, leads: list[int], ids: np.ndarray
) -> pl.DataFrame:  # fmt: skip
    """Return the keys and 11 quantiles (capacity shares) of `primaries` at the native origins and lead indices."""
    c_all = native + lay.NATIVE_OFFSET_STEPS
    prim, cc = np.meshgrid(primaries, c_all, indexing="ij")
    prim, cc = prim.ravel(), cc.ravel()
    samples = tr.SampleSet(primary=prim, c_idx=cc, is_val=np.zeros(prim.size, bool))
    frames = []
    for lead in leads:
        for start in range(0, prim.size, BATCH):
            sel = np.arange(start, min(start + BATCH, prim.size))
            ones = np.ones(len(sel), bool)
            drop = (ones, np.zeros(len(sel), bool)) if MINE.mask_history else None
            b = tr.batch_for(
                source=src, samples=samples, index=sel, lead=np.full(len(sel), lead),
                ensemble=True, cut=10**9, dropout=drop,
            )  # fmt: skip
            assert src.ens is not None
            assert bool(
                src.ens.filled[
                    b.inputs.primary[:, None],
                    (b.c_idx // 48 - src.ens.first_day)[:, None],
                    (48 * b.inputs.lead_day[:, None] + src.steps),
                ].all()
            ), "a scored row has no ensemble inputs"  # ty: ignore[unresolved-attribute]
            with torch.no_grad():
                q = mdl.standardised_quantiles(out=model(b.inputs).out)
            p = prim[sel]
            q = (
                q.cpu().numpy() * data.target_std[p][:, None, None]
                + data.target_mean[p][:, None, None]
            )
            q = np.sort(q, axis=-1)
            tidx = b.target_idx.cpu().numpy()
            origin_us = (
                prep422.GRID_START_US + (cc[sel] - lay.NATIVE_OFFSET_STEPS) * prep422.STEP_US
            )
            frames.append(
                pl.DataFrame(
                    {
                        "time_series_id": np.repeat(ids[p], 48),
                        "origin_us": np.repeat(origin_us, 48),
                        "valid_us": prep422.GRID_START_US + tidx.reshape(-1) * prep422.STEP_US,
                        **{
                            name: q[:, :, i].reshape(-1).astype(np.float32)
                            for i, name in enumerate(pred.QCOLS)
                        },
                    }
                )
            )
    utc = pl.Datetime("us", "UTC")
    frame = pl.concat(frames).with_columns(
        power_fcst_init_time=pl.col("origin_us")
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .cast(utc),
        valid_time=pl.col("valid_us").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").cast(utc),
        time_series_id=pl.col("time_series_id").cast(pl.Int64),
    )
    out = frame.select(*com.KEYS, *pred.QCOLS).sort(*com.KEYS)
    assert (
        out.select(pl.all_horizontal(pl.col(c).is_finite() for c in pred.QCOLS)).to_series().all()
    ), "a forecast quantile is not finite"
    assert not out.select(list(com.KEYS)).is_duplicated().any(), "a forecast key is repeated"
    return out


def check_km3(*, data, device: str) -> None:
    """Check KM3 (input half): the history power of an origin inside a window equals the clipped
    standardised shifted panel.

    The panel is pass 0's. The follow-share half of KM3 runs in `e00726_checks.py` on the stored files.
    """
    r = 0
    month_idx = data.cutoff_idx
    shifted = com.all_shifted_panel(clean=data.clean, month_start_idx=month_idx, r=r)
    z = (shifted - data.target_mean[:, None]) / data.target_std[:, None]
    z32 = z.astype(np.float32)
    src = data.source.with_power_in(power_in=torch.from_numpy(z32).to(device))
    depth, length, _sign = com.event_arrays()
    wanted = [p for p in range(16) if abs(depth[r, p]) == 0.5 and length[r, p] >= 15]
    assert wanted, "pass 0 has no 0.50 shift of 15 days or more"
    c = month_idx + 10 * 48 + 18 + lay.NATIVE_OFFSET_STEPS
    prim = np.array(wanted, dtype=np.int64)
    cc = np.full(len(wanted), c, dtype=np.int64)
    samples = tr.SampleSet(primary=prim, c_idx=cc, is_val=np.zeros(len(wanted), bool))
    b = tr.batch_for(
        source=src, samples=samples, index=np.arange(len(wanted)), lead=np.zeros(len(wanted), int),
        ensemble=True, cut=10**9, dropout=None,
    )  # fmt: skip
    hist_idx = b.hist_idx.cpu().numpy()
    visible = b.inputs.hist_visible.cpu().numpy()
    expect = np.clip(z32[prim[:, None, None], hist_idx], -mdl.POWER_CLIP, mdl.POWER_CLIP)
    expect = np.where(visible & np.isfinite(expect), expect, 0.0)
    worst = float(np.abs(expect - b.inputs.hist_power.cpu().numpy()).max())
    shifted_tokens = visible & (
        np.abs(shifted[prim[:, None, None], hist_idx] - data.clean[prim[:, None, None], hist_idx])
        > 0
    )
    com.blind_report(
        name="km3_input",
        passed=worst <= 1e-6 and bool(shifted_tokens.any()),
        values={"max_abs_difference": worst, "n_shifted_visible_tokens": int(shifted_tokens.sum())},
    )


def run() -> None:
    month, seed, device = MINE.month, MINE.seed, MINE.device
    com.cap_gpu(device=device)
    outs = {
        r: com.msa_output(
            world=MINE.world if MINE.world != "host" else com.msa_host_world(k=MINE.sets[0]),
            arm=ARM_NAME, month=month, seed=seed, r=r,
        )
        for r in passes
    }  # fmt: skip
    todo = [r for r in passes if not com.job_done(out=outs[r])]
    print(f"{MINE.world} {ARM_NAME} {month} s{seed}: {len(todo)} passes to do", flush=True)
    if not todo and not MINE.check_km3:
        return
    data_dir = ML_EXPERIMENTS_DATA_DIR
    spec = arms.resolve(arm=MINE.arm, data_dir=data_dir)
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    saved = arms.root(data_dir=data_dir) / "work" / "standardiser" / f"{month}.sha256"
    assert saved.exists() and saved.read_text() == data.fingerprint, (
        "standardiser differs from the training phase"
    )
    assert not spec.placebo
    import e00650_common as c650  # ty: ignore[unresolved-import]

    com.require(
        data.cutoff_idx == c650.month_start_idx(month=month),
        "the fold's cutoff index differs from experiment 650's month start index",
    )
    model = bank713.load_model(
        spec=spec, arm=MINE.arm, month=month, seed=seed, device=device, data_dir=data_dir
    )
    weather_statement(month=month)
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    labels = k578.sites().sort("primary")["label"].to_list()
    native = origins_of_candidate(month=month)
    leads = com.lead_indices(
        lo_hours=com.LEAD_BANDS["24-240h"][0], hi_hours=com.LEAD_BANDS["24-240h"][1]
    )
    print(f"origins {len(native)}, lead indices {leads}", flush=True)
    sets = com.read_sets() if MINE.world == "host" else []
    mean, std = data.target_mean, data.target_std
    if MINE.check_km3:
        check_km3(data=data, device=device)
    for r in todo:
        started = time.monotonic()
        if MINE.world == "zero":
            panel, primaries = data.clean, np.arange(16)
        elif MINE.world == "all":
            panel = com.all_shifted_panel(clean=data.clean, month_start_idx=data.cutoff_idx, r=r)
            primaries = np.arange(16)
        else:
            members = sets[MINE.sets[0]]
            panel = com.host_panel(
                clean=data.clean, month_start_idx=data.cutoff_idx, r=r, members=members
            )
            primaries = np.array(sorted(members))
        z = (panel - mean[:, None]) / std[:, None]
        src = data.source.with_power_in(power_in=torch.from_numpy(z.astype(np.float32)).to(device))
        frame = forecast_pass(
            model=model,
            src=src,
            data=data,
            primaries=primaries,
            native=native,
            leads=leads,
            ids=ids,
        )
        com.write_parquet_atomic(frame=frame, path=outs[r])
        print(
            f"{MINE.world} {ARM_NAME} {month} s{seed} r{r:02d}: {frame.height} rows, {time.monotonic() - started:.1f} s",
            flush=True,
        )
    if MINE.world == "all" and MINE.arm == "M1" and not MINE.mask_history:
        diagnostic = com.clip_diagnostic(
            clean=data.clean,
            target_mean=mean,
            target_std=std,
            month_start_idx=data.cutoff_idx,
            labels=labels,
        )
        for row in diagnostic:
            print(f"clip {month} {row}", flush=True)
        com.write_json_atomic(
            obj=diagnostic, path=com.data_root() / "diagnostics" / f"clip_{month}.json"
        )
    com.write_json_atomic(
        obj=com.job_record(
            extra={
                "frozen_713": RECORD_713,
                "world": MINE.world,
                "arm": ARM_NAME,
                "month": month,
                "seed": seed,
                "passes": todo,
                "origins": len(native),
                "leads": leads,
                "argv": ARGV,
            },
        ),
        path=com.data_root()
        / "logs"
        / f"msa_{MINE.world}_{ARM_NAME}_{month}_s{seed}_{os.getpid()}.json",
    )


if __name__ == "__main__":
    run()

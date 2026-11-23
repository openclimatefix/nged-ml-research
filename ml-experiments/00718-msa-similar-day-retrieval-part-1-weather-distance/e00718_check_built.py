# ruff: noqa: E501
"""Experiment 718, B1: the checks that need a built fold's tables (design sections 10.1 and 11), on the CPU.

Usage:
    run.sh e00718_check_built.py MONTH [--samples 1000]         # prediction phase: KR8, KR3, KR7, KR1 (a)
    run.sh e00718_check_built.py MONTH --kr2c [--days 12]       # training phase: KR2 (c)

The default mode loads the fold in the prediction phase and every table that `e00718_build_tables.py`
wrote for it, with the provenance asserted, and checks them against the loaded arrays:

- KR8: `check_table` on every table, independent of the builder's eligibility function;
- KR3: `--samples` random samples per arm (`R`, `U`, `U27`, `UX`) over all origin days equal the
  independent recomputation, for the `ens` table, and a tenth of that for the `era5` table;
- KR7: the age-bucket tensor of 256 assembled samples per arm equals the layout's bucket function of
  the assembled ages, and the ages equal the table's;
- KR1 (a): the `Z` tables with the stored stage 3 state of `M*` reproduce experiment 713's stored
  forecast of its first scored origins (seed 0).

`--kr2c` loads the fold in the training phase and rebuilds the tables of the arms present for random
training days, which must equal the stored tables (built in the prediction phase) bit for bit. It
writes nothing under the data folder.
"""

import argparse
import sys
import time
from dataclasses import replace

import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00718_arms as arms
import e00718_batch as fb
import e00718_build_tables as bt
import e00718_check_kr3 as ck
import e00718_check_real as cr
import e00718_fit as fit_mod
import e00718_layout as lay
import e00718_tables as tab
import e00718_weather_lookahead as lookahead
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

KR3_ARMS = ("R", "U", "U27", "UX")


def load_tables(
    *, month: str, phase: str, fingerprint: str, digest: str, limit: int | None = None
) -> dict[str, dict]:
    """Return the stored tables of every arm that was built for `month`, with their provenance asserted."""
    out = {}
    for arm in sorted(lay.ROWS_PER_ARM):
        paths = {
            v: arms.table_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, version=v)
            for v in (fb.TABLE_ERA5, fb.TABLE_ENS)
        }
        if not all(p.exists() for p in paths.values()):
            continue
        expect = fit_mod.table_meta(
            arm=arm,
            month=month,
            fingerprint=fingerprint,
            phase=phase,
            digest=digest,
            limit_days=limit,
        )
        out[arm] = {v: fit_mod.load_table(path=p, expect=expect) for v, p in paths.items()}
    return out


KR1A_TOLERANCE_GPU: float = 1e-6
"""The pre-registered tolerance of KR1 (a); the stored forecasts were written on the GPU, so only a GPU run meets it."""
KR1A_TOLERANCE_CPU_SMOKE: float = 1e-5
"""The tolerance of the CPU smoke reading of KR1 (a): the CPU path differs from the GPU's by about 6e-6."""


def kr1a(
    *,
    data: inp.FoldData,
    tables: dict[str, dict],
    month: str,
    source,  # noqa: ANN001
    device: str = "cpu",
) -> str:
    """KR1 (a): arm `Z` with `M*`'s stage 3 state reproduces experiment 713's stored forecast of the first scored origins.

    On the CPU this is a smoke reading at 1e-5. On `cuda` it is the required check, at the pre-registered
    1e-6.
    """
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_arms as a713  # ty: ignore[unresolved-import]
    import e00713_score as s713  # ty: ignore[unresolved-import]
    import e00718_model as f9
    import e00718_predict as pred
    import polars as pl

    star = a713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    state_path = a713.model_file(
        data_dir=ML_EXPERIMENTS_DATA_DIR, arm=star, month=month, seed=0, stage=3
    )
    net = f9.F9Net(seed=0)
    net.load_mstar_state(torch.load(state_path, map_location="cpu"))
    net.to(device)
    assert data.test_origin_idx is not None
    in_table = np.isin(
        (data.test_origin_idx + lay713.NATIVE_OFFSET_STEPS) // lay713.STEPS_PER_DAY,
        tables["Z"][fb.TABLE_ENS].days,
    )
    few = replace(data, test_origin_idx=data.test_origin_idx[in_table][:2])
    f9source = fb.F9Source(base=source, tables=tables["Z"])
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    frame = pred.forecast_frame(
        model=net,
        data=few,
        source=f9source,
        seed=0,
        mask_history=False,
        grid_start_us=prep422.GRID_START_US,
        time_series_ids=ids,
    )
    stored = pl.read_parquet(
        a713.fit_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm=star, month=month, seed=0)
    )
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    joined = frame.join(stored, on=keys, suffix="_stored")
    assert joined.height == frame.height > 0, (joined.height, frame.height)
    worst = max(float((joined[c] - joined[f"{c}_stored"]).abs().max()) for c in s713.QCOLS)  # ty: ignore[unresolved-attribute]
    tolerance = KR1A_TOLERANCE_GPU if device == "cuda" else KR1A_TOLERANCE_CPU_SMOKE
    label = (
        "KR1 (a)"
        if device == "cuda"
        else "KR1 (a) CPU smoke (the required check runs on the GPU at 1e-6)"
    )
    assert worst <= tolerance, (
        f"{label}: the largest quantile difference is {worst}, above {tolerance}"
    )
    return f"{label}: arm Z with the stored {star} stage 3 state reproduces {joined.height:,} stored forecast rows; the largest difference of a quantile is {worst:.2e} (tolerance {tolerance:.0e})"


def kr7(*, source, tables: dict[str, dict], arm: str, n: int = 256) -> str:  # noqa: ANN001
    """KR7: the assembled age buckets are the layout's function of the assembled ages, and the ages are the table's."""
    f9 = fb.F9Source(base=source, tables=tables[arm])
    rng = np.random.default_rng(7)
    table = tables[arm][fb.TABLE_ENS]
    primary = rng.choice(table.primaries, size=n)
    # The ensemble table covers fewer days than the row tables of a full fold, and a batch assembles only
    # origin days inside it.
    ens = source.ens
    c_all = lay.contract_idx_of_day(day=table.days)
    inside = table.days[
        (c_all // lay.STEPS_PER_DAY - ens.first_day >= 0)
        & (c_all // lay.STEPS_PER_DAY - ens.first_day < ens.filled.shape[1])
    ]
    assert inside.size > 0, "no origin day of the table is inside the ensemble table"
    day = rng.choice(inside, size=n)
    lead = rng.integers(0, lay.N_LEAD_DAYS, size=n)
    batch = f9.assemble(
        primary=torch.from_numpy(primary), c_idx=torch.from_numpy(lay.contract_idx_of_day(day=day)),
        lead_day=torch.from_numpy(lead), ensemble=True,
    )  # fmt: skip
    ages = batch.ages.cpu().numpy()
    # A fixed row (arms Z and U27) is in the table but can hold no visible power, so `row_exists` is not
    # the table's presence, and the absent filler age is a valid age. The table's own ages are the reference.
    c_idx = lay.contract_idx_of_day(day=day)
    table_ages, _ = f9.rows_of(primary=primary, c_idx=c_idx, lead_day=lead, ensemble=True)
    exists = table_ages >= 0
    assert np.array_equal(np.where(exists, ages, -1), table_ages), (
        "an assembled age differs from the table"
    )
    assert batch.inputs.age_bucket is not None
    want = lay.age_bucket(ages=table_ages)
    got = batch.inputs.age_bucket.cpu().numpy()
    bad = np.argwhere(got != want)
    assert not bad.size, (
        f"KR7 {arm}: {len(bad)} age buckets differ from the layout's function; first at {bad[:3].tolist()}: "
        f"got {got[tuple(bad[0])]}, want {want[tuple(bad[0])]}, age {ages[tuple(bad[0])]}, exists {exists[tuple(bad[0])]}"
    )
    stored = table.lookup(primary=primary, day=day, lead_day=lead)
    assert np.array_equal(np.where(exists, ages, -1)[stored >= 0], stored[stored >= 0]), (
        "an assembled age differs from the table"
    )
    return f"KR7 {arm}: {n} assembled samples carry the layout's age buckets and the table's ages"


def predict_mode(*, month: str, samples: int, limit: int | None) -> list[str]:
    """Run KR8, KR3, KR7, and KR1 (a) on the built tables of `month`."""
    cr.cpu_only_cuda_calls()
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=0)
    )
    assert pipe.fold.PHASE == "predict"
    data = inp.build_fold_data(pipe=pipe, device="cpu")
    source = data.source
    assert source.ens is not None
    digest = fit_mod.inputs_digest(source=source, cutoff_idx=data.cutoff_idx)
    tables = load_tables(
        month=month, phase="predict", fingerprint=data.fingerprint, digest=digest, limit=limit
    )
    assert tables, f"no table of fold {month} was built"
    lines = [
        f"fold {month}: tables of {sorted(tables)} loaded with provenance asserted (full builds, predict phase)"
    ]
    finite = np.isfinite(source.power_z.cpu().numpy())
    for by_version in tables.values():
        for version, table in by_version.items():
            line = lookahead.check_table(table=table, finite=finite)
            lines.append(f"KR8 {version}: {line}")
            assert line.startswith("PASS"), line
    holidays = lay713.england_bank_holidays(years=range(2019, 2029))
    arrays = tab.pool_arrays_from_source(source=source, with_calendar=True, holidays=set(holidays))
    primaries = np.arange(source.power_z.shape[0])
    for arm in KR3_ARMS:
        if arm not in tables:
            continue
        for version, n, provider in (
            (fb.TABLE_ENS, samples, tab.ens_provider(ens=source.ens)),
            (fb.TABLE_ERA5, max(samples // 10, 50), tab.era5_provider(arrays=arrays)),
        ):
            started = time.process_time()
            table = tables[arm][version]
            out, _ = ck.check_arm(
                arm=arm, source=source, days=table.days, provider=provider, n_samples=n, seed=11,
                primaries=primaries, table=table, holidays=holidays,
            )  # fmt: skip
            lines.append(
                f"{out[0].replace('KR3', 'KR3 built ' + version)} [{time.process_time() - started:.0f} CPU s]"
            )
    lines += [kr7(source=source, tables=tables, arm=arm) for arm in tables]
    if "Z" in tables:
        lines.append(kr1a(data=data, tables=tables, month=month, source=source))
    else:
        lines.append("KR1 (a): not run, because the tables of arm Z were not built")
    return lines


def kr1a_gpu_mode(*, month: str) -> list[str]:
    """The required KR1 (a): the stored forecasts reproduced on the GPU to 1e-6, on the built tables."""
    assert torch.cuda.is_available(), "KR1 (a) at 1e-6 needs the GPU"
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=0)
    )
    assert pipe.fold.PHASE == "predict"
    data = inp.build_fold_data(pipe=pipe, device="cuda")
    source = data.source
    digest = fit_mod.inputs_digest(source=source, cutoff_idx=data.cutoff_idx)
    tables = load_tables(
        month=month, phase="predict", fingerprint=data.fingerprint, digest=digest, limit=None
    )
    return [kr1a(data=data, tables=tables, month=month, source=source, device="cuda")]


def kr2c_mode(*, month: str, n_days: int, limit: int | None) -> list[str]:
    """KR2 (c): tables rebuilt in the training phase for training days equal the stored tables."""
    cr.cpu_only_cuda_calls()
    pipe = inp.load_pipeline(month=month, phase="train", chain_root=inp.chain_root_of_seed(seed=0))
    assert pipe.fold.PHASE == "train"
    data = inp.build_fold_data(pipe=pipe, device="cpu")
    source = data.source
    assert source.ens is not None
    digest = fit_mod.inputs_digest(source=source, cutoff_idx=data.cutoff_idx)
    tables = load_tables(
        month=month, phase="predict", fingerprint=data.fingerprint, digest=digest, limit=limit
    )
    assert tables, f"no table of fold {month} was built"
    holidays = lay713.england_bank_holidays(years=range(2019, 2029))
    some = next(iter(tables.values()))[fb.TABLE_ENS]
    cutoff_day = data.cutoff_idx // lay713.STEPS_PER_DAY
    training = some.days[some.days < cutoff_day - lay713.N_LEAD_DAYS - 1]
    if len(training) == 0:
        return ["KR2 (c): the stored tables hold no training day (a limited smoke build): not run"]
    days = np.sort(
        np.random.default_rng(5).choice(training, size=min(n_days, len(training)), replace=False)
    )
    built = bt.build_tables(
        source=source, arms_to_build=sorted(tables), days=days, holidays=holidays
    )
    lines = []
    for arm, by_version in built.items():
        for version, table in by_version.items():
            stored = tables[arm][version]
            di = np.searchsorted(stored.days, days)
            assert np.array_equal(table.ages, stored.ages[:, di]), (
                f"arm {arm} {version}: training-phase rows differ from the stored table"
            )
            assert np.array_equal(table.measure, stored.measure[:, di])
            assert np.array_equal(table.no_weather, stored.no_weather[:, di])
        lines.append(
            f"KR2 (c) {arm}: {len(days)} training days (before the cutoff day {cutoff_day}) rebuilt in the training phase equal the stored tables, both versions"
        )
    return lines


def main() -> int:
    """Run the checks of one phase."""
    parser = argparse.ArgumentParser()
    parser.add_argument("month")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--kr2c", action="store_true")
    parser.add_argument(
        "--kr1a-gpu",
        action="store_true",
        help="the required KR1 (a) at 1e-6 on the GPU (a short job)",
    )
    parser.add_argument("--days", type=int, default=12)
    parser.add_argument(
        "--limit-days",
        type=int,
        default=None,
        help="for smoke tests: expect tables built with this --limit-days",
    )
    args = parser.parse_args()
    if args.kr1a_gpu:
        lines = kr1a_gpu_mode(month=args.month)
    elif args.kr2c:
        lines = kr2c_mode(month=args.month, n_days=args.days, limit=args.limit_days)
    else:
        lines = predict_mode(month=args.month, samples=args.samples, limit=args.limit_days)
    print("\n".join(lines))
    print("e00718_check_built: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

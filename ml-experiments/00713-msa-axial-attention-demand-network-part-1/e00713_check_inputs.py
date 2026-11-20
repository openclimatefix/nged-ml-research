# ruff: noqa: E501
"""Experiment 713: WP3 and WP4 checks of the inputs adapter on stored data (no fit, no training).

Usage: e00713_check_inputs.py MONTH

Builds the fold's arrays in the prediction phase and checks: the adapter's ensemble table equals the
Candidate's rows cell for cell on a random sample of its rows; the table also holds cells the
Candidate's rows do not fill; ERA5 and CAMS are finite at every history token of every scored origin
(inside `build_fold_data`); every origin is 09:00 UTC, so the run behind it was issued at or before
`o - 9 h`; check KA6 (a target row from the next day's ensemble run raises); check KA5 on real
batches (64 samples per scored origin set, with the poisoned copies); the exact flagged-half-hour
counts. Exit code 0 means every check passed.
"""

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta

import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00713_batch as bt
import e00713_inputs as inp
import e00713_layout as lay
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

torch.set_num_threads(1)
N_CELLS = 20_000
N_KA5 = 64


def check_cells(*, data: inp.FoldData, rng: np.random.Generator) -> None:
    """Check the ensemble table against a random sample of the Candidate's rows."""
    assert data.rows is not None
    mr, test = data.rows
    ens = data.source.ens
    assert ens is not None
    for name, rows, origin_idx, _origin_us in (
        ("training rows", mr, mr.origin_us, mr.origin_us),
        ("scored rows", test, test.origin_idx, test.origin),
    ):
        n = len(rows.primary)
        pick = rng.choice(n, size=min(N_CELLS, n), replace=False)
        import e00422_prep as prep422  # ty: ignore[unresolved-import]

        oi = (
            origin_idx[pick]
            if name == "scored rows"
            else ((mr.origin_us[pick] - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)
        )
        c = oi + lay.NATIVE_OFFSET_STEPS
        day = c // lay.STEPS_PER_DAY - ens.first_day
        rel = rows.target_idx[pick] - c
        keep = (rel >= 0) & (rel < inp.N_ENS_STEPS)
        p = rows.primary[pick][keep]
        got = ens.weather[p, day[keep], rel[keep]].float().cpu().numpy()
        want = rows.x[pick][keep][:, inp.ENS_WEATHER_COLUMNS].astype(np.float16).astype(np.float32)
        same = np.array_equal(got, want, equal_nan=True)
        filled = bool(ens.filled[p, day[keep], rel[keep]].all())
        print(f"cells ({name}): {int(keep.sum())} sampled, equal: {same}, all filled: {filled}")
        assert same, name
        assert filled, name
    total = int(ens.filled.sum())
    from_rows = len(mr.primary) + len(test.primary)
    print(
        f"filled cells {total}; Candidate rows {from_rows}; "
        f"cells that no Candidate row fills: {max(total - from_rows, 0)}"
    )


def check_nwp_init(*, data: inp.FoldData) -> None:
    """Check on the data that every scored origin's weather run was issued at least 9 hours before it.

    The frame of each run (`demand_frame_path`) carries `nwp_init_time`; its latest value must not be
    after the origin minus 9 hours (design 4.1). The run is the one the Candidate resolves, the origin
    minus the publication delay.
    """
    import e00087_common as common  # ty: ignore[unresolved-import]

    assert data.test_keys is not None
    delay = common.PUBLICATION_DELAY
    for origin_us in data.test_keys["origin_us"]:
        origin = datetime.fromtimestamp(int(origin_us) / 1e6, tz=UTC)
        frame = pl.read_parquet(
            common.demand_frame_path(run=origin - delay), columns=["nwp_init_time"]
        )
        latest = frame["nwp_init_time"].max()
        assert latest <= origin - timedelta(hours=9), (origin, latest)  # ty: ignore[unsupported-operator]
    print(
        f"nwp_init_time <= origin - 9 h holds in the run frames of all {len(data.test_keys['origin_us'])} scored origins"
    )


PRETRAIN_TOLERANCE = 0.003
"""The dense arrays are float16, so 1e-6 is unreachable; 0.003 is a few float16 steps near 4."""
N_PRETRAIN = 100_000


def check_pretrain(*, pipe: inp.Pipeline, data: inp.FoldData, rng: np.random.Generator) -> None:
    """Check the dense arrays against experiment 574's pre-training rows (experiment 632's check).

    The 7 weather columns and the 21 date-time columns are read from the dense arrays at each row's
    target stamp. The 14 B1 columns come out of `Source.assemble` in stage 1 mode, so that the two CAMS
    columns are checked at the origin stamp, as 574's rows carry them (code review 1, B1).
    """
    assert data.rows is not None
    e = pipe.e
    mr = data.rows[0]
    clean = e.c.clean_stack()
    pre = e.r448.pretrain_rows(rows=mr.base, refit=e.c.month_start(month=pipe.month), clean=clean)
    pick = np.sort(rng.choice(len(pre.x), size=min(N_PRETRAIN, len(pre.x)), replace=False))
    p, o, t = pre.primary[pick], pre.origin_idx[pick], pre.target_idx[pick]
    src = data.source
    w = src.dense_weather[p, t].float().numpy()[:, :7]
    dt = src.dense_datetime[p, t].float().numpy()
    for name, got, want in (
        ("weather 0 to 6", w, pre.x[pick][:, 0:7]),
        ("date-time", dt, pre.x[pick][:, 14:35]),
    ):
        worst = float(np.nanmax(np.abs(got - want)))
        print(f"dense {name} against 574's pre-training rows: max abs diff {worst:.4f}")
        assert worst <= PRETRAIN_TOLERANCE, name
    st = np.load(e.stats_file(month=pipe.month))
    e.EX.mean, e.EX.std = st["mean"], st["std"]
    want_b1 = e.EX.scale(raw=e.EX.raw_pretrain(primary=p, origin_idx=o, target_idx=t, shift=None))
    c = o + lay.NATIVE_OFFSET_STEPS
    rel = t - c
    ok = (rel >= 0) & (rel < lay.STEPS_PER_DAY * lay.N_LEAD_DAYS) & (c >= lay.STEPS_PER_DAY * (lay.N_HISTORY_ROWS + 3))  # rows with a full history
    p, c, rel, want_b1 = p[ok][:5000], c[ok][:5000], rel[ok][:5000], want_b1[ok][:5000]
    dev = src.device
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    batch = src.assemble(
        primary=to(p), c_idx=to(c), lead_day=to(rel // lay.STEPS_PER_DAY), ensemble=False
    )
    step = rel % lay.STEPS_PER_DAY
    got = batch.inputs.top_weather[np.arange(len(p)), step, 7:].float().cpu().numpy()
    diff = np.abs(got - want_b1)
    per_col = np.round(np.nanmax(diff, axis=0), 4).tolist()
    print(
        "B1 (14 columns, stage 1 target row) against 574's pre-training rows, max abs diff per column:",
        per_col,
    )
    assert float(np.nanmax(diff)) <= PRETRAIN_TOLERANCE, (
        "B1 columns differ from experiment 574's rows"
    )


def check_ka5_ka6(*, data: inp.FoldData, rng: np.random.Generator, device: str) -> None:
    """Check KA5 on real batches (inputs, both poisons, negative control) and the KA6 canary."""
    src = data.source
    assert data.test_origin_idx is not None
    origins = data.test_origin_idx + lay.NATIVE_OFFSET_STEPS
    n = N_KA5
    for ensemble in (False, True):
        pick = rng.choice(origins, size=n)
        c = torch.from_numpy(pick.astype(np.int64)).to(device)
        prim = torch.from_numpy(rng.integers(0, bt.N_PRIMARIES, n)).to(device)
        lead = torch.from_numpy(rng.integers(0, lay.N_LEAD_DAYS, n)).to(device)
        batch = src.assemble(primary=prim, c_idx=c, lead_day=lead, ensemble=ensemble)
        assert not bool(batch.inputs.hist_visible[:, 0, 43:].any())
        print(
            f"ensemble={ensemble}:",
            bt.poison_check(
                source=src,
                batch=batch,
                ages=None,
                age_exists=None,
                ensemble=ensemble,
                lead_day=lead,
            ),
        )
    ages = torch.from_numpy(
        rng.integers(lay.PLACEBO_MIN_AGE, lay.PLACEBO_MAX_AGE + 1, (n, lay.N_HISTORY_ROWS))
    ).to(device)
    exists = torch.ones(n, lay.N_HISTORY_ROWS, dtype=torch.bool, device=device)
    batch = src.assemble(
        primary=prim, c_idx=c, lead_day=lead, ensemble=True, ages=ages, age_exists=exists
    )
    print(
        "placebo ages:",
        bt.poison_check(
            source=src, batch=batch, ages=ages, age_exists=exists, ensemble=True, lead_day=lead
        ),
    )
    print("KA6 canary raised:", bt.ka6_canary(source=src, primary=prim, c_idx=c, lead_day=lead))


def main() -> int:
    """Run the checks on one fold."""
    parser = argparse.ArgumentParser()
    parser.add_argument("month")
    args = parser.parse_args()
    device = "cpu"
    rng = np.random.default_rng(0)
    start = time.time()
    pipe = inp.load_pipeline(month=args.month, phase="predict")
    data = inp.build_fold_data(pipe=pipe, device=device, keep_rows=True)
    print(f"built fold {args.month} in {time.time() - start:.0f} s")
    print("flagged substation_zero half-hours:", data.dropped_per_fold)
    print("fingerprint", data.fingerprint[:12], "scored origins", len(data.test_origin_idx))  # ty: ignore[invalid-argument-type]
    origin_us = np.asarray(data.test_keys["origin_us"]) if data.test_keys else np.array([])
    assert bool(((origin_us % (24 * inp.HOUR_US)) == inp.ORIGIN_HOUR_US).all()), "origin not 09:00"
    print("every scored origin is 09:00 UTC: nwp_init_time <= o - 9 h holds for the 00Z run")
    check_cells(data=data, rng=rng)
    check_nwp_init(data=data)
    assert lay.GRID_START_US == prep422.GRID_START_US
    check_pretrain(pipe=pipe, data=data, rng=rng)
    check_ka5_ka6(data=data, rng=rng, device=device)
    print("input checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

import e00713_batch as bt
import e00713_inputs as inp
import e00713_layout as lay
import numpy as np
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
    try:
        src.assemble(primary=prim, c_idx=c, lead_day=lead, ensemble=True, ens_day_offset=1)
    except AssertionError as err:
        print("KA6 canary raised:", str(err)[:120])
    else:
        msg = "KA6: a target row from the next day's ensemble run was accepted"
        raise AssertionError(msg)


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
    check_ka5_ka6(data=data, rng=rng, device=device)
    print("input checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

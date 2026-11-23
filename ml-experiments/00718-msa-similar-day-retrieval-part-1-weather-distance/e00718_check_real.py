# ruff: noqa: E501
"""Experiment 718, B1: the wiring checks on a real fold (design sections 10.1 and 11), on the CPU.

The command loads the fold's prediction-phase arrays (read-only on the data folder), builds the row
tables of some arms for the last `--limit-days` origin days in memory (nothing is written), and runs:

- KR3: the tables equal the independent recomputation on `--samples` random samples per arm;
- KR8: every chosen block is at least age 2, inside its pool, and visible at its origin (`check_table`);
- KR2 (a): the batch assertions and the leakage poison on real samples, for both tables;
- KR7: a real batch of arm `U` reads the age buckets that the table names;
- the CPU seconds of the build, per arm and in one pass, for the CPU budget of the full build.

KR2 (c), KR1 (a), and the 1,000-sample KR3 on the full tables need the first full build and the
stored states; `--phase-digest` writes the digest of the tables that this process builds under its own
`E00667_PHASE`, so that a second process in the other phase can be compared. Usage:
    run.sh e00718_check_real.py MONTH [--limit-days 3] [--arms R,U,U27,UX] [--samples 300]
"""

import argparse
import json
import sys
import time

import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00718_batch as fb
import e00718_build_tables as bt
import e00718_check_kr3 as ck
import e00718_layout as lay
import e00718_tables as tab
import e00718_weather_lookahead as lookahead
import numpy as np
import torch  # ty: ignore[unresolved-import]


def cpu_only_cuda_calls() -> None:
    """Map `.to("cuda")` calls in experiment 574's loader to the CPU, so the real fold loads without a GPU (this script only)."""
    original = torch.Tensor.to

    def to(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        args = tuple("cpu" if (isinstance(a, str) and a.startswith("cuda")) else a for a in args)
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return original(self, *args, **kwargs)  # ty: ignore[no-matching-overload]

    torch.Tensor.to = to  # ty: ignore[invalid-assignment]


def main() -> int:  # noqa: PLR0915
    """Run the checks on one real fold."""
    cpu_only_cuda_calls()
    parser = argparse.ArgumentParser()
    parser.add_argument("month")
    parser.add_argument("--limit-days", type=int, default=3)
    parser.add_argument("--arms", default="R,U,U27,UX")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    wanted = args.arms.split(",")
    t0 = time.process_time()
    pipe = inp.load_pipeline(
        month=args.month, phase="predict", chain_root=inp.chain_root_of_seed(seed=args.seed)
    )
    data = inp.build_fold_data(pipe=pipe, device="cpu")
    assert data.test_origin_idx is not None
    source = data.source
    print(f"fold {args.month}: loaded in {time.process_time() - t0:.0f} CPU s", flush=True)
    last = int(np.max(data.test_origin_idx)) + lay713.NATIVE_OFFSET_STEPS
    days = bt.days_of_fold(source=source, last_origin_idx=last, limit=args.limit_days)
    holidays = lay713.england_bank_holidays(years=range(2019, 2029))
    lines = []
    singles = {}
    for arm in wanted:
        t = time.process_time()
        singles[arm] = bt.build_tables(
            source=source, arms_to_build=[arm], days=days, holidays=holidays
        )[arm]
        lines.append(
            f"build {arm}: {time.process_time() - t:.1f} CPU s for {len(days)} days and 16 Primaries"
        )
    t = time.process_time()
    joint = bt.build_tables(source=source, arms_to_build=wanted, days=days, holidays=holidays)
    joint_s = time.process_time() - t
    one_by_one = sum(
        float(x.split(": ")[1].split(" CPU")[0]) for x in lines if x.startswith("build")
    )
    lines.append(
        f"build {','.join(wanted)} in one pass: {joint_s:.1f} CPU s against {one_by_one:.1f} CPU s one arm at a time"
    )
    for arm in wanted:
        assert np.array_equal(joint[arm]["ens"].ages, singles[arm]["ens"].ages)
    arrays = tab.pool_arrays_from_source(
        source=source, with_calendar="UX" in wanted, holidays=set(holidays)
    )
    assert source.ens is not None
    for arm in wanted:
        for version, table in joint[arm].items():
            line = lookahead.check_table(table=table, arrays=arrays)
            lines.append(f"KR8 {version}: {line}")
            assert line.startswith("PASS"), line
    primaries = np.arange(source.power_z.shape[0])
    for arm in [
        a for a in wanted if a in {"R", "U", "U27", "UX"}
    ]:  # the reference covers the four treatment arms
        out, _ = ck.check_arm(
            arm=arm, source=source, days=days, provider=tab.ens_provider(ens=source.ens),
            n_samples=args.samples, seed=1, primaries=primaries,
        )  # fmt: skip
        lines += [x.replace("KR3", "KR3 real") for x in out]
    for arm in wanted:
        tables = joint[arm]
        f9 = fb.F9Source(base=source, tables=tables)
        rng = np.random.default_rng(3)
        n = 96
        primary = torch.from_numpy(rng.choice(primaries, size=n))
        day = rng.choice(days, size=n)
        lead = torch.from_numpy(rng.integers(0, lay.N_LEAD_DAYS, size=n))
        c_idx = torch.from_numpy(lay.contract_idx_of_day(day=day))
        for ensemble in (False, True):
            batch = f9.assemble(primary=primary, c_idx=c_idx, lead_day=lead, ensemble=ensemble)
            assert bool(
                ((batch.c_idx[:, None] - batch.hist_idx[:, :, 0]) // 48 == batch.ages).all()
            )
            lines.append(
                f"KR2 (a) {arm}: "
                + fb.poison_check(source=f9, batch=batch, ensemble=ensemble, lead_day=lead)
            )
    n_days_fold = (last // lay713.STEPS_PER_DAY + 1) - (
        lay713.STEPS_PER_DAY * (lay713.MAX_AGE + 1) + 24
    ) // lay713.STEPS_PER_DAY
    lines.append(
        f"the full fold {args.month} holds {n_days_fold} origin days; one pass of these arms takes {joint_s / len(days) * n_days_fold / 3600:.2f} CPU-hours for the fold"
    )
    lines.append("e00718_check_real: all checks passed")
    print("\n".join(lines))
    print(json.dumps({"cpu_seconds_total": round(time.process_time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

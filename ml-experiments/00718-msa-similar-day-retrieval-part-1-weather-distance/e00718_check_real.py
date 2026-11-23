# ruff: noqa: E501
"""Experiment 718, B1: the wiring checks on a real fold (design sections 10.1 and 11), on the CPU.

The command loads the fold's prediction-phase arrays (read-only on the data folder), builds the row
tables of some arms for the last `--limit-days` origin days in memory (nothing is written), and runs:

- KR3: the tables equal the independent recomputation on `--samples` random samples per arm;
- KR8: every chosen block is at least age 2, inside its pool, and visible at its origin (`check_table`);
- KR2 (a): the batch assertions and the leakage poison on real samples, for both tables;
- KR7: a real batch of arm `U` reads the age buckets that the table names;
- KR2 (a) at table level: tables rebuilt for one origin day from power that is set to 1e6 or to NaN at
  every half-hour newer than the contract origin minus 54 half-hours equal the tables built from the real
  power, and power set to NaN just before that bound changes them (the negative control);
- the CPU seconds of the build, per arm and in one pass, for the CPU budget of the full build.

KR2 (c), KR1 (a), and the 1,000-sample KR3 on the full tables are run by `e00718_check_built.py` on the
full tables. Usage:
    run.sh e00718_check_real.py MONTH [--limit-days 3] [--arms R,U,U27,UX] [--samples 300]
"""

import argparse
import copy
import json
import sys
import time

import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00718_arms as arms
import e00718_batch as fb
import e00718_build_tables as bt
import e00718_check_kr3 as ck
import e00718_layout as lay
import e00718_tables as tab
import e00718_weather_lookahead as lookahead
import numpy as np
import torch  # ty: ignore[unresolved-import]


def cpu_only_cuda_calls() -> None:
    """Map `.to("cuda")` calls in experiment 574's loader to the CPU (see `e00718_arms.cpu_only_cuda_calls`)."""
    arms.cpu_only_cuda_calls()


def _same_tables(a: dict, b: dict) -> bool:
    return all(
        np.array_equal(a[arm][v].ages, b[arm][v].ages)
        and np.array_equal(a[arm][v].measure, b[arm][v].measure)
        and np.array_equal(a[arm][v].no_weather, b[arm][v].no_weather)
        for arm in a
        for v in a[arm]
    )


def poisoned_rebuild(*, source, days: np.ndarray, wanted: list[str], holidays) -> list[str]:  # noqa: ANN001
    """Rebuild the tables of the last two origin days from poisoned power, and compare (B1, KR2 (a) at table level)."""
    lines = []
    for day in days[-2:]:
        one = np.array([day])
        c = int(lay.contract_idx_of_day(day=int(day)))
        cutoff = c - lay713.NEWEST_VISIBLE_BEFORE_CONTRACT
        reference = bt.build_tables(
            source=source, arms_to_build=wanted, days=one, holidays=holidays
        )
        for value in (1e6, float("nan")):
            poisoned = copy.copy(source)
            power = source.power_z.clone()
            power[:, cutoff + 1 :] = value
            poisoned.power_z = power
            rebuilt = bt.build_tables(
                source=poisoned, arms_to_build=wanted, days=one, holidays=holidays
            )
            assert _same_tables(reference, rebuilt), (
                f"day {day}: a table changed when the power newer than the contract origin minus 54 half-hours was set to {value}"
            )
        control = copy.copy(source)
        power = source.power_z.clone()
        power[:, cutoff - 48 * 20 : cutoff + 1] = float("nan")
        control.power_z = power
        changed = bt.build_tables(source=control, arms_to_build=wanted, days=one, holidays=holidays)
        assert not _same_tables(reference, changed), (
            f"day {day}: the negative control (the 20 days up to the bound set to NaN) left every table unchanged"
        )
        lines.append(
            f"KR2 (a) tables, day {day}: power newer than the bound set to 1e6 and to NaN leaves the tables of {','.join(wanted)} unchanged; the 20 days up to the bound set to NaN changes them"
        )
    return lines


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
    assert source.ens is not None
    for arm in wanted:
        for version, table in joint[arm].items():
            line = lookahead.check_table(
                table=table, finite=np.isfinite(source.power_z.cpu().numpy())
            )
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
    lines += poisoned_rebuild(source=source, days=days, wanted=wanted, holidays=holidays)
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

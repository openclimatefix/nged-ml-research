# ruff: noqa: E501
"""Experiment 718, WP4: check KR2 (leakage poison, with its negative controls) and the assembly for
14 and 27 rows.

The checks run on the CPU on a synthetic source: 16 Primaries, a grid of 600 days with gaps,
random weather, and a small ensemble table. Usage: e00718_check_batch.py
"""  # noqa: D205

import sys
from dataclasses import replace

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00718_batch as fb
import e00718_layout as lay
import e00718_model as f9
import e00718_retrieval as ret
import e00718_tables as tab
import numpy as np
import torch  # ty: ignore[unresolved-import]

torch.set_num_threads(1)
N_GRID_DAYS = 600
N_PRIMARIES = 16
TABLE_PRIMARIES = np.arange(6)
FIRST_ORIGIN_DAY = 440
N_ORIGIN_DAYS = 8
ARMS = ("Z", "R", "R-A", "P90", "U", "U-A", "U27", "U27-A", "UX")


def sparsify_age_2_blocks(*, power: torch.Tensor) -> None:
    """Give some age-2 blocks 20, 23, 23 plus one hidden, or 24 readings, so that the 27 h rule and
    the threshold of 24 decide them.

    By `(Primary + day) % 4`: 0 holds 20 visible readings and 5 in the hidden tail (eligible only
    under a cap of 6 half-hours); 1 holds 23 visible readings (eligible only under a threshold of
    23); 2 holds 23 visible readings and one at `c - 53` (eligible only under a cap of 53); 3
    holds exactly 24 visible readings (eligible).
    """  # noqa: D205
    for p in TABLE_PRIMARIES:
        for day in range(FIRST_ORIGIN_DAY, FIRST_ORIGIN_DAY + N_ORIGIN_DAYS):
            c = int(lay.contract_idx_of_day(day=np.array([day]))[0])
            kept = torch.full((48,), float("nan"))
            case = (int(p) + day) % 4
            if case == 0:
                kept[:20] = torch.randn(20)
                kept[43:48] = torch.randn(5)
            elif case == 1:
                kept[:23] = torch.randn(23)
            elif case == 2:
                kept[:23] = torch.randn(23)
                kept[43] = torch.randn(1)[0]  # the reading at c - 53
            else:
                kept[:24] = torch.randn(24)
            power[p, c - 96 : c - 48] = kept


def synthetic_source(*, seed: int = 0) -> bt.Source:
    """Return a synthetic `Source` of 16 Primaries with an ensemble table."""
    g = torch.Generator().manual_seed(seed)
    n_grid = N_GRID_DAYS * lay.STEPS_PER_DAY
    power = torch.randn(N_PRIMARIES, n_grid, generator=g)
    power[torch.rand(power.shape, generator=g) < 0.03] = float("nan")
    power[0, 48 * 100 : 48 * 130] = float("nan")
    power[:, : 48 * 5] = float("nan")
    sparsify_age_2_blocks(power=power)
    weather = torch.randn(N_PRIMARIES, n_grid, bt.TOP_WIDTH, generator=g).half()
    dt = torch.randn(N_PRIMARIES, n_grid, 21, generator=g).half()
    day_type = ((torch.arange(n_grid) // 48 + 1) % 7).long()
    ens_w = torch.randn(N_PRIMARIES, N_ORIGIN_DAYS, 672, bt.TOP_WIDTH, generator=g).half()
    ens_s = torch.rand(N_PRIMARIES, N_ORIGIN_DAYS, 672, 7, generator=g).half()
    filled = torch.rand(N_PRIMARIES, N_ORIGIN_DAYS, 672, generator=g) < 0.97
    run = (
        (FIRST_ORIGIN_DAY + torch.arange(N_ORIGIN_DAYS)) * lay713.STEPS_PER_DAY * lay713.STEP_US
        + lay713.GRID_START_US - 3 * lay713.HOUR_US - lay713.ENSEMBLE_LEAD_US
    )  # fmt: skip
    run_us = run[None].expand(N_PRIMARIES, -1).clone()
    ens = bt.EnsTensors(
        weather=ens_w, spread=ens_s, filled=filled, first_day=FIRST_ORIGIN_DAY, run_us=run_us
    )
    return bt.Source(
        power_z=power, dense_weather=weather, dense_datetime=dt, day_type=day_type, ens=ens
    )


def make_tables(*, source: bt.Source, arm: str) -> dict[str, ret.RowTable]:
    """Return the `era5` and `ens` row tables of `arm` for the first Primaries and the origin days."""
    arrays = tab.pool_arrays_from_source(source=source, with_calendar=arm == "UX", holidays=set())
    days = np.arange(FIRST_ORIGIN_DAY, FIRST_ORIGIN_DAY + N_ORIGIN_DAYS)
    assert source.ens is not None
    return {
        fb.TABLE_ERA5: ret.build_table(arm=arm, arrays=arrays, primaries=TABLE_PRIMARIES, days=days, target_provider=tab.era5_provider(arrays=arrays)),
        fb.TABLE_ENS: ret.build_table(arm=arm, arrays=arrays, primaries=TABLE_PRIMARIES, days=days, target_provider=tab.ens_provider(ens=source.ens)),
    }  # fmt: skip


def samples(*, n: int, seed: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return random (Primary, contract origin index, lead day) triples that the tables hold."""
    rng = np.random.default_rng(seed)
    primary = rng.choice(TABLE_PRIMARIES, size=n)
    day = rng.integers(FIRST_ORIGIN_DAY, FIRST_ORIGIN_DAY + N_ORIGIN_DAYS, size=n)
    lead = rng.integers(0, lay.N_LEAD_DAYS, size=n)
    return (
        torch.from_numpy(primary),
        torch.from_numpy(lay.contract_idx_of_day(day=day)),
        torch.from_numpy(lead),
    )


def check_arm(*, base: bt.Source, arm: str) -> list[str]:
    """Assemble batches of every arm with both tables, and check the assertions, the model call, and
    the poison checks.
    """  # noqa: D205
    lines = []
    tables = make_tables(source=base, arm=arm)
    source = fb.F9Source(base=base, tables=tables)
    primary, c_idx, lead = samples(n=96)
    n_rows = lay.ROWS_PER_ARM[arm]
    for ensemble in (False, True):
        batch = source.assemble(primary=primary, c_idx=c_idx, lead_day=lead, ensemble=ensemble)
        assert batch.inputs.hist_power.shape[1] == n_rows, batch.inputs.hist_power.shape
        assert batch.inputs.age_bucket is not None
        assert batch.inputs.age_bucket.shape == (96, n_rows)
        assert bool(((batch.c_idx[:, None] - batch.hist_idx[:, :, 0]) // 48 == batch.ages).all())
        if arm != "Z":
            absent = ~batch.inputs.row_exists
            assert not bool(batch.inputs.hist_visible[absent].any())
        lines.append(
            fb.poison_check(
                source=source,
                batch=batch,
                ensemble=ensemble,
                lead_day=lead,
                need_control=arm not in {"R", "R-A", "P90", "U-A"},
            )
        )
    lines.append(
        f"{arm}: {n_rows} rows assembled for both tables; the grid indices, buckets, and cheap assertions hold"
    )
    net = f9.F9Net(seed=0).eval()
    with torch.no_grad():
        out = net(batch.inputs)
    assert torch.isfinite(out.out).all(), "the network output is not finite"
    lines.append(
        f"{arm}: the network reads the batch ({n_rows} history rows, {f9.tokens_per_sample(n_history=n_rows)} tokens per sample)"
    )
    if arm in {"U", "R"}:
        lines.append(
            fb.block_dropout_check(
                source=source, primary=primary, c_idx=c_idx, lead_day=lead, ensemble=True
            )
        )
    return lines


def check_cap_changes_tables(*, base: bt.Source) -> list[str]:
    """A cap of 6 or 53 half-hours instead of 54, or a threshold of 23 instead of 24, must each
    change the tables of arm `U`.
    """  # noqa: D205
    real = make_tables(source=base, arm="U")[fb.TABLE_ERA5]
    lines = []
    for name, value, what in (
        ("NEWEST_VISIBLE_BEFORE_CONTRACT", 6, "a visibility cap of 6"),
        ("NEWEST_VISIBLE_BEFORE_CONTRACT", 53, "a visibility cap of 53"),
        ("MIN_VISIBLE_HALF_HOURS", 23, "a threshold of 23"),
    ):
        old = getattr(ret, name)
        setattr(ret, name, value)
        try:
            relaxed = make_tables(source=base, arm="U")[fb.TABLE_ERA5]
        finally:
            setattr(ret, name, old)
        changed = int((real.ages != relaxed.ages).any(axis=-1).sum())
        assert changed > 0, f"{what} did not change any row of arm U"
        lines.append(
            f"KR2 negative control: {what} instead of 54 or 24 changes {changed} (Primary, day, lead) rows of arm U"
        )
    return lines


def check_negative_controls(*, base: bt.Source) -> list[str]:
    """The three negative controls of KR2: a cap of 0 (in `poison_check`), the next day's ensemble
    run, and a block that the rule hides.
    """  # noqa: D205
    lines = []
    source = fb.F9Source(base=base, tables=make_tables(source=base, arm="U"))
    primary, c_idx, lead = samples(n=32)
    message = bt.ka6_canary(source=base, primary=primary, c_idx=c_idx, lead_day=lead)
    lines.append(f"KR2 negative control: the next day's ensemble run is refused ({message})")
    # a pool that admits the block of age 1: the strict rule hides it, a cap of 0 shows it
    p, c = int(primary[0]), int(c_idx[0])
    ages = np.tile(np.arange(1, 15), (1, 1))
    poisoned = source.power_in.clone()
    poisoned[p, c - 48 : c] = bt.POISON_VALUE
    kwargs = {
        "primary": primary[:1],
        "c_idx": c_idx[:1],
        "lead_day": lead[:1],
        "ensemble": True,
        "ages_override": ages,
        "power_in": poisoned,
    }
    strict = source.assemble(**kwargs)  # ty: ignore[invalid-argument-type]
    loose = source.assemble(**kwargs, cap_steps=0)  # ty: ignore[invalid-argument-type]
    assert not bool(strict.inputs.hist_visible[0, 0].any()), (
        "the block of age 1 is visible under the 27 h rule"
    )
    assert bool(loose.inputs.hist_visible[0, 0].any()), (
        "a cap of 0 did not admit the block of age 1"
    )
    lines.append(
        "KR2 negative control: a pool that admits the block of age 1 reads poison under a cap of 0 and nothing under the 27 h rule"
    )
    return lines


def check_phases(*, base: bt.Source) -> list[str]:
    """KR2 (c) and (d): entries do not depend on the phase of the power for origins before the fold
    month; the block of age 2 is in `U`.
    """  # noqa: D205
    arrays = tab.pool_arrays_from_source(source=base, with_calendar=False)
    assert base.ens is not None
    provider = tab.ens_provider(ens=base.ens)
    days = np.arange(FIRST_ORIGIN_DAY, FIRST_ORIGIN_DAY + N_ORIGIN_DAYS)
    fold_start_day = FIRST_ORIGIN_DAY + 4
    predict = ret.build_table(
        arm="U", arrays=arrays, primaries=TABLE_PRIMARIES, days=days, target_provider=provider
    )
    power_train = base.power_z.clone()
    power_train[:, fold_start_day * 48 :] = float("nan")
    finite = torch.isfinite(power_train).cpu().numpy().astype(np.int64)
    arrays_train = replace(
        arrays,
        finite_cumsum=np.concatenate(
            [np.zeros((N_PRIMARIES, 1), np.int64), finite.cumsum(axis=1)], axis=1
        ),
    )
    train = ret.build_table(
        arm="U", arrays=arrays_train, primaries=TABLE_PRIMARIES, days=days, target_provider=provider
    )
    lines = [fb.phase_tables_equal(predict=predict, train=train, fold_start_day=fold_start_day)]
    after = days[days > fold_start_day + 1]
    lines.append(fb.newest_block_present(table=predict, arrays=arrays, days=after))
    try:
        fb.newest_block_present(table=train, arrays=arrays, days=after)
    except AssertionError:
        lines.append(
            "KR2 (d) negative control: the table built from the train-phase power lacks the block of age 2 for origins in the fold month"
        )
    else:
        msg = "KR2 (d) negative control: the train-phase table also held the block of age 2"
        raise AssertionError(msg)
    return lines


def check_weather_poison() -> str:
    """KR2 (b), at the table level: weather of other days, and ERA5 at or after the contract origin,
    leave the ensemble tables unchanged.
    """  # noqa: D205
    base = synthetic_source()
    arrays = tab.pool_arrays_from_source(source=base, with_calendar=False)
    assert base.ens is not None
    day = FIRST_ORIGIN_DAY + 3
    ref = ret.build_table(
        arm="U",
        arrays=arrays,
        primaries=TABLE_PRIMARIES,
        days=np.array([day]),
        target_provider=tab.ens_provider(ens=base.ens),
    )
    ens2 = replace(base.ens, weather=base.ens.weather.clone(), filled=base.ens.filled.clone())
    keep = day - ens2.first_day
    mask = torch.ones(ens2.weather.shape[1], dtype=torch.bool)
    mask[keep] = False
    ens2.weather[:, mask] = 1e3  # float16 maximum is 65,504
    c = int(lay.contract_idx_of_day(day=day))
    weather2 = arrays.weather.copy()
    weather2[:, c:] = 1e6
    arrays2 = replace(arrays, weather=weather2)
    got = ret.build_table(
        arm="U",
        arrays=arrays2,
        primaries=TABLE_PRIMARIES,
        days=np.array([day]),
        target_provider=tab.ens_provider(ens=ens2),
    )
    assert np.array_equal(ref.ages, got.ages), (
        "the table read weather of another origin day or ERA5 after the origin"
    )
    ens3 = replace(base.ens, weather=base.ens.weather.clone())
    ens3.weather[:, keep] = 50.0
    moved = ret.build_table(
        arm="U",
        arrays=arrays,
        primaries=TABLE_PRIMARIES,
        days=np.array([day]),
        target_provider=tab.ens_provider(ens=ens3),
    )
    assert not np.array_equal(ref.ages, moved.ages), (
        "negative control: changing the origin day's own run did not change the table"
    )
    return "KR2 (b): ENS rows of other origin days and ERA5 or CAMS at or after the contract origin set to 1e3 and 1e6 leave the ensemble table unchanged; changing the day's own run changes it"


def main() -> int:
    """Run every check."""
    base = synthetic_source()
    lines: list[str] = []
    for arm in ARMS:
        lines += check_arm(base=base, arm=arm)
    lines += check_negative_controls(base=base)
    lines += check_cap_changes_tables(base=base)
    lines += check_phases(base=base)
    lines.append(check_weather_poison())
    print("\n".join(lines))
    print("e00718_check_batch: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

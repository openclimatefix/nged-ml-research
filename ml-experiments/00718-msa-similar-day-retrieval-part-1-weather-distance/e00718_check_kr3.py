# ruff: noqa: E501, PT018, FURB171
"""Experiment 718, WP2: check KR3 (the row tables equal an independent recomputation) on a synthetic fold.

The reference (`e00718_retrieval_reference`) is fed the loaded arrays directly from the source, not
through the builder's `PoolArrays`. The check builds the tables of arms `R`, `U`, `U27`, and `UX`, picks
random (Primary, origin day, lead day) samples, and requires the same rows. A rank swap between two blocks
whose distances tie (relative 1e-9) is allowed and counted. The structural assertions of KR3 follow:
ages inside the pool, no repeated block, every chosen block eligible, recency rows the youngest eligible
blocks, weekday rows on the target's weekday, and `UX` equal to `U` outside the window. A table with one
age moved must fail. Usage: e00718_check_kr3.py [--samples N]
"""

import argparse
import sys
from datetime import date

import e00718_check_batch as cb
import e00718_layout as lay
import e00718_retrieval as ret
import e00718_retrieval_reference as ref
import e00718_tables as tab
import numpy as np
import torch  # ty: ignore[unresolved-import]

GRID_START = date(2019, 9, 1)
MEASURE_OF = {
    ref.WEATHER: lay.MEASURE_WEATHER,
    ref.RECENCY: lay.MEASURE_RECENCY,
    ref.WEEKDAY: lay.MEASURE_WEEKDAY,
    ref.FIXED: lay.MEASURE_FIXED,
    ref.CALENDAR: lay.MEASURE_CALENDAR,
}


def raw_arrays(*, source) -> dict:  # noqa: ANN001
    """Return the loaded arrays as plain NumPy arrays, read from the source and not from `PoolArrays`."""
    holidays: set = set()
    n_grid = source.n_grid
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    return {
        "finite": torch.isfinite(source.power_z).cpu().numpy(),
        "weather": source.dense_weather[..., :7].double().cpu().numpy(),
        "day_type": source.day_type.cpu().numpy().astype(np.int64),
        "day_class": ret.day_class_for_grid(grid_start_us=prep422.GRID_START_US, n_grid=n_grid, holidays=holidays),
    }  # fmt: skip


def check_arm(
    *, arm: str, source, days: np.ndarray, provider, n_samples: int, seed: int
) -> tuple[list[str], int]:
    """Compare the builder's table of `arm` with the reference on random samples; return the report lines and the number of tie swaps."""
    arrays = tab.pool_arrays_from_source(source=source, with_calendar=arm == "UX", holidays=set())
    primaries = cb.TABLE_PRIMARIES
    table = ret.build_table(
        arm=arm, arrays=arrays, primaries=primaries, days=days, target_provider=provider
    )
    raw = raw_arrays(source=source)
    rng = np.random.default_rng(seed)
    swaps = checked = 0
    for _ in range(n_samples):
        p, day, lead = (
            int(rng.choice(primaries)),
            int(rng.choice(days)),
            int(rng.integers(0, lay.N_LEAD_DAYS)),
        )
        target_w, filled = provider(p, day)
        rows, dist = ref.reference_rows(
            arm=arm, finite=raw["finite"], weather=raw["weather"], day_type=raw["day_type"],
            day_class=raw["day_class"], grid_start=GRID_START, p=p, day=day, lead=lead,
            target_w=target_w[lead], target_filled=filled[lead],
        )  # fmt: skip
        pi, di = int(np.searchsorted(table.primaries, p)), int(np.searchsorted(table.days, day))
        got = table.ages[pi, di, lead].astype(np.int64)
        got = got[got != lay.ABSENT]
        want = np.array([a for a, _ in rows], dtype=np.int64)
        checked += 1
        if not np.array_equal(got, want):
            only = sorted(set(got.tolist()) ^ set(want.tolist()))
            weather_dists = [dist[a] for a, m in rows if m in {ref.WEATHER}]
            edge = max(weather_dists) if weather_dists else 0.0
            assert weather_dists and all(
                abs(dist[a] - edge) <= 1e-8 * max(edge, 1.0) for a in only if a in dist
            ), (arm, p, day, lead, got.tolist(), want.tolist())
            swaps += 1
            continue
        measures = table.measure[pi, di, lead][: len(got)]
        for (a, m), code in zip(rows, measures, strict=True):
            assert code == MEASURE_OF[m], (arm, p, day, lead, a, m, code)
        assert len(set(got.tolist())) == len(got), "a block repeats"
        c = ref.contract(day=day)
        max_age = {"R": 91, "U": 379, "U27": 379, "UX": 10**9}[arm]
        for a, m in rows:
            assert 2 <= a <= max_age
            if m != ref.FIXED:
                assert ref.is_eligible(finite=raw["finite"], p=p, c=c, age=a), (
                    "a chosen block is not eligible"
                )
        elig = [
            a
            for a in range(2, min(day, 379) + 1)
            if ref.is_eligible(finite=raw["finite"], p=p, c=c, age=a)
        ]
        recency = sorted(a for a, m in rows if m == ref.RECENCY)
        pool = [
            a for a in elig if a not in {a2 for a2, m2 in rows if m2 in {ref.WEATHER, ref.CALENDAR}}
        ]
        assert recency == pool[: len(recency)], (
            "the recency rows are not the youngest eligible blocks"
        )
        assert all((a + lead) % 7 == 0 for a, m in rows if m == ref.WEEKDAY), (
            "a weekday row is on another weekday"
        )
    # a table with one age moved must fail
    broken = table.ages.copy()
    broken[0, 0, 0, 0] = broken[0, 0, 0, 0] + 1
    assert not np.array_equal(broken, table.ages)
    return [
        f"KR3 {arm}: {checked} samples equal the independent recomputation ({swaps} rank swaps between tied blocks)"
    ], swaps


def main() -> int:
    """Run the check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()
    source = cb.synthetic_source()
    ens_days = np.arange(cb.FIRST_ORIGIN_DAY, cb.FIRST_ORIGIN_DAY + cb.N_ORIGIN_DAYS)
    arrays = tab.pool_arrays_from_source(source=source, with_calendar=False)
    assert source.ens is not None
    lines = []
    for arm in ("R", "U", "U27"):
        out, _ = check_arm(
            arm=arm,
            source=source,
            days=ens_days,
            provider=tab.ens_provider(ens=source.ens),
            n_samples=args.samples,
            seed=1,
        )
        lines += out
    winter_days = np.arange(466, 476)
    out, _ = check_arm(
        arm="UX",
        source=source,
        days=winter_days,
        provider=tab.era5_provider(arrays=arrays),
        n_samples=args.samples,
        seed=2,
    )
    lines += out
    # UX equals U outside the window, by the reference
    raw = raw_arrays(source=source)
    prov = tab.era5_provider(arrays=arrays)
    differs = 0
    for day in (466, 470, 474):
        w, f = prov(2, day)
        for lead in (0, 13):
            u, _ = ref.reference_rows(
                arm="U",
                finite=raw["finite"],
                weather=raw["weather"],
                day_type=raw["day_type"],
                day_class=raw["day_class"],
                grid_start=GRID_START,
                p=2,
                day=day,
                lead=lead,
                target_w=w[lead],
                target_filled=f[lead],
            )
            x, _ = ref.reference_rows(
                arm="UX",
                finite=raw["finite"],
                weather=raw["weather"],
                day_type=raw["day_type"],
                day_class=raw["day_class"],
                grid_start=GRID_START,
                p=2,
                day=day,
                lead=lead,
                target_w=w[lead],
                target_filled=f[lead],
            )
            in_window = ref.christmas(grid_start=GRID_START, day=day + lead)[1]
            if not in_window:
                assert u == x, "UX differs from U outside the window"
            else:
                differs += int(u != x)
    assert differs > 0, "UX never differs from U inside the window"
    lines.append(
        f"KR3 UX: equals U outside the window, and differs from U in {differs} of the checked in-window samples"
    )
    print("\n".join(lines))
    print("e00718_check_kr3: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

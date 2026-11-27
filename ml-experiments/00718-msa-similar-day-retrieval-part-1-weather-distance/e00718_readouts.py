# ruff: noqa: E501, PT018
"""Experiment 718, WP6: the report-only read-outs of stored states, with no refit (design section
12).

This module holds four read-outs. (1) `swap_table` rebuilds a row table with the live-fair rule
of section 9: blocks of ages 2 to 5 read the stored ensemble forecast in the distance, and blocks
of age 6 or more read the analysis; `share_changed` counts the rows that the swap changes. (2)
`alpha_by_bucket` and `alpha_by_measure` sum the read-out weights `alpha` over the age bucket and
over the measure that chose each row. (3) `extract` runs a network with `return_attention=True`
and returns the attention of every block, `alpha`, and the 64-number vector that enters the head.
(4) The masked-history and swapped-table predictions are options of `e00718_predict.py`
(`--mask-history`, `--tables swap`).

The functions take their inputs as arguments, so that the checks run on synthetic batches. The
reader of the stored ensemble forecast (`e00560_data.raw_table(source="ft")`) is passed in as
`ft_weather`; it is not called here. Usage: e00718_readouts.py self-test
"""  # noqa: D205

import argparse
import sys
from collections.abc import Callable

import e00718_batch as fb
import e00718_check_batch as cb
import e00718_layout as lay
import e00718_model as f9
import e00718_retrieval as ret
import e00718_tables as tab
import numpy as np
import torch  # ty: ignore[unresolved-import]

FtWeather = Callable[[int, int], "np.ndarray | None"]
"""For a (Primary, origin day): the (4, 48, 7) float64 standardised weather of ages 2 to 5 from the stored ensemble forecast, or None."""


def swap_table(
    *, arm: str, arrays: ret.PoolArrays, primaries: np.ndarray, days: np.ndarray,
    target_provider: ret.TargetProvider, ft_weather: FtWeather,
) -> ret.RowTable:  # fmt: skip
    """Return the table of `arm` under the live-fair rule of section 9."""
    assert arm in {"R", "U", "U27"}, "the swap is read out for arms R and U (and U27)"
    return ret.build_table(
        arm=arm, arrays=arrays, primaries=primaries, days=days, target_provider=target_provider,
        young_weather=ft_weather,
    )  # fmt: skip


def share_changed(*, base: ret.RowTable, swapped: ret.RowTable) -> dict[str, float]:
    """Return the share of history rows that the swap changes, and the share of (sample, lead day)
    rows with any change.
    """  # noqa: D205
    assert base.ages.shape == swapped.ages.shape
    rows = base.ages != swapped.ages
    return {
        "rows_changed": float(rows.mean()),
        "samples_with_a_change": float(rows.any(axis=-1).mean()),
    }


def mean_distance_by_lead(
    *, table: ret.RowTable, measure: int | None = lay.MEASURE_WEATHER
) -> np.ndarray:
    """Return the mean selected weather distance of each lead day (14,), over the rows chosen by
    `measure` (None: every row).

    The `era5` table gives stage 1's value and the `ens` table gives stages 2 and 3's, which is
    the comparison that section 12 asks for. Rows with no distance (absent rows, targets with no
    weather) are left out.
    """  # noqa: D205
    assert table.distance is not None, "the table holds no distances: rebuild it"
    d = table.distance.astype(np.float64)
    d = np.where(np.isfinite(d), d, np.nan)  # calendar rows carry an infinite distance
    if measure is not None:
        d = np.where(table.measure == measure, d, np.nan)
    flat = np.moveaxis(d, 2, 0).reshape(lay.N_LEAD_DAYS, -1)
    count = np.isfinite(flat).sum(axis=1)
    return np.where(count > 0, np.nansum(flat, axis=1) / np.maximum(count, 1), np.nan)


def alpha_by_bucket(*, alpha: torch.Tensor, age_bucket: torch.Tensor) -> np.ndarray:
    """Return the mean total read-out weight of each of the 24 age buckets.

    `alpha` is (B, H, 48) with weights that sum to 1 over the rows H of a (sample, half-hour), or
    to 0 where nothing is visible; `age_bucket` is (B, H). The result is (24,) and sums to the
    share of (sample, half-hour) cells that have a visible row.
    """
    mass = alpha.sum(dim=2) / alpha.shape[2]
    out = torch.zeros(lay.N_BUCKETS, dtype=mass.dtype)
    out.index_add_(0, age_bucket.reshape(-1), mass.reshape(-1))
    return (out / alpha.shape[0]).numpy()


def alpha_by_measure(*, alpha: torch.Tensor, measure: torch.Tensor) -> np.ndarray:
    """Return the mean total read-out weight of each measure code (weather, recency, weekday, fixed,
    calendar, random).
    """  # noqa: D205
    mass = alpha.sum(dim=2) / alpha.shape[2]
    out = torch.zeros(len(lay.MEASURE_NAMES), dtype=mass.dtype)
    out.index_add_(0, measure.reshape(-1), mass.reshape(-1))
    return (out / alpha.shape[0]).numpy()


def extract(*, model: f9.F9Net, inputs: f9.F9Inputs) -> dict[str, torch.Tensor]:
    """Return the attention of every block, `alpha`, and the head's input vector, with the network
    in evaluation mode.
    """  # noqa: D205
    model.eval()
    with torch.no_grad():
        out = model(inputs, return_attention=True)
    assert out.attention is not None and out.target_vector is not None
    return {**out.attention, "alpha": out.alpha, "target_vector": out.target_vector}


def internals_for_case(
    *, model: f9.F9Net, source: fb.F9Source, primary: int, c_idx: int, lead_day: int
) -> dict[str, np.ndarray]:
    """Return the read-out weights, the across-day attention, and the rows of one (Primary, contract
    origin, lead day) case.

    The arrays are `alpha` (H, 48), `across` (48, heads, 1 + H) of the last block (the target
    row's query), `ages` (H,), `measure` (H,), `hist_power` (H, 48) in standardised units, and
    `hist_idx` (H,).
    """  # noqa: D205
    dev = source.device
    one = lambda v: torch.tensor([v], dtype=torch.long, device=dev)  # noqa: E731
    batch = source.assemble(
        primary=one(primary), c_idx=one(c_idx), lead_day=one(lead_day), ensemble=False
    )
    got = extract(model=model, inputs=batch.inputs)
    last = max(int(k.split("_")[1]) for k in got if k.startswith("across_"))
    return {
        "alpha": got["alpha"][0].cpu().numpy(),
        "across": got[f"across_{last}"][0].cpu().numpy(),
        "ages": batch.ages[0].cpu().numpy(),
        "measure": batch.measure[0].cpu().numpy(),
        "hist_power": batch.inputs.hist_power[0].cpu().numpy(),
        "hist_idx": batch.hist_idx[0, :, 0].cpu().numpy(),
    }


def self_test() -> int:
    """Check the four read-outs on a synthetic fold."""
    source = cb.synthetic_source()
    arrays = tab.pool_arrays_from_source(source=source, with_calendar=False)

    days = np.arange(cb.FIRST_ORIGIN_DAY, cb.FIRST_ORIGIN_DAY + cb.N_ORIGIN_DAYS)
    provider = tab.era5_provider(arrays=arrays)
    base = ret.build_table(
        arm="U", arrays=arrays, primaries=cb.TABLE_PRIMARIES, days=days, target_provider=provider
    )
    same = swap_table(
        arm="U",
        arrays=arrays,
        primaries=cb.TABLE_PRIMARIES,
        days=days,
        target_provider=provider,
        ft_weather=lambda *_: None,
    )
    assert np.array_equal(base.ages, same.ages), (
        "a swap with no stored forecast must change nothing"
    )
    shifted = swap_table(
        arm="U", arrays=arrays, primaries=cb.TABLE_PRIMARIES, days=days, target_provider=provider,
        ft_weather=lambda p, d: arrays.weather[p][lay.block_index(contract_idx=lay.contract_idx_of_day(day=d), ages=np.arange(2, 6))] + 3.0,
    )  # fmt: skip
    changed = share_changed(base=base, swapped=shifted)
    assert 0.0 < changed["rows_changed"] < 1.0, changed
    dist = mean_distance_by_lead(table=base)
    assert (
        dist.shape == (lay.N_LEAD_DAYS,)
        and bool(np.isfinite(dist).all())
        and bool((dist >= 0).all())
    )
    assert base.distance is not None
    manual = base.distance[:, :, 3][base.measure[:, :, 3] == lay.MEASURE_WEATHER].astype(np.float64)
    assert abs(dist[3] - manual.mean()) < 1e-9
    lines = [
        f"distance: the mean selected distance by lead day is read from the table ({dist.min():.2f} to {dist.max():.2f} on the synthetic fold)",
        f"swap: no stored forecast changes no row, and a shifted forecast for ages 2 to 5 changes {100 * changed['rows_changed']:.1f}% of rows ({100 * changed['samples_with_a_change']:.1f}% of samples)",
    ]
    fsource = fb.F9Source(base=source, tables={fb.TABLE_ERA5: base, fb.TABLE_ENS: base})
    primary, c_idx, lead = cb.samples(n=24)
    batch = fsource.assemble(primary=primary, c_idx=c_idx, lead_day=lead, ensemble=True)
    net = f9.F9Net(seed=0)
    got = extract(model=net, inputs=batch.inputs)
    alpha = got["alpha"]
    by_bucket = alpha_by_bucket(alpha=alpha, age_bucket=batch.inputs.age_bucket)  # ty: ignore[invalid-argument-type]
    by_measure = alpha_by_measure(alpha=alpha, measure=batch.measure)
    assert by_bucket.shape == (lay.N_BUCKETS,) and by_measure.shape == (len(lay.MEASURE_NAMES),)
    covered = float((alpha.sum(dim=1) > 0).float().mean())
    assert abs(by_bucket.sum() - covered) < 1e-5 and abs(by_measure.sum() - covered) < 1e-5, (
        by_bucket.sum(),
        by_measure.sum(),
        covered,
    )
    assert set(got) >= {"within_0", "across_0", "within_1", "across_1", "alpha", "target_vector"}
    plain = net.eval()(batch.inputs).out
    with torch.no_grad():
        again = net(batch.inputs, return_attention=True).out
    assert torch.equal(plain, again), "recording the attention changed the forecast"
    lines.append(
        "alpha: the weight of the buckets and of the measures each sum to the share of cells with a visible row"
    )
    lines.append(
        "attention: the within-day and across-day attention of both blocks and the head vector are returned, and recording changes no forecast bit"
    )
    print("\n".join(lines))
    print("e00718_readouts: all checks passed")
    return 0


def main() -> int:
    """Run the self-test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["self-test"])
    parser.parse_args()
    return self_test()


if __name__ == "__main__":
    sys.exit(main())

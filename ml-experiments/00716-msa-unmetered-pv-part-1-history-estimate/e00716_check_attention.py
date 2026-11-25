# ruff: noqa: E501, PT018
"""Experiment 716: the checks of `e00716_attention_extract.py` that need no data (CPU, random tensors).

Usage: run.sh e00716_check_attention.py

A randomised `H` network and the synthetic source of `e00716_check_batch.py`: the weights of each
target token sum to 1 over the 15 slots, `return_attention=True` leaves the forecast unchanged, the
weekday part of the summary never exceeds the mean weight, the cases are those of the three allowed
hosts, `max_difference` finds a planted difference, and `extract` refuses an existing output before it
reads any data.
"""

import sys
import tempfile
from pathlib import Path
from typing import Final

import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_attention_extract as ex
import e00716_batch as pvb
import e00716_check_batch as cb
import e00716_model as pvm
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

WEEKDAY_SLOTS: Final = {1: [6, 13], 7: [7, 14]}
"""The history slots whose row has the target's weekday: slot k is age k+1, and lead day L adds L-1 days."""
HOSTS_TO_INDEX: Final = {"D11": 3, "D19": 7, "D21": 12}


def randomised_h(*, seed: int) -> pvm.PvNet:
    """Return an `H` network whose every parameter is random, so that no zero layer hides a difference."""
    torch.manual_seed(seed)
    net = pvm.PvNet(arm="H", seed=seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return net.eval()


def check_attention() -> list[str]:
    """Check the extracted arrays on the synthetic source."""
    source, tables = cb.synthetic_source()
    ps = pvb.PvSource(source=source, tables=tables)
    model = randomised_h(seed=0)
    c_all = (cb.FIRST_DAY + np.arange(cb.N_DAYS - 1)) * 48 + 24
    shortwave = source.dense_weather[:, :, ex.SHORTWAVE_COLUMN].float().numpy()
    cases = ex.pick_cases(shortwave=shortwave, primaries=HOSTS_TO_INDEX, c_all=c_all)
    assert len(cases) == len(ex.CASE_HOSTS) * len(ex.CASE_KINDS)
    assert {v.primary for v in cases.values()} == set(HOSTS_TO_INDEX.values())
    assert all(v.contract_idx in set(c_all.tolist()) for v in cases.values())
    zeros, ones = np.zeros(16), np.ones(16)
    case = next(iter(cases.values()))
    arrays = ex.case_arrays(model=model, source=ps, case=case, target_mean=zeros, target_std=ones)
    assert arrays["across"].shape == (48, 4, ex.N_SLOTS), arrays["across"].shape
    assert np.allclose(arrays["across"].sum(axis=-1), 1.0, atol=1e-4)
    assert arrays["alpha"].shape == (14, 48) and arrays["q"].shape == (48, 11)
    assert arrays["hist_pv"].shape == (14, 48) and arrays["hist_power"].shape == (14, 48)
    plain = ex.run_model(
        model=model,
        source=ps,
        primary=np.array([case.primary]),
        c_idx=np.array([case.contract_idx]),
        lead=0,
    )[0]
    batch = ex.pvt.batch_for(
        source=ps, samples=ex.tr.SampleSet(primary=np.array([case.primary]), c_idx=np.array([case.contract_idx]), is_val=np.zeros(1, bool)),
        index=np.arange(1), lead=np.zeros(1, dtype=np.int64), cut=10**9,
    )  # fmt: skip
    with torch.no_grad():
        assert torch.equal(model(batch.inputs).out, plain.out), (
            "return_attention changed the forecast"
        )
    summary = ex.summary_arrays(model=model, source=ps, c_all=c_all)
    for lead in (1, 7):
        mean = summary[f"summary_lead{lead}/mean_weight"]
        same = summary[f"summary_lead{lead}/same_weekday_weight"]
        assert mean.shape == (16, ex.N_SLOTS) and np.allclose(mean.sum(axis=1), 1.0, atol=1e-4)
        weekday_slots = WEEKDAY_SLOTS[lead]
        other = [k for k in range(ex.N_SLOTS) if k not in weekday_slots]
        assert np.array_equal(same[:, weekday_slots], mean[:, weekday_slots]), lead
        assert (same[:, other] == 0).all(), lead
        assert (summary[f"summary_lead{lead}/count"] == len(c_all)).all()
    return ["attention arrays: shapes, softmax sums, unchanged forecast, summary bounds"]


def planted_shortwave(*, dull_spike: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return a (16, G) shortwave array with planted patterns, and the scored origins (`c_all`).

    Every Primary holds the same pattern on a base of 1.0. Without `dull_spike`, the history rows of
    origin X (index 20) hold 5.0 and its target day 0.05, and origin Y (index 2) has a target day of 9.0.
    With `dull_spike`, a spike of 50.0 lies in the day before the origin of origin W (index 10), which
    no history row reads, and W's target day is 0.01.
    """
    c_all = (cb.FIRST_DAY + np.arange(cb.N_DAYS - 1)) * 48 + 24
    row = np.ones(cb.G)
    if dull_spike:
        w = int(c_all[10])
        row[w - 48 : w] = 50.0
        row[w : w + 48] = 0.01
    else:
        x, y = int(c_all[20]), int(c_all[2])
        row[lay.history_grid_index(contract_idx=np.array([x])).ravel()] = 5.0
        row[y : y + 48] = 9.0
        row[x : x + 48] = 0.05
    return np.tile(row, (16, 1)), c_all


def check_picks() -> list[str]:
    """Check the history window and the picks on planted shortwave patterns."""
    ramp = np.arange(cb.G, dtype=np.float64)
    c_all = (cb.FIRST_DAY + np.arange(cb.N_DAYS - 1)) * 48 + 24
    assert np.allclose(ex.history_means(shortwave=ramp, c_all=c_all), c_all - 384.5)
    shortwave, c_all = planted_shortwave(dull_spike=False)
    cases = ex.pick_cases(shortwave=shortwave, primaries=HOSTS_TO_INDEX, c_all=c_all)
    for host in ex.CASE_HOSTS:
        dull = cases[ex.case_key(host=host, kind="dull_after_sunny")]
        clear = cases[ex.case_key(host=host, kind="clear_day")]
        assert dull.contract_idx == int(c_all[20]), dull
        assert clear.contract_idx == int(c_all[2]), clear
        assert abs(dull.history_mean - 5.0) < 1e-9 and abs(dull.target_mean - 0.05) < 1e-9
    shortwave, c_all = planted_shortwave(dull_spike=True)
    try:
        ex.pick_cases(shortwave=shortwave, primaries=HOSTS_TO_INDEX, c_all=c_all)
    except AssertionError:
        pass
    else:
        msg = "a spike that no history row reads made a dull-after-sunny case"
        raise AssertionError(msg)
    return [
        "picks: history window is the 14 history rows, planted picks found, a spike outside the rows picks nothing"
    ]


def check_max_difference() -> list[str]:
    """Check that `max_difference` reads the matching rows and finds a planted difference."""
    origin_us, step = 1_700_000_000_000_000, 1_800_000_000
    valid_us = origin_us + np.arange(48) * step
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(size=(48, 11)), axis=1).astype(np.float32)
    mean = rng.normal(size=48).astype(np.float32)
    stored = pl.DataFrame(
        {
            "time_series_id": np.full(48, 7, dtype=np.int32),
            "power_fcst_init_time": pl.Series(np.full(48, origin_us)).cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            "valid_time": pl.from_epoch(pl.Series(valid_us), time_unit="us").dt.replace_time_zone("UTC"),
            **{name: q[:, i] for i, name in enumerate(ex.QUANTILE_COLUMNS)},
            "mean": mean,
        }
    )  # fmt: skip

    def compare(*, quantiles: np.ndarray) -> float:
        return ex.max_difference(
            stored=stored,
            time_series_id=7,
            origin_us=origin_us,
            valid_us=valid_us,
            quantiles=quantiles,
            mean=mean,
        )

    assert compare(quantiles=q) == 0.0
    bumped = q.copy()
    bumped[10, 4] += 0.25
    found = compare(quantiles=bumped)
    assert abs(found - 0.25) < 1e-6, found
    return ["max_difference: zero on equal forecasts, 0.25 on a planted 0.25"]


def check_refusal() -> list[str]:
    """Check that `extract` stops on an existing output before it reads any data."""
    with tempfile.TemporaryDirectory(dir=Path.home() / ".cache") as tmp:
        data_dir = Path(tmp)
        path = ex.attention_file(data_dir=data_dir, arm="H", month="2026-05", seed=0)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
        try:
            ex.extract(
                month="2026-05", seed=0, arm_names=["H"], device="cpu", data_dir=data_dir, accept=[]
            )
        except FileExistsError:
            pass
        else:
            msg = "extract did not refuse an existing output"
            raise AssertionError(msg)
        assert arms.root(data_dir=data_dir, arm="H") in path.parents
    return ["extract: refuses to overwrite a stored output"]


def main() -> int:
    """Run every check and print one line per check."""
    torch.set_num_threads(1)
    lines = check_attention() + check_picks() + check_max_difference() + check_refusal()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

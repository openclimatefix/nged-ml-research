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

import e00716_arms as arms
import e00716_attention_extract as ex
import e00716_batch as pvb
import e00716_check_batch as cb
import e00716_model as pvm
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

HOSTS_TO_INDEX = {"D11": 3, "D19": 7, "D21": 12}


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
    assert {v[0] for v in cases.values()} == set(HOSTS_TO_INDEX.values())
    assert all(int(v[1]) in set(c_all.tolist()) for v in cases.values())
    zeros, ones = np.zeros(16), np.ones(16)
    case = next(iter(cases.values()))
    arrays = ex.case_arrays(model=model, source=ps, case=case, target_mean=zeros, target_std=ones)
    assert arrays["across"].shape == (48, 4, ex.N_SLOTS), arrays["across"].shape
    assert np.allclose(arrays["across"].sum(axis=-1), 1.0, atol=1e-4)
    assert arrays["alpha"].shape == (14, 48) and arrays["q"].shape == (48, 11)
    assert arrays["hist_pv"].shape == (14, 48) and arrays["hist_power"].shape == (14, 48)
    plain = ex.run_model(
        model=model, source=ps, primary=np.array([case[0]]), c_idx=np.array([case[1]]), lead=0
    )[0]
    batch = ex.pvt.batch_for(
        source=ps, samples=ex.tr.SampleSet(np.array([case[0]]), np.array([case[1]]), np.zeros(1, bool)),
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
        assert (same <= mean + 1e-9).all() and (same[:, 0] == 0).all()
        assert (summary[f"summary_lead{lead}/count"] == len(c_all)).all()
    return ["attention arrays: shapes, softmax sums, unchanged forecast, summary bounds"]


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
    lines = check_attention() + check_max_difference() + check_refusal()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

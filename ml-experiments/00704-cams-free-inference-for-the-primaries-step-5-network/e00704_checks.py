# ruff: noqa: D103
"""Experiments 703 and 704: checks that the chain scripts run after the pre-training step.

Usage (with `E00704_ARM` set for `proxy-mean`):
- `python e00704_checks.py stats ARM STATS_NPZ [REFERENCE_NPZ]`: the stored B1 standardiser.
  `drop`: mean exactly 0 and std 1e-6 for the two CAMS columns (positions from
  `c562.TABLE_ARMS["B1"]`). `replay`: the file equals REFERENCE_NPZ (the original experiment's
  statistics) exactly. `proxy`: the file exists and holds finite values.
- `E00704_ARM=proxy python e00704_checks.py proxy-mean`: on a sample of pre-training rows of three
  Primaries, the patched `cams_prev_day_mean` equals the unpatched one (CAMS hourly copied onto
  both half-hours of each hour) to float32 tolerance, and the patched `cams_prev_day_csi` is
  finite wherever the unpatched one is.
"""

import sys

import e00437_features as f437  # ty: ignore[unresolved-import]
import e00562_columns as c562  # ty: ignore[unresolved-import]
import numpy as np

CAMS_COLUMNS = ("cams_prev_day_mean", "cams_prev_day_csi")


def stats(*, arm: str, path: str, reference: str | None) -> None:
    found = np.load(path)
    if arm == "drop":
        names = c562.TABLE_ARMS["B1"]
        for column in CAMS_COLUMNS:
            i = names.index(column)
            mean, std = float(found["mean"][i]), float(found["std"][i])
            assert mean == 0.0, (column, mean)
            assert np.isclose(std, 1e-6, rtol=1e-6, atol=0.0), (column, std)
            print(f"{path}: {column} (position {i}) mean {mean} std {std}: ok")
    elif arm == "replay":
        assert reference, "give the original experiment's stats file"
        ref = np.load(reference)
        for key in ("mean", "std"):
            assert np.array_equal(found[key], ref[key]), f"{key} differs from {reference}"
        print(f"{path}: equals {reference} exactly: ok")
    else:
        assert np.isfinite(found["mean"]).all()
        assert np.isfinite(found["std"]).all()
        print(f"{path}: finite: ok")


def proxy_mean() -> None:
    import e00704_arm as arm  # ty: ignore[unresolved-import]

    assert arm.ARM == "proxy", "set E00704_ARM=proxy"
    original = arm._orig_pretrain_extras
    local_date = f437.local_date_codes()
    columns = ("t_mean_3h", *CAMS_COLUMNS)
    rng = np.random.default_rng(0)
    for row in list(f437.prep.primaries().iter_rows(named=True))[:3]:
        dense = f437.dense_arrays(row=row, local_date=local_date)
        n = 20_000
        valid_idx = rng.integers(2_000, len(dense["valid_shortwave"]) - 400, size=n)
        origin_idx = valid_idx - rng.integers(1, 300, size=n)
        valid = f437.GRID_US + valid_idx * f437.STEP_US
        origin = f437.GRID_US + origin_idx * f437.STEP_US
        a = original(
            row=row, dense=dense, columns=columns, valid_us=valid, origin_us=origin, max_steps=704
        )
        b = arm.pretrain_extras(
            row=row, dense=dense, columns=columns, valid_us=valid, origin_us=origin, max_steps=704
        )
        assert np.array_equal(a[:, 0], b[:, 0], equal_nan=True)
        both = np.isfinite(a[:, 1]) & np.isfinite(b[:, 1])
        worst = float(np.abs(a[both, 1] - b[both, 1]).max())
        mismatch = int((np.isfinite(a[:, 1]) != np.isfinite(b[:, 1])).sum())
        csi_missing = int((np.isfinite(a[:, 2]) & ~np.isfinite(b[:, 2])).sum())
        print(
            row["site"],
            "mean: max abs diff",
            worst,
            "W/m2 over",
            int(both.sum()),
            "rows; finite mismatches",
            mismatch,
            "; csi missing where CAMS has it",
            csi_missing,
        )
        assert worst <= 1e-3, "the patched mean differs from CAMS"
        assert mismatch == 0


if __name__ == "__main__":
    {
        "stats": lambda: stats(
            arm=sys.argv[2], path=sys.argv[3], reference=sys.argv[4] if len(sys.argv) > 4 else None
        ),
        "proxy-mean": proxy_mean,
    }[sys.argv[1]]()

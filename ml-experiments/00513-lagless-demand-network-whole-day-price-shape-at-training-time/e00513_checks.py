# ruff: noqa: PT018
# ruff: noqa: E501
"""Checks of the shape features, run before any fit: `uv run --no-sync python e00513_checks.py`."""

import e00513_data as data
import e00513_shape as shape
import numpy as np
import polars as pl


def main() -> None:
    """Assert the feature definitions on synthetic days and report the real price coverage."""
    # A synthetic 48-half-hour day with a known minimum, maximum, and evening peak.
    v = np.linspace(10.0, 57.0, 48)
    v[3] = 1.0
    v[36] = 100.0
    minutes = np.arange(48) * 30
    f = shape.day_features(values=v, local_minutes=minutes)
    assert f.shape == (48, 10)
    assert f[3, 2] == 0.0 and f[10, 2] == 3.5, "hours from the minimum"
    assert f[36, 3] == 0.0 and f[30, 3] == 3.0, "hours to the maximum"
    assert f[:, 4].sum() == 8 and f[:, 5].sum() == 8 and f[3, 4] == 1 and f[36, 5] == 1
    assert f[0, 6] == 1.0 and f[0, 7] == 100.0 and f[0, 8] == 99.0
    assert abs(f[:, 0].min()) < 1e-12 and abs(f[:, 0].max() - 1) < 1e-12
    expected = v[32:38].mean() - v[0:10].mean()
    assert abs(f[0, 9] - expected) < 1e-9, "evening minus overnight"
    assert abs(f[:, 1].mean()) < 1e-9
    # Day lengths on the real grid: 46 and 50 half-hours on the clock-change days.
    n = data.grid_length()
    price = data.price_values()
    _, day_id, missing = shape.grid_features(
        grid_start_us=data.prep422.GRID_START_US, grid_length=n, values=price
    )
    length = np.bincount(day_id)
    print("day lengths:", dict(zip(*np.unique(length, return_counts=True), strict=True)))
    assert set(np.unique(length[~missing])) <= {46, 48, 50}, "unexpected day length"
    # Placebo: never the day itself, no donor used twice, same length and month.
    tables = data.ShapeTables(fit_until_idx=n // 2)
    d = tables.donor
    live = d >= 0
    assert (d[live] != np.nonzero(live)[0]).all(), "placebo donor is the day itself"
    assert len(np.unique(d[live])) == live.sum(), "a donor is used twice"
    assert (length[d[live]] == length[np.nonzero(live)[0]]).all()
    first = tables.day_local.astype("datetime64[D]").astype("datetime64[M]").astype(int) % 12
    assert (first[d[live]] == first[live]).all(), "placebo donor is in another calendar month"
    print(
        f"days {len(missing)}: price-missing {int(missing.sum())}, "
        f"placebo-unmatched {tables.unmatched}, national-missing {int(tables.national_missing.sum())}"
    )
    by_year = (
        pl.DataFrame({"year": tables.day_local.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970, "missing": missing})
        .group_by("year").agg(pl.col("missing").sum()).sort("year")
    )  # fmt: skip
    print(by_year)
    print("shape checks passed")


if __name__ == "__main__":
    main()

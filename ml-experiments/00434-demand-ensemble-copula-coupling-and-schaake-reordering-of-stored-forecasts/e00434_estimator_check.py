"""Compare two estimators of the CRPS on the same stored quantiles, for one run of arm A.

The stored experiments score the CRPS as twice the pinball loss averaged over 11 uneven levels
(`studies.eval_protocol.metrics.quantile_score`). A forecast of 51 values at the levels k/52 scores
the CRPS as the energy form of the sample, or equally as twice the mean pinball loss over 51 evenly
spaced levels. This script prints both on the quantiles of one run (2025-10, the fourth run), so
that the page can say how much of the gap between the two is the estimator.

```bash
DATA_PATH_INTERNAL=... PYTHONPATH=<folders of experiments 87, 377, 381, 434> \
  uv run --no-sync python ml-experiments/00434-*/e00434_estimator_check.py
```
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00434_score as sc
import numpy as np
import polars as pl

series = adapter.series_table()
ctx = sc.cleaned_context(series=series)
grid = sc.Grid(series=series, actuals=ctx["actuals"])
st = sc.stored_arm_a(month="2025-10")
ids = series["time_series_id"].to_list()
order = {s: i for i, s in enumerate(ids)}
init = st["power_fcst_init_time"].unique().sort()[3]
sub = (
    st.filter(pl.col("power_fcst_init_time") == init)
    .with_columns(p=pl.col("time_series_id").replace_strict(order, return_dtype=pl.Int64))
    .sort("p", "valid_time")
)
q = sub.select(sc.LEVELS).to_numpy().reshape(16, 702, 11) * 100
i0 = grid.index(t=init) + 1
y = grid.values[:, i0 : i0 + 702]
ok = ~np.isnan(y)


def pin(q: np.ndarray, lv: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the pinball loss per row and level."""
    e = y[..., None] - q
    return np.maximum(lv * e, (lv - 1) * e)


c11 = 2 * pin(q, sc.QUANTILE_LEVELS, y).mean(-1)
v = sc.fifty_one_values(quantiles=q)
c51 = 2 * pin(v, np.arange(1, 52) / 52, y).mean(-1)
cs = sc.crps_of_sorted(values=v, actual=np.nan_to_num(y))
print(
    "11-level", c11[ok].mean(), "51-level pinball", c51[ok].mean(), "51 sample form", cs[ok].mean()
)

"""Step 0: edge width and constant-versus-proportional shape on the verified transfers.

    DATA_PATH_INTERNAL=... PYTHONPATH=. uv run --no-sync python e00653_step0.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl
from e00653_common import DAY, E580, GRID_START, WORK, Panel, idx_of
from e00653_segment import step_width

VERIFIED_OFFSET: int = (
    idx_of(when=GRID_START) + int((GRID_START - datetime(2019, 9, 1, tzinfo=UTC)).total_seconds() // 1800)
)
"""Half-hours from 2019-09-01 (the verified tables' origin) to 2022-01-01 (this grid's origin)."""
PAIRS: dict[str, tuple[str, str]] = {"D3/D22": ("D3", "D22"), "D5/D16": ("D5", "D16")}


def measure(*, power: np.ndarray, start: int, end: int) -> dict[str, float]:
    """Width of the onset and end steps, and the fit of a constant and a proportional shape."""
    w_on = step_width(xh=power, t=start)
    w_off = step_width(xh=power, t=end)
    days = np.arange(1, 8)
    base = np.full(48, np.nan)
    for hh in range(48):
        before = [power[start - k * DAY + hh - (start % DAY)] for k in days]
        after = [power[end + k * DAY + hh - (end % DAY)] for k in days]
        base[hh] = np.nanmean(before + after)
    t = np.arange(start + 12, end - 12)
    b = base[t % DAY]
    d = power[t] - b
    ok = np.isfinite(d) & np.isfinite(b)
    d, b = d[ok], b[ok]
    c = d.mean()
    alpha = float((d * b).sum() / (b * b).sum())
    return {
        "onset_width_hh": w_on,
        "end_width_hh": w_off,
        "mean_extra_mw": float(c),
        "var_constant": float(np.var(d - c)),
        "alpha": alpha,
        "var_proportional": float(np.var(d - alpha * b)),
        "n": int(ok.sum()),
    }


def main() -> None:
    """Measure both transfers and write the medians."""
    panel = Panel()
    verified = pl.read_parquet(E580 / "events_verified.parquet")
    out: dict[str, dict] = {}
    for name, (a, b) in PAIRS.items():
        row = verified.filter((pl.col("primary_a") == a) & (pl.col("primary_b") == b))
        if row.height == 0:
            row = verified.filter((pl.col("primary_a") == b) & (pl.col("primary_b") == a))
        r = row.to_dicts()[0]
        res = {}
        for site, tag in ((r["primary_a"], "a"), (r["primary_b"], "b")):
            shift = VERIFIED_OFFSET
            start, end = r[f"onset_{tag}_idx"] - shift, r[f"end_{tag}_idx"] - shift
            res[site] = measure(power=panel.power[site], start=start, end=end) | {"start": start, "end": end}
        out[name] = res
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "step0.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

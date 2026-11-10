"""Post hoc diagnostic for check 1: refit the four real edges with constant-MW steps only.

    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 PYTHONPATH=<folder of 660>:. \
        uv run --no-sync python e00686_constant_only.py

Writes `constant_only.json` beside the other tables. The point estimates use the same window and
fit as `e00686_timing.py` with the profile share fixed at 0. There is no bootstrap.
"""

from __future__ import annotations

import json

import e00660_data as data  # ty: ignore[unresolved-import]
import e00686_timing as t
import numpy as np
from e00660_ev1_real_edges import verified_rows  # ty: ignore[unresolved-import]


def main() -> None:
    """Fit and write the four gaps."""
    t.SHARES = (0.0,)
    info = data.series_info()
    rows = verified_rows()
    power = {s: np.load(data.E327 / f"power/power_{info[s]['id']}.npy").astype(float) for s in info}
    fcst = {s: data.load_forecast(site=s) for s in info}
    out = []
    for name, ev in rows.items():
        d, r = t.DONOR[name], t.RECIPIENT[name]
        for edge, k in (("onset", 0), ("end", 1)):
            c = (ev[d][k] + ev[r][k]) // 2
            yd, _ = t.window(site=d, pair_centre=c, forecasts=fcst, power=power)
            yr, _ = t.window(site=r, pair_centre=c, forecasts=fcst, power=power)
            fd = t.donor_profile(f=fcst[d][c - t.HALF : c + t.HALF])
            s, _, _ = t.fit_pair(yd=yd, yr=yr, fd=fd, reverse=edge == "end")
            stat = t.lr_stat(s=s, n_total=2 * len(yd))
            out.append({"pair": name, "edge": edge, "gap_hh": s["gap_hh"], "statistic": stat})
    (t.WORK / "constant_only.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()

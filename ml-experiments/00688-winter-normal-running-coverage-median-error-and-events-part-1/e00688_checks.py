"""Error of the median forecast by stage, and the size of the oracle shift (CPU only).

Run it from the repository root with `DATA_PATH_INTERNAL` set to the production data folder and
`PYTHONPATH` set to the folders of experiments 668, 683, and 688.

The stored files hold quantiles only, so no mean forecast exists; the median's absolute and signed
error (percent of capacity, the two winter months, arm 2) stand beside the coverage contrasts.
"""

import e00688_diagnose as dg  # ty: ignore[unresolved-import]
import numpy as np


def main() -> None:
    """Print the median's error by stage, and the shift sizes and the rows left out."""
    rows = dg.load_rows()
    spans = dg.event_spans(kind="primary")
    evt = dg.event_mask(spans=spans, series=rows["series"], vidx=rows["vidx"])
    winter = np.isin(rows["month"], dg.WINTER)
    for stage in ("none", "events", "shift", "events+shift"):
        q, use = dg.stage_quantiles(rows=rows, which="scaled", stage=stage, evt=evt)
        sel = winter & use
        err = rows["y"][sel] - q[sel, 5]
        print(stage, f"MAE {np.abs(err).mean():.2f}", f"mean error {err.mean():.2f}", sel.sum())
    q, _ = dg.stage_quantiles(rows=rows, which="scaled", stage="events+shift", evt=evt)
    shift = q[:, 5] - rows["scaled"][:, 5]
    for month in dg.WINTER:
        sel = rows["month"] == month
        print(
            month,
            f"mean |shift| {np.abs(shift[sel]).mean():.2f}",
            f"max {np.abs(shift[sel]).max():.2f}",
        )
    print("rows with a valid time in the final-test window, left out:", int((~rows["inwin"]).sum()))


if __name__ == "__main__":
    main()

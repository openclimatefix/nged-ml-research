# ruff: noqa: D103
"""Experiment 693: count the injected events per scored month on the deciding bank of bank F."""

from datetime import UTC, datetime, timedelta
from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00693_headdata as hd
import numpy as np

FOLD_MONTHS: Final = {
    "2025-11": ("2025-11", "2025-12", "2026-01"),
    "2026-02": ("2026-02", "2026-03", "2026-04"),
    "2026-05": ("2026-05", "2026-06"),
}


def main() -> None:
    lines = ["month, events over 48 passes, events per pass, runs per Primary-year implied"]
    for fold, months in FOLD_MONTHS.items():
        counts = dict.fromkeys(months, 0)
        for r in range(hd.DEC_PASSES):
            w = np.load(
                hd.ROOT
                / f"P7_D50_{fold}_headbank"
                / "work"
                / "headbank"
                / f"dec_r{r:02d}_world.npz"
            )
            for spans in hd.runs_of(mask=w["own_active"]).values():
                for onset, _end in spans:
                    month = (k578.GRID_START + timedelta(minutes=30 * onset)).strftime("%Y-%m")
                    if month in counts:
                        counts[month] += 1
        for m, n in counts.items():
            lines.append(
                f"{m}: {n} events over {hd.DEC_PASSES} passes, {n / hd.DEC_PASSES:.1f} per pass"
            )
    out = hd.ROOT / "scores" / "event_counts.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _ = (UTC, datetime)


if __name__ == "__main__":
    main()

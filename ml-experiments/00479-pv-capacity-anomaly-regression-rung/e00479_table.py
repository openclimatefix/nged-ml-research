"""Write `nged_table.md` for experiment 479: capacity at every Primary, none validated."""

import textwrap
from typing import Final

import e00479_common as common
import polars as pl
from e00479_charts import DOCS_DIR

STATUS: Final[dict[str, str]] = {
    "D5": "export limit: the clip-free solar term does not represent it",
    "D12": "reverse flow: the estimate is negative and is not a capacity",
    "D15": "generator case: not run",
    "D18": "history only from 2025-03-01; report-only",
}
MVA_STATUS: Final[str] = "MVA-only: apparent power, reactive power assumed zero"


def main() -> None:
    """Write the table in the docs folder."""
    cap = pl.read_parquet(common.WORK_DIR / "capacity_table.parquet").filter(
        pl.col("prior") == "none", pl.col("cloud")
    )
    e474 = pl.read_parquet(common.E474_WORK_DIR / "nged_table.parquet").filter(
        pl.col("season_year") == common.LATEST_SEASON_YEAR
    )
    rows = []
    for site in sorted((*common.ALL_SITES, "D15"), key=lambda s: int(s[1:])):
        mine = cap.filter(pl.col("site") == site)
        old = e474.filter(pl.col("site") == site)
        if mine.is_empty():
            rows.append(f"| {site} | {STATUS[site]} | - | - | - |")
            continue
        r = mine.row(0, named=True)
        status = STATUS.get(
            site, MVA_STATUS if site in common.MVA_ONLY_SITES else "signed MW reading"
        )
        old_text = f"{old['ac_p50'][0]:.1f}" if len(old) else "-"
        rows.append(
            f"| {site} | {status}; unvalidated | {r['p10']:.1f} / {r['p50']:.1f} / {r['p90']:.1f} "
            f"| {r['dc_1.25_p10']:.1f} / {r['dc_1.25_p50']:.1f} / "
            f"{r['dc_1.25_p90']:.1f} | {old_text} |"
        )
    intro = (
        "Season-year July 2025 to June 2026, in megawatts. Estimator: anomaly regression with the "
        "two cloud terms. No row is validated, because the estimator recovers only 0.73 to 0.83 of "
        "an injected farm (see the experiment page). The intervals are the 10th to 90th percentile "
        "of a month-block bootstrap and do not include that bias, so the true capacity is likely "
        "higher."
    )
    header = (
        "| Primary | Status | AC capacity, MW (10th / 50th / 90th) | DC capacity at 1.25, MW "
        "(10th / 50th / 90th) | Experiment 474 AC median, MW |\n|---|---|---|---|---|\n"
    )
    title = "# Unmetered solar capacity at each Primary, anomaly regression (NGED table)"
    text = (
        "\n\n".join([title, "\n".join(textwrap.wrap(intro, 100)), header + "\n".join(rows)]) + "\n"
    )
    (DOCS_DIR / "nged_table.md").write_text(text)


if __name__ == "__main__":
    main()

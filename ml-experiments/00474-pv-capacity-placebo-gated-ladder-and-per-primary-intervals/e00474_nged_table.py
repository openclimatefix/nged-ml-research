"""Build the readable per-Primary table for NGED from experiment 474's stage 1 outputs.

Writes `work/nged_table.parquet` (private, all season-years) and `nged_table.md` into the docs
folder (latest season-year, July 2025 to June 2026). Substations are named; no metered generator
appears.
"""

import argparse

import e00474_charts as charts
import e00474_common as common
import numpy as np
import polars as pl

D12_STATUS = (
    "reverse flow; fit through the meter (physical fleet shape), bootstrap of about 30 resamples"
)


def d12_row(*, ladder: pl.DataFrame) -> dict | None:
    """Return D12's row from the meter-fit bootstrap, divided by the tilt-35 recoveries."""
    path = common.WORK_DIR / "d12_bootstrap.parquet"
    if not path.exists():
        return None
    boot = pl.read_parquet(path)
    ratios = ladder.filter(
        pl.col("rung") == "C1b_physics_tilt35",
        pl.col("kind") == "real",
        pl.col("stat") == "ratio",
        pl.col("b") == 0,
    )["value"].to_numpy()
    rng = np.random.default_rng(5)
    ac = np.array(
        [rng.choice(boot["capacity_mw"].to_numpy()) / rng.choice(ratios) for _ in range(400)]
    )
    ratio = rng.triangular(common.DC_AC_LOW, common.DC_AC_CENTRE, common.DC_AC_HIGH, 400)
    dc = ac * ratio
    q = lambda v, p: float(np.percentile(v, p))  # noqa: E731
    return {
        "site": "D12",
        "season_year": 2024,
        "ac_p10": q(ac, 10),
        "ac_p50": q(ac, 50),
        "ac_p90": q(ac, 90),
        "dc_p10": q(dc, 10),
        "dc_p50": q(dc, 50),
        "dc_p90": q(dc, 90),
        "pinned_mw": None,
        "rung": "meter fit, C1b_physics_tilt35",
        "status": D12_STATUS,
        "coverage_easy": 0.75,
        "coverage_cloud_load": 0.0,
        "placebo_p95_max": 0.246,
    }


def main() -> None:
    """Write the table files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="table_R4_pooled_table_era5_cams_mean_cloudterm")
    arguments = parser.parse_args()
    table = pl.read_parquet(common.WORK_DIR / f"{arguments.table}.parquet")
    ladder = pl.concat(
        [pl.read_parquet(p) for p in sorted(common.WORK_DIR.glob("ladder_D*.parquet"))]
    )
    d12 = d12_row(ladder=ladder)
    if d12 is not None:
        table = pl.concat([table, pl.DataFrame([d12])], how="diagonal")
    table.write_parquet(common.WORK_DIR / "nged_table.parquet")
    latest = table.filter(pl.col("season_year") == 2025)
    d12_latest = table.filter(pl.col("site") == "D12")
    latest = pl.concat([latest, d12_latest], how="diagonal").unique(subset="site", keep="first")
    lines = [
        "# Unmetered solar capacity at each Primary (NGED table)",
        "",
        "Season-year July 2025 to June 2026, in megawatts. Capacities come from the",
        "pooled solar-shape model driven by the mean of ERA5 and CAMS irradiance, with two",
        "cloud terms; D12 uses the fit through the meter.",
        "",
        (
            "| Primary | Status | AC capacity, MW (10th / 50th / 90th) "
            "| DC capacity, MW (10th / 50th / 90th) | Confidence |"
        ),
        "|---|---|---|---|---|",
    ]
    for row in latest.sort(pl.col("site").str.slice(1).cast(pl.Int64)).iter_rows(named=True):
        status = row["status"] or "signed MW reading"
        note = "" if row["site"] != "D12" else "recovery tests used the physical fleet shape; "
        lines.append(
            f"| {row['site']} | {status} "
            f"| {row['ac_p10']:.1f} / {row['ac_p50']:.1f} / {row['ac_p90']:.1f} "
            f"| {row['dc_p10']:.1f} / {row['dc_p50']:.1f} / {row['dc_p90']:.1f} "
            f"| {note}coverage {row['coverage_easy']:.0%} / {row['coverage_cloud_load']:.0%}, "
            f"placebo {row['placebo_p95_max']:.2f} |"
        )
    lines += [
        "",
        "**Confidence column.** Coverage is the share of injected metered-farm capacities that the",
        "80% interval contained in tests, first without and then with a cloud-following load.",
        "Placebo is the 95th percentile of the share of a fake farm (a day-of-year profile with",
        "another day's anomaly) that the fit found. The placebo test fails when it exceeds 0.20, so",  # noqa: E501
        "it failed on every row. Status terms: MVA-only means the reading is apparent power with no",  # noqa: E501
        "sign; the other Primaries have signed readings. D12 shows its season-year 2024 value",
        "(July 2024 to June 2025), the latest the fit through the meter was run for.",
    ]
    (charts.DOCS_DIR / "nged_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

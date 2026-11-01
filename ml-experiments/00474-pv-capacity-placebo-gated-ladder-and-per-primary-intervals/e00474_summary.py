"""Summarise experiment 474's ladder: raw recovery, placebo gate, and interval coverage by rung.

Coverage of the injected capacity at the nominal 80% level is read two ways: from the month-block
bootstrap alone, and from the bootstrap widened by the spread of recoveries of the other Primaries
(leave-one-Primary-out). The cloud-following load case uses the same recoveries.
"""

import e00474_common as common
import numpy as np
import polars as pl


def main() -> None:
    """Print the table and write `work/rung_summary.parquet`."""
    d = pl.concat([pl.read_parquet(p) for p in sorted(common.WORK_DIR.glob("ladder_D*.parquet"))])
    point = d.filter(pl.col("b") == 0)
    rows = []
    for rung in sorted(d["rung"].unique().to_list()):
        r = point.filter(pl.col("rung") == rung, pl.col("stat") == "recovery")
        real = r.filter(pl.col("kind") == "real")["value"].to_numpy()
        placebo = r.filter(pl.col("kind") == "placebo")
        p95_by_site = placebo.group_by("site").agg(pl.col("value").quantile(0.95))["value"]
        profile = r.filter(pl.col("kind") == "profile")["value"].median()
        ratios = d.filter(
            pl.col("rung") == rung, pl.col("kind") == "real", pl.col("stat") == "ratio"
        )
        cells = {}
        for (site, farm, mw), g in ratios.group_by("site", "farm", "mw"):
            cells[(site, farm, mw)] = g.sort("b")["value"].to_numpy()
        cloud_cells = {}
        for (site, farm, mw), g in d.filter(
            pl.col("rung") == rung, pl.col("kind") == "cloud", pl.col("stat") == "ratio"
        ).group_by("site", "farm", "mw"):
            cloud_cells[(site, farm, mw)] = g.sort("b")["value"].to_numpy()

        def coverage(items: dict) -> tuple[float, float]:
            boot, wide = [], []
            for (site, farm, mw), v in items.items():
                others = np.array([c[0] for k, c in cells.items() if k[0] != site])
                if not len(others):
                    continue
                lo, hi = np.percentile(v[1:] / np.median(others), [10, 90])
                boot.append(lo <= 1 <= hi)
                lo, hi = np.percentile(np.concatenate([v[1:] / o for o in others]), [10, 90])
                wide.append(lo <= 1 <= hi)
            return float(np.mean(boot)), float(np.mean(wide))

        boot, wide = coverage(cells)
        cboot, cwide = coverage(cloud_cells) if cloud_cells else (float("nan"),) * 2
        rows.append(
            {
                "rung": rung,
                "n_real": len(real),
                "recovery_median": float(np.median(real)),
                "recovery_min": float(real.min()),
                "placebo_p95_median_over_primaries": float(p95_by_site.median()),
                "placebo_p95_max": float(p95_by_site.max()),
                "profile_only": float(profile),
                "passes_recovery": bool(np.median(real) >= common.PASS_RECOVERY),
                "passes_placebo": bool(p95_by_site.max() < common.GATE_LIMIT),
                "coverage_boot": boot,
                "coverage_boot_plus_recovery_spread": wide,
                "cloud_cov_boot": cboot,
                "cloud_cov_boot_plus_spread": cwide,
            }
        )
    out = pl.DataFrame(rows)
    with pl.Config(tbl_rows=30, tbl_cols=20, fmt_str_lengths=45, tbl_width_chars=250):
        print(out)
    out.write_parquet(common.WORK_DIR / "rung_summary.parquet")


if __name__ == "__main__":
    main()

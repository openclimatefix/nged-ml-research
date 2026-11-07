# ruff: noqa: E501
"""Experiment 634, part 1: re-score the stored donor-day probe recoveries in injected-energy units (CPU).

The probe returns a capacity in MW from a regression on the physics unit curve `u`. A recovery in
MW is `delta_mw / mw`. The injected energy over the probe cells is `mw * sum(farm)`, and the energy
the probe implies is `delta_mw * sum(u)`, with `u` driven by the Primary's own weather at the
probe cells. So `recovery_energy = recovery_mw * sum(u) / sum(farm)`. The sums run over the probe
cells (12 dates, half-hour ends from 10:00 to 15:00 UTC) where the farm has no gap. A second ratio
over all daylight pretraining rows of the six Primaries is the cross-check of experiment 624's audit.

Usage: python e00634_energy.py
"""

import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00623_common as k  # ty: ignore[unresolved-import]
import e00623_probe as p623  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00634-pv-probe-capacity-step-world-and-energy-recovery"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
SOURCES = {
    "621": ML_EXPERIMENTS_DATA_DIR / k.PLAIN_NAME / "work" / "recoveries.csv",
    "623": ML_EXPERIMENTS_DATA_DIR / k.NAME / "work" / "recoveries.csv",
    "624": ML_EXPERIMENTS_DATA_DIR / "00624-pv-probe-contrastive-injection-arms" / "work" / "recoveries.csv",
}
"""Stored recoveries by experiment; experiment 621's file has no `arm` column (plain network)."""


class Std:
    """The frame's standardiser, in the form `raw_weather` reads."""

    def __init__(self, *, mean: np.ndarray, std: np.ndarray) -> None:
        """Hold the first seven weather columns' mean and standard deviation."""
        self.mean, self.std = mean, std


def energy_ratios() -> pl.DataFrame:
    """Return, per cutoff, Primary and farm, the farm-over-unit-curve energy ratio on the probe cells and on all rows."""
    out = []
    for cutoff in k.CUTOFFS:
        frame = p623.load_frame(cutoff=cutoff)
        std = Std(mean=frame["std_mean"], std=frame["std_std"])
        for site, i in k.six_indices().items():
            probe = p623.cells_and_donors(frame=frame, primary=i)
            cells = np.unique(probe["base"])
            x_all, t_all = probe["x"], probe["t_us"]
            sel = frame["primary"] == i
            x_rows, t_rows = frame["x"][sel], k.time_us(target_idx=frame["target_idx"][sel])
            # donor basis: the regression's own rows, the donor's clearness rescaled to the base sun
            b, d = probe["base"], probe["donor"]
            ghi_d, temp_d = k.raw_weather(x=x_all[d], standardiser=std)
            kt = np.clip(ghi_d / np.maximum(k.haurwitz_w_m2(cos_zenith=x_all[d, 14].astype(np.float64)), 1.0), 0.0, p623.KT_MAX)
            ghi_in = kt * k.haurwitz_w_m2(cos_zenith=x_all[b, 14].astype(np.float64))
            u_donor = k.unit_curve(ghi=ghi_in, x=x_all[b], times_us=t_all[b], temp=temp_d)
            for label, xx, tt in (("cells", x_all[cells], t_all[cells]), ("rows", x_rows, t_rows), ("donors", x_all[d], t_all[d])):
                if label == "donors":
                    u = u_donor
                else:
                    ghi, temp = k.raw_weather(x=xx, standardiser=std)
                    u = k.unit_curve(ghi=ghi, x=xx, times_us=tt, temp=temp)
                for farm in k.FARMS:
                    f = p9.series_at(times_us=tt, series=p5.farm_norm_series(farm=farm), shuffle_salt=None)
                    ok = np.isfinite(f) & (u > 0.02)
                    out.append(
                        {
                            "cutoff": cutoff, "site": site, "farm": farm, "basis": label,
                            "ratio": float(f[ok].sum() / u[ok].sum()), "n": int(ok.sum()),
                        }
                    )  # fmt: skip
    return pl.DataFrame(out)


def main() -> None:
    """Write `work/energy_ratios.csv` and `work/energy_recoveries.csv`, and print the summary tables."""
    WORK.mkdir(parents=True, exist_ok=True)
    ratios = energy_ratios()
    ratios.write_csv(WORK / "energy_ratios.csv")
    cells = ratios.filter(pl.col("basis") == "cells").select("cutoff", "site", "farm", pl.col("ratio").alias("ratio_cells"))
    rows = ratios.filter(pl.col("basis") == "rows").select("cutoff", "site", "farm", pl.col("ratio").alias("ratio_rows"))
    donors = ratios.filter(pl.col("basis") == "donors").select("cutoff", "site", "farm", pl.col("ratio").alias("ratio_donors"))
    parts = []
    for exp, path in SOURCES.items():
        df = pl.read_csv(path)
        if "arm" not in df.columns:
            df = df.with_columns(arm=pl.lit("plain"))
        df = df.filter((pl.col("model") == "plain") & (pl.col("world") == "farm")).with_columns(
            experiment=pl.lit(exp),
            j=pl.col("site").replace_strict({s: j for j, s in enumerate(k.SIX)}),
        )
        df = df.with_columns(
            farm=(pl.col("j") + pl.col("rotation")).map_elements(lambda v: k.FARMS[v % len(k.FARMS)], return_dtype=pl.String)
        )
        parts.append(df)
    rec = pl.concat(parts, how="diagonal_relaxed").join(cells, on=["cutoff", "site", "farm"]).join(rows, on=["cutoff", "site", "farm"]).join(donors, on=["cutoff", "site", "farm"])
    rec = rec.with_columns(
        recovery_mw=pl.col("delta_mw") / pl.col("mw"),
        recovery_energy_cells=pl.col("delta_mw") / pl.col("mw") / pl.col("ratio_cells"),
        recovery_energy_rows=pl.col("delta_mw") / pl.col("mw") / pl.col("ratio_rows"),
        recovery_energy_donors=pl.col("delta_mw") / pl.col("mw") / pl.col("ratio_donors"),
    )
    assert (rec["recovery_mw"] - rec["recovery"]).abs().max() < 1e-6, "recovery column is not delta_mw / mw"
    rec = rec.filter(pl.col("arm") != "a")  # arm (a) of experiment 624 reads zero everywhere; its ratio is not informative
    rec.write_csv(WORK / "energy_recoveries.csv")
    cols = ["recovery_mw", "recovery_energy_cells", "recovery_energy_rows", "recovery_energy_donors"]
    agg = [pl.col(c).median().alias(f"{c}_median") for c in cols]
    pl.Config.set_tbl_rows(60)
    print("ratio farm/unit-curve (cells, rows, donors), by site:")
    print(ratios.group_by("basis", "site").agg(pl.col("ratio").min().alias("min"), pl.col("ratio").max().alias("max")).sort("basis", "site"))
    print("by experiment and arm, median over everything:")
    print(rec.group_by("experiment", "arm").agg(*agg).sort("experiment", "arm"))
    print("by site:")
    print(rec.group_by("arm", "site").agg(*agg).sort("arm", "site"))
    print("by farm:")
    print(rec.group_by("arm", "farm").agg(*agg).sort("arm", "farm"))
    print("coefficient of variation of the site medians, and of the farm medians:")
    for key in ("site", "farm"):
        per = rec.group_by("arm", key).agg(*agg)
        print(per.group_by("arm").agg(*[(pl.col(f"{c}_median").std() / pl.col(f"{c}_median").mean()).alias(f"cv_{key}_{c}") for c in cols]))


if __name__ == "__main__":
    main()

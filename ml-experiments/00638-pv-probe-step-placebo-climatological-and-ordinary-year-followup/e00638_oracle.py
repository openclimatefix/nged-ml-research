# ruff: noqa: E501
"""Experiment 638: the linear oracle, the read a regression of the injected megawatts would give.

For each step and shape, regress the injected megawatts of the probe's donor rows (the real farm's
or the climatological shape, after the start day) on the physics unit curve and the temperature, with
the probe's own cells and estimator. The raw result per injected megawatt is what a perfect linear
learner that cannot tell the date would read. It is compared with the networks' raw reads.

Usage: python e00638_oracle.py
"""

import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00634_probe as p  # ty: ignore[unresolved-import]
import e00638_common as k
import numpy as np
import polars as pl


def main() -> None:
    """Write `work/oracle.csv` and print the oracle beside the network reads."""
    pl.Config.set_tbl_rows(100)
    six = k634.six_indices()
    series = {f: p5.farm_norm_series(farm=f) for f in k634.FARMS}
    clim = {f: k.climatological_profile(series=series[f]) for f in k634.FARMS}
    out = []
    for cutoff in k.CUTOFFS:
        frame = p.load_frame(cutoff=cutoff)
        for j, (site, i) in enumerate(six.items()):
            pr = p.cells_and_donors(frame=frame, primary=i)
            x, b, d, cell = pr["x"], pr["base"], pr["donor"], pr["cell"]
            sm, ss = frame["std_mean"], frame["std_std"]
            ghi_d = np.clip(x[d, 0].astype(float) * ss[0] + sm[0], 0, None)
            temp = x[d, 2].astype(float) * ss[2] + sm[2]
            kt = np.clip(
                ghi_d / np.maximum(k634.haurwitz_w_m2(cos_zenith=x[d, 14].astype(float)), 1),
                0,
                p.KT_MAX,
            )
            ghi_in = kt * k634.haurwitz_w_m2(cos_zenith=x[b, 14].astype(float))
            u = k634.unit_curve(ghi=ghi_in, x=x[b], times_us=pr["t_us"][b], temp=temp)
            td, dday = pr["t_us"][d], pr["days"][d]
            farm = k634.FARMS[j % 4]
            shapes = {
                "real": p9.series_at(times_us=td, series=series[farm], shuffle_salt=None),
                "climatological": p9.series_at(times_us=td, series=clim[farm], shuffle_salt=None),
            }
            for key, start in k.STEP_STARTS.items():
                after = (dday >= int(np.datetime64(start, "D").astype(np.int64))).astype(float)
                for kind, shape in shapes.items():
                    ok = np.isfinite(shape)
                    y = np.where(ok, -4.0 * np.nan_to_num(shape) * after, 0.0)
                    raw = p.fit(y=y, cell=cell, regressors=[u, temp], w=ok.astype(float)) / 4.0
                    out.append(
                        {
                            "cutoff": cutoff,
                            "site": site,
                            "key": key,
                            "kind": kind,
                            "oracle_raw": raw,
                        }
                    )
    df = pl.DataFrame(out)
    df.write_csv(k.WORK / "oracle.csv")
    oracle = df.group_by("key", "kind", "cutoff").agg(oracle_raw=pl.col("oracle_raw").median())
    reads = pl.read_csv(k.WORK / "step_reads_shift0.csv")
    names = {k.real_condition(key=key): (key, "real") for key in k.STEP_STARTS}
    names |= {f"climstep_{key}_mw4": (key, "climatological") for key in k.STEP_STARTS}
    reads = reads.filter(pl.col("condition").is_in(list(names))).with_columns(
        key=pl.col("condition").replace_strict({c: v[0] for c, v in names.items()}),
        kind=pl.col("condition").replace_strict({c: v[1] for c, v in names.items()}),
    )
    net = reads.group_by("key", "kind", "cutoff", "site").agg(v=pl.col("recovery_raw").median())
    net = net.group_by("key", "kind", "cutoff").agg(network_raw=pl.col("v").median())
    print(oracle.join(net, on=["key", "kind", "cutoff"], how="left").sort("key", "kind", "cutoff"))


if __name__ == "__main__":
    main()

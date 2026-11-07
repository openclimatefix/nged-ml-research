# ruff: noqa: E501, ANN001
"""Experiment 635: recoveries per injected MW and per injected energy, placebos, CRPS, and the pass bars.

Arm a is experiment 621's plain probe at the 2026-06 cutoff (its stored probe results). Arms b and c
are this experiment's. Every injected network is paired with the uninjected twin that has the same
arm, seed, Primary and input column. Writes `work/recoveries.csv` and `work/summary.json`.
"""

import json

import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00621_common as k  # ty: ignore[unresolved-import]
import e00621_probe as p621  # ty: ignore[unresolved-import]
import e00635_common as k5
import numpy as np
import polars as pl

PROBES_621 = k.WORK / "probes" / k5.CUTOFF
WORLDS = {
    "farm": "farm",
    "phys": "phys",
    "profile": "profile",
    "shuffled": "shuffled",
    "cloud": "cloud",
}


def load() -> pl.DataFrame:
    """Return every probe result with arm, condition, injected MW and world."""
    frames = []
    for f in sorted(PROBES_621.glob("*.parquet")):
        name = f.stem
        d = pl.read_parquet(f).filter(pl.col("model") == "plain")
        d = d.with_columns(
            arm=pl.lit("a"), condition=pl.lit("control_r0" if name == "control" else name)
        )
        frames.append(d.select("arm", "condition", "seed", "site", "capacity_mw", "boot"))
        if name == "control":  # arm a has one input column, so one twin serves both rotations
            frames.append(frames[-1].with_columns(condition=pl.lit("control_r1")))
    for arm in k5.ARMS:
        for f in sorted((k5.WORK / "probes" / arm).glob("*.parquet")):
            d = pl.read_parquet(f).with_columns(arm=pl.lit(arm))
            frames.append(d.select("arm", "condition", "seed", "site", "capacity_mw", "boot"))
    df = pl.concat(frames)
    meta = {n: k5.split(name=n)[0] for n in df["condition"].unique().to_list()}
    return df.with_columns(
        world=pl.col("condition").replace_strict({n: c.kind for n, c in meta.items()}),
        mw=pl.col("condition").replace_strict({n: c.mw for n, c in meta.items()}),
        rotation=pl.col("condition").map_elements(
            lambda n: k5.split(name=n)[1], return_dtype=pl.Int64
        ),
        mshuf=pl.col("condition").str.ends_with("_mshuf"),
    )


def energy_factor(*, frame, site: str, farm: str, kind: str, shuffled: bool = False) -> float:
    """Return (sum of the unit curve over the probed base half-hours) over (sum of the farm's shape there).

    Recovery per injected energy is the recovery per injected MW times this factor, because the probe's
    capacity multiplies the unit curve while the injection multiplies the farm's shape.
    """
    i = k.six_indices()[site]
    pr = p621.cells_and_donors(frame=frame, primary=i)
    b = np.unique(pr["base"])
    x, t = pr["x"], pr["t_us"]
    s_mean, s_std = frame["std_mean"], frame["std_std"]
    ghi = np.clip(x[b, 0].astype(np.float64) * s_std[0] + s_mean[0], 0.0, None)
    temp = x[b, 2].astype(np.float64) * s_std[2] + s_mean[2]
    u = k.unit_curve(ghi=ghi, x=x[b], times_us=t[b], temp=temp)
    if kind == "phys":
        return 1.0
    series = p5.farm_norm_series(farm=farm)
    if kind in {"profile", "shuffled"}:
        series = p9.split_profile_anomaly(series=series)[0]
    shape = np.nan_to_num(p9.series_at(times_us=t[b], series=series, shuffle_salt=None))
    return float(u.sum() / max(shape.sum(), 1e-9))


def recoveries(df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per injected network with its recovery per injected MW."""
    twin = df.filter(pl.col("world") == "control").select(
        "arm", "seed", "site", twin_name=pl.col("condition"), twin_mw=pl.col("capacity_mw")
    )
    inj = df.filter(pl.col("world") != "control").with_columns(
        twin_name=pl.col("condition").map_elements(
            lambda n: k5.twin(name=n), return_dtype=pl.String
        )
    )
    j = inj.join(twin, on=["arm", "seed", "site", "twin_name"])
    return j.with_columns(recovery=(pl.col("capacity_mw") - pl.col("twin_mw")) / pl.col("mw"))


def main() -> None:
    """Print the recoveries and write the CSV and JSON."""
    rec = recoveries(load())
    frame = p621.load_frame(cutoff=k5.CUTOFF)
    out = []
    for r in rec.iter_rows(named=True):
        j = k.SIX.index(r["site"])
        farm = k.FARMS[(j + r["rotation"]) % len(k.FARMS)]
        f = (
            energy_factor(frame=frame, site=r["site"], farm=farm, kind=r["world"])
            if r["world"] != "cloud"
            else float("nan")
        )
        out.append(r["recovery"] * f)
    rec = rec.with_columns(recovery_energy=pl.Series(out)).drop("boot")
    rec.write_csv(k5.WORK / "recoveries.csv")
    summary: dict = {}
    for (arm, world, mw, mshuf), g in rec.group_by(
        ["arm", "world", "mw", "mshuf"], maintain_order=True
    ):
        v = g["recovery"].to_numpy()
        e = g["recovery_energy"].to_numpy()
        summary[f"{arm}|{world}|{mw}|{'mshuf' if mshuf else ''}"] = {
            "n": len(v), "median_per_mw": float(np.median(v)), "p95_per_mw": float(np.percentile(v, 95)),
            "median_per_energy": float(np.nanmedian(e)),
            "by_site": {s: float(np.median(g.filter(pl.col("site") == s)["recovery"])) for s in k.SIX},
        }  # fmt: skip
    (k5.WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    for key, v in summary.items():
        print(
            key,
            {
                a: (round(b, 3) if isinstance(b, float) else b)
                for a, b in v.items()
                if a != "by_site"
            },
        )


if __name__ == "__main__":
    main()

# ruff: noqa: E501, C408, PLR0912
"""Experiment 509: paired contrasts (injected minus injection-off) from the cube, with cluster bootstraps.

Reads `work/cube.npz` (written by `e00509_paired.py`) and writes `work/contrasts.csv` and
`work/pit.csv`. A cluster is one (month, donor) pair; the bootstrap resamples the 15 clusters with
replacement, B times, and recomputes every statistic from the resampled sums (so the widening factor
is re-solved within each resample). The interval is the percentile interval at the Bonferroni level
for 48 contrasts (alpha 0.05 / 48, two-sided). The 12 to 36 hour band and all non-confirmatory
contrasts carry the same interval but are descriptive.

```bash
export DATA_PATH_INTERNAL=<private data folder>
uv run --no-sync python ml-experiments/00509-*/e00509_contrasts.py
```
"""

from typing import Final

import e00509_paired as p  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

B: Final[int] = 20000
SEED: Final[int] = 509
ALPHA: Final[float] = 0.05 / 48
SIGNED_DONORS: Final[tuple[str, ...]] = ()  # filled from experiment 482's summary below
MEASURES: Final[tuple[str, ...]] = (
    "cov_q01", "cov_q05", "cov_q95", "cov_q99", "cov_c90", "width90", "pin05", "pin95",
    "factor_q95", "factor_q05", "pit_deviation",
)  # fmt: skip


def donor_groups() -> dict[str, str]:
    """Return the metering group of each donor from experiment 482's summary."""
    s = pl.read_parquet(p.W482 / "summary" / "2025-10.parquet").filter(pl.col("role") == "donor")
    return {d: g for d, g in s.select("donor", "group").unique().iter_rows() if d in p.DONORS}


def factor(*, hist: np.ndarray) -> np.ndarray:
    """Return the 95th percentile of the ratio, from histograms (B, bins)."""
    total = hist.sum(axis=-1, keepdims=True)
    cum = np.cumsum(hist, axis=-1) / np.maximum(total, 1)
    k = np.argmax(cum >= 0.95, axis=-1)
    out = k * p.STEP
    saturated = k >= p.N_RATIO - 1  # the last bin also holds every ratio above RATIO_MAX
    return np.where((total[..., 0] > 0) & ~saturated, out, np.nan)


def measures(*, s: np.ndarray) -> dict[str, np.ndarray]:
    """Return every measure from summed statistics (..., K)."""
    n = np.maximum(s[..., 0], 1)
    pit_share = s[..., p.N_SCALAR : p.N_SCALAR + 10] / n[..., None]
    base = p.N_SCALAR + 10
    return {
        "cov_q01": s[..., 1] / n,
        "cov_q05": s[..., 2] / n,
        "cov_q95": s[..., 3] / n,
        "cov_q99": s[..., 4] / n,
        "cov_c90": s[..., 5] / n,
        "width90": 100.0 * s[..., 6] / n,
        "pin05": 100.0 * s[..., 7] / n,
        "pin95": 100.0 * s[..., 8] / n,
        "factor_q95": factor(hist=s[..., base : base + p.N_RATIO]),
        "factor_q05": factor(hist=s[..., base + p.N_RATIO :]),
        "pit_deviation": np.abs(pit_share - 0.1).sum(axis=-1),
    }


def main() -> None:
    """Write the contrast and PIT tables."""
    z = np.load(p.WORK / "cube.npz")
    cube, clusters = z["cube"], [str(c) for c in z["clusters"]]
    groups = donor_groups()
    sets = {
        "all": np.arange(len(clusters)),
        "signed_six": np.array(
            [i for i, c in enumerate(clusters) if groups[c.split("|")[1]] == "signed_six"]
        ),
        "mva_eight": np.array(
            [i for i, c in enumerate(clusters) if groups[c.split("|")[1]] == "mva_eight"]
        ),
    }
    rng = np.random.default_rng(SEED)
    records, pit_records = [], []
    for set_name, members in sets.items():
        if len(members) == 0:
            continue
        w = rng.multinomial(len(members), np.full(len(members), 1 / len(members)), size=B).astype(
            np.float64
        )
        for si, sid in enumerate(z["scenarios"]):
            for ai, arm in enumerate(z["arms"]):
                for ci, cls in enumerate(z["classes"]):
                    for bi, band in enumerate(z["bands"]):
                        cells = cube[members][:, si, ai, :, ci, bi, :]  # (C, 2, K)
                        point = {
                            v: measures(s=cells[:, vi].sum(axis=0))
                            for vi, v in enumerate(z["variants"])
                        }
                        if point["injected"]["cov_q95"] is None or cells[:, 0, 0].sum() == 0:
                            continue
                        boot = {
                            v: measures(s=w @ cells[:, vi]) for vi, v in enumerate(z["variants"])
                        }
                        n_rows = int(cells[:, 0, 0].sum())
                        for m in MEASURES:
                            inj, cln = float(point["injected"][m]), float(point["clean"][m])
                            if m == "width90":
                                d_point = inj / cln - 1.0
                                d_boot = boot["injected"][m] / boot["clean"][m] - 1.0
                            else:
                                d_point = inj - cln
                                d_boot = boot["injected"][m] - boot["clean"][m]
                            lo, hi = np.nanquantile(d_boot, [ALPHA / 2, 1 - ALPHA / 2])
                            records.append(
                                dict(
                                    clusters=set_name,
                                    n_clusters=len(members),
                                    scenario=int(sid),
                                    arm=str(arm),
                                    cls=str(cls),
                                    band=str(band),
                                    measure=m,
                                    n_rows=n_rows,
                                    injected=inj,
                                    clean=cln,
                                    diff=d_point,
                                    boot_sd=float(np.nanstd(d_boot)),
                                    lo_bonferroni=float(lo),
                                    hi_bonferroni=float(hi),
                                )
                            )
                        if set_name == "all":
                            for vi, v in enumerate(z["variants"]):
                                s = cells[:, vi].sum(axis=0)
                                share = s[p.N_SCALAR : p.N_SCALAR + 10] / max(s[0], 1)
                                for b, val in enumerate(share):
                                    pit_records.append(
                                        dict(
                                            scenario=int(sid),
                                            arm=str(arm),
                                            cls=str(cls),
                                            band=str(band),
                                            variant=str(v),
                                            bin=b,
                                            share=float(val),
                                        )
                                    )
    pl.DataFrame(records).write_csv(p.WORK / "contrasts.csv", float_precision=5)
    pl.DataFrame(pit_records).write_csv(p.WORK / "pit.csv", float_precision=5)
    print(f"{len(records):,} contrast rows")


if __name__ == "__main__":
    main()

"""Experiment 508, Part A post hoc: D12 read with the thresholds of the nearest estimable bin.

D12's observed ratio lies in the 3 to 6 bin, which Part A could not calibrate (more than 99% of the
injected days cross zero). This post hoc script applies the frozen thresholds of the 1.5 to 3 bin,
the nearest estimable bin, to D12's shoulders, and is labelled an extrapolation.

    PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-*:ml-experiments/00508-* \
        uv run --no-sync python ml-experiments/00508-*/e00508_part_a_posthoc.py
"""

import json

import e00506_data as data  # ty: ignore[unresolved-import]
import e00508_common as common  # ty: ignore[unresolved-import]
import e00508_geometry as geo  # ty: ignore[unresolved-import]
import e00508_part_a as part_a  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BIN: int = 1


def main() -> None:
    """Apply the 1.5 to 3 bin thresholds to D12 and print dip shares and placebos."""
    rng = np.random.default_rng(5082)
    ref = geo.Mats(site="D12", extra={"farm": np.zeros(len(data.grid()))})
    band = geo.calibrate_shoulder(mats=ref)
    frozen = json.loads((common.WORK / "partA_thresholds.json").read_text())
    st = part_a.day_stats(mats=ref, level=0.0, band=band)
    rows = []
    for arm in common.ARMS:
        for window in common.SHOULDERS:
            t = frozen[f"{arm}|{BIN}|{window}"]
            stat, dark, k = st[f"{window}_{arm}"], st[f"{window}_dark_{arm}"], st[f"{window}_k"]
            have = st["qualifies"] & np.isfinite(stat)
            sunny, cloudy = have & (k >= data.SUNNY_K), have & (k < common.CLOUDY_K)
            dip = stat < t
            ks, ns, kc, nc = (int((dip & m).sum()) for m in (sunny, sunny, cloudy, cloudy))
            ns, nc = int(sunny.sum()), int(cloudy.sum())
            uniq, inv = np.unique(ref.block, return_inverse=True)
            per = np.stack(
                [
                    np.bincount(inv, weights=x.astype(float), minlength=len(uniq))
                    for x in (dip & sunny, sunny, dip & cloudy, cloudy)
                ]
            )
            w = np.stack(
                [
                    np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
                    for _ in range(20_000)
                ]
            ).astype(float)
            a, b, c, d = (w @ per[i] for i in range(4))
            draws = ((a + 0.5) / (b + 1)) / ((c + 0.5) / (d + 1))
            ok = sunny & np.isfinite(dark)
            dark_dip = dark < t
            free = ok & ~dark_dip
            rows.append(
                {
                    "arm": arm,
                    "window": window,
                    "threshold": t,
                    "n_sunny": ns,
                    "n_cloudy": nc,
                    "dips_sunny": ks,
                    "dips_cloudy": kc,
                    "sunny_to_cloudy": ((ks + 0.5) / (ns + 1)) / ((kc + 0.5) / (nc + 1)),
                    "ratio_lo": float(np.percentile(draws, 2.5)),
                    "ratio_hi": float(np.percentile(draws, 97.5)),
                    "dark_window_dip_share": float(dark_dip[ok].mean()),
                    "shoulder_dip_share_without_dark_dip": float(dip[free].mean()),
                    "n_free": int(free.sum()),
                }
            )
    table = pl.DataFrame(rows)
    table.write_parquet(common.WORK / "partA_d12_posthoc.parquet")
    pl.Config.set_tbl_cols(20)
    print(table)


if __name__ == "__main__":
    main()

# ruff: noqa: E501, C408
"""Experiment 509: coverage and width on rows with a lag inside a labelled event, against rows with no label.

Sensitivity only. Reads experiment 454's stored forecasts (LL, S1, B-mix, K2-S1: three seed cells
averaged level by level) and experiment 470's A-pool, with 454's row flags. An *event row* has any of
the three lags inside a labelled event (`lag_real`). A *no-label row* has `lag_real`, `tgt_real`, and
`win_real` all false. Actuals are the NRA-corrected actual (headline: the metered actual minus
experiment 435's normal-running-arrangement correction) and the metered actual (secondary). A cluster
is one (month, Primary) pair; the bootstrap resamples clusters and reports the event minus no-label
difference with its standard deviation and 2.8 standard deviations as the detectable effect.
NGED's labels are incomplete, so no-label rows may hold unlogged events.

```bash
export DATA_PATH_INTERNAL=<private data folder>
PYTHONPATH=<this folder>:<repo>/src uv run --no-sync python ml-experiments/00509-*/e00509_real.py
```
"""

from typing import Final

import e00509_paired as p  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MVA_ONLY: Final[frozenset[int]] = frozenset(
    {0, 1, 2, 4, 8, 13, 14, 15}
)  # D2 D3 D4 D8 D12 D19 D21 D22
REPORT_ONLY: Final[frozenset[int]] = frozenset({10, 12})  # D15 D18
NRA_FILE: Final = (
    p.ROOT
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
    / "nra_correction_share.npz"
)
B: Final[int] = 4000
FLAGS: Final[tuple[str, ...]] = ("lag_real", "tgt_real", "win_real")


def group_of(*, primary: int) -> str:
    """Return the metering group of a Primary index."""
    return (
        "report_only"
        if primary in REPORT_ONLY
        else "mva_eight"
        if primary in MVA_ONLY
        else "signed_six"
    )


def main() -> None:
    """Write `work/real_events.csv`."""
    nra = np.load(NRA_FILE)["f1.0"]
    stats: dict[tuple, np.ndarray] = {}
    for month in p.MONTHS:
        cells = sorted((p.W454 / "predictions" / "K2-S1").glob(f"{month}_s*_w-1.parquet"))
        base = (
            pl.read_parquet(cells[0])
            .select(*p.KEYS, "lead_hours", "actual", *FLAGS)
            .with_columns(pl.col(p.KEYS).cast(pl.Int64))
        )
        for arm in p.ARMS:
            q = base.join(
                p.clean_forecast(arm=arm, month=month), on=p.KEYS, how="left", maintain_order="left"
            )
            qs = q.select(p.LEVELS).to_numpy().astype(np.float64)
            ok = ~np.isnan(qs).any(axis=1)
            prim = q["primary"].to_numpy()
            tgt = q["target"].to_numpy()
            lead = q["lead_hours"].to_numpy().astype(np.float64)
            event = q["lag_real"].to_numpy()
            nolabel = ~(
                q["lag_real"].to_numpy() | q["tgt_real"].to_numpy() | q["win_real"].to_numpy()
            )
            metered = q["actual"].to_numpy().astype(np.float64)
            actuals = {"nra": metered - nra[prim, tgt], "metered": metered}
            for pr in np.unique(prim):
                for band, (lo, hi) in p.BANDS.items():
                    inband = ok & (prim == pr) & (lead >= lo) & (lead <= hi)
                    for name, mask in (("event", inband & event), ("nolabel", inband & nolabel)):
                        for act, y in actuals.items():
                            stats[month, int(pr), arm, band, name, act] = p.cell_stats(
                                q=qs[mask], y=y[mask]
                            )
    rng = np.random.default_rng(509)
    records = []
    for group in ("signed_six", "mva_eight", "report_only"):
        cl = sorted({(m, pr) for (m, pr, *_) in stats if group_of(primary=pr) == group})
        w = rng.multinomial(len(cl), np.full(len(cl), 1 / len(cl)), size=B).astype(np.float64)
        for arm in p.ARMS:
            for band in p.BANDS:
                for act in ("nra", "metered"):
                    ev = np.array([stats[m, pr, arm, band, "event", act] for m, pr in cl])
                    nl = np.array([stats[m, pr, arm, band, "nolabel", act] for m, pr in cl])
                    n_event_clusters = int((ev[:, 0] > 0).sum())
                    for measure, col in (("cov_q05", 2), ("cov_q95", 3), ("cov_c90", 5)):
                        e_pt = ev[:, col].sum() / max(ev[:, 0].sum(), 1)
                        n_pt = nl[:, col].sum() / max(nl[:, 0].sum(), 1)
                        e_b = (w @ ev[:, col]) / np.maximum(w @ ev[:, 0], 1)
                        n_b = (w @ nl[:, col]) / np.maximum(w @ nl[:, 0], 1)
                        records.append(
                            dict(
                                group=group,
                                arm=arm,
                                band=band,
                                actual=act,
                                measure=measure,
                                n_event=int(ev[:, 0].sum()),
                                n_nolabel=int(nl[:, 0].sum()),
                                event_clusters=n_event_clusters,
                                n_clusters=len(cl),
                                event=e_pt,
                                nolabel=n_pt,
                                diff=e_pt - n_pt,
                                boot_sd=float(np.std(e_b - n_b)),
                                mde=2.8 * float(np.std(e_b - n_b)),
                            )
                        )
                    e_w = ev[:, 6].sum() / max(ev[:, 0].sum(), 1)
                    n_w = nl[:, 6].sum() / max(nl[:, 0].sum(), 1)
                    e_wb = (w @ ev[:, 6]) / np.maximum(w @ ev[:, 0], 1)
                    n_wb = (w @ nl[:, 6]) / np.maximum(w @ nl[:, 0], 1)
                    records.append(
                        dict(
                            group=group,
                            arm=arm,
                            band=band,
                            actual=act,
                            measure="width90",
                            n_event=int(ev[:, 0].sum()),
                            n_nolabel=int(nl[:, 0].sum()),
                            event_clusters=n_event_clusters,
                            n_clusters=len(cl),
                            event=100 * e_w,
                            nolabel=100 * n_w,
                            diff=100 * (e_w - n_w),
                            boot_sd=100 * float(np.std(e_wb - n_wb)),
                            mde=2.8 * 100 * float(np.std(e_wb - n_wb)),
                        )
                    )
    out = pl.DataFrame(records)
    out.write_csv(p.WORK / "real_events.csv", float_precision=5)
    print(
        out.filter(
            (pl.col("band") == "d1_10")
            & (pl.col("actual") == "nra")
            & pl.col("measure").is_in(["cov_q95", "width90"])
            & (pl.col("group") != "report_only")
        )
    )


if __name__ == "__main__":
    main()

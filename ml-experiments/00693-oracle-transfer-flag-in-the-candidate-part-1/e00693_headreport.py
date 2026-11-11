# ruff: noqa: D103, E501, PLR0912, PERF401
"""Experiment 693: score the prevailing head arms and print the planned contrasts (CPU).

Reads `<root>/head_scores/<arm>_<fold>_<kind>_<label>.parquet` from `e00693_headfit.py` and writes
`<root>/scores/head_report.md` and `head_contrasts.csv`. Scores are in points of capacity; the
truth is the shifted (injected) power on a bank and the observed power on real rows.

Bank contrasts (C2, C3, C5 on bank F) resample clusters of (fold, pass, Primary) stratified by
fold, paired across arms. Real-row contrasts (C4) resample calendar months, paired across arms.
Intervals: 2,000 resamples; the Bonferroni level is 1 - 0.05 / 3 for the family C2, C3, C4-prevailing
(C1 and C4-NRA are not run because the stage 0 gate stopped the no-retrain-arm retraining).

Usage: python e00693_headreport.py
"""

from datetime import UTC, datetime
from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00693_headdata as hd
import e00693_real as real693
import numpy as np
import polars as pl

SCORES: Final = hd.ROOT / "head_scores"
OUT: Final = hd.ROOT / "scores"
HEAD_FOLDS: Final = ("2026-02", "2026-05")
FIRST_VI: Final[int] = int(
    (datetime(2026, 2, 1, tzinfo=UTC) - k578.GRID_START).total_seconds() // 1800
)
CUTOFF: Final[int] = hd.CUTOFF_IDX
N_BOOT: Final[int] = 2000
FAMILY: Final[int] = 3
LEVELS: Final = {"95%": 0.95, "Bonferroni": 1 - 0.05 / FAMILY, "90%": 0.90}


def read(*, arm: str, kind: str, folds: tuple[str, ...]) -> pl.DataFrame:
    frames = []
    for fold in folds:
        for f in sorted(SCORES.glob(f"{arm}_{fold}_{kind}_*.parquet")):
            frames.append(pl.read_parquet(f).with_columns(fold=pl.lit(fold)))
    return pl.concat(frames) if frames else pl.DataFrame()


def paired(
    *, a: pl.DataFrame, b: pl.DataFrame, keys: list[str], metric: str = "crps"
) -> pl.DataFrame:
    """Return the rows both arms scored, with the per-row difference `d = a - b` of `metric`."""
    j = a.select(*keys, "fold", pl.col(metric).alias("a")).join(
        b.select(*keys, pl.col(metric).alias("b")), on=keys, how="inner"
    )
    assert j.height == a.height == b.height, (
        f"the arms scored different rows: {a.height}, {b.height}, {j.height}"
    )
    return j.with_columns(d=pl.col("a") - pl.col("b"))


def interval(*, clusters: pl.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """Return the pooled mean difference and its intervals; `clusters` has columns strat, s, n."""
    groups = [g.select("s", "n").to_numpy() for _, g in clusters.group_by("strat")]
    point = sum(g[:, 0].sum() for g in groups) / sum(g[:, 1].sum() for g in groups)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = n = 0.0
        for g in groups:
            pick = g[rng.integers(0, len(g), len(g))]
            s += pick[:, 0].sum()
            n += pick[:, 1].sum()
        draws[b] = s / n
    out = {"mean": float(point), "n_clusters": float(sum(len(g) for g in groups))}
    for name, lvl in LEVELS.items():
        lo, hi = np.quantile(draws, [(1 - lvl) / 2, 1 - (1 - lvl) / 2])
        out[f"{name} low"], out[f"{name} high"] = float(lo), float(hi)
    return out


def bank_contrast(
    *, a: str, b: str, phase: int, kind: str, rng: np.random.Generator
) -> dict | None:
    ra = read(arm=a, kind=kind, folds=HEAD_FOLDS).filter(
        (pl.col("phase") == phase) & (pl.col("vi") >= FIRST_VI) & (pl.col("vi") < CUTOFF)
    )
    rb = read(arm=b, kind=kind, folds=HEAD_FOLDS).filter(
        (pl.col("phase") == phase) & (pl.col("vi") >= FIRST_VI) & (pl.col("vi") < CUTOFF)
    )
    if ra.is_empty():
        return None
    j = paired(a=ra, b=rb, keys=["prim", "oi", "vi", "pass_"])
    clusters = (
        j.group_by("fold", "pass_", "prim")
        .agg(s=pl.col("d").sum(), n=pl.len())
        .rename({"fold": "strat"})
    )
    return {"rows": j.height, **interval(clusters=clusters, rng=rng)}


def event_masks() -> dict[str, np.ndarray]:
    labels = k578.sites()["label"].to_list()
    n = hd.CUTOFF_IDX + 15 * hd.PER
    return {
        "verified": real693.mask_of(spans=real693.verified_spans(), labels=labels, length=n),
        "label+verified": real693.mask_of(spans=real693.verified_spans(), labels=labels, length=n)
        | real693.mask_of(spans=real693.labelled_spans(), labels=labels, length=n),
    }


def real_slices(*, df: pl.DataFrame, active: np.ndarray) -> pl.DataFrame:
    """Add the real-row scenario: 0 outside every event and at least 7 days after; 1 target inside;
    2 within 7 days after; 3 origin inside only."""
    after = hd.ended_within(active=active, days=7)
    near = active | after
    prim, oi, vi = (
        df["prim"].to_numpy().astype(np.int64),
        df["oi"].to_numpy().astype(np.int64),
        df["vi"].to_numpy().astype(np.int64),
    )
    clamp = lambda t: np.clip(t, 0, active.shape[1] - 1)  # noqa: E731
    origin_bad = near[prim, clamp(oi - 12)] | near[prim, clamp(oi - 48)]
    sc = np.where(
        active[prim, clamp(vi)], 1, np.where(after[prim, clamp(vi)], 2, np.where(origin_bad, 3, 0))
    )
    return df.with_columns(scenario=pl.Series(sc.astype(np.int8)))


def month_blocks(*, j: pl.DataFrame) -> pl.DataFrame:
    month = pl.from_epoch(
        (pl.col("vi") * 1800 + hd.GRID_US // 1_000_000), time_unit="s"
    ).dt.strftime("%Y-%m")
    return (
        j.with_columns(strat=pl.lit("real"), month=month)
        .group_by("strat", "month")
        .agg(s=pl.col("d").sum(), n=pl.len())
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(693)
    lines: list[str] = []
    rows: list[dict] = []
    masks = event_masks()
    arms = ("CAND", "B", "OB", "A0", "A1", "A2")
    lines.append("## Rows scored per arm, fold, and row set")
    for arm in arms:
        for kind in ("real", "bankheld", "bankdec"):
            d = read(arm=arm, kind=kind, folds=hd.FOLDS)
            if d.height:
                lines.append(
                    f"{arm:5s} {kind:9s} "
                    + ", ".join(f"{f}: {int((d['fold'] == f).sum()):,}" for f in hd.FOLDS)
                )
    lines.append("")
    lines.append(
        "## Bank F: mean CRPS and normalised MAE by phase, February to June 2026 (heads of folds 2026-02 and 2026-05)"
    )
    for kind in ("bankdec", "bankheld"):
        for arm in arms:
            d = read(arm=arm, kind=kind, folds=HEAD_FOLDS)
            if d.is_empty():
                continue
            d = d.filter((pl.col("vi") >= FIRST_VI) & (pl.col("vi") < CUTOFF))
            g = (
                d.group_by("phase")
                .agg(
                    rows=pl.len(),
                    crps=pl.col("crps").mean(),
                    mae=pl.col("ae").mean(),
                    share_p_pos=(pl.col("p") > 0).mean(),
                )
                .sort("phase")
            )
            for r in g.iter_rows(named=True):
                lines.append(
                    f"{kind:9s} {arm:5s} P{r['phase']}: rows {r['rows']:>10,} CRPS {r['crps']:.3f} MAE {r['mae']:.3f} share with p>0 {r['share_p_pos']:.3f}"
                )
    lines.append("")
    lines.append("## Planned contrasts, bank F (points of capacity, negative is better)")
    for kind, tag in (
        ("bankdec", "deciding bank (experiment 660 generator)"),
        ("bankheld", "check bank (experiment 650 held-out passes)"),
    ):
        for name, a, b, phase in (
            ("C2 A1-A2 P1", "A1", "A2", 1),
            ("C3 A1-A2 P2", "A1", "A2", 2),
            ("C2' A1-A0 P1", "A1", "A0", 1),
            ("C3' A1-A0 P2", "A1", "A0", 2),
            ("C5 A2-A0 P1", "A2", "A0", 1),
            ("C5 A2-A0 P2", "A2", "A0", 2),
            ("A0-CAND P1", "A0", "CAND", 1),
            ("A0-CAND P2", "A0", "CAND", 2),
            ("OB-B P1", "OB", "B", 1),
            ("OB-B P2", "OB", "B", 2),
        ):
            res = bank_contrast(a=a, b=b, phase=phase, kind=kind, rng=rng)
            if res is None:
                continue
            rows.append({"set": kind, "contrast": name, **res})
            lines.append(
                f"{tag}: {name}: {res['mean']:+.3f} (95% [{res['95% low']:+.3f}, {res['95% high']:+.3f}]; Bonferroni [{res['Bonferroni low']:+.3f}, {res['Bonferroni high']:+.3f}]; 90% [{res['90% low']:+.3f}, {res['90% high']:+.3f}]; {int(res['rows']):,} rows, {int(res['n_clusters'])} clusters)"
            )
    lines.append("")
    lines.append(
        "## Real rows: CRPS and MAE by scenario, February to June 2026 (descriptive except C4)"
    )
    for def_name, active in masks.items():
        for arm in arms:
            d = read(arm=arm, kind="real", folds=HEAD_FOLDS)
            if d.is_empty():
                continue
            d = real_slices(
                df=d.filter((pl.col("vi") >= FIRST_VI) & (pl.col("vi") < CUTOFF)), active=active
            )
            g = (
                d.group_by("scenario")
                .agg(rows=pl.len(), crps=pl.col("crps").mean(), mae=pl.col("ae").mean())
                .sort("scenario")
            )
            lines.append(
                f"{def_name:15s} {arm:5s} "
                + "; ".join(
                    f"S{r['scenario']} rows {r['rows']:,} CRPS {r['crps']:.3f} MAE {r['mae']:.3f}"
                    for r in g.iter_rows(named=True)
                )
            )
    lines.append(
        "scenarios: S0 event-free (target, origin, and 7 days after), S1 target inside an event, S2 target within 7 days after an event, S3 origin inside or just after an event only"
    )
    lines.append("")
    lines.append(
        "## C4 (harm): A1 minus A0 on event-free real rows, month blocks, February to June 2026"
    )
    for def_name, active in masks.items():
        for a, b, name in (
            ("A1", "A0", "C4 A1-A0"),
            ("A2", "A0", "C5 A2-A0"),
            ("A1", "CAND", "A1-CAND"),
            ("A0", "CAND", "A0-CAND"),
        ):
            ra = read(arm=a, kind="real", folds=HEAD_FOLDS).filter(
                (pl.col("vi") >= FIRST_VI) & (pl.col("vi") < CUTOFF)
            )
            rb = read(arm=b, kind="real", folds=HEAD_FOLDS).filter(
                (pl.col("vi") >= FIRST_VI) & (pl.col("vi") < CUTOFF)
            )
            if ra.is_empty():
                continue
            j = paired(a=ra, b=rb, keys=["prim", "oi", "vi"])
            j = real_slices(df=j, active=active).filter(pl.col("scenario") == 0)
            res = interval(clusters=month_blocks(j=j), rng=rng)
            rows.append({"set": f"real {def_name}", "contrast": name, "rows": j.height, **res})
            lines.append(
                f"{def_name}: {name}: {res['mean']:+.3f} (95% [{res['95% low']:+.3f}, {res['95% high']:+.3f}]; Bonferroni [{res['Bonferroni low']:+.3f}, {res['Bonferroni high']:+.3f}]; 90% [{res['90% low']:+.3f}, {res['90% high']:+.3f}]; {j.height:,} rows, {int(res['n_clusters'])} month blocks)"
            )
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "head_contrasts.csv")
    (OUT / "head_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

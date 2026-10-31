"""Audit the pooled contrasts of experiment 404 (CPU only, no fit).

Row 1 of the Opus review: reconcile the event counts, re-pool the contrasts weighted by events,
report single-event months apart, and rescore every arm raw, with the 1.15 widening of the
experiment-428 protocol, and with the experiment-435 table P (factors refitted on the arm's own
earlier months, clean real rows, per group and lead band, so no month sees its own rows).
Row 2: rescore the rows inside real events against experiment 435's NRA-corrected actuals.

```bash
PYTHONPATH=<as e00404_fit.py shows>:<ml-experiments/00435-* of the uncertainty-stage0 worktree> \
    uv run --no-sync python ml-experiments/00404-*/e00404_audit.py cells \
    --arms A0-inj B0 B1 B3 B6 S1 B4c --reference B-mix
uv run --no-sync python ml-experiments/00404-*/e00404_audit.py pool
uv run --no-sync python ml-experiments/00404-*/e00404_audit.py events
```
"""

import argparse
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import e00435_tailcore as tailcore  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

AUDIT_DIR: Final = fit.WORK_DIR / "audit"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
CELLS: Final[tuple[tuple[str, int], ...]] = tuple(
    (m, s) for m in MONTHS for s in ((0, 1, 2) if m == "2026-06" else (0,))
)
GROUP_NAME: Final[dict[str, str]] = {
    "signed_six": "signed-MW six",
    "mva_eight": "MVA-only eight",
    "report_only": "D15 and D18",
}
REAL_CLASSES: Final[tuple[str, ...]] = ("harm_real", "tgt_real", "clean_real")
VARIANTS: Final[tuple[str, ...]] = ("raw", "widen1.15", "tableP")
DEPTHS: Final[tuple[float, ...]] = (0.75, 1.0, 1.25)
TAILS: Final[tuple[str, ...]] = tuple(tailcore.TAIL_LEVELS)


def has(*, arm: str, month: str, seed: int) -> bool:
    """Return whether the arm has real-world forecasts for the cell."""
    return score.prediction_path(arm=arm, month=month, seed=seed, world=-1).exists()


def real_frame(*, arm: str, month: str, seed: int) -> pl.DataFrame:
    """Return an arm's real-world forecasts for a cell, with the 435 group and lead band."""
    frame = report.load(arm=arm, month=month, seed=seed, worlds=[-1])
    return frame.with_columns(
        group435=pl.col("group").replace(GROUP_NAME),
        lead_band=tailcore.lead_band_expr(),
    ).filter(pl.col("lead_band").is_not_null())


def rescore(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Recompute the pinball score of a frame after its quantiles or actuals changed."""
    return frame.with_columns(crps=score.pinball_points(frame=frame))


def table_p_factors(*, arm: str, month: str) -> pl.DataFrame:
    """Return table P factors fitted on the arm's clean real rows of earlier months (seed 0)."""
    earlier = [m for m in MONTHS if m < month and has(arm=arm, month=m, seed=0)]
    if not earlier:
        return pl.DataFrame(
            schema={"group": pl.String, "lead_band": pl.String, "tail": pl.String,
                    "factor": pl.Float64, "n_fit": pl.Int64}
        )  # fmt: skip
    rows = pl.concat(
        [
            real_frame(arm=arm, month=m, seed=0)
            .filter(pl.col("cls") == "clean_real")
            .select("group435", "lead_band", "actual", "q50", *TAILS)
            .rename({"group435": "group"})
            for m in earlier
        ]
    ).cast(dict.fromkeys(("actual", "q50", *TAILS), pl.Float64))
    return prov.fit_factors(fit=rows)


def variant_frame(*, frame: pl.DataFrame, variant: str, factors: pl.DataFrame) -> pl.DataFrame:
    """Return the frame with its tail quantiles replaced according to the variant."""
    if variant == "raw":
        return frame
    typed = frame.cast(dict.fromkeys(("q50", *TAILS), pl.Float64))
    if variant == "widen1.15":
        out = tailcore.widen(frame=typed, factor=tailcore.WIDENING)
    else:
        out = prov.apply_factors(
            frame=typed.rename({"group435": "group_p"}).with_columns(group=pl.col("group_p")),
            factors=factors,
        )
    return rescore(frame=out.cast(dict.fromkeys(("q50", *TAILS), pl.Float32)))


def cells(*, arms: list[str], reference: str) -> None:
    """Write one row per variant, arm, group, class, and cell with the contrast to the reference."""
    real = score.real_event_mask()
    out = []
    for month, seed in CELLS:
        if not has(arm=reference, month=month, seed=seed):
            continue
        ref0 = real_frame(arm=reference, month=month, seed=seed)
        clusters = report.cluster_ids(frame=ref0, real=real)
        factors = {
            a: table_p_factors(arm=a, month=month)
            for a in [reference, *arms]
            if has(arm=a, month=month, seed=seed)
        }
        for variant in VARIANTS:
            ref = variant_frame(frame=ref0, variant=variant, factors=factors[reference])
            for arm in arms:
                if arm not in factors:
                    continue
                frame0 = real_frame(arm=arm, month=month, seed=seed)
                frame = variant_frame(frame=frame0, variant=variant, factors=factors[arm])
                for group in GROUP_NAME:
                    for cls in REAL_CLASSES:
                        mask = ((ref0["group"] == group) & (ref0["cls"] == cls)).to_numpy()
                        if mask.sum() == 0:
                            continue
                        a, b = frame.filter(pl.Series(mask)), ref.filter(pl.Series(mask))
                        cl = clusters[mask]
                        point, low, high = report.contrast(a=a, b=b, cluster=cl)
                        drop, _, _ = (
                            report.contrast(a=a, b=b, cluster=cl, drop_largest=True)
                            if len(np.unique(cl)) > 1
                            else (float("nan"),) * 3
                        )
                        out.append(
                            {
                                "variant": variant, "arm": arm, "group": group, "cls": cls,
                                "month": month, "seed": seed, "rows": int(mask.sum()),
                                "events": len(np.unique(cl)),
                                "arm_score": float(a["crps"].mean()),  # ty: ignore[invalid-argument-type]
                                "ref_score": float(b["crps"].mean()),  # ty: ignore[invalid-argument-type]
                                "diff": point, "low": low, "high": high, "diff_drop_largest": drop,
                            }
                        )  # fmt: skip
        print(f"{month} seed {seed} done")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out).write_parquet(AUDIT_DIR / "cells.parquet")


def pool() -> None:
    """Pool the per-cell contrasts three ways and write the table."""
    cell = pl.read_parquet(AUDIT_DIR / "cells.parquet")
    month_level = cell.group_by("variant", "arm", "group", "cls", "month").agg(
        pl.col("diff").mean(), pl.col("events").first(), pl.col("diff_drop_largest").mean()
    )
    rows = []
    for key, part in month_level.group_by("variant", "arm", "group", "cls", maintain_order=True):
        d = part["diff"].to_numpy()
        w = part["events"].to_numpy().astype(float)
        multi = part.filter(pl.col("events") > 1)
        single = part.filter(pl.col("events") == 1)
        loo = [float(np.delete(d, i).mean()) for i in range(len(d))] if len(d) > 1 else [np.nan]
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        rows.append(
            {
                "variant": key[0], "arm": key[1], "group": key[2], "cls": key[3],
                "months": len(d), "events_total": int(w.sum()),
                "mean_of_months": float(d.mean()),
                "event_weighted": float((d * w).sum() / w.sum()),
                "multi_event_months": multi.height,
                "multi_event_mean": float(multi["diff"].mean()) if multi.height else float("nan"),  # ty: ignore[invalid-argument-type]
                "single_event_months": single.height,
                "single_event_values": ", ".join(
                    f"{m}:{v:+.2f}" for m, v in zip(single["month"], single["diff"], strict=True)
                ),
                "leave_one_month_out_min": float(np.nanmin(loo)),
                "leave_one_month_out_max": float(np.nanmax(loo)),
                "mde_80pct_power": 2.8 * sd / np.sqrt(len(d)) if len(d) > 1 else float("nan"),
            }
        )  # fmt: skip
    table = pl.DataFrame(rows).sort("variant", "group", "cls", "arm")
    table.write_csv(AUDIT_DIR / "pooled_event_weighted.csv", float_precision=3)
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(20)
    print(table)


def events() -> None:
    """Reconcile event counts: padded runs, unpadded events, and bootstrap clusters per month."""
    real = score.real_event_mask()
    sites = fit.prep.primaries()["site"].to_list()
    groups = report.group_of(sites=sites)
    corrected = pl.read_parquet(
        score.ML_EXPERIMENTS_DATA_DIR
        / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored"
        / "work"
        / "events_corrected.parquet"
    )
    rows = []
    for month in MONTHS:
        ref0 = real_frame(arm="B-mix", month=month, seed=0)
        lo, hi = int(ref0["target"].min()), int(ref0["target"].max())  # ty: ignore[invalid-argument-type]
        clusters = report.cluster_ids(frame=ref0, real=real)
        for group in GROUP_NAME:
            padded_runs = 0
            unpadded = 0
            for p, site in enumerate(sites):
                if groups[p] != group:
                    continue
                run = np.diff(np.concatenate([[0], real[p, lo : hi + 1].astype(np.int8)])) == 1
                padded_runs += int(run.sum())
                unpadded += corrected.filter(
                    (pl.col("label") == site)
                    & (pl.col("end_idx") >= lo)
                    & (pl.col("onset_idx") <= hi)
                ).height
            entry = {"month": month, "group": group, "padded_runs": padded_runs,
                     "unpadded_events": unpadded}  # fmt: skip
            for cls in ("harm_real", "tgt_real"):
                mask = ((ref0["group"] == group) & (ref0["cls"] == cls)).to_numpy()
                entry[f"clusters_{cls}"] = len(np.unique(clusters[mask])) if mask.any() else 0
                entry[f"rows_{cls}"] = int(mask.sum())
            rows.append(entry)
    table = pl.DataFrame(rows)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(AUDIT_DIR / "event_counts.csv")
    pl.Config.set_tbl_rows(100)
    print(table)


def nra(*, arms: list[str], reference: str) -> None:
    """Rescore the rows inside real events against NRA-corrected actuals (row 2 of the review)."""
    shares = {f: np.load(prov.WORK_DIR / "nra_correction_share.npz")[f"f{f}"] for f in DEPTHS}
    real = score.real_event_mask()
    out = []
    for month, seed in CELLS:
        if not has(arm=reference, month=month, seed=seed):
            continue
        ref0 = real_frame(arm=reference, month=month, seed=seed)
        clusters = report.cluster_ids(frame=ref0, real=real)
        for group in GROUP_NAME:
            mask = ((ref0["group"] == group) & (ref0["cls"] == "tgt_real")).to_numpy()
            if not mask.any():
                continue
            primary = ref0["primary"].to_numpy()[mask].astype(np.int64)
            grid = np.clip(ref0["target"].to_numpy()[mask].astype(np.int64), 0, 121247)
            cl = clusters[mask]
            for arm in [reference, *arms]:
                if not has(arm=arm, month=month, seed=seed):
                    continue
                frame = real_frame(arm=arm, month=month, seed=seed).filter(pl.Series(mask))
                for depth in ("measured_as_metered", *DEPTHS):
                    actual = frame["actual"].to_numpy().astype(np.float64)
                    if depth != "measured_as_metered":
                        actual = actual - shares[depth][primary, grid]
                    scored = rescore(frame=frame.with_columns(actual=pl.Series(actual)))
                    out.append(
                        {
                            "arm": arm, "group": group, "month": month, "seed": seed,
                            "depth": str(depth), "rows": int(mask.sum()),
                            "events": len(np.unique(cl)),
                            "score": float(scored["crps"].mean()),  # ty: ignore[invalid-argument-type]
                            "crps": scored["crps"].to_numpy(), "cluster": cl,
                        }
                    )  # fmt: skip
    # Contrasts per cell, with and without the largest event.
    rows = []
    frame = pl.DataFrame(
        [{k: v for k, v in r.items() if k not in {"crps", "cluster"}} for r in out]
    )
    refs = {
        (r["group"], r["month"], r["seed"], r["depth"]): r for r in out if r["arm"] == reference
    }
    for r in out:
        if r["arm"] == reference:
            continue
        ref = refs[(r["group"], r["month"], r["seed"], r["depth"])]
        diff = r["crps"] - ref["crps"]  # ty: ignore[unsupported-operator]
        codes, inverse = np.unique(r["cluster"], return_inverse=True)
        totals = np.abs(np.bincount(inverse, weights=diff, minlength=len(codes)))
        keep = inverse != int(np.argmax(totals))
        rows.append(
            {
                "arm": r["arm"], "group": r["group"], "month": r["month"], "seed": r["seed"],
                "depth": r["depth"], "events": r["events"], "diff": float(diff.mean()),
                "diff_drop_largest": float(diff[keep].mean()) if len(codes) > 1 else float("nan"),
            }
        )  # fmt: skip
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(AUDIT_DIR / "nra_scores.parquet")
    table = pl.DataFrame(rows)
    table.write_parquet(AUDIT_DIR / "nra_cells.parquet")
    summary = (
        table.group_by("arm", "group", "depth", "month")
        .agg(pl.col("diff").mean(), pl.col("diff_drop_largest").mean(), pl.col("events").first())
        .group_by("arm", "group", "depth")
        .agg(
            pl.col("diff").mean().alias("mean_of_months"),
            ((pl.col("diff") * pl.col("events")).sum() / pl.col("events").sum()).alias(
                "event_weighted"
            ),
            pl.col("diff_drop_largest").mean().alias("drop_largest_mean"),
            pl.col("events").sum().alias("events_total"),
            pl.len().alias("months"),
        )
        .sort("group", "depth", "arm")
    )
    summary.write_csv(AUDIT_DIR / "nra_summary.csv", float_precision=3)
    pl.Config.set_tbl_rows(400)
    print(summary)


def main() -> None:
    """Run one audit step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cells", "pool", "events", "nra"))
    parser.add_argument(
        "--arms", nargs="+", default=["A0-inj", "B0", "B1", "B3", "B6", "S1", "B4c"]
    )
    parser.add_argument("--reference", default="B-mix")
    args = parser.parse_args()
    if args.command == "cells":
        cells(arms=args.arms, reference=args.reference)
    elif args.command == "pool":
        pool()
    elif args.command == "events":
        events()
    else:
        nra(arms=args.arms, reference=args.reference)


if __name__ == "__main__":
    main()

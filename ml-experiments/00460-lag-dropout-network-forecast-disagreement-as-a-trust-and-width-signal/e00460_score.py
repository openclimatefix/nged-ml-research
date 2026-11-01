# ruff: noqa: E501, PLR0912, PLR0915, PLC0206, PERF401, PT018
"""Score experiment 460: minimum detectable effect, AUC of the spread, contrasts, and tails.

All scores are the mass-weighted pinball loss in points of capacity (CRPS-like), seeds (cells)
averaged within a row, on experiment 404's real-world rows. Row classes are experiment 404's
`report.load` classes (`clean_real` is called "rows with no labelled event" in the write-up).
Rows inside real events are scored against experiment 435's NRA-corrected actuals (headline) and the
metered actuals (secondary).

Commands (each writes CSV files under the experiment's `work/score/` folder):

- `mde`: seed-to-seed spread of B-mix, run before any contrast is read.
- `auc`: AUC of the spread signals (real harm rows against rows with no labelled event, and the
  known-answer control on the injected worlds).
- `contrasts`: absolute scores, contrasts with intervals, and tails, raw and after table P.

```bash
source /tmp/e460_env.sh
uv run --no-sync python ml-experiments/00460-*/e00460_score.py mde
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import e00435_tailcore as tailcore  # ty: ignore[unresolved-import]
import e00460_arms as arms  # ty: ignore[unresolved-import]
import e00460_heads as heads_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.stats import rankdata

SCORE_DIR: Final[Path] = arms.WORK_DIR / "score"
MONTHS: Final[tuple[str, ...]] = arms.MONTHS
GROUPS: Final[tuple[str, ...]] = ("signed_six", "mva_eight", "report_only")
CLASSES: Final[tuple[str, ...]] = ("clean_real", "harm_real", "tgt_real")
BOOTSTRAP: Final[int] = 2000
AUC_BOOTSTRAP: Final[int] = 400
AUC_NEGATIVES_PER_MONTH: Final[int] = 40_000
MIN_POSITIVES_PER_BAND: Final[int] = 20
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "S1d", "S1"),
    ("C2", "S1d-c", "B-mix"),
    ("C3", "S1d-d", "B-mix"),
    ("C4", "S1d-c", "S1w"),
    ("C5", "S1d-d", "S1-ens3"),
    ("C6", "S1d-c", "S1d-c0"),
    ("C7", "S1-ens3", "B-mix"),
    ("C8", "S1d", "B-mix"),  # post hoc from here: not among the 42 pre-registered
    ("C9", "S1d-c", "S1d"),
    ("C10", "S1d-d", "S1d"),
    ("C11", "S1", "B-mix"),
    ("C12", "S1d", "S1d-c0"),
)
ARM_NAMES: Final[tuple[str, ...]] = (
    "S1", "S1d", "S1d-c", "S1d-d", "B-mix", "S1w", "S1-ens3", "S1d-c0",
)  # fmt: skip
TAIL_ARMS: Final[tuple[str, ...]] = ("B-mix", "S1", "S1-ens3", "S1d")
"""Arms that also get a table P (fitted like arm c0's, with one bin) for the tails table."""
BAND_ALL: Final[str] = "all"
BAND_12_36: Final[str] = "12-36h"


def cells_of(*, arm: str, month: str) -> list[int]:
    """Return the fit-seed cells an arm has for a month."""
    return [
        c for c in heads_mod.SEEDS if arms.prediction_path(arm=arm, month=month, cell=c).exists()
    ]


def cell_frame(*, arm: str, month: str, cell: int, nra: bool, shares: np.ndarray) -> pl.DataFrame:
    """Return an arm's real-world cell with the scores, classes, and (optionally) NRA actuals."""
    fit.PRED_DIR = arms.WORK_DIR / "predictions"
    frame = report.load(arm=arm, month=month, seed=cell, worlds=[-1])
    if nra:
        primary = frame["primary"].to_numpy().astype(np.int64)
        grid = np.clip(frame["target"].to_numpy().astype(np.int64), 0, shares.shape[1] - 1)
        inside = (frame["cls"] == "tgt_real").to_numpy()
        actual = frame["actual"].to_numpy().astype(np.float64)
        actual = np.where(inside, actual - shares[primary, grid], actual)
        frame = frame.with_columns(actual=pl.Series(actual))
    frame = frame.with_columns(crps=score.pinball_points(frame=frame))
    exprs = []
    for name, tau in tailcore.TAIL_LEVELS.items():
        error = pl.col("actual") - pl.col(name)
        exprs.append(pl.max_horizontal(tau * error, (tau - 1.0) * error).alias(f"pin_{name}"))
        exprs.append((pl.col("actual") <= pl.col(name)).cast(pl.Float64).alias(f"cov_{name}"))
    return frame.with_columns(*exprs, lead_band=tailcore.lead_band_expr()).filter(
        pl.col("lead_band").is_not_null()
    )


def arm_month(*, arm: str, month: str, nra: bool, shares: np.ndarray) -> pl.DataFrame:
    """Return an arm's month with the metrics averaged over its cells, on identical rows."""
    frames = [
        cell_frame(arm=arm, month=month, cell=c, nra=nra, shares=shares)
        for c in cells_of(arm=arm, month=month)
    ]
    first = frames[0]
    metric_columns = ["crps", *(f"{p}_{n}" for n in tailcore.TAIL_LEVELS for p in ("pin", "cov"))]
    for f in frames[1:]:
        assert f["target"].equals(first["target"]) and f["origin"].equals(first["origin"])
    mean = {c: np.mean([f[c].to_numpy() for f in frames], axis=0) for c in metric_columns}
    return first.with_columns(**{c: pl.Series(v) for c, v in mean.items()})


def table_p_arm(*, arm: str, month: str, nra: bool, shares: np.ndarray) -> pl.DataFrame | None:
    """Return the arm's month after a table P fitted (per cell) on earlier months, or None."""
    earlier = [m for m in MONTHS if m < month]
    if not earlier or not cells_of(arm=arm, month=month):
        return None
    out = []
    for cell in cells_of(arm=arm, month=month):
        fit_rows = pl.concat(
            [
                arms.with_groups(
                    frame=pl.read_parquet(
                        arms.prediction_path(arm=arm, month=m, cell=cell)
                    ).with_columns(**{arms.SPREAD: pl.lit(0.0)})
                )
                for m in earlier
            ]
        ).cast(dict.fromkeys(("actual", "q50", *tailcore.TAIL_LEVELS), pl.Float64))
        factors = arms.fit_widening(fit_rows=fit_rows, edges={}, n_bins=1)
        frame = pl.read_parquet(arms.prediction_path(arm=arm, month=month, cell=cell))
        grouped = arms.with_groups(frame=frame.with_columns(**{arms.SPREAD: pl.lit(0.0)}))
        widened = arms.apply_widening(
            frame=grouped.cast(dict.fromkeys(("q50", *tailcore.TAIL_LEVELS), pl.Float64)),
            factors=factors, edges={}, n_bins=1,
        )  # fmt: skip
        joined = frame.join(
            widened.select(*heads_mod.KEY_COLUMNS, *tailcore.TAIL_LEVELS),
            on=list(heads_mod.KEY_COLUMNS), how="left", suffix="_w",
        )  # fmt: skip
        for n in tailcore.TAIL_LEVELS:
            joined = joined.with_columns(pl.coalesce(f"{n}_w", n).alias(n)).drop(f"{n}_w")
        path = arms.WORK_DIR / "predictions" / f"{arm}+P" / f"{month}_s{cell}_w-1.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        joined.cast(dict.fromkeys(score.LEVEL_NAMES, pl.Float32)).write_parquet(path)
        out.append(cell)
    return arm_month(arm=f"{arm}+P", month=month, nra=nra, shares=shares)


def nra_shares() -> np.ndarray:
    """Return experiment 435's NRA correction share at the labelled depth (factor 1.0)."""
    return np.load(prov.WORK_DIR / "nra_correction_share.npz")["f1.0"]


def mde() -> None:
    """Print and write the seed-to-seed spread of B-mix on rows with no labelled event."""
    shares = nra_shares()
    rows = []
    for month in MONTHS:
        cells = {
            c: cell_frame(arm="B-mix", month=month, cell=c, nra=False, shares=shares)
            for c in cells_of(arm="B-mix", month=month)
        }
        for group in GROUPS:
            means = {
                c: float(
                    f.filter((pl.col("group") == group) & (pl.col("cls") == "clean_real"))[  # ty: ignore[invalid-argument-type]
                        "crps"
                    ].mean()
                )  # ty: ignore[invalid-argument-type]
                for c, f in cells.items()
            }
            for a in means:
                for b in means:
                    if a < b:
                        rows.append(
                            {"month": month, "group": group, "pair": f"{a}-{b}",
                             "diff": means[a] - means[b]}
                        )  # fmt: skip
    table = pl.DataFrame(rows)
    summary = table.group_by("group").agg(
        pl.col("diff").std().alias("sd_of_seed_pair_differences"),
        pl.len().alias("n_pairs"),
        (2.0 * pl.col("diff").std() / np.sqrt(len(MONTHS))).alias("minimum_detectable_effect"),
    )
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(SCORE_DIR / "mde_pairs.csv")
    summary.write_csv(SCORE_DIR / "mde.csv", float_precision=4)
    print(summary)


def auc_of(*, pos: np.ndarray, neg: np.ndarray) -> float:
    """Return the area under the ROC curve for a signal that is larger on positive rows."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_inputs(*, real: np.ndarray) -> dict:
    """Return per (group, seed, month) positive and negative signal arrays for real and injected."""
    out: dict = {}
    rng = np.random.default_rng(0)
    for month in MONTHS:
        for seed in heads_mod.SEEDS:
            for world in heads_mod.WORLDS:
                frame = pl.read_parquet(heads_mod.passes_path(month=month, seed=seed, world=world))
                frame = arms.with_groups(frame=frame).with_columns(world=pl.lit(world))
                groups = frame["group"].to_numpy()
                if world < 0:
                    inside = (frame["tgt_real"] | frame["tgt_post"]).to_numpy()
                    pos = (frame["win_real"] & ~frame["tgt_real"] & ~frame["tgt_post"]).to_numpy()
                    neg = ~inside & ~frame["win_real"].to_numpy()
                    clusters = report.cluster_ids(
                        frame=frame.with_columns(world=pl.lit(-1)), real=real
                    )
                else:
                    clean = ~frame["win_inj"] & ~frame["tgt_inj"]
                    pos = (frame["win_inj"] & ~frame["tgt_inj"]).to_numpy()
                    neg = clean.to_numpy()
                    clusters = np.zeros(frame.height, dtype=np.int64)
                for g in groups_names():
                    sel = groups == g
                    p_idx, n_idx = np.flatnonzero(sel & pos), np.flatnonzero(sel & neg)
                    if len(n_idx) > AUC_NEGATIVES_PER_MONTH:
                        n_idx = rng.choice(n_idx, size=AUC_NEGATIVES_PER_MONTH, replace=False)
                    entry = {
                        "lead": frame["lead_hours"].to_numpy(),
                        "mean": frame["spread_mean"].to_numpy(),
                        "max": frame["spread_max"].to_numpy(),
                        "band": frame["lead_band"].to_numpy(),
                        "cluster": clusters,
                    }
                    out[(g, "real" if world < 0 else "injected", seed, month, world)] = {
                        k: (v[p_idx], v[n_idx]) for k, v in entry.items()
                    }
    return out


def groups_names() -> list[str]:
    """Return the group names used in the passes tables."""
    return list(arms.GROUP_NAME.values())


def pooled_auc(*, data: dict, group: str, kind: str, signal: str, months: list[str]) -> float:
    """Return the AUC pooled over `months` per seed (and world), averaged over seeds."""
    values = []
    for seed in heads_mod.SEEDS:
        keys = [
            k for k in data if k[0] == group and k[1] == kind and k[2] == seed and k[3] in months
        ]
        pos = np.concatenate([data[k][signal][0] for k in keys])
        neg = np.concatenate([data[k][signal][1] for k in keys])
        values.append(auc_of(pos=pos, neg=neg))
    return float(np.nanmean(values))


def stratified_auc(*, data: dict, group: str, signal: str) -> float:
    """Return the mean over lead bands (with enough positives) of the pooled real-world AUC."""
    bands = list(tailcore.LEAD_BANDS)
    values = []
    for band in bands:
        per_seed = []
        for seed in heads_mod.SEEDS:
            keys = [k for k in data if k[:3] == (group, "real", seed)]
            pos = np.concatenate([data[k][signal][0][data[k]["band"][0] == band] for k in keys])
            neg = np.concatenate([data[k][signal][1][data[k]["band"][1] == band] for k in keys])
            if len(pos) >= MIN_POSITIVES_PER_BAND:
                per_seed.append(auc_of(pos=pos, neg=neg))
        if per_seed:
            values.append(float(np.mean(per_seed)))
    return float(np.mean(values)) if values else float("nan")


def auc() -> None:
    """Write the AUC table: point, month-block interval, event-cluster interval, and checks."""
    real = score.real_event_mask()
    data = auc_inputs(real=real)
    rng = np.random.default_rng(0)
    rows = []
    for group in groups_names():
        for signal in ("mean", "max", "lead"):
            point = pooled_auc(
                data=data, group=group, kind="real", signal=signal, months=list(MONTHS)
            )
            draws = []
            for _ in range(AUC_BOOTSTRAP):
                picked = [MONTHS[i] for i in rng.integers(0, len(MONTHS), len(MONTHS))]
                draws.append(
                    pooled_auc_with_repeats(data=data, group=group, signal=signal, months=picked)
                )
            low_m, high_m = np.nanquantile(draws, [0.025, 0.975])
            loo = [
                pooled_auc(data=data, group=group, kind="real", signal=signal,
                           months=[m for m in MONTHS if m != left])
                for left in MONTHS
            ]  # fmt: skip
            cluster_draws = cluster_bootstrap(data=data, group=group, signal=signal, rng=rng)
            low_c, high_c = np.nanquantile(cluster_draws, [0.025, 0.975])
            rows.append(
                {
                    "group": group, "signal": signal, "auc": point,
                    "month_low": float(low_m), "month_high": float(high_m),
                    "event_low": float(low_c), "event_high": float(high_c),
                    "leave_one_month_min": float(np.nanmin(loo)),
                    "leave_one_month_max": float(np.nanmax(loo)),
                    "lead_stratified": stratified_auc(data=data, group=group, signal=signal),
                    "injected_world_auc": pooled_auc(
                        data=data, group=group, kind="injected", signal=signal, months=list(MONTHS)
                    ),
                    "positives": int(sum(len(data[k][signal][0]) for k in data
                                         if k[0] == group and k[1] == "real" and k[2] == 0)),
                }
            )  # fmt: skip
    table = pl.DataFrame(rows)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(SCORE_DIR / "auc.csv", float_precision=3)
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=220):
        print(table)


def pooled_auc_with_repeats(*, data: dict, group: str, signal: str, months: list[str]) -> float:
    """Return the AUC over a month list that may repeat months, averaged over seeds."""
    values = []
    for seed in heads_mod.SEEDS:
        pos = np.concatenate([data[(group, "real", seed, m, -1)][signal][0] for m in months])
        neg = np.concatenate([data[(group, "real", seed, m, -1)][signal][1] for m in months])
        values.append(auc_of(pos=pos, neg=neg))
    return float(np.nanmean(values))


def cluster_bootstrap(*, data: dict, group: str, signal: str, rng: np.random.Generator) -> list:
    """Return AUC draws that resample the real events of the positive rows (negatives fixed)."""
    draws = []
    per_seed = []
    for seed in heads_mod.SEEDS:
        keys = [(group, "real", seed, m, -1) for m in MONTHS]
        pos_values = np.concatenate([data[k][signal][0] for k in keys])
        pos_cluster = np.concatenate(
            [data[k]["cluster"][0] + 10_000_000 * i for i, k in enumerate(keys)]
        )
        neg = np.concatenate([data[k][signal][1] for k in keys])
        codes, inverse = np.unique(pos_cluster, return_inverse=True)
        per_seed.append((pos_values, inverse, len(codes), neg))
    for _ in range(AUC_BOOTSTRAP):
        values = []
        for pos_values, inverse, n_codes, neg in per_seed:
            if n_codes == 0:
                continue
            pick = rng.integers(0, n_codes, n_codes)
            counts = np.bincount(pick, minlength=n_codes)
            pos = np.repeat(pos_values, counts[inverse])
            values.append(auc_of(pos=pos, neg=neg))
        draws.append(float(np.nanmean(values)) if values else float("nan"))
    return draws


def month_block_interval(
    *, month_diffs: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Return a 95% bootstrap interval of the mean over months."""
    draws = month_diffs[rng.integers(0, len(month_diffs), (BOOTSTRAP, len(month_diffs)))].mean(1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def contrasts() -> None:
    """Write absolute scores, contrasts with intervals, and the tails table."""
    shares = nra_shares()
    real = score.real_event_mask()
    rng = np.random.default_rng(0)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[tuple[str, str, bool], pl.DataFrame] = {}
    for nra in (True, False):
        for arm in ARM_NAMES:
            for month in MONTHS:
                if cells_of(arm=arm, month=month):
                    frames[(arm, month, nra)] = arm_month(
                        arm=arm, month=month, nra=nra, shares=shares
                    )
        for arm in TAIL_ARMS:
            for month in MONTHS:
                f = table_p_arm(arm=arm, month=month, nra=nra, shares=shares)
                if f is not None:
                    frames[(f"{arm}+P", month, nra)] = f
    clusters = {
        month: report.cluster_ids(
            frame=frames[("B-mix", month, False)].with_columns(world=pl.lit(-1)), real=real
        )
        - 10_000_000 * i
        for i, month in enumerate(MONTHS)
    }
    absolute, contrast_rows, tail_rows = [], [], []
    for nra in (True, False):
        actual_name = "nra" if nra else "metered"
        for band in (BAND_ALL, BAND_12_36):
            for group in GROUPS:
                for cls in CLASSES:
                    mask_of = lambda f, g=group, c=cls, b=band: (  # noqa: E731
                        ((f["group"] == g) & (f["cls"] == c)).to_numpy()
                        & ((f["lead_band"] == BAND_12_36).to_numpy() if b == BAND_12_36
                           else np.ones(f.height, dtype=bool))
                    )  # fmt: skip
                    for arm in [*ARM_NAMES, *(f"{a}+P" for a in TAIL_ARMS)]:
                        months = [m for m in MONTHS if (arm, m, nra) in frames]
                        if not months:
                            continue
                        per_month = []
                        for m in months:
                            f = frames[(arm, m, nra)]
                            sel = f.filter(pl.Series(mask_of(f)))
                            if sel.height:
                                per_month.append(sel)
                        if not per_month:
                            continue
                        record = {"actual": actual_name, "band": band, "group": group, "cls": cls,
                                  "arm": arm, "months": len(months),
                                  "rows": sum(p.height for p in per_month),
                                  "crps": float(np.mean([p["crps"].mean() for p in per_month]))}  # fmt: skip
                        absolute.append(record)
                        tail_rows.append(
                            {**{k: v for k, v in record.items() if k != "crps"},
                             **{c: float(np.mean([p[c].mean() for p in per_month]))
                                for c in per_month[0].columns
                                if c.startswith(("pin_", "cov_"))}}
                        )  # fmt: skip
                    for name, a_arm, b_arm in CONTRASTS:
                        months = [m for m in MONTHS if (a_arm, m, nra) in frames
                                  and (b_arm, m, nra) in frames]  # fmt: skip
                        diffs, parts_a, parts_b, parts_c = [], [], [], []
                        for m in months:
                            fa, fb = frames[(a_arm, m, nra)], frames[(b_arm, m, nra)]
                            keys = list(heads_mod.KEY_COLUMNS)
                            assert fa[keys].equals(fb[keys]), f"{a_arm} and {b_arm} rows differ"
                            sel = mask_of(fb)
                            if not sel.any():
                                continue
                            parts_a.append(fa.filter(pl.Series(sel)))
                            parts_b.append(fb.filter(pl.Series(sel)))
                            parts_c.append(clusters[m][sel])
                            diffs.append(
                                (
                                    m,
                                    float(parts_a[-1]["crps"].mean())
                                    - float(parts_b[-1]["crps"].mean()),
                                )
                            )  # ty: ignore[unsupported-operator]
                        if not diffs:
                            continue
                        month_diffs = np.array([d for _, d in diffs])
                        cluster = np.concatenate(parts_c)
                        a_all, b_all = pl.concat(parts_a), pl.concat(parts_b)
                        if cls == "clean_real":
                            low, high = month_block_interval(month_diffs=month_diffs, rng=rng)
                            point = float(month_diffs.mean())
                            drop = float("nan")
                        else:
                            point, low, high = report.contrast(a=a_all, b=b_all, cluster=cluster)
                            drop = (
                                report.contrast(a=a_all, b=b_all, cluster=cluster, drop_largest=True)[0]
                                if len(np.unique(cluster)) > 1 else float("nan")
                            )  # fmt: skip
                        loo = [float(np.mean([d for mm, d in diffs if mm != left]))
                               for left in months if len(diffs) > 1]  # fmt: skip
                        contrast_rows.append(
                            {"contrast": name, "a": a_arm, "b": b_arm, "actual": actual_name,
                             "band": band, "group": group, "cls": cls, "months": len(diffs),
                             "rows": a_all.height, "events": len(np.unique(cluster)),
                             "score_a": float(a_all["crps"].mean()), "score_b": float(b_all["crps"].mean()),  # ty: ignore[invalid-argument-type]
                             "diff": point, "low": low, "high": high,
                             "months_a_better": int((month_diffs < 0).sum()),
                             "diff_drop_largest_event": drop,
                             "leave_one_month_min": min(loo) if loo else float("nan"),
                             "leave_one_month_max": max(loo) if loo else float("nan")}
                        )  # fmt: skip
    pl.DataFrame(absolute).write_csv(SCORE_DIR / "absolute.csv", float_precision=4)
    pl.DataFrame(contrast_rows).write_csv(SCORE_DIR / "contrasts.csv", float_precision=4)
    pl.DataFrame(tail_rows).write_csv(SCORE_DIR / "tails.csv", float_precision=4)
    print("written", SCORE_DIR)


def main() -> None:
    """Run one scoring step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("mde", "auc", "contrasts"))
    args = parser.parse_args()
    {"mde": mde, "auc": auc, "contrasts": contrasts}[args.command]()


if __name__ == "__main__":
    main()

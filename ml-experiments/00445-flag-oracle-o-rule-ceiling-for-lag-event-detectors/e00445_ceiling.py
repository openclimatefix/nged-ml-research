"""Score the O-rule arms of experiment 445 against B-mix and experiment 404's stored forecasts.

Row classes are experiment 404's (`e00445_report.load`): `harm_real`, `tgt_real` ("inside real
events"), `clean_real` on the clean world -1, and `harm_inj`, `tgt_inj`, `clean_inj` on the injected
worlds 200 to 202. Groups are the signed-MW six and the MVA-only eight. A contrast is arm minus
B-mix on the same rows (negative favours the arm), with a 95% event-cluster bootstrap interval,
with and without the largest event (real classes). Cells are pooled by appending their rows, with
event clusters kept apart per cell.

```bash
<PYTHONPATH as e00445_orule.py shows> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00445-*/e00445_ceiling.py table
```
"""

import argparse
import json
from pathlib import Path
from typing import Final

import e00445_fit as fit  # ty: ignore[unresolved-import]
import e00445_orule as orule  # ty: ignore[unresolved-import]
import e00445_report as report  # ty: ignore[unresolved-import]
import e00445_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

REFERENCE: Final[str] = "B-mix"
STORED_ARMS: Final[tuple[str, ...]] = ("A0-inj", "B1", "B3", "B6", "S1")
"""Experiment 404 arms whose stored final forecasts are scored beside the O-rule, where present."""
POOLED_CELLS: Final[tuple[tuple[str, int], ...]] = orule.CELLS[:5]
SEED_CELLS: Final[tuple[tuple[str, int], ...]] = orule.CELLS[5:]
GROUPS: Final[tuple[str, ...]] = ("signed_six", "mva_eight")
CLASSES: Final[tuple[str, ...]] = (
    "harm_real",
    "clean_real",
    "tgt_real",
    "harm_inj",
    "clean_inj",
    "tgt_inj",
)
CLASS_LABEL: Final[dict[str, str]] = {
    "harm_real": "real harm",
    "clean_real": "real clean",
    "tgt_real": "inside real events",
    "harm_inj": "injected harm",
    "clean_inj": "injected clean",
    "tgt_inj": "inside injected events",
}
TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
LEVEL_VALUE: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
OUT_DIR: Final[Path] = orule.OUT_DIR


def arm_path(*, arm: str, month: str, seed: int, world: int) -> Path:
    """Return where an arm's forecast frame lives: this experiment's folder or experiment 404's."""
    if arm in orule.ARMS:
        return orule.prediction_path(arm=arm, month=month, seed=seed, world=world)
    return score.prediction_path(arm=arm, month=month, seed=seed, world=world)


def load_cell(
    *, arm: str, month: str, seed: int, worlds: list[int], cell: int
) -> pl.DataFrame | None:
    """Return an arm's scored and classified rows for one cell, or None if the arm is not stored."""
    sites = fit.prep.primaries()["site"].to_list()
    groups = report.group_of(sites=sites)
    parts = []
    for world in worlds:
        path = arm_path(arm=arm, month=month, seed=seed, world=world)
        if not path.exists():
            return None
        parts.append(pl.read_parquet(path).with_columns(world=pl.lit(world)))
    frame = pl.concat(parts, how="diagonal_relaxed")
    frame = frame.with_columns(
        crps=score.pinball_points(frame=frame),
        group=pl.Series(groups[frame["primary"].to_numpy()]),
        cell=pl.lit(cell),
    )
    for name in TAIL_LEVELS:
        level = LEVEL_VALUE[name]
        diff = pl.col("actual") - pl.col(name)
        frame = frame.with_columns(
            **{
                f"pin_{name}": 100.0 * pl.max_horizontal(level * diff, (level - 1.0) * diff),
                f"below_{name}": (pl.col("actual") <= pl.col(name)).cast(pl.Float64),
            }
        )
    injected = pl.col("world") >= 0
    frame = frame.with_columns(
        cls=pl.when(injected & pl.col("tgt_inj"))
        .then(pl.lit("tgt_inj"))
        .when(injected & pl.col("win_inj"))
        .then(pl.lit("harm_inj"))
        .when(injected)
        .then(pl.lit("clean_inj"))
        .when(pl.col("tgt_real") | pl.col("tgt_post"))
        .then(pl.lit("tgt_real"))
        .when(pl.col("win_real"))
        .then(pl.lit("harm_real"))
        .otherwise(pl.lit("clean_real"))
    )
    assert frame["cls"].null_count() == 0
    return frame


def load_pool(
    *, arm: str, cells: tuple[tuple[str, int], ...], reference_rows: dict[int, pl.DataFrame] | None
) -> pl.DataFrame | None:
    """Return an arm's rows pooled over cells (None if any cell is missing), keyed like B-mix."""
    worlds = [*score.SCORING_SEEDS, -1]
    parts = []
    for i, (month, seed) in enumerate(cells):
        frame = load_cell(arm=arm, month=month, seed=seed, worlds=worlds, cell=i)
        if frame is None:
            return None
        if reference_rows is not None:
            ref = reference_rows[i]
            for key in ("primary", "origin", "target", "world"):
                assert np.array_equal(frame[key].to_numpy(), ref[key].to_numpy()), (arm, month, key)
            assert frame["cls"].equals(ref["cls"]), (arm, month, "cls")
        parts.append(frame)
    return pl.concat(parts, how="diagonal_relaxed")


def cluster_ids(*, frame: pl.DataFrame, real: np.ndarray) -> np.ndarray:
    """Return event cluster ids as experiment 404's, except for rows inside real events.

    Experiment 404 clusters a real row by the event its lags fall in, which suits harm rows. A row
    whose target is inside (or within 96 half-hours after) a padded real event is clustered by the
    padded run it belongs to, the latest run that started at or before the target.
    """
    ids = report.cluster_ids(frame=frame, real=real)
    target_rows = ((frame["world"] < 0) & (frame["cls"] == "tgt_real")).to_numpy()
    primary = frame["primary"].to_numpy().astype(np.int64)
    target = frame["target"].to_numpy().astype(np.int64)
    for p in np.unique(primary[target_rows]):
        run = np.cumsum(np.diff(np.concatenate([[0], real[p].astype(np.int8)])) == 1)
        sel = np.flatnonzero(target_rows & (primary == p))
        ids[sel] = -1 - (p * 10_000 + run[np.clip(target[sel], 0, real.shape[1] - 1)])
    return ids


def pooled_clusters(*, frame: pl.DataFrame, real: np.ndarray) -> np.ndarray:
    """Return event cluster ids, kept apart per cell."""
    out = np.zeros(frame.height, dtype=np.int64)
    for cell in frame["cell"].unique().to_list():
        sel = (frame["cell"] == cell).to_numpy()
        ids = cluster_ids(frame=frame.filter(pl.Series(sel)), real=real)
        out[sel] = ids + cell * 10**12
    return out


def summary_rows(*, arm: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the mean score, tail pinball loss, and tail coverage by group and class."""
    aggs = [
        pl.len().alias("rows"),
        pl.col("crps").mean().alias("crps"),
        ((pl.col("actual") >= pl.col("q05")) & (pl.col("actual") <= pl.col("q95")))
        .cast(pl.Float64)
        .mean()
        .alias("inside90"),
    ]
    for name in TAIL_LEVELS:
        aggs += [
            pl.col(f"pin_{name}").mean().alias(f"pin_{name}"),
            pl.col(f"below_{name}").mean().alias(f"below_{name}"),
        ]
    return frame.group_by("group", "cls").agg(aggs).with_columns(arm=pl.lit(arm))


def run_table(*, cells: tuple[tuple[str, int], ...], label: str) -> dict:
    """Print and return the summary and contrasts for the pooled cells."""
    real = score.real_event_mask()
    worlds = [*score.SCORING_SEEDS, -1]
    reference_cells = {
        i: load_cell(arm=REFERENCE, month=m, seed=s, worlds=worlds, cell=i)
        for i, (m, s) in enumerate(cells)
    }
    assert all(v is not None for v in reference_cells.values())
    frames: dict[str, pl.DataFrame] = {}
    for arm in (REFERENCE, *orule.ARMS, *STORED_ARMS):
        pooled = load_pool(arm=arm, cells=cells, reference_rows=reference_cells)  # ty: ignore[invalid-argument-type]
        if pooled is None:
            print(f"[{label}] arm {arm}: not stored for every cell, skipped")
            continue
        frames[arm] = pooled
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    summary = pl.concat([summary_rows(arm=a, frame=f) for a, f in frames.items()])
    print(f"== {label}: mean score and tails by arm, group, class")
    print(summary.sort("group", "cls", "arm"))
    reference = frames[REFERENCE]
    clusters = pooled_clusters(frame=reference, real=real)
    contrasts = []
    for arm, frame in frames.items():
        if arm == REFERENCE:
            continue
        for group in GROUPS:
            for cls in CLASSES:
                mask = ((reference["group"] == group) & (reference["cls"] == cls)).to_numpy()
                if mask.sum() == 0:
                    continue
                for dropped in (False, True) if cls.endswith("_real") else (False,):
                    point, low, high = report.contrast(
                        a=frame.filter(pl.Series(mask)),
                        b=reference.filter(pl.Series(mask)),
                        cluster=clusters[mask],
                        drop_largest=dropped,
                    )
                    n_clusters = len(np.unique(clusters[mask]))
                    contrasts.append(
                        {"set": label, "arm": arm, "group": group, "cls": cls,
                         "largest_removed": dropped, "contrast": point, "low": low, "high": high,
                         "rows": int(mask.sum()), "clusters": n_clusters}
                    )  # fmt: skip
                    print(
                        f"{arm} minus {REFERENCE} | {group} | {CLASS_LABEL[cls]}"
                        f"{' (largest event removed)' if dropped else ''}: "
                        f"{point:+.3f} [{low:+.3f}, {high:+.3f}] ({n_clusters} clusters)"
                    )
    return {"summary": summary, "contrasts": pl.DataFrame(contrasts)}


def per_cell(*, arm: str) -> None:
    """Print each cell's contrast of an arm with B-mix inside real events at the six."""
    real = score.real_event_mask()
    worlds = [*score.SCORING_SEEDS, -1]
    print(f"== per cell: {arm} minus {REFERENCE}, signed six, by class (all events)")
    for i, (month, seed) in enumerate(orule.CELLS):
        a = load_cell(arm=arm, month=month, seed=seed, worlds=worlds, cell=i)
        b = load_cell(arm=REFERENCE, month=month, seed=seed, worlds=worlds, cell=i)
        if a is None or b is None:
            continue
        clusters = cluster_ids(frame=b, real=real)
        text = []
        for cls in ("harm_real", "tgt_real", "harm_inj"):
            mask = ((b["group"] == "signed_six") & (b["cls"] == cls)).to_numpy()
            if mask.sum() == 0:
                continue
            p, lo, hi = report.contrast(
                a=a.filter(pl.Series(mask)), b=b.filter(pl.Series(mask)), cluster=clusters[mask]
            )
            text.append(f"{CLASS_LABEL[cls]} {p:+.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"{month} s{seed}: " + "; ".join(text))


def counts() -> None:
    """Reconcile the real event counts: corrected events, padded runs, and events per month."""
    events = pl.read_parquet(
        score.ML_EXPERIMENTS_DATA_DIR
        / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored"
        / "work"
        / "events_corrected.parquet"
    )
    sites = fit.prep.primaries()["site"].to_list()
    groups = dict(zip(sites, report.group_of(sites=sites), strict=True))
    print(f"corrected events: {events.height}; columns {events.columns}")
    by_group = (
        events.with_columns(group=pl.col("label").replace_strict(groups, default="other"))
        .group_by("group")
        .len()
    )
    print(by_group)
    real = score.real_event_mask()
    runs_per_primary = [
        int((np.diff(np.concatenate([[0], r.astype(np.int8)])) == 1).sum()) for r in real
    ]
    print(
        "padded runs per Primary:",
        dict(zip(sites, runs_per_primary, strict=True)),
        "total",
        sum(runs_per_primary),
    )
    for month, seed in orule.CELLS[:5]:
        frame = pl.read_parquet(
            score.prediction_path(arm=REFERENCE, month=month, seed=seed, world=-1)
        )
        lo, hi = int(frame["target"].min()), int(frame["target"].max())  # ty: ignore[invalid-argument-type]
        overlapping = events.filter((pl.col("onset_idx") <= hi) & (pl.col("end_idx") >= lo))
        in_class = frame.filter(pl.col("tgt_real") | pl.col("tgt_post"))
        print(
            f"{month}: targets {lo}-{hi}; corrected events overlapping {overlapping.height}; "
            f"rows with target in a padded event {in_class.height:,}"
        )


def main() -> None:
    """Print the tables and save them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("table", "counts"))
    args = parser.parse_args()
    if args.command == "counts":
        counts()
        return
    result = run_table(cells=POOLED_CELLS, label="pooled 5 months, seed 0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result["summary"].write_parquet(OUT_DIR / "summary_pooled.parquet")
    result["contrasts"].write_parquet(OUT_DIR / "contrasts_pooled.parquet")
    robust = [
        run_table(cells=(cell,), label=f"{cell[0]} seed {cell[1]} (robustness)")["contrasts"]
        for cell in SEED_CELLS
    ]
    pl.concat(robust).write_parquet(OUT_DIR / "contrasts_seeds.parquet")
    for arm in ("O-hard", "O-soft"):
        per_cell(arm=arm)
    (OUT_DIR / "done.json").write_text(json.dumps({"ok": True}))


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Score E-oracle beside E3, S1, and R-inj on the same rows (no fit).

Reads experiment 428's stored forecasts of R-inj, E3, and S1 (seeds 0 to 2, linked into this
experiment's predictions folder by the `link` command) and this experiment's E-oracle forecasts. Row
sets are experiment 428's (`clean`, `inj_clean`, `inj_harm`, `real_event`, `real_harm`) plus
`real_event_ex_largest` and `real_harm_ex_largest`, which drop the rows tied to the real event with
the most scored target rows (target inside it, or, for the harm rows, a window position inside it).
Scores, contrasts, intervals, and the minimum detectable effect are experiment 428's methods
(`e00428_score`).

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00438-* DATA_PATH_INTERNAL=... \
  uv run --no-sync python ml-experiments/00438-*/e00438_score.py link
  uv run --no-sync python ml-experiments/00438-*/e00438_score.py score
```
"""

import argparse
from collections.abc import Callable
from typing import Final

import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import e00438_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

REFERENCES: Final[tuple[str, ...]] = ("R-inj", "E3", "S1")
ARMS: Final[tuple[str, ...]] = (*REFERENCES, run.ARM)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "O1 E-oracle - E3": (run.ARM, "E3"),
    "O2 E-oracle - S1": (run.ARM, "S1"),
    "O3 E-oracle - R-inj": (run.ARM, "R-inj"),
    "O4 S1 - R-inj": ("S1", "R-inj"),
    "O5 E3 - R-inj": ("E3", "R-inj"),
}
EXTRA_FLAGS: Final[tuple[str, ...]] = ("big_tgt", "big_win")
ROW_SETS: Final[dict[str, tuple[str, Callable[[dict[str, np.ndarray]], np.ndarray]]]] = {
    **score.ROW_SETS,
    "real_event_ex_largest": ("real", lambda f: f["tgt_real"] & ~f["big_tgt"]),
    "real_harm_ex_largest": ("real", lambda f: ~f["tgt_post"] & f["win_real"] & ~f["big_win"]),
}


def link() -> None:
    """Link experiment 428's stored forecasts of the reference arms into this predictions folder."""
    target = run.WORK_DIR / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for arm in REFERENCES:
        source = data.WORK_DIR / "predictions" / arm
        if not (target / arm).exists():
            (target / arm).symlink_to(source, target_is_directory=True)


def event_runs(*, real: np.ndarray) -> dict[int, np.ndarray]:
    """Return, for each Primary, the (start, stop) grid positions of its padded real events."""
    out = {}
    for p in range(real.shape[0]):
        edges = np.diff(np.concatenate([[0], real[p].astype(np.int8), [0]]))
        out[p] = np.stack([np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]], axis=1)
    return out


def grid_index(*, us: np.ndarray) -> np.ndarray:
    """Return the dense-grid position of microsecond timestamps."""
    return (us - prep422.GRID_START_US) // prep422.STEP_US


def largest_event(
    *, months: list[str], seeds: tuple[int, ...], runs: dict[int, np.ndarray]
) -> tuple[int, int, int, dict]:
    """Return (primary, start, stop) of the real event with the most scored target rows, and counts."""
    counts: dict[tuple[int, int, int], int] = {}
    for month in months:
        frame = score.read_arm(arm="R-inj", month=month, world=0, seeds=seeds, factor=1.0)
        keep = frame["tgt_real"].to_numpy().astype(bool) & np.isfinite(frame["actual"].to_numpy())
        primary = frame["primary"].to_numpy()[keep]
        target = grid_index(us=frame["valid_us"].to_numpy()[keep])
        for p, t in zip(primary, target, strict=True):
            hit = runs[int(p)]
            i = np.searchsorted(hit[:, 0], t, side="right") - 1
            if i >= 0 and t < hit[i, 1]:
                key = (int(p), int(hit[i, 0]), int(hit[i, 1]))
                counts[key] = counts.get(key, 0) + 1
    best = max(counts, key=counts.__getitem__)
    return (*best, counts)  # ty: ignore[invalid-return-type]


def extra_flags(*, frame: pl.DataFrame, big: tuple[int, int, int]) -> pl.DataFrame:
    """Add `big_tgt` and `big_win`: the row's target, or a position of its windows, is in `big`."""
    p, start, stop = big
    primary = frame["primary"].to_numpy()
    target = grid_index(us=frame["valid_us"].to_numpy())
    origin = grid_index(us=frame["origin_us"].to_numpy())
    big_tgt = (primary == p) & (target >= start) & (target < stop)
    big_win = np.zeros(frame.height, dtype=bool)
    rows = np.nonzero(primary == p)[0]
    if len(rows):
        index = win.window_indices(
            origin=torch.from_numpy(origin[rows]), target=torch.from_numpy(target[rows]), days=3
        ).numpy()
        usable = index <= (origin[rows] - win.NEWEST_AGE)[:, None, None]
        big_win[rows] = ((index >= start) & (index < stop) & usable).reshape(len(rows), -1).any(1)
    return frame.with_columns(big_tgt=pl.Series(big_tgt), big_win=pl.Series(big_win))


def collect(
    *, months: list[str], seeds: tuple[int, ...], factor: float, big: tuple[int, int, int]
) -> pl.DataFrame:
    """Return per arm, month, world, row set, group, and origin day: row count and score sums."""
    records = []
    flag_names = [*score._flags(), *EXTRA_FLAGS]
    for month in months:
        for set_name, worlds in score.WORLD_SETS.items():
            for arm in ARMS:
                for world in worlds:
                    frame = score.read_arm(
                        arm=arm, month=month, world=world, seeds=seeds, factor=factor
                    )
                    frame = extra_flags(frame=frame, big=big)
                    day = (frame["origin_us"].to_numpy() // (86_400 * 1_000_000)).astype(int)
                    groups = score.group_of(primary=frame["primary"].to_numpy())
                    actual = frame["actual"].to_numpy()
                    flags = {n: frame[n].to_numpy().astype(bool) for n in flag_names}
                    for row_set, (which, rule) in ROW_SETS.items():
                        if which != set_name:
                            continue
                        keep = rule(flags) & np.isfinite(actual)
                        for group, in_group in groups.items():
                            sel = keep & in_group
                            for d in np.unique(day[sel]):
                                m = sel & (day == d)
                                record = {
                                    "arm": arm, "month": month, "world": world, "row_set": row_set,
                                    "group": group, "day": int(d), "n": int(m.sum()),
                                    "score": float(frame["score"].to_numpy()[m].sum()),
                                }  # fmt: skip
                                for s in seeds:
                                    record[f"score_s{s}"] = float(
                                        frame[f"score_s{s}"].to_numpy()[m].sum()
                                    )
                                for tail in score.TAILS:
                                    q = frame[tail].to_numpy()[m]
                                    level = score.LEVEL_OF[tail]
                                    err = actual[m] - q
                                    record[f"pin_{tail}"] = float(
                                        np.maximum(level * err, (level - 1) * err).sum()
                                    )
                                    record[f"cov_{tail}"] = float((actual[m] <= q).sum())
                                records.append(record)
    return pl.DataFrame(records)


def main() -> None:
    """Link the references, or score the arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("link", "score"))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    parser.add_argument("--raw", action="store_true", help="no 1.15 widening")
    args = parser.parse_args()
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    if args.command == "link":
        link()
        return
    seeds = tuple(args.seeds)
    score.CONTRASTS = CONTRASTS  # ty: ignore[invalid-assignment]
    runs = event_runs(real=data.real_event_mask())
    p, start, stop, counts = largest_event(months=args.months, seeds=seeds, runs=runs)
    print(
        f"real events with scored target rows: {len(counts)}; scored target rows: {sum(counts.values())}; "
        f"largest: Primary {data.sites()[p]}, {counts[(p, start, stop)]} rows"
    )
    table = collect(
        months=args.months,
        seeds=seeds,
        factor=1.0 if args.raw else score.WIDENING,
        big=(p, start, stop),
    )
    out = run.WORK_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out / "blocks.parquet")
    with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=60):
        absolute = score.summarise(table=table, seeds=seeds)
        absolute.write_parquet(out / "absolute.parquet")
        print(absolute.sort("row_set", "group", "arm"))
        mde = score.minimum_detectable_effect(table=table, seeds=seeds)
        mde.write_parquet(out / "mde.parquet")
        print(mde.sort("row_set", "group"))
        contrasts = score.contrast_table(table=table, rng=np.random.default_rng(0))
        contrasts.write_parquet(out / "contrasts.parquet")
        print(contrasts.sort("contrast", "row_set", "group").drop("fold_differences"))


if __name__ == "__main__":
    main()

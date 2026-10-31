# ruff: noqa: E501
"""Score the ensembling control of experiment 445.

Question: is B-mix's edge over a single head an ensembling gain? Heads of one kind are averaged
(`e00445_ensemble`), and the averages are scored against the single heads and B-mix on experiment
404's row classes, with event-cluster intervals.

Part 1 (2026-06, fit seeds 0 to 2): row-aligned averages over seeds. "Single lagged" is the mean
score of A0-inj seeds 0 to 2, "B-mix" the mean of the seed-matched B-mix arms, "E-lag-2" the mean of
the three pairs, "E-lag-3" the 3-way mixture, and likewise for the lagless heads.
Part 2 (all five months, seed 0): lagless-head mixtures against the single lagless head.

```bash
<PYTHONPATH as e00445_orule.py shows> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00445-*/e00445_ensemble_table.py
```
"""

from typing import Final

import e00445_ceiling as ceiling  # ty: ignore[unresolved-import]
import e00445_ensemble as ens  # ty: ignore[unresolved-import]
import e00445_orule as orule  # ty: ignore[unresolved-import]
import e00445_report as report  # ty: ignore[unresolved-import]
import e00445_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MONTH: Final[str] = ens.LAGGED_MONTH
SEEDS: Final[tuple[int, ...]] = ens.LAGGED_SEEDS
WORLDS: Final[list[int]] = [*score.SCORING_SEEDS, -1]
GROUPS: Final[tuple[str, ...]] = ceiling.GROUPS
CLASSES: Final[tuple[str, ...]] = ("clean_real", "harm_real", "tgt_real", "clean_inj", "harm_inj")


def crps_of(*, arm: str, seed: int, month: str = MONTH) -> pl.DataFrame:
    """Return an arm's classified rows (month, seed-0 row set) from the loader."""
    frame = ceiling.load_cell(arm=arm, month=month, seed=seed, worlds=WORLDS, cell=0)
    assert frame is not None, f"{arm} seed {seed} month {month} is not stored"
    return frame


def averaged(*, arms: list[tuple[str, int]], ref: pl.DataFrame) -> np.ndarray:
    """Return the row-wise mean score of several arm files, after checking the rows agree."""
    total = np.zeros(ref.height)
    for arm, seed in arms:
        frame = crps_of(arm=arm, seed=seed)
        for key in ("primary", "origin", "target", "world"):
            assert np.array_equal(frame[key].to_numpy(), ref[key].to_numpy()), (arm, seed, key)
        assert frame["cls"].equals(ref["cls"]), (arm, seed, "cls")
        total += frame["crps"].to_numpy()
    return total / len(arms)


def families(*, ref: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the row-wise score of each family of arms in part 1."""
    pairs = ("01", "02", "12")
    return {
        "single lagged (A0-inj)": averaged(arms=[("A0-inj", s) for s in SEEDS], ref=ref),
        "B-mix (seed-matched)": averaged(arms=[("B-mix", s) for s in SEEDS], ref=ref),
        "lagged x2": averaged(arms=[(f"E-lag-{p}", 0) for p in pairs], ref=ref),
        "lagged x3": averaged(arms=[("E-lag-012", 0)], ref=ref),
        "single lagless": averaged(arms=[(f"LL-s{s}", 0) for s in SEEDS], ref=ref),
        "lagless x2": averaged(arms=[(f"E-ll-{p}", 0) for p in pairs], ref=ref),
        "lagless x3": averaged(arms=[("E-ll-012", 0)], ref=ref),
        "lagless x5": averaged(arms=[("E-ll-01234", 0)], ref=ref),
        "O-floor (post hoc)": averaged(arms=[("O-floor", 0)], ref=ref),
    }


def part_one() -> None:
    """Print part 1: means, contrasts, and the gain ratio, at 2026-06."""
    real = score.real_event_mask()
    ref = crps_of(arm="B-mix", seed=0)
    fam = families(ref=ref)
    clusters = ceiling.cluster_ids(frame=ref, real=real)
    pl.Config.set_tbl_rows(100)
    print(f"== part 1: {MONTH}, fit seeds 0 to 2, mean score by family")
    rows = []
    for group in GROUPS:
        for cls in CLASSES:
            mask = ((ref["group"] == group) & (ref["cls"] == cls)).to_numpy()
            rows.append(
                {"group": group, "cls": cls, "rows": int(mask.sum()),
                 **{k: round(float(v[mask].mean()), 3) for k, v in fam.items()}}
            )  # fmt: skip
    print(pl.DataFrame(rows))
    pairs = [
        ("B-mix (seed-matched)", "single lagged (A0-inj)"),
        ("lagged x2", "single lagged (A0-inj)"),
        ("lagged x3", "single lagged (A0-inj)"),
        ("B-mix (seed-matched)", "lagged x2"),
        ("B-mix (seed-matched)", "lagged x3"),
        ("lagless x2", "single lagless"),
        ("lagless x3", "single lagless"),
        ("lagless x5", "single lagless"),
        ("B-mix (seed-matched)", "lagless x2"),
        ("O-floor (post hoc)", "B-mix (seed-matched)"),
    ]
    out = []
    for group in GROUPS:
        for cls in CLASSES:
            mask = ((ref["group"] == group) & (ref["cls"] == cls)).to_numpy()
            if not mask.any():
                continue
            for a, b in pairs:
                for dropped in (False, True) if cls.endswith("_real") else (False,):
                    point, low, high = report.contrast(
                        a=pl.DataFrame({"crps": fam[a][mask]}),
                        b=pl.DataFrame({"crps": fam[b][mask]}),
                        cluster=clusters[mask],
                        drop_largest=dropped,
                    )
                    out.append(
                        {"group": group, "cls": cls, "a": a, "b": b, "largest_removed": dropped,
                         "contrast": point, "low": low, "high": high}
                    )  # fmt: skip
                    note = " (largest removed)" if dropped else ""
                    print(
                        f"{group} | {cls}{note}: {a} minus {b}: {point:+.3f} [{low:+.3f}, {high:+.3f}]"
                    )
    pl.DataFrame(out).write_parquet(orule.OUT_DIR / "ensemble_part1.parquet")
    for cls in ("clean_real", "tgt_real", "clean_inj", "harm_inj"):
        mask = ((ref["group"] == "signed_six") & (ref["cls"] == cls)).to_numpy()
        gb = fam["single lagged (A0-inj)"][mask].mean() - fam["B-mix (seed-matched)"][mask].mean()
        ge = fam["single lagged (A0-inj)"][mask].mean() - fam["lagged x2"][mask].mean()
        g3 = fam["single lagged (A0-inj)"][mask].mean() - fam["lagged x3"][mask].mean()
        print(
            f"gain ratio signed six {cls}: G_B {gb:.3f}, G_E(x2) {ge:.3f}, ratio {ge / gb:.2f}; G_E(x3) {g3:.3f}, ratio {g3 / gb:.2f}"
        )


def part_two() -> None:
    """Print part 2: lagless-head mixtures against the single lagless head, five months pooled."""
    real = score.real_event_mask()
    cells = ceiling.POOLED_CELLS
    ref_cells = {
        i: ceiling.load_cell(arm="B-mix", month=m, seed=s, worlds=WORLDS, cell=i)
        for i, (m, s) in enumerate(cells)
    }
    arms = ["LL-s0", "E-ll-01", "E-ll-012", "E-ll-01234", "B-mix", "A0-inj", "O-floor"]
    frames = {a: ceiling.load_pool(arm=a, cells=cells, reference_rows=ref_cells) for a in arms}  # ty: ignore[invalid-argument-type]
    ref = frames["B-mix"]
    assert ref is not None
    clusters = ceiling.pooled_clusters(frame=ref, real=real)
    print("== part 2: five months, seed 0, mean score by arm")
    rows = []
    for group in GROUPS:
        for cls in CLASSES:
            mask = ((ref["group"] == group) & (ref["cls"] == cls)).to_numpy()
            rows.append(
                {"group": group, "cls": cls, "rows": int(mask.sum()),
                 **{a: round(float(f["crps"].to_numpy()[mask].mean()), 3) for a, f in frames.items() if f is not None}}
            )  # fmt: skip
    print(pl.DataFrame(rows))
    pairs = [("E-ll-01", "LL-s0"), ("E-ll-012", "LL-s0"), ("E-ll-01234", "LL-s0"),
             ("B-mix", "LL-s0"), ("B-mix", "E-ll-01234"), ("B-mix", "A0-inj")]  # fmt: skip
    out = []
    for group in GROUPS:
        for cls in CLASSES:
            mask = ((ref["group"] == group) & (ref["cls"] == cls)).to_numpy()
            for a, b in pairs:
                for dropped in (False, True) if cls.endswith("_real") else (False,):
                    point, low, high = report.contrast(
                        a=frames[a].filter(pl.Series(mask)),  # ty: ignore[unresolved-attribute]
                        b=frames[b].filter(pl.Series(mask)),  # ty: ignore[unresolved-attribute]
                        cluster=clusters[mask],
                        drop_largest=dropped,
                    )
                    out.append(
                        {"group": group, "cls": cls, "a": a, "b": b, "largest_removed": dropped,
                         "contrast": point, "low": low, "high": high}
                    )  # fmt: skip
                    note = " (largest removed)" if dropped else ""
                    print(
                        f"{group} | {cls}{note}: {a} minus {b}: {point:+.3f} [{low:+.3f}, {high:+.3f}]"
                    )
    pl.DataFrame(out).write_parquet(orule.OUT_DIR / "ensemble_part2.parquet")


def main() -> None:
    """Print both parts."""
    part_one()
    part_two()


if __name__ == "__main__":
    main()

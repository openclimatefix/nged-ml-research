# ruff: noqa: E501
"""Part 1 of experiment 522: how far the stored Primary embeddings moved from their random start.

Reads the stored pretrained weights of experiment 448's L0PT (4 folds x 5 seeds), experiment 425's
arm A (the same folds and seeds), the fine-tuned L0PT networks of experiment 448, the pretrained
`MIX` weights of experiment 511, and one fine-tuned L-1d network of experiment 511. Regenerates
each seed's random start, then reports per-dimension spread, singular values, displacement from the
start, per-Primary movement, and rank correlations of the 120-pair distance matrices. Writes
`work/part1.json`. CPU only, no fitting.

```bash
DATA_PATH_INTERNAL=<private data folder> OMP_NUM_THREADS=2 \
    uv run --no-sync python ml-experiments/00522-*/e00522_part1.py
```
"""

import itertools
import json
from typing import Final

import e00522_common as common  # ty: ignore[unresolved-import]
import numpy as np

BOOTSTRAPS: Final[int] = 10_000
PERMUTATIONS: Final[int] = 2_000
VARIANCE_SHARE: Final[float] = 0.9


def effective_dimensions(*, embedding: np.ndarray) -> dict[str, object]:
    """Return the singular values of the centred 16 x 8 matrix and the dimensions for 90% of variance."""
    centred = embedding - embedding.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    share = np.cumsum(singular**2) / np.sum(singular**2)
    return {
        "singular_values": singular.round(3).tolist(),
        "dims_for_90pct": int(np.searchsorted(share, VARIANCE_SHARE) + 1),
        "dim_std": embedding.std(axis=0, ddof=1).round(3).tolist(),
    }


def bootstrap_mean(*, values: np.ndarray, seed: int) -> dict[str, float]:
    """Return the mean and the 95% percentile bootstrap interval over the cells (fold, seed)."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
        "n_cells": len(values),
    }


def load_all() -> dict[str, dict[tuple[str, int], np.ndarray]]:
    """Return every stored embedding, keyed by network family and (fold, seed)."""
    out: dict[str, dict[tuple[str, int], np.ndarray]] = {
        "L0PT": {},
        "A": {},
        "L0PT_finetuned": {},
    }
    for fold, seed in itertools.product(common.FOLDS, common.SEEDS):
        out["L0PT"][fold, seed] = common.embedding_of_state(
            path=common.L0PT_PRETRAINED / f"{fold}_s{seed}.pt"
        )
        out["A"][fold, seed] = common.embedding_of_state(
            path=common.A_PRETRAINED / f"{fold}_s{seed}.pt"
        )
        out["L0PT_finetuned"][fold, seed] = common.embedding_of_state(
            path=common.E448 / "work" / "models" / "L0PT" / fold / f"s{seed}" / "model.pt"
        )
    return out


def displacement_table(
    *, tables: dict[str, dict[tuple[str, int], np.ndarray]]
) -> dict[str, object]:
    """Return the displacement from the regenerated start for every family, as shares of the start's norm."""
    out: dict[str, object] = {}
    for family, cells in tables.items():
        shares = []
        per_primary = []
        for (_fold, seed), emb in cells.items():
            start = common.regenerated_start(seed=seed)
            shares.append(np.linalg.norm(emb - start) / np.linalg.norm(start))
            per_primary.append(np.linalg.norm(emb - start, axis=1) / np.linalg.norm(start, axis=1))
        out[family] = {
            "displacement_share_of_start_norm": bootstrap_mean(values=np.array(shares), seed=1),
            "per_primary_mean_share": np.mean(per_primary, axis=0).round(3).tolist(),
            "per_primary_min_cell_share": np.min(per_primary, axis=0).round(3).tolist(),
            "per_primary_max_cell_share": np.max(per_primary, axis=0).round(3).tolist(),
        }
    return out


def alignment(*, tables: dict[str, dict[tuple[str, int], np.ndarray]]) -> dict[str, object]:
    """Return rank correlations of the 120-pair distance matrices, with and without the start removed."""

    def moved(*, family: str, fold: str, seed: int) -> np.ndarray:
        return tables[family][fold, seed] - common.regenerated_start(seed=seed)

    def corr(*, a: np.ndarray, b: np.ndarray) -> float:
        return common.spearman(
            a=common.distance_vector(embedding=a), b=common.distance_vector(embedding=b)
        )

    out: dict[str, object] = {}
    same_seed = [
        (f1, f2, s) for s in common.SEEDS for f1, f2 in itertools.combinations(common.FOLDS, 2)
    ]
    out["L0PT_same_seed_across_folds_stored"] = _summary(
        values=[corr(a=tables["L0PT"][f1, s], b=tables["L0PT"][f2, s]) for f1, f2, s in same_seed]
    )
    out["L0PT_same_seed_across_folds_movement"] = _summary(
        values=[
            corr(a=moved(family="L0PT", fold=f1, seed=s), b=moved(family="L0PT", fold=f2, seed=s))
            for f1, f2, s in same_seed
        ]
    )
    cross_seed = [
        (fold, s1, s2)
        for fold in common.FOLDS
        for s1, s2 in itertools.combinations(common.SEEDS, 2)
    ]
    out["L0PT_across_seeds_same_fold_stored"] = _summary(
        values=[
            corr(a=tables["L0PT"][fold, s1], b=tables["L0PT"][fold, s2])
            for fold, s1, s2 in cross_seed
        ]
    )
    out["L0PT_across_seeds_same_fold_movement"] = _summary(
        values=[
            corr(
                a=moved(family="L0PT", fold=fold, seed=s1),
                b=moved(family="L0PT", fold=fold, seed=s2),
            )
            for fold, s1, s2 in cross_seed
        ]
    )
    cells = list(itertools.product(common.FOLDS, common.SEEDS))
    out["A_vs_L0PT_same_cell_stored"] = _summary(
        values=[corr(a=tables["A"][c], b=tables["L0PT"][c]) for c in cells]
    )
    out["A_vs_L0PT_same_cell_movement"] = _summary(
        values=[
            corr(
                a=moved(family="A", fold=c[0], seed=c[1]),
                b=moved(family="L0PT", fold=c[0], seed=c[1]),
            )
            for c in cells
        ]
    )
    out["L0PT_pretrained_vs_finetuned_stored"] = _summary(
        values=[corr(a=tables["L0PT"][c], b=tables["L0PT_finetuned"][c]) for c in cells]
    )
    null = common.spearman_null(
        a=tables["L0PT"]["2026-01", 0],
        b=tables["L0PT"]["2026-01", 1],
        n_permutations=PERMUTATIONS,
        seed=7,
    )
    out["permutation_null_spearman_120_pairs"] = {
        "mean": float(null.mean()),
        "sd": float(null.std()),
        "q95": float(np.quantile(null, 0.95)),
        "n_permutations": PERMUTATIONS,
    }
    return out


def _summary(*, values: list[float]) -> dict[str, float]:
    """Return the number of matrix pairs and the range and mean of their rank correlations."""
    v = np.array(values)
    return {
        "n_matrix_pairs": len(v),
        "pairs_per_matrix": 120,
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def drift_table(*, tables: dict[str, dict[tuple[str, int], np.ndarray]]) -> dict[str, object]:
    """Return how far daily fine-tuning moved each Primary's vector, from the pretrained vector."""
    cells = list(itertools.product(common.FOLDS, common.SEEDS))
    share = []
    per_primary = []
    for c in cells:
        pre, fine = tables["L0PT"][c], tables["L0PT_finetuned"][c]
        share.append(np.linalg.norm(fine - pre) / np.linalg.norm(pre))
        per_primary.append(np.linalg.norm(fine - pre, axis=1) / np.linalg.norm(pre, axis=1))
    out: dict[str, object] = {
        "finetune_move_share_of_pretrained_norm": bootstrap_mean(values=np.array(share), seed=2),
        "per_primary_mean_share": np.mean(per_primary, axis=0).round(4).tolist(),
    }
    mix = common.embedding_of_state(
        path=common.E511 / "work" / "pretrained" / "MIX" / "2026-01_s0.pt"
    )
    l1d = common.embedding_of_state(
        path=common.E511 / "work" / "models" / "L-1d" / "2026-01_d10_s0.pt"
    )
    start = common.regenerated_start(seed=0)
    pre = tables["L0PT"]["2026-01", 0]
    out["L-1d_2026-01_d10_s0"] = {
        "from_L0PT_pretrained_share": float(np.linalg.norm(l1d - pre) / np.linalg.norm(pre)),
        "L0PT_pretrained_from_start_share": float(
            np.linalg.norm(pre - start) / np.linalg.norm(start)
        ),
        "from_start_share": float(np.linalg.norm(l1d - start) / np.linalg.norm(start)),
        "MIX_pretrained_from_start_share": float(
            np.linalg.norm(mix - start) / np.linalg.norm(start)
        ),
    }
    return out


def main() -> None:
    """Compute and write Part 1."""
    tables = load_all()
    spread = {
        f"{fam}|{fold}|s{seed}": effective_dimensions(embedding=emb)
        for fam, cells in tables.items()
        if fam != "L0PT_finetuned"
        for (fold, seed), emb in cells.items()
    }
    starts = {
        f"start|s{s}": effective_dimensions(embedding=common.regenerated_start(seed=s))
        for s in common.SEEDS
    }
    movement_dims = {
        f"L0PT|{fold}|s{seed}": effective_dimensions(
            embedding=tables["L0PT"][fold, seed] - common.regenerated_start(seed=seed)
        )
        for fold, seed in itertools.product(common.FOLDS, common.SEEDS)
    }
    report = {
        "regeneration_check_corr_stored_vs_start_same_seed": [
            float(
                np.corrcoef(
                    tables["L0PT"][fold, seed].ravel(),
                    common.regenerated_start(seed=seed).ravel(),
                )[0, 1]
            )
            for fold, seed in itertools.product(common.FOLDS, common.SEEDS)
        ],
        "dimensions_stored_and_start": {"stored": spread, "start": starts},
        "dimensions_of_movement": movement_dims,
        "displacement": displacement_table(tables=tables),
        "alignment": alignment(tables=tables),
        "drift": drift_table(tables=tables),
    }
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "part1.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps({k: v for k, v in report.items() if k != "dimensions_of_movement"}, indent=1)[
            :6000
        ]
    )


if __name__ == "__main__":
    main()

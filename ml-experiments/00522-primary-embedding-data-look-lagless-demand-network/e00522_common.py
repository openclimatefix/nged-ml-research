"""Shared pieces of experiment 522: stored-weight paths, embedding loaders, and distance helpers.

The Primary embedding is the 16 x 8 table `embedding.weight` of experiment 397's `PlainMdn` (the
files of experiments 448 and 510 prefix it with `net.`). Its random start is the first draw after
`torch.manual_seed(seed)`, because the embedding is the first layer built, so the start is
regenerated rather than stored.
"""

from pathlib import Path
from typing import Final

import numpy as np
import torch  # ty: ignore[unresolved-import]
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from studies.sources import ML_EXPERIMENTS_DATA_DIR

N_PRIMARIES: Final[int] = 16
EMBEDDING_WIDTH: Final[int] = 8
FOLDS: Final[tuple[str, ...]] = ("2025-10", "2026-01", "2026-03", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00522-primary-embedding-data-look-lagless-demand-network" / "work"
)


def _folder(*, prefix: str) -> Path:
    """Return the private data folder of the experiment whose name starts with `prefix`."""
    (match,) = ML_EXPERIMENTS_DATA_DIR.glob(f"{prefix}-*")
    return match


E448: Final[Path] = _folder(prefix="00448")
E425: Final[Path] = _folder(prefix="00425")
E511: Final[Path] = _folder(prefix="00511")
L0PT_PRETRAINED: Final[Path] = E448 / "work" / "pretrained" / "L0PT"
A_PRETRAINED: Final[Path] = E425 / "work" / "pretrained" / "A"


def embedding_of_state(*, path: Path) -> np.ndarray:
    """Return the 16 x 8 embedding of a stored state dict (the key has a `net.` prefix in some files)."""
    state = torch.load(path, map_location="cpu")
    key = "embedding.weight" if "embedding.weight" in state else "net.embedding.weight"
    return state[key].numpy().astype(np.float64)


def regenerated_start(*, seed: int) -> np.ndarray:
    """Return the random start of the embedding of a network built after `torch.manual_seed(seed)`."""
    torch.manual_seed(seed)
    return (
        torch.nn.Embedding(N_PRIMARIES, EMBEDDING_WIDTH).weight.detach().numpy().astype(np.float64)
    )


def distance_vector(*, embedding: np.ndarray) -> np.ndarray:
    """Return the 120 pairwise Euclidean distances between the 16 rows (upper triangle, row-major)."""
    return pdist(embedding)


def spearman(*, a: np.ndarray, b: np.ndarray) -> float:
    """Return the Spearman rank correlation of two vectors."""
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def permuted_distance_vector(*, embedding: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Return the distance vector after relabelling the 16 Primaries by `order`."""
    return pdist(embedding[order])


def spearman_null(*, a: np.ndarray, b: np.ndarray, n_permutations: int, seed: int) -> np.ndarray:
    """Return the Spearman correlations of distance matrix `a` against relabelled copies of `b`.

    Args:
        a: A 16 x 16-derived embedding matrix (rows are Primaries).
        b: A second embedding matrix.
        n_permutations: How many random relabellings of the 16 Primaries to draw.
        seed: The seed of the relabelling draws.

    Returns:
        One correlation per relabelling, which is the null distribution of the correlation when the
        order of the Primaries carries no information.
    """
    rng = np.random.default_rng(seed)
    da = distance_vector(embedding=a)
    return np.array(
        [
            spearman(
                a=da,
                b=permuted_distance_vector(embedding=b, order=rng.permutation(N_PRIMARIES)),
            )
            for _ in range(n_permutations)
        ]
    )

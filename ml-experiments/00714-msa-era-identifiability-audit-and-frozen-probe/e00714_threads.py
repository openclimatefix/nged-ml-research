"""Experiment 714: limit BLAS and OpenMP to one thread and Polars to 2. Import before numpy."""

import os
from typing import Final

THREAD_VARIABLES: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"
)  # fmt: skip
for _name in THREAD_VARIABLES:
    os.environ[_name] = "1"
os.environ["POLARS_MAX_THREADS"] = "2"

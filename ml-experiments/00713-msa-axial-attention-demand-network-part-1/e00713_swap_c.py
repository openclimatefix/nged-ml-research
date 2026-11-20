# ruff: noqa: E501
"""Experiment 713: swap (c), the stored Candidate with ERA5 at its lag times (design 4.3).

Usage: e00713_swap_c.py MONTH SEED [--device cuda]

The stored Candidate states (`00667-*/P7_D50_<month>/work/models/P5/`) predict with
`e00560_data.raw_table(source="pre")` attached as the expected-input table in place of the `ft` table.
The script runs experiment 667's own prediction code (`e00667_cand.run_fit`) with `E00667_ROOT` set to
this experiment's `swap_c` folder. It links the stored model states and the stored prediction tables
into that folder (a link inside this experiment's data folder only), so that `run_fit` loads the states
and skips the fine-tune; nothing under `00667-*/` is written. The forecasts are
`swap_c/P7_D50_<month>/work/fits/P5/<month>_d0_s<seed>.parquet`.
"""

import argparse
import os
import sys
from pathlib import Path

import e00713_inputs as inp
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT_SWAP: Path = ML_EXPERIMENTS_DATA_DIR / inp.NAME / "swap_c"


def link_stored(*, month: str, seed: int) -> None:
    """Link experiment 667's stored states and prediction table into the swap folder (read-only sources)."""
    stored = inp.ROOT_667 / f"P7_D50_{month}" / "work" / "models" / "P5" / f"{month}_d0_s{seed}.pt"
    assert stored.exists(), stored
    dest = ROOT_SWAP / f"P7_D50_{month}" / "work" / "models" / "P5" / stored.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.symlink_to(stored)
    table = inp.ROOT_667 / "tables" / "predict" / f"T_{month}_s{seed}.npz"
    if table.exists():
        t_dest = ROOT_SWAP / "tables" / "predict" / table.name
        t_dest.parent.mkdir(parents=True, exist_ok=True)
        if not t_dest.exists():
            t_dest.symlink_to(table)


def main() -> int:
    """Run the Candidate's prediction with the ERA5 table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    month, seed = args.month, args.seed
    link_stored(month=month, seed=seed)
    os.environ.update(
        E00667_ROOT=str(ROOT_SWAP), E00667_MONTH=month, E00667_PHASE="predict",
        CHAIN667=str(inp.chain_dir(root=inp.ROOT_667, month=month)),
    )  # fmt: skip
    sys.argv = [
        "e00667_cand.py",
        "P7+D50",
        "fit",
        "--work-suffix",
        f"_{month}",
        "--months",
        month,
        "--seeds",
        str(seed),
    ]
    import e00667_cand as cand  # ty: ignore[unresolved-import]

    e = cand.e
    import e00560_data as d560  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    original = e.attach_tables

    def attach_pre(*, standardiser) -> None:  # noqa: ANN001
        """Attach the `ft` table (its run origins feed the lag-age checks), then replace the table with `pre`."""
        original(standardiser=standardiser)
        cols, _ = d560.raw_table(source="pre", cache=e.W560 / "expected")
        table = d560.standardised_table(cols=cols, standardiser=standardiser)
        e.CTX["table_ft"] = torch.from_numpy(table).to("cuda")
        del cols

    e.attach_tables = attach_pre
    cand.run_fit(device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

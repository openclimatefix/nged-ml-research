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
import hashlib
import os
import sys
from pathlib import Path

import e00713_inputs as inp
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT_SWAP: Path = ML_EXPERIMENTS_DATA_DIR / inp.NAME / "swap_c"


def link_file(*, source: Path, dest: Path) -> None:
    """Make `dest` a link to `source`, replacing a dangling link; the parent folder is a real directory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dest) and not dest.exists():
        dest.unlink()  # a dangling link: `exists()` is False although the name is taken
    if not dest.exists():
        dest.symlink_to(source)


def link_tree(*, source: Path, dest: Path) -> Path:
    """Mirror `source` as real directories under `dest`, with a link for every file; return `dest`.

    No write through a path under `dest` can reach a folder of experiment 667, and a rename of a file
    replaces its link and not its target.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        target = dest / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            link_file(source=path, dest=target)
    assert not str(dest.resolve()).startswith(str(inp.ROOT_667.resolve())), dest
    return dest


def link_stored(*, month: str, seed: int) -> None:
    """Link experiment 667's stored states, prediction table, and standardiser hash into the swap folder."""
    fold = inp.ROOT_667 / f"P7_D50_{month}"
    stored = fold / "work" / "models" / "P5" / f"{month}_d0_s{seed}.pt"
    assert stored.exists(), stored
    link_file(
        source=stored, dest=ROOT_SWAP / f"P7_D50_{month}" / "work" / "models" / "P5" / stored.name
    )
    sha = fold / f"standardiser_{month}.sha256"  # `check_fingerprint` of the predict phase reads it
    assert sha.exists(), sha
    link_file(source=sha, dest=ROOT_SWAP / f"P7_D50_{month}" / sha.name)
    table = inp.ROOT_667 / "tables" / "predict" / f"T_{month}_s{seed}.npz"
    if table.exists():
        link_file(source=table, dest=ROOT_SWAP / "tables" / "predict" / table.name)


def link_chain(*, month: str) -> Path:
    """Return a `work` folder under the swap folder that holds file links to experiment 667's chain."""
    return link_tree(
        source=inp.chain_dir(root=inp.ROOT_667, month=month),
        dest=ROOT_SWAP / "chains" / month / "work",
    )


def redirect_inputs() -> None:
    """Point experiment 667's extended-input paths at a linked copy under the swap folder.

    `e00667_fold.install` writes stub run frames under `inputs/frames` and the `W560` cache under
    `inputs/w560` of `e00667_extend.OUT_ROOT`. Both then land in the swap folder, and the files that
    exist already are links to the stored ones. The call must come before `e00667_cand` is imported.
    """
    import e00667_extend as ext  # ty: ignore[unresolved-import]

    link_tree(source=inp.ROOT_667 / "inputs", dest=ROOT_SWAP / "inputs")
    ext.OUT_ROOT = ROOT_SWAP  # ty: ignore[invalid-assignment]
    ext.RAW_DIR = ROOT_SWAP / "inputs" / "raw"  # ty: ignore[invalid-assignment]


def snapshot_667() -> str:
    """Return a hash of every path, size, and modification time below experiment 667's folder."""
    h = hashlib.sha256()
    for path in sorted(inp.ROOT_667.rglob("*")):
        st = path.lstat()
        h.update(f"{path} {st.st_size} {st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def main() -> int:
    """Run the Candidate's prediction with the ERA5 table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    month, seed = args.month, args.seed
    before = snapshot_667()
    link_stored(month=month, seed=seed)
    redirect_inputs()
    os.environ.update(
        E00667_ROOT=str(ROOT_SWAP), E00667_MONTH=month, E00667_PHASE="predict",
        CHAIN667=str(link_chain(month=month)),
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
        e.CTX["table_ft"] = torch.from_numpy(table).to(args.device)
        del cols

    e.attach_tables = attach_pre
    cand.run_fit(device=args.device)
    assert snapshot_667() == before, "swap (c) changed a path under experiment 667's folder"
    return 0


if __name__ == "__main__":
    sys.exit(main())

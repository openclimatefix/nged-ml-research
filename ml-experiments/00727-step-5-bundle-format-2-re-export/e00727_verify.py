# ruff: noqa: E501
"""Experiment 727: prove that a format 2 bundle's `.npz` weights equal its `.pt` weights, and its hashes hold.

Usage: `python e00727_verify.py BUNDLE_DIR`. For every seed the script loads `seed<N>.pt` with
`torch.load` and `seed<N>.npz` with `numpy.load`, and requires the same names, shapes, dtypes, and
bit-identical values. It also recomputes the SHA-256 of every file that `manifest.json` lists, checks
that the listing covers every file except `manifest.json`, and checks the format 2 fields.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]


def sha256(*, path: Path) -> str:
    """Return the hex SHA-256 of a file.

    Args:
        path: The file to hash.

    Returns:
        The digest as lower-case hex.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Run the weight-equality and hash checks; exit with status 1 on any failure."""
    root = Path(sys.argv[1]).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["bundle_format_version"] == "2", manifest["bundle_format_version"]
    assert manifest["model_release"] == 0
    assert isinstance(manifest["model_release"], int)
    levels = json.loads((root / "config.json").read_text())["components"]["step5"][
        "quantile_levels"
    ]
    assert levels == [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99]
    files = manifest["components"]["step5"]["files"]
    for path in (*files["weights"], *files["weights_research_torch"]):
        assert (root / path).is_file(), path
    weights = root / "components" / "step5" / "weights"
    n_arrays = 0
    for seed in manifest["components"]["step5"]["seeds"]:
        state = torch.load(weights / f"seed{seed}.pt", map_location="cpu", weights_only=True)
        with np.load(weights / f"seed{seed}.npz") as arrays:
            assert sorted(arrays.files) == sorted(state), f"seed {seed}: array names differ"
            for name, tensor in state.items():
                array = arrays[name]
                want = tensor.numpy()
                assert array.shape == want.shape, (seed, name, array.shape, want.shape)
                assert array.dtype == want.dtype, (seed, name, array.dtype, want.dtype)
                assert array.tobytes() == want.tobytes(), (seed, name, "values differ")
                n_arrays += 1
    seed0 = (
        next(iter(sorted((root / "fixtures" / "step5").glob("origin=*"))))
        / "outputs_native.parquet"
    )
    columns = pl.read_parquet_schema(seed0)
    assert all(f"q{round(lv * 100):02d}_mean" in columns for lv in levels), (
        "13-level columns missing"
    )
    assert not any(name.startswith(("q25", "q40", "q60", "q75")) for name in columns)
    listed = manifest["sha256"]
    on_disk = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    assert set(listed) == on_disk, sorted(set(listed) ^ on_disk)
    for name, digest in listed.items():
        assert sha256(path=root / name) == digest, f"hash mismatch: {name}"
    print(
        f"OK: {n_arrays} arrays equal bit for bit; {len(listed)} file hashes match; format 2 fields set"
    )


if __name__ == "__main__":
    main()

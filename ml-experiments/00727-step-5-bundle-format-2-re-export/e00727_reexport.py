# ruff: noqa: E501, PLR0915
"""Experiment 727: re-export an experiment 705 step-5 bundle (format 1) as bundle format 2.

Usage: `python e00727_reexport.py --src BUNDLE_V1 --out BUNDLE_V2` (the output folder must be
`<00727 data folder>/bundles/<arm>/<month>`). The script makes three changes to the format 1 bundle:

1. It writes each seed's weights as `components/step5/weights/seed<N>.npz`, numpy arrays keyed by the
   `state_dict` names. The `.pt` files stay.
2. It sets `bundle_format_version` to "2" in `manifest.json` and adds `model_release` (0).
3. It sets `quantile_levels` in `config.json` to NGED's 13 delivery levels and regenerates the three
   quantile fixtures of every origin on the CPU, from the stored standardised input matrix and the
   `.npz` weights, with the functions of this folder's `reference_check.py`.

The raw network outputs, inputs, and ENS fixtures are copied unchanged. The script then compares the
quantiles at the 7 levels that format 1 also holds with the format 1 fixtures, and stops when one
differs by more than the format 1 tolerance of the reference check.
"""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
import reference_check as rc
import torch  # ty: ignore[unresolved-import]

HERE: Final = Path(__file__).resolve().parent
LEVELS_V2: Final = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99)
LEVEL_NAMES_V1: Final = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
SHARED_NAMES: Final = tuple(name for name in LEVEL_NAMES_V1 if name in rc.LEVEL_NAMES)
QUANTILE_TOLERANCE: Final = 2e-5
RAW_TOLERANCE: Final = 1e-5
MODEL_RELEASE: Final = 0
SOURCE_CODE_FILES: Final = ("e00727_reexport.py", "e00727_verify.py", "reference_check.py")


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


def write_json(*, obj: object, path: Path) -> None:
    """Write `obj` as indented JSON with a trailing newline.

    Args:
        obj: A JSON-serialisable object.
        path: The file to write.
    """
    path.write_text(json.dumps(obj, indent=1) + "\n")


def git_state() -> dict:
    """Return the research repository's commit and whether tracked files differ from it.

    Returns:
        The commit hash, the count of modified tracked files, and the count of untracked files.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=HERE
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True, cwd=HERE
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty_tracked_files": sum(not line.startswith("??") for line in status),
        "untracked_files": sum(line.startswith("??") for line in status),
    }


def write_weights(*, step5: Path, seeds: list[int]) -> None:
    """Write `seed<N>.npz` next to each `seed<N>.pt`.

    Args:
        step5: The bundle's `components/step5` folder.
        seeds: The seeds the bundle holds.
    """
    for seed in seeds:
        state = torch.load(
            step5 / "weights" / f"seed{seed}.pt", map_location="cpu", weights_only=True
        )
        arrays = {name: tensor.numpy() for name, tensor in state.items()}
        assert all(array.dtype != object for array in arrays.values())
        np.savez(step5 / "weights" / f"seed{seed}.npz", **arrays)


def removed_lag_inputs(*, stored: np.ndarray, config: dict) -> np.ndarray:
    """Return the standardised matrix with both lags set to the missing encoding.

    Args:
        stored: The stored standardised input matrix of one origin.
        config: The `components.step5` section of `config.json`.

    Returns:
        A copy of `stored` with the day-lag and week-lag columns set to value 0, flag 1, and the
        missing age code.
    """
    out = stored.copy()
    for start, key in ((37, "day"), (40, "week")):
        age = config["lags"]["missing_age_code"][key]
        out[:, start], out[:, start + 1], out[:, start + 2] = 0.0, 1.0, age
    return out


def level_columns(*, q: np.ndarray, suffix: str, names: tuple[str, ...]) -> pl.DataFrame:
    """Return a frame with one column per level.

    Args:
        q: Quantiles, shape (rows, levels).
        suffix: Text appended to each level name, such as `_s0` or `_mw`.
        names: The level names, in the order of the columns of `q`.

    Returns:
        A frame whose columns are `<name><suffix>`.
    """
    return pl.DataFrame({f"{name}{suffix}": q[:, i] for i, name in enumerate(names)})


def shared_columns(*, new: np.ndarray) -> np.ndarray:
    """Return the columns of `new` at the 7 levels that format 1 also holds.

    Args:
        new: Quantiles at the 13 format 2 levels, shape (rows, 13).

    Returns:
        Shape (rows, 7), ordered as `SHARED_NAMES`.
    """
    return new[:, [rc.LEVEL_NAMES.index(name) for name in SHARED_NAMES]]


def old_columns(*, frame: pl.DataFrame, suffix: str, prefix: str = "") -> np.ndarray:
    """Return the format 1 columns at the shared levels.

    Args:
        frame: A format 1 fixture frame.
        suffix: The column suffix, such as `_s0`, `_mean`, or `_mw`.
        prefix: Text before the level name.

    Returns:
        Shape (rows, 7), ordered as `SHARED_NAMES`, as float32 (the fixture dtype).
    """
    return frame.select([f"{prefix}{name}{suffix}" for name in SHARED_NAMES]).to_numpy()


def regenerate_fixtures(*, out: Path, src: Path) -> dict:
    """Rewrite the quantile fixtures of every origin at the 13 levels and compare with format 1.

    Args:
        out: The new bundle, with its weights, config, and manifest already at format 2.
        src: The format 1 bundle.

    Returns:
        The largest differences, per stage, between the format 2 values and the format 1 or stored
        values, at the 7 shared levels.

    Raises:
        RuntimeError: If a difference exceeds its tolerance.
    """
    assert tuple(f"q{round(lv * 100):02d}" for lv in LEVELS_V2) == rc.LEVEL_NAMES
    bundle = rc.load_bundle(out)
    config = bundle["config"]["components"]["step5"]
    seeds = bundle["manifest"]["components"]["step5"]["seeds"]
    table = rc.expected_table(bundle=bundle)
    capacity_all = np.array([p["capacity_mw"] for p in bundle["primaries"]["primaries"]])
    worst = dict.fromkeys(
        (
            "raw_output",
            "native_seed",
            "native_seed_mean",
            "contract_mw_over_capacity",
            "lags_removed",
        ),
        0.0,
    )
    for origin_dir in sorted((out / "fixtures" / "step5").glob("origin=*")):
        old_dir = src / "fixtures" / "step5" / origin_dir.name
        rows = pl.read_parquet(origin_dir / "input_matrix.parquet")
        old_native = pl.read_parquet(old_dir / "outputs_native.parquet")
        old_contract = pl.read_parquet(old_dir / "outputs_contract.parquet")
        old_removed = pl.read_parquet(old_dir / "outputs_lags_removed.parquet")
        names = [c["name"] for c in bundle["columns"]]
        stored = rows.select([f"std__{name}" for name in names]).to_numpy().astype(np.float32)
        primary = rows["primary_index"].to_numpy().astype(np.int64)
        xt, pt = torch.from_numpy(stored), torch.from_numpy(primary)
        xr = torch.from_numpy(removed_lag_inputs(stored=stored, config=config))
        native = old_native.select("primary_index", "valid_time")
        removed = old_removed.select("primary_index", "valid_time")
        per_seed = []
        with torch.no_grad():
            for seed in seeds:
                weights = bundle["weights"][seed]
                raw = rc.forward(w=weights, x=xt, primary=pt, table=table, config=config)
                want_raw = old_native.select([f"raw_out_{k}_s{seed}" for k in range(9)]).to_numpy()
                worst["raw_output"] = max(
                    worst["raw_output"], float(np.abs(raw.numpy() - want_raw).max())
                )
                q = rc.native_fractions(raw=raw, primary=primary, bundle=bundle)
                per_seed.append(q)
                worst["native_seed"] = max(
                    worst["native_seed"],
                    float(
                        np.abs(
                            shared_columns(new=q)
                            - old_columns(frame=old_native, suffix=f"_s{seed}")
                        ).max()
                    ),
                )
                native = native.hstack(
                    level_columns(q=q, suffix=f"_s{seed}", names=rc.LEVEL_NAMES)
                ).hstack(old_native.select([f"raw_out_{k}_s{seed}" for k in range(9)]))
                raw_removed = rc.forward(w=weights, x=xr, primary=pt, table=table, config=config)
                q_removed = rc.native_fractions(raw=raw_removed, primary=primary, bundle=bundle)
                worst["lags_removed"] = max(
                    worst["lags_removed"],
                    float(
                        np.abs(
                            shared_columns(new=q_removed)
                            - old_columns(frame=old_removed, suffix=f"_s{seed}")
                        ).max()
                    ),
                )
                removed = removed.hstack(
                    level_columns(q=q_removed, suffix=f"_s{seed}", names=rc.LEVEL_NAMES)
                )
        mean = np.sort(np.mean(np.stack(per_seed), axis=0, dtype=np.float32), axis=1)
        worst["native_seed_mean"] = max(
            worst["native_seed_mean"],
            float(
                np.abs(
                    shared_columns(new=mean) - old_columns(frame=old_native, suffix="_mean")
                ).max()
            ),
        )
        native = native.hstack(level_columns(q=mean, suffix="_mean", names=rc.LEVEL_NAMES))
        valid = rows["valid_time"].dt.epoch("us").to_numpy().astype("datetime64[us]")
        origin = rows["origin_time"].dt.epoch("us").to_numpy().astype("datetime64[us]")
        mask, mw_stack, mw_mean = rc.contract_rows(
            per_seed=per_seed, primary=primary, valid=valid, origin=origin, bundle=bundle
        )
        assert old_contract.height == int(mask.sum())
        assert (
            rows.filter(pl.Series(mask))
            .select(pl.col("time_series_id").cast(pl.Int8), "valid_time")
            .equals(old_contract.select("time_series_id", "valid_time"))
        ), "contract rows differ in order from format 1"
        capacity = capacity_all[primary[mask]]
        worst["contract_mw_over_capacity"] = max(
            worst["contract_mw_over_capacity"],
            float(
                (
                    np.abs(
                        shared_columns(new=mw_mean[mask])
                        - old_columns(frame=old_contract, suffix="_mw")
                    )
                    / capacity[:, None]
                ).max()
            ),
        )
        contract = old_contract.select(
            "time_series_id", "valid_time", "nwp_init_time", "power_fcst_init_time"
        ).hstack(level_columns(q=mw_mean[mask], suffix="_mw", names=rc.LEVEL_NAMES))
        for k, seed in enumerate(seeds):
            contract = contract.hstack(
                level_columns(q=mw_stack[k][mask], suffix=f"_s{seed}_mw", names=rc.LEVEL_NAMES)
            )
            got = shared_columns(new=mw_stack[k][mask])
            want = old_columns(frame=old_contract, suffix=f"_s{seed}_mw")
            worst["contract_mw_over_capacity"] = max(
                worst["contract_mw_over_capacity"],
                float((np.abs(got - want) / capacity[:, None]).max()),
            )
        native.write_parquet(origin_dir / "outputs_native.parquet", compression="zstd")
        removed.write_parquet(origin_dir / "outputs_lags_removed.parquet", compression="zstd")
        contract.write_parquet(origin_dir / "outputs_contract.parquet", compression="zstd")
        print(origin_dir.name, {k: f"{v:.2e}" for k, v in worst.items()}, flush=True)
    limits = {
        "raw_output": RAW_TOLERANCE,
        "native_seed": QUANTILE_TOLERANCE,
        "native_seed_mean": QUANTILE_TOLERANCE,
        "contract_mw_over_capacity": QUANTILE_TOLERANCE,
        "lags_removed": QUANTILE_TOLERANCE,
    }
    for name, limit in limits.items():
        if worst[name] > limit:
            msg = f"{name}: {worst[name]:.3e} exceeds {limit:g}; stop and report"
            raise RuntimeError(msg)
    return worst


def main() -> None:
    """Build the format 2 bundle from a format 1 bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="the format 1 bundle")
    parser.add_argument("--out", required=True, help="<00727 folder>/bundles/<arm>/<month>")
    args = parser.parse_args()
    torch.set_num_threads(1)
    src, out = Path(args.src).resolve(), Path(args.out).resolve()
    assert out.parent.parent.parent.name.startswith("00727-"), (
        f"--out must be <00727 folder>/bundles/<arm>/<month>: {out}"
    )
    assert out.parent.name == json.loads((src / "manifest.json").read_text())["arm"]
    assert out.name == json.loads((src / "manifest.json").read_text())["refit_month"]
    if out.exists():
        print("skip: output exists", out)
        return
    final, out = out, out.with_name(out.name + ".partial")
    if out.exists():
        shutil.rmtree(out)
    old_manifest = json.loads((src / "manifest.json").read_text())
    assert old_manifest["bundle_format_version"] == "1"
    shutil.copytree(src, out)
    (out / "manifest.json").unlink()
    step5 = out / "components" / "step5"
    seeds = old_manifest["components"]["step5"]["seeds"]
    write_weights(step5=step5, seeds=seeds)
    config = json.loads((out / "config.json").read_text())
    config["components"]["step5"]["quantile_levels"] = list(LEVELS_V2)
    write_json(obj=config, path=out / "config.json")
    for name in ("reference_check.py", "bundle_readme.md"):
        target = "README.md" if name == "bundle_readme.md" else name
        shutil.copyfile(HERE / name, out / target)
    manifest: dict[str, Any] = {
        **old_manifest,
        "bundle_format_version": "2",
        "model_release": MODEL_RELEASE,
    }
    write_json(obj=manifest, path=out / "manifest.json")
    worst = regenerate_fixtures(out=out, src=src)
    manifest["created_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["components"]["step5"]["files"]["weights"] = [
        f"components/step5/weights/seed{s}.npz" for s in seeds
    ]
    manifest["components"]["step5"]["files"]["weights_research_torch"] = [
        f"components/step5/weights/seed{s}.pt" for s in seeds
    ]
    manifest["export_checks"] = {
        "format_1_export": old_manifest["export_checks"],
        "format_1_export_note": "These checks describe the format 1 fixtures; the quantile fixtures of this bundle are regenerated, so see format_2_reexport for them.",
        "format_2_reexport": {
            "library_versions": {
                "torch": importlib.metadata.version("torch"),
                "numpy": importlib.metadata.version("numpy"),
                "polars": importlib.metadata.version("polars"),
            },
            "source_bundle": str(src),
            "source_manifest_sha256": sha256(path=src / "manifest.json"),
            "quantile_levels": list(LEVELS_V2),
            "levels_shared_with_format_1": list(SHARED_NAMES),
            "max_abs_difference_to_format_1_at_shared_levels": worst,
            "note": "outputs_native, outputs_contract, and outputs_lags_removed are regenerated on the CPU at 13 levels; every other fixture is copied from the source bundle",
        },
    }
    manifest["format_2_reexport_git"] = git_state()
    manifest["code_sha256"] = {
        **old_manifest["code_sha256"],
        **{name: sha256(path=HERE / name) for name in SOURCE_CODE_FILES if (HERE / name).exists()},
    }
    manifest.pop("sha256", None)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest["sha256"] = {str(p.relative_to(out)): sha256(path=p) for p in files}
    write_json(obj=manifest, path=out / "manifest.json")
    out.rename(final)
    print("bundle written", final, len(files) + 1, "files", flush=True)


if __name__ == "__main__":
    main()

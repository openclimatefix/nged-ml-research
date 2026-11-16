# ruff: noqa: E501
"""Experiment 699: every assertion that the plan names, one line printed per check.

Usage: python e00699_checks.py COMMAND

- `hash-inputs`: write `inputs.sha256` (the first fold's inputs, once, before any fit).
- `all`: run every check on whatever the data folder holds, and exit non-zero on the first failure.

The checks cover the holdout cut, the input hashes, the seed ranges of the five worlds, the share
of rows with a shifted lag in each stage, the time and target guards, the equality of the banks'
shifts across arms, the origin-time weather counts, the equal training budget, the B1 statistics, the
initial weights, and the absence of any write under experiment 677's data folder.
"""

import hashlib
import json
import re
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from e00699_common import (
    ARMS,
    BANK_BASE_581,
    BANK_BASE_660,
    FOLDS,
    HOLDOUT,
    NEW_ARMS,
    ROOT,
    ROOT_677,
    SEED_RANGES,
    SEEDS,
    SHARE_RANGE,
    STAGE_SEED_BASE,
    STORED_SEEDS,
    bank_folder,
)

EXPERIMENTS = Path(__file__).resolve().parent.parent
GRID_START_US = 1_567_296_000_000_000
HOLDOUT_US = int(HOLDOUT.timestamp() * 1e6)
HOLDOUT_IDX = (HOLDOUT_US - GRID_START_US) // 1_800_000_000
INPUTS_HASH_FILE = ROOT / "inputs.sha256"
"""Written before the first fit: any file under experiment 677's data folder newer than it is a write."""
INPUT_FILES = {
    "series_capacity": ROOT_677.parent
    / "backtest_year_2025-11_to_2026-10"
    / "series_capacity.parquet",
}


def ok(name: str, detail: str = "") -> None:
    """Print one passing check."""
    print(f"PASS {name} {detail}".rstrip(), flush=True)


def chains() -> list[tuple[str, str]]:
    """Return the (sub-folder, month) pairs that hold a chain."""
    found = []
    for sub in ("clean", "stage2", "early"):
        base = ROOT / "roots" / sub / "chains" / "5"
        found += [(sub, p.name) for p in sorted(base.glob("20??-??"))] if base.exists() else []
    return found


def sha256_of(*, path: Path) -> str:
    """Return the SHA-256 hash of a file's content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def input_tables() -> dict[str, Path]:
    """Return the input tables that every fold reads, keyed by a short name."""
    from e00677_extend import E607, OUT_ROOT  # ty: ignore[unresolved-import]

    tables = dict(INPUT_FILES)
    tables.update(
        {
            f"raw_summary_{p.stem}": p
            for p in sorted((OUT_ROOT / "inputs" / "raw" / "summary").glob("*.npz"))
        }
    )
    tables["era5_features"] = E607 / "era5_features.parquet"
    tables["observed"] = E607 / "observed.parquet"
    return tables


def hash_inputs() -> None:
    """Write the hash of every input table (and of the ensemble-summary file listing)."""
    from e00677_extend import E607  # ty: ignore[unresolved-import]

    lines = [f"{sha256_of(path=path)}  {name}" for name, path in input_tables().items()]
    listing = hashlib.sha256()
    for path in sorted((E607 / "ens_summary").glob("init=*.parquet")):
        listing.update(f"{path.name} {path.stat().st_size}\n".encode())
    lines.append(f"{listing.hexdigest()}  ens_summary_listing")
    target = ROOT / "inputs.sha256"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    print("wrote", target, len(lines), "hashes")


def check_input_hashes() -> None:
    """Assert that every input table still has the hash written before the first fit."""
    stored = ROOT / "inputs.sha256"
    assert stored.exists(), "run `hash-inputs` before the first fit"
    want = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in stored.read_text().splitlines()
    }
    for name, path in input_tables().items():
        assert sha256_of(path=path) == want[name], f"input table changed: {name}"
    ok("input hashes", f"{len(want)} tables unchanged")


def check_holdout() -> None:
    """Assert that no forecast, bank, or per-row score file holds a valid time at or after the holdout."""
    count = 0
    for sub in ("clean", "stage2", "early", "stored"):
        base = ROOT / "roots" / sub
        for path in [
            *base.glob("arms/*/work/fits/*/*.parquet"),
            *base.glob("backtests/*.parquet"),
            *base.glob("P0_D50_*/work/bank/*.parquet"),
        ]:
            latest = pl.scan_parquet(path).select(pl.col("valid_time").max()).collect().item()
            assert latest < HOLDOUT, f"{path} holds a valid time of {latest}"
            count += 1
        for path in base.glob("scores/*/rows.parquet"):
            latest = pl.scan_parquet(path).select(pl.col("valid_idx").max()).collect().item()
            assert latest < HOLDOUT_IDX, f"{path} holds a valid index of {latest}"
            count += 1
    ok("holdout cut", f"{count} files hold no valid time on or after {HOLDOUT:%Y-%m-%d}")


def read_injection(*, sub: str, month: str) -> dict:
    """Return a chain's `injection.json` (empty for the clean chain)."""
    path = ROOT / "roots" / sub / "chains" / "5" / month / "work" / "injection.json"
    return json.loads(path.read_text()) if path.exists() else {}


def stage3_log(*, sub: str, month: str) -> tuple[float, int] | None:
    """Return the share and the event count that stage 3's builder printed, from the fit log."""
    log = ROOT / "roots" / sub / "logs" / f"{month}_5w_fit.log"
    if not log.exists():
        return None
    found = re.findall(
        r"(\d+) synthetic events; share of rows with a shifted lag ([0-9.]+)", log.read_text()
    )
    return (float(found[-1][1]), int(found[-1][0])) if found else None


def stage3_time_guard(*, number: int, cutoff: int, first: int) -> None:
    """Assert that stage 3's world (seed YYYYMM) changes no half-hour at or after the cutoff.

    Stage 3 runs in experiment 677's own process, which hides power from the cutoff on, and the generator
    leaves a missing value missing. This re-runs the generator with stage 3's seed on a clean grid whose
    power is missing from the cutoff on, and finds the last half-hour that it changed.
    """
    import e00581_bank as bank  # ty: ignore[unresolved-import]

    clean = np.ones((15, cutoff + 2000))
    clean[:, cutoff:] = np.nan
    fixed, _ = bank.synthetic_clean(clean=clean, first=first, last=cutoff, seed=number)
    changed = np.flatnonzero(
        (np.nan_to_num(fixed, nan=-9) != np.nan_to_num(clean, nan=-9)).any(axis=0)
    )
    assert changed.size > 0, (number, "stage 3 world changed nothing")
    assert changed.max() < cutoff, (number, "stage 3 world reaches the cutoff")


def check_worlds() -> None:
    """Assert the seed ranges, the share of rows with a shifted lag, and the time and target guards."""
    source = next(EXPERIMENTS.glob("00591-*/e00591_arm.py")).read_text()
    assert 'seed=int(month.replace("-", ""))' in source, (
        "stage 3's seed is no longer the month number"
    )
    cand = next(EXPERIMENTS.glob("00677-*/e00677_cand.py")).read_text()
    assert "BANK_SEED_BASE: Final = 9_000_000" in cand, "the 581 bank seed base changed"
    seen = 0
    for sub, month in chains():
        number = int(month.replace("-", ""))
        info = read_injection(sub=sub, month=month)
        for stage in ("stage1", "stage2"):
            if stage not in info:
                continue
            world = info[stage]
            lo, hi = SEED_RANGES[stage]
            assert lo <= world["seed"] <= hi, (sub, month, stage, world["seed"])
            assert world["seed"] == STAGE_SEED_BASE[stage] + number
            assert SHARE_RANGE[0] <= world["share"] <= SHARE_RANGE[1], (
                sub,
                month,
                stage,
                world["share"],
            )
            assert world["last_changed_idx"] < world["cutoff_idx"], (
                sub,
                month,
                stage,
                "a world changed power at or after the cutoff",
            )
            assert world["asserted_in_process"], (sub, month, stage)
            seen += 1
        if sub != "clean" and "stage2" in info:
            stage3_time_guard(
                number=number, cutoff=info["stage2"]["cutoff_idx"], first=info["stage2"]["first"]
            )
        log = stage3_log(sub=sub, month=month)
        if log is not None and sub in {"clean", "stage2", "early"}:
            share, n_events = log
            if (ROOT / "roots" / sub / "logs" / f"{month}_5w_fit.log").exists():
                assert SHARE_RANGE[0] <= share <= SHARE_RANGE[1], (sub, month, "stage3", share)
                lo, hi = SEED_RANGES["stage3"]
                assert lo <= number <= hi
                assert n_events > 0
                seen += 1
        assert SEED_RANGES["bank581"][0] <= BANK_BASE_581 + number <= SEED_RANGES["bank581"][1]
        seed_file = ROOT / "roots" / sub / "bank660" / f"{month}_seed.json"
        if seed_file.exists():
            seed = json.loads(seed_file.read_text())["seed"]
            assert SEED_RANGES["bank660"][0] <= seed <= SEED_RANGES["bank660"][1]
            assert seed == BANK_BASE_660 + number
    ok(
        "worlds",
        f"seed ranges, shares in {SHARE_RANGE}, time guard, target guard on {seen} stage records",
    )


def check_banks() -> None:
    """Assert that every arm's bank holds the same shifts row for row, in each fold and for each seed."""
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    n_files = 0
    for suffix in ("", "_b660"):
        for month in FOLDS:
            reference = None
            for arm_key in ("s3_stored", "none_stored", *NEW_ARMS):
                arm = ARMS[arm_key]
                if suffix and arm.stored:
                    continue
                folder = bank_folder(arm=arm, month=month, suffix=suffix)
                if not folder.exists():
                    continue
                for seed in STORED_SEEDS if arm.stored else SEEDS:
                    path = folder / f"{month}_d0_s{seed}.parquet"
                    if not path.exists():
                        continue
                    frame = pl.read_parquet(path, columns=[*keys, "lag_shift"]).sort(keys)
                    assert frame["valid_time"].max() < HOLDOUT  # ty: ignore[unsupported-operator]
                    if reference is None:
                        reference = frame
                    assert frame.height == reference.height, (path, frame.height, reference.height)
                    assert frame.select(keys).equals(reference.select(keys)), path
                    assert np.array_equal(
                        frame["lag_shift"].to_numpy(), reference["lag_shift"].to_numpy()
                    ), path
                    n_files += 1
            if reference is not None and suffix == "":
                first_valid = reference.filter(pl.col("lag_shift") != 0)["valid_time"].min()
                idx = (
                    int(first_valid.timestamp() * 1e6)  # ty: ignore[unresolved-attribute]
                    - GRID_START_US
                ) // 1_800_000_000
                for chain in ("early", "stage2"):
                    info = read_injection(sub=chain, month=month)
                    for stage in ("stage1", "stage2"):
                        if stage in info:
                            assert info[stage]["last_changed_idx"] < idx, (
                                month,
                                chain,
                                stage,
                                "a training world reaches the bank's first shift",
                            )
    ok(
        "banks",
        f"{n_files} bank files hold identical lag shifts per fold, and training worlds end before the bank's first shift",
    )


def check_weather() -> None:
    """Assert the origin-time weather counts on the predict-phase copies and on every exported file."""
    n = 0
    for path in sorted((ROOT / "checks").glob("*/*/20??-??_pfit.json")):
        month = path.name[:7]
        data = json.loads(path.read_text())
        year, mon = (int(v) for v in month.split("-"))
        days = (
            np.datetime64(f"{year + (mon == 12)}-{(mon % 12) + 1:02d}-01")
            - np.datetime64(f"{year}-{mon:02d}-01")
        ).astype(int)
        assert data["origins_weather_checked"] == days, (
            path,
            data["origins_weather_checked"],
            days,
        )
        n += 1
    m = 0
    for path in sorted((ROOT / "roots").glob("*/backtests/*.parquet")):
        frame = (
            pl.scan_parquet(path)
            .select(
                ((pl.col("power_fcst_init_time") - pl.col("nwp_init_time")) >= timedelta(hours=3))
                .all()
                .alias("early_enough")
            )
            .collect()
        )
        assert frame["early_enough"].item(), (
            f"{path}: a run issued after the 09:00 UTC native origin"
        )
        m += 1
    ok(
        "weather",
        f"origins_weather_checked equals the origin days in {n} predict-phase files; nwp_init_time at or before the native origin in {m} exported files",
    )


def parse_epochs(*, log: Path) -> list[int]:
    """Return the epoch counts that a pre-training log prints."""
    return (
        [int(v) for v in re.findall(r"(\d+) epochs, val NLL", log.read_text())]
        if log.exists()
        else []
    )


def check_budget() -> None:  # noqa: PLR0912
    """Assert equal row counts between chains and against experiment 677, and report the epochs."""
    pretrain_rows: dict[str, dict[str, int]] = {}
    for sub, month in chains():
        log = ROOT / "roots" / sub / "logs" / f"{month}_5w_pretrain.log"
        if log.exists():
            found = re.findall(r"pretraining rows ([0-9,]+)", log.read_text())
            if found:
                pretrain_rows.setdefault(month, {})[sub] = int(found[-1].replace(",", ""))
    for month, by_chain in pretrain_rows.items():
        assert len(set(by_chain.values())) == 1, (month, by_chain)
        stored_log = ROOT_677 / "logs" / f"{month}_5w_pretrain.log"
        if stored_log.exists():
            found = re.findall(r"pretraining rows ([0-9,]+)", stored_log.read_text())
            if found:
                assert int(found[-1].replace(",", "")) == next(iter(by_chain.values())), (
                    month,
                    "differs from experiment 677",
                )
    step2: dict[str, dict[str, set[int]]] = {}
    for sub, month in chains():
        for path in sorted(
            (ROOT / "roots" / sub / "chains" / "5" / month / "work" / "logs" / "step2" / "P5").glob(
                "*.json"
            )
        ):
            step2.setdefault(month, {}).setdefault(sub, set()).add(
                json.loads(path.read_text())["rows"]
            )
    for month, by_chain in step2.items():
        values = {v for s in by_chain.values() for v in s}
        assert len(values) == 1, (month, by_chain)
        stored = (
            ROOT_677
            / "chains"
            / "5"
            / month
            / "work"
            / "logs"
            / "step2"
            / "P5"
            / f"{month}_s0.json"
        )
        if stored.exists():
            assert json.loads(stored.read_text())["rows"] in values, (
                month,
                "differs from experiment 677",
            )
    stage3: dict[str, dict[str, set[int]]] = {}
    for path in sorted((ROOT / "checks").glob("*/*/logs/20??-??_d0_s*.json")):
        month = path.name[:7]
        data = json.loads(path.read_text())
        assert "rows" in data, (path, "a stage-3 fit log without `rows`: a stub was copied")
        stage3.setdefault(month, {}).setdefault(path.parts[-4], set()).add(int(data["rows"]))
    for month, by_chain in stage3.items():
        values = {v for s in by_chain.values() for v in s}
        assert len(values) == 1, (month, "stage-3 rows differ between chains", by_chain)
        assert "clean" in by_chain, (month, "no clean chain logs")
        assert len(by_chain) >= 2, (
            month,
            "chains with logs",
            list(by_chain),
        )
        for chain in by_chain:
            for folder in sorted((ROOT / "checks" / chain).glob("*/logs")):
                n_logs = len(list(folder.glob(f"{month}_d0_s*.json")))
                assert n_logs in (0, len(SEEDS)), (chain, folder, month, n_logs, "seeds with rows")
    print(
        "  stage-3 row counts are compared between chains only: experiment 677's stored stage-3 logs are"
        " 'reused' stubs without `rows`, so stage-3 rows cannot be compared with 677"
    )
    epochs: dict[tuple[str, str], list[int]] = {}
    for sub, month in chains():
        epochs.setdefault((sub, "stage1"), []).extend(
            parse_epochs(log=ROOT / "roots" / sub / "logs" / f"{month}_5w_pretrain.log")
        )
        for path in (
            ROOT / "roots" / sub / "chains" / "5" / month / "work" / "logs" / "step2" / "P5"
        ).glob("*.json"):
            epochs.setdefault((sub, "stage2"), []).append(json.loads(path.read_text())["epochs"])
    for (sub, stage), values in sorted(epochs.items()):
        if values:
            print(
                f"  mean epochs, chain {sub}, {stage}: {np.mean(values):.1f} over {len(values)} fits"
            )
    ok(
        "equal training budget",
        f"stage-1 counts for {len(pretrain_rows)} folds, stage-2 counts for {len(step2)}, stage-3 counts for {len(stage3)} equal across chains",
    )


def check_b1() -> None:
    """Assert that the B1 statistics written by the new pre-trainings equal experiment 677's stored files."""
    n = 0
    for sub, month in chains():
        mine = (
            ROOT
            / "roots"
            / sub
            / "chains"
            / "3b"
            / month
            / "work"
            / "stats"
            / "B1"
            / f"{month}.npz"
        )
        stored = ROOT_677 / "chains" / "3b" / month / "work" / "stats" / "B1" / f"{month}.npz"
        if mine.exists() and not mine.is_symlink() and stored.exists():
            a, b = np.load(mine), np.load(stored)
            assert np.allclose(a["mean"], b["mean"], rtol=1e-6)
            assert np.allclose(a["std"], b["std"], rtol=1e-6), (sub, month)
            n += 1
    ok("B1 statistics", f"{n} files equal experiment 677's")


def check_init_weights() -> None:
    """Assert that the recorded initial weights of each seed equal a fresh network built with that seed."""
    import e00591_model as model_module  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    n = 0
    for sub, month in chains():
        info = read_injection(sub=sub, month=month).get("init_weights", {})
        for seed, record in info.items():
            torch.manual_seed(int(seed))
            network = model_module.AuxLagMdn(
                n_primaries=record["n_primaries"], stage="pre", n_knots=record["n_knots"], n_b1=record["n_b1"], kind="shared",
            )  # fmt: skip
            digest = hashlib.sha256()
            for key, value in sorted(network.state_dict().items()):
                digest.update(key.encode())
                digest.update(value.detach().cpu().numpy().tobytes())
            assert digest.hexdigest() == record["sha256"], (sub, month, seed)
            n += 1
    ok("initial weights", f"{n} recorded hashes equal a fresh network built with the same seed")


def check_no_write_to_677() -> None:
    """Assert that no file under experiment 677's data folder is newer than this experiment's start."""
    result = subprocess.run(
        ["find", str(ROOT_677), "-type", "f", "-newer", str(INPUTS_HASH_FILE), "-print", "-quit"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert not result.stdout.strip(), (
        f"written under experiment 677's folder: {result.stdout.strip()}"
    )
    ok("experiment 677's data folder", f"no file newer than {INPUTS_HASH_FILE.name}")


def main() -> None:
    """Run the command named on the command line."""
    command = sys.argv[1]
    if command == "hash-inputs":
        hash_inputs()
        return
    assert command == "all", command
    for check in (
        check_input_hashes,
        check_holdout,
        check_worlds,
        check_banks,
        check_weather,
        check_budget,
        check_b1,
        check_init_weights,
        check_no_write_to_677,
    ):
        check()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

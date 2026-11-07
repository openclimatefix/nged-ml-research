# ruff: noqa: E501, D103, ANN001
"""Experiment 614: the final-test runner (one pass over origins 2026-07-01 to 2026-10-31, never run twice).

The runner has two modes, set in the configuration file. `development` mode runs on origins before
2026-07-01 and refuses any later origin. `final` mode runs on exactly the final-test origins and needs
three things: `approved_by_maintainer` (a date, filled in by the maintainer after freezing the roster),
the command-line flag `--run-final-test`, and no lock file.

What a run does, in order:

1. Reads the configuration and checks the origin range against the mode.
2. Refuses to run if the lock file exists, whatever its status. A failed run leaves its lock, and
   only the maintainer may delete it.
3. Checks every listed artefact of every candidate against its listed SHA-256 hash.
4. Creates the lock file atomically (exclusive create) with the configuration hash, the runner hash,
   the git commit, and each candidate's artefact hashes and combined hash.
5. For each daily origin (00:00 UTC ensemble run plus 9 hours), builds the input rows from experiment
   607's extended tables: the weather summary of the run issued at or before the origin minus 9 hours,
   and the day and week power lags (experiment 558's capped lags). It asserts, row by row, that the
   weather run is the newest run at or before the origin minus 9 hours and that its table row carries
   that origin, that the valid time is after the origin, and that both lag times are at least 24 hours
   before the origin. Any violation stops the run.
6. Passes the rows to each candidate's predictor, which returns the 11 quantiles, and asserts that the
   predictor changed neither the row order nor the row count and returned finite, non-crossing
   quantiles.
7. Writes each candidate's forecasts as Parquet (one file per candidate), scores each candidate once
   against the observed power (continuous ranked probability score and q05-q95 coverage, by lead
   band), writes `scores.csv`, and marks the lock `completed`.

A predictor is a class `Predictor(artefacts: dict[str, Path])` with `predict(inputs: pl.DataFrame) ->
np.ndarray` of shape (rows, 11). The runner gives it only the columns listed in `INPUT_COLUMNS` plus the
weather summary columns. Observed power at the valid time is never given to a predictor.

The configuration's `inputs` hold `ens_summary_dir` (experiment 607's folder of `init=YYYY-MM-DD.parquet`
files), `observed` (its observed-power cache), and `capacity_csv` (a file with `time_series_id` and
`capacity`, in the unit of the cache, used to express the continuous ranked probability score as points
of capacity). With no event labels for the final window, the scores cover every row with an observed
reading, not only event-free rows.

Usage: python e00614_final_runner.py CONFIG.json [--hash-artefacts | --check | --run-final-test]
"""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00558_lags as lags  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

FINAL_FIRST: Final = datetime(2026, 7, 1, tzinfo=UTC)
FINAL_LAST: Final = datetime(2026, 10, 31, tzinfo=UTC)
PUBLICATION_DELAY: Final = timedelta(hours=9)
HORIZON_STEPS: Final = 480
STEP: Final = timedelta(minutes=30)
GRID_START: Final = datetime(2024, 4, 1, tzinfo=UTC)
QCOLS: Final = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
TAUS: Final = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
KEYS: Final = ["time_series_id", "power_fcst_init_time", "valid_time"]
INPUT_COLUMNS: Final = [
    *KEYS,
    "nwp_init_time",
    "lead_hours",
    "lag_day_time",
    "lag_week_time",
    "power_lag_day",
    "power_lag_week",
]
BANDS: Final = (("0-24 h", 0.0, 24.0), ("24-120 h", 24.0, 120.0), ("120-240 h", 120.0, 240.0))


class RunnerError(Exception):
    """A rule of the final test was broken; the run stops."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parse_day(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def artefact_paths(*, cand: dict) -> dict[str, Path]:
    root = Path(cand["artefact_root"])
    return {rel: root / rel for rel in cand["artefacts"]}


def verify_artefacts(*, cand: dict) -> str:
    """Check every artefact against its listed hash and return the candidate's combined hash."""
    lines = []
    for rel, path in artefact_paths(cand=cand).items():
        if not path.is_file():
            raise RunnerError(f"{cand['id']}: artefact {rel} does not exist at {path}")
        got = sha256_file(path)
        if got != cand["artefacts"][rel]:
            raise RunnerError(
                f"{cand['id']}: artefact {rel} hash {got} differs from the listed {cand['artefacts'][rel]}"
            )
        lines.append(f"{rel}:{got}")
    if not lines:
        raise RunnerError(f"{cand['id']}: no artefacts listed")
    return sha256_text("\n".join(sorted(lines)))


def check_config(*, cfg: dict, final_flag: bool) -> tuple[datetime, datetime]:
    first, last = parse_day(cfg["origin_first"]), parse_day(cfg["origin_last"])
    if first > last:
        raise RunnerError("origin_first is after origin_last")
    ids = [c["id"] for c in cfg["candidates"]]
    if len(ids) != len(set(ids)) or not ids:
        raise RunnerError("candidate ids must be unique and not empty")
    if cfg["mode"] == "development":
        if last >= FINAL_FIRST or final_flag:
            raise RunnerError("development mode never opens origins from 2026-07-01 onwards")
    elif cfg["mode"] == "final":
        if not final_flag:
            raise RunnerError("final mode needs --run-final-test")
        if not cfg.get("approved_by_maintainer"):
            raise RunnerError(
                "final mode needs `approved_by_maintainer` (a date) in the configuration"
            )
        if (first, last) != (FINAL_FIRST, FINAL_LAST):
            raise RunnerError("final mode runs on exactly the origins 2026-07-01 to 2026-10-31")
    else:
        raise RunnerError(f"unknown mode {cfg['mode']!r}")
    return first, last


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"


def create_lock(*, path: Path, cfg_text: str, hashes: dict[str, str], cfg: dict) -> None:
    """Create the lock file with an exclusive create, so that two runs cannot both start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "started",
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": cfg["mode"],
        "run_name": cfg["run_name"],
        "origin_first": cfg["origin_first"],
        "origin_last": cfg["origin_last"],
        "config_sha256": sha256_text(cfg_text),
        "runner_sha256": sha256_file(Path(__file__)),
        "git_head": git_head(),
        "candidates": {
            c["id"]: {"artefacts": c["artefacts"], "combined_sha256": hashes[c["id"]]}
            for c in cfg["candidates"]
        },
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise RunnerError(
            f"the lock file {path} exists: this test has been started before and is never run twice; only the maintainer may delete the lock"
        ) from None
    with os.fdopen(fd, "w") as f:
        json.dump(record, f, indent=1)


def finish_lock(*, path: Path, outputs: dict[str, str]) -> None:
    record = json.loads(path.read_text())
    record |= {
        "status": "completed",
        "finished_utc": datetime.now(UTC).isoformat(),
        "outputs_sha256": outputs,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    path.chmod(0o644)
    tmp.replace(path)
    path.chmod(0o444)


def idx_of(*, t: pl.Series) -> np.ndarray:
    return (
        (t.dt.epoch("us").to_numpy() - int(GRID_START.timestamp() * 1e6))
        // int(STEP.total_seconds() * 1e6)
    ).astype(np.int64)


def time_of(*, idx: np.ndarray) -> pl.Series:
    us = idx.astype(np.int64) * int(STEP.total_seconds() * 1e6) + int(GRID_START.timestamp() * 1e6)
    return pl.Series(us).cast(pl.Datetime("us", "UTC"))


def origin_rows(*, origin: datetime, ens_dir: Path, observed: pl.DataFrame) -> pl.DataFrame:
    """Return the input rows of one origin, after asserting the timing rules on every row."""
    run = origin - PUBLICATION_DELAY
    newest_allowed = origin - PUBLICATION_DELAY
    path = ens_dir / f"init={run:%Y-%m-%d}.parquet"
    if not path.is_file():
        raise RunnerError(f"origin {origin}: the ensemble run {run} is missing in {ens_dir}")
    runs = sorted(
        datetime.strptime(p.name[5:15], "%Y-%m-%d").replace(tzinfo=UTC)
        for p in ens_dir.glob("init=*.parquet")
    )
    newest = max(r for r in runs if r <= newest_allowed)
    if newest != run:
        raise RunnerError(
            f"origin {origin}: the newest run at or before the origin minus 9 h is {newest}, not {run}"
        )
    ens = pl.read_parquet(path)
    n_before = ens.height
    ens = ens.filter(pl.col("valid_time") < pl.lit(origin + HORIZON_STEPS * STEP))
    if ens.is_empty():
        raise RunnerError(f"origin {origin}: no rows within 240 h")
    bad_origin = ens.filter(pl.col("power_fcst_init_time") != pl.lit(origin))
    if bad_origin.height:
        raise RunnerError(
            f"origin {origin}: {bad_origin.height} of {n_before} rows of the run carry another power_fcst_init_time"
        )
    bad_after = ens.filter(pl.col("valid_time") <= pl.col("power_fcst_init_time"))
    if bad_after.height:
        raise RunnerError(
            f"origin {origin}: {bad_after.height} rows have a valid time not after the origin"
        )
    if ens.select(pl.struct("time_series_id", "valid_time").is_duplicated().any()).item():
        raise RunnerError(f"origin {origin}: duplicate (series, valid time) rows")
    target = idx_of(t=ens["valid_time"])
    origin_idx = np.full_like(target, int((origin - GRID_START) / STEP))
    try:
        day, week = lags.capped_lag_indices(target_idx=target, origin_idx=origin_idx)
    except AssertionError as err:
        raise RunnerError(f"origin {origin}: the capped lag rule failed: {err}") from err
    if ((origin_idx - day) < 48).any() or ((origin_idx - week) < 48).any():
        raise RunnerError(
            f"origin {origin}: a power lag is stamped less than 24 h before the origin"
        )
    lag_day_time, lag_week_time = time_of(idx=day), time_of(idx=week)
    if (
        lag_day_time.dt.epoch("us").to_numpy()
        > int((origin - timedelta(hours=24)).timestamp() * 1e6)
    ).any():
        raise RunnerError(f"origin {origin}: a day lag is younger than 24 h")
    rows = ens.with_columns(
        nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")),
        lead_hours=(
            (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
        ).cast(pl.Float32),
        lag_day_time=lag_day_time,
        lag_week_time=lag_week_time,
    )
    for name, col in (("power_lag_day", "lag_day_time"), ("power_lag_week", "lag_week_time")):
        rows = rows.join(
            observed.rename({"valid_time": col, "power": name}),
            on=["time_series_id", col],
            how="left",
        )
    return rows


def predict_once(*, predictor, rows: pl.DataFrame, cand_id: str) -> np.ndarray:
    q = np.asarray(predictor.predict(rows.clone()), dtype=np.float64)
    if q.shape != (rows.height, len(QCOLS)):
        raise RunnerError(
            f"{cand_id}: predictor returned shape {q.shape}, expected {(rows.height, len(QCOLS))}"
        )
    if not np.isfinite(q).all():
        raise RunnerError(f"{cand_id}: predictor returned non-finite quantiles")
    if (np.diff(q, axis=1) < -1e-9).any():
        raise RunnerError(f"{cand_id}: predictor returned crossing quantiles")
    return q.astype(np.float32)


def score(*, fc: pl.DataFrame, observed: pl.DataFrame, capacity: pl.DataFrame) -> pl.DataFrame:
    """Return CRPS (twice the mean pinball loss over the 11 levels, as points of capacity) and q05-q95 coverage by lead band."""
    df = (
        fc.join(
            observed.rename({"power": "observed"}), on=["time_series_id", "valid_time"], how="inner"
        )
        .filter(pl.col("observed").is_not_null())
        .join(capacity, on="time_series_id", how="inner")
    )
    y = df["observed"].to_numpy().astype(np.float64)
    q = df.select(QCOLS).to_numpy().astype(np.float64)
    err = y[:, None] - q
    pin = np.maximum(TAUS[None, :] * err, (TAUS[None, :] - 1) * err)
    crps = 2 * pin.mean(axis=1) / df["capacity"].to_numpy() * 100
    hit = (q[:, 1] <= y) & (y <= q[:, 9])
    lead = df["lead_hours"].to_numpy()
    out = []
    for name, lo, hi in (("all", 0.0, 1e9), *BANDS):
        sel = (lead >= lo) & (lead < hi)
        if sel.any():
            out.append(
                {
                    "band": name,
                    "n_rows": int(sel.sum()),
                    "crps_points_of_capacity": float(crps[sel].mean()),
                    "coverage_q05_q95": float(hit[sel].mean()),
                }
            )
    return pl.DataFrame(out)


def load_predictor(*, cand: dict) -> object:
    module, cls = cand["predictor"].split(":")
    return getattr(importlib.import_module(module), cls)(artefact_paths(cand=cand))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument(
        "--hash-artefacts",
        action="store_true",
        help="print the SHA-256 of every listed artefact and stop",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="check the configuration, artefact hashes, and timing rules on every origin; predict nothing, write no lock",
    )
    p.add_argument("--run-final-test", action="store_true")
    args = p.parse_args(argv)
    cfg_text = Path(args.config).read_text()
    cfg = json.loads(cfg_text)
    if args.hash_artefacts:
        for cand in cfg["candidates"]:
            for rel, path in artefact_paths(cand=cand).items():
                print(cand["id"], rel, sha256_file(path))
        return
    first, last = check_config(cfg=cfg, final_flag=args.run_final_test)
    lock = Path(cfg["lock_file"])
    out_dir = Path(cfg["output_dir"])
    if lock.exists() and not args.check:
        raise RunnerError(
            f"the lock file {lock} exists: this test has been started before and is never run twice; only the maintainer may delete the lock"
        )
    hashes = {c["id"]: verify_artefacts(cand=c) for c in cfg["candidates"]}
    ens_dir = Path(cfg["inputs"]["ens_summary_dir"])
    observed = pl.read_parquet(
        cfg["inputs"]["observed"], columns=["time_series_id", "valid_time", "power"]
    )
    capacity = pl.read_csv(cfg["inputs"]["capacity_csv"]).select(
        pl.col("time_series_id").cast(pl.Int32), "capacity"
    )
    origins = [
        first + timedelta(days=k) + PUBLICATION_DELAY for k in range((last - first).days + 1)
    ]
    if args.check:
        for origin in origins:
            origin_rows(origin=origin, ens_dir=ens_dir, observed=observed)
        print(
            f"check passed: {len(origins)} origins, candidates {list(hashes)}, no lock written, nothing predicted"
        )
        return
    create_lock(path=lock, cfg_text=cfg_text, hashes=hashes, cfg=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictors = {c["id"]: load_predictor(cand=c) for c in cfg["candidates"]}
    frames: dict[str, list[pl.DataFrame]] = {cid: [] for cid in predictors}
    n_rows = 0
    for origin in origins:
        rows = origin_rows(origin=origin, ens_dir=ens_dir, observed=observed)
        n_rows += rows.height
        for cid, predictor in predictors.items():
            q = predict_once(predictor=predictor, rows=rows, cand_id=cid)
            frames[cid].append(
                rows.select([*KEYS, "lead_hours"]).hstack(
                    [pl.Series(name, q[:, i]) for i, name in enumerate(QCOLS)]
                )
            )
    outputs, score_rows = {}, []
    for cid, parts in frames.items():
        fc = pl.concat(parts).sort(KEYS)
        path = out_dir / f"forecasts_{cid}.parquet"
        fc.write_parquet(path, compression="zstd", compression_level=9, statistics=True)
        outputs[path.name] = sha256_file(path)
        score_rows.append(
            score(fc=fc, observed=observed, capacity=capacity).with_columns(candidate=pl.lit(cid))
        )
        print(cid, fc.height, "rows written", flush=True)
    scores = pl.concat(score_rows).select(
        "candidate", "band", "n_rows", "crps_points_of_capacity", "coverage_q05_q95"
    )
    scores.write_csv(out_dir / "scores.csv")
    outputs["scores.csv"] = sha256_file(out_dir / "scores.csv")
    finish_lock(path=lock, outputs=outputs)
    print(scores)
    print(
        f"done: {len(origins)} origins, {n_rows} input rows per candidate set, lock {lock} completed"
    )


if __name__ == "__main__":
    try:
        main()
    except RunnerError as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        raise SystemExit(2) from err

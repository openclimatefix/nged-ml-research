# ruff: noqa: E501, D103, PLR0915
"""Experiment 614: prove the final-test runner on a development month with a dummy configuration.

Runs the runner as a command-line program on origins 2026-06-01 to 2026-06-07 with two dummy frozen
candidates, and checks that: a run completes and writes forecasts and scores; the lock records each
candidate's hash; a second run is refused; a changed artefact is refused; a wrong-origin row in an
ensemble file stops the run; a development configuration with an origin in July is refused; and a
final configuration without the maintainer's approval is refused. Writes `dev_test_report.txt`.

Usage: python e00614_dev_test.py OUTDIR
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

RUNNER = Path(__file__).with_name("e00614_final_runner.py")
E607 = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        capture_output=True,
        text=True,
        env=os.environ,
        check=False,
    )


def main(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    art = out / "artefacts"
    art.mkdir(parents=True)
    observed = pl.read_parquet(
        E607 / "observed.parquet", columns=["time_series_id", "valid_time", "power"]
    )
    prior = (
        observed.filter(pl.col("valid_time") < pl.datetime(2026, 5, 1, time_zone="UTC"))
        .group_by("time_series_id")
        .agg(pl.col("power").mean().alias("m"), pl.col("power").std().alias("s"))
    )
    (art / "persistence.json").write_text(json.dumps({"scale": 0.1, "fallback": 5.0}))
    (art / "weather_only.json").write_text(
        json.dumps(
            {
                "mean": {str(i): m for i, m in prior.select("time_series_id", "m").iter_rows()},
                "std": {str(i): s for i, s in prior.select("time_series_id", "s").iter_rows()},
            }
        )
    )
    shutil.copy(
        ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07/substations.csv",
        out / "capacity.csv",
    )
    base = {
        "run_name": "dev-month-dummy",
        "mode": "development",
        "origin_first": "2026-06-01",
        "origin_last": "2026-06-07",
        "approved_by_maintainer": None,
        "inputs": {
            "ens_summary_dir": str(E607 / "ens_summary"),
            "observed": str(E607 / "observed.parquet"),
            "capacity_csv": str(out / "capacity.csv"),
        },
        "output_dir": str(out / "run1_out"),
        "lock_file": str(out / "run1.lock"),
        "candidates": [
            {
                "id": "dummy_persistence",
                "network_class": 1,
                "predictor": "e00614_dummy_predictors:PersistencePredictor",
                "artefact_root": str(art),
                "artefacts": {"persistence.json": ""},
            },
            {
                "id": "dummy_weather_only",
                "network_class": 3,
                "predictor": "e00614_dummy_predictors:WeatherOnlyPredictor",
                "artefact_root": str(art),
                "artefacts": {"weather_only.json": ""},
            },
        ],
    }
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(base))
    hashed = run(str(cfg_path), "--hash-artefacts")
    lines = [f"hash-artefacts exit {hashed.returncode}:\n{hashed.stdout.strip()}"]
    for line in hashed.stdout.splitlines():
        cid, rel, digest = line.split()
        next(c for c in base["candidates"] if c["id"] == cid)["artefacts"][rel] = digest  # ty: ignore[invalid-assignment]
    cfg_path.write_text(json.dumps(base, indent=1))
    results: dict[str, bool] = {}

    def record(name: str, ok: bool, detail: str) -> None:
        results[name] = ok
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    r = run(str(cfg_path), "--check")
    record(
        "check (no lock, no prediction)",
        r.returncode == 0 and not Path(base["lock_file"]).exists(),
        r.stdout.strip() or r.stderr.strip(),
    )
    r = run(str(cfg_path))
    lock = (
        json.loads(Path(base["lock_file"]).read_text()) if Path(base["lock_file"]).exists() else {}
    )
    record(
        "first run completes",
        r.returncode == 0 and lock.get("status") == "completed",
        r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip(),
    )
    out_files = sorted(p.name for p in Path(base["output_dir"]).glob("*"))
    record(
        "writes Parquet forecasts and scores",
        out_files
        == [
            "forecasts_dummy_persistence.parquet",
            "forecasts_dummy_weather_only.parquet",
            "scores.csv",
        ],
        str(out_files),
    )
    record(
        "lock records the candidate hashes",
        set(lock.get("candidates", {})) == {"dummy_persistence", "dummy_weather_only"}
        and all(len(v["combined_sha256"]) == 64 for v in lock.get("candidates", {}).values()),
        json.dumps({k: v["combined_sha256"][:12] for k, v in lock.get("candidates", {}).items()}),
    )
    fc = pl.read_parquet(Path(base["output_dir"]) / "forecasts_dummy_persistence.parquet")
    record(
        "forecasts cover 7 origins, 16 Primaries, leads 0.5 to 239.5 h",
        fc["power_fcst_init_time"].n_unique() == 7
        and float(fc["lead_hours"].max()) == 239.5  # ty: ignore[invalid-argument-type]
        and fc.height == 7 * 16 * 479,
        f"rows {fc.height}, origins {fc['power_fcst_init_time'].n_unique()}, max lead {fc['lead_hours'].max()}",
    )
    r = run(str(cfg_path))
    record(
        "second run is refused",
        r.returncode == 2 and "never run twice" in r.stderr,
        r.stderr.strip(),
    )
    # a changed artefact
    cfg2 = json.loads(cfg_path.read_text()) | {
        "lock_file": str(out / "run2.lock"),
        "output_dir": str(out / "run2_out"),
    }
    (art / "persistence.json").write_text(json.dumps({"scale": 0.2, "fallback": 5.0}))
    (out / "config_tampered.json").write_text(json.dumps(cfg2))
    r = run(str(out / "config_tampered.json"))
    record(
        "a changed artefact is refused and writes no lock",
        r.returncode == 2
        and "differs from the listed" in r.stderr
        and not (out / "run2.lock").exists(),
        r.stderr.strip(),
    )
    (art / "persistence.json").write_text(json.dumps({"scale": 0.1, "fallback": 5.0}))
    # a wrong-origin row in one ensemble file
    bad_dir = out / "ens_bad"
    bad_dir.mkdir()
    for f in (E607 / "ens_summary").glob("init=2026-06-0[1-7].parquet"):
        (bad_dir / f.name).symlink_to(f)
    for f in (E607 / "ens_summary").glob("init=2026-05-*.parquet"):
        (bad_dir / f.name).symlink_to(f)
    target = bad_dir / "init=2026-06-03.parquet"
    target.unlink()
    pl.read_parquet(E607 / "ens_summary" / "init=2026-06-03.parquet").with_row_index().with_columns(
        power_fcst_init_time=pl.when(pl.col("index") == 5)
        .then(pl.col("power_fcst_init_time") + pl.duration(hours=1))
        .otherwise(pl.col("power_fcst_init_time"))
    ).drop("index").write_parquet(target)
    cfg3 = json.loads(cfg_path.read_text()) | {
        "lock_file": str(out / "run3.lock"),
        "output_dir": str(out / "run3_out"),
    }
    cfg3["inputs"] = {**cfg3["inputs"], "ens_summary_dir": str(bad_dir)}
    (out / "config_bad_row.json").write_text(json.dumps(cfg3))
    r = run(str(out / "config_bad_row.json"), "--check")
    record(
        "one wrong-origin row stops the check",
        r.returncode == 2 and "carry another power_fcst_init_time" in r.stderr,
        r.stderr.strip(),
    )
    r = run(str(out / "config_bad_row.json"))
    record(
        "one wrong-origin row stops the run",
        r.returncode == 2 and "carry another power_fcst_init_time" in r.stderr,
        r.stderr.strip(),
    )
    # a missing ensemble run (the newest run is not available): remove 2026-06-05
    missing_dir = out / "ens_missing"
    shutil.copytree(bad_dir, missing_dir, symlinks=True)
    (missing_dir / "init=2026-06-03.parquet").unlink()
    (missing_dir / "init=2026-06-03.parquet").symlink_to(
        E607 / "ens_summary" / "init=2026-06-03.parquet"
    )
    (missing_dir / "init=2026-06-05.parquet").unlink()
    cfg4 = json.loads(cfg_path.read_text()) | {
        "lock_file": str(out / "run4.lock"),
        "output_dir": str(out / "run4_out"),
    }
    cfg4["inputs"] = {**cfg4["inputs"], "ens_summary_dir": str(missing_dir)}
    (out / "config_missing_run.json").write_text(json.dumps(cfg4))
    r = run(str(out / "config_missing_run.json"), "--check")
    record(
        "a missing ensemble run stops the check",
        r.returncode == 2 and "is missing" in r.stderr,
        r.stderr.strip(),
    )
    # development mode refuses July
    cfg5 = json.loads(cfg_path.read_text()) | {
        "origin_last": "2026-07-01",
        "lock_file": str(out / "run5.lock"),
    }
    (out / "config_july.json").write_text(json.dumps(cfg5))
    r = run(str(out / "config_july.json"))
    record(
        "development mode refuses an origin in July",
        r.returncode == 2 and "never opens" in r.stderr and not (out / "run5.lock").exists(),
        r.stderr.strip(),
    )
    # final mode without approval, then without the flag
    cfg6 = json.loads(cfg_path.read_text()) | {
        "mode": "final",
        "origin_first": "2026-07-01",
        "origin_last": "2026-10-31",
        "lock_file": str(out / "run6.lock"),
    }
    (out / "config_final.json").write_text(json.dumps(cfg6))
    r = run(str(out / "config_final.json"))
    record(
        "final mode without the flag is refused",
        r.returncode == 2 and "--run-final-test" in r.stderr,
        r.stderr.strip(),
    )
    r = run(str(out / "config_final.json"), "--run-final-test")
    record(
        "final mode without the maintainer's approval is refused",
        r.returncode == 2
        and "approved_by_maintainer" in r.stderr
        and not (out / "run6.lock").exists(),
        r.stderr.strip(),
    )
    # the 24 h lag rule: with the cap lowered to 12 h, the rule must stop the run
    sys.path.insert(0, str(RUNNER.parent))
    import e00558_lags  # ty: ignore[unresolved-import]
    import e00614_final_runner as runner  # ty: ignore[unresolved-import]

    saved = e00558_lags.CAP_STEPS
    e00558_lags.CAP_STEPS = 24
    try:
        runner.origin_rows(
            origin=runner.parse_day("2026-06-02") + runner.PUBLICATION_DELAY,
            ens_dir=E607 / "ens_summary",
            observed=observed,
        )
        record("a lag younger than 24 h stops the run", False, "no error raised")
    except runner.RunnerError as err:
        record(
            "a lag younger than 24 h stops the run",
            "younger than 24 h" in str(err) or "24 h" in str(err),
            str(err),
        )
    finally:
        e00558_lags.CAP_STEPS = saved
    lines.append("scores of the dummy candidates (development month, not meaningful as skill):")
    lines.append(pl.read_csv(Path(base["output_dir"]) / "scores.csv").__str__())
    lines.append(f"{sum(results.values())} of {len(results)} checks passed")
    (out / "dev_test_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

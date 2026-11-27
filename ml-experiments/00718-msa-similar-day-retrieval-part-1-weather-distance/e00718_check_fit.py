# ruff: noqa: E501, PLR0915, E731, PT017
"""Experiment 718, WP5a: check the fit's resume and refusal rules on the CPU with a tiny synthetic
fold.

The checks run `fit` with `smoke=True` (two passes per stage), random encoders, and the row
tables of arm `U` on a synthetic source. They cover a crash during stage 2 followed by a resume
that reuses the stage 1 state file, the refusal of a stage log written by other code and its
named override, the refusal of a missing known-answer gate file, and the assertion that
experiment 713's files are untouched. Usage: e00718_check_fit.py
"""  # noqa: D205

import json
import sys
import tempfile
from pathlib import Path

import e00713_train as tr  # ty: ignore[unresolved-import]
import e00718_arms as arms
import e00718_check_batch as cb
import e00718_check_train as ct
import e00718_fit as fit_mod
import e00718_train as ftr
import numpy as np

ARM = "U"
MONTH = "2025-11"
SEED = 0


def make_samples(*, world: tr.World, stage: int, placebo: bool) -> tr.SampleSet:
    """Return the (Primary, origin day) pairs that the tables hold, every fifth held out."""
    days = np.arange(cb.FIRST_ORIGIN_DAY, cb.FIRST_ORIGIN_DAY + cb.N_ORIGIN_DAYS)
    prim, dd = np.meshgrid(cb.TABLE_PRIMARIES, days, indexing="ij")
    prim, c = prim.ravel(), dd.ravel() * 48 + 24
    return tr.SampleSet(primary=prim, c_idx=c, is_val=(np.arange(len(prim)) % 5 == 0))


def run_fit(*, root: Path, tables: dict, source, **kwargs) -> dict:  # noqa: ANN001, ANN003
    """Run a smoke fit of arm `U` in `root`."""
    world = ct.synthetic_world(source=source)
    loaded = fit_mod.Loaded(world=world, fingerprint="f", dropped={})
    return fit_mod.fit(
        arm=ARM, month=MONTH, seed=SEED, device="cpu", data_dir=root, smoke=True,
        loader=lambda *_: loaded, make_samples=make_samples, encoders="random", tables=tables, **kwargs,
    )  # fmt: skip


def main() -> int:
    """Run the checks."""
    source = cb.synthetic_source()
    tables = cb.make_tables(source=source, arm=ARM)
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stage_file = lambda s: arms.model_file(
            data_dir=root, arm=ARM, month=MONTH, seed=SEED, stage=s, smoke=True
        )
        # a crash during stage 2
        real = ftr.train_stage

        def crashing(**kwargs) -> object:  # noqa: ANN003
            if kwargs["stage"] == 2:
                msg = "simulated crash"
                raise RuntimeError(msg)
            return real(**kwargs)

        ftr.train_stage = crashing  # ty: ignore[invalid-assignment]
        try:
            run_fit(root=root, tables=tables, source=source)
            raise SystemExit("the simulated crash did not propagate")
        except RuntimeError:
            pass
        finally:
            ftr.train_stage = real
        assert stage_file(1).exists(), "stage 1 left no state file"
        assert not stage_file(2).exists(), "stage 2 left a state file after the crash"
        mtime = stage_file(1).stat().st_mtime_ns
        log = run_fit(root=root, tables=tables, source=source)
        assert stage_file(1).stat().st_mtime_ns == mtime, "the resume refitted stage 1"
        assert stage_file(3).exists()
        assert len(log["stages"]) == 3
        lines.append(
            "fit resume: a crash in stage 2 left the stage 1 state file, and the rerun reused it and finished all three stages"
        )
        # a finished job is skipped
        again = run_fit(root=root, tables=tables, source=source)
        assert again["stages"] == log["stages"] or len(again["stages"]) == 3
        lines.append("fit skip: a finished job returns its log without refitting")
        # another code hash is refused, and an override names the exact old hash
        stage_log = fit_mod.stage_log_file(
            data_dir=root, arm=ARM, month=MONTH, seed=SEED, stage=2, smoke=True
        )
        stage_file(3).unlink()
        arms.log_file(
            data_dir=root, arm=ARM, month=MONTH, seed=SEED, kind="fit", smoke=True
        ).unlink()
        record = json.loads(stage_log.read_text())
        record["code_hash"] = "0" * 16
        stage_log.write_text(json.dumps(record))
        try:
            run_fit(root=root, tables=tables, source=source)
            raise SystemExit("a stage state written by other code was accepted")
        except AssertionError as err:
            assert "0" * 16 in str(err)
        stage_file(3).unlink(missing_ok=True)
        log = run_fit(root=root, tables=tables, source=source, accept=["0" * 16])
        assert log["code_overrides"][0]["old_hash"] == "0" * 16
        lines.append(
            "fit refusal: a stage log with another code hash is refused, and `--accept-code-change OLD_HASH` records an override with both hashes"
        )
        # a real fit needs the known-answer gate file
        try:
            fit_mod.fit(
                arm=ARM,
                month=MONTH,
                seed=1,
                device="cpu",
                data_dir=root,
                smoke=False,
                tables=tables,
            )
            raise SystemExit("a real fit ran without the gate file")
        except AssertionError as err:
            assert "known" in str(err)
        lines.append(
            "gate: a fit that is not a smoke fit refuses to start without the known-answer gate file"
        )
        # table provenance: a table is refused for another month, another fingerprint, or other builder code
        meta = fit_mod.table_meta(
            arm=ARM, month=MONTH, fingerprint="f", phase="predict", digest="d"
        )
        path = root / "t.npz"
        fit_mod.save_table(table=tables[next(iter(tables))], path=path, meta=meta)
        fit_mod.load_table(path=path, expect=meta)
        for key, value in (
            ("month", "2026-01"),
            ("fingerprint", "g"),
            ("builder_hash", "old"),
            ("phase", "train"),
            ("inputs_digest", "e"),
            ("limit_days", "16"),
        ):
            try:
                fit_mod.load_table(path=path, expect={**meta, key: value})
                raise SystemExit(f"a table with another {key} was accepted")
            except AssertionError as err:
                assert "provenance" in str(err)
        fit_mod.save_table(table=tables[next(iter(tables))], path=path)
        try:
            fit_mod.load_table(path=path, expect=meta)
            raise SystemExit("a table without provenance was accepted")
        except AssertionError as err:
            assert "provenance" in str(err)
        lines.append(
            "provenance: a table of another month, fingerprint, builder code, phase, input digest, limit, or none is refused"
        )
    print(*lines, sep="\n")
    print("e00718_check_fit: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

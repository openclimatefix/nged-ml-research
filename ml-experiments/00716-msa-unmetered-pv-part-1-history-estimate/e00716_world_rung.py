# ruff: noqa: E501, PT018
"""Experiment 716: choose the rung of the synthetic worlds from the NumPy references (WP5a; design section 8.2).

Usage: run.sh e00716_world_rung.py [--fold 2026-01] [--rungs 1 2 3 4 5] [--force] [--after N]
       run.sh e00716_world_rung.py --selftest

This is a CPU task that reads the dense weather table and fits no neural network. For each rung in
the fixed order, it builds worlds WH and WT for world draws 0 to 7, fits the naive and the oracle ridge
reference of each Primary, and prints the `gap` of each world and draw (the naive reference's MAE of
the median minus the oracle's, on solar-block validation tokens pooled over the four Primaries, on the
236 days after the cutoff). A rung passes when, in both worlds, the mean `gap` over the eight draws and
the `gap` of draw 0 are both at least 0.010 (design addendum 1); a rung whose mean passes and whose
draw 0 fails is a failed rung. It stops at the first passing rung and writes `work/world_rung.json`
with every `gap` it computed, the eight draws, their mean and standard deviation, and `held_out_days`. If no rung passes, it writes the file with `rung: null` and exits with status 2, so that the
manager is told and F7 stops before any PV fit. A rung whose file `work/world_rung/rung<N>.json`
exists is not recomputed unless `--force` is given.

`--after N` steps the ladder up after the early-stopping pre-check failed at rung N: rungs up to N are kept in the
file but the first passing rung above N is chosen, and `known_answer.py` then fits `M` again at that rung.

`--selftest` checks the block ridge against a direct solve and the world's formula on small arrays.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_pv as pv
import e00716_worlds as wd
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME_713_CACHE = "00713-msa-axial-attention-demand-network-part-1"


def fold_cutoff_idx(*, fold: str) -> int:
    """Return the grid index of the first day of the fold's month at 00:00 UTC."""
    year, month = (int(x) for x in fold.split("-"))
    stamp_us = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1e6)
    return int((stamp_us - lay.GRID_START_US) // lay.STEP_US)


def load_dense4() -> wd.Dense4:
    """Read the stored dense table of experiment 713 (read-only) and the calendar of the whole grid."""
    path = ML_EXPERIMENTS_DATA_DIR / NAME_713_CACHE / "cache" / "dense_raw.npy"
    raw = np.load(path, mmap_mode="r")
    n_grid = raw.shape[1]
    holidays = lay.england_bank_holidays(years=range(2019, 2029))
    day_type = lay.day_types_for_grid(
        grid_start_us=lay.GRID_START_US, n_grid=n_grid, holidays=holidays
    )
    half_hour = lay.local_half_hours_for_grid(grid_start_us=lay.GRID_START_US, n_grid=n_grid)
    doy = pv.day_of_year(grid_start_us=lay.GRID_START_US, n_grid=n_grid)
    return wd.dense4_from_raw(
        raw=raw, day_type=day_type.astype(np.int64), half_hour=half_hour.astype(np.int64), doy=doy
    )


def rung_record_problems(*, record: dict) -> list[str]:
    """Return why a stored rung or `world_rung.json` record is not an eight-draw record on the current validation window (empty when it is).

    A record from the 60-day, single-draw rule lacks `held_out_days`, `draws`, `mean`, and `sd`, and is
    refused. So is a record with no results, a rung with no worlds, a world with the wrong number of
    draws, and (for `world_rung.json`) a chosen rung that is absent from `results` or does not pass.
    """
    out = []
    if record.get("held_out_days") != wd.HELD_OUT_DAYS:
        out.append(
            f"held_out_days is {record.get('held_out_days')}, not {wd.HELD_OUT_DAYS}: the record is from another window"
        )
    results = record.get("results", {"_": record})
    if not results:
        out.append("no rung results")
    for number, res in results.items():
        gaps = res.get("gaps", {})
        if set(gaps) != {"WH", "WT"}:
            out.append(f"rung {number}: holds worlds {sorted(gaps)}, not WH and WT")
        for kind, g in gaps.items():
            missing = [k for k in ("draws", "mean", "sd") if k not in g]
            if missing or len(g.get("draws", [])) != wd.N_DRAWS:
                out.append(
                    f"rung {number} {kind}: lacks the {wd.N_DRAWS}-draw fields {missing or 'draws'}"
                )
    chosen = record.get("rung")
    if "results" in record and chosen is not None:
        res = results.get(str(chosen))
        if res is None:
            out.append(f"the chosen rung {chosen} is absent from the results")
        elif not out and not rung_passes(gaps=res["gaps"]):
            out.append(f"the chosen rung {chosen} does not pass the rule")
    return out


def stored_rung_problems(*, name: str, stored: dict) -> list[str]:
    """Return why a stored `rung<N>.json` cannot be reused: other world code, a dirty tree, or a record that is not an eight-draw record of the current window (F12)."""
    return [
        *arms.artefact_problems(what=name, record=stored, files=arms.WORLD_FILES),
        *rung_record_problems(record=stored),
    ]


def rung_passes(*, gaps: dict[str, dict]) -> bool:
    """Return whether a rung passes: in every world, the mean `gap` over the draws and the `gap` of draw 0 are both at least `GAP_MIN`."""
    return bool(
        gaps and all(g["mean"] >= wd.GAP_MIN and g["draws"][0] >= wd.GAP_MIN for g in gaps.values())
    )


def summarise_draws(*, results: list[dict[str, wd.GapResult]]) -> dict[str, dict]:
    """Return per world: draw 0's full result under `gap`, and the draws, their mean, and their standard deviation (ddof 1)."""
    out = {}
    for kind in results[0]:
        gaps = [r[kind].gap for r in results]
        first = results[0][kind]
        out[kind] = {
            "gap": first.gap, "gap_per_primary": first.gap_per_primary, "mae_naive": first.mae_naive,
            "mae_oracle": first.mae_oracle, "penalties": first.penalties, "draws": gaps,
            "mean": float(np.mean(gaps)), "sd": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
        }  # fmt: skip
    return out


def run_rungs(
    *,
    dense: wd.Dense4,
    cutoff_idx: int,
    rungs: list[int],
    out_dir: Path,
    force: bool,
    after: int = 0,
) -> dict:
    """Compute the `gap` of WH and WT at draws 0 to 7 per rung in order and stop at the first rung that passes.

    A rung passes when, in both worlds, the mean over the draws and the `gap` of draw 0 reach `GAP_MIN`.
    A rung numbered `after` or lower is read or computed and recorded but never chosen: this is the step up
    after the early-stopping pre-check failed at rung `after` (design 8.2), so the ladder continues from the next rung.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    chosen: int | None = None
    for number in rungs:
        rung = wd.RUNGS[number - 1]
        path = out_dir / f"rung{number}.json"
        stored = json.loads(path.read_text()) if path.exists() and not force else None
        problems = stored_rung_problems(name=path.name, stored=stored) if stored is not None else []
        if stored is not None and problems:
            print(f"rung {number}: recomputed ({'; '.join(problems)})", flush=True)
        if stored is not None and not problems:
            record = stored
            print(f"rung {number}: read from {path.name}", flush=True)
        else:
            started = time.monotonic()
            per_draw = []
            for draw in range(wd.N_DRAWS):
                world = wd.build_world(dense=dense, rung=rung, seed=draw)
                found = {
                    kind: wd.gap_of_world(world=world, dense=dense, kind=kind, cutoff_idx=cutoff_idx, log=False)
                    for kind in ("WH", "WT")
                }  # fmt: skip
                per_draw.append(found)
                print(
                    f"rung {number} draw {draw}: "
                    + ", ".join(f"{k} gap {r.gap:.4f}" for k, r in found.items()),
                    flush=True,
                )
            gaps = summarise_draws(results=per_draw)
            for kind, g in gaps.items():
                print(
                    f"rung {number} {kind}: draw 0 {g['gap']:.4f}, mean {g['mean']:.4f}, sd {g['sd']:.4f}, "
                    f"draws {[round(x, 4) for x in g['draws']]}",
                    flush=True,
                )
            record = {
                "rung": number, "settings": vars(rung), "gaps": gaps, "seconds": round(time.monotonic() - started, 1),
                "passes": rung_passes(gaps=gaps), "held_out_days": wd.HELD_OUT_DAYS, "n_draws": wd.N_DRAWS,
                "code": arms.code_record(),
            }  # fmt: skip
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(record, indent=1, default=str))
            tmp.rename(path)
        results[str(number)] = record
        if record["passes"] and number > after:
            chosen = number
            break
    return {
        "rung": chosen,
        "gap_min": wd.GAP_MIN,
        "held_out_days": wd.HELD_OUT_DAYS,
        "n_draws": wd.N_DRAWS,
        "results": results,
        "stepped_up_after": after or None,
    }


def precheck_path(*, work: Path, rung: int) -> Path:
    """Return the file that `e00716_known_answer.py precheck` writes for `rung`: `work/known_answers/precheck_rung<N>.json` (`known_answer.precheck_file` must agree)."""
    return work / "known_answers" / f"precheck_rung{rung}.json"


def step_up_problems(*, after: int, precheck_path: Path) -> list[str]:
    """Return why `--after N` is refused: it needs `precheck_rung<N>.json` to exist, to record a failure, and to come from the current world code."""
    if not precheck_path.exists():
        return [
            f"{precheck_path.name} does not exist: the step up follows a failed pre-check at rung {after}"
        ]
    record = json.loads(precheck_path.read_text())
    out = arms.artefact_problems(what=precheck_path.name, record=record, files=arms.WORLD_FILES)
    if record.get("pass") is not False or record.get("rung") != after:
        out.append(f"{precheck_path.name} does not record a failed pre-check at rung {after}")
    return out


def check_finite_weather(*, cutoff_idx: int) -> str:
    """Run the finite-weather check of the validation window on the stored dense table and the B1 tables of experiment 713 (design addendum 1, F11)."""
    import e00632_data as d632  # ty: ignore[unresolved-import]

    cache = ML_EXPERIMENTS_DATA_DIR / NAME_713_CACHE / "cache"
    raw = np.load(cache / "dense_raw.npy", mmap_mode="r")
    b1_grid = np.load(cache / "dense_b1_grid.npy", mmap_mode="r")
    cams = np.load(cache / "dense_b1_cams.npy")
    stamp_us = lay.GRID_START_US + np.arange(raw.shape[1], dtype=np.int64) * lay.STEP_US
    day_code = np.clip(stamp_us // d632.DAY_US - 1 - d632.DAY0, 0, d632.N_DAYS - 1)
    return wd.check_validation_weather_finite(
        raw=raw, b1_grid=b1_grid, cams=cams, day_code=day_code, cutoff_idx=cutoff_idx
    )


def selftest_records() -> None:
    """Check that a rung file with no code record is not reused and that a step up needs a failed pre-check record."""
    import tempfile

    stale = {"rung": 4, "passes": True, "gaps": {}}
    assert arms.artefact_problems(what="rung4.json", record=stale, files=arms.WORLD_FILES), (
        "a rung file with no code record was accepted"
    )
    clean = {**arms.code_record(), "git_dirty": False}
    with tempfile.TemporaryDirectory() as tmp:
        check = Path(tmp) / "precheck_rung4.json"
        assert step_up_problems(after=4, precheck_path=check), (
            "a step up with no pre-check was accepted"
        )
        check.write_text(json.dumps({"rung": 4, "pass": True, **clean}))
        assert step_up_problems(after=4, precheck_path=check), (
            "a step up after a passed pre-check was accepted"
        )
        check.write_text(json.dumps({"rung": 4, "pass": False, **clean}))
        problems = step_up_problems(after=4, precheck_path=check)
        assert not problems, problems
    print(
        "a step up needs a failed pre-check record of that rung, and a rung file with no code record is not reused"
    )


def selftest_rule() -> None:
    """Check each condition of the rung rule and of the record checks on its own, so that removing any one of them fails a test."""
    clean = {**arms.code_record(), "git_dirty": False}

    def world(first: float, rest: float, n: int = wd.N_DRAWS) -> dict:
        draws = [first] + [rest] * (n - 1)
        return {"gap": first, "draws": draws, "mean": float(np.mean(draws)), "sd": 0.001}

    both = lambda g: {"WH": g, "WT": g}  # noqa: E731
    assert rung_passes(gaps=both(world(0.0146, 0.0125)))
    assert not rung_passes(gaps=both(world(0.012, 0.0075))), (
        "draw 0 passes and the mean fails: the rung must fail"
    )
    assert not rung_passes(gaps=both(world(0.009, 0.0125))), (
        "the mean passes and draw 0 fails: the rung must fail"
    )
    assert not rung_passes(gaps={"WH": world(0.0146, 0.0125), "WT": world(0.012, 0.0075)}), (
        "one world's mean fails"
    )
    one = {**world(0.0146, 0.0125, n=1)}
    rec = {
        "rung": 4,
        "held_out_days": wd.HELD_OUT_DAYS,
        "n_draws": 1,
        "gaps": both(one),
        "code": clean,
    }
    assert rung_record_problems(record=rec), "a record with one draw (and mean and sd) was accepted"
    assert stored_rung_problems(name="rung4.json", stored=rec), (
        "a stored single-draw rung file with a current code record was reused"
    )
    full = {**rec, "gaps": both(world(0.0146, 0.0125)), "n_draws": wd.N_DRAWS}
    assert not stored_rung_problems(name="rung4.json", stored=full), stored_rung_problems(
        name="rung4.json", stored=full
    )
    top = {"rung": 4, "held_out_days": wd.HELD_OUT_DAYS, "code": clean}
    assert rung_record_problems(record={**top, "results": {}}), (
        "a record with empty results was accepted"
    )
    assert rung_record_problems(record={**top, "results": {"4": {"gaps": {}}}}), (
        "a rung with no worlds was accepted"
    )
    assert rung_record_problems(record={**top, "results": {"3": full}}), (
        "a chosen rung absent from the results was accepted"
    )
    assert rung_record_problems(
        record={**top, "results": {"4": {"gaps": both(world(0.012, 0.0075))}}}
    ), "a chosen rung that fails was accepted"
    assert not rung_record_problems(
        record={**top, "results": {"4": {"gaps": both(world(0.0146, 0.0125))}}}
    )
    print(
        "rung rule: draw 0 and the mean are each required, and records with one draw, no results, or a failing chosen rung are refused"
    )


def selftest_ladder(*, dense: wd.Dense4, real_days: int) -> None:
    """Check the step up, the mean-and-draw-0 rule, and the refusal of a single-draw record on stored toy rung files."""
    wd.HELD_OUT_DAYS = real_days  # ty: ignore[invalid-assignment]
    # the step up: with the pre-check failed at rung 4, the chosen rung must be above 4
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # rung 3: the mean passes and draw 0 fails, so the rung fails; rung 4 passes; rung 5 passes (the rescue case)
        toy = {
            1: (0.004, 0.004),
            2: (0.005, 0.005),
            3: (0.012, 0.009),
            4: (0.0129, 0.0146),
            5: (0.0149, 0.0147),
        }
        for n, (mean, first) in toy.items():
            gaps = {
                k: {
                    "mean": mean,
                    "draws": [first] + [mean] * (wd.N_DRAWS - 1),
                    "sd": 0.001,
                    "gap": first,
                }
                for k in ("WH", "WT")
            }
            (out / f"rung{n}.json").write_text(
                json.dumps(
                    {
                        "rung": n,
                        "passes": rung_passes(gaps=gaps),
                        "gaps": gaps,
                        "held_out_days": wd.HELD_OUT_DAYS,
                        "n_draws": wd.N_DRAWS,
                        "code": {**arms.code_record(), "git_dirty": False},
                    }
                )
            )
        assert not json.loads((out / "rung3.json").read_text())["passes"], (
            "a rung whose mean passes and whose draw 0 fails must fail"
        )
        assert not rung_record_problems(record=json.loads((out / "rung5.json").read_text()))
        old = {"rung": 4, "passes": True, "gaps": {"WH": {"gap": 0.0165}}}
        assert rung_record_problems(record=old), "a single-draw, 60-day record was accepted"
        assert (
            run_rungs(dense=dense, cutoff_idx=0, rungs=[1, 2, 3, 4, 5], out_dir=out, force=False)[
                "rung"
            ]
            == 4
        )
        stepped = run_rungs(
            dense=dense, cutoff_idx=0, rungs=[1, 2, 3, 4, 5], out_dir=out, force=False, after=4
        )
        assert stepped["rung"] == 5 and stepped["stepped_up_after"] == 4, stepped
    print(
        "the ladder steps up to rung 5 after a failed pre-check at rung 4, a rung whose mean passes and whose draw 0 fails is failed, and a single-draw record is refused"
    )


def selftest() -> int:
    """Check the block ridge against a direct solve, the world's formula, and the order of the gap on a toy world."""
    rng = np.random.default_rng(0)
    n, p, k = 60, 12, 3
    hs = rng.normal(size=(n, p))
    z = rng.normal(size=(n, wd.N_LEADS, 48, k))
    z[..., 0] = 1.0
    y = rng.normal(size=(n, wd.N_LEADS, 48)) + z[..., 1] * 0.5 + (hs[:, 0])[:, None, None]
    piece = wd.piece_of(hs=hs, z=z, y=y)
    fit = wd.prepare(piece=piece)
    lam = 37.0
    w, v = wd.solve(fit=fit, penalty=lam)
    for h in (0, 17, 47):
        rows = np.concatenate(
            [np.repeat(hs, wd.N_LEADS, axis=0), z[:, :, h].reshape(-1, k)], axis=1
        )
        gram = rows.T @ rows
        gram[:p, :p] += lam * np.eye(p)
        gram[p:, p:] += 1e-8 * np.eye(k)
        direct = np.linalg.solve(gram, rows.T @ y[:, :, h].reshape(-1))
        assert np.allclose(direct[:p], w[h], atol=1e-8) and np.allclose(
            direct[p:], v[h], atol=1e-8
        ), h
    a, b = hs[:40], hs[40:]
    sub = wd.piece_of(hs=a, z=z[:40], y=y[:40]) + wd.piece_of(hs=b, z=z[40:], y=y[40:])
    assert (
        np.allclose(sub.a, piece.a) and np.allclose(sub.b, piece.b) and np.allclose(sub.q, piece.q)
    )
    print("block ridge equals the direct normal equations (3 half-hours), and pieces add")
    # a toy world on a small grid
    n_grid = 48 * 400
    steps = np.arange(n_grid)
    hour = (steps % 48) / 2.0
    elevation = np.sin(np.pi * np.clip((hour - 4.0) / 16.0, 0.0, 1.0))
    season = 0.6 + 0.4 * np.sin(2 * np.pi * steps / (48 * 365))
    cosz = np.clip(0.9 * elevation * season, 0.0, 1.0) - 0.2 * (elevation <= 0)
    sky = np.repeat(rng.uniform(0.2, 1.0, size=(4, n_grid // 48)), 48, axis=1)
    dense = wd.Dense4(
        ghi=1000.0 * np.clip(cosz, 0, None)[None] * sky, temperature_c=np.tile(8.0 + 6 * np.sin(steps / 48 / 58.0), (4, 1)),
        dew_c=np.tile(4.0 + 3 * np.sin(steps / 48 / 40.0), (4, 1)), precip=np.zeros((4, n_grid)),
        cos_zenith=np.tile(cosz, (4, 1)), sin_azimuth=np.tile(np.sin(np.pi + (hour - 12) / 12 * np.pi / 2), (4, 1)),
        cos_azimuth=np.tile(np.cos(np.pi + (hour - 12) / 12 * np.pi / 2), (4, 1)),
        day_type=rng.integers(0, 8, n_grid), half_hour=(steps % 48).astype(np.int64), doy=(1.0 + (steps // 48) % 365),
    )  # fmt: skip
    real_days = wd.HELD_OUT_DAYS
    wd.HELD_OUT_DAYS = 60  # ty: ignore[invalid-assignment]  # the toy grid is 400 days long
    world = wd.build_world(dense=dense, rung=wd.RUNGS[4])
    assert np.isfinite(world.y).all()
    night = dense.cos_zenith <= 0
    assert np.abs(world.resid[night]).max() < 6 * wd.NOISE_SD, (
        "the night tokens must carry only noise"
    )
    assert (world.d >= 0).all() and world.d.max() <= 1.0 + 1e-9
    res = wd.gap_of_world(world=world, dense=dense, kind="WH", cutoff_idx=48 * 300, log=False)
    assert res.mae_oracle <= res.mae_naive + 0.002, (res.mae_naive, res.mae_oracle)
    res_t = wd.gap_of_world(world=world, dense=dense, kind="WT", cutoff_idx=48 * 300, log=False)
    assert res_t.mae_oracle <= res_t.mae_naive + 0.002, (res_t.mae_naive, res_t.mae_oracle)
    print(
        f"toy world: WH naive {res.mae_naive:.4f} oracle {res.mae_oracle:.4f}; WT naive {res_t.mae_naive:.4f} oracle {res_t.mae_oracle:.4f}"
    )
    selftest_ladder(dense=dense, real_days=real_days)
    selftest_rule()
    selftest_records()
    print("selftest passed")
    return 0


def main() -> int:
    """Run the self-test or the ladder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="2026-01")
    parser.add_argument("--rungs", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--after",
        type=int,
        default=0,
        help="step up: never choose a rung numbered at or below this (the rung whose early-stopping pre-check failed)",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    work = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="Z") / "work"
    if args.after:
        problems = step_up_problems(
            after=args.after, precheck_path=precheck_path(work=work, rung=args.after)
        )
        assert not problems, "; ".join(problems)
    dense = load_dense4()
    cutoff = fold_cutoff_idx(fold=args.fold)
    print(check_finite_weather(cutoff_idx=cutoff), flush=True)
    summary = run_rungs(
        dense=dense,
        cutoff_idx=cutoff,
        rungs=args.rungs,
        out_dir=work / "world_rung",
        force=args.force,
        after=args.after,
    )
    summary["fold"] = args.fold
    summary["code"] = arms.code_record()
    path = work / "world_rung.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=str))
    tmp.rename(path)
    if summary["rung"] is None:
        print("NO RUNG PASSED: tell the manager; F7 stops before any PV fit", flush=True)
        return 2
    print(f"chosen rung: {summary['rung']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

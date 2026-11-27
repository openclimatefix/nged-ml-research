# ruff: noqa: E501
"""Experiment 733: the stage functions that read the real files (banks, channels, limits, gallery).

`e00733_cli.install_runtime` puts these functions into `STAGE_FUNCTIONS` when the command line starts,
never at import. Every function takes `(args, root)` and writes under `root` only.

A debug run on a scratch data root may pass `--limit-segments`, which keeps only the first segments
of each bank. The official run on the real data folder refuses that argument.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Final

import e00733_channels as ch
import e00733_core as core
import e00733_paths as paths
import e00733_rerun as rr
import e00733_scoring as scoring
import e00733_stages as st
import e00733_synth as sy
import e00733_world as world
import numpy as np

TRAIN_WINDOW_START: Final = "2024-04-01"
"""The first day of the fold-coverage training window: the first run of experiment 607."""
SERIES_ID_D8: Final = 26
BANK_FILES: Final = {
    "train_s3": ("train", 3), "train_s4": ("train", 4), "validation": ("validation", None),
    "test": ("test", None),
}  # fmt: skip
"""The synthetic world's banks and the experiment 730 bank (kind, seed) behind each."""


class RuntimeError733(RuntimeError):
    """A stage cannot run as asked (a refused argument or a missing input)."""


def epoch_day(*, text: str) -> int:
    """Return the epoch day of an ISO date."""
    return int(np.datetime64(text, "D").astype(np.int64))


def fold_days(*, fold: str) -> tuple[int, int]:
    """Return the first and the last epoch day of a fold month."""
    first = np.datetime64(fold + "-01", "D")
    last = (first.astype("datetime64[M]") + np.timedelta64(1, "M")).astype(
        "datetime64[D]"
    ) - np.timedelta64(1, "D")
    return int(first.astype(np.int64)), int(last.astype(np.int64))


def check_limit_allowed(*, args: argparse.Namespace) -> int | None:
    """Return `--limit-segments`, refused unless a scratch data root was given.

    Args:
        args: The parsed arguments.

    Returns:
        The number of segments to keep per bank, or None.

    Raises:
        RuntimeError733: When the limit is asked for without `--data-root`.
    """
    limit = getattr(args, "limit_segments", None)
    if limit is not None and getattr(args, "data_root", None) is None:
        msg = "--limit-segments is for a scratch data root; the official run builds every segment"
        raise RuntimeError733(msg)
    return limit


# --------------------------------------------------------------------------- banks


def reading_ok_on_grid(*, cal: Any, use: np.ndarray, n_grid: int) -> np.ndarray:
    """Return whether D8's reading exists at each grid index, from experiment 729's calendar."""
    epoch = cal.dates.astype("datetime64[D]").astype(np.int64)
    ok = np.isfinite(cal.reading) & use[:, None]
    gi = ch.grid_index(
        day=np.repeat(epoch[:, None], 48, axis=1), hh=np.tile(np.arange(48), (len(epoch), 1))
    )
    inside = ok & (gi >= 0) & (gi < n_grid)
    out = np.zeros(n_grid, dtype=bool)
    out[gi[inside]] = True
    return out


def stage_banks(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Check the banks' digests, write the run marker, and build the world's four banks.

    The weather cache is built once from experiment 713's pipeline (on the CPU). A bank whose cache
    file exists is not built again.

    Args:
        args: The parsed arguments (`--limit-segments`).
        root: This experiment's data folder.

    Returns:
        The counts of each bank.
    """
    import e00729_banks as b729  # ty: ignore[unresolved-import]
    import e00730_banks as b730  # ty: ignore[unresolved-import]
    import e00733_adapters as adapters
    import e00733_leakage as lk

    limit = check_limit_allowed(args=args)
    n_checked = lk.assert_bank_digests(root_730=paths.root_730())
    print(f"banks checked against their digest: {n_checked}")
    digest = str(b730.banks_digest_id(root=paths.root_730()))
    st.write_run_marker(root=root, banks_digest=digest)
    if not (world.world_dir(root=root) / world.WEATHER_FILE).exists():
        adapters.build_weather_cache(root=root, device="cpu")
    cache = world.read_weather_cache(root=root)
    run_days_dt = sy.summary_run_days(summary_dirs=paths.ens_summary_dirs())
    run_days = {int(np.datetime64(d, "D").astype("int64")) for d in run_days_dt}
    runs = world.RunCache(
        summary_dir=paths.ens_summary_dirs(), series_id=SERIES_ID_D8, weather=cache
    )
    env, _ctx = world.make_channel_env(root_732=paths.root_732(), root_730=paths.root_730())
    banks = {
        k: b729.load_bank(path=b730.bank_path(root=paths.root_730(), kind=kind, seed=seed))
        for k, (kind, seed) in BANK_FILES.items()
    }
    constants = world.reading_constants(
        segments=[s for k in ("train_s3", "train_s4") for s in banks[k].segments]
    )
    result: dict[str, Any] = {}
    for kind in world.KINDS:
        if world.world_path(root=root, kind=kind).exists():
            result[kind] = {"built_earlier": True}
            continue
        segments, counts = world.build_bank_world(
            bank=banks[kind], env=env, runs=runs, run_days=run_days, constants=constants,
            limit=limit,
        )  # fmt: skip
        if not segments:
            msg = f"the bank {kind} holds no covered segment, so the world has nothing to read"
            raise st.CorrectnessFailureError(msg)
        world.write_world(root=root, kind=kind, segments=segments, extra={**counts, "limit": limit})
        result[kind] = counts
        print(f"bank {kind}: {counts}")
    world.write_meta(
        root=root, payload={"constants": constants, "limit": limit, "banks_digest": digest}
    )
    return result


# --------------------------------------------------------------------------- channels


def rerun_all_origins(
    *, windows: Any, env: Any, first: int, last: int, run_exists: Any
) -> rr.RerunOutputs:  # fmt: skip
    """Re-run the untangling network on every origin day `first` to `last` (epoch days).

    An origin without the ensemble runs of days D - 1 and D has zero outputs and flags off.

    Args:
        windows: The real-window builder over a span that holds every origin and its windows.
        env: The channel environment (the frozen networks).
        first: The first origin, as an epoch day.
        last: The last origin, as an epoch day.
        run_exists: `(epoch_day) -> bool`, whether the 00Z run of an epoch day exists.

    Returns:
        The outputs for the consecutive grid days `first - GRID_START` to `last - GRID_START`.

    Raises:
        RuntimeError733: When no origin is covered (nothing was re-run).
    """
    offset = ch.GRID_START_EPOCH_DAY
    days = np.arange(first, last + 1)
    covered = [int(d) for d in days if run_exists(int(d) - 1) and run_exists(int(d))]
    if not covered:
        msg = "no origin has the ensemble runs of days D - 1 and D, so nothing was re-run"
        raise RuntimeError733(msg)
    got = rr.rerun_origins(
        origin_days=[d - offset for d in covered], build_window=windows.build_window,
        predict=env.predict, n_seeds=env.n_seeds,
    )  # fmt: skip
    values = np.zeros((len(days), 2, 48, core.N_CHANNELS), dtype=np.float32)
    ok = np.zeros((len(days), 2, 48), dtype=bool)
    where = np.asarray(covered) - first
    values[where], ok[where] = got.values, got.ok
    return rr.RerunOutputs(origin_days=days - offset, values=values, ok=ok)


def slice_rerun(
    *, outputs: rr.RerunOutputs, first_grid_day: int, last_grid_day: int
) -> rr.RerunOutputs:
    """Return the outputs of the origins `first_grid_day` to `last_grid_day` (grid day numbers)."""
    keep = (outputs.origin_days >= first_grid_day) & (outputs.origin_days <= last_grid_day)
    return rr.RerunOutputs(
        origin_days=outputs.origin_days[keep], values=outputs.values[keep], ok=outputs.ok[keep]
    )


def fold_coverages(
    *, reading_ok: np.ndarray, channel_ok: np.ndarray, folds: tuple[str, ...] = core.ALL_FOLDS
) -> dict[str, float]:
    """Return the coverage of each fold's training window and of each fold month.

    Args:
        reading_ok: Whether D8's reading exists at each grid index.
        channel_ok: Whether an untangled channel exists at each grid index.
        folds: The fold months.

    Returns:
        The share by name `<fold> train` and `<fold> month`.
    """
    offset = ch.GRID_START_EPOCH_DAY
    start = epoch_day(text=TRAIN_WINDOW_START)
    out: dict[str, float] = {}
    for fold in folds:
        first, last = fold_days(fold=fold)
        for name, lo, hi in (("train", start, first - 1), ("month", first, last)):
            origins = np.arange(lo, hi + 1) - offset
            out[f"{fold} {name}"] = ch.coverage(
                origin_days=origins, reading_ok=reading_ok, channel_ok=channel_ok
            )
    return out


def real_neighbours(*, cal: Any) -> dict[str, np.ndarray]:
    """Return the three neighbours' real series on experiment 729's calendar (read only).

    The synthetic world's neighbours (`ChannelEnv.neighbours`) lie on the base calendar of
    experiment 730's banks and have another length, so the real windows need their own.
    """
    import e00722_guard as guard  # ty: ignore[unresolved-import]
    import e00732_neighbours as nb732  # ty: ignore[unresolved-import]

    return nb732.build_arrays(dates=cal.dates, log=guard.ReadLog(), name="neighbours_real_days")


def stage_channels(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Apply the rule, write `channels.json` and the table, re-run every origin, and check coverage.

    The re-run outputs of every origin from 2024-04-01 (the first run of experiment 607) to the end
    of the last fold are written once per fold to `channels/rerun/<fold>.npz`, covering the origins
    from that date to the fold's last day.

    Args:
        args: The parsed arguments.
        root: This experiment's data folder.

    Returns:
        The rule's choice, the coverage by fold, and the re-run counts.
    """
    import e00733_real as rl

    reading = ch.read_reading_732(root=paths.root_732())
    choice = ch.decide(reading=reading)
    outputs = ch.load_outputs(choice=choice, root_732=paths.root_732(), root_730=paths.root_730())
    table = ch.build_table(outputs=outputs)
    cache = world.read_weather_cache(root=root)
    values_dense, channel_ok = table.dense(n_grid=cache.n_grid)
    del values_dense
    env, ctx = world.make_channel_env(root_732=paths.root_732(), root_730=paths.root_730())
    cal = ctx.table.calendar
    use = np.isfinite(cal.reading).any(axis=1)
    reading_ok = reading_ok_on_grid(cal=cal, use=use, n_grid=cache.n_grid)
    coverages = fold_coverages(reading_ok=reading_ok, channel_ok=channel_ok)
    for name, share in coverages.items():
        print(f"channel coverage {name}: {share:.3f}")
    ch.assert_coverage(coverages=coverages)
    ch.write_choice(path=paths.channels_json(root=root), choice=choice)
    ch.write_table(table=table, path=paths.table_path(root=root))
    first = epoch_day(text=TRAIN_WINDOW_START)
    last = fold_days(fold=core.ALL_FOLDS[-1])[1]
    cal_last = int(cal.dates[-1].astype("datetime64[D]").astype(np.int64))
    windows = rl.build_real_windows(
        cal=cal, reading_rule_use=use, holidays=ctx.cal.holidays, stats=ctx.stats,
        medians=env.medians, neighbours=real_neighbours(cal=cal), summary_dir=paths.ens_summary_dirs(),
        first_epoch_day=first - 8, last_epoch_day=min(last + 2, cal_last),
    )  # fmt: skip
    all_outputs = rerun_all_origins(
        windows=windows, env=env, first=first, last=last,
        run_exists=lambda d: windows_run_exists(windows=windows, epoch_day=d),
    )  # fmt: skip
    n_covered = int(all_outputs.ok.any(axis=(1, 2)).sum())
    for fold in core.ALL_FOLDS:
        rr.write_rerun(
            outputs=slice_rerun(
                outputs=all_outputs, first_grid_day=first - ch.GRID_START_EPOCH_DAY,
                last_grid_day=fold_days(fold=fold)[1] - ch.GRID_START_EPOCH_DAY,
            ),
            path=paths.rerun_dir(root=root) / f"{fold}.npz",
        )  # fmt: skip
    poison = poison_check(
        root=root, cal=cal, neighbours=real_neighbours(cal=cal), use=use, ctx=ctx, env=env,
        outputs=all_outputs, first_epoch_day=first,
    )  # fmt: skip
    return {
        "choice": choice.usable,
        "poison": poison,
        "coverage": coverages,
        "origins": len(all_outputs.origin_days),
        "origins_with_rerun": n_covered,
        "missing_ens_columns": windows.missing_ens_columns,
    }


POISON_FILE: Final = "scores/poison_test.json"
"""The record of the poison test on one real origin, which G14 reads."""


def poison_check(
    *, root: Path, cal: Any, neighbours: dict[str, np.ndarray], use: np.ndarray, ctx: Any, env: Any,
    outputs: rr.RerunOutputs, first_epoch_day: int,
) -> dict[str, Any]:  # fmt: skip
    """Re-run one real origin with NaN and 1e6 in every input after the limit and record the result.

    Args:
        root: This experiment's data folder (the record is `scores/poison_test.json`).
        cal: The calendar of the whole day table.
        neighbours: The neighbours' real series on the calendar axis.
        use: Whether each calendar day holds a reading.
        ctx: The channel context (`holidays`, `stats`).
        env: The channel environment (`medians`, `predict`, `n_seeds`).
        outputs: The re-run of every origin (the origin is the last one with a visible output).
        first_epoch_day: The first origin day, as an epoch day.

    Returns:
        The record: `passed`, `n_poison`, and `origin_day`.

    Raises:
        RuntimeError733: When no origin has a visible re-run output, or the re-run changed.
    """
    import e00733_real as rl

    covered = np.flatnonzero(outputs.ok.any(axis=(1, 2)))
    if not len(covered):
        msg = "no origin has a visible re-run output, so the poison test would check nothing"
        raise RuntimeError733(msg)
    origin = int(outputs.origin_days[covered[-1]])
    o_epoch = origin + ch.GRID_START_EPOCH_DAY

    def make(c: Any, nb: Any) -> Any:
        return rl.build_real_windows(
            cal=c, reading_rule_use=use, holidays=ctx.cal.holidays, stats=ctx.stats,
            medians=env.medians, neighbours=nb, summary_dir=paths.ens_summary_dirs(),
            first_epoch_day=o_epoch - 8, last_epoch_day=o_epoch + 2,
        )  # fmt: skip

    record: dict[str, Any] = {"origin_day": origin, "first_epoch_day": first_epoch_day}
    try:
        n_poison = rl.poisoned_origin_run(
            make_windows=make, cal=cal, neighbours=neighbours, origin_day=origin,
            predict=env.predict, n_seeds=env.n_seeds,
        )  # fmt: skip
    except rr.RerunError as exc:
        st.write_json(
            path=root / POISON_FILE, payload={**record, "passed": False, "n_poison": 0, "message": str(exc)}
        )  # fmt: skip
        msg = f"the poison test failed: {exc}"
        raise st.CorrectnessFailureError(msg) from exc
    record = {**record, "passed": n_poison > 0, "n_poison": n_poison}
    st.write_json(path=root / POISON_FILE, payload=record)
    return record


def poison_passed(*, root: Path) -> bool:
    """Return whether `scores/poison_test.json` records a passed poison test on at least one value."""
    path = root / POISON_FILE
    if not path.exists():
        return False
    rec = json.loads(path.read_text())
    return bool(rec.get("passed")) and int(rec.get("n_poison", 0)) > 0


def windows_run_exists(*, windows: Any, epoch_day: int) -> bool:
    """Return whether the 00Z ensemble run of an epoch day exists (a gap is data, not an error)."""
    try:
        windows.run(epoch_day=epoch_day)
    except rr.RerunError:
        return False
    return True


# --------------------------------------------------------------------------- limits


def train_segments(*, root: Path) -> list[sy.SegmentArrays]:
    """Return the segments of the synthetic training banks of seeds 3 and 4."""
    return [
        s for kind in ("train_s3", "train_s4") for s in world.read_world(root=root, kind=kind)[0]
    ]


def stage_limits(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Fit B-clim on the training banks and write `limits.json` (g, B-clim, and the counts).

    Args:
        args: The parsed arguments.
        root: This experiment's data folder.

    Returns:
        The content written.
    """
    import e00733_baselines as bl

    clim = bl.fit_clim(segments=train_segments(root=root))
    counts: dict[str, int] = {}
    for kind in world.KINDS:
        segments, _extra = world.read_world(root=root, kind=kind)
        index = sy.enumerate_samples(segments=segments)
        counts[kind] = len(index)
        counts[f"{kind}_scored"] = int(sy.scored(index=index).sum())
    payload = core.limits_payload(b_clim=clim.as_dict(), n_synthetic=counts)
    st.write_json(path=root / "limits.json", payload=payload)
    print(
        f"limits.json written: g {payload['smallest_gain']}, B-clim overall {clim.overall:.4f}, {counts}"
    )
    return payload


# --------------------------------------------------------------------------- gallery


def stage_gallery(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Draw panels G10 to G15 on the world's banks, check each against its list, write the check.

    G10 to G13 and G15 read the test bank; G14 reads the validation bank. The panels are drawn into
    `PRIVATE/gallery/`.

    Args:
        args: The parsed arguments.
        root: This experiment's data folder.

    Returns:
        The content of `panel_check.json`.
    """
    import e00730_core as core730  # ty: ignore[unresolved-import]
    import e00733_gallery as gal

    cache = world.read_weather_cache(root=root)
    meta = world.read_meta(root=root)["constants"]
    sun_up = world.sun_up_threshold(cache=cache)
    test, _ = world.read_world(root=root, kind="test")
    valid, _ = world.read_world(root=root, kind="validation")
    limits = json.loads((root / "limits.json").read_text())
    picked = [s for s in test if (s.present & s.covered).any()][:3]
    if len(picked) < 3:
        msg = (
            f"the test bank has {len(picked)} segments with a covered origin, and G10 to G13 need 3"
        )
        raise st.CorrectnessFailureError(msg)
    origins = [gal.pick_origin(seg=s) for s in picked]
    reading_points = [(np.asarray(s.reading_z) * meta["std"] + meta["mean"]) * 100.0 for s in test]
    clim = np.asarray(limits["b_clim"]["share"])
    results = [gal.g10_layout(seg=picked[0], origin_day=origins[0])]
    results.append(gal.g11_target_days(segments=picked, origin_days=origins))
    results.append(
        gal.g12_weather_correlation(
            segments=test, ens_col=world.SHORTWAVE_COLUMN, cams_col=world.SKY_COLUMN
        )
    )
    results.append(
        gal.g13_channels_against_truth(
            segments=test, shortwave_col=world.SHORTWAVE_COLUMN, sun_up=sun_up,
            min_reverse_points=float(core730.DECIDABLE_READING_POINTS), reading_points=reading_points,
        )
    )  # fmt: skip
    results.append(gal.g14_rerun_difference(segments=valid, poison_passed=poison_passed(root=root)))
    results.append(
        gal.g15_base_rate(
            segments=test, clim=clim, shortwave_col=world.SHORTWAVE_COLUMN, sun_up=sun_up
        )
    )
    digest = world.read_meta(root=root)["banks_digest"]
    payload = gal.panel_check(results=results, banks_digest=digest)
    gal.write_panel_check(root=root, payload=payload)
    for r in results:
        print(f"gallery {r.panel}: {'pass' if r.passed else 'FAIL'} {dict(r.items)}")
        gal.draw(result=r, path=root / "PRIVATE" / "gallery" / f"{r.panel}.svg")
    return payload


# --------------------------------------------------------------------------- the fits


def check_max_passes_allowed(*, args: argparse.Namespace) -> int | None:
    """Return `--max-passes`, refused unless a scratch data root was given.

    Args:
        args: The parsed arguments.

    Returns:
        The pass cap, or None.

    Raises:
        RuntimeError733: When a pass cap is asked for without `--data-root`.
    """
    cap = getattr(args, "max_passes", None)
    if cap is not None and getattr(args, "data_root", None) is None:
        msg = "--max-passes is for a scratch data root; the official run fits to the plan's caps"
        raise RuntimeError733(msg)
    return cap


def synthetic_loader(*, root: Path) -> Any:
    """Return the loader of the synthetic fits: `(arm, seed, device) -> SyntheticInputs`.

    The training banks of seeds 3 and 4, the validation bank, and the constants are read once.

    Args:
        root: This experiment's data folder (the world's cache and `limits.json`).

    Returns:
        The loader, a `fit.SyntheticLoader`.
    """
    import e00733_adapters as adapters
    import e00733_fit as fit
    import e00733_leakage as leak
    import e00733_model as model

    cached: dict[str, fit.SyntheticInputs] = {}

    def load(arm: str, seed: int, device: str) -> fit.SyntheticInputs:
        if "inputs" not in cached:
            leak.assert_synthetic_train_banks(
                seeds=[
                    int(b[1]) for b in BANK_FILES.values() if b[0] == "train" and b[1] is not None
                ],
                fit_seeds_732=list(core.SOURCE_SEEDS),
            )
            leak.assert_synthetic_inputs_clean(
                input_names=list(model.UntInputs.__dataclass_fields__)
            )
            train = train_segments(root=root)
            valid = world.read_world(root=root, kind="validation")[0]
            limits = json.loads((root / "limits.json").read_text())
            overall = float(limits["b_clim"]["overall"])
            cached["inputs"] = fit.SyntheticInputs(
                train_segments=train, train_index=sy.enumerate_samples(segments=train),
                val_segments=valid, val_index=sy.enumerate_samples(segments=valid),
                d8_index=adapters.d8_index(),
                target_std=float(world.read_meta(root=root)["constants"]["std"]),
                clim_logit=float(np.log(overall / (1.0 - overall))),
            )  # fmt: skip
        return cached["inputs"]

    return load


def make_run_job(
    *, root: Path, device: str, max_passes: int | None, fit_root: Path | None = None
) -> Any:
    """Return `run_job(arm, fold, seed) -> record`: a real-fold or a synthetic fit by the arm's place.

    Args:
        root: This experiment's data folder (inputs).
        device: "cuda" or "cpu".
        max_passes: Replaces the plan's caps (scratch debug runs only).
        fit_root: Where the record and the state are written (default `root`).

    Returns:
        The function.
    """
    import e00733_adapters as adapters
    import e00733_fit as fit

    out = root if fit_root is None else fit_root
    real = adapters.real_loader(root=root)
    synthetic = synthetic_loader(root=root)

    def run_job(arm: str, fold: str | None, seed: int) -> dict[str, Any]:
        if core.ARMS[arm].where == "real":
            if fold is None:
                msg = f"the real-fold arm {arm} needs a fold"
                raise RuntimeError733(msg)
            return fit.fit_real(
                root=out, arm=arm, fold=fold, seed=seed, device=device, loader=real,
                max_passes=max_passes,
            )  # fmt: skip
        return fit.fit_synthetic(
            root=out, arm=arm, seed=seed, device=device, loader=synthetic, max_epochs=max_passes
        )

    return run_job


def make_predicting_job(
    *, root: Path, device: str, max_passes: int | None, rows: str = "test"
) -> Any:  # fmt: skip
    """Return `run(arm, fold, seed) -> record`: the fit, then its stored prediction sets.

    Args:
        root: This experiment's data folder.
        device: "cuda" or "cpu".
        max_passes: Replaces the plan's caps (scratch debug runs only).
        rows: The rows predicted ("test" for the chain).

    Returns:
        The function. A fit that raised leaves no prediction; a prediction that raised fails the job.
    """
    import e00733_fit as fit
    import e00733_leakage as leak
    import e00733_predictions as pr

    run_job = make_run_job(root=root, device=device, max_passes=max_passes)

    def run(arm: str, fold: str | None, seed: int) -> dict[str, Any]:
        record = run_job(arm, fold, seed)
        leak.assert_limits_before_fits(
            limits=root / "limits.json",
            fit_records=[fit.record_path(root=root, arm=arm, fold=fold, seed=seed)],
        )
        pr.predict_job(root=root, arm=arm, fold=fold, seed=seed, device=device, rows=rows)
        return record

    return run


def stage_predict(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Write the stored prediction sets of one fit (`--arm`, `--fold`, `--seed`) or of every fit.

    With no `--arm`, the stage predicts every fit that has a record, in the plan's order. `--rows
    validation` predicts the held-out validation rows instead of the test rows.

    Args:
        args: The parsed arguments (`--rows`, `--device`).
        root: This experiment's data folder.

    Returns:
        The names of the sets written or found, and the number of fits predicted.

    Raises:
        RuntimeError733: When no fit exists to predict (the stage would pass on an empty counter).
    """
    import e00733_fit as fit
    import e00733_predictions as pr

    rows = getattr(args, "rows", "test")
    if args.arm is not None:
        jobs = [(args.arm, args.fold, args.seed)]
    else:
        jobs = [
            job
            for family in core.FIT_ORDER
            for job in st.family_jobs(family=family)
            if fit.record_path(root=root, arm=job[0], fold=job[1], seed=job[2]).exists()
        ]
    done: list[str] = []
    for arm, fold, seed in jobs:
        done += [
            str(p.name)
            for p in pr.predict_job(
                root=root, arm=arm, fold=fold, seed=seed, device=args.device, rows=rows
            )
        ]
    if not jobs:
        msg = "no fit exists, so the predict stage would write nothing"
        raise RuntimeError733(msg)
    return {"fits": len(jobs), "sets": done}


def stage_fit(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Run one fit (`--arm`, `--fold`, `--seed`) and return its record.

    Args:
        args: The parsed arguments.
        root: This experiment's data folder.

    Returns:
        The fit's record.

    Raises:
        RuntimeError733: When no arm was given.
    """
    if args.arm is None:
        msg = "the fit stage needs --arm (and --fold for a real-fold arm)"
        raise RuntimeError733(msg)
    cap = check_max_passes_allowed(args=args)
    run_job = make_run_job(root=root, device=args.device, max_passes=cap)
    return run_job(args.arm, args.fold, args.seed)


def keep_first_job(
    family: str, jobs: list[tuple[str, str | None, int]]
) -> list[tuple[str, str | None, int]]:
    """Return the first job of a family (a scratch debug run fits one job per family)."""
    return jobs[:1]


def build_chain_steps(
    *, root: Path, args: argparse.Namespace, run_one: Any, counter: dict[str, int]
) -> st.ChainSteps:
    """Return the chain's steps: the job runner, the two gates, and the job selection.

    The F0r gate and the screen point read the stored predictions, which the chain writes after each
    fit. A scratch run may pass `--skip-gates` to run the fits alone.

    Args:
        root: This experiment's data folder.
        args: The parsed arguments (`--skip-gates`, `--first-job-per-family`).
        run_one: `(arm, fold, seed) -> record`.
        counter: Receives the number of fits run (`run`).

    Returns:
        The steps.

    Raises:
        RuntimeError733: When `--skip-gates` or `--first-job-per-family` is given without a scratch
            data root.
    """
    scratch = getattr(args, "data_root", None) is not None
    skip = bool(getattr(args, "skip_gates", False))
    first = bool(getattr(args, "first_job_per_family", False))
    if (skip or first) and not scratch:
        msg = "--skip-gates and --first-job-per-family are for a scratch data root"
        raise RuntimeError733(msg)

    def run_job(arm: str, fold: str | None, seed: int) -> dict:
        record = run_one(arm, fold, seed)
        counter["run"] = counter.get("run", 0) + 1
        return record

    def f0r_gate() -> None:
        scoring.f0r_gate(root=root, require_all=not scratch)

    def screen_point() -> float:
        return scoring.screen_point(root=root)

    return st.ChainSteps(
        run_job=run_job,
        f0r_gate=(lambda: None) if skip else f0r_gate,
        screen_point=(lambda: 0.0) if skip else screen_point,
        select_jobs=keep_first_job if first else (lambda _f, jobs: jobs),
    )


def run_chain_stage(*, root: Path, args: argparse.Namespace, run_one: Any) -> dict[str, Any]:
    """Run the chain over `run_one` and return the status; a chain that ran no fit is a failure."""
    counter: dict[str, int] = {}
    status = st.run_chain(
        root=root, steps=build_chain_steps(root=root, args=args, run_one=run_one, counter=counter)
    )
    if status.get("state") == "complete" and counter.get("run", 0) == 0:
        msg = "the chain ran no fit, so a pass would rest on an empty counter"
        st.write_chain_status(root=root, state="failed", reason=msg, failed=[])
        raise RuntimeError733(msg)
    return status


def stage_chain(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Run the fit families in the plan's order (the dropped arms are skipped)."""
    cap = check_max_passes_allowed(args=args)
    run_one = make_predicting_job(root=root, device=args.device, max_passes=cap)
    return run_chain_stage(root=root, args=args, run_one=run_one)


def measure_timing(args: argparse.Namespace, root: Path) -> dict[str, float]:
    """Measure the timing gate: one fold of FU at seed 0 and 5 passes of YU, in a scratch folder.

    The fits are written under `scores/timing_fits`, so no chain job reads them.

    Args:
        args: The parsed arguments (`--device`, `--max-passes` on a scratch root).
        root: This experiment's data folder.

    Returns:
        The measurements that `timing_decision` reads.
    """
    cap = check_max_passes_allowed(args=args)
    sub = root / "scores" / "timing_fits"
    run_job = make_run_job(root=root, device=args.device, max_passes=cap, fit_root=sub)
    real = run_job("FU", core.SCREEN_FOLDS[0], 0)
    passes = 5 if cap is None else cap
    synthetic_cap = make_run_job(root=root, device=args.device, max_passes=passes, fit_root=sub)
    syn = synthetic_cap("YU", None, 0)
    per_pass = float(syn["seconds"]) / passes
    return {
        "seconds_per_real_fit": float(real["seconds"]),
        "seconds_per_synthetic_fit": per_pass * 80,
        "peak_gpu_gb": max(float(real["peak_gpu_gb"]), float(syn["peak_gpu_gb"])),
        "peak_rss_gb": max(float(real["peak_rss_gb"]), float(syn["peak_rss_gb"])),
    }

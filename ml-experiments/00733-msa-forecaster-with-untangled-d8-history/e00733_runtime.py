# ruff: noqa: E501
"""Experiment 733: the stage functions that read the real files (banks, channels, limits, gallery).

`e00733_cli.install_runtime` puts these functions into `STAGE_FUNCTIONS` when the command line starts,
never at import. Every function takes `(args, root)` and writes under `root` only.

A debug run on a scratch data root may pass `--limit-segments`, which keeps only the first segments
of each bank. The official run on the real data folder refuses that argument.
"""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
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
from e00733_channels import StoppedError

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
        if kind == "test":
            sy.assert_enough_test_segments_counted(
                n_segments=counts["n_segments"], n_left_out=counts["n_left_out"]
            )
            st.record_pre_fit_guard(
                root=root, name="test_segment_coverage",
                payload={k: counts[k] for k in ("n_segments", "n_left_out")},
            )  # fmt: skip
        write_origin_time_example(root=root, kind=kind, segments=segments)
        world.write_world(root=root, kind=kind, segments=segments, extra={**counts, "limit": limit})
        result[kind] = counts
        print(f"bank {kind}: {counts}")
    world.write_meta(
        root=root, payload={"constants": constants, "limit": limit, "banks_digest": digest}
    )
    return result


def write_origin_time_example(
    *, root: Path, kind: str, segments: Sequence[sy.SegmentArrays]
) -> str:
    """Check every covered origin of a bank's segments and write the first example row privately.

    The synthetic world's origin-time weather assertion runs over every target row of every covered
    origin here, and again on every batch. The example row names a base date, so it goes to
    `PRIVATE/origin_time_example_<kind>.txt` and the public log gets only the count.

    Args:
        root: This experiment's data folder.
        kind: The bank's kind.
        segments: The bank's built segments.

    Returns:
        The example row.

    Raises:
        CorrectnessFailureError: When a target token fails the assertion, or no origin is covered.
    """
    example, n_tokens = None, 0
    for seg in segments:
        days = np.flatnonzero(seg.covered)
        for k in range(sy.TOP_ROWS):
            times = sy.OriginTimes.empty(b=len(days))
            for n, d in enumerate(days):
                times.take(n=n, seg=seg, d=int(d), k=k)
            n_tokens += times.assert_origin_time_weather()
            example = example or times.example()
    if example is None:
        msg = f"the bank {kind} holds no covered target token for the origin-time weather assertion"
        raise st.CorrectnessFailureError(msg)
    (paths.private_dir(root=root) / f"origin_time_example_{kind}.txt").write_text(example + "\n")
    print(f"bank {kind}: origin-time weather asserted on {n_tokens} target tokens")
    return example


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
    guards = assert_base_day_guards(
        base_days=bank_base_days(root_730=paths.root_730()), channel_ok=channel_ok
    )
    print(f"base-day guards passed: {guards}")
    st.record_pre_fit_guard(root=root, name="base_day_guards", payload=guards)
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
        "base_day_guards": guards,
    }


HISTORY_DAYS_BEFORE_ORIGIN: Final = 15
"""A sample at origin day D reads history days D - 15 to D - 1 (the oldest row has age 14 or 15)."""
BANK_KINDS: Final = (("test", None), ("validation", None), ("shifted", None), ("train", 0), ("train", 1), ("train", 2), ("train", 3), ("train", 4))  # fmt: skip


def bank_base_days(*, root_730: Path) -> np.ndarray:
    """Return the sorted real base days (epoch days) of every bank of experiment 730 that exists.

    Args:
        root_730: Experiment 730's data folder.

    Returns:
        The unique base days.

    Raises:
        RuntimeError733: When no bank holds a base day (the guard would read nothing).
    """
    import e00729_banks as b729  # ty: ignore[unresolved-import]
    import e00730_banks as b730  # ty: ignore[unresolved-import]

    days: list[np.ndarray] = []
    for kind, seed in BANK_KINDS:
        path = b730.bank_path(root=root_730, kind=kind, seed=seed)
        if not path.exists():
            continue
        for seg in b729.load_bank(path=path).segments:
            dates = np.asarray(seg.dates).astype("datetime64[D]").astype(np.int64)
            days.append(dates[np.asarray(seg.cal_index) >= 0])
    if not days:
        msg = "no bank of experiment 730 holds a base day"
        raise RuntimeError733(msg)
    return np.unique(np.concatenate(days))


def assert_base_day_guards(*, base_days: np.ndarray, channel_ok: np.ndarray) -> dict[str, int]:
    """Check that the base days precede every fold's history and that no history or channel touches one.

    The fold layout gives each fold's first stage 2 or stage 3 origin (`TRAIN_WINDOW_START`) and last
    origin; a sample at origin day D reads the history days D - 15 to D - 1. The two guards are
    `assert_base_days_precede_history` and `assert_no_overlap` (the number of samples whose history
    overlaps a base day, and the number of untangled channel days that sit on a base day).

    Args:
        base_days: The base days of experiment 730's banks, epoch days.
        channel_ok: `(n_grid,)` whether an untangled channel exists at each half-hour of the grid.

    Returns:
        The count of base days, samples, and channel days read.

    Raises:
        CorrectnessFailureError: When a guard fails (verdict row 1).
        RuntimeError733: When no channel exists in the table.
    """
    import e00733_leakage as lk

    first = epoch_day(text=TRAIN_WINDOW_START)
    first_history = dict.fromkeys(core.ALL_FOLDS, first - HISTORY_DAYS_BEFORE_ORIGIN)
    lk.assert_base_days_precede_history(
        base_days_epoch=base_days, first_history_epoch_day_by_fold=first_history
    )
    present = np.unique(
        ch.GRID_START_EPOCH_DAY + np.flatnonzero(np.asarray(channel_ok)) // core.STEPS_PER_DAY
    )
    if present.size == 0:
        msg = "the channel table holds no day, so the overlap guard would read nothing"
        raise RuntimeError733(msg)
    origins = np.arange(first, max(fold_days(fold=f)[1] for f in core.ALL_FOLDS) + 1)
    n_samples, n_channels = lk.count_overlap(
        sample_first_history_day=origins - HISTORY_DAYS_BEFORE_ORIGIN, sample_last_history_day=origins - 1,
        base_days_epoch=base_days, channel_days_present=present,
    )  # fmt: skip
    lk.assert_no_overlap(n_samples=n_samples, n_channels=n_channels)
    return {
        "base_days": int(base_days.size),
        "samples": int(origins.size),
        "channel_days": int(present.size),
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


def run_stage_in_new_process(
    *, stage: str, arm: str, fold: str | None, seed: int, device: str, root: Path,
    data_root: str | None, max_passes: int | None, rows: str = "test",
) -> None:  # fmt: skip
    """Run `stage` (fit or predict) of one job in a new Python process and map its exit code.

    The plan keeps experiment 716's rule that the process that fits never reads the scored rows, so
    the chain runs each fit and each prediction in a process of its own. The new process inherits the
    slot's cores, memory scope, and environment, and the parent holds no data while it waits.

    Args:
        stage: "fit" or "predict".
        arm: The arm.
        fold: The fold (None on the synthetic world).
        seed: The seed.
        device: "cuda" or "cpu".
        root: This experiment's data folder (for the messages).
        data_root: The scratch data root named on the command line, or None.
        max_passes: The scratch pass cap, or None.
        rows: The rows a prediction covers.

    Raises:
        CorrectnessFailureError: When the process exits 4.
        StoppedError: When the process exits 3.
        RuntimeError733: When the process exits with any other code.
    """
    import subprocess
    import sys

    command = [
        sys.executable, str(Path(__file__).with_name("e00733_cli.py")), stage, "--arm", arm,
        "--seed", str(seed), "--device", device,
    ]  # fmt: skip
    if fold is not None:
        command += ["--fold", fold]
    if data_root is not None:
        command += ["--data-root", data_root]
    if max_passes is not None:
        command += ["--max-passes", str(max_passes)]
    if stage == "predict":
        command += ["--rows", rows]
    done = subprocess.run(command, check=False)
    if done.returncode == st.EXIT_OK:
        return
    where = f"{stage} of {arm} {fold} seed {seed} under {root.name}"
    if done.returncode == st.EXIT_INVALID:
        raise st.CorrectnessFailureError(f"a correctness failure in the {where} (exit 4)")
    if done.returncode == st.EXIT_STOPPED:
        raise StoppedError(f"the {where} stopped the line (exit 3)")
    msg = f"the {where} exited with code {done.returncode}"
    raise RuntimeError733(msg)


def job_record_path(*, root: Path, arm: str, fold: str | None, seed: int) -> Path:
    """Return the file that holds one job's measured times and memory."""
    import e00733_fit as fit

    return root / "fits" / f"{fit.job_name(arm=arm, fold=fold, seed=seed)}.job.json"


def write_job_record(
    *, root: Path, arm: str, fold: str | None, seed: int, fit_record: Mapping[str, Any],
    predict_seconds: float, predict_peak_rss_gb: float,
) -> dict[str, Any]:  # fmt: skip
    """Write one job's seconds and peak memory, and stop the line when the memory is too high.

    Args:
        root: This experiment's data folder.
        arm: The arm.
        fold: The fold, or None.
        seed: The seed.
        fit_record: The fit's record.
        predict_seconds: The wall seconds of the prediction (every set, the masked and stored
            variants, and the target-poison check).
        predict_peak_rss_gb: The prediction process's peak resident memory.

    Returns:
        The job record.

    Raises:
        StoppedError: When a process of the job peaked above `core.RSS_STOP_GB` (verdict row 0).
    """
    record = {
        "fit_seconds": float(fit_record["seconds"]), "predict_seconds": float(predict_seconds),
        "fit_peak_rss_gb": float(fit_record["peak_rss_gb"]),
        "predict_peak_rss_gb": float(predict_peak_rss_gb),
    }  # fmt: skip
    st.write_json(path=job_record_path(root=root, arm=arm, fold=fold, seed=seed), payload=record)
    peak = max(record["fit_peak_rss_gb"], record["predict_peak_rss_gb"])
    if peak > core.RSS_STOP_GB:
        msg = (
            f"the job {arm} {fold} seed {seed} peaked at {peak:.1f} GB of resident memory, above "
            f"the stop line of {core.RSS_STOP_GB} GB (verdict row 0)"
        )
        raise StoppedError(msg)
    return record


def make_predicting_job(
    *, root: Path, device: str, max_passes: int | None, rows: str = "test",
    data_root: str | None = None, new_process: bool = True,
) -> Any:  # fmt: skip
    """Return `run(arm, fold, seed) -> record`: the fit, then its stored prediction sets.

    Args:
        root: This experiment's data folder.
        device: "cuda" or "cpu".
        max_passes: Replaces the plan's caps (scratch debug runs only).
        rows: The rows predicted ("test" for the chain).
        data_root: The scratch data root of the command line, passed on to the new processes.
        new_process: Whether the fit and the prediction each run in a process of their own (the
            chain). The timing gate runs both in this process so that it can cap the passes.

    Returns:
        The function. A fit that raised leaves no prediction; a prediction that raised fails the
        job. The record it returns is the fit's, plus `predict_seconds` and `predict_peak_rss_gb`.
    """
    import e00733_fit as fit
    import e00733_leakage as leak
    import e00733_predictions as pr

    run_job = make_run_job(root=root, device=device, max_passes=max_passes)

    def run(arm: str, fold: str | None, seed: int) -> dict[str, Any]:
        if new_process:
            run_stage_in_new_process(
                stage="fit", arm=arm, fold=fold, seed=seed, device=device, root=root,
                data_root=data_root, max_passes=max_passes,
            )  # fmt: skip
            record = json.loads(
                fit.record_path(root=root, arm=arm, fold=fold, seed=seed).read_text()
            )
        else:
            record = run_job(arm, fold, seed)
        leak.assert_limits_before_fits(
            limits=root / "limits.json",
            fit_records=[fit.record_path(root=root, arm=arm, fold=fold, seed=seed)],
        )
        started = time.monotonic()
        if new_process:
            run_stage_in_new_process(
                stage="predict", arm=arm, fold=fold, seed=seed, device=device, root=root,
                data_root=data_root, max_passes=None, rows=rows,
            )  # fmt: skip
            predicted = json.loads(
                predict_record_path(root=root, arm=arm, fold=fold, seed=seed).read_text()
            )
            seconds, peak = float(predicted["seconds"]), float(predicted["peak_rss_gb"])
        else:
            pr.predict_job(root=root, arm=arm, fold=fold, seed=seed, device=device, rows=rows)
            seconds, peak = time.monotonic() - started, fit.peak_rss_gb()
        job = write_job_record(
            root=root, arm=arm, fold=fold, seed=seed, fit_record=record, predict_seconds=seconds,
            predict_peak_rss_gb=peak,
        )  # fmt: skip
        return {
            **record,
            "predict_seconds": job["predict_seconds"],
            "predict_peak_rss_gb": job["predict_peak_rss_gb"],
        }

    return run


def predict_record_path(*, root: Path, arm: str, fold: str | None, seed: int) -> Path:
    """Return the file in which the predict stage records its seconds and peak memory."""
    import e00733_fit as fit

    return root / "fits" / f"{fit.job_name(arm=arm, fold=fold, seed=seed)}.predict.json"


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
        started = time.monotonic()
        done += [
            str(p.name)
            for p in pr.predict_job(
                root=root, arm=arm, fold=fold, seed=seed, device=args.device, rows=rows
            )
        ]
        st.write_json(
            path=predict_record_path(root=root, arm=arm, fold=fold, seed=seed),
            payload={
                "seconds": round(time.monotonic() - started, 1),
                "peak_rss_gb": fit.peak_rss_gb(),
            },
        )
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
    run_one = make_predicting_job(
        root=root, device=args.device, max_passes=cap, data_root=getattr(args, "data_root", None)
    )
    return run_chain_stage(root=root, args=args, run_one=run_one)


def measure_timing(args: argparse.Namespace, root: Path) -> dict[str, float]:
    """Measure the timing gate: one fold of FU at seed 0 and 5 passes of YU, each fit and predicted.

    Each measured job is a whole chain job: the fit, then every prediction set (for FU the three
    stored variants, with the masked and stored forecasts), and the target-poison check, on the
    held-out validation rows. The jobs run in a scratch folder, `scores/timing_scratch`, that links
    the inputs, so no chain job reads them and no test row is read.

    Args:
        args: The parsed arguments (`--device`, `--max-passes` on a scratch root).
        root: This experiment's data folder.

    Returns:
        The measurements that `timing_decision` reads: seconds of a whole real job, seconds of a
        whole synthetic job at 80 passes, the peaks, and the seconds of the smoke test and of this
        measurement.
    """
    import e00733_smokerun as smokerun

    started = time.monotonic()
    cap = check_max_passes_allowed(args=args)
    scratch = smokerun.scratch_root(outer=root / "scores" / "timing_scratch", root=root)
    smokerun.link_inputs(root=root, scratch=scratch)
    real_job = make_predicting_job(
        root=scratch, device=args.device, max_passes=cap, rows="validation", new_process=False
    )
    real = real_job("FU", core.SCREEN_FOLDS[0], 0)
    passes = 5 if cap is None else cap
    syn_job = make_predicting_job(
        root=scratch, device=args.device, max_passes=passes, rows="validation", new_process=False
    )
    syn = syn_job("YU", None, 0)
    per_pass = float(syn["seconds"]) / passes
    smoke_path = root / st.SMOKE_FILE
    smoke_seconds = (
        float(st.read_json(path=smoke_path).get("seconds", 0.0)) if smoke_path.exists() else 0.0
    )
    return {
        "seconds_per_real_fit": float(real["seconds"]) + float(real["predict_seconds"]),
        "seconds_per_synthetic_fit": per_pass * 80 + float(syn["predict_seconds"]),
        "peak_gpu_gb": max(float(real["peak_gpu_gb"]), float(syn["peak_gpu_gb"])),
        "peak_rss_gb": max(
            float(real["peak_rss_gb"]), float(syn["peak_rss_gb"]), float(real["predict_peak_rss_gb"]),
            float(syn["predict_peak_rss_gb"]),
        ),
        "smoke_seconds": smoke_seconds,
        "timing_gate_seconds": time.monotonic() - started,
    }  # fmt: skip

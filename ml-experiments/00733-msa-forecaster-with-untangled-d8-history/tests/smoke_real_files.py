# ruff: noqa: E501
"""Smoke run of experiment 733's part-1 modules on the real files. Read only, CPU only, no GPU.

Run: `taskset -c 22-27 python tests/smoke_real_files.py` with `DATA_PATH_INTERNAL` set. The script
reads every real file that the part-1 modules read (experiment 732's verdict, scores, stored
real-day outputs, and networks; experiment 730's stored outputs, banks, and networks; experiment
607's ensemble summaries), writes nothing, and prints counts and shares only, never a date or a
value of a real untangled channel. A check fails when its counter is empty, so no check passes on
nothing.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np

_FOLDER = Path(__file__).parent.parent
sys.path.insert(0, str(_FOLDER))

import e00733_channels as ch  # noqa: E402
import e00733_core as core  # noqa: E402
import e00733_leakage as lk  # noqa: E402
import e00733_paths as paths  # noqa: E402
import e00733_real as rl  # noqa: E402
import e00733_rerun as rr  # noqa: E402

paths.pin_threads()
FAILURES: list[str] = []
D8_SERIES_ID = 26


def check(*, name: str, ok: bool, detail: str) -> None:
    """Print one check and record a failure."""
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def nonempty(*, name: str, n: int) -> bool:
    """Record a failure when a counter is empty, and return whether it holds anything."""
    check(name=f"{name} is not empty", ok=n > 0, detail=f"{n}")
    return n > 0


def smoke_rule_and_outputs() -> ch.ChannelChoice | None:
    """Apply the rule to experiment 732's real files and read the stored outputs of the chosen source."""
    root_732, root_730 = paths.root_732(), paths.root_730()
    reading = ch.read_reading_732(root=root_732)
    check(
        name="732 verdict and chain status read",
        ok=reading is not None and reading.chain_complete,
        detail=f"row {None if reading is None else reading.row}",
    )
    if reading is None:
        return None
    print(f"       checks read from real.json: {dict(reading.checks)}")
    check(
        name="experiment 732's H2 check is read",
        ok=isinstance(reading.checks["h2"], bool),
        detail=f"H2 {reading.checks['h2']}",
    )
    choice = ch.decide(reading=reading)
    print(
        f"       rule: source {choice.p_neg_source}, usable {choice.usable}, dropped {choice.dropped}"
    )
    out = ch.load_outputs(choice=choice, root_732=root_732, root_730=root_730)
    n = len(out.day)
    nonempty(name="stored rows of the chosen source", n=n)
    table = ch.build_table(outputs=out)
    check(
        name="grid indices unique and sorted",
        ok=bool((np.diff(table.grid_index) > 0).all()),
        detail=f"{len(table.grid_index)} rows",
    )
    for i, name in enumerate(core.CHANNEL_NAMES):
        n_on = int((table.values[:, i] > 1e-3).sum())
        if name in choice.usable:
            nonempty(name=f"rows where the usable channel {name} is above 0.001", n=n_on)
        else:
            check(name=f"dropped channel {name} is zero", ok=n_on == 0, detail=f"{n_on}")
    # The two stored sources must hold the same rows in the same order: 730's S has no keys.
    seeds = core.SOURCE_SEEDS
    s730 = [
        np.load(root_730 / "predictions" / ch.JOB_730.format(seed=s) / f"{ch.STORED_SET}.npz")["p"]
        for s in seeds
    ]
    u732 = [
        np.load(root_732 / "predictions" / ch.JOB_732.format(seed=s) / f"{ch.STORED_SET}.npz")["p"]
        for s in seeds
    ]
    check(
        name="730 and 732 hold the same number of era rows",
        ok=len(s730[0]) == len(u732[0]),
        detail=f"{len(s730[0])} and {len(u732[0])}",
    )
    corr = float(np.corrcoef(np.mean(s730, axis=0), np.mean(u732, axis=0))[0, 1])
    check(
        name="730's S and 732's U* agree row by row (row order is the same)",
        ok=corr > 0.5,
        detail=f"correlation {corr:.2f}",
    )
    shuffled = float(np.corrcoef(np.mean(s730, axis=0), np.roll(np.mean(u732, axis=0), 7))[0, 1])
    check(
        name="the alignment check can fail (a rolled copy disagrees)",
        ok=shuffled < corr - 0.2,
        detail=f"rolled correlation {shuffled:.2f}",
    )
    fallback = ch.decide(
        reading=ch.Reading732(
            row=0, solar_word="void", transfer_word="void", checks={}, chain_complete=True
        )
    )
    out_s = ch.load_outputs(choice=fallback, root_732=root_732, root_730=root_730)
    check(
        name="730's fallback reads rows keyed from 732's files",
        ok=len(out_s.day) == n and np.array_equal(out_s.day, out.day),
        detail=f"{len(out_s.day)} rows",
    )
    return choice


def smoke_banks() -> None:
    """Recompute every stored bank's digest with experiment 729's code and compare it with 730's summary."""
    n = lk.assert_bank_digests(root_730=paths.root_730())
    nonempty(name="banks checked against their digest", n=n)


def smoke_ensemble() -> None:
    """Read the ensemble summaries of two consecutive days and check the units."""
    day = (date(2025, 6, 15) - date(1970, 1, 1)).days
    prev = rr.read_ens_run(
        summary_dir=paths.ens_summary_dir(), run_epoch_day=day - 1, series_id=D8_SERIES_ID
    )
    cur = rr.read_ens_run(
        summary_dir=paths.ens_summary_dir(), run_epoch_day=day, series_id=D8_SERIES_ID
    )
    nonempty(name="ensemble valid times of the D-1 run", n=len(prev.valid_us))
    nonempty(name="ensemble valid times of the D run", n=len(cur.valid_us))
    stamps = rr.window_stamps_us(day_epoch=day - 1)
    got = rr.ens_for_window(
        previous=prev, current=cur, origin_epoch_day=day, window_stamps_us=stamps
    )
    n_found = int(np.isfinite(got["temperature"]).sum())
    nonempty(name="window stamps with an ensemble value", n=n_found)
    check(
        name="ensemble temperature is in degrees Celsius",
        ok=np.nanmin(got["temperature"]) > -30 and np.nanmax(got["temperature"]) < 45,
        detail="range plausible",
    )
    check(
        name="ensemble wind is in metres per second",
        ok=np.nanmin(got["wind"]) >= 0 and np.nanmax(got["wind"]) < 60,
        detail="range plausible",
    )
    check(
        name="ensemble precipitation is in kg m-2 s-1 (needs x3600 for mm/h)",
        ok=np.nanmin(got["precip"]) >= 0 and np.nanmax(got["precip"]) < 0.05,
        detail="range plausible",
    )
    std = rr.standardise_ens(
        temperature=got["temperature"],
        wind=got["wind"],
        precip_kg_m2_s=got["precip"],
        t_mean=10.0,
        t_sd=5.0,
        w_mean=4.0,
        w_sd=2.0,
    )
    check(
        name="standardised precipitation is finite and non-negative",
        ok=bool(np.nanmin(std["precip"]) >= 0),
        detail="ok",
    )
    from e00733_channels import GRID_START_EPOCH_DAY

    hidden = rr.window_hidden(
        origin_day=day - GRID_START_EPOCH_DAY, day=day - GRID_START_EPOCH_DAY - 1
    )
    nonempty(name="hidden columns of the window of day D - 1", n=int(hidden.sum()))
    have = np.isfinite(got["temperature"])
    check(
        name="the ensemble covers every hidden column of the window of day D - 1",
        ok=bool(have[hidden].all()),
        detail=f"{int(have[hidden].sum())} of {int(hidden.sum())} hidden columns",
    )
    check(
        name="the ensemble is read nowhere else",
        ok=not bool(have[~hidden].any()),
        detail=f"{int(have[~hidden].sum())} visible columns hold a value",
    )


def smoke_networks(choice: ch.ChannelChoice | None) -> None:
    """Load every stored network on the CPU and run it on a zero window of the right shape."""
    import torch

    torch.set_num_threads(1)
    sources = [("U*", paths.root_732()), ("S", paths.root_730())]
    for source, _root in sources:
        nets = [
            rr.load_network(
                source=source,
                seed=s,
                root_732=paths.root_732(),
                root_730=paths.root_730(),
                device="cpu",
            )
            for s in core.SOURCE_SEEDS
        ]
        c_in = nets[0].state_dict()[next(iter(nets[0].state_dict()))].shape[1]
        windows = np.zeros((2, c_in, rr.STEPS), dtype=np.float32)
        out = rr.predict_windows(nets=nets, windows=windows, source=source, n_base=c_in)
        check(
            name=f"{source} networks give (seeds, n, 3, 48)",
            ok=out.shape == (3, 2, 3, 48),
            detail=str(out.shape),
        )
        check(
            name=f"{source} outputs are finite and P(X<0) in 0 to 1",
            ok=bool(np.isfinite(out).all() and out[:, :, 0].min() >= 0 and out[:, :, 0].max() <= 1),
            detail="ok",
        )
        check(name=f"{source} first-layer input width", ok=c_in > 0, detail=f"{c_in} channels")


def _real_inputs():  # noqa: ANN202
    """Return experiment 729's context, the three neighbours' real series, and their medians (read only)."""
    import e00722_guard as guard
    import e00729_stages as stages729
    import e00732_neighbours as nb732
    import e00732_stages as st732

    ctx = stages729.build_context()
    cal = ctx.table.calendar
    neighbours = nb732.build_arrays(
        dates=cal.dates, log=guard.ReadLog(), name="neighbours_real_days"
    )
    medians = {
        k: float(v) for k, v in st732.read_json(path=paths.root_732() / st732.MEDIANS_FILE).items()
    }
    return ctx, cal, neighbours, medians


def _epoch(*, date64: np.datetime64) -> int:
    return int(date64.astype("datetime64[D]").astype(np.int64))


def smoke_real_windows() -> None:
    """Build real windows, re-run one origin, reproduce stored outputs.

    Poison the inputs after the limit, and count the coverage.
    """
    ctx, cal, neighbours, medians = _real_inputs()
    use = np.isfinite(cal.reading).any(axis=1)
    epoch_days = cal.dates.astype("datetime64[D]").astype(np.int64)
    nonempty(name="real calendar days with a reading", n=int(use.sum()))
    nets = [
        rr.load_network(
            source="U*", seed=s, root_732=paths.root_732(), root_730=paths.root_730(), device="cpu"
        )
        for s in core.SOURCE_SEEDS
    ]

    def predict(w: np.ndarray) -> np.ndarray:
        return rr.predict_windows(nets=nets, windows=w, source="U*", n_base=w.shape[1])

    def make(*, calendar, nb, first: int, last: int) -> rl.RealWindows:  # noqa: ANN001
        return rl.build_real_windows(
            cal=calendar, reading_rule_use=use, holidays=ctx.cal.holidays, stats=ctx.stats, medians=medians,
            neighbours=nb, summary_dir=paths.ens_summary_dir(), first_epoch_day=first, last_epoch_day=last,
        )  # fmt: skip

    # per-fold coverage: every day of a fold month is an origin; the base days of experiment 729 end in 2023
    # and the stored ensemble summaries start in 2024, so the base days are not the origins of these folds
    origin_days, labels = [], []
    for fold in core.ALL_FOLDS:
        first = np.datetime64(fold + "-01", "D")
        days = np.arange(first, (first.astype("datetime64[M]") + 1).astype("datetime64[D]"))
        origin_days += [int(d.astype(np.int64)) - ch.GRID_START_EPOCH_DAY for d in days]
        labels += [fold] * len(days)
    cover = rl.ens_coverage_by_split(
        origin_days=np.array(origin_days),
        splits=np.array(labels),
        summary_dir=paths.ens_summary_dir(),
    )
    for fold, counts in sorted(cover.items()):
        print(
            f"       ensemble coverage, fold {fold}: {counts['covered']} of {counts['origins']} origins"
        )
    nonempty(
        name="fold origins whose ensemble runs cover the hidden columns",
        n=sum(c["covered"] for c in cover.values()),
    )

    # one origin: the first day of the first fold whose two windows have readings and whose ensemble runs exist
    table_epoch = {int(d) for d in epoch_days[use]}
    test_days = [
        d
        for d, lab in zip(origin_days, labels, strict=True)
        if lab == core.F0R_FOLDS[0] and d + ch.GRID_START_EPOCH_DAY - 3 in table_epoch
    ]
    chosen = None
    for origin in test_days[3:]:
        o_epoch = origin + ch.GRID_START_EPOCH_DAY
        try:
            windows = make(calendar=cal, nb=neighbours, first=o_epoch - 8, last=o_epoch + 2)
            got = rr.rerun_origins(
                origin_days=[origin],
                build_window=windows.build_window,
                predict=predict,
                n_seeds=len(nets),
            )
        except rr.RerunError:
            continue
        if got.ok.any():
            chosen = (origin, got, windows)
            break
    if not nonempty(
        name="fold origins that can be re-run on real windows", n=int(chosen is not None)
    ):
        return
    origin, got, windows = chosen  # ty: ignore[not-iterable]
    check(
        name="one-origin re-run has the shape (1, 2, 48, 3)",
        ok=got.values.shape == (1, 2, 48, 3),
        detail=str(got.values.shape),
    )
    check(name="one-origin re-run is finite", ok=bool(np.isfinite(got.values).all()), detail="ok")
    nonempty(
        name="half-hours of the re-run that are visible and hold a reading", n=int(got.ok.sum())
    )
    check(
        name="hidden columns without an ensemble value",
        ok=windows.missing_ens_columns == 0,
        detail=f"{windows.missing_ens_columns}",
    )

    # poison run: NaN and 1e6 after the limit
    o_epoch = origin + ch.GRID_START_EPOCH_DAY
    n_poison = rl.poisoned_origin_run(
        make_windows=lambda c, nb: make(calendar=c, nb=nb, first=o_epoch - 8, last=o_epoch + 2),
        cal=cal, neighbours=neighbours, origin_day=origin, predict=predict, n_seeds=len(nets),
    )  # fmt: skip
    nonempty(name="poison values the re-run ignored", n=n_poison)

    # no-hiding reproduction of the stored outputs
    arrays = []
    for s in core.SOURCE_SEEDS:
        with np.load(
            paths.root_732() / "predictions" / ch.JOB_732.format(seed=s) / f"{ch.STORED_SET}.npz"
        ) as z:
            arrays.append({k: z[k] for k in z.files})
    stored = ch.StoredOutputs(
        day=arrays[0]["row_date"].astype(np.int64), hh=arrays[0]["row_hh"].astype(np.int64),
        values=np.stack([ch.mean_over_seeds(arrays=arrays, key=k) for k in ("p", "pv", "step_p")], axis=1).astype(np.float32),
    )  # fmt: skip
    have = {int(d) for d in np.unique(stored.day)}
    month = lambda d: str(np.datetime64(d, "D").astype("datetime64[M]"))  # noqa: E731
    interior = sorted(
        d for d in have if d - 1 in have and d + 1 in have and month(d - 1) == month(d + 1)
    )
    nonempty(
        name="stored era days with both neighbouring days stored in the same month", n=len(interior)
    )
    pick = interior[:: max(1, len(interior) // 12)][:12]
    wide = make(
        calendar=cal, nb=neighbours, first=int(epoch_days.min()) + 2, last=int(epoch_days.max()) - 2
    )
    out = rl.reproduce_stored(windows=wide, nets=nets, stored=stored, epoch_days=pick)
    check(
        name="outputs with nothing hidden reproduce the stored outputs",
        ok=out["rows"] > 0,
        detail=f"{out['rows']} rows, largest differences {out['max_p_neg']:.1e} (P(X<0)), {out['max_step']:.1e} (step)",
    )


def main() -> int:
    """Run every smoke check and return the exit status."""
    print("read-only smoke run of the part-1 modules on the real files (CPU only)")
    choice = smoke_rule_and_outputs()
    smoke_banks()
    smoke_ensemble()
    smoke_networks(choice)
    smoke_real_windows()
    print(f"{len(FAILURES)} failed checks: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

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
        name="a missing check is not guessed",
        ok=reading.checks["k2"] is None or isinstance(reading.checks["k2"], bool),
        detail="K2 is absent from real.json" if reading.checks["k2"] is None else "K2 present",
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


def main() -> int:
    """Run every smoke check and return the exit status."""
    print("read-only smoke run of the part-1 modules on the real files (CPU only)")
    choice = smoke_rule_and_outputs()
    smoke_banks()
    smoke_ensemble()
    smoke_networks(choice)
    print(f"{len(FAILURES)} failed checks: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

# ruff: noqa: E402
"""Experiment 730: offline calibration of the simulator ranges for the one permitted re-tune.

The script builds small synthetic training banks (a seed that no fit and no gate uses) for a list
of candidate range sets, and prints for each candidate the share of days with a daily minimum
below 1.0 MVA and the share of half-hours below 1.0 MVA, in the engine-only and the
engine-and-solar worlds, as ratios to the real shares that `scores/gate_attempt1.json` holds. It
reads the base readings that the banks are built on and prints aggregates only. It reads no
network output and no score, because no fit exists when it runs. Run it through `run.sh`:

    SLOT_CPUS=<n> run.sh --go e00730_gate_calibration.py
"""

import e00730_paths as paths

paths.pin_threads()

import json
import sys

import e00729_banks as banks729
import e00729_core as core729
import e00729_score as score729
import e00730_stages as stages
import numpy as np

CALIBRATION_SEED: int = 9
"""Training-bank seed 9 is used by no fit and no gate (the gate reads seed 0)."""
EVERY_NTH_MONTH: int = 2
ATTEMPT_FILE: str = "scores/gate_attempt1.json"
BASE_RANGES = {k: tuple(v) for k, v in core729.RANGES.items()}


def candidates() -> dict[str, dict]:
    """Return the candidate range sets: engine spells and frequency, then solar capacity."""
    out: dict[str, dict] = {"registered": {}}
    for spells_hi in (0.5, 1.0):
        for spell_days_hi in (3, 6):
            for no_run in ((0.2, 0.7), (0.05, 0.5)):
                for ramp_hi in (2.0, 4.0):
                    out[
                        f"spells<={spells_hi}, spell_days<={spell_days_hi}, "
                        f"no_run={no_run}, ramp<={ramp_hi}"
                    ] = {
                        "spells_per_30_days": (0.0, spells_hi),
                        "spell_days": (2, spell_days_hi),
                        "no_run_share": no_run,
                        "ramp_half_hours": (0.0, ramp_hi),
                    }
    return out


def candidates_two() -> dict[str, dict]:
    """Return the second grid: spell length, run frequency, and solar capacity."""
    out: dict[str, dict] = {}
    for spell_days_hi in (3, 4, 5):
        for no_run in ((0.05, 0.5), (0.1, 0.6)):
            for pv in ((5.0, 150.0), (10.0, 200.0), (20.0, 250.0)):
                out[f"spells<=1.0, spell_days<={spell_days_hi}, no_run={no_run}, pv={pv}"] = {
                    "spells_per_30_days": (0.0, 1.0),
                    "spell_days": (2, spell_days_hi),
                    "no_run_share": no_run,
                    "solar_capacity_points": pv,
                }
    return out


RE_TUNE: dict[str, tuple] = {
    "spells_per_30_days": (0.0, 1.0),
    "spell_days": (2, 3),
    "no_run_share": (0.05, 0.5),
    "solar_capacity_points": (20.0, 250.0),
}
"""The chosen range set: the four ranges that the one permitted re-tune changes."""
CANDIDATES: dict[str, dict] = (
    candidates_two() if "2" in sys.argv[1:] else {"re-tune": RE_TUNE} if "reach" in sys.argv[1:]
    else candidates()
)  # fmt: skip


def main() -> int:
    """Print the share ratios of every candidate."""
    root = paths.data_root()
    real = json.loads((root / ATTEMPT_FILE).read_text())["shares"]
    ctx = stages.real_context()
    inputs = stages.bank_inputs(ctx=ctx)
    months = sorted(inputs.months["training"])[::EVERY_NTH_MONTH]
    use = {m: inputs.use["training"][m] for m in months}
    results = []
    for name, override in CANDIDATES.items():
        core729.RANGES.clear()
        core729.RANGES.update(BASE_RANGES)
        core729.RANGES.update(override)
        bank = banks729.build_bank(
            cal=inputs.cal, kind="train", seed=CALIBRATION_SEED, months=months, month_use=use
        )
        if "reach" in sys.argv[1:]:
            test = banks729.build_bank(
                cal=inputs.cal, kind="test", seed=None, months=inputs.months["test"],
                month_use=inputs.use["test"],
            )  # fmt: skip
            rows = score729.rows_from_segments(segments=test.segments, holidays=inputs.cal.holidays)
            print("reachability counts:", score729.reachability(rows=rows))
        parts, ratios = [], []
        for era, worlds in (("net", ("engine",)), ("solar", ("both",))):
            day, hh = banks729.bank_shares_by_world(bank=bank, worlds=worlds)
            ratios += [day / real[era]["real_day"], hh / real[era]["real_half_hour"]]
            parts.append(f"{era}: day {ratios[-2]:.2f}x, half-hour {ratios[-1]:.2f}x")
        worst = max(abs(np.log(r)) for r in ratios)
        results.append((worst, f"{name}: " + "; ".join(parts)))
    for worst, line in sorted(results)[:15]:
        print(f"worst log-ratio {worst:.2f} | {line}")
    print(f"candidates tried: {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

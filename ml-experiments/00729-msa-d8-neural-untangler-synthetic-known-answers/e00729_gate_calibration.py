# ruff: noqa: E402
# ruff: noqa: E501
"""Experiment 729: offline calibration of the simulator ranges for the realism gate's share checks.

The script builds small synthetic banks (training months, a seed that no fit or gate uses) for a list
of candidate range sets and prints, for each candidate, the share of days with a daily minimum below
1.0 MVA and the share of half-hours below 1.0 MVA in the engine-only and the engine-and-solar worlds,
beside the real shares of the page and their ratio. It reads D8's base readings (to build the
synthetic worlds on them) and prints aggregates only. It reads no network output and no score; no
fit exists when it runs. Run it through `run.sh` with the argument `cpu`:

    SLOT_CPU=<n> run.sh --go e00729_gate_calibration.py cpu
"""

import e00729_paths as paths

paths.pin_threads()

import sys
from datetime import date

import e00729_banks as banks
import e00729_core as core
import e00729_data as data
import numpy as np

CALIBRATION_SEED: int = 9
"""Training-bank seed 9 (root 7299) is used by no fit and no gate."""
EVERY_NTH_MONTH: int = 2
REAL = {"net": (0.1715, 0.0119), "solar": (0.5364, 0.0454)}
"""The real shares of the net and solar eras (days under 1.0 MVA at the minimum, half-hours under it)."""
BASE_RANGES = {k: tuple(v) for k, v in core.RANGES.items()}
BASE_SPELL_LEVEL = core.SPELL_LEVEL_RANGE


def grid() -> dict[str, dict]:
    """Return the grid of candidate range sets: spells, solar capacity, and engine ramps."""
    out: dict[str, dict] = {"current": {}}
    for spells_hi in (0.5, 1.0, 2.0):
        for spell_days_hi in (3, 6, 19):
            for pv in ((5.0, 150.0), (20.0, 200.0), (30.0, 250.0), (40.0, 300.0)):
                for ramp_hi in (2.0, 4.0):
                    out[
                        f"spells<={spells_hi}, spell_days<={spell_days_hi}, pv={pv}, ramp<={ramp_hi}"
                    ] = {
                        "spells_per_30_days": (0.0, spells_hi),
                        "spell_days": (2, spell_days_hi),
                        "solar_capacity_points": pv,
                        "ramp_half_hours": (0.0, ramp_hi),
                    }
    return out


def grid_two() -> dict[str, dict]:
    """Return the second grid, around the best set of the first: engine frequency and PV size."""
    out: dict[str, dict] = {}
    for no_run in ((0.2, 0.7), (0.05, 0.5)):
        for evening_hi in (6.0, 4.0):
            for share_lo in (0.6, 0.4):
                for pv in ((40.0, 300.0), (40.0, 400.0), (60.0, 400.0)):
                    for cap in (120.0, 180.0):
                        out[
                            f"no_run={no_run}, evening<={evening_hi}, share>={share_lo}, pv={pv}, cap<={cap}"
                        ] = {
                            "spells_per_30_days": (0.0, 0.5),
                            "spell_days": (2, 6),
                            "ramp_half_hours": (0.0, 4.0),
                            "no_run_share": no_run,
                            "evening_hours": (2.0, evening_hi),
                            "unit_output_share": (share_lo, 1.0),
                            "solar_capacity_points": pv,
                            "total_points_cap": (0.0, cap),
                        }
    return out


CANDIDATES: dict[str, dict] = grid_two() if "2" in sys.argv[1:] else grid()
SORT_BY_WORST: bool = True


def shares_for(*, ctx_cal, months: list[str], month_use: dict) -> dict[str, tuple[float, float]]:  # noqa: ANN001
    """Return the share pair of the engine-only and the engine-and-solar worlds of a small bank."""
    bank = banks.build_bank(
        cal=ctx_cal, kind="train", seed=CALIBRATION_SEED, months=months, month_use=month_use
    )
    return {
        "net": banks.bank_shares_by_world(bank=bank, worlds=("engine",)),
        "solar": banks.bank_shares_by_world(bank=bank, worlds=("both",)),
    }


def main() -> int:
    """Print the shares of every candidate."""
    table = data.load_day_table()
    cal = data.select_days(cal=table.calendar, mask=table.is_base)
    months, n_days, bsp_days = data.month_table(cal=cal)
    split = core.month_splits(months=months, n_days=n_days, bsp_days=bsp_days)
    days = [d.astype("datetime64[D]").astype(date) for d in cal.dates]
    day_split = np.array(core.assign_day_splits(days=days, splits=split))
    keys = np.array([str(d)[:7] for d in cal.dates])
    training = sorted(m for m, s in split.items() if s == "training")[::EVERY_NTH_MONTH]
    use = {m: (keys == m) & (day_split == "training") for m in training}
    print(f"candidates: {len(CANDIDATES)}; training months used: {len(training)}", flush=True)
    results = []
    for name, override in CANDIDATES.items():
        core.RANGES.update({k: tuple(v) for k, v in BASE_RANGES.items()})
        core.RANGES.update(override)
        got = shares_for(ctx_cal=cal, months=training, month_use=use)
        parts, ratios = [], []
        for era, (day, hh) in got.items():
            rd, rh = REAL[era]
            ratios += [day / rd, hh / rh]
            parts.append(f"{era}: day {day / rd:.2f}x, half-hour {hh / rh:.2f}x")
        worst = max(abs(np.log(r)) for r in ratios)
        results.append((worst, f"{name}: " + "; ".join(parts)))
    for worst, line in sorted(results)[:15] if SORT_BY_WORST else results:
        print(f"worst log-ratio {worst:.2f} | {line}")
    print(f"candidates tried: {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

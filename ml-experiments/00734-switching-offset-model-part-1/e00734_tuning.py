"""Grids and file rules of experiment 734's tuning, including the extension of the H-own grid.

The first tuning ran 12 H-own settings, rho_on in (1e-4, 3e-4, 1e-3) crossed with two values of
rho_off and two of rho_cont, and found the best rho_on on the grid's lowest value. The extension
adds rho_on in (3e-5, 1e-5), crossed with the same rho_off and rho_cont values, as 8 new settings.
They are named `own12` to `own19`, so every file and done marker of the first 27 settings stays
valid. The functions here are pure (they take the output folder as an argument) so that the tests
can run them on a temporary folder.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Final

import e00734_events as ev
import e00734_hmm as hmm
import numpy as np

SETTINGS: Final[tuple[tuple[float, float, float], ...]] = tuple(
    (on, off, cont)
    for on in (1e-4, 3e-4, 1e-3)
    for off in (1 / 60, 1 / 240)
    for cont in (0.05, 0.2)
)
"""The 12 (rho_on, rho_off, rho_cont) settings of H-own in the first tuning."""
SETTINGS_EXT: Final[tuple[tuple[float, float, float], ...]] = tuple(
    (on, off, cont) for on in (3e-5, 1e-5) for off in (1 / 60, 1 / 240) for cont in (0.05, 0.2)
)
"""The 8 settings of the extension: two lower rho_on values with the same rho_off and rho_cont."""
SETTINGS_EXT2: Final[tuple[tuple[float, float, float], ...]] = (
    # one parameter at a time from the best setting of the first extension (1e-5, 1/60, 0.2)
    (3e-6, 1 / 60, 0.2),
    (1e-6, 1 / 60, 0.2),
    (1e-5, 1 / 20, 0.2),
    (1e-5, 1 / 8, 0.2),
    (1e-5, 1 / 60, 0.5),
    (1e-5, 1 / 60, 0.8),
    # the full cross of the six new values, for the interactions
    *((on, off, cont) for on in (3e-6, 1e-6) for off in (1 / 20, 1 / 8) for cont in (0.5, 0.8)),
)
"""The 14 settings of the second and last extension.

The first six move one parameter at a time away from the best setting of the first extension, so
that every parameter has three or more values with its best-so-far value in the interior. The
last eight cross the new values of rho_on, rho_off, and rho_cont with one another.
"""
ALL_SETTINGS: Final[tuple[tuple[float, float, float], ...]] = (
    SETTINGS + SETTINGS_EXT + SETTINGS_EXT2
)
CPU_HOURS_PER_SETTING: Final[float] = 0.51
"""Measured: 8 settings over the 30 tuning worlds and the real record took 4.1 CPU-hours."""
EXT2_CAP_CPU_HOURS: Final[float] = 8.0
"""The manager's cap on the second extension."""
MAX_H_OWN_EXTENSIONS: Final[int] = 2
"""The H-own grid is extended twice and never again."""
GRID_LEVELS: Final[tuple[str, ...]] = ("first", "extended", "extended2")
STATE_FILE: Final[str] = "extension_state.json"
RHO_KEYS: Final[tuple[str, ...]] = ("rho_on", "rho_off", "rho_cont")
V1_FILES: Final[tuple[str, ...]] = (
    "choice_own.json",
    "tuning_own.parquet",
    "tuning.parquet",
    "params.json",
    "params.sha256",
    "g3b.json",
)
"""The outputs of the first tuning, copied to `*_v1` before the extension overwrites them."""
V2_FILES: Final[tuple[str, ...]] = (*V1_FILES, "edge_check.json")
"""The outputs of the first extension, copied to `*_v2` before the second extension overwrites."""


def ext_names() -> list[str]:
    """The setting names of the first extension (`own12` to `own19`)."""
    return [f"own{i:02d}" for i in range(len(SETTINGS), len(SETTINGS) + len(SETTINGS_EXT))]


def ext2_names() -> list[str]:
    """The setting names of the second extension (`own20` onward)."""
    first = len(SETTINGS) + len(SETTINGS_EXT)
    return [f"own{i:02d}" for i in range(first, first + len(SETTINGS_EXT2))]


def versioned_name(*, name: str, version: str) -> str:
    """`choice_own.json` becomes `choice_own_v1.json` for version `v1`."""
    path = Path(name)
    return f"{path.stem}_{version}{path.suffix}"


def v1_name(*, name: str) -> str:
    """`choice_own.json` becomes `choice_own_v1.json`."""
    return versioned_name(name=name, version="v1")


def backup(*, out: Path, version: str, names: tuple[str, ...]) -> list[str]:
    """Copy each existing output to its `_<version>` name, never replacing an existing copy.

    Args:
        out: The experiment's output folder.
        version: `v1` or `v2`.
        names: The files to copy.

    Returns:
        The names of the files copied now.
    """
    copied = []
    for name in names:
        src, dst = out / name, out / versioned_name(name=name, version=version)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(dst.name)
    return copied


def backup_v1(*, out: Path, names: tuple[str, ...] = V1_FILES) -> list[str]:
    """Copy each existing first-tuning output to its `_v1` name, never replacing a `_v1` file."""
    return backup(out=out, version="v1", names=names)


def backup_v2(*, out: Path, names: tuple[str, ...] = V2_FILES) -> list[str]:
    """Copy each existing first-extension output to its `_v2` name, never replacing a `_v2` file."""
    return backup(out=out, version="v2", names=names)


def extension_complete(*, out: Path, keys: list[str], names: list[str] | None = None) -> bool:
    """Whether every setting of an extension has a file for every world and the real record."""
    wanted = ext_names() if names is None else names
    return all((out / "tune" / f"{name}_{key}.parquet").exists() for name in wanted for key in keys)


def grid_level(*, out: Path, keys: list[str]) -> int:
    """0 for the first grid, 1 once the first extension is complete, 2 once the second is."""
    if extension_complete(out=out, keys=keys) and extension_complete(
        out=out, keys=keys, names=ext2_names()
    ):
        return 2
    return 1 if extension_complete(out=out, keys=keys) else 0


def projected_cpu_hours(*, n_settings: int) -> float:
    """CPU-hours of `n_settings` new H-own settings over the 30 worlds and the real record."""
    return n_settings * CPU_HOURS_PER_SETTING


# extension state ------------------------------------------------------------------------------


def read_state(*, out: Path) -> dict:
    """The extension counters: `h_own` (0, 1, or 2), and the PELT, kappa, and tau extensions."""
    path = out / STATE_FILE
    state = {"h_own": 0, "pelt": None, "kappa": None, "tau": None}
    if path.exists():
        state |= json.loads(path.read_text())
    elif any((out / "tune").glob("own12_*.parquet")) if (out / "tune").exists() else False:
        state["h_own"] = 1
    return state


def _write_state(*, out: Path, state: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{STATE_FILE}.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.rename(out / STATE_FILE)


def register_h_own_extension(*, out: Path, number: int) -> None:
    """Record that the H-own grid is being extended for the `number`-th time.

    A run that resumes after a crash registers the same number again and passes. A third
    extension raises, so no re-run of `tune` can extend the grid again.

    Raises:
        ValueError: If `number` is above `MAX_H_OWN_EXTENSIONS` or skips an extension.
    """
    state = read_state(out=out)
    if number > MAX_H_OWN_EXTENSIONS:
        raise ValueError(
            f"extension {number} refused: the H-own grid is extended {MAX_H_OWN_EXTENSIONS} "
            "times at most, and edge hits after that are reported, not extended"
        )
    if number > state["h_own"] + 1:
        raise ValueError(f"extension {number} refused: extension {state['h_own'] + 1} comes first")
    if number == state["h_own"] + 1:
        if projected_cpu_hours(n_settings=len(SETTINGS_EXT2)) > EXT2_CAP_CPU_HOURS:
            raise ValueError("the second extension would pass its CPU-hour cap")
        _write_state(out=out, state=state | {"h_own": number})


def extension_values(
    *,
    grid: tuple[float, ...],
    chosen: float,
    n: int = 3,
    max_ratio: float = 2.0,
    floor: float = 0.0,
) -> tuple[str, list[float]]:
    """Up to `n` new grid values beyond the edge where `chosen` sits, continuing the grid's ratio.

    Args:
        grid: The existing grid values.
        chosen: The chosen value.
        n: Most values to add.
        max_ratio: Largest ratio between neighbouring values of the new part.
        floor: The natural lower limit of the parameter (0 for kappa, which cannot go below it).

    Returns:
        The side (`low` or `high`) and the new values, nearest to the grid first.

    Raises:
        ValueError: If `chosen` is not on an edge, or the edge is the natural limit.
    """
    values = sorted(grid)
    if chosen == values[0]:
        side, ratio = "low", values[1] / values[0] if values[0] > 0 else max_ratio
    elif chosen == values[-1]:
        side, ratio = "high", values[-1] / values[-2]
    else:
        raise ValueError(f"{chosen:g} is interior to its grid; there is nothing to extend")
    ratio = min(ratio, max_ratio)
    if side == "low":
        if values[0] <= floor:
            raise ValueError(f"{chosen:g} is the natural lower limit of the parameter")
        new = [values[0] / ratio ** (k + 1) for k in range(n)]
        return side, [v for v in new if v > floor]
    return side, [values[-1] * ratio ** (k + 1) for k in range(n)]


TAU_CANDIDATES: Final[tuple[float, ...]] = (0.4, 0.3, 0.2)
"""The lower tau values that `--extend-tau` may add below the floor of the tau grid."""


def lower_taus(
    *,
    grid: tuple[float, ...],
    chosen: float,
    within_limits: Callable[[float], bool],
    candidates: tuple[float, ...] = TAU_CANDIDATES,
) -> list[float]:
    """The tau values below the grid floor at which both called rates stay within E1-J-sym's.

    The called rates come from the stored events of the real record, which hold every event with
    its confidence, so no decoding is needed to check them.

    Args:
        grid: The tau grid.
        chosen: The chosen tau.
        within_limits: Whether both called rates at a tau are at or below E1-J-sym's.
        candidates: The values to try, at most three.

    Returns:
        The eligible new values, highest first.

    Raises:
        ValueError: If the choice is not the grid floor, or a call rate binds so that no lower tau
            is eligible.
    """
    if chosen != min(grid):
        raise ValueError(f"tau {chosen:g} is not the floor of its grid; there is nothing to extend")
    new = [t for t in candidates[:3] if t < min(grid) and within_limits(t)]
    if not new:
        raise ValueError("a called rate binds at the floor, so no lower tau is eligible")
    return new


def register_other_extension(*, out: Path, arm: str, side: str, values: list[float]) -> list[float]:
    """Record the one extension of the PELT penalty, the kappa, or the H-nbr tau grid.

    Args:
        out: The experiment's output folder.
        arm: `pelt`, `kappa`, or `tau`.
        side: The edge that is extended.
        values: The new grid values, at most three.

    Returns:
        The values of the extension (those already stored when it was registered before).

    Raises:
        ValueError: If a different extension of the same grid was registered earlier, or the
            values are more than three or would pass the CPU-hour cap.
    """
    state = read_state(out=out)
    old = state.get(arm)
    if old is not None:
        if [round(v, 9) for v in old["values"]] != [round(v, 9) for v in values]:
            raise ValueError(f"the {arm} grid was extended once already, with {old['values']}")
        return old["values"]
    if len(values) > 3 or projected_cpu_hours(n_settings=len(values)) > EXT2_CAP_CPU_HOURS:
        raise ValueError("an extension adds at most 3 values and stays under the CPU-hour cap")
    _write_state(out=out, state=state | {arm: {"side": side, "values": values}})
    return values


def nbr_tag(*, out: Path) -> str:
    """The name of the H-nbr tuning outputs, so that old outputs are reused only while valid.

    The pass-1 results of H-nbr depend on the H-own values of rho_on, rho_off, and rho_cont. The
    outputs made under the first choice (`choice_own_v1.json`) are `nbr`, those made under the
    first extension's choice (`choice_own_v2.json`) are `nbr2`, and those made under the second
    extension's choice are `nbr3`. The tag of the current choice is the first of these whose H-own
    values equal it. With no `_v1` file no extension has happened, and the tag is `nbr`.
    """
    now = out / "choice_own.json"
    v1, v2 = out / "choice_own_v1.json", out / "choice_own_v2.json"
    if not v1.exists() or not now.exists():
        return "nbr"
    new = json.loads(now.read_text())["h_own"]

    def same(path: Path) -> bool:
        old = json.loads(path.read_text())["h_own"]
        return all(abs(old[k] - new[k]) <= 1e-15 * max(1.0, abs(old[k])) for k in RHO_KEYS)

    if same(v1):
        return "nbr"
    if not v2.exists():
        return "nbr2"
    return "nbr2" if same(v2) else "nbr3"


G3B_LIMIT: Final[float] = 0.5
"""Gate G3b: calls per Primary-year on synthetic noise must stay under this."""


def g3b_rates(
    *, arms: dict[str, tuple[hmm.Priors, float]], years: float = 2.0, seed: int = 734
) -> dict[str, dict]:
    """Calls per Primary-year of each arm on synthetic Student-t noise, each read at its own tau.

    Every arm decodes the same noise series with its own priors, so the gate covers H-own at its
    tau and H-nbr at its lower tau. The series is synthetic: no scored world is read.

    Args:
        arms: For each arm, its priors and its tau.
        years: Length of the noise series in years.
        seed: Seed of the noise.

    Returns:
        For each arm, its tau, the calls per Primary-year, and whether it is under `G3B_LIMIT`.
    """
    n = int(years * 365 * hmm.BLOCKS_PER_DAY)
    x = np.random.default_rng(seed).standard_t(4, n)
    out = {}
    for arm, (priors, tau) in arms.items():
        post, o_path, l_path = hmm.decode(x=x, f=np.zeros(n), sigma=1.0, scale=np.ones(4), p=priors)
        events = ev.extract(
            site="noise",
            o_path=o_path,
            l_path=l_path,
            post=post,
            xh=np.repeat(x, ev.BLOCK),
            sigma=1.0,
            record_end=n * ev.BLOCK,
        )
        rate = len(ev.called(events=events, tau=tau)) / years
        out[arm] = {"tau": tau, "called_per_primary_year": rate, "passed": rate < G3B_LIMIT}
    return out

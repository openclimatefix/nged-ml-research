# ruff: noqa: E501, PT018, PIE808
"""Experiment 726: constants, paths, panels, slices, and checkout assertions shared by every script.

This module imports no neural-network library. Everything here is plain numpy, polars, and the
standard library, so the unit tests run it on tiny synthetic data.

**Output rule.** Every file this experiment writes lies under its own data folder
(`data_root()`), and `refuse_foreign_path` raises for any other path. The data folders of
experiments 581, 650, 667, 709, 713, 714, and 715 are read-only inputs.
"""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00581_bank as bank  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00726-msa-candidate-follow-share-on-injected-shift-bank"
NAME_667: Final = "00667-one-year-backtest-of-the-two-output-demand-network-at-primaries"
NAME_713: Final = "00713-msa-axial-attention-demand-network-part-1"
FORBIDDEN_PREFIXES: Final = ("00581-", "00650-", "00667-", "00709-", "00713-", "00714-", "00715-")
"""A path that has one of these folder names among its parts is never written."""

FOLD_MONTHS: Final = (
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
BANK_FOLDS: Final = ("2025-11", "2026-01", "2026-04", "2026-06")
"""The four fold months of experiment 709's bank."""
SEEDS_ALL: Final = (0, 1, 2, 3, 4)
SEEDS_TWO: Final = (0, 1)
N_PASSES: Final = bank.N_PASSES
PER_DAY: Final = bank.PER_DAY
DAY_US: Final = 86_400_000_000
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
KEYS: Final = ("time_series_id", "power_fcst_init_time", "valid_time")
HOLDOUT_UTC: Final = "2026-07-01"
"""No forecast has a valid time on or after this date."""
LEAD_BAND_COUNTED: Final = (24.0, 120.0)
"""Hours after the native origin, lower bound included and upper bound excluded (experiment 581)."""
LEAD_BANDS: Final = {"24-120h": (24.0, 120.0), "120-240h": (120.0, 240.0), "24-240h": (24.0, 240.0)}
MAX_ORIGIN_DAY: Final = 20
"""The last origin day (counted from 0) that any file of this experiment holds."""
MAX_LEAD_HOURS: Final = 240.0
CLIP_SD: Final = 5.0
"""Experiment 713's clip on the standardised history power."""

SIX_SIGNED_MW: Final = ("D5", "D9", "D10", "D11", "D14", "D16")
EIGHT_MVA: Final = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
APART: Final = ("D15", "D18")
GROUPS: Final = {"six_signed_mw": SIX_SIGNED_MW, "eight_mva": EIGHT_MVA, "apart": APART}

SLICE_AFTER: Final = "after0-7"
SLICE_LASTING: Final = "lasting"
SLICE_D4END: Final = "d4-end"
SLICES: Final = (SLICE_AFTER, SLICE_LASTING, SLICE_D4END)

CODE_HASH_713: Final = "e85b84e2ea04b3cb"
HEAD_713: Final = "7c29114a8"
HEAD_F4: Final = "ac3e768e4"
GPU_CAP_GB: Final = 6.0
CANDIDATE_WORLDS: Final = (
    "e581_host",
    "e581_all",
    "zero",
    "e650",
    "bank709",
    "g2",
    "kc6_one",
    "kc6_set",
)
"""The injected worlds of the Candidate module. `e581_host` (the counted world) and `e581_all` cut the
test rows to the scorer's rows; the rest forecast every row of the Primaries they read."""
MSA_WORLDS: Final = ("all", "zero", "host")


# ---------------------------------------------------------------- paths and the output rule


def base_root() -> Path:
    """Return this experiment's data folder, whatever the smoke setting."""
    return ML_EXPERIMENTS_DATA_DIR / NAME


def data_root() -> Path:
    """Return the folder that jobs write under: `smoke/` of this experiment's folder when `E00726_SMOKE` is set.

    The smoke test writes there, so phase 1 never treats a smoke file as a finished job.
    """
    return base_root() / "smoke" if os.environ.get("E00726_SMOKE") else base_root()


def folder_of(*, prefix: str) -> Path:
    """Return the data folder of the experiment whose folder name starts with `prefix` (read only)."""
    found = sorted(ML_EXPERIMENTS_DATA_DIR.glob(f"{prefix}-*"))
    if len(found) != 1:
        msg = f"expected one folder {prefix}-* under {ML_EXPERIMENTS_DATA_DIR}, found {found}"
        raise FileNotFoundError(msg)
    return found[0]


def refuse_foreign_path(path: Path) -> Path:
    """Return `path` if it lies under this experiment's data folder; raise otherwise.

    A path with the folder name of any experiment in `FORBIDDEN_PREFIXES` among its parts is
    refused as well, so a symbolic link into another experiment's folder cannot slip through.
    """
    resolved = path.resolve()
    root = data_root().resolve()
    if not resolved.is_relative_to(root):
        msg = f"refusing to write outside {root}: {resolved}"
        raise ValueError(msg)
    for part in resolved.parts:
        if part.startswith(FORBIDDEN_PREFIXES):
            msg = f"refusing to write under another experiment's folder: {resolved}"
            raise ValueError(msg)
    return path


def write_parquet_atomic(*, frame: pl.DataFrame, path: Path) -> None:
    """Write `frame` to `path` through a temporary file, after the output rule has passed."""
    refuse_foreign_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(tmp)
    tmp.rename(path)


def write_json_atomic(*, obj: object, path: Path) -> None:
    """Write `obj` as JSON to `path` through a temporary file, after the output rule has passed."""
    refuse_foreign_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.rename(path)


def touch_marker(*, path: Path) -> None:
    """Create the empty marker `path` (a job with no rows), after the output rule has passed."""
    refuse_foreign_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def candidate_output(
    *, world: str, month: str, r: int, seed: int, k: int | None = None, data: Path | None = None
) -> Path:
    """Return the path of one Candidate forecast file.

    The world is in the folder name and in the file name, and the host-only world carries the set
    number after the world name: `candidate/<world>/<month>/<world>[_k<set>]_<month>_r<pass>_d0_s<seed>.parquet`.
    """
    stem = world if k is None else f"{world}_k{k}"
    base = data if data is not None else data_root()
    return base / "candidate" / world / month / f"{stem}_{month}_r{r:02d}_d0_s{seed}.parquet"


def msa_output(
    *, world: str, arm: str, month: str, seed: int, r: int, data: Path | None = None
) -> Path:
    """Return the path of one MSA forecast file: `msa/<world>/<arm>_<month>_s<seed>/pass<NN>.parquet`."""
    base = data if data is not None else data_root()
    return base / "msa" / world / f"{arm}_{month}_s{seed}" / f"pass{r:02d}.parquet"


def job_done(*, out: Path) -> bool:
    """Return whether the job that writes `out` has finished: the file or its `.empty` marker exists."""
    return out.exists() or out.with_suffix(".empty").exists()


@dataclass(frozen=True)
class CandidateRoot:
    """The folder of one Candidate chain: the stored weights, tables, and chain of some seeds."""

    tag: str
    root: Path
    seeds: tuple[int, ...]

    def work(self, *, month: str) -> Path:
        """Return `P7_D50_<month>/work` of this root."""
        return self.root / f"P7_D50_{month}" / "work"

    def chain(self, *, month: str) -> Path:
        """Return the K1 chain's `work` folder of `month`."""
        return self.root / "chains" / month / "work"

    def model_file(self, *, month: str, seed: int) -> Path:
        """Return the saved weights of (`month`, `seed`)."""
        return self.work(month=month) / "models" / "P5" / f"{month}_d0_s{seed}.pt"

    def table_file(self, *, month: str, seed: int) -> Path:
        """Return the prediction-phase table of (`month`, `seed`)."""
        return self.root / "tables" / "predict" / f"T_{month}_s{seed}.npz"

    def stored_fit(self, *, month: str, seed: int) -> Path:
        """Return the stored forecast on the unshifted panel."""
        return self.work(month=month) / "fits" / "P5" / f"{month}_d0_s{seed}.parquet"

    def stored_inject(self, *, month: str, r: int, seed: int) -> Path:
        """Return experiment 667's stored forecast of experiment 650's events (check KC2 only)."""
        return self.work(month=month) / "inject" / f"{month}_r{r:02d}_d0_s{seed}.parquet"

    def require_seed(self, *, seed: int) -> None:
        """Raise unless `seed` belongs to this root."""
        if seed not in self.seeds:
            msg = f"seed {seed} does not belong to root {self.tag} (seeds {self.seeds})"
            raise ValueError(msg)


def candidate_root(*, tag: str) -> CandidateRoot:
    """Return the root `e667` (seeds 0 and 1) or `seeds713` (seeds 2 to 4)."""
    if tag == "e667":
        return CandidateRoot(tag=tag, root=ML_EXPERIMENTS_DATA_DIR / NAME_667, seeds=(0, 1))
    if tag == "seeds713":
        return CandidateRoot(
            tag=tag, root=ML_EXPERIMENTS_DATA_DIR / NAME_713 / "candidate_seeds", seeds=(2, 3, 4)
        )
    msg = f"unknown root {tag!r}: use e667 or seeds713"
    raise ValueError(msg)


def root_of_seed(*, seed: int) -> CandidateRoot:
    """Return the root that holds the Candidate's weights of `seed`."""
    return candidate_root(tag="e667" if seed in {0, 1} else "seeds713")


def msa_host_world(*, k: int) -> str:
    """Return the folder name of the MSA host-only world for set `k`, shared by the writer and the reader."""
    return f"host_k{k}"


def msa_fit_file(*, arm: str, month: str, seed: int) -> Path:
    """Return experiment 713's stored forecast of `arm` on the unshifted panel."""
    return ML_EXPERIMENTS_DATA_DIR / NAME_713 / "work" / "fits" / arm / f"{month}_s{seed}.parquet"


def msa_model_file(*, arm: str, month: str, seed: int) -> Path:
    """Return experiment 713's stage 3 state of `arm`."""
    return (
        ML_EXPERIMENTS_DATA_DIR / NAME_713 / "work" / "models" / arm / f"{month}_s{seed}_stage3.pt"
    )


def cap_gpu(*, device: str) -> None:
    """Cap this process's GPU memory at `GPU_CAP_GB` (or the `GPU_CAP_GB` environment variable)."""
    import torch  # ty: ignore[unresolved-import]

    if device.startswith("cuda"):
        cap = float(os.environ.get("GPU_CAP_GB", GPU_CAP_GB))
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(cap * 2**30 / total)


# ---------------------------------------------------------------- frames


def normalise_frame(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return `frame` with `time_series_id` as Int64 and both time columns as UTC microseconds."""
    utc = pl.Datetime("us", "UTC")
    exprs = [pl.col("time_series_id").cast(pl.Int64)]
    for column in ("power_fcst_init_time", "valid_time"):
        dtype = frame.schema[column]
        assert isinstance(dtype, pl.Datetime), (column, dtype)
        col = pl.col(column)
        col = (
            col.dt.replace_time_zone("UTC")
            if dtype.time_zone is None
            else col.dt.convert_time_zone("UTC")
        )
        exprs.append(col.cast(utc))
    return frame.with_columns(exprs)


def read_forecast(*, path: Path, columns: Sequence[str] = QCOLS) -> pl.DataFrame:
    """Return the keys and the named quantile columns of a forecast file, normalised."""
    return normalise_frame(frame=pl.read_parquet(path, columns=[*KEYS, *columns]))


def seed_mean(*, frames: Sequence[pl.DataFrame], columns: Sequence[str] = QCOLS) -> pl.DataFrame:
    """Return the per-key mean of the quantile columns over the seeds' frames, sorted across levels.

    A key that any frame lacks is dropped. The 11 columns are sorted after the mean, as experiment
    713's stored files are. A frame with a duplicated key raises.
    """
    n = len(frames)
    assert n > 0
    for frame in frames:
        assert not frame.select(list(KEYS)).is_duplicated().any(), (
            "a forecast frame has a repeated key"
        )
    if n == 1:
        merged = frames[0].select(*KEYS, *columns)
    else:
        merged = (
            pl.concat([f.select(*KEYS, *columns) for f in frames])
            .group_by(list(KEYS))
            .agg(*[pl.col(c).mean() for c in columns], pl.len().alias("_n"))
            .filter(pl.col("_n") == n)
            .drop("_n")
        )
    values = merged.select(list(columns)).to_numpy().astype(np.float64)
    values.sort(axis=1)
    return merged.select(*KEYS).with_columns(
        [pl.Series(c, values[:, i]) for i, c in enumerate(columns)]
    )


# ---------------------------------------------------------------- the injected panels


def month_start_us(*, month: str) -> int:
    """Return the first microsecond of `month` (UTC) as a Unix timestamp in microseconds."""
    from datetime import UTC, datetime

    return int(datetime.fromisoformat(month + "-01").replace(tzinfo=UTC).timestamp() * 1_000_000)


def all_shifted_panel(*, clean: np.ndarray, month_start_idx: int, r: int) -> np.ndarray:
    """Return experiment 581's panel of pass `r`: every Primary carries its shift."""
    shifted, _rows = bank.contaminate(clean=clean, month_start_idx=month_start_idx, r=r)
    return shifted


def host_panel(
    *, clean: np.ndarray, month_start_idx: int, r: int, members: Collection[int]
) -> np.ndarray:
    """Return the panel of pass `r` in which only the Primaries `members` carry their shifts."""
    shifted = all_shifted_panel(clean=clean, month_start_idx=month_start_idx, r=r)
    out = clean.copy()
    rows = sorted(members)
    out[rows] = shifted[rows]
    return out


def event_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (signed depth, length in days, sign) per pass and Primary, shape (24, 16)."""
    depth = np.zeros((N_PASSES, 16))
    length = np.zeros((N_PASSES, 16), dtype=np.int64)
    sign = np.zeros((N_PASSES, 16), dtype=np.int64)
    for r in range(N_PASSES):
        for p in range(16):
            depth[r, p], length[r, p], sign[r, p] = bank.event(primary=p, r=r)
    return depth, length, sign


def full_panel(*, injected: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Return an experiment 709 panel padded with the cleaned power to the length of `clean`.

    Experiment 709's panels end on 2026-06-30, so the rest of the grid holds the cleaned power
    (the rule of `e00713_era.full_panel`).
    """
    injected = np.asarray(injected, dtype=np.float64)
    assert injected.shape[0] == clean.shape[0]
    assert injected.shape[1] <= clean.shape[1]
    out = clean.astype(np.float64).copy()
    out[:, : injected.shape[1]] = injected
    return out


def check_panels(
    *,
    clean: np.ndarray,
    month_start_idx: int,
    passes: Sequence[int],
    sets: Sequence[Sequence[int]],
) -> list[str]:
    """Check KC4: return the failures of the all-shifted and host-only panels (empty when all pass)."""
    failures: list[str] = []
    end = month_start_idx + bank.END_DAY * PER_DAY
    for r in passes:
        shifted = all_shifted_panel(clean=clean, month_start_idx=month_start_idx, r=r)
        _, rows = bank.contaminate(clean=clean, month_start_idx=month_start_idx, r=r)
        for p in range(clean.shape[0]):
            signed, length, _sign = bank.event(primary=p, r=r)
            on = end - length * PER_DAY
            row = rows[p]
            if (row["onset_idx"], row["end_idx"]) != (on, end) or row["depth"] != abs(signed):
                failures.append(f"pass {r} primary {p}: event row differs from e00581_bank.event")
            inside = np.zeros(clean.shape[1], bool)
            inside[on:end] = True
            if not np.array_equal(shifted[p, ~inside], clean[p, ~inside], equal_nan=True):
                failures.append(f"pass {r} primary {p}: not the cleaned power outside the window")
            want = np.where(np.isfinite(clean[p, inside]), clean[p, inside] + signed, np.nan)
            if not np.allclose(shifted[p, inside], want, atol=1e-12, equal_nan=True):
                failures.append(
                    f"pass {r} primary {p}: not the cleaned power plus the shift inside"
                )
            if not np.array_equal(np.isnan(shifted[p]), np.isnan(clean[p])):
                failures.append(f"pass {r} primary {p}: NaN pattern differs from the cleaned power")
        for k, members in enumerate(sets):
            host = host_panel(clean=clean, month_start_idx=month_start_idx, r=r, members=members)
            for p in range(clean.shape[0]):
                same = np.array_equal(
                    host[p], shifted[p] if p in members else clean[p], equal_nan=True
                )
                if not same:
                    failures.append(f"pass {r} set {k} primary {p}: host-only panel is wrong")
    return failures


def clip_diagnostic(
    *,
    clean: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    month_start_idx: int,
    labels: Sequence[str],
    passes: Sequence[int] = tuple(range(N_PASSES)),
) -> list[dict]:
    """Return, by depth, sign, and group, the share of shifted half-hours beyond the clip and range.

    `beyond_clip` is the share of the shifted half-hours of a window whose standardised value is
    beyond plus or minus `CLIP_SD`. `out_of_range` is the share above capacity or below zero.
    """
    end = month_start_idx + bank.END_DAY * PER_DAY
    cells: dict[tuple[float, int, str], list[int]] = {}
    group_of = {label: name for name, labels_in in GROUPS.items() for label in labels_in}
    for r in passes:
        for p in range(clean.shape[0]):
            signed, length, sign = bank.event(primary=p, r=r)
            window = clean[p, end - length * PER_DAY : end]
            window = window[np.isfinite(window)] + signed
            z = (window - target_mean[p]) / target_std[p]
            key = (abs(signed), sign, group_of.get(labels[p], "other"))
            tally = cells.setdefault(key, [0, 0, 0])
            tally[0] += window.size
            tally[1] += int((np.abs(z) > CLIP_SD).sum())
            tally[2] += int(((window > 1.0) | (window < 0.0)).sum())
    return [
        {
            "depth": depth,
            "sign": sign,
            "group": group,
            "n_half_hours": n,
            "beyond_clip": beyond / n if n else float("nan"),
            "out_of_range": out / n if n else float("nan"),
        }
        for (depth, sign, group), (n, beyond, out) in sorted(cells.items())
    ]


# ---------------------------------------------------------------- the host-only sets


def assert_symmetric(*, neighbours: Mapping[int, Sequence[int]]) -> None:
    """Raise unless every neighbour list is symmetric and has no self-loop."""
    for a, others in neighbours.items():
        for b in others:
            assert a != b, f"Primary {a} lists itself as a neighbour"
            assert a in neighbours.get(b, ()), f"neighbour lists are not symmetric: {a} -> {b}"


def greedy_colouring(*, neighbours: Mapping[int, Sequence[int]]) -> list[list[int]]:
    """Return sets of Primaries in which no member is a neighbour of another member.

    The Primaries are taken by descending degree (ties by index), and each joins the lowest set
    that holds none of its neighbours.
    """
    assert_symmetric(neighbours=neighbours)
    order = sorted(neighbours, key=lambda p: (-len(set(neighbours[p])), p))
    colour: dict[int, int] = {}
    for p in order:
        used = {colour[q] for q in neighbours[p] if q in colour}
        colour[p] = next(c for c in range(len(neighbours) + 1) if c not in used)
    n_sets = max(colour.values()) + 1
    return [sorted(p for p, c in colour.items() if c == k) for k in range(n_sets)]


def check_sets(*, sets: Sequence[Sequence[int]], neighbours: Mapping[int, Sequence[int]]) -> None:
    """Raise unless each Primary is in exactly one set and no set holds two neighbours."""
    flat = [p for members in sets for p in members]
    assert sorted(flat) == sorted(neighbours), "a Primary is in no set or in two sets"
    for members in sets:
        for p in members:
            clash = set(members) & set(neighbours[p])
            assert not clash, f"set {list(members)} holds neighbours {p} and {sorted(clash)}"


def neighbour_lists() -> tuple[dict[int, list[int]], list[str], np.ndarray]:
    """Return (neighbour lists, D labels, time series ids) per Primary index, as the Candidate reads them."""
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00650_common as c650  # ty: ignore[unresolved-import]

    nb_of, _capacity, _is_mw = c650.world()
    sites = k578.sites().sort("primary")
    return nb_of, sites["label"].to_list(), sites["time_series_id"].to_numpy()


def sets_file() -> Path:
    """Return the file that holds the host-only sets, written by the preflight."""
    return base_root() / "candidate" / "sets.json"


def read_sets() -> list[list[int]]:
    """Return the host-only sets that the preflight wrote."""
    path = sets_file()
    assert path.exists(), "run `e00726_checks.py kp` first: it writes the host-only sets"
    return json.loads(path.read_text())["sets"]


# ---------------------------------------------------------------- test rows, origins, slices


def cut_mask(
    *,
    origin_idx: np.ndarray,
    target_idx: np.ndarray,
    primary: np.ndarray,
    month_start_idx: int,
    members: Collection[int] | None,
    max_origin_day: int = MAX_ORIGIN_DAY,
    max_lead_hours: float = MAX_LEAD_HOURS,
) -> np.ndarray:
    """Return the test rows the scorer reads: origin day 0 to `max_origin_day`, lead up to `max_lead_hours`.

    With `members`, only the rows of those Primaries are kept. The mask is applied to the subset
    that is passed to `test_block`, never to the test rows that `FoldSignals` checks.
    """
    day = (origin_idx - month_start_idx) // PER_DAY
    lead_hours = (target_idx - origin_idx) / 2.0
    keep = (day >= 0) & (day <= max_origin_day) & (lead_hours <= max_lead_hours)
    if members is not None:
        keep &= np.isin(primary, sorted(members))
    return keep


def origin_day_of(*, origin_us: np.ndarray, month: str) -> np.ndarray:
    """Return the origin day, counted from 0 at the month start (floor of the elapsed days)."""
    return (np.asarray(origin_us, dtype=np.int64) - month_start_us(month=month)) // DAY_US


def event_columns(
    *,
    frame: pl.DataFrame,
    month: str,
    r: int,
    primary_of_id: Mapping[int, int],
) -> pl.DataFrame:
    """Return `frame` with the event of each row's Primary and the slice flags.

    Added columns: `primary`, `depth` (unsigned), `sign`, `length_days`, `lead_hours`, `origin_day`,
    `d_on` and `d_end` (days from the onset and from the end to the native origin), and one Boolean
    column per slice. A row of a Primary outside `primary_of_id` is dropped.
    """
    depth, length, sign = event_arrays()
    ids = np.array(sorted(primary_of_id), dtype=np.int64)
    prims = np.array([primary_of_id[i] for i in ids], dtype=np.int64)
    lookup = pl.DataFrame({"time_series_id": ids, "primary": prims})
    out = frame.join(lookup, on="time_series_id", how="inner")
    p = out["primary"].to_numpy()
    origin_us = out["power_fcst_init_time"].dt.epoch("us").to_numpy()
    valid_us = out["valid_time"].dt.epoch("us").to_numpy()
    since_month = (origin_us - month_start_us(month=month)) / DAY_US
    d_end = since_month - bank.END_DAY
    d_on = d_end + length[r, p]
    after = (d_end >= 0) & (d_end < 7)
    lasting = (d_on >= 7) & (d_on < 28) & (d_end < 0)
    d4_end = (d_on >= 4) & (d_end < 0)
    return out.with_columns(
        depth=pl.Series(np.abs(depth[r, p])),
        sign=pl.Series(sign[r, p]),
        length_days=pl.Series(length[r, p]),
        lead_hours=pl.Series((valid_us - origin_us) / 3_600_000_000),
        origin_day=pl.Series(np.floor(since_month).astype(np.int64)),
        d_on=pl.Series(d_on),
        d_end=pl.Series(d_end),
        **{
            SLICE_AFTER: pl.Series(after),
            SLICE_LASTING: pl.Series(lasting),
            SLICE_D4END: pl.Series(d4_end),
        },
    )


def follow_share(
    *, shifted: np.ndarray, base: np.ndarray, sign: np.ndarray, depth: np.ndarray
) -> np.ndarray:
    """Return experiment 581's `follow_q50` per row: (shifted - base) times the sign over the depth."""
    return (shifted - base) * sign / depth


# ---------------------------------------------------------------- checkouts and hashes


def require(condition: object, message: str) -> None:
    """Raise `RuntimeError` unless `condition` holds; unlike `assert`, `PYTHONOPTIMIZE` cannot remove it."""
    if not condition:
        raise RuntimeError(message)


def peak_memory() -> dict[str, float]:
    """Return this process's peak host memory in GB and, when torch is loaded, its peak GPU memory in GB."""
    import resource

    out = {"host_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        out["gpu_allocated_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        out["gpu_reserved_peak_gb"] = torch.cuda.max_memory_reserved() / 2**30
    return out


def git_state(*, folder: Path) -> tuple[str, bool]:
    """Return (HEAD sha, whether `folder` has uncommitted changes) of the checkout holding `folder`."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=folder,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return head, bool(dirty)


def module_folder(*, name: str) -> Path:
    """Return the folder of the module `name` as the import system resolves it, without importing it."""
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None, f"module {name} is not on the path"
    return Path(spec.origin).resolve().parent


def resolved_recipe_hashes() -> dict[str, str]:
    """Return the sha256 (16 hex digits) of every loaded `e?????_*` module, as the import system resolved it."""
    out = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and re.fullmatch(r"e\d{5}_\w+", name):
            out[f"{name}:{Path(path).parent.name[:5]}"] = hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest()[:16]
    return out


def assert_frozen_713() -> dict:
    """Assert experiment 713's code hash, HEAD, and clean checkout; return the record for the log.

    The check imports `e00713_arms` from wherever `EXTRA_PYTHONPATH` puts it.
    """
    import e00713_arms as arms  # ty: ignore[unresolved-import]

    folder = Path(arms.__file__).resolve().parent
    require(arms.code_hash() == CODE_HASH_713, f"code hash {arms.code_hash()}, not {CODE_HASH_713}")
    head, dirty = git_state(folder=folder)
    require(head.startswith(HEAD_713), f"experiment 713's checkout is at {head}, not {HEAD_713}")
    require(not dirty, f"experiment 713's checkout {folder} is not clean")
    return {"code_hash": arms.code_hash(), "head": head, "dirty": dirty, "folder": str(folder)}


def assert_pinned_f4() -> dict:
    """Assert that `e00715_era` resolves from a clean checkout at the pinned F4 commit; return the record."""
    folder = module_folder(name="e00715_era")
    head, dirty = git_state(folder=folder)
    require(head.startswith(HEAD_F4), f"F4's checkout is at {head}, not {HEAD_F4}")
    require(not dirty, f"F4's checkout {folder} is not clean")
    return {"head": head, "dirty": dirty, "folder": str(folder)}


def job_record(*, extra: Mapping[str, object] | None = None) -> dict:
    """Return the record that every job logs: commits, hashes of the resolved recipe modules, and `extra`."""
    here = Path(__file__).resolve().parent
    head, dirty = git_state(folder=here)
    return {
        "commit_726": head,
        "dirty_726": dirty,
        "resolved_hashes": resolved_recipe_hashes(),
        "peaks": peak_memory(),
        **(extra or {}),
    }


# ---------------------------------------------------------------- the stub-frame redirect


def install_stub_redirect(*, e: object) -> None:
    """Make the pipeline's stand-in demand frames read from experiment 667 and write only here.

    `e00667_fold.install` writes a one-row stand-in frame into experiment 667's `inputs/frames`
    for each run that has none. This wrapper runs first: a frame that exists is returned as it is,
    and a missing one is written under this experiment's folder, so `install` never writes into
    experiment 667's folder.
    """
    import e00667_extend as ext  # ty: ignore[unresolved-import]

    stub_dir = data_root() / "inputs" / "frames"
    real_path = e.common.demand_frame_path  # ty: ignore[unresolved-attribute]

    def demand_frame_path(*, run: object) -> Path:
        path = real_path(run=run)
        if path.exists():
            return path
        stub_667 = ext.OUT_ROOT / "inputs" / "frames" / f"init={run:%Y-%m-%d}.parquet"  # ty: ignore[invalid-argument-type]
        if stub_667.exists():
            return stub_667
        stub = stub_dir / stub_667.name
        if not stub.exists():
            summary = ext.E607 / "ens_summary" / f"init={run:%Y-%m-%d}.parquet"  # ty: ignore[invalid-argument-type]
            origin = (
                pl.scan_parquet(summary).select(pl.col("power_fcst_init_time").unique()).collect()
            )
            assert origin.height == 1, summary
            write_parquet_atomic(frame=origin.with_columns(nwp_init_time=pl.lit(run)), path=stub)
        return stub

    e.common.demand_frame_path = demand_frame_path  # ty: ignore[unresolved-attribute]


def prime_pipeline(*, month: str) -> Any:
    """Import experiment 574's module for `month` in the prediction phase, with the stub redirect.

    The environment variables `E00667_PHASE=predict` (from `run.sh`) and `E00667_MONTH` must be
    set before any module that reads them is imported, so this function sets the month and
    asserts the phase.
    """
    require(os.environ.get("E00667_PHASE") == "predict", "set E00667_PHASE=predict (run.sh does)")
    require(os.environ.get("E00667_MONTH", month) == month, "E00667_MONTH names another month")
    os.environ["E00667_MONTH"] = month
    os.environ["E00667_ROOT"] = str(data_root())
    sys.argv = ["e00574_fit.py", "fit", "--arm", "P5", "--months", month, "--seeds", "0"]
    import e00574_fit as e  # ty: ignore[unresolved-import]

    install_stub_redirect(e=e)
    return e


# ---------------------------------------------------------------- MSA lead days and blind checks


def lead_indices(
    *, lo_hours: float, hi_hours: float, offset_steps: int = 6, steps_per_day: int = 48
) -> list[int]:
    """Return the lead indices of experiment 713's layout whose rows reach into [`lo_hours`, `hi_hours`).

    Lead index `k` holds the 48 half-hours that start `offset_steps` half-hours (the contract
    origin) plus `k` days after the native origin. The band is derived, never hard-coded.
    """
    first = offset_steps * 0.5
    day_hours = steps_per_day * 0.5
    out = []
    for k in range(0, 15):
        start = first + day_hours * k
        end = start + day_hours
        if start < hi_hours and end > lo_hours:
            out.append(k)
    return out


def blind_report(*, name: str, passed: bool, values: Mapping[str, object]) -> bool:
    """Print only PASS or FAIL for `name` and store the values in a file that nobody reads before phase 1.

    Checks KM2, KM3, and KC6 measure how far a forecast of the treatment or the reference moves, so
    their values must not reach a log that a person reads before the counted run.
    """
    write_json_atomic(
        obj={"name": name, "passed": passed, "values": dict(values)},
        path=data_root() / "checks" / f"blind_{name}.json",
    )
    print(f"{name} {'PASS' if passed else 'FAIL'}", flush=True)
    return passed

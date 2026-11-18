# ruff: noqa: E501
"""Experiment 706, stage 2: build the tables, fit the learned arms, and write their predictions.

Run it from the repository root with the environment of `CLAUDE.md` set, on a machine where
`uv sync --group gpu` has been run (use `uv run --no-sync`), and the folder of experiment 650 on
`PYTHONPATH` for `--prepare` (it holds the generator of the injected shifts):

    export PYTHONPATH=$(ls -d ml-experiments/00650-*)
    uv run --no-sync python ml-experiments/00706-*/e00706_s2_fit.py --prepare --n-jobs 4
    uv run --no-sync python ml-experiments/00706-*/e00706_s2_fit.py --smoke --device cuda
    uv run --no-sync python ml-experiments/00706-*/e00706_s2_fit.py --n-jobs 2 --device cuda
    uv run --no-sync python ml-experiments/00706-*/e00706_s2_fit.py --assemble --n-jobs 4

Steps:

- `--prepare` builds the tables of `e00706_s2_data` (CPU, one process, tens of minutes). A table
  that
  exists is kept; `--only` limits it to `graphs`, `real`, `ramp`, or one bank month.
- `--smoke` runs one fold (2026-03), one seed, a few epochs, for each arm on the real graph, writes
  to `s2/smoke/`, and prints the time and the peak memory of each fit and a projection of the full
  run. Use it to time one fit before the full run.
- The default mode runs the 240 fits (two arms, two graphs, six folds from 2026-01 to 2026-06, ten
  seeds), `--n-jobs` at a time, each in its own process. Every fit writes its own job file
  `s2/jobs/<arm>-<graph>-<yyyymm>-s<seed>.npz` and its state file `s2/models/<...>.pt` through
  temporary files, and a job whose file exists is skipped, so a crashed worker is rerun alone by
  running the same command again. `--fold`, `--seed`, `--arm`, and `--graph` restrict the jobs.
- `--assemble` averages the seeds' corrections, writes the prediction sets `n-rule`, `n-flag-real`,
  `n-flag-rewired`, `n-gate-real`, and `n-gate-rewired` with `write_predictions` (origins 2026-01-01
  to 2026-06-30), and stops with the list of missing jobs if a job file is absent.

Training. A fold month M trains on the real rows, the bank rows of months before M, and own-only
ramp rows, whose valid times are before the start of M minus 7 days. Rows whose valid-time week
number is 3 modulo 4 are the validation rows (so validation is grouped by valid time). Half the
steps are ramp steps (a batch of own-only ramp rows, truth is the shifted power); the other half
are mixed steps (half real rows, half bank rows). Bank rows are scored against clean power, real
rows against the observed power. Adam, learning rate 2e-3, batches of 4096 rows, 150 steps per
epoch, at most 40 epochs, early stopping on the mean validation loss with patience 6, and the
weights of the best epoch are kept. The validation loss averages the real, bank, and ramp rows
with equal weight.

Validation is week mod 4 inside the training period, and not the last training month: the first
fold (2026-01) has only about 24 days of training rows, so holding out a month would leave it none.
The validation weeks lie between training weeks, so the validation loss is a little optimistic and
early stopping may run later than a held-out month would allow. The test month and the 7-day
embargo before it are never used for training or validation.
"""

import argparse
import dataclasses
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import e00706_common as c  # ty: ignore[unresolved-import]
import e00706_s2_data as d  # ty: ignore[unresolved-import]
import numpy as np
from studies.eval_protocol.metrics import MEDIAN_INDEX
from studies.experiment_store import write_predictions

if TYPE_CHECKING:
    import torch  # ty: ignore[unresolved-import]
    from e00706_s2_models import Batch  # ty: ignore[unresolved-import]

JOBS_DIR: Final[Path] = d.S2_DIR / "jobs"
MODELS_DIR: Final[Path] = d.S2_DIR / "models"
SMOKE_DIR: Final[Path] = d.S2_DIR / "smoke"
ARMS: Final[tuple[str, ...]] = ("n-flag", "n-gate")
N_SEEDS: Final[int] = 10
BATCH_SIZE: Final[int] = 4096
STEPS_PER_EPOCH: Final[int] = 150
MAX_EPOCHS: Final[int] = 40
PATIENCE: Final[int] = 6
LEARNING_RATE: Final[float] = 2e-3
VALIDATION_ROWS: Final[int] = 40_000
VALIDATION_MODULO: Final[int] = 4
VALIDATION_REMAINDER: Final[int] = 3
SMOKE_MONTH: Final[str] = "2026-03"
SMOKE_EPOCHS: Final[int] = 2
SMOKE_STEPS: Final[int] = 40
SHIFT_FLOOR_POINTS: Final[float] = 3.0
"""Rows with a shift of at least this many points (3% of capacity) are the rows of the shift-in-force tables."""
CHUNK: Final[int] = 262_144
VALIDATION_NOTE: Final[str] = (
    "VALIDATION: week mod 4 inside the training period (not the last training month, which the "
    "first fold lacks), so early stopping is slightly optimistic; the test month is never used."
)
SET_NAMES: Final[dict[tuple[str, str], str]] = {
    ("n-flag", "real"): "n-flag-real",
    ("n-flag", "rewired"): "n-flag-rewired",
    ("n-gate", "real"): "n-gate-real",
    ("n-gate", "rewired"): "n-gate-rewired",
}


@dataclasses.dataclass(frozen=True, slots=True)
class Job:
    """One fit: an arm, a graph, a fold month, and a seed."""

    arm: str
    graph: str
    month: str
    seed: int

    @property
    def key(self) -> str:
        """The job's file stem."""
        return f"{self.arm}-{self.graph}-{self.month.replace('-', '')}-s{self.seed:02d}"


def all_jobs(
    *,
    folds: tuple[str, ...],
    seeds: tuple[int, ...],
    arms: tuple[str, ...],
    graphs: tuple[str, ...],
) -> list[Job]:
    """Return the jobs, fold by fold and seed by seed."""
    return [
        Job(arm=arm, graph=graph, month=month, seed=seed)
        for month in folds
        for seed in seeds
        for arm in arms
        for graph in graphs
    ]


# --------------------------------------------------------------------------- tensors for one fit


def _device_table(
    *, table: dict[str, np.ndarray], graph: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Return an entry table's tensors on `device`, with the arrays of one graph selected."""
    import torch  # ty: ignore[unresolved-import]

    flag = np.concatenate([table["flag_own"], table["flag_nb"][graph]], axis=1)
    out = {
        "gate": table["gate"],
        "nb_gate": table["nb_gate"][graph],
        "box": table["box"],
        "slow": table["slow"],
        "flag": flag,
        "flag_y": table["flag_y"].astype(np.int64),
    }
    if "ramp_s0" in table:
        out["ramp_s0"] = table["ramp_s0"].astype(np.float32)
        out["ramp_sign"] = table["ramp_sign"].astype(np.float32)
    return {name: torch.as_tensor(values, device=device) for name, values in out.items()}


class Rows:
    """A set of rows on a device: their entries, positions, and (for the bank) their stored
    forecasts."""

    def __init__(
        self,
        *,
        table: dict[str, torch.Tensor],
        dense: dict[str, torch.Tensor],
        entry: np.ndarray,
        k: np.ndarray,
        o: np.ndarray,
        h: np.ndarray,
        device: torch.device,
        q: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        """Move the row index arrays to `device`; `q` and `y` are given for bank rows only."""
        import torch  # ty: ignore[unresolved-import]

        self.table = table
        self.dense = dense
        self.device = device
        self.size = len(entry)
        self.entry = torch.as_tensor(entry.astype(np.int64), device=device)
        self.k = torch.as_tensor(k.astype(np.int64), device=device)
        self.o = torch.as_tensor(o.astype(np.int64), device=device)
        self.h = torch.as_tensor(h.astype(np.int64), device=device)
        self.q = None if q is None else torch.as_tensor(q.astype(np.float32), device=device)
        self.y = None if y is None else torch.as_tensor(y.astype(np.float32), device=device)

    def batch(self, *, index: torch.Tensor) -> Batch:
        """Return the rows `index` (a long tensor on the device) as a `Batch`."""
        import e00706_s2_models as m  # ty: ignore[unresolved-import]
        import torch  # ty: ignore[unresolved-import]

        entry, k, o, h = self.entry[index], self.k[index], self.o[index], self.h[index]
        q = self.dense["q"][k, o, h] if self.q is None else self.q[index]
        if self.y is None:
            y = self.dense["truth"][k, o, h]
            if "ramp_s0" in self.table:
                stamp = (c.ORIGIN_STAMP + c.PER_DAY * o + h).float()
                rise = torch.clamp(
                    (stamp - self.table["ramp_s0"][entry]) / d.RAMP_HALF_HOURS, 0.0, 1.0
                )
                y = y + self.table["ramp_sign"][entry] * d.RAMP_POINTS * rise
        else:
            y = self.y[index]
        return m.Batch(
            entry=entry,
            band=self.dense["band"][h],
            h=h,
            q=q,
            y=y,
            x=torch.nan_to_num(self.dense["weather"][k, o, h], nan=0.0),
            flag_y=self.table["flag_y"][entry],
        )


def _load_dense(*, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Return the dense tensors on `device` and the dense arrays in memory."""
    import torch  # ty: ignore[unresolved-import]

    arrays = d.load_arrays(path=d.CACHE_DIR / "real_dense.npz")
    dense = {
        "q": torch.as_tensor(arrays["q_nra"], device=device),
        "truth": torch.as_tensor(arrays["truth"], device=device),
        "weather": torch.as_tensor(arrays["weather"], device=device),
        "band": torch.as_tensor(c.BAND_OF_H.astype(np.int64), device=device),
    }
    return dense, arrays


def _split(*, stamp: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the training and validation positions among `ok`, grouped by the week of the valid
    time."""
    week = stamp // (7 * c.PER_DAY)
    validation = week % VALIDATION_MODULO == VALIDATION_REMAINDER
    return np.flatnonzero(ok & ~validation), np.flatnonzero(ok & validation)


def _implicit_rows(
    *,
    arrays: dict[str, np.ndarray],
    table: dict[str, np.ndarray],
    entries: np.ndarray,
    limit: int,
    first_origin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (entry, h) of every row of `entries` that may train: finite, before `limit`,
    covered.

    A row is eligible when its target power and base median are finite, its valid time is before
    `limit`, its row is at least 1, its origin is at least `first_origin`, and its entry is covered.
    """
    k = table["series"][entries].astype(np.int64)
    o = table["origin"][entries].astype(np.int64)
    h = np.arange(c.N_H)[None, :]
    stamp = c.ORIGIN_STAMP + c.PER_DAY * o[:, None] + h
    ok = (
        np.isfinite(arrays["truth"][k, o])
        & np.isfinite(arrays["q_nra"][k, o, :, MEDIAN_INDEX])
        & (stamp < limit)
        & (h >= 1)
        & (o[:, None] >= first_origin)
        & table["covered"][entries][:, None]
    )
    entry_index, row = np.nonzero(ok)
    return entries[entry_index], row


type Source = tuple[Rows, np.ndarray, np.ndarray]
"""A row set with the positions of its training rows and of its validation rows."""


class FitData:
    """Everything one fit reads: the training and validation rows of the three sources and the test
    rows."""

    def __init__(self, *, month: str, graph: str, device: torch.device) -> None:
        """Load the caches for fold `month` and move them to `device`."""
        self.month = month
        graph_index = d.GRAPH_NAMES.index(graph)
        limit = d.embargo_stamp(month=month)
        self.dense, arrays = _load_dense(device=device)
        self.arrays = arrays
        real_table_np = d.load_arrays(path=d.CACHE_DIR / "real_entries.npz")
        real_table = _device_table(table=real_table_np, graph=graph_index, device=device)
        entries = np.arange(len(real_table_np["origin"]))
        entry, row = _implicit_rows(
            arrays=arrays,
            table=real_table_np,
            entries=entries,
            limit=limit,
            first_origin=c.FIRST_ORIGIN_DEC,
        )
        self.real = self._source(
            table=real_table,
            table_np=real_table_np,
            entry=entry,
            row=row,
            limit=limit,
            device=device,
        )
        ramp_np = d.load_arrays(path=d.CACHE_DIR / f"ramp_train_{month}_entries.npz")
        ramp_table = _device_table(table=ramp_np, graph=graph_index, device=device)
        entry, row = _implicit_rows(
            arrays=arrays,
            table=ramp_np,
            entries=np.arange(len(ramp_np["origin"])),
            limit=limit,
            first_origin=c.FIRST_ORIGIN_DEC,
        )
        self.ramp = self._source(
            table=ramp_table, table_np=ramp_np, entry=entry, row=row, limit=limit, device=device
        )
        self.bank = self._bank_source(
            month=month, graph_index=graph_index, limit=limit, device=device
        )
        self.graph_index = graph_index
        self.device = device

    def _source(
        self,
        *,
        table: dict[str, torch.Tensor],
        table_np: dict[str, np.ndarray],
        entry: np.ndarray,
        row: np.ndarray,
        limit: int,
        device: torch.device,
    ) -> Source:
        """Return an implicit row set with its training and validation positions."""
        k = table_np["series"][entry].astype(np.int64)
        o = table_np["origin"][entry].astype(np.int64)
        stamp = c.ORIGIN_STAMP + c.PER_DAY * o + row
        train, validation = _split(stamp=stamp, ok=stamp < limit)
        rows = Rows(table=table, dense=self.dense, entry=entry, k=k, o=o, h=row, device=device)
        return rows, train, validation

    def _bank_source(
        self,
        *,
        month: str,
        graph_index: int,
        limit: int,
        device: torch.device,
    ) -> Source:
        """Return the bank rows of the months before `month` with their training and validation
        positions."""
        months = [m for m in d.BANK_MONTHS if m < month]
        tables, rows, offset = [], [], 0
        for bank_month in months:
            table = d.load_arrays(path=d.CACHE_DIR / f"bank_{bank_month}_entries.npz")
            part = d.load_arrays(path=d.CACHE_DIR / f"bank_{bank_month}_rows.npz")
            part["entry"] = part["entry"] + offset
            offset += len(table["origin"])
            tables.append(table)
            rows.append(part)
        table_np = d.concat_tables(tables=tables)
        part = d.concat_tables(tables=rows)
        ok = (
            (part["stamp"] < limit)
            & table_np["covered"][part["entry"]]
            & (part["h"] >= 1)
            & np.isfinite(part["q"][:, MEDIAN_INDEX])
            & np.isfinite(part["y"])
        )
        train, validation = _split(stamp=part["stamp"].astype(np.int64), ok=ok)
        table = _device_table(table=table_np, graph=graph_index, device=device)
        row_set = Rows(
            table=table,
            dense=self.dense,
            entry=part["entry"],
            k=part["k"],
            o=part["o"],
            h=part["h"],
            device=device,
            q=part["q"],
            y=part["y"],
        )
        return row_set, train, validation


# --------------------------------------------------------------------------- fitting


def _evaluate(
    *,
    arm: str,
    network: torch.nn.Module,
    sources: dict[str, Source],
    validation_index: dict[str, torch.Tensor],
    levels: torch.Tensor,
) -> float:
    """Return the mean validation loss over the three sources, each with equal weight."""
    import e00706_s2_models as m  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    losses = []
    network.eval()
    with torch.no_grad():
        for name, (rows, _, _) in sources.items():
            index = validation_index[name]
            if len(index) == 0:
                continue
            batch = rows.batch(index=index)
            loss, _ = m.training_loss(
                arm=arm, network=network, table=rows.table, batch=batch, levels=levels
            )
            losses.append(float(loss))
    network.train()
    return float(np.mean(losses))


def train_network(
    *, job: Job, data: FitData, epochs: int, steps: int, device: torch.device
) -> tuple[torch.nn.Module, list[float]]:
    """Train one network and return it with the best weights loaded, and its validation curve."""
    import e00706_s2_models as m  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    fold = d.FOLD_MONTHS.index(job.month)
    arm_index = ARMS.index(job.arm)
    graph_index = d.GRAPH_NAMES.index(job.graph)
    rng = np.random.default_rng([c.EXPERIMENT_ID, 4, fold, job.seed, arm_index, graph_index])
    torch.manual_seed(int(rng.integers(2**31 - 1)))
    network = m.new_network(arm=job.arm).to(device)
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    levels = m.levels_tensor(device=device)
    sources: dict[str, Source] = {"real": data.real, "bank": data.bank, "ramp": data.ramp}
    validation_index: dict[str, torch.Tensor] = {}
    for name, (_, train, validation) in sources.items():
        if len(train) == 0 and name != "ramp":
            msg = f"{job.key}: no training rows in the {name} source"
            raise ValueError(msg)
        chosen = np.sort(rng.permutation(validation)[:VALIDATION_ROWS])
        validation_index[name] = torch.as_tensor(chosen.astype(np.int64), device=device)

    def draw(*, name: str, count: int) -> tuple[Rows, Batch]:
        rows, train, _ = sources[name]
        position = train[rng.integers(len(train), size=count)]
        return rows, rows.batch(index=torch.as_tensor(position.astype(np.int64), device=device))

    best, best_state, stale, curve = np.inf, None, 0, []
    for _epoch in range(epochs):
        for _ in range(steps):
            parts: list[tuple[str, int]]
            if len(sources["ramp"][1]) > 0 and rng.random() < 0.5:
                parts = [("ramp", BATCH_SIZE)]
            else:
                parts = [("real", BATCH_SIZE // 2), ("bank", BATCH_SIZE // 2)]
            loss: Any = 0.0
            for name, count in parts:
                if len(sources[name][1]) == 0:
                    continue
                rows, batch = draw(name=name, count=count)
                part, _ = m.training_loss(
                    arm=job.arm, network=network, table=rows.table, batch=batch, levels=levels
                )
                loss = loss + part / len(parts)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        score = _evaluate(
            arm=job.arm,
            network=network,
            sources=sources,
            validation_index=validation_index,
            levels=levels,
        )
        curve.append(score)
        if score < best:
            best, stale = score, 0
            best_state = {
                name: value.detach().clone() for name, value in network.state_dict().items()
            }
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    return network, curve


def predict_rows(
    *,
    job: Job,
    network: torch.nn.Module,
    rows: Rows,
    zero_neighbours: bool,
    shift_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the correction of every row of `rows` as float32 and the mean gate over `shift_mask`
    rows.

    Args:
        job: The job, for the arm.
        network: The fitted network.
        rows: The rows.
        zero_neighbours: Zero the neighbour inputs.
        shift_mask: Rows over which the mean gate is taken; None for all rows.

    Returns:
        The corrections (N,) and the mean of the 16 gates (NaN for arm N-flag or an empty mask).
    """
    import e00706_s2_models as m  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    out = np.empty(rows.size, dtype=np.float32)
    gate_sum = np.zeros(m.N_GATE_INPUTS)
    count = 0
    network.eval()
    with torch.no_grad():
        for start in range(0, rows.size, CHUNK):
            index = torch.arange(start, min(start + CHUNK, rows.size), device=rows.device)
            batch = rows.batch(index=index)
            result = m.apply_network(
                arm=job.arm,
                network=network,
                table=rows.table,
                batch=batch,
                zero_neighbours=zero_neighbours,
            )
            out[start : start + len(index)] = result.correction.cpu().numpy()
            if result.gates is not None:
                chosen = (
                    np.ones(len(index), bool)
                    if shift_mask is None
                    else shift_mask[start : start + len(index)]
                )
                if chosen.any():
                    gates = result.gates.cpu().numpy()[chosen]
                    gate_sum += gates.sum(axis=0)
                    count += int(chosen.sum())
    mean_gate = gate_sum / count if count else np.full(m.N_GATE_INPUTS, np.nan)
    return out, mean_gate


def _entry_probabilities(*, network: torch.nn.Module, table: dict[str, torch.Tensor]) -> np.ndarray:
    """Return arm N-flag's switching probability of every entry of `table`."""
    import torch  # ty: ignore[unresolved-import]

    with torch.no_grad():
        return torch.sigmoid(network(flags=table["flag"])).cpu().numpy()


def _synchronise(*, device: torch.device) -> None:
    """Wait for the queued work on a CUDA device, so that a timer reads the finished work."""
    import torch  # ty: ignore[unresolved-import]

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_job(  # noqa: PLR0915
    *, job: Job, device_name: str, epochs: int, steps: int, out_dir: Path, models_dir: Path
) -> dict[str, float]:
    """Fit one job and write its job file and its state file; return its time and memory."""
    import torch  # ty: ignore[unresolved-import]

    started = time.time()
    device = torch.device(device_name)
    data = FitData(month=job.month, graph=job.graph, device=device)
    _synchronise(device=device)
    load_seconds = time.time() - started
    train_started = time.time()
    network, curve = train_network(job=job, data=data, epochs=epochs, steps=steps, device=device)
    _synchronise(device=device)
    train_seconds = time.time() - train_started
    month = job.month
    graph_index = d.GRAPH_NAMES.index(job.graph)
    arrays = data.arrays
    origins = c.origin_of_month(month=month)
    n_series = arrays["q_nra"].shape[0]
    real_np = d.load_arrays(path=d.CACHE_DIR / "real_entries.npz")
    real_table = _device_table(table=real_np, graph=graph_index, device=device)
    k, o, h = (
        a.reshape(-1)
        for a in np.meshgrid(np.arange(n_series), origins, np.arange(c.N_H), indexing="ij")
    )
    real_rows = Rows(
        table=real_table, dense=data.dense, entry=k * c.N_ORIGINS + o, k=k, o=o, h=h, device=device
    )
    bank_table_np = d.load_arrays(path=d.CACHE_DIR / f"bank_{month}_entries.npz")
    bank_part = d.load_arrays(path=d.CACHE_DIR / f"bank_{month}_rows.npz")
    bank_rows = Rows(
        table=_device_table(table=bank_table_np, graph=graph_index, device=device),
        dense=data.dense,
        entry=bank_part["entry"],
        k=bank_part["k"],
        o=bank_part["o"],
        h=bank_part["h"],
        device=device,
        q=bank_part["q"],
        y=bank_part["y"],
    )
    ramp_np = d.load_arrays(path=d.CACHE_DIR / f"ramp_test_{month}_entries.npz")
    n_ramp = len(ramp_np["origin"])
    ramp_entry = np.repeat(np.arange(n_ramp), c.N_H)
    ramp_h = np.tile(np.arange(c.N_H), n_ramp)
    ramp_rows = Rows(
        table=_device_table(table=ramp_np, graph=graph_index, device=device),
        dense=data.dense,
        entry=ramp_entry,
        k=ramp_np["series"][ramp_entry],
        o=ramp_np["origin"][ramp_entry],
        h=ramp_h,
        device=device,
    )
    stamp = c.ORIGIN_STAMP + c.PER_DAY * ramp_np["origin"][ramp_entry].astype(np.int64) + ramp_h
    ramp_value = d.ramp_shift(
        stamp=stamp, start=ramp_np["ramp_s0"][ramp_entry], sign=ramp_np["ramp_sign"][ramp_entry]
    )
    result: dict[str, np.ndarray] = {
        "val_curve": np.array(curve),
        "best_epoch": np.array(int(np.argmin(curve))),
    }
    bank_in_force = np.abs(bank_part["shift"]) >= SHIFT_FLOOR_POINTS
    ramp_in_force = np.abs(ramp_value) >= SHIFT_FLOOR_POINTS
    for suffix, zero in (("", False), ("_z", True)):
        corr, gate = predict_rows(job=job, network=network, rows=real_rows, zero_neighbours=zero)
        result[f"corr_real{suffix}"] = corr.reshape(n_series, len(origins), c.N_H)
        result[f"gate_real{suffix}"] = gate
        corr, gate = predict_rows(
            job=job, network=network, rows=bank_rows, zero_neighbours=zero, shift_mask=bank_in_force
        )
        result[f"corr_bank{suffix}"] = corr
        result[f"gate_bank{suffix}"] = gate
        corr, gate = predict_rows(
            job=job, network=network, rows=ramp_rows, zero_neighbours=zero, shift_mask=ramp_in_force
        )
        result[f"corr_ramp{suffix}"] = corr.reshape(n_ramp, c.N_H)
        result[f"gate_ramp{suffix}"] = gate
    if job.arm == "n-flag":
        result["p_real"] = _entry_probabilities(network=network, table=real_table)[
            (np.arange(n_series)[:, None] * c.N_ORIGINS + origins[None, :]).reshape(-1)
        ].reshape(n_series, len(origins))
        result["p_bank"] = _entry_probabilities(network=network, table=bank_rows.table)
        result["p_ramp"] = _entry_probabilities(network=network, table=ramp_rows.table)
    total_seconds = time.time() - started
    result["seconds"] = np.array(total_seconds)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    state_path = models_dir / f"{job.key}.pt"
    temporary = state_path.with_suffix(".tmp")
    torch.save(network.state_dict(), temporary)
    temporary.replace(state_path)
    d.save_arrays(path=out_dir / f"{job.key}.npz", arrays=result)
    peak_cuda = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    return {
        "load_seconds": load_seconds,
        "train_seconds": train_seconds,
        "predict_seconds": total_seconds - load_seconds - train_seconds,
        "total_seconds": total_seconds,
        "epochs": float(len(curve)),
        "peak_host_gb": c.peak_memory_gb(),
        "peak_cuda_gb": peak_cuda,
    }


def _run_job_safely(
    args: tuple[Job, str, int, int, Path, Path],
) -> tuple[str, str | dict[str, float]]:
    """Run a job in a worker process; return its key and its statistics or its error text."""
    job, device_name, epochs, steps, out_dir, models_dir = args
    try:
        return job.key, run_job(
            job=job,
            device_name=device_name,
            epochs=epochs,
            steps=steps,
            out_dir=out_dir,
            models_dir=models_dir,
        )
    except Exception as error:  # noqa: BLE001
        return job.key, f"{type(error).__name__}: {error}"


def run_jobs(
    *,
    jobs: list[Job],
    n_jobs: int,
    device_name: str,
    epochs: int,
    steps: int,
    out_dir: Path,
    models_dir: Path,
) -> None:
    """Run the jobs whose files do not exist, `n_jobs` at a time, and print each job's time and
    memory."""
    todo = [job for job in jobs if not (out_dir / f"{job.key}.npz").exists()]
    print(f"JOBS {len(jobs)} requested, {len(todo)} to run, {n_jobs} at a time on {device_name}")
    print(VALIDATION_NOTE)
    arguments = [(job, device_name, epochs, steps, out_dir, models_dir) for job in todo]
    failed = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=context) as pool:
        for key, outcome in pool.map(_run_job_safely, arguments):
            if isinstance(outcome, str):
                failed.append(key)
                print(f"FAILED {key}: {outcome}", flush=True)
            else:
                print(
                    f"DONE {key}: {outcome['total_seconds']:.0f} s ({outcome['epochs']:.0f} epochs), "
                    f"host peak {outcome['peak_host_gb']:.2f} GB, GPU peak {outcome['peak_cuda_gb']:.2f} GB",
                    flush=True,
                )
    print(f"FINISHED, {len(failed)} failed: {failed}")


def smoke(*, device_name: str) -> None:
    """Fit one fold, one seed, a few epochs, per arm on the real graph; print the timings and a
    projection.

    The projection charges every one of the 240 jobs with its own load of `FitData` (measured
    separately from the training), a full-length training (`MAX_EPOCHS` epochs of
    `STEPS_PER_EPOCH` steps, at the per-step time of the smoke fit, validation passes included, so
    it ignores early stopping and is an upper bound), and its prediction time.
    """
    jobs = all_jobs(folds=(SMOKE_MONTH,), seeds=(0,), arms=ARMS, graphs=("real",))
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    print(VALIDATION_NOTE)
    total_seconds = 0.0
    jobs_per_arm = len(d.GRAPH_NAMES) * len(d.FOLD_MONTHS) * N_SEEDS
    for job in jobs:
        stats = run_job(
            job=job,
            device_name=device_name,
            epochs=SMOKE_EPOCHS,
            steps=SMOKE_STEPS,
            out_dir=SMOKE_DIR / "jobs",
            models_dir=SMOKE_DIR / "models",
        )
        per_step = stats["train_seconds"] / (SMOKE_EPOCHS * SMOKE_STEPS)
        full_train = per_step * MAX_EPOCHS * STEPS_PER_EPOCH
        per_job = stats["load_seconds"] + full_train + stats["predict_seconds"]
        total_seconds += jobs_per_arm * per_job
        print(
            f"SMOKE {job.key}: FitData load {stats['load_seconds']:.1f} s; train_network alone "
            f"{stats['train_seconds']:.1f} s for {SMOKE_EPOCHS * SMOKE_STEPS} steps "
            f"({1000 * per_step:.0f} ms a step); predicting {stats['predict_seconds']:.1f} s; host peak "
            f"{stats['peak_host_gb']:.2f} GB, GPU peak {stats['peak_cuda_gb']:.2f} GB; one job of "
            f"{MAX_EPOCHS * STEPS_PER_EPOCH} steps would take at most {per_job / 60:.1f} min "
            f"({stats['load_seconds']:.0f} s of loading, {full_train / 60:.1f} min of training)"
        )
    print(
        f"SMOKE PROJECTION: {jobs_per_arm * len(ARMS)} jobs, each loading its own FitData, take at most "
        f"{total_seconds / 3600:.1f} h run one at a time; divide by the number of jobs run at once "
        "only to the extent that the device is not already saturated"
    )


# --------------------------------------------------------------------------- assembling


def assemble(*, n_jobs: int) -> None:
    """Average the seeds, write the five prediction sets, and stop if any job file is missing."""
    jobs = all_jobs(
        folds=d.FOLD_MONTHS, seeds=tuple(range(N_SEEDS)), arms=ARMS, graphs=d.GRAPH_NAMES
    )
    missing = [job.key for job in jobs if not (JOBS_DIR / f"{job.key}.npz").exists()]
    if missing:
        msg = f"{len(missing)} job files are missing, such as {missing[:5]}; run the fits first"
        raise FileNotFoundError(msg)
    inputs = c.load_inputs(load_prevailing=False, n_jobs=n_jobs)
    real_np = d.load_arrays(path=d.CACHE_DIR / "real_entries.npz")
    n_series = len(inputs.ids)
    corrections: dict[str, np.ndarray] = {}
    weight = real_np["w_rule"].reshape(n_series, c.N_ORIGINS)
    box = real_np["box"].reshape(n_series, c.N_ORIGINS, c.N_BANDS)
    corrections["n-rule"] = (weight[..., None] * box[:, :, c.BAND_OF_H]).astype(np.float32)
    for (arm, graph), name in SET_NAMES.items():
        dense = np.zeros((n_series, c.N_ORIGINS, c.N_H), dtype=np.float32)
        for month in d.FOLD_MONTHS:
            origins = c.origin_of_month(month=month)
            seeds = [
                d.load_arrays(
                    path=JOBS_DIR / f"{Job(arm=arm, graph=graph, month=month, seed=seed).key}.npz"
                )["corr_real"]
                for seed in range(N_SEEDS)
            ]
            dense[:, origins, :] = np.mean(seeds, axis=0)
        corrections[name] = dense
    origins = np.arange(c.FIRST_ORIGIN_JAN, c.N_ORIGINS)
    for name, correction in corrections.items():
        destination = d.S2_DIR.parent / "predictions" / name
        if destination.exists():
            print(f"  {name}: exists, skipped")
            continue
        quantiles = c.apply_correction(inputs=inputs, quantiles=inputs.q_nra, correction=correction)
        frame = c.dense_to_frame(inputs=inputs, quantiles=quantiles, origins=origins)
        shift = float(np.abs(correction[:, origins, :]).mean())
        print(f"  {name}: {frame.height:,} rows, mean |correction| {shift:.3f} points")
        path = write_predictions(
            experiment_id=c.EXPERIMENT_ID,
            name=name,
            frame=frame,
            kind="quantiles",
            row_set_id=c.ROW_SET_ID_FROM_JANUARY,
        )
        print(f"  {name}: wrote {path}")


def main() -> None:
    """Parse the command line and run one of the modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="build the cached tables")
    parser.add_argument(
        "--only", default=None, help="with --prepare: graphs, real, ramp, or a bank month"
    )
    parser.add_argument("--smoke", action="store_true", help="time one fold, one seed, few epochs")
    parser.add_argument("--assemble", action="store_true", help="write the prediction sets")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="processes (threads for --prepare and --assemble loading)",
    )
    parser.add_argument("--device", default="cuda", help="torch device, cuda or cpu")
    parser.add_argument("--fold", default=None, help="one fold month, YYYY-MM")
    parser.add_argument("--seed", type=int, default=None, help="one seed, 0 to 9")
    parser.add_argument("--arm", default=None, choices=ARMS)
    parser.add_argument("--graph", default=None, choices=d.GRAPH_NAMES)
    arguments = parser.parse_args()
    if arguments.prepare:
        inputs = c.load_inputs(load_prevailing=False, n_jobs=arguments.n_jobs)
        d.prepare_all(inputs=inputs, only=arguments.only)
        return
    if arguments.assemble:
        assemble(n_jobs=arguments.n_jobs)
        return
    if arguments.smoke:
        smoke(device_name=arguments.device)
        return
    jobs = all_jobs(
        folds=d.FOLD_MONTHS if arguments.fold is None else (arguments.fold,),
        seeds=tuple(range(N_SEEDS)) if arguments.seed is None else (arguments.seed,),
        arms=ARMS if arguments.arm is None else (arguments.arm,),
        graphs=d.GRAPH_NAMES if arguments.graph is None else (arguments.graph,),
    )
    run_jobs(
        jobs=jobs,
        n_jobs=arguments.n_jobs,
        device_name=arguments.device,
        epochs=MAX_EPOCHS,
        steps=STEPS_PER_EPOCH,
        out_dir=JOBS_DIR,
        models_dir=MODELS_DIR,
    )


if __name__ == "__main__":
    main()

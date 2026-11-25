# ruff: noqa: E501, PT018, D205
"""Experiment 715: the neighbour tensors of a batch, the masks, the neighbour dropout, and the
per-batch assertions.

`F4Source` wraps experiment 713's `Source`. It calls `Source.assemble` for the host matrix, then builds
the neighbour tensors from an explicit array `neighbour_power`, set by the caller and never inferred.
Experiment 713's stage 3 replaces `power_in` for all 16 Primaries (`e00713_train.synthetic_source`),
so
without an explicit array the neighbours would read the same shifted array as the host. Every source
therefore carries a `label` that names the array the neighbour slots read, and `ARRAYS_OF_MODE` states
which array each of the five modes of design section 5.2 reads.

The neighbour token at slot `s`, age `a`, and column `j` has the same grid index as the host's history
token at age `a` and column `j` (arms `F4`, `F4-R`), or the grid index of the shuffled age (arm `F4-T`).
It is visible only if its grid index is at most the native origin's grid index minus 48 (the host's own
function `e00713_layout.visible_by_time`), the reading is finite (a flagged half-hour is NaN), and the
D15 and D18 cut before 2025-03-01 does not remove it. A slot removed by neighbour dropout is masked whole.
"""

from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00715_model as f4
import e00715_neighbours as nb
import numpy as np
import torch  # ty: ignore[unresolved-import]

Mode = Literal[
    "stage2",
    "stage3_main",
    "stage3_mirror",
    "predict_real",
    "predict_injected",
    "predict_injected_rewired",
]
ArrayName = Literal["cleaned", "shifted", "mirror", "injected"]

NEIGHBOUR_DROPOUT: Final = 0.15
"""Each filled slot is masked whole with this probability per sample in training."""
SHORT_HISTORY_CUT: Final = date(2025, 3, 1)
GRID_START_DATE: Final = date(2019, 9, 1)
SHORT_HISTORY_CUT_IDX: Final = (SHORT_HISTORY_CUT - GRID_START_DATE).days * lay.STEPS_PER_DAY
"""Grid index of the first half-hour of 2025-03-01: D15 and D18 are not read before it."""
ARRAYS_OF_MODE: Final[dict[Mode, tuple[ArrayName, ArrayName]]] = {
    "stage2": ("cleaned", "cleaned"),
    "stage3_main": ("shifted", "cleaned"),
    "stage3_mirror": ("mirror", "mirror"),
    "predict_real": ("cleaned", "cleaned"),
    "predict_injected": ("injected", "injected"),
    "predict_injected_rewired": ("injected", "cleaned"),
}
"""For each mode of design section 5.2, the array the host's history reads and the array the neighbour slots read.

`predict_injected_rewired` is the mode of the rewired arms (`F4-R`, `F4-WR`) on an injected world: a rewired
arm has no real neighbour, so every rewired slot reads the cleaned array and no slot shows an injected
change (design section 9 rule 4, as amended).
"""
DROP_KEY: Final = 4_150_715


@dataclass
class F4Assembled:
    """One assembled batch: the host batch of experiment 713 and the neighbour tensors."""

    host: bt.Assembled
    nbr: f4.NeighbourInputs
    nbr_idx: torch.Tensor  # (B, 14, 48) grid index of each neighbour token
    slot_primary: torch.Tensor  # (B, 6) the Primary in each slot, -1 where empty


def check_origin_days(*, ens: bt.EnsTensors, c_idx: torch.Tensor | np.ndarray, what: str) -> None:
    """Raise a readable error if a contract origin lies outside the days the ensemble table holds.

    The table `ens.run_us` (and every other tensor of `EnsTensors`) holds origin days
    `first_day` to `first_day + n_days - 1`. A gather outside that range is a device-side assert on
    the GPU and an unreadable crash (experiment 713's J1 gate); this check stops it on the CPU.
    """
    c = np.asarray(c_idx.cpu() if isinstance(c_idx, torch.Tensor) else c_idx)
    day = c // lay.STEPS_PER_DAY - ens.first_day
    n_days = ens.filled.shape[1]
    if ens.run_us is not None:
        assert ens.run_us.shape[1] == n_days, (ens.run_us.shape, n_days)
    if day.size and (int(day.min()) < 0 or int(day.max()) >= n_days):
        msg = (
            f"{what}: origin day index {int(day.min())} to {int(day.max())} lies outside the "
            f"ensemble table, which holds day indices 0 to {n_days - 1} "
            f"(day numbers {ens.first_day} to {ens.first_day + n_days - 1})"
        )
        raise IndexError(msg)


class F4Source:
    """A host `Source`, an explicit neighbour power array, and the slot tables of one arm."""

    def __init__(
        self, *, host: bt.Source, neighbour_power: torch.Tensor, mw_scale: np.ndarray,
        table: nb.NeighbourTable, mode: nb.SlotMode, rewire_seed: int, label: ArrayName,
    ) -> None:  # fmt: skip
        """Hold the arrays and build the per-Primary slot tables for `mode`.

        Args:
            host: Experiment 713's source; its `power_in` is the array the host's history reads.
            neighbour_power: (16, G) standardised power that the neighbour slots read, each Primary
                with its own `target_mean` and `target_std`; NaN where missing or flagged.
            mw_scale: (16,) `target_std * capacity_mw`, the megawatts of one standard deviation.
            table: The neighbour table.
            mode: `real`, `rewired`, `shuffled`, or `none`.
            rewire_seed: The seed of the rewired draw (the seed of the neural network).
            label: The name of the array that `neighbour_power` is (asserted by the mode table).
        """
        self.host = host
        self.neighbour_power = neighbour_power
        self.label = label
        self.mode = mode
        self.rewire_seed = rewire_seed
        self.table = table
        dev = host.device
        self.device = dev
        slot, exists, rel = nb.slot_arrays(
            table=table, hosts=np.arange(nb.N_PRIMARIES), mode=mode, seed=rewire_seed
        )
        self.slot_table = torch.from_numpy(slot).to(dev)
        self.rel_table = torch.from_numpy(rel).to(dev)
        self.exists_table = torch.from_numpy(exists).to(dev)
        scale = np.asarray(mw_scale, dtype=np.float64)
        self._mw_scale = scale
        self.ratio = torch.from_numpy((scale[None, :] / scale[:, None]).astype(np.float32)).to(dev)
        """`ratio[h, q]` = MW per standard deviation of `q` over that of host `h`."""
        cut = np.zeros(nb.N_PRIMARIES, dtype=np.int64)
        for label_ in nb.PAIR:
            cut[nb.LABELS.index(label_)] = SHORT_HISTORY_CUT_IDX
        self.cut_idx = torch.from_numpy(cut).to(dev)

    def with_arrays(
        self, *, host_power_in: torch.Tensor | None, neighbour_power: torch.Tensor, label: ArrayName
    ) -> F4Source:
        """Return a source with another host history array and another neighbour array."""
        host = (
            self.host if host_power_in is None else self.host.with_power_in(power_in=host_power_in)
        )
        return F4Source(
            host=host, neighbour_power=neighbour_power, mw_scale=self.scale_np(), table=self.table,
            mode=self.mode, rewire_seed=self.rewire_seed, label=label,
        )  # fmt: skip

    def with_mode(self, *, mode: nb.SlotMode, rewire_seed: int | None = None) -> F4Source:
        """Return a source with the same arrays and another slot mode."""
        return F4Source(
            host=self.host, neighbour_power=self.neighbour_power, mw_scale=self.scale_np(),
            table=self.table, mode=mode,
            rewire_seed=self.rewire_seed if rewire_seed is None else rewire_seed, label=self.label,
        )  # fmt: skip

    def scale_np(self) -> np.ndarray:
        """Return `mw_scale`, the megawatts of one standard deviation of each Primary."""
        return self._mw_scale

    def assemble(
        self,
        *,
        primary: torch.Tensor,
        c_idx: torch.Tensor,
        lead_day: torch.Tensor,
        ensemble: bool,
        ages: torch.Tensor | None = None,
        age_exists: torch.Tensor | None = None,
        mask_all_power: torch.Tensor | None = None,
        mask_newest_row: torch.Tensor | None = None,
        target_end: int | None = None,
        cap_steps: int = lay.POWER_CAP_STEPS,
        nbr_ages: torch.Tensor | None = None,
        nbr_age_exists: torch.Tensor | None = None,
        drop_slots: torch.Tensor | None = None,
        mask_all_slots: bool = False,
        neighbour_power: torch.Tensor | None = None,
        host_power_in: torch.Tensor | None = None,
        ens_day_offset: int = 0,
    ) -> F4Assembled:
        """Return the batch of samples (Primary, contract origin `c_idx`, lead day).

        The host arguments are those of `Source.assemble`, which experiment 713's source documents.

        Args:
            primary: (B,) long, the host Primary of each sample.
            c_idx: (B,) long, the contract origin's grid index.
            lead_day: (B,) long, the zero-based lead day.
            ensemble: Whether the target row reads the ensemble weather.
            ages: (B, 14) long, the host's shuffled ages (experiment 713's placebo); None for 2 to 15.
            age_exists: (B, 14) bool, False for a shuffled row with no block.
            mask_all_power: (B,) bool, mask every history power token (experiment 713's block dropout).
            mask_newest_row: (B,) bool, mask the newest history row's power.
            target_end: Grid index at and after which targets are NaN, or None.
            cap_steps: The number of half-hours of the newest row that are hidden.
            nbr_ages: (B, 14) long, the ages at which the slots read their neighbour (arm `F4-T`);
                None for ages 2 to 15.
            nbr_age_exists: (B, 14) bool, False for a shuffled row with no block.
            drop_slots: (B, 6) bool, neighbour dropout: True masks the slot whole.
            mask_all_slots: Mask every slot (the read-out of design section 7.4).
            neighbour_power: Override of the array the slots read (check KN2).
            host_power_in: Override of the array the host's history reads (check KN2).
            ens_day_offset: Shift, in days, of the ensemble run read (experiment 713's canary).

        Returns:
            The host batch, the neighbour tensors, the slot index, and the slot's Primary.
        """
        if ensemble and self.host.ens is not None:
            check_origin_days(ens=self.host.ens, c_idx=c_idx, what="assemble")
        host = self.host.assemble(
            primary=primary, c_idx=c_idx, lead_day=lead_day, ensemble=ensemble, ages=ages,
            age_exists=age_exists, mask_all_power=mask_all_power, mask_newest_row=mask_newest_row,
            target_end=target_end, cap_steps=cap_steps, power_in=host_power_in,
            ens_day_offset=ens_day_offset,
        )  # fmt: skip
        power = self.neighbour_power if neighbour_power is None else neighbour_power
        b = len(primary)
        steps = self.host.steps
        ages_n = (
            self.host.default_ages[None, :].expand(b, lay.N_HISTORY_ROWS)
            if nbr_ages is None
            else nbr_ages
        )
        nidx = c_idx[:, None, None] - lay.STEPS_PER_DAY * ages_n[:, :, None] + steps  # (B, 14, 48)
        assert int(nidx.min()) >= 0 and int(nidx.max()) < power.shape[1], (
            "neighbour rows leave the grid"
        )
        slot = self.slot_table[primary]  # (B, 6), -1 where empty
        exists = slot >= 0
        q = slot.clamp_min(0)
        raw = power[q[:, :, None, None], nidx[:, None]]  # (B, 6, 14, 48)
        newest = c_idx - lay.NATIVE_OFFSET_STEPS - cap_steps
        by_time = nidx <= newest[:, None, None]
        cut_ok = nidx[:, None] >= self.cut_idx[q][:, :, None, None]
        vis = by_time[:, None] & cut_ok & torch.isfinite(raw) & exists[:, :, None, None]
        if nbr_age_exists is not None:
            vis = vis & nbr_age_exists[:, None, :, None]
        slot_exists = exists
        if drop_slots is not None:
            slot_exists = exists & ~drop_slots
        if mask_all_slots:
            slot_exists = torch.zeros_like(exists)
        vis = vis & slot_exists[:, :, None, None]
        zero = torch.zeros_like(raw)
        z_own = torch.where(vis, raw, zero).clamp(-mdl.POWER_CLIP, mdl.POWER_CLIP)
        ratio = self.ratio[primary[:, None], q][:, :, None, None]
        z_host = torch.where(vis, raw * ratio, zero).clamp(-mdl.POWER_CLIP, mdl.POWER_CLIP)
        nbr = f4.NeighbourInputs(
            slot_index=q, slot_exists=slot_exists, slot_relation=self.rel_table[primary],
            z_own=z_own, z_host=z_host, visible=vis,
        )  # fmt: skip
        out = F4Assembled(host=host, nbr=nbr, nbr_idx=nidx, slot_primary=slot)
        if cap_steps == lay.POWER_CAP_STEPS:
            check_neighbours(batch=out, c_idx=c_idx, shuffled=nbr_ages is not None)
        return out


def check_neighbours(*, batch: F4Assembled, c_idx: torch.Tensor, shuffled: bool) -> None:
    """Row-by-row assertions that cost nothing, run on every batch.

    No visible neighbour token is newer than the contract origin minus 27 hours; a token that is
    not visible carries zero on both scales; a visible token belongs to a filled slot; an empty
    slot holds no visible token; and a shuffled row is older than 16 days.
    """
    nbr = batch.nbr
    limit = (c_idx - lay.NEWEST_VISIBLE_BEFORE_CONTRACT)[:, None, None, None]
    assert bool(((batch.nbr_idx[:, None] <= limit) | ~nbr.visible).all()), (
        "neighbour power newer than origin - 27 h"
    )
    assert bool((nbr.z_own[~nbr.visible] == 0).all()) and bool(
        (nbr.z_host[~nbr.visible] == 0).all()
    ), "a masked neighbour token carries power"
    assert not bool(nbr.visible[~nbr.slot_exists].any()), (
        "a visible token in an empty or dropped slot"
    )
    assert bool((nbr.slot_index[nbr.slot_exists] >= 0).all())
    if shuffled:
        age = (c_idx[:, None, None] - batch.nbr_idx) // lay.STEPS_PER_DAY
        assert bool((age >= 1).all()), "a shuffled row is not older"


def drop_slots_for(*, n: int, key: tuple[int, ...]) -> np.ndarray:
    """Return the (n, 6) neighbour-dropout masks of one batch.

    The draw comes from a NumPy generator keyed on `key` (the seed, the stage, and the pass), so
    experiment 713's sample order, lead-day draws, and block-dropout draws stay identical.
    """
    rng = np.random.default_rng([DROP_KEY, *key])
    return rng.random((n, nb.MAX_SLOTS)) < NEIGHBOUR_DROPOUT


def host_scale_z(
    *, power_own: np.ndarray, mw_scale: np.ndarray, host: int, other: int
) -> np.ndarray:
    """Return the neighbour's deviation on the host's scale, from `z_own`, by plain NumPy (the KN9 reference)."""
    mw = power_own * mw_scale[other]
    return mw / mw_scale[host]


def visible_share(*, batch: F4Assembled) -> tuple[int, int]:
    """Return (visible neighbour tokens, tokens of the filled slots) of a batch, for the scorer's printout."""
    filled = int(batch.nbr.slot_exists.sum()) * lay.N_HISTORY_ROWS * lay.STEPS_PER_DAY
    return int(batch.nbr.visible.sum()), filled


def arrays_for_mode(
    *, mode: Mode, named: dict[ArrayName, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, ArrayName]:
    """Return (host history array, neighbour array, neighbour array name) that `mode` reads from `named`."""
    host_name, nbr_name = ARRAYS_OF_MODE[mode]
    return named[host_name], named[nbr_name], nbr_name

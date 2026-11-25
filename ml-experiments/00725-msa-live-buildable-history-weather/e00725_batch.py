# ruff: noqa: E501
"""Experiment 725: `LiveSource`, experiment 713's `Source` with the live rule for the history weather.

`assemble` calls the parent, then rewrites the history weather of the batch (`hist_weather`) by the
table of `e00725_rule`: each cell reads the analysis array (ERA5 or CAMS), the ENS value of the token's
own half-hour, or the CAMS day summary of D - 3. Every other input is the parent's.

**`with_power_in` is overridden.** Experiment 713's `Source.with_power_in` builds a plain `Source`, and
`e00713_train.synthetic_source` calls it for every stage 3 fit. Without the override, stage 3 would train
on the analysis weather and no assertion would raise. `with_power_in` here returns a shallow copy of
this object, so the arm, the rule, the arrays, and the call counters are carried over. Check KL9 counts
the calls of `assemble` and asserts the type of the source after `synthetic_source`.

**Training and scored cells differ.** In the training phase, an ENS cell with no `ft` row (or no stored
B1 row) reads the analysis, and the cell is counted (check KL7). At prediction, such a cell reads the
standardised value 0, as the Candidate's `standardised_table` does, and the cell is counted too.
"""

import copy
from dataclasses import dataclass, replace
from typing import Any, Final

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00725_rule as rule
import torch  # ty: ignore[unresolved-import]

N_ENS: Final = rule.N_ENS_COLUMNS
ERA5_LAG_STEPS: Final = rule.ERA5_LAG_HOURS * 2
CAMS_COLUMN_SLICE: Final = slice(19, 21)
GRID_DAY_US: Final = 86_400 * 1_000_000


@dataclass
class CellPlan:
    """Which cells of a batch read the ENS value.

    Attributes:
        wanted: (B, 14, 48, 19) bool, the cells the rule says read the ENS value.
        used: (B, 14, 48, 19) bool, wanted cells whose `ft` row and stored B1 row exist.
        fallback: (B, 14, 48, 19) bool, wanted cells with no row (an analysis in training, zero at prediction).
        origin_idx: (B, 14, 48) int64, the grid index of the `ft` run's forecast origin (-1 where none).
    """

    wanted: torch.Tensor
    used: torch.Tensor
    fallback: torch.Tensor
    origin_idx: torch.Tensor


class LiveSource(bt.Source):
    """Experiment 713's source with the live history weather (arms `L`, `E`; arm `Z` is the parent's batch)."""

    def __init__(
        self, *, base: bt.Source, rule_name: str, phase: str, ens_z: torch.Tensor,
        ens_found: torch.Tensor, ens_origin: torch.Tensor, cams_days: torch.Tensor, day0: int,
        grid_start_us: int, target_summary_back_days: int | None = None, live_summary: bool = True,
        verify: bool = True,
    ) -> None:  # fmt: skip
        """Hold the parent's arrays and the live-rule arrays.

        Args:
            base: The fold's `Source` as `e00713_inputs.build_fold_data` built it.
            rule_name: `L`, `Z`, or `E`.
            phase: `train` (a missing ENS cell reads the analysis) or `predict` (it reads 0).
            ens_z: (16, G, 19) float16 standardised ENS weather and stored B1 columns, on the device.
            ens_found: (16, G) bool, whether the cell has its `ft` row and stored B1 row.
            ens_origin: (16, G) int64, the grid index of the `ft` run's forecast origin (-1 if none).
            cams_days: (16, N_DAYS, 2) float32 standardised CAMS day summaries by UTC day code.
            day0: `e00632_data.DAY0`, the day code of slot 0.
            grid_start_us: The grid start in microseconds.
            target_summary_back_days: Report-only read-out `T3`: the target row's CAMS day summary is
                that of UTC day D minus this many days (3), instead of D - 1. None leaves the target row.
            live_summary: False makes the day summary the analysis array's (check KL1 only).
            verify: False skips the availability assertions of `check_cells` (check KL1 only).
        """
        assert rule_name in rule.RULES, rule_name
        assert phase in {"train", "predict"}, phase
        self.__dict__.update(base.__dict__)
        self.rule_name, self.phase = rule_name, phase
        self.ens_z, self.ens_found, self.ens_origin = ens_z, ens_found, ens_origin
        self.cams_days, self.day0 = cams_days, day0
        self.grid_day0 = grid_start_us // GRID_DAY_US
        self.target_summary_back_days = target_summary_back_days
        self.live_summary, self.verify = live_summary, verify
        dev = self.device
        codes = rule.cell_sources(rule=rule_name)  # ty: ignore[invalid-argument-type]
        self.codes = torch.from_numpy(codes).to(dev)
        self.rule_ens = torch.from_numpy(rule.ens_mask(rule=rule_name)).to(dev)  # ty: ignore[invalid-argument-type]
        self.counters: dict[str, int] = {
            "assemble_calls": 0, "ens_cells": 0, "fallback_cells": 0, "cells": 0,
        }  # fmt: skip

    def with_power_in(self, *, power_in: torch.Tensor) -> LiveSource:  # ty: ignore[invalid-method-override]
        """Return a source of the same type that reads `power_in` as history and keeps the targets.

        The copy is shallow: the arrays and the counters dictionary are shared with this object.
        """
        new = copy.copy(self)
        new.power_in = power_in
        return new

    def analysis_copy(self) -> LiveSource:
        """Return a copy whose rule reads the analysis in every cell, through the rewrite code path (check KL1)."""
        new = copy.copy(self)
        new.rule_ens = torch.zeros_like(self.rule_ens)
        new.live_summary, new.verify = False, False
        new.counters = dict.fromkeys(self.counters, 0)
        return new

    def with_rule(self, *, rule_name: str) -> LiveSource:
        """Return a copy of this source that reads another rule's table, with fresh counters."""
        assert rule_name in rule.RULES, rule_name
        new = copy.copy(self)
        new.rule_name = rule_name
        new.codes = torch.from_numpy(rule.cell_sources(rule=rule_name)).to(self.device)  # ty: ignore[invalid-argument-type]
        new.rule_ens = torch.from_numpy(rule.ens_mask(rule=rule_name)).to(self.device)  # ty: ignore[invalid-argument-type]
        new.counters = dict.fromkeys(self.counters, 0)
        return new

    def plan(self, *, primary: torch.Tensor, hist_idx: torch.Tensor) -> CellPlan:
        """Return the cells of a batch that read the ENS value, from the rule and the cache."""
        p3 = primary[:, None, None]
        found = self.ens_found[p3, hist_idx][..., None]
        wanted = self.rule_ens[None].expand(len(primary), -1, -1, -1)
        return CellPlan(
            wanted=wanted, used=wanted & found, fallback=wanted & ~found,
            origin_idx=self.ens_origin[p3, hist_idx],
        )  # fmt: skip

    def summary_slots(
        self, *, c_idx: torch.Tensor, hist_idx: torch.Tensor, live: bool
    ) -> torch.Tensor:
        """Return the (B, 14, 48) slot of the CAMS day summary of each token, clipped as experiment 713 clips."""
        token_day = hist_idx // lay.STEPS_PER_DAY
        day = token_day - 1
        if live:
            day = torch.minimum(
                day, (c_idx // lay.STEPS_PER_DAY - rule.CAMS_LAST_DAY_BEFORE)[:, None, None]
            )
        return (self.grid_day0 + day - self.day0).clamp(0, self.cams_days.shape[1] - 1)

    def check_cells(self, *, c_idx: torch.Tensor, hist_idx: torch.Tensor, plan: CellPlan) -> None:
        """Check KL4: every cell the batch reads was available at the native origin.

        Args:
            c_idx: The contract-origin grid index of each sample.
            hist_idx: The grid index of every history token.
            plan: The cell plan that holds the stored forecast origin of every ENS cell.

        Raises:
            AssertionError: If an ENS cell has no stored forecast origin, a forecast origin after the
                token's stamp or after the native origin, an ERA5 cell newer than the native origin
                minus 120 h, a CAMS cell whose source stamp is after 00:00 UTC on day D - 2, or a day
                summary newer than UTC day D - 3.
        """
        native = (c_idx - lay.NATIVE_OFFSET_STEPS)[:, None, None]
        used_tok = plan.used.any(dim=-1)
        org = plan.origin_idx
        assert bool((org[used_tok] > 0).all()), "an ENS cell has no stored forecast origin"
        assert bool((org <= hist_idx)[used_tok].all()), (
            "an ENS forecast origin is after the token's stamp"
        )
        assert bool((org <= native)[used_tok].all()), (
            "an ENS forecast origin is after the native origin"
        )
        era5_tok = (self.codes[..., list(rule.ERA5_COLUMNS)] == rule.SRC_ERA5).any(dim=-1)
        if bool(era5_tok.any()):
            assert bool((hist_idx[:, era5_tok] <= (native[:, :, 0] - ERA5_LAG_STEPS)).all()), (
                "an ERA5 cell is newer than the native origin minus 120 h"
            )
        cams_tok = self.codes[..., 0] == rule.SRC_CAMS
        if bool(cams_tok.any()):
            src = hist_idx[:, cams_tok] + hist_idx[:, cams_tok] % 2
            lim = (c_idx // lay.STEPS_PER_DAY - 2) * lay.STEPS_PER_DAY
            assert bool((src <= lim[:, None]).all()), (
                "a CAMS cell has a source stamp after 00:00 UTC on D - 2"
            )
        slots = self.summary_slots(c_idx=c_idx, hist_idx=hist_idx, live=True)
        last_slot = (
            self.grid_day0 + c_idx // lay.STEPS_PER_DAY - rule.CAMS_LAST_DAY_BEFORE - self.day0
        )
        assert bool((slots <= last_slot[:, None, None]).all()), (
            "a CAMS day summary is newer than UTC day D - 3"
        )

    def assemble(self, **kwargs: Any) -> bt.Assembled:  # ty: ignore[invalid-method-override]
        """Assemble a batch with the parent, then apply the live rule to the history weather.

        Args:
            **kwargs: The arguments of the parent's `assemble`.

        Returns:
            The parent's batch, with the history weather replaced by the live rule's cells.

        Raises:
            ValueError: For a placebo batch (`ages` given), which the rule's table does not cover.
        """
        out = super().assemble(**kwargs)
        self.counters["assemble_calls"] += 1
        if kwargs.get("ages") is not None:
            msg = "the live rule has no table for placebo ages"
            raise ValueError(msg)
        inputs = out.inputs
        hist, top = inputs.hist_weather, inputs.top_weather
        if self.rule_name != "Z":
            primary, hist_idx = inputs.primary, out.hist_idx
            plan = self.plan(primary=primary, hist_idx=hist_idx)
            if self.verify:
                self.check_cells(c_idx=out.c_idx, hist_idx=hist_idx, plan=plan)
            p3 = primary[:, None, None]
            ens = self.ens_z[p3, hist_idx].float()
            hist = hist.clone()
            new = torch.where(plan.used, ens, hist[..., :N_ENS])
            if self.phase == "predict":
                new = torch.where(plan.fallback, torch.zeros_like(new), new)
            hist[..., :N_ENS] = new
            slots = self.summary_slots(c_idx=out.c_idx, hist_idx=hist_idx, live=self.live_summary)
            hist[..., CAMS_COLUMN_SLICE] = self.cams_days[p3, slots].half().float()
            self.counters["cells"] += int(plan.wanted.sum())
            self.counters["ens_cells"] += int(plan.used.sum())
            self.counters["fallback_cells"] += int(plan.fallback.sum())
        if self.target_summary_back_days is not None and kwargs.get("ensemble"):
            top = top.clone()
            day = out.c_idx // lay.STEPS_PER_DAY - self.target_summary_back_days
            slot = (self.grid_day0 + day - self.day0).clamp(0, self.cams_days.shape[1] - 1)
            summary = self.cams_days[inputs.primary, slot].half().float()
            # a half-hour with no ensemble row keeps the parent's zero (the parent's weather is zero where `filled` is false)
            ens_day = (out.c_idx // lay.STEPS_PER_DAY - self.ens.first_day)[:, None]
            rel = lay.STEPS_PER_DAY * kwargs["lead_day"][:, None] + self.steps
            filled = self.ens.filled[inputs.primary[:, None], ens_day, rel]
            top[..., CAMS_COLUMN_SLICE] = torch.where(
                filled[..., None], summary[:, None, :], top[..., CAMS_COLUMN_SLICE]
            )
        return bt.Assembled(
            inputs=replace(inputs, hist_weather=hist, top_weather=top), target=out.target,
            target_idx=out.target_idx, hist_idx=out.hist_idx, c_idx=out.c_idx,
        )  # fmt: skip


def build_live_source(
    *, base: bt.Source, rule_name: str, phase: str, ens_z: torch.Tensor, ens_found: torch.Tensor,
    ens_origin: torch.Tensor, cams_days: torch.Tensor, day0: int, grid_start_us: int,
    target_summary_back_days: int | None = None, live_summary: bool = True,
) -> LiveSource:  # fmt: skip
    """Return a `LiveSource` around `base` (keyword-only wrapper for the call sites)."""
    return LiveSource(
        base=base, rule_name=rule_name, phase=phase, ens_z=ens_z, ens_found=ens_found,
        ens_origin=ens_origin, cams_days=cams_days, day0=day0, grid_start_us=grid_start_us,
        target_summary_back_days=target_summary_back_days, live_summary=live_summary,
    )  # fmt: skip

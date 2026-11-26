# ruff: noqa: E501, PT018
"""Experiment 716: batches with the four PV tensors (design section 4.1, WP3).

`PvSource` wraps experiment 713's `Source`: `assemble` calls `Source.assemble` unchanged and adds
`hist_pv`, `hist_pv_ok`, `top_pv`, and `top_pv_ok` from its own arrays. The history estimate is read
at each history token's own grid index; the target-row estimate is read from the table of the
origin's own 00Z run (or, for `HT-S`, of the donor run), keyed as `EnsTensors` is. The mask dropout
draws from a generator of its own, keyed by (seed, stage, pass, batch), so the trunk's dropout masks
and the global torch stream equal arm `Z`'s.
"""

from dataclasses import dataclass

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00716_model as pvm
import numpy as np
import torch  # ty: ignore[unresolved-import]

MASK_DROPOUT_SHARE: float = 0.10
"""The share of training samples that get each of the two PV masks."""
CAMS_RULE_AGE3_FIRST_MASKED_COLUMN: int = 25
"""Columns 25 to 47 of the age-3 row are not buildable live under the 36-hour rule (design section 9.2)."""


@dataclass
class PvAssembled:
    """One assembled batch with PV tensors."""

    inputs: pvm.PvInputs
    target: torch.Tensor
    target_idx: torch.Tensor
    hist_idx: torch.Tensor
    c_idx: torch.Tensor


@dataclass
class PvTables:
    """The PV arrays of one arm and fold, on the device.

    Attributes:
        hist_p: (16, G) float32, `p` at every grid stamp (the treatment, or a placebo's input).
        hist_ok: (16, G) bool.
        top_p0: (16, n_days, 672, 2) float32 target-row estimate.
        top_filled: (16, n_days, 672) bool.
        first_day: The day number of table day 0 (the same as `EnsTensors.first_day`).
        run_us: (16, n_days) int64 issue time of the run behind each row of `top_p0`, 0 where absent.
        exception_days: Table days whose donor run is issued after the origin (the `HT-S` exception);
            the issue-time assertion skips them.
    """

    hist_p: torch.Tensor
    hist_ok: torch.Tensor
    top_p0: torch.Tensor
    top_filled: torch.Tensor
    first_day: int
    run_us: torch.Tensor
    exception_days: frozenset[int] = frozenset()


def pv_dropout_masks(
    *, seed: int, stage: int, epoch: int, batch: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two mask draws of one training batch, each (n,) bool.

    The draws are (CAMS-rule mask, mask of 713's convention).

    The draw uses its own generator keyed by (seed, stage, pass, batch), never the global torch
    stream, so every PV arm and placebo of one seed and the arm `Z` see the same trunk dropout. A
    sample gets the CAMS-rule mask with probability 0.10, and another 0.10 get the mask of ages 2 to 5.
    """
    u = np.random.default_rng([716, seed, stage, epoch, batch]).random(n)
    cams = u < MASK_DROPOUT_SHARE
    convention = (u >= MASK_DROPOUT_SHARE) & (u < 2 * MASK_DROPOUT_SHARE)
    return cams, convention


def apply_ok_mask(
    *, ok: torch.Tensor, cams_rule: torch.Tensor | None, convention: torch.Tensor | None
) -> torch.Tensor:
    """Return `hist_pv_ok` with the CAMS-rule mask and the mask of ages 2 to 5 applied.

    The CAMS-rule mask covers the age-2 row and columns 25 to 47 of the age-3 row.

    `cams_rule` and `convention` are (B,) bool or None. Rows are the history rows 0 to 13 (ages 2 to 15).
    """
    ok = ok.clone()
    if cams_rule is not None:
        ok[:, 0] &= ~cams_rule[:, None]
        ok[:, 1, CAMS_RULE_AGE3_FIRST_MASKED_COLUMN:] &= ~cams_rule[:, None]
    if convention is not None:
        ok[:, :4] &= ~convention[:, None, None]
    return ok


class PvSource:
    """A 713 `Source` plus the PV arrays of one arm."""

    def __init__(self, *, source: bt.Source, tables: PvTables, world: bool = False) -> None:
        """Hold the wrapped source and the tables; both sit on one device.

        `world` is true for the known-answer worlds, whose target row reads the dense ERA5-and-CAMS
        weather (stage 1's weather) and whose source has no ensemble table.
        """
        assert world or source.ens is not None, (
            "F7 reads the ensemble table in stages 2 and 3 and at prediction"
        )
        if not world:
            assert tables.first_day == source.ens.first_day, (
                tables.first_day,
                source.ens.first_day,
            )  # ty: ignore[unresolved-attribute]
        self.source, self.tables, self.world = source, tables, world
        self.device = source.device

    def with_power_in(self, *, power_in: torch.Tensor) -> PvSource:
        """Return a source whose history power is `power_in` (stage 3's synthetic shifts); the PV arrays are unchanged."""
        return PvSource(
            source=self.source.with_power_in(power_in=power_in),
            tables=self.tables,
            world=self.world,
        )

    @property
    def ens(self) -> bt.EnsTensors | None:
        """Return the wrapped source's ensemble table (713's `make_samples` reads it)."""
        return self.source.ens

    @property
    def steps(self) -> torch.Tensor:
        """Return the 48 column indices of the wrapped source."""
        return self.source.steps

    @property
    def power_z(self) -> torch.Tensor:
        """Return the wrapped source's standardised power."""
        return self.source.power_z

    def assemble(
        self, *, primary: torch.Tensor, c_idx: torch.Tensor, lead_day: torch.Tensor, ensemble: bool = True,
        mask_all_power: torch.Tensor | None = None, mask_newest_row: torch.Tensor | None = None,
        target_end: int | None = None, cams_rule: torch.Tensor | None = None,
        convention: torch.Tensor | None = None, power_in: torch.Tensor | None = None,
        cap_steps: int = lay.POWER_CAP_STEPS, hist_p_override: torch.Tensor | None = None,
        top_p0_override: torch.Tensor | None = None,
    ) -> PvAssembled:  # fmt: skip
        """Return the batch: 713's inputs and the PV tensors.

        Args:
            primary: (B,) long.
            c_idx: (B,) long, the grid index of the 12:00 UTC contract origin.
            lead_day: (B,) long, `k - 1`.
            ensemble: Must be true (F7 starts at stage 2).
            mask_all_power: 713's block dropout of all history power.
            mask_newest_row: 713's block dropout of the age-2 row.
            target_end: Targets at grid indices at or after this are NaN.
            cams_rule: (B,) bool, the CAMS-rule PV mask (training dropout or a read-out).
            convention: (B,) bool, the mask of ages 2 to 5.
            power_in: Override of the power read as history (check KA5).
            cap_steps: The power cap in steps; 0 is the negative control.
            hist_p_override: (16, G) a replacement history `p` (the poison copy of KF2).
            top_p0_override: (16, n_days, 672, 2) a replacement target table (the poison copy of KF2).

        Returns:
            The assembled batch of inputs and PV tensors.
        """
        assert ensemble, "F7 fits and predicts with the ensemble target row"
        out = self.source.assemble(
            primary=primary, c_idx=c_idx, lead_day=lead_day, ensemble=not self.world,
            mask_all_power=mask_all_power, mask_newest_row=mask_newest_row, target_end=target_end,
            power_in=power_in, cap_steps=cap_steps,
        )  # fmt: skip
        tb = self.tables
        p3 = primary[:, None, None]
        hist_p_all = tb.hist_p if hist_p_override is None else hist_p_override
        hist_pv = hist_p_all[p3, out.hist_idx]
        hist_ok = tb.hist_ok[p3, out.hist_idx] & torch.isfinite(hist_pv)
        hist_ok = apply_ok_mask(ok=hist_ok, cams_rule=cams_rule, convention=convention)
        hist_pv = torch.where(hist_ok, hist_pv, torch.zeros_like(hist_pv))
        day = (c_idx // lay.STEPS_PER_DAY - tb.first_day)[:, None]
        rel = lay.STEPS_PER_DAY * lead_day[:, None] + self.source.steps
        p2 = primary[:, None]
        top_table = tb.top_p0 if top_p0_override is None else top_p0_override
        top_pv = top_table[p2, day, rel]
        top_ok = tb.top_filled[p2, day, rel] & torch.isfinite(top_pv).all(dim=-1)
        top_pv = torch.where(top_ok[..., None], top_pv, torch.zeros_like(top_pv))
        if not self.world:
            self.check_run_times(primary=primary, day=day[:, 0], c_idx=c_idx)
        inputs = pvm.PvInputs(
            **out.inputs.__dict__, hist_pv=hist_pv, hist_pv_ok=hist_ok, top_pv=top_pv, top_pv_ok=top_ok
        )  # fmt: skip
        batch = PvAssembled(
            inputs=inputs,
            target=out.target,
            target_idx=out.target_idx,
            hist_idx=out.hist_idx,
            c_idx=c_idx,
        )
        if cap_steps == lay.POWER_CAP_STEPS:
            check_pv_cheap(batch=batch)
        return batch

    def check_run_times(
        self, *, primary: torch.Tensor, day: torch.Tensor, c_idx: torch.Tensor
    ) -> None:
        """Assert that the run behind each target-row estimate was issued at least 9 hours before the native origin.

        `HT-S`'s listed exception days (donor run later than the origin) are skipped.
        """
        tb = self.tables
        run = tb.run_us[primary, day]
        present = run > 0
        if tb.exception_days:
            skip = torch.isin(
                day + tb.first_day, torch.tensor(sorted(tb.exception_days), device=day.device)
            )
            present = present & ~skip
        origin = lay.GRID_START_US + (c_idx - lay.NATIVE_OFFSET_STEPS) * lay.STEP_US
        assert bool((run[present] <= (origin - lay.ENSEMBLE_LEAD_US)[present]).all()), (
            "a PV target-row estimate comes from a run issued after the origin minus 9 hours"
        )


def check_pv_cheap(*, batch: PvAssembled) -> None:
    """Row-by-row assertions on every batch: a PV token is only where its flag says, and is finite and in range."""
    inp = batch.inputs
    assert bool((inp.hist_pv[~inp.hist_pv_ok] == 0).all()), "a masked history token carries PV"
    assert bool((inp.top_pv[~inp.top_pv_ok] == 0).all()), "a masked target token carries PV"
    assert bool(torch.isfinite(inp.hist_pv).all()) and bool(torch.isfinite(inp.top_pv).all()), (
        "non-finite PV"
    )
    assert float(inp.hist_pv.min()) >= 0.0 and float(inp.hist_pv.max()) <= 10.0 + 1e-6, (
        "history p outside 0 to 10"
    )
    assert bool((batch.hist_idx < batch.target_idx[:, :1, None]).all()), "history after the target"

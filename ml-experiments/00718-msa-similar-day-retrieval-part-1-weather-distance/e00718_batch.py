# ruff: noqa: E501
"""Experiment 718: batch assembly for any number of history rows chosen by a row table (design section 5, 15).

`F9Source` wraps experiment 713's `Source` and a row table per target-weather version (`era5` for
stage 1, `ens` for stages 2 and 3 and prediction). `Source.assemble` of experiment 713 already takes
the ages of the history rows as an argument of any width, applies the 27 h rule, and sets `row_exists`
from any visible finite reading (`has_reading`, line 160), so this module calls it unchanged and adds
three things around it:

- the ages come from the row table, with an absent row marked `row_exists` False;
- the 4.5% block dropout hides the row of age 2 wherever that row sits, and nothing when the sample
  holds no such row (experiment 713 hid slot 0);
- the age-bucket tensor, which an assertion ties to the grid indices of the rows.

`poison_check` is rewritten because experiment 713's negative control is skipped whenever `ages` is
given (`e00713_batch.py` line 323), and every F9 arm passes `ages`.
"""

from dataclasses import dataclass
from typing import Final

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00718_layout as lay
import e00718_model as f9
import e00718_retrieval as ret
import numpy as np
import torch  # ty: ignore[unresolved-import]

ABSENT_FILLER_AGE: Final = 2
"""The age that an absent row takes in the index arithmetic; `row_exists` is False for it."""
TABLE_ERA5: Final = "era5"
TABLE_ENS: Final = "ens"


@dataclass
class F9Assembled:
    """One assembled batch with the history rows of a row table."""

    inputs: f9.F9Inputs
    target: torch.Tensor  # (B, 48)
    target_idx: torch.Tensor  # (B, 48)
    hist_idx: torch.Tensor  # (B, H, 48)
    c_idx: torch.Tensor  # (B,)
    ages: torch.Tensor  # (B, H) long, ABSENT_FILLER_AGE where a row is absent
    measure: torch.Tensor  # (B, H) long, the measure code of each row


class F9Source:
    """Experiment 713's dense arrays on the device, and the row tables."""

    def __init__(self, *, base: bt.Source, tables: dict[str, ret.RowTable]) -> None:
        """Hold the source and the tables (`era5` and `ens`); stage 1 reads the first, the others the second."""
        self.base = base
        self.tables = tables
        self.device = base.device
        self.n_grid = base.n_grid
        self.ens = base.ens
        self.power_in = base.power_in
        self.power_z = base.power_z

    def with_power_in(self, *, power_in: torch.Tensor) -> F9Source:
        """Return a source that reads `power_in` as history and keeps the targets and the tables."""
        return F9Source(base=self.base.with_power_in(power_in=power_in), tables=self.tables)

    def rows_of(
        self, *, primary: np.ndarray, c_idx: np.ndarray, lead_day: np.ndarray, ensemble: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the (B, H) ages and measure codes of the samples from the table of the stage."""
        table = self.tables[TABLE_ENS if ensemble else TABLE_ERA5]
        day = c_idx // lay.STEPS_PER_DAY
        return (
            table.lookup(primary=primary, day=day, lead_day=lead_day),
            table.lookup_measure(primary=primary, day=day, lead_day=lead_day),
        )

    def assemble(
        self, *, primary: torch.Tensor, c_idx: torch.Tensor, lead_day: torch.Tensor, ensemble: bool,
        mask_all_power: torch.Tensor | None = None, mask_newest_row: torch.Tensor | None = None,
        target_end: int | None = None, cap_steps: int = lay713.POWER_CAP_STEPS,
        power_in: torch.Tensor | None = None, ens_day_offset: int = 0,
        ages_override: np.ndarray | None = None,
    ) -> F9Assembled:  # fmt: skip
        """Return the batch of samples (Primary, contract origin index, lead day).

        Args:
            primary: (B,) long.
            c_idx: (B,) long, the grid index of the 12:00 UTC contract origin.
            lead_day: (B,) long, `k - 1`.
            ensemble: Read the target row from the ensemble table, and the ensemble row table.
            mask_all_power: (B,) bool, block dropout: no history power at all.
            mask_newest_row: (B,) bool, block dropout: the row of age 2 has no power, wherever it sits.
            target_end: Targets at grid indices at or after this are NaN.
            cap_steps: The power cap in steps before the native origin (0 is a negative control).
            power_in: Override of the power read as history (the poison checks).
            ens_day_offset: Days added to the origin's day when the ensemble row is looked up (must be 0).
            ages_override: (B, H) ages that replace the table's (the negative controls of the poison check).

        Returns:
            The assembled batch of the samples.
        """
        p_np, c_np, k_np = (x.detach().cpu().numpy() for x in (primary, c_idx, lead_day))
        if ages_override is None:
            ages_np, measure_np = self.rows_of(
                primary=p_np, c_idx=c_np, lead_day=k_np, ensemble=ensemble
            )
        else:
            ages_np, measure_np = ages_override, np.full(ages_override.shape, lay.MEASURE_FIXED)
        present = ages_np >= 0
        dev = self.device
        ages_t = torch.from_numpy(np.where(present, ages_np, ABSENT_FILLER_AGE)).to(dev)
        exists_t = torch.from_numpy(present).to(dev)
        out = self.base.assemble(
            primary=primary, c_idx=c_idx, lead_day=lead_day, ensemble=ensemble, ages=ages_t,
            age_exists=exists_t, mask_all_power=mask_all_power, mask_newest_row=None,
            target_end=target_end, cap_steps=cap_steps, power_in=power_in, ens_day_offset=ens_day_offset,
        )  # fmt: skip
        inp = out.inputs
        if mask_newest_row is not None:
            hide = mask_newest_row[:, None] & (ages_t == lay.MIN_AGE) & exists_t
            visible = inp.hist_visible & ~hide[:, :, None]
            inp = inp.with_power(
                power=torch.where(visible, inp.hist_power, torch.zeros_like(inp.hist_power)),
                visible=visible,
            )
        bucket = torch.from_numpy(lay.age_bucket(ages=np.where(present, ages_np, -1))).to(dev)
        assert bool(
            (out.c_idx[:, None] - out.hist_idx[:, :, 0] == lay.STEPS_PER_DAY * ages_t).all()
        ), "the grid indices disagree with the ages"
        grid_age = (out.c_idx[:, None] - out.hist_idx[:, :, 0]) // lay.STEPS_PER_DAY
        grid_bucket = torch.from_numpy(lay.age_bucket(ages=grid_age.cpu().numpy())).to(dev)
        assert bool((bucket[exists_t] == grid_bucket[exists_t]).all()), (
            "the bucket tensor is not the function of the grid indices"
        )
        f9_inputs = f9.F9Inputs(**inp.__dict__, age_bucket=bucket)
        batch = F9Assembled(
            inputs=f9_inputs, target=out.target, target_idx=out.target_idx, hist_idx=out.hist_idx,
            c_idx=out.c_idx, ages=ages_t, measure=torch.from_numpy(measure_np).to(dev),
        )  # fmt: skip
        if cap_steps == lay713.POWER_CAP_STEPS:
            bt.check_cheap(
                batch=out.__class__(
                    inputs=inp,
                    target=out.target,
                    target_idx=out.target_idx,
                    hist_idx=out.hist_idx,
                    c_idx=out.c_idx,
                )
            )
        return batch


# ---------------------------------------------------------------------------------- the checks of KR2


def poison_check(
    *,
    source: F9Source,
    batch: F9Assembled,
    ensemble: bool,
    lead_day: torch.Tensor,
    limit: int = 64,
    need_control: bool = True,
) -> str:
    """KR2 (a) on the first `limit` samples of `batch`: poisoned copies change no input.

    For each sample, the power of its Primary newer than `c - 54` is replaced by 1e6, and in a second
    run by NaN, in a private copy. The rebuilt tensors must equal the batch's, bit for bit. The negative
    control rebuilds the sample from the poisoned copy with a cap of 0 and the ages of the batch, and
    must change the inputs wherever the sample holds a row that the cap-0 rule makes visible (a row of
    age 2, whose newest half-hours lie between 27 h and 24.5 h before the contract origin).
    `need_control` is False for arms whose rows do not always include age 2 (a random or weather-nearest pool).
    """
    n = min(limit, len(batch.c_idx))
    controls = 0
    for i in range(n):
        p = int(batch.inputs.primary[i])
        c = int(batch.c_idx[i])
        ages = batch.ages[i : i + 1].cpu().numpy()
        ages = np.where(batch.inputs.row_exists[i : i + 1].cpu().numpy(), ages, lay.ABSENT)
        for value in (bt.POISON_VALUE, float("nan")):
            copy = source.power_in.clone()
            copy[p, c - lay713.NEWEST_VISIBLE_BEFORE_CONTRACT + 1 :] = value
            one = source.assemble(
                primary=batch.inputs.primary[i : i + 1], c_idx=batch.c_idx[i : i + 1],
                lead_day=lead_day[i : i + 1], ensemble=ensemble, power_in=copy, ages_override=ages,
            )  # fmt: skip
            for name in ("hist_power", "hist_visible", "row_exists", "age_bucket"):
                assert torch.equal(getattr(one.inputs, name)[0], getattr(batch.inputs, name)[i]), (
                    f"sample {i}: {name} read a poisoned reading ({value})"
                )
        has_age2 = bool(((batch.ages[i] == lay.MIN_AGE) & batch.inputs.row_exists[i]).any())
        if not has_age2:
            continue
        copy = source.power_in.clone()
        copy[p, c - lay713.NEWEST_VISIBLE_BEFORE_CONTRACT + 1 :] = bt.POISON_VALUE
        try:
            loose = source.assemble(
                primary=batch.inputs.primary[i : i + 1], c_idx=batch.c_idx[i : i + 1],
                lead_day=lead_day[i : i + 1], ensemble=ensemble, cap_steps=0, power_in=copy, ages_override=ages,
            )  # fmt: skip
        except AssertionError:
            controls += 1
            continue
        assert not torch.equal(loose.inputs.hist_power[0], batch.inputs.hist_power[i]), (
            f"sample {i}: the negative control (cap 0) did not change the inputs"
        )
        controls += 1
    assert controls > 0 or not need_control, (
        "no sample held a row of age 2, so the negative control did not run"
    )
    return f"KR2 (a): {n} samples identical under poison 1e6 and NaN; the cap-0 negative control changed {controls} samples"


def block_dropout_check(
    *,
    source: F9Source,
    primary: torch.Tensor,
    c_idx: torch.Tensor,
    lead_day: torch.Tensor,
    ensemble: bool,
) -> str:
    """KR5, the input-level part: the newest-row draw hides the row of age 2 wherever it sits, and nothing otherwise."""
    on = torch.ones(len(c_idx), dtype=torch.bool, device=c_idx.device)
    off = ~on
    a = source.assemble(
        primary=primary, c_idx=c_idx, lead_day=lead_day, ensemble=ensemble, mask_newest_row=on
    )
    b = source.assemble(
        primary=primary, c_idx=c_idx, lead_day=lead_day, ensemble=ensemble, mask_newest_row=off
    )
    is_age2 = (a.ages == lay.MIN_AGE) & a.inputs.row_exists
    assert not bool(a.inputs.hist_visible[is_age2].any()), (
        "a row of age 2 stayed visible under the newest-row draw"
    )
    assert bool((a.inputs.hist_power[is_age2] == 0).all())
    other = ~is_age2
    assert torch.equal(a.inputs.hist_visible[other], b.inputs.hist_visible[other]), (
        "the draw hid a row that is not of age 2"
    )
    assert torch.equal(a.inputs.hist_power[other], b.inputs.hist_power[other])
    n_with = int(is_age2.any(dim=1).sum())
    return f"KR5 (input level): the draw hides the row of age 2 in {n_with} of {len(c_idx)} samples and no other row"


def phase_tables_equal(*, predict: ret.RowTable, train: ret.RowTable, fold_start_day: int) -> str:
    """KR2 (c): entries of origins before the fold month are bit-identical under the two phases of power.

    Entries of origins on or after `fold_start_day` may differ, because the train-phase power is NaN
    from the fold month's start. The function asserts equality before it and counts the differences
    after it, so that a caller can show the negative control.
    """
    before = predict.days < fold_start_day
    assert before.any(), "no origin day before the fold month"
    assert np.array_equal(predict.ages[:, before], train.ages[:, before]), (
        "a training entry depends on the phase of the power"
    )
    assert np.array_equal(predict.measure[:, before], train.measure[:, before])
    after = ~before
    n_diff = (
        int((predict.ages[:, after] != train.ages[:, after]).any(axis=-1).sum())
        if after.any()
        else 0
    )
    return f"KR2 (c): {int(before.sum())} origin days before the fold month are bit-identical; {n_diff} entries after it differ (the control)"


def newest_block_present(*, table: ret.RowTable, arrays: ret.PoolArrays, days: np.ndarray) -> str:
    """KR2 (d): the table of `U` holds the block of age 2 wherever that block is eligible (under the prediction-phase power)."""
    assert table.arm in {"U", "UX", "U27"}
    n = missing = 0
    for pi, p in enumerate(table.primaries):
        for day in days:
            di = int(np.searchsorted(table.days, day))
            assert table.days[di] == day
            c = int(lay.contract_idx_of_day(day=int(day)))
            if not bool(
                ret.eligible_ages(
                    arrays=arrays, primary=int(p), contract_idx=c, ages=np.array([2])
                )[0]
            ):
                continue
            for lead in range(lay.N_LEAD_DAYS):
                n += 1
                missing += int(not (table.ages[pi, di, lead] == 2).any())
    assert missing == 0, f"{missing} of {n} samples lack the eligible block of age 2"
    return f"KR2 (d): the block of age 2 is in the table of {table.arm} in all {n} samples where it is eligible"

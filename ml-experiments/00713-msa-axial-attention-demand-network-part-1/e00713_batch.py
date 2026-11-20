# ruff: noqa: E501, PT018
"""Experiment 713: batch assembly on the device from the dense arrays (section 2.2).

A `Source` holds, for 16 Primaries and every half-hour of the grid, the standardised power, the
standardised weather and B1 columns (ERA5 and CAMS), the standardised date-time columns, and the day
type. For the stages that read the ensemble forecast it also holds an `EnsTensors` table of the target
row's inputs keyed by (Primary, origin day, position in the 672 contract half-hours).

`Source.assemble` cuts one batch out of these arrays by index. It applies the 27-hour power rule
(`e00713_layout`), the block dropout, the placebo rows, and the stage's target cut, and it runs the
cheap assertions of the design on every batch. `poison_check` is check KA5: it rebuilds a sample from
a copy of the arrays in which every reading newer than the rule allows is replaced by 1e6 (and, in a
second run, by NaN), and requires the inputs to be bit-identical.
"""

from dataclasses import dataclass
from typing import Final

import e00713_layout as lay
import e00713_model as mdl
import torch  # ty: ignore[unresolved-import]

N_PRIMARIES: Final = 16
TOP_WIDTH: Final = mdl.N_WEATHER + mdl.N_B1
"""Columns of a weather array: 7 weather columns, then 14 B1 columns."""
POISON_VALUE: Final = 1e6


@dataclass
class EnsTensors:
    """The target row's ensemble inputs, on the device.

    Attributes:
        weather: (16, n_days, 672, 21) float16: 7 ensemble-mean columns, 14 B1 columns.
        spread: (16, n_days, 672, 7) float16.
        filled: (16, n_days, 672) bool: whether the row exists.
        first_day: The day number (grid index // 48) of day index 0.
        run_us: (16, n_days) int64: the issue time of the ensemble run behind each (Primary, origin day)
            in microseconds, 0 where the row is absent. It is the origin minus 9 hours, set by
            `ens_table` from the 09:00 origin it checked, not read from the run. The assertion in
            `Source.assemble` is therefore a structural check of the code path (a later edit that
            moves the run, as the KA6 canary does, is caught); the check against the stored
            `nwp_init_time` column is `e00713_check_inputs.check_nwp_init`, on the scored origins.
    """

    weather: torch.Tensor
    spread: torch.Tensor
    filled: torch.Tensor
    first_day: int
    run_us: torch.Tensor | None = None  # (16, n_days) int64: issue time of the run behind each row


@dataclass
class Assembled:
    """One assembled batch."""

    inputs: mdl.MsaInputs
    target: torch.Tensor  # (B, 48) standardised power, NaN where the token is not a target
    target_idx: torch.Tensor  # (B, 48) grid index of each target token
    hist_idx: torch.Tensor  # (B, 14, 48) grid index of each history token
    c_idx: torch.Tensor  # (B,)


CAMS_COLUMNS: Final = [19, 20]
"""The two `cams_prev_day_*` columns of the 21 weather columns (7 weather, 12 B1 grid, 2 CAMS)."""


class Source:
    """Dense arrays of one fold on the device, and batch assembly by index."""

    def __init__(
        self,
        *,
        power_z: torch.Tensor,
        dense_weather: torch.Tensor,
        dense_datetime: torch.Tensor,
        day_type: torch.Tensor,
        ens: EnsTensors | None = None,
        power_in: torch.Tensor | None = None,
    ) -> None:
        """Hold the arrays.

        Args:
            power_z: (16, G) standardised cleaned power; NaN where flagged, missing, or hidden.
            dense_weather: (16, G, 21) float16, ERA5 and CAMS weather and B1, standardised.
            dense_datetime: (16, G, 21) float16, standardised.
            day_type: (G,) long, the day type of every grid index (`e00713_layout.day_types_for_grid`).
            ens: The ensemble target-row table, or None for stage 1.
            power_in: The power read as history, when it differs from the targets (synthetic shifts,
                an injected bank); defaults to `power_z`.
        """
        self.power_z = power_z
        self.power_in = power_z if power_in is None else power_in
        self.dense_weather, self.dense_datetime = dense_weather, dense_datetime
        self.day_type, self.ens = day_type, ens
        self.n_grid = power_z.shape[1]
        self.device = power_z.device
        self.steps = torch.arange(lay.STEPS_PER_DAY, device=self.device)
        self.default_ages = torch.from_numpy(lay.history_ages()).to(self.device)

    def with_power_in(self, *, power_in: torch.Tensor) -> Source:
        """Return a source that reads `power_in` as history and keeps the targets."""
        return Source(
            power_z=self.power_z, dense_weather=self.dense_weather,
            dense_datetime=self.dense_datetime, day_type=self.day_type, ens=self.ens,
            power_in=power_in,
        )  # fmt: skip

    def assemble(  # noqa: PLR0915
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
        power_in: torch.Tensor | None = None,
        ens_day_offset: int = 0,
    ) -> Assembled:
        """Return the batch of samples (Primary, contract origin index `c_idx`, lead day).

        Args:
            primary: (B,) long.
            c_idx: (B,) long, the grid index of the 12:00 UTC contract origin.
            lead_day: (B,) long, `k - 1`.
            ensemble: Read the target row from the ensemble table (stages 2 and 3, prediction)
                instead of ERA5 and CAMS (stage 1).
            ages: (B, 14) long history ages (the placebo); None for ages 2 to 15.
            age_exists: (B, 14) bool, False for a placebo slot with no block.
            mask_all_power: (B,) bool, block dropout: no history power at all.
            mask_newest_row: (B,) bool, block dropout: the age-2 row has no power.
            target_end: Targets at grid indices at or after this are NaN.
            cap_steps: The power cap in steps before the native origin (48; 0 is the negative
                control of check KA5).
            power_in: Override of the power read as history (check KA5).
            ens_day_offset: Days added to the origin's day when the ensemble row is looked up. It must
                be 0: the canary of check KA6 passes 1 and requires the assertion to raise.

        Returns:
            The batch, with its model inputs, target row, and grid indices.
        """
        b = len(primary)
        power = self.power_in if power_in is None else power_in
        steps = self.steps
        p3 = primary[:, None, None]
        ages_b = self.default_ages[None, :].expand(b, lay.N_HISTORY_ROWS) if ages is None else ages
        hist_idx = c_idx[:, None, None] - lay.STEPS_PER_DAY * ages_b[:, :, None] + steps
        assert int(hist_idx.min()) >= 0 and int(hist_idx.max()) < self.n_grid, (
            "history leaves the grid"
        )
        newest = c_idx - lay.NATIVE_OFFSET_STEPS - cap_steps
        by_time = hist_idx <= newest[:, None, None]
        raw = power[p3, hist_idx]
        finite = torch.isfinite(raw)
        has_reading = (finite & by_time).any(dim=2)
        row_exists = has_reading if age_exists is None else (has_reading & age_exists)
        visible = by_time & finite & row_exists[:, :, None]
        if mask_all_power is not None:
            visible &= ~mask_all_power[:, None, None]
        if mask_newest_row is not None:
            visible[:, 0] &= ~mask_newest_row[:, None]
        hist_power = torch.where(visible, raw, torch.zeros_like(raw))
        top_idx = c_idx[:, None] + lay.STEPS_PER_DAY * lead_day[:, None] + steps
        assert int(top_idx.max()) < self.n_grid, "target row leaves the grid"
        p2 = primary[:, None]
        if ensemble:
            assert self.ens is not None, "this source has no ensemble table"
            assert ens_day_offset == 0, (
                "the ensemble run is not the origin's own run (issued after the origin)"
            )
            day = (c_idx // lay.STEPS_PER_DAY - self.ens.first_day + ens_day_offset)[:, None]
            n_days = self.ens.filled.shape[1]
            assert bool(((day >= 0) & (day < n_days)).all()), (
                f"an origin day lies outside the ensemble table (0 to {n_days - 1}): "
                f"{int(day.min())} to {int(day.max())}"
            )  # raised here, not as a device-side assert in the gather below
            if self.ens.run_us is not None:
                run = self.ens.run_us[primary, day[:, 0]]
                origin = lay.GRID_START_US + (c_idx - lay.NATIVE_OFFSET_STEPS) * lay.STEP_US
                present = run > 0
                assert bool((run[present] <= (origin - lay.ENSEMBLE_LEAD_US)[present]).all()), (
                    "nwp_init_time is later than the origin minus 9 hours"
                )
            rel = lay.STEPS_PER_DAY * lead_day[:, None] + steps
            top_weather = self.ens.weather[p2, day, rel].float()
            top_spread = self.ens.spread[p2, day, rel].float()
            filled = self.ens.filled[p2, day, rel]
        else:
            top_weather = self.dense_weather[p2, top_idx].float()
            # the two CAMS columns of the target row are the origin's, as in experiment 574's
            # pre-training rows and in stages 2 and 3: the dense array holds them per grid stamp, so
            # read them at the origin stamp and not at each target half-hour (code review 1, B1)
            origin_stamp = c_idx - lay.NATIVE_OFFSET_STEPS
            cams = self.dense_weather[primary, origin_stamp][:, CAMS_COLUMNS].float()
            top_weather[:, :, CAMS_COLUMNS] = cams[:, None, :]
            top_spread = torch.zeros(b, lay.STEPS_PER_DAY, mdl.N_SPREAD, device=self.device)
            filled = torch.ones(b, lay.STEPS_PER_DAY, dtype=torch.bool, device=self.device)
        target = self.power_z[p2, top_idx].clone()
        target[~filled] = float("nan")
        if ensemble:
            # a target row with a gap in its ensemble weather is dropped from the loss (code review 1,
            # S1): the unfilled tokens carry zero weather, which prediction never sees
            target[~filled.all(dim=1)] = float("nan")
        if target_end is not None:
            target[top_idx >= target_end] = float("nan")
        inputs = mdl.MsaInputs(
            primary=primary, lead_day=lead_day, top_weather=top_weather, top_spread=top_spread,
            top_datetime=self.dense_datetime[p2, top_idx].float(),
            hist_weather=self.dense_weather[p3, hist_idx].float(),
            hist_datetime=self.dense_datetime[p3, hist_idx].float(),
            hist_power=hist_power.clamp(-mdl.POWER_CLIP, mdl.POWER_CLIP), hist_visible=visible,
            row_exists=row_exists, day_type_top=self.day_type[top_idx].long(),
            day_type_hist=self.day_type[hist_idx].long(),
        )  # fmt: skip
        out = Assembled(
            inputs=inputs, target=target, target_idx=top_idx, hist_idx=hist_idx, c_idx=c_idx
        )
        if cap_steps == lay.POWER_CAP_STEPS:
            check_cheap(batch=out)
        return out


def ka6_canary(
    *, source: Source, primary: torch.Tensor, c_idx: torch.Tensor, lead_day: torch.Tensor
) -> str:
    """Check KA6: a table keyed by the run of the next day fails the origin-time assertion on the data.

    The canary copies the source with every present run moved one day later and assembles the samples
    whose (Primary, origin day) row exists. The data assertion in `Source.assemble` must raise.
    """
    import copy
    import dataclasses

    ens = source.ens
    assert ens is not None
    assert ens.run_us is not None, "the source carries no run times"
    day = c_idx // lay.STEPS_PER_DAY - ens.first_day
    n_days = ens.run_us.shape[1]
    day_min, day_max = (
        int(day.min().cpu()),
        int(day.max().cpu()),
    )  # checked before the gather, so the error is readable
    if day_min < 0 or day_max >= n_days:
        msg = f"KA6 canary origin day index {day_min} to {day_max} is outside the ensemble table (0 to {n_days - 1})"
        raise IndexError(msg)
    present = ens.run_us[primary, day] > 0
    assert bool(present.any()), "no sample of the canary has an ensemble row"
    bad = copy.copy(source)
    later = torch.where(ens.run_us > 0, ens.run_us + 24 * lay.HOUR_US, ens.run_us)
    bad.ens = dataclasses.replace(ens, run_us=later)
    try:
        bad.assemble(
            primary=primary[present],
            c_idx=c_idx[present],
            lead_day=lead_day[present],
            ensemble=True,
        )
    except AssertionError as err:
        return str(err)
    msg = "KA6: a target row from the next day's ensemble run was accepted"
    raise AssertionError(msg)


def check_cheap(*, batch: Assembled) -> None:
    """Row-by-row assertions that cost nothing, run on every batch.

    No visible history token is newer than the contract origin minus 27 hours; a token that is not
    visible carries power 0; the targets are all after the native origin; every history token is
    older than the target row's first token.
    """
    inp = batch.inputs
    limit = (batch.c_idx - lay.NEWEST_VISIBLE_BEFORE_CONTRACT)[:, None, None]
    assert bool(((batch.hist_idx <= limit) | ~inp.hist_visible).all()), (
        "power newer than origin - 27 h"
    )
    assert bool((inp.hist_power[~inp.hist_visible] == 0).all()), "a masked token carries power"
    assert bool((batch.target_idx > (batch.c_idx - lay.NATIVE_OFFSET_STEPS)[:, None]).all())
    assert bool((batch.hist_idx < batch.target_idx[:, :1, None]).all()), "history after the target"
    assert not bool(inp.hist_visible[~inp.row_exists].any()), "a visible token in an absent row"


def poison_check(
    *,
    source: Source,
    batch: Assembled,
    ages: torch.Tensor | None,
    age_exists: torch.Tensor | None,
    ensemble: bool,
    lead_day: torch.Tensor,
    limit: int = 64,
) -> str:
    """Check KA5 on the first `limit` samples of `batch`: poisoned copies change no input.

    For each sample the power of its Primary newer than `c - 54` is replaced by 1e6 in a private copy,
    and in a second run by NaN. The rebuilt history tensors must equal the batch's, bit for bit. The
    negative control rebuilds the sample with a cap of 0 on the poisoned copy: the inputs must then
    differ (or the assembly must raise).
    """
    n = min(limit, len(batch.c_idx))
    for i in range(n):
        p = int(batch.inputs.primary[i])
        c = int(batch.c_idx[i])
        for value in (POISON_VALUE, float("nan")):
            copy = source.power_in.clone()
            copy[p, c - lay.NEWEST_VISIBLE_BEFORE_CONTRACT + 1 :] = value
            one = source.assemble(
                primary=batch.inputs.primary[i : i + 1], c_idx=batch.c_idx[i : i + 1],
                lead_day=lead_day[i : i + 1], ensemble=ensemble,
                ages=None if ages is None else ages[i : i + 1],
                age_exists=None if age_exists is None else age_exists[i : i + 1],
                power_in=copy,
            )  # fmt: skip
            for name in ("hist_power", "hist_visible", "row_exists"):
                assert torch.equal(getattr(one.inputs, name)[0], getattr(batch.inputs, name)[i]), (
                    f"sample {i}: {name} read a poisoned reading ({value})"
                )
        # negative control: with a cap of 0 the poison must reach the newest default row
        if ages is not None:
            continue
        copy = source.power_in.clone()
        copy[p, c - lay.NEWEST_VISIBLE_BEFORE_CONTRACT + 1 :] = POISON_VALUE
        try:
            loose = source.assemble(
                primary=batch.inputs.primary[i : i + 1], c_idx=batch.c_idx[i : i + 1],
                lead_day=lead_day[i : i + 1], ensemble=ensemble, cap_steps=0, power_in=copy,
            )  # fmt: skip
        except AssertionError:
            continue
        assert not torch.equal(loose.inputs.hist_power[0], batch.inputs.hist_power[i]), (
            f"sample {i}: the negative control did not change the inputs"
        )
    return f"KA5 (poisoned copies, 1e6 and NaN): {n} samples identical"

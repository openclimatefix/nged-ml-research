"""Experiment 718: the retrieval arrays of a fold, the target-weather providers, and the table
builder (WP1, WP5a).

`pool_arrays_from_source` reads the arrays of experiment 713's `Source` (power finite flags, the
7 weather columns of the ERA5 and CAMS array, and the day type), and the providers read the
target row's weather: ERA5 at the target half-hours for stage 1, and the ensemble-mean columns of
the origin day's 00Z run for stages 2 and 3 and prediction. Both are the arrays that the network
reads, converted to float64 before the distance. The standardised weather of experiment 713 is
stored as float16, and the design asks for float64 from the source rows before the cast: a
departure, recorded on the page, that leaves the builder and the independent recomputation of KR3
on one array.
"""  # noqa: D205

import e00718_layout as lay
import e00718_retrieval as ret
import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00713_batch import EnsTensors, Source  # ty: ignore[unresolved-import]


def pool_arrays_from_source(
    *, source: Source, with_calendar: bool, holidays: set | None = None
) -> ret.PoolArrays:
    """Return the retrieval arrays of the fold from experiment 713's source.

    `power_z` is NaN where the reading is flagged, missing, or hidden, so its finite flags are
    the eligibility rule's input. In the training phase the fold month's power is NaN, and the
    tables must be built in the prediction phase (design section 2.4).
    """
    finite = torch.isfinite(source.power_z).cpu().numpy().astype(np.int64)
    cumsum = np.concatenate(
        [np.zeros((finite.shape[0], 1), np.int64), finite.cumsum(axis=1)], axis=1
    )
    weather = source.dense_weather[..., : ret.N_WEATHER].double().cpu().numpy()
    day_type = source.day_type.cpu().numpy().astype(np.int64)
    day_class = None
    offsets, window, _ = ret.christmas_arrays(n_days=source.n_grid // lay.STEPS_PER_DAY)
    if with_calendar:
        import e00422_prep as prep422  # ty: ignore[unresolved-import]

        assert holidays is not None
        day_class = ret.day_class_for_grid(
            grid_start_us=prep422.GRID_START_US, n_grid=source.n_grid, holidays=holidays
        )
    return ret.PoolArrays(
        finite_cumsum=cumsum, weather=weather, day_type=day_type, day_class=day_class,
        christmas_offset=offsets, in_window=window,
    )  # fmt: skip


def era5_provider(*, arrays: ret.PoolArrays) -> ret.TargetProvider:
    """Return the stage 1 provider: the target row's weather is ERA5 and CAMS at the target
    half-hours, all filled.
    """  # noqa: D205

    def provide(primary: int, day: int) -> tuple[np.ndarray, np.ndarray]:
        c = int(lay.contract_idx_of_day(day=day))
        idx = np.minimum(c + np.arange(lay.N_LEAD_DAYS * lay.STEPS_PER_DAY), arrays.n_grid - 1)
        weather = arrays.weather[primary][idx].reshape(
            lay.N_LEAD_DAYS, lay.STEPS_PER_DAY, ret.N_WEATHER
        )
        return weather, np.ones((lay.N_LEAD_DAYS, lay.STEPS_PER_DAY), dtype=bool)

    return provide


def ens_provider(*, ens: EnsTensors) -> ret.TargetProvider:
    """Return the stage 2, stage 3, and prediction provider: the 7 ensemble-mean columns of the
    origin day's 00Z run.
    """  # noqa: D205
    weather = ens.weather[..., : ret.N_WEATHER]
    filled = ens.filled
    n_days = filled.shape[1]

    def provide(primary: int, day: int) -> tuple[np.ndarray, np.ndarray]:
        d = day - ens.first_day
        if not 0 <= d < n_days:
            return (
                np.zeros((lay.N_LEAD_DAYS, lay.STEPS_PER_DAY, ret.N_WEATHER)),
                np.zeros((lay.N_LEAD_DAYS, lay.STEPS_PER_DAY), dtype=bool),
            )
        w = (
            weather[primary, d]
            .double()
            .cpu()
            .numpy()
            .reshape(lay.N_LEAD_DAYS, lay.STEPS_PER_DAY, ret.N_WEATHER)
        )
        f = filled[primary, d].cpu().numpy().reshape(lay.N_LEAD_DAYS, lay.STEPS_PER_DAY)
        return w, f

    return provide

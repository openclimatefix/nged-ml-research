"""Primary demand research default 1, extended to 240 hours: stage 2 refitted for each month.

Reads each substation's stage 1 forecasts for every run (`e00377_forecast.py`), applies the stage 2
mean corrections of experiment 155's arm d (the winter offset and recipe R, both already monthly and
causal), then fits the spread for each scored month on the corrected residuals of earlier months
only: runs whose origin and valid time are both before the month's first instant, outside the
hand-filtered events. The spread has six lead bands, so that 240 to 336 hours has its own band.
D8 borrows the residuals of its six donors (the protocol's labelled exception) and, from fold
month 2025-08, also gets an own-residual arm (under 12 months of history until 2026-04). Output:
one file per substation and arm.

    E00377_SOURCE=control|mva E00377_ARM=a0 uv run --no-sync python \
        ml-experiments/00377-*/e00377s3_stack.py --sites D10
"""

import argparse
from datetime import UTC, datetime
from typing import Final

import e00377s2_common as common
import e00377s2_corrections as corr
import numpy as np
import polars as pl
from studies.dp_ens_weather import PROTOCOL_LEAD_BAND_UPPER_HOURS
from studies.dp_gam import ResidualSpread
from studies.eval_protocol.slices import fold_plan

LEAD_EDGES_HOURS: Final[tuple[float, ...]] = PROTOCOL_LEAD_BAND_UPPER_HOURS
"""Upper edges of the protocol's six lead bands, reaching 336 hours."""
N_BANDS: Final[int] = len(LEAD_EDGES_HOURS)
FIRST_FOLD_MONTH: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)
"""The first month with 12 months of ENS runs before it (ENS starts 2024-04-01)."""
LAST_FOLD_MONTH: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
D8_FIRST_OWN_MONTH: Final[datetime] = datetime(2025, 8, 1, tzinfo=UTC)
"""D8's stage 1 forecasts start with the 2025-07-01 origin (its series is masked before 2025-04-01
and the first fit needs 90 days), so its own residuals exist from this fold month."""
"""D8's history from 2025-04-01 reaches 12 months at this fold month."""
D8_DONOR_MONTHS_FROM: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
DONOR_ARM: Final[str] = "d_donor"
OWN_ARM: Final[str] = "d"
MIN_ROWS_PER_BAND: Final[int] = 200


def lead_band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the lead band index of each lead, the last band taking anything beyond 336 hours."""
    return np.minimum(
        np.searchsorted(np.array(LEAD_EDGES_HOURS), lead_hours, side="left"), N_BANDS - 1
    )


def fit_spread(*, residual: pl.DataFrame) -> list[ResidualSpread]:
    """Fit one centred residual table per lead band.

    Args:
        residual: Columns `resid`, `hour_band`, `quarter`, `lead_hours`.

    Returns:
        The six spreads.

    Raises:
        ValueError: If a lead band holds fewer than `MIN_ROWS_PER_BAND` residuals.
    """
    lead = lead_band(lead_hours=residual["lead_hours"].to_numpy())
    out = []
    for index in range(N_BANDS):
        chosen = lead == index
        if chosen.sum() < MIN_ROWS_PER_BAND:
            msg = (
                f"lead band {index} holds {int(chosen.sum())} residuals, under {MIN_ROWS_PER_BAND}"
            )
            raise ValueError(msg)
        out.append(
            ResidualSpread.fit(
                residuals=residual["resid"].to_numpy()[chosen],
                band=residual["hour_band"].to_numpy().astype(np.int64)[chosen],
                quarter=residual["quarter"].to_numpy().astype(np.int64)[chosen],
                centre=True,
            )
        )
    return out


def corrected_frame(*, site: str) -> pl.DataFrame:
    """Return stage 1's frame at a site with arm d's corrected mean in `mean` and the residual."""
    base = corr.add_calendar(frame=common.read_base(site=site))
    mean_a = base["mean"].to_numpy().astype(np.float64)
    offset, _ = corr.winter_offset_shift(frame=base, variant="winter")
    shift_d = offset + corr.recipe_r_shift(frame=base, forecast=mean_a + offset)
    mean_d = mean_a + shift_d
    return base.with_columns(
        pl.Series("mean", mean_d.astype(np.float32)),
        resid=pl.col("actual").cast(pl.Float64) - pl.Series(mean_d),
    )


def residuals_before(*, frames: list[pl.DataFrame], cut: datetime) -> pl.DataFrame:
    """Return the usable residual rows of earlier runs: origin and valid time before `cut`."""
    return pl.concat(
        [
            f.filter(
                pl.col("power_fcst_init_time") < cut,
                pl.col("valid_time") < cut,
                ~pl.col("switching_mask"),
                pl.col("resid").is_finite(),
            ).select("resid", "hour_band", "quarter", "lead_hours")
            for f in frames
        ]
    )


def with_fold_spread(
    *, frame: pl.DataFrame, pools: list[pl.DataFrame], months: list[datetime], arm: str
) -> pl.DataFrame:
    """Return the quantiles of each month's runs, from a spread fitted on earlier months only."""
    pieces = []
    for month in months:
        rows = frame.filter(
            pl.col("power_fcst_init_time") >= month,
            pl.col("power_fcst_init_time")
            < month.replace(year=month.year + (month.month == 12), month=month.month % 12 + 1),
        )
        if rows.is_empty():
            continue
        spreads = fit_spread(residual=residuals_before(frames=pools, cut=month))
        mean = rows["mean"].to_numpy().astype(np.float64)
        band = rows["hour_band"].to_numpy().astype(np.int64)
        quarter = rows["quarter"].to_numpy().astype(np.int64)
        lead = lead_band(lead_hours=rows["lead_hours"].to_numpy())
        quantiles = np.empty((rows.height, len(common.QUANTILES)))
        for index in range(N_BANDS):
            chosen = lead == index
            if chosen.any():
                table = spreads[index].table[band[chosen], quarter[chosen]]
                quantiles[chosen] = mean[chosen, None] + table
        pieces.append(
            rows.with_columns(
                *[
                    pl.Series(name, quantiles[:, position].astype(np.float32))
                    for position, name in enumerate(common.QUANTILES)
                ],
                arm=pl.lit(arm),
            )
        )
    return pl.concat(pieces)


def eligible_months(
    *, first_reading: datetime, first_month: datetime = FIRST_FOLD_MONTH
) -> list[datetime]:
    """Return the fold months with at least 12 months of power history and of ENS runs."""
    months = []
    month = first_month
    while month <= LAST_FOLD_MONTH:
        months.append(month)
        month = month.replace(year=month.year + (month.month == 12), month=month.month % 12 + 1)
    plan = fold_plan(first_reading=first_reading, scored_months=months)
    return plan.filter(pl.col("enough_history"))["scored_month"].to_list()


def write(*, frame: pl.DataFrame, site: str, arm: str) -> None:
    """Write one arm's rows atomically."""
    target = common.OUT_DIR / f"stack_{arm}_{site}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    frame.select([*common.BASE_COLUMNS, "arm"]).write_parquet(temporary)
    temporary.rename(target)


def main() -> None:
    """Build the stack's quantiles for the named sites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--first-readings", required=True, help="site=ISO,site=ISO")
    args = parser.parse_args()
    first = {
        pair.split("=")[0]: datetime.fromisoformat(pair.split("=")[1]).replace(tzinfo=UTC)
        for pair in args.first_readings.split(",")
    }
    for site in args.sites:
        frame = corrected_frame(site=site)
        if site == "D8":
            donors = [corrected_frame(site=donor) for donor in common.DONORS]
            donor_months = [
                m for m in eligible_months(first_reading=first["D2"]) if m >= D8_DONOR_MONTHS_FROM
            ]
            write(
                frame=with_fold_spread(
                    frame=frame, pools=donors, months=donor_months, arm=DONOR_ARM
                ),
                site=site,
                arm=DONOR_ARM,
            )
            own = eligible_months(first_reading=first["D2"], first_month=D8_FIRST_OWN_MONTH)
            write(
                frame=with_fold_spread(frame=frame, pools=[frame], months=own, arm=OWN_ARM),
                site=site,
                arm=OWN_ARM,
            )
            continue
        months = eligible_months(first_reading=first[site])
        write(
            frame=with_fold_spread(frame=frame, pools=[frame], months=months, arm=OWN_ARM),
            site=site,
            arm=OWN_ARM,
        )
        print(f"{site}: {len(months)} months, first {months[0].date()}")


if __name__ == "__main__":
    main()

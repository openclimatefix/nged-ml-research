# ruff: noqa: D103, E501
"""Experiment 647, step 2: R0 (K1 shifted by its own recent error) as code, and the reproduction assert.

R0 shifts all 11 quantiles of K1 by `resid * exp(-lead / tau)`, where `resid` is the mean of observed
power minus K1's day-ahead median (the forecast of each valid time with a lead of 24 to below 48 h)
over the valid UTC days `origin_day - 8` to `origin_day - 2`. The newest reading is then 33 h old at
the 09:00 UTC origin. Power is a share of capacity. The assert compares the result on the stored
export with the design answer's table within 0.01 points of capacity; the script writes its tables
before it asserts, so a failure still leaves the evidence.

Usage: python e00647_r0.py
"""

import itertools

import e00647_common as c647  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

Q = list(c647.QNAMES)


def daily_residual(
    *, df: pl.DataFrame, lead_lo: float = c647.R0_LEAD_H[0], lead_hi: float = c647.R0_LEAD_H[1],
    obs: str = "observed_power", how: str = "mean",
) -> pl.DataFrame:  # fmt: skip
    """Return (time_series_id, day, r): the daily mean of `obs` minus the day-ahead median, per valid UTC day.

    `df` holds the forecasts (columns `lead_hours`, `valid_time`, `q50`, `obs`); rows without an
    observation are dropped.
    """
    da = df.filter(
        (pl.col("lead_hours") >= lead_lo)
        & (pl.col("lead_hours") < lead_hi)
        & pl.col(obs).is_not_null()
    ).with_columns(r=pl.col(obs) - pl.col("q50"), day=pl.col("valid_time").dt.truncate("1d"))
    g = da.group_by("time_series_id", "day")
    return g.agg(r=pl.col("r").mean() if how == "mean" else pl.col("r").median())


def origin_residual(
    *, origins: pl.DataFrame, daily: pl.DataFrame, first_day: int = c647.R0_FIRST_DAY,
    last_day: int = c647.R0_LAST_DAY, min_days: int = 1,
) -> pl.DataFrame:  # fmt: skip
    """Return (time_series_id, od, res, n_days) for the mean daily residual of each origin.

    The mean runs over the days `od - last_day` to `od - first_day`.
    `origins` holds the distinct (time_series_id, od) pairs. An origin with fewer than `min_days`
    daily residuals gets no row, so R0 cannot be formed there.
    """
    parts = [
        origins.with_columns(day=pl.col("od") - pl.duration(days=k)).join(
            daily, on=["time_series_id", "day"], how="inner"
        )
        for k in range(first_day, last_day + 1)
    ]
    return (
        pl.concat(parts)
        .group_by("time_series_id", "od")
        .agg(res=pl.col("r").mean(), n_days=pl.len())
        .filter(pl.col("n_days") >= min_days)
    )


def shifted(*, q: np.ndarray, res: np.ndarray, lead: np.ndarray, tau_h: float) -> np.ndarray:
    """Return `q` (rows x 11) shifted by `res * exp(-lead / tau_h)`."""
    return q + (res * np.exp(-lead / tau_h))[:, None]


def score_table(*, x: pl.DataFrame, zero_fill: bool = False) -> dict[str, float]:
    """Return CRPS (points of capacity) of K1 and R0 on the event-free and verified rows.

    Rows without a residual are dropped, or with `zero_fill` kept unshifted (R0 equals K1 there).
    """
    out: dict[str, float] = {}
    if zero_fill:
        x = x.with_columns(pl.col("res").fill_null(0.0))
    for tag, flt in (("ef", pl.col("event_free")), ("ver", pl.col("in_verified_event"))):
        s = x.filter(flt & pl.col("observed_power").is_not_null() & pl.col("res").is_not_null())
        y, q, res, lead = (
            s["observed_power"].to_numpy(), s.select(Q).to_numpy(), s["res"].to_numpy(),
            s["lead_hours"].to_numpy(),
        )  # fmt: skip
        out[f"{tag}_rows"] = float(s.height)
        out[f"{tag}_K1"] = float(c647.crps_points(y=y, q=q).mean())
        for tau in c647.TAU_HOURS:
            out[f"{tag}_{tau}"] = float(
                c647.crps_points(y=y, q=shifted(q=q, res=res, lead=lead, tau_h=tau)).mean()
            )
    return out


def main() -> None:
    c647.WORK.mkdir(parents=True, exist_ok=True)
    df = c647.load_export(network="K1")
    origins = df.select("time_series_id", "od").unique()
    scored = df.filter(pl.col("lead_hours") >= 24)
    n_ef_all = scored.filter(pl.col("event_free") & pl.col("observed_power").is_not_null()).height
    n_ver_all = scored.filter(
        pl.col("in_verified_event") & pl.col("observed_power").is_not_null()
    ).height

    def run(
        *,
        lo: float,
        hi: float,
        d0: int,
        d1: int,
        how: str = "mean",
        min_days: int = 1,
        zero_fill: bool = False,
    ) -> dict[str, float]:
        res = origin_residual(origins=origins, daily=daily_residual(df=df, lead_lo=lo, lead_hi=hi, how=how),
                              first_day=d0, last_day=d1, min_days=min_days)  # fmt: skip
        return score_table(
            x=scored.join(res, on=["time_series_id", "od"], how="left"), zero_fill=zero_fill
        )

    zero_fill = {"zero_fill": True}
    main_spec = run(
        lo=c647.R0_LEAD_H[0],
        hi=c647.R0_LEAD_H[1],
        d0=c647.R0_FIRST_DAY,
        d1=c647.R0_LAST_DAY,
        **zero_fill,  # ty: ignore[invalid-argument-type]
    )
    rows = [
        {
            "spec": "main (days 2-8, lead 24-48 h, mean, no shift where R0 cannot be formed)",
            **main_spec,
        }
    ]
    for lo_hi, (d0, d1) in itertools.product(
        ((0, 24), (24, 48), (24, 240)), ((1, 7), (2, 8), (2, 7), (1, 8))
    ):
        rows.append({"spec": f"zero-fill: days {d0}-{d1}, lead {lo_hi[0]}-{lo_hi[1]} h", **run(lo=lo_hi[0], hi=lo_hi[1], d0=d0, d1=d1, **zero_fill)})  # ty: ignore[invalid-argument-type]  # fmt: skip
    rows.append({"spec": "main, rows without R0 dropped", **run(lo=24, hi=48, d0=2, d1=8)})
    for lo_hi, (d0, d1), how in itertools.product(
        ((0, 24), (24, 48), (24, 240)), ((1, 7), (2, 8), (2, 7), (3, 9), (1, 8)), ("mean",)
    ):
        rows.append({"spec": f"variant: days {d0}-{d1}, lead {lo_hi[0]}-{lo_hi[1]} h, {how}", **run(lo=lo_hi[0], hi=lo_hi[1], d0=d0, d1=d1, how=how)})  # fmt: skip
    rows.append(
        {
            "spec": "variant: days 2-8, lead 24-48 h, median",
            **run(lo=24, hi=48, d0=2, d1=8, how="median"),
        }
    )
    rows.append({"spec": "variant: days 2-8, lead 24-48 h, all 7 days needed", **run(lo=24, hi=48, d0=2, d1=8, min_days=7)})  # fmt: skip
    tab = pl.DataFrame(rows)
    keys = list(c647.DESIGN_TABLE)
    tab = tab.with_columns(
        max_abs_diff_vs_design=pl.max_horizontal(
            *[(pl.col(k) - v).abs() for k, v in c647.DESIGN_TABLE.items()]
        )
    )
    tab.write_csv(c647.WORK / "r0_reproduction.csv")
    with pl.Config(tbl_rows=40, tbl_cols=14, fmt_str_lengths=60):
        print(tab.select("spec", *keys, "max_abs_diff_vs_design"))
    print(f"scored rows with an observation: event-free {n_ef_all}, verified {n_ver_all}")  # fmt: skip
    print("design answer:", c647.DESIGN_TABLE)
    unformable = (
        scored.join(
            origin_residual(origins=origins, daily=daily_residual(df=df)),
            on=["time_series_id", "od"],
            how="anti",
        )
        .filter(pl.col("event_free") & pl.col("observed_power").is_not_null())
        .height
    )
    print(
        f"event-free rows whose origin has no daily residual (R0 = K1 there): {unformable} of {n_ef_all}"
    )
    for k, v in c647.DESIGN_TABLE.items():
        print(f"  {k}: ours {main_spec[k]:.3f} design {v:.3f} diff {main_spec[k] - v:+.3f} {'ok' if abs(main_spec[k] - v) <= c647.DESIGN_TOLERANCE else 'FAIL'}")  # fmt: skip
    bad = {
        k: (main_spec[k], v)
        for k, v in c647.DESIGN_TABLE.items()
        if abs(main_spec[k] - v) > c647.DESIGN_TOLERANCE
    }
    best = tab.sort("max_abs_diff_vs_design").row(0, named=True)
    print(f"closest variant: {best['spec']} (max diff {best['max_abs_diff_vs_design']:.3f})")
    assert not bad, (
        f"R0 does not reproduce the design answer's table within {c647.DESIGN_TOLERANCE}: {bad}"
    )
    print("R0 REPRODUCTION PASS")


if __name__ == "__main__":
    main()

# ruff: noqa: B007, DTZ001, E501, D103
"""Experiment 614: the 14-days-either-side example figure of the December 2025 D5/D16 load-shift.

Reads the exported K1 and K3 files only. Day-ahead means the forecast issued at 09:00 UTC the day
before the valid day. One forecast network per panel, two panels per figure (D5 and D16), observed power in
black, the gate in a thin strip under the panel from the stored `gate_on` column, dashed lines at the
maintainer's switch times, and the verified window of experiment 580 shaded. Coverage is scored
against an estimate of the normal running arrangement, which these panels do not draw.

Usage: python e00614_figures.py OUTDIR
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00588_clean_panels as cp  # ty: ignore[unresolved-import]
import e00614_export as ex
import numpy as np
import polars as pl

SWITCH = [datetime(2025, 12, 30, 1, 0), datetime(2026, 1, 14, 11, 30)]
IQ = {0.05: 0, 0.5: 1, 0.95: 2}
PRIMARIES = ("D5", "D16")


def day_ahead(*, network: str, primary: str, lo: datetime, hi: datetime) -> pl.DataFrame:
    sub = pl.read_csv(ex.OUT / "substations.csv")
    frames = [pl.read_parquet(ex.OUT / f"{network}_{m}.parquet") for m in ("2025-12", "2026-01")]
    return (
        pl.concat(frames)
        .join(sub.select("time_series_id", "primary_name", "capacity"), on="time_series_id")
        .filter(
            (pl.col("primary_name") == primary)
            & (pl.col("valid_time") >= lo.replace(tzinfo=UTC))
            & (pl.col("valid_time") < hi.replace(tzinfo=UTC))
            & (pl.col("power_fcst_init_time").dt.hour() == 9)
            & (
                pl.col("power_fcst_init_time").dt.date()
                == (pl.col("valid_time").dt.date() - pl.duration(days=1))
            )
        )
        .sort("valid_time")
    )


def step_check(*, primary: str) -> str:
    """Return which window matches the observed step: mean power in the 48 half-hours either side of each date."""
    frames = pl.concat(
        [pl.read_parquet(ex.OUT / f"K1_{m}.parquet") for m in ("2025-12", "2026-01")]
    )
    sub = pl.read_csv(ex.OUT / "substations.csv")
    tsid = sub.filter(pl.col("primary_name") == primary)["time_series_id"][0]
    obs = (
        frames.filter(pl.col("time_series_id") == tsid)
        .unique("valid_time")
        .sort("valid_time")
        .select("valid_time", "observed_power")
    )
    out = []
    for name, t in (
        ("2025-12-29 00:00 (verified window start)", datetime(2025, 12, 29)),
        ("2025-12-30 01:00 (given start)", datetime(2025, 12, 30, 1)),
    ):
        before = obs.filter(
            (pl.col("valid_time") >= t.replace(tzinfo=UTC) - timedelta(hours=24))
            & (pl.col("valid_time") < t.replace(tzinfo=UTC))
        )["observed_power"].mean()
        after = obs.filter(
            (pl.col("valid_time") >= t.replace(tzinfo=UTC))
            & (pl.col("valid_time") < t.replace(tzinfo=UTC) + timedelta(hours=24))
        )["observed_power"].mean()
        out.append(
            f"{primary}: mean power {before:.2f} in the day before {name}, {after:.2f} in the day after"
        )
    return "; ".join(out)


def figure(*, network: str, title_label: str, out: Path) -> None:
    lo, hi = SWITCH[0] - timedelta(days=14), SWITCH[1] + timedelta(days=14)
    cp.SHADE = cp.entry("Verified window (exp. 580)", "band", cp.SHADE_COLOUR, 0.7)
    units, ydoms = [], []
    data = {p: day_ahead(network=network, primary=p, lo=lo, hi=hi) for p in PRIMARIES}
    for p, df in data.items():
        cap = df["capacity"]
        q = np.stack([(df[c] / cap).to_numpy() for c in ("q05", "q50", "q95")], axis=1)
        obs = (df["observed_power"] / cap).to_numpy()
        ydoms.append(cp.common_ydom([(q, obs)], iq=IQ))
    for i, (p, df) in enumerate(data.items()):
        cap = df["capacity"]
        q = np.stack([(df[c] / cap).to_numpy() for c in ("q05", "q50", "q95")], axis=1)
        obs = (df["observed_power"] / cap).to_numpy()
        x = [t.replace(tzinfo=None) for t in df["valid_time"].to_list()]
        strip = None
        if network == "K1":
            on = df["gate_on"].to_numpy()
            strip = [{"x0": x[j], "x1": x[min(j + 1, len(x) - 1)]} for j in np.flatnonzero(on)]
            print(p, "gate on half-hours:", int(on.sum()), "of", len(on), flush=True)
        units.append(
            cp.arm_panel(
                x=x, q=q, iq=IQ, obs=obs, xdom=(lo, hi), title=f"{p}: {title_label}", ydom=ydoms[i],
                xtype="T", xtitle="Date (UTC)" if i == 1 else None, strip=strip or None,
                span=(datetime(2025, 12, 29, 1), datetime(2026, 1, 13, 11)), rules=SWITCH,
            )
        )  # fmt: skip
    cp.RULES = cp.entry("Maintainer's switch times", "rule", cp.RULE_COLOUR)
    key = [*cp.FORECAST_ENTRIES, cp.OBS, cp.SHADE, cp.RULES]
    if network == "K1":
        key.append(cp.GATE)
    chart = cp.figure(
        units=units,
        title="Day-ahead forecasts through the December 2025 load-shift",
        key=key,
    )
    chart.save(str(out))
    if out.suffix == ".svg":
        chart.save(str(out.with_suffix(".png")), scale_factor=2)


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cp.PANEL_HEIGHT = 300
    for p in PRIMARIES:
        print(step_check(primary=p), flush=True)
    figure(
        network="K1",
        title_label="K1, lag neural network with the gated band",
        out=outdir / "example_K1_2025-12.svg",
    )
    figure(network="K3", title_label="K3, lag-free neural network", out=outdir / "example_K3_2025-12.svg")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

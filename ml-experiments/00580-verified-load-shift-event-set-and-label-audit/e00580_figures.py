# ruff: noqa: ANN001, D103, D205, E501
"""Experiment 580, step 4: figures. Hand-check plots for the long labelled events and the telemetry transfers,
and event 146 / event 55 before and after correction against the day-ahead forecast.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00578_common as k  # ty: ignore[unresolved-import]
import e00578_follow2 as f2  # ty: ignore[unresolved-import]
import e00580_verify as v  # ty: ignore[unresolved-import]
import matplotlib as mpl
import matplotlib.dates
import matplotlib.ticker
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

mpl.use("svg")
import matplotlib.pyplot as plt

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00580-verified-load-shift-event-set-and-label-audit"
)
START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
DAY = 48
plt.rcParams.update(
    {"font.size": 7, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False}
)


def when(i: float) -> datetime:
    return START + timedelta(minutes=30 * float(i))


def daily(x: np.ndarray, lo: int, hi: int) -> tuple[list[datetime], np.ndarray]:
    seg = x[lo:hi]
    n = len(seg) // DAY
    m = np.nanmean(seg[: n * DAY].reshape(n, DAY), axis=1)
    return [when(lo + DAY * i + DAY / 2) for i in range(n)], m


def tidy(ax) -> None:
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d %b %y"))
    ax.tick_params(axis="x", labelsize=6)


def mark(ax, *, on: float, en: float, lab_on: int | None, lab_en: int | None) -> None:
    if lab_on is not None:
        ax.axvspan(
            when(lab_on),
            when(lab_en + 1),  # ty: ignore[unsupported-operator]
            color=theme.BRAND_ORANGE,
            alpha=0.15,
            lw=0,
        )
    ax.axvline(when(on), color=theme.DATA_BLUE, lw=1)
    ax.axvline(when(en), color=theme.DATA_BLUE, lw=1, ls="--")


def long_events(
    *, st: pl.DataFrame, x: np.ndarray, labels: list[str], name: str, ids: list[int]
) -> None:
    ncol = 2
    nrow = (len(ids) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, 1.55 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, eid in zip(axes.flat, ids, strict=False):
        ax.set_visible(True)
        r = st.filter(pl.col("event_id") == eid).row(0, named=True)
        pad = 25 * DAY
        lo, hi = max(r["label_onset_idx"] - pad, 0), min(r["label_end_idx"] + pad, x.shape[1])
        t, m = daily(x[labels.index(r["label"])], lo, hi)
        ax.plot(t, m, color=theme.BLACK_1, lw=0.8)
        mark(
            ax,
            on=r["fit_onset_idx"],
            en=r["fit_end_idx"],
            lab_on=r["label_onset_idx"],
            lab_en=r["label_end_idx"],
        )
        cls = r["final_class"].split(" (")[0].replace("unpaired step", "one-Primary step")
        ax.set_title(
            f"Event {eid}, {r['label']}: label {r['label_depth_mw']:+.2f} MW, fit {r['fit_depth_mw']:+.2f} MW; {cls}",
            fontsize=6.5,
            loc="left",
        )
        tidy(ax)
        ax.set_ylabel("share of capacity", fontsize=6)
    fig.suptitle(
        "Daily mean power (black). Orange: labelled span. Blue lines: fitted onset (solid) and end (dashed) step.",
        fontsize=7,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def transfers(
    *, tr: pl.DataFrame, x: np.ndarray, labels: list[str], name: str, ids: list[int]
) -> None:
    nrow = (len(ids) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(7.1, 1.55 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, tid in zip(axes.flat, ids, strict=False):
        ax.set_visible(True)
        r = tr.filter(pl.col("transfer_id") == tid).row(0, named=True)
        pad = 20 * DAY
        lo, hi = max(r["onset_a_idx"] - pad, 0), min(r["end_a_idx"] + pad, x.shape[1])
        for lab, colour in (
            (r["primary_a"], theme.BRAND_ORANGE),
            (r["primary_b"], theme.DATA_BLUE),
        ):
            t, m = daily(x[labels.index(lab)], lo, hi)
            ax.plot(t, m - np.nanmedian(m), color=colour, lw=0.8, label=lab)
        ax.axvline(when(r["onset_a_idx"]), color="grey", lw=0.8)
        ax.axvline(when(r["end_a_idx"]), color="grey", lw=0.8, ls="--")
        ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=2)
        ax.set_title(
            f"{r['primary_a']} {r['depth_a_mw']:+.2f} MW, {r['primary_b']} {r['depth_b_mw']:+.2f} MW, {r['length_days']:.0f} days; {r['source']}",
            fontsize=6.5,
            loc="left",
        )
        ax.set_ylabel("share of capacity,\nminus median", fontsize=6)
        tidy(ax)
    fig.suptitle(
        "Daily mean power at both Primaries of each telemetry transfer; grey lines: fitted onset (solid) and end (dashed).",
        fontsize=7,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def before_after(
    *, st: pl.DataFrame, tr: pl.DataFrame, x: np.ndarray, labels: list[str], dayahead: np.ndarray
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.2))
    for ax, (eid, tid, lab) in zip(axes, ((146, 23, "D5"), (55, 23, "D16")), strict=True):
        r = st.filter(pl.col("event_id") == eid).row(0, named=True)
        m = tr.filter(pl.col("transfer_id") == tid).row(0, named=True)
        p = labels.index(lab)
        cap = r["capacity_mw"]
        lo, hi = r["label_onset_idx"] - 10 * DAY, r["label_end_idx"] + 10 * DAY
        obs = x[p].copy()
        by_label = v.correct(
            observed=obs,
            depth=r["label_depth_mw"] / cap,
            onset=r["label_onset_idx"],
            end=r["label_end_idx"] + 1,
        )
        fit = (m["depth_a_mw"] if m["primary_a"] == lab else m["depth_b_mw"]) / cap
        by_tel = v.correct(observed=obs, depth=fit, onset=m["onset_a_idx"], end=m["end_a_idx"])
        for series, colour, text, lw in (
            (obs, theme.BRAND_ORANGE, "Observed", 1.0),
            (by_label, theme.DATA_BLUE, "Corrected with the label", 0.8),
            (by_tel, theme.DATA_GREEN, "Corrected with the telemetry fit", 0.8),
            (
                dayahead[p],
                theme.DATA_PURPLE,
                "Day-ahead forecast (v3+D5 median, lead 24 to 48 h)",
                0.7,
            ),
        ):
            t, y = daily(series, lo, hi)
            ax.plot(t, y, color=colour, lw=lw, label=text)
        ax.axvspan(
            when(r["label_onset_idx"]),
            when(r["label_end_idx"] + 1),
            color=theme.BRAND_ORANGE,
            alpha=0.12,
            lw=0,
        )
        ax.axvspan(
            when(m["onset_a_idx"]), when(m["end_a_idx"]), color=theme.DATA_GREEN, alpha=0.12, lw=0
        )
        ax.set_title(
            f"{lab}, event {eid}: label {r['label_depth_mw']:+.2f} MW from {when(r['label_onset_idx']):%d %b %Y} (orange band); telemetry fit {fit * cap:+.2f} MW from {when(m['onset_a_idx']):%d %b %Y} (green band)",
            fontsize=7,
            loc="left",
        )
        ax.set_ylabel("daily mean, share of capacity")
        ax.legend(fontsize=6, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "before_after_events_146_and_55.svg")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev, labels, x, _, _, _, _ = v.load_series()
    del ev
    st = pl.read_parquet(v.WORK / "events_labelled_status.parquet")
    tr = pl.read_parquet(v.WORK / "events_verified.parquet")
    long_ids = st.filter(pl.col("label_length_days") >= 7).sort("event_id")["event_id"].to_list()
    long_events(st=st, x=x, labels=labels, name="long_events_hand_check_1", ids=long_ids[:12])
    long_events(st=st, x=x, labels=labels, name="long_events_hand_check_2", ids=long_ids[12:])
    tids = tr.filter(pl.col("in_label_period"))["transfer_id"].to_list()
    transfers(tr=tr, x=x, labels=labels, name="telemetry_transfers_in_label_period", ids=tids)
    outside = tr.filter(~pl.col("in_label_period"))["transfer_id"].to_list()
    transfers(tr=tr, x=x, labels=labels, name="telemetry_transfers_before_labels", ids=outside)
    # day-ahead forecast (lead 24 to 48 h)
    sm = k.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    allm = (*k.MONTHS, *f2.NEW)
    fr = (
        f2.load(root=f2._Root(), tag="V3D5", months=allm)
        .with_columns(
            lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
            prim=pl.col("time_series_id").replace_strict(prim_of),
            idx=(
                (pl.col("valid_time").dt.epoch("us") - k.prep.GRID_START_US) // k.prep.STEP_US
            ).cast(pl.Int64),
        )
        .filter((pl.col("lead") >= 24) & (pl.col("lead") < 48))
    )
    grid = np.full(x.shape, np.nan, np.float32)
    g = fr.group_by(["prim", "idx"]).agg(pl.col("q50").mean())
    # `prim` holds the Primary index in the site map's numbering
    p, i, q = g["prim"].to_numpy(), g["idx"].to_numpy(), g["q50"].to_numpy()
    grid[p, i] = q
    before_after(st=st, tr=tr, x=x, labels=labels, dayahead=grid)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()

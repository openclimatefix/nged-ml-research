"""Real-record tables, the four scenarios, the discriminator, and the figures of 658.

    DATA_PATH_INTERNAL=... PYTHONPATH=<folder of experiment 653>:. \
        uv run --no-sync python e00658_report_real.py \
        <real|scenarios|discriminator|examples> <prefix>

`prefix` is `main` for the first look and `v2main` for the second look.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from e00653_common import (
    DAY,
    E580,
    GRID_LENGTH,
    GRID_START,
    SCORED_FROM,
    SERIES_IDS,
    Panel,
    block_means,
)
from e00653_inject import OFFSET_2019
from e00653_slow import fit_slow, to_half_hours
from e00658_report import YEARS, picked
from e00658_train import WORK, setup
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00658-edge-state-event-table-neural-network"
)
BLUE, ORANGE, GREY = "#1f4e9c", "#e8730c", "#888888"
E653: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00653-edge-state-event-table-penalised-segmentation"


def ts(i: int) -> str:
    """UTC time of a half-hour index."""
    return (GRID_START + timedelta(minutes=30 * int(i))).strftime("%Y-%m-%d %H:%M")


def labelled(panel: Panel) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Logged shifts of more than 1 day in the scored years, and LR_b20's alarms."""
    lab = pl.read_parquet(E580 / "events_labelled_status.parquet").filter(
        (pl.col("label_length_days") > 1)
        & pl.col("label").is_in(panel.sites)
        & (pl.col("label_onset_idx") - OFFSET_2019 >= SCORED_FROM)
    )
    names = dict(zip(SERIES_IDS, panel.sites, strict=True))
    lr = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00254-lsc-alarm-precision-against-full-log-with-chance-base-rate/alarms_LR_b20.parquet"
    )
    lr = (
        lr.with_columns(site=pl.col("series").replace_strict(names, default=None))
        .filter(pl.col("onset") >= SCORED_FROM)
        .unique(subset=["site", "onset"])
    )
    return lab, lr


def found(lab: pl.DataFrame, starts: dict[str, list[int]]) -> tuple[int, int]:
    """Logged shifts matched strictly (4 half-hours) and loosely (48)."""
    strict = loose = 0
    for r in lab.iter_rows(named=True):
        on = r["label_onset_idx"] - OFFSET_2019
        s = starts.get(r["label"], [])
        strict += any(abs(x - on) <= 4 for x in s)
        loose += any(abs(x - on) <= 48 for x in s)
    return strict, loose


def real(prefix: str) -> None:
    """Events per Primary-year, logged shifts, the rewired-graph placebo, the verified pair."""
    panel = Panel()
    n_prim = len(panel.sites)
    lab, lr = labelled(panel)
    print(
        f"logged shifts of more than 1 day in scored years: {lab.height}; "
        f"LR_b20 alarms {lr.height} "
        f"({lr.height / (n_prim * YEARS):.2f} per Primary-year)"
    )
    base = {p: lr.filter(pl.col("site") == p)["onset"].to_list() for p in panel.sites}
    print("LR_b20 strict/loose", found(lab, base))
    e653 = pl.read_parquet(E653 / "real_events_masked.parquet").filter(
        pl.col("start") >= SCORED_FROM
    )
    print(
        "experiment 653:",
        e653.height,
        f"{e653.height / (n_prim * YEARS):.2f} per Primary-year",
        found(
            lab, {p: e653.filter(pl.col("primary") == p)["start"].to_list() for p in panel.sites}
        ),
        "edge events",
        e653.filter(pl.col("cls") == "edge_transfer").height,
    )
    for s in (0, 1, 2):
        tag = f"{prefix}_seed{s}"
        ev = pl.read_parquet(WORK / f"real_events_{tag}.parquet").filter(
            pl.col("start") >= SCORED_FROM
        )
        mine = {p: ev.filter(pl.col("primary") == p)["start"].to_list() for p in panel.sites}
        per = (
            ev.group_by("primary")
            .len()
            .with_columns(rate=pl.col("len") / YEARS)
            .sort("rate", descending=True)
        )
        d8 = per.filter(pl.col("primary") == "D8")["rate"].to_list()
        rw = json.loads((WORK / f"rewired_{tag}.json").read_text())
        print(
            f"\n{tag} costs {picked(tag)}: {ev.height} events, "
            f"{ev.height / (n_prim * YEARS):.2f} per Primary-year, "
            f"strict/loose {found(lab, mine)}; classes {dict(ev.group_by('cls').len().iter_rows())}"
        )
        print(
            f"   median Primary rate {per['rate'].median():.2f}, D8 {d8}; "
            f"edge events real {rw['real']}, "
            f"rewired median {np.median(rw['rewired'])} "
            f"(5-95% {np.percentile(rw['rewired'], 5)}-{np.percentile(rw['rewired'], 95)}), "
            f"experiment 653 decoder on same rewires median {np.median(rw['rewired_e653pair'])}"
        )
        for r in rw["grid"]:
            rate = r["n_scored"] / (n_prim * YEARS)
            if abs(rate - 6.57) < 1.0:
                print("   grid point near the 653 rate:", r, round(rate, 2))
        print("   verified pair (D5, D16):")
        for site in ("D5", "D16"):
            for r in (
                pl.read_parquet(WORK / f"real_events_{tag}.parquet")
                .filter(
                    (pl.col("primary") == site)
                    & (pl.col("start") > 69900 - 400)
                    & (pl.col("start") < 70800)
                    & (pl.col("units") > 3)
                )
                .iter_rows(named=True)
            ):
                print(
                    f"      {site} {ts(r['start'])} -> {ts(r['end']) if r['end'] > 0 else 'open'} "
                    f"{r['amp_mw']:+.2f} MW {r['cls']} partners={r['partners']}"
                )
    ev = pl.read_parquet(WORK / f"real_events_{prefix}_seed0.parquet")
    allab = pl.read_parquet(E580 / "events_labelled_status.parquet")
    spans = [
        (r["label"], r["label_onset_idx"] - OFFSET_2019, r["label_end_idx"] - OFFSET_2019)
        for r in allab.iter_rows(named=True)
    ]
    ver = pl.read_parquet(E580 / "events_verified.parquet")
    for r in ver.iter_rows(named=True):
        spans += [
            (r["primary_a"], r["onset_a_idx"] - OFFSET_2019, r["end_a_idx"] - OFFSET_2019),
            (r["primary_b"], r["onset_b_idx"] - OFFSET_2019, r["end_b_idx"] - OFFSET_2019),
        ]
    free = 0
    for site in panel.sites:
        busy = np.zeros(GRID_LENGTH, bool)
        for s, a, b in spans:
            if s == site:
                busy[max(0, a - 14 * DAY) : max(0, b + 14 * DAY)] = True
        busy[:SCORED_FROM] = True
        free += int((~busy & np.isfinite(panel.residual[site])).sum())
    for name, e in (
        ("network seed 0", ev.filter(pl.col("start") >= SCORED_FROM)),
        ("experiment 653", e653),
    ):
        n = sum(
            not any(
                s == r["primary"]
                and r["start"] < b + 14 * DAY
                and (GRID_LENGTH if r["end"] < 0 else r["end"]) > a - 14 * DAY
                for s, a, b in spans
            )
            for r in e.iter_rows(named=True)
        )
        print(
            f"event-free stratum {name}: {n} events, "
            f"{n / (free / (DAY * 365.25)):.2f} per event-free Primary-year"
        )


def scenarios(prefix: str) -> None:
    """Normalised MAE of the median forecast in the four scenarios, seed 0."""
    panel, res, sigma = setup()
    ev = pl.read_parquet(WORK / f"real_events_{prefix}_seed0.parquet")
    rows = []
    for site in panel.sites:
        r = panel.residual[site]
        y = block_means(x=res[site])
        sl = to_half_hours(slow=fit_slow(y=y, sigma=sigma[site], arm="masked"))
        corr = r.copy()
        corr[: len(sl)] -= sl
        inside = np.zeros(GRID_LENGTH, bool)
        after = np.zeros(GRID_LENGTH, bool)
        near = np.zeros(GRID_LENGTH, bool)
        for e in ev.filter(pl.col("primary") == site).iter_rows(named=True):
            end = GRID_LENGTH if e["end"] < 0 else e["end"]
            inside[e["start"] : end] = True
            after[end : end + 7 * DAY] = True
            near[e["start"] : end + 7 * DAY] = True
            corr[e["start"] : end] -= e["amp_mw"]
        for name, m0 in {
            "all rows": np.ones(GRID_LENGTH, bool),
            "outside events, 7+ days after": ~near,
            "inside events": inside,
            "7 days after an event": after & ~inside,
        }.items():
            m = m0 & (np.arange(GRID_LENGTH) >= SCORED_FROM) & np.isfinite(r)
            rows.append(
                {
                    "scenario": name,
                    "n": int(m.sum()),
                    "before": float(np.abs(r[m]).sum() / panel.capacity[site]),
                    "after": float(np.abs(corr[m]).sum() / panel.capacity[site]),
                }
            )
    d = (
        pl.DataFrame(rows)
        .group_by("scenario", maintain_order=True)
        .agg(
            pl.col("n").sum(),
            (100 * pl.col("before").sum() / pl.col("n").sum()).alias("nMAE_before_pct"),
            (100 * pl.col("after").sum() / pl.col("n").sum()).alias("nMAE_after_table_pct"),
        )
    )
    print(d)


def _features(*, x: np.ndarray, t: int, sigma: float) -> list[float]:
    """Window features of one event onset: step size, 10-90% width, noise after, slope before."""
    from e00653_segment import step_width

    pre, post = x[max(0, t - 96) : t - 4], x[t + 4 : t + 100]
    step = (np.nanmedian(post) - np.nanmedian(pre)) / sigma
    w = step_width(xh=x, t=t)
    return [
        abs(step),
        0.0 if np.isnan(w) else w,
        float(np.nanstd(post) / sigma),
        float(np.nanstd(pre) / sigma),
    ]


def discriminator() -> None:
    """Can a logistic classifier tell injected edge events from real high-confidence events?"""
    from e00653_inject import GenParams
    from e00658_data import build_world
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    panel, res, sigma = setup()
    inj, real_rows = [], []
    for k in range(6):
        w = build_world(
            panel=panel,
            residual=res,
            sigma=sigma,
            rng=np.random.default_rng(1000 + 17 * k + 7),
            lo=SCORED_FROM,
            hi=GRID_LENGTH,
            gen=GenParams(),
            upto=None,
        )
        inj.extend(
            _features(x=w.residual[t.site], t=t.start, sigma=sigma[t.site])
            for t in w.truth
            if t.kind in ("edge", "permanent_transfer") and t.units >= 4 and t.amp_mw < 0
        )
    ver = pl.read_parquet(E580 / "events_verified.parquet")
    for r in ver.iter_rows(named=True):
        real_rows.extend(
            _features(x=res[p], t=on - OFFSET_2019, sigma=sigma[p])
            for p, on in ((r["primary_a"], r["onset_a_idx"]), (r["primary_b"], r["onset_b_idx"]))
            if p in sigma
        )
    e653 = pl.read_parquet(E653 / "real_events_masked.parquet").filter(
        (pl.col("cls") == "edge_transfer") & (pl.col("start") >= SCORED_FROM)
    )
    real_rows.extend(
        _features(x=res[r["primary"]], t=r["start"], sigma=sigma[r["primary"]])
        for r in e653.iter_rows(named=True)
    )
    xi, xr = np.array(inj), np.array(real_rows)
    xr = xr[np.isfinite(xr).all(axis=1)]
    xi = xi[np.isfinite(xi).all(axis=1)]
    print(f"injected edge events {len(xi)}, real high-confidence events {len(xr)}")
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(500):
        tr = rng.choice(len(xi), size=len(xi), replace=True)
        te_real = rng.choice(len(xr), size=len(xr), replace=True)
        x = np.vstack([xi[tr], xr[te_real]])
        y = np.r_[np.zeros(len(tr)), np.ones(len(te_real))]
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(
            (x - x.mean(0)) / (x.std(0) + 1e-9), y
        )
        oob = np.setdiff1d(np.arange(len(xi)), tr)
        z = np.vstack([xi[oob], xr])
        zz = (z - x.mean(0)) / (x.std(0) + 1e-9)
        aucs.append(
            roc_auc_score(np.r_[np.zeros(len(oob)), np.ones(len(xr))], clf.predict_proba(zz)[:, 1])
        )
    print(
        "AUC injected vs real (bootstrap, out-of-bag injected):",
        round(float(np.mean(aucs)), 3),
        "95% interval",
        np.round(np.percentile(aucs, [2.5, 97.5]), 3),
    )


def examples(prefix: str) -> None:
    """Event time series for D5 and D16 (seed 0), with observed power and the K1 baseline."""
    panel = Panel()
    ev = pl.read_parquet(WORK / f"real_events_{prefix}_seed0.parquet")
    d = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
    k1 = pl.concat(
        [pl.read_parquet(d / f"{f}.parquet") for f in ["K1_2025-12", "K1_2026-01"]]
    ).filter((pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36))
    names = dict(zip(panel.sites, SERIES_IDS, strict=True))
    lo, hi = "2025-12-15", "2026-01-21"
    lo_i = int((np.datetime64(lo) - np.datetime64("2022-01-01")) / np.timedelta64(30, "m"))
    hi_i = int((np.datetime64(hi) - np.datetime64("2022-01-01")) / np.timedelta64(30, "m"))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for ax, site in zip(axes, ("D5", "D16"), strict=True):
        t = np.arange(lo_i, hi_i)
        tt = np.datetime64("2022-01-01T00:00") + t.astype("timedelta64[30m]")
        ax.plot(tt, panel.power[site][t], color="k", lw=0.8, label="observed power")
        kk = k1.filter(pl.col("time_series_id") == names[site]).sort("valid_time")
        ax.plot(
            kk["valid_time"].dt.replace_time_zone(None).to_numpy(),
            kk["q50"].to_numpy(),
            color=ORANGE,
            lw=0.8,
            label="K1 day-ahead median",
        )
        for e in ev.filter(pl.col("primary") == site).iter_rows(named=True):
            if hi_i > e["start"] > lo_i - 2000 and e["units"] >= 3:
                end = hi_i if e["end"] < 0 else e["end"]
                a = np.datetime64("2022-01-01T00:00") + np.timedelta64(30 * e["start"], "m")
                b = np.datetime64("2022-01-01T00:00") + np.timedelta64(30 * min(end, hi_i), "m")
                ax.axvspan(a, b, color=BLUE, alpha=0.12)
                ax.text(
                    a,
                    ax.get_ylim()[1],
                    f" {e['amp_mw']:+.1f} MW {e['cls']}",
                    fontsize=6,
                    va="top",
                    color=BLUE,
                )
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_xlim(np.datetime64(lo), np.datetime64(hi))
        ax.set_ylabel(f"{site} power (MW)")
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / f"figure_event_series_D5_D16_{prefix}.svg")
    plt.close(fig)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "discriminator":
        discriminator()
    else:
        {"real": real, "scenarios": scenarios, "examples": examples}[stage](sys.argv[2])

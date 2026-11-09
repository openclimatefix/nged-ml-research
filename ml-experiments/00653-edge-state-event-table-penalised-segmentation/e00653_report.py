"""Print every number the page quotes and draw its figures.

DATA_PATH_INTERNAL=... PYTHONPATH=. uv run --no-sync python e00653_report.py
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Final, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from e00653_common import DAY, E580, GRID_LENGTH, GRID_START, SCORED_FROM, SERIES_IDS, WORK, Panel
from e00653_inject import OFFSET_2019
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00653-edge-state-event-table-penalised-segmentation"
)
BANDS: Final[list[str]] = ["3h-1d", "1-7d", "7-21d", "21-60d", "60d+", "permanent"]
YEARS: Final[float] = (GRID_LENGTH - SCORED_FROM) / (365.25 * DAY)
BLUE, ORANGE, GREY = "#1f4e9c", "#e8730c", "#888888"


def band_col(df: pl.DataFrame) -> pl.DataFrame:
    """Add the duration band of each truth row."""
    d = pl.col("days")
    return df.with_columns(
        band=pl.when(d < 0)
        .then(pl.lit("permanent"))
        .when(d < 1)
        .then(pl.lit("3h-1d"))
        .when(d < 7)
        .then(pl.lit("1-7d"))
        .when(d < 21)
        .then(pl.lit("7-21d"))
        .when(d < 60)
        .then(pl.lit("21-60d"))
        .otherwise(pl.lit("60d+"))
    )


def wrec(df: pl.DataFrame, col: str) -> float:
    """MW-weighted share with `col` true."""
    w = df["amp_mw"].abs()
    return (
        float((df[col].cast(pl.Float64) * w).sum()) / float(w.sum()) if df.height else float("nan")
    )


def injected() -> None:  # noqa: PLR0915 - one linear printout of many tables
    """Tables of the injected worlds."""
    t = band_col(pl.read_parquet(WORK / "injected_truth_scored.parquet"))
    pos = t.filter(pl.col("kind").is_in(["edge", "solo", "permanent_transfer", "permanent_shift"]))
    big = pos.filter((pl.col("units") >= 4) & ((pl.col("days") >= 1) | (pl.col("days") < 0)))
    print(
        "== P1: recall of injected events of 4+ noise units and 1+ day, by spec and slow-level arm"
    )
    for (spec, arm), g in big.group_by(["spec", "arm"], maintain_order=True):
        e = g.filter(pl.col("edge_truth"))
        print(
            f"{spec:14s} {arm:7s} n={g.height:4d} found MW-weighted={wrec(g, 'found'):.3f} "
            f"unweighted={g['found'].mean():.3f} | edge transfers n={e.height:3d} "
            "right-class MW-weighted="
            f"{wrec(e, 'right_class'):.3f} "
            f"unweighted={e['right_class'].mean() if e.height else float('nan'):.3f}"
        )
    d = big.filter((pl.col("spec") == "default") & (pl.col("arm") == "masked"))
    print(
        "\n== P2: start within 2 h (4 half-hours) and end within 6 h (12), misses count as failures"
    )
    ends_df = d.filter(pl.col("end") > 0)
    print(
        f"start {np.mean([(not math.isnan(a)) and abs(a) <= 4 for a in d['start_err_hh']]):.3f}  "
        "end (events with an end) "
        f"{np.mean([(not math.isnan(a)) and abs(a) <= 12 for a in ends_df['end_err_hh']]):.3f}"
    )
    print(
        "\n== Recall by duration band, default spec, three slow-level arms "
        "(all positive events 4+ units)"
    )
    b4 = pos.filter((pl.col("units") >= 4) & (pl.col("spec") == "default"))
    for arm in ("masked", "huber", "rw30"):
        row = []
        for b in BANDS:
            g = b4.filter((pl.col("arm") == arm) & (pl.col("band") == b))
            row.append(f"{b}: {wrec(g, 'found'):.2f} (n={g.height})")
        print(arm, " | ".join(row))
    print(
        "\n== Absorption by band (median share of the event's MW taken by the slow level), "
        "sparse long-event worlds,"
    )
    print("   4+ units, events with no other injected event at the same Primary within 30 days")
    ab = []
    iso_t = band_col(pl.read_parquet(WORK / "isolated_truth_scored.parquet"))
    for (_spec, _world, _arm), g in iso_t.group_by(["spec", "world", "arm"]):
        ends = np.where(g["end"].to_numpy() < 0, GRID_LENGTH, g["end"].to_numpy())
        st = g["start"].to_numpy()
        sites = g["site"].to_numpy()
        grp = g["group"].to_numpy()
        ok = np.ones(g.height, bool)
        for i in range(g.height):
            m = (
                (sites == sites[i])
                & (grp != grp[i])
                & (st < ends[i] + 30 * DAY)
                & (ends > st[i] - 30 * DAY)
            )
            ok[i] = not m.any()
        ab.append(g.with_columns(isolated=pl.Series(ok)))
    isol = pl.concat(ab).filter(
        pl.col("isolated")
        & (pl.col("units") >= 4)
        & pl.col("absorption").is_finite()
        & pl.col("kind").is_in(["edge", "solo", "permanent_transfer", "permanent_shift"])
    )
    for arm in ("masked", "huber", "rw30"):
        row = []
        for b in BANDS:
            g = isol.filter((pl.col("arm") == arm) & (pl.col("band") == b))
            row.append(
                f"{b}: {g['absorption'].median() if g.height else float('nan'):.2f} (n={g.height})"
            )
        print(arm, " | ".join(row))
    print("\n== 95% intervals, default spec, masked arm (whole worlds resampled, 2000 draws)")
    rng = np.random.default_rng(3)
    dm = big.filter((pl.col("spec") == "default") & (pl.col("arm") == "masked"))
    worlds = dm["world"].unique().to_list()
    for col, sub in (("found", dm), ("right_class", dm.filter(pl.col("edge_truth")))):
        vals = []
        for _ in range(2000):
            pick = rng.choice(worlds, size=len(worlds))
            g = pl.concat([sub.filter(pl.col("world") == w) for w in pick])
            vals.append(wrec(g, col))
        print(col, f"{wrec(sub, col):.3f}", np.round(np.percentile(vals, [2.5, 97.5]), 3))
    print("\n== Decoded class for each injected kind (default spec, masked), share of events")
    allk = t.filter(
        (pl.col("spec") == "default") & (pl.col("arm") == "masked") & (pl.col("units") >= 2)
    )
    print(
        allk.group_by("kind", "decoded_cls")
        .len()
        .pivot(on="decoded_cls", index="kind", values="len")
        .fill_null(0)
    )
    print("\n== Hard negatives called an edge transfer (share of each kind)")
    for k in ("routine", "ramp", "pulse", "solo", "placebo"):
        g = allk.filter(pl.col("kind") == k)
        print(k, g.height, f"{(g['decoded_cls'] == 'edge_transfer').mean():.3f}")
    print(
        "\n== Temporary against permanent, events whose truth start is 180+ days before the "
        "record end, found"
    )
    c = pos.filter(
        (pl.col("spec") == "default")
        & (pl.col("arm") == "masked")
        & pl.col("found")
        & (pl.col("start") < GRID_LENGTH - 180 * DAY)
    )
    c = c.with_columns(
        truth=pl.when(pl.col("days") < 0).then(pl.lit("permanent")).otherwise(pl.lit("temporary"))
    )
    print(c.group_by("truth", "decoded_kind").len())
    wrong = c.filter(
        ((pl.col("truth") == "permanent") & (pl.col("decoded_kind") != "permanent"))
        | ((pl.col("truth") == "temporary") & (pl.col("decoded_kind") == "permanent"))
    ).height
    print("long and permanent confused:", wrong, "of", c.height)
    print("\n== False events from solar growth and seasonal bias, per injected Primary-year")
    fp = pl.read_parquet(WORK / "growth_bias_false_events.parquet")
    print(
        fp.group_by("spec", "arm").agg(
            (pl.col("new_events").sum() / (pl.col("n_sites").sum() * YEARS)).alias(
                "per_primary_year"
            )
        )
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(BANDS))
    for i, (arm, col) in enumerate((("masked", BLUE), ("huber", ORANGE), ("rw30", GREY))):
        v = [
            wrec(b4.filter((pl.col("arm") == arm) & (pl.col("band") == b)), "found") for b in BANDS
        ]
        ax.bar(
            x + (i - 1) * 0.27,
            v,
            0.27,
            color=col,
            label={
                "masked": "masked, frozen (default)",
                "huber": "Huber, no mask",
                "rw30": "30-day random walk",
            }[arm],
        )
    ax.set_xticks(x, BANDS)
    ax.set_ylabel("Share of injected MW found")
    ax.set_xlabel("Event duration")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "figure_recall_by_band.svg")
    fig.savefig("/tmp/e653_png/recall.png", dpi=90)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (arm, col) in enumerate((("masked", BLUE), ("huber", ORANGE), ("rw30", GREY))):
        v = [
            float(
                cast(
                    "float | None",
                    isol.filter((pl.col("arm") == arm) & (pl.col("band") == b))[
                        "absorption"
                    ].median(),
                )
                or 0.0
            )
            for b in BANDS
        ]
        ax.bar(x + (i - 1) * 0.27, v, 0.27, color=col, label=arm)
    ax.set_xticks(x, BANDS)
    ax.set_ylabel("Share of event taken by the slow level")
    ax.set_xlabel("Event duration")
    ax.axhline(0.25, color="k", lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure_absorption_by_band.svg")
    fig.savefig("/tmp/e653_png/absorb.png", dpi=90)
    plt.close(fig)


def ts(i: int) -> str:
    """UTC time of a half-hour index."""
    return (GRID_START + timedelta(minutes=30 * int(i))).strftime("%Y-%m-%d %H:%M")


def real() -> None:  # noqa: PLR0915 - one linear printout of many tables
    """Tables of the real record."""
    panel = Panel()
    ev = pl.read_parquet(WORK / "real_events_masked.parquet").filter(pl.col("start") >= SCORED_FROM)
    n_prim = len(panel.sites)
    print("== Real record, scored years (2024-01-01 to 2026-06-30), default arm")
    print(f"events {ev.height}, per Primary-year {ev.height / (n_prim * YEARS):.2f}")
    print(ev.group_by("cls").len().sort("len", descending=True))
    per = (
        ev.group_by("primary")
        .len()
        .with_columns(rate=pl.col("len") / YEARS)
        .sort("rate", descending=True)
    )
    print(per.head(5))
    med = per["rate"].median()
    print("median rate", med, "D8 rate", per.filter(pl.col("primary") == "D8")["rate"].to_list())
    print("edges that carry data:", len(panel.edges))
    rw = json.loads((WORK / "rewired.json").read_text())
    print(
        f"edge events real {rw['real']}; rewired (100 draws) median {np.median(rw['rewired'])} "
        f"90% range {np.percentile(rw['rewired'], 5)}-{np.percentile(rw['rewired'], 95)}; "
        "real share of rewired median "
        f"{np.median(rw['rewired']) / max(rw['real'], 1):.2f}"
    )
    lab = pl.read_parquet(E580 / "events_labelled_status.parquet").filter(
        (pl.col("label_length_days") > 1)
        & pl.col("label").is_in(panel.sites)
        & (pl.col("label_onset_idx") - OFFSET_2019 >= SCORED_FROM)
    )
    print(
        "\n== Logged shifts of more than 1 day starting 2024-01-01 or later at the 15 Primaries: "
        f"{lab.height}"
    )
    names = dict(zip(SERIES_IDS, panel.sites, strict=True))
    lr = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00254-lsc-alarm-precision-against-full-log-with-chance-base-rate/alarms_LR_b20.parquet"
    )
    lr = lr.with_columns(site=pl.col("series").replace_strict(names, default=None)).filter(
        pl.col("onset") >= SCORED_FROM
    )
    lr = lr.unique(subset=["site", "onset"])

    def found(starts: dict[str, list[int]]) -> tuple[int, int]:
        strict = loose = 0
        for r in lab.iter_rows(named=True):
            on = r["label_onset_idx"] - OFFSET_2019
            s = starts.get(r["label"], [])
            strict += any(abs(x - on) <= 4 for x in s)
            loose += any(abs(x - on) <= 48 for x in s)
        return strict, loose

    mine = {p: ev.filter(pl.col("primary") == p)["start"].to_list() for p in panel.sites}
    base = {p: lr.filter(pl.col("site") == p)["onset"].to_list() for p in panel.sites}
    target = lr.height / (n_prim * YEARS)
    print(
        f"LR_b20 alarms in scored years {lr.height} ({target:.2f} per Primary-year): "
        f"strict/loose found {found(base)}"
    )
    print(
        f"this table, all events: {ev.height} "
        f"({ev.height / (n_prim * YEARS):.2f} per Primary-year): "
        f"strict/loose {found(mine)}"
    )
    for thr in np.linspace(0.3, 0.99, 40):
        e2 = ev.filter(pl.col("confidence") >= thr)
        if e2.height / (n_prim * YEARS) <= target:
            m2 = {p: e2.filter(pl.col("primary") == p)["start"].to_list() for p in panel.sites}
            print(
                f"equal alarm rate (confidence >= {thr:.2f}): {e2.height} events, "
                f"strict/loose {found(m2)}"
            )
            break
    print(
        "\n== Event-free stratum: decoded events more than 14 days from every labelled or "
        "verified event, scored years"
    )
    allab = pl.read_parquet(E580 / "events_labelled_status.parquet")
    spans = [
        (r["label"], r["label_onset_idx"] - OFFSET_2019, r["label_end_idx"] - OFFSET_2019)
        for r in allab.iter_rows(named=True)
    ]
    ver = pl.read_parquet(E580 / "events_verified.parquet")
    shift = OFFSET_2019
    for r in ver.iter_rows(named=True):
        spans += [
            (r["primary_a"], r["onset_a_idx"] - shift, r["end_a_idx"] - shift),
            (r["primary_b"], r["onset_b_idx"] - shift, r["end_b_idx"] - shift),
        ]
    free_half_hours = 0
    for site in panel.sites:
        busy = np.zeros(GRID_LENGTH, bool)
        for lab_site, a, b in spans:
            if lab_site == site:
                busy[max(0, a - 14 * DAY) : max(0, b + 14 * DAY)] = True
        busy[:SCORED_FROM] = True
        free_half_hours += int((~busy & np.isfinite(panel.residual[site])).sum())
    print(
        f"event-free half-hours {free_half_hours} = "
        f"{free_half_hours / (DAY * 365.25):.1f} Primary-years of {n_prim * YEARS:.1f}"
    )
    for arm in ("masked", "huber", "rw30"):
        e = pl.read_parquet(WORK / f"real_events_{arm}.parquet").filter(
            pl.col("start") >= SCORED_FROM
        )
        n = 0
        for r in e.iter_rows(named=True):
            end = GRID_LENGTH if r["end"] < 0 else r["end"]
            if not any(
                site == r["primary"] and r["start"] < b + 14 * DAY and end > a - 14 * DAY
                for site, a, b in spans
            ):
                n += 1
        print(
            f"{arm}: {n} decoded events in the stratum, "
            f"{n / (free_half_hours / (DAY * 365.25)):.2f} per event-free Primary-year; "
            f"all rows {e.height / (n_prim * YEARS):.2f}"
        )
    print("\n== Verified and descriptive cases (all arms)")
    for arm in ("masked", "huber", "rw30"):
        e = pl.read_parquet(WORK / f"real_events_{arm}.parquet")
        slows = np.load(WORK / f"real_slow_{arm}.npz")
        for site, a, b in (
            ("D5", 69900, 70800),
            ("D16", 69900, 70800),
            ("D3", 49300, 50800),
            ("D22", 49300, 50800),
            ("D10", 63000, 66000),
            ("D12", 63000, 66000),
        ):
            for r in e.filter(
                (pl.col("primary") == site)
                & (pl.col("start") > a - 400)
                & (pl.col("start") < b)
                & (pl.col("units") > 3)
            ).iter_rows(named=True):
                s0, s1 = (r["start"] - 14 * DAY) // 12, (r["start"] + 7 * DAY) // 12
                move = (slows[site][s1] - slows[site][s0]) * 100 / panel.capacity[site]
                print(
                    f"{arm:7s} {site:4s} {ts(r['start'])} -> "
                    f"{ts(r['end']) if r['end'] > 0 else 'open'} "
                    f"{r['amp_mw']:+.2f} MW {r['units']:.1f} units {r['cls']} "
                    f"partners={r['partners']} "
                    f"slow level moved {move:+.1f} points of capacity"
                )


def scenarios() -> None:
    """Normalised MAE of the median forecast in the four scenarios, by decoded events."""
    panel = Panel()
    ev = pl.read_parquet(WORK / "real_events_masked.parquet")
    slows = np.load(WORK / "real_slow_masked.npz")
    from e00653_core import Frozen  # noqa: F401
    from e00653_slow import to_half_hours

    print(
        "\n== Four scenarios (decoded events), normalised MAE of the median forecast = "
        "mean |observed - GAM forecast| / effective capacity, in per cent"
    )
    print(
        "   'after table' = the same after removing the slow level and the decoded events "
        "(hindsight)"
    )
    rows = []
    for site in panel.sites:
        r = panel.residual[site]
        sl = to_half_hours(slow=slows[site])
        corr = r.copy()
        corr[: len(sl)] -= sl
        inside = np.zeros(GRID_LENGTH, bool)
        after = np.zeros(GRID_LENGTH, bool)
        near = np.zeros(GRID_LENGTH, bool)
        for e in ev.filter(pl.col("primary") == site).iter_rows(named=True):
            end = GRID_LENGTH if e["end"] < 0 else e["end"]
            inside[e["start"] : end] = True
            after[end : end + 7 * DAY] = True
            near[max(0, e["start"] - 0) : end + 7 * DAY] = True
            corr[e["start"] : end] -= e["amp_mw"]
        sc = {
            "all rows": np.ones(GRID_LENGTH, bool),
            "outside events, 7+ days after": ~near,
            "inside events": inside,
            "7 days after an event": after & ~inside,
        }
        for name, m_scenario in sc.items():
            m = m_scenario & (np.arange(GRID_LENGTH) >= SCORED_FROM) & np.isfinite(r)
            rows.append(
                {
                    "site": site,
                    "scenario": name,
                    "n": int(m.sum()),
                    "before": float(np.abs(r[m]).sum()),
                    "after": float(np.abs(corr[m]).sum()),
                    "cap": panel.capacity[site],
                }
            )
    d = pl.DataFrame(rows)
    out = (
        d.with_columns(b=pl.col("before") / pl.col("cap"), a=pl.col("after") / pl.col("cap"))
        .group_by("scenario", maintain_order=True)
        .agg(
            pl.col("n").sum(),
            (100 * pl.col("b").sum() / pl.col("n").sum()).alias("nMAE_before_pct"),
            (100 * pl.col("a").sum() / pl.col("n").sum()).alias("nMAE_after_table_pct"),
        )
    )
    print(out)


def examples() -> None:
    """Event time series: observed power, baseline forecast K1, and the decoded events."""
    panel = Panel()
    ev = pl.read_parquet(WORK / "real_events_masked.parquet")
    D = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
    cases = [
        (
            "D5_D16",
            ["D5", "D16"],
            "2025-12-29 01:00",
            "2025-12-15",
            "2026-01-21",
            ["K1_2025-12", "K1_2026-01"],
        )
    ]
    names = dict(zip(panel.sites, SERIES_IDS, strict=True))
    for tag, sites, _start, lo, hi, files in cases:
        k1 = pl.concat([pl.read_parquet(D / f"{f}.parquet") for f in files]).filter(
            (pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36)
        )
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
        for ax, site in zip(axes, sites, strict=True):
            lo_i, hi_i = (
                int((np.datetime64(lo) - np.datetime64("2022-01-01")) / np.timedelta64(30, "m")),
                int((np.datetime64(hi) - np.datetime64("2022-01-01")) / np.timedelta64(30, "m")),
            )
            t = np.arange(lo_i, hi_i)
            tt = np.datetime64("2022-01-01T00:00") + t.astype("timedelta64[30m]")
            ax.plot(tt, panel.power[site][t], color="k", lw=0.8, label="observed power")
            kk = k1.filter(pl.col("time_series_id") == names[site]).sort("valid_time")
            kt = kk["valid_time"].dt.replace_time_zone(None).to_numpy()
            ax.plot(kt, kk["q50"].to_numpy(), color=ORANGE, lw=0.8, label="K1 day-ahead median")
            for e in ev.filter(pl.col("primary") == site).iter_rows(named=True):
                if hi_i > e["start"] > lo_i - 2000 and e["units"] >= 3:
                    end = hi_i if e["end"] < 0 else e["end"]
                    ax.axvspan(
                        np.datetime64("2022-01-01T00:00") + np.timedelta64(30 * e["start"], "m"),
                        np.datetime64("2022-01-01T00:00")
                        + np.timedelta64(30 * min(end, hi_i), "m"),
                        color=BLUE,
                        alpha=0.12,
                    )
                    ax.text(
                        np.datetime64("2022-01-01T00:00") + np.timedelta64(30 * e["start"], "m"),
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
        fig.savefig(OUT / f"figure_event_series_{tag}.svg")
        fig.savefig(f"/tmp/e653_png/{tag}.png", dpi=90)
        plt.close(fig)


if __name__ == "__main__":
    import sys

    {"injected": injected, "real": real, "scenarios": scenarios, "examples": examples}[
        sys.argv[1]
    ]()

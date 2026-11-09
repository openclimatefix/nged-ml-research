"""Print every number the page of experiment 657 quotes and draw its figures.

    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 \
        PYTHONPATH=ml-experiments/00653-edge-state-event-table-penalised-segmentation:. \
        uv run --no-sync python e00657_report.py <part>

Parts: `gain`, `events` (P1 to P4, equal width, logged shifts, event-free stratum),
`recall` (P2 and recall by band), `scenarios` (four-scenario normalised MAE), `figures`, `all`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, GRID_START, SCORED_FROM, SERIES_IDS, Panel  # ty: ignore[unresolved-import]
from e00653_inject import OFFSET_2019  # ty: ignore[unresolved-import]
from e00653_slow import to_half_hours  # ty: ignore[unresolved-import]
from e00657_solar import (
    ARM_NAMES,
    N_DAYS,
    SIX,
    WORK,
    SolarInputs,
    build,
    busy_mask,
    midday_nmae,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00657-event-table-with-explicit-solar-term-against-plain-arm"
)
PNG: Final[Path] = WORK / "png"
E580: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00580-verified-load-shift-event-set-and-label-audit/work"
)
YEARS: Final[float] = (GRID_LENGTH - SCORED_FROM) / (365.25 * DAY)
BANDS: Final[list[str]] = ["3h-1d", "1-7d", "7-21d", "21-60d", "60d+", "permanent"]
COLOURS: Final[dict[str, str]] = {"plain": "#306BFF", "solar": "#FF4901", "placebo": "#888888"}
K1_COLOUR: Final[str] = "#10C5F7"
N_BOOT: Final[int] = 2000
POSITIVE_KINDS: Final[list[str]] = ["edge", "solo", "permanent_transfer", "permanent_shift"]
SCORED_DAY0: Final[int] = SCORED_FROM // DAY


def load_events(*, arm: str) -> pl.DataFrame:
    """Decoded real events of an arm that start in the scored years."""
    return pl.read_parquet(WORK / f"real_events_{arm}.parquet").filter(
        pl.col("start") >= SCORED_FROM
    )


def month_of_day() -> np.ndarray:
    """Month index (months since 2022-01) of each day on the grid."""
    days = np.datetime64(GRID_START.replace(tzinfo=None), "D") + np.arange(N_DAYS)
    months = days.astype("datetime64[M]")
    return (months - np.datetime64("2022-01", "M")).astype(int)


def free_days(*, inp: SolarInputs, sites: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Per site, the scored-year days with no half-hour within 14 days of a labelled event."""
    out = {}
    for s in sites:
        busy = busy_mask(spans=inp.spans, site=s, pad_days=14).reshape(N_DAYS, DAY).any(axis=1)
        finite = np.isfinite(inp.res[s]).reshape(N_DAYS, DAY).mean(axis=1) >= 0.5
        ok = ~busy & finite
        ok[:SCORED_DAY0] = False
        out[s] = ok
    return out


def in_stratum(*, inp: SolarInputs, events: pl.DataFrame) -> pl.DataFrame:
    """Events more than 14 days from every labelled or verified event, tagged sunny or not."""
    keep = []
    for r in events.iter_rows(named=True):
        end = GRID_LENGTH if r["end"] < 0 else r["end"]
        free = not any(
            site == r["primary"] and r["start"] < b + 14 * DAY and end > a - 14 * DAY
            for site, a, b in inp.spans
        )
        keep.append(free)
    free_events = events.filter(pl.Series(keep))
    sunny = []
    for r in free_events.iter_rows(named=True):
        sd = inp.sunny_day.get(r["primary"])
        sunny.append(bool(sd[r["start"] // DAY]) if sd is not None else False)
    return free_events.with_columns(sunny=pl.Series(sunny, dtype=pl.Boolean))


def rates(
    *, inp: SolarInputs, events: pl.DataFrame, sites: tuple[str, ...] = SIX
) -> dict[str, float]:
    """Counts and per-Primary-year rates in the event-free stratum, by start-day type."""
    free = free_days(inp=inp, sites=sites)
    sunny_days = sum(int((free[s] & inp.sunny_day[s]).sum()) for s in sites)
    other_days = sum(int((free[s] & ~inp.sunny_day[s]).sum()) for s in sites)
    ev = in_stratum(inp=inp, events=events.filter(pl.col("primary").is_in(list(sites))))
    n_s = int(ev["sunny"].sum())
    n_o = ev.height - n_s
    return {
        "n_sunny": n_s,
        "n_other": n_o,
        "days_sunny": sunny_days,
        "days_other": other_days,
        "rate_sunny": n_s / (sunny_days / 365.25),
        "rate_other": n_o / (other_days / 365.25),
    }


def month_counts(*, inp: SolarInputs, events: pl.DataFrame) -> dict[str, np.ndarray]:
    """Per-month event and day counts (sunny and other) at the six, for the bootstrap."""
    mon = month_of_day()
    n_m = int(mon.max()) + 1
    free = free_days(inp=inp, sites=SIX)
    out = {k: np.zeros(n_m) for k in ("n_sunny", "n_other", "d_sunny", "d_other")}
    for s in SIX:
        np.add.at(out["d_sunny"], mon[free[s] & inp.sunny_day[s]], 1)
        np.add.at(out["d_other"], mon[free[s] & ~inp.sunny_day[s]], 1)
    ev = in_stratum(inp=inp, events=events.filter(pl.col("primary").is_in(list(SIX))))
    for r in ev.iter_rows(named=True):
        out["n_sunny" if r["sunny"] else "n_other"][mon[r["start"] // DAY]] += 1
    return out


def rel_change(
    *, a: dict[str, np.ndarray], b: dict[str, np.ndarray], idx: np.ndarray, kind: str
) -> float:
    """Relative change of the rate in `b` against `a` over the months `idx`."""
    ra = a[f"n_{kind}"][idx].sum() / a[f"d_{kind}"][idx].sum()
    rb = b[f"n_{kind}"][idx].sum() / b[f"d_{kind}"][idx].sum()
    return float(rb / ra - 1) if ra > 0 else float("nan")


def gain_part(*, inp: SolarInputs) -> None:
    """Print the gain and P5."""
    print("== Gain and P5")
    print({k: round(v, 4) for k, v in inp.stats.items()})
    print("capacity C (MW):", {k: round(inp.capacity[k], 2) for k in SIX})
    for name, lo, hi in (("tuning", 0, SCORED_FROM), ("scored", SCORED_FROM, GRID_LENGTH)):
        v = {}
        for a in ARM_NAMES:
            r = {s: inp.res[s] + inp.delta[a][s] for s in inp.panel.sites}
            v[a] = midday_nmae(inp=inp, residual=r, lo=lo, hi=hi)
        print(
            f"P5 {name}: plain {v['plain']:.3f}% solar {v['solar']:.3f}% "
            f"({100 * (v['solar'] / v['plain'] - 1):+.2f}%) placebo {v['placebo']:.3f}% "
            f"({100 * (v['placebo'] / v['plain'] - 1):+.2f}%)"
        )
    print(f"gate: g = {inp.gain:.4f}; under 0.05: {inp.gain < 0.05}")


def matched(*, panel: Panel, events: pl.DataFrame, sites: tuple[str, ...]) -> tuple[int, int, int]:
    """Logged shifts over 1 day found (strict +-4 half-hours, loose +-48), and their count."""
    lab = pl.read_parquet(E580 / "events_labelled_status.parquet").filter(
        (pl.col("label_length_days") > 1)
        & pl.col("label").is_in(list(sites))
        & (pl.col("label_onset_idx") - OFFSET_2019 >= SCORED_FROM)
    )
    starts = {p: events.filter(pl.col("primary") == p)["start"].to_list() for p in panel.sites}
    strict = loose = 0
    for r in lab.iter_rows(named=True):
        on = r["label_onset_idx"] - OFFSET_2019
        s = starts.get(r["label"], [])
        strict += any(abs(x - on) <= 4 for x in s)
        loose += any(abs(x - on) <= 48 for x in s)
    return strict, loose, lab.height


def thin(*, events: pl.DataFrame, n: int) -> pl.DataFrame:
    """Keep the `n` most confident events (raising the confidence threshold)."""
    if events.height <= n:
        return events
    return events.sort("confidence", descending=True).head(n)


def events_part(*, inp: SolarInputs) -> None:
    """P1, P3, P4, equal width, logged shifts, and the event-free stratum."""
    ev = {a: load_events(arm=a) for a in ARM_NAMES}
    print("== Event counts (scored years 2024-01-01 to 2026-06-30)")
    for a in ARM_NAMES:
        six = ev[a].filter(pl.col("primary").is_in(list(SIX)))
        print(f"{a}: all 15 Primaries {ev[a].height}, six MW Primaries {six.height}")
    print(
        "\n== Event-free stratum (>14 days from every labelled or verified event), six MW Primaries"
    )
    r = {a: rates(inp=inp, events=ev[a]) for a in ARM_NAMES}
    for a in ARM_NAMES:
        x = r[a]
        print(
            f"{a:8s} sunny-midday days: {x['n_sunny']} events / {x['days_sunny'] / 365.25:.2f} "
            f"Primary-years = {x['rate_sunny']:.2f} per Primary-year | other days: "
            f"{x['n_other']} / "
            f"{x['days_other'] / 365.25:.2f} = {x['rate_other']:.2f}"
        )
    n15 = {a: in_stratum(inp=inp, events=ev[a]).height for a in ARM_NAMES}
    mc = {a: month_counts(inp=inp, events=ev[a]) for a in ARM_NAMES}
    rng = np.random.default_rng(657)
    months = np.flatnonzero(mc["plain"]["d_sunny"] + mc["plain"]["d_other"] > 0)
    print("\n== P1 / P3 / P4 (relative change against the plain arm; month bootstrap, 95%)")
    for kind, label in (("sunny", "P1 sunny-midday"), ("other", "P3 non-sunny")):
        for a in ("solar", "placebo"):
            point = rel_change(a=mc["plain"], b=mc[a], idx=months, kind=kind)
            boots = [
                rel_change(
                    a=mc["plain"], b=mc[a], idx=rng.choice(months, size=len(months)), kind=kind
                )
                for _ in range(N_BOOT)
            ]
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            print(f"{label} {a}: {100 * point:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    fall_solar = r["plain"]["rate_sunny"] - r["solar"]["rate_sunny"]
    fall_placebo = r["plain"]["rate_sunny"] - r["placebo"]["rate_sunny"]
    ratio = fall_placebo / fall_solar if fall_solar > 0 else float("nan")
    print(
        f"P4: sunny rate fall, solar {fall_solar:+.3f}, placebo {fall_placebo:+.3f} "
        "per Primary-year; "
        f"placebo/solar = {ratio:.2f} (undefined when the solar fall is not positive)"
    )
    p1 = (r["solar"]["rate_sunny"] / r["plain"]["rate_sunny"] - 1) * 100
    print(
        f"P1 holds (fall of 20% or more): {p1 <= -20}; P3 holds (|change| < 10%): "
        f"{abs(r['solar']['rate_other'] / r['plain']['rate_other'] - 1) < 0.10}"
    )
    print("\n== Equal width: the arm with more events at the six is thinned to the other's count")
    n6 = {a: ev[a].filter(pl.col("primary").is_in(list(SIX))).height for a in ARM_NAMES}
    for a in ("solar", "placebo"):
        target = min(n6["plain"], n6[a])
        pa = thin(events=ev["plain"].filter(pl.col("primary").is_in(list(SIX))), n=target)
        sa = thin(events=ev[a].filter(pl.col("primary").is_in(list(SIX))), n=target)
        rp, rs = rates(inp=inp, events=pa), rates(inp=inp, events=sa)
        print(
            f"plain vs {a} at {target} events each: sunny rate {rp['rate_sunny']:.2f} vs "
            f"{rs['rate_sunny']:.2f} ({rp['n_sunny']} vs {rs['n_sunny']} events); other-day rate "
            f"{rp['rate_other']:.2f} vs {rs['rate_other']:.2f} ({rp['n_other']} vs {rs['n_other']})"
        )
    print("\n== Logged shifts (over 1 day, scored years) matched: strict / loose of n")
    for a in ARM_NAMES:
        s15 = matched(panel=inp.panel, events=ev[a], sites=tuple(inp.panel.sites))
        s6 = matched(panel=inp.panel, events=ev[a], sites=SIX)
        print(
            f"{a:8s} 15 Primaries {s15[0]}/{s15[1]} of {s15[2]} | "
            f"six MW Primaries {s6[0]}/{s6[1]} of {s6[2]}"
        )
    print("\n== Event-free stratum per arm: events per event-free Primary-year, 15 Primaries")
    free15 = free_days(inp=inp, sites=tuple(inp.panel.sites))
    py = sum(int(v.sum()) for v in free15.values()) / 365.25
    for a in ARM_NAMES:
        n = n15[a]
        print(
            f"{a:8s} {n} events / {py:.1f} event-free Primary-years = {n / py:.2f} "
            f"(all rows {ev[a].height / (15 * YEARS):.2f})"
        )


def wrec(df: pl.DataFrame, col: str) -> float:
    """MW-weighted share with `col` true."""
    w = df["amp_mw"].abs()
    return (
        float((df[col].cast(pl.Float64) * w).sum()) / float(w.sum()) if df.height else float("nan")
    )


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


def truth_six() -> pl.DataFrame:
    """Scored positive injected events of 4 units or more at the six Primaries."""
    t = band_col(pl.read_parquet(WORK / "injected_truth_scored.parquet"))
    return t.filter(
        pl.col("site").is_in(list(SIX))
        & pl.col("kind").is_in(POSITIVE_KINDS)
        & (pl.col("units") >= 4)
    )


def recall_part() -> None:
    """P2 and recall by duration band."""
    t = truth_six()
    big = t.filter((pl.col("days") >= 1) | (pl.col("days") < 0))
    worlds = sorted(big["world"].unique().to_list())
    print(
        "== P2: MW-weighted recall of injected events of 4+ units and 1+ day at the six Primaries"
    )
    per = {
        a: np.array(
            [
                wrec(big.filter((pl.col("arm") == a) & (pl.col("world") == w)), "found")
                for w in worlds
            ]
        )
        for a in ARM_NAMES
    }
    n_ev = big.filter(pl.col("arm") == "plain").height
    for a in ARM_NAMES:
        pooled = wrec(big.filter(pl.col("arm") == a), "found")
        print(
            f"{a:8s} pooled {pooled:.3f} (n={big.filter(pl.col('arm') == a).height}), "
            f"per world {np.round(per[a], 3)}"
        )
    rng = np.random.default_rng(3)
    for a in ("solar", "placebo"):
        d = per[a] - per["plain"]
        boots = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(
            f"paired difference {a} - plain over {len(worlds)} worlds ({n_ev} events): "
            f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; holds (>= -0.03): {d.mean() >= -0.03}"
        )
    print(
        "\n== Recall by duration band (all positive events of 4+ units at the six), MW-weighted (n)"
    )
    for a in ARM_NAMES:
        row = []
        for b in BANDS:
            g = t.filter((pl.col("arm") == a) & (pl.col("band") == b))
            row.append(f"{b}: {wrec(g, 'found'):.2f} ({g.height})")
        print(f"{a:8s}", " | ".join(row))


def scenarios_part(*, inp: SolarInputs) -> None:
    """Normalised MAE of the median forecast in the four scenarios, per arm, for 15 and the six."""
    print(
        "== Four scenarios: mean |observed - forecast| / effective capacity in per cent; "
        "'after' removes the slow level and the decoded events (hindsight)"
    )
    for a in ARM_NAMES:
        ev = pl.read_parquet(WORK / f"real_events_{a}.parquet")
        slows = np.load(WORK / f"real_slow_{a}.npz")
        rows = []
        for site in inp.panel.sites:
            r = inp.panel.residual[site] + inp.delta[a][site]
            corr = r - to_half_hours(slow=slows[site])
            inside = np.zeros(GRID_LENGTH, bool)
            after = np.zeros(GRID_LENGTH, bool)
            near = np.zeros(GRID_LENGTH, bool)
            for e in ev.filter(pl.col("primary") == site).iter_rows(named=True):
                end = GRID_LENGTH if e["end"] < 0 else e["end"]
                inside[e["start"] : end] = True
                after[end : end + 7 * DAY] = True
                near[e["start"] : end + 7 * DAY] = True
                corr[e["start"] : end] -= e["amp_mw"]
            sc = {
                "all rows": np.ones(GRID_LENGTH, bool),
                "outside events, 7+ days after": ~near,
                "inside events": inside,
                "7 days after an event": after & ~inside,
            }
            for name, m_sc in sc.items():
                m = m_sc & (np.arange(GRID_LENGTH) >= SCORED_FROM) & np.isfinite(r)
                rows.append(
                    {
                        "site": site,
                        "scenario": name,
                        "n": int(m.sum()),
                        "before": float(np.abs(r[m]).sum()) / inp.panel.capacity[site],
                        "after": float(np.abs(corr[m]).sum()) / inp.panel.capacity[site],
                    }
                )
        d = pl.DataFrame(rows)
        for label, sites in (("15 Primaries", inp.panel.sites), ("six MW Primaries", list(SIX))):
            out = (
                d.filter(pl.col("site").is_in(sites))
                .group_by("scenario", maintain_order=True)
                .agg(
                    pl.col("n").sum(),
                    (100 * pl.col("before").sum() / pl.col("n").sum()).alias("nMAE_before_pct"),
                    (100 * pl.col("after").sum() / pl.col("n").sum()).alias("nMAE_after_table_pct"),
                )
            )
            print(f"\n-- arm {a}, {label}")
            print(out)


def optimise(path: Path) -> None:
    """Run svgo on a figure and strip trailing whitespace."""
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=False,
    )
    text = path.read_text()
    path.write_text("\n".join(line.rstrip() for line in text.split("\n")))


def save(fig: plt.Figure, name: str) -> None:
    """Save SVG (optimised) and a PNG for checking the render."""
    OUT.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(PNG / f"{name}.png", dpi=90)
    plt.close(fig)
    optimise(OUT / f"{name}.svg")


def figures_part(*, inp: SolarInputs) -> None:  # noqa: PLR0915 - one linear figure routine
    """Figures (a) events per Primary-year by stratum, (b) recall by band, (c) D5 and D16 series."""
    ev = {a: load_events(arm=a) for a in ARM_NAMES}
    r = {a: rates(inp=inp, events=ev[a]) for a in ARM_NAMES}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, a in enumerate(ARM_NAMES):
        vals = [r[a]["rate_sunny"], r[a]["rate_other"]]
        counts = [r[a]["n_sunny"], r[a]["n_other"]]
        x = np.arange(2) + (i - 1) * 0.27
        ax.bar(x, vals, 0.27, color=COLOURS[a], label=a)
        for xi, v, c in zip(x, vals, counts, strict=True):
            ax.text(float(xi), float(v), f"n={c}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(2), ["Sunny-midday days", "Other days"])
    ax.set_ylabel("Decoded events per Primary-year")
    ax.set_title("Event-free stratum, six megawatt Primaries, 2024-01 to 2026-06", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.tight_layout()
    save(fig, "figure_events_per_primary_year")

    t = truth_six()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(BANDS))
    for i, a in enumerate(ARM_NAMES):
        v = [wrec(t.filter((pl.col("arm") == a) & (pl.col("band") == b)), "found") for b in BANDS]
        ax.bar(x + (i - 1) * 0.27, v, 0.27, color=COLOURS[a], label=a)
    ax.set_xticks(x, BANDS)
    ax.set_ylabel("Share of injected MW found")
    ax.set_xlabel("Event duration")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout()
    save(fig, "figure_recall_by_band")

    k1_dir = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
    k1 = pl.concat(
        [pl.read_parquet(k1_dir / f"{f}.parquet") for f in ("K1_2025-12", "K1_2026-01")]
    ).filter((pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36))
    ids = dict(zip(inp.panel.sites, SERIES_IDS, strict=True))
    lo, hi = "2025-12-15", "2026-01-21"
    lo_i = int((np.datetime64(lo) - np.datetime64("2022-01-01")) / np.timedelta64(30, "m"))
    hi_i = int((np.datetime64(hi) - np.datetime64("2022-01-01")) / np.timedelta64(30, "m"))
    t_idx = np.arange(lo_i, hi_i)
    tt = np.datetime64("2022-01-01T00:00") + t_idx.astype("timedelta64[30m]")

    def at(i: int) -> np.datetime64:
        return np.datetime64("2022-01-01T00:00") + np.timedelta64(30 * int(i), "m")

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.4), sharex=True)
    for j, site in enumerate(("D5", "D16")):
        top, bot = axes[2 * j], axes[2 * j + 1]
        top.plot(tt, inp.panel.power[site][t_idx], color="k", lw=0.8, label="observed power")
        kk = k1.filter(pl.col("time_series_id") == ids[site]).sort("valid_time")
        top.plot(
            kk["valid_time"].dt.replace_time_zone(None).to_numpy(),
            kk["q50"].to_numpy(),
            color=K1_COLOUR,
            lw=0.9,
            label="K1 day-ahead median",
        )
        top.set_ylabel(f"{site} power (MW)")
        top.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        plain = inp.res[site][t_idx]
        solar = plain + inp.delta["solar"][site][t_idx]
        bot.plot(tt, plain, color=COLOURS["plain"], lw=0.8, label="plain residual")
        bot.plot(tt, solar, color=COLOURS["solar"], lw=0.8, label="solar-term residual")
        bot.axhline(0, color="k", lw=0.4)
        for a in ("plain", "solar"):
            for e in ev[a].filter(pl.col("primary") == site).iter_rows(named=True):
                end = hi_i if e["end"] < 0 else e["end"]
                if e["units"] >= 3 and e["start"] < hi_i and end > lo_i:
                    bot.axvspan(
                        at(max(e["start"], lo_i)),
                        at(min(end, hi_i)),
                        ymin=0 if a == "plain" else 0.5,
                        ymax=0.5 if a == "plain" else 1,
                        color=COLOURS[a],
                        alpha=0.15,
                    )
        bot.set_ylabel(f"{site} residual (MW)")
        bot.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].set_xlim(np.datetime64(lo), np.datetime64(hi))
    fig.tight_layout()
    save(fig, "figure_event_series_D5_D16")


def main() -> None:
    """Run the requested part of the report."""
    part = sys.argv[1]
    inp = build()
    if part in ("gain", "all"):
        gain_part(inp=inp)
    if part in ("events", "all"):
        events_part(inp=inp)
    if part in ("recall", "all"):
        recall_part()
    if part in ("scenarios", "all"):
        scenarios_part(inp=inp)
    if part in ("figures", "all"):
        figures_part(inp=inp)


if __name__ == "__main__":
    main()

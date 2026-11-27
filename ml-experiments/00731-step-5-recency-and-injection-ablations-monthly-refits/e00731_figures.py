# ruff: noqa: E501, D103, ANN001, ANN201
"""Experiment 731: figures and the tails table (SVG, public; CSV, private) from the stored fits.

README. This script draws the figures of experiment 731 after the fits and the scoring have run.
It relies on `e00731_score` (paths, the 677 scorer module, the site map), 677
(`e00677_score.seed_mean`), 668 (`e00668_score_lib`, `e00668_slices_primaries.event_sets`), and on
the files that `e00731_k1.py` writes: `arms/k1_<ARM>_sp1/work/history/*.json` (the per-epoch loss
history of every fine-tune) and `work/worlds/<month>.npz` (the injected shifts of every month).
PYTHONPATH: source `ml-experiments/00524*/env.sh` from the worktree root; set `DATA_PATH_INTERNAL`.
Usage: `python e00731_figures.py [--out DIR] [--only loss worlds forecasts events tails]`.

Figures (Altair, `dataviz` skill: `plotting`'s main data colours, blue and orange for the two
conditions, a direction-of-better line in every subtitle, a labelled zero or reference rule):

- `loss_curves.svg`: training and validation negative log-likelihood against the epoch, one panel per
  arm, for one refit month and seed 0, with the stored (selected) epoch marked. The training curve is
  the unweighted mean of the training batches, with dropout on; the validation curve is the unweighted
  mean that early stopping reads;
- `injected_worlds_<month>.svg`: the 24-hour lag input of one Primary with and without the injected
  shifts of that refit month (arm R0's world);
- `forecast_examples.svg`: a day-ahead and a 14-day forecast of one Primary from one origin;
- `event_<n>_<label>.svg`: at least 6 load-shift events, observed power and the 24 to 48 hours-ahead
  median of R0 and of the stored injection-free arm, with the caveat that the clean-lag rows may hold
  events that no label covers;
- `tails.csv` and the printed table: for each arm, the share of observations at or below the forecast
  quantiles at levels 0.01, 0.05, 0.95, and 0.99 (nominal level is the quantile's own level), and the
  pinball loss at those levels in points of capacity.

Only Primaries appear, so the figures carry no NGED generator series.
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00731_score as S  # noqa: N812  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BLUE: Final = "#306BFF"
ORANGE: Final = "#FF4901"
SKY: Final = "#10C5F7"
BLACK: Final = "#222222"
WIDTH: Final = 640
LOSS_MONTH: Final = "2026-01"
WORLD_MONTHS: Final = ("2026-01", "2026-07")
EXAMPLE_PRIMARY: Final = "D5"
EXAMPLE_ORIGIN: Final = datetime(2026, 1, 14, 12, tzinfo=UTC)
CAVEAT: Final = "Events with no label may sit in the lag inputs of the plotted rows."
ARM_TEXT: Final = {
    "R0": "R0 (replay of arm 9a)",
    "W0": "W0 (equal weights)",
    "W30": "W30 (half-life 30 days)",
    "W270": "W270 (half-life 270 days)",
    "WP": "WP (placebo: ages permuted)",
    "IP": "IP (events with depth zero)",
    "B0": "B0 (equal weights, no events)",
}


def wrap(text: str, *, width: int = 100) -> list[str]:
    """Split `text` into lines of at most `width` characters, at spaces."""
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


def site_map():
    import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]

    return pl.read_csv(next(s668.R.glob("00521-*/site_map.csv"))).sort("primary")


def loss_curves(*, out: Path, month: str = LOSS_MONTH, seed: int = 0) -> None:
    panels = []
    stored = []
    for arm, text in ARM_TEXT.items():
        path = (
            S.ROOT_731 / "arms" / f"k1_{arm}_sp1" / "work" / "history" / f"{month}_d0_s{seed}.json"
        )
        if not path.exists():
            print("missing history", path)
            continue
        h = json.loads(path.read_text())
        rows = []
        for i, r in enumerate(h["history"], start=1):
            rows.append(
                {
                    "epoch": i,
                    "loss": r["train_nll_unweighted"],
                    "series": "Training (unweighted, dropout on)",
                }
            )
            rows.append(
                {
                    "epoch": i,
                    "loss": r["val_nll"],
                    "series": "Validation (what early stopping reads)",
                }
            )
        df = pl.DataFrame(rows)
        limit = h.get("max_epochs")
        if limit is not None and h["epochs_run"] >= limit:
            stop = f"ran to the {limit}-epoch limit"
        else:
            stop = f"early stopping ended the fit after epoch {h['epochs_run']}"
        stored.append(
            {
                "arm": arm,
                "stored_epoch": h["stored_epoch"],
                "epochs_run": h["epochs_run"],
                "stop": stop,
            }
        )
        lines = (
            alt.Chart(df)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1, format="d")),
                y=alt.Y("loss:Q", title="NLL (lower is better)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=[
                            "Training (unweighted, dropout on)",
                            "Validation (what early stopping reads)",
                        ],
                        range=[BLUE, ORANGE],
                    ),
                    legend=alt.Legend(orient="top", title=None) if not panels else None,
                ),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"epoch": [h["stored_epoch"]]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=BLACK, strokeDash=[4, 3])
            .encode(x="epoch:Q")
        )
        label = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame(
                    {"epoch": [h["stored_epoch"]], "text": [f"stored epoch {h['stored_epoch']}"]}
                )
            )
            .mark_text(align="left", dx=4, dy=-4, color=BLACK, baseline="top")
            .encode(x="epoch:Q", y=alt.value(4), text="text:N")
        )
        panels.append(
            (lines + rule + label).properties(width=WIDTH - 90, height=110, title=f"{text}: {stop}")
        )
    if not panels:
        return
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Training and validation loss by epoch, one fold and one seed per arm",
            subtitle=wrap(
                f"Refit month {month}, seed {seed}. Negative log-likelihood of the Gaussian mixture, lower is better. The dashed line marks the epoch whose weights are stored; each panel title says why the fit ended."
            ),
        )
    )
    chart.save(out / "loss_curves.svg")
    pl.DataFrame(stored).write_csv(out / "loss_curves_stored_epochs.csv")
    print("loss curves: stored epochs", stored)


def worlds(*, out: Path, months=WORLD_MONTHS, primary: str = EXAMPLE_PRIMARY) -> None:
    lib = S.sc.lib
    site = site_map()
    ids = sorted(site["time_series_id"].to_list())
    capacity = (
        pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
        .filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")["capacity"]
        .to_numpy()
    )
    grid = lib.cleaned_grid(ids=ids)
    row = ids.index(int(site.filter(pl.col("label") == primary)["time_series_id"][0]))
    share = grid[row] / capacity[row]
    panels = []
    for month in months:
        f = S.ROOT_731 / "arms" / "k1_R0_sp1" / "work" / "worlds" / f"{month}.npz"
        if not f.exists():
            print("missing world", f)
            continue
        w = np.load(f)
        shift = w["shift"][int(site.filter(pl.col("label") == primary)["primary"][0])]
        first = int(w["first"])
        nz = np.flatnonzero(shift)
        if not len(nz):
            continue
        lo, hi = max(0, int(nz[0]) - 20 * 48), min(len(shift), int(nz[0]) + 70 * 48)
        t = np.arange(lo, hi) + first
        # `build_month_syn` adds the shift at the grid time of the lagged observation itself
        # (the label of a row is `shift[primary, lag index]`), so both series are plotted on the
        # time of the lagged observation
        base = share[t] * 100.0
        df = pl.DataFrame(
            {
                "day": (np.arange(lo, hi) - lo) / 48.0,
                "Without the shifts (observed)": base,
                "With the injected shifts": base + shift[lo:hi] * 100.0,
            }
        ).unpivot(index="day", variable_name="lag input", value_name="share")
        start = lib.GRID_START + timedelta(minutes=30 * int(t[0]))
        panels.append(
            alt.Chart(df)  # ty: ignore[unresolved-attribute]
            .mark_line(clip=True)
            .encode(
                x=alt.X(
                    "day:Q", title="Days since the window start (time of the lagged observation)"
                ),
                y=alt.Y(
                    "share:Q",
                    title="24-hour lag input (% of capacity)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "lag input:N",
                    scale=alt.Scale(
                        domain=["Without the shifts (observed)", "With the injected shifts"],
                        range=[BLUE, ORANGE],
                    ),
                    legend=alt.Legend(orient="top", title=None) if not panels else None,
                ),
            )
            .properties(
                width=WIDTH - 90,
                height=150,
                title=f"Refit {month}: window starts {start:%Y-%m-%d}; {int(w['n_events'])} events over all Primaries",
            )
        )
    if panels:
        alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                f"Injected switching events in the lag input of Primary {primary}",
                subtitle=wrap(
                    "The fine-tune sees the orange series as its 24-hour lag input and the blue series is what was observed. Targets are never shifted. One world per refit month is shared by the three seeds; each shift is 15, 30, or 50% of capacity, up or down."
                ),
            )
        ).save(out / "injected_worlds.svg")
        print("worlds drawn for", months)


def truth_frame():
    lib = S.sc.lib
    return pl.read_parquet(lib.FOLDER / "ground_truth_primaries.parquet").select(
        "time_series_id", "valid_time", "power"
    )


def seed_mean_window(
    *,
    name: str,
    tsid: int,
    lo: datetime,
    hi: datetime,
    lead: tuple[float, float] | None,
    origin: datetime | None = None,
) -> pl.DataFrame:
    lf = pl.scan_parquet(S.ROOT_731 / "backtests" / f"{name}.parquet").filter(
        (pl.col("time_series_id") == tsid)
        & (pl.col("valid_time") >= lo)
        & (pl.col("valid_time") < hi)
    )
    if origin is not None:
        lf = lf.filter(pl.col("power_fcst_init_time") == origin)
    if lead is not None:
        hours = (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
        lf = lf.filter((hours >= lead[0]) & (hours < lead[1]))
    return (
        lf.group_by("valid_time")
        .agg(pl.col(["q05", "q50", "q95"]).mean())
        .sort("valid_time")
        .collect()
    )


def forecasts(*, out: Path) -> None:
    site = site_map()
    tsid = int(site.filter(pl.col("label") == EXAMPLE_PRIMARY)["time_series_id"][0])
    truth = truth_frame().filter(pl.col("time_series_id") == tsid)
    series = {"R0": ("arm_R0_sp1", BLUE), "B0": ("arm_B0_sp1", ORANGE)}
    panels = []
    for title, hours in (("Day ahead (first 24 hours)", 24), ("14 days ahead", 24 * 14)):
        hi = EXAMPLE_ORIGIN + timedelta(hours=hours)
        obs = truth.filter((pl.col("valid_time") >= EXAMPLE_ORIGIN) & (pl.col("valid_time") < hi))
        obs_df = pl.DataFrame(
            {
                "time": obs["valid_time"].to_list(),
                "mw": obs["power"].to_list(),
                "series": "Observed",
            }
        )
        layers = [
            alt.Chart(obs_df)  # ty: ignore[unresolved-attribute]
            .mark_line(color=BLACK, clip=True)
            .encode(
                x=alt.X("time:T", title="Valid time (UTC)"),
                y=alt.Y("mw:Q", title="Power (MW)", scale=alt.Scale(zero=False)),
            )
        ]
        for name, colour in series.values():
            f = seed_mean_window(
                name=name, tsid=tsid, lo=EXAMPLE_ORIGIN, hi=hi, lead=None, origin=EXAMPLE_ORIGIN
            )
            if not f.height:
                continue
            df = pl.DataFrame(
                {
                    "time": f["valid_time"].to_list(),
                    "q05": f["q05"].to_list(),
                    "q50": f["q50"].to_list(),
                    "q95": f["q95"].to_list(),
                }
            )
            layers.append(
                alt.Chart(df)  # ty: ignore[unresolved-attribute]
                .mark_area(opacity=0.18, color=colour)
                .encode(x="time:T", y="q05:Q", y2="q95:Q")
            )
            layers.append(alt.Chart(df).mark_line(color=colour).encode(x="time:T", y="q50:Q"))  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(*layers).properties(width=WIDTH - 90, height=170, title=title))
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            f"Forecasts of Primary {EXAMPLE_PRIMARY} from the origin of {EXAMPLE_ORIGIN:%Y-%m-%d} 12:00 UTC",
            subtitle=wrap(
                "Black: observed power. Blue: arm R0, median line and 5 to 95% band. Orange: arm B0 (equal weights, no injected events). Seed mean of three seeds; the closer the median to the black line, the better."
            ),
        )
    ).save(out / "forecast_examples.svg")
    print("forecast examples drawn")


def events(*, out: Path, n_min: int = 6) -> None:
    import e00668_slices_primaries as s668  # ty: ignore[unresolved-import]

    lib = S.sc.lib
    site = site_map()
    sets = s668.event_sets(site=site)
    table = pl.concat(
        [
            sets["verified transfer"].select("label", "start", "end"),
            sets["event table, over 7 days, 2 units or more"].select("label", "start", "end"),
        ]
    )
    table = table.with_columns(length=pl.col("end") - pl.col("start")).sort(
        "length", descending=True
    )
    year_lo, year_hi = datetime(2025, 11, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC)

    def at(idx: int) -> datetime:
        return lib.GRID_START + timedelta(minutes=30 * int(idx))

    drawn = 0
    truth_all = truth_frame()
    for ev in table.iter_rows(named=True):
        tsid = int(site.filter(pl.col("label") == ev["label"])["time_series_id"][0])
        lo, hi = (
            max(year_lo, at(ev["start"]) - timedelta(days=10)),
            min(year_hi, at(ev["end"]) + timedelta(days=10)),
        )
        if hi - lo < timedelta(days=5):
            continue
        obs = truth_all.filter(
            (pl.col("time_series_id") == tsid)
            & (pl.col("valid_time") >= lo)
            & (pl.col("valid_time") < hi)
        )
        obs_df = pl.DataFrame({"time": obs["valid_time"].to_list(), "mw": obs["power"].to_list()})
        layers = [
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame({"a": [max(at(ev["start"]), lo)], "b": [min(at(ev["end"]), hi)]})
            )
            .mark_rect(color="#DDDDDD", opacity=0.6)
            .encode(x="a:T", x2="b:T"),
            alt.Chart(obs_df)  # ty: ignore[unresolved-attribute]
            .mark_line(color=BLACK, clip=True)
            .encode(
                x=alt.X("time:T", title="Valid time (UTC)"),
                y=alt.Y("mw:Q", title="Power (MW)", scale=alt.Scale(zero=False)),
            ),
        ]
        for name, colour in (("arm_R0_sp1", BLUE), ("stored_5wo_sp1", ORANGE)):
            f = seed_mean_window(name=name, tsid=tsid, lo=lo, hi=hi, lead=(24.0, 48.0))
            df = pl.DataFrame({"time": f["valid_time"].to_list(), "mw": f["q50"].to_list()})
            layers.append(
                alt.Chart(df)  # ty: ignore[unresolved-attribute]
                .mark_line(color=colour, opacity=0.9, clip=True)
                .encode(x="time:T", y="mw:Q")
            )
        alt.layer(*layers).properties(
            width=WIDTH - 90,
            height=190,
            title=alt.TitleParams(
                f"Event {drawn + 1}: Primary {ev['label']}, the grey band marks the event ({(ev['end'] - ev['start']) / 48:.0f} days)",
                subtitle=wrap(
                    f"Black: observed power. Blue: median forecast 24 to 48 hours ahead of arm R0 (trained with injected events). Orange: the same of the stored arm without injected events. Seed mean. {CAVEAT}"
                ),
            ),
        ).save(out / f"event_{drawn + 1:02d}_{ev['label']}.svg")
        drawn += 1
        if drawn >= 12:
            break
    assert drawn >= n_min or drawn == table.height, ("fewer than 6 event plots", drawn)
    print("event plots drawn:", drawn)


def tails(*, out: Path) -> None:
    """Write the tails table on the rows of 677's joint pass, with the scorer's own observed power.

    The rows are the 3,879,313 shared rows of pass A (`e00731_score.shared_keys`), and the observed
    power is the cleaned grid that `e00677_score.main` scores against, not the raw ground truth.
    """
    lib = S.sc.lib
    keys = S.shared_keys(paths=[p for _, p in S.ladder_systems()])
    assert len(keys) == S.EXPECTED_ROWS, len(keys)
    ids = sorted(site_map()["time_series_id"].to_list())
    cap_by_id = (
        pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
        .filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")["capacity"]
        .to_numpy()
        .astype(np.float64)
    )
    grid = lib.cleaned_grid(ids=ids)
    levels = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
    rows = []
    for name, text in [
        *((f"arm_{a}_sp1", a) for a in S.ARMS),
        ("stored_9a_sp1", "9a"),
        ("stored_5wo_sp1", "5wo"),
    ]:
        if not (S.ROOT_731 / "backtests" / f"{name}.parquet").exists():
            continue
        f = S.sc.seed_mean(name=name, seeds=S.SEEDS)
        vidx = lib.grid_index(times=f["valid_time"])
        oidx = lib.grid_index(times=f["power_fcst_init_time"])
        key = S.key_of(tsid=f["time_series_id"].to_numpy(), origin=oidx, valid=vidx)
        keep = np.isin(key, keys)
        assert int(keep.sum()) == S.EXPECTED_ROWS, (name, int(keep.sum()))
        row = np.searchsorted(ids, f["time_series_id"].to_numpy())[keep]
        y = grid[row, vidx[keep]]
        cap = cap_by_id[row]
        rec = {"arm": text, "half_hours": int(keep.sum())}
        for col, tau in levels.items():
            q = f[col].to_numpy().astype(np.float64)[keep]
            rec[f"share_below_{col}"] = float((y <= q).mean())
            d = y - q
            rec[f"pinball_{col}"] = float(
                (np.maximum(tau * d, (tau - 1.0) * d) / cap * 100.0).mean()
            )
        rows.append(rec)
        print(rec, flush=True)
    if rows:
        df = pl.DataFrame(rows)
        df.write_csv(out / "tails.csv")
        print(df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=S.ROOT_731 / "figures")
    parser.add_argument(
        "--only",
        nargs="+",
        default=["loss", "worlds", "forecasts", "events", "tails"],
        choices=["loss", "worlds", "forecasts", "events", "tails"],
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if "loss" in args.only:
        loss_curves(out=args.out)
    if "worlds" in args.only:
        worlds(out=args.out)
    if "forecasts" in args.only:
        forecasts(out=args.out)
    if "events" in args.only:
        events(out=args.out)
    if "tails" in args.only:
        tails(out=args.out)


if __name__ == "__main__":
    main()

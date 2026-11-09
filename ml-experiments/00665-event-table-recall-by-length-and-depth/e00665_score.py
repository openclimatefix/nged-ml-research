"""Re-score the stored outputs of experiments 653 and 658 by event length and depth.

DATA_PATH_INTERNAL=... PYTHONPATH=../00653-edge-state-event-table-penalised-segmentation \
    uv run --no-sync python e00665_score.py injected|real
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY, E580, GRID_LENGTH, SCORED_FROM, SERIES_IDS, WORK, Panel
from e00653_inject import OFFSET_2019
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E658: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00658-edge-state-event-table-neural-network"
E254: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00254-lsc-alarm-precision-against-full-log-with-chance-base-rate"
)
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00665-event-table-recall-by-length-and-depth"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
KINDS: Final[list[str]] = ["edge", "solo", "permanent_transfer", "permanent_shift"]
CELLS: Final[list[str]] = [
    "long, deep",
    "long, shallow",
    "medium, deep",
    "medium, shallow",
    "short, deep",
    "short, shallow",
]
"""The six types, the long and deep cell first."""
DEEP_UNITS: Final[float] = 8.0
MIN_UNITS: Final[float] = 2.0
OTHER: Final[tuple[str, str]] = ("none", "unmeasured")
"""Rows in no cell: under 2 noise units, and depth not measurable (too few readings)."""
MIN_READINGS: Final[int] = 48
"""Half-hours with a reading needed in each window to measure the depth of a logged shift."""
YEARS: Final[float] = (GRID_LENGTH - SCORED_FROM) / (365.25 * DAY)
N_BOOT: Final[int] = 2000
METHODS: Final[tuple[str, str]] = ("653 default", "658 second look")


def length_of(*, days: float | None) -> str:
    """Length class of an event; `None` or a negative number means it has no end."""
    if days is None or days < 0 or days > 7:
        return "long"
    return "medium" if days > 1 else "short"


def cell_of(*, days: float | None, units: float) -> str:
    """The six-type cell of an event."""
    return f"{length_of(days=days)}, {'deep' if units >= DEEP_UNITS else 'shallow'}"


def wrec(*, df: pl.DataFrame, col: str) -> float:
    """MW-weighted share of rows with `col` true."""
    w = df["amp_mw"].abs()
    if not df.height or float(w.sum()) == 0.0:
        return float("nan")
    return float((df[col].cast(pl.Float64) * w).sum()) / float(w.sum())


def with_cells(*, df: pl.DataFrame) -> pl.DataFrame:
    """Keep injected positive events of 2 noise units or more and add the cell column."""
    d = df.filter(pl.col("kind").is_in(KINDS) & (pl.col("units") >= MIN_UNITS))
    cells = [
        cell_of(days=r["days"], units=r["units"])
        for r in d.select("days", "units").iter_rows(named=True)
    ]
    return d.with_columns(
        cell=pl.Series(cells),
        right_class=pl.col("right_class").cast(pl.Float64),
        found=pl.col("found").cast(pl.Boolean),
    )


def load_injected() -> dict[str, list[pl.DataFrame]]:
    """Injected truth rows (default spec) per method, one frame per seed."""
    t653 = pl.read_parquet(WORK / "injected_truth_scored.parquet").filter(
        (pl.col("spec") == "default") & (pl.col("arm") == "masked")
    )
    out: dict[str, list[pl.DataFrame]] = {METHODS[0]: [with_cells(df=t653)], METHODS[1]: []}
    for s in SEEDS:
        costs = json.loads((E658 / f"costs_v2main_seed{s}.json").read_text())
        d = pl.read_parquet(E658 / f"injected_v2main_seed{s}.parquet").filter(
            (pl.col("arm") == "net")
            & (pl.col("spec") == "default")
            & (pl.col("on") == costs["on"])
            & (pl.col("boost") == costs["boost"])
            & (pl.col("cutoff") == costs["cutoff"])
        )
        out[METHODS[1]].append(with_cells(df=d))
    return out


def load_pulses() -> dict[str, list[pl.DataFrame]]:
    """Injected pulses of 2 noise units or more per method, one frame per seed."""
    t653 = pl.read_parquet(WORK / "injected_truth_scored.parquet").filter(
        (pl.col("spec") == "default") & (pl.col("arm") == "masked")
    )
    out = {METHODS[0]: [t653], METHODS[1]: []}
    for s in SEEDS:
        costs = json.loads((E658 / f"costs_v2main_seed{s}.json").read_text())
        out[METHODS[1]].append(
            pl.read_parquet(E658 / f"injected_v2main_seed{s}.parquet").filter(
                (pl.col("arm") == "net")
                & (pl.col("spec") == "default")
                & (pl.col("on") == costs["on"])
                & (pl.col("boost") == costs["boost"])
                & (pl.col("cutoff") == costs["cutoff"])
            )
        )
    return {
        k: [f.filter((pl.col("kind") == "pulse") & (pl.col("units") >= MIN_UNITS)) for f in v]
        for k, v in out.items()
    }


def timing(*, df: pl.DataFrame) -> dict[str, float]:
    """Start within 2 hours and end within 6 hours (misses fail), and median errors when found."""
    s = df["start_err_hh"].to_numpy()
    start_ok = float(np.mean(np.isfinite(s) & (np.abs(s) <= 4))) if df.height else float("nan")
    ends = df.filter(pl.col("end") > 0)
    e = ends["end_err_hh"].to_numpy()
    end_ok = float(np.mean(np.isfinite(e) & (np.abs(e) <= 12))) if ends.height else float("nan")
    f = df.filter(pl.col("found"))
    fs = np.abs(f["start_err_hh"].to_numpy())
    fe = np.abs(f.filter(pl.col("end") > 0)["end_err_hh"].to_numpy())
    return {
        "start_within_2h": start_ok,
        "end_within_6h": end_ok,
        "median_start_err_h": float(np.nanmedian(fs)) / 2
        if np.isfinite(fs).any()
        else float("nan"),
        "median_end_err_h": float(np.nanmedian(fe)) / 2 if np.isfinite(fe).any() else float("nan"),
    }


def cell_stats(*, frames: list[pl.DataFrame], cell: str) -> dict[str, float]:
    """Counts, recall, pairing, and timing of one cell, averaged over the seeds' frames."""
    parts = [f.filter(pl.col("cell") == cell) for f in frames]
    edges = [p.filter(pl.col("edge_truth")) for p in parts]
    tm = [timing(df=p) for p in parts]
    return {
        "n_events": float(parts[0].height),
        "n_edge": float(edges[0].height),
        "found_mw": float(np.nanmean([wrec(df=p, col="found") for p in parts])),
        "found_unweighted": float(np.mean([p["found"].mean() for p in parts])),
        "paired_mw": float(np.nanmean([wrec(df=e, col="right_class") for e in edges]))
        if edges[0].height
        else float("nan"),
        **{k: float(np.nanmean([t[k] for t in tm])) for k in tm[0]},
    }


def boot_intervals(*, frames: list[pl.DataFrame]) -> dict[str, dict[str, tuple[float, float]]]:
    """95% intervals per cell for found and paired, resampling whole worlds (seeds averaged)."""
    rng = np.random.default_rng(3)
    worlds = sorted(frames[0]["world"].unique().to_list())
    agg: dict[tuple[str, int, int], tuple[float, float, float, float]] = {}
    for si, f in enumerate(frames):
        for (cell, world), g in f.group_by(["cell", "world"]):
            e = g.filter(pl.col("edge_truth"))
            w = g["amp_mw"].abs()
            we = e["amp_mw"].abs()
            agg[(cell, si, world)] = (
                float((g["found"].cast(pl.Float64) * w).sum()),
                float(w.sum()),
                float((e["right_class"] * we).sum()) if e.height else 0.0,
                float(we.sum()),
            )
    draws = rng.choice(len(worlds), size=(N_BOOT, len(worlds)))
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for cell in CELLS:
        fnd = np.zeros(N_BOOT)
        prd = np.zeros(N_BOOT)
        for si in range(len(frames)):
            tab = np.array([agg.get((cell, si, w), (0.0, 0.0, 0.0, 0.0)) for w in worlds])
            sel = tab[draws]
            with np.errstate(invalid="ignore", divide="ignore"):
                fnd += sel[..., 0].sum(1) / sel[..., 1].sum(1) / len(frames)
                prd += sel[..., 2].sum(1) / sel[..., 3].sum(1) / len(frames)
        out[cell] = {
            "found": tuple(np.round(np.nanpercentile(fnd, [2.5, 97.5]), 3).tolist()),  # type: ignore[dict-item]
            "paired": tuple(np.round(np.nanpercentile(prd, [2.5, 97.5]), 3).tolist()),  # type: ignore[dict-item]
        }
    return out


def gate(*, frames: dict[str, list[pl.DataFrame]]) -> None:
    """Reproduce the stored headline recalls (events of 4+ units lasting 1+ day)."""
    for name, fr in frames.items():
        f_, p_ = [], []
        for d in fr:
            b = d.filter((pl.col("units") >= 4) & ((pl.col("days") >= 1) | (pl.col("days") < 0)))
            f_.append(wrec(df=b, col="found"))
            p_.append(wrec(df=b.filter(pl.col("edge_truth")), col="right_class"))
        print(f"GATE {name}: found {np.mean(f_):.3f} paired {np.nanmean(p_):.3f}")


def injected() -> None:
    """Per-cell tables of the injected worlds."""
    OUT.mkdir(exist_ok=True)
    frames = load_injected()
    gate(frames=frames)
    res: dict[str, dict[str, dict[str, object]]] = {}
    for name, fr in frames.items():
        iv = boot_intervals(frames=fr)
        res[name] = {}
        print(f"\n== {name} ({len(fr)} frame(s))")
        for cell in CELLS:
            st = cell_stats(frames=fr, cell=cell)
            res[name][cell] = {**st, "interval": iv[cell]}
            few = st["n_events"] < 20 or st["n_edge"] < 10
            print(
                f"{cell:16s} n={st['n_events']:4.0f} edges={st['n_edge']:3.0f}"
                f"{' (too few)' if few else ''} found MW {st['found_mw']:.3f} {iv[cell]['found']}"
                f" unweighted {st['found_unweighted']:.3f} | paired {st['paired_mw']:.3f}"
                f" {iv[cell]['paired']} | start<=2h {st['start_within_2h']:.2f}"
                f" end<=6h {st['end_within_6h']:.2f} | median err (found) start"
                f" {st['median_start_err_h']:.1f} h end {st['median_end_err_h']:.1f} h"
            )
        if len(fr) > 1:
            print("per seed found MW by cell:")
            for cell in CELLS:
                print(
                    f"  {cell:16s}",
                    [round(wrec(df=f.filter(pl.col("cell") == cell), col="found"), 3) for f in fr],
                )
    print("\n== Pulses (30 minutes to 2.5 hours), kept out of the six cells")
    pulses: dict[str, dict[str, float]] = {}
    for name, raw in load_pulses().items():
        pulses[name] = {
            "n": float(raw[0].height),
            "found_mw": float(np.nanmean([wrec(df=p, col="found") for p in raw])),
        }
        print(f"{name:16s} n={raw[0].height} found MW {pulses[name]['found_mw']:.3f}")
    res["pulses"] = pulses  # type: ignore[assignment]
    (OUT / "injected_by_cell.json").write_text(json.dumps(res, indent=1))


def depth_units(*, resid: np.ndarray, sigma: float, onset: int, end: int | None) -> float:
    """Measured step in noise units: 3 days after onset (or to the end) against 3 days before."""
    stop = onset + 3 * DAY if end is None else min(onset + 3 * DAY, max(end, onset + 1))
    after = resid[onset:stop]
    before = resid[max(0, onset - 3 * DAY) : onset]
    if np.isfinite(after).sum() < MIN_READINGS or np.isfinite(before).sum() < MIN_READINGS:
        return float("nan")
    return abs(float(np.nanmedian(after)) - float(np.nanmedian(before))) / sigma


def real() -> None:  # noqa: PLR0915, PLR0912 - one linear printout
    """Real-record tables: events called, logged shifts matched, and the event-free stratum."""
    panel = Panel()
    resid = panel.mask_common_mode(residual=panel.residual)
    sigma = json.loads((WORK / "noise_units.json").read_text())["sigma_block_mw"]
    n_prim = len(panel.sites)
    names = dict(zip(SERIES_IDS, panel.sites, strict=True))
    res: dict[str, object] = {}
    scored_years = sum(
        int(np.isfinite(panel.residual[x][SCORED_FROM:GRID_LENGTH]).sum()) for x in panel.sites
    ) / (DAY * 365.25)
    print(f"Scored Primary-years with a reading: {scored_years:.2f} (nominal {n_prim * YEARS:.2f})")
    res["scored_primary_years"] = scored_years

    methods: dict[str, pl.DataFrame] = {}
    e653 = pl.read_parquet(WORK / "real_events_masked.parquet").filter(
        pl.col("start") >= SCORED_FROM
    )
    methods["653 default"] = e653.select("primary", "start", pl.col("end").alias("end_"), "units")
    for s in SEEDS:
        e = pl.read_parquet(E658 / f"real_events_v2main_seed{s}.parquet").filter(
            pl.col("start") >= SCORED_FROM
        )
        methods[f"658 second look seed {s}"] = e.select(
            "primary", "start", pl.col("end").alias("end_"), "units"
        )
    for tag, f in (("LR_b20", "alarms_LR_b20"), ("RO-NC", "alarms_RO_b80")):
        a = (
            pl.read_parquet(E254 / f"{f}.parquet")
            .with_columns(primary=pl.col("series").replace_strict(names, default=None))
            .filter((pl.col("onset") >= SCORED_FROM) & pl.col("primary").is_not_null())
        )
        a = a.unique(subset=["primary", "onset"])
        rows = []
        for r in a.iter_rows(named=True):
            rs = resid[r["primary"]]
            rows.append(
                {
                    "primary": r["primary"],
                    "start": r["onset"],
                    "end_": r["end"],
                    "units": depth_units(
                        resid=rs, sigma=sigma[r["primary"]], onset=r["onset"], end=r["end"]
                    ),
                }
            )
        methods[tag] = pl.DataFrame(
            rows,
            schema={"primary": pl.String, "start": pl.Int64, "end_": pl.Int64, "units": pl.Float64},
        )

    for m, d in methods.items():
        assert d["start"].max() < GRID_LENGTH, m  # no row from 2026-07-01 onward

    def days_of(*, start: int, end: int) -> float | None:
        return None if end < 0 else (end - start) / DAY

    print("== Events (or alarms) per method in the scored years, per Primary-year, by cell")
    print("   (events under 2 noise units, or with unmeasured depth, fall in no cell)")
    per_cell: dict[str, dict[str, float]] = {}
    for m, d in methods.items():
        cells = [
            cell_of(days=days_of(start=r["start"], end=r["end_"]), units=r["units"])
            if np.isfinite(r["units"]) and r["units"] >= MIN_UNITS
            else ("unmeasured" if not np.isfinite(r["units"]) else "none")
            for r in d.iter_rows(named=True)
        ]
        d = d.with_columns(cell=pl.Series(cells, dtype=pl.String))  # noqa: PLW2901
        methods[m] = d
        per_cell[m] = {
            c: float((d["cell"] == c).sum()) / (n_prim * YEARS) for c in [*CELLS, *OTHER]
        }
        print(
            f"{m:26s} total {d.height:4d} ({d.height / scored_years:.2f}/yr) | "
            + " | ".join(f"{c} {int((d['cell'] == c).sum())} ({per_cell[m][c]:.2f})" for c in CELLS)
        )
    res["per_primary_year"] = per_cell
    print("\n== Same counts with depth measured in the residual panel for every method")
    measured: dict[str, dict[str, float]] = {}
    for m in ("653 default", "658 second look seed 0"):
        d = methods[m]
        um = [
            depth_units(
                resid=resid[r["primary"]],
                sigma=sigma[r["primary"]],
                onset=r["start"],
                end=None if r["end_"] < 0 else r["end_"],
            )
            for r in d.iter_rows(named=True)
        ]
        cm = [
            cell_of(days=days_of(start=r["start"], end=r["end_"]), units=u)
            if np.isfinite(u) and u >= MIN_UNITS
            else "other"
            for r, u in zip(d.iter_rows(named=True), um, strict=True)
        ]
        cnt = pl.Series(cm).value_counts()
        measured[m] = {
            c: float(cnt.filter(pl.col("") == c)["count"].sum()) / scored_years
            for c in [*CELLS, "other"]
        }
        print(
            m, " | ".join(f"{c} {v * scored_years:.0f} ({v:.2f})" for c, v in measured[m].items())
        )
    res["per_primary_year_measured_depth"] = measured

    lab = pl.read_parquet(E580 / "events_labelled_status.parquet").filter(
        (pl.col("label_length_days") > 1)
        & pl.col("label").is_in(panel.sites)
        & (pl.col("label_onset_idx") - OFFSET_2019 >= SCORED_FROM)
    )
    logged = []
    for r in lab.iter_rows(named=True):
        on = r["label_onset_idx"] - OFFSET_2019
        end = r["label_end_idx"] - OFFSET_2019
        u = depth_units(resid=resid[r["label"]], sigma=sigma[r["label"]], onset=on, end=end)
        logged.append(
            {
                "primary": r["label"],
                "onset": on,
                "days": r["label_length_days"],
                "units": u,
                "cell": cell_of(days=r["label_length_days"], units=u)
                if np.isfinite(u) and u >= MIN_UNITS
                else ("unmeasured" if not np.isfinite(u) else "none"),
            }
        )
    lg = pl.DataFrame(logged)
    print(f"\n== Logged shifts of more than 1 day, scored years: {lg.height}")
    print(lg.group_by("cell").len().sort("cell"))
    mat: dict[str, dict[str, list[int]]] = {}
    for m, d in methods.items():
        starts = {p: d.filter(pl.col("primary") == p)["start"].to_list() for p in panel.sites}
        out: dict[str, list[int]] = {}
        tot = [0, 0]
        for cell in [*CELLS, *OTHER]:
            sub = lg.filter(pl.col("cell") == cell)
            strict = sum(
                any(abs(x - r["onset"]) <= 4 for x in starts[r["primary"]])
                for r in sub.iter_rows(named=True)
            )
            loose = sum(
                any(abs(x - r["onset"]) <= 48 for x in starts[r["primary"]])
                for r in sub.iter_rows(named=True)
            )
            out[cell] = [sub.height, strict, loose]
            tot[0] += strict
            tot[1] += loose
        mat[m] = out
        print(
            f"{m:26s} strict/loose overall {tot} | "
            + " | ".join(f"{c}: {v[1]}/{v[2]} of {v[0]}" for c, v in out.items())
        )
    res["logged_matched"] = mat

    allab = pl.read_parquet(E580 / "events_labelled_status.parquet")
    spans = [
        (r["label"], r["label_onset_idx"] - OFFSET_2019, r["label_end_idx"] - OFFSET_2019)
        for r in allab.iter_rows(named=True)
    ]
    for r in pl.read_parquet(E580 / "events_verified.parquet").iter_rows(named=True):
        spans += [
            (r["primary_a"], r["onset_a_idx"] - OFFSET_2019, r["end_a_idx"] - OFFSET_2019),
            (r["primary_b"], r["onset_b_idx"] - OFFSET_2019, r["end_b_idx"] - OFFSET_2019),
        ]
    assert all(b >= a for _, a, b in spans)  # every span ends after it starts
    free_hh = 0
    busy_by: dict[str, np.ndarray] = {}
    for site in panel.sites:
        busy = np.zeros(GRID_LENGTH, bool)
        for lab_site, a, b in spans:
            if lab_site == site:
                busy[max(0, a - 14 * DAY) : max(0, b + 14 * DAY)] = True
        busy[:SCORED_FROM] = True
        busy_by[site] = busy
        free_hh += int((~busy & np.isfinite(panel.residual[site])).sum())
    free_years = free_hh / (DAY * 365.25)
    print(f"\n== Event-free stratum: {free_years:.1f} Primary-years of {scored_years:.1f}")
    ef: dict[str, dict[str, float]] = {}
    for m, d in methods.items():
        row = dict.fromkeys([*CELLS, *OTHER], 0.0)
        for r in d.iter_rows(named=True):
            end = GRID_LENGTH if r["end_"] < 0 else r["end_"]
            b = busy_by[r["primary"]]
            seg = b[r["start"] : max(end, r["start"] + 1)]
            if not seg.any():
                row[r["cell"]] += 1
        ef[m] = {c: v / free_years for c, v in row.items()}
        print(f"{m:26s} " + " | ".join(f"{c} {v:.2f}" for c, v in ef[m].items()))
    res["event_free_per_primary_year"] = ef
    res["free_primary_years"] = free_years
    OUT.mkdir(exist_ok=True)
    (OUT / "real_by_cell.json").write_text(json.dumps(res, indent=1))
    for m in ("653 default", "LR_b20"):
        d = methods[m]
        starts = {p: d.filter(pl.col("primary") == p)["start"].to_list() for p in panel.sites}
        s = sum(
            any(abs(x - on) <= 4 for x in starts[p])
            for p, on in zip(lg["primary"], lg["onset"], strict=True)
        )
        loose = sum(
            any(abs(x - on) <= 48 for x in starts[p])
            for p, on in zip(lg["primary"], lg["onset"], strict=True)
        )
        print(f"GATE {m}: all events strict/loose {s}/{loose} (stored: 653 8/17, LR_b20 19/24)")


if __name__ == "__main__":
    {"injected": injected, "real": real}[sys.argv[1]]()

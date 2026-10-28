# ruff: noqa: E501
"""Build Primary demand research default 3 at 168 to 240 hours at every Primary of experiment 377.

Rows are experiment 377's stage-2 forecasts with a lead above 168 hours and up to 240 hours,
joined to the member responses of `e00378_members.py`. For each fold month, the fits use the rows
of the trailing 365 days whose run had completed (origin plus 240 hours) before the month's first
origin, exactly as experiments 319 and 322 do, and the functions are theirs:

- **x1.** Experiment 319's two-number solar summary S, factors `(f, s)` per 24-hour lead band.
- **r319.** Experiment 319's blend with the manual heuristic's median, weight per lead band and
  season group, fitted on the rows of the fold's stage 2 (not on the old convex chain).
- **d3.** Default 3: r319 on half-hours with the sun above 5 degrees, x1 elsewhere. Below 168 hours
  the forecast is experiment 377's stage 2 unchanged (`e00378_assemble.py`).

Fits are pooled within a group: the signed-MW group (D5, D9, D10, D11, D14, D16) and the MVA-only
group (D2, D3, D4, D12, D19, D21, D22). D8 (donor spread), D15, and D18 do not enter any fit;
D8 takes the MVA-only fits and D15 and D18 take the signed-MW fits. A fit with fewer than 200
rows in any lead band falls back to the identity (no summary change, no blend), and the month is
listed in `fit_log.json`.

    PYTHONPATH=<folders of experiments 184, 202, 317, 319, 320, 377> E00377_SOURCE=control \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00378-*/e00378_arms.py
"""

import json
import os
from datetime import timedelta
from typing import Final

import e00319_arms as a319  # ty: ignore[unresolved-import]
import e00320_arms as a320  # ty: ignore[unresolved-import]
import e00377_common as c377  # ty: ignore[unresolved-import]
import e00378_common as common
import numpy as np
import polars as pl

QUANTILES: Final[tuple[str, ...]] = common.QUANTILES
FIT_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "mw": ("D5", "D9", "D10", "D11", "D14", "D16"),
    "mva": ("D2", "D3", "D4", "D12", "D19", "D21", "D22"),
}
APPLY_GROUP: Final[dict[str, str]] = {
    **{s: "mw" for s in FIT_GROUPS["mw"]},
    **{s: "mva" for s in FIT_GROUPS["mva"]},
    "D18": "mw",
    "D15": "mw",
    "D8": "mva",
}
SITES: Final[tuple[str, ...]] = tuple(APPLY_GROUP)
MIN_ROWS: Final[int] = 200


def arms_source(site: str) -> str:
    """Return the source a site's arms run under: D15 joins the signed-MW sites, whose fit it takes."""
    return "control" if site == "D15" else common.SOURCE_OF_SITE[site]


def stage_two_rows(*, site: str) -> pl.DataFrame:
    """Return experiment 377's stage-2 rows of a site with a lead in (168, 240] hours."""
    path = common.stage_path(site=site, stage="dp_preds_s2")
    if site == "D8":
        path = path.with_name("stack_d_donor_D8.parquet")
    return pl.read_parquet(path).filter(
        (pl.col("lead_hours") > common.STORED_HORIZON_HOURS)
        & (pl.col("lead_hours") <= common.HORIZON_HOURS)
    )


def load_rows(*, cache, source: str) -> tuple[pl.DataFrame, np.ndarray]:  # noqa: ANN001
    """Return the labelled long-lead rows of the sites in one source, and their (rows, 51) responses."""
    frames, deltas = [], []
    for site in (s for s in SITES if arms_source(s) == source):
        chain = stage_two_rows(site=site).with_columns(period=pl.lit("scored"))
        pieces = []
        for period in ("tuning", "scored"):
            path = common.MEMBERS_DIR / f"{site}_{period}.npz"
            if not path.exists():
                continue
            stored = np.load(path)
            keys = pl.DataFrame(
                {
                    "run": pl.Series(stored["run"]).cast(chain["run"].dtype),
                    "valid_time": pl.Series(stored["valid_time"]).dt.replace_time_zone("UTC"),
                    "n_members": stored["n_members"],
                    "row": np.arange(len(stored["run"])),
                    "period_members": pl.Series([period] * len(stored["run"])),
                }
            )
            part = chain.join(keys, on=["run", "valid_time"], how="inner", validate="1:1")
            deltas.append(stored["delta_m"][part["row"].to_numpy()].astype(np.float64))
            pieces.append(part.drop("row"))
        site_frame = pl.concat(pieces)
        if site_frame.height != chain.height:
            msg = f"{site}: {chain.height - site_frame.height} stage-2 rows have no member responses"
            raise ValueError(msg)
        index = np.searchsorted(
            cache.end_times, site_frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        elevation = cache.sites[site].elevation_deg[index]
        kind = np.where(elevation >= a319.DAYLIGHT_ELEVATION_DEG, "daylight", "other")
        frames.append(
            site_frame.with_columns(
                site=pl.lit(site),
                group=pl.lit(APPLY_GROUP[site]),
                kind=pl.Series(kind),
                heuristic_median=pl.Series(
                    a319.heuristic_median(cache=cache, site=site, frame=site_frame)
                ),
            )
        )
    return pl.concat(frames), np.concatenate(deltas)


def guarded_factors(*, frame: pl.DataFrame, delta: np.ndarray, rows: np.ndarray) -> list:
    """Return `a319.fit_factors`, or the identity `(1, 1)` per band when a band is thin."""
    sub = frame[rows]
    band = a319.common.long_band(lead_hours=sub["lead_hours"].to_numpy())
    day = ((sub["kind"] == "daylight") & (sub["n_members"] > 0)).to_numpy()
    if min(int((day & (band == b)).sum()) for b in range(3)) < MIN_ROWS:
        return [(1.0, 1.0)] * 3
    return a319.fit_factors(frame=frame, delta=delta, rows=rows)


def replay(
    *, frame: pl.DataFrame, delta: np.ndarray, in_fit: np.ndarray, scored: np.ndarray, label: str
) -> tuple[dict[str, np.ndarray], dict]:
    """Replay every fold month of the `scored` rows and return the arms' quantiles and the fit log."""
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    month = frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    usable = (~frame["switching_mask"] & frame["actual"].is_finite()).to_numpy()
    group = frame["group"].to_numpy()
    horizon = np.timedelta64(int(timedelta(hours=common.HORIZON_HOURS).total_seconds()), "s")
    q0 = frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
    out = {name: q0.copy() for name in ("x1", "r319", "d3")}
    log: dict = {}
    for m in sorted(set(month[scored])):
        for g in sorted(set(group)):
            rows_m = np.flatnonzero((month == m) & (group == g) & scored)
            if len(rows_m) == 0:
                continue
            first = origin[rows_m].min()
            completed = (
                (origin + horizon <= first)
                & (origin >= first - np.timedelta64(common.WINDOW_DAYS, "D"))
                & usable
                & in_fit
                & (group == g)
            )
            fit_rows = np.flatnonzero(completed)
            band = a319.common.long_band(lead_hours=frame["lead_hours"].to_numpy()[fit_rows])
            thin = len(fit_rows) == 0 or min(int((band == b).sum()) for b in range(3)) < MIN_ROWS
            if thin:
                factors = [(1.0, 1.0)] * 3
                weights = np.ones((3, 2))
            else:
                factors = guarded_factors(frame=frame, delta=delta, rows=fit_rows)
                weights = a319.fit_weights(frame=frame, rows=fit_rows)
            x1 = a319.apply_x1(frame=frame, delta=delta, rows=rows_m, factors=factors)
            w_table = a320._table_w(frame=frame, rows=rows_m, weights=weights)  # noqa: SLF001
            heuristic = frame["heuristic_median"].to_numpy()[rows_m]
            r319 = a320.apply_blend(base=x1, heuristic=heuristic, w=w_table)
            day = (frame["kind"] == "daylight").to_numpy()[rows_m][:, None]
            out["x1"][rows_m] = x1
            out["r319"][rows_m] = r319
            out["d3"][rows_m] = np.where(day, r319, x1)
            log[f"{label}/{g}/{m}"] = {
                "n_fit_rows": len(fit_rows),
                "identity_fallback": bool(thin),
                "factors": factors,
                "weights": weights.tolist(),
            }
            print(label, g, m, len(fit_rows), "identity" if thin else "fitted", flush=True)
    return out, log


def run_source(*, source: str) -> pl.DataFrame:
    """Replay every fold month for the sites of one source and return their rows with arm quantiles."""
    cache = c377.load_cache()
    frame, delta = load_rows(cache=cache, source=source)
    site = frame["site"].to_numpy()
    in_fit = np.array([s in FIT_GROUPS[APPLY_GROUP[s]] for s in site])
    out, log = replay(
        frame=frame, delta=delta, in_fit=in_fit, scored=np.ones(frame.height, dtype=bool), label=source
    )
    keep = frame.select(
        "site",
        "group",
        "run",
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        "actual",
        "switching_mask",
        "kind",
        "heuristic_median",
        *QUANTILES,
    ).rename({q: f"base_{q}" for q in QUANTILES})
    columns = [
        pl.Series(f"{name}_{q}", values[:, k].astype(np.float32))
        for name, values in out.items()
        for k, q in enumerate(QUANTILES)
    ]
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    keep = keep.with_columns(columns)
    keep.write_parquet(common.ARMS_DIR / f"long_lead_{source}.parquet")
    common.write_json_atomic(path=common.ARMS_DIR / f"fit_log_{source}.json", content=log)
    return keep


def main() -> None:
    """Run the sources named by `E00377_SOURCE` (one per process, because 377 reads it on import)."""
    source = os.environ["E00377_SOURCE"]
    keep = run_source(source=source)
    print(json.dumps({"source": source, "rows": keep.height}))


if __name__ == "__main__":
    main()

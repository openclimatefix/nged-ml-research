"""Apply experiment 202's two-number solar summary S at 0.5 to 168 hours at every Primary.

Experiment 377's stage 2 has no S at any lead, and default 1 has S below 168 hours (experiments
202 and 251). Rows are experiment 377's stage-2 rows with a lead of 168 hours or less, joined to the
member responses of `e00378_members.py --low`. For each fold month, the factors `(f, s)` per
calibration lead band are fitted by experiment 202's `fit_factors` on the daylight rows of the
trailing 365 days whose run had completed (origin plus 168 hours) before the month's first origin,
pooled within the group of `e00378_arms.py`; a band with fewer than 200 daylight rows falls back to
the identity. The result is applied with experiment 202's `apply_arm`.

    PYTHONPATH=<folders of experiments 184, 202, 319, 320, 377> E00377_SOURCE=control \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00378-*/e00378_arms_low.py
"""

import json
import os
from datetime import timedelta

import e00202_arms as a202  # ty: ignore[unresolved-import]
import e00377_common as c377  # ty: ignore[unresolved-import]
import e00378_arms as arms
import e00378_common as common
import numpy as np
import polars as pl
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours

QUANTILES = common.QUANTILES


def load_low(*, cache, source: str) -> tuple[pl.DataFrame, np.ndarray]:  # noqa: ANN001
    """Return the labelled rows of at most 168 hours of one source's sites and their responses."""
    frames, deltas = [], []
    for site in (s for s in arms.SITES if arms.arms_source(s) == source):
        path = common.stage_path(site=site, stage="dp_preds_s2")
        if site == "D8":
            path = path.with_name("stack_d_donor_D8.parquet")
        chain = pl.read_parquet(path).filter(pl.col("lead_hours") <= common.STORED_HORIZON_HOURS)
        pieces = []
        for period in ("tuning", "scored"):
            stored_path = common.MEMBERS_DIR / f"low_{site}_{period}.npz"
            if not stored_path.exists():
                continue
            stored = np.load(stored_path)
            keys = pl.DataFrame(
                {
                    "run": pl.Series(stored["run"]).cast(chain["run"].dtype),
                    "valid_time": pl.Series(stored["valid_time"]).dt.replace_time_zone("UTC"),
                    "n_members": stored["n_members"],
                    "row": np.arange(len(stored["run"])),
                }
            )
            part = chain.join(keys, on=["run", "valid_time"], how="inner", validate="1:1")
            deltas.append(stored["delta_m"][part["row"].to_numpy()].astype(np.float64))
            pieces.append(part.drop("row"))
        site_frame = pl.concat(pieces)
        if site_frame.height != chain.height:
            msg = (
                f"{site}: {chain.height - site_frame.height} stage-2 rows have no member responses"
            )
            raise ValueError(msg)
        index = np.searchsorted(
            cache.end_times, site_frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        elevation = cache.sites[site].elevation_deg[index]
        kind = np.where(elevation >= 5.0, "daylight", "other")
        frames.append(
            site_frame.with_columns(
                site=pl.lit(site), group=pl.lit(arms.APPLY_GROUP[site]), kind=pl.Series(kind)
            )
        )
    return pl.concat(frames), np.concatenate(deltas)


def main() -> None:
    """Replay every fold month and write the S-corrected quantiles at 168 hours or less."""
    source = os.environ["E00377_SOURCE"]
    cache = c377.load_cache()
    frame, delta = load_low(cache=cache, source=source)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    month = frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    usable = (~frame["switching_mask"] & frame["actual"].is_finite()).to_numpy()
    group = frame["group"].to_numpy()
    in_fit = np.array([s in arms.FIT_GROUPS[arms.APPLY_GROUP[s]] for s in frame["site"].to_numpy()])
    day = ((frame["kind"] == "daylight") & (frame["n_members"] > 0)).to_numpy()
    lead_band = lead_band_of_hours(lead_hours=frame["lead_hours"].to_numpy())
    horizon = np.timedelta64(int(timedelta(hours=common.STORED_HORIZON_HOURS).total_seconds()), "s")
    out = frame.select(list(QUANTILES)).to_numpy().astype(np.float64)
    log: dict = {}
    for m in sorted(set(month)):
        for g in sorted(set(group)):
            rows_m = np.flatnonzero((month == m) & (group == g))
            if len(rows_m) == 0:
                continue
            first = origin[rows_m].min()
            fit_rows = np.flatnonzero(
                (origin + horizon <= first)
                & (origin >= first - np.timedelta64(common.WINDOW_DAYS, "D"))
                & usable
                & in_fit
                & (group == g)
            )
            counts = [
                int((day[fit_rows] & (lead_band[fit_rows] == b)).sum()) for b in range(N_LEAD_BANDS)
            ]
            thin = min(counts) < arms.MIN_ROWS
            if thin:
                factors = [(1.0, 1.0)] * N_LEAD_BANDS
            else:
                sub = frame[fit_rows]
                stats = {
                    k: v.astype(np.float32).astype(np.float64)
                    for k, v in a202.summaries(delta=delta[fit_rows]).items()
                }
                factors = a202.fit_factors(
                    frame=sub,
                    make_points=lambda s, stats=stats: np.nan_to_num(
                        a202.gaussian_points(stats=stats, spread=s)
                    ),
                    fit_spread=True,
                )
            stats_m = {
                k: v.astype(np.float32).astype(np.float64)
                for k, v in a202.summaries(delta=delta[rows_m]).items()
            }
            sub_m = frame[rows_m]
            q = a202.apply_arm(
                frame=sub_m,
                make_points=lambda s, stats=stats_m: np.nan_to_num(
                    a202.gaussian_points(stats=stats, spread=s)
                ),
                factors=factors,
            )
            out[rows_m] = np.sort(q, axis=1)
            log[f"{source}/{g}/{m}"] = {
                "daylight_rows_per_band": counts,
                "identity": bool(thin),
                "factors": factors,
            }
            print(source, g, m, counts, "identity" if thin else "fitted", flush=True)
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
        *QUANTILES,
    ).rename({q: f"base_{q}" for q in QUANTILES})
    keep = keep.with_columns(
        [pl.Series(f"x1_{q}", out[:, k].astype(np.float32)) for k, q in enumerate(QUANTILES)]
    )
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    keep.write_parquet(common.ARMS_DIR / f"low_lead_{source}.parquet")
    common.write_json_atomic(path=common.ARMS_DIR / f"fit_log_low_{source}.json", content=log)
    print(json.dumps({"source": source, "rows": keep.height}))


if __name__ == "__main__":
    main()

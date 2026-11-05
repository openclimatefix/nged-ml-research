# ruff: noqa: E501, PLR0912
"""Experiment 580, step 2: verify the labelled load-shift events from the telemetry alone.

The labels are used only as candidate windows (approximate dates, and which Primaries a transfer
touches). Depth and dates come from the telemetry.

**Sign convention.** A fitted depth is signed MW: positive means the Primary carries *more* power
after the onset than before it. `corrected = observed - depth_fit_mw` inside `[onset, end)`
removes the event (see `correct()` and its test in `e00580_check_sign.py`).

**Step.** At grid index t with window w half-hours, the step is the mean over the half-hours of
the day of (median of that half-hour over the w after t) minus (median of the same half-hour over
the w before t), for w of 1 or 3 whole days; for a shorter window it is the plain median
difference. Power is the cleaned power over capacity (fraction of effective capacity).

**Noise threshold.** For each Primary and window, a percentile of |step| over dates more than 4
days from every labelled onset and end at that Primary, at least `MIN_SHARE` of capacity. The
strict rule uses the 99th percentile for each Primary alone; the pair rule uses the 95th
percentile for each Primary of a transfer, because a mirrored step at two Primaries on the same
dates is far less likely by chance than a step at one.

**Verified step pair at one Primary.** Onset step and end step both exceed the threshold, have
opposite signs, and their sizes are within a factor of `SIZE_FACTOR` of each other. **Verified
transfer.** Every Primary of the label cluster (labels with the same onset and end, within 2 days)
passes, and the onset steps conserve load: the sum of signed MW over the cluster is under
`CONSERVE` of the largest member. A passing Primary in a cluster that fails to conserve, or a
Primary with no partner, is an unpaired step (possible level change).
"""

import warnings
from pathlib import Path
from typing import Final

import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00580-verified-load-shift-event-set-and-label-audit"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
EVENTS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored/work/events_corrected.parquet"
)
DAY: Final[int] = 48
SEARCH: Final[int] = 3 * DAY
MIN_SHARE: Final[float] = 0.05
NULL_PCT: Final[float] = 99.0
PAIR_PCT: Final[float] = 95.0
DATE_TOL: Final[int] = DAY
MIN_LONG: Final[int] = 7 * DAY
MAX_LONG: Final[int] = 60 * DAY
NULL_EXCLUDE: Final[int] = 4 * DAY
SIZE_FACTOR: Final[float] = 2.0
CONSERVE: Final[float] = 0.5
CLUSTER_TOL: Final[int] = 2 * DAY
SCAN_CONSERVE: Final[float] = 0.3
DEDUPE: Final[int] = 3 * DAY
CHUNK: Final[int] = 20000
WINDOWS: Final[tuple[int, ...]] = (3 * DAY, DAY, 12)


def window_for(*, steps: int) -> int:
    """Return the step window (half-hours) for an event of `steps` half-hours: 3 days, 1 day or 6 h."""
    if steps >= 3 * DAY:
        return 3 * DAY
    if steps >= DAY:
        return DAY
    return 12


def step_series(*, x: np.ndarray, w: int) -> np.ndarray:
    """Return the step at every grid index t (NaN where a window is short of data): after minus before."""
    n = len(x)
    out = np.full(n, np.nan, np.float32)
    sw = sliding_window_view(x, 2 * w)  # row i covers x[i : i + 2w]; t = i + w
    for lo in range(0, sw.shape[0], CHUNK):
        blk = sw[lo : lo + CHUNK]
        before, after = blk[:, :w], blk[:, w:]
        ok = (np.isfinite(before).mean(axis=1) >= 0.5) & (np.isfinite(after).mean(axis=1) >= 0.5)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if w % DAY == 0:
                k = w // DAY
                mb = np.nanmedian(before.reshape(-1, k, DAY), axis=1)
                ma = np.nanmedian(after.reshape(-1, k, DAY), axis=1)
                d = np.nanmean(ma - mb, axis=1)
            else:
                d = np.nanmedian(after, axis=1) - np.nanmedian(before, axis=1)
        d[~ok] = np.nan
        out[lo + w : lo + w + len(d)] = d
    return out


def noise_threshold(*, steps: np.ndarray, near_label: np.ndarray, pct: float) -> float:
    """Return the 99th percentile of |step| on dates away from labelled onsets and ends, floored at MIN_SHARE."""
    keep = np.isfinite(steps) & ~near_label
    keep &= (np.arange(len(steps)) % 12) == 0  # one date per 6 h limits dependence between draws
    return max(float(np.nanpercentile(np.abs(steps[keep]), pct)), MIN_SHARE)


def peak(*, s: np.ndarray, centre: int, radius: int) -> tuple[float, int]:
    """Return the step of the largest |step| within `radius` of `centre`, and its grid index."""
    lo, hi = max(centre - radius, 0), min(centre + radius + 1, len(s))
    seg = s[lo:hi]
    if not np.isfinite(seg).any():
        return float("nan"), -1
    j = int(np.nanargmax(np.abs(seg)))
    return float(seg[j]), lo + j


def correct(*, observed: np.ndarray, depth: float, onset: int, end: int) -> np.ndarray:
    """Return the counterfactual (no event): observed minus the signed depth inside [onset, end)."""
    out = observed.copy()
    out[onset:end] -= depth
    return out


def load_series() -> tuple[pl.DataFrame, list[str], np.ndarray, dict[str, float], dict, dict, dict]:
    """Return events, Primary labels, power, capacities, step series, strict and pair thresholds (cached)."""
    ev = pl.read_parquet(EVENTS)
    labels = sorted(set(ev["label"].to_list()), key=lambda s: int(s[1:]))
    row_of = {lab: i for i, lab in enumerate(labels)}
    x = c.clean_stack()
    cap = {
        r["label"]: r["capacity_mw"]
        for r in ev.group_by("label").agg(pl.col("capacity_mw").first()).iter_rows(named=True)
    }
    cache = WORK / "step_series.npz"
    series: dict[tuple[str, int], np.ndarray] = {}
    thr: dict[tuple[str, int], float] = {}
    thr95: dict[tuple[str, int], float] = {}
    if cache.exists():
        z = np.load(cache)
        for w in WINDOWS:
            for lab in labels:
                series[(lab, w)] = z[f"s_{lab}_{w}"]
                thr[(lab, w)] = float(z[f"t99_{lab}_{w}"])
                thr95[(lab, w)] = float(z[f"t95_{lab}_{w}"])
        return ev, labels, x, cap, series, thr, thr95
    n = x.shape[1]
    out: dict[str, np.ndarray | np.float64] = {}
    for w in WINDOWS:
        for lab in labels:
            st = step_series(x=x[row_of[lab]], w=w)
            near = np.zeros(n, bool)
            for r in ev.filter(pl.col("label") == lab).iter_rows(named=True):
                for t in (r["onset_idx"], r["end_idx"] + 1):
                    near[max(t - NULL_EXCLUDE, 0) : t + NULL_EXCLUDE + 1] = True
            series[(lab, w)] = st
            thr[(lab, w)] = noise_threshold(steps=st, near_label=near, pct=NULL_PCT)
            thr95[(lab, w)] = noise_threshold(steps=st, near_label=near, pct=PAIR_PCT)
            out[f"s_{lab}_{w}"] = st
            out[f"t99_{lab}_{w}"] = np.float64(thr[(lab, w)])
            out[f"t95_{lab}_{w}"] = np.float64(thr95[(lab, w)])
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)  # ty: ignore[invalid-argument-type]
    return ev, labels, x, cap, series, thr, thr95


def fit_event(*, r: dict, series: dict, thr: dict, thr95: dict) -> dict:
    """Fit onset and end steps of one labelled event and apply the strict (own) rule."""
    lab = r["label"]
    steps = r["end_idx"] - r["onset_idx"] + 1
    w = window_for(steps=steps)
    s = series[(lab, w)]
    on, t_on = peak(s=s, centre=r["onset_idx"], radius=SEARCH)
    en, t_en = peak(s=s, centre=r["end_idx"] + 1, radius=SEARCH)

    def reasons(th: float) -> list[str]:
        out = []
        if steps < 12:
            return ["under 6 hours"]
        if not (np.isfinite(on) and np.isfinite(en)):
            return ["no telemetry"]
        if abs(on) < th:
            out.append("onset step below noise")
        if abs(en) < th:
            out.append("end step below noise")
        if on * en > 0:
            out.append("onset and end steps have the same sign")
        lo, hi = sorted((abs(on), abs(en)))
        if lo > 0 and hi / lo > SIZE_FACTOR:
            out.append("onset and end steps differ in size")
        if t_en <= t_on:
            out.append("end step before onset step")
        return out

    strict, pair = reasons(thr[(lab, w)]), reasons(thr95[(lab, w)])
    return {
        "event_id": r["event_id"],
        "label": lab,
        "label_depth_mw": r["depth_mw"],
        "label_onset_idx": r["onset_idx"],
        "label_end_idx": r["end_idx"],
        "label_length_days": r["length_days"],
        "capacity_mw": r["capacity_mw"],
        "window_half_hours": w,
        "noise_p99_share": thr[(lab, w)],
        "noise_p95_share": thr95[(lab, w)],
        "fit_onset_idx": t_on,
        "fit_end_idx": t_en,
        "fit_onset_step_share": on,
        "fit_end_step_share": en,
        "fit_depth_mw": 0.5 * (on - en) * r["capacity_mw"]
        if np.isfinite(on + en)
        else float("nan"),
        "strict_reasons": "; ".join(strict),
        "strict_pass": not strict,
        "pair_reasons": "; ".join(pair),
        "pair_pass": not pair,
    }


def verify_events(*, ev: pl.DataFrame, series: dict, thr: dict, thr95: dict) -> pl.DataFrame:
    """Return one row per labelled event with the telemetry fit and its verdict."""
    df = pl.DataFrame(
        [fit_event(r=r, series=series, thr=thr, thr95=thr95) for r in ev.iter_rows(named=True)],
        infer_schema_length=None,
    )
    d = df.to_dicts()
    cluster = [-1] * len(d)
    k = 0
    for i, a in enumerate(d):
        if cluster[i] >= 0:
            continue
        cluster[i] = k
        for j in range(i + 1, len(d)):
            b = d[j]
            if (
                cluster[j] < 0
                and abs(a["label_onset_idx"] - b["label_onset_idx"]) <= CLUSTER_TOL
                and abs(a["label_end_idx"] - b["label_end_idx"]) <= CLUSTER_TOL
            ):
                cluster[j] = k
        k += 1
    df = df.with_columns(cluster=pl.Series(cluster))
    status, why, members, pair_status = [], [], [], []
    for a in df.to_dicts():
        grp = df.filter(pl.col("cluster") == a["cluster"])
        members.append(",".join(sorted(grp["label"].to_list(), key=lambda s: int(s[1:]))))
        if a["label_length_days"] * 24 < 6:
            status.append("too short to check")
            pair_status.append("too short to check")
            why.append(a["strict_reasons"])
            continue
        for col, store in (("strict_pass", status), ("pair_pass", pair_status)):
            if not a[col]:
                store.append("rejected")
            elif grp.height == 1:
                store.append("unpaired step")
            else:
                ok = grp.filter(pl.col(col))
                mw = [
                    0.5 * (g["fit_onset_step_share"] - g["fit_end_step_share"]) * g["capacity_mw"]
                    for g in ok.iter_rows(named=True)
                ]
                on_t = ok["fit_onset_idx"].to_list()
                dates_ok = max(on_t) - min(on_t) <= 2 * DATE_TOL
                conserve = ok.height == grp.height and abs(sum(mw)) <= CONSERVE * max(
                    abs(m) for m in mw
                )
                store.append("verified transfer" if conserve and dates_ok else "unpaired step")
        if a["strict_pass"]:
            why.append(
                "" if status[-1] == "verified transfer" else "partner fails or load not conserved"
            )
        else:
            why.append(a["strict_reasons"])
    return df.with_columns(
        status=pl.Series(status),
        pair_status=pl.Series(pair_status),
        reason=pl.Series(why),
        cluster_members=pl.Series(members),
        label_depth_over_fit=pl.col("label_depth_mw") / pl.col("fit_depth_mw"),
    )


def peaks_of(*, s: np.ndarray, th: float) -> list[tuple[int, float]]:
    """Return (index, step) of each local maximum of |step| above `th` within a +-1 day neighbourhood."""
    a = np.abs(np.nan_to_num(s, nan=0.0))
    idx = np.flatnonzero(a >= th)
    out: list[tuple[int, float]] = []
    taken = -(10**9)
    # greedy: largest first
    for i in idx[np.argsort(-a[idx])]:
        if abs(int(i) - taken) < 0:
            continue
        if any(abs(int(i) - j) <= DAY for j, _ in out):
            continue
        out.append((int(i), float(s[i])))
    return sorted(out)


def scan_pairs(
    *, ev: pl.DataFrame, labels: list[str], cap: dict[str, float], series: dict, thr95: dict
) -> pl.DataFrame:
    """Return sustained mirrored pairs (7 to 60 days) found in the telemetry alone, and whether labelled."""
    pk = {lab: peaks_of(s=series[(lab, 3 * DAY)], th=thr95[(lab, 3 * DAY)]) for lab in labels}
    rows = []
    for i, p in enumerate(labels):
        for q in labels[i + 1 :]:
            for tp, sp in pk[p]:
                for tq, sq in pk[q]:
                    if abs(tp - tq) > DATE_TOL or sp * sq >= 0:
                        continue
                    mp, mq = sp * cap[p], sq * cap[q]
                    if abs(mp + mq) > SCAN_CONSERVE * max(abs(mp), abs(mq)):
                        continue
                    ends_p = [
                        (t, v)
                        for t, v in pk[p]
                        if MIN_LONG <= t - tp <= MAX_LONG
                        and v * sp < 0
                        and 0.5 <= abs(v / sp) <= SIZE_FACTOR
                    ]
                    ends_q = [
                        (t, v)
                        for t, v in pk[q]
                        if MIN_LONG <= t - tq <= MAX_LONG
                        and v * sq < 0
                        and 0.5 <= abs(v / sq) <= SIZE_FACTOR
                    ]
                    best = None
                    for tep, vp in ends_p:
                        for teq, vq in ends_q:
                            if abs(tep - teq) <= DATE_TOL and (best is None or tep < best[0]):
                                best = (tep, vp, teq, vq)
                    if best is None:
                        continue
                    rows.append(
                        {
                            "mismatch": abs(mp + mq) / max(abs(mp), abs(mq)),
                            "primary_a": p,
                            "primary_b": q,
                            "onset_a": tp,
                            "onset_b": tq,
                            "end_a": best[0],
                            "end_b": best[2],
                            "onset_step_a_share": sp,
                            "onset_step_b_share": sq,
                            "end_step_a_share": best[1],
                            "end_step_b_share": best[3],
                            "depth_a_mw": mp,
                            "depth_b_mw": mq,
                            "length_days": (best[0] - tp) / DAY,
                        }
                    )
    df = pl.DataFrame(rows, infer_schema_length=None).sort("mismatch")
    keep: list[dict] = []
    for r in df.iter_rows(named=True):
        if not any(
            k["primary_a"] == r["primary_a"]
            and k["primary_b"] == r["primary_b"]
            and abs(k["onset_a"] - r["onset_a"]) <= DEDUPE
            and abs(k["end_a"] - r["end_a"]) <= DEDUPE
            for k in keep
        ):
            keep.append(r)
    df = pl.DataFrame(keep, infer_schema_length=None).sort("onset_a")
    labelled = []
    for r in df.iter_rows(named=True):
        hit = ev.filter(
            pl.col("label").is_in([r["primary_a"], r["primary_b"]])
            & ((pl.col("onset_idx") - r["onset_a"]).abs() <= SEARCH)
            & ((pl.col("end_idx") - r["end_a"]).abs() <= SEARCH)
        )
        labelled.append(hit["event_id"].to_list())
    return df.with_columns(labelled_event_ids=pl.Series(labelled, dtype=pl.List(pl.Int64)))


if __name__ == "__main__":
    ev_, labels_, x_, cap_, series_, thr_, thr95_ = load_series()
    out = verify_events(ev=ev_, series=series_, thr=thr_, thr95=thr95_)
    WORK.mkdir(parents=True, exist_ok=True)
    out.write_parquet(WORK / "events_verified_all.parquet")
    for name in ("status", "pair_status"):
        print(name, out[name].value_counts().sort(name).rows())
        print(
            " >=7 d",
            out.filter(pl.col("label_length_days") >= 7)[name].value_counts().sort(name).rows(),
        )
    cand = scan_pairs(ev=ev_, labels=labels_, cap=cap_, series=series_, thr95=thr95_)
    cand.write_parquet(WORK / "candidate_pairs.parquet")
    print(
        "candidate pairs",
        cand.height,
        "unlabelled",
        cand.filter(pl.col("labelled_event_ids").list.len() == 0).height,
    )

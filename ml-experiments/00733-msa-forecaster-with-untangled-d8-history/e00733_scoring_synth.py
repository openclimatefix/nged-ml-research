# ruff: noqa: E501
"""Experiment 733: scoring the synthetic world (C3, C4, R5, R6, R7, R11) from the stored archives.

The scored rows are the samples of the test bank (the validation bank on validation rows) at lead
days 2 to 10 (lead index 1 to 9), on the half-hours that hold a sign. Each arm's probability and
quantiles are averaged over the seeds present, and the contrasts pair the arms on the same
half-hours with experiment 729's two-way bootstrap.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import e00733_baselines as baselines
import e00733_core as core
import e00733_predict as pred
import e00733_score as sc
import e00733_store as store
import e00733_synth as sy
import e00733_world as world
import numpy as np

SYNTHETIC_ARMS: Final = ("Y0", "YU", "YT")
N_RESAMPLES_VALIDATION: Final = 1000
GATHER_CHUNK: Final = 512


def clim_from_limits(*, root: Path) -> baselines.Clim:
    """Return B-clim as `limits.json` stored it (a share of -1 marks an empty cell)."""
    b = json.loads((root / "limits.json").read_text())["b_clim"]
    share = np.asarray(b["share"], dtype=np.float64)
    return baselines.Clim(
        share=np.where(share < 0.0, np.nan, share), count=np.asarray(b["count"]), overall=float(b["overall"])
    )  # fmt: skip


def archive_path(*, root: Path, arm: str, seed: int, rows: str) -> Path:
    """Return the archive of one synthetic fit's predictions on the test or the validation bank."""
    return (
        store.set_folder(root=root, name=store.set_name(arm=arm, fold=None, seed=seed))
        / f"{rows}.npz"
    )


def read_arm(
    *, root: Path, arm: str, seeds: Sequence[int], rows: str
) -> dict[str, np.ndarray] | None:  # fmt: skip
    """Return an arm's seed-mean probability and quantiles, or None when a seed's archive is missing.

    Args:
        root: This experiment's data folder.
        arm: Y0, YU, or YT.
        seeds: The seeds the arm must hold.
        rows: "test" or "validation".

    Returns:
        `p_neg` `(N, 48)`, `quantiles` `(N, 48, 11)` sorted, `target_z`, `sign_truth`, and the keys
        `seg`, `origin_day`, `lead_index`; None when any seed is missing.

    Raises:
        sc.ScoreError: When the seeds' archives hold other keys.
    """
    paths = [archive_path(root=root, arm=arm, seed=s, rows=rows) for s in seeds]
    if not paths or not all(p.exists() for p in paths):
        return None
    got = [pred.read_synthetic(path=p) for p in paths]
    first = got[0]
    for g in got[1:]:
        for key in ("seg", "origin_day", "lead_index"):
            if not np.array_equal(g[key], first[key]):
                msg = f"the seeds of {arm} hold other samples ({key})"
                raise sc.ScoreError(msg)
    return {
        "p_neg": np.mean([g["p_neg"] for g in got], axis=0),
        "quantiles": np.sort(np.mean([g["quantiles"] for g in got], axis=0), axis=-1),
        **{k: first[k] for k in ("target_z", "sign_truth", "seg", "origin_day", "lead_index")},
    }  # fmt: skip


def seeds_held(*, root: Path, arm: str, rows: str) -> tuple[int, ...]:
    """Return the seeds whose archive exists, all of them required on test rows.

    A validation score (the smoke test, a scratch run) reads the seeds that exist, because it checks
    the code and its readings are interim. The official test score reads an arm only when every seed
    of `core.SYNTHETIC_SEEDS` exists (a missing seed makes the arm's readings void).
    """
    held = tuple(
        s
        for s in core.SYNTHETIC_SEEDS
        if archive_path(root=root, arm=arm, seed=s, rows=rows).exists()
    )
    return held if rows == "validation" or len(held) == len(core.SYNTHETIC_SEEDS) else ()


def scored_mask(*, lead_index: np.ndarray) -> np.ndarray:
    """Return the samples at lead days 2 to 10 (lead index 1 to 9), `(N,)`."""
    days = np.asarray(lead_index) + 1
    return (days >= core.SYNTHETIC_LEAD_DAYS[0]) & (days <= core.SYNTHETIC_LEAD_DAYS[-1])


def baseline_probabilities(
    *, segments: Sequence[Any], index: Any, clim: baselines.Clim, d8_index: int, target_std: float
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return B-clim, B-hist, and O for every sample, each `(N, 48)`.

    Args:
        segments: The bank's segments.
        index: Its samples.
        clim: B-clim.
        d8_index: The Primary index of D8.
        target_std: D8's target standard deviation.

    Returns:
        `b_clim`, `b_hist`, and `o`.
    """
    n = len(index)
    b_clim = np.zeros((n, core.STEPS_PER_DAY))
    b_hist = np.zeros((n, core.STEPS_PER_DAY))
    for start in range(0, n, GATHER_CHUNK):
        rows = np.arange(start, min(start + GATHER_CHUNK, n))
        c = baselines.clim_for_rows(clim=clim, segments=segments, index=index, rows=rows)
        batch = sy.gather_batch(
            segments=segments, index=index, rows=rows, d8_index=d8_index, target_std=target_std, mode="untangled",
        )  # fmt: skip
        b_clim[rows] = c
        b_hist[rows] = baselines.b_hist_for_batch(batch=batch, clim=c)
    o = baselines.o_for_rows(segments=segments, index=index, rows=np.arange(n))
    return {"b_clim": b_clim, "b_hist": b_hist, "o": o}


def segment_months(*, segments: Sequence[Any]) -> np.ndarray:
    """Return each segment's month id (the local month of its first base day), the resampling stratum."""
    out = np.zeros(len(segments), dtype=np.int64)
    for i, seg in enumerate(segments):
        days = np.flatnonzero(seg.present)
        out[i] = int(seg.month[days[0]]) if len(days) else 0
    return out


def masked(*, values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Return `values.ravel()` with NaN where `keep` is false."""
    return np.where(keep.ravel(), np.asarray(values, dtype=np.float64).ravel(), np.nan)


def crps_rows(*, quantiles: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    """Return the CRPS of each half-hour (twice the mean pinball loss over the 11 levels), standardised units."""
    from studies.eval_protocol.metrics import quantile_score

    q = quantiles.reshape(-1, quantiles.shape[-1]).astype(np.float64)
    return quantile_score(actual=np.asarray(target_z, dtype=np.float64).ravel(), quantiles=q)


def synthetic_readings(
    *, root: Path, rows: str, n_resamples: int | None = None
) -> dict[str, Any]:  # fmt: skip
    """Read C3, C4, R5, R6, R7, R11, and the sign-forecast inputs from the stored archives.

    Args:
        root: This experiment's data folder.
        rows: "test" or "validation".
        n_resamples: The bootstrap resamples (default the plan's 20,000, or 1,000 on validation).

    Returns:
        `contrasts` (name to experiment 729's reading or None), `sign_inputs` (None when YU or
        B-clim is unread), `log_loss` (the mean log-loss difference beside each Brier contrast),
        `r11`, and `n_scored`. An arm with a missing seed gives None for the contrasts that read it.
    """
    n_boot = (
        n_resamples
        if n_resamples is not None
        else (core.N_RESAMPLES_SYNTHETIC if rows == "test" else N_RESAMPLES_VALIDATION)
    )
    held = {a: seeds_held(root=root, arm=a, rows=rows) for a in SYNTHETIC_ARMS}
    arms = {
        a: (read_arm(root=root, arm=a, seeds=held[a], rows=rows) if held[a] else None)
        for a in SYNTHETIC_ARMS
    }
    out: dict[str, Any] = {
        "contrasts": dict.fromkeys(("C3", "C4", "R5", "R6", "R7")), "sign_inputs": None,
        "log_loss": {}, "r11": {"status": "void", "reason": "YU is unread"}, "n_scored": 0,
    }  # fmt: skip
    yu = arms["YU"]
    if yu is None:
        return out
    segments, _ = world.read_world(root=root, kind="validation" if rows == "validation" else "test")
    index = sy.enumerate_samples(segments=segments)
    for a in arms.values():
        if a is not None and len(a["seg"]) != len(index):
            msg = "a stored synthetic archive holds another number of samples than the bank"
            raise sc.ScoreError(msg)
    import e00733_adapters as adapters

    meta = world.read_meta(root=root)["constants"]
    base = baseline_probabilities(
        segments=segments, index=index, clim=clim_from_limits(root=root), d8_index=adapters.d8_index(),
        target_std=float(meta["std"]),
    )  # fmt: skip
    truth = yu["sign_truth"]
    keep = scored_mask(lead_index=index.lead_index)[:, None] & np.isfinite(truth)
    n_scored = int(keep.sum())
    if n_scored == 0:
        msg = "no half-hour of the synthetic bank is scored, so the readings would rest on nothing"
        raise sc.ScoreError(msg)
    out["n_scored"] = n_scored
    srows = sc.synthetic_rows(
        seg=index.seg, origin_day=index.origin_day, seg_month=segment_months(segments=segments)
    )

    def brier(p: np.ndarray) -> np.ndarray:
        return masked(values=(np.asarray(p) - truth) ** 2, keep=keep & np.isfinite(p))

    def log_loss(p: np.ndarray) -> np.ndarray:
        return masked(
            values=sc.log_loss_rows(p=p, truth=truth).reshape(truth.shape),
            keep=keep & np.isfinite(p),
        )

    def read(first: np.ndarray, second: np.ndarray) -> Any:
        return sc.synthetic_contrast(rows=srows, first=first, second=second, n_resamples=n_boot)

    def mean_diff(first: np.ndarray, second: np.ndarray) -> float:
        d = first - second
        return float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")

    y0, yt = arms["Y0"], arms["YT"]
    c = out["contrasts"]
    c["C3"] = read(brier(yu["p_neg"]), brier(base["b_hist"]))
    out["log_loss"]["C3"] = mean_diff(log_loss(yu["p_neg"]), log_loss(base["b_hist"]))
    c["R7"] = read(brier(yu["p_neg"]), brier(base["o"]))
    if y0 is not None:
        c["C4"] = read(brier(yu["p_neg"]), brier(y0["p_neg"]))
        out["log_loss"]["C4"] = mean_diff(log_loss(yu["p_neg"]), log_loss(y0["p_neg"]))
        c["R6"] = read(
            masked(values=crps_rows(quantiles=yu["quantiles"], target_z=yu["target_z"]), keep=keep),
            masked(values=crps_rows(quantiles=y0["quantiles"], target_z=y0["target_z"]), keep=keep),
        )  # fmt: skip
        if yt is not None:
            c["R5"] = read(brier(yt["p_neg"]), brier(y0["p_neg"]))
    ok = keep.ravel()
    out["sign_inputs"] = sc.sign_forecast_inputs(
        p_yu=yu["p_neg"].ravel()[ok], p_clim=base["b_clim"].ravel()[ok], truth=truth.ravel()[ok],
        seeds_present=len(held["YU"]) == len(core.SYNTHETIC_SEEDS),
    )  # fmt: skip
    out["r11"] = reading_r11(
        yu=yu, segments=segments, index=index, keep=keep, meta=meta,
    )  # fmt: skip
    return out


def reading_r11(
    *, yu: dict[str, np.ndarray], segments: Sequence[Any], index: Any, keep: np.ndarray, meta: dict[str, Any]
) -> dict[str, Any]:  # fmt: skip
    """Return R11: YU's sign head by lead band and by reading band, with the import half-hours' mean P(X<0).

    The reading bands are under 1 MVA, 1 to 3 MVA, and 3 MVA and above (the absolute reading at the
    target half-hour). The mean P(X<0) on import half-hours of at least 1 MVA sits beside the pooled
    harm limit of experiment 730.

    Args:
        yu: YU's seed-mean arrays.
        segments: The bank's segments.
        index: Its samples.
        keep: The scored half-hours `(N, 48)`.
        meta: The reading constants (`mean`, `std`).

    Returns:
        The tables.
    """
    import e00578_common as k578  # ty: ignore[unresolved-import]

    sites = k578.sites()
    capacity = float(sites.filter(sites["label"] == "D8")["capacity_mw"][0])
    mva = np.full(yu["p_neg"].shape, np.nan)
    for r in range(len(index)):
        seg = segments[int(index.seg[r])]
        t = sy.target_slots(
            origin_day=index.origin_day[r : r + 1], lead_index=index.lead_index[r : r + 1]
        )[0]
        mva[r] = np.abs((np.asarray(seg.reading_z)[t] * meta["std"] + meta["mean"]) * capacity)
    p, truth = yu["p_neg"], yu["sign_truth"]
    lo, hi = core.READING_BANDS_MVA
    bands = {
        "under 1 MVA": mva < lo,
        "1 to 3 MVA": (mva >= lo) & (mva < hi),
        "3 MVA and above": mva >= hi,
    }

    def table(mask: np.ndarray) -> dict[str, Any]:
        m = mask & keep & np.isfinite(mva)
        if not m.any():
            return {"n": 0}
        return {
            "n": int(m.sum()), "brier": float(np.mean((p[m] - truth[m]) ** 2)),
            "mean_p": float(p[m].mean()), "share_reverse": float(truth[m].mean()),
            "reliability": sc.reliability_term(p=p[m], truth=truth[m]),
        }  # fmt: skip

    lead_day = (np.asarray(index.lead_index) + 1)[:, None] * np.ones_like(p)
    out: dict[str, Any] = {
        "by_lead_band": {},
        "by_reading_band": {n: table(m) for n, m in bands.items()},
    }
    for a, b in core.BAND_LEAD_DAYS:
        out["by_lead_band"][f"days {a} to {b}"] = table((lead_day >= a) & (lead_day <= b))
    imports = (truth == 0.0) & (mva >= lo) & keep
    out["import_half_hours_of_at_least_1_mva"] = {
        "n": int(imports.sum()),
        "mean_p_neg": float(p[imports].mean()) if imports.any() else float("nan"),
        "harm_limit_730": core.HARM_LIMIT_730,
    }
    return out

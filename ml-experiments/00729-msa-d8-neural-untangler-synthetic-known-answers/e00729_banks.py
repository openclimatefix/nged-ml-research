"""Experiment 729: build, hash, and store the synthetic banks, and the realism gate's functions.

A bank is a list of segments of one kind. Training banks hold 32 draws per training month per seed
and draw their own no-injection share. Validation, test, and shifted-mix banks hold 8 draws per
month in separate seed spaces, are built once, and are hashed so that every arm reads identical
bytes. No function prints a month or a date.
"""

import hashlib
import os
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import e00729_core as core
import e00729_sim as sim
import numpy as np

BankKind = Literal["train", "validation", "test", "shifted"]
SPLIT_OF_KIND: Final = {
    "train": "training", "validation": "validation", "test": "test", "shifted": "test",
}  # fmt: skip
SEED_BASE: Final = {
    "train": core.TRAIN_SEED_BASE, "validation": core.EVAL_SEED_BASE,
    "test": core.EVAL_SEED_BASE, "shifted": core.SHIFTED_SEED_BASE,
}  # fmt: skip
"""The first entry of each bank kind's `SeedSequence`. Test and validation share a base but differ
in the month index, so their draws differ and never meet a training seed."""


@dataclass
class Bank:
    """A hashed list of segments of one kind.

    Attributes:
        kind: "train", "validation", "test", or "shifted".
        seed: The training seed, or None for the shared evaluation banks.
        root: The first entry of the seed sequence that built every segment.
        none_share: The no-injection share of each (month) sub-bank, by month.
        segments: The segments.
        digest: The SHA-256 of the bank's arrays.
    """

    kind: BankKind
    seed: int | None
    root: int
    none_share: dict[str, float]
    segments: list[sim.Segment]
    digest: str


def bank_root(*, kind: BankKind, seed: int | None) -> int:
    """Return the first seed-sequence entry of a bank: 7290 plus the seed, or the evaluation base."""
    if kind == "train":
        assert seed is not None
        return core.TRAIN_SEED_BASE + seed
    return SEED_BASE[kind]


def seed_spaces_disjoint(*, train_seeds: Sequence[int]) -> bool:
    """Return whether no training root equals an evaluation root (the leakage check's core)."""
    train = {bank_root(kind="train", seed=s) for s in train_seeds}
    evals = {bank_root(kind=k, seed=None) for k in ("validation", "test", "shifted")}
    return not (train & evals)


def bank_digest(*, segments: Sequence[sim.Segment]) -> str:
    """Return the SHA-256 of the segments' truth, reading, and BSP arrays and world labels."""
    h = hashlib.sha256()
    for s in segments:
        h.update(s.world.encode())
        h.update(s.month.encode())
        for arr in (s.x, s.r, s.bsp, s.engine, s.solar):
            h.update(np.ascontiguousarray(np.nan_to_num(arr, nan=-1.0)).tobytes())
    return h.hexdigest()


def build_bank(
    *, cal: sim.Calendar, kind: BankKind, seed: int | None, months: Sequence[str],
    month_use: dict[str, np.ndarray],
) -> Bank:  # fmt: skip
    """Build one bank.

    Args:
        cal: The calendar of base days.
        kind: The bank kind.
        seed: The training seed (None for evaluation banks).
        months: The months of this bank (training months, or validation or test months).
        month_use: For each month, which base days of `cal` the segments may use.

    Returns:
        The bank.
    """
    root = bank_root(kind=kind, seed=seed)
    n_draws = core.SEGMENT_DRAWS_TRAIN if kind == "train" else core.SEGMENT_DRAWS_EVAL
    segments: list[sim.Segment] = []
    none_share: dict[str, float] = {}
    for index, month in enumerate(sorted(months)):
        ss = np.random.SeedSequence([root, index])
        children = ss.spawn(n_draws + 1)
        rng_mix = np.random.default_rng(children[0])
        if kind == "train":
            share = core.draw_none_share(rng=rng_mix)
        elif kind == "shifted":
            share = core.SHIFTED_NONE_SHARE
        else:
            share = core.TEST_MIX["none"]
        none_share[month] = share
        worlds = core.draw_world_types(rng=rng_mix, n=n_draws, none_share=share)
        for draw in range(n_draws):
            segments.append(
                sim.simulate_segment(
                    cal=cal,
                    month=month,
                    use=month_use[month],
                    world=worlds[draw],
                    rng=np.random.default_rng(children[draw + 1]),
                )  # fmt: skip
            )
    return Bank(
        kind=kind, seed=seed, root=root, none_share=none_share, segments=segments,
        digest=bank_digest(segments=segments),
    )  # fmt: skip


def save_bank(*, bank: Bank, path: Path) -> None:
    """Write a bank atomically (a temporary file in the same folder, then a rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(bank, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(path)


def load_bank(*, path: Path) -> Bank:
    """Read a bank and check that its stored digest matches its arrays."""
    bank = pickle.loads(path.read_bytes())  # noqa: S301
    assert bank_digest(segments=bank.segments) == bank.digest, "a stored bank changed on disk"
    return bank


def reverse_counts(*, bank: Bank) -> tuple[int, int]:
    """Return the reverse-flow half-hours outside the dead band, in all segments and in BSP ones."""
    total = bsp = 0
    for s in bank.segments:
        n = int(((s.x < -core.DEAD_BAND_POINTS) & s.valid).sum())
        total += n
        if s.has_bsp:
            bsp += n
    return total, bsp


# --------------------------------------------------------------------------- the realism gate

REAL_MIN_POINTS: Final = core.GATE_MIN_VALUE_MVA / core.CAPACITY_MVA * core.POINTS
"""1.0 MVA in points: the level under which a half-hour counts as near zero."""


def near_zero_shares(*, r: np.ndarray) -> tuple[float, float]:
    """Return the share of days with a daily minimum under 1.0 MVA and of half-hours under it.

    Args:
        r: Readings in points, shape `(days, 48)`, NaN where missing.

    Returns:
        The two shares, over days that hold at least one finite half-hour.
    """
    ok = np.isfinite(r).any(axis=1)
    if not ok.any():
        return float("nan"), float("nan")
    day_min = np.nanmin(r[ok], axis=1)
    finite = np.isfinite(r)
    return float((day_min < REAL_MIN_POINTS).mean()), float((r[finite] < REAL_MIN_POINTS).mean())


def bank_shares_by_world(*, bank: Bank, worlds: Sequence[str]) -> tuple[float, float]:
    """Return `near_zero_shares` over the base days of a bank's segments of the given worlds."""
    masked = []
    for s in bank.segments:
        if s.world in worlds:
            on = s.cal_index >= 0
            masked.append(np.where(s.valid[on], s.r[on], np.nan))
    if not masked:
        return float("nan"), float("nan")
    return near_zero_shares(r=np.vstack(masked))


@dataclass(frozen=True)
class StepFit:
    """A regression of BSP steps on engine steps.

    Attributes:
        slope: The slope of the BSP step per point of engine step.
        se: The slope's standard error.
        resid_sd: The residual standard deviation in points.
    """

    slope: float
    se: float
    resid_sd: float


def step_fit(*, d_bsp: np.ndarray, d_eng: np.ndarray, weights: np.ndarray | None = None) -> StepFit:
    """Fit `d_bsp = slope * d_eng + noise` by (weighted) least squares through the origin.

    Args:
        d_bsp: Half-hour changes of the BSP meter (NaN allowed).
        d_eng: Half-hour changes of the engine output.
        weights: Optional row weights.

    Returns:
        The slope, its standard error, and the residual standard deviation.
    """
    ok = np.isfinite(d_bsp) & np.isfinite(d_eng)
    w = np.ones(ok.sum()) if weights is None else weights[ok]
    x, y = d_eng[ok], d_bsp[ok]
    sxx = float((w * x * x).sum())
    if sxx <= 0 or len(x) < 3:
        return StepFit(float("nan"), float("nan"), float("nan"))
    slope = float((w * x * y).sum() / sxx)
    sigma2 = float((w * (y - slope * x) ** 2).sum() / max(w.sum() - 1.0, 1.0))
    return StepFit(slope, float(np.sqrt(sigma2 / sxx)), float(np.sqrt(sigma2)))


def size_bins(*, d_eng: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the quintile bin (0 to 4) of each absolute engine step."""
    inside = np.searchsorted(edges, np.abs(d_eng), side="right") - 1
    return np.clip(inside, 0, core.GATE_STEP_BINS - 1)


def real_bin_edges(*, d_eng: np.ndarray) -> np.ndarray:
    """Return the 6 edges of the quintile bins of the real record's absolute step size."""
    ok = np.abs(d_eng[np.isfinite(d_eng)])
    return np.quantile(ok, np.linspace(0.0, 1.0, core.GATE_STEP_BINS + 1))


def size_matched_weights(*, d_eng: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return row weights so that each size bin holds the real record's share (one fifth) of rows.

    Args:
        d_eng: A synthetic stretch's engine steps.
        edges: The real record's quintile edges from `real_bin_edges`.

    Returns:
        One weight per row (0 where the step is not finite); the weights sum to the finite count.
    """
    bins = size_bins(d_eng=d_eng, edges=edges)
    finite = np.isfinite(d_eng)
    weights = np.zeros(len(d_eng))
    for b in range(core.GATE_STEP_BINS):
        sel = finite & (bins == b)
        if sel.any():
            weights[sel] = (1.0 / core.GATE_STEP_BINS) / (sel.sum() / finite.sum())
    return weights


def bsp_gate(
    *, real: StepFit, real_ci: tuple[float, float], synthetic: Sequence[StepFit]
) -> dict[str, bool | float]:
    """Return the gate's verdict on the synthetic BSP steps against the real relation.

    Args:
        real: The real record's fit.
        real_ci: The 95% interval of the real slope.
        synthetic: One size-matched fit per synthetic stretch.

    Returns:
        Each condition and whether the gate passes: the real slope's interval overlaps -1.0 to
        -0.6, and the real standard error and residual sd each lie in the 5th to 95th percentile of
        the synthetic stretches.
    """
    lo_b, hi_b = -core.RANGES["beta"][1], -core.RANGES["beta"][0]
    overlap = real_ci[0] <= hi_b and real_ci[1] >= lo_b
    se = np.array([f.se for f in synthetic])
    sd = np.array([f.resid_sd for f in synthetic])
    se_in = bool(np.nanpercentile(se, 5) <= real.se <= np.nanpercentile(se, 95))
    sd_in = bool(np.nanpercentile(sd, 5) <= real.resid_sd <= np.nanpercentile(sd, 95))
    return {"slope_overlap": overlap, "se_inside": se_in, "sd_inside": sd_in,
            "pass": bool(overlap and se_in and sd_in)}  # fmt: skip

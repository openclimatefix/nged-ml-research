"""Decoding of worlds and of the real record by every arm of experiment 734.

A `WorldDecoder` holds one world's block panels and a cache of pass-1 posteriors, so H-nbr's pass 1
runs once for all values of kappa, and H-nbr-noshared reuses H-own's pass. Nothing here reads a
file except through the `Rig` and the frozen calibrations it is given.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import e00734_events as ev
import e00734_hmm as hmm
import e00734_panel as pan
import e00734_pelt as pl_pelt
import e00734_store as store
import numpy as np
import polars as pl
from e00653_common import E580, GRID_LENGTH, SCORED_FROM, WORK  # ty: ignore[unresolved-import]
from e00653_inject import OFFSET_2019  # ty: ignore[unresolved-import]
from e00653_segment import Event  # ty: ignore[unresolved-import]

if TYPE_CHECKING:
    from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]

BLOCK: Final[int] = 12
H_ARMS: Final[tuple[str, ...]] = ("H-own", "H-nbr", "H-nbr-noramp", "H-nbr-noshared")
ARMS: Final[tuple[str, ...]] = (*H_ARMS, "PELT")
Arm = Literal["H-own", "H-nbr", "H-nbr-noramp", "H-nbr-noshared", "PELT"]


@dataclass(frozen=True)
class ArmParams:
    """The tuned values of one arm."""

    priors: hmm.Priors
    tau: float
    penalty: float = 0.0


@dataclass(frozen=True)
class Context:
    """The rig and the frozen quantities that every world shares."""

    rig: Rig
    sigma653: dict[str, float]
    cal_cm: pan.Calibration
    cal_sh: pan.Calibration
    neighbours: dict[str, tuple[str, ...]]


def load_context(*, sigma_from: Literal["tune", "whole"] = "tune") -> Context:
    """Load experiment 660's rig and freeze the calibrations of both inputs on the real record."""
    from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]

    rig = Rig()
    power = rig.panel.power
    sigma653 = json.loads((WORK / "noise_units.json").read_text())["sigma_block_mw"]
    return Context(
        rig=rig,
        sigma653=sigma653,
        cal_cm=pan.calibrate(rig=rig, power=power, arm_input="cm", sigma_from=sigma_from),
        cal_sh=pan.calibrate(rig=rig, power=power, arm_input="shared", sigma_from=sigma_from),
        neighbours={s: tuple(v) for s, v in rig.p653.neighbours.items()},
    )


def arm_params(*, tuned: dict, arm: str) -> ArmParams:
    """Build an arm's `ArmParams` from the dict stored in `params.json`."""
    own, nbr = tuned["h_own"], tuned["h_nbr"]
    base = {"rho_on": own["rho_on"], "rho_off": own["rho_off"], "rho_cont": own["rho_cont"]}
    if arm == "H-own":
        return ArmParams(priors=hmm.Priors(**base), tau=own["tau"])
    if arm == "PELT":
        return ArmParams(priors=hmm.Priors(**base), tau=1.0, penalty=tuned["pelt"]["penalty"])
    return ArmParams(
        priors=hmm.Priors(**base, kappa=nbr["kappa"], ramp_rule=arm != "H-nbr-noramp"),
        tau=nbr["tau"],
    )


def record_end_of(*, power: dict[str, np.ndarray], limit: int = GRID_LENGTH) -> int:
    """One past the last half-hour with a reading at any Primary, at most `limit`."""
    end = 0
    for v in power.values():
        ok = np.flatnonzero(np.isfinite(v[:limit]))
        end = max(end, int(ok[-1]) + 1 if len(ok) else 0)
    return end


@dataclass
class ArmResult:
    """Events (all, not yet called) and posteriors of one arm in one world."""

    events: list[Event]
    posts: dict[str, hmm.Posterior]
    blocks: pan.BlockPanel


class WorldDecoder:
    """Decodes one world with any arm, caching block panels and pass-1 posteriors."""

    def __init__(
        self,
        *,
        ctx: Context,
        power: dict[str, np.ndarray],
        cal_cm: pan.Calibration,
        cal_sh: pan.Calibration,
        resid_cm: dict[str, np.ndarray] | None = None,
        resid_raw: dict[str, np.ndarray] | None = None,
        n_blocks: int = pan.N_BLOCKS,
    ) -> None:
        """Hold the world's power, its optional residual overrides, and the cut length."""
        self.ctx, self.power, self.n_blocks = ctx, power, n_blocks
        self.cal = {"cm": cal_cm, "shared": cal_sh}
        self.override = {"cm": resid_cm, "shared": resid_raw}
        self._resid: dict[str, dict[str, np.ndarray]] = {}
        self._panels: dict[str, pan.BlockPanel] = {}
        self._pass1: dict[tuple, dict[str, hmm.Posterior]] = {}
        self.record_end = min(record_end_of(power=power), n_blocks * BLOCK)

    def resid(self, *, kind: str) -> dict[str, np.ndarray]:
        """The half-hourly input residual of `cm` or `shared`."""
        if kind not in self._resid:
            rig = self.ctx.rig
            given = self.override[kind]
            if given is not None:
                self._resid[kind] = given
            elif kind == "cm":
                self._resid[kind] = pan.common_mode_residual(rig=rig, power=self.power)
            else:
                self._resid[kind] = pan.raw_residual(rig=rig, power=self.power)
        return self._resid[kind]

    def panel(self, *, kind: str) -> pan.BlockPanel:
        """Block panel of an input, built once."""
        if kind not in self._panels:
            self._panels[kind] = pan.build_blocks(
                rig=self.ctx.rig,
                power=self.power,
                arm_input=kind,  # ty: ignore[invalid-argument-type]
                loading=None,
                switch=None,
                sigma_from="tune",
                calibration=self.cal[kind],
                resid=self.resid(kind=kind),
                n_blocks=self.n_blocks,
            )
        return self._panels[kind]

    def xh(self, *, kind: str, site: str) -> np.ndarray:
        """Half-hourly input minus the slow level and the shared term, megawatts."""
        p = self.panel(kind=kind)
        level = store.blocks_to_half_hours(level=p.slow[site] + p.shared[site])
        return self.resid(kind=kind)[site][: self.n_blocks * BLOCK] - level

    def _fb(
        self, *, kind: str, site: str, p: hmm.Priors, boost: tuple | None = None
    ) -> hmm.Posterior:
        panel = self.panel(kind=kind)
        return hmm.forward_backward(
            x=panel.observation(site=site),
            f=panel.forecast[site],
            sigma=panel.sigma[site],
            scale=panel.scale[site],
            p=p,
            boost_up=None if boost is None else boost[0],
            boost_down=None if boost is None else boost[1],
        )

    def pass1(self, *, kind: str, p: hmm.Priors) -> dict[str, hmm.Posterior]:
        """Pass-1 posteriors of every Primary with kappa 0, cached by input and prior setting."""
        key = (kind, p.rho_on, p.rho_off, p.rho_cont, p.ramp_rule)
        if key not in self._pass1:
            plain = hmm.Priors(
                rho_on=p.rho_on, rho_off=p.rho_off, rho_cont=p.rho_cont, ramp_rule=p.ramp_rule
            )
            self._pass1[key] = {
                s: self._fb(kind=kind, site=s, p=plain) for s in self.panel(kind=kind).sites
            }
        return self._pass1[key]

    def run_arm(self, *, arm: str, params: ArmParams) -> ArmResult:
        """Decode every Primary with one arm and return all of its events (before calling)."""
        if arm == "PELT":
            return self._run_pelt(params=params)
        kind = "cm" if arm in ("H-own", "H-nbr-noshared") else "shared"
        panel = self.panel(kind=kind)
        p = params.priors
        first = self.pass1(kind=kind, p=p)
        posts: dict[str, hmm.Posterior] = {}
        events: list[Event] = []
        for s in panel.sites:
            boost = None
            if arm != "H-own" and p.kappa > 0:
                boost = hmm.mirror_boost(
                    site=s,
                    pass1=first,
                    neighbours=panel.neighbours[s],
                    unknown=panel.unknown,
                    kappa=p.kappa,
                )
            post = first[s] if boost is None else self._fb(kind=kind, site=s, p=p, boost=boost)
            o_path, l_path, _ = hmm.viterbi(
                x=panel.observation(site=s),
                f=panel.forecast[s],
                sigma=panel.sigma[s],
                scale=panel.scale[s],
                p=p,
                boost_up=None if boost is None else boost[0],
                boost_down=None if boost is None else boost[1],
            )
            posts[s] = post
            events += ev.extract(
                site=s,
                o_path=o_path,
                l_path=l_path,
                post=post,
                xh=self.xh(kind=kind, site=s),
                sigma=panel.sigma[s],
                record_end=self.record_end,
            )
        return ArmResult(events=events, posts=posts, blocks=panel)

    def _run_pelt(self, *, params: ArmParams) -> ArmResult:
        panel = self.panel(kind="cm")
        events: list[Event] = []
        for s in panel.sites:
            events += pl_pelt.pelt_events(
                site=s,
                y=panel.y[s],
                slow=panel.slow[s],
                sigma=panel.sigma[s],
                xh=self.xh(kind="cm", site=s),
                penalty_units=params.penalty,
                record_end=self.record_end,
            )
        return ArmResult(events=events, posts={}, blocks=panel)

    def final_events(self, *, result: ArmResult, tau: float) -> list[Event]:
        """The called events of an arm, graded."""
        return ev.grade(
            events=ev.called(events=result.events, tau=tau), neighbours=self.ctx.neighbours
        )


def events_frame(*, events: list[Event]) -> pl.DataFrame:
    """A table of events with their confidence and grade."""
    rows = [
        {
            "primary": e.primary,
            "start": e.start,
            "end": -1 if e.end is None else e.end,
            "amp_mw": e.amp_mw,
            "confidence": e.confidence,
            "cls": e.cls,
            "grade": e.kind,
            "partners": ",".join(e.partners),
        }
        for e in events
    ]
    schema = {
        "primary": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
        "amp_mw": pl.Float64,
        "confidence": pl.Float64,
        "cls": pl.String,
        "grade": pl.String,
        "partners": pl.String,
    }
    return pl.DataFrame(rows, schema=schema)


def event_free_masks(*, rig: Rig, mask_tuning_years: bool = True) -> dict[str, np.ndarray]:
    """Boolean masks, per Primary, of half-hours within 14 days of a label or before 2024-01-01.

    Time outside the mask is "event-free" in the sense of experiment 670's `event_free`. With
    `mask_tuning_years=False` the years before 2024-01-01 are masked only near a label, which is
    the event-free time of 2022 and 2023 that stage 0 reads.
    """
    day = 48
    spans: list[tuple[str, int, int]] = []
    lab = pl.read_parquet(E580 / "events_labelled_status.parquet")
    spans += [
        (r["label"], r["label_onset_idx"] - OFFSET_2019, r["label_end_idx"] - OFFSET_2019)
        for r in lab.iter_rows(named=True)
    ]
    for r in pl.read_parquet(E580 / "events_verified.parquet").iter_rows(named=True):
        spans += [
            (r["primary_a"], r["onset_a_idx"] - OFFSET_2019, r["end_a_idx"] - OFFSET_2019),
            (r["primary_b"], r["onset_b_idx"] - OFFSET_2019, r["end_b_idx"] - OFFSET_2019),
        ]
    busy_by = {}
    for site in rig.p653.sites:
        busy = np.zeros(GRID_LENGTH, bool)
        for s, a, b in spans:
            if s == site:
                busy[max(0, a - 14 * day) : max(0, b + 14 * day)] = True
        if mask_tuning_years:
            busy[:SCORED_FROM] = True
        busy_by[site] = busy
    return busy_by

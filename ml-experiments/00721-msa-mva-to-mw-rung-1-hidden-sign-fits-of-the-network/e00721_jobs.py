# ruff: noqa: E501
"""Experiment 721: the arrays of an arm, its network, and the `JobInputs` that the fit loop runs.

`WorldInputs` implements `e00721_fit.JobInputs` over a function that builds one job's `ArmWorld`.
The real loader (`e00721_real`) and the synthetic mini world of the end-to-end check
(`e00721_e2e`) both supply that function, so the chain from batch to scored rows is one piece of code.
"""

from collections.abc import Callable

import e00721_paths  # noqa: F401

# isort: split
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_model as pvm  # ty: ignore[unresolved-import]
import e00721_arms as arms
import e00721_core as core
import e00721_fit as fit
import e00721_train as trn
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]


def arm_arrays(
    *, spec: core.ArmSpec, signed: np.ndarray, fold_view: np.ndarray, signed_view: np.ndarray
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return the arrays an arm trains on, in capacity fractions, from one world's three views.

    `signed_view` is the signed power at the visible signed Primaries and the folded reading
    elsewhere. Arm `S` also
    reads the signed truth at the hidden pair. `clean_fold` is the all-folded history that a supervised Primary's
    sample reads when its history is folded.
    """
    if spec.readings == "signed":
        target = np.where(np.isfinite(signed), signed, fold_view)
        plain = target
    else:
        target = signed_view
        plain = signed_view
    return {"target": target, "plain": plain, "fold": fold_view}


def build_model(
    *, spec: core.ArmSpec, seed: int, gate: np.ndarray, shift: np.ndarray, device: str
) -> torch.nn.Module:
    """Return the network of an arm: `PvNet` `Z` for `P0`, `PvNet` `HT` for `S`, `SgNet` for the `SG` arms."""
    torch.manual_seed(seed)
    if spec.name == "P0":
        model: torch.nn.Module = pvm.PvNet(arm="Z", seed=seed)
    elif spec.name == "S":
        model = pvm.PvNet(arm="HT", seed=seed)
    else:
        sg = arms.SgNet(seed=seed, sign_head=spec.sign_head)
        sg.set_gate(gate=gate, force_open=spec.force_gate_open)
        sg.set_shift(shift=shift)
        model = sg
    return model.to(device)


class WorldInputs:
    """`JobInputs` over a builder of one job's `ArmWorld`."""

    def __init__(
        self, *, make_world: Callable[[fit.Job], trn.ArmWorld], device: str = "cpu"
    ) -> None:
        """Hold the builder; worlds, models, and samples are built once per job."""
        self.make_world, self.device = make_world, device
        self.worlds: dict[str, trn.ArmWorld] = {}
        self.models: dict[str, torch.nn.Module] = {}
        self.samples: dict[tuple[str, int], tr.SampleSet] = {}
        self.loaded: set[str] = set()
        """Jobs whose fitted weights were loaded: the predict phase, in which nothing is recalibrated or retrained."""

    def world_of(self, *, job: fit.Job) -> trn.ArmWorld:
        """Return the job's `ArmWorld`, built on first use."""
        if job.key not in self.worlds:
            self.worlds[job.key] = self.make_world(job)
        return self.worlds[job.key]

    def model_of(self, *, job: fit.Job) -> torch.nn.Module:
        """Return the job's network, built on first use."""
        if job.key not in self.models:
            aw = self.world_of(job=job)
            self.models[job.key] = build_model(
                spec=aw.spec, seed=job.seed, gate=aw.gate, shift=aw.shift, device=self.device
            )
        return self.models[job.key]

    def stages(self, *, job: fit.Job) -> tuple[int, ...]:
        """Return (1,) in the synthetic world and (1, 2, 3) on a real fold."""
        return (1,) if job.world == "synthetic" else (1, 2, 3)

    def build_trainable(self, *, job: fit.Job, stage: int) -> trn.StageTrainable:
        """Return the stage's trainable; the first stage also sets the scale of the PV paths."""
        assert job.key not in self.loaded, (
            f"{job.key}: the predict phase never builds a trainable (it would recalibrate the PV scales)"
        )
        aw = self.world_of(job=job)
        model = self.model_of(job=job)
        key = (job.key, stage)
        if key not in self.samples:
            world = tr.World(
                source=aw.sources[stage], cutoff_idx=aw.world.cutoff_idx, clean=aw.world.clean,
                target_mean=aw.world.target_mean, target_std=aw.world.target_std, month=aw.world.month,
                placebo_pool=None,
            )  # fmt: skip
            self.samples[key] = tr.make_samples(
                world=world, stage=stage, placebo=False, primaries=aw.train_primaries
            )
            if stage == 1:
                aw.log["scale"] = trn.pv_calibration(
                    model=model, aw=aw, samples=self.samples[key], source=aw.sources[1], cut=tr.target_end(world=aw.world, stage=1)
                )  # fmt: skip
        aw.world.source = aw.sources[stage]
        return trn.StageTrainable(model=model, aw=aw, stage=stage, samples=self.samples[key])

    def predict(self, *, job: fit.Job) -> pl.DataFrame:
        """Return the scored rows of the job's current network."""
        aw = self.world_of(job=job)
        assert aw.score_c_idx is not None
        assert aw.stamp_us is not None
        rows = trn.predict_rows(
            model=self.model_of(job=job), aw=aw, c_idx=aw.score_c_idx, primaries=aw.score_primaries, stamp_us=aw.stamp_us,
        )  # fmt: skip
        return rows.with_columns(
            pl.lit(job.fold).alias("fold"), pl.lit(job.seed, dtype=pl.Int64).alias("seed"),
            pl.lit(job.arm).alias("arm"), pl.lit(job.world).alias("world"),
        )  # fmt: skip

    def network_state(self, *, job: fit.Job) -> dict:
        """Return a CPU copy of the job's network state."""
        return {k: v.detach().cpu().clone() for k, v in self.model_of(job=job).state_dict().items()}

    def load_network_state(self, *, job: fit.Job, state: dict) -> None:
        """Load `state` into the job's network."""
        self.model_of(job=job).load_state_dict(state)
        self.loaded.add(job.key)

    def sign_tokens(self, *, job: fit.Job) -> dict[str, np.ndarray] | None:
        """Return the sign tokens of a synthetic `SG-A` fit, or `None` for any other job.

        The tokens are the exporters' `ρ`, and the non-exporters' `ρ` with the gate forced open, on
        tokens that no training batch read. This method rebuilds no sample set and reruns no
        calibration, so the loaded network's weights and PV scales are the fitted ones.
        """
        aw = self.world_of(job=job)
        if job.world != "synthetic" or not aw.spec.sign_head:
            return None
        assert aw.score_c_idx is not None
        model = self.model_of(job=job)
        exp = trn.sign_tokens_of(
            model=model,
            aw=aw,
            primaries=aw.exporters,
            c_idx=aw.score_c_idx,
            cutoff_idx=aw.world.cutoff_idx,
        )
        old = model.export_gate.clone()
        model.set_gate(gate=np.ones(16), force_open=True)
        non = trn.sign_tokens_of(
            model=model,
            aw=aw,
            primaries=aw.non_exporters,
            c_idx=aw.score_c_idx,
            cutoff_idx=aw.world.cutoff_idx,
        )
        model.export_gate.copy_(old)
        return {f"exporters_{k}": v for k, v in exp.items()} | {
            f"open_{k}": v for k, v in non.items()
        }

# ruff: noqa: E501, PT018, PLR0915, ANN003
"""Experiment 716: KF13 and KF13b, the copied training loop against experiment 713's (WP4 check; CPU, synthetic world).

Usage: run.sh e00716_check_train.py

KF13: experiment 713's `train_stage` and this experiment's copy, with every PV path absent (network
`Z`), train stage 2 for one pass (about 40 steps at a batch of 32) from one stored state with one seed
and one sample set, under `torch.use_deterministic_algorithms(True)`. The two state dictionaries must be
bit-identical.

KF13b: the same, with the full network of arm `HT` (all three paths, random non-zero PV tables) and
the new group's rate scaled to zero. Every PV contribution is an exact zero at the start, so the trunk
must come out bit-identical to experiment 713's. This tests the two optimiser groups, the two clips,
the dedicated generators, and the mask-dropout generator.
"""

import copy
import sys

import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_batch as pvb
import e00716_check_model as ckm
import e00716_model as pvm
import e00716_train as pvt
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

N_DAYS_GRID = 170
FIRST_DAY = 40
N_ENS_DAYS = 100
SEED = 0


def synthetic_world(*, seed: int = 0) -> tuple[tr.World, pvb.PvTables]:
    """Return a world of 170 days with ensemble rows on days 40 to 139, and random non-zero PV tables."""
    g = torch.Generator().manual_seed(seed)
    n_grid = 48 * N_DAYS_GRID
    power = 6.0 * torch.randn(
        16, n_grid, generator=g
    )  # a wide target, so that the trunk gradient clip of 1.0 binds
    power[torch.rand(16, n_grid, generator=g) < 0.02] = float("nan")
    run = torch.zeros(16, N_ENS_DAYS, dtype=torch.int64)
    c = (FIRST_DAY + torch.arange(N_ENS_DAYS)) * 48 + 24
    run[:] = (
        lay.GRID_START_US + (c - lay.NATIVE_OFFSET_STEPS) * lay.STEP_US - lay.ENSEMBLE_LEAD_US
    )[None, :]
    ens = bt.EnsTensors(
        weather=torch.randn(16, N_ENS_DAYS, 672, 21, generator=g).half(),
        spread=torch.rand(16, N_ENS_DAYS, 672, 7, generator=g).half(),
        filled=torch.ones(16, N_ENS_DAYS, 672, dtype=torch.bool), first_day=FIRST_DAY, run_us=run,
    )  # fmt: skip
    source = bt.Source(
        power_z=power, dense_weather=torch.randn(16, n_grid, 21, generator=g).half(),
        dense_datetime=torch.randn(16, n_grid, 21, generator=g).half(),
        day_type=torch.randint(0, 8, (n_grid,), generator=g), ens=ens,
    )  # fmt: skip
    tables = pvb.PvTables(
        hist_p=torch.rand(16, n_grid, generator=g) * 4, hist_ok=torch.rand(16, n_grid, generator=g) > 0.1,
        top_p0=torch.rand(16, N_ENS_DAYS, 672, 2, generator=g) * 3,
        top_filled=torch.ones(16, N_ENS_DAYS, 672, dtype=torch.bool), first_day=FIRST_DAY, run_us=run,
    )  # fmt: skip
    world = tr.World(
        source=source, cutoff_idx=48 * 140, clean=power.numpy().astype(np.float64),
        target_mean=np.zeros(16), target_std=np.ones(16), month="2025-11", placebo_pool=None,
    )  # fmt: skip
    return world, tables


def state_equal(*, a: dict, b: dict, keys: list[str]) -> list[str]:
    """Return the keys whose tensors differ in any bit."""
    return [k for k in keys if not torch.equal(a[k], b[k])]


def check_groups() -> None:
    """KF8's group half: the optimiser groups, rates, decays, warm-up flags, and clips equal design section 6."""
    for encoders in ("trained", "frozen"):
        for stage in (2, 3):
            m = pvm.PvNet(arm="HT", seed=0)
            if encoders == "frozen":
                m.freeze_encoders()
            groups = pvt.param_groups(model=m, stage=stage, encoders=encoders)
            names = {id(p): n for n, p in m.named_parameters()}
            by_name = {names[id(p)]: g for g in groups for p in g["params"]}
            for n, g in by_name.items():
                if n.startswith(pvm.NEW_PREFIX):
                    lr, wd = pvt.NEW_RATES.get(n, pvt.NEW_DEFAULT)
                    assert (g["lr"], g["weight_decay"], g["warmup"], g["trunk"]) == (
                        lr,
                        wd,
                        False,
                        False,
                    ), (n, g["lr"])
                    assert (lr, wd) in {(1e-2, 0.0), (3e-3, 0.0), (2e-3, 1e-3)}, (n, lr, wd)
                else:
                    assert (
                        g["trunk"] and g["warmup"] and g["weight_decay"] == tr.WEIGHT_DECAY == 1e-3
                    ), n
                    spread = n.startswith("proj_s")
                    want = tr.SPREAD_LR if (spread and stage == 2) else tr.FINE_TUNE_LR
                    if encoders == "trained" and "encoder" in n:
                        want = tr.FINE_TUNE_LR
                    assert g["lr"] == want, (n, g["lr"], want)
            gains = {
                n: (g["lr"], g["weight_decay"]) for n, g in by_name.items() if n in pvt.GAIN_NAMES
            }
            assert gains and all(v[1] == 0.0 for v in gains.values()), gains
            assert tr.GRAD_CLIP == 1.0
    print(
        "KF8 groups: trunk 2e-4 (proj_s 5e-4 in stage 2) with decay 1e-3 and warm-up; gains 1e-2 and 3e-3 with decay 0; W1, W2, pv_head_in 2e-3 with decay 1e-3; no warm-up on the new groups; clip 1.0 on each group alone"
    )


class ClipRecorder:
    """Record the norm that `clip_grad_norm_` returns, and optionally clip globally (the negative control)."""

    def __init__(self, *, global_params: list | None = None) -> None:
        """Hold the norms and the all-parameter list of a global clip."""
        self.norms: list[float] = []
        self.global_params = global_params
        self.original = torch.nn.utils.clip_grad_norm_

    def __call__(self, params, max_norm, *args, **kwargs):  # noqa: ANN001, ANN002, ANN204
        """Clip as `torch` does, recording the norm of each call (or clipping every parameter together)."""
        if self.global_params is not None:
            return self.original(self.global_params, max_norm, *args, **kwargs)
        out = self.original(params, max_norm, *args, **kwargs)
        self.norms.append(float(out))
        return out


def run_mode(*, encoders: str) -> None:
    """Run KF13, KF13b, the live-group control, and the global-clip control with `encoders` trained or frozen."""
    torch.use_deterministic_algorithms(True)
    tr.BATCH = 32  # a smaller batch gives about 40 steps in one pass
    world, tables = synthetic_world()
    samples = tr.make_samples(world=world, stage=2, placebo=False)
    assert (~samples.is_val).sum() > 400 and samples.is_val.sum() > 20, (
        len(samples),
        samples.is_val.sum(),
    )
    start = ckm.randomised_m_star(seed=3).state_dict()

    ref = mdl.MsaNet()
    ref.load_state_dict(start)
    if encoders == "frozen":
        ref.freeze_encoders()
    recorder = ClipRecorder()
    torch.nn.utils.clip_grad_norm_ = recorder  # ty: ignore[invalid-assignment]
    try:
        tr.train_stage(
            model=ref,
            world=world,
            stage=2,
            samples=samples,
            seed=SEED,
            encoders=encoders,
            max_passes=1,
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = recorder.original
    ref_state = copy.deepcopy(ref.state_dict())
    start_copy = copy.deepcopy(start)
    moved = [k for k in start if not torch.equal(ref_state[k], start_copy[k])]
    assert len(moved) > 20, "the reference loop barely moved the weights"
    bound = sum(1 for x in recorder.norms if x > tr.GRAD_CLIP)
    assert recorder.norms[0] > tr.GRAD_CLIP and bound > len(recorder.norms) // 2, (
        "the trunk clip does not bind, so a global clip would pass the check",
        recorder.norms[:5],
    )
    print(
        f"[{encoders}] the trunk gradient norm at step 1 is {recorder.norms[0]:.2f}; the clip of {tr.GRAD_CLIP} binds in {bound} of {len(recorder.norms)} steps"
    )

    pv_world = tr.World(
        source=pvb.PvSource(source=world.source, tables=tables), cutoff_idx=world.cutoff_idx, clean=world.clean,
        target_mean=world.target_mean, target_std=world.target_std, month=world.month, placebo_pool=None,
    )  # fmt: skip

    def pv_model(arm: str) -> pvm.PvNet:
        model = pvm.PvNet(arm=arm, seed=SEED)  # ty: ignore[invalid-argument-type]
        model.load_m_star(state=copy.deepcopy(start))
        if encoders == "frozen":
            model.freeze_encoders()
        return model

    for arm, scale, label in (("Z", 1.0, "KF13"), ("HT", 0.0, "KF13b")):
        model = pv_model(arm)
        pvt.train_stage(
            model=model,
            world=pv_world,
            stage=2,
            samples=samples,
            seed=SEED,
            encoders=encoders,
            max_passes=1,
            new_lr_scale=scale,
        )
        got = model.state_dict()
        bad = state_equal(a=got, b=ref_state, keys=list(ref_state))
        assert not bad, (
            f"{label} [{encoders}]: the trunk differs from experiment 713's in {bad[:5]}"
        )
        if arm == "HT":
            assert [k for k in got if k not in ref_state], "no new parameters in HT"
        print(
            f"{label} [{encoders}]: {len(ref_state)} trunk tensors bit-identical to experiment 713's after one pass ({int((~samples.is_val).sum()) // 32} steps)"
        )

    # Negative control 1: a live new group (scale 1) must change the trunk, or the check could not fail.
    model = pv_model("HT")
    pvt.train_stage(
        model=model,
        world=pv_world,
        stage=2,
        samples=samples,
        seed=SEED,
        encoders=encoders,
        max_passes=1,
        new_lr_scale=1.0,
    )
    bad = state_equal(a=model.state_dict(), b=ref_state, keys=list(ref_state))
    assert bad, "negative control: a live PV group left the trunk unchanged"
    print(
        f"negative control [{encoders}]: with the new group live, {len(bad)} trunk tensors differ"
    )

    # Negative control 2: a global clip (one norm over the trunk and the new group) must change the trunk.
    model = pv_model("HT")
    torch.nn.utils.clip_grad_norm_ = ClipRecorder(  # ty: ignore[invalid-assignment]
        global_params=[p for p in model.parameters() if p.requires_grad]
    )  # ty: ignore[invalid-assignment]
    try:
        pvt.train_stage(
            model=model,
            world=pv_world,
            stage=2,
            samples=samples,
            seed=SEED,
            encoders=encoders,
            max_passes=1,
            new_lr_scale=0.0,
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = recorder.original
    bad = state_equal(a=model.state_dict(), b=ref_state, keys=list(ref_state))
    assert bad, (
        "negative control: a global clip left the trunk bit-identical, so KF13b cannot see the two clips"
    )
    print(f"negative control [{encoders}]: with one global clip, {len(bad)} trunk tensors differ")


def run() -> None:
    """Run KF13 and KF13b with trained and with frozen encoders, and the group half of KF8."""
    check_groups()
    for encoders in ("trained", "frozen"):
        run_mode(encoders=encoders)


def main() -> int:
    """Run the checks."""
    run()
    print("e00716_check_train: all checks passed")
    if "--record" in sys.argv:
        for name in ("KF13", "KF13b"):
            arms.write_wiring_record(data_dir=ML_EXPERIMENTS_DATA_DIR, name=name, passed=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ruff: noqa: E501
"""One real epoch of every arm on the CPU, from the real stored banks, neighbour table, and overlays.

The invented-bank tests cannot reach what only the real files hold, so this test builds the
environment exactly as the chain does (`real_env` with the real-day sets, the real neighbours, the
medians, and the zero-fill mask) and fits one epoch of each arm into a throwaway folder. It needs the
private data folder, so it runs only when `RUN_REAL_FIT=1` (on the workstation, with
`CUDA_VISIBLE_DEVICES=` and a slot's cores):

    DATA_PATH_INTERNAL=<data folder> RUN_REAL_FIT=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=<this experiment's folder> \
    taskset -c 16-21 <python of the gpu venv> -m pytest --run-studies tests/test_e00732_real_fit.py

It reads the data folder and writes only under `tmp_path`.
"""

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

import e00732_cli as cli
import e00732_core as core
import e00732_fit as fit
import e00732_stages as stages

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_FIT") != "1",
    reason="reads the private data folder; set RUN_REAL_FIT=1",
)

ALL_ARMS = tuple(core.ARMS)


@pytest.fixture(scope="module")
def real(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, stages.Env]:
    """Return a copy of the gate records in a throwaway folder, experiment 730's folder, and the real environment."""
    real_root = cli.paths.data_root()
    scratch = tmp_path_factory.mktemp("real_fit")
    for item in real_root.iterdir():
        if item.name not in ("fits", "figures", "networks", "predictions"):
            target = scratch / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy(item, target)
    ctx = cli.real_context()
    base = cli.real_base(ctx=ctx)
    env = cli.real_env(root=real_root, ctx=ctx, base=base, with_real=True)
    return scratch, cli.paths.root_730(), env


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_one_real_epoch_of_each_arm_stores_a_record_and_predictions(
    real: tuple[Path, Path, stages.Env], arm: str
) -> None:
    scratch, root730, env = real
    record = fit.fit_one(
        root=scratch,
        root730=root730,
        env=env,
        arm=arm,
        seed=0,
        device="cpu",
        log=print,
        max_epochs=1,
    )
    assert record["epochs"] == 1
    assert (scratch / "networks").is_dir()
    assert any((scratch / "predictions").glob("*/test.npz"))

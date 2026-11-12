# ruff: noqa: E501
"""Experiment 696: the pre-training hooks that `e00696_k1.py` installs on experiment 574's fit module.

`StateHooks` wraps two functions of `e00574_fit`: the pass-through wrapper of `e00448_run.pretrain_rows`
keeps the arrays that locate each pre-training row, and the wrapper of `train` adds the engine-state
columns, their checks, and the encoder probe to a pre-training call. A stage 2 or stage 3 call passes
through after a check that its input has the original column layout. The class takes everything it
reads from the environment as arguments, so the checks of `e00696_checks.py` run it on synthetic tensors
with a fake `train`.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00695_common import READING_DELAY  # ty: ignore[unresolved-import]
from e00696_common import N_STATE, Arm
from e00696_probe import probe_model
from e00696_state import hide_half_of_origin_days, rows_state

ALIGNMENT_SAMPLE = 10_000
ROW_ARRAYS = ("primary", "target_idx", "origin_idx", "y", "is_val")


class StateHooks:
    """The state columns, checks, and probe of one arm, one fold, and one process."""

    def __init__(
        self, *, spec: Arm, month: str, run_root: Path, state_file: Path, host_row: int | None,
        raw_stack: Callable[[], np.ndarray], stored_pretrained: Callable[[int], Path] | None,
        probe_only: bool,
    ) -> None:  # fmt: skip
        """Keep the arm's settings.

        Args:
            spec: The arm.
            month: The fold's test month.
            run_root: The arm's run folder, where the check and probe records go.
            state_file: The arm's state file for the fold (unused when the arm reads no state).
            host_row: The host's stack row, for the probe arms.
            raw_stack: Returns the pipeline's stack with the host's real reading and no engine.
            stored_pretrained: Maps a seed to experiment 695's stored pre-trained weights, for the
                probe-only arm.
            probe_only: Whether the pre-training call loads stored weights and runs the probe alone.
        """
        self.spec, self.month, self.run_root = spec, month, run_root
        self.state_file, self.host_row, self.raw_stack = state_file, host_row, raw_stack
        self.stored_pretrained, self.probe_only = stored_pretrained, probe_only
        self.capture: dict[str, np.ndarray] = {}
        self.wide: torch.Tensor | None = None
        self.grid: np.ndarray | None = None
        self.raw_row: np.ndarray | None = None

    def wrap_pretrain_rows(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Return `original` with a pass-through that keeps the rows' locating arrays."""

        def pretrain_rows(**kwargs: Any) -> Any:
            pre = original(**kwargs)
            for name in ROW_ARRAYS:
                self.capture[name] = np.asarray(getattr(pre, name)).copy()
            return pre

        return pretrain_rows

    def write_check(self, *, name: str, content: dict) -> None:
        """Write one check's record under the arm's run folder."""
        folder = self.run_root / "checks"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.json").write_text(json.dumps(content, indent=1, default=float))

    def state_rows(self) -> tuple[np.ndarray, int, int]:
        """Return the arm's state columns of every pre-training row, the series' row, and the fold's refit half-hour."""
        stored = np.load(self.state_file)
        row, refit = int(stored["row"]), int(stored["refit"])
        self.grid = stored["state"]
        out = rows_state(
            grid_state=self.grid, row=row, primary=self.capture["primary"],
            target_idx=self.capture["target_idx"],
        )  # fmt: skip
        if self.spec.kind == "hidden":
            out, _ = hide_half_of_origin_days(
                state=out,
                origin_idx=self.capture["origin_idx"],
                seed=int(self.month.replace("-", "")),
                rows=self.capture["primary"] == row,
            )
        return out, row, refit

    def check_alignment(self, *, wide: torch.Tensor, width: int, row: int, refit: int) -> dict:
        """Check the state columns of 10,000 sampled pre-training rows against the state file, row by row.

        A row of the series reads the file's value at its own target half-hour, every other row reads
        zeros, a hidden row (arm S1d) reads zeros and the flag, and every row's target half-hour is at
        or before the fold's refit minus 3.5 hours.
        """
        assert self.grid is not None
        target_idx, primary = self.capture["target_idx"], self.capture["primary"]
        assert int(target_idx.max()) <= refit - READING_DELAY, (target_idx.max(), refit)
        rng = np.random.default_rng(0)
        sample = rng.choice(len(primary), size=min(ALIGNMENT_SAMPLE, len(primary)), replace=False)
        got = wide[torch.from_numpy(sample).to(wide.device), width:].cpu().numpy()
        worst, n_mine, n_nonzero = 0.0, 0, 0
        for i, r in enumerate(sample):
            if int(primary[r]) == row:
                n_mine += 1
                hidden = self.spec.kind == "hidden" and got[i, N_STATE - 1] == 1.0
                want = (
                    np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    if hidden
                    else self.grid[int(target_idx[r])]
                )
            else:
                want = np.zeros(N_STATE, dtype=np.float32)
            worst = max(worst, float(np.abs(got[i] - want).max()))
            n_nonzero += int(np.abs(got[i, : N_STATE - 1]).max() > 0.0)
        assert worst == 0.0, f"a state column is misaligned with its row (largest gap {worst})"
        return {
            "sampled_rows": len(sample), "rows_of_the_series": n_mine,
            "rows_with_a_nonzero_state": n_nonzero, "largest_gap": worst,
            "latest_target_idx": int(target_idx.max()), "refit_minus_delay": refit - READING_DELAY,
        }  # fmt: skip

    def check_liveness(self, *, model: torch.nn.Module) -> dict:
        """Return the largest first-layer `mixer_pre` weight on each state column; assert it is live."""
        weight = model.mixer_pre[0].weight.detach()[:, -N_STATE:]  # ty: ignore[not-subscriptable, unresolved-attribute]
        largest = weight.abs().amax(dim=0).cpu().tolist()
        live = all(v > 0.0 for v in largest[: N_STATE - 1])
        if self.spec.kind == "zero":
            assert all(v == 0.0 for v in largest), f"arm SZ moved a state weight: {largest}"
        else:
            assert live, (
                f"arm {self.spec.name}: a state weight stayed at zero, so the patch did not take: {largest}"
            )
        return {"largest_abs_weight_per_state_column": largest, "live": live}

    def raw_points_of_host(self) -> np.ndarray:
        """Return each pre-training row's host reading without the engine, in points of capacity."""
        assert self.host_row is not None
        if self.raw_row is None:
            self.raw_row = np.asarray(self.raw_stack()[self.host_row])
        raw = self.raw_row
        mine = self.capture["primary"] == self.host_row
        out = np.full(len(mine), np.nan, dtype=np.float32)
        out[mine] = raw[self.capture["target_idx"][mine]] * 100.0
        return out

    def wrap_train(
        self, original: Callable[..., tuple[int, float]]
    ) -> Callable[..., tuple[int, float]]:
        """Return experiment 574's `train` with the state columns, their checks, and the probe added in pre-training."""

        def train(
            *, model: torch.nn.Module, x: torch.Tensor, seed: int, **kwargs: Any
        ) -> tuple[int, float]:
            if getattr(model, "stage", None) != "pre":
                if self.spec.kind != "none":
                    label = int(getattr(model, "label_col", -1))
                    assert x.shape[1] in {label, label + 2}, (
                        "a stage 2 or stage 3 tensor carries an engine-state column",
                        tuple(x.shape),
                    )
                return original(model=model, x=x, seed=seed, **kwargs)
            width = x.shape[1]
            record: dict = {
                "arm": self.spec.name,
                "month": self.month,
                "seed": seed,
                "rows": len(x),
            }
            x_for_training = x
            if self.spec.kind != "none":
                if self.wide is None or self.wide.shape[0] != len(x):
                    self.wide = None
                    state, row, refit = self.state_rows()
                    wide = torch.empty((len(x), width + N_STATE), dtype=x.dtype, device=x.device)
                    wide[:, width:] = torch.from_numpy(state).to(x.device)
                    self.wide = wide
                    record["alignment"] = self.check_alignment(
                        wide=wide, width=width, row=row, refit=refit
                    )
                self.wide[:, :width].copy_(x)
                x_for_training = self.wide
            if self.probe_only:
                assert self.stored_pretrained is not None
                stored = self.stored_pretrained(seed)
                model.load_state_dict(torch.load(stored, map_location=x.device))
                epochs, best = 0, float("nan")
                record["loaded_from"] = str(stored)
            else:
                epochs, best = original(model=model, x=x_for_training, seed=seed, **kwargs)
            record |= {"epochs": epochs, "val_nll": best}
            if self.spec.kind != "none":
                record["liveness"] = self.check_liveness(model=model)
            if self.spec.probe:
                assert self.host_row is not None
                record["probe"] = probe_model(
                    model=model, x=x, primary=self.capture["primary"], raw_points=self.raw_points_of_host(),
                    is_val=self.capture["is_val"], target_idx=self.capture["target_idx"],
                    host_row=self.host_row, seed=seed,
                )  # fmt: skip
                (self.run_root / "probe").mkdir(parents=True, exist_ok=True)
                (self.run_root / "probe" / f"{self.month}_s{seed}.json").write_text(
                    json.dumps(record["probe"], indent=1, default=float)
                )
            self.write_check(name=f"{self.month}_s{seed}_pretrain", content=record)
            print(
                f"E00696 {self.spec.name} {self.month} s{seed}: {json.dumps(record, default=float)}",
                flush=True,
            )
            return epochs, best

        return train

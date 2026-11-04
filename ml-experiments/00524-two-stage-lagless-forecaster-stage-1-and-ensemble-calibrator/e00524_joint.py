# ruff: noqa: E501
r"""S2-J of experiment 524: stage 1 and the stage-2 form-B head fine-tuned jointly on ENS rows.

Each weekly refit starts from the fold's frozen-stage-1 weights (`e00524_stage1.py`, ERA5 and CAMS
rows only, no lead, no spread) and a zero-initialised form-B head, then trains both on the cutoff's
ENS rows under experiment 510's row rule with the negative log-likelihood. Stage 1 trains at a
fraction of the pretraining rate (5e-4): `0.4` is experiment 511's L-1d fine-tune rate (2e-4, so
only the missing lead input differs from L-1d), `0.1` is the second arm. The head trains at the
stage-2 rate (3e-3). Early stopping uses the held-out weeks, as experiment 510 does. No cross-fit is
needed, because stage 1 and the head are fitted on the same rows.

```bash
cd <worktree>; source ml-experiments/00524-*/env.sh
systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00524-*/e00524_joint.py --ratio 0.4
```
"""

import argparse
import time
from datetime import datetime
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as fit510  # ty: ignore[unresolved-import]
import e00524_stage1 as s1  # ty: ignore[unresolved-import]
import e00524_stage2 as s2  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

K: Final[int] = c.COMPONENTS
N_AUX: Final[int] = 6


class Joint(torch.nn.Module):
    """Stage 1 followed by the form-B head; input columns are stage 1's inputs, then the 6 side inputs."""

    def __init__(self, *, stage1: torch.nn.Module, n_stage1_inputs: int) -> None:
        """Wrap a stage-1 network with a zero-initialised head."""
        super().__init__()
        self.stage1 = stage1
        self.head = s2.FormB()
        self.n = n_stage1_inputs

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Return the calibrated 9 raw outputs."""
        out = self.stage1(x[:, : self.n], p)
        return self.head(out, x[:, self.n :])


def train_joint(
    *, model: Joint, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
    is_val: torch.Tensor, seed: int, stage1_lr: float, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Train with weighted NLL and early stopping (experiment 510's loop with two learning rates)."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        val_idx = val_idx[
            torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS].to(device)
        ]
    opt = torch.optim.AdamW(
        [
            {"params": model.stage1.parameters(), "lr": stage1_lr},
            {"params": model.head.parameters(), "lr": s2.STAGE2_LR},
        ],
        weight_decay=c.WEIGHT_DECAY,
    )
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            nll = base.mixture_nll(out=model(x[b], p[b]), z=z[b], components=K)
            loss = (w[b] * nll).sum() / w[b].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        base.mixture_nll(out=model(x[i], p[i]), z=z[i], components=K)
                        for i in val_idx.split(c.BATCH * 2)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def run(*, months: list[str], seeds: list[int], ratio: float, device: str) -> None:
    """Fit S2-J at each weekly cutoff and write the arm's month forecasts."""
    arm = f"S2-J-{ratio:g}x"
    clean = c.clean_stack()
    for month in months:
        todo = [
            s
            for s in seeds
            if not (s1.WORK_DIR / "seed_preds" / arm / f"{month}_s{s}.parquet").exists()
        ]
        if not todo:
            continue
        mr = c.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        columns = list(mr.base.columns)
        n_primaries = len(mr.base.target_mean)
        aux_all = s2.aux_inputs(x=mr.x, columns=columns, lead=mr.lead, primary=mr.primary)
        aux_test = s2.aux_inputs(x=test.x, columns=columns, lead=test.lead, primary=test.primary)
        for seed in todo:
            _, keep = s2.load_stage1(
                path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries
            )
            x_all = np.concatenate([np.ascontiguousarray(mr.x[:, keep]), aux_all], axis=1)
            x_test = torch.from_numpy(
                np.concatenate([np.ascontiguousarray(test.x[:, keep]), aux_test], axis=1)
            )
            d = {
                "primary_test": test.primary, "origin_test": test.origin, "valid_test": test.valid,
                "target_mean": mr.base.target_mean, "target_std": mr.base.target_std,
            }  # fmt: skip
            frames = []
            for week, cut in enumerate(c.cutoffs(month=month)):
                started = time.monotonic()
                refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=c.HALF_LIFE_90)
                assert refit.report["rows_past_valid_cut"] == 0
                assert refit.report["rows_past_origin_cut"] == 0
                idx = refit.index
                prim = mr.primary[idx]
                z = (refit.y - mr.base.target_mean[prim]) / mr.base.target_std[prim]
                t = {
                    "x": torch.from_numpy(x_all[idx]).to(device),
                    "z": torch.from_numpy(z.astype(np.float32)).to(device),
                    "w": torch.from_numpy(refit.weight).to(device),
                    "p": torch.from_numpy(prim).to(device),
                    "v": torch.from_numpy(refit.is_val).to(device),
                }
                stage1, _ = s2.load_stage1(
                    path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries
                )
                model = Joint(stage1=stage1, n_stage1_inputs=len(keep)).to(device)
                epochs, best = train_joint(
                    model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                    stage1_lr=ratio * c.PRETRAIN_LR, device=device,
                )  # fmt: skip
                lo, hi = fit510.window_of(month=month, week=week)
                sel = (test.origin >= lo) & (test.origin < hi)
                out = np.zeros((len(test.primary), 3 * K), dtype=np.float32)
                with torch.no_grad():
                    rows = np.nonzero(sel)[0]
                    for a in range(0, len(rows), 65_536):
                        sl = rows[a : a + 65_536]
                        out[sl] = (
                            model(
                                x_test[sl].to(device), torch.from_numpy(test.primary[sl]).to(device)
                            )
                            .cpu()
                            .numpy()
                        )
                frames.append(s2.quantile_frame(out=out, d=d, select=sel))
                print(
                    f"{arm} {month} s{seed} w{week + 1}: {epochs} epochs, val NLL {best:.4f}, {len(idx):,} rows, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
                del t, model
                torch.cuda.empty_cache()
            s2.write_arm(arm=arm, month=month, seed=seed, frame=pl.concat(frames))
        del mr, test
    _ = datetime


def main() -> None:
    """Run S2-J."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--ratio", type=float, default=0.4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(months=args.months, seeds=args.seeds, ratio=args.ratio, device=args.device)


if __name__ == "__main__":
    main()

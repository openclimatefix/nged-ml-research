"""Recompute the mixture parameters of experiment 425's arm A networks (a forward pass, no fit).

Experiment 425 stored arm A's quantiles only, so a density mixture of arm A seeds with lagless
networks needs the Gaussian mixture itself. This script loads each saved arm A network (pooled
plain network, summary scheme), forecasts every test origin of a month exactly as experiment 425's
`fit397.forecast_month` did, and keeps the 3-component mixture in capacity fractions. It then
checks the recomputed 11 quantiles against `seed_preds/A` (maximum difference under 5e-4) and
writes `work/heads/A/<month>_s<seed>.parquet` under this experiment's private folder: the row keys
(`primary`, `origin`, `target` on experiment 404's grid) and the mixture.

```bash
source .e470env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00470-*/e00470_heads.py --month 2025-10 --seed 0
```
"""

import argparse
import json
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E425: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00425-best-mdn-architecture-assembly-with-ablations" / "work"
)
WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00470-first-release-demand-forecaster-assembly-and-comparison"
    / "work"
)
GRID_START_US: Final[int] = int(common.GRID_START.timestamp() * 1_000_000)
STEP_US: Final[int] = 1_800_000_000
CHECK_TOLERANCE: Final[float] = 5e-4
"""Largest accepted difference (capacity fraction) between a recomputed and a stored quantile."""
CHUNK: Final[int] = 8192


def head_path(*, month: str, seed: int) -> Path:
    """Return where an arm A head is stored."""
    return WORK_DIR / "heads" / "A" / f"{month}_s{seed}.parquet"


def load_network(
    *, month: str, seed: int
) -> tuple[fit397.Fitted, prep.Standardiser, np.ndarray, np.ndarray]:
    """Return arm A's pooled network, its standardiser, and the target mean and deviation."""
    folder = E425 / "models" / "A" / month / f"s{seed}"
    meta = json.loads((folder / "meta.json").read_text())
    setting = fit397.Setting(**meta["setting"])
    names, _, layout = prep.feature_names(scheme="summary")
    model = fit397.make_model(
        arm=meta["arm"], setting=setting, layout=layout,
        n_primaries=len(meta["target_mean"]), n_inputs=len(names),
    )  # fmt: skip
    model.load_state_dict(torch.load(folder / "pooled.pt", map_location="cpu"))
    model.eval()
    with np.load(folder / "standardiser.npz") as stored:
        standardiser = prep.Standardiser(
            mean=stored["mean"].copy(), std=stored["std"].copy(), mask=stored["mask"].copy()
        )
    fitted = fit397.Fitted(
        model=model, setting=setting, arm=meta["arm"], seed=seed,
        epochs=meta["epochs"]["None"], val_nll=meta["val_nll"]["None"],
    )  # fmt: skip
    return fitted, standardiser, np.array(meta["target_mean"]), np.array(meta["target_std"])


@torch.no_grad()
def forecast_heads(*, month: str, seed: int) -> pl.DataFrame:
    """Return the month's keys and 3-component mixture (capacity fractions), per valid time."""
    fitted, standardiser, t_mean, t_std = load_network(month=month, seed=seed)
    k = fitted.setting.components
    pieces = []
    for origin in prep.test_origins(month=month):
        for row, keys, x in prep.run_inputs(
            origin=origin, scheme="summary", standardiser=standardiser
        ):
            p = int(row["primary"])
            parts = []
            for start in range(0, len(x), CHUNK):
                xb = torch.from_numpy(x[start : start + CHUNK])
                pb = torch.full((len(xb),), p, dtype=torch.int64)
                out = fit397.forward(model=fitted.model, x=xb, primary=pb)
                log_w, mean, std = fit397.models.mixture_parts(out=out.double(), components=k)
                parts.append(torch.cat([log_w, mean, std], dim=1).numpy())
            raw = np.concatenate(parts)
            raw[:, k : 2 * k] = raw[:, k : 2 * k] * t_std[p] + t_mean[p]
            raw[:, 2 * k :] = raw[:, 2 * k :] * t_std[p]
            names = [f"{n}{i}" for n in ("lw", "mu", "sd") for i in range(k)]
            frame = keys.select(
                "time_series_id", "power_fcst_init_time", "valid_time"
            ).with_columns(primary=pl.lit(p, dtype=pl.Int16))
            pieces.append(frame.hstack(pl.DataFrame(raw, schema=names, orient="row")))
    out = pl.concat(pieces)
    us = lambda c: ((pl.col(c).dt.epoch("us") - GRID_START_US) // STEP_US).cast(pl.Int32)  # noqa: E731
    return out.with_columns(origin=us("power_fcst_init_time"), target=us("valid_time"))


def check_against_stored(*, frame: pl.DataFrame, month: str, seed: int) -> float:
    """Return the largest |recomputed - stored| quantile difference over the shared rows."""
    import e00404_models as models404  # ty: ignore[unresolved-import]

    stored = pl.read_parquet(E425 / "seed_preds" / "A" / f"{month}_s{seed}.parquet")
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    joined = frame.join(stored, on=keys, validate="1:1")
    assert joined.height == stored.height, "recomputed rows differ from stored rows"
    k = 3
    names = [f"{n}{i}" for n in ("lw", "mu", "sd") for i in range(k)]
    t = torch.tensor(joined.select(names).to_numpy(), dtype=torch.float64)
    q = models404.mixture_quantiles(log_w=t[:, :k], mean=t[:, k : 2 * k], std=t[:, 2 * k :])
    q = torch.sort(q, dim=1).values.numpy()
    levels = [c for c in stored.columns if c.startswith("q")]
    return float(np.abs(q - joined.select(levels).to_numpy()).max())


def main() -> None:
    """Recompute, check, and store one head."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.time()
    frame = forecast_heads(month=args.month, seed=args.seed)
    err = check_against_stored(frame=frame, month=args.month, seed=args.seed)
    print(f"{args.month} s{args.seed}: {frame.height} rows, max |recomputed - stored| {err:.2e}, "
          f"{time.time() - started:.0f} s", flush=True)  # fmt: skip
    assert err < CHECK_TOLERANCE, f"arm A head differs from stored quantiles by {err}"
    target = head_path(month=args.month, seed=args.seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target.with_name(target.name + ".tmp"))
    target.with_name(target.name + ".tmp").rename(target)


if __name__ == "__main__":
    main()

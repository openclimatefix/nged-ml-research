# ruff: noqa: E501
"""Fit the band heads month by month on as-of residuals and write their quantiles.

Arms: `x` XGBoost quantile offsets around the stored median; `m` a three-component Gaussian mixture
density network; `xshuf` and `mshuf` the same heads fitted on training features permuted across rows
(the placebo). A fit at the start of calendar month T uses every row, at all leads and all eight
Primaries, whose reading arrived (valid time plus 6 hours) before T, and is applied to the rows
whose forecast origin is in month T. Quantiles are the stored median plus the head's offset minus
the head's own median offset, so the median is unchanged, then sorted.

    PYTHONPATH=ml-experiments/00368-d8-conditional-band-head-on-gam-stack-residuals \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00368-*/e00368_fit.py --arm x
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from datetime import UTC, datetime
from typing import Final

import e00368_features as feat
import numpy as np
import polars as pl
import torch
import xgboost as xgb

LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
MEDIAN_INDEX: Final[int] = 5
MONTH_STARTS: Final[list[datetime]] = [
    datetime(2025 + (7 + i) // 12, (7 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(11)
]
"""Fit times: 2025-08-01 to 2026-06-01."""
MAX_TRAIN_ROWS: Final[int] = 300_000
SEED: Final[int] = 368


def design(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the feature matrix, NaN where a trailing mean is unknown."""
    return frame.select(list(feat.FEATURES)).to_numpy().astype(np.float32)


def fit_predict_xgboost(*, train_x, train_y, test_x, threads: int) -> np.ndarray:  # noqa: ANN001
    """Return the 11 offsets of an XGBoost multi-quantile model on the test rows."""
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=LEVELS,
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        min_child_weight=200,
        subsample=0.8,
        colsample_bytree=1.0,  # the arms differ in rows, never in columns (colsample invalidates ablations)
        tree_method="hist",
        n_jobs=threads,
        random_state=SEED,
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x), dtype=np.float64)


class Mixture(torch.nn.Module):
    """A three-component Gaussian mixture density network."""

    def __init__(self, *, n_in: int) -> None:
        """Build the two-layer body and the nine-output head."""
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(n_in, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU()
        )
        self.head = torch.nn.Linear(64, 9)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-mutable-override]
        """Return log-weights, means, and standard deviations."""
        out = self.head(self.body(x))
        return (
            torch.log_softmax(out[:, :3], dim=1),
            out[:, 3:6],
            torch.nn.functional.softplus(out[:, 6:]) + 0.05,
        )


def fit_predict_mdn(*, train_x, train_y, test_x, threads: int) -> np.ndarray:  # noqa: ANN001
    """Return the 11 offsets of a mixture density network on the test rows."""
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    mean = np.nanmean(train_x, axis=0)
    scale = np.nanstd(train_x, axis=0) + 1e-6

    def prepare(x: np.ndarray) -> torch.Tensor:
        z = (x - mean) / scale
        return torch.from_numpy(
            np.concatenate(
                [np.nan_to_num(z), np.isnan(x[:, 8:9]).astype(np.float32)], axis=1
            ).astype(np.float32)
        )

    x_train, y_train = prepare(train_x), torch.from_numpy(train_y.astype(np.float32))
    net = Mixture(n_in=x_train.shape[1])
    optimiser = torch.optim.Adam(net.parameters(), lr=2e-3)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(12):
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 4096):
            batch = order[start : start + 4096]
            log_w, mu, sigma = net(x_train[batch])
            z = (y_train[batch, None] - mu) / sigma
            log_p = log_w - 0.5 * z**2 - torch.log(sigma) - 0.9189385
            loss = -torch.logsumexp(log_p, dim=1).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    out = np.empty((len(test_x), len(LEVELS)))
    x_test = prepare(test_x)
    with torch.no_grad():
        for start in range(0, len(x_test), 20_000):
            log_w, mu, sigma = net(x_test[start : start + 20_000])
            weights = log_w.exp()
            low = (mu - 8 * sigma).min(dim=1).values
            high = (mu + 8 * sigma).max(dim=1).values
            for j, level in enumerate(LEVELS):
                lo, hi = low.clone(), high.clone()
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    cdf = (weights * torch.distributions.Normal(mu, sigma).cdf(mid[:, None])).sum(
                        dim=1
                    )
                    below = cdf < level
                    lo = torch.where(below, mid, lo)
                    hi = torch.where(below, hi, mid)
                out[start : start + 20_000, j] = (0.5 * (lo + hi)).numpy()
    return out


def run_month(*, frame: pl.DataFrame, fit_at: datetime, arm: str, threads: int) -> pl.DataFrame:
    """Fit on rows read before `fit_at`, and return the new quantiles of the rows originating in its month."""
    fit_ns = int(fit_at.timestamp() * 1e9)
    end = datetime(fit_at.year + fit_at.month // 12, fit_at.month % 12 + 1, 1, tzinfo=UTC)
    train = frame.filter(
        (pl.col("valid_ns") + feat.LATENCY_NS <= fit_ns) & pl.col("residual").is_finite()
    )
    test = frame.filter(
        (pl.col("origin_ns") >= fit_ns) & (pl.col("origin_ns") < int(end.timestamp() * 1e9))
    )
    if train.height > MAX_TRAIN_ROWS:
        train = train.sample(n=MAX_TRAIN_ROWS, seed=SEED)
    train_x, train_y, test_x = design(frame=train), train["residual"].to_numpy(), design(frame=test)
    if arm.endswith("shuf"):
        train_x = train_x[np.random.default_rng(SEED).permutation(len(train_x))]
    fit = fit_predict_xgboost if arm.startswith("x") else fit_predict_mdn
    offsets = fit(train_x=train_x, train_y=train_y, test_x=test_x, threads=threads)
    offsets = offsets - offsets[:, MEDIAN_INDEX : MEDIAN_INDEX + 1]
    values = np.sort(test["median"].to_numpy()[:, None] + offsets, axis=1)
    return test.select(*feat.lib.KEY, "site").with_columns(
        [
            pl.Series(name, values[:, i].astype(np.float32))
            for i, name in enumerate(feat.lib.QUANTILES)
        ]
    )


def main() -> None:
    """Fit one arm for every month and write `work/heads/<arm>.parquet`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["x", "m", "xshuf", "mshuf"], required=True)
    parser.add_argument("--threads", type=int, default=3)
    arguments = parser.parse_args()
    frame = pl.read_parquet(feat.WORK / "features.parquet")
    out_dir = feat.WORK / "heads"
    out_dir.mkdir(exist_ok=True)
    parts = []
    for fit_at in MONTH_STARTS:
        part = run_month(frame=frame, fit_at=fit_at, arm=arguments.arm, threads=arguments.threads)
        parts.append(part)
        print(arguments.arm, fit_at.date(), part.height, flush=True)
    tmp = out_dir / f"{arguments.arm}.parquet.tmp"
    pl.concat(parts).write_parquet(tmp)
    tmp.rename(out_dir / f"{arguments.arm}.parquet")


if __name__ == "__main__":
    main()

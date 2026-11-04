# ruff: noqa: E501
"""Copy of experiment 531's refit and head functions (part A of experiment 531): experiment 404's lag-free head scored with the weather of the run published at the forecast origin.

Experiment 404 stored the lag-free head only as output tables, never as weights. This script therefore refits
the table's "further" module (the one that forecasts the scored month) with 404's own code and seed, checks that
the refit reproduces the stored table, and then feeds it either 404's freshest-run weather (reproduction) or the
weather of 511's rows (the run published at the forecast origin).
"""

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00454_build as b454  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

SEEDS = (0, 1, 2)


def fit_further(
    *, month: str, seed: int, device: str
) -> tuple[models.LaglessModule, lagless.Scaler, lagless.Pool]:
    """Refit the further module of 404's table (copy of the last fit in `crossfit_table`); return it, its scaler, and the pool it forecasts."""
    refit = prep.refit_of(month=month)
    end = int(common.grid_index(times=refit))
    start = max(lagless.data.FIRST_VALID_INDEX, end - lagless.WINDOW_STEPS)
    train = lagless.build_pool(lo=start, hi=end)
    after = lagless.build_pool(lo=end, hi=end + lagless.FURTHER_STEPS)
    age_days = (end - train.grid) / common.STEPS_PER_DAY
    weight = (0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)).astype(np.float32)
    fit_mask = np.isfinite(train.y)
    scaler = lagless.Scaler(
        x=train.x[fit_mask], site=train.site[fit_mask], y=train.y[fit_mask], n_primaries=16
    )
    z = scaler.target(y=train.y[fit_mask], site=train.site[fit_mask])
    module = lagless.fit_module(
        x=torch.tensor(scaler.inputs(x=train.x[fit_mask]), device=device), z=torch.tensor(z, device=device),
        site=torch.tensor(train.site[fit_mask], device=device), weight=torch.tensor(weight[fit_mask], device=device), seed=seed,
    )  # fmt: skip
    module.eval()
    return module, scaler, after


@torch.no_grad()
def head(
    *,
    module: models.LaglessModule,
    scaler: lagless.Scaler,
    x_raw: np.ndarray,
    site: np.ndarray,
    device: str,
) -> b454.Head:
    """Return the mixture (log weights, means, standard deviations in capacity fractions) for raw inputs."""
    out = []
    for a in range(0, len(x_raw), 65_536):
        xs = torch.tensor(scaler.inputs(x=x_raw[a : a + 65_536]), device=device)
        out.append(module(xs, torch.tensor(site[a : a + 65_536], device=device)).double().cpu())
    log_w, mean, std = models.mixture_parts(out=torch.cat(out), components=models.COMPONENTS)
    y_mean = torch.tensor(scaler.y_mean)[torch.from_numpy(site)][:, None]
    y_std = torch.tensor(scaler.y_std)[torch.from_numpy(site)][:, None]
    return log_w, mean * y_std + y_mean, std * y_std

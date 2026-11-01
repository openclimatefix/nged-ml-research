# ruff: noqa: E501
r"""Fine-tune experiment 425's arm A with the out-of-fold GAM forecast as an input or as a base.

Both arms start from experiment 425's stored pretrained arm A weights (pretrained without any GAM
value, on ERA5-and-CAMS rows), so the GAM enters in fine-tuning only and arm A is the matched
reference. Each arm adds two columns to the fine-tuning and scoring rows: the GAM forecast in
standardised units, `z_gam = (gam / 100 - target_mean) / target_std` (the GAM is in percent of
effective capacity), and `miss`, which is 1 where no GAM value exists (D8 before 2025-07, D15,
D18) and then `z_gam` is 0.

- `Gin` (A-gam-in): the network reads both columns. The two new first-layer weight columns start at
  zero, so fine-tuning starts from the pretrained arm A.
- `Gbase` (A-gam-base): the network does not read the columns. Its mixture means are
  `inner_mean + (1 - miss) * (z_gam - m0)`, where `m0` is the mean of the frozen pretrained
  network's whole mixture. At the start of fine-tuning the mixture's mean is the GAM forecast
  and its component offsets, weights, and spreads are the pretrained network's. Rows with
  `miss = 1` are the pretrained network unchanged.

```bash
export PYTHONPATH=<as e00425_fit.py, plus ml-experiments/00457-*>
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00457-*/e00457_fit.py fit --device cuda
```
"""

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep  # ty: ignore[unresolved-import]
import e00457_stage0 as stage0  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

WORK_DIR: Final[Path] = stage0.WORK_DIR
OUT_DIR: Final[Path] = WORK_DIR / "work"
ARMS: Final[tuple[str, ...]] = ("Gin", "Gbase")
FIVE_FOLDS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03", "2025-12")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
US_PER_DAY: Final[int] = 86_400 * 1_000_000
INPUT_CLIP_SD: Final[float] = 5.0
"""The network inputs are clipped at 5 standard deviations; so is the GAM value."""
GAM_PER_UNIT: Final[float] = 100.0
"""The GAM is in percent of effective capacity; the network's targets are fractions."""


def gam_table() -> pl.DataFrame:
    """Return the stage-0 GAM forecasts of the 14 Primaries, keyed for the fold rows and the keys."""
    series = base.primaries().select("primary", "site", "time_series_id")
    frames = []
    for site in (*stage0.SOURCES["control"][1], *stage0.SOURCES["mva"][1]):
        frame = pl.read_parquet(WORK_DIR / f"gam_{site}.parquet")
        frames.append(frame.with_columns(pl.lit(site).alias("site")))
    first_us = base.FIRST_RUN.timestamp() * 1_000_000
    joined = (
        pl.concat(frames)
        .join(series, on=["site", "time_series_id"], how="inner", validate="m:1")
        .with_columns(
            ((pl.col("power_fcst_init_time").dt.epoch("us") - int(first_us)) // US_PER_DAY)
            .cast(pl.Int64)
            .alias("origin_day"),
            pl.col("valid_time").dt.epoch("us").alias("valid_us"),
            (pl.col("gam_points") / GAM_PER_UNIT).alias("gam_fraction"),
        )
        .select(
            "primary", "time_series_id", "power_fcst_init_time", "valid_time", "origin_day",
            "valid_us", "gam_fraction",
        )
    )  # fmt: skip
    assert joined["primary"].n_unique() == len(frames), "a site was dropped in the Primary join"
    return joined


def gam_columns(*, gam_fraction: np.ndarray, primary: np.ndarray, fold: prep.EnsFold) -> np.ndarray:
    """Return `[z_gam, miss]` (rows, 2): the standardised GAM value and its missing flag."""
    miss = ~np.isfinite(gam_fraction)
    z = (gam_fraction - fold.target_mean[primary]) / fold.target_std[primary]
    z = np.where(miss, 0.0, np.clip(z, -INPUT_CLIP_SD, INPUT_CLIP_SD))
    return np.stack([z, miss.astype(np.float64)], axis=1).astype(np.float32)


def fold_gam(*, fold: prep.EnsFold, table: pl.DataFrame) -> np.ndarray:
    """Return the GAM columns of the fold's fine-tuning rows (key: Primary, origin day, valid time)."""
    rows = pl.DataFrame(
        {"primary": fold.primary, "origin_day": fold.origin_day, "valid_us": fold.valid}
    ).with_row_index("row")
    joined = rows.join(
        table.select("primary", "origin_day", "valid_us", "gam_fraction"),
        on=["primary", "origin_day", "valid_us"],
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    assert joined.height == len(fold.primary)
    return gam_columns(
        gam_fraction=joined["gam_fraction"].fill_null(float("nan")).to_numpy(),
        primary=fold.primary,
        fold=fold,
    )


class GamNet(nn.Module):
    """The pretrained network plus the GAM, as an extra input (`in`) or a base (`base`)."""

    def __init__(self, *, inner: nn.Module, mode: str, frozen: nn.Module | None, k: int) -> None:
        """Wrap `inner`; `frozen` is the pretrained copy that `base` subtracts."""
        super().__init__()
        self.inner, self.mode, self.k = inner, mode, k
        self.frozen = frozen
        if frozen is not None:
            for p in frozen.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the raw mixture outputs for rows whose last two columns are `z_gam` and `miss`."""
        if self.mode == "in":
            return self.inner(x, primary)
        out = self.inner(x[:, :-2], primary)
        assert self.frozen is not None
        with torch.no_grad():
            self.frozen.eval()
            f0 = self.frozen(x[:, :-2], primary)
        z_gam, present = x[:, -2:-1], 1.0 - x[:, -1:]
        k = self.k
        pretrained_mean = (torch.softmax(f0[..., :k], dim=-1) * f0[..., k : 2 * k]).sum(
            -1, keepdim=True
        )
        shift = present * (z_gam - pretrained_mean)
        return torch.cat([out[..., :k], out[..., k : 2 * k] + shift, out[..., 2 * k :]], dim=-1)


def build(*, arm: str, fold: prep.EnsFold, month: str, seed: int, device: str) -> GamNet:
    """Return the wrapped network, loaded from experiment 425's pretrained arm A weights."""
    n = fold.x.shape[1]
    stored = torch.load(
        fit425.pretrained_path(arm=fit425.ARMS["A"], cut=True, month=month, seed=seed),
        map_location="cpu",
    )
    plain = fit425.new_model(arm=fit425.ARMS["A"], fold=fold)
    plain.load_state_dict(stored)
    if arm == "Gbase":
        return GamNet(inner=plain, mode="base", frozen=copy.deepcopy(plain), k=3).to(device)
    wide = fit397.make_model(
        arm="plain_pooled",
        setting=fit425.SETTING,
        layout=fold.layout,
        n_primaries=len(fold.target_mean),
        n_inputs=n + 2,
    )
    state = {key: value.clone() for key, value in stored.items()}
    old = stored["net.0.weight"]
    new = torch.zeros(old.shape[0], old.shape[1] + 2)
    new[:, :n], new[:, n + 2 :] = old[:, :n], old[:, n:]
    state["net.0.weight"] = new
    wide.load_state_dict(state)
    return GamNet(inner=wide, mode="in", frozen=None, k=3).to(device)


def forecast_month(
    *, model: GamNet, fold: prep.EnsFold, month: str, table: pl.DataFrame, device: str
) -> pl.DataFrame:
    """Return the test month's quantiles (fraction of capacity) for one fitted network."""
    fitted = fit397.Fitted(
        model=model, setting=fit425.SETTING, arm="plain_pooled", seed=0, epochs=0, val_nll=0.0
    )
    lookup = table.select("time_series_id", "power_fcst_init_time", "valid_time", "gam_fraction")
    pieces = []
    for origin in base.test_origins(month=month):
        for row, keys, x in base.run_inputs(
            origin=origin, scheme="summary", standardiser=fold.standardiser
        ):
            p = int(row["primary"])
            joined = keys.join(
                lookup, on=["time_series_id", "power_fcst_init_time", "valid_time"],
                how="left", validate="1:1", maintain_order="left",
            )  # fmt: skip
            gam = gam_columns(
                gam_fraction=joined["gam_fraction"].fill_null(float("nan")).to_numpy(),
                primary=np.full(len(x), p),
                fold=fold,
            )
            q, _, _ = fit397.predict_z_quantiles(
                fitted=fitted,
                x=np.concatenate([x, gam], axis=1),
                primary=p,
                members=1,
                device=device,
            )
            q = np.sort(q * fold.target_std[p] + fold.target_mean[p], axis=1).astype(np.float32)
            pieces.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                )
            )
    return pl.concat(pieces)


def seed_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's seed-level raw quantile forecast is stored."""
    return OUT_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def run(*, months: list[str], arms: list[str], seeds: list[int], device: str) -> None:
    """Fine-tune and forecast each arm, fold, and seed that has no stored forecast."""
    table = gam_table()
    for month in months:
        todo = [
            (a, s)
            for a in arms
            for s in seeds
            if not seed_path(arm=a, month=month, seed=s).exists()
        ]
        if not todo:
            continue
        fold = prep.build_ens_fold(refit=base.refit_of(month=month), cut=True)
        gam = fold_gam(fold=fold, table=table)
        print(
            f"fold {month}: {len(fold.x):,} rows; GAM present on {int((gam[:, 1] == 0).sum()):,}",
            flush=True,
        )
        x = np.concatenate([fold.x, gam], axis=1)
        t = fit425.tensors(
            x=x,
            y=fold.y,
            primary=fold.primary,
            weight=fold.weight,
            is_val=fold.is_val,
            fold=fold,
            device=device,
        )
        setting = replace(fit425.SETTING, lr=fit425.FINE_TUNE_LR)
        for arm, seed in todo:
            started = time.monotonic()
            torch.manual_seed(seed)
            model = build(arm=arm, fold=fold, month=month, seed=seed, device=device)
            model, epochs, best = fit397.train_network(
                model=model,
                setting=setting,
                x=t["x"],
                z=t["z"],
                w=t["w"],
                primary=t["p"],
                is_val=t["v"],
                seed=seed,
            )  # ty: ignore[invalid-assignment]
            q = forecast_month(model=model, fold=fold, month=month, table=table, device=device)  # ty: ignore[invalid-argument-type]
            common.atomic_write(frame=q, target=seed_path(arm=arm, month=month, seed=seed))
            log = {"epochs": epochs, "val_nll": best, "seconds": round(time.monotonic() - started)}
            (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "logs" / f"{arm}_{month}_s{seed}.json").write_text(json.dumps(log))
            print(f"{arm} {month} s{seed}: {log}", flush=True)
        del t
        torch.cuda.empty_cache()


def check(*, month: str) -> None:
    """Print how the GAM value relates to the target on a fold's fine-tuning rows."""
    table = gam_table()
    fold = prep.build_ens_fold(refit=base.refit_of(month=month), cut=True)
    gam = fold_gam(fold=fold, table=table)
    present = gam[:, 1] == 0
    z = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
    print(f"rows {len(fold.y):,}; GAM present {int(present.sum()):,}")
    print("corr(z_gam, z)", float(np.corrcoef(gam[present, 0], z[present])[0, 1]))
    resid = (gam[present, 0] - z[present]) * fold.target_std[fold.primary][present]
    print(
        "GAM minus target, fraction of capacity: mean",
        float(resid.mean()),
        "mean abs",
        float(np.abs(resid).mean()),
    )


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "check"))
    parser.add_argument("--months", nargs="+", default=list(FIVE_FOLDS))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "check":
        check(month=args.months[0])
    else:
        run(months=args.months, arms=args.arms, seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()

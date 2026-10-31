# ruff: noqa: E501, B023
"""Forecast a test month with saved networks of experiment 428, and the row-class flags.

`forecast_rows` is the single function every later step calls: it takes rows and a world, optionally
a function that modifies the raw recent power (the encroachment sweep and later lag-trust
evaluations), recomputes every input from the modified power, and returns the 11 quantiles, the
mixture parameters, and (for the encoder) the attention mass per window and the "no day" token. The
networks run in evaluation mode with dropout off.

Output of the `predict` command: for each arm, month, and seed, one parquet per world under
`predictions/<arm>/<month>_s<seed>_w<world>.parquet` with `primary`, `origin_us`, `valid_us`, `lead`,
`actual` (fraction of capacity, NaN where unusable), q01 to q99 in fractions of capacity, the row
class flags (`win_inj`, `win_real`, `tgt_real`, `tgt_post`), and for encoders `mass_d1..d3`,
`mass_null`, and `masked_q01..masked_q99` (the same network with all recent power hidden).

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00428-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00428-*/e00428_predict.py --arms R E3 --months 2026-01 --seeds 0
```
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_models as models  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

PRED_DIR: Final[Path] = data.WORK_DIR / "predictions"
LEVEL_NAMES: Final[tuple[str, ...]] = tuple(f"q{round(100 * q):02d}" for q in base.LEVELS)
CHUNK: Final[int] = 8_192
EVAL_WORLDS: Final[tuple[int, ...]] = (0, 4, 5, 6)
"""World 0 is the real, clean series; worlds 4 to 6 are the held-out injected worlds."""
POST_EVENT_STEPS: Final[int] = 96


@dataclass
class Loaded:
    """A saved network with what it needs to forecast."""

    arm: fit.Arm
    model: torch.nn.Module
    standardiser: prep.Standardiser
    target_mean: np.ndarray
    target_std: np.ndarray
    power_mean: float
    power_std: float
    seed: int


def load(*, arm_name: str, month: str, seed: int, device: str) -> Loaded:
    """Load a saved network."""
    folder = fit.MODELS_DIR / arm_name / month / f"s{seed}"
    meta = json.loads((folder / "meta.json").read_text())
    z = np.load(folder / "standardiser.npz")
    arm = fit.ARMS[arm_name]
    model = fit.make_model(
        arm=arm, n_inputs=meta["n_inputs"], n_primaries=len(z["target_mean"])
    ).to(device)
    model.load_state_dict(torch.load(folder / "model.pt", map_location=device))
    model.eval()
    return Loaded(
        arm=arm,
        model=model,
        standardiser=prep.Standardiser(mean=z["mean"], std=z["std"], mask=z["mask"]),
        target_mean=z["target_mean"],
        target_std=z["target_std"],
        power_mean=meta["power_mean"],
        power_std=meta["power_std"],
        seed=seed,
    )


def mixture_of(
    *, out: torch.Tensor, components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return log weights, means, and standard deviations of the raw outputs, in double precision."""
    return base.mixture_parts(out=out.double(), components=components)


def quantiles_of(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, loaded: Loaded, primary: torch.Tensor
) -> torch.Tensor:  # fmt: skip
    """Return the sorted 11 quantiles as fractions of capacity, shape (rows, 11)."""
    q = base.mixture_quantiles(log_w=log_w, mean=mean, std=std)
    mu = torch.from_numpy(loaded.target_mean).to(q.device)[primary][:, None]
    sd = torch.from_numpy(loaded.target_std).to(q.device)[primary][:, None]
    return torch.sort(q * sd + mu, dim=1).values


def forecast_rows(
    *,
    loaded: Loaded,
    inputs: fit.Inputs,
    rows: torch.Tensor,
    world: torch.Tensor,
    modify: Callable[[dict], None] | None = None,
    masked: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the forecast of `rows` read from `world`, after `modify` edits the raw power.

    Args:
        loaded: The network.
        inputs: The rows' GPU inputs and the power stack.
        rows: Row numbers into `inputs`.
        world: World id per row.
        modify: Edits `raw["lags"]` or `raw["raw"]` in place (recent power as fractions of
            capacity) before every input is recomputed from it.
        masked: Hide every recent-power input (the forecast of the network with no days shown).

    Returns:
        `q` (rows, 11), `log_w`, `mean`, `std` (rows, components, z space), and `mass`
        (rows, days + 1) for an encoder, else absent.
    """
    k = fit.SETTING.components
    with torch.no_grad():
        raw = inputs.raw(arm=loaded.arm, rows=rows, world=world)
        if modify is not None:
            modify(raw)
        x, primary, windows = inputs.assemble(arm=loaded.arm, raw=raw, masked=masked)
        result: dict[str, torch.Tensor] = {}
        if isinstance(loaded.model, models.DayEncoderMdn):
            out, mass = loaded.model(x, primary, windows, details=True)
            result["mass"] = mass
        else:
            out = fit.forward(model=loaded.model, x=x, primary=primary, windows=windows)
        log_w, mean, std = mixture_of(out=out, components=k)
        result.update(
            q=quantiles_of(log_w=log_w, mean=mean, std=std, loaded=loaded, primary=primary),
            log_w=log_w, mean=mean, std=std,
        )  # fmt: skip
    return result


def post_event_mask(*, padded: np.ndarray) -> np.ndarray:
    """Return the padded real-event mask extended by `POST_EVENT_STEPS` after each event's end."""
    mask = padded.copy()
    edges = np.diff(np.concatenate([[0], padded.astype(np.int8), [0]]))
    for end in np.nonzero(edges == -1)[0]:
        mask[end : end + POST_EVENT_STEPS] = True
    return mask[: padded.size]


class Flags:
    """Row-class flags from the injected event codes and the padded real-event mask."""

    def __init__(
        self, *, code: np.ndarray, real: np.ndarray, power: np.ndarray, device: str
    ) -> None:
        """Hold the masks on `device`: code (worlds, primaries, grid), real (primaries, grid)."""
        self.code = torch.from_numpy(code).to(device)
        self.power = torch.from_numpy(power).to(device)
        self.real = torch.from_numpy(real).to(device)
        self.post = torch.from_numpy(np.stack([post_event_mask(padded=r) for r in real])).to(device)

    def of(self, *, inputs: fit.Inputs, rows: torch.Tensor, world: torch.Tensor) -> dict:
        """Return boolean flags per row: `win_inj`, `win_real`, `tgt_real`, `tgt_post`."""
        primary = inputs.primary[rows]
        origin, target = inputs.origin[rows], inputs.target[rows]
        index = win.window_indices(origin=origin, target=target, days=3)
        size = self.real.shape[-1]
        inside = (index >= 0) & (index < size) & (index <= (origin - win.NEWEST_AGE)[:, None, None])
        safe = index.clamp(0, size - 1)
        p = primary[:, None, None]
        injected = self.code[world[:, None, None], p, safe] != 0
        real = self.real[p, safe]
        observed = torch.isfinite(self.power[world[:, None, None], p, safe])
        inside = inside & observed
        return {
            "win_inj": (injected & inside).flatten(1).any(1),
            "win_real": (real & inside).flatten(1).any(1),
            "tgt_inj": self.code[world, primary, target] != 0,
            "tgt_real": self.real[primary, target],
            "tgt_post": self.post[primary, target],
        }


def bmix_quantiles(
    *, lagged: Loaded, lagless: Loaded, inputs: fit.Inputs, rows: torch.Tensor, world: torch.Tensor
) -> torch.Tensor:
    """Return the quantiles of the fixed 50/50 mixture of two networks' distributions.

    The two networks standardise the target alike only if their target statistics match, so the
    mixture is formed on the capacity scale: each component's mean and spread are converted from
    z space with its own network's target statistics before the components are pooled.
    """
    pieces = []
    for loaded in (lagged, lagless):
        res = forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)
        primary = inputs.primary[rows]
        mu = torch.from_numpy(loaded.target_mean).to(rows.device)[primary][:, None]
        sd = torch.from_numpy(loaded.target_std).to(rows.device)[primary][:, None]
        pieces.append((res["log_w"] - float(np.log(2.0)), res["mean"] * sd + mu, res["std"] * sd))
    log_w = torch.cat([p[0] for p in pieces], dim=1)
    mean = torch.cat([p[1] for p in pieces], dim=1)
    std = torch.cat([p[2] for p in pieces], dim=1)
    q = base.mixture_quantiles(log_w=log_w, mean=mean, std=std)
    return torch.sort(q, dim=1).values


def predict_bmix(*, month: str, seed: int, device: str, power: torch.Tensor, flags: Flags,
                 y_clean: np.ndarray) -> list[str]:  # fmt: skip
    """Forecast B-mix (R-inj and L0 of the same fold and seed) for the scored worlds."""
    lagged = load(arm_name="R-inj", month=month, seed=seed, device=device)
    lagless = load(arm_name="L0", month=month, seed=seed, device=device)
    test = data.build_test(month=month, standardiser=lagged.standardiser)
    inputs = fit.Inputs(
        power=power, x=test.x, primary=test.primary, origin_idx=test.origin_idx,
        target_idx=test.target_idx, power_mean=lagged.power_mean, power_std=lagged.power_std,
        device=device,
    )  # fmt: skip
    lines = []
    for w in EVAL_WORLDS:
        target = PRED_DIR / "Bmix" / f"{month}_s{seed}_w{w}.parquet"
        if target.exists():
            continue
        columns: dict[str, list[np.ndarray]] = {n: [] for n in LEVEL_NAMES}
        flag_columns: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(test.x), CHUNK):
            rows = torch.arange(start, min(start + CHUNK, len(test.x)), device=device)
            world = torch.full((len(rows),), w, dtype=torch.long, device=device)
            with torch.no_grad():
                q = bmix_quantiles(lagged=lagged, lagless=lagless, inputs=inputs, rows=rows,
                                   world=world)  # fmt: skip
            for i, name in enumerate(LEVEL_NAMES):
                columns[name].append(q[:, i].float().cpu().numpy())
            for name, value in flags.of(inputs=inputs, rows=rows, world=world).items():
                flag_columns.setdefault(name, []).append(value.cpu().numpy())
        frame = pl.DataFrame(
            {
                "primary": test.primary, "origin_us": test.origin, "valid_us": test.valid,
                "lead": test.lead, "actual": y_clean[test.primary, test.target_idx],
                **{k: np.concatenate(v) for k, v in columns.items()},
                **{k: np.concatenate(v) for k, v in flag_columns.items()},
            }
        )  # fmt: skip
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        lines.append(f"Bmix {month} s{seed} w{w}: {frame.height:,} rows")
    return lines


def predict_month(*, arm_name: str, month: str, seed: int, device: str, power: torch.Tensor,
                  flags: Flags, y_clean: np.ndarray) -> list[str]:  # fmt: skip
    """Forecast a test month for the real and held-out injected worlds; return progress lines."""
    loaded = load(arm_name=arm_name, month=month, seed=seed, device=device)
    test = data.build_test(month=month, standardiser=loaded.standardiser)
    inputs = fit.Inputs(
        power=power, x=test.x, primary=test.primary, origin_idx=test.origin_idx,
        target_idx=test.target_idx, power_mean=loaded.power_mean, power_std=loaded.power_std,
        device=device,
    )  # fmt: skip
    actual = y_clean[test.primary, test.target_idx]
    lines = []
    for w in EVAL_WORLDS:
        target = PRED_DIR / arm_name / f"{month}_s{seed}_w{w}.parquet"
        if target.exists():
            continue
        pieces: dict[str, list[np.ndarray]] = {}

        def add(name: str, values: torch.Tensor) -> None:
            pieces.setdefault(name, []).append(values.detach().cpu().numpy())

        for start in range(0, len(test.x), CHUNK):
            rows = torch.arange(start, min(start + CHUNK, len(test.x)), device=device)
            world = torch.full((len(rows),), w, dtype=torch.long, device=device)
            res = forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)
            for i, name in enumerate(LEVEL_NAMES):
                add(name, res["q"][:, i].float())
            if "mass" in res:
                for d in range(loaded.arm.days):
                    add(f"mass_d{d + 1}", res["mass"][:, d])
                add("mass_null", res["mass"][:, -1])
                masked = forecast_rows(
                    loaded=loaded, inputs=inputs, rows=rows, world=world, masked=True
                )
                for i, name in enumerate(LEVEL_NAMES):
                    add(f"masked_{name}", masked["q"][:, i].float())
            for name, value in flags.of(inputs=inputs, rows=rows, world=world).items():
                add(name, value)
        frame = pl.DataFrame(
            {
                "primary": test.primary, "origin_us": test.origin, "valid_us": test.valid,
                "lead": test.lead, "actual": actual,
                **{k: np.concatenate(v) for k, v in pieces.items()},
            }
        )  # fmt: skip
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        lines.append(f"{arm_name} {month} s{seed} w{w}: {frame.height:,} rows")
    return lines


def main() -> None:
    """Forecast the requested arms, months, and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(fit.ARMS))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    power_np, code = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(args.device)
    flags = Flags(code=code, real=data.real_event_mask(), power=power_np, device=args.device)
    for month in args.months:
        for seed in args.seeds:
            if "Bmix" in args.arms:
                for line in predict_bmix(
                    month=month, seed=seed, device=args.device, power=power, flags=flags,
                    y_clean=power_np[0],
                ):  # fmt: skip
                    print(line, flush=True)
            for name in [a for a in args.arms if a != "Bmix"]:
                for line in predict_month(
                    arm_name=name, month=month, seed=seed, device=args.device, power=power,
                    flags=flags, y_clean=power_np[0],
                ):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()

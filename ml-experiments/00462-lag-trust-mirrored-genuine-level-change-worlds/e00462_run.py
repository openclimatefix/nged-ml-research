# ruff: noqa: E501, PLR0912, PT018
"""Fit and forecast S3-amb and S3-rw-amb: experiment 440's S3 network on experiment 446's ambiguous worlds.

**Arms.**

- `S3-amb`: experiment 440's S3 network (S1's three lags and the surprise at each, plus 24 window
  statistics and 30 neighbour columns, 54 extra columns) trained on the four training worlds of
  experiment 446's arm U1n-2: experiment 428's worlds 0 to 3 plus genuine level changes in 2% of
  half-hours, the target moving with the lags and no mirror at the neighbours. The recipe is
  experiment 446's: token dropout 0.2 on each lag and on each lag's surprise, and the origin of the
  lags drawn 0 to 7 hours before the usual one. A dropped lag also zeroes its 8 own and 10
  neighbour columns.
- `S3-rw-amb`: the same with random neighbours that keep each Primary's degree. The random matrix
  is used in the features only. The worlds are the same.
- References (forecast only): `U1n-2` (experiment 446's networks, S1 inputs), `S1` and `R-inj`
  (experiment 428's networks), forecast on this experiment's held-out worlds 7 to 9. Their forecasts
  on worlds 0, 4, 5, and 6 are experiment 446's and 428's stored files, linked by `link`.

**Held-out worlds 7 to 9** are the real clean series plus genuine level changes drawn with seeds 500
to 502 (`e00462_worlds`), each mirrored with probability 0.5 at 1 or 2 neighbours.

```bash
source /tmp/pp462.sh; export DATA_PATH_INTERNAL=<private data folder>
PYTHONPATH=$PP OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00462-*/e00462_run.py fit --arms S3-amb S3-rw-amb --months 2026-01 --seeds 0
uv run --no-sync python ml-experiments/00462-*/e00462_run.py predict --arms S3-amb U1n-2 S1 R-inj --months 2026-01 --seeds 0
uv run --no-sync python ml-experiments/00462-*/e00462_run.py link
uv run --no-sync python ml-experiments/00462-*/e00462_run.py selftest
```

`--smoke` fits one epoch and writes under `work/smoke/`.
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_models as models428  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import e00440_features as feat  # ty: ignore[unresolved-import]
import e00440_run as run440  # ty: ignore[unresolved-import]
import e00446_run as run446  # ty: ignore[unresolved-import]
import e00462_worlds as worlds
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00397_prep import refit_of  # ty: ignore[unresolved-import]

WORK_DIR: Final[Path] = worlds.WORK_DIR
MODELS_DIR: Path = WORK_DIR / "models"
PRED_DIR: Path = WORK_DIR / "predictions"
LOCK_DIR: Path = WORK_DIR / "locks"
ARM_NAMES: Final[tuple[str, ...]] = ("S3-amb", "S3-rw-amb")
RANDOM_NEIGHBOURS: Final[dict[str, str]] = {"S3-amb": "S3", "S3-rw-amb": "S3-rw"}
"""The key of experiment 440's neighbour matrices that each arm uses."""
REFERENCE_ARMS: Final[tuple[str, ...]] = ("U1n-2", "S1", "R-inj")
GEN_WORLDS: Final[tuple[int, ...]] = (7, 8, 9)
S1_COLUMNS: Final[int] = run440.S1_COLUMNS
N_EXTRA: Final[int] = feat.N_S3
LINK_WORLDS: Final[tuple[int, ...]] = pr.EVAL_WORLDS
"""Worlds of the stored forecasts that `link` links (0, 4, 5, 6)."""
STORED: Final[dict[str, Path]] = {
    "U1n-2": run446.WORK_DIR / "predictions" / "U1n-2",
    "S1": data.WORK_DIR / "predictions" / "S1",
    "R-inj": data.WORK_DIR / "predictions" / "R-inj",
    "Bmix": data.WORK_DIR / "predictions" / "Bmix",
    "S3": run440.WORK_DIR / "predictions" / "S3",
    "S3-rw": run440.WORK_DIR / "predictions" / "S3-rw",
}
"""Stored forecasts on worlds 0, 4, 5, and 6 that the scorer reads beside this experiment's own."""


def lag_of_column() -> torch.Tensor:
    """Return (54,) the lag (0, 1, or 2) that each of the 54 extra columns describes."""
    own = [lag for lag in range(3) for _ in range(feat.N_OWN_PER_LAG)]
    neighbour = [lag for lag in range(3) for _ in range(feat.N_NEIGHBOUR_PER_LAG)]
    return torch.tensor(own + neighbour)


class AmbInputs(run440.WindowInputs, run446.Inputs446):
    """Experiment 440's window and neighbour columns on experiment 446's ambiguous-world training.

    The training augmentation (token dropout, origin draw) and the shifted target come from
    `Inputs446`; the 54 extra columns come from `WindowInputs`. The extra columns of a dropped or
    unobserved lag are zeroed, so a hidden lag leaves no trace in its window columns.
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Build the inputs and name the neighbour matrix of each arm."""
        super().__init__(**kwargs)
        for arm, key in RANDOM_NEIGHBOURS.items():
            self._neighbours[arm] = self._neighbours[key]
        self._lag_of_column = lag_of_column().to(self.x.device)

    def assemble(self, *, arm: fit.Arm, raw: dict, masked: bool = False) -> tuple:
        """Return `(x, primary, None)`: S1's columns, then the 54 extra columns of the visible lags."""
        x_in, primary, _ = fit.Inputs.assemble(self, arm=arm, raw=raw, masked=masked)
        lags = torch.full_like(raw["lags"], float("nan")) if masked else raw["lags"]
        surprise = self.lag_surprise(raw=raw, lags=lags)
        if "spread_drop" in raw and not masked:
            surprise = torch.where(raw["spread_drop"], torch.zeros_like(surprise), surprise)
        extra = torch.zeros(len(primary), N_EXTRA, device=primary.device)
        if not masked:
            lag_grid = win.lag_indices(origin=raw["origin"], target=raw["target"])
            half = win.take(
                power=self.surprise[1][None], world=torch.zeros_like(primary), primary=primary,
                index=lag_grid,
            )  # fmt: skip
            step = torch.nan_to_num(raw["lags"] - raw["lags_unshifted"], nan=0.0)
            shift = torch.nan_to_num(step / half, nan=0.0)
            for w in torch.unique(raw["world"]).tolist():
                sel = raw["world"] == w
                extra[sel] = feat.row_features(
                    stack=self.stack(world=int(w)), primary=primary[sel], lag_grid=lag_grid[sel],
                    origin=raw["origin"][sel], shift=shift[sel],
                    neighbours=self._neighbours[arm.name], weights=self._weights,
                )  # fmt: skip
            present = torch.isfinite(raw["lags"]).float()
            extra = extra * present[:, self._lag_of_column]
        return torch.cat([x_in, surprise, extra], dim=1), primary, None


def install() -> None:
    """Register the arms, redirect the output folders, and widen the network of this experiment's arms."""
    run446.install()
    for name in ARM_NAMES:
        fit.ARMS[name] = fit.Arm(name, "lags", 0, True, surprise=True)
    if getattr(fit.make_model, "_installed_462", False):
        return
    previous = fit.make_model

    def make_model(*, arm: fit.Arm, n_inputs: int, n_primaries: int) -> torch.nn.Module:
        if arm.name not in ARM_NAMES:
            return previous(arm=arm, n_inputs=n_inputs, n_primaries=n_primaries)
        return models428.LaggedMdn(
            n_inputs=n_inputs + S1_COLUMNS + N_EXTRA, n_primaries=n_primaries,
            hidden=fit.SETTING.hidden, layers=fit.SETTING.layers,
            components=fit.SETTING.components, dropout=fit.SETTING.dropout,
        )  # fmt: skip

    make_model._installed_462 = True  # ty: ignore[unresolved-attribute]
    fit.make_model = make_model  # ty: ignore[invalid-assignment]


def use_models(*, arm_name: str) -> None:
    """Point experiment 428's loader at the folder that holds the arm's saved networks."""
    if arm_name in ARM_NAMES:
        fit.MODELS_DIR = MODELS_DIR
    elif arm_name == "U1n-2":
        fit.MODELS_DIR = run446.REAL_MODELS_DIR
    else:
        fit.MODELS_DIR = data.WORK_DIR / "models"


def fit_one(
    *, arm_name: str, month: str, seed: int, device: str, rows_cache: dict, stack_cache: dict,
) -> str:  # fmt: skip
    """Fit and save one arm, fold, and seed; return a progress line."""
    target = MODELS_DIR / arm_name / month / f"s{seed}"
    if (target / "meta.json").exists():
        return f"{arm_name} {month} s{seed}: exists"
    cell = run446.Cell(kind="fit", arm=arm_name, month=month, seed=seed)
    if not cell.held:
        return f"{arm_name} {month} s{seed}: locked by another process"
    if (target / "meta.json").exists():
        cell.release()
        return f"{arm_name} {month} s{seed}: exists"
    try:
        spec = run446.ARMS446["U1n-2"]
        arm = fit.ARMS[arm_name]
        started = time.monotonic()
        if "stack" not in stack_cache:
            stack_cache["stack"] = run446.train_stacks(spec=spec, device=device)
        power, target_power, clean_power = stack_cache["stack"]
        if month not in rows_cache:
            rows_cache.clear()
            stack, _ = data.load_stack(include_eval=False)
            rows_cache[month] = data.build_fold(refit=refit_of(month=month), clean=stack[0])
        rows: data.Rows = rows_cache[month]
        inputs = AmbInputs(
            power=power, x=rows.x, primary=rows.primary, origin_idx=rows.origin_idx,
            target_idx=rows.target_idx, power_mean=rows.power_mean, power_std=rows.power_std,
            device=device, surprise=fit.surprise_for(arm=arm, month=month, seed=seed, device=device),
            clean_power=clean_power, target_power=target_power,
        )  # fmt: skip
        inputs.target_stats = fit.target_stats(
            mean=rows.target_mean, std=rows.target_std, device=device
        )
        w = torch.from_numpy(rows.weight).to(device)
        is_val = torch.from_numpy(rows.is_val).to(device)
        torch.manual_seed(seed)
        model = fit.make_model(
            arm=arm, n_inputs=rows.x.shape[1], n_primaries=len(rows.target_mean)
        ).to(device)
        n_parameters = models428.count_parameters(model=model)
        epochs, best = run446.train(
            spec=spec, arm=arm, model=model, inputs=inputs, w=w, is_val=is_val, seed=seed,
            device=device,
        )  # fmt: skip
        target.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target / "model.pt")
        np.savez(
            target / "standardiser.npz",
            mean=rows.standardiser.mean, std=rows.standardiser.std, mask=rows.standardiser.mask,
            target_mean=rows.target_mean, target_std=rows.target_std,
        )  # fmt: skip
        import json

        (target / "meta.json").write_text(
            json.dumps(
                {
                    "arm": arm_name, "month": month, "seed": seed, "epochs": epochs,
                    "val_nll": best, "power_mean": rows.power_mean, "power_std": rows.power_std,
                    "n_rows": len(rows.x), "unusable_targets": rows.dropped,
                    "n_inputs": int(rows.x.shape[1]), "parameters": n_parameters,
                    "worlds": "experiment 446 U1n-2 (2% genuine, no mirror)",
                }
            )
        )  # fmt: skip
        return (
            f"{arm_name} {month} s{seed}: {epochs} epochs, val loss {best:.4f}, "
            f"{time.monotonic() - started:.0f} s"
        )
    finally:
        cell.release()


def eval_stack() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(power, code, shift, onset, level_code)` over worlds 0 to 9.

    Worlds 0 to 6 are experiment 428's; worlds 7 to 9 are the real clean series plus this
    experiment's held-out level changes (own events and mirrors). `code` is zero for worlds 7 to 9.
    `shift`, `onset`, and `level_code` hold the three held-out worlds only (3, primaries, grid).
    """
    power, code = data.load_stack(include_eval=True)
    nonneg = run446.nonnegative_primaries()
    loaded = [worlds.load_world(seed=s) for s in worlds.HELD_OUT_SEEDS]
    gen = np.stack(
        [
            run446.worlds.apply_shift(power=power[0], shift=s, nonnegative=nonneg)
            for s, _, _ in loaded
        ]
    )
    return (
        np.concatenate([power, gen]),
        np.concatenate([code, np.zeros(gen.shape, dtype=np.int8)]),
        np.stack([s for s, _, _ in loaded]),
        np.stack([o for _, o, _ in loaded]),
        np.stack([c for _, _, c in loaded]),
    )


def level_kind(*, test: data.TestRows, level_code: np.ndarray) -> np.ndarray:
    """Return the kind of level change at the row's lags: the code of the first lag that has one.

    0 none, 1 own event with no mirror, 2 own event with a mirror (the donor), 3 mirror received
    (the recipient). The lags are taken newest first (the newest reading, the daily lag, the weekly lag).
    """
    lags = win.lag_indices(
        origin=torch.from_numpy(test.origin_idx), target=torch.from_numpy(test.target_idx)
    ).numpy()
    size = level_code.shape[1]
    inside = (lags >= 0) & (lags < size)
    codes = np.where(inside, level_code[test.primary[:, None], np.clip(lags, 0, size - 1)], 0)
    kind = np.zeros(len(test.primary), dtype=np.int8)
    for j in (2, 1, 0):
        kind = np.where(codes[:, j] != 0, codes[:, j], kind)
    return kind


def predict_month(
    *, arm_name: str, month: str, seed: int, device: str, power: torch.Tensor,
    stack: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], flags: pr.Flags,
) -> list[str]:  # fmt: skip
    """Forecast a test month in every world not yet on disk; return progress lines.

    This experiment's arms are forecast in worlds 0, 4, 5, 6, and 7 to 9; the reference arms only
    in worlds 7 to 9 (their other worlds are stored files, linked by `link`).
    """
    power_np, _, shift, onset, level_code = stack
    mine = arm_name in ARM_NAMES
    todo_worlds = [*pr.EVAL_WORLDS, *GEN_WORLDS] if mine else list(GEN_WORLDS)

    def path_of(w: int) -> Path:
        return PRED_DIR / arm_name / f"{month}_s{seed}_w{w}.parquet"

    missing = [w for w in todo_worlds if not path_of(w).exists()]
    if not missing:
        return [f"{arm_name} {month} s{seed}: exists"]
    cell = run446.Cell(kind="predict", arm=arm_name, month=month, seed=seed)
    if not cell.held:
        return [f"{arm_name} {month} s{seed}: locked by another process"]
    try:
        use_models(arm_name=arm_name)
        loaded = pr.load(arm_name=arm_name, month=month, seed=seed, device=device)
        test = data.build_test(month=month, standardiser=loaded.standardiser)
        inputs_class = AmbInputs if mine else fit.Inputs
        inputs = inputs_class(
            power=power, x=test.x, primary=test.primary, origin_idx=test.origin_idx,
            target_idx=test.target_idx, power_mean=loaded.power_mean, power_std=loaded.power_std,
            device=device,
            surprise=fit.surprise_for(arm=loaded.arm, month=month, seed=seed, device=device),
        )  # fmt: skip
        inputs.target_stats = fit.target_stats(
            mean=loaded.target_mean, std=loaded.target_std, device=device
        )
        clean = power_np[0]
        lines = []
        for w in missing:
            gen = w in GEN_WORLDS
            pieces: dict[str, list[np.ndarray]] = {}
            for start in range(0, len(test.x), pr.CHUNK):
                rows = torch.arange(start, min(start + pr.CHUNK, len(test.x)), device=device)
                world = torch.full((len(rows),), w, dtype=torch.long, device=device)
                res = run446.forecast(loaded=loaded, inputs=inputs, rows=rows, world=world)
                for i, name in enumerate(pr.LEVEL_NAMES):
                    pieces.setdefault(name, []).append(res["q"][:, i].float().cpu().numpy())
                for name, value in flags.of(inputs=inputs, rows=rows, world=world).items():
                    pieces.setdefault(name, []).append(value.cpu().numpy())
            if mine and hasattr(inputs, "_stacks"):
                inputs._stacks.clear()
            extra: dict[str, np.ndarray] = {}
            if gen:
                g = w - GEN_WORLDS[0]
                extra = run446.generated_columns(test=test, shift=shift[g], onset=onset[g])
                extra["level_kind"] = level_kind(test=test, level_code=level_code[g])
            actual = (
                power_np[w][test.primary, test.target_idx] if gen
                else clean[test.primary, test.target_idx]
            )  # fmt: skip
            frame = pl.DataFrame(
                {
                    "primary": test.primary, "origin_us": test.origin, "valid_us": test.valid,
                    "lead": test.lead, "actual": actual,
                    "actual_clean": clean[test.primary, test.target_idx],
                    **{k: np.concatenate(v) for k, v in pieces.items()},
                    **extra,
                }
            )  # fmt: skip
            target = path_of(w)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            frame.write_parquet(temporary)
            temporary.rename(target)
            lines.append(f"{arm_name} {month} s{seed} w{w}: {frame.height:,} rows")
        return lines
    finally:
        cell.release()


def link() -> int:
    """Link the stored forecasts of the reference arms on worlds 0, 4, 5, and 6 into this folder."""
    n = 0
    for arm, source_dir in STORED.items():
        target = PRED_DIR / arm
        target.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.parquet"):
            world = int(source.stem.split("_w")[-1])
            if world not in LINK_WORLDS:
                continue
            seed = int(source.stem.split("_s")[1].split("_")[0])
            if seed > 1:
                continue
            path = target / source.name
            if not path.is_symlink() and not path.exists():
                path.symlink_to(source)
                n += 1
    return n


def rates(*, month: str = "2026-01", per_world: int = 400_000, seed: int = 0) -> pl.DataFrame:
    """Return the Bayes-optimal share of a step followed, from the training worlds' row mixture.

    For a sample of fold rows read from each training world 0 to 3, a *jump at a lag* is a lag whose
    power differs from the clean series (a load-shift donor or recipient, or a genuine level
    change). A row has a *mirror* if any neighbour of its Primary differs from the clean series at
    the newest lag's time. The target moved if the newest lag lies in a genuine level change and the target carries the same
    step (a load-shift jump never counts). The mirror is coarser than what the network sees: any neighbour
    differing from the clean series counts, whatever its sign or size, so the bound is not the
    best a network with the sign-agreement columns could reach.
    For k = 1, 2, 3 (the first k lags all in a jump), the share of such rows whose target moved is
    the share of a step that a forecaster with no other information should follow, with and
    without a mirror.
    """
    stack, _ = data.load_stack(include_eval=False)
    clean = stack[0]
    rows = data.build_fold(refit=refit_of(month=month), clean=clean)
    neighbours = worlds.neighbour_lists(sites=data.sites())
    rng = np.random.default_rng(seed)
    records = []
    for w, world_seed in enumerate(worlds.TRAIN_SEEDS):
        genuine, _, _ = worlds.load_world(seed=world_seed)
        injected = np.nan_to_num(stack[w] - clean) + genuine
        neighbour_jump = np.zeros(injected.shape, dtype=bool)
        for p, nbrs in enumerate(neighbours):
            for q in nbrs:
                neighbour_jump[p] |= injected[q] != 0
        pick = rng.choice(len(rows.primary), size=min(per_world, len(rows.primary)), replace=False)
        origin = torch.from_numpy(rows.origin_idx[pick])
        target = torch.from_numpy(rows.target_idx[pick])
        primary = rows.primary[pick]
        lags = win.lag_indices(origin=origin, target=target).numpy()
        size = injected.shape[1]
        ok = ((lags >= 0) & (lags < size)).all(axis=1)
        safe = np.clip(lags, 0, size - 1)
        jump = injected[primary[:, None], safe] != 0
        at_lag = genuine[primary, safe[:, 0]]
        at_target = genuine[primary, np.clip(rows.target_idx[pick], 0, size - 1)]
        moved = (at_lag != 0) & (at_target == at_lag)
        mirror = neighbour_jump[primary, safe[:, 0]]
        for k in (1, 2, 3):
            in_jump = ok & jump[:, :k].all(axis=1)
            for has_mirror in (True, False):
                sel = in_jump & (mirror == has_mirror)
                records.append(
                    {"world": w, "k": k, "mirror": has_mirror, "rows": int(sel.sum()),
                     "target_moved": int((sel & moved).sum())}
                )  # fmt: skip
    return (
        pl.DataFrame(records)
        .group_by("k", "mirror")
        .agg(pl.col("rows").sum(), pl.col("target_moved").sum())
        .with_columns(bayes_share=pl.col("target_moved") / pl.col("rows"))
        .sort("k", "mirror")
    )


def check(*, month: str = "2026-01", device: str = "cuda") -> None:
    """Check on real rows that a mirror reaches the neighbour columns and a dropped lag leaves none."""
    stack = eval_stack()
    power = torch.from_numpy(stack[0]).to(device)
    arm = fit.ARMS["S3-amb"]
    arm_rw = fit.ARMS["S3-rw-amb"]
    loaded_dir = MODELS_DIR / "S3-amb" / month / "s0"
    assert (loaded_dir / "meta.json").exists(), "fit (or smoke fit) S3-amb first"
    import json

    meta = json.loads((loaded_dir / "meta.json").read_text())
    z = np.load(loaded_dir / "standardiser.npz")
    test = data.build_test(
        month=month,
        standardiser=run446.prep.Standardiser(mean=z["mean"], std=z["std"], mask=z["mask"]),
    )
    inputs = AmbInputs(
        power=power, x=test.x, primary=test.primary, origin_idx=test.origin_idx,
        target_idx=test.target_idx, power_mean=meta["power_mean"], power_std=meta["power_std"],
        device=device, surprise=fit.surprise_for(arm=arm, month=month, seed=0, device=device),
    )  # fmt: skip
    kind = level_kind(test=test, level_code=stack[4][0])
    donors = np.nonzero(kind == worlds.OWN_MIRRORED)[0][:2_000]
    assert len(donors) > 100, "the test month must hold rows with a mirrored level change"
    rows = torch.from_numpy(donors).to(device)
    out = {}
    for name, a in (("S3-amb", arm), ("S3-rw-amb", arm_rw)):
        for w in (0, 7):
            world = torch.full((len(rows),), w, dtype=torch.long, device=device)
            raw = inputs.raw(arm=a, rows=rows, world=world)
            x, _, _ = inputs.assemble(arm=a, raw=raw)
            out[name, w] = x[:, -N_EXTRA:].cpu()
    own_change = (out["S3-amb", 7][:, :24] - out["S3-amb", 0][:, :24]).abs().max(dim=1).values
    neighbour_change = (out["S3-amb", 7][:, 24:] - out["S3-amb", 0][:, 24:]).abs().max(dim=1).values
    print(f"rows with a mirrored level change at a lag: {len(donors)}")
    print(
        f"share whose own window columns differ from the clean world: {(own_change > 1e-4).float().mean():.3f}"
    )
    print(
        f"share whose neighbour columns differ from the clean world: {(neighbour_change > 1e-4).float().mean():.3f}"
    )
    assert float((own_change > 1e-4).float().mean()) > 0.9, (
        "the donor's own columns must show the step"
    )
    assert float((neighbour_change > 1e-4).float().mean()) > 0.5, (
        "the mirror must reach S3's neighbour columns"
    )
    rw_change = (out["S3-rw-amb", 7][:, 24:] - out["S3-rw-amb", 0][:, 24:]).abs().max(dim=1).values
    print(
        f"S3-rw-amb: share whose (random) neighbour columns differ: {(rw_change > 1e-4).float().mean():.3f}"
    )
    world = torch.full((len(rows),), 7, dtype=torch.long, device=device)
    raw = inputs.raw(arm=arm, rows=rows, world=world)
    raw["lags"][:, 1] = float("nan")
    x, _, _ = inputs.assemble(arm=arm, raw=raw)
    extra = x[:, -N_EXTRA:].cpu()
    dropped = lag_of_column() == 1
    assert float(extra[:, dropped].abs().max()) == 0.0, "a dropped lag must leave no window columns"
    assert float(extra[:, ~dropped].abs().max()) > 0.0
    print("check passed")


def selftest() -> None:
    """Check the dropped-lag mask, the column layout, and that the arms read the same worlds as U1n-2."""
    mapping = lag_of_column()
    assert len(mapping) == N_EXTRA == 54
    assert mapping[:24].tolist() == [0] * 8 + [1] * 8 + [2] * 8
    assert mapping[24:].tolist() == [0] * 10 + [1] * 10 + [2] * 10
    present = torch.tensor([[1.0, 0.0, 1.0]])
    mask = present[:, mapping]
    assert float(mask[0, :8].min()) == 1.0 and float(mask[0, 8:16].max()) == 0.0
    assert float(mask[0, 24:34].min()) == 1.0 and float(mask[0, 34:44].max()) == 0.0
    assert float(mask[0, 44:].min()) == 1.0, "the weekly lag's columns stay"
    for seed in worlds.TRAIN_SEEDS:
        mine = worlds.load_world(seed=seed)[0]
        theirs = run446.worlds.load_shift(rate=0.02, trimmed=False, seed=seed)[0]
        assert np.array_equal(mine, theirs), f"training world {seed} must be experiment 446's"
    random = feat.random_neighbours(matrix=np.array([[1, 2], [0, -1], [0, -1]]))
    assert all(p not in random[p] for p in range(3))
    print("selftest passed")


def redirect_to_smoke() -> None:
    """Send fits, forecasts, and locks to `work/smoke/`, and cap every fit at one epoch."""
    global MODELS_DIR, PRED_DIR, LOCK_DIR
    root = WORK_DIR / "smoke"
    MODELS_DIR, PRED_DIR, LOCK_DIR = root / "models", root / "predictions", root / "locks"
    run446.LOCK_DIR = LOCK_DIR
    fit.MAX_EPOCHS = 1


def main() -> None:
    """Fit, forecast, link, or self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict", "link", "selftest", "check", "rates"))
    parser.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        redirect_to_smoke()
    install()
    pr.PRED_DIR = PRED_DIR
    run446.LOCK_DIR = LOCK_DIR
    if args.command == "selftest":
        selftest()
        return
    if args.command == "check":
        check(device=args.device)
        return
    if args.command == "rates":
        table = rates()
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        table.write_csv(WORK_DIR / "bayes_shares.csv")
        print(table)
        return
    if args.command == "link":
        print(f"linked {link()} forecast files")
        return
    if args.command == "fit":
        fit.MODELS_DIR = MODELS_DIR
        cache: dict = {}
        stack_cache: dict = {}
        for month in args.months:
            for seed in args.seeds:
                for name in args.arms:
                    print(
                        fit_one(
                            arm_name=name, month=month, seed=seed, device=args.device,
                            rows_cache=cache, stack_cache=stack_cache,
                        ),
                        flush=True,
                    )  # fmt: skip
        return
    stack = eval_stack()
    power = torch.from_numpy(stack[0]).to(args.device)
    flags = pr.Flags(code=stack[1], real=data.real_event_mask(), power=stack[0], device=args.device)
    for month in args.months:
        for seed in args.seeds:
            for name in args.arms:
                for line in predict_month(
                    arm_name=name, month=month, seed=seed, device=args.device, power=power,
                    stack=stack, flags=flags,
                ):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()

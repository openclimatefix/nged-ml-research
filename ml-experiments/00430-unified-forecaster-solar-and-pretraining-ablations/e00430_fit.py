# ruff: noqa: D105, E501
"""Fit the pooled solar networks of experiment 430 on the GPU and store their quantiles.

Rows, refits, and scoring follow experiment 345's solar-only pooled network (itself experiment
310's): farms A to F pooled over the six day bands, four quarterly refits from 2025-09, each
training on rows whose valid time is at least one day before the refit and predicting the runs of
its three months. Own history only. One job is one (arm, seed, refit); its prediction table holds
every farm and band and is written atomically and skipped when it exists.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00430-*/e00430_fit.py ARM [--seeds 0,1,2] [--refits 2025-09,...]
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00310_common as c310  # noqa: E402  # ty: ignore[unresolved-import]
import e00310_features as feats  # noqa: E402  # ty: ignore[unresolved-import]
import e00310_fit as f310  # noqa: E402  # ty: ignore[unresolved-import]
import e00430_common as common  # noqa: E402
import e00430_net as net  # noqa: E402
import e00430_plant as plant_module  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]
from studies.tail_calibration import level_column  # noqa: E402

DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFITS: Final[tuple[datetime, ...]] = f310.REFIT_STARTS
INFERENCE_CHUNK: Final[int] = 20_000
ALL_COLUMNS: Final[tuple[str, ...]] = (*common.GATE_COLUMNS, "clear_sky")
FLAGS: Final[tuple[str, ...]] = ("temp_missing",)
PRETRAIN_LEAD_RANGE_HOURS: Final[tuple[float, float]] = (12.0, 252.0)


@dataclass
class Piece:
    """Rows from one farm and band (or one farm's ERA5 and CAMS history), as float32 arrays."""

    columns: dict[str, np.ndarray]
    geo: np.ndarray
    members: np.ndarray
    y: np.ndarray
    ceiling: np.ndarray
    is_val: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __len__(self) -> int:
        return len(self.y)


# ---------------------------------------------------------------- data


@cache
def solar_base(farm: str, band: str) -> pl.DataFrame:
    """Return a farm-band's rows with every column the networks can read (plant-free)."""
    frame = feats.feature_frame(farm=feats.load_band_farm(farm, band), scale=1.0, ceiling=1e9)
    return frame.with_columns(
        temperature=pl.col("temperature_mean"),
        hour_sin=(2 * np.pi * pl.col("hour") / 24.0).sin(),
        hour_cos=(2 * np.pi * pl.col("hour") / 24.0).cos(),
        doy_sin=(2 * np.pi * pl.col("doy") / 366.0).sin(),
        doy_cos=(2 * np.pi * pl.col("doy") / 366.0).cos(),
        temp_missing=pl.lit(0.0),
    )


@cache
def member_array(farm: str, band: str) -> np.ndarray:
    """Return the (rows, 51, 2) shortwave and temperature members, rows in `solar_base` order."""
    arrays = [np.load(common.MEMBER_DIR / f"{farm}_d{day}.npy") for day in c310.BANDS[band]]
    return np.concatenate(arrays)


@cache
def farm_ceiling(farm: str, band: str, cutoff: datetime) -> float:
    """Return the farm-band's output ceiling (0.995 quantile) from rows before the cutoff."""
    return feats.scale_and_ceiling(farm=feats.load_band_farm(farm, band), cutoff=cutoff)[1]


def _columns(frame: pl.DataFrame, rows: np.ndarray | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for c in ALL_COLUMNS:
        if c in frame.columns and frame[c].null_count() < frame.height:
            values = frame[c].to_numpy().astype(np.float32)
            out[c] = values if rows is None else values[rows]
    return out


def ens_piece(*, farm: str, band: str, cutoff: datetime, need_members: bool) -> Piece:
    """Return a farm-band's training rows before the cutoff, with targets and a validation flag."""
    frame = solar_base(farm, band)
    rows = (frame["time"] < cutoff).to_numpy()
    members = (
        member_array(farm, band)[rows] if need_members else np.zeros((rows.sum(), 1, 2), np.float32)
    )
    return Piece(
        columns=_columns(frame, rows),
        geo=np.stack(
            [
                frame["zenith_deg"].to_numpy(),
                frame["sun_azimuth_deg"].to_numpy(),
                frame["doy"].to_numpy(),
            ],
            axis=1,
        ).astype(np.float32)[rows],
        members=members.astype(np.float32),
        y=frame["output"].to_numpy().astype(np.float32)[rows],
        ceiling=np.full(rows.sum(), farm_ceiling(farm, band, cutoff), dtype=np.float32),
        is_val=(frame["time"] >= cutoff - timedelta(days=common.VALIDATION_DAYS)).to_numpy()[rows],
    )


@cache
def era5_pieces(seed: int) -> tuple[Piece, ...]:
    """Return one piece per farm of ERA5 and CAMS history before the ensemble store starts.

    The weather columns the history holds are shortwave, beam, diffuse, temperature, the sun's
    position, the clear-sky index, and the clock; the lead is drawn uniformly from the ensemble's
    12 to 252 hours so the lead column is not constant, and the other ensemble columns are absent.
    A farm with no history before the ensemble store starts (farm E) contributes no piece.
    """
    rng = np.random.default_rng(10_000 + seed)
    pieces = []
    for farm in c310.FARMS:
        frame = (
            pl.read_parquet(c310.FARM_ROWS)
            .filter(pl.col("site") == farm)
            .filter(
                pl.col("time") < common.PRETRAIN_END,
                pl.col("elevation_deg") > 0,
                pl.col("ghi_w_m2").is_finite(),
                pl.col("output").is_finite(),
            )
            .sort("time")
            .with_columns(
                hour=(pl.col("time").dt.hour() + pl.col("time").dt.minute() / 60.0),
                doy=pl.col("time").dt.ordinal_day().cast(pl.Float64),
            )
            .with_columns(
                temperature=pl.col("temperature_c"),
                hour_sin=(2 * np.pi * pl.col("hour") / 24.0).sin(),
                hour_cos=(2 * np.pi * pl.col("hour") / 24.0).cos(),
                doy_sin=(2 * np.pi * pl.col("doy") / 366.0).sin(),
                doy_cos=(2 * np.pi * pl.col("doy") / 366.0).cos(),
                ghi_mean=pl.col("ghi_w_m2"),
                dni_mean=pl.col("dni_w_m2"),
                dhi_mean=pl.col("dhi_w_m2"),
                azimuth_sin=(pl.col("sun_azimuth_deg") * np.pi / 180).sin(),
                azimuth_cos=(pl.col("sun_azimuth_deg") * np.pi / 180).cos(),
                csi=(pl.col("ghi_w_m2") / pl.col("clear_sky").clip(lower_bound=1.0)).clip(0.0, 1.5),
                temp_missing=pl.lit(0.0),
            )
        )
        n = frame.height
        if n == 0:
            continue
        frame = frame.with_columns(
            lead_hours=pl.Series(rng.uniform(*PRETRAIN_LEAD_RANGE_HOURS, size=n))
        )
        last = frame["time"].max()
        assert isinstance(last, datetime)
        output = frame["output"].to_numpy().astype(np.float32)
        pieces.append(
            Piece(
                columns=_columns(frame, None),
                geo=np.stack(
                    [
                        frame["zenith_deg"].to_numpy(),
                        frame["sun_azimuth_deg"].to_numpy(),
                        frame["doy"].to_numpy(),
                    ],
                    axis=1,
                ).astype(np.float32),
                members=np.stack(
                    [frame["ghi_w_m2"].to_numpy(), frame["temperature_c"].to_numpy()], axis=1
                ).astype(np.float32)[:, None, :],
                y=output,
                ceiling=np.full(n, np.quantile(output, c310.CEILING_QUANTILE), dtype=np.float32),
                is_val=(frame["time"] >= last - timedelta(days=common.VALIDATION_DAYS)).to_numpy(),
            )
        )
    return tuple(pieces)


def standardiser(
    *, pieces: list[Piece], columns: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    """Return each column's mean and deviation over the pieces that hold the column."""
    stats: dict[str, tuple[float, float]] = {}
    for c in columns:
        if c in FLAGS:
            continue
        total = count = square = 0.0
        for piece in pieces:
            if c in piece.columns:
                v = piece.columns[c].astype(np.float64)
                total, square, count = total + v.sum(), square + (v**2).sum(), count + v.size
        if count == 0:
            stats[c] = (0.0, 1.0)
            continue
        mean = total / count
        stats[c] = (mean, float(np.sqrt(max(square / count - mean**2, 0.0))) + 1e-6)
    return stats


def matrix(*, piece: Piece, columns: tuple[str, ...], stats: dict) -> np.ndarray:
    """Return the piece's standardised inputs; a column the piece lacks is 0."""
    out = np.zeros((len(piece), len(columns)), dtype=np.float32)
    for j, c in enumerate(columns):
        if c in FLAGS:
            if c in piece.columns:
                out[:, j] = piece.columns[c]
        elif c in piece.columns:
            mean, std = stats[c]
            out[:, j] = (piece.columns[c] - mean) / std
    return out


@cache
def fixed_stats() -> dict[str, tuple[float, float]]:
    """Return the standardiser of the ensemble rows before 2025-08-31, shared by the pretraining arms."""
    path = common.DATA_DIR / "fixed_stats.json"
    if path.exists():
        return {k: (v[0], v[1]) for k, v in json.loads(path.read_text()).items()}
    pieces = [
        ens_piece(farm=f, band=b, cutoff=common.FIXED_STATS_CUTOFF, need_members=False)
        for f in c310.FARMS
        for b in c310.BAND_ORDER
    ]
    stats = standardiser(pieces=pieces, columns=ALL_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats))
    return stats


# ---------------------------------------------------------------- network


def build_plant(*, config: common.Config, pieces: list[Piece]) -> plant_module.Plant | None:
    """Return the arm's plant, with its scale fitted by least squares on the training rows."""
    if config.plant == "none":
        return None
    numerator = denominator = 0.0
    outputs = []
    for p in pieces:
        mean = p.members.mean(axis=1)
        unit = plant_module.unit_response(
            ghi=torch.tensor(mean[:, 0], dtype=torch.float64),
            temperature_c=torch.tensor(mean[:, 1], dtype=torch.float64),
            zenith_deg=torch.tensor(p.geo[:, 0], dtype=torch.float64),
            azimuth_deg=torch.tensor(p.geo[:, 1], dtype=torch.float64),
            doy=torch.tensor(p.geo[:, 2], dtype=torch.float64),
            tilt_deg=common.PLANT_TILT_DEG,
            panel_azimuth_deg=common.PLANT_AZIMUTH_DEG,
        ).numpy()
        numerator += float(np.sum(p.y * unit))
        denominator += float(np.sum(unit**2))
        outputs.append(p.y)
    clip = float(np.quantile(np.concatenate(outputs), c310.CEILING_QUANTILE))
    return plant_module.Plant(
        tilt_deg=common.PLANT_TILT_DEG,
        azimuth_deg=common.PLANT_AZIMUTH_DEG,
        scale=numerator / denominator,
        clip=clip,
        learn=config.plant == "e2e",
    )


def to_batch(
    *,
    piece: Piece,
    columns: tuple[str, ...],
    stats: dict,
    need_members: bool,
    rows: slice | None = None,
) -> net.Batch:
    """Return a piece (or a slice of it) as GPU tensors."""
    x = matrix(piece=piece, columns=columns, stats=stats)
    members = piece.members if need_members else piece.members[:, :1]
    sl = rows if rows is not None else slice(None)
    return net.Batch(
        x=torch.tensor(x[sl], device=DEVICE),
        geo=torch.tensor(piece.geo[sl], device=DEVICE),
        members=torch.tensor(members[sl], device=DEVICE),
        y=torch.tensor(piece.y[sl], device=DEVICE),
        ceiling=torch.tensor(piece.ceiling[sl], device=DEVICE),
    )


def train_network(
    *,
    pieces: list[Piece],
    config: common.Config,
    seed: int,
    stats: dict,
    learning_rate: float,
    init_state: dict | None = None,
) -> net.UnifiedNet:
    """Fit one network on the pieces: AdamW, 20 epochs, the best epoch on the last 61 days."""
    columns = config.columns
    need_members = config.member_route or config.plant != "none"
    torch.manual_seed(seed)
    plant = build_plant(config=config, pieces=pieces)
    model = net.UnifiedNet(n_static=len(columns), config=config, plant=plant).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)
    # Piece-by-piece standardisation (a piece lacking a column holds 0 there), joined on the GPU.
    parts = [
        to_batch(piece=p, columns=columns, stats=stats, need_members=need_members) for p in pieces
    ]
    all_batch = net.Batch(
        x=torch.cat([b.x for b in parts]),
        geo=torch.cat([b.geo for b in parts]),
        members=torch.cat([b.members for b in parts]),
        y=torch.cat([b.y for b in parts]),
        ceiling=torch.cat([b.ceiling for b in parts]),
    )
    del parts
    is_val = np.concatenate([p.is_val for p in pieces])
    fit_rows = torch.tensor(np.flatnonzero(~is_val), device=DEVICE)
    val_rows = torch.tensor(np.flatnonzero(is_val), device=DEVICE)
    plant_params = [p for n, p in model.named_parameters() if n.startswith("plant.")]
    other = [p for n, p in model.named_parameters() if not n.startswith("plant.")]
    optimiser = torch.optim.AdamW(
        [
            {"params": other, "weight_decay": common.WEIGHT_DECAY},
            {"params": plant_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    best, best_state = float("inf"), None

    def gather(index: torch.Tensor) -> net.Batch:
        return net.Batch(
            x=all_batch.x[index],
            geo=all_batch.geo[index],
            members=all_batch.members[index],
            y=all_batch.y[index],
            ceiling=all_batch.ceiling[index],
        )

    for _ in range(common.EPOCHS):
        model.train()
        order = fit_rows[torch.randperm(len(fit_rows), generator=generator, device=DEVICE)]
        for start in range(0, len(order), common.BATCH):
            loss = model.nll(gather(order[start : start + common.BATCH]))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            total = sum(
                float(model.nll(gather(val_rows[s : s + INFERENCE_CHUNK])))
                * len(val_rows[s : s + INFERENCE_CHUNK])
                for s in range(0, len(val_rows), INFERENCE_CHUNK)
            ) / len(val_rows)
        if total < best:
            best, best_state = total, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


def pretrained_state(*, config: common.Config, seed: int) -> dict:
    """Return the ERA5 and CAMS pretrained weights for the config (fitted once per seed, then stored)."""
    key = f"m{int(config.member_route)}_p{config.plant}_s{int(config.drop_spread)}_c{int(config.clear_sky_input)}_a{int(config.atoms)}"
    path = common.DATA_DIR / "pretrained" / f"{key}_s{seed}.pt"
    if path.exists():
        return torch.load(path, map_location=DEVICE)
    plain = common.Config(
        member_route=config.member_route,
        drop_spread=config.drop_spread,
        clear_sky_input=config.clear_sky_input,
        plant=config.plant,
        atoms=config.atoms,
        pretrain="none",
    )
    model = train_network(
        pieces=list(era5_pieces(seed)),
        config=plain,
        seed=seed,
        stats=fixed_stats(),
        learning_rate=common.LEARNING_RATE,
    )
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


# ---------------------------------------------------------------- jobs


def predict_rows(
    *,
    model: net.UnifiedNet,
    config: common.Config,
    stats: dict,
    start: datetime,
    end: datetime,
    cutoff: datetime,
    seed: int,
    arm: str,
) -> pl.DataFrame:
    """Return every farm and band's scored rows of the window with the model's 11 quantiles."""
    need_members = config.member_route or config.plant != "none"
    only_era5 = config.pretrain == "only"
    frames = []
    for farm in c310.FARMS:
        for band in c310.BAND_ORDER:
            frame = solar_base(farm, band)
            keep = (
                ((frame["init_time"] >= start) & (frame["init_time"] < end))
                & (frame["time"] < c310.SCORED_END)
            ).to_numpy()
            rows = frame.filter(pl.Series(keep))
            members = member_array(farm, band)[keep]
            columns = _columns(rows, None)
            if only_era5:
                columns = {k: v for k, v in columns.items() if k in common.AVAILABLE_IN_ERA5}
            piece = Piece(
                columns=columns,
                geo=np.stack(
                    [
                        rows["zenith_deg"].to_numpy(),
                        rows["sun_azimuth_deg"].to_numpy(),
                        rows["doy"].to_numpy(),
                    ],
                    axis=1,
                ).astype(np.float32),
                members=members if need_members else members[:, :1],
                y=np.zeros(rows.height, dtype=np.float32),
                ceiling=np.full(rows.height, farm_ceiling(farm, band, cutoff), dtype=np.float32),
            )
            out = np.empty((rows.height, len(c310.LEVELS)), dtype=np.float32)
            for s in range(0, rows.height, INFERENCE_CHUNK):
                batch = to_batch(
                    piece=piece,
                    columns=config.columns,
                    stats=stats,
                    need_members=need_members,
                    rows=slice(s, s + INFERENCE_CHUNK),
                )
                out[s : s + INFERENCE_CHUNK] = net.quantiles(model=model, batch=batch).cpu().numpy()
            frames.append(
                rows.select("time", "init_time", "lead_hours").with_columns(
                    time_series_id=pl.lit(farm),
                    arm=pl.lit(arm),
                    seed=pl.lit(seed),
                    setting=pl.lit("own_history"),
                    band=pl.lit(band),
                    actual=rows["output"].cast(pl.Float64),
                    **{
                        level_column(level=lv): pl.Series(out[:, i])
                        for i, lv in enumerate(c310.LEVELS)
                    },
                )
            )
    return pl.concat(frames)


def pred_path(*, arm: str, seed: int, start: datetime) -> Path:
    """Return a job's prediction file."""
    return common.PRED_DIR / arm / f"s{seed}_{start:%Y-%m}.parquet"


def run_job(*, arm: str, config: common.Config, seed: int, start: datetime) -> str:
    """Fit one (arm, seed, refit) and write its prediction table."""
    path = pred_path(arm=arm, seed=seed, start=start)
    if path.exists():
        return f"{arm} s{seed} {start:%Y-%m} skipped"
    began = time.time()
    cutoff = start - timedelta(days=f310.REFIT_GAP_DAYS)
    need_members = config.member_route or config.plant != "none"
    pieces = [
        ens_piece(farm=f, band=b, cutoff=cutoff, need_members=need_members)
        for f in c310.FARMS
        for b in c310.BAND_ORDER
    ]
    stats = (
        fixed_stats() if config.fixed_stats else standardiser(pieces=pieces, columns=config.columns)
    )
    if config.pretrain in ("finetune", "only"):
        state = pretrained_state(config=config, seed=seed)
        if config.pretrain == "finetune":
            model = train_network(
                pieces=pieces,
                config=config,
                seed=seed,
                stats=stats,
                learning_rate=common.FINE_TUNE_LEARNING_RATE,
                init_state=state,
            )
        else:
            plant = build_plant(config=config, pieces=list(era5_pieces(seed)))
            model = net.UnifiedNet(n_static=len(config.columns), config=config, plant=plant).to(
                DEVICE
            )
            model.load_state_dict(state)
            model.eval()
    else:
        model = train_network(
            pieces=pieces, config=config, seed=seed, stats=stats, learning_rate=common.LEARNING_RATE
        )
    end = f310.month_add(start=start, months=f310.REFIT_MONTHS)
    table = predict_rows(
        model=model,
        config=config,
        stats=stats,
        start=start,
        end=end,
        cutoff=cutoff,
        seed=seed,
        arm=arm,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    table.write_parquet(tmp)
    tmp.replace(path)
    del model
    torch.cuda.empty_cache()
    return f"{arm} s{seed} {start:%Y-%m} {table.height} rows {time.time() - began:.0f}s"


def main() -> None:
    """Run the named arms, seeds, and refits one after another in this process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+")
    parser.add_argument("--seeds", default=",".join(map(str, common.SEEDS_ABLATION)))
    parser.add_argument("--refits", default=",".join(f"{r:%Y-%m}" for r in REFITS))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    refits = [r for r in REFITS if f"{r:%Y-%m}" in args.refits.split(",")]
    for arm in args.arms:
        for seed in seeds:
            for start in refits:
                print(run_job(arm=arm, config=common.ARMS[arm], seed=seed, start=start), flush=True)


if __name__ == "__main__":
    main()

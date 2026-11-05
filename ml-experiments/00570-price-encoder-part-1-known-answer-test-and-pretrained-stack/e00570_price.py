# ruff: noqa: E501, ANN001, ANN201
"""Price series on the half-hourly grid, and the learnt price encoder of experiment 570.

Prices come from Elexon's market index (provider `APXMIDP`) and system price tables, 2019-09 to
2026-09, stored privately under this experiment's data folder (`market_early/`). Every price window
may look forward (the price is an oracle available only at training time and at oracle scoring).

Channels on the grid (T half-hours from `GRID_START_US`): 0 the market index price, 1 the system
price minus the market index price, 2 the net imbalance volume, 3 the presence mask of channel 0.
Channels 0 to 2 are transformed causally: signed log with the trailing 28-day median of |p| as the
scale, minus the trailing 28-day median, divided by the trailing 28-day interquartile range.
"""

import datetime as dt
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from sklearn.ensemble import HistGradientBoostingRegressor
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from torch import nn  # ty: ignore[unresolved-import]

STEP_US: Final[int] = 1_800_000_000
STEPS_PER_DAY: Final[int] = 48
TRAIL: Final[int] = 28 * STEPS_PER_DAY
FINE_HALF: Final[int] = 48  # fine window: target - 24 h to target + 24 h, 97 values
MEDIUM_HALF: Final[int] = 84  # medium window: 168 hourly means, +/- 3.5 days
COARSE_HALF: Final[int] = 14  # coarse window: 28 days, target day - 14 to + 13
N_CHANNELS: Final[int] = 4
SHAPE_DIM: Final[int] = 6
LATENT: Final[int] = 16
REGIMES: Final[int] = 8
DATA = next(ML_EXPERIMENTS_DATA_DIR.glob("00570-*")) / "market_early"

KINDS: Final = ("real", "res", "typical", "noise", "const")
"""Series kinds: the real transformed price, its calendar residual, the typical shape, noise, constant."""


def _trail_median(s: pl.Series, q: float) -> pl.Series:
    return s.rolling_quantile(q, window_size=TRAIL, min_samples=TRAIL // 4)


def load_raw(*, grid_start_us: int, n_steps: int) -> np.ndarray:
    """Return raw channels (3, T) of price, system-minus-index, and net imbalance volume (NaN = missing)."""
    mid = pl.read_parquet(DATA / "elexon_mid_apx_2019-09_to_2026-09.parquet")
    sp = pl.read_parquet(DATA / "elexon_system_prices_2019-09_to_2026-09.parquet")
    out = np.full((3, n_steps), np.nan, dtype=np.float64)

    def idx(frame: pl.DataFrame) -> np.ndarray:
        return ((frame["time"].dt.epoch("us").to_numpy() - grid_start_us) // STEP_US).astype(np.int64)

    mid = mid.with_columns(price=pl.when(pl.col("volume_mwh") > 0).then(pl.col("price_gbp_per_mwh")))
    i = idx(mid)
    ok = (i >= 0) & (i < n_steps)
    out[0, i[ok]] = mid["price"].to_numpy()[ok]
    j = idx(sp)
    okj = (j >= 0) & (j < n_steps)
    out[1, j[okj]] = sp["system_sell_price_gbp_per_mwh"].to_numpy()[okj]
    out[2, j[okj]] = sp["net_imbalance_volume_mwh"].to_numpy()[okj]
    out[1] = out[1] - out[0]
    return out


def transform(raw: np.ndarray) -> np.ndarray:
    """Return channels (4, T): three causally normalised signed-log series (0 where missing) and the mask."""
    chans = []
    mask = np.isfinite(raw[0])
    for k in range(3):
        s = pl.Series(raw[k]).fill_nan(None)
        scale = _trail_median(s.abs(), 0.5).fill_null(1.0).clip(lower_bound=1.0)
        a = np.sign(raw[k]) * np.log1p(np.abs(raw[k]) / scale.to_numpy())
        a_s = pl.Series(a).fill_nan(None)
        med = _trail_median(a_s, 0.5).fill_null(0.0).to_numpy()
        iqr = (_trail_median(a_s, 0.75) - _trail_median(a_s, 0.25)).fill_null(1.0).clip(lower_bound=0.05).to_numpy()
        z = (a - med) / iqr
        chans.append(np.where(np.isfinite(z), z, 0.0))
    return np.stack([*chans, mask.astype(np.float64)]).astype(np.float32)


def calendar_frame(*, grid_start_us: int, n_steps: int) -> pl.DataFrame:
    """Return the grid's half-hour of day, day type (0 weekday, 1 Saturday, 2 Sunday), and month."""
    t = pl.DataFrame({"t": pl.from_epoch(pl.Series(grid_start_us + STEP_US * np.arange(n_steps)), time_unit="us")})
    wd = pl.col("t").dt.weekday()
    return t.select(
        hh=pl.col("t").dt.hour() * 2 + pl.col("t").dt.minute() // 30,
        dtype=pl.when(wd <= 5).then(0).when(wd == 6).then(1).otherwise(2),
        month=pl.col("t").dt.month(),
    )


def residualise(*, chans: np.ndarray, cal: pl.DataFrame, fit_until: int) -> np.ndarray:
    """Subtract a calendar-only gradient-boosted tree's prediction from channels 0 and 1 (fitted before `fit_until`)."""
    out = chans.copy()
    X = cal.select("hh", "dtype", "month").to_numpy().astype(np.float32)
    ok = chans[3] > 0
    train = ok.copy()
    train[fit_until:] = False
    for k in (0, 1):
        tree = HistGradientBoostingRegressor(max_iter=60, max_depth=4, random_state=0)
        tree.fit(X[train], chans[k][train])
        out[k] = np.where(ok, chans[k] - tree.predict(X), 0.0)
    return out.astype(np.float32)


def typical(*, chans: np.ndarray, cal: pl.DataFrame, fit_until: int) -> np.ndarray:
    """Return the mean of each channel for (month, half-hour, day type), from rows before `fit_until`, on the whole grid."""
    out = chans.copy()
    keys = cal.with_columns(i=pl.int_range(pl.len()))
    for k in range(3):
        v = pl.DataFrame({"i": np.arange(chans.shape[1]), "v": chans[k], "m": chans[3]})
        train = keys.join(v, on="i").filter((pl.col("i") < fit_until) & (pl.col("m") > 0))
        mean = train.group_by("hh", "dtype", "month").agg(pl.col("v").mean().alias("mean"))
        out[k] = keys.join(mean, on=["hh", "dtype", "month"], how="left", maintain_order="left")["mean"].fill_null(0.0).to_numpy()
    out[3] = 1.0
    return out.astype(np.float32)


def noise(*, like: np.ndarray, seed: int) -> np.ndarray:
    """Return Gaussian noise channels of the same shape, mask all ones."""
    rng = np.random.default_rng(seed)
    out = rng.standard_normal(like.shape).astype(np.float32)
    out[3] = 1.0
    return out


def constant(*, like: np.ndarray) -> np.ndarray:
    """Return zero channels with the mask all ones."""
    out = np.zeros_like(like)
    out[3] = 1.0
    return out


def shape_features(*, chans: np.ndarray) -> np.ndarray:
    """Return hand-made shape vector (6, T) from channel 0: day rank, minus day mean, 12-step range, cheap4h, dear4h, evening-minus-night."""
    n = chans.shape[1]
    day = np.arange(n) // STEPS_PER_DAY
    df = pl.DataFrame({"d": day, "p": chans[0], "i": np.arange(n)})
    df = df.with_columns(
        rank=(pl.col("p").rank("average").over("d") - 1) / (STEPS_PER_DAY - 1),
        dev=pl.col("p") - pl.col("p").mean().over("d"),
        rng=pl.col("p").rolling_max(13, center=True, min_samples=1) - pl.col("p").rolling_min(13, center=True, min_samples=1),
    )
    rank_ = df["rank"].to_numpy()
    cheap = (rank_ < 8 / STEPS_PER_DAY).astype(np.float32)
    dear = (rank_ >= 1 - 8 / STEPS_PER_DAY).astype(np.float32)
    hh = np.arange(n) % STEPS_PER_DAY
    eve = pl.DataFrame({"d": day, "p": np.where((hh >= 32) & (hh < 38), chans[0], np.nan)}).with_columns(
        e=pl.col("p").mean().over("d")
    )["e"].to_numpy()
    night = pl.DataFrame({"d": day, "p": np.where(hh < 10, chans[0], np.nan)}).with_columns(
        e=pl.col("p").mean().over("d")
    )["e"].to_numpy()
    en = np.nan_to_num(eve - night)
    out = np.stack([rank_, df["dev"].to_numpy(), df["rng"].to_numpy(), cheap, dear, en])
    return (np.nan_to_num(out) * (chans[3] > 0)).astype(np.float32)


def daily_stats(*, chans: np.ndarray) -> np.ndarray:
    """Return per-day stats (4, D): mean, max, min, evening-minus-overnight of channel 0."""
    n = chans.shape[1] // STEPS_PER_DAY * STEPS_PER_DAY
    d = chans[0, :n].reshape(-1, STEPS_PER_DAY)
    return np.stack([d.mean(1), d.max(1), d.min(1), d[:, 32:38].mean(1) - d[:, :10].mean(1)]).astype(np.float32)


class Bundle:
    """The device tensors one price encoder reads: channels, shape vector, hourly means, daily stats."""

    def __init__(self, *, parts: list[np.ndarray], device: str) -> None:
        """Build the bundle from channel arrays (4, T) laid end to end; each T is a multiple of 48."""
        assert all(p.shape[1] % STEPS_PER_DAY == 0 for p in parts)
        self.part_len = parts[0].shape[1]
        self.T = sum(p.shape[1] for p in parts)
        hourly = [(p[:3] + np.roll(p[:3], -1, axis=1)) / 2 for p in parts]
        self.chans = torch.from_numpy(np.concatenate(parts, axis=1)).to(device)
        self.hourly = torch.from_numpy(np.concatenate(hourly, axis=1).astype(np.float32)).to(device)
        self.shape = torch.from_numpy(np.concatenate([shape_features(chans=p) for p in parts], axis=1)).to(device)
        self.daily = torch.from_numpy(np.concatenate([daily_stats(chans=p) for p in parts], axis=1)).to(device)

    def windows(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return fine (B,4,97), medium (B,3,168), coarse (B,4,28), shape (B,6) windows for target indices."""
        dev = idx.device
        fine_pos = idx[:, None] + torch.arange(-FINE_HALF, FINE_HALF + 1, device=dev)[None]
        med_pos = idx[:, None] + 2 * torch.arange(-MEDIUM_HALF, MEDIUM_HALF, device=dev)[None]
        day = idx // STEPS_PER_DAY
        coarse_pos = day[:, None] + torch.arange(-COARSE_HALF, COARSE_HALF, device=dev)[None]
        fine = self.chans[:, fine_pos.clamp(0, self.T - 1)].permute(1, 0, 2)
        medium = self.hourly[:, med_pos.clamp(0, self.T - 1)].permute(1, 0, 2)
        coarse = self.daily[:, coarse_pos.clamp(0, self.daily.shape[1] - 1)].permute(1, 0, 2)
        return {"fine": fine, "medium": medium, "coarse": coarse, "shape": self.shape[:, idx.clamp(0, self.T - 1)].T}


def glu_tower(c_in: int, k: int, stride: int = 1) -> nn.Sequential:
    """Return a two-layer gated convolution tower with 16 output channels."""
    return nn.Sequential(
        nn.Conv1d(c_in, 32, k, stride=stride, padding=k // 2), nn.GLU(dim=1),
        nn.Conv1d(16, 32, 3, padding=1), nn.GLU(dim=1),
    )  # fmt: skip


class PriceEncoder(nn.Module):
    """Three convolutional towers, attention over the fine window, a regime bank, and the shape vector.

    Output: a 16-number latent, or a learned null vector where the presence flag is 0.
    """

    def __init__(self) -> None:
        """Build the encoder."""
        super().__init__()
        self.fine = glu_tower(N_CHANNELS, 7)
        self.medium = glu_tower(3, 5, stride=2)
        self.coarse = glu_tower(4, 3)
        self.pos = nn.Embedding(STEPS_PER_DAY, 16)
        self.proj = nn.Conv1d(N_CHANNELS, 16, 1)
        self.attn = nn.MultiheadAttention(16, 2, batch_first=True)
        self.protos = nn.Parameter(torch.randn(REGIMES, STEPS_PER_DAY) * 0.1)
        self.tau = nn.Parameter(torch.tensor(5.0))
        self.out = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 16 + REGIMES + SHAPE_DIM, 64), nn.GELU(), nn.Linear(64, LATENT)
        )
        self.null = nn.Parameter(torch.zeros(LATENT))
        self.last_regimes: torch.Tensor | None = None

    @staticmethod
    def _pool(h: torch.Tensor) -> torch.Tensor:
        return torch.cat([h.mean(-1), h.amax(-1)], dim=-1)

    def forward(self, w: dict[str, torch.Tensor], idx: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Return (B, 16) latents; rows with `present` 0 get the null vector."""
        fine = w["fine"]
        hf = self.fine(fine)
        hours = ((idx[:, None] + torch.arange(-FINE_HALF, FINE_HALF + 1, device=idx.device)[None]) % STEPS_PER_DAY)
        tokens = self.proj(fine).transpose(1, 2) + self.pos(hours)
        att, _ = self.attn(tokens[:, FINE_HALF : FINE_HALF + 1], tokens, tokens)
        day_vec = fine[:, 0, FINE_HALF - 24 : FINE_HALF + 24]
        day_vec = day_vec / (day_vec.norm(dim=-1, keepdim=True) + 1e-6)
        sim = day_vec @ (self.protos / (self.protos.norm(dim=-1, keepdim=True) + 1e-6)).T
        reg = torch.softmax(self.tau * sim, dim=-1)
        self.last_regimes = reg
        z = torch.cat(
            [self._pool(hf[:, :16]), self._pool(self.medium(w["medium"])), self._pool(self.coarse(w["coarse"])), att[:, 0], reg, w["shape"]],
            dim=-1,
        )
        lat = self.out(z)
        return torch.where(present[:, None] > 0.5, lat, self.null[None].expand_as(lat))

    def aux_loss(self) -> torch.Tensor:
        """Return the regime bank's entropy and prototype-diversity penalties (small, fixed weights)."""
        reg = self.last_regimes
        assert reg is not None
        mean = reg.mean(0)
        ent_batch = -(mean * (mean + 1e-8).log()).sum()  # high = prototypes all used
        p = self.protos / (self.protos.norm(dim=-1, keepdim=True) + 1e-6)
        sim = (p @ p.T - torch.eye(REGIMES, device=p.device)).abs().mean()
        return 0.01 * (-ent_batch) + 0.01 * sim


def build_series(*, kind: str, grid_start_us: int, n_steps: int, fit_until: int, seed: int = 0) -> np.ndarray:
    """Return channels (4, T) of a series kind for a fold whose price statistics are fitted before `fit_until`."""
    chans = transform(load_raw(grid_start_us=grid_start_us, n_steps=n_steps))
    cal = calendar_frame(grid_start_us=grid_start_us, n_steps=n_steps)
    if kind == "real":
        return chans
    if kind == "res":
        return residualise(chans=chans, cal=cal, fit_until=fit_until)
    if kind == "typical":
        return typical(chans=chans, cal=cal, fit_until=fit_until)
    if kind == "noise":
        return noise(like=chans, seed=seed)
    if kind == "const":
        return constant(like=chans)
    raise ValueError(kind)


def donor_shift(*, n_steps: int, grid_start_us: int, seed: int) -> np.ndarray:
    """Return, for each grid day, the shift in half-hours to a donor day (shuffled-day control P1).

    The donor has the same calendar month and day type and lies at least 3 days earlier, so that it is
    strictly before the row's origin. Days with no such donor keep shift 0 and are counted by the caller.
    """
    rng = np.random.default_rng(seed)
    n_days = n_steps // STEPS_PER_DAY
    start = dt.datetime.fromtimestamp(grid_start_us / 1e6, dt.UTC)
    days = [start + dt.timedelta(days=int(d)) for d in range(n_days)]
    key = np.array([(d.month, min(d.weekday(), 5) if d.weekday() < 5 else d.weekday()) for d in days])
    code = key[:, 0] * 10 + np.where(key[:, 1] < 5, 0, key[:, 1] - 4)  # month, weekday/sat/sun
    shift = np.zeros(n_days, dtype=np.int64)
    by: dict[int, np.ndarray] = {c: np.nonzero(code == c)[0] for c in np.unique(code)}
    for d in range(n_days):
        cand = by[int(code[d])]
        cand = cand[cand <= d - 3]
        if len(cand):
            shift[d] = (int(rng.choice(cand)) - d) * STEPS_PER_DAY
    return shift


def selftest() -> None:
    """Print shapes and a forward pass."""
    n = 140_016
    gs = int(dt.datetime(2019, 1, 1, tzinfo=dt.UTC).timestamp() * 1e6)
    chans = build_series(kind="real", grid_start_us=gs, n_steps=n, fit_until=n)
    b = Bundle(parts=[chans, chans], device="cpu")
    idx = torch.tensor([60_000, 70_000 + n])
    w = b.windows(idx)
    print({k: tuple(v.shape) for k, v in w.items()})
    enc = PriceEncoder()
    print(enc(w, idx, torch.tensor([1.0, 0.0])).shape, float(enc.aux_loss()))
    print("present share", float(chans[3].mean()), "nonzero mid", float((chans[0] != 0).mean()))
    print("donor shifts filled", float((donor_shift(n_steps=n, grid_start_us=gs, seed=0) != 0).mean()))
    print(Path(str(DATA)).exists())


if __name__ == "__main__":
    selftest()

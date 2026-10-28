# ruff: noqa: E501
"""Score the Kalman-transient arms (control, K0 off, KA adaptive gain) from experiment 364's stage columns.

    DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data OMP_NUM_THREADS=2 \
        uv run --no-sync python ml-experiments/00367-*/e00367_score.py > printed_report.txt

Stage columns (experiment 364): `S1` plain GAM plus Kalman level, `S2` plus Kalman transient,
`S4` the stack. The transient is `S2__q50 - S1__q50`. Arms at stack level shift all 11 quantiles of
`S4` by the change in the transient, so the band keeps its shape (an approximation, labelled).
"""

import json
import os
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

DATA: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
WORK: Final[Path] = DATA / "00367-d8-kalman-transient-off-and-adaptive-short-lead" / "work"
SRC: Final[Path] = DATA / "00364-d8-stage-blame-and-analogue-recent-power" / "work"
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
SITES: Final[dict[str, str]] = {"D8": "D8_weekly", "D2": "D2_monthly", "D4": "D4_monthly"}
WINDOW_DAYS: Final[int] = 90
MIN_ROWS: Final[int] = 14 * 48
SHORT_LEAD_H: Final[float] = 24.0
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-24h": (0.0, 24.0), "24-72h": (24.0, 72.0), "72-168h": (72.0, 168.0),
}  # fmt: skip
N_RESAMPLES: Final[int] = 20_000
ARMS: Final[tuple[str, ...]] = ("control", "K0", "KA", "K0_stage", "KA_stage")


def pinball_score(*, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 quantiles, per row (q is rows x 11)."""
    diff = actual[:, None] - q
    return 2.0 * np.mean(np.maximum(LEVELS * diff, (LEVELS - 1.0) * diff), axis=1)


def adaptive_gain(*, wide: pl.DataFrame) -> np.ndarray:
    """Return the per-row gain g of arm KA: trailing benefit of the transient over its size, in [0, 1]."""
    valid = wide["valid_time"].dt.epoch("s").to_numpy().astype(np.float64)
    init = wide["power_fcst_init_time"].dt.epoch("s").to_numpy().astype(np.float64)
    lead = wide["lead_hours"].to_numpy()
    actual = wide["actual"].to_numpy().astype(np.float64)
    s1, s2 = (wide[f"{s}__q50"].to_numpy().astype(np.float64) for s in ("S1", "S2"))
    short = lead <= SHORT_LEAD_H
    benefit = np.abs(s1 - actual) - np.abs(s2 - actual)
    size = np.abs(s2 - s1)
    order = np.argsort(valid[short])
    v = valid[short][order]
    cb = np.concatenate([[0.0], np.cumsum(benefit[short][order])])
    cs = np.concatenate([[0.0], np.cumsum(size[short][order])])
    runs = np.unique(init)
    gain_by_run = {}
    for t in runs:
        hi = np.searchsorted(v, t, side="right")
        lo = np.searchsorted(v, t - WINDOW_DAYS * 86400, side="right")
        n = hi - lo
        if n < MIN_ROWS or cs[hi] - cs[lo] <= 0:
            gain_by_run[t] = 1.0
        else:
            gain_by_run[t] = float(np.clip((cb[hi] - cb[lo]) / (cs[hi] - cs[lo]), 0.0, 1.0))
    return np.array([gain_by_run[t] for t in init])


def arm_quantiles(*, wide: pl.DataFrame, gain: np.ndarray) -> dict[str, np.ndarray]:
    """Return each arm's 11 quantiles (rows x 11)."""
    short = wide["lead_hours"].to_numpy() <= SHORT_LEAD_H
    transient = (wide["S2__q50"] - wide["S1__q50"]).to_numpy().astype(np.float64)
    s4 = wide.select([f"S4__{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    s2 = wide.select([f"S2__{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    removed = {"K0": np.where(short, 1.0, 0.0), "KA": np.where(short, 1.0 - gain, 0.0)}
    out = {"control": s4}
    for arm, frac in removed.items():
        out[arm] = s4 - (frac * transient)[:, None]
        out[f"{arm}_stage"] = s2 - (frac * transient)[:, None]
    return out


class Bootstrap:
    """Month-resampled intervals for one row subset (shared draws, so contrasts are paired)."""

    def __init__(self, *, months: np.ndarray, seed: int = 0) -> None:  # noqa: D107
        self.labels = np.unique(months)
        self.months = months
        draws = np.random.default_rng(seed).integers(
            0, self.labels.size, (N_RESAMPLES, self.labels.size)
        )
        self.counts = (draws[:, :, None] == np.arange(self.labels.size)).sum(axis=1).astype(float)
        self.n = np.array([(months == m).sum() for m in self.labels], dtype=float)

    def mean(self, *, values: np.ndarray) -> list[float]:
        """Return the mean and its 95% interval."""
        sums = np.array([values[self.months == m].sum() for m in self.labels])
        draws = (self.counts @ sums) / (self.counts @ self.n)
        return [float(sums.sum() / self.n.sum()), *np.quantile(draws, [0.025, 0.975]).tolist()]

    def stats(self, *, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return per-month (n, sx, sy, sxx, syy, sxy)."""
        return np.array([
            [m.sum(), x[m].sum(), y[m].sum(), (x[m] ** 2).sum(), (y[m] ** 2).sum(), (x[m] * y[m]).sum()]
            for m in (self.months == lab for lab in self.labels)
        ])  # fmt: skip

    @staticmethod
    def corr(s: np.ndarray) -> np.ndarray:
        """Return Pearson correlation from summed statistics."""
        n, sx, sy, sxx, syy, sxy = (s[..., k] for k in range(6))
        return (sxy / n - sx / n * sy / n) / np.sqrt(
            (sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2)
        )

    def corr_diff(self, *, x: np.ndarray, base: np.ndarray, y: np.ndarray) -> list[float]:
        """Return corr(x, y) minus corr(base, y) with its interval."""
        a, b = self.stats(x=x, y=y), self.stats(x=base, y=y)
        d = self.corr(self.counts @ a) - self.corr(self.counts @ b)
        return [
            float(self.corr(a.sum(0)) - self.corr(b.sum(0))),
            *np.quantile(d, [0.025, 0.975]).tolist(),
        ]


def tail_stats(*, q: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Return coverage (share of actuals below the quantile) and pinball loss at q01, q05, q95, q99."""
    out = {}
    for name in ("q01", "q05", "q95", "q99"):
        i = QUANTILES.index(name)
        tau = LEVELS[i]
        d = actual - q[:, i]
        out[f"{name}_coverage"] = float((actual <= q[:, i]).mean())
        out[f"{name}_pinball"] = float(np.maximum(tau * d, (tau - 1) * d).mean())
    return out


def slices(*, wide: pl.DataFrame, site: str) -> dict[str, np.ndarray]:
    """Return boolean row masks: all, lead bands, day types."""
    lead = wide["lead_hours"].to_numpy()
    out = {"all": np.ones(wide.height, bool)}
    out |= {k: (lead > lo) & (lead <= hi) for k, (lo, hi) in LEAD_BANDS.items()}
    if site == "D8":
        flag = np.array(json.loads((SRC / "step_day_flags.json").read_text()))
        out |= {"step_days": flag, "no_step_days": ~flag}
    out["cold_winter_night"] = wide["cold_night"].to_numpy()
    out["winter_night"] = wide["winter_night"].to_numpy()
    out["sunny_apr_jun"] = wide["sunny"].to_numpy() & np.isin(
        wide["valid_time"].dt.month().to_numpy(), [4, 5, 6]
    )
    return out


def main() -> None:
    """Score every arm at every site and write `work/scores.json`."""
    WORK.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for site, key in SITES.items():
        wide = pl.read_parquet(SRC / f"wide_{key}.parquet")
        actual = wide["actual"].to_numpy().astype(np.float64)
        gain = adaptive_gain(wide=wide)
        qs = arm_quantiles(wide=wide, gain=gain)
        score = {a: pinball_score(q=q, actual=actual) for a, q in qs.items()}
        gate = float(np.abs(score["control"] - wide["s_S4"].to_numpy()).max())
        med = {a: q[:, QUANTILES.index("q50")] for a, q in qs.items()}
        res: dict = {"rows": wide.height, "gate_max_abs_diff_vs_stored_s_S4": gate,
                     "gain_mean_short_lead": float(gain[wide["lead_hours"].to_numpy() <= SHORT_LEAD_H].mean()),
                     "slices": {}}  # fmt: skip
        months = wide["month"].to_numpy()
        for name, mask in slices(wide=wide, site=site).items():
            if mask.sum() == 0:
                continue
            boot = Bootstrap(months=months[mask])
            block: dict = {"n": int(mask.sum()), "arms": {}, "contrasts": {}, "tails": {}}
            for a in ARMS:
                block["arms"][a] = {
                    "score": boot.mean(values=score[a][mask]),
                    "mae": boot.mean(values=np.abs(med[a][mask] - actual[mask])),
                    "corr": [float(boot.corr(boot.stats(x=med[a][mask], y=actual[mask]).sum(0)))],
                }
                block["tails"][a] = tail_stats(q=qs[a][mask], actual=actual[mask])
            for a, b in [("K0", "control"), ("KA", "control"), ("KA", "K0"),
                         ("K0_stage", "control"), ("KA_stage", "control")]:  # fmt: skip
                block["contrasts"][f"{a}-{b}"] = {
                    "score": boot.mean(values=score[a][mask] - score[b][mask]),
                    "corr": boot.corr_diff(x=med[a][mask], base=med[b][mask], y=actual[mask]),
                }
            res["slices"][name] = block
        results[site] = res
        print(site, "rows", wide.height, "gate max |diff| vs s_S4:", round(gate, 6),
              "mean KA gain at <=24h:", round(res["gain_mean_short_lead"], 3))  # fmt: skip
        for name, block in res["slices"].items():
            line = {a: round(v["score"][0], 3) for a, v in block["arms"].items()}
            print(f"  {name:18s} n={block['n']:6d}", line)
            for c, v in block["contrasts"].items():
                if c in ("K0-control", "KA-control"):
                    print(
                        f"      {c}: score {np.round(v['score'], 3).tolist()} corr {np.round(v['corr'], 3).tolist()}"
                    )
    (WORK / "scores.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

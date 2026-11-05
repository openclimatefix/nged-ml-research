# ruff: noqa: E501
"""Experiment 577: how static each arm's learnt offset is (descriptive), and the offset plots.

Reads the last fold's (2026-06) pretrained weights, seeds 0 and 1, which hold the offset of every
month (monthly arms) or day (O2) from 2019-09. An offset is converted to a share of capacity by the
Primary's target standard deviation. A month is event-free for a Primary when no labelled event lies
within one month of it. Measures, per arm, pooled over the Primaries and both seeds:

- the median absolute month-to-month change in event-free months (share of capacity);
- the share of event-free month pairs whose change is under 0.2% of capacity;
- the number of month-to-month changes of 1% of capacity or more, per Primary-year (all months).

Writes `static.json` and `offsets_<D-number>.svg`-style figures into the experiment's docs folder.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00510_common as c  # ty: ignore[unresolved-import]
import e00577_model_variants as mv  # ty: ignore[unresolved-import]
import matplotlib as mpl
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

mpl.use("Agg")
import matplotlib.pyplot as plt

NAME: Final = "00577-static-between-steps-offsets-and-probabilistic-offset-widening"
W: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
W565: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00565-drift-embedding-variants-six-designs" / "work"
MONTH: Final = "2026-06"
SEEDS: Final = (0, 1)
START: Final = datetime(2019, 9, 1, tzinfo=UTC)
COLOURS: Final = {"D5": "#306BFF", "O1": "#FF4901", "O2": "#17E58F", "O3": "#B701FF"}
SITES: Final = pl.read_csv(next(ML_EXPERIMENTS_DATA_DIR.glob("00521*/site_map.csv")))
SMALL: Final = 0.002
JUMP: Final = 0.01


def monthly_offsets(*, arm: str, seed: int, sd: np.ndarray) -> np.ndarray:
    """Return the arm's offset at the start of each month, (Primaries, months), as a share of capacity."""
    path = (W565 / "pretrained_D5" / "real") if arm == "D5" else (W / f"pretrained_{arm}" / "real")
    state = torch.load(path / f"{MONTH}_s{seed}.pt", map_location="cpu")
    if arm == "O3":
        gate = (torch.sigmoid(state["log_alpha"]) * (mv.HC_ZETA - mv.HC_GAMMA) + mv.HC_GAMMA).clamp(
            0, 1
        )
        knots = torch.cumsum(gate * state["amp"], dim=1).numpy()
    elif arm == "O2":
        daily = state["knots"][:, :, 0].numpy()
        first = mv.MONTH_FIRST_DAY[: np.searchsorted(mv.MONTH_FIRST_DAY, daily.shape[1])]
        knots = daily[:, first]
    else:
        knots = state["knots"][:, :, 0].numpy()
    return knots * sd[:, None]


def event_free(*, n_months: int) -> np.ndarray:
    """Return (Primaries, months) True where no labelled event lies within one month."""
    free = np.ones((16, n_months), dtype=bool)
    grid, step = c.prep422.GRID_START_US, c.prep422.STEP_US
    for p, spans in c.labelled_events().items():
        for onset, end in spans:
            t0, t1 = (datetime.fromtimestamp((grid + i * step) / 1e6, UTC) for i in (onset, end))
            m0 = (t0.year - START.year) * 12 + t0.month - START.month
            m1 = (t1.year - START.year) * 12 + t1.month - START.month
            free[p, max(m0 - 1, 0) : min(m1 + 2, n_months)] = False
    return free


def main() -> None:
    """Compute the measures and draw the offset plots."""
    sd = np.nanstd(c.clean_stack(), axis=1)
    arms = ("D5", "O1", "O2", "O3")
    off = {a: [monthly_offsets(arm=a, seed=s, sd=sd) for s in SEEDS] for a in arms}
    n = min(o.shape[1] for a in arms for o in off[a])
    free = event_free(n_months=n)
    pair_free = free[:, :-1] & free[:, 1:]
    out: dict = {}
    for a in arms:
        change = np.concatenate([np.abs(np.diff(o[:, :n], axis=1)) for o in off[a]], axis=0)
        mask = np.concatenate([pair_free] * len(SEEDS), axis=0)
        ch = change[mask]
        out[a] = {
            "median_abs_change_event_free": float(np.median(ch)),
            "share_under_0.2pct_event_free": float((ch < SMALL).mean()),
            "jumps_1pct_per_primary_year": float(
                (change >= JUMP).sum() / (len(SEEDS) * 16 * (n - 1) / 12)
            ),
            "event_free_pairs": int(mask.sum()),
        }
        print(a, out[a], flush=True)
    (W / "static.json").write_text(json.dumps(out, indent=1))
    spans = c.labelled_events()
    label = dict(zip(SITES["primary"].to_list(), SITES["label"].to_list(), strict=True))
    busiest = sorted(spans, key=lambda p: -len(spans[p]))[:4]
    fig, axes = plt.subplots(len(busiest), 1, figsize=(7.5, 2.1 * len(busiest)), sharex=True)
    months = np.arange(n)
    for ax, p in zip(axes, busiest, strict=True):
        for onset, end in spans[p]:
            t0, t1 = (
                datetime.fromtimestamp((c.prep422.GRID_START_US + i * c.prep422.STEP_US) / 1e6, UTC)
                for i in (onset, end)
            )
            ax.axvspan(
                (t0 - START).days / 30.44,
                max((t1 - START).days / 30.44, (t0 - START).days / 30.44 + 0.3),
                color="#cccccc",
                alpha=0.7,
                lw=0,
            )
        for a in arms:
            ax.plot(months, 100 * off[a][0][p, :n], color=COLOURS[a], lw=1.4, label=a)
        ax.set_ylabel(f"{label[p]}\noffset, % of capacity", fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0].legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.4))
    axes[-1].set_xlabel("months since 2019-09 (grey bands: labelled events)", fontsize=8)
    fig.tight_layout()
    dest = (
        Path(__file__).parents[2]
        / "docs"
        / "ml-experiments"
        / "00577-static-between-steps-offsets-and-probabilistic-offset-widening"
    )
    fig.savefig(dest / "offsets_by_arm.svg")


if __name__ == "__main__":
    main()

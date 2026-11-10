# ruff: noqa: ANN001, ANN202, PLR0915
"""Shared constants and the figure renderer of experiment 692.

A candidate is a donor end (a decoded fall) of E1-J-sym with its largest decoded partner. The figure
shows the candidate with no label, no truth mark, no date, and neutral Primary names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00692-event-table-reviewer-calibration-gate"
DAY: Final[int] = 48
PAD_DAYS: Final[int] = 7
CAP_DAYS: Final[int] = 60
OPEN_DAYS: Final[int] = 30
"""Days shown after the onset when the candidate has no end inside the record."""
ZOOM_HH: Final[int] = 96
"""Half-hours either side of an onset or an end in the zoom panels (48 hours)."""
BLOCK: Final[int] = 12
COLOURS: Final[dict[str, str]] = {
    "donor": "#1f77b4",
    "recipient": "#d95f02",
    "grey": "#9a9a9a",
    "forecast": "#7a7a7a",
}


def _slice(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    out = np.full(hi - lo, np.nan)
    a, b = max(lo, 0), min(hi, len(x))
    if b > a:
        out[a - lo : b - lo] = x[a:b]
    return out


def _block_mean(y: np.ndarray) -> np.ndarray:
    n = len(y) // BLOCK * BLOCK
    with np.errstate(all="ignore"):
        return np.nanmean(y[:n].reshape(-1, BLOCK), axis=1)


def render(
    *,
    path: Path,
    donor: dict,
    recipient: dict,
    others: list[dict],
    median_resid_units: np.ndarray,
    onset: int,
    end: int | None,
    donor_step: tuple[int, int | None, float],
    recipient_step: tuple[int, int | None, float],
) -> None:
    """Draw one candidate. Each site dict holds `power`, `forecast`, `resid` (MW) and `sigma`.

    The whole-grid arrays are indexed in half-hours. `donor_step` and `recipient_step` are the
    decoded (start, end, amplitude in noise units) drawn as a horizontal line.
    """
    stop = onset + OPEN_DAYS * DAY if end is None else end
    lo = onset - PAD_DAYS * DAY
    hi = min(stop + PAD_DAYS * DAY, lo + CAP_DAYS * DAY)
    windows = [
        ("Overview (6-hour means; days from the donor onset)", lo, hi, True),
        ("Zoom at the donor onset (hours from it)", onset - ZOOM_HH, onset + ZOOM_HH, False),
        (
            "Zoom at the end (hours from it)" if end is not None else "No end inside the record",
            (end or onset) - ZOOM_HH,
            (end or onset) + ZOOM_HH,
            False,
        ),
    ]
    fig, axes = plt.subplots(6, 3, figsize=(11.5, 11.5), dpi=90)
    rows = [
        ("Donor: power and expected power (MW)", donor, "power"),
        ("Donor: residual (noise units)", donor, "resid"),
        ("Recipient: power and expected power (MW)", recipient, "power"),
        ("Recipient: residual (noise units)", recipient, "resid"),
        ("Other neighbours' residual (noise units)", donor, "others"),
        ("Median residual of all Primaries (noise units)", donor, "median"),
    ]
    for c, (title, a, b, overview) in enumerate(windows):
        if c == 2 and end is None:
            for r in range(6):
                axes[r, c].axis("off")
            axes[0, c].set_title(title, fontsize=8)
            continue
        t_ref = onset if c < 2 or end is None else end
        axes[0, c].set_title(title, fontsize=8)
        for r, (label, site, kind) in enumerate(rows):
            ax = axes[r, c]
            ax.tick_params(labelsize=7)
            ax.grid(color="#e5e5e5", linewidth=0.5)
            if c == 0:
                ax.set_ylabel(label, fontsize=6.5)

            def series(y: np.ndarray, a=a, b=b, overview=overview, t_ref=t_ref):
                v = _slice(y, a, b)
                if overview:
                    return (np.arange(len(v) // BLOCK) * BLOCK + BLOCK / 2 + a - onset) / DAY, (
                        _block_mean(v)
                    )
                return (np.arange(len(v)) + a - t_ref) / 2, v

            if kind == "power":
                colour = COLOURS["donor"] if site is donor else COLOURS["recipient"]
                x, y = series(site["power"])
                ax.plot(x, y, color=colour, linewidth=1)
                x, y = series(site["forecast"])
                ax.plot(x, y, color=COLOURS["forecast"], linewidth=1, linestyle="--")
            elif kind == "resid":
                colour = COLOURS["donor"] if site is donor else COLOURS["recipient"]
                x, y = series(site["resid"] / site["sigma"])
                ax.plot(x, y, color=colour, linewidth=1)
                step = donor_step if site is donor else recipient_step
                s0, s1, amp = step
                s1 = (stop + 10 * DAY) if s1 is None else s1
                s0, s1 = max(s0, a), min(s1, b)
                if s1 > s0:
                    xs = [
                        (s0 - onset) / DAY if overview else (s0 - t_ref) / 2,
                        (s1 - onset) / DAY if overview else (s1 - t_ref) / 2,
                    ]
                    ax.plot(xs, [amp, amp], color="black", linewidth=1)
                ax.axhline(0, color="#bbbbbb", linewidth=0.5)
            elif kind == "others":
                for o in others:
                    x, y = series(o["resid"] / o["sigma"])
                    ax.plot(x, y, color=COLOURS["grey"], linewidth=0.8)
                ax.axhline(0, color="#bbbbbb", linewidth=0.5)
            else:
                x, y = series(median_resid_units)
                ax.plot(x, y, color="#444444", linewidth=1)
                ax.axhline(0, color="#bbbbbb", linewidth=0.5)
    axes[0, 0].legend(
        handles=[
            plt.Line2D([], [], color=COLOURS["donor"], label="donor power or residual"),
            plt.Line2D([], [], color=COLOURS["forecast"], linestyle="--", label="expected power"),
            plt.Line2D([], [], color="black", label="decoded step"),
        ],
        fontsize=6,
        loc="best",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

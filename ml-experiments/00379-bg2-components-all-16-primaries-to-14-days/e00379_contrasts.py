# ruff: noqa: E501, D107, RUF007, PLR0912, PLR0915
"""Paired contrasts, per-Primary tables, and tail tables of experiment 379, from the scored rows.

Reads `work/scores/rows_<tag>.parquet` written by `e00379_score.py` and prints every number the
write-up quotes. The interval is the percentile interval of a bootstrap that resamples whole
calendar months of the forecast origin, the procedure of `studies.eval_protocol.metrics`
(`month_bootstrap_contrast`), computed from each month's total so that millions of rows resample
in milliseconds. `--check` compares one contrast with the module's own function.

    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00379-*/e00379_contrasts.py
"""

import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00379-bg2-components-all-16-primaries-to-14-days" / "work"
)
SCORES: Final[Path] = WORK / "scores"
MEMBERS: Final[tuple[str, ...]] = ("W", "M", "AN", "PT", "ST")
STAIRCASE: Final[tuple[str, ...]] = (
    "s1_frozen",
    "s2_stack",
    "s3_monthly",
    "s4_weekly",
    "s5_blend_no_offset",
    "bg2",
)
HEADLINE_BAND: Final[str] = "lead_days_1_10"
BANDS: Final[tuple[str, ...]] = (
    HEADLINE_BAND,
    "lead_day_1",
    "lead_days_5_7",
    "lead_days_8_10",
    "lead_days_11_14",
)
SIGNED: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "signed-MW": SIGNED,
    "MVA-only with D8": MVA,
    "MVA-only without D8": tuple(x for x in MVA if x != "D8"),
    "short-history (report only)": ("D15", "D18"),
}
TAILS: Final[tuple[str, ...]] = (
    "pinball_q01",
    "pinball_q05",
    "pinball_q95",
    "pinball_q99",
    "coverage_q01",
    "coverage_q05",
    "coverage_q95",
    "coverage_q99",
    "twcrps",
)
SEED: Final[int] = 0


def interval(
    *, sums: np.ndarray, counts: np.ndarray, level: float, n_resamples: int
) -> tuple[float, float]:
    """Return the percentile interval of the resampled mean, resampling whole months."""
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, counts.size, size=(n_resamples, counts.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (100.0 - level) / 2.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


class Wide:
    """One row per half-hour with each arm's per-row metrics as columns."""

    def __init__(self, *, tag: str) -> None:
        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
        rows = pl.scan_parquet(SCORES / f"rows_{tag}.parquet")
        self.arms = sorted(rows.select("arm").unique().collect().to_series().to_list())
        base = (
            rows.filter(pl.col("arm") == self.arms[0])
            .select(*keys, "label", "group", *BANDS)
            .sort(keys)
            .collect()
        )
        self.base = base.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        self.metrics: dict[str, dict[str, np.ndarray]] = {}
        for arm in self.arms:
            one = (
                rows.filter(pl.col("arm") == arm)
                .select(*keys, "crps", "mae_median", *TAILS)
                .sort(keys)
                .collect()
            )
            if not (one.select(keys).equals(base.select(keys))):
                msg = f"arm {arm} does not hold the shared rows"
                raise ValueError(msg)
            self.metrics[arm] = {c: one[c].to_numpy() for c in ("crps", "mae_median", *TAILS)}
        self.label = self.base["label"].to_numpy()
        self.month = self.base["month"].to_numpy()
        self.band = {b: self.base[b].to_numpy() for b in BANDS}

    def mask(self, *, labels: tuple[str, ...], band: str) -> np.ndarray:
        """Return the rows of some Primaries in a lead band."""
        return np.isin(self.label, labels) & self.band[band]

    def contrast(
        self, *, treatment: str, reference: str, mask: np.ndarray, level: float, n: int
    ) -> tuple[float, float, float, int]:
        """Return the mean paired CRPS difference, its interval, and the half-hour count."""
        diff = (
            self.metrics[treatment]["crps"][mask].astype(np.float64)
            - self.metrics[reference]["crps"][mask]
        )
        months, index = np.unique(self.month[mask], return_inverse=True)
        sums = np.bincount(index, weights=diff, minlength=months.size)
        counts = np.bincount(index, minlength=months.size).astype(np.float64)
        low, high = interval(sums=sums, counts=counts, level=level, n_resamples=n)
        return float(diff.mean()), low, high, int(mask.sum())


def pairs(*, tag: str) -> list[tuple[str, str, str]]:
    """Return the registered contrasts as (label, treatment, reference)."""
    if tag == "stack":
        return [("C6 stack_377 vs bg2", "stack_377", "bg2")]
    out = [(f"C1 drop {m}", "bg2", f"drop_{m}") for m in MEMBERS]
    out += [("C2 offset", "bg2", "s5_blend_no_offset"), ("C3 spread", "bg2", "a5_no_spread")]
    out += [(f"C4 {b}-{a}", b, a) for a, b in zip(STAIRCASE, STAIRCASE[1:], strict=False)]
    out += [(f"C5 only {m}", f"only_{m}", "bg2") for m in MEMBERS]
    out += [("C5 only W vs only M", "only_W", "only_M")]
    out += [
        ("C7 bg2 vs manual_heuristic", "bg2", "manual_heuristic"),
        ("C7 bg2 vs climatology", "bg2", "climatology"),
    ]
    return out


def main() -> None:
    """Print the tables."""
    tag = "stack" if "--stack" in sys.argv else "components"
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(260)
    wide = Wide(tag=tag)
    print(f"arms: {wide.arms}; shared half-hours: {wide.base.height:,}")
    if "--sign-checks" in sys.argv:
        print(
            "== Audit triggers 3 and 4: CRPS and MAE-of-median contrasts, point estimates, days 1 to 10 =="
        )
        for label, t, r in pairs(tag=tag):
            for gname, labels in list(GROUPS.items())[:3]:
                m = wide.mask(labels=labels, band=HEADLINE_BAND)
                d_crps = float((wide.metrics[t]["crps"][m] - wide.metrics[r]["crps"][m]).mean())
                d_mae = float(
                    (wide.metrics[t]["mae_median"][m] - wide.metrics[r]["mae_median"][m]).mean()
                )
                flag = (
                    "SIGN DISAGREES"
                    if d_crps * d_mae < 0 and max(abs(d_crps), abs(d_mae)) > 0.005
                    else ""
                )
                print(f"{label:28s} {gname:20s} crps {d_crps:+.3f} mae {d_mae:+.3f} {flag}")
        print("== Trigger 4: leave-one-out sign flips (CRPS point estimate) ==")
        for label, t, r in pairs(tag=tag):
            for gname in ("signed-MW", "MVA-only with D8"):
                labels = GROUPS[gname]
                full = float(
                    (
                        wide.metrics[t]["crps"][wide.mask(labels=labels, band=HEADLINE_BAND)]
                        - wide.metrics[r]["crps"][wide.mask(labels=labels, band=HEADLINE_BAND)]
                    ).mean()
                )
                for left_out in labels:
                    keep = tuple(x for x in labels if x != left_out)
                    m = wide.mask(labels=keep, band=HEADLINE_BAND)
                    d = float((wide.metrics[t]["crps"][m] - wide.metrics[r]["crps"][m]).mean())
                    if d * full < 0 and abs(full) > 0.005:
                        print(f"FLIP {label} {gname} without {left_out}: {full:+.3f} -> {d:+.3f}")
        return
    if "--check-only" in sys.argv:
        rows = pl.read_parquet(SCORES / f"rows_{tag}.parquet")
        table = pl.read_parquet(SCORES / f"slice_table_{tag}.parquet")
        t = rows.filter(pl.col("arm") == "bg2")
        r = rows.filter(pl.col("arm") == "drop_W")
        module = metrics.headline_month_bootstrap_contrast(
            treatment=t,
            reference=r,
            metric="crps",
            where=(pl.col("group") == "signed-MW") & pl.col(HEADLINE_BAND),
            slice_table=table,
        )
        m = wide.mask(labels=SIGNED, band=HEADLINE_BAND)
        mine = wide.contrast(treatment="bg2", reference="drop_W", mask=m, level=99.0, n=20_000)
        print("CHECK module", module, "mine", mine)
        return
    all_labels = tuple(sorted(wide.base["label"].unique().to_list()))
    print("\n== Mean CRPS (points), by group and band ==")
    for band in BANDS:
        for gname, labels in GROUPS.items():
            m = wide.mask(labels=labels, band=band)
            line = "  ".join(f"{a}={wide.metrics[a]['crps'][m].mean():.3f}" for a in wide.arms)
            print(f"{band:16s} {gname:28s} n={int(m.sum()):>9,}  {line}")
    print("\n== Mean CRPS (points) per Primary, days 1 to 10 and days 11 to 14 ==")
    for band in (HEADLINE_BAND, "lead_days_11_14"):
        print(band)
        for lab in all_labels:
            m = wide.mask(labels=(lab,), band=band)
            line = "  ".join(f"{a}={wide.metrics[a]['crps'][m].mean():.3f}" for a in wide.arms)
            print(f"  {lab:4s} n={int(m.sum()):>8,}  {line}")
    print("\n== Contrasts, treatment minus reference, CRPS; 99% (20,000) headline; (95%, 2,000) ==")
    for label, t, r in pairs(tag=tag):
        for band in (HEADLINE_BAND, "lead_days_8_10", "lead_days_11_14"):
            for gname, labels in GROUPS.items():
                m = wide.mask(labels=labels, band=band)
                d, lo, hi, n = wide.contrast(treatment=t, reference=r, mask=m, level=99.0, n=20_000)
                _, lo95, hi95, _ = wide.contrast(
                    treatment=t, reference=r, mask=m, level=95.0, n=2_000
                )
                print(
                    f"{label:28s} {band:16s} {gname:28s} {d:+.3f} 99% ({lo:+.3f} to {hi:+.3f}) "
                    f"95% ({lo95:+.3f} to {hi95:+.3f}) n={n:,}"
                )
    print("\n== Per-Primary contrasts (95%, 2,000), days 1 to 10 ==")
    for label, t, r in pairs(tag=tag):
        print(f"-- {label}")
        for lab in all_labels:
            m = wide.mask(labels=(lab,), band=HEADLINE_BAND)
            d, lo, hi, n = wide.contrast(treatment=t, reference=r, mask=m, level=95.0, n=2_000)
            print(f"  {lab:4s} {d:+.3f} ({lo:+.3f} to {hi:+.3f}) n={n:,}")
    print("\n== Leave-one-out (95%, 2,000), days 1 to 10 ==")
    for label, t, r in pairs(tag=tag):
        for gname in ("signed-MW", "MVA-only with D8"):
            labels = GROUPS[gname]
            for left_out in labels:
                keep = tuple(x for x in labels if x != left_out)
                m = wide.mask(labels=keep, band=HEADLINE_BAND)
                d, lo, hi, _ = wide.contrast(treatment=t, reference=r, mask=m, level=95.0, n=2_000)
                print(
                    f"{label:28s} {gname:18s} without {left_out:4s} {d:+.3f} ({lo:+.3f} to {hi:+.3f})"
                )
    print("\n== Tail measures, days 1 to 10, per group (mean over rows) ==")
    for gname, labels in GROUPS.items():
        m = wide.mask(labels=labels, band=HEADLINE_BAND)
        print(gname)
        for a in wide.arms:
            line = "  ".join(
                f"{c.replace('pinball_', 'pb').replace('coverage_', 'cov')}={wide.metrics[a][c][m].mean():.3f}"
                for c in TAILS
            )
            print(f"  {a:20s} {line}")
    print(
        "\n== Tail measures per Primary, days 1 to 10 (bg2, only_W, manual_heuristic, climatology) =="
    )
    for lab in all_labels:
        m = wide.mask(labels=(lab,), band=HEADLINE_BAND)
        print(lab)
        for a in [
            x
            for x in ("bg2", "only_W", "stack_377", "manual_heuristic", "climatology")
            if x in wide.arms
        ]:
            line = "  ".join(
                f"{c.replace('pinball_', 'pb').replace('coverage_', 'cov')}={wide.metrics[a][c][m].mean():.3f}"
                for c in TAILS
            )
            print(f"  {a:18s} {line}")
    print(
        "\n== Worst Primary per arm (CRPS days 1 to 10) and number of Primaries bg2 beats only_W =="
    )
    for a in wide.arms:
        per = {
            lab: wide.metrics[a]["crps"][wide.mask(labels=(lab,), band=HEADLINE_BAND)].mean()
            for lab in all_labels
        }
        worst = max(per, key=per.__getitem__)
        print(f"  {a:20s} worst {worst} {per[worst]:.3f}")


if __name__ == "__main__":
    main()

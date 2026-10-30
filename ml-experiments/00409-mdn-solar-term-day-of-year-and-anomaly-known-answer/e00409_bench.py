# ruff: noqa: E501
"""The least-squares read-out, the noise check, and the known-answer gates of experiment 409.

`dC_LS` is the capacity change (MW) that best explains an injected series from the ensemble unit
shape, on the fit's own rows and weights: `eff * sum(w * d * u) / sum(w * u**2)` over rows of the
as-of season-year with positive unit shape, where `d` is the injected series' effect on the target
(fraction of effective capacity) and `u` the unit shape. A day bootstrap gives its standard error.

```bash
PYTHONPATH=... uv run --no-sync python ml-experiments/00409-*/e00409_bench.py ls    # writes bench.json
PYTHONPATH=... uv run --no-sync python ml-experiments/00409-*/e00409_bench.py noise # before any gate
PYTHONPATH=... uv run --no-sync python ml-experiments/00409-*/e00409_bench.py gates # after the fits
```
"""

import argparse
import json
from typing import Final

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00409_prep as prep  # ty: ignore[unresolved-import]
import numpy as np

REFIT: Final[str] = "2026-06"
BENCH_PATH: Final = prep.WORK / "bench.json"
N_BOOT: Final[int] = 200
OLD_KAT: Final = (
    prep.ML_EXPERIMENTS_DATA_DIR
    / "00405-mdn-explicit-pv-term-learned-capacity-known-answer"
    / "work"
    / "kat"
    / "C2"
)
GATE_SITES: Final = pv.SIGNED_SIX


def ls_readout(
    *, d: np.ndarray, u: np.ndarray, w: np.ndarray, eff: float, day: np.ndarray, seed: int = 0
) -> tuple[float, float]:
    """Return `dC_LS` in MW and its day-bootstrap standard error."""

    def one(idx: np.ndarray) -> float:
        return eff * float(
            (w[idx] * d[idx] * u[idx]).sum() / max((w[idx] * u[idx] ** 2).sum(), 1e-12)
        )

    all_idx = np.arange(len(d))
    days, inverse = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(seed)
    by_day = [np.nonzero(inverse == i)[0] for i in range(len(days))]
    boots = [
        one(np.concatenate([by_day[i] for i in rng.integers(0, len(days), len(days))]))
        for _ in range(N_BOOT)
    ]
    return one(all_idx), float(np.std(boots))


def compute_ls() -> dict:
    """Return `dC_LS` for every (condition, shuffle) and Primary, from folds of the control and condition."""
    refit = base.refit_of(month=REFIT)
    table = base.primaries().filter(base.pl.col("site").is_in(list(GATE_SITES)))
    control = prep.build_fold(refit=refit, injection="control")
    out: dict = {}
    for cond in prep.CONDITIONS[1:]:
        fold = prep.build_fold(refit=refit, injection=cond)
        assert fold.y.shape == control.y.shape, "conditions must train on identical rows"
        for row in table.iter_rows(named=True):
            p, site = int(row["primary"]), row["site"]
            sel = (
                (fold.primary == p)
                & (fold.year_cell == fold.as_of_year_cell)
                & (fold.extras[:, 0] > 0)
                & ~fold.is_val
            )
            d = (control.y - fold.y)[sel] / 1.0
            est, se = ls_readout(
                d=d,
                u=fold.extras[sel, 0],
                w=fold.weight[sel],
                eff=float(row["effective_capacity_mw"]),
                day=fold.valid[sel] // prep.DAY_US,
            )
            out.setdefault(cond, {})[site] = {"ls_mw": est, "ls_se": se, "rows": int(sel.sum())}
        del fold
    out["placebo_shuffles"] = shuffle_benchmarks(control=control, table=table)
    return out


def shuffle_benchmarks(*, control: prep.Fold, table: base.pl.DataFrame) -> dict:
    """Return `dC_LS` of the shuffled farm series for several shuffles (the placebo benchmark)."""
    import e00405_prep as p5  # ty: ignore[unresolved-import]

    out: dict = {}
    for kind in ("profile", "anomaly"):
        series = p5.farm_norm_series(farm=prep.FARM)
        if kind == "anomaly":
            series = prep.split_profile_anomaly(series=series)[1]
        for salt in range(1, prep.N_BENCH_SHUFFLES + 1):
            for row in table.iter_rows(named=True):
                p, site = int(row["primary"]), row["site"]
                sel = (
                    (control.primary == p)
                    & (control.year_cell == control.as_of_year_cell)
                    & (control.extras[:, 0] > 0)
                    & ~control.is_val
                )
                f = prep.series_at(times_us=control.valid[sel], series=series, shuffle_salt=salt)
                inc = (
                    prep.SHARE
                    * p5.studies_capacity_2025(site=site)
                    / float(row["effective_capacity_mw"])
                )
                ok = ~np.isnan(f)
                d = f[ok] * inc
                est, _ = ls_readout(
                    d=d,
                    u=control.extras[sel, 0][ok],
                    w=control.weight[sel][ok],
                    eff=float(row["effective_capacity_mw"]),
                    day=control.valid[sel][ok] // prep.DAY_US,
                )
                out.setdefault(kind, {}).setdefault(site, []).append(est)
    return out


def noise_check(*, bench: dict) -> dict:
    """Return, per Primary, the expected SD of the held-anomaly recovery from experiment 405's seed spread."""
    out = {}
    for site in GATE_SITES:

        def cap(cond: str, s: int, site: str = site) -> float:
            r = json.loads((OLD_KAT / f"{cond}_s{s}.json").read_text())
            return r["capacity_mw"][site][r["as_of_year_cell"]]

        diffs = [cap("held", s) - cap("control", s) for s in range(3)]
        sd_mw = float(np.std(diffs, ddof=1))
        ls = bench["held_anomaly"][site]["ls_mw"]
        out[site] = {
            "seed_sd_mw": sd_mw,
            "ls_mw": ls,
            "expected_ratio_sd": sd_mw / abs(ls) if ls else None,
        }
    return out


def recovery(*, arm: str, cond: str, site: str, denominators: dict) -> tuple[float, float]:
    """Return a condition's seed-mean learned increment over the control, divided by `denominators`, and its SE."""
    d = prep.WORK / "kat" / arm

    def caps(c: str) -> np.ndarray:
        out = []
        for f in sorted(d.glob(f"{c}_s*.json")):
            r = json.loads(f.read_text())
            out.append(r["capacity_mw"][site][r["as_of_year_cell"]])
        return np.array(out)

    inc, ctl = caps(cond), caps("control")
    n = min(len(inc), len(ctl))
    diff = inc[:n] - ctl[:n] if n else np.array([np.nan])
    return float(diff.mean() / denominators[site]), float(
        diff.std(ddof=1) / np.sqrt(n) / abs(denominators[site])
    ) if n > 1 else float("nan")


def gates(*, bench: dict) -> dict:
    """Evaluate K0, K1, K2 for C2 and report K3 and the profile variant (descriptive)."""
    res: dict = {}
    for variant, held, placebo in (
        ("anomaly", "held_anomaly", "placebo_anomaly"),
        ("profile", "held_profile", "placebo_profile"),
    ):
        den = {s: bench[held][s]["ls_mw"] for s in GATE_SITES}
        shuffles = bench["placebo_shuffles"][variant]
        rows = {}
        for s in GATE_SITES:
            r_held, se_held = recovery(arm="C2", cond=held, site=s, denominators=den)
            r_pl, se_pl = recovery(arm="C2", cond=placebo, site=s, denominators=den)
            bench_pl = float(np.mean(shuffles[s]) / den[s])
            rows[s] = {
                "held": r_held,
                "held_se": se_held,
                "placebo": r_pl,
                "placebo_se": se_pl,
                "benchmark": bench_pl,
            }
        res[variant] = rows
    a = res["anomaly"]
    k1_band = [0.5 <= a[s]["held"] <= 1.5 for s in GATE_SITES]
    k1_sep = [
        (a[s]["held"] - a[s]["placebo"]) > 2 * float(np.hypot(a[s]["held_se"], a[s]["placebo_se"]))
        for s in GATE_SITES
    ]
    excess = np.array([a[s]["placebo"] - a[s]["benchmark"] for s in GATE_SITES])
    pos_a = {
        s: recovery(
            arm="C2", cond="pos_anomaly", site=s, denominators={s: bench["pos_anomaly"][s]["ls_mw"]}
        )[0]
        for s in GATE_SITES
    }
    pos_p = {
        s: recovery(
            arm="C2", cond="pos_profile", site=s, denominators={s: bench["pos_profile"][s]["ls_mw"]}
        )[0]
        for s in GATE_SITES
    }
    res["gates"] = {
        "K0_pos_anomaly_in_0.8_1.2_at_4": sum(0.8 <= v <= 1.2 for v in pos_a.values()) >= 4,
        "K0_pos_profile_at_most_0.30_at_4": sum(abs(v) <= 0.30 for v in pos_p.values()) >= 4,
        "K1_band_at_4": sum(k1_band) >= 4,
        "K1_separation_at_4": sum(k1_sep) >= 4,
        "K2_median_excess_le_0.20": bool(np.median(excess) <= 0.20),
        "K2_max_excess_le_0.20": bool(excess.max() <= 0.20),
        "pos_anomaly": pos_a,
        "pos_profile": pos_p,
    }
    return res


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ls", "noise", "gates"))
    args = parser.parse_args()
    if args.command == "ls":
        BENCH_PATH.write_text(json.dumps(compute_ls(), indent=1))
        print(f"wrote {BENCH_PATH}")
        return
    bench = json.loads(BENCH_PATH.read_text())
    result = noise_check(bench=bench) if args.command == "noise" else gates(bench=bench)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

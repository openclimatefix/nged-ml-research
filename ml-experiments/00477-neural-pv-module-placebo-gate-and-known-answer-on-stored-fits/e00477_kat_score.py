# ruff: noqa: E501
"""Read out the known-answer fits of experiment 477: the gate limit first, then the recoveries.

The equivalent capacity of a fit under an injected condition is the least-squares regression, over
the stored daylight rows of one Primary, of the fall in its median forecast (against the same seed
set's control) on the ensemble unit shape: `-sum(d * u) / sum(u * u)` in MW. Recovery is that
equivalent divided by the injected MW. `A0-doy` has no capacity, so only this read-out applies to it;
`C3` also has its learned capacity, whose increment over control is reported beside it.

```bash
. <the environment script that sets DATA_PATH_INTERNAL and PYTHONPATH>
uv run --no-sync python ml-experiments/00477-*/e00477_kat_score.py gate       # A0-doy only
uv run --no-sync python ml-experiments/00477-*/e00477_kat_score.py recovery   # needs the gate file
```
"""

import argparse
import json
from typing import Final

import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00477_inject as inj  # ty: ignore[unresolved-import]
import numpy as np

KAT: Final = inj.WORK / "kat"
GATE_PATH: Final = inj.WORK / "gate_limit.json"
RECOVERY_PATH: Final = inj.WORK / "recovery.json"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SIX: Final[tuple[str, ...]] = pv.SIGNED_SIX
REPORT_ONLY: Final[tuple[str, ...]] = ("D15", "D18")
GATE_MARGIN: Final[float] = 0.20


def table() -> dict[int, str]:
    """Return the primary index to site label for the signed Primaries."""
    import e00397_prep as base  # ty: ignore[unresolved-import]

    return {
        int(r["primary"]): r["site"]
        for r in base.primaries().iter_rows(named=True)
        if r["site"] in pv.SIGNED_SIX
    }


def load(*, arm: str, cond: str, seed: int) -> dict:
    """Return a fit's stored read-out arrays."""
    z = np.load(KAT / arm / f"{cond}_s{seed}.npz")
    return {k: z[k] for k in z.files}


def mean_control(*, arm: str) -> dict:
    """Return the seed-mean control read-out, after checking all seeds cover the same rows."""
    runs = [load(arm=arm, cond="control", seed=s) for s in SEEDS]
    for r in runs[1:]:
        assert np.array_equal(r["valid"], runs[0]["valid"])
        assert np.array_equal(r["primary"], runs[0]["primary"])
        assert np.array_equal(r["lead"], runs[0]["lead"])
    return {**runs[0], "median_mw": np.mean([r["median_mw"] for r in runs], axis=0)}


def equivalent(*, arm: str, cond: str, seed: int, control: dict) -> dict[str, float]:
    """Return the equivalent capacity (MW) by site for one fit."""
    run = load(arm=arm, cond=cond, seed=seed)
    assert np.array_equal(run["valid"], control["valid"])
    assert np.array_equal(run["primary"], control["primary"])
    assert np.array_equal(run["lead"], control["lead"])
    d = run["median_mw"].astype(np.float64) - control["median_mw"].astype(np.float64)
    u = run["unit_shape"].astype(np.float64)
    out = {}
    for p, site in table().items():
        m = run["primary"] == p
        out[site] = float(-(d[m] * u[m]).sum() / (u[m] ** 2).sum())
    return out


def least_squares_readout(*, cond: str, control: dict) -> dict[str, float]:
    """Return the least-squares read-out (MW) of the injected series on the unit shape, by site."""
    function, mw = inj.condition_function(condition=cond)
    assert function is not None
    f = np.nan_to_num(function(control["valid"]), nan=0.0) * mw
    u = control["unit_shape"].astype(np.float64)
    return {
        site: float(
            (f[control["primary"] == p] * u[control["primary"] == p]).sum()
            / (u[control["primary"] == p] ** 2).sum()
        )
        for p, site in table().items()
    }


def mw_of(*, cond: str) -> float:
    """Return a condition's injected megawatts."""
    return inj.condition_function(condition=cond)[1]


def gate() -> None:
    """Compute and store the gate limit from `A0-doy`'s placebo and profile-only equivalents."""
    control = mean_control(arm="A0-doy")
    limits: dict[str, dict] = {}
    per: dict[str, dict[str, list[float]]] = {}
    for cond in ("placebo_C_mw4", "profile_C_mw4"):
        per[cond] = {site: [] for site in table().values()}
        for s in SEEDS:
            for site, v in equivalent(arm="A0-doy", cond=cond, seed=s, control=control).items():
                per[cond][site].append(v / mw_of(cond=cond))
    for site in table().values():
        placebo = float(np.mean(per["placebo_C_mw4"][site]))
        profile = float(np.mean(per["profile_C_mw4"][site]))
        limits[site] = {
            "a0_placebo": placebo,
            "a0_profile_only": profile,
            "limit": max(placebo, profile),
        }
        print(
            f"{site}: A0-doy placebo {placebo:+.3f}, profile only {profile:+.3f}, limit {max(placebo, profile):+.3f}"
        )
    GATE_PATH.write_text(json.dumps(limits, indent=1))
    print(f"stored {GATE_PATH}")


def recovery() -> None:
    """Print and store the recovery of `C3` and `C3-nolag` against the stored gate limit."""
    assert GATE_PATH.exists(), "run the gate stage and commit its number before reading recoveries"
    limits = json.loads(GATE_PATH.read_text())
    out: dict = {}
    sites = table()
    for arm in ("C3", "C3-nolag", "A0-doy"):
        control = mean_control(arm=arm)
        cap_control = None
        if arm != "A0-doy":
            cap_control = [
                json.loads((KAT / arm / f"control_s{s}.json").read_text()) for s in SEEDS
            ]
        for cond in sorted(p.stem.rsplit("_s", 1)[0] for p in (KAT / arm).glob("*_s0.json")):
            if cond == "control":
                continue
            mw = mw_of(cond=cond)
            eq = {site: [] for site in sites.values()}
            dcap = {site: [] for site in sites.values()}
            for s in SEEDS:
                for site, v in equivalent(arm=arm, cond=cond, seed=s, control=control).items():
                    eq[site].append(v / mw)
                if cap_control is not None:
                    run = json.loads((KAT / arm / f"{cond}_s{s}.json").read_text())
                    cell = run["as_of_year_cell"]
                    for site in sites.values():
                        base_cap = np.mean([c["capacity_mw"][site][cell] for c in cap_control])
                        dcap[site].append((run["capacity_mw"][site][cell] - base_cap) / mw)
            held = cond.replace("placebo", "held").replace("profile", "held")
            ls = None
            if held in inj.CONDITIONS:
                ls = least_squares_readout(cond=held, control=control)
            for site in sites.values():
                key = f"{arm}|{cond}|{site}"
                out[key] = {
                    "equivalent_ratio_mean": float(np.mean(eq[site])),
                    "equivalent_ratio_sd": float(np.std(eq[site], ddof=1)),
                    "capacity_ratio_mean": float(np.mean(dcap[site])) if dcap[site] else None,
                    "capacity_ratio_sd": float(np.std(dcap[site], ddof=1)) if dcap[site] else None,
                    "least_squares_ratio": None if ls is None else ls[site] / mw,
                    "gate_limit": limits[site]["limit"],
                }
                r = out[key]
                print(
                    f"{key}: equivalent {r['equivalent_ratio_mean']:+.2f} +- {r['equivalent_ratio_sd']:.2f}"
                    + (f", capacity {r['capacity_ratio_mean']:+.2f}" if dcap[site] else "")
                    + (f", LS {r['least_squares_ratio']:.2f}" if ls else "")
                    + f", gate {limits[site]['limit']:+.2f}"
                )
    RECOVERY_PATH.write_text(json.dumps(out, indent=1))
    print(f"stored {RECOVERY_PATH}")


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("gate", "recovery"))
    args = parser.parse_args()
    gate() if args.stage == "gate" else recovery()


if __name__ == "__main__":
    main()

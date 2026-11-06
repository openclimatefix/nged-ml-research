# ruff: noqa: E501, D103, PLR0912, PLR0915
"""Experiment 600: print the pre-registered tests and the headline tables from the scored files.

Reads `data/ml-experiments/00600-*/<arm>/{bank_event_phase,event_free,real_event_phase,gate_check}`
for arms c0, c1, a1 and writes `report.txt` and `tests.csv` beside them. Usage: python e00600_report.py
"""

import json

import e00600_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

ARMS = ("c0", "c1", "a1")
LABEL = {"c0": "C0", "c1": "C1", "a1": "A1"}
N_BOOT = 20000
RNG = np.random.default_rng(600)
LONG, MID = "120-240 h", "24-120 h"
LONGEVENTS = pl.col("length_days") >= 15


def boot(*, vals: np.ndarray, level: float) -> tuple[float, float, float]:
    """Return the mean and the central `level` event-bootstrap interval."""
    v = vals[np.isfinite(vals)]
    m = v[RNG.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    a = (1 - level) / 2 * 100
    return float(v.mean()), float(np.percentile(m, a)), float(np.percentile(m, 100 - a))


BONF = 1 - 0.05 / k.N_COUNTED


def load(arm: str, name: str) -> pl.DataFrame:
    return pl.read_csv(k.WORK / arm / name)


def event_means(
    *, bank: pl.DataFrame, system: str, band: str, phase: str, group: str, metric: str
) -> pl.DataFrame:
    sub = bank.filter(
        (pl.col("system") == system)
        & (pl.col("band") == band)
        & (pl.col("phase") == phase)
        & LONGEVENTS
    )
    if group != "both groups":
        sub = sub.filter(pl.col("group") == group)
    return sub.select(
        "month", "pass", "primary", "sign", "depth", "length_days", pl.col(metric).alias(metric)
    )


def main() -> None:
    lines: list[str] = []
    tests: list[dict] = []
    out = lambda s="": lines.append(s)  # noqa: E731
    groups = ("both groups", *k.GROUPS_SCORED)
    for arm in ARMS:
        chk = json.loads((k.WORK / arm / "gate_check.json").read_text())
        out(f"## {LABEL[arm]}: gate state equals experiment 592's: {chk}")
    for arm in ("c1", "a1"):
        bank = load(arm, "bank_event_phase.csv")
        L = LABEL[arm]
        out(
            f"\n## {L}: injected 15 and 30 day events, q05-q95 coverage (mean over Primary-events, 95% event bootstrap)"
        )
        for band in (k.BANDS[0], MID, LONG):
            for phase in k.PHASES:
                for grp in groups:
                    for system in ("plain", "G4"):
                        e = event_means(
                            bank=bank,
                            system=system,
                            band=band,
                            phase=phase,
                            group=grp,
                            metric="cover90",
                        )
                        if e.height < 20:
                            continue
                        mean, lo, hi = boot(vals=e["cover90"].to_numpy(), level=0.95)
                        tests.append(
                            {
                                "arm": L,
                                "system": system,
                                "band": band,
                                "phase": phase,
                                "group": grp,
                                "n_events": e.height,
                                "cover90": mean,
                                "lo": lo,
                                "hi": hi,
                            }
                        )
        # P1: coverage day 4 to end at 120-240 h
        e = event_means(
            bank=bank,
            system="G4",
            band=LONG,
            phase="days4-end",
            group="both groups",
            metric="cover90",
        )
        mean, lo, hi = boot(vals=e["cover90"].to_numpy(), level=0.95)
        _, blo, bhi = boot(vals=e["cover90"].to_numpy(), level=BONF)
        out(
            f"P1 {L}+G4 day 4 to end at {LONG}: {mean:.3f} (95% {lo:.3f} to {hi:.3f}; Bonferroni {blo:.3f} to {bhi:.3f}; MDE {(bhi - blo) / 2:.3f}), n={e.height}; holds={mean >= 0.90}"
        )
        e = event_means(
            bank=bank,
            system="G4",
            band=MID,
            phase="days4-end",
            group="both groups",
            metric="cover90",
        )
        out(
            f"P4 {L}+G4 day 4 to end at {MID}: {e['cover90'].mean():.4f}, n={e.height} (experiment 597: {'0.9132' if arm == 'c1' else '0.925'})"
        )
        # P2: gate value (G4 minus plain), per phase, long minus mid
        for phase in ("after0-7", "days4-end"):
            gains = {}
            for band in (MID, LONG):
                g = event_means(
                    bank=bank,
                    system="G4",
                    band=band,
                    phase=phase,
                    group="both groups",
                    metric="cover90",
                ).rename({"cover90": "g4"})
                p = event_means(
                    bank=bank,
                    system="plain",
                    band=band,
                    phase=phase,
                    group="both groups",
                    metric="cover90",
                ).rename({"cover90": "plain"})
                keys = ["month", "pass", "primary", "sign", "depth", "length_days"]
                gains[band] = (
                    g.join(p, on=keys)
                    .with_columns(gain=pl.col("g4") - pl.col("plain"))
                    .select(*keys, "gain")
                )
            j = gains[LONG].join(
                gains[MID],
                on=["month", "pass", "primary", "sign", "depth", "length_days"],
                suffix="_mid",
            )
            d = (j["gain"] - j["gain_mid"]).to_numpy()
            mean, lo, hi = boot(vals=d, level=0.95)
            _, blo, bhi = boot(vals=d, level=BONF)
            gl, gm = float(j["gain"].mean()), float(j["gain_mid"].mean())  # ty: ignore[invalid-argument-type]
            out(
                f"P2 {L} gate value, {phase}: {MID} {gm:+.3f}, {LONG} {gl:+.3f}, difference {mean:+.3f} (95% {lo:+.3f} to {hi:+.3f}; Bonferroni {blo:+.3f} to {bhi:+.3f}; MDE {(bhi - blo) / 2:.3f}), n={j.height}; holds={mean >= 0}; refuted={bhi < 0}"
            )
        # follow ratio and safe side, day 4 to end, long
        for system in ("plain", "G4"):
            for band in (MID, LONG):
                for phase in ("days4-end", "after0-7"):
                    f = event_means(
                        bank=bank,
                        system=system,
                        band=band,
                        phase=phase,
                        group="both groups",
                        metric="follow_q50",
                    )
                    s = event_means(
                        bank=bank,
                        system=system,
                        band=band,
                        phase=phase,
                        group="both groups",
                        metric="nra_side95",
                    )
                    out(
                        f"{L} {system} {band} {phase}: follow {f['follow_q50'].mean():.3f}, safe-side {s['nra_side95'].mean():.3f}"
                    )
        # tail pinball
        for system in ("plain", "G4"):
            for band in (MID, LONG):
                r = []
                for m in ("pin01", "pin05", "pin95", "pin99"):
                    e = event_means(
                        bank=bank,
                        system=system,
                        band=band,
                        phase="days4-end",
                        group="both groups",
                        metric=m,
                    )
                    r.append(f"{m} {e[m].mean() * 100:.2f}")  # ty: ignore[unsupported-operator]
                out(
                    f"{L} {system} {band} day 4 to end tail pinball (points of capacity): "
                    + ", ".join(r)
                )
    out(
        "\n## Why the gate adds less at long lead: share of rows whose forecast was issued while the G4 wide band was on, injected 15 and 30 day events"
    )
    for arm in ("c1", "a1"):
        bank = load(arm, "bank_event_phase.csv").filter(LONGEVENTS & (pl.col("system") == "G4"))
        for phase in ("days4-end", "after0-7"):
            r = [
                f"{b} {bank.filter((pl.col('band') == b) & (pl.col('phase') == phase))['share_origin_days_wide'].mean():.3f}"
                for b in (k.BANDS[0], MID, LONG)
            ]
            out(f"{LABEL[arm]} {phase}: " + ", ".join(r))
    out(
        "\n## Leave one Primary out: P1 coverage (G4, day 4 to end, 120-240 h) and P2 difference of gains (after0-7)"
    )
    for arm in ("c1", "a1"):
        bank = load(arm, "bank_event_phase.csv")
        for prim in sorted(bank["primary"].unique().to_list()):
            b = bank.filter(pl.col("primary") != prim)
            e = event_means(
                bank=b,
                system="G4",
                band=LONG,
                phase="days4-end",
                group="both groups",
                metric="cover90",
            )
            keys = ["month", "pass", "primary", "sign", "depth", "length_days"]
            gs = {}
            for band in (MID, LONG):
                g = event_means(
                    bank=b,
                    system="G4",
                    band=band,
                    phase="after0-7",
                    group="both groups",
                    metric="cover90",
                ).rename({"cover90": "g4"})
                q = event_means(
                    bank=b,
                    system="plain",
                    band=band,
                    phase="after0-7",
                    group="both groups",
                    metric="cover90",
                ).rename({"cover90": "plain"})
                gs[band] = (
                    g.join(q, on=keys)
                    .with_columns(gain=pl.col("g4") - pl.col("plain"))
                    .select(*keys, "gain")
                )
            j = gs[LONG].join(gs[MID], on=keys, suffix="_mid")
            out(
                f"{LABEL[arm]} without Primary {prim}: P1 {e['cover90'].mean():.3f}; P2 difference {(j['gain'] - j['gain_mid']).mean():+.4f}"
            )
    ef = pl.concat(
        [load(a, "event_free.csv").with_columns(arm=pl.lit(LABEL[a])) for a in ARMS],
        how="diagonal_relaxed",
    )
    out("\n## Event-free rows: CRPS (points of capacity) and coverage by arm, system, band, group")
    for row in (
        ef.filter(pl.col("group").is_in(list(k.GROUPS_SCORED)))
        .sort("arm", "group", "band", "system")
        .iter_rows(named=True)
    ):
        out(
            f"{row['arm']} {row['system']} {row['group']} {row['band']}: n={row['n_rows']}, CRPS {row['crps']:.3f}, cover90 {row['cover90']:.3f}, pin05 {row['pin05']:.3f}, pin95 {row['pin95']:.3f}"
        )
    out(
        "\n## Event-free P3: assembled minus plain pinball (points of capacity, Bonferroni interval, MDE), both groups"
    )
    for row in (
        ef.filter((pl.col("group") == "both groups") & (pl.col("system") == "G4"))
        .sort("arm", "band")
        .iter_rows(named=True)
    ):
        for m in ("pin05", "pin95", "crps"):
            out(
                f"{row['arm']} {row['band']} {m}: {row[m + '_minus_plain']:+.4f} ({row[m + '_lo']:+.4f} to {row[m + '_hi']:+.4f}; MDE {row[m + '_mde']:.4f}); counted={row['counted']}"
            )
    pl.DataFrame(tests).write_csv(k.WORK / "tests.csv")
    (k.WORK / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

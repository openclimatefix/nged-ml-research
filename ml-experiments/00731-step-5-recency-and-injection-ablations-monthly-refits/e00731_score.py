# ruff: noqa: PLR0915, PLR0912, E501, PERF401, PT018
"""Experiment 731: score the new arms and read the pre-registered contrasts (private CSV, CPU only).

README. This script runs the scoring step of experiment 731. It relies on experiment 677
(`e00677_score.main`, the per-arm scorer; `e00677_slices.block_bootstrap`, the 7-day block
bootstrap; `e00677_export`'s conventions), 667 (`e00667_export.contract_rows`), 668
(`e00668_slices_primaries.event_sets` and `masks`, the load-shift scenarios), and
`studies.eval_protocol.slices` (the Primary groups). PYTHONPATH: source `ml-experiments/00524*/env.sh`
from the worktree root, and set `DATA_PATH_INTERNAL`. Usage: `python e00731_score.py [--stage
export|rows|contrasts|all] [--arms R0 W0 ...] [--continue-after-gate-failure]`.

Stages:

- `export`: turn each arm's fits into `backtests/arm_<ARM>_sp1.parquet` (the layout of 677's
  `e00677_export`), and link the stored arms 9a and `5wo` read only as `stored_9a_sp1` and
  `stored_5wo_sp1`, so they can be scored per seed.
- `rows`: score each arm with 677's per-arm scorer into the 731 folder (`scores/arm_<ARM>_sp1_s012`,
  and one folder per seed `_s0`, `_s1`, `_s2`). 677's scores folder is only read.
- `contrasts`: pass A, pass B, the contrasts, the bank follow, and the median minus mean.

**Pass A** is 677's joint pass over its 31 systems, unchanged: the rows on which every one of the
29 stored ladder systems (the 5 early steps, and steps 5 to 8 with and without injected events, each
quarterly, quarterly with seeds 0 to 2, and monthly), the manual heuristic, and climatology have a
scored row. It must reproduce step5w@1 = 7.3946 (95% interval 7.1369 to 7.6713) on 3,879,313 rows
with 2,000 draws and seed 0, or the script stops. **Pass B** adds the new arms to the same
intersection, asserts that the row count and step5w@1's mean are unchanged, and reads the contrasts.

Contrasts (arm minus reference, in points of capacity; reference R0, and the stored arm 9a as the
secondary reference): C1 W0-R0, C2 W30-R0, C3 W270-R0, C4 5wo-R0, C5 B0-R0, C6 WP-R0, C7 IP-R0, C8
IP-5wo, and WP-W0. Each is read for the normalised MAE of the median and for CRPS, with a paired
7-day-block bootstrap of origin days (10,000 draws, one set of draws for every contrast, so the
contrasts are paired), 99.5% intervals, for the six signed-MW Primaries, the eight MVA-only
Primaries, the eight without D8, each Primary, the 16 together, and each seed alone (seed s of the
arm minus seed s of the reference); and leave-one-Primary-out for the six, the eight, and the 16.
The scenarios of experiment 677's slice code (all rows, outside events and 7 days after one,
inside events, the 7 days after an event ends) are read for the two event sets that have rows.

The sums are linear, so the script keeps for each system only a 16 x (number of 7-day blocks) table
of per-Primary sums of each loss, one arm at a time. **Expected peak memory** (not measured): about 6
GB inside 677's scorer for one arm (11.6 million forecast rows with 14 columns, plus a group-by),
and under 2 GB in the contrast stage (five columns of 3.9 million rows for one arm at a time).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SHORT_HISTORY_LABELS, SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME_731: Final = "00731-step-5-recency-and-injection-ablations-monthly-refits"
NAME_677: Final = "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
ROOT_731: Final[Path] = Path(os.environ.get("R731", ML_EXPERIMENTS_DATA_DIR / NAME_731))
ROOT_677: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME_677
assert "00731" in str(ROOT_731), ROOT_731
assert "00731" not in str(ROOT_677), ROOT_677
os.environ["E00677_ROOT"] = str(ROOT_731)  # 677's scorer then writes under this experiment only

import e00667_export as x667  # noqa: E402  # ty: ignore[unresolved-import]
import e00668_slices_primaries as s668  # noqa: E402  # ty: ignore[unresolved-import]
import e00677_score as sc  # noqa: E402  # ty: ignore[unresolved-import]
import e00677_slices as s677  # noqa: E402  # ty: ignore[unresolved-import]

assert sc.ROOT == ROOT_731, (sc.ROOT, ROOT_731)
S_LIB: Final = sc.lib
ARMS: Final = ("R0", "W0", "W30", "W270", "WP", "IP", "B0")
MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
)  # fmt: skip
SEEDS: Final = (0, 1, 2)
STORED: Final = {"9a": "step5w_with_injected_sp1", "5wo": "step5wo_without_injected_sp1"}
"""The stored monthly arms: arm 9a and the injection-free arm, in 677's backtest file names."""
N_BOOT: Final = 10_000
LEVEL: Final = 0.995
BLOCK_DAYS: Final = 7
DAY: Final = 48
EXPECTED_ROWS: Final = 3_879_313
EXPECTED_9A: Final = (7.3946, 7.1369, 7.6713)
"""Pass A's mean, 95% lower, and 95% upper limit of step5w@1 on the all-row MAE."""
GATE_TOLERANCE: Final = 0.05
CONTRASTS: Final = (
    ("C1", "W0", "R0"),
    ("C2", "W30", "R0"),
    ("C3", "W270", "R0"),
    ("C4", "5wo", "R0"),
    ("C5", "B0", "R0"),
    ("C6", "WP", "R0"),
    ("C7", "IP", "R0"),
    ("C8", "IP", "5wo"),
    ("C9", "WP", "W0"),
)
"""(name, arm, reference). C1 to C5 are the family; C6 to C8 are placebos and the control; C9 is extra."""
SECONDARY: Final = tuple((f"{a} vs 9a", a, "9a") for a in (*ARMS, "5wo"))
SETTINGS: Final = ("joint", "s0", "s1", "s2")
SCENARIO_SETS: Final = ("event table, over 7 days, 2 units or more", "verified transfer")


# ----------------------------------------------------------------------------- stage: export


def arm_backtest(*, arm: str) -> Path:
    """Write the arm's contract rows for the 12 months and 3 seeds, as 677's export does."""
    target = ROOT_731 / "backtests" / f"arm_{arm}_sp1.parquet"
    if target.exists():
        return target
    parts, keys = [], {}
    for month in MONTHS:
        for seed in SEEDS:
            path = (
                ROOT_731
                / "arms"
                / f"k1_{arm}_sp1"
                / "work"
                / "fits"
                / "P5"
                / f"{month}_d0_s{seed}.parquet"
            )
            raw = pl.read_parquet(path, columns=[*x667.KEYS, *x667.QCOLS])
            ordered = raw.sort(x667.KEYS)
            keys.setdefault(month, ordered.select(x667.KEYS))
            assert ordered.select(x667.KEYS).equals(keys[month]), ("seeds differ", arm, month, seed)
            assert raw.height == raw.unique(x667.KEYS).height, ("duplicate keys", arm, month, seed)
            parts.append(x667.contract_rows(mean=ordered).with_columns(seed=pl.lit(seed, pl.Int8)))
    frame = pl.concat(parts).with_columns(
        step=pl.lit(f"arm_{arm}"), version=pl.lit("731"), NRA_or_prevailing=pl.lit("NRA")
    )
    assert frame.height == frame.unique(["seed", *x667.KEYS]).height, "duplicate keys across months"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(target)
    print("exported", target.name, frame.height, "rows", flush=True)
    return target


def link_stored() -> None:
    """Link the stored arms' backtest files read only into this experiment's folder."""
    for short, stem in STORED.items():
        link = ROOT_731 / "backtests" / f"stored_{short}_sp1.parquet"
        source = ROOT_677 / "backtests" / f"{stem}.parquet"
        assert source.exists(), source
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(source)


def score_names(*, arms: list[str]) -> list[str]:
    """Return the names of the backtest files to score: the new arms and the two stored arms."""
    return [f"arm_{a}_sp1" for a in arms] + [f"stored_{s}_sp1" for s in STORED]


def stage_rows(*, arms: list[str]) -> None:
    """Score every arm for the three seeds together and for each seed alone, skipping done work."""
    for arm in arms:
        arm_backtest(arm=arm)
    link_stored()
    for name in score_names(arms=arms):
        for seeds in (SEEDS, *[(s,) for s in SEEDS]):
            out = ROOT_731 / "scores" / f"{name}_s{''.join(map(str, seeds))}" / "rows.parquet"
            if out.exists():
                continue
            sc.main(name, None, seeds)


# ----------------------------------------------------------------------------- stage: contrasts


def rows_path(*, system: str, setting: str) -> Path:
    """Return the per-row score file of one system in one setting (joint or one seed)."""
    tag = "s012" if setting == "joint" else setting
    if system in ARMS:
        return ROOT_731 / "scores" / f"arm_{system}_sp1_{tag}" / "rows.parquet"
    return ROOT_731 / "scores" / f"stored_{system}_sp1_{tag}" / "rows.parquet"


def rows_path_677(*, system: str) -> Path:
    """Return 677's own stored per-row score file of arm 9a or `5wo` (read only; used by pass A and the check)."""
    return ROOT_677 / "scores" / f"{STORED[system]}_s012" / "rows.parquet"


def ladder_systems() -> list[tuple[str, Path]]:
    """Return the 29 stored ladder systems of 677's joint pass (the other two are the heuristic and climatology)."""
    out = []
    for step in ("1", "2", "3a", "3b", "4"):
        out.append((f"step{step}", ROOT_677 / "scores" / f"step{step}_{_version(step)}_sp3_s012"))
    for n in "5678":
        for tail in ("w", "wo"):
            step = f"{n}{tail}"
            base = f"step{step}_{_version(step)}"
            out.append((f"step{step}", ROOT_677 / "scores" / f"{base}_sp3_s01234"))
            out.append((f"step{step}!", ROOT_677 / "scores" / f"{base}_sp3_s012"))
            out.append((f"step{step}@1", ROOT_677 / "scores" / f"{base}_sp1_s012"))
    assert len(out) == 29, len(out)
    return [(n, p / "rows.parquet") for n, p in out]


def _version(step: str) -> str:
    import e00677_export as exp  # ty: ignore[unresolved-import]

    return exp.VERSION[step]


def key_of(*, tsid: np.ndarray, origin: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pack (series, origin index, valid index) into one sortable int64."""
    assert origin.max() < (1 << 20) and valid.max() < (1 << 20)
    return (tsid.astype(np.int64) << 40) | (origin.astype(np.int64) << 20) | valid.astype(np.int64)


def ok_keys(*, path: Path) -> np.ndarray:
    """Return the sorted keys of the rows that a score file marks as scored."""
    f = pl.read_parquet(path, columns=["time_series_id", "origin_idx", "valid_idx", "ok"])
    f = f.filter(pl.col("ok"))
    k = key_of(
        tsid=f["time_series_id"].to_numpy(),
        origin=f["origin_idx"].to_numpy(),
        valid=f["valid_idx"].to_numpy(),
    )
    return np.sort(k)


def shared_keys(*, paths: list[Path]) -> np.ndarray:
    """Return the keys scored in every file (677's joint pass)."""
    keys = ok_keys(path=paths[0])
    for p in paths[1:]:
        keys = np.intersect1d(keys, ok_keys(path=p), assume_unique=True)
    return keys


def values_on(*, path: Path, keys: np.ndarray, columns: dict[str, str]) -> dict[str, np.ndarray]:
    """Return the named score columns of a file on `keys`, in the order of `keys` (all must be scored)."""
    f = pl.read_parquet(
        path, columns=["time_series_id", "origin_idx", "valid_idx", "ok", *columns.values()]
    )
    k = key_of(
        tsid=f["time_series_id"].to_numpy(),
        origin=f["origin_idx"].to_numpy(),
        valid=f["valid_idx"].to_numpy(),
    )
    order = np.argsort(k, kind="stable")
    pos = np.searchsorted(k[order], keys)
    assert (pos < len(k)).all() and (k[order][np.minimum(pos, len(k) - 1)] == keys).all(), (
        "the file lacks some shared rows",
        path,
    )
    take = order[pos]
    assert f["ok"].to_numpy()[take].all(), ("a shared row is not scored", path)
    return {n: f[c].to_numpy()[take].astype(np.float64) for n, c in columns.items()}


class Frame:
    """The shared rows' index arrays, block ids, and scenario masks, built once."""

    def __init__(self, *, keys: np.ndarray, site: pl.DataFrame) -> None:
        """Decode the keys, number the 7-day blocks, and build the load-shift scenario masks."""
        self.labels = site["label"].to_list()
        primary_of_tsid = dict(
            zip(site["time_series_id"].to_list(), range(len(self.labels)), strict=True)
        )
        # the site map is ordered by Primary index; map each time-series id to its Primary index
        self.prim_of_tsid = np.full(128, -1, dtype=np.int64)
        for tsid, i in primary_of_tsid.items():
            self.prim_of_tsid[int(tsid)] = i
        tsid = keys >> 40
        self.origin = (keys >> 20) & 0xFFFFF
        self.valid = keys & 0xFFFFF
        self.prim = self.prim_of_tsid[tsid]
        assert (self.prim >= 0).all()
        day = self.origin // DAY
        self.block_ids, inverse = np.unique(day // BLOCK_DAYS, return_inverse=True)
        self.nb = len(self.block_ids)
        self.blk = inverse.astype(np.int64)
        self.n_origin_days = len(np.unique(day))
        self.counts = self.block_sums(values=np.ones(len(keys)))
        label_arr = np.asarray(self.labels)[self.prim]
        rows = pl.DataFrame({"label": label_arr, "valid_idx": self.valid})
        sets = s668.event_sets(site=site)
        self.scenarios: dict[str, np.ndarray] = {"all rows": np.ones(len(keys), dtype=bool)}
        for name in SCENARIO_SETS:
            for part, mask in s668.masks(rows=rows, events=sets[name]).items():
                self.scenarios[f"{name} | {part}"] = mask
        print(
            "event sets:",
            {k: v.height for k, v in sets.items()},
            "scenario rows:",
            {k: int(m.sum()) for k, m in self.scenarios.items()},
            flush=True,
        )

    def block_sums(self, *, values: np.ndarray) -> np.ndarray:
        """Return the per-Primary, per-block sums of `values`, shape (16, blocks)."""
        n = len(self.labels)
        return np.bincount(
            self.prim * self.nb + self.blk, weights=values, minlength=n * self.nb
        ).reshape(n, self.nb)

    def scenario_sums(self, *, values: np.ndarray) -> dict[str, np.ndarray]:
        """Return the per-block sums of `values` on each scenario's rows."""
        return {
            k: np.bincount(self.blk[m], weights=values[m], minlength=self.nb)
            for k, m in self.scenarios.items()
        }


class Sums:
    """Per-Primary and per-scenario block sums of one system's MAE and CRPS in each setting."""

    def __init__(self) -> None:
        """Start empty; `add` fills one setting."""
        self.prim: dict[tuple[str, str], np.ndarray] = {}
        self.scn: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def add(self, *, frame: Frame, setting: str, mae: np.ndarray, crps: np.ndarray) -> None:
        """Store the sums of one setting."""
        for metric, v in (("mae", mae), ("crps", crps)):
            self.prim[(metric, setting)] = frame.block_sums(values=v)
            self.scn[(metric, setting)] = frame.scenario_sums(values=v)


def groups(*, labels: list[str]) -> dict[str, list[int]]:
    """Return each reported group as the Primary indices it contains."""
    idx = {lab: i for i, lab in enumerate(labels)}
    six = [idx[x] for x in SIGNED_MW_LABELS]
    eight = [idx[x] for x in MVA_ONLY_LABELS]
    out = {
        "six signed-MW": six,
        "eight MVA-only": eight,
        "eight without D8": [i for i in eight if labels[i] != "D8"],
        "all 16": list(range(len(labels))),
    }
    for lab in labels:
        out[f"Primary {lab}" + (" (reported only)" if lab in SHORT_HISTORY_LABELS else "")] = [
            idx[lab]
        ]
    for base in ("six signed-MW", "eight MVA-only", "all 16"):
        for i in out[base]:
            out[f"{base} minus {labels[i]}"] = [j for j in out[base] if j != i]
    return out


def interval(
    *, diff: np.ndarray, count: np.ndarray, pick: np.ndarray
) -> tuple[float, float, float]:
    """Return the pooled mean of the paired difference and its 99.5% block-bootstrap interval.

    Args:
        diff: Per-block sums of the paired difference.
        count: Per-block row counts.
        pick: Bootstrap draws of block indices, shape (draws, blocks).

    Returns:
        The pooled mean, the lower limit, and the upper limit, or three NaNs when no rows exist.
    """
    if count.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    draws = diff[pick].sum(axis=1) / np.maximum(count[pick].sum(axis=1), 1)
    lo, hi = np.quantile(draws, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    return float(diff.sum() / count.sum()), float(lo), float(hi)


def stage_contrasts(*, arms: list[str], continue_after_gate_failure: bool) -> None:
    """Run pass A, pass B, the contrasts, the bank follow, and the median minus mean."""
    out = ROOT_731 / "scores" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    site = pl.read_csv(next(s668.R.glob("00521-*/site_map.csv"))).sort("primary")
    # ---------------- pass A: 677's joint pass, unchanged
    ladder = ladder_systems()
    paths_a = [p for _, p in ladder]
    keys_a = shared_keys(paths=paths_a)
    print("PASS A: shared rows of 677's 31 systems:", len(keys_a), flush=True)
    assert len(keys_a) == EXPECTED_ROWS, (len(keys_a), EXPECTED_ROWS)
    frame = Frame(keys=keys_a, site=site)
    ref9a = values_on(
        path=rows_path_677(system="9a"),
        keys=keys_a,
        columns={"mae": "Model_mae", "crps": "Model_crps"},
    )
    day = ((keys_a >> 20) & 0xFFFFF) // DAY
    draws = s677.block_bootstrap(day=day, values=ref9a["mae"])
    mean_a = float(ref9a["mae"].mean())
    lo_a, hi_a = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    print(
        f"PASS A step5w@1 MAE {mean_a:.4f} (95% {lo_a:.4f} to {hi_a:.4f}) on {len(keys_a):,} rows"
    )
    for got, want in zip((mean_a, lo_a, hi_a), EXPECTED_9A, strict=True):
        assert abs(got - want) < 5e-5 + 1e-9, (
            "reproduction gate: pass A differs from 677",
            got,
            want,
        )
    print("REPRODUCTION GATE (pass A): passed", flush=True)
    # ---------------- pass B: the new arms join the same intersection
    # The scoring inputs (the cleaned grid) changed after experiment 677 ran: the rows below that 677
    # scored and this script's scorer does not, or the reverse, are counted and printed. Pass B uses
    # the rows that both scorings share, so every system in a contrast is scored by one scorer.
    keys_b = keys_a
    for system in (*arms, "9a", "5wo"):
        keys_b = keys_b[
            np.isin(
                keys_b, ok_keys(path=rows_path(system=system, setting="joint")), assume_unique=True
            )
        ]
    print(
        f"PASS B: {len(keys_b):,} shared rows of {len(keys_a):,} in pass A "
        f"({len(keys_a) - len(keys_b):,} rows are scored by experiment 677's scorer and not by this one)",
        flush=True,
    )
    assert len(keys_b) > 0.99 * len(keys_a), "pass B lost more than 1% of pass A's rows"
    keys_a = keys_b
    frame = Frame(keys=keys_a, site=site)
    N_B = len(keys_b)
    systems: dict[str, Sums] = {}
    for system in (*arms, "9a", "5wo"):
        sums = Sums()
        for setting in SETTINGS:
            v = values_on(
                path=rows_path(system=system, setting=setting),
                keys=keys_a,
                columns={"mae": "Model_mae", "crps": "Model_crps"},
            )
            sums.add(frame=frame, setting=setting, mae=v["mae"], crps=v["crps"])
        systems[system] = sums
        if system in STORED:  # the rescoring must reproduce 677's stored per-row scores
            mine = values_on(
                path=rows_path(system=system, setting="joint"),
                keys=keys_a,
                columns={"mae": "Model_mae", "crps": "Model_crps"},
            )
            theirs = values_on(
                path=rows_path_677(system=system),
                keys=keys_a,
                columns={"mae": "Model_mae", "crps": "Model_crps"},
            )
            for metric in ("mae", "crps"):
                worst = float(np.abs(mine[metric] - theirs[metric]).max())
                assert worst < 1e-4, (
                    "rescored stored arm differs from 677's rows",
                    system,
                    metric,
                    worst,
                )
            print(
                f"rescored {system} equals 677's stored per-row scores on the shared rows (checked MAE and CRPS)"
            )
        print("summed", system, flush=True)
    print(
        f"stored arm 9a rescored on pass B's {N_B:,} rows: MAE "
        f"{float(systems['9a'].prim[('mae', 'joint')].sum() / N_B):.4f} "
        f"(677's pass A value {EXPECTED_9A[0]})"
    )
    assert frame.counts.sum() == N_B
    heur = values_on(
        path=ROOT_677
        / "scores"
        / "step1_ens_only_sp3_s012"
        / "rows.parquet",  # as in 677's joint pass
        keys=keys_a,
        columns={
            "heur_mae": "Manual heuristic_mae",
            "clim_mae": "Climatology_mae",
            "heur_crps": "Manual heuristic_crps",
            "clim_crps": "Climatology_crps",
        },
    )
    for sysname in ("heur", "clim"):
        sums = Sums()
        sums.add(
            frame=frame, setting="joint", mae=heur[f"{sysname}_mae"], crps=heur[f"{sysname}_crps"]
        )
        systems[sysname] = sums
    # ---------------- reproduction gate of R0 against arm 9a
    if "R0" in systems:
        r0 = systems["R0"].prim[("mae", "joint")].sum() / N_B
        n9 = systems["9a"].prim[("mae", "joint")].sum() / N_B
        print(
            f"R0 seed-mean MAE {r0:.4f}, arm 9a {n9:.4f}, difference {r0 - n9:+.4f} (gate {GATE_TOLERANCE})"
        )
        if abs(r0 - n9) > GATE_TOLERANCE:
            print("REPRODUCTION GATE (R0): FAILED. The month-by-month comparison follows.")
            month_table(keys=keys_a, frame=frame, systems=("R0", "9a"))
            if not continue_after_gate_failure:
                sys.exit(3)
        else:
            print("REPRODUCTION GATE (R0): passed", flush=True)
    # ---------------- contrasts
    rng = np.random.default_rng(0)
    pick = rng.integers(0, frame.nb, size=(N_BOOT, frame.nb))
    grp = groups(labels=frame.labels)
    records = []
    mean_records = []
    for system, sums in systems.items():
        for metric in ("mae", "crps"):
            for setting in SETTINGS:
                if (metric, setting) not in sums.prim:
                    continue
                for g, members in grp.items():
                    s = sums.prim[(metric, setting)][members].sum(axis=0)
                    c = frame.counts[members].sum(axis=0)
                    mean_records.append(
                        {"system": system, "group": g, "metric": metric, "setting": setting, "mean": float(s.sum() / c.sum()) if c.sum() else float("nan"), "rows": int(c.sum())}
                    )  # fmt: skip
    for kind, table in (("primary", CONTRASTS), ("vs 9a", SECONDARY)):
        for cname, a, b in table:
            if a not in systems or b not in systems:
                continue
            for metric in ("mae", "crps"):
                for setting in SETTINGS:
                    if (metric, setting) not in systems[a].prim or (metric, setting) not in systems[
                        b
                    ].prim:
                        continue
                    d_all = systems[a].prim[(metric, setting)] - systems[b].prim[(metric, setting)]
                    for g, members in grp.items():
                        d = d_all[members].sum(axis=0)
                        c = frame.counts[members].sum(axis=0)
                        m, lo, hi = interval(diff=d, count=c, pick=pick)
                        records.append(
                            {"kind": kind, "contrast": cname, "arm": a, "reference": b, "group": g, "metric": metric, "setting": setting, "diff": m, "lo": lo, "hi": hi, "rows": int(c.sum())}
                        )  # fmt: skip
                    if setting != "joint":
                        continue
                    ds = {
                        k: systems[a].scn[(metric, setting)][k]
                        - systems[b].scn[(metric, setting)][k]
                        for k in frame.scenarios
                    }
                    for k, d in ds.items():
                        c = np.bincount(frame.blk[frame.scenarios[k]], minlength=frame.nb).astype(
                            float
                        )
                        m, lo, hi = interval(diff=d, count=c, pick=pick)
                        records.append(
                            {"kind": kind, "contrast": cname, "arm": a, "reference": b, "group": f"scenario: {k}", "metric": metric, "setting": setting, "diff": m, "lo": lo, "hi": hi, "rows": int(c.sum())}
                        )  # fmt: skip
    contrasts = pl.DataFrame(records)
    contrasts.write_csv(out / "contrasts.csv")
    pl.DataFrame(mean_records).write_csv(out / "system_means.csv")
    # ---------------- bank follow and median minus mean
    follow = bank_follow(arms=arms)
    follow.write_csv(out / "bank_follow.csv")
    mmm = median_minus_mean(arms=arms)
    mmm.write_csv(out / "median_minus_mean.csv")
    report(contrasts=contrasts, follow=follow, mmm=mmm, means=pl.DataFrame(mean_records), out=out)


def month_table(*, keys: np.ndarray, frame: Frame, systems: tuple[str, str]) -> None:
    """Print the MAE of two systems by origin month (the gate's month-by-month comparison)."""
    start = np.datetime64(S_LIB.GRID_START.replace(tzinfo=None), "m")
    month = (
        (start + frame.origin * 30).astype("datetime64[M]").astype(str)
    )  # calendar month of the origin
    rows = []
    for name in systems:
        v = values_on(
            path=rows_path(system=name, setting="joint"), keys=keys, columns={"mae": "Model_mae"}
        )["mae"]
        for m in np.unique(month):
            rows.append(
                {"system": name, "origin_month": str(m), "mae": float(v[month == m].mean())}
            )
    print(pl.DataFrame(rows).pivot(on="system", index="origin_month", values="mae"))


BANK_CHECK: Final = {"9a": 0.07101, "5wo": 0.12553}
"""The copied share of 677's stored arms 9a and `5wo` on the held-out bank, 0 to 14 days, monthly refits."""


def bank_follow(*, arms: list[str]) -> pl.DataFrame:
    """Return the share of an injected shift that the median copies, 0 to 14 days (677's `bank` measure).

    Pooling is 677's (`e00677_bank_score.py`): per (arm, seed, month) the least-squares slope
    sum(change * shift) / sum(shift ** 2) over the scored rows whose 1-day lag is shifted; the months
    are pooled with the number of shifted rows as weights; the arm's value is the mean over seeds.
    Every arm must have all 12 months and 3 seeds, and the stored arms must give 0.07101 and 0.12553.
    """
    folders = {
        **{a: (ROOT_731, f"P0_D50_{{m}}_{a}_sp1") for a in arms},
        "9a": (ROOT_677, "P0_D50_{m}_5w_sp1"),
        "5wo": (ROOT_677, "P0_D50_{m}_5wo_sp1"),
    }
    rows = []
    for arm, (root, pattern) in folders.items():
        for month in MONTHS:
            for seed in SEEDS:
                f = root / pattern.format(m=month) / "work" / "bank" / f"{month}_d0_s{seed}.parquet"
                assert f.exists(), ("missing bank file", arm, month, seed, f)
                d = (
                    pl.read_parquet(f, columns=["median_base", "median_bank", "lag_shift"])
                    .filter(pl.col("lag_shift") != 0)
                    .select(
                        num=(
                            (pl.col("median_bank") - pl.col("median_base")) * pl.col("lag_shift")
                        ).sum(),
                        den=(pl.col("lag_shift") ** 2).sum(),
                        rows=pl.len(),
                    )
                )
                rows.append({"arm": arm, "month": month, "seed": seed, **d.row(0, named=True)})
    per = pl.DataFrame(rows).with_columns(follow=pl.col("num") / pl.col("den"))
    assert per.height == len(folders) * len(MONTHS) * len(SEEDS), per.height
    seed_level = per.group_by("arm", "seed").agg(
        follow=(pl.col("follow") * pl.col("rows")).sum() / pl.col("rows").sum(),
        months=pl.len(),
        rows=pl.col("rows").sum(),
    )
    out = (
        seed_level.group_by("arm")
        .agg(
            follow=pl.col("follow").mean(),
            follow_sd_over_seeds=pl.col("follow").std(),
            seeds=pl.len(),
            months_per_seed=pl.col("months").mean(),
            rows_per_seed=pl.col("rows").mean(),
        )
        .sort("arm")
    )
    assert out["seeds"].to_list() == [len(SEEDS)] * out.height
    assert out["months_per_seed"].to_list() == [float(len(MONTHS))] * out.height
    for arm, want in BANK_CHECK.items():
        got = float(out.filter(pl.col("arm") == arm)["follow"][0])
        assert abs(got - want) < 2e-5, (
            "bank follow of the stored arm differs from 677's",
            arm,
            got,
            want,
        )
    print("bank follow: stored arms reproduce 677's", BANK_CHECK)
    return out


def median_minus_mean(*, arms: list[str]) -> pl.DataFrame:
    """Return, per arm and Primary, the median minus the mean of the quantile forecast, in points of capacity.

    The mean of the forecast distribution is approximated from the 11 quantiles by the midpoint rule
    over the probability levels (the ends reach 0 and 1). It is averaged over every forecast row and
    seed, so it needs no seed mean. Streams the backtest file, so memory stays small.
    """
    import e00668_score_lib as lib  # ty: ignore[unresolved-import]

    taus = np.asarray(lib.TAUS)
    edges = np.concatenate([[0.0], (taus[1:] + taus[:-1]) / 2, [1.0]])
    weight = np.diff(edges)
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet").select(
        "time_series_id", "capacity"
    )
    out = []
    for system, name in [
        *((a, f"arm_{a}_sp1") for a in arms),
        *((s, f"stored_{s}_sp1") for s in STORED),
    ]:
        f = ROOT_731 / "backtests" / f"{name}.parquet"
        if not f.exists():
            continue
        mean_q = sum(pl.col(q) * float(w) for q, w in zip(lib.QCOLS, weight, strict=True))
        r = (
            pl.scan_parquet(f)
            .with_columns(time_series_id=pl.col("time_series_id").cast(pl.Int64))
            .group_by("time_series_id")
            .agg(gap=(pl.col("q50") - mean_q).mean(), rows=pl.len())
            .collect(engine="streaming")
            .join(
                capacity.with_columns(pl.col("time_series_id").cast(pl.Int64)), on="time_series_id"
            )
            .with_columns(
                median_minus_mean=pl.col("gap") / pl.col("capacity") * 100.0, system=pl.lit(system)
            )
        )
        out.append(r.select("system", "time_series_id", "median_minus_mean", "rows"))
    return pl.concat(out) if out else pl.DataFrame()


def holds(*, contrasts: pl.DataFrame, name: str, group: str, metric: str = "mae") -> dict:
    """Return the joint interval of a contrast and whether all three seeds share the point estimate's sign."""
    sel = contrasts.filter(
        (pl.col("contrast") == name) & (pl.col("group") == group) & (pl.col("metric") == metric)
        & (pl.col("kind") == "primary")
    )  # fmt: skip
    joint = sel.filter(pl.col("setting") == "joint").row(0, named=True)
    seeds = sel.filter(pl.col("setting") != "joint")["diff"].to_list()
    same = bool(seeds) and all(np.sign(x) == np.sign(joint["diff"]) for x in seeds)
    return {**joint, "seeds": seeds, "same_sign": same}


def report(
    *,
    contrasts: pl.DataFrame,
    follow: pl.DataFrame,
    mmm: pl.DataFrame,
    means: pl.DataFrame,
    out: Path,
) -> None:
    """Print every number the write-up quotes and apply the pre-registered verdict rules mechanically."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(60)
    main_groups = ("six signed-MW", "eight MVA-only", "eight without D8", "all 16")
    print("\n=== SYSTEM MEANS (joint, all 16, normalised MAE and CRPS, points of capacity) ===")
    print(
        means.filter((pl.col("group") == "all 16") & (pl.col("setting") == "joint"))
        .pivot(on="metric", index="system", values="mean")
        .sort("system")
    )
    print("\n=== CONTRASTS, arm minus reference, joint seeds, 99.5% intervals ===")
    for kind in ("primary", "vs 9a"):
        for metric in ("mae", "crps"):
            t = contrasts.filter(
                (pl.col("kind") == kind) & (pl.col("metric") == metric) & (pl.col("setting") == "joint")
                & pl.col("group").is_in(main_groups)
            )  # fmt: skip
            print(f"\n-- {kind}, {metric}")
            print(t.select("contrast", "group", "diff", "lo", "hi", "rows"))
    print("\n=== PER SEED (seed s of the arm minus seed s of the reference), MAE, main groups ===")
    t = contrasts.filter(
        (pl.col("kind") == "primary") & (pl.col("metric") == "mae") & (pl.col("setting") != "joint")
        & pl.col("group").is_in(main_groups[:2])
    )  # fmt: skip
    print(t.pivot(on="setting", index=["contrast", "group"], values="diff"))
    print("\n=== PER PRIMARY, MAE, primary contrasts (diff, lo, hi) ===")
    t = contrasts.filter(
        (pl.col("kind") == "primary") & (pl.col("metric") == "mae") & (pl.col("setting") == "joint")
        & pl.col("group").str.starts_with("Primary")
    )  # fmt: skip
    print(t.select("contrast", "group", "diff", "lo", "hi"))
    print(
        "\n=== LEAVE-ONE-PRIMARY-OUT, MAE: range of the point estimate and number of intervals that exclude zero ==="
    )
    lo = contrasts.filter(
        (pl.col("kind") == "primary") & (pl.col("metric") == "mae") & (pl.col("setting") == "joint")
        & pl.col("group").str.contains(" minus ")
    ).with_columns(base=pl.col("group").str.split(" minus ").list.get(0))  # fmt: skip
    print(
        lo.group_by("contrast", "base")
        .agg(
            min=pl.col("diff").min(),
            max=pl.col("diff").max(),
            sign_flips=((pl.col("diff") > 0).any() & (pl.col("diff") < 0).any()),
            excluding_zero=((pl.col("lo") > 0) | (pl.col("hi") < 0)).sum(),
            n=pl.len(),
        )
        .sort("contrast", "base")
    )
    print("\n=== SCENARIOS (load-shift slices of 677), MAE, primary contrasts ===")
    t = contrasts.filter(
        (pl.col("kind") == "primary") & (pl.col("metric") == "mae") & pl.col("group").str.starts_with("scenario")
    )  # fmt: skip
    print(t.select("contrast", "group", "diff", "lo", "hi", "rows"))
    print("\n=== COPIED SHARE OF INJECTED SHIFTS (bank, 0 to 14 days) ===")
    print(follow)
    print("\n=== MEDIAN MINUS MEAN FORECAST per Primary, points of capacity ===")
    if mmm.height:
        print(
            mmm.pivot(on="system", index="time_series_id", values="median_minus_mean").sort(
                "time_series_id"
            )
        )
        mmm.write_csv(out / "median_minus_mean.csv")
    # ---- the pre-registered verdict, read mechanically
    print("\n=== PRE-REGISTERED READINGS (interim until the reviews) ===")
    share = {r["arm"]: r["follow"] for r in follow.iter_rows(named=True)}
    verdict_parts = {}
    for g in ("six signed-MW", "eight MVA-only"):
        c1, c6 = (
            holds(contrasts=contrasts, name="C1", group=g),
            holds(contrasts=contrasts, name="C6", group=g),
        )
        verdict_parts[g] = {
            "C1": c1,
            "C6": c6,
            "weights_help": c1["lo"] > 0 and c6["lo"] > 0 and c1["same_sign"] and c6["same_sign"],
            "C1_within_0.02": c1["lo"] >= -0.02 and c1["hi"] <= 0.02,
            "C1_below_zero": c1["hi"] < 0,
        }
        print(
            g,
            {k: v for k, v in verdict_parts[g].items() if not isinstance(v, dict)},
            "C1",
            round(c1["diff"], 4),
            (round(c1["lo"], 4), round(c1["hi"], 4)),
            "C6",
            round(c6["diff"], 4),
            (round(c6["lo"], 4), round(c6["hi"], 4)),
        )
    groups_ = list(verdict_parts.values())
    helped = [v["weights_help"] for v in groups_]
    c1_holds = [v["C1"]["lo"] > 0 and v["C1"]["same_sign"] for v in groups_]
    # the pre-registered order: the first rule that applies decides
    if all(helped):
        verdict = "supported"
    elif any(c1 and not h for c1, h in zip(c1_holds, helped, strict=True)):
        verdict = "inconclusive"  # C1 holds at a group where the control C6 fails
    elif sum(helped) == 1:
        verdict = "partly supported"
    elif any(v["C1_below_zero"] for v in groups_):
        verdict = "refuted"
    elif all(v["C1_within_0.02"] for v in groups_):
        verdict = "not supported"
    else:
        verdict = "inconclusive"
    print("RECENCY VERDICT (mechanical, interim):", verdict)
    print(
        "\nArm replaces R0 only if MAE interval wholly below zero at both groups, CRPS upper limit < +0.02, "
    )
    print(
        "point estimate at most -0.02, same sign in all seeds, and copied share at most R0's + 0.02:"
    )
    for arm in ("W0", "W30", "W270", "WP", "IP", "B0", "5wo"):
        name = {
            "W0": "C1",
            "W30": "C2",
            "W270": "C3",
            "5wo": "C4",
            "B0": "C5",
            "WP": "C6",
            "IP": "C7",
        }[arm]
        ok = True
        for g in ("six signed-MW", "eight MVA-only"):
            mae = holds(contrasts=contrasts, name=name, group=g)
            crps = holds(contrasts=contrasts, name=name, group=g, metric="crps")
            ok &= mae["hi"] < 0 and mae["diff"] <= -0.02 and crps["hi"] < 0.02 and mae["same_sign"]
        ok &= share.get(arm, float("nan")) <= share.get("R0", float("nan")) + 0.02
        print(f"  {arm}: replaces R0 = {bool(ok)}")
    print("\nCSV files written to", out)


def main() -> None:
    """Parse the command line and run the chosen stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("export", "rows", "contrasts", "all"), default="all")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--continue-after-gate-failure", action="store_true")
    args = parser.parse_args()
    if args.stage in {"export", "rows", "all"}:
        stage_rows(arms=args.arms)
    if args.stage in {"contrasts", "all"}:
        stage_contrasts(
            arms=args.arms, continue_after_gate_failure=args.continue_after_gate_failure
        )


if __name__ == "__main__":
    main()

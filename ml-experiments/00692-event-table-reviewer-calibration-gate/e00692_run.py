"""Experiment 692: build and render the calibration and realness figures for the reviewer.

Run from the repository root (see experiment 680's `e00680_run.py` for the environment):

    P=ml-experiments
    export PYTHONPATH=$P/00692-event-table-reviewer-calibration-gate:\
$P/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search:\
$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00660-shared-generator-of-injected-load-transfer-events:\
$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00665-event-table-recall-by-length-and-depth
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python $P/00692-*/e00692_run.py candidates
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python $P/00692-*/e00692_run.py render
"""

from __future__ import annotations

import sys
import warnings
from datetime import UTC, datetime

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM
from e00653_inject import OFFSET_2019
from e00680_run import CONFIGS, Rig, decode_variants, params653
from e00692_common import OUT, render
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import e00660_generator as gen  # ty: ignore[unresolved-import]  # isort: skip

warnings.filterwarnings("ignore")
SEEDS = range(900, 940)
E580 = ML_EXPERIMENTS_DATA_DIR / "00580-verified-load-shift-event-set-and-label-audit/work"
E680 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search"
)
START_TOL = 24
N_TRUE, N_TOP_FALSE, N_RANDOM_FALSE, N_HARD = 30, 20, 30, 20
N_REAL_REALNESS = 30
FINAL_TEST_FROM = int(
    (datetime(2026, 7, 1, tzinfo=UTC) - datetime(2022, 1, 1, tzinfo=UTC)).days * DAY
)


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def candidates_of(*, events: list, world, kind: str, seed: int) -> list[dict]:
    """Donor ends of decoded transfers that start in the scored years, labelled by the truth."""
    truth = world.events.filter(
        (pl.col("label") == "edge_transfer")
        & (pl.col("role") == "donor")
        & (pl.col("start") >= SCORED_FROM)
    ).to_dicts()
    hard = world.events.filter(
        (pl.col("label").is_in(["hard_negative", "shift"]))
        | pl.col("role").is_in(["placebo_a", "placebo_b", "common_mode"])
    ).to_dicts()
    transfers = [e for e in events if e.cls.endswith("transfer") and e.start >= SCORED_FROM]
    out = []
    for e in transfers:
        if e.amp_mw >= 0 or not e.partners:
            continue
        partner = max(e.partners, key=lambda p: abs(e.partner_amp.get(p, 0.0)))
        rec = [
            r
            for r in transfers
            if r.primary == partner and r.amp_mw > 0 and abs(r.start - e.start) <= 60
        ]
        r = min(rec, key=lambda r: abs(r.start - e.start)) if rec else None
        end = GRID_LENGTH if e.end is None else e.end
        right = False
        for t in truth:
            t_end = GRID_LENGTH if t["end"] is None else t["end"]
            if (
                t["site"] == e.primary
                and abs(t["start"] - e.start) <= START_TOL
                and _iou((t["start"], t_end), (e.start, end)) >= 0.5
                and partner in list(t["partners"])
            ):
                right = True
        is_hard = any(
            h["site"] in (e.primary, partner) and abs(h["start"] - e.start) <= 2 * DAY for h in hard
        )
        out.append(
            {
                "kind": kind,
                "seed": seed,
                "donor": e.primary,
                "recipient": partner,
                "onset": e.start,
                "end": -1 if e.end is None else e.end,
                "donor_mw": float(e.amp_mw),
                "donor_units_abs": float(e.units),
                "confidence": float(e.confidence),
                "rec_start": -1 if r is None else r.start,
                "rec_end": -2 if r is None else (-1 if r.end is None else r.end),
                "rec_mw": float(e.partner_amp.get(partner, 0.0)) if r is None else float(r.amp_mw),
                "rec_units": float("nan") if r is None else float(r.units),
                "true": right,
                "hard": is_hard,
            }
        )
    return out


def run_candidates() -> None:
    """Decode the 40 scored worlds and list every donor end of E1-J-sym and of the rewired graph."""
    rig = Rig()
    params = params653()
    rows = []
    for seed in SEEDS:
        world = gen.inject(
            panel=rig.panel, config=CONFIGS["default"], seed=seed, start_times=rig.starts
        )
        residual = rig.residual_of(world=world)
        variants, _ = decode_variants(
            rig=rig,
            residual=residual,
            frozen=rig.frozen,
            params=params,
            rng=np.random.default_rng(seed),
        )
        rows += candidates_of(events=variants["E1-J-sym"], world=world, kind="sym", seed=seed)
        rows += candidates_of(
            events=variants["E1-J-sym rewired"], world=world, kind="rewired", seed=seed
        )
        n_ends = sum(
            e.cls.endswith("transfer") and e.start >= SCORED_FROM for e in variants["E1-J-sym"]
        )
        print(seed, n_ends, len(rows), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(OUT / "candidates.parquet")


def _site(*, rig: Rig, power: dict, resid: dict, s: str) -> dict:
    return {
        "power": power[s],
        "forecast": rig.forecast[s],
        "resid": resid[s],
        "sigma": float(rig.frozen.sigma[s]),
    }


def _draw(
    *,
    rig: Rig,
    power: dict,
    resid: dict,
    path,
    donor: str,
    rec: str,
    onset: int,
    end: int | None,
    d_step: tuple,
    r_step: tuple,
) -> None:
    sites = rig.panel.sites
    raw = {s: power[s] - rig.forecast[s] for s in sites}
    stack = np.stack([raw[s] / rig.p653._scale(raw[s]) for s in sites])  # noqa: SLF001
    med = np.nanmedian(stack, axis=0)
    nb = [
        s
        for s in sorted(set(rig.p653.neighbours[donor]) | set(rig.p653.neighbours[rec]))
        if s not in (donor, rec) and s in sites
    ][:4]
    render(
        path=path,
        donor=_site(rig=rig, power=power, resid=resid, s=donor),
        recipient=_site(rig=rig, power=power, resid=resid, s=rec),
        others=[_site(rig=rig, power=power, resid=resid, s=s) for s in nb],
        median_resid_units=med,
        onset=onset,
        end=end,
        donor_step=d_step,
        recipient_step=r_step,
    )


def run_render() -> None:
    """Pick the calibration and realness sets, draw one figure each, and write the answer key."""
    rig = Rig()
    cand = pl.read_parquet(OUT / "candidates.parquet")
    rng = np.random.default_rng(692)
    sym = cand.filter(pl.col("kind") == "sym").with_row_index("cid")
    rew = cand.filter(pl.col("kind") == "rewired").with_row_index("cid")
    true = sym.filter(pl.col("true") & ~pl.col("hard"))
    false = sym.filter(~pl.col("true") & ~pl.col("hard"))
    hard = sym.filter(~pl.col("true") & pl.col("hard"))
    print(
        {
            "sym": sym.height,
            "true": true.height,
            "false": false.height,
            "hard": hard.height,
            "rewired": rew.height,
        }
    )
    pick: list[tuple[str, dict]] = []

    def take(df: pl.DataFrame, n: int, stratum: str) -> pl.DataFrame:
        idx = rng.choice(df.height, size=min(n, df.height), replace=False)
        sel = df[sorted(int(i) for i in idx)]
        pick.extend((stratum, r) for r in sel.to_dicts())
        return sel

    take(true, N_TRUE, "true")
    top = false.sort("confidence", descending=True).head(N_TOP_FALSE)
    pick.extend(("false_top_confidence", r) for r in top.to_dicts())
    rest = false.filter(~pl.col("cid").is_in(top["cid"]))
    take(rest, N_RANDOM_FALSE, "false_random")
    n_hard = min(N_HARD, hard.height)
    take(hard, n_hard, "hard_negative_injected_placebo")
    if n_hard < N_HARD:
        take(rew, N_HARD - n_hard, "hard_negative_rewired")
    used = {(r["seed"], r["onset"], r["donor"]) for _, r in pick}
    realness_inj = sym.filter(
        pl.struct(["seed", "onset", "donor"]).map_elements(
            lambda r: (r["seed"], r["onset"], r["donor"]) not in used, return_dtype=pl.Boolean
        )
    )
    inj_for_realness = take(realness_inj, N_REAL_REALNESS, "realness_injected")
    del inj_for_realness
    # Render injected figures, one world at a time.
    key = []
    order = rng.permutation(len(pick))
    by_seed: dict[int, list[tuple[int, str, dict]]] = {}
    for rank, i in enumerate(order):
        stratum, r = pick[int(i)]
        by_seed.setdefault(r["seed"], []).append((rank, stratum, r))
    sets = {"realness_injected": "realness"}
    for seed, items in sorted(by_seed.items()):
        world = gen.inject(
            panel=rig.panel, config=CONFIGS["default"], seed=seed, start_times=rig.starts
        )
        resid = rig.residual_of(world=world)
        for rank, stratum, r in items:
            fset = sets.get(stratum, "calibration")
            cid = f"{fset}_{rank:03d}"
            end = None if r["end"] < 0 else r["end"]
            sig_d = rig.frozen.sigma[r["donor"]]
            sig_r = rig.frozen.sigma[r["recipient"]]
            rs = r["onset"] if r["rec_start"] < 0 else r["rec_start"]
            re_ = None if r["rec_end"] == -1 else (end if r["rec_end"] == -2 else r["rec_end"])
            _draw(
                rig=rig,
                power=world.power,
                resid=resid,
                path=OUT / "figures" / fset / f"{cid}.png",
                donor=r["donor"],
                rec=r["recipient"],
                onset=r["onset"],
                end=end,
                d_step=(r["onset"], end, r["donor_mw"] / sig_d),
                r_step=(rs, re_, r["rec_mw"] / sig_r),
            )
            key.append(
                {
                    "figure": cid,
                    "set": fset,
                    "stratum": stratum,
                    "source": "injected",
                    **{
                        k: r[k]
                        for k in (
                            "seed",
                            "donor",
                            "recipient",
                            "onset",
                            "end",
                            "confidence",
                            "true",
                        )
                    },
                }
            )
        print("rendered world", seed, flush=True)
    # Real cases: the two verified transfers, and real E1-J-sym transfer candidates.
    real_resid = rig.real_residual
    ver = pl.read_parquet(E580 / "events_verified.parquet")
    for a, b, tag in (("D5", "D16", "2025-12"), ("D3", "D22", "2024-10")):
        v = ver.filter(
            (pl.col("primary_a") == a)
            & (pl.col("primary_b") == b)
            & pl.col("onset_a").str.starts_with(tag)
        ).row(0, named=True)
        on, en = v["onset_a_idx"] - OFFSET_2019, v["end_a_idx"] - OFFSET_2019
        rs_, re2 = v["onset_b_idx"] - OFFSET_2019, v["end_b_idx"] - OFFSET_2019

        def step(s: str, on_: int, en_: int) -> float:
            inside = np.nanmean(real_resid[s][on_ + DAY : en_ - DAY])
            before = np.nanmean(real_resid[s][on_ - 7 * DAY : on_ - 12])
            return float((inside - before) / rig.frozen.sigma[s])

        fid = f"calibration_{len(key) + 1000:04d}"
        _draw(
            rig=rig,
            power=rig.panel.power,
            resid=real_resid,
            path=OUT / "figures" / "calibration" / f"{fid}.png",
            donor=a,
            rec=b,
            onset=on,
            end=en,
            d_step=(on, en, step(a, on, en)),
            r_step=(rs_, re2, step(b, rs_, re2)),
        )
        key.append(
            {
                "figure": fid,
                "set": "calibration",
                "stratum": "verified_real",
                "source": "real",
                "seed": -1,
                "donor": a,
                "recipient": b,
                "onset": on,
                "end": en,
                "confidence": float("nan"),
                "true": True,
            }
        )
        rid = f"realness_{len(key) + 600:03d}"
        _draw(
            rig=rig,
            power=rig.panel.power,
            resid=real_resid,
            path=OUT / "figures" / "realness" / f"{rid}.png",
            donor=a,
            rec=b,
            onset=on,
            end=en,
            d_step=(on, en, step(a, on, en)),
            r_step=(rs_, re2, step(b, rs_, re2)),
        )
        key.append(
            {
                "figure": rid,
                "set": "realness",
                "stratum": "realness_real_verified",
                "source": "real",
                "seed": -1,
                "donor": a,
                "recipient": b,
                "onset": on,
                "end": en,
                "confidence": float("nan"),
                "true": False,
            }
        )
    real = pl.read_parquet(E680 / "real_events_E1-J-sym.parquet").filter(
        pl.col("cls").str.ends_with("transfer")
        & (pl.col("amp_mw") < 0)
        & (pl.col("start") >= SCORED_FROM)
        & (pl.col("start") < FINAL_TEST_FROM)
    )
    reals = real.to_dicts()
    rng.shuffle(reals)
    n_done = 0
    allr = pl.read_parquet(E680 / "real_events_E1-J-sym.parquet")
    for e in reals:
        if n_done >= N_REAL_REALNESS:
            break
        parts = [p for p in e["partners"].split(",") if p in rig.panel.sites]
        if not parts:
            continue
        rec = allr.filter(
            pl.col("primary").is_in(parts)
            & (pl.col("amp_mw") > 0)
            & ((pl.col("start") - e["start"]).abs() <= 60)
            & pl.col("cls").str.ends_with("transfer")
        )
        if rec.height == 0:
            continue
        r = rec.sort("amp_mw", descending=True).row(0, named=True)
        end = None if e["end"] < 0 else e["end"]
        fid = f"realness_{n_done + 500:03d}"
        sd, sr = rig.frozen.sigma[e["primary"]], rig.frozen.sigma[r["primary"]]
        _draw(
            rig=rig,
            power=rig.panel.power,
            resid=real_resid,
            path=OUT / "figures" / "realness" / f"{fid}.png",
            donor=e["primary"],
            rec=r["primary"],
            onset=e["start"],
            end=end,
            d_step=(e["start"], end, e["amp_mw"] / sd),
            r_step=(r["start"], None if r["end"] < 0 else r["end"], r["amp_mw"] / sr),
        )
        key.append(
            {
                "figure": fid,
                "set": "realness",
                "stratum": "realness_real",
                "source": "real",
                "seed": -1,
                "donor": e["primary"],
                "recipient": r["primary"],
                "onset": e["start"],
                "end": -1 if end is None else end,
                "confidence": float("nan"),
                "true": False,
            }
        )
        n_done += 1
    pl.DataFrame(key).write_parquet(OUT / "key.parquet")
    print(pl.DataFrame(key).group_by(["set", "stratum"]).len())


if __name__ == "__main__":
    {"candidates": run_candidates, "render": run_render}[sys.argv[1]]()

# ruff: noqa: E501, PLR0915, B023
"""Build the arms of experiment 460 from the cached heads, with arms c and d fitted on earlier months.

Arms (real world only; forecast tables in experiment 404's layout under `work/predictions/<arm>/`):

- `S1` (a): the stored S1 network of experiment 404, recomputed and checked against 404's tables.
- `S1d` (b): the lag-dropout network with all lags.
- `S1d-c` (c): arm b with its tail quantiles widened by a factor that depends on the mean spread.
- `S1d-d` (d): arm b mixed with its no-lags pass, with a weight that depends on the mean spread.
- `B-mix` (e): equal-weight mixture of the A0-inj network and the frozen lagless table.
- `S1w` (f): arm a with a constant 1.15 widening.
- `S1-ens3` (g): equal-weight density mixture of the S1 networks of fit seeds 0 to 2 (one cell).
- `S1d-c0` (control): arm b after table P (experiment 435), with no spread.

Arms c, c0, and d are fitted per fit seed on the real rows of the earlier months of the same seed
(all rows with a lead of 0 to 240 hours, no event label used), so 2025-10 has none of them. The
spread bins are the quartiles of the fitting rows' mean spread per group. Arm c's factors are
table P's per (group, bin, lead band, tail) and arm d's weights come from a grid search on the
fitting rows' mean CRPS; both are then made non-decreasing in the bin by a running maximum.

```bash
source /tmp/e460_env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00460-*/e00460_arms.py
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import e00435_tailcore as tailcore  # ty: ignore[unresolved-import]
import e00460_heads as heads_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

WORK_DIR: Final[Path] = heads_mod.WORK_DIR
MONTHS: Final[tuple[str, ...]] = heads_mod.MONTHS
SEEDS: Final[tuple[int, ...]] = heads_mod.SEEDS
BATCH: Final[int] = heads_mod.BATCH
CHECK_TOLERANCE: Final[float] = 1e-4
BIN_QUANTILES: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
N_BINS: Final[int] = len(BIN_QUANTILES) + 1
WEIGHT_GRID: Final[tuple[float, ...]] = tuple(round(0.1 * i, 1) for i in range(11))
FIT_SUBSAMPLE: Final[int] = 60_000
GROUP_NAME: Final[dict[str, str]] = {
    "signed_six": "signed-MW six",
    "mva_eight": "MVA-only eight",
    "report_only": "D15 and D18",
}
Head = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
SPREAD: Final[str] = "spread_mean"
"""The spread signal that arms c and d use (the pre-registered primary signal)."""


def prediction_path(*, arm: str, month: str, cell: int) -> Path:
    """Return where an arm's real-world forecast table is stored."""
    return WORK_DIR / "predictions" / arm / f"{month}_s{cell}_w-1.parquet"


def load_head(*, kind: str, month: str, seed: int) -> Head:
    """Return a cached head ("S", "A", "L") or the S1d "all" or "none" pass ("Sd:all")."""
    if kind.startswith("Sd:"):
        saved = torch.load(
            heads_mod.heads_path(kind="Sd", month=month, seed=seed), weights_only=False
        )
        return tuple(saved[kind[3:]])  # ty: ignore[invalid-return-type]
    saved = torch.load(heads_mod.heads_path(kind=kind, month=month, seed=seed), weights_only=False)
    return tuple(saved["head"])  # ty: ignore[invalid-return-type]


def mix(*, heads: list[Head], weights: list[torch.Tensor] | None = None, device: str) -> np.ndarray:
    """Return the sorted 11 quantiles (rows, 11) of a density mixture of the heads.

    `weights` holds one weight tensor (rows,) per head; None means equal weights.
    """
    k = len(heads)
    logs = [
        torch.full((len(h[0]), 1), -float(np.log(k)), dtype=torch.float64)
        if weights is None
        else torch.log(w.double().clamp(1e-9, 1.0))[:, None]  # ty: ignore[unresolved-attribute]
        for h, w in zip(heads, weights or [None] * k, strict=True)  # ty: ignore[invalid-argument-type]
    ]
    log_w = torch.cat([h[0] + extra for h, extra in zip(heads, logs, strict=True)], dim=1)
    mean = torch.cat([h[1] for h in heads], dim=1)
    std = torch.cat([h[2] for h in heads], dim=1)
    out = []
    for start in range(0, len(log_w), BATCH):
        sl = slice(start, start + BATCH)
        out.append(
            models.mixture_quantiles(
                log_w=log_w[sl].to(device), mean=mean[sl].to(device), std=std[sl].to(device)
            ).cpu()
        )
    return torch.sort(torch.cat(out), dim=1).values.numpy().astype(np.float64)


def crps_points(*, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the score in points of capacity per row (twice the mass-weighted pinball loss)."""
    diff = actual[:, None] - q
    levels = np.array(models.LEVELS)[None, :]
    loss = np.maximum(levels * diff, (levels - 1.0) * diff)
    return 200.0 * (loss * np.asarray(common.ATOM_MASSES, dtype=np.float64)[None, :]).sum(1)


def with_groups(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the 435 group name and the lead band to a table, keeping leads of 0 to 240 hours."""
    sites = report.fit.prep.primaries()["site"].to_list()
    groups = report.group_of(sites=sites)
    return frame.with_columns(
        group=pl.Series([GROUP_NAME[g] for g in groups[frame["primary"].to_numpy()]]),
        lead_band=tailcore.lead_band_expr(),
    ).filter(pl.col("lead_band").is_not_null())


def bin_edges(*, fit_rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the spread quartiles of the fitting rows per group."""
    return {
        str(g): np.quantile(part[SPREAD].to_numpy(), BIN_QUANTILES)
        for (g,), part in fit_rows.group_by("group")
    }


def bin_of(*, frame: pl.DataFrame, edges: dict[str, np.ndarray]) -> np.ndarray:
    """Return the spread bin (0 to 3) of each row, using its group's edges."""
    out = np.zeros(frame.height, dtype=np.int64)
    group = frame["group"].to_numpy()
    spread = frame[SPREAD].to_numpy()
    for g, e in edges.items():
        sel = group == g
        out[sel] = np.searchsorted(e, spread[sel], side="right")
    return out


def monotone(*, factors: pl.DataFrame, n_bins: int) -> pl.DataFrame:
    """Make table P factors non-decreasing in the bin; a missing bin takes the running maximum.

    `factors` has the keys `group` ("<group>|<bin>"), `lead_band`, `tail`, and `factor`.
    """
    split = factors.with_columns(
        base=pl.col("group").str.split("|").list.get(0),
        bin=pl.col("group").str.split("|").list.get(1).cast(pl.Int64),
    )
    rows = []
    for (base, band, tail), part in split.group_by("base", "lead_band", "tail"):
        have = dict(zip(part["bin"].to_list(), part["factor"].to_list(), strict=True))
        running = 1.0
        for b in range(n_bins):
            running = max(running, have.get(b, running))
            rows.append(
                {"group": f"{base}|{b}", "lead_band": band, "tail": tail, "factor": running}
            )
    return pl.DataFrame(rows)


def fit_widening(
    *, fit_rows: pl.DataFrame, edges: dict[str, np.ndarray], n_bins: int
) -> pl.DataFrame:
    """Return the monotone table P factors per (group, bin, lead band, tail)."""
    binned = fit_rows.with_columns(
        bin=pl.Series(
            bin_of(frame=fit_rows, edges=edges) if n_bins > 1 else np.zeros(fit_rows.height, int)
        )
    ).with_columns(group=pl.col("group") + "|" + pl.col("bin").cast(pl.String))
    raw = prov.fit_factors(fit=binned)
    return monotone(factors=raw, n_bins=n_bins)


def apply_widening(
    *, frame: pl.DataFrame, factors: pl.DataFrame, edges: dict[str, np.ndarray], n_bins: int
) -> pl.DataFrame:
    """Widen the tail quantiles of a table by the factors of each row's group, bin, and band."""
    bins = bin_of(frame=frame, edges=edges) if n_bins > 1 else np.zeros(frame.height, dtype=int)
    keyed = frame.with_columns(group_key=pl.col("group") + "|" + pl.Series(bins.astype(str)))
    typed = keyed.cast(dict.fromkeys(("q50", *tailcore.TAIL_LEVELS), pl.Float64)).with_columns(
        group=pl.col("group_key")
    )
    out = prov.apply_factors(frame=typed, factors=factors)
    return out.with_columns(group=frame["group"]).drop("group_key")


def fit_weights(
    *, fits: list[tuple[str, int]], edges: dict[str, np.ndarray], seed: int, device: str
) -> dict[tuple[str, int], float]:
    """Return the no-lags weight per (group, bin) that minimises the fitting rows' mean CRPS."""
    rng = np.random.default_rng(seed)
    pools: dict[str, list[tuple[Head, Head, np.ndarray, np.ndarray]]] = {}
    for month, s in fits:
        frame = with_groups(
            frame=pl.read_parquet(passes_for(month=month, seed=s)).with_row_index("row")
        )
        h_all = load_head(kind="Sd:all", month=month, seed=s)
        h_none = load_head(kind="Sd:none", month=month, seed=s)
        for (g,), part in frame.group_by("group"):
            rows = part["row"].to_numpy().astype(np.int64)
            if len(rows) > FIT_SUBSAMPLE:
                rows = np.sort(rng.choice(rows, size=FIT_SUBSAMPLE, replace=False))
            sel = torch.from_numpy(rows)
            take = lambda h: tuple(t[sel] for t in h)  # noqa: E731
            sub = frame.filter(pl.col("row").is_in(rows.tolist())).sort("row")
            pools.setdefault(str(g), []).append(
                (
                    take(h_all),
                    take(h_none),
                    sub["actual"].to_numpy().astype(np.float64),  # ty: ignore[invalid-argument-type]
                    bin_of(frame=sub, edges=edges),
                )
            )
    chosen: dict[tuple[str, int], float] = {}
    for g, items in pools.items():
        actual = np.concatenate([i[2] for i in items])
        bins = np.concatenate([i[3] for i in items])
        scores = np.zeros((len(actual), len(WEIGHT_GRID)))
        for j, w in enumerate(WEIGHT_GRID):
            q = np.concatenate(
                [
                    mix(
                        heads=[i[0], i[1]],
                        weights=[torch.full((len(i[2]),), 1.0 - w), torch.full((len(i[2]),), w)],
                        device=device,
                    )
                    for i in items
                ]
            )
            scores[:, j] = crps_points(q=q, actual=actual)
        running = 0.0
        for b in range(N_BINS):
            sel = bins == b
            best = WEIGHT_GRID[int(np.argmin(scores[sel].mean(0)))] if sel.any() else running
            running = max(running, best)
            chosen[(g, b)] = running
    return chosen


def passes_for(*, month: str, seed: int) -> Path:
    """Return the real-world five-pass table of a cell."""
    return heads_mod.passes_path(month=month, seed=seed, world=-1)


def write_arm(*, arm: str, month: str, cell: int, base: pl.DataFrame, q: np.ndarray) -> None:
    """Write an arm's table: the base columns with the 11 quantiles replaced."""
    names = list(score.LEVEL_NAMES)
    frame = base.with_columns(
        **{n: pl.Series(q[:, i]).cast(pl.Float32) for i, n in enumerate(names)}
    )
    path = prediction_path(arm=arm, month=month, cell=cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def check_stored(*, month: str, q_s1: np.ndarray, q_bmix: np.ndarray, keys: dict) -> list[str]:
    """Compare the recomputed seed-0 S1 and B-mix with experiment 404's stored forecasts."""
    names = list(score.LEVEL_NAMES)
    lines = []
    for arm, q in (("S1", q_s1), ("B-mix", q_bmix)):
        stored = pl.read_parquet(score.prediction_path(arm=arm, month=month, seed=0, world=-1))
        for k in heads_mod.KEY_COLUMNS:
            assert np.array_equal(stored[k].to_numpy(), keys[k]), f"stored {arm} {k} differs"
        err = float(np.abs(q - stored[names].to_numpy()).max())
        lines.append(f"{month} seed 0 {arm}: max |recomputed - stored| {err:.2e}")
        assert err < CHECK_TOLERANCE, lines[-1]
    return lines


def build_month(*, month: str, device: str) -> None:
    """Write every arm of a month, for each fit seed (cell)."""
    earlier = [m for m in MONTHS if m < month]
    q_s1: dict[int, np.ndarray] = {}
    for cell in SEEDS:
        base_full = pl.read_parquet(passes_for(month=month, seed=cell))
        keep = [c for c in base_full.columns if not c.startswith("med_")]
        base = base_full.select(keep)
        keys = {k: base[k].to_numpy() for k in heads_mod.KEY_COLUMNS}
        s_head = load_head(kind="S", month=month, seed=cell)
        a_head = load_head(kind="A", month=month, seed=cell)
        l_head = load_head(kind="L", month=month, seed=cell)
        d_all = load_head(kind="Sd:all", month=month, seed=cell)
        d_none = load_head(kind="Sd:none", month=month, seed=cell)
        q_s1[cell] = mix(heads=[s_head], device=device)
        q_bmix = mix(heads=[a_head, l_head], device=device)
        if cell == 0:
            lines = check_stored(month=month, q_s1=q_s1[0], q_bmix=q_bmix, keys=keys)
            (WORK_DIR / "checks").mkdir(parents=True, exist_ok=True)
            (WORK_DIR / "checks" / f"{month}.txt").write_text("\n".join(lines) + "\n")
            print("\n".join(lines), flush=True)
        write_arm(arm="S1", month=month, cell=cell, base=base, q=q_s1[cell])
        write_arm(arm="B-mix", month=month, cell=cell, base=base, q=q_bmix)
        write_arm(arm="S1d", month=month, cell=cell, base=base,
                  q=base_full.select(list(score.LEVEL_NAMES)).to_numpy().astype(np.float64))  # fmt: skip
        widened = tailcore.widen(
            frame=pl.read_parquet(prediction_path(arm="S1", month=month, cell=cell)).cast(
                dict.fromkeys(("q50", *tailcore.TAIL_LEVELS), pl.Float64)
            ),
            factor=tailcore.WIDENING,
        )
        path = prediction_path(arm="S1w", month=month, cell=cell)
        path.parent.mkdir(parents=True, exist_ok=True)
        widened.cast(dict.fromkeys(score.LEVEL_NAMES, pl.Float32)).write_parquet(path)
        if not earlier:
            print(f"{month} s{cell}: no earlier months, arms c, c0, d skipped", flush=True)
            continue
        fits = [(m, cell) for m in earlier]
        fit_rows = pl.concat(
            [with_groups(frame=pl.read_parquet(passes_for(month=m, seed=s))) for m, s in fits]
        ).cast(dict.fromkeys(("actual", "q50", SPREAD, *tailcore.TAIL_LEVELS), pl.Float64))
        edges = bin_edges(fit_rows=fit_rows)
        test_full = base_full.select(keep)
        for arm, n_bins in (("S1d-c", N_BINS), ("S1d-c0", 1)):
            factors = fit_widening(fit_rows=fit_rows, edges=edges, n_bins=n_bins)
            out = apply_widening(
                frame=with_groups(frame=test_full).cast(
                    dict.fromkeys(("q50", SPREAD, *tailcore.TAIL_LEVELS), pl.Float64)
                ),
                factors=factors, edges=edges, n_bins=n_bins,
            )  # fmt: skip
            full = test_full.join(
                out.select(*heads_mod.KEY_COLUMNS, *tailcore.TAIL_LEVELS),
                on=list(heads_mod.KEY_COLUMNS),
                how="left",
                suffix="_w",
            )
            for n in tailcore.TAIL_LEVELS:
                full = full.with_columns(pl.coalesce(f"{n}_w", n).alias(n)).drop(f"{n}_w")
            path = prediction_path(arm=arm, month=month, cell=cell)
            path.parent.mkdir(parents=True, exist_ok=True)
            full.cast(dict.fromkeys(score.LEVEL_NAMES, pl.Float32)).write_parquet(path)
            factors.write_csv(WORK_DIR / "checks" / f"factors_{arm}_{month}_s{cell}.csv")
        weights = fit_weights(fits=fits, edges=edges, seed=cell, device=device)
        full_groups = base_full.select(keep).with_columns(
            group=pl.Series(
                [
                    GROUP_NAME[g]
                    for g in report.group_of(sites=report.fit.prep.primaries()["site"].to_list())[
                        base_full["primary"].to_numpy()
                    ]
                ]
            )
        )
        bins = bin_of(frame=full_groups, edges=edges)
        w = np.array(
            [
                weights.get((g, b), 0.0)
                for g, b in zip(full_groups["group"].to_list(), bins, strict=True)
            ]
        )
        q_d = mix(
            heads=[d_all, d_none],
            weights=[torch.from_numpy(1.0 - w), torch.from_numpy(w)],
            device=device,
        )
        write_arm(arm="S1d-d", month=month, cell=cell, base=base, q=q_d)
        pl.DataFrame(
            [{"group": g, "bin": b, "weight": v} for (g, b), v in weights.items()]
        ).write_csv(WORK_DIR / "checks" / f"weights_{month}_s{cell}.csv")
        print(f"{month} s{cell}: arms c, c0, d written", flush=True)
    heads = [load_head(kind="S", month=month, seed=c) for c in SEEDS]
    for c in SEEDS:
        saved = torch.load(heads_mod.heads_path(kind="S", month=month, seed=c), weights_only=False)
        assert np.array_equal(
            saved["keys"]["target"],
            pl.read_parquet(passes_for(month=month, seed=0))["target"].to_numpy(),
        )
    base = pl.read_parquet(passes_for(month=month, seed=0))
    base = base.select([c for c in base.columns if not c.startswith("med_")])
    write_arm(arm="S1-ens3", month=month, cell=0, base=base, q=mix(heads=heads, device=device))


def main() -> None:
    """Build the arms of the given months."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for month in args.months:
        build_month(month=month, device=args.device)


if __name__ == "__main__":
    main()

# ruff: noqa: E501
r"""Part 3 of experiment 522: does the network use its Primary embedding vectors?

Scores the 2026-01 test rows (all leads up to 240 hours, rows with a cleaned actual) of three stored
networks, with the embedding replaced in turn by: the stored vectors (identity), 20 random
derangements of the 16 vectors, the mean vector, each Primary's regenerated random start, and 20
draws of fresh N(0, 1) vectors. Only the embedding table changes: the input standardiser and each
Primary's target mean and standard deviation stay the stored ones. The networks are the pretrained
L0PT weights (fold 2026-01, seed 0) and the fine-tuned L0PT weights of experiment 448, and the
fine-tuned arm A of experiment 425 (a lagged network with the same embedding). CRPS is the 11-level
quantile score of `e00087_common.pinball_quantile_score` on the raw (unwidened) quantiles, in
percentage points of effective capacity. Writes `work/part3.json` and `work/part3_per_day.npz`.

```bash
source <(printf 'export PYTHONPATH=%s\\n' ml-experiments/00087-*:ml-experiments/00397-*:ml-experiments/00418-*:ml-experiments/00422-*:ml-experiments/00428-*:ml-experiments/00510-*:ml-experiments/00522-*)
DATA_PATH_INTERNAL=<private data folder> OMP_NUM_THREADS=2 \\
    uv run --no-sync python ml-experiments/00522-*/e00522_part3.py
```
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00087_common as c87  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data428  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00522_common as common  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

MONTH: Final[str] = "2026-01"
DRAWS: Final[int] = 20
BOOTSTRAPS: Final[int] = 2_000
REPORT_ONLY: Final[frozenset[str]] = frozenset({"D15", "D18"})
CHUNK: Final[int] = 65_536
STEPS_PER_DAY: Final[int] = 48


@dataclass
class Rows:
    """The scored rows of one network: inputs, Primary index, actuals, and origin day."""

    x: np.ndarray
    primary: np.ndarray
    actual: np.ndarray
    day: np.ndarray


def derangement(*, rng: np.random.Generator) -> np.ndarray:
    """Return a random permutation of 0..15 with no fixed point."""
    while True:
        order = rng.permutation(common.N_PRIMARIES)
        if not np.any(order == np.arange(common.N_PRIMARIES)):
            return order


def embedding_module(*, model: torch.nn.Module) -> torch.nn.Embedding:
    """Return the embedding layer of a `PlainMdn`."""
    layer = model.embedding  # ty: ignore[unresolved-attribute]
    assert isinstance(layer, torch.nn.Embedding)
    return layer


def predict(
    *,
    model: torch.nn.Module,
    rows: Rows,
    table: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    """Return per-row CRPS (fraction of capacity) with the embedding table replaced by `table`."""
    layer = embedding_module(model=model)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(table.astype(np.float32)))
    out = np.empty(len(rows.actual))
    for a in range(0, len(rows.actual), CHUNK):
        sl = slice(a, a + CHUNK)
        x = torch.from_numpy(rows.x[sl])
        p = torch.from_numpy(rows.primary[sl])
        q = c510.serving_quantiles(model=model, x=x, p=p, outlier=False).numpy()
        q = np.sort(
            q * target_std[rows.primary[sl]][:, None] + target_mean[rows.primary[sl]][:, None],
            axis=1,
        )
        out[sl] = c87.pinball_quantile_score(quantiles=q, actual=rows.actual[sl])
    return out


def per_primary_day(*, crps: np.ndarray, rows: Rows, n_days: int) -> np.ndarray:
    """Return the mean CRPS per (Primary, origin day), shape (16, n_days), NaN where a cell has no row."""
    total = np.zeros((common.N_PRIMARIES, n_days))
    count = np.zeros((common.N_PRIMARIES, n_days))
    np.add.at(total, (rows.primary, rows.day), crps)
    np.add.at(count, (rows.primary, rows.day), 1.0)
    with np.errstate(invalid="ignore"):
        return total / count


def l0pt_rows(*, standardiser_path: Path, clean: np.ndarray) -> Rows:
    """Return experiment 448's 2026-01 test rows of the lag-free network, with cleaned actuals."""
    z = np.load(standardiser_path)
    standardiser = prep.Standardiser(mean=z["mean"], std=z["std"], mask=z["mask"])
    test = data428.build_test(month=MONTH, standardiser=standardiser)
    actual = clean[test.primary, test.target_idx]
    keep = np.isfinite(actual)
    origin_day = ((test.origin - int(test.origin.min())) // (86_400 * 1_000_000)).astype(np.int64)
    return Rows(
        x=test.x[keep], primary=test.primary[keep], actual=actual[keep], day=origin_day[keep]
    )


def a_rows(*, standardiser_path: Path, clean: np.ndarray) -> Rows:
    """Return the 2026-01 test rows of arm A of experiment 425 (summary inputs including the lags)."""
    z = np.load(standardiser_path)
    standardiser = prep.Standardiser(mean=z["mean"], std=z["std"], mask=z["mask"])
    xs, ps, ys, ds = [], [], [], []
    origins = prep.test_origins(month=MONTH)
    for day, origin in enumerate(origins):
        for row, keys, x in prep.run_inputs(
            origin=origin, scheme="summary", standardiser=standardiser
        ):
            valid_idx = np.asarray(c87.grid_index(times=keys["valid_time"]))
            actual = clean[int(row["primary"]), valid_idx]
            keep = np.isfinite(actual) & (keys["lead_hours"].to_numpy() <= 240)
            xs.append(x[keep])
            ps.append(np.full(int(keep.sum()), int(row["primary"]), dtype=np.int64))
            ys.append(actual[keep])
            ds.append(np.full(int(keep.sum()), day, dtype=np.int64))
    return Rows(
        x=np.concatenate(xs),
        primary=np.concatenate(ps),
        actual=np.concatenate(ys),
        day=np.concatenate(ds),
    )


def load_l0pt(*, path: Path, n_inputs: int) -> torch.nn.Module:
    """Return a lag-free network with stored weights, in evaluation mode."""
    model = c510.new_model(outlier=False, n_inputs=n_inputs, n_primaries=common.N_PRIMARIES)
    c510.load_pretrained(model=model, path=path)
    return model.eval()


def load_a(*, path: Path) -> torch.nn.Module:
    """Return arm A of experiment 425 with its stored weights, in evaluation mode."""
    state = torch.load(path, map_location="cpu")
    model = base.PlainMdn(
        n_inputs=state["net.0.weight"].shape[1] - common.EMBEDDING_WIDTH, hidden=64, layers=2,
        components=3, dropout=0.1, n_primaries=common.N_PRIMARIES,
    )  # fmt: skip
    model.load_state_dict(state)
    return model.eval()


def run_network(
    *, name: str, model: torch.nn.Module, rows: Rows, target_mean: np.ndarray, target_std: np.ndarray, seed: int,
    include_start: bool, sites: list[str],
) -> dict[str, object]:  # fmt: skip
    """Score every arm of Part 3 for one network and return per-Primary changes against the identity."""
    stored = embedding_module(model=model).weight.detach().numpy().astype(np.float64).copy()
    n_days = int(rows.day.max()) + 1
    identity = per_primary_day(
        crps=predict(
            model=model, rows=rows, target_mean=target_mean, target_std=target_std, table=stored
        ),
        rows=rows,
        n_days=n_days,
    )
    arms: dict[str, list[np.ndarray]] = {}
    rng = np.random.default_rng(seed)
    arms["derangement"] = [
        per_primary_day(
            crps=predict(
                model=model,
                rows=rows,
                target_mean=target_mean,
                target_std=target_std,
                table=stored[derangement(rng=rng)],
            ),
            rows=rows,
            n_days=n_days,
        )
        for _ in range(DRAWS)
    ]
    arms["mean_vector"] = [
        per_primary_day(
            crps=predict(
                model=model,
                rows=rows,
                target_mean=target_mean,
                target_std=target_std,
                table=np.tile(stored.mean(axis=0), (common.N_PRIMARIES, 1)),
            ),
            rows=rows,
            n_days=n_days,
        )
    ]
    if include_start:
        arms["start_vector"] = [
            per_primary_day(
                crps=predict(
                    model=model,
                    rows=rows,
                    target_mean=target_mean,
                    target_std=target_std,
                    table=common.regenerated_start(seed=seed),
                ),
                rows=rows,
                n_days=n_days,
            )
        ]
    arms["fresh_random"] = [
        per_primary_day(
            crps=predict(
                model=model,
                rows=rows,
                target_mean=target_mean,
                target_std=target_std,
                table=np.random.default_rng(1000 + i).normal(size=stored.shape),
            ),
            rows=rows,
            n_days=n_days,
        )
        for i in range(DRAWS)
    ]
    saved: dict[str, Any] = {k: np.stack(v) for k, v in arms.items()}
    np.savez(
        common.WORK_DIR / f"part3_per_day_{name}.npz",
        identity=identity,
        **saved,
    )
    return summarise(identity=identity, arms=arms, sites=sites, seed=seed)


def summarise(
    *, identity: np.ndarray, arms: dict[str, list[np.ndarray]], sites: list[str], seed: int
) -> dict[str, object]:
    """Return per-Primary CRPS change (percentage points), the headline mean over scored Primaries, and a day-block interval."""
    scored = np.array([s not in REPORT_ONLY for s in sites])
    rng = np.random.default_rng(seed)
    n_days = identity.shape[1]
    resamples = rng.integers(0, n_days, size=(BOOTSTRAPS, n_days))
    out: dict[str, object] = {
        "identity_crps_by_primary_pct": (100 * np.nanmean(identity, axis=1)).round(3).tolist(),
        "identity_crps_mean_scored_pct": float(
            100 * np.nanmean(np.nanmean(identity, axis=1)[scored])
        ),
    }
    for name, draws in arms.items():
        delta = np.mean([d - identity for d in draws], axis=0)  # (16, days)
        per_primary = 100 * np.nanmean(delta, axis=1)
        boot = np.array(
            [100 * np.nanmean(np.nanmean(delta[:, r], axis=1)[scored]) for r in resamples]
        )
        draw_means = [100 * np.nanmean(np.nanmean(d - identity, axis=1)[scored]) for d in draws]
        out[name] = {
            "change_by_primary_pct": per_primary.round(3).tolist(),
            "change_mean_scored_pct": float(100 * np.nanmean(np.nanmean(delta, axis=1)[scored])),
            "day_block_interval_95": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "draw_sd_of_scored_mean": float(np.std(draw_means)) if len(draw_means) > 1 else None,
            "n_draws": len(draws),
        }
    return out


def main() -> None:
    """Score Part 3 for the three stored networks."""
    torch.set_num_threads(1)
    clean_map = prep422.cleaned_arrays()
    sites = prep.primaries()["site"].to_list()
    clean = np.stack([clean_map[s] for s in sites])
    report: dict[str, object] = {"sites": sites, "month": MONTH}
    folder = common.E448 / "work" / "models" / "L0PT" / MONTH / "s0"
    rows = l0pt_rows(standardiser_path=folder / "standardiser.npz", clean=clean)
    z = np.load(folder / "standardiser.npz")
    report["l0pt_rows"] = len(rows.actual)
    for name, path in (
        ("L0PT_pretrained", common.L0PT_PRETRAINED / f"{MONTH}_s0.pt"),
        ("L0PT_finetuned", folder / "model.pt"),
    ):
        model = load_l0pt(path=path, n_inputs=rows.x.shape[1])
        report[name] = run_network(
            name=name, model=model, rows=rows, target_mean=z["target_mean"], target_std=z["target_std"], seed=0,
            include_start=True, sites=sites,
        )  # fmt: skip
        print(name, "done", flush=True)
    a_folder = common.E425 / "work" / "models" / "A" / MONTH / "s0"
    meta_a = json.loads((a_folder / "meta.json").read_text())
    rows_a = a_rows(standardiser_path=a_folder / "standardiser.npz", clean=clean)
    report["a_rows"] = len(rows_a.actual)
    model_a = load_a(path=a_folder / "pooled.pt")
    start = common.regenerated_start(seed=0)
    stored = embedding_module(model=model_a).weight.detach().numpy().astype(np.float64)
    report["A_start_corr"] = float(np.corrcoef(stored.ravel(), start.ravel())[0, 1])
    report["A"] = run_network(
        name="A", model=model_a, rows=rows_a, target_mean=np.array(meta_a["target_mean"]), target_std=np.array(meta_a["target_std"]), seed=0,
        include_start=report["A_start_corr"] > 0.8, sites=sites,
    )  # fmt: skip
    (common.WORK_DIR / "part3.json").write_text(json.dumps(report, indent=1, default=float))
    for name in ("L0PT_pretrained", "L0PT_finetuned", "A"):
        net = report[name]
        assert isinstance(net, dict)
        print(
            name,
            "identity",
            round(net["identity_crps_mean_scored_pct"], 3),
            {
                k: round(v["change_mean_scored_pct"], 3)
                for k, v in net.items()
                if isinstance(v, dict)
            },
        )


if __name__ == "__main__":
    main()

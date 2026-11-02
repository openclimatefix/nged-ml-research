# ruff: noqa: E501
"""Part 2 of experiment 522: do the Primary embeddings' movements follow physical behaviour?

Builds ten counted per-Primary descriptors from the 730 days of cleaned power before each fold's
refit (the fold's training window), runs two-sided Mantel tests (Spearman correlation, 10,000
permutations of the 16 labels) of the seed-averaged distance matrix of the embedding movement
(stored vector minus regenerated start) against each descriptor's distance matrix, and reports the
descriptive companions: the stored-vector view, cosine distance, the other three folds, the
correlation of each descriptor with the movement norm, a hierarchical clustering, and a
planted-signal control. Writes `work/part2.json` and `work/descriptors_<fold>.parquet`.

```bash
source <(printf 'export PYTHONPATH=%s\\n' ml-experiments/00087-*:ml-experiments/00397-*:ml-experiments/00418-*:ml-experiments/00422-*:ml-experiments/00522-*)
DATA_PATH_INTERNAL=<private data folder> OMP_NUM_THREADS=2 \\
    uv run --no-sync python ml-experiments/00522-*/e00522_part2.py
```
"""

import json
from typing import Final

import e00087_common as c87  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00522_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

PERMUTATIONS: Final[int] = 10_000
COUNTED: Final[tuple[str, ...]] = (
    "median_demand", "p95_demand", "load_factor", "profile_winter", "profile_summer",
    "weekend_ratio_winter", "weekend_ratio_summer", "temperature_slope_winter",
    "temperature_slope_summer", "shortwave_sensitivity_midday", "metering",
)  # fmt: skip
PROFILES: Final[frozenset[str]] = frozenset({"profile_winter", "profile_summer"})
PLANTED_PERMUTATIONS: Final[int] = 1_000
PROBES: Final[int] = 1
"""The leave-one-Primary-out probe of the metering type is the one counted test besides the Mantel tests."""
TESTS: Final[int] = len(COUNTED) + PROBES
LEVEL: Final[float] = 0.05 / TESTS
WINDOW_DAYS: Final[int] = 730
MIN_HALF_HOURS: Final[int] = 48 * 20
"""A season counts for a Primary if its window holds at least 20 weekdays and 20 weekend days of cleaned data."""
HALF_HOURS_PER_DAY: Final[int] = 48


def local_frame() -> pl.DataFrame:
    """Return the grid's local (Europe/London) month, weekday flag, and half-hour of day."""
    t = c87.grid_times().dt.convert_time_zone("Europe/London")
    return pl.DataFrame(
        {
            "month": t.dt.month(),
            "weekend": t.dt.weekday() >= 6,
            "half_hour": t.dt.hour() * 2 + t.dt.minute() // 30,
            "date": t.dt.date(),
        }
    )


def descriptors(*, fold: str, clean: dict[str, np.ndarray], local: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each descriptor for the 16 Primaries, from the 730 days before the fold's refit.

    Scalar descriptors are arrays of length 16; the two profiles are 16 x 48 shape arrays (each
    Primary's weekday profile divided by its own mean, so that the profile carries no demand level).
    """
    refit = prep.refit_of(month=fold)
    stop = int(c87.grid_index(times=refit))
    start = stop - WINDOW_DAYS * HALF_HOURS_PER_DAY
    month = local["month"].to_numpy()[start:stop]
    weekend = local["weekend"].to_numpy()[start:stop]
    half = local["half_hour"].to_numpy()[start:stop]
    day = local["date"].to_numpy()[start:stop]
    winter = np.isin(month, (12, 1, 2))
    summer = np.isin(month, (6, 7, 8))
    out: dict[str, list] = {name: [] for name in COUNTED}
    table = prep.primaries()
    for row in table.iter_rows(named=True):
        y = clean[row["site"]][start:stop]
        ok = np.isfinite(y)
        weather = prep418.dense_weather(series_id=int(row["time_series_id"]), site=row["site"])[start:stop]
        temp = weather[:, prep.WEATHER.index("temperature_2m")]
        sw = weather[:, prep.WEATHER.index("shortwave")]
        p95 = float(np.quantile(y[ok], 0.95))
        out["metering"].append(1.0 if row["units"] == "MW" else 0.0)
        out["median_demand"].append(float(np.median(y[ok])))
        out["p95_demand"].append(p95)
        out["load_factor"].append(float(y[ok].mean() / p95))
        for name, season in (("winter", winter), ("summer", summer)):
            weekday = ok & season & ~weekend
            enough = int(weekday.sum()) >= MIN_HALF_HOURS and int((ok & season & weekend).sum()) >= MIN_HALF_HOURS
            if not enough:
                out[f"profile_{name}"].append(np.full(HALF_HOURS_PER_DAY, np.nan))
                out[f"weekend_ratio_{name}"].append(np.nan)
                out[f"temperature_slope_{name}"].append(np.nan)
                continue
            profile = np.array([y[weekday & (half == h)].mean() for h in range(HALF_HOURS_PER_DAY)])
            out[f"profile_{name}"].append(profile / profile.mean())
            out[f"weekend_ratio_{name}"].append(float(y[ok & season & weekend].mean() / y[weekday].mean()))
            daily = pl.DataFrame({"d": day[ok & season], "y": y[ok & season], "t": temp[ok & season]}).group_by("d").agg(
                pl.col("y").mean(), pl.col("t").mean(), pl.len().alias("n")
            ).filter(pl.col("n") >= 40)
            slope = np.polyfit(daily["t"].to_numpy(), daily["y"].to_numpy(), 1)[0]
            out[f"temperature_slope_{name}"].append(float(slope))
        midday = ok & summer & ~weekend & (half >= 22) & (half <= 27) & np.isfinite(sw)
        out["shortwave_sensitivity_midday"].append(
            float(np.polyfit(sw[midday], y[midday], 1)[0]) if int(midday.sum()) >= 120 else np.nan
        )
    return {k: np.array(v) for k, v in out.items()}


def finite_rows(*, values: np.ndarray) -> np.ndarray:
    """Return the Primaries whose descriptor is finite (a short-history Primary may lack a season)."""
    return np.isfinite(values).all(axis=tuple(range(1, values.ndim)))


def descriptor_matrix(*, name: str, values: np.ndarray) -> np.ndarray:
    """Return the distance matrix of a descriptor (z-scored absolute difference, or profile Euclidean)."""
    if name in PROFILES:
        return squareform(pdist(values))
    z = (values - values.mean()) / values.std()
    return squareform(pdist(z[:, None]))


def metering_probe(*, fold: str, view: str) -> dict[str, object]:
    """Return the leave-one-Primary-out linear probe of the metering type (MW = 1, MVA = 0) from the 8-wide vector.

    A logistic regression (L2, C = 1) is fitted on 15 Primaries and predicts the 16th, for each of the
    5 seeds. D15 and D18 belong to the group of their metering unit and are among the 16. The exact
    binomial (Clopper-Pearson) interval is on the 16 majority-vote predictions across seeds.
    """
    from scipy.stats import beta  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    labels = np.array([1 if u == "MW" else 0 for u in prep.primaries()["units"].to_list()])
    votes = np.zeros((len(common.SEEDS), common.N_PRIMARIES))
    for k, seed in enumerate(common.SEEDS):
        emb = common.embedding_of_state(path=common.L0PT_PRETRAINED / f"{fold}_s{seed}.pt")
        if view == "movement":
            emb = emb - common.regenerated_start(seed=seed)
        for held in range(common.N_PRIMARIES):
            keep = np.arange(common.N_PRIMARIES) != held
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(emb[keep], labels[keep])
            votes[k, held] = clf.predict(emb[held : held + 1])[0]
    per_seed = (votes == labels).mean(axis=1)
    majority = (votes.mean(axis=0) > 0.5) == labels.astype(bool)
    hits = int(majority.sum())
    n = common.N_PRIMARIES
    lo = float(beta.ppf(0.025 / TESTS, hits, n - hits + 1)) if hits > 0 else 0.0
    hi = float(beta.ppf(1 - 0.025 / TESTS, hits + 1, n - hits)) if hits < n else 1.0
    return {
        "accuracy_per_seed": per_seed.round(3).tolist(),
        "majority_vote_accuracy": hits / n,
        "interval_bonferroni_95": [lo, hi],
        "share_of_largest_group": float(max(labels.mean(), 1 - labels.mean())),
        "n_mw": int(labels.sum()),
        "n_mva": int((1 - labels).sum()),
    }


def mantel(*, a: np.ndarray, b: np.ndarray, seed: int, permutations: int = PERMUTATIONS) -> dict[str, float]:
    """Return the Spearman correlation of two distance matrices and its two-sided permutation p-value.

    Args:
        a: The n x n distance matrix of the embedding view.
        b: The n x n distance matrix of the descriptor; its labels are permuted.
        seed: The seed of the permutations.
        permutations: How many random relabellings to draw.

    Returns:
        The observed correlation, the number of Primaries and pairs, and the two-sided p-value with
        the +1 correction.
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    upper = np.triu_indices(n, k=1)
    ra = rankdata(a[upper])
    observed = float(np.corrcoef(ra, rankdata(b[upper]))[0, 1])
    count = 0
    for _ in range(permutations):
        order = rng.permutation(n)
        r = float(np.corrcoef(ra, rankdata(b[np.ix_(order, order)][upper]))[0, 1])
        count += abs(r) >= abs(observed) - 1e-12
    return {"rho": observed, "p": (count + 1) / (permutations + 1), "n_primaries": n, "n_pairs": len(upper[0])}


def mantel_on_finite(*, a: np.ndarray, name: str, values: np.ndarray, seed: int) -> dict[str, float]:
    """Return `mantel` on the Primaries that have the descriptor."""
    keep = finite_rows(values=values)
    return mantel(a=a[np.ix_(keep, keep)], b=descriptor_matrix(name=name, values=values[keep]), seed=seed)


def movement_distance(*, fold: str, view: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the seed-averaged 16 x 16 distance matrix and the mean row norm of an embedding view."""
    matrices, norms = [], []
    for seed in common.SEEDS:
        emb = common.embedding_of_state(path=common.L0PT_PRETRAINED / f"{fold}_s{seed}.pt")
        if view == "movement":
            emb = emb - common.regenerated_start(seed=seed)
        matrices.append(squareform(pdist(emb, metric=metric)))
        norms.append(np.linalg.norm(emb, axis=1))
    return np.mean(matrices, axis=0), np.mean(norms, axis=0)


def planted_signal_detection(*, values: np.ndarray, name: str, signal_share: float, draws: int, seed: int) -> float:
    """Return the share of synthetic movements, built from one descriptor, that the Mantel test detects at `LEVEL`.

    A synthetic 16 x 8 movement is the standardised descriptor times a random unit vector plus
    Gaussian noise, scaled so that `signal_share` of the variance is the descriptor's. For a profile,
    the first principal component of the profiles is the signal.
    """
    rng = np.random.default_rng(seed)
    values = values[finite_rows(values=values)]
    if name in PROFILES:
        centred = values - values.mean(axis=0)
        signal = np.linalg.svd(centred, full_matrices=False)[0][:, 0]
    else:
        signal = values
    z = (signal - signal.mean()) / signal.std()
    target = descriptor_matrix(name=name, values=values)
    hits = 0
    for draw in range(draws):
        u = rng.normal(size=common.EMBEDDING_WIDTH)
        u /= np.linalg.norm(u)
        noise = rng.normal(size=(common.N_PRIMARIES, common.EMBEDDING_WIDTH))
        synthetic = np.sqrt(signal_share) * z[:, None] * u[None, :] * np.sqrt(common.EMBEDDING_WIDTH) + np.sqrt(1 - signal_share) * noise
        result = mantel(a=squareform(pdist(synthetic)), b=target, seed=seed + draw, permutations=PLANTED_PERMUTATIONS)
        hits += result["p"] < LEVEL
    return hits / draws


def _norm_corr(*, norms: np.ndarray, values: np.ndarray) -> float:
    """Return the Spearman correlation of the movement norm with a descriptor (profiles: their mean absolute deviation)."""
    v = np.abs(values - np.nanmean(values, axis=0)).mean(axis=1) if values.ndim == 2 else values
    keep = np.isfinite(v)
    return float(np.corrcoef(rankdata(norms[keep]), rankdata(v[keep]))[0, 1])


def main() -> None:
    """Compute and write Part 2."""
    clean = prep422.cleaned_arrays()
    local = local_frame()
    table = prep.primaries()
    sites = table["site"].to_list()
    signed = [s for s, u in zip(sites, table["units"].to_list(), strict=True) if u == "MW"]
    report: dict[str, object] = {"sites": sites, "signed_mw_sites": signed, "level": LEVEL, "counted_tests": len(COUNTED)}
    for fold in common.FOLDS:
        desc = descriptors(fold=fold, clean=clean, local=local)
        flat = {k: (v if v.ndim == 1 else np.nanmean(v, axis=1)) for k, v in desc.items()}
        pl.DataFrame({"site": sites, **{k: v.tolist() for k, v in flat.items()}}).write_parquet(common.WORK_DIR / f"descriptors_{fold}.parquet")
        views = {
            ("movement", "euclidean"): movement_distance(fold=fold, view="movement", metric="euclidean"),
            ("stored", "euclidean"): movement_distance(fold=fold, view="stored", metric="euclidean"),
            ("movement", "cosine"): movement_distance(fold=fold, view="movement", metric="cosine"),
        }
        fold_out: dict[str, object] = {
            "metering_probe_movement": metering_probe(fold=fold, view="movement"),
            "metering_probe_stored": metering_probe(fold=fold, view="stored"),
        }
        for (view, metric), (dist, norms) in views.items():
            tests = {}
            for i, name in enumerate(COUNTED):
                res = mantel_on_finite(a=dist, name=name, values=desc[name], seed=100 + i)
                tests[name] = res | {"passes_level": res["p"] < LEVEL}
            fold_out[f"{view}_{metric}"] = tests
            if view == "movement" and metric == "euclidean":
                fold_out["movement_norm_spearman_with_descriptor"] = {
                    name: _norm_corr(norms=norms, values=desc[name])
                    for name in COUNTED
                }
                link = linkage(squareform(dist), method="average")
                fold_out["clusters_movement_4"] = dict(zip(sites, fcluster(link, 4, criterion="maxclust").tolist(), strict=True))
                fold_out["linkage"] = link.tolist()
                # Classical multidimensional scaling of the averaged distance matrix, for the figure.
                d2 = dist**2
                j = np.eye(16) - np.ones((16, 16)) / 16
                vals, vecs = np.linalg.eigh(-0.5 * j @ d2 @ j)
                top = np.argsort(vals)[::-1][:2]
                fold_out["mds_2d"] = (vecs[:, top] * np.sqrt(np.maximum(vals[top], 0))).tolist()
                fold_out["distance_matrix_movement"] = dist.tolist()
        report[fold] = fold_out
        if fold == "2026-01":
            desc_flat = {k: v.tolist() for k, v in flat.items()}
            report["descriptors_2026-01"] = desc_flat
            report["planted_signal_detection"] = {
                f"{name}|share{share}": planted_signal_detection(values=desc[name], name=name, signal_share=share, draws=100, seed=900)
                for name in COUNTED
                if name != "metering"
                for share in (0.25, 0.5)
            }
            sign = [s in signed for s in sites]
            flow = [float((clean[s][:] < 0).mean()) if g else None for s, g in zip(sites, sign, strict=True)]
            report["reverse_flow_share_signed_mw"] = dict(zip(sites, flow, strict=True))
    (common.WORK_DIR / "part2.json").write_text(json.dumps(report, indent=1, default=float))
    summary = {
        fold: {k: {n: (round(t["rho"], 2), round(t["p"], 4)) for n, t in v.items()} for k, v in report[fold].items() if k.startswith(("movement_", "stored_"))}
        for fold in common.FOLDS
    }
    print(json.dumps(summary["2026-01"], indent=1))
    print(json.dumps(report["planted_signal_detection"], indent=0))


if __name__ == "__main__":
    main()

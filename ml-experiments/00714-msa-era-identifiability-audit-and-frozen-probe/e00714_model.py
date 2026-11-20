"""Experiment 714: the classifier, the grouped cross-validation, and the bootstrap.

**Classifier.** A logistic regression with an L2 penalty on standardised features (numpy and scipy;
scikit-learn is not a declared dependency). The standardising mean and standard deviation, the
median used to fill a missing value, and the penalty strength come from the training passes only.
The penalty is chosen by an inner split into 4 groups of whole passes.

**Outer split.** Leave one fold month out: the classifier is fitted on the passes of 3 fold months
and scores the passes of the fourth, so no pass and no month seen in fitting is scored.

**Sensitivity.** A boosted tree of depth 2 (XGBoost, 1 thread, fixed settings) on the same features.

**Intervals.** A two-stage bootstrap of the out-of-fold predictions: whole passes are resampled
within each fold month, then fold months are resampled. The predictions are not refitted per draw.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import rankdata

PENALTIES: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0, 100.0)
"""Grid of the L2 penalty on the summed log-loss."""
INNER_GROUPS: Final[int] = 4
XGB_PARAMS: Final[dict] = {
    "max_depth": 2, "n_estimators": 80, "learning_rate": 0.1, "n_jobs": 1,
    "tree_method": "hist", "verbosity": 0, "random_state": 0,
}  # fmt: skip
CONFIDENCE: Final[float] = 1.0 - 0.05 / 6.0
"""Level of the interval of each of the 6 planned contrasts (99.17%)."""


def auroc(*, label: np.ndarray, score: np.ndarray) -> float:
    """Return the area under the ROC curve of `score` for the positive label, ties averaged.

    Args:
        label: Boolean or 0/1 labels.
        score: Scores; higher means positive.

    Returns:
        The AUROC, or NaN if one class is empty.
    """
    positive = label.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class Logistic:
    """A fitted logistic regression on standardised features.

    Attributes:
        weight: Coefficient of each standardised feature.
        bias: Intercept.
        mean: Training mean of each feature.
        scale: Training standard deviation of each feature (1 where it was 0).
        fill: Training median of each feature, used for a missing value.
        penalty: The chosen penalty.
    """

    weight: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    fill: np.ndarray
    penalty: float

    def standardise(self, *, x: np.ndarray) -> np.ndarray:
        """Fill missing values and standardise with the training statistics."""
        filled = np.where(np.isfinite(x), x, self.fill)
        return (filled - self.mean) / self.scale

    def decision(self, *, x: np.ndarray) -> np.ndarray:
        """Return the log-odds of the positive class."""
        return self.standardise(x=x) @ self.weight + self.bias

    def proba(self, *, x: np.ndarray) -> np.ndarray:
        """Return the predicted probability of the positive class."""
        return 1.0 / (1.0 + np.exp(-self.decision(x=x)))


def _fit_standardised(*, z: np.ndarray, y: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    """Fit weights and bias on standardised features by L-BFGS."""

    def loss(params: np.ndarray) -> tuple[float, np.ndarray]:
        w, b = params[:-1], params[-1]
        s = z @ w + b
        value = np.sum(np.logaddexp(0.0, s) - y * s) + 0.5 * penalty * float(w @ w)
        p = 1.0 / (1.0 + np.exp(-s))
        grad = np.concatenate([z.T @ (p - y) + penalty * w, [np.sum(p - y)]])
        return float(value), grad

    result = minimize(loss, np.zeros(z.shape[1] + 1), jac=True, method="L-BFGS-B")
    return result.x[:-1], float(result.x[-1])


def fit_logistic(*, x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> Logistic:
    """Fit a logistic regression, choosing the penalty by an inner split of whole passes.

    Args:
        x: Features, shape (rows, columns), NaN where missing.
        y: 0/1 labels.
        groups: Pass identifier of each row, for the inner split.
        seed: Seed of the inner split.

    Returns:
        The fitted classifier, refitted on all rows at the chosen penalty.
    """
    fill = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    filled = np.where(np.isfinite(x), x, fill)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (filled - mean) / scale
    names = np.unique(groups)
    rng = np.random.default_rng(seed)
    assign = dict(zip(names, rng.permutation(len(names)) % INNER_GROUPS, strict=True))
    fold = np.array([assign[g] for g in groups])
    best, best_loss = PENALTIES[0], np.inf
    for penalty in PENALTIES:
        total = 0.0
        for f in range(INNER_GROUPS):
            train, test = fold != f, fold == f
            if test.sum() == 0 or len(np.unique(y[train])) < 2:
                continue
            w, b = _fit_standardised(z=z[train], y=y[train], penalty=penalty)
            s = z[test] @ w + b
            total += float(np.sum(np.logaddexp(0.0, s) - y[test] * s))
        if total < best_loss:
            best, best_loss = penalty, total
    w, b = _fit_standardised(z=z, y=y, penalty=best)
    return Logistic(weight=w, bias=b, mean=mean, scale=scale, fill=fill, penalty=best)


def out_of_fold(
    *, x: np.ndarray, y: np.ndarray, groups: np.ndarray, months: np.ndarray, model: str = "logistic"
) -> np.ndarray:
    """Return out-of-fold probabilities by leaving each fold month out in turn.

    Args:
        x: Features, shape (rows, columns).
        y: 0/1 labels.
        groups: Pass identifier of each row.
        months: Fold month of each row.
        model: `logistic` or `tree` (a boosted tree of depth 2).

    Returns:
        The predicted probability of the positive class for every row, from a classifier that did
        not see the row's fold month.
    """
    out = np.full(len(y), np.nan)
    for month in np.unique(months):
        test = months == month
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        if model == "logistic":
            fitted = fit_logistic(x=x[train], y=y[train], groups=groups[train])
            out[test] = fitted.proba(x=x[test])
        else:
            from xgboost import XGBClassifier

            tree = XGBClassifier(**XGB_PARAMS)
            tree.fit(x[train], y[train])
            out[test] = tree.predict_proba(x[test])[:, 1]
    return out


def transfer_scores(
    *, x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """Fit on one set of events and score another set (for the cross-generator reading)."""
    fitted = fit_logistic(x=x_train, y=y_train, groups=groups)
    return fitted.proba(x=x_test)


def resample_indices(
    *, months: np.ndarray, passes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw row indices by resampling whole passes within fold months, then fold months.

    Args:
        months: Fold month of each row.
        passes: Pass identifier of each row (unique across months).
        rng: The random generator.

    Returns:
        An index array into the rows.
    """
    by_month: dict[str, dict[str, np.ndarray]] = {}
    for month in np.unique(months):
        sel = np.flatnonzero(months == month)
        by_month[month] = {p: sel[passes[sel] == p] for p in np.unique(passes[sel])}
    chosen = rng.choice(list(by_month), size=len(by_month), replace=True)
    parts: list[np.ndarray] = []
    for month in chosen:
        pass_ids = list(by_month[month])
        parts.extend(
            by_month[month][p] for p in rng.choice(pass_ids, size=len(pass_ids), replace=True)
        )
    return np.concatenate(parts)


def bootstrap_draws(
    *, months: np.ndarray, passes: np.ndarray, n_draws: int, seed: int
) -> list[np.ndarray]:
    """Return `n_draws` resampled index arrays, reusable across feature sets (paired draws)."""
    rng = np.random.default_rng(seed)
    return [resample_indices(months=months, passes=passes, rng=rng) for _ in range(n_draws)]


def interval(*, values: np.ndarray, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Return the percentile interval of bootstrap values at the given confidence."""
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.nanpercentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def bootstrap_auroc(*, label: np.ndarray, score: np.ndarray, draws: list[np.ndarray]) -> np.ndarray:
    """Return the AUROC of each bootstrap draw."""
    return np.array([auroc(label=label[d], score=score[d]) for d in draws])


def calibration_table(*, label: np.ndarray, proba: np.ndarray, bins: int = 10) -> pl.DataFrame:
    """Return the mean predicted probability and observed share in equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(proba, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = which == b
        if sel.any():
            rows.append(
                {"bin": b, "n": int(sel.sum()), "mean_pred": float(proba[sel].mean()),
                 "observed": float(label[sel].mean())}
            )  # fmt: skip
    return pl.DataFrame(rows)

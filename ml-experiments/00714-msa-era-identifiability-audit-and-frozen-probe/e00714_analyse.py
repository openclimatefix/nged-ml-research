"""Experiment 714: fit the classifiers, score the six contrasts and the placebos, write the tables.

    .venv/bin/python ml-experiments/00714-*/e00714_analyse.py [--smoke]

Reads `features/{bank,g2,gas}.parquet` (`features_smoke/` with `--smoke`), written by
`e00714_build.py`, and writes under
`results/`: `oof_<world>_k<k>.parquet` (out-of-fold probabilities by feature set),
`primary_contrasts.json` (the six pre-registered contrasts), `placebos.json`,
`sensitivity.json`, `coefficients.json`, `secondary_cues_by_kind.csv`, `calibration.csv`, and
`outcome_table.md`. The pre-registered contrasts use the worlds `bank` and `g2` only. The world
`gas` (intermittent steps) enters the secondary per-cue and per-kind readings only.
`--smoke` uses 200 bootstrap draws and 3 label permutations and writes under `results_smoke/`.
"""

import e00714_threads  # noqa: F401  (must run before numpy)

# isort: split

import argparse
import json
from pathlib import Path
from typing import Final

import e00714_common as common
import e00714_features as feats
import e00714_model as model
import numpy as np
import polars as pl

FEATURE_DIR: Final[Path] = common.DATA_DIR / "features"
HOST: Final[list[str]] = list(feats.HOST_FEATURES)
NBR: Final[list[str]] = list(feats.NEIGHBOUR_FEATURES)
STEP: Final[list[str]] = list(feats.STEP_FEATURES)
OLD: Final[list[str]] = list(feats.OLD_FEATURES)
RW: Final[list[str]] = ["rw_mirror_share", "rw_mirror_z"]
FEATURE_SETS: Final[dict[str, list[str]]] = {
    "host": HOST,
    "host_old": HOST + OLD,
    "host_nbr": HOST + NBR,
    "host_nbr_rewired": HOST + RW,
    "host_steps": HOST + STEP,
    "known_answer": [f"ka_{n}" for n in HOST + NBR],
}
CUE_SETS: Final[dict[str, list[str]]] = {
    "cue_flat": ["flat"],
    "cue_mirror": NBR,
    "cue_steps": STEP,
    "cue_flat_mirror": ["flat", *NBR],
    "cue_flat_steps": ["flat", *STEP],
    "cue_all_three": ["flat", *NBR, *STEP],
}
PRIMARY_SETS: Final[tuple[str, ...]] = (
    "host", "host_old", "host_nbr", "host_nbr_rewired", "known_answer", "host_steps"
)  # fmt: skip
MIRRORED: Final[tuple[str, ...]] = ("full", "partial")
PERMUTATIONS: Final[int] = 20
DRAWS: Final[int] = 10_000
SECONDARY_DRAWS: Final[int] = 1000
STRATIFIED_DRAWS: Final[int] = 1000
SITE_PRIOR_LIMIT: Final[float] = 0.60
SECONDARY_CONFIDENCE: Final[float] = 0.95
TOLERANCE: Final[float] = 0.03
NO_EVENT_TOLERANCE: Final[float] = 0.05
HIDDEN_GAIN_LIMIT: Final[float] = 0.02
BANK_ENOUGH_ORIGINS: Final[tuple[int, ...]] = (7, 14)


def load_world(*, world: str, smoke: bool) -> pl.DataFrame:
    """Read a world's feature table, or raise if it is missing."""
    folder = FEATURE_DIR.with_name("features_smoke") if smoke else FEATURE_DIR
    return pl.read_parquet(folder / f"{world}.parquet")


def matrix(*, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the feature matrix of the named columns, NaN where missing."""
    return frame.select(columns).cast(pl.Float64).to_numpy()


def oof_for_sets(*, frame: pl.DataFrame, sets: dict[str, list[str]], tree: bool) -> pl.DataFrame:
    """Return out-of-fold probabilities of each feature set for the rows of `frame`."""
    y = frame["label"].to_numpy().astype(float)
    months = frame["month"].to_numpy()
    out = {}
    for name, columns in sets.items():
        x = matrix(frame=frame, columns=columns)
        out[name] = model.out_of_fold(x=x, y=y, months=months)
        if tree and name in ("host", "host_nbr"):
            out[f"{name}_tree"] = model.out_of_fold(x=x, y=y, months=months, model="tree")
    return pl.DataFrame(out)


def mirrored_mask(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the rows that are permanent changes or temporary moves with a metered mirror."""
    return (frame["label"] == 1).to_numpy() | frame["mirror_state"].is_in(MIRRORED).to_numpy()


def hidden_mask(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the rows that are permanent changes or temporary moves with a hidden mirror."""
    return (frame["label"] == 1).to_numpy() | (frame["mirror_state"] == "hidden").to_numpy()


def in_force_mask(*, frame: pl.DataFrame) -> np.ndarray:
    """Return permanent changes and the temporary moves still in force at the data cut."""
    return (frame["label"] == 1).to_numpy() | frame["in_force"].to_numpy()


def negative_size_mask(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the rows whose injected change at the scored host is a decrease."""
    return (frame["size_true"] < 0).to_numpy()


def _sub_draws(*, draws: list[np.ndarray], mask: np.ndarray | None) -> list[np.ndarray]:
    """Keep, in each draw, the rows inside the mask (the arrays keep their full length)."""
    return draws if mask is None else [d[mask[d]] for d in draws]


def point_and_interval(
    *, label: np.ndarray, score: np.ndarray, draws: list[np.ndarray], mask: np.ndarray | None = None
) -> dict[str, float]:
    """Return an AUROC with its 99.17% interval over paired bootstrap draws, on the mask's rows."""
    sel = np.ones(len(label), dtype=bool) if mask is None else mask
    boot = model.bootstrap_auroc(label=label, score=score, draws=_sub_draws(draws=draws, mask=mask))
    low, high = model.interval(values=boot)
    return {"auroc": model.auroc(label=label[sel], score=score[sel]), "low": low, "high": high}


def contrast(
    *, label: np.ndarray, a: np.ndarray, b: np.ndarray, draws: list[np.ndarray],
    mask: np.ndarray | None = None,
) -> dict[str, float]:  # fmt: skip
    """Return AUROC(a) - AUROC(b) with its 99.17% interval, point and draws on the mask's rows."""
    sel = np.ones(len(label), dtype=bool) if mask is None else mask
    sub = _sub_draws(draws=draws, mask=mask)
    diff = model.bootstrap_auroc(label=label, score=a, draws=sub) - model.bootstrap_auroc(
        label=label, score=b, draws=sub
    )
    low, high = model.interval(values=diff)
    return {
        "diff": model.auroc(label=label[sel], score=a[sel])
        - model.auroc(label=label[sel], score=b[sel]),
        "low": low, "high": high,
    }  # fmt: skip


def c6_contrast(
    *, bank: tuple[pl.DataFrame, np.ndarray, list[np.ndarray]],
    second: tuple[pl.DataFrame, np.ndarray, list[np.ndarray]],
    masks: tuple[np.ndarray | None, np.ndarray | None], seed: int,
) -> dict[str, float]:  # fmt: skip
    """Return the bank's AUROC minus the second generator's, with independent worlds' draws.

    Args:
        bank: The bank's frame, scores, and draws.
        second: The second generator's frame, scores, and draws.
        masks: Optional row masks of the bank and of the second generator.
        seed: Seed of the random pairing of the two worlds' draws.

    Returns:
        The difference and its 99.17% interval.
    """
    est = []
    boots = []
    for (frame, score, draws), mask in zip((bank, second), masks, strict=True):
        label = frame["label"].to_numpy()
        sel = np.ones(len(label), dtype=bool) if mask is None else mask
        est.append(model.auroc(label=label[sel], score=score[sel]))
        boots.append(
            model.bootstrap_auroc(
                label=label, score=score, draws=_sub_draws(draws=draws, mask=mask)
            )
        )
    np.random.default_rng(seed).shuffle(boots[1])  # independent worlds: pair the draws at random
    low, high = model.interval(values=boots[0] - boots[1])
    return {"diff": est[0] - est[1], "low": low, "high": high}


def mirror_share_extremes(*, tables: dict[str, pl.DataFrame]) -> dict:
    """Return the fraction of rows at exactly 0 and at the clip of `mirror_share`, by class.

    The no-event column (`np_mirror_share`) scores the real panel at the same hosts and onsets.
    """
    result: dict = {}
    for world in ("bank", "g2"):
        frame = tables[world].filter(pl.col("k") == 14)
        for column in ("mirror_share", "np_mirror_share"):
            for label, name in ((1, "permanent"), (0, "temporary")):
                values = frame.filter(pl.col("label") == label)[column].drop_nulls().to_numpy()
                result[f"{world}/{column}/{name}"] = {
                    "median": float(np.median(values)),
                    "at_zero": float(np.mean(values == 0.0)),
                    "at_clip": float(np.mean(np.abs(values) >= feats.MIRROR_CLIP)),
                }
    return result


def boot_limit(draws: int) -> int:
    """Return the number of bootstrap draws used for the slower site-stratified readings."""
    return min(draws, STRATIFIED_DRAWS)


def site_prior_readings(
    *, tables: dict[str, pl.DataFrame],
    store: dict[tuple[str, int], tuple[pl.DataFrame, pl.DataFrame, list[np.ndarray]]], draws: int,
) -> dict:  # fmt: skip
    """Return how far the Primary alone predicts the class, and the AUROC within each Primary.

    The site-prior AUROC scores each event by its Primary's share of permanent changes in the other
    fold months, from the truth tables alone. The site-stratified AUROC ranks only pairs of a
    permanent and a temporary event at the same Primary, so it cannot use the Primary's identity.
    Both are secondary readings that sit beside C1 to C3 and C6.
    """
    result: dict = {}
    for world in ("bank", "g2"):
        frame = tables[world].filter(pl.col("k") == 14).sort("pass_id", "event")
        sites = frame["host"].to_numpy()
        label = frame["label"].to_numpy()
        prior = model.site_prior_scores(sites=sites, label=label, months=frame["month"].to_numpy())
        result[f"{world}/site_prior_auroc"] = model.auroc(label=label, score=prior)
        for k in common.ORIGIN_DAYS:
            f, preds, boot = store[(world, k)]
            s, y, p = f["host"].to_numpy(), f["label"].to_numpy(), preds["host"].to_numpy()
            values = np.array(
                [
                    model.site_stratified_auroc(sites=s[d], label=y[d], score=p[d])
                    for d in boot[:draws]
                ]
            )
            low, high = model.interval(values=values, confidence=SECONDARY_CONFIDENCE)
            result[f"{world}/k{k}/site_stratified_auroc"] = {
                "auroc": model.site_stratified_auroc(sites=s, label=y, score=p),
                "low95": low, "high95": high,
            }  # fmt: skip
    return result


def primary(*, tables: dict[str, pl.DataFrame], draws: int, out: Path, tree: bool) -> dict:
    """Fit the primary sets on `bank` and `g2`, compute C1 to C6 and their companion readings."""
    result: dict = {"auroc": {}, "contrasts": {}, "sign_only": {}, "secondary": {}}
    store: dict[tuple[str, int], tuple[pl.DataFrame, pl.DataFrame, list[np.ndarray]]] = {}
    for world in ("bank", "g2"):
        for k in common.ORIGIN_DAYS:
            frame = tables[world].filter(pl.col("k") == k).sort("pass_id", "event")
            sets = {n: FEATURE_SETS[n] for n in PRIMARY_SETS}
            preds = oof_for_sets(frame=frame, sets=sets, tree=tree)
            preds.with_columns(
                frame.select("pass_id", "month", "kind", "label", "mirror_state")
            ).write_parquet(out / f"oof_{world}_k{k}.parquet")
            boot = model.bootstrap_draws(
                months=frame["month"].to_numpy(), passes=frame["pass_id"].to_numpy(),
                n_draws=draws, seed=714_000 + k,
            )  # fmt: skip
            store[(world, k)] = (frame, preds, boot)
            label = frame["label"].to_numpy()
            for name in preds.columns:
                result["auroc"][f"{world}/k{k}/{name}"] = point_and_interval(
                    label=label, score=preds[name].to_numpy(), draws=boot
                )
            sign = (frame["size"] > 0).to_numpy().astype(float)
            result["sign_only"][f"{world}/k{k}"] = point_and_interval(
                label=label, score=sign, draws=boot
            )
    result["mirror_share_extremes"] = mirror_share_extremes(tables=tables)
    result["site_prior"] = site_prior_readings(tables=tables, store=store, draws=boot_limit(draws))
    for name, k in (("C1", 7), ("C2", 14), ("C3", 28)):
        result["contrasts"][name] = result["auroc"][f"g2/k{k}/host"]
    frame, preds, boot = store[("g2", 14)]
    label = frame["label"].to_numpy()
    host, old, nbr = (preds[n].to_numpy() for n in ("host", "host_old", "host_nbr"))
    result["contrasts"]["C4"] = contrast(label=label, a=old, b=host, draws=boot)
    mirrored, hidden = mirrored_mask(frame=frame), hidden_mask(frame=frame)
    result["contrasts"]["C5"] = contrast(label=label, a=nbr, b=host, draws=boot, mask=mirrored)
    result["contrasts"]["C5_auroc_host_nbr_mirrored"] = point_and_interval(
        label=label, score=nbr, draws=boot, mask=mirrored
    )
    result["contrasts"]["C5_auroc_host_nbr_hidden"] = point_and_interval(
        label=label, score=nbr, draws=boot, mask=hidden
    )
    result["contrasts"]["C5_gain_hidden"] = contrast(
        label=label, a=nbr, b=host, draws=boot, mask=hidden
    )
    bank = (
        store[("bank", 14)][0],
        store[("bank", 14)][1]["host"].to_numpy(),
        store[("bank", 14)][2],
    )
    second = (frame, host, boot)
    result["contrasts"]["C6"] = c6_contrast(
        bank=bank, second=second, masks=(None, None), seed=714_006
    )
    # companion readings (secondary): sign cannot help on decreases only; moves in force at the cut
    result["secondary"]["C6_decreases_only"] = c6_contrast(
        bank=bank, second=second,
        masks=(negative_size_mask(frame=bank[0]), negative_size_mask(frame=frame)), seed=714_007,
    )  # fmt: skip
    result["secondary"]["C6_in_force"] = c6_contrast(
        bank=bank, second=second,
        masks=(in_force_mask(frame=bank[0]), in_force_mask(frame=frame)), seed=714_008,
    )  # fmt: skip
    for name, k in (("C1", 7), ("C2", 14), ("C3", 28)):
        f, p, b = store[("g2", k)]
        result["secondary"][f"{name}_in_force"] = point_and_interval(
            label=f["label"].to_numpy(), score=p["host"].to_numpy(), draws=b,
            mask=in_force_mask(frame=f),
        )  # fmt: skip
        result["secondary"][f"{name}_decreases_only"] = point_and_interval(
            label=f["label"].to_numpy(), score=p["host"].to_numpy(), draws=b,
            mask=negative_size_mask(frame=f),
        )  # fmt: skip
    return result


def decisions(*, result: dict, placebo: dict) -> dict[str, bool]:
    """Apply every part of the pre-registered decision rule to the contrasts and the placebos.

    Precedence: any failed audit flag makes the run `inconclusive_pending_audit`, and no outcome
    is then stated. Otherwise outcome 3 applies only where the second generator does not clear
    the threshold (`bank_inflated_g2_enough` names the case where both are high), and the
    neighbour-row stop is the complement of the neighbour-row gain test.
    """
    c, auroc = result["contrasts"], result["auroc"]
    enough_g2 = c["C1"]["low"] >= 0.70 and c["C2"]["low"] >= 0.70
    not_enough = c["C1"]["high"] < 0.70 or c["C2"]["high"] < 0.70
    bank_enough = all(auroc[f"bank/k{k}/host"]["low"] >= 0.70 for k in BANK_ENOUGH_ORIGINS)
    gain_clause = c["C5"]["low"] >= 0.05
    neighbour = gain_clause and c["C5_auroc_host_nbr_mirrored"]["auroc"] >= 0.85
    hidden_small = abs(c["C5_gain_hidden"]["diff"]) < HIDDEN_GAIN_LIMIT
    shuffled_ok = all(placebo[f"{w}/shuffled_labels"]["within_0.03"] for w in ("bank", "g2"))
    rewired_ok = all(
        abs(auroc[f"{w}/k14/host_nbr_rewired"]["auroc"] - auroc[f"{w}/k14/host"]["auroc"])
        <= TOLERANCE
        for w in ("bank", "g2")
    )
    no_event_ok = all(
        abs(placebo[f"{w}/no_event_placebo"]["auroc"] - 0.5) <= NO_EVENT_TOLERANCE
        for w in ("bank", "g2")
    )
    site_prior_ok = result["site_prior"]["g2/site_prior_auroc"] <= SITE_PRIOR_LIMIT
    bank_inflated = c["C6"]["low"] > 0.10
    return {
        "outcome_1_host_rows_not_enough": not_enough,
        "outcome_2_host_rows_enough_on_g2": enough_g2,
        "outcome_2_host_rows_enough_on_bank_and_g2": enough_g2 and bank_enough,
        "neither_enough_nor_not_enough": not enough_g2 and not not_enough,
        "outcome_3_injector_artefact": bank_inflated and not enough_g2,
        "bank_inflated_g2_enough": bank_inflated and enough_g2,
        "outcome_4_neighbour_rows_pass": neighbour,
        "outcome_4_hidden_mirror_gain_under_0.02": hidden_small,
        "outcome_4_full": neighbour and hidden_small,
        "f4_stopped_gain_interval_low_under_0.05": not gain_clause,
        "known_answer_pass": auroc["bank/k14/known_answer"]["auroc"] > 0.95,
        "shuffled_label_within_0.03": shuffled_ok,
        "rewired_within_0.03_bank_and_g2": rewired_ok,
        "no_event_within_0.05": no_event_ok,
        "g2_site_prior_auroc_within_limit": site_prior_ok,
        "inconclusive_pending_audit": not (
            shuffled_ok and rewired_ok and no_event_ok and site_prior_ok
        ),
    }


def placebos(*, tables: dict[str, pl.DataFrame], permutations: int) -> dict:
    """Run the shuffled-label placebo (a check of the splits) and the no-event placebo."""
    result: dict = {}
    rng = np.random.default_rng(714_007)
    for world in ("bank", "g2"):
        frame = tables[world].filter(pl.col("k") == 14).sort("pass_id", "event")
        x = matrix(frame=frame, columns=HOST)
        groups, months = frame["pass_id"].to_numpy(), frame["month"].to_numpy()
        label = frame["label"].to_numpy()
        scores = []
        for _ in range(permutations):
            shuffled = label.copy()
            for p in np.unique(groups):
                idx = np.flatnonzero(groups == p)
                shuffled[idx] = rng.permutation(label[idx])
            pred = model.out_of_fold(x=x, y=shuffled.astype(float), months=months)
            scores.append(model.auroc(label=shuffled, score=pred))
        result[f"{world}/shuffled_labels"] = {
            "mean": float(np.mean(scores)), "min": float(np.min(scores)),
            "max": float(np.max(scores)), "n": permutations,
            "within_0.03": bool(abs(np.mean(scores) - 0.5) <= TOLERANCE),
        }  # fmt: skip
        # the no-event placebo: the classifier fitted on genuine events scores the real panel
        x_np = matrix(frame=frame, columns=[f"np_{n}" for n in HOST])
        pred = np.full(len(label), np.nan)
        for month in np.unique(months):
            test = months == month
            fitted = model.fit_logistic(
                x=x[~test], y=label[~test].astype(float), blocks=months[~test]
            )
            pred[test] = fitted.proba(x=x_np[test])
        result[f"{world}/no_event_placebo"] = {"auroc": model.auroc(label=label, score=pred)}
    return result


def sensitivity(*, tables: dict[str, pl.DataFrame], out: Path) -> dict:
    """Return the pre-registered sensitivity readings (long moves dropped, kinds, mirror states)."""
    result: dict = {}
    for world in ("bank", "g2"):
        for k in common.ORIGIN_DAYS:
            frame = tables[world].filter(pl.col("k") == k).sort("pass_id", "event")
            keep = ~((frame["truth"] == "switch") & (frame["band"] == "60-120")).to_numpy()
            sub = frame.filter(pl.Series(keep))
            preds = oof_for_sets(frame=sub, sets={"host": HOST}, tree=False)
            result[f"{world}/k{k}/drop_60_120"] = {
                "auroc": model.auroc(label=sub["label"].to_numpy(), score=preds["host"].to_numpy()),
                "n": sub.height,
            }
            full = pl.read_parquet(out / f"oof_{world}_k{k}.parquet")
            label = full["label"].to_numpy()
            for kind in full.filter(pl.col("label") == 1)["kind"].unique().to_list():
                sel = (full["kind"] == kind).to_numpy() | (label == 0)
                result[f"{world}/k{k}/kind/{kind}"] = {
                    "auroc_host": model.auroc(label=label[sel], score=full["host"].to_numpy()[sel]),
                    "n_pos": int((full["kind"] == kind).sum()),
                }
            for state in ("full", "partial", "hidden"):
                sel = (label == 1) | (full["mirror_state"] == state).to_numpy()
                if (full["mirror_state"] == state).any():
                    result[f"{world}/k{k}/mirror/{state}"] = {
                        "auroc_host": model.auroc(
                            label=label[sel], score=full["host"].to_numpy()[sel]
                        ),
                        "auroc_host_nbr": model.auroc(
                            label=label[sel], score=full["host_nbr"].to_numpy()[sel]
                        ),
                        "n_neg": int((full["mirror_state"] == state).sum()),
                    }
    return result


def coefficients(*, tables: dict[str, pl.DataFrame]) -> dict:
    """Return the standardised coefficient of each feature, fitted on all rows of each world."""
    result: dict = {}
    for world in ("bank", "g2"):
        for k in common.ORIGIN_DAYS:
            frame = tables[world].filter(pl.col("k") == k).sort("pass_id", "event")
            for name in ("host", "host_nbr", "host_old"):
                columns = FEATURE_SETS[name]
                fitted = model.fit_logistic(
                    x=matrix(frame=frame, columns=columns),
                    y=frame["label"].to_numpy().astype(float),
                    blocks=frame["month"].to_numpy(),
                )
                order = np.argsort(-np.abs(fitted.weight))
                result[f"{world}/k{k}/{name}"] = {
                    "penalty": fitted.penalty, "bias": fitted.bias,
                    "standardised": {columns[i]: float(fitted.weight[i]) for i in order},
                }  # fmt: skip
    return result


RAW_CUES: Final[dict[str, tuple[str, float]]] = {
    "steps_per_day": ("steps", 1.0),
    "high_fraction": ("high", -1.0),
    "flatness": ("flat", -1.0),
    "mirror_share": ("mirror_share", -1.0),
    "mirror_z": ("mirror_z", -1.0),
}
"""Raw single cues, with the sign that makes a higher score mean a permanent change of the kind
named by the plan's prediction: more steps, a lower high-level share, a less flat profile, and no
mirror (a mirror marks a temporary move)."""


def secondary(*, tables: dict[str, pl.DataFrame], draws: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return each cue alone and together, and the raw single cues, for each permanent kind.

    The classifier of a cue set is fitted on every event of a world (bank, or g2 with gas) and
    scores held-out fold months. Positives are one permanent kind. Negatives are all temporary
    moves, those still in force at the data cut (sustained), those with a metered mirror, or those
    with a hidden mirror. The raw cues are the feature values themselves, in the direction that the
    plan's prediction states, with no fitting. Every reading is secondary.

    Args:
        tables: The feature tables of the worlds `bank`, `g2` and `gas`.
        draws: The number of bootstrap draws.

    Returns:
        The fitted-cue table and the raw-cue table.
    """
    rows, raw_rows = [], []
    worlds = {"bank": tables["bank"], "g2_gas": pl.concat([tables["g2"], tables["gas"]])}
    for world, big in worlds.items():
        for k in common.ORIGIN_DAYS:
            frame = big.filter(pl.col("k") == k).sort("pass_id", "event")
            preds = oof_for_sets(frame=frame, sets=CUE_SETS, tree=False)
            label = frame["label"].to_numpy()
            months, passes = frame["month"].to_numpy(), frame["pass_id"].to_numpy()
            kinds = frame["kind"].to_numpy()
            state = frame["mirror_state"].to_numpy()
            temporary = label == 0
            negatives = {
                "all_temporary": temporary,
                "sustained": temporary & frame["in_force"].to_numpy(),
                "metered_mirror": temporary & np.isin(state, MIRRORED),
                "hidden_mirror": temporary & (state == "hidden"),
            }
            boot = model.bootstrap_draws(
                months=months, passes=passes, n_draws=draws, seed=714_100 + k
            )
            scores = {cue: preds[cue].to_numpy() for cue in CUE_SETS}
            raw = {
                name: sign * frame[column].to_numpy().astype(float)
                for name, (column, sign) in RAW_CUES.items()
            }
            for kind in np.unique(kinds[label == 1]):
                positive = kinds == kind
                for neg_name, neg in negatives.items():
                    sel = positive | neg
                    sub = [d[sel[d]] for d in boot]
                    for cue, table in (*((c, scores[c]) for c in CUE_SETS), *raw.items()):
                        values = model.bootstrap_auroc(label=positive, score=table, draws=sub)
                        low, high = model.interval(values=values, confidence=SECONDARY_CONFIDENCE)
                        entry = {
                            "world": world, "k": k, "kind": kind, "negatives": neg_name,
                            "cue": cue,
                            "auroc": model.auroc(label=positive[sel], score=table[sel]),
                            "low95": low, "high95": high,
                            "n_pos": int(positive.sum()), "n_neg": int(neg.sum()),
                        }  # fmt: skip
                        (rows if cue in CUE_SETS else raw_rows).append(entry)
    return pl.DataFrame(rows), pl.DataFrame(raw_rows)


def transfer_reading(*, tables: dict[str, pl.DataFrame]) -> dict:
    """Fit on one world's other months and score the other world's month (secondary).

    The classifier that scores month m of the destination world never saw month m of either world,
    but it saw the same Primaries, which the two worlds share.
    """
    result = {}
    for src, dst in (("bank", "g2"), ("g2", "bank")):
        for k in common.ORIGIN_DAYS:
            a = tables[src].filter(pl.col("k") == k).sort("pass_id", "event")
            b = tables[dst].filter(pl.col("k") == k).sort("pass_id", "event")
            score = model.transfer_scores(
                x_source=matrix(frame=a, columns=HOST),
                y_source=a["label"].to_numpy().astype(float),
                months_source=a["month"].to_numpy(), x_dest=matrix(frame=b, columns=HOST),
                months_dest=b["month"].to_numpy(),
            )  # fmt: skip
            result[f"{src}_to_{dst}/k{k}"] = model.auroc(label=b["label"].to_numpy(), score=score)
    return result


def outcome_table(*, result: dict, flags: dict[str, bool]) -> str:
    """Return the outcome table in Markdown: the six contrasts, the companions, and the outcome."""
    c = result["contrasts"]
    lines = [
        "| Contrast | Quantity | Estimate | Interval (99.17%) |",
        "|---|---|---|---|",
    ]
    names = {
        "C1": "host rows, AUROC at 7 days, second generator",
        "C2": "host rows, AUROC at 14 days, second generator",
        "C3": "host rows, AUROC at 28 days, second generator",
        "C4": "gain at 14 days from six older rows, second generator",
        "C5": "gain at 14 days from neighbour rows, metered-mirror events, second generator",
        "C6": "host rows at 14 days, bank minus second generator",
    }
    for key, text in names.items():
        value = c[key]
        est = value.get("auroc", value.get("diff"))
        lines.append(f"| {key} | {text} | {est:.3f} | [{value['low']:.3f}, {value['high']:.3f}] |")
    lines += [
        "",
        "Sign-only AUROC (score is the sign of the measured change), beside C1 to C3 and C6:",
        "",
    ]
    lines += ["| World and origin | Sign-only AUROC | Interval (99.17%) |", "|---|---|---|"]
    for key, value in result["sign_only"].items():
        lines.append(
            f"| {key} | {value['auroc']:.3f} | [{value['low']:.3f}, {value['high']:.3f}] |"
        )
    lines += [
        "",
        (
            "Site-prior AUROC (the Primary alone, from the other fold months' labels) and "
            "the AUROC within each Primary (95% interval, secondary):"
        ),
        "",
        "| Reading | Value | Interval (95%) |",
        "|---|---|---|",
    ]
    for key, value in result["site_prior"].items():
        if isinstance(value, dict):
            lines.append(
                f"| {key} | {value['auroc']:.3f} | [{value['low95']:.3f}, {value['high95']:.3f}] |"
            )
        else:
            lines.append(f"| {key} | {value:.3f} | |")
    lines += [
        "",
        (
            "Precedence: `inconclusive_pending_audit` overrides every outcome. Outcome 3 applies "
            "only when the second generator does not clear 0.70."
        ),
        "",
        "| Decision flag | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in flags.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    draws = 200 if args.smoke else DRAWS
    out = common.DATA_DIR / ("results_smoke" if args.smoke else "results")
    out.mkdir(parents=True, exist_ok=True)
    tables = {w: load_world(world=w, smoke=args.smoke) for w in ("bank", "g2", "gas")}
    for world, frame in tables.items():
        print(world, frame.height, "rows")
    result = primary(tables=tables, draws=draws, out=out, tree=True)
    placebo = placebos(tables=tables, permutations=3 if args.smoke else PERMUTATIONS)
    flags = decisions(result=result, placebo=placebo)
    (out / "primary_contrasts.json").write_text(json.dumps({**result, "flags": flags}, indent=1))
    (out / "placebos.json").write_text(json.dumps(placebo, indent=1))
    (out / "sensitivity.json").write_text(
        json.dumps(
            {**sensitivity(tables=tables, out=out), "transfer": transfer_reading(tables=tables)},
            indent=1,
        )
    )
    (out / "coefficients.json").write_text(json.dumps(coefficients(tables=tables), indent=1))
    fitted, raw = secondary(tables=tables, draws=100 if args.smoke else SECONDARY_DRAWS)
    fitted.write_csv(out / "secondary_cues_by_kind.csv")
    raw.write_csv(out / "raw_cues_by_kind.csv")
    frame = tables["g2"].filter(pl.col("k") == 14).sort("pass_id", "event")
    preds = pl.read_parquet(out / "oof_g2_k14.parquet")
    model.calibration_table(
        label=frame["label"].to_numpy(), proba=preds["host"].to_numpy()
    ).write_csv(out / "calibration.csv")
    text = outcome_table(result=result, flags=flags)
    (out / "outcome_table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

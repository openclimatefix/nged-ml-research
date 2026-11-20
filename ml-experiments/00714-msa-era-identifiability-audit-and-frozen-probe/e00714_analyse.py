"""Experiment 714: fit the classifiers, score the six contrasts and the placebos, write the tables.

    .venv/bin/python ml-experiments/00714-*/e00714_analyse.py [--smoke]

Reads `features/{bank,g2,gas}.parquet` (written by `e00714_build.py`) and writes under
`results/`: `oof_<world>_k<k>.parquet` (out-of-fold probabilities by feature set),
`primary_contrasts.json` (the six pre-registered contrasts), `placebos.json`,
`sensitivity.json`, `coefficients.json`, `secondary_cues_by_kind.csv`, `calibration.csv`, and
`outcome_table.md`. The pre-registered contrasts use the worlds `bank` and `g2` only. The world
`gas` (intermittent steps) enters the secondary per-cue and per-kind readings only.
`--smoke` uses 200 bootstrap draws and 3 label permutations and writes under `results_smoke/`.
"""

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
RW: Final[list[str]] = ["rw_mirror_share", "rw_mirror_corr"]
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
SECONDARY_CONFIDENCE: Final[float] = 0.95
TOLERANCE: Final[float] = 0.03


def load_world(*, world: str) -> pl.DataFrame:
    """Read a world's feature table, or raise if it is missing."""
    return pl.read_parquet(FEATURE_DIR / f"{world}.parquet")


def matrix(*, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the feature matrix of the named columns, NaN where missing."""
    return frame.select(columns).cast(pl.Float64).to_numpy()


def oof_for_sets(*, frame: pl.DataFrame, sets: dict[str, list[str]], tree: bool) -> pl.DataFrame:
    """Return out-of-fold probabilities of each feature set for the rows of `frame`."""
    y = frame["label"].to_numpy().astype(float)
    groups, months = frame["pass_id"].to_numpy(), frame["month"].to_numpy()
    out = {}
    for name, columns in sets.items():
        x = matrix(frame=frame, columns=columns)
        out[name] = model.out_of_fold(x=x, y=y, groups=groups, months=months)
        if tree and name in ("host", "host_nbr"):
            out[f"{name}_tree"] = model.out_of_fold(
                x=x, y=y, groups=groups, months=months, model="tree"
            )
    return pl.DataFrame(out)


def mirrored_mask(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the rows that are permanent changes or temporary moves with a metered mirror."""
    return (frame["label"] == 1).to_numpy() | frame["mirror_state"].is_in(MIRRORED).to_numpy()


def hidden_mask(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the rows that are permanent changes or temporary moves with a hidden mirror."""
    return (frame["label"] == 1).to_numpy() | (frame["mirror_state"] == "hidden").to_numpy()


def point_and_interval(
    *, label: np.ndarray, score: np.ndarray, draws: list[np.ndarray]
) -> dict[str, float]:
    """Return an AUROC with its 99.17% interval over paired bootstrap draws."""
    boot = model.bootstrap_auroc(label=label, score=score, draws=draws)
    low, high = model.interval(values=boot)
    return {"auroc": model.auroc(label=label, score=score), "low": low, "high": high}


def contrast(
    *, label: np.ndarray, a: np.ndarray, b: np.ndarray, draws: list[np.ndarray]
) -> dict[str, float]:
    """Return AUROC(a) - AUROC(b) with its 99.17% interval over paired draws."""
    diff = model.bootstrap_auroc(label=label, score=a, draws=draws) - model.bootstrap_auroc(
        label=label, score=b, draws=draws
    )
    low, high = model.interval(values=diff)
    return {
        "diff": model.auroc(label=label, score=a) - model.auroc(label=label, score=b),
        "low": low, "high": high,
    }  # fmt: skip


def primary(*, tables: dict[str, pl.DataFrame], draws: int, out: Path, tree: bool) -> dict:
    """Fit the primary sets on `bank` and `g2` and compute the six contrasts."""
    result: dict = {"auroc": {}, "contrasts": {}}
    oof: dict[tuple[str, int], tuple[pl.DataFrame, pl.DataFrame]] = {}
    boots: dict[tuple[str, int], list[np.ndarray]] = {}
    for world in ("bank", "g2"):
        for k in common.ORIGIN_DAYS:
            frame = tables[world].filter(pl.col("k") == k).sort("pass_id", "event")
            preds = oof_for_sets(
                frame=frame, sets={n: FEATURE_SETS[n] for n in PRIMARY_SETS}, tree=tree
            )
            oof[(world, k)] = (frame, preds)
            preds.with_columns(
                frame.select("pass_id", "month", "kind", "label", "mirror_state")
            ).write_parquet(out / f"oof_{world}_k{k}.parquet")
            boots[(world, k)] = model.bootstrap_draws(
                months=frame["month"].to_numpy(), passes=frame["pass_id"].to_numpy(),
                n_draws=draws, seed=714_000 + k,
            )  # fmt: skip
            label = frame["label"].to_numpy()
            for name in preds.columns:
                result["auroc"][f"{world}/k{k}/{name}"] = point_and_interval(
                    label=label, score=preds[name].to_numpy(), draws=boots[(world, k)]
                )
    g2_frames = {k: oof[("g2", k)] for k in common.ORIGIN_DAYS}
    for name, k in (("C1", 7), ("C2", 14), ("C3", 28)):
        result["contrasts"][name] = result["auroc"][f"g2/k{k}/host"]
    frame, preds = g2_frames[14]
    label = frame["label"].to_numpy()
    result["contrasts"]["C4"] = contrast(
        label=label,
        a=preds["host_old"].to_numpy(),
        b=preds["host"].to_numpy(),
        draws=boots[("g2", 14)],
    )
    mask = mirrored_mask(frame=frame)
    sub_draws = [d[mask[d]] for d in boots[("g2", 14)]]
    result["contrasts"]["C5"] = contrast(
        label=label, a=preds["host_nbr"].to_numpy(), b=preds["host"].to_numpy(), draws=sub_draws
    )
    result["contrasts"]["C5_auroc_host_nbr_mirrored"] = point_and_interval(
        label=label, score=preds["host_nbr"].to_numpy(), draws=sub_draws
    )
    result["contrasts"]["C5_auroc_host_nbr_hidden"] = point_and_interval(
        label=label,
        score=preds["host_nbr"].to_numpy(),
        draws=[d[hidden_mask(frame=frame)[d]] for d in boots[("g2", 14)]],
    )
    result["contrasts"]["C5_gain_hidden"] = contrast(
        label=label, a=preds["host_nbr"].to_numpy(), b=preds["host"].to_numpy(),
        draws=[d[hidden_mask(frame=frame)[d]] for d in boots[("g2", 14)]],
    )  # fmt: skip
    bank_frame, bank_preds = oof[("bank", 14)]
    rng = np.random.default_rng(714_006)
    b_boot = model.bootstrap_auroc(
        label=bank_frame["label"].to_numpy(),
        score=bank_preds["host"].to_numpy(),
        draws=boots[("bank", 14)],
    )
    g_boot = model.bootstrap_auroc(
        label=label, score=preds["host"].to_numpy(), draws=boots[("g2", 14)]
    )
    rng.shuffle(g_boot)  # independent worlds: pair the draws at random
    low, high = model.interval(values=b_boot - g_boot)
    result["contrasts"]["C6"] = {
        "diff": result["auroc"]["bank/k14/host"]["auroc"] - result["auroc"]["g2/k14/host"]["auroc"],
        "low": low, "high": high,
    }  # fmt: skip
    return result


def decisions(*, result: dict) -> dict[str, bool | None]:
    """Apply the pre-registered decision rule to the six contrasts."""
    c = result["contrasts"]
    auroc = result["auroc"]
    return {
        "host_rows_enough": c["C1"]["low"] >= 0.70 and c["C2"]["low"] >= 0.70,
        "host_rows_not_enough": c["C1"]["high"] < 0.70 or c["C2"]["high"] < 0.70,
        "injector_artefact": c["C6"]["low"] > 0.10,
        "neighbour_rows_pass": c["C5"]["low"] >= 0.05
        and c["C5_auroc_host_nbr_mirrored"]["auroc"] >= 0.85,
        "known_answer_pass": auroc["bank/k14/known_answer"]["auroc"] > 0.95,
        "rewired_within_0.03": abs(
            auroc["g2/k14/host_nbr_rewired"]["auroc"] - auroc["g2/k14/host"]["auroc"]
        )
        <= TOLERANCE,
    }


def placebos(*, tables: dict[str, pl.DataFrame], permutations: int, out: Path) -> dict:
    """Run the shuffled-label placebo and the no-event placebo."""
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
            pred = model.out_of_fold(x=x, y=shuffled.astype(float), groups=groups, months=months)
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
                x=x[~test], y=label[~test].astype(float), groups=groups[~test]
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
                    groups=frame["pass_id"].to_numpy(),
                )
                order = np.argsort(-np.abs(fitted.weight))
                result[f"{world}/k{k}/{name}"] = {
                    "penalty": fitted.penalty, "bias": fitted.bias,
                    "standardised": {columns[i]: float(fitted.weight[i]) for i in order},
                }  # fmt: skip
    return result


def secondary(*, tables: dict[str, pl.DataFrame], draws: int) -> pl.DataFrame:
    """Return each cue alone and together, for each permanent kind against the temporary moves.

    The classifier of a cue set is fitted on every event of a world (bank, or g2 with gas) and
    scores held-out fold months. Positives are one permanent kind. Negatives are all temporary
    moves, those with a metered mirror, or those with a hidden mirror. Every reading is secondary.
    """
    rows = []
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
                "metered_mirror": temporary & np.isin(state, MIRRORED),
                "hidden_mirror": temporary & (state == "hidden"),
            }
            boot = model.bootstrap_draws(
                months=months, passes=passes, n_draws=draws, seed=714_100 + k
            )
            for kind in np.unique(kinds[label == 1]):
                for neg_name, neg in negatives.items():
                    sel = (kinds == kind) | neg
                    for cue in CUE_SETS:
                        score = preds[cue].to_numpy()
                        values = model.bootstrap_auroc(
                            label=(kinds == kind),
                            score=score,
                            draws=[d[sel[d]] for d in boot],
                        )
                        low, high = model.interval(values=values, confidence=SECONDARY_CONFIDENCE)
                        rows.append(
                            {
                                "world": world, "k": k, "kind": kind, "negatives": neg_name,
                                "cue": cue,
                                "auroc": model.auroc(label=(kinds == kind)[sel], score=score[sel]),
                                "low95": low, "high95": high,
                                "n_pos": int((kinds == kind).sum()), "n_neg": int(neg.sum()),
                            }
                        )  # fmt: skip
    return pl.DataFrame(rows)


def transfer_reading(*, tables: dict[str, pl.DataFrame]) -> dict:
    """Fit on one generator's events and score the other's (secondary)."""
    result = {}
    for src, dst in (("bank", "g2"), ("g2", "bank")):
        for k in common.ORIGIN_DAYS:
            a = tables[src].filter(pl.col("k") == k).sort("pass_id", "event")
            b = tables[dst].filter(pl.col("k") == k).sort("pass_id", "event")
            score = model.transfer_scores(
                x_train=matrix(frame=a, columns=HOST), y_train=a["label"].to_numpy().astype(float),
                groups=a["pass_id"].to_numpy(), x_test=matrix(frame=b, columns=HOST),
            )  # fmt: skip
            result[f"{src}_to_{dst}/k{k}"] = model.auroc(label=b["label"].to_numpy(), score=score)
    return result


def outcome_table(*, result: dict, flags: dict) -> str:
    """Return the outcome table in Markdown: the six contrasts with their intervals."""
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
    lines += ["", "| Decision flag | Value |", "|---|---|"]
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
    tables = {w: load_world(world=w) for w in ("bank", "g2", "gas")}
    for world, frame in tables.items():
        print(world, frame.height, "rows")
    result = primary(tables=tables, draws=draws, out=out, tree=True)
    flags = decisions(result=result)
    (out / "primary_contrasts.json").write_text(json.dumps({**result, "flags": flags}, indent=1))
    (out / "placebos.json").write_text(
        json.dumps(
            placebos(tables=tables, permutations=3 if args.smoke else PERMUTATIONS, out=out),
            indent=1,
        )
    )
    (out / "sensitivity.json").write_text(
        json.dumps(
            {**sensitivity(tables=tables, out=out), "transfer": transfer_reading(tables=tables)},
            indent=1,
        )
    )
    (out / "coefficients.json").write_text(json.dumps(coefficients(tables=tables), indent=1))
    secondary(tables=tables, draws=100 if args.smoke else SECONDARY_DRAWS).write_csv(
        out / "secondary_cues_by_kind.csv"
    )
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

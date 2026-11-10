"""Experiment 692: score the reviewer's answers against the answer key.

    uv run --no-sync python e00692_score.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "00692-event-table-reviewer-calibration-gate"
N_BOOT = 4000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of a share."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def load_answers(folder: Path) -> pl.DataFrame:
    """Every JSON line in the folder, last answer per figure."""
    rows = []
    for f in sorted(folder.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pl.DataFrame(rows).unique(subset="figure", keep="last") if rows else pl.DataFrame()


def main() -> None:
    """Print the calibration-gate numbers and write `scores.json`."""
    key = pl.read_parquet(OUT / "answer_key_private" / "key.parquet")
    cand = pl.read_parquet(OUT / "answer_key_private" / "candidates.parquet").filter(
        pl.col("kind") == "sym"
    )
    ans = load_answers(OUT / "answers" / "calibration")
    cal = key.filter(pl.col("set") == "calibration").join(ans, on="figure", how="left")
    res: dict = {"n_figures": cal.height, "n_answered": int(cal["verdict"].is_not_null().sum())}
    cal = cal.with_columns(
        pl.col("verdict").eq("transfer").fill_null(False).alias("called"),
        (pl.col("stratum").is_in(["true", "verified_real"])).alias("positive"),
    )
    tp = cal.filter(pl.col("called") & pl.col("positive")).height
    fp = cal.filter(pl.col("called") & ~pl.col("positive")).height
    npos = cal.filter(pl.col("positive")).height
    res["tp"], res["fp"], res["n_positive"] = tp, fp, npos
    res["precision"] = tp / (tp + fp) if tp + fp else float("nan")
    res["precision_ci"] = wilson(tp, tp + fp)
    res["recall"] = tp / npos
    res["recall_ci"] = wilson(tp, npos)
    called_pos = cal.filter(pl.col("called") & pl.col("positive"))
    right_dir = called_pos.filter(pl.col("direction") == "as_drawn").height
    res["direction_right"] = (right_dir, called_pos.height)
    ver = cal.filter(pl.col("stratum") == "verified_real")
    res["verified_called"] = {r["donor"] + "_" + r["recipient"]: r["verdict"] for r in ver.to_dicts()}
    by = (
        cal.group_by("stratum")
        .agg(pl.len().alias("n"), pl.col("called").sum().alias("called"))
        .sort("stratum")
    )
    res["by_stratum"] = by.to_dicts()
    # Natural mix of the 40 worlds' candidates.
    n_true = cand.filter(pl.col("true")).height
    n_false_plain = cand.filter(~pl.col("true") & ~pl.col("hard")).height
    n_false_hard = cand.filter(~pl.col("true") & pl.col("hard")).height
    res["natural"] = {"true": n_true, "false_plain": n_false_plain, "false_hard": n_false_hard}
    res["false_share_before"] = (n_false_plain + n_false_hard) / cand.height

    def stratum_flags(name: str) -> np.ndarray:
        return cal.filter(pl.col("stratum") == name)["called"].to_numpy().astype(float)

    pools = {
        "true": np.concatenate([stratum_flags("true"), stratum_flags("verified_real")]),
        "plain": stratum_flags("false_random"),
        "hard": stratum_flags("hard_negative_injected_placebo"),
    }
    rng = np.random.default_rng(692)

    def share(p: dict[str, np.ndarray]) -> float:
        tpr, fpr_p, fpr_h = (p[k].mean() if len(p[k]) else 0.0 for k in ("true", "plain", "hard"))
        fa = n_false_plain * fpr_p + n_false_hard * fpr_h
        return float(fa / (fa + n_true * tpr)) if fa + n_true * tpr else float("nan")

    res["false_share_after"] = share(pools)
    boots = [
        share({k: rng.choice(v, size=len(v)) for k, v in pools.items() if len(v)})
        for _ in range(N_BOOT)
    ]
    res["false_share_after_ci"] = (
        float(np.nanpercentile(boots, 2.5)),
        float(np.nanpercentile(boots, 97.5)),
    )
    res["kept_true_share"] = float(pools["true"].mean())
    res["wrong"] = (
        cal.filter(pl.col("called") != pl.col("positive"))
        .select(["figure", "stratum", "verdict", "confidence", "reason"])
        .to_dicts()
    )
    # Realness.
    ra = load_answers(OUT / "answers" / "realness")
    if ra.height:
        rk = key.filter(pl.col("set") == "realness").join(ra, on="figure", how="inner")
        rk = rk.with_columns((pl.col("source") == "real").alias("is_real"))
        said_real = rk["answer"].eq("real")
        real = rk["is_real"]
        acc = float((said_real == real).mean())  # ty: ignore[invalid-argument-type]
        tpr_r = float(said_real.filter(real).mean())  # ty: ignore[invalid-argument-type]
        tnr = float((~said_real).filter(~real).mean())  # ty: ignore[invalid-argument-type]
        res["realness"] = {
            "n": rk.height,
            "n_real": int(real.sum()),
            "accuracy": acc,
            "balanced_accuracy": (tpr_r + tnr) / 2,
            "real_called_real": tpr_r,
            "injected_called_injected": tnr,
        }
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "wrong"}, indent=1, default=float))


if __name__ == "__main__":
    main()

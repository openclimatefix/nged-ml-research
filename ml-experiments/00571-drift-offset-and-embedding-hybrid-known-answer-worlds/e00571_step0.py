# ruff: noqa: E402
# ruff: noqa: E501, ANN001
"""Experiment 571, step 0: do the joint hybrid's embedding and offset compete for the same slow variance?

Reads experiment 565's stored hybrid (arm H) fits, one per fold month and seed (day 14, the newest
fit of the fold). For each Primary and each monthly knot, computes (a) the embedding's level path:
the mean change in the mixture location (standardised units) when the drift embedding moves from its
Primary-average (mean of the Primary's knots) to the knot's value, over a fixed random sample of the
fold's test rows; and (b) the offset's level path: the offset knot minus its Primary mean. The offset
adds the same amount to all three component means, so it is a location shift exactly. Reports the
correlation of the two paths and the share of slow variance each takes.

Usage: python e00571_step0.py
"""

import json
import sys

sys.argv = ["e00565_fit.py", "fit", "--arm", "H"]  # the H arm of experiment 565, parsed on import

import e00397_models as base  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00559_model_drift as mdr  # ty: ignore[unresolved-import]
import e00565_fit as f565  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00571-drift-offset-and-embedding-hybrid-known-answer-worlds"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME
ROWS_PER_PRIMARY = 4000
SEEDS = (0, 1)
DAY = 14


def location(*, model, x, p, vec) -> torch.Tensor:
    """Return the mixture location of rows with every row's drift embedding set to `vec` (rows, 4)."""
    model.drift = lambda tau, primary: vec
    with torch.no_grad():
        out = mdr.ModularMdnDrift.forward(model, x, p).double()
    log_w, mean, _ = base.mixture_parts(out=out, components=m.COMPONENTS)
    return (log_w.exp() * mean).sum(-1)


def main() -> None:
    """Compute and print the level-path comparison for every fold month and seed."""
    clean = c.clean_stack()
    sites = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR
        / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
    )
    label = dict(zip(sites["primary"].to_list(), sites["label"].to_list(), strict=True))
    group = dict(zip(sites["primary"].to_list(), sites["group"].to_list(), strict=True))
    rows = []
    for month in c.MONTHS:
        mr = f565.build_month(month=month, clean=clean)
        test = f565.build_test(month=month, standardiser=mr.base.standardiser)
        n_p = len(mr.base.target_mean)
        rng = np.random.default_rng(0)
        pick = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(test.primary == q),
                    min(ROWS_PER_PRIMARY, (test.primary == q).sum()),
                    replace=False,
                )
                for q in range(n_p)
            ]
        )
        x = torch.from_numpy(test.x[pick]).float().cuda()
        p = torch.from_numpy(test.primary[pick]).long().cuda()
        for seed in SEEDS:
            model = f565.new_model(n_primaries=n_p).cuda()
            model.load_state_dict(
                torch.load(
                    f565.model_file(tag="H", month=month, day=DAY, seed=seed), map_location="cuda"
                )
            )
            model.eval()
            knots = model.all_knots().detach()  # (P, K, 4)
            off = model.all_offsets().detach()  # (P, K)
            for q in range(n_p):
                sel = p == q
                xs, ps = x[sel], p[sel]
                avg = knots[q].mean(0, keepdim=True).expand(len(xs), -1)
                base_loc = location(model=model, x=xs, p=ps, vec=avg).mean()
                emb_path = np.array(
                    [
                        float(
                            location(
                                model=model, x=xs, p=ps, vec=knots[q, j][None].expand(len(xs), -1)
                            ).mean()
                            - base_loc
                        )
                        for j in range(knots.shape[1])
                    ]
                )
                off_path = (off[q] - off[q].mean()).cpu().numpy().astype(np.float64)
                for span, sl in (("all", slice(None)), ("last12", slice(-12, None))):
                    e, o = emb_path[sl], off_path[sl]
                    e, o = e - e.mean(), o - o.mean()
                    ve, vo = float((e**2).mean()), float((o**2).mean())
                    rows.append(
                        {
                            "month": month,
                            "seed": seed,
                            "primary": q,
                            "label": label[q],
                            "group": group[q],
                            "span": span,
                            "corr": float(np.corrcoef(e, o)[0, 1])
                            if ve > 0 and vo > 0
                            else float("nan"),
                            "var_emb": ve,
                            "var_off": vo,
                            "share_emb": ve / (ve + vo) if ve + vo > 0 else float("nan"),
                            "var_sum": float(((e + o) ** 2).mean()),
                        }
                    )
        del mr, test, x, p
        torch.cuda.empty_cache()
        print(month, "done", flush=True)
    frame = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.write_csv(OUT / "step0_paths.csv")
    summary = (
        frame.group_by("group", "span")
        .agg(
            pl.col("corr").mean().alias("mean_corr"),
            pl.col("corr").median().alias("median_corr"),
            (pl.col("corr") >= 0.5).mean().alias("share_corr_ge_0.5"),
            (pl.col("corr").abs() < 0.3).mean().alias("share_abs_corr_lt_0.3"),
            pl.col("share_emb").mean().alias("mean_share_emb"),
            pl.col("var_emb").mean(),
            pl.col("var_off").mean(),
            pl.len().alias("n"),
        )
        .sort("group", "span")
    )
    print(summary)
    (OUT / "step0_summary.json").write_text(json.dumps(summary.to_dicts(), indent=1))


if __name__ == "__main__":
    main()

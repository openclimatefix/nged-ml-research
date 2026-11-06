# ruff: noqa: E501, ANN001, ANN202, E402, D103, PLR0915, PLR0912, PERF401, ANN201
"""Experiment 602: staged probe of the modular network's three encoders (GPU, inference only).

For one network (P5: experiment 574; C1: experiment 584; A1: experiment 591) and each stored daily
refit, forecast the day-ahead rows (lead 24 to 48 h) three ways through the trained mixer, with the
activated embeddings of the encoders that are not yet "switched on" replaced by their mean over a
sample of the fold's fine-tuning rows (clean lags; a probe that is off-distribution):

1. `s1`: datetime encoder only (weather and lag embeddings = mean);
2. `s2`: datetime + weather (lag embedding = mean);
3. `s3`: all three encoders (the full network; checked against the network's own forward pass).

Cases: `real` (observed lags; the December 2025 pair and the following month), `free` (observed lags
in a month with no verified event), `inject` (experiment 581's injected bank, passes 0 to 23, in
2026-03). Usage: python e00602_probe.py NET CASE [--seeds 0]
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("net", choices=("P5", "C1", "A1"))
_p.add_argument("case", choices=("real", "free", "inject"))
_p.add_argument("--seeds", nargs="+", type=int, default=[0])
_p.add_argument("--months", nargs="+", default=None)
MINE = _p.parse_args()
sys.argv = ["e00574_fit.py", "fit", "--arm", "P5", "--seeds", *map(str, MINE.seeds)]

import e00510_common as c  # ty: ignore[unresolved-import]
import e00574_fit as e  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00602-forecasts-by-encoder-stage-of-the-modular-network"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME / "probe" / MINE.net
W574 = e.W
if MINE.net == "C1":
    e.W = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6" / "c1" / "work"
elif MINE.net == "A1":
    import e00591_model as mymodel  # ty: ignore[unresolved-import]

    e.W = (
        ML_EXPERIMENTS_DATA_DIR
        / "00591-shared-lag-encoder-auxiliary-shift-depth-loss"
        / "A1"
        / "work"
    )

    def _new_model(*, n_primaries: int, stage: str = "ft"):
        model = mymodel.AuxLagMdn(
            n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, kind="shared"
        )
        model.table = e.CTX.get(f"table_{stage}")
        return model

    e.new_model = _new_model
if MINE.net != "P5":
    e.stats_file = lambda *, month: W574 / "stats" / e.SPEC.b1 / f"{month}.npz"

LEAD_BAND = (24.0, 48.0)
N_MEAN_ROWS = 200_000
LEVELS = {"q05": 1, "q50": 5, "q95": 9}


def embeddings(*, model, x, p):
    """Return the activated weather, datetime, and lag embeddings and the Primary embedding of rows `x`."""
    m = e.m
    emb = model.embedding(p)
    weather_in = [x[:, m.WEATHER_COLS], emb]
    if model.n_b1:
        weather_in.append(x[:, e.mod.b1_cols(n_b1=model.n_b1)])
    act = torch.nn.functional.gelu
    hw = act(model.weather(torch.cat(weather_in, dim=-1)))
    hd = act(model.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
    hl = act(model.lag_embedding(x=x, primary=p, emb=emb))
    return hw, hd, hl, emb


def mixer_out(*, model, x, p, hw, hd, hl, emb):
    """Return the nine raw outputs (offset added to the means) for the given embeddings."""
    m = e.m
    core = torch.cat([hw, hd, hl, emb], dim=-1)
    raw = model.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))
    off = model.offset(x[:, e.mod.TAU], p)
    k = m.COMPONENTS
    return torch.cat([raw[:, :k], raw[:, k : 2 * k] + off, raw[:, 2 * k :]], dim=-1)


def staged_quantiles(*, model, x, p, means) -> dict[str, torch.Tensor]:
    """Return standardised q05, q50, q95 of the stages and the leave-one-encoder-out forecasts.

    `s1`: datetime only. `w`: weather only. `s2`: datetime and weather. `s3`: all three (the full
    network). `loo_w`, `loo_d`, `loo_l`: the full network with one encoder's embedding set to its mean.
    """
    hw, hd, hl, emb = embeddings(model=model, x=x, p=p)
    mw, md, ml = (v.expand(len(x), -1) for v in means)
    out = {}
    stages = {
        "s1": (mw, hd, ml), "w": (hw, md, ml), "s2": (hw, hd, ml), "s3": (hw, hd, hl),
        "loo_w": (mw, hd, hl), "loo_d": (hw, md, hl), "loo_l": (hw, hd, ml),
    }  # fmt: skip
    for name, (a, b, c_) in stages.items():
        raw = mixer_out(model=model, x=x, p=p, hw=a, hd=b, hl=c_, emb=emb)
        log_w, mean, std = e.base.mixture_parts(out=raw.double(), components=e.m.COMPONENTS)
        q = e.base.mixture_quantiles(log_w=log_w, mean=mean, std=std)
        out[name] = q[:, [LEVELS["q05"], LEVELS["q50"], LEVELS["q95"]]]
    return out


def attributions(*, model, x, p) -> np.ndarray:
    """Return gradient times input of the mixture mean for the weather, datetime, and lag blocks.

    Each value sums over the block's dimensions: the signed contribution of that embedding to the
    standardised mixture mean. A sensitivity measure, not attention.
    """
    with torch.no_grad():
        hw, hd, hl, emb = embeddings(model=model, x=x, p=p)
    leaves = [v.detach().clone().requires_grad_(True) for v in (hw, hd, hl)]
    with torch.enable_grad():
        raw = mixer_out(model=model, x=x, p=p, hw=leaves[0], hd=leaves[1], hl=leaves[2], emb=emb)
        k = e.m.COMPONENTS
        log_w, mean, _ = e.base.mixture_parts(out=raw, components=k)
        y = (log_w.exp() * mean).sum(dim=-1).sum()
        y.backward()
    return torch.stack([(v.grad * v.detach()).sum(dim=-1) for v in leaves], dim=1).cpu().numpy()


def mean_embeddings(*, model, mr, pool: np.ndarray, device: str) -> tuple[torch.Tensor, ...]:
    """Return the mean activated (weather, datetime, lag) embeddings over a sample of `pool`.

    `pool` holds the cut's fine-tuning row indices.
    """
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(pool, size=min(N_MEAN_ROWS, len(pool)), replace=False))
    sums = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for a in range(0, len(idx), 50_000):
            sl = idx[a : a + 50_000]
            x = torch.from_numpy(mr.x[sl]).to(device)
            p = torch.from_numpy(mr.primary[sl]).to(device)
            emb = model.embedding(p)
            wi = [x[:, e.m.WEATHER_COLS], emb]
            if model.n_b1:
                wi.append(x[:, e.mod.b1_cols(n_b1=model.n_b1)])
            act = torch.nn.functional.gelu
            hs = (
                act(model.weather(torch.cat(wi, dim=-1))),
                act(model.datetime(torch.cat([x[:, e.m.DATETIME_COLS], emb], dim=-1))),
                act(model.lag_embedding(x=x, primary=p, emb=emb)),
            )
            for j, h in enumerate(hs):
                sums[j] = sums[j] + h.sum(0)
    return tuple((s / len(idx))[None, :] for s in sums)


def main(*, device: str = "cuda") -> None:
    clean = e.c.clean_stack()
    default = {"real": ["2025-12", "2026-01"], "free": ["2025-10"], "inject": ["2026-03"]}
    months = MINE.months or default[MINE.case]
    passes = list(range(bank.N_PASSES)) if MINE.case == "inject" else [-1]
    for month in months:
        cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
        mr = e.k.build_month(month=month, clean=clean)
        test = e.build_test(month=month, standardiser=mr.base.standardiser)
        assert test.x.shape[1] == mr.x.shape[1], (test.x.shape, mr.x.shape)
        start = e.c.month_start(month=month)
        start_idx = (e.c.us(t=start) - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
        lead = (test.target_idx - test.origin_idx) * 0.5
        in_band = (lead >= LEAD_BAND[0]) & (lead < LEAD_BAND[1])
        sl = slice(37, 45)
        for i, cut in enumerate(cuts):
            day = e.f.day_of(month=month, cut=cut)
            lo, hi = e.k.window_of(month=month, cadence=e.CADENCE, index=i)
            idx = np.flatnonzero((test.origin >= lo) & (test.origin < hi) & in_band)
            if not len(idx):
                continue
            for seed in e.ARGS.seeds:
                model = e.new_model(n_primaries=len(mr.base.target_mean)).to(device)
                model.load_state_dict(
                    torch.load(
                        e.model_file(tag=e.SPEC.tag, month=month, day=day, seed=seed),
                        map_location=device,
                    )
                )
                model.eval()
                model.table = e.CTX["table_ft"]
                refit = e.c.select_rows(rows=mr, cutoff=cut, half_life_days=e.SPEC.half_life)
                refit = e.embargoed(mr=mr, refit=refit, cut=cut)
                means = None
                for r in passes:
                    target = OUT / f"{MINE.case}_{month}_r{r:02d}_d{day}_s{seed}.parquet"
                    if target.exists():
                        continue
                    if means is None:
                        means = mean_embeddings(model=model, mr=mr, pool=refit.index, device=device)
                    x = test.x[idx].copy()
                    if r >= 0:
                        fixed, _ = bank.contaminate(clean=clean, month_start_idx=start_idx, r=r)
                        x[:, sl] = e.block_of(
                            clean=fixed, primary=test.primary[idx], target_idx=test.target_idx[idx],
                            origin_idx=test.origin_idx[idx],
                        )  # fmt: skip
                    prim = test.primary[idx]
                    mu, sd = mr.base.target_mean[prim], mr.base.target_std[prim]
                    cols: dict[str, np.ndarray] = {}
                    full_check = 0.0
                    attr = []
                    for a in range(0, len(idx), 25_000):
                        attr.append(
                            attributions(
                                model=model,
                                x=torch.from_numpy(x[a : a + 25_000]).to(device),
                                p=torch.from_numpy(prim[a : a + 25_000]).to(device),
                            )
                        )
                    attr = np.concatenate(attr) * sd[:, None]
                    for j, name in enumerate(("w", "d", "l")):
                        cols[f"g_{name}"] = attr[:, j]
                    with torch.no_grad():
                        parts: dict[str, list] = {}
                        for a in range(0, len(idx), 50_000):
                            xb = torch.from_numpy(x[a : a + 50_000]).to(device)
                            pb = torch.from_numpy(prim[a : a + 50_000]).to(device)
                            st = staged_quantiles(model=model, x=xb, p=pb, means=means)
                            ref = c.serving_quantiles(model=model, x=xb, p=pb, outlier=False)
                            full_check = max(
                                full_check, float((ref[:, [1, 5, 9]] - st["s3"]).abs().max())
                            )
                            for s, q in st.items():
                                parts.setdefault(s, []).append(q.float().cpu().numpy())
                    for s, ql in parts.items():
                        q = np.concatenate(ql) * sd[:, None] + mu[:, None]
                        q = np.sort(q, axis=1)
                        for j, name in enumerate(("q05", "q50", "q95")):
                            cols[f"{s}_{name}"] = q[:, j]
                    assert full_check < 1e-4, (
                        f"stage 3 differs from the network's forward pass: {full_check}"
                    )
                    frame = pl.DataFrame(
                        {
                            "primary": prim,
                            "origin_idx": test.origin_idx[idx],
                            "target_idx": test.target_idx[idx],
                            **cols,
                        }
                    )
                    e.atomic_frame(frame=frame, target=target)
                    print(
                        f"{MINE.net} {MINE.case} {month} r{r:02d} d{day} s{seed}: {len(idx)} rows, stage-3 check {full_check:.2e}",
                        flush=True,
                    )
                del model
        del mr, test
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, E501, E731, PERF401, PLR0915, PT018, PTH123
r"""Stage 0 of the 524 remaining arms: an adaptor in weather space.

Interim and unaudited. Stage 0 maps the ENS mean weather (the seven `mean_*` stage-1 inputs)
towards the ERA5 and CAMS weather at the same valid time, then frozen stage 1 reads the
corrected values. Pairs at a weekly cutoff: ERA5 variables with valid time <= cutoff - 5 days,
shortwave (CAMS) <= cutoff - 1 day, origin <= cutoff, valid time within 730 days. Form 1: per
variable and lead band (8 bands of 30 h) weighted least-squares bias and scale. Form 2: 2 x 32
multilayer perceptron on the 7 variables, lead / 240, sin and cos of the hour; residual. Arms:
`S0-A-F1`, `S0-A-F2` (stage 0 + frozen stage 1), `S0-S2` (stage 0 of the chosen form, then a
form-B stage 2 trained on cross-fitted stage-1 outputs of the corrected ENS rows).
"""

import argparse
import time

import e00397_prep as prep397  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as fit510  # ty: ignore[unresolved-import]
import e00524_rest_score as rs
import e00524_stage1 as s1
import e00524_stage2 as s2
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

WEATHER = prep397.WEATHER
NW = len(WEATHER)
SW = WEATHER.index("shortwave")
PRECIP = WEATHER.index("precipitation")
DAY = c.DAY_US
BINS = 8
BIN_H = 30.0
SEED_PREDS = rs.SEED_PREDS
_DENSE: dict = {}


def dense_targets(*, valid_us, primary, columns, standardiser) -> tuple[np.ndarray, list[int]]:
    """Return the ERA5 and CAMS weather at each row's valid time on the stage-1 input scale (NaN where missing)."""
    if not _DENSE:
        _DENSE.update(prep418.era5_dense_cache())
    sites = c.sites()
    idx = ((valid_us - c.prep422.GRID_START_US) // c.prep422.STEP_US).astype(np.int64)
    raw = np.full((len(idx), NW), np.nan, dtype=np.float64)
    for p, site in enumerate(sites):
        r = np.nonzero(primary == p)[0]
        if len(r):
            ok = (idx[r] >= 0) & (idx[r] < len(_DENSE[site]))
            raw[r[ok]] = _DENSE[site][idx[r[ok]]]
    raw[:, PRECIP] = np.log1p(np.clip(raw[:, PRECIP], 0, None))
    pos = [columns.index(f"mean_{w}") for w in WEATHER]
    t = (raw - standardiser.mean[pos]) / standardiser.std[pos]
    return np.clip(t, -prep397.INPUT_CLIP_SD, prep397.INPUT_CLIP_SD).astype(np.float32), pos


def pair_mask(*, t, valid_us, origin_us, cu) -> np.ndarray:
    """Return which of the 7 targets of each row a pair may use at cutoff `cu` (microseconds)."""
    m = np.isfinite(t)
    base = (origin_us <= cu) & (valid_us > cu - c.WINDOW_DAYS * DAY)
    age = np.full(NW, 5 * DAY, dtype=np.int64)
    age[SW] = 1 * DAY
    m &= base[:, None] & (valid_us[:, None] <= cu - age[None, :])
    return m


class Form1:
    """Per variable and lead band: x' = a x + b, by weighted least squares."""

    def __init__(self, *, x, t, m, lead, w):
        band = np.clip((lead / BIN_H).astype(int), 0, BINS - 1)
        self.a = np.ones((NW, BINS))
        self.b = np.zeros((NW, BINS))
        for v in range(NW):
            ok = m[:, v]
            ww = w[ok].astype(np.float64)
            bi = band[ok]
            xs, ts = x[ok, v].astype(np.float64), t[ok, v].astype(np.float64)
            sw = np.bincount(bi, ww, BINS)
            mx = np.bincount(bi, ww * xs, BINS) / np.maximum(sw, 1e-9)
            mt = np.bincount(bi, ww * ts, BINS) / np.maximum(sw, 1e-9)
            cov = np.bincount(bi, ww * (xs - mx[bi]) * (ts - mt[bi]), BINS) / np.maximum(sw, 1e-9)
            var = np.bincount(bi, ww * (xs - mx[bi]) ** 2, BINS) / np.maximum(sw, 1e-9)
            good = (sw > 100) & (var > 1e-6)
            self.a[v, good] = cov[good] / var[good]
            self.b[v, good] = mt[good] - self.a[v, good] * mx[good]
        self.band = None

    def __call__(self, x, side):
        band = np.clip((side[:, 0] * 240.0 / BIN_H).astype(int), 0, BINS - 1)
        out = x.copy()
        for v in range(NW):
            out[:, v] = self.a[v, band] * x[:, v] + self.b[v, band]
        return out.astype(np.float32)


class Form2(torch.nn.Module):
    """2 x 32 multilayer perceptron on the 7 variables, lead / 240, sin hour, cos hour; residual on the 7 variables."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(NW + 3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, NW),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, side):
        return x + self.net(torch.cat([x, side], dim=1))

    def apply(self, x, side):
        out = []
        with torch.no_grad():
            for a in range(0, len(x), 262_144):
                out.append(
                    self(
                        torch.from_numpy(x[a : a + 262_144]),
                        torch.from_numpy(side[a : a + 262_144]),
                    ).numpy()
                )
        return np.concatenate(out).astype(np.float32)

    __call__ = lambda self, x, side: (
        self.apply(x, side)
        if isinstance(x, np.ndarray)
        else torch.nn.Module.__call__(self, x, side)
    )


def side_of(aux):
    return np.ascontiguousarray(aux[:, [0, 3, 4]])


def train_form2(
    *, x, side, t, m, w, origin_day, seed, epochs=20, patience=4, lr=3e-3, device="cpu", init=None
):
    """Train form 2 with weighted masked mean squared error; every 5th origin day is the early-stopping set."""
    torch.manual_seed(seed)
    model = Form2() if init is None else init
    model = model.to(device)
    val = origin_day % 5 == 4
    tt = {
        k: torch.from_numpy(v).to(device)
        for k, v in {
            "x": x,
            "s": side,
            "t": np.nan_to_num(t),
            "m": m.astype(np.float32),
            "w": w,
        }.items()
    }
    tr, vi = np.nonzero(~val)[0], np.nonzero(val)[0]
    vi = np.random.default_rng(0).permutation(vi)[:200_000]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c.WEIGHT_DECAY)
    rng = np.random.default_rng(seed)

    def loss(b):
        e = (model(tt["x"][b], tt["s"][b]) - tt["t"][b]) ** 2 * tt["m"][b]
        return (e.sum(1) * tt["w"][b]).sum() / (tt["m"][b].sum(1) * tt["w"][b]).sum().clamp_min(
            1e-9
        )

    best, state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(tr)[: int(c.ROW_FRACTION * len(tr))]
        for a in range(0, len(order), 8192):
            b = torch.from_numpy(order[a : a + 8192]).to(device)
            v = loss(b)
            opt.zero_grad()
            v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss(torch.from_numpy(vi).to(device)))
        if v < best - 1e-6:
            best, stale = v, 0
            state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model.cpu()


def fit_stage0(*, form, x, side, t, m, lead, w, origin_day, seed):
    if form == 1:
        return Form1(x=x, t=t, m=m, lead=lead, w=w)
    return train_form2(x=x, side=side, t=t, m=m, w=w, origin_day=origin_day, seed=seed)


def apply_stage0(model, x, side):
    return model(x, side)


def mse_report(model, x, side, t, m):
    """Mean squared error to the ERA5 and CAMS target on the pair rows, before and after (diagnostic)."""
    after = model(x, side)
    r = {}
    for v, name in enumerate(WEATHER):
        ok = m[:, v]
        r[name] = (
            float(np.mean((x[ok, v] - t[ok, v]) ** 2)),
            float(np.mean((after[ok, v] - t[ok, v]) ** 2)),
        )
    return r


def xfit_outputs(*, xk, x7, valid_us, primary, mean_pos, models, edges):
    """Cross-fitted stage-1 outputs of the (corrected) ENS rows: the block-left-out network for each row."""
    idx = (valid_us - c.prep422.GRID_START_US) // c.prep422.STEP_US
    block = np.clip(np.searchsorted(edges, idx, side="right") - 1, 0, s1.BLOCKS)
    block = np.where(idx >= edges[-1], s1.BLOCKS, block)
    out = np.zeros((len(xk), 3 * c.COMPONENTS), dtype=np.float32)
    for bl in range(s1.BLOCKS + 1):
        sel = np.nonzero(block == bl)[0]
        for a in range(0, len(sel), 1_000_000):
            sl = sel[a : a + 1_000_000]
            xb = xk[sl].copy()
            xb[:, mean_pos] = x7[sl]
            out[sl] = s2.raw_outputs(
                model=models[bl], x=np.ascontiguousarray(xb), primary=primary[sl]
            )
    return out


def run(*, months, seeds, arms, stage2_form):
    torch.set_num_threads(4)
    clean = c.clean_stack()
    report = []
    for month in months:
        todo = [
            (a, s)
            for a in arms
            for s in seeds
            if not (SEED_PREDS / a / f"{month}_s{s}.parquet").exists()
        ]
        if not todo:
            continue
        t0 = time.monotonic()
        mr = c.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        columns = list(mr.base.columns)
        n_primaries = len(mr.base.target_mean)
        window_start = c.us(t=c.month_start(month=month)) - c.WINDOW_DAYS * DAY
        rows = np.nonzero(mr.valid_us > window_start)[0]
        d0 = dict(np.load(s2.CACHE / f"{month}_s0.npz"))
        assert np.array_equal(d0["valid_train"], mr.valid_us[rows]) and np.array_equal(
            d0["valid_test"], test.valid
        )
        _, keep = s2.load_stage1(path=s1.STAGE1 / f"{month}_s0.pt", n_primaries=n_primaries)
        mean_cols = [columns.index(f"mean_{w}") for w in WEATHER]
        mean_pos = [keep.index(i) for i in mean_cols]
        xk_tr = np.ascontiguousarray(mr.x[rows][:, keep])
        xk_te = np.ascontiguousarray(test.x[:, keep])
        x7_tr, x7_te = xk_tr[:, mean_pos].copy(), xk_te[:, mean_pos].copy()
        t_tr, _ = dense_targets(
            valid_us=mr.valid_us[rows],
            primary=mr.primary[rows],
            columns=columns,
            standardiser=mr.base.standardiser,
        )
        aux_tr = s2.aux_inputs(
            x=mr.x[rows], columns=columns, lead=mr.lead[rows], primary=mr.primary[rows]
        )
        aux_te = s2.aux_inputs(x=test.x, columns=columns, lead=test.lead, primary=test.primary)
        side_tr, side_te = side_of(aux_tr), side_of(aux_te)
        valid_tr, origin_tr, prim_tr, lead_tr = (
            mr.valid_us[rows],
            mr.origin_us[rows],
            mr.primary[rows],
            mr.lead[rows],
        )
        origin_day = ((origin_tr - c.us(t=c.prep.FIRST_RUN)) // DAY).astype(np.int64)
        z_tr = d0["z_train"]
        # sanity: the ENS mean is correlated with the ERA5/CAMS target
        ok = np.isfinite(t_tr[:, 0])
        print(
            f"{month}: rows {len(rows):,}; corr(ENS temp, target) {np.corrcoef(x7_tr[ok, 0][:200000], t_tr[ok, 0][:200000])[0, 1]:.3f}; setup {time.monotonic() - t0:.0f} s",
            flush=True,
        )
        del mr
        for seed in seeds:
            todo_s = [a for a in arms if not (SEED_PREDS / a / f"{month}_s{seed}.parquet").exists()]
            if not todo_s:
                continue
            full, _ = s2.load_stage1(
                path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries
            )
            models = []
            meta = s1.json.loads((s1.CROSSFIT / f"{month}_s{seed}_b0.json").read_text())
            edges = np.array(meta["edges"])
            if "S0-S2" in todo_s:
                models = [
                    s2.load_stage1(
                        path=s1.CROSSFIT / f"{month}_s{seed}_b{b}.pt", n_primaries=n_primaries
                    )[0]
                    for b in range(s1.BLOCKS)
                ] + [full]
            weeks = {a: [] for a in todo_s}
            for week, cut in enumerate(c.cutoffs(month=month)):
                cu = c.us(t=cut)
                lo, hi = fit510.window_of(month=month, week=week)
                sel = (test.origin >= lo) & (test.origin < hi)
                m = pair_mask(t=t_tr, valid_us=valid_tr, origin_us=origin_tr, cu=cu)
                age = (cu - valid_tr) / DAY
                w = (0.5 ** (age / s2.HALF_LIFE)).astype(np.float32)
                forms = {"S0-A-F1": 1, "S0-A-F2": 2, "S0-S2": stage2_form}
                for arm in todo_s:
                    t1 = time.monotonic()
                    form = forms[arm]
                    ok = m.any(1)
                    st0 = fit_stage0(
                        form=form,
                        x=x7_tr[ok],
                        side=side_tr[ok],
                        t=t_tr[ok],
                        m=m[ok],
                        lead=lead_tr[ok],
                        w=w[ok],
                        origin_day=origin_day[ok],
                        seed=seed,
                    )
                    rep = mse_report(
                        st0,
                        x7_tr[ok][:300000],
                        side_tr[ok][:300000],
                        t_tr[ok][:300000],
                        m[ok][:300000],
                    )
                    report.append(
                        (
                            month,
                            seed,
                            week,
                            arm,
                            *[f"{k}:{a:.4f}->{b:.4f}" for k, (a, b) in rep.items()],
                        )
                    )
                    idx = np.nonzero(sel)[0]
                    x7c = st0(x7_te[idx], side_te[idx])
                    xb = xk_te[idx].copy()
                    xb[:, mean_pos] = x7c
                    out = np.zeros((len(test.primary), 3 * c.COMPONENTS), dtype=np.float32)
                    out[idx] = s2.raw_outputs(
                        model=full, x=np.ascontiguousarray(xb), primary=test.primary[idx]
                    )
                    if arm == "S0-S2":
                        keep_rows = (
                            (valid_tr <= cu - c.CUTOFF_AGE_US)
                            & (origin_tr <= cu)
                            & (valid_tr > cu - c.WINDOW_DAYS * DAY)
                        )
                        ri = np.nonzero(keep_rows)[0]
                        x7_all = st0(x7_tr[ri], side_tr[ri])
                        out_tr = xfit_outputs(
                            xk=xk_tr[ri],
                            x7=x7_all,
                            valid_us=valid_tr[ri],
                            primary=prim_tr[ri],
                            mean_pos=mean_pos,
                            models=models,
                            edges=edges,
                        )
                        stage2 = s2.train_stage2(
                            form="B",
                            loss="nll",
                            out=out_tr,
                            aux=aux_tr[ri],
                            z=z_tr[ri],
                            weight=w[ri],
                            origin_day=origin_day[ri],
                            seed=seed,
                        )
                        with torch.no_grad():
                            out[idx] = stage2(
                                torch.from_numpy(out[idx]), torch.from_numpy(aux_te[idx])
                            ).numpy()
                    d = {
                        "primary_test": test.primary,
                        "origin_test": test.origin,
                        "valid_test": test.valid,
                        "target_mean": d0["target_mean"],
                        "target_std": d0["target_std"],
                    }
                    weeks[arm].append(s2.quantile_frame(out=out, d=d, select=sel))
                    print(
                        f"  {month} s{seed} w{week + 1} {arm}: {time.monotonic() - t1:.0f} s",
                        flush=True,
                    )
            for arm, frames in weeks.items():
                s2.W = rs.WORK  # write under addon_rest/work
                target = SEED_PREDS / arm / f"{month}_s{seed}.parquet"
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                pl.concat(frames).write_parquet(tmp)
                tmp.rename(target)
    rs.OUT.mkdir(parents=True, exist_ok=True)
    with open(rs.OUT / "stage0_pair_mse.csv", "a") as f:
        f.writelines(";".join(map(str, r)) + "\n" for r in report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", nargs="+", default=list(c.MONTHS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--arms", nargs="+", default=["S0-A-F1", "S0-A-F2"])
    p.add_argument("--stage2-form", type=int, default=2)
    a = p.parse_args()
    run(months=a.months, seeds=a.seeds, arms=a.arms, stage2_form=a.stage2_form)


if __name__ == "__main__":
    main()

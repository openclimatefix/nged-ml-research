# ruff: noqa: ANN001, D103, E501, PLR0912, PLR0915, PTH123
"""Add-on to experiment 524 (interim, unaudited, not for merge): S2-M, member-wise stage 1.

Each of the 51 ensemble members goes through the frozen stage-1 network (no refit). Arms:
`S2-M-0`: exact quantiles of the equal-weight 153-component mixture, no stage 2. `S2-M-0c`: the
same mixture collapsed to 3 components (component k of every member moment-matched into one
component, weights averaged), no stage 2. `S2-M-B`: the collapsed mixture through the form-B
(nll) calibrator of S2-F-B-nll, refitted at each weekly cutoff exactly as `e00524_stage2.py fit`
does (S2-J's per-week weights were never stored).

Commands: `members`, `build`, `rows`, `baseline`, `stats`.
"""

import argparse
import time

import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep397  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as fit510  # ty: ignore[unresolved-import]
import e00510_score as s510  # ty: ignore[unresolved-import]
import e00524_stage1 as s1
import e00524_stage2 as s2
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

OUT = s1.WORK_DIR.parent / "addon_s2m"
WORK = OUT / "work"
SEED_PREDS = WORK / "seed_preds"
K = c.COMPONENTS
M = 51
WEATHER = prep397.WEATHER


def member_weather(*, d: dict) -> np.ndarray:
    """Return raw member weather (rows, 51, 7) for the cache's test rows, in the cache's row order."""
    table = c.prep.primaries().select("primary", "time_series_id")
    n = len(d["primary_test"])
    keys = (
        pl.DataFrame(
            {
                "i": np.arange(n),
                "primary": d["primary_test"],
                "power_fcst_init_time": d["origin_test"],
                "valid_time": d["valid_test"],
            }
        )
        .with_columns(
            pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        )
        .join(table, on="primary")
    )
    out = np.full((n, M, len(WEATHER)), np.nan, dtype=np.float32)
    origins = keys["power_fcst_init_time"].unique().sort().to_list()
    for origin in origins:
        run = origin - c.common.PUBLICATION_DELAY
        frame = pl.read_parquet(
            c.common.demand_frame_path(run=run),
            columns=[
                "time_series_id",
                "power_fcst_init_time",
                "valid_time",
                "ensemble_member",
                *WEATHER,
            ],
        )
        part = keys.filter(pl.col("power_fcst_init_time") == origin)
        j = part.join(
            frame, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="inner"
        ).sort("i", "ensemble_member")
        counts = j.group_by("i").len()
        assert (counts["len"] == M).all(), f"{origin}: a row has fewer than {M} members"
        assert j["i"].n_unique() == part.height, f"{origin}: test rows without ensemble rows"
        out[j["i"].unique(maintain_order=True).to_numpy()] = (
            j.select(WEATHER).to_numpy().astype(np.float32).reshape(-1, M, len(WEATHER))
        )
    assert np.isfinite(out).all()
    return out


def member_inputs(
    *, x: np.ndarray, columns: list[str], raw: np.ndarray, standardiser
) -> np.ndarray:
    """Return stage-1 inputs per member: `x` with the seven mean-weather columns replaced."""
    pos = [columns.index(f"mean_{w}") for w in WEATHER]
    xm = np.repeat(x[:, None, :], M, axis=1)
    v = raw.astype(np.float64)
    pi = WEATHER.index("precipitation")
    v[..., pi] = np.log1p(np.clip(v[..., pi], 0, None))
    for j, p in enumerate(pos):
        z = (v[..., j] - standardiser.mean[p]) / standardiser.std[p]
        xm[..., p] = np.clip(z, -prep397.INPUT_CLIP_SD, prep397.INPUT_CLIP_SD)
    return xm


def collapse(*, out: np.ndarray) -> np.ndarray:
    """Return 3-component raw outputs from member outputs (rows, 51, 9): component k of every member moment-matched."""
    t = torch.from_numpy(out).double()
    log_w, mean, std = base.mixture_parts(out=t, components=K)
    w = log_w.exp()
    wk = w.mean(1)
    mu = (w * mean).sum(1) / (w.sum(1))
    second = (w * (std**2 + mean**2)).sum(1) / w.sum(1)
    sd = torch.sqrt((second - mu**2).clamp_min(1e-8))
    logit = torch.log(wk)
    # shift the logits to the scale stage 1 emits (softmax is shift-invariant, the calibrator's input is not)
    logit = logit - logit.mean(1, keepdim=True) + t[..., :K].mean((1, 2))[:, None]
    raw = torch.cat([logit, mu, torch.log(sd)], dim=1)
    return raw.float().numpy()


def exact_quantiles(*, out: np.ndarray, d: dict, rows: np.ndarray) -> np.ndarray:
    """Return sorted un-standardised 11 quantiles of the equal-weight 153-component mixture."""
    parts = []
    for a in range(0, len(rows), 8_192):
        sl = rows[a : a + 8_192]
        t = torch.from_numpy(out[sl]).double()
        log_w, mean, std = base.mixture_parts(out=t, components=K)
        n = len(sl)
        log_w = (log_w - float(np.log(M))).reshape(n, M * K)
        q = base.mixture_quantiles(
            log_w=log_w, mean=mean.reshape(n, M * K), std=std.reshape(n, M * K)
        )
        p = d["primary_test"][sl]
        q = (
            q * torch.from_numpy(d["target_std"][p])[:, None]
            + torch.from_numpy(d["target_mean"][p])[:, None]
        )
        parts.append(torch.sort(q, dim=1).values.float().numpy())
    return np.concatenate(parts)


def frame_of(*, q: np.ndarray, d: dict, rows: np.ndarray) -> pl.DataFrame:
    """Return the forecast frame in experiment 510's layout for the given test rows."""
    table = c.prep.primaries().select("primary", "time_series_id")
    frame = (
        pl.DataFrame(
            {
                "primary": d["primary_test"][rows],
                "power_fcst_init_time": d["origin_test"][rows],
                "valid_time": d["valid_test"][rows],
            }
        )
        .with_columns(
            pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        )
        .join(table, on="primary", how="left", maintain_order="left")
        .drop("primary")
    )
    levels = pl.DataFrame(q, schema=list(c.common.LEVEL_COLUMNS), orient="row")
    return frame.select(
        pl.col("time_series_id").cast(pl.Int32), "power_fcst_init_time", "valid_time"
    ).hstack(levels)


def write(*, arm: str, month: str, seed: int, frame: pl.DataFrame) -> None:
    target = SEED_PREDS / arm / f"{month}_s{seed}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(target)


def members(*, months: list[str], seeds: list[int]) -> None:
    torch.set_num_threads(2)
    clean = c.clean_stack()
    timing = []
    for month in months:
        if all(
            (SEED_PREDS / a / f"{month}_s{s}.parquet").exists()
            for a in ("S2-M-0", "S2-M-0c", "S2-M-B")
            for s in seeds
        ):
            continue
        refit = c.month_start(month=month)
        rows = data.build_fold(refit=refit, clean=clean)
        test = data.build_test(month=month, standardiser=rows.standardiser)
        columns = list(rows.columns)
        n_primaries = len(rows.target_mean)
        raw = None
        for seed in seeds:
            d = dict(np.load(s2.CACHE / f"{month}_s{seed}.npz"))
            for k in ("origin_test", "valid_test", "primary_test"):
                key = {"origin_test": "origin", "valid_test": "valid", "primary_test": "primary"}[k]
                assert np.array_equal(d[k], getattr(test, key)), (
                    f"{month}: cache row order differs ({k})"
                )
            if raw is None:
                raw = member_weather(d=d)
            model, keep = s2.load_stage1(
                path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries
            )
            n = len(test.x)
            out = np.zeros((n, M, 3 * K), dtype=np.float32)
            started = time.monotonic()
            for a in range(0, n, 4_096):
                sl = slice(a, a + 4_096)
                xm = member_inputs(
                    x=test.x[sl], columns=columns, raw=raw[sl], standardiser=rows.standardiser
                )
                xm = np.ascontiguousarray(xm[:, :, keep]).reshape(-1, len(keep))
                pm = np.repeat(test.primary[sl], M)
                out[sl] = s2.raw_outputs(model=model, x=xm, primary=pm).reshape(-1, M, 3 * K)
                if a == 0 and seed == seeds[0]:
                    # check: the mean-weather input reproduces the cached mean-input outputs
                    xc = np.ascontiguousarray(test.x[sl][:, keep])
                    chk = s2.raw_outputs(model=model, x=xc, primary=test.primary[sl])
                    assert np.abs(chk - d["out_test"][sl]).max() < 1e-4, "mean-input check failed"
            secs = time.monotonic() - started
            timing.append((month, seed, n, secs))
            print(
                f"{month} s{seed}: stage-1 over {n:,} rows x {M} members, {secs:.0f} s", flush=True
            )
            spread = float(out[..., K : 2 * K].mean(2).std(1).mean())
            print(f"  member spread of mixture mean (standardised units) {spread:.3f}", flush=True)
            col = collapse(out=out.reshape(n, M, 3 * K))
            cuts = c.cutoffs(month=month)
            wanted = [
                a
                for a in ("S2-M-0", "S2-M-0c", "S2-M-B")
                if not (SEED_PREDS / a / f"{month}_s{seed}.parquet").exists()
            ]
            if not wanted:
                continue
            weeks = {a: [] for a in wanted}
            for week, cut in enumerate(cuts):
                lo, hi = fit510.window_of(month=month, week=week)
                sel = (d["origin_test"] >= lo) & (d["origin_test"] < hi)
                idx = np.nonzero(sel)[0]
                if "S2-M-0" in weeks:
                    weeks["S2-M-0"].append(
                        frame_of(q=exact_quantiles(out=out, d=d, rows=idx), d=d, rows=idx)
                    )
                if "S2-M-0c" in weeks:
                    weeks["S2-M-0c"].append(s2.quantile_frame(out=col, d=d, select=sel))
                if "S2-M-B" not in weeks:
                    continue
                cu = c.us(t=cut)
                tr = (
                    (d["valid_train"] <= cu - c.CUTOFF_AGE_US)
                    & (d["origin_train"] <= cu)
                    & (d["valid_train"] > cu - c.WINDOW_DAYS * c.DAY_US)
                )
                age = (cu - d["valid_train"][tr]) / c.DAY_US
                weight = (0.5 ** (age / s2.HALF_LIFE)).astype(np.float32)
                origin_day = (
                    (d["origin_train"][tr] - c.us(t=c.prep.FIRST_RUN)) // c.DAY_US
                ).astype(np.int64)
                t0 = time.monotonic()
                stage2 = s2.train_stage2(
                    form="B",
                    loss="nll",
                    out=d["out_train"][tr],
                    aux=d["aux_train"][tr],
                    z=d["z_train"][tr],
                    weight=weight,
                    origin_day=origin_day,
                    seed=seed,
                )
                with torch.no_grad():
                    cal = stage2(
                        torch.from_numpy(col[sel]), torch.from_numpy(d["aux_test"][sel])
                    ).numpy()
                full = np.zeros_like(col)
                full[sel] = cal
                weeks["S2-M-B"].append(s2.quantile_frame(out=full, d=d, select=sel))
                print(
                    f"  {month} s{seed} w{week + 1}: stage 2 {time.monotonic() - t0:.0f} s",
                    flush=True,
                )
            for arm, frames in weeks.items():
                write(arm=arm, month=month, seed=seed, frame=pl.concat(frames))
            del out, col
        del raw, rows, test
    (OUT).mkdir(parents=True, exist_ok=True)
    with open(OUT / "timing.csv", "a") as f:
        f.writelines(",".join(map(str, r)) + "\n" for r in timing)


# ---- scoring -----------------------------------------------------------------------------

NEW = ("S2-M-0", "S2-M-0c", "S2-M-B")
OLD = ("S2-0F", "S2-F-B-nll", "S2-J-0.4x", "S2-J-0.1x", "L-Ww")
SOURCE = s1.WORK_DIR


def configure_scoring() -> None:
    """Point experiment 510's scorer at the add-on folder."""
    s510.WORK_DIR = WORK
    s510.SEED_PREDS = SEED_PREDS
    s510.ARMS = dict.fromkeys(NEW, (0, 1))
    s510.CONTRASTS = (
        ("S2-M-0", "L-Ww"),
        ("S2-M-B", "L-Ww"),
        ("S2-M-0c", "L-Ww"),
        ("S2-M-0", "S2-J-0.1x"),
        ("S2-M-B", "S2-J-0.1x"),
        ("S2-M-0", "S2-J-0.4x"),
        ("S2-M-B", "S2-J-0.4x"),
        ("S2-M-B", "S2-M-0"),
        ("S2-M-B", "S2-F-B-nll"),
        ("S2-M-0", "S2-0F"),
        ("S2-M-0c", "S2-M-0"),
    )
    s510.TESTS = 4 * len(s510.CONTRASTS)
    s510.configure()
    s510.link_stored()
    # existing arms: link pooled predictions and the two-seed singles from experiment 524's folder
    pred = WORK / "predictions"
    for arm in (*OLD, "L-W", "L-W-s0", "L-W-s1", "L-Ww-s0", "L-Ww-s1", "K2-S1"):
        link = pred / arm
        src = SOURCE / "predictions" / arm
        if not link.exists() and src.exists():
            link.symlink_to(src, target_is_directory=True)
    sp = s510.score470
    sp.CANDIDATES = (*NEW,)
    sp.CONTROLS = (*OLD, "K2-S1")
    sp.MAIN_ARMS = (*NEW, *OLD, "K2-S1")
    sp.SINGLES = s510.SINGLES
    sp.CONTRASTS = s510.CONTRASTS
    sp.TESTS = s510.TESTS


def build_scores() -> None:
    configure_scoring()
    s510.build()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("members", "build", "rows", "baseline", "stats"))
    p.add_argument("--months", nargs="+", default=list(c.MONTHS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    a = p.parse_args()
    if a.command == "members":
        members(months=a.months, seeds=a.seeds)
        return
    configure_scoring()
    {
        "build": s510.build,
        "rows": s510.score470.rows_command,
        "baseline": s510.baseline,
        "stats": s510.score470.stats_command,
    }[a.command]()


if __name__ == "__main__":
    main()

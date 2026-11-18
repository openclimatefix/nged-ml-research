# ruff: noqa: E501, PLR0915
"""Reference implementation of K1, table T, and the Candidate's inference, checked against a bundle's fixtures.

This file is copied into every bundle as `reference_check.py`. It imports nothing from the research
repository, reads only files inside the bundle directory, and needs only `torch`, `numpy`, and
`polars` (the Parquet reader). Production can copy the functions below.

    python reference_check.py [BUNDLE_DIR] [--tolerance 1e-5] [--quantile-tolerance 2e-5]

For every fixture origin and every seed the script (1) runs K1's forward pass with both lags masked
on the stored K1 input rows of the three origins that fill the newest table T entries, and compares
the medians with the stored ones; (2) rebuilds those table T entries by the freshest-origin rule and
compares them with the stored entries; (3) rebuilds the Candidate's 60 base columns from the raw
columns, the lag values, and the expected-weather rows, and its signal columns from the power
history and table T; (4) runs the Candidate's forward pass and the quantile read-out on the stored
matrix and on the rebuilt matrix; (5) compares the quantiles, the seed mean, and the 12:00 contract
megawatts with the stored ones. Exit status 1 if any error exceeds a tolerance. The stored values
come from a GPU pass; the CPU pass differs by float32 rounding.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
import torch.nn.functional as F  # noqa: N812  # ty: ignore[unresolved-import]

LEVEL_NAMES = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
K1_QUANTILE_COLUMNS = (1, 5, 9)
"""Positions of q05, q50, and q95 among the 11 levels: what K1's lags-masked read-out keeps."""
BASES = ("gap", "ew1", "ew3", "ew7")
LAGS = ("day", "week")
SCALE = 10.0
CLIP = 10.0


def read_json(path: Path) -> dict:
    """Return the JSON file's content."""
    return json.loads(path.read_text())


def load_bundle(root: Path) -> dict:
    """Load every non-fixture file of the bundle into one dict."""
    manifest = read_json(root / "manifest.json")
    assert manifest["bundle_format_version"] == "1", manifest["bundle_format_version"]
    k1, cand = root / "components" / "k1", root / "components" / "candidate"
    bundle = {
        "manifest": manifest,
        "config": read_json(root / "config.json"),
        "primaries": read_json(root / "primaries.json"),
        "columns": read_json(k1 / "columns.json"),
        "standardisers": read_json(k1 / "standardisers.json"),
        "expected_rows": pl.read_parquet(k1 / "expected_weather_rows.parquet"),
        "signal_columns": read_json(cand / "signal_columns.json"),
        "neighbours": read_json(cand / "neighbours.json"),
    }
    for name, folder in (("k1", k1), ("candidate", cand)):
        bundle[f"weights_{name}"] = {
            seed: torch.load(
                folder / "weights" / f"seed{seed}.pt", map_location="cpu", weights_only=True
            )
            for seed in manifest["components"][name]["seeds"]
        }
    return bundle


# --- base input columns -------------------------------------------------------------------------


def standardise_x37(*, raw: np.ndarray, standardisers: dict, clip_sd: float) -> np.ndarray:
    """Return the first 37 columns: (raw - mean) / std in float32, clipped, NaN replaced by 0."""
    mean = np.asarray(standardisers["x37"]["mean"], dtype=np.float32)
    std = np.asarray(standardisers["x37"]["std"], dtype=np.float32)
    z = np.clip((raw - mean) / std, -clip_sd, clip_sd)
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


def lag_columns(
    *, power: np.ndarray, lag_index: np.ndarray, target_index: np.ndarray, which: int, bundle: dict
) -> np.ndarray:
    """Return one lag's value, missing flag, and age code (rows, 3), float32."""
    config = bundle["config"]["components"]["candidate"]["lags"]
    stats = bundle["standardisers"]["lag"]["day" if which == 0 else "week"]
    missing = ~np.isfinite(power) | (lag_index < 0)
    mean = np.float64(stats["mean"])
    std = max(np.float64(stats["std"]), np.float64(config["std_floor"]))
    z = np.clip((power.astype(np.float64) - mean) / std, -config["clip_sd"], config["clip_sd"])
    age = (target_index - lag_index) * 0.5 / config["age_scale_hours"]
    out = np.zeros((len(power), 3), dtype=np.float32)
    out[:, 0] = np.where(missing, 0.0, z)
    out[:, 1] = missing
    out[:, 2] = np.where(missing, config["missing_age_code"][("day", "week")[which]], age)
    return out


def knot_position(*, valid_us: np.ndarray, config: dict) -> np.ndarray:
    """Return the offset-knot position: months since the first knot month plus the fraction of the month."""
    t = valid_us.astype("datetime64[us]")
    start = t.astype("datetime64[M]")
    lo, hi = (
        start.astype("datetime64[us]"),
        (start + np.timedelta64(1, "M")).astype("datetime64[us]"),
    )
    frac = (t - lo) / (hi - lo)
    first = np.datetime64(config["offset_knots"]["first_knot_month"], "M")
    return ((start - first).astype(np.float64) + frac).astype(np.float32)


def build_base_matrix(*, rows: pl.DataFrame, lag_power: dict, bundle: dict) -> np.ndarray:
    """Return the (rows, 60) standardised base matrix from the fixture's raw columns.

    `rows` holds the `input_matrix.parquet` rows of one origin; `lag_power` maps
    (primary_index, grid_index) to the cleaned power fraction (NaN when missing).
    """
    names = [c["name"] for c in bundle["columns"]]
    config = bundle["config"]["components"]["candidate"]
    standardisers = bundle["standardisers"]
    x = np.zeros((rows.height, config["n_base_columns"]), dtype=np.float32)
    raw37 = rows.select([f"raw__{name}" for name in names[:37]]).to_numpy().astype(np.float32)
    x[:, :37] = standardise_x37(
        raw=raw37, standardisers=standardisers, clip_sd=config["input_clip_sd"]
    )
    primary = rows["primary_index"].to_numpy()
    target = rows["target_grid_index"].to_numpy()
    for which, name in enumerate(LAGS):
        idx = rows[f"raw__{name}_lag_grid_index"].to_numpy().astype(np.int64)
        power = np.array(
            [
                lag_power.get((int(p), int(i)), np.nan) if i >= 0 else np.nan
                for p, i in zip(primary, idx, strict=True)
            ],
            dtype=np.float32,
        )
        start, stop = config["column_slices"][("day_lag", "week_lag")[which]]
        x[:, start:stop] = lag_columns(
            power=power, lag_index=idx, target_index=target, which=which, bundle=bundle
        )
        x[:, config["lag_grid_index_columns"][which]] = idx
    x[:, config["knot_position_column"]] = knot_position(
        valid_us=rows["valid_time"].dt.epoch("us").to_numpy(), config=config
    )
    b1 = rows.select([f"raw__{name}" for name in names[46:]]).to_numpy().astype(np.float32)
    mean = np.asarray(standardisers["b1"]["mean"], dtype=np.float32)
    std = np.asarray(standardisers["b1"]["std"], dtype=np.float32)
    z = (b1 - mean) / std
    x[:, 46:] = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
    return x


def expected_table(*, bundle: dict) -> dict[tuple[int, int], torch.Tensor]:
    """Return the standardised expected-weather rows by (primary, grid index), float16 as in the research network."""
    config = bundle["config"]["components"]["candidate"]
    positions = config["expected_weather"]["x37_positions"]
    mean = np.asarray(bundle["standardisers"]["x37"]["mean"], dtype=np.float32)[positions]
    std = np.asarray(bundle["standardisers"]["x37"]["std"], dtype=np.float32)[positions]
    names = [bundle["columns"][i]["name"] for i in positions]
    frame = bundle["expected_rows"]
    raw = frame.select([f"raw__{n}" for n in names]).to_numpy().astype(np.float32)
    z = np.clip((raw - mean) / std, -config["input_clip_sd"], config["input_clip_sd"])
    z16 = np.nan_to_num(z, nan=0.0).astype(np.float16)
    keys = zip(frame["primary_index"].to_list(), frame["grid_index"].to_list(), strict=True)
    return {key: torch.from_numpy(z16[i]) for i, key in enumerate(keys)}


def weather_present(*, bundle: dict) -> dict[tuple[int, int], bool]:
    """Return, by (primary, grid index), whether the expected-weather table holds a run for that half-hour."""
    frame = bundle["expected_rows"]
    origin = frame["run_origin_time"].dt.epoch("us").to_numpy()
    keys = zip(frame["primary_index"].to_list(), frame["grid_index"].to_list(), strict=True)
    return {key: bool(origin[i] > 0) for i, key in enumerate(keys)}


# --- network ------------------------------------------------------------------------------------


def mlp(*, w: dict, prefix: str, x: torch.Tensor, layers: tuple[int, ...]) -> torch.Tensor:
    """Apply Linear layers named `prefix.<i>` with an exact GELU between them (dropout is off at inference)."""
    for k, i in enumerate(layers):
        x = F.linear(x, w[f"{prefix}.{i}.weight"], w[f"{prefix}.{i}.bias"])
        if k < len(layers) - 1:
            x = F.gelu(x)
    return x


def forward(
    *, w: dict, x: torch.Tensor, primary: torch.Tensor, table: dict, config: dict, n_extra: int
) -> torch.Tensor:
    """Return the nine raw outputs (three logits, three means, three log standard deviations), float32.

    `x` has the 60 base columns, then `n_extra` signal columns (none for K1). The signal columns
    are appended to the input of each lag encoder.
    """
    cols = config["column_slices"]
    n = len(x)
    emb = w["embedding.weight"][primary]
    weather_in = torch.cat(
        [x[:, cols["weather"][0] : cols["weather"][1]], emb, x[:, cols["b1"][0] : cols["b1"][1]]],
        dim=-1,
    )
    hw = F.gelu(mlp(w=w, prefix="weather", x=weather_in, layers=(0, 3)))
    hd = F.gelu(
        mlp(
            w=w,
            prefix="datetime",
            x=torch.cat([x[:, cols["datetime"][0] : cols["datetime"][1]], emb], dim=-1),
            layers=(0, 3),
        )
    )
    n_weather = cols["weather"][1] - cols["weather"][0]
    width = len(config["expected_weather"]["x37_positions"])
    extra = x[:, config["n_base_columns"] : config["n_base_columns"] + n_extra]
    outs = []
    for j, name in enumerate(LAGS):
        idx = x[:, config["lag_grid_index_columns"][j]].long()
        # A missing lag (grid index < 0) reads a zero row; any other absent key is a bundle error.
        rows = torch.stack(
            [
                table[(int(p), int(i))] if i >= 0 else torch.zeros(width, dtype=torch.float16)
                for p, i in zip(primary.tolist(), idx.tolist(), strict=True)
            ]
        ).float()
        pad = torch.zeros(n, config["n_b1"])
        ehw = F.gelu(
            mlp(
                w=w,
                prefix="weather",
                x=torch.cat([rows[:, :n_weather], emb, pad], dim=-1),
                layers=(0, 3),
            )
        )
        ehd = F.gelu(
            mlp(
                w=w,
                prefix="datetime",
                x=torch.cat([rows[:, n_weather:], emb], dim=-1),
                layers=(0, 3),
            )
        )
        lag_id = w["lag_id.weight"][torch.full((n,), j, dtype=torch.long)]
        lag_cols = cols[f"{name}_lag"]
        parts = [x[:, lag_cols[0] : lag_cols[1]], ehw, ehd, emb, lag_id]
        if n_extra:
            parts.append(extra)
        outs.append(mlp(w=w, prefix="lag.0", x=torch.cat(parts, dim=-1), layers=(0, 3)))
    hl = F.gelu(torch.cat(outs, dim=-1))
    core = torch.cat([hw, hd, hl, emb], dim=-1)
    mixer_in = torch.cat(
        [
            core,
            x[:, cols["spread"][0] : cols["spread"][1]],
            x[:, cols["lead"][0] : cols["lead"][1]],
        ],
        dim=-1,
    )
    out = mlp(w=w, prefix="mixer_ft", x=mixer_in, layers=(0, 3, 6))
    knots = torch.cat([w["knots_old"], w["knot_new"][:, None, :]], dim=1)
    n_knots = knots.shape[1]
    tau = x[:, config["knot_position_column"]].clamp(0.0, float(n_knots - 1))
    lo = tau.floor().long().clamp(max=n_knots - 2)
    frac = (tau - lo.to(tau.dtype))[:, None]
    offset = (1.0 - frac) * knots[primary, lo] + frac * knots[primary, lo + 1]
    k = config["mixture_components"]
    return torch.cat([out[:, :k], out[:, k : 2 * k] + offset, out[:, 2 * k :]], dim=-1)


def quantiles(*, raw: torch.Tensor, config: dict) -> torch.Tensor:
    """Return the 11 standardised quantiles of the mixture, by bisection on its CDF in float64."""
    k = config["mixture_components"]
    out = raw.double()
    log_w = torch.log_softmax(out[:, :k], dim=-1)
    mean = out[:, k : 2 * k]
    lo_s, hi_s = config["log_std_clamp"]
    std = torch.exp(out[:, 2 * k :].clamp(lo_s, hi_s))
    levels = torch.tensor(config["quantile_levels"], dtype=torch.float64)
    w = log_w.exp()
    limit = config["bisection_range"]
    lo = torch.full((len(w), len(levels)), -float(limit), dtype=torch.float64)
    hi = torch.full_like(lo, float(limit))
    for _ in range(config["bisection_steps"]):
        mid = 0.5 * (lo + hi)
        z = (mid[..., None] - mean[:, None, :]) / (std[:, None, :] * float(np.sqrt(2.0)))
        cdf = (w[:, None, :] * 0.5 * (1.0 + torch.erf(z))).sum(-1)
        below = cdf < levels
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def native_fractions(*, raw: torch.Tensor, primary: np.ndarray, bundle: dict) -> np.ndarray:
    """Return the sorted quantiles as a fraction of capacity, float32, at the native 09:00 origin."""
    config = bundle["config"]["components"]["candidate"]
    target = bundle["standardisers"]["target"]
    mean = torch.tensor(target["mean"], dtype=torch.float64)[torch.from_numpy(primary)][:, None]
    std = torch.tensor(target["std"], dtype=torch.float64)[torch.from_numpy(primary)][:, None]
    q = torch.sort(quantiles(raw=raw, config=config) * std + mean, dim=1).values
    return q.float().numpy()


def mask_lags(*, x: np.ndarray, config: dict) -> np.ndarray:
    """Return a copy of `x` with both lags set to the missing encoding (value 0, flag 1, largest age code)."""
    out = x.copy()
    for start, age in (
        (config["column_slices"]["day_lag"][0], config["lags"]["missing_age_code"]["day"]),
        (config["column_slices"]["week_lag"][0], config["lags"]["missing_age_code"]["week"]),
    ):
        out[:, start], out[:, start + 1], out[:, start + 2] = 0.0, 1.0, age
    return out


def k1_lags_masked(
    *, x: np.ndarray, primary: np.ndarray, weights: dict, table: dict, bundle: dict
) -> np.ndarray:
    """Return K1's lags-masked q05, q50, and q95 (fraction of capacity), (rows, 3), float32."""
    config = bundle["config"]["components"]["k1"]
    xm = torch.from_numpy(mask_lags(x=x, config=config))
    pt = torch.from_numpy(primary)
    with torch.no_grad():
        raw = forward(w=weights, x=xm, primary=pt, table=table, config=config, n_extra=0)
    return native_fractions(raw=raw, primary=primary, bundle=bundle)[:, K1_QUANTILE_COLUMNS]


# --- table T and the signal columns ---------------------------------------------------------------


def rebuild_table_t(
    *,
    primary: np.ndarray,
    origin_idx: np.ndarray,
    target_idx: np.ndarray,
    median: np.ndarray,
    config: dict,
) -> dict[tuple[int, int], tuple[float, int]]:
    """Return table T entries by (primary, grid index): (median, source origin index).

    Each entry is the lags-masked median of the freshest forecast origin that precedes the grid
    index, among forecasts whose lead is positive and at most `max_lead_half_hours`.
    """
    lead = target_idx - origin_idx
    keep = (lead > 0) & (lead <= config["table_t"]["max_lead_half_hours"]) & np.isfinite(median)
    entries: dict[tuple[int, int], tuple[float, int]] = {}
    for p, o, t, m in zip(
        primary[keep], origin_idx[keep], target_idx[keep], median[keep], strict=True
    ):
        key = (int(p), int(t))
        if key not in entries or o > entries[key][1]:
            entries[key] = (float(m), int(o))
    return entries


def ew_average(*, res: np.ndarray, half_life_days: int, min_weight: float) -> np.ndarray:
    """Return (16, grid): the exponentially weighted average of `res` up to and including each index.

    The weight of a reading `k` half-hours old is `0.5 ** (k / (48 * half_life_days))`. Missing
    readings carry no weight. NaN where the weights fall under `min_weight` of a full history.
    The recursion starts at the first column of `res`, with zero state.
    """
    alpha = 0.5 ** (1.0 / (48 * half_life_days))
    fin = np.isfinite(res)
    v = np.where(fin, res, 0.0).astype(np.float64)
    f = fin.astype(np.float64)
    num = np.empty_like(v)
    den = np.empty_like(v)
    num[:, 0], den[:, 0] = v[:, 0], f[:, 0]
    for t in range(1, v.shape[1]):
        num[:, t] = v[:, t] + alpha * num[:, t - 1]
        den[:, t] = f[:, t] + alpha * den[:, t - 1]
    full = 1.0 / (1.0 - alpha)
    ok = den >= min_weight * full
    return np.where(ok, num / np.maximum(den, 1e-12), np.nan).astype(np.float32)


def signal_block(
    *,
    primary: np.ndarray,
    day_idx: np.ndarray,
    week_idx: np.ndarray,
    day_miss: np.ndarray,
    week_miss: np.ndarray,
    power: np.ndarray,
    table_median: np.ndarray,
    first_idx: int,
    present: dict[tuple[int, int], bool],
    neighbours: dict,
    names: list[str],
    config: dict,
) -> np.ndarray:
    """Return the (rows, len(names)) float32 signal columns of the Candidate.

    `power` and `table_median` are (16, window) arrays whose column 0 is grid index `first_idx`
    (power as a fraction of effective capacity, table T's lags-masked median as a fraction of
    capacity; NaN where absent). A row reads only grid indices at or before its lag times, which are
    at least 24 h before its origin.
    """
    min_weight = config["signals"]["ew_min_weight"]
    res = (power - table_median).astype(np.float32)
    arrays = {"gap": res}
    for base in BASES[1:]:
        arrays[base] = ew_average(res=res, half_life_days=int(base[2:]), min_weight=min_weight)
    n = len(primary)
    cols: dict[str, np.ndarray] = {name: np.zeros(n, np.float32) for name in names}
    lag_info = (("day", day_idx, day_miss), ("week", week_idx, week_miss))
    own: dict[str, np.ndarray] = {}
    for lag, idx, miss in lag_info:
        absent = miss | (idx < 0)
        i = np.clip(idx, 0, None) - first_idx
        assert (i[~absent] >= 0).all(), "a lag lies before the history window"
        for base, arr in arrays.items():
            val = np.where(absent, np.nan, arr[primary, i])
            own[f"{base}_{lag}"] = val
            cols[f"{base}_{lag}"] = np.nan_to_num(val) * SCALE
            cols[f"{base}_{lag}_missing"] = (~np.isfinite(val) & ~absent).astype(np.float32)
        cols[f"exp_missing_{lag}"] = np.array(
            [
                float(int(j) >= 0 and not present[(int(p), int(j))])
                for p, j in zip(primary, idx, strict=True)
            ],
            dtype=np.float32,
        )
    qlist = [f"{b}_{lag}" for lag in LAGS for b in BASES]
    capacity = np.asarray(neighbours["capacity_mw"], dtype=np.float64)
    of = {int(p): list(qs) for p, qs in neighbours["of"].items()}
    for p in range(len(capacity)):
        rows = np.flatnonzero(primary == p)
        qs = of.get(p, [])
        if len(rows) == 0 or not qs:
            cols["nb_none"][rows] = 1.0
            continue
        vals = np.full((len(qs), len(qlist), len(rows)), np.nan, np.float32)
        for k, q in enumerate(qs):
            scale = capacity[q] / capacity[p]
            for lag, idx, miss in lag_info:
                absent = (miss | (idx < 0))[rows]
                i = np.clip(idx[rows], 0, None) - first_idx
                for base, arr in arrays.items():
                    vals[k, qlist.index(f"{base}_{lag}")] = np.where(
                        absent, np.nan, arr[q, i] * scale
                    )
        own_rows = np.stack([own[q][rows] for q in qlist])
        fin = np.isfinite(vals)
        have = np.isfinite(
            np.stack(
                [
                    np.where(
                        (day_miss | (day_idx < 0))[rows],
                        np.nan,
                        arrays["gap"][q, np.clip(day_idx[rows], 0, None) - first_idx],
                    )
                    for q in qs
                ]
            )
        )
        count = have.sum(axis=0)
        cols["nb_count"][rows] = count
        cols["nb_none"][rows] = count == 0
        z = np.where(fin, vals, 0.0)
        nfin = fin.sum(axis=0)
        summ = z.sum(axis=0)
        mean = np.where(nfin > 0, summ / np.maximum(nfin, 1), 0.0)
        vmin = np.where(nfin > 0, np.where(fin, vals, np.inf).min(axis=0), 0.0)
        vmax = np.where(nfin > 0, np.where(fin, vals, -np.inf).max(axis=0), 0.0)
        own0 = np.nan_to_num(own_rows)
        for j, q in enumerate(qlist):
            for stat, arr in zip(
                ("sum", "mean", "min", "max"), (summ, mean, vmin, vmax), strict=True
            ):
                cols[f"nb_{stat}_{q}"][rows] = arr[j] * SCALE
            cols[f"own_plus_nb_{q}"][rows] = (own0[j] + summ[j]) * SCALE
    out = np.stack([cols[name] for name in names], axis=1).astype(np.float32)
    np.clip(out, -CLIP, CLIP, out=out)
    return out


# --- contract rows ------------------------------------------------------------------------------


def contract_rows(
    *,
    per_seed: list[np.ndarray],
    primary: np.ndarray,
    valid: np.ndarray,
    origin: np.ndarray,
    bundle: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask, per-seed MW stacked, seed-mean MW) of the 12:00 contract rows.

    Each seed's fractions are multiplied by the Primary's `capacity_mw` in float64 and rounded to
    float32; the seeds are averaged in float32 and each row is sorted.
    """
    shared = bundle["config"]["shared"]
    capacity = np.array([p["capacity_mw"] for p in bundle["primaries"]["primaries"]])[primary]
    shift = np.timedelta64(
        shared["contract_origin_hour_utc"] - shared["native_origin_hour_utc"], "h"
    )
    contract_origin = origin + shift
    horizon = np.timedelta64(shared["horizon_days"], "D")
    mask = (valid >= contract_origin) & (valid < contract_origin + horizon)
    mw = [(q.astype(np.float64) * capacity[:, None]).astype(np.float32) for q in per_seed]
    mean = np.sort(np.mean(np.stack(mw), axis=0, dtype=np.float32), axis=1)
    return mask, np.stack(mw), mean


# --- check --------------------------------------------------------------------------------------


def lag_power_of(*, frame: pl.DataFrame) -> dict[tuple[int, int], float]:
    """Return the fixture's lag power by (primary, grid index)."""
    return {
        (int(p), int(i)): float(v)
        for p, i, v in zip(
            frame["primary_index"], frame["grid_index"], frame["power_fraction"], strict=True
        )
    }


def check_origin(
    *, root: Path, origin_dir: str, bundle: dict, table: dict, present: dict, worst: dict
) -> None:
    """Reproduce one fixture origin and update the largest errors in `worst`."""
    config = bundle["config"]["components"]["candidate"]
    k1_config = bundle["config"]["components"]["k1"]
    seeds = bundle["manifest"]["components"]["candidate"]["seeds"]
    names = [c["name"] for c in bundle["columns"]]
    signal_names = bundle["signal_columns"]["names"]
    cand = root / "fixtures" / "candidate" / origin_dir
    k1 = root / "fixtures" / "k1" / origin_dir
    rows = pl.read_parquet(cand / "input_matrix.parquet")
    lags = pl.read_parquet(cand / "lag_and_expected_rows.parquet")
    history = pl.read_parquet(cand / "signal_history.parquet")
    hist_rows = pl.read_parquet(k1 / "history_rows.parquet")
    entries = pl.read_parquet(k1 / "table_t_entries.parquet")
    out = pl.read_parquet(cand / "outputs_native.parquet")
    primary = rows["primary_index"].to_numpy().astype(np.int64)
    pt = torch.from_numpy(primary)

    base = build_base_matrix(rows=rows, lag_power=lag_power_of(frame=lags), bundle=bundle)
    stored_base = rows.select([f"std__{n}" for n in names]).to_numpy().astype(np.float32)
    worst["input_matrix_base"] = max(
        worst["input_matrix_base"], float(np.abs(base - stored_base).max())
    )

    # K1 on the history rows, and table T by the freshest-origin rule
    h_prim = hist_rows["primary_index"].to_numpy().astype(np.int64)
    h_x = hist_rows.select([f"std__{n}" for n in names]).to_numpy().astype(np.float32)
    g_first = int(history["grid_index"].to_numpy().min())
    g_last = int(history["grid_index"].to_numpy().max())
    width = g_last - g_first + 1
    power = np.full((16, width), np.nan, np.float32)
    power[history["primary_index"].to_numpy(), history["grid_index"].to_numpy() - g_first] = (
        history["power_fraction"].to_numpy()
    )
    e_prim = entries["primary_index"].to_numpy()
    e_idx = entries["grid_index"].to_numpy()
    per_seed_rebuilt = []
    for seed in seeds:
        q = k1_lags_masked(
            x=h_x, primary=h_prim, weights=bundle["weights_k1"][seed], table=table, bundle=bundle
        )
        want = hist_rows.select([f"k1_{lv}_s{seed}" for lv in ("q05", "q50", "q95")]).to_numpy()
        worst["k1_lags_masked_median"] = max(
            worst["k1_lags_masked_median"], float(np.abs(q[:, 1] - want[:, 1]).max())
        )
        worst["k1_lags_masked_q05_q95"] = max(
            worst["k1_lags_masked_q05_q95"], float(np.abs(q[:, [0, 2]] - want[:, [0, 2]]).max())
        )
        rebuilt = rebuild_table_t(
            primary=h_prim,
            origin_idx=hist_rows["origin_grid_index"].to_numpy(),
            target_idx=hist_rows["target_grid_index"].to_numpy(),
            median=q[:, 1],
            config=config,
        )
        stored_t = entries[f"t_median_s{seed}"].to_numpy()
        stored_o = entries[f"t_origin_grid_index_s{seed}"].to_numpy()
        got = np.array([rebuilt[(int(p), int(i))][0] for p, i in zip(e_prim, e_idx, strict=True)])
        got_o = np.array([rebuilt[(int(p), int(i))][1] for p, i in zip(e_prim, e_idx, strict=True)])
        worst["table_t_entries"] = max(
            worst["table_t_entries"], float(np.abs(got - stored_t).max())
        )
        assert np.array_equal(got_o, stored_o), "a table T entry has a different source origin"
        t_median = np.full((16, width), np.nan, np.float32)
        t_median[
            history["primary_index"].to_numpy(), history["grid_index"].to_numpy() - g_first
        ] = history[f"t_median_s{seed}"].to_numpy()
        stored_median = t_median.copy()
        t_median[e_prim, e_idx - g_first] = got.astype(
            np.float32
        )  # the rebuilt entries replace the stored ones
        per_seed_rebuilt.append((t_median, stored_median))
    day_idx = rows["raw__day_lag_grid_index"].to_numpy().astype(np.int64)
    week_idx = rows["raw__week_lag_grid_index"].to_numpy().astype(np.int64)
    day_miss, week_miss = base[:, 38] > 0.5, base[:, 41] > 0.5
    n_extra = config["n_extra"]
    start = config["n_base_columns"]
    per_seed_q = []
    for k, seed in enumerate(seeds):
        weights = bundle["weights_candidate"][seed]
        t_rebuilt, _ = per_seed_rebuilt[k]
        extras = signal_block(
            primary=primary, day_idx=day_idx, week_idx=week_idx, day_miss=day_miss, week_miss=week_miss,
            power=power, table_median=t_rebuilt, first_idx=g_first, present=present,
            neighbours=bundle["neighbours"], names=signal_names, config=config,
        )  # fmt: skip
        stored_extras = (
            rows.select([f"extra_s{seed}__{n}" for n in signal_names]).to_numpy().astype(np.float32)
        )
        worst["input_matrix_signals"] = max(
            worst["input_matrix_signals"], float(np.abs(extras - stored_extras).max())
        )
        stored_x = np.concatenate([stored_base, stored_extras], axis=1)
        rebuilt_x = np.concatenate([base, extras], axis=1)
        assert stored_x.shape[1] == start + n_extra
        with torch.no_grad():
            raw = forward(
                w=weights,
                x=torch.from_numpy(stored_x),
                primary=pt,
                table=table,
                config=config,
                n_extra=n_extra,
            )
            raw_end = forward(
                w=weights,
                x=torch.from_numpy(rebuilt_x),
                primary=pt,
                table=table,
                config=config,
                n_extra=n_extra,
            )
        want_raw = out.select([f"raw_out_{j}_s{seed}" for j in range(9)]).to_numpy()
        worst["raw_output"] = max(worst["raw_output"], float(np.abs(raw.numpy() - want_raw).max()))
        want = out.select([f"{lv}_s{seed}" for lv in LEVEL_NAMES]).to_numpy()
        q_stored = native_fractions(raw=raw, primary=primary, bundle=bundle)
        q_end = native_fractions(raw=raw_end, primary=primary, bundle=bundle)
        worst["native_seed_stored_matrix"] = max(
            worst["native_seed_stored_matrix"], float(np.abs(q_stored - want).max())
        )
        worst["native_seed"] = max(worst["native_seed"], float(np.abs(q_end - want).max()))
        per_seed_q.append(q_end)
    mean = np.sort(np.mean(np.stack(per_seed_q), axis=0, dtype=np.float32), axis=1)
    want_mean = out.select([f"{lv}_mean" for lv in LEVEL_NAMES]).to_numpy()
    worst["native_seed_mean"] = max(
        worst["native_seed_mean"], float(np.abs(mean - want_mean).max())
    )
    valid = rows["valid_time"].dt.epoch("us").to_numpy().astype("datetime64[us]")
    origin = rows["origin_time"].dt.epoch("us").to_numpy().astype("datetime64[us]")
    mask, _, mw_mean = contract_rows(
        per_seed=per_seed_q, primary=primary, valid=valid, origin=origin, bundle=bundle
    )
    contract = pl.read_parquet(cand / "outputs_contract.parquet")
    assert contract.height == int(mask.sum()), (contract.height, int(mask.sum()))
    want_mw = contract.select([f"{lv}_mw" for lv in LEVEL_NAMES]).to_numpy()
    capacity = np.array([p["capacity_mw"] for p in bundle["primaries"]["primaries"]])[primary[mask]]
    worst["contract_mw_over_capacity"] = max(
        worst["contract_mw_over_capacity"],
        float((np.abs(mw_mean[mask] - want_mw) / capacity[:, None]).max()),
    )
    assert k1_config["n_extra"] == 0
    print(f"{origin_dir}: rows {rows.height}, history rows {hist_rows.height}", flush=True)


STAGE_TOLERANCE = {
    "k1_lags_masked_median": "tolerance",
    "k1_lags_masked_q05_q95": "tolerance",
    "table_t_entries": "tolerance",
    "input_matrix_base": "tolerance",
    "input_matrix_signals": "tolerance",
    "raw_output": "tolerance",
    "native_seed_stored_matrix": "quantile_tolerance",
    "native_seed": "quantile_tolerance",
    "native_seed_mean": "quantile_tolerance",
    "contract_mw_over_capacity": "quantile_tolerance",
}


def check(*, root: Path, tolerance: float, quantile_tolerance: float) -> bool:
    """Reproduce every fixture; print the largest errors; return True when all are within tolerance."""
    bundle = load_bundle(root)
    table = expected_table(bundle=bundle)
    present = weather_present(bundle=bundle)
    worst = dict.fromkeys(STAGE_TOLERANCE, 0.0)
    for origin_dir in sorted((root / "fixtures" / "candidate").glob("origin=*")):
        check_origin(
            root=root,
            origin_dir=origin_dir.name,
            bundle=bundle,
            table=table,
            present=present,
            worst=worst,
        )
    limits = {"tolerance": tolerance, "quantile_tolerance": quantile_tolerance}
    ok = True
    for name, value in worst.items():
        limit = limits[STAGE_TOLERANCE[name]]
        ok &= value <= limit
        print(
            f"max error {name}: {value:.3e} ({'ok' if value <= limit else 'FAIL'}, tolerance {limit:g})"
        )
    return ok


def main() -> None:
    """Run the check on the bundle that contains this file, or on the directory given."""
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", nargs="?", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="inputs, table T, K1 medians, raw outputs"
    )
    parser.add_argument(
        "--quantile-tolerance", type=float, default=2e-5,
        help="quantiles (fraction of capacity; MW errors are divided by capacity)",
    )  # fmt: skip
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = datetime.now(UTC)
    ok = check(
        root=Path(args.bundle), tolerance=args.tolerance, quantile_tolerance=args.quantile_tolerance
    )
    print(
        f"{'PASSED' if ok else 'FAILED'} in {(datetime.now(UTC) - started).total_seconds():.0f} s"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

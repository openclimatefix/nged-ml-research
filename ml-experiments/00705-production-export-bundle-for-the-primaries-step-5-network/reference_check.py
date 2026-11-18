# ruff: noqa: E501, PLR0915
"""Reference implementation of the step-5 network's inference, checked against a bundle's fixtures.

This file is copied into every bundle. It imports nothing from the research repository, reads only
files inside the bundle directory, and needs only `torch`, `numpy`, and `polars` (the Parquet
reader). Production can copy the functions below.

    python reference_check.py [BUNDLE_DIR] [--tolerance 1e-5] [--quantile-tolerance 2e-5]

For every fixture origin and every seed the script (1) rebuilds the 60-column input matrix from the
raw columns of `input_matrix.parquet`, the lag values and the expected-weather rows, with the
standardisers of `components/step5/standardisers.json`, and compares it with the stored
standardised matrix; (2) runs the forward pass on CPU and the quantile read-out; (3) compares the 11
quantiles with the stored ones (fraction of capacity, native 09:00 origin), after the seed average
and sort, and after the 12:00 relabel times capacity. Exit status 1 if any error exceeds the
tolerance. The stored forecasts come from a GPU pass; the CPU pass differs by float32 rounding
(about 3e-6 in the raw outputs), which the quantile bisection amplifies to about 1e-5 of capacity.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812

LEVEL_NAMES = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")


def read_json(path: Path) -> dict:
    """Return the JSON file's content."""
    return json.loads(path.read_text())


def load_bundle(root: Path) -> dict:
    """Load every non-fixture file of the bundle into one dict."""
    manifest = read_json(root / "manifest.json")
    assert manifest["bundle_format_version"] == "1", manifest["bundle_format_version"]
    step5 = root / "components" / "step5"
    bundle = {
        "manifest": manifest,
        "config": read_json(root / "config.json"),
        "primaries": read_json(root / "primaries.json"),
        "columns": read_json(step5 / "columns.json"),
        "standardisers": read_json(step5 / "standardisers.json"),
        "expected_rows": pl.read_parquet(step5 / "expected_weather_rows.parquet"),
    }
    bundle["weights"] = {
        seed: torch.load(
            step5 / "weights" / f"seed{seed}.pt", map_location="cpu", weights_only=True
        )
        for seed in manifest["components"]["step5"]["seeds"]
    }
    return bundle


# --- inputs -------------------------------------------------------------------------------------


def standardise_x37(*, raw: np.ndarray, standardisers: dict, clip_sd: float) -> np.ndarray:
    """Return the first 37 columns: (raw - mean) / std in float32, clipped, NaN replaced by 0.

    The research standardiser holds float32 means and standard deviations, so the arithmetic is float32.
    """
    mean = np.asarray(standardisers["x37"]["mean"], dtype=np.float32)
    std = np.asarray(standardisers["x37"]["std"], dtype=np.float32)
    z = np.clip((raw - mean) / std, -clip_sd, clip_sd)
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


def lag_columns(
    *, power: np.ndarray, lag_index: np.ndarray, target_index: np.ndarray, which: int, bundle: dict
) -> np.ndarray:
    """Return one lag's value, missing flag, and age code (rows, 3), float32.

    `power` is the lag's cleaned power as a fraction of the Primary's effective capacity (NaN when
    missing), `lag_index` the grid index of the lag's half-hour (negative when absent), `which` 0 for
    the day lag and 1 for the week lag.
    """
    config = bundle["config"]["components"]["step5"]["lags"]
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


def build_input_matrix(*, rows: pl.DataFrame, lag_power: dict, bundle: dict) -> np.ndarray:
    """Return the (rows, 60) standardised matrix from the fixture's raw columns.

    `rows` holds the `input_matrix.parquet` rows of one origin; `lag_power` maps
    (primary_index, grid_index) to the cleaned power fraction (NaN when missing).
    """
    names = [c["name"] for c in bundle["columns"]]
    config = bundle["config"]["components"]["step5"]
    standardisers = bundle["standardisers"]
    n = rows.height
    x = np.zeros((n, config["n_columns"]), dtype=np.float32)
    raw37 = rows.select([f"raw__{name}" for name in names[:37]]).to_numpy().astype(np.float32)
    x[:, :37] = standardise_x37(
        raw=raw37, standardisers=standardisers, clip_sd=config["input_clip_sd"]
    )
    primary = rows["primary_index"].to_numpy()
    target = rows["target_grid_index"].to_numpy()
    for which, name in enumerate(("day", "week")):
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
    """Return the standardised expected-weather rows by (primary, grid index).

    The rows are float16, as in the research network.
    """
    config = bundle["config"]["components"]["step5"]
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


# --- network ------------------------------------------------------------------------------------


def mlp(*, w: dict, prefix: str, x: torch.Tensor, layers: tuple[int, ...]) -> torch.Tensor:
    """Apply Linear layers named `prefix.<i>` with an exact GELU between them (dropout is off at inference)."""
    for k, i in enumerate(layers):
        x = F.linear(x, w[f"{prefix}.{i}.weight"], w[f"{prefix}.{i}.bias"])
        if k < len(layers) - 1:
            x = F.gelu(x)
    return x


def forward(
    *, w: dict, x: torch.Tensor, primary: torch.Tensor, table: dict, config: dict
) -> torch.Tensor:
    """Return the nine raw outputs (three logits, three means, three log standard deviations), float32."""
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
    outs = []
    for j, name in enumerate(("day", "week")):
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
    config = bundle["config"]["components"]["step5"]
    target = bundle["standardisers"]["target"]
    mean = torch.tensor(target["mean"], dtype=torch.float64)[torch.from_numpy(primary)][:, None]
    std = torch.tensor(target["std"], dtype=torch.float64)[torch.from_numpy(primary)][:, None]
    q = torch.sort(quantiles(raw=raw, config=config) * std + mean, dim=1).values
    return q.float().numpy()


def contract_rows(
    *,
    per_seed: list[np.ndarray],
    primary: np.ndarray,
    valid: np.ndarray,
    origin: np.ndarray,
    bundle: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask, per-seed MW list stacked, seed-mean MW) of the 12:00 contract rows.

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


def check(*, root: Path, tolerance: float, quantile_tolerance: float) -> bool:
    """Reproduce every fixture; print the largest errors; return True when all are within tolerance."""
    bundle = load_bundle(root)
    config = bundle["config"]["components"]["step5"]
    table = expected_table(bundle=bundle)
    seeds = bundle["manifest"]["components"]["step5"]["seeds"]
    ok = True
    worst = {
        "input_matrix": 0.0,
        "raw_output": 0.0,
        "native_seed": 0.0,
        "native_seed_mean": 0.0,
        "contract_mw_over_capacity": 0.0,
        "lags_removed": 0.0,
    }
    for origin_dir in sorted((root / "fixtures" / "step5").glob("origin=*")):
        rows = pl.read_parquet(origin_dir / "input_matrix.parquet")
        lags = pl.read_parquet(origin_dir / "lag_and_expected_rows.parquet")
        lag_power = {
            (int(p), int(i)): float(v)
            for p, i, v in zip(
                lags["primary_index"], lags["grid_index"], lags["power_fraction"], strict=True
            )
        }
        x = build_input_matrix(rows=rows, lag_power=lag_power, bundle=bundle)
        names = [c["name"] for c in bundle["columns"]]
        stored = rows.select([f"std__{name}" for name in names]).to_numpy().astype(np.float32)
        err = float(np.abs(x - stored).max())
        worst["input_matrix"] = max(worst["input_matrix"], err)
        out = pl.read_parquet(origin_dir / "outputs_native.parquet")
        assert out.select("primary_index", "valid_time").equals(
            rows.select("primary_index", "valid_time")
        )
        primary = rows["primary_index"].to_numpy().astype(np.int64)
        xt, pt = torch.from_numpy(stored), torch.from_numpy(primary)
        per_seed = []
        with torch.no_grad():
            for seed in seeds:
                weights = bundle["weights"][seed]
                raw = forward(w=weights, x=xt, primary=pt, table=table, config=config)
                want_raw = out.select([f"raw_out_{k}_s{seed}" for k in range(9)]).to_numpy()
                worst["raw_output"] = max(
                    worst["raw_output"], float(np.abs(raw.numpy() - want_raw).max())
                )
                q = native_fractions(raw=raw, primary=primary, bundle=bundle)
                per_seed.append(q)
                want = out.select([f"{lv}_s{seed}" for lv in LEVEL_NAMES]).to_numpy()
                worst["native_seed"] = max(worst["native_seed"], float(np.abs(q - want).max()))
                if origin_dir.name == "origin=" + config["variant_origin"]:
                    xr = stored.copy()
                    for c0, age in (
                        (37, config["lags"]["missing_age_code"]["day"]),
                        (40, config["lags"]["missing_age_code"]["week"]),
                    ):
                        xr[:, c0], xr[:, c0 + 1], xr[:, c0 + 2] = 0.0, 1.0, age
                    ro = forward(
                        w=weights, x=torch.from_numpy(xr), primary=pt, table=table, config=config
                    )
                    qr = native_fractions(raw=ro, primary=primary, bundle=bundle)
                    rem = pl.read_parquet(origin_dir / "outputs_lags_removed.parquet")
                    wantr = rem.select([f"{lv}_s{seed}" for lv in LEVEL_NAMES]).to_numpy()
                    worst["lags_removed"] = max(
                        worst["lags_removed"], float(np.abs(qr - wantr).max())
                    )
        mean = np.sort(np.mean(np.stack(per_seed), axis=0, dtype=np.float32), axis=1)
        want_mean = out.select([f"{lv}_mean" for lv in LEVEL_NAMES]).to_numpy()
        worst["native_seed_mean"] = max(
            worst["native_seed_mean"], float(np.abs(mean - want_mean).max())
        )
        valid = rows["valid_time"].dt.epoch("us").to_numpy().astype("datetime64[us]")
        origin = rows["origin_time"].dt.epoch("us").to_numpy().astype("datetime64[us]")
        mask, _, mw_mean = contract_rows(
            per_seed=per_seed, primary=primary, valid=valid, origin=origin, bundle=bundle
        )
        contract = pl.read_parquet(origin_dir / "outputs_contract.parquet")
        assert contract.height == int(mask.sum()), (contract.height, int(mask.sum()))
        want_mw = contract.select([f"{lv}_mw" for lv in LEVEL_NAMES]).to_numpy()
        capacity = np.array([p["capacity_mw"] for p in bundle["primaries"]["primaries"]])[
            primary[mask]
        ]
        rel = float((np.abs(mw_mean[mask] - want_mw) / capacity[:, None]).max())
        worst["contract_mw_over_capacity"] = max(worst["contract_mw_over_capacity"], rel)
        print(
            f"{origin_dir.name}: rows {rows.height}, input matrix max abs error {err:.2e}",
            flush=True,
        )
    for name, value in worst.items():
        limit = tolerance if name in {"input_matrix", "raw_output"} else quantile_tolerance
        verdict = "ok" if value <= limit else "FAIL"
        ok &= value <= limit
        print(f"max error {name}: {value:.3e} ({verdict}, tolerance {limit:g})")
    return ok


def main() -> None:
    """Run the check on the bundle that contains this file, or on the directory given."""
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", nargs="?", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="inputs and raw network outputs"
    )
    parser.add_argument(
        "--quantile-tolerance",
        type=float,
        default=2e-5,
        help="quantiles (fraction of capacity; MW errors are divided by capacity)",
    )
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

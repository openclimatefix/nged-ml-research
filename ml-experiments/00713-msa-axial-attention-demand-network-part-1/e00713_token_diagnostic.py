# ruff: noqa: E501
"""Experiment 713: report-only diagnostic of the token's scale and of the forecast's sensitivity to history power.

Usage: e00713_token_diagnostic.py ARM MONTH SEED [--device cuda] [--random-init]

For one fitted network (`work/` stage 3 state; `--random-init` builds an untrained network for a smoke
test and labels the file), on `N_SAMPLES` real batches of one fold (random Primaries and test origins, lead
days 1 and 7), the script writes `work/token_diagnostic/<arm>_<month>_s<seed>.json` with:

(a) **The mean length of each part of the token.** The L2 length over the 64 numbers of the weather part
(`proj_w` of the weather encoder), the date-time part, the Primary table, the time-of-day table, the
day-age table, and the power part (`z * power_u + power_b` on visible history tokens, with its
`z * power_u` and `power_b` pieces), on visible history tokens; and the lead, ensemble-spread, and
`m_target` parts on the target row.

(b) **How far the forecast moves with history power.** The sensitivity of the standardised mixture mean
of the 48 target tokens to each visible history reading: the sum over the visible tokens of the absolute
derivative with respect to `z`, averaged over the samples. The derivative is split into the part
through the read-out `base` and `spread` (the trunk output held fixed) and the part through the trunk
(`tokens -> blocks -> h0`).

(c) **The distance between the masked token and a `z = 0` token.** The L2 distance between the token
with `m_hist` and the same token at `z = 0` (`power_b` in place of `m_hist`), before and after the
first LayerNorm of the first block.

The script is report-only: it fits nothing, changes no stored forecast, and feeds no verdict.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

import e00713_arms as arms
import e00713_bank as bank
import e00713_inputs as inp
import e00713_layout as lay
import e00713_model as mdl
import e00713_train as tr
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from torch.nn import functional  # ty: ignore[unresolved-import]

torch.set_num_threads(1)
N_SAMPLES: Final = 256
LEADS: Final = (0, 6)


def out_file(*, arm: str, month: str, seed: int, random_init: bool) -> Path:
    """Return the diagnostic's own file."""
    suffix = "_RANDOM_INIT_SMOKE_TEST" if random_init else ""
    return (
        arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR)
        / "work"
        / "token_diagnostic"
        / f"{arm}_{month}_s{seed}{suffix}.json"
    )


def part_lengths(*, model: mdl.MsaNet, x: mdl.MsaInputs) -> dict[str, float]:
    """Return the mean L2 length of each token part on visible history tokens and on the target row."""
    b = len(x.primary)
    emb = model.embedding(x.primary)
    weather = torch.cat([x.top_weather[:, None], x.hist_weather], dim=1)
    datetime_ = torch.cat([x.top_datetime[:, None], x.hist_datetime], dim=1)
    rows = weather.shape[1]
    e = emb[:, None, None, :].expand(b, rows, mdl.N_COLS, emb.shape[-1])
    w_in = torch.cat([weather[..., : mdl.N_WEATHER], e, weather[..., mdl.N_WEATHER :]], dim=-1)
    pw = model.proj_w(functional.gelu(model.weather(w_in)))
    pd = model.proj_d(functional.gelu(model.datetime(torch.cat([datetime_, e], dim=-1))))
    vis = x.hist_visible & x.row_exists[:, :, None]  # (B, 14, 48)
    z = x.hist_power.clamp(-mdl.POWER_CLIP, mdl.POWER_CLIP)

    def mean_len(v: torch.Tensor, mask: torch.Tensor) -> float:
        return float(v.norm(dim=-1)[mask].mean())

    tok = model.tok_emb(x.primary)[:, None, None, :].expand(b, 14, mdl.N_COLS, mdl.WIDTH)
    col = model.col_emb.weight[None, None].expand(b, 14, mdl.N_COLS, mdl.WIDTH)
    age = model.age_emb.weight[None, :, None, :].expand(b, 14, mdl.N_COLS, mdl.WIDTH)
    zu = z[..., None] * model.power_u
    power = zu + model.power_b
    top = torch.ones(b, mdl.N_COLS, dtype=torch.bool, device=z.device)
    return {
        "weather": mean_len(pw[:, 1:], vis),
        "datetime": mean_len(pd[:, 1:], vis),
        "primary": mean_len(tok, vis),
        "time_of_day": mean_len(col, vis),
        "day_age": mean_len(age, vis),
        "power": mean_len(power, vis),
        "power_z_times_u": mean_len(zu, vis),
        "power_b": float(model.power_b.norm()),
        "m_hist": float(model.m_hist.norm()),
        "target_weather": mean_len(pw[:, :1].squeeze(1), top),
        "target_datetime": mean_len(pd[:, :1].squeeze(1), top),
        "lead": float(model.lead_emb(x.lead_day).norm(dim=-1).mean()),
        "spread": mean_len(model.proj_s(x.top_spread), top),
        "m_target": float(model.m_target.norm()),
    }


def sensitivity(*, model: mdl.MsaNet, x: mdl.MsaInputs) -> dict[str, float]:
    """Return the summed absolute derivative of the mixture mean with respect to visible history power, by path."""
    z_tokens = x.hist_power.clone().requires_grad_(True)
    z_readout = x.hist_power.clone().requires_grad_(True)
    tokens_in = mdl.MsaInputs(**{**x.__dict__, "hist_power": z_tokens})
    readout_in = mdl.MsaInputs(**{**x.__dict__, "hist_power": z_readout})
    h, row_exists = model._tokens(tokens_in)
    for block in model.blocks:
        h = block(h, row_exists=row_exists)
    h0, base, spread, no_history, _ = model._readout(h, readout_in)
    out = model.head(
        torch.cat([h0, base[..., None], spread[..., None], no_history[..., None]], dim=-1)
    )
    k = mdl.m559.COMPONENTS
    out = torch.cat(
        [out[..., :k], out[..., k : 2 * k] + base[..., None], out[..., 2 * k :]], dim=-1
    )
    y = mdl.standardised_mixture_mean(
        out=out
    ).sum()  # samples are independent, so one backward pass
    y.backward()
    vis = (x.hist_visible & x.row_exists[:, :, None]).to(z_tokens.dtype)
    assert z_tokens.grad is not None
    assert z_readout.grad is not None
    via_trunk = (z_tokens.grad * vis).abs().sum(dim=(1, 2)).mean()
    via_readout = (z_readout.grad * vis).abs().sum(dim=(1, 2)).mean()
    return {"through_trunk": float(via_trunk), "through_readout_base": float(via_readout)}


def mask_distance(*, model: mdl.MsaNet, x: mdl.MsaInputs) -> dict[str, float]:
    """Return the L2 distance between the `m_hist` token and the `z = 0` token, before and after the first LayerNorm."""
    with torch.no_grad():
        # a token differs between the two cases only in its power part, so the rest cancels before the norm
        # only if the LayerNorm is linear; measure both on a real token instead
        masked = mdl.MsaInputs(**{**x.__dict__, "hist_visible": torch.zeros_like(x.hist_visible)})
        zero = mdl.MsaInputs(
            **{
                **x.__dict__,
                "hist_power": torch.zeros_like(x.hist_power),
                "hist_visible": torch.ones_like(x.hist_visible),
            }
        )
        t_masked, _ = model._tokens(masked)
        t_zero, _ = model._tokens(zero)
        block = model.blocks[0]
        ln: torch.nn.Module = block.ln_within or block.ln_across or block.ln_transition  # ty: ignore[invalid-assignment]
        raw = (t_masked[:, 1:] - t_zero[:, 1:]).norm(dim=-1).mean()
        normed = (ln(t_masked[:, 1:]) - ln(t_zero[:, 1:])).norm(dim=-1).mean()
        scale = ln(t_zero[:, 1:]).norm(dim=-1).mean()
    return {
        "raw": float(raw),
        "after_first_layernorm": float(normed),
        "token_length_after_first_layernorm": float(scale),
    }


def main() -> int:
    """Measure one fitted network on real batches and write the file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("month")
    parser.add_argument("seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-init", action="store_true")
    args = parser.parse_args()
    path = out_file(arm=args.arm, month=args.month, seed=args.seed, random_init=args.random_init)
    spec = arms.resolve(arm=args.arm, data_dir=ML_EXPERIMENTS_DATA_DIR)
    assert not spec.switches.top_row_only, "arm N has no history tokens"
    pipe = inp.load_pipeline(
        month=args.month, phase="predict", chain_root=inp.chain_root_of_seed(seed=args.seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=args.device)
    if args.random_init:
        torch.manual_seed(0)
        model = mdl.MsaNet(switches=spec.switches).to(args.device).eval()
    else:
        model = bank.load_model(
            spec=spec,
            arm=args.arm,
            month=args.month,
            seed=args.seed,
            device=args.device,
            data_dir=ML_EXPERIMENTS_DATA_DIR,
        )
    assert data.test_origin_idx is not None
    rng = np.random.default_rng(713)
    c_all = data.test_origin_idx + lay.NATIVE_OFFSET_STEPS
    pool = tr.block_pool(clean=data.clean)
    result: dict = {
        "arm": args.arm,
        "month": args.month,
        "seed": args.seed,
        "random_init": args.random_init,
        "n_samples": N_SAMPLES,
    }
    for lead in LEADS:
        primary = rng.integers(0, 16, N_SAMPLES)
        c = rng.choice(c_all, N_SAMPLES)
        samples = tr.SampleSet(primary, c, np.zeros(N_SAMPLES, bool))
        if spec.placebo:
            samples.ages = tr.placebo_ages_for(primary=primary, c_idx=c, pool=pool)
        b = tr.batch_for(
            source=data.source,
            samples=samples,
            index=np.arange(N_SAMPLES),
            lead=np.full(N_SAMPLES, lead),
            ensemble=True,
            cut=10**9,
            dropout=None,
        )
        with torch.no_grad():
            lengths = part_lengths(model=model, x=b.inputs)
        result[f"lead_day_{lead + 1}"] = {
            "part_lengths": lengths,
            "sensitivity_abs_sum_per_sample": sensitivity(model=model, x=b.inputs),
            "mask_distance": mask_distance(model=model, x=b.inputs),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.rename(path)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

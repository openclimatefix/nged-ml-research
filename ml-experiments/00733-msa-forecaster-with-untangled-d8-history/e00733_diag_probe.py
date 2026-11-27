# ruff: noqa: E501
"""Experiment 733: post hoc probe of what a fitted Y0 network knows about the sign of its history.

Question: can the network already read the sign of each history half-hour from its own hidden
state, without the untangler's signals? The probe runs a fitted network on CPU over samples of the
synthetic VALIDATION bank (never test rows) and takes, for each history token, the encoder output:
the token's 64-value state after the last axial-attention block and its layer-normalised form, which
the network's read-out layers take (`UntangledNet.ln_readout`). A logistic regression then predicts
the true sign of the token's half-hour (reverse flow, X < 0) from that state. The fit uses tokens
of some segments, and the scores come from tokens of other segments. Beside the probe sit the
untangler's stored probability of reverse flow at the same tokens and the training base rate. The
YU network is probed too; its history tokens carry the untangler's signals, so its probe reads
them directly. The reading is post hoc, report-only, and changes no verdict.

Run it through `run.sh --go e00733_diag_probe.py`; it writes `diagnostics/probe.json` and
`diagnostics/probe.svg` under the experiment's data folder.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_diag as diag
import e00733_paths as paths
import e00733_predictions as predictions
import e00733_synth as sy
import e00733_train as train
import e00733_world as world
import numpy as np

# isort: split
import altair as alt
import e00729_figures as figures729
import polars as pl
import torch  # ty: ignore[unresolved-import]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

N_SAMPLES: Final = 2000
"""The validation samples whose history tokens are probed."""
TOKENS_PER_SAMPLE: Final = 40
TEST_SHARE: Final = 0.3
BATCH: Final = 100
SAMPLE_SEED: Final = 7331
SPLIT_SEED: Final = 7332
ARMS: Final = ("Y0", "YU")
STATES: Final = ("layer_normed", "after_blocks")
"""`layer_normed` is `ln_readout(x)`, which the read-out reads; `after_blocks` is `x` before it."""


def collect(
    *, net: Any, segments: list, index: sy.SampleIndex, rows: np.ndarray, mode: str,
    mode_reference: str, d8_index: int, target_std: float, rng_seed: int,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return the probed tokens: hidden states, true sign, untangler probability, and segment.

    A token is probed when its history row exists, its reading is visible, the true sign is known,
    and the untangler has an output there. The same tokens are probed for every network, because
    the untangler's mask comes from the `mode_reference` batch.

    Args:
        net: The fitted network (evaluation mode).
        segments: The validation bank's segments.
        index: Its samples.
        rows: The samples to run.
        mode: The network's channel mode (`absent` for Y0, `untangled` for YU).
        mode_reference: The mode whose untangler flag selects the tokens (`untangled`).
        d8_index: The Primary index of D8.
        target_std: D8's target standard deviation.
        rng_seed: Seeds the choice of tokens inside each sample.

    Returns:
        `layer_normed` and `after_blocks` `(T, 64)`, `truth`, `p_untangler`, `seg` `(T,)`.
    """
    rng = np.random.default_rng(rng_seed)
    keys = ("layer_normed", "after_blocks", "truth", "p_untangler", "seg")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start : start + BATCH]
            args: dict[str, Any] = {
                "segments": segments,
                "index": index,
                "rows": chunk,
                "d8_index": d8_index,
                "target_std": target_std,
            }
            ref = sy.gather_batch(**args, mode=mode_reference)  # ty: ignore[invalid-argument-type]
            batch = ref if mode == mode_reference else sy.gather_batch(**args, mode=mode)  # ty: ignore[invalid-argument-type]
            truth = ref.history_x_neg
            valid = (
                ref.arrays["row_exists"][:, :, None] & ref.arrays["hist_visible"]
                & ref.arrays["hist_u_ok"] & np.isfinite(truth)
            )  # fmt: skip
            pick = diag.pick_tokens(valid=valid, per_sample=TOKENS_PER_SAMPLE, rng=rng)
            inp = train.synthetic_arrays_to_inputs(batch=batch, device="cpu")
            x, row_exists = net._tokens(inp)
            for block in net.blocks:
                x = block(x, row_exists=row_exists)
            hist_x = x[:, 1:].numpy()
            hist_h = net.ln_readout(x)[:, 1:].numpy()
            parts["layer_normed"].append(hist_h[pick])
            parts["after_blocks"].append(hist_x[pick])
            parts["truth"].append(truth[pick])
            parts["p_untangler"].append(ref.arrays["hist_u"][..., core.P_NEG][pick])
            parts["seg"].append(np.broadcast_to(index.seg[chunk][:, None, None], pick.shape)[pick])
    return {k: np.concatenate(v) for k, v in parts.items()}


def fit_probe(
    *, features: np.ndarray, truth: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray
) -> dict[str, Any]:  # fmt: skip
    """Fit a logistic regression on the train tokens and score it on the test tokens.

    Args:
        features: `(T, F)` hidden states.
        truth: `(T,)` the true sign (1.0 for reverse flow).
        train_mask: The probe-train tokens.
        test_mask: The probe-test tokens (no segment in common with the train tokens).

    Returns:
        The probe's accuracy, Brier score, AUC, base rate, and token count on the test tokens.
    """
    scaler = StandardScaler().fit(features[train_mask])
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(scaler.transform(features[train_mask]), truth[train_mask] > 0.5)
    p = clf.predict_proba(scaler.transform(features[test_mask]))[:, 1]
    return diag.probability_scores(p=p, y=truth[test_mask])


def probe_one_net(*, tokens: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    """Return the probe scores of one network's tokens, with the untangler and base-rate rows."""
    train_mask, test_mask = diag.split_by_segment(
        seg=tokens["seg"], test_share=TEST_SHARE, seed=seed
    )
    truth = tokens["truth"]
    base = float(np.mean(truth[train_mask] > 0.5))
    out: dict[str, Any] = {
        "probe": {s: fit_probe(features=tokens[s], truth=truth, train_mask=train_mask, test_mask=test_mask) for s in STATES},
        "untangler": diag.probability_scores(p=tokens["p_untangler"][test_mask], y=truth[test_mask]),
        "base_rate_predictor": diag.probability_scores(p=np.full(int(test_mask.sum()), base), y=truth[test_mask]),
        "n_train_tokens": int(train_mask.sum()), "n_test_tokens": int(test_mask.sum()),
        "n_train_segments": len(np.unique(tokens["seg"][train_mask])),
        "n_test_segments": len(np.unique(tokens["seg"][test_mask])),
    }  # fmt: skip
    return out


def run_probe(*, root: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    """Probe the fitted Y0 and YU networks of `seeds` on validation-bank samples."""
    import e00733_adapters as adapters

    segments, _ = world.read_world(root=root, kind="validation")
    index = sy.enumerate_samples(segments=segments)
    meta = world.read_meta(root=root)["constants"]
    rows = np.sort(
        np.random.default_rng(SAMPLE_SEED).choice(
            len(index), size=min(N_SAMPLES, len(index)), replace=False
        )
    )
    result: dict[str, Any] = {
        "n_samples": len(rows),
        "n_validation_samples": len(index),
        "nets": {},
    }
    for arm in ARMS:
        for seed in seeds:
            net = predictions.load_net(root=root, arm=arm, fold=None, seed=seed, device="cpu")
            tokens = collect(
                net=net, segments=segments, index=index, rows=rows, mode="absent" if arm == "Y0" else "untangled",
                mode_reference="untangled", d8_index=adapters.d8_index(), target_std=float(meta["std"]), rng_seed=SAMPLE_SEED,
            )  # fmt: skip
            result["nets"][f"{arm}_s{seed}"] = probe_one_net(tokens=tokens, seed=SPLIT_SEED)
            print(f"{arm} seed {seed}: {result['nets'][f'{arm}_s{seed}']['probe']['layer_normed']}")
    result["mean"] = mean_over_seeds(nets=result["nets"])
    return result


def mean_over_seeds(*, nets: dict[str, Any]) -> dict[str, Any]:
    """Return, for each arm, the seed-mean accuracy, Brier score, and AUC of each reading."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        mine = [v for k, v in nets.items() if k.startswith(arm + "_")]
        readings = {
            **{f"probe_{s}": [m["probe"][s] for m in mine] for s in STATES},
            "untangler": [m["untangler"] for m in mine], "base_rate_predictor": [m["base_rate_predictor"] for m in mine],
        }  # fmt: skip
        out[arm] = {
            name: {
                k: float(np.nanmean([r[k] for r in rs]))
                for k in ("accuracy", "brier", "auc", "base_rate")
            }
            for name, rs in readings.items()
        }
    return out


def bar_rows(*, mean: dict[str, Any]) -> pl.DataFrame:
    """Return one row per reading and metric (Brier, AUC) for the bars of the figure."""
    spec = (
        ("Y0 probe", "Y0", "probe_layer_normed"), ("YU probe", "YU", "probe_layer_normed"),
        ("Untangler output", "Y0", "untangler"), ("Base rate", "Y0", "base_rate_predictor"),
    )  # fmt: skip
    out = [
        {"reading": label, "metric": metric, "value": mean[arm][key][metric]}
        for label, arm, key in spec
        for metric in ("brier", "auc")
    ]
    return pl.DataFrame(out)


def probe_chart(*, rows: pl.DataFrame) -> alt.VConcatChart:
    """Return the two bar panels: probe Brier score and AUC, by reading."""
    order = ["Y0 probe", "YU probe", "Untangler output", "Base rate"]
    colours = {
        "Y0 probe": figures729.GREY,
        "YU probe": figures729.BLUE,
        "Untangler output": figures729.ORANGE,
        "Base rate": "#BBBBBB",
    }
    colour = alt.Color(
        "reading:N", scale=alt.Scale(domain=order, range=[colours[n] for n in order]), legend=None
    )
    panels = []
    for metric, title, note in (
        (
            "brier",
            "Brier score (smaller is better)",
            "Brier score of the history sign on probe-test tokens",
        ),
        (
            "auc",
            "AUC (larger is better; 0.5 is chance)",
            "Area under the ROC curve on the same tokens",
        ),
    ):
        data = rows.filter(pl.col("metric") == metric)
        base = alt.Chart(data).encode(
            y=alt.Y("reading:N", sort=order, title=None),
            x=alt.X("value:Q", title=title),
            color=colour,
        )  # ty: ignore[unresolved-attribute]
        bars = base.mark_bar(aria=False)  # ty: ignore[unresolved-attribute]
        labels = base.mark_text(align="left", dx=3, aria=False).encode(  # ty: ignore[unresolved-attribute]
            text=alt.Text("value:Q", format=".3f")
        )  # ty: ignore[unresolved-attribute]
        panels.append((bars + labels).properties(
            width=figures729.WIDTH_PX - 120, height=90,
            title=alt.TitleParams(note, anchor="start", fontSize=12),
        ))  # fmt: skip
    return alt.vconcat(*panels, title=alt.TitleParams(
        ["Can a network read the sign of its history from its own hidden state?",
         "Synthetic validation days. Probe: logistic regression on the 64-value state of each history",
            "token, scored on tokens of segments the regression never saw. Mean of three seeds."],
        anchor="start", fontSize=12, subtitleFontSize=11,
    ))  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write `probe.json` and `probe.svg`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(core.SYNTHETIC_SEEDS))
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)
    paths.pin_threads()
    torch.set_num_threads(1)
    root = args.data_root or paths.data_root()
    folder = diag.diagnostics_folder(root=root)
    result = run_probe(root=root, seeds=tuple(args.seeds))
    result["label"] = "post hoc, report-only; validation-bank samples only"
    (folder / "probe.json").write_text(json.dumps(result, indent=1))
    figures729.save_svg(
        chart=probe_chart(rows=bar_rows(mean=result["mean"])), path=folder / "probe.svg"
    )
    print(json.dumps(result["mean"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

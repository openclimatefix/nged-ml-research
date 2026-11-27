# ruff: noqa: E501, PERF401
"""Experiment 733: attention maps of the forecaster (torch).

For D8's target tokens, the across-day attention (which of the 15 rows each half-hour of the target
row reads) and the within-row attention of the target row come from a forward pass of a saved state
with `return_attention=True`. Every tensor leaves the device through `e00733_train.to_numpy`. The
learned gain `gamma` of the history path is returned beside each map, as the plan prints it.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import e00733_figures as figs
import e00733_fit as fit
import e00733_model as model
import e00733_paths as paths  # noqa: F401
import e00733_train as train
import numpy as np
import polars as pl

# isort: split
import altair as alt
import torch  # ty: ignore[unresolved-import]

TARGET_ROW: Final = 0
"""The index of the target row among the 15 rows of the across-day attention."""


class AttentionError(ValueError):
    """The attention weights are not in the shape the maps need."""


def extract(
    *, net: model.UntangledNet, inp: model.UntInputs, device: str, sample: int = 0
) -> dict[str, np.ndarray]:
    """Return the attention maps and the readout weights of one sample from a forward pass.

    Args:
        net: The fitted network.
        inp: A batch of inputs (on `device`).
        device: "cuda" or "cpu".
        sample: The batch row to extract.

    Returns:
        `across_<i>` `(48, 15)` (head mean), `within_<i>` `(48, 48)` (target row, head mean),
        `alpha`
        `(14, 48)` the similar-day read-out weights, and `gamma` `(64,)` (empty when the arm has no
        path).

    Raises:
        AttentionError: When the network returns no attention, a weight has the wrong shape, or the
            forecast with the flag differs from the forecast without it.
    """
    net.eval()
    with torch.no_grad():
        on_device = inp.to(device)
        out = net(on_device, return_attention=True)
        plain = net(on_device)
    if not torch.equal(plain.out, out.out):
        msg = "the forecast with return_attention=True differs from the forecast without it"
        raise AttentionError(msg)
    if out.attention is None:
        msg = "the network returned no attention weights"
        raise AttentionError(msg)
    maps: dict[str, np.ndarray] = {}
    for name, weights in out.attention.items():
        w = train.to_numpy(tensor=weights)[sample]
        if name.startswith("across"):
            if w.ndim != 3 or w.shape[0] != 48:
                msg = f"the across-day weights {name} have shape {w.shape}, not (48, heads, rows)"
                raise AttentionError(msg)
            maps[name] = w.mean(axis=1)
        else:
            if w.ndim != 4:
                msg = (
                    f"the within-row weights {name} have shape {w.shape}, not (rows, heads, 48, 48)"
                )
                raise AttentionError(msg)
            maps[name] = w[TARGET_ROW].mean(axis=0)
    maps["alpha"] = train.to_numpy(tensor=out.alpha)[sample]
    gamma = getattr(net, "u_gamma", None)
    maps["gamma"] = train.to_numpy(tensor=gamma) if gamma is not None else np.zeros(0)
    return maps


def gamma_line(*, gamma: np.ndarray) -> str:
    """Return the line printed beside a map: the size of the learned gain of the history path.

    (no value of a series).
    """
    if gamma.size == 0:
        return "no untangled history path"
    return f"learned gain of the history path: mean absolute {float(np.abs(gamma).mean()):.3f}, largest {float(np.abs(gamma).max()):.3f}"


def heatmap(*, matrix: np.ndarray, x_title: str, y_title: str, title: str) -> alt.Chart:
    """Return a heatmap of an attention matrix (rows on the y axis).

    With one colour scale from 0 to the largest weight.
    """
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        msg = "an attention map must be a finite two-dimensional array"
        raise AttentionError(msg)
    rows, cols = matrix.shape
    data = pl.DataFrame(
        {
            "x": np.tile(np.arange(cols), rows),
            "y": np.repeat(np.arange(rows), cols),
            "weight": matrix.ravel(),
        }
    )
    return (
        alt.Chart(data)
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:O", title=x_title, axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("y:O", title=y_title, axis=alt.Axis(labelOverlap=True)),
            color=alt.Color(
                "weight:Q",
                scale=alt.Scale(scheme="blues", domain=[0, float(matrix.max())]),
                title="Attention weight",
            ),
        )
        .properties(
            width=figs.WIDTH_PX / 2 - 40, height=150, title=alt.TitleParams(title, anchor="start")
        )
    )


def attention_figure(
    *, maps: Mapping[str, np.ndarray], label: str, compact: bool = False
) -> alt.VConcatChart:
    """Return one case's figure.

    The across-day and the within-row maps of each block and the read-out weights, with the gain
    line. A compact figure draws the across-day map of the last block and the read-out weights only,
    which keeps one arm's file under 500 KB.
    """
    panels = []
    across = sorted(k for k in maps if k.startswith("across"))
    for name in across[-1:] if compact else across:
        panels.append(
            heatmap(
                matrix=maps[name].T,
                x_title="Half-hour of the target day",
                y_title="Row (0 target, 1 to 14 history)",
                title=f"{label}: across-day attention, block {name[-1]}",
            )
        )
    panels.append(
        heatmap(
            matrix=maps["alpha"],
            x_title="Half-hour of the target day",
            y_title="History row",
            title=f"{label}: read-out weights of the similar-day base",
        )
    )
    first_within = [] if compact else sorted(k for k in maps if k.startswith("within"))[:1]
    for name in first_within:
        panels.append(
            heatmap(
                matrix=maps[name],
                x_title="Half-hour read",
                y_title="Half-hour of the target row",
                title=f"{label}: within-row attention of the target row, block {name[-1]}",
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(gamma_line(gamma=maps["gamma"]), anchor="start", fontSize=11)
    )


def write_case(*, maps: Mapping[str, np.ndarray], label: str, path: Path) -> None:
    """Draw one case and write it as SVG (run `svgo` on the file before committing it)."""
    figs.save_svg(chart=attention_figure(maps=maps, label=label), path=path)


# --------------------------------------------------------------------------- part 5c: the `attention` stage

REAL_CASES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("2026-05", ("sunny day", "dull day")),
    ("2026-01", ("winter day",)),
)
"""The fold of each real case; the sunny and the dull day have the highest and the lowest mean
shortwave over the target day, and the winter day has the median."""
REAL_ARMS: Final = (("FU", "FU"), ("F0r", "M*"))
"""The arms drawn on the real cases and the label of each (F0r reproduces M* to 1e-6 of capacity)."""
SYNTHETIC_ARMS: Final = ("YU", "Y0")
LEAD_INDEX: Final = 1
"""The zero-based lead index of lead day 2."""


def pick_real_cases(
    *, shortwave: np.ndarray, names: tuple[str, ...]
) -> dict[str, int]:  # fmt: skip
    """Return the position of each named case among the origins, from the mean target-day shortwave.

    Args:
        shortwave: The mean shortwave over the lead-day-2 target day of each origin.
        names: The cases of the fold: any of "sunny day" (highest), "dull day" (lowest), and
            "winter day" (median).

    Returns:
        The origin position of each case; the positions differ between the cases.

    Raises:
        AttentionError: When there are too few origins to give each case its own, or a name is unknown.
    """
    finite = np.flatnonzero(np.isfinite(shortwave))
    if len(finite) < len(names):
        msg = f"the fold holds {len(finite)} origins with a shortwave value, and {len(names)} cases need one each"
        raise AttentionError(msg)
    order = finite[np.argsort(shortwave[finite], kind="stable")]
    out: dict[str, int] = {}
    for name in names:
        if name == "sunny day":
            out[name] = int(order[-1])
        elif name == "dull day":
            out[name] = int(order[0])
        elif name == "winter day":
            out[name] = int(order[(len(order) - 1) // 2])
        else:
            msg = f"unknown attention case {name!r}"
            raise AttentionError(msg)
    if len(set(out.values())) != len(out):
        msg = "two cases fall on the same origin"
        raise AttentionError(msg)
    return out


def pick_synthetic_case(
    *, segments: Any, sources: list[str], index: Any
) -> int:  # fmt: skip
    """Return the sample row (lead day 2) of an S2 segment with the most reverse flow on its target day.

    Args:
        segments: The world's segments.
        sources: Each segment's solar source.
        index: The samples.

    Returns:
        The row of the sample.

    Raises:
        AttentionError: When no lead-day-2 target day of an S2 segment has reverse flow.
    """
    import e00733_synth as sy

    rows = np.flatnonzero(index.lead_index == LEAD_INDEX)
    best, best_n = -1, 0
    for r in rows:
        seg = int(index.seg[r])
        if sources[seg] != "S2":
            continue
        slots = sy.target_slots(
            origin_day=index.origin_day[r : r + 1], lead_index=index.lead_index[r : r + 1]
        )[0]
        n = int(np.nansum(segments[seg].x_neg[slots]))
        if n > best_n:
            best, best_n = int(r), n
    if best < 0:
        msg = (
            "no S2 segment has reverse flow on a lead-day-2 target day, so no synthetic case exists"
        )
        raise AttentionError(msg)
    return best


def open_fold(
    *, root: Path, arm: str, fold: str, seed: int, device: str, rows: str
) -> tuple[Any, Any, np.ndarray]:
    """Return the fold's network, its source with the arm's channels, and the native origins drawn from.

    Args:
        root: This experiment's data folder.
        arm: The arm.
        fold: The fold.
        seed: The seed.
        device: "cuda" or "cpu".
        rows: "test" or "validation".

    Returns:
        The network, the `UntSource` (with `data` attached as `.data`), and the native origin indices.
    """
    import e00713_inputs as inp713  # ty: ignore[unresolved-import]
    import e00713_train as tr713  # ty: ignore[unresolved-import]
    import e00733_adapters as adapters
    import e00733_predictions as pr

    net = pr.load_net(root=root, arm=arm, fold=fold, seed=seed, device=device)
    pipe = inp713.load_pipeline(
        month=fold, phase="predict" if rows == "test" else "train", chain_root=inp713.chain_root_of_seed(seed=seed)
    )  # fmt: skip
    if device == "cpu":
        adapters.patch_cpu_tables(pipe=pipe)
    data = adapters.build_fold_data_with_run_files(pipe=pipe, device=device)
    tensors, _ = adapters.arm_tensors_for(arm=arm, base=root, data=data, device=device)
    source = train.UntSource(source=data.source, tensors=tensors, root=root)
    if rows == "test":
        origins = data.test_origin_idx
    else:
        world_ = tr713.World(
            source=source, cutoff_idx=data.cutoff_idx, clean=data.clean, target_mean=data.target_mean,
            target_std=data.target_std, month=fold, placebo_pool=None,
        )  # fmt: skip
        origins = pr.validation_origins(world_=world_, data=data)
    source.data = data  # ty: ignore[unresolved-attribute]
    return net, source, np.asarray(origins)


def real_case_maps(
    *, net: model.UntangledNet, source: Any, d8: int, origin: int, device: str
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return the attention maps of D8's lead-day-2 target tokens at one native origin.

    Args:
        net: The fitted network.
        source: The `UntSource` of the fold.
        d8: D8's Primary index.
        origin: The native origin index.
        device: "cuda" or "cpu".

    Returns:
        `extract`'s maps.
    """
    import e00713_layout as lay  # ty: ignore[unresolved-import]
    import e00713_train as tr713  # ty: ignore[unresolved-import]

    samples = tr713.SampleSet(
        np.array([d8]), np.array([origin + lay.NATIVE_OFFSET_STEPS]), np.zeros(1, bool)
    )
    b = train.batch_for(
        source=source, samples=samples, index=np.arange(1), lead=np.array([LEAD_INDEX]), cut=10**9
    )
    return extract(net=net, inp=b.inputs, device=device)


def mean_target_shortwave(*, source: Any, d8: int, origins: np.ndarray) -> np.ndarray:
    """Return the mean reanalysis shortwave over each origin's lead-day-2 target day.

    The shortwave is column 0 of the dense weather.
    """
    import e00713_layout as lay  # ty: ignore[unresolved-import]

    dense = source.source.dense_weather
    out = np.full(len(origins), np.nan)
    for i, o in enumerate(origins):
        c = int(o) + lay.NATIVE_OFFSET_STEPS + LEAD_INDEX * lay.STEPS_PER_DAY
        out[i] = float(dense[d8, c : c + lay.STEPS_PER_DAY, 0].float().mean())
    return out


def cases_of_folds(
    *, root: Path, rows: str, seed: int
) -> tuple[list[tuple[str, tuple[str, ...]]], list[str]]:
    """Return the fold of each real case, with a note when validation rows fall back to a fitted fold.

    Args:
        root: This experiment's data folder.
        rows: "test" or "validation".
        seed: The seed.

    Returns:
        The cases of each fold and the notes.

    Raises:
        AttentionError: When no fold holds a fit of both FU and F0r.
    """
    import e00733_scoring as scoring

    arms = (("FU", None), ("F0r", None))
    plan = scoring.choose_plan(
        root=root, arms=arms, wanted_seeds=(seed,), rows=rows, screen_ok=True, partial_ok=True
    )
    if plan is None:
        msg = "no fold holds a fit and a stored forecast of both FU and F0r, so no attention map can be drawn"
        raise AttentionError(msg)
    wanted = list(REAL_CASES)
    missing = [f for f, _ in wanted if f not in plan.folds]
    if not missing:
        return wanted, []
    if rows == "test":
        msg = f"the folds {missing} have no fit of FU and F0r"
        raise AttentionError(msg)
    note = f"folds {missing} are not fitted in this root; their cases are drawn from the fold {plan.folds[0]}"
    return [(plan.folds[0] if f in missing else f, names) for f, names in wanted], [note]


def draw_real_cases(
    *, root: Path, rows: str, device: str, seed: int, wanted: Any
) -> dict[str, Any]:
    """Draw FU's and M*'s maps on each real case and return the files, shapes, and gains."""
    import e00733_adapters as adapters

    out = root / "figures"
    d8 = adapters.d8_index()
    files: list[str] = []
    cases: dict[str, dict[str, list[int]]] = {}
    gains: dict[str, float] = {}
    for fold, names in wanted:
        opened = {
            label: open_fold(root=root, arm=arm, fold=fold, seed=seed, device=device, rows=rows)
            for arm, label in REAL_ARMS
        }
        _, source, origins = opened["FU"]
        picked = pick_real_cases(
            shortwave=mean_target_shortwave(source=source, d8=d8, origins=origins), names=names
        )
        for name, pos in picked.items():
            maps = {
                label: real_case_maps(
                    net=net, source=src, d8=d8, origin=int(origins[pos]), device=device
                )
                for label, (net, src, _) in opened.items()
            }
            for label, m in maps.items():
                if label == "M*":
                    m["gamma"] = np.zeros(0)
                gains[f"{label} {fold} {name}"] = (
                    float(np.abs(m["gamma"]).mean()) if m["gamma"].size else 0.0
                )
            for label, m in maps.items():
                fname = (
                    f"attention_{label.replace('*', 'star')}_{name.replace(' ', '_')}_{fold}.svg"
                )
                figs.save_svg(
                    chart=attention_figure(maps=m, label=label, compact=True), path=out / fname
                )
                files.append(fname)
            key = f"{name} ({fold})"
            if key in cases:
                msg = f"the case {key} is drawn twice"
                raise AttentionError(msg)
            cases[key] = {label: list(m["alpha"].shape) for label, m in maps.items()}
    return {"files": files, "cases": cases, "gains": gains}


def draw_synthetic_case(*, root: Path, rows: str, device: str, seed: int) -> dict[str, Any]:
    """Draw YU's and Y0's maps on the lead-day-2 target day of an S2 segment with reverse flow."""
    import e00733_adapters as adapters
    import e00733_predictions as pr
    import e00733_signdata as signdata
    import e00733_synth as sy
    import e00733_world as world

    kind = "validation" if rows == "validation" else "test"
    segments, _ = world.read_world(root=root, kind=kind)
    index = sy.enumerate_samples(segments=segments)
    sources = signdata.segment_sources(root=root, kind=kind, segments=segments)
    row = pick_synthetic_case(segments=segments, sources=sources, index=index)
    meta = world.read_meta(root=root)["constants"]
    maps, gains = {}, {}
    for arm in SYNTHETIC_ARMS:
        net = pr.load_net(root=root, arm=arm, fold=None, seed=seed, device=device)
        batch = sy.gather_batch(
            segments=segments,
            index=index,
            rows=np.array([row]),
            d8_index=adapters.d8_index(),
            target_std=float(meta["std"]),
            mode=fit.synthetic_mode(arm=arm),
        )
        maps[arm] = extract(
            net=net, inp=train.synthetic_arrays_to_inputs(batch=batch, device=device), device=device
        )
        if fit.synthetic_mode(arm=arm) == "absent":
            maps[arm]["gamma"] = np.zeros(0)
        gains[f"{arm} synthetic"] = (
            float(np.abs(maps[arm]["gamma"]).mean()) if maps[arm]["gamma"].size else 0.0
        )
    names = []
    for arm, m in maps.items():
        names.append(f"attention_{arm}_synthetic_reverse_flow.svg")
        figs.save_svg(
            chart=attention_figure(maps=m, label=arm, compact=True),
            path=root / "figures" / names[-1],
        )
    return {
        "files": names,
        "cases": {
            "synthetic day with reverse flow": {a: list(m["alpha"].shape) for a, m in maps.items()}
        },
        "gains": gains,
    }


def run_attention(*, root: Path, rows: str, device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    """Draw the attention maps of FU and M* on the real cases and of YU and Y0 on a synthetic case.

    A fold missing from the fitted arms falls back, on validation rows only, to the fold that
    exists, and the note says so.

    Args:
        root: This experiment's data folder.
        rows: "test" or "validation".
        device: "cuda" or "cpu".
        seed: The seed.

    Returns:
        `files`, `cases` (the read-out shape by case), `gains` (the learned gain by arm), and `notes`.
    """
    (root / "figures").mkdir(parents=True, exist_ok=True)
    wanted, notes = cases_of_folds(root=root, rows=rows, seed=seed)
    real = draw_real_cases(root=root, rows=rows, device=device, seed=seed, wanted=wanted)
    synthetic = draw_synthetic_case(root=root, rows=rows, device=device, seed=seed)
    return {
        "files": real["files"] + synthetic["files"], "cases": {**real["cases"], **synthetic["cases"]},
        "gains": {**real["gains"], **synthetic["gains"]}, "notes": notes,
    }  # fmt: skip

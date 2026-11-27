# ruff: noqa: E501
"""Experiment 733: the predict stage, which writes a stored prediction set for each fitted network (torch).

`e00733_store` names the sets (`<job>`, `<job>_masked` for R4, `<job>_stored` for R12, each with
`_val` for validation rows) and reads them. A real-fold set is written through
`experiment_store.write_predictions`, and a synthetic fit writes one archive per split under
`predictions/<job>/<split>.npz`. Test rows are the fold's scored origins, or the synthetic test
bank. Validation rows are the fold's held-out validation origins whose targets lie before the
fold's start, or the synthetic validation bank; no test month is read for them.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import e00733_adapters as adapters
import e00733_core as core
import e00733_fit as fit
import e00733_leakage as leak
import e00733_model as model
import e00733_predict as pred
import e00733_stages as st
import e00733_store as store
import e00733_synth as sy
import e00733_train as train
import e00733_world as world
import numpy as np

# isort: split
import e00713_layout as lay  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]


def validation_origins(*, world_: object, data: object) -> np.ndarray:
    """Return the native origin indices of the fold's validation origins that have every ensemble row.

    The origins are those of the stage 3 samples that experiment 713's purge holds out. An origin
    is kept only when every Primary has ensemble inputs for all 14 lead days, so a forecast of it
    needs no row that the training window lacks.

    Args:
        world_: Experiment 713's `World` of the fold, built on the training phase.
        data: Experiment 713's `FoldData` (for the ensemble table).

    Returns:
        The sorted native origin indices.

    Raises:
        store.PredictionError: When the fold holds no validation origin with every ensemble row.
    """
    import e00713_train as tr713  # ty: ignore[unresolved-import]

    samples = tr713.make_samples(world=world_, stage=3, placebo=False)  # ty: ignore[invalid-argument-type]
    contract = np.unique(samples.c_idx[samples.is_val])
    ens = data.source.ens  # ty: ignore[unresolved-attribute]
    filled = ens.filled.cpu().numpy()
    day = contract // lay.STEPS_PER_DAY - ens.first_day
    inside = (day >= 0) & (day < filled.shape[1])
    keep = np.zeros(len(contract), dtype=bool)
    keep[inside] = filled[:, day[inside], :].all(axis=(0, 2))
    origins = contract[keep] - lay.NATIVE_OFFSET_STEPS
    if not len(origins):
        msg = "the fold holds no validation origin with ensemble inputs for every lead day"
        raise store.PredictionError(msg)
    return np.sort(origins)


def load_net(
    *, root: Path, arm: str, fold: str | None, seed: int, device: str
) -> model.UntangledNet:
    """Return the fitted network of a job, in evaluation mode, on `device`.

    Args:
        root: This experiment's data folder.
        arm: The arm.
        fold: The fold, or None on the synthetic world.
        seed: The seed.
        device: "cuda" or "cpu".

    Returns:
        The network.

    Raises:
        store.PredictionError: When the fit's state file is missing.
    """
    spec = core.ARMS[arm]
    if fold is None:
        path = fit.network_path(root=root, arm=arm, fold=None, seed=seed)
        mode = fit.synthetic_mode(arm=arm)
        net = model.UntangledNet(seed=seed, path=mode != "absent", sign_head=True, clim_logit=0.0)
    else:
        path = fit.network_path(root=root, arm=arm, fold=fold, seed=seed, stage=3)
        net = model.UntangledNet(seed=seed)
    if not path.exists():
        msg = f"{spec.name} has no fitted network at {path.name}: fit it before predicting"
        raise store.PredictionError(msg)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(device).eval()


def target_poison_check(
    *, root: Path, arm: str, fold: str, seed: int, device: str, net: model.UntangledNet, data: object,
    origins: np.ndarray, ids: np.ndarray, grid_start_us: int,
) -> int:  # fmt: skip
    """Check that NaN in every channel value of the target rows of two origins changes no forecast.

    The check forecasts two origins twice, with the channel table as stored and with its values at
    every target half-hour of those origins set to NaN. A forecast that reads a target-row value is
    non-finite or different, and the check raises. The count of poisoned cells goes to
    `scores/target_poison.json`.

    Args:
        root: This experiment's data folder.
        arm: The arm (it must read channels).
        fold: The fold.
        seed: The seed.
        device: "cuda" or "cpu".
        net: The fitted network.
        data: Experiment 713's `FoldData` of the fold.
        origins: The native origin indices of the fold's rows.
        ids: Each Primary's `time_series_id`.
        grid_start_us: The grid start in microseconds.

    Returns:
        The number of flagged target cells that were poisoned.
    """
    pick = origins[:2]
    target = (
        pick[:, None] + lay.NATIVE_OFFSET_STEPS + np.arange(lay.STEPS_PER_DAY * lay.N_LEAD_DAYS)
    ).ravel()

    def run(poison: bool) -> tuple[np.ndarray, int]:
        tensors, _ = adapters.arm_tensors_for(
            arm=arm, base=root, data=data, device=device, poison_idx=target if poison else None
        )  # fmt: skip
        source = train.UntSource(source=data.source, tensors=tensors)  # ty: ignore[unresolved-attribute]
        frame = pred.forecast_frame(
            net=net, test_origin_idx=pick, source=source, target_mean=data.target_mean,  # ty: ignore[unresolved-attribute]
            target_std=data.target_std, seed=seed, grid_start_us=grid_start_us, time_series_ids=ids,  # ty: ignore[unresolved-attribute]
        )  # fmt: skip
        flagged = adapters.table_flags(base=root, n_grid=int(data.clean.shape[1]))  # ty: ignore[unresolved-attribute]
        return frame.select(*pred.QCOLS).to_numpy(), int(
            flagged[target[target < len(flagged)]].sum()
        )

    n = leak.assert_poisoned_targets_change_nothing(run=run)
    path = root / "scores" / "target_poison.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    record[fit.job_name(arm=arm, fold=fold, seed=seed)] = n
    st.write_json(path=path, payload=record)
    return n


def predict_fold(
    *, root: Path, arm: str, fold: str, seed: int, device: str,
    variants: Sequence[str | None] = (None,), rows: str = "test",
) -> dict[str | None, Path]:  # fmt: skip
    """Predict one real-fold fit and write one prediction set per variant.

    The fold's pipeline is built once. A set that exists is not written again.

    Args:
        root: This experiment's data folder.
        arm: F0r, FU, FU-S, or FS.
        fold: The fold.
        seed: The seed.
        device: "cuda" or "cpu".
        variants: The variants (None for the arm itself, "masked", "stored").
        rows: "test" (the scored origins) or "validation" (held-out validation origins only).

    Returns:
        The set folder of each variant.

    Raises:
        store.PredictionError: When the arm is not a fitted real-fold arm, a variant is not allowed for
            the arm, or the keys differ from M*'s stored file.
        CorrectnessFailureError: When the standardiser differs from the fit's.
    """
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00578_common as k578  # ty: ignore[unresolved-import]
    import e00713_inputs as inp713  # ty: ignore[unresolved-import]
    import e00713_train as tr713  # ty: ignore[unresolved-import]

    from studies import experiment_store

    spec = core.ARMS[arm]
    if spec.where != "real" or spec.source == "stored":
        msg = f"{arm} is not a fitted real-fold arm"
        raise store.PredictionError(msg)
    if any(v is not None for v in variants) and arm != "FU":
        msg = "the masked and stored variants are readings of FU (R4, R12)"
        raise store.PredictionError(msg)
    names = {
        v: store.set_name(arm=arm, fold=fold, seed=seed, variant=v, rows=rows) for v in variants
    }
    todo = [v for v in variants if not store.set_folder(root=root, name=names[v]).exists()]
    out = {v: store.set_folder(root=root, name=names[v]) for v in variants}
    if not todo:
        return out
    record = json.loads(fit.record_path(root=root, arm=arm, fold=fold, seed=seed).read_text())
    net = load_net(root=root, arm=arm, fold=fold, seed=seed, device=device)
    pipe = inp713.load_pipeline(
        month=fold, phase="predict" if rows == "test" else "train",
        chain_root=inp713.chain_root_of_seed(seed=seed),
    )  # fmt: skip
    if device == "cpu":
        adapters.patch_cpu_tables(pipe=pipe)
    data = inp713.build_fold_data(pipe=pipe, device=device, keep_rows=False)
    if data.fingerprint != record["fingerprint"]:
        msg = "the standardiser of the prediction phase differs from the fit's (verdict row 1)"
        raise st.CorrectnessFailureError(msg)
    ids = k578.sites().sort("primary")["time_series_id"].to_numpy()
    holdout = None
    if rows == "test":
        assert data.test_origin_idx is not None
        origins = data.test_origin_idx
    else:
        tensors, _ = adapters.arm_tensors_for(arm=arm, base=root, data=data, device=device)
        world_ = tr713.World(
            source=train.UntSource(source=data.source, tensors=tensors), cutoff_idx=data.cutoff_idx,
            clean=data.clean, target_mean=data.target_mean, target_std=data.target_std, month=fold, placebo_pool=None,
        )  # fmt: skip
        origins = validation_origins(world_=world_, data=data)
        holdout = _utc_from_grid(index=data.cutoff_idx)
    if rows == "test" and todo[0] is None and spec.channels:
        target_poison_check(
            root=root, arm=arm, fold=fold, seed=seed, device=device, net=net, data=data,
            origins=origins, ids=ids, grid_start_us=prep422.GRID_START_US,
        )  # fmt: skip
    for variant in todo:
        tensors, _ = adapters.arm_tensors_for(
            arm=arm, base=root, data=data, device=device, variant=variant
        )
        source = train.UntSource(source=data.source, tensors=tensors)
        frame = pred.forecast_frame(
            net=net, test_origin_idx=origins, source=source, target_mean=data.target_mean,
            target_std=data.target_std, seed=seed, grid_start_us=prep422.GRID_START_US,
            time_series_ids=ids, holdout_start=holdout,
        )  # fmt: skip
        if frame.is_empty():
            msg = f"the {rows} prediction of {arm} {fold} seed {seed} holds no row"
            raise store.PredictionError(msg)
        if rows == "test":
            store.assert_keys_match_m_star(frame=frame, fold=fold, seed=seed)
        experiment_store.write_predictions(
            experiment_id=store.EXPERIMENT_ID, name=names[variant], frame=frame, kind="quantiles",
            row_set_id=store.row_set_id(rows=rows), data_dir=store.data_dir(root=root),
        )  # fmt: skip
    return out


def _utc_from_grid(*, index: int) -> Any:
    """Return the datetime of a grid index (experiment 422's grid) in UTC."""
    import datetime as dt

    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    return dt.datetime.fromtimestamp(
        (prep422.GRID_START_US + index * prep422.STEP_US) / 1e6, tz=dt.UTC
    )


def predict_synthetic_arm(
    *, root: Path, arm: str, seed: int, device: str, rows: str = "test"
) -> Path:  # fmt: skip
    """Predict one synthetic fit on the test bank (or the validation bank) and write its archive.

    Args:
        root: This experiment's data folder.
        arm: Y0, YU, or YT.
        seed: The seed.
        device: "cuda" or "cpu".
        rows: "test" or "validation".

    Returns:
        The archive's path, `predictions/<job>/<split>.npz`.
    """
    path = (
        store.set_folder(root=root, name=store.set_name(arm=arm, fold=None, seed=seed))
        / f"{rows}.npz"
    )
    if path.exists():
        return path
    net = load_net(root=root, arm=arm, fold=None, seed=seed, device=device)
    kind = "validation" if rows == "validation" else "test"
    segments, _ = world.read_world(root=root, kind=kind)
    index = sy.enumerate_samples(segments=segments)
    meta = world.read_meta(root=root)["constants"]
    arrays = pred.predict_synthetic(
        net=net, segments=segments, index=index, d8_index=adapters.d8_index(),
        target_std=float(meta["std"]), mode=fit.synthetic_mode(arm=arm), device=device,
    )  # fmt: skip
    pred.check_stored_synthetic(arrays=arrays, n_expected=len(index))
    return pred.write_synthetic(arrays=arrays, path=path)


def predict_job(
    *, root: Path, arm: str, fold: str | None, seed: int, device: str, rows: str = "test"
) -> list[Path]:  # fmt: skip
    """Predict one job by the arm's place: a real-fold fit (with R4 and R12 for FU) or a synthetic fit.

    Args:
        root: This experiment's data folder.
        arm: The arm.
        fold: The fold, or None on the synthetic world.
        seed: The seed.
        device: "cuda" or "cpu".
        rows: "test" or "validation".

    Returns:
        The paths written or found.
    """
    if core.ARMS[arm].where == "synthetic":
        return [predict_synthetic_arm(root=root, arm=arm, seed=seed, device=device, rows=rows)]
    if fold is None:
        msg = f"the real-fold arm {arm} needs a fold"
        raise store.PredictionError(msg)
    variants = store.VARIANTS if arm == "FU" else (None,)
    return list(
        predict_fold(
            root=root, arm=arm, fold=fold, seed=seed, device=device, variants=variants, rows=rows
        ).values()
    )

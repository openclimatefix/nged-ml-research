# ruff: noqa: E501
"""Experiment 721: the training stages, the validation loss, the prediction rows, and the sign tokens.

The loop follows experiment 716's `train_stage` (two optimiser groups, a trunk clip, a clip of the new
parameters, a 300-step warm-up) with these differences, and nothing else:

1. Stages 1, 2, and 3 train from scratch with random encoders.
2. Each batch comes from an `SgSource`: a sample of a supervised Primary has its history rows folded
   with probability 0.5 in the arms with a sign head.
3. The loss is the folded likelihood at the Primaries whose reading is folded (arms `SG-*`), the plain
   mixture likelihood elsewhere, plus the cross-entropy of `ρ` against the true sign.
4. The validation loss that early stopping reads is the loss at the hidden pair, against the folded reading.
5. One `StageTrainable` is one stage's fit, with `train_pass`, `validate`, and state access, so that
   `e00721_fit` can stop, rerun, and continue it.
"""

import time
from dataclasses import dataclass, field

import e00721_paths  # noqa: F401

# isort: split
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_train as pvt  # ty: ignore[unresolved-import]
import e00721_arms as arms
import e00721_batch as sgb
import e00721_core as core
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00558_lags import block_dropout_masks  # ty: ignore[unresolved-import]

PRIMARY_BATCH_DEFAULT = tr.BATCH


@dataclass
class ArmWorld:
    """What one fit of one arm in one world and fold reads.

    Attributes:
        spec: The arm.
        world: 713's `World`; its `source` is the stage-1 `SgSource` and `clean` is the target array.
        sources: The `SgSource` of each stage (1 reads dense weather, 2 and 3 the ensemble table).
        clean_plain: `(16, G)` the history in capacity fractions (shifted in stage 3).
        clean_fold: `(16, G)` the all-folded history in capacity fractions.
        folded_target: `(16,)` bool, the Primaries whose target is a folded reading.
        shift: `(16,)` mean over standard deviation of each Primary's target.
        q_v: `(16,)` the reactive floor in standard deviations.
        gate: `(16,)` the export gate `e_i`.
        val_primaries: The Primaries whose tokens the validation loss reads.
        seed: The seed of the fit.
        fold: The fold label of the prediction rows.
        signed_truth: `(16, G)` signed truth in capacity fractions, NaN where unknown (never read in training).
        reading: `(16, G)` the folded reading in capacity fractions.
        batch_size: The training batch size.
        predict_stage: The stage whose source builds the scored rows.
        score_c_idx: The contract origins of the scored rows.
        score_primaries: The Primaries whose rows are scored.
        train_primaries: The Primaries that make the training pairs, or `None` for all 16.
        stamp_us: `(G,)` the microsecond stamp of every grid index.
        exporters: The synthetic world's exporting Primaries (empty elsewhere).
        non_exporters: The synthetic world's non-exporting Primaries (empty elsewhere).
        log: Facts that the stages record for the job's log (the scale of the PV paths).
    """

    spec: arms.ArmSpec
    world: tr.World
    sources: dict[int, sgb.SgSource]
    clean_plain: np.ndarray
    clean_fold: np.ndarray
    folded_target: np.ndarray
    shift: np.ndarray
    q_v: np.ndarray
    gate: np.ndarray
    val_primaries: tuple[int, ...]
    seed: int
    fold: str
    signed_truth: np.ndarray
    reading: np.ndarray
    batch_size: int = PRIMARY_BATCH_DEFAULT
    predict_stage: int = 2
    score_c_idx: np.ndarray | None = None
    score_primaries: tuple[int, ...] = ()
    train_primaries: tuple[int, ...] | None = None
    stamp_us: np.ndarray | None = None
    exporters: tuple[int, ...] = ()
    non_exporters: tuple[int, ...] = ()
    log: dict = field(default_factory=dict)


def token_loss(
    *, out: torch.Tensor, target: torch.Tensor, primary: torch.Tensor, aw: ArmWorld, folded_arm: bool
) -> torch.Tensor:  # fmt: skip
    """Return the negative log-likelihood `(B, 48)` of each target token (NaN targets give a finite number)."""
    z = torch.nan_to_num(target)
    plain = mdl.per_token_nll(out=out, target=z)
    if not folded_arm:
        return plain
    dev = out.device
    ft = torch.as_tensor(aw.folded_target, device=dev)[primary][:, None]
    shift = torch.as_tensor(aw.shift, dtype=out.dtype, device=dev)[primary]
    q_v = torch.as_tensor(aw.q_v, dtype=out.dtype, device=dev)[primary]
    return torch.where(ft, arms.folded_nll(out=out, reading_z=z, shift=shift, q_v=q_v), plain)


def sign_cross_entropy(*, model: torch.nn.Module, batch: sgb.SgAssembled) -> torch.Tensor:
    """Return the mean cross-entropy of `ρ` against the true sign over the labelled history tokens (0 with none)."""
    rho = getattr(model, "rho_live", None)
    if rho is None or not getattr(model, "sign_head", False):
        return torch.zeros(())
    sign = batch.hist_sign
    ok = torch.isfinite(sign)
    if not bool(ok.any()):
        return rho.sum() * 0.0
    r = rho[ok].clamp(1e-6, 1 - 1e-6)
    s = sign[ok]
    return -(s * torch.log(r) + (1 - s) * torch.log(1 - r)).mean()


def to_device(*, a: np.ndarray, dev: torch.device) -> torch.Tensor:
    """Return `a` as a tensor on `dev`."""
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev)


def batch_for(
    *, source: sgb.SgSource, samples: tr.SampleSet, index: np.ndarray, lead: np.ndarray, cut: int,
    dropout: tuple[np.ndarray, np.ndarray] | None = None, pv_masks: tuple[np.ndarray, np.ndarray] | None = None,
    fold_hist: np.ndarray | None = None,
) -> sgb.SgAssembled:  # fmt: skip
    """Assemble the batch of `samples[index]` at lead days `lead`."""
    dev = source.device
    to = lambda a: to_device(a=a, dev=dev)  # noqa: E731
    return source.assemble(
        primary=to(samples.primary[index]), c_idx=to(samples.c_idx[index]), lead_day=to(lead),
        mask_all_power=None if dropout is None else to(dropout[0]),
        mask_newest_row=None if dropout is None else to(dropout[1]),
        cams_rule=None if pv_masks is None else to(pv_masks[0]),
        convention=None if pv_masks is None else to(pv_masks[1]),
        fold_hist=None if fold_hist is None else to(fold_hist), target_end=cut,
    )  # fmt: skip


def shifted_source(*, aw: ArmWorld, samples: tr.SampleSet) -> sgb.SgSource:
    """Return the stage 3 source, with clean targets.

    Both history arrays carry experiment 581's lag-only shifts of one schedule.
    """
    import e00581_bank as bank  # ty: ignore[unresolved-import]

    first = int(samples.c_idx.min()) - tr.SYNTHETIC_FIRST_MARGIN
    last = int(samples.c_idx.max()) + 672
    seed = int(aw.world.month.replace("-", ""))
    z = []
    for clean in (aw.clean_plain, aw.clean_fold):
        fixed, _ = bank.synthetic_clean(clean=clean, first=first, last=last, seed=seed)
        z.append(
            torch.from_numpy(
                ((fixed - aw.world.target_mean[:, None]) / aw.world.target_std[:, None]).astype(
                    np.float32
                )
            ).to(aw.sources[3].device)
        )
    return aw.sources[3].with_hist(plain_z=z[0], fold_z=z[1])


def pv_calibration(
    *, model: torch.nn.Module, aw: ArmWorld, samples: tr.SampleSet, source: sgb.SgSource, cut: int
) -> dict:
    """Set the stored scales `r_h` and `r_t` of the PV paths and return them.

    The scales come from the training samples of the first stage.
    """
    if not getattr(getattr(model, "pv", None), "any_path", False):
        return {}
    train_idx = np.nonzero(~samples.is_val)[0]
    batches = []
    for start in range(0, len(train_idx), aw.batch_size * 2):
        sel = train_idx[start : start + aw.batch_size * 2]
        lead = (sel % lay.N_LEAD_DAYS).astype(np.int64)
        batches.append(
            batch_for(source=source, samples=samples, index=sel, lead=lead, cut=cut).inputs
        )
    return model.calibrate_scale(batches=batches)  # ty: ignore[invalid-argument-type]


class StageTrainable:
    """One stage's fit of one arm: the trainer of `e00721_fit.Trainable`."""

    def __init__(
        self, *, model: torch.nn.Module, aw: ArmWorld, stage: int, samples: tr.SampleSet
    ) -> None:
        """Build the optimiser and the stage's source; the model is trained in place."""
        self.model, self.aw, self.stage, self.samples = model, aw, stage, samples
        assert stage in {1, 2, 3}, stage
        self.ensemble = stage >= 2
        self.source = shifted_source(aw=aw, samples=samples) if stage == 3 else aw.sources[stage]
        self.cut = tr.target_end(world=aw.world, stage=stage)
        self.folded_arm = aw.spec.loss == "folded"
        self.fold_probability = core.FOLD_HISTORY_SIGN_PROB if aw.spec.sign_head else 0.0
        torch.manual_seed(aw.seed * 7 + stage)
        self.optimiser = torch.optim.AdamW(
            pvt.param_groups(model=model, stage=stage, encoders="random")
        )
        trunk = {id(p) for g in self.optimiser.param_groups if g["trunk"] for p in g["params"]}
        self.trunk_params = [p for p in model.parameters() if id(p) in trunk]
        self.new_params = [
            p for g in self.optimiser.param_groups if not g["trunk"] for p in g["params"]
        ]
        self.train_idx = np.nonzero(~samples.is_val)[0]
        val_ok = np.isin(samples.primary, np.array(aw.val_primaries))
        self.val_idx = np.nonzero(samples.is_val & val_ok)[0]
        self.epoch, self.step = 0, 0

    def train_pass(self) -> None:
        """Run one pass over the training pairs."""
        aw, samples, model = self.aw, self.samples, self.model
        model.train()
        rng = np.random.default_rng([aw.seed, self.stage, self.epoch])
        order = rng.permutation(self.train_idx)
        lead_all = rng.integers(0, lay.N_LEAD_DAYS, size=len(order))
        for batch_index, start in enumerate(range(0, len(order), aw.batch_size)):
            sel = order[start : start + aw.batch_size]
            native = samples.c_idx[sel] - lay.NATIVE_OFFSET_STEPS
            drop = block_dropout_masks(
                origin_idx=native, seed=(aw.seed * 1009 + self.stage) * 1009 + self.epoch
            )
            fold_rng = np.random.default_rng([721, aw.seed, self.stage, self.epoch, batch_index])
            fold_hist = fold_rng.random(len(sel)) < self.fold_probability
            masks = None
            if self.ensemble and getattr(model, "pv", None) is not None and model.pv.any_path:
                import e00716_batch as pvb  # ty: ignore[unresolved-import]

                masks = pvb.pv_dropout_masks(
                    seed=aw.seed, stage=self.stage, epoch=self.epoch, batch=batch_index, n=len(sel)
                )
            b = batch_for(
                source=self.source, samples=samples, index=sel, lead=lead_all[start : start + aw.batch_size],
                cut=self.cut, dropout=drop, pv_masks=masks, fold_hist=fold_hist,
            )  # fmt: skip
            w = tr.token_weights(
                target_idx=b.target_idx, world=aw.world, stage=self.stage
            ) * torch.isfinite(b.target)
            out = model(b.inputs).out
            nll = token_loss(
                out=out,
                target=b.target,
                primary=b.inputs.primary,
                aw=aw,
                folded_arm=self.folded_arm,
            )
            loss = (w * nll).sum() / w.sum().clamp_min(
                1e-9
            ) + core.SIGN_CE_WEIGHT * sign_cross_entropy(model=model, batch=b)
            self.step += 1
            for g in self.optimiser.param_groups:
                g["lr"] = g["base_lr"] * (
                    min(1.0, self.step / tr.WARMUP_STEPS) if g["warmup"] else 1.0
                )
            self.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trunk_params, tr.GRAD_CLIP)
            if self.new_params:
                torch.nn.utils.clip_grad_norm_(self.new_params, tr.GRAD_CLIP)
            self.optimiser.step()
        self.epoch += 1

    def validate(self) -> float:
        """Return the unweighted loss at the hidden pair on the held-out pairs.

        The loss is against the folded reading, with no mask.
        """
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for lead_day in tr.VALIDATION_LEAD_DAYS:
                for start in range(0, len(self.val_idx), self.aw.batch_size * 2):
                    sel = self.val_idx[start : start + self.aw.batch_size * 2]
                    b = batch_for(
                        source=self.source, samples=self.samples, index=sel,
                        lead=np.full(len(sel), lead_day, dtype=np.int64), cut=self.cut,
                    )  # fmt: skip
                    nll = token_loss(
                        out=self.model(b.inputs).out, target=b.target, primary=b.inputs.primary, aw=self.aw,
                        folded_arm=self.folded_arm,
                    )  # fmt: skip
                    ok = torch.isfinite(b.target)
                    total += float(nll[ok].sum())
                    count += int(ok.sum())
        return total / max(count, 1)

    def get_state(self) -> dict:
        """Return a copy of the model's state."""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_state(self, state: dict) -> None:
        """Load a state into the model."""
        self.model.load_state_dict(state)


# --------------------------------------------------------------------------- prediction and sign read-outs


def predict_rows(
    *, model: torch.nn.Module, aw: ArmWorld, c_idx: np.ndarray, primaries: tuple[int, ...], stamp_us: np.ndarray,
    batch: int = 128,
) -> pl.DataFrame:  # fmt: skip
    """Return the scored rows of `model` at the contract origins `c_idx` for `primaries`.

    One row per (Primary, origin, target half-hour) with a finite signed truth or reading and a lead
    of 24 to
    239 hours. The columns are the keys of `e00721_score.KEYS`, `actual` (signed truth, or the reading where
    the sign is unknown), `reading`, `q01` to `q99` (the signed mixture) and `fq01` to `fq99` (the folded
    reading's quantiles), all in capacity fractions.
    """
    import e00721_score as score

    model.eval()
    source = aw.sources[aw.predict_stage]
    dev = source.device
    prim_all = np.repeat(np.array(primaries), len(c_idx))
    c_all = np.tile(np.asarray(c_idx), len(primaries))
    mean = torch.as_tensor(aw.world.target_mean, dtype=torch.float32, device=dev)
    std = torch.as_tensor(aw.world.target_std, dtype=torch.float32, device=dev)
    frames = []
    with torch.no_grad():
        for k in range(lay.N_LEAD_DAYS):
            for start in range(0, len(prim_all), batch):
                p = to_device(a=prim_all[start : start + batch], dev=dev)
                c = to_device(a=c_all[start : start + batch], dev=dev)
                lead = torch.full_like(p, k)
                b = source.assemble(primary=p, c_idx=c, lead_day=lead, fold_hist=None)
                out = model(b.inputs).out
                q = (
                    mdl.standardised_quantiles(out=out) * std[p][:, None, None]
                    + mean[p][:, None, None]
                )
                shift = torch.as_tensor(aw.shift, dtype=torch.float32, device=dev)[p]
                qv = torch.as_tensor(aw.q_v, dtype=torch.float32, device=dev)[p]
                fq = (
                    arms.folded_quantiles(out=out, shift=shift, q_v=qv) * std[p][:, None, None]
                    + mean[p][:, None, None]
                )
                if aw.spec.loss != "folded":
                    fq = q
                tidx = b.target_idx.cpu().numpy()
                pn = p.cpu().numpy()[:, None].repeat(tidx.shape[1], axis=1)
                signed = aw.signed_truth[pn, tidx]
                reading = aw.reading[pn, tidx]
                actual = np.where(np.isfinite(signed), signed, reading)
                native = (c.cpu().numpy() - lay.NATIVE_OFFSET_STEPS)[:, None].repeat(
                    tidx.shape[1], axis=1
                )
                lead_h = (tidx - native) * 0.5
                keep = (
                    np.isfinite(actual)
                    & (lead_h >= core.LEAD_MIN_HOURS)
                    & (lead_h < core.LEAD_MAX_HOURS_EXCLUSIVE)
                )
                cols = {
                    "time_series_id": pn[keep].astype(np.int64),
                    "power_fcst_init_time": stamp_us[native][keep],
                    "valid_time": stamp_us[tidx][keep],
                    "fold": np.full(int(keep.sum()), aw.fold),
                    "seed": np.full(int(keep.sum()), aw.seed, dtype=np.int64),
                    "actual": actual[keep].astype(np.float64),
                    "reading": reading[keep].astype(np.float64),
                    "signed_known": np.isfinite(signed)[keep],
                }
                qa, fa = q.cpu().numpy(), fq.cpu().numpy()
                for i, name in enumerate(score.QCOLS):
                    cols[name] = qa[..., i][keep].astype(np.float64)
                    cols[score.FOLDED_QCOLS[i]] = fa[..., i][keep].astype(np.float64)
                frames.append(pl.DataFrame(cols))
    df = pl.concat(frames)
    return df.with_columns(
        pl.col("valid_time").cast(pl.Datetime("us")),
        pl.col("power_fcst_init_time").cast(pl.Datetime("us")),
    )


def sign_tokens_of(
    *, model: torch.nn.Module, aw: ArmWorld, samples: tr.SampleSet, primaries: tuple[int, ...], max_pairs: int = 400,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return `rho`, the true sign, the signed power in points, `u`, and a token id of each token.

    The tokens are the distinct history tokens of `primaries`.

    The tokens come from the held-out pairs, with the history folded. The truth is read here and never in
    training.
    """
    model.eval()
    source = aw.sources[aw.predict_stage]
    idx = np.nonzero(samples.is_val & np.isin(samples.primary, np.array(primaries)))[0][:max_pairs]
    rho, x, u, tid, prim = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(idx), aw.batch_size):
            sel = idx[start : start + aw.batch_size]
            b = batch_for(
                source=source, samples=samples, index=sel, lead=np.zeros(len(sel), dtype=np.int64), cut=10**9,
                fold_hist=np.ones(len(sel), dtype=bool),
            )  # fmt: skip
            model(b.inputs)
            r = getattr(model, "last_rho", None)
            if r is None:
                continue
            gi = b.hist_idx.cpu().numpy()
            pr = (
                b.inputs.primary.cpu()
                .numpy()[:, None, None]
                .repeat(gi.shape[1], 1)
                .repeat(gi.shape[2], 2)
            )
            vis = b.inputs.hist_visible.cpu().numpy() & (
                b.inputs.hist_u.cpu().numpy() >= core.U_MIN
            )
            rho.append(r.cpu().numpy()[vis])
            x.append(aw.signed_truth[pr, gi][vis] * 100.0)
            u.append(b.inputs.hist_u.cpu().numpy()[vis])
            tid.append((pr.astype(np.int64) * 10**7 + gi)[vis])
            prim.append(pr[vis])
    if not rho:
        return {k: np.zeros(0) for k in ("rho", "x_points", "u", "token_id", "primary")}
    return {
        "rho": np.concatenate(rho), "x_points": np.concatenate(x), "u": np.concatenate(u),
        "token_id": np.concatenate(tid), "primary": np.concatenate(prim),
    }  # fmt: skip


def timing_record(*, started: float) -> float:
    """Return the seconds since `started`."""
    return time.monotonic() - started

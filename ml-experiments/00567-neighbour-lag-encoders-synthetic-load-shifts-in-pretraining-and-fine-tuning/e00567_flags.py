# ruff: noqa: E501, E402, ANN001, ANN201, PLR0915, D103
r"""Score the auxiliary flag head of the N1-aux arm on held-out synthetic events and on real labelled events.

For each fold and fit seed the script loads the saved coarse fine-tuned model, computes the flag logit
of each of the 10 lag slots on the test rows (lags as observed for real events, and under the held-out
exact and imperfect-conservation worlds for synthetic events), and writes one row per (fold, seed,
variant, slot kind) with the area under the ROC curve, and the recall at a false-alarm rate of 5%.

A real lag is positive when its source half-hour lies between an event's onset and end at that slot's
Primary (labels are hindsight and are used for scoring only); a slot with a missing lag is excluded.
False alarms on slots with no labelled event are an upper bound, because NGED's labels are
incomplete and 28% of events name no recipient.

```bash
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00567-*/e00567_flags.py --arm N1aux-r7
```
"""

import argparse
import sys

_argv = sys.argv
parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True)
parser.add_argument("--months", nargs="+", default=None)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ARGS = parser.parse_args(_argv[1:])
sys.argv = [_argv[0], "coarse"]

import e00510_common as c  # ty: ignore[unresolved-import]
import e00567_fit as F  # ty: ignore[unresolved-import]
import e00567_model as m  # ty: ignore[unresolved-import]
import e00567_score as sc  # ty: ignore[unresolved-import]
import e00567_world as w  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

sys.argv = _argv
SLOT_KINDS = {
    "own_day": [0],
    "own_week": [1],
    "neighbour_day": [2, 4, 6, 8],
    "neighbour_week": [3, 5, 7, 9],
}


def auc(*, score: np.ndarray, label: np.ndarray) -> float:
    """Return the area under the ROC curve by the rank statistic (NaN when a class is empty)."""
    pos, neg = int((label == 1).sum()), int((label == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def recall_at_false_alarm(
    *, score: np.ndarray, label: np.ndarray, rate: float = 0.05
) -> tuple[float, float]:
    """Return (recall, threshold): the share of positives above the threshold that flags `rate` of negatives."""
    neg = score[label == 0]
    pos = score[label == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan")
    threshold = float(np.quantile(neg, 1.0 - rate))
    return float((pos > threshold).mean()), threshold


def real_labels(
    *, ev: sc.Events, slot_primary: np.ndarray, slot_idx: np.ndarray, usable: np.ndarray
) -> np.ndarray:
    """Return 1, 0, or -1 (excluded) per slot: the lag's source time lies inside a labelled event at that Primary."""
    out = np.full(slot_primary.shape, -1, dtype=np.int8)
    for j in range(slot_primary.shape[1]):
        inside, _ = ev.mask(
            primary=np.where(slot_primary[:, j] >= 0, slot_primary[:, j], 0),
            idx=slot_idx[:, j],
            pad_before=0,
            pad_after=0,
        )
        out[:, j] = np.where(usable[:, j], inside.astype(np.int8), -1)
    return out


def main() -> None:
    arm = F.make_arm(tag=ARGS.arm)
    assert arm.aux, "the flag head exists only in the aux arm"
    months = ARGS.months or list(F.MONTHS)
    ev = sc.Events()
    t = F.topo()
    rows_out: list[dict] = []
    for month in months:
        cell = F.Cell(month=month, device="cuda")
        nbr = t.index[arm.graph]
        test = cell.test
        kw = {
            "primary": test.primary,
            "target_idx": test.target_idx,
            "origin_idx": test.origin_idx,
            "lead": test.lead,
        }
        first = F.cut_index(month=month) - 14 * 48
        for seed in ARGS.seeds:
            held = {}
            for kind, name in (("exact", "hx"), ("imperfect", "hi")):
                held[name] = F.heldout_world(
                    kind=kind,
                    month=month,
                    seed=seed,
                    clean=cell.clean,
                    first=first,
                    last=int(test.target_idx.max()) + 1,
                    tag=f"fold{month}",
                )
            mfile = F.model_path(tag=arm.tag, month=month, key="coarse", seed=seed)
            model = F.new_net(arm=arm, n_primaries=cell.n_primaries, stage="ft")
            model.load_state_dict(torch.load(mfile, map_location="cpu"))
            model = model.to("cuda").eval()
            model.table = F.b.CTX["table_ft"]
            vs = cell.variants(arm=arm, held=held)
            for name in ("base", "hx", "hi"):
                x = vs[name]
                logits = np.empty((len(x), m.SLOTS), dtype=np.float32)
                with torch.no_grad():
                    for a in range(0, len(x), 65_536):
                        xb = torch.from_numpy(x[a : a + 65_536]).to("cuda")
                        pb = torch.from_numpy(test.primary[a : a + 65_536]).to("cuda")
                        logits[a : a + 65_536] = model.forward_aux(xb, pb)[1].float().cpu().numpy()
                day = x[:, 43].astype(np.int64)
                week = x[:, 44].astype(np.int64)
                sp, si = F.slots_of(primary=test.primary, day_idx=day, week_idx=week, nbr=nbr)
                if name == "base":
                    usable = np.stack(
                        [
                            (x[:, 38] < 0.5),
                            (x[:, 41] < 0.5),
                            *[
                                (x[:, 45 + 6 * (j // 2) + (1 if j % 2 == 0 else 4)] < 0.5)
                                & (sp[:, 2 + j] >= 0)
                                for j in range(2 * m.K)
                            ],
                        ],
                        axis=1,
                    )
                    label = real_labels(ev=ev, slot_primary=sp, slot_idx=si, usable=usable)
                else:
                    power, code = held[name]
                    label, _ = w.aux_labels(
                        world=power, clean=cell.clean, code=code, slot_primary=sp, slot_idx=si
                    )
                    # block-dropout is not applied at test time, so only missing lags are excluded (by aux_labels)
                for kind, slots in SLOT_KINDS.items():
                    s_ = logits[:, slots].ravel()
                    l_ = label[:, slots].ravel()
                    keep = l_ >= 0
                    rec, thr = recall_at_false_alarm(score=s_[keep], label=l_[keep])
                    rows_out.append({"month": month, "seed": seed, "variant": name, "slot": kind, "positives": int((l_ == 1).sum()), "negatives": int((l_ == 0).sum()),
                                     "auc": auc(score=s_[keep], label=l_[keep]), "recall_at_5pct_false_alarm": rec, "threshold": thr})  # fmt: skip
                print(month, seed, name, flush=True)
            del model
        del cell
        torch.cuda.empty_cache()
    out = F.WORK / "scores" / f"flags_{arm.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows_out).write_csv(out)
    print("wrote", out, c.MONTHS)


if __name__ == "__main__":
    main()

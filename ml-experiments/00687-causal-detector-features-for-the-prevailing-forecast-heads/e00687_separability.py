# ruff: noqa: E501, D103, ANN001
"""Experiment 687, step 1: does adding D1 to D5 separate 'this lag lies inside an injected shift'?

Logistic regression on each input and its absolute value (shift signs are random), standardised
inputs, folds that hold out the scored month. Three scorers: `base` (experiment 606's origin
summaries), `plus` (summaries plus D1 to D5 at the day and week cutoffs), and `placebo` (summaries
plus D1 to D5 of an unrelated Primary of the same metering group). Rows are the bank's origins.
AUROC is reported per bank version, per lag label, and per family (training passes below 200,
held-out passes from 200), with a month-block bootstrap (5 blocks) and an event-block bootstrap
(Primary x pass).

Usage: python e00687_separability.py
"""

import json
import warnings

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00687_features as f
import e00687_tables as t
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BANKS = ("M", "S", "X")
D_COLS = [i for i, n in enumerate(f.FEATURE_NAMES) if n != "cu_ret"]
OUT = t.WORK / "separability.json"


def placebo_map() -> np.ndarray:
    """Return for each Primary an unrelated Primary of the same metering group (not itself, not a neighbour)."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    nb_of = t.neighbour_map()
    grp = k578.sites()["group"].to_list()
    out = np.zeros(len(grp), int)
    for p, g in enumerate(grp):
        cand = [q for q in range(len(grp)) if q != p and q not in nb_of[p] and str(grp[q]).startswith("signed-MW") == str(g).startswith("signed-MW")]  # fmt: skip
        out[p] = cand[(p * 7) % len(cand)]
    return out


def load_month(*, month: str, bank: str, pmap: np.ndarray) -> dict:
    b = np.load(c650.WORK / "head_data" / f"bank_{month}.npz")
    z = np.load(t.WORK / "feats" / f"{month}_{bank}.npz")
    key = {
        (int(p), int(o), int(r)): i
        for i, (p, o, r) in enumerate(zip(z["prim"], z["oi"], z["pass_"], strict=True))
    }
    org = np.array(
        [
            key[(int(p), int(o), int(r))]
            for p, o, r in zip(b["org_prim"], b["org_oi"], b["org_pass"], strict=True)
        ]
    )
    plc = np.array(
        [
            key[(int(pmap[p]), int(o), int(r))]
            for p, o, r in zip(b["org_prim"], b["org_oi"], b["org_pass"], strict=True)
        ]
    )
    base = b["feat"].astype(np.float64)
    own = z["feat"][org][:, :2, :][:, :, D_COLS].reshape(len(org), -1)
    oth = z["feat"][plc][:, :2, :][:, :, D_COLS].reshape(len(org), -1)
    return {
        "base": base, "plus": np.concatenate([base, own], axis=1), "placebo": np.concatenate([base, oth], axis=1),
        "y_day": z["in_day"][org], "y_week": z["in_week"][org], "pass_": b["org_pass"], "prim": b["org_prim"],
        "usable": b["gate"] == b["gate"],
    }  # fmt: skip


def fit_score(*, x_tr, y_tr, x_te) -> np.ndarray:
    # shifts have random signs, so every scorer also gets the absolute value of each input
    x_tr, x_te = (np.concatenate([a, np.abs(a)], axis=1) for a in (x_tr, x_te))
    sc = StandardScaler().fit(x_tr)
    m = LogisticRegression(max_iter=500, C=1.0).fit(np.clip(sc.transform(x_tr), -10, 10), y_tr)
    return m.predict_proba(np.clip(sc.transform(x_te), -10, 10))[:, 1]


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main() -> None:
    warnings.filterwarnings("ignore")
    pmap = placebo_map()
    res: dict = {}
    for bank in BANKS:
        data = {m: load_month(month=m, bank=bank, pmap=pmap) for m in c650.MONTHS}
        for label in ("y_day", "y_week"):
            for scorer in ("base", "plus", "placebo"):
                per_month = {}
                scores = {}
                for m in c650.MONTHS:
                    tr = [k for k in c650.MONTHS if k != m]
                    fam_tr = [
                        data[k]["pass_"] < 200 for k in tr
                    ]  # train on the training family only
                    x_tr = np.concatenate(
                        [data[k][scorer][mk] for k, mk in zip(tr, fam_tr, strict=True)]
                    )
                    y_tr = np.concatenate(
                        [data[k][label][mk] for k, mk in zip(tr, fam_tr, strict=True)]
                    )
                    s = fit_score(x_tr=x_tr, y_tr=y_tr, x_te=data[m][scorer])
                    scores[m] = s
                    for fam, mask in (
                        ("train_family", data[m]["pass_"] < 200),
                        ("held_out", data[m]["pass_"] >= 200),
                    ):
                        per_month[(m, fam)] = auroc(data[m][label][mask], s[mask])
                for fam in ("train_family", "held_out"):
                    res[f"{bank}|{label}|{scorer}|{fam}"] = {
                        "per_month": {m: per_month[(m, fam)] for m in c650.MONTHS},
                        "mean": float(np.nanmean([per_month[(m, fam)] for m in c650.MONTHS])),
                    }
                # pooled event-block bootstrap on the difference is done in the report from stored scores
                np.savez(t.WORK / "sep_scores" / f"{bank}_{label}_{scorer}.npz", **{m: scores[m] for m in c650.MONTHS}) if (t.WORK / "sep_scores").exists() else None  # fmt: skip  # ty: ignore[invalid-argument-type]
        print(bank, "done", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    for bank in BANKS:
        for label in ("y_day", "y_week"):
            for fam in ("train_family", "held_out"):
                b0 = res[f"{bank}|{label}|base|{fam}"]["mean"]
                p1 = res[f"{bank}|{label}|plus|{fam}"]["mean"]
                pl = res[f"{bank}|{label}|placebo|{fam}"]["mean"]
                print(
                    f"{bank} {label} {fam}: base {b0:.3f} plus {p1:.3f} (+{p1 - b0:.3f}) placebo {pl:.3f}"
                )


if __name__ == "__main__":
    (t.WORK / "sep_scores").mkdir(parents=True, exist_ok=True)
    main()

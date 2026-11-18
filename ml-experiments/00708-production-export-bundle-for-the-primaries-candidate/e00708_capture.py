# ruff: noqa: ANN001, ANN002, ANN003, ANN201, D103, E402, E501, PT018
"""Experiment 708: capture the Candidate's prediction-pass state by running experiment 667's code.

The driver reruns the prediction phase of `e00667_cand.py P7+D50 fit` (which first rebuilds K1's
lags-masked table T) for one refit month in a scratch root, with the arm patch of experiment 704
installed first (`e00704_arm`, `drop`). The stored fine-tuned Candidate weights, and the file that
fingerprints the standardiser, are linked into the scratch root, so nothing is trained and nothing
is written under another experiment's data folder (the chain's own reads of experiment 667's cached
inputs are unchanged). Hooks on module attributes record what the pass holds only in process
memory:

- the 37-column raw matrix before `Standardiser.apply` and the raw B1 columns, for the month's test
  rows (in test order);
- the final test rows of the month and of the month before (the rows whose K1 forecasts fill table
  T), and K1's lags-masked q05, q50, and q95 for both, per seed;
- the Candidate's input matrix (60 columns plus the signal columns), raw network outputs, and sorted
  quantiles of every seed for the rows of the fixture origins, and the whole window's quantiles for a
  comparison with the stored forecasts;
- the fold's standardiser, target mean and standard deviation, lag statistics, B1 statistics, the
  neighbour list and capacities, the cleaned power, and the rows of the expected-weather table that
  the fixtures read.

Usage (through `run_export.sh`): `python e00708_capture.py --month 2026-10 --origins 2026-10-01 ...
--chain-work CHAIN --source-root P7_D50_2026-10 --scratch SCRATCH --out DUMP.npz`.
"""

import argparse
import json
import os
import re
import runpy
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--month", required=True)
PARSER.add_argument("--origins", nargs="+", required=True, help="UTC dates of the fixture origins")
PARSER.add_argument("--chain-work", required=True, help="CHAIN667: pre-training and step-2 weights")
PARSER.add_argument("--source-root", required=True, help="folder P7_D50_<month> of the stored fit")
PARSER.add_argument("--scratch", required=True, help="scratch root for this capture")
PARSER.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
PARSER.add_argument("--out", required=True)
ARGS = PARSER.parse_args()

os.environ.update(
    E00704_ARM="drop",
    E00667_MONTH=ARGS.month,
    E00667_PHASE="predict",
    E00667_ROOT=ARGS.scratch,
    CHAIN667=ARGS.chain_work,
)
SCRATCH = Path(ARGS.scratch)
WORK = SCRATCH / f"P7_D50_{ARGS.month}"
MODELS = WORK / "work" / "models" / "P5"
MODELS.mkdir(parents=True, exist_ok=True)
shutil.rmtree(
    WORK / "work" / "fits", ignore_errors=True
)  # a stored forecast would make the pass skip the model
shutil.rmtree(SCRATCH / "tables", ignore_errors=True)  # the table is rebuilt from K1's weights
SOURCE = Path(ARGS.source_root)
for _seed in ARGS.seeds:
    _name = f"{ARGS.month}_d0_s{_seed}.pt"
    _link = MODELS / _name
    if _link.is_symlink() or _link.exists():
        _link.unlink()
    assert (SOURCE / "work" / "models" / "P5" / _name).is_file(), _name
    _link.symlink_to(SOURCE / "work" / "models" / "P5" / _name)
_fingerprint = f"standardiser_{ARGS.month}.sha256"
shutil.copyfile(SOURCE / _fingerprint, WORK / _fingerprint)

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00562_columns as c562  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00704_arm  # noqa: F401  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

ORIGINS_US = np.array(
    [
        int(datetime.fromisoformat(d).replace(hour=9, tzinfo=UTC).timestamp()) * 1_000_000
        for d in ARGS.origins
    ],
    dtype=np.int64,
)
MONTH_START_US = (
    int(datetime.fromisoformat(ARGS.month + "-01").replace(tzinfo=UTC).timestamp()) * 1_000_000
)
GRID_START_US = int(datetime(2019, 9, 1, tzinfo=UTC).timestamp()) * 1_000_000
STEP_US = 1_800_000_000
HISTORY_ORIGINS = (2, 3, 4)
"""K1 forecasts of the origins this many days before a fixture origin fill table T up to the newest entry the Candidate reads."""
STATE: dict = {
    "in_test": False,
    "raw37": [],
    "b1": None,
    "calls": [],
    "last_path": None,
    "tests": {},
    "k1": {},
}

_orig_build_test = data.build_test


def build_test(*, month, standardiser):
    STATE["in_test"] = True
    STATE["raw37"] = []
    try:
        out = _orig_build_test(month=month, standardiser=standardiser)
    finally:
        STATE["in_test"] = False
    STATE.setdefault("raw37_of", {})[month] = np.concatenate(STATE["raw37"])
    return out


data.build_test = build_test  # before experiment 574 saves it as `_build_test`

_orig_apply = prep.Standardiser.apply


def apply(self, *, x):
    if STATE["in_test"]:
        STATE["raw37"].append(np.array(x, dtype=np.float32))
        STATE["standardiser"] = self
    return _orig_apply(self, x=x)


prep.Standardiser.apply = apply

_orig_raw_real = c562.FoldColumns._raw_real


def raw_real(self, *, arm, primary, origin_idx, valid_idx):
    raw = _orig_raw_real(self, arm=arm, primary=primary, origin_idx=origin_idx, valid_idx=valid_idx)
    if (
        arm == "B1"
        and len(origin_idx)
        and int(np.min(origin_idx)) * STEP_US + GRID_START_US >= MONTH_START_US
    ):
        STATE["b1"] = {
            "raw": np.array(raw, dtype=np.float32),
            "origin_idx": np.array(origin_idx),
            "valid_idx": np.array(valid_idx),
            "primary": np.array(primary),
        }
    return raw


c562.FoldColumns._raw_real = raw_real

_orig_load = torch.load


def load(f, *args, **kwargs):
    STATE["last_path"] = str(f)
    return _orig_load(f, *args, **kwargs)


torch.load = load  # ty: ignore[invalid-assignment]

_orig_masked = s598.masked_quantiles


def masked_quantiles(*, model, x, primary, target_mean, target_std, device, batch=262_144):
    q = _orig_masked(
        model=model,
        x=x,
        primary=primary,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        batch=batch,
    )
    known = {id(t.x): month for month, t in STATE["tests"].items()}
    if id(x) in known:
        found = re.search(r"_s(\d+)\.pt$", STATE["last_path"])
        assert found is not None, STATE["last_path"]
        STATE["k1"][(known[id(x)], int(found.group(1)))] = q
    return q


s598.masked_quantiles = masked_quantiles

# the fold modules wrap experiment 574's `build_test`, so import it now with the same arguments as
# experiment 667's candidate script, and record the final test rows of every month
sys.argv = [
    "e00574_fit.py", "fit", "--arm", "P5", "--months", ARGS.month, "--seeds", *map(str, ARGS.seeds),
]  # fmt: skip
import e00574_fit as fit  # ty: ignore[unresolved-import]

_final_build_test = fit.build_test


def final_build_test(*, month, standardiser):
    test = _final_build_test(month=month, standardiser=standardiser)
    STATE["tests"][month] = test
    return test


fit.build_test = final_build_test

_orig_forecast = c510.forecast_frame


def forecast_frame(*, model, outlier, test, select, rows, widen, device):
    frame = _orig_forecast(
        model=model,
        outlier=outlier,
        test=test,
        select=select,
        rows=rows,
        widen=widen,
        device=device,
    )
    if widen != 1.0:
        return frame
    found = re.search(r"_s(\d+)\.pt$", STATE["last_path"])
    assert found is not None, STATE["last_path"]
    seed = int(found.group(1))
    sel_idx = np.nonzero(select)[0]
    fixture = np.nonzero(np.isin(test.origin[sel_idx], ORIGINS_US))[0]  # positions in the frame
    rows_idx = sel_idx[fixture]
    x = torch.from_numpy(test.x[rows_idx]).to(device)
    p = torch.from_numpy(test.primary[rows_idx]).to(device)
    with torch.no_grad():
        out = model(x, p).float().cpu().numpy()
    names = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
    q = frame.select(names).to_numpy()
    STATE["calls"].append(
        {
            "seed": seed,
            "primary": np.array(test.primary[rows_idx]),
            "origin": np.array(test.origin[rows_idx]),
            "target_idx": np.array(test.target_idx[rows_idx]),
            "x": np.array(test.x[rows_idx]),
            "raw_out": out,
            "q_fix": q[fixture],
            "q_all": q,
            "key_series": frame["time_series_id"].to_numpy(),
            "key_origin": frame["power_fcst_init_time"].dt.epoch("us").to_numpy(),
            "key_valid": frame["valid_time"].dt.epoch("us").to_numpy(),
        }
    )
    STATE["rows"] = rows
    return frame


c510.forecast_frame = forecast_frame

cand = next(Path(__file__).resolve().parents[1].glob("00667-*")) / "e00667_cand.py"
sys.argv = [
    str(cand), "P7+D50", "fit", "--work-suffix", f"_{ARGS.month}", "--months", ARGS.month,
    "--seeds", *map(str, ARGS.seeds),
]  # fmt: skip
cand_globals = runpy.run_path(str(cand), run_name="__main__")

# --- after the run: collect the fold's in-memory state -----------------------------------------
test = STATE["tests"][ARGS.month]
ref = STATE["tests"][cand_globals["fold"].PREVIOUS_MONTH]
std = STATE["standardiser"]
raw37 = STATE["raw37_of"][ARGS.month]
assert len(raw37) == len(test.x), (len(raw37), len(test.x))
assert np.array_equal(std.apply(x=raw37), test.x[:, :37]), (
    "the raw matrix does not standardise to the test matrix"
)
b1 = STATE["b1"]
assert b1 is not None and len(b1["raw"]) == len(test.x), "B1 capture missing"
assert np.array_equal(b1["primary"], test.primary) and np.array_equal(
    b1["valid_idx"], test.target_idx
)
assert len(STATE["calls"]) == len(ARGS.seeds), len(STATE["calls"])
rows = STATE["rows"]
ctx = fit.CTX


def key_of(*, primary, origin_idx, target_idx):
    return (
        (primary.astype(np.int64) << 48)
        | (origin_idx.astype(np.int64) << 24)
        | target_idx.astype(np.int64)
    )


test_key = key_of(primary=test.primary, origin_idx=test.origin_idx, target_idx=test.target_idx)
assert len(np.unique(test_key)) == len(test_key)
order = np.argsort(test_key)
first = STATE["calls"][0]
fixture_key = key_of(
    primary=first["primary"],
    origin_idx=(first["origin"] - GRID_START_US) // STEP_US,
    target_idx=first["target_idx"],
)
pos = order[np.searchsorted(test_key[order], fixture_key)]
assert np.array_equal(test_key[pos], fixture_key), "fixture rows not found in the month's test rows"
for call in STATE["calls"]:
    assert np.array_equal(call["origin"], first["origin"]) and np.array_equal(
        call["target_idx"], first["target_idx"]
    ), "fixture rows differ between seeds"
    assert np.array_equal(call["x"][:, :60], test.x[pos][:, :60]), (
        "base columns differ from the test rows"
    )

nb = cand_globals["NB"]
columns = cand_globals["COLUMNS"]
sig = cand_globals["sig"]
signal_names = [sig.ALL_NAMES[c] for c in columns]
n_extra = len(columns)
assert first["x"].shape[1] == 60 + n_extra, first["x"].shape

# K1 history rows: the rows of the three origins before each fixture origin, from the test rows and the previous month's
hist_parts = []
for o_us in ORIGINS_US:
    o_idx = int((o_us - GRID_START_US) // STEP_US)
    for k in HISTORY_ORIGINS:
        for month, t in ((ARGS.month, test), (cand_globals["fold"].PREVIOUS_MONTH, ref)):
            lead = t.target_idx - t.origin_idx
            sel = np.nonzero((t.origin_idx == o_idx - 48 * k) & (lead >= 1) & (lead <= 48))[0]
            if len(sel):
                hist_parts.append((o_idx, k, month, t, sel))
hist = {
    k: []
    for k in (
        "fixture_origin_idx",
        "k",
        "source",
        "x",
        "primary",
        "origin_idx",
        "target_idx",
        "origin",
        "valid",
    )
}
hist_q = {s: [] for s in ARGS.seeds}
for o_idx, k, month, t, sel in hist_parts:
    hist["fixture_origin_idx"].append(np.full(len(sel), o_idx))
    hist["k"].append(np.full(len(sel), k))
    hist["source"].append(np.full(len(sel), int(month == ARGS.month)))
    hist["x"].append(np.array(t.x[sel]))
    hist["primary"].append(np.array(t.primary[sel]))
    hist["origin_idx"].append(np.array(t.origin_idx[sel]))
    hist["target_idx"].append(np.array(t.target_idx[sel]))
    hist["origin"].append(np.array(t.origin[sel]))
    hist["valid"].append(np.array(t.valid[sel]))
    for s in ARGS.seeds:
        hist_q[s].append(STATE["k1"][(month, s)][sel])
hist = {k: np.concatenate(v) for k, v in hist.items()}
for s in ARGS.seeds:
    hist[f"k1_q_s{s}"] = np.concatenate(hist_q[s])
# every (fixture origin, history origin) pair has all 16 Primaries and 48 leads
assert len(hist["x"]) == len(ORIGINS_US) * len(HISTORY_ORIGINS) * 16 * 48, len(hist["x"])

# expected-weather rows read by the fixtures: the lag windows of each fixture origin and every lag index of the history rows
need_parts = []
for o_us in ORIGINS_US:
    o_idx = int((o_us - GRID_START_US) // STEP_US)
    window = np.arange(o_idx - 48 - 335, o_idx - 48 + 1)
    need_parts.append(
        np.stack([np.repeat(np.arange(16), len(window)), np.tile(window, 16)], axis=1)
    )
for xs, ps in ((hist["x"], hist["primary"]), (first["x"], first["primary"])):
    for col in (43, 44):
        idx = xs[:, col].astype(np.int64)
        need_parts.append(np.stack([ps[idx >= 0], idx[idx >= 0]], axis=1))
need = np.unique(np.concatenate(need_parts).astype(np.int64), axis=0)
table_rows = (
    ctx["table_ft"][torch.from_numpy(need[:, 0]).cuda(), torch.from_numpy(need[:, 1]).cuda()]
    .cpu()
    .numpy()
)
out: dict[str, np.ndarray] = {
    "fixture_pos": pos,
    "x_base": test.x[pos][:, :60],
    "raw37": raw37[pos],
    "b1_raw": b1["raw"][pos],
    "primary": test.primary[pos],
    "origin": test.origin[pos],
    "valid": test.valid[pos],
    "lead": test.lead[pos],
    "target_idx": test.target_idx[pos],
    "origin_idx": test.origin_idx[pos],
    "std_mean": std.mean,
    "std_std": std.std,
    "std_mask": std.mask,
    "target_mean": rows.target_mean,
    "target_std": rows.target_std,
    "lag_stats": np.asarray(ctx["stats"]),
    "b1_mean": np.asarray(fit.EX.mean),
    "b1_std": np.asarray(fit.EX.std),
    "knots": np.array(fit.KNOTS),
    "table_need": need,
    "table_std_f16": table_rows,
    "table_origin": np.asarray(ctx["origin_ft"])[need[:, 0], need[:, 1]],
    "clean": np.asarray(ctx["clean"], dtype=np.float32),
    "n_test": np.array(len(test.x)),
    "nb_capacity": np.asarray(nb.capacity, dtype=np.float64),
    "nb_of_flat": np.array([[p, q] for p, qs in nb.of.items() for q in qs], dtype=np.int64),
    "signal_columns": np.array(columns),
    "signal_names": np.array(signal_names),
    **{f"hist_{k}": v for k, v in hist.items()},
}
for call in STATE["calls"]:
    tag = f"s{call['seed']}"
    out[f"raw_out_{tag}"] = call["raw_out"]
    out[f"q_fix_{tag}"] = call["q_fix"]
    out[f"x_{tag}"] = call["x"]
    out[f"q_all_{tag}"] = call["q_all"]
    out[f"key_series_{tag}"] = call["key_series"]
    out[f"key_origin_{tag}"] = call["key_origin"]
    out[f"key_valid_{tag}"] = call["key_valid"]
Path(ARGS.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(ARGS.out, **out)
meta = {"seeds": ARGS.seeds, "month": ARGS.month, "n_extra": n_extra, "signal_names": signal_names}
Path(ARGS.out).with_suffix(".json").write_text(json.dumps(meta))
print("E00708 capture written", ARGS.out, flush=True)

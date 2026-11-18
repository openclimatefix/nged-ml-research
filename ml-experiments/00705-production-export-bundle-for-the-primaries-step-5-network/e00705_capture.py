# ruff: noqa: ANN001, ANN002, ANN003, ANN201, D103, E402, E501, PT018
"""Experiment 705: capture the step-5 network's prediction-pass state by running experiment 677's code.

The driver reruns experiment 677's prediction phase (`e00677_k1.py k1_5w fit --step 5` with
`E00677_PHASE=predict`) for one refit month in a scratch root, with the arm patch of experiment 704
installed first (`e00704_arm`, selected by `--arm`). The stored fine-tuned weights are linked into the
scratch root, so nothing is trained and nothing is written under any other experiment's data
folder. Hooks on module attributes record what the pass holds only in process memory:

- the 37-column raw matrix before `Standardiser.apply` (test rows, in test order),
- the raw B1 columns that `FoldColumns._raw_real` returns for the month's test rows,
- the final input matrix, the nine raw network outputs and the sorted quantiles of every seed, for
  the rows of the fixture origins (plus the whole window's quantiles of the main call, for a
  comparison with the stored forecasts),
- the fold's standardiser, target mean and standard deviation, lag statistics, B1 mean and standard
  deviation, knot count, and the rows of the expected-weather table that the fixture rows read.

Usage (from the worktree root, through `run_capture.sh`):
`python e00705_capture.py --arm replay --month 2026-04 --origins 2026-04-01 ... --out DUMP.npz`.
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
PARSER.add_argument("--arm", required=True, choices=("replay", "proxy", "drop"))
PARSER.add_argument("--month", required=True)
PARSER.add_argument("--origins", nargs="+", required=True, help="UTC dates of the fixture origins")
PARSER.add_argument(
    "--source-root", required=True, help="folder holding arms/k1_5w_sp1/work/models"
)
PARSER.add_argument("--chain-work", required=True, help="CHAIN677: pre-training and step-2 weights")
PARSER.add_argument(
    "--stats-work", required=True, help="STATS677: folder holding stats/B1/<month>.npz"
)
PARSER.add_argument("--scratch", required=True, help="scratch root for this capture")
PARSER.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
PARSER.add_argument("--out", required=True)
ARGS = PARSER.parse_args()

os.environ.update(
    E00704_ARM=ARGS.arm,
    E00677_MONTH=ARGS.month,
    E00677_SPAN="1",
    E00677_PHASE="predict",
    E00677_ROOT=ARGS.scratch,
    CHAIN677=ARGS.chain_work,
    STATS677=ARGS.stats_work,
)
SCRATCH = Path(ARGS.scratch)
MODELS = SCRATCH / "arms" / "k1_5w_sp1" / "work" / "models" / "P5"
MODELS.mkdir(parents=True, exist_ok=True)
shutil.rmtree(
    SCRATCH / "arms" / "k1_5w_sp1" / "work" / "fits", ignore_errors=True
)  # a stored forecast here would make the pass skip the model
for _seed in ARGS.seeds:
    _name = f"{ARGS.month}_d0_s{_seed}.pt"
    _source = Path(ARGS.source_root) / "arms" / "k1_5w_sp1" / "work" / "models" / "P5" / _name
    assert _source.is_file(), _source
    _link = MODELS / _name
    if _link.is_symlink() or _link.exists():
        _link.unlink()
    _link.symlink_to(_source)

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00562_columns as c562  # ty: ignore[unresolved-import]
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
STATE: dict = {
    "in_test": False,
    "raw37": [],
    "b1": None,
    "calls": [],
    "last_path": None,
    "test_ref": None,
}

_orig_build_test = data.build_test


def build_test(*, month, standardiser):
    STATE["in_test"] = True
    try:
        return _orig_build_test(month=month, standardiser=standardiser)
    finally:
        STATE["in_test"] = False


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
    fit = sys.modules["e00574_fit"]
    variants = fit.CTX.get("variants", {})
    kind = next((name for name, v in variants.items() if v is test.x), "main")
    if kind not in {"main", "removed"} or widen != 1.0:
        return frame
    found = re.search(r"_s(\d+)\.pt$", STATE["last_path"])
    assert found is not None, STATE["last_path"]
    seed = int(found.group(1))
    if STATE["test_ref"] is None:
        STATE["test_ref"] = test
    sel_idx = np.nonzero(select)[0]
    fixture = np.nonzero(np.isin(test.origin[sel_idx], ORIGINS_US))[0]  # positions in the frame
    rows_idx = sel_idx[fixture]
    x = torch.from_numpy(test.x[rows_idx]).to(device)
    p = torch.from_numpy(test.primary[rows_idx]).to(device)
    with torch.no_grad():
        out = model(x, p).float().cpu().numpy()
    names = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
    q = frame.select(names).to_numpy()
    call = {
        "seed": seed,
        "kind": kind,
        "rows_idx": rows_idx,
        "x": np.array(test.x[rows_idx]),
        "raw_out": out,
        "q_fix": q[fixture],
        "n_select": len(sel_idx),
    }
    if kind == "main":
        call["q_all"] = q
        call["key_series"] = frame["time_series_id"].to_numpy()
        call["key_origin"] = frame["power_fcst_init_time"].dt.epoch("us").to_numpy()
        call["key_valid"] = frame["valid_time"].dt.epoch("us").to_numpy()
    STATE["calls"].append(call)
    STATE["rows"] = rows
    return frame


c510.forecast_frame = forecast_frame

e677 = next(Path(__file__).resolve().parents[1].glob("00677-*")) / "e00677_k1.py"
sys.argv = [
    str(e677),
    "k1_5w",
    "fit",
    "--step",
    "5",
    "--synthetic",
    "1",
    "--months",
    ARGS.month,
    "--seeds",
    *map(str, ARGS.seeds),
]
runpy.run_path(str(e677), run_name="__main__")

# --- after the run: collect the fold's in-memory state -----------------------------------------
fit = sys.modules["e00574_fit"]
test = STATE["test_ref"]
assert test is not None, "no forecast call was captured"
std = STATE["standardiser"]
raw37 = np.concatenate(STATE["raw37"])
assert len(raw37) == len(test.x), (len(raw37), len(test.x))
again = std.apply(x=raw37)
assert np.array_equal(again, test.x[:, :37]), (
    "the captured raw matrix does not standardise to the test matrix"
)
rows = STATE["rows"]
fixture_all = np.nonzero(np.isin(test.origin, ORIGINS_US))[0]
b1 = STATE["b1"]
assert b1 is not None and len(b1["raw"]) == len(test.x), "B1 capture missing"
assert np.array_equal(b1["primary"], test.primary) and np.array_equal(
    b1["valid_idx"], test.target_idx
)

main_calls = [call for call in STATE["calls"] if call["kind"] == "main"]
x_ref = main_calls[0]["x"]
ctx = fit.CTX
lag_idx = np.concatenate([x_ref[:, 43], x_ref[:, 44]]).astype(np.int64)
lag_p = np.concatenate([test.primary[fixture_all]] * 2)
need = np.unique(np.stack([lag_p[lag_idx >= 0], lag_idx[lag_idx >= 0]], axis=1), axis=0)
table_rows = (
    ctx["table_ft"][torch.from_numpy(need[:, 0]).cuda(), torch.from_numpy(need[:, 1]).cuda()]
    .cpu()
    .numpy()
)
out: dict[str, np.ndarray] = {
    "fixture_rows": fixture_all,
    "x": test.x[fixture_all],
    "raw37": raw37[fixture_all],
    "b1_raw": b1["raw"][fixture_all],
    "primary": test.primary[fixture_all],
    "origin": test.origin[fixture_all],
    "valid": test.valid[fixture_all],
    "lead": test.lead[fixture_all],
    "target_idx": test.target_idx[fixture_all],
    "origin_idx": test.origin_idx[fixture_all],
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
}
for call in STATE["calls"]:
    tag = f"s{call['seed']}_{call['kind']}"
    assert np.array_equal(call["rows_idx"], fixture_all), "fixture rows differ between calls"
    out[f"raw_out_{tag}"] = call["raw_out"]
    out[f"q_fix_{tag}"] = call["q_fix"]
    out[f"x_{tag}"] = call["x"]
    if call["kind"] == "main":
        out[f"q_all_{tag}"] = call["q_all"]
        out[f"key_series_{tag}"] = call["key_series"]
        out[f"key_origin_{tag}"] = call["key_origin"]
        out[f"key_valid_{tag}"] = call["key_valid"]
Path(ARGS.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(ARGS.out, **out)
meta = {"seeds": ARGS.seeds, "n_calls": len(STATE["calls"]), "arm": ARGS.arm, "month": ARGS.month}
Path(ARGS.out).with_suffix(".json").write_text(json.dumps(meta))
print("E00705 capture written", ARGS.out, flush=True)

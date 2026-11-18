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

# --- the dense table T: K1's lags-masked forecast at every daily origin of the history window -------
# The research table has whole-day holes (purge windows around held-out weeks), so a production
# store of K1's daily forecasts is denser than the table the Candidate trained on. The history rows
# are every daily origin's first 48 half-hours, built by experiment 574's `build_test` (the code that
# built the fold month's test rows) with the fold's standardiser, for the months the window touches.
HISTORY_DAYS = 180
ORIGIN_IDX = np.array([int((o - GRID_START_US) // STEP_US) for o in ORIGINS_US])
LAST_T_IDX = int(ORIGIN_IDX.max()) - 48  # the newest grid index that any fixture origin reads
ORIGIN_LO = int(ORIGIN_IDX.min()) - 48 - HISTORY_DAYS * 48 - 96
ORIGIN_HI = (
    LAST_T_IDX - 48
)  # the freshest origin that table T entries up to LAST_T_IDX can come from
months = []
cursor = datetime.fromtimestamp((GRID_START_US + ORIGIN_LO * STEP_US) / 1e6, UTC).replace(
    day=1, hour=0
)
stop = datetime.fromtimestamp((GRID_START_US + ORIGIN_HI * STEP_US) / 1e6, UTC)
while cursor <= stop:
    months.append(f"{cursor:%Y-%m}")
    cursor = cursor.replace(year=cursor.year + (cursor.month == 12), month=cursor.month % 12 + 1)
dense: dict[str, list] = {k: [] for k in ("x", "primary", "origin_idx", "target_idx")}
for month in months:
    t = STATE["tests"].get(month)
    owned = t is None
    if owned:
        fit.CHECK["months"].setdefault(month, {})
        t = _final_build_test(month=month, standardiser=std)
    lead = t.target_idx - t.origin_idx
    keep = np.nonzero(
        (lead >= 1) & (lead <= 48) & (t.origin_idx >= ORIGIN_LO) & (t.origin_idx <= ORIGIN_HI)
    )[0]
    for k, v in (
        ("x", t.x),
        ("primary", t.primary),
        ("origin_idx", t.origin_idx),
        ("target_idx", t.target_idx),
    ):
        dense[k].append(np.array(v[keep]))
    STATE.get("raw37_of", {}).pop(month, None)
    print("dense history rows", month, len(keep), flush=True)
    del t
dense_np = {k: np.concatenate(v) for k, v in dense.items()}
n_days = (ORIGIN_HI - ORIGIN_LO) // 48 + 1
assert len(dense_np["x"]) == n_days * 16 * 48, (len(dense_np["x"]), n_days)
assert len(
    np.unique(
        key_of(
            primary=dense_np["primary"],
            origin_idx=dense_np["origin_idx"],
            target_idx=dense_np["target_idx"],
        )
    )
) == len(dense_np["x"])

aux = cand_globals["aux"]
step2_dir = Path(ARGS.chain_work) / "step2" / "P5"
device = "cuda"
tm, ts = rows.target_mean, rows.target_std
dense_q: dict[int, np.ndarray] = {}
dense_table: dict = {}
sparse_table: dict = {}
for s in ARGS.seeds:
    k1 = aux.AuxLagMdn(
        n_primaries=16, stage="ft", n_knots=fit.KNOTS, n_b1=fit.N_B1, kind="shared"
    ).to(device)
    k1.load_state_dict(_orig_load(step2_dir / f"{ARGS.month}_s{s}.pt", map_location=device))
    k1.table = ctx["table_ft"]
    dense_q[s] = _orig_masked(
        model=k1,
        x=dense_np["x"],
        primary=dense_np["primary"],
        target_mean=tm,
        target_std=ts,
        device=device,
    )
    dense_table[s] = s598.build_lag_free_table(
        prim=dense_np["primary"], origin_idx=dense_np["origin_idx"], target_idx=dense_np["target_idx"],
        median=dense_q[s][:, 1], n_primaries=16, n_grid=cand_globals["GRID_SHAPE"][1],
    )  # fmt: skip
    z_sparse = np.load(Path(ARGS.scratch) / "tables" / "predict" / f"T_{ARGS.month}_s{s}.npz")
    sparse_table[s] = s598.LagFreeTable(
        median=z_sparse["median"], origin_idx=z_sparse["origin_idx"]
    )
    del k1
torch.cuda.empty_cache()

# the Candidate's forecasts with the dense table, through the research code path
cand_models = {}
for s in ARGS.seeds:
    model = cand_globals["new_model"](n_primaries=16).to(device)
    model.load_state_dict(
        _orig_load(
            SOURCE / "work" / "models" / "P5" / f"{ARGS.month}_d0_s{s}.pt", map_location=device
        )
    )
    model.eval()
    cand_models[s] = model
origin_ft = np.asarray(ctx["origin_ft"])
NB_READ = cand_globals["NB_READ"]


def extras_of(*, x_rows, prim, origin_idx_rows, power, table, weather_origin) -> np.ndarray:
    """Return the Candidate's signal columns for rows, through experiment 613's `signal_columns`."""
    return sig.signal_columns(
        prim=prim, origin_idx=origin_idx_rows, day_idx=x_rows[:, 43].astype(np.int64),
        week_idx=x_rows[:, 44].astype(np.int64), day_missing=x_rows[:, 38] > 0.5,
        week_missing=x_rows[:, 41] > 0.5, power_lag=power, power_sig=power, table=table,
        nb=NB_READ, weather_origin=weather_origin, columns=columns,
    )  # fmt: skip


def outputs_of(*, model, x_rows, prim, expected_table) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw outputs and sorted quantiles (fraction of capacity), as experiment 510 does."""
    model.table = expected_table
    xt = torch.from_numpy(x_rows).to(device)
    p = torch.from_numpy(prim).to(device)
    with torch.no_grad():
        raw = model(xt, p).float().cpu().numpy()
    q = c510.serving_quantiles(model=model, x=xt, p=p, outlier=False)
    mu = torch.from_numpy(tm).to(device)[p][:, None]
    sd = torch.from_numpy(ts).to(device)[p][:, None]
    return raw, torch.sort(q * sd + mu, dim=1).values.float().cpu().numpy()


fix_x = test.x[pos][:, :60]
fix_prim, fix_oidx = test.primary[pos], test.origin_idx[pos]
dense_out: dict[str, np.ndarray] = {}
sparse_replication = {}
for s in ARGS.seeds:
    # the research table reproduces the stored extras and the stored quantiles (a check of this function)
    again = extras_of(
        x_rows=fix_x,
        prim=fix_prim,
        origin_idx_rows=fix_oidx,
        power=ctx["clean"],
        table=sparse_table[s],
        weather_origin=origin_ft,
    )
    raw_s, q_s = outputs_of(
        model=cand_models[s],
        x_rows=np.concatenate([fix_x, again], axis=1),
        prim=fix_prim,
        expected_table=ctx["table_ft"],
    )
    call = next(c for c in STATE["calls"] if c["seed"] == s)
    sparse_replication[s] = {
        "extras_max_abs_diff": float(np.abs(again - call["x"][:, 60:]).max()),
        "quantiles_max_abs_diff": float(np.abs(q_s - call["q_fix"]).max()),
        "raw_out_max_abs_diff": float(np.abs(raw_s - call["raw_out"]).max()),
    }
    extras_d = extras_of(
        x_rows=fix_x,
        prim=fix_prim,
        origin_idx_rows=fix_oidx,
        power=ctx["clean"],
        table=dense_table[s],
        weather_origin=origin_ft,
    )
    raw_d, q_d = outputs_of(
        model=cand_models[s],
        x_rows=np.concatenate([fix_x, extras_d], axis=1),
        prim=fix_prim,
        expected_table=ctx["table_ft"],
    )
    (
        dense_out[f"extras_dense_s{s}"],
        dense_out[f"raw_out_dense_s{s}"],
        dense_out[f"q_fix_dense_s{s}"],
    ) = extras_d, raw_d, q_d
print("sparse-table replication", sparse_replication, flush=True)

# dense table against the research table where the research table is filled by test rows of the same origin
overlap = {}
for s in ARGS.seeds:
    both = (
        np.isfinite(dense_table[s].median)
        & np.isfinite(sparse_table[s].median)
        & (dense_table[s].origin_idx == sparse_table[s].origin_idx)
    )
    overlap[s] = {
        "entries_with_equal_source_origin": int(both.sum()),
        "max_abs_diff": float(
            np.abs(dense_table[s].median[both] - sparse_table[s].median[both]).max()
        ),
        "research_entries_filled": int(np.isfinite(sparse_table[s].median).sum()),
        "dense_entries_filled": int(np.isfinite(dense_table[s].median).sum()),
    }
print("dense vs research table", overlap, flush=True)

# --- the synthetic case: one fixture origin with absent power, an emptied table T span, and a missing weather run ---
V_US = ORIGINS_US[1]
V_IDX = int((V_US - GRID_START_US) // STEP_US)
v_sel = np.nonzero(test.origin[pos] == V_US)[0]
v_prim = fix_prim[v_sel]
clean_mod = np.array(ctx["clean"], dtype=np.float32)
nb_of = cand_globals["NB"].of
with_nb = [p for p in range(16) if nb_of.get(p)]
without_nb = [p for p in range(16) if not nb_of.get(p)]
last = V_IDX - 48
case = {
    "own_day_lag_absent": 3,
    "own_week_lag_absent": 5,
    "long_gap_before_day_lag": 6,
    "own_and_neighbours_absent": next(p for p in with_nb if p > 6),
    "neighbours_absent_at_day_lag": next(p for p in with_nb if p > 8),
    "table_t_empty_long": 2,
    "table_t_empty_at_day_lag": 11,
    "weather_run_missing_day": 4,
    "weather_run_missing_week": 8,
    "primary_without_neighbours": without_nb[0] if without_nb else None,
}
clean_mod[case["own_day_lag_absent"], V_IDX - 120 : V_IDX - 47] = np.nan
clean_mod[case["own_week_lag_absent"], last - 335 : last - 299] = np.nan
p_c = case["own_and_neighbours_absent"]
clean_mod[[p_c, *nb_of[p_c]], last - 1700 : last - 299] = (
    np.nan
)  # long enough for the 7-day average to lose its weight
clean_mod[case["long_gap_before_day_lag"], last - 1700 : last - 59] = np.nan
p_d = case["neighbours_absent_at_day_lag"]
clean_mod[nb_of[p_d], V_IDX - 120 : V_IDX - 47] = np.nan
t_blank = [  # (Primary, first grid index, one past the last)
    (case["table_t_empty_long"], last - 200, last - 99),
    (case["table_t_empty_at_day_lag"], last - 19, last + 1),
]
weather_blank = [
    (case["weather_run_missing_day"], V_IDX - 60, V_IDX - 29),
    (case["weather_run_missing_week"], last - 335, last - 305),
]
v_oft = origin_ft.copy()
v_table = ctx["table_ft"].clone()
for p, a, b in weather_blank:
    v_oft[p, a:b] = 0
    v_table[p, a:b] = 0
v_x = np.array(fix_x[v_sel])
v_x[:, 37:45] = fit.block_of(
    clean=clean_mod, primary=v_prim, target_idx=test.target_idx[pos][v_sel],
    origin_idx=fix_oidx[v_sel],
)  # fmt: skip
v_out: dict[str, np.ndarray] = {"v_x": v_x, "v_clean": clean_mod}
v_branches = {}
for s in ARGS.seeds:
    t_med, t_org = dense_table[s].median.copy(), dense_table[s].origin_idx.copy()
    for p, a, b in t_blank:
        t_med[p, a:b] = np.nan
        t_org[p, a:b] = -1
    table_mod = s598.LagFreeTable(median=t_med, origin_idx=t_org)
    ex = extras_of(
        x_rows=v_x,
        prim=v_prim,
        origin_idx_rows=fix_oidx[v_sel],
        power=clean_mod,
        table=table_mod,
        weather_origin=v_oft,
    )
    raw_v, q_v = outputs_of(
        model=cand_models[s],
        x_rows=np.concatenate([v_x, ex], axis=1),
        prim=v_prim,
        expected_table=v_table,
    )
    v_out[f"v_extras_s{s}"], v_out[f"v_raw_out_s{s}"], v_out[f"v_q_s{s}"] = ex, raw_v, q_v
    if s == ARGS.seeds[0]:
        v_branches = {
            n: int((ex[:, signal_names.index(n)] > 0.5).sum())
            for n in signal_names
            if n.endswith(("_missing", "_none")) or n.startswith("exp_missing")
        }
print("synthetic case rows with each flag set", v_branches, flush=True)

# expected-weather rows read by the fixtures: the lag windows of each fixture origin and every lag index of the history rows
need_parts = []
for o_idx in ORIGIN_IDX:
    window = np.arange(o_idx - 48 - 335, o_idx - 48 + 1)
    need_parts.append(
        np.stack([np.repeat(np.arange(16), len(window)), np.tile(window, 16)], axis=1)
    )
for xs, ps in ((dense_np["x"], dense_np["primary"]), (first["x"], first["primary"])):
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
    "table_origin": origin_ft[need[:, 0], need[:, 1]],
    "clean": np.asarray(ctx["clean"], dtype=np.float32),
    "n_test": np.array(len(test.x)),
    "nb_capacity": np.asarray(nb.capacity, dtype=np.float64),
    "nb_of_flat": np.array([[p, q] for p, qs in nb.of.items() for q in qs], dtype=np.int64),
    "signal_columns": np.array(columns),
    "signal_names": np.array(signal_names),
    "t_blank": np.array(t_blank, dtype=np.int64),
    "weather_blank": np.array(weather_blank, dtype=np.int64),
    "v_sel": v_sel,
    **{f"dense_{k}": v for k, v in dense_np.items()},
    **dense_out,
    **v_out,
}
for s in ARGS.seeds:
    out[f"dense_k1_q_s{s}"] = dense_q[s]
    out[f"dense_t_median_s{s}"] = dense_table[s].median
    out[f"dense_t_origin_idx_s{s}"] = dense_table[s].origin_idx
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
meta = {
    "seeds": ARGS.seeds,
    "month": ARGS.month,
    "n_extra": n_extra,
    "signal_names": signal_names,
    "history_days": HISTORY_DAYS,
    "dense_origin_range_idx": [ORIGIN_LO, ORIGIN_HI],
    "dense_months": months,
    "sparse_replication": {str(k): v for k, v in sparse_replication.items()},
    "dense_vs_research_table": {str(k): v for k, v in overlap.items()},
    "synthetic_case": case,
    "synthetic_flag_counts": v_branches,
    "synthetic_origin": f"{datetime.fromtimestamp(int(V_US) / 1e6, UTC):%Y-%m-%d}",
}
Path(ARGS.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
print("E00708 capture written", ARGS.out, flush=True)

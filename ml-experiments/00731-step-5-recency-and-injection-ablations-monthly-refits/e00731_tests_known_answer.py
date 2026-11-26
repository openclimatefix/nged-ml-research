# ruff: noqa: E501, PT018, PLR0915
"""Experiment 731: known-answer tests of the weights, the placebo, the injection, and the weather look-ahead.

README. This script checks the code of steps 3 and 4 of experiment 731 (`e00731_k1.py`). Run it with
`python`, not pytest; it asserts and prints. It relies on 677 (`e00677_k1.py`, the original, and
`e00677_fold`), 574 (`e00574_fit`), 510 (`e00510_common.select_rows`), 581 (`e00581_bank`), 591
(`e00591_arm.build_month_syn`), and 710 (`e00710_weights.permuted_age_days`). PYTHONPATH: source
`ml-experiments/00524*/env.sh` from the worktree root, and set `DATA_PATH_INTERNAL`.

Usage: `python e00731_tests_known_answer.py [--skip-build]`. Tests, in this process:

1. the unpatched part of `e00731_k1.py` equals `e00677_k1.py`;
2. the root assertion refuses a folder without `00731` (a child process that must fail);
3. the total weight of the 710 placebo stays within 5% of the true total (both totals are printed).
   The placebo ages are not a permutation of the true ages, because calendar months differ in length
   and 710 clips a row to its new month, so only the total is asserted;
4. the synthetic generator at depth 0.3: the RNG draws are replayed, and the shift of a mid-day
   onset event is exactly +-0.3 over its length and 0 on the half-hour before it;
5. the weather look-ahead: for every origin 2026-07-01 to 2026-10-31 the ensemble summary file's run
   time is at most the origin minus 9 hours, and one example row is printed. **This check depends on
   experiment 607's `init=YYYY-MM-DD.parquet` file naming** (the date is the 00Z run); the check also
   reads each file's own `power_fcst_init_time` and requires it to equal the file's date plus 9 hours,
   which ties the file name to its rows.

Child processes (`--child ARM`; `--child TRAIN` runs test 9), which `exec` the setup of `e00731_k1.py` (everything before the first
command) so that the real patches are tested, on a fixed CPU core pair:

6. R0: weights 1, 0.5, and 0.25 at ages 0, 90, and 180 days (the formula, and the original
   `select_rows` at ages just past its cut), and the E3 halving of the 7 days after the 7-day embargo;
7. WP: the E3 window is unchanged by the placebo, and the placebo's total weight is printed;
8. IP, one real month, CPU only, build only: the lag block of the IP arm equals the plain unshifted
   `build_month` block on all rows and on the columns `x[:, 37:45]`, and a depth of 0.3 shifts the lag
   values by the label's sign where the label is nonzero. This builds one month of rows (about 4
   million rows, a few GB, about a minute on 2 cores): `--skip-build` leaves it out. At depth 0.3,
   the lag value moves by a scale per Primary that equals the scale of `e.block_of` on a series with
   0.3 added to that Primary alone;
9. TRAIN: `e.train` on random rows gives identical state dicts and the same stored epoch with and
   without the patch-C wrappers (`row_nll_recorded`, `select_rows_report`).
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import numpy as np

HERE: Final = Path(__file__).resolve().parent
REPO: Final = HERE.parents[1]
K1_731: Final = HERE / "e00731_k1.py"
K1_677: Final = next((REPO / "ml-experiments").glob("00677-*/e00677_k1.py"))
NAME: Final = "00731-step-5-recency-and-injection-ablations-monthly-refits"
MONTH: Final = "2026-01"
BEGIN: Final = "# >>> E00731 PATCH BEGIN"
END: Final = "# <<< E00731 PATCH END"
CORES: Final = os.environ.get("CORES", "20-21")


def data_root() -> Path:
    """Return the private data folder."""
    return Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"


def strip_patches(*, text: str) -> str:
    """Return `text` without the lines from a patch's BEGIN line to its END line, inclusive."""
    out, inside = [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(BEGIN):
            inside = True
        if not inside:
            out.append(line)
        if line.lstrip().startswith(END):
            inside = False
    assert not inside, "unterminated patch block"
    return "".join(out)


def test_unpatched_equals_original() -> None:
    """Assert that the unpatched part of the copy equals the original from `import argparse` on."""
    mine = K1_731.read_text()
    original = K1_677.read_text()
    n_blocks = mine.count(BEGIN)
    assert n_blocks == mine.count(END) == 3, n_blocks
    cut = "import argparse\n"
    assert strip_patches(text=mine.partition(cut)[2]) == original.partition(cut)[2]
    print(
        f"[1] unpatched part of e00731_k1.py equals e00677_k1.py (3 patch blocks, {len(mine.splitlines())} lines)"
    )
    # the placebo wrapper sits inside the E3 wrapper: it is installed before the E3 block captures `select_rows`
    assert mine.index("e.c.select_rows = select_rows_wp") < mine.index(
        "_select_rows = e.c.select_rows"
    )
    print("    the WP wrapper is installed before the E3 wrapper captures select_rows (inside E3)")


def child_env(*, arm: str, root: Path, month: str = MONTH) -> dict[str, str]:
    """Return the environment of a child process, as `e00731_run.sh` sets it for the train phase."""
    r677 = next(data_root().glob("00677-*"))
    return {
        **os.environ,
        "E00677_ROOT": str(root),
        "E00677_SPAN": "1",
        "E00677_MONTH": month,
        "E00677_PHASE": "train",
        "E00731_ARM": arm,
        "CHAIN677": str(r677 / "chains" / "5" / month / "work"),
        "STATS677": str(r677 / "chains" / "3b" / month / "work"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "NGED_S3_BUCKET_URL": "x",
        "NGED_S3_BUCKET_ACCESS_KEY": "x",
        "NGED_S3_BUCKET_SECRET": "x",
    }


def run_child(
    *, arm: str, root: Path, extra: list[str] | None = None
) -> subprocess.CompletedProcess:
    """Run this script as a child for one arm and return the finished process."""
    cmd = [
        "taskset",
        "-c",
        CORES,
        sys.executable,
        str(Path(__file__)),
        "--child",
        arm,
        *(extra or []),
    ]
    return subprocess.run(
        cmd,
        env=child_env(arm="R0" if arm == "TRAIN" else arm, root=root),
        capture_output=True,
        text=True,
        check=False,
    )


def test_root_assertion() -> None:
    """Assert that the copy refuses an output folder without `00731` in its path (no write happens)."""
    fake = Path("/tmp/not-an-experiment/00677-fake")  # a path only: nothing is created
    argv = [
        "taskset", "-c", CORES, sys.executable, str(K1_731), "k1_R0", "fit", "--step", "5",
        "--months", MONTH, "--seeds", "0",
    ]  # fmt: skip
    proc = subprocess.run(
        argv, env=child_env(arm="R0", root=fake), capture_output=True, text=True, check=False
    )
    assert (
        proc.returncode != 0
        and "experiment 731 writes only under its own data folder" in proc.stderr
    ), proc.stderr[-800:]
    assert not fake.exists()
    print("[2] the root assertion refuses a folder without 00731 and creates nothing")


def fake_rows(*, cutoff: datetime, n_days: int = 760) -> SimpleNamespace:
    """Return a stand-in `MonthRows` with half-hourly valid times and no real data."""
    from e00510_common import DAY_US, us  # ty: ignore[unresolved-import]

    c = us(t=cutoff)
    valid = c - (np.arange(n_days * 48, dtype=np.int64) + 1) * (DAY_US // 48)
    valid = np.sort(valid)
    return SimpleNamespace(
        valid_us=valid,
        origin_us=valid - 3 * DAY_US,
        y=np.zeros(len(valid), dtype=np.float32),
        is_val=np.zeros(len(valid), dtype=bool),
        primary=np.zeros(len(valid), dtype=np.int64),
    )


def test_placebo_total_weight() -> None:
    """Assert that the placebo's total weight stays within 5% of the true total, and print both."""
    import e00510_common as c  # ty: ignore[unresolved-import]
    import e00710_weights as w710  # ty: ignore[unresolved-import]

    for yyyymm in (202511, 202601, 202610):
        cutoff = datetime(yyyymm // 100, yyyymm % 100, 1, tzinfo=UTC)
        rows = fake_rows(cutoff=cutoff)
        refit_us = c.us(t=cutoff)
        floor_us = refit_us - c.WINDOW_DAYS * c.DAY_US
        keep = (rows.valid_us <= refit_us - c.CUTOFF_AGE_US) & (rows.valid_us > floor_us)
        valid = rows.valid_us[keep]
        true_age = w710.age_days(valid_us=valid, refit_us=refit_us)
        perm_age = w710.permuted_age_days(
            valid_us=valid, refit_us=refit_us, floor_us=floor_us, fold_yyyymm=yyyymm
        )
        w_true = 0.5 ** (true_age / 90.0)
        w_perm = 0.5 ** (perm_age / 90.0)
        same_multiset = np.mean(np.sort(true_age) == np.sort(perm_age))
        ratio = w_perm.sum() / w_true.sum()
        print(
            f"[3] {yyyymm}: rows {len(valid):,}; total weight true {w_true.sum():,.1f}, placebo {w_perm.sum():,.1f} "
            f"(ratio {ratio:.4f}); share of sorted ages equal {same_multiset:.3f} (not a permutation: months differ in length)"
        )
        assert abs(ratio - 1) < 0.05, ratio
        assert np.array_equal(
            perm_age,
            w710.permuted_age_days(
                valid_us=valid, refit_us=refit_us, floor_us=floor_us, fold_yyyymm=yyyymm
            ),
        ), "not deterministic"


def test_synthetic_depth() -> None:
    """Assert the synthetic generator at depth 0.3 shifts a constant series by 0.3, with a mid-day onset."""
    import e00581_bank as bank  # ty: ignore[unresolved-import]

    saved = bank.DEPTHS
    bank.DEPTHS = (0.3, 0.3, 0.3)
    try:
        clean = np.ones((2, 48 * 400), dtype=np.float64)
        out, n_events = bank.synthetic_clean(clean=clean, first=0, last=clean.shape[1], seed=202601)
    finally:
        bank.DEPTHS = saved
    shift = out - clean
    values = np.unique(np.round(np.abs(shift), 6))
    assert set(values.tolist()) <= {0.0, 0.3, 0.6, 0.9}, values  # 0.6 and 0.9: overlapping events
    onsets = []
    for p in range(clean.shape[0]):
        nz = np.flatnonzero(shift[p] != 0)
        onsets.extend(int(i) for i in nz[np.r_[True, np.diff(nz) > 1]])
    mid_day = [i for i in onsets if i % 48 != 0]
    assert mid_day, onsets
    # replay the generator's draws and check one isolated mid-day event exactly
    rng = np.random.default_rng(202601)
    events = []
    for p in range(clean.shape[0]):
        covered = 0
        while covered < bank.SYN_SHARE * clean.shape[1]:
            length = int(rng.choice(bank.LENGTHS_DAYS)) * bank.PER_DAY
            on = int(rng.integers(0, max(1, clean.shape[1] - length)))
            depth = float(rng.choice((0.3, 0.3, 0.3))) * float(rng.choice([-1.0, 1.0]))
            events.append((p, on, length, depth))
            covered += length
    assert len(events) == n_events, (len(events), n_events)
    isolated = [
        (p, on, ln, d)
        for (p, on, ln, d) in events
        if on % 48 != 0
        and on > 0
        and on + ln < clean.shape[1]
        and all(
            q != p or (o + n2 <= on or o >= on + ln)
            for (q, o, n2, _) in events
            if (q, o) != (p, on)
        )
    ]
    assert isolated, "no isolated mid-day event to check"
    p, on, ln, d = isolated[0]
    assert np.allclose(shift[p, on : on + ln], d, atol=1e-9), (p, on, ln, d)
    assert shift[p, on - 1] == 0 and shift[p, on + ln] == 0
    first = on
    print(
        f"[4] depth 0.3: {n_events} events, shift magnitudes {values.tolist()}; "
        f"example onset at half-hour {first} = day {first // 48} {(first % 48) // 2:02d}:{30 * (first % 2):02d} UTC (mid-day)"
    )
    bank.DEPTHS = (0.0, 0.0, 0.0)
    try:
        zero, _ = bank.synthetic_clean(clean=clean, first=0, last=clean.shape[1], seed=202601)
    finally:
        bank.DEPTHS = saved
    assert np.array_equal(zero, clean), "depth 0 must leave the series unchanged"
    # same random draws: the events of depth 0 sit where the events of depth 0.3 sit
    print(
        "    depth 0 leaves the series unchanged (same draws, so onsets, lengths, and signs equal arm 9a's)"
    )


def test_weather_lookahead() -> None:
    """Assert, from the ensemble summary files alone, run time <= origin - 9 h for every origin 2026-07 to 2026-10."""
    import polars as pl

    summary = next(data_root().glob("00607-*")) / "work" / "ens_summary"
    day = datetime(2026, 7, 1, tzinfo=UTC)
    n, example = 0, None
    while day < datetime(2026, 11, 1, tzinfo=UTC):
        path = summary / f"init={day:%Y-%m-%d}.parquet"
        assert path.exists(), path
        frame = pl.scan_parquet(path).select("power_fcst_init_time", "valid_time").collect()
        origin = frame["power_fcst_init_time"].unique()
        assert origin.len() == 1, path
        origin_t = origin[0]
        run_t = day  # the file name is the 00Z run's date
        assert origin_t == day + timedelta(hours=9), ("file name and rows disagree", path, origin_t)
        assert run_t <= origin_t - timedelta(hours=9), (path, run_t, origin_t)
        assert frame["valid_time"].min() > origin_t - timedelta(hours=9), path
        example = example or (run_t, origin_t, path.name)
        n += 1
        day += timedelta(days=1)
    assert n == 123, n
    print(
        f"[5] weather look-ahead: {n} origins, run time <= origin - 9 h; example run {example[0]:%Y-%m-%d %H:%M} UTC, origin {example[1]:%Y-%m-%d %H:%M} UTC ({example[2]})"
    )


# ----------------------------------------------------------------------------- children


def exec_setup(*, arm_name: str) -> dict:
    """Run the setup of `e00731_k1.py` (everything before its first command) and return its namespace."""
    src = K1_731.read_text()
    setup = src.split('\nif MINE.command == "pretrain":')[0]
    sys.argv = [str(K1_731), f"k1_{arm_name}", "fit", "--step", "5", "--months", MONTH, "--seeds", "0",
                "--synthetic", "0" if arm_name == "B0" else "1"]  # fmt: skip
    ns: dict = {"__name__": "e00731_k1_setup", "__file__": str(K1_731)}
    exec(compile(setup, str(K1_731), "exec"), ns)  # noqa: S102
    return ns


def child_weights(*, arm: str) -> dict:
    """Test the real patched `select_rows` of the arm on stand-in rows and return a JSON-able summary."""
    ns = exec_setup(arm_name=arm)
    e = ns["e"]
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    rows = fake_rows(cutoff=cutoff)
    refit_us = e.c.us(t=cutoff)
    result: dict = {"arm": arm}
    half_life = e.SPEC.half_life
    final = e.c.select_rows(rows=rows, cutoff=cutoff, half_life_days=half_life)
    age_kept = (refit_us - rows.valid_us[final.index]) / e.c.DAY_US
    window = (age_kept >= 7) & (age_kept < 14)
    if arm == "R0":
        formula = 0.5 ** (np.array([0.0, 90.0, 180.0]) / 90.0)
        assert np.allclose(formula, [1.0, 0.5, 0.25]), formula
        original = ns["_select_rows"](
            rows=rows, cutoff=cutoff, half_life_days=90.0
        )  # 510's, unwrapped
        age_o = (refit_us - rows.valid_us[original.index]) / e.c.DAY_US
        for a in (10.0, 90.0, 180.0, 360.0):
            i = int(np.argmin(np.abs(age_o - a)))
            assert abs(original.weight[i] - 0.5 ** (age_o[i] / 90.0)) < 1e-6
        i0 = int(np.argmin(age_o))
        print(
            f"[6] weights: formula at ages 0, 90, 180 days = {formula.tolist()}; original select_rows at age {age_o[i0]:.3f} d = {original.weight[i0]:.4f}, at ages 90 and 180 = {[round(float(original.weight[int(np.argmin(np.abs(age_o - a)))]), 4) for a in (90, 180)]}"
        )
        expected = 0.5 ** (age_kept / 90.0) * np.where(window, 0.5, 1.0)
        assert np.allclose(final.weight, expected, rtol=1e-5), "E3 halving differs from the formula"
        print(
            f"[6] E3: {int(window.sum())} rows aged 7 to 14 days are halved; weights equal 0.5 ** (age / 90) times the halving"
        )
    elif arm in {"W0", "B0"}:
        assert np.allclose(final.weight, np.where(window, 0.5, 1.0)), (
            "equal weights keep E3's halving"
        )
        print(f"[6] {arm}: equal weights (E3 halving kept on {int(window.sum())} rows)")
    elif arm in {"W30", "W270"}:
        expected = 0.5 ** (age_kept / half_life) * np.where(window, 0.5, 1.0)
        assert np.allclose(final.weight, expected, rtol=1e-5)
        print(f"[6] {arm}: half-life {half_life} days, weights equal the formula")
    elif arm == "WP":
        import e00710_weights as w710  # ty: ignore[unresolved-import]

        floor_us = refit_us - e.c.WINDOW_DAYS * e.c.DAY_US
        keep = (rows.valid_us <= refit_us - e.c.CUTOFF_AGE_US) & (rows.valid_us > floor_us)
        perm = w710.permuted_age_days(
            valid_us=rows.valid_us[keep], refit_us=refit_us, floor_us=floor_us, fold_yyyymm=202601
        )
        wp = 0.5 ** (perm / 90.0)
        plain = ns["_wp_inner"](rows=rows, cutoff=cutoff, half_life_days=90.0)
        assert np.array_equal(final.index, plain.index)
        expected = wp * np.where(window, 0.5, 1.0)  # E3's window from the TRUE ages
        assert np.allclose(final.weight, expected, rtol=1e-5, atol=1e-9), (
            "E3 window or placebo differs"
        )
        assert not np.allclose(final.weight, plain.weight * np.where(window, 0.5, 1.0)), (
            "the placebo changed nothing"
        )
        r0_total = float((plain.weight * np.where(window, 0.5, 1.0)).sum())
        print(
            f"[7] WP: E3 window ({int(window.sum())} rows, true ages 7 to 14 days) unchanged by the placebo; total weight placebo {float(final.weight.sum()):,.1f} versus R0 {r0_total:,.1f} (ratio {float(final.weight.sum()) / r0_total:.4f}); report {final.report.get('placebo_total_weight'):.1f}"
        )
        result["ratio"] = float(final.weight.sum()) / r0_total
    return result


def child_build(*, arm: str) -> dict:
    """Build one real month with the IP patch and compare its lag block with the plain build."""
    assert arm == "IP"
    ns = exec_setup(arm_name="IP")
    e, armmod = ns["e"], ns["arm"]
    clean = e.c.clean_stack()

    def attach_tables_cpu(*, standardiser) -> None:  # noqa: ANN001
        """Build the expected-input table on the CPU (the original uploads it to the GPU)."""
        import torch  # ty: ignore[unresolved-import]

        cols, origin = e.d560.raw_table(source="ft", cache=e.W560 / "expected")
        e.CTX["origin_ft"] = np.asarray(origin)
        e.CTX["table_ft"] = torch.from_numpy(
            e.d560.standardised_table(cols=cols, standardiser=standardiser)
        )

    e.attach_tables = attach_tables_cpu
    plain_builder = armmod._build_month_p5
    mr = plain_builder(month=MONTH, clean=clean)
    sl = slice(37, 45)
    plain_block = mr.x[:, sl].copy()
    plain_x_other = mr.x[:, :37].copy()
    armmod._build_month_p5 = lambda **_: (
        mr
    )  # the same rows; `build_month_syn` overwrites the lag block in place
    armmod.build_month_syn(month=MONTH, clean=clean)
    assert np.array_equal(mr.x[:, sl], plain_block, equal_nan=True), (
        "IP's lag block differs from the plain block"
    )
    assert np.array_equal(mr.x[:, :37], plain_x_other, equal_nan=True)
    lab0 = armmod.AUX[id(mr)].copy()
    assert not np.nan_to_num(lab0).any(), "IP's depth labels must all be zero"
    print(
        f"[8] IP: lag block equals the plain unshifted block on all {len(mr.x):,} rows and columns x[:, 37:45]; all depth labels are zero"
    )
    # depth 0.3 with the real generator's draws: the label and the lag value move together
    armmod.bank.DEPTHS = (0.3, 0.3, 0.3)
    armmod.build_month_syn(month=MONTH, clean=clean)
    lab = armmod.AUX[id(mr)]
    for j, col in enumerate((37, 40)):
        moved = (
            np.abs(np.nan_to_num(lab[:, j])) > 1e-3
        )  # float32 noise of cancelling shifts is below this
        delta = mr.x[:, col] - plain_block[:, col - 37]
        assert moved.any(), "no shifted lag rows"
        assert np.all(
            np.isin(np.round(np.abs(lab[moved, j]).astype(np.float64), 4), [0.3, 0.6, 0.9])
        ), (
            "label is not a multiple of 0.3",
            np.unique(np.round(np.abs(lab[moved, j]).astype(np.float64), 4))[:20],
        )
        agree = np.sign(delta[moved]) == np.sign(lab[moved, j])
        assert agree.mean() > 0.999, (  # a few rows may sit at a clipped value
            "lag value does not move with the label",
            float(agree.mean()),
            np.unique(np.round(delta[moved][:2000], 3))[:12],
            np.unique(np.round(lab[moved, j][:2000], 3))[:12],
            np.isnan(delta[moved]).mean(),
        )
        assert np.abs(np.nan_to_num(delta[~moved])).max() < 1e-3, "an unshifted row moved"
        ratio = delta[moved] / lab[moved, j]
        print(
            f"[8] depth 0.3, lag column {col}: {int(moved.sum()):,} rows shifted, sign agrees on {agree.mean():.5f}; value change per unit label: median {np.median(ratio):.4f}, range {ratio.min():.4f} to {ratio.max():.4f}"
        )
    # direct check of the shift size: add 0.3 to one Primary's whole series and read `e.block_of`
    step_us, grid_us = e.c.prep422.STEP_US, e.c.prep422.GRID_START_US
    origin_idx = (mr.origin_us - grid_us) // step_us
    for prim in range(clean.shape[0]):
        rows_p = np.flatnonzero(mr.primary == prim)[::40]
        fixed_p = clean.copy()
        fixed_p[prim] = np.where(np.isfinite(fixed_p[prim]), fixed_p[prim] + 0.3, fixed_p[prim])
        direct = e.block_of(
            clean=fixed_p,
            primary=mr.primary[rows_p],
            target_idx=mr.target_idx[rows_p],
            origin_idx=origin_idx[rows_p],
        )[:, 0]
        direct = (direct - plain_block[rows_p, 0]) / 0.3
        in_build = (
            (mr.primary == prim)
            & (np.abs(np.nan_to_num(lab[:, 0])) > 0.299)
            & (np.abs(np.nan_to_num(lab[:, 0])) < 0.301)
        )
        built = (mr.x[in_build, 37] - plain_block[in_build, 0]) / lab[in_build, 0]
        d_med, b_med = float(np.nanmedian(direct)), float(np.nanmedian(built))
        assert abs(d_med - b_med) < 1e-3 * max(1.0, abs(d_med)), (prim, d_med, b_med)
        near = float(np.mean(np.abs(built[np.isfinite(built)] - b_med) < 1e-3 * abs(b_med)))
        assert near > 0.9, (
            prim,
            b_med,
            near,
        )  # rows at a clipped value, mostly at the short-history Primaries
        print(
            f"[8] Primary {prim}: scale of a 0.3 shift in the lag value, direct {d_med:.4f}, through the injection {b_med:.4f} ({near:.4f} of rows within 0.1%)"
        )
    # one example row for the weather look-ahead: run time, origin, oldest lag time
    grid_start = e.c.prep422.GRID_START_US
    step = e.c.prep422.STEP_US
    last_origin = mr.origin_us.max()
    rows = np.flatnonzero(mr.origin_us == last_origin)
    lag_idx = np.concatenate([mr.x[rows, 43], mr.x[rows, 44]]).astype(np.int64)
    oldest = datetime.fromtimestamp((grid_start + lag_idx[lag_idx >= 0].min() * step) / 1e6, UTC)
    origin = datetime.fromtimestamp(last_origin / 1e6, UTC)
    run = origin - timedelta(hours=9)
    assert oldest < origin
    print(
        f"[5] example row (month {MONTH}, last origin): ensemble run {run:%Y-%m-%d %H:%M} UTC, origin {origin:%Y-%m-%d %H:%M} UTC, oldest lag time {oldest:%Y-%m-%d %H:%M} UTC"
    )
    return {"arm": arm, "rows": len(mr.x)}


def child_train() -> dict:
    """Train a toy network on random rows with and without the patch-C wrappers; the fit must not change."""
    import copy
    import textwrap

    import torch  # ty: ignore[unresolved-import]

    ns = exec_setup(arm_name="R0")
    e = ns["e"]

    class Toy(torch.nn.Module):
        """Stand-in with the attributes `e.train` reads (weather, datetime, mixer_ft)."""

        def __init__(self) -> None:
            super().__init__()
            self.weather = torch.nn.Linear(4, 4)
            self.datetime = torch.nn.Linear(4, 4)
            self.mixer_ft = torch.nn.Linear(4, 1)

        def forward(self, x, p):  # noqa: ANN001, ANN202
            return self.mixer_ft(torch.tanh(self.weather(x) + self.datetime(x))).squeeze(-1)

    def toy_nll(*, out, z, p):  # noqa: ANN001, ANN202
        return (out - z) ** 2

    g = torch.Generator().manual_seed(1)
    n = 4000
    x = torch.randn(n, 4, generator=g)
    z = x[:, 0] * 0.5 + 0.3 * torch.randn(n, generator=g)
    w = torch.rand(n, generator=g) + 0.1
    p = torch.zeros(n, dtype=torch.long)
    is_val = torch.zeros(n, dtype=torch.bool)
    is_val[-800:] = True
    torch.manual_seed(0)
    first = Toy()
    second = copy.deepcopy(first)
    e.row_nll = toy_nll
    kw = {"x": x, "z": z, "w": w, "p": p, "is_val": is_val, "seed": 3, "lr": 5e-3,
          "max_epochs": 25, "fresh_lr": None, "frozen": False, "anchor": None, "lam": 0.0,
          "device": "cpu"}  # fmt: skip
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    rows = fake_rows(cutoff=cutoff)
    weights_before = e.c.select_rows(rows=rows, cutoff=cutoff, half_life_days=90.0).weight
    epochs_a, best_a = e.train(model=first, **kw)
    source = K1_731.read_text()
    block = source.split("(C: diagnostics")[1].split(END)[0].split("\n", 1)[1]
    block = block.split("    _outer_select_rows = e.c.select_rows\n", 1)[1]
    block = (
        "    _outer_select_rows = e.c.select_rows\n" + block
    )  # drops the hash-marker section (it writes a file)
    exec(compile(textwrap.dedent(block), "patch_c", "exec"), ns)  # noqa: S102
    assert e.row_nll is not toy_nll, "the recording wrapper is not installed"
    weights_after = e.c.select_rows(rows=rows, cutoff=cutoff, half_life_days=90.0).weight
    assert np.array_equal(weights_before, weights_after), "select_rows_report changed the weights"
    ns["HIST"].reset()
    epochs_b, best_b = e.train(model=second, **kw)
    assert (epochs_a, best_a) == (epochs_b, best_b), ((epochs_a, best_a), (epochs_b, best_b))
    for (name, a), (_, b) in zip(
        first.state_dict().items(), second.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), ("state differs", name)
    history, stored = ns["HIST"].finish()
    assert len(history) == epochs_b
    assert abs(history[stored - 1]["val_nll"] - best_b) < 1e-5, (history[stored - 1], best_b)
    print(
        f"[9] patch C leaves the fit unchanged: identical state dicts, {epochs_b} epochs, stored epoch {stored}, validation NLL {best_b:.6f} (history {history[stored - 1]['val_nll']:.6f})"
    )
    return {"epochs": epochs_b, "stored": stored}


def main() -> None:
    """Run the tests in this process, then the children."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--only", nargs="+", default=None, help="run only these children (for example IP)"
    )
    args = parser.parse_args()
    if args.child:
        if args.child == "TRAIN":
            print("CHILD_RESULT " + json.dumps(child_train()))
            return
        fn = child_build if args.child == "IP" else child_weights
        print("CHILD_RESULT " + json.dumps(fn(arm=args.child)))
        return
    if not args.only:
        test_unpatched_equals_original()
        test_placebo_total_weight()
        test_synthetic_depth()
        test_weather_lookahead()
        test_root_assertion()
    root = data_root() / NAME
    arms = ("R0", "W0", "W30", "W270", "WP", "B0", "TRAIN", *(() if args.skip_build else ("IP",)))
    for arm in args.only or arms:
        proc = run_child(arm=arm, root=root)
        sys.stdout.write(
            "".join(ln + "\n" for ln in proc.stdout.splitlines() if ln.startswith(("[", "    ")))
        )
        if proc.returncode != 0 or "CHILD_RESULT" not in proc.stdout:
            print(f"CHILD {arm} FAILED\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
            sys.exit(1)
    print(
        "ALL KNOWN-ANSWER TESTS PASSED"
        + (" (the IP build test was skipped)" if args.skip_build else "")
    )


if __name__ == "__main__":
    main()

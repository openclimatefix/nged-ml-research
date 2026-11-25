# ruff: noqa: D103, E501, I001, PIE808, PERF401
"""Experiment 726: the preflight KP and the known-answer checks of the design (section 6).

Usage (every command reads or writes files only; none loads a network):
    e00726_checks.py kp                      # CPU preflight: files, paths, colouring, checkouts; writes candidate/sets.json
    e00726_checks.py kc4 --month 2026-01     # the all-shifted and host-only panels (needs E00667_PHASE=predict)
    e00726_checks.py kw --month 2026-01      # experiment 713's weather statement and data check
    e00726_checks.py compare kc1|kc2|kc3|kc6|km1|km4|km2|km3 [...]   # file comparisons after the GPU jobs
    e00726_checks.py selftest                # KS1, KS2, KS3 on synthetic frames

KM2, KM3, and KC6 print PASS or FAIL only (`com.blind_report`), because their values measure how
far a forecast moves and a person must not read them before the counted run. Every tolerance is
fixed here, in code, before any forecast exists: 1e-4 of capacity for KC1 to KC3, KC6, KM1, and
KM4; 1e-6 for KM2; and a follow share above 0.02 for the 0.50 shifts in KM3.
"""

import argparse
import difflib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import e00726_common as com
import e00726_score as sc
import numpy as np
import polars as pl

TOL: Final = 1e-4
TOL_ZERO: Final = 1e-6
KM3_MIN_SHARE: Final = 0.02
# folds, seeds, and passes of each comparison, as the design's table fixes them
KC1_CASES: Final = (("2025-11", 0), ("2025-11", 2), ("2026-04", 0), ("2026-04", 2), ("2026-02", 3))
KC2_CASES: Final = tuple((m, r, s) for m in ("2025-11", "2026-06") for r in (0, 7) for s in (0, 1))
KC3_CASES: Final = tuple((m, r, s) for m in ("2025-11", "2026-06") for r in (0, 5) for s in (0, 1))
KM_MONTHS: Final = ("2025-11", "2026-04")


# ---------------------------------------------------------------- file comparison


def compare_frames(
    *, mine: pl.DataFrame, stored: pl.DataFrame, columns: Sequence[str] = com.QCOLS
) -> tuple[bool, float, int]:
    """Return (key sets equal, largest absolute difference of any column, row count) of two frames.

    The frames hold the keys and the columns. The largest difference is taken on the shared keys,
    and `nan` when the key sets differ.
    """
    a = com.normalise_frame(frame=mine)
    b = com.normalise_frame(frame=stored)
    ka = a.select(list(com.KEYS))
    kb = b.select(list(com.KEYS))
    same_keys = (
        ka.height == kb.height
        and ka.join(kb, on=list(com.KEYS), how="anti").is_empty()
        and kb.join(ka, on=list(com.KEYS), how="anti").is_empty()
    )
    joined = a.join(b, on=list(com.KEYS), how="inner", suffix="_stored")
    if joined.is_empty():
        return same_keys, float("nan"), 0
    worst = max(
        float(np.abs((joined[c] - joined[f"{c}_stored"]).to_numpy()).max()) for c in columns
    )
    return same_keys, (worst if same_keys else float("nan")), joined.height


def restrict_to(*, frame: pl.DataFrame, like: pl.DataFrame, month: str) -> pl.DataFrame:
    """Keep the rows of `frame` whose origin day is 0 to 20 and whose lead lies in the range of `like`."""
    lead = lambda f: (  # noqa: E731
        (f["valid_time"].dt.epoch("us") - f["power_fcst_init_time"].dt.epoch("us")) / 3_600_000_000
    )
    lo, hi = float(lead(like).min()), float(lead(like).max())
    return sc.restrict_rows(frame=frame, month=month, lead_lo=lo, lead_hi=hi + 1e-6)


def report(*, name: str, case: str, same_keys: bool, worst: float, tol: float, n: int) -> bool:
    ok = bool(same_keys and worst <= tol)
    print(
        f"{name} {case}: keys {'equal' if same_keys else 'DIFFER'}, max |diff| {worst:.2e} "
        f"(tolerance {tol:.0e}), {n} shared rows -> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def read_mine(*, path: Path, month: str) -> pl.DataFrame:
    assert path.exists(), f"{path} is missing"
    return sc.restrict_rows(
        frame=com.read_forecast(path=path), month=month, lead_lo=0.0, lead_hi=1e9
    )


def check_kc1() -> bool:
    ok = True
    for month, seed in KC1_CASES:
        mine = read_mine(
            path=com.candidate_output(world="zero", month=month, r=0, seed=seed), month=month
        )
        stored = com.read_forecast(
            path=com.root_of_seed(seed=seed).stored_fit(month=month, seed=seed)
        )
        stored = restrict_to(frame=stored, like=mine, month=month)
        ok &= report(name="KC1", case=f"{month} s{seed}", **_cmp(mine, stored, TOL))
    return ok


def _cmp(mine: pl.DataFrame, stored: pl.DataFrame, tol: float) -> dict:
    same, worst, n = compare_frames(mine=mine, stored=stored)
    return {"same_keys": same, "worst": worst, "tol": tol, "n": n}


def check_kc2() -> bool:
    ok = True
    for month, r, seed in KC2_CASES:
        mine_path = com.candidate_output(world="e650", month=month, r=r, seed=seed)
        stored_path = com.root_of_seed(seed=seed).stored_inject(month=month, r=r, seed=seed)
        assert mine_path.stat().st_mtime > 0
        assert com.NAME in str(mine_path), "KC2 must read its own file, never 667's"
        mine = read_mine(path=mine_path, month=month)
        stored = restrict_to(frame=com.read_forecast(path=stored_path), like=mine, month=month)
        ok &= report(name="KC2", case=f"{month} r{r} s{seed}", **_cmp(mine, stored, TOL))
    return ok


def check_kc3() -> bool:
    ok = True
    for month, r, seed in KC3_CASES:
        mine = read_mine(
            path=com.candidate_output(world="bank709", month=month, r=r, seed=seed), month=month
        )
        stored_path = (
            com.folder_of(prefix="00709")
            / "bank"
            / f"none_{month}_s{seed}"
            / f"pass{r:02d}.parquet"
        )
        stored = restrict_to(frame=com.read_forecast(path=stored_path), like=mine, month=month)
        lost = stored.select(list(com.KEYS)).join(
            mine.select(list(com.KEYS)), on=list(com.KEYS), how="anti"
        )
        print(
            f"KC3 {month} r{r} s{seed}: {stored.height} stored keys, {lost.height} not in this run's file",
            flush=True,
        )
        ok &= lost.is_empty()
        mine = mine.join(stored.select(list(com.KEYS)), on=list(com.KEYS), how="inner")
        ok &= report(
            name="KC3", case=f"{month} r{r} s{seed} (every stored key)", **_cmp(mine, stored, TOL)
        )
    return ok


def check_kc6(*, month: str, seed: int, r: int, primary: int, set_index: int) -> bool:
    """KC6: the forecast of a Primary the shift cannot reach equals the stored forecast, and the cut equals the uncut file."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    nb_of, _labels, ids = com.neighbour_lists()
    one = read_mine(
        path=com.candidate_output(world="kc6_one", month=month, r=r, seed=seed), month=month
    )
    stored = restrict_to(
        frame=com.read_forecast(
            path=com.root_of_seed(seed=seed).stored_fit(month=month, seed=seed)
        ),
        like=one,
        month=month,
    )
    reached = {primary} | {p for p, nb in nb_of.items() if primary in nb}
    far = [int(ids[p]) for p in range(len(ids)) if p not in reached]
    same, worst, _n = compare_frames(
        mine=one.filter(pl.col("time_series_id").is_in(far)),
        stored=stored.filter(pl.col("time_series_id").is_in(far)),
    )
    unreached_ok = bool(same and worst <= TOL)
    mover = int(ids[primary])
    moved = compare_frames(
        mine=one.filter(pl.col("time_series_id") == mover),
        stored=stored.filter(pl.col("time_series_id") == mover),
    )[1]
    members = com.read_sets()[set_index]
    cut = read_mine(
        path=com.candidate_output(world="e581_host", month=month, r=r, seed=seed, k=set_index),
        month=month,
    )
    uncut = read_mine(
        path=com.candidate_output(world="kc6_set", month=month, r=r, seed=seed, k=set_index),
        month=month,
    )
    uncut = restrict_to(frame=uncut, like=cut, month=month)
    member_ids = [int(ids[p]) for p in members]
    cut_same, cut_worst, _ = compare_frames(
        mine=cut.filter(pl.col("time_series_id").is_in(member_ids)),
        stored=uncut.filter(pl.col("time_series_id").is_in(member_ids)),
    )
    passed = unreached_ok and moved > TOL and cut.height > 0 and bool(cut_same and cut_worst <= TOL)
    del k578
    return com.blind_report(
        name="kc6",
        passed=passed,
        values={
            "unreached_same_keys": same,
            "unreached_max_diff": worst,
            "shifted_primary_max_diff": moved,
            "cut_same_keys": cut_same,
            "cut_max_diff": cut_worst,
        },
    )


def check_km1() -> bool:
    ok = True
    for month in KM_MONTHS:
        for arm in ("M1", "K", "E", "N", "M1-nohist"):
            mine = read_mine(
                path=com.msa_output(world="zero", arm=arm, month=month, seed=0, r=0), month=month
            )
            stored = restrict_to(
                frame=com.read_forecast(path=com.msa_fit_file(arm=arm, month=month, seed=0)),
                like=mine,
                month=month,
            )
            ok &= report(name="KM1", case=f"{arm} {month} s0", **_cmp(mine, stored, TOL))
    return ok


def km4_host_path(*, month: str, set_index: int) -> Path:
    """Return the file of `M1` in the host-only world for one set, the path the MSA command writes."""
    return com.msa_output(world=com.msa_host_world(k=set_index), arm="M1", month=month, seed=0, r=0)


def check_km4(*, month: str, set_index: int) -> bool:
    _nb, _labels, ids = com.neighbour_lists()
    members = [int(ids[p]) for p in com.read_sets()[set_index]]
    host = read_mine(path=km4_host_path(month=month, set_index=set_index), month=month)
    everything = read_mine(
        path=com.msa_output(world="all", arm="M1", month=month, seed=0, r=0), month=month
    )
    sel = pl.col("time_series_id").is_in(members)
    return report(
        name="KM4",
        case=f"{month} set {set_index}",
        **_cmp(host.filter(sel), everything.filter(sel), TOL),
    )


def _slice_follow(*, arm: str, month: str, r: int, seed: int, zero_arm: str) -> dict[str, float]:
    """Return the largest absolute follow share per slice of one pass, against the arm's own zero-world forecast."""
    primary_of_id, _labels = sc.site_table()
    shifted = read_mine(
        path=com.msa_output(world="all", arm=arm, month=month, seed=seed, r=r), month=month
    )
    base = read_mine(
        path=com.msa_output(world="zero", arm=zero_arm, month=month, seed=seed, r=0), month=month
    )
    both = shifted.select(*com.KEYS, pl.col("q50").alias("s")).join(
        base.select(*com.KEYS, pl.col("q50").alias("b")), on=list(com.KEYS), how="inner"
    )
    rows = com.event_columns(frame=both, month=month, r=r, primary_of_id=primary_of_id)
    follow = com.follow_share(
        shifted=rows["s"].to_numpy(),
        base=rows["b"].to_numpy(),
        sign=rows["sign"].to_numpy(),
        depth=rows["depth"].to_numpy(),
    )
    rows = rows.with_columns(follow=pl.Series(follow))
    return {
        s: float(np.abs(rows.filter(pl.col(s))["follow"].to_numpy()).max())
        for s in com.SLICES
        if rows.filter(pl.col(s)).height
    }


def check_km2() -> bool:
    """KM2: arm `N`, and `M1` with every history row masked, follow share 0 to 1e-6 in every slice."""
    worst: dict[str, float] = {}
    for month in KM_MONTHS:
        for r in (0, 12):
            for arm, zero_arm in (("N", "N"), ("M1-nohist", "M1-nohist")):
                for s, v in _slice_follow(
                    arm=arm, month=month, r=r, seed=0, zero_arm=zero_arm
                ).items():
                    worst[f"{arm}|{month}|r{r}|{s}"] = v
    passed = bool(worst) and max(worst.values()) <= TOL_ZERO
    return com.blind_report(name="km2", passed=passed, values=worst)


def check_km3() -> bool:
    """KM3, follow-share half: `M1` on pass 0 of fold 2025-11 follows the 0.50 shifts by more than 0.02."""
    primary_of_id, _labels = sc.site_table()
    month = "2025-11"
    shifted = read_mine(
        path=com.msa_output(world="all", arm="M1", month=month, seed=0, r=0), month=month
    )
    base = read_mine(path=com.msa_fit_file(arm="M1", month=month, seed=0), month=month)
    both = shifted.select(*com.KEYS, pl.col("q50").alias("s")).join(
        base.select(*com.KEYS, pl.col("q50").alias("b")), on=list(com.KEYS), how="inner"
    )
    rows = com.event_columns(frame=both, month=month, r=0, primary_of_id=primary_of_id)
    rows = rows.filter(pl.col("depth").round(2) == 0.5, pl.col(com.SLICE_LASTING))
    share = (
        float(
            com.follow_share(
                shifted=rows["s"].to_numpy(),
                base=rows["b"].to_numpy(),
                sign=rows["sign"].to_numpy(),
                depth=rows["depth"].to_numpy(),
            ).mean()
        )
        if rows.height
        else float("nan")
    )
    input_file = com.data_root() / "checks" / "blind_km3_input.json"
    input_passed = input_file.exists() and bool(json.loads(input_file.read_text())["passed"])
    com.blind_report(name="km3_input_seen", passed=input_passed, values={})
    return com.blind_report(
        name="km3_follow",
        passed=bool(share > KM3_MIN_SHARE) and input_passed,
        values={"follow_share_050": share},
    )


# ---------------------------------------------------------------- the preflight


ARM_SEEDS: Final = {"M1": (0, 1, 2, 3, 4), "K": (0, 1), "E": (0, 1), "N": (0,)}
"""The seeds at which experiment 713 stored each arm (`K` and `E` have seeds 0 and 1 only)."""


def bank709_pairs() -> list[tuple[str, int, int]]:
    """Return the (month, pass, seed) cases of the KC3 comparison."""
    return list(KC3_CASES)


def expected_inputs() -> list[Path]:
    """Return every stored file that a job or a check reads, per root, arm, seed, and fold month."""
    e713 = com.ML_EXPERIMENTS_DATA_DIR / com.NAME_713
    files: list[Path] = [
        e713 / "cache" / f"{n}.npy" for n in ("dense_raw", "dense_b1_grid", "dense_b1_cams")
    ]
    for month in com.FOLD_MONTHS:
        files.append(e713 / "work" / "standardiser" / f"{month}.sha256")
        for tag in ("e667", "seeds713"):
            files.append(
                com.candidate_root(tag=tag).chain(month=month) / "stats" / "B1" / f"{month}.npz"
            )
        for seed in com.SEEDS_ALL:
            root = com.root_of_seed(seed=seed)
            files += [
                root.model_file(month=month, seed=seed),
                root.table_file(month=month, seed=seed),
                root.stored_fit(month=month, seed=seed),
            ]
        for arm, seeds in ARM_SEEDS.items():
            for seed in seeds:
                files += [
                    com.msa_fit_file(arm=arm, month=month, seed=seed),
                    com.msa_model_file(arm=arm, month=month, seed=seed),
                ]
    for month in KM_MONTHS:
        files.append(com.msa_fit_file(arm="M1-nohist", month=month, seed=0))
    for month, r, seed in KC2_CASES:
        files.append(com.root_of_seed(seed=seed).stored_inject(month=month, r=r, seed=seed))
    e709 = com.folder_of(prefix="00709")
    for month, r, seed in KC3_CASES:
        files += [
            e709 / "worlds" / f"bank_{month}_p{r}.pkl",
            e709 / "bank" / f"none_{month}_s{seed}" / f"pass{r:02d}.parquet",
        ]
    e714 = com.folder_of(prefix="00714")
    for month in com.BANK_FOLDS:
        found = sorted((e714 / "worlds" / "g2").glob(f"{month}_p*.npz"))
        files += found or [e714 / "worlds" / "g2" / f"{month}_p<none found>.npz"]
    return files


def check_kp(*, strict: bool) -> bool:
    ok = True
    missing = [p for p in expected_inputs() if not p.exists()]
    print(f"KP inputs: {len(missing)} missing of {len(expected_inputs())}", flush=True)
    for p in missing[:10]:
        print(f"  missing {p}", flush=True)
    ok &= not missing
    probes = [
        com.candidate_output(world=w, month=m, r=0, seed=0)
        for w in com.CANDIDATE_WORLDS
        for m in com.FOLD_MONTHS
    ] + [
        com.msa_output(world=w, arm="M1", month=m, seed=0, r=0)
        for w in com.MSA_WORLDS
        for m in com.FOLD_MONTHS
    ]
    for path in probes:
        com.refuse_foreign_path(path)
    print(f"KP outputs: {len(probes)} probe paths all inside this experiment's folder", flush=True)
    nb_of, labels, _ids = com.neighbour_lists()
    com.assert_symmetric(neighbours=nb_of)
    sets = com.greedy_colouring(neighbours=nb_of)
    com.check_sets(sets=sets, neighbours=nb_of)
    print(f"KP colouring: {len(sets)} sets {[[labels[p] for p in s] for s in sets]}", flush=True)
    assert len(sets) <= 6, "the colouring needs more than 6 sets: stop and ask"
    if len(sets) != 4:
        print(f"KP WARNING: the design expects 4 sets, the colouring found {len(sets)}", flush=True)
    com.write_json_atomic(obj={"sets": sets, "labels": labels}, path=com.sets_file())
    record = {}
    if strict:
        record["frozen_713"] = com.assert_frozen_713()
        record["pinned_f4"] = com.assert_pinned_f4()
        ok &= check_g2_copy()
    print(f"KP checkouts: {record}", flush=True)
    print(f"KP {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def g2_core(text: str) -> list[str]:
    """Return the lines of `g2_panel` from the import to the write of the generator's power.

    A trailing comment is stripped from each line, so `# ty: ignore[...]` on the import does not
    stop the match. Blank lines and comment lines are dropped.
    """
    first = "import e00714_generator2 as g2"
    last = "out[:, lo : lo + n] = gen.power.power"
    block = text[text.index("def g2_panel(") :].split("\n\n\n", maxsplit=1)[0]
    lines = [re.sub(r"\s+#.*$", "", x).strip() for x in block.splitlines()]
    lines = [x for x in lines if x and not x.startswith("#")]
    return lines[lines.index(first) : lines.index(last) + 1]


def g2_returns_touched(text: str) -> bool:
    """Return whether the `g2_panel` block of `text` holds the return line of the copy."""
    block = text[text.index("def g2_panel(") :].split("\n\n\n", maxsplit=1)[0]
    return "return out, list(gen.touched)" in [x.strip() for x in block.splitlines()]


def check_g2_copy() -> bool:
    """Compare the copy of `g2_panel` with F4's, body only, and check the return line of the copy."""
    f4_file = com.module_folder(name="e00715_era") / "e00715_era.py"
    mine_text = (Path(__file__).resolve().parent / "e00726_candidate_panel.py").read_text()
    a, b = g2_core(f4_file.read_text()), g2_core(mine_text)
    diff = [
        d for d in difflib.unified_diff(a, b, lineterm="", n=0)
        if d[:1] in "+-" and d[:3] not in {"+++", "---"}
    ]  # fmt: skip
    returns = g2_returns_touched(mine_text)
    print(
        f"KP g2_panel copy: {len(a)} lines compared, {len(diff)} differ, return line ok: {returns}",
        flush=True,
    )
    for d in diff[:10]:
        print(f"  {d}", flush=True)
    return not diff and returns and bool(a)


def check_kc4(*, month: str) -> bool:
    e = com.prime_pipeline(month=month)
    clean = e.c.clean_stack()
    grid, step = e.c.prep422.GRID_START_US, e.c.prep422.STEP_US
    month_idx = (com.month_start_us(month=month) - grid) // step
    failures = com.check_panels(
        clean=clean,
        month_start_idx=int(month_idx),
        passes=list(range(com.N_PASSES)),
        sets=com.read_sets(),
    )
    for f in failures[:20]:
        print(f"KC4 {month}: {f}", flush=True)
    print(
        f"KC4 {month}: {'PASS' if not failures else 'FAIL'} ({len(failures)} failures)", flush=True
    )
    return not failures


def check_kw(*, month: str) -> bool:
    import e00713_weather_lookahead as wl  # ty: ignore[unresolved-import]
    from datetime import UTC, datetime

    y, m = (int(v) for v in month.split("-"))
    contract = datetime(y, m, 10, 12, tzinfo=UTC)
    print("\n".join(wl.statement(contract=contract, primary="D5")), flush=True)
    ok, lines = wl.check_data(contract=contract)
    print("\n".join(lines), flush=True)
    print(f"KW {month}: {'PASS' if ok else 'FAIL'}", flush=True)
    return bool(ok)


# ---------------------------------------------------------------- synthetic self-tests


def selftest_ks1() -> bool:
    """KS1: the scorer returns 0.37 in every slice for a raising and a lowering pass, and 0 for equal frames."""
    primary_of_id = {1000 + p: p for p in range(16)}
    month, ok = "2026-01", True
    ms = com.month_start_us(month=month)
    for r in (0, 13):
        origins = [ms + int(d * com.DAY_US) for d in range(0, 21)]
        rows = [
            {
                "time_series_id": 1000 + p,
                "power_fcst_init_time": o,
                "valid_time": o + 30 * 3_600_000_000,
                "base": 0.5,
            }
            for p in range(16)
            for o in origins
        ]
        frame = pl.DataFrame(rows).with_columns(
            pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        )
        ev = com.event_columns(frame=frame, month=month, r=r, primary_of_id=primary_of_id)
        signed = ev["sign"].to_numpy() * ev["depth"].to_numpy()
        for factor, want in ((0.37, 0.37), (0.0, 0.0)):
            shifted = ev["base"].to_numpy() + factor * signed
            follow = com.follow_share(
                shifted=shifted,
                base=ev["base"].to_numpy(),
                sign=ev["sign"].to_numpy(),
                depth=ev["depth"].to_numpy(),
            )
            for s in com.SLICES:
                mask = ev[s].to_numpy()
                if mask.any() and not np.allclose(follow[mask], want, atol=1e-9):
                    ok = False
    print(f"KS1 {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def selftest_ks2() -> bool:
    """KS2: the origin days of each slice for pass 0 and pass 13 of fold 2026-01 equal the table of section 2."""
    month, ok = "2026-01", True
    primary_of_id = {1000 + p: p for p in range(16)}
    ms = com.month_start_us(month=month)
    frame = pl.DataFrame(
        {
            "time_series_id": [1000 + p for p in range(16) for _ in range(29)],
            "power_fcst_init_time": [ms + d * com.DAY_US for _ in range(16) for d in range(29)],
            "valid_time": [
                ms + d * com.DAY_US + 30 * 3_600_000_000 for _ in range(16) for d in range(29)
            ],
        }
    ).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )
    for r in (0, 13):
        ev = com.event_columns(frame=frame, month=month, r=r, primary_of_id=primary_of_id)
        for p in range(16):
            one = ev.filter(pl.col("primary") == p)
            length = int(one["length_days"][0])
            days = {s: sorted(one.filter(pl.col(s))["origin_day"].to_list()) for s in com.SLICES}
            want_after = list(range(14, 21))
            want_lasting = [d for d in range(0, 14) if 7 <= d - (14 - length) < 28]
            want_d4 = [d for d in range(0, 14) if d - (14 - length) >= 4]
            ok &= days[com.SLICE_AFTER] == want_after
            ok &= days[com.SLICE_LASTING] == want_lasting
            ok &= days[com.SLICE_D4END] == want_d4
    print(f"KS2 {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def selftest_ks3() -> bool:
    """KS3: the join assertions of the same-origin rule fire on a frame that breaks each one."""
    month = "2026-01"
    ms = com.month_start_us(month=month)
    good = pl.DataFrame(
        {
            "time_series_id": [1, 1],
            "power_fcst_init_time": [ms, ms + com.DAY_US],
            "valid_time": [ms + 30 * 3_600_000_000, ms + com.DAY_US + 30 * 3_600_000_000],
            "A:q50:s": [0.1, 0.2],
        }
    ).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )
    pairs = good.select("time_series_id", "power_fcst_init_time")
    sc.assert_same_origin(joined=good, expected_pairs=pairs, month=month, levels=("q50",))
    broken = {
        "repeated key": pl.concat([good, good]),
        "non-finite": good.with_columns(pl.Series("A:q50:s", [0.1, float("nan")])),
        "origin before month": good.with_columns(
            pl.col("power_fcst_init_time") - pl.duration(days=40)
        ),
        "dropped origin": good.head(1),
    }
    fired = 0
    for name, frame in broken.items():
        try:
            sc.assert_same_origin(joined=frame, expected_pairs=pairs, month=month, levels=("q50",))
        except AssertionError:
            fired += 1
        else:
            print(f"KS3: no assertion fired for {name}", flush=True)
    ok = fired == len(broken)
    print(f"KS3 {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    kp = sub.add_parser("kp")
    kp.add_argument(
        "--no-checkouts", action="store_true", help="skip the frozen-checkout assertions"
    )
    for name in ("kc4", "kw"):
        p = sub.add_parser(name)
        p.add_argument("--month", required=True, choices=com.FOLD_MONTHS)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("which", choices=("kc1", "kc2", "kc3", "kc6", "km1", "km2", "km3", "km4"))
    cmp_.add_argument("--month", default="2025-11")
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.add_argument("--pass", dest="r", type=int, default=0)
    cmp_.add_argument("--primary", type=int, default=0)
    cmp_.add_argument("--set", dest="set_index", type=int, default=0)
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "kp":
        return int(not check_kp(strict=not args.no_checkouts))
    if args.command == "kc4":
        return int(not check_kc4(month=args.month))
    if args.command == "kw":
        return int(not check_kw(month=args.month))
    if args.command == "selftest":
        return int(not all([selftest_ks1(), selftest_ks2(), selftest_ks3()]))
    runs = {
        "kc1": check_kc1,
        "kc2": check_kc2,
        "kc3": check_kc3,
        "km1": check_km1,
        "km2": check_km2,
        "km3": check_km3,
    }
    if args.which == "kc6":
        ok = check_kc6(
            month=args.month,
            seed=args.seed,
            r=args.r,
            primary=args.primary,
            set_index=args.set_index,
        )
    elif args.which == "km4":
        ok = check_km4(month=args.month, set_index=args.set_index)
    else:
        ok = runs[args.which]()
    return int(not ok)


if __name__ == "__main__":
    sys.exit(main())

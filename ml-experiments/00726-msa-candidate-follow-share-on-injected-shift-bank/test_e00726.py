# ruff: noqa: E402, PT018, PT011, ANN201, ANN003, D103, E501, PERF401
"""Tests of experiment 726's panels, slices, follow share, scorer, and readings. Synthetic data only."""

import sys
from pathlib import Path

HERE = Path(__file__).parent
for folder in (HERE, HERE.parent / "00581-robust-multiweek-lag-encoder-x3"):
    sys.path.insert(0, str(folder))

import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00726_checks as chk
import e00726_common as com
import e00726_score as sc
import numpy as np
import polars as pl
import pytest

MONTH = "2026-01"
N_SITES = 16
N_STEPS = 70 * com.PER_DAY
MONTH_IDX = 32 * com.PER_DAY


def clean_panel() -> np.ndarray:
    rng = np.random.default_rng(0)
    panel = 0.5 + 0.1 * rng.standard_normal((N_SITES, N_STEPS))
    panel[3, 100:110] = np.nan
    return panel


def test_greedy_colouring_gives_independent_sets() -> None:
    nb = {0: [1, 2], 1: [0], 2: [0, 3], 3: [2], 4: []}
    sets = com.greedy_colouring(neighbours=nb)
    com.check_sets(sets=sets, neighbours=nb)
    assert sorted(p for s in sets for p in s) == [0, 1, 2, 3, 4]
    assert len(sets) == 2


def test_asymmetric_neighbour_lists_are_refused() -> None:
    with pytest.raises(AssertionError):
        com.greedy_colouring(neighbours={0: [1], 1: []})
    with pytest.raises(AssertionError):
        com.check_sets(sets=[[0, 1]], neighbours={0: [1], 1: [0]})


def test_host_panel_shifts_only_the_members() -> None:
    clean = clean_panel()
    r, members = 0, {2, 5}
    allp = com.all_shifted_panel(clean=clean, month_start_idx=MONTH_IDX, r=r)
    host = com.host_panel(clean=clean, month_start_idx=MONTH_IDX, r=r, members=members)
    for p in range(N_SITES):
        want = allp[p] if p in members else clean[p]
        assert np.array_equal(host[p], want, equal_nan=True)
    assert not np.array_equal(allp, clean, equal_nan=True)
    assert (
        com.check_panels(clean=clean, month_start_idx=MONTH_IDX, passes=[0, 13], sets=[[2, 5]])
        == []
    )


def test_all_shifted_panel_matches_bank_event_inside_window() -> None:
    clean = clean_panel()
    r = 13
    allp = com.all_shifted_panel(clean=clean, month_start_idx=MONTH_IDX, r=r)
    end = MONTH_IDX + bank.END_DAY * com.PER_DAY
    for p in range(N_SITES):
        signed, length, _ = bank.event(primary=p, r=r)
        window = slice(end - length * com.PER_DAY, end)
        ok = np.isfinite(clean[p, window])
        assert np.allclose(allp[p, window][ok], clean[p, window][ok] + signed)
        assert np.array_equal(allp[p, end:], clean[p, end:], equal_nan=True)


def test_cut_mask_keeps_origin_days_lead_and_members() -> None:
    origin = np.array(
        [MONTH_IDX - 1, MONTH_IDX, MONTH_IDX + 20 * 48, MONTH_IDX + 21 * 48, MONTH_IDX]
    )
    target = origin + np.array([48, 48, 48, 48, 600])
    primary = np.array([1, 1, 1, 1, 1])
    keep = com.cut_mask(
        origin_idx=origin,
        target_idx=target,
        primary=primary,
        month_start_idx=MONTH_IDX,
        members=None,
    )
    assert keep.tolist() == [False, True, True, False, False]
    other = com.cut_mask(
        origin_idx=origin,
        target_idx=target,
        primary=primary,
        month_start_idx=MONTH_IDX,
        members={2},
    )
    assert not other.any()


def test_output_path_guard() -> None:
    inside = com.candidate_output(world="zero", month=MONTH, r=0, seed=0)
    assert com.refuse_foreign_path(inside) == inside
    assert com.NAME in str(inside) and "zero" in inside.name
    with pytest.raises(ValueError):
        com.refuse_foreign_path(com.data_root().parent / "00667-something" / "x.parquet")
    with pytest.raises(ValueError):
        com.refuse_foreign_path(com.data_root() / "00667-link" / "x.parquet")
    host = com.candidate_output(world="e581_host", month=MONTH, r=3, seed=1, k=2)
    assert "e581_host_k2" in host.name


def test_candidate_root_seeds() -> None:
    com.root_of_seed(seed=1).require_seed(seed=1)
    with pytest.raises(ValueError):
        com.candidate_root(tag="e667").require_seed(seed=3)
    assert com.root_of_seed(seed=3).tag == "seeds713"


def frame_for(*, r: int, days: range, lead_hours: float = 30.0) -> pl.DataFrame:
    ms = com.month_start_us(month=MONTH)
    rows = [
        (1000 + p, ms + d * com.DAY_US, ms + d * com.DAY_US + int(lead_hours * 3_600_000_000))
        for p in range(N_SITES)
        for d in days
    ]
    return pl.DataFrame(
        rows, schema=["time_series_id", "power_fcst_init_time", "valid_time"], orient="row"
    ).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )


@pytest.mark.parametrize("r", [0, 13])
def test_slices_follow_the_design_table(r: int) -> None:
    ids = {1000 + p: p for p in range(N_SITES)}
    ev = com.event_columns(
        frame=frame_for(r=r, days=range(29)), month=MONTH, r=r, primary_of_id=ids
    )
    for p in range(N_SITES):
        one = ev.filter(pl.col("primary") == p)
        length = int(one["length_days"][0])
        after = sorted(one.filter(pl.col(com.SLICE_AFTER))["origin_day"].to_list())
        lasting = sorted(one.filter(pl.col(com.SLICE_LASTING))["origin_day"].to_list())
        assert after == list(range(14, 21))
        assert lasting == [d for d in range(14) if 7 <= d - (14 - length) < 28]
        d4 = sorted(one.filter(pl.col(com.SLICE_D4END))["origin_day"].to_list())
        assert d4 == [d for d in range(14) if d - (14 - length) >= 4]


def test_lasting_slice_examples_of_the_design() -> None:
    """15-day shifts count at origin days 6 to 13, and 30-day shifts at days 0 to 11."""
    ids = {1000 + p: p for p in range(N_SITES)}
    seen = set()
    for r in range(bank.N_PASSES):
        ev = com.event_columns(
            frame=frame_for(r=r, days=range(21)), month=MONTH, r=r, primary_of_id=ids
        )
        for length, want in ((15, list(range(6, 14))), (30, list(range(12)))):
            sel = ev.filter(pl.col("length_days") == length, pl.col(com.SLICE_LASTING))
            if not sel.is_empty():
                seen.add(length)
                assert sorted(sel["origin_day"].unique().to_list()) == want
    assert seen == {15, 30}


def test_follow_share_recovers_the_known_fraction() -> None:
    ids = {1000 + p: p for p in range(N_SITES)}
    for r in (0, 13):
        ev = com.event_columns(
            frame=frame_for(r=r, days=range(21)), month=MONTH, r=r, primary_of_id=ids
        )
        signed = ev["sign"].to_numpy() * ev["depth"].to_numpy()
        base = np.full(ev.height, 0.5)
        got = com.follow_share(
            shifted=base + 0.37 * signed,
            base=base,
            sign=ev["sign"].to_numpy(),
            depth=ev["depth"].to_numpy(),
        )
        assert np.allclose(got, 0.37)
        zero = com.follow_share(
            shifted=base, base=base, sign=ev["sign"].to_numpy(), depth=ev["depth"].to_numpy()
        )
        assert np.allclose(zero, 0.0)


def keyed(values: list[list[float]], *, ids: list[int] | None = None) -> pl.DataFrame:
    ms = com.month_start_us(month=MONTH)
    n = len(values)
    data = {
        "time_series_id": ids or [1] * n,
        "power_fcst_init_time": [ms + i * com.DAY_US for i in range(n)],
        "valid_time": [ms + i * com.DAY_US + 30 * 3_600_000_000 for i in range(n)],
    }
    for j, q in enumerate(com.QCOLS):
        data[q] = [v[j] for v in values]
    return pl.DataFrame(data).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )


def test_seed_mean_sorts_levels_and_drops_missing_keys() -> None:
    a = keyed([[3.0, 2.0, *([1.0] * 9)], [1.0] * 11])
    b = keyed([[1.0, 2.0, *([3.0] * 9)]])
    out = com.seed_mean(frames=[a, b])
    assert out.height == 1
    row = out.select(list(com.QCOLS)).row(0)
    assert list(row) == sorted(row)
    with pytest.raises(AssertionError):
        com.seed_mean(frames=[pl.concat([a, a])])


def test_same_origin_assertions() -> None:
    assert chk.selftest_ks3()
    assert chk.selftest_ks2()
    assert chk.selftest_ks1()


def test_compare_frames() -> None:
    a = keyed([[0.1] * 11, [0.2] * 11])
    b = a.with_columns(pl.col("q50") + 5e-5)
    same, worst, n = chk.compare_frames(mine=a, stored=b)
    assert same and n == 2 and 4e-5 < worst < 6e-5
    same, worst, _ = chk.compare_frames(mine=a, stored=b.head(1))
    assert not same and np.isnan(worst)


def test_bootstrap_interval_properties() -> None:
    rng = np.random.default_rng(1)
    sums = rng.normal(0.30, 0.02, (4, 24))
    counts = np.ones((4, 24))
    low, high = sc.bootstrap_interval(sums=sums, counts=counts, draws=2000)
    assert low < sc.point_estimate(sums=sums, counts=counts) < high
    assert 0.2 < low < high < 0.4
    narrow = sc.bootstrap_interval(sums=sums, counts=counts, draws=2000, level=80.0)
    assert narrow[0] > low and narrow[1] < high
    zero = sc.bootstrap_interval(sums=np.zeros((4, 24)), counts=counts, draws=500)
    assert zero == (0.0, 0.0)


def cell(d: float, low: float, high: float, *, single: list[float] | None = None, **kw):
    single = single if single is not None else [d] * 5
    return sc.read_cell(
        d=d,
        low=low,
        high=high,
        single=single,
        split_difference=kw.get("split", 0.0),
        d_without_050=kw.get("no050", d),
        plan=kw.get("plan", "five"),
    )


def test_read_cell_table() -> None:
    assert cell(0.10, 0.06, 0.14) == "M1 follows more"
    assert cell(-0.10, -0.14, -0.06) == "M1 follows less"
    assert cell(0.03, 0.01, 0.05) == "small difference"
    assert cell(0.0, -0.02, 0.02) == "no detectable difference"
    assert cell(0.0, -0.10, 0.10) == "cannot tell"


def test_read_cell_guards() -> None:
    assert cell(0.10, 0.06, 0.14, split=0.04) == "cannot tell"
    assert cell(0.10, 0.06, 0.14, no050=-0.01) == "cannot tell"
    assert cell(0.10, 0.06, 0.14, single=[0.1, 0.1, 0.1, -0.1, -0.1]) != "M1 follows more"
    assert cell(0.10, 0.06, 0.14, single=[0.1, 0.1, 0.1, 0.1, -0.1]) == "M1 follows more"
    assert cell(0.10, 0.06, 0.14, single=[0.1, 0.1], plan="two", split=9.0) == "M1 follows more"
    assert cell(0.10, 0.06, 0.14, single=[0.1, -0.1], plan="two") != "M1 follows more"
    # a no-difference reading does not depend on the sign rule
    assert cell(0.0, -0.02, 0.02, no050=0.01) == "no detectable difference"


def verdict(more: str, *others: str, d: float = 0.1) -> str:
    cells = [
        ("T1", "six_signed_mw"),
        ("T1", "eight_mva"),
        ("L1", "six_signed_mw"),
        ("L1", "eight_mva"),
    ]
    readings = dict(zip(cells, [more, *others], strict=True))
    return sc.verdict_of(
        readings=readings,  # ty: ignore[invalid-argument-type]
        d_of=dict.fromkeys(cells, d),
    )


def test_verdict_rows() -> None:
    mf, ml, sd, nd, ct = (
        "M1 follows more",
        "M1 follows less",
        "small difference",
        "no detectable difference",
        "cannot tell",
    )
    assert verdict(mf, mf, mf, mf) == "supported"
    assert verdict(mf, ct, nd, ct) == "partly supported"
    assert verdict(ml, ml, mf, mf) == "refuted"
    assert verdict(sd, sd, sd, sd) == "small gain"
    assert verdict(nd, nd, nd, nd) == "not supported"
    assert verdict(ct, nd, ct, ct) == "inconclusive"


def test_lead_indices_cover_the_band() -> None:
    idx = com.lead_indices(lo_hours=24.0, hi_hours=240.0)
    assert idx == sorted(idx) and idx[0] <= 1 and idx[-1] >= 9
    short = com.lead_indices(lo_hours=24.0, hi_hours=120.0)
    assert set(short) <= set(idx)


def test_clip_diagnostic_counts_beyond_the_clip() -> None:
    clean = np.full((N_SITES, N_STEPS), 0.5)
    out = com.clip_diagnostic(
        clean=clean,
        target_mean=np.full(N_SITES, 0.5),
        target_std=np.full(N_SITES, 0.05),
        month_start_idx=MONTH_IDX,
        labels=[f"D{p}" for p in range(N_SITES)],
        passes=[0],
    )
    assert out and all(0.0 <= row["beyond_clip"] <= 1.0 for row in out)
    assert any(row["beyond_clip"] == 1.0 for row in out)


def test_collect_and_contrast_on_stored_like_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A synthetic pair of networks: M1 follows 0.6 and C follows 0.2, so the contrast is 0.4."""
    events = []
    for r in range(3):
        for p in range(N_SITES):
            events.append(
                {
                    "primary": p,
                    "depth": 0.3,
                    "sign": 1,
                    "length_days": 7,
                    "slice": com.SLICE_AFTER,
                    "month": MONTH,
                    "pass_r": r,
                    "M1|q50": 0.6,
                    "C_host|q50": 0.2,
                }
            )
    frame = pl.DataFrame(events)
    out = sc.contrast(
        events=frame,
        a="M1",
        b="C_host",
        slice_name=com.SLICE_AFTER,
        primaries=list(range(N_SITES)),
        months=[MONTH],
        draws=500,
    )
    assert (
        out["D"] == pytest.approx(0.4)
        and out["low"] == pytest.approx(0.4)
        and out["n_events"] == 48
    )
    assert (
        sc.contrast(
            events=frame,
            a="M1",
            b="C_host",
            slice_name=com.SLICE_AFTER,
            primaries=[0],
            months=[MONTH],
            draws=None,
            max_depth=0.15,
        )["n_events"]
        == 0
    )
    del tmp_path, monkeypatch


SETS = [list(range(4 * k, 4 * k + 4)) for k in range(4)]


def write_stored_like(
    *, root: Path, factors: dict[int, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write stored-like files for `C_host` (4 set files per pass) and `M1` under `root`.

    The shifted median of seed `s` equals the base median plus `factors[s]` times the signed depth,
    on all 11 quantiles. The base files stand in for the stored fits.
    """
    ids = {1000 + p: p for p in range(N_SITES)}
    keys = frame_for(r=0, days=range(21))
    base_q = np.array([0.30 + 0.01 * k for k in range(11)])

    def with_q(frame: pl.DataFrame, shift: np.ndarray) -> pl.DataFrame:
        return frame.with_columns(
            [pl.Series(q, base_q[j] + shift) for j, q in enumerate(com.QCOLS)]
        )

    zero = np.zeros(keys.height)
    base = with_q(keys, zero)
    for seed in factors:
        (root / "base").mkdir(parents=True, exist_ok=True)
        base.write_parquet(root / "base" / f"C_s{seed}.parquet")
        base.write_parquet(root / "base" / f"M1_s{seed}.parquet")
    monkeypatch.setattr(com, "data_root", lambda: root / "mine")
    monkeypatch.setattr(
        com, "msa_fit_file", lambda *, arm, month, seed: root / "base" / f"{arm}_s{seed}.parquet"
    )
    monkeypatch.setattr(
        com.CandidateRoot,
        "stored_fit",
        lambda self, *, month, seed: root / "base" / f"C_s{seed}.parquet",
    )
    monkeypatch.setattr(com, "read_sets", lambda: SETS)
    for r in (0, 13):
        ev = com.event_columns(frame=keys, month=MONTH, r=r, primary_of_id=ids)
        signed = ev["sign"].to_numpy() * ev["depth"].to_numpy()
        # event_columns drops nothing here, but keep the row order of `keys`
        assert ev.height == keys.height
        for seed, factor in factors.items():
            shifted = with_q(ev.select(*com.KEYS), factor * signed)
            out = com.msa_output(world="all", arm="M1", month=MONTH, seed=seed, r=r)
            out.parent.mkdir(parents=True, exist_ok=True)
            shifted.write_parquet(out)
            for k, members in enumerate(SETS):
                part = shifted.filter(pl.col("time_series_id").is_in([1000 + p for p in members]))
                path = com.candidate_output(world="e581_host", month=MONTH, r=r, seed=seed, k=k)
                path.parent.mkdir(parents=True, exist_ok=True)
                part.write_parquet(path)


@pytest.mark.parametrize(("factors", "want"), [({0: 0.30, 1: 0.44}, 0.37), ({0: 0.0, 1: 0.0}, 0.0)])
def test_collect_returns_the_known_follow_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factors: dict[int, float], want: float
) -> None:
    write_stored_like(root=tmp_path, factors=factors, monkeypatch=monkeypatch)
    ids = {1000 + p: p for p in range(N_SITES)}
    got = sc.collect(
        nets=(sc.C_HOST, sc.M1),
        seeds=tuple(factors),
        months=[MONTH],
        passes=[0, 13],
        primary_of_id=ids,
        n_sets=4,
    )
    for net in ("C_host", "M1"):
        for slice_name in com.SLICES:
            level = sc.level_of(
                events=got.events,
                net=net,
                slice_name=slice_name,
                primaries=list(range(16)),
                months=[MONTH],
            )
            assert level == pytest.approx(want, abs=1e-5), (net, slice_name)
    swapped = sc.contrast(
        events=got.events, a="M1", b="C_host", slice_name=com.SLICE_AFTER, primaries=list(range(16)),
        months=[MONTH], draws=None,
    )  # fmt: skip
    assert swapped["D"] == pytest.approx(0.0, abs=1e-5)


def test_host_only_world_with_a_lost_set_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_stored_like(root=tmp_path, factors={0: 0.37}, monkeypatch=monkeypatch)
    com.candidate_output(world="e581_host", month=MONTH, r=0, seed=0, k=2).unlink()
    ids = {1000 + p: p for p in range(N_SITES)}
    with pytest.raises((AssertionError, OSError)):
        sc.collect(
            nets=(sc.C_HOST, sc.M1),
            seeds=(0,),
            months=[MONTH],
            passes=[0],
            primary_of_id=ids,
            n_sets=4,
        )


def test_msa_host_world_path_is_shared_by_writer_and_reader() -> None:
    assert chk.km4_host_path(month=MONTH, set_index=2) == com.msa_output(
        world=com.msa_host_world(k=2), arm="M1", month=MONTH, seed=0, r=0
    )
    assert "host_k2" in str(chk.km4_host_path(month=MONTH, set_index=2))


def test_expected_inputs_ask_only_for_files_that_exist_by_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(com, "folder_of", lambda *, prefix: tmp_path / prefix)
    files = {str(p) for p in chk.expected_inputs()}
    assert not any(
        "/fits/K/" in f and f.endswith(("_s2.parquet", "_s3.parquet", "_s4.parquet")) for f in files
    )
    assert not any(
        "/fits/E/" in f and f.endswith(("_s2.parquet", "_s3.parquet", "_s4.parquet")) for f in files
    )
    assert any("stage3" in f for f in files) and any("sha256" in f for f in files)
    assert any("dense_raw" in f for f in files) and any("B1" in f for f in files)


def test_read_cell_with_a_missing_value_cannot_tell() -> None:
    nan = float("nan")
    assert cell(0.10, 0.06, 0.14, split=nan) == "cannot tell"
    assert cell(nan, 0.06, 0.14) == "cannot tell"
    assert cell(0.10, 0.06, 0.14, no050=nan) == "cannot tell"


def test_gates_raise_without_assert() -> None:
    with pytest.raises(RuntimeError):
        com.require(False, "x")
    com.require(True, "x")


def test_crps_from_quantiles_is_zero_for_a_perfect_point_forecast_and_grows_with_error() -> None:
    import e00726_readings as rd

    q = np.tile(np.linspace(0.4, 0.6, 11), (3, 1))  # in capacity shares
    actual = np.array([0.5, 0.5, 0.9])
    crps = rd.crps_from_quantiles(quantiles=q, actual=actual)
    assert crps[0] == pytest.approx(crps[1]) and crps[2] > crps[0]
    sharp = rd.crps_from_quantiles(quantiles=np.full((1, 11), 0.5), actual=np.array([0.5]))
    assert sharp[0] == pytest.approx(0.0)


G2_TEXT = """
def g2_panel(*, clean, month, index):
    \"\"\"Doc.\"\"\"
    import e00714_generator2 as g2  # ty: ignore[unresolved-import]

    gen = g2.read_pass(world="g2", month=month, index=index)
    out = clean.astype(np.float64).copy()
    lo, n = gen.power.start_hh, gen.power.power.shape[1]
    out[:, lo : lo + n] = gen.power.power
    return out, list(gen.touched)
"""


def test_g2_core_ignores_the_trailing_type_comment() -> None:
    core = chk.g2_core(G2_TEXT)
    assert core[0] == "import e00714_generator2 as g2"
    assert core[-1] == "out[:, lo : lo + n] = gen.power.power" and len(core) == 5
    assert core == chk.g2_core(G2_TEXT.replace("  # ty: ignore[unresolved-import]", ""))
    assert chk.g2_returns_touched(G2_TEXT)
    assert not chk.g2_returns_touched(G2_TEXT.replace("list(gen.touched)", "gen.truth"))


def test_interval_coverage() -> None:
    import e00726_readings as rd

    got = rd.interval_coverage(
        lower=np.array([0.1, 0.1, 0.1]),
        upper=np.array([0.5, 0.5, 0.5]),
        actual=np.array([0.3, 0.6, np.nan]),
    )
    assert got[0] == 1.0 and got[1] == 0.0 and np.isnan(got[2])


def clip_rows(*, month_scale: float) -> list[dict]:
    return [
        {"depth": 0.5, "sign": 1, "group": "six_signed_mw", "n_half_hours": int(100 * month_scale),
         "beyond_clip": 0.0, "out_of_range": 0.5},
        {"depth": 0.15, "sign": 1, "group": "six_signed_mw", "n_half_hours": 100,
         "beyond_clip": 0.0, "out_of_range": 0.0},
    ]  # fmt: skip


def test_clip_by_cell_weights_months_by_half_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.setattr(com, "data_root", lambda: tmp_path)
    (tmp_path / "diagnostics").mkdir()
    rows_a = clip_rows(month_scale=1.0)
    rows_b = clip_rows(month_scale=3.0)
    rows_b[0]["out_of_range"] = 0.1
    (tmp_path / "diagnostics" / "clip_2025-11.json").write_text(json.dumps(rows_a))
    (tmp_path / "diagnostics" / "clip_2026-01.json").write_text(json.dumps(rows_b))
    diag = sc.read_clip_diagnostics(months=["2025-11", "2026-01"])
    cell = sc.clip_by_cell(diagnostics=diag).filter(pl.col("depth") == 0.5).row(0, named=True)
    assert cell["n_half_hours"] == 400
    assert cell["out_of_range"] == pytest.approx((0.5 * 100 + 0.1 * 300) / 400)
    with pytest.raises(RuntimeError):
        sc.read_clip_diagnostics(months=["2026-04"])


def test_breakdown_rows_join_the_clip_diagnostics() -> None:
    events = []
    for r in range(com.N_PASSES):
        for p in range(N_SITES):
            events.append(
                {"primary": p, "depth": 0.5, "sign": 1, "length_days": 15, "slice": com.SLICE_AFTER,
                 "month": MONTH, "pass_r": r, "M1|q50": 0.6, "C_host|q50": 0.2}
            )  # fmt: skip
    clip = sc.clip_by_cell(
        diagnostics=pl.DataFrame(clip_rows(month_scale=1.0)).with_columns(month=pl.lit(MONTH))
    )
    rows = sc.breakdown_rows(
        events=pl.DataFrame(events), groups={"six_signed_mw": [0, 1, 2], "eight_mva": [3, 4]},
        months=[MONTH], clip=clip,
    )  # fmt: skip
    assert len(rows) == 2 * 2 * 2 * 3 * 4
    hit = [
        r for r in rows
        if (r["slice"], r["group"], r["sign"], r["depth"], r["length_days"])
        == (com.SLICE_AFTER, "six_signed_mw", 1, 0.5, 15)
    ]  # fmt: skip
    assert (
        len(hit) == 1
        and hit[0]["D"] == pytest.approx(0.4)
        and hit[0]["out_of_range"] == pytest.approx(0.5)
    )
    assert hit[0]["n_events"] == 3 * com.N_PASSES
    empty = next(r for r in rows if r["length_days"] == 3 and r["sign"] == 1 and r["depth"] == 0.5)
    assert empty["n_events"] == 0 and np.isnan(empty["D"])
    other = next(r for r in rows if r["group"] == "eight_mva" and r["sign"] == -1)
    assert np.isnan(other["out_of_range"])

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

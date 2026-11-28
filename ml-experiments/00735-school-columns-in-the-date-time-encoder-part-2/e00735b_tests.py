# ruff: noqa: E501, PLR0912, PLR0915, PT018, B023
"""Experiment 735 part 2: known-answer tests of the columns, the network, the lag path, and the runners.

README. Run with plain `python`, from the worktree root, with the experiment folders on `PYTHONPATH`
(source `ml-experiments/00524*/env.sh`) and `DATA_PATH_INTERNAL` set:
`python ml-experiments/00735-school-*/e00735b_tests.py`. Every function named `test_*` also runs under
`pytest ml-experiments/00735-school-*/e00735b_tests.py` when the file is named on the command line (the
file name does not match the repository's `python_files` patterns, as part 1's test file does not).
Each test prints PASS or raises. The network tests need `torch` and run on the CPU with synthetic
tensors; they are skipped, with a printed note, when it is not installed. No test fits a network, reads
a month of rows, or writes outside a temporary folder.

The runner `e00735b_k1.py` executes on import, so the attribute test reads its source: every
`module.attribute` it takes from `e00574_fit`, from `e00591_arm`, or from an imported module exists.
"""

import ast
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import e00735_columns as p1  # ty: ignore[unresolved-import]
import e00735b_columns as cols  # ty: ignore[unresolved-import]
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS: list[str] = []
HOME_PREFIX = "/" + "home/jack"
"""Built from two parts, so that this file does not itself hold an absolute path under the maintainer's home folder."""


def passed(name: str) -> None:
    """Record and print one passing test."""
    RESULTS.append(name)
    print(f"PASS {name}", flush=True)


def stamp_us(*, year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """Return a UTC instant in microseconds since the epoch."""
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()) * 1_000_000


def block_of(
    *, arm: str, stamps: Sequence[Any] | np.ndarray, primary: Sequence[Any] | np.ndarray
) -> np.ndarray:
    """Return an arm's columns for stamps and Primaries."""
    c, cp = cols.coast_arrays()
    return cols.extra_columns(
        valid_us=np.asarray(stamps, dtype=np.int64), primary=np.asarray(primary, dtype=np.int64),
        arm=cols.ARMS[arm], c_true=c, c_permuted=cp,
    )  # fmt: skip


# ----------------------------------------------------------------------------- imports


def _top_level_names(*, path: Path) -> set[str]:
    """Return the names that a module's top level defines, from its source (without importing it)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def _attribute_chains(*, tree: ast.AST) -> list[list[str]]:
    """Return every loaded attribute chain `a.b.c` that starts at a plain name, as `["a", "b", "c"]`."""
    chains = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            chain, cur = [node.attr], node.value
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chains.append([cur.id, *reversed(chain)])
    return chains


def test_every_attribute_exists() -> None:
    """Every `module.attribute` that an experiment-735-part-2 script takes from an imported module exists."""
    fit_path = next(REPO.glob("ml-experiments/00574-*/e00574_fit.py"))
    arm_path = next(REPO.glob("ml-experiments/00591-*/e00591_arm.py"))
    fit_names, arm_names = _top_level_names(path=fit_path), _top_level_names(path=arm_path)
    fit_aliases = {  # the imports of experiment 574's module, which the runner reaches as `e`
        node.names[0].asname or node.names[0].name: node.names[0].name
        for node in ast.walk(ast.parse(fit_path.read_text()))
        if isinstance(node, ast.Import)
    }
    local = {  # names that `e00735b_stage1.pretrain_arm` binds to attributes of `e`
        "c": "e00510_common", "k": "e00511_common", "r448": "e00448_run", "d560": "e00560_data",
        "lg": "e00572_lags",
    }  # fmt: skip
    checked = 0
    for path in sorted(HERE.glob("e00735b_*.py")):
        if path.name == Path(__file__).name:
            continue
        tree = ast.parse(path.read_text())
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    aliases[a.asname or a.name.split(".")[0]] = a.name
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mod = importlib.import_module(node.module)
                for a in node.names:
                    assert hasattr(mod, a.name), (path.name, node.module, a.name)
                    checked += 1
        is_runner = path.name == "e00735b_k1.py"
        for chain in _attribute_chains(tree=tree):
            base, rest = chain[0], chain[1:]
            if base == "e" and path.name in {"e00735b_k1.py", "e00735b_stage1.py"}:
                assert rest[0] in fit_names, (path.name, "e00574_fit has no", rest[0])
                checked += 1
                if rest[0] in fit_aliases and len(rest) > 1:
                    mod = importlib.import_module(fit_aliases[rest[0]])
                    assert hasattr(mod, rest[1]), (path.name, fit_aliases[rest[0]], rest[1])
                    checked += 1
            elif base == "arm" and is_runner:
                assert rest[0] in arm_names, (path.name, "e00591_arm has no", rest[0])
                checked += 1
            elif base in local and path.name == "e00735b_stage1.py":
                assert hasattr(importlib.import_module(local[base]), rest[0]), (
                    path.name,
                    base,
                    rest[0],
                )
                checked += 1
            elif base in aliases:
                name = aliases[base]
                try:
                    mod = importlib.import_module(name)
                except ModuleNotFoundError:
                    print(f"  skipped {path.name}: {name} is not importable here")
                    continue
                obj = mod
                for attr in rest[:2]:
                    if not hasattr(obj, attr):
                        break
                    obj = getattr(obj, attr)
                    checked += 1
                else:
                    continue
                assert hasattr(obj, attr), (path.name, name, rest)
    assert checked > 100, checked
    passed(f"every attribute taken from an imported module exists ({checked} checked)")


# ----------------------------------------------------------------------------- calendar


def test_calendar_covers_stage_one() -> None:
    """The calendar covers every valid time from 2019-12-01, also read 21 days earlier, and has sane term gaps."""
    cal = cols.calendar()
    first = p1.code_of(day=cols.FIRST_PRETRAIN_DATE)
    assert cal.first_code <= first - cols.PLACEBO_SHIFT_DAYS, (cal.first_code, first)
    assert cal.last_code >= p1.code_of(day=date(2026, 7, 31)), cal.last_code
    table = cols.terms()
    p1.validate_terms(table=table)  # ordered, no overlap, weekdays, 4 to 8 weeks, gaps
    assert table["first_day"].min() == date(2019, 9, 3)
    blocks = cols.blocks()
    in_2019_21 = [b for b in blocks if date(2019, 10, 1) <= b.first <= date(2021, 8, 1)]
    assert len(in_2019_21) == 12, len(in_2019_21)
    for b in in_2019_21:
        if b.first.month in {12, 4} or (b.first.month == 3 and b.first.day >= 28):
            assert 16 <= b.days <= 19, (b.first, b.days)  # Christmas and Easter
        elif b.first.month == 7:
            assert 43 <= b.days <= 51, (b.first, b.days)  # summer
        else:
            assert 7 <= b.days <= 10, (b.first, b.days)  # a half-term week
    # a known-answer date of each kind: Christmas 2019 holiday, a term day, the 2020 Easter block
    s = block_of(arm="SKE", stamps=[stamp_us(year=2019, month=12, day=25, hour=12),
                                    stamp_us(year=2020, month=1, day=14, hour=12),
                                    stamp_us(year=2020, month=4, day=8, hour=12)], primary=[0, 0, 0])  # fmt: skip
    assert s[:, 0].tolist() == [1.0, 0.0, 1.0], s[:, 0]
    assert s[1, 0:3].sum() == 0.0, (
        "a term day has no school column"
    )  # 2020-01-14 is a Tuesday in term 3
    passed(
        "the calendar covers 2019-12-01 minus 21 days, its terms validate, and its blocks have plausible lengths"
    )


def test_raises_outside_calendar() -> None:
    """A valid time outside the calendar raises `ValueError`, and `grid_table` marks it NaN."""
    c, cp = cols.coast_arrays()
    before = np.asarray([stamp_us(year=2019, month=8, day=1)])
    after = np.asarray([stamp_us(year=2026, month=12, day=15)])
    for stamps in (before, after):
        try:
            cols.extra_columns(
                valid_us=stamps,
                primary=np.asarray([0]),
                arm=cols.ARMS["SKE"],
                c_true=c,
                c_permuted=cp,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("a time outside the calendar did not raise")
    early = np.asarray(
        [stamp_us(year=2019, month=9, day=20, hour=12)]
    )  # inside, but 21 days earlier is not
    cols.extra_columns(
        valid_us=early, primary=np.asarray([0]), arm=cols.ARMS["SKE"], c_true=c, c_permuted=cp
    )
    try:
        cols.extra_columns(
            valid_us=early,
            primary=np.asarray([0]),
            arm=cols.ARMS["SKE-S21"],
            c_true=c,
            c_permuted=cp,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a shifted read before the calendar did not raise")
    table = cols.grid_table(
        arm=cols.ARMS["SKE"], first_us=stamp_us(year=2019, month=8, day=30), step_us=1_800_000_000 * 48,
        n_grid=6, c_true=c, c_permuted=cp,
    )  # fmt: skip
    assert np.isnan(table[0, 0]).all() and np.isfinite(table[0, 5]).all(), table[0, :, 0]
    passed("a valid time outside the calendar raises, and the lag table marks it NaN")


# ----------------------------------------------------------------------------- columns


def test_columns_depend_only_on_valid_time_and_primary() -> None:
    """The columns are a function of the valid time and the Primary: no other argument, and row order is irrelevant."""
    parameters = set(inspect.signature(cols.extra_columns).parameters)
    assert parameters == {"valid_us", "primary", "arm", "c_true", "c_permuted"}, parameters
    rng = np.random.default_rng(1)
    first = stamp_us(year=2020, month=1, day=1)
    stamps = first + rng.integers(0, 2200 * 48, size=3000) * 1_800_000_000
    primary = rng.integers(0, 16, size=3000)
    for arm in ("SKE", "SKE-S21", "SKE-CP"):
        a = block_of(arm=arm, stamps=list(stamps), primary=list(primary))
        order = rng.permutation(3000)
        b = block_of(arm=arm, stamps=list(stamps[order]), primary=list(primary[order]))
        assert np.array_equal(a[order], b), arm
        twice = block_of(
            arm=arm, stamps=[*stamps[:50], *stamps[:50]], primary=[*primary[:50], *primary[:50]]
        )
        assert np.array_equal(twice[:50], twice[50:]), arm
        assert a.dtype == np.float32 and a.shape == (3000, 8), (a.dtype, a.shape)
    none = block_of(arm="R1", stamps=list(stamps[:5]), primary=list(primary[:5]))
    assert none.shape == (5, 0)
    passed(
        "the 8 columns depend only on the valid time and the Primary (no other argument; row order does not matter)"
    )


def test_shift_and_permutation_definitions() -> None:
    """SKE-S21 shifts S1 to S3, K2, and K3 by 21 days and nothing else; SKE-CP permutes `c` in K1 to K5 only."""
    c, cp = cols.coast_arrays()
    rng = np.random.default_rng(2)
    first = stamp_us(year=2021, month=1, day=1, hour=12)
    stamps = (
        first + rng.integers(0, 1900, size=4000) * 86_400_000_000
    )  # midday, clear of clock changes
    primary = rng.integers(0, 16, size=4000)
    ske = block_of(arm="SKE", stamps=list(stamps), primary=list(primary))
    s21 = block_of(arm="SKE-S21", stamps=list(stamps), primary=list(primary))
    cpa = block_of(arm="SKE-CP", stamps=list(stamps), primary=list(primary))
    earlier = block_of(arm="SKE", stamps=list(stamps - 21 * 86_400_000_000), primary=list(primary))
    # the shifted arm's S1 to S3 are SKE's at the date 21 days earlier; its K2 and K3 use those S1 and S2
    assert np.array_equal(s21[:, 0:3], earlier[:, 0:3])
    assert np.array_equal(s21[:, 4], c[primary].astype(np.float32) * s21[:, 0])
    assert np.array_equal(s21[:, 5], c[primary].astype(np.float32) * s21[:, 1])
    # K1, K4, and K5 are the real ones
    for j in (3, 6, 7):
        assert np.array_equal(s21[:, j], ske[:, j]), j
    assert (s21[:, 0:3] != ske[:, 0:3]).any(axis=1).mean() > 0.05
    # the permuted arm's school columns are real; its K columns are the permuted `c` times the real factors
    assert np.array_equal(cpa[:, 0:3], ske[:, 0:3])
    factor = np.stack([np.ones(4000), ske[:, 0], ske[:, 1], 0 * ske[:, 0], 0 * ske[:, 0]], axis=1)[
        :, :3
    ]
    assert np.array_equal(cpa[:, 3], cp[primary]), "K1 is the permuted closeness"
    assert np.array_equal(cpa[:, 4], cp[primary].astype(np.float32) * ske[:, 0])
    assert np.array_equal(cpa[:, 5], cp[primary].astype(np.float32) * ske[:, 1])
    for j in (6, 7):  # the flags are the same dates: the ratio to `c` matches
        ratio_true = np.where(ske[:, j] != 0, ske[:, j] / c[primary], 0.0)
        ratio_perm = np.where(cpa[:, j] != 0, cpa[:, j] / cp[primary], 0.0)
        assert np.allclose(ratio_true, ratio_perm, atol=1e-5), j
    del factor
    perm = p1.derange(c=c.astype(np.float64))
    assert np.array_equal(cp, c[perm]), "the permutation is part 1's (seed 735)"
    assert (cp != c).all()
    passed(
        "SKE-S21 shifts S1 to S3, K2, and K3 by 21 days (K1, K4, K5 stay real); SKE-CP permutes `c` in K1 to K5 (part 1's permutation)"
    )


def test_stamp_minus_15_minutes_convention() -> None:
    """K4 and K5 belong to the local date of the stamp minus 15 minutes, so the 00:00 stamp is the day before."""
    c, _ = cols.coast_arrays()
    p = 3
    # Thursday 2026-01-01 is a bank holiday, and a winter date has no clock offset: the stamp
    # 2026-01-01 00:00 ends 12-31, the stamp 00:30 begins 01-01, and the stamp 2026-01-02 00:00 still belongs to 01-01.
    k4 = block_of(
        arm="SKE",
        stamps=[stamp_us(year=2026, month=1, day=1, hour=0), stamp_us(year=2026, month=1, day=1, hour=0, minute=30),
                stamp_us(year=2026, month=1, day=2, hour=0), stamp_us(year=2026, month=1, day=2, hour=0, minute=30)],
        primary=[p] * 4,
    )[:, 6]  # fmt: skip
    assert k4.tolist() == [0.0, c[p], c[p], 0.0], k4
    # Saturday 2026-02-14 to Sunday 02-15: the Saturday 00:00 stamp is Friday, the Monday 00:00 stamp is Sunday
    k5 = block_of(
        arm="SKE",
        stamps=[stamp_us(year=2026, month=2, day=14, hour=0), stamp_us(year=2026, month=2, day=14, hour=0, minute=30),
                stamp_us(year=2026, month=2, day=16, hour=0), stamp_us(year=2026, month=2, day=16, hour=0, minute=30)],
        primary=[p] * 4,
    )[:, 7]  # fmt: skip
    assert k5.tolist() == [0.0, c[p], c[p], 0.0], k5
    # the same convention holds across a clock change: stamp 23:30 UTC on 2025-10-25 begins Sunday 10-26 (BST)
    k5_dst = block_of(
        arm="SKE",
        stamps=[stamp_us(year=2025, month=10, day=25, hour=23, minute=0), stamp_us(year=2025, month=10, day=25, hour=23, minute=30)],
        primary=[p] * 2,
    )[:, 7]  # fmt: skip
    assert k5_dst.tolist() == [c[p], c[p]], (
        k5_dst
    )  # 23:00 UTC ends Saturday 10-25 BST (00:00 BST on 10-26 minus 15 min is Saturday)
    passed(
        "K4 and K5 use the stamp-minus-15-minutes local date (00:00 ends the day before; 00:30 begins the day)"
    )


def test_lag_table_equals_row_path() -> None:
    """A lag row's columns (the table at the lag's grid index) equal the row path's columns at the same valid time and Primary."""
    c, cp = cols.coast_arrays()
    first_us, step_us = stamp_us(year=2019, month=9, day=1), 1_800_000_000
    n_grid = 62_000  # to 2022-03; the calendar covers it
    rng = np.random.default_rng(3)
    idx = rng.integers(
        7000, n_grid, size=5000
    )  # from 2020-01 on, so that the 21-day shift is covered
    primary = rng.integers(0, 16, size=5000)
    for name in ("SKE", "SKE-S21", "SKE-CP"):
        arm = cols.ARMS[name]
        table = cols.grid_table(
            arm=arm, first_us=first_us, step_us=step_us, n_grid=n_grid, c_true=c, c_permuted=cp
        )
        assert table.shape == (16, n_grid, 8) and table.dtype == np.float32
        row_path = cols.extra_columns(
            valid_us=first_us + idx.astype(np.int64) * step_us, primary=primary, arm=arm, c_true=c, c_permuted=cp
        )  # fmt: skip
        assert np.array_equal(table[primary, idx], row_path), name
    passed(
        "the lag path's table equals the row path's columns at the same valid time and Primary (SKE, SKE-S21, SKE-CP; exact)"
    )


def test_check_block_catches_silent_columns() -> None:
    """`check_block` raises on a zero K1, a NaN, a constant school column, and a wrong shape or dtype."""
    c, cp = cols.coast_arrays()
    arm = cols.ARMS["SKE"]
    first = stamp_us(year=2023, month=9, day=1)
    stamps = first + np.arange(0, 600 * 48, 7) * 1_800_000_000
    primary = np.arange(len(stamps)) % 16
    block = cols.extra_columns(valid_us=stamps, primary=primary, arm=arm, c_true=c, c_permuted=cp)
    cols.check_block(arm=arm, block=block, valid_us=stamps, kind="train")
    for name, bad in (
        ("zero K1", np.where(np.arange(8) == 3, 0.0, block).astype(np.float32)),
        ("NaN", np.where(np.arange(8) == 5, np.nan, block).astype(np.float32)),
        ("constant S2", np.where(np.arange(8) == 1, 0.0, block).astype(np.float32)),
        ("float64", block.astype(np.float64)),
        ("wrong width", block[:, :7]),
    ):
        try:
            cols.check_block(arm=arm, block=bad, valid_us=stamps, kind="train")
        except AssertionError:
            continue
        raise AssertionError(f"check_block accepted a block with {name}")
    passed(
        "check_block catches a zero K1, a NaN, a constant school column, a float64 block, and a wrong width"
    )


# ----------------------------------------------------------------------------- network


def _torch_or_skip():  # noqa: ANN202
    try:
        import torch  # ty: ignore[unresolved-import]
    except ModuleNotFoundError:
        print("SKIP network tests: torch is not installed")
        return None
    return torch


def test_r1_equals_677_network() -> None:
    """R1's network (no new column) gives experiment 677's `AuxLagMdn` outputs on the same weights, labels attached or not."""
    torch = _torch_or_skip()
    if torch is None:
        return
    import e00591_model as aux  # ty: ignore[unresolved-import]
    import e00735b_model as model  # ty: ignore[unresolved-import]

    n_b1, n_knots, n_primaries, rows = 14, 6, 16, 600
    for stage in ("pre", "ft"):
        torch.manual_seed(0)
        theirs = aux.AuxLagMdn(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, kind="shared", detector=False
        )  # fmt: skip
        torch.manual_seed(0)
        mine = model.make_network(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, n_new=0
        )
        assert list(mine.state_dict()) == list(theirs.state_dict())
        assert all(
            torch.equal(a, b)
            for a, b in zip(mine.state_dict().values(), theirs.state_dict().values(), strict=True)
        ), "R1 and 677's network draw the same initial weights from the same seed"
        mine.load_state_dict(theirs.state_dict())
        with (
            torch.no_grad()
        ):  # random weights in every layer, so that equality is not an accident of `init_head`
            gen = torch.Generator().manual_seed(5)
            for p in theirs.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
        mine.load_state_dict(theirs.state_dict())
        rng = np.random.default_rng(0)
        x = rng.normal(size=(rows, 60)).astype(np.float32)
        x[:, 43:45] = rng.integers(-1, 400, size=(rows, 2))
        table = torch.from_numpy(rng.normal(size=(n_primaries, 400, 28)).astype(np.float32))
        theirs.table = mine.table = table
        primary = torch.from_numpy(rng.integers(0, n_primaries, size=rows))
        labels = rng.normal(size=(rows, 2)).astype(np.float32)
        for net in (theirs, mine):
            net.eval()
        for with_labels in (False, True):
            xin = np.concatenate([x, labels], axis=1) if with_labels and stage == "ft" else x
            a = theirs(torch.from_numpy(xin), primary).detach()
            b = mine(torch.from_numpy(xin), primary).detach()
            assert torch.equal(a, b), (stage, with_labels, float((a - b).abs().max()))
        assert float(a.std()) > 0.05
        assert mine.label_col == theirs.label_col == 60
    passed(
        "R1's network equals experiment 677's AuxLagMdn: same keys, same seeded initial weights, equal outputs on the same weights (stages pre and ft, labels attached and not)"
    )


def _ske_network(*, torch, stage: str = "ft", seed: int = 0):  # noqa: ANN001, ANN202
    import e00735b_model as model  # ty: ignore[unresolved-import]

    torch.manual_seed(seed)
    return model.make_network(n_primaries=16, stage=stage, n_knots=6, n_b1=14, n_new=8)


def test_columns_reach_the_encoder_in_row_and_lag_paths() -> None:
    """The date-time encoder's input holds the 8 columns at the row's and at each lag's own time; they change the output."""
    torch = _torch_or_skip()
    if torch is None:
        return
    c, cp = cols.coast_arrays()
    arm = cols.ARMS["SKE"]
    first_us, step_us, n_grid, rows = (
        stamp_us(year=2019, month=9, day=1),
        1_800_000_000,
        62_000,
        400,
    )
    rng = np.random.default_rng(4)
    target = rng.integers(20_000, n_grid, size=rows)
    primary = rng.integers(0, 16, size=rows)
    day, week = target - 48, target - 336
    day[:20] = -1
    x = rng.normal(size=(rows, 60)).astype(np.float32)
    x[:, 43], x[:, 44] = day, week
    block = cols.extra_columns(
        valid_us=first_us + target.astype(np.int64) * step_us,
        primary=primary,
        arm=arm,
        c_true=c,
        c_permuted=cp,
    )
    xin = torch.from_numpy(np.concatenate([x, block], axis=1))
    table_new = torch.from_numpy(
        cols.grid_table(
            arm=arm, first_us=first_us, step_us=step_us, n_grid=n_grid, c_true=c, c_permuted=cp
        )
    )
    labels = torch.from_numpy(rng.normal(size=(rows, 2)).astype(np.float32))
    for stage in ("pre", "ft"):
        net = _ske_network(torch=torch, stage=stage)
        with (
            torch.no_grad()
        ):  # random weights everywhere, as the freshly initialised head may output a constant
            gen = torch.Generator().manual_seed(6)
            for p in net.parameters():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
        assert net.datetime[0].in_features == 21 + 8 + 8 and net.label_col == 68
        net.table = torch.zeros(16, n_grid, 28)
        net.table_new = table_new
        net.eval()
        seen: list = []
        net.datetime.register_forward_pre_hook(
            lambda _m, args: seen.append(args[0].detach().clone())
        )
        pid = torch.from_numpy(primary)
        out = net(xin, pid).detach()
        assert len(seen) == 3
        row_in, day_in, week_in = (s.numpy() for s in seen)
        assert np.array_equal(row_in[:, 21:29], block), "row path"
        for path_in, idx in ((day_in, day), (week_in, week)):
            valid = first_us + np.where(idx >= 0, idx, target).astype(np.int64) * step_us
            want = cols.extra_columns(
                valid_us=valid, primary=primary, arm=arm, c_true=c, c_permuted=cp
            )
            want[idx < 0] = 0.0
            assert np.array_equal(path_in[:, 21:29], want), "lag path"
        assert (day_in[:20, 21:29] == 0).all(), "a missing lag reads zeros"
        if stage == "ft":
            with_labels = net(torch.cat([xin, labels], dim=1), pid).detach()
            assert torch.equal(with_labels, out), (
                "attached labels do not change an evaluation-mode output"
            )
        # the row columns move the output
        changed = xin.clone()
        changed[:, 60:68] = 1.0 - changed[:, 60:68]
        assert not torch.allclose(net(changed, pid).detach(), out, atol=1e-5), (
            "row columns have no effect"
        )
        # the lag columns move the output
        net.table_new = 1.0 - table_new
        assert not torch.allclose(net(xin, pid).detach(), out, atol=1e-5), (
            "lag columns have no effect"
        )
        net.table_new = None
        try:
            net(xin, pid)
        except AssertionError:
            pass
        else:
            raise AssertionError("a network with 8 columns ran without its lag table")
    passed(
        "the date-time encoder's input holds the 8 columns at the row and at each lag's own time (zeros for a missing lag), and each path changes the output"
    )


def test_paired_initialisation_and_state_dict() -> None:
    """SKE's initial weights equal R1's except the 8 new first-layer columns, which do not depend on the seed."""
    torch = _torch_or_skip()
    if torch is None:
        return
    import e00735b_model as model  # ty: ignore[unresolved-import]

    r1 = model.make_network(
        n_primaries=16, stage="pre", n_knots=6, n_b1=14, n_new=0
    )  # unseeded, then seeded below
    for seed in (0, 1):
        torch.manual_seed(seed)
        r1 = model.make_network(n_primaries=16, stage="pre", n_knots=6, n_b1=14, n_new=0)
        ske = _ske_network(torch=torch, stage="pre", seed=seed)
        a, b = r1.state_dict(), ske.state_dict()
        assert list(a) == list(b)
        for key in a:
            if key == "datetime.0.weight":
                assert b[key].shape == (64, 37) and a[key].shape == (64, 29)
                assert torch.equal(b[key][:, :21], a[key][:, :21]) and torch.equal(
                    b[key][:, 29:], a[key][:, 21:]
                )
            else:
                assert torch.equal(a[key], b[key]), key
        assert float(b["datetime.0.weight"][:, 21:29].abs().max()) <= 1.0 / np.sqrt(37) + 1e-6
    w0 = _ske_network(torch=torch, stage="pre", seed=0).state_dict()["datetime.0.weight"][:, 21:29]
    w1 = _ske_network(torch=torch, stage="pre", seed=1).state_dict()["datetime.0.weight"][:, 21:29]
    assert torch.equal(w0, w1), "the new columns' initial weights are the same for every seed"
    # a stage-1 state loads into the stage-2 network: every key but `mixer_ft` and the knots
    pre = _ske_network(torch=torch, stage="pre", seed=0)
    ft = _ske_network(torch=torch, stage="ft", seed=0)
    raw = {
        k: v for k, v in pre.state_dict().items() if k != "knots" and not k.startswith("mixer_ft.")
    }
    missing = ft.load_state_dict(raw, strict=False)
    assert all(
        k.startswith("mixer_ft.") or k in {"knots_old", "knot_new"} for k in missing.missing_keys
    ), missing
    assert not missing.unexpected_keys
    passed(
        "SKE starts from R1's initial weights plus 8 new first-layer columns that do not depend on the seed; stage-1 weights load into stage 2"
    )


def test_stage_one_matrix_layout() -> None:
    """The 8 columns sit at input index 60 to 67 in the pre-training matrix, the row, and the network, and the lags are at 43 and 44."""
    import e00574_model as mod  # ty: ignore[unresolved-import]
    import e00735b_stage1 as stage1  # ty: ignore[unresolved-import]

    assert mod.b1_cols(n_b1=14).stop == 60 and mod.TAU + 1 + 14 == 60
    first_new = mod.b1_cols(n_b1=14).stop
    torch = _torch_or_skip()
    if torch is not None:
        net = _ske_network(torch=torch)
        assert net.new_cols == slice(first_new, first_new + 8) and net.label_col == first_new + 8
    c, cp = cols.coast_arrays()
    tables = stage1.NewColumnTables(
        arm=cols.ARMS["SKE"], first_us=stamp_us(year=2019, month=9, day=1), step_us=1_800_000_000, device="cpu",
        c_true=c, c_permuted=cp,
    )  # fmt: skip
    if torch is not None:
        t = tables.get(n_grid=20_000)
        assert t.shape == (16, 20_000, 8) and tables.get(n_grid=20_000) is t
    passed(
        "the 8 columns sit at input index 60 to 67 (after the B1 columns), the labels after them"
    )


def test_history_follows_the_stopping_rule() -> None:
    """`History` stores the epoch that experiment 574's early-stopping rule stores (best strictly below by 1e-5)."""
    torch = _torch_or_skip()
    if torch is None:
        return
    import e00735b_stage1 as stage1  # ty: ignore[unresolved-import]

    hist = stage1.History()
    val = [1.0, 0.9, 0.9000001, 0.8, 0.8, 0.85]
    for v in val:
        hist.add(
            nll=torch.tensor([2.0, 4.0], requires_grad=True)
        )  # a training batch (gradients on)
        with torch.no_grad():
            hist.add(nll=torch.tensor([v, v]))
    rows, stored = hist.finish()
    assert len(rows) == 6 and stored == 4, (len(rows), stored)
    assert (
        abs(rows[3]["val_nll"] - 0.8) < 1e-6 and abs(rows[0]["train_nll_unweighted"] - 3.0) < 1e-9
    )
    passed(
        "History records one row per epoch and stores the epoch of experiment 574's early-stopping rule"
    )


# ----------------------------------------------------------------------------- scorer, phases, shell


def test_period_of_and_verdict() -> None:
    """The 8-month `period_of` excludes July 2026 valid dates; the verdict maps readings to the plan's rules."""
    os.environ.setdefault(
        "R735", str(Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / cols.NAME)
    )
    import e00735b_score as score  # ty: ignore[unresolved-import]

    s1 = score.s1
    assert s1.MONTHS == score.MONTHS and len(s1.MONTHS) == 8 and s1.GATE_MONTH == "2026-03"
    m = np.asarray([0, 7, 7, 3])
    v = np.asarray([20_000, s1.HOLDOUT_CODE - 1, s1.HOLDOUT_CODE, 20_100])
    assert s1.period_of(m=m, vday=v).tolist() == [
        "held-in",
        "held-in",
        "july to september",
        "held-in",
    ]
    assert abs(score.FAMILY_LEVEL - 0.99) < 1e-12

    def reading(*, est: float, lo: float, hi: float, months: int = 8, fits: int = 24) -> dict:
        return {
            "estimate": est,
            "lower": lo,
            "upper": hi,
            "months_negative": months,
            "fits_negative": fits,
        }

    gain = reading(est=-0.05, lo=-0.09, hi=-0.01)
    null = reading(est=0.0, lo=-0.02, hi=0.02)
    harm = reading(est=0.05, lo=0.02, hi=0.09)
    small = reading(est=-0.01, lo=-0.02, hi=-0.001)
    bound_ok = {"estimate": 0.0, "lower": float("nan"), "upper": 0.01}
    bound_bad = {"estimate": 0.0, "lower": float("nan"), "upper": 0.05}

    def readings(**over: dict) -> dict:
        base = dict.fromkeys(score.REQUIRED_READINGS, null)
        base["year held-in SKE six"] = base["year held-in SKE eight"] = bound_ok
        return {**base, **over}

    assert score.verdict(r=readings())["verdict"] == "not supported"
    full = readings(
        **{"C1 six": gain, "C2 six": gain, "C1 eight": gain, "C2 eight": gain, "C3": gain}
    )
    assert score.verdict(r=full)["verdict"] == "supported"
    school_only = readings(**{"C1 six": gain, "C2 six": gain})
    out = score.verdict(r=school_only)
    assert (
        out["verdict"] == "partly supported"
        and out["school_claim_six"]
        and not out["school_claim_eight"]
    ), out
    coast_only = readings(C3=gain)
    assert score.verdict(r=coast_only)["verdict"] == "partly supported"
    assert (
        score.verdict(r=readings(C3=gain, **{"CP-R1": gain}))["coast_claim"] is False
    )  # a failed placebo
    assert score.verdict(r=readings(C3=gain, **{"inland C3": gain}))["coast_claim"] is False
    assert (
        score.verdict(r=readings(C3=gain, **{"year held-in SKE eight": bound_bad}))["coast_claim"]
        is False
    )
    assert score.verdict(r=readings(**{"C1 six": harm, "C2 six": harm}))["verdict"] == "refuted"
    assert (
        score.verdict(r=readings(**{"C1 six": harm}))["verdict"] != "refuted"
    )  # needs both at the same group
    assert score.verdict(r=readings(**{"C1 six": small}))["verdict"] == "small gain"
    assert (
        score.verdict(r=readings(C3=reading(est=0.2, lo=-0.3, hi=0.7)))["verdict"] == "inconclusive"
    )
    try:
        score.verdict(r={})
    except ValueError:
        pass
    else:
        raise AssertionError("a missing reading did not raise")
    passed(
        "period_of excludes July 2026 valid dates in the eighth month; the verdict maps readings to the plan's rules 1 to 6"
    )


def test_failure_record_stores_traceback() -> None:
    """A failed stage's record holds the exception message and the traceback, and a failed chain exits with 20, not the busy code 75."""
    import e00735b_phases as phases  # ty: ignore[unresolved-import]

    with tempfile.TemporaryDirectory(prefix="e735b_") as tmp:
        root = Path(tmp) / "00735-smoke"
        (root / "logs").mkdir(parents=True)
        (root / "logs" / "2026-01_SKE_step2.log").write_text(
            'some output\nTraceback (most recent call last):\n  File "x.py", line 1, in <module>\nAssertionError: lag index off\n'
        )
        path = phases.failure_record(root=root, arm="SKE", month="2026-01", stage="step2", code=20)
        record = json.loads(path.read_text())
        assert record["exception_message"] == "AssertionError: lag index off", record
        assert (
            "Traceback (most recent call last)" in record["traceback"]
            and "line 1" in record["traceback"]
        )
        assert record["exit_code"] == 20 and record["arm"] == "SKE"
    assert (
        phases.EXIT_BUSY == 75
        and phases.EXIT_FAILED == 20
        and phases.EXIT_BUSY != phases.EXIT_FAILED
    )
    assert (
        len(
            {
                phases.EXIT_BUSY,
                phases.EXIT_FAILED,
                phases.EXIT_STOP,
                phases.EXIT_USAGE,
                phases.EXIT_GATE_FAILED,
            }
        )
        == 5
    )
    shell = (HERE / "e00735b_run.sh").read_text()
    assert "EXIT_BUSY=75" in shell and "return 20" in shell and "return $EXIT_BUSY" in shell
    passed(
        "the failure record stores the exception message and the traceback; a failed chain exits with 20, a busy GPU with 75"
    )


def test_phase_runner_markers_and_stop_line() -> None:
    """The phase runner refuses a phase with no review marker, stops at 6.5 GPU-process-hours, and needs the gate before `fit`."""
    import e00735b_phases as phases  # ty: ignore[unresolved-import]

    assert phases.STOP_HOURS == 6.5 or "STOP_HOURS" in os.environ, phases.STOP_HOURS
    with tempfile.TemporaryDirectory(prefix="e735b_") as tmp:
        root = Path(tmp) / "00735-smoke"
        for phase in phases.PHASES:
            env = {
                **os.environ,
                "R735": str(root),
                "DATA_PATH_INTERNAL": os.environ["DATA_PATH_INTERNAL"],
            }
            code = subprocess.call(
                [sys.executable, str(HERE / "e00735b_phases.py"), phase],
                env=env,
                stdout=subprocess.DEVNULL,
            )
            assert code == 2, (phase, code)  # no marker: nothing starts
        (root / "review_markers").mkdir(parents=True)
        (root / "review_markers" / "fit").write_text("ok")
        code = subprocess.call(
            [sys.executable, str(HERE / "e00735b_phases.py"), "fit"],
            env=env,
            stdout=subprocess.DEVNULL,
        )
        assert code == 2, "fit without a passed replay gate must refuse"
        (root / "scores").mkdir()
        (root / "scores" / "gate.json").write_text(json.dumps({"passed": True}))
        log = root / "logs" / "phase_times.csv"
        log.parent.mkdir(parents=True)
        log.write_text(
            "arm,month,phase,device,start_utc,end_utc\n"
            "R1,2025-11,fit,cuda,2026-01-01T00:00:00Z,2026-01-01T06:30:00Z\n"
        )
        assert abs(phases.gpu_hours(csv_paths=[log]) - 6.5) < 1e-9
        out = subprocess.run(
            [sys.executable, str(HERE / "e00735b_phases.py"), "fit", "--dry-run"],
            env={**os.environ, "R735": str(root)}, capture_output=True, text=True, check=False,
        )  # fmt: skip
        assert out.returncode == 21 and "over the line" in out.stdout, (
            out.returncode,
            out.stdout[-300:],
        )
    passed(
        "the phase runner refuses a phase without its marker, refuses fit before the gate, and stops at the 6.5 GPU-process-hour line"
    )


def test_run_script_and_hash_lists() -> None:
    """The driver never writes under 677's folder, and the hashed files match between the runner and the driver."""
    shell = (HERE / "e00735b_run.sh").read_text()
    assert (
        '"$R735" == *school-columns-in-the-date-time-encoder-part-2*' in shell
        and '[[ "$R735" != *00677* ]]' in shell
        and '[[ "$R735" != *coast-and-school* ]]' in shell
    )
    assert "R735=$REPO/.claude/scratch/e735b_smoke/00735-smoke" in shell
    assert subprocess.call(["bash", "-n", str(HERE / "e00735b_run.sh")]) == 0
    runner = (HERE / "e00735b_k1.py").read_text()
    assert (
        "root=E00735B_ROOT_OK" in runner
        and "_cols.check_output_root(" in runner
        and "_cols.check_output_root(root=str(ROOT))" in runner
    )
    tree = ast.parse(runner)
    listed: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "HASHED_FILES":
            listed = [
                ast.unparse(elt).split("/")[-1].strip(" ')\"")
                if isinstance(elt, ast.BinOp)
                else "p1:" + ast.unparse(elt)
                for elt in node.value.elts  # ty: ignore[unresolved-attribute]
            ]
    assert listed[:4] == [
        "e00735b_k1.py",
        "e00735b_columns.py",
        "e00735b_model.py",
        "e00735b_stage1.py",
    ], listed
    cat = next(line for line in shell.splitlines() if line.startswith("CODE_HASH="))
    order = [
        cat.index(n)
        for n in (
            "e00735b_k1.py",
            "e00735b_columns.py",
            "e00735b_model.py",
            "e00735b_stage1.py",
            "e00735_columns.py",
            "e00735_diagnostic.py",
        )
    ]
    assert order == sorted(order), cat
    assert len(listed) == 6
    for path in sorted(HERE.glob("e00735b_*")):
        if path.name == Path(__file__).name:
            continue
        assert HOME_PREFIX not in path.read_text(), (
            path.name,
            "names an absolute path under the maintainer's home",
        )
    passed(
        "the driver and runner write only under experiment 735's folder, the code-hash file lists agree, and no file names a home path"
    )


def test_output_root_guards() -> None:
    """Every guard accepts the experiment's own and the smoke folder, and rejects part 1's, 677's, and any other."""
    own = "/x/data/ml-experiments/00735-school-columns-in-the-date-time-encoder-part-2"
    smoke = "/x/.claude/scratch/e735b_smoke/00735-smoke"
    part1 = (
        "/x/data/ml-experiments/00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1"
    )
    e677 = "/x/data/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time"
    cols.check_output_root(root=own)
    cols.check_output_root(root=smoke)
    for bad in (
        part1,
        e677,
        "/x/data/ml-experiments/00735-other",
        f"{own}/00677",
        f"{smoke}/coast-and-school",
    ):
        try:
            cols.check_output_root(root=bad)
        except AssertionError:
            continue
        raise AssertionError(f"check_output_root accepted {bad}")
    import e00735b_phases as phases  # ty: ignore[unresolved-import]

    saved = os.environ.get("R735")
    try:
        for root, ok in ((own, True), (smoke, True), (part1, False), (e677, False)):
            os.environ["R735"] = root
            try:
                phases.root_folder()
            except AssertionError:
                assert not ok, root
            else:
                assert ok, root
            try:
                cols.data_root()
            except AssertionError:
                assert not ok, root
            else:
                assert ok, root
    finally:
        if saved is None:
            os.environ.pop("R735", None)
        else:
            os.environ["R735"] = saved
    for root in (part1, e677):
        env = {**os.environ, "R735": root, "DATA_PATH_INTERNAL": os.environ["DATA_PATH_INTERNAL"]}
        code = subprocess.call(
            ["bash", str(HERE / "e00735b_run.sh"), "R1", "2026-01", "step2"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert code == 2, (root, code)  # a rejected root exits 2 before anything runs
    for name in ("e00735b_k1.py", "e00735b_score.py"):
        text = (HERE / name).read_text()
        assert "check_output_root" in text or "coast-and-school" in text, name
    passed(
        "every output-root guard rejects part 1's and experiment 677's folders and accepts the experiment's own and the smoke folder"
    )


def test_lag_table_check_catches_an_offset() -> None:
    """`check_table_matches_rows` passes on the true table and raises on a one-step offset or a wrong Primary."""
    torch = _torch_or_skip()
    if torch is None:
        return
    import e00735b_stage1 as stage1  # ty: ignore[unresolved-import]

    c, cp = cols.coast_arrays()
    arm = cols.ARMS["SKE"]
    first_us, step_us, n_grid = stamp_us(year=2019, month=9, day=1), 1_800_000_000, 60_000
    table = torch.from_numpy(
        cols.grid_table(
            arm=arm, first_us=first_us, step_us=step_us, n_grid=n_grid, c_true=c, c_permuted=cp
        )
    )
    rng = np.random.default_rng(8)
    idx = rng.integers(20_000, 50_000, size=5000)
    primary = rng.integers(0, 16, size=5000)
    valid = first_us + idx.astype(np.int64) * step_us
    block = cols.extra_columns(valid_us=valid, primary=primary, arm=arm, c_true=c, c_permuted=cp)
    stage1.check_table_matches_rows(
        table=table,
        block=block,
        primary=primary,
        valid_us=valid,
        first_us=first_us,
        step_us=step_us,
    )
    late = np.roll(np.arange(len(table[0])), 1)  # a table one step late
    for name, bad in (
        ("one-step offset", table[:, torch.from_numpy(late)]),
        ("wrong Primary", table.flip(0)),
    ):
        try:
            stage1.check_table_matches_rows(
                table=bad,
                block=block,
                primary=primary,
                valid_us=valid,
                first_us=first_us,
                step_us=step_us,
            )
        except AssertionError:
            continue
        raise AssertionError(f"the check accepted a table with a {name}")
    passed(
        "the lag-table check passes on the true table and raises on a one-step offset and on a wrong Primary"
    )


FAKE_677_LOG = """2025-11: pretraining rows 7,144,058
  pretrain P5 2025-11 s0: 45 epochs, val NLL 0.3144, 54 s, dropout {'share_all_lags_masked': 0.23141833394969638, 'share_day_lag_missing': 0.30076518654823303}
  pretrain P5 2025-11 s1: 60 epochs, val NLL 0.3072, 70 s, dropout {'share_all_lags_masked': 0.242239774649086, 'share_day_lag_missing': 0.2957655191421509}
  pretrain P5 2025-11 s2: 60 epochs, val NLL 0.3186, 70 s, dropout {'share_all_lags_masked': 0.2548102213055941, 'share_day_lag_missing': 0.31151390}
"""


def _fake_mine(
    *, epochs: tuple[int, int, int], nll: tuple[str, str, str], change_dropout: bool = False
) -> str:
    """Return a fake R1 stage-1 log built from the fake 677 log with new epochs and NLLs."""
    lines = []
    for seed, (line, ep, v) in enumerate(
        zip(FAKE_677_LOG.splitlines()[1:], epochs, nll, strict=True)
    ):
        head, tail = line.split(": ", 1)
        tail = tail.split(", dropout ", 1)[1]
        if change_dropout and seed == 1:
            tail = tail.replace("0.2422", "0.2423")
        lines.append(f"{head.replace('P5', 'R1')}: {ep} epochs, val NLL {v}, 50 s, dropout {tail}")
    return "\n".join(lines)


def test_r1_reproduction_check() -> None:
    """The R1 check parses both logs, accepts equality, flags an epoch gap of up to 5 within 3e-3, skips missing seeds, and stops on anything else."""
    import e00735b_phases as phases  # ty: ignore[unresolved-import]

    ref = phases.parse_pretrain_log(text=FAKE_677_LOG)
    assert set(ref) == {0, 1, 2} and ref[0]["epochs"] == 45 and ref[1]["nll"] == 0.3072
    assert ref[0]["dropout"]["share_all_lags_masked"] == 0.23141833394969638
    seeds = [0, 1, 2]
    same = phases.parse_pretrain_log(
        text=_fake_mine(epochs=(45, 60, 60), nll=("0.3144", "0.3072", "0.3186"))
    )
    assert phases.compare_pretrain(mine=same, reference=ref, seeds=seeds)[0] == "ok"
    close = phases.parse_pretrain_log(
        text=_fake_mine(epochs=(45, 60, 60), nll=("0.3150", "0.3079", "0.3190"))
    )
    assert phases.compare_pretrain(mine=close, reference=ref, seeds=seeds)[0] == "ok"  # within 1e-3
    flag = phases.parse_pretrain_log(
        text=_fake_mine(epochs=(46, 60, 60), nll=("0.3154", "0.3072", "0.3186"))
    )
    assert phases.compare_pretrain(mine=flag, reference=ref, seeds=seeds)[0] == "flag"
    for name, text in (
        (
            "NLL off by 4e-3 at equal epochs",
            _fake_mine(epochs=(45, 60, 60), nll=("0.3184", "0.3072", "0.3186")),
        ),
        (
            "one epoch and NLL off by 5e-3",
            _fake_mine(epochs=(46, 60, 60), nll=("0.3194", "0.3072", "0.3186")),
        ),
        ("six epochs", _fake_mine(epochs=(51, 60, 60), nll=("0.3144", "0.3072", "0.3186"))),
        (
            "dropout shares",
            _fake_mine(
                epochs=(45, 60, 60), nll=("0.3144", "0.3072", "0.3186"), change_dropout=True
            ),
        ),
    ):
        status, _ = phases.compare_pretrain(
            mine=phases.parse_pretrain_log(text=text), reference=ref, seeds=seeds
        )
        assert status == "mismatch", name
    gap5 = phases.parse_pretrain_log(
        text=_fake_mine(epochs=(50, 60, 60), nll=("0.3170", "0.3072", "0.3186"))
    )
    assert (
        phases.compare_pretrain(mine=gap5, reference=ref, seeds=seeds)[0] == "flag"
    )  # 5 epochs, NLL within 3e-3
    status, notes = phases.compare_pretrain(mine={0: same[0]}, reference=ref, seeds=seeds)
    assert status == "ok" and sum("skipped" in n for n in notes) == 2, (
        status,
        notes,
    )  # a seed missing from R1's log
    status, notes = phases.compare_pretrain(mine=same, reference={0: ref[0]}, seeds=seeds)
    assert status == "ok" and sum("skipped" in n for n in notes) == 2, (
        status,
        notes,
    )  # a seed missing from 677's log
    nearly = {
        **same,
        0: {**same[0], "dropout": {k: v * (1 + 1e-9) for k, v in same[0]["dropout"].items()}},
    }
    assert (
        phases.compare_pretrain(mine=nearly, reference=ref, seeds=seeds)[0] == "ok"
    )  # math.isclose, relative 1e-6
    with tempfile.TemporaryDirectory(prefix="e735b_") as tmp:
        data = Path(tmp) / "data"
        (data / "ml-experiments" / "00677-fake" / "logs").mkdir(parents=True)
        (data / "ml-experiments" / "00677-fake" / "logs" / "2025-11_5w_pretrain.log").write_text(
            FAKE_677_LOG
        )
        (data / "ml-experiments" / "00677-fake" / "logs" / "2026-02_5w_pretrain.log").write_text(
            "2026-02: stage 1 exists, nothing was trained\n"
        )
        root = Path(tmp) / "00735-smoke"
        (root / "logs").mkdir(parents=True)
        saved = os.environ["DATA_PATH_INTERNAL"]
        os.environ["DATA_PATH_INTERNAL"] = str(data)
        try:
            assert (
                phases.r1_reproduction_check(root=root, month="2026-03") == 0
            )  # no 677 log: skipped
            assert (
                phases.r1_reproduction_check(root=root, month="2025-11") == phases.EXIT_FAILED
            )  # R1's log is missing
            (root / "logs" / "2025-11_R1_pretrain.log").write_text(
                _fake_mine(epochs=(45, 60, 60), nll=("0.3144", "0.3072", "0.3186"))
            )
            assert phases.r1_reproduction_check(root=root, month="2025-11") == 0
            # 677 trained no stage 1 in 2026-02: its log has no line for any seed, so the month is skipped
            (root / "logs" / "2026-02_R1_pretrain.log").write_text(
                "  pretrain R1 2026-02 s0: 12 epochs, val NLL 0.5000, 5 s, dropout {'a': 0.1}\n"
            )
            assert phases.r1_reproduction_check(root=root, month="2026-02") == 0
            # a resumed stage 1 whose log lacks seed 1
            lines = _fake_mine(epochs=(45, 60, 60), nll=("0.3144", "0.3072", "0.3186")).splitlines()
            (root / "logs" / "2025-11_R1_pretrain.log").write_text("\n".join([lines[0], lines[2]]))
            assert phases.r1_reproduction_check(root=root, month="2025-11") == 0
            (root / "logs" / "2025-11_R1_pretrain.log").write_text(
                _fake_mine(
                    epochs=(45, 60, 60), nll=("0.3144", "0.3072", "0.3186"), change_dropout=True
                )
            )
            assert phases.r1_reproduction_check(root=root, month="2025-11") == phases.EXIT_FAILED
        finally:
            os.environ["DATA_PATH_INTERNAL"] = saved
        record = json.loads((root / "failures" / "2025-11_R1_reproduction.json").read_text())
        assert (
            "Traceback (most recent call last)" in record["traceback"]
            and "dropout shares differ" in record["exception_message"]
        ), record
        out = subprocess.run(
            [sys.executable, str(HERE / "e00735b_phases.py"), "fit", "--dry-run"],
            env={**os.environ, "R735": str(root)}, capture_output=True, text=True, check=False,
        )  # fmt: skip
        order = [line for line in out.stdout.splitlines() if line.startswith("dry run:")]
        assert order[0].startswith("dry run: R1 2025-11") and order[1].startswith(
            "dry run: SKE 2025-11"
        ), (out.returncode, order[:3], out.stdout[-300:])
    passed(
        "the R1 reproduction check parses the logs, accepts, flags, and stops as specified, writes the traceback record, and R1 runs first in each month"
    )


def _gate_fixture(*, torch, arm_name: str, rows: int = 900):  # noqa: ANN001, ANN202
    """Return a network, an input matrix, Primaries, a validation mask, and valid times for the gate."""
    import e00735b_model as model_module  # ty: ignore[unresolved-import]

    c, cp = cols.coast_arrays()
    arm = cols.ARMS[arm_name]
    first_us, step_us, n_grid = stamp_us(year=2019, month=9, day=1), 1_800_000_000, 62_000
    rng = np.random.default_rng(9)
    target = rng.integers(20_000, n_grid, size=rows)
    primary = rng.integers(0, 16, size=rows)
    valid = first_us + target.astype(np.int64) * step_us
    x = rng.normal(size=(rows, 60)).astype(np.float32)
    day, week = target - 48, target - 336
    day[:50] = -1
    x[:, 43], x[:, 44] = day, week
    block = cols.extra_columns(valid_us=valid, primary=primary, arm=arm, c_true=c, c_permuted=cp)
    torch.manual_seed(0)
    net = model_module.make_network(n_primaries=16, stage="ft", n_knots=6, n_b1=14, n_new=arm.width)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(6)
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
    net.table = torch.zeros(16, n_grid, 28)
    if arm.columns:
        net.table_new = torch.from_numpy(
            cols.grid_table(
                arm=arm, first_us=first_us, step_us=step_us, n_grid=n_grid, c_true=c, c_permuted=cp
            )
        )
    xin = torch.from_numpy(np.concatenate([x, block], axis=1))
    is_val = torch.from_numpy(rng.random(rows) < 0.3)
    return net, xin, torch.from_numpy(primary), is_val, valid, arm, (first_us, step_us, c, cp)


def test_real_run_stage_gate_and_tags() -> None:
    """`gate_stage` writes its JSON for an arm with the columns, shows none for R1, raises on a wrong table, and the stage tag catches changes."""
    torch = _torch_or_skip()
    if torch is None:
        return
    import e00735b_stage1 as stage1  # ty: ignore[unresolved-import]

    with tempfile.TemporaryDirectory(prefix="e735b_") as tmp:
        out = Path(tmp)
        for name in ("SKE", "SKE-S21", "SKE-CP", "R1"):
            net, x, primary, is_val, valid, arm, (first_us, step_us, c, cp) = _gate_fixture(
                torch=torch, arm_name=name
            )
            if not arm.columns:
                x = x[:, :60]
            for stage in (1, 2, 3):
                payload = stage1.gate_stage(
                    model=net, x=x, primary=primary, is_val=is_val, valid_us=valid, arm=arm, stage=stage,
                    month="2026-03", seed=0, out_dir=out, first_us=first_us, step_us=step_us, c_true=c, c_permuted=cp, n_rows=300,
                )  # fmt: skip
                assert (out / f"checks_stage_{stage}_2026-03_s0.json").exists() or stage != 3
                paths = payload["batches"]["train"]["paths"]
                if arm.columns:
                    assert set(paths) == {"row", "day lag", "week lag"} and all(
                        p["equals_expected"] for p in paths.values()
                    )
                    assert (
                        all(col["distinct"] > 1 for col in paths["row"]["columns"])
                        and len(paths["row"]["columns"]) == 8
                    )
                else:
                    assert all(
                        p["columns"] == [] and p["input_width"] == 21 + net.embedding.embedding_dim
                        for p in paths.values()
                    )
            (out / "checks_stage_3_2026-03_s0.json").rename(out / f"{name}_stage3.json.bak")
            (out / "checks_stage_2_2026-03_s0.json").rename(out / f"{name}_stage2.json.bak")
            (out / "checks_stage_1_2026-03_s0.json").rename(out / f"{name}_stage1.json.bak")
        # the table the lag path reads is wrong by one step: the gate raises
        net, x, primary, is_val, valid, arm, (first_us, step_us, c, cp) = _gate_fixture(
            torch=torch, arm_name="SKE"
        )
        net.table_new = torch.roll(net.table_new, shifts=1, dims=1)
        try:
            stage1.gate_stage(
                model=net, x=x, primary=primary, is_val=is_val, valid_us=valid, arm=arm, stage=2, month="2026-03", seed=0,
                out_dir=out, first_us=first_us, step_us=step_us, c_true=c, c_permuted=cp, n_rows=300,
            )  # fmt: skip
        except AssertionError:
            pass
        else:
            raise AssertionError("the gate accepted a lag table that is one step late")
        # the stage tag
        net, *_ = _gate_fixture(torch=torch, arm_name="SKE")
        tag = out / "m_s0.tag.json"
        arm = cols.ARMS["SKE"]
        stage1.write_stage_tag(path=tag, arm=arm, model=net)
        stage1.check_stage_tag(path=tag, arm=arm, model=net)
        with torch.no_grad():
            net.datetime[0].weight[0, 0] += 1e-3
        for what in ("weight", "table", "arm", "missing"):
            bad, *_ = _gate_fixture(torch=torch, arm_name="SKE")
            check_arm = arm
            if what == "weight":
                bad.load_state_dict(net.state_dict())
            elif what == "table":
                bad.table_new = bad.table_new + 1.0
                stage1.write_stage_tag(path=tag, arm=arm, model=bad)
                bad.table_new = bad.table_new + 1.0
            elif what == "arm":
                stage1.write_stage_tag(path=tag, arm=arm, model=bad)
                check_arm = cols.ARMS["SKE-CP"]
            else:
                tag.unlink()
            try:
                stage1.check_stage_tag(path=tag, arm=check_arm, model=bad)
            except AssertionError:
                continue
            raise AssertionError(f"the tag check accepted a changed {what}")
        # the tag hashes the whole date-time encoder (a bias moves it) and compares a table of another length by prefix
        net, *_ = _gate_fixture(torch=torch, arm_name="SKE")
        stage1.write_stage_tag(path=tag, arm=arm, model=net)
        shorter, *_ = _gate_fixture(torch=torch, arm_name="SKE")
        shorter.table_new = shorter.table_new[:, :50_000].clone()
        stage1.check_stage_tag(
            path=tag, arm=arm, model=shorter
        )  # the same table, cut: its prefix matches
        shorter.table_new[:, 30_000] += 1.0
        for model_bad, what in ((shorter, "an altered table of another length"),):
            try:
                stage1.check_stage_tag(path=tag, arm=arm, model=model_bad)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"the tag check accepted {what}")
        biased, *_ = _gate_fixture(torch=torch, arm_name="SKE")
        with torch.no_grad():
            biased.datetime[0].bias[0] += 1e-3
        try:
            stage1.check_stage_tag(path=tag, arm=arm, model=biased)
        except AssertionError:
            pass
        else:
            raise AssertionError("the tag check accepted a changed bias")
    passed(
        "the real-run gate writes the JSON, shows no columns for R1, raises on a late lag table, the stage tag catches a changed weight, table, arm, or missing file, and the whole encoder state and a table prefix are hashed"
    )


def test_json_mode_coverage_and_contrasts() -> None:
    """`--from-json` asserts coverage, rows, school 0 and 1, non-constant columns, and the plan's contrasts between arms."""
    torch = _torch_or_skip()
    if torch is None:
        return
    import e00735b_check_stages as check_stages  # ty: ignore[unresolved-import]
    import e00735b_stage1 as stage1  # ty: ignore[unresolved-import]

    arms = ("R1", "SKE", "SKE-S21", "SKE-CP")

    def write_all(*, folder: Path, skip: tuple | None = None) -> None:
        for name in arms:
            net, x, primary, is_val, valid, arm, (first_us, step_us, c, cp) = _gate_fixture(
                torch=torch, arm_name=name, rows=1500
            )
            for stage in (1, 2, 3):
                if skip == (name, stage):
                    continue
                stage1.gate_stage(
                    model=net, x=x if arm.columns else x[:, :60], primary=primary, is_val=is_val, valid_us=valid, arm=arm,
                    stage=stage, month="2026-03", seed=0, out_dir=folder / name, first_us=first_us, step_us=step_us,
                    c_true=c, c_permuted=cp, n_rows=1200,
                )  # fmt: skip

    kwargs: dict[str, Any] = {"arms": arms, "months": ("2026-03",), "seeds": (0,)}
    with tempfile.TemporaryDirectory(prefix="e735b_") as tmp:
        folder = Path(tmp)
        write_all(folder=folder)
        frame = check_stages.read_json_checks(folder=folder, **kwargs)
        assert set(frame["arm"]) == set(arms) and set(frame.filter(arm="R1")["column"]) == {"none"}
        assert len(frame.filter(arm="SKE", stage=2, batch="train", path="row")) == 8
        for bad, message in (
            ({**kwargs, "seeds": (0, 1)}, "a missing seed"),
            ({**kwargs, "months": ("2026-03", "2026-04")}, "a missing month"),
            ({**kwargs, "arms": ("R1", "SKE")}, "an unexpected arm"),
        ):
            try:
                check_stages.read_json_checks(folder=folder, **bad)  # ty: ignore[invalid-argument-type]
            except AssertionError:
                continue
            raise AssertionError(f"the coverage check accepted {message}")
        # an empty batch, a school column without a 1, and an arm pair on different rows
        path = folder / "SKE" / "checks_stage_2_2026-03_s0.json"
        record = json.loads(path.read_text())
        original = path.read_text()
        record["batches"]["val"]["n_rows"] = 0
        path.write_text(json.dumps(record))
        for what in ("no rows",):
            try:
                check_stages.read_json_checks(folder=folder, **kwargs)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"the check accepted {what}")
        path.write_text(original)
        for arm_name, field, value, what in (
            ("SKE", "max", 0.0, "a school column without a 1"),
            ("SKE-S21", "row_hash", "x", "different rows"),
        ):
            target = next(
                p
                for p in folder.rglob("checks_stage_1_2026-03_s0.json")
                if json.loads(p.read_text())["arm"] == arm_name
            )
            text = target.read_text()
            record = json.loads(text)
            if field == "max":
                record["batches"]["train"]["paths"]["row"]["columns"][0]["max"] = value
            else:
                record["batches"]["train"][field] = value
            target.write_text(json.dumps(record))
            try:
                check_stages.read_json_checks(folder=folder, **kwargs)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"the check accepted {what}")
            target.write_text(text)
        check_stages.read_json_checks(folder=folder, **kwargs)  # restored: passes again
    with tempfile.TemporaryDirectory(prefix="e735b_") as tmp:
        folder = Path(tmp)
        write_all(folder=folder, skip=("SKE-CP", 3))
        try:
            check_stages.read_json_checks(folder=folder, **kwargs)
        except AssertionError:
            pass
        else:
            raise AssertionError("a missing stage file was accepted")
    # the contrasts: an S21 file whose school sample equals SKE's, and a CP file whose coast means equal SKE's, are rejected
    with tempfile.TemporaryDirectory(prefix="e735b_") as tmp:
        folder = Path(tmp)
        write_all(folder=folder)
        files = {
            json.loads(p.read_text())["arm"]: p
            for p in folder.rglob("checks_stage_1_2026-03_s0.json")
        }
        ske = json.loads(files["SKE"].read_text())
        for arm_name, patch in (("SKE-S21", "school_sample"), ("SKE-CP", "per_primary_mean")):
            text = files[arm_name].read_text()
            record = json.loads(text)
            record["batches"]["train"]["paths"]["row"][patch] = ske["batches"]["train"]["paths"][
                "row"
            ][patch]
            files[arm_name].write_text(json.dumps(record))
            try:
                check_stages.read_json_checks(folder=folder, **kwargs)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"the contrast of {arm_name} with SKE was not enforced")
            files[arm_name].write_text(text)
    passed(
        "the table mode asserts coverage, non-empty batches, school 0 and 1, and the plan's SKE-S21 and SKE-CP contrasts"
    )


def test_b1_stats_check_comes_first_and_gate_is_wired() -> None:
    """Stage 1 compares the B1 statistics right after they are written, and the runner wires the gate into stages 1, 2, and 3."""
    stage1_text = (HERE / "e00735b_stage1.py").read_text()
    assert (
        stage1_text.index("check_b1_stats(month)")
        < stage1_text.index("x_gpu[:, mod.b1_cols")
        < stage1_text.index("for seed in todo")
    )
    assert stage1_text.index("e.pretrain_b1_columns(") < stage1_text.index("check_b1_stats(month)")
    assert "gate_stage(" in stage1_text and "write_stage_tag(" in stage1_text
    runner = (HERE / "e00735b_k1.py").read_text()
    assert runner.count("np.load(_OWN_STATS") == 1, "the statistics are compared in one place only"
    assert (
        "e.train = _gated(inner=train_recorded, stage=2)" in runner
        and "e.train = _gated(inner=e.train, stage=3)" in runner
    )
    assert "check_b1_stats=_check_b1_stats" in runner
    passed(
        "stage 1 compares the B1 statistics before any seed trains, and the gate and the tag check are wired into the three stages"
    )


def test_failed_basenames_do_not_clash() -> None:
    """No basename of this folder's scripts is the basename of another script under `ml-experiments/` or `studies/`."""
    mine = {p.name for p in HERE.glob("e00735b_*")}
    seen: dict[str, Path] = {}
    for root in ("ml-experiments", "studies"):
        for p in (REPO / root).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".sh"} and p.parent != HERE:
                seen.setdefault(p.name, p)
    clash = sorted(mine & set(seen))
    assert not clash, clash
    passed(f"no basename of the {len(mine)} files clashes with another script")


def main() -> None:
    """Run every test."""
    tests = [
        v
        for k, v in sorted(globals().items(), key=lambda kv: _order(kv[0]))
        if k.startswith("test_") and callable(v)
    ]
    for test in tests:
        test()
    print(f"\n{len(RESULTS)} tests passed")


def _order(name: str) -> int:
    """Return the line of a test function, so that the tests run in file order."""
    fn = globals()[name]
    return inspect.getsourcelines(fn)[1] if callable(fn) and name.startswith("test_") else 0


if __name__ == "__main__":
    main()

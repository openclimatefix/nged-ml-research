"""Tests of the K1 and K2 row alignment (pure Python, no torch)."""

from types import SimpleNamespace

import e00730_leakage as leak
import e00730_stages as stages
import numpy as np
import pytest


def rows(*, date: list[int], hh: list[int]) -> SimpleNamespace:
    return SimpleNamespace(date=np.array(date, dtype=np.int64), hh=np.array(hh, dtype=np.int64))


K2 = rows(date=[10, 10, 10, 11, 11], hh=[0, 1, 2, 0, 1])


def test_gappy_record_aligns() -> None:
    k1 = rows(date=[10, 10, 11, 11], hh=[0, 2, 0, 1])  # half-hour 1 of day 10 has no record row
    index = stages.align_k2_to_k1(k1=k1, k2=K2)
    assert index.tolist() == [0, 2, 3, 4]


def test_identical_sets_align() -> None:
    assert stages.align_k2_to_k1(k1=K2, k2=K2).tolist() == [0, 1, 2, 3, 4]


def test_k1_key_missing_from_k2_fails() -> None:
    k1 = rows(date=[10, 12], hh=[0, 0])
    with pytest.raises(leak.CorrectnessFailureError):
        stages.align_k2_to_k1(k1=k1, k2=K2)


def test_duplicated_key_fails() -> None:
    k1 = rows(date=[10, 10], hh=[0, 0])
    with pytest.raises(leak.CorrectnessFailureError):
        stages.align_k2_to_k1(k1=k1, k2=K2)
    k2 = rows(date=[10, 10, 11], hh=[0, 0, 0])
    with pytest.raises(leak.CorrectnessFailureError):
        stages.align_k2_to_k1(k1=rows(date=[10], hh=[0]), k2=k2)


def test_off_by_one_hh_shift_fails() -> None:
    k1 = rows(date=[10, 10, 11], hh=[1, 3, 1])  # 3 on day 10 is absent from K2
    with pytest.raises(leak.CorrectnessFailureError):
        stages.align_k2_to_k1(k1=k1, k2=K2)


def test_k2_smaller_than_k1_fails() -> None:
    k1 = rows(date=[10, 10, 10, 11, 11, 11], hh=[0, 1, 2, 0, 1, 2])
    with pytest.raises(leak.CorrectnessFailureError):
        stages.align_k2_to_k1(k1=k1, k2=K2)

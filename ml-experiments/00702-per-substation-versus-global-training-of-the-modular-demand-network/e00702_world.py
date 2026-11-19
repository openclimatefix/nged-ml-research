# ruff: noqa: E501, PT018
"""Experiment 702: the patches that make one Primary's training world, and nothing else.

`e00702_fit.py` calls `restrict_to_primary` before it imports `e00574_fit`, because that module
builds its B1-column helper (`EX`) at import. After the call, every loader that reads experiment
87's `series_table()` sees one Primary, so `prep.primaries()`, `fd._series_and_arrays()`, the
cleaned power, the training rows, the pre-training rows, and the lag statistics all hold that
Primary alone. The stored expected-input table, which holds all 16 Primaries' weather columns, is
sliced to the Primary and never written.

The functions take the modules as arguments (or import them inside), so the tests can pass small
stand-ins and no earlier experiment is needed to import this file.
"""

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

FLAGGED_COLUMNS: Final[tuple[str, str]] = ("time_series_id", "valid_time")


def primary_labels(*, series_table: Callable[[], pl.DataFrame]) -> list[str]:
    """Return the 16 Primary labels in `prep.primaries()` order (sorted by label number)."""
    table = series_table().filter(pl.col("substation_type") == "Primary")
    return table["site"].to_list()


def one_site_series_table(
    *, original: Callable[[], pl.DataFrame], label: str
) -> Callable[[], pl.DataFrame]:
    """Return a `series_table` that holds only the substation named `label`.

    Args:
        original: Experiment 87's `series_table`.
        label: The Primary's label, for example `D5`.

    Returns:
        A function with the original's name and no arguments, returning a one-row frame.
    """
    keep = original().filter((pl.col("substation_type") == "Primary") & (pl.col("site") == label))
    assert keep.height == 1, (label, keep.height)

    def series_table() -> pl.DataFrame:
        return keep

    return series_table


def restrict_to_primary(*, common: Any, label: str) -> int:
    """Patch `common.series_table` to one Primary and return that Primary's index among the 16.

    Args:
        common: The module `e00087_common`.
        label: The Primary's label.

    Returns:
        The Primary's index in the pooled order, which the shift schedule of the global arm uses.
    """
    labels = primary_labels(series_table=common.series_table)
    assert label in labels, (label, labels)
    index = labels.index(label)
    common.series_table = one_site_series_table(original=common.series_table, label=label)
    return index


def one_site_cleaned_arrays(*, prep422: Any, fd: Any, common: Any, flagged_keys: Path) -> dict:
    """Return the one Primary's cleaned power over capacity, as `e00422_prep.cleaned_arrays` does.

    Differences from the original: the flagged half-hours are scanned with a filter on the one
    series, so the other series' flags are never collected, and the short-history cut applies only
    if the Primary is one of the labels that have it (the original indexes the dict by label and
    raises a `KeyError` for a Primary that is not in the 16).
    """
    series, arrays = fd._series_and_arrays()
    stale = prep422.stale
    primary = series.filter(pl.col("substation_type") == "Primary")
    assert primary.height == 1, primary.height
    row = next(primary.iter_rows(named=True))
    flagged = (
        pl.scan_parquet(flagged_keys)
        .filter(pl.col("time_series_id") == row["time_series_id"])
        .select(*FLAGGED_COLUMNS)
        .collect()
    )
    power = arrays[row["site"]].power.copy()
    index = np.asarray(common.grid_index(times=flagged["valid_time"]))
    power[index[(index >= 0) & (index < power.size)]] = np.nan
    clean = {row["site"]: power}
    if row["site"] in stale.SHORT_HISTORY_LABELS:
        cut = int(common.grid_index(times=stale.SHORT_HISTORY_CUT))
        clean[row["site"]][:cut] = np.nan
    return clean


def sliced_raw_table(*, index: int) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Return a `raw_table` that reads the stored 16-row cache, keeps one row, and never writes.

    Args:
        index: The Primary's row in the 16-row cache.

    Returns:
        A function with the signature of `e00560_data.raw_table`. It refuses to build a cache that is missing,
        because a build would loop over `prep.primaries()` and write a one-row cache that the
        pooled arms would then read.
    """

    def raw_table(*, source: str, cache: Path) -> tuple[np.ndarray, np.ndarray]:
        f_col = cache / f"expected_{source}_cols.npy"
        f_org = cache / f"expected_{source}_origin.npy"
        assert f_col.exists() and f_org.exists(), f"no stored expected-input table in {cache}"
        cols, origin = np.load(f_col, mmap_mode="r"), np.load(f_org, mmap_mode="r")
        assert cols.shape[0] == origin.shape[0] and index < cols.shape[0], cols.shape
        return cols[index : index + 1], origin[index : index + 1]

    return raw_table


def schedule_trick(
    *,
    original: Callable[..., tuple[np.ndarray, int]],
    index: int,
    n_primaries: int,
    first: int,
    last: int,
) -> Callable[..., tuple[np.ndarray, int]]:
    """Return a `synthetic_clean` that gives one Primary the shifts it has in the global arm.

    `synthetic_clean` draws its schedule from one generator, row after row, and its draw count per
    row depends on `first` and `last` and never on the data. The global arm's rows hold the 16
    Primaries, so Primary `index` is the `index`-th block of draws. The returned function embeds the
    one-row power array at row `index` of an array of `n_primaries` rows (the rest NaN), calls the
    original with the fold's `first` and `last`, and returns row `index`. Arguments `first` and
    `last` that the caller passes (the Primary's own span) are only checked to lie inside the fold's.

    Args:
        original: `e00581_bank.synthetic_clean`.
        index: The Primary's index among the 16.
        n_primaries: 16.
        first: The fold's first target grid index.
        last: The fold's last target grid index.

    Returns:
        A function with the original's signature.
    """
    span = (first, last)

    def synthetic_clean(
        *, clean: np.ndarray, first: int, last: int, seed: int
    ) -> tuple[np.ndarray, int]:
        assert clean.shape[0] == 1, clean.shape
        assert span[0] <= first and last <= span[1], (
            "the Primary's span leaves the fold's",
            span,
            first,
            last,
        )
        big = np.full((n_primaries, clean.shape[1]), np.nan, dtype=clean.dtype)
        big[index] = clean[0]
        fixed, n_events = original(clean=big, first=span[0], last=span[1], seed=seed)
        assert np.isnan(fixed[np.arange(n_primaries) != index]).all()
        return fixed[index : index + 1], n_events

    return synthetic_clean


LABEL_TOKEN: Final[re.Pattern[str]] = re.compile(r"(?<![A-Za-z0-9])D\d+(?![0-9])")
"""A Primary's label (`D5`, `D22`) as a whole token inside a path."""


def other_primary_in_path(*, path: str | Path, own: str, labels: Iterable[str]) -> str | None:
    """Return the label of another Primary that the path names, or None.

    Args:
        path: A file path or a URL.
        own: The label of the process's Primary.
        labels: All 16 labels.

    Returns:
        The label of the first other Primary that the path names, or None.
    """
    others = set(labels) - {own}
    for token in LABEL_TOKEN.findall(str(path)):
        if token in others:
            return token
    return None


def guard_readers(
    *, own: str, labels: Iterable[str], modules: dict[str, tuple[object, str]]
) -> list[str]:
    """Wrap file readers so that opening another Primary's per-site file raises, and return the log of every path opened.

    Args:
        own: The label of the process's Primary.
        labels: All 16 labels.
        modules: A name for each reader, mapped to (module, attribute), such as
            `{"np.load": (np, "load")}`. Each reader takes the path as its first positional argument
            or as the keyword `file` or `source`.

    Returns:
        The list that every opened path is appended to, so a caller can write it to a log.
    """
    labels = tuple(labels)
    opened: list[str] = []
    for name, (module, attribute) in modules.items():
        original = getattr(module, attribute)

        def guarded(
            *args: Any, _original: Callable = original, _name: str = name, **kwargs: Any
        ) -> Any:
            path = args[0] if args else kwargs.get("file", kwargs.get("source"))
            if isinstance(path, str | Path):
                other = other_primary_in_path(path=path, own=own, labels=labels)
                if other is not None:
                    msg = f"{_name} opened {path}, which names Primary {other}; this process holds {own} only"
                    raise PermissionError(msg)
                opened.append(str(path))
            return _original(*args, **kwargs)

        setattr(module, attribute, guarded)
    return opened

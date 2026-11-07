# ruff: noqa: ANN001, ANN003, ANN202, D103, E501, PLR0915, PT018
"""Experiment 640: single-cutoff folds at a chosen fit age, patched onto the development pipeline.

The development pipeline (experiments 510, 511, 574, 591, 613) works on folds named by calendar
month: the fold's refit is the first day of its test month, the networks are fitted on data before
that day, and refits follow on days 0 and 14. Experiment 640 needs a fold whose fit ends `A` weeks
before the scored month. A fold name is therefore `YYYY-MM-AA`, the development month `YYYY-MM` and
the fit age `AA` in weeks (for example `2025-10-08`):

- the single cutoff is 00:00 UTC on the first day of the month minus `A` weeks, and there is no
  refit inside the month;
- everything fitted (pre-training rows, standardiser, B1 statistics, knot count, fine-tune rows,
  early-stopping rows) uses that cutoff, because experiment 574's code reads the fold's refit time
  through `prep.refit_of`, which this module redirects to the cutoff;
- the scored origins are the whole real month, so the fit age at an origin is `A` weeks plus the
  days into the month;
- in the training phase (`E00640_PHASE=train`) the cleaned power is set to NaN from the cutoff
  onwards, so no code path can read a later reading.

The name ends in digits only because experiment 591's synthetic-shift builder seeds with
`int(month.replace("-", ""))`. Every parser of a fold name that this module does not patch fails
loudly: `prep.refit_of` and `prep.test_origins` are patched here, and `DriftColumns` is patched here
too. Nothing in an earlier experiment's folder is edited.
"""

import os
import re
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np

MONTHS: Final = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
AGES_WEEKS: Final = (0, 4, 8, 16, 26)
PHASE: Final = os.environ.get("E00640_PHASE", "")
"""`train`: the power from the cutoff onwards is hidden. `predict`: forecast the real month."""
TRAIN_ONLY: Final = PHASE == "train"

_FOLD: Final = re.compile(r"(\d{4}-\d{2})-(\d{2})")
_STATE: dict[str, bool] = {"scoring_window": False}
"""`scoring_window`: while true, `prep.refit_of` returns the real month start (experiment 428's
`build_test` reads its window through it)."""


def fold_name(*, month: str, age_weeks: int) -> str:
    return f"{month}-{age_weeks:02d}"


def parse(fold: str) -> tuple[str, int]:
    """Return (development month, fit age in weeks) of a fold name; raise on anything else."""
    match = _FOLD.fullmatch(fold)
    assert match, f"not a fit-age fold name: {fold!r}"
    month, age = match.group(1), int(match.group(2))
    assert month in MONTHS and age in AGES_WEEKS, fold
    return month, age


def is_fold(name: str) -> bool:
    return _FOLD.fullmatch(name) is not None


def real_start(*, fold: str) -> datetime:
    return datetime.fromisoformat(f"{parse(fold)[0]}-01").replace(tzinfo=UTC)


def real_stop(*, fold: str) -> datetime:
    return (real_start(fold=fold) + timedelta(days=32)).replace(day=1)


def cutoff_of(*, fold: str) -> datetime:
    """Return the fold's single fit cutoff: the month start minus the fit age."""
    return real_start(fold=fold) - timedelta(weeks=parse(fold)[1])


def install(e, *, folds: list[str]) -> None:
    """Patch experiment 574's module `e` (or experiment 565's) for the fit-age folds `folds`."""
    assert PHASE in {"train", "predict"}, "set E00640_PHASE to train or predict"
    assert folds and all(is_fold(f) for f in folds), folds
    if TRAIN_ONLY:
        assert len(folds) == 1, "the training phase runs one cutoff per process"
    import e00559_extra as extra  # ty: ignore[unresolved-import]

    c, k, prep = e.c, e.k, e.prep
    cutoff_set = {cutoff_of(fold=f) for f in folds}

    # 1. the fold's refit time (and so month_start, fit_until, pre-training, B1 statistics) is the cutoff
    _refit_of, _test_origins = prep.refit_of, prep.test_origins

    def refit_of(*, month):
        if not is_fold(month):
            return _refit_of(month=month)
        if _STATE["scoring_window"]:
            return real_start(fold=month)
        return cutoff_of(fold=month)

    def test_origins(*, month):
        return _test_origins(month=parse(month)[0] if is_fold(month) else month)

    prep.refit_of, prep.test_origins = refit_of, test_origins

    # 2. one cutoff, scored over the real month
    _month_stop = c.month_stop

    def cutoffs(*, month, cadence=None):
        return [cutoff_of(fold=month)]

    def window_of(*, month, cadence, index):
        assert index == 0
        return c.us(t=real_start(fold=month)), c.us(t=real_stop(fold=month))

    def month_stop(*, month):
        return real_stop(fold=month) if is_fold(month) else _month_stop(month=month)

    _extension = c._extension

    def extension(*, refit, last, clean, rows):
        """Return experiment 510's later-cutoff rows; none exist when the only cutoff is the refit."""
        if refit not in cutoff_set:
            return _extension(refit=refit, last=last, clean=clean, rows=rows)
        assert last == refit, "a fit-age fold has one cutoff"
        empty = lambda a, dtype=None: np.empty((0, *a.shape[1:]), dtype=dtype or a.dtype)  # noqa: E731
        return {
            "x": empty(rows.x, np.float32),
            "y": empty(rows.y),
            "origin": np.empty(0, np.int64),
            "valid": np.empty(0, np.int64),
            "target_idx": np.empty(0, np.int64),
            "lead": empty(rows.lead),
            "primary": np.empty(0, np.int64),
            "is_val": empty(rows.is_val),
        }

    c._extension = extension
    k.cutoffs, k.window_of, c.month_stop, c.cutoffs = cutoffs, window_of, month_stop, cutoffs
    e.cutoffs = cutoffs

    # 3. the test rows are the real month's, whatever the cutoff
    _inner = e._build_test

    def build_test_window(*, month, standardiser):
        if not is_fold(month):
            return _inner(month=month, standardiser=standardiser)
        _STATE["scoring_window"] = True
        try:
            return _inner(month=month, standardiser=standardiser)
        finally:
            _STATE["scoring_window"] = False

    e._build_test = build_test_window

    # 4. the knot count of the drift column follows the cutoff's month
    _drift_init = extra.DriftColumns.__init__

    def drift_init(self, *, month, mr):
        if is_fold(month):
            month = f"{cutoff_of(fold=month):%Y-%m}"
        _drift_init(self, month=month, mr=mr)

    extra.DriftColumns.__init__ = drift_init

    if TRAIN_ONLY:
        (fold,) = folds
        _clean_stack = c.clean_stack
        cut = int((c.us(t=cutoff_of(fold=fold)) - c.prep422.GRID_START_US) // c.prep422.STEP_US)

        def clean_stack():
            """Return the cleaned power with every half-hour from the cutoff hidden."""
            clean = _clean_stack().copy()
            clean[:, cut:] = np.nan
            assert not np.isfinite(clean[:, cut:]).any()
            return clean

        c.clean_stack = clean_stack

        def build_test(*, month, standardiser):
            """Keep the scored month closed while the networks are trained."""
            assert is_fold(month)

        def window_frame(**_):
            return None

        e.data.build_test = build_test
        e.build_test = build_test
        e.f.window_frame = window_frame
        e.window_frame = window_frame
    else:

        def fine_tune(**_):
            raise AssertionError("the predict phase only reads fitted models; none may be refitted")

        e.fine_tune = fine_tune

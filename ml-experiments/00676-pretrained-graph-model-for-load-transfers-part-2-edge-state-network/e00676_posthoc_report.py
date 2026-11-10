"""Experiment 676, Part A: per-cell shares of the post hoc hybrids (experiment 678's report code)."""

# ruff: noqa: E501

from __future__ import annotations

import e00678_report as rep  # ty: ignore[unresolved-import]
from e00676_posthoc import base

rep.OUT = base.OUT
if __name__ == "__main__":
    rep.main()

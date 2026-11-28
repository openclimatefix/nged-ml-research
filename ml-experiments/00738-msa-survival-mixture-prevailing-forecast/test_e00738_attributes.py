"""Check that every attribute taken from an imported module exists, lazy imports included."""
# ruff: noqa: D102, D103, D107, E402, N815

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
# the tests never write to the experiment's data folder
os.environ.setdefault("E00738_OUT", str(Path(tempfile.gettempdir()) / "scratch" / "e738_tests"))
os.environ.setdefault("E00738_SCRATCH", "1")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _d in HERE.parent.glob("*"):
    if _d.is_dir() and _d.name[:5].isdigit() and int(_d.name[:5]) in {
        11, 87, 377, 378, 381, 391, 397, 400, 402, 404, 407, 410, 418, 422, 425, 428, 435, 437, 448,
        454, 470, 510, 511, 513, 515, 521, 545, 558, 559, 560, 562, 565, 568, 569, 572, 574, 578,
        581, 632, 667, 668, 709, 647, 650, 652, 654, 713, 726,
    }:  # fmt: skip
        sys.path.append(str(_d))
FOLDER = Path(__file__).parent

import ast
import importlib

torch = pytest.importorskip("torch", reason="the 713 modules need torch (workstation only)")

SKIPPED_ROOTS = frozenset({"__future__"})


def _module_or_none(name: str) -> object | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class Checker(ast.NodeVisitor):
    """Walk a module and check each `alias.attr` and each `from m import n` against the module."""

    def __init__(self, *, file: str) -> None:
        self.file = file
        self.scopes: list[dict[str, str]] = [{}]
        self.problems: list[str] = []
        self.unimportable: list[str] = []

    def _bind(self, node: ast.Import | ast.ImportFrom) -> None:
        scope = self.scopes[-1]
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in SKIPPED_ROOTS:
                    continue
                scope[a.asname or a.name.split(".")[0]] = (
                    a.name if a.asname else a.name.split(".")[0]
                )
                if _module_or_none(a.name) is None:
                    self.unimportable.append(f"{self.file}:{node.lineno} import {a.name}")
            return
        if node.module is None or node.module in SKIPPED_ROOTS:
            return
        module = _module_or_none(node.module)
        if module is None:
            self.unimportable.append(f"{self.file}:{node.lineno} from {node.module}")
            return
        for a in node.names:
            sub = _module_or_none(f"{node.module}.{a.name}")
            if not hasattr(module, a.name) and sub is None:
                self.problems.append(
                    f"{self.file}:{node.lineno} from {node.module} import {a.name}"
                )
            if sub is not None:
                scope[a.asname or a.name] = f"{node.module}.{a.name}"

    def visit_Import(self, node: ast.Import) -> None:
        self._bind(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._bind(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scopes.append(dict(self.scopes[-1]))
        self.generic_visit(node)
        self.scopes.pop()

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            target = self.scopes[-1].get(node.value.id)
            if target is not None:
                module = _module_or_none(target)
                if module is not None and not hasattr(module, node.attr):
                    sub = _module_or_none(f"{target}.{node.attr}")
                    if sub is None:
                        self.problems.append(
                            f"{self.file}:{node.lineno} {node.value.id}.{node.attr} ({target})"
                        )
        self.generic_visit(node)


def test_every_attribute_taken_from_an_imported_module_exists_lazy_imports_included() -> None:
    """A call to a function that does not exist fails only when its line runs, so check them all."""
    problems: list[str] = []
    unimportable: list[str] = []
    files = sorted(FOLDER.glob("e00738_*.py"))
    assert len(files) >= 17
    for f in files:
        checker = Checker(file=f.name)
        checker.visit(ast.parse(f.read_text()))
        problems += checker.problems
        unimportable += checker.unimportable
    assert not problems, "\n".join(problems)
    assert not unimportable, "modules that could not be imported:\n" + "\n".join(unimportable)


def test_the_checker_finds_a_missing_attribute_and_a_missing_name() -> None:
    src = (
        "import json\n"
        "def f():\n"
        "    import os as o\n"
        "    from json import dumps, nonexistent_name\n"
        "    return json.no_such_function, o.path, dumps\n"
    )
    checker = Checker(file="synthetic.py")
    checker.visit(ast.parse(src))
    text = "\n".join(checker.problems)
    assert "json.no_such_function" in text
    assert "nonexistent_name" in text
    assert "o.path" not in text

# ruff: noqa: E501
"""Guard tests of experiment 733.

Every attribute of an imported experiment module exists, and every `.numpy()` follows `.cpu()`.
"""

import ast
import importlib

import e00733_paths  # noqa: F401  (puts the sibling experiments on sys.path)
import pytest
from ast_guards import (
    FOLDER,
    attribute_uses,
    defined_names,
    module_aliases,
    numpy_calls_without_cpu,
    own_sources,
)

SIBLING_PREFIXES = ("e00", "studies")


def test_modules_exist() -> None:
    """The six part-1 modules exist."""
    assert set(own_sources()) == {"core", "paths", "channels", "rerun", "leakage", "inputs"}


@pytest.mark.parametrize("module", sorted(own_sources()))
def test_every_attribute_of_an_imported_module_exists(module: str) -> None:
    """Every `alias.attribute` read from an imported experiment module is defined there (no typo, no stale name)."""
    path = own_sources()[module]
    tree = ast.parse(path.read_text())
    checked = 0
    for alias, name in module_aliases(tree=tree).items():
        if not name.startswith(SIBLING_PREFIXES):
            continue
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError:
            continue  # a sibling that is absent from this checkout (experiment 732 before its merge)
        for attr in attribute_uses(tree=tree, alias=alias):
            assert hasattr(mod, attr), f"{path.name}: {name}.{attr} does not exist"
            checked += 1
    own_alias = {a: n for a, n in module_aliases(tree=tree).items() if n.startswith("e00733_")}
    for alias, name in own_alias.items():
        target = ast.parse((FOLDER / f"{name}.py").read_text())
        for attr in attribute_uses(tree=tree, alias=alias):
            assert attr in defined_names(tree=target), f"{path.name}: {name}.{attr} does not exist"
            checked += 1
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith(SIBLING_PREFIXES):
            try:
                mod = importlib.import_module(n.module)
            except ModuleNotFoundError:
                continue
            for a in n.names:
                assert hasattr(mod, a.name), f"{path.name}: {n.module}.{a.name} does not exist"
                checked += 1
    assert checked > 0 or module == "core"


def test_the_attribute_check_detects_a_missing_name() -> None:
    """The check fails on a name that does not exist (it cannot pass vacuously)."""
    tree = ast.parse("import e00733_core as core\ncore.NO_SUCH_NAME\n")
    target = ast.parse((FOLDER / "e00733_core.py").read_text())
    uses = attribute_uses(tree=tree, alias="core")
    assert uses == {"NO_SUCH_NAME"}
    assert not uses <= defined_names(tree=target)


@pytest.mark.parametrize("module", sorted(own_sources()))
def test_every_numpy_call_follows_cpu(module: str) -> None:
    """Every `.numpy()` in the modules is called on a `.cpu()` result."""
    assert numpy_calls_without_cpu(source=own_sources()[module].read_text()) == []


def test_the_numpy_check_detects_a_bare_call() -> None:
    """The check fails on `x.numpy()` and passes on `x.cpu().numpy()`."""
    assert numpy_calls_without_cpu(source="y = x.numpy()\n") == [1]
    assert numpy_calls_without_cpu(source="y = x.detach().cpu().numpy()\n") == []


def test_no_absolute_home_path_in_the_modules() -> None:
    """No module or test names an absolute path under the home folder (a repository test fails on one)."""
    needle = "/ho" + "me/" + "jack"
    for path in [*own_sources().values(), *FOLDER.glob("tests/*.py")]:
        assert needle not in path.read_text(), path.name

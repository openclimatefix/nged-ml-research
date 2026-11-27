"""AST helpers of the guard tests. They read module sources and never import them."""

import ast
from pathlib import Path

FOLDER = Path(__file__).parent.parent
OWN_MODULES = ("core", "paths", "channels", "rerun", "leakage", "inputs")


def own_sources() -> dict[str, Path]:
    """Return the module name and path of every experiment module that exists."""
    return {
        m: FOLDER / f"e00733_{m}.py" for m in OWN_MODULES if (FOLDER / f"e00733_{m}.py").exists()
    }


def attribute_uses(*, tree: ast.AST, alias: str) -> set[str]:
    """Return the attribute names read from the module imported under `alias`."""
    return {
        n.attr for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == alias
    }  # fmt: skip


def module_aliases(*, tree: ast.AST) -> dict[str, str]:
    """Return alias -> module name for every `import x as y` and `import x`."""
    out: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name] = a.name
    return out


def defined_names(*, tree: ast.Module) -> set[str]:
    """Return the names a module defines at top level (functions, classes, assignments, imports)."""
    names: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, ast.Import | ast.ImportFrom):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return names


def numpy_calls_without_cpu(*, source: str) -> list[int]:
    """Return the line numbers of every `.numpy()` call whose receiver is not a `.cpu()` call."""
    bad: list[int] = []
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "numpy":
            recv = n.func.value
            if not (
                isinstance(recv, ast.Call)
                and isinstance(recv.func, ast.Attribute)
                and recv.func.attr == "cpu"
            ):
                bad.append(n.lineno)
    return bad

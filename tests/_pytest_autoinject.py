"""Parallelise a full-suite run with `pytest-xdist`; leave a targeted run serial.

This cannot be a plain `addopts = "-n auto"`, because worker start-up costs a few seconds whether
one test runs or the whole suite does. That is a good trade for the full suite and a bad one for a
run that names a single test.

So `-n auto` is added only when the invocation names no explicit path or node id. A run that
already carries its own `-n` keeps it, because the injected `-n auto` is prepended and argparse
keeps the last value. A `-k` or `-m` selection does not suppress the injection. `-s` and
`--capture=no` do suppress it, because under xdist a worker's uncaptured output never reaches the
controller.

This has to be a separately loaded plugin (`addopts = "-p _pytest_autoinject"`) rather than a
`pytest_load_initial_conftests` hook in the root `conftest.py`. That hook fires while pytest loads
the root `conftest.py` itself, so a hook defined there is never registered in time. A `-p` plugin
loads before `pytest-xdist` registers its `-n` option, and pytest parses `args` a third time
afterwards, which is what makes `-n` and `capture` visible on `known_args_namespace` here.
"""

import pytest


def pytest_load_initial_conftests(early_config: pytest.Config, args: list[str]) -> None:
    if not early_config.pluginmanager.hasplugin("xdist"):
        # `-p no:xdist` leaves xdist unloaded; injecting `-n` with no plugin to read it is a pytest
        # usage error, not a silent fallback to serial.
        return
    ns = early_config.known_args_namespace
    if not ns.file_or_dir and ns.capture != "no":
        args[:0] = ["-n", "auto"]

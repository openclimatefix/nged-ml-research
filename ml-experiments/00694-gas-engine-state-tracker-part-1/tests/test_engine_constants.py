"""The loader of the engine's private constants reads a file and names the file in every error."""

import json
from datetime import UTC, datetime
from pathlib import Path

import engine_constants as loader
import pytest

# Invented numbers: the real constants live in a private file that no test reads.
INVENTED = {"a_date": "2001-02-03", "a_size": 4.5, "an_integer": 7, "a_string": "text"}


@pytest.fixture
def constants_file(tmp_path: Path) -> Path:
    path = tmp_path / "engine_constants.json"
    path.write_text(json.dumps(INVENTED))
    return path


def test_a_number_and_a_date_are_read(constants_file: Path):
    assert loader.engine_float(key="a_size", path=constants_file) == 4.5
    assert loader.engine_float(key="an_integer", path=constants_file) == 7.0
    assert loader.engine_datetime(key="a_date", path=constants_file) == datetime(
        2001, 2, 3, tzinfo=UTC
    )


def test_a_missing_file_is_named(tmp_path: Path):
    missing = tmp_path / "nowhere.json"
    with pytest.raises(loader.EngineConstantsError, match=r"nowhere\.json"):
        loader.engine_float(key="a_size", path=missing)


def test_a_missing_key_is_named(constants_file: Path):
    with pytest.raises(loader.EngineConstantsError, match="'absent'") as error:
        loader.engine_float(key="absent", path=constants_file)
    assert str(constants_file) in str(error.value)


def test_a_wrong_type_is_refused_without_echoing_the_value(constants_file: Path):
    with pytest.raises(loader.EngineConstantsError, match="must be a number") as error:
        loader.engine_float(key="a_string", path=constants_file)
    assert "text" not in str(error.value)
    with pytest.raises(loader.EngineConstantsError, match="YYYY-MM-DD string"):
        loader.engine_datetime(key="a_size", path=constants_file)
    with pytest.raises(loader.EngineConstantsError, match="not a YYYY-MM-DD date") as error:
        loader.engine_datetime(key="a_string", path=constants_file)
    assert "text" not in str(error.value)
    assert error.value.__cause__ is None


def test_a_string_is_read_and_a_number_is_refused(constants_file: Path):
    assert loader.engine_string(key="a_string", path=constants_file) == "text"
    with pytest.raises(loader.EngineConstantsError, match="non-empty string"):
        loader.engine_string(key="a_size", path=constants_file)


def test_a_boolean_is_not_a_number(tmp_path: Path):
    path = tmp_path / "engine_constants.json"
    path.write_text(json.dumps({"flag": True}))
    with pytest.raises(loader.EngineConstantsError, match="must be a number"):
        loader.engine_float(key="flag", path=path)


def test_a_file_that_is_not_a_json_object_is_refused(tmp_path: Path):
    path = tmp_path / "engine_constants.json"
    path.write_text("[1, 2]")
    with pytest.raises(loader.EngineConstantsError, match="not a JSON object"):
        loader.engine_float(key="a_size", path=path)
    path.write_text("not json")
    with pytest.raises(loader.EngineConstantsError, match="not JSON"):
        loader.engine_float(key="a_size", path=path)


def test_the_environment_variable_overrides_the_default_path(
    constants_file: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(loader.PATH_ENV_VAR, str(constants_file))
    assert loader.constants_path() == constants_file
    assert loader.engine_float(key="a_size") == 4.5

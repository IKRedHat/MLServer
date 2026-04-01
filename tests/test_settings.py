import os
import sys
import json
import pytest
from unittest.mock import patch

from mlserver import Settings, ModelSettings
from mlserver.settings import (_extra_sys_path, _get_import_path, _reload_module)
from mlserver.model import MLModel


def test_extra_sys_path():
    test_path = "/tmp/test_path"
    with _extra_sys_path(test_path):
        assert test_path in sys.path
    assert test_path not in sys.path


def test_get_import_path():
    class MockClass:
        pass

    import_path = _get_import_path(MockClass)
    assert import_path == "tests.test_settings.MockClass"


def test_reload_module():
    with patch("importlib.import_module") as mock_import_module:
        _reload_module("module.path")
        mock_import_module.assert_called_once_with("module")


def test_reload_module_empty_import_path():
    with patch("importlib.import_module") as mock_import_module:
        _reload_module("")
        mock_import_module.assert_not_called()


@pytest.fixture
def model_settings_file(tmp_path):
    model_settings = {
        "name": "test-model",
        "implementation": "tests.test_settings.MockModel",
    }
    model_settings_path = tmp_path / "model-settings.json"
    with open(model_settings_path, "w") as f:
        json.dump(model_settings, f)
    return str(model_settings_path)


class MockModel(MLModel):
    pass


def test_model_settings_parse_file(model_settings_file):
    model_settings = ModelSettings.parse_file(model_settings_file)
    assert model_settings.name == "test-model"
    assert model_settings._source == model_settings_file


def test_model_settings_model_validate():
    model_settings_dict = {
        "name": "test-model",
        "implementation": "tests.test_settings.MockModel",
        "_source": "test-source"
    }
    model_settings = ModelSettings.model_validate(model_settings_dict)
    assert model_settings.name == "test-model"
    assert model_settings._source == "test-source"


def test_model_settings_implementation():
    model_settings = ModelSettings(
        name="test-model", implementation="tests.test_settings.MockModel"
    )
    assert model_settings.implementation == MockModel


def test_model_settings_implementation_with_source(tmp_path):
    model_file = tmp_path / "model.py"
    model_file.write_text("class MyModel:\n    pass")
    settings_file = tmp_path / "model-settings.json"
    settings_file.write_text(json.dumps({
        "name": "my-model",
        "implementation": "model.MyModel"
    }))

    model_settings = ModelSettings.parse_file(str(settings_file))
    assert model_settings.implementation.__name__ == "MyModel"


def test_model_settings_version():
    model_settings = ModelSettings(
        name="test-model", implementation="tests.test_settings.MockModel"
    )
    assert model_settings.version is None

    model_settings = ModelSettings(
        name="test-model",
        implementation="tests.test_settings.MockModel",
        parameters={"version": "v1"},
    )
    assert model_settings.version == "v1"